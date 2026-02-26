import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from typing import List, Dict, Any, Optional
from openai import (
    AsyncOpenAI,
    APIError, APIStatusError, APIConnectionError, APITimeoutError,
    AuthenticationError, PermissionDeniedError, BadRequestError,
    NotFoundError, ConflictError, UnprocessableEntityError,
    RateLimitError, InternalServerError,
)
from tqdm.asyncio import tqdm_asyncio
from sys import exit
from contextchecker.stats import GLOBAL_STATS


# ───────────────────────────────────────────────────────────────
#  RETRY MATRIX
#  Each strategy is a complete request configuration.
#  On capability errors, we advance to the next strategy.
#  First real request discovers the working strategy (serialized).
#  All subsequent requests use the locked strategy concurrently.
# ───────────────────────────────────────────────────────────────

@dataclass
class RetryStrategy:
    """One level in the retry matrix. Readable and explicit."""
    name: str
    reasoning_effort: Optional[str] = None   # "low", "medium", "high" — OpenAI standard. None = don't send.
    use_schema: bool = True                  # Strict JSON Schema (structured output, constrained decoding)
    use_json_object: bool = False            # Loose JSON (valid JSON, shape not enforced)
    temperature: float = 0.0


# Best case at top, vanilla at bottom.
# On capability errors (BadRequest, UnsupportedParams), we walk down.
RETRY_MATRIX = [
    RetryStrategy("Reasoning + Schema",  reasoning_effort="low",  use_schema=True),
    RetryStrategy("Schema Only",                                  use_schema=True),
    RetryStrategy("Reasoning + JSON",    reasoning_effort="low",  use_json_object=True),
    RetryStrategy("JSON Only",                                    use_json_object=True),
    RetryStrategy("Vanilla")
]


class ErrorAction(Enum):
    """What to do when an API error occurs."""
    FATAL = "fatal"   # Exit program — unrecoverable
    SKIP  = "skip"    # Return "", continue batch — per-item failure
    RETRY = "retry"   # Backoff and retry — transient


class LLMClient:
    def __init__(self, api_key: str, model: str, base_url: Optional[str] = None, concurrency: int = 10):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.concurrency = concurrency
        
        # OpenAI SDK client — only created if base_url is set (direct endpoint mode)
        if self.base_url:
            self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        else:
            self.client = None  # LiteLLM mode — no direct client needed
        
        self._connection_verified = False
        self.sem = asyncio.Semaphore(self.concurrency)

        # Retry matrix state (OpenAI SDK path only)
        self._strategy_index = 0
        self._strategy_discovered = False   # True after first successful request
        self._discovery_lock = asyncio.Lock()  # Serializes strategy discovery
        self._cache_hit_logged = False  # Only log cache hint once

        sdk_mode = "OpenAI SDK" if self.base_url else "LiteLLM"
        print(f"🔧 LLMClient initialized: {self.model} via {sdk_mode}")


    @property
    def strategy(self) -> RetryStrategy:
        """Current retry strategy."""
        return RETRY_MATRIX[self._strategy_index]


    def _next_strategy(self) -> bool:
        """Advance to next strategy. Returns True if advanced, False if at bottom."""
        if self._strategy_index < len(RETRY_MATRIX) - 1:
            self._strategy_index += 1
            print(f"   ⬇️  Next strategy: '{self.strategy.name}'")
            return True
        return False


    # ───────────────────────────────────────────────────────────────
    #  CENTRAL ERROR HANDLER
    #  Order matters! Subclasses MUST be checked before parents.
    # ───────────────────────────────────────────────────────────────

    def _handle_api_error(self, e: Exception, attempt: int = 0, max_retries: int = 0) -> ErrorAction:

        # ── FATAL: Auth / Permissions / Not Found ─────────────────

        if isinstance(e, AuthenticationError):
            print(f"\n⛔ AUTH ERROR ({self.model})")
            print(f"   API Key rejected or expired.")
            print(f"   Key: {self.api_key[:6]}...")
            print(f"   Error: {e}")
            return ErrorAction.FATAL

        if isinstance(e, PermissionDeniedError):
            print(f"\n⛔ PERMISSION DENIED ({self.model})")
            print(f"   Your API key is valid but lacks access to this resource.")
            print(f"   Check your plan/tier or model permissions.")
            print(f"   Error: {e}")
            return ErrorAction.FATAL

        if isinstance(e, NotFoundError):
            print(f"\n⛔ NOT FOUND: Model '{self.model}' does not exist on {self.base_url}")
            print(f"   Error: {e}")
            return ErrorAction.FATAL

        if e.__class__.__name__ == 'BudgetExceededError':
            print(f"\n⛔ BUDGET EXCEEDED — LiteLLM proxy budget limit reached.")
            print(f"   Error: {e}")
            return ErrorAction.FATAL

        # ── SKIP: Per-item failures ───────────────────────────────

        if e.__class__.__name__ == 'ContextWindowExceededError':
            print(f"⚠️  CONTEXT WINDOW EXCEEDED ({self.model}): Input too long. Skipping.")
            print(f"   Details: {str(e)[:300]}")
            return ErrorAction.SKIP

        if e.__class__.__name__ == 'ContentPolicyViolationError':
            print(f"⚠️  CONTENT POLICY VIOLATION ({self.model}): Safety filter triggered. Skipping.")
            print(f"   Details: {str(e)[:300]}")
            return ErrorAction.SKIP

        if e.__class__.__name__ == 'UnsupportedParamsError':
            print(f"⚠️  UNSUPPORTED PARAMS ({self.model}): {str(e)[:300]}")
            return ErrorAction.SKIP

        if e.__class__.__name__ == 'JSONSchemaValidationError':
            print(f"⚠️  SCHEMA VALIDATION FAILED ({self.model}): {str(e)[:300]}")
            return ErrorAction.SKIP

        if isinstance(e, UnprocessableEntityError):
            print(f"⚠️  UNPROCESSABLE ENTITY ({self.model}): {str(e)[:300]}")
            return ErrorAction.SKIP

        # ── CONFIG ERROR: BadRequest base (after subclass checks!) ─

        if isinstance(e, BadRequestError):   
            error_text = ""
            if hasattr(e, 'body') and isinstance(e.body, dict):
                error_text = str(e.body).lower()
            else:
                error_text = str(e).lower()        
            if "invalid model" in error_text or "model name" in error_text:
                print(f"\n⛔ CRITICAL: Model Error for '{self.model}' - Not found!")
                
                # 3. String-Catching für den Prefix (wie von dir gefordert)
                if "/" in self.model:
                    prefix, actual_model = self.model.split("/", 1)
                    print(f"💡 HINT: You are using the prefix '{prefix}/'.")
                    print(f"   A possible error cause is that when using a custom base_url (Proxy/Local),")
                    print(f"   you MUST NOT use a provider prefix, since it is not using LiteLLM. The provider information is only for the LiteLLM SDK.")
                    print(f"   -> If that is the case: Change model to '{actual_model}' instead of '{self.model}'\n")
                else:
                    print(f"💡 HINT: The model name was rejected by your base_url. Call `/v1/models` to check available models.\n")
                
                return ErrorAction.FATAL
        
            # Fallback für alle anderen 400er Fehler (Context Window zu groß, falsches Schema etc.)
            print(f"⚠️ BAD REQUEST ({self.model}): {str(e)[:300]}")
            return ErrorAction.SKIP

        # ── RETRY: Transient errors ───────────────────────────────

        retry_label = f"Attempt {attempt + 1}/{max_retries + 1}"

        if isinstance(e, RateLimitError):
            print(f"🔄 RATE LIMITED ({self.model}) — {retry_label}")
            return ErrorAction.RETRY

        if isinstance(e, APITimeoutError):
            print(f"🔄 TIMEOUT ({self.model}) — {retry_label}")
            return ErrorAction.RETRY

        if isinstance(e, APIConnectionError):
            print(f"🔄 CONNECTION ERROR ({self.model}) — {retry_label}")
            return ErrorAction.RETRY

        if isinstance(e, InternalServerError) or e.__class__.__name__ == 'ServiceUnavailableError':
            print(f"🔄 SERVER ERROR ({self.model}) — {retry_label}")
            return ErrorAction.RETRY

        if isinstance(e, ConflictError):
            print(f"🔄 CONFLICT ({self.model}) — {retry_label}")
            return ErrorAction.RETRY

        if isinstance(e, APIError):
            # Generic APIError fallback — treat as retryable

            # 1. Spezifischer Check auf 402 (Insufficient Credits / Payment Required)
            status_code = getattr(e, "status_code", None)
            
            if status_code == 402:
                print(f"\n⛔ CRITICAL ERROR (402): Out of Credits or Context too large for {self.model}.")
                print(f"    Error: {e}")
                return ErrorAction.FATAL

            # 2. Generic APIError fallback — treat as retryable (z.B. 500, 502)
            print(f"🔄 API ERROR ({self.model}) — {retry_label}: {str(e)[:300]}")
            return ErrorAction.RETRY

        # ── UNKNOWN ───────────────────────────────────────────────

        print(f"💥 UNEXPECTED ERROR ({self.model}): {type(e).__name__}: {str(e)[:300]}")
        return ErrorAction.SKIP


    # ───────────────────────────────────────────────────────────────
    #  GENERATE
    # ───────────────────────────────────────────────────────────────

    async def generate(self, messages: List[Dict], schema: Any = None, max_retries=2, **kwargs) -> str:
        """
        Runs one LLM request.
        - base_url set     → OpenAI SDK (direct endpoint, no provider prefix needed)
        - base_url not set → LiteLLM (provider routing via model prefix, e.g. 'openrouter/...')
        - First request discovers the best strategy (serialized via lock).
        - All subsequent requests use the locked strategy concurrently.
        """
        if not self._connection_verified:
            await self.check_connection()

        # ── Discovery: serialize the first request to walk the matrix alone ──
        # All other requests wait at the lock until discovery is done.
        discovering = False
        if self.base_url and not self._strategy_discovered:
            await self._discovery_lock.acquire()
            if self._strategy_discovered:
                # Someone else discovered while we waited — release and continue
                self._discovery_lock.release()
            else:
                discovering = True
                print(f"🔬 Discovering best strategy for {self.model}...")

        try:
            async with self.sem:
                last_error = None
                attempt = 0

                while attempt <= max_retries:
                    try:
                        if self.base_url:
                            # ── OpenAI SDK Path ────────────────────────────
                            # kwargs go in first, strategy overwrites on top
                            strategy = self.strategy
                            call_kwargs = {
                                "model": self.model,
                                "messages": messages,
                                **kwargs,
                                "temperature": strategy.temperature,
                            }

                            # Strategy controls reasoning — always overwrites
                            if strategy.reasoning_effort:
                                call_kwargs["reasoning_effort"] = strategy.reasoning_effort
                            else:
                                call_kwargs.pop("reasoning_effort", None)

                            # Strategy controls output format — always overwrites
                            if schema:
                                if strategy.use_schema:
                                    call_kwargs["response_format"] = schema
                                elif strategy.use_json_object:
                                    call_kwargs["response_format"] = {"type": "json_object"}
                                else:
                                    # Vanilla mode — no response_format, inject JSON instructions into prompt. Temporary solution
                                    call_kwargs.pop("response_format", None)
                                    schema_json = json.dumps(schema.model_json_schema(), indent=2)
                                    patched_messages = list(messages)
                                    patched_messages[-1] = {
                                        **patched_messages[-1],
                                        "content": patched_messages[-1]["content"]
                                            + f"\n\nRespond ONLY with valid JSON matching this schema:\n{schema_json}"
                                    }
                                    call_kwargs["messages"] = patched_messages

                            response = await self.client.chat.completions.parse(**call_kwargs)

                        else:
                            # ── LiteLLM Path (no matrix, passthrough) ─────
                            import litellm
                            litellm.suppress_debug_info = True
                            from litellm import acompletion

                            call_kwargs = {
                                "model": self.model,
                                "messages": messages,
                                "api_key": self.api_key,
                                "drop_params": False,
                                **kwargs
                            }
                            if schema:
                                call_kwargs["response_format"] = schema

                            response = await acompletion(**call_kwargs)

                        # ── Success ────────────────────────────────────
                        if hasattr(response, 'usage') and response.usage:
                            GLOBAL_STATS.update(response.usage.model_dump())

                        # Lock strategy on first success
                        if discovering and not self._strategy_discovered:
                            self._strategy_discovered = True
                            print(f"   🔒 Strategy locked: '{self.strategy.name}'")

                        # Cache hint (only log once to avoid spam)
                        if not self._cache_hit_logged:
                            cache_hit = getattr(response, '_hidden_params', {}).get('cache_hit', False)
                            if cache_hit:
                                print(f"   💾 Cache hit detected — provider is caching responses.")
                                self._cache_hit_logged = True

                        return response.choices[0].message.content

                    except Exception as e:
                        action = self._handle_api_error(e, attempt, max_retries)

                        # During discovery: advance strategy on capability errors
                        # This does NOT count as a retry attempt except if it is a fatal error

                        if action == ErrorAction.FATAL:
                            exit(f"FATAL: {type(e).__name__} — Cannot continue.")

                        is_capability_error = (
                            isinstance(e, BadRequestError) or e.__class__.__name__ == 'UnsupportedParamsError'
                        ) and not (
                            e.__class__.__name__ in ('ContextWindowExceededError', 'ContentPolicyViolationError')
                        )

                        if is_capability_error and discovering and self._next_strategy():
                            continue  # same attempt counter, just different strategy


                        elif action == ErrorAction.SKIP:
                            GLOBAL_STATS.log_error()
                            return ""

                        elif action == ErrorAction.RETRY:
                            if attempt < max_retries:
                                wait_time = 0.5 * (attempt + 1)
                                print(f"   ⏳ Waiting {wait_time}s before retry...")
                                await asyncio.sleep(wait_time)
                                attempt += 1
                                continue
                            else:
                                last_error = e
                                break

                # All retries exhausted
                print(f"🔴 FAILED after {attempt + 1} attempts. Last error: {str(last_error)[:100]}")
                GLOBAL_STATS.log_error()
                return ""

        finally:
            # Release the discovery lock if we hold it
            if discovering:
                self._strategy_discovered = True  # lock at whatever level, even on failure
                if self._discovery_lock.locked():
                    self._discovery_lock.release()


    async def generate_batch(self, tasks_data: List[Dict], description="Processing") -> List[str]:
        """
        Batch helper. Expects a list of dicts with args for self.generate(), e.g.:
        [{'messages': [...], 'schema': MyModel}, ...]
        """
        if not self._connection_verified:
            await self.check_connection()

        tasks = [self.generate(**task_args) for task_args in tasks_data]
        return await tqdm_asyncio.gather(*tasks, desc=description)


    async def check_connection(self):
        """Pre-flight check: verifies API reachability and authentication."""
        if not self.base_url:
            # LiteLLM mode — no direct endpoint to check, skip pre-flight
            print(f"📡 LiteLLM mode ({self.model}) — skipping pre-flight connection check.")
            self._connection_verified = True
            return

        print(f"📡 Testing connection to {self.base_url}/models...")
        try:
            await self.client.models.list()
            print("   ✅ Connection confirmed. Server reachable")
            self._connection_verified = True

        except AuthenticationError as e:
            print(f"\n❌ FATAL: Authentication Failed.")
            print(f"   Key: {self.api_key[:6]}...")
            print(f"   Error: {e}")
            exit("FATAL: Auth Error — check your API key.")

        except PermissionDeniedError as e:
            print(f"\n❌ FATAL: Permission Denied.")
            print(f"   Your key is valid but cannot access this endpoint.")
            print(f"   Error: {e}")
            exit("FATAL: Permission Denied — check your API plan/tier.")

        except NotFoundError as e:
            # /v1/models may not exist on custom endpoints (vllm, sglang, etc.)
            print(f"   ⚡ /models endpoint not available — skipping pre-flight check.")
            print(f"   (This is normal for custom providers like vllm, sglang, etc.)")
            self._connection_verified = True

        except APIConnectionError as e:
            print(f"\n❌ FATAL: Cannot connect to API endpoint.")
            print(f"   URL: {self.base_url}")
            print(f"   Error: {e}")
            print(f"   Check: Is the URL correct? Is the server running? Firewall/proxy issues?")
            print(f"   Skip modell check with ---------------------------------------------------------------arg")
            exit(f"FATAL: Cannot connect to {self.base_url}")

        except APITimeoutError as e:
            print(f"\n❌ FATAL: Connection timed out during pre-flight check.")
            print(f"   URL: {self.base_url}")
            print(f"   Error: {e}")
            exit(f"FATAL: Timeout connecting to {self.base_url}")

        except Exception as e:
            print(f"\n❌ FATAL: Unexpected error during connection check.")
            print(f"   Type: {type(e).__name__}")
            print(f"   Error: {str(e)}")
            exit(f"FATAL: {type(e).__name__} in check_connection: {e}")