import asyncio
from typing import List, Dict, Any, Optional
from openai import AsyncOpenAI
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, BadRequestError, AuthenticationError, NotFoundError, InternalServerError, APITimeoutError, APIError
from litellm import acompletion, APIError as LiteLLMError, supports_reasoning, get_model_info
from tqdm.asyncio import tqdm_asyncio
from sys import exit
from src.stats import GLOBAL_STATS 

class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, concurrency: int = 10):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        self.concurrency = concurrency
        
        # Instanz für Raw-Requests
        self.client = AsyncOpenAI(base_url=self.base_url, api_key=self.api_key)
        self._connection_verified = False
        
        # DIE AMPEL (SEMAPHORE) - Gilt für ALLE Requests dieses Clients
        self.sem = asyncio.Semaphore(self.concurrency)

    async def generate(self, messages: List[Dict], schema: Any = None,max_retries=2, **kwargs) -> str:
        """
        Führt EINEN Request aus.
        - Nutzt 'litellm', wenn ein Pydantic-Schema übergeben wird (für Cross-Provider Support).
        - Nutzt 'openai', wenn kein Schema da ist (für Raw Speed).
        - Retries up to X times.
        """
        if self._connection_verified == False:
            await self.check_connection()

        async with self.sem:
            last_error = None

            for attempt in range(max_retries +1):
                try:
                    if schema:
                        # Fall A: Structured Output via LiteLLM (übersetzt Pydantic für Anthropic/OpenAI etc.)
                        response = await acompletion(
                            model=self.model,
                            messages=messages,
                            api_key=self.api_key,
                            base_url=self.base_url,
                            response_format=schema, 
                            drop_params=False, # Ignoriert Parameter, die der Provider nicht kennt mit True was auch def ist. Reasoning effort?
                            **kwargs
                        )
                        if hasattr(response, 'usage'):
                            GLOBAL_STATS.update(response.usage.model_dump())
                        # Gibt den rohen JSON-String zurück (noch kein Objekt!)
                        return response.choices[0].message.content
                    
                    else:
                        # Fall B: Raw Text via OpenAI SDK
                        response = await self.client.chat.completions.create(
                            model=self.model,
                            messages=messages,
                            **kwargs
                        )
                        return response.choices[0].message.content
                
                # Critical Errors

                except AuthenticationError:
                    print(f"⛔ AUTH ERROR ({self.model}): API Key rejected/expired.")
                    exit("FATAL: Sudden Auth error after previous successfull connection! API Key expired?")

                except NotFoundError:
                    print(f"⛔ MODEL ERROR: Model '{self.model}' not found on server.")
                    exit("FATAL: MODEL ERROR: Model '{self.model}' not found on server.")

                except BadRequestError as e:
                    # WARNING: ONLY EXIT AFTER CERTAIN MODEL DEATH!! --> CAN BE REASONING ERROR
                    print(f"CONFIG ERROR: Could not use model '{self.model}'. Provider is missing or unkown.")
                    print(f"Details: {e}")
                    print(f"Hint: If using a custom endpoint (vllm, sglang), try adding 'openai/' prefix (e.g., 'openai/{self.model}')\n")
                    exit(f"CRITICAL: Could not use model '{self.model}'. Provider is missing or unkown.")
                    # 


                # 2. RETRIABLE (Warten & nochmal)
                except (APIConnectionError, RateLimitError, InternalServerError, APITimeoutError, LiteLLMError) as e:
                    if attempt < max_retries:
                        # Exponentieller Backoff (0.5s, 1.0s)
                        wait_time = 0.5 * (attempt + 1)
                        await asyncio.sleep(wait_time)
                        continue 
                    else:
                        last_error = e

                # 3. CATCH-ALL
                except Exception as e:
                    print(f"💥 SYSTEM ERROR: {e}")
                    return ""

            # Loop zu Ende -> Fail
            print(f"🔴 FAILED after {max_retries} retries. Reason: {str(last_error)[:100]}")
            GLOBAL_STATS.log_error()
            return ""
        

    async def generate_batch(self, tasks_data: List[Dict], description="Processing") -> List[str]:
        """
        Helper für Batch-Verarbeitung.
        Erwartet eine Liste von Dicts mit args für self.generate(), z.B.:
        [{'messages': [...], 'schema': MyModel}, ...]
        """
        if self._connection_verified == False:
            await self.check_connection()

        tasks = [self.generate(**task_args) for task_args in tasks_data]
        return await tqdm_asyncio.gather(*tasks, desc=description)
    

    async def check_connection(self):
        print(f"📡 Testing connection to {self.base_url}...")
        try:
            # Wir speichern das Ergebnis nicht, wir wollen nur wissen ob kein Fehler fliegt
            await self.client.models.list()
            print("   ✅ Connection confirmed.")
            self._connection_verified = True
            #info = get_model_info(model="deepseek/deepseek-chat") # not advised because often times pseudonyms or proxies are used that will make fetching impossible
            #info = supports_reasoning(model="openrouter/gemini") == True

        except AuthenticationError as e:
            print(f"\n❌ FATAL: Authentication Failed.")
            print(f"   Key: {self.api_key[:5]}...")
            print(f"   Error: {e}")
            exit("FATAL: Auth Error in check_connection")

        except APIConnectionError as e:
            print(f"\n❌ FATAL: Cannot connect to API endpoint.")
            print(f"   Url: {self.base_url}")
            print(f"   Error: {e}")
            exit(f"FATAL: Cannot connect to API endpoint: {self.base_url}") 

        except Exception as e:
            print(f"\n❌ FATAL: Unexpected error during connection check.")
            print(f"   Error: {str(e)}")
            exit(f"FATAL: System Error in check_connection: {e}")