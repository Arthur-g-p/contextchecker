"""
Enterprise Endpoint Capability Stress Test
=========================================
Goal: diagnose where failures come from (LiteLLM client gatekeeping vs upstream/proxy behavior)
and whether prefixing with `openai/` changes LiteLLM behavior.

What this test answers:
1) Does LiteLLM gatekeep params (e.g., reasoning_effort) BEFORE hitting the server?
2) Does `openai/<model>` reduce that gatekeeping (vs `openrouter/<model>`, raw, etc.)?
3) Can we call the proxy-advertised Llama model ID and at least get Vanilla Text to succeed?
4) Is the LiteLLM *proxy server* interfering (rewriting model IDs, collapsing to groups like "llama-3")?
5) OpenAI SDK baseline: if OpenAI SDK succeeds where LiteLLM fails, the issue is LiteLLM client-side.
   If both fail, the issue is proxy/upstream routing or the model truly not supported.

NOTE:
- This script intentionally tries "weird" model strings to observe rewriting/normalization.
- Fix obvious typos in keys/URLs. The OpenRouter base URL below is the typical OpenAI-compatible endpoint.
"""

import asyncio
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel

import litellm
from litellm import acompletion
from litellm.exceptions import (
    APIError as LiteLLMAPIError,
    BadRequestError as LiteLLMBadRequestError,
)

from openai import AsyncOpenAI
from openai import (
    APIConnectionError,
    RateLimitError,
    BadRequestError as OpenAIBadRequestError,
    AuthenticationError,
    NotFoundError,
    InternalServerError,
    APITimeoutError,
    APIError as OpenAIAPIError,
)

# Quiet logs
os.environ["LITELLM_LOG"] = "ERROR"

# ========= CONFIG =========
REAL_OPENROUTER_KEY = "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"  # <-- set
LOCAL_PROXY_BASE_URL = "http://localhost:4000/v1"
LOCAL_PROXY_KEY = "sk-1234"

# OpenRouter's OpenAI-compatible base URL (common):
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Optional headers OpenRouter sometimes likes (safe to leave empty)
OPENROUTER_EXTRA_HEADERS = {
    # "HTTP-Referer": "https://yourdomain.tld",
    # "X-Title": "YourAppName",
}

# ========= SCHEMA =========
class DummySchema(BaseModel):
    reason: str
    reason2: str

def pydantic_json_schema(model: BaseModel) -> Dict[str, Any]:
    # pydantic v2: model_json_schema, v1: schema()
    try:
        return model.model_json_schema()  # type: ignore[attr-defined]
    except Exception:
        return model.schema()  # type: ignore[attr-defined]

# ========= TEST MATRIX =========
@dataclass(frozen=True)
class Combo:
    name: str
    use_schema: bool
    use_json_object: bool
    reasoning_effort: Optional[str]
    # LiteLLM-only knobs
    litellm_drop_params: bool
    litellm_allow_vendor_params: bool

COMBOS: List[Combo] = [
    Combo("Vanilla Text",             use_schema=False, use_json_object=False, reasoning_effort=None,  litellm_drop_params=False, litellm_allow_vendor_params=False),
    Combo("JSON Object",              use_schema=False, use_json_object=True,  reasoning_effort=None,  litellm_drop_params=False, litellm_allow_vendor_params=False),
    Combo("JSON Schema (Strict)",     use_schema=True,  use_json_object=False, reasoning_effort=None,  litellm_drop_params=False, litellm_allow_vendor_params=False),

    Combo("Reasoning Param (raw)",    use_schema=False, use_json_object=False, reasoning_effort="low", litellm_drop_params=False, litellm_allow_vendor_params=False),
    Combo("Reasoning Param (drop)",   use_schema=False, use_json_object=False, reasoning_effort="low", litellm_drop_params=True,  litellm_allow_vendor_params=False),
    Combo("Reasoning Param (allow)",  use_schema=False, use_json_object=False, reasoning_effort="low", litellm_drop_params=False, litellm_allow_vendor_params=True),

    Combo("Reasoning + JSON Object",  use_schema=False, use_json_object=True,  reasoning_effort="low", litellm_drop_params=False, litellm_allow_vendor_params=True),
    Combo("Reasoning + JSON Schema",  use_schema=True,  use_json_object=False, reasoning_effort="low", litellm_drop_params=False, litellm_allow_vendor_params=True),
]

MESSAGES = [
    {"role": "user", "content": "Give two main reasons for the downfall of western civilization. Reply in JSON."}
]

# ========= TARGETS =========
@dataclass(frozen=True)
class Target:
    name: str
    base_url: str
    api_key: str
    extra_headers: Dict[str, str]

TARGETS_OPENAI_COMPAT = [
    Target("LOCAL_PROXY",    LOCAL_PROXY_BASE_URL, LOCAL_PROXY_KEY, {}),
    Target("OPENROUTER_OAI", OPENROUTER_BASE_URL,  REAL_OPENROUTER_KEY, OPENROUTER_EXTRA_HEADERS),
]

# LiteLLM can also route "direct via provider" without base_url (openrouter/<slug>).
# This target is LiteLLM-only (OpenAI SDK can't use it).
TARGET_LITELLM_PROVIDER_OPENROUTER = Target("OPENROUTER_LITELLM_PROVIDER", base_url="", api_key=REAL_OPENROUTER_KEY, extra_headers=OPENROUTER_EXTRA_HEADERS)

# ========= MODEL VARIANTS =========
# Proxy advertises EXACTLY these (from your /v1/models):
PROXY_MODELS = [
    "gemini",
    "deepseek-r1-0528",
    "openrouter/meta-llama/llama-3-70b-instruct",
]

# Known-good OpenRouter slugs from your earlier successful run:
OPENROUTER_MODELS_OAI_COMPAT = [
    "google/gemini-3-flash-preview",          # known to work in your logs
    "meta-llama/llama-3-70b-instruct",        # may or may not be available to your key
    # Add more if you know they exist on your key:
    # "anthropic/claude-3.5-sonnet",
    # "openai/gpt-4o-mini",
]

# Also include the "proxy-style" llama id as a deliberate stress case against OpenRouter directly:
OPENROUTER_MODELS_STRESS = [
    "openrouter/meta-llama/llama-3-70b-instruct",  # expected to fail on OpenRouter OAI endpoint
]

def build_model_variants(base_model: str, want_openai_prefix: bool = True) -> List[str]:
    """
    Variants to probe LiteLLM normalization and proxy behavior.
    - raw: base_model
    - openai/<base_model>: forces LiteLLM to treat call as OpenAI adapter (often reduces provider inference)
    - openrouter/<base_model>: triggers openrouter adapter inference in LiteLLM client
    """
    out = [base_model]
    if want_openai_prefix:
        out.append(f"openai/{base_model}")
    out.append(f"openrouter/{base_model}")
    # Special: if base already begins with openrouter/, allow "double prefix" variant
    if base_model.startswith("openrouter/"):
        out.append(f"openrouter/{base_model}")  # openrouter/openrouter/...
    return out

# ========= RESULT TRACKING =========
@dataclass
class Result:
    method: str                # "litellm" or "openai_sdk"
    target: str
    model: str
    combo: str
    ok: bool
    classification: str        # e.g. "OK", "GATEKEEP_PARAMS", "UPSTREAM_INVALID_MODEL", ...
    detail: str

def classify_litellm_error(e: Exception) -> Tuple[str, str]:
    s = str(e)

    # The one you care about: LiteLLM blocks before request is made
    if "UnsupportedParamsError" in s or "does not support parameters" in s:
        return ("GATEKEEP_PARAMS", s)

    if "LLM Provider NOT provided" in s:
        return ("CLIENT_MODEL_PREFIX", s)

    # Upstream invalid model indicators
    if "not a valid model ID" in s or "Invalid model name" in s:
        return ("UPSTREAM_INVALID_MODEL", s)

    if "Authentication" in s or "invalid api key" in s.lower() or "expired" in s.lower():
        return ("AUTH", s)

    if "RateLimit" in s or "429" in s:
        return ("RATE_LIMIT", s)

    if "timeout" in s.lower():
        return ("TIMEOUT", s)

    return ("LITELLM_ERROR", s)

def classify_openai_sdk_error(e: Exception) -> Tuple[str, str]:
    s = str(e)

    if isinstance(e, AuthenticationError):
        return ("AUTH", s)
    if isinstance(e, NotFoundError):
        return ("NOT_FOUND", s)
    if isinstance(e, RateLimitError):
        return ("RATE_LIMIT", s)
    if isinstance(e, APITimeoutError):
        return ("TIMEOUT", s)
    if isinstance(e, APIConnectionError):
        return ("CONNECTION", s)
    if isinstance(e, InternalServerError):
        return ("UPSTREAM_5XX", s)
    if isinstance(e, OpenAIBadRequestError):
        # upstream invalid model / invalid request shape
        if "not a valid model" in s.lower() or "invalid model" in s.lower() or "not found" in s.lower():
            return ("UPSTREAM_INVALID_MODEL", s)
        return ("UPSTREAM_4XX", s)
    if isinstance(e, OpenAIAPIError):
        return ("OPENAI_SDK_ERROR", s)

    return ("SDK_ERROR", s)

def short(s: str, n: int = 220) -> str:
    s = s.replace("\n", " ")
    return s if len(s) <= n else (s[: n - 3] + "...")

# ========= CALLERS =========
async def call_with_litellm(target: Target, model: str, combo: Combo) -> Result:
    # Decide response_format for LiteLLM
    response_format = None
    if combo.use_schema:
        response_format = DummySchema
    elif combo.use_json_object:
        # LiteLLM accepts dict response_format too
        response_format = {"type": "json_object"}

    kwargs: Dict[str, Any] = {}
    if combo.reasoning_effort is not None:
        kwargs["reasoning_effort"] = combo.reasoning_effort

    if combo.litellm_allow_vendor_params and combo.reasoning_effort is not None:
        # VIP pass to avoid client-side blocking (when LiteLLM supports this for the chosen adapter path)
        kwargs["allowed_openai_params"] = ["reasoning_effort"]

    # How LiteLLM decides where to send:
    # - If target.name == OPENROUTER_LITELLM_PROVIDER: do NOT pass base_url, rely on `openrouter/<slug>`
    # - Otherwise: treat as OpenAI-compatible endpoint and pass base_url
    if target.base_url:
        kwargs["base_url"] = target.base_url

    # Optional headers
    if target.extra_headers:
        kwargs["extra_headers"] = target.extra_headers

    try:
        resp = await acompletion(
            model=model,
            messages=MESSAGES,
            api_key=target.api_key,
            response_format=response_format,
            drop_params=combo.litellm_drop_params,
            **kwargs,
        )

        # Detect "reasoning" fields (best-effort; varies by provider)
        msg = resp.choices[0].message
        has_reasoning = False
        for attr in ["reasoning_content", "reasoning", "thinking"]:
            if hasattr(msg, attr) and getattr(msg, attr) is not None:
                has_reasoning = True
                break

        # Collect a tiny snippet for sanity
        content = getattr(msg, "content", None)
        snippet = short(content or "", 140)
        detail = f"has_reasoning={has_reasoning} content='{snippet}'"
        return Result("litellm", target.name, model, combo.name, True, "OK", detail)

    except Exception as e:
        cls, det = classify_litellm_error(e)
        return Result("litellm", target.name, model, combo.name, False, cls, short(det))

async def call_with_openai_sdk(target: Target, model: str, combo: Combo) -> Result:
    # OpenAI SDK does not "gatekeep" vendor params; it forwards extra_body.
    # response_format:
    response_format: Optional[Dict[str, Any]] = None
    if combo.use_schema:
        schema = pydantic_json_schema(DummySchema)
        response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "DummySchema",
                "schema": schema,
                "strict": True,
            },
        }
    elif combo.use_json_object:
        response_format = {"type": "json_object"}

    extra_body: Dict[str, Any] = {}
    if combo.reasoning_effort is not None:
        extra_body["reasoning_effort"] = combo.reasoning_effort

    # Note: target.base_url must be OpenAI-compatible.
    client = AsyncOpenAI(
        api_key=target.api_key,
        base_url=target.base_url,
        default_headers=target.extra_headers or None,
        timeout=35.0,
    )

    try:
        resp = await client.chat.completions.create(
            model=model,
            messages=MESSAGES,
            response_format=response_format,
            extra_body=extra_body if extra_body else None,
        )
        msg = resp.choices[0].message

        # Best-effort detect reasoning-like fields
        dumped = msg.model_dump() if hasattr(msg, "model_dump") else {}
        has_reasoning = any(k in dumped and dumped.get(k) for k in ["reasoning", "reasoning_content", "thinking"])

        snippet = short(getattr(msg, "content", "") or "", 140)
        detail = f"has_reasoning={has_reasoning} content='{snippet}'"
        return Result("openai_sdk", target.name, model, combo.name, True, "OK", detail)

    except Exception as e:
        cls, det = classify_openai_sdk_error(e)
        return Result("openai_sdk", target.name, model, combo.name, False, cls, short(det))

# ========= RUNNER =========
def print_header(title: str) -> None:
    print("\n" + "#" * 90)
    print(title)
    print("#" * 90)

def print_model_header(model: str) -> None:
    print("\n" + "=" * 90)
    print(f"🤖 MODEL: {model}")
    print("=" * 90)

def print_result(r: Result) -> None:
    icon = "✅" if r.ok else "⛔"
    print(f"{icon} [{r.method:<9}] [{r.target:<22}] {r.combo:<22} | model='{r.model}' | {r.classification} | {r.detail}")

def summarize(results: List[Result]) -> None:
    print_header("SUMMARY")

    # Overall counts
    total = len(results)
    ok = sum(1 for r in results if r.ok)
    fail = total - ok
    print(f"Total tests: {total} | ✅ OK: {ok} | ⛔ Fail: {fail}")

    # Fail buckets
    buckets: Dict[str, int] = {}
    for r in results:
        if not r.ok:
            buckets[r.classification] = buckets.get(r.classification, 0) + 1
    if buckets:
        print("\nFailure buckets:")
        for k, v in sorted(buckets.items(), key=lambda kv: (-kv[1], kv[0])):
            print(f"  - {k:<22}: {v}")

    # Key diagnostic: LiteLLM gatekeeping incidence
    gk = [r for r in results if (not r.ok and r.method == "litellm" and r.classification == "GATEKEEP_PARAMS")]
    print(f"\nLiteLLM param-gatekeeps: {len(gk)}")
    if gk[:8]:
        for r in gk[:8]:
            print(f"  - [{r.target}] {r.model} :: {r.combo} -> {short(r.detail, 160)}")
        if len(gk) > 8:
            print(f"  ... +{len(gk)-8} more")

    # Key diagnostic: Vanilla Text should succeed for proxy models (your expectation)
    vanilla_proxy_fails = [
        r for r in results
        if (r.combo == "Vanilla Text" and not r.ok and r.target == "LOCAL_PROXY")
    ]
    print(f"\nLOCAL_PROXY 'Vanilla Text' failures: {len(vanilla_proxy_fails)}")
    if vanilla_proxy_fails:
        for r in vanilla_proxy_fails[:10]:
            print(f"  - [{r.method}] model='{r.model}' -> {r.classification} :: {short(r.detail, 200)}")

    # Compare LiteLLM vs OpenAI SDK for same target/model/combo
    # (If SDK succeeds and LiteLLM fails, it’s client-side (gatekeeping/rewriting).)
    index: Dict[Tuple[str, str, str], Dict[str, Result]] = {}
    for r in results:
        key = (r.target, r.model, r.combo)
        index.setdefault(key, {})[r.method] = r

    mismatches = []
    for key, by_method in index.items():
        if "litellm" in by_method and "openai_sdk" in by_method:
            if by_method["openai_sdk"].ok and not by_method["litellm"].ok:
                mismatches.append((key, by_method["litellm"], by_method["openai_sdk"]))

    print(f"\nCases where OpenAI SDK OK but LiteLLM FAIL (strong hint: LiteLLM client-side issue): {len(mismatches)}")
    for (target, model, combo), lres, sres in mismatches[:12]:
        print(f"  - [{target}] model='{model}' combo='{combo}' :: LiteLLM={lres.classification} | SDK=OK")

async def main() -> None:
    results: List[Result] = []
    print_header("🚀 Starting Enterprise Endpoint Capability Stress Test")

    # -------------------------------
    # 1) LOCAL PROXY tests
    # -------------------------------
    print_header("TARGET: LOCAL_PROXY (OpenAI-compatible endpoint)")
    for base_model in PROXY_MODELS:
        variants = build_model_variants(base_model, want_openai_prefix=True)

        # Keep the print readable: show which variants are being tested
        print_model_header(f"BASE='{base_model}' VARIANTS={variants}")

        for model_variant in variants:
            for combo in COMBOS:
                # LiteLLM call to proxy
                r1 = await call_with_litellm(TARGETS_OPENAI_COMPAT[0], model_variant, combo)
                results.append(r1)
                print_result(r1)

                # OpenAI SDK call to proxy (uses EXACT model string, no provider prefixes!)
                # Only run SDK if the model_variant is "raw" (no openai/ or openrouter/ prefix),
                # because the proxy won't list "openai/<id>" as a model.
                if model_variant == base_model:
                    r2 = await call_with_openai_sdk(TARGETS_OPENAI_COMPAT[0], base_model, combo)
                    results.append(r2)
                    print_result(r2)

                # tiny pacing to keep logs readable / reduce rate-limit flukes
                await asyncio.sleep(0.05)

    # -------------------------------
    # 2) DIRECT OPENROUTER tests (OpenAI-compatible base_url)
    # -------------------------------
    print_header("TARGET: OPENROUTER_OAI (OpenAI-compatible endpoint)")
    for base_model in OPENROUTER_MODELS_OAI_COMPAT + OPENROUTER_MODELS_STRESS:
        variants = build_model_variants(base_model, want_openai_prefix=True)
        print_model_header(f"BASE='{base_model}' VARIANTS={variants}")

        for model_variant in variants:
            for combo in COMBOS:
                # LiteLLM to OpenRouter as OpenAI-compatible endpoint (base_url provided)
                r1 = await call_with_litellm(TARGETS_OPENAI_COMPAT[1], model_variant, combo)
                results.append(r1)
                print_result(r1)

                # OpenAI SDK to OpenRouter (ONLY with raw base_model, not prefixed variants)
                if model_variant == base_model:
                    r2 = await call_with_openai_sdk(TARGETS_OPENAI_COMPAT[1], base_model, combo)
                    results.append(r2)
                    print_result(r2)

                await asyncio.sleep(0.05)

    # -------------------------------
    # 3) DIRECT OPENROUTER tests via LiteLLM provider routing (openrouter/<slug>)
    # -------------------------------
    print_header("TARGET: OPENROUTER_LITELLM_PROVIDER (LiteLLM provider routing, no base_url)")
    provider_models = [
        "openrouter/google/gemini-3-flash-preview",  # known good in your previous logs
        "openrouter/meta-llama/llama-3-70b-instruct", # may or may not exist for your key
        # include a deliberate weird one:
        "openrouter/openrouter/meta-llama/llama-3-70b-instruct",
    ]
    for model in provider_models:
        print_model_header(model)
        for combo in COMBOS:
            r = await call_with_litellm(TARGET_LITELLM_PROVIDER_OPENROUTER, model, combo)
            results.append(r)
            print_result(r)
            await asyncio.sleep(0.05)

    summarize(results)

if __name__ == "__main__":
    # If you want deeper LiteLLM debugging, uncomment:
    # litellm._turn_on_debug()
    asyncio.run(main())
