import asyncio
import os
from pydantic import BaseModel
from litellm import acompletion
from litellm.exceptions import APIError, BadRequestError
from litellm import acompletion, APIError as LiteLLMError, supports_reasoning, get_model_info
from openai import AsyncOpenAI, APIConnectionError, RateLimitError, BadRequestError, AuthenticationError, NotFoundError, InternalServerError, APITimeoutError, APIError

# Damit wir keinen Spam in der Konsole haben
os.environ["LITELLM_LOG"] = "ERROR" 

# ==========================================
# 🔑 TRAG HIER DEINEN ECHTEN OPENROUTER KEY EIN
# ==========================================
REAL_OPENROUTER_KEY = "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" 

# Ein simples Schema für den Test
class DummySchema(BaseModel):
    reason: str
    reason2: str

async def test_calls():
    # Deine Modelle
    models = [
        #"openrouter/gemini", # Oder gemini-flash, je nachdem was du hast
        #"openrouter/deepseek-r1-0528",  # OpenRouter Prefix wichtig!
        #"google/gemini-3-flash-preview",
        "openrouter/openrouter/meta-llama/llama-3-70b-instruct"
        "openai/meta-llama/llama-3-70b-instruct",
        "openai/openrouter/meta-llama/llama-3-70b-instruct"
    ]
    
    # Die Matrix der Wahrheit
    combinations = [
        {"reasoning": None,  "drop": False, "schema": False, "force": False, "name": "Vanilla Text"},
        {"reasoning": None,  "drop": False, "schema": True,  "force": False, "name": "Vanilla JSON"},
        
        # Der Türsteher blockiert das hier:
        {"reasoning": "low", "drop": False, "schema": False, "force": False, "name": "Reasoning (LiteLLM Gatekeeper)"},
        
        # Der Türsteher wirft das Reasoning weg:
        {"reasoning": "low", "drop": True,  "schema": False, "force": False, "name": "Reasoning (Drop/Safe)"},
        
        # HIER IST DIE MAGIE: Wir zwingen LiteLLM, es durchzulassen!
        {"reasoning": "low", "drop": False, "schema": False, "force": True,  "name": "Reasoning (Force Bypass!)"},
        {"reasoning": "low", "drop": False, "schema": True,  "force": True,  "name": "Reasoning JSON (Force Bypass!)"},
    ]

    messages = [{"role": "user", "content": "What is are the two main reasons for the downfall of western civilazation? Reply in JSON."}]

    # 🌍 HIER DEFINIEREN WIR BEIDE ZIELE (Proxy & Direct)
    endpoints = [
        {"name": "LOCAL PROXY", "base_url": "http://localhost:4000/v1", "api_key": "sk-1234"},
        {"name": "DIRECT OPENROUTER", "base_url": None, "api_key": REAL_OPENROUTER_KEY} # base_url=None lässt LiteLLM automatisch an OpenRouter senden
    ]

    print("🚀 Starting API Capability Stress Test...\n")

    for endpoint in endpoints:
        print(f"\n{'#'*70}")
        print(f"🌍 RUNNING TESTS AGAINST: {endpoint['name']}")
        print(f"{'#'*70}\n")

        for model in models:
            print(f"{'='*50}")
            print(f"🤖 MODEL: {model}")
            print(f"{'='*50}")
            last_error = ""
            
            for combo in combinations:
                if "LLM Provider NOT provided" in last_error or "Invalid model name" in last_error:
                    print("skipping dead model: "+model)
                    continue

                schema_param = DummySchema if combo["schema"] else None
                reasoning_param = combo["reasoning"]
                drop_param = combo["drop"]
                force_bypass = combo["force"]
                
                print(f"Test: {combo['name']:<30} | Drop: {str(drop_param):<5} | Schema: {str(combo['schema']):<5}")
                
                kwargs = {}
                if reasoning_param:
                    kwargs["reasoning_effort"] = reasoning_param
                    
                # Der VIP Pass für LiteLLM
                if force_bypass:
                    kwargs["allowed_openai_params"] = ["reasoning_effort"]
                    
                # Base URL nur anhängen, wenn wir den Proxy testen (bei Direct lassen wir LiteLLM routen)
                if endpoint["base_url"]:
                    kwargs["base_url"] = endpoint["base_url"]

                try:
                    response = await acompletion(
                        model=model,
                        messages=messages,
                        response_format=schema_param,
                        drop_params=drop_param,
                        api_key=endpoint["api_key"],
                        **kwargs
                    )
                    
                    has_reasoning = hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content is not None
                    print(f"  ✅ SUCCESS (Actually Reasoned: {has_reasoning})")
                    
                except BadRequestError as e:
                    error_msg = str(e)
                    # Wir checken den String auf die typischen Naming/Provider Fehler
                    if "LLM Provider NOT provided" in error_msg or "Invalid model name" in error_msg:
                        print(f"⛔ MODEL NAME ERROR: Check your provider prefix or model name '{model}'.")
                        print(f"   Details: {error_msg}")
                        last_error = error_msg
                        continue
                    else:
                        print(f"⛔ BAD REQUEST ({model}): Check Context Window or JSON structure. Details: {e}")

                except AuthenticationError:
                    print(f"⛔ AUTH ERROR ({model}): API Key is invalid or expired.")

                except NotFoundError:
                    print(f"⛔ NOT FOUND ERROR ({model}): Endpoint or Model missing.")

                # 2. RETRIABLE (Warten & nochmal)
                except (APIConnectionError, RateLimitError, InternalServerError, APITimeoutError, LiteLLMError) as e:
                    print(f"RETRIABLE Details: {e}")

                # 3. CATCH-ALL
                except Exception as e:
                    print(f"💥 SYSTEM ERROR: {e}")

if __name__ == "__main__":
    asyncio.run(test_calls())