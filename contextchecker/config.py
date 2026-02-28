import os
import json
import sys
from dotenv import load_dotenv

# Load .env from project root (works for CLI, debugger, and direct scripts)
load_dotenv()


EXTRACTOR_API_KEY = os.getenv("EXTRACTOR_API_KEY")
if not EXTRACTOR_API_KEY:
    print("CRITICAL: EXTRACTOR_API_KEY is missing from .env file.")
    sys.exit("CRITICAL: EXTRACTOR_API_KEY is missing from .env file.")


CHECKER_API_KEY = os.getenv("CHECKER_API_KEY")
if not CHECKER_API_KEY:
    print("CRITICAL: CHECKER_API_KEY is missing from .env file.")
    sys.exit("CRITICAL: CHECKER_API_KEY is missing from .env file.")

LLM_TIMEOUT = float(os.getenv("LLM_TIMEOUT", "120.0"))


def _load_prompts():
    """
    Internal function to load prompts once.
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    prompt_path = os.path.join(current_dir, 'prompt_map.json')
    
    try:
        with open(prompt_path, 'r') as file:
            print(f"Successfully loaded prompts from {prompt_path}")
            return json.load(file)
    except FileNotFoundError:
        print(f"CRITICAL: Could not find prompt_map.json file at {prompt_path}.")
        sys.exit(f"CRITICAL: Could not find prompt_map.json file at {prompt_path}.") 
    except json.JSONDecodeError:
        print(f"CRITICAL: prompt_map.json is not valid JSON.")
        sys.exit(f"CRITICAL: prompt_map.json is not valid JSON.") 
    except Exception as e:
        print(f"An unexpected error occurred loading prompts: {e}")
        sys.exit(f"An unexpected error occurred loading prompts: {e}") 

# 2. Execution (Runs ONCE on first import)
PROMPTS = _load_prompts()

# src/config.py

# Die "Matrix" ist einfach eine Liste von Versuchen.
EXTRACTION_STRATEGIES = [
    # 1. Der "Professor": Denkt nach, nutzt striktes Schema. (Beste Qualität)
    {
        "name": "high_precision",
        "params": {
            "temperature": 0.2, # research the best temp for your model!
            "max_tokens": 4000,
            "response_format": {"type": "json_schema", "hi": 2}, # https://github.com/open-webui/open-webui/issues/14867
            # Provider-Spezifisch: Reasoning aktivieren
            "extra_body": {"include_reasoning": True} 
        }
    },
    # 2. Der "Buchhalter": Kein Reasoning, nur Schema. (Schneller, kein Token-Overflow)
    {
        "name": "standard_structured",
        "params": {
            "temperature": 0.0,
            "max_tokens": 2000,
            "response_format": {"type": "json_schema","hi": 2},
            "extra_body": {"include_reasoning": False} # Reasoning explizit AUS
        }
    },
    # 3. Der "Notnagel": Text Mode + Regex. (Wenn Schema-Parsing crasht)
    {
        "name": "text_fallback",
        "params": {
            "temperature": 0.0,
            "response_format": None, # Text Mode
        },
        "use_repair_parser": True
    }
]