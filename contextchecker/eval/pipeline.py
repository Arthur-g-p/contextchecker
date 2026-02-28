"""
Output naming utilities for pipeline results.

These handle smart filename generation for meta-eval pipeline results.
Kept separate for now — may be refactored into utils.py later.
"""

import re


# --- Max filename length (Windows = 260, safety margin) ---
MAX_FILENAME_LEN = 200


def _abbreviate_model(model: str) -> str:
    """
    Shorten model names for filenames.
    
    'openai/gpt-oss-120b' → 'gpt-oss-120b'
    'openrouter/meta-llama/llama-3-70b-instruct' → 'llama-3-70b-instruct'
    """
    parts = model.split("/")
    short = parts[-1] if len(parts) > 1 else model
    short = re.sub(r'^(meta-|google-|anthropic-|openai-)', '', short)
    return short


def _extract_response_model(filename: str) -> str:
    """
    Extract the response model name from a msmarco filename.
    
    'msmarco_gpt4_answers.json' → 'gpt4'
    'msmarco_alpaca_7B_answers.json' → 'alpaca_7B'
    'msmarco_llama2_70b_chat_answers.json' → 'llama2_70b_chat'
    """
    name = filename.replace("msmarco_", "").replace("_answers", "").replace(".json", "")
    name = re.sub(r'_full.*$', '', name) if not name.endswith("_full") else name.replace("_full", "")
    return name or "unknown"


def build_output_filename(source_filename: str, extractor_model: str, checker_model: str) -> str:
    """
    Build a descriptive output filename. Catches path-too-long errors.
    
    Returns: 'meta_msmarco_{response_model}_ext-{extractor}_chk-{checker}.json'
    """
    resp_model = _extract_response_model(source_filename)
    ext_short = _abbreviate_model(extractor_model)
    chk_short = _abbreviate_model(checker_model)

    filename = f"meta_msmarco_{resp_model}_ext-{ext_short}_chk-{chk_short}.json"

    if len(filename) > MAX_FILENAME_LEN:
        ext_short = ext_short[:20]
        chk_short = chk_short[:20]
        filename = f"meta_msmarco_{resp_model}_ext-{ext_short}_chk-{chk_short}.json"

    if len(filename) > MAX_FILENAME_LEN:
        import hashlib
        h = hashlib.md5(f"{extractor_model}_{checker_model}".encode()).hexdigest()[:8]
        filename = f"meta_msmarco_{resp_model}_{h}.json"

    return filename
