# utils.py
def format_prompt(template: str, placeholders: dict[str, str]) -> str:
    """
    Safely inserts strings into double-curly brace placeholders.
    """
    formatted_prompt = template
    # Iterate over the placeholders
    for key, value in placeholders.items():
        target = "{{" + key + "}}"
        formatted_prompt = formatted_prompt.replace(target, str(value))
    
    return formatted_prompt

def preflight_check_refchecker_input_file():
    # find optimal handling and informing user that refrences or whatever are sometimes empty. reject empty questions.
    pass

def preflight_check_ragchecker_input_file():
    pass

def preflight_check_evaluation_input_file():
    pass