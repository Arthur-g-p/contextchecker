import json

queries = [
    'average price for an engagement ring', 
    'history of bose',  
    'where to find a stud finder', 
    "what is the meaning of non hodgkin's lymphoma?",
    'when did sears open',
    'average conversion rate for retail store',
    'where is mount sinai in the bible',
    'what are the stages of normal labor',
    'what is a full form of i.c.c in cricket',
    'average temperature in san juan',
    'grace vanderwaal instagram',
    'what happens to body when quitting drinking',
    'what temperature should brats be cooked to',
    'can marijuana mental illness',
    'meaning of the name makayla',
    'egyptians are what ethnicity',
    'how to renew your passport on guam',
    'do you have to pay to get papers notarized?'
]

with open('results/checked_msmarco_gpt4_answers_full.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('fp_analysis.txt', 'w', encoding='utf-8') as out:
    for item in data:
        q = str(item.get('question', '')).lower()
        if any(query in q for query in queries):
            out.write(f"\n========================================\n")
            out.write(f"QUESTION: {item.get('question')}\n")
            out.write(f"========================================\n")
            out.write("CONTEXT:\n")
            for c in item.get('context', []):
                out.write(f" - {c}\n")
                
            out.write("\n========================================\n")
            out.write(f"LLM RESPONSE (The text being extracted from):\n")
            out.write(f"{item.get('response', 'N/A')}\n")
            out.write(f"========================================\n")
            
            out.write("\nGROUND TRUTH (Claude):\n")
            for gt in item.get('claude2_response_kg', []):
                out.write(f" - {gt.get('triplet')}\n")
                
            out.write("\nPREDICTED (GPT-OSS-120B):\n")
            for pred in item.get('openai/gpt-oss-120b_response_kg', []):
                out.write(f" - {pred.get('claim')}\n")
            out.write("\n\n")

print("Analysis written to fp_analysis.txt successfully.")
