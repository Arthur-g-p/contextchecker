##
#the actual meta evaluation. it looks at the numbers between abstention rates and looks for correlations!!!
#here are many good numbers. so wie eine wetterrader das immer sonne anzeigt. hat 99% wahrscheinlichkeit aber nicht das was wir brauchen

#I could run this complete eval pipeline to test a model from front to end. However: thats the meta eval. what about testing each seperatly in this eval script?
#tract eval --> in paper: they tasked a model to add missing triplets did T/F label on existing ones. then looked for human correlation on this task.

#however we are working here with a gt, which we want to utilize. Which is the reason for all the NLI CHecks to find best prompts

#checker eval --> comparing verdics to gt
#metaeval --> what we just said

#übrigens: div by zero crash.

import asyncio
from contextchecker.extractor import Extractor
from contextchecker.checker import Checker
import json
import sys
from contextchecker.schemas import InputItem
from typing import List

INPUT_FILE = 'example/example_in_ref.json'
OUTPUT_FILE = 'output_results.json'

# complete process: get all

# run solo script from cli
# input

async def main():
    # First read the msmacro file to actually get the prompts we look for
    print(f"Loading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Input file not found. Please create example/example_in_ref.json")
        return

	# again, preflight check (standard)
    
    # Prepare data for batch processing
    all_responses = [item["response"] for item in data]
    all_references = [item["reference"] for item in data]



	# init checkers
    extractor = Extractor(baseapi = "http://localhost:4000/v1", model = "openrouter/gemini") # openrouter so litellm knows what this is and how to speak to it.
    checker = Checker(baseapi = "http://localhost:4000/v1", model = "openrouter/gemini")


    # --- 4. OUTPUT TO FILE ---
    OUTPUT_FILE = "results_final.json"
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    


if __name__ == "__main__":
    asyncio.run(main())