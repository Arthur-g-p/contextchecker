import asyncio
from contextchecker.extractor import Extractor
from contextchecker.checker import Checker
import json
import sys
from contextchecker.schemas import InputItem
from typing import List

#INPUT_FILE = 'example/example_in_ref.json'
INPUT_FILE = 'example/content.json'
OUTPUT_FILE = 'output_results.json'


def preflight_check(data: List[dict]) -> List[InputItem]:
    print("Running checks...")
    valid_items = []
    skipped_count = 0
    
    total_response_chars = 0
    
    for idx, raw_item in enumerate(data):
        try:
            # 1. Validierung: Passt die Struktur?
            item = InputItem(**raw_item)
            
            # 2. Sanity Check: Sind die Felder leer?
            if not item.response.strip() or not item.reference.strip():
                print(f"[WARN] Item {idx} skipped: Empty response or reference.")
                skipped_count += 1
                continue
                
            valid_items.append(item)
            total_response_chars += len(item.response)
            
        except ValidationError as e:
            print(f"   [ERROR] Item {idx} invalid structure: {e}")
            skipped_count += 1

    # --- ESTIMATION ---
    n = len(valid_items)
    if n == 0:
        print("No valid items found! Aborting.")
        sys.exit("No valid items found in file.")

    # Grobe Schätzung (1 Token ~= 4 Chars)
    est_extract_tokens = total_response_chars / 4
    # Annahme: Extractor findet im Schnitt 5 Claims pro Text
    est_claims = n * 5 
    
    print("\n📊 Mission Report:")
    print(f"   • Input File:      {len(data)} items")
    print(f"   • Valid Items:     {n} (Ready to process)")
    print(f"   • Skipped:         {skipped_count} (Bad format)")
    print(f"   • Est. Extractor:  {n} Requests (~{int(est_extract_tokens)} Input Tokens)")
    print(f"   • Est. Checker:    ~{est_claims} Requests (Depends on claims found)")
    print("-" * 40)

async def main():
    # Read JSON file (input json example)
    print(f"Loading data from {INPUT_FILE}...")
    try:
        with open(INPUT_FILE, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print("Input file not found. Please create example/example_in_ref.json")
        return

    #clean_data = preflight_check(data)


    # data quality checking and if the format works. Estimation of requests!

    # Prepare data for batch processing
    all_responses = [item["response"] for item in data]
    all_references = [item["context"] for item in data]


    # Extractor (sentence)
    extractor = Extractor(model = "gemini", baseapi = "http://localhost:4000/v1") # openrouter so litellm knows what this is and how to speak to it.
    checker = Checker(model = "gemini", baseapi = "http://localhost:4000/v1")
    # check if checker works before the extractor runs too!!!
    
    # PHASE 1: MASS EXTRACTION ---
    print(f"Phase 1: Extracting Claims from {len(all_responses)} items...")
    # Ein einziger Aufruf für ALLES. Der Extractor regelt Semaphore & TQDM.
    extraction_results = await extractor.extract_batch(all_responses)

    print(f"\n🚀 Phase 2: Checking...")
    # empty fields?!!
    
    tasks_triplets = []
    tasks_references = []
    
    stats_claims_found = 0

    for i, result in enumerate(extraction_results):
        # Result kann None sein (wenn API Error) oder leere Triplets haben
        if result and result.triplets:
            claims_clean = [str(t) for t in result.triplets]
            
            tasks_triplets.append(claims_clean)
            tasks_references.append(all_references[i]) # Referenz passend zum Item
            stats_claims_found += len(claims_clean)
        else:
            # Fallback für leere Ergebnisse (damit Index i synchron bleibt) ????????????
            tasks_triplets.append([]) 
            tasks_references.append(all_references[i])

    print(f"   • Info: Found {stats_claims_found} claims to check.")

    # no claims to check == ABORT!!! WARN!!
    verdicts_batch = await checker.check_batch(tasks_triplets, tasks_references)
    print(f"\n🚀 Phase 3: Merging results...")

    final_output = []

    for i, item in enumerate(data):
        # 1. Hole die Zwischenergebnisse für Index i
        extract_res = extraction_results[i] # Das Pydantic Objekt oder None
        item_verdicts = verdicts_batch[i]   # Die Liste von Verdicts für dieses Item
        
        # 2. Baue die "claims" Struktur
        # Wir müssen sicherstellen, dass Triplets und Verdicts die gleiche Länge haben
        claims_entry = []
        
        if extract_res and extract_res.triplets and item_verdicts:
            # ZIP verbindet das i-te Triplet mit dem i-ten Verdict
            for triplet, verdict in zip(extract_res.triplets, item_verdicts):
                claims_entry.append({
                    "triplet": [triplet.subject, triplet.predicate, triplet.object],
                    "verdict": verdict.label, # Angenommen dein Verdict ist ein Objekt, sonst nur 'verdict'
                    "explanation": getattr(verdict, 'explanation', "") # Falls vorhanden
                })
        
        # 3. Baue das finale Objekt
        entry = {
            "question": item.get("question", ""),
            "reference": item.get("reference", ""),
            "response": item.get("response", ""),
            "claims": claims_entry
        }
        
        final_output.append(entry)


    # --- 4. OUTPUT TO FILE ---
    OUTPUT_FILE = "results_final.json"
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=2, ensure_ascii=False)
    


if __name__ == "__main__":
    asyncio.run(main())