import sys
import glob
import json
import os
import asyncio
from typing import List, Optional
from pydantic import BaseModel, ValidationError

# Imports aus deiner eigenen Library
from src.extractor import Extractor
from src.checker import Checker

# --- CONFIG ---
DATA_DIR = r"C:\Users\Arthur\ragcheckerLite\data\noisy_context"
OUTPUT_DIR = "results"
FILE_PATTERN = "*_answers_full.json" # Findet alle Modelle (msmarco_gpt4..., msmarco_claude2...)

# --- DATENMODELL ---
class KGItem(BaseModel):
    triplet: List[str]
    human_label: str

# 2. Binde es in dein Hauptmodell ein
class InputItem(BaseModel):
    id: str
    response: str
    question: str
    context: List[str]
    # Jetzt weiß Pydantic genau, wie die Liste aussehen muss
    claude2_response_kg: List[KGItem] 

    class Config:
        extra = "ignore"

def preflight_check(data: List[dict], filename: str) -> List[InputItem]:
    """Validiert die Datenstruktur und gibt eine Schätzung ab."""
    print(f"\n🔍 Preflight Check for: {filename}")
    valid_items = []
    skipped_count = 0
    total_chars = 0
    
    for idx, raw_item in enumerate(data):
        try:
            # 1. Validierung & Parsing
            item = InputItem(**raw_item)
            
            # 2. Sanity Check (Leere Inhalte?)
            if not item.response.strip() or not item.context:
                skipped_count += 1
                continue
                
            valid_items.append(item)
            total_chars += len(item.response)
            
        except ValidationError:
            skipped_count += 1
            print("CRITICAL: the evaluation data does not match the expected format.")

    # --- STATS ---
    n = len(valid_items)
    if n == 0:
        print(f"   [WARN] No valid items in {filename}. Skipping file.")
        return []

    est_claims = n * 4 # Annahme: ~4 Claims pro Response
    print(f"   • Valid Items:    {n}")
    print(f"   • Skipped:        {skipped_count}")
    print(f"   • Est. Extractor: {n} Requests")
    print(f"   • Est. Checker:   ~{est_claims} Requests (Contexts will be joined)")
    
    return valid_items
async def process_single_file(filepath: str, extractor: Extractor, checker: Checker):
    """Führt die komplette Pipeline für EINE Datei aus und speichert im RefChecker-Format."""
    filename = os.path.basename(filepath)
    
    # 1. LOAD
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except Exception as e:
        print(f"Error loading {filename}: {e}")
        return

    # 2. PREFLIGHT
    valid_items = preflight_check(raw_data, filename)
    if not valid_items:
        return

    # Daten vorbereiten
    all_responses = [item.response for item in valid_items]
    # Context ist Liste -> String Join für Checker
    all_references = [item.context for item in valid_items]  

    # 3. PHASE 1: MASS EXTRACTION
    print(f"   🚀 Phase 1: Extracting claims with {extractor.model}...")
    try:
        extraction_results = await extractor.extract_batch(all_responses)
    except Exception as e:
        print(f"   [CRITICAL FAIL] Extraction died: {e}")
        return

    # 4. PHASE 2: CHECKING
    print(f"   🚀 Phase 2: Verifying claims with {checker.model}...")
    tasks_triplets = []
    tasks_references = []
    stats_claims = 0

    for i, result in enumerate(extraction_results):
        if result and result.triplets:
            # Pydantic Triplet -> String für Checker
            claims_clean = [str(t) for t in result.triplets]
            tasks_triplets.append(claims_clean)
            tasks_references.append(all_references[i])
            stats_claims += len(claims_clean)
        else:
            tasks_triplets.append([])
            tasks_references.append(all_references[i])

    if stats_claims == 0:
        print("   [WARN] No claims found in entire file. Skipping check.")
        # Wir speichern trotzdem, damit man sieht, dass nichts gefunden wurde
        verdicts_batch = [[] for _ in valid_items]
    else:
        try:
            verdicts_batch = await checker.check_batch(tasks_triplets, tasks_references)
        except Exception as e:
            print(f"   [CRITICAL FAIL] Checking died: {e}")
            return

    # 5. MERGE & SAVE (RefChecker Format)
    print(f"   💾 Phase 3: Merging & Saving...")
    final_output = []
    
    # Dynamische Keys basierend auf Model-Namen
    kg_key = f"{extractor.model}_response_kg"
    label_key = f"{checker.model}_label"

    for i, item in enumerate(valid_items):
        extract_res = extraction_results[i]
        item_verdicts = verdicts_batch[i]
        
        # Wir kopieren alle existierenden Daten aus dem Input (damit nichts verloren geht)
        entry = item.model_dump()
        
        knowledge_graph = []
        if extract_res and extract_res.triplets and item_verdicts:
            for triplet, verdict in zip(extract_res.triplets, item_verdicts):
                knowledge_graph.append({
                    "claim": [triplet.subject, triplet.predicate, triplet.object],
                    label_key: verdict.label,      # Z.B. "openai/gpt-4o_label": "Entailment"
                    "explanation": getattr(verdict, 'explanation', "")
                })
        
        # Speichern unter dem dynamischen Key
        entry[kg_key] = knowledge_graph
        final_output.append(entry)

    # Speichern
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    out_path = os.path.join(OUTPUT_DIR, f"checked_{filename}")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(final_output, f, indent=4, ensure_ascii=False)
    
    print(f"   ✅ Done. Saved to {out_path}")


async def main():
    print("🔌 Initializing Engine...")
    extractor = Extractor(baseapi = "http://localhost:4000/v1", model = "openrouter/gemini") # openrouter so litellm knows what this is and how to speak to it.
    checker = Checker(baseapi = "http://localhost:4000/v1", model = "openrouter/gemini")

    # Alle Files finden
    pattern = os.path.join(DATA_DIR, FILE_PATTERN)
    files = glob.glob(pattern)
    
    print(f"Found {len(files)} files to process in {DATA_DIR}")

    # Loop über alle gefundenen Dateien
    for filepath in files:
        await process_single_file(filepath, extractor, checker)
        print("-" * 50)

if __name__ == "__main__":
    # Windows Selector Event Loop Fix (falls nötig)
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
    asyncio.run(main())