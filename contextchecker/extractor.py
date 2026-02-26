from typing import List
import random
import contextchecker.config as config
from contextchecker.utils import format_prompt
from contextchecker.llmclient import LLMClient
from contextchecker.schemas import ExtractionResult, MissingClaimsResult, ValidationBatchResult
import string


PUNCTUATION = "!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~"
REMOVE_PUNCTUATION = str.maketrans('', '', PUNCTUATION)

REFUSAL_PHRASES = (
    "i dont know",
    "i cannot answer",
    "not provided in the context",
    "i dont have enough information",
    "information not provided"
)

def _is_full_abstention(text: str, threshold: float = 0.85) -> bool:
    if not text or not text.strip():
        return True

    # 1. Clean (C-optimiert)
    clean_text = text.lower().translate(REMOVE_PUNCTUATION)
    
    # 2. Normalize (C-optimiert)
    clean_text = " ".join(clean_text.split())
    
    if not clean_text:
        return True

    text_length = len(clean_text)

    # 3. Coverage Check
    for phrase in REFUSAL_PHRASES:
        if phrase in clean_text:
            if (len(phrase) / text_length) >= threshold:
                return True
                
    return False

class Extractor:
    def __init__(self, model: str = "", baseapi: str = None, concurrency: int = 10, 
                 validate_pct: int = 0, add_missing_pct: int = 0):
        self.model = model
        self.client = LLMClient(
            api_key=config.EXTRACTOR_API_KEY,
            model=model,
            base_url=baseapi,
            concurrency=concurrency 
        )
        self.prompts = config.PROMPTS
        
        # 0 bis 100 Prozent
        self.validate_pct = validate_pct
        self.add_missing_pct = add_missing_pct


    async def extract_batch(self, responses: List[str]) -> List[ExtractionResult]:
        print(f"🚀 Phase 1: Base Extraction for {len(responses)} responses...")

        # --- 1. BASE EXTRACTION ---
        results = [None] * len(responses)
        batch_tasks = []
        task_indices = []

        # --- 1. PRE-FILTER & TASK CREATION ---
        for i, response in enumerate(responses):
            if _is_full_abstention(response):
                # Sofortiges Abstain, kein LLM Call
                results[i] = ExtractionResult(triplets=[])
            else:
                # Relevanter Text, ab in den Batch
                prompt_text = format_prompt(self.prompts["extractor_prompt"], {"text": response})
                batch_tasks.append({
                    "messages": [
                        {"role": "system", "content": "Extract knowledge triplets."},
                        {"role": "user", "content": prompt_text}
                    ],
                    "schema": ExtractionResult,
                    "temperature": 0.0,
                    "reasoning_effort": "low"
                })
                task_indices.append(i)

        # --- 2. EXECUTE BATCH (Nur für die, die nicht gefiltert wurden) ---
        if batch_tasks:
            raw_responses = await self.client.generate_batch(batch_tasks, description="Extracting")

            # --- 3. PARSING & MAPPING ---
            for idx_in_batch, original_idx in enumerate(task_indices):
                raw_text = raw_responses[idx_in_batch]
                try:
                    results[original_idx] = ExtractionResult.model_validate_json(raw_text)
                except Exception as e:
                    # Retry-Matrix Platzhalter
                    results[original_idx] = ExtractionResult(triplets=[])
                    
        return results


        #await self.calc_prec(responses, results)
        #await self.calc_score(responses, results)
        # --- 2. OPTIONAL: VALIDATION (Bad Claims filtern) ---
        if self.validate_pct > 0:
            #results = await self._run_validation(responses, results)
            pass

        # --- 3. OPTIONAL: ADD MISSING CLAIMS ---
        if self.add_missing_pct > 0:
            #results = await self._run_add_missing(responses, results)
            pass

        return results
    

    async def _run_validation(self, texts: List[str], extraction_results: List[ExtractionResult]) -> List[ValidationBatchResult]:
        print(f"🔍 Running Validation on {len(texts)} texts...")
        val_tasks = []
        indices_to_validate = []
        
        # Leere Ergebnisse vorbereiten, damit die Indizes am Ende stimmen
        final_results = [None] * len(texts)

        for i, (text, ext_res) in enumerate(zip(texts, extraction_results)):
            if ext_res and len(ext_res.triplets) > 0:
                claims_str = "\n".join([str(t) for t in ext_res.triplets])
                prompt_text = format_prompt(self.prompts["claim_verifcation_prompt"], {"text": text, "claims_str": claims_str})
                
                val_tasks.append({
                    "messages": [{"role": "user", "content": prompt_text}],
                    "schema": ValidationBatchResult, # Hier nutzen wir den Wrapper!
                    "temperature": 0.0
                })
                indices_to_validate.append(i)

        if not val_tasks:
            return final_results

        raw_vals = await self.client.generate_batch(val_tasks, description="Validating TP/FP")

        for idx_in_batch, original_idx in enumerate(indices_to_validate):
            try:
                final_results[original_idx] = ValidationBatchResult.model_validate_json(raw_vals[idx_in_batch])
            except Exception as e:
                print(f"Validation parsing failed for index {original_idx}: {e}")
                
        return final_results

    async def _run_validation_old(self, texts: List[str], current_results: List[ExtractionResult]) -> List[ExtractionResult]:
        print(f"🧹 Phase 2: Validating claims (Target: {self.validate_pct}% of items)...")
        
        val_tasks = []
        indices_to_validate = []

        # Entscheiden, wer validiert wird
        for i, (text, ext_res) in enumerate(zip(texts, current_results)):
            # Random Check UND es müssen überhaupt Claims da sein
            if len(ext_res.triplets) > 0 and random.randint(1, 100) <= self.validate_pct:
                claims_str = "\n".join([str(t) for t in ext_res.triplets])
                prompt_text = format_prompt(self.prompts["claim_verifcation_prompt"], {"text": text, "claims_str": claims_str})

                val_tasks.append({
                    "messages": [{"role": "user", "content": prompt_text}],
                    #"schema": ValidationResult,
                    "temperature": 0.0
                })
                indices_to_validate.append(i)

        if not val_tasks:
            return current_results

        # Batch abschicken
        raw_vals = await self.client.generate_batch(val_tasks, description="Validating")

        # Ergebnisse anwenden (Filtern)
        for idx_in_batch, original_idx in enumerate(indices_to_validate):
            try:
                val_obj = ValidationResult.model_validate_json(raw_vals[idx_in_batch])
                original_claims = current_results[original_idx].triplets
                
                # Wir behalten nur die Claims, bei denen das LLM 'True' (is_faithful) gesagt hat.
                # zip() ist sicher, solange das LLM die Listenlänge respektiert hat.
                filtered_claims = [
                    claim for claim, is_valid in zip(original_claims, val_obj.is_faithful) 
                    if is_valid
                ]
                # Überschreiben mit sauberer Liste
                current_results[original_idx].triplets = filtered_claims
                
            except Exception:
                pass # Wenn Validator crasht, behalten wir die Original-Claims

        return current_results


    async def _run_add_missing(self, texts: List[str], extraction_results: List[ExtractionResult]) -> List[MissingClaimsResult]:
        print(f"➕ Running Missing Claims Adder on {len(texts)} texts...")
        add_tasks = []
        
        for text, ext_res in zip(texts, extraction_results):
            claims_str = "\n".join([str(t) for t in ext_res.triplets]) if ext_res and ext_res.triplets else "None"
            prompt = f"Text: {text}\nAlready extracted:\n{claims_str}\nExtract ONLY missing atomic facts."
            
            add_tasks.append({
                "messages": [{"role": "user", "content": prompt}],
                "schema": MissingClaimsResult,
                "temperature": 0.2
            })

        raw_adds = await self.client.generate_batch(add_tasks, description="Finding FN")
        
        final_results = []
        for raw in raw_adds:
            try:
                final_results.append(MissingClaimsResult.model_validate_json(raw))
            except Exception as e:
                final_results.append(MissingClaimsResult(triplets=[]))
                
        return final_results


    # ----------------------------------------------------------------
    # 2. DIE PUBLIC API (Die Funktionen, die du aufrufst)
    # ----------------------------------------------------------------

    async def calc_prec(self, texts: List[str], extraction_results: List[ExtractionResult]) -> float:
        """
        Berechnet NUR die Precision. Ruft den Validator im Hintergrund auf.
        """
        assert len(texts) == len(extraction_results), "Mismatched lists!"
        
        # 1. API Call
        validation_batches = await self._run_validation(texts, extraction_results)
        
        # 2. Zählen
        tp, fp = 0, 0
        for val_batch in validation_batches:
            if val_batch:
                for val in val_batch.results:
                    if val.is_faithful:
                        tp += 1
                    else:
                        fp += 1
                        
        # 3. Mathe
        if (tp + fp) == 0:
            return 0.0
            
        precision = tp / (tp + fp)
        print(f"📊 Precision: {precision*100:.2f}% (TP: {tp}, FP: {fp})")
        return precision

    async def calc_score(self, texts: List[str], extraction_results: List[ExtractionResult]) -> dict:
        """
        Berechnet Precision, Recall und F1. Ruft Validator UND Adder auf.
        """
        assert len(texts) == len(extraction_results), "Mismatched lists!"
        
        # 1. API Calls (laufen nacheinander, da sie den gleichen Semaphore Pool nutzen)
        validation_batches = await self._run_validation(texts, extraction_results)
        missing_batches = await self._run_add_missing(texts, extraction_results)
        
        # 2. Zählen (TP und FP)
        tp, fp = 0, 0
        for val_batch in validation_batches:
            if val_batch:
                for val in val_batch.results:
                    if val.is_faithful:
                        tp += 1
                    else:
                        fp += 1
                        
        # 3. Zählen (FN)
        fn = 0
        for missing_batch in missing_batches:
            if missing_batch:
                fn += len(missing_batch.triplets)
                
        # 4. Mathe
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        stats = {
            "Precision": round(precision * 100, 2),
            "Recall": round(recall * 100, 2),
            "F1_Score": round(f1 * 100, 2),
            "Raw": {"TP": tp, "FP": fp, "FN": fn}
        }
        
        print(f"📊 Final Scores: Precision: {stats['Precision']}% | Recall: {stats['Recall']}% | F1: {stats['F1_Score']}%")
        return stats


    async def _run_add_missing_old(self, texts: List[str], current_results: List[ExtractionResult]) -> List[ExtractionResult]:  
        print(f"➕ Phase 3: Adding missing claims (Target: {self.add_missing_pct}% of items)...")
        
        add_tasks = []
        indices_to_add = []

        for i, (text, ext_res) in enumerate(zip(texts, current_results)):
            if random.randint(1, 100) <= self.add_missing_pct:
                claims_str = "\n".join([str(t) for t in ext_res.triplets]) if ext_res.triplets else "None"
                prompt = f"Text: {text}\nAlready extracted:\n{claims_str}\nExtract ONLY missing atomic facts."
                
                add_tasks.append({
                    "messages": [{"role": "user", "content": prompt}],
                    "schema": MissingClaimsResult,
                    "temperature": 0.2 # Etwas mehr Kreativität für Recall
                })
                indices_to_add.append(i)

        if not add_tasks:
            return current_results

        raw_adds = await self.client.generate_batch(add_tasks, description="Adding Claims")

        # Ergebnisse anwenden (Anhängen)
        for idx_in_batch, original_idx in enumerate(indices_to_add):
            try:
                add_obj = MissingClaimsResult.model_validate_json(raw_adds[idx_in_batch])
                # Füge die neuen Claims einfach hinten dran (Formatierung als Triplet-Objekt vorausgesetzt)
                # *Hier müsstest du idealerweise sicherstellen, dass die neuen Claims auch Triplets sind*
                current_results[original_idx].triplets.extend(add_obj.missing_claims) 
            except Exception:
                pass 

        return current_results
    

        """
        Temporäre Benchmarking-Funktion, um Precision, Recall und F1 
        wie im RefChecker-Paper zu berechnen.
        """
        print("\n" + "="*50)
        print("🔬 STARTING EXTRACTOR META-EVALUATION")
        print("="*50)

        total_tp = 0  # True Positives (Wahre Claims)
        total_fp = 0  # False Positives (Halluzinationen)
        total_fn = 0  # False Negatives (Vergessene Claims)

        for i, (text, ext_res) in enumerate(zip(texts, current_results)):
            if not ext_res.triplets:
                print(f"Skipping Item {i}: No claims extracted initially.")
                continue

            claims_str = "\n".join([str(t) for t in ext_res.triplets])

            # ---------------------------------------------------------
            # SCHRITT 1: VERIFICATION (T/F LABELER) -> Für Precision
            # ---------------------------------------------------------
            val_prompt = format_prompt(self.prompts["claim_verifcation_prompt"], {
                "text": text, 
                "claims_str": claims_str
            })
            
            # Wir rufen den LLMClient auf (hier ohne Schema, wenn dein Prompt das JSON selbst erzwingt, 
            # ansonsten pass das schema=ValidationResult an)
            raw_val = await self.client.generate(
                messages=[{"role": "user", "content": val_prompt}],
                schema=ValidationResult, # WICHTIG: Erwartet is_faithful: List[bool]
                temperature=0.0
            )

            try:
                val_obj = ValidationResult.model_validate_json(raw_val)
                is_faithful_list = val_obj.is_faithful
                
                # Zählen & Fehler printen
                item_tp = 0
                item_fp = 0
                print(f"\n--- Item {i} False Claims ---")
                
                for claim, is_true in zip(ext_res.triplets, is_faithful_list):
                    if is_true:
                        item_tp += 1
                    else:
                        item_fp += 1
                        print(f"❌ Hallucination: {claim}")
                        
                if item_fp == 0:
                    print("✅ All extracted claims are faithful.")

                total_tp += item_tp
                total_fp += item_fp

            except Exception as e:
                print(f"Failed to parse validation for Item {i}: {e}")
                continue

            # ---------------------------------------------------------
            # SCHRITT 2: COMPLETION (ADDER) -> Für Recall
            # ---------------------------------------------------------
            add_prompt = format_prompt(self.prompts["complete_claim_set_prompt"], {
                "text": text, 
                "claims_str": claims_str
            })

            raw_add = await self.client.generate(
                messages=[{"role": "user", "content": add_prompt}],
                schema=MissingClaimsResult, # WICHTIG: Erwartet missing_claims: List[str]
                temperature=0.2
            )

            try:
                add_obj = MissingClaimsResult.model_validate_json(raw_add)
                item_fn = len(add_obj.missing_claims)
                total_fn += item_fn
                
                if item_fn > 0:
                    print(f"⚠️ Missed {item_fn} claims:")
                    for mc in add_obj.missing_claims:
                        print(f"   - {mc}")
            except Exception as e:
                print(f"Failed to parse missing claims for Item {i}: {e}")
                continue

        # ---------------------------------------------------------
        # SCHRITT 3: DIE ABRECHNUNG (MATH)
        # ---------------------------------------------------------
        print("\n" + "="*50)
        print("📊 FINAL METRICS (Paper Style)")
        print("="*50)
        print(f"Total Extracted Claims: {total_tp + total_fp}")
        print(f"  - True Positives (Faithful): {total_tp}")
        print(f"  - False Positives (Errors):  {total_fp}")
        print(f"Total Missing Facts (FN):      {total_fn}")
        print("-" * 50)

        # Division by zero protection
        precision = (total_tp / (total_tp + total_fp)) * 100 if (total_tp + total_fp) > 0 else 0.0
        recall = (total_tp / (total_tp + total_fn)) * 100 if (total_tp + total_fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"Precision: {precision:.2f}% (How accurate was the extraction?)")
        print(f"Recall:    {recall:.2f}% (How complete was the extraction?)")
        print(f"F1 Score:  {f1:.2f}  (Overall Quality)")
        print("="*50 + "\n")