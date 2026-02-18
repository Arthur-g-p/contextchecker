import os
import glob
import json
import numpy as np
from sklearn.metrics import f1_score, accuracy_score
from scipy import stats

# --- CONFIG ---
RESULTS_DIR = "results"  # Wo liegen deine 'checked_*.json' Files?
GT_KEY = "claude2_response_kg" # Das ist die Ground Truth vom Menschen

class MetaEvaluator:
    def __init__(self, extractor_model: str, checker_model: str):
        # Damit wissen wir, welche Keys wir auslesen müssen (z.B. "openrouter/gemini_label")
        self.kg_key = f"{extractor_model}_response_kg"
        self.label_key = f"{checker_model}_label"

    def evaluate_file(self, filepath: str):
        filename = os.path.basename(filepath)
        print(f"\n📊 Evaluating: {filename}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading file: {e}")
            return

        # Listen für Metriken
        gt_factual_list = []   # True/False (Ist der ganze Text wahr?)
        pred_factual_list = [] 
        
        gt_halu_rates = []     # 0.0 bis 1.0 (Wieviel % sind Lügen?)
        pred_halu_rates = []

        valid_samples = 0
        empty_predictions = 0

        for item in data:
            # 1. GROUND TRUTH CHECK
            # Wenn keine menschlichen Labels da sind, können wir nicht evaluieren.
            if GT_KEY not in item or not item[GT_KEY]:
                continue
            
            # GT: Ist ALLES 'Entailment'?
            gt_is_clean = all(t['human_label'] == 'Entailment' for t in item[GT_KEY])
            # GT: Halluzinations-Rate
            gt_rate = len([c for c in item[GT_KEY] if c['human_label'] != 'Entailment']) / len(item[GT_KEY])
            
            # 2. PREDICTION CHECK (Dein Tool)
            # Wir müssen prüfen, ob dein Tool überhaupt lief (Key existiert)
            if self.kg_key not in item:
                continue

            tool_kg = item[self.kg_key]
            
            # Logic aus dem originalen Paper (RefChecker):
            if not tool_kg:
                # "Silence is Truth" (Wenn nichts extrahiert wurde, nehmen wir an, es ist wahr)!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                # Das ist methodisch fragwürdig, aber nötig für Vergleichbarkeit.
                pred_is_clean = True 
                pred_rate = 0.0
                empty_predictions += 1
            else:
                # Pred: Ist ALLES 'Entailment'?
                pred_is_clean = all(c.get(self.label_key) == 'Entailment' for c in tool_kg)
                # Pred: Rate berechnen
                non_entailments = [c for c in tool_kg if c.get(self.label_key) != 'Entailment']
                pred_rate = len(non_entailments) / len(tool_kg)

            # Listen füllen
            gt_factual_list.append(gt_is_clean)
            pred_factual_list.append(pred_is_clean)
            
            gt_halu_rates.append(gt_rate)
            pred_halu_rates.append(pred_rate)
            
            valid_samples += 1

        if valid_samples == 0:
            print("   [WARN] No samples with Ground Truth found.")
            return

        # 3. CALCULATE METRICS
        self._print_metrics(gt_factual_list, pred_factual_list, gt_halu_rates, pred_halu_rates, valid_samples, empty_predictions)

    def _print_metrics(self, gt_fact, pred_fact, gt_rate, pred_rate, n, empty_cnt):
        # Classification Metrics
        acc = accuracy_score(gt_fact, pred_fact) * 100
        f1_fact = f1_score(gt_fact, pred_fact, pos_label=True) * 100
        f1_nonfact = f1_score(gt_fact, pred_fact, pos_label=False) * 100
        
        # Correlation Metrics (Pearson = Linear, Spearman = Rank)
        # Wir fangen Fehler ab, falls Variance 0 ist (z.B. wenn alle Rates 0 sind)
        print(gt_rate)

        print(pred_rate)
        try:
            pearson = stats.pearsonr(gt_rate, pred_rate).statistic * 100
            spearman = stats.spearmanr(gt_rate, pred_rate).statistic * 100
        except Exception:
            pearson = 0.0
            spearman = 0.0

        print(f"   • Samples:      {n} (Empty Preds: {empty_cnt})")
        print(f"   ------------------------------------------------")
        print(f"   • Accuracy:     {acc:.2f}%")
        print(f"   • F1 (Factual): {f1_fact:.2f}")
        print(f"   • F1 (Hallu):   {f1_nonfact:.2f}  <-- Wichtigster Wert!")
        print(f"   ------------------------------------------------")
        print(f"   • Pearson:      {pearson:.2f}")
        print(f"   • Spearman:     {spearman:.2f}")

if __name__ == "__main__":
    # Hier stellst du ein, welche Modelle du im vorherigen Schritt genutzt hast
    EXTRACTOR = "openai/gpt-oss-120b"
    CHECKER = "openai/gpt-oss-120b"
    
    evaluator = MetaEvaluator(EXTRACTOR, CHECKER)
    
    # Suche alle Ergebnis-Dateien
    files = glob.glob(os.path.join(RESULTS_DIR, "checked_*.json"))
    
    if not files:
        print(f"No files found in {RESULTS_DIR}")
    else:
        for f in files:
            evaluator.evaluate_file(f)