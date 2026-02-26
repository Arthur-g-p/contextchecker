import os
import glob
import json
from contextchecker.utils import f1_score, accuracy_score, confusion_matrix, stats
# --- CONFIG ---
RESULTS_DIR = "results"  
GT_KEY = "claude2_response_kg" 

class MetaEvaluator:
    def __init__(self, extractor_model: str, checker_model: str):
        self.kg_key = f"{extractor_model}_response_kg"
        self.label_key = f"{checker_model}_label"

    def evaluate_file(self, filepath: str, ignore_abstentions: bool = True):
        """
        Evaluates the generated labels against the ground truth.
        
        Args:
            filepath: Path to the JSON file.
            ignore_abstentions: 
                - True: Skips items where the model abstained (returned []).
                - False (Legacy Mode/RefChecker): Treats abstentions as 100% factual/0% hallucination.
        """
        filename = os.path.basename(filepath)
        mode_str = "HONEST MODE (Abstentions skipped)" if ignore_abstentions else "LEGACY MODE (Silence is Truth)"
        
        print(f"\n{'='*60}")
        print(f"📊 EVALUATING: {filename}")
        print(f"⚙️  MODE: {mode_str}")
        print(f"{'='*60}")
        
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading file: {e}")
            return

        # Listen für die Metriken
        gt_factual_list = []   
        pred_factual_list = [] 
        gt_halu_rates = []     
        pred_halu_rates = []

        # Counter für lückenlose Mathematik
        total_items = len(data)
        missing_gt_count = 0
        missing_pred_key_count = 0
        abstention_count = 0
        evaluated_count = 0

        for item in data:
            # 1. GROUND TRUTH CHECK
            if GT_KEY not in item or not item[GT_KEY]:
                missing_gt_count += 1
                continue
            
            # 2. PREDICTION KEY CHECK (Lief das Tool überhaupt?)
            if self.kg_key not in item:
                missing_pred_key_count += 1
                continue

            tool_kg = item[self.kg_key]
            
            # 3. ABSTENTION CHECK (Leere Extraktion / "Weiß ich nicht")
            if not tool_kg:
                abstention_count += 1
                
                if ignore_abstentions:
                    # HONEST MODE: Wir werten dieses Item nicht für die Halluzinations-Rate.
                    # Grund: Wer nichts behauptet, kann nicht lügen.
                    continue 
                else:
                    # LEGACY MODE (RefChecker Logic):
                    # Wir zwingen einen perfekten Score in die Liste.
                    # Das bläht die False Negatives (FN) und True Negatives (TN) massiv auf.
                    pred_is_clean = True 
                    pred_rate = 0.0
            else:
                # 4a. EVALUATION (Normaler Durchlauf mit Claims)
                # --- Prediction ---
                pred_is_clean = all(c.get(self.label_key) == 'Entailment' for c in tool_kg)
                non_entailments = [c for c in tool_kg if c.get(self.label_key) != 'Entailment']
                pred_rate = len(non_entailments) / len(tool_kg)

            evaluated_count += 1

            # --- Ground Truth (Wird in beiden Modes berechnet, da wir es vergleichen müssen) ---
            gt_is_clean = all(t.get('human_label') == 'Entailment' for t in item[GT_KEY])
            gt_rate = len([c for c in item[GT_KEY] if c.get('human_label') != 'Entailment']) / len(item[GT_KEY])
            
            # --- Append ---
            gt_factual_list.append(gt_is_clean)
            pred_factual_list.append(pred_is_clean)
            gt_halu_rates.append(gt_rate)
            pred_halu_rates.append(pred_rate)

        if evaluated_count == 0:
            print("   [WARN] No valid samples found after filtering.")
            return

        self._print_metrics(
            gt_factual_list, pred_factual_list, 
            gt_halu_rates, pred_halu_rates, 
            total_items, missing_gt_count, missing_pred_key_count, 
            abstention_count, evaluated_count, ignore_abstentions
        )

    def _print_metrics(self, gt_fact, pred_fact, gt_rate, pred_rate, 
                       total, missing_gt, missing_pred_key, abstains, evaluated, ignore_abstentions):
        
        acc = accuracy_score(gt_fact, pred_fact) * 100
        f1_fact = f1_score(gt_fact, pred_fact, pos_label=True) * 100
        f1_nonfact = f1_score(gt_fact, pred_fact, pos_label=False) * 100
        
        gt_is_hallu = [not x for x in gt_fact]
        pred_is_hallu = [not x for x in pred_fact]
        tn, fp, fn, tp = confusion_matrix(gt_is_hallu, pred_is_hallu, labels=[False, True]).ravel()
        
        try:
            pearson = stats.pearsonr(gt_rate, pred_rate).statistic * 100
            spearman = stats.spearmanr(gt_rate, pred_rate).statistic * 100
        except Exception:
            pearson = 0.0
            spearman = 0.0

        base_for_eval = total - missing_gt - missing_pred_key
        abstain_rate = (abstains / base_for_eval) * 100 if base_for_eval > 0 else 0

        # --- OUTPUT ---
        print("\n📈 DATA FLOW:")
        print(f"   • Total Items in JSON:       {total}")
        print(f"   • Skipped (No Ground Truth): -{missing_gt}")
        print(f"   • Skipped (Missing Tool Key):-{missing_pred_key}")
        print(f"   ------------------------------------------------")
        print(f"   • Base for Evaluation:       {base_for_eval}")
        
        if ignore_abstentions:
            print(f"   • Abstentions Skipped:       -{abstains} ({abstain_rate:.1f}% Abstention Rate)")
            print(f"   • Actively Evaluated:        {evaluated}")
        else:
            print(f"   • Abstentions Treated as 0%: {abstains} ({abstain_rate:.1f}% Abstention Rate)")
            print(f"   • Actively Evaluated:        {evaluated}")

        print("\n🎯 CLASSIFICATION METRICS:")
        print(f"   • Accuracy:                  {acc:.2f}%")
        print(f"   • F1 (Factual):              {f1_fact:.2f}")
        print(f"   • F1 (Hallucination):        {f1_nonfact:.2f}  <-- {'Honest Rate' if ignore_abstentions else 'Distorted by Abstentions'}")
        
        print("\n🔬 HALLUCINATION BREAKDOWN (Of the evaluated ones):")
        print(f"   • True Positives (Tool caught real hallu):   {tp}")
        print(f"   • False Positives (Tool cried wolf):         {fp}")
        print(f"   • False Negatives (Tool missed real hallu):  {fn} {'<-- Massively inflated in Legacy Mode!' if not ignore_abstentions else ''}")
        print(f"   • True Negatives (Tool confirmed facts):     {tn}")
        
        print("\n📉 CORRELATION METRICS:")
        print(f"   • Pearson:                   {pearson:.2f}")
        print(f"   • Spearman:                  {spearman:.2f}\n")


if __name__ == "__main__":
    EXTRACTOR = "openai/gpt-oss-120b"
    CHECKER = "openai/gpt-oss-120b"
    
    evaluator = MetaEvaluator(EXTRACTOR, CHECKER)
    files = glob.glob(os.path.join(RESULTS_DIR, "checked_*.json"))
    
    if not files:
        print(f"No files found in {RESULTS_DIR}")
    else:
        for f in files:
            # 1. Wir testen den ehrlichen Modus (Kein "Silence is truth")
            evaluator.evaluate_file(f, ignore_abstentions=False)
            
            # 2. Wir testen den Legacy Modus (RefChecker Style)
            # evaluator.evaluate_file(f, ignore_abstentions=False)