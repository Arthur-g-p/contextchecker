import os
import json
import gc
from sys import exit
from typing import List
from tqdm import tqdm

# --- CONFIG ---
GT_KEY = "claude2_response_kg"
ENTAILMENT_THRESHOLD = 0.5  # NLI score above this = match


def _require_nli():
    """Lazy import guard — fails fast with helpful message."""
    try:
        import torch  # noqa: F401
        from transformers import pipeline  # noqa: F401
    except ImportError:
        print("\n" + "=" * 60)
        print("⛔ MISSING DEPENDENCY: torch / transformers")
        print("=" * 60)
        print("Extractor evaluation requires a local NLI model.")
        print("Install with:\n")
        print("   pip install contextchecker[eval]")
        print()
        exit("Cannot run extractor eval without NLI dependencies.")


def _detect_device() -> int:
    """Detect best available device. Returns device index for transformers pipeline."""
    import torch

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_mem / (1024 ** 3)
        print(f"   🟢 GPU detected: {gpu_name} ({vram:.1f} GB VRAM)")
        print(f"   → Running NLI inference on GPU (device=0)")
        return 0
    else:
        print(f"   🟡 No GPU detected — running on CPU")
        print(f"   → This will be slower but works fine.")
        return -1


class ExtractorEvaluator:
    """
    Triplet-level extraction evaluation using NLI (zero-shot classification).
    
    Compares predicted triplets against GT triplets by converting both 
    to natural language and using an NLI model to check semantic equivalence.
    
    This isolates extractor quality from checker quality.
    
    Requires: pip install contextchecker[eval]
    """

    def __init__(self, extractor_model: str, nli_model: str = "facebook/bart-large-mnli",
                 threshold: float = ENTAILMENT_THRESHOLD):
        _require_nli()
        import torch
        from transformers import pipeline

        self.extractor_model = extractor_model
        self.nli_model_name = nli_model
        self.threshold = threshold

        # Key for predicted triplets: "{model}_response_kg"
        self.pred_key = f"{extractor_model}_response_kg"

        print(f"\n🧠 NLI MODEL SETUP:")
        print(f"   Model: {nli_model}")
        device = _detect_device()

        print(f"   Loading model...")
        self.nli = pipeline(
            "zero-shot-classification",
            model=nli_model,
            device=device
        )
        print(f"   ✅ Model loaded and ready.")
        self._device = device
        self._torch = torch

    @staticmethod
    def _triplet_to_str(triplet: list) -> str:
        """Convert [subject, predicate, object] → natural language string."""
        return f"{triplet[0]} {triplet[1]} {triplet[2]}"

    def _check_equivalence(self, gt_str: str, pred_str: str) -> tuple[bool, float]:
        """
        Use NLI to check if predicted triplet is semantically equivalent to GT.
        
        Uses zero-shot classification: premise=gt_str, hypothesis=pred_str
        Returns (is_match, score)
        """
        template = "This implies that {}"
        result = self.nli(gt_str, candidate_labels=[pred_str], hypothesis_template=template, multi_label=True)
        score = result['scores'][0]
        return score >= self.threshold, score

    def _match_triplets(self, gt_strs: List[str], pred_strs: List[str]) -> dict:
        """
        Greedy 1:1 matching: for each predicted triplet, find best GT match via NLI.
        
        Returns dict with tp, fp, fn counts and match details.
        """
        gt_matched = [False] * len(gt_strs)

        tp = 0
        fp = 0

        for pred_str in pred_strs:
            best_score = 0.0
            best_j = -1

            for j, gt_str in enumerate(gt_strs):
                if gt_matched[j]:
                    continue
                is_match, score = self._check_equivalence(gt_str, pred_str)
                if score > best_score:
                    best_score = score
                    best_j = j

            if best_j >= 0 and best_score >= self.threshold:
                gt_matched[best_j] = True
                tp += 1
            else:
                fp += 1

        fn = sum(1 for m in gt_matched if not m)

        return {"tp": tp, "fp": fp, "fn": fn}

    def evaluate_file(self, filepath: str):
        """Evaluate extractor quality by comparing predicted vs GT triplets using NLI."""
        filename = os.path.basename(filepath)

        print(f"\n{'='*60}")
        print(f"🔬 EXTRACTOR EVAL: {filename}")
        print(f"   Extractor Model: {self.extractor_model}")
        print(f"   NLI Model:       {self.nli_model_name}")
        print(f"   Pred Key:        {self.pred_key}")
        print(f"   Threshold:       {self.threshold}")
        print(f"{'='*60}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            exit(f"Cannot load {filepath}: {e}")

        total_items = len(data)
        missing_gt = 0
        missing_pred = 0
        total_tp = total_fp = total_fn = 0
        gt_counts = []
        pred_counts = []
        items_evaluated = 0

        # Pre-filter valid items for progress bar
        valid_items = []
        for item in data:
            if GT_KEY not in item or not item[GT_KEY]:
                missing_gt += 1
                continue
            if self.pred_key not in item or not item[self.pred_key]:
                missing_pred += 1
                continue
            valid_items.append(item)

        if not valid_items:
            print(f"\n❌ No valid items found.")
            print(f"   Missing GT ({GT_KEY}): {missing_gt}")
            print(f"   Missing Pred ({self.pred_key}): {missing_pred}")
            exit("Nothing to evaluate. Check your data and model name.")

        print(f"\n   📋 {len(valid_items)} items to evaluate ({missing_gt} no GT, {missing_pred} no predictions)\n")

        for item in tqdm(valid_items, desc="NLI matching", unit="item"):
            # Convert triplets to strings
            gt_strs = []
            for t in item[GT_KEY]:
                triplet = t.get("triplet", [])
                if len(triplet) >= 3:
                    gt_strs.append(self._triplet_to_str(triplet))

            pred_strs = []
            for t in item[self.pred_key]:
                triplet = t.get("triplet", [])
                if len(triplet) >= 3:
                    pred_strs.append(self._triplet_to_str(triplet))

            if not gt_strs:
                missing_gt += 1
                continue

            gt_counts.append(len(gt_strs))
            pred_counts.append(len(pred_strs))

            # NLI matching
            result = self._match_triplets(gt_strs, pred_strs)
            total_tp += result["tp"]
            total_fp += result["fp"]
            total_fn += result["fn"]
            items_evaluated += 1

        # --- METRICS ---
        self._print_metrics(
            total_items, missing_gt, missing_pred, items_evaluated,
            total_tp, total_fp, total_fn, gt_counts, pred_counts
        )

        # --- CLEANUP ---
        self._cleanup()

    def _cleanup(self):
        """Free NLI model memory."""
        print("   🧹 Cleaning up NLI model from memory...")
        del self.nli
        gc.collect()
        if self._torch.cuda.is_available():
            self._torch.cuda.empty_cache()
        print("   ✅ Memory freed.")

    def _print_metrics(self, total_items, missing_gt, missing_pred,
                       items_evaluated, tp, fp, fn, gt_counts, pred_counts):

        print(f"\n📈 DATA FLOW:")
        print(f"   • Total Items:               {total_items}")
        print(f"   • Skipped (No GT):           -{missing_gt}")
        print(f"   • Skipped (No Predictions):  -{missing_pred}")
        print(f"   ------------------------------------------------")
        print(f"   • Items Evaluated:            {items_evaluated}")
        print(f"   • Total GT Triplets:          {sum(gt_counts)}")
        print(f"   • Total Pred Triplets:        {sum(pred_counts)}")

        total_nli_calls = sum(g * p for g, p in zip(gt_counts, pred_counts))
        print(f"   • Total NLI Comparisons:      {total_nli_calls}")

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        print(f"\n🎯 EXTRACTION QUALITY (NLI-based matching, threshold={self.threshold}):")
        print(f"   • True Positives:    {tp}")
        print(f"   • False Positives:   {fp}  (predicted but no GT match)")
        print(f"   • False Negatives:   {fn}  (GT missed by extractor)")
        print(f"   ------------------------------------------------")
        print(f"   • Precision:         {precision:.3f}")
        print(f"   • Recall:            {recall:.3f}")
        print(f"   • F1:                {f1:.3f}")

        if gt_counts:
            avg_gt = sum(gt_counts) / len(gt_counts)
            avg_pred = sum(pred_counts) / len(pred_counts)
            print(f"\n📊 TRIPLET COUNT DISTRIBUTION:")
            print(f"   • Avg GT triplets/item:      {avg_gt:.1f}")
            print(f"   • Avg Pred triplets/item:    {avg_pred:.1f}")
            print(f"   • Delta (Pred - GT):         {avg_pred - avg_gt:+.1f}")
        print()
