import os
import json
import asyncio
from typing import List, Tuple
from contextchecker.utils import accuracy_score, classification_report, confusion_matrix
from collections import Counter

from contextchecker.checker import Checker

# --- CONFIG ---
DATA_DIR = "results"
GT_KEY = "claude2_response_kg"
LABELS = ["Entailment", "Contradiction", "Neutral"]


class CheckerEvaluator:
    """
    Triplet-level NLI evaluation.
    
    Runs the checker on GT triplets (claude2_response_kg) and compares
    the checker's predicted verdicts 1:1 against human_label.
    
    This isolates checker quality from extractor quality.
    """

    def __init__(self, checker: Checker):
        self.checker = checker

    def _prepare_gt_data(self, data: list) -> Tuple[List[List[str]], List[str], List[str], list]:
        """
        Extracts GT triplets and references from the data.
        
        Returns:
            claims_batch: List[List[str]] — GT triplets as strings per item
            references_batch: List[str] — joined context per item
            gt_labels_flat: List[str] — flat list of human_labels (aligned with checker output)
            skipped_info: dict with skip counts
        """
        claims_batch = []
        references_batch = []
        gt_labels_per_doc = []  # List[List[str]] — mirrors claims_batch structure

        missing_gt = 0
        missing_context = 0
        empty_gt = 0

        for item in data:
            # 1. Need GT triplets
            if GT_KEY not in item or not item[GT_KEY]:
                missing_gt += 1
                continue

            # 2. Need context/reference
            context = item.get("context", [])
            if not context:
                missing_context += 1
                continue

            gt_triplets = item[GT_KEY]

            # 3. Convert GT triplets to claim strings (same format as main.py)
            claims = []
            labels = []
            for t in gt_triplets:
                triplet = t.get("triplet", [])
                label = t.get("human_label")
                
                if not triplet or not label:
                    continue
                
                # Format as string: "(subject, predicate, object)"
                claim_str = f"({triplet[0]}, {triplet[1]}, {triplet[2]})"
                claims.append(claim_str)
                labels.append(label)

            if not claims:
                empty_gt += 1
                continue

            claims_batch.append(claims)
            references_batch.append("\n".join(context))
            gt_labels_per_doc.append(labels)

        skip_info = {
            "missing_gt": missing_gt,
            "missing_context": missing_context,
            "empty_gt": empty_gt
        }

        return claims_batch, references_batch, gt_labels_per_doc, skip_info

    async def evaluate_file(self, filepath: str):
        """
        Runs the checker on GT triplets and evaluates against human_label.
        """
        filename = os.path.basename(filepath)

        print(f"\n{'='*60}")
        print(f"🔬 CHECKER EVAL: {filename}")
        print(f"{'='*60}")

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error loading file: {e}")
            return

        total_items = len(data)

        # --- PHASE 1: PREPARE GT DATA ---
        print(f"   📋 Preparing GT data from {total_items} items...")
        claims_batch, references_batch, gt_labels_per_doc, skip_info = self._prepare_gt_data(data)

        items_to_check = len(claims_batch)
        total_claims = sum(len(c) for c in claims_batch)

        print(f"\n📈 DATA FLOW:")
        print(f"   • Total Items in JSON:       {total_items}")
        print(f"   • Skipped (No GT):           -{skip_info['missing_gt']}")
        print(f"   • Skipped (No Context):      -{skip_info['missing_context']}")
        print(f"   • Skipped (Empty GT):        -{skip_info['empty_gt']}")
        print(f"   ------------------------------------------------")
        print(f"   • Items to Check:            {items_to_check}")
        print(f"   • Total GT Triplets:         {total_claims}")

        if items_to_check == 0:
            print("   [WARN] Nothing to evaluate.")
            return

        # --- PHASE 2: RUN CHECKER ON GT TRIPLETS ---
        print(f"\n   🚀 Running checker on {total_claims} GT triplets...")
        verdicts_batch = await self.checker.check_batch(claims_batch, references_batch)

        # --- PHASE 3: COMPARE 1:1 ---
        gt_labels_flat = []
        pred_labels_flat = []
        parse_errors = 0

        for doc_gt_labels, doc_verdicts in zip(gt_labels_per_doc, verdicts_batch):
            for gt_label, verdict in zip(doc_gt_labels, doc_verdicts):
                gt_labels_flat.append(gt_label)
                pred_labels_flat.append(verdict.label)
                if verdict.claim == "Error":
                    parse_errors += 1

        # --- PHASE 4: METRICS ---
        self._print_metrics(gt_labels_flat, pred_labels_flat, parse_errors)

    def _print_metrics(self, gt_labels, pred_labels, parse_errors):
        
        n = len(gt_labels)
        
        # --- LABEL DISTRIBUTION ---
        gt_dist = Counter(gt_labels)
        pred_dist = Counter(pred_labels)

        print(f"\n📊 LABEL DISTRIBUTION (n={n}):")
        print(f"   {'Label':<16} {'GT':>8} {'Pred':>8} {'Delta':>8}")
        print(f"   {'-'*44}")
        for label in LABELS:
            g = gt_dist.get(label, 0)
            p = pred_dist.get(label, 0)
            delta = p - g
            sign = "+" if delta > 0 else ""
            print(f"   {label:<16} {g:>8} {p:>8} {sign}{delta:>7}")
        
        if parse_errors:
            print(f"\n   ⚠️  Parse Errors (defaulted to Neutral): {parse_errors}")

        # --- CLASSIFICATION ---
        acc = accuracy_score(gt_labels, pred_labels) * 100
        print(f"\n🎯 TRIPLET-LEVEL CLASSIFICATION (1:1 aligned):")
        print(f"   • Accuracy: {acc:.2f}%")
        print(f"   • Total Triplets Evaluated: {n}")

        report = classification_report(
            gt_labels, pred_labels,
            labels=LABELS,
            zero_division=0,
            digits=3
        )
        print(f"\n{report}")

        # --- CONFUSION MATRIX ---
        cm = confusion_matrix(gt_labels, pred_labels, labels=LABELS)
        print("   CONFUSION MATRIX (rows=GT, cols=Pred):")
        header = "   ".join(f"{'Entail':>10}" if l == "Entailment" else f"{l[:6]:>10}" for l in LABELS)
        print(f"   {'':>16} {header}")
        for i, label in enumerate(LABELS):
            short = "Entail" if label == "Entailment" else label[:6]
            row = "   ".join(f"{v:>10}" for v in cm[i])
            print(f"   {short:<16} {row}")
        print()


if __name__ == "__main__":
    # --- CONFIGURE ---
    MODEL = "openai/gpt-oss-120b"
    BASE_API = "http://localhost:4000/v1"
    INPUT_FILE = os.path.join(DATA_DIR, "checked_msmarco_gpt4_answers_full.json")

    checker = Checker(model=MODEL, baseapi=BASE_API)
    evaluator = CheckerEvaluator(checker)

    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    asyncio.run(evaluator.evaluate_file(INPUT_FILE))
