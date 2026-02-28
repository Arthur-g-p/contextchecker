# utils.py
import json
import os
from sys import exit


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


def match_triplets_to_references_for_batch_checker(triplets, references, context_string, data):
    tasks_triplets = []
    tasks_references = []

    for i, result in enumerate(triplets):
        if result and result.triplets:
            claims_clean = [str(t) for t in result.triplets]
            tasks_triplets.append(claims_clean)
            tasks_references.append("\n".join(data[i][context_string]))
        else:
            tasks_triplets.append([])
            tasks_references.append("\n".join(data[i][context_string]))
    
    return tasks_triplets, tasks_references
    
# ----------------------------------------------------------------------------------
#  INPUT FORMAT NORMALIZATION
# ----------------------------------------------------------------------------------

def _get_reference(item: dict) -> str | None:
    """
    Auto-detect and normalize reference from any supported input format.
    
    Supports 4 combos:
        "reference": "single string"     -> "single string"
        "context":   ["p1", "p2"]        -> "p1\\np2"
        "reference": ["p1", "p2"]        -> "p1\\np2"
        "context":   "single string"     -> "single string"
    
    Priority: context > reference (context is the richer format).
    Returns None if neither key found or value is empty.
    """
    for key in ("context", "reference"):
        val = item.get(key)
        if val is None:
            continue
        if isinstance(val, list):
            joined = "\n".join(str(v) for v in val if str(v).strip())
            return joined if joined else None
        if isinstance(val, str) and val.strip():
            return val
    return None


# ----------------------------------------------------------------------------------
#  SMART DATA LOADING
# ----------------------------------------------------------------------------------

def load_and_validate_json(filepath: str, context: str, verbose: bool = False) -> list[dict]:
    """
    Load a JSON file, auto-detect format, skip bad items, return clean items.
    
    Handles both flat arrays [...] and wrapped {"_meta": {...}, "data": [...]}.
    
    Skip rules:
        - Missing reference AND context  -> HARD SKIP
        - Missing question               -> HARD SKIP
        - Empty response                 -> KEEP (extractor handles -> abstention)
    
    verbose=False: prints summary counts
    verbose=True:  also prints per-item details (index + question)
    
    Returns the cleaned list of items (skipped items removed).
    Exits on fatal errors (file not found, corrupt JSON, all items skipped).
    """
    if not os.path.exists(filepath):
        exit(f"File not found: {filepath}")

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = json.load(f)
    except Exception as e:
        exit(f"Cannot load {filepath}: {e}")

    # Handle wrapped format: {"_meta": {...}, "data": [...]}
    if isinstance(raw, dict) and "data" in raw:
        data = raw["data"]
    elif isinstance(raw, list):
        data = raw
    else:
        exit(f"{filepath}: Expected a JSON array or dict with 'data' key.")

    if not isinstance(data, list) or len(data) == 0:
        exit(f"{filepath}: Expected a non-empty JSON array.")

    # --- Scan and skip bad items ---
    valid_items = []
    skip_no_ref = []
    skip_no_question = []
    warn_empty_response = []

    for idx, item in enumerate(data):
        q = item.get("question", "").strip() if isinstance(item.get("question"), str) else ""
        ref = _get_reference(item)
        resp = item.get("response", "")

        # HARD SKIP: no reference
        if ref is None:
            skip_no_ref.append((idx, q or "(no question)"))
            continue

        # HARD SKIP: no question
        if not q:
            skip_no_question.append((idx, "(missing question)"))
            continue

        # WARN: empty response (keep - flows through as abstention)
        if not isinstance(resp, str) or not resp.strip():
            warn_empty_response.append((idx, q))

        valid_items.append(item)

    # --- Print report ---
    total = len(data)
    fname = os.path.basename(filepath)

    print(f"\nData loaded: {fname} ({context})")
    print(f"   Total: {total}")

    if skip_no_ref:
        print(f"   SKIPPED (no reference/context): {len(skip_no_ref)}")
        if verbose:
            for idx, q in skip_no_ref:
                print(f"      [{idx}] {q}")

    if skip_no_question:
        print(f"   SKIPPED (no question): {len(skip_no_question)}")
        if verbose:
            for idx, q in skip_no_question:
                print(f"      [{idx}]")

    if warn_empty_response:
        print(f"   Empty response (-> abstention): {len(warn_empty_response)}")
        if verbose:
            for idx, q in warn_empty_response:
                print(f"      [{idx}] {q}")

    if not valid_items:
        exit(f"All {total} items skipped. No processable data in {filepath}. "
             f"If your data is corrupted, reload from git.")

    print(f"   Ready to process: {len(valid_items)}")

    return valid_items


# ----------------------------------------------------------------------------------
#  PREFLIGHT CHECKS -- format-specific validation
# ----------------------------------------------------------------------------------

def preflight_check_msmarco_input_file(filepath: str, verbose: bool = False) -> list[dict]:
    """
    Validate an msmarco data file has the expected structure.
    Required: response, context, claude2_response_kg (GT triplets with human_label)
    """
    data = load_and_validate_json(filepath, "msmarco input", verbose=verbose)

    # Spot-check GT triplets structure (on first item that has them)
    for item in data:
        sample_kg = item.get("claude2_response_kg", [])
        if sample_kg and isinstance(sample_kg, list) and len(sample_kg) > 0:
            first_triplet = sample_kg[0]
            if "triplet" not in first_triplet:
                print(f"\nGT triplets missing 'triplet' key in claude2_response_kg")
                print(f"   Got: {list(first_triplet.keys())}")
                exit("Invalid msmarco data structure. Reload from git.")
            if "human_label" not in first_triplet:
                print(f"\nWarning: GT triplets missing 'human_label' -- evaluation will be limited")
            break

    return data


def preflight_check_evaluation_input_file(filepath: str, extractor_model: str, checker_model: str) -> list[dict]:
    """Validate a pipeline results file for meta evaluation."""
    data = load_and_validate_json(filepath, "meta evaluation")
    ext_key = f"{extractor_model}_response_kg"
    sample = data[0]
    if ext_key not in sample:
        print(f"\nMissing key '{ext_key}' -- was the pipeline run with this extractor model?")
        print(f"   Available keys: {list(sample.keys())}")
        exit("Data validation failed.")
    return data


# ----------------------------------------------------------------------------------
#  OUTPUT METADATA -- crash-safe _meta builder
# ----------------------------------------------------------------------------------

def build_meta(*, extractor_model: str = None, checker_model: str = None,
               source_file: str = None, timestamp_start: str = None,
               timestamp_end: str = None, duration_seconds: float = None,
               status: str = "complete", total: int = None,
               skipped: dict = None, abstentions: int = None,
               token_stats: dict = None) -> dict:
    """
    Build crash-safe _meta dict for output JSON. Never raises.
    Any field that fails gets set to None rather than crashing the save.
    """
    try:
        from importlib.metadata import version as pkg_version
        ver = pkg_version("contextchecker")
    except Exception:
        ver = None

    meta = {}
    try:
        meta["version"] = ver
        meta["extractor_model"] = extractor_model
        meta["checker_model"] = checker_model
        meta["source_file"] = source_file
        meta["timestamp_start"] = timestamp_start
        meta["timestamp_end"] = timestamp_end
        meta["duration_seconds"] = round(duration_seconds, 2) if duration_seconds else None
        meta["status"] = status
        meta["total"] = total
        meta["skipped"] = skipped or {}
        meta["abstentions"] = abstentions
        meta["token_stats"] = token_stats or {}
    except Exception:
        meta["_error"] = "Failed to build complete _meta"

    return meta











#  CUSTOM METRICS (Drop-in replacements to avoid heavy scipy/scikit-learn dependencies)
import math
from collections import Counter

def accuracy_score(y_true, y_pred):
    if not y_true:
        return 0.0
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)

def confusion_matrix(y_true, y_pred, labels=None):
    if labels is None:
        labels = list(sorted(set(y_true) | set(y_pred)))
    
    label_to_idx = {label: i for i, label in enumerate(labels)}
    n_labels = len(labels)
    
    matrix = [[0] * n_labels for _ in range(n_labels)]
    
    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            matrix[label_to_idx[t]][label_to_idx[p]] += 1
            
    class MatrixHelper(list):
        def ravel(self):
            return [item for row in self for item in row]
            
    return MatrixHelper(matrix)

def f1_score(y_true, y_pred, pos_label=True):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p == pos_label)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t != pos_label and p == pos_label)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == pos_label and p != pos_label)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)

class _StatsResult:
    def __init__(self, stat):
        self.statistic = stat

def pearsonr(x, y):
    n = len(x)
    if n == 0:
        return _StatsResult(0.0)
        
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_sq = sum(xi * xi for xi in x)
    sum_y_sq = sum(yi * yi for yi in y)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    
    numerator = n * sum_xy - sum_x * sum_y
    denom_x = n * sum_x_sq - sum_x * sum_x
    denom_y = n * sum_y_sq - sum_y * sum_y
    
    if denom_x <= 0 or denom_y <= 0:
        return _StatsResult(0.0)
        
    return _StatsResult(numerator / math.sqrt(denom_x * denom_y))

def _rank_data(a):
    n = len(a)
    ivec = list(enumerate(a))
    ivec.sort(key=lambda x: x[1])
    svec = [x[1] for x in ivec]
    sumranks = 0
    dupcount = 0
    newarray = [0.0] * n
    for i in range(n):
        sumranks += i
        dupcount += 1
        if i == n - 1 or svec[i] != svec[i + 1]:
            averank = sumranks / dupcount + 1
            for j in range(i - dupcount + 1, i + 1):
                newarray[ivec[j][0]] = averank
            sumranks = 0
            dupcount = 0
    return newarray

def spearmanr(x, y):
    rank_x = _rank_data(x)
    rank_y = _rank_data(y)
    return pearsonr(rank_x, rank_y)

def classification_report(y_true, y_pred, labels=None, zero_division=0, digits=2):
    if labels is None:
        labels = list(sorted(set(y_true) | set(y_pred)))
        
    report = []
    max_label_len = max([len(str(l)) for l in labels] + [12])
    header = f"{'':<{max_label_len}} {'precision':>10} {'recall':>10} {'f1-score':>10} {'support':>10}"
    report.append(header)
    report.append("")
    
    total_support = 0
    macro_p = macro_r = macro_f1 = 0
    weighted_p = weighted_r = weighted_f1 = 0
    
    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else zero_division
        recall = tp / (tp + fn) if (tp + fn) > 0 else zero_division
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else zero_division
        
        total_support += support
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f1 += f1 * support
        
        row = f"{str(label):<{max_label_len}} {precision:>10.{digits}f} {recall:>10.{digits}f} {f1:>10.{digits}f} {support:>10}"
        report.append(row)
        
    report.append("")
    
    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    accuracy = correct / len(y_true) if y_true else 0
    report.append(f"{'accuracy':<{max_label_len}} {'':>10} {'':>10} {accuracy:>10.{digits}f} {total_support:>10}")
    
    n_labels = len(labels)
    macro_p /= n_labels
    macro_r /= n_labels
    macro_f1 /= n_labels
    report.append(f"{'macro avg':<{max_label_len}} {macro_p:>10.{digits}f} {macro_r:>10.{digits}f} {macro_f1:>10.{digits}f} {total_support:>10}")
    
    if total_support > 0:
        weighted_p /= total_support
        weighted_r /= total_support
        weighted_f1 /= total_support
    report.append(f"{'weighted avg':<{max_label_len}} {weighted_p:>10.{digits}f} {weighted_r:>10.{digits}f} {weighted_f1:>10.{digits}f} {total_support:>10}")
        
    return "\n".join(report)

class stats:
    pearsonr = staticmethod(pearsonr)
    spearmanr = staticmethod(spearmanr)