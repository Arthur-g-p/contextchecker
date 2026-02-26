# utils.py
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

def preflight_check_refchecker_input_file():
    # find optimal handling and informing user that refrences or whatever are sometimes empty. reject empty questions.
    pass

def preflight_check_ragchecker_input_file():
    pass

def preflight_check_evaluation_input_file():
    pass

# ----------------------------------------------------------------------------------
#  CUSTOM METRICS (Drop-in replacements to avoid heavy scipy/scikit-learn dependencies)
# ----------------------------------------------------------------------------------
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