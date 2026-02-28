# Evaluation Guide

ContextChecker evaluates its pipeline at **three levels**, each isolating a different component.

## Overview

| Command | What it evaluates | Input needed | API? | Extra deps? |
|---------|-------------------|-------------|------|-------------|
| `eval extractor` | Are predicted triplets semantically the same as GT? | GT data with both `claude2_response_kg` and `{model}_response_kg` | ❌ | ✅ `pip install contextchecker[eval]` |
| `eval checker` | Does the checker label GT triplets correctly? | GT data with `claude2_response_kg` (has `human_label`) + `context` | ✅ costs tokens | ❌ |
| `eval meta --data` | End-to-end pipeline accuracy at item level | Pipeline output (from `run` or `results/`) | ❌ | ❌ |
| `eval meta --source` | Full pipeline run + evaluation on raw msmarco data | Raw data from `data/noisy_context/` | ✅ costs tokens | ❌ |
| `eval full` | All three in sequence | Same as above (auto-skips unavailable components) | ✅ | optional |

## Data Requirements

### GT Data Format

The ground-truth (GT) data files contain items with human-annotated triplets:

```json
{
  "query": "How long does NEXUS processing take?",
  "response": "It takes approximately 10-14 weeks.",
  "context": ["Reference passage 1", "Reference passage 2"],
  "claude2_response_kg": [
    {
      "triplet": ["NEXUS application", "processing time", "10-14 weeks"],
      "human_label": "Entailment"
    }
  ]
}
```

### What Each Eval Needs

**Extractor Eval** — compares predicted vs GT triplets using a local NLI model:
- Requires: `claude2_response_kg` (GT) AND `{extractor_model}_response_kg` (predicted)
- The extractor must have already been run on this data
- No API calls needed, but requires `pip install contextchecker[eval]` for NLI model

**Checker Eval** — runs the checker fresh on GT triplets:  
- Requires: `claude2_response_kg` with `human_label` AND `context`
- Makes live API calls to check each GT triplet → **costs tokens**
- Does NOT need prior pipeline results

**Meta Eval** — evaluates the full pipeline output (two modes):
- `--data`: Uses existing pipeline output with `{extractor_model}_response_kg` + `{checker_model}_label`. Fully offline.
- `--source`: Runs the full pipeline (extract + check) on raw data first, then evaluates. **Costs tokens.**
- Output naming (with `--source`): `meta_msmarco_{response_model}_ext-{extractor}_chk-{checker}.json`

## Typical Workflows

### 1. Evaluate a new checker model
```bash
# Just checker — uses GT triplets, no extractor needed
python cli.py eval checker \
  --data data/gt_annotated.json \
  --checker-model openai/gpt-4o \
  --checker-base-api http://localhost:4000/v1
```

### 2. Run pipeline + evaluate on raw msmarco data
```bash
# Runs extraction + checking, saves results, then evaluates
python cli.py eval meta \
  --source data/noisy_context/msmarco_gpt4_answers.json \
  --extractor-model openai/gpt-oss-120b \
  --checker-model openai/gpt-oss-120b \
  --extractor-base-api http://localhost:4000/v1 \
  --checker-base-api http://localhost:4000/v1
# Output: results/meta_msmarco_gpt4_ext-gpt-oss-120b_chk-gpt-oss-120b.json
```

### 3. Evaluate existing pipeline results (offline)
```bash
python cli.py eval meta \
  --data results/checked_msmarco_gpt4_answers_full.json \
  --extractor-model openai/gpt-oss-120b \
  --checker-model openai/gpt-oss-120b
```

### 4. Full model check (all components)
```bash
python cli.py eval full \
  --data results/checked_msmarco_gpt4_answers_full.json \
  --extractor-model openai/gpt-oss-120b \
  --checker-model openai/gpt-oss-120b \
  --extractor-base-api http://localhost:4000/v1 \
  --checker-base-api http://localhost:4000/v1
```

### 5. Evaluate extraction quality only
```bash
# Requires: pip install contextchecker[eval]
python cli.py eval extractor \
  --data results/checked_msmarco_gpt4_answers_full.json \
  --extractor-model openai/gpt-oss-120b
```

## Installing Optional Dependencies

Extractor evaluation uses a local NLI model (default: `facebook/bart-large-mnli`).
These are heavy (~1.6GB) and not installed by default. Works on GPU (even older cards like GTX 1660 Ti) and CPU:

```bash
pip install contextchecker[eval]
```

This installs `torch`, `transformers`, and `sentencepiece`.

## Metrics Reference

### Extractor Eval
- **Precision**: % of predicted triplets that match a GT triplet (via NLI)
- **Recall**: % of GT triplets captured by the extractor
- **F1**: Harmonic mean of precision and recall

### Checker Eval
- **Accuracy**: % of GT triplets correctly labeled
- **Confusion Matrix**: GT labels (rows) vs predicted labels (columns)
- **Per-label F1**: Entailment / Contradiction / Neutral

### Meta Eval
- **Accuracy**: Item-level factuality classification
- **F1 (Factual/Hallucination)**: Per-class F1 scores
- **Pearson/Spearman**: Correlation between GT and predicted hallucination rates
- **Honest vs Legacy mode**: `--honest` skips abstentions; `--no-honest` treats them as 0% hallucination
