import typer
import asyncio
import sys
import glob
import os
import json
import time
from datetime import datetime, timezone

from contextchecker.utils import load_and_validate_json, _get_reference, build_meta

app = typer.Typer(help="ContextChecker — RAG Hallucination Checking Pipeline. See EVALUATION.md for full documentation.")
eval_app = typer.Typer(help="Evaluation subcommands. See EVALUATION.md for data requirements and workflows.")
app.add_typer(eval_app, name="eval")

# --- Shared defaults ---
DEFAULT_MODEL = None
DEFAULT_API = None


def _async_run(coro):
    """Run async code with Windows support. Returns the coroutine result."""
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    return asyncio.run(coro)


def _resolve_data_files(data: str | None) -> list[str]:
    """Resolve data argument to a list of file paths."""
    if data:
        if not os.path.exists(data):
            typer.echo(f"❌ File not found: {data}")
            raise typer.Exit(1)
        return [data]
    files = glob.glob(os.path.join("results", "checked_*.json"))
    if not files:
        typer.echo("❌ No files found in results/. Use --data to specify a file.")
        raise typer.Exit(1)
    return files


# Validation is now centralized in contextchecker.utils (load_and_validate_json, preflight_check_*)


# ═══════════════════════════════════════════════════════════════
#  RUN — Full extract + check pipeline
# ═══════════════════════════════════════════════════════════════

@app.command()
def run(
    input: str = typer.Option("example/example_in_ref.json", help="Path to input JSON"),
    output: str = typer.Option("results_final.json", help="Path to output JSON"),
    extractor_model: str = typer.Option(DEFAULT_MODEL, help="Extractor model name"),
    checker_model: str = typer.Option(DEFAULT_MODEL, help="Checker model name"),
    extractor_base_api: str = typer.Option(DEFAULT_API, help="Base API URL for extractor"),
    checker_base_api: str = typer.Option(DEFAULT_API, help="Base API URL for checker"),
):
    """Full extract + check pipeline on your own data.

    Input needs 'response' + ('reference' or 'context') + 'question'.
    Output can be evaluated with 'eval meta --data'.
    """
    if not extractor_model:
        typer.echo("--extractor-model is required")
        raise typer.Exit(1)
    if not checker_model:
        typer.echo("--checker-model is required")
        raise typer.Exit(1)

    # --- Load + auto-detect format ---
    data = load_and_validate_json(input, "run")
    responses = [item.get("response", "") for item in data]
    references = [_get_reference(item) or "" for item in data]

    typer.echo(f"\nPipeline: {len(data)} items from {input}")
    typer.echo(f"   Extractor: {extractor_model}" + (f" @ {extractor_base_api}" if extractor_base_api else ""))
    typer.echo(f"   Checker:   {checker_model}" + (f" @ {checker_base_api}" if checker_base_api else ""))

    from contextchecker.extractor import Extractor
    from contextchecker.checker import Checker
    from contextchecker.pipeline import Pipeline
    from contextchecker.stats import GLOBAL_STATS

    t_start = time.time()
    ts_start = datetime.now(timezone.utc).isoformat()

    pipe = Pipeline(
        Extractor(model=extractor_model, baseapi=extractor_base_api),
        Checker(model=checker_model, baseapi=checker_base_api)
    )
    results = _async_run(pipe.run(responses, references))

    t_end = time.time()
    ts_end = datetime.now(timezone.utc).isoformat()

    # --- Caller merges results ---
    output_items = []
    abstentions = 0
    for item, result in zip(data, results):
        claims_entry = []
        if result.extraction and result.extraction.triplets:
            for triplet, verdict in zip(result.extraction.triplets, result.verdicts):
                claims_entry.append({
                    "triplet": [triplet.subject, triplet.predicate, triplet.object],
                    "verdict": verdict.label,
                    "explanation": verdict.explanation,
                })
        else:
            abstentions += 1
        output_items.append({
            "question": item.get("question", ""),
            "reference": _get_reference(item) or "",
            "response": item.get("response", ""),
            "claims": claims_entry,
        })

    # --- Save with _meta ---
    total_claims = sum(len(r.verdicts) for r in results)
    meta = build_meta(
        extractor_model=extractor_model, checker_model=checker_model,
        source_file=os.path.basename(input),
        timestamp_start=ts_start, timestamp_end=ts_end,
        duration_seconds=t_end - t_start,
        total=len(data), abstentions=abstentions,
        token_stats=GLOBAL_STATS.snapshot(),
    )
    wrapped = {"_meta": meta, "data": output_items}

    os.makedirs(os.path.dirname(output) or ".", exist_ok=True)
    with open(output, 'w', encoding='utf-8') as f:
        json.dump(wrapped, f, indent=2, ensure_ascii=False)

    print(f"\n{GLOBAL_STATS}")
    typer.echo(f"Done: {len(data)} items, {total_claims} claims, {abstentions} abstentions -> {output}")




# ═══════════════════════════════════════════════════════════════
#  EVAL EXTRACTOR — NLI-based triplet comparison
# ═══════════════════════════════════════════════════════════════

@eval_app.command("extractor")
def eval_extractor(
    data: str = typer.Option(None, help="Path to JSON with GT + predicted triplets (or auto-detect from results/)"),
    extractor_model: str = typer.Option(DEFAULT_MODEL, help="Extractor model name (determines {model}_response_kg key)"),
    nli_model: str = typer.Option("facebook/bart-large-mnli", help="Local NLI model for semantic comparison"),
):
    """Evaluate extraction quality via NLI-based triplet matching.

    Compares predicted triplets ({model}_response_kg) against GT triplets
    (claude2_response_kg) using a local NLI model. No API calls needed,
    but requires: pip install contextchecker[eval]

    See EVALUATION.md for details.
    """
    files = _resolve_data_files(data)
    for f in files:
        load_and_validate_json(f, "eval extractor")

    from contextchecker.eval.extractoreval import ExtractorEvaluator

    evaluator = ExtractorEvaluator(extractor_model=extractor_model, nli_model=nli_model)

    for f in files:
        evaluator.evaluate_file(f)


# ═══════════════════════════════════════════════════════════════
#  EVAL CHECKER — Run checker on GT triplets
# ═══════════════════════════════════════════════════════════════

@eval_app.command("checker")
def eval_checker(
    data: str = typer.Option(..., help="Path to GT JSON with claude2_response_kg + context"),
    checker_model: str = typer.Option(DEFAULT_MODEL, help="Checker model name"),
    checker_base_api: str = typer.Option(DEFAULT_API, help="Base API URL for checker"),
):
    """Run checker on GT triplets and evaluate 1:1 against human labels.

    Requires GT data with 'claude2_response_kg' (with human_label) and 'context'.
    Makes live API calls — costs tokens.

    See EVALUATION.md for details.
    """
    load_and_validate_json(data, "eval checker")

    from contextchecker.checker import Checker
    from contextchecker.eval.checkereval import CheckerEvaluator

    checker = Checker(model=checker_model, baseapi=checker_base_api)
    evaluator = CheckerEvaluator(checker)
    _async_run(evaluator.evaluate_file(data))


# ═══════════════════════════════════════════════════════════════
#  EVAL META — End-to-end pipeline evaluation
# ═══════════════════════════════════════════════════════════════

@eval_app.command("meta")
def eval_meta(
    source: str = typer.Option(None, help="Path to RAW data (e.g. data/noisy_context/msmarco_gpt4_answers.json). Runs the full pipeline first."),
    data: str = typer.Option(None, help="Path to EXISTING pipeline output (already has {model}_response_kg + {model}_label). Skips pipeline."),
    extractor_model: str = typer.Option(DEFAULT_MODEL, help="Extractor model name"),
    checker_model: str = typer.Option(DEFAULT_MODEL, help="Checker model name"),
    extractor_base_api: str = typer.Option(DEFAULT_API, help="Base API URL for extractor (only with --source)"),
    checker_base_api: str = typer.Option(DEFAULT_API, help="Base API URL for checker (only with --source)"),
    honest: bool = typer.Option(True, help="Honest mode: skip abstentions. --no-honest = legacy/RefChecker mode"),
):
    """Meta-evaluation: end-to-end factuality comparison.

    Two modes:

    1. --source: Runs the FULL pipeline (extract + check) on raw data,
       saves results, then evaluates. Costs API tokens.

       Example: eval meta --source data/noisy_context/msmarco_gpt4_answers.json

    2. --data: Evaluates EXISTING pipeline output (offline, no API calls).
       Requires {model}_response_kg and {model}_label keys.

       Example: eval meta --data results/checked_msmarco_gpt4_answers_full.json

    See EVALUATION.md for details.
    """
    if source and data:
        typer.echo("❌ Use either --source (run pipeline) or --data (evaluate existing results), not both.")
        raise typer.Exit(1)
    
    if not source and not data:
        typer.echo("❌ Use either --source (run pipeline) or --data (evaluate existing results).")
        raise typer.Exit(1)

    from contextchecker.eval.metaeval import MetaEvaluator

    if source:
        # --- MODE 1: Run pipeline first, then evaluate ---
        if not extractor_model or not checker_model:
            typer.echo("--extractor-model and --checker-model are required with --source")
            raise typer.Exit(1)

        # Preflight validates file exists + required keys + warns about anomalies
        from contextchecker.utils import preflight_check_msmarco_input_file
        msmarco_data = preflight_check_msmarco_input_file(source)

        # --- Caller maps format via _get_reference (auto-detects context/reference) ---
        responses = [item.get("response", "") for item in msmarco_data]
        references = [_get_reference(item) or "" for item in msmarco_data]

        from contextchecker.extractor import Extractor
        from contextchecker.checker import Checker
        from contextchecker.pipeline import Pipeline
        from contextchecker.stats import GLOBAL_STATS

        t_start = time.time()
        ts_start = datetime.now(timezone.utc).isoformat()

        pipe = Pipeline(
            Extractor(model=extractor_model, baseapi=extractor_base_api),
            Checker(model=checker_model, baseapi=checker_base_api)
        )
        results = _async_run(pipe.run(responses, references))

        t_end = time.time()
        ts_end = datetime.now(timezone.utc).isoformat()

        # --- Caller merges results back into msmarco format ---
        ext_key = f"{extractor_model}_response_kg"
        label_key = f"{checker_model}_label"
        abstentions = 0
        for item, result in zip(msmarco_data, results):
            kg_entries = []
            if result.extraction and result.extraction.triplets:
                for triplet, verdict in zip(result.extraction.triplets, result.verdicts):
                    kg_entries.append({
                        "claim": [triplet.subject, triplet.predicate, triplet.object],
                        label_key: verdict.label,
                    })
            else:
                abstentions += 1
            item[ext_key] = kg_entries

        # --- Caller saves with _meta + smart naming ---
        from contextchecker.eval.pipeline import build_output_filename
        out_dir = "results"
        os.makedirs(out_dir, exist_ok=True)
        out_filename = build_output_filename(os.path.basename(source), extractor_model, checker_model)
        out_path = os.path.join(out_dir, out_filename)

        total_claims = sum(len(r.verdicts) for r in results)
        meta = build_meta(
            extractor_model=extractor_model, checker_model=checker_model,
            source_file=os.path.basename(source),
            timestamp_start=ts_start, timestamp_end=ts_end,
            duration_seconds=t_end - t_start,
            total=len(msmarco_data), abstentions=abstentions,
            token_stats=GLOBAL_STATS.snapshot(),
        )
        wrapped = {"_meta": meta, "data": msmarco_data}

        with open(out_path, 'w', encoding='utf-8') as f:
            json.dump(wrapped, f, indent=2, ensure_ascii=False)

        print(f"\n{GLOBAL_STATS}")
        typer.echo(f"Pipeline done: {len(msmarco_data)} items, {total_claims} claims, {abstentions} abstentions -> {out_path}")

        typer.echo(f"\n{'---'*20}")
        typer.echo(f"  Now running MetaEvaluator on pipeline output...")
        typer.echo(f"{'---'*20}")

        evaluator = MetaEvaluator(extractor_model, checker_model)
        evaluator.evaluate_file(out_path, ignore_abstentions=honest)

    else:
        # --- MODE 2: Evaluate existing results ---
        data_files = _resolve_data_files(data)
        evaluator = MetaEvaluator(extractor_model, checker_model)
        for f in data_files:
            evaluator.evaluate_file(f, ignore_abstentions=honest)


# ═══════════════════════════════════════════════════════════════
#  EVAL FULL — Run all available evaluations
# ═══════════════════════════════════════════════════════════════

@eval_app.command("full")
def eval_full(
    data: str = typer.Option(None, help="Path to data JSON (or auto-detect from results/)"),
    extractor_model: str = typer.Option(DEFAULT_MODEL, help="Extractor model name"),
    checker_model: str = typer.Option(DEFAULT_MODEL, help="Checker model name"),
    extractor_base_api: str = typer.Option(DEFAULT_API, help="Base API URL for extractor"),
    checker_base_api: str = typer.Option(DEFAULT_API, help="Base API URL for checker"),
    nli_model: str = typer.Option("facebook/bart-large-mnli", help="NLI model for extractor eval"),
    honest: bool = typer.Option(True, help="Honest mode for meta eval"),
):
    """Run all evaluations: extractor + checker + meta.

    Runs each component eval in sequence, skipping unavailable ones
    (e.g. extractor eval if contextchecker[eval] not installed).

    See EVALUATION.md for data requirements and workflows.
    """
    files = _resolve_data_files(data)

    print("\n" + "█" * 60)
    print("  FULL MODEL EVALUATION")
    print(f"  Extractor: {extractor_model} @ {extractor_base_api}")
    print(f"  Checker:   {checker_model} @ {checker_base_api}")
    print("█" * 60)

    # ── 1. EXTRACTOR EVAL (needs NLI model — optional dep) ───
    print(f"\n{'─'*60}")
    print(f"  PHASE 1/3: EXTRACTOR EVALUATION")
    print(f"{'─'*60}")
    try:
        from contextchecker.eval.extractoreval import ExtractorEvaluator
        evaluator = ExtractorEvaluator(extractor_model=extractor_model, nli_model=nli_model)
        for f in files:
            evaluator.evaluate_file(f)
    except SystemExit:
        print("   ⚠️  Skipping extractor eval (install contextchecker[eval] to enable)")

    # ── 2. CHECKER EVAL (needs API — costs tokens) ───────────
    print(f"\n{'─'*60}")
    print(f"  PHASE 2/3: CHECKER EVALUATION")
    print(f"{'─'*60}")
    from contextchecker.checker import Checker
    from contextchecker.eval.checkereval import CheckerEvaluator

    checker = Checker(model=checker_model, baseapi=checker_base_api)
    checker_evaluator = CheckerEvaluator(checker)
    for f in files:
        _async_run(checker_evaluator.evaluate_file(f))

    # ── 3. META EVAL (offline) ────────────────────────────────
    print(f"\n{'─'*60}")
    print(f"  PHASE 3/3: META EVALUATION (end-to-end)")
    print(f"{'─'*60}")
    from contextchecker.eval.metaeval import MetaEvaluator

    meta_evaluator = MetaEvaluator(extractor_model, checker_model)
    for f in files:
        meta_evaluator.evaluate_file(f, ignore_abstentions=honest)

    print("\n" + "█" * 60)
    print("  FULL EVALUATION COMPLETE")
    print("█" * 60 + "\n")


if __name__ == "__main__":
    app()
