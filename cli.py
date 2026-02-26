import typer
import asyncio
import sys
import glob
import os
import json

app = typer.Typer(help="ContextChecker — RAG Hallucination Checking Pipeline. See EVALUATION.md for full documentation.")
eval_app = typer.Typer(help="Evaluation subcommands. See EVALUATION.md for data requirements and workflows.")
app.add_typer(eval_app, name="eval")

# --- Shared defaults ---
DEFAULT_MODEL = None
DEFAULT_API = None


def _async_run(coro):
    """Helper to run async code with proper Windows support."""
    if sys.platform == 'win32':
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    asyncio.run(coro)


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


def _validate_data_keys(filepath: str, required_keys: list[str], context: str) -> list:
    """Load JSON and validate that required keys exist in at least some items."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        typer.echo(f"❌ Cannot load {filepath}: {e}")
        raise typer.Exit(1)

    if not isinstance(data, list) or len(data) == 0:
        typer.echo(f"❌ {filepath}: Expected a non-empty JSON array.")
        raise typer.Exit(1)

    # Check first item for required keys
    sample = data[0]
    missing = [k for k in required_keys if k not in sample]
    if missing:
        typer.echo(f"\n❌ DATA VALIDATION FAILED for '{context}'")
        typer.echo(f"   File: {filepath}")
        typer.echo(f"   Missing keys: {missing}")
        typer.echo(f"   Available keys: {list(sample.keys())}")
        typer.echo(f"\n   See EVALUATION.md for required data format.\n")
        raise typer.Exit(1)

    return data


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
    """Full extract + check pipeline. Output can be evaluated with 'eval meta'."""
    typer.echo(f"Running full pipeline: {input} → {output}")
    typer.echo(f"Extractor: {extractor_model} @ {extractor_base_api}")
    typer.echo(f"Checker:   {checker_model} @ {checker_base_api}")
    typer.echo("(Use main.py directly for now — CLI integration coming soon)")


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
        _validate_data_keys(f, ["claude2_response_kg"], "eval extractor")

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
    _validate_data_keys(data, ["claude2_response_kg", "context"], "eval checker")

    from contextchecker.checker import Checker
    from contextchecker.eval.checkereval import CheckerEvaluator

    checker = Checker(model=checker_model, baseapi=checker_base_api)
    evaluator = CheckerEvaluator(checker)
    _async_run(evaluator.evaluate_file(data))


# ═══════════════════════════════════════════════════════════════
#  EVAL META — End-to-end pipeline evaluation (offline)
# ═══════════════════════════════════════════════════════════════

@eval_app.command("meta")
def eval_meta(
    data: str = typer.Option(None, help="Path to pipeline output JSON (or auto-detect from results/)"),
    extractor_model: str = typer.Option(DEFAULT_MODEL, help="Extractor model name"),
    checker_model: str = typer.Option(DEFAULT_MODEL, help="Checker model name"),
    honest: bool = typer.Option(True, help="Honest mode: skip abstentions. --no-honest = legacy/RefChecker mode"),
):
    """Meta-evaluation: item-level factuality comparison (offline).

    Requires pipeline output with both extraction ({model}_response_kg) and
    checking ({model}_label) already completed. Run 'contextchecker run' first,
    or use existing results from results/.

    See EVALUATION.md for details.
    """
    from contextchecker.eval.metaeval import MetaEvaluator

    evaluator = MetaEvaluator(extractor_model, checker_model)

    for f in _resolve_data_files(data):
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
