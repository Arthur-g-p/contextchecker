"""
The Pipeline. Does extract + check. That's it.

Takes responses[] + references[], returns PipelineItem[] (same length, same order).
Does NOT load files, does NOT save files, does NOT filter. The caller handles all of that.

Usage:
    pipe = Pipeline(Extractor(model="x", baseapi="y"), Checker(model="z"))
    results = await pipe.run(responses, references)

    # Query results:
    for r in results:
        if r.extraction and r.extraction.triplets:
            for triplet, verdict in zip(r.extraction.triplets, r.verdicts):
                print(f"{triplet.subject} | {triplet.predicate} | {triplet.object}")
                print(f"  → {verdict.label}: {verdict.explanation}")
        else:
            print("Abstention / no claims found")
"""

from dataclasses import dataclass
from typing import List

from contextchecker.extractor import Extractor
from contextchecker.checker import Checker
from contextchecker.schemas import ExtractionResult, Verdict


@dataclass
class PipelineItem:
    """One item's results after the pipeline ran."""
    extraction: ExtractionResult | None   # None = extractor error/abstention
    verdicts: list[Verdict]               # empty if no triplets were found


class Pipeline:
    """
    Dumb pipeline: extract claims, check them against references.
    
    No I/O, no filtering, no opinions. output_len == input_len always.
    """

    def __init__(self, extractor: Extractor, checker: Checker):
        self.extractor = extractor
        self.checker = checker

    async def run(self, responses: list[str], references: list[str]) -> list[PipelineItem]:
        """
        Run the full extract + check pipeline.
        
        Args:
            responses: The LLM responses to extract claims from
            references: The reference texts to check claims against (same length)
        
        Returns:
            List[PipelineItem] — one per input item, same order
        
        Edge cases (handled gracefully, NOT fatal):
            - Empty response string → extractor returns ExtractionResult(triplets=[])
              → checker gets 0 claims → skip → empty verdicts. Item preserved.
            - Extractor API error → extraction is None → empty verdicts. Item preserved.
            - Checker API error on one claim → Verdict(label="Neutral", explanation="Parsing Failed")
              → item still has partial results. Item preserved.
            - All items empty → returns list of empty PipelineItems. No crash.
        
        Known limitation (TODO: interim result caching):
            - If the process dies mid-run (e.g. 402 Payment Required after 50/100 items),
              all extraction results are lost. Next step: cache interim results to disk
              so we can resume from where we left off.
        """
        assert len(responses) == len(references), \
            f"responses ({len(responses)}) and references ({len(references)}) must be same length"

        # --- PHASE 1: EXTRACTION ---
        extraction_results = await self.extractor.extract_batch(responses)

        # --- PHASE 2: BUILD CHECKER INPUT ---
        # Convert ExtractionResult triplets → string claims for checker
        claims_batch: list[list[str]] = []
        for result in extraction_results:
            if result and result.triplets:
                claims_batch.append([str(t) for t in result.triplets])
            else:
                claims_batch.append([])

        # --- PHASE 3: CHECKING ---
        verdicts_batch = await self.checker.check_batch(claims_batch, references)

        # --- PHASE 4: ZIP INTO RESULTS ---
        items: list[PipelineItem] = []
        for extraction, verdicts in zip(extraction_results, verdicts_batch):
            items.append(PipelineItem(extraction=extraction, verdicts=verdicts))

        return items
