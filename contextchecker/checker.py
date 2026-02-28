import json
import os
from openai import AsyncOpenAI
from typing import List
import contextchecker.config as config
from contextchecker.utils import format_prompt
from contextchecker.llmclient import LLMClient
from contextchecker.schemas import Verdict

#Should the checker include the Q?!?!

class Checker:
    def __init__(self, model: str = "", baseapi: str = None, concurrency: int = 10): 
        self.model = model
        self.client = LLMClient(
            api_key=config.CHECKER_API_KEY,
            model=model,
            base_url=baseapi,
            concurrency=concurrency 
        )
        self.prompts = config.PROMPTS


    async def check_batch(self, claims_batch: List[List[str]], references_batch: List[str]) -> List[List[Verdict]]:
        """
        Checks a batch of documents. 
        Input: A list where each item is a list of claims for each refrence
        Output: A list where each item is a list of Verdicts for each claim.
        """
        print(f"Preparing checking of {len(claims_batch)} responses...")

        # check for empty triplets or abstentions!
        # 1. Flatten the structure for the API Batch
        # We need to map every single claim to its specific reference text.
        flat_tasks = []
        doc_lengths = [] # Keeps track of which document the claim belongs to

        for claims, ref in zip(claims_batch, references_batch):
            is_abstain = len(claims) == 0
            if is_abstain:
                doc_lengths.append(0)
                continue
                
            # Store the count so we know how to rebuild later
            doc_lengths.append(len(claims)) 
            
            for claim in claims:
                # Prepare the prompt/task as before
                prompt_text = format_prompt(self.prompts["checker_prompt"], {"claim": claim, "reference": ref})
                flat_tasks.append({
                    "messages": [{"role": "system", "content": "You are a precise fact-checking assistant."},
                                 {"role": "user", "content": prompt_text}],
                    "schema": Verdict,
                    "temperature": 0.0
                })

        # 2. Execute Flattened Batch (Protected by Semaphore & TQDM)
        if not flat_tasks:
            print("No claims to check.")
            return [[] for _ in claims_batch]

        print(f"Executing {len(flat_tasks)} individual checks...")
        raw_responses = await self.client.generate_batch(flat_tasks, description="Verifying Claims")
        # 3. Reconstruct using an Iterator (The "Pythonic" part)
        results_iter = iter(raw_responses) # Turn list into a stream we can consume
        structured_results = []

        for count in doc_lengths:
            # Consume exactly 'count' items from the flat results
            doc_verdicts = []
            for _ in range(count):
                raw_text = next(results_iter)
                try:
                    verdict = Verdict.model_validate_json(raw_text)
                    doc_verdicts.append(verdict)
                except Exception as e:
                    print(f"[Checker Error] Parsing failed: {e}")
                    doc_verdicts.append(Verdict(claim="Error", verdict="Neutral", explanation="Parsing Failed"))
            
            structured_results.append(doc_verdicts)

        return structured_results

    async def check_single(self, claims: List[str], reference: str) -> List[Verdict]:
        """
        Checks a list of claims against a single reference text.
        """
        print(f"Preparing checking of {len(claims)} claims on {len(reference)} refrences...")

        if not claims:
            return []

        # 1. Prepare Batch Tasks
        batch_tasks = []
        for claim in claims:
            # We format the prompt: "Is {claim} supported by {reference}?"
            prompt_text = format_prompt(
                self.prompts["checker_prompt"], 
                {"claim": claim, "reference": reference}
            )
            
            messages = [
                {"role": "system", "content": "You are a fact-checking assistant."},
                {"role": "user", "content": prompt_text}
            ]
            
            # Add to batch
            # Only content-related params — execution params are owned by the strategy
            batch_tasks.append({
                "messages": messages,
                "schema": Verdict, # Force structured output!
                "temperature": 0.0
            })

        # 2. Execute Batch (Protected by Semaphore & TQDM)
        # description="" prevents TQDM spam if we call this inside another loop
        raw_responses = await self.client.generate_batch(batch_tasks, description="Checking")

        # 3. Validate & Parse
        results = []
        for raw_text in raw_responses:
            try:
                # Validierung gegen das Pydantic Schema
                verdict = Verdict.model_validate_json(raw_text)
                results.append(verdict)
            except Exception as e:
                print(f"[Checker Error] Could not parse verdict: {e}")
                # Fallback: Create a Neutral verdict on error
                results.append(Verdict(claim="Error", label="Neutral", explanation="Parsing Failed"))
        
        return results