from typing import List
import src.config as config
from src.utils import format_prompt
from src.llmclient import LLMClient
from src.schemas import ExtractionResult

class Extractor:
    def __init__(self, baseapi, model, concurrency=15):
        self.model=model
        self.client = LLMClient(
            base_url=baseapi,
            api_key=config.EXTRACTOR_API_KEY,
            model=model,
            concurrency=concurrency 
        )
        self.prompts = config.PROMPTS

    async def extract(self, response: str):
        """
        Ein intelligenter Worker-Task.
        Er beißt sich an einem Text fest und lässt erst los, wenn er ein Ergebnis hat
        oder die Matrix erschöpft ist. Only takes the unformated response as input
        """
        prompt_text = format_prompt(self.prompts["extractor_prompt"], {"text": response})
        messages = [{"role": "user", "content": prompt_text}]

        for i in 3:
            try:
                
                pass
            except:
                pass

    async def extract_batch(self, responses: List[str]):
        print(f"Preparing extraction for {len(responses)} responses...")

        batch_tasks = []
        for response in responses:
            prompt_text = format_prompt(self.prompts["extractor_prompt"], {"text": response})
            
            messages = [
                {"role": "system", "content": "Extract knowledge triplets."},
                {"role": "user", "content": prompt_text}
            ]
            
            # Wir packen alles in ein Dict, was der Client.generate() braucht
            batch_tasks.append({
                "messages": messages,
                "schema": ExtractionResult, # Wir fordern JSON an!
                "temperature": 0.0,
                "reasoning_effort":"low", 
            })

        # 2. Ausführen (Client kümmert sich um Semaphore & TQDM)
        raw_responses = await self.client.generate_batch(batch_tasks, description="Extracting")

        # 3. Validierung & Parsing (Hier ist die Fachlogik!)
        results = []
        for raw_text in raw_responses:
            try:
                # Versuch 1: Ist es valides JSON?
                parsed_obj = ExtractionResult.model_validate_json(raw_text)
                results.append(parsed_obj)
            except Exception as e:
                if raw_text is None:
                    print(f"[Validation Failed] Got an EMPTY TEXT")
                else:
                    print(f"[Validation Failed] Got: {raw_text[:50]}... Error: {e}")
                # --- MATRIX LOGIK START ---
                # Hier könntest du jetzt sagen:
                # "Okay, Parsing fehlgeschlagen. Ich rufe self.client.generate() nochmal auf,
                # aber diesmal OHNE Schema (schema=None) und lasse es reparieren."
                # --------------------------
                
                results.append(None) # Oder Error-Objekt

        return results
    
    def parse_extraction(self, responses):
        # See if convertable to JSON

        # Return list of claims
        return