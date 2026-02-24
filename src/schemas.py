from pydantic import BaseModel, Field
from typing import List, Literal

from typing import Optional

# input for refchecker
class InputItem(BaseModel):
    reference: str
    question: str
    response: str

# --- EXTRACTOR OUTPUT ---
class Triplet(BaseModel):
    subject: str
    predicate: str
    object: str

class ExtractionResult(BaseModel):
    triplets: List[Triplet]

# --- CHECKER OUTPUT ---
class Verdict(BaseModel):
    claim: str # The triplet converted back to string for reference
    label: Literal["Entailment", "Contradiction", "Neutral"]
    explanation: str = Field(description="Short reasoning for the verdict")

class VerdictResult(BaseModel):
    verdicts: List[Verdict]

class ClaimValidationResult(BaseModel):
    claim: str = Field(description="The triplet converted back to string")
    is_faithful: bool = Field(description="True if supported by text, False if hallucinated")
    explanation: str = Field(description="Short reasoning for the verdict")

class MissingClaimsResult(BaseModel):
    triplets: List[Triplet]

# Der entscheidende Wrapper für den Batch-Call!
class ValidationBatchResult(BaseModel):
    results: List[ClaimValidationResult]
