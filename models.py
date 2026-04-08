from pydantic import BaseModel
from typing import Optional, List
from enum import Enum

class TaskType(str, Enum):
    RELEVANCE = "relevance"
    HALLUCINATION = "hallucination"
    FULL_JUDGMENT = "full_judgment"

class RAGAction(BaseModel):
    # Agent's judgment output
    relevant_chunk_ids: Optional[List[int]] = None      # for relevance task
    hallucinated_claims: Optional[List[str]] = None     # for hallucination task
    relevance_score: Optional[float] = None             # for full judgment
    faithfulness_score: Optional[float] = None          # for full judgment
    citation_accuracy_score: Optional[float] = None     # for full judgment
    reasoning: Optional[str] = None

class RAGObservation(BaseModel):
    query: str
    retrieved_chunks: List[str]
    chunk_ids: List[int]
    generated_answer: Optional[str] = None
    cited_sources: Optional[List[int]] = None
    task_type: TaskType
    instructions: str

class RAGReward(BaseModel):
    score: float          # 0.0 - 1.0
    feedback: str
    partial_scores: Optional[dict] = None