import random
from typing import Optional

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import RAGAction, RAGObservation, RAGReward, TaskType
except ImportError:
    from models import RAGAction, RAGObservation, RAGReward, TaskType

try:
    from .dataset import TASKS
except ImportError:
    from dataset import TASKS


class RagJudgeEnvEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.current_task_type = None
        self.current_data = None
        self.done = False
        self.steps = 0
        self.max_steps = 8

    def reset(self, seed: Optional[int] = None, episode_id: Optional[str] = None, **kwargs) -> RAGObservation:
        self.done = False
        self.steps = 0

        task_type = kwargs.get("task_type", None)
        if task_type is None:
            task_type = random.choice(["relevance", "hallucination", "full_judgment"])

        self.current_task_type = TaskType(task_type)
        pool = TASKS[task_type]
        weights = [s.get("weight", 1) for s in pool]
        self.current_data = random.choices(pool, weights=weights, k=1)[0]

        return self._build_observation()

    def step(self, action: RAGAction, timeout_s: Optional[float] = None, **kwargs) -> RAGObservation:
        self.steps += 1
        reward_obj = self._grade(action)
        self.done = True

        obs = self._build_observation()
        obs.reward = reward_obj.score
        obs.done = self.done
        # stash full reward for callers that need feedback/partial_scores
        obs.metadata["feedback"] = reward_obj.feedback
        if reward_obj.partial_scores:
            obs.metadata["partial_scores"] = reward_obj.partial_scores
        return obs

    @property
    def state(self) -> State:
        return State(
            episode_id=None,
            step_count=self.steps,
        )

    def _build_observation(self) -> RAGObservation:
        d = self.current_data
        if self.current_task_type == TaskType.RELEVANCE:
            return RAGObservation(
                query=d["query"],
                retrieved_chunks=d["chunks"],
                chunk_ids=list(range(len(d["chunks"]))),
                task_type=self.current_task_type,
                instructions="Identify which chunk IDs are relevant to the query. Set relevant_chunk_ids in your action."
            )
        elif self.current_task_type == TaskType.HALLUCINATION:
            return RAGObservation(
                query=d["query"],
                retrieved_chunks=[d["context"]],
                chunk_ids=[0],
                generated_answer=d["answer"],
                task_type=self.current_task_type,
                instructions="Identify hallucinated claims in the answer not supported by context. Set hallucinated_claims in your action."
            )
        else:
            return RAGObservation(
                query=d["query"],
                retrieved_chunks=d["chunks"],
                chunk_ids=list(range(len(d["chunks"]))),
                generated_answer=d["answer"],
                cited_sources=d["cited_ids"],
                task_type=self.current_task_type,
                instructions="Score relevance, faithfulness, and citation accuracy between 0.0 and 1.0."
            )

    def _grade(self, action: RAGAction) -> RAGReward:
        d = self.current_data

        if self.current_task_type == TaskType.RELEVANCE:
            predicted = set(action.relevant_chunk_ids or [])
            ground_truth = set(d["relevant_ids"])
            if not ground_truth:
                score = 1.0 if not predicted else 0.0
            else:
                tp = len(predicted & ground_truth)
                fp = len(predicted - ground_truth)
                fn = len(ground_truth - predicted)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                score = (2 * precision * recall / (precision + recall)
                         if (precision + recall) > 0 else 0.0)
            return RAGReward(
                score=round(score, 2),
                feedback=f"F1 score against ground truth relevant chunks: {score:.2f}"
            )

        elif self.current_task_type == TaskType.HALLUCINATION:
            predicted = [c.lower().strip() for c in (action.hallucinated_claims or [])]
            ground_truth = [h.lower().strip() for h in d["hallucinations"]]
            matched = sum(
                1 for gt in ground_truth
                if any(gt in p or p in gt for p in predicted)
            )
            score = matched / len(ground_truth) if ground_truth else 1.0
            over_flag_penalty = max(0, len(predicted) - len(ground_truth)) * 0.1
            score = max(0.0, round(score - over_flag_penalty, 2))
            return RAGReward(
                score=score,
                feedback=f"Detected {matched}/{len(ground_truth)} hallucinations. Penalty: {over_flag_penalty}"
            )

        else:  # full_judgment
            gt = d["ground_truth"]
            scores = {
                "relevance": 1.0 - abs((action.relevance_score or 0) - gt["relevance"]),
                "faithfulness": 1.0 - abs((action.faithfulness_score or 0) - gt["faithfulness"]),
                "citation": 1.0 - abs((action.citation_accuracy_score or 0) - gt["citation_accuracy"])
            }
            scores = {k: max(0.0, round(v, 2)) for k, v in scores.items()}
            final = round((scores["relevance"] * 0.3 +
                          scores["faithfulness"] * 0.4 +
                          scores["citation"] * 0.3), 2)
            return RAGReward(
                score=final,
                feedback="Weighted score: relevance 30%, faithfulness 40%, citation 30%",
                partial_scores=scores
            )
