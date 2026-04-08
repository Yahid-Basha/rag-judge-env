import os
import json
import re
from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

try:
    from server.rag_judge_env_environment import RagJudgeEnvEnvironment
except ImportError:
    from rag_judge_env.server.rag_judge_env_environment import RagJudgeEnvEnvironment

try:
    from models import RAGAction, TaskType
except ImportError:
    from rag_judge_env.models import RAGAction, TaskType

API_KEY = os.getenv("HF_TOKEN") or os.getenv("API_KEY")
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
BENCHMARK = "rag_judge_env"

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

TASK_TYPES = ["relevance", "hallucination", "full_judgment"]


def extract_json(text: str) -> str:
    """Strip markdown code fences if present and return raw JSON string."""
    text = text.strip()
    # Remove ```json ... ``` or ``` ... ``` fences
    match = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
    if match:
        return match.group(1).strip()
    return text


def build_prompt(obs) -> str:
    base = f"""You are a RAG evaluation expert.

Query: {obs.query}

Retrieved Chunks:
{chr(10).join(f'[{i}] {c}' for i, c in zip(obs.chunk_ids, obs.retrieved_chunks))}
"""
    if obs.generated_answer:
        base += f"\nGenerated Answer: {obs.generated_answer}"
    if obs.cited_sources:
        base += f"\nCited chunk IDs: {obs.cited_sources}"

    base += f"\n\nTask Instructions: {obs.instructions}"
    base += "\n\nRespond ONLY with valid JSON matching the RAGAction schema (fields: relevant_chunk_ids, hallucinated_claims, relevance_score, faithfulness_score, citation_accuracy_score, reasoning). No markdown, no explanation."
    return base


def run_task(task_type: str):
    env = RagJudgeEnvEnvironment()
    obs = env.reset(task_type=task_type)

    print(f"[START] task={task_type} env={BENCHMARK} model={MODEL_NAME}")

    prompt = build_prompt(obs)
    error_msg = "null"
    action_json = "{}"

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=500
        )
        raw = response.choices[0].message.content.strip()
        clean = extract_json(raw)
        data = json.loads(clean)
        action = RAGAction(**data)
        action_json = json.dumps(data)
    except Exception as e:
        error_msg = str(e).replace("\n", " ")[:120]
        action = RAGAction(reasoning=f"parse error: {e}")
        action_json = json.dumps({"reasoning": f"parse error: {e}"[:80]})

    obs_next, reward, done, info = env.step(action)

    print(f"[STEP] step=1 action={action_json} reward={reward.score:.2f} done={str(done).lower()} error={error_msg}")
    print(f"[END] success={str(reward.score > 0.5).lower()} steps=1 score={reward.score:.2f} rewards={reward.score:.2f}")

    return reward.score


if __name__ == "__main__":
    for task in TASK_TYPES:
        run_task(task)
