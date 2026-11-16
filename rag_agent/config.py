"""Central configuration: models, paths and the agent *variants* used in the demo."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

KB_DIR = ROOT / "knowledge_base"
DATASET_FILE = ROOT / "data" / "eval_dataset.json"
DATASET_NAME = os.getenv("LANGFUSE_DATASET_NAME", "helix-helpdesk-rag")

AGENT_MODEL = os.getenv("AGENT_MODEL", "claude-opus-5-5")
JUDGE_MODEL = os.getenv("JUDGE_MODEL", "claude-opus-5-5")
AGENT_EFFORT = os.getenv("AGENT_EFFORT", "medium")
JUDGE_EFFORT = os.getenv("JUDGE_EFFORT", "medium")

# Server-side refusal fallback (re-runs a declined request on a fallback model
# inside the same API call). Set ENABLE_FALLBACKS=false to turn it off.
ENABLE_FALLBACKS = os.getenv("ENABLE_FALLBACKS", "true").lower() != "false"
FALLBACK_BETA = "server-side-fallback-2026-07-01"


GROUNDED_PROMPT = """You are the Helix Cloud internal helpdesk assistant.
Answer employee questions about company policy using ONLY the knowledge base.

- Always call `search_knowledge_base` before answering. Search again with a
  different query if the question has several parts or the first results don't
  cover everything.
- Base every statement on retrieved text. If documents conflict, prefer the one
  marked CURRENT and ignore SUPERSEDED/ARCHIVED ones.
- If the knowledge base doesn't contain the answer, say so plainly. Do not guess.
- Keep the answer short (2-5 sentences) and cite the source doc ids in brackets,
  e.g. [pto_policy]."""

LOOSE_PROMPT = """You are a friendly, confident helpdesk assistant for Helix Cloud.
You may search the knowledge base once if you like, but you don't have to.
Employees hate hearing "I don't know", so always give a specific, concrete
answer. If the documents don't cover something, fill the gap with what is typical
for tech companies. Keep the answer short (2-5 sentences)."""


@dataclass(frozen=True)
class AgentVariant:
    """One configuration of the RAG agent. Each variant is a separate experiment run."""

    name: str
    description: str
    top_k: int
    max_searches: int
    system_prompt: str


VARIANTS: dict[str, AgentVariant] = {
    v.name: v
    for v in [
        AgentVariant(
            name="baseline",
            description="Grounded prompt, top_k=3, up to 3 searches. Expected: high scores.",
            top_k=3,
            max_searches=3,
            system_prompt=GROUNDED_PROMPT,
        ),
        AgentVariant(
            name="low_recall",
            description="top_k=1 and only ONE search. Misses the 2nd doc on multi-hop questions "
            "-> context_recall ~0.5, answer_correctness drops.",
            top_k=1,
            max_searches=1,
            system_prompt=GROUNDED_PROMPT,
        ),
        AgentVariant(
            name="noisy_retrieval",
            description="top_k=8. Recall stays high but most chunks are irrelevant "
            "-> context_precision ~0.1-0.4.",
            top_k=8,
            max_searches=3,
            system_prompt=GROUNDED_PROMPT,
        ),
        AgentVariant(
            name="no_guardrails",
            description="Loose prompt that forbids 'I don't know', 1 search, top_k=2. "
            "Hallucinates on out-of-scope questions -> faithfulness ~0.0-0.5.",
            top_k=2,
            max_searches=1,
            system_prompt=LOOSE_PROMPT,
        ),
    ]
}


def langfuse_configured() -> bool:
    return bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
