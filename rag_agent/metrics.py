"""Evaluation metrics, written as Langfuse experiment evaluators.

Every metric is a number in [0, 1] and most are *fractions*, which is why you
see values like 0.2, 0.33, 0.5, 0.67 in Langfuse rather than just 0 / 1:

Retrieval (deterministic, no LLM):
  context_precision  = retrieved chunks from a gold doc / retrieved chunks
  context_recall     = gold docs retrieved / gold docs

Generation (Claude as judge; the judge only labels, the score math is in code):
  faithfulness       = answer claims supported by retrieved context / all claims
  answer_correctness = reference key facts covered by the answer / key facts
                       (covered = 1, partially covered = 0.5)
  answer_relevance   = 1-5 rubric mapped to 0, 0.25, 0.5, 0.75, 1

Composite:
  rag_score          = mean of the metrics above that apply to the item
"""

from __future__ import annotations

import json
from typing import Any, Literal

import anthropic
from langfuse import Evaluation
from pydantic import BaseModel, Field

from .config import ENABLE_FALLBACKS, FALLBACK_BETA, JUDGE_EFFORT, JUDGE_MODEL
from .tracing import observation

_client: anthropic.Anthropic | None = None


def _anthropic() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


def set_judge_client(client: anthropic.Anthropic) -> None:
    """Inject a client (used by tests)."""
    global _client
    _client = client


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {"answer": value}
        except json.JSONDecodeError:
            return {"answer": value}
    return {}


# ---------------------------------------------------------------------------
# Retrieval metrics (deterministic)
# ---------------------------------------------------------------------------
def retrieval_evaluator(*, input: Any, output: Any, expected_output: Any, metadata: Any = None, **_: Any):
    out, exp = _as_dict(output), _as_dict(expected_output)
    gold = set(exp.get("gold_doc_ids") or [])
    if not gold:
        # Out-of-scope question: there is nothing correct to retrieve, so
        # precision/recall are undefined. Skip rather than report a fake 0 or 1.
        return []

    chunk_ids: list[str] = out.get("retrieved_chunk_ids") or []
    chunk_docs = [cid.split("#", 1)[0] for cid in chunk_ids]
    retrieved_docs = set(out.get("retrieved_doc_ids") or chunk_docs)

    relevant_chunks = sum(1 for d in chunk_docs if d in gold)
    precision = relevant_chunks / len(chunk_docs) if chunk_docs else 0.0
    hit_docs = gold & retrieved_docs
    recall = len(hit_docs) / len(gold)

    return [
        Evaluation(
            name="context_precision",
            value=round(precision, 3),
            comment=f"{relevant_chunks}/{len(chunk_docs)} retrieved chunks come from gold docs {sorted(gold)}",
        ),
        Evaluation(
            name="context_recall",
            value=round(recall, 3),
            comment=f"found {len(hit_docs)}/{len(gold)} gold docs; missing: {sorted(gold - hit_docs) or 'none'}",
        ),
    ]


# ---------------------------------------------------------------------------
# LLM-as-judge metrics
# ---------------------------------------------------------------------------
class ClaimVerdict(BaseModel):
    claim: str
    supported: bool = Field(description="True only if the CONTEXT explicitly states this claim.")


class FactVerdict(BaseModel):
    fact: str
    verdict: Literal["covered", "partially_covered", "missing", "contradicted"]


class Judgment(BaseModel):
    claims: list[ClaimVerdict]
    facts: list[FactVerdict]
    relevance: int = Field(description="1-5: how directly the answer addresses the question.")
    relevance_reason: str


JUDGE_SYSTEM = """You are a strict, impartial evaluator of a retrieval-augmented helpdesk assistant.
You label; you do not compute scores. Follow the instructions exactly."""

JUDGE_TEMPLATE = """<question>
{question}
</question>

<retrieved_context>
{context}
</retrieved_context>

<answer>
{answer}
</answer>

<reference_key_facts>
{facts}
</reference_key_facts>

Tasks:
1. claims: Split the ANSWER into atomic factual claims (numbers, rules, conditions,
   procedures). Skip citations, pleasantries, and statements that information is
   unavailable. For each claim set supported=true ONLY if the RETRIEVED CONTEXT
   explicitly states it. Claims from general knowledge, or from a document marked
   SUPERSEDED when presented as current, are NOT supported. If the answer makes no
   factual claims, return an empty list.
2. facts: For EACH reference key fact (same order, copy the text), judge the ANSWER:
   covered / partially_covered / missing / contradicted.
   {facts_note}
3. relevance: 1-5, how directly the ANSWER addresses what was asked, regardless of
   correctness (5 = fully on point, 3 = partly on point or padded, 1 = off-topic).
   Give a one-sentence relevance_reason."""


def judge(question: str, answer: str, contexts: list[str], key_facts: list[str]) -> Judgment:
    prompt = JUDGE_TEMPLATE.format(
        question=question,
        context="\n\n---\n\n".join(contexts) if contexts else "(nothing was retrieved)",
        answer=answer or "(empty answer)",
        facts="\n".join(f"- {f}" for f in key_facts) if key_facts else "(none provided)",
        facts_note="" if key_facts else "No key facts were provided: return an empty list.",
    )
    kwargs: dict[str, Any] = dict(
        model=JUDGE_MODEL,
        max_tokens=16000,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
        output_format=Judgment,
        output_config={"effort": JUDGE_EFFORT},
    )
    if ENABLE_FALLBACKS:
        kwargs.update(betas=[FALLBACK_BETA], fallbacks="default")

    with observation("llm-judge", as_type="evaluator", model=JUDGE_MODEL, input={"prompt": prompt}) as span:
        response = _anthropic().beta.messages.parse(**kwargs)
        if response.stop_reason == "refusal" or response.parsed_output is None:
            raise RuntimeError(f"judge returned no verdict (stop_reason={response.stop_reason})")
        span.update(output=response.parsed_output.model_dump())
        return response.parsed_output


_FACT_POINTS = {"covered": 1.0, "partially_covered": 0.5, "missing": 0.0, "contradicted": 0.0}


def scores_from_judgment(j: Judgment, has_key_facts: bool) -> list[Evaluation]:
    evals: list[Evaluation] = []

    if j.claims:
        supported = sum(c.supported for c in j.claims)
        unsupported = [c.claim for c in j.claims if not c.supported]
        evals.append(Evaluation(
            name="faithfulness",
            value=round(supported / len(j.claims), 3),
            comment=f"{supported}/{len(j.claims)} claims supported by context."
            + (f" Unsupported: {unsupported}" if unsupported else ""),
        ))
    else:
        evals.append(Evaluation(name="faithfulness", value=1.0,
                                comment="No factual claims (abstained) — nothing unsupported."))

    if has_key_facts and j.facts:
        pts = sum(_FACT_POINTS[f.verdict] for f in j.facts)
        evals.append(Evaluation(
            name="answer_correctness",
            value=round(pts / len(j.facts), 3),
            comment="; ".join(f"{f.verdict}: {f.fact}" for f in j.facts),
        ))

    rel = min(max(j.relevance, 1), 5)
    evals.append(Evaluation(name="answer_relevance", value=(rel - 1) / 4, comment=f"{rel}/5 — {j.relevance_reason}"))
    return evals


def llm_judge_evaluator(*, input: Any, output: Any, expected_output: Any, metadata: Any = None, **_: Any):
    out, exp, inp = _as_dict(output), _as_dict(expected_output), _as_dict(input)
    question = inp.get("question") or inp.get("answer") or str(input)
    key_facts = exp.get("key_facts") or []
    j = judge(question, out.get("answer", ""), out.get("contexts") or [], key_facts)
    return scores_from_judgment(j, has_key_facts=bool(key_facts))


# ---------------------------------------------------------------------------
# Composite + run-level
# ---------------------------------------------------------------------------
COMPONENT_METRICS = ("context_precision", "context_recall", "faithfulness", "answer_correctness", "answer_relevance")


def rag_score_composite(*, input: Any, output: Any, expected_output: Any, metadata: Any, evaluations: list[Evaluation], **_: Any):
    vals = {e.name: e.value for e in evaluations if e.name in COMPONENT_METRICS and isinstance(e.value, (int, float))}
    if not vals:
        return []
    return Evaluation(
        name="rag_score",
        value=round(sum(vals.values()) / len(vals), 3),
        comment="mean of " + ", ".join(f"{k}={v}" for k, v in vals.items()),
    )


def average_run_evaluator(*, item_results: list[Any], **_: Any):
    """Run-level averages, shown on the experiment run in Langfuse."""
    buckets: dict[str, list[float]] = {}
    for r in item_results:
        for e in r.evaluations:
            if isinstance(e.value, (int, float)):
                buckets.setdefault(e.name, []).append(float(e.value))
    return [
        Evaluation(name=f"avg_{name}", value=round(sum(v) / len(v), 3), comment=f"n={len(v)}")
        for name, v in sorted(buckets.items())
    ]


ITEM_EVALUATORS = [retrieval_evaluator, llm_judge_evaluator]
