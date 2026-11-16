"""Helix Cloud helpdesk RAG agent — Langfuse evaluation demo.

    python main.py variants                       # list agent variants
    python main.py ask "How many PTO days do I get?" [--variant baseline] [--score]
    python main.py seed                           # upload data/eval_dataset.json to Langfuse
    python main.py eval --variant all             # run experiments (one Langfuse run per variant)
    python main.py eval --variant baseline --local   # run + score locally, no Langfuse needed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any

from rag_agent.agent import RAGAgent
from rag_agent.config import (
    AGENT_MODEL,
    DATASET_FILE,
    DATASET_NAME,
    JUDGE_MODEL,
    VARIANTS,
    langfuse_configured,
)
from rag_agent.metrics import (
    average_run_evaluator,
    judge,
    llm_judge_evaluator,
    rag_score_composite,
    retrieval_evaluator,
    scores_from_judgment,
)
from rag_agent.tracing import get_langfuse, observation

METRIC_ORDER = ["context_precision", "context_recall", "faithfulness", "answer_correctness", "answer_relevance", "rag_score"]


def load_items() -> list[dict[str, Any]]:
    return json.loads(DATASET_FILE.read_text(encoding="utf-8"))


def to_experiment_item(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "input": {"question": row["question"]},
        "expected_output": {
            "reference_answer": row["reference_answer"],
            "key_facts": row["key_facts"],
            "gold_doc_ids": row["gold_doc_ids"],
        },
        "metadata": {"item_id": row["id"], "category": row["category"]},
    }


def require_langfuse() -> Any:
    if not langfuse_configured():
        sys.exit("LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY are not set (see .env.example), "
                 "or pass --local to evaluate without Langfuse.")
    return get_langfuse()


# ---------------------------------------------------------------------------
# ask
# ---------------------------------------------------------------------------
def cmd_ask(args: argparse.Namespace) -> None:
    agent = RAGAgent(args.variant)
    lf = get_langfuse()

    with observation("helpdesk-question", as_type="span", input={"question": args.question},
                     metadata={"variant": args.variant}) as root:
        if lf is not None:
            from langfuse import propagate_attributes

            ctx = propagate_attributes(trace_name="helpdesk-question", tags=[f"variant:{args.variant}", "ask"])
        else:
            import contextlib

            ctx = contextlib.nullcontext()
        with ctx:
            result = agent.run(args.question)
            root.update(output={"answer": result.answer})

            evals = []
            if args.score:
                # Online evaluation: no reference answer, so only reference-free metrics.
                j = judge(args.question, result.answer, [c.render() for c in result.retrieved], key_facts=[])
                evals = scores_from_judgment(j, has_key_facts=False)
                for e in evals:
                    root.score_trace(name=e.name, value=e.value, comment=e.comment)

    print(f"\nQ: {args.question}\n\nA: {result.answer}\n")
    print(f"searches: {result.searches}")
    print(f"retrieved: {[c.chunk_id for c in result.retrieved]}")
    print(f"tokens: in={result.input_tokens} out={result.output_tokens}")
    for e in evals:
        print(f"{e.name:>18}: {e.value:.2f}  {e.comment}")
    if lf is not None:
        lf.flush()
        print("\nTrace sent to Langfuse.")


# ---------------------------------------------------------------------------
# seed
# ---------------------------------------------------------------------------
def cmd_seed(_: argparse.Namespace) -> None:
    lf = require_langfuse()
    rows = load_items()
    try:
        lf.create_dataset(
            name=DATASET_NAME,
            description="Helix Cloud helpdesk RAG eval set: single-hop, multi-hop, "
            "outdated/conflicting docs, and out-of-scope questions.",
            metadata={"source": str(DATASET_FILE.name)},
        )
    except Exception as e:  # dataset already exists -> items are upserted below
        print(f"(create_dataset: {e.__class__.__name__}; continuing with upsert)")
    for row in rows:
        item = to_experiment_item(row)
        lf.create_dataset_item(dataset_name=DATASET_NAME, id=row["id"], **item)
    lf.flush()
    print(f"Seeded {len(rows)} items into Langfuse dataset '{DATASET_NAME}'.")


# ---------------------------------------------------------------------------
# eval
# ---------------------------------------------------------------------------
def make_task(agent: RAGAgent):
    async def rag_agent_task(*, item: Any, **_: Any) -> dict[str, Any]:
        inp = item["input"] if isinstance(item, dict) else item.input
        result = await asyncio.to_thread(agent.run, inp["question"])
        return result.to_output()

    return rag_agent_task


async def llm_judge(**kwargs: Any):
    return await asyncio.to_thread(llm_judge_evaluator, **kwargs)


def summarize(variant: str, rows: list[tuple[dict[str, Any], dict[str, float]]]) -> None:
    """Print per-item scores plus averages per metric and per category."""
    print(f"\n=== {variant}: {VARIANTS[variant].description}")
    header = f"{'item':<24}{'category':<18}" + "".join(f"{m[:11]:>12}" for m in METRIC_ORDER)
    print(header)
    print("-" * len(header))
    by_cat: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    overall: dict[str, list[float]] = defaultdict(list)
    for meta, scores in rows:
        cells = "".join(f"{scores[m]:>12.2f}" if m in scores else f"{'—':>12}" for m in METRIC_ORDER)
        print(f"{meta.get('item_id', '?'):<24}{meta.get('category', '?'):<18}{cells}")
        for m, v in scores.items():
            by_cat[meta.get("category", "?")][m].append(v)
            overall[m].append(v)
    print("-" * len(header))
    for cat, ms in sorted(by_cat.items()):
        cells = "".join(f"{sum(ms[m]) / len(ms[m]):>12.2f}" if ms.get(m) else f"{'—':>12}" for m in METRIC_ORDER)
        print(f"{'avg':<24}{cat:<18}{cells}")
    cells = "".join(f"{sum(overall[m]) / len(overall[m]):>12.2f}" if overall.get(m) else f"{'—':>12}" for m in METRIC_ORDER)
    print(f"{'AVG':<24}{'(all)':<18}{cells}")


async def run_local(variant: str, items: list[dict[str, Any]], concurrency: int) -> None:
    agent = RAGAgent(variant)
    task = make_task(agent)
    sem = asyncio.Semaphore(concurrency)

    async def one(item: dict[str, Any]):
        async with sem:
            output = await task(item=item)
            kw = dict(input=item["input"], output=output, expected_output=item["expected_output"], metadata=item["metadata"])
            evals = list(retrieval_evaluator(**kw)) + list(await llm_judge(**kw))
            comp = rag_score_composite(**kw, evaluations=evals)
            evals += comp if isinstance(comp, list) else [comp]
            return item["metadata"], {e.name: float(e.value) for e in evals}

    rows = await asyncio.gather(*(one(i) for i in items))
    summarize(variant, list(rows))


def run_langfuse(variant: str, limit: int | None, concurrency: int) -> None:
    lf = require_langfuse()
    dataset = lf.get_dataset(DATASET_NAME)
    if limit:
        dataset.items = dataset.items[:limit]
    run_name = f"{variant}-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}"
    result = dataset.run_experiment(
        name=f"helpdesk-rag-{variant}",
        run_name=run_name,
        description=VARIANTS[variant].description,
        task=make_task(RAGAgent(variant)),
        evaluators=[retrieval_evaluator, llm_judge],
        composite_evaluator=rag_score_composite,
        run_evaluators=[average_run_evaluator],
        max_concurrency=concurrency,
        metadata={"variant": variant, "agent_model": AGENT_MODEL, "judge_model": JUDGE_MODEL,
                  "top_k": str(VARIANTS[variant].top_k), "max_searches": str(VARIANTS[variant].max_searches)},
    )
    rows = []
    for r in result.item_results:
        meta = r.item["metadata"] if isinstance(r.item, dict) else (r.item.metadata or {})
        rows.append((meta, {e.name: float(e.value) for e in r.evaluations if isinstance(e.value, (int, float))}))
    summarize(variant, rows)
    if result.experiment_url:
        print(f"\nLangfuse run: {result.experiment_url}")
    lf.flush()


def cmd_eval(args: argparse.Namespace) -> None:
    variants = list(VARIANTS) if args.variant == "all" else [args.variant]
    items = [to_experiment_item(r) for r in load_items()]
    if args.category:
        items = [i for i in items if i["metadata"]["category"] == args.category]
    if args.limit:
        items = items[: args.limit]
    for v in variants:
        if args.local:
            asyncio.run(run_local(v, items, args.concurrency))
        else:
            run_langfuse(v, args.limit, args.concurrency)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("variants", help="list agent variants")

    a = sub.add_parser("ask", help="ask one question (traced to Langfuse if configured)")
    a.add_argument("question")
    a.add_argument("--variant", default="baseline", choices=list(VARIANTS))
    a.add_argument("--score", action="store_true", help="also score faithfulness/relevance with the LLM judge")

    sub.add_parser("seed", help="create/update the Langfuse dataset from data/eval_dataset.json")

    e = sub.add_parser("eval", help="run the evaluation experiment")
    e.add_argument("--variant", default="baseline", choices=["all", *VARIANTS])
    e.add_argument("--local", action="store_true", help="score locally and print; no Langfuse needed")
    e.add_argument("--limit", type=int, help="only the first N items")
    e.add_argument("--category", help="(local mode) only items of this category")
    e.add_argument("--concurrency", type=int, default=4)

    args = p.parse_args()
    if args.cmd == "variants":
        for v in VARIANTS.values():
            print(f"{v.name:<16} top_k={v.top_k} max_searches={v.max_searches}  {v.description}")
    elif args.cmd == "ask":
        cmd_ask(args)
    elif args.cmd == "seed":
        cmd_seed(args)
    elif args.cmd == "eval":
        cmd_eval(args)


if __name__ == "__main__":
    main()
