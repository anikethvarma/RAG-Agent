# RAG Agent: Langfuse evaluation demo

This is a small RAG agent built on Claude. It answers employee questions for a
fictional company, **Helix Cloud**, using a set of internal policy documents.
It sends a trace for every run to **Langfuse** and scores each answer with
**six metrics between 0 and 1**.

The demo is designed so the scores spread out. You get values like **0.12,
0.33, 0.5, 0.67**, not only 0 or 1:

- The company and its policies are made up, so Claude can't answer from memory.
  The scores reflect what the agent retrieved.
- Most metrics are **fractions**: supported claims divided by all claims, gold
  docs found divided by gold docs, key facts covered divided by key facts.
- The dataset includes **traps** on purpose: questions that need two documents,
  an outdated policy that conflicts with the current one, and questions the
  documents don't answer.
- There are **4 agent variants**. Each one is configured to fail on a different
  metric.

## How it works

```
question ─► Claude (claude-opus-5-5) ──tool_use──► search_knowledge_base (BM25, top_k)
               ▲                                        │
               └────────────── tool_result ◄────────────┘
                     ...loop until answer...

Langfuse trace:
  rag-agent (agent)
  ├─ claude-call-1 (generation)        tokens, stop_reason
  ├─ search_knowledge_base (retriever) query → chunk ids + scores
  ├─ claude-call-2 (generation)
  └─ evaluators: retrieval_evaluator, llm_judge (+ llm-judge generation), rag_score_composite
```

| Path | What it is |
|---|---|
| `knowledge_base/*.md` | 11 policy documents. `travel_policy_2023.md` is **superseded** on purpose. |
| `data/eval_dataset.json` | 14 questions with a reference answer, `key_facts`, `gold_doc_ids` and `category` |
| `rag_agent/retriever.py` | BM25 retriever with no dependencies. Each `## section` is one chunk. |
| `rag_agent/agent.py` | Claude tool-use loop. Each Claude call and each search becomes its own Langfuse observation. |
| `rag_agent/metrics.py` | Metrics, written as Langfuse experiment evaluators |
| `rag_agent/config.py` | Models and the 4 **variants** |
| `main.py` | CLI: `ask`, `seed`, `eval`, `variants` |

## Metrics

| Metric | How it's computed | Typical "bad" value |
|---|---|---|
| `context_precision` | retrieved chunks from a gold doc ÷ retrieved chunks (no LLM) | 0.12–0.38 with top_k=8 |
| `context_recall` | gold docs retrieved ÷ gold docs (no LLM) | 0.5 on multi-hop with top_k=1 |
| `faithfulness` | answer claims supported by the retrieved context ÷ all claims (Claude labels each claim) | 0.0–0.5 when the agent makes things up |
| `answer_correctness` | reference key facts covered ÷ key facts (covered = 1, partially covered = 0.5) | 0.33–0.67 on partial answers |
| `answer_relevance` | 1–5 score from the judge, mapped to 0, 0.25, 0.5, 0.75 or 1 | 0.5 when the answer is padded or only partly on topic |
| `rag_score` | average of the metrics above that apply to the item (composite evaluator) | |

The judge only **labels** claims and facts. The scores are calculated in
Python, so the same labels always give the same numbers. Retrieval metrics are
skipped for out-of-scope questions, because nothing in the documents is correct
to retrieve. Each run also logs `avg_*` scores for the whole run.

## Variants: one Langfuse experiment run each

| Variant | Config | What you'll see in Langfuse |
|---|---|---|
| `baseline` | grounded prompt, top_k=3, ≤3 searches | Mostly 0.8–1.0. `context_precision` is around 0.67 when weak chunks fill the remaining slots. |
| `low_recall` | top_k=1, **1 search** | `context_recall` **0.5** on multi-hop questions. `answer_correctness` drops to about 0.33–0.67. |
| `noisy_retrieval` | **top_k=8** | `context_precision` **0.12–0.62**. Recall stays at 1.0. |
| `no_guardrails` | prompt tells it never to say "I don't know" and to fill gaps with typical company policy | `faithfulness` **0.0–0.5** and `answer_correctness` around 0 on the 401(k) and dog questions |

Dataset categories: `single_hop` (6), `multi_hop` (4), `outdated_conflict` (2),
`out_of_scope` (2).

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
```

```bash
pip install -r requirements.txt
```

```bash
cp .env.example .env
```

Then fill in `ANTHROPIC_API_KEY`, `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`
and `LANGFUSE_BASE_URL`. You can use Langfuse Cloud (free tier) or a
self-hosted instance.

## Run the demo

1. Ask one question. This creates a trace with nested generation and retriever
   spans. `--score` adds online faithfulness and relevance scores to the trace.

   ```bash
   python main.py ask "What is the hotel cap in London?" --score
   ```

2. Upload the eval set as a Langfuse **dataset** (`helix-helpdesk-rag`). You
   can run this again safely: items are upserted by id.

   ```bash
   python main.py seed
   ```

3. Run all four variants as **experiment runs** on that dataset:

   ```bash
   python main.py eval --variant all
   ```

4. In Langfuse, open **Datasets → helix-helpdesk-rag → Runs**. Select the runs
   and compare them. You'll see the average score per metric for each run.
   Click an item to see its scores side by side for each variant. Click a score
   to read its explanation, for example *"1/8 retrieved chunks come from gold
   docs"* or *"Unsupported: ['The 401(k) match is 4%']"*.

Each `eval` also prints a table in the terminal, with a row per item and
averages per category:

```
item                    category           context_pre context_rec faithfulnes answer_corr answer_rele   rag_score
q05-london-conference   multi_hop                 1.00        0.50        1.00        0.33        0.75        0.72
...
avg                     multi_hop                 1.00        0.50        ...
```

Useful flags:

```bash
python main.py eval --variant low_recall --local                 # no Langfuse needed; prints scores only
python main.py eval --variant no_guardrails --local --category out_of_scope
python main.py eval --variant baseline --limit 3                 # quick, cheap check
```

## Cost note

Each item makes about 2–3 agent calls plus 1 judge call. Running all four
variants on 14 items is about 200 Claude requests. Use `--limit` or `--local`
with a single variant for a quick rehearsal. To change the model, set
`AGENT_MODEL` or `JUDGE_MODEL` in `.env`. Server-side refusal fallback
(`fallbacks="default"`) is on by default. Set `ENABLE_FALLBACKS=false` to turn
it off.

## Tests

The tests run offline with a fake Claude client. They need no API keys:

```bash
python -m pytest -q
```

## Extending

- To use your own documents, put them in `knowledge_base/` as markdown files
  with `##` sections. Then add rows to `data/eval_dataset.json` with
  `gold_doc_ids` set to the file names without `.md`.
- To try another failure mode, add an entry to `VARIANTS` in
  `rag_agent/config.py` and pass its name with `--variant`.
- To use embeddings, replace `BM25Retriever`. The metrics only use the doc id
  part of each chunk id.
