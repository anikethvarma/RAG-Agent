# RAG Agent (free OpenRouter model)

A simple retrieval-augmented agent that runs on a **free LLM from
[OpenRouter](https://openrouter.ai)**. It answers employee questions for a
fictional company, **Helix Cloud**, using a small set of internal policy
documents.

## How it works

```
question ─► free LLM (OpenRouter) ──tool call──► search_knowledge_base (BM25, top_k)
                 ▲                                      │
                 └──────────── tool result ◄────────────┘
                       ...loop until it answers...
```

The model decides what to search for, can search more than once for
multi-part questions, and answers only from what it retrieved, citing the
source doc ids. If the documents don't cover the question, it says so.

There are two modes:

| Mode | How it works | Needs a model with tool calling? |
|---|---|---|
| `agent` (default) | The model calls the search tool itself, as often as it needs | Yes |
| `simple` | The question is used to retrieve documents, then the model answers in one call | No |

If the chosen model doesn't support tool calling, the agent switches to
`simple` mode automatically.

| Path | What it is |
|---|---|
| `knowledge_base/*.md` | 11 policy documents (each `## section` is one chunk) |
| `rag_agent/retriever.py` | Dependency-free BM25 retriever |
| `rag_agent/agent.py` | Agent loop using the `openai` SDK pointed at OpenRouter |
| `rag_agent/config.py` | Model, mode, retrieval settings, prompts |
| `main.py` | CLI: one question, interactive chat, or list free models |
| `test_questions.xlsx` | 32 test questions with expected answers |

## Setup

1. Create a free API key at https://openrouter.ai/keys.

2. Install:

   ```bash
   python -m venv .venv && source .venv/bin/activate
   ```

   ```bash
   pip install -r requirements.txt
   ```

3. Configure:

   ```bash
   cp .env.example .env
   ```

   Then set `OPENROUTER_API_KEY` in `.env`.

## Usage

See which free models are available right now (tool-capable ones are listed first):

```bash
python main.py --models
```

Ask one question:

```bash
python main.py "What is the hotel cap in London?"
```

Or start an interactive chat:

```bash
python main.py
```

To use a different free model, set `MODEL` in `.env` to an id from
`--models`, preferably one with `tools = yes`.

## Configuration

Set these in `.env` (all optional except the key):

| Variable | Default | Meaning |
|---|---|---|
| `OPENROUTER_API_KEY` | — | your OpenRouter key (required) |
| `MODEL` | `meta-llama/llama-3.3-70b-instruct:free` | any free OpenRouter model id |
| `MODE` | `agent` | `agent` or `simple` |
| `TOP_K` | `3` | chunks returned per search |
| `MAX_SEARCHES` | `3` | searches allowed per question (agent mode) |
| `TEMPERATURE` | `0.1` | sampling temperature |
| `KB_DIR` | `knowledge_base/` | folder of markdown documents |

## Notes on free models

- Free models on OpenRouter have rate limits (requests per minute and per day).
  The client retries rate-limited requests a few times; if you still hit limits,
  wait a bit or switch models.
- The list of free models changes over time. If the default model is no longer
  available, pick another from `python main.py --models`.
- Smaller free models follow instructions less reliably. If answers ignore the
  documents, try a larger model or `MODE=simple`.

## Testing

Use `test_questions.xlsx` for manual testing: run each question, paste the
answer, and mark Pass / Partial / Fail. The Summary tab scores the results.

Automated tests run offline with a fake client (no key needed):

```bash
python -m pytest -q
```

## Using your own documents

Put markdown files in `knowledge_base/` (or point `KB_DIR` elsewhere), with
`##` headings to split sections, and update the prompts in
`rag_agent/config.py` to describe them.
