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
| `app.py` | Chat UI (Streamlit) |
| `main.py` | CLI: one question, interactive chat, or list free models |
| `.claude/skills/langfuse/` | Langfuse agent skill (from github.com/langfuse/skills) |
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

## Chat UI

```bash
streamlit run app.py
```

This opens a chat page in your browser (http://localhost:8501). It has:

- a chat with memory, so follow-up questions work ("And in London?")
- a sidebar to pick the free model (🔧 marks models that support tool calling),
  the mode, and the search settings
- a **Tools the agent can use** panel
- under each answer, an expandable list of every tool call the agent made: the
  search query, the text it got back, the source documents and token counts

If `OPENROUTER_API_KEY` isn't in `.env`, the sidebar asks for it.

## Tools the agent can use

The agent has one tool:

| Tool | Input | What it does |
|---|---|---|
| `search_knowledge_base` | `query` (text) | Keyword (BM25) search over the documents in `knowledge_base/`. Returns the `TOP_K` best-matching sections, each labelled with its doc id. |

The model decides when to call it and with what query, and can call it up to
`MAX_SEARCHES` times per question. It has no other access: no web, no files
outside the knowledge base, no code execution. In `simple` mode the model
calls no tools; the search runs automatically with the question.

To add a tool, define it next to `SEARCH_TOOL` in `rag_agent/agent.py`, add it
to the `tools` list in `_complete`, and handle its name in `_run_agent`.

## Command line

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

## Tracing with Langfuse (optional)

Every question is traced to [Langfuse](https://langfuse.com) when keys are set.
Without keys, tracing is off and the app works as before.

1. Create a free account at https://langfuse.com/cloud (or self-host), create a
   project, and copy its API keys from **Settings → API Keys**.
2. Add them to `.env`:

   ```
   LANGFUSE_PUBLIC_KEY=pk-lf-...
   LANGFUSE_SECRET_KEY=sk-lf-...
   LANGFUSE_BASE_URL=https://cloud.langfuse.com
   LANGFUSE_TRACING_ENVIRONMENT=development
   ```

   Use `https://us.cloud.langfuse.com` for the US region.

3. Ask a question in the UI or CLI, then open **Tracing** in Langfuse.

What each trace looks like (one trace per question / chat turn):

```
answer-question          agent       input: the user's question · output: the answer
├─ generate-response     generation  messages + tool definitions in; reply, tool calls and reasoning out;
│                                    model, input/output tokens, cost (from OpenRouter)
├─ search-knowledge-base retriever   input: the search query · output: retrieved chunks (doc, score, text)
├─ generate-response     generation
└─ ...
```

- **Sessions:** all turns of one chat share a `session_id` (one per UI
  conversation, cleared with "Clear chat"; one per interactive CLI run), so
  **Sessions** in Langfuse replays whole conversations.
- **Tags:** `ui` or `cli`, to filter by where a question came from.
- **Environment:** set with `LANGFUSE_TRACING_ENVIRONMENT`, so local test
  traces stay apart from production.
- **Metadata on the root:** model, mode, `top_k`, searches made, sources used,
  token totals, and `fell_back_to_simple` when a model rejected tool calls.
- **Errors:** a failed model call is recorded as an `ERROR` generation with the
  error message.

### Not seeing traces?

Run the connection check:

```bash
python main.py --check-tracing
```

It reports whether the keys were found (and in which `.env`), checks them
against your Langfuse host, sends a test trace named `tracing-check`, and prints
a link to it. The CLI also prints whether tracing is on at startup, plus a trace
link after each answer. The UI shows the same status, a **Test Langfuse
connection** button in the sidebar, and a **View trace in Langfuse** link under
each answer.

Common causes:

| Symptom | Fix |
|---|---|
| "tracing OFF: … not found" | The keys aren't in `.env` next to `main.py`, or the lines still start with `#`. |
| `401 Invalid credentials. Confirm that you've configured the correct host.` | The keys are from another region. Set `LANGFUSE_BASE_URL=https://us.cloud.langfuse.com` for US projects (EU is the default). |
| The check passes but you don't see traces | Make sure you're in the project the keys belong to, and that the **Environment** filter in Langfuse includes your `LANGFUSE_TRACING_ENVIRONMENT` (or `default`). |
| You edited `.env` but nothing changed | Restart `streamlit run app.py`; `.env` is read at startup. |

Names are kept stable (`answer-question`, `generate-response`,
`search-knowledge-base`) because Langfuse evaluators and dashboards select
observations by name.

The Langfuse agent skill (from https://github.com/langfuse/skills) is installed
in `.claude/skills/langfuse`, so Claude Code sessions in this repo can use it
for further Langfuse work: evaluations, datasets, prompt management.

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
