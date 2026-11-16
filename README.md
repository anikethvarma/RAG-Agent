# RAG Agent

A simple retrieval-augmented agent built on Claude. It answers employee
questions for a fictional company, **Helix Cloud**, using a small set of
internal policy documents.

## How it works

```
question ─► Claude ──tool_use──► search_knowledge_base (BM25, top_k)
              ▲                          │
              └──── tool_result ◄────────┘
                 ...loop until it answers...
```

Claude decides what to search for, can search more than once for multi-part
questions, and answers only from what it retrieved, citing the source doc ids.
If the documents don't cover the question, it says so.

| Path | What it is |
|---|---|
| `knowledge_base/*.md` | 11 policy documents (each `## section` is one chunk) |
| `rag_agent/retriever.py` | Dependency-free BM25 retriever |
| `rag_agent/agent.py` | Claude tool-use loop |
| `rag_agent/config.py` | Model, retrieval settings, system prompt |
| `main.py` | CLI: one question, or interactive chat |

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

Then set `ANTHROPIC_API_KEY` in `.env`.

## Usage

Ask one question:

```bash
python main.py "What is the hotel cap in London?"
```

Or start an interactive chat:

```bash
python main.py
```

Example questions:

- How many PTO days do I get, and how do they accrue?
- I lost my laptop on a trip. Who do I tell and how quickly?
- What's the per diem, and can I fly business class on a 14-hour flight?
- What is the 401(k) match? *(not in the documents — the agent should say so)*

## Configuration

Set these in `.env` (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `claude-opus-5-5` | Claude model |
| `EFFORT` | `medium` | `low` / `medium` / `high` / `xhigh` / `max` |
| `TOP_K` | `3` | chunks returned per search |
| `MAX_SEARCHES` | `3` | searches allowed per question |
| `KB_DIR` | `knowledge_base/` | folder of markdown documents |
| `ENABLE_FALLBACKS` | `true` | server-side refusal fallback |

## Using your own documents

Put markdown files in `knowledge_base/` (or point `KB_DIR` elsewhere), with
`##` headings to split sections, and update `SYSTEM_PROMPT` in
`rag_agent/config.py` to describe them.

## Tests

Offline, with a fake Claude client — no API key needed:

```bash
python -m pytest -q
```
