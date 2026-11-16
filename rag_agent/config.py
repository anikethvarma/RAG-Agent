"""Configuration: OpenRouter model, retrieval settings and the system prompt."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = ROOT / ".env"


def _load_env_file(path: Path) -> None:
    """Load .env, also when Windows saved it as UTF-16 (e.g. PowerShell `echo ... > .env`)."""
    if not path.is_file():
        return
    head = path.read_bytes()[:4]
    if head.startswith((b"\xff\xfe", b"\xfe\xff")):
        encoding = "utf-16"
    else:
        encoding = "utf-8-sig"  # also strips a UTF-8 byte-order mark (Notepad)
    load_dotenv(path, encoding=encoding)


_load_env_file(ENV_FILE)

# Langfuse tracing is optional. Without keys, turn it off explicitly so the SDK
# doesn't log an authentication error on every run. The app reports the status at
# startup (see rag_agent/tracing.py), so this never fails silently.
LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY") and os.getenv("LANGFUSE_SECRET_KEY"))
if not LANGFUSE_ENABLED:
    os.environ.setdefault("LANGFUSE_TRACING_ENABLED", "false")
    from langfuse.logger import langfuse_logger  # imported after load_dotenv(), as Langfuse requires

    langfuse_logger.setLevel(logging.ERROR)  # hide the expected "no public_key" warning

KB_DIR = Path(os.getenv("KB_DIR", ROOT / "knowledge_base"))

OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
# Any free OpenRouter model (ids ending in ":free"). Run `python main.py --models`
# to see which free models are available right now and which support tool calling.
MODEL = os.getenv("MODEL", "meta-llama/llama-3.3-70b-instruct:free")

# "agent"  = the model calls a search tool itself (needs a model with tool support)
# "simple" = retrieve first, then ask the model (works with every model)
# The agent mode falls back to simple automatically if the model rejects tools.
MODE = os.getenv("MODE", "agent")

TOP_K = int(os.getenv("TOP_K", "3"))
MAX_SEARCHES = int(os.getenv("MAX_SEARCHES", "3"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))

SYSTEM_PROMPT = """You are the Helix Cloud internal helpdesk assistant.
Answer employee questions about company policy using ONLY the knowledge base.

- Always call `search_knowledge_base` before answering. Search again with a
  different query if the question has several parts or the first results don't
  cover everything.
- Base every statement on retrieved text. If documents conflict, prefer the one
  marked CURRENT and ignore SUPERSEDED/ARCHIVED ones.
- If the knowledge base doesn't contain the answer, say so plainly. Do not guess.
- Keep the answer short (2-5 sentences) and cite the source doc ids in brackets,
  e.g. [pto_policy]."""

SIMPLE_PROMPT = """You are the Helix Cloud internal helpdesk assistant.
Answer the employee's question using ONLY the documents below.

- If documents conflict, prefer the one marked CURRENT and ignore SUPERSEDED/ARCHIVED ones.
- If the documents don't contain the answer, say so plainly. Do not guess.
- Keep the answer short (2-5 sentences) and cite the source doc ids in brackets,
  e.g. [pto_policy].

<documents>
{context}
</documents>"""
