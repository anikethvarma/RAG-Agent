"""Configuration: model, retrieval settings and the system prompt."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

KB_DIR = Path(os.getenv("KB_DIR", ROOT / "knowledge_base"))

MODEL = os.getenv("MODEL", "claude-opus-5-5")
EFFORT = os.getenv("EFFORT", "medium")
TOP_K = int(os.getenv("TOP_K", "3"))
MAX_SEARCHES = int(os.getenv("MAX_SEARCHES", "3"))

# Server-side refusal fallback (re-runs a declined request on a fallback model
# inside the same API call). Set ENABLE_FALLBACKS=false to turn it off.
ENABLE_FALLBACKS = os.getenv("ENABLE_FALLBACKS", "true").lower() != "false"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

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
