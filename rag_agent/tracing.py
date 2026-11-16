"""Thin wrapper so the agent runs with or without Langfuse credentials."""

from __future__ import annotations

import contextlib
from typing import Any

from .config import langfuse_configured


class _NoopObservation:
    def update(self, **_: Any) -> None:  # mirrors LangfuseSpan.update
        pass

    def score(self, **_: Any) -> None:
        pass

    def score_trace(self, **_: Any) -> None:
        pass


_client = None


def get_langfuse():
    """Return the Langfuse client, or None when keys aren't configured."""
    global _client
    if _client is None and langfuse_configured():
        from langfuse import get_client

        _client = get_client()
    return _client


def observation(name: str, as_type: str = "span", **kwargs: Any):
    """Context manager yielding a Langfuse observation (or a no-op stand-in)."""
    lf = get_langfuse()
    if lf is None:
        return contextlib.nullcontext(_NoopObservation())
    return lf.start_as_current_observation(name=name, as_type=as_type, **kwargs)


def to_jsonable(obj: Any) -> Any:
    """Convert SDK content blocks / nested structures into plain JSON for trace payloads."""
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
