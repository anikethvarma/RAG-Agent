"""Langfuse tracing status and a connection check, so tracing never fails silently."""

from __future__ import annotations

import os
from dataclasses import dataclass
from importlib.metadata import version

from .config import ENV_FILE, LANGFUSE_ENABLED

DEFAULT_BASE_URL = "https://cloud.langfuse.com"  # the SDK's default when no URL is set


def _mask(key: str | None) -> str:
    return f"{key[:10]}…{key[-4:]}" if key and len(key) > 16 else ("(set)" if key else "(missing)")


@dataclass
class TracingStatus:
    enabled: bool
    base_url: str
    public_key: str
    secret_key: str
    environment: str
    env_file: str
    sdk_version: str

    def summary(self) -> str:
        if self.enabled:
            return (f"Langfuse tracing ON → {self.base_url} (public key {self.public_key}, "
                    f"environment '{self.environment}')")
        return (f"Langfuse tracing OFF: LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY not found "
                f"(looked in {self.env_file} and the shell environment)")


def tracing_status() -> TracingStatus:
    return TracingStatus(
        enabled=LANGFUSE_ENABLED,
        base_url=os.getenv("LANGFUSE_BASE_URL") or os.getenv("LANGFUSE_HOST") or f"{DEFAULT_BASE_URL} (default)",
        public_key=_mask(os.getenv("LANGFUSE_PUBLIC_KEY")),
        secret_key=_mask(os.getenv("LANGFUSE_SECRET_KEY")),
        environment=os.getenv("LANGFUSE_TRACING_ENVIRONMENT") or "default",
        env_file=f"{ENV_FILE} ({'found' if ENV_FILE.is_file() else 'NOT found'})",
        sdk_version=version("langfuse"),
    )


def check_tracing() -> tuple[bool, str]:
    """Verify credentials and send one test trace. Returns (ok, message incl. trace URL)."""
    status = tracing_status()
    if not status.enabled:
        return False, status.summary()

    from langfuse import get_client

    lf = get_client()
    try:
        if not lf.auth_check():
            return False, "Langfuse rejected the keys."
    except Exception as err:  # wrong keys, wrong region/host, or network blocked
        body = getattr(err, "body", None)
        detail = body.get("message") if isinstance(body, dict) and body.get("message") else str(err)[:300]
        return False, (
            f"Could not authenticate with {status.base_url}: {err.__class__.__name__}: {detail}\n"
            "Check that the keys belong to a project in this region. EU: https://cloud.langfuse.com, "
            "US: https://us.cloud.langfuse.com (set LANGFUSE_BASE_URL)."
        )

    with lf.start_as_current_observation(as_type="span", name="tracing-check",
                                         input="connection test", output="ok") as span:
        trace_id = span.trace_id
    lf.flush()
    try:
        url = lf.get_trace_url(trace_id=trace_id)
    except Exception:
        url = None
    return True, (f"Credentials OK and a test trace 'tracing-check' was sent to {status.base_url}."
                  + (f"\nOpen it: {url}" if url else f"\nTrace id: {trace_id}")
                  + f"\nIn Langfuse, make sure the environment filter includes '{status.environment}'.")


def trace_url(trace_id: str | None) -> str | None:
    """Link to a trace in the Langfuse UI (None when tracing is off or the URL can't be built)."""
    if not (LANGFUSE_ENABLED and trace_id):
        return None
    from langfuse import get_client

    try:
        return get_client().get_trace_url(trace_id=trace_id)
    except Exception:
        return None
