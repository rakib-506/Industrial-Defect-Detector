"""Server entry point: `python -m src.api`.

Hosting platforms assign the port at runtime rather than letting the image pick
one. Render (and Heroku, Cloud Run, Fly and others) pass it in `PORT`; Hugging
Face Spaces instead expects the port declared in the README to be the one bound.
Reading `PORT` with a sensible default satisfies both, so the same image runs on
either without a rebuild.
"""

from __future__ import annotations

import os

import uvicorn

# Used when nothing sets PORT: local runs, and platforms that expect the image to
# choose (Hugging Face Spaces, whose app_port is 7860).
DEFAULT_PORT = 7860


def resolve_port(raw: str | None = None) -> int:
    """Port to bind, from `PORT`, falling back to `DEFAULT_PORT`.

    An unset, empty or non-numeric value falls back rather than crashing: a
    container that refuses to start is much harder to diagnose on a hosted
    platform than one that starts on an unexpected port and says so.
    """
    if raw is None:
        raw = os.environ.get("PORT", "")
    raw = raw.strip()
    if not raw:
        return DEFAULT_PORT
    try:
        port = int(raw)
    except ValueError:
        print(f"[api] PORT={raw!r} is not a number; using {DEFAULT_PORT}")
        return DEFAULT_PORT
    if not (1 <= port <= 65535):
        print(f"[api] PORT={port} is out of range; using {DEFAULT_PORT}")
        return DEFAULT_PORT
    return port


def main() -> None:
    port = resolve_port()
    # 0.0.0.0, not localhost: the platform's router reaches the container from
    # outside its network namespace.
    host = os.environ.get("HOST", "0.0.0.0")  # noqa: S104 - required in a container
    print(f"[api] binding {host}:{port}")
    uvicorn.run(
        "src.api.main:app",
        host=host,
        port=port,
        # One worker: each would load its own copy of the memory banks, and the
        # backbone plus banks already dominates memory on a small instance.
        workers=1,
        log_level=os.environ.get("LOG_LEVEL", "info"),
    )


if __name__ == "__main__":
    main()
