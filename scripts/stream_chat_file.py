#!/usr/bin/env python3
"""Stream a text file through an OpenAI-compatible chat-completions server."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", help="Text file to send as the user message.")
    parser.add_argument("--host", default="127.0.0.1", help="Server host. Default: 127.0.0.1")
    parser.add_argument("--port", type=int, default=8000, help="Server port. Default: 8000")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Full OpenAI-compatible base URL. Overrides --host/--port, e.g. http://127.0.0.1:8000/v1",
    )
    parser.add_argument("--model", default=None, help="Model id. Defaults to the first model from /v1/models.")
    parser.add_argument("--api-key", default=None, help="API key. Defaults to OPENAI_API_KEY, then EMPTY.")
    parser.add_argument("--system", default=None, help="Optional system message.")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument(
        "--no-labels",
        action="store_true",
        help="Do not print [thinking]/[content] markers when the stream switches fields.",
    )
    return parser.parse_args()


def import_openai():
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit(
            "error: missing OpenAI SDK. Install it with `python -m pip install openai`."
        ) from exc
    return OpenAI


def chunk_to_dict(obj: Any) -> dict[str, Any]:
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return {}


def delta_value(delta: Any, name: str) -> str:
    if isinstance(delta, dict):
        value = delta.get(name)
    else:
        value = getattr(delta, name, None)
    return value if isinstance(value, str) else ""


def choice_delta(chunk: Any) -> Any:
    choices = getattr(chunk, "choices", None)
    if choices is None and isinstance(chunk, dict):
        choices = chunk.get("choices")
    if not choices:
        return None

    choice = choices[0]
    if isinstance(choice, dict):
        return choice.get("delta") or {}
    return getattr(choice, "delta", None)


def text_fields(delta: Any) -> tuple[str, str]:
    thinking = (
        delta_value(delta, "reasoning_content")
        or delta_value(delta, "reasoning")
        or delta_value(delta, "thinking")
    )
    content = delta_value(delta, "content")
    return thinking, content


def print_piece(label: str, text: str, *, state: dict[str, str], no_labels: bool) -> None:
    if not text:
        return
    if not no_labels and state.get("label") != label:
        prefix = "\n" if state.get("started") else ""
        print(f"{prefix}[{label}]", flush=True)
        state["label"] = label
        state["started"] = "1"
    print(text, end="", flush=True)


def resolve_model(client: Any) -> str:
    models = client.models.list()
    data = getattr(models, "data", None)
    if not data and isinstance(models, dict):
        data = models.get("data")
    if not data:
        raise RuntimeError("/v1/models returned no models")
    first = data[0]
    model_id = first.get("id") if isinstance(first, dict) else getattr(first, "id", None)
    if not model_id:
        raise RuntimeError("/v1/models returned a model without an id")
    return model_id


def main() -> None:
    args = parse_args()
    OpenAI = import_openai()

    path = Path(args.file)
    prompt = path.read_text(encoding="utf-8")

    base_url = args.base_url or f"http://{args.host}:{args.port}/v1"
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or "EMPTY"
    client = OpenAI(base_url=base_url, api_key=api_key, timeout=args.timeout)

    model = args.model or resolve_model(client)
    messages: list[dict[str, str]] = []
    if args.system is not None:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": prompt})

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        reasoning_effort="high",
    )

    state: dict[str, str] = {}
    for chunk in stream:
        delta = choice_delta(chunk)
        if delta is None:
            continue

        thinking, content = text_fields(delta)
        if thinking:
            print_piece("thinking", thinking, state=state, no_labels=args.no_labels)
        if content:
            print_piece("content", content, state=state, no_labels=args.no_labels)

        # Some OpenAI-compatible servers put vendor-specific fields in the raw
        # serialized chunk even when the SDK object has no typed attribute.
        raw_delta = choice_delta(chunk_to_dict(chunk))
        if raw_delta and raw_delta is not delta:
            raw_thinking, raw_content = text_fields(raw_delta)
            if raw_thinking and raw_thinking != thinking:
                print_piece("thinking", raw_thinking, state=state, no_labels=args.no_labels)
            if raw_content and raw_content != content:
                print_piece("content", raw_content, state=state, no_labels=args.no_labels)

    if state.get("started"):
        print()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(1)
