"""Evaluate the frozen contexts with one user-selected OpenRouter model."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from pathlib import Path

import evaluate_context_sufficiency as evaluation
from dotenv import load_dotenv

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
PROVIDER = {"only": ["openai"], "allow_fallbacks": False, "require_parameters": True}


def payload(request):
    return {"model": "openai/gpt-4o", "temperature": 0, "seed": 2026, "max_tokens": 400,
            "response_format": {"type": "json_object"}, "provider": PROVIDER,
            "messages": evaluation.messages(request)}


def run(args):
    prepared = args.prepared.resolve()
    manifest = evaluation.read(prepared / "manifest.json")
    for name in ("requests.jsonl", "mapping.jsonl", "judge_prompt.txt"):
        if evaluation.sha(prepared / name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Frozen input changed: {name}")
    if (prepared / "judge_prompt.txt").read_text() != evaluation.PROMPT + "\n":
        raise ValueError("Frozen rubric changed")
    load_dotenv(args.env_file)
    key = os.environ["OPENROUTER_API_KEY"]
    output = evaluation.fresh(args.output)
    evaluation.write(output / "run_config.json", {
        "started_at_utc": datetime.now(UTC).isoformat(), "prepared": str(prepared),
        "endpoint": ENDPOINT, "model": "openai/gpt-4o", "temperature": 0, "seed": 2026,
        "max_tokens": 400, "response_format": {"type": "json_object"}, "provider": PROVIDER,
        "minimum_request_interval_seconds": args.interval,
        "transport_retries": "up_to_5_only_429_or_5xx; fail_on_explicit_zero_limit; no_retry_received_judgments",
        "input_sha256": {name: evaluation.sha(prepared / name) for name in (
            "requests.jsonl", "mapping.jsonl", "judge_prompt.txt")},
        "source_sha256": {str(p): evaluation.sha(p) for p in (
            Path(__file__).resolve(), Path(evaluation.__file__).resolve())},
        "python": platform.python_version(), "packages": {p: importlib.metadata.version(p) for p in (
            "python-dotenv", "numpy", "pyarrow", "tokenizers")},
        "command": sys.argv, "credentials": "external temporary file; never saved in artifacts"})
    count, resolved_model, last_start = 0, None, 0
    requests = evaluation.rows(prepared / "requests.jsonl")
    try:
        with (output / "responses.jsonl").open("x", encoding="utf-8") as raw, \
                (output / "judgments.jsonl").open("x", encoding="utf-8") as scored, \
                (output / "transport_events.jsonl").open("x", encoding="utf-8") as events:
            for request in requests:
                body = json.dumps(payload(request), ensure_ascii=False).encode()
                for attempt in range(6):
                    time.sleep(max(0, last_start + args.interval - time.monotonic()))
                    last_start = time.monotonic()
                    call = urllib.request.Request(ENDPOINT, data=body, headers={
                        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
                    try:
                        with urllib.request.urlopen(call, timeout=120) as response:
                            value = json.load(response)
                        break
                    except urllib.error.HTTPError as error:
                        limit_headers = {k: v for k, v in error.headers.items()
                                         if "ratelimit" in k.lower() or k.lower() == "retry-after"}
                        events.write(json.dumps({"request_id": request["request_id"], "attempt": attempt + 1,
                            "http_status": error.code, "body": error.read().decode()[:2000].replace(key, "[REDACTED]"),
                            "limit_headers": limit_headers, "time_utc": datetime.now(UTC).isoformat()}) + "\n")
                        events.flush()
                        zero = any("limit-req" in k.lower() and str(v).strip() == "0" for k, v in limit_headers.items())
                        if attempt == 5 or zero or (error.code != 429 and not 500 <= error.code < 600):
                            raise RuntimeError(f"OpenRouter HTTP {error.code}; no substituted judgments") from None
                        time.sleep(min(30, 2 ** (attempt + 1)))
                raw.write(json.dumps({"request_id": request["request_id"], "response": value}, ensure_ascii=False) + "\n")
                raw.flush()
                if value.get("error"):
                    raise ValueError("API returned an error inside HTTP 200")
                if not value.get("model") or (resolved_model is not None and value["model"] != resolved_model):
                    raise ValueError("Missing or changing model identifier")
                resolved_model = value["model"]
                if value.get("provider", "OpenAI").lower() != "openai":
                    raise ValueError("Unexpected judge provider")
                choice = value["choices"][0]
                if choice["finish_reason"] != "stop":
                    raise ValueError("Truncated or incomplete judge output")
                verdict = evaluation.validate_judgment(json.loads(choice["message"]["content"]))
                scored.write(json.dumps({"request_id": request["request_id"], **verdict}, ensure_ascii=False) + "\n")
                scored.flush()
                count += 1
                if count == 1 or count % 10 == 0 or count == len(requests):
                    print(json.dumps({"judged": count, "total": len(requests), "model": resolved_model}), flush=True)
        evaluation.analyze(prepared, output)
    except Exception as error:
        evaluation.write(output / "failure.json", {"status": "incomplete_no_substitutions", "judged": count,
            "error_type": type(error).__name__, "returned_model": resolved_model,
            "finished_at_utc": datetime.now(UTC).isoformat()})
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--interval", type=float, default=1.2)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
