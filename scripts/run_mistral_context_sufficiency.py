"""Run the frozen blind context rubric through the user's Mistral API."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import platform
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import evaluate_context_sufficiency as evaluation
from dotenv import load_dotenv
from mistralai.client import Mistral
from mistralai.client.errors import MistralError


def retryable_error(code, headers):
    if code == 429 and any(k.lower().startswith("x-ratelimit-limit-req") and str(v).strip() == "0"
                           for k, v in headers.items()):
        return False
    return code == 429 or (isinstance(code, int) and 500 <= code < 600)


def run(args):
    prepared = args.prepared.resolve()
    manifest = evaluation.read(prepared / "manifest.json")
    for name in ("requests.jsonl", "mapping.jsonl", "judge_prompt.txt"):
        if evaluation.sha(prepared / name) != manifest["artifact_sha256"][name]:
            raise ValueError(f"Frozen input changed: {name}")
    if (prepared / "judge_prompt.txt").read_text() != evaluation.PROMPT + "\n":
        raise ValueError("Frozen rubric changed")
    load_dotenv(args.env_file, override=False)
    if not os.environ.get("MISTRAL_API_KEY"):
        raise ValueError("MISTRAL_API_KEY missing")
    output = evaluation.fresh(args.output)
    evaluation.write(output / "run_config.json", {
        "status": "frozen_before_first_judgment", "started_at_utc": datetime.now(UTC).isoformat(),
        "prepared": str(prepared), "prepared_manifest_sha256": evaluation.sha(prepared / "manifest.json"),
        "input_sha256": {name: evaluation.sha(prepared / name) for name in (
            "requests.jsonl", "mapping.jsonl", "judge_prompt.txt")},
        "source_sha256": {str(p): evaluation.sha(p) for p in (
            Path(__file__).resolve(), Path(evaluation.__file__).resolve())},
        "sdk": "mistralai", "endpoint": "https://api.mistral.ai/v1/chat/completions",
        "requested_model": args.model, "temperature": 0, "random_seed": 2026,
        "max_tokens": 400, "response_format": {"type": "json_object"},
        "minimum_request_interval_seconds": args.interval,
        "sdk_retries": "disabled", "transport_retries": "up_to_5_only_HTTP_429_or_5xx; no retry of received judgments",
        "python": platform.python_version(), "command": sys.argv,
        "packages": {p: importlib.metadata.version(p) for p in (
            "mistralai", "python-dotenv", "pyarrow", "numpy", "tokenizers")},
        "credentials": "MISTRAL_API_KEY loaded from external temporary file; never saved in artifacts",
    })
    resolved_model, last_start, count = None, 0, 0
    requests = evaluation.rows(prepared / "requests.jsonl")
    client = Mistral(api_key=os.environ["MISTRAL_API_KEY"], retry_config=None, timeout_ms=120000)
    try:
        with (output / "responses.jsonl").open("x", encoding="utf-8") as raw, \
                (output / "judgments.jsonl").open("x", encoding="utf-8") as scored, \
                (output / "transport_events.jsonl").open("x", encoding="utf-8") as events:
            for request in requests:
                for attempt in range(6):
                    time.sleep(max(0, last_start + args.interval - time.monotonic()))
                    last_start = time.monotonic()
                    try:
                        response = client.chat.complete(
                            model=args.model, messages=evaluation.messages(request), temperature=0,
                            random_seed=2026, max_tokens=400, response_format={"type": "json_object"},
                            retries=None,
                        )
                        break
                    except MistralError as error:
                        code = getattr(error, "status_code", None)
                        events.write(json.dumps({"request_id": request["request_id"], "attempt": attempt + 1,
                            "status_code": code, "exception_type": type(error).__name__,
                            "response_body": str(getattr(error, "body", ""))[:2000].replace(os.environ["MISTRAL_API_KEY"], "[REDACTED]"),
                            "limit_headers": {k: v for k, v in getattr(error, "headers", {}).items()
                                              if "ratelimit" in k.lower() or k.lower() == "retry-after"},
                            "time_utc": datetime.now(UTC).isoformat()}) + "\n")
                        events.flush()
                        if not retryable_error(code, getattr(error, "headers", {})):
                            raise RuntimeError(f"Mistral transport failed: {type(error).__name__}, status={code}") from None
                        if attempt == 5:
                            raise RuntimeError(f"Mistral transport retries exhausted, status={code}") from None
                        time.sleep(min(30, 2 ** (attempt + 1)))
                value = response.model_dump(mode="json")
                raw.write(json.dumps({"request_id": request["request_id"], "response": value}, ensure_ascii=False) + "\n")
                raw.flush()
                if not value.get("model"):
                    raise ValueError("Response does not identify model")
                if resolved_model is not None and value["model"] != resolved_model:
                    raise ValueError("Returned model changed within run")
                resolved_model = value["model"]
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
        evaluation.write(output / "mistral_execution.json", {"status": "completed_judging_pending_content_review",
            "finished_at_utc": datetime.now(UTC).isoformat(), "returned_model": resolved_model,
            "judged": count, "total": len(requests)})
    except Exception as error:
        evaluation.write(output / "failure.json", {"status": "incomplete_no_substitutions", "judged": count,
            "error_type": type(error).__name__, "returned_model": resolved_model,
            "finished_at_utc": datetime.now(UTC).isoformat()})
        raise
    finally:
        client.__exit__(None, None, None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--env-file", required=True, type=Path)
    parser.add_argument("--model", default="mistral-large-latest")
    parser.add_argument("--interval", default=1.2, type=float)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
