"""OpenRouter payload and temporary-budget policy without network calls."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_openrouter_context_sufficiency import payload, temporary_budget_error


def test_payload_is_blind_and_fixed_to_one_provider():
    value = payload({"question": "Q", "context": "C", "reference_answer": "SECRET", "variant": "fixed"})
    assert json.loads(value["messages"][1]["content"]) == {"question": "Q", "context": "C"}
    assert value["temperature"] == 0
    assert value["provider"] == {"only": ["openai"], "allow_fallbacks": False, "require_parameters": True}
    assert "SECRET" not in json.dumps(value)


def test_only_explicit_inflight_402_is_temporary():
    assert temporary_budget_error(402, {"error": {"metadata": {"reason": "in_flight_budget_exhausted"}}})
    assert not temporary_budget_error(402, {"error": {"metadata": {"reason": "insufficient_credits"}}})
    assert not temporary_budget_error(402, {})
