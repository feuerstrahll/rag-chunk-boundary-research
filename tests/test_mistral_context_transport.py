"""Transport-only fixtures, no API calls or experimental judgments."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_mistral_context_sufficiency import retryable_error


@pytest.mark.parametrize("code,headers,expected", [
    (429, {"x-ratelimit-limit-req-minute": "0"}, False),
    (429, {"x-ratelimit-limit-req-minute": "60", "x-ratelimit-remaining-req-minute": "0"}, True),
    (429, {}, True),
    (403, {}, False),
    (500, {}, True),
])
def test_retry_policy(code, headers, expected):
    assert retryable_error(code, headers) is expected
