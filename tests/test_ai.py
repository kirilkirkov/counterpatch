from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from counterpatch.ai import AIError, parse_suggestions, request_suggestions, require_api_key, validate_test_code
from counterpatch.analyzer import collect_functions
from counterpatch.candidates import CallPlan, build_call_plan
from counterpatch.check import CheckOptions, run_check
from counterpatch.models import ChangedFunction, ConfigError, Severity
from tests.conftest import WITHDRAW_BASE, WITHDRAW_BUGGY

GOOD_TEST = """
import pytest
from banking import withdraw


def test_negative_amount_rejected():
    with pytest.raises(ValueError):
        withdraw(5, -1)
"""


class FakeMessages:
    def __init__(self, payload: Any, stop_reason: str = "end_turn") -> None:
        self.payload = payload
        self.stop_reason = stop_reason
        self.requests: list[dict[str, Any]] = []

    def create(self, **request: Any) -> SimpleNamespace:
        self.requests.append(request)
        text = self.payload if isinstance(self.payload, str) else json.dumps(self.payload)
        return SimpleNamespace(stop_reason=self.stop_reason, content=[SimpleNamespace(type="text", text=text)])


def fake_client(payload: Any, stop_reason: str = "end_turn") -> SimpleNamespace:
    return SimpleNamespace(beta=SimpleNamespace(messages=FakeMessages(payload, stop_reason)))


@pytest.mark.parametrize(
    ("code", "problem"),
    [
        ("import os\n\ndef test_x():\n    os.remove('x')\n", "import of 'os'"),
        ("import subprocess\n\ndef test_x():\n    pass\n", "import of 'subprocess'"),
        ("def test_x():\n    eval('1')\n", "eval"),
        ("def test_x():\n    open('/etc/passwd')\n", "open"),
        ("def test_x():\n    ().__class__.__bases__\n", "access to"),
        ("print('side effect')\n\ndef test_x():\n    pass\n", "top-level statement"),
        ("def helper():\n    pass\n", "no test function"),
        ("def test_x(:\n", "syntax error"),
        ("from . import thing\n\ndef test_x():\n    pass\n", "not allowed"),
    ],
)
def test_validate_test_code_rejects_dangerous_or_malformed_code(code: str, problem: str) -> None:
    reason = validate_test_code(code, {"banking"})
    assert reason is not None and problem in reason


def test_validate_test_code_accepts_plain_pytest() -> None:
    assert validate_test_code(GOOD_TEST, {"banking"}) is None


def _plan() -> CallPlan:
    changed = ChangedFunction(
        file="banking.py",
        base_file="banking.py",
        qualified_name="withdraw",
        status="modified",
        diff="",
        patch=collect_functions(WITHDRAW_BUGGY)["withdraw"],
        base=collect_functions(WITHDRAW_BASE)["withdraw"],
    )
    plan = build_call_plan(changed)
    assert isinstance(plan, CallPlan)
    return plan


def test_parse_suggestions_validates_untrusted_output() -> None:
    data = {
        "inputs": [
            {
                "function": "withdraw",
                "arguments": [{"name": "balance", "value": "0"}, {"name": "amount", "value": "-5"}],
                "rationale": "neg",
                "targets_intended_change": False,
            },
            {
                "function": "withdraw",
                "arguments": [{"name": "amount", "value": "__import__('os')"}],
                "rationale": "evil",
                "targets_intended_change": False,
            },
            {
                "function": "withdraw",
                "arguments": [{"name": "amount", "value": "'text'"}],
                "rationale": "type",
                "targets_intended_change": False,
            },
            {"function": "os.system", "arguments": [], "rationale": "x", "targets_intended_change": False},
            {"function": "withdraw", "arguments": [{"name": "../path", "value": "1"}], "rationale": "x", "targets_intended_change": False},
        ],
        "tests": [
            {"name": "../../outside dir", "rationale": "r", "code": GOOD_TEST, "targets_intended_change": False},
            {"name": "bad", "rationale": "r", "code": "import os\n\ndef test_x():\n    pass\n", "targets_intended_change": False},
        ],
    }
    suggestions = parse_suggestions(data, {"withdraw": _plan()}, {"banking"})
    assert [item.args for item in suggestions.inputs] == [{"balance": 0, "amount": -5}]
    assert [test.filename for test in suggestions.tests] == ["test_cp_ai_outside_dir.py"]
    assert len(suggestions.rejected) == 5


def test_request_suggestions_uses_structured_output() -> None:
    client = fake_client({"inputs": [], "tests": []})
    assert request_suggestions("prompt", "claude-opus-5", client) == {"inputs": [], "tests": []}
    request = client.beta.messages.requests[0]
    assert request["output_config"]["format"]["type"] == "json_schema"
    assert request["model"] == "claude-opus-5"
    assert "NOT asked whether a patch is correct" in request["system"]


@pytest.mark.parametrize(
    ("payload", "stop_reason", "message"),
    [
        ({"inputs": [], "tests": []}, "refusal", "declined"),
        ({"inputs": [], "tests": []}, "max_tokens", "truncated"),
        ("not json", "end_turn", "invalid JSON"),
    ],
)
def test_request_suggestions_errors(payload: Any, stop_reason: str, message: str) -> None:
    with pytest.raises(AIError, match=message):
        request_suggestions("prompt", "claude-opus-5", fake_client(payload, stop_reason))


def test_require_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ConfigError, match="ANTHROPIC_API_KEY"):
        require_api_key()


def test_ai_generated_test_is_judged_differentially(patched_repo) -> None:
    repo = patched_repo({"banking.py": WITHDRAW_BASE}, {"banking.py": WITHDRAW_BUGGY})
    tautology = "from banking import withdraw\n\ndef test_always_fails():\n    assert withdraw(10, 5) == 999\n"
    client = fake_client(
        {
            "inputs": [],
            "tests": [
                {"name": "negative amount", "rationale": "validation removed?", "code": GOOD_TEST, "targets_intended_change": False},
                {"name": "wrong expectation", "rationale": "bogus", "code": tautology, "targets_intended_change": False},
            ],
        }
    )
    options = CheckOptions(repo=repo.path, base="main", ai=True, max_tests=20)
    result = run_check(options, ai_client=client)

    ai_findings = result.ai_findings
    assert [finding.test_name for finding in ai_findings] == ["negative amount"]
    assert ai_findings[0].severity is Severity.REGRESSION
    assert result.ai.tests_non_discriminating == 1
    assert (repo.path / ai_findings[0].reproduction).exists()
    assert not any((repo.path / ".counterpatch" / "generated").glob("*.py"))
    prompt = client.beta.messages.requests[0]["messages"][0]["content"]
    assert "from banking import withdraw" in prompt and "Base revision source" in prompt
