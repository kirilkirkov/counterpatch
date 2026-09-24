from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from counterpatch.analyzer import collect_functions
from counterpatch.candidates import CallPlan, build_call_plan
from counterpatch.models import ChangedFunction, CheckResult, ConfigError, Finding, FunctionReport, Observation, Severity
from counterpatch.reporter import github_annotations, render_json, render_markdown, render_text
from counterpatch.reproduction import render_reproduction, value_source, write_reproduction
from counterpatch.utils import safe_join, sanitize_filename

SOURCE = (
    "def withdraw(balance: int, amount: int) -> int:\n    if amount <= 0:\n        raise ValueError('x')\n    return balance - amount\n"
)


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("withdraw negative amount", "test_withdraw_negative_amount.py"),
        ("../../etc/passwd", "test_etc_passwd.py"),
        ("test_Already Prefixed!", "test_already_prefixed.py"),
        ("", "test_case.py"),
        ("a" * 300, "test_" + "a" * 75 + ".py"),
    ],
)
def test_sanitize_filename(name: str, expected: str) -> None:
    assert sanitize_filename(name) == expected


@pytest.mark.parametrize("name", ["../evil.py", "sub/evil.py", "..", "", "a\\b.py", "x\x00.py"])
def test_safe_join_prevents_path_traversal(tmp_path: Path, name: str) -> None:
    with pytest.raises(ConfigError):
        safe_join(tmp_path, name)
    assert safe_join(tmp_path, "test_ok.py") == (tmp_path / "test_ok.py").resolve()


def test_value_source_handles_special_values() -> None:
    assert value_source(math.inf) == 'float("inf")'
    assert value_source([1, math.nan]) == '[1, float("nan")]'
    assert value_source((1,)) == "(1,)"
    assert value_source(set()) == "set()"
    assert value_source({"a": "b"}) == "{'a': 'b'}"


def _plan() -> CallPlan:
    changed = ChangedFunction(
        file="src/banking.py",
        base_file="src/banking.py",
        qualified_name="withdraw",
        status="modified",
        diff="",
        patch=collect_functions(SOURCE)["withdraw"],
        base=collect_functions(SOURCE)["withdraw"],
    )
    plan = build_call_plan(changed)
    assert isinstance(plan, CallPlan)
    return plan


def _finding() -> Finding:
    return Finding(
        file="src/banking.py",
        function="withdraw",
        signature="withdraw(balance: int, amount: int) -> int",
        severity=Severity.REGRESSION,
        kind="accepts-rejected-input",
        summary="Previously rejected input is now accepted",
        source="deterministic",
        args={"balance": 0, "amount": -1},
        call="withdraw(0, -1)",
        base=Observation(status="raised", exception_type="ValueError", exception_message="Amount must be positive"),
        patch=Observation(status="returned", return_value_repr="1", return_value_source="1"),
        line=1,
        reproduction=".counterpatch/reproductions/test_withdraw_accepts_rejected_input.py",
        reproduction_verified=True,
        reproduction_note="passes on base, fails on patch",
    )


def test_reproduction_file_is_runnable_pytest(tmp_path: Path) -> None:
    path = write_reproduction(tmp_path, _plan(), _finding(), "src", "banking", "main @ abc123")
    assert path == tmp_path / ".counterpatch" / "reproductions" / "test_withdraw_accepts_rejected_input.py"
    text = path.read_text()
    compile(text, str(path), "exec")
    assert "with pytest.raises(ValueError):\n        withdraw(0, -1)" in text
    assert 'parents[2] / "src"' in text
    assert (tmp_path / ".counterpatch" / ".gitignore").exists()


def test_reproduction_for_changed_return_value() -> None:
    finding = _finding()
    finding.kind = "return-value-changed"
    finding.base = Observation(status="returned", return_value_repr="[1, 2]", return_value_source="[1, 2]")
    text = render_reproduction(_plan(), finding, ".", "banking", "main")
    assert "result = withdraw(0, -1)\n    assert result == [1, 2]" in text
    assert "import pytest" not in text


def _result(**kwargs) -> CheckResult:
    report = FunctionReport(
        file="src/banking.py", function="withdraw", signature="withdraw()", status="tested", candidates=24, executions=48
    )
    return CheckResult(
        base_ref="origin/main",
        base_sha="abc123",
        patch_label="HEAD",
        task="Refactor",
        functions=[report],
        changed_files=["src/banking.py"],
        **kwargs,
    )


def test_render_text_success_and_failure() -> None:
    clean = _result()
    text = render_text(clean)
    assert "✓ No regression counterexample found." in text
    assert "This does not prove the patch is correct." in text
    assert "Candidates explored: 24" in text
    assert clean.exit_code == 0

    failing = _result()
    failing.functions[0].findings.append(_finding())
    text = render_text(failing)
    assert "✗ Regression candidate found" in text
    assert "    balance = 0\n    amount = -1" in text
    assert "raises ValueError('Amount must be positive')" in text
    assert "pytest .counterpatch/reproductions/test_withdraw_accepts_rejected_input.py -q" in text
    assert failing.exit_code == 1


def test_divergence_exit_codes_and_headline() -> None:
    result = _result()
    finding = _finding()
    finding.severity = Severity.DIVERGENCE
    result.functions[0].findings.append(finding)
    assert "⚠ Behavioral divergence found" in render_text(result)
    assert result.exit_code == 0
    result.fail_on_divergence = True
    assert result.exit_code == 1


def test_error_and_timeout_exit_codes() -> None:
    assert _result(errors=["AI failed"]).exit_code == 2
    assert _result(timed_out=True).exit_code == 2
    regression_and_timeout = _result(timed_out=True)
    regression_and_timeout.functions[0].findings.append(_finding())
    assert regression_and_timeout.exit_code == 1


def test_json_markdown_and_annotations() -> None:
    result = _result()
    result.functions[0].findings.append(_finding())
    data = json.loads(render_json(result))
    assert data["exit_code"] == 1 and data["findings"][0]["kind"] == "accepts-rejected-input"
    assert "Regression candidate: `withdraw`" in render_markdown(result)
    [annotation] = github_annotations(result)
    assert annotation.startswith("::error file=src/banking.py,line=1,title=CounterPatch::")
