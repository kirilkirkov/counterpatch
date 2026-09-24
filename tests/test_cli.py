from __future__ import annotations

import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from counterpatch import __version__
from counterpatch.cli import app
from tests.conftest import WITHDRAW_BASE, WITHDRAW_BUGGY, WITHDRAW_EQUIVALENT

runner = CliRunner()


def invoke(*args: str):
    return runner.invoke(app, list(args))


def test_version_and_help() -> None:
    assert invoke("--version").output.strip() == f"counterpatch {__version__}"
    assert invoke("--help").exit_code == 0
    result = invoke("check", "--help")
    assert result.exit_code == 0
    # Typer forces Rich colors when GITHUB_ACTIONS/FORCE_COLOR is set, even under CliRunner.
    plain = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
    for option in ["--base", "--task", "--task-file", "--ai", "--max-tests", "--timeout", "--verbose"]:
        assert option in plain


def test_exit_code_0_when_patch_preserves_behavior(patched_repo) -> None:
    repo = patched_repo({"banking.py": WITHDRAW_BASE}, {"banking.py": WITHDRAW_EQUIVALENT})
    result = invoke("check", "--repo", str(repo.path), "--base", "main", "--task", "Refactor withdrawal validation")
    assert result.exit_code == 0, result.output
    assert "✓ No regression counterexample found." in result.output
    assert "This does not prove the patch is correct." in result.output
    assert (repo.path / ".counterpatch" / "report.json").exists()


def test_exit_code_1_when_regression_found(patched_repo) -> None:
    repo = patched_repo({"banking.py": WITHDRAW_BASE}, {"banking.py": WITHDRAW_BUGGY})
    result = invoke("check", "--repo", str(repo.path), "--base", "main", "--max-tests", "30")
    assert result.exit_code == 1, result.output
    assert "✗ Regression candidate found" in result.output
    assert "withdraw(0, 0)" in result.output


def test_exit_code_2_for_configuration_and_git_errors(patched_repo, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = patched_repo({"banking.py": WITHDRAW_BASE}, {"banking.py": WITHDRAW_BUGGY})
    missing_base = invoke("check", "--repo", str(repo.path), "--base", "does-not-exist")
    assert missing_base.exit_code == 2 and "does not exist" in missing_base.output

    not_a_repo = tmp_path / "plain"
    not_a_repo.mkdir()
    assert invoke("check", "--repo", str(not_a_repo)).exit_code == 2

    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    no_key = invoke("check", "--repo", str(repo.path), "--base", "main", "--ai")
    assert no_key.exit_code == 2 and "ANTHROPIC_API_KEY" in no_key.output

    missing_task = invoke("check", "--repo", str(repo.path), "--base", "main", "--task-file", str(tmp_path / "nope.md"))
    assert missing_task.exit_code == 2


def test_exit_code_2_when_timeout_budget_is_exhausted(patched_repo) -> None:
    slow = "import time\n\n\ndef f(x: int) -> int:\n    time.sleep(0.2)\n    return x\n"
    repo = patched_repo({"slow.py": slow}, {"slow.py": slow.replace("return x", "return x  # unchanged behavior")})
    result = invoke("check", "--repo", str(repo.path), "--base", "main", "--timeout", "2")
    assert result.exit_code == 2, result.output
    assert "--timeout budget" in result.output


def test_timeout_budget_shorter_than_call_timeout_still_tests(patched_repo) -> None:
    repo = patched_repo({"banking.py": WITHDRAW_BASE}, {"banking.py": WITHDRAW_BUGGY})
    result = invoke("check", "--repo", str(repo.path), "--base", "main", "--timeout", "3", "--call-timeout", "5")
    assert result.exit_code == 1, result.output


def test_task_file_is_used_to_classify_intended_changes(patched_repo, tmp_path: Path) -> None:
    base = """
    def validate_username(username: str) -> bool:
        if len(username) > 20:
            raise ValueError("username too long")
        return True
    """
    repo = patched_repo({"users.py": base}, {"users.py": base.replace("> 20", "> 32")})
    issue = tmp_path / "issue.md"
    issue.write_text("Allow usernames up to 32 characters.\n")

    without_task = invoke("check", "--repo", str(repo.path), "--base", "main")
    assert without_task.exit_code == 1

    with_task = invoke("check", "--repo", str(repo.path), "--base", "main", "--task-file", str(issue))
    assert with_task.exit_code == 0, with_task.output
    assert "⚠ Behavioral divergence found" in with_task.output
    assert "Possibly intended" in with_task.output

    strict = invoke("check", "--repo", str(repo.path), "--base", "main", "--task-file", str(issue), "--fail-on-divergence")
    assert strict.exit_code == 1
