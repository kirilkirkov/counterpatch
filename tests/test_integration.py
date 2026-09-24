"""End-to-end tests: real Git repositories, real subprocess execution, real CounterPatch runs."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from counterpatch.check import CheckOptions, run_check
from counterpatch.models import Severity
from tests.conftest import WITHDRAW_BASE, WITHDRAW_BUGGY

ROOT = Path(__file__).resolve().parents[1]


def cli(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "counterpatch", "check", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=300,
    )


def test_counterpatch_finds_regression_in_real_repository(patched_repo) -> None:
    repo = patched_repo(
        {
            "src/banking.py": WITHDRAW_BASE,
            "tests/test_banking.py": "from banking import withdraw\n\ndef test_ok():\n    assert withdraw(100, 30) == 70\n",
        },
        {"src/banking.py": WITHDRAW_BUGGY},
    )
    completed = cli(repo.path, "--base", "main", "--task", "Refactor withdrawal validation")
    assert completed.returncode == 1, completed.stdout + completed.stderr
    output = completed.stdout
    assert "✗ Regression candidate found" in output
    assert "balance = 0\n    amount = 0" in output
    assert "raises ValueError('Amount must be positive')" in output
    assert "returns 0" in output

    reproduction = repo.path / ".counterpatch" / "reproductions" / "test_withdraw_accepts_rejected_input.py"
    assert reproduction.exists()
    assert "verified: passes on base, fails on patch" in output
    rerun = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(reproduction)],
        cwd=repo.path,
        capture_output=True,
        text=True,
    )
    assert rerun.returncode == 1 and "DID NOT RAISE" in rerun.stdout

    assert len(repo.git("worktree", "list").splitlines()) == 1
    assert repo.git("status", "--porcelain", "--untracked-files=no") == ""


def test_uncommitted_changes_are_checked_and_preserved(make_repo) -> None:
    repo = make_repo({"banking.py": WITHDRAW_BASE})
    repo.write("banking.py", WITHDRAW_BUGGY)
    result = run_check(CheckOptions(repo=repo.path, base="HEAD", max_tests=30))
    assert result.patch_label.startswith("working tree")
    assert result.regressions
    assert (repo.path / "banking.py").read_text() == WITHDRAW_BUGGY.lstrip("\n")
    assert "banking.py" in repo.git("status", "--porcelain")
    assert not list(repo.path.rglob("__pycache__"))


def test_username_scenarios_intended_vs_unrelated(patched_repo) -> None:
    base = """
    def validate_username(username: str) -> bool:
        if not username:
            raise ValueError("username cannot be empty")
        if len(username) > 20:
            raise ValueError("username too long")
        return True
    """
    intended = base.replace("> 20", "> 32")
    buggy = """
    def validate_username(username: str) -> bool:
        if len(username) > 32:
            raise ValueError("username too long")
        return True
    """
    task = "Allow usernames up to 32 characters"

    good_repo = patched_repo({"users.py": base}, {"users.py": intended})
    good = run_check(CheckOptions(repo=good_repo.path, base="main", task=task))
    assert not good.regressions
    assert good.divergences and good.divergences[0].intent_note.startswith("Possibly intended")
    assert good.exit_code == 0

    bad_repo = patched_repo({"users.py": base}, {"users.py": buggy})
    bad = run_check(CheckOptions(repo=bad_repo.path, base="main", task=task))
    [regression] = bad.regressions
    assert regression.args == {"username": ""}
    assert regression.base.exception_message == "username cannot be empty"
    assert regression.patch.return_value_repr == "True"
    assert regression.reproduction_verified


def test_methods_crashes_hangs_and_skips(patched_repo) -> None:
    base = """
    class Cart:
        def __init__(self) -> None:
            self.items: list[int] = []

        def average(self, total: int, count: int) -> float:
            if count <= 0:
                return 0.0
            return total / count

        def countdown(self, n: int) -> int:
            steps = 0
            while n > 0:
                n -= 1
                steps += 1
            return steps


    class Gateway:
        def __init__(self, client) -> None:
            self.client = client

        def charge(self, amount: int) -> int:
            return amount
    """
    patch = (
        base.replace("            if count <= 0:\n                return 0.0\n", "")
        .replace("while n > 0:", "while n != 0:")
        .replace("return amount", "return amount + 1")
    )
    repo = patched_repo({"shop.py": base}, {"shop.py": patch})
    result = run_check(CheckOptions(repo=repo.path, base="main", max_tests=40, call_timeout=1))

    by_function = {report.function: report for report in result.functions}
    [average] = by_function["Cart.average"].findings
    assert (average.severity, average.kind) == (Severity.REGRESSION, "new-crash")
    assert average.args["count"] == 0 and average.patch.exception_type == "ZeroDivisionError"

    [countdown] = by_function["Cart.countdown"].findings
    assert (countdown.severity, countdown.kind) == (Severity.REGRESSION, "new-hang")
    assert countdown.args == {"n": -1}

    gateway = by_function["Gateway.charge"]
    assert gateway.status == "skipped"
    assert "could not construct Gateway()" in (gateway.skip_reason or "")


def test_no_python_changes(patched_repo) -> None:
    repo = patched_repo({"README.md": "a\n", "app.py": "x = 1\n"}, {"README.md": "b\n"})
    result = run_check(CheckOptions(repo=repo.path, base="main"))
    assert result.functions == [] and result.exit_code == 0


def test_demo_script_runs_end_to_end(tmp_path: Path) -> None:
    script = ROOT / "examples" / "demo_project" / "run_demo.sh"
    env = {**os.environ, "PATH": f"{Path(sys.executable).parent}{os.pathsep}{os.environ['PATH']}", "TMPDIR": str(tmp_path)}
    completed = subprocess.run(["bash", str(script)], capture_output=True, text=True, env=env, timeout=300)
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "3 passed" in completed.stdout
    assert "✗ Regression candidate found" in completed.stdout
    assert "with pytest.raises(ValueError):" in completed.stdout
    # The reproduction is re-run with plain pytest: it fails on the patch and passes on the base.
    patch_run, base_run = completed.stdout.split("expected to FAIL", 1)[1].split("expected to PASS", 1)
    assert "1 failed" in patch_run and "DID NOT RAISE" in patch_run
    assert "1 passed" in base_run


def test_patch_that_breaks_module_import_is_an_error_not_a_pass(patched_repo) -> None:
    base = "def f(x: int) -> int:\n    return x\n"
    repo = patched_repo({"mod.py": base}, {"mod.py": "import does_not_exist_anywhere\n\n" + base})
    result = run_check(CheckOptions(repo=repo.path, base="main"))
    assert result.exit_code == 2
    assert "fails to import, but the base revision imports" in result.errors[0]


def test_syntax_error_in_patch_exits_2(patched_repo) -> None:
    repo = patched_repo({"mod.py": "def f(x: int) -> int:\n    return x\n"}, {"mod.py": "def f(x: int) -> int:\n    return x +\n"})
    assert run_check(CheckOptions(repo=repo.path, base="main")).exit_code == 2


def test_module_level_constant_change_reaches_functions_that_use_it(patched_repo) -> None:
    base = 'LIMIT = 100\n\n\ndef within_limit(amount: int) -> bool:\n    if amount > LIMIT:\n        raise ValueError("over")\n    return True\n'
    repo = patched_repo({"limits.py": base}, {"limits.py": base.replace("LIMIT = 100", "LIMIT = 1000")})
    [regression] = run_check(CheckOptions(repo=repo.path, base="main")).regressions
    assert regression.args == {"amount": 101}
    assert regression.source == "deterministic"
    intended = run_check(CheckOptions(repo=repo.path, base="main", task="Raise LIMIT to 1000"))
    assert not intended.regressions and intended.divergences[0].intent_note.startswith("Possibly intended")


def test_intended_change_cannot_hide_an_unrelated_divergence(patched_repo) -> None:
    base = """
    def label(name: str) -> str:
        if len(name) > 20:
            raise ValueError("too long")
        if name == "admin":
            return "staff"
        return "user"
    """
    patch = base.replace("> 20", "> 32").replace('return "staff"', 'return "user"')
    repo = patched_repo({"users.py": base}, {"users.py": patch})
    result = run_check(CheckOptions(repo=repo.path, base="main", task="Allow names up to 32 characters"))
    [finding] = result.findings
    assert finding.args == {"name": "admin"}
    assert finding.intent_note is None


def test_no_false_positives_for_slow_imports_float_noise_or_stdin(patched_repo) -> None:
    base = "import time\ntime.sleep(0.2)\n\n\ndef mean3(a: float, b: float, c: float) -> float:\n    return (a + b + c) / 3\n"
    patch = (
        "import time\ntime.sleep(1.5)\n\n\ndef mean3(a: float, b: float, c: float) -> float:\n"
        "    try:\n        input()\n    except EOFError:\n        pass\n    return a / 3 + b / 3 + c / 3\n"
    )
    repo = patched_repo({"stats.py": base}, {"stats.py": patch})
    result = run_check(CheckOptions(repo=repo.path, base="main", task="Refactor mean3", call_timeout=1, max_tests=40))
    assert result.findings == [] and result.exit_code == 0
    assert result.functions[0].candidates >= 20


def test_results_are_deterministic_across_runs(patched_repo) -> None:
    repo = patched_repo({"banking.py": WITHDRAW_BASE}, {"banking.py": WITHDRAW_BUGGY})
    first, second = (run_check(CheckOptions(repo=repo.path, base="main", max_tests=60)) for _ in range(2))
    summary = [(f.args, f.kind, [o.args for o in f.other_examples]) for f in first.findings]
    assert summary == [(f.args, f.kind, [o.args for o in f.other_examples]) for f in second.findings]
    assert first.functions[0].candidates == second.functions[0].candidates


def test_sigterm_cleans_up_worktree_and_workers(patched_repo) -> None:
    import signal
    import time

    base = "import time\n\n\ndef slow(x: int) -> int:\n    time.sleep(0.5)\n    return x\n"
    repo = patched_repo({"slow.py": base}, {"slow.py": base.replace("return x", "return -x")})
    process = subprocess.Popen(
        [sys.executable, "-m", "counterpatch", "check", "--base", "main"], cwd=repo.path, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    deadline = time.monotonic() + 30
    while len(repo.git("worktree", "list").splitlines()) < 2 and time.monotonic() < deadline:
        time.sleep(0.05)
    time.sleep(0.5)
    process.send_signal(signal.SIGTERM)
    assert process.wait(timeout=30) == 2
    assert len(repo.git("worktree", "list").splitlines()) == 1
    leftover = subprocess.run(["pgrep", "-f", str(repo.path)], capture_output=True, text=True).stdout
    assert leftover.strip() == ""
