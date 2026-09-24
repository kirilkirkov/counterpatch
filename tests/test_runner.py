from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

from counterpatch.runner import Worker, run_process, run_pytest

TARGET = """
import os
import signal
import time

calls = []


def add(a: int, b: int) -> int:
    print("adding")
    return a + b


def fail(x: int) -> int:
    raise ValueError(f"bad {x}")


def hang(x: int) -> int:
    while True:
        time.sleep(0.01)


def segfault(x: int) -> int:
    os.kill(os.getpid(), signal.SIGSEGV)


def mutate(items: list) -> None:
    items.append(1)


def noisy(x: int) -> int:
    os.write(1, b"garbage on fd 1\\n")
    return x


class Counter:
    def __init__(self) -> None:
        self.count = 0

    def bump(self, by: int) -> int:
        self.count += by
        return self.count


class NeedsArgs:
    def __init__(self, dependency) -> None:
        self.dependency = dependency

    def run(self, x: int) -> int:
        return x
"""


@pytest.fixture
def workspace(tmp_path: Path) -> Path:
    (tmp_path / "target.py").write_text(TARGET)
    return tmp_path


def request(workspace: Path, qualname: str, args: list, kind: str = "function", module: str = "target") -> dict:
    return {
        "module": module,
        "file": str((workspace / (module.replace(".", "/") + ".py")).resolve()),
        "qualname": qualname,
        "kind": kind,
        "roots": [str(workspace)],
        "args": args,
    }


def test_worker_observes_return_exception_stdout_and_mutation(workspace: Path) -> None:
    with Worker(sys.executable, workspace, "test") as worker:
        observation = worker.call(request(workspace, "add", [["a", "1", False], ["b", "2", False]]), 5)
        assert observation.status == "returned"
        assert observation.return_value_repr == "3" and observation.return_value_source == "3"
        assert observation.stdout == "adding\n"

        observation = worker.call(request(workspace, "fail", [["x", "7", False]]), 5)
        assert (observation.status, observation.exception_type, observation.exception_message) == ("raised", "ValueError", "bad 7")

        observation = worker.call(request(workspace, "mutate", [["items", "[]", False]]), 5)
        assert observation.args_after == {"items": "[1]"}

        observation = worker.call(request(workspace, "noisy", [["x", "4", False]]), 5)
        assert observation.return_value_repr == "4"


def test_worker_instance_methods_and_construction_errors(workspace: Path) -> None:
    with Worker(sys.executable, workspace, "test") as worker:
        observation = worker.call(request(workspace, "Counter.bump", [["by", "2", False]], "instance"), 5)
        assert observation.return_value_repr == "2"
        assert observation.state_after == "{'count': 2}"
        observation = worker.call(request(workspace, "NeedsArgs.run", [["x", "1", False]], "instance"), 5)
        assert observation.status == "error"
        assert "could not construct NeedsArgs()" in (observation.detail or "")


def test_worker_timeout_and_restart(workspace: Path) -> None:
    with Worker(sys.executable, workspace, "test") as worker:
        observation = worker.call(request(workspace, "hang", [["x", "1", False]]), 0.5)
        assert observation.status == "timeout"
        observation = worker.call(request(workspace, "add", [["a", "1", False], ["b", "1", False]]), 5)
        assert observation.return_value_repr == "2"
        assert worker.restarts == 1


def test_worker_crash_is_an_observation_not_a_counterpatch_crash(workspace: Path) -> None:
    with Worker(sys.executable, workspace, "test") as worker:
        observation = worker.call(request(workspace, "segfault", [["x", "1", False]]), 5)
        assert observation.status == "crashed"
        assert "SIGSEGV" in (observation.detail or "")
        assert worker.call(request(workspace, "add", [["a", "2", False], ["b", "2", False]]), 5).return_value_repr == "4"


def test_worker_import_errors_are_harness_errors(workspace: Path) -> None:
    (workspace / "broken.py").write_text("import does_not_exist_anywhere\n")
    with Worker(sys.executable, workspace, "test") as worker:
        observation = worker.call(request(workspace, "f", [], module="broken"), 5)
        assert (observation.status, observation.phase) == ("error", "import")
        assert "ModuleNotFoundError" in (observation.detail or "")


def test_worker_refuses_modules_resolving_to_another_file(workspace: Path) -> None:
    (workspace / "a").mkdir()
    (workspace / "b").mkdir()
    (workspace / "a" / "utils.py").write_text("def f() -> str:\n    return 'a'\n")
    (workspace / "b" / "utils.py").write_text("def f() -> str:\n    return 'b'\n")
    with Worker(sys.executable, workspace, "test") as worker:
        first = {
            **request(workspace, "f", [], module="utils"),
            "file": str((workspace / "a" / "utils.py").resolve()),
            "roots": [str(workspace / "a")],
        }
        assert worker.call(first, 5).return_value_repr == "'a'"
        second = {**first, "file": str((workspace / "b" / "utils.py").resolve()), "roots": [str(workspace / "b")]}
        observation = worker.call(second, 5)
    assert observation.status == "error"
    assert "expected" in (observation.detail or "")
    with Worker(sys.executable, workspace, "test") as worker:
        observation = worker.call({**request(workspace, "dumps", [], module="json"), "file": str(workspace / "json.py")}, 5)
    assert observation.status == "error"


def test_slow_import_is_not_counted_against_the_call_timeout(workspace: Path) -> None:
    (workspace / "slow.py").write_text("import time\ntime.sleep(1.5)\n\ndef f(x: int) -> int:\n    return x\n")
    with Worker(sys.executable, workspace, "test") as worker:
        observation = worker.call(request(workspace, "f", [["x", "3", False]], module="slow"), 0.5)
    assert observation.return_value_repr == "3"


def test_target_reading_stdin_cannot_consume_protocol(workspace: Path) -> None:
    (workspace / "asks.py").write_text(
        "def ask(x: int) -> str:\n    try:\n        return input()\n    except EOFError:\n        return 'eof'\n"
    )
    with Worker(sys.executable, workspace, "test") as worker:
        assert worker.call(request(workspace, "ask", [["x", "1", False]], module="asks"), 5).return_value_repr == "'eof'"
        assert worker.call(request(workspace, "add", [["a", "1", False], ["b", "1", False]]), 5).return_value_repr == "2"


def test_float_noise_has_equal_comparison_keys(workspace: Path) -> None:
    (workspace / "floats.py").write_text(
        "def a() -> float:\n    return (0.0 + -1.0 + 86.0) / 3\n\ndef b() -> float:\n    return 0.0 / 3 + -1.0 / 3 + 86.0 / 3\n\ndef c() -> float:\n    return 28.34\n"
    )
    with Worker(sys.executable, workspace, "test") as worker:
        a, b, c = (worker.call(request(workspace, name, [], module="floats"), 5) for name in "abc")
    assert a.return_value_repr != b.return_value_repr
    assert a.return_value_key == b.return_value_key != c.return_value_key


def test_worker_does_not_leak_secret_environment(workspace: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (workspace / "envcheck.py").write_text("import os\n\ndef read() -> str:\n    return os.environ.get('ANTHROPIC_API_KEY', 'absent')\n")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-secret")
    with Worker(sys.executable, workspace, "test") as worker:
        observation = worker.call(request(workspace, "read", [], module="envcheck"), 5)
    assert observation.return_value_repr == "'absent'"


def test_run_process_timeout(tmp_path: Path) -> None:
    result = run_process([sys.executable, "-c", "import time; time.sleep(30)"], tmp_path, dict(os.environ), 0.5)
    assert result.timed_out
    result = run_process([sys.executable, "-c", "import sys; print('hi'); sys.exit(3)"], tmp_path, dict(os.environ), 10)
    assert (result.returncode, result.stdout.strip(), result.timed_out) == (3, "hi", False)


def test_run_pytest_outcomes(tmp_path: Path) -> None:
    (tmp_path / "test_pass.py").write_text("def test_ok():\n    assert True\n")
    (tmp_path / "test_fail.py").write_text("def test_bad():\n    assert 1 == 2\n")
    (tmp_path / "test_broken.py").write_text("import nope_not_installed\n\ndef test_x():\n    pass\n")
    files = [tmp_path / "test_pass.py", tmp_path / "test_fail.py", tmp_path / "test_broken.py"]
    run = run_pytest(sys.executable, tmp_path, files, [], 60)
    assert run.outcomes == {"test_pass.py": "passed", "test_fail.py": "failed", "test_broken.py": "error"}
    assert not (tmp_path / ".pytest_cache").exists()
