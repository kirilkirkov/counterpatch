"""Subprocess execution: probe workers and pytest runs.

Target code never runs inside the CounterPatch process. Function probes run in a
persistent worker subprocess per revision (restarted after a timeout or crash);
generated tests and reproductions run in fresh ``python -m pytest`` subprocesses.
Process isolation is not a security sandbox.
"""

from __future__ import annotations

import json
import os
import selectors
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ElementTree
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from counterpatch.models import HarnessError, Observation
from counterpatch.utils import subprocess_env, truncate

WORKER_SCRIPT = Path(__file__).with_name("_worker.py")


@dataclass
class ProcessResult:
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration: float


def _kill_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        process.kill()


def run_process(command: list[str], cwd: Path, env: dict[str, str], timeout: float) -> ProcessResult:
    """Run ``command`` with a hard timeout; the whole process group is killed on expiry."""
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            errors="replace",
            start_new_session=True,
        )
    except OSError as error:
        raise HarnessError(f"could not start {command[0]}: {error}") from error
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        _kill_group(process)
        stdout, stderr = process.communicate()
        timed_out = True
    return ProcessResult(process.returncode, stdout, stderr, timed_out, time.monotonic() - started)


def describe_returncode(returncode: int | None) -> str:
    if returncode is not None and returncode < 0:
        try:
            return f"killed by {signal.Signals(-returncode).name}"
        except ValueError:
            return f"killed by signal {-returncode}"
    return f"exit code {returncode}"


class Worker:
    """A persistent probe subprocess bound to one workspace (revision)."""

    def __init__(self, python: str, workspace: Path, label: str) -> None:
        self.python = python
        self.workspace = workspace
        self.label = label
        self.process: subprocess.Popen[bytes] | None = None
        self._buffer = b""
        self._log = tempfile.TemporaryFile()
        self.restarts = 0
        self._prepared: set[tuple[str, str]] = set()

    def _start(self) -> subprocess.Popen[bytes]:
        if self.process is None or self.process.poll() is not None:
            self._buffer = b""
            self._prepared.clear()
            try:
                self.process = subprocess.Popen(
                    [self.python, str(WORKER_SCRIPT), str(self.workspace)],
                    cwd=self.workspace,
                    env=subprocess_env(),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=self._log,
                    start_new_session=True,
                )
            except OSError as error:
                raise HarnessError(f"could not start Python interpreter {self.python!r}: {error}") from error
        return self.process

    def call(self, request: dict[str, Any], timeout: float, import_timeout: float = 120.0) -> Observation:
        """Execute one candidate. Importing the target module is done first, under its own
        timeout, so slow imports are never mistaken for a hanging function."""
        ready = self.prepare(request["module"], request["file"], request["roots"], import_timeout)
        if ready.status == "error":
            return ready
        observation = self._exchange(request, timeout)
        if observation.status == "timeout":
            observation.timeout_seconds = timeout
        return observation

    def prepare(self, module: str, file: str, roots: list[str], timeout: float) -> Observation:
        """Import ``module`` (verified to be ``file``) once per worker process.

        Returns an observation with status ``ready`` or ``error`` (phase ``import`` or
        ``resolve``); an import that hangs or kills the interpreter is an import error.
        """
        if (module, file) in self._prepared:
            return Observation(status="ready")
        ready = self._exchange({"op": "prepare", "module": module, "file": file, "roots": roots}, timeout)
        if ready.status in {"timeout", "crashed"}:
            return Observation(status="error", phase="import", detail=f"importing {module} {ready.describe()}")
        if ready.status == "ready":
            self._prepared.add((module, file))
        return ready

    def _exchange(self, request: dict[str, Any], timeout: float) -> Observation:
        process = self._start()
        assert process.stdin is not None and process.stdout is not None
        try:
            process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            return self._died(process)

        deadline = time.monotonic() + timeout
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)
        try:
            while b"\n" not in self._buffer:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    self._terminate()
                    return Observation(status="timeout", detail=f"no result within {timeout:g}s")
                chunk = os.read(process.stdout.fileno(), 65536)
                if not chunk:
                    return self._died(process)
                self._buffer += chunk
        finally:
            selector.close()
        line, self._buffer = self._buffer.split(b"\n", 1)
        try:
            data = json.loads(line)
            if data.get("status") == "ready":
                return Observation(status="ready")
            return Observation.from_dict(data)
        except (ValueError, TypeError) as error:
            raise HarnessError(f"{self.label} worker sent an invalid response: {error}") from error

    def _died(self, process: subprocess.Popen[bytes]) -> Observation:
        returncode = process.wait()
        self.process = None
        self.restarts += 1
        return Observation(status="crashed", detail=f"{describe_returncode(returncode)}; {self.stderr_tail()}")

    def stderr_tail(self, limit: int = 300) -> str:
        try:
            self._log.seek(0, os.SEEK_END)
            size = self._log.tell()
            self._log.seek(max(0, size - limit))
            return self._log.read().decode("utf-8", "replace").strip()
        except OSError:
            return ""

    def _terminate(self) -> None:
        if self.process is not None:
            _kill_group(self.process)
            self.process.wait()
            self.process = None
            self.restarts += 1

    def close(self) -> None:
        if self.process is not None:
            try:
                if self.process.stdin:
                    self.process.stdin.close()
                self.process.wait(timeout=2)
            except (subprocess.TimeoutExpired, OSError):
                _kill_group(self.process)
                self.process.wait()
            finally:
                for stream in (self.process.stdout,):
                    if stream:
                        stream.close()
                self.process = None
        self._log.close()

    def __enter__(self) -> Worker:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


@dataclass
class PytestRun:
    """Outcome of one pytest subprocess; ``outcomes`` is keyed by test file name."""

    process: ProcessResult
    outcomes: dict[str, str] = field(default_factory=dict)  # passed | failed | error | skipped
    messages: dict[str, str] = field(default_factory=dict)


def run_pytest(python: str, workspace: Path, test_files: list[Path], import_roots: list[Path], timeout: float) -> PytestRun:
    """Run pytest on ``test_files`` inside ``workspace`` and collect per-file outcomes.

    Project ``addopts`` are neutralized and the cache provider is disabled so the run
    does not write ``.pytest_cache`` into the developer's tree.
    """
    workspace = workspace.resolve()
    test_files = [path.resolve() for path in test_files]
    with tempfile.TemporaryDirectory(prefix="counterpatch-junit-") as temp_dir:
        junit = Path(temp_dir) / "junit.xml"
        command = [
            python,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            "-o",
            "addopts=",
            "--continue-on-collection-errors",
            "-o",
            "junit_family=xunit2",
            f"--junitxml={junit}",
            "--rootdir",
            str(workspace),
            *[str(path) for path in test_files],
        ]
        result = run_process(command, workspace, subprocess_env(import_roots), timeout)
        if "No module named pytest" in result.stderr:
            raise HarnessError(f"pytest is not installed for {python}")
        run = PytestRun(process=result)
        if junit.exists():
            _parse_junit(junit, test_files, run)
    for path in test_files:
        if path.name not in run.outcomes:
            run.outcomes[path.name] = "error"
            run.messages[path.name] = "timed out" if result.timed_out else truncate(result.stdout[-600:] + result.stderr[-300:])
    return run


def _parse_junit(junit: Path, test_files: list[Path], run: PytestRun) -> None:
    rank = {"passed": 0, "skipped": 1, "failed": 2, "error": 3}
    stems = {path.stem: path.name for path in test_files}
    try:
        root = ElementTree.parse(junit).getroot()
    except ElementTree.ParseError:
        return
    for case in root.iter("testcase"):
        classname = case.get("classname", "") + "." + case.get("name", "")
        name = next((file for stem, file in stems.items() if f".{stem}." in f".{classname}."), None)
        if name is None and len(test_files) == 1:
            name = test_files[0].name
        if name is None:
            continue
        outcome, message = "passed", ""
        for child in case:
            if child.tag in {"failure", "error", "skipped"}:
                outcome = {"failure": "failed", "error": "error", "skipped": "skipped"}[child.tag]
                message = child.get("message", "") or (child.text or "")
        if rank[outcome] >= rank.get(run.outcomes.get(name, "passed"), 0) or name not in run.outcomes:
            run.outcomes[name] = outcome
            if message:
                run.messages[name] = truncate(message, 600)
