from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from collections.abc import Callable
from pathlib import Path

import pytest

GIT_ENV = {
    "GIT_AUTHOR_NAME": "Test",
    "GIT_AUTHOR_EMAIL": "test@example.com",
    "GIT_COMMITTER_NAME": "Test",
    "GIT_COMMITTER_EMAIL": "test@example.com",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_CONFIG_SYSTEM": os.devnull,
}

WITHDRAW_BASE = """
def withdraw(balance: int, amount: int) -> int:
    if amount <= 0:
        raise ValueError("Amount must be positive")

    if amount > balance:
        raise ValueError("Insufficient balance")

    return balance - amount
"""

WITHDRAW_BUGGY = """
def withdraw(balance: int, amount: int) -> int:
    if amount <= balance:
        return balance - amount

    raise ValueError("Insufficient balance")
"""

WITHDRAW_EQUIVALENT = """
def withdraw(balance: int, amount: int) -> int:
    if amount <= 0:
        raise ValueError("Amount must be positive")
    if balance < amount:
        raise ValueError("Insufficient balance")
    return balance - amount
"""


def git(repo: Path, *args: str) -> str:
    completed = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True, env={**os.environ, **GIT_ENV}, check=True)
    return completed.stdout


class Repo:
    """A throwaway Git repository for tests."""

    def __init__(self, path: Path) -> None:
        self.path = path
        path.mkdir(parents=True, exist_ok=True)
        git(path, "init", "-q", "-b", "main")

    def write(self, relative: str, content: str) -> Path:
        target = self.path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")
        return target

    def commit(self, message: str = "commit") -> str:
        git(self.path, "add", "-A")
        git(self.path, "commit", "-q", "--allow-empty", "-m", message)
        return git(self.path, "rev-parse", "HEAD").strip()

    def branch(self, name: str) -> None:
        git(self.path, "checkout", "-q", "-b", name)

    def git(self, *args: str) -> str:
        return git(self.path, *args)


@pytest.fixture
def make_repo(tmp_path: Path) -> Callable[..., Repo]:
    counter = iter(range(1000))

    def factory(files: dict[str, str] | None = None) -> Repo:
        repo = Repo(tmp_path / f"repo{next(counter)}")
        for relative, content in (files or {}).items():
            repo.write(relative, content)
        repo.commit("base")
        return repo

    return factory


@pytest.fixture
def patched_repo(make_repo: Callable[..., Repo]) -> Callable[[dict[str, str], dict[str, str]], Repo]:
    """Repo with a base commit on ``main`` and a patch commit on branch ``patch``."""

    def factory(base: dict[str, str], patch: dict[str, str]) -> Repo:
        repo = make_repo(base)
        repo.branch("patch")
        for relative, content in patch.items():
            repo.write(relative, content)
        repo.commit("patch")
        return repo

    return factory


@pytest.fixture
def python() -> str:
    return sys.executable
