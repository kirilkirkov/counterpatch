"""Git plumbing: resolving the base revision, parsing diffs, temporary base worktrees."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from counterpatch.models import CounterPatchError, FileDiff, FileStatus, Hunk

_HUNK_HEADER = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
DEFAULT_BASE_CANDIDATES = ("origin/main", "main", "origin/master", "master")


class GitError(CounterPatchError):
    """A Git command failed or the repository is not in a usable state."""


def run_git(args: list[str], cwd: Path, check: bool = True) -> str:
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
    try:
        completed = subprocess.run(
            ["git", "-c", "core.quotePath=false", *args],
            cwd=cwd,
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )
    except FileNotFoundError as error:
        raise GitError("git executable not found on PATH") from error
    except subprocess.TimeoutExpired as error:
        raise GitError(f"git {' '.join(args)} timed out") from error
    if check and completed.returncode != 0:
        message = completed.stderr.strip() or completed.stdout.strip()
        raise GitError(f"git {' '.join(args)} failed: {message}")
    return completed.stdout


def repo_root(cwd: Path) -> Path:
    try:
        return Path(run_git(["rev-parse", "--show-toplevel"], cwd).strip())
    except GitError as error:
        raise GitError(f"{cwd} is not inside a Git repository") from error


def rev_exists(repo: Path, rev: str) -> bool:
    completed = run_git(["rev-parse", "--verify", "--quiet", f"{rev}^{{commit}}"], repo, check=False)
    return bool(completed.strip())


def default_base(repo: Path) -> str:
    for candidate in DEFAULT_BASE_CANDIDATES:
        if rev_exists(repo, candidate):
            return candidate
    raise GitError("Could not find a default base branch (tried origin/main, main, origin/master, master). Pass --base.")


def resolve_base(repo: Path, base: str) -> str:
    """Return the merge-base commit between ``base`` and ``HEAD``.

    Comparing against the merge-base mirrors ``git diff base...HEAD``: only changes made
    on the current branch are analyzed, not unrelated commits that landed on ``base``.
    """
    if not rev_exists(repo, base):
        raise GitError(f"Base revision {base!r} does not exist. In GitHub Actions use actions/checkout with fetch-depth: 0.")
    completed = run_git(["merge-base", base, "HEAD"], repo, check=False)
    sha = completed.strip()
    if not sha:
        raise GitError(f"No merge-base between {base!r} and HEAD. The history may be shallow; fetch full history (fetch-depth: 0).")
    return sha


def short_sha(repo: Path, sha: str) -> str:
    return run_git(["rev-parse", "--short", sha], repo).strip()


def has_uncommitted_changes(repo: Path) -> bool:
    return bool(run_git(["status", "--porcelain", "--untracked-files=no"], repo).strip())


def diff_python_files(repo: Path, base_sha: str) -> list[FileDiff]:
    """Zero-context diff of tracked ``.py`` files between ``base_sha`` and the working tree."""
    output = run_git(
        ["diff", "--no-color", "--no-ext-diff", "-U0", "-M", base_sha, "--", "*.py"],
        repo,
    )
    return parse_unified_diff(output)


def parse_unified_diff(text: str) -> list[FileDiff]:
    """Parse ``git diff`` output (any context size) into :class:`FileDiff` objects."""
    files: list[FileDiff] = []
    current: FileDiff | None = None
    hunk: Hunk | None = None

    for line in text.splitlines():
        if line.startswith("diff --git "):
            current = FileDiff(old_path=None, new_path=None, status=FileStatus.MODIFIED)
            files.append(current)
            hunk = None
            old_guess, new_guess = _paths_from_diff_git_line(line)
            current.old_path, current.new_path = old_guess, new_guess
            continue
        if current is None:
            continue
        if hunk is None or line[:1] not in {"+", "-", " ", "\\"}:
            if line.startswith("new file mode"):
                current.status = FileStatus.ADDED
                current.old_path = None
            elif line.startswith("deleted file mode"):
                current.status = FileStatus.DELETED
                current.new_path = None
            elif line.startswith("rename from "):
                current.status = FileStatus.RENAMED
                current.old_path = line[len("rename from ") :]
            elif line.startswith("rename to "):
                current.status = FileStatus.RENAMED
                current.new_path = line[len("rename to ") :]
            elif line.startswith("--- "):
                current.old_path = _strip_prefix(line[4:], "a/")
            elif line.startswith("+++ "):
                current.new_path = _strip_prefix(line[4:], "b/")
            elif line.startswith("@@"):
                match = _HUNK_HEADER.match(line)
                if match:
                    hunk = Hunk(
                        old_start=int(match.group(1)),
                        old_count=int(match.group(2)) if match.group(2) is not None else 1,
                        new_start=int(match.group(3)),
                        new_count=int(match.group(4)) if match.group(4) is not None else 1,
                    )
                    current.hunks.append(hunk)
            continue
        if line.startswith("\\"):
            continue
        hunk.lines.append(line)

    for file_diff in files:
        if file_diff.status is FileStatus.ADDED:
            file_diff.old_path = None
        if file_diff.status is FileStatus.DELETED:
            file_diff.new_path = None
    return files


def _strip_prefix(path: str, prefix: str) -> str | None:
    path = path.strip()
    if path == "/dev/null":
        return None
    if path.startswith('"') and path.endswith('"'):
        path = path[1:-1]
    return path[len(prefix) :] if path.startswith(prefix) else path


def _paths_from_diff_git_line(line: str) -> tuple[str | None, str | None]:
    rest = line[len("diff --git ") :]
    if rest.startswith("a/") and " b/" in rest:
        old, new = rest.split(" b/", 1)
        return old[2:], new
    return None, None


def show_file(repo: Path, sha: str, path: str) -> str | None:
    """Return the content of ``path`` at ``sha``, or ``None`` if it does not exist there."""
    try:
        return run_git(["show", f"{sha}:{path}"], repo)
    except GitError:
        return None


@contextmanager
def base_worktree(repo: Path, sha: str) -> Iterator[Path]:
    """Check out ``sha`` into a temporary, detached worktree and always remove it.

    Safety properties:

    * the developer's working tree, index, branches and other worktrees are untouched
      (no global ``git worktree prune``: only this worktree's own metadata is removed);
    * repository hooks (e.g. ``post-checkout``) do not run for the temporary checkout;
    * cleanup runs after exceptions, KeyboardInterrupt and SIGTERM (see ``cli``).
    """
    temp_dir = Path(tempfile.mkdtemp(prefix="counterpatch-")).resolve()
    worktree = temp_dir / "base"
    no_hooks = temp_dir / "no-hooks"
    no_hooks.mkdir()
    try:
        run_git(["-c", f"core.hooksPath={no_hooks}", "worktree", "add", "--detach", "--quiet", str(worktree), sha], repo)
        yield worktree
    finally:
        admin_dir = _worktree_admin_dir(worktree)
        run_git(["worktree", "remove", "--force", str(worktree)], repo, check=False)
        if admin_dir is not None and admin_dir.is_dir():
            shutil.rmtree(admin_dir, ignore_errors=True)
        shutil.rmtree(temp_dir, ignore_errors=True)


def _worktree_admin_dir(worktree: Path) -> Path | None:
    """``.git/worktrees/<id>`` for ``worktree``, read from its ``.git`` file."""
    try:
        content = (worktree / ".git").read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not content.startswith("gitdir:"):
        return None
    admin_dir = Path(content.removeprefix("gitdir:").strip())
    return admin_dir if admin_dir.parent.name == "worktrees" else None
