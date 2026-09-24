from __future__ import annotations

from pathlib import Path

import pytest

from counterpatch.git import (
    GitError,
    base_worktree,
    diff_python_files,
    has_uncommitted_changes,
    parse_unified_diff,
    repo_root,
    resolve_base,
    show_file,
)
from counterpatch.models import FileStatus

SAMPLE_DIFF = """\
diff --git a/src/app.py b/src/app.py
index 1111111..2222222 100644
--- a/src/app.py
+++ b/src/app.py
@@ -3 +3 @@ def f():
-    return 1
+    return 2
@@ -10,0 +11,2 @@ def g():
+    x = 1
+    y = 2
@@ -20,3 +22 @@ def h():
-    a
-    b
-    c
+    d
diff --git a/new.py b/new.py
new file mode 100644
index 0000000..3333333
--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+def added():
+    return 1
diff --git a/old.py b/old.py
deleted file mode 100644
index 4444444..0000000
--- a/old.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def removed():
-    return 1
diff --git a/pkg/before.py b/pkg/after.py
similarity index 90%
rename from pkg/before.py
rename to pkg/after.py
index 5555555..6666666 100644
--- a/pkg/before.py
+++ b/pkg/after.py
@@ -2 +2 @@
-    return "old"
+    return "new"
"""


def test_parse_unified_diff_statuses_and_paths() -> None:
    files = parse_unified_diff(SAMPLE_DIFF)
    assert [(f.old_path, f.new_path, f.status) for f in files] == [
        ("src/app.py", "src/app.py", FileStatus.MODIFIED),
        (None, "new.py", FileStatus.ADDED),
        ("old.py", None, FileStatus.DELETED),
        ("pkg/before.py", "pkg/after.py", FileStatus.RENAMED),
    ]


def test_parse_unified_diff_line_ranges() -> None:
    modified = parse_unified_diff(SAMPLE_DIFF)[0]
    assert [(h.old_start, h.old_count, h.new_start, h.new_count) for h in modified.hunks] == [
        (3, 1, 3, 1),
        (10, 0, 11, 2),
        (20, 3, 22, 1),
    ]
    assert modified.changed_new_lines() == {3, 11, 12, 22}
    assert modified.changed_old_lines() == {3, 20, 21, 22}
    assert modified.hunks[2].lines == ["-    a", "-    b", "-    c", "+    d"]


def test_parse_pure_rename_without_hunks() -> None:
    text = "diff --git a/a.py b/b.py\nsimilarity index 100%\nrename from a.py\nrename to b.py\n"
    [file_diff] = parse_unified_diff(text)
    assert (file_diff.old_path, file_diff.new_path, file_diff.status) == ("a.py", "b.py", FileStatus.RENAMED)
    assert file_diff.hunks == []


def test_diff_detects_changed_python_files_including_uncommitted(make_repo) -> None:
    repo = make_repo({"app.py": "def f():\n    return 1\n", "README.md": "hi\n", "keep.py": "x = 1\n"})
    base = repo.git("rev-parse", "HEAD").strip()
    repo.write("app.py", "def f():\n    return 2\n")
    repo.write("README.md", "changed\n")
    files = diff_python_files(repo.path, base)
    assert [f.path for f in files] == ["app.py"]


def test_diff_detects_renamed_file(make_repo) -> None:
    body = "def f():\n    return 1\n" + "".join(f"# line {i}\n" for i in range(20))
    repo = make_repo({"old_name.py": body})
    base = repo.git("rev-parse", "HEAD").strip()
    repo.git("mv", "old_name.py", "new_name.py")
    [file_diff] = diff_python_files(repo.path, base)
    assert file_diff.status is FileStatus.RENAMED
    assert (file_diff.old_path, file_diff.new_path) == ("old_name.py", "new_name.py")


def test_resolve_base_uses_merge_base(make_repo) -> None:
    repo = make_repo({"a.py": "x = 1\n"})
    fork_point = repo.git("rev-parse", "HEAD").strip()
    repo.branch("feature")
    repo.write("a.py", "x = 2\n")
    repo.commit("feature work")
    repo.git("checkout", "-q", "main")
    repo.write("b.py", "y = 1\n")
    repo.commit("main moved on")
    repo.git("checkout", "-q", "feature")
    assert resolve_base(repo.path, "main") == fork_point


def test_resolve_base_errors(make_repo, tmp_path: Path) -> None:
    repo = make_repo({"a.py": "x = 1\n"})
    with pytest.raises(GitError, match="does not exist"):
        resolve_base(repo.path, "no-such-branch")
    outside = tmp_path / "not-a-repo"
    outside.mkdir()
    with pytest.raises(GitError, match="not inside a Git repository"):
        repo_root(outside)


def test_show_file_missing_returns_none(make_repo) -> None:
    repo = make_repo({"a.py": "x = 1\n"})
    head = repo.git("rev-parse", "HEAD").strip()
    assert show_file(repo.path, head, "a.py") == "x = 1\n"
    assert show_file(repo.path, head, "missing.py") is None


def test_base_worktree_is_cleaned_up_and_working_tree_untouched(make_repo) -> None:
    repo = make_repo({"a.py": "x = 1\n"})
    base = repo.git("rev-parse", "HEAD").strip()
    repo.write("a.py", "x = 2  # uncommitted\n")
    assert has_uncommitted_changes(repo.path)

    with base_worktree(repo.path, base) as workspace:
        created = workspace
        assert (workspace / "a.py").read_text() == "x = 1\n"
        assert len(repo.git("worktree", "list").splitlines()) == 2

    assert not created.exists()
    assert len(repo.git("worktree", "list").splitlines()) == 1
    assert (repo.path / "a.py").read_text() == "x = 2  # uncommitted\n"
    assert repo.git("branch", "--show-current").strip() == "main"


def test_base_worktree_cleanup_after_exception(make_repo) -> None:
    repo = make_repo({"a.py": "x = 1\n"})
    base = repo.git("rev-parse", "HEAD").strip()
    with pytest.raises(RuntimeError), base_worktree(repo.path, base) as workspace:
        created = workspace
        raise RuntimeError("boom")
    assert not created.exists()
    assert len(repo.git("worktree", "list").splitlines()) == 1


def test_base_worktree_preserves_other_worktrees_and_skips_hooks(make_repo, tmp_path: Path) -> None:
    repo = make_repo({"a.py": "x = 1\n"})
    base = repo.git("rev-parse", "HEAD").strip()
    user_worktree = tmp_path / "user-worktree"
    repo.git("worktree", "add", "-q", "--detach", str(user_worktree), base)
    user_worktree.rename(tmp_path / "moved-away")  # e.g. on an unmounted disk: "prunable"
    hook = repo.path / ".git" / "hooks" / "post-checkout"
    marker = tmp_path / "hook-ran"
    hook.write_text(f"#!/bin/sh\ntouch {marker}\n")
    hook.chmod(0o755)

    with base_worktree(repo.path, base):
        pass

    assert not marker.exists(), "repository hooks must not run for the temporary checkout"
    assert len(repo.git("worktree", "list").splitlines()) == 2, "the user's other worktree metadata must survive"
