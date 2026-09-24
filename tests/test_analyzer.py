from __future__ import annotations

from pathlib import Path

from counterpatch.analyzer import (
    analyze_changes,
    collect_functions,
    find_related_tests,
    is_test_path,
    literal_calls_in_tests,
)
from counterpatch.git import diff_python_files
from counterpatch.utils import module_for_path

SOURCE = """
import functools


def top(a: int, b: str = "x", *args, c: bool = False, **kwargs) -> int:
    def helper():
        return 1
    return a


async def fetch(url: str) -> str:
    return url


class Account:
    def __init__(self) -> None:
        self.balance = 0

    def deposit(self, amount: int) -> None:
        self.balance += amount

    @staticmethod
    def validate(amount: int) -> bool:
        return amount > 0

    @classmethod
    def create(cls, owner: str) -> "Account":
        return cls()

    @property
    def is_empty(self) -> bool:
        return self.balance == 0

    @functools.cache
    def cached(self, x: int) -> int:
        return x

    class Inner:
        def method(self, value: int) -> int:
            return value
"""


def test_collect_functions_finds_functions_methods_and_kinds() -> None:
    functions = collect_functions(SOURCE)
    assert set(functions) == {
        "top",
        "fetch",
        "Account.__init__",
        "Account.deposit",
        "Account.validate",
        "Account.create",
        "Account.is_empty",
        "Account.cached",
        "Account.Inner.method",
    }
    assert "helper" not in functions
    assert functions["top"].method_kind == "function"
    assert functions["Account.deposit"].method_kind == "instance"
    assert functions["Account.validate"].method_kind == "static"
    assert functions["Account.create"].method_kind == "class"
    assert functions["Account.is_empty"].is_property
    assert functions["fetch"].is_async
    assert functions["Account.Inner.method"].class_name == "Account.Inner"
    assert functions["Account.cached"].start_line == functions["Account.cached"].node.lineno - 1


def test_collect_functions_parameters_and_signature() -> None:
    top = collect_functions(SOURCE)["top"]
    assert [(p.name, p.kind, p.annotation, p.default) for p in top.params] == [
        ("a", "positional_or_keyword", "int", None),
        ("b", "positional_or_keyword", "str", "'x'"),
        ("args", "var_positional", None, None),
        ("c", "keyword_only", "bool", "False"),
        ("kwargs", "var_keyword", None, None),
    ]
    assert top.signature == "top(a: int, b: str='x', *args, c: bool=False, **kwargs) -> int"


def _analyze(repo):
    base = repo.git("rev-parse", "main").strip()
    return analyze_changes(repo.path, base, diff_python_files(repo.path, base))


def test_analyze_changes_maps_diff_to_functions_and_methods(patched_repo) -> None:
    base_source = """
    def unchanged(x: int) -> int:
        return x


    def changed(x: int) -> int:
        return x + 1


    class Shop:
        def price(self, amount: int) -> int:
            return amount * 2

        def other(self) -> None:
            pass
    """
    patch_source = base_source.replace("return x + 1", "return x + 2").replace("amount * 2", "amount * 3")
    repo = patched_repo({"shop.py": base_source}, {"shop.py": patch_source})
    analysis = _analyze(repo)
    assert [f.qualified_name for f in analysis.changed_functions] == ["changed", "Shop.price"]
    changed = analysis.changed_functions[0]
    assert changed.status == "modified"
    assert "-    return x + 1" in changed.diff and "+    return x + 2" in changed.diff
    assert changed.signature == "changed(x: int) -> int"


def test_analyze_changes_added_and_deleted_functions(patched_repo) -> None:
    repo = patched_repo(
        {"mod.py": "def old(x: int) -> int:\n    return x\n"},
        {"mod.py": "def new(x: int) -> int:\n    return x\n"},
    )
    statuses = {f.qualified_name: f.status for f in _analyze(repo).changed_functions}
    assert statuses == {"new": "added", "old": "deleted"}


def test_analyze_changes_deletion_inside_function(patched_repo) -> None:
    repo = patched_repo(
        {"mod.py": "def f(x: int) -> int:\n    if x < 0:\n        raise ValueError\n    return x\n"},
        {"mod.py": "def f(x: int) -> int:\n    return x\n"},
    )
    [changed] = _analyze(repo).changed_functions
    assert changed.qualified_name == "f"


def test_analyze_changes_new_file_syntax_error_and_tests(patched_repo) -> None:
    repo = patched_repo(
        {"ok.py": "x = 1\n"},
        {
            "added.py": "def fresh(v: int) -> int:\n    return v\n",
            "broken.py": "def oops(:\n",
            "tests/test_ok.py": "def test_x():\n    assert True\n",
        },
    )
    analysis = _analyze(repo)
    assert [(f.qualified_name, f.status) for f in analysis.changed_functions] == [("fresh", "added")]
    assert analysis.test_files == ["tests/test_ok.py"]
    assert len(analysis.parse_errors) == 1 and analysis.parse_errors[0].startswith("broken.py")


def test_analyze_changes_without_python_changes(patched_repo) -> None:
    repo = patched_repo({"README.md": "a\n"}, {"README.md": "b\n"})
    analysis = _analyze(repo)
    assert analysis.python_files == [] and analysis.changed_functions == []


def test_is_test_path() -> None:
    assert is_test_path("tests/test_a.py")
    assert is_test_path("pkg/a_test.py")
    assert is_test_path("conftest.py")
    assert is_test_path(".counterpatch/reproductions/test_x.py")
    assert not is_test_path("src/testing_utils.py")


def test_module_for_path_layouts(tmp_path: Path) -> None:
    (tmp_path / "pkg" / "sub").mkdir(parents=True)
    (tmp_path / "pkg" / "__init__.py").write_text("")
    (tmp_path / "pkg" / "sub" / "__init__.py").write_text("")
    assert module_for_path(tmp_path, "pkg/sub/mod.py") == (".", "pkg.sub.mod")
    assert module_for_path(tmp_path, "src/banking.py") == ("src", "banking")
    assert module_for_path(tmp_path, "top.py") == (".", "top")


def test_related_tests_and_literal_calls(tmp_path: Path) -> None:
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_banking.py").write_text(
        "from banking import withdraw\n\n"
        "def test_a():\n    assert withdraw(100, 30) == 70\n    withdraw(amount=5, balance=1)\n    withdraw(x, 2)\n"
    )
    (tmp_path / "tests" / "test_other.py").write_text("def test_b():\n    pass\n")
    related = find_related_tests(tmp_path, "banking", "src/banking.py")
    assert [path.name for path in related] == ["test_banking.py"]
    calls = literal_calls_in_tests(related[0].read_text(), "withdraw")
    assert calls == [([100, 30], {}), ([], {"amount": 5, "balance": 1})]
