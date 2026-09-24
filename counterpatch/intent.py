"""Task/PR description handling.

The task description is context, not truth. It is used in two conservative ways:

* a divergence whose input is plausibly *about* the requested change (it touches a
  number or concept the task mentions) is downgraded to a "behavioral divergence";
* a task that declares a behavior-preserving change (refactor, cleanup, "no
  functional change") upgrades unexplained divergences to regression candidates.
"""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from counterpatch.models import ConfigError

MAX_TASK_CHARS = 20_000
_NUMBER = re.compile(r"(?<![#\w.])-?\d+(?:\.\d+)?(?!\w)")
_PRESERVING = [
    r"\brefactor",
    r"\bclean ?up\b",
    r"\bcleanups?\b",
    r"\bsimplif",
    r"\brestructur",
    r"\breorgani[sz]",
    r"\brename",
    r"\btidy",
    r"\bno (functional|behaviou?r(al)?) changes?",
    r"\bbehaviou?r[- ]preserving",
    r"\bwithout changing (the |any )?behaviou?r",
    r"\boptimi[sz]",
    r"\bperformance\b",
    r"\bspeed ?up\b",
]


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _non_ascii(value: Any) -> bool:
    return isinstance(value, str) and any(ord(character) > 127 for character in value)


_KEYWORD_CHECKS: dict[str, Callable[[Any], bool]] = {
    "empty": lambda value: isinstance(value, (str, bytes, list, tuple, dict, set)) and len(value) == 0,
    "blank": lambda value: isinstance(value, str) and value.strip() == "",
    "whitespace": lambda value: isinstance(value, str) and value != "" and value.strip() != value,
    "negative": lambda value: _is_number(value) and value < 0,
    "zero": lambda value: _is_number(value) and value == 0,
    "none": lambda value: value is None,
    "null": lambda value: value is None,
    "optional": lambda value: value is None,
    "missing": lambda value: value is None,
    "unicode": _non_ascii,
    "emoji": _non_ascii,
    "non-ascii": _non_ascii,
    "float": lambda value: isinstance(value, float),
    "decimal": lambda value: isinstance(value, float),
    "nan": lambda value: isinstance(value, float) and math.isnan(value),
    "infinity": lambda value: isinstance(value, float) and math.isinf(value),
    "newline": lambda value: isinstance(value, str) and "\n" in value,
    "case": lambda value: isinstance(value, str) and value != value.lower(),
    "duplicate": lambda value: isinstance(value, (list, tuple)) and len(value) != len({repr(item) for item in value}),
    "duplicates": lambda value: isinstance(value, (list, tuple)) and len(value) != len({repr(item) for item in value}),
}


def load_task(task: str | None, task_file: Path | None) -> str | None:
    """Combine ``--task`` and ``--task-file`` into one description (or ``None``)."""
    parts: list[str] = []
    if task and task.strip():
        parts.append(task.strip())
    if task_file is not None:
        if not task_file.is_file():
            raise ConfigError(f"--task-file {task_file} does not exist")
        try:
            text = task_file.read_text(encoding="utf-8", errors="replace").strip()
        except OSError as error:
            raise ConfigError(f"could not read --task-file {task_file}: {error}") from error
        if text:
            parts.append(text)
    if not parts:
        return None
    return "\n\n".join(parts)[:MAX_TASK_CHARS]


@dataclass
class Intent:
    text: str | None
    numbers: set[float] = field(default_factory=set)
    keywords: set[str] = field(default_factory=set)
    preserves_behavior: bool = False

    @classmethod
    def from_text(cls, text: str | None) -> Intent:
        if not text:
            return cls(text=None)
        lowered = text.lower()
        numbers = {float(match) for match in _NUMBER.findall(text)}
        keywords = {keyword for keyword in _KEYWORD_CHECKS if re.search(rf"\b{re.escape(keyword)}\b", lowered)}
        preserves = any(re.search(pattern, lowered) for pattern in _PRESERVING)
        return cls(text=text, numbers=numbers, keywords=keywords, preserves_behavior=preserves)

    def relatedness(self, args: dict[str, Any], diff: DiffNumbers, names: set[str]) -> str | None:
        """Explain why ``args`` plausibly concern the requested change, or return ``None``.

        Deliberately narrow, because a match hides a finding from the exit code:

        * a number from the task only counts if the function's diff adds or removes that
          literal (so "fixes ticket 1" does not make every input near 1 "intended");
          inputs within 1 of it, or between an old limit removed by the diff and it, match;
        * a keyword ("empty", "negative"...) only counts if the task also names the
          parameter or the function (``names``).
        """
        if not self.text:
            return None
        eligible = sorted(number for number in self.numbers if number in diff.added | diff.removed)
        mentioned = self._mentions(names)
        for name, value in args.items():
            measure, label = _measure(name, value)
            if measure is not None:
                for number in eligible:
                    if abs(measure - number) <= 1:
                        return f"the task mentions {_fmt(number)} and {label} = {_fmt(measure)}"
                    for limit in sorted(diff.removed - {number}):
                        if min(limit, number) - 1 <= measure <= max(limit, number) + 1:
                            return f"{label} = {_fmt(measure)} lies between the old limit {_fmt(limit)} and the task's {_fmt(number)}"
            if not (mentioned & {name.lower()} or mentioned - {arg.lower() for arg in args}):
                continue
            for keyword in sorted(self.keywords):
                try:
                    matches = _KEYWORD_CHECKS[keyword](value)
                except Exception:  # noqa: BLE001 - defensive against odd values
                    matches = False
                if matches:
                    return f"the task mentions {keyword!r} and {name} = {value!r:.60}"
        return None

    def _mentions(self, names: set[str]) -> set[str]:
        """Which of ``names`` (parameter/function names) the task refers to, allowing plurals."""
        words = set(re.findall(r"[a-z0-9_]+", (self.text or "").lower()))
        found = set()
        for name in names:
            lowered = name.lower()
            if lowered in words or f"{lowered}s" in words or (lowered.endswith("s") and lowered[:-1] in words):
                found.add(lowered)
        return found


@dataclass
class DiffNumbers:
    """Numeric literals on removed (old limits) and added (new limits) lines of a function diff."""

    removed: set[float] = field(default_factory=set)
    added: set[float] = field(default_factory=set)

    @classmethod
    def from_diff(cls, diff: str) -> DiffNumbers:
        numbers = cls()
        for line in diff.splitlines():
            if line[:1] not in {"-", "+"} or line.startswith(("---", "+++")):
                continue
            code = line[1:].split("#", 1)[0]
            found = {float(match) for match in re.findall(r"(?<![\w.])\d+(?:\.\d+)?(?![\w.])", code)}
            (numbers.removed if line.startswith("-") else numbers.added).update(found)
        return numbers


def _measure(name: str, value: Any) -> tuple[float | None, str]:
    if _is_number(value):
        return (float(value) if math.isfinite(value) else None), name
    if isinstance(value, (str, bytes, list, tuple, dict, set)):
        return float(len(value)), f"len({name})"
    return None, name


def _fmt(number: float) -> str:
    return str(int(number)) if float(number).is_integer() else f"{number:g}"
