# CounterPatch — Adversarial Regression Testing for Code Patches

**Attack your patch before users do.**

CounterPatch finds reproducible regressions in code changes. It generates adversarial inputs for the functions a patch touches, runs each input against both the **base** and the **patched** revision, and reports inputs whose behavior changed unexpectedly. Every finding comes with a minimal counterexample and a plain pytest reproduction.

It does not ask an LLM whether your patch looks correct. Generators propose inputs; runtime execution decides what actually happened.

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](pyproject.toml)

> Status: v0.1.0, alpha. Python projects on Linux and macOS. Not yet published to PyPI or the GitHub Marketplace.

## Find the bug your tests missed

A "refactor" of `withdraw()` keeps every existing test green (`3 passed`). CounterPatch disagrees (real output, trimmed):

```text
$ counterpatch check --base main --task "Refactor withdrawal validation."

✗ Regression candidate found

Function:
    withdraw(balance: int, amount: int) -> int

Minimal counterexample:
    balance = 0
    amount = 0

Base revision:
    raises ValueError('Amount must be positive')

Patched revision:
    returns 0

Also differs for:
    balance=0, amount=-1: base raises ValueError('Amount must be positive'); patch returns 1

Reproduction:
    .counterpatch/reproductions/test_withdraw_accepts_rejected_input.py  (verified: passes on base, fails on patch)
```

The same input was executed on both revisions: the base rejects it, the patch accepts it. That differential evidence is what makes it a regression candidate, and CounterPatch exits with code `1` so CI fails.

## Installation

Requires Python 3.11+, Git, and Linux or macOS.

CounterPatch is not on PyPI yet. Install it from source into the **same environment as the project you want to test**, since it imports your code:

```bash
git clone https://github.com/OWNER/counterpatch
pip install -e ./counterpatch            # deterministic mode
pip install -e "./counterpatch[ai]"      # optional: adds the Anthropic SDK for --ai
```

## Quick start

Inside a Git repository with changes on a branch (or uncommitted changes):

```bash
counterpatch check --base main
```

Describe the intended change so CounterPatch can tell deliberate behavior changes from regressions:

```bash
counterpatch check \
  --base main \
  --task "Refactor withdrawal validation"
```

To see it on a real buggy patch, run the self-contained demo. It creates a throwaway repository, shows that the existing tests pass, runs CounterPatch, and then re-runs the generated reproduction with plain pytest on both revisions:

```bash
examples/demo_project/run_demo.sh
```

All options are listed under [CLI reference](#cli-reference).

## GitHub Action

This repository is also a composite GitHub Action for pull request testing. Add `.github/workflows/counterpatch.yml` (also available as [`examples/workflows/counterpatch.yml`](examples/workflows/counterpatch.yml)):

```yaml
name: CounterPatch

on:
  pull_request:

permissions:
  contents: read

jobs:
  counterpatch:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0 # CounterPatch needs the base commit

      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"

      - run: pip install -e . # your project and its dependencies

      - uses: OWNER/counterpatch@v1
        with:
          base: ${{ github.event.pull_request.base.sha }}
          task: ${{ github.event.pull_request.title }}
```

`OWNER/counterpatch@v1` is a placeholder until the action is published.

- **Findings** appear as PR annotations and in the job summary.
- **Reproductions** are uploaded as the `counterpatch-reproductions` artifact.
- **Inputs:** `base` (defaults to the PR base SHA), `task`, `task-file`, `ai`, `ai-model`, `max-tests`, `timeout`, `call-timeout`, `fail-on-divergence`, `working-directory`, `python`, `upload-reproductions`.
- **Output:** `exit-code`.

It is a composite action on purpose: CounterPatch must import your code, so it runs in the Python environment your workflow already set up. Run `actions/setup-python` and install your project first.

## How it works

```text
              Git diff
                 │
                 ▼
         Changed functions
                 │
        ┌────────┴────────┐
        ▼                 ▼
 deterministic        optional AI
  candidates           candidates
        │                 │
        └────────┬────────┘
                 ▼
          run each candidate
           /            \
          ▼              ▼
   base revision    patched revision
   (git worktree)   (your working tree)
           \            /
            ▼          ▼
         compare behavior
                 │
                 ▼
    confirm in fresh processes
                 │
                 ▼
     shrink to a minimal input
                 │
                 ▼
        reproducible pytest
```

1. CounterPatch maps the diff onto changed functions and methods with Python's `ast`. A changed module-level constant (such as `LIMIT = 100`) also marks the functions that read it.
2. It builds candidate inputs from signatures, type annotations and the code itself.
3. Each candidate runs in a subprocess against the base revision (checked out into a temporary `git worktree`) and against your working tree. Return values, exception types, printed output and argument mutations are compared.
4. Before a difference is reported, it is re-run twice per revision in fresh processes. Nondeterministic or state-dependent differences are dropped.
5. The input is shrunk to the simplest one that still shows the same difference. It is then written out as a pytest file, which is run on both revisions. The report says whether it was verified (passes on the base, fails on the patch).

Your working tree, branches and other worktrees are never modified. The temporary checkout is removed even after errors or a cancelled CI job.

## Why base-vs-patch matters

A generated test that fails tells you almost nothing on its own. The expectation may be wrong, or the code may have behaved that way all along. CounterPatch only reports a regression when the **same input** behaves acceptably on the base revision and not on the patch:

```text
base PASS / patch PASS     → no counterexample
base PASS / patch FAIL     → regression candidate
base FAIL / patch FAIL     → non-discriminating candidate (ignored)
base FAIL / patch PASS     → possible improvement (not reported as a problem)
different valid results    → behavioral divergence
```

When CounterPatch calls a function directly, "patch FAIL" means one of the following:

- **Validation removed:** the base rejects the input with an exception it raises deliberately, and the patch accepts it.
- **New crash:** the patch raises a generic crash (`TypeError`, `IndexError`, `ZeroDivisionError`, …) where the base returned.
- **New hang:** the patch hangs or kills the interpreter where the base completed.

A changed return value is a **behavioral divergence**. It is printed as a warning (exit code 0), because it may be intended.

The task description tunes this conservatively:

- If the task says the change preserves behavior ("refactor", "cleanup", "no functional change"), divergences become regression candidates.
- If a divergence plausibly *is* the requested change, it is labeled "possibly intended". Example: accepting a 25-character username when the task says *"Allow usernames up to 32 characters"* and the diff moves the limit from 20 to 32.
- Crashes and hangs are never excused by the task.

## Deterministic mode

The core engine needs no AI, no network and no API key. For each changed function it tries values that tend to break code:

```text
integers   0, 1, -1, boundary - 1, boundary, boundary + 1, very large and very negative values
strings    "", " ", "\n", unicode, emoji, NUL, very long strings, lengths around len() limits
other      None for optionals, True/False, empty / single / duplicate / boundary-sized collections
```

Boundaries are discovered from the source code of both revisions. A comparison such as `if amount <= 0`, `if len(name) > 32` or `if amount > LIMIT` yields `-1, 0, 1`, `31, 32, 33` or `LIMIT ± 1`. Literal arguments from existing tests that call the function are reused too.

After a deterministic sweep, [Hypothesis](https://hypothesis.readthedocs.io/) explores further with strategies biased toward those boundary values. That property-based testing phase finds inputs no fixed list contains; the test suite checks that it finds a change which only triggers for `x > 5000 and x % 7 == 3`. Findings are then **shrunk**: Hypothesis' shrinker followed by a greedy pass turns something like `balance=918273, amount=-9281` into `balance=0, amount=0`. Hypothesis runs derandomized, so results are reproducible in CI.

Functions CounterPatch cannot call on its own are listed as **not automatically exercisable**, with the reason. They are never reported as tested. Examples are methods whose class needs constructor arguments, and parameters of types it cannot build.

## Optional AI mode

AI proposes additional adversarial hypotheses. It does not decide whether the patch is correct.

```bash
export ANTHROPIC_API_KEY=...
counterpatch check --ai --task "Refactor withdrawal validation"
```

CounterPatch sends the task, the changed functions (base and patch), their diff and a few related tests. It asks for concrete inputs and small pytest tests that could falsify the intended behavior.

- **Suggested inputs** go through the same differential engine, shrinking and reproduction as deterministic ones.
- **Suggested tests** can reach functions deterministic mode has to skip. They run on both revisions, and only base PASS / patch FAIL counts.
- **Model output is untrusted:** it is validated, written only under `.counterpatch/`, and executed in subprocesses.

The default model is `claude-opus-5`. Change it with `--ai-model` or `COUNTERPATCH_AI_MODEL`. Without `--ai`, no API key is needed or read.

## Reproductions

Every regression candidate is saved as an ordinary pytest file that encodes the base revision's behavior:

```python
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from banking import withdraw


def test_withdraw_accepts_rejected_input() -> None:
    with pytest.raises(ValueError):
        withdraw(0, 0)
```

```bash
pytest .counterpatch/reproductions/test_withdraw_accepts_rejected_input.py -q
```

It depends only on pytest and your code, passes on the base and fails on the patch. Move it into your test suite once the bug is fixed, or delete it if the new behavior is intended.

CounterPatch writes to `.counterpatch/`:

- `reproductions/`: the saved tests.
- `report.json`: a machine-readable report of the last run.
- `generated/`: scratch space.

The directory ships its own `.gitignore` for the report and scratch files.

## Exit codes

| Code | Meaning |
|---|---|
| `0` | No regression counterexample was found by the tests executed. Divergences may be printed as warnings. |
| `1` | Regression candidate found (or any divergence with `--fail-on-divergence`). |
| `2` | Error: configuration, Git, parsing, timeout, AI, or a changed module that no longer imports. |

A regression candidate takes precedence over errors and timeouts. Exit code `0` never means the patch is proven correct. Testing cannot prove arbitrary software correct.

## Security

**CounterPatch executes your project's code, and optionally AI-generated test code. Run it on untrusted repositories only inside isolated CI environments.**

- Subprocess isolation protects CounterPatch from crashes and hangs. **It is not a security sandbox**: code under test can access files and the network like any process you run.
- Secret-looking environment variables (`*API_KEY*`, `*TOKEN*`, `*SECRET*`, …) are removed before running target code. This is defense in depth, not a guarantee.
- In GitHub Actions, use the `pull_request` trigger with `permissions: contents: read`. **Never** run CounterPatch under `pull_request_target` with a checkout of the PR's code. That hands attacker-controlled code your secrets and a write token.
- The action passes the PR title/body through environment variables, never into shell scripts, so a PR description cannot inject commands.
- GitHub does not expose secrets to workflows from forked PRs. AI mode therefore only works for same-repository branches; deterministic mode works everywhere.

## Limitations

- **Python 3.11+ only**, on Linux and macOS (no Windows support yet).
- **Callable functions:** only functions with primitive or simply annotated arguments (`int`, `str`, `float`, `bool`, `bytes`, `None`, `Optional`, `Literal`, and lists/tuples/sets/dicts of these) are exercised automatically.
- **Methods and other skips:** instance methods need a class constructible without arguments. Properties, dunder methods and incompatible signature changes are skipped.
- **New code:** newly added functions have no base behavior to compare against.
- **Side effects:** importing a module runs its top-level code, and calls perform their side effects (files, network, databases) on both revisions.
- **Comparison:** exception messages are ignored, and floats are compared to 12 significant digits.
- **Scope of changes:** module-level changes are followed one level; functions reading a changed name are covered, callers of those functions are not. Untracked new files are not part of the diff.
- **Task heuristics:** classification of intended changes relies on numbers touched by the diff, keywords that name a parameter, and behavior-preserving wording. Prefer passing the PR title over a long PR body.

## CLI reference

```text
counterpatch check [OPTIONS]

  --base REV              Base revision (default: origin/main, main, origin/master or master)
  --task TEXT             Description of the intended change
  --task-file PATH        Read the description from a file
  --ai                    Also generate candidates with the Anthropic API (needs ANTHROPIC_API_KEY)
  --ai-model MODEL        Model for --ai (env: COUNTERPATCH_AI_MODEL)
  --max-tests N           Exploration budget per changed function (default 100; shrinking is extra)
  --timeout SECONDS       Overall time budget (default 600); exceeding it exits with code 2
  --call-timeout SECONDS  Time a single call may take before it counts as a hang (default 5)
  --fail-on-divergence    Exit 1 on behavioral divergences too
  --python PATH           Interpreter used to run your code (default: the one running CounterPatch)
  --repo PATH             Repository to check (default: current directory)
  -v, --verbose           Progress and extra details

counterpatch --version
counterpatch --help
```

The base is the merge-base of `--base` and `HEAD`, like `git diff main...HEAD`. The patch is your working tree, so uncommitted changes to tracked files are included.

## Architecture

For contributors, a map of the main modules in [`counterpatch/`](counterpatch/):

| Module | Responsibility |
|---|---|
| `git.py` | merge-base, diff parsing, temporary base worktree |
| `analyzer.py` | changed functions and methods, module-level changes, related tests |
| `candidates.py` | boundary discovery, value pools, Hypothesis strategies, call plans |
| `runner.py`, `_worker.py` | subprocess workers per revision (timeouts, crash recovery), pytest runs |
| `differential.py` | behavior comparison, classification, fresh-process confirmation |
| `intent.py` | task handling |
| `shrinking.py` | sweep, Hypothesis exploration, minimization |
| `reproduction.py` | pytest reproductions |
| `ai.py` | optional AI candidate generation and validation |
| `check.py`, `reporter.py`, `cli.py` | orchestration, output, command line |

## Development

```bash
python -m venv .venv && . .venv/bin/activate
pip install -e ".[dev]"
pytest -q
ruff check . && ruff format --check .
```

The suite includes integration tests that build real Git repositories with a buggy patch and run CounterPatch end to end. It needs no API key; AI tests use mocked responses.

## Roadmap

- Constructing simple object arguments (dataclasses, classes with primitive constructors)
- Stateful testing of methods through call sequences
- Running existing tests that cover changed functions against both revisions
- Optional container sandboxing for untrusted repositories
- SARIF output for GitHub code scanning
- PyPI and GitHub Marketplace releases

## Contributing

Issues and pull requests are welcome. One rule matters more than any other:

> Never classify something as a regression without differential runtime evidence.

Please add tests for behavior changes (an integration test when results change end to end), and run `pytest` and `ruff` before opening a PR.

## License

[MIT](LICENSE)
