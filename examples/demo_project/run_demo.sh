#!/usr/bin/env bash
# CounterPatch demo: the existing test suite passes on a buggy "refactor",
# but CounterPatch finds a minimal input that behaves differently.
#
# Usage: examples/demo_project/run_demo.sh
# Requires `counterpatch` and `pytest` on PATH (pip install -e . from the repo root).
set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORK_DIR="$(mktemp -d "${TMPDIR:-/tmp}/counterpatch-demo.XXXXXX")"
trap 'rm -rf "$WORK_DIR"' EXIT

step() { printf '\n\033[1m$ %s\033[0m\n' "$*"; }

cp -R "$DEMO_DIR/src" "$DEMO_DIR/tests" "$DEMO_DIR/pyproject.toml" "$WORK_DIR/"
cd "$WORK_DIR"
git init -q -b main
git -c user.name=demo -c user.email=demo@example.com add .
git -c user.name=demo -c user.email=demo@example.com commit -q -m "Add banking module"

git checkout -q -b refactor-withdraw
cp "$DEMO_DIR/patches/banking.py" src/banking.py
git -c user.name=demo -c user.email=demo@example.com commit -q -am "Refactor withdrawal validation"

step "git diff main -- src/banking.py"
git --no-pager diff main -- src/banking.py

step "pytest -q"
python -m pytest -q -p no:cacheprovider

step 'counterpatch check --base main --task "Refactor withdrawal validation."'
set +e
counterpatch check --base main --task "Refactor withdrawal validation."
status=$?
set -e

REPRO=.counterpatch/reproductions/test_withdraw_accepts_rejected_input.py
if [[ -f "$REPRO" ]]; then
  step "cat $REPRO"
  cat "$REPRO"

  # Independent proof, without CounterPatch: plain pytest on both revisions.
  step "pytest $REPRO -q   # on the patch: expected to FAIL"
  if python -m pytest -q -p no:cacheprovider "$REPRO"; then
    echo "unexpected: the reproduction passed on the patch" >&2
    exit 3
  fi

  step "git checkout main -- src/banking.py && pytest $REPRO -q   # on the base: expected to PASS"
  git checkout -q main -- src/banking.py
  python -m pytest -q -p no:cacheprovider "$REPRO"
  git checkout -q refactor-withdraw -- src/banking.py
fi

printf '\ncounterpatch exit code: %s\n' "$status"
exit "$status"
