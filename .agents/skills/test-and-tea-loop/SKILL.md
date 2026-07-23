---
name: test-and-tea-loop
description: Self-correcting test/run loop for Python & PyTorch projects using pytest or dry-run scripts. Diagnoses tracebacks, fixes source code, and re-verifies.
triggers:
  - on_artifact_generation
  - on_code_modification
capabilities:
  - terminal_execution
  - filesystem_read_write
---

# Test-and-Tea Loop

Automated run → diagnose → fix → re-verify loop for Python & PyTorch projects using `pytest` or dry-run execution scripts.

## Phase 1: Environment & Test Runner Detection
1. **Locate Test Framework**: Check for `tests/`, `pytest.ini`, or test scripts (`test_*.py`).
2. **Establish Execution Command**: Default to `pytest` or dry-run execution (e.g. `python train.py --help` or `python -m py_compile train.py`).

## Phase 2: The Self-Correction Execution Loop
Execute the test command in the local environment terminal. If execution fails:
1. **Parse Traceback Logs**: Capture stdout/stderr. Extract exact file paths, line numbers, exception types (e.g. `RuntimeError`, `ValueError`, `ModuleNotFoundError`, `AttributeError`), and stack traces.
2. **Diagnose and Fix**:
    - For **syntax/import errors**, fix module imports or parameter syntax in target `.py` files.
    - For **tensor shape / device mismatches**, inspect the corresponding PyTorch model forward pass or dataset batch collation.
    - For **logical assertion failures**, update code implementation to satisfy expected numerical or structural invariants.
3. **Recurse**: Re-run the execution command.
4. **Loop Limit**: Stop and alert the user if execution fails 3 consecutive times on the same root cause.

## Phase 3: Final Verification & Polish
Once execution completes with exit code 0:
1. **Re-Verify**: Confirm all tests pass without errors or warnings.
2. **Linting Check**: Run `ruff check` or `python -m py_compile` to ensure clean syntax.
3. **Present Status**: Report the success state concisely to the user.
