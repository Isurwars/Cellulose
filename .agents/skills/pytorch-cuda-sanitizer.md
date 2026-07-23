---
name: pytorch-cuda-sanitizer
description: Instruments the PyTorch runtime with CUDA memory diagnostics, autograd anomaly detection, and tensor shape checkers to catch NaN gradients and memory leaks.
triggers:
  - post_compilation
  - pre_test_execution
capabilities:
  - terminal_execution
  - filesystem_read_write
  - flag_injection
dependencies:
  - test-and-tea-loop
---

# PyTorch & CUDA Sanitizer Validator

This skill adds runtime memory, gradient safety, and shape checks to your agentic PyTorch workflow. It instruments training/test execution, monitors CUDA memory allocation, traps Autograd anomalies, and isolates NaN loss propagation.

## Execution Pipeline

### Phase 1: Environment & Diagnostic Instrumentation
1. **Detect CUDA Device**: Check GPU availability (`torch.cuda.is_available()`).
2. **Inject Debug Flags**: Enable PyTorch anomaly detection and synchronous CUDA launches when debugging runtime faults:
    ```python
    import torch
    torch.autograd.set_detect_anomaly(True)
    ```
    Or via environment variables:
    ```bash
    CUDA_LAUNCH_BLOCKING=1 PYTHONWARNINGS=default python train.py
    ```

### Phase 2: Runtime Execution & Trap Capture
1. **Execute Test/Training Binaries**: Run target scripts under the instrumented environment.
2. **Capture Diagnostics & Intercept Faults**:
    - **`RuntimeError: CUDA out of memory`**: Analyze batch size, gradient accumulation, or un-detached computation graphs.
    - **`RuntimeError: Function '...' returned nan values`**: Intercept Autograd backprop failures.
    - **`RuntimeError: Expected all tensors to be on the same device`**: Pinpoint CPU/GPU device mismatches.
    - **`IndexError / ShapeMismatch`**: Intercept tensor dim misalignment in scatter ops or pooling.

### Phase 3: Root-Cause Remediation & Resolution Loop
If a runtime error or warning trap is captured:
1. **Parse the Traceback**: Extract exact Python script name, line number, module name, and stack frame.
2. **Apply Fixes**:
    - **For CUDA OOM**: Add `.detach()` / `.item()` calls for logged metrics, or wrap inference in `torch.no_grad()`.
    - **For Autograd NaN**: Add numerical stability clamps (`torch.clamp(x, min=1e-8)`) or max-shifted log-softmax operations.
    - **For Device Mismatches**: Ensure `.to(device)` is applied to all input tensors, target labels, and module parameters.
3. **Re-Validate**: Re-run Phase 2 until execution completes cleanly with zero uncaught exceptions.
