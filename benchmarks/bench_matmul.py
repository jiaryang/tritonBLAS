#!/usr/bin/env python3
"""
TritonBLAS Matrix Multiplication Benchmark with Accuracy Validation

This benchmark compares different matrix multiplication implementations:
- torch.matmul (baseline reference)
- torch.compile (Inductor Triton kernels)
- TritonBLAS (Stream-K implementation)

Features:
- Performance comparison (TFLOPS, latency)
- Numerical accuracy validation against torch.matmul reference
- Support for various data types (fp16, fp32, bf16, int8)
- CSV export for analysis
- Configurable accuracy tolerances

Usage Examples:
  # Basic performance benchmark
  python bench_matmul.py --input-yaml config.yaml --print-verbose

  # With accuracy checking
  python bench_matmul.py --input-yaml config.yaml --check-accuracy --print-verbose

  # Compare all implementations with accuracy
  python bench_matmul.py --input-yaml config.yaml --check-accuracy \\
    --torch-compile-modes all --tritonblas-modes all \\
    --output-csv results.csv --print-verbose

  # Custom accuracy tolerance for fp16
  python bench_matmul.py --input-yaml config.yaml --check-accuracy \\
    --accuracy-tolerance 1e-2 --tritonblas-modes streamk

  # Disable torch.matmul baseline (useful for torch-free environments)
  python bench_matmul.py --input-yaml config.yaml --disable-torch-matmul \\
    --tritonblas-modes persistent --print-verbose
"""

import argparse
import csv
import gc
import os
import random
import shutil
from collections import OrderedDict
from pathlib import Path

import triton
import tritonblas
import torch
import yaml
from tqdm import tqdm

# ============================================================
# Helper functions
# ============================================================

def setup_torch_compile_config():
    import torch._inductor.config
    import torch._dynamo.config
    torch._inductor.config.triton.unique_kernel_names = True
    torch._inductor.config.triton.unique_user_kernel_names = True
    torch._inductor.config.coordinate_descent_tuning = True
    torch._inductor.config.freezing = True
    torch._inductor.config.max_autotune = True
    torch._dynamo.config.recompile_limit = 256


def clear_torch_compile_cache(print_verbose=False):
    """Clear torch.compile in-memory and on-disk caches before a benchmark pass."""
    try:
        import torch._dynamo
        torch._dynamo.reset()
    except Exception as exc:
        if print_verbose:
            print(f"WARNING: torch._dynamo.reset() failed: {exc}")

    try:
        from torch._inductor.utils import clear_inductor_caches
        clear_inductor_caches()
    except Exception as exc:
        if print_verbose:
            print(f"WARNING: torch._inductor cache reset failed: {exc}")

    cache_paths = []
    try:
        from torch._inductor.runtime.cache_dir_utils import cache_dir, triton_cache_dir
        cache_paths.append(Path(cache_dir()))
        if torch.cuda.is_available():
            cache_paths.append(Path(triton_cache_dir(torch.cuda.current_device())))
    except Exception as exc:
        if print_verbose:
            print(f"WARNING: could not resolve torch.compile cache dirs: {exc}")

    for env_name in ("TORCHINDUCTOR_CACHE_DIR", "TRITON_CACHE_DIR"):
        env_path = os.environ.get(env_name)
        if env_path:
            cache_paths.append(Path(env_path))

    seen = set()
    for path in sorted(cache_paths, key=lambda p: len(p.parts), reverse=True):
        if path in seen:
            continue
        seen.add(path)
        if path.exists():
            if print_verbose:
                print(f"Clearing torch.compile cache: {path}")
            shutil.rmtree(path, ignore_errors=True)


class BenchmarkRunError(RuntimeError):
    def __init__(self, message, partial_results, original_exception):
        super().__init__(message)
        self.partial_results = list(partial_results)
        self.original_exception = original_exception


def str_to_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.replace("torch.", "")
    return getattr(torch, dtype_str)


def init_by_size_and_type(size, dtype, init_type):
    if init_type == "hpl":
        return torch.empty(size, device="cuda", dtype=dtype).uniform_(-0.5, 0.5)
    elif init_type == "trig_float":
        M, N = size
        return torch.arange(0, M * N, device="cuda", dtype=torch.float32).reshape(M, N).sin().to(dtype=dtype)
    elif init_type == "zeros":
        return torch.zeros(size, dtype=dtype, device="cuda")
    elif init_type == "randn":
        return torch.randn(size, dtype=torch.float32, device="cuda").to(dtype)
    elif init_type == "increasing":
        # Create tensor with values increasing from 0 to M-1 along rows and 0 to N-1 along columns
        # Each element (i, j) = i + j (normalized to avoid overflow for large matrices)
        M, N = size
        row_indices = torch.arange(0, M, device="cuda", dtype=torch.float32).unsqueeze(1)  # Shape: (M, 1)
        col_indices = torch.arange(0, N, device="cuda", dtype=torch.float32).unsqueeze(0)  # Shape: (1, N)
        result = (row_indices + col_indices) / max(M, N)  # Normalize to avoid large values
        return result.to(dtype=dtype)
    else:
        raise ValueError(f"Unsupported init_type: {init_type}")


def print_tensor_debug(name, tensor, max_elements=16):
    """
    Print tensor info and sample values for debugging.

    Args:
        name: str - Name of the tensor for display
        tensor: torch.Tensor - Tensor to print
        max_elements: int - Maximum number of elements to show
    """
    print(f"\n{'='*60}")
    print(f"  {name}:")
    print(f"    Shape: {tensor.shape}, dtype: {tensor.dtype}")
    print(f"    Min: {tensor.min().item():.6f}, Max: {tensor.max().item():.6f}, Mean: {tensor.float().mean().item():.6f}")

    # Print corner values for visualization
    m, n = tensor.shape
    rows_to_show = min(4, m)
    cols_to_show = min(8, n)

    print(f"    Top-left {rows_to_show}x{cols_to_show} corner:")
    for i in range(rows_to_show):
        row_vals = [f"{tensor[i, j].item():10.4f}" for j in range(cols_to_show)]
        suffix = " ..." if n > cols_to_show else ""
        print(f"      [{', '.join(row_vals)}{suffix}]")
    if m > rows_to_show:
        print(f"      ... ({m - rows_to_show} more rows)")
    print(f"{'='*60}")


def print_differences(reference, test, max_diffs=20):
    """
    Print only the positions where reference and test differ.

    Args:
        reference: torch.Tensor - Reference result
        test: torch.Tensor - Test result to compare
        max_diffs: int - Maximum number of differences to print
    """
    diff = torch.abs(reference - test)
    # Find non-zero differences
    nonzero_mask = diff > 0
    nonzero_indices = torch.nonzero(nonzero_mask)

    num_diffs = nonzero_indices.shape[0]
    print(f"\n{'='*60}")
    print(f"  DIFFERENCES FOUND: {num_diffs} elements differ")
    print(f"{'='*60}")

    if num_diffs == 0:
        print("  No differences found!")
        return

    # Sort by error magnitude (largest first)
    diff_values = diff[nonzero_mask]
    sorted_indices = torch.argsort(diff_values, descending=True)

    print(f"  Showing top {min(max_diffs, num_diffs)} differences (sorted by magnitude):")
    print(f"  {'Index':<15} {'Reference':<15} {'Test':<15} {'Abs Diff':<15} {'Rel Diff':<15}")
    print(f"  {'-'*75}")

    for i in range(min(max_diffs, num_diffs)):
        idx = sorted_indices[i]
        pos = nonzero_indices[idx]
        row, col = pos[0].item(), pos[1].item()
        ref_val = reference[row, col].item()
        test_val = test[row, col].item()
        abs_diff = diff[row, col].item()
        rel_diff = abs_diff / abs(ref_val) if abs(ref_val) > 1e-8 else abs_diff

        print(f"  [{row:4d},{col:4d}]     {ref_val:<15.4f} {test_val:<15.4f} {abs_diff:<15.4f} {rel_diff:<15.6f}")

    if num_diffs > max_diffs:
        print(f"  ... and {num_diffs - max_diffs} more differences")
    print(f"{'='*60}")


def check_accuracy(reference_result, test_result, impl_name, tolerance=None, relative_tolerance=None):
    """
    Check numerical accuracy between reference and test implementations.

    Uses torch.testing.assert_close() style comparison with dtype-appropriate tolerances.
    This matches the tritonblas_matmul.py reference implementation.

    Tolerance guidelines (from tritonblas_matmul.py):
    - bf16/fp16: atol=0.5, rtol=0.05
    - fp32: atol=1e-2, rtol=1e-3
    - fp8/int8 (quantized): atol=2.0, rtol=0.2

    Args:
        reference_result: torch.Tensor - Reference result from torch.matmul
        test_result: torch.Tensor - Result from implementation being tested
        impl_name: str - Name of implementation for logging
        tolerance: float - Absolute tolerance (atol). If None, uses dtype-based default.
        relative_tolerance: float - Relative tolerance (rtol). If None, uses dtype-based default.

    Returns:
        tuple: (is_accurate: bool, max_abs_error: float, max_rel_error: float)
    """
    if reference_result.shape != test_result.shape:
        print(f"❌ {impl_name}: Shape mismatch! Reference: {reference_result.shape}, Test: {test_result.shape}")
        return False, float('inf'), float('inf')

    # Determine dtype-appropriate tolerances if not specified
    # Based on tritonblas_matmul.py reference implementation
    dtype = reference_result.dtype
    if tolerance is None or relative_tolerance is None:
        if dtype in (torch.float16, torch.bfloat16):
            default_atol, default_rtol = 0.5, 0.05
        elif dtype == torch.float32:
            default_atol, default_rtol = 1e-2, 1e-3
        elif dtype in (torch.int8, torch.uint8):
            default_atol, default_rtol = 2.0, 0.2
        elif hasattr(torch, 'float8_e4m3fn') and dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            default_atol, default_rtol = 2.0, 0.2
        else:
            default_atol, default_rtol = 0.5, 0.05

        if tolerance is None:
            tolerance = default_atol
        if relative_tolerance is None:
            relative_tolerance = default_rtol

    # Convert to float32 for comparison (matches tritonblas_matmul.py)
    ref_f32 = reference_result.to(torch.float32)
    test_f32 = test_result.to(torch.float32)

    # Calculate absolute error
    abs_error = torch.abs(ref_f32 - test_f32)
    max_abs_error = torch.max(abs_error).item()

    # Calculate relative error only for elements with significant magnitude
    ref_abs = torch.abs(ref_f32)
    magnitude_threshold = 1.0
    significant_mask = ref_abs > magnitude_threshold

    if significant_mask.any():
        rel_error_significant = abs_error[significant_mask] / ref_abs[significant_mask]
        max_rel_error = torch.max(rel_error_significant).item()
    else:
        max_rel_error = 0.0

    # Use torch.testing.assert_close style check
    # This is what tritonblas_matmul.py uses for verification
    try:
        torch.testing.assert_close(
            test_f32, ref_f32,
            atol=tolerance, rtol=relative_tolerance
        )
        is_accurate = True
    except AssertionError:
        is_accurate = False

    return is_accurate, max_abs_error, max_rel_error


def benchmark_shape_key(row):
    return (
        row.get("m"),
        row.get("n"),
        row.get("k"),
        row.get("transA"),
        row.get("transB"),
        row.get("in_dtype"),
        row.get("out_dtype"),
    )


def fill_speedups_vs_torch(results):
    torch_by_shape = {
        benchmark_shape_key(row): row.get("torch_tflops")
        for row in results
        if row.get("torch_tflops")
    }

    for row in results:
        torch_tflops = torch_by_shape.get(benchmark_shape_key(row))
        if not torch_tflops:
            continue
        compile_tflops = row.get("torch_compile_tflops")
        triton_tflops = row.get("tritonblas_tflops")
        if compile_tflops:
            row["speedup_compile_vs_torch"] = compile_tflops / torch_tflops
        if triton_tflops:
            row["speedup_triton_vs_torch"] = triton_tflops / torch_tflops


def format_vs_torch(speedup):
    if speedup is None:
        return "N/A vs torch"
    percent = (speedup - 1.0) * 100.0
    return f"{speedup:.2f}x, {percent:+.1f}% vs torch"


def format_accuracy(status):
    if status is None:
        return ""
    return f", {'✅' if status else '❌'}acc"


def format_selected_streamk(status):
    if status is None:
        return ""
    return f", streamk={'yes' if status else 'no'}"


def print_benchmark_summary(results):
    grouped_rows = OrderedDict()
    for row in results:
        grouped_rows.setdefault(benchmark_shape_key(row), []).append(row)

    for case_idx, ((m, n, k, transA, transB, in_dtype, _), rows) in enumerate(grouped_rows.items(), start=1):
        print(f"\nTest {case_idx}: [M={m},N={n},K={k},trans={transA}{transB}] dtype={in_dtype}")

        for row in rows:
            if row.get("torch_tflops"):
                print(
                    f"  torch.matmul: {row['torch_tflops']:.3f} TF/s "
                    f"({row['ms_torch']:.2f} ms, 100.0%)"
                )

        for row in rows:
            if row.get("torch_compile_tflops"):
                mode = row.get("torch_compile_mode") or "default"
                print(
                    f"  torch.compile[{mode}]: {row['torch_compile_tflops']:.3f} TF/s "
                    f"({row['ms_compile']:.2f} ms, "
                    f"{format_vs_torch(row.get('speedup_compile_vs_torch'))}"
                    f"{format_selected_streamk(row.get('torch_compile_selected_streamk'))}"
                    f"{format_accuracy(row.get('accuracy_compile'))})"
                )

        for row in rows:
            if row.get("tritonblas_tflops"):
                mode = row.get("tritonblas_mode") or "tritonblas"
                print(
                    f"  TritonBLAS[{mode}]: {row['tritonblas_tflops']:.3f} TF/s "
                    f"({row['ms_triton']:.2f} ms, "
                    f"{format_vs_torch(row.get('speedup_triton_vs_torch'))}"
                    f"{format_accuracy(row.get('accuracy_triton'))})"
                )


def resolve_csv_path(output_csv):
    if output_csv:
        path = Path(output_csv)
        if path.suffix.lower() != ".csv":
            path = path.with_suffix(".csv")
        return path
    return Path("benchmark_results.csv")


def csv_mode_name(mode):
    return str(mode).replace(".", "_").replace("-", "_")


REPORT_STREAMK_SELECTION_MODES = {"streamk", "streamk_tuning"}


def merged_benchmark_rows(results, include_error_details=False):
    grouped_rows = OrderedDict()
    compile_modes = []
    tritonblas_modes = []
    seen_compile_modes = set()
    seen_tritonblas_modes = set()

    for row in results:
        grouped_rows.setdefault(benchmark_shape_key(row), []).append(row)

        compile_mode = row.get("torch_compile_mode")
        if row.get("torch_compile_tflops") is not None and compile_mode not in seen_compile_modes:
            compile_modes.append(compile_mode or "default")
            seen_compile_modes.add(compile_mode)

        tritonblas_mode = row.get("tritonblas_mode")
        if row.get("tritonblas_tflops") is not None and tritonblas_mode not in seen_tritonblas_modes:
            tritonblas_modes.append(tritonblas_mode or "tritonblas")
            seen_tritonblas_modes.add(tritonblas_mode)

    fieldnames = [
        "m", "n", "k", "transA", "transB", "in_dtype", "out_dtype", "flops",
        "",
        "torch_tflops", "ms_torch",
    ]
    for mode in compile_modes:
        mode_name = csv_mode_name(mode)
        fieldnames.extend([
            "",
            f"torch_compile_{mode_name}_tflops",
            f"ms_torch_compile_{mode_name}",
            f"speedup_torch_compile_{mode_name}_vs_torch",
            f"accuracy_torch_compile_{mode_name}",
        ])
        if mode in REPORT_STREAMK_SELECTION_MODES:
            fieldnames.append(f"torch_compile_{mode_name}_selected_streamk")
        if include_error_details:
            fieldnames.extend([
                f"max_abs_error_torch_compile_{mode_name}",
                f"max_rel_error_torch_compile_{mode_name}",
            ])
    for mode in tritonblas_modes:
        mode_name = csv_mode_name(mode)
        fieldnames.extend([
            "",
            f"tritonblas_{mode_name}_tflops",
            f"ms_tritonblas_{mode_name}",
            f"speedup_tritonblas_{mode_name}_vs_torch",
            f"accuracy_tritonblas_{mode_name}",
        ])
        if include_error_details:
            fieldnames.extend([
                f"max_abs_error_tritonblas_{mode_name}",
                f"max_rel_error_tritonblas_{mode_name}",
            ])
    merged_rows = []
    for (m, n, k, transA, transB, in_dtype, out_dtype), rows in grouped_rows.items():
        merged = {
            "m": m, "n": n, "k": k,
            "transA": transA, "transB": transB,
            "in_dtype": in_dtype, "out_dtype": out_dtype,
        }

        for row in rows:
            if row.get("flops") is not None:
                merged["flops"] = row.get("flops")

            if row.get("torch_tflops") is not None:
                merged["torch_tflops"] = row.get("torch_tflops")
                merged["ms_torch"] = row.get("ms_torch")

            if row.get("torch_compile_tflops") is not None:
                mode_name = csv_mode_name(row.get("torch_compile_mode") or "default")
                speedup = row.get("speedup_compile_vs_torch")
                merged[f"torch_compile_{mode_name}_tflops"] = row.get("torch_compile_tflops")
                merged[f"ms_torch_compile_{mode_name}"] = row.get("ms_compile")
                merged[f"speedup_torch_compile_{mode_name}_vs_torch"] = speedup
                merged[f"accuracy_torch_compile_{mode_name}"] = row.get("accuracy_compile")
                if (row.get("torch_compile_mode") or "default") in REPORT_STREAMK_SELECTION_MODES:
                    merged[f"torch_compile_{mode_name}_selected_streamk"] = row.get(
                        "torch_compile_selected_streamk"
                    )
                if include_error_details:
                    merged[f"max_abs_error_torch_compile_{mode_name}"] = row.get("max_abs_error_compile")
                    merged[f"max_rel_error_torch_compile_{mode_name}"] = row.get("max_rel_error_compile")

            if row.get("tritonblas_tflops") is not None:
                mode_name = csv_mode_name(row.get("tritonblas_mode") or "tritonblas")
                speedup = row.get("speedup_triton_vs_torch")
                merged[f"tritonblas_{mode_name}_tflops"] = row.get("tritonblas_tflops")
                merged[f"ms_tritonblas_{mode_name}"] = row.get("ms_triton")
                merged[f"speedup_tritonblas_{mode_name}_vs_torch"] = speedup
                merged[f"accuracy_tritonblas_{mode_name}"] = row.get("accuracy_triton")
                if include_error_details:
                    merged[f"max_abs_error_tritonblas_{mode_name}"] = row.get("max_abs_error_triton")
                    merged[f"max_rel_error_tritonblas_{mode_name}"] = row.get("max_rel_error_triton")

        merged_rows.append(merged)

    return fieldnames, merged_rows


def write_csv(output_csv, results, include_error_details=False):
    path = resolve_csv_path(output_csv)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames, rows = merged_benchmark_rows(results, include_error_details=include_error_details)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"✅ CSV report saved to '{path}'")


# ============================================================
# Benchmark core
# ============================================================

def bench_matmul(
    input_yaml: str,
    init_type: str,
    print_verbose=False,
    shuffle_benchmark=True,
    enable_streamk=False,
    torch_compile=False,
    torch_compile_mode=None,
    dynamic=False,
    work_stealing=False,
    tritonblas_mode=None,
    enable_accuracy_check=False,
    accuracy_tolerance=1e-3,
    enable_torch_matmul=True,
    print_case_results=True,
    progress_label=None,
):
    with open(input_yaml, "r") as f:
        dataset = yaml.safe_load(f)

    if torch_compile:
        setup_torch_compile_config()

    benchmark_results = []

    dataset_tuples = [
        (
            case["m"], case["n"], case["k"],
            str_to_dtype(case["in_dtype"]),
            str_to_dtype(case["out_dtype"]),
            case["transA"], case["transB"],
        )
        for case in dataset
    ]
    if shuffle_benchmark:
        random.shuffle(dataset_tuples)

    count = 0

    try:
        for m, n, k, in_dtype, out_dtype, transA, transB in (
            tqdm(dataset_tuples) if not print_verbose else dataset_tuples
        ):
            # === prepare shapes ===
            if transA == "T": A_size = (m, k)
            else: A_size = (k, m)
            if transB == "T": B_size = (k, n)
            else: B_size = (n, k)

            A = init_by_size_and_type(A_size, in_dtype, init_type)
            B = init_by_size_and_type(B_size, in_dtype, init_type)
            if transA == "N": A = A.T
            if transB == "N": B = B.T
            C = torch.zeros((m, n), device="cuda", dtype=out_dtype)

            # FLOPs & TFLOPs conversion
            flops = 2 * m * n * k
            tflops = lambda ms: (flops * 1e-12) / (ms * 1e-3)

            # ------------------------------------------------------------
            # 1️⃣ Torch.matmul (baseline)
            # ------------------------------------------------------------
            ms_torch, perf_torch = None, None
            if enable_torch_matmul:
                ms_torch = triton.testing.do_bench(lambda: torch.matmul(A, B), warmup=20, rep=100)
                perf_torch = tflops(ms_torch)

            # ------------------------------------------------------------
            # 2️⃣ Torch.compile (Inductor Triton kernel)
            # ------------------------------------------------------------
            ms_compile, perf_compile = None, None
            compiled_fn = None
            torch_compile_selected_streamk = None
            torch_compile_best_kernel = None
            if torch_compile:
                os.environ.pop("TORCHINDUCTOR_LAST_AUTOTUNE_BEST_KERNEL", None)
                os.environ.pop("TORCHINDUCTOR_LAST_AUTOTUNE_EVENT", None)
                compiled_fn = torch.compile(torch.matmul, dynamic=dynamic)
                ms_compile = triton.testing.do_bench(lambda: compiled_fn(A, B), warmup=20, rep=100)
                perf_compile = tflops(ms_compile)
                torch_compile_best_kernel = os.environ.get("TORCHINDUCTOR_LAST_AUTOTUNE_BEST_KERNEL")
                if torch_compile_best_kernel:
                    torch_compile_selected_streamk = "streamk" in torch_compile_best_kernel

            # ------------------------------------------------------------
            # 3️⃣ TritonBLAS
            # ------------------------------------------------------------
            ms_triton, perf_triton = None, None
            run_tritonblas = tritonblas_mode is not None
            if run_tritonblas:
                selector = tritonblas.OrigamiMatmulSelector(
                    m, n, k, A.dtype, B.dtype, C.dtype, A.device, streamk=enable_streamk
                )
                cfg = tritonblas.matmul_preamble(selector)

                def matmul_triton():
                    tritonblas.matmul_lt(
                        A, B, C, selector, cfg,
                        enable_streamk=enable_streamk,
                        work_stealing=work_stealing,
                    )

                def reset_triton():
                    cfg.reset(streamk=enable_streamk, work_stealing=work_stealing)

                ms_triton = tritonblas.do_bench(
                    matmul_triton, reset_fn=reset_triton, n_warmup=20, n_repeat=100
                )
                perf_triton = tflops(ms_triton)

            # ------------------------------------------------------------
            # Accuracy Check (if enabled)
            # ------------------------------------------------------------
            accuracy_compile, accuracy_triton = None, None
            max_abs_error_compile, max_abs_error_triton = None, None
            max_rel_error_compile, max_rel_error_triton = None, None

            if enable_accuracy_check and (enable_torch_matmul or torch_compile or run_tritonblas):
                # Compute reference result (torch.matmul)
                reference_result = torch.matmul(A, B)

                # Check torch.compile
                if torch_compile:
                    if compiled_fn is None:
                        compiled_fn = torch.compile(torch.matmul, dynamic=dynamic)
                    compile_result = compiled_fn(A, B)
                    accuracy_compile, max_abs_error_compile, max_rel_error_compile = check_accuracy(
                        reference_result, compile_result, "torch.compile", accuracy_tolerance, accuracy_tolerance
                    )

                    # Print debug info when accuracy check fails
                    if not accuracy_compile and print_verbose:
                        print(f"\n❌ torch.compile ACCURACY FAILURE for [M={m},N={n},K={k}]")
                        # Print only the differences (most useful for debugging)
                        print_differences(reference_result, compile_result, max_diffs=50)
                        # Also print summary stats
                        diff = torch.abs(reference_result - compile_result)
                        max_idx = torch.argmax(diff)
                        max_row = max_idx // n
                        max_col = max_idx % n
                        print(f"  Max error location: [{max_row}, {max_col}]")
                        print(f"  Reference value: {reference_result[max_row, max_col].item():.6f}")
                        print(f"  torch.compile value: {compile_result[max_row, max_col].item():.6f}")
                        print(f"  Difference: {diff[max_row, max_col].item():.6e}")

                # Check TritonBLAS
                if run_tritonblas:
                    C_triton = torch.zeros((m, n), device="cuda", dtype=out_dtype)
                    selector = tritonblas.OrigamiMatmulSelector(
                        m, n, k, A.dtype, B.dtype, C_triton.dtype, A.device,
                        streamk=enable_streamk
                    )
                    cfg = tritonblas.matmul_preamble(selector)
                    tritonblas.matmul_lt(
                        A, B, C_triton, selector, cfg,
                        enable_streamk=enable_streamk,
                        work_stealing=work_stealing,
                    )
                    triton_result = C_triton
                    accuracy_triton, max_abs_error_triton, max_rel_error_triton = check_accuracy(
                        reference_result, triton_result, "TritonBLAS", accuracy_tolerance, accuracy_tolerance
                    )

                    # Print debug info when accuracy check fails
                    if not accuracy_triton and print_verbose:
                        print(f"\n❌ TritonBLAS ACCURACY FAILURE for [M={m},N={n},K={k}]")
                        # Print only the differences (most useful for debugging)
                        print_differences(reference_result, triton_result, max_diffs=50)
                        # Also print summary stats
                        diff = torch.abs(reference_result - triton_result)
                        max_idx = torch.argmax(diff)
                        max_row = max_idx // n
                        max_col = max_idx % n
                        print(f"  Max error location: [{max_row}, {max_col}]")
                        print(f"  Reference value: {reference_result[max_row, max_col].item():.6f}")
                        print(f"  TritonBLAS value: {triton_result[max_row, max_col].item():.6f}")
                        print(f"  Difference: {diff[max_row, max_col].item():.6e}")

            # ------------------------------------------------------------
            # Speedup vs Torch baseline
            # ------------------------------------------------------------
            speedup_compile = perf_compile / perf_torch if (perf_compile and perf_torch) else None
            speedup_triton = perf_triton / perf_torch if (perf_triton and perf_torch) else None

            # ------------------------------------------------------------
            # Logging
            # ------------------------------------------------------------
            if print_verbose and print_case_results:
                test_case_id = count + 1
                msg = f"Test {test_case_id}: [M={m},N={n},K={k},trans={transA}{transB}] dtype={in_dtype}"
                if perf_torch:
                    msg += f" | Torch={perf_torch:.3f} TF/s ({ms_torch:.2f} ms)"
                else:
                    msg += " | Torch=DISABLED"
                msg += " "
                if perf_compile:
                    acc_str = ""
                    if enable_accuracy_check and accuracy_compile is not None:
                        acc_str = f", {'✅' if accuracy_compile else '❌'}acc"
                    speedup_str = f"{speedup_compile:.2f}x" if speedup_compile is not None else "N/A"
                    msg += f"| compile={perf_compile:.3f} ({ms_compile:.2f} ms, {speedup_str}{acc_str}) "
                if perf_triton:
                    acc_str = ""
                    if enable_accuracy_check and accuracy_triton is not None:
                        acc_str = f", {'✅' if accuracy_triton else '❌'}acc"
                    speedup_str = f"{speedup_triton:.2f}x" if speedup_triton is not None else "N/A"
                    mode_label = tritonblas_mode or "tritonblas"
                    msg += f"| Triton[{mode_label}]={perf_triton:.3f} ({ms_triton:.2f} ms, {speedup_str}{acc_str})"
                print(msg)

                if enable_accuracy_check and any([accuracy_compile is not None, accuracy_triton is not None]):
                    acc_details = "    Accuracy Details: "
                    if accuracy_compile is not None:
                        status = "✅" if accuracy_compile else "❌"
                        acc_details += f"compile: {status} (abs_err={max_abs_error_compile:.2e}, rel_err={max_rel_error_compile:.2e}) "
                    if accuracy_triton is not None:
                        status = "✅" if accuracy_triton else "❌"
                        acc_details += f"Triton: {status} (abs_err={max_abs_error_triton:.2e}, rel_err={max_rel_error_triton:.2e}) "
                    print(acc_details)

            if print_verbose and not print_case_results:
                case_id = count + 1
                label = progress_label or "benchmark"
                msg = (
                    f"  [{case_id}/{len(dataset_tuples)}] {label}: "
                    f"[M={m},N={n},K={k},trans={transA}{transB}] dtype={in_dtype}"
                )
                if perf_torch:
                    msg += f" -> {perf_torch:.3f} TF/s ({ms_torch:.2f} ms)"
                elif perf_compile:
                    acc_str = format_accuracy(accuracy_compile) if enable_accuracy_check else ""
                    streamk_str = format_selected_streamk(torch_compile_selected_streamk)
                    msg += f" -> {perf_compile:.3f} TF/s ({ms_compile:.2f} ms{streamk_str}{acc_str})"
                elif perf_triton:
                    acc_str = format_accuracy(accuracy_triton) if enable_accuracy_check else ""
                    msg += f" -> {perf_triton:.3f} TF/s ({ms_triton:.2f} ms{acc_str})"
                else:
                    msg += " -> completed"
                print(msg, flush=True)

            # ------------------------------------------------------------
            # Record
            # ------------------------------------------------------------
            metrics = {
                "m": m, "n": n, "k": k,
                "in_dtype": str(in_dtype), "out_dtype": str(out_dtype),
                "transA": transA, "transB": transB,
                "flops": flops,
                "torch_tflops": perf_torch,
                "torch_compile_tflops": perf_compile,
                "tritonblas_tflops": perf_triton,
                "speedup_compile_vs_torch": speedup_compile,
                "speedup_triton_vs_torch": speedup_triton,
                "ms_torch": ms_torch,
                "ms_compile": ms_compile,
                "ms_triton": ms_triton,
                "torch_compile_mode": torch_compile_mode if torch_compile else None,
                "tritonblas_mode": tritonblas_mode,
                "enable_streamk": enable_streamk,
                "work_stealing": work_stealing,
                "torch_compile_dynamic": dynamic,
                "torch_compile_selected_streamk": torch_compile_selected_streamk,
                "torch_compile_best_kernel": torch_compile_best_kernel,
                "accuracy_compile": accuracy_compile,
                "accuracy_triton": accuracy_triton,
                "max_abs_error_compile": max_abs_error_compile,
                "max_abs_error_triton": max_abs_error_triton,
                "max_rel_error_compile": max_rel_error_compile,
                "max_rel_error_triton": max_rel_error_triton,
                "accuracy_tolerance": accuracy_tolerance if enable_accuracy_check else None,
            }
            benchmark_results.append(metrics)
            count += 1
    except Exception as exc:
        raise BenchmarkRunError(
            f"Benchmark failed after {len(benchmark_results)} completed cases",
            benchmark_results,
            exc,
        ) from exc
    finally:
        gc.collect()

    return benchmark_results


TORCH_COMPILE_STREAMK_ENV_KEYS = (
    "TORCHINDUCTOR_ENABLE_STREAMK",
    "TORCHINDUCTOR_STREAMK_AUTOTUNE",
    "TORCHINDUCTOR_STREAMK_ONLY",
)


def torch_compile_env(**enabled_flags):
    env = {name: None for name in TORCH_COMPILE_STREAMK_ENV_KEYS}
    env.update(enabled_flags)
    return env


TORCH_COMPILE_MODE_CONFIGS = OrderedDict(
    [
        ("default", {"env": torch_compile_env()}),
        ("streamk", {"env": torch_compile_env(TORCHINDUCTOR_ENABLE_STREAMK=1)}),
        (
            "streamk_tuning",
            {
                "env": torch_compile_env(
                    TORCHINDUCTOR_ENABLE_STREAMK=1,
                    TORCHINDUCTOR_STREAMK_AUTOTUNE=1,
                )
            },
        ),
        (
            "force_streamk",
            {
                "env": torch_compile_env(
                    TORCHINDUCTOR_ENABLE_STREAMK=1,
                    TORCHINDUCTOR_STREAMK_ONLY=1,
                )
            },
        ),
        (
            "force_streamk_tuning",
            {
                "env": torch_compile_env(
                    TORCHINDUCTOR_ENABLE_STREAMK=1,
                    TORCHINDUCTOR_STREAMK_ONLY=1,
                    TORCHINDUCTOR_STREAMK_AUTOTUNE=1,
                )
            },
        ),
    ]
)


def apply_env_overrides(env_overrides):
    previous_values = {}
    for name, value in env_overrides.items():
        previous_values[name] = os.environ.get(name)
        if value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = str(value)
    return previous_values


def restore_env_overrides(previous_values):
    for name, previous_value in previous_values.items():
        if previous_value is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous_value


def resolve_torch_compile_modes(args):
    if not args.torch_compile_modes:
        return []

    requested_modes = []
    for mode in args.torch_compile_modes:
        if mode == "all":
            requested_modes.extend(TORCH_COMPILE_MODE_CONFIGS.keys())
        else:
            requested_modes.append(mode)

    modes = []
    seen = set()
    for mode in requested_modes:
        if mode not in seen:
            modes.append(mode)
            seen.add(mode)
    return modes


TRITONBLAS_MODE_CONFIGS = OrderedDict(
    [
        ("persistent", (False, False)),
        ("streamk", (True, False)),
        ("work_stealing", (False, True)),
        ("streamk_work_stealing", (True, True)),
    ]
)


def resolve_tritonblas_modes(args):
    if not args.tritonblas_modes:
        return []

    requested_modes = []
    for mode in args.tritonblas_modes:
        if mode == "all":
            requested_modes.extend(TRITONBLAS_MODE_CONFIGS.keys())
        else:
            requested_modes.append(mode)

    modes = []
    seen = set()
    for mode in requested_modes:
        if mode not in seen:
            modes.append(mode)
            seen.add(mode)
    return modes


# ============================================================
# CLI
# ============================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare Torch, Torch.compile, TritonBLAS performance and accuracy (TFLOPS, ms, errors vs torch.matmul baseline)."
    )
    parser.add_argument("--input-yaml", type=str, required=True)
    parser.add_argument("--output-csv", type=str, default="",
                        help="CSV output path for benchmark results.")
    parser.add_argument("--init_type", type=str, default="randn",
                        choices=["hpl","trig_float","zeros","randn","increasing"],
                        help="Initialization type: randn (random normal), hpl (uniform -0.5 to 0.5), "
                             "trig_float (sin of indices), zeros, increasing (i+j normalized)")
    parser.add_argument("--shuffle-bench", action="store_true")
    parser.add_argument("--print-verbose", action="store_true")
    parser.add_argument("--tritonblas-modes", nargs="+",
                        choices=["persistent", "streamk", "work_stealing", "streamk_work_stealing", "all"],
                        help="TritonBLAS modes to benchmark. Use 'all' to run persistent, streamk, "
                             "work_stealing, and streamk_work_stealing.")
    parser.add_argument("--torch-compile-modes", nargs="+",
                        choices=[
                            "default",
                            "streamk",
                            "streamk_tuning",
                            "force_streamk",
                            "force_streamk_tuning",
                            "all",
                        ],
                        help="torch.compile modes to benchmark. Use 'all' to run every configured "
                             "torch.compile mode: default, streamk, streamk_tuning, "
                             "force_streamk, and force_streamk_tuning.")
    parser.add_argument("--dynamic", action="store_true")
    parser.add_argument("--check-accuracy", action="store_true",
                        help="Check numerical accuracy against torch.matmul reference")
    parser.add_argument("--accuracy-tolerance", type=float, default=1e-2,
                        help="Tolerance for accuracy checks (default: 1e-2)")
    parser.add_argument("--debug-csv-errors", action="store_true",
                        help="Include max_abs_error and max_rel_error columns in the merged CSV.")
    parser.add_argument("--disable-torch-matmul", action="store_true",
                        help="Disable torch.matmul baseline benchmark")
    args = parser.parse_args()

    benchmark_results = []
    pending_exception = None
    torch_compile_modes = resolve_torch_compile_modes(args)
    tritonblas_modes = resolve_tritonblas_modes(args)

    try:
        if not args.disable_torch_matmul:
            if args.print_verbose:
                print("\n--- Baseline: torch.matmul ---")
            baseline_results = bench_matmul(
                args.input_yaml,
                args.init_type,
                shuffle_benchmark=args.shuffle_bench,
                print_verbose=args.print_verbose,
                enable_streamk=False,
                torch_compile=False,
                torch_compile_mode=None,
                dynamic=False,
                work_stealing=False,
                tritonblas_mode=None,
                enable_accuracy_check=args.check_accuracy,
                accuracy_tolerance=args.accuracy_tolerance,
                enable_torch_matmul=not args.disable_torch_matmul,
                print_case_results=False,
                progress_label="torch.matmul",
            )
            benchmark_results.extend(baseline_results)

        for mode in torch_compile_modes:
            if args.print_verbose:
                print(f"\n--- torch.compile mode: {mode} ---")
            clear_torch_compile_cache(print_verbose=args.print_verbose)
            env_overrides = TORCH_COMPILE_MODE_CONFIGS[mode].get("env", {})
            previous_env = apply_env_overrides(env_overrides)
            try:
                compile_results = bench_matmul(
                    args.input_yaml,
                    args.init_type,
                    shuffle_benchmark=args.shuffle_bench,
                    print_verbose=args.print_verbose,
                    enable_streamk=False,
                    torch_compile=True,
                    torch_compile_mode=mode,
                    dynamic=args.dynamic,
                    work_stealing=False,
                    tritonblas_mode=None,
                    enable_accuracy_check=args.check_accuracy,
                    accuracy_tolerance=args.accuracy_tolerance,
                    enable_torch_matmul=False,
                    print_case_results=False,
                    progress_label=f"torch.compile[{mode}]",
                )
            finally:
                restore_env_overrides(previous_env)
            benchmark_results.extend(compile_results)

        for mode in tritonblas_modes:
            enable_streamk, work_stealing = TRITONBLAS_MODE_CONFIGS[mode]
            if args.print_verbose:
                print(f"\n--- TritonBLAS mode: {mode} ---")

            results = bench_matmul(
                args.input_yaml,
                args.init_type,
                shuffle_benchmark=args.shuffle_bench,
                print_verbose=args.print_verbose,
                enable_streamk=enable_streamk,
                torch_compile=False,
                torch_compile_mode=None,
                dynamic=False,
                work_stealing=work_stealing,
                tritonblas_mode=mode,
                enable_accuracy_check=args.check_accuracy,
                accuracy_tolerance=args.accuracy_tolerance,
                enable_torch_matmul=False,
                print_case_results=False,
                progress_label=f"TritonBLAS[{mode}]",
            )
            benchmark_results.extend(results)

        fill_speedups_vs_torch(benchmark_results)
        if args.print_verbose:
            print("\n--- Benchmark Summary ---")
            print_benchmark_summary(benchmark_results)
    except BenchmarkRunError as exc:
        if exc.partial_results:
            benchmark_results.extend(exc.partial_results)
            fill_speedups_vs_torch(benchmark_results)
        pending_exception = exc.original_exception
    except Exception as exc:
        pending_exception = exc
    finally:
        if benchmark_results or args.output_csv:
            gc.collect()
            try:
                write_csv(args.output_csv, benchmark_results, include_error_details=args.debug_csv_errors)
            except OSError as exc:
                if pending_exception is None:
                    pending_exception = exc
                else:
                    print(f"WARNING: failed to write partial CSV results: {exc}")

    if pending_exception is not None:
        raise pending_exception
