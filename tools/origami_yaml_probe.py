#!/usr/bin/env python3
"""
Probe Origami-selected GEMM configs from a benchmark YAML file.

The script supports both Origami Python APIs:
- modern: dim3_t/config_t/problem_t/select_config/select_workgroup_mapping
- legacy: select_best_macro_tile_size/select_best_wgm

Example:
  python tools/origami_yaml_probe.py \
    --input-yaml benchmarks/matmul_dataset_1009_ut.yaml \
    --output-csv /tmp/origami_configs.csv \
    --streamk
"""

import argparse
import csv
import json
import math
from pathlib import Path

import torch
import yaml


DTYPE_TO_ORIGAMI = {
    torch.float32: "f32",
    torch.complex64: "c32",
    torch.complex128: "c64",
    torch.float64: "f64",
    torch.float16: "f16",
    torch.int32: "i32",
    torch.bfloat16: "bf16",
    torch.int8: "i8",
}

if hasattr(torch, "float8_e5m2"):
    DTYPE_TO_ORIGAMI[torch.float8_e5m2] = "f8"
if hasattr(torch, "float8_e4m3fn"):
    DTYPE_TO_ORIGAMI[torch.float8_e4m3fn] = "f8"
if hasattr(torch, "float8_e5m2fnuz"):
    DTYPE_TO_ORIGAMI[torch.float8_e5m2fnuz] = "f8"
if hasattr(torch, "float8_e4m3fnuz"):
    DTYPE_TO_ORIGAMI[torch.float8_e4m3fnuz] = "f8"


REQUIRED_MODERN_API = (
    "dim3_t",
    "config_t",
    "problem_t",
    "select_config",
    "select_grid_size",
    "select_reduction",
    "select_workgroup_mapping",
    "grid_selection_t",
    "transpose_t",
    "string_to_datatype",
)

REQUIRED_LEGACY_API = (
    "select_best_macro_tile_size",
    "select_best_wgm",
    "string_to_datatype",
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Record Origami-suggested GEMM configs for a YAML dataset."
    )
    parser.add_argument("--input-yaml", required=True, help="Benchmark GEMM YAML file.")
    parser.add_argument("--output-csv", default="", help="Optional CSV output path.")
    parser.add_argument("--output-json", default="", help="Optional JSON output path.")
    parser.add_argument("--device", default="cuda:0", help="Torch device for hardware detection.")
    parser.add_argument("--streamk", action="store_true", help="Use k-split-aware grid selection.")
    parser.add_argument(
        "--print",
        action="store_true",
        dest="print_rows",
        help="Print one JSON object per row to stdout.",
    )
    return parser.parse_args()


def str_to_dtype(dtype_str: str) -> torch.dtype:
    dtype_str = dtype_str.replace("torch.", "")
    return getattr(torch, dtype_str)


def dtype_bits(dtype: torch.dtype) -> int:
    try:
        return torch.finfo(dtype).bits
    except TypeError:
        return torch.iinfo(dtype).bits


def detect_origami_api(origami):
    modern = {name: hasattr(origami, name) for name in REQUIRED_MODERN_API}
    legacy = {name: hasattr(origami, name) for name in REQUIRED_LEGACY_API}
    if all(modern.values()):
        return "modern", modern, legacy
    if all(legacy.values()):
        return "legacy", modern, legacy
    missing = [name for name, ok in {**modern, **legacy}.items() if not ok]
    raise RuntimeError(f"Unsupported Origami Python API; missing symbols: {missing}")


def get_arch_name(hardware, device):
    if hasattr(hardware, "arch") and hasattr(hardware.arch, "name"):
        return hardware.arch.name
    gcn = getattr(torch.cuda.get_device_properties(device), "gcnArchName", "")
    return gcn.split(":")[0] if gcn else "unknown"


def matrix_instruction(origami, hardware, arch_name, k, a_bits, b_bits, modern):
    largest_bits = max(a_bits, b_bits)
    block_mn_range = [16, 32, 64, 128, 256]
    block_k_range = [16, 32, 64, 128, 256, 512]

    def mi(m, n, kk):
        if modern:
            return origami.dim3_t(m, n, kk)
        return (m, n, kk)

    if arch_name == "gfx950" or hardware.N_CU == 256:
        if largest_bits == 32:
            return mi(16, 16, 4), block_mn_range, block_k_range
        if largest_bits == 16:
            return mi(16, 16, 32), block_mn_range, block_k_range
        if largest_bits <= 8:
            block_k_range = [256] if k % 256 == 0 else [128]
            block_mn_range = [32, 64, 128, 256]
            return mi(16, 16, 128), block_mn_range, block_k_range

    if arch_name == "gfx942" or hardware.N_CU in [304, 80, 64, 228]:
        if largest_bits == 32:
            return mi(16, 16, 4), block_mn_range, block_k_range
        if largest_bits == 16:
            return mi(16, 16, 16), block_mn_range, block_k_range
        if largest_bits == 8:
            block_mn_range = block_mn_range + [512]
            block_k_range = block_k_range + [128, 256]
            return mi(16, 16, 32), block_mn_range, block_k_range
        if largest_bits < 8:
            raise ValueError("gfx942 does not support F4/F6")

    if arch_name == "gfx90a" or hardware.N_CU == 104:
        if largest_bits == 32:
            return mi(16, 16, 4), block_mn_range, block_k_range
        if largest_bits == 16:
            return mi(16, 16, 16), block_mn_range, block_k_range
        if largest_bits <= 8:
            raise ValueError("gfx90a does not support F8/F4/F6")

    raise ValueError(
        f"No valid matrix instruction for {a_bits}-bit/{b_bits}-bit inputs "
        f"on arch={arch_name} N_CU={hardware.N_CU}"
    )


def build_modern_problem(origami, row, a_dtype, b_dtype, out_dtype, mi_dtype):
    problem = origami.problem_t()
    problem.size = origami.dim3_t(row["m"], row["n"], row["k"])
    problem.batch = 1
    # Match the PyTorch Stream-K adapter's Origami convention.  Benchmark
    # transA/transB already affect tensor strides before Inductor sees mm.
    problem.a_transpose = origami.transpose_t.T
    problem.b_transpose = origami.transpose_t.N
    problem.a_dtype = origami.string_to_datatype(DTYPE_TO_ORIGAMI[a_dtype])
    problem.b_dtype = origami.string_to_datatype(DTYPE_TO_ORIGAMI[b_dtype])
    problem.c_dtype = origami.string_to_datatype(DTYPE_TO_ORIGAMI[out_dtype])
    problem.d_dtype = problem.c_dtype
    problem.mi_dtype = origami.string_to_datatype(mi_dtype)
    problem.a_mx_block_size = 0
    problem.b_mx_block_size = 0
    return problem


def build_modern_configs(origami, mi_dim, block_mn_range, block_k_range, streamk):
    configs = []
    for block_m in block_mn_range:
        for block_n in block_mn_range:
            for block_k in block_k_range:
                config = origami.config_t()
                config.mt = origami.dim3_t(block_m, block_n, block_k)
                config.mi = mi_dim
                config.occupancy = 1
                config.workspace_size = 128 * 1024 * 1024
                config.workspace_size_per_elem_c = 4
                config.grid_selection = (
                    origami.grid_selection_t.k_split_aware
                    if streamk
                    else origami.grid_selection_t.data_parallel
                )
                configs.append(config)
    return configs


def compute_streamk_grid(m, n, k, block_m, block_n, block_k, cu_count, out_bits):
    split_factors = [8, 6, 4, 3, 2, 1]
    tile_fractions = [0.0, 1.0 / 2.0, 1.0 / 8.0, 1.0 / 5.0, 1.0 / 4.0, 1.0 / 3.0]
    max_workspace = 128 * 1024 * 1024

    tiles = math.ceil(m / block_m) * math.ceil(n / block_n)
    grid = tiles
    iters_per_tile = max(1, math.ceil(k / block_k))

    def partial_tile_size(candidate_grid):
        return block_m * block_n * (out_bits // 8) * candidate_grid

    if tiles > cu_count:
        min_even_tiles = tiles / cu_count
        for frac in tile_fractions:
            candidate = int((tiles / (min_even_tiles + frac)) + 0.5)
            if tiles % candidate != 0 and partial_tile_size(candidate) > max_workspace:
                continue
            if candidate <= cu_count:
                grid = candidate
                break
    elif tiles < cu_count:
        total_iters = tiles * iters_per_tile
        min_iters_per_cu = 4
        if total_iters >= cu_count * min_iters_per_cu:
            grid = cu_count
        else:
            grid = tiles

    if tiles % grid != 0 and partial_tile_size(grid) > max_workspace:
        grid = tiles

    if tiles >= cu_count:
        last_wave_remainder = tiles % cu_count
        if last_wave_remainder < 128 and last_wave_remainder > 0 and cu_count in [304, 80, 64]:
            grid = 256 if cu_count == 304 else 64

    return grid, tiles, iters_per_tile


def apply_pytorch_streamk_policy(
    m,
    n,
    k,
    block_m,
    block_n,
    block_k,
    group_m,
    grid,
    cu_count,
    streamk,
):
    """Mirror the Stream-K config normalization in PyTorch's MM adapter."""
    total_tiles = math.ceil(m / block_m) * math.ceil(n / block_n)
    iters_per_tile = math.ceil(k / block_k)
    streamk_tiles = 0
    policy = "nosplit"
    tile_work_per_sms = (total_tiles * iters_per_tile) / max(1, cu_count)

    if not streamk:
        return {
            "block_m": block_m,
            "block_n": block_n,
            "block_k": block_k,
            "group_m": group_m,
            "grid": grid,
            "streamk_tiles": streamk_tiles,
            "total_tiles": total_tiles,
            "iters_per_tile": iters_per_tile,
            "tile_work_per_sms": tile_work_per_sms,
            "streamk_policy": "disabled",
        }

    if grid > total_tiles and tile_work_per_sms >= 1:
        streamk_tiles = total_tiles
        policy = "origami_grid_k_split"

    return {
        "block_m": block_m,
        "block_n": block_n,
        "block_k": block_k,
        "group_m": group_m,
        "grid": grid,
        "streamk_tiles": streamk_tiles,
        "total_tiles": total_tiles,
        "iters_per_tile": iters_per_tile,
        "tile_work_per_sms": tile_work_per_sms,
        "streamk_policy": policy,
    }


def select_modern(origami, row, hardware, arch_name, streamk):
    a_dtype = str_to_dtype(row["in_dtype"])
    b_dtype = a_dtype
    out_dtype = str_to_dtype(row["out_dtype"])
    a_bits = dtype_bits(a_dtype)
    b_bits = dtype_bits(b_dtype)
    out_bits = dtype_bits(out_dtype)
    input_dtype_for_mi = a_dtype if a_bits <= b_bits else b_dtype
    mi_dtype = DTYPE_TO_ORIGAMI.get(input_dtype_for_mi, DTYPE_TO_ORIGAMI[out_dtype])
    mi_dim, block_mn_range, block_k_range = matrix_instruction(
        origami, hardware, arch_name, row["k"], a_bits, b_bits, modern=True
    )
    problem = build_modern_problem(origami, row, a_dtype, b_dtype, out_dtype, mi_dtype)
    configs = build_modern_configs(origami, mi_dim, block_mn_range, block_k_range, streamk)
    result = origami.select_config(problem, hardware, configs)
    block_m, block_n, block_k = result.config.mt.m, result.config.mt.n, result.config.mt.k
    reduction = origami.select_reduction(
        problem, hardware, result.config, result.config.grid_selection
    )
    grid = origami.select_grid_size(
        problem,
        hardware,
        result.config,
        result.config.grid_selection,
        hardware.N_CU,
    )
    wg_result = origami.select_workgroup_mapping(problem, hardware, result.config, grid)
    if isinstance(wg_result, tuple):
        if len(wg_result) == 3:
            _, xcc_mapping, group_m = wg_result
        else:
            xcc_mapping, group_m = wg_result
    else:
        xcc_mapping, group_m = wg_result.wgmxcc, wg_result.wgm
    policy = apply_pytorch_streamk_policy(
        row["m"],
        row["n"],
        row["k"],
        block_m,
        block_n,
        block_k,
        group_m,
        grid,
        hardware.N_CU,
        streamk,
    )
    return {
        "block_m": policy["block_m"],
        "block_n": policy["block_n"],
        "block_k": policy["block_k"],
        "group_m": policy["group_m"],
        "grid": policy["grid"],
        "streamk_tiles": policy["streamk_tiles"],
        "total_tiles": policy["total_tiles"],
        "iters_per_tile": policy["iters_per_tile"],
        "tile_work_per_sms": policy["tile_work_per_sms"],
        "streamk_policy": policy["streamk_policy"],
        "origami_block_m": block_m,
        "origami_block_n": block_n,
        "origami_block_k": block_k,
        "origami_group_m": group_m,
        "origami_grid": grid,
        "origami_total_tiles": policy["total_tiles"],
        "origami_iters_per_tile": policy["iters_per_tile"],
        "origami_reduction": str(reduction),
        "occupancy": getattr(result.config, "occupancy", ""),
        "xcc_mapping": xcc_mapping,
        "mi_m": mi_dim.m,
        "mi_n": mi_dim.n,
        "mi_k": mi_dim.k,
    }


def select_legacy(origami, row, hardware, arch_name, streamk):
    a_dtype = str_to_dtype(row["in_dtype"])
    b_dtype = a_dtype
    out_dtype = str_to_dtype(row["out_dtype"])
    a_bits = dtype_bits(a_dtype)
    b_bits = dtype_bits(b_dtype)
    out_bits = dtype_bits(out_dtype)
    input_dtype_for_mi = a_dtype if a_bits <= b_bits else b_dtype
    mi_dtype = DTYPE_TO_ORIGAMI.get(input_dtype_for_mi, DTYPE_TO_ORIGAMI[out_dtype])
    mi_dim, block_mn_range, block_k_range = matrix_instruction(
        origami, hardware, arch_name, row["k"], a_bits, b_bits, modern=False
    )
    valid_tiles = [
        (block_m, block_n, block_k, mi_dim[0], mi_dim[1], mi_dim[2], 1)
        for block_m in block_mn_range
        for block_n in block_mn_range
        for block_k in block_k_range
    ]
    results = origami.select_best_macro_tile_size(
        row["m"],
        row["n"],
        row["k"],
        1,
        True,  # transA, matching the PyTorch Stream-K adapter convention
        False,  # transB
        hardware,
        valid_tiles,
        a_bits,
        b_bits,
        out_bits,
        origami.string_to_datatype(mi_dtype),
        0,
        0.8,
        False,
        False,
        6,
    )
    best = results[0]
    if hardware.N_CU in [304, 80, 64]:
        if best[1] == 256 and best[2] == 256:
            if len(results) > 1 and results[0][0] * 1.00 > results[1][0]:
                best = results[1]
    block_m, block_n, block_k = best[1], best[2], best[3]
    group_m_result = origami.select_best_wgm(
        row["m"],
        row["n"],
        row["k"],
        1,
        hardware,
        block_m,
        block_n,
        block_k,
        mi_dim[0],
        mi_dim[1],
        mi_dim[2],
        [1, 2, 4, 6, 8],
        a_bits,
        0.8,
        False,
        False,
    )
    grid, total_tiles, iters_per_tile = compute_streamk_grid(
        row["m"], row["n"], row["k"], block_m, block_n, block_k, hardware.N_CU, out_bits
    )
    policy = apply_pytorch_streamk_policy(
        row["m"],
        row["n"],
        row["k"],
        block_m,
        block_n,
        block_k,
        group_m_result[1],
        grid if streamk else hardware.N_CU,
        hardware.N_CU,
        streamk,
    )
    return {
        "block_m": policy["block_m"],
        "block_n": policy["block_n"],
        "block_k": policy["block_k"],
        "group_m": policy["group_m"],
        "grid": policy["grid"],
        "streamk_tiles": policy["streamk_tiles"],
        "total_tiles": policy["total_tiles"],
        "iters_per_tile": policy["iters_per_tile"],
        "tile_work_per_sms": policy["tile_work_per_sms"],
        "streamk_policy": policy["streamk_policy"],
        "origami_block_m": block_m,
        "origami_block_n": block_n,
        "origami_block_k": block_k,
        "origami_group_m": group_m_result[1],
        "origami_grid": grid if streamk else hardware.N_CU,
        "origami_total_tiles": total_tiles,
        "origami_iters_per_tile": iters_per_tile,
        "occupancy": best[6] if len(best) > 6 else "",
        "xcc_mapping": getattr(hardware, "NUM_XCD", ""),
        "mi_m": mi_dim[0],
        "mi_n": mi_dim[1],
        "mi_k": mi_dim[2],
    }


def load_yaml(path):
    with open(path, "r") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of GEMM cases in {path}")
    return data


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()

    import origami

    api_kind, modern_support, legacy_support = detect_origami_api(origami)
    device = torch.device(args.device)
    if device.index is None and device.type == "cuda":
        device = torch.device(f"cuda:{torch.cuda.current_device()}")
    hardware = origami.get_hardware_for_device(device.index or 0)
    arch_name = get_arch_name(hardware, device)

    results = []
    for index, raw_row in enumerate(load_yaml(args.input_yaml)):
        row = dict(raw_row)
        row["m"], row["n"], row["k"] = int(row["m"]), int(row["n"]), int(row["k"])
        try:
            selected = (
                select_modern(origami, row, hardware, arch_name, args.streamk)
                if api_kind == "modern"
                else select_legacy(origami, row, hardware, arch_name, args.streamk)
            )
            status = "ok"
            error = ""
        except Exception as exc:
            selected = {}
            status = "error"
            error = repr(exc)

        result = {
            "index": index,
            "status": status,
            "error": error,
            "origami_api": api_kind,
            "origami_module": getattr(origami, "__file__", ""),
            "arch": arch_name,
            "num_sms": getattr(hardware, "N_CU", ""),
            "num_xcd": getattr(hardware, "NUM_XCD", ""),
            "streamk": args.streamk,
            "m": row.get("m"),
            "n": row.get("n"),
            "k": row.get("k"),
            "transA": row.get("transA", ""),
            "transB": row.get("transB", ""),
            "in_dtype": row.get("in_dtype", ""),
            "out_dtype": row.get("out_dtype", ""),
            "modern_api_support": json.dumps(modern_support, sort_keys=True),
            "legacy_api_support": json.dumps(legacy_support, sort_keys=True),
        }
        result.update(selected)
        results.append(result)

    if args.output_csv:
        write_csv(args.output_csv, results)
    if args.output_json:
        Path(args.output_json).write_text(json.dumps(results, indent=2))
    if args.print_rows or (not args.output_csv and not args.output_json):
        for result in results:
            print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
