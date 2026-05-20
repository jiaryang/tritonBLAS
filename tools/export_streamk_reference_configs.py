#!/usr/bin/env python3
"""Export Stream-K reference configs from benchmark logs.

This is a debug helper for seeding a newer PyTorch/Inductor run with configs
observed in an older run.  The output can be consumed by
TORCHINDUCTOR_STREAMK_REFERENCE_CONFIG_PATH.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path


SELECTOR_CONFIG_RE = re.compile(r"selector config: (\{.*\})")
BENCHMARK_RE = re.compile(
    r"torch\.compile\[[^\]]+\]: "
    r"\[M=(?P<M>\d+),N=(?P<N>\d+),K=(?P<K>\d+),trans=(?P<trans>[A-Z]+)\] "
    r"dtype=torch\.(?P<dtype>[a-z0-9_]+) -> "
    r"(?P<tflops>[0-9.]+) TF/s \((?P<ms>[0-9.]+) ms"
)


def parse_log(path: Path) -> list[dict[str, object]]:
    pending_configs: list[dict[str, object]] = []
    entries: list[dict[str, object]] = []

    for line in path.read_text(errors="replace").splitlines():
        config_match = SELECTOR_CONFIG_RE.search(line)
        if config_match:
            try:
                config = ast.literal_eval(config_match.group(1))
            except (SyntaxError, ValueError):
                continue
            if isinstance(config, dict):
                pending_configs.append(config)
            continue

        benchmark_match = BENCHMARK_RE.search(line)
        if not benchmark_match or not pending_configs:
            continue

        config = pending_configs.pop(0)
        groups = benchmark_match.groupdict()
        entries.append(
            {
                "M": int(groups["M"]),
                "N": int(groups["N"]),
                "K": int(groups["K"]),
                "dtype": groups["dtype"],
                "trans": groups["trans"],
                "tflops": float(groups["tflops"]),
                "ms": float(groups["ms"]),
                "config": config,
            }
        )

    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("log", type=Path, help="Benchmark log to parse")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Output JSON path. Defaults to stdout.",
    )
    args = parser.parse_args()

    entries = parse_log(args.log)
    payload = {
        "source": str(args.log),
        "configs": entries,
    }
    text = json.dumps(payload, indent=2, sort_keys=True)

    if args.output:
        args.output.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
