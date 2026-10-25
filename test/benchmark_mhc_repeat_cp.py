# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""GPU-event bandwidth: selected input reads plus output writes, decimal TB/s.

Allocations, comparisons and transfers are outside timing. Rotating buffers
exceed cache. The 1 TB/s gate covers full chunks, not launch-bound tiny tails.
API/collective/service latency is not included in this device-time measurement.
"""

import argparse
import json
import statistics
from pathlib import Path

import torch
from lightop.attention import mhc


def benchmark(rows, hidden, hc, cp, rank, slots=6, iterations=60, trials=5):
    local_rows = len(range(rank, rows, cp))
    sources = [
        torch.empty((rows, hidden), device="cuda", dtype=torch.bfloat16).normal_()
        for _ in range(slots)
    ]
    outputs = [
        torch.empty((local_rows, hc, hidden), device="cuda", dtype=torch.bfloat16)
        for _ in range(slots)
    ]
    op = mhc.mhc_repeat_cp_sglang_out
    for i in range(30):
        op(sources[i % slots], hc, cp, rank, outputs[i % slots])
    torch.cuda.synchronize()
    timings = []
    for _ in range(trials):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for i in range(iterations):
            op(sources[i % slots], hc, cp, rank, outputs[i % slots])
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / iterations)
    read_bytes = local_rows * hidden * 2
    write_bytes = read_bytes * hc
    median_ms = statistics.median(timings)
    return {
        "rows": rows,
        "hidden": hidden,
        "cp_rank": rank,
        "hc": hc,
        "cp": cp,
        "selected_read_bytes": read_bytes,
        "write_bytes": write_bytes,
        "rotating_slots": slots,
        "working_set_bytes": slots * (rows * hidden * 2 + write_bytes),
        "trial_ms": timings,
        "median_ms": median_ms,
        "bandwidth_tb_s": (read_bytes + write_bytes) / (median_ms * 1e9),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-tb-s", type=float, default=1.0)
    args = parser.parse_args()
    shapes = [
        (32768, hidden, 4, 8, rank) for hidden in [4096, 5120] for rank in [0, 3, 7]
    ]
    shapes += [(32769, 6144, 3, 5, 4), (49151, 3072, 2, 3, 2), (16384, 8192, 8, 1, 0)]
    cases = [benchmark(*shape) for shape in shapes]
    result = {
        "device": torch.cuda.get_device_properties(0).gcnArchName,
        "torch": torch.__version__,
        "hip": torch.version.hip,
        "definition": "(selected input bytes + output bytes) / GPU-event time / 1e12",
        "threshold_tb_s": args.min_tb_s,
        "cases": cases,
        "passed": all(case["bandwidth_tb_s"] >= args.min_tb_s for case in cases),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    if not result["passed"]:
        raise SystemExit("Bandwidth threshold failed; raw trials were preserved")


if __name__ == "__main__":
    main()
