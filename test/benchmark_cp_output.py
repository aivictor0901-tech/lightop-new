# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

"""Offline GPU-event bandwidth check with six rotating CP gather buffers."""

import argparse
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-rows", type=int, default=4096)
    parser.add_argument("--cp", type=int, default=8)
    parser.add_argument("--widths", type=int, nargs="+", default=[5120, 15360, 20480])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.local_rows < 1 or args.cp < 1 or any(width < 1 for width in args.widths):
        parser.error("positive local rows, CP, and widths are required")
    import torch
    from lightop.attention.cp_output import interleave_cp_output_

    rows = args.cp * args.local_rows
    moved = sum((row % args.cp) * args.local_rows + row // args.cp != row
                for row in range(rows))
    results = []
    for width in args.widths:
        buffers = [torch.empty((rows, width), device="cuda", dtype=torch.bfloat16)
                   for _ in range(6)]
        for tensor in buffers:
            interleave_cp_output_(tensor, args.cp)
        torch.cuda.synchronize()
        samples = []
        for _ in range(9):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for repetition in range(24):
                interleave_cp_output_(buffers[repetition % 6], args.cp)
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) / 24)
        elapsed = statistics.median(samples)
        transferred = 2 * moved * width * buffers[0].element_size()
        results.append({"shape": [rows, width], "cp": args.cp, "moved_rows": moved,
                        "read_write_bytes": transferred, "median_ms": elapsed,
                        "effective_TB_s": transferred / (elapsed * 1e9) if moved else None,
                        "event_ms": samples, "rotation_buffers": 6,
                        "scope": "operator on the selected isolated GPU; not communication or service throughput"})
        del tensor, buffers
    payload = json.dumps(results, indent=2)
    if args.output:
        args.output.write_text(payload + "\n")
    print(payload)


if __name__ == "__main__":
    main()
