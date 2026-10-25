# CP output interleave without a full-size copy

`lightop.attention.cp_output.interleave_cp_output_` converts a completed,
contiguous, exclusively owned CP gather from rank order to token order in place.
For `CP * L` rows, destination row `i` reads original row
`(i % CP) * L + i // CP`. Disjoint permutation cycles permit register tiles and
need no second output tensor. Elements are copied as bytes, without arithmetic.

The short-cycle kernel handles suitable power-of-two layouts. A general cycle
kernel covers other legal CP sizes, row counts, hidden widths, storage offsets,
and element widths. The physical gather must contain `CP * L` rows. Logical
uneven and bounded-replay layouts keep SGLang's existing `gather_index` selection.

The caller must finish communication before mutation. The overlap finish path
waits for its event before calling the operator. Cycle plans inspect CPU shape
metadata only and cache small device index arrays. Warm a shape before graph
capture; captured plans retain their device storage for process lifetime.

For `[32768, 3, 5120]` BF16, the required gather is 960 MiB. The former
`transpose().reshape()` additionally allocates 960 MiB. The warm in-place call
has zero extra PyTorch allocated tensor bytes; the first shape also creates a
small cycle plan. This is an operator allocation result, not a whole-service
VRAM reduction or an allocator peak for the entire model.

The existing `csrc/export.cpp` provides both full-library and focused bindings.
The full build includes `csrc/cp_output/interleave.cu`; `setup_cp_output.py`
selects `LIGHTOP_CP_OUTPUT_ONLY` for an optional small extension. The Python API
uses the focused extension when available and otherwise the full `lightop.op`.
No additional `.cpp` or `.inc` source is required.

Run GPU contracts only on an isolated GPU with sufficient free memory:

```bash
pytest -q test/test_cp_output_interleave.py
pytest -q test/test_cp_output_plan.py
python test/benchmark_cp_output.py --output cp-output-bandwidth.json
```

The measured deployment uses Prefill chunk 32768, static fraction 0.844, and
the existing 1M KV capacity. Standalone effective bandwidth for the 960 MiB
geometry was approximately 1.016 TB/s on physical a12r4n07 GPU0, using GPU events
and six rotating buffers. This is neither communication nor service throughput,
and the other measured geometries were below 1 TB/s. Full model precision and
long stability results must be reported separately from operator correctness.
