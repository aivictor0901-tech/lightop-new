# mHC broadcast after round-robin CP selection

The API **mhc_repeat_cp_sglang(x, hc_mult, cp_size, cp_rank)** computes exactly:

~~~python
x[cp_rank::cp_size].unsqueeze(1).repeat(1, hc_mult, 1)
~~~

The output is contiguous [selected_rows, hc_mult, hidden]. It does not
materialize a global [tokens, hc_mult, hidden] intermediate. The operation
copies storage bits, including BF16 NaN payloads and signed zero.

## Shape and storage contract

- The input is a two-dimensional strided GPU tensor in a ROCm build.
- Token count and hidden width may be zero. There is no 32768-token limit and
  no hidden-width whitelist.
- HC and CP are any positive integers; the rank is in [0, CP).
- Token counts need not be divisible by CP. A rank with no selected rows returns
  a correctly shaped empty tensor.
- Contiguous, transposed, column-strided, offset and expanded inputs are accepted.
  Ordinary non-quantized element storage widths of 1, 2, 4, 8 and 16 bytes are
  supported. Negative strides, unresolved conjugate/negative views and autograd
  inputs are outside this inference API.
- Aligned rows use 16-byte vector loads; other layouts use the general element
  copy. The vector path is an optimization, not an acceptance restriction.

The native **mhc_repeat_cp_sglang_out(x, hc, cp, rank, out)** API accepts a
preallocated contiguous output with the exact shape, dtype and device. Input
and output must not overlap. This API allows fixed workspaces and allocation-free
bandwidth measurements. Both APIs use the current stream and device guard, and
record the input/output storage on the execution stream for asynchronous lifetime
safety. The caller must establish producer/consumer stream ordering.

## Building

The normal LightOp build includes the operator and exports all three APIs.
For an existing installation, the optional small extension can be built with:

~~~bash
PYTORCH_ROCM_ARCH=gfx936 MAX_JOBS=2 \
  python setup_mhc_repeat_cp.py build_ext --build-lib build/mhc-extension
~~~

The optional extension is **lightop._mhc_repeat_cp**. Use it with this revision's
python/lightop/attention/mhc.py; the frontend selects it when installed and
otherwise selects the normal lightop.op implementation. Rebuild for the target
architecture/PyTorch ABI. This revision was measured on gfx936 with Torch 2.11
and DTK 26.04; other hardware has not been measured.

## SGLang integration

Set **SGLANG_DSV4_FUSE_MHC_REPEAT_CP_SPLIT=1** before starting SGLang. The V4.1
entry helper must allow arbitrary hidden/HC/CP values, uneven row counts and
padded CPU metadata. It checks consistency of the physical hidden, input-ID,
position and cache-location row counts.

The existing layout/mode dispatch still owns sharding: legacy round-robin
Prefill can use this operation; CP-v2 already shards its inputs, zigzag has a
different mapping, and TBO partitions global rows before sharding its children.
These semantic branches keep their corresponding implementations. Normal
collective padding is independent of this copy; enabling it does not change
chunk size, KV capacity or the context window.

## Correctness, bandwidth and memory

~~~bash
python -m pytest -q test/test_mhc_repeat_cp_sglang.py test/test_mhc_repeat_cp_v41.py
python test/benchmark_mhc_repeat_cp.py --output bandwidth.json --min-tb-s 1.0
~~~

Tests cover arbitrary sizes, uneven CP ownership, empty ranks, HC/CP changes,
multiple element widths, non-contiguous storage, alias rejection, current-stream
execution and asynchronous input lifetime.

Bandwidth is effective (selected input bytes + output bytes) / GPU-event time.
Only selected rows count as input traffic. Six rotating buffers exceed cache;
allocation, comparisons and transfers are outside the timing window. Results
are decimal TB/s and are device timings, not collective or serving throughput.
The 1 TB/s threshold applies to the bandwidth-bound large cases; tiny tails are
launch-bound and are checked for correctness.

On gfx936, nine measured large cases passed at **1.045–1.240 TB/s**. The actual
V4.1 geometry [32768,5120], HC4, CP8 measured **1.064–1.077 TB/s** across tested
ranks. Raw per-trial values are retained with the deployment evidence.

At the same V4.1 geometry, an isolated process measured the complete old entry
sequence (global repeat, release embedding, then local slice copy) against direct
selected repeat. Peak allocated bytes, including the input, were **1600 MiB**
versus **480 MiB**, a **1120 MiB** reduction. These are entry-phase allocator
measurements, not whole-service peak reductions.
