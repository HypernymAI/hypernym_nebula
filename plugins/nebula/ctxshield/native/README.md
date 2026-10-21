# ctxshield native core

**What ships in the plugin: the CPU scorer only.** On macOS that is the module `_native.cpython-313-darwin.so`; on Linux the
fully static programs in `ctxshield/static/` (built reproducibly from `scorer/` — see `static_build/README.md`). Every
shipped file is pinned by sha256 in `PLATFORMS.json` and checked before it loads or runs. The Metal and CUDA backends
below are research source kept with the core; they are not built into the plugin, and the loader refuses any backend
that is not built.

Build doc: `__production_steps/2026-09-27_dfc_solid_state_shield_classifier_build.md` Part IX (§9.1–9.6),
feature spec §5.2c + r2 §5.2d, build log §14 S3 / S8. The C++ featurizer IS the feature spec;
`ctxshield/features_ref.py` is the slow audit reference that `ctxshield.parity` checks it against.

## Layout

| path | what |
|---|---|
| `include/shield.h` | C++ core API (FieldSet, Scratch, tokenize_doc, featurize, featurize_score, gather_dot) + the **plain-C ABI** (`shield_scorer_create/score/destroy`, `shield_backend_available`) every backend implements |
| `src/featurize.cpp` | normalize (utf8proc, segmented) → tokenize into one arena → stream-hash FNV-1a → open-addressing accumulator → sublinear + L2 |
| `src/score.cpp` | head-interleaved all-heads gather-dot (NEON, 4 accumulator sets, prefetch; scalar path for x86) |
| `src/c_abi.cpp` | C ABI, CPU backend, CPU tokenization + parallel packing shared by the GPU hosts |
| `src/backend.h`, `src/parallel.h` | internal backend interface, thread fan-out, per-thread scratch |
| `src/metal/shield.metal`, `src/metal/metal_backend.mm`, `src/metal/embed.cmake` | Metal kernels, Obj-C++ host (shared buffers, double-buffered chunks), metallib embedded in the extension |
| `src/cuda/shield.cu`, `src/cuda/cuda_backend.cpp`, `src/cuda/shield_cuda.h` | CUDA kernels (sm_89) + host (pinned buffers, 2 streams) — research source, not built into the plugin |
| `python/bind.cpp` | pybind11 module `ctxshield._native` |
| `backends.json` | backend manifest: `backends.cuda.verified=false` until the L4 run passes |
| `bench/bench_cpu.py`, `bench/bench_metal.py`, `bench/bench_kernel.cpp` | benchmarks (results in `bench/results/`) |
| `golden/golden_v1.bin`, `golden/golden_format.py`, `golden/make_golden.py` | CUDA verification golden file |
| `cuda/run_on_l4.sh`, `cuda/check_golden.py`, `cuda/check_syntax.sh`, `cuda/shim/` | the L4 one-shot, its checker, the Mac-side syntax check |
| `third_party/utf8proc/` | vendored utf8proc 2.9.0 (Unicode 15.1), MIT, per-file `SHA256SUMS` |

## Build

```bash
source .venv/bin/activate
bash ctxshield/native/build.sh metal    # CPU + Metal (default on macOS)
bash ctxshield/native/build.sh cpu      # CPU only
# SHIELD_PARITY_DATASET=<dataset dir> adds a 500-unit feature/score parity run (and Metal) to the gate
```

`build.sh` builds into `native/build-<mode>/`, runs `tests/test_features.py` + `tests/test_backends.py`
(and the optional parity) against a **staging copy** of the package, and only then installs the
extension with copy-beside + atomic rename (a process that already loaded the old `.so` keeps its
inode — nothing is overwritten under a running trainer).

## Backends (T12 — backend honesty)

`features.score_logits(..., backend='cpu'|'metal'|'cuda')`, `features.make_scorer(backend, ...)`,
`_native.Scorer(backend, fields, delex, W, We, b)`, `_native.score_batch_metal(...)`.
A backend that is not built, not available, or (CUDA) not verified **raises**; nothing ever falls
back to the CPU. `Scorer.last_stats['backend']` is the backend that ran (the DFC writes it as
`shield_kernel`). CUDA is refused twice while unverified: by the Python loader
(`backends.json` `backends.cuda.verified=false`) and by the native layer (built with
`SHIELD_CUDA_VERIFIED=0`), unless `SHIELD_CUDA_UNVERIFIED_OK=true` (the L4 verification run only).

## CPU design (step 8 speed pass — same features, same values; parity EXACT)

* Tokens are spans into one reused per-thread arena laid out `' ' tok ' ' tok ' '`, so `w1`, `w2`
  (`tok_a 0x20 tok_b`) and the padded `' '+tok+' '` char n-gram windows are contiguous arena ranges:
  no Token objects, no per-token strings, no padded string, no code-point vectors for ASCII tokens.
* Normalization is segmented: a position just before an ASCII character is a normalization
  boundary for NFKC (± case fold) — ASCII has combining class 0, is never the second element of a
  composition, and nothing reorders across it — so ASCII runs are lower-cased byte-wise and only
  `[ASCII starter] + [non-ASCII run]` segments go through `utf8proc_decompose` +
  `utf8proc_normalize_utf32` (= `utf8proc_map` without malloc / re-encode / re-decode).
* Hashing is two-phase (all hashes of a doc, then inserts) into an open-addressing table sized to
  ≥ 4× the exact insertion count, generation-tagged (never cleared); home-slot hits/claims are
  branch-free.
* Scoring: finalize (sublinear, L2 with 4 partial sums) + gather-dot with 4 accumulator sets and W
  prefetch; the doc's `Features` are never materialized on the scoring path.

## Metal design (§9.3)

CPU tokenization (P-cores) packs token spans (`GpuTok` = a, len, w2_end, meta; 16 B) into
`MTLStorageModeShared` buffers; `score_tg_7600` runs one threadgroup (512 threads) per document:
w1/w2 one token per thread, char n-gram windows parallel over the document's arena positions (one
binary search per thread), a threadgroup open-addressing table of 7,600 packed u32 slots
(index+1 in bits 0–18, signed count in bits 19–31), simd reductions, head-interleaved float4
gather-dot. A document that claims > 3/4 of the slots or whose count leaves ±4,095 sets its
overflow flag; the host re-runs exactly those documents with `score_dev` (device-memory table ≥ 2×
the doc's insertions) — counted in `n_overflow_docs`. Chunks of 16,384 docs are double-buffered so
the CPU tokenizes chunk c+1 while the GPU runs chunk c. §9.3's kernels A and B are fused (the hashes
never leave the threadgroup).

## CUDA (§9.4) — research source, not built into the plugin

Line-for-line port of the Metal kernels: one block (256) per doc, 8,192-slot shared table
(34,952 B dynamic shared memory, opt-in attribute set), `__ldg` gathers, warp-shuffle reductions,
2 streams, pinned buffers, sm_89. On the Mac: `bash native/cuda/check_syntax.sh` keeps the sources
compiler-clean (clang CUDA host + sm_89 device passes against `cuda/shim/`, the host C++, and the CPU
core for x86_64) — syntax/types only, **not** correctness. On the L4: `bash native/cuda/run_on_l4.sh`.

## Golden file (`golden/golden_v1.bin`, format version 1)

2,000 **synthetic** units (400 per head group × 5 groups) × both specs = 10 records, expected logits
from the NEON CPU kernel on the M1 Ultra. Synthetic because build doc §2b forbids sessions/units in
`ctxshield/` and the file travels to a rented VM; real-unit correctness of the CPU featurizer is the
5,000-unit `ctxshield.parity` run. The texts cover ASCII / multi-byte / combining / CJK / emoji,
V2 placeholders, long tokens, empty fields and documents that overflow the on-chip tables.
Regenerate: `python3 native/golden/make_golden.py` (deterministic, seed 20260927).

All integers little-endian. `str` = u32 byte length + UTF-8 bytes.

```
magic            8 bytes  "CTXSGLD1"
format_version   u32      1
n_records        u32
meta             str      JSON: created, generator(+sha256), synthetic, seed, per_group, spec_sha{V1,V2},
                          native{utf8proc,unicode,dim_bits}, expected_backend, weights recipe, tolerance_prob
record × n_records:
  group          str      e.g. "pilot|reply"
  n_fields       u32
  field names    str × n_fields (sorted)
  delex          u32      0 = V1, 1 = V2
  hpad           u32      4 | 8 | 12 | 16
  weight_seed    u64
  scales         f32 × 3  w_scale, we_scale, b_scale
  weights_sha256 32 bytes sha256(W ‖ We ‖ b as float32 LE)   — verifies the regenerated weights
  indices_sha256 32 bytes sha256(indptr int64 LE ‖ indices int32 LE) of featurize_batch — CPU features exact
  n_docs         u32
  docs           str × (n_docs × n_fields), row-major, fields in the sorted order
  logits         f32 × (n_docs × hpad)   expected CPU logits
  nnz            u32 × n_docs            active hashed features per doc
```

Weights (not stored; regenerated exactly on any platform, `golden_format.make_weights`):
`u_k = (splitmix64(seed + k·0x9E3779B97F4A7C15) >> 40) / 2^24 · 2 − 1`, k = 1…; `W = u(seed)·w_scale`
reshaped `[2^18, hpad]`, `We = u(seed+1)·we_scale` `[n_fields+2, hpad]`, `b = u(seed+2)·b_scale`.

Pass rule (`cuda/check_golden.py`): CPU on the L4 host — feature indices sha EXACT and
|p − p_golden| ≤ 1e-5; CUDA — |p − p_golden| ≤ 1e-5 and |p − p_cpu| ≤ 1e-5 (p = sigmoid(logit)).
