# ctxshield — shield for contexts

A local, static detector for agent abuse patterns in coding sessions (fabricated confession,
decision handback, claim beyond evidence, forced repetition, unauthorized action). No network,
no model server: hashed n-gram heads scored by a native C++ / Metal (CUDA blocked out) kernel.

Build plan, provenance and every number: ClaudeStorm
`__production_steps/2026-09-27_dfc_solid_state_shield_classifier_build.md`.

## Boundary (enforced by `tests/test_boundary.py`)
- This package never imports ClaudeStorm (`core`, `workers`, `tools`, `ux`, `claudestorm`).
- It contains no sessions, units, labels or benchmark databases. Training reads a generic dataset
  directory (`ctxshield.dataset`) written by the private bridge (`tools/shield_bridge/` in ClaudeStorm).

## Layout
```
ctxshield/      dataset.py (dataset contract) · shape.py (payload cap helpers, byte-exact)
native/         C++ / Metal / CUDA  (step 3+); native/static_build/ = the fully static Linux scorer
                (ctxshield/static/ctxshield-scorer-linux-*, spoken to by ctxshield/_native_static.py)
models/         trained weights + manifest (shipped only after the leakage gate)
tests/
```

## Status
Step 2 of the runbook: skeleton + dataset contract. Nothing trained yet.
