#!/usr/bin/env bash
# ONE-SHOT CUDA VERIFICATION on an NVIDIA L4 (build doc §9.4). NOT runnable on the Mac.
#   Machine: GCP g2-standard-8 (1× L4, sm_89), CUDA 12.x toolkit (nvcc), cmake ≥ 3.20, python3 with
#            numpy + pybind11 (uv pip install numpy pybind11), a C++20 host compiler.
#   Input:   this ctxshield/ tree (includes native/golden/golden_v1.bin — synthetic text only).
#   Steps:   build with -DSHIELD_CUDA=ON (sm_89) into native/build-cuda (NOT installed over the
#            package), stage the package, run the Python tests, then check_golden.py:
#            CPU-x86 features exact + logits ≤ 1e-5 vs golden, CUDA ≤ 1e-5 vs golden and vs CPU,
#            overflow docs counted, CPU-vs-CUDA bench + crossover → native/cuda/results/cuda_l4_<stamp>.json
#   After:   copy the results JSON back, log it in the build doc §14; ONLY THEN may
#            native/backends.json set backends.cuda.verified=true (and a rebuild pass
#            -DSHIELD_CUDA_VERIFIED=1). Until then every CUDA request is refused by the loader.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NATIVE="$HERE/.."
PKG="$NATIVE/.."
PY="${PYTHON:-python3}"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="$HERE/results/cuda_l4_$STAMP.json"
mkdir -p "$HERE/results"

echo "== device"; nvidia-smi
echo "== toolkit"; nvcc --version
"$PY" -c "import numpy, pybind11; print('numpy', numpy.__version__, 'pybind11', pybind11.__version__)"

echo "== build (CUDA, sm_89)"
BUILD="$NATIVE/build-cuda"
cmake -S "$NATIVE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)" -DPython_EXECUTABLE="$(command -v "$PY")" \
  -DSHIELD_METAL=OFF -DSHIELD_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build "$BUILD" -j
SO="$(ls "$BUILD"/_native*.so)"

echo "== stage"
STAGE="$BUILD/stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/ctxshield" "$STAGE/tests" "$STAGE/native/golden" "$STAGE/native/cuda"
cp "$PKG"/ctxshield/*.py "$STAGE/ctxshield/"
cp "$SO" "$STAGE/ctxshield/"
cp "$PKG"/tests/*.py "$STAGE/tests/"
cp "$NATIVE/backends.json" "$STAGE/native/"
cp "$NATIVE/golden/golden_format.py" "$NATIVE/golden/golden_v1.bin" "$STAGE/native/golden/"
cp "$HERE/check_golden.py" "$STAGE/native/cuda/"

echo "== tests (CPU path on x86 + backend honesty; CUDA is still refused here)"
( cd "$STAGE" && env -u SHIELD_CUDA_UNVERIFIED_OK "$PY" tests/test_features.py && \
  env -u SHIELD_CUDA_UNVERIFIED_OK "$PY" tests/test_backends.py )

echo "== golden parity + bench (SHIELD_CUDA_UNVERIFIED_OK=true for THIS run only)"
( cd "$STAGE" && SHIELD_CUDA_UNVERIFIED_OK=true "$PY" native/cuda/check_golden.py \
    --golden native/golden/golden_v1.bin --backends cpu,cuda --bench --out "$OUT" )
echo "RESULT: $OUT"
echo "Copy it back, log it in the build doc §14, and only then flip backends.cuda.verified."
