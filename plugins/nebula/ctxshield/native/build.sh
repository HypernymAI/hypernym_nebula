#!/usr/bin/env bash
# Build the ctxshield native module, test it from a STAGING copy, and only then install it next to
# the Python package (atomic rename, so a process that has the old .so loaded keeps its inode).
#   bash ctxshield/native/build.sh cpu      CPU only
#   bash ctxshield/native/build.sh metal    CPU + Metal (Apple; default on macOS)
#   bash ctxshield/native/build.sh cuda     CPU + CUDA (UNTESTED; for the L4 run — native/cuda/run_on_l4.sh)
# Optional: SHIELD_PARITY_DATASET=<dataset dir> adds a 500-unit parity run to the pre-install gate.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PKG="$HERE/../ctxshield"
MODE="${1:-metal}"
PY="${PYTHON:-python3}"
METAL=OFF; CUDA=OFF
case "$MODE" in
  cpu) ;;
  metal) METAL=ON ;;
  cuda) CUDA=ON ;;
  *) echo "usage: build.sh cpu|metal|cuda" >&2; exit 2 ;;
esac
BUILD="$HERE/build-$MODE"
cmake -S "$HERE" -B "$BUILD" -DCMAKE_BUILD_TYPE=Release \
  -Dpybind11_DIR="$("$PY" -m pybind11 --cmakedir)" \
  -DPython_EXECUTABLE="$(command -v "$PY")" \
  -DSHIELD_METAL="$METAL" -DSHIELD_CUDA="$CUDA"
cmake --build "$BUILD" -j
SO="$(ls "$BUILD"/_native*.so)"

# ── pre-install gate: tests run against a staging copy of the package ──
STAGE="$BUILD/stage"
rm -rf "$STAGE"
mkdir -p "$STAGE/ctxshield" "$STAGE/tests"
cp "$PKG"/*.py "$STAGE/ctxshield/"
cp "$SO" "$STAGE/ctxshield/"
cp "$HERE/../tests/"*.py "$STAGE/tests/"
mkdir -p "$STAGE/native"
cp "$HERE/backends.json" "$STAGE/native/"
( cd "$STAGE" && "$PY" tests/test_features.py && "$PY" tests/test_backends.py )
if [ -n "${SHIELD_PARITY_DATASET:-}" ]; then
  ( cd "$STAGE" && "$PY" -m ctxshield.parity --features --scores --dataset "$SHIELD_PARITY_DATASET" --n 500 )
  if [ "$METAL" = ON ]; then
    ( cd "$STAGE" && "$PY" -m ctxshield.parity --scores --backend metal --dataset "$SHIELD_PARITY_DATASET" --n 500 )
  fi
else
  echo "NOTE: SHIELD_PARITY_DATASET not set — dataset parity NOT part of this gate (run ctxshield.parity)"
fi

# ── install: copy beside, then atomic rename ──
DEST="$PKG/$(basename "$SO")"
cp "$SO" "$DEST.tmp.$$"
mv -f "$DEST.tmp.$$" "$DEST"
ls -la "$DEST"
