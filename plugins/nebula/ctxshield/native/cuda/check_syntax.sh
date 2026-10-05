#!/usr/bin/env bash
# Keep the CUDA sources compiler-clean on a machine without a CUDA toolkit (build doc §9.4):
#   1. shield.cu host pass    (clang -x cuda --cuda-host-only, shim declarations)
#   2. shield.cu device pass  (clang -x cuda --cuda-device-only --cuda-gpu-arch=sm_89)
#   3. cuda_backend.cpp as host C++ against the shim
#   4. the CPU core compiled for x86_64 (the non-NEON scoring path the L4 host runs)
# This proves syntax/types only — NOT correctness. Only run_on_l4.sh on a real L4 verifies CUDA.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
N="$HERE/.."
SHIM="$HERE/shim/cuda_syntax_shim.h"
CXX="${CXX:-clang++}"
common=(-nocudainc -nocudalib -fsyntax-only -Wall -Wextra -DSHIELD_CUDA_SYNTAX_SHIM -include "$SHIM" -I"$N/src/cuda")
"$CXX" -std=c++17 -x cuda --cuda-host-only "${common[@]}" "$N/src/cuda/shield.cu"
echo "  ✓ shield.cu host pass"
"$CXX" -std=c++17 -x cuda --cuda-device-only --cuda-gpu-arch=sm_89 "${common[@]}" "$N/src/cuda/shield.cu"
echo "  ✓ shield.cu device pass (sm_89)"
"$CXX" -std=c++20 -fsyntax-only -Wall -Wextra -DSHIELD_CUDA_SYNTAX_SHIM -DSHIELD_HAVE_CUDA=1 -include "$SHIM" \
  -I"$N/include" -I"$N/src" -I"$N/src/cuda" "$N/src/cuda/cuda_backend.cpp"
echo "  ✓ cuda_backend.cpp (host C++)"
for f in featurize.cpp score.cpp c_abi.cpp; do
  "$CXX" -std=c++20 -target x86_64-apple-macos11 -fsyntax-only -Wall -Wextra -DSHIELD_HAVE_CUDA=1 \
    -I"$N/include" -I"$N/src" -I"$N/third_party/utf8proc" -DUTF8PROC_STATIC "$N/src/$f"
done
echo "  ✓ CPU core for x86_64 (non-NEON path), SHIELD_HAVE_CUDA=1"
echo "SYNTAX CLEAN (not a correctness check)"
