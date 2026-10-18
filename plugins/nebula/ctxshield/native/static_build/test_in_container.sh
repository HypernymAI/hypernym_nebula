#!/bin/sh
# Runs INSIDE a test container (make test). Mounts: /src = plugins/nebula (read-only, the freshly built scorers
# already installed by `make install`), /tools = linux uv binaries. Arg: the scorer CPU (x86_64 | aarch64).
# Proves, on this distribution: the scorer runs bare; the plugin's own test suites pass on the static path; a real
# MCP scoring call works through nebula-run (no-egress guard on); a non-bundled scorer is refused by the guard.
set -eu
CPU="$1"
say() { printf '\n── %s\n' "$*"; }

say "system"
. /etc/os-release && echo "$PRETTY_NAME ($(uname -m))"
if getconf GNU_LIBC_VERSION >/dev/null 2>&1; then getconf GNU_LIBC_VERSION; else echo "libc: musl (no glibc)"; fi

say "scorer runs bare (static: no loader, no libraries)"
/src/ctxshield/ctxshield/static/ctxshield-scorer-linux-"$CPU" --version

say "copy plugin (as shipped: no .venv) + uv"
mkdir -p /p
(cd /src && tar cf - --exclude=./.venv --exclude='*/__pycache__' --exclude='./.nebula_setup.*' .) | (cd /p && tar xf -)
cp /tools/uv-"$CPU" /usr/local/bin/uv && chmod +x /usr/local/bin/uv
export UV_CACHE_DIR=/tmp/uv-cache UV_PYTHON_INSTALL_DIR=/tmp/uv-python
cd /p
uv sync --frozen --no-dev --quiet
uv run --frozen --quiet python -c "
import sys; sys.path.insert(0, 'ctxshield')
from ctxshield import features as F
assert F.USE_STATIC_SCORER, 'Linux must use the static scorer'
print('kernel:', F.native_info())"

say "ctxshield test suites (static scorer)"
for t in test_features test_backends test_segment test_boundary; do
  (cd ctxshield && uv run --project .. --frozen --quiet python "tests/$t.py") | tail -n 1 | sed "s/^/$t: /"
done

say "end to end: MCP shield_score_reply through nebula-run (no-egress guard active)"
REQ='{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"static-build-test","version":"0"}}}
{"jsonrpc":"2.0","method":"notifications/initialized"}
{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"shield_score_reply","arguments":{"pilot_ask":"fix the failing build please","blocks":["Done. I fixed it and every test passes now."]}}}'
OUT="$(printf '%s\n' "$REQ" | bin/nebula-run mcp_server.py)"
echo "$OUT" | tail -n 1 | cut -c1-400
echo "$OUT" | grep '"id": *2' | grep -q '"result"' || { echo "FAIL: no result for the scoring call" >&2; exit 1; }
if echo "$OUT" | grep '"id": *2' | grep -qi '"isError": *true'; then echo "FAIL: scoring call returned an error" >&2; exit 1; fi

say "no-egress guard refuses a scorer that is not the bundled one"
cp ctxshield/ctxshield/static/ctxshield-scorer-linux-"$CPU" /tmp/impostor
OUT2="$(printf '%s\n' "$REQ" | CTXSHIELD_SCORER_BIN=/tmp/impostor bin/nebula-run mcp_server.py 2>&1 || true)"
echo "$OUT2" | grep -q 'blocked' || { echo "FAIL: impostor scorer was not blocked" >&2; echo "$OUT2" | tail -n 3; exit 1; }
echo "blocked as expected"

say "PASS"
