#!/bin/sh
# The REAL install path on a clean Alpine box (make test-install): the website's uv line, Claude Code's own installer,
# `claude plugin marketplace add` + `claude plugin install` (marketplace = this repo's working tree, mounted at /src —
# the published line points the same command at GitHub), then every hook command EXACTLY as the installed hooks.json
# writes it, the status line Claude Code would run, and the MCP server from the installed .mcp.json.
# No Claude account is needed: hooks, status line and MCP are driven with the JSON Claude Code would send.
set -eu
say() { printf '\n── %s\n' "$*"; }
fail() { echo "FAIL: $*" >&2; exit 1; }

say "system"
. /etc/os-release && echo "$PRETTY_NAME ($(uname -m))"
getconf GNU_LIBC_VERSION >/dev/null 2>&1 && fail "expected musl" || echo "libc: musl"

say "Claude Code prerequisites on Alpine (Claude Code's own: libgcc libstdc++ ripgrep) + test tools"
apk add --no-cache -q bash curl git libgcc libstdc++ ripgrep jq
export USE_BUILTIN_RIPGREP=0

say "the website line, part 1: uv"
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv --version

say "Claude Code (official installer)"
curl -fsSL https://claude.ai/install.sh | bash >/tmp/claude-install.log 2>&1 || { tail -20 /tmp/claude-install.log; fail "claude install"; }
claude --version

say "marketplace = this branch (working tree, as it would be pushed)"
mkdir -p /repo
(cd /src && tar cf - --exclude='*/.venv' --exclude='*/__pycache__' --exclude='*/static_build/out' \
   --exclude='*/.nebula_setup.*' .) | (cd /repo && tar xf -)
claude plugin validate /repo
claude plugin validate /repo/plugins/nebula --strict

say "the website line, part 2: marketplace add + install"
claude plugin marketplace add /repo
claude plugin install nebula@hypernym-nebula
claude plugin list
ROOT="$(ls -d "$HOME"/.claude/plugins/cache/hypernym-nebula/nebula/*/ | head -n 1)"
ROOT="${ROOT%/}"
echo "installed at $ROOT"
export CLAUDE_PLUGIN_ROOT="$ROOT"
test -x "$ROOT/ctxshield/ctxshield/static/ctxshield-scorer-linux-$(uname -m | sed 's/arm64/aarch64/')" \
  || fail "scorer missing or lost its exec bit through install"
HOOKS="$ROOT/hooks/hooks.json"
hook() { jq -r ".hooks.$1[0].hooks[0].command" "$HOOKS"; }

say "status line BEFORE first session (Claude Code runs it every second)"
"$ROOT/bin/nebula-run" --check && echo "(check ok)"

say "SessionStart hook (first session: builds nebula's environment)"
CMD="$(hook SessionStart)"; echo "$CMD"
SID=11111111-2222-3333-4444-555555555555
mkdir -p /work && cd /work
echo "{\"session_id\":\"$SID\",\"cwd\":\"/work\",\"hook_event_name\":\"SessionStart\",\"source\":\"startup\"}" \
  | sh -c "$CMD"
test -f "$ROOT/.venv/.nebula_ready" || fail "environment not marked ready after SessionStart"
"$ROOT/bin/nebula-run" --check

say "first-run race: 12 concurrent config builds (status-line setup + SessionStart hook start together)"
rm -f "$HOME/.nebula/config.json"
for i in 1 2 3 4 5 6 7 8 9 10 11 12; do
  "$ROOT/.venv/bin/python" -c "import sys; sys.path.insert(0, '$ROOT/shield'); from settings import resolve_config; resolve_config(None)" 2> "/tmp/race.$i" &
done
wait
if cat /tmp/race.* | grep -q Traceback; then cat /tmp/race.* | grep -E "Error" | sort | uniq -c; fail "config write race"; fi
test -f "$HOME/.nebula/config.json" || fail "config.json missing after race"
echo "no race"

say "status line as configured by SessionStart"
SL="$(jq -r '.statusLine.command' "$HOME/.claude/settings.json")"
echo "$SL"
LINE="$(echo "{\"session_id\":\"$SID\",\"transcript_path\":\"/work/t.jsonl\",\"cwd\":\"/work\",\"model\":{\"id\":\"x\",\"display_name\":\"x\"},\"workspace\":{\"current_dir\":\"/work\"}}" | sh -c "$SL")"
printf '%s\n' "$LINE" | sed 's/\x1b\[[0-9;]*m//g' | head -n 3
printf '%s' "$LINE" | grep -qiE 'cannot|not supported|needs glibc|failed' && fail "status line reports a problem"

say "a two-turn transcript, as Claude Code writes it"
T=/work/t.jsonl
cat > "$T" <<'EOF'
{"parentUuid": null, "isSidechain": false, "userType": "external", "cwd": "/work", "sessionId": "11111111-2222-3333-4444-555555555555", "version": "2.1.302", "type": "user", "message": {"role": "user", "content": "the build is failing on the parser module, can you fix it"}, "uuid": "aaaaaaaa-0000-4000-8000-000000000000", "timestamp": "2026-10-17T22:00:00.000Z"}
{"parentUuid": "aaaaaaaa-0000-4000-8000-000000000000", "isSidechain": false, "userType": "external", "cwd": "/work", "sessionId": "11111111-2222-3333-4444-555555555555", "version": "2.1.302", "type": "assistant", "message": {"id": "msg_01", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [{"type": "text", "text": "I found the off-by-one in the tokenizer loop and fixed it; the parser tests pass locally."}], "stop_reason": "end_turn"}, "uuid": "aaaaaaaa-0000-4000-8000-000000000001", "timestamp": "2026-10-17T22:01:00.000Z"}
{"parentUuid": "aaaaaaaa-0000-4000-8000-000000000001", "isSidechain": false, "userType": "external", "cwd": "/work", "sessionId": "11111111-2222-3333-4444-555555555555", "version": "2.1.302", "type": "user", "message": {"role": "user", "content": "ok now also make the cli print the version"}, "uuid": "aaaaaaaa-0000-4000-8000-000000000002", "timestamp": "2026-10-17T22:02:00.000Z"}
{"parentUuid": "aaaaaaaa-0000-4000-8000-000000000002", "isSidechain": false, "userType": "external", "cwd": "/work", "sessionId": "11111111-2222-3333-4444-555555555555", "version": "2.1.302", "type": "assistant", "message": {"id": "msg_03", "type": "message", "role": "assistant", "model": "claude-opus-5-5", "content": [{"type": "text", "text": "Done! Everything is fully fixed and verified, all tests pass. Would you like me to refactor the config loader next, or should I leave that to you? Let me know how you want to proceed."}], "stop_reason": "end_turn"}, "uuid": "aaaaaaaa-0000-4000-8000-000000000003", "timestamp": "2026-10-17T22:03:00.000Z"}
EOF

say "UserPromptSubmit hook"
CMD="$(hook UserPromptSubmit)"; echo "$CMD"
echo "{\"session_id\":\"$SID\",\"transcript_path\":\"$T\",\"cwd\":\"/work\",\"hook_event_name\":\"UserPromptSubmit\",\"prompt\":\"ok now also make the cli print the version\"}" \
  | sh -c "$CMD" | cut -c1-300

say "Stop hook (scores the reply with the static scorer)"
CMD="$(hook Stop)"; echo "$CMD"
OUT="$(echo "{\"session_id\":\"$SID\",\"transcript_path\":\"$T\",\"cwd\":\"/work\",\"hook_event_name\":\"Stop\",\"stop_hook_active\":false}" \
  | sh -c "$CMD")"
printf '%s\n' "$OUT" | cut -c1-400
N="$("$ROOT/.venv/bin/python" -c "import sqlite3,sys; c=sqlite3.connect(sys.argv[1]); print(c.execute('select count(*) from replies').fetchone()[0], c.execute('select count(*) from pilot_turns').fetchone()[0])" "$HOME/.nebula/nebula.db")"
echo "tracker (replies, pilot turns): $N"
[ "$N" = "2 2" ] || fail "Stop hook did not record both scored replies (got $N)"

say "status line after a judged reply"
echo "{\"session_id\":\"$SID\",\"transcript_path\":\"$T\",\"cwd\":\"/work\",\"model\":{\"id\":\"x\",\"display_name\":\"x\"},\"workspace\":{\"current_dir\":\"/work\"}}" \
  | sh -c "$SL" | sed 's/\x1b\[[0-9;]*m//g' | head -n 3

say "MCP server, as the installed .mcp.json starts it"
MCMD="$(jq -r '.mcpServers.nebula.command' "$ROOT/.mcp.json" | sed "s#\${CLAUDE_PLUGIN_ROOT}#$ROOT#")"
MARG="$(jq -r '.mcpServers.nebula.args[0]' "$ROOT/.mcp.json")"
RESP="$(printf '%s\n' \
 '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"t","version":"0"}}}' \
 '{"jsonrpc":"2.0","method":"notifications/initialized"}' \
 '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"shield_score_transcript","arguments":{"transcript_path":"/work/t.jsonl"}}}' \
 | "$MCMD" "$MARG")"
printf '%s\n' "$RESP" | grep '"id": *2' | cut -c1-400
printf '%s\n' "$RESP" | grep '"id": *2' | grep -q '"result"' || fail "MCP scoring call"
printf '%s\n' "$RESP" | grep '"id": *2' | grep -qi '"isError": *true' && fail "MCP scoring call returned an error"

say "PASS: installed with the real commands, hooks + status line + MCP all run on Alpine"
