"""
NO-EGRESS GUARANTEE — imported FIRST by every nebula entry point (hooks, badge, MCP server, sync, backfill, status,
privacy, badge installer, session start). On import it installs a CPython audit hook (PEP 578) that RAISES on every
network operation the interpreter can perform, and on launching any program except a short allowlist.

  - audit hooks cannot be removed once installed (by design of PEP 578) — no code later in the process can undo this
  - blocked: socket connect / bind / sendto / sendmsg / name resolution, urllib requests, http/ftp/smtp/imap/poplib/
    telnetlib connections, webbrowser, os.system, os.exec*/spawn*/posix_spawn, and subprocess of anything not allowed
  - allowed programs: `ps` and `osascript` (terminal-tab background colour, termbg.py — local IPC with Terminal.app,
    no network) and this same Python interpreter (the badge's catch-up sync)
  - the MCP server talks to Claude Code over stdin/stdout pipes — no socket is opened

tests/test_privacy.py proves it: every entry point imports this first (static check), no network module is imported
anywhere (static check), and a real connection attempt inside a hook process raises (runtime check).
NO .get(). Imports at top.
"""
import os
import sys

BLOCKED_EVENTS = frozenset({
    'socket.connect', 'socket.bind', 'socket.sendto', 'socket.sendmsg', 'socket.getaddrinfo',
    'socket.gethostbyname', 'socket.gethostbyaddr', 'socket.getnameinfo',
    'urllib.Request', 'http.client.connect', 'ftplib.connect', 'smtplib.connect', 'imaplib.open', 'poplib.connect',
    'telnetlib.Telnet.open', 'webbrowser.open', 'os.system', 'os.exec', 'os.spawn', 'os.posix_spawn', 'os.startfile',
})
ALLOWED_PROGRAMS = frozenset({'ps', 'osascript'})
_SELF = os.path.realpath(sys.executable)


class EgressBlocked(PermissionError):
    pass


def _allowed_program(exe) -> bool:
    if isinstance(exe, (bytes, os.PathLike)):
        exe = os.fsdecode(exe)
    if not isinstance(exe, str):
        return False
    if os.path.realpath(exe) == _SELF:
        return True
    base = os.path.basename(exe)
    return base in ALLOWED_PROGRAMS and (exe == base or exe in (f'/bin/{base}', f'/usr/bin/{base}'))


def _hook(event: str, args) -> None:
    if event in BLOCKED_EVENTS:
        raise EgressBlocked(f"nebula no-egress: '{event}' is blocked — this plugin never uses the network")
    if event == 'subprocess.Popen':
        exe, argv = args[0], args[1]
        prog = exe if exe is not None else (argv[0] if isinstance(argv, (list, tuple)) and argv else argv)
        if not _allowed_program(prog):
            raise EgressBlocked(f"nebula no-egress: launching {prog!r} is blocked (allowed: ps, osascript, itself)")


if not getattr(sys, '_ctxshield_noegress', False):
    sys.addaudithook(_hook)
    sys._ctxshield_noegress = True
