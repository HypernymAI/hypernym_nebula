# Fully static Linux scorer

On Linux, nebula's native kernel is **one fully static program per CPU**:
`ctxshield/ctxshield/static/ctxshield-scorer-linux-{x86_64,aarch64}`. It loads nothing from the system (no libc,
libstdc++, libgcc or dynamic loader), so the same file runs on every distribution: glibc or musl (Alpine), old or new.
macOS uses the pybind11 module `_native.cpython-313-darwin.so` instead, because macOS has no static libc.

## Why a program and not a `.so`

A Python extension is loaded into the interpreter's process and must share the interpreter's libc. It can't be
fully static, and a musl build can't load into a glibc Python (or the other way round). A separate program has no
such tie, so it is linked `-static -no-pie` against musl and libstdc++ and spoken to over a pipe.

## How a score is computed on Linux

```
shield code ──► ctxshield/features.py ──► _native_static.py ──stdin/stdout──► ctxshield-scorer
               (picks the core)          (pipe client, same API as _native)  (same C++ kernel as _native)
```

1. **Choosing the core:** `features.py` imports `_native_static as _native` when `sys.platform` is Linux, or when
   `CTXSHIELD_SCORER=static`. Otherwise it imports the pybind11 `_native`. Every caller (`score_logits`,
   `featurize`, `make_scorer`, tests via `features._native`) goes through that one name.
2. **Starting the scorer:** on first use in a process, the client starts the bundled binary for
   `platform.machine()` and sends HELLO. It checks the protocol, `DIM_BITS` and ABI against its own constants and
   raises on any mismatch.
   - There is one scorer per OS process. A forked child starts its own; the client tracks the owning pid.
   - At exit, `atexit` closes stdin, and the scorer exits on EOF.
   - If the scorer dies, the next call raises with its exit code. It is never restarted silently.
3. **Weights stay resident:** `Scorer(...)` sends the weights once (CREATE) and gets a handle back; `score()` sends
   only the docs.
   - `score_batch()` (the API takes weights on every call) keeps up to 16 scorers keyed by the weights' *content*.
     It compares against a private copy with `np.array_equal`, so mutating an array in place can't return stale
     scores.
4. **Errors:** a bad request (wrong shapes, unknown backend, invalid UTF-8) comes back as an error response, which
   the client raises as `RuntimeError`, and the scorer keeps serving. A broken stream (bad magic, short read) makes
   the scorer exit with code 2.
5. **Backends:** CPU only. `metal` and `cuda` report unavailable and refuse to create, with the same messages as a
   pybind module built without them.

The kernel files (`src/featurize.cpp`, `src/score.cpp`, `src/c_abi.cpp`, `third_party/utf8proc`) are shared with
the pybind module. Only `scorer/scorer_main.cpp` (transport) and `python/bind.cpp` (pybind) differ, so both cores
give bit-identical results for the same CPU baseline.

## Wire protocol (CXS1)

Little-endian, one response per request, strictly in order. The full op table lives at the top of
`scorer/scorer_main.cpp`, which is the only definition. If you change it, bump `kProtocol` there and `PROTOCOL` in
`_native_static.py` together.

```
request  "CXS1" u32 op     u64 len payload      ops: 1 HELLO  2 AVAILABLE  3 FEATURIZE  4 CREATE  5 SCORE  6 DESTROY
response "CXR1" u32 status u64 len payload      status 0 ok, 1 error (payload = message)
docs     u64 n_docs  u32 n_fields  u64 lens[n_docs*n_fields]  concatenated UTF-8
```

## Things outside ctxshield that depend on this

- **`shield/noegress.py`:** every nebula entry point blocks launching programs except an allowlist. On top of that,
  it allows the two bundled scorer paths and nothing else. A same-named binary elsewhere, or a
  `CTXSHIELD_SCORER_BIN` pointing outside the plugin, is blocked, so that override only works outside nebula's
  entry points (development and tests).
- **`bin/nebula-run`:** on Linux, accepts x86-64 / ARM64 and requires the scorer for this CPU to exist and be
  executable. There is no libc check.
- **`shield/nebula_statusline.py`:** the shared status-line launcher (`~/.hypernym-nebula/bin/statusline`) runs the
  absolute path of the interpreter that wrote it, not `python3` from PATH, because minimal Linux has none. Only the
  newest copy (by `VERSION`) rewrites the launcher.
- **Settings writers** (`settings.py`, `customize.py`, `privacy.py`): write through per-process temp files
  (`<name>.<pid>.tmp`). On first run, the status line's `--check` and the SessionStart hook both run the one-time
  setup at the same moment.
- **`native/PLATFORMS.json`:** sha256, CPU baseline and test matrix of every shipped core (`make install` writes the
  Linux entries).
- **Git:** the binaries must stay mode 755; plugin installs keep the exec bit.

## Build and test

You need Docker with buildx. On an x86 host the ARM64 build and distro tests run under qemu emulation: slower, but
no extra setup. `test-install` is the exception: it runs Claude Code itself, whose installer doesn't finish under
emulation, so it defaults to the host's own CPU. Run it on an ARM64 machine (Apple Silicon Docker, or an ARM64 CI
runner) to cover ARM64.
Every image and tool version is pinned at the top of the `Makefile`.

```
make                  # build both scorers into out/ (Alpine; fails unless no NEEDED entries and no interpreter)
make verify           # run each scorer bare on alpine, debian 12, ubuntu 18.04 and busybox (no libc at all)
make install          # copy into ctxshield/static/ (mode 755) and update PLATFORMS.json
make test             # install, then everything below
make test-x86_64      # per CPU, on alpine / debian 12 / ubuntu 18.04 (test_in_container.sh): the ctxshield
make test-aarch64     #   suites on the static core, an MCP shield_score_reply through nebula-run, and
                      #   noegress refusing a non-bundled scorer
make test-install     # this host's CPU (INSTALL_CPUS to override), on clean Alpine (test_claude_install.sh): the
                      #   published uv line, Claude Code's
                      #   installer, claude plugin validate / marketplace add / install from this tree, then
                      #   every hook command exactly as the installed hooks.json writes it, a 12-way first-run
                      #   config race, the status line, a Stop hook that must record both scored replies, and MCP
make clean
```

| File | Role |
|---|---|
| `Dockerfile.build` | Alpine build of `ctxshield-scorer` (`SHIELD_PYTHON=OFF SHIELD_STATIC=ON`, CPU baseline from `SHIELD_TARGET_ARCH`). |
| `test_in_container.sh` | The per-distribution test. |
| `test_claude_install.sh` | The real install flow on Alpine. |
| `update_platforms.py` | Writes the Linux entries of `PLATFORMS.json`. |

CMake options used here (`../CMakeLists.txt`):
- `SHIELD_PYTHON=OFF` skips pybind11 and Python headers.
- `SHIELD_STATIC=ON` links `-static -no-pie -s` (Linux only).
- `SHIELD_TARGET_ARCH` selects the portable baseline: `x86-64-v2` or `armv8-a`.

To build the scorer dynamically on a Mac for development: `cmake -DSHIELD_PYTHON=OFF -DSHIELD_METAL=OFF`, then run
anything with `CTXSHIELD_SCORER=static CTXSHIELD_SCORER_BIN=<build>/ctxshield-scorer`.

## When you change something

- **Kernel** (`include/`, `src/`, `third_party/`) or **transport** (`scorer/`): run `make test`, then commit both
  binaries in `ctxshield/ctxshield/static/` together with `PLATFORMS.json`. The macOS `.so` is rebuilt separately
  (`../build.sh`). Both cores must keep producing identical scores.
- **The pipe client:** keep its API identical to `python/bind.cpp`. `tests/test_backends.py` runs against whichever
  core `features.py` picked.
