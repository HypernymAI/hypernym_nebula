#!/usr/bin/env python3
"""
Record the installed static scorers in native/PLATFORMS.json (sha256, targets, tested distributions) and drop the
glibc-only pybind11 Linux modules from the manifest (Linux now runs the static scorer). Run by `make install`.

    python3 update_platforms.py <PLATFORMS.json> <static dir> <stamp>
NO .get().
"""
import hashlib
import json
import sys
from pathlib import Path

TESTED = ['Alpine 3.20 (musl)', 'Debian 12 (glibc 2.36)', 'Ubuntu 18.04 (glibc 2.27)', 'busybox (no libc)']
BASELINE = {'x86_64': 'x86-64-v2', 'aarch64': 'armv8-a'}


def main() -> None:
    manifest, static_dir, stamp = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
    m = json.loads(manifest.read_text())
    mods = m['modules']
    for k in [k for k in mods if k.endswith('-linux-gnu.so')]:
        del mods[k]
    for cpu in ('x86_64', 'aarch64'):
        f = static_dir / f'ctxshield-scorer-linux-{cpu}'
        mods[f'static/{f.name}'] = {
            'targets': [f'linux-{cpu}'],
            'kind': 'static PIE executable (musl + libstdc++ linked in, ASLR, stack protector, FORTIFY, full RELRO), '
                    'reproducible build, CXS1 pipe protocol',
            'floor': 'no libc requirement (glibc or musl): nothing is loaded from the system',
            'cpu_baseline': BASELINE[cpu],
            'sha256': hashlib.sha256(f.read_bytes()).hexdigest(),
            'tested': TESTED,
        }
    m['built'] = stamp
    m['backend'] = 'cpu only (no Metal / CUDA in shipped modules)'
    manifest.write_text(json.dumps(m, indent=1, ensure_ascii=False) + '\n')
    print(f"PLATFORMS.json: {len(mods)} modules")


if __name__ == '__main__':
    main()
