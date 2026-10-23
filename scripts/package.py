#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION = (ROOT / "VERSION").read_text().strip()
PREFIX = f"despoina-simulator-{VERSION}/"
DIST = ROOT / "dist"
ARCHIVE = DIST / f"despoina-simulator-{VERSION}.tar.gz"

def main() -> None:
    if subprocess.run(["git", "status", "--porcelain"], cwd=ROOT,
                      check=True, capture_output=True, text=True).stdout:
        raise SystemExit("package requires a clean standalone repository")
    subprocess.run(["python3", "scripts/check_public_scope.py"], cwd=ROOT, check=True)
    DIST.mkdir(exist_ok=True)
    subprocess.run([
        "git", "archive", "--format=tar.gz", f"--prefix={PREFIX}",
        f"--output={ARCHIVE}", "HEAD"
    ], cwd=ROOT, check=True)
    digest = hashlib.sha256(ARCHIVE.read_bytes()).hexdigest()
    print(f"PACKAGE_PASS {ARCHIVE.relative_to(ROOT)} sha256={digest}")

if __name__ == "__main__":
    main()
