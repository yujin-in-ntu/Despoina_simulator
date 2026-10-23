#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Prove the internal corruption hook exists only in the fault-test binary."""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path


MARKER = b"DESPOINA_INTERNAL_TEST_CORRUPT_DCIM_OUTPUT"


def help_text(binary: Path) -> bytes:
    result = subprocess.run(
        [str(binary), "--help"], check=True, capture_output=True
    )
    return result.stdout + result.stderr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normal", type=Path, required=True)
    parser.add_argument("--fault", type=Path, required=True)
    args = parser.parse_args()
    normal = args.normal.resolve()
    fault = args.fault.resolve()

    if MARKER in normal.read_bytes():
        raise SystemExit("normal release binary contains the internal fault marker")
    if MARKER not in fault.read_bytes():
        raise SystemExit("fault-test binary does not contain the internal fault marker")
    if MARKER in help_text(normal) or MARKER in help_text(fault):
        raise SystemExit("internal fault marker leaked through command-line help")
    print("FAULT_BINARY_ISOLATION_PASS")


if __name__ == "__main__":
    main()
