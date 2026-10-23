#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Minimal executable contract checks for the public reviewer interface."""
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
BINARY = ROOT / "build/despoina_sim"
FAULT_MARKER = "DESPOINA_INTERNAL_TEST_CORRUPT_DCIM_OUTPUT"


help_result = subprocess.run(
    [str(BINARY), "--help"], check=True, capture_output=True, text=True
)
help_text = help_result.stdout + help_result.stderr
for required in (
    "--profile despoina-h6-dcim-final",
    "--program_root",
    "--dataset",
    "--f-init",
    "--w-init-l1",
    "--w-init-l2",
):
    assert required in help_text, required
for forbidden in (
    "--experimental",
    "--tile_size",
    "--arch-variant",
    "--cim-mode",
    FAULT_MARKER,
):
    assert forbidden not in help_text, forbidden

conflict = subprocess.run(
    [
        str(BINARY),
        "--profile",
        "despoina-h6-dcim-final",
        "--tile_size",
        "32",
    ],
    capture_output=True,
    text=True,
)
assert conflict.returncode != 0
assert "unavailable in the closed official profile" in conflict.stderr

missing = subprocess.run(
    [str(BINARY), "--profile", "despoina-h6-dcim-final"],
    capture_output=True,
    text=True,
)
assert missing.returncode != 0
assert "requires explicit --program_root" in missing.stderr

print("PUBLIC_CLI_CONTRACT_PASS")
