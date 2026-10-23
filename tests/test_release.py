#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Static integration assertions for the standalone real simulator release."""
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MAKEFILE = (ROOT / "Makefile").read_text()
SIM = ROOT / "src/NeuraSim/src"

required = [
    SIM / "code/main.cpp",
    SIM / "lib/neuratile/neuratile.cpp",
    SIM / "lib/pe/dual_mode_spmm.cpp",
    SIM / "lib/pe/pe_array_spmm.cpp",
    SIM / "lib/pe/pe_cluster.cpp",
    SIM / "lib/pe/work_stealing_scheduler.cpp",
    SIM / "lib/pe/delta_f_aware_indexer.cpp",
    SIM / "lib/pe/dynamic_row_mean_operator.h",
    SIM / "lib/neuracim/neuracim.cpp",
    SIM / "lib/result_sink/result_sink.cpp",
    ROOT / "third_party/DRAMsim3-1.0.0/src/memory_system.cc",
]
for path in required:
    assert path.is_file(), path
    if path.suffix == ".cpp":
        assert path.name in MAKEFILE

for excluded in ("cambricon_dg", "race", "idgnn", "rowmean_baselines", "mongo"):
    assert not (SIM / "lib" / excluded).exists(), excluded

source = "\n".join(
    path.read_text(errors="ignore")
    for path in (SIM / "lib").rglob("*")
    if path.is_file()
)
baseline_surface = re.compile(r"\bRACE\b|\bI[-_]?DGNN\b|\bCambricon(?:-DG)?\b", re.I)
baseline_cli = re.compile(
    r"baselines-only|race-(?:tiles|pes-per-tile|freq-ghz|hbm-bw-gbps)|"
    r"idgnn-(?:engines|freq-ghz|ddr-bw-gbps)|"
    r"cambricon-(?:clusters|pes-per-cluster|freq-ghz|hbm-bw-gbps)",
    re.I,
)
assert not baseline_surface.search(source)
assert not baseline_cli.search(source)
assert not re.search(r"hybrid[-_ ]?[ab]", source, re.I)
unresolved_marker = re.compile(
    r"(?<![A-Za-z0-9_])(?:TODO|FIXME|HACK|XXX)(?![A-Za-z0-9_])|"
    r"^\s*#\s*if\s+0(?=$|[^A-Za-z0-9_])",
    re.I | re.M,
)
for probe in ("todo한글", "한글FIXME", "HACK注記", "XXX메모"):
    assert unresolved_marker.search(probe), probe
for probe in ("methodology", "prefixFIXMEsuffix", "has_xxx"):
    assert not unresolved_marker.search(probe), probe
assert not unresolved_marker.search(source)
assert not re.search(r"(?<![A-Za-z0-9_])system\s*\(", source)
assert not re.search(r"(?<![A-Za-z0-9_])popen\s*\(", source)
assert "mail@kaustubh" not in source
assert "neurachip.us" not in source
assert "ResultCollection" in source
assert "component_metrics.jsonl" in source
assert (ROOT / "configs/runtime/neurosim_official_dcim_primitive.json").is_file()
assert "DESPOINA_DCIM_PRIMITIVE_JSON" in source
assert "DESPOINA_DCIM_RAW_JSON" not in source

binary = ROOT / "build/despoina_sim"
help_result = subprocess.run(
    [str(binary), "--help"], check=True, capture_output=True, text=True
)
help_text = help_result.stdout + help_result.stderr
for forbidden in (
    "--experimental", "--tile_size", "--arch-variant", "--cim-mode",
    "--baselines-only", "--race-", "--idgnn-", "--cambricon-",
    "DESPOINA_INTERNAL_TEST_CORRUPT_DCIM_OUTPUT",
):
    assert forbidden not in help_text, forbidden
for required_help in (
    "--profile despoina-h6-dcim-final", "--program_root", "--dataset",
    "--f-init", "--w-init-l1", "--w-init-l2",
):
    assert required_help in help_text, required_help

expanded = subprocess.run(
    [str(binary), "--help", "--experimental"],
    check=True, capture_output=True, text=True,
).stdout
assert "--arch-variant" in expanded
assert not baseline_surface.search(expanded)
assert not baseline_cli.search(expanded)
assert "DESPOINA_INTERNAL_TEST_CORRUPT_DCIM_OUTPUT" not in expanded

conflict = subprocess.run(
    [
        str(binary), "--profile", "despoina-h6-dcim-final",
        "--tile_size", "32",
    ],
    capture_output=True, text=True,
)
assert conflict.returncode != 0
assert "unavailable in the closed official profile" in conflict.stderr

missing_inputs = subprocess.run(
    [str(binary), "--profile", "despoina-h6-dcim-final"],
    capture_output=True, text=True,
)
assert missing_inputs.returncode != 0
assert "requires explicit --program_root" in missing_inputs.stderr

print("RELEASE_STATIC_INTEGRATION_PASS")
