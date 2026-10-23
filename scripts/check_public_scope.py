#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import hashlib
import re
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN_PARTS = {
    "reference", "reproduction", "results", "plots", "checkpoints",
    "reviewer", "table10", "hotcrp", "weights", "mongo",
}
FORBIDDEN_IMPL = re.compile(r"(^|/)(cambricon_dg|race|idgnn|rowmean_baselines)(/|$)", re.I)
FORBIDDEN_BASELINE_SURFACE = re.compile(
    r"\bRACE\b|\bI[-_]?DGNN\b|\bCambricon(?:-DG)?\b", re.I
)
FORBIDDEN_CLI = re.compile(
    r"baselines-only|race-(?:tiles|pes-per-tile|freq-ghz|hbm-bw-gbps)|"
    r"idgnn-(?:engines|freq-ghz|ddr-bw-gbps)|"
    r"cambricon-(?:clusters|pes-per-cluster|freq-ghz|hbm-bw-gbps)",
    re.I,
)
UNRESOLVED_MARKER = re.compile(
    r"(?<![A-Za-z0-9_])(?:TODO|FIXME|HACK|XXX)(?![A-Za-z0-9_])|"
    r"^\s*#\s*if\s+0(?=$|[^A-Za-z0-9_])",
    re.I | re.M,
)
STALE_DCIM_ALIAS = re.compile(r"hybrid[-_ ]?[ab]", re.I)
FORBIDDEN_TEXT = re.compile(
    r"drive\.google|github\.com/yujin-in-ntu|REVIEW_DATA_ROOT|AE_AUTHORITY_ROOT|"
    r"output_to_user|HotCRP|main-results|exhaustive-full-results|TABLE10_|"
    r"neurachip\.us|mail@kaustubh|(?<![A-Za-z0-9_])system\s*\(|"
    r"(?<![A-Za-z0-9_])popen\s*\(",
    re.I,
)
ABSOLUTE_PATH = re.compile(r"(?:^|[\s'\"])(?:/Users/|/home/|[A-Za-z]:\\Users\\)")
SECRET = re.compile(
    r"(?i)(api[_-]?key|access[_-]?token|secret|password)\s*[:=]\s*['\"][^'\"]{8,}"
)
ALLOWED_SUFFIXES = {
    "", ".a", ".cc", ".cff", ".cpp", ".h", ".hpp", ".ini", ".json",
    ".md", ".py", ".rst", ".txt",
}
REQUIRED = {
    "README.md", "LICENSE.md", "NOTICE", "THIRD_PARTY_LICENSES.md",
    "CITATION.cff", ".zenodo.json", "VERSION", "SHA256SUMS",
    "LICENSES/Apache-2.0.txt", "LICENSES/CC-BY-4.0.txt",
    "scripts/run_synthetic_smoke.py",
    "src/NeuraSim/src/code/main.cpp",
    "src/NeuraSim/src/lib/neuratile/neuratile.cpp",
    "src/NeuraSim/src/lib/pe/dual_mode_spmm.cpp",
    "src/NeuraSim/src/lib/pe/pe_array_spmm.cpp",
    "src/NeuraSim/src/lib/pe/pe_cluster.cpp",
    "src/NeuraSim/src/lib/pe/work_stealing_scheduler.cpp",
    "src/NeuraSim/src/lib/pe/dynamic_row_mean_operator.h",
    "src/NeuraSim/src/lib/neuracim/neuracim.cpp",
    "src/NeuraSim/src/lib/result_sink/result_sink.cpp",
    "src/NeuraSim/src/lib/result_sink/result_sink.h",
    "configs/runtime/despoina_official_dcim_params.json",
    "configs/runtime/neurosim_official_dcim_primitive.json",
    "configs/runtime/official_parameter_provenance.json",
    "configs/runtime/digital_sequential_raw.json",
}

def tracked() -> list[str]:
    if (ROOT / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=ROOT, check=True, capture_output=True
        ).stdout
        return sorted(p.decode() for p in result.split(b"\0") if p)
    return sorted(
        path.relative_to(ROOT).as_posix()
        for path in ROOT.rglob("*")
        if path.is_file() and path.parts[len(ROOT.parts)] not in {"build", "dist"}
    )

def main() -> None:
    for probe in ("todo한글", "한글FIXME", "HACK注記", "XXX메모"):
        if not UNRESOLVED_MARKER.search(probe):
            raise SystemExit(f"unresolved-marker scanner self-test failed: {probe}")
    for probe in ("methodology", "prefixFIXMEsuffix", "has_xxx"):
        if UNRESOLVED_MARKER.search(probe):
            raise SystemExit(f"unresolved-marker scanner false positive: {probe}")
    paths = tracked()
    if len(paths) != len(set(paths)) or paths != sorted(paths):
        raise SystemExit("tracked allowlist is not sorted/unique")
    missing = sorted(REQUIRED - set(paths))
    if missing:
        raise SystemExit(f"required public files missing: {missing}")
    findings: list[str] = []
    for rel in paths:
        path = ROOT / rel
        parts = {part.lower() for part in Path(rel).parts}
        if parts & FORBIDDEN_PARTS or FORBIDDEN_IMPL.search(rel):
            findings.append(f"forbidden path: {rel}")
        if not path.exists():
            findings.append(f"missing tracked entry: {rel}")
            continue
        if path.is_symlink() or not path.is_file():
            findings.append(f"non-regular tracked entry: {rel}")
            continue
        if path.stat().st_size > 2 * 1024 * 1024:
            findings.append(f"file exceeds 2 MiB: {rel}")
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            findings.append(f"unapproved extension: {rel}")
        raw = path.read_bytes()
        if b"\0" in raw:
            findings.append(f"NUL/binary content: {rel}")
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            findings.append(f"non-UTF-8 content: {rel}")
            continue
        scanner_sources = {"scripts/check_public_scope.py", "tests/test_release.py"}
        if rel not in scanner_sources and FORBIDDEN_TEXT.search(text):
            findings.append(f"forbidden private/reviewer content: {rel}")
        if rel not in scanner_sources and ABSOLUTE_PATH.search(text):
            findings.append(f"possible author-machine absolute path: {rel}")
        if SECRET.search(text):
            findings.append(f"possible secret assignment: {rel}")
        owned_source = rel.startswith("src/NeuraSim/src/")
        if owned_source and FORBIDDEN_BASELINE_SURFACE.search(text):
            findings.append(f"competitor baseline recurrence: {rel}")
        if owned_source and FORBIDDEN_CLI.search(text):
            findings.append(f"competitor CLI recurrence: {rel}")
        first_party_public = (
            not rel.startswith("third_party/") and rel not in scanner_sources
        )
        if first_party_public and UNRESOLVED_MARKER.search(text):
            findings.append(f"unresolved marker/dead preprocessor block: {rel}")
        if first_party_public and STALE_DCIM_ALIAS.search(text):
            findings.append(f"stale DCIM architecture alias: {rel}")
        if rel.startswith(("src/NeuraSim/src/", "src/NeuraSim/tests/", "scripts/", "tests/")):
            first_lines = "\n".join(text.splitlines()[:5])
            if "SPDX-License-Identifier:" not in first_lines:
                findings.append(f"missing file-scoped SPDX identifier: {rel}")
    existing_paths = [p for p in paths if (ROOT / p).is_file()]
    if sum((ROOT / p).stat().st_size for p in existing_paths) > 20 * 1024 * 1024:
        findings.append("tracked public tree exceeds 20 MiB")
    zenodo = json.loads((ROOT / ".zenodo.json").read_text())
    version = (ROOT / "VERSION").read_text().strip()
    if zenodo.get("version") != version:
        findings.append("Zenodo metadata version does not match VERSION")
    if [c.get("name") for c in zenodo["creators"]] != ["Lee, Yujin", "Jang, Minsu"]:
        findings.append("creator order drift")
    if "license" in zenodo:
        findings.append("blanket Zenodo license field is not valid for this mixed-license archive")
    citation = (ROOT / "CITATION.cff").read_text()
    citation_version = re.search(r"(?m)^version:\s*(\S+)\s*$", citation)
    if not citation_version or citation_version.group(1) != version:
        findings.append("CITATION.cff version does not match VERSION")
    if re.search(r"(?m)^license\s*:", citation):
        findings.append("blanket CITATION.cff license field is not valid for this mixed-license archive")
    message = re.search(r'(?m)^message:\s*["\']?(.*?)["\']?$', citation)
    if not message or "LICENSE.md" not in message.group(1) or "NOTICE" not in message.group(1):
        findings.append("CITATION.cff message must point to LICENSE.md and NOTICE")
    readme = (ROOT / "README.md").read_text()
    if "file-scoped" not in readme or "LICENSE.md" not in readme or "NOTICE" not in readme:
        findings.append("README mixed-license guidance missing")
    checksum_entries: dict[str, str] = {}
    for line in (ROOT / "SHA256SUMS").read_text().splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match.group(2) in checksum_entries:
            findings.append("SHA256SUMS has malformed or duplicate entries")
            continue
        checksum_entries[match.group(2)] = match.group(1)
    expected_checksum_paths = set(paths) - {"SHA256SUMS"}
    if set(checksum_entries) != expected_checksum_paths:
        findings.append("SHA256SUMS path set does not match the public tree")
    else:
        for rel, expected in checksum_entries.items():
            if hashlib.sha256((ROOT / rel).read_bytes()).hexdigest() != expected:
                findings.append(f"SHA256 mismatch: {rel}")
    if findings:
        raise SystemExit("PUBLIC_SCOPE_FAIL\n" + "\n".join(findings))
    print(
        f"PUBLIC_SCOPE_PASS files={len(paths)} "
        f"bytes={sum((ROOT/p).stat().st_size for p in existing_paths)}"
    )

if __name__ == "__main__":
    main()
