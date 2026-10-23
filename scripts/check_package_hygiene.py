#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Audit the committed release archive without extracting it."""
from __future__ import annotations

import hashlib
import re
import tarfile
from pathlib import Path, PurePosixPath


ROOT = Path(__file__).resolve().parents[1]
VERSION = (ROOT / "VERSION").read_text().strip()
PREFIX = f"despoina-simulator-{VERSION}"
ARCHIVE = ROOT / "dist" / f"{PREFIX}.tar.gz"


def main() -> None:
    with tarfile.open(ARCHIVE, "r:gz") as stream:
        members = stream.getmembers()
        names = [PurePosixPath(member.name) for member in members]
        for member, path in zip(members, names):
            if path.is_absolute() or ".." in path.parts:
                raise SystemExit(f"unsafe archive path: {member.name}")
            if not path.parts or path.parts[0] != PREFIX:
                raise SystemExit(f"archive is not single-rooted: {member.name}")
            if ".git" in path.parts:
                raise SystemExit(f"Git metadata in archive: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise SystemExit(f"non-regular archive entry: {member.name}")

        files = {
            path.relative_to(PREFIX).as_posix(): stream.extractfile(member).read()
            for member, path in zip(members, names)
            if member.isfile()
        }
    checksum_text = files.get("SHA256SUMS", b"").decode("utf-8")
    checksums: dict[str, str] = {}
    for line in checksum_text.splitlines():
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match or match.group(2) in checksums:
            raise SystemExit("archive SHA256SUMS is malformed or duplicated")
        checksums[match.group(2)] = match.group(1)
    if set(checksums) != set(files) - {"SHA256SUMS"}:
        raise SystemExit("archive SHA256SUMS path set mismatch")
    for path, expected in checksums.items():
        if hashlib.sha256(files[path]).hexdigest() != expected:
            raise SystemExit(f"archive checksum mismatch: {path}")
    print(f"PACKAGE_HYGIENE_PASS files={len(files)} root={PREFIX}")


if __name__ == "__main__":
    main()
