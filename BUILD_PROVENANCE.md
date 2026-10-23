# Build provenance

This standalone real-simulator tree was assembled from the exact source snapshot
`a439368fe03406a4e5120ef5eae7403d50c6ab4f` using an explicit file allowlist.
The snapshot identifier is included for byte-level provenance only; this
standalone repository contains no parent commits, remote configuration, private
branch name, or prior repository history.

The included simulator C++ files are source-derived from that commit. Changes
are limited to standalone dependency/build integration, license/change headers,
safe removal of external process/service hooks, fail-closed output handling,
and the bounded synthetic test/packaging surface. Scientific algorithms were
not replaced with a surrogate.

The 1.0.0 release is a zero-parent-lineage successor of public root
`fc0054510fa056183870138482190890f8b72c21`. Its FP32 PE authority is the
validated public-source treatment recorded by authority commit
`1afd8d141679e0fd173be4eea6e4b7595ade6a39`: Accelergy 0.4 with the Aladdin
estimator reports 38.55419140625 pJ/action at 45 nm, scaled to 22 nm by
`(22/45)^2`. See `configs/runtime/official_parameter_provenance.json` for raw
and output hashes and exact consumers.

The Despoina official DCIM/factorized row-mean path uses only the eligible
NeuroSim 1.4 SRAM read primitive, the Accelergy digital datapath constants, and
CACTI FIFO-access constants listed in that manifest. The tools themselves are
not bundled. The official runtime SHA-256 gates the reduced
`neurosim_official_dcim_primitive.json`; the generic raw NeuroSim artifact is
provenance-only and is not a runtime input. Generic analog/ACIM fields and full
processing-unit accumulation, transport, area, leakage, total energy,
unavailable latency, and weight-program fields are not consumers of the
official profile.

The energy claim is limited to supported-component steady-state read/compute
and native FIFO accesses. Resident-weight source/replicated bytes and load
events are recorded, while program latency and program energy are unsupported
and excluded. This is not a full-chip or deployment-energy claim.

## 1.1.0

Version 1.1.0 keeps the 1.0.0 tree and changes five simulator sources
(`dual_mode_spmm.{h,cpp}`, `neuratile.{h,cpp}`, `globalcounters.cpp`) so the
datapath follows the paper's description. The changes were developed and
validated in the authors' artifact repository and merged three-way onto this
tree; the release-specific accuracy gate was adapted to compare accumulated Z
rows. See the "Release 1.1.0" section of `README.md` for the full list. No
parameter, energy constant, input or gate threshold changed.
