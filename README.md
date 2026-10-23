# Despoina self-contained simulator

This public release is not intended to serve as a complete provenance archive of the paper’s experimental environment. Its purpose is to provide the core Despoina implementation as a self-contained, executable simulator.

The official profile exercises the NeuraSim-derived simulator driver, Despoina
PE/cluster/array path, work-stealing scheduler, cache and staging structures,
IndexStore, the Despoina official DCIM/factorized row-mean path, and vendored
DRAMsim3.

The validated release scope is deterministic, steady-state, resident-weight
inference. One-time model deployment and weight programming are outside the
Figure 8(b) accounting scope and are not presented as supported energy terms.
The release makes no paper-results-reproduction claim and contains no datasets,
trained weights, checkpoints, or publication results.

## Release 1.1.0: datapath aligned with the paper description

Version 1.0.0 (DOI `10.5281/zenodo.21510101`) is the release cited in the
MICRO 2026 artifact appendix. Version 1.1.0 changes the simulator so that its
datapath follows the paper's description (§IV-B–E, Figure 3). Simulator outputs
therefore differ from 1.0.0. The reviewer interface, official profile, inputs
and gates are unchanged.

- **ΔY streaming.** The PE accumulator starts at zero and holds only the
  three-phase change of a row, ΔY = Ph1 + Ph2 − Ph3. The row-mean-scaled ΔY is
  quantized and sent to DCIM once per changed row. No previous Y is stored, so
  the old-Y HBM snapshot of 1.0.0 is gone (its counters report zero).
- **Z accumulation.** The committed Z row is updated in place,
  Z′ = Z_old · (d_old/d_new) + Q(ΔY)·W, which keeps the row mean exact in Z
  space. Whole Z layers are placed in the 4 MB Post-Combination Buffer in layer
  order while they fit; the remaining layers read-modify-write Z in HBM. Z
  traffic of every layer is charged.
- **NPF state.** H′ is committed only for rows that pass ‖dH‖∞ > τ; rows below τ
  keep H_prev, so their change accumulates until it passes. H lives in HBM:
  H′ writes and H_prev reads are charged. The output layer is not τ-gated and
  needs no H_prev read.
- **Distance bound.** Each row of a τ-gated layer keeps an upper bound
  b ≥ ‖NormReLU(Z) − H_prev‖∞, grown by ‖NormReLU(Z_new) − NormReLU(Z_old)‖∞ when
  Z changes. A changed row with b ≤ τ cannot pass, so it is dropped without
  reading H_prev. Decisions are identical to the exact comparison; the run
  fails closed if a skipped row would have propagated. The bound (4 B per row)
  is read and written in HBM. Its compute, 774 scalar operations per changed
  row counted from the executed code, is charged as 392 FP32 fused-MAC events
  (two scalar operations per fused MAC, rounded up to a multiple of eight) in
  the row-mean energy owner, and on the 2,048-operation-per-cycle PE lanes as
  ceil(774 × changed rows / 2,048) cycles per layer, serially.
- **Area model.** The Z buffer is priced at the configured 4 MB.
- **Accuracy gate.** The higher-layer official DCIM comparison now reads the
  accumulated Z rows, because DCIM emits Q(ΔY)·W rather than complete rows.

## Validated platform and dependencies

The release gate is run on macOS with Apple Clang, CMake, Make, Python 3,
Homebrew Boost.Program_options, and Homebrew `libomp`. It is also validated in
an Ubuntu 24.04 container with GCC 13.3, CMake, Make, Python 3,
Boost.Program_options, and GNU OpenMP.

## Official reviewer interface

The public reviewer path has four commands:

```sh
make build
make test
make smoke
make verify
```

`make test` runs the official parameter/energy/accuracy contract unit, the
real dynamic row-mean operator unit, and a minimal executable CLI contract
check. `make verify` is exactly the concise reviewer gate: `make test` followed
by `make smoke`.

`make smoke` generates all inputs locally. It uses deterministic nonuniform
features, two non-identity signed/permuted weight matrices, a degree-changing
edge insertion, and a sparse feature update. It runs the real t0+t1
two-layer simulator path twice and requires byte-identical selected evidence.
Both the per-step official DCIM comparison and final higher-layer comparison must
report `valid=true` and `pass=true`. Scheduler, staging, IndexStore, dynamic
row mean, DCIM, and DRAMsim3 evidence must be nonzero.

A separate logical-tile reference micro-model was deliberately removed from the
shipped tree because it could be mistaken for the production simulator path.
Its relevant real-path coverage is provided by the production-driver smoke,
including two-layer DCIM dependency, FIFO/staging activity, scheduler work,
resident-weight evidence, and accuracy-gate checks.

The official command selected by the smoke is equivalent to:

```sh
build/despoina_sim \
  --profile despoina-h6-dcim-final \
  --program_root INPUT/programs --dataset DATASET \
  --timesteps 2 --delta-dir INPUT/deltas \
  --f-init INPUT/features.bin \
  --w-init-l1 INPUT/weights_l1.bin \
  --w-init-l2 INPUT/weights_l2.bin \
  --dram-config third_party/DRAMsim3-1.0.0/configs/HBM2_8Gb_x128.ini
```

Only explicit dataset/input/output naming, timestep count, propagation tau, and
DRAM configuration are official-profile overrides. Architecture,
SpMM/DCIM/indexer modes, normalization semantics, D128, 16 lanes, FIFO16,
parameter hashes, and final FP32/22 nm energy settings are fixed. Conflicting
architecture controls, implicit input defaults, missing feature/weight files,
and generated random weights are rejected. Non-official research controls are
outside the validated claim and are shown only by `--help --experimental`.

## Maintainer-only release checks

The reviewer path intentionally excludes micro-model, fault-injection, static
sanitation, public-scope, and archive-hygiene checks. Maintainers run them
separately:

```sh
make maintainer-check
```

This target retains the A-load burst and HAGG staging micro-units, runs the
release/static integration assertions and public-scope audit, builds a separate
fault-enabled binary, proves the ordinary `build/despoina_sim` contains no
active corruption hook, verifies runtime corruption fails closed only in the
fault binary, and creates and audits the release archive. The ordinary release
build never enables the test fault-injection macro.

## Parameter and energy provenance

`configs/runtime/official_parameter_provenance.json` is the machine-readable
authority for every official Accelergy, CACTI, and NeuroSim constant. The typed
Despoina official DCIM parameters and reduced
`configs/runtime/neurosim_official_dcim_primitive.json` are SHA-256 gated at
runtime. The generic raw NeuroSim output is retained only for provenance and is
never parsed by the official runtime. Full processing-unit accumulation,
transport, area, leakage, generic analog/ACIM, unavailable latency, and weight
programming fields are excluded by the reduced primitive and compile-time
consumer assertions.

FP32 PE actions use 38.55419140625 pJ/action at 45 nm and
`E22 = E45 × (22/45)^2`, yielding 9.214927723765432 pJ/action. Main PE
multiply-plus-add counters are converted to fused-MAC events by division by
two. Dynamic row-mean FP32 scalar operations remain separate events and are
never divided by two or grouped as an eight-lane FP16 event. The NPF
distance-bound compute added in 1.1.0 is booked in the same owner as fused-MAC
events, already converted from scalar operations as described above.

The reported Despoina official DCIM energy is supported-component steady-state
read/compute plus native FIFO-access energy. It is neither full-chip energy nor
model-deployment energy. The runtime quantizes and hash-checks the resident
weight image and records its bytes/load events, but does not claim weight
program latency or energy; zero program timing means not applicable.

## Reproducible archive

After committing a clean standalone tree:

```sh
make package
```

The package contains one root directory and is reproducible for a fixed commit.

## Licensing

Licensing is file-scoped. See `LICENSE.md`, `NOTICE`, and
`THIRD_PARTY_LICENSES.md`. NeuraSim-derived files are CC BY 4.0 with upstream
DOI `10.5281/zenodo.10896280` and modification notices. New Despoina files are
Apache-2.0. Vendored dependencies retain their own licenses.
