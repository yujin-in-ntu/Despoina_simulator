#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate and run a redistribution-safe t0+t1 Despoina workload."""
from __future__ import annotations

import argparse
import json
import os
import struct
import subprocess
from pathlib import Path


N = 4
D = 128


def lines(path: Path, values: list[object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{value}\n" for value in values))


def matrix(
    root: Path,
    name: str,
    rows: int,
    cols: int,
    row_ptr: list[int],
    col_idx: list[int],
    data: list[float],
    cursor: int,
    with_csc: bool,
) -> tuple[int, list[tuple[int, object]]]:
    directory = root / name
    directory.mkdir(parents=True, exist_ok=True)
    entries: list[tuple[int, object]] = []

    def place(values: list[object]) -> int:
        nonlocal cursor
        address = cursor
        entries.extend((cursor + 4 * index, value) for index, value in enumerate(values))
        cursor += 4 * len(values)
        return address

    row_address = place(row_ptr)
    col_address = place(col_idx)
    data_address = place(data)
    col_ptr: list[int] = []
    row_idx: list[int] = []
    if with_csc:
        buckets: list[list[int]] = [[] for _ in range(cols)]
        for row in range(rows):
            for offset in range(row_ptr[row], row_ptr[row + 1]):
                buckets[col_idx[offset]].append(row)
        col_ptr = [0]
        for bucket in buckets:
            row_idx.extend(bucket)
            col_ptr.append(len(row_idx))
        col_ptr_address = place(col_ptr)
        row_indices_address = place(row_idx)
    else:
        col_ptr_address = -1
        row_indices_address = -1
    header = [
        f"name,{name[-1]}",
        f"x_dim,{rows}",
        f"y_dim,{cols}",
        f"type,{'CSR_CSC' if with_csc else 'CSR'}",
        f"nnz,{len(col_idx)}",
        f"row_ptr_address,{row_address}",
        f"col_indices_address,{col_address}",
        f"data_address,{data_address}",
        f"col_ptr_address,{col_ptr_address}",
        f"row_indices_address,{row_indices_address}",
    ]
    (directory / "header.csv").write_text("\n".join(header) + "\n")
    lines(directory / "row_ptr.txt", row_ptr)
    lines(directory / "col_indices.txt", col_idx)
    lines(directory / "data.txt", data)
    if with_csc:
        lines(directory / "col_ptr.txt", col_ptr)
        lines(directory / "row_indices.txt", row_idx)
    return cursor, entries


def generate(root: Path) -> tuple[Path, Path, Path, Path, Path]:
    programs = root / "programs"
    dataset = programs / "synthetic4"
    matrix_root = dataset / "matrix"
    (dataset / "bin").mkdir(parents=True, exist_ok=True)
    (dataset / "bin/spmm.csv").write_text("# PE-array runtime entry\n")
    row_ptr = [0, 1, 2, 3, 4]
    col_idx = [1, 2, 3, 0]
    cursor = 0
    all_entries: list[tuple[int, object]] = []
    cursor, entries = matrix(
        matrix_root, "mat_A", N, N, row_ptr, col_idx, [1.0] * 4, cursor, True
    )
    all_entries.extend(entries)
    b_cols = [0, 1, 2, 3]
    cursor, entries = matrix(
        matrix_root, "mat_B", N, D, row_ptr, b_cols, [1.0, -0.5, 0.25, 0.75],
        cursor, True
    )
    all_entries.extend(entries)
    cursor, entries = matrix(
        matrix_root, "mat_C", N, D, row_ptr, [0, 0, 0, 0], [0.0] * 4,
        cursor, False
    )
    all_entries.extend(entries)
    raw = dataset / "raw_data"
    raw.mkdir(parents=True, exist_ok=True)
    payload = "".join(f"{address},{value}\n" for address, value in all_entries)
    for name in ("dram_raw_data.csv", "dram_raw_data_0.csv", "dram_raw_data_1.csv"):
        (raw / name).write_text(payload)

    deltas = root / "deltas"
    deltas.mkdir(parents=True, exist_ok=True)
    (deltas / "delta_a_t1.txt").write_text(
        "# Add one directed edge; row 0 degree changes from 1 to 2.\nadd 0 2 1.0\n"
    )
    (deltas / "delta_f_t1.txt").write_text(
        "# Update one 128-wide feature row; omitted values are zero.\n1 0.125 -0.0625 0.03125\n"
    )
    features = root / "features.bin"
    with features.open("wb") as stream:
        for row in range(N):
            for col in range(D):
                value = ((row + 1) * ((col % 11) - 5)) / 32.0
                stream.write(struct.pack("<f", value))
    weights_l1 = root / "weights_l1.bin"
    with weights_l1.open("wb") as stream:
        for row in range(D):
            for col in range(D):
                target = (row * 5 + 7) % D
                value = (-1.0 if row % 3 == 0 else 1.0) if col == target else 0.0
                stream.write(struct.pack("<f", value))
    weights_l2 = root / "weights_l2.bin"
    with weights_l2.open("wb") as stream:
        for row in range(D):
            for col in range(D):
                target = (row * 3 + 1) % D
                value = (-0.5 if row % 2 else 0.5) if col == target else 0.0
                stream.write(struct.pack("<f", value))
    return programs, deltas, features, weights_l1, weights_l2


def selected(stats: dict) -> dict:
    t1 = stats["per_timestep"][1]
    dynamic = t1["dynamic_row_mean"]
    return {
        "timesteps": stats["timesteps"],
        "modes": [row["mode_used"].lower() for row in stats["per_timestep"]],
        "cycles": [row["cycles"] for row in stats["per_timestep"]],
        "phase1_nnz": t1["phase1"]["nnz"],
        "phase2_nnz": t1["phase2"]["nnz"],
        "phase3_nnz": t1["phase3"]["nnz"],
        "degree_rows": dynamic["graph_changed_degree_rows"],
        "degree_writes": dynamic["degree_write_events"],
        "degree_reads": dynamic["degree_read_events"],
        "reciprocals": dynamic["reciprocal_events"],
        "residual_elements": dynamic["residual_correction_elements"],
        "dram_requests": t1["dram_requests"],
        "a_load_dram_requests": t1["a_load_dram_requests"],
        "index_hbm_cycles": t1["index_update_dram_cycles"],
        "y_accuracy_exact": t1["y_accuracy"]["exact"],
        "dcim_accuracy_valid": t1["accuracy"]["valid"],
        "dcim_accuracy_pass": t1["accuracy"]["pass"],
        "e2e_accuracy_valid": t1["e2e_accuracy"]["valid"],
        "e2e_accuracy_pass": t1["e2e_accuracy"]["pass"],
        "per_layer_rows": t1.get("per_layer_propagating_rows", []),
        "per_layer_macs": t1.get("per_layer_mac_ops", []),
        "dcim_rows": t1["cim_rows"],
        "dcim_macs": t1["dcim_scalar_macs_exact_once"],
        "scheduler_tasks": t1["phase1"]["nnz"] + t1["phase2"]["nnz"] +
                           t1["phase3"]["nnz"],
        "cache_hits": t1["cache_hits"],
        "cache_misses": t1["cache_misses"],
        "staging_rows": stats["hagg_staging"]["max_occupancy_rows"],
        "energy": t1["official_fp32_22nm_energy"],
    }


def run_once(binary: Path, output: Path, fixture: Path, name: str) -> dict:
    run_dir = output / name
    run_dir.mkdir(parents=True)
    (run_dir / "results_neurachip").mkdir()
    programs, deltas, features, weights_l1, weights_l2 = generate(fixture)
    repo = Path(__file__).resolve().parents[1]
    argv = [
        str(binary), "--profile", "despoina-h6-dcim-final",
        "-n", name, "-r", str(programs), "-x", "synthetic4",
        "--delta-dir", str(deltas), "--f-init", str(features),
        "--timesteps", "2", "--propagation-tau", "0.0",
        "--dram-config",
        str(repo / "third_party/DRAMsim3-1.0.0/configs/HBM2_8Gb_x128.ini"),
        "--w-init-l1", str(weights_l1), "--w-init-l2", str(weights_l2),
    ]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "DESPOINA_DCIM_PARAMS_JSON":
            str(repo / "configs/runtime/despoina_official_dcim_params.json"),
        "DESPOINA_DCIM_PRIMITIVE_JSON":
            str(repo / "configs/runtime/neurosim_official_dcim_primitive.json"),
    })
    result = subprocess.run(
        argv, cwd=run_dir, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    (run_dir / "stdout.log").write_text(result.stdout)
    (run_dir / "argv.json").write_text(json.dumps(argv, indent=2) + "\n")
    if result.returncode or result.stdout.count("Program End") != 1:
        raise SystemExit(f"simulator failed ({name}, rc={result.returncode})")
    result_dir = run_dir / "results_neurachip" / name
    stats = json.loads((result_dir / "pe_array_stats.json").read_text())
    if not (result_dir / "simulation_results.json").is_file():
        raise SystemExit("missing simulation_results.json")
    if not (result_dir / "dramsim3_pe_output/dramsim3.json").is_file():
        raise SystemExit("missing DRAMsim3 PE statistics")
    summary = selected(stats)
    (run_dir / "selected_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def run_fault_negative(binary: Path, output: Path) -> None:
    run_dir = output / "runtime_dcim_corruption"
    run_dir.mkdir(parents=True)
    (run_dir / "results_neurachip").mkdir()
    programs, deltas, features, weights_l1, weights_l2 = generate(
        output / "fault_fixture"
    )
    repo = Path(__file__).resolve().parents[1]
    argv = [
        str(binary), "--profile", "despoina-h6-dcim-final",
        "-n", "runtime_dcim_corruption", "-r", str(programs),
        "-x", "synthetic4", "--delta-dir", str(deltas),
        "--f-init", str(features), "--timesteps", "2",
        "--propagation-tau", "0.0", "--dram-config",
        str(repo / "third_party/DRAMsim3-1.0.0/configs/HBM2_8Gb_x128.ini"),
        "--w-init-l1", str(weights_l1), "--w-init-l2", str(weights_l2),
    ]
    env = os.environ.copy()
    env.update({
        "OMP_NUM_THREADS": "1",
        "DESPOINA_DCIM_PARAMS_JSON":
            str(repo / "configs/runtime/despoina_official_dcim_params.json"),
        "DESPOINA_DCIM_PRIMITIVE_JSON":
            str(repo / "configs/runtime/neurosim_official_dcim_primitive.json"),
        "DESPOINA_INTERNAL_TEST_CORRUPT_DCIM_OUTPUT": "1",
    })
    result = subprocess.run(
        argv, cwd=run_dir, env=env, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False
    )
    (run_dir / "stdout.log").write_text(result.stdout)
    (run_dir / "argv.json").write_text(json.dumps(argv, indent=2) + "\n")
    if result.returncode == 0 or "Program End" in result.stdout:
        raise SystemExit("corrupted runtime DCIM output did not fail closed")
    if "official DCIM accuracy comparison failed" not in result.stdout:
        raise SystemExit(
            "negative run failed for the wrong reason; accuracy rejection missing"
        )
    evidence = {
        "fault": "post-DCIM causal output element corruption",
        "interface": "test-only environment hook; no CLI option",
        "returncode": result.returncode,
        "program_end": False,
        "accuracy_rejection": True,
    }
    (output / "negative_summary.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True) + "\n"
    )
    print("RUNTIME_DCIM_CORRUPTION_FAIL_CLOSED_PASS",
          json.dumps(evidence, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--negative-only", action="store_true")
    args = parser.parse_args()
    binary = args.binary.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.negative_only:
        run_fault_negative(binary, output)
        return
    first = run_once(binary, output, output / "fixture1", "synthetic_repeat_1")
    second = run_once(binary, output, output / "fixture2", "synthetic_repeat_2")
    if first != second:
        raise SystemExit(f"non-deterministic structural/cycle summary: {first} != {second}")
    if first["timesteps"] != 2 or first["modes"] != ["full", "incremental"]:
        raise SystemExit(f"unexpected production-driver modes: {first}")
    if first["degree_rows"] != 1 or first["degree_writes"] != 1:
        raise SystemExit(f"dynamic row-mean path not exercised: {first}")
    if first["cycles"][1] <= 0 or first["dram_requests"] <= 0:
        raise SystemExit(f"cycle/DRAM path not exercised: {first}")
    if not (first["y_accuracy_exact"] and first["dcim_accuracy_valid"] and
            first["dcim_accuracy_pass"] and first["e2e_accuracy_valid"] and
            first["e2e_accuracy_pass"]):
        raise SystemExit(f"functional/accuracy instrumentation missing: {first}")
    if len(first["per_layer_rows"]) < 2 or first["dcim_rows"] <= 0 or first["dcim_macs"] <= 0:
        raise SystemExit(f"two-layer DCIM dependency path not exercised: {first}")
    if first["scheduler_tasks"] <= 0 or first["staging_rows"] <= 0:
        raise SystemExit(f"scheduler/staging path not exercised: {first}")
    if first["cache_hits"] + first["cache_misses"] <= 0:
        raise SystemExit(f"cache path not exercised: {first}")
    (output / "deterministic_summary.json").write_text(
        json.dumps(first, indent=2, sort_keys=True) + "\n"
    )
    print("REAL_DESPOINA_SYNTHETIC_SMOKE_PASS", json.dumps(first, sort_keys=True))


if __name__ == "__main__":
    main()
