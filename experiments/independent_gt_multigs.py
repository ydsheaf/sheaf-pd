#!/usr/bin/env python3
"""
Independent Ground-Truth Validation: eta vs GRT overflow at multiple grid sizes.

Loads the GCell overflow JSON from OpenROAD GRT (46x47 fine grid),
aggregates into coarser gs x gs grids, computes eta and cell_density
at each resolution, and runs 2-fold CV regression against aggregated
GRT usage as the ground-truth congestion signal.

Reports: gs, n_active_gcells, CV-R2(density), CV-R2(eta+density), Delta
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(__file__))
from run_batch import load_design, DESIGNS
from eta_shield_placement import compute_gcell_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "gr")
DBU_PER_MICRON = 2000  # nangate45


def load_grt_data():
    """Load the GCell overflow JSON from GRT."""
    json_path = os.path.join(RESULTS_DIR, "gcell_overflow_congested_gcd_nangate45.json")
    with open(json_path) as f:
        return json.load(f)


def aggregate_grt_usage(grt_data, gs, die_area):
    """Aggregate GRT GCell usage into a coarser gs x gs grid.

    The GRT grid has its own (gx, gy) coordinates with boundaries
    x_grids/y_grids in DBU. We convert each GRT GCell center to microns,
    map it into our gs x gs grid, and sum usage values.
    """
    x_grids = np.array(grt_data["x_grids"]) / DBU_PER_MICRON  # to microns
    y_grids = np.array(grt_data["y_grids"]) / DBU_PER_MICRON

    x_min = die_area["x_min"]
    y_min = die_area["y_min"]
    x_max = die_area["x_max"]
    y_max = die_area["y_max"]
    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs

    usage_map = np.zeros((gs, gs))

    for gcell in grt_data["gcells"]:
        gx_grt = gcell["gx"]
        gy_grt = gcell["gy"]

        # GRT GCell center in microns
        if gx_grt + 1 < len(x_grids):
            cx = (x_grids[gx_grt] + x_grids[gx_grt + 1]) / 2.0
        else:
            cx = x_grids[gx_grt]
        if gy_grt + 1 < len(y_grids):
            cy = (y_grids[gy_grt] + y_grids[gy_grt + 1]) / 2.0
        else:
            cy = y_grids[gy_grt]

        # Map to our coarse grid
        our_gx = min(int((cx - x_min) / gcell_w), gs - 1)
        our_gy = min(int((cy - y_min) / gcell_h), gs - 1)
        our_gx = max(0, our_gx)
        our_gy = max(0, our_gy)

        usage_map[our_gy, our_gx] += gcell["usage"]

    return usage_map


def compute_cell_density(positions, widths, heights, die_area, gs):
    """Compute cell area density per G-cell."""
    N = len(positions)
    x_min, y_min = die_area["x_min"], die_area["y_min"]
    x_max, y_max = die_area["x_max"], die_area["y_max"]
    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs
    gcell_area = gcell_w * gcell_h

    density_map = np.zeros((gs, gs))
    ncells_map = np.zeros((gs, gs), dtype=int)

    for idx in range(N):
        gx = min(int((positions[idx, 0] - x_min) / gcell_w), gs - 1)
        gy = min(int((positions[idx, 1] - y_min) / gcell_h), gs - 1)
        gx, gy = max(0, gx), max(0, gy)
        density_map[gy, gx] += widths[idx] * heights[idx] / gcell_area
        ncells_map[gy, gx] += 1

    return density_map, ncells_map


def crossval_r2(X, y, fold_mask):
    """2-fold cross-validated R-squared."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if np.std(X, axis=0).max() < 1e-10:
        return 0.0

    scores = []
    for train_mask, test_mask in [(fold_mask, ~fold_mask), (~fold_mask, fold_mask)]:
        if train_mask.sum() < 2 or test_mask.sum() < 2:
            return 0.0
        reg = LinearRegression().fit(X[train_mask], y[train_mask])
        y_pred = reg.predict(X[test_mask])
        scores.append(r2_score(y[test_mask], y_pred))
    return float(np.mean(scores))


def main():
    # Load GRT data
    grt_data = load_grt_data()
    print(f"GRT grid: {grt_data['grid_nx']}x{grt_data['grid_ny']}, "
          f"{len(grt_data['gcells'])} gcells with demand")

    # Load design
    positions, widths, heights, die_area, label = load_design("gcd_nangate45")
    if positions is None:
        print("ERROR: Could not load design")
        return

    N = len(positions)
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)
    print(f"\nCells: {N}, median {med_w:.3f} x {med_h:.3f} um, diag={diag:.4f}")
    print(f"Die area: ({die_area['x_min']:.1f}, {die_area['y_min']:.1f}) to "
          f"({die_area['x_max']:.1f}, {die_area['y_max']:.1f})")

    r_factors = [1.5, 2.0, 3.0]
    grid_sizes = [6, 8, 10]

    all_results = []

    print(f"\n{'='*80}")
    print(f"{'gs':>4s}  {'n_active':>8s}  {'r/diag':>6s}  "
          f"{'CV-R2(dens)':>12s}  {'CV-R2(eta+dens)':>16s}  {'Delta':>8s}")
    print(f"{'='*80}")

    for gs in grid_sizes:
        # Aggregate GRT usage into our grid
        usage_map = aggregate_grt_usage(grt_data, gs, die_area)

        # Compute cell density
        density_map, ncells_map = compute_cell_density(
            positions, widths, heights, die_area, gs
        )

        # Active G-cells: have cells AND have GRT usage
        active_mask = (ncells_map.ravel() >= 2) & (usage_map.ravel() > 0)
        n_active = int(active_mask.sum())

        y = usage_map.ravel()[active_mask]
        cell_dens = density_map.ravel()[active_mask]

        if n_active < 4 or np.std(y) < 1e-10:
            print(f"{gs:>4d}  {n_active:>8d}  -- not enough active G-cells --")
            continue

        # 2-fold split: odd vs even index
        fold_mask = np.array([i % 2 == 0 for i in range(n_active)])

        # Baseline: cell_density alone
        cv_r2_dens = crossval_r2(cell_dens, y, fold_mask)

        gs_result = {
            "gs": gs,
            "n_active_gcells": n_active,
            "cv_r2_density": float(cv_r2_dens),
            "per_radius": [],
        }

        for r_factor in r_factors:
            r = r_factor * diag

            # Compute per-G-cell eta
            gains, eta_map, sigma_map, dbar_map, gcell_assign, nc_map = \
                compute_gcell_metrics(positions, widths, heights, die_area, gs, r)

            eta = eta_map.ravel()[active_mask]

            # eta + density
            X_combined = np.column_stack([eta, cell_dens])
            cv_r2_combined = crossval_r2(X_combined, y, fold_mask)

            delta = cv_r2_combined - cv_r2_dens
            delta_pct = delta * 100

            print(f"{gs:>4d}  {n_active:>8d}  {r_factor:>6.1f}  "
                  f"{cv_r2_dens:>12.4f}  {cv_r2_combined:>16.4f}  {delta_pct:>+7.1f}%")

            gs_result["per_radius"].append({
                "r_factor": float(r_factor),
                "r": float(r),
                "cv_r2_eta_density": float(cv_r2_combined),
                "delta": float(delta),
                "delta_pct": float(delta_pct),
                "eta_max": float(np.max(eta_map)),
                "eta_mean_active": float(np.mean(eta)),
            })

        all_results.append(gs_result)
        print(f"{'─'*80}")

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"{'gs':>4s}  {'n_active':>8s}  {'CV-R2(dens)':>12s}  "
          f"{'CV-R2(eta+dens)':>16s}  {'Delta':>8s}  {'r/diag':>6s}")
    print(f"{'─'*60}")

    for res in all_results:
        # Pick best radius for each gs
        best = max(res["per_radius"], key=lambda x: x["cv_r2_eta_density"])
        print(f"{res['gs']:>4d}  {res['n_active_gcells']:>8d}  "
              f"{res['cv_r2_density']:>12.4f}  "
              f"{best['cv_r2_eta_density']:>16.4f}  "
              f"{best['delta_pct']:>+7.1f}%  "
              f"{best['r_factor']:>6.1f}")

    # Save
    output = {
        "design": "gcd_nangate45",
        "ground_truth": "GRT gcell usage (guide-based)",
        "source_file": "gcell_overflow_congested_gcd_nangate45.json",
        "grt_grid": f"{grt_data['grid_nx']}x{grt_data['grid_ny']}",
        "dbu_per_micron": DBU_PER_MICRON,
        "cell_diag_um": float(diag),
        "n_cells": N,
        "r_factors": r_factors,
        "grid_sizes": grid_sizes,
        "method": "2-fold CV (odd/even index), density alone vs eta+density",
        "results": all_results,
    }

    os.makedirs(RESULTS_DIR, exist_ok=True)
    out_path = os.path.join(RESULTS_DIR, "independent_gt_gcd_multigs.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
