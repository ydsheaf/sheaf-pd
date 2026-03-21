#!/usr/bin/env python3
"""
Correlate η with independent GRT guide-based usage for AES nangate45.

Pipeline:
1. Load gcell_overflow_congested_aes_nangate45.json (GRT ground truth)
2. Aggregate GRT GCells into coarser grid (gs=10)
3. Load AES placement, compute η at multiple radii
4. Linear regression + 2-fold CV: density alone vs η+density
5. Save results to independent_gt_aes.json
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from scipy.linalg import svd
from scipy.spatial import KDTree
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

sys.path.insert(0, os.path.dirname(__file__))
from run_batch import (
    parse_lef_macros, parse_def_components, PlacedCell,
    build_overlap_coboundary, compute_eta, theory_eta,
    DESIGNS, load_design,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "gr")
GT_JSON = os.path.join(RESULTS_DIR, "gcell_overflow_congested_aes_nangate45.json")


def load_grt_ground_truth(json_path):
    """Load GRT gcell overflow JSON and return structured data."""
    with open(json_path) as f:
        data = json.load(f)
    return data


def aggregate_grt_to_coarse_grid(grt_data, gs):
    """Aggregate fine GRT GCells into a coarser gs x gs grid.

    Maps GRT gcell coordinates (in DBU) to a uniform gs x gs grid
    over the die area. Returns usage_map and capacity_map.
    """
    x_grids = grt_data["x_grids"]
    y_grids = grt_data["y_grids"]
    die_xmin = x_grids[0]
    die_xmax = x_grids[-1]
    die_ymin = y_grids[0]
    die_ymax = y_grids[-1]

    coarse_w = (die_xmax - die_xmin) / gs
    coarse_h = (die_ymax - die_ymin) / gs

    usage_map = np.zeros((gs, gs))
    capacity_map = np.zeros((gs, gs))

    for gcell in grt_data["gcells"]:
        gx_fine = gcell["gx"]
        gy_fine = gcell["gy"]

        # Get center of this fine gcell in DBU
        if gx_fine + 1 < len(x_grids):
            cx = (x_grids[gx_fine] + x_grids[gx_fine + 1]) / 2.0
        else:
            cx = x_grids[gx_fine]
        if gy_fine + 1 < len(y_grids):
            cy = (y_grids[gy_fine] + y_grids[gy_fine + 1]) / 2.0
        else:
            cy = y_grids[gy_fine]

        # Map to coarse grid
        cgx = min(int((cx - die_xmin) / coarse_w), gs - 1)
        cgy = min(int((cy - die_ymin) / coarse_h), gs - 1)
        cgx = max(0, cgx)
        cgy = max(0, cgy)

        usage_map[cgy, cgx] += gcell["usage"]
        capacity_map[cgy, cgx] += gcell["capacity"]

    return usage_map, capacity_map


def compute_eta_map_subsampled(positions, widths, heights, die_area, gs,
                                r_interact, max_svd_cells=3000):
    """Compute per-G-cell η with subsampling for SVD.

    For cell density, use ALL cells. For SVD-based η, subsample
    to max_svd_cells (center-biased).
    """
    N = len(positions)

    if die_area is not None:
        x_min, y_min = die_area["x_min"], die_area["y_min"]
        x_max, y_max = die_area["x_max"], die_area["y_max"]
    else:
        margin = 1.0
        x_min, y_min = positions.min(axis=0) - margin
        x_max, y_max = positions.max(axis=0) + margin

    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs
    gcell_area = gcell_w * gcell_h

    # --- Cell density map: use ALL cells ---
    density_map = np.zeros((gs, gs))
    ncells_all_map = np.zeros((gs, gs), dtype=int)
    for idx in range(N):
        gx = min(int((positions[idx, 0] - x_min) / gcell_w), gs - 1)
        gy = min(int((positions[idx, 1] - y_min) / gcell_h), gs - 1)
        gx, gy = max(0, gx), max(0, gy)
        density_map[gy, gx] += widths[idx] * heights[idx] / gcell_area
        ncells_all_map[gy, gx] += 1

    # --- Subsample for SVD ---
    if N > max_svd_cells:
        rng = np.random.default_rng(42)
        # Stratified subsampling: sample proportionally from each G-cell
        # to ensure coverage across the die
        gcell_cells_all = defaultdict(list)
        for idx in range(N):
            gx = min(int((positions[idx, 0] - x_min) / gcell_w), gs - 1)
            gy = min(int((positions[idx, 1] - y_min) / gcell_h), gs - 1)
            gx, gy = max(0, gx), max(0, gy)
            gcell_cells_all[(gx, gy)].append(idx)

        # Sample proportionally, minimum 2 per occupied G-cell if possible
        n_occupied = sum(1 for v in gcell_cells_all.values() if len(v) >= 2)
        per_cell_min = min(2, max_svd_cells // max(n_occupied, 1))
        remaining = max_svd_cells
        selected = []
        for key, idxs in gcell_cells_all.items():
            if len(idxs) < 2:
                continue
            n_take = max(per_cell_min, int(len(idxs) / N * max_svd_cells))
            n_take = min(n_take, len(idxs), remaining)
            chosen = rng.choice(idxs, size=n_take, replace=False)
            selected.extend(chosen)
            remaining -= n_take
            if remaining <= 0:
                break

        idx_sorted = np.array(selected[:max_svd_cells])
        pos_sub = positions[idx_sorted]
        w_sub = widths[idx_sorted]
        h_sub = heights[idx_sorted]
        print(f"    Subsampled {N} -> {len(idx_sorted)} cells (stratified) for SVD")
    else:
        pos_sub = positions
        w_sub = widths
        h_sub = heights

    N_sub = len(pos_sub)

    # Assign subsampled cells to G-cells
    gcell_cells = defaultdict(list)
    for idx in range(N_sub):
        gx = min(int((pos_sub[idx, 0] - x_min) / gcell_w), gs - 1)
        gy = min(int((pos_sub[idx, 1] - y_min) / gcell_h), gs - 1)
        gx, gy = max(0, gx), max(0, gy)
        gcell_cells[(gx, gy)].append(idx)

    # Build global edge set
    tree = KDTree(pos_sub)
    global_pairs = tree.query_pairs(r=r_interact)
    global_edges = [(min(i, j), max(i, j)) for i, j in global_pairs]

    eta_map = np.zeros((gs, gs))
    ncells_sub_map = np.zeros((gs, gs), dtype=int)

    for gx in range(gs):
        for gy in range(gs):
            cell_idxs = gcell_cells.get((gx, gy), [])
            nc = len(cell_idxs)
            ncells_sub_map[gy, gx] = nc
            if nc < 2:
                continue

            idx_set = set(cell_idxs)
            old_to_new = {old: new for new, old in enumerate(sorted(idx_set))}
            local_edges = [(old_to_new[i], old_to_new[j])
                           for i, j in global_edges
                           if i in idx_set and j in idx_set]

            nE = len(local_edges)
            if nE == 0:
                continue

            dbar_local = 2.0 * nE / nc

            # SVD for exact η if small enough, else theory
            if nc <= 400 and nE <= 3000:
                local_pos = pos_sub[sorted(idx_set)]
                delta = build_overlap_coboundary(local_pos, local_edges, n_v=2)
                sv = svd(delta, compute_uv=False)
                rank = int(np.sum(sv > 1e-10))
                eta_alpha = (nE - rank) / nE if nE > 0 else 0.0
            else:
                eta_alpha = theory_eta(dbar_local, 2)

            eta_map[gy, gx] = eta_alpha

    return eta_map, density_map, ncells_all_map, ncells_sub_map


def cv_r2(X, y, n_folds=2, seed=42):
    """Compute cross-validated R² with n_folds.

    Standardizes features before fitting. Uses multiple seeds and
    averages for stability when n_folds < N.
    """
    from sklearn.preprocessing import StandardScaler

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    if n_folds >= len(y):
        # LOO
        from sklearn.model_selection import LeaveOneOut
        loo = LeaveOneOut()
        preds = np.zeros_like(y)
        for train_idx, test_idx in loo.split(X_scaled):
            model = LinearRegression()
            model.fit(X_scaled[train_idx], y[train_idx])
            preds[test_idx] = model.predict(X_scaled[test_idx])
        ss_res = np.sum((y - preds)**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # For k-fold, average over multiple seeds for stability
    all_r2 = []
    for s in range(5):
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=seed + s)
        preds = np.zeros_like(y)
        counts = np.zeros_like(y)
        for train_idx, test_idx in kf.split(X_scaled):
            model = LinearRegression()
            model.fit(X_scaled[train_idx], y[train_idx])
            preds[test_idx] += model.predict(X_scaled[test_idx])
            counts[test_idx] += 1
        preds /= np.maximum(counts, 1)
        mask = counts > 0
        ss_res = np.sum((y[mask] - preds[mask])**2)
        ss_tot = np.sum((y[mask] - np.mean(y[mask]))**2)
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
        all_r2.append(r2)
    return np.mean(all_r2)


def main():
    print("=" * 70)
    print("Independent GT Correlation: AES nangate45")
    print("=" * 70)

    # Step 1: Load GRT ground truth
    print(f"\nLoading GRT ground truth: {GT_JSON}")
    grt_data = load_grt_ground_truth(GT_JSON)
    print(f"  GRT grid: {grt_data['grid_nx']} x {grt_data['grid_ny']}")
    print(f"  GCells with data: {len(grt_data['gcells'])}")
    print(f"  Summary: {json.dumps(grt_data['summary'], indent=2)}")

    # Step 2: Load AES placement
    print("\nLoading AES placement...")
    positions, widths, heights, die_area, _ = load_design("aes_nangate45")
    if positions is None:
        print("ERROR: Could not load AES design")
        return

    N = len(positions)
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)
    print(f"  N={N}, med_w={med_w:.3f}, med_h={med_h:.3f}, diag={diag:.3f}")

    # Step 3: Compute η at multiple radii and grid sizes, correlate with GT usage
    gs = 10  # primary grid size as specified
    r_factors = [1.0, 1.5, 2.0, 3.0, 4.0]
    r_values = [diag * f for f in r_factors]

    usage_map, capacity_map = aggregate_grt_to_coarse_grid(grt_data, gs)
    print(f"\n  Aggregated GRT to {gs}x{gs} grid")
    print(f"  Total usage: {usage_map.sum():.0f}")
    print(f"  Max usage per coarse cell: {usage_map.max():.0f}")
    print(f"  Non-zero coarse cells: {np.sum(usage_map > 0)}")

    results_per_r = []
    best_cv_r2_eta_density = -999
    best_result = None

    print(f"\n  {'r':>6s}  {'r/diag':>6s}  {'LOO-R2(dens)':>12s}  "
          f"{'LOO-R2(e+d)':>12s}  {'LOO-R2(eta)':>12s}  "
          f"{'R2(eta)':>8s}  {'R2(dens)':>8s}")
    print(f"  {'─'*80}")

    for r_interact in r_values:
        eta_map, density_map, ncells_all, ncells_sub = \
            compute_eta_map_subsampled(
                positions, widths, heights, die_area, gs,
                r_interact, max_svd_cells=3000
            )

        # Use ncells from ALL cells for coverage check
        mask = (ncells_all.ravel() >= 2) & (usage_map.ravel() > 0)

        # Fill in eta for cells that have enough total cells but not enough subsampled
        eta_filled = eta_map.copy()
        for gx in range(gs):
            for gy in range(gs):
                if ncells_all[gy, gx] >= 2 and ncells_sub[gy, gx] < 2:
                    nc = ncells_all[gy, gx]
                    gcell_w_local = (die_area["x_max"] - die_area["x_min"]) / gs
                    gcell_h_local = (die_area["y_max"] - die_area["y_min"]) / gs
                    gcell_area_local = gcell_w_local * gcell_h_local
                    cell_density_per_area = nc / gcell_area_local
                    dbar_est = cell_density_per_area * np.pi * r_interact**2
                    dbar_est = min(dbar_est, nc - 1)
                    eta_filled[gy, gx] = theory_eta(dbar_est, 2) if dbar_est > 0 else 0.0

        n_valid = mask.sum()
        if n_valid < 6:
            print(f"  {r_interact:6.3f}  {r_interact/diag:6.1f}  "
                  f"  (only {n_valid} valid cells, skipping)")
            continue

        eta_flat = eta_filled.ravel()[mask]
        density_flat = density_map.ravel()[mask]
        usage_flat = usage_map.ravel()[mask]

        # Simple R² (Pearson)
        r2_eta = np.corrcoef(eta_flat, usage_flat)[0, 1]**2 if np.std(eta_flat) > 1e-10 else 0
        r2_dens = np.corrcoef(density_flat, usage_flat)[0, 1]**2 if np.std(density_flat) > 1e-10 else 0

        # LOO CV-R²
        X_dens = density_flat.reshape(-1, 1)
        loo_r2_dens = cv_r2(X_dens, usage_flat, n_folds=n_valid)

        X_eta_dens = np.column_stack([eta_flat, density_flat])
        loo_r2_ed = cv_r2(X_eta_dens, usage_flat, n_folds=n_valid)

        X_eta = eta_flat.reshape(-1, 1)
        loo_r2_eta_alone = cv_r2(X_eta, usage_flat, n_folds=n_valid) if np.std(eta_flat) > 1e-10 else 0.0

        # 2-fold CV-R²
        cv2_r2_dens = cv_r2(X_dens, usage_flat, n_folds=2)
        cv2_r2_ed = cv_r2(X_eta_dens, usage_flat, n_folds=2)

        print(f"  {r_interact:6.3f}  {r_interact/diag:6.1f}  "
              f"{loo_r2_dens:12.4f}  {loo_r2_ed:12.4f}  {loo_r2_eta_alone:12.4f}  "
              f"{r2_eta:8.4f}  {r2_dens:8.4f}  n={n_valid}")

        result_r = {
            "r": float(r_interact),
            "r_over_diag": float(r_interact / diag),
            "n_valid_cells": int(n_valid),
            "R2_eta": float(r2_eta),
            "R2_density": float(r2_dens),
            "CV2_R2_density_alone": float(cv2_r2_dens),
            "CV2_R2_eta_plus_density": float(cv2_r2_ed),
            "LOO_R2_density_alone": float(loo_r2_dens),
            "LOO_R2_eta_plus_density": float(loo_r2_ed),
            "LOO_R2_eta_alone": float(loo_r2_eta_alone),
            "eta_improvement_LOO": float(loo_r2_ed - loo_r2_dens),
        }
        results_per_r.append(result_r)

        if loo_r2_ed > best_cv_r2_eta_density:
            best_cv_r2_eta_density = loo_r2_ed
            best_result = result_r

    if best_result is None:
        print("\nERROR: No valid correlations computed")
        return

    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY: Independent GT Correlation for AES nangate45")
    print(f"{'='*70}")
    print(f"  Best radius: r={best_result['r']:.3f} "
          f"(r/diag={best_result['r_over_diag']:.1f})")
    print(f"  n_valid G-cells:       {best_result['n_valid_cells']}")
    print(f"  LOO-R2(density alone)  = {best_result['LOO_R2_density_alone']:.4f}")
    print(f"  LOO-R2(eta+density)    = {best_result['LOO_R2_eta_plus_density']:.4f}")
    print(f"  LOO-R2(eta alone)      = {best_result['LOO_R2_eta_alone']:.4f}")
    print(f"  eta improvement (LOO)  = {best_result['eta_improvement_LOO']:.4f}")
    print(f"  Pearson R2(eta alone)  = {best_result['R2_eta']:.4f}")
    print(f"  Pearson R2(density)    = {best_result['R2_density']:.4f}")

    eta_imp = best_result['eta_improvement_LOO']
    if eta_imp > 0:
        print(f"\n  >>> eta ADDS VALUE over density alone "
              f"(+{eta_imp:.4f} LOO-R2)")
    else:
        print(f"\n  >>> density alone sufficient "
              f"(eta adds {eta_imp:.4f})")

    # Compare with demand-proxy results (from validate_congested.py)
    print(f"\n  Comparison with demand-proxy results:")
    print(f"    Demand proxy: R2(eta)=0.40, R2(RUDY)=0.11")
    print(f"    Independent GT: R2(eta)={best_result['R2_eta']:.4f}, "
          f"R2(density)={best_result['R2_density']:.4f}")

    # Save
    output = {
        "design": "aes_nangate45",
        "N": N,
        "gs": gs,
        "max_svd_cells": 3000,
        "ground_truth": "GRT guide-based usage (gcell_overflow_congested_aes_nangate45.json)",
        "grt_summary": grt_data["summary"],
        "results_per_radius": results_per_r,
        "best": best_result,
        "comparison_demand_proxy": {
            "R2_eta_demand_proxy": 0.40,
            "R2_rudy_demand_proxy": 0.11,
        },
    }

    out_path = os.path.join(RESULTS_DIR, "independent_gt_aes.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {out_path}")


if __name__ == "__main__":
    main()
