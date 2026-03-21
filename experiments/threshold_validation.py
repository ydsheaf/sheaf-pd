#!/usr/bin/env python3
"""
E5: Validate eta behavior at the phase transition threshold Dbar ~ 4
=====================================================================

For each design (gcd_nangate45, aes_nangate45):
1. Sweep interaction radius r to find r where mean Dbar ~ 4 (PS1 transition)
2. At that r, compute per-G-cell:
   - eta_alpha (SVD)
   - eta_theory = max(0, 1 - 4/Dbar_alpha)
   - Binary threshold classification (Dbar > 4 vs <= 4)
3. Parse congested GR demand from log files (NB lines)
4. Validate:
   - Dbar < 4 (eta=0 predicted) => fraction with low demand
   - Dbar > 4 (eta>0 predicted) => fraction with high demand
5. Precision/recall of "eta>0 predicts high demand" using median demand

Usage:
  python experiments/threshold_validation.py
"""

import json
import os
import re
import sys
import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree

sys.path.insert(0, os.path.dirname(__file__))
from run_batch import (
    parse_lef_macros, parse_def_components, PlacedCell,
    build_overlap_coboundary, compute_eta, theory_eta,
    DESIGNS, load_design,
)
from eta_shield_placement import compute_gcell_metrics
from validate_congested import compute_per_gcell_demand

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "gr")
os.makedirs(RESULTS_DIR, exist_ok=True)

GS = 10  # G-cell grid size


def parse_net_bboxes_from_log(log_path):
    """Parse net bounding boxes from congested GR log (NB lines)."""
    net_bboxes = []
    die_area_um = None

    with open(log_path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("NB "):
                parts = line.split()
                if len(parts) >= 5:
                    net_bboxes.append({
                        "x_min": float(parts[1]),
                        "y_min": float(parts[2]),
                        "x_max": float(parts[3]),
                        "y_max": float(parts[4]),
                    })
            elif line.startswith("DIE_UM "):
                parts = line.split()
                if len(parts) >= 5:
                    die_area_um = {
                        "x_min": float(parts[1]),
                        "y_min": float(parts[2]),
                        "x_max": float(parts[3]),
                        "y_max": float(parts[4]),
                    }

    return net_bboxes, die_area_um


def sweep_radius_for_dbar(positions, widths, heights, die_area, gs,
                          target_dbar=4.0, n_v=2):
    """Sweep interaction radius r to find r where mean Dbar ~ 4.

    Returns the best r, and the metrics at that r.
    """
    N = len(positions)
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)

    # Subsample large designs for SVD tractability
    if N > 1500:
        rng = np.random.default_rng(42)
        idx = rng.choice(N, 1500, replace=False)
        pos_sub = positions[idx]
        w_sub = widths[idx]
        h_sub = heights[idx]
    else:
        pos_sub = positions
        w_sub = widths
        h_sub = heights

    # Sweep multipliers of cell diagonal (wider range for dense designs)
    multipliers = np.arange(0.5, 12.1, 0.25)
    best_r = None
    best_diff = float("inf")
    best_result = None

    print(f"  Sweeping r to find mean Dbar ~ {target_dbar}:")
    print(f"  {'r':>7s}  {'r/diag':>7s}  {'mean_Dbar':>10s}  {'diff':>8s}  {'eta_max':>8s}")
    print(f"  {'─' * 50}")

    for mult in multipliers:
        r = diag * mult
        gains, eta_map, sigma_map, dbar_map, gcell_assign, ncells_map = \
            compute_gcell_metrics(pos_sub, w_sub, h_sub, die_area, gs, r)

        # Mean Dbar over active G-cells (those with >= 2 cells)
        active_mask = ncells_map >= 2
        if active_mask.sum() < 3:
            continue

        active_dbar = dbar_map[active_mask]
        mean_dbar = float(np.mean(active_dbar))
        diff = abs(mean_dbar - target_dbar)

        print(f"  {r:7.3f}  {mult:7.2f}  {mean_dbar:10.2f}  {diff:8.2f}  "
              f"{float(np.max(eta_map)):8.3f}")

        if diff < best_diff:
            best_diff = diff
            best_r = r
            best_result = {
                "r": r,
                "r_over_diag": mult,
                "mean_dbar": mean_dbar,
                "eta_map": eta_map.copy(),
                "dbar_map": dbar_map.copy(),
                "ncells_map": ncells_map.copy(),
                "gcell_assign": dict(gcell_assign),
                "gains": gains.copy(),
                "sigma_map": sigma_map.copy(),
            }

    print(f"\n  Best r = {best_r:.3f} (r/diag = {best_r/diag:.2f}), "
          f"mean Dbar = {best_result['mean_dbar']:.2f}")

    return best_result


def run_threshold_validation(design_name, gs=GS):
    """Run threshold validation for one design."""
    print(f"\n{'=' * 70}")
    print(f"THRESHOLD VALIDATION: {design_name}")
    print(f"{'=' * 70}")

    # 1. Load design
    positions, widths, heights, die_area, label = load_design(design_name)
    if positions is None:
        print(f"  ERROR: Could not load {design_name}")
        return None

    N = len(positions)
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)
    print(f"  Cells: {N}, median size: {med_w:.3f} x {med_h:.3f}, diag: {diag:.3f}")

    # 2. Sweep r to find Dbar ~ 4
    result = sweep_radius_for_dbar(positions, widths, heights, die_area, gs)
    if result is None:
        print("  ERROR: Could not find suitable r")
        return None

    eta_map = result["eta_map"]
    dbar_map = result["dbar_map"]
    ncells_map = result["ncells_map"]
    r_opt = result["r"]

    # 3. Compute per-G-cell eta_theory = max(0, 1 - 4/Dbar_alpha)
    eta_theory_map = np.zeros_like(dbar_map)
    for gy in range(gs):
        for gx in range(gs):
            d = dbar_map[gy, gx]
            eta_theory_map[gy, gx] = max(0.0, 1.0 - 4.0 / d) if d > 0 else 0.0

    # 4. Load demand from congested GR log
    log_path = os.path.join(RESULTS_DIR, f"congested_{design_name}.log")
    if not os.path.exists(log_path):
        print(f"  WARNING: No GR log at {log_path}, skipping demand analysis")
        return None

    net_bboxes, die_area_gr = parse_net_bboxes_from_log(log_path)
    print(f"  Parsed {len(net_bboxes)} net bounding boxes from GR log")

    # Use GR die area if available, else placement die area
    die_for_demand = die_area_gr if die_area_gr else die_area
    if die_for_demand is None:
        print("  ERROR: No die area available")
        return None

    demand_map = compute_per_gcell_demand(net_bboxes, die_for_demand, gs)

    # 5. Classification analysis
    active_mask = ncells_map >= 2
    n_active = int(active_mask.sum())

    # Flatten active G-cells
    dbar_flat = dbar_map[active_mask]
    eta_svd_flat = eta_map[active_mask]
    eta_th_flat = eta_theory_map[active_mask]
    demand_flat = demand_map[active_mask]

    # Threshold classification
    below_threshold = dbar_flat <= 4.0   # eta should be 0
    above_threshold = dbar_flat > 4.0    # eta should be > 0

    n_below = int(below_threshold.sum())
    n_above = int(above_threshold.sum())

    # Median demand as high/low boundary
    median_demand = float(np.median(demand_flat))
    high_demand = demand_flat > median_demand
    low_demand = demand_flat <= median_demand

    print(f"\n  Active G-cells: {n_active}")
    print(f"  Below threshold (Dbar <= 4): {n_below}")
    print(f"  Above threshold (Dbar > 4):  {n_above}")
    print(f"  Median demand: {median_demand:.4f}")

    # 6. Validation: Dbar < 4 => low demand?
    if n_below > 0:
        frac_low_below = float(low_demand[below_threshold].sum()) / n_below
    else:
        frac_low_below = float("nan")

    # 7. Validation: Dbar > 4 => high demand?
    if n_above > 0:
        frac_high_above = float(high_demand[above_threshold].sum()) / n_above
    else:
        frac_high_above = float("nan")

    print(f"\n  Dbar <= 4 (eta=0 predicted): {frac_low_below:.1%} have low demand  "
          f"(validates eta=0 -> safe)")
    print(f"  Dbar > 4  (eta>0 predicted): {frac_high_above:.1%} have high demand "
          f"(validates eta>0 -> congested)")

    # 8. Precision / recall of binary classifier "eta>0 predicts high demand"
    # Predicted positive: eta_theory > 0  (equivalently, Dbar > 4)
    # Actual positive: high demand (above median)
    pred_positive = above_threshold
    actual_positive = high_demand

    tp = int((pred_positive & actual_positive).sum())
    fp = int((pred_positive & ~actual_positive).sum())
    fn = int((~pred_positive & actual_positive).sum())
    tn = int((~pred_positive & ~actual_positive).sum())

    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    f1 = 2 * precision * recall / max(1e-10, precision + recall)
    accuracy = (tp + tn) / max(1, n_active)

    print(f"\n  Binary classifier: 'eta > 0 predicts high demand'")
    print(f"  {'─' * 45}")
    print(f"  TP={tp:4d}  FP={fp:4d}")
    print(f"  FN={fn:4d}  TN={tn:4d}")
    print(f"  Precision: {precision:.3f}")
    print(f"  Recall:    {recall:.3f}")
    print(f"  F1:        {f1:.3f}")
    print(f"  Accuracy:  {accuracy:.3f}")

    # 9. Additional: correlation between eta_svd and eta_theory
    if np.std(eta_svd_flat) > 1e-10 and np.std(eta_th_flat) > 1e-10:
        corr_svd_th = float(np.corrcoef(eta_svd_flat, eta_th_flat)[0, 1])
    else:
        corr_svd_th = float("nan")

    if np.std(eta_svd_flat) > 1e-10 and np.std(demand_flat) > 1e-10:
        corr_eta_demand = float(np.corrcoef(eta_svd_flat, demand_flat)[0, 1])
    else:
        corr_eta_demand = float("nan")

    print(f"\n  Correlation(eta_SVD, eta_theory): {corr_svd_th:.4f}")
    print(f"  Correlation(eta_SVD, demand):     {corr_eta_demand:.4f}")

    # Build output
    output = {
        "design": design_name,
        "N": N,
        "gs": gs,
        "r_optimal": float(r_opt),
        "r_over_diag": float(r_opt / diag),
        "mean_dbar_at_r": float(result["mean_dbar"]),
        "n_active_gcells": n_active,
        "n_below_threshold": n_below,
        "n_above_threshold": n_above,
        "median_demand": median_demand,
        "frac_low_demand_below_threshold": float(frac_low_below) if not np.isnan(frac_low_below) else None,
        "frac_high_demand_above_threshold": float(frac_high_above) if not np.isnan(frac_high_above) else None,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "confusion_matrix": {"TP": tp, "FP": fp, "FN": fn, "TN": tn},
        "corr_eta_svd_vs_theory": float(corr_svd_th) if not np.isnan(corr_svd_th) else None,
        "corr_eta_svd_vs_demand": float(corr_eta_demand) if not np.isnan(corr_eta_demand) else None,
        "per_gcell": {
            "dbar": dbar_flat.tolist(),
            "eta_svd": eta_svd_flat.tolist(),
            "eta_theory": eta_th_flat.tolist(),
            "demand": demand_flat.tolist(),
        },
    }

    return output


def main():
    designs = ["gcd_nangate45", "aes_nangate45"]
    all_results = {}

    for design_name in designs:
        result = run_threshold_validation(design_name, gs=GS)
        if result is not None:
            all_results[design_name] = result

    # Summary table
    print(f"\n{'=' * 80}")
    print("THRESHOLD VALIDATION SUMMARY")
    print(f"{'=' * 80}")
    print(f"  {'Design':>20s}  {'r/diag':>7s}  {'mean_Dbar':>10s}  "
          f"{'Below->Low':>11s}  {'Above->High':>12s}  "
          f"{'Prec':>6s}  {'Rec':>6s}  {'F1':>6s}  {'Acc':>6s}")
    print(f"  {'─' * 95}")

    for design_name, r in all_results.items():
        below_low = r.get("frac_low_demand_below_threshold")
        above_high = r.get("frac_high_demand_above_threshold")
        below_str = f"{below_low:.1%}" if below_low is not None else "N/A"
        above_str = f"{above_high:.1%}" if above_high is not None else "N/A"
        print(f"  {design_name:>20s}  {r['r_over_diag']:7.2f}  "
              f"{r['mean_dbar_at_r']:10.2f}  "
              f"{below_str:>11s}  {above_str:>12s}  "
              f"{r['precision']:6.3f}  {r['recall']:6.3f}  "
              f"{r['f1']:6.3f}  {r['accuracy']:6.3f}")

    # Save results
    json_path = os.path.join(RESULTS_DIR, "threshold_validation.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {json_path}")


if __name__ == "__main__":
    main()
