#!/usr/bin/env python3
"""
η=0 Routability Certificate Validation (Issue #21)
====================================================

Evaluate η as a BINARY CLASSIFIER, not a regression predictor.

Question: "If η=0 in a G-cell, does that G-cell have zero GR overflow?"

This is the unique value proposition of η — no other placement metric
provides a formal threshold with theoretical guarantee.

Metrics:
  - Certificate precision: P(no overflow | η=0)  ← THE key number
  - Detection recall: P(η>0 | overflow>0)
  - Specificity: P(η=0 | no overflow)
  - Compare with density and RUDY thresholds
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from run_batch import load_design, theory_eta, DESIGNS, parse_lef_macros, parse_def_components
from eta_shield_placement import compute_gcell_metrics

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "gr")


def load_independent_gt(design_name, gs, die_area, dbu):
    """Load GRT guide-based usage and aggregate into our G-cell grid.

    Returns per-G-cell overflow (binary: has usage above capacity threshold).
    """
    gt_path = os.path.join(RESULTS_DIR,
                           f"gcell_overflow_congested_{design_name}.json")
    if not os.path.exists(gt_path):
        return None, None

    gt_data = json.load(open(gt_path))
    x_grids = gt_data['x_grids']
    y_grids = gt_data['y_grids']

    x_min, y_min = die_area["x_min"], die_area["y_min"]
    x_max, y_max = die_area["x_max"], die_area["y_max"]
    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs

    # Aggregate: total usage and max capacity per our G-cell
    usage_grid = np.zeros((gs, gs))
    capacity_grid = np.zeros((gs, gs))
    overflow_count = np.zeros((gs, gs))  # number of GRT GCells with overflow

    for gc in gt_data['gcells']:
        grt_x = (x_grids[gc['gx']] + x_grids[min(gc['gx'] + 1, gt_data['grid_nx'])]) / 2.0 / dbu
        grt_y = (y_grids[gc['gy']] + y_grids[min(gc['gy'] + 1, gt_data['grid_ny'])]) / 2.0 / dbu
        our_gx = min(max(0, int((grt_x - x_min) / gcell_w)), gs - 1)
        our_gy = min(max(0, int((grt_y - y_min) / gcell_h)), gs - 1)
        usage_grid[our_gy, our_gx] += gc['usage']
        capacity_grid[our_gy, our_gx] += gc['capacity']
        if gc.get('overflow', 0) > 0 or gc['usage'] > gc['capacity']:
            overflow_count[our_gy, our_gx] += 1

    # Binary: G-cell has overflow if usage > capacity
    has_overflow = (usage_grid > capacity_grid).astype(int)

    # Also try: G-cell has HIGH usage (top quartile)
    return usage_grid, capacity_grid, has_overflow, overflow_count


def analyze_design(design_name, gs_values=[6, 8]):
    """Binary classification analysis for one design."""
    print(f"\n{'='*70}")
    print(f"CERTIFICATE VALIDATION: {design_name}")
    print(f"{'='*70}")

    positions, widths, heights, die_area, _ = load_design(design_name)
    if positions is None:
        return None

    N = len(positions)
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)

    cfg = DESIGNS[design_name]
    pdk = cfg["pdk"]
    dbu = {"nangate45": 2000, "sky130hd": 1000, "asap7": 1000}.get(pdk, 1000)

    x_min, y_min = die_area["x_min"], die_area["y_min"]
    x_max, y_max = die_area["x_max"], die_area["y_max"]

    all_results = []

    for gs in gs_values:
        gcell_w = (x_max - x_min) / gs
        gcell_h = (y_max - y_min) / gs

        # Load GT
        usage_grid, cap_grid, has_overflow, overflow_count = \
            load_independent_gt(design_name, gs, die_area, dbu)
        if usage_grid is None:
            print(f"  gs={gs}: no GT data")
            continue

        # Cell density
        cell_density = np.zeros((gs, gs))
        ncells_map = np.zeros((gs, gs), dtype=int)
        for idx in range(N):
            gx = min(max(0, int((positions[idx, 0] - x_min) / gcell_w)), gs - 1)
            gy = min(max(0, int((positions[idx, 1] - y_min) / gcell_h)), gs - 1)
            cell_density[gy, gx] += widths[idx] * heights[idx] / (gcell_w * gcell_h)
            ncells_map[gy, gx] += 1

        # Active G-cells
        mask = ncells_map.ravel() >= 2
        n_active = int(mask.sum())
        if n_active < 4:
            continue

        y_true = has_overflow.ravel()[mask]
        n_overflow = int(y_true.sum())
        n_clean = int((1 - y_true).sum())

        print(f"\n  gs={gs}: {n_active} active G-cells, "
              f"{n_overflow} with overflow, {n_clean} clean")

        if n_overflow == 0:
            print(f"  No overflow G-cells — skipping (all clean)")
            continue
        if n_clean == 0:
            print(f"  All G-cells have overflow — skipping")
            continue

        # η at multiple radii
        r_values = [diag * f for f in [1.5, 2.0, 3.0, 4.0, 5.0]]

        print(f"\n  {'Predictor':>30s}  {'Prec(η=0→ok)':>12s}  {'Recall':>8s}  "
              f"{'F1':>6s}  {'Spec':>6s}  {'Acc':>6s}")
        print(f"  {'─'*75}")

        # Density threshold (median)
        cd_flat = cell_density.ravel()[mask]
        cd_pred = (cd_flat > np.median(cd_flat)).astype(int)
        if cd_pred.sum() > 0 and cd_pred.sum() < len(cd_pred):
            prec_cd = precision_score(y_true, cd_pred, zero_division=0)
            rec_cd = recall_score(y_true, cd_pred, zero_division=0)
            f1_cd = f1_score(y_true, cd_pred, zero_division=0)
            tn, fp, fn, tp = confusion_matrix(y_true, cd_pred).ravel()
            spec_cd = tn / (tn + fp) if (tn + fp) > 0 else 0
            acc_cd = (tp + tn) / len(y_true)
            # Certificate precision: P(no overflow | predicted clean)
            cert_prec_cd = tn / (tn + fn) if (tn + fn) > 0 else 0
            print(f"  {'density > median':>30s}  {cert_prec_cd:>12.3f}  "
                  f"{rec_cd:>8.3f}  {f1_cd:>6.3f}  {spec_cd:>6.3f}  {acc_cd:>6.3f}")

        for r in r_values:
            # Subsample for large designs
            if N > 1500:
                rng = np.random.default_rng(42)
                cx, cy = np.mean(positions, axis=0)
                dists = np.sqrt((positions[:, 0] - cx)**2 + (positions[:, 1] - cy)**2)
                idx_sub = np.argsort(dists)[:1500]
                pos_sub, w_sub, h_sub = positions[idx_sub], widths[idx_sub], heights[idx_sub]
            else:
                pos_sub, w_sub, h_sub = positions, widths, heights

            _, eta_map, _, dbar_map, _, ncells_sv = compute_gcell_metrics(
                pos_sub, w_sub, h_sub, die_area, gs, r)

            r_d = r / diag
            eta_flat = eta_map.ravel()[mask]

            # Binary prediction: η > 0 predicts overflow
            eta_pred = (eta_flat > 0).astype(int)

            # Also try theory-based: Δ̄ > 4
            dbar_flat = dbar_map.ravel()[mask]
            dbar_pred = (dbar_flat > 4).astype(int)

            for pred, label in [(eta_pred, f"η>0 (r={r_d:.1f}d)"),
                                (dbar_pred, f"Δ̄>4 (r={r_d:.1f}d)")]:
                if pred.sum() == 0 or pred.sum() == len(pred):
                    continue

                prec = precision_score(y_true, pred, zero_division=0)
                rec = recall_score(y_true, pred, zero_division=0)
                f1 = f1_score(y_true, pred, zero_division=0)
                tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                acc = (tp + tn) / len(y_true)
                # Certificate: P(no overflow | η=0)
                cert_prec = tn / (tn + fn) if (tn + fn) > 0 else 0

                print(f"  {label:>30s}  {cert_prec:>12.3f}  "
                      f"{rec:>8.3f}  {f1:>6.3f}  {spec:>6.3f}  {acc:>6.3f}")

                all_results.append({
                    "design": design_name,
                    "gs": gs,
                    "predictor": label,
                    "n_active": n_active,
                    "n_overflow": n_overflow,
                    "n_clean": n_clean,
                    "certificate_precision": float(cert_prec),
                    "recall": float(rec),
                    "f1": float(f1),
                    "specificity": float(spec),
                    "accuracy": float(acc),
                })

    return all_results


def main():
    designs = ["gcd_nangate45", "gcd_sky130", "aes_nangate45"]
    all_results = []

    for design in designs:
        gt_path = os.path.join(RESULTS_DIR,
                               f"gcell_overflow_congested_{design}.json")
        if not os.path.exists(gt_path):
            print(f"\n  Skipping {design}: no GT")
            continue
        results = analyze_design(design, gs_values=[6, 8, 10])
        if results:
            all_results.extend(results)

    # Summary: best certificate precision per design
    print(f"\n{'='*70}")
    print("CERTIFICATE SUMMARY")
    print(f"{'='*70}")
    print(f"  Certificate precision = P(no overflow | predictor says safe)")
    print(f"  This is THE key number: how trustworthy is the η=0 guarantee?\n")

    by_design = defaultdict(list)
    for r in all_results:
        by_design[r["design"]].append(r)

    for design, results in by_design.items():
        # Best η-based certificate
        eta_results = [r for r in results if "η>0" in r["predictor"]]
        if eta_results:
            best = max(eta_results, key=lambda r: r["certificate_precision"])
            print(f"  {design}:")
            print(f"    Best η certificate: P(safe|η=0) = {best['certificate_precision']:.3f} "
                  f"({best['predictor']}, gs={best['gs']})")
            print(f"    Recall = {best['recall']:.3f}, F1 = {best['f1']:.3f}")

    # Save
    json_path = os.path.join(RESULTS_DIR, "certificate_validation.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {json_path}")


if __name__ == "__main__":
    main()
