#!/usr/bin/env python3
"""
η + RUDY Combined Predictor: Does η add information beyond RUDY?
================================================================

RUDY measures density (additive, per-net).
η measures topological constraint interaction (non-additive, per-edge).

If η captures something RUDY doesn't, then:
    R²(α·η + β·RUDY, demand) > R²(RUDY, demand)

The improvement quantifies the "topological information content" of η
beyond what density alone provides.

This is the key question for the ICCAD paper: η doesn't need to beat
RUDY alone — it needs to ADD information to RUDY.

Uses the congested gcd_nangate45 data from validate_congested.py.
"""

import json
import os
import sys
import numpy as np
from scipy.spatial import KDTree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(__file__))
from run_batch import (
    build_overlap_coboundary, theory_eta,
    DESIGNS, load_design,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "gr")


def load_congested_data(design_name):
    """Load pre-computed congested GR data."""
    json_path = os.path.join(RESULTS_DIR, f"congested_{design_name}.json")
    if not os.path.exists(json_path):
        print(f"  No congested data for {design_name}. Run validate_congested.py first.")
        return None
    with open(json_path) as f:
        return json.load(f)


def compute_net_rudy(net_bboxes, die_area, gs):
    """Compute net-based RUDY (Rectangular Uniform wire DensitY) per G-cell.

    For each net bounding box:
      1. HPWL = bbox_width + bbox_height
      2. demand = HPWL / bbox_area  (wire density estimate)
      3. Distribute to each G-cell weighted by overlap fraction

    This is the standard RUDY formula used in placement tools.
    """
    x_min, y_min = die_area["x_min"], die_area["y_min"]
    x_max, y_max = die_area["x_max"], die_area["y_max"]
    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs

    rudy_map = np.zeros((gs, gs))

    for net in net_bboxes:
        bbox_w = net["x_max"] - net["x_min"]
        bbox_h = net["y_max"] - net["y_min"]
        hpwl = bbox_w + bbox_h

        if hpwl < 1e-6:
            # All pins at same location — skip
            continue

        bbox_w = max(bbox_w, gcell_w * 0.1)  # minimum width
        bbox_h = max(bbox_h, gcell_h * 0.1)  # minimum height
        bbox_area = bbox_w * bbox_h

        # RUDY demand = HPWL / bbox_area
        demand = hpwl / bbox_area if bbox_area > 1e-12 else 0

        # Which G-cells does this net overlap?
        gx_lo = max(0, int((net["x_min"] - x_min) / gcell_w))
        gx_hi = min(gs - 1, int((net["x_max"] - x_min) / gcell_w))
        gy_lo = max(0, int((net["y_min"] - y_min) / gcell_h))
        gy_hi = min(gs - 1, int((net["y_max"] - y_min) / gcell_h))

        for gx in range(gx_lo, gx_hi + 1):
            for gy in range(gy_lo, gy_hi + 1):
                # Compute overlap fraction between net bbox and G-cell
                gcell_xmin = x_min + gx * gcell_w
                gcell_xmax = gcell_xmin + gcell_w
                gcell_ymin = y_min + gy * gcell_h
                gcell_ymax = gcell_ymin + gcell_h

                overlap_xmin = max(net["x_min"], gcell_xmin)
                overlap_xmax = min(net["x_max"], gcell_xmax)
                overlap_ymin = max(net["y_min"], gcell_ymin)
                overlap_ymax = min(net["y_max"], gcell_ymax)

                if overlap_xmax > overlap_xmin and overlap_ymax > overlap_ymin:
                    overlap_area = (overlap_xmax - overlap_xmin) * (overlap_ymax - overlap_ymin)
                    rudy_map[gy, gx] += demand * (overlap_area / bbox_area)

    return rudy_map


def compute_features(design_name, gs=6):
    """Compute per-G-cell features: η, net-based RUDY, cell_density, Δ̄, n_cells."""
    from eta_shield_placement import compute_gcell_metrics
    from validate_congested import compute_per_gcell_demand

    positions, widths, heights, die_area, _ = load_design(design_name)
    if positions is None:
        return None

    N = len(positions)
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)

    x_min, y_min = die_area["x_min"], die_area["y_min"]
    x_max, y_max = die_area["x_max"], die_area["y_max"]
    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs
    gcell_area = gcell_w * gcell_h

    # Cell area density (NOT RUDY — this is just sum of cell areas per G-cell)
    cell_density_map = np.zeros((gs, gs))
    ncells_raw = np.zeros((gs, gs), dtype=int)
    for idx in range(N):
        gx = min(int((positions[idx, 0] - x_min) / gcell_w), gs - 1)
        gy = min(int((positions[idx, 1] - y_min) / gcell_h), gs - 1)
        gx, gy = max(0, gx), max(0, gy)
        cell_density_map[gy, gx] += widths[idx] * heights[idx] / gcell_area
        ncells_raw[gy, gx] += 1

    # η at multiple radii
    r_values = [diag * f for f in [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]]
    eta_maps = {}
    dbar_maps = {}

    for r in r_values:
        gains, eta_map, sigma_map, dbar_map, gcell_assign, ncells_map = \
            compute_gcell_metrics(positions, widths, heights, die_area, gs, r)
        eta_maps[r] = eta_map
        dbar_maps[r] = dbar_map

    # Parse net bounding boxes from GR log
    log_path = os.path.join(RESULTS_DIR, f"congested_{design_name}.log")
    if os.path.exists(log_path):
        with open(log_path) as f:
            log_text = f.read()
        net_bboxes = []
        for line in log_text.split('\n'):
            if line.startswith('NB '):
                parts = line.split()
                if len(parts) >= 5:
                    net_bboxes.append({
                        "x_min": float(parts[1]),
                        "y_min": float(parts[2]),
                        "x_max": float(parts[3]),
                        "y_max": float(parts[4]),
                    })

        # Net-based RUDY: HPWL/bbox_area distributed over G-cells
        rudy_map = compute_net_rudy(net_bboxes, die_area, gs)
        # GR demand (used as ground truth target)
        demand_map = compute_per_gcell_demand(net_bboxes, die_area, gs)
        print(f"  Parsed {len(net_bboxes)} net bounding boxes from GR log")
    else:
        print("  No GR log found — cannot compute demand")
        return None

    return {
        "rudy_map": rudy_map,
        "cell_density_map": cell_density_map,
        "eta_maps": eta_maps,
        "dbar_maps": dbar_maps,
        "demand_map": demand_map,
        "ncells_map": ncells_raw,
        "r_values": r_values,
        "diag": diag,
        "gs": gs,
    }


def _fit_r2(X, y):
    """Fit linear regression and return R². Returns 0 if degenerate."""
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if np.std(X, axis=0).max() < 1e-10:
        return 0.0, None
    reg = LinearRegression().fit(X, y)
    return r2_score(y, reg.predict(X)), reg


def run_regression(features, design_name):
    """Compare prediction accuracy: η, RUDY, cell_density, and combinations."""
    print(f"\n{'='*70}")
    print(f"η + RUDY Combined Predictor (net-based RUDY): {design_name}")
    print(f"{'='*70}")

    mask = features["ncells_map"].ravel() >= 2
    if mask.sum() < 5:
        print("  Not enough active G-cells")
        return None

    y = features["demand_map"].ravel()[mask]
    rudy = features["rudy_map"].ravel()[mask].reshape(-1, 1)
    cell_dens = features["cell_density_map"].ravel()[mask].reshape(-1, 1)
    ncells = features["ncells_map"].ravel()[mask].reshape(-1, 1)

    if np.std(y) < 1e-10:
        print("  No variance in demand")
        return None

    # RUDY alone (net-based)
    r2_rudy, _ = _fit_r2(rudy, y)

    # Cell density alone
    r2_cell_dens, _ = _fit_r2(cell_dens, y)

    print(f"\n  {'Predictor':>35s}  {'R²':>8s}  {'Δ vs RUDY':>10s}")
    print(f"  {'─'*60}")
    print(f"  {'RUDY alone (net-based)':>35s}  {r2_rudy:>8.4f}  {'baseline':>10s}")
    print(f"  {'cell_density alone':>35s}  {r2_cell_dens:>8.4f}  {r2_cell_dens-r2_rudy:>+10.4f}")
    print()

    best_combined_r2 = r2_rudy
    best_r = None
    results = []

    for r in features["r_values"]:
        eta = features["eta_maps"][r].ravel()[mask].reshape(-1, 1)
        dbar = features["dbar_maps"][r].ravel()[mask].reshape(-1, 1)
        r_over_diag = r / features["diag"]

        # η alone
        r2_eta, _ = _fit_r2(eta, y)

        # η + RUDY
        r2_eta_rudy, reg_eta_rudy = _fit_r2(np.hstack([eta, rudy]), y)

        # η + cell_density
        r2_eta_cd, _ = _fit_r2(np.hstack([eta, cell_dens]), y)

        # η + RUDY + cell_density
        r2_eta_rudy_cd, _ = _fit_r2(np.hstack([eta, rudy, cell_dens]), y)

        # ALL = η + RUDY + cell_density + Δ̄ + n_cells
        r2_all, _ = _fit_r2(np.hstack([eta, rudy, cell_dens, dbar, ncells]), y)

        print(f"  --- r = {r_over_diag:.1f}·diag ---")
        print(f"  {'η alone':>35s}  {r2_eta:>8.4f}  {r2_eta-r2_rudy:>+10.4f}")
        print(f"  {'η + RUDY':>35s}  {r2_eta_rudy:>8.4f}  {r2_eta_rudy-r2_rudy:>+10.4f}")
        print(f"  {'η + cell_density':>35s}  {r2_eta_cd:>8.4f}  {r2_eta_cd-r2_rudy:>+10.4f}")
        print(f"  {'η + RUDY + cell_density':>35s}  {r2_eta_rudy_cd:>8.4f}  {r2_eta_rudy_cd-r2_rudy:>+10.4f}")
        print(f"  {'ALL (η+RUDY+cd+Δ̄+n)':>35s}  {r2_all:>8.4f}  {r2_all-r2_rudy:>+10.4f}")
        print()

        coef_eta = float(reg_eta_rudy.coef_[0]) if reg_eta_rudy else 0.0
        coef_rudy = float(reg_eta_rudy.coef_[1]) if reg_eta_rudy else 0.0

        entry = {
            "r": float(r),
            "r_over_diag": float(r_over_diag),
            "r2_eta": float(r2_eta),
            "r2_rudy": float(r2_rudy),
            "r2_cell_density": float(r2_cell_dens),
            "r2_eta_rudy": float(r2_eta_rudy),
            "r2_eta_cell_density": float(r2_eta_cd),
            "r2_eta_rudy_cell_density": float(r2_eta_rudy_cd),
            "r2_all": float(r2_all),
            "coef_eta": coef_eta,
            "coef_rudy": coef_rudy,
        }
        results.append(entry)

        if r2_eta_rudy > best_combined_r2:
            best_combined_r2 = r2_eta_rudy
            best_r = r

    # Summary
    print(f"  {'─'*60}")
    if best_r:
        r_d = best_r / features["diag"]
        print(f"  η ADDS INFORMATION: best R²(η+RUDY) = {best_combined_r2:.4f} "
              f"at r={r_d:.1f}·diag")
        print(f"  Improvement over RUDY alone: {best_combined_r2 - r2_rudy:+.4f}")
    else:
        print(f"  η does NOT add information beyond RUDY")

    output = {
        "design": design_name,
        "gs": features["gs"],
        "r2_rudy_alone": float(r2_rudy),
        "r2_cell_density_alone": float(r2_cell_dens),
        "best_r2_combined": float(best_combined_r2),
        "best_r": float(best_r) if best_r else None,
        "improvement": float(best_combined_r2 - r2_rudy),
        "per_radius": results,
    }

    json_path = os.path.join(RESULTS_DIR, f"eta_plus_rudy_{design_name}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    plot_combined(results, r2_rudy, r2_cell_dens, design_name, features["diag"])

    return output


def plot_combined(results, r2_rudy, r2_cell_dens, design_name, diag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Does η Add Information Beyond RUDY? — {design_name}\n"
                 f"(net-based RUDY, not cell density)", fontsize=14)

    r_diags = [r["r_over_diag"] for r in results]

    # (a) R² comparison across radii
    ax = axes[0]
    ax.axhline(y=r2_rudy, color='blue', ls='--', lw=2, label='RUDY alone (net-based)')
    ax.axhline(y=r2_cell_dens, color='orange', ls=':', lw=2, label='cell_density alone')
    ax.plot(r_diags, [r["r2_eta"] for r in results],
            'r-o', label='η alone', markersize=6)
    ax.plot(r_diags, [r["r2_eta_rudy"] for r in results],
            'g-s', label='η + RUDY', markersize=6, lw=2)
    ax.plot(r_diags, [r["r2_eta_rudy_cell_density"] for r in results],
            'purple', ls='-', marker='^', label='η + RUDY + cell_density', markersize=6)
    ax.plot(r_diags, [r["r2_all"] for r in results],
            'k-D', label='ALL', markersize=5, alpha=0.7)
    ax.set_xlabel("r / diag")
    ax.set_ylabel("R² vs GR demand")
    ax.set_title("(a) Prediction accuracy by predictor set")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)

    # (b) η coefficient in combined model
    ax = axes[1]
    coef_eta = [r["coef_eta"] for r in results]
    coef_rudy = [r["coef_rudy"] for r in results]
    ax.plot(r_diags, coef_eta, 'r-o', label='β_η (η coefficient)', markersize=6)
    ax.plot(r_diags, coef_rudy, 'b-s', label='β_RUDY (RUDY coefficient)', markersize=6)
    ax.axhline(y=0, color='gray', ls='-', alpha=0.3)
    ax.set_xlabel("r / diag")
    ax.set_ylabel("Regression coefficient")
    ax.set_title("(b) η coefficient in combined model\n(>0 means η adds info)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f"eta_plus_rudy_{design_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--design", type=str, default="gcd_nangate45")
    parser.add_argument("--gs", type=int, default=6)
    args = parser.parse_args()

    features = compute_features(args.design, gs=args.gs)
    if features:
        run_regression(features, args.design)


if __name__ == "__main__":
    main()
