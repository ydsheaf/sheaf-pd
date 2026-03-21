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


def compute_features(design_name, gs=6):
    """Compute per-G-cell features: η, RUDY, Δ̄, n_cells, and variants."""
    from eta_shield_placement import compute_gcell_metrics
    from validate_congested import compute_per_gcell_demand, run_congested_gr

    positions, widths, heights, die_area, _ = load_design(design_name)
    if positions is None:
        return None

    N = len(positions)
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)

    # RUDY
    x_min, y_min = die_area["x_min"], die_area["y_min"]
    x_max, y_max = die_area["x_max"], die_area["y_max"]
    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs
    gcell_area = gcell_w * gcell_h

    rudy_map = np.zeros((gs, gs))
    ncells_raw = np.zeros((gs, gs), dtype=int)
    for idx in range(N):
        gx = min(int((positions[idx, 0] - x_min) / gcell_w), gs - 1)
        gy = min(int((positions[idx, 1] - y_min) / gcell_h), gs - 1)
        gx, gy = max(0, gx), max(0, gy)
        rudy_map[gy, gx] += widths[idx] * heights[idx] / gcell_area
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

    # GR demand (need net bboxes — load from log if available)
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
        demand_map = compute_per_gcell_demand(net_bboxes, die_area, gs)
    else:
        print("  No GR log found — cannot compute demand")
        return None

    return {
        "rudy_map": rudy_map,
        "eta_maps": eta_maps,
        "dbar_maps": dbar_maps,
        "demand_map": demand_map,
        "ncells_map": ncells_raw,
        "r_values": r_values,
        "diag": diag,
        "gs": gs,
    }


def run_regression(features, design_name):
    """Compare prediction accuracy of η, RUDY, and η+RUDY combined."""
    print(f"\n{'='*70}")
    print(f"η + RUDY Combined Predictor: {design_name}")
    print(f"{'='*70}")

    mask = features["ncells_map"].ravel() >= 2
    if mask.sum() < 5:
        print("  Not enough active G-cells")
        return None

    y = features["demand_map"].ravel()[mask]
    rudy = features["rudy_map"].ravel()[mask].reshape(-1, 1)

    if np.std(y) < 1e-10:
        print("  No variance in demand")
        return None

    # RUDY alone
    reg_rudy = LinearRegression().fit(rudy, y)
    r2_rudy = r2_score(y, reg_rudy.predict(rudy))

    print(f"\n  {'Predictor':>25s}  {'R²':>8s}  {'Δ vs RUDY':>10s}")
    print(f"  {'─'*50}")
    print(f"  {'RUDY alone':>25s}  {r2_rudy:>8.4f}  {'baseline':>10s}")

    best_combined_r2 = r2_rudy
    best_r = None
    results = []

    for r in features["r_values"]:
        eta = features["eta_maps"][r].ravel()[mask].reshape(-1, 1)
        dbar = features["dbar_maps"][r].ravel()[mask].reshape(-1, 1)

        # η alone
        if np.std(eta) > 1e-10:
            reg_eta = LinearRegression().fit(eta, y)
            r2_eta = r2_score(y, reg_eta.predict(eta))
        else:
            r2_eta = 0.0

        # Δ̄ alone
        if np.std(dbar) > 1e-10:
            reg_dbar = LinearRegression().fit(dbar, y)
            r2_dbar = r2_score(y, reg_dbar.predict(dbar))
        else:
            r2_dbar = 0.0

        # η + RUDY
        X_combined = np.hstack([eta, rudy])
        reg_combined = LinearRegression().fit(X_combined, y)
        r2_combined = r2_score(y, reg_combined.predict(X_combined))

        # η + RUDY + Δ̄
        X_full = np.hstack([eta, rudy, dbar])
        reg_full = LinearRegression().fit(X_full, y)
        r2_full = r2_score(y, reg_full.predict(X_full))

        # η + RUDY + Δ̄ + n_cells
        ncells = features["ncells_map"].ravel()[mask].reshape(-1, 1)
        X_all = np.hstack([eta, rudy, dbar, ncells])
        reg_all = LinearRegression().fit(X_all, y)
        r2_all = r2_score(y, reg_all.predict(X_all))

        r_over_diag = r / features["diag"]
        delta_combined = r2_combined - r2_rudy
        delta_full = r2_full - r2_rudy

        print(f"  {'η(r=%.1f·d)' % r_over_diag:>25s}  {r2_eta:>8.4f}  {r2_eta-r2_rudy:>+10.4f}")
        print(f"  {'Δ̄(r=%.1f·d)' % r_over_diag:>25s}  {r2_dbar:>8.4f}  {r2_dbar-r2_rudy:>+10.4f}")
        print(f"  {'η+RUDY(r=%.1f·d)' % r_over_diag:>25s}  {r2_combined:>8.4f}  {delta_combined:>+10.4f}")
        print(f"  {'η+RUDY+Δ̄(r=%.1f·d)' % r_over_diag:>25s}  {r2_full:>8.4f}  {delta_full:>+10.4f}")
        print(f"  {'ALL(r=%.1f·d)' % r_over_diag:>25s}  {r2_all:>8.4f}  {r2_all-r2_rudy:>+10.4f}")
        print()

        entry = {
            "r": float(r),
            "r_over_diag": float(r_over_diag),
            "r2_eta": float(r2_eta),
            "r2_dbar": float(r2_dbar),
            "r2_rudy": float(r2_rudy),
            "r2_eta_rudy": float(r2_combined),
            "r2_eta_rudy_dbar": float(r2_full),
            "r2_all": float(r2_all),
            "coef_eta": float(reg_combined.coef_[0]),
            "coef_rudy": float(reg_combined.coef_[1]),
        }
        results.append(entry)

        if r2_combined > best_combined_r2:
            best_combined_r2 = r2_combined
            best_r = r

    # Summary
    print(f"  {'─'*50}")
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
        "best_r2_combined": float(best_combined_r2),
        "best_r": float(best_r) if best_r else None,
        "improvement": float(best_combined_r2 - r2_rudy),
        "per_radius": results,
    }

    json_path = os.path.join(RESULTS_DIR, f"eta_plus_rudy_{design_name}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    plot_combined(results, r2_rudy, design_name, features["diag"])

    return output


def plot_combined(results, r2_rudy, design_name, diag):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Does η Add Information Beyond RUDY? — {design_name}", fontsize=14)

    r_diags = [r["r_over_diag"] for r in results]

    # (a) R² comparison across radii
    ax = axes[0]
    ax.axhline(y=r2_rudy, color='blue', ls='--', lw=2, label='RUDY alone')
    ax.plot(r_diags, [r["r2_eta"] for r in results],
            'r-o', label='η alone', markersize=6)
    ax.plot(r_diags, [r["r2_eta_rudy"] for r in results],
            'g-s', label='η + RUDY', markersize=6, lw=2)
    ax.plot(r_diags, [r["r2_all"] for r in results],
            'purple', ls='-', marker='^', label='η + RUDY + Δ̄ + n_cells', markersize=6)
    ax.set_xlabel("r / diag")
    ax.set_ylabel("R² vs GR demand")
    ax.set_title("(a) Prediction accuracy by predictor set")
    ax.legend(fontsize=8)
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
    features = compute_features("gcd_nangate45", gs=6)
    if features:
        run_regression(features, "gcd_nangate45")


if __name__ == "__main__":
    main()
