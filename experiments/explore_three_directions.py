#!/usr/bin/env python3
"""
Three explorations beyond η-shield (Issue #12)
===============================================

1. σ_min early warning: Does σ_min predict congestion BEFORE η > 0?
   Gradually compress a design, track (σ_min, η) trajectory.
   Hypothesis: σ_min → 0 continuously while η stays at 0, then η jumps.

2. Collapse dynamics: Start from generic 2D, snap to rows, then grid.
   Measure rank and η at each stage. Show η is monotonically increasing.
   This is the "drone collapse" metaphor made precise.

3. Excess η decomposition: η_actual = η_generic + η_gap.
   η_generic = max(0, 1-4/Δ̄) is density. η_gap = grid structure.
   Show this decomposition varies by PDK (SKY130 >> NanGate45 >> ASAP7).

Usage:
  python experiments/explore_three_directions.py
"""

import json
import os
import sys
import numpy as np
from scipy.linalg import svd
from scipy.spatial import KDTree
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from run_batch import (
    build_overlap_coboundary, compute_eta, theory_eta,
    DESIGNS, load_design,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "explore")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# Exploration 1: σ_min Early Warning
# ═══════════════════════════════════════════════════════════════════

def explore_sigma_min(design_name="gcd_nangate45"):
    """Gradually increase density, track σ_min and η.

    Start from the real placement (sparse, η≈0), progressively
    compress toward center. At each compression level, compute
    σ_min and η. Plot the trajectory.

    The prediction (from sheaf-swarm eta_signal_geometry.md):
    σ_min approaches 0 BEFORE η jumps from 0 to >0.
    """
    print(f"\n{'='*70}")
    print(f"EXPLORATION 1: σ_min Early Warning — {design_name}")
    print(f"{'='*70}")

    positions, widths, heights, die_area, _ = load_design(design_name)
    if positions is None:
        return None

    N = len(positions)
    # Subsample for SVD tractability
    if N > 500:
        rng = np.random.default_rng(42)
        cx, cy = np.mean(positions, axis=0)
        dists = np.sqrt((positions[:, 0] - cx)**2 + (positions[:, 1] - cy)**2)
        idx = np.argsort(dists)[:500]
        positions = positions[idx]
        widths = widths[idx]
        heights = heights[idx]
        N = len(positions)

    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)
    r_interact = diag * 1.5

    centroid = np.mean(positions, axis=0)

    # Sweep compression from 1.0 (no compression) to 0.15 (extreme)
    compress_factors = np.concatenate([
        np.linspace(1.0, 0.5, 20),
        np.linspace(0.48, 0.2, 20),
        np.linspace(0.19, 0.10, 10),
    ])

    results = []
    print(f"\n  {'factor':>8s}  {'|E|':>6s}  {'Δ̄':>6s}  {'rank':>5s}  "
          f"{'η':>8s}  {'σ_min':>8s}  {'σ_2':>8s}  {'phase':>6s}")
    print(f"  {'─'*70}")

    for cf in compress_factors:
        # Compress all cells toward centroid
        pos = centroid + (positions - centroid) * cf

        tree = KDTree(pos)
        pairs = tree.query_pairs(r=r_interact)
        edges = [(min(i, j), max(i, j)) for i, j in pairs]
        nE = len(edges)

        if nE < 2:
            results.append({
                "compress": float(cf), "nE": nE, "dbar": 0,
                "eta": 0, "sigma_min": float('inf'), "sigma_2": float('inf'),
                "rank": 0,
            })
            continue

        dbar = 2.0 * nE / N
        delta = build_overlap_coboundary(pos, edges, n_v=2)
        sv = svd(delta, compute_uv=False)
        rank = int(np.sum(sv > 1e-10))
        eta = (nE - rank) / nE if nE > 0 else 0.0
        sigma_min = float(sv[rank - 1]) if rank > 0 else 0.0
        # Second smallest singular value (for gap analysis)
        sigma_2 = float(sv[rank - 2]) if rank > 1 else sigma_min

        phase = "safe" if eta == 0 else "congested"

        entry = {
            "compress": float(cf),
            "nE": int(nE),
            "dbar": float(dbar),
            "eta": float(eta),
            "eta_generic": float(theory_eta(dbar, 2)),
            "sigma_min": float(sigma_min),
            "sigma_2": float(sigma_2),
            "rank": int(rank),
        }
        results.append(entry)

        if len(results) % 5 == 0 or eta > 0:
            print(f"  {cf:8.3f}  {nE:6d}  {dbar:6.2f}  {rank:5d}  "
                  f"{eta:8.4f}  {sigma_min:8.4f}  {sigma_2:8.4f}  {phase:>6s}")

    # Find the transition point
    transition_idx = next(
        (i for i, r in enumerate(results) if r["eta"] > 0), len(results)
    )

    # Compute early warning window
    if transition_idx > 0 and transition_idx < len(results):
        # How many steps before transition did σ_min start dropping?
        sigma_at_transition = results[transition_idx]["sigma_min"]
        pre_transition = results[:transition_idx]
        # Find where σ_min drops below 50% of its max
        sigma_max = max(r["sigma_min"] for r in pre_transition) if pre_transition else 1
        warning_idx = next(
            (i for i, r in enumerate(pre_transition)
             if r["sigma_min"] < sigma_max * 0.5 and r["sigma_min"] > 0),
            transition_idx
        )
        warning_window = transition_idx - warning_idx

        print(f"\n  Phase transition at compress={results[transition_idx]['compress']:.3f} "
              f"(Δ̄={results[transition_idx]['dbar']:.2f})")
        print(f"  σ_min warning started at compress={results[warning_idx]['compress']:.3f}")
        print(f"  Early warning window: {warning_window} steps "
              f"({results[warning_idx]['compress']:.3f} → "
              f"{results[transition_idx]['compress']:.3f})")
    else:
        warning_window = 0

    output = {
        "exploration": "sigma_min_early_warning",
        "design": design_name,
        "N": N,
        "r_interact": float(r_interact),
        "transition_idx": transition_idx,
        "warning_window": warning_window,
        "trajectory": results,
    }

    json_path = os.path.join(RESULTS_DIR, f"sigma_min_{design_name}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    # Plot
    plot_sigma_min(results, design_name, transition_idx)

    return output


def plot_sigma_min(results, design_name, transition_idx):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"σ_min Early Warning: {design_name}", fontsize=14)

    cfs = [r["compress"] for r in results]
    etas = [r["eta"] for r in results]
    sigmas = [r["sigma_min"] for r in results]
    dbars = [r["dbar"] for r in results]

    # Normalize σ_min for comparison
    sigma_max = max(s for s in sigmas if np.isfinite(s) and s > 0) if any(
        np.isfinite(s) and s > 0 for s in sigmas) else 1
    sigmas_norm = [min(s / sigma_max, 2.0) if np.isfinite(s) else 0 for s in sigmas]

    # (a) η and σ_min vs compression
    ax = axes[0]
    ax.plot(cfs, etas, 'r-o', markersize=3, label='η', linewidth=2)
    ax.plot(cfs, sigmas_norm, 'b-s', markersize=3, label='σ_min (normalized)', linewidth=2)
    if transition_idx < len(cfs):
        ax.axvline(x=cfs[transition_idx], color='gray', ls='--', alpha=0.7,
                   label=f'η>0 at cf={cfs[transition_idx]:.2f}')
    ax.set_xlabel("Compression factor (1=original, 0=fully compressed)")
    ax.set_ylabel("Value")
    ax.set_title("(a) η and σ_min vs compression")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.invert_xaxis()

    # (b) η and σ_min vs Δ̄
    ax = axes[1]
    ax.plot(dbars, etas, 'r-o', markersize=3, label='η (measured)', linewidth=2)
    dbar_curve = np.linspace(0.5, max(dbars) * 1.1, 200)
    eta_curve = [theory_eta(d, 2) for d in dbar_curve]
    ax.plot(dbar_curve, eta_curve, 'r--', alpha=0.5, label='η (generic, n_v=2)')
    ax.plot(dbars, sigmas_norm, 'b-s', markersize=3, label='σ_min (norm)', linewidth=2)
    ax.axvline(x=4, color='gray', ls='--', alpha=0.7, label='Δ̄=4 threshold')
    ax.set_xlabel("Δ̄ (average degree)")
    ax.set_ylabel("Value")
    ax.set_title("(b) Phase transition at Δ̄=4")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) σ_min zoom near transition
    ax = axes[2]
    if transition_idx > 5:
        start = max(0, transition_idx - 15)
        end = min(len(results), transition_idx + 10)
        zoom_cfs = cfs[start:end]
        zoom_etas = etas[start:end]
        zoom_sigmas = sigmas[start:end]

        ax.plot(zoom_cfs, zoom_etas, 'r-o', markersize=5, label='η', linewidth=2)
        ax2 = ax.twinx()
        ax2.plot(zoom_cfs, zoom_sigmas, 'b-s', markersize=5, label='σ_min (raw)', linewidth=2)
        ax2.set_ylabel("σ_min (raw)", color='blue')
        ax.set_xlabel("Compression factor")
        ax.set_ylabel("η", color='red')
        ax.set_title("(c) Zoom: σ_min leads η transition")
        ax.invert_xaxis()
        ax.grid(True, alpha=0.3)

        # Mark warning window
        if transition_idx < len(cfs):
            ax.axvline(x=cfs[transition_idx], color='red', ls=':', alpha=0.7)
            ax.fill_betweenx([0, 1], cfs[transition_idx],
                             cfs[max(0, transition_idx-5)],
                             alpha=0.1, color='blue', label='warning window')
        ax.legend(loc='upper left', fontsize=8)
        ax2.legend(loc='upper right', fontsize=8)
    else:
        ax.text(0.5, 0.5, "No clear transition\n(η>0 from start)",
                ha='center', va='center', transform=ax.transAxes)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f"sigma_min_{design_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════
# Exploration 2: Collapse Dynamics
# ═══════════════════════════════════════════════════════════════════

def explore_collapse(design_name="gcd_nangate45"):
    """Measure η at each stage of the DOF collapse chain.

    Stage 0: Generic 2D positions (add small random perturbation to break grid)
    Stage 1: Real 2D placement (continuous x, quantized y to rows)
    Stage 2: Row-snapped (snap y to nearest row, keep x continuous)
    Stage 3: Grid-snapped (snap both x and y to grid)

    At each stage, compute rank(δ) and η for multiple radii.
    Prediction: η₀ ≤ η₁ ≤ η₂ ≤ η₃ (monotonic through collapse).
    """
    print(f"\n{'='*70}")
    print(f"EXPLORATION 2: Collapse Dynamics — {design_name}")
    print(f"{'='*70}")

    positions, widths, heights, die_area, _ = load_design(design_name)
    if positions is None:
        return None

    N = len(positions)
    if N > 500:
        rng = np.random.default_rng(42)
        cx, cy = np.mean(positions, axis=0)
        dists = np.sqrt((positions[:, 0] - cx)**2 + (positions[:, 1] - cy)**2)
        idx = np.argsort(dists)[:500]
        positions = positions[idx]
        widths = widths[idx]
        heights = heights[idx]
        N = len(positions)

    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)

    # Detect row structure
    y_coords = positions[:, 1]
    y_unique = np.unique(np.round(y_coords, decimals=3))
    n_rows = len(y_unique)
    if n_rows > 1:
        row_pitch = np.median(np.diff(np.sort(y_unique)))
    else:
        row_pitch = med_h

    # Detect x grid
    x_coords = positions[:, 0]
    x_diffs = np.diff(np.sort(x_coords))
    x_diffs_nonzero = x_diffs[x_diffs > 1e-6]
    site_width = np.median(x_diffs_nonzero) if len(x_diffs_nonzero) > 0 else med_w

    print(f"  N={N}, n_rows={n_rows}, row_pitch={row_pitch:.4f}, "
          f"site_width={site_width:.4f}")

    # Create 4 stages
    rng = np.random.default_rng(42)

    # Stage 0: Generic — perturb positions by ~10% of cell size
    pos_generic = positions.copy()
    pos_generic += rng.normal(0, diag * 0.3, size=pos_generic.shape)

    # Stage 1: Real placement (as-is)
    pos_real = positions.copy()

    # Stage 2: Row-snapped (y to nearest row, x continuous)
    pos_rowsnap = positions.copy()
    for i in range(N):
        nearest_row = y_unique[np.argmin(np.abs(y_unique - pos_rowsnap[i, 1]))]
        pos_rowsnap[i, 1] = nearest_row

    # Stage 3: Grid-snapped (both x and y)
    pos_gridsnap = pos_rowsnap.copy()
    x_min = np.min(x_coords)
    for i in range(N):
        n_sites = round((pos_gridsnap[i, 0] - x_min) / site_width)
        pos_gridsnap[i, 0] = x_min + n_sites * site_width

    stages = [
        ("generic", pos_generic),
        ("real", pos_real),
        ("row-snap", pos_rowsnap),
        ("grid-snap", pos_gridsnap),
    ]

    # Sweep radius for each stage
    r_values = sorted(set([
        diag * f for f in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0]
    ]))

    all_results = {}

    for stage_name, pos in stages:
        stage_results = []
        for r in r_values:
            tree = KDTree(pos)
            pairs = tree.query_pairs(r=r)
            edges = [(min(i, j), max(i, j)) for i, j in pairs]
            nE = len(edges)
            if nE < 2:
                continue
            dbar = 2.0 * nE / N

            delta = build_overlap_coboundary(pos, edges, n_v=2)
            sv = svd(delta, compute_uv=False)
            rank = int(np.sum(sv > 1e-10))
            eta = (nE - rank) / nE

            stage_results.append({
                "r": float(r),
                "nE": int(nE),
                "dbar": float(dbar),
                "rank": int(rank),
                "eta": float(eta),
                "eta_generic": float(theory_eta(dbar, 2)),
            })

            if nE > 10000:
                break

        all_results[stage_name] = stage_results

    # Print comparison table
    print(f"\n  {'r':>6s}  {'Δ̄':>6s}  "
          f"{'η_generic':>10s}  {'η_real':>10s}  {'η_rowsnap':>10s}  {'η_gridsnap':>10s}  "
          f"{'gap_row':>8s}  {'gap_grid':>8s}")
    print(f"  {'─'*80}")

    # Align by closest dbar
    for entry_g in all_results.get("generic", []):
        r = entry_g["r"]
        dbar = entry_g["dbar"]

        def find_eta(stage, r_target):
            entries = all_results.get(stage, [])
            for e in entries:
                if abs(e["r"] - r_target) < 0.01:
                    return e["eta"]
            return None

        eta_gen = entry_g["eta"]
        eta_real = find_eta("real", r)
        eta_row = find_eta("row-snap", r)
        eta_grid = find_eta("grid-snap", r)

        if eta_real is not None and eta_row is not None and eta_grid is not None:
            gap_row = eta_row - eta_real
            gap_grid = eta_grid - eta_row
            print(f"  {r:6.2f}  {dbar:6.2f}  "
                  f"{eta_gen:10.4f}  {eta_real:10.4f}  {eta_row:10.4f}  {eta_grid:10.4f}  "
                  f"{gap_row:+8.4f}  {gap_grid:+8.4f}")

    # Check monotonicity
    monotonic_violations = 0
    for entry_g in all_results.get("generic", []):
        r = entry_g["r"]
        etas = []
        for stage in ["generic", "real", "row-snap", "grid-snap"]:
            for e in all_results.get(stage, []):
                if abs(e["r"] - r) < 0.01:
                    etas.append(e["eta"])
                    break
        if len(etas) == 4:
            for i in range(3):
                if etas[i+1] < etas[i] - 0.01:  # tolerance
                    monotonic_violations += 1

    print(f"\n  Monotonicity violations: {monotonic_violations}")
    if monotonic_violations == 0:
        print(f"  ✓ η monotonically increases through collapse chain")
    else:
        print(f"  ✗ {monotonic_violations} violations — collapse doesn't always increase η")

    output = {
        "exploration": "collapse_dynamics",
        "design": design_name,
        "N": N,
        "n_rows": int(n_rows),
        "row_pitch": float(row_pitch),
        "site_width": float(site_width),
        "stages": all_results,
        "monotonic_violations": monotonic_violations,
    }

    json_path = os.path.join(RESULTS_DIR, f"collapse_{design_name}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    plot_collapse(all_results, design_name)
    return output


def plot_collapse(all_results, design_name):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Collapse Dynamics: {design_name}", fontsize=14)

    colors = {"generic": "#2ca02c", "real": "#1f77b4",
              "row-snap": "#ff7f0e", "grid-snap": "#d62728"}
    markers = {"generic": "o", "real": "s", "row-snap": "^", "grid-snap": "D"}

    # (a) η vs Δ̄ for each stage
    ax = axes[0]
    for stage, entries in all_results.items():
        dbars = [e["dbar"] for e in entries]
        etas = [e["eta"] for e in entries]
        ax.plot(dbars, etas, f'{markers[stage]}-', color=colors[stage],
                label=stage, markersize=5, linewidth=2)

    dbar_curve = np.linspace(0.5, 15, 200)
    ax.plot(dbar_curve, [theory_eta(d, 2) for d in dbar_curve],
            'k--', alpha=0.4, label='generic theory (n_v=2)')
    ax.plot(dbar_curve, [theory_eta(d, 1) for d in dbar_curve],
            'k:', alpha=0.4, label='collinear theory (n_v=1)')
    ax.axvline(x=4, color='gray', ls='--', alpha=0.3)
    ax.set_xlabel("Δ̄")
    ax.set_ylabel("η")
    ax.set_title("(a) η vs Δ̄ at each collapse stage")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.0)

    # (b) Gap (η_stage - η_generic) vs Δ̄
    ax = axes[1]
    generic_entries = {e["r"]: e for e in all_results.get("generic", [])}
    for stage in ["real", "row-snap", "grid-snap"]:
        entries = all_results.get(stage, [])
        gaps = []
        dbars_gap = []
        for e in entries:
            ge = generic_entries.get(e["r"])
            if ge:
                gaps.append(e["eta"] - ge["eta"])
                dbars_gap.append(e["dbar"])
        if gaps:
            ax.plot(dbars_gap, gaps, f'{markers[stage]}-', color=colors[stage],
                    label=f'{stage} gap', markersize=5, linewidth=2)

    ax.axhline(y=0, color='gray', ls='-', alpha=0.3)
    ax.set_xlabel("Δ̄")
    ax.set_ylabel("η_stage - η_generic (gap)")
    ax.set_title("(b) Genericity gap at each stage")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f"collapse_{design_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════
# Exploration 3: Excess η Decomposition
# ═══════════════════════════════════════════════════════════════════

def explore_excess_eta(designs=None):
    """Decompose η into density and structural components across PDKs.

    η_actual = η_density + η_structural
    where:
      η_density = max(0, 1-4/Δ̄) — pure density (Theorem PS1)
      η_structural = η_actual - η_density — grid/collinearity (Theorem G'')

    Show that η_structural is PDK-dependent:
      SKY130 >> NanGate45 >> ASAP7
    """
    if designs is None:
        designs = ["gcd_nangate45", "gcd_sky130", "gcd_asap7"]

    print(f"\n{'='*70}")
    print(f"EXPLORATION 3: Excess η Decomposition")
    print(f"{'='*70}")

    all_results = {}

    for design_name in designs:
        positions, widths, heights, die_area, _ = load_design(design_name)
        if positions is None:
            continue

        N = len(positions)
        if N > 500:
            rng = np.random.default_rng(42)
            cx, cy = np.mean(positions, axis=0)
            dists = np.sqrt((positions[:, 0] - cx)**2 + (positions[:, 1] - cy)**2)
            idx = np.argsort(dists)[:500]
            positions = positions[idx]
            widths = widths[idx]
            heights = heights[idx]
            N = len(positions)

        med_w = np.median(widths)
        med_h = np.median(heights)
        diag = np.sqrt(med_w**2 + med_h**2)

        # PDK info
        pdk = DESIGNS[design_name]["pdk"]
        if pdk == "sky130hd":
            site_width = 0.46
            row_pitch = 2.72
        elif pdk == "nangate45":
            site_width = 0.19
            row_pitch = 1.40
        elif pdk == "asap7":
            site_width = 0.054
            row_pitch = 0.27
        else:
            site_width = med_w
            row_pitch = med_h

        diag_dist = np.sqrt(site_width**2 + row_pitch**2)

        r_values = sorted(set([
            diag * f for f in [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0, 10.0, 15.0]
        ]))

        results = []
        for r in r_values:
            tree = KDTree(positions)
            pairs = tree.query_pairs(r=r)
            edges = [(min(i, j), max(i, j)) for i, j in pairs]
            nE = len(edges)
            if nE < 2:
                continue
            dbar = 2.0 * nE / N

            delta = build_overlap_coboundary(positions, edges, n_v=2)
            sv = svd(delta, compute_uv=False)
            rank = int(np.sum(sv > 1e-10))
            eta_actual = (nE - rank) / nE

            eta_density = theory_eta(dbar, 2)
            eta_structural = max(0, eta_actual - eta_density)

            # Structural fraction
            structural_frac = eta_structural / max(eta_actual, 1e-10) if eta_actual > 0 else 0

            # Phase based on r vs diagonal distance
            if r < row_pitch:
                phase = "H-only"
            elif r < diag_dist:
                phase = "H+V"
            else:
                phase = "diagonal"

            results.append({
                "r": float(r),
                "r_over_diag": float(r / diag_dist),
                "nE": int(nE),
                "dbar": float(dbar),
                "eta_actual": float(eta_actual),
                "eta_density": float(eta_density),
                "eta_structural": float(eta_structural),
                "structural_frac": float(structural_frac),
                "rank": int(rank),
                "rank_generic": min(nE, 2 * N),
                "phase": phase,
            })

            if nE > 15000:
                break

        all_results[design_name] = {
            "pdk": pdk,
            "N": N,
            "site_width": site_width,
            "row_pitch": row_pitch,
            "diag_dist": diag_dist,
            "data": results,
        }

    # Print comparison
    print(f"\n  {'Design':>25s}  {'PDK':>10s}  {'site':>6s}  {'pitch':>6s}  "
          f"{'diag':>6s}  {'peak η_struct':>12s}  {'at Δ̄':>6s}")
    print(f"  {'─'*80}")

    for design, info in all_results.items():
        data = info["data"]
        if data:
            peak_struct = max(d["eta_structural"] for d in data)
            peak_entry = [d for d in data if d["eta_structural"] == peak_struct][0]
            print(f"  {design:>25s}  {info['pdk']:>10s}  "
                  f"{info['site_width']:>6.3f}  {info['row_pitch']:>6.2f}  "
                  f"{info['diag_dist']:>6.3f}  {peak_struct:>12.4f}  "
                  f"{peak_entry['dbar']:>6.2f}")

    # Detailed table per design
    for design, info in all_results.items():
        print(f"\n  {design} ({info['pdk']}):")
        print(f"    {'r':>6s}  {'r/diag':>6s}  {'Δ̄':>6s}  "
              f"{'η_actual':>8s}  {'η_density':>9s}  {'η_struct':>8s}  "
              f"{'%struct':>7s}  {'phase':>8s}")
        print(f"    {'─'*70}")
        for d in info["data"]:
            print(f"    {d['r']:>6.2f}  {d['r_over_diag']:>6.2f}  {d['dbar']:>6.2f}  "
                  f"{d['eta_actual']:>8.4f}  {d['eta_density']:>9.4f}  "
                  f"{d['eta_structural']:>8.4f}  "
                  f"{d['structural_frac']*100:>6.1f}%  {d['phase']:>8s}")

    output = {
        "exploration": "excess_eta_decomposition",
        "designs": all_results,
    }

    json_path = os.path.join(RESULTS_DIR, "excess_eta_decomposition.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)

    plot_excess_eta(all_results)
    return output


def plot_excess_eta(all_results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_designs = len(all_results)
    fig, axes = plt.subplots(1, n_designs, figsize=(6 * n_designs, 5))
    if n_designs == 1:
        axes = [axes]

    fig.suptitle("Excess η Decomposition: Density vs Structure", fontsize=14)

    for ax, (design, info) in zip(axes, all_results.items()):
        data = info["data"]
        dbars = [d["dbar"] for d in data]
        eta_actual = [d["eta_actual"] for d in data]
        eta_density = [d["eta_density"] for d in data]
        eta_struct = [d["eta_structural"] for d in data]

        ax.fill_between(dbars, 0, eta_density, alpha=0.3, color='blue',
                        label='η_density (PS1)')
        ax.fill_between(dbars, eta_density, eta_actual, alpha=0.3, color='red',
                        label='η_structural (G\'\')')
        ax.plot(dbars, eta_actual, 'ko-', markersize=4, label='η_actual (SVD)',
                linewidth=2)
        ax.plot(dbars, eta_density, 'b--', alpha=0.7, linewidth=1.5)

        # Mark diagonal threshold
        diag_dist = info["diag_dist"]
        ax.axvline(x=4, color='gray', ls=':', alpha=0.5)

        ax.set_xlabel("Δ̄")
        ax.set_ylabel("η")
        ax.set_title(f"{design}\n({info['pdk']}, site={info['site_width']:.3f})")
        ax.legend(fontsize=7, loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.0)
        ax.set_xlim(0, min(max(dbars) * 1.1, 20))

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "excess_eta_decomposition.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved: {fig_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════════╗")
    print("║  Three Explorations Beyond η-Shield                     ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # 1. σ_min early warning
    r1 = explore_sigma_min("gcd_nangate45")

    # 2. Collapse dynamics
    r2a = explore_collapse("gcd_nangate45")
    r2b = explore_collapse("gcd_sky130")

    # 3. Excess η decomposition
    r3 = explore_excess_eta(["gcd_nangate45", "gcd_sky130", "gcd_asap7"])

    print(f"\n{'='*70}")
    print("ALL EXPLORATIONS COMPLETE")
    print(f"{'='*70}")
    print(f"  Results in: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
