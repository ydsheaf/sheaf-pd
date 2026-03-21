#!/usr/bin/env python3
"""
η-Shield Placement Optimizer — Phase 1 (Issue #12)
===================================================

Ports the η-shield from sheaf-swarm (Theorem 11) to VLSI placement.

Standard analytical placement iteratively moves cells to resolve overlaps.
When a G-cell is congested (η_α > 0), the standard approach oscillates
(same phenomenon as CBF-QP oscillation in sheaf-swarm).

η-shield: scale cell displacement by max(ε, 1 - η_α) per G-cell.
- Congested G-cells → smaller steps → less oscillation
- Uncongested G-cells → full step → fast convergence

Setup: compress one quadrant of the design to create **localized** congestion.
This produces spatial variation in η — some G-cells have η > 0 (congested),
others η = 0 (routable). The shield should help differentially.

A/B comparison:
  1. standard:     fixed step size, no shielding
  2. eta_shield:   step *= max(ε, 1 - η_α) per G-cell
  3. sigma_shield: step *= min(1, σ_min / σ_threshold)

Usage:
  python experiments/eta_shield_placement.py
  python experiments/eta_shield_placement.py --design gcd_nangate45
  python experiments/eta_shield_placement.py --design aes_nangate45 --iters 200
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
from collections import defaultdict
from scipy.linalg import svd
from scipy.spatial import KDTree

sys.path.insert(0, os.path.dirname(__file__))
from run_batch import (
    parse_lef_macros, parse_def_components, PlacedCell,
    build_overlap_coboundary, compute_eta, theory_eta,
    DESIGNS, load_design,
)

# ─── Parameters ───
ETA_GAIN_FLOOR = 0.2       # minimum step fraction even at η→1
SIGMA_THRESHOLD = 0.3      # σ_min below this → start reducing step
STEP_SIZE = 0.4            # base step size (fraction of overlap to resolve)
K_ANCHOR = 0.05            # anchor force toward original position
N_ITERS = 80               # placement iterations
GCELL_GRID = 6             # G-cell grid size for per-region η


# ═══════════════════════════════════════════════════════════════════
# Congestion Setup: Localized Compression
# ═══════════════════════════════════════════════════════════════════

def create_congested_placement(positions, widths, heights, die_area,
                               compress_region="quadrant", compress_factor=0.5,
                               seed=42):
    """Create a placement with localized congestion.

    Compresses cells in one region toward the center of that region,
    creating overlaps. Cells outside the region stay put.

    Args:
        compress_region: "quadrant" (bottom-left), "center", or "stripe"
        compress_factor: how much to compress (0.3 = compress to 30% of area)
    """
    pos = positions.copy()
    N = len(pos)

    if die_area is not None:
        cx = (die_area["x_min"] + die_area["x_max"]) / 2
        cy = (die_area["y_min"] + die_area["y_max"]) / 2
    else:
        cx, cy = np.mean(pos, axis=0)

    if compress_region == "quadrant":
        # Compress cells in bottom-left quadrant
        mask = (pos[:, 0] < cx) & (pos[:, 1] < cy)
    elif compress_region == "center":
        # Compress cells near center
        dists = np.sqrt((pos[:, 0] - cx)**2 + (pos[:, 1] - cy)**2)
        r_half = np.median(dists)
        mask = dists < r_half
    elif compress_region == "stripe":
        # Compress cells in middle horizontal stripe
        y_range = pos[:, 1].max() - pos[:, 1].min()
        y_mid = np.median(pos[:, 1])
        mask = np.abs(pos[:, 1] - y_mid) < y_range * 0.2
    else:
        raise ValueError(f"Unknown compress_region: {compress_region}")

    n_compressed = int(mask.sum())

    # Compress: move selected cells toward their centroid
    if n_compressed > 0:
        centroid = pos[mask].mean(axis=0)
        pos[mask] = centroid + (pos[mask] - centroid) * compress_factor

        # Add small random jitter to break symmetry
        rng = np.random.default_rng(seed)
        med_w = np.median(widths)
        pos[mask] += rng.normal(0, med_w * 0.1, size=(n_compressed, 2))

    return pos, mask, n_compressed


# ═══════════════════════════════════════════════════════════════════
# Overlap Detection & Force Computation
# ═══════════════════════════════════════════════════════════════════

def compute_overlaps(positions, widths, heights):
    """Find all overlapping cell pairs. Returns list of (i, j, overlap_area)."""
    N = len(positions)
    max_dim = max(np.max(widths), np.max(heights)) * 2.5
    tree = KDTree(positions)
    pairs = tree.query_pairs(r=max_dim)

    overlaps = []
    for i, j in pairs:
        dx = abs(positions[i, 0] - positions[j, 0])
        dy = abs(positions[i, 1] - positions[j, 1])
        min_dx = (widths[i] + widths[j]) / 2.0
        min_dy = (heights[i] + heights[j]) / 2.0

        ox = max(0, min_dx - dx)
        oy = max(0, min_dy - dy)
        if ox > 0 and oy > 0:
            overlaps.append((i, j, ox * oy))

    return overlaps


def compute_forces(positions, widths, heights, overlaps, anchor_positions,
                   k_anchor):
    """Compute repulsion + anchor forces.

    Repulsion: push overlapping cells apart (proportional to overlap).
    Anchor: pull cells toward their original positions (prevents drift).
    """
    N = len(positions)
    forces = np.zeros((N, 2))

    # Repulsion from overlaps
    for i, j, area in overlaps:
        dp = positions[i] - positions[j]

        # Per-axis overlap resolution
        dx = abs(dp[0])
        dy = abs(dp[1])
        min_dx = (widths[i] + widths[j]) / 2.0
        min_dy = (heights[i] + heights[j]) / 2.0

        # Choose axis with smaller overlap to resolve (like real legalizers)
        ox = max(0, min_dx - dx)
        oy = max(0, min_dy - dy)

        if ox < oy:
            # Push in x
            push = np.array([ox * np.sign(dp[0] + 1e-12), 0.0])
        else:
            # Push in y
            push = np.array([0.0, oy * np.sign(dp[1] + 1e-12)])

        forces[i] += push / 2.0
        forces[j] -= push / 2.0

    # Anchor toward original position
    if k_anchor > 0 and anchor_positions is not None:
        anchor_force = k_anchor * (anchor_positions - positions)
        forces += anchor_force

    return forces


def dykstra_projection(positions, widths, heights, overlaps, n_rounds=5):
    """Iterative constraint projection (Dykstra-style).

    For each overlap, project both cells to resolve it. Iterating over
    all overlaps multiple rounds mimics the CBF-QP iterative projector
    from sheaf-swarm. When η > 0, this OSCILLATES because constraints
    are contradictory — resolving one overlap creates another.

    Returns the total displacement for each cell.
    """
    N = len(positions)
    pos = positions.copy()

    for _round in range(n_rounds):
        for i, j, area in overlaps:
            dp = pos[i] - pos[j]
            dx = abs(dp[0])
            dy = abs(dp[1])
            min_dx = (widths[i] + widths[j]) / 2.0
            min_dy = (heights[i] + heights[j]) / 2.0

            ox = max(0, min_dx - dx)
            oy = max(0, min_dy - dy)

            if ox <= 0 and oy <= 0:
                continue  # already resolved

            # Project: move each cell by half the minimum separation
            if ox < oy and ox > 0:
                shift = ox / 2.0 * np.sign(dp[0] + 1e-12)
                pos[i, 0] += shift
                pos[j, 0] -= shift
            elif oy > 0:
                shift = oy / 2.0 * np.sign(dp[1] + 1e-12)
                pos[i, 1] += shift
                pos[j, 1] -= shift

    return pos - positions  # total displacement


# ═══════════════════════════════════════════════════════════════════
# Per-G-cell η and σ_min Computation
# ═══════════════════════════════════════════════════════════════════

def compute_gcell_metrics(positions, widths, heights, die_area, gs, r_interact):
    """Compute per-G-cell η_α, σ_min, Δ̄, and shield gains.

    Returns:
        gains: array (gs, gs) of per-G-cell gain factors
        eta_map: array (gs, gs) of η values
        sigma_min_map: array (gs, gs) of σ_min values
        dbar_map: array (gs, gs) of average degrees
        gcell_assign: dict mapping cell index → (gx, gy)
        ncells_map: array (gs, gs) of cell counts
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

    # Assign cells to G-cells
    gcell_cells = defaultdict(list)
    gcell_assign = {}
    for idx in range(N):
        gx = min(int((positions[idx, 0] - x_min) / gcell_w), gs - 1)
        gy = min(int((positions[idx, 1] - y_min) / gcell_h), gs - 1)
        gx, gy = max(0, gx), max(0, gy)
        gcell_cells[(gx, gy)].append(idx)
        gcell_assign[idx] = (gx, gy)

    # Build edge set using overlap-scale radius
    tree = KDTree(positions)
    global_pairs = tree.query_pairs(r=r_interact)
    global_edges = [(min(i, j), max(i, j)) for i, j in global_pairs]

    eta_map = np.zeros((gs, gs))
    sigma_min_map = np.full((gs, gs), np.inf)
    dbar_map = np.zeros((gs, gs))
    ncells_map = np.zeros((gs, gs), dtype=int)
    gains = np.ones((gs, gs))

    for gx in range(gs):
        for gy in range(gs):
            cell_idxs = gcell_cells.get((gx, gy), [])
            nc = len(cell_idxs)
            ncells_map[gy, gx] = nc
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
            dbar_map[gy, gx] = dbar_local

            local_pos = positions[sorted(idx_set)]

            # SVD for exact η and σ_min
            if nc <= 400 and nE <= 3000:
                delta = build_overlap_coboundary(local_pos, local_edges, n_v=2)
                sv = svd(delta, compute_uv=False)
                rank = int(np.sum(sv > 1e-10))
                eta_alpha = (nE - rank) / nE if nE > 0 else 0.0
                sigma_min_val = float(sv[rank - 1]) if rank > 0 else 0.0
            else:
                eta_alpha = theory_eta(dbar_local, 2)
                sigma_min_val = 1.0

            eta_map[gy, gx] = eta_alpha
            sigma_min_map[gy, gx] = sigma_min_val
            gains[gy, gx] = max(ETA_GAIN_FLOOR, 1.0 - eta_alpha)

    return gains, eta_map, sigma_min_map, dbar_map, gcell_assign, ncells_map


# ═══════════════════════════════════════════════════════════════════
# Placement Iteration
# ═══════════════════════════════════════════════════════════════════

def iterate_placement(positions, widths, heights, die_area, anchor_positions,
                      mode, n_iters, gs, r_interact, step_size, k_anchor):
    """Run placement legalization with a given shielding mode.

    Uses Dykstra-style constraint projection (like sheaf-swarm CBF-QP):
    each iteration projects over all overlap constraints multiple rounds.
    When η > 0, the projector oscillates — the shield dampens this.

    Returns trajectory dict with per-iteration metrics, and final positions.
    """
    pos = positions.copy()
    N = len(pos)

    traj = {
        "mode": mode,
        "overlap_counts": [],
        "overlap_areas": [],
        "eta_max": [],
        "eta_congested_frac": [],
        "sigma_min_global": [],
        "gain_min": [],
        "gain_mean": [],
        "displacement_rms": [],
    }

    gains = sigma_min_map = gcell_assign = ncells_map = None

    for it in range(n_iters):
        # 1. Detect overlaps
        overlaps = compute_overlaps(pos, widths, heights)
        n_overlaps = len(overlaps)
        total_area = sum(a for _, _, a in overlaps)
        traj["overlap_counts"].append(n_overlaps)
        traj["overlap_areas"].append(float(total_area))

        if n_overlaps == 0:
            for _ in range(n_iters - it - 1):
                traj["overlap_counts"].append(0)
                traj["overlap_areas"].append(0.0)
                traj["eta_max"].append(0.0)
                traj["eta_congested_frac"].append(0.0)
                traj["sigma_min_global"].append(float('inf'))
                traj["gain_min"].append(1.0)
                traj["gain_mean"].append(1.0)
                traj["displacement_rms"].append(0.0)
            break

        # 2. Dykstra projection: iterate over all constraints
        proj_disp = dykstra_projection(pos, widths, heights, overlaps,
                                        n_rounds=5)

        # Add anchor force
        if k_anchor > 0 and anchor_positions is not None:
            proj_disp += k_anchor * (anchor_positions - pos)

        # 3. Compute per-G-cell metrics (every few iters)
        if it % 3 == 0 or it < 5 or gains is None:
            gains, eta_map, sigma_min_map, dbar_map, gcell_assign, ncells_map = \
                compute_gcell_metrics(pos, widths, heights, die_area, gs,
                                      r_interact)

        eta_max = float(np.max(eta_map))
        n_congested = int(np.sum(eta_map > 0))
        n_active = int(np.sum(ncells_map >= 2))
        congested_frac = n_congested / max(1, n_active)

        finite_sigmas = sigma_min_map[np.isfinite(sigma_min_map)]
        sigma_min_g = float(np.min(finite_sigmas)) \
            if len(finite_sigmas) > 0 else float('inf')

        traj["eta_max"].append(eta_max)
        traj["eta_congested_frac"].append(float(congested_frac))
        traj["sigma_min_global"].append(sigma_min_g)

        # 4. Per-cell gain
        per_cell_gain = np.ones(N)
        if mode == "eta_shield":
            # SVD-measured η (includes grid structure artifacts)
            for idx in range(N):
                gx, gy = gcell_assign.get(idx, (0, 0))
                per_cell_gain[idx] = gains[gy, gx]

        elif mode == "theory_shield":
            # Theoretical η = max(0, 1-4/Δ̄) — pure density, no grid artifacts
            # This is the correct analog of sheaf-swarm Theorem 11:
            # grid structure (Theorem G'') is physics, not congestion.
            for idx in range(N):
                gx, gy = gcell_assign.get(idx, (0, 0))
                dbar_local = dbar_map[gy, gx]
                eta_th = theory_eta(dbar_local, 2)  # n_v=2 generic
                per_cell_gain[idx] = max(ETA_GAIN_FLOOR, 1.0 - eta_th)

        elif mode == "sigma_shield":
            for idx in range(N):
                gx, gy = gcell_assign.get(idx, (0, 0))
                sm = sigma_min_map[gy, gx]
                if np.isfinite(sm):
                    per_cell_gain[idx] = max(ETA_GAIN_FLOOR,
                                             min(1.0, sm / SIGMA_THRESHOLD))

        traj["gain_min"].append(float(np.min(per_cell_gain)))
        traj["gain_mean"].append(float(np.mean(per_cell_gain)))

        # 5. Apply shielded displacement
        displacement = proj_disp * step_size * per_cell_gain[:, np.newaxis]
        pos += displacement

        # Clamp to die area
        if die_area is not None:
            pos[:, 0] = np.clip(pos[:, 0], die_area["x_min"],
                                die_area["x_max"])
            pos[:, 1] = np.clip(pos[:, 1], die_area["y_min"],
                                die_area["y_max"])

        disp_rms = float(np.sqrt(np.mean(displacement**2)))
        traj["displacement_rms"].append(disp_rms)

        if it % 10 == 0 or it == n_iters - 1:
            g_min, g_mean = traj["gain_min"][-1], traj["gain_mean"][-1]
            print(f"  [{mode:>13s}] it {it:3d}: "
                  f"ov={n_overlaps:4d} area={total_area:8.2f} "
                  f"η_max={eta_max:.3f} cong={congested_frac:.0%} "
                  f"g=[{g_min:.2f},{g_mean:.2f}] rms={disp_rms:.4f}")

    traj["final_overlaps"] = traj["overlap_counts"][-1]
    traj["converged_iter"] = next(
        (i for i, c in enumerate(traj["overlap_counts"]) if c == 0),
        n_iters
    )

    return traj, pos


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_comparison(results, design_name, results_dir):
    """Plot A/B comparison of placement modes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"η-Shield Placement A/B: {design_name}", fontsize=14)

    colors = {"standard": "#d62728", "eta_shield": "#2ca02c",
              "theory_shield": "#ff7f0e", "sigma_shield": "#1f77b4"}
    labels = {"standard": "Standard", "eta_shield": "η-Shield (SVD)",
              "theory_shield": "η-Shield (Thm PS1)",
              "sigma_shield": "σ-Shield"}

    # (a) Overlap count
    ax = axes[0, 0]
    for mode, traj in results.items():
        c = traj["overlap_counts"]
        lbl = f'{labels[mode]} (→{traj["final_overlaps"]})'
        ax.plot(range(len(c)), c, color=colors[mode], label=lbl, linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Overlap count")
    ax.set_title("(a) Overlap count convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Overlap area
    ax = axes[0, 1]
    for mode, traj in results.items():
        a = traj["overlap_areas"]
        ax.plot(range(len(a)), a, color=colors[mode], label=labels[mode], lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Total overlap area (μm²)")
    ax.set_title("(b) Overlap area convergence")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) η_max trajectory
    ax = axes[0, 2]
    for mode, traj in results.items():
        e = traj["eta_max"]
        ax.plot(range(len(e)), e, color=colors[mode], label=labels[mode], lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("η_max")
    ax.set_title("(c) Worst G-cell η")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.0)

    # (d) Congested fraction
    ax = axes[1, 0]
    for mode, traj in results.items():
        cf = traj["eta_congested_frac"]
        ax.plot(range(len(cf)), cf, color=colors[mode], label=labels[mode], lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Fraction of G-cells with η>0")
    ax.set_title("(d) Congested G-cell fraction")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (e) Gain statistics
    ax = axes[1, 1]
    for mode, traj in results.items():
        if mode == "standard":
            continue
        ax.fill_between(range(len(traj["gain_min"])),
                        traj["gain_min"], traj["gain_mean"],
                        alpha=0.2, color=colors[mode])
        ax.plot(traj["gain_min"], color=colors[mode], ls="--", alpha=0.7,
                label=f'{labels[mode]} min')
        ax.plot(traj["gain_mean"], color=colors[mode], lw=2,
                label=f'{labels[mode]} mean')
    ax.axhline(y=ETA_GAIN_FLOOR, color="gray", ls=":", alpha=0.5,
               label=f"Floor ε={ETA_GAIN_FLOOR}")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Gain factor")
    ax.set_title("(e) Shield gain")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.1)

    # (f) Displacement RMS
    ax = axes[1, 2]
    for mode, traj in results.items():
        d = traj["displacement_rms"]
        ax.plot(range(len(d)), d, color=colors[mode], label=labels[mode], lw=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("RMS displacement (μm)")
    ax.set_title("(f) Cell movement magnitude")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(results_dir, f"eta_shield_{design_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved figure: {fig_path}")


# ═══════════════════════════════════════════════════════════════════
# Main Experiment
# ═══════════════════════════════════════════════════════════════════

def run_experiment(design_name, n_iters=N_ITERS, gs=GCELL_GRID,
                   step_size=STEP_SIZE, k_anchor=K_ANCHOR,
                   compress_factor=0.4, compress_region="quadrant",
                   modes=None, results_dir=None):
    """Run A/B experiment: compress one region, then legalize."""
    if modes is None:
        modes = ["standard", "eta_shield", "sigma_shield"]
    if results_dir is None:
        results_dir = os.path.join(os.path.dirname(__file__), "results", "shield")
    os.makedirs(results_dir, exist_ok=True)

    # Load design
    positions, widths, heights, die_area, label = load_design(design_name)
    if positions is None:
        return None

    N = len(positions)
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)

    # Interaction radius: ~ cell diagonal (overlap scale, not 3x)
    r_interact = diag * 1.5
    print(f"\n  Cells: {N}, median {med_w:.3f}×{med_h:.3f} μm, diag={diag:.3f}")
    print(f"  r_interact = {r_interact:.3f} μm (overlap scale)")

    # Save original positions as anchor
    anchor_positions = positions.copy()

    # Create congested placement
    congested_pos, compress_mask, n_compressed = create_congested_placement(
        positions, widths, heights, die_area,
        compress_region=compress_region,
        compress_factor=compress_factor,
    )
    overlaps_init = compute_overlaps(congested_pos, widths, heights)
    print(f"\n  Compression: {compress_region}, factor={compress_factor}")
    print(f"  Compressed {n_compressed}/{N} cells → {len(overlaps_init)} overlaps")

    # Initial η snapshot
    gains0, eta0, sigma0, dbar0, _, ncells0 = compute_gcell_metrics(
        congested_pos, widths, heights, die_area, gs, r_interact
    )
    n_congested0 = int(np.sum(eta0 > 0))
    n_active0 = int(np.sum(ncells0 >= 2))
    print(f"  Initial η: max={np.max(eta0):.3f}, "
          f"congested G-cells={n_congested0}/{n_active0}")
    print(f"  Initial Δ̄: max={np.max(dbar0):.1f}, "
          f"mean(active)={np.mean(dbar0[ncells0>=2]):.1f}")

    # Run each mode from the SAME initial positions
    results = {}
    for mode in modes:
        print(f"\n{'─'*60}")
        print(f"  Mode: {mode}")
        print(f"{'─'*60}")
        traj, final_pos = iterate_placement(
            congested_pos, widths, heights, die_area, anchor_positions,
            mode, n_iters, gs, r_interact, step_size, k_anchor,
        )
        results[mode] = traj

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {design_name} "
          f"(compress={compress_region}@{compress_factor}, "
          f"N={N}, gs={gs})")
    print(f"{'='*70}")
    print(f"  {'Mode':>15s}  {'Init ov':>8s}  {'Final ov':>8s}  "
          f"{'Resolved':>8s}  {'Conv@':>6s}  "
          f"{'η_max₀':>7s}  {'η_max_f':>7s}")
    print(f"  {'─'*65}")

    summary = {
        "design": design_name, "N": N, "gs": gs,
        "compress_region": compress_region,
        "compress_factor": compress_factor,
        "r_interact": float(r_interact),
        "n_compressed": n_compressed,
        "initial_overlaps": len(overlaps_init),
        "modes": {},
    }

    for mode, traj in results.items():
        init_ov = traj["overlap_counts"][0]
        final_ov = traj["final_overlaps"]
        resolved = init_ov - final_ov
        conv = traj["converged_iter"]
        eta_init = traj["eta_max"][0] if traj["eta_max"] else 0
        eta_final = traj["eta_max"][-1] if traj["eta_max"] else 0

        conv_str = str(conv) if conv < n_iters else "—"
        print(f"  {mode:>15s}  {init_ov:>8d}  {final_ov:>8d}  "
              f"{resolved:>8d}  {conv_str:>6s}  "
              f"{eta_init:>7.3f}  {eta_final:>7.3f}")

        summary["modes"][mode] = {
            "final_overlaps": final_ov,
            "converged_iter": conv,
            "resolved": resolved,
            "resolution_rate": resolved / max(1, init_ov),
            "eta_max_init": float(eta_init),
            "eta_max_final": float(eta_final),
            "overlap_counts": traj["overlap_counts"],
            "overlap_areas": traj["overlap_areas"],
            "eta_max": traj["eta_max"],
            "eta_congested_frac": traj["eta_congested_frac"],
            "gain_min": traj["gain_min"],
            "gain_mean": traj["gain_mean"],
            "displacement_rms": traj["displacement_rms"],
        }

    # Save
    json_path = os.path.join(results_dir, f"eta_shield_{design_name}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {json_path}")

    plot_comparison(results, design_name, results_dir)

    return summary


def main():
    parser = argparse.ArgumentParser(description="η-Shield Placement A/B Test")
    parser.add_argument("--design", type=str, default="gcd_nangate45",
                        choices=list(DESIGNS.keys()))
    parser.add_argument("--iters", type=int, default=N_ITERS)
    parser.add_argument("--grid", type=int, default=GCELL_GRID)
    parser.add_argument("--step", type=float, default=STEP_SIZE)
    parser.add_argument("--anchor", type=float, default=K_ANCHOR)
    parser.add_argument("--compress", type=float, default=0.4,
                        help="Compression factor (smaller = more congested)")
    parser.add_argument("--region", type=str, default="quadrant",
                        choices=["quadrant", "center", "stripe"])
    parser.add_argument("--modes", type=str,
                        default="standard,eta_shield,theory_shield")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    modes = args.modes.split(",")

    if args.all:
        designs = ["gcd_nangate45", "gcd_sky130", "aes_nangate45", "gcd_asap7"]
    else:
        designs = [args.design]

    all_results = {}
    for design in designs:
        result = run_experiment(
            design, n_iters=args.iters, gs=args.grid,
            step_size=args.step, k_anchor=args.anchor,
            compress_factor=args.compress, compress_region=args.region,
            modes=modes,
        )
        if result is not None:
            all_results[design] = result

    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-DESIGN SUMMARY")
        print(f"{'='*70}")
        print(f"  {'Design':>25s}  {'Std final':>10s}  {'η-sh final':>10s}  "
              f"{'Δ overlaps':>10s}  {'η adv.':>8s}")
        print(f"  {'─'*70}")
        for design, result in all_results.items():
            std_f = result["modes"].get("standard", {}).get("final_overlaps", -1)
            eta_f = result["modes"].get("eta_shield", {}).get("final_overlaps", -1)
            delta_ov = std_f - eta_f
            adv = f"{delta_ov:+d}" if delta_ov != 0 else "="
            print(f"  {design:>25s}  {std_f:>10d}  {eta_f:>10d}  "
                  f"{delta_ov:>10d}  {adv:>8s}")


if __name__ == "__main__":
    main()
