#!/usr/bin/env python3
"""
η-Guided Cell Migration (Issue #12, Phase 3)
=============================================

Simultaneous optimization of HPWL (reachability) and DRC (safety).

Like drones that fly to waypoints while avoiding collisions:
- Goal (reachability): minimize HPWL / wirelength
- Constraint (safety): no overlaps (η_α = 0 for all G-cells)
- These two objectives are ORTHOGONAL (not the same potential)

When η_α > 0 in a G-cell, the answer is NOT "slow down" (η-shield,
which we showed is marginal). The answer is "migrate cells OUT" —
reduce local density until Δ̄ < 4 and η → 0.

Architecture (matches "Resolving Conflicting Constraints" layered approach):

  Layer 1 (Diagnostic):  Per-G-cell η computation
                         Identifies which regions have unsatisfiable constraints

  Layer 2 (Migration):   Move cells from η>0 G-cells to η=0 G-cells
                         Target: reduce Δ̄_α to below threshold (=4)
                         Constraint: minimize HPWL increase

  Layer 3 (Legalization): Dykstra projection within each G-cell
                          Now η=0, so projection converges cleanly

This is the placement analog of:
  "MARL learns to proactively avoid conflict regions" (arXiv:2505.02293)

Usage:
  python experiments/eta_migration.py
  python experiments/eta_migration.py --design gcd_nangate45 --iters 30
"""

import argparse
import json
import os
import sys
import numpy as np
from collections import defaultdict
from scipy.spatial import KDTree
from scipy.linalg import svd

sys.path.insert(0, os.path.dirname(__file__))
from run_batch import (
    build_overlap_coboundary, theory_eta,
    DESIGNS, load_design,
)
from eta_shield_placement import (
    compute_overlaps, compute_gcell_metrics, dykstra_projection,
    create_congested_placement, ETA_GAIN_FLOOR,
)


# ═══════════════════════════════════════════════════════════════════
# HPWL Computation (proxy for reachability)
# ═══════════════════════════════════════════════════════════════════

def compute_hpwl(positions, anchor_positions):
    """Total displacement from anchor (proxy for HPWL degradation).

    In real placement, HPWL = Σ_net (max_x - min_x + max_y - min_y).
    Without net info, we use total displacement as proxy: moving cells
    further from their original position increases wirelength.
    """
    return float(np.sum(np.abs(positions - anchor_positions)))


def compute_hpwl_per_cell(positions, anchor_positions):
    """Per-cell displacement from anchor."""
    return np.sum(np.abs(positions - anchor_positions), axis=1)


# ═══════════════════════════════════════════════════════════════════
# Cell Migration Strategies
# ═══════════════════════════════════════════════════════════════════

def identify_migration_candidates(positions, widths, heights, die_area,
                                  anchor_positions, eta_map, dbar_map,
                                  ncells_map, gcell_assign, gs,
                                  max_migrations=None):
    """Identify cells to migrate and target G-cells.

    Donor G-cells: η_α > 0 (congested)
    Receiver G-cells: η_α = 0 AND have capacity (Δ̄ < 3)
    Candidates: cells in donor G-cells with highest HPWL cost
                (cells furthest from anchor → moving them costs less)

    Returns list of (cell_idx, source_gcell, target_gcell, priority).
    """
    if die_area is not None:
        x_min, y_min = die_area["x_min"], die_area["y_min"]
        x_max, y_max = die_area["x_max"], die_area["y_max"]
    else:
        margin = 1.0
        x_min, y_min = positions.min(axis=0) - margin
        x_max, y_max = positions.max(axis=0) + margin

    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs

    # Find donor and receiver G-cells
    donors = []    # (gx, gy, η_α, n_excess)
    receivers = [] # (gx, gy, capacity, center_x, center_y)

    for gx in range(gs):
        for gy in range(gs):
            eta_local = eta_map[gy, gx]
            dbar_local = dbar_map[gy, gx]
            nc = ncells_map[gy, gx]

            if eta_local > 0 and nc >= 3:
                # How many cells to remove to get Δ̄ < 4?
                # Δ̄ = 2|E|/|V| ≈ n·π·r²/A → need n_target = n · (4/Δ̄)
                n_target = max(2, int(nc * 4.0 / max(dbar_local, 0.1)))
                n_excess = nc - n_target
                if n_excess > 0:
                    donors.append((gx, gy, eta_local, n_excess))

            elif eta_local == 0 and dbar_local < 3.0:
                # Capacity = how many cells can this G-cell absorb?
                center_x = x_min + (gx + 0.5) * gcell_w
                center_y = y_min + (gy + 0.5) * gcell_h
                capacity = max(1, int((3.0 - dbar_local) / 0.5))
                receivers.append((gx, gy, capacity, center_x, center_y))

    if not donors or not receivers:
        return []

    # For each donor, find candidate cells to migrate
    migrations = []
    hpwl_per_cell = compute_hpwl_per_cell(positions, anchor_positions)

    # Invert gcell_assign: gcell → list of cell indices
    gcell_to_cells = defaultdict(list)
    for idx, (gx, gy) in gcell_assign.items():
        gcell_to_cells[(gx, gy)].append(idx)

    for gx_d, gy_d, eta_d, n_excess in sorted(donors, key=lambda x: -x[2]):
        cells_in_donor = gcell_to_cells.get((gx_d, gy_d), [])
        if not cells_in_donor:
            continue

        # Sort by HPWL: cells already far from anchor are cheapest to move
        cells_sorted = sorted(cells_in_donor,
                              key=lambda i: -hpwl_per_cell[i])

        for cell_idx in cells_sorted[:n_excess]:
            # Find nearest receiver with capacity
            best_receiver = None
            best_dist = float('inf')
            for gx_r, gy_r, cap, cx_r, cy_r in receivers:
                if cap <= 0:
                    continue
                dist = abs(positions[cell_idx, 0] - cx_r) + \
                       abs(positions[cell_idx, 1] - cy_r)
                if dist < best_dist:
                    best_dist = dist
                    best_receiver = (gx_r, gy_r, cx_r, cy_r)

            if best_receiver is not None:
                gx_r, gy_r, cx_r, cy_r = best_receiver
                priority = eta_d * (1.0 / (best_dist + 0.01))
                migrations.append({
                    "cell": cell_idx,
                    "source": (gx_d, gy_d),
                    "target": (gx_r, gy_r),
                    "target_center": (cx_r, cy_r),
                    "eta_source": float(eta_d),
                    "hpwl_cost": float(best_dist),
                    "priority": float(priority),
                })
                # Reduce receiver capacity
                for k, (gx, gy, cap, cx, cy) in enumerate(receivers):
                    if gx == gx_r and gy == gy_r:
                        receivers[k] = (gx, gy, cap - 1, cx, cy)
                        break

    # Sort by priority (highest first)
    migrations.sort(key=lambda m: -m["priority"])

    # Rate-limit: cap number of migrations per iteration
    N = len(positions)
    if max_migrations is None:
        max_migrations = max(1, N // 20)
    if len(migrations) > max_migrations:
        migrations = migrations[:max_migrations]

    return migrations


def execute_migration(positions, widths, heights, migrations,
                      migration_fraction=0.5):
    """Move cells toward target G-cell centers.

    Doesn't teleport — moves a fraction of the way each iteration.
    This allows the diagnostic (η) to update and guide further migrations.

    migration_fraction: how much of the way to move per iteration (0.5 = halfway)
    """
    pos = positions.copy()
    n_migrated = 0

    for m in migrations:
        idx = m["cell"]
        cx, cy = m["target_center"]
        target = np.array([cx, cy])

        # Move fraction of the way toward target center
        displacement = (target - pos[idx]) * migration_fraction
        pos[idx] += displacement
        n_migrated += 1

    return pos, n_migrated


# ═══════════════════════════════════════════════════════════════════
# Migration Loop
# ═══════════════════════════════════════════════════════════════════

def run_migration(positions, widths, heights, die_area, anchor_positions,
                  mode, n_iters, gs, r_interact, migration_fraction=0.3,
                  max_migrations=None):
    """Run iterative migration + legalization.

    Modes:
      - standard: Dykstra only, no migration
      - migrate: η-guided migration + Dykstra legalization
      - migrate_greedy: migrate most-overlapping cells (no η)
    """
    pos = positions.copy()
    N = len(pos)

    traj = {
        "mode": mode,
        "overlap_counts": [],
        "overlap_areas": [],
        "hpwl": [],
        "eta_max": [],
        "n_congested": [],
        "n_migrated": [],
    }

    for it in range(n_iters):
        # 1. Compute metrics
        overlaps = compute_overlaps(pos, widths, heights)
        n_overlaps = len(overlaps)
        total_area = sum(a for _, _, a in overlaps)
        hpwl = compute_hpwl(pos, anchor_positions)

        gains, eta_map, sigma_map, dbar_map, gcell_assign, ncells_map = \
            compute_gcell_metrics(pos, widths, heights, die_area, gs, r_interact)

        eta_max = float(np.max(eta_map))
        n_congested = int(np.sum(eta_map > 0))

        traj["overlap_counts"].append(n_overlaps)
        traj["overlap_areas"].append(float(total_area))
        traj["hpwl"].append(float(hpwl))
        traj["eta_max"].append(eta_max)
        traj["n_congested"].append(n_congested)

        if n_overlaps == 0:
            traj["n_migrated"].append(0)
            # Pad
            for _ in range(n_iters - it - 1):
                traj["overlap_counts"].append(0)
                traj["overlap_areas"].append(0.0)
                traj["hpwl"].append(float(hpwl))
                traj["eta_max"].append(0.0)
                traj["n_congested"].append(0)
                traj["n_migrated"].append(0)
            break

        # 2. Migration phase (if applicable)
        n_migrated = 0
        if mode == "migrate" and n_congested > 0:
            migrations = identify_migration_candidates(
                pos, widths, heights, die_area, anchor_positions,
                eta_map, dbar_map, ncells_map, gcell_assign, gs,
                max_migrations=max_migrations)
            if migrations:
                pos, n_migrated = execute_migration(
                    pos, widths, heights, migrations, migration_fraction)

        elif mode == "migrate_greedy":
            # Greedy: move most-overlapping cells toward less dense areas
            overlap_count = np.zeros(N)
            for i, j, _ in overlaps:
                overlap_count[i] += 1
                overlap_count[j] += 1
            worst_cells = np.argsort(-overlap_count)[:max(1, N // 20)]
            centroid = np.mean(pos, axis=0)
            for idx in worst_cells:
                # Push away from centroid (spread out)
                dp = pos[idx] - centroid
                dist = np.linalg.norm(dp) + 1e-10
                pos[idx] += dp / dist * migration_fraction * np.median(widths) * 2
                n_migrated += 1

        traj["n_migrated"].append(n_migrated)

        # 3. Legalization phase: Dykstra projection
        overlaps_now = compute_overlaps(pos, widths, heights)
        if overlaps_now:
            proj_disp = dykstra_projection(pos, widths, heights, overlaps_now,
                                            n_rounds=5)
            pos += proj_disp * 0.3

        # 4. Anchor force (gentle pull back toward original)
        pos += 0.03 * (anchor_positions - pos)

        # Clamp
        if die_area is not None:
            pos[:, 0] = np.clip(pos[:, 0], die_area["x_min"], die_area["x_max"])
            pos[:, 1] = np.clip(pos[:, 1], die_area["y_min"], die_area["y_max"])

        if it % 5 == 0 or it == n_iters - 1:
            print(f"  [{mode:>15s}] it {it:3d}: "
                  f"ov={n_overlaps:4d} area={total_area:8.2f} "
                  f"hpwl={hpwl:8.1f} η_max={eta_max:.3f} "
                  f"cong={n_congested:2d} migr={n_migrated:2d}")

    traj["final_overlaps"] = traj["overlap_counts"][-1]
    traj["final_hpwl"] = traj["hpwl"][-1]
    traj["converged_iter"] = next(
        (i for i, c in enumerate(traj["overlap_counts"]) if c == 0),
        n_iters
    )

    return traj, pos


# ═══════════════════════════════════════════════════════════════════
# Visualization
# ═══════════════════════════════════════════════════════════════════

def plot_migration(results, design_name, results_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"η-Guided Cell Migration: {design_name}", fontsize=14)

    colors = {"standard": "#d62728", "migrate": "#2ca02c",
              "migrate_greedy": "#1f77b4"}
    labels = {"standard": "Standard (Dykstra only)",
              "migrate": "η-guided migration",
              "migrate_greedy": "Greedy migration"}

    # (a) Overlap count
    ax = axes[0, 0]
    for mode, traj in results.items():
        ax.plot(traj["overlap_counts"], color=colors[mode],
                label=labels[mode], linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Overlap count")
    ax.set_title("(a) Overlap count")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) Overlap area
    ax = axes[0, 1]
    for mode, traj in results.items():
        ax.plot(traj["overlap_areas"], color=colors[mode],
                label=labels[mode], linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Overlap area (μm²)")
    ax.set_title("(b) Overlap area")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) HPWL (reachability cost)
    ax = axes[0, 2]
    for mode, traj in results.items():
        ax.plot(traj["hpwl"], color=colors[mode],
                label=labels[mode], linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("HPWL proxy (total displacement)")
    ax.set_title("(c) HPWL cost (reachability)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (d) η_max
    ax = axes[1, 0]
    for mode, traj in results.items():
        ax.plot(traj["eta_max"], color=colors[mode],
                label=labels[mode], linewidth=2)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("η_max")
    ax.set_title("(d) Worst G-cell η (safety)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.0)

    # (e) Safety-Reachability tradeoff
    ax = axes[1, 1]
    for mode, traj in results.items():
        ax.plot(traj["hpwl"], traj["overlap_areas"],
                color=colors[mode], label=labels[mode],
                linewidth=2, marker='o', markersize=2)
        # Mark start and end
        ax.plot(traj["hpwl"][0], traj["overlap_areas"][0],
                'o', color=colors[mode], markersize=10, alpha=0.5)
        ax.plot(traj["hpwl"][-1], traj["overlap_areas"][-1],
                's', color=colors[mode], markersize=10)
    ax.set_xlabel("HPWL (reachability cost) →")
    ax.set_ylabel("Overlap area (safety cost) →")
    ax.set_title("(e) Safety vs Reachability tradeoff")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (f) Migrations per iteration
    ax = axes[1, 2]
    for mode, traj in results.items():
        if mode != "standard":
            ax.bar(range(len(traj["n_migrated"])), traj["n_migrated"],
                   color=colors[mode], alpha=0.5, label=labels[mode])
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Cells migrated")
    ax.set_title("(f) Migration activity")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(results_dir, f"eta_migration_{design_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved figure: {fig_path}")


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description="η-Guided Cell Migration")
    parser.add_argument("--design", type=str, default="gcd_nangate45")
    parser.add_argument("--iters", type=int, default=40)
    parser.add_argument("--grid", type=int, default=6)
    parser.add_argument("--compress", type=float, default=0.3)
    parser.add_argument("--region", type=str, default="center")
    parser.add_argument("--migrate-frac", type=float, default=0.3)
    args = parser.parse_args()

    results_dir = os.path.join(os.path.dirname(__file__), "results", "migration")
    os.makedirs(results_dir, exist_ok=True)

    # Load
    positions, widths, heights, die_area, _ = load_design(args.design)
    if positions is None:
        return

    N = len(positions)
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)
    r_interact = diag * 1.5

    anchor_positions = positions.copy()

    # Create congested placement
    congested_pos, _, n_comp = create_congested_placement(
        positions, widths, heights, die_area,
        compress_region=args.region, compress_factor=args.compress)

    overlaps_init = compute_overlaps(congested_pos, widths, heights)
    hpwl_init = compute_hpwl(congested_pos, anchor_positions)
    print(f"\n  N={N}, compressed {n_comp} cells → {len(overlaps_init)} overlaps")
    print(f"  Initial HPWL proxy: {hpwl_init:.1f}")

    # Run modes
    modes = ["standard", "migrate", "migrate_greedy"]
    results = {}

    for mode in modes:
        print(f"\n{'─'*60}")
        print(f"  Mode: {mode}")
        print(f"{'─'*60}")
        traj, final_pos = run_migration(
            congested_pos, widths, heights, die_area, anchor_positions,
            mode, args.iters, args.grid, r_interact, args.migrate_frac)
        results[mode] = traj

    # Summary
    print(f"\n{'='*70}")
    print(f"RESULTS: {args.design}")
    print(f"{'='*70}")
    print(f"  {'Mode':>17s}  {'Overlaps':>9s}  {'Area':>8s}  "
          f"{'HPWL':>8s}  {'η_max':>6s}  {'Pareto?':>7s}")
    print(f"  {'─'*60}")

    std_ov = results["standard"]["final_overlaps"]
    std_hpwl = results["standard"]["final_hpwl"]

    for mode, traj in results.items():
        ov = traj["final_overlaps"]
        area = traj["overlap_areas"][-1]
        hpwl = traj["final_hpwl"]
        eta = traj["eta_max"][-1]

        # Pareto: better overlaps AND better or equal HPWL?
        pareto = "✓" if (ov < std_ov and hpwl <= std_hpwl * 1.1) else \
                 "≈" if mode == "standard" else "✗"

        print(f"  {mode:>17s}  {ov:>9d}  {area:>8.2f}  "
              f"{hpwl:>8.1f}  {eta:>6.3f}  {pareto:>7s}")

    # Save
    summary = {
        "design": args.design, "N": N,
        "compress": args.compress, "region": args.region,
        "initial_overlaps": len(overlaps_init),
        "initial_hpwl": float(hpwl_init),
        "modes": {mode: {
            "final_overlaps": traj["final_overlaps"],
            "final_hpwl": traj["final_hpwl"],
            "overlap_counts": traj["overlap_counts"],
            "overlap_areas": traj["overlap_areas"],
            "hpwl": traj["hpwl"],
            "eta_max": traj["eta_max"],
            "n_migrated": traj["n_migrated"],
        } for mode, traj in results.items()},
    }
    json_path = os.path.join(results_dir, f"eta_migration_{args.design}.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Saved: {json_path}")

    plot_migration(results, args.design, results_dir)


if __name__ == "__main__":
    main()
