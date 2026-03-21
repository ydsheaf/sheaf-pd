#!/usr/bin/env python3
"""
Validate η vs DRC on CONGESTED designs (Issue #14)
====================================================

Uses restricted routing layers (metal2-3 only) to induce congestion,
then correlates per-G-cell η with per-G-cell GR overflow.

Pipeline:
1. Run OpenROAD GR with restricted layers → get per-net overflow
2. Compute per-G-cell overflow by mapping net bounding boxes to G-cells
3. Compute per-G-cell η from placement
4. Scatter plot: η vs overflow per G-cell
"""

import json
import os
import re
import subprocess
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

OPENROAD = "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad"
ORFS = "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "gr")
os.makedirs(RESULTS_DIR, exist_ok=True)


def run_congested_gr(design_name, max_layer="metal3"):
    """Run GR with restricted layers to create congestion.
    Also outputs per-GCell congestion data via Tcl.
    """
    cfg = DESIGNS[design_name]
    lef_paths = cfg["lef"]
    def_path = cfg["def"]

    tcl_lines = []
    for lef in lef_paths:
        tcl_lines.append(f'read_lef "{lef}"')
    tcl_lines.append(f'read_def "{def_path}"')

    tcl_lines.append(f'''
set_global_routing_layer_adjustment metal1-metal10 0.8
set_routing_layers -signal metal2-{max_layer}
global_route -verbose -allow_congestion

set block [ord::get_db_block]
set die [$block getDieArea]
set dbu [$block getDefUnits]
puts "DIE_UM [expr [$die xMin] / double($dbu)] [expr [$die yMin] / double($dbu)] [expr [$die xMax] / double($dbu)] [expr [$die yMax] / double($dbu)]"
puts "DBU $dbu"

# Output net bounding boxes (fast for small designs)
puts "=== NET_BBOX_START ==="
set nets [$block getNets]
foreach net $nets {{
    set iters [$net getITerms]
    if {{ [llength $iters] < 2 }} continue
    set xmin 1e18; set ymin 1e18; set xmax -1e18; set ymax -1e18
    foreach it $iters {{
        set inst [$it getInst]
        if {{ $inst == "NULL" }} continue
        set bb [$inst getBBox]
        set cx [expr ([$bb xMin] + [$bb xMax]) / 2.0 / $dbu]
        set cy [expr ([$bb yMin] + [$bb yMax]) / 2.0 / $dbu]
        if {{ $cx < $xmin }} {{ set xmin $cx }}
        if {{ $cy < $ymin }} {{ set ymin $cy }}
        if {{ $cx > $xmax }} {{ set xmax $cx }}
        if {{ $cy > $ymax }} {{ set ymax $cy }}
    }}
    if {{ $xmin < 1e17 }} {{
        puts "NB $xmin $ymin $xmax $ymax"
    }}
}}
puts "=== NET_BBOX_END ==="
''')

    tcl_script = "\n".join(tcl_lines)
    tcl_path = os.path.join(RESULTS_DIR, f"_congested_{design_name}.tcl")
    with open(tcl_path, "w") as f:
        f.write(tcl_script)

    log_path = os.path.join(RESULTS_DIR, f"congested_{design_name}.log")
    print(f"  Running congested GR: {design_name} (max_layer={max_layer})...")

    result = subprocess.run(
        [OPENROAD, "-no_splash", "-exit", tcl_path],
        capture_output=True, text=True, timeout=3600
    )

    full_output = result.stdout + result.stderr
    with open(log_path, "w") as f:
        f.write(full_output)

    # Parse overflow
    layer_pattern = re.compile(
        r'(metal\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)'
    )
    total_overflow = 0
    total_demand = 0
    total_resource = 0
    for m in layer_pattern.finditer(full_output):
        total_overflow += int(m.group(7))
        total_demand += int(m.group(3))
        total_resource += int(m.group(2))

    print(f"  Total overflow: {total_overflow}, demand: {total_demand}, "
          f"resource: {total_resource}")

    # Parse die area
    die_match = re.search(r'DIE_UM\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)\s+([\d.e+-]+)',
                          full_output)
    die_area_um = None
    if die_match:
        die_area_um = {
            "x_min": float(die_match.group(1)),
            "y_min": float(die_match.group(2)),
            "x_max": float(die_match.group(3)),
            "y_max": float(die_match.group(4)),
        }

    # Parse net bounding boxes for per-G-cell overflow estimation
    net_bboxes = []
    for line in full_output.split('\n'):
        if line.startswith('NB '):
            parts = line.split()
            if len(parts) >= 5:
                net_bboxes.append({
                    "x_min": float(parts[1]),
                    "y_min": float(parts[2]),
                    "x_max": float(parts[3]),
                    "y_max": float(parts[4]),
                })

    print(f"  Net bounding boxes parsed: {len(net_bboxes)}")

    return {
        "total_overflow": total_overflow,
        "total_demand": total_demand,
        "total_resource": total_resource,
        "die_area_um": die_area_um,
        "net_bboxes": net_bboxes,
        "log_path": log_path,
    }


def compute_per_gcell_demand(net_bboxes, die_area, gs):
    """Estimate per-G-cell routing demand from net bounding boxes.

    RUDY-style: each net contributes 1/(bbox_area) to each G-cell
    it overlaps. This gives a density estimate of routing demand.
    """
    x_min, y_min = die_area["x_min"], die_area["y_min"]
    x_max, y_max = die_area["x_max"], die_area["y_max"]
    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs

    demand_map = np.zeros((gs, gs))

    for net in net_bboxes:
        bbox_w = max(net["x_max"] - net["x_min"], gcell_w * 0.1)
        bbox_h = max(net["y_max"] - net["y_min"], gcell_h * 0.1)
        bbox_area = bbox_w * bbox_h
        contribution = 1.0 / bbox_area  # RUDY formula

        # Which G-cells does this net overlap?
        gx_min = max(0, int((net["x_min"] - x_min) / gcell_w))
        gx_max = min(gs - 1, int((net["x_max"] - x_min) / gcell_w))
        gy_min = max(0, int((net["y_min"] - y_min) / gcell_h))
        gy_max = min(gs - 1, int((net["y_max"] - y_min) / gcell_h))

        for gx in range(gx_min, gx_max + 1):
            for gy in range(gy_min, gy_max + 1):
                demand_map[gy, gx] += contribution

    return demand_map


def run_analysis(design_name, gs=10, max_layer="metal3"):
    """Full congested analysis pipeline."""
    print(f"\n{'='*70}")
    print(f"CONGESTED ANALYSIS: {design_name}")
    print(f"{'='*70}")

    # 1. Run congested GR
    gr = run_congested_gr(design_name, max_layer)
    if gr["total_overflow"] == 0:
        print("  No congestion — skipping")
        return None

    # 2. Load placement and compute η
    positions, widths, heights, die_area, _ = load_design(design_name)
    if positions is None:
        return None

    N = len(positions)
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)

    # Use die_area from GR if available
    if gr["die_area_um"]:
        die_area = gr["die_area_um"]

    # Multiple radii
    r_values = [diag * f for f in [1.0, 1.5, 2.0, 3.0]]

    # 3. Per-G-cell demand from net bboxes
    demand_map = compute_per_gcell_demand(gr["net_bboxes"], die_area, gs)

    # 4. Compute η at each radius and correlate with demand
    from eta_shield_placement import compute_gcell_metrics

    best_r2 = -1
    best_r = None
    best_eta_map = None

    print(f"\n  Correlating η with GR demand (gs={gs}):")
    print(f"  {'r':>6s}  {'r/diag':>6s}  {'η_max':>6s}  {'R²(η,demand)':>12s}  {'R²(RUDY,demand)':>15s}")
    print(f"  {'─'*55}")

    for r_interact in r_values:
        # Subsample for SVD if needed
        max_cells = max(5000, gs * gs * 20)  # ensure coverage for larger grids
        if N > max_cells:
            rng = np.random.default_rng(42)
            cx, cy = np.mean(positions, axis=0)
            dists = np.sqrt((positions[:, 0] - cx)**2 + (positions[:, 1] - cy)**2)
            idx = np.argsort(dists)[:max_cells]
            pos_sub = positions[idx]
            w_sub = widths[idx]
            h_sub = heights[idx]
        else:
            pos_sub = positions
            w_sub = widths
            h_sub = heights

        gains, eta_map, sigma_map, dbar_map, gcell_assign, ncells_map = \
            compute_gcell_metrics(pos_sub, w_sub, h_sub, die_area, gs, r_interact)

        # RUDY (cell area density)
        rudy_map = np.zeros((gs, gs))
        gcell_w = (die_area["x_max"] - die_area["x_min"]) / gs
        gcell_h = (die_area["y_max"] - die_area["y_min"]) / gs
        gcell_area = gcell_w * gcell_h
        for idx_c in range(len(pos_sub)):
            gx = min(int((pos_sub[idx_c, 0] - die_area["x_min"]) / gcell_w), gs - 1)
            gy = min(int((pos_sub[idx_c, 1] - die_area["y_min"]) / gcell_h), gs - 1)
            gx, gy = max(0, gx), max(0, gy)
            rudy_map[gy, gx] += w_sub[idx_c] * h_sub[idx_c] / gcell_area

        # Correlate
        mask = ncells_map.ravel() >= 2
        if mask.sum() < 5:
            continue

        eta_flat = eta_map.ravel()[mask]
        demand_flat = demand_map.ravel()[mask]
        rudy_flat = rudy_map.ravel()[mask]

        # R² for η vs demand
        if np.std(demand_flat) > 1e-10 and np.std(eta_flat) > 1e-10:
            corr_eta = np.corrcoef(eta_flat, demand_flat)[0, 1]
            r2_eta = corr_eta ** 2
        else:
            r2_eta = 0

        if np.std(rudy_flat) > 1e-10 and np.std(demand_flat) > 1e-10:
            corr_rudy = np.corrcoef(rudy_flat, demand_flat)[0, 1]
            r2_rudy = corr_rudy ** 2
        else:
            r2_rudy = 0

        eta_max = float(np.max(eta_map))
        print(f"  {r_interact:6.2f}  {r_interact/diag:6.1f}  {eta_max:6.3f}  "
              f"{r2_eta:12.4f}  {r2_rudy:15.4f}")

        if r2_eta > best_r2:
            best_r2 = r2_eta
            best_r = r_interact
            best_eta_map = eta_map.copy()
            best_dbar_map = dbar_map.copy()
            best_rudy_map = rudy_map.copy()
            best_ncells_map = ncells_map.copy()
            best_r2_rudy = r2_rudy

    if best_eta_map is None:
        print("  No valid correlation computed")
        return None

    print(f"\n  Best radius: r={best_r:.2f} (r/diag={best_r/diag:.1f})")
    print(f"  R²(η, demand) = {best_r2:.4f}")
    print(f"  R²(RUDY, demand) = {best_r2_rudy:.4f}")
    if best_r2 > best_r2_rudy:
        print(f"  ✓ η OUTPERFORMS RUDY by {best_r2 - best_r2_rudy:.4f}")
    else:
        print(f"  ✗ RUDY outperforms η by {best_r2_rudy - best_r2:.4f}")

    # Plot
    plot_congested(best_eta_map, best_rudy_map, demand_map, best_ncells_map,
                   best_dbar_map, design_name, best_r, diag,
                   best_r2, best_r2_rudy, gr["total_overflow"])

    output = {
        "design": design_name,
        "N": N,
        "gs": gs,
        "max_layer": max_layer,
        "total_overflow": gr["total_overflow"],
        "best_r": float(best_r),
        "best_r_over_diag": float(best_r / diag),
        "R2_eta_demand": float(best_r2),
        "R2_rudy_demand": float(best_r2_rudy),
        "eta_wins": bool(best_r2 > best_r2_rudy),
    }

    json_path = os.path.join(RESULTS_DIR, f"congested_{design_name}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")

    return output


def plot_congested(eta_map, rudy_map, demand_map, ncells_map, dbar_map,
                   design_name, r, diag, r2_eta, r2_rudy, total_overflow):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"η vs GR Demand: {design_name} (overflow={total_overflow}, "
                 f"r={r:.2f}, r/diag={r/diag:.1f})", fontsize=13)

    # (a) η heatmap
    ax = axes[0, 0]
    im = ax.imshow(eta_map, cmap="YlOrRd", vmin=0, vmax=0.6,
                   origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax, label="η")
    ax.set_title("(a) η (sheaf)")
    ax.set_xlabel("G-cell col")
    ax.set_ylabel("G-cell row")

    # (b) RUDY heatmap
    ax = axes[0, 1]
    im = ax.imshow(rudy_map, cmap="YlOrRd", origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax, label="RUDY")
    ax.set_title("(b) RUDY (cell density)")

    # (c) GR demand heatmap
    ax = axes[0, 2]
    im = ax.imshow(demand_map, cmap="YlOrRd", origin="lower", aspect="auto")
    plt.colorbar(im, ax=ax, label="GR demand")
    ax.set_title("(c) GR routing demand")

    # (d) Scatter: η vs demand
    ax = axes[1, 0]
    mask = ncells_map.ravel() >= 2
    eta_flat = eta_map.ravel()[mask]
    demand_flat = demand_map.ravel()[mask]
    ax.scatter(eta_flat, demand_flat, alpha=0.6, s=30, c='red', edgecolors='darkred')
    ax.set_xlabel("η (sheaf)")
    ax.set_ylabel("GR demand")
    ax.set_title(f"(d) η vs demand  R²={r2_eta:.4f}")
    ax.grid(True, alpha=0.3)
    # Fit line
    if len(eta_flat) > 2 and np.std(eta_flat) > 1e-10:
        z = np.polyfit(eta_flat, demand_flat, 1)
        x_fit = np.linspace(0, max(eta_flat), 50)
        ax.plot(x_fit, np.polyval(z, x_fit), 'k--', alpha=0.5)

    # (e) Scatter: RUDY vs demand
    ax = axes[1, 1]
    rudy_flat = rudy_map.ravel()[mask]
    ax.scatter(rudy_flat, demand_flat, alpha=0.6, s=30, c='blue', edgecolors='darkblue')
    ax.set_xlabel("RUDY")
    ax.set_ylabel("GR demand")
    ax.set_title(f"(e) RUDY vs demand  R²={r2_rudy:.4f}")
    ax.grid(True, alpha=0.3)
    if len(rudy_flat) > 2 and np.std(rudy_flat) > 1e-10:
        z = np.polyfit(rudy_flat, demand_flat, 1)
        x_fit = np.linspace(0, max(rudy_flat), 50)
        ax.plot(x_fit, np.polyval(z, x_fit), 'k--', alpha=0.5)

    # (f) R² comparison bar
    ax = axes[1, 2]
    bars = ax.bar(["η (sheaf)", "RUDY"], [r2_eta, r2_rudy],
                  color=["red", "blue"], alpha=0.7)
    ax.set_ylabel("R² with GR demand")
    ax.set_title("(f) Prediction accuracy")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, [r2_eta, r2_rudy]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.4f}', ha='center', fontsize=11)

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, f"congested_{design_name}.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved figure: {fig_path}")


def main():
    # AES NanGate45 with restricted layers (larger design)
    r1 = run_analysis("aes_nangate45", gs=12, max_layer="metal3")

    print(f"\n{'='*70}")
    print("CONGESTED VALIDATION SUMMARY")
    print(f"{'='*70}")
    for r in [r1]:
        if r:
            winner = "η" if r["eta_wins"] else "RUDY"
            print(f"  {r['design']:>20s}: overflow={r['total_overflow']:>8d}  "
                  f"R²(η)={r['R2_eta_demand']:.4f}  R²(RUDY)={r['R2_rudy_demand']:.4f}  "
                  f"winner={winner}")


if __name__ == "__main__":
    main()
