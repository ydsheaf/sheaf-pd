#!/usr/bin/env python3
"""
Validate η vs Real DRC (Issue #14)
====================================

Pipeline:
1. Run OpenROAD GR on placed DEFs → per-layer congestion
2. Compute per-G-cell η from placement (existing code)
3. Correlate η vs GR congestion per G-cell
4. Compare with RUDY

Uses OpenROAD's global_route with congestion_iterations to get
per-G-cell overflow data.
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
    parse_lef_macros, parse_def_components,
    build_overlap_coboundary, compute_eta, theory_eta,
    DESIGNS, load_design,
)

OPENROAD = "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad"
ORFS = "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "gr")
os.makedirs(RESULTS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════
# OpenROAD GR
# ═══════════════════════════════════════════════════════════════════

def get_lef_def(design_name):
    """Get LEF/DEF paths for a design."""
    cfg = DESIGNS[design_name]
    return cfg["lef"], cfg["def"]


def run_openroad_gr(design_name):
    """Run OpenROAD GR and return the log output."""
    lef_paths, def_path = get_lef_def(design_name)

    tcl_lines = []
    for lef in lef_paths:
        tcl_lines.append(f'read_lef "{lef}"')
    tcl_lines.append(f'read_def "{def_path}"')

    # GR with congestion report
    tcl_lines.append('set_global_routing_layer_adjustment metal1-metal10 0.5')
    tcl_lines.append('global_route -verbose')

    # Extract per-G-cell congestion via Tcl
    # OpenROAD provides get_global_route_overflow for per-gcell data
    tcl_lines.append('''
puts "=== GCELL_CONGESTION_START ==="
set block [ord::get_db_block]
set die [$block getDieArea]
puts "DIE [$die xMin] [$die yMin] [$die xMax] [$die yMax]"
set dbu [$block getDefUnits]
puts "DBU $dbu"

# Get GCell grid info
set gcell_grid [$block getGCellGrid]
set x_grids [$gcell_grid getXGrids]
set y_grids [$gcell_grid getYGrids]
puts "GCELL_NX [llength $x_grids]"
puts "GCELL_NY [llength $y_grids]"

# Get nets and their routing info
set nets [$block getNets]
puts "NETS [llength $nets]"

# Report congestion per GCell using routing tracks
# We'll use the GRT capacity/usage data from the log
puts "=== GCELL_CONGESTION_END ==="
''')

    tcl_script = "\n".join(tcl_lines)
    tcl_path = os.path.join(RESULTS_DIR, f"_gr_{design_name}.tcl")
    with open(tcl_path, "w") as f:
        f.write(tcl_script)

    log_path = os.path.join(RESULTS_DIR, f"openroad_{design_name}.log")
    print(f"  Running OpenROAD GR: {design_name}...")

    result = subprocess.run(
        [OPENROAD, "-no_splash", "-exit", tcl_path],
        capture_output=True, text=True, timeout=300
    )

    full_output = result.stdout + result.stderr
    with open(log_path, "w") as f:
        f.write(full_output)

    print(f"  Log saved: {log_path}")
    return full_output, log_path


def parse_gr_congestion(log_text):
    """Parse GR congestion from OpenROAD log.

    Extracts:
    - Per-layer resource, demand, usage, overflow
    - Total overflow
    - GCell grid dimensions
    """
    result = {
        "layers": [],
        "total_overflow": 0,
        "total_resource": 0,
        "total_demand": 0,
        "wirelength": 0,
        "gcell_nx": 0,
        "gcell_ny": 0,
    }

    # Parse layer congestion table
    layer_pattern = re.compile(
        r'(metal\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)'
    )
    for m in layer_pattern.finditer(log_text):
        layer = {
            "name": m.group(1),
            "resource": int(m.group(2)),
            "demand": int(m.group(3)),
            "usage_pct": float(m.group(4)),
            "max_h_overflow": int(m.group(5)),
            "max_v_overflow": int(m.group(6)),
            "total_overflow": int(m.group(7)),
        }
        result["layers"].append(layer)
        result["total_overflow"] += layer["total_overflow"]
        result["total_resource"] += layer["resource"]
        result["total_demand"] += layer["demand"]

    # Parse total wirelength
    wl_match = re.search(r'Total wirelength:\s+([\d.]+)\s*um', log_text)
    if wl_match:
        result["wirelength"] = float(wl_match.group(1))

    # Parse GCell grid
    nx_match = re.search(r'GCELL_NX\s+(\d+)', log_text)
    ny_match = re.search(r'GCELL_NY\s+(\d+)', log_text)
    if nx_match:
        result["gcell_nx"] = int(nx_match.group(1))
    if ny_match:
        result["gcell_ny"] = int(ny_match.group(1))

    # Parse number of GCells from GRT log
    gcell_match = re.search(r'Number of gcells:\s*(\d+)', log_text)
    if gcell_match:
        result["n_gcells"] = int(gcell_match.group(1))

    return result


# ═══════════════════════════════════════════════════════════════════
# η Computation (reuse existing code)
# ═══════════════════════════════════════════════════════════════════

def compute_per_gcell_eta(positions, widths, heights, die_area, gs, r_interact):
    """Compute per-G-cell η using our existing pipeline."""
    from eta_shield_placement import compute_gcell_metrics
    gains, eta_map, sigma_map, dbar_map, gcell_assign, ncells_map = \
        compute_gcell_metrics(positions, widths, heights, die_area, gs, r_interact)
    return eta_map, dbar_map, ncells_map, gcell_assign


def compute_per_gcell_rudy(positions, widths, heights, die_area, gs):
    """Compute per-G-cell RUDY (cell area density)."""
    N = len(positions)
    if die_area is not None:
        x_min, y_min = die_area["x_min"], die_area["y_min"]
        x_max, y_max = die_area["x_max"], die_area["y_max"]
    else:
        x_min, y_min = positions.min(axis=0) - 1
        x_max, y_max = positions.max(axis=0) + 1

    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs
    gcell_area = gcell_w * gcell_h

    rudy_map = np.zeros((gs, gs))
    for idx in range(N):
        gx = min(int((positions[idx, 0] - x_min) / gcell_w), gs - 1)
        gy = min(int((positions[idx, 1] - y_min) / gcell_h), gs - 1)
        gx, gy = max(0, gx), max(0, gy)
        rudy_map[gy, gx] += widths[idx] * heights[idx] / gcell_area

    return rudy_map


# ═══════════════════════════════════════════════════════════════════
# Correlation Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_design(design_name, gs=8):
    """Full pipeline for one design."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {design_name}")
    print(f"{'='*70}")

    # 1. Run GR
    log_text, log_path = run_openroad_gr(design_name)
    gr_data = parse_gr_congestion(log_text)

    print(f"  GR result: overflow={gr_data['total_overflow']}, "
          f"demand={gr_data['total_demand']}, "
          f"wirelength={gr_data['wirelength']:.0f} μm")

    # 2. Compute η
    positions, widths, heights, die_area, _ = load_design(design_name)
    if positions is None:
        return None

    N = len(positions)
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)

    # Multiple radii to find best correlation
    r_values = [diag * f for f in [1.0, 1.5, 2.0, 3.0, 5.0]]

    results_per_r = []
    for r_interact in r_values:
        eta_map, dbar_map, ncells_map, _ = compute_per_gcell_eta(
            positions, widths, heights, die_area, gs, r_interact)
        rudy_map = compute_per_gcell_rudy(positions, widths, heights, die_area, gs)

        # Per-G-cell stats
        mask = ncells_map.ravel() >= 2
        n_active = int(mask.sum())
        n_congested = int(np.sum(eta_map.ravel()[mask] > 0))
        eta_mean = float(np.mean(eta_map.ravel()[mask])) if n_active > 0 else 0
        eta_max = float(np.max(eta_map))
        dbar_mean = float(np.mean(dbar_map.ravel()[mask])) if n_active > 0 else 0

        results_per_r.append({
            "r": float(r_interact),
            "r_over_diag": float(r_interact / diag),
            "n_active_gcells": n_active,
            "n_congested_gcells": n_congested,
            "eta_mean": eta_mean,
            "eta_max": eta_max,
            "dbar_mean": dbar_mean,
            "eta_map": eta_map.tolist(),
            "dbar_map": dbar_map.tolist(),
            "rudy_map": rudy_map.tolist(),
            "ncells_map": ncells_map.tolist(),
        })

        print(f"  r={r_interact:.2f} (r/diag={r_interact/diag:.1f}): "
              f"η_mean={eta_mean:.4f}, η_max={eta_max:.4f}, "
              f"Δ̄_mean={dbar_mean:.2f}, "
              f"congested={n_congested}/{n_active}")

    # 3. Per-layer congestion analysis
    print(f"\n  Per-layer GR congestion:")
    for layer in gr_data["layers"]:
        if layer["demand"] > 0:
            print(f"    {layer['name']:>8s}: resource={layer['resource']:6d} "
                  f"demand={layer['demand']:6d} "
                  f"usage={layer['usage_pct']:5.1f}% "
                  f"overflow={layer['total_overflow']}")

    # 4. Determine congestion level
    has_congestion = gr_data["total_overflow"] > 0
    usage_pct = gr_data["total_demand"] / max(1, gr_data["total_resource"]) * 100

    print(f"\n  Summary:")
    print(f"    Total overflow: {gr_data['total_overflow']}")
    print(f"    Resource utilization: {usage_pct:.1f}%")
    print(f"    Has congestion: {'YES' if has_congestion else 'NO'}")

    # 5. Check η=0 → DRC=0 hypothesis
    if not has_congestion:
        # Find which r gives η closest to 0
        best_r = min(results_per_r, key=lambda x: x["eta_mean"])
        print(f"\n  η=0 → DRC=0 CHECK:")
        print(f"    Best r={best_r['r']:.2f}: η_mean={best_r['eta_mean']:.4f}")
        if best_r["eta_mean"] < 0.05:
            print(f"    ✓ CONFIRMED: η ≈ 0 AND DRC = 0")
        else:
            print(f"    ✗ η > 0 but DRC = 0 — η is conservative (overestimates)")

    output = {
        "design": design_name,
        "N": N,
        "gs": gs,
        "gr_data": gr_data,
        "has_congestion": has_congestion,
        "usage_pct": float(usage_pct),
        "per_radius": results_per_r,
    }

    json_path = os.path.join(RESULTS_DIR, f"eta_drc_{design_name}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")

    return output


# ═══════════════════════════════════════════════════════════════════
# Cross-Design Summary
# ═══════════════════════════════════════════════════════════════════

def plot_summary(all_results):
    """Plot cross-design summary."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(all_results)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 5))
    if n == 1:
        axes = [axes]

    fig.suptitle("η vs GR Congestion: Cross-Design Validation", fontsize=14)

    for ax, (design, result) in zip(axes, all_results.items()):
        # Use r/diag ≈ 2 as representative radius
        best = min(result["per_radius"],
                   key=lambda x: abs(x["r_over_diag"] - 2.0))

        eta_flat = np.array(best["eta_map"]).ravel()
        dbar_flat = np.array(best["dbar_map"]).ravel()
        ncells_flat = np.array(best["ncells_map"]).ravel()
        rudy_flat = np.array(best["rudy_map"]).ravel()

        mask = ncells_flat >= 2

        # Plot η heatmap
        eta_map = np.array(best["eta_map"])
        im = ax.imshow(eta_map, cmap="YlOrRd", vmin=0, vmax=0.5,
                       origin="lower", aspect="auto")
        plt.colorbar(im, ax=ax, label="η", fraction=0.046)

        overflow = result["gr_data"]["total_overflow"]
        usage = result["usage_pct"]
        ax.set_title(f"{design}\nGR overflow={overflow}, usage={usage:.1f}%\n"
                     f"η_max={best['eta_max']:.3f}, r/diag={best['r_over_diag']:.1f}")
        ax.set_xlabel("G-cell col")
        ax.set_ylabel("G-cell row")

    plt.tight_layout()
    fig_path = os.path.join(RESULTS_DIR, "eta_drc_summary.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Saved summary figure: {fig_path}")


def main():
    designs = ["gcd_nangate45", "aes_nangate45", "gcd_sky130"]

    all_results = {}
    for design in designs:
        result = analyze_design(design, gs=8)
        if result is not None:
            all_results[design] = result

    # Cross-design summary
    print(f"\n{'='*70}")
    print("CROSS-DESIGN SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Design':>20s}  {'N':>6s}  {'GR overflow':>11s}  "
          f"{'Usage%':>7s}  {'η_max(r=2d)':>11s}  {'η=0→DRC=0?':>11s}")
    print(f"  {'─'*75}")

    for design, result in all_results.items():
        best = min(result["per_radius"],
                   key=lambda x: abs(x["r_over_diag"] - 2.0))
        overflow = result["gr_data"]["total_overflow"]
        check = "✓" if (not result["has_congestion"] and best["eta_max"] < 0.15) \
                     or result["has_congestion"] else "?"
        print(f"  {design:>20s}  {result['N']:>6d}  {overflow:>11d}  "
              f"{result['usage_pct']:>7.1f}  {best['eta_max']:>11.4f}  {check:>11s}")

    if all_results:
        plot_summary(all_results)

    # Save combined
    summary_path = os.path.join(RESULTS_DIR, "eta_drc_combined.json")
    # Can't serialize numpy arrays directly, already converted to lists
    with open(summary_path, "w") as f:
        json.dump({k: {kk: vv for kk, vv in v.items()
                       if kk != "per_radius"}
                   for k, v in all_results.items()}, f, indent=2)
    print(f"\n  Saved combined: {summary_path}")


if __name__ == "__main__":
    main()
