#!/usr/bin/env python3
"""
Large-scale η=0 certificate sweep (Issue #22)
===============================================

For each ORFS design:
1. Run congested GR (restricted layers)
2. Extract per-GCell guide-based overflow
3. Compute per-G-cell η
4. Report certificate precision P(safe|η=0)

Automates the full pipeline for all available designs.
"""

import json
import os
import re
import subprocess
import sys
import numpy as np
from collections import defaultdict
from scipy.linalg import svd
from scipy.spatial import KDTree
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from run_batch import (
    parse_lef_macros, parse_def_components,
    build_overlap_coboundary, compute_eta, theory_eta,
    DESIGNS, load_design,
)
from eta_shield_placement import compute_gcell_metrics

OPENROAD = "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts/tools/install/OpenROAD/bin/openroad"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "sweep")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Layer names per PDK
LAYER_CONFIG = {
    "nangate45": {
        "restrict": "metal2-metal3",
        "adjust_layers": "metal1-metal10",
        "adjust_value": 0.8,
        "dbu": 2000,
    },
    "sky130hd": {
        "restrict": "met2-met3",
        "adjust_layers": "met1-met5",
        "adjust_value": 0.95,
        "dbu": 1000,
    },
    "asap7": {
        "restrict": "M2-M3",
        "adjust_layers": "M1-M9",
        "adjust_value": 0.8,
        "dbu": 1000,
    },
}


def run_congested_gr_and_extract(design_name, timeout=1800):
    """Run congested GR + extract per-GCell overflow. Returns path to JSON."""
    cfg = DESIGNS[design_name]
    pdk = cfg["pdk"]
    lcfg = LAYER_CONFIG.get(pdk)
    if lcfg is None:
        print(f"  Unknown PDK {pdk} — skipping")
        return None

    json_path = os.path.join(RESULTS_DIR, f"gcell_{design_name}.json")
    log_path = os.path.join(RESULTS_DIR, f"gr_{design_name}.log")

    # Check if already done
    if os.path.exists(json_path):
        print(f"  Already have: {json_path}")
        return json_path

    # Build Tcl script
    tcl_lines = []
    for lef in cfg["lef"]:
        tcl_lines.append(f'read_lef "{lef}"')
    tcl_lines.append(f'read_def "{cfg["def"]}"')

    tcl_lines.append(f'''
set_global_routing_layer_adjustment {lcfg["adjust_layers"]} {lcfg["adjust_value"]}
set_routing_layers -signal {lcfg["restrict"]}
global_route -verbose -allow_congestion
''')

    # Extract GCell grid + segments
    tcl_lines.append(f'''
set block [ord::get_db_block]
set dbu [$block getDefUnits]
set die [$block getDieArea]
set gcell_grid [$block getGCellGrid]
set x_grids [$gcell_grid getGridX]
set y_grids [$gcell_grid getGridY]
set nx [expr {{[llength $x_grids] - 1}}]
set ny [expr {{[llength $y_grids] - 1}}]

puts "GRID $nx $ny"
puts "DBU $dbu"
puts "DIE [$die xMin] [$die yMin] [$die xMax] [$die yMax]"

# Write segments for guide counting
set seg_file "{RESULTS_DIR}/seg_{design_name}.txt"
grt::write_segments $seg_file

# Output grid coordinates
puts "XGRIDS $x_grids"
puts "YGRIDS $y_grids"
''')

    tcl_path = os.path.join(RESULTS_DIR, f"_gr_{design_name}.tcl")
    with open(tcl_path, "w") as f:
        f.write("\n".join(tcl_lines))

    print(f"  Running GR for {design_name}...")
    try:
        result = subprocess.run(
            [OPENROAD, "-no_splash", "-exit", tcl_path],
            capture_output=True, text=True, timeout=timeout,
        )
        full_output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        print(f"  TIMEOUT after {timeout}s")
        return None

    with open(log_path, "w") as f:
        f.write(full_output)

    # Parse total overflow from GR report
    total_overflow = 0
    layer_pattern = re.compile(
        r'(?:metal|met|M)\d+\s+(\d+)\s+(\d+)\s+([\d.]+)%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)'
    )
    for m in layer_pattern.finditer(full_output):
        total_overflow += int(m.group(6))

    # Parse grid info
    grid_match = re.search(r'GRID\s+(\d+)\s+(\d+)', full_output)
    dbu_match = re.search(r'DBU\s+(\d+)', full_output)
    die_match = re.search(r'DIE\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)', full_output)
    xgrids_match = re.search(r'XGRIDS\s+(.+)', full_output)
    ygrids_match = re.search(r'YGRIDS\s+(.+)', full_output)

    if not all([grid_match, dbu_match, die_match, xgrids_match, ygrids_match]):
        print(f"  Failed to parse grid info")
        return None

    nx = int(grid_match.group(1))
    ny = int(grid_match.group(2))
    dbu = int(dbu_match.group(1))
    x_grids = [int(x) for x in xgrids_match.group(1).split()]
    y_grids = [int(y) for y in ygrids_match.group(1).split()]

    # Parse segments and compute per-GCell guide count
    seg_file = os.path.join(RESULTS_DIR, f"seg_{design_name}.txt")
    gcell_usage = defaultdict(int)

    if os.path.exists(seg_file):
        with open(seg_file) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 6:
                    try:
                        x1, y1 = int(parts[0]), int(parts[1])
                        x2, y2 = int(parts[3]), int(parts[4])
                        # Find which GCells this segment spans
                        for x_coord in [x1, x2]:
                            for y_coord in [y1, y2]:
                                gx = max(0, min(nx - 1,
                                    next((i for i in range(len(x_grids)-1)
                                          if x_grids[i+1] > x_coord), nx-1)))
                                gy = max(0, min(ny - 1,
                                    next((i for i in range(len(y_grids)-1)
                                          if y_grids[i+1] > y_coord), ny-1)))
                                gcell_usage[(gx, gy)] += 1
                    except (ValueError, StopIteration):
                        continue

    # Compute capacity per GCell (rough: tracks per pitch)
    gcell_cap = max(1, int(
        (x_grids[1] - x_grids[0]) / max(280, lcfg.get("min_pitch", 280)) * 0.5
    )) if len(x_grids) > 1 else 10

    # Build GCell list
    gcells = []
    for gy in range(ny):
        for gx in range(nx):
            usage = gcell_usage.get((gx, gy), 0)
            gcells.append({
                "gx": gx, "gy": gy,
                "usage": usage,
                "capacity": gcell_cap,
                "overflow": max(0, usage - gcell_cap),
            })

    output = {
        "design": design_name,
        "pdk": pdk,
        "grid_nx": nx, "grid_ny": ny,
        "x_grids": x_grids, "y_grids": y_grids,
        "dbu": dbu,
        "total_overflow": total_overflow,
        "gcells": gcells,
    }

    with open(json_path, "w") as f:
        json.dump(output, f)
    print(f"  Saved: {json_path} (overflow={total_overflow})")

    # Clean up segment file (can be large)
    if os.path.exists(seg_file):
        seg_size = os.path.getsize(seg_file)
        if seg_size > 10_000_000:
            os.remove(seg_file)
            print(f"  Removed large segment file ({seg_size//1000000}MB)")

    return json_path


def validate_certificate(design_name, gs=6):
    """Compute η and validate certificate against GR overflow."""
    positions, widths, heights, die_area, _ = load_design(design_name)
    if positions is None:
        return None

    N = len(positions)
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)

    cfg = DESIGNS[design_name]
    pdk = cfg["pdk"]
    dbu = LAYER_CONFIG.get(pdk, {}).get("dbu", 1000)

    # Load GCell overflow
    json_path = os.path.join(RESULTS_DIR, f"gcell_{design_name}.json")
    if not os.path.exists(json_path):
        return None
    gt = json.load(open(json_path))

    if gt["total_overflow"] == 0:
        # No congestion — can only confirm η=0 direction
        pass

    # Aggregate GT into our grid
    x_min, y_min = die_area["x_min"], die_area["y_min"]
    x_max, y_max = die_area["x_max"], die_area["y_max"]
    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs

    usage_grid = np.zeros((gs, gs))
    cap_grid = np.zeros((gs, gs))
    x_grids = gt['x_grids']
    y_grids = gt['y_grids']

    for gc in gt['gcells']:
        grt_x = (x_grids[gc['gx']] + x_grids[min(gc['gx']+1, gt['grid_nx'])]) / 2.0 / dbu
        grt_y = (y_grids[gc['gy']] + y_grids[min(gc['gy']+1, gt['grid_ny'])]) / 2.0 / dbu
        our_gx = min(max(0, int((grt_x - x_min) / gcell_w)), gs-1)
        our_gy = min(max(0, int((grt_y - y_min) / gcell_h)), gs-1)
        usage_grid[our_gy, our_gx] += gc['usage']
        cap_grid[our_gy, our_gx] += gc['capacity']

    has_overflow = (usage_grid > cap_grid).astype(int)

    # Compute η
    # Subsample for large designs
    if N > 1500:
        rng = np.random.default_rng(42)
        cx, cy = np.mean(positions, axis=0)
        dists = np.sqrt((positions[:,0]-cx)**2 + (positions[:,1]-cy)**2)
        idx = np.argsort(dists)[:1500]
        pos_sub, w_sub, h_sub = positions[idx], widths[idx], heights[idx]
    else:
        pos_sub, w_sub, h_sub = positions, widths, heights

    r = diag * 2.0
    _, eta_map, _, dbar_map, _, ncells_map = compute_gcell_metrics(
        pos_sub, w_sub, h_sub, die_area, gs, r)

    # Also cell density
    ncells_full = np.zeros((gs, gs), dtype=int)
    cell_density = np.zeros((gs, gs))
    for idx in range(N):
        gx = min(max(0, int((positions[idx,0]-x_min)/gcell_w)), gs-1)
        gy = min(max(0, int((positions[idx,1]-y_min)/gcell_h)), gs-1)
        ncells_full[gy,gx] += 1
        cell_density[gy,gx] += widths[idx]*heights[idx]/(gcell_w*gcell_h)

    # Active G-cells
    mask = ncells_full.ravel() >= 2
    n_active = int(mask.sum())

    y_true = has_overflow.ravel()[mask]
    n_overflow = int(y_true.sum())
    n_clean = n_active - n_overflow

    eta_pred = (eta_map.ravel()[mask] > 0).astype(int)
    dens_pred = (cell_density.ravel()[mask] > np.median(cell_density.ravel()[mask])).astype(int)

    result = {
        "design": design_name,
        "pdk": pdk,
        "N": N,
        "gs": gs,
        "r_over_diag": 2.0,
        "total_overflow": gt["total_overflow"],
        "n_active": n_active,
        "n_overflow": n_overflow,
        "n_clean": n_clean,
    }

    if n_overflow == 0 or n_clean == 0:
        # Can't compute binary metrics, but record η stats
        result["certificate"] = "N/A (no overflow)" if n_overflow == 0 else "N/A (all overflow)"
        result["eta_max"] = float(np.max(eta_map))
        result["eta_mean_active"] = float(np.mean(eta_map.ravel()[mask]))
        if n_overflow == 0 and np.max(eta_map.ravel()[mask]) == 0:
            result["certificate"] = "TRIVIAL (η=0 everywhere, no overflow)"
        return result

    # Compute metrics for η
    tn, fp, fn, tp = confusion_matrix(y_true, eta_pred).ravel()
    result["eta_cert_precision"] = float(tn / (tn + fn)) if (tn + fn) > 0 else None
    result["eta_recall"] = float(recall_score(y_true, eta_pred, zero_division=0))
    result["eta_f1"] = float(f1_score(y_true, eta_pred, zero_division=0))
    result["eta_specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    result["eta_accuracy"] = float((tp + tn) / len(y_true))

    # Density
    if dens_pred.sum() > 0 and dens_pred.sum() < len(dens_pred):
        tn_d, fp_d, fn_d, tp_d = confusion_matrix(y_true, dens_pred).ravel()
        result["dens_cert_precision"] = float(tn_d / (tn_d + fn_d)) if (tn_d + fn_d) > 0 else None
        result["dens_f1"] = float(f1_score(y_true, dens_pred, zero_division=0))

    return result


def main():
    designs = list(DESIGNS.keys())
    print(f"Sweeping {len(designs)} designs\n")

    all_results = []

    for design_name in designs:
        print(f"\n{'─'*60}")
        print(f"  {design_name}")
        print(f"{'─'*60}")

        # Step 1: Run GR + extract
        gt_path = run_congested_gr_and_extract(design_name, timeout=1800)
        if gt_path is None:
            print(f"  SKIP: GR failed")
            continue

        # Step 2: Validate certificate
        result = validate_certificate(design_name, gs=6)
        if result is None:
            print(f"  SKIP: validation failed")
            continue

        all_results.append(result)

        # Print result
        cert = result.get("eta_cert_precision")
        if cert is not None:
            print(f"  P(safe|η=0) = {cert:.3f}, "
                  f"recall = {result['eta_recall']:.3f}, "
                  f"F1 = {result['eta_f1']:.3f}")
        else:
            print(f"  Certificate: {result.get('certificate', 'N/A')}")

    # Summary table
    print(f"\n{'='*70}")
    print("η=0 ROUTABILITY CERTIFICATE — FULL SWEEP")
    print(f"{'='*70}")
    print(f"  {'Design':>25s}  {'PDK':>10s}  {'N':>6s}  {'Overflow':>8s}  "
          f"{'P(safe|η=0)':>11s}  {'Recall':>7s}  {'F1':>6s}")
    print(f"  {'─'*80}")

    for r in all_results:
        cert = r.get("eta_cert_precision")
        rec = r.get("eta_recall")
        f1 = r.get("eta_f1")
        cert_s = f"{cert:.3f}" if cert is not None else r.get("certificate", "—")[:15]
        rec_s = f"{rec:.3f}" if rec is not None else "—"
        f1_s = f"{f1:.3f}" if f1 is not None else "—"
        print(f"  {r['design']:>25s}  {r['pdk']:>10s}  {r['N']:>6d}  "
              f"{r['total_overflow']:>8d}  {cert_s:>11s}  {rec_s:>7s}  {f1_s:>6s}")

    # Save
    json_path = os.path.join(RESULTS_DIR, "certificate_sweep.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Saved: {json_path}")


if __name__ == "__main__":
    main()
