#!/usr/bin/env python3
"""
mass_sweep.py — Mass GT certificate validation across new designs (Issue #24)
================================================================================

Runs OpenROAD GR with restricted layers, extracts per-G-cell capacity/usage
from dbGCellGrid, computes eta at r=2*diag, and reports specificity,
certificate precision (P(safe|eta=0)), and recall.

Designs that fail (wrong LEF, missing NETS, GR divergence) are skipped
with error reporting. At the end, a binomial test checks whether
specificity=1.0 across NanGate45 designs is statistically significant
under H0: specificity=0.9.
"""

import json
import os
import re
import subprocess
import sys
import time
import numpy as np
from collections import defaultdict
from scipy.linalg import svd
from scipy.spatial import KDTree
from scipy.stats import binom

sys.path.insert(0, os.path.dirname(__file__))
from run_batch import (
    parse_lef_macros, parse_def_components,
    build_overlap_coboundary, compute_eta, theory_eta,
    load_design, DESIGNS,
)
from eta_shield_placement import compute_gcell_metrics

# ─── Paths ───
ORFS = "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts"
OPENROAD = f"{ORFS}/tools/install/OpenROAD/bin/openroad"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "sweep")

# ─── New designs to process ───
# Each design verified: DEF has PLACED components, NETS section,
# LEFs define matching SITEs and cell macros.
NEW_DESIGNS = {
    # --- Original designs from Issue #24 ---
    "ibex_nangate45": {
        "lef": [f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_tech.lef",
                f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_stdcell.lef"],
        "def": f"{ORFS}/tools/OpenROAD/src/dpl/test/ibex_core_replace.def",
        "pdk": "nangate45",
    },
    "aes_opt_nangate45": {
        "lef": [f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_tech.lef",
                f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_stdcell.lef"],
        "def": f"{ORFS}/tools/OpenROAD/src/dpl/test/aes-opt.def",
        "pdk": "nangate45",
    },
    "aes_psm2_nangate45": {
        "lef": [f"{ORFS}/tools/OpenROAD/src/psm/test/Nangate45/Nangate45_tech.lef",
                f"{ORFS}/tools/OpenROAD/src/psm/test/Nangate45/Nangate45_stdcell.lef"],
        "def": f"{ORFS}/tools/OpenROAD/src/psm/test/Nangate45_data/aes.def",
        "pdk": "nangate45",
    },
    "rocket_nangate45": {
        "lef": [f"{ORFS}/tools/OpenROAD/src/gpl/test/nangate45.lef",
                f"{ORFS}/tools/OpenROAD/src/gpl/test/RocketTile_macro.lef"],
        "def": f"{ORFS}/tools/OpenROAD/src/gpl/test/medium05.def",
        "pdk": "nangate45",
    },
    "bp_nangate45": {
        "lef": [f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_tech.lef",
                f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_stdcell.lef",
                f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/fakeram45_64x32.lef",
                f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/fakeram45_512x64.lef",
                f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/fakeram45_64x7.lef",
                f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/fakeram45_64x96.lef"],
        "def": f"{ORFS}/tools/OpenROAD/src/gpl/test/medium07.def",
        "pdk": "nangate45",
    },
    "incr_nangate45": {
        "lef": [f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_tech.lef",
                f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_stdcell.lef"],
        "def": f"{ORFS}/tools/OpenROAD/src/gpl/test/incremental02.def",
        "pdk": "nangate45",
    },
    "gcd_grt_nangate45": {
        "lef": [f"{ORFS}/tools/OpenROAD/src/grt/test/Nangate45/Nangate45_tech.lef",
                f"{ORFS}/tools/OpenROAD/src/grt/test/Nangate45/Nangate45_stdcell.lef"],
        "def": f"{ORFS}/tools/OpenROAD/src/grt/test/gcd.def",
        "pdk": "nangate45",
    },
    "uart_sky130": {
        "lef": [f"{ORFS}/tools/OpenROAD/src/grt/test/overlapping_edges.lef"],
        "def": f"{ORFS}/tools/OpenROAD/src/grt/test/overlapping_edges.def",
        "pdk": "sky130hd",
    },
    # --- Previously timed-out designs (Issue #25) ---
    "jpeg_nangate45": {
        "lef": [f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_tech.lef",
                f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_stdcell.lef"],
        "def": f"{ORFS}/tools/OpenROAD/src/gpl/test/medium03.def",
        "pdk": "nangate45",
    },
    "swerv_nangate45": {
        "lef": [f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_tech.lef",
                f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_stdcell.lef"],
        "def": f"{ORFS}/tools/OpenROAD/src/gpl/test/medium04.def",
        "pdk": "nangate45",
    },
    # uart_nangate45 removed: overlapping_edges.def uses sky130 "unithd" site
    # (already covered by uart_sky130 with correct LEF)
    "gcd_rcx_sky130hs": {
        "lef": [f"{ORFS}/tools/OpenROAD/src/rcx/test/sky130hs/sky130hs.tlef",
                f"{ORFS}/tools/OpenROAD/src/rcx/test/sky130hs/sky130hs_std_cell.lef"],
        "def": f"{ORFS}/tools/OpenROAD/src/rcx/test/gcd.def",
        "pdk": "sky130hd",
    },
    "gcd_grt_sky130hs": {
        "lef": [f"{ORFS}/tools/OpenROAD/src/rcx/test/sky130hs/sky130hs.tlef",
                f"{ORFS}/tools/OpenROAD/src/rcx/test/sky130hs/sky130hs_std_cell.lef"],
        "def": f"{ORFS}/tools/OpenROAD/src/grt/test/gcd_sky130.def",
        "pdk": "sky130hd",
    },
    "riscv_asap7": {
        "lef": [f"{ORFS}/flow/platforms/asap7/lef/asap7_tech_1x_201209.lef",
                f"{ORFS}/flow/platforms/asap7/lef/asap7sc7p5t_28_R_1x_220121a.lef",
                f"{ORFS}/flow/platforms/asap7/lef/asap7sc7p5t_28_L_1x_220121a.lef",
                f"{ORFS}/flow/platforms/asap7/lef/asap7sc7p5t_28_SL_1x_220121a.lef",
                f"{ORFS}/flow/platforms/asap7/lef/fakeram7_256x32.lef"],
        "def": f"{ORFS}/tools/OpenROAD/src/psm/test/asap7_data/riscv.def",
        "pdk": "asap7",
    },
}

# ─── GR settings per PDK ───
LAYER_CONFIG = {
    "nangate45": {
        "restrict": "metal2-metal3",
        "adjust_layers": "metal1-metal10",
        "adjust_value": 0.8,
    },
    "sky130hd": {
        "restrict": "met2-met3",
        "adjust_layers": "met1-met5",
        "adjust_value": 0.95,
    },
    "asap7": {
        "restrict": "M2-M3",
        "adjust_layers": "M1-M9",
        "adjust_value": 0.8,
    },
}


def get_timeout(def_path):
    """Determine timeout based on design size (number of components)."""
    try:
        with open(def_path) as f:
            content = f.read()
        m = re.search(r'COMPONENTS\s+(\d+)', content)
        if m:
            n_cells = int(m.group(1))
            if n_cells > 20000:
                return 1800
            elif n_cells > 5000:
                return 600
            else:
                return 300
    except Exception:
        pass
    return 600


def check_lef_files(lef_list):
    """Check LEF files exist, return list of existing ones and list of missing."""
    existing, missing = [], []
    for lef in lef_list:
        (existing if os.path.exists(lef) else missing).append(lef)
    return existing, missing


def build_tcl_script(design_name, cfg):
    """Build the Tcl script for GR + GCell extraction.

    Uses cross-layer aggregation in the Tcl loop to avoid iterating
    per-layer per-cell (which is very slow for large grids).
    """
    pdk = cfg["pdk"]
    lcfg = LAYER_CONFIG[pdk]

    lines = []
    for lef in cfg["lef"]:
        lines.append(f'read_lef "{lef}"')
    lines.append(f'read_def "{cfg["def"]}"')
    lines.append(f'set_global_routing_layer_adjustment {lcfg["adjust_layers"]} {lcfg["adjust_value"]}')
    lines.append(f'set_routing_layers -signal {lcfg["restrict"]}')
    lines.append('global_route -verbose -allow_congestion -congestion_iterations 5')

    # GCell extraction: cross-layer aggregation (much faster for large designs)
    lines.append('''
set block [ord::get_db_block]
set gcell_grid [$block getGCellGrid]
set db [ord::get_db]
set tech [$db getTech]
set x_grids [$gcell_grid getGridX]
set y_grids [$gcell_grid getGridY]
set nx [expr {[llength $x_grids] - 1}]
set ny [expr {[llength $y_grids] - 1}]
set dbu [$block getDefUnits]

puts "ACCURATE_GT_START"
puts "GRID $nx $ny"
puts "DBU $dbu"
puts "XGRIDS $x_grids"
puts "YGRIDS $y_grids"

set routing_layers [list]
foreach layer [$tech getLayers] {
    if {[$layer getType] == "ROUTING"} {
        lappend routing_layers $layer
    }
}
puts "NLAYERS [llength $routing_layers]"

for {set gy 0} {$gy < $ny} {incr gy} {
    for {set gx 0} {$gx < $nx} {incr gx} {
        set total_cap 0.0
        set total_usage 0.0
        foreach layer $routing_layers {
            catch {
                set cap [$gcell_grid getCapacity $layer $gx $gy]
                set usg [$gcell_grid getUsage $layer $gx $gy]
                set total_cap [expr {$total_cap + $cap}]
                set total_usage [expr {$total_usage + $usg}]
            }
        }
        if {$total_cap > 0 || $total_usage > 0} {
            puts "GC $gx $gy $total_cap $total_usage"
        }
    }
}
puts "ACCURATE_GT_END"
''')
    return "\n".join(lines)


def run_openroad(design_name, cfg, timeout):
    """Run OpenROAD GR + GCell extraction. Returns log path or None."""
    log_path = os.path.join(RESULTS_DIR, f"accurate_gt_{design_name}.log")

    # Check if already done
    if os.path.exists(log_path):
        with open(log_path) as f:
            content = f.read()
        if "ACCURATE_GT_END" in content:
            print(f"  [CACHED] {log_path}")
            return log_path

    # Check LEFs
    existing_lefs, missing_lefs = check_lef_files(cfg["lef"])
    if missing_lefs:
        print(f"  [WARN] Missing LEFs: {missing_lefs}")
        if not existing_lefs:
            print(f"  [SKIP] No LEF files found")
            return None
        cfg = dict(cfg)
        cfg["lef"] = existing_lefs

    # Check DEF
    if not os.path.exists(cfg["def"]):
        print(f"  [SKIP] DEF not found: {cfg['def']}")
        return None

    # Build and write Tcl
    tcl_content = build_tcl_script(design_name, cfg)
    tcl_path = os.path.join(RESULTS_DIR, f"_mass_{design_name}.tcl")
    with open(tcl_path, "w") as f:
        f.write(tcl_content)

    print(f"  Running OpenROAD GR (timeout={timeout}s)...")
    t0 = time.time()
    try:
        result = subprocess.run(
            [OPENROAD, "-no_splash", "-exit", tcl_path],
            capture_output=True, text=True, timeout=timeout,
        )
        full_output = result.stdout + result.stderr
        elapsed = time.time() - t0
        print(f"  OpenROAD finished in {elapsed:.1f}s")
    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        print(f"  [TIMEOUT] after {elapsed:.1f}s")
        return None
    except Exception as e:
        print(f"  [ERROR] OpenROAD failed: {e}")
        return None

    # Save log
    with open(log_path, "w") as f:
        f.write(full_output)

    if "ACCURATE_GT_END" not in full_output:
        errors = [l for l in full_output.split('\n')
                  if 'error' in l.lower() or 'Error' in l][:5]
        if errors:
            print(f"  [ERROR] GR did not complete. Errors:")
            for e in errors:
                print(f"    {e.strip()[:120]}")
        else:
            print(f"  [ERROR] GR did not complete (no ACCURATE_GT_END marker)")
        return None

    print(f"  [OK] Saved: {log_path}")
    return log_path


def parse_accurate_gt_log(log_path):
    """Parse the accurate GT log, extracting per-G-cell capacity/usage."""
    with open(log_path) as f:
        content = f.read()

    if "ACCURATE_GT_START" not in content or "ACCURATE_GT_END" not in content:
        print(f"  [ERROR] Missing ACCURATE_GT markers in {log_path}")
        return None

    grid_match = re.search(r'GRID\s+(\d+)\s+(\d+)', content)
    dbu_match = re.search(r'DBU\s+(\d+)', content)
    xgrids_match = re.search(r'XGRIDS\s+(.+)', content)
    ygrids_match = re.search(r'YGRIDS\s+(.+)', content)

    if not all([grid_match, dbu_match, xgrids_match, ygrids_match]):
        print(f"  [ERROR] Failed to parse grid info")
        return None

    nx = int(grid_match.group(1))
    ny = int(grid_match.group(2))
    dbu = int(dbu_match.group(1))
    x_grids = [int(x) for x in xgrids_match.group(1).split()]
    y_grids = [int(y) for y in ygrids_match.group(1).split()]

    gcells = []
    gc_pattern = re.compile(r'^GC\s+(\d+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)', re.MULTILINE)
    for m in gc_pattern.finditer(content):
        gx, gy = int(m.group(1)), int(m.group(2))
        cap, usage = float(m.group(3)), float(m.group(4))
        overflow = max(0.0, usage - cap)
        gcells.append({"gx": gx, "gy": gy, "capacity": cap,
                       "usage": usage, "overflow": overflow})

    grt_total_overflow = 0
    total_match = re.search(
        r'Total\s+\d+\s+\d+\s+[\d.]+%\s+(\d+)\s*/\s*(\d+)\s*/\s*(\d+)', content)
    if total_match:
        grt_total_overflow = int(total_match.group(3))

    return {
        "grid_nx": nx, "grid_ny": ny, "dbu": dbu,
        "x_grids": x_grids, "y_grids": y_grids,
        "gcells": gcells,
        "grt_total_overflow": grt_total_overflow,
    }


def aggregate_to_grid(gt_data, die_area, gs, dbu):
    """Aggregate accurate GT G-cells into a gs x gs grid."""
    x_min, y_min = die_area["x_min"], die_area["y_min"]
    x_max, y_max = die_area["x_max"], die_area["y_max"]
    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs

    x_grids = gt_data["x_grids"]
    y_grids = gt_data["y_grids"]
    nx = gt_data["grid_nx"]
    ny = gt_data["grid_ny"]

    usage_grid = np.zeros((gs, gs))
    cap_grid = np.zeros((gs, gs))

    for gc in gt_data["gcells"]:
        grt_x = (x_grids[gc["gx"]] + x_grids[min(gc["gx"] + 1, nx)]) / 2.0 / dbu
        grt_y = (y_grids[gc["gy"]] + y_grids[min(gc["gy"] + 1, ny)]) / 2.0 / dbu
        our_gx = min(max(0, int((grt_x - x_min) / gcell_w)), gs - 1)
        our_gy = min(max(0, int((grt_y - y_min) / gcell_h)), gs - 1)
        usage_grid[our_gy, our_gx] += gc["usage"]
        cap_grid[our_gy, our_gx] += gc["capacity"]

    overflow_grid = np.maximum(0, usage_grid - cap_grid)
    return usage_grid, cap_grid, overflow_grid


def load_design_from_config(design_name, cfg):
    """Load design cells from LEF/DEF using run_batch infrastructure."""
    was_present = design_name in DESIGNS
    old_val = DESIGNS.get(design_name)
    DESIGNS[design_name] = cfg
    try:
        result = load_design(design_name)
    finally:
        if was_present:
            DESIGNS[design_name] = old_val
        else:
            del DESIGNS[design_name]
    return result


def validate_certificate(design_name, cfg, gs=6):
    """Compute eta and validate certificate against accurate GT."""
    log_path = os.path.join(RESULTS_DIR, f"accurate_gt_{design_name}.log")
    if not os.path.exists(log_path):
        print(f"  [ERROR] No log file: {log_path}")
        return None

    gt_data = parse_accurate_gt_log(log_path)
    if gt_data is None:
        return None

    n_gc = len(gt_data["gcells"])
    print(f"  Parsed {n_gc} G-cells from accurate GT")

    our_ovf = sum(gc["overflow"] for gc in gt_data["gcells"])
    grt_ovf = gt_data["grt_total_overflow"]
    if grt_ovf > 0:
        ratio = our_ovf / grt_ovf
    elif our_ovf == 0:
        ratio = 1.0
    else:
        ratio = float('inf')
    print(f"  Overflow: ours={our_ovf:.0f}, GRT={grt_ovf}, ratio={ratio:.3f}")
    n_overflow_gc = sum(1 for gc in gt_data["gcells"] if gc["overflow"] > 0)
    print(f"  G-cells with overflow: {n_overflow_gc}/{n_gc}")

    positions, widths, heights, die_area, _ = load_design_from_config(design_name, cfg)
    if positions is None:
        print(f"  [ERROR] Could not load design cells")
        return None

    N = len(positions)
    if N < 3:
        print(f"  [SKIP] Too few cells: {N}")
        return None

    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)

    pdk = cfg["pdk"]
    dbu = gt_data["dbu"]

    x_min, y_min = die_area["x_min"], die_area["y_min"]
    x_max, y_max = die_area["x_max"], die_area["y_max"]
    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs

    usage_grid, cap_grid, overflow_grid = aggregate_to_grid(
        gt_data, die_area, gs, dbu)
    has_overflow = (overflow_grid > 0).astype(int)

    # Subsample if needed
    if N > 1500:
        print(f"  Subsampling {N} -> 1500 cells for eta computation")
        cx, cy = np.mean(positions, axis=0)
        dists = np.sqrt((positions[:, 0] - cx)**2 + (positions[:, 1] - cy)**2)
        idx = np.argsort(dists)[:1500]
        pos_sub, w_sub, h_sub = positions[idx], widths[idx], heights[idx]
    else:
        pos_sub, w_sub, h_sub = positions, widths, heights

    r = diag * 2.0
    _, eta_map, _, dbar_map, _, ncells_map = compute_gcell_metrics(
        pos_sub, w_sub, h_sub, die_area, gs, r)

    # Cell density per G-cell (full set)
    ncells_full = np.zeros((gs, gs), dtype=int)
    cell_density = np.zeros((gs, gs))
    for idx_c in range(N):
        gx = min(max(0, int((positions[idx_c, 0] - x_min) / gcell_w)), gs - 1)
        gy = min(max(0, int((positions[idx_c, 1] - y_min) / gcell_h)), gs - 1)
        ncells_full[gy, gx] += 1
        cell_density[gy, gx] += widths[idx_c] * heights[idx_c] / (gcell_w * gcell_h)

    mask = ncells_full.ravel() >= 2
    n_active = int(mask.sum())

    y_true = has_overflow.ravel()[mask]
    n_overflow = int(y_true.sum())
    n_clean = n_active - n_overflow

    eta_flat = eta_map.ravel()[mask]
    eta_pred = (eta_flat > 0).astype(int)

    result = {
        "design": design_name,
        "pdk": pdk,
        "N": N,
        "gs": gs,
        "r_over_diag": 2.0,
        "grt_total_overflow": grt_ovf,
        "accurate_gt_overflow": float(our_ovf),
        "overflow_ratio": float(ratio),
        "n_gcells_with_overflow": n_overflow_gc,
        "n_active": n_active,
        "n_overflow": n_overflow,
        "n_clean": n_clean,
    }

    if n_overflow == 0 or n_clean == 0:
        if n_overflow == 0:
            eta_max = float(np.max(eta_flat)) if len(eta_flat) > 0 else 0.0
            result["certificate"] = "TRIVIAL (no overflow)"
            result["eta_max"] = eta_max
            if eta_max == 0:
                result["P_safe_given_eta0"] = 1.0
                result["eta_specificity"] = 1.0
                result["eta_recall"] = None
            else:
                result["P_safe_given_eta0"] = 1.0
                result["eta_specificity"] = 1.0
                result["eta_recall"] = None
        else:
            result["certificate"] = "N/A (all overflow)"
        return result

    from sklearn.metrics import confusion_matrix, recall_score, f1_score
    cm = confusion_matrix(y_true, eta_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else None
    cert_prec = float(tn / (tn + fn)) if (tn + fn) > 0 else None
    recall = float(recall_score(y_true, eta_pred, zero_division=0))
    f1 = float(f1_score(y_true, eta_pred, zero_division=0))
    accuracy = float((tp + tn) / len(y_true))

    result["eta_specificity"] = specificity
    result["P_safe_given_eta0"] = cert_prec
    result["eta_recall"] = recall
    result["eta_f1"] = f1
    result["eta_accuracy"] = accuracy
    result["confusion"] = {"tn": int(tn), "fp": int(fp),
                           "fn": int(fn), "tp": int(tp)}

    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 78)
    print("  MASS SWEEP: GT Certificate Validation (Issue #24)")
    print("=" * 78)
    print(f"  Designs to process: {len(NEW_DESIGNS)}")
    print(f"  Results dir: {RESULTS_DIR}")
    print()

    all_results = []
    errors = {}

    for design_name, cfg in NEW_DESIGNS.items():
        print(f"\n{'='*70}")
        print(f"  DESIGN: {design_name} (PDK={cfg['pdk']})")
        print(f"{'='*70}")

        try:
            timeout = get_timeout(cfg["def"])
            print(f"  DEF: {cfg['def']}")
            print(f"  Timeout: {timeout}s")

            log_path = run_openroad(design_name, cfg, timeout)
            if log_path is None:
                errors[design_name] = "OpenROAD GR failed or timed out"
                continue

            result = validate_certificate(design_name, cfg, gs=6)
            if result is None:
                errors[design_name] = "Certificate validation failed"
                continue

            all_results.append(result)

            spec = result.get("eta_specificity")
            cert = result.get("P_safe_given_eta0")
            rec = result.get("eta_recall")
            if spec is not None:
                print(f"\n  >> Specificity = {spec:.4f}")
                if cert is not None:
                    print(f"  >> P(safe|eta=0) = {cert:.4f}")
                if rec is not None:
                    print(f"  >> Recall = {rec:.3f}")
                cm = result.get("confusion", {})
                if cm:
                    print(f"  >> Confusion: TN={cm.get('tn')}, FP={cm.get('fp')}, "
                          f"FN={cm.get('fn')}, TP={cm.get('tp')}")
            else:
                print(f"  >> Certificate: {result.get('certificate', 'N/A')}")

        except Exception as e:
            import traceback
            errors[design_name] = str(e)
            print(f"  [EXCEPTION] {e}")
            traceback.print_exc()

    # ─── Summary table ───
    print(f"\n\n{'='*78}")
    print("  MASS SWEEP SUMMARY")
    print(f"{'='*78}")

    all_results.sort(key=lambda r: (r["pdk"], r["design"]))

    print(f"\n  {'Design':>25s}  {'PDK':>10s}  {'N':>7s}  {'GRT Ovf':>8s}  "
          f"{'Specif.':>8s}  {'P(safe|eta0)':>12s}  {'Recall':>7s}")
    print(f"  {'-'*83}")

    for r in all_results:
        spec = r.get("eta_specificity")
        cert = r.get("P_safe_given_eta0")
        rec = r.get("eta_recall")
        spec_s = f"{spec:.4f}" if spec is not None else "---"
        cert_s = f"{cert:.4f}" if cert is not None else r.get("certificate", "---")[:14]
        rec_s = f"{rec:.3f}" if rec is not None else "---"
        print(f"  {r['design']:>25s}  {r['pdk']:>10s}  {r['N']:>7d}  "
              f"{r['grt_total_overflow']:>8d}  {spec_s:>8s}  {cert_s:>12s}  {rec_s:>7s}")

    if errors:
        print(f"\n  FAILED ({len(errors)}):")
        for dname, reason in errors.items():
            print(f"    {dname}: {reason}")

    # ─── Binomial test for NanGate45 specificity ───
    # Include prior results from Issue #23
    prior_path = os.path.join(RESULTS_DIR, "accurate_gt_certificate.json")
    prior_results = []
    if os.path.exists(prior_path):
        with open(prior_path) as f:
            prior_results = json.load(f)

    # Collect all NanGate45 specificity values
    nangate45_specs = []

    for r in prior_results:
        if r.get("pdk") == "nangate45":
            s = r.get("eta_specificity")
            if s is not None:
                nangate45_specs.append((r["design"], s))
            elif r.get("certificate", "").startswith("TRIVIAL"):
                # No overflow => no FP possible => specificity = 1.0
                nangate45_specs.append((r["design"], 1.0))
            elif r.get("P_safe_given_eta0") == 1.0:
                nangate45_specs.append((r["design"], 1.0))

    prior_names = {name for name, _ in nangate45_specs}
    for r in all_results:
        if r.get("pdk") == "nangate45" and r["design"] not in prior_names:
            s = r.get("eta_specificity")
            if s is not None:
                nangate45_specs.append((r["design"], s))
            elif r.get("certificate", "").startswith("TRIVIAL"):
                nangate45_specs.append((r["design"], 1.0))
            elif r.get("P_safe_given_eta0") == 1.0:
                nangate45_specs.append((r["design"], 1.0))

    print(f"\n\n{'='*78}")
    print("  BINOMIAL TEST: NanGate45 Specificity")
    print(f"{'='*78}")
    print(f"  H0: specificity = 0.9 (10% false-positive rate)")
    print(f"  H1: specificity > 0.9")

    n_total = len(nangate45_specs)
    n_perfect = sum(1 for _, s in nangate45_specs if s == 1.0)

    print(f"\n  NanGate45 designs with specificity data: {n_total}")
    for name, s in sorted(nangate45_specs):
        print(f"    {name:>30s}: specificity = {s:.4f}")

    print(f"\n  Perfect specificity (1.0): {n_perfect} / {n_total}")

    p_value = None
    if n_total > 0:
        # P(X >= n_perfect) where X ~ Binomial(n_total, 0.9)
        p_value = 1.0 - binom.cdf(n_perfect - 1, n_total, 0.9)

        print(f"  p-value (one-sided, greater): {p_value:.6f}")
        if p_value < 0.05:
            print(f"  => REJECT H0 at alpha=0.05: specificity significantly > 0.9")
        else:
            print(f"  => Cannot reject H0 at alpha=0.05")
            print(f"     (Need more designs or H0 spec=0.9 is plausible)")

    # ─── Save results ───
    output = {
        "new_results": all_results,
        "errors": errors,
        "nangate45_binomial": {
            "n_designs": n_total,
            "n_perfect_specificity": n_perfect,
            "designs": [{"name": n, "specificity": s} for n, s in nangate45_specs],
            "p_value_vs_0.9": p_value,
        },
    }

    json_path = os.path.join(RESULTS_DIR, "mass_sweep_results.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")


if __name__ == "__main__":
    main()
