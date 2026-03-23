#!/usr/bin/env python3
"""
Held-out certificate validation on NEW NanGate45 designs (Issue #26).
=====================================================================

The first 6 designs (from Issue #23) are "training" where n_min=50 was
discovered. These 10 new designs are the held-out "test set."

For each design:
1. Generate Tcl script for GR + G-cell extraction
2. Run OpenROAD with timeout 600s
3. Parse GT from log, compute eta per G-cell
4. For n_min in [2, 10, 50, 100]: count FN and FP
5. Report the held-out result at n_min=50
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

sys.path.insert(0, os.path.dirname(__file__))
from run_batch import (
    parse_lef_macros, parse_def_components,
    build_overlap_coboundary, compute_eta, theory_eta,
    DESIGNS, load_design,
)
from eta_shield_placement import compute_gcell_metrics
from mass_sweep import parse_accurate_gt_log, aggregate_to_grid

# --- Paths ---
ORFS = "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts"
OPENROAD = f"{ORFS}/tools/install/OpenROAD/bin/openroad"
RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "sweep")

NANGATE45_LEFS = [
    f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_tech.lef",
    f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/Nangate45_stdcell.lef",
]

NEW_DESIGNS = {
    # --- Large designs with full placement + nets ---
    "rocket_nangate45": {
        "def": f"{ORFS}/tools/OpenROAD/src/gpl/test/medium05.def",
        "lef": NANGATE45_LEFS + [
            f"{ORFS}/tools/OpenROAD/src/gpl/test/Nangate45/fakeram45_64x32.lef",
        ],
        "pdk": "nangate45",
        "desc": "RocketTile, 42485 comp, 43429 nets",
    },
    "ibex_opt_nangate45": {
        "def": f"{ORFS}/tools/OpenROAD/src/dpl/test/ibex-opt.def",
        "lef": NANGATE45_LEFS,
        "pdk": "nangate45",
        "desc": "ibex-opt (DPL), 34184 comp, 33171 nets",
    },
    "aes_dpl_nangate45": {
        "def": f"{ORFS}/tools/OpenROAD/src/dpl/test/aes_cipher_top_replace.def",
        "lef": NANGATE45_LEFS,
        "pdk": "nangate45",
        "desc": "aes_cipher_top_replace (DPL), 21340 comp, 19675 nets",
    },
    "gcd_odb_nangate45": {
        "def": f"{ORFS}/tools/OpenROAD/src/odb/test/data/gcd/gcd_nangate45_route.def",
        "lef": NANGATE45_LEFS,
        "pdk": "nangate45",
        "desc": "gcd ODB routed, 1877 comp",
    },
    "gcd_opt_nangate45": {
        "def": f"{ORFS}/tools/OpenROAD/src/dpl/test/gcd-opt.def",
        "lef": NANGATE45_LEFS,
        "pdk": "nangate45",
        "desc": "gcd-opt (DPL), 549 comp, 364 nets",
    },
    "gcd_dpl_nangate45": {
        "def": f"{ORFS}/tools/OpenROAD/src/dpl/test/gcd_replace.def",
        "lef": NANGATE45_LEFS,
        "pdk": "nangate45",
        "desc": "gcd_replace (DPL), 549 comp, 364 nets",
    },
    "gcd_gpl_nangate45": {
        "def": f"{ORFS}/tools/OpenROAD/src/gpl/test/design/nangate45/gcd/gcd.def",
        "lef": NANGATE45_LEFS,
        "pdk": "nangate45",
        "desc": "gcd GPL test, 294 comp, 364 nets",
    },
    "gcd_incr_nangate45": {
        "def": f"{ORFS}/tools/OpenROAD/src/gpl/test/incremental01.def",
        "lef": NANGATE45_LEFS,
        "pdk": "nangate45",
        "desc": "gcd incremental01, 294 comp, 364 nets",
    },
    "grt_wire_nangate45": {
        "def": f"{ORFS}/tools/OpenROAD/src/grt/test/report_wire_length7.def",
        "lef": [f"{ORFS}/tools/OpenROAD/src/grt/test/Nangate45/Nangate45_tech.lef",
                f"{ORFS}/tools/OpenROAD/src/grt/test/Nangate45/Nangate45_stdcell.lef"],
        "pdk": "nangate45",
        "desc": "GRT wire_length test, 676 comp, 579 nets",
    },
    "dbsta_nangate45": {
        "def": f"{ORFS}/tools/OpenROAD/src/dbSta/test/report_cell_usage.def",
        "lef": NANGATE45_LEFS,
        "pdk": "nangate45",
        "desc": "dbSta cell_usage, 676 comp, 579 nets",
    },
    "max_disp_nangate45": {
        "def": f"{ORFS}/tools/OpenROAD/src/dpl/test/max_disp1.def",
        "lef": NANGATE45_LEFS,
        "pdk": "nangate45",
        "desc": "max_disp1 (DPL), 1768 comp, 130 nets",
    },
    "gcd_psm_nangate45": {
        "def": f"{ORFS}/tools/OpenROAD/src/psm/test/Nangate45_data/gcd.def",
        "lef": [f"{ORFS}/tools/OpenROAD/src/psm/test/Nangate45/Nangate45_tech.lef",
                f"{ORFS}/tools/OpenROAD/src/psm/test/Nangate45/Nangate45_stdcell.lef"],
        "pdk": "nangate45",
        "desc": "gcd PSM test, 624 comp",
    },
}

# --- GR settings for NanGate45 ---
LAYER_CONFIG = {
    "restrict": "metal2-metal3",
    "adjust_layers": "metal1-metal10",
    "adjust_value": 0.8,
}


def build_tcl_script(design_name, cfg):
    """Build the Tcl script for GR + GCell extraction."""
    lcfg = LAYER_CONFIG
    lines = []
    for lef in cfg["lef"]:
        lines.append(f'read_lef "{lef}"')
    lines.append(f'read_def "{cfg["def"]}"')
    lines.append(f'set_global_routing_layer_adjustment {lcfg["adjust_layers"]} {lcfg["adjust_value"]}')
    lines.append(f'set_routing_layers -signal {lcfg["restrict"]}')
    lines.append('global_route -verbose -allow_congestion -congestion_iterations 5')
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


def run_openroad(design_name, cfg, timeout=600):
    """Run OpenROAD GR + GCell extraction. Returns log path or None."""
    log_path = os.path.join(RESULTS_DIR, f"heldout_{design_name}.log")

    # Check if already done
    if os.path.exists(log_path):
        with open(log_path) as f:
            content = f.read()
        if "ACCURATE_GT_END" in content:
            print(f"  [CACHED] {log_path}")
            return log_path

    # Build and write Tcl
    tcl_content = build_tcl_script(design_name, cfg)
    tcl_path = os.path.join(RESULTS_DIR, f"_heldout_{design_name}.tcl")
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
        # Check for common errors
        if "GRT-0010" in full_output:
            print(f"  [SKIP] Unplaced instances (GRT-0010)")
        elif "GRT-0228" in full_output:
            print(f"  [SKIP] Edge usage overflow (GRT-0228)")
        else:
            errors = [l for l in full_output.split('\n')
                      if 'error' in l.lower() or 'Error' in l][:5]
            if errors:
                print(f"  [ERROR] GR did not complete:")
                for e in errors:
                    print(f"    {e.strip()[:120]}")
            else:
                print(f"  [ERROR] GR did not complete (no ACCURATE_GT_END)")
        return None

    print(f"  [OK] Saved: {log_path}")
    return log_path


def load_design_from_config(design_name, cfg):
    """Load design cells from LEF/DEF."""
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


def validate_scoped_certificate(design_name, cfg, gs=6, n_min_list=None):
    """Compute eta and validate scoped certificate at various n_min thresholds.

    The scoped certificate restricts analysis to G-cells with at least n_min
    cells. At n_min=50, we expect the certificate to have zero FN (no false
    negatives where eta=0 but overflow occurs).

    Returns dict with results for each n_min threshold.
    """
    if n_min_list is None:
        n_min_list = [2, 10, 50, 100]

    log_path = os.path.join(RESULTS_DIR, f"heldout_{design_name}.log")
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

    # Load design cells
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
    dbu = gt_data["dbu"]

    x_min, y_min = die_area["x_min"], die_area["y_min"]
    x_max, y_max = die_area["x_max"], die_area["y_max"]
    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs

    # Aggregate GT into our grid
    usage_grid, cap_grid, overflow_grid = aggregate_to_grid(
        gt_data, die_area, gs, dbu)
    has_overflow = (overflow_grid > 0).astype(int)

    # Compute eta -- NO subsampling per issue instructions.
    # SVD is done per G-cell (not globally), so even for large designs
    # the per-G-cell cell count is manageable. Subsampling would reduce
    # cell density within each G-cell, causing dbar to drop and eta=0
    # everywhere (false negatives).
    r = diag * 2.0
    print(f"  Computing eta with FULL {N} cells (per-G-cell SVD, r={r:.3f})...")
    _, eta_map, _, dbar_map, _, ncells_map = compute_gcell_metrics(
        positions, widths, heights, die_area, gs, r)

    # Cell count per G-cell using FULL cell set (not subsampled)
    ncells_full = np.zeros((gs, gs), dtype=int)
    for idx_c in range(N):
        gx = min(max(0, int((positions[idx_c, 0] - x_min) / gcell_w)), gs - 1)
        gy = min(max(0, int((positions[idx_c, 1] - y_min) / gcell_h)), gs - 1)
        ncells_full[gy, gx] += 1

    # Results for each n_min threshold
    result = {
        "design": design_name,
        "pdk": "nangate45",
        "N": N,
        "gs": gs,
        "r_over_diag": 2.0,
        "grt_total_overflow": grt_ovf,
        "accurate_gt_overflow": float(our_ovf),
        "overflow_ratio": float(ratio),
        "n_gcells_with_overflow": n_overflow_gc,
        "ncells_full": ncells_full.tolist(),
        "eta_map": [[float(v) for v in row] for row in eta_map],
        "has_overflow": has_overflow.tolist(),
        "thresholds": {},
    }

    for n_min in n_min_list:
        mask = ncells_full.ravel() >= n_min
        n_active = int(mask.sum())

        y_true = has_overflow.ravel()[mask]
        n_overflow = int(y_true.sum())
        n_clean = n_active - n_overflow

        eta_flat = eta_map.ravel()[mask]
        eta_pred = (eta_flat > 0).astype(int)

        threshold_result = {
            "n_min": n_min,
            "n_active": n_active,
            "n_overflow": n_overflow,
            "n_clean": n_clean,
        }

        if n_active == 0:
            threshold_result["status"] = "no active G-cells"
            result["thresholds"][str(n_min)] = threshold_result
            continue

        if n_overflow == 0:
            # No overflow: certificate is trivially true
            eta_max = float(np.max(eta_flat)) if len(eta_flat) > 0 else 0.0
            threshold_result["status"] = "TRIVIAL (no overflow)"
            threshold_result["eta_max"] = eta_max
            threshold_result["FN"] = 0
            threshold_result["FP"] = int(np.sum(eta_pred))
            threshold_result["TN"] = n_clean - int(np.sum(eta_pred))
            threshold_result["TP"] = 0
            threshold_result["P_safe_given_eta0"] = 1.0
        elif n_clean == 0:
            threshold_result["status"] = "all overflow"
            threshold_result["FN"] = int(np.sum(eta_pred == 0))
            threshold_result["FP"] = 0
            threshold_result["TN"] = 0
            threshold_result["TP"] = int(np.sum(eta_pred))
        else:
            from sklearn.metrics import confusion_matrix as cm_func
            cm = cm_func(y_true, eta_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()
            threshold_result["TN"] = int(tn)
            threshold_result["FP"] = int(fp)
            threshold_result["FN"] = int(fn)
            threshold_result["TP"] = int(tp)
            threshold_result["P_safe_given_eta0"] = float(tn / (tn + fn)) if (tn + fn) > 0 else None
            threshold_result["specificity"] = float(tn / (tn + fp)) if (tn + fp) > 0 else None
            threshold_result["status"] = "computed"

        result["thresholds"][str(n_min)] = threshold_result

    return result


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    print("=" * 78)
    print("  HELD-OUT CERTIFICATE VALIDATION (Issue #26)")
    print("  10 NEW NanGate45 designs — held-out test set")
    print("=" * 78)
    print(f"  Results dir: {RESULTS_DIR}")
    print()

    # --- Step 1: Run OpenROAD for all designs ---
    print("\n" + "=" * 78)
    print("  PHASE 1: Run OpenROAD GR + G-cell extraction")
    print("=" * 78)

    succeeded = {}
    failed = {}

    for design_name, cfg in NEW_DESIGNS.items():
        print(f"\n{'='*70}")
        print(f"  DESIGN: {design_name} — {cfg['desc']}")
        print(f"  DEF: {cfg['def']}")
        print(f"{'='*70}")

        log_path = run_openroad(design_name, cfg, timeout=600)
        if log_path is None:
            failed[design_name] = "GR failed or timed out"
        else:
            succeeded[design_name] = log_path

    print(f"\n\n  GR Results: {len(succeeded)} succeeded, {len(failed)} failed")
    for name, reason in failed.items():
        print(f"    FAILED: {name} — {reason}")

    # --- Step 2: Validate certificate for all succeeded designs ---
    print("\n" + "=" * 78)
    print("  PHASE 2: Scoped certificate analysis")
    print("=" * 78)

    all_results = []
    analysis_errors = {}

    for design_name in succeeded:
        cfg = NEW_DESIGNS[design_name]
        print(f"\n{'='*70}")
        print(f"  ANALYZING: {design_name}")
        print(f"{'='*70}")

        try:
            result = validate_scoped_certificate(design_name, cfg, gs=6,
                                                  n_min_list=[2, 10, 50, 100])
            if result is None:
                analysis_errors[design_name] = "validation returned None"
                continue
            all_results.append(result)

            # Print summary for this design
            for n_min_key in ["2", "10", "50", "100"]:
                t = result["thresholds"].get(n_min_key)
                if t is None:
                    continue
                fn = t.get("FN", "?")
                fp = t.get("FP", "?")
                p_safe = t.get("P_safe_given_eta0")
                p_safe_s = f"{p_safe:.4f}" if p_safe is not None else "---"
                print(f"    n_min={n_min_key:>3s}: active={t['n_active']:>3d}, "
                      f"overflow={t['n_overflow']:>2d}, "
                      f"FN={fn}, FP={fp}, P(safe|eta=0)={p_safe_s}")

        except Exception as e:
            import traceback
            analysis_errors[design_name] = str(e)
            print(f"  [EXCEPTION] {e}")
            traceback.print_exc()

    # --- Step 3: Summary ---
    print(f"\n\n{'='*78}")
    print("  HELD-OUT CERTIFICATE SUMMARY")
    print(f"{'='*78}")

    n_min_list = [2, 10, 50, 100]

    print(f"\n  {'Design':>25s}  {'N':>7s}  {'GRT Ovf':>8s}  |", end="")
    for nm in n_min_list:
        print(f"  n>={nm:>3d}: FN/FP", end="")
    print()
    print(f"  {'-'*90}")

    total_fn = {nm: 0 for nm in n_min_list}
    total_fp = {nm: 0 for nm in n_min_list}
    total_active = {nm: 0 for nm in n_min_list}
    total_overflow = {nm: 0 for nm in n_min_list}

    for r in all_results:
        print(f"  {r['design']:>25s}  {r['N']:>7d}  {r['grt_total_overflow']:>8d}  |", end="")
        for nm in n_min_list:
            t = r["thresholds"].get(str(nm), {})
            fn = t.get("FN", "?")
            fp = t.get("FP", "?")
            print(f"  {nm:>3d}: {fn}/{fp}    ", end="")
            if isinstance(fn, int):
                total_fn[nm] += fn
            if isinstance(fp, int):
                total_fp[nm] += fp
            total_active[nm] += t.get("n_active", 0)
            total_overflow[nm] += t.get("n_overflow", 0)
        print()

    print(f"  {'-'*90}")
    print(f"  {'TOTAL':>25s}  {'':>7s}  {'':>8s}  |", end="")
    for nm in n_min_list:
        print(f"  {nm:>3d}: {total_fn[nm]}/{total_fp[nm]}    ", end="")
    print()

    # --- KEY RESULT ---
    print(f"\n\n  {'='*60}")
    print(f"  KEY RESULT: Held-out test at n_min=50")
    print(f"  {'='*60}")
    print(f"  Total designs succeeded: {len(all_results)}")
    print(f"  Total active G-cells (n>=50): {total_active[50]}")
    print(f"  Total G-cells with overflow: {total_overflow[50]}")
    print(f"  Total FN (eta=0 but overflow): {total_fn[50]}")
    print(f"  Total FP (eta>0 but no overflow): {total_fp[50]}")
    if total_active[50] > 0:
        fn_rate = total_fn[50] / max(1, total_overflow[50]) if total_overflow[50] > 0 else 0
        print(f"  FN rate: {fn_rate:.4f}")

    if analysis_errors:
        print(f"\n  Analysis errors ({len(analysis_errors)}):")
        for name, reason in analysis_errors.items():
            print(f"    {name}: {reason}")

    if failed:
        print(f"\n  GR failures ({len(failed)}):")
        for name, reason in failed.items():
            print(f"    {name}: {reason}")

    # --- Save results ---
    # Strip large fields for JSON output
    save_results = []
    for r in all_results:
        r_save = dict(r)
        r_save.pop("ncells_full", None)
        r_save.pop("eta_map", None)
        r_save.pop("has_overflow", None)
        save_results.append(r_save)

    output = {
        "description": "Held-out scoped certificate validation on 10 new NanGate45 designs",
        "issue": 26,
        "designs_attempted": list(NEW_DESIGNS.keys()),
        "designs_succeeded": list(succeeded.keys()),
        "designs_failed": failed,
        "analysis_errors": analysis_errors,
        "results": save_results,
        "summary": {
            n_min: {
                "total_active": total_active[n_min],
                "total_overflow": total_overflow[n_min],
                "total_FN": total_fn[n_min],
                "total_FP": total_fp[n_min],
            }
            for n_min in n_min_list
        },
        "key_result_n_min_50": {
            "total_FN": total_fn[50],
            "total_FP": total_fp[50],
            "total_active": total_active[50],
            "total_overflow": total_overflow[50],
        },
    }

    json_path = os.path.join(RESULTS_DIR, "heldout_certificate.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")


if __name__ == "__main__":
    main()
