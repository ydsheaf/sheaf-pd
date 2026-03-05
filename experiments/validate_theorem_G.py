#!/usr/bin/env python3
"""
validate_theorem_G.py — Validate Theorem G (Row-Based Genericity Gap)

Tests:
  1. rank(δ) = (|V| - c) + min(|E_inter|, c + |V|)  [rank formula]
  2. gap = max(0, c + |V| - |E_inter|) / |E|          [gap formula]
  3. n_v^eff = rank(δ) / |V| interpolates 1 → 2       [effective stalk dim]
  4. SKY130 designs have larger gap than ASAP7/NanGate  [PDK dependence]

Uses existing run_all.py infrastructure for DEF parsing and coboundary.
"""

import sys
import os
import json
import numpy as np
from scipy.linalg import svd
from scipy.spatial import KDTree
from collections import defaultdict

# Import from run_all.py
sys.path.insert(0, os.path.dirname(__file__))
from run_all import (
    parse_lef_macros, parse_def_components,
    build_overlap_coboundary, compute_eta, theory_eta,
)

# ── Design configs (same as run_all.py) ──

ORFS = "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts"
PD_RESULTS = "/mnt/storage1/users/ydwu/claude_projects/sheaf-pd/experiments/results/drt"

# Use SAME DEFs as run_all.py for consistency
DESIGNS = {
    "gcd_preroute": {
        "pdk": "nangate45",
        "def": f"{ORFS}/tools/OpenROAD/src/drt/test/gcd_nangate45_preroute.def",
        "lef": [
            f"{ORFS}/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_tech.lef",
            f"{ORFS}/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_stdcell.lef",
        ],
    },
    "gcd_sky130": {
        "pdk": "sky130hd",
        "def": f"{ORFS}/flow/tutorials/scripts/drt/gcd/4_cts.def",
        "lef": [
            f"{ORFS}/flow/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef",
            f"{ORFS}/flow/platforms/sky130hd/lef/sky130_fd_sc_hd_merged.lef",
        ],
    },
    "aes_preroute": {
        "pdk": "nangate45",
        "def": f"{ORFS}/tools/OpenROAD/src/drt/test/aes_nangate45_preroute.def",
        "lef": [
            f"{ORFS}/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_tech.lef",
            f"{ORFS}/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_stdcell.lef",
        ],
    },
}

# Also check for placed DEFs (stage 3)
for name, cfg in list(DESIGNS.items()):
    placed_def = cfg["def"].replace("2_floorplan.def", "3_place.def")
    if os.path.exists(placed_def):
        cfg["def"] = placed_def


def identify_rows(positions, tol_frac=0.01):
    """Identify rows by clustering y-coordinates.

    tol_frac: fraction of median cell height used as tolerance.
    Returns: row_id per cell (int array), number of rows m.
    """
    y_coords = positions[:, 1]

    # Cluster y-coordinates: sort, then group by proximity
    y_sorted_idx = np.argsort(y_coords)
    y_sorted = y_coords[y_sorted_idx]

    # Use median gap between distinct y-values as tolerance
    y_unique = np.unique(np.round(y_sorted, decimals=4))
    if len(y_unique) <= 1:
        return np.zeros(len(positions), dtype=int), 1

    gaps = np.diff(y_unique)
    tol = np.median(gaps) * tol_frac if len(gaps) > 0 else 0.01

    # Assign row IDs
    row_ids = np.zeros(len(positions), dtype=int)
    current_row = 0
    row_y = y_sorted[0]

    for idx in y_sorted_idx:
        if abs(y_coords[idx] - row_y) > tol:
            current_row += 1
            row_y = y_coords[idx]
        row_ids[idx] = current_row

    m = current_row + 1
    return row_ids, m


def partition_edges(edges, row_ids):
    """Partition edges into intra-row and inter-row."""
    E_intra = []
    E_inter = []
    for i, j in edges:
        if row_ids[i] == row_ids[j]:
            E_intra.append((i, j))
        else:
            E_inter.append((i, j))
    return E_intra, E_inter


def count_intra_components(N, E_intra, row_ids, m):
    """Count connected components of the intra-row subgraph.

    Uses union-find for efficiency.
    """
    parent = list(range(N))
    rank = [0] * N

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px == py:
            return
        if rank[px] < rank[py]:
            px, py = py, px
        parent[py] = px
        if rank[px] == rank[py]:
            rank[px] += 1

    for i, j in E_intra:
        union(i, j)

    components = len(set(find(i) for i in range(N)))
    return components


def theorem_G_rank(N, c, E_inter_count):
    """Predicted rank from Theorem G."""
    return (N - c) + min(E_inter_count, c + N)


def validate_design(name, cfg, results_dir, max_cells=1500):
    """Run Theorem G validation on one design."""
    print(f"\n{'='*70}")
    print(f"Design: {name} (PDK: {cfg['pdk']})")
    print(f"{'='*70}")

    # Check if DEF exists
    if not os.path.exists(cfg["def"]):
        print(f"  [SKIP] DEF not found: {cfg['def']}")
        return None

    # Parse cells
    cell_sizes = parse_lef_macros(cfg["lef"])
    cells, meta = parse_def_components(cfg["def"])
    dbu = meta.get("units", 1000)

    # Filter and compute centers
    # NOTE: parse_def_components already converts x,y to microns (divides by dbu)
    valid_cells = []
    for c_obj in cells:
        if c_obj.macro in cell_sizes:
            c_obj.width, c_obj.height = cell_sizes[c_obj.macro]
        elif c_obj.width == 0:
            continue
        c_obj.compute_center()
        if not c_obj.is_filler and not c_obj.is_tap:
            valid_cells.append(c_obj)

    N_total = len(valid_cells)
    print(f"  Total cells: {N_total}")

    if N_total == 0:
        print("  [SKIP] No valid cells")
        return None

    # Subsample for SVD tractability
    positions = np.array([(c_obj.cx, c_obj.cy) for c_obj in valid_cells])
    widths = np.array([c_obj.width for c_obj in valid_cells])
    heights = np.array([c_obj.height for c_obj in valid_cells])

    if N_total > max_cells:
        cx_mid = np.median(positions[:, 0])
        cy_mid = np.median(positions[:, 1])
        dists = np.sqrt((positions[:, 0] - cx_mid)**2 +
                        (positions[:, 1] - cy_mid)**2)
        idx_sub = np.argsort(dists)[:max_cells]
        positions = positions[idx_sub]
        widths = widths[idx_sub]
        heights = heights[idx_sub]

    N = len(positions)
    print(f"  Using {N} cells (subsampled)" if N < N_total else f"  Using {N} cells")

    # Identify rows
    row_ids, m = identify_rows(positions)
    print(f"  Rows detected: {m}")
    cells_per_row = np.bincount(row_ids)
    print(f"  Cells per row: min={cells_per_row.min()}, median={int(np.median(cells_per_row))}, max={cells_per_row.max()}")

    # Cell size stats
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)
    print(f"  Cell size: {med_w:.3f} x {med_h:.3f} μm, aspect={med_h/med_w:.2f}")

    # Sweep interaction radius — start VERY small to see intra-row regime
    tree = KDTree(positions)
    row_pitch = med_h  # approximate row-to-row distance
    r_values = sorted(set([
        med_w * 0.6, med_w * 0.8, med_w * 1.0,  # sub-cell: mostly intra-row
        med_w * 1.5, med_w * 2.0, med_w * 3.0,   # cell-scale
        row_pitch * 0.8, row_pitch * 1.0,          # row pitch boundary
        row_pitch * 1.2, row_pitch * 1.5,          # cross-row begins
        row_pitch * 2.0, row_pitch * 3.0,          # multi-row
        diag * 2.0, diag * 3.0, diag * 5.0,       # large radii
        diag * 10.0,
    ]))

    print(f"\n  {'r':>6} {'|E|':>6} {'E_intr':>6} {'E_inte':>6} {'c':>5} "
          f"{'rank_G':>6} {'rank_δ':>6} {'err':>5} "
          f"{'η_row':>6} {'η_gen':>6} {'gap':>6} {'nv_eff':>6}")
    print("  " + "-" * 85)

    sweep_results = []

    for r in r_values:
        pairs = tree.query_pairs(r=r)
        edges = sorted((min(i, j), max(i, j)) for i, j in pairs)
        nE = len(edges)
        if nE < 5:
            continue

        dbar = 2.0 * nE / N

        # Partition edges
        E_intra, E_inter = partition_edges(edges, row_ids)
        nE_intra = len(E_intra)
        nE_inter = len(E_inter)

        # Count intra-row connected components
        c = count_intra_components(N, E_intra, row_ids, m)

        # Theorem G predicted rank
        rank_G = theorem_G_rank(N, c, nE_inter)

        # Actual rank
        delta = build_overlap_coboundary(positions, edges, n_v=2)
        res = compute_eta(delta)
        rank_actual = res["rank"]

        # Metrics
        eta_row = 1.0 - rank_G / nE if nE > 0 else 0
        eta_generic = max(0.0, 1.0 - 2 * N / nE) if nE > 0 else 0
        gap = max(0, (c + N - nE_inter)) / nE if nE > 0 else 0
        nv_eff = rank_G / N if N > 0 else 0
        rank_err = rank_actual - rank_G

        entry = {
            "r": float(r),
            "nE": nE,
            "nE_intra": nE_intra,
            "nE_inter": nE_inter,
            "c": c,
            "dbar": float(dbar),
            "rank_theorem_G": rank_G,
            "rank_actual": rank_actual,
            "rank_error": rank_err,
            "eta_row": float(eta_row),
            "eta_generic": float(eta_generic),
            "eta_actual": float(res["eta"]),
            "gap_predicted": float(gap),
            "gap_actual": float(res["eta"] - eta_generic),
            "nv_eff": float(nv_eff),
            "f_intra": float(nE_intra / nE) if nE > 0 else 0,
        }
        sweep_results.append(entry)

        print(f"  {r:6.2f} {nE:6d} {nE_intra:6d} {nE_inter:6d} {c:5d} "
              f"{rank_G:6d} {rank_actual:6d} {rank_err:+5d} "
              f"{eta_row:6.3f} {eta_generic:6.3f} {gap:6.3f} {nv_eff:6.2f}")

        if nE > 15000:
            print("  [stopping: matrix too large]")
            break

    # Summary
    output = {
        "design": name,
        "pdk": cfg["pdk"],
        "N": N,
        "N_total": N_total,
        "m_rows": m,
        "cell_aspect": float(med_h / med_w),
        "sweep": sweep_results,
    }

    json_path = os.path.join(results_dir, f"theorem_G_{name}.json")
    with open(json_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")

    return output


def main():
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []
    for name, cfg in DESIGNS.items():
        result = validate_design(name, cfg, results_dir)
        if result:
            all_results.append(result)

    # Cross-design summary
    print("\n" + "=" * 70)
    print("CROSS-DESIGN SUMMARY")
    print("=" * 70)
    print(f"\n  {'Design':>25s} {'PDK':>10s} {'aspect':>7s} {'m_rows':>6s} "
          f"{'max_gap':>8s} {'nv_best':>7s}")
    print("  " + "-" * 75)

    for r in all_results:
        if not r["sweep"]:
            continue
        max_gap = max(e["gap_predicted"] for e in r["sweep"])
        # Find nv_eff at moderate radius
        mid = len(r["sweep"]) // 2
        nv = r["sweep"][mid]["nv_eff"] if mid < len(r["sweep"]) else 0
        print(f"  {r['design']:>25s} {r['pdk']:>10s} {r['cell_aspect']:>7.2f} "
              f"{r['m_rows']:>6d} {max_gap:>8.3f} {nv:>7.2f}")

    summary_path = os.path.join(results_dir, "theorem_G_summary.json")
    with open(summary_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nFull results: {summary_path}")


if __name__ == '__main__':
    main()
