#!/usr/bin/env python3
"""
validate_theorem_Gp.py — Validate Theorem G' (Direction-Based Rank Bound)

Tests G-P4: rank(δ) ≈ Σ_k (|V_k| - c_k) where groups are by direction.

For each edge, the restriction map direction is (Δx, Δy) up to scaling.
Edges sharing the same direction constrain the same 1D projection.
"""

import sys, os, json, numpy as np
from scipy.linalg import svd
from scipy.spatial import KDTree
from collections import defaultdict

sys.path.insert(0, os.path.dirname(__file__))
from run_all import (
    parse_lef_macros, parse_def_components,
    build_overlap_coboundary, compute_eta,
)

ORFS = "/mnt/storage1/users/ydwu/claude_projects/OpenROAD-flow-scripts"

DESIGNS = {
    "gcd_preroute": {
        "def": f"{ORFS}/tools/OpenROAD/src/drt/test/gcd_nangate45_preroute.def",
        "lef": [
            f"{ORFS}/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_tech.lef",
            f"{ORFS}/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_stdcell.lef",
        ],
    },
    "gcd_sky130": {
        "def": f"{ORFS}/flow/tutorials/scripts/drt/gcd/4_cts.def",
        "lef": [
            f"{ORFS}/flow/platforms/sky130hd/lef/sky130_fd_sc_hd.tlef",
            f"{ORFS}/flow/platforms/sky130hd/lef/sky130_fd_sc_hd_merged.lef",
        ],
    },
    "aes_preroute": {
        "def": f"{ORFS}/tools/OpenROAD/src/drt/test/aes_nangate45_preroute.def",
        "lef": [
            f"{ORFS}/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_tech.lef",
            f"{ORFS}/tools/OpenROAD/src/drt/test/Nangate45/Nangate45_stdcell.lef",
        ],
    },
}


def direction_key(dx, dy, tol=0.001):
    """Normalize (dx, dy) to a canonical direction key.

    Map to (1, dy/dx) if |dx| > |dy|, else (dx/dy, 1).
    Handle sign: always positive first nonzero component.
    Quantize to avoid floating point issues.
    """
    if abs(dx) < tol and abs(dy) < tol:
        return (0, 0)

    # Normalize by the larger component
    norm = max(abs(dx), abs(dy))
    ndx, ndy = dx / norm, dy / norm

    # Canonical sign: first nonzero is positive
    if ndx < -tol or (abs(ndx) < tol and ndy < -tol):
        ndx, ndy = -ndx, -ndy

    # Quantize to grid (avoid float equality issues)
    return (round(ndx, 4), round(ndy, 4))


def compute_direction_rank(positions, edges):
    """Compute Theorem G' direction-based rank bound.

    Groups edges by direction, computes per-group rank contribution.
    """
    N = len(positions)

    # Group edges by direction
    groups = defaultdict(list)
    for idx, (i, j) in enumerate(edges):
        dx = positions[i, 0] - positions[j, 0]
        dy = positions[i, 1] - positions[j, 1]
        key = direction_key(dx, dy)
        groups[key].append((i, j))

    K = len(groups)

    # Per-group rank contribution: |V_k| - c_k
    total_rank_Gp = 0
    group_stats = []

    for key, group_edges in sorted(groups.items(), key=lambda x: -len(x[1])):
        # Find vertices in this group
        verts = set()
        for i, j in group_edges:
            verts.add(i)
            verts.add(j)
        V_k = len(verts)

        # Count connected components via union-find
        parent = {v: v for v in verts}
        rank_uf = {v: 0 for v in verts}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px == py:
                return
            if rank_uf[px] < rank_uf[py]:
                px, py = py, px
            parent[py] = px
            if rank_uf[px] == rank_uf[py]:
                rank_uf[px] += 1

        for i, j in group_edges:
            union(i, j)

        c_k = len(set(find(v) for v in verts))
        rank_k = V_k - c_k

        total_rank_Gp += rank_k

        group_stats.append({
            "direction": list(key),
            "n_edges": len(group_edges),
            "n_vertices": V_k,
            "n_components": c_k,
            "rank_contribution": rank_k,
        })

    return total_rank_Gp, K, group_stats


def validate_design(name, cfg, results_dir, max_cells=1500):
    """Run Theorem G' validation."""
    print(f"\n{'='*70}")
    print(f"Design: {name}")
    print(f"{'='*70}")

    if not os.path.exists(cfg["def"]):
        print(f"  [SKIP] DEF not found")
        return None

    cell_sizes = parse_lef_macros(cfg["lef"])
    cells, meta = parse_def_components(cfg["def"])

    valid_cells = []
    for c in cells:
        if c.macro in cell_sizes:
            c.width, c.height = cell_sizes[c.macro]
        elif c.width == 0:
            continue
        c.compute_center()
        if not c.is_filler and not c.is_tap:
            valid_cells.append(c)

    positions = np.array([(c.cx, c.cy) for c in valid_cells])
    widths = np.array([c.width for c in valid_cells])
    heights = np.array([c.height for c in valid_cells])
    N_total = len(positions)

    if N_total > max_cells:
        cx_mid, cy_mid = np.median(positions, axis=0)
        dists = np.linalg.norm(positions - [cx_mid, cy_mid], axis=1)
        idx = np.argsort(dists)[:max_cells]
        positions = positions[idx]
        widths = widths[idx]
        heights = heights[idx]

    N = len(positions)
    med_h = np.median(heights)
    med_w = np.median(widths)
    diag = np.sqrt(med_w**2 + med_h**2)
    print(f"  N={N}, cell={med_w:.3f}×{med_h:.3f}μm")

    tree = KDTree(positions)

    # Test at several radii
    r_values = sorted(set([
        med_w * 1.5, med_h * 1.0, med_h * 1.2,
        diag * 1.0, diag * 1.5, diag * 2.0,
        diag * 3.0, diag * 5.0,
    ]))

    print(f"\n  {'r':>6} {'|E|':>6} {'K':>4} {'rank_Gp':>7} {'rank_δ':>6} "
          f"{'err':>5} {'η_Gp':>6} {'η_act':>6}")
    print("  " + "-" * 60)

    sweep = []
    for r in r_values:
        pairs = tree.query_pairs(r=r)
        edges = sorted((min(i, j), max(i, j)) for i, j in pairs)
        nE = len(edges)
        if nE < 10 or nE > 15000:
            continue

        # Theorem G' prediction
        rank_Gp, K, group_stats = compute_direction_rank(positions, edges)

        # Actual rank
        delta = build_overlap_coboundary(positions, edges, n_v=2)
        res = compute_eta(delta)
        rank_actual = res["rank"]

        err = rank_actual - rank_Gp
        eta_Gp = 1 - rank_Gp / nE if nE > 0 else 0
        eta_actual = res["eta"]

        entry = {
            "r": float(r),
            "nE": nE,
            "K_directions": K,
            "rank_Gp": rank_Gp,
            "rank_actual": rank_actual,
            "rank_error": err,
            "eta_Gp": float(eta_Gp),
            "eta_actual": float(eta_actual),
            "top_groups": group_stats[:5],
        }
        sweep.append(entry)

        print(f"  {r:6.2f} {nE:6d} {K:4d} {rank_Gp:7d} {rank_actual:6d} "
              f"{err:+5d} {eta_Gp:6.3f} {eta_actual:6.3f}")

    output = {"design": name, "N": N, "sweep": sweep}

    path = os.path.join(results_dir, f"theorem_Gp_{name}.json")
    with open(path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"  Saved: {path}")
    return output


def main():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    for name, cfg in DESIGNS.items():
        validate_design(name, cfg, results_dir)


if __name__ == '__main__':
    main()
