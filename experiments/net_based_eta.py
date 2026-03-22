#!/usr/bin/env python3
"""
Net-based η: sheaf cohomology on the NETLIST graph (Issue #20)
==============================================================

Instead of proximity graph (cells within distance r), use the actual
netlist as the constraint graph. Edge (i,j) exists iff cells i and j
share a net.

This measures routing constraint topology, not placement density.
The key hypothesis: net-based η captures routing-specific structure
that proximity-based η (which is basically density) misses.

Usage:
  python experiments/net_based_eta.py
  python experiments/net_based_eta.py --design aes_nangate45 --gs 10
"""

import argparse
import json
import os
import re
import sys
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy.linalg import svd
from scipy.spatial import KDTree
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(__file__))
from run_batch import (
    parse_lef_macros, parse_def_components,
    build_overlap_coboundary, compute_eta, theory_eta,
    DESIGNS, load_design,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results", "gr")


# ═══════════════════════════════════════════════════════════════════
# DEF Net Parser
# ═══════════════════════════════════════════════════════════════════

def parse_def_nets(def_path):
    """Parse NETS section from DEF. Returns list of nets, each a list of instance names."""
    with open(def_path) as f:
        content = f.read()

    # Find NETS section
    nets_match = re.search(r'^NETS\s+(\d+)\s*;(.+?)^END NETS',
                           content, re.MULTILINE | re.DOTALL)
    if not nets_match:
        print("  WARNING: No NETS section found")
        return []

    n_nets = int(nets_match.group(1))
    nets_text = nets_match.group(2)

    nets = []
    # Each net: - netname ( inst pin ) ( inst pin ) ... ;
    net_pattern = re.compile(r'-\s+(\S+)\s+((?:\(\s*\S+\s+\S+\s*\)\s*)+)')
    pin_pattern = re.compile(r'\(\s*(\S+)\s+(\S+)\s*\)')

    for m in net_pattern.finditer(nets_text):
        net_name = m.group(1)
        pins_text = m.group(2)
        instances = []
        for pm in pin_pattern.finditer(pins_text):
            inst_name = pm.group(1)
            if inst_name != "PIN":  # skip I/O pins
                instances.append(inst_name)
        if len(instances) >= 2:
            nets.append({"name": net_name, "instances": instances})

    print(f"  Parsed {len(nets)} nets (of {n_nets} declared)")
    return nets


# ═══════════════════════════════════════════════════════════════════
# Net-based Constraint Graph
# ═══════════════════════════════════════════════════════════════════

def build_netlist_graph(nets, cell_name_to_idx, max_net_degree=20):
    """Build constraint graph from netlist.

    For each net with k cells:
    - If k <= max_net_degree: add all k*(k-1)/2 pairwise edges (clique)
    - If k > max_net_degree: use star decomposition (connect all to first cell)

    Returns list of (i, j) edges (deduplicated).
    """
    edge_set = set()

    for net in nets:
        # Map instance names to cell indices
        idxs = []
        for inst in net["instances"]:
            if inst in cell_name_to_idx:
                idxs.append(cell_name_to_idx[inst])
        idxs = list(set(idxs))  # deduplicate

        if len(idxs) < 2:
            continue

        if len(idxs) <= max_net_degree:
            # Clique decomposition
            for i, j in combinations(sorted(idxs), 2):
                edge_set.add((i, j))
        else:
            # Star decomposition (connect all to first)
            hub = idxs[0]
            for spoke in idxs[1:]:
                edge_set.add((min(hub, spoke), max(hub, spoke)))

    edges = sorted(edge_set)
    return edges


# ═══════════════════════════════════════════════════════════════════
# Per-G-cell Net-based η
# ═══════════════════════════════════════════════════════════════════

def compute_net_eta_per_gcell(positions, widths, heights, die_area,
                              net_edges, gs, n_v=2, max_cells_svd=500):
    """Compute per-G-cell η using netlist graph."""
    N = len(positions)
    x_min, y_min = die_area["x_min"], die_area["y_min"]
    x_max, y_max = die_area["x_max"], die_area["y_max"]
    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs

    # Assign cells to G-cells
    gcell_cells = defaultdict(list)
    gcell_assign = {}
    for idx in range(N):
        gx = min(max(0, int((positions[idx, 0] - x_min) / gcell_w)), gs - 1)
        gy = min(max(0, int((positions[idx, 1] - y_min) / gcell_h)), gs - 1)
        gcell_cells[(gx, gy)].append(idx)
        gcell_assign[idx] = (gx, gy)

    eta_map = np.zeros((gs, gs))
    dbar_map = np.zeros((gs, gs))
    ncells_map = np.zeros((gs, gs), dtype=int)
    nedges_map = np.zeros((gs, gs), dtype=int)

    for gx in range(gs):
        for gy in range(gs):
            cell_idxs = gcell_cells.get((gx, gy), [])
            nc = len(cell_idxs)
            ncells_map[gy, gx] = nc
            if nc < 2:
                continue

            idx_set = set(cell_idxs)
            old_to_new = {old: new for new, old in enumerate(sorted(idx_set))}

            # Local edges from netlist (both endpoints in this G-cell)
            local_edges = [(old_to_new[i], old_to_new[j])
                           for i, j in net_edges
                           if i in idx_set and j in idx_set]

            nE = len(local_edges)
            nedges_map[gy, gx] = nE
            if nE == 0:
                continue

            dbar_local = 2.0 * nE / nc
            dbar_map[gy, gx] = dbar_local

            local_pos = positions[sorted(idx_set)]

            if nc <= max_cells_svd and nE <= 5000:
                delta = build_overlap_coboundary(local_pos, local_edges, n_v=n_v)
                sv = svd(delta, compute_uv=False)
                rank = int(np.sum(sv > 1e-10))
                eta_map[gy, gx] = (nE - rank) / nE if nE > 0 else 0.0
            else:
                eta_map[gy, gx] = theory_eta(dbar_local, n_v)

    return eta_map, dbar_map, ncells_map, nedges_map, gcell_assign


# ═══════════════════════════════════════════════════════════════════
# Main Analysis
# ═══════════════════════════════════════════════════════════════════

def analyze_design(design_name, gs=8):
    """Full pipeline: parse nets → build graph → compute η → correlate with GT."""
    print(f"\n{'='*70}")
    print(f"NET-BASED η: {design_name}")
    print(f"{'='*70}")

    cfg = DESIGNS[design_name]
    def_path = cfg["def"]

    # 1. Load design
    positions, widths, heights, die_area, _ = load_design(design_name)
    if positions is None:
        return None
    N = len(positions)
    med_w = np.median(widths)
    med_h = np.median(heights)
    diag = np.sqrt(med_w**2 + med_h**2)

    # Build cell name → index mapping
    cells, metadata = parse_def_components(def_path)
    # Filter same way as load_design
    cell_sizes = parse_lef_macros(cfg["lef"])
    logic_cells = []
    for c in cells:
        if c.is_filler or c.is_tap:
            continue
        if c.macro in cell_sizes:
            c.width, c.height = cell_sizes[c.macro]
        c.compute_center()
        logic_cells.append(c)

    cell_name_to_idx = {c.name: i for i, c in enumerate(logic_cells)}
    print(f"  Cells: {N}, cell_name_to_idx: {len(cell_name_to_idx)}")

    # 2. Parse nets
    nets = parse_def_nets(def_path)

    # 3. Build netlist graph
    net_edges = build_netlist_graph(nets, cell_name_to_idx)
    net_dbar = 2.0 * len(net_edges) / N if N > 0 else 0
    print(f"  Netlist graph: {len(net_edges)} edges, Δ̄={net_dbar:.2f}")

    # 4. Also build proximity graph for comparison
    r_prox = diag * 3.0
    tree = KDTree(positions)
    prox_pairs = tree.query_pairs(r=r_prox)
    prox_edges = [(min(i, j), max(i, j)) for i, j in prox_pairs]
    prox_dbar = 2.0 * len(prox_edges) / N
    print(f"  Proximity graph (r=3d): {len(prox_edges)} edges, Δ̄={prox_dbar:.2f}")

    # 5. Compute per-G-cell η for both graphs
    x_min, y_min = die_area["x_min"], die_area["y_min"]
    x_max, y_max = die_area["x_max"], die_area["y_max"]
    gcell_w = (x_max - x_min) / gs
    gcell_h = (y_max - y_min) / gs

    # Net-based η
    eta_net, dbar_net, ncells_map, nedges_net, gcell_assign = \
        compute_net_eta_per_gcell(positions, widths, heights, die_area,
                                  net_edges, gs)

    # Proximity-based η
    from eta_shield_placement import compute_gcell_metrics
    _, eta_prox, _, dbar_prox, _, _ = compute_gcell_metrics(
        positions, widths, heights, die_area, gs, r_prox)

    # Cell density
    cell_density = np.zeros((gs, gs))
    for idx in range(N):
        gx = min(max(0, int((positions[idx, 0] - x_min) / gcell_w)), gs - 1)
        gy = min(max(0, int((positions[idx, 1] - y_min) / gcell_h)), gs - 1)
        cell_density[gy, gx] += widths[idx] * heights[idx] / (gcell_w * gcell_h)

    # 6. Load independent GT
    gt_path = os.path.join(RESULTS_DIR,
                           f"gcell_overflow_congested_{design_name}.json")
    if not os.path.exists(gt_path):
        print(f"  No independent GT found at {gt_path}")
        print(f"  Run extract_gcell_overflow first.")
        return None

    gt_data = json.load(open(gt_path))
    dbu = metadata.get("units", 2000)

    # Aggregate GT GCells into our grid
    gt_grid = np.zeros((gs, gs))
    x_grids = gt_data['x_grids']
    y_grids = gt_data['y_grids']
    for gc in gt_data['gcells']:
        grt_x = (x_grids[gc['gx']] + x_grids[min(gc['gx'] + 1, gt_data['grid_nx'])]) / 2.0 / dbu
        grt_y = (y_grids[gc['gy']] + y_grids[min(gc['gy'] + 1, gt_data['grid_ny'])]) / 2.0 / dbu
        our_gx = min(max(0, int((grt_x - x_min) / gcell_w)), gs - 1)
        our_gy = min(max(0, int((grt_y - y_min) / gcell_h)), gs - 1)
        gt_grid[our_gy, our_gx] += gc['usage']

    # 7. Regression
    mask = ncells_map.ravel() >= 2
    n_active = int(mask.sum())
    if n_active < 5:
        print(f"  Not enough active G-cells ({n_active})")
        return None

    y = gt_grid.ravel()[mask]
    if np.std(y) < 1e-10:
        print("  No variance in GT")
        return None

    cd = cell_density.ravel()[mask].reshape(-1, 1)
    e_net = eta_net.ravel()[mask].reshape(-1, 1)
    e_prox = eta_prox.ravel()[mask].reshape(-1, 1)

    fold = np.array([i % 2 == 0 for i in range(n_active)])

    def cv_r2(X, y, fold):
        scores = []
        for tr, te in [(fold, ~fold), (~fold, fold)]:
            if tr.sum() < 2 or te.sum() < 2:
                return 0.0
            reg = LinearRegression().fit(X[tr], y[tr])
            scores.append(r2_score(y[te], reg.predict(X[te])))
        return float(np.mean(scores))

    r2_cd = cv_r2(cd, y, fold)
    r2_net = cv_r2(e_net, y, fold) if np.std(e_net) > 1e-10 else 0.0
    r2_prox = cv_r2(e_prox, y, fold) if np.std(e_prox) > 1e-10 else 0.0
    r2_net_cd = cv_r2(np.hstack([e_net, cd]), y, fold) if np.std(e_net) > 1e-10 else r2_cd
    r2_prox_cd = cv_r2(np.hstack([e_prox, cd]), y, fold) if np.std(e_prox) > 1e-10 else r2_cd
    r2_all = cv_r2(np.hstack([e_net, e_prox, cd]), y, fold) \
        if np.std(e_net) > 1e-10 and np.std(e_prox) > 1e-10 else r2_net_cd

    print(f"\n  === RESULTS (gs={gs}, {n_active} active G-cells) ===")
    print(f"  {'Predictor':>30s}  {'CV-R²':>8s}  {'Δ vs density':>12s}")
    print(f"  {'─'*55}")
    print(f"  {'cell density':>30s}  {r2_cd:>8.4f}  {'baseline':>12s}")
    print(f"  {'η_net (netlist) alone':>30s}  {r2_net:>8.4f}  {r2_net-r2_cd:>+12.4f}")
    print(f"  {'η_prox (proximity) alone':>30s}  {r2_prox:>8.4f}  {r2_prox-r2_cd:>+12.4f}")
    print(f"  {'η_net + density':>30s}  {r2_net_cd:>8.4f}  {r2_net_cd-r2_cd:>+12.4f}")
    print(f"  {'η_prox + density':>30s}  {r2_prox_cd:>8.4f}  {r2_prox_cd-r2_cd:>+12.4f}")
    print(f"  {'η_net + η_prox + density':>30s}  {r2_all:>8.4f}  {r2_all-r2_cd:>+12.4f}")

    net_vs_prox = r2_net_cd - r2_prox_cd
    print(f"\n  η_net vs η_prox (both + density): {net_vs_prox:+.4f}")
    if net_vs_prox > 0.005:
        print(f"  ✓ NET-BASED η WINS over proximity-based η")
    elif net_vs_prox < -0.005:
        print(f"  ✗ Proximity-based η wins")
    else:
        print(f"  ≈ About the same")

    # Stats
    print(f"\n  Net-based: η_max={float(np.max(eta_net)):.4f}, "
          f"mean Δ̄={float(np.mean(dbar_net[ncells_map>=2])):.2f}, "
          f"edges/G-cell={float(np.mean(nedges_net[ncells_map>=2])):.1f}")
    print(f"  Prox-based: η_max={float(np.max(eta_prox)):.4f}, "
          f"mean Δ̄={float(np.mean(dbar_prox[ncells_map>=2])):.2f}")

    output = {
        "design": design_name,
        "gs": gs,
        "n_active": n_active,
        "n_nets": len(nets),
        "n_net_edges": len(net_edges),
        "net_dbar": float(net_dbar),
        "n_prox_edges": len(prox_edges),
        "prox_dbar": float(prox_dbar),
        "cv_r2_density": float(r2_cd),
        "cv_r2_eta_net_alone": float(r2_net),
        "cv_r2_eta_prox_alone": float(r2_prox),
        "cv_r2_eta_net_density": float(r2_net_cd),
        "cv_r2_eta_prox_density": float(r2_prox_cd),
        "cv_r2_all": float(r2_all),
        "net_improvement": float(r2_net_cd - r2_cd),
        "prox_improvement": float(r2_prox_cd - r2_cd),
        "net_vs_prox": float(net_vs_prox),
    }

    json_path = os.path.join(RESULTS_DIR, f"net_eta_{design_name}.json")
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Saved: {json_path}")

    return output


def main():
    parser = argparse.ArgumentParser(description="Net-based η")
    parser.add_argument("--design", type=str, default=None)
    parser.add_argument("--gs", type=int, default=8)
    args = parser.parse_args()

    if args.design:
        designs = [args.design]
    else:
        designs = ["gcd_nangate45", "gcd_sky130", "aes_nangate45"]

    all_results = {}
    for design in designs:
        # Check if independent GT exists
        gt_path = os.path.join(RESULTS_DIR,
                               f"gcell_overflow_congested_{design}.json")
        if not os.path.exists(gt_path):
            print(f"\n  Skipping {design}: no independent GT")
            continue

        result = analyze_design(design, gs=args.gs)
        if result:
            all_results[design] = result

    if len(all_results) > 1:
        print(f"\n{'='*70}")
        print("CROSS-DESIGN: NET vs PROXIMITY η")
        print(f"{'='*70}")
        print(f"  {'Design':>20s}  {'density':>8s}  {'η_net+d':>8s}  "
              f"{'η_prox+d':>8s}  {'net wins?':>9s}")
        print(f"  {'─'*60}")
        for design, r in all_results.items():
            wins = "✓" if r["net_vs_prox"] > 0.005 else \
                   "✗" if r["net_vs_prox"] < -0.005 else "≈"
            print(f"  {design:>20s}  {r['cv_r2_density']:>8.4f}  "
                  f"{r['cv_r2_eta_net_density']:>8.4f}  "
                  f"{r['cv_r2_eta_prox_density']:>8.4f}  {wins:>9s}")


if __name__ == "__main__":
    main()
