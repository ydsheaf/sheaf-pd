#!/usr/bin/env python3
"""
Analog η: Sheaf cohomology on analog layout constraints (Issue #30)
====================================================================

Extracts constraints from ALIGN benchmark .const.json files,
builds the constraint sheaf, and computes η.

Constraint types → sheaf edges:
  SymmetricBlocks: pairs must be mirror-symmetric → n_e = 2 (x-mirror, y-match)
  GroupBlocks: devices placed together → n_e = 1 (proximity)
  SameTemplate: identical W/L → n_e = 2 (ΔW=0, ΔL=0)
  Order: ordering constraint → n_e = 1 (directional)
  Align: alignment → n_e = 1 (coordinate match)

Vertex stalk: n_v = 2 (x, y position) — minimum DOF per device
"""

import json
import os
import sys
import numpy as np
from collections import defaultdict
from itertools import combinations
from scipy.linalg import svd

ALIGN = "/mnt/storage1/users/ydwu/claude_projects/ALIGN"
EXAMPLES = os.path.join(ALIGN, "examples")


def load_constraints(design_dir):
    """Load constraint JSON from ALIGN example directory."""
    const_files = [f for f in os.listdir(design_dir)
                   if f.endswith('.const.json')]
    if not const_files:
        return None, None

    design_name = os.path.basename(design_dir)

    all_constraints = []
    for cf in const_files:
        with open(os.path.join(design_dir, cf)) as f:
            constraints = json.load(f)
        all_constraints.extend(constraints)

    return design_name, all_constraints


def load_spice_devices(design_dir):
    """Parse SPICE netlist to find device instances."""
    sp_files = [f for f in os.listdir(design_dir)
                if f.endswith('.sp') or f.endswith('.spice')]
    if not sp_files:
        return []

    devices = []
    for sf in sp_files:
        with open(os.path.join(design_dir, sf)) as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('*') or line.startswith('.'):
                    continue
                # MOSFET: M<name> drain gate source bulk model ...
                if line.upper().startswith('M'):
                    parts = line.split()
                    if len(parts) >= 6:
                        devices.append(parts[0].lower())

    return list(set(devices))


def extract_constraint_graph(constraints, devices=None):
    """Build constraint graph from ALIGN constraints.

    Returns:
        vertices: list of device/block names
        edges: list of (i, j, constraint_type, n_e)
        n_e_per_edge: edge stalk dimension for each edge
    """
    # Collect all named instances
    all_instances = set()
    if devices:
        all_instances.update(d.lower() for d in devices)

    # Parse constraints to find instances
    for c in constraints:
        ctype = c.get("constraint", "")
        if ctype in ("GroupBlocks", "SameTemplate"):
            for inst in c.get("instances", []):
                all_instances.add(inst.lower())
            if "instance_name" in c:
                all_instances.add(c["instance_name"].lower())
        elif ctype == "SymmetricBlocks":
            for pair in c.get("pairs", []):
                for inst in pair:
                    all_instances.add(inst.lower())
        elif ctype in ("Order", "Align", "AlignInOrder"):
            for inst in c.get("instances", []):
                all_instances.add(inst.lower())

    vertices = sorted(all_instances)
    v_to_idx = {v: i for i, v in enumerate(vertices)}

    edges = []

    for c in constraints:
        ctype = c.get("constraint", "")

        if ctype == "SymmetricBlocks":
            # Each pair in SymmetricBlocks creates a symmetry constraint
            # Pairs: [[a], [b, c]] means b,c are symmetric pair, a is on axis
            for pair in c.get("pairs", []):
                if len(pair) == 2:
                    # Symmetric pair: (a, b) must be mirror-symmetric
                    a, b = pair[0].lower(), pair[1].lower()
                    if a in v_to_idx and b in v_to_idx:
                        edges.append((v_to_idx[a], v_to_idx[b],
                                      "symmetry", 2))  # n_e=2: x-mirror + y-match
                elif len(pair) == 1:
                    # On-axis: self-symmetry (position constrained to axis)
                    pass  # No edge needed

        elif ctype == "GroupBlocks":
            # All instances in group must be proximate
            instances = [inst.lower() for inst in c.get("instances", [])
                         if inst.lower() in v_to_idx]
            for a, b in combinations(instances, 2):
                edges.append((v_to_idx[a], v_to_idx[b],
                              "group", 1))  # n_e=1: proximity

        elif ctype == "SameTemplate":
            # Matching: same W and L
            instances = [inst.lower() for inst in c.get("instances", [])
                         if inst.lower() in v_to_idx]
            for a, b in combinations(instances, 2):
                edges.append((v_to_idx[a], v_to_idx[b],
                              "matching", 2))  # n_e=2: ΔW=0, ΔL=0

        elif ctype in ("Order", "AlignInOrder"):
            # Sequential ordering: each consecutive pair has an order constraint
            instances = [inst.lower() for inst in c.get("instances", [])
                         if inst.lower() in v_to_idx]
            for k in range(len(instances) - 1):
                edges.append((v_to_idx[instances[k]], v_to_idx[instances[k + 1]],
                              "order", 1))  # n_e=1: directional

        elif ctype == "Align":
            # Alignment: all instances share a coordinate
            instances = [inst.lower() for inst in c.get("instances", [])
                         if inst.lower() in v_to_idx]
            for k in range(len(instances) - 1):
                edges.append((v_to_idx[instances[k]], v_to_idx[instances[k + 1]],
                              "align", 1))  # n_e=1: coordinate match

    return vertices, edges


def compute_analog_eta(vertices, edges, n_v=2):
    """Compute η for analog constraint graph.

    Builds coboundary with mixed edge stalk dimensions.
    """
    N = len(vertices)
    nE = len(edges)

    if nE == 0 or N < 2:
        return {"eta": 0, "rank": 0, "nE": 0, "N": N,
                "dim_C0": n_v * N, "dim_C1": 0}

    # Total C1 dimension (sum of edge stalk dims)
    dim_C1 = sum(ne for _, _, _, ne in edges)
    dim_C0 = n_v * N

    # Build coboundary δ: C0 → C1
    # For each edge (i, j, type, n_e):
    #   δ maps (s_i, s_j) ∈ R^{n_v} × R^{n_v} to R^{n_e}
    #   via restriction maps ρ_{i→e} and ρ_{j→e}

    delta = np.zeros((dim_C1, dim_C0))
    row_offset = 0

    rng = np.random.default_rng(42)

    for i, j, ctype, n_e in edges:
        if ctype == "symmetry":
            # Symmetry about vertical axis:
            # Constraint 1: x_i + x_j = 2*x_center (x-mirror)
            # Constraint 2: y_i - y_j = 0 (y-match)
            # ρ_{i→e} = [[1, 0], [0, 1]], ρ_{j→e} = [[1, 0], [0, -1]]
            # δ = ρ_i s_i - ρ_j s_j
            delta[row_offset, i * n_v] = 1      # x_i
            delta[row_offset, j * n_v] = 1      # + x_j (mirror: sum = const)
            delta[row_offset + 1, i * n_v + 1] = 1  # y_i
            delta[row_offset + 1, j * n_v + 1] = -1  # - y_j (match: diff = 0)

        elif ctype == "matching":
            # Matching: ΔW = 0, ΔL = 0
            # With n_v=2 (position only), matching adds constraints
            # beyond position. Use random restriction maps for genericity.
            delta[row_offset, i * n_v:i * n_v + n_v] = rng.standard_normal(n_v)
            delta[row_offset, j * n_v:j * n_v + n_v] = -delta[row_offset, i * n_v:i * n_v + n_v]
            delta[row_offset + 1, i * n_v:i * n_v + n_v] = rng.standard_normal(n_v)
            delta[row_offset + 1, j * n_v:j * n_v + n_v] = -delta[row_offset + 1, i * n_v:i * n_v + n_v]

        elif ctype == "group":
            # Proximity: ||p_i - p_j|| < R
            # ρ = (p_i - p_j) / ||p_i - p_j|| (direction)
            direction = rng.standard_normal(n_v)
            direction /= np.linalg.norm(direction)
            delta[row_offset, i * n_v:i * n_v + n_v] = direction
            delta[row_offset, j * n_v:j * n_v + n_v] = -direction

        elif ctype == "order":
            # Order: y_i > y_j (top-to-bottom)
            delta[row_offset, i * n_v + 1] = 1   # y_i
            delta[row_offset, j * n_v + 1] = -1  # - y_j

        elif ctype == "align":
            # Align: y_i = y_j (horizontal alignment)
            delta[row_offset, i * n_v + 1] = 1
            delta[row_offset, j * n_v + 1] = -1

        row_offset += n_e

    # SVD to compute rank
    sv = svd(delta, compute_uv=False)
    rank = int(np.sum(sv > 1e-10))
    dim_H1 = dim_C1 - rank
    eta = dim_H1 / dim_C1 if dim_C1 > 0 else 0

    # Theory prediction
    avg_ne = dim_C1 / nE if nE > 0 else 1
    dbar = 2 * nE / N
    eta_theory = max(0, 1 - 2 * n_v / (avg_ne * dbar))

    # Constraint type breakdown
    type_counts = defaultdict(int)
    for _, _, ctype, _ in edges:
        type_counts[ctype] += 1

    return {
        "eta": float(eta),
        "eta_theory": float(eta_theory),
        "rank": int(rank),
        "nE": int(nE),
        "N": int(N),
        "dim_C0": int(dim_C0),
        "dim_C1": int(dim_C1),
        "dim_H1": int(dim_H1),
        "dbar": float(dbar),
        "avg_ne": float(avg_ne),
        "type_counts": dict(type_counts),
    }


def analyze_design(design_dir):
    """Full analysis of one ALIGN example."""
    design_name, constraints = load_constraints(design_dir)
    if constraints is None:
        return None

    # Filter out power/ground/clock (not layout constraints)
    layout_constraints = [c for c in constraints
                          if c.get("constraint") not in
                          ("PowerPorts", "GroundPorts", "ClockPorts",
                           "ConfigureCompiler", "AspectRatio",
                           "MultiConnection", "GuardRing",
                           "HorizontalDistance", "VerticalDistance")]

    if not layout_constraints:
        return None

    devices = load_spice_devices(design_dir)
    vertices, edges = extract_constraint_graph(layout_constraints, devices)

    if len(edges) == 0:
        return None

    result = compute_analog_eta(vertices, edges, n_v=2)
    result["design"] = design_name
    result["n_constraints_raw"] = len(constraints)
    result["n_layout_constraints"] = len(layout_constraints)
    result["vertices"] = vertices

    return result


def main():
    print("=" * 70)
    print("ANALOG η: Sheaf Cohomology on ALIGN Benchmark Constraints")
    print("=" * 70)

    results = []

    # Scan all examples
    for design_dir in sorted(os.listdir(EXAMPLES)):
        full_path = os.path.join(EXAMPLES, design_dir)
        if not os.path.isdir(full_path):
            continue

        result = analyze_design(full_path)
        if result is None:
            continue

        results.append(result)
        eta = result["eta"]
        eta_th = result["eta_theory"]
        print(f"  {result['design']:>45s}  "
              f"V={result['N']:>3d}  E={result['nE']:>3d}  "
              f"Δ̄={result['dbar']:>5.2f}  "
              f"η={eta:>6.4f}  η_th={eta_th:>6.4f}  "
              f"H¹={result['dim_H1']:>3d}/{result['dim_C1']:>3d}  "
              f"{result['type_counts']}")

    print(f"\n{'=' * 70}")
    print(f"SUMMARY: {len(results)} designs analyzed")
    print(f"{'=' * 70}")

    n_feasible = sum(1 for r in results if r["eta"] == 0)
    n_infeasible = sum(1 for r in results if r["eta"] > 0)

    print(f"  Feasible (η=0): {n_feasible}")
    print(f"  Infeasible (η>0): {n_infeasible}")

    if n_infeasible > 0:
        print(f"\n  Infeasible designs:")
        for r in results:
            if r["eta"] > 0:
                print(f"    {r['design']}: η={r['eta']:.4f}, "
                      f"Δ̄={r['dbar']:.2f}, H¹={r['dim_H1']}/{r['dim_C1']}")
                print(f"      Constraints: {r['type_counts']}")
                print(f"      → {r['dim_H1']} constraints are structurally unsatisfiable")

    # Save
    out_path = os.path.join(os.path.dirname(__file__), "results", "analog")
    os.makedirs(out_path, exist_ok=True)
    json_path = os.path.join(out_path, "analog_eta_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Saved: {json_path}")


if __name__ == "__main__":
    main()
