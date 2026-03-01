#!/usr/bin/env python3
"""
Verify hierarchical sheaf composition theorems.

Tests:
1. Partition rank decomposition: Σ rank(δ_i) ≤ rank(δ) ≤ Σ rank(δ_i) + |E_cross|
2. Recursive η: η_residual ≈ η^L after L levels
3. Commander count formula
4. Hexagonal tiling: dual graph has Δ̄ ≈ 6 at every level
"""

import torch
import numpy as np
import time
import json
import sys

sys.path.insert(0, '.')
from theory.gpu_sweep import (
    build_proximity_graph_chunked,
    build_coboundary_gpu_fast,
    compute_h1_gpu,
)


def partition_agents(positions, k, method='voronoi'):
    """Partition N agents into k groups.

    method='voronoi': k random centers, assign by nearest.
    method='grid': regular grid partition.
    """
    N = positions.shape[0]
    d = positions.shape[1]

    if method == 'voronoi':
        # Random Voronoi centers
        rng = np.random.default_rng(123)
        centers = rng.uniform(0, 1, (k, d))
        # Assign each agent to nearest center
        dists = np.linalg.norm(
            positions[:, None, :] - centers[None, :, :], axis=2
        )
        labels = np.argmin(dists, axis=1)
    elif method == 'grid':
        # Grid partition
        cells_per_dim = int(np.ceil(k ** (1.0/d)))
        coords = np.floor(positions * cells_per_dim).astype(int)
        coords = np.clip(coords, 0, cells_per_dim - 1)
        if d == 2:
            labels = coords[:, 0] * cells_per_dim + coords[:, 1]
        else:
            labels = coords[:, 0]
        # Remap to 0..k-1
        unique = np.unique(labels)
        remap = {v: i for i, v in enumerate(unique)}
        labels = np.array([remap[l] for l in labels])
    else:
        raise ValueError(f"Unknown method: {method}")

    return labels


def verify_partition_decomposition(N, d, r_prox, n_v, k, device, seed=42):
    """Verify: Σ rank(δ_i) ≤ rank(δ) ≤ Σ rank(δ_i) + |E_cross|."""
    rng = np.random.default_rng(seed)
    pos_np = rng.uniform(0, 1, (N, d)).astype(np.float32)
    vel_np = (rng.standard_normal((N, d)) * 0.1).astype(np.float32)

    positions = torch.from_numpy(pos_np).to(device)
    velocities = torch.from_numpy(vel_np).to(device)

    # Full graph
    edges = build_proximity_graph_chunked(positions, r_prox)
    nE = edges.shape[0]
    if nE == 0:
        return None

    # Full coboundary and rank
    delta_full = build_coboundary_gpu_fast(
        positions, velocities, edges, d, n_v, cbf_type="velocity"
    )
    _, rank_full = compute_h1_gpu(delta_full)
    eta_full = (nE - rank_full) / nE

    # Partition
    labels = partition_agents(pos_np, k, method='voronoi')

    edges_np = edges.cpu().numpy()
    edge_labels_src = labels[edges_np[:, 0]]
    edge_labels_tgt = labels[edges_np[:, 1]]
    is_internal = edge_labels_src == edge_labels_tgt
    is_cross = ~is_internal
    n_cross = int(is_cross.sum())

    sum_rank = 0
    group_results = []

    for g in range(k):
        mask_v = labels == g
        idx_v = np.where(mask_v)[0]
        if len(idx_v) < 2:
            continue

        # Internal edges for this group
        mask_e = is_internal & (edge_labels_src == g)
        idx_e = np.where(mask_e)[0]
        if len(idx_e) == 0:
            group_results.append({
                'group': g, 'nV': len(idx_v), 'nE': 0,
                'rank': 0, 'eta': 0, 'avg_deg': 0
            })
            continue

        # Remap vertex indices
        old_to_new = {old: new for new, old in enumerate(idx_v)}
        group_edges_np = edges_np[idx_e]
        group_edges_remapped = np.array([
            [old_to_new[e[0]], old_to_new[e[1]]]
            for e in group_edges_np
        ])
        group_edges = torch.from_numpy(group_edges_remapped).to(device)

        group_pos = positions[idx_v]
        group_vel = velocities[idx_v]

        delta_g = build_coboundary_gpu_fast(
            group_pos, group_vel, group_edges, d, n_v, cbf_type="velocity"
        )
        h1_g, rank_g = compute_h1_gpu(delta_g)
        nE_g = len(idx_e)
        eta_g = h1_g / nE_g if nE_g > 0 else 0

        sum_rank += rank_g
        group_results.append({
            'group': g, 'nV': len(idx_v), 'nE': nE_g,
            'rank': rank_g, 'eta': float(eta_g),
            'avg_deg': 2.0 * nE_g / len(idx_v)
        })

        del delta_g, group_edges, group_pos, group_vel

    del delta_full, positions, velocities, edges
    torch.cuda.empty_cache()

    # Verify bounds
    lb_ok = sum_rank <= rank_full
    ub_ok = rank_full <= sum_rank + n_cross

    n_internal = nE - n_cross
    # Weighted average η
    eta_weighted = sum(
        r['nE'] * r['eta'] for r in group_results
    ) / n_internal if n_internal > 0 else 0

    return {
        'N': N, 'k': k, 'nE': nE, 'n_internal': n_internal, 'n_cross': n_cross,
        'cross_fraction': n_cross / nE,
        'rank_full': rank_full, 'sum_rank_groups': sum_rank,
        'rank_gap': rank_full - sum_rank,
        'lb_ok': lb_ok, 'ub_ok': ub_ok,
        'eta_full': float(eta_full),
        'eta_weighted_groups': float(eta_weighted),
        'group_results': group_results,
    }


def verify_recursive_eta(N, d, r_prox, n_v, device, n_levels=4, k_per_level=10, seed=42):
    """Verify that recursive partitioning gives η_residual ≈ η^L."""
    rng = np.random.default_rng(seed)
    pos_np = rng.uniform(0, 1, (N, d)).astype(np.float32)
    vel_np = (rng.standard_normal((N, d)) * 0.1).astype(np.float32)

    positions = torch.from_numpy(pos_np).to(device)
    velocities = torch.from_numpy(vel_np).to(device)

    edges = build_proximity_graph_chunked(positions, r_prox)
    nE = edges.shape[0]
    if nE == 0:
        return None

    delta = build_coboundary_gpu_fast(
        positions, velocities, edges, d, n_v, cbf_type="velocity"
    )
    _, rank = compute_h1_gpu(delta)
    eta_0 = (nE - rank) / nE

    del delta
    torch.cuda.empty_cache()

    results = [{
        'level': 0, 'N': N, 'nE': nE, 'eta': float(eta_0),
        'predicted_residual': float(eta_0),
    }]

    # For higher levels, we simulate the dual graph
    # Each level's "agents" are the commanders from previous level
    # Their proximity graph is the dual of the Voronoi partition
    current_N = N
    current_pos = pos_np.copy()
    current_vel = vel_np.copy()
    eta_product = eta_0

    for level in range(1, n_levels):
        # Partition into k groups
        k = min(k_per_level, current_N // 3)
        if k < 3:
            break

        labels = partition_agents(current_pos, k, method='voronoi')

        # Commander positions = group centroids
        new_pos = np.zeros((k, d), dtype=np.float32)
        new_vel = np.zeros((k, d), dtype=np.float32)
        for g in range(k):
            mask = labels == g
            if mask.sum() > 0:
                new_pos[g] = current_pos[mask].mean(axis=0)
                new_vel[g] = current_vel[mask].mean(axis=0)

        # Commander proximity graph
        # r_prox for commanders: scale by sqrt(k/N) to maintain similar Δ̄
        r_cmd = r_prox * np.sqrt(current_N / k)
        r_cmd = min(r_cmd, 0.5)  # cap

        cmd_positions = torch.from_numpy(new_pos).to(device)
        cmd_velocities = torch.from_numpy(new_vel).to(device)

        cmd_edges = build_proximity_graph_chunked(cmd_positions, r_cmd)
        nE_cmd = cmd_edges.shape[0]
        if nE_cmd == 0:
            break

        avg_deg_cmd = 2.0 * nE_cmd / k

        delta_cmd = build_coboundary_gpu_fast(
            cmd_positions, cmd_velocities, cmd_edges, d, n_v, cbf_type="velocity"
        )
        _, rank_cmd = compute_h1_gpu(delta_cmd)
        eta_cmd = (nE_cmd - rank_cmd) / nE_cmd

        eta_product *= eta_cmd

        results.append({
            'level': level,
            'N': k,
            'nE': nE_cmd,
            'avg_deg': float(avg_deg_cmd),
            'eta': float(eta_cmd),
            'eta_product': float(eta_product),
            'predicted_residual': float(eta_0 ** (level + 1)),
        })

        del delta_cmd, cmd_positions, cmd_velocities, cmd_edges
        torch.cuda.empty_cache()

        current_N = k
        current_pos = new_pos
        current_vel = new_vel

    return results


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if device.type == 'cuda':
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    d = 2
    n_v = 2 * d  # = 4

    # =====================================================
    # Test 1: Partition rank decomposition
    # =====================================================
    print(f"\n{'='*80}")
    print("  TEST 1: Partition Rank Decomposition")
    print(f"{'='*80}")
    print(f"  Verify: Σ rank(δ_i) ≤ rank(δ) ≤ Σ rank(δ_i) + |E_cross|")
    print()

    test_configs = [
        (200, 6, 4),   # N=200, Δ̄≈6, k=4
        (200, 6, 10),  # N=200, Δ̄≈6, k=10
        (200, 6, 20),  # N=200, Δ̄≈6, k=20
        (500, 6, 5),   # N=500, Δ̄≈6, k=5
        (500, 6, 25),  # N=500, Δ̄≈6, k=25
        (500, 8, 10),  # N=500, Δ̄≈8, k=10
        (1000, 6, 10), # N=1000, Δ̄≈6, k=10
        (1000, 6, 50), # N=1000, Δ̄≈6, k=50
    ]

    print(f"{'N':>6} {'k':>4} {'|E|':>7} {'|E_int|':>7} {'|E_x|':>6} "
          f"{'ε_x':>6} {'Σrank':>7} {'rank':>7} {'gap':>5} "
          f"{'η_full':>7} {'η_wt':>7} {'bounds':>7}")
    print("-" * 90)

    all_partition_results = []
    for N, target_deg, k in test_configs:
        r_prox = np.sqrt(target_deg / (np.pi * N))
        n_seeds = 5
        for seed in range(n_seeds):
            res = verify_partition_decomposition(
                N, d, r_prox, n_v, k, device, seed=seed * 1000 + N
            )
            if res is None:
                continue

            status = "OK" if (res['lb_ok'] and res['ub_ok']) else "FAIL"
            print(f"{N:>6} {k:>4} {res['nE']:>7} {res['n_internal']:>7} "
                  f"{res['n_cross']:>6} {res['cross_fraction']:>5.1%} "
                  f"{res['sum_rank_groups']:>7} {res['rank_full']:>7} "
                  f"{res['rank_gap']:>5} "
                  f"{res['eta_full']:>6.1%} {res['eta_weighted_groups']:>6.1%} "
                  f"{status:>7}", flush=True)
            all_partition_results.append(res)

    # Summary
    n_ok = sum(1 for r in all_partition_results if r['lb_ok'] and r['ub_ok'])
    n_total = len(all_partition_results)
    print(f"\nBounds verified: {n_ok}/{n_total}")

    # Cross-edge fraction vs k
    print(f"\n  Cross-edge fraction vs k (N=500, Δ̄≈6):")
    for r in all_partition_results:
        if r['N'] == 500:
            print(f"    k={r['k']:>3}: ε_cross = {r['cross_fraction']:.1%}, "
                  f"rank_gap = {r['rank_gap']} "
                  f"({r['rank_gap']/r['n_cross']:.1%} of |E_cross| used)")

    # =====================================================
    # Test 2: Recursive η
    # =====================================================
    print(f"\n{'='*80}")
    print("  TEST 2: Recursive η (Hierarchical Command)")
    print(f"{'='*80}")
    print()

    Ns = [500, 1000, 2000]
    target_deg = 6
    n_seeds = 5

    for N in Ns:
        r_prox = np.sqrt(target_deg / (np.pi * N))
        print(f"\n  N = {N}, Δ̄ ≈ {target_deg}:")
        print(f"  {'Level':>5} {'N_ℓ':>6} {'|E_ℓ|':>6} {'Δ̄_ℓ':>5} "
              f"{'η_ℓ':>7} {'Π η':>8} {'η₀^L':>8} {'match':>6}")
        print("  " + "-" * 65)

        for seed in range(n_seeds):
            levels = verify_recursive_eta(
                N, d, r_prox, n_v, device,
                n_levels=4, k_per_level=max(10, N // 50),
                seed=seed * 1000 + N
            )
            if levels is None:
                continue

            for lv in levels:
                pred = lv.get('predicted_residual', lv['eta'])
                actual = lv.get('eta_product', lv['eta'])
                match = "~" if abs(actual - pred) / max(pred, 1e-6) < 0.5 else "X"
                avg_deg = lv.get('avg_deg', target_deg)
                print(f"  {lv['level']:>5} {lv['N']:>6} {lv['nE']:>6} "
                      f"{avg_deg:>5.1f} "
                      f"{lv['eta']:>6.1%} "
                      f"{actual:>7.2%} {pred:>7.2%} "
                      f"{match:>6}", flush=True)
            if seed < n_seeds - 1:
                print("  " + "." * 65)

    # =====================================================
    # Test 3: Commander count scaling
    # =====================================================
    print(f"\n{'='*80}")
    print("  TEST 3: Commander Count Scaling")
    print(f"{'='*80}")
    print()

    # Theoretical predictions for different scales
    scenarios = [
        ("Indoor show", 100, 6, 50, 3e8, 0.01),
        ("City block", 1000, 6, 200, 3e8, 0.01),
        ("Military op", 10000, 6, 1000, 3e8, 0.1),
        ("LEO constellation", 1e6, 6, 1e4, 3e8, 1.0),
        ("Dyson sphere", 1e18, 6, 1e12, 3e8, 1.0),
    ]

    print(f"{'Scenario':>20} {'N':>10} {'η|E|':>12} {'C_η(B=100)':>12} "
          f"{'C_geom':>12} {'C_total':>12} {'Levels':>8}")
    print("-" * 90)

    eta = 0.05  # η ≈ 5% at Δ̄=6
    for name, N, dbar, R_system, c_s, dt in scenarios:
        nE = N * dbar / 2
        eta_constraints = eta * nE

        # η-based commander count
        B = 100  # constraints per commander per timestep
        C_eta = eta_constraints / B

        # Geometry-based (light-cone)
        # For d=2 surface: C_geom = Area / (π R_cmd²)
        R_cmd = c_s * dt
        if R_system > 0:
            C_geom = (R_system / R_cmd) ** 2 if R_cmd < R_system else 1
        else:
            C_geom = 1

        C_total = max(C_eta, C_geom)

        # Levels needed for η_residual < 10⁻⁶
        if eta > 0 and eta < 1:
            levels = int(np.ceil(np.log(1e-6) / np.log(eta)))
        else:
            levels = 1

        print(f"{name:>20} {N:>10.0e} {eta_constraints:>12.2e} "
              f"{C_eta:>12.2e} {C_geom:>12.2e} {C_total:>12.2e} {levels:>8}")

    # =====================================================
    # Save results
    # =====================================================
    outfile = "theory/hierarchical_results.json"
    save_data = {
        'partition_tests': [{
            k: v for k, v in r.items() if k != 'group_results'
        } for r in all_partition_results],
    }
    with open(outfile, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    print(f"\nResults saved to {outfile}")


if __name__ == "__main__":
    main()
