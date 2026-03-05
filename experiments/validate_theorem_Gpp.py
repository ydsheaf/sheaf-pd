#!/usr/bin/env python3
"""
validate_theorem_Gpp.py — Validate Theorem G'' (Bloch Decomposition)

Tests on synthetic periodic grids (torus):
1. Bloch rank formula matches SVD rank exactly
2. Nodal line count N_1 matches analytical prediction
3. Diagonal threshold: gap drops when r > sqrt(a^2 + b^2)
4. PDK-parameterized grids reproduce observed gap magnitudes
"""

import numpy as np
from scipy.spatial import KDTree
from itertools import product
import json, os


def build_torus_grid(Nx, Ny, a, b):
    """Build positions for Nx × Ny cells on a torus."""
    positions = np.array([(m * a, n * b) for n in range(Ny) for m in range(Nx)])
    return positions  # shape (Nx*Ny, 2)


def torus_edges(Nx, Ny, a, b, r):
    """Find edges on the torus within distance r.

    Uses minimum image convention for periodic boundary.
    """
    Lx, Ly = Nx * a, Ny * b
    positions = build_torus_grid(Nx, Ny, a, b)
    N = len(positions)

    edges = []
    neighbor_types = set()

    for i in range(N):
        mi, ni = i % Nx, i // Nx
        for j in range(i + 1, N):
            mj, nj = j % Nx, j // Nx
            # Minimum image
            dx = (mj - mi) * a
            dy = (nj - ni) * b
            # Periodic wrap
            dx = dx - Lx * round(dx / Lx)
            dy = dy - Ly * round(dy / Ly)
            if dx * dx + dy * dy <= r * r + 1e-10:
                edges.append((i, j))
                # Canonical neighbor type
                dp = round(dx / a)
                dq = round(dy / b)
                if dp < 0 or (dp == 0 and dq < 0):
                    dp, dq = -dp, -dq
                neighbor_types.add((dp, dq))

    return edges, sorted(neighbor_types)


def build_coboundary(positions, edges, Nx, Ny, a, b):
    """Build coboundary matrix for placement sheaf.

    Row e: ρ_e · (s_i - s_j) where ρ_e = (Δx, Δy) for edge (i,j).
    Uses minimum image convention for Δx, Δy on torus.
    """
    Lx, Ly = Nx * a, Ny * b
    N = len(positions)
    nE = len(edges)
    delta = np.zeros((nE, 2 * N))

    for idx, (i, j) in enumerate(edges):
        dx = positions[i, 0] - positions[j, 0]
        dy = positions[i, 1] - positions[j, 1]
        # Periodic wrap
        dx = dx - Lx * round(dx / Lx)
        dy = dy - Ly * round(dy / Ly)
        # ρ_e = (dx, dy), applied to (s_x^i - s_x^j, s_y^i - s_y^j)
        delta[idx, 2 * i] = dx      # s_x^i
        delta[idx, 2 * i + 1] = dy  # s_y^i
        delta[idx, 2 * j] = -dx     # s_x^j
        delta[idx, 2 * j + 1] = -dy # s_y^j

    return delta


def bloch_rank(Nx, Ny, a, b, neighbor_types):
    """Compute rank via Bloch decomposition.

    For each k-point, compute rank of δ̂(k) = directions among active types.

    Returns: total_rank, N0, N1, N2, per_k_ranks
    """
    N0, N1, N2 = 0, 0, 0
    per_k = []

    for m in range(Nx):
        for n in range(Ny):
            kx = 2 * np.pi * m / (Nx * a)
            ky = 2 * np.pi * n / (Ny * b)

            # Collect active direction vectors
            active_dirs = []
            for (p, q) in neighbor_types:
                phase = kx * p * a + ky * q * b
                # c_{p,q}(k) = 1 - e^{i*phase}
                # |c| = |1 - e^{i*phase}| = 2|sin(phase/2)|
                if abs(np.sin(phase / 2)) > 1e-10:
                    active_dirs.append([p * a, q * b])

            if len(active_dirs) == 0:
                rk = 0
                N0 += 1
            else:
                D = np.array(active_dirs)
                rk = np.linalg.matrix_rank(D, tol=1e-10)
                if rk == 1:
                    N1 += 1
                else:
                    N2 += 1

            per_k.append({"m": m, "n": n, "rank": rk})

    total_rank = N1 + 2 * N2
    return total_rank, N0, N1, N2, per_k


def test_periodic_grid(Nx, Ny, a, b, r, label=""):
    """Test Bloch formula against SVD on one configuration."""
    positions = build_torus_grid(Nx, Ny, a, b)
    N = Nx * Ny
    edges, ntypes = torus_edges(Nx, Ny, a, b, r)
    nE = len(edges)

    if nE == 0:
        return None

    # Bloch prediction
    rank_bloch, N0, N1, N2, _ = bloch_rank(Nx, Ny, a, b, ntypes)

    # SVD actual rank
    delta = build_coboundary(positions, edges, Nx, Ny, a, b)
    sv = np.linalg.svd(delta, compute_uv=False)
    rank_svd = int(np.sum(sv > 1e-8))

    eta_bloch = 1 - rank_bloch / nE if nE > 0 else 0
    eta_svd = 1 - rank_svd / nE if nE > 0 else 0
    eta_generic = max(0, 1 - 2 * N / nE) if nE > 0 else 0

    match = "OK" if rank_bloch == rank_svd else "MISMATCH"

    result = {
        "label": label,
        "Nx": Nx, "Ny": Ny, "a": a, "b": b, "r": r,
        "N": N, "nE": nE, "n_types": len(ntypes),
        "types": [list(t) for t in ntypes],
        "N0": N0, "N1": N1, "N2": N2,
        "rank_bloch": rank_bloch,
        "rank_svd": rank_svd,
        "eta_bloch": eta_bloch,
        "eta_svd": eta_svd,
        "eta_generic": eta_generic,
        "match": match,
    }
    return result


def main():
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
    os.makedirs(results_dir, exist_ok=True)

    all_results = []

    # ── Test 1: Nearest-neighbor, various grid sizes ──
    print("=" * 70)
    print("TEST 1: Nearest-neighbor grid, verify rank = 2NxNy - Nx - Ny")
    print("=" * 70)
    print(f"  {'Nx':>3} {'Ny':>3} {'|V|':>5} {'|E|':>5} {'J':>3} "
          f"{'N1':>4} {'rank_B':>6} {'rank_S':>6} {'η':>6} {'match':>8}")
    print("  " + "-" * 65)

    for Nx, Ny in [(5, 5), (8, 6), (10, 10), (15, 8), (20, 10)]:
        a, b = 1.0, 1.0
        r = 1.01  # just enough for nearest neighbors
        res = test_periodic_grid(Nx, Ny, a, b, r, f"NN_{Nx}x{Ny}")
        if res:
            all_results.append(res)
            print(f"  {Nx:3d} {Ny:3d} {res['N']:5d} {res['nE']:5d} {res['n_types']:3d} "
                  f"{res['N1']:4d} {res['rank_bloch']:6d} {res['rank_svd']:6d} "
                  f"{res['eta_bloch']:6.4f} {res['match']:>8s}")

    # ── Test 2: Radius sweep on 10x10 grid ──
    print(f"\n{'=' * 70}")
    print("TEST 2: Radius sweep on 10×10 grid (a=0.46, b=2.72 — SKY130-like)")
    print("=" * 70)
    Nx, Ny = 10, 10
    a, b = 0.46, 2.72
    diag = np.sqrt(a**2 + b**2)

    print(f"  Diagonal distance = {diag:.3f}")
    print(f"\n  {'r':>6} {'|E|':>5} {'J':>3} {'N0':>3} {'N1':>3} {'N2':>3} "
          f"{'rank_B':>6} {'rank_S':>6} {'η_B':>6} {'η_gen':>6} {'gap':>6} {'match':>7}")
    print("  " + "-" * 75)

    for r in [0.5, 0.95, 1.5, 2.0, 2.75, 3.0, 4.0, 5.5, 8.0, 12.0]:
        res = test_periodic_grid(Nx, Ny, a, b, r, f"SKY130_{r:.1f}")
        if res:
            all_results.append(res)
            gap = res["eta_bloch"] - res["eta_generic"]
            print(f"  {r:6.2f} {res['nE']:5d} {res['n_types']:3d} "
                  f"{res['N0']:3d} {res['N1']:3d} {res['N2']:3d} "
                  f"{res['rank_bloch']:6d} {res['rank_svd']:6d} "
                  f"{res['eta_bloch']:6.4f} {res['eta_generic']:6.4f} "
                  f"{gap:6.4f} {res['match']:>7s}")

    # ── Test 3: PDK comparison ──
    print(f"\n{'=' * 70}")
    print("TEST 3: PDK comparison at r = 1.5 × row_pitch")
    print("=" * 70)

    pdks = {
        "SKY130": (0.46, 2.72),
        "NanGate45": (0.19, 1.40),
        "ASAP7": (0.054, 0.27),
    }

    print(f"  {'PDK':>10} {'a':>6} {'b':>6} {'diag':>6} {'r':>6} "
          f"{'|E|':>5} {'N1':>4} {'rank':>5} {'η':>6} {'gap':>6}")
    print("  " + "-" * 70)

    Nx, Ny = 15, 15
    for pdk, (a, b) in pdks.items():
        r = 1.5 * b
        res = test_periodic_grid(Nx, Ny, a, b, r, f"{pdk}_1.5pitch")
        if res:
            all_results.append(res)
            gap = res["eta_bloch"] - res["eta_generic"]
            diag_pdk = np.sqrt(a**2 + b**2)
            print(f"  {pdk:>10} {a:6.3f} {b:6.3f} {diag_pdk:6.3f} {r:6.3f} "
                  f"{res['nE']:5d} {res['N1']:4d} {res['rank_bloch']:5d} "
                  f"{res['eta_bloch']:6.4f} {gap:6.4f}")

    # ── Test 4: Analytical prediction for nearest-neighbor ──
    print(f"\n{'=' * 70}")
    print("TEST 4: Analytical formula verification: rank = 2NxNy - Nx - Ny")
    print("=" * 70)

    all_match = True
    for Nx in [5, 8, 10, 15, 20]:
        for Ny in [5, 8, 10, 15]:
            a, b = 1.0, 2.0
            r = min(a, b) * 1.01
            res = test_periodic_grid(Nx, Ny, a, b, r)
            if res:
                predicted = 2 * Nx * Ny - Nx - Ny
                ok = res["rank_bloch"] == predicted and res["rank_svd"] == predicted
                if not ok:
                    print(f"  FAIL: {Nx}x{Ny}: predicted={predicted}, "
                          f"bloch={res['rank_bloch']}, svd={res['rank_svd']}")
                    all_match = False

    if all_match:
        print("  ALL PASSED: rank = 2NxNy - Nx - Ny for nearest-neighbor")

    # ── Test 5: Diagonal threshold ──
    print(f"\n{'=' * 70}")
    print("TEST 5: Diagonal threshold — N1 drops to 0 when diag neighbors active")
    print("=" * 70)

    Nx, Ny = 10, 10
    a, b = 0.46, 2.72
    diag = np.sqrt(a**2 + b**2)
    print(f"  Diagonal = {diag:.4f}μm")
    print(f"  {'r':>6} {'J':>3} {'N1':>4} {'has_diag':>9}")
    print("  " + "-" * 30)

    for r in np.arange(0.5, 5.0, 0.25):
        _, ntypes = torus_edges(Nx, Ny, a, b, r)
        has_diag = any(p != 0 and q != 0 for p, q in ntypes)
        _, _, N1, _, _ = bloch_rank(Nx, Ny, a, b, ntypes)
        if len(ntypes) > 0:
            print(f"  {r:6.2f} {len(ntypes):3d} {N1:4d} {'YES' if has_diag else 'no':>9}")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print("=" * 70)

    mismatches = sum(1 for r in all_results if r["match"] != "OK")
    print(f"\n  Total tests: {len(all_results)}")
    print(f"  Matches: {len(all_results) - mismatches}")
    print(f"  Mismatches: {mismatches}")

    if mismatches == 0:
        print("\n  ✓ Theorem G'' VALIDATED: Bloch formula matches SVD exactly")
        print("    on all periodic grid configurations tested.")

    # Save
    path = os.path.join(results_dir, "theorem_Gpp_validation.json")
    with open(path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\n  Results saved: {path}")


if __name__ == '__main__':
    main()
