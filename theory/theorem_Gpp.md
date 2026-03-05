# Theorem G'' (Bloch Decomposition of Placement Coboundary)

Derives an exact rank formula for the placement sheaf coboundary on
periodic grids, connecting sheaf cohomology with solid-state band theory.

---

## 1. Setup: Periodic Grid

Consider N_x × N_y cells on a rectangular lattice (torus Z_{N_x} × Z_{N_y}):
- Site width a, row pitch b
- Cell (m,n) at position (m·a, n·b)
- Interaction radius r determines neighbor types

### Neighbor types

    J = {(p, q) : p²a² + q²b² ≤ r², (p,q) ≠ (0,0)} / ±

(undirected: identify (p,q) with (-p,-q)). Each type has N_x·N_y edges
on the torus.

### Coboundary

For edge e connecting (m,n) to (m+p, n+q):

    (δs)_e = (p·a)(s_x^{m,n} − s_x^{m+p,n+q}) + (q·b)(s_y^{m,n} − s_y^{m+p,n+q})

This is the R^2 → R^1 restriction map ρ_e = (p·a, q·b) applied to the
stalk difference.

---

## 2. Bloch Transform

The torus Z_{N_x} × Z_{N_y} has character group (dual torus) with
N_x · N_y characters:

    k = (2πm/(N_x·a), 2πn/(N_y·b)),  m = 0,...,N_x−1,  n = 0,...,N_y−1

### Fourier ansatz

    s_x^{m,n} = ŝ_x(k) · exp(i(k_x·m·a + k_y·n·b))
    s_y^{m,n} = ŝ_y(k) · exp(i(k_x·m·a + k_y·n·b))

### Fiber coboundary

Substituting into the kernel equation:

    (δ̂(k)ŝ)^{(p,q)} = (1 − e^{iφ(k,p,q)}) · [(p·a)ŝ_x + (q·b)ŝ_y]

where φ(k,p,q) = k_x·p·a + k_y·q·b is the Bloch phase.

**Key factorization.** Each row of δ̂(k) factors as:

    row_{(p,q)}(k) = c_{p,q}(k) · (p·a, q·b)

where c_{p,q}(k) = 1 − e^{iφ(k,p,q)} is a scalar phase factor.

---

## 3. Rank Formula

### Theorem G'' (Bloch Rank Decomposition)

For the periodic placement coboundary on Z_{N_x} × Z_{N_y}:

    rank(δ) = Σ_k rank(δ̂(k))

where:

    rank(δ̂(k)) = dim span{(p·a, q·b) : c_{p,q}(k) ≠ 0}

That is, the rank at each k-point equals the number of linearly
independent direction vectors among the ACTIVE neighbor types (those
whose Bloch phase ≠ 0 mod 2π).

### Proof sketch

The Fourier transform block-diagonalizes δ into N_x·N_y independent
fiber problems. Each fiber δ̂(k) is a |J|×2 matrix (one row per
undirected neighbor type, two columns for x,y DOFs).

Row (p,q) of δ̂(k) = c_{p,q}(k) · (p·a, q·b).

Since direction vectors in R^2 can span at most 2 dimensions:

    rank(δ̂(k)) ∈ {0, 1, 2}

The total rank is the sum over k-points because the Fourier blocks are
independent (direct sum).  ∎

### Corollary: Explicit counting

Classify k-points by their rank:

    N_0 = #{k : all c_{p,q}(k) = 0}                    (rank 0)
    N_1 = #{k : active directions span only R^1}         (rank 1)
    N_2 = #{k : active directions span R^2}              (rank 2)

Then:

    rank(δ) = 0·N_0 + 1·N_1 + 2·N_2
    N_0 + N_1 + N_2 = N_x · N_y

Always: N_0 = 1 (only k = 0 kills all phases).

    rank(δ) = N_1 + 2N_2 = N_1 + 2(N_x·N_y − 1 − N_1)
            = 2|V| − 2 − N_1

**The rank deficiency equals N_1 + 2: one rank-1 deficit per
k-point on the "nodal lines", plus 2 for the zero mode.**

---

## 4. Nearest-Neighbor Case (Worked Example)

### Setup

Neighbor types: (1,0) and (0,1) (horizontal and vertical only).
Directions: (a, 0) and (0, b) — always linearly independent.

### Activity conditions

- (1,0) active ⟺ k_x·a ≠ 0 mod 2π ⟺ m ≠ 0
- (0,1) active ⟺ k_y·b ≠ 0 mod 2π ⟺ n ≠ 0

### Classification

    k = (0, 0):        both inactive → rank 0    (1 point)
    k_x = 0, k_y ≠ 0:  only (0,1) → rank 1      (N_y − 1 points)
    k_x ≠ 0, k_y = 0:  only (1,0) → rank 1      (N_x − 1 points)
    k_x ≠ 0, k_y ≠ 0:  both active → rank 2     ((N_x−1)(N_y−1) points)

### Result

    N_1 = (N_x − 1) + (N_y − 1) = N_x + N_y − 2

    rank(δ) = 2|V| − 2 − (N_x + N_y − 2) = 2N_x N_y − N_x − N_y

    |E| = 2N_x N_y  (on torus: N_x N_y horizontal + N_x N_y vertical)

    η = 1 − rank/|E| = (N_x + N_y) / (2N_x N_y)

For N_x = N_y = N:

    η = 1/N  →  0 as N → ∞

**Note:** This requires r ≥ max(a, b) so that BOTH horizontal and
vertical neighbors are within range. When r < b (only horizontal
neighbors active): rank = N_y(N_x − 1) and η = 1 − (N_x − 1)/(N_x N_y).
This single-direction regime produces massive gap.

**Validated:** Bloch rank matches SVD on all tested grid sizes (5×5 to
20×15). The rank-1 k-points form "nodal lines" k_x = 0 and k_y = 0
in the Brillouin zone.

---

## 5. With Diagonal Neighbors

### Setup

Add neighbor types (1,1) and (1,−1). Directions: (a,b) and (a,−b).

### Key observation

At k_x = 0, k_y ≠ 0: type (0,1) has direction (0,b), type (1,1) has
direction (a,b) — linearly independent! So rank = 2.

At k_x ≠ 0, k_y = 0: type (1,0) has direction (a,0), type (1,1) has
direction (a,b) — linearly independent! So rank = 2.

### Result

    N_1 = 0,  rank(δ) = 2|V| − 2

**Diagonal neighbors eliminate all rank-1 k-points.** The gap vanishes
(η → 2/|E| → 0).

**Physical interpretation:** Diagonal neighbors provide "off-axis"
constraint directions that fill in the nodal lines of the Brillouin zone.

---

## 6. General Formula: Nodal Lines

### Definition

For neighbor types J = {(p_j, q_j)}, define the **nodal set** at k:

    Null(k) = {j ∈ J : c_{p_j,q_j}(k) = 0}

and the **active direction set**:

    D(k) = {(p_j·a, q_j·b) : j ∉ Null(k)}

Then rank(δ̂(k)) = dim span D(k).

### Nodal line condition

c_{p,q}(k) = 0 ⟺ k_x·p·a + k_y·q·b = 0 mod 2π

Each neighbor type (p,q) defines a "nodal line" in the Brillouin zone
where it is inactive. The rank drops at k-points where all active
neighbor directions become collinear.

### For row-based placement (Source I connection)

Intra-row types: (p, 0) for various p. Their nodal lines are all
{k_y : any} when k_x = 0, i.e., the SAME nodal line. This is Source I
(row collinearity) in Fourier language: all horizontal neighbors become
inactive simultaneously.

The rank-1 points are precisely {k : k_x = 0, k_y ≠ 0} when
the only non-horizontal neighbor type is (0,1). Adding diagonal
neighbors breaks this degeneracy.

---

## 7. Application to Standard Cell Placement

### Non-uniform grids

Real placements have non-uniform cell widths. The exact Bloch
decomposition requires perfect periodicity, but we can use it as
a **predictor** for the periodic component of rank loss:

    rank_deficiency ≈ rank_deficiency_periodic + boundary_correction

For large grids (N_x, N_y ≫ 1), the boundary correction is O(√|V|)
while the periodic part is O(|V|), so the Bloch formula dominates.

### PDK dependence from Bloch theory

The key parameter is: **how many neighbor types are active, and do
they span R^2?**

For interaction radius r:
- Number of neighbor types |J| ≈ π r² / (a·b)
- If r > √(a² + b²) (diagonal distance): diagonal types included → N_1 = 0

| PDK | a (site) | b (pitch) | diag √(a²+b²) | Effect |
|-----|----------|-----------|----------------|--------|
| SKY130 | 0.46μm | 2.72μm | 2.76μm | Need r > 2.76 for diag |
| NanGate45 | 0.19μm | 1.40μm | 1.41μm | Need r > 1.41 for diag |
| ASAP7 | 0.054μm | 0.27μm | 0.275μm | Need r > 0.275 for diag |

**SKY130 has the largest diagonal distance** → at moderate r (comparable
to row pitch), only horizontal/vertical neighbors exist → large N_1 →
large gap. This explains why SKY130 shows the biggest grid quantization
effect.

### Connection to Theorem G

Theorem G captures Source I (k_x = 0 nodal line, N_y − 1 rank-1 points).
Theorem G'' captures all three sources via the full nodal line structure.

    gap_G   = (N_y − 1 + ... ) / |E|   (collinearity only)
    gap_G'' = N_1 / |E|                 (all nodal lines)

---

## 8. Experimental Validation

### Bloch vs SVD (synthetic periodic grids)

All tests: Bloch rank = SVD rank (exact match, 18/18 configurations).

**SKY130-like grid (10×10, a=0.46, b=2.72):**

| r (μm) | |E| | J types | N_1 | rank | η | gap | Phase |
|---------|-----|---------|-----|------|------|------|-------|
| 0.50 | 100 | 1 | 90 | 90 | 0.100 | 0.100 | H only |
| 0.95 | 200 | 2 | 90 | 90 | 0.550 | 0.550 | H only |
| 2.00 | 400 | 4 | 90 | 90 | 0.775 | 0.275 | H only |
| 2.75 | 550 | 6 | 18 | 180 | 0.673 | 0.036 | H+V |
| 3.00 | 950 | 10 | 0 | 198 | 0.792 | 0.002 | diag |

**Three phases:**
1. **r < b (row pitch)**: Only horizontal neighbors → N_1 = N_x·N_y − N_y = 90 → gap LARGE
2. **b < r < √(a²+b²)**: H + V neighbors → N_1 = N_x + N_y − 2 = 18 → gap moderate
3. **r > √(a²+b²)**: Diagonal neighbors → N_1 = 0 → gap ≈ 0

**Key insight:** The SKY130 gap at moderate r comes entirely from being
in Phase 1 (only horizontal neighbors). The transition is SHARP.

### Diagonal threshold

N_1 drops from 90 → 18 → 0 at r = 2.72μm (row pitch) and r = 2.76μm
(diagonal distance). The 0.04μm window is where "row awareness" kicks in.

---

## 9. Testable Predictions

### G''-P1: Periodic grid rank

On a synthetic N_x × N_y torus with site width a, row pitch b,
and interaction radius r, the Bloch formula gives the EXACT rank.

### G''-P2: Nodal line count predicts gap

For real placements, approximate the gap as:

    gap ≈ N_1 / |E|

where N_1 is computed from the periodic approximation using the
median cell width as site width.

### G''-P3: Diagonal threshold

The gap drops sharply when r crosses √(a² + b²) (diagonal distance),
because diagonal neighbor types eliminate the rank-1 nodal lines.

### G''-P4: η formula for periodic grids

For nearest-neighbor interactions on N × N torus:

    η = 1/N

For beyond-nearest-neighbor (r > diagonal):

    η = 2/(|J| · N²)  →  0

---

## 10. Solid-State Analogy

| Solid state | Placement sheaf |
|-------------|-----------------|
| Crystal lattice Z^d | Standard cell grid Z_{N_x} × Z_{N_y} |
| Hamiltonian H(k) | Fiber coboundary δ̂(k) |
| Bloch wavevector k | Character of Z_{N_x} × Z_{N_y} |
| Band structure E_n(k) | Singular values σ_i(k) of δ̂(k) |
| Band gap | rank(δ̂(k)) < 2 (rank deficiency) |
| Nodal surface | {k : det(δ̂(k)^T δ̂(k)) = 0} |
| Density of states | Distribution of σ values |
| Topological insulator | Sheaf with protected rank deficiency |

The rank-1 k-points form "nodal lines" in the Brillouin zone, analogous
to the nodal surfaces of band Hamiltonians. The total rank deficiency
(= gap) is determined by the measure of these nodal sets.

---

## 11. Why This Beats RUDY

RUDY (Rectangular Uniform wire DensitY) estimates congestion by counting
nets passing through each tile. It captures |E| (edge count = density)
but NOT the direction structure.

The Bloch formula captures:

1. **Direction diversity**: how many independent constraint directions
   exist at each k-point. RUDY counts edges; we count INDEPENDENT edges.

2. **Grid-scale periodicity**: the specific PDK grid parameters (a, b)
   determine which k-points have rank deficiency. RUDY is grid-agnostic.

3. **Transition radius**: r* = √(a² + b²) marks where diagonal neighbors
   activate. For r < r*, the effective dimension is < 2 (→ more constraints
   needed). RUDY doesn't model this transition.

**Prediction:** η outperforms RUDY specifically for:
- Coarse-grid PDKs (SKY130 > NanGate45 > ASAP7)
- Moderate interaction radius (r ≈ row pitch)
- Designs with high row utilization (many intra-row edges)

For fine-grid PDKs or large r, η → η_generic ≈ RUDY, and there's
no advantage. The value-add is in the coarse-grid, moderate-r regime
where RUDY's density assumption breaks down.
