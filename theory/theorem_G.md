# Theorem G: Row-Based Genericity Gap

Proves that row-based standard cell placement has a nonzero genericity
gap, and characterizes three sources of rank deficiency in the placement
coboundary.

**Connection to rigidity theory.** Theorem G is a Laman-style counting
argument (Laman 1970) adapted to the sheaf setting: the rank of the
coboundary is bounded by a combinatorial count on the constraint graph,
with the specific structure of row-based placement reducing rank below
the generic Laman bound.

---

## 1. Setup

Let G = (V, E) be the constraint graph of a standard cell placement
with |V| cells arranged in rows. The placement sheaf has:

- Vertex stalks: F(v) = R^2 (position DOFs)
- Edge stalks: F(e) = R^1 (scalar overlap constraint)
- Restriction maps: ρ_e = 2(p_i − p_j)^T ∈ R^{1×2}

Partition the edge set:

    E = E_intra ∪ E_inter

where:
- E_intra = {(i,j) ∈ E : y_i = y_j} (same-row edges)
- E_inter = {(i,j) ∈ E : y_i ≠ y_j} (cross-row edges)

Let c = number of connected components of the subgraph (V, E_intra).

---

## 2. Three Sources of Rank Deficiency

### Source I: Row collinearity

For an intra-row edge e = (i,j) with y_i = y_j:

    ρ_e = 2(x_i − x_j, 0)^T

The y-component is identically zero. Intra-row edges only constrain
x-displacement differences, wasting their y-component.

### Source II: Quantized row y-coordinates

Cells are placed in m discrete rows with fixed y-coordinates
{Y_1, ..., Y_m}. Inter-row edges between rows k and l all share
the same Δy = Y_l − Y_k. This creates repeated structure in the
restriction maps:

    ρ_e = 2(x_i − x_j, Y_l − Y_k)^T

With only m(m−1)/2 distinct Δy values, inter-row restriction maps
are far from general position.

### Source III: Quantized x-grid

Standard cells are placed on a grid with step Δx_grid (the site
width). Cell x-coordinates are integer multiples of Δx_grid. This
quantizes the Δx component of restriction maps, creating additional
linear dependencies when multiple edges share the same (Δx, Δy).

---

## 3. Theorem G (Row Collinearity Upper Bound)

**Assumptions.**

(G1) Within each row, cells have distinct x-coordinates:
     for all (i,j) ∈ E_intra, x_i ≠ x_j.

(G2) Inter-row edges connect cells at distinct heights:
     for all (i,j) ∈ E_inter, y_i ≠ y_j.

(G3) The inter-row constraint equations are **transversal** to the
     intra-row kernel: for each inter-row edge e = (i,j) with i in
     component α, j in component β, the equation
     (x_i−x_j)(c_α−c_β) + (y_i−y_j)(s_y^i−s_y^j) = 0
     is not in the span of previous inter-row equations (up to
     min(|E_inter|, c+|V|) equations).

     Note: (G3) fails for grid-quantized placements where cell
     coordinates are integer multiples of site width. In that case,
     the formula below is an upper bound (see Section 6).

**Statement.** Under (G1)-(G3):

    rank(δ) = (|V| − c) + min(|E_inter|, c + |V|)

In general (without G3), this is an **upper bound**:

    rank(δ) ≤ (|V| − c) + min(|E_inter|, c + |V|)

**Corollary (Collinearity Gap Lower Bound).**

    gap_actual ≥ gap_collinear = max(0, c + |V| − |E_inter|) / |E|

Equality holds when (G3) is satisfied. For grid-quantized placements,
gap_actual > gap_collinear due to Sources II and III.

---

## 4. Proof

### Step 1: Kernel characterization

A vector s = (s_x, s_y) ∈ R^{2|V|} is in ker(δ) iff for every edge
e = (i,j):

    (x_i − x_j)(s_x^i − s_x^j) + (y_i − y_j)(s_y^i − s_y^j) = 0

### Step 2: Intra-row constraints fix x-displacements

For (i,j) ∈ E_intra, y_i = y_j, so:

    (x_i − x_j)(s_x^i − s_x^j) = 0

By (G1), x_i ≠ x_j, so s_x^i = s_x^j for all intra-row connected
pairs. Therefore s_x is constant on each connected component of
(V, E_intra): c free parameters.

The y-displacements s_y are unconstrained by intra-row edges.

    rank_intra = |V| − c

### Step 3: Residual DOF space

After intra-row constraints, the residual kernel is (c + |V|)-dimensional:
- c_1, ..., c_c ∈ R (x-constant per component)
- s_y^1, ..., s_y^{|V|} ∈ R (free y-displacements)

### Step 4: Inter-row constraints (Schur complement argument)

Partition the coboundary matrix by edge type:

    δ = [δ_intra]    (|E_intra| × 2|V|)
        [δ_inter]    (|E_inter| × 2|V|)

From Step 2, δ_intra has rank |V| − c and its kernel K_intra is
(c + |V|)-dimensional (c x-constants + |V| y-DOFs).

The total rank satisfies:

    rank(δ) = rank(δ_intra) + rank(δ_inter|_{K_intra})

where δ_inter|_{K_intra} is the restriction of δ_inter to the
kernel of δ_intra (the Schur complement). This is the (|E_inter|) ×
(c + |V|) matrix whose rows are the inter-row equations in the
residual DOF coordinates (c_α, c_β, s_y^i, s_y^j).

For (i,j) ∈ E_inter, i in component α, j in component β:

    (δ_inter|_{K_intra})_e = (x_i − x_j)(c_α − c_β) + (y_i − y_j)(s_y^i − s_y^j)

Under (G3) (transversality), this matrix has rank min(|E_inter|, c + |V|).

### Step 5: Total

    rank(δ) = rank(δ_intra) + rank(δ_inter|_{K_intra})
            = (|V| − c) + min(|E_inter|, c + |V|)  ∎

The Schur complement structure ensures additivity: the intra-row
and inter-row contributions operate on complementary subspaces.

---

## 5. Theorem G' (Grid Quantization Rank Loss)

Theorem G assumes (G3): inter-row maps in general position. In real
standard cell placements, (G3) fails due to Sources II and III. This
section quantifies the additional rank loss.

### Setup

Let the placement have:
- m rows at y-coordinates Y_1, ..., Y_m
- Site width Δx_grid (x-coordinates are multiples of Δx_grid)

### Distinct direction vectors

For each edge e = (i,j), the restriction map direction is:

    d_e = (p_i − p_j) / ‖p_i − p_j‖ ∈ S^1

With quantized coordinates, the set of distinct directions is finite.
Let D = {d_1, ..., d_K} be the set of distinct (Δx, Δy) vectors
(up to scaling) among all edges. Group edges by direction:

    E_k = {e ∈ E : d_e = d_k}

### Scalar sheaf decomposition

Each direction d_k = (a_k, b_k) defines a **projection** π_k : R^2 → R^1
by π_k(s_x, s_y) = a_k·s_x + b_k·s_y. Under this projection, all edges
in group E_k have scalar restriction maps:

    ρ_e^{(k)} = λ_e ∈ R^{1×1}    (scalar, since direction is fixed)

This defines a **scalar sheaf F_k** on the subgraph (V_k, E_k):
- Vertex stalks: F_k(v) = R^1 (the projected displacement π_k(s_v))
- Edge stalks: F_k(e) = R^1
- Restriction maps: scalar multiples λ_e

The rank contribution from group k is the rank of this scalar sheaf's
coboundary, which equals |V_k| − c_k (vertices minus components) by
standard 1D sheaf theory (Theorem H with n_v = n_e = 1).

**Interpretation.** Theorem G' decomposes the R^2-valued placement
sheaf into K independent R^1-valued scalar sheaves, one per direction.
The total rank is bounded by the sum of scalar sheaf ranks.

**Saturation.** Each vertex v has 2 DOFs (s_x^v, s_y^v). At most 2
linearly independent projections can constrain it. If vertex v appears
in K_v > 2 direction groups, only 2 of them contribute independent
information. The overcounting is:

    overcount = Σ_v max(0, K_v − 2)

where K_v = number of direction groups incident to vertex v. This
explains why rank_Gp > rank_actual for large r (many directions per
vertex).

### Theorem G' (Direction-Based Rank Bound)

    rank(δ) ≤ Σ_k (|V_k| − c_k)

where the sum runs over distinct direction groups k = 1, ..., K, V_k is
the vertex set of edges in group k, and c_k is the number of connected
components of (V_k, E_k).

**Caveat: vertex sharing.** This bound overcounts when direction groups
share vertices. Each vertex has 2 DOFs; once two independent directions
constrain it, additional groups add nothing. The bound is therefore
tighter than Theorem G but still overestimates rank when many groups
overlap at shared vertices. See experimental validation below.

### Corollary (Grid Gap)

    gap_grid = 1 − Σ_k (|V_k| − c_k) / |E|

This is strictly positive whenever many edges share directions (large
|E_k| relative to |V_k|).

---

## 6. Experimental Validation

### Data

Three designs tested, sweeping interaction radius r:

| Design | PDK | |V| | Rows | Cell aspect |
|--------|-----|-----|------|-------------|
| gcd_preroute | NanGate45 | 357 | 42 | 1.84 (tall) |
| gcd_sky130 | SKY130 | 370 | 68 | 0.84 (square) |
| aes_preroute | NanGate45 | 1500* | 85 | 1.34 (tall) |

*subsampled from 16591

### Key finding: Theorem G is an upper bound, not exact

gcd_sky130 at representative radii:

| r (μm) | |E| | rank_G | rank_actual | error | η_actual | η_generic |
|---------|-----|--------|-------------|-------|----------|-----------|
| 3.22 | 82 | 82 | 82 | 0 | 0.000 | 0.000 |
| 6.44 | 453 | 453 | 392 | −61 | 0.135 | 0.000 |
| 8.43 | 733 | 730 | 466 | −264 | 0.364 | 0.000 |
| 9.66 | 966 | 740 | 509 | −231 | 0.473 | 0.234 |
| 12.65 | 1626 | 740 | 588 | −152 | 0.638 | 0.545 |
| 42.15 | 13954 | 740 | 734 | −6 | 0.947 | 0.947 |

**Observations:**

1. **rank_actual < rank_G** at all radii where |E| is moderate. Theorem G
   overestimates rank because (G3) fails — inter-row maps are NOT in
   general position due to grid quantization (Sources II + III).

2. **The gap is LARGE at moderate radii.** At r=8.43 (Δ̄ ≈ 4):
   η_actual = 0.364 vs η_generic = 0.000. The generic formula predicts
   NO deficiency, but 36% of constraints are rank-deficient.

3. **The gap closes at large radii.** At r=42.15: rank_actual ≈ 740 ≈ 2|V|.
   With enough edges from diverse directions, the rank approaches generic.

4. **Matches PS1 observation:** gcd_sky130 R²(n_v=1) = 0.955 > R²(n_v=2)
   = 0.886. The effective n_v is between 1 and 2, consistent with the
   partial rank deficiency observed.

### Rank deficiency decomposition

The total rank deficiency decomposes as:

    rank_generic − rank_actual = (rank_generic − rank_G) + (rank_G − rank_actual)
                                  ↑ collinearity gap       ↑ grid quantization gap
                                  (Theorem G)               (Theorem G')

At gcd_sky130, r=8.43:
- rank_generic = min(733, 740) = 733
- rank_G = 730 (collinearity removes 3)
- rank_actual = 466 (grid quantization removes 264 more!)
- **Grid quantization dominates for coarse-grid PDKs:** 264 / 267 = 99%
  of rank loss for SKY130 (site width 0.46μm). For fine-grid PDKs
  (e.g., ASAP7, site width 0.054μm), collinearity becomes the more
  significant factor as the grid approaches continuous.

This means the dominant physics fingerprint is PDK-dependent: coarse
grids have their rank loss dominated by grid quantization, while fine
grids are closer to the collinearity-only prediction of Theorem G.

### gcd_preroute (NanGate45) comparison

| r (μm) | |E| | rank_G | rank_actual | error | η_actual |
|---------|-----|--------|-------------|-------|----------|
| 2.28 | 441 | 432 | 436 | +4 | 0.011 |
| 2.80 | 587 | 559 | 544 | −15 | 0.073 |
| 3.19 | 870 | 714 | 675 | −39 | 0.224 |
| 4.20 | 1219 | 714 | 690 | −24 | 0.434 |

NanGate45 shows smaller grid quantization effect (error −39 max vs
SKY130's −264). Consistent with NanGate45's finer grid and more
varied cell widths, which create more diverse restriction map directions.

### aes_preroute (NanGate45, large)

| r (μm) | |E| | rank_G | rank_actual | error | η_actual |
|---------|-----|--------|-------------|-------|----------|
| 2.09 | 1448 | 1444 | 1443 | −1 | 0.003 |
| 3.13 | 3060 | 2958 | 2629 | −329 | 0.141 |
| 3.49 | 3678 | 3000 | 2810 | −190 | 0.236 |
| 5.24 | 7706 | 3000 | 3000 | 0 | 0.611 |

At large r (5.24+), rank_actual = 3000 = 2|V| — generic regime reached.
Grid quantization effect is visible at moderate r (3.13: error = −329).

---

## 7. Special Cases (preserved from original)

### Case 1: Single row, all edges intra-row

m = 1 row, E_inter = ∅, c = 1 (connected row).

    rank(δ) = |V| − 1
    η_row = 1 − (|V| − 1)/|E| → 1 − 2/Δ̄  (n_v^eff = 1)

### Case 2: Dense inter-row (|E_inter| ≥ c + |V|, generic)

    rank(δ) = 2|V| = rank_generic,  gap = 0

### Case 3: Dense inter-row, quantized grid (real placement)

    rank_G = 2|V| but rank_actual < 2|V|

Theorem G predicts no gap, but grid quantization still reduces rank.
The gap_grid from Theorem G' captures this.

---

## 8. Effective Stalk Dimension

Define n_v^eff such that η = max(0, 1 − 2·n_v^eff / Δ̄):

    n_v^eff = rank(δ) / |V|

From experiments:

| Design | PDK | n_v^eff (at Δ̄ ≈ 4) | Best PS1 n_v |
|--------|-----|---------------------|-------------|
| gcd_sky130 | SKY130 | 1.26 | 1 |
| gcd_preroute | NanGate45 | 1.93 | 2 |
| aes_preroute | NanGate45 | 1.75 | 2 |

**Interpretation:** n_v^eff interpolates between 1 (pure row, no
inter-row info) and 2 (fully generic). The interpolation point depends
on PDK grid structure, not just row collinearity.

---

## 9. Physical Interpretation

### Why the gap exists

Three effects compound:

1. **Row collinearity** (Source I): y-DOF is unused for same-row edges.
   This is the ONLY effect Theorem G captures.

2. **Quantized Δy** (Source II): only m distinct row heights →
   inter-row edges have at most m(m−1)/2 distinct y-differences.
   Many edges share the same direction, reducing rank.

3. **Quantized Δx** (Source III): site-width grid → x-differences
   are discrete. Combined with Source II, creates repeated restriction
   map vectors.

### PDK dependence

| PDK | Site width | Row pitch | Grid effect |
|-----|-----------|-----------|-------------|
| SKY130 | 0.46μm | 2.72μm | STRONG (coarse grid, few directions) |
| NanGate45 | 0.19μm | 1.40μm | MODERATE (finer grid) |
| ASAP7 | 0.054μm | 0.27μm | WEAK (very fine grid → near-generic) |

Finer PDK grids → more distinct directions → closer to generic.
This explains why ASAP7 fits n_v=2 perfectly (R²=0.9999) while
SKY130 needs n_v=1.

---

## 10. Relation to Other Results

### Conjecture G (sheaf-swarm)

Theorem G partially resolves Conjecture G. The collinearity effect
(Source I) is fully characterized. Sources II and III require
Theorem G' for complete characterization.

### Meta-theorem (sheaf-swarm #110)

The genericity gap for placement is:

    gap = gap_collinear + gap_grid

Both are "physics fingerprints":
- gap_collinear: row-based layout structure
- gap_grid: discrete site grid and row quantization

Combined, they measure how standard cell placement deviates from
generic position — exactly the content the sheaf captures beyond
pure dimension counting.

### Theorem PS1 (this repo)

Theorem G + G' explain WHY PS1 fits with n_v=1 for SKY130:
the grid quantization (Source III) is so strong that the effective
rank is closer to |V| (n_v=1) than 2|V| (n_v=2).

---

## 11. Testable Predictions

### G-P1: Direction diversity predicts n_v^eff

Compute K = number of distinct direction vectors among edges.
Prediction: n_v^eff ≈ min(2, K / |V|).

### G-P2: Finer grid → smaller gap

Across PDKs, finer site width → more directions → higher n_v^eff.
Prediction: n_v^eff positively correlates with 1/Δx_grid.

### G-P3: Radius transition

For each design, there exists r* where n_v^eff transitions from
≈1 (intra-row dominated) to ≈2 (generic). Prediction:
r* ≈ row_pitch · sqrt(|V|/m) (geometric mean of row pitch and
average intra-row cell count).

### G-P4: Direction decomposition (Theorem G')

    rank_Gp = Σ_k (|V_k| − c_k)  vs  rank_actual

**Result: G' is tighter than G but still overestimates at large r.**

gcd_sky130:

| r | |E| | K dirs | rank_Gp | rank_actual | error |
|---|-----|--------|---------|-------------|-------|
| 3.26 | 84 | 16 | 84 | 84 | 0 |
| 6.32 | 443 | 62 | 440 | 387 | −53 |
| 8.43 | 733 | 108 | 721 | 466 | −255 |

The overcounting comes from vertex sharing between direction groups.
Each vertex has only 2 DOFs, but many direction groups pass through
it. The exact rank requires accounting for this vertex saturation.

**Open problem:** Find a rank formula tighter than G' that accounts
for vertex DOF limits (2 per vertex) across shared direction groups.
