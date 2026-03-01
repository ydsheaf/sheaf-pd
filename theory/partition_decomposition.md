# Partition Decomposition of η

Adapted from sheaf-drone/theory/hierarchical_sheaf.md (Parts 1-2).
Placement context: partition = G-cell grid, groups = G-cells, crossing edges = inter-G-cell nets.

## Problem Setup

N cells in ℝ², constraint graph G = (V, E) with average degree Δ̄.
Safety sheaf F with stalk dim n_v, scalar edges (n_e = 1).
η = dim H¹/|E| = cohomological deficiency ratio.

**Question**: Partition V into k groups V₁, ..., V_k (= G-cells).
What is the relationship between:
- η(G, F) = deficiency of the full chip
- η(G_i, F|_{V_i}) = deficiency of each G-cell
- The "inter-G-cell" deficiency from crossing edges (inter-cell nets)?

---

## Part 1: Partition Rank Decomposition

### Setup

Partition V = V₁ ∪ ... ∪ V_k (disjoint G-cells).
Define:
- G_i = (V_i, E_i) = subgraph induced by V_i (edges with BOTH endpoints in V_i)
- E_cross = edges with endpoints in different G-cells
- E = E₁ ∪ ... ∪ E_k ∪ E_cross (disjoint)
- δ_i = coboundary of F restricted to G_i
- δ = coboundary of full G

### Theorem PD1 (Partition Rank Decomposition)

$$\sum_{i=1}^k \text{rank}(\delta_i) \leq \text{rank}(\delta) \leq \sum_{i=1}^k \text{rank}(\delta_i) + |E_{\text{cross}}|$$

**Proof of lower bound**: δ restricted to rows E_i and columns V_i
is block-diagonal diag(δ₁, ..., δ_k). Rank of block diagonal = sum of ranks.

**Proof of upper bound**: δ has |E| rows total.
rank(δ) ≤ Σ rank(δ_i) + rank(δ|_{E_cross})
         ≤ Σ rank(δ_i) + |E_cross|
since the crossing-edge submatrix has at most |E_cross| rows.

### Corollary (η Decomposition)

Let ε_i = |E_i|/|E| be the fraction of edges in G-cell i,
    ε_cross = |E_cross|/|E| the crossing fraction (= inter-cell net fraction).
Then:

$$\sum_i \varepsilon_i (1 - \eta_i) \leq 1 - \eta \leq \sum_i \varepsilon_i (1 - \eta_i) + \varepsilon_{\text{cross}}$$

Or equivalently:

$$\sum_i \varepsilon_i \eta_i - \varepsilon_{\text{cross}} \leq \eta \leq \sum_i \varepsilon_i \eta_i$$

**Interpretation (placement context)**:
- η ≤ Σ ε_i η_i : global η is at most the weighted average of G-cell η's
  (inter-cell nets can only HELP by adding new rank)
- η ≥ Σ ε_i η_i - ε_cross : inter-cell nets contribute at most ε_cross of rank
  (they can't fix more deficiency than their number)

**Placement implication**: A placement with low per-G-cell η_i values
guarantees low global η, regardless of inter-cell routing. This justifies
per-G-cell routability prediction.

---

## Part 2: Hierarchical Placement

### Model

In hierarchical placement, cells are partitioned at multiple levels.
At each level, the partition creates G-cells with internal edges and crossing edges.

### Theorem PD2 (Minimum Resource Count)

If each G-cell can handle at most B constraint violations
(limited by routing resources: tracks, vias, metal layers), then
the minimum number of G-cells requiring special attention is:

$$C \geq \frac{\eta \cdot |E|}{B}$$

With Theorem H (generic case, Δ̄ > 2n_v):

$$C \geq \frac{(1 - 2n_v/\bar\Delta) \cdot |E|}{B} = \frac{(1 - 2n_v/\bar\Delta) \cdot N\bar\Delta/2}{B}$$

### Example: Standard cell placement

N = 100,000 cells, Δ̄ = 8, n_v = 2 (2D position), η ≈ max(0, 1-4/8) = 0.5.
|E| = 400,000. Need to handle 200,000 constraint violations.
If each G-cell handles B = 1,000 → need C ≥ 200 congested G-cells.
