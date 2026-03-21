# Placement-Sheaf Mapping Theorem

Formalizes the exact mapping from VLSI standard cell placement to the
sheaf-cohomological framework of sheaf-swarm (Theorems A-H).

---

## 1. Problem Setup

### Standard Cell Placement

Given N standard cells in 2D:
- Cell i has center position p_i = (x_i, y_i) in R^2
- Cell i has dimensions (w_i, h_i) (width, height)
- Spacing requirement: s_ij >= 0 between cells i and j (from design rules)

The **overlap constraint** between cells i, j is:

    h_{ij}^{rect} = max(0, |x_i - x_j| - (w_i + w_j)/2 - s_ij)
                  + max(0, |y_i - y_j| - (h_i + h_j)/2 - s_ij)

This is non-smooth (absolute values, max). For the sheaf construction
we require smooth barrier functions, so we use the standard **CBF
linearization**:

    h_{ij} = ||p_i - p_j||^2 - delta_{ij}^2

where delta_{ij} = effective half-size:

    delta_{ij} = sqrt((w_i + w_j)^2/4 + (h_i + h_j)^2/4) + s_ij

This gives h_{ij} > 0 iff the bounding circles of cells i, j do not
overlap. Note: bounding-circle relaxation is standard in analytical
placement (e.g., HPWL relaxation, ePlace density smoothing).

### Constraint Graph

Define the constraint graph G = (V, E):
- V = {1, ..., N}, one vertex per cell
- E = {(i,j) : cells i, j have potential overlap interaction}

In practice, E is defined by a proximity radius r:

    (i,j) in E  iff  ||p_i - p_j|| < r

where r is chosen so that only nearby cells interact (matching the
local nature of placement constraints). The average degree is:

    Delta_bar = 2|E| / |V|

which measures the average number of overlap interactions per cell.

---

## 2. The Placement Sheaf

### Definition

The **placement sheaf** F on G is a cellular sheaf defined by:

| Component | Definition | Dimension |
|-----------|-----------|-----------|
| Vertex stalk F(v) | R^{n_v} = R^2 (position DOFs: x, y) | n_v = 2 |
| Edge stalk F(e) | R^{n_e} = R^1 (one scalar overlap constraint) | n_e = 1 |
| Restriction map rho_{i -> (i,j)} | nabla_{p_i} h_{ij}^T in R^{1 x 2} | (1 x n_v) matrix |

Explicitly, for edge e = (i,j) with overlap CBF h_{ij} = ||p_i - p_j||^2 - delta_{ij}^2:

    rho_{i -> e} = nabla_{p_i} h_{ij} = 2(p_i - p_j)^T = [2(x_i - x_j),  2(y_i - y_j)]

    rho_{j -> e} = nabla_{p_j} h_{ij} = -2(p_i - p_j)^T = [-2(x_i - x_j),  -2(y_i - y_j)]

### Coboundary Map

The coboundary delta: C^0(G, F) -> C^1(G, F) acts on a global section
s = (s_1, ..., s_N) in R^{2N}. We adopt the **same-restriction-map
convention** used in sheaf_cohomology.py and Theorem E of sheaf-swarm:
both endpoints of an edge share the same restriction map
rho_e = 2(p_i - p_j)^T, and the coboundary takes the difference:

    (delta s)(e_{ij}) = rho_e . s_i - rho_e . s_j
                      = 2(p_i - p_j)^T . (s_i - s_j)

This is the directional derivative dh_{ij} measuring how a differential
displacement (s_i - s_j) affects the overlap margin.

**Remark on sign conventions.** An alternative convention assigns the
per-vertex gradients rho_{i->e} = nabla_{p_i} h_{ij} = 2(p_i - p_j)^T
and rho_{j->e} = nabla_{p_j} h_{ij} = -2(p_i - p_j)^T, giving
(delta s)(e) = rho_{i->e} s_i - rho_{j->e} s_j = 2(p_i-p_j)^T(s_i+s_j).
For the overlap CBF, rho_{i->e} = -rho_{j->e}, so the two conventions
differ by an edge reorientation and produce coboundary matrices of
identical rank. We use the same-rho convention throughout for consistency
with the sheaf-swarm codebase.

---

## 3. Proof: Overlap Constraint IS a CBF

### Proposition (Overlap = CBF)

The overlap constraint h_{ij} = ||p_i - p_j||^2 - delta_{ij}^2 is a
valid control barrier function for the placement dynamics.

### Proof

**Step 1: Barrier property.** Define the safe set:

    S = {(p_1, ..., p_N) : h_{ij}(p_i, p_j) >= 0 for all (i,j) in E}

The set {h_{ij} >= 0} = {||p_i - p_j|| >= delta_{ij}} is the complement of
the overlap region. h_{ij} is smooth (polynomial in positions), and
h_{ij} = 0 exactly on the boundary of the safe set. Therefore h_{ij}
is a valid barrier function.

**Step 2: Relative degree.** For the first-order "placement dynamics"

    dp_i/dt = u_i    (direct position control, as in analytical placement)

we have:

    dh_{ij}/dt = nabla_{p_i} h_{ij} . u_i + nabla_{p_j} h_{ij} . u_j
               = 2(p_i - p_j)^T . (u_i - u_j)

The control input u_i appears explicitly in dh/dt, so h_{ij} has
relative degree 1 with respect to the placement dynamics. This is
exactly the CBF condition: we can enforce dh/dt >= -alpha . h (forward
invariance) by choosing u_i appropriately.

**Step 3: Gradient = restriction map.** The gradient

    nabla_{p_i} h_{ij} = 2(p_i - p_j)^T in R^{1 x 2}

is precisely the restriction map rho_{i -> (i,j)} of the placement sheaf.
This is the CBF-sheaf coupling of Theorem E from sheaf-swarm.

Therefore, by Theorem E, the placement sheaf is a well-defined cellular
sheaf whose restriction maps are derived from a valid CBF.  qed

---

## 4. Drone-Placement Mapping Table

| Drone Swarm | VLSI Placement | Mathematical Object |
|-------------|----------------|---------------------|
| Agent i | Standard cell i | Vertex v_i in V |
| Position p_i in R^d | Cell center (x_i, y_i) in R^2 | Element of stalk F(v_i) |
| Safety radius delta | Half-diagonal + spacing delta_{ij} | CBF parameter |
| CBF h_{ij} = \|\|p_i - p_j\|\|^2 - (2 delta)^2 | Overlap h_{ij} = \|\|p_i - p_j\|\|^2 - delta_{ij}^2 | Barrier function on edge e_{ij} |
| Proximity graph G(N, r) | Constraint graph (nearby cells) | Graph G = (V, E) |
| Average degree Delta_bar | Average cell interactions | 2\|E\|/\|V\| |
| Agent velocity v_i | (not used: position-only) | -- |
| Control input u_i (acceleration) | Cell displacement u_i (placement update) | Element of R^d |
| CBF-QP: min \|\|u - u_nom\|\|^2 s.t. safety | Legalization: min displacement s.t. no overlap | Constrained optimization |
| H^1 = 0: all safety constraints satisfiable | H^1 = 0: necessary condition for overlap feasibility (under generic position) | Sheaf cohomology vanishing |
| eta > 0: fraction of unsatisfiable constraints | eta > 0: fraction of unresolvable overlaps | Cohomological deficiency ratio |
| N_max: max agents before eta > 0 | N_max: max cells per G-cell before congestion | Theorem D bound |
| Congestion density | G-cell utilization | Local Delta_bar |

### Key Simplification

Placement is **simpler** than the drone problem:
- d = 2 (all placement is 2D)
- Position-only CBF (no velocity dependence): n_v = d = 2
- First-order dynamics dp/dt = u (direct position control)
- No need for HOCBF (which would give n_v = 2d = 4)

This means the phase transition occurs at the LOWER threshold:

    Delta_bar_c = 2 n_v = 2 . 2 = 4

compared to Delta_bar_c = 4d = 8 for HOCBF-equipped drones.

---

## 5. Main Theorem: Placement Feasibility via eta

### Theorem PS1 (Placement Sheaf Deficiency)

Let G = (V, E) be the constraint graph of a standard cell placement
with N cells in R^2. Let F be the placement sheaf (n_v = 2, n_e = 1)
with restriction maps rho_{i->e} = nabla_{p_i} h_{ij}. If the
restriction maps are in general position (which holds when no three
cell centers are collinear -- see Remark below), then:

    eta = dim H^1(G, F) / |E| = max(0, 1 - 2 . n_v / Delta_bar)
                                = max(0, 1 - 4 / Delta_bar)

**Proof.** By Theorem E (sheaf-swarm), the placement sheaf is a valid
cellular sheaf with scalar edge stalks and vertex stalk dimension
n_v = 2. By Theorem H (sheaf-swarm), for generic restriction maps:

    eta = max(0, 1 - 2 n_v / Delta_bar)

For the placement sheaf, n_v = 2, giving eta = max(0, 1 - 4/Delta_bar).

It remains to verify that the CBF-derived restriction maps are
sufficiently generic. The restriction map for edge (i,j) is
rho_{i->e} = 2(p_i - p_j)^T in R^{1 x 2}. Two rows of the coboundary
delta corresponding to edges (i,j) and (i,k) sharing vertex i are:

    row_{ij}: [..., 2(p_i - p_j)^T, ..., -2(p_i - p_j)^T, ..., 0, ...]
    row_{ik}: [..., 2(p_i - p_k)^T, ..., 0, ..., -2(p_i - p_k)^T, ...]

These rows are linearly independent whenever (p_i - p_j) and (p_i - p_k)
are linearly independent, i.e., when cells i, j, k are not collinear.
For generic placements (no three centers collinear), the coboundary
achieves maximal rank, and Theorem H applies.  qed

### Remark (Genericity Condition)

The condition "no three cell centers are collinear" is equivalent to
requiring that for each vertex i with neighbors j, k in the constraint
graph, the vectors (p_i - p_j) and (p_i - p_k) are linearly
independent. This fails only on a measure-zero set in position space.

In practice, standard cell placement on a grid may have many collinear
centers (e.g., cells in the same row). However, perturbation analysis
shows that the generic formula eta = max(0, 1 - 4/Delta_bar) remains
a tight upper bound: collinearity can only INCREASE eta (reduce rank),
so the generic formula is a LOWER bound on the actual eta.

For row-based placement where all cells in a row share the same
y-coordinate, the effective stalk dimension drops to n_v_eff = 1 within
each row (only x-displacement contributes to rank). This gives
eta_row = max(0, 1 - 2/Delta_bar_row) for intra-row constraints, and
the full n_v = 2 applies only for inter-row interactions. This is
precisely the CBF genericity gap (Conjecture G from sheaf-swarm).

**EDA caveat.** For row-based placements, the generic position assumption
typically does NOT hold: cells sharing a row are collinear by
construction. In this regime, eta = 0 is best understood as a
**proxy** for feasibility, not a guarantee. The cohomological analysis
provides a necessary condition (if eta > 0 the placement is certainly
overconstrained), but eta = 0 alone does not ensure that a legal
placement exists -- the bounding-circle relaxation and the loss of
genericity both introduce gaps between the sheaf-theoretic prediction
and the true combinatorial feasibility of the placement problem.

---

## 6. Phase Transition at Delta_bar = 4

### Corollary PS1.1 (Placement Phase Transition)

The placement undergoes a phase transition at Delta_bar_c = 4:

- **Delta_bar < 4**: eta = 0. This is a necessary condition for
  simultaneous satisfiability of all overlap constraints under generic
  position: there exists a sufficient number of DOFs for a satisfying
  displacement. The placement is **potentially feasible**.

- **Delta_bar > 4**: eta = 1 - 4/Delta_bar > 0. A fraction eta of
  overlap constraints are **topologically unsatisfiable** -- no
  rearrangement of cells can resolve them without violating other
  constraints. The placement is **congested**.

### Interpretation

The average degree Delta_bar measures the average number of overlap
interactions per cell. When each cell interacts with fewer than 4
neighbors on average, the constraint system is underdetermined (2 DOFs
per cell vs. < 2 constraints per cell, since each constraint involves
2 cells). When Delta_bar > 4, the system becomes overdetermined.

This directly parallels the drone swarm result:
- Drones (position-only, d=2): Delta_bar_c = 2d = 4
- Placement (position-only, d=2): Delta_bar_c = 2 . 2 = 4

The critical density is identical because the mathematical structure
is identical: scalar constraints on 2D positions.

### Connection to Congestion

The constraint graph density Delta_bar is directly related to
placement congestion:

    Delta_bar_local(x) = (number of cell-pair interactions in neighborhood of x)
                       ~ (local cell density)^2 . pi . r^2

For a G-cell with area A containing n cells:
- Local density: rho = n / A
- Expected local degree: Delta_bar ~ rho . pi . r^2 = n . pi . r^2 / A

The congestion threshold Delta_bar > 4 translates to:

    n > 4A / (pi . r^2)

This gives an upper bound on the number of cells per G-cell before
congestion onset, directly from Theorem D (sheaf-swarm):

    N_max^{G-cell} = 4A / (pi . r^2)

where r is the interaction radius (determined by cell sizes and spacing
rules). For modern standard cell libraries:
- Typical cell width: 0.2-2 um
- Typical cell height: 1-2 um (fixed by row height)
- Interaction radius r ~ 2-4 um (a few cell widths)
- G-cell area: 10x10 um^2 to 50x50 um^2

Example: A = 20x20 = 400 um^2, r = 3 um:
N_max = 4 . 400 / (pi . 9) ~ 57 cells per G-cell.

---

## 7. Concrete Example: 10 Cells in a Row

### Setup

Consider N = 10 identical unit-square cells (w_i = h_i = 1) placed
along the x-axis with centers at positions:

    p_i = (i . s, 0)    for i = 0, 1, ..., 9

where s is the cell spacing (center-to-center distance).

Interaction radius: r = 2 (cells interact with immediate neighbors).

### Constraint Graph

For spacing s = 1.5 (dense but not overlapping):
- Each interior cell has 2 neighbors (left and right)
- Boundary cells have 1 neighbor
- |V| = 10, |E| = 9 (path graph)
- Delta_bar = 2 . 9 / 10 = 1.8

For spacing s = 0.8 (overlapping, need interaction radius r = 2.5):
- Each interior cell has 4 neighbors (2 left, 2 right)
- |V| = 10, |E| = 16
- Delta_bar = 2 . 16 / 10 = 3.2

For spacing s = 0.5 (very dense, r = 3):
- Each interior cell has 5+ neighbors
- |V| = 10, |E| = 25
- Delta_bar = 2 . 25 / 10 = 5.0

### Computing eta

| Spacing s | |E| | Delta_bar | eta = max(0, 1 - 4/Delta_bar) | Interpretation |
|-----------|-----|-----------|-------------------------------|----------------|
| 1.5 | 9 | 1.8 | 0 | Easily legalizable |
| 0.8 | 16 | 3.2 | 0 | Still legalizable (below threshold) |
| 0.5 | 25 | 5.0 | 0.20 | 20% of overlaps unresolvable |
| 0.3 | 35 | 7.0 | 0.43 | 43% unresolvable -- severe congestion |

### Verification

For the s = 0.5 case (Delta_bar = 5, eta = 0.2):
- dim C^0 = n_v . |V| = 2 . 10 = 20
- dim C^1 = n_e . |E| = 1 . 25 = 25
- rank(delta) = min(20, 25) = 20 (generically)
- dim H^1 = 25 - 20 = 5
- eta = 5 / 25 = 0.20

The 5 unsatisfiable constraints are the "excess" beyond what 20 DOFs
can handle. These correspond to the cells that MUST overlap regardless
of how they are rearranged -- precisely the congestion hotspot.

### Practical Implication

At s = 0.8 (Delta_bar = 3.2 < 4), the constraint system has enough DOFs
that a legalizer may resolve all overlaps (eta = 0 is necessary for
feasibility but not a guarantee, since the CBF is a bounding-circle
relaxation of the true rectangular constraint).
At s = 0.5 (Delta_bar = 5 > 4), approximately 20% of overlap constraints
are inherently unsatisfiable -- the placement is too dense. The only
resolution is to move cells to a different region (global re-placement)
or add whitespace.

---

## 8. Higher-Order Extensions

### HOCBF for Placement with Timing

If placement also encodes timing constraints (slack, arrival time), the
state per cell becomes (x_i, y_i, t_i, ...) with n_v > 2. The HOCBF
form from sheaf-swarm applies:

    h_tilde_{ij} = dh_{ij}/dt + alpha . h_{ij}

where dh/dt encodes timing-driven objectives. This gives:

| Placement model | State per cell | n_v | Delta_bar_c |
|-----------------|---------------|-----|-------------|
| Position-only | (x, y) | 2 | 4 |
| + slack | (x, y, slack) | 3 | 6 |
| + arrival time | (x, y, t_arr, t_dep) | 4 | 8 |
| Full timing | (x, y, slack, delay, ...) | k | 2k |

Each additional DOF increases the phase transition threshold by 2,
allowing denser placements before eta > 0. This explains why
timing-driven placement can tolerate higher local density: the
additional degrees of freedom (timing slack) provide more room for
the constraint system.

### Connection to ePlace / RePlAce

Modern analytical placers (ePlace, RePlAce, DREAMPlace) use a smoothed
density function:

    phi(p) = sum_i w_i . bell(p - p_i)

where bell() is a bell-shaped smoothing kernel. The density constraint
phi(p) <= rho_target is enforced globally.

The sheaf construction provides a LOCAL version of this: eta_i > 0
in G-cell i iff the local density exceeds the phase transition.
This is more informative than a single global phi, because it
identifies WHICH regions are congested and by how much.

---

## 9. Summary

### What We Proved

1. **The overlap constraint h_{ij} = ||p_i - p_j||^2 - delta_{ij}^2
   is a valid CBF** for placement dynamics dp/dt = u (Proposition,
   Section 3).

2. **The placement sheaf is well-defined** with vertex stalks R^2
   (position) and edge stalks R^1 (overlap), with restriction maps
   given by the CBF gradient (Section 2, via Theorem E of sheaf-swarm).

3. **The cohomological deficiency ratio** has the closed form
   eta = max(0, 1 - 4/Delta_bar) for generic placements (Theorem PS1,
   via Theorem H of sheaf-swarm).

4. **Phase transition at Delta_bar = 4**: placements with fewer than 4
   average overlap interactions per cell satisfy the necessary condition
   for feasibility (eta = 0); those with more have an irreducible
   fraction of violations (Corollary PS1.1). For row-based placements
   where generic position may not hold, eta = 0 is a proxy for
   feasibility rather than a guarantee.

### What Remains

- Computational verification on real placement benchmarks (ISPD, ICCAD)
- Quantifying the genericity gap for row-based placements (collinear cells)
- Connection to GR congestion via PD1 (next file: gcell_routability.md)
- Timing-aware extension with n_v > 2
