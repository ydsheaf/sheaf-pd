# Topological Shield: Beyond Geometric Safety Filtering

A framework for safety shielding when pairwise constraints are
collectively infeasible — the regime where CBF-based shields fail.

---

## 1. The Problem: Geometric Shields Have a Blind Spot

Standard safety filtering (CBF, Hamilton-Jacobi, barrier certificates)
assumes the safe set is non-empty:

    ∃ u : ḣ_e(x, u) + α h_e(x) ≥ 0   ∀ e ∈ E

When this holds, the shield projects the nominal control onto the safe
set: u_safe = argmin ||u - u_nom||² s.t. safety constraints.

**The blind spot**: When the constraint set is **collectively infeasible**
— each constraint individually has a solution, but no single u satisfies
all constraints simultaneously — the QP projection oscillates and the
shield fails silently.

This happens when the system is **topologically overconstrained**:
more independent constraints than degrees of freedom.

---

## 2. Sheaf Cohomology Detects the Blind Spot

### Construction

Given a constraint graph G = (V, E) with:
- Vertex stalks F(v) = ℝ^{n_v} (per-agent DOFs)
- Edge stalks F(e) = ℝ^{n_e} (per-constraint dimension)
- Restriction maps ρ_{v→e} = ∇_{x_v} h_e (CBF gradient)

The coboundary map δ: C⁰(G, F) → C¹(G, F) encodes ALL constraint
gradients simultaneously. Its cokernel H¹(G, F) = C¹ / im(δ) measures
the space of constraint configurations that NO control input can satisfy.

### The Deficiency Ratio

    η = dim H¹(G, F) / |E|

- η = 0: all constraints simultaneously satisfiable → geometric shield works
- η > 0: fraction η of constraints are structurally unsatisfiable →
  geometric shield WILL oscillate → need topological shield

### Closed Form (Theorem H, sheaf-swarm)

For generic constraint configurations:

    η = max(0, 1 - 2n_v / Δ̄)

where Δ̄ = average vertex degree. Phase transition at Δ̄ = 2n_v.

---

## 3. The Topological Shield

### Architecture (Two-Layer)

    Layer 1 (Topological):  Measure η → scale action by g = max(ε, 1-η)
    Layer 2 (Geometric):    Standard CBF projection on scaled action

The topological layer sits ABOVE the geometric layer. It preprocesses
the nominal control to reduce demand amplitude, preventing the geometric
layer from oscillating on infeasible constraint sets.

### Why g = 1-η is Optimal (Theorem 11, sheaf-swarm)

The gain g = 1-η is the unique solution to the capacity-matching problem:

    max g   (don't slow down unnecessarily)
    s.t.    average constraint violation = 0

When η|E| constraints are infeasible, the CBF projector amplifies demand
on the remaining (1-η)|E| feasible constraints by factor 1/(1-η).
Setting g = 1-η cancels this amplification exactly.

**Proof**: The effective demand under gain g is g/(1-η) per feasible
constraint. Setting g/(1-η) = 1 gives g = 1-η. ∎

### Locality

η can be computed from local information only:

    η∞ = max(0, 1 - 2n_v / Δ̄)

Each agent needs only its local degree Δ̄ (obtainable via O(log N) gossip
rounds). No centralized computation, no SVD, no training.

---

## 4. Validated Instances

### Instance 1: Drone Swarm (sheaf-swarm)

| Component | Instantiation |
|---|---|
| Agent | Drone with position p_i ∈ ℝ^d |
| Constraint | CBF h_{ij} = ‖p_i - p_j‖² - d²_min ≥ 0 |
| n_v | 2d (position + velocity) or d (position only) |
| n_e | 1 (scalar safety margin) |
| Phase transition | Δ̄ = 2d (position-only) or 4d (with velocity) |
| Shield | v_des ← max(0.2, 1-η) · v_nom |

**Results** (N=128, Δ̄≈8, d=3):
- Standard CBF: 78.1% safety rate
- η-shield: **100%** safety rate
- GCBF+ (100 GPU-hours training): comparable to η-shield
- Feasible subspace RL (PPO): no improvement over η-shield

### Instance 2: VLSI Placement (sheaf-pd)

| Component | Instantiation |
|---|---|
| Agent | Standard cell with center (x_i, y_i) |
| Constraint | Overlap h_{ij} = ‖p_i - p_j‖² - δ²_{ij} ≥ 0 |
| n_v | 2 (position only, d=2) |
| n_e | 1 (scalar overlap margin) |
| Phase transition | Δ̄ = 4 |
| Shield | step_size ← max(0.2, 1-η_α) per G-cell |

**Results**:
- η as diagnostic: R² > 0.95 for η vs Δ̄ across 8+ designs
- η=0 → DRC=0: **confirmed** via OpenROAD GR (3 designs, 2 PDKs)
- η-shield for step size: -29% residual overlap area (NanGate45)
- η decomposition: η = η_density + η_structural, with
  η_structural PDK-dependent (SKY130 >> NanGate45 >> ASAP7)

### Comparison

| Aspect | Drone (real-time) | Placement (offline) |
|---|---|---|
| η as shield | **Dominant** (100% vs 78%) | Marginal (-29% area) |
| η as diagnostic | Useful (σ_min early warning) | **Dominant** (replaces RUDY) |
| RL improvement | None (Theorem 11 optimal) | None (convex problem) |

**Key insight**: The topological shield's value depends on the domain.
In real-time safety (drones), the shield prevents catastrophic failure.
In offline optimization (placement), the diagnostic value dominates.

---

## 5. The General Framework

### API

```python
class TopologicalShield:
    """Generic topological safety filter."""

    def __init__(self, constraint_graph, stalk_dim_v, stalk_dim_e):
        """
        Args:
            constraint_graph: edges = pairwise constraints
            stalk_dim_v: DOFs per agent (n_v)
            stalk_dim_e: constraint dimension (n_e, usually 1)
        """
        self.n_v = stalk_dim_v
        self.n_e = stalk_dim_e

    def compute_eta(self, positions, constraint_fn):
        """Compute η from current state.

        Local version: η = max(0, 1 - 2*n_v / mean_degree)
        Exact version: SVD of coboundary matrix
        """
        dbar = self.mean_degree(positions)
        return max(0, 1 - 2 * self.n_v / dbar)

    def shield(self, u_nominal, eta):
        """Apply topological shield.

        Returns: u_shielded = max(ε, 1-η) * u_nominal
        """
        gain = max(self.epsilon, 1 - eta)
        return gain * u_nominal

    def diagnose(self, positions):
        """Per-region η heatmap for diagnostics."""
        # Partition space into regions, compute η per region
        ...
```

### Requirements for a New Instance

To instantiate the topological shield for a new domain:

1. **Constraint graph**: Define which agents interact (edges)
2. **CBF**: Define pairwise safety function h_e(x_i, x_j)
3. **Stalk dimensions**: n_v = DOFs per agent, n_e = constraints per pair
4. **Verify genericity**: Check that CBF gradients are in general position
   (holds for almost all configurations in ℝ^d)

The closed-form η = max(0, 1-2n_v/Δ̄) then applies automatically.

---

## 6. What This Framework Does NOT Do

1. **Not a replacement for CBF**: The topological shield modulates the
   geometric shield, doesn't replace it. You still need CBF for per-edge
   safety enforcement.

2. **Not learning-based**: Theorem 11 proves g=1-η is optimal for scalar
   gain. RL cannot improve upon it (validated in both instances).

3. **Not a solver for infeasible constraints**: When η > 0, some
   constraints CANNOT be satisfied. The shield gracefully degrades
   (reduces speed / accepts violations) rather than solving the
   infeasibility. Resolving infeasibility requires changing the topology
   (removing agents, migrating cells).

4. **Radius-dependent**: η depends on the interaction radius r, which
   determines the constraint graph density. Choosing r too large makes η
   conservative (over-predicts infeasibility). The correct r matches the
   physical constraint scale (safety radius for drones, cell diagonal
   for placement).

---

## 7. Open Problems

### Theoretical

1. **Non-scalar edge stalks**: Theorem H assumes n_e = 1. Multi-dimensional
   constraints (e.g., vector-valued safety margins) need a generalized
   rank formula.

2. **Non-uniform gains**: Theorem 11 proves optimality for a single scalar
   gain g. Per-region or per-agent gains could potentially do better, but
   our experiments (per-G-cell RL, sheaf-pd) showed no improvement.

3. **Dynamic graphs**: When the constraint graph changes over time
   (agents move in/out of range), η(t) is a stochastic process. Its
   statistics (mean, variance, autocorrelation) determine the shield's
   effectiveness.

### Empirical

4. **η > 0 → DRC > 0 direction**: Validated η=0 → DRC=0 via OpenROAD GR.
   Need congested/poorly-placed designs to test the reverse direction.

5. **Third instance**: Need a non-EDA, non-drone instance to claim
   generality. Candidates: multi-robot path planning, network flow,
   resource allocation.

6. **Comparison with HMARL-CBF**: Recent work (OpenReview, Sep 2025)
   combines hierarchical MARL with CBF. Direct comparison with
   η-shield on the same benchmark would clarify the value of the
   topological layer.

---

## 8. Relation to Prior Work

| Work | What it does | What it misses |
|---|---|---|
| CBF (Ames et al.) | Per-edge geometric safety | Collective infeasibility |
| GCBF+ (Fan et al.) | Learned graph CBF, N≤1024 | Can't compute η (GNN is local) |
| HMARL-CBF (2025) | Hierarchical MARL + CBF | No topological analysis |
| SOS barriers | Polynomial barrier synthesis | Assumes feasibility |
| Layered Safety (2025) | MARL avoids conflict regions | Heuristic conflict detection |
| **This work** | **Cohomological η + shield** | **Scalar gain only (so far)** |

The key differentiator: we provide a **closed-form, computable measure**
of collective infeasibility (η) with a **provably optimal shield** (g=1-η).
No training, no heuristics, one line of code.
