# sheaf-pd

**Placement as Drone Swarm: Sheaf-Cohomological Routability Prediction**

Standard cells = agents, overlap = collision, congestion = density.
Theorem H directly applies: η predicts placement routability.

## Core Mapping

| Drone Swarm | VLSI Placement |
|-------------|---------------|
| Agent position x_i | Cell center (x_i, y_i) |
| Safety radius δ | Half-perimeter + spacing |
| CBF h_ij = ‖x_i-x_j‖² - (2δ)² | Overlap constraint |
| Congestion density ρ(x) | G-cell utilization |
| Δ̄ (average degree) | Average cell-pair interactions |
| η > 0 | Unroutable placement |

## Key Insight

    η_∞ = max(0, 1 - 2n_v / Δ̄)

- **η = 0**: placement is routable (all constraints satisfiable)
- **η > 0**: fraction η of constraints are topologically unfixable
- G-cell level: compute η per G-cell → routability heatmap
- Chip level: global η predicts overall routability

## Theorems (from sheaf-swarm)

| Theorem | Application to Placement |
|---------|------------------------|
| H (closed form) | η_∞ = max(0, 1-2n_v/Δ̄) predicts routability |
| D (N_max) | Maximum cells per G-cell before η > 0 |
| E (CBF-sheaf) | Overlap/spacing constraints → sheaf restriction maps |

## New Results

| Result | Name | Statement |
|--------|------|-----------|
| PD1 | Partition Decomposition | η decomposes across G-cell partition with tight bounds |
| PD2 | Hierarchical η | Multi-level placement hierarchy inherits η bounds |

## Architecture

```
theory/          Partition decomposition theorem, placement-sheaf mapping
experiments/     G-cell η computation on OpenROAD benchmarks
paper/           ICCAD submission
```

## Relation to Trilogy

- **Prologue**: [sheaf-swarm](https://github.com/ydsheaf/sheaf-swarm) — Theorems A-H, drones
- **Part I**: [sheaf-solver](https://github.com/ydsheaf/sheaf-solver) — Logic Sheaf, floating-point
- **Part II**: [sheaf-drc](https://github.com/ydsheaf/sheaf-drc) — Physical Sheaf, DRC/timing
- **Part III**: [sheaf-unified](https://github.com/ydsheaf/sheaf-unified) — Bridge, universality
- **Application**: this repo — Placement routability via η
