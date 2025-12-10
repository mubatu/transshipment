# Transshipment Problem Solver

Operations Research course project — solves a multi-layer transshipment linear program using PuLP with comprehensive sensitivity analysis.

## Problem Description

Minimizes total transportation cost through a **4-layer network**:

```
Supply (4) → T1 (3) → T2 (4) → Demand (5)
```

### Network Data

| Layer | Nodes | Capacities/Demands |
|-------|-------|-------------------|
| Supply | S₀, S₁, S₂, S₃ | 191, 233, 276, 385 |
| T1 | T1₀, T1₁, T1₂ | Flow conservation |
| T2 | T2₀, T2₁, T2₂, T2₃ | Flow conservation |
| Demand | D₀, D₁, D₂, D₃, D₄ | 102, 289, 127, 210, 352 |

### Mathematical Formulation

**Decision Variables:**
- `x[i][j]` — flow from Supply i to T1 j
- `y[j][k]` — flow from T1 j to T2 k  
- `z[k][l]` — flow from T2 k to Demand l

**Objective:** Minimize total transportation cost

**Constraints:**
- Supply capacity (≤)
- Flow conservation at T1 and T2 (=)
- Demand satisfaction (≥)

## Requirements

```bash
pip install pulp
```

## Usage

**Jupyter Notebook** (recommended): Run `main.ipynb`

**CLI:**
```bash
python main.py Q2      # Base model solution
python main.py Q3a     # Sensitivity: supply capacity S₃ (slack constraint)
python main.py Q3b     # Sensitivity: supply capacity S₀ (binding constraint)
python main.py Q3c     # Sensitivity: demand D₀
python main.py Q3d     # Sensitivity: cost c₀₂ (basic variable)
python main.py Q3e     # Sensitivity: cost c₀₀ (non-basic variable)
python main.py Q3f     # Large cost change causing basis change
python main.py Q3g     # Arc restriction (forbid S₀→T1₂)
python main.py Q3h     # Maximum demand increase feasibility
```

## Results Summary

### Q2 — Optimal Solution
- **Optimal cost:** 60,983
- **Basic variables:** x₀₂=191, x₁₂=233, x₂₁=276, x₃₁=380, y₁₃=656, y₂₃=424, z₃ᵢ serves all demand

### Q3 — Sensitivity Analysis

| Question | Analysis | Key Finding |
|----------|----------|-------------|
| Q3a | S₃ +1 | Shadow price = 0 (slack), no cost change |
| Q3b | S₀ +1 | Shadow price = -9, cost decreases by 9 |
| Q3c | D₀ +1 | Shadow price = 61, cost increases by 61 |
| Q3d | c₀₂ -1 | Basis unchanged, cost decreases by 191 |
| Q3e | c₀₀ -1 | Reduced cost drops (22→21), x₀₀ stays non-basic |
| Q3f | c₀₀ -27 | Basis change: x₀₀ enters, x₀₂ leaves |
| Q3g | x₀₂ = 0 | Cost increases by 3,579 |
| Q3h | D₁ max Δ | Δmax = 5 before infeasibility |

## Files

- `main.py` — CLI interface
- `main.ipynb` — Interactive notebook with outputs
- `report.pdf` — Full analysis report
- `description.pdf` — Problem description
