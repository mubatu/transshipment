# Transshipment Problem Solver

Operations Research course project — solves a multi-layer transshipment LP using PuLP.

## Problem

Minimizes transportation cost through a 4-layer network:
- **4 Supply nodes** → **3 T1 nodes** → **4 T2 nodes** → **5 Demand nodes**

## Requirements

```bash
pip install pulp
```

## Usage

**Jupyter Notebook** (recommended): Run `main.ipynb`

**CLI**:
```bash
python main.py Q2      # Base model solution
python main.py Q3a     # Sensitivity: supply capacity S_3
python main.py Q3b     # Sensitivity: supply capacity S_0
python main.py Q3c     # Sensitivity: demand D_0
python main.py Q3d     # Sensitivity: cost c_{0,2} (basic var)
python main.py Q3e     # Sensitivity: cost c_{0,0} (non-basic var)
python main.py Q3f     # Large cost change causing basis change
python main.py Q3g     # Arc restriction (forbid S_0→T1_2)
python main.py Q3h     # Max demand increase feasibility
```

## Output

- Optimal cost: **60,983**
- Shadow prices, reduced costs, and basis analysis for sensitivity questions
