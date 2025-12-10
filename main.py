import pulp

### 1. Data ###
nSupplyNodes = 4
nT1Nodes = 3
nT2Nodes = 4
nDemandNodes = 5

costMatrixStoT1 = [
    [28, 27, 11],
    [16, 21, 11],
    [11, 10, 24],
    [27, 17, 25],
]

costMatrixT1toT2 = [
    [47, 34, 43, 69],
    [61, 46, 32, 33],
    [47, 68, 31, 30],
]

costMatrixT2toD = [
    [15, 18, 19, 16, 18],
    [19, 15, 18, 13, 14],
    [17, 15, 19, 17, 18],
    [11, 12, 11, 12, 12],
]

supplyCapacities = [191, 233, 276, 385]
demandQuantities = [102, 289, 127, 210, 352]

# Index sets
# I use 0-based indexing in the code.
S  = range(nSupplyNodes)
T1 = range(nT1Nodes)
T2 = range(nT2Nodes)
D  = range(nDemandNodes)

### 2. Model ###
model = pulp.LpProblem("MultiLayerTransshipment", pulp.LpMinimize)

# Decision variables: x_ij, y_jk, z_kl >= 0
x = pulp.LpVariable.dicts("x", (S, T1), lowBound=0, cat="Continuous")
y = pulp.LpVariable.dicts("y", (T1, T2), lowBound=0, cat="Continuous")
z = pulp.LpVariable.dicts("z", (T2, D), lowBound=0, cat="Continuous")

### 3. Objective function ###
model += (
    pulp.lpSum(costMatrixStoT1[i][j] * x[i][j] for i in S for j in T1)
    + pulp.lpSum(costMatrixT1toT2[j][k] * y[j][k] for j in T1 for k in T2)
    + pulp.lpSum(costMatrixT2toD[k][l] * z[k][l] for k in T2 for l in D)
), "TotalTransportationCost"

### 4. Constraints ###

# Supply capacity: sum_j x_ij <= S_i
for i in S:
    model += (
        pulp.lpSum(x[i][j] for j in T1) <= supplyCapacities[i],
        f"Supply_{i}"
    )

# Flow conservation at 1st transshipment layer: sum_i x_ij = sum_k y_jk
for j in T1:
    model += (
        pulp.lpSum(x[i][j] for i in S)
        == pulp.lpSum(y[j][k] for k in T2),
        f"Flow_T1_{j}"
    )

# Flow conservation at 2nd transshipment layer: sum_j y_jk = sum_l z_kl
for k in T2:
    model += (
        pulp.lpSum(y[j][k] for j in T1)
        == pulp.lpSum(z[k][l] for l in D),
        f"Flow_T2_{k}"
    )

# Demand satisfaction: sum_k z_kl >= D_l
for l in D:
    model += (
        pulp.lpSum(z[k][l] for k in T2) >= demandQuantities[l],
        f"Demand_{l}"
    )

### 5. Output model ###
print("========== MODEL ==========")
print(model) # objective + all constraints

### 6. Solving the model ###
status = model.solve(pulp.PULP_CBC_CMD(msg=False))

print("\n========== SOLUTION STATUS ==========")
print("Status:", pulp.LpStatus[status])

print("\n========== OPTIMAL OBJECTIVE ==========")
print("Minimum total cost z* =", pulp.value(model.objective))

### 7. Print basic variables ###
print("\n========== BASIC VARIABLES ==========")
for v in model.variables():
    if v.varValue is not None and v.varValue > 1e-6:
        print(f"{v.name} = {v.varValue}")