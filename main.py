import sys
import pulp

# Original data
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


def solve_transshipment(supply_caps=None, demand_quants=None):
    """
    This builds and solves the transshipment LP.
    If arguments are None, it uses the original data.
    It returns the PuLP model object.
    """
    # I use copies so we never modify global data
    if supply_caps is None:
        supply_caps = supplyCapacities.copy()
    if demand_quants is None:
        demand_quants = demandQuantities.copy()

    S = range(len(supply_caps))
    T1 = range(nT1Nodes)
    T2 = range(nT2Nodes)
    D = range(len(demand_quants))

    model = pulp.LpProblem("MultiLayerTransshipment", pulp.LpMinimize)

    # Decision variables
    x = pulp.LpVariable.dicts("x", (S, T1), lowBound=0, cat="Continuous")
    y = pulp.LpVariable.dicts("y", (T1, T2), lowBound=0, cat="Continuous")
    z = pulp.LpVariable.dicts("z", (T2, D), lowBound=0, cat="Continuous")

    # Objective function
    model += (
        pulp.lpSum(costMatrixStoT1[i][j] * x[i][j] for i in S for j in T1)
        + pulp.lpSum(costMatrixT1toT2[j][k] * y[j][k] for j in T1 for k in T2)
        + pulp.lpSum(costMatrixT2toD[k][l] * z[k][l] for k in T2 for l in D)
    ), "TotalTransportationCost"

    # Supply capacity constraints
    for i in S:
        model += (
            pulp.lpSum(x[i][j] for j in T1) <= supply_caps[i],
            f"Supply_{i}",
        )

    # Flow conservation at first transshipment layer
    for j in T1:
        model += (
            pulp.lpSum(x[i][j] for i in S)
            == pulp.lpSum(y[j][k] for k in T2),
            f"Flow_T1_{j}",
        )

    # Flow conservation at second transshipment layer
    for k in T2:
        model += (
            pulp.lpSum(y[j][k] for j in T1)
            == pulp.lpSum(z[k][l] for l in D),
            f"Flow_T2_{k}",
        )

    # Demand satisfaction constraints
    for l in D:
        model += (
            pulp.lpSum(z[k][l] for k in T2) >= demand_quants[l],
            f"Demand_{l}",
        )

    # Solving the model
    model.solve(pulp.PULP_CBC_CMD(msg=False))
    return model

### Runners for each question ###
def run_Q2():
    """Output the Question 2 results."""
    model = solve_transshipment()

    print("========== MODEL ==========")
    print(model)  # objective + all constraints

    print("\n========== SOLUTION STATUS ==========")
    print("Status:", pulp.LpStatus[model.status])

    print("\n========== OPTIMAL OBJECTIVE ==========")
    print("Minimum total cost z* =", pulp.value(model.objective))

    print("\n========== BASIC VARIABLES ==========")
    for v in model.variables():
        if v.varValue is not None and v.varValue > 1e-6:
            print(f"{v.name} = {v.varValue}")


def run_Q3a():
    """
    Output the Question 3a results.
    """
    # Base model
    base_model = solve_transshipment()
    base_obj = pulp.value(base_model.objective)
    base_shadow_S3 = base_model.constraints["Supply_3"].pi

    # Modified supplies: increase S_3 by 1 (index 3)
    supply_caps_3a = supplyCapacities.copy()
    supply_caps_3a[3] += 1

    model_3a = solve_transshipment(supply_caps=supply_caps_3a)
    new_obj = pulp.value(model_3a.objective)
    new_shadow_S3 = model_3a.constraints["Supply_3"].pi

    print("========== Q3a: CHANGE IN SUPPLY CAPACITY AT S_3 ==========")
    print(f"Original capacity S_3: {supplyCapacities[3]}")
    print(f"New capacity S_3:      {supply_caps_3a[3]}\n")

    print("----- Objective values -----")
    print("Base objective value:           ", base_obj)
    print("Objective after change (Q3a):   ", new_obj)
    print("Change in objective (new - base):", new_obj - base_obj, "\n")

    print("----- Shadow price of Supply_3 -----")
    print("Base model shadow price (Supply_3): ", base_shadow_S3)
    print("Q3a model shadow price (Supply_3):  ", new_shadow_S3)

def run_Q3b():
    """
    Output the Question 3b results.
    """
    # Base model
    base_model = solve_transshipment()
    base_obj = pulp.value(base_model.objective)
    base_shadow_S0 = base_model.constraints["Supply_0"].pi

    # Modified supplies: increase S_0 by 1 (index 0)
    supply_caps_3b = supplyCapacities.copy()
    supply_caps_3b[0] += 1

    model_3b = solve_transshipment(supply_caps=supply_caps_3b)
    new_obj = pulp.value(model_3b.objective)

    print("========== Q3b: CHANGE IN SUPPLY CAPACITY AT S_0 ==========")
    print(f"Original capacity S_0: {supplyCapacities[0]}")
    print(f"New capacity S_0:      {supply_caps_3b[0]}\n")

    print("----- Objective values -----")
    print("Base objective value:           ", base_obj)
    print("Objective after change (Q3b):   ", new_obj)
    print("Change in objective (new - base):", new_obj - base_obj, "\n")

    print("----- Shadow price of Supply_0 -----")
    print("Base model shadow price (Supply_0): ", base_shadow_S0)

def run_Q3c():
    """
    Output the Question 3c results.
    The demand quantity of demand node D_0 increases by one unit.
    """
    # Base model
    base_model = solve_transshipment()
    base_obj = pulp.value(base_model.objective)
    base_shadow_D0 = base_model.constraints["Demand_0"].pi

    # Modified demands: increase D_0 by 1 (index 0)
    demand_3c = demandQuantities.copy()
    demand_3c[0] += 1

    model_3c = solve_transshipment(demand_quants=demand_3c)
    new_obj = pulp.value(model_3c.objective)

    print("========== Q3c: CHANGE IN DEMAND AT D_0 ==========")
    print(f"Original demand D_0: {demandQuantities[0]}")
    print(f"New demand D_0:      {demand_3c[0]}\n")

    print("----- Objective values -----")
    print("Base objective value:           ", base_obj)
    print("Objective after change (Q3c):   ", new_obj)
    print("Change in objective (new - base):", new_obj - base_obj, "\n")

    print("----- Shadow price of Demand_0 -----")
    print("Base model shadow price (Demand_0): ", base_shadow_D0)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 main.py Q2")
        print("  python3 main.py Q3a")
        print("  python3 main.py Q3b")
        print("  python3 main.py Q3c")
        sys.exit(1)

    task = sys.argv[1]

    if task == "Q2":
        run_Q2()
    elif task == "Q3a":
        run_Q3a()
    elif task == "Q3b":
        run_Q3b()
    elif task == "Q3c":
        run_Q3c()
    else:
        print("Unknown argument:", task)
        print("Use 'Q2', 'Q3a', 'Q3b' or 'Q3c'.")