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


def solve_transshipment(
    supply_caps=None,
    demand_quants=None,
    costStoT1=None,
    costT1toT2=None,
    costT2toD=None,
):
    """
    This builds and solves the transshipment LP.
    If arguments are None, it uses the original data.
    It returns the PuLP model object.
    """
    # I use copies so we never modify global lists
    if supply_caps is None:
        supply_caps = supplyCapacities.copy()
    if demand_quants is None:
        demand_quants = demandQuantities.copy()

    # I use given cost matrices or fall back to globals
    if costStoT1 is None:
        costStoT1 = costMatrixStoT1
    if costT1toT2 is None:
        costT1toT2 = costMatrixT1toT2
    if costT2toD is None:
        costT2toD = costMatrixT2toD

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
        pulp.lpSum(costStoT1[i][j] * x[i][j] for i in S for j in T1)
        + pulp.lpSum(costT1toT2[j][k] * y[j][k] for j in T1 for k in T2)
        + pulp.lpSum(costT2toD[k][l] * z[k][l] for k in T2 for l in D)
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

def run_Q3d():
    """
    Output the Question 3d results.
    """
    # Base model
    base_model = solve_transshipment()
    base_obj = pulp.value(base_model.objective)

    # Non-zero variables in base model
    base_nonzero = {
        v.name: v.varValue
        for v in base_model.variables()
        if v.varValue is not None and v.varValue > 1e-6
    }

    # For reference: value of x_0_2 in the base solution
    x_0_2_val = base_nonzero.get("x_0_2", 0.0)

    # Modified cost matrix, build a modified cost matrix for StoT1
    costStoT1_3d = [row.copy() for row in costMatrixStoT1]
    costStoT1_3d[0][2] -= 1 # Decrease c_{0,2} by 1

    # Solve the model with the modified cost matrix
    model_3d = solve_transshipment(costStoT1=costStoT1_3d)
    new_obj = pulp.value(model_3d.objective)

    # Non-zero variables in the modified model
    new_nonzero = {
        v.name: v.varValue
        for v in model_3d.variables()
        if v.varValue is not None and v.varValue > 1e-6
    }

    # Compare supports and values
    base_support = set(base_nonzero.keys())
    new_support = set(new_nonzero.keys())
    same_support = (base_support == new_support)

    # Max absolute difference in values for variables present in both
    common_vars = base_support & new_support
    max_diff = 0.0
    for name in common_vars:
        diff = abs(base_nonzero[name] - new_nonzero[name])
        if diff > max_diff:
            max_diff = diff

    print("========== Q3d: CHANGE IN COST c_{0,2} (S_0 -> T1_2) ==========")
    print(f"Original cost c_0,2: {costMatrixStoT1[0][2]}")
    print(f"New cost c_0,2:      {costStoT1_3d[0][2]}\n")

    print("----- Non-zero variables comparison (base --> Q3d) -----")
    all_vars = sorted(set(base_nonzero.keys()) | set(new_nonzero.keys()))
    for name in all_vars:
        base_val = base_nonzero.get(name, 0.0)
        new_val = new_nonzero.get(name, 0.0)
        print(f"{name} = {base_val} --> {new_val}")
    print()

    print("----- Basis / support comparison -----")
    print("Same set of non-zero variable names?:", same_support)
    print("Max abs difference in common variable values:", max_diff)
    print()

    print("----- Objective values -----")
    print("Base objective value:            ", base_obj)
    print("Objective after change (Q3d):    ", new_obj)
    print("Change in objective (new - base):", new_obj - base_obj, "\n")

def run_Q3e():
    """
    Output the Question 3e results.
    """
    # Base model
    base_model = solve_transshipment()
    base_obj = pulp.value(base_model.objective)
    base_vars = base_model.variablesDict()
    base_x00 = base_vars["x_0_0"].varValue
    base_redcost_x00 = base_vars["x_0_0"].dj

    # Build a modified cost matrix for StoT1 (deep copy)
    costStoT1_3e = [row.copy() for row in costMatrixStoT1]
    costStoT1_3e[0][0] -= 1 # Decrease c_{0,0} by 1

    # Solve the model with the modified cost matrix
    model_3e = solve_transshipment(costStoT1=costStoT1_3e)
    new_obj = pulp.value(model_3e.objective)
    new_vars = model_3e.variablesDict()
    new_x00 = new_vars["x_0_0"].varValue
    new_redcost_x00 = new_vars["x_0_0"].dj

    print("========== Q3e: CHANGE IN COST c_{0,0} (S_0 -> T1_0) ==========")
    print(f"Original cost c_0,0: {costMatrixStoT1[0][0]}")
    print(f"New cost c_0,0:      {costStoT1_3e[0][0]}\n")

    print("----- x_0_0 values -----")
    print("x_0_0 (base model):   ", base_x00)
    print("x_0_0 (Q3e model):    ", new_x00, "\n")

    print("----- Reduced cost of x_0_0 -----")
    print("Reduced cost x_0_0 (base):", base_redcost_x00)
    print("Reduced cost x_0_0 (Q3e): ", new_redcost_x00, "\n")

    print("----- Objective values -----")
    print("Base objective value:            ", base_obj)
    print("Objective after change (Q3e):    ", new_obj)
    print("Change in objective (new - base):", new_obj - base_obj)

def run_Q3f():
    """
    Output the Question 3f results.
    """
    # Base model
    base_model = solve_transshipment()
    base_obj = pulp.value(base_model.objective)
    base_vars = base_model.variablesDict()
    base_x00 = base_vars["x_0_0"].varValue
    base_x02 = base_vars["x_0_2"].varValue

    # Build a modified cost matrix for StoT1
    costStoT1_3f = [row.copy() for row in costMatrixStoT1]
    costStoT1_3f[0][0] -= 27 # Decrease c_{0,0} by 27

    # Solve the model with the modified cost matrix
    model_3f = solve_transshipment(costStoT1=costStoT1_3f)
    new_obj = pulp.value(model_3f.objective)
    new_vars = model_3f.variablesDict()
    new_x00 = new_vars["x_0_0"].varValue
    new_x02 = new_vars["x_0_2"].varValue

    print("========== Q3f: LARGE CHANGE IN COST c_{0,0} (S_0 -> T1_0) ==========")
    print(f"Original cost c_0,0: {costMatrixStoT1[0][0]}")
    print(f"New cost c_0,0:      {costStoT1_3f[0][0]}\n")

    print("----- x_0_0 and x_0_2 values -----")
    print("x_0_0 (base model):   ", base_x00, " --> ", new_x00)
    print("x_0_2 (base model):   ", base_x02, " --> ", new_x02, "\n")

    print("----- Basis Change Check -----")
    if new_x00 > 0:
        print(">> RESULT: Basis CHANGED. x_0_0 entered the basis.")
    else:
        print(">> RESULT: Basis did NOT change.")
    print()

    print("----- Objective values -----")
    print("Base objective value:            ", base_obj)
    print("Objective after change (Q3f):    ", new_obj)
    print("Change in objective (new - base):", new_obj - base_obj)

def run_Q3g():
    """
    Output the Question 3g results.
    """
    # Base model
    base_model = solve_transshipment()
    base_obj = pulp.value(base_model.objective)
    base_vars = base_model.variablesDict()
    base_x02 = base_vars["x_0_2"].varValue

    # Modified model: forbid S_0 -> T1_2, build the same model, then add x_0_2 = 0
    model_3g = solve_transshipment()
    vars_3g = model_3g.variablesDict()
    x_0_2_var = vars_3g["x_0_2"]

    # Add the constraint x_0_2 = 0
    model_3g += (x_0_2_var == 0), "Ban_S0_T1_2"

    # Solve with the added restriction
    model_3g.solve(pulp.PULP_CBC_CMD(msg=False))
    new_obj = pulp.value(model_3g.objective)
    new_vars = model_3g.variablesDict()
    new_x02 = new_vars["x_0_2"].varValue

    print("========== Q3g: FORBID ARC S_0 -> T1_2 (x_0_2 = 0) ==========")
    print("x_0_2 in base model:   ", base_x02)
    print("x_0_2 in Q3g model:    ", new_x02, "\n")

    print("----- Objective values -----")
    print("Base objective value:            ", base_obj)
    print("Objective after change (Q3g):    ", new_obj)
    print("Change in objective (new - base):", new_obj - base_obj)

def run_Q3h():
    """
    Output the Question 3h results.
    """
    # Base model
    base_model = solve_transshipment()
    base_obj = pulp.value(base_model.objective)

    print("========== Q3h: INCREASE DEMAND AT D_1 ==========")
    print("Original demand vector:", demandQuantities)
    print("Base objective value:  ", base_obj, "\n")

    max_feasible_increase = None

    # Try increasing D_1 by k = 0,1,2,...,6 and see when it becomes infeasible
    for k in range(1, 7):
        demand_3h = demandQuantities.copy()
        demand_3h[1] += k  # increase D_1

        model_k = solve_transshipment(demand_quants=demand_3h)
        status_k = pulp.LpStatus[model_k.status]

        print(f"Increase Δ = {k}:  D_1 = {demandQuantities[1]} + {k} = {demand_3h[1]}")
        print("  Status:", status_k)

        if status_k == "Optimal":
            obj_k = pulp.value(model_k.objective)
            print("  Objective value:", obj_k)
            max_feasible_increase = k
        else:
            print("  Model infeasible or not optimal.")
        print()

    print("----- SUMMARY -----")
    print("Largest tested Δ with an optimal solution:", max_feasible_increase)
    print("So it is expected that Δ_max =", max_feasible_increase, "for D_1.")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python3 main.py Q2")
        print("  python3 main.py Q3a")
        print("  python3 main.py Q3b")
        print("  python3 main.py Q3c")
        print("  python3 main.py Q3d")
        print("  python3 main.py Q3e")
        print("  python3 main.py Q3f")
        print("  python3 main.py Q3g")
        print("  python3 main.py Q3h")
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
    elif task == "Q3d":
        run_Q3d()
    elif task == "Q3e":
        run_Q3e()
    elif task == "Q3f":
        run_Q3f()
    elif task == "Q3g":
        run_Q3g()
    elif task == "Q3h":
        run_Q3h()
    else:
        print("Unknown argument:", task)
        print("Use 'Q2', 'Q3a', 'Q3b', 'Q3c', 'Q3d', 'Q3e', 'Q3f', 'Q3g' or 'Q3h'.")
