import pulp
import pandas as pd

from core.data import (
    operators,
    days,
    availability,
    wage,
    bachelor_ops,
    master_ops,
    required_daily_hours,
)

from core.constraints import add_base_constraints


def solve_task2_s1(baseline_cost):
    model = pulp.LpProblem("Task2_Senerio1", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "x", ((i, d) for i in operators for d in days), lowBound=0, cat="Integer"
    )

    H_max = pulp.LpVariable("H_max", lowBound=0)
    H_min = pulp.LpVariable("H_min", lowBound=0)

    # Objective: minimise fairness gap
    model += H_max - H_min

    # Base constraints (Task 1)
    add_base_constraints(
        model,
        x,
        operators,
        days,
        availability,
        bachelor_ops,
        master_ops,
        required_daily_hours,
    )

    # Fairness linking
    for i in operators:
        total_i = pulp.lpSum(x[(i, d)] for d in days)
        model += total_i <= H_max
        model += total_i >= H_min

    # Cost cap (+1.8%)
    cost_cap = 1.018 * baseline_cost
    model += (
        pulp.lpSum(wage[i] * x[(i, d)] for i in operators for d in days) <= cost_cap
    )

    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    # Build result
    allocation = pd.DataFrame(0.0, index=operators, columns=days)
    for i in operators:
        for d in days:
            allocation.loc[i, d] = pulp.value(x[(i, d)]) or 0.0
    allocation = allocation.round(2)

    weekly_totals = allocation.sum(axis=1).round(2)
    cost = round(sum(wage[i] * weekly_totals.loc[i] for i in operators), 2)
    gap = round(float(weekly_totals.max() - weekly_totals.min()), 2)
    cost_increase_pct = round((cost - baseline_cost) / baseline_cost * 100, 2)

    results = {
        "name": "Task 2 - Senerio 1",
        "allocation": allocation,
        "weekly_totals": weekly_totals,
        "cost": cost,
        "cost_increase_pct": cost_increase_pct,
        "gap": gap,
    }

    return results
