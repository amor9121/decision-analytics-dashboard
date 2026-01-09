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


def solve_task2_s2(baseline_cost):
    # -----------------------
    # Stage 1: minimise gap
    # -----------------------
    model_a = pulp.LpProblem("Task2_S2A_MinGap", pulp.LpMinimize)

    x_a = pulp.LpVariable.dicts(
        "x", ((i, d) for i in operators for d in days), lowBound=0, cat="Integer"
    )

    H_max_a = pulp.LpVariable("H_max", lowBound=0)
    H_min_a = pulp.LpVariable("H_min", lowBound=0)

    model_a += H_max_a - H_min_a

    add_base_constraints(
        model_a,
        x_a,
        operators,
        days,
        availability,
        bachelor_ops,
        master_ops,
        required_daily_hours,
    )

    for i in operators:
        total_i = pulp.lpSum(x_a[(i, d)] for d in days)
        model_a += total_i <= H_max_a
        model_a += total_i >= H_min_a

    model_a.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[model_a.status] != "Optimal":
        return {
            "name": "Task 2 - S2 (Stage 1 infeasible)",
            "allocation": None,
            "weekly_totals": None,
            "cost": None,
            "cost_increase_pct": None,
            "gap": None,
        }

    min_gap = pulp.value(H_max_a - H_min_a)

    # -----------------------
    # Stage 2: minimise cost given min_gap
    # -----------------------
    model_b = pulp.LpProblem("Task2_S2B_MinCost_GivenMinGap", pulp.LpMinimize)

    x_b = pulp.LpVariable.dicts(
        "x", ((i, d) for i in operators for d in days), lowBound=0, cat="Integer"
    )

    # objective: minimise total cost
    model_b += pulp.lpSum(wage[i] * x_b[(i, d)] for i in operators for d in days)

    add_base_constraints(
        model_b,
        x_b,
        operators,
        days,
        availability,
        bachelor_ops,
        master_ops,
        required_daily_hours,
    )

    # fairness linking + enforce min gap
    H_max_b = pulp.LpVariable("H_max", lowBound=0)
    H_min_b = pulp.LpVariable("H_min", lowBound=0)

    for i in operators:
        total_i = pulp.lpSum(x_b[(i, d)] for d in days)
        model_b += total_i <= H_max_b
        model_b += total_i >= H_min_b

    model_b += (H_max_b - H_min_b) <= min_gap + 1e-6

    model_b.solve(pulp.PULP_CBC_CMD(msg=False))
    if pulp.LpStatus[model_b.status] != "Optimal":
        return {
            "name": "Task 2 - S2 (Stage 2 infeasible)",
            "allocation": None,
            "weekly_totals": None,
            "cost": None,
            "cost_increase_pct": None,
            "gap": None,
        }

    # Build result
    allocation = pd.DataFrame(0.0, index=operators, columns=days)
    for i in operators:
        for d in days:
            allocation.loc[i, d] = pulp.value(x_b[(i, d)]) or 0.0
    allocation = allocation.round(2)

    weekly_totals = allocation.sum(axis=1).round(2)
    cost = round(sum(wage[i] * weekly_totals.loc[i] for i in operators), 2)
    gap = round(float(weekly_totals.max() - weekly_totals.min()), 2)
    cost_increase_pct = round((cost - baseline_cost) / baseline_cost * 100, 2)

    results = {
        "name": "Task 2 - Senerio 2",
        "allocation": allocation,
        "weekly_totals": weekly_totals,
        "cost": cost,
        "cost_increase_pct": cost_increase_pct,
        "gap": gap,
    }

    return results

