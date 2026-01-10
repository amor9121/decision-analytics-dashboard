import pulp
import pandas as pd

from core.data import (
    operators,
    days,
    skills,
    availability,
    wage,
    bachelor_ops,
    master_ops,
    required_daily_hours,
    min_skill_hours,
    s,
)

from core.constraints import add_base_constraints, add_skill_constraints


def solve_task3(baseline_cost):
    model = pulp.LpProblem("Task3", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "x", ((i, d) for i in operators for d in days), lowBound=0, cat="Integer"
    )

    # objective
    model += pulp.lpSum(wage[i] * x[(i, d)] for i in operators for d in days)

    # constraints
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

    add_skill_constraints(model, x, operators, days, skills, s, min_skill_hours)

    # solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    # allocation
    allocation = pd.DataFrame(0.0, index=operators, columns=days)
    for i in operators:
        for d in days:
            allocation.loc[i, d] = pulp.value(x[(i, d)]) or 0.0
    allocation = allocation.round(2)

    weekly_totals = allocation.sum(axis=1).round(2)
    cost = round(pulp.value(model.objective), 2)
    cost_increase_pct = round((cost - baseline_cost) / baseline_cost * 100, 2)

    # skill coverage per day
    skill_coverage = {}

    for d in days:
        skill_coverage[d] = {
            "Programming": sum(
                s[(i, "Programming")] * allocation.loc[i, d] for i in operators
            ),
            "Troubleshooting": sum(
                s[(i, "Troubleshooting")] * allocation.loc[i, d] for i in operators
            ),
        }

    tables = {
        "skill_coverage": (
            pd.DataFrame(skill_coverage)
            .T
            .reset_index()
            .rename(columns={"index": "Day"})
        )
    }

    results = {
        "name": "Task 3",
        "allocation": allocation,
        "weekly_totals": weekly_totals,
        "cost": cost,
        "cost_increase_pct": cost_increase_pct,
        "gap": None,
        "skill_coverage": skill_coverage,
        "tables": tables,
    }

    return results
