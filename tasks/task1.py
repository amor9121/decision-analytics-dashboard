# ===== DEBUG ONLY (REMOVE BEFORE SUBMISSION) =====
# import sys
# from pathlib import Path
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#    sys.path.insert(0, str(PROJECT_ROOT))
# =================================================

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


def solve_task1():
    model = pulp.LpProblem("Task1", pulp.LpMinimize)

    x = pulp.LpVariable.dicts(
        "x", ((i, d) for i in operators for d in days), lowBound=0, cat="Integer"
    )

    # Objective
    model += pulp.lpSum(wage[i] * x[(i, d)] for i in operators for d in days)

    # Constraints
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

    # Solve
    model.solve(pulp.PULP_CBC_CMD(msg=False))

    # Build result
    allocation = pd.DataFrame(0.0, index=operators, columns=days)
    for i in operators:
        for d in days:
            allocation.loc[i, d] = pulp.value(x[(i, d)]) or 0.0
    allocation = allocation.round(2)

    weekly_totals = allocation.sum(axis=1).round(2)
    cost = round(pulp.value(model.objective), 2)

    results = {
        "name": "Task 1",
        "allocation": allocation,
        "weekly_totals": weekly_totals,
        "cost": cost,
        "cost_increase_pct": 0.0,
        "gap": None,
    }

    return results


# ===== DEBUG ONLY (REMOVE BEFORE SUBMISSION) =====
if __name__ == "__main__":
    from core.utils import print_results

    t1 = solve_task1()
    print_results([t1], days, wage)
# =================================================
