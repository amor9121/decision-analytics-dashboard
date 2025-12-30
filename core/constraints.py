# =========================
# Shared base constraints
# =========================

import pulp


def add_base_constraints(
    model,
    x,
    operators,
    days,
    availability,
    bachelor_ops,
    master_ops,
    required_daily_hours,
):

    # (1) Daily coverage
    for d in days:
        model += pulp.lpSum(x[(i, d)] for i in operators) == required_daily_hours

    # (2) Availability
    for i in operators:
        for d in days:
            model += x[(i, d)] <= availability[i][d]

    # (3) Minimum weekly hours
    for i in bachelor_ops:
        model += pulp.lpSum(x[(i, d)] for d in days) >= 8

    for i in master_ops:
        model += pulp.lpSum(x[(i, d)] for d in days) >= 7


def add_fairness_constraints(model, x, operators, days):
    H_max = pulp.LpVariable("H_max", lowBound=0)
    H_min = pulp.LpVariable("H_min", lowBound=0)

    for i in operators:
        total_i = pulp.lpSum(x[(i, d)] for d in days)
        model += total_i <= H_max
        model += total_i >= H_min

    return H_max, H_min


def add_skill_constraints(model, x, operators, days, skills, s, min_skill_hours):
    for d in days:
        for k in skills:
            model += (
                pulp.lpSum(s[(i, k)] * x[(i, d)] for i in operators) >= min_skill_hours
            )
