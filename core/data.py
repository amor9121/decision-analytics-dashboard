# =========================
# Shared sets & parameters
# =========================

operators = ["E.Khan", "Y.Chen", "A.Taylor", "R.Zidane", "R.Perez", "C.Santos"]
days = ["Mon", "Tue", "Wed", "Thu", "Fri"]

bachelor_ops = ["E.Khan", "Y.Chen", "A.Taylor", "R.Zidane"]
master_ops = ["R.Perez", "C.Santos"]

availability = {
    "E.Khan": {"Mon": 6, "Tue": 0, "Wed": 6, "Thu": 0, "Fri": 6},
    "Y.Chen": {"Mon": 0, "Tue": 6, "Wed": 0, "Thu": 6, "Fri": 0},
    "A.Taylor": {"Mon": 4, "Tue": 8, "Wed": 4, "Thu": 0, "Fri": 4},
    "R.Zidane": {"Mon": 5, "Tue": 5, "Wed": 5, "Thu": 0, "Fri": 5},
    "R.Perez": {"Mon": 3, "Tue": 0, "Wed": 3, "Thu": 8, "Fri": 0},
    "C.Santos": {"Mon": 0, "Tue": 0, "Wed": 0, "Thu": 6, "Fri": 2},
}

wage = {
    "E.Khan": 25,
    "Y.Chen": 26,
    "A.Taylor": 24,
    "R.Zidane": 23,
    "R.Perez": 28,
    "C.Santos": 30,
}

required_daily_hours = 14

skills = ["Programming", "Troubleshooting"]

min_skill_hours = 6


# skill indicator s[i,k]
s = {(i, k): 0 for i in operators for k in skills}

for i in ["E.Khan", "Y.Chen", "R.Perez", "C.Santos"]:
    s[(i, "Programming")] = 1

for i in ["A.Taylor", "R.Zidane", "C.Santos"]:
    s[(i, "Troubleshooting")] = 1
