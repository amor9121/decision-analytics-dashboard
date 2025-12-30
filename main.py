from tasks.task1 import solve_task1
from tasks.task2_s1 import solve_task2_s1
from tasks.task2_s2 import solve_task2_s2
from tasks.task3 import solve_task3
from core.utils import print_results, export_csv
from core.data import days, wage


# baseline
base = solve_task1()
baseline_cost = base["cost"]

# tasks
t1 = solve_task1()
s1 = solve_task2_s1(baseline_cost)
s2 = solve_task2_s2(baseline_cost)
t3 = solve_task3(baseline_cost)

# print results
results = [t1, s1, s2, t3]
print_results(results, days, wage)


# export CSV
export_csv([t1, s1, s2, t3], days, wage, filepath="outputs/scheduling_results.csv")
