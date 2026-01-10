import sys
from contextlib import redirect_stdout
import warnings
from statsmodels.tools.sm_exceptions import ConvergenceWarning

from tasks.task1 import solve_task1
from tasks.task2_s1 import solve_task2_s1
from tasks.task2_s2 import solve_task2_s2
from tasks.task3 import solve_task3
from tasks.task4 import solve_task4
from tasks.task5 import solve_task5
from tasks.task6 import solve_task6
from utils.cli_utils import print_results
from core.data import days, wage


warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", message="overflow encountered in multiply")

# baseline
base = solve_task1()
baseline_cost = base["cost"]

# tasks
t1 = solve_task1()
s1 = solve_task2_s1(baseline_cost)
s2 = solve_task2_s2(baseline_cost)
t3 = solve_task3(baseline_cost)
t4 = solve_task4()
t5 = solve_task5()
t6 = solve_task6("data/AED4weeks.csv")

# print results
results = [t1, s1, s2, t3, t4, t5, t6]
print_results(results, days, wage)

with open("outputs/results.txt", "w", encoding="utf-8") as f:
    with redirect_stdout(f):
        print_results(results, days, wage)