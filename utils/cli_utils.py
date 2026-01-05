from sklearn.metrics import confusion_matrix
import pandas as pd


def print_task1_2(r: dict, days, wage):
    print("\n[Tasks 1–2] Scheduling & cost scenarios")

    if r.get("status") is not None:
        print("Status:", r.get("status"))

    if r.get("cost") is not None:
        print("Total cost (£):", r.get("cost"))

    if r.get("cost_increase_pct") is not None:
        print("Cost increase (%):", r.get("cost_increase_pct"))

    allocation = r.get("allocation")
    if allocation is not None:
        table = allocation.copy()

        table["Weekly Total"] = table[days].sum(axis=1)
        table["Hourly Wage (£)"] = table.index.map(lambda i: wage.get(i, ""))
        table["Weekly Wage (£)"] = table.index.map(
            lambda i: (table.loc[i, "Weekly Total"] * wage[i]) if i in wage else ""
        )

        daily = table[days].sum(axis=0)
        daily["Weekly Total"] = table["Weekly Total"].sum()
        daily["Hourly Wage (£)"] = ""
        daily["Weekly Wage (£)"] = table["Weekly Wage (£)"].sum()

        table.loc["Daily Total"] = daily
        table = table.round(2)

        print("\nFinal schedule table:")
        print(table)


def print_task3(r: dict):
    print("\n[Task 3] Optimisation & coverage")

    if r.get("gap") is not None:
        print("Fairness gap:", r.get("gap"))

    if r.get("skill_coverage") is not None:
        print("\nSkill coverage per day:")
        for d, cov in r["skill_coverage"].items():
            print(
                f"{d}: Programming={cov['Programming']:.2f}, "
                f"Troubleshooting={cov['Troubleshooting']:.2f}"
            )


def print_task4(r: dict):
    print("\n[Task 4] Descriptive analysis")

    if r.get("breach_rate_pct") is not None:
        print("Breach rate (%):", r.get("breach_rate_pct"))

    if r.get("breach_counts") is not None:
        print("\nBreach counts:")
        print(r["breach_counts"])

    if r.get("kpi_table") is not None:
        print("\nKPI table:")
        print(r["kpi_table"])

    if r.get("age_summary") is not None:
        print("\nAge summary:")
        print(r["age_summary"])

    if r.get("los_summary") is not None:
        print("\nLoS summary:")
        print(r["los_summary"])

    if r.get("summary") is not None:
        print("\nSummary text:")
        print(r["summary"])


def print_task5(r: dict):
    print("\n[Task 5] Statistical analysis")

    if r.get("stats") is not None:
        print("\nStats:")
        for k, v in r["stats"].items():
            print(f"- {k}: {v}")

    if r.get("tables") is not None:
        print("\nTables:")
        for name, t in r["tables"].items():
            print(f"\n{name}:")
            print(t)

    if r.get("figures") is not None:
        print("\nFigures generated (see plots)")


def print_task6(r: dict):
    print("\n[Task 6] Breach prediction (ML)")

    metrics = r.get("summary")
    if not isinstance(metrics, pd.DataFrame):
        metrics = r.get("metrics")

    if metrics is not None:
        print("\nModel comparison table:")
        print(metrics.round(3))

    final_model = r.get("final_model") or r.get("best_model_name")
    if final_model is not None:
        print("\nFinal model:", final_model)

    if r.get("notes") is not None:
        print("\nNotes:")
        for k, v in r["notes"].items():
            print(f"- {k}: {v}")

    td = r.get("test_data", {})
    if "y_test" in td and "y_pred_best" in td:
        cm = confusion_matrix(td["y_test"], td["y_pred_best"])
        print("\nConfusion Matrix [ [TN FP], [FN TP] ]:")
        print(cm)

    if r.get("plots") is not None:
        print("\nPlots:", ", ".join(r["plots"].keys()))


def print_results(results, days, wage):
    for r in results:
        print("\n" + "=" * 90)
        print(r.get("name", "Result"))
        print("=" * 90)

        name = r.get("name", "")

        if name in ["Task 1", "Task 2", "Task 2 - Scenario 1", "Task 2 - Scenario 2"]:
            print_task1_2(r, days, wage)

        elif name == "Task 3":
            print_task3(r)

        elif name == "Task 4":
            print_task4(r)

        elif name == "Task 5":
            print_task5(r)

        elif name == "Task 6":
            print_task6(r)
