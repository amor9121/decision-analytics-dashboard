import numpy as np


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
    print("\n[Task 6] Machine learning model analysis")

    # ---- Summary ----
    if r.get("summary") is not None:
        print("\nSummary:")
        print(r["summary"])

    # ---- Key metrics ----
    print("\nMetrics:")
    acc = r.get("accuracy", None)
    if isinstance(acc, (int, float)):
        print(f"- Test accuracy: {acc:.2%}")

    roc_auc_val = r.get("roc_auc", None)
    if isinstance(roc_auc_val, (int, float)):
        print(f"- ROC-AUC: {roc_auc_val:.4f}")

    # ---- Selected features ----
    if r.get("selected_features") is not None:
        print("\nSelected features:")
        for f in r["selected_features"]:
            print(f"- {f}")

    # ---- Classification report ----
    if r.get("classification_report") is not None:
        print("\nClassification report:")
        print(r["classification_report"])

    # ---- Cross-validation ----
    cv_recall = r.get("cv_recall", None)
    cv_auc = r.get("cv_auc", None)

    if cv_recall is not None or cv_auc is not None:
        print("\nCross-validation:")
        if cv_recall is not None:
            print(f"- Recall (5-fold mean): {float(np.mean(cv_recall)):.2%}")
        if cv_auc is not None:
            print(f"- ROC-AUC (5-fold mean): {float(np.mean(cv_auc)):.4f}")

    # ---- Odds ratio table ----
    if r.get("odds_ratio_table") is not None:
        print("\nOdds ratio table:")
        print(r["odds_ratio_table"])

    # ---- Figures ----
    if r.get("figures") is not None:
        print("\nFigures generated (see plots):")
        print(f"- Total figures: {len(r['figures'])}")


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
