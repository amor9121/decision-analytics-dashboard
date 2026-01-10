import numpy as np
import pandas as pd


def _print_schedule_table(allocation, days, wage, title="Final schedule table"):
    if allocation is None:
        return

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

    print(f"\n{title}:")
    print(table)


def print_task1_2(r: dict, days, wage):

    if r.get("status") is not None:
        print("Status:", r.get("status"))

    if r.get("cost") is not None:
        print("Total cost (£):", r.get("cost"))

    if r.get("cost_increase_pct") is not None:
        print("Cost increase (%):", r.get("cost_increase_pct"))

    allocation = r.get("allocation")
    if allocation is not None:
        _print_schedule_table(allocation, days, wage, title="Final schedule table")


def print_task3(r: dict, days, wage):

    # ---- headline metrics ----
    if r.get("gap") is not None:
        print("Fairness gap:", r.get("gap"))

    # ---- skill coverage ----
    if r.get("skill_coverage") is not None:
        print("\nSkill coverage per day:")
        for d, cov in r["skill_coverage"].items():
            prog = cov.get("Programming", None)
            trb = cov.get("Troubleshooting", None)
            if isinstance(prog, (int, float)) and isinstance(trb, (int, float)):
                print(f"{d}: Programming={prog:.2f}, Troubleshooting={trb:.2f}")
            else:
                print(f"{d}: {cov}")

    # ---- skill match (whatever your Task3 already provides) ----
    skill_match = r.get("skill_match")
    if skill_match is None and isinstance(r.get("tables"), dict):
        skill_match = r["tables"].get("skill_match")

    if skill_match is not None:
        print("\nSkill match:")
        if isinstance(skill_match, pd.DataFrame):
            print(skill_match)
        elif isinstance(skill_match, dict):
            try:
                print(pd.DataFrame(skill_match))
            except Exception:
                print(skill_match)
        else:
            print(skill_match)

    #  ---- print schedule table  ----
    allocation = r.get("allocation")
    if allocation is None and isinstance(r.get("tables"), dict):
        allocation = r["tables"].get("allocation")

    if allocation is not None:
        _print_schedule_table(allocation, days, wage, title="Final schedule table")


def print_task4(r: dict):

    br = r.get("breach_rate_pct", None)
    if isinstance(br, (int, float)):
        print(f"\nBreach rate (%): {br:.2f}")

    if r.get("breach_counts") is not None:
        print("\nBreach counts:")
        print(r["breach_counts"].to_string(index=False))

    if r.get("kpi_table") is not None:
        print("\nKPI table:")
        print(r["kpi_table"].to_string(index=False))

    if r.get("age_summary") is not None:
        print("\nAge summary:")
        print(r["age_summary"].to_string())

    if r.get("los_summary") is not None:
        print("\nLoS summary:")
        print(r["los_summary"].to_string())

    figs = r.get("figures", None)
    if isinstance(figs, dict):
        print(f"\nFigures generated: {len(figs)}")
        for k in figs.keys():
            print(" -", k)

    if r.get("summary"):
        print("\nSummary:")
        print(r["summary"])


def print_task5(r: dict):
    print("\n[Task 5] Statistical analysis")

    # ---- Summary ----
    if r.get("summary"):
        print("\nSummary:")
        print(r["summary"])

    # ---- Stats ----
    stats = r.get("stats", {})
    if isinstance(stats, dict) and stats:
        print("\nStats:")
        for k, v in stats.items():
            print(f"- {k}: {v}")

    # ---- Tables (ALL) ----
    tables = r.get("tables", {})
    if isinstance(tables, dict) and tables:
        print("\nTables:")
        for name, t in tables.items():
            print(f"\n{name}:")
            try:
                # Works for pandas DataFrame/Series
                if hasattr(t, "to_string"):
                    print(t.to_string(index=False))
                else:
                    print(t)
            except Exception as e:
                print(f"[Could not print table '{name}']: {type(e).__name__}: {e}")
                print(t)
    else:
        print("\n[No tables found]")

    # ---- Logit summaries (text) ----
    if r.get("logit_m1_summary"):
        print("\nlogit_m1_summary:")
        print(r["logit_m1_summary"])

    if r.get("logit_m2_summary"):
        print("\nlogit_m2_summary:")
        print(r["logit_m2_summary"])

    # ---- Figures ----
    figs = r.get("figures", {})
    if isinstance(figs, dict):
        print(f"\nFigures generated: {len(figs)}")
        for k in figs.keys():
            print(" -", k)
    else:
        print("\nFigures: not available")


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

        if name in [
            "Task 1",
            "Task 2",
            "Task 2 - Senerio 1",
            "Task 2 - Senerio 2",
        ]:
            print_task1_2(r, days, wage)

        elif name in ["Task 3"]:
            print_task3(r, days, wage)

        elif name == "Task 4":
            print_task4(r)

        elif name == "Task 5":
            print_task5(r)

        elif name == "Task 6":
            print_task6(r)

        else:
            # fallback: don't fail silently
            print("[Unknown result type]")
            print("Available keys:", sorted(list(r.keys())))
