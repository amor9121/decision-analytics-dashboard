def print_results(results, days, wage):
    for r in results:
        name = r.get("name", "Scenario")

        print("\n" + "=" * 90)
        print(name)
        print("=" * 90)

        # ---- Common summary ----
        if "status" in r:
            print("Status:", r.get("status"))

        if r.get("cost") is not None:
            print("Total cost (£):", r.get("cost"))

        if r.get("cost_increase_pct") is not None:
            print("Cost increase (%):", r.get("cost_increase_pct"))

        if r.get("gap") is not None:
            print("Fairness gap:", r.get("gap"))

        # ---- AED Task 4: descriptive outputs ----
        if (
            r.get("age_summary") is not None
            or r.get("los_summary") is not None
            or r.get("kpi_table") is not None
            or r.get("breach_counts") is not None
        ):
            print("\n[AED Task 4] Key outputs")

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

            if r.get("fig1") is not None or r.get("fig2") is not None:
                print(
                    "\nFigures: fig1/fig2 generated (view in Streamlit or save them)."
                )

        # ---- AED Task 5: statistical outputs ----
        if (
            r.get("stats") is not None
            or r.get("tables") is not None
            or r.get("figures") is not None
        ):
            print("\n[AED Task 5] Key outputs")

            stats = r.get("stats", {})
            if stats:
                print("\nStats:")
                for k, v in stats.items():
                    print(f"- {k}: {v}")

            tables = r.get("tables", {})
            if tables:
                print("\nTables:")
                for tname, tval in tables.items():
                    print(f"\n{tname}:")
                    print(tval)

            figs = r.get("figures")
            if figs is not None:
                try:
                    print(
                        f"\nFigures generated: {len(figs)} (view in Streamlit / save as images)"
                    )
                except Exception:
                    print("\nFigures generated (view in Streamlit / save as images)")

        # ---- Skill coverage (Task 3) ----
        if r.get("skill_coverage") is not None:
            print("\nSkill coverage per day (should be >=6 each):")
            for d, cov in r["skill_coverage"].items():
                print(
                    f"{d}: Programming={cov['Programming']:.2f}, "
                    f"Troubleshooting={cov['Troubleshooting']:.2f}"
                )

        # ---- Scheduling table (Task 1–3 / Task 2 scenarios) ----
        allocation = r.get("allocation")
        if allocation is not None:
            final_table = allocation.copy()

            final_table["Weekly Total"] = final_table[days].sum(axis=1)
            final_table["Hourly Wage (£)"] = final_table.index.map(
                lambda i: wage.get(i, "")
            )
            final_table["Weekly Wage (£)"] = final_table.index.map(
                lambda i: (
                    (final_table.loc[i, "Weekly Total"] * wage[i]) if i in wage else ""
                )
            )

            daily_total = final_table[days].sum(axis=0)
            daily_total["Weekly Total"] = final_table["Weekly Total"].sum()
            daily_total["Hourly Wage (£)"] = ""
            daily_total["Weekly Wage (£)"] = final_table["Weekly Wage (£)"].sum()

            final_table.loc["Daily Total"] = daily_total
            final_table = final_table.round(2)

            print("\nFinal Schedule Table:")
            print(final_table)

            print(
                f"\nTotal weekly operational hours: "
                f"{final_table.loc['Daily Total', 'Weekly Total']}"
            )
