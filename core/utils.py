import os
import pandas as pd


def print_results(results, days, wage):

    for r in results:
        print("\n" + "=" * 90)
        print(r.get("name", "Scenario"))
        print("=" * 90)

        # ---- Summary ----
        print("Total cost (£):", r.get("cost"))

        if r.get("cost_increase_pct") is not None:
            print("Cost increase (%):", r.get("cost_increase_pct"))

        if r.get("gap") is not None:
            print("Fairness gap:", r.get("gap"))

        allocation = r.get("allocation")

        # ---- Skill coverage (Task 3 only) ----
        if r.get("skill_coverage") is not None:
            print("\nSkill coverage per day (should be >=6 each):")
            for d, cov in r["skill_coverage"].items():
                print(
                    f"{d}: Programming={cov['Programming']:.2f}, "
                    f"Troubleshooting={cov['Troubleshooting']:.2f}"
                )

        # ---- Final table ----
        if allocation is not None:
            final_table = allocation.copy()

            final_table["Weekly Total"] = final_table[days].sum(axis=1)
            final_table["Hourly Wage (£)"] = final_table.index.map(
                lambda i: wage[i] if i in wage else ""
            )
            final_table["Weekly Wage (£)"] = final_table.index.map(
                lambda i: (
                    final_table.loc[i, "Weekly Total"] * wage[i] if i in wage else ""
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


def export_csv(results, days, wage, filepath="outputs/scheduling_grouped.csv"):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    cols = days + ["Weekly Total", "Hourly Wage (£)", "Weekly Wage (£)"]
    blocks = []

    for r in results:
        name = r.get("name", "Task")
        allocation = r.get("allocation")
        if allocation is None:
            continue

        table = allocation.copy()
        table["Weekly Total"] = table[days].sum(axis=1)
        table["Hourly Wage (£)"] = table.index.map(lambda i: wage.get(i, ""))
        table["Weekly Wage (£)"] = table.index.map(
            lambda i: table.loc[i, "Weekly Total"] * wage[i] if i in wage else ""
        )

        daily = table[days].sum(axis=0)
        daily["Weekly Total"] = table["Weekly Total"].sum()
        daily["Hourly Wage (£)"] = ""
        daily["Weekly Wage (£)"] = table["Weekly Wage (£)"].sum()
        table.loc["Daily Total"] = daily

        table = table[cols].round(2).reset_index(drop=True)

        title_row = pd.DataFrame([[name] + [""] * (len(cols) - 1)], columns=cols)
        header_row = pd.DataFrame([cols], columns=cols)

        blocks.extend([title_row, header_row, table])

    out = pd.concat(blocks, ignore_index=True)

    out.to_csv(filepath, index=False, header=False)
    print(f"Saved: {filepath}")
