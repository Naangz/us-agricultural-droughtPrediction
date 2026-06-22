from __future__ import annotations

from pathlib import Path
import csv

ROOT = Path(__file__).resolve().parent

SCENARIO_EXPORTS = [
    ("Kansas", "Scenario 2", ROOT / "kansas" / "output_weekly_kansas_scenario2"),
    ("Kansas", "Scenario 2A", ROOT / "kansas" / "output_weekly_kansas_scenario2A"),
    ("Kansas", "Scenario 2B", ROOT / "kansas" / "output_weekly_kansas_scenario2B"),
    ("Kansas", "Scenario 2C", ROOT / "kansas" / "output_weekly_kansas_scenario2C"),
    ("Nebraska", "Scenario 2", ROOT / "nebraska" / "output_weekly_nebraska_scenario2"),
    ("Nebraska", "Scenario 2A", ROOT / "nebraska" / "output_weekly_nebraska_scenario2A"),
    ("Nebraska", "Scenario 2B", ROOT / "nebraska" / "output_weekly_nebraska_scenario2B"),
    ("Nebraska", "Scenario 2C", ROOT / "nebraska" / "output_weekly_nebraska_scenario2C")]

EXPORT_FILES = {
    "feature_model_mutual_information": "feature_model_mutual_information.csv",
    "selected_feature_ranking": "selected_feature_ranking.csv",
    "pruned_feature_pairs": "pruned_feature_pairs.csv",
    "feature_feature_abs_correlation_selected": "feature_feature_abs_correlation_selected.csv",
}

def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))

def read_selected_correlation_long(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        rows = list(reader)
    if not rows:
        return []

    header = rows[0]
    if len(header) < 2:
        return []

    feature_cols = header[1:]
    long_rows: list[dict[str, str]] = []
    for row in rows[1:]:
        if not row:
            continue
        feature_row = row[0]
        values = row[1:]
        for feature_col, value in zip(feature_cols, values):
            long_rows.append(
                {
                    "feature_row": feature_row,
                    "feature_col": feature_col,
                    "abs_correlation": value,
                }
            )
    return long_rows

def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

def build_status_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for region, scenario, folder in SCENARIO_EXPORTS:
        row = {
            "region": region,
            "scenario": scenario,
            "output_folder": str(folder.relative_to(ROOT)),
        }
        for export_key, filename in EXPORT_FILES.items():
            row[export_key] = "present" if (folder / filename).exists() else "missing"
        rows.append(row)
    return rows

def combine_rows() -> dict[str, list[dict[str, str]]]:
    combined = {
        "mi": [],
        "ranking": [],
        "pruned": [],
        "corr_selected": [],
    }

    for region, scenario, folder in SCENARIO_EXPORTS:
        for row in read_csv_rows(folder / EXPORT_FILES["feature_model_mutual_information"]):
            row = dict(row)
            row["region"] = region
            row["scenario"] = scenario
            combined["mi"].append(row)

        for row in read_csv_rows(folder / EXPORT_FILES["selected_feature_ranking"]):
            row = dict(row)
            row["region"] = region
            row["scenario"] = scenario
            combined["ranking"].append(row)

        for row in read_csv_rows(folder / EXPORT_FILES["pruned_feature_pairs"]):
            row = dict(row)
            row["region"] = region
            row["scenario"] = scenario
            combined["pruned"].append(row)

        for row in read_selected_correlation_long(folder / EXPORT_FILES["feature_feature_abs_correlation_selected"]):
            row = dict(row)
            row["region"] = region
            row["scenario"] = scenario
            combined["corr_selected"].append(row)

    return combined

def write_summary_markdown(status_rows: list[dict[str, str]], combined: dict[str, list[dict[str, str]]]) -> None:
    summary_path = ROOT / "scenario2_feature_exports_summary.md"
    lines: list[str] = []
    lines.append("# Scenario 2 Feature Export Summary")
    lines.append("")
    lines.append("Dokumen ini merangkum status file ekspor CSV untuk Scenario 2, 2A, 2B, dan 2C pada Kansas dan Nebraska.")
    lines.append("")
    lines.append("## Status Export")
    lines.append("")
    lines.append("| Region | Scenario | Folder | MI CSV | Ranking CSV | Pruned CSV | Selected Corr CSV |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- |")
    for row in status_rows:
        lines.append(
            f"| {row['region']} | {row['scenario']} | `{row['output_folder']}` | "
            f"{row['feature_model_mutual_information']} | "
            f"{row['selected_feature_ranking']} | "
            f"{row['pruned_feature_pairs']} | "
            f"{row['feature_feature_abs_correlation_selected']} |"
        )
    lines.append("")
    lines.append("## Root Output Files")
    lines.append("")
    lines.append("- `scenario2_feature_export_status.csv`")
    lines.append("- `scenario2_feature_model_mutual_information_all.csv`")
    lines.append("- `scenario2_selected_feature_ranking_all.csv`")
    lines.append("- `scenario2_pruned_feature_pairs_all.csv`")
    lines.append("- `scenario2_selected_feature_correlations_all.csv`")
    lines.append("")
    lines.append("## Available Rows")
    lines.append("")
    lines.append(f"- Mutual information rows: `{len(combined['mi'])}`")
    lines.append(f"- Selected ranking rows: `{len(combined['ranking'])}`")
    lines.append(f"- Pruned pair rows: `{len(combined['pruned'])}`")
    lines.append(f"- Selected correlation rows: `{len(combined['corr_selected'])}`")
    if not any(combined.values()):
        lines.append("")
        lines.append("Belum ada data yang berhasil digabung. Jalankan ulang script Scenario 2/2A/2B/2C agar CSV ekspor baru terbentuk.")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def main() -> None:
    status_rows = build_status_rows()
    combined = combine_rows()

    write_csv(
        ROOT / "scenario2_feature_export_status.csv",
        [
            "region",
            "scenario",
            "output_folder",
            "feature_model_mutual_information",
            "selected_feature_ranking",
            "pruned_feature_pairs",
            "feature_feature_abs_correlation_selected"],
        status_rows,
    )
    write_csv(
        ROOT / "scenario2_feature_model_mutual_information_all.csv",
        ["feature", "mutual_information_score", "rank", "region", "scenario"],
        combined["mi"],
    )
    write_csv(
        ROOT / "scenario2_selected_feature_ranking_all.csv",
        ["feature", "mi_score", "max_abs_corr_prev", "max_corr_prev_feat", "rank", "region", "scenario"],
        combined["ranking"],
    )
    write_csv(
        ROOT / "scenario2_pruned_feature_pairs_all.csv",
        ["feature_dropped", "feature_kept", "abs_correlation", "region", "scenario"],
        combined["pruned"],
    )
    write_csv(
        ROOT / "scenario2_selected_feature_correlations_all.csv",
        ["feature_row", "feature_col", "abs_correlation", "region", "scenario"],
        combined["corr_selected"],
    )
    write_summary_markdown(status_rows, combined)

    print("Wrote root outputs:")
    print(" - scenario2_feature_export_status.csv")
    print(" - scenario2_feature_model_mutual_information_all.csv")
    print(" - scenario2_selected_feature_ranking_all.csv")
    print(" - scenario2_pruned_feature_pairs_all.csv")
    print(" - scenario2_selected_feature_correlations_all.csv")
    print(" - scenario2_feature_exports_summary.md")

if __name__ == "__main__":
    main()
