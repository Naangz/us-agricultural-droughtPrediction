"""
Temporal heatmaps of county-level USDM D0+ area percentage.

This script uses the original USDM export files before decumulation or argmax.
It creates one heatmap for Kansas and one for Nebraska using the same 0-100%
color scale so both regions can be compared directly.

Run from the repository root:
    python visualization/dataset/plot_county_drought_heatmap.py
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = Path(__file__).resolve().parent
VALUE_COLUMN = "D0"
COLORBAR_LABEL = "Persentase Wilayah D0 atau Lebih Parah (%)"


@dataclass(frozen=True)
class RegionConfig:
    key: str
    label: str
    usdm_path: Path
    county_selection_path: Path


REGIONS = {
    "kansas": RegionConfig(
        key="kansas",
        label="Kansas",
        usdm_path=REPO_ROOT / "KAN_dm_export_20100101_20251231.csv",
        county_selection_path=REPO_ROOT / "County_selection_KS_20.csv",
    ),
    "nebraska": RegionConfig(
        key="nebraska",
        label="Nebraska",
        usdm_path=REPO_ROOT / "NEB_dm_export_20100101_20251231.csv",
        county_selection_path=REPO_ROOT / "County_selection_NE_20.csv",
    ),
}


def configure_plot_style() -> None:
    sns.set_theme(style="white", context="paper")
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Segoe UI", "Helvetica"],
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 10,
        }
    )


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")
    return pd.read_csv(path)


def validate_columns(df: pd.DataFrame, required_columns: set[str], source_name: str) -> None:
    missing = sorted(required_columns.difference(df.columns))
    if missing:
        raise ValueError(f"{source_name} missing required columns: {missing}")


def prepare_heatmap_matrix(
    usdm: pd.DataFrame,
    county_selection: pd.DataFrame,
    value_column: str = VALUE_COLUMN,
) -> pd.DataFrame:
    """
    Return county x week matrix without inter-county aggregation.

    `pivot` is intentionally used instead of `pivot_table`; duplicate county-week
    pairs should raise an error because each heatmap cell must represent exactly
    one county in one USDM week.
    """
    validate_columns(
        usdm,
        {"FIPS", "County", "ValidStart", value_column},
        "USDM dataset",
    )
    validate_columns(county_selection, {"FIPS", "County"}, "County selection")

    selection = county_selection[["FIPS", "County"]].copy()
    selection["FIPS"] = selection["FIPS"].astype(int)
    selection = selection.sort_values("County", key=lambda series: series.str.lower()).reset_index(
        drop=True
    )
    selected_fips = selection["FIPS"].tolist()
    county_order = selection["County"].tolist()

    data = usdm[["FIPS", "County", "ValidStart", value_column]].copy()
    data["FIPS"] = data["FIPS"].astype(int)
    data = data[data["FIPS"].isin(selected_fips)].copy()
    data["week_start"] = pd.to_datetime(data["ValidStart"], errors="coerce")
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce")

    invalid_dates = int(data["week_start"].isna().sum())
    if invalid_dates:
        raise ValueError(f"Ditemukan {invalid_dates} baris dengan ValidStart tidak valid.")

    invalid_values = int(data[value_column].isna().sum())
    if invalid_values:
        raise ValueError(f"Ditemukan {invalid_values} baris dengan nilai {value_column} tidak valid.")

    duplicate_rows = int(data.duplicated(["County", "week_start"]).sum())
    if duplicate_rows:
        raise ValueError(f"Ditemukan {duplicate_rows} duplikasi county-week pada data USDM.")

    matrix = data.pivot(index="County", columns="week_start", values=value_column)
    matrix = matrix.reindex(index=county_order)
    matrix = matrix.reindex(columns=sorted(matrix.columns))
    return matrix


def region_title(region_key: str) -> str:
    label = REGIONS[region_key].label
    return f"Persentase Wilayah County pada Kondisi D0 atau Lebih Parah di {label}"


def year_tick_positions(dates: pd.DatetimeIndex) -> tuple[list[float], list[str]]:
    if len(dates) == 0:
        return [], []

    date_series = pd.Series(pd.DatetimeIndex(dates))
    positions = []
    labels = []
    for year, group in date_series.groupby(date_series.dt.year, sort=True):
        first_position = float(group.index.min())
        last_position = float(group.index.max())
        positions.append(((first_position + 0.5) + (last_position + 0.5)) / 2)
        labels.append(str(year))
    return positions, labels


def region_output_path(output_dir: Path, region_key: str) -> Path:
    return output_dir / f"heatmap_county_drought_{region_key}.png"


def county_summary_output_path(output_dir: Path, region_key: str) -> Path:
    return output_dir / f"summary_county_drought_{region_key}.csv"


def year_summary_output_path(output_dir: Path, region_key: str) -> Path:
    return output_dir / f"summary_year_drought_{region_key}.csv"


def interpretation_output_path(output_dir: Path) -> Path:
    return output_dir / "interpretasi_heatmap_county_drought.md"


def compute_county_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    valid_counts = matrix.notna().sum(axis=1)
    summary = pd.DataFrame(
        {
            "county": matrix.index,
            "mean_d0_plus": matrix.mean(axis=1),
            "median_d0_plus": matrix.median(axis=1),
            "pct_weeks_d0_plus_gt_0": matrix.gt(0).sum(axis=1) / valid_counts * 100,
            "pct_weeks_d0_plus_eq_100": matrix.eq(100).sum(axis=1) / valid_counts * 100,
        }
    ).reset_index(drop=True)
    return summary


def compute_year_summary(matrix: pd.DataFrame) -> pd.DataFrame:
    records = []
    years = sorted(pd.DatetimeIndex(matrix.columns).year.unique())
    for year in years:
        year_values = matrix.loc[:, pd.DatetimeIndex(matrix.columns).year == year].to_numpy().ravel()
        values = pd.Series(year_values, dtype="float64").dropna()
        records.append(
            {
                "year": int(year),
                "mean_d0_plus": float(values.mean()),
                "median_d0_plus": float(values.median()),
                "pct_county_weeks_d0_plus_eq_100": float(values.eq(100).mean() * 100),
            }
        )
    return pd.DataFrame.from_records(records)


def export_summaries(
    matrix: pd.DataFrame,
    output_dir: Path,
    region_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    output_dir.mkdir(parents=True, exist_ok=True)
    county_summary = compute_county_summary(matrix)
    year_summary = compute_year_summary(matrix)
    county_summary.round(2).to_csv(
        county_summary_output_path(output_dir, region_key),
        index=False,
        encoding="utf-8",
    )
    year_summary.round(2).to_csv(
        year_summary_output_path(output_dir, region_key),
        index=False,
        encoding="utf-8",
    )
    return county_summary, year_summary


def interpretation_for_region(
    region_label: str,
    county_summary: pd.DataFrame,
    year_summary: pd.DataFrame,
) -> str:
    county_by_mean = county_summary.sort_values(
        ["mean_d0_plus", "pct_weeks_d0_plus_gt_0", "county"],
        ascending=[False, False, True],
    ).iloc[0]
    county_by_full = county_summary.sort_values(
        ["pct_weeks_d0_plus_eq_100", "mean_d0_plus", "county"],
        ascending=[False, False, True],
    ).iloc[0]
    year_by_mean = year_summary.sort_values(
        ["mean_d0_plus", "pct_county_weeks_d0_plus_eq_100", "year"],
        ascending=[False, False, True],
    ).iloc[0]
    year_by_full = year_summary.sort_values(
        ["pct_county_weeks_d0_plus_eq_100", "mean_d0_plus", "year"],
        ascending=[False, False, True],
    ).iloc[0]

    return (
        f"## {region_label}\n\n"
        f"Berdasarkan ringkasan numerik per county, rata-rata D0+ tertinggi terdapat pada "
        f"{county_by_mean['county']} (mean {county_by_mean['mean_d0_plus']:.2f}%, "
        f"median {county_by_mean['median_d0_plus']:.2f}%, dan "
        f"{county_by_mean['pct_weeks_d0_plus_gt_0']:.2f}% minggu memiliki D0+ > 0). "
        f"County dengan proporsi minggu D0+ = 100% tertinggi adalah "
        f"{county_by_full['county']} ({county_by_full['pct_weeks_d0_plus_eq_100']:.2f}% minggu). "
        f"Pada ringkasan tahunan seluruh county-week, tahun dengan mean D0+ tertinggi adalah "
        f"{int(year_by_mean['year'])} (mean {year_by_mean['mean_d0_plus']:.2f}%, "
        f"median {year_by_mean['median_d0_plus']:.2f}%). "
        f"Tahun dengan proporsi county-week D0+ = 100% tertinggi adalah "
        f"{int(year_by_full['year'])} ({year_by_full['pct_county_weeks_d0_plus_eq_100']:.2f}%).\n"
    )


def write_interpretation(
    summaries: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    output_dir: Path,
) -> Path:
    lines = [
        "# Interpretasi Terverifikasi Heatmap D0+",
        "",
        "Interpretasi ini dihitung dari ringkasan numerik USDM asli, bukan dari inspeksi visual heatmap.",
        "",
    ]
    for region_key, (county_summary, year_summary) in summaries.items():
        lines.append(interpretation_for_region(REGIONS[region_key].label, county_summary, year_summary))
    output_path = interpretation_output_path(output_dir)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def plot_heatmap(matrix: pd.DataFrame, region_key: str, output_path: Path) -> Path:
    configure_plot_style()

    fig_width = 15.5
    fig_height = 8.2
    fig, ax = plt.subplots(figsize=(fig_width, fig_height), facecolor="white")

    sns.heatmap(
        matrix,
        ax=ax,
        cmap="YlOrRd",
        vmin=0,
        vmax=100,
        linewidths=0,
        cbar_kws={"label": COLORBAR_LABEL},
    )

    tick_positions, tick_labels = year_tick_positions(pd.DatetimeIndex(matrix.columns))
    ax.set_xticks(tick_positions)
    ax.set_xticklabels(tick_labels, rotation=90, ha="center", va="top", fontsize=11)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    ax.set_xlabel("Week start")
    ax.set_ylabel("County")
    ax.set_title(region_title(region_key), pad=14)

    colorbar = ax.collections[0].colorbar
    colorbar.set_label(COLORBAR_LABEL, fontsize=12)
    colorbar.ax.tick_params(labelsize=10)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return output_path


def create_region_heatmap(config: RegionConfig, output_dir: Path = OUTPUT_DIR) -> Path:
    usdm = read_csv(config.usdm_path)
    county_selection = read_csv(config.county_selection_path)
    matrix = prepare_heatmap_matrix(usdm, county_selection)
    export_summaries(matrix, output_dir, config.key)
    return plot_heatmap(matrix, config.key, region_output_path(output_dir, config.key))


def main() -> None:
    output_paths = []
    summaries = {}
    for config in REGIONS.values():
        usdm = read_csv(config.usdm_path)
        county_selection = read_csv(config.county_selection_path)
        matrix = prepare_heatmap_matrix(usdm, county_selection)
        summaries[config.key] = export_summaries(matrix, OUTPUT_DIR, config.key)
        output_paths.append(plot_heatmap(matrix, config.key, region_output_path(OUTPUT_DIR, config.key)))
    interpretation_path = write_interpretation(summaries, OUTPUT_DIR)
    for output_path in output_paths:
        print(f"Heatmap disimpan di: {output_path}")
    print(f"Interpretasi disimpan di: {interpretation_path}")


if __name__ == "__main__":
    main()
