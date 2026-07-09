"""
Seasonality analysis for weekly NASA POWER + USDM integrated datasets.

This script is descriptive, not predictive. It tests whether the weekly weather
signals show empirical support for an annual 52-week cycle.

Run from the repository root:
    python seasonality_analysis/analyze_seasonality.py
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


VARIABLES = ["PRECTOTCORR", "T2M", "RH2M"]
DECOMPOSITION_VARIABLES = ["T2M", "PRECTOTCORR"]
DEFAULT_MAX_LAG = 104
DEFAULT_PERIOD = 52


@dataclass(frozen=True)
class VariableSummary:
    variable: str
    acf_lag_52: float
    ci_half_width: float
    is_significant: bool
    is_positive: bool
    is_local_peak: bool
    seasonal_profile_range: float
    seasonal_profile_std: float
    seasonal_strength: float | None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Analyze annual seasonality in weekly integrated NASA POWER + USDM data."
    )
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("Integrated_weekly_NEB_20counties.csv"),
        help="Input weekly integrated CSV. Defaults to Nebraska 20-county dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("seasonality_analysis"),
        help="Directory for PNG figures and Markdown report.",
    )
    parser.add_argument(
        "--max-lag",
        type=int,
        default=DEFAULT_MAX_LAG,
        help="Maximum ACF lag in weeks. Use at least 104 for two years.",
    )
    parser.add_argument(
        "--period",
        type=int,
        default=DEFAULT_PERIOD,
        help="Seasonal period in weeks. Defaults to 52.",
    )
    return parser


def configure_plot_style() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "figure.dpi": 140,
            "savefig.dpi": 300,
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_dataset(data_path: Path) -> pd.DataFrame:
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    required = {"FIPS", "week_start", *VARIABLES}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"], errors="coerce")
    if df["week_start"].isna().any():
        bad_count = int(df["week_start"].isna().sum())
        raise ValueError(f"Found {bad_count} rows with invalid week_start values.")

    return df.sort_values(["FIPS", "week_start"]).reset_index(drop=True)


def aggregate_weekly_region(df: pd.DataFrame, variables: Iterable[str]) -> pd.DataFrame:
    """
    Average all counties per week to estimate one regional time series.

    This avoids treating county rows as independent temporal observations while
    preserving the regional annual signal used by the weekly model.
    """
    variables = list(variables)
    weekly = (
        df.groupby("week_start", as_index=False)[variables]
        .mean(numeric_only=True)
        .sort_values("week_start")
        .reset_index(drop=True)
    )
    return weekly


def regularize_weekly_series(weekly: pd.DataFrame, variables: Iterable[str]) -> pd.DataFrame:
    variables = list(variables)
    weekly = weekly.set_index("week_start").sort_index()
    inferred = pd.infer_freq(weekly.index)
    freq = inferred if inferred is not None else "7D"
    full_index = pd.date_range(weekly.index.min(), weekly.index.max(), freq=freq)
    regular = weekly.reindex(full_index)
    regular.index.name = "week_start"
    regular[variables] = regular[variables].interpolate(method="time").ffill().bfill()
    return regular.reset_index()


def compute_acf_values(values: Iterable[float], nlags: int) -> np.ndarray:
    series = pd.Series(values, dtype="float64").dropna().to_numpy()
    if len(series) <= nlags:
        raise ValueError(f"Need more than {nlags} observations to compute ACF to that lag.")

    centered = series - np.mean(series)
    denominator = float(np.dot(centered, centered))
    if denominator == 0.0:
        raise ValueError("Cannot compute ACF for a constant series.")

    acf = np.empty(nlags + 1, dtype="float64")
    acf[0] = 1.0
    for lag in range(1, nlags + 1):
        acf[lag] = float(np.dot(centered[:-lag], centered[lag:]) / denominator)
    return acf


def confidence_half_width(n_observations: int) -> float:
    return 1.96 / math.sqrt(n_observations)


def interpret_acf_at_lag(
    acf_value: float, ci_half_width: float, local_values: Iterable[float]
) -> dict[str, bool]:
    local_values = list(local_values)
    local_max = max(local_values) if local_values else acf_value
    return {
        "is_significant": abs(acf_value) > ci_half_width,
        "is_positive": acf_value > 0,
        "is_local_peak": acf_value >= local_max - 1e-12,
    }


def plot_acf(
    weekly: pd.DataFrame,
    variable: str,
    max_lag: int,
    period: int,
    output_path: Path,
) -> tuple[np.ndarray, float]:
    values = weekly[variable].to_numpy(dtype="float64")
    acf = compute_acf_values(values, max_lag)
    ci = confidence_half_width(len(values))
    lags = np.arange(max_lag + 1)

    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    ax.axhspan(-ci, ci, color="#4C72B0", alpha=0.14, label="95% confidence interval")
    ax.axhline(0, color="#333333", linewidth=0.9)
    ax.vlines(lags, 0, acf, color="#4C72B0", linewidth=1.1)
    ax.plot(lags, acf, marker="o", markersize=2.6, linewidth=0, color="#4C72B0")
    ax.axvline(period, color="#C44E52", linestyle="--", linewidth=1.6, label=f"Lag {period}")

    lag_value = acf[period]
    ax.scatter([period], [lag_value], color="#C44E52", s=45, zorder=5)
    y_offset = 0.08 if lag_value < 0.85 else -0.14
    ax.annotate(
        f"ACF({period}) = {lag_value:.3f}",
        xy=(period, lag_value),
        xytext=(period + 5, lag_value + y_offset),
        arrowprops={"arrowstyle": "->", "color": "#C44E52", "lw": 1.0},
        fontsize=10,
        color="#333333",
    )
    ax.set_title(f"Autocorrelation Function (ACF): {variable}")
    ax.set_xlabel("Lag (weeks)")
    ax.set_ylabel("Autocorrelation")
    ax.set_xlim(0, max_lag)
    ax.set_ylim(min(-1.0, acf.min() - 0.05), 1.05)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return acf, ci


def add_week_of_year(weekly: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    prof = weekly.copy()
    prof["week_of_year"] = prof["week_start"].dt.isocalendar().week.astype(int)
    dropped_week_53 = int((prof["week_of_year"] == 53).sum())
    prof = prof[prof["week_of_year"].between(1, 52)].copy()
    return prof, dropped_week_53


def seasonal_profile(weekly: pd.DataFrame, variables: Iterable[str]) -> tuple[pd.DataFrame, int]:
    prof, dropped_week_53 = add_week_of_year(weekly)
    profile = prof.groupby("week_of_year")[list(variables)].mean().reindex(range(1, 53))
    return profile, dropped_week_53


def plot_seasonal_profile(profile: pd.DataFrame, output_path: Path) -> None:
    colors = {"PRECTOTCORR": "#4C72B0", "T2M": "#C44E52", "RH2M": "#55A868"}
    fig, axes = plt.subplots(3, 1, figsize=(11, 9), sharex=True, constrained_layout=True)
    for ax, variable in zip(axes, VARIABLES):
        ax.plot(
            profile.index,
            profile[variable],
            color=colors[variable],
            linewidth=2.0,
            marker="o",
            markersize=3.0,
        )
        ax.set_title(f"Seasonal Profile: {variable}")
        ax.set_ylabel("Mean")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("ISO week of year")
    axes[-1].set_xlim(1, 52)
    axes[-1].set_xticks(range(1, 53, 4))
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def decompose_series(values: pd.Series, period: int) -> tuple[pd.DataFrame, str]:
    series = values.astype("float64").dropna()
    try:
        from statsmodels.tsa.seasonal import STL

        result = STL(series, period=period, robust=True).fit()
        components = pd.DataFrame(
            {
                "observed": series,
                "trend": result.trend,
                "seasonal": result.seasonal,
                "residual": result.resid,
            }
        )
        return components, "STL decomposition"
    except Exception:
        trend = series.rolling(window=period, center=True, min_periods=period // 2).mean()
        detrended = series - trend
        phase = pd.Series(np.arange(len(series)) % period, index=series.index)
        seasonal_lookup = detrended.groupby(phase).mean()
        seasonal_lookup = seasonal_lookup - seasonal_lookup.mean()
        seasonal = phase.map(seasonal_lookup).astype("float64")
        seasonal.index = series.index
        residual = series - trend - seasonal
        components = pd.DataFrame(
            {
                "observed": series,
                "trend": trend,
                "seasonal": seasonal,
                "residual": residual,
            }
        )
        return components, "classical additive decomposition fallback"


def seasonal_strength(components: pd.DataFrame) -> float | None:
    valid = components[["seasonal", "residual"]].dropna()
    if valid.empty:
        return None
    residual_var = float(np.var(valid["residual"], ddof=1))
    combined_var = float(np.var(valid["seasonal"] + valid["residual"], ddof=1))
    if combined_var == 0.0 or math.isnan(combined_var):
        return None
    return max(0.0, 1.0 - residual_var / combined_var)


def plot_decomposition(
    weekly: pd.DataFrame,
    variable: str,
    period: int,
    output_path: Path,
) -> tuple[float | None, str]:
    series = weekly.set_index("week_start")[variable]
    components, method = decompose_series(series, period)

    fig, axes = plt.subplots(4, 1, figsize=(11, 8.8), sharex=True, constrained_layout=True)
    component_config = [
        ("observed", "Observed", "#333333"),
        ("trend", "Trend", "#4C72B0"),
        ("seasonal", "Seasonal", "#55A868"),
        ("residual", "Residual", "#C44E52"),
    ]
    for ax, (column, title, color) in zip(axes, component_config):
        ax.plot(components.index, components[column], color=color, linewidth=1.6)
        ax.set_title(f"{title}: {variable}")
        ax.set_ylabel(title)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Week start")
    fig.suptitle(f"{method} (period = {period} weeks)", fontsize=13, fontweight="bold")
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    return seasonal_strength(components), method


def classify_evidence(summary: VariableSummary) -> str:
    score = 0
    if summary.is_significant and summary.is_positive:
        score += 1
    if summary.is_local_peak:
        score += 1
    if summary.seasonal_strength is not None and summary.seasonal_strength >= 0.45:
        score += 1

    if score >= 3:
        return "strong"
    if score == 2:
        return "moderate"
    if score == 1:
        return "weak"
    return "limited"


def interpretation_sentence(summary: VariableSummary) -> str:
    evidence = classify_evidence(summary)
    significance = "outside" if summary.is_significant else "inside"
    peak_text = "a local maximum" if summary.is_local_peak else "not a local maximum"
    strength_text = (
        "not computed"
        if summary.seasonal_strength is None
        else f"{summary.seasonal_strength:.3f}"
    )
    return (
        f"- **{summary.variable}**: ACF at lag 52 = {summary.acf_lag_52:.3f}, "
        f"{significance} the 95% CI (+/- {summary.ci_half_width:.3f}), and is "
        f"{peak_text} in lags 50-54. Seasonal profile range = "
        f"{summary.seasonal_profile_range:.3f}; seasonal strength = {strength_text}. "
        f"Overall evidence: **{evidence}**."
    )


def write_report(
    output_path: Path,
    data_path: Path,
    n_rows: int,
    n_counties: int,
    date_min: pd.Timestamp,
    date_max: pd.Timestamp,
    dropped_week_53: int,
    decomposition_methods: dict[str, str],
    summaries: list[VariableSummary],
) -> None:
    evidence_counts = {level: 0 for level in ["strong", "moderate", "weak", "limited"]}
    for summary in summaries:
        evidence_counts[classify_evidence(summary)] += 1

    annual_supported = evidence_counts["strong"] + evidence_counts["moderate"] >= 2
    conclusion = (
        "Penggunaan sequence length 52 minggu memiliki dasar empiris yang layak, "
        "karena sebagian besar variabel utama menunjukkan indikasi siklus tahunan."
        if annual_supported
        else "Penggunaan sequence length 52 minggu perlu diposisikan sebagai asumsi "
        "berbasis kalender/agronomi, karena bukti statistik seasonality tahunan pada "
        "variabel utama tidak konsisten kuat."
    )

    lines = [
        "# Seasonality Analysis Report",
        "",
        "## Metode",
        "",
        f"- Dataset: `{data_path}`.",
        f"- Jumlah baris: {n_rows:,}; jumlah county unik: {n_counties:,}.",
        f"- Rentang waktu: {date_min.date()} sampai {date_max.date()}.",
        "- Unit analisis: rata-rata seluruh county per `week_start` sebelum ACF, profil musiman, dan dekomposisi.",
        "- Alasan agregasi: setiap county berbagi kalender mingguan yang sama, sehingga menghitung ACF pada seluruh baris county dapat memperlakukan observasi spasial sebagai replikasi temporal. Rata-rata regional mengurangi noise lokal dan menguji seasonality temporal dataset secara lebih langsung.",
        "- ACF dihitung sampai lag 104 minggu. Confidence interval menggunakan pendekatan white-noise `+/- 1.96/sqrt(n)`.",
        "- Seasonal profile dihitung dari rata-rata setiap ISO week-of-year 1-52.",
        f"- Baris dengan ISO week 53 yang tidak masuk profil 1-52: {dropped_week_53}.",
        "- Seasonal decomposition memakai STL dengan `period=52` jika `statsmodels` tersedia; jika tidak, script memakai fallback dekomposisi aditif klasik berbasis moving average 52 minggu.",
        "",
        "## Output Grafik",
        "",
        "- `acf_prec.png`: ACF PRECTOTCORR.",
        "- `acf_t2m.png`: ACF T2M.",
        "- `acf_rh2m.png`: ACF RH2M.",
        "- `seasonal_profile.png`: profil rata-rata week-of-year.",
        "- `decomposition_t2m.png`: trend, seasonal, residual T2M.",
        "- `decomposition_prec.png`: trend, seasonal, residual PRECTOTCORR.",
        "",
        "## Interpretasi ACF dan Profil Musiman",
        "",
    ]
    lines.extend(interpretation_sentence(summary) for summary in summaries)
    lines.extend(
        [
            "",
            "## Metode Dekomposisi yang Dipakai",
            "",
        ]
    )
    for variable, method in decomposition_methods.items():
        lines.append(f"- **{variable}**: {method} dengan period 52 minggu.")

    lines.extend(
        [
            "",
            "## Kesimpulan",
            "",
            conclusion,
            "",
            "Kesimpulan ini bersifat objektif terhadap hasil deskriptif: lag 52 yang signifikan dan profil week-of-year yang jelas mendukung sequence 52 minggu, sedangkan variabel dengan ACF lag 52 lemah menunjukkan bahwa seasonality tahunan tidak sama kuat pada semua sinyal cuaca.",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")


def run_analysis(data_path: Path, output_dir: Path, max_lag: int, period: int) -> list[VariableSummary]:
    if max_lag < period * 2:
        raise ValueError(f"max_lag should be at least {period * 2} for two annual cycles.")

    configure_plot_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(data_path)
    weekly = aggregate_weekly_region(df, VARIABLES)
    weekly = regularize_weekly_series(weekly, VARIABLES)
    profile, dropped_week_53 = seasonal_profile(weekly, VARIABLES)
    plot_seasonal_profile(profile, output_dir / "seasonal_profile.png")

    output_names = {
        "PRECTOTCORR": "acf_prec.png",
        "T2M": "acf_t2m.png",
        "RH2M": "acf_rh2m.png",
    }
    summaries: list[VariableSummary] = []
    acf_results: dict[str, tuple[np.ndarray, float]] = {}
    for variable in VARIABLES:
        acf, ci = plot_acf(weekly, variable, max_lag, period, output_dir / output_names[variable])
        acf_results[variable] = (acf, ci)

    decomposition_methods: dict[str, str] = {}
    strengths: dict[str, float | None] = {variable: None for variable in VARIABLES}
    decomposition_outputs = {
        "T2M": "decomposition_t2m.png",
        "PRECTOTCORR": "decomposition_prec.png",
    }
    for variable in DECOMPOSITION_VARIABLES:
        strength, method = plot_decomposition(
            weekly, variable, period, output_dir / decomposition_outputs[variable]
        )
        strengths[variable] = strength
        decomposition_methods[variable] = method

    for variable in VARIABLES:
        acf, ci = acf_results[variable]
        local_start = max(0, period - 2)
        local_end = min(len(acf), period + 3)
        acf_flags = interpret_acf_at_lag(acf[period], ci, acf[local_start:local_end])
        profile_values = profile[variable].dropna()
        summaries.append(
            VariableSummary(
                variable=variable,
                acf_lag_52=float(acf[period]),
                ci_half_width=float(ci),
                is_significant=acf_flags["is_significant"],
                is_positive=acf_flags["is_positive"],
                is_local_peak=acf_flags["is_local_peak"],
                seasonal_profile_range=float(profile_values.max() - profile_values.min()),
                seasonal_profile_std=float(profile_values.std(ddof=1)),
                seasonal_strength=strengths.get(variable),
            )
        )

    write_report(
        output_dir / "seasonality_report.md",
        data_path=data_path,
        n_rows=len(df),
        n_counties=int(df["FIPS"].nunique()),
        date_min=df["week_start"].min(),
        date_max=df["week_start"].max(),
        dropped_week_53=dropped_week_53,
        decomposition_methods=decomposition_methods,
        summaries=summaries,
    )

    print("Seasonality analysis completed.")
    print(f"Data: {data_path}")
    print(f"Output directory: {output_dir}")
    for summary in summaries:
        print(
            f"{summary.variable}: ACF(52)={summary.acf_lag_52:.3f}, "
            f"CI=+/-{summary.ci_half_width:.3f}, evidence={classify_evidence(summary)}"
        )
    return summaries


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_analysis(args.data_path, args.output_dir, args.max_lag, args.period)


if __name__ == "__main__":
    main()
