"""Visualize weekly drought predictions from saved BiLSTM models.

This utility rebuilds the deterministic preprocessing used by the weekly
scenario scripts, loads existing ``best_model.keras`` files, and writes
per-week prediction CSV/PNG outputs. It does not train or fine-tune models.
"""

from __future__ import annotations

import argparse
import ast
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator


SEED = 42
SEQ_LENGTH = 52
TRAIN_END_DATE = "2019-12-31"
TEST_START_DATE = "2022-01-01"

DROUGHT_COLS = ["None", "D0", "D1", "D2", "D3", "D4"]
PMF_COLS = ["PMF_None", "PMF_D0", "PMF_D1", "PMF_D2", "PMF_D3", "PMF_D4"]
LABEL_MAP = {0: "None", 1: "D0", 2: "D1", 3: "D2", 4: "D3", 5: "D4"}
CLASS_NAMES = [LABEL_MAP[i] for i in range(6)]

WEATHER_FEATURES = ["ALLSKY_SFC_SW_DWN", "PRECTOTCORR", "PS", "RH2M", "T2M", "WS2M"]
WEATHER_LAG_FEATURES = [
    "PREC_lag1",
    "PREC_lag2",
    "PREC_lag4",
    "PREC_lag8",
    "T2M_lag1",
    "T2M_lag2",
    "T2M_lag4",
    "T2M_lag8",
    "RH2M_lag1",
    "RH2M_lag2",
    "RH2M_lag4",
    "RH2M_lag8",
]
ROLLING_FEATURES = [
    "PREC_roll4_mean",
    "PREC_roll4_std",
    "PREC_roll12_mean",
    "PREC_roll12_std",
    "T2M_roll4_mean",
    "T2M_roll12_mean",
]
SEASON_FEATURES = ["week_sin", "week_cos"]
DROUGHT_LAG_FEATURES = [
    "None_lag1",
    "D0_lag1",
    "D1_lag1",
    "D2_lag1",
    "D3_lag1",
    "D4_lag1",
    "None_lag2",
    "D0_lag2",
    "D1_lag2",
    "D2_lag2",
    "D3_lag2",
    "D4_lag2",
]
BASELINE_FEATURES = (
    WEATHER_FEATURES + WEATHER_LAG_FEATURES + ROLLING_FEATURES + SEASON_FEATURES + DROUGHT_LAG_FEATURES
)


@dataclass(frozen=True)
class RegionConfig:
    name: str
    data_file: str
    folder: str


@dataclass(frozen=True)
class ScenarioConfig:
    name: str
    output_folder_name: str


REGIONS = {
    "kansas": RegionConfig("kansas", "Integrated_weekly_KAN_20counties.csv", "kansas"),
    "nebraska": RegionConfig("nebraska", "Integrated_weekly_NEB_20counties.csv", "nebraska"),
}

SCENARIOS = {
    "scenario1": "output_weekly_{region}_20counties",
    "scenario2": "output_weekly_{region}_scenario2",
    "scenario2A": "output_weekly_{region}_scenario2A",
    "scenario2B": "output_weekly_{region}_scenario2B",
    "scenario2C": "output_weekly_{region}_scenario2C",
    "scenario3": "output_weekly_{region}_scenario3",
    "scenario4": "output_weekly_{region}_scenario4",
    "scenario5": "output_weekly_{region}_scenario5",
    "scenario6": "output_weekly_{region}_scenario6",
    "scenario7": "output_weekly_{region}_scenario7",
}
SCENARIO2_VARIANTS = ["scenario2", "scenario2A", "scenario2B", "scenario2C"]


def decumulate_drought(row: pd.Series) -> pd.Series:
    pmf_d4 = row["D4"]
    pmf_d3 = max(0.0, row["D3"] - row["D4"])
    pmf_d2 = max(0.0, row["D2"] - row["D3"])
    pmf_d1 = max(0.0, row["D1"] - row["D2"])
    pmf_d0 = max(0.0, row["D0"] - row["D1"])
    pmf_none = max(0.0, row["None"])
    return pd.Series([pmf_none, pmf_d0, pmf_d1, pmf_d2, pmf_d3, pmf_d4])


def add_drought_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[PMF_COLS] = df.apply(decumulate_drought, axis=1)
    df["PMF_Sum"] = df[PMF_COLS].sum(axis=1)
    df["Label"] = df[PMF_COLS].idxmax(axis=1).apply(lambda name: PMF_COLS.index(name))
    return df


def add_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    df_fe = df.copy().sort_values(["FIPS", "week_start"]).reset_index(drop=True)

    for lag in [1, 2, 4, 8]:
        df_fe[f"PREC_lag{lag}"] = df_fe.groupby("FIPS")["PRECTOTCORR"].shift(lag)
        df_fe[f"T2M_lag{lag}"] = df_fe.groupby("FIPS")["T2M"].shift(lag)
        df_fe[f"RH2M_lag{lag}"] = df_fe.groupby("FIPS")["RH2M"].shift(lag)

    for window in [4, 12]:
        df_fe[f"PREC_roll{window}_mean"] = df_fe.groupby("FIPS")["PRECTOTCORR"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )
        df_fe[f"PREC_roll{window}_std"] = df_fe.groupby("FIPS")["PRECTOTCORR"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0.0)
        )
        df_fe[f"T2M_roll{window}_mean"] = df_fe.groupby("FIPS")["T2M"].transform(
            lambda x: x.shift(1).rolling(window, min_periods=1).mean()
        )

    iso_week = df_fe["week_start"].dt.isocalendar().week.astype(int)
    df_fe["week_sin"] = np.sin(2 * np.pi * iso_week / 52.0)
    df_fe["week_cos"] = np.cos(2 * np.pi * iso_week / 52.0)

    for col in DROUGHT_COLS:
        df_fe[f"{col}_lag1"] = df_fe.groupby("FIPS")[col].shift(1)
        df_fe[f"{col}_lag2"] = df_fe.groupby("FIPS")[col].shift(2)

    return df_fe


def read_csv_features(csv_path: Path, model_feature_count: int | None = None) -> list[str]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Feature selection file not found: {csv_path}")

    frame = pd.read_csv(csv_path)
    if "rank" in frame.columns:
        frame = frame.sort_values("rank")
    if "feature" not in frame.columns:
        raise ValueError(f"Feature selection file has no 'feature' column: {csv_path}")

    features = frame["feature"].astype(str).tolist()
    if model_feature_count is not None:
        features = features[:model_feature_count]
    return features


def parse_selected_features_from_summary(summary_path: Path) -> list[str] | None:
    if not summary_path.exists():
        return None

    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Selected features:\s*(\[[^\n]+\])", text)
    if not match:
        return None

    parsed = ast.literal_eval(match.group(1))
    return [str(item) for item in parsed]


def parse_class_multipliers(summary_path: Path) -> list[float]:
    if not summary_path.exists():
        return [1.0] * 6

    text = summary_path.read_text(encoding="utf-8", errors="ignore")
    match = re.search(r"Class multipliers:\s*(\[[^\n]+\])", text)
    if not match:
        return [1.0] * 6

    values = ast.literal_eval(match.group(1))
    if len(values) != 6:
        raise ValueError(f"Expected 6 class multipliers in {summary_path}, got {len(values)}")
    return [float(value) for value in values]


def get_scenario_features(
    scenario: str,
    output_folder: Path,
    model_feature_count: int | None = None,
) -> list[str]:
    if scenario in {"scenario1"}:
        return list(BASELINE_FEATURES)
    if scenario in {"scenario2", "scenario2A", "scenario2B", "scenario2C"}:
        return read_csv_features(output_folder / "selected_feature_ranking.csv", model_feature_count)
    if scenario == "scenario3":
        selected = parse_selected_features_from_summary(output_folder / "results_summary.txt")
        if selected is not None:
            return selected
        return read_csv_features(output_folder / "permutation_importance.csv", model_feature_count)
    if scenario == "scenario4":
        return list(WEATHER_FEATURES)
    if scenario == "scenario5":
        return list(WEATHER_FEATURES + WEATHER_LAG_FEATURES + SEASON_FEATURES)
    if scenario == "scenario6":
        return list(WEATHER_FEATURES + WEATHER_LAG_FEATURES + ROLLING_FEATURES + SEASON_FEATURES)
    if scenario == "scenario7":
        return list(DROUGHT_LAG_FEATURES)
    raise ValueError(f"Unknown scenario: {scenario}")


def load_region_frame(repo_root: Path, region: RegionConfig) -> pd.DataFrame:
    data_path = repo_root / region.data_file
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    if "ValidEnd" in df.columns:
        df["ValidEnd"] = pd.to_datetime(df["ValidEnd"])
    df = df.sort_values(["FIPS", "week_start"]).reset_index(drop=True)
    return add_weekly_features(add_drought_labels(df))


def scale_features(df_fe: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    missing = [col for col in feature_cols if col not in df_fe.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    df_scaled = df_fe.dropna(subset=feature_cols + ["Label"]).reset_index(drop=True).copy()
    train_df = df_scaled[df_scaled["week_start"] <= TRAIN_END_DATE]
    if train_df.empty:
        raise ValueError("No training rows available for scaler fit.")

    train_min = train_df[feature_cols].min()
    train_range = train_df[feature_cols].max() - train_min
    safe_range = train_range.replace(0, np.nan)
    scaled_values = (df_scaled[feature_cols] - train_min) / safe_range
    df_scaled.loc[:, feature_cols] = scaled_values.fillna(0.0)
    return df_scaled


def create_sequences_with_metadata(
    df_input: pd.DataFrame,
    feature_columns: list[str],
    seq_length: int = SEQ_LENGTH,
    start_date: str | None = TEST_START_DATE,
    end_date: str | None = None,
    id_col: str = "FIPS",
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    x_values: list[np.ndarray] = []
    y_values: list[int] = []
    metadata: list[dict[str, object]] = []

    for fips, group in df_input.groupby(id_col):
        group = group.sort_values("week_start")
        feats = group[feature_columns].values
        labels = group["Label"].values
        dates = group["week_start"].values
        if len(group) < seq_length:
            continue

        for i in range(seq_length - 1, len(group)):
            target_week = pd.Timestamp(dates[i])
            if start_date is not None and target_week < pd.Timestamp(start_date):
                continue
            if end_date is not None and target_week > pd.Timestamp(end_date):
                continue

            x_values.append(feats[i - seq_length + 1 : i + 1])
            y_values.append(int(labels[i]))
            metadata.append(
                {
                    "FIPS": fips,
                    "target_week": target_week,
                    "actual_label": int(labels[i]),
                }
            )

    return np.array(x_values), np.array(y_values), pd.DataFrame(metadata)


def build_prediction_frame(
    metadata: pd.DataFrame,
    probabilities: np.ndarray,
    multipliers: Iterable[float] | None = None,
) -> pd.DataFrame:
    class_multipliers = np.array(list(multipliers) if multipliers is not None else [1.0] * 6)
    if class_multipliers.shape != (6,):
        raise ValueError("multipliers must contain exactly 6 values.")

    raw_labels = np.argmax(probabilities, axis=1)
    tuned_labels = np.argmax(probabilities * class_multipliers, axis=1)

    frame = metadata.copy()
    frame["actual_class"] = frame["actual_label"].map(LABEL_MAP)
    frame["pred_label_raw"] = raw_labels
    frame["pred_class_raw"] = [LABEL_MAP[int(label)] for label in raw_labels]
    frame["pred_label"] = tuned_labels
    frame["pred_class"] = [LABEL_MAP[int(label)] for label in tuned_labels]
    frame["is_correct"] = frame["actual_label"].to_numpy() == tuned_labels

    for idx, name in LABEL_MAP.items():
        frame[f"prob_{name}"] = probabilities[:, idx]

    return frame


def select_week_window(
    predictions: pd.DataFrame,
    week_count: int = 16,
    position: str = "last",
) -> pd.DataFrame:
    unique_weeks = pd.Index(pd.to_datetime(predictions["target_week"].drop_duplicates())).sort_values()
    if len(unique_weeks) <= week_count:
        selected_weeks = unique_weeks
    elif position == "first":
        selected_weeks = unique_weeks[:week_count]
    elif position == "middle":
        start = (len(unique_weeks) - week_count) // 2
        selected_weeks = unique_weeks[start : start + week_count]
    elif position == "last":
        selected_weeks = unique_weeks[-week_count:]
    else:
        raise ValueError("position must be one of: first, middle, last")

    return predictions[pd.to_datetime(predictions["target_week"]).isin(selected_weeks)].copy()


def build_weekly_class_counts(predictions: pd.DataFrame) -> pd.DataFrame:
    plot_data = predictions.copy()
    plot_data["target_week"] = pd.to_datetime(plot_data["target_week"])
    plot_data["pred_class"] = plot_data["pred_class"].fillna("None")
    return (
        plot_data.groupby(["target_week", "pred_class"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=CLASS_NAMES, fill_value=0)
    )


def plot_weekly_class_counts(predictions: pd.DataFrame, out_path: Path, title: str) -> None:
    counts = build_weekly_class_counts(predictions)

    fig, ax = plt.subplots(figsize=(18, 7))
    x_positions = np.arange(len(counts.index))
    bottoms = np.zeros(len(counts.index), dtype=float)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(CLASS_NAMES)))

    for class_name, color in zip(CLASS_NAMES, colors):
        values = counts[class_name].to_numpy(dtype=float)
        ax.bar(x_positions, values, bottom=bottoms, label=class_name, color=color)
        bottoms += values

    ax.set_title(title)
    ax.set_xlabel("Predicted week")
    ax.set_ylabel("County count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(title="Predicted class", ncol=6, loc="upper center", bbox_to_anchor=(0.5, -0.16))
    tick_labels = [pd.Timestamp(idx).strftime("%Y-%m-%d") for idx in counts.index]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=7)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return ax


def plot_class_counts_on_axis(ax, predictions: pd.DataFrame, title: str) -> None:
    counts = build_weekly_class_counts(predictions)
    x_positions = np.arange(len(counts.index))
    bottoms = np.zeros(len(counts.index), dtype=float)
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(CLASS_NAMES)))

    for class_name, color in zip(CLASS_NAMES, colors):
        values = counts[class_name].to_numpy(dtype=float)
        ax.bar(x_positions, values, bottom=bottoms, label=class_name, color=color)
        bottoms += values

    tick_labels = [pd.Timestamp(idx).strftime("%Y-%m-%d") for idx in counts.index]
    ax.set_title(title)
    ax.set_xlabel("Predicted week")
    ax.set_ylabel("County count")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)


def plot_combined_scenario_class_counts(
    scenario_predictions: dict[str, pd.DataFrame],
    out_path: Path,
    title: str,
) -> list:
    fig, axes_grid = plt.subplots(2, 2, figsize=(22, 12), sharey=True)
    axes = list(axes_grid.ravel())

    for ax, scenario in zip(axes, SCENARIO2_VARIANTS):
        if scenario not in scenario_predictions:
            raise ValueError(f"Missing predictions for {scenario}")
        plot_class_counts_on_axis(ax, scenario_predictions[scenario], scenario)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.suptitle(title, fontsize=16)
    fig.legend(handles, labels, title="Predicted class", ncol=6, loc="lower center")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    return axes


def plot_weekly_accuracy(predictions: pd.DataFrame, out_path: Path, title: str) -> None:
    accuracy = predictions.groupby("target_week")["is_correct"].mean()
    sample_count = predictions.groupby("target_week").size()

    fig, ax1 = plt.subplots(figsize=(18, 6))
    ax1.plot(accuracy.index, accuracy.values, marker="o", linewidth=1.5, color="#2f5597")
    ax1.set_ylim(0, 1.05)
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Predicted week")
    ax1.set_title(title)
    ax1.grid(alpha=0.3)

    ax2 = ax1.twinx()
    ax2.bar(sample_count.index, sample_count.values, alpha=0.18, color="#888888", width=5)
    ax2.set_ylabel("County predictions")

    fig.autofmt_xdate(rotation=45)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def plot_county_week_heatmap(predictions: pd.DataFrame, out_path: Path, title: str) -> None:
    pivot = predictions.pivot_table(
        index="FIPS",
        columns="target_week",
        values="pred_label",
        aggfunc="first",
    ).sort_index()

    fig, ax = plt.subplots(figsize=(18, max(5, len(pivot) * 0.35)))
    image = ax.imshow(pivot.values, aspect="auto", interpolation="nearest", cmap="YlOrRd", vmin=0, vmax=5)
    ax.set_title(title)
    ax.set_xlabel("Predicted week")
    ax.set_ylabel("FIPS")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([str(value) for value in pivot.index], fontsize=8)

    columns = list(pivot.columns)
    step = max(1, len(columns) // 18)
    tick_positions = np.arange(0, len(columns), step)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([pd.Timestamp(columns[i]).strftime("%Y-%m-%d") for i in tick_positions], rotation=90, fontsize=7)

    cbar = fig.colorbar(image, ax=ax, ticks=range(6))
    cbar.ax.set_yticklabels(CLASS_NAMES)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


def get_model_feature_count(model_path: Path) -> int:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    return int(model.input_shape[-1])


def predict_probabilities(model_path: Path, x_seq: np.ndarray) -> np.ndarray:
    import tensorflow as tf

    model = tf.keras.models.load_model(model_path, compile=False)
    return model.predict(x_seq, verbose=0)


def output_folder_for(repo_root: Path, region: str, scenario: str) -> Path:
    region_config = REGIONS[region]
    folder_name = SCENARIOS[scenario].format(region=region)
    return repo_root / region_config.folder / folder_name


def run_scenario(
    repo_root: Path,
    output_root: Path,
    region: str,
    scenario: str,
    dry_run: bool = False,
) -> None:
    region_config = REGIONS[region]
    model_output_folder = output_folder_for(repo_root, region, scenario)
    model_path = model_output_folder / "best_model.keras"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model_feature_count = None if dry_run else get_model_feature_count(model_path)
    feature_cols = get_scenario_features(scenario, model_output_folder, model_feature_count)
    if model_feature_count is not None and len(feature_cols) != model_feature_count:
        raise ValueError(
            f"{region}/{scenario}: feature count mismatch, "
            f"features={len(feature_cols)} model={model_feature_count}"
        )

    df_fe = load_region_frame(repo_root, region_config)
    df_scaled = scale_features(df_fe, feature_cols)
    x_seq, _, metadata = create_sequences_with_metadata(df_scaled, feature_cols, start_date=TEST_START_DATE)

    destination = output_root / region / scenario
    destination.mkdir(parents=True, exist_ok=True)

    if dry_run:
        print(
            f"[dry-run] {region}/{scenario}: {len(feature_cols)} features, "
            f"{x_seq.shape[0]} prediction windows -> {destination}"
        )
        return

    if x_seq.size == 0:
        raise ValueError(f"{region}/{scenario}: no prediction windows created.")

    probabilities = predict_probabilities(model_path, x_seq)
    multipliers = parse_class_multipliers(model_output_folder / "results_summary.txt")
    predictions = build_prediction_frame(metadata, probabilities, multipliers)

    csv_path = destination / "weekly_predictions.csv"
    predictions.to_csv(csv_path, index=False)

    plot_weekly_class_counts(
        predictions,
        destination / "weekly_predicted_class_counts.png",
        f"{region.title()} {scenario}: predicted drought class per week",
    )
    plot_weekly_class_counts(
        select_week_window(predictions, week_count=16, position="last"),
        destination / "weekly_predicted_class_counts_16weeks.png",
        f"{region.title()} {scenario}: predicted drought class per week (last 16 weeks)",
    )
    plot_weekly_accuracy(
        predictions,
        destination / "weekly_accuracy.png",
        f"{region.title()} {scenario}: weekly prediction accuracy",
    )
    plot_county_week_heatmap(
        predictions,
        destination / "county_week_predicted_class_heatmap.png",
        f"{region.title()} {scenario}: county-week predicted class",
    )

    print(f"{region}/{scenario}: wrote {csv_path}")


def load_prediction_csv(csv_path: Path) -> pd.DataFrame:
    return pd.read_csv(csv_path, parse_dates=["target_week"])


def write_combined_scenario2_visuals(output_root: Path, region: str) -> list[Path]:
    scenario_predictions = {}
    for scenario in SCENARIO2_VARIANTS:
        csv_path = output_root / region / scenario / "weekly_predictions.csv"
        if not csv_path.exists():
            return []
        scenario_predictions[scenario] = load_prediction_csv(csv_path)

    destination = output_root / region / "scenario2_combined"
    destination.mkdir(parents=True, exist_ok=True)

    full_path = destination / "weekly_predicted_class_counts_scenario2_variants.png"
    plot_combined_scenario_class_counts(
        scenario_predictions,
        full_path,
        f"{region.title()}: scenario 2 variants predicted drought class per week",
    )

    short_predictions = {
        scenario: select_week_window(predictions, week_count=16, position="last")
        for scenario, predictions in scenario_predictions.items()
    }
    short_path = destination / "weekly_predicted_class_counts_scenario2_variants_16weeks.png"
    plot_combined_scenario_class_counts(
        short_predictions,
        short_path,
        f"{region.title()}: scenario 2 variants predicted drought class per week (last 16 weeks)",
    )

    return [full_path, short_path]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "output")
    parser.add_argument("--regions", nargs="+", default=["kansas", "nebraska"], choices=sorted(REGIONS))
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["all"],
        help="Scenario keys, e.g. scenario1 scenario7, or all.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate preprocessing without loading models/predicting.")
    return parser.parse_args()


def normalize_scenarios(values: list[str]) -> list[str]:
    if values == ["all"]:
        return list(SCENARIOS)
    unknown = [value for value in values if value not in SCENARIOS]
    if unknown:
        raise ValueError(f"Unknown scenario(s): {unknown}. Valid: {list(SCENARIOS)}")
    return values


def main() -> None:
    args = parse_args()
    repo_root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    scenarios = normalize_scenarios(args.scenarios)

    for region in args.regions:
        for scenario in scenarios:
            run_scenario(repo_root, output_root, region, scenario, dry_run=args.dry_run)
        if not args.dry_run and all(scenario in scenarios for scenario in SCENARIO2_VARIANTS):
            written = write_combined_scenario2_visuals(output_root, region)
            for path in written:
                print(f"{region}/scenario2_combined: wrote {path}")


if __name__ == "__main__":
    main()
