from __future__ import annotations

import argparse
import csv
from pathlib import Path

SEED = 42
SEQ_LENGTH = 52
TRAIN_END_DATE = "2019-12-31"
VAL_START_DATE = "2020-01-01"
VAL_END_DATE = "2021-12-31"
TEST_START_DATE = "2022-01-01"
LABEL_MAP = {0: "None", 1: "D0", 2: "D1", 3: "D2", 4: "D3", 5: "D4"}
SCENARIO_ORDER = ["1", "2", "2A", "2B", "2C", "3", "4", "5", "6", "7"]
SCENARIO_NAMES = {
    "1": "Scenario 1: Baseline",
    "2": "Scenario 2: Correlation-aware Feature Selection",
    "2A": "Scenario 2A: Correlation-aware Feature Selection",
    "2B": "Scenario 2B: Correlation-aware Feature Selection",
    "2C": "Scenario 2C: Correlation-aware Feature Selection",
    "3": "Scenario 3: Macro F1-driven Feature Selection",
    "4": "Scenario 4 - Weather Only",
    "5": "Scenario 5 - Weather + Lag Only",
    "6": "Scenario 6 - No Drought History",
    "7": "Scenario 7 - Drought History Only",
}
SCENARIO_ALIASES = {
    "1": "1",
    "SCENARIO1": "1",
    "2": "2",
    "SCENARIO2": "2",
    "2A": "2A",
    "SCENARIO2A": "2A",
    "2B": "2B",
    "SCENARIO2B": "2B",
    "2C": "2C",
    "SCENARIO2C": "2C",
    "3": "3",
    "SCENARIO3": "3",
    "4": "4",
    "SCENARIO4": "4",
    "5": "5",
    "SCENARIO5": "5",
    "6": "6",
    "SCENARIO6": "6",
    "7": "7",
    "SCENARIO7": "7",
    "ALL": "ALL",
}
BASELINE_FEATURES = [
    "ALLSKY_SFC_SW_DWN", "PRECTOTCORR", "PS", "RH2M", "T2M", "WS2M",
    "PREC_lag1", "PREC_lag2", "PREC_lag4", "PREC_lag8",
    "T2M_lag1", "T2M_lag2", "T2M_lag4", "T2M_lag8",
    "RH2M_lag1", "RH2M_lag2", "RH2M_lag4", "RH2M_lag8",
    "PREC_roll4_mean", "PREC_roll4_std", "PREC_roll12_mean", "PREC_roll12_std",
    "T2M_roll4_mean", "T2M_roll12_mean",
    "week_sin", "week_cos",
    "None_lag1", "D0_lag1", "D1_lag1", "D2_lag1", "D3_lag1", "D4_lag1",
    "None_lag2", "D0_lag2", "D1_lag2", "D2_lag2", "D3_lag2", "D4_lag2",
    "heat_dry_stress",
]
SCENARIO_FEATURES = {
    "1": BASELINE_FEATURES,
    "4": ["ALLSKY_SFC_SW_DWN", "PRECTOTCORR", "PS", "RH2M", "T2M", "WS2M"],
    "5": [
        "ALLSKY_SFC_SW_DWN", "PRECTOTCORR", "PS", "RH2M", "T2M", "WS2M",
        "PREC_lag1", "PREC_lag2", "PREC_lag4", "PREC_lag8",
        "T2M_lag1", "T2M_lag2", "T2M_lag4", "T2M_lag8",
        "RH2M_lag1", "RH2M_lag2", "RH2M_lag4", "RH2M_lag8",
        "week_sin", "week_cos",
    ],
    "6": [
        "ALLSKY_SFC_SW_DWN", "PRECTOTCORR", "PS", "RH2M", "T2M", "WS2M",
        "PREC_lag1", "PREC_lag2", "PREC_lag4", "PREC_lag8",
        "T2M_lag1", "T2M_lag2", "T2M_lag4", "T2M_lag8",
        "RH2M_lag1", "RH2M_lag2", "RH2M_lag4", "RH2M_lag8",
        "PREC_roll4_mean", "PREC_roll4_std", "PREC_roll12_mean", "PREC_roll12_std",
        "T2M_roll4_mean", "T2M_roll12_mean",
        "week_sin", "week_cos",
        "heat_dry_stress",
    ],
    "7": [
        "None_lag1", "D0_lag1", "D1_lag1", "D2_lag1", "D3_lag1", "D4_lag1",
        "None_lag2", "D0_lag2", "D1_lag2", "D2_lag2", "D3_lag2", "D4_lag2",
    ],
}


def normalize_scenario_key(value: str) -> str:
    key = value.strip().upper().replace("-", "")
    if key not in SCENARIO_ALIASES:
        raise ValueError(f"Unsupported scenario '{value}'. Choose from {', '.join(SCENARIO_ORDER)} or 'all'.")
    return SCENARIO_ALIASES[key]


def scenario_output_dir_name(key: str) -> str:
    if key == "1":
        return "output_weekly_kansas_20counties"
    return f"output_weekly_kansas_scenario{key}"


def load_selected_features_from_csv(csv_path: Path) -> list[str]:
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "feature" not in reader.fieldnames:
            raise ValueError(f"Expected 'feature' column in {csv_path}")
        return [row["feature"] for row in reader]


def load_feature_list_from_summary(summary_path: Path) -> list[str]:
    lines = summary_path.read_text(encoding="utf-8").splitlines()
    capture = False
    features: list[str] = []
    for line in lines:
        if line.strip() == "Selected features:":
            capture = True
            continue
        if not capture:
            continue
        if not line.startswith("  "):
            break
        features.append(line.strip())
    if not features:
        raise ValueError(f"Could not parse selected features from {summary_path}")
    return features


def get_feature_columns_for_scenario(scenario_key: str, output_dir: Path) -> list[str]:
    if scenario_key in {"2", "2A", "2B", "2C"}:
        return load_selected_features_from_csv(output_dir / "selected_feature_ranking.csv")
    if scenario_key == "3":
        return load_feature_list_from_summary(output_dir / "results_summary.txt")
    return SCENARIO_FEATURES[scenario_key]


def decumulate_drought(row):
    import pandas as pd

    pmf_d4 = row["D4"]
    pmf_d3 = max(0.0, row["D3"] - row["D4"])
    pmf_d2 = max(0.0, row["D2"] - row["D3"])
    pmf_d1 = max(0.0, row["D1"] - row["D2"])
    pmf_d0 = max(0.0, row["D0"] - row["D1"])
    pmf_none = max(0.0, row["None"])
    return pd.Series([pmf_none, pmf_d0, pmf_d1, pmf_d2, pmf_d3, pmf_d4])


def build_base_feature_frame(data_path: Path):
    import numpy as np
    import pandas as pd

    df = pd.read_csv(data_path)
    df["week_start"] = pd.to_datetime(df["week_start"])
    df["ValidEnd"] = pd.to_datetime(df["ValidEnd"])
    df = df.sort_values(["FIPS", "week_start"]).reset_index(drop=True)

    pmf_cols = ["PMF_None", "PMF_D0", "PMF_D1", "PMF_D2", "PMF_D3", "PMF_D4"]
    df[pmf_cols] = df.apply(decumulate_drought, axis=1)
    df["PMF_Sum"] = df[pmf_cols].sum(axis=1)
    df["Label"] = df[pmf_cols].idxmax(axis=1).apply(lambda x: pmf_cols.index(x))

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

    for col in ["None", "D0", "D1", "D2", "D3", "D4"]:
        df_fe[f"{col}_lag1"] = df_fe.groupby("FIPS")[col].shift(1)
        df_fe[f"{col}_lag2"] = df_fe.groupby("FIPS")[col].shift(2)

    df_fe["heat_dry_stress"] = df_fe["T2M"] * (1.0 - df_fe["RH2M"] / 100.0)
    return df_fe


def create_sequences_from_df(
    df_input,
    feature_columns: list[str],
    label_col: str,
    seq_length: int = SEQ_LENGTH,
    id_col: str = "FIPS",
    start_date: str | None = None,
    end_date: str | None = None,
) -> tuple:
    import numpy as np
    import pandas as pd

    x_values, y_values = [], []
    for _, group in df_input.groupby(id_col):
        group = group.sort_values("week_start")
        feats = group[feature_columns].values
        labels = group[label_col].values
        dates = group["week_start"].values
        if len(group) < seq_length:
            continue
        for i in range(seq_length - 1, len(group)):
            target_date = pd.Timestamp(dates[i])
            if start_date is not None and target_date < pd.Timestamp(start_date):
                continue
            if end_date is not None and target_date > pd.Timestamp(end_date):
                continue
            x_values.append(feats[i - seq_length + 1:i + 1])
            y_values.append(labels[i])
    return np.array(x_values), np.array(y_values)


def tune_class_multipliers(
    y_true,
    y_prob,
    n_classes: int = 6,
    n_iter: int = 2500,
    seed: int = SEED,
) -> tuple:
    import numpy as np
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(seed)
    best_m = np.ones(n_classes, dtype=np.float32)
    base_pred = np.argmax(y_prob, axis=1)
    best_score = f1_score(y_true, base_pred, average="macro", zero_division=0)
    for _ in range(n_iter):
        cand = np.exp(rng.normal(loc=0.0, scale=0.30, size=n_classes)).astype(np.float32)
        cand = np.clip(cand, 0.55, 1.8)
        pred = np.argmax(y_prob * cand, axis=1)
        score = f1_score(y_true, pred, average="macro", zero_division=0)
        if score > best_score:
            best_score = score
            best_m = cand
    return best_m, float(best_score)


def extract_existing_summary_metadata(summary_path: Path) -> dict[str, str]:
    metadata: dict[str, str] = {}
    if not summary_path.exists():
        return metadata
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if ":" not in line:
            continue
        if line.startswith("Best trial:"):
            metadata["best_trial"] = line.split(":", 1)[1].strip()
        elif line.startswith("Best trial config:"):
            metadata["best_trial_config"] = line.split(":", 1)[1].strip()
        elif line.startswith("Trial leaderboard:"):
            break
    return metadata


def write_confusion_matrix(output_dir: Path, y_true, y_pred) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(LABEL_MAP))))
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[LABEL_MAP[i] for i in range(len(LABEL_MAP))],
        yticklabels=[LABEL_MAP[i] for i in range(len(LABEL_MAP))],
        ax=axes[0],
    )
    axes[0].set_title("Confusion Matrix (Counts)")
    axes[0].set_xlabel("Predicted")
    axes[0].set_ylabel("Actual")

    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        xticklabels=[LABEL_MAP[i] for i in range(len(LABEL_MAP))],
        yticklabels=[LABEL_MAP[i] for i in range(len(LABEL_MAP))],
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=140)
    plt.close(fig)


def write_per_class_f1(output_dir: Path, per_class_f1, macro_f1: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([LABEL_MAP[i] for i in range(len(LABEL_MAP))], per_class_f1)
    for bar, value in zip(bars, per_class_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{value:.3f}", ha="center")
    ax.axhline(macro_f1, color="red", linestyle="--", label=f"Macro F1: {macro_f1:.4f}")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "per_class_f1.png", dpi=140)
    plt.close(fig)


def write_summary(
    output_dir: Path,
    scenario_key: str,
    feature_cols: list[str],
    metadata: dict[str, str],
    class_multipliers,
    raw_val_macro_f1: float,
    tuned_val_macro_f1: float,
    accuracy_raw: float,
    macro_f1_raw: float,
    weighted_f1_raw: float,
    accuracy_tuned: float,
    macro_f1_tuned: float,
    weighted_f1_tuned: float,
    report_raw: str,
    report_tuned: str,
    per_class_f1_raw,
    per_class_f1_tuned,
) -> None:
    summary_path = output_dir / "results_summary.txt"
    with summary_path.open("w", encoding="utf-8") as handle:
        handle.write(f"BiLSTM Weekly Kansas - {SCENARIO_NAMES[scenario_key]}\n")
        if "best_trial" in metadata:
            handle.write(f"Best trial: {metadata['best_trial']}\n")
        if "best_trial_config" in metadata:
            handle.write(f"Best trial config: {metadata['best_trial_config']}\n")
        handle.write(f"Best model path: {output_dir / 'best_model.keras'}\n")
        handle.write(f"Seq Length: {SEQ_LENGTH}\n")
        handle.write(f"Feature count: {len(feature_cols)}\n")
        handle.write(f"Val Macro F1 (raw/tuned): {raw_val_macro_f1:.4f} / {tuned_val_macro_f1:.4f}\n")
        handle.write(f"Class multipliers: {class_multipliers.tolist()}\n")
        handle.write("Selected features:\n")
        for feature in feature_cols:
            handle.write(f"  {feature}\n")
        handle.write("\n--- RESULTS ---\n")
        handle.write(f"Accuracy (raw): {accuracy_raw:.4f}\n")
        handle.write(f"Macro F1 (raw): {macro_f1_raw:.4f}\n")
        handle.write(f"Weighted F1 (raw): {weighted_f1_raw:.4f}\n")
        handle.write(f"Accuracy (tuned): {accuracy_tuned:.4f}\n")
        handle.write(f"Macro F1 (tuned): {macro_f1_tuned:.4f}\n")
        handle.write(f"Weighted F1 (tuned): {weighted_f1_tuned:.4f}\n\n")
        handle.write("Per-class F1 (raw):\n")
        for idx in range(len(LABEL_MAP)):
            handle.write(f"  {LABEL_MAP[idx]}: {per_class_f1_raw[idx]:.4f}\n")
        handle.write("\nPer-class F1 (tuned):\n")
        for idx in range(len(LABEL_MAP)):
            handle.write(f"  {LABEL_MAP[idx]}: {per_class_f1_tuned[idx]:.4f}\n")
        handle.write("\n======================================================================\n")
        handle.write("CLASSIFICATION REPORT (RAW)\n")
        handle.write("======================================================================\n")
        handle.write(report_raw)
        handle.write("\n======================================================================\n")
        handle.write("CLASSIFICATION REPORT (TUNED)\n")
        handle.write("======================================================================\n")
        handle.write(report_tuned)
        handle.write("\n======================================================================\n")


def load_model_input_width(model) -> int:
    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    return int(input_shape[-1])


def refresh_scenario(scenario_key: str, kansas_dir: Path) -> None:
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.preprocessing import MinMaxScaler

    root_dir = kansas_dir.parent
    data_path = root_dir / "Integrated_weekly_KAN_20counties.csv"
    output_dir = kansas_dir / scenario_output_dir_name(scenario_key)
    model_path = output_dir / "best_model.keras"

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Best model not found for scenario {scenario_key}: {model_path}")

    feature_cols = get_feature_columns_for_scenario(scenario_key, output_dir)
    df_fe = build_base_feature_frame(data_path)
    df_fe = df_fe.dropna(subset=feature_cols + ["Label"]).reset_index(drop=True)

    train_df = df_fe[df_fe["week_start"] <= TRAIN_END_DATE].copy()
    scaler = MinMaxScaler()
    scaler.fit(train_df[feature_cols])
    df_fe.loc[:, feature_cols] = scaler.transform(df_fe[feature_cols])

    x_val, y_val = create_sequences_from_df(
        df_fe, feature_cols, "Label", SEQ_LENGTH, start_date=VAL_START_DATE, end_date=VAL_END_DATE
    )
    x_test, y_test = create_sequences_from_df(
        df_fe, feature_cols, "Label", SEQ_LENGTH, start_date=TEST_START_DATE, end_date=None
    )

    model = tf.keras.models.load_model(model_path, compile=False)
    model_feature_width = load_model_input_width(model)
    if model_feature_width != len(feature_cols):
        raise ValueError(
            f"Scenario {scenario_key} feature mismatch: model expects {model_feature_width}, "
            f"but recreated features contain {len(feature_cols)}"
        )

    val_prob = model.predict(x_val, verbose=0)
    class_multipliers, tuned_val_macro_f1 = tune_class_multipliers(y_val, val_prob)
    raw_val_macro_f1 = f1_score(y_val, np.argmax(val_prob, axis=1), average="macro", zero_division=0)

    test_prob = model.predict(x_test, verbose=0)
    y_pred_raw = np.argmax(test_prob, axis=1)
    y_pred_tuned = np.argmax(test_prob * class_multipliers, axis=1)
    present_classes = sorted(set(y_test) | set(y_pred_tuned))
    target_names = [LABEL_MAP[idx] for idx in present_classes]

    report_raw = classification_report(
        y_test, y_pred_raw, labels=present_classes, target_names=target_names, digits=4, zero_division=0
    )
    report_tuned = classification_report(
        y_test, y_pred_tuned, labels=present_classes, target_names=target_names, digits=4, zero_division=0
    )

    accuracy_raw = accuracy_score(y_test, y_pred_raw)
    macro_f1_raw = f1_score(y_test, y_pred_raw, average="macro", zero_division=0)
    weighted_f1_raw = f1_score(y_test, y_pred_raw, average="weighted", zero_division=0)
    accuracy_tuned = accuracy_score(y_test, y_pred_tuned)
    macro_f1_tuned = f1_score(y_test, y_pred_tuned, average="macro", zero_division=0)
    weighted_f1_tuned = f1_score(y_test, y_pred_tuned, average="weighted", zero_division=0)
    per_class_f1_raw = f1_score(y_test, y_pred_raw, labels=list(range(len(LABEL_MAP))), average=None, zero_division=0)
    per_class_f1_tuned = f1_score(
        y_test, y_pred_tuned, labels=list(range(len(LABEL_MAP))), average=None, zero_division=0
    )

    write_confusion_matrix(output_dir, y_test, y_pred_tuned)
    write_per_class_f1(output_dir, per_class_f1_tuned, macro_f1_tuned)
    write_summary(
        output_dir=output_dir,
        scenario_key=scenario_key,
        feature_cols=feature_cols,
        metadata=extract_existing_summary_metadata(output_dir / "results_summary.txt"),
        class_multipliers=class_multipliers,
        raw_val_macro_f1=raw_val_macro_f1,
        tuned_val_macro_f1=tuned_val_macro_f1,
        accuracy_raw=accuracy_raw,
        macro_f1_raw=macro_f1_raw,
        weighted_f1_raw=weighted_f1_raw,
        accuracy_tuned=accuracy_tuned,
        macro_f1_tuned=macro_f1_tuned,
        weighted_f1_tuned=weighted_f1_tuned,
        report_raw=report_raw,
        report_tuned=report_tuned,
        per_class_f1_raw=per_class_f1_raw,
        per_class_f1_tuned=per_class_f1_tuned,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Refresh Kansas scenario outputs from existing best_model.keras files without retraining."
    )
    parser.add_argument("--scenario", required=True, help="Scenario key such as 1, 2A, 3, 7, or 'all'.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    requested = normalize_scenario_key(args.scenario)
    scenarios = SCENARIO_ORDER if requested == "ALL" else [requested]
    kansas_dir = Path(__file__).resolve().parent
    for scenario_key in scenarios:
        print(f"Refreshing scenario {scenario_key}...")
        refresh_scenario(scenario_key, kansas_dir)
        print(f"Scenario {scenario_key} complete.")


if __name__ == "__main__":
    main()
