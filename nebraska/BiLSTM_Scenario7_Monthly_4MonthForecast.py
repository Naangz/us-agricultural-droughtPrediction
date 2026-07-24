# Scenario 7: Monthly Drought History Only
# Forecast target: drought class 4 months after the input history window.

import json
import os
import pickle
import random
from collections import Counter

import numpy as np
import pandas as pd


SEED = 42
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NEBRASKA_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, "Integrated_weekly_NEB_20counties.csv")
OUTPUT_FOLDER = os.path.join(NEBRASKA_DIR, "output_monthly_nebraska_scenario7_avg_horizon4")

SCENARIO_NAME = "Scenario 7 - Monthly Drought History Only (Lagged, 4-Month Forecast)"
TIME_COL = "month_start"
SEQ_LENGTH = 12
FORECAST_HORIZON = 4
BATCH_SIZE = 64
EPOCHS = 120

TRAIN_END_DATE = "2020-12-31"
VAL_START_DATE = "2021-01-01"
VAL_END_DATE = "2022-12-31"
TEST_START_DATE = "2023-01-01"

USE_TARGETED_CLASS_BOOST = False
CLASS_WEIGHT_BOOST = {0: 1.2, 1: 1.2, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}

DROUGHT_COLS = ["None", "D0", "D1", "D2", "D3", "D4"]
PMF_COLS = ["PMF_None", "PMF_D0", "PMF_D1", "PMF_D2", "PMF_D3", "PMF_D4"]
LABEL_MAP = {0: "None", 1: "D0", 2: "D1", 3: "D2", 4: "D3", 5: "D4"}


def decumulate_drought(row):
    pmf_d4 = row["D4"]
    pmf_d3 = max(0.0, row["D3"] - row["D4"])
    pmf_d2 = max(0.0, row["D2"] - row["D3"])
    pmf_d1 = max(0.0, row["D1"] - row["D2"])
    pmf_d0 = max(0.0, row["D0"] - row["D1"])
    pmf_none = max(0.0, row["None"])
    return pd.Series([pmf_none, pmf_d0, pmf_d1, pmf_d2, pmf_d3, pmf_d4])


def add_drought_labels(df):
    df = df.copy()
    df[PMF_COLS] = df.apply(decumulate_drought, axis=1)
    df["PMF_Sum"] = df[PMF_COLS].sum(axis=1)
    df["Label"] = df[PMF_COLS].idxmax(axis=1).apply(lambda x: PMF_COLS.index(x))
    return df


def aggregate_weekly_to_monthly(df, date_col="week_start", id_col="FIPS"):
    """Average weekly drought percentages into county-month rows."""
    monthly = df.copy()
    monthly[date_col] = pd.to_datetime(monthly[date_col])
    if "ValidEnd" in monthly.columns:
        monthly["ValidEnd"] = pd.to_datetime(monthly["ValidEnd"])

    monthly["month_start"] = monthly[date_col].dt.to_period("M").dt.to_timestamp()
    monthly = monthly.sort_values([id_col, date_col]).reset_index(drop=True)
    agg_spec = {col: "mean" for col in DROUGHT_COLS}
    agg_spec[date_col] = "max"
    if "ValidEnd" in monthly.columns:
        agg_spec["ValidEnd"] = "max"
    monthly = monthly.groupby([id_col, "month_start"], as_index=False).agg(agg_spec)
    return monthly.sort_values([id_col, "month_start"]).reset_index(drop=True)


def prepare_monthly_drought_history_features(df, id_col="FIPS", time_col=TIME_COL):
    df_fe = df.copy().sort_values([id_col, time_col]).reset_index(drop=True)

    for col in DROUGHT_COLS:
        df_fe[f"{col}_lag1"] = df_fe.groupby(id_col)[col].shift(1)
        df_fe[f"{col}_lag2"] = df_fe.groupby(id_col)[col].shift(2)

    feature_cols = [
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

    df_fe = df_fe.dropna(subset=feature_cols + ["Label"]).reset_index(drop=True)
    return df_fe, feature_cols


def create_sequences_from_df(
    df_input,
    feature_columns,
    label_col,
    seq_length=SEQ_LENGTH,
    forecast_horizon=FORECAST_HORIZON,
    id_col="FIPS",
    time_col=TIME_COL,
    start_date=None,
    end_date=None,
    return_dates=False,
):
    X, y = [], []
    input_end_dates, target_dates = [], []

    for _, group in df_input.groupby(id_col):
        group = group.sort_values(time_col)
        feats = group[feature_columns].values
        labels = group[label_col].values
        dates = group[time_col].values

        if len(group) < seq_length + forecast_horizon:
            continue

        for input_end_i in range(seq_length - 1, len(group) - forecast_horizon):
            target_i = input_end_i + forecast_horizon
            target_date = pd.Timestamp(dates[target_i])

            if start_date is not None and target_date < pd.Timestamp(start_date):
                continue
            if end_date is not None and target_date > pd.Timestamp(end_date):
                continue

            X.append(feats[input_end_i - seq_length + 1:input_end_i + 1])
            y.append(labels[target_i])
            input_end_dates.append(pd.Timestamp(dates[input_end_i]))
            target_dates.append(target_date)

    X = np.array(X)
    y = np.array(y)
    if return_dates:
        return X, y, input_end_dates, target_dates
    return X, y


def get_balancer(name, seed):
    name = name.upper()
    if name == "NONE":
        return None

    from imblearn.over_sampling import ADASYN, SMOTE, BorderlineSMOTE, RandomOverSampler
    from imblearn.combine import SMOTEENN, SMOTETomek

    if name == "ROS":
        return RandomOverSampler(random_state=seed)
    if name == "SMOTE":
        return SMOTE(random_state=seed)
    if name == "BSMOTE":
        return BorderlineSMOTE(random_state=seed)
    if name == "ADASYN":
        return ADASYN(random_state=seed)
    if name == "SMOTETOMEK":
        return SMOTETomek(random_state=seed)
    if name == "SMOTEENN":
        return SMOTEENN(random_state=seed)
    raise ValueError(f"Unknown balancer: {name}")


def compute_focal_alpha(y_labels, n_classes, mode="inverse_train_seq", manual_alpha=None):
    if mode == "none":
        return None
    if mode == "manual":
        if manual_alpha is None or len(manual_alpha) != n_classes:
            raise ValueError("manual_alpha must be provided with length equal to n_classes.")
        alpha = np.array(manual_alpha, dtype=np.float32)
    elif mode == "inverse_train_seq":
        counts = np.bincount(y_labels.astype(int), minlength=n_classes).astype(np.float32)
        inv = 1.0 / np.maximum(counts, 1.0)
        alpha = inv / inv.sum()
    else:
        raise ValueError(f"Unknown FOCAL_ALPHA_MODE: {mode}")
    return alpha.tolist()


def categorical_focal_loss(gamma=2.0, alpha=None):
    import tensorflow as tf
    from tensorflow.keras import backend as K

    def focal_loss(y_true, y_pred):
        eps = K.epsilon()
        y_pred_clipped = K.clip(y_pred, eps, 1.0 - eps)
        ce = -y_true * K.log(y_pred_clipped)
        fw = K.pow(1.0 - y_pred_clipped, gamma)
        loss = fw * ce
        if alpha is not None:
            loss = tf.constant(alpha, dtype=tf.float32) * loss
        return K.sum(loss, axis=-1)

    return focal_loss


def build_model(
    seq_length,
    n_features,
    n_classes,
    lstm_units=(64, 32),
    dropout=0.3,
    dense_units=64,
    lr=1e-3,
    focal_gamma=2.0,
    focal_alpha=None,
):
    import tensorflow as tf
    from tensorflow.keras.layers import LSTM, BatchNormalization, Bidirectional, Dense, Dropout, Input
    from tensorflow.keras.models import Sequential

    u1, u2 = lstm_units
    model = Sequential(
        [
            Input(shape=(seq_length, n_features)),
            Bidirectional(LSTM(u1, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)),
            BatchNormalization(),
            Dropout(dropout),
            Bidirectional(LSTM(u2, dropout=0.2, recurrent_dropout=0.1)),
            BatchNormalization(),
            Dropout(dropout),
            Dense(dense_units, activation="relu"),
            Dropout(min(0.4, dropout)),
            Dense(n_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=categorical_focal_loss(gamma=focal_gamma, alpha=focal_alpha),
        metrics=["accuracy"],
    )
    return model


def prepare_training_data(x_train, y_train, balancer_name, seed):
    balancer = get_balancer(balancer_name, seed)
    if balancer is None:
        return x_train, y_train

    n_samples_local, n_steps_local, n_features_local = x_train.shape
    x_train_flat = x_train.reshape(n_samples_local, n_steps_local * n_features_local)
    x_bal_flat, y_bal = balancer.fit_resample(x_train_flat, y_train)
    x_bal = x_bal_flat.reshape(-1, n_steps_local, n_features_local)
    return x_bal, y_bal


def compute_class_weights(y_labels, num_classes):
    from sklearn.utils.class_weight import compute_class_weight

    y_labels = y_labels.astype(int)
    weights = np.ones(num_classes, dtype=np.float32)
    present_classes = np.unique(y_labels)
    present_weights = compute_class_weight(class_weight="balanced", classes=present_classes, y=y_labels)
    weights[present_classes] = present_weights.astype(np.float32)
    weights = weights / weights.mean()

    cw_dict = dict(enumerate(weights.tolist()))
    if USE_TARGETED_CLASS_BOOST:
        for cls_idx, boost in CLASS_WEIGHT_BOOST.items():
            cw_dict[cls_idx] = cw_dict.get(cls_idx, 1.0) * float(boost)
        mean_cw = np.mean(list(cw_dict.values()))
        cw_dict = {k: v / mean_cw for k, v in cw_dict.items()}
    return cw_dict


def tune_class_multipliers(y_true, y_prob, n_classes=6, n_iter=2500, seed=SEED):
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


def print_class_distribution(title, labels):
    dist = Counter(labels)
    print(f"\n{title} class distribution:")
    for idx in range(6):
        print(f"  {LABEL_MAP[idx]:>4s}: {dist.get(idx, 0)}")


def main():
    import matplotlib.pyplot as plt
    import seaborn as sns
    import tensorflow as tf
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
    from sklearn.preprocessing import LabelEncoder, MinMaxScaler
    from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    os.environ["PYTHONHASHSEED"] = str(SEED)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"

    print(f"TensorFlow version: {tf.__version__}")
    print("Config ready")
    print(f"Data path: {DATA_PATH}")
    print(f"Scenario: {SCENARIO_NAME}")
    print(f"Temporal resolution: monthly")
    print(f"Sequence length: {SEQ_LENGTH} months")
    print(f"Forecast horizon: {FORECAST_HORIZON} months")
    print(f"Output folder: {OUTPUT_FOLDER}")

    df_weekly = pd.read_csv(DATA_PATH)
    df_monthly = aggregate_weekly_to_monthly(df_weekly)
    df_monthly = add_drought_labels(df_monthly)
    df_monthly = df_monthly.sort_values(["FIPS", TIME_COL]).reset_index(drop=True)

    print(f"Weekly shape: {df_weekly.shape}")
    df_fe, feature_cols = prepare_monthly_drought_history_features(df_monthly)

    print("Monthly aggregation: average weekly drought percentages per county-month")
    print(f"Monthly shape: {df_monthly.shape}")
    print(f"Feature rows after lag dropna: {len(df_fe):,}")
    print(f"Unique counties: {df_fe['FIPS'].nunique()}")
    print(f"Monthly range: {df_fe[TIME_COL].min().date()} to {df_fe[TIME_COL].max().date()}")

    print("=" * 60)
    print("SCENARIO:", SCENARIO_NAME)
    print(f"Features used: {feature_cols}")
    print("=" * 60)

    train_feature_end = pd.Timestamp(TRAIN_END_DATE) - pd.DateOffset(months=FORECAST_HORIZON)
    scaler_fit_df = df_fe[df_fe[TIME_COL] <= train_feature_end]
    if scaler_fit_df.empty:
        raise ValueError("No rows available for train-only scaler fit.")

    scaler = MinMaxScaler()
    scaler.fit(scaler_fit_df[feature_cols])
    df_fe.loc[:, feature_cols] = scaler.transform(df_fe[feature_cols])
    print(f"Scaling done using features through {train_feature_end.date()} only.")

    X_train_seq, y_train_seq, train_input_end, train_target_dates = create_sequences_from_df(
        df_fe,
        feature_cols,
        "Label",
        SEQ_LENGTH,
        FORECAST_HORIZON,
        start_date=None,
        end_date=TRAIN_END_DATE,
        return_dates=True,
    )
    X_val_seq, y_val_seq, val_input_end, val_target_dates = create_sequences_from_df(
        df_fe,
        feature_cols,
        "Label",
        SEQ_LENGTH,
        FORECAST_HORIZON,
        start_date=VAL_START_DATE,
        end_date=VAL_END_DATE,
        return_dates=True,
    )
    X_test_seq, y_test, test_input_end, test_target_dates = create_sequences_from_df(
        df_fe,
        feature_cols,
        "Label",
        SEQ_LENGTH,
        FORECAST_HORIZON,
        start_date=TEST_START_DATE,
        end_date=None,
        return_dates=True,
    )

    if len(X_train_seq) == 0 or len(X_val_seq) == 0 or len(X_test_seq) == 0:
        raise ValueError("One or more splits produced zero sequences. Check dates, sequence length, and horizon.")

    print(f"X_train_seq: {X_train_seq.shape}")
    print(f"X_val_seq:   {X_val_seq.shape}")
    print(f"X_test_seq:  {X_test_seq.shape}")
    print(f"Train input end range: {min(train_input_end).date()} to {max(train_input_end).date()}")
    print(f"Train target range:    {min(train_target_dates).date()} to {max(train_target_dates).date()}")
    print(f"Val target range:      {min(val_target_dates).date()} to {max(val_target_dates).date()}")
    print(f"Test target range:     {min(test_target_dates).date()} to {max(test_target_dates).date()}")
    print_class_distribution("Train target", y_train_seq)
    print_class_distribution("Val target", y_val_seq)
    print_class_distribution("Test target", y_test)

    n_steps, n_features = X_train_seq.shape[1], X_train_seq.shape[2]
    num_classes = 6
    y_val_enc = to_categorical(y_val_seq, num_classes=num_classes)
    y_test_enc = to_categorical(y_test, num_classes=num_classes)

    trial_configs = [
        {
            "name": "ros_focal_no_cw_96x48",
            "balancer": "ROS",
            "use_class_weight": True,
            "focal_gamma": 1.5,
            "focal_alpha_mode": "none",
            "manual_focal_alpha": None,
            "lstm_units": (96, 48),
            "dropout": 0.30,
            "dense_units": 64,
            "lr": 8e-4,
            "patience": 14,
        },
        {
            "name": "none_focal_inv_cw_64x32",
            "balancer": "NONE",
            "use_class_weight": True,
            "focal_gamma": 2.0,
            "focal_alpha_mode": "inverse_train_seq",
            "manual_focal_alpha": None,
            "lstm_units": (64, 32),
            "dropout": 0.35,
            "dense_units": 64,
            "lr": 1e-3,
            "patience": 16,
        },
        {
            "name": "ros_focal_inv_no_cw_128x64",
            "balancer": "ROS",
            "use_class_weight": True,
            "focal_gamma": 2.0,
            "focal_alpha_mode": "inverse_train_seq",
            "manual_focal_alpha": None,
            "lstm_units": (128, 64),
            "dropout": 0.28,
            "dense_units": 96,
            "lr": 6e-4,
            "patience": 14,
        },
    ]

    class MacroF1Callback(Callback):
        def __init__(self, x_val, y_val):
            super().__init__()
            self.x_val = x_val
            self.y_val = y_val

        def on_epoch_end(self, epoch, logs=None):
            if logs is None:
                logs = {}
            val_pred = self.model.predict(self.x_val, verbose=0)
            val_pred_label = np.argmax(val_pred, axis=1)
            val_true_label = np.argmax(self.y_val, axis=1)
            score = f1_score(val_true_label, val_pred_label, average="macro", zero_division=0)
            logs["val_macro_f1"] = score
            print(f" - val_macro_f1: {score:.4f}", end="")

    trial_results = []
    best_trial = None
    best_val_macro_f1 = -1.0
    best_model = None
    best_history = None

    print("\nStarting hyperparameter trials...")
    for i, cfg in enumerate(trial_configs, start=1):
        print("\n" + "=" * 80)
        print(f"Trial {i}/{len(trial_configs)}: {cfg['name']}")
        print(cfg)

        X_train_bal, y_train_bal = prepare_training_data(X_train_seq, y_train_seq, cfg["balancer"], SEED + i)
        y_train_enc = to_categorical(y_train_bal, num_classes=num_classes)

        focal_alpha = compute_focal_alpha(
            y_train_bal,
            n_classes=num_classes,
            mode=cfg["focal_alpha_mode"],
            manual_alpha=cfg["manual_focal_alpha"],
        )
        print("Focal alpha:", focal_alpha)
        print(f"Before balancing: {Counter(y_train_seq)}")
        print(f"After balancing:  {Counter(y_train_bal)}")

        model = build_model(
            SEQ_LENGTH,
            n_features,
            num_classes,
            lstm_units=cfg["lstm_units"],
            dropout=cfg["dropout"],
            dense_units=cfg["dense_units"],
            lr=cfg["lr"],
            focal_gamma=cfg["focal_gamma"],
            focal_alpha=focal_alpha,
        )

        trial_ckpt = os.path.join(OUTPUT_FOLDER, f"best_model_{cfg['name']}.keras")
        callbacks = [
            MacroF1Callback(X_val_seq, y_val_enc),
            EarlyStopping(
                monitor="val_macro_f1",
                mode="max",
                patience=cfg["patience"],
                restore_best_weights=True,
                verbose=1,
            ),
            ModelCheckpoint(trial_ckpt, monitor="val_macro_f1", mode="max", save_best_only=True, verbose=0),
            ReduceLROnPlateau(
                monitor="val_macro_f1",
                mode="max",
                factor=0.5,
                patience=max(4, cfg["patience"] // 3),
                min_lr=1e-6,
                verbose=1,
            ),
        ]

        fit_kwargs = {}
        if cfg["use_class_weight"]:
            class_weight_dict = compute_class_weights(y_train_bal, num_classes)
            print("Class weights:", class_weight_dict)
            fit_kwargs["class_weight"] = class_weight_dict
        else:
            print("Class weights: not used")

        history = model.fit(
            X_train_bal,
            y_train_enc,
            validation_data=(X_val_seq, y_val_enc),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
            **fit_kwargs,
        )

        val_pred_prob = model.predict(X_val_seq, verbose=0)
        val_pred = np.argmax(val_pred_prob, axis=1)
        val_macro_f1 = f1_score(y_val_seq, val_pred, average="macro", zero_division=0)
        trial_results.append({"name": cfg["name"], "val_macro_f1": float(val_macro_f1), "balancer": cfg["balancer"]})
        print(f"Trial {cfg['name']} val macro-F1: {val_macro_f1:.4f}")

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_trial = cfg
            best_model = model
            best_history = history

    print("\nTrial leaderboard (by validation macro-F1):")
    for row in sorted(trial_results, key=lambda x: x["val_macro_f1"], reverse=True):
        print(f"  {row['name']}: {row['val_macro_f1']:.4f} (balancer={row['balancer']})")

    print(f"\nBest trial: {best_trial['name']} with val macro-F1 {best_val_macro_f1:.4f}")
    best_model.save(os.path.join(OUTPUT_FOLDER, "best_model.keras"))

    val_best_prob = best_model.predict(X_val_seq, verbose=0)
    class_multipliers, tuned_val_macro_f1 = tune_class_multipliers(
        y_val_seq,
        val_best_prob,
        n_classes=num_classes,
        n_iter=2500,
        seed=SEED,
    )
    raw_val_macro_f1 = f1_score(y_val_seq, np.argmax(val_best_prob, axis=1), average="macro", zero_division=0)
    print("\nClass multiplier calibration (from validation only):")
    print(f"  raw val macro-F1:   {raw_val_macro_f1:.4f}")
    print(f"  tuned val macro-F1: {tuned_val_macro_f1:.4f}")
    print(f"  multipliers: {class_multipliers}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(best_history.history["loss"], label="Train")
    axes[0].plot(best_history.history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(best_history.history["accuracy"], label="Train")
    axes[1].plot(best_history.history["val_accuracy"], label="Val")
    axes[1].set_title("Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "training_history.png"), dpi=140)
    plt.close(fig)

    y_pred_prob = best_model.predict(X_test_seq, verbose=0)
    y_pred_raw = np.argmax(y_pred_prob, axis=1)
    y_pred = np.argmax(y_pred_prob * class_multipliers, axis=1)

    present_classes = sorted(set(y_test) | set(y_pred))
    target_names = [LABEL_MAP[i] for i in present_classes]
    present_classes_raw = sorted(set(y_test) | set(y_pred_raw))
    target_names_raw = [LABEL_MAP[i] for i in present_classes_raw]

    report_raw = classification_report(
        y_test,
        y_pred_raw,
        labels=present_classes_raw,
        target_names=target_names_raw,
        digits=4,
        zero_division=0,
    )
    report = classification_report(
        y_test,
        y_pred,
        labels=present_classes,
        target_names=target_names,
        digits=4,
        zero_division=0,
    )
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    macro_f1_raw = f1_score(y_test, y_pred_raw, average="macro", zero_division=0)
    accuracy_raw = accuracy_score(y_test, y_pred_raw)

    print("=" * 70)
    print("CLASSIFICATION REPORT")
    print("=" * 70)
    print(report)
    print("=" * 70)
    print(f"Accuracy (raw):    {accuracy_raw:.4f}")
    print(f"Macro F1 (raw):    {macro_f1_raw:.4f}")
    print(f"Accuracy:          {accuracy:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")
    print(f"Weighted F1:       {weighted_f1:.4f}")
    print("=" * 70)

    cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[LABEL_MAP[i] for i in range(num_classes)],
        yticklabels=[LABEL_MAP[i] for i in range(num_classes)],
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
        xticklabels=[LABEL_MAP[i] for i in range(num_classes)],
        yticklabels=[LABEL_MAP[i] for i in range(num_classes)],
        ax=axes[1],
    )
    axes[1].set_title("Confusion Matrix (Normalized)")
    axes[1].set_xlabel("Predicted")
    axes[1].set_ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "confusion_matrix.png"), dpi=140)
    plt.close(fig)

    per_class_f1_raw = f1_score(y_test, y_pred_raw, labels=list(range(num_classes)), average=None, zero_division=0)
    per_class_f1 = f1_score(y_test, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)

    per_class_acc = []
    per_class_acc_raw = []
    for i in range(num_classes):
        total = np.sum(y_test == i)
        acc = np.sum((y_test == i) & (y_pred == i)) / total if total > 0 else 0.0
        acc_raw = np.sum((y_test == i) & (y_pred_raw == i)) / total if total > 0 else 0.0
        per_class_acc.append(acc)
        per_class_acc_raw.append(acc_raw)

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([LABEL_MAP[i] for i in range(num_classes)], per_class_f1)
    for bar, val in zip(bars, per_class_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f"{val:.3f}", ha="center")
    ax.axhline(macro_f1, color="red", linestyle="--", label=f"Macro F1: {macro_f1:.4f}")
    ax.set_ylim(0, 1.05)
    ax.set_title("Per-Class F1")
    ax.grid(alpha=0.3, axis="y")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "per_class_f1.png"), dpi=140)
    plt.close(fig)

    summary_path = os.path.join(OUTPUT_FOLDER, "results_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("BiLSTM Monthly Nebraska Scenario 7 - 4-Month Forecast\n")
        f.write(f"Best trial: {best_trial['name']}\n")
        f.write(f"Best trial config: {best_trial}\n")
        f.write(f"Seq Length: {SEQ_LENGTH} months\n")
        f.write(f"Forecast Horizon: {FORECAST_HORIZON} months\n")
        f.write(f"Val Macro F1 (best trial): {best_val_macro_f1:.4f}\n")
        f.write(f"Val Macro F1 (raw/tuned): {raw_val_macro_f1:.4f} / {tuned_val_macro_f1:.4f}\n")
        f.write(f"Class multipliers: {class_multipliers.tolist()}\n")
        f.write("Selected features:\n")
        for feat in feature_cols:
            f.write(f"  {feat}\n")
        f.write(f"Accuracy (raw): {accuracy_raw:.4f}\n")
        f.write(f"Macro F1 (raw): {macro_f1_raw:.4f}\n")
        f.write(f"Accuracy: {accuracy:.4f}\n")
        f.write(f"Macro F1: {macro_f1:.4f}\n")
        f.write(f"Weighted F1: {weighted_f1:.4f}\n\n")
        f.write("Trial leaderboard:\n")
        for row in sorted(trial_results, key=lambda x: x["val_macro_f1"], reverse=True):
            f.write(f"  {row['name']}: {row['val_macro_f1']:.4f} (balancer={row['balancer']})\n")
        f.write("\nPer-class F1 (raw):\n")
        for i in range(num_classes):
            f.write(f"  {LABEL_MAP[i]}: {per_class_f1_raw[i]:.4f}\n")
        f.write("\nPer-class F1:\n")
        for i in range(num_classes):
            f.write(f"  {LABEL_MAP[i]}: {per_class_f1[i]:.4f}\n")
        f.write("\nPer-class Accuracy:\n")
        for i in range(num_classes):
            f.write(f"  {LABEL_MAP[i]}: {per_class_acc[i]:.4f}\n")
        f.write("\nPer-class Accuracy (raw):\n")
        for i in range(num_classes):
            f.write(f"  {LABEL_MAP[i]}: {per_class_acc_raw[i]:.4f}\n")
        f.write("\nClassification Report (raw):\n")
        f.write(report_raw)
        f.write("\nClassification Report:\n")
        f.write(report)

    print(f"Results saved to {summary_path}")

    artifacts = {
        "tokenizer.pkl": scaler,
        "scaler.pkl": scaler,
    }
    for filename, artifact in artifacts.items():
        artifact_path = os.path.join(OUTPUT_FOLDER, filename)
        with open(artifact_path, "wb") as f:
            pickle.dump(artifact, f)
        print(f"Saved: {artifact_path}")

    le = LabelEncoder()
    le.classes_ = np.array(["None", "D0", "D1", "D2", "D3", "D4"], dtype=object)
    le_path = os.path.join(OUTPUT_FOLDER, "label_encoder.pkl")
    with open(le_path, "wb") as f:
        pickle.dump(le, f)
    print(f"Saved: {le_path}")

    config_data = {
        "temporal_resolution": "monthly",
        "forecast_horizon_months": int(FORECAST_HORIZON),
        "max_sequence_length": int(SEQ_LENGTH),
        "feature_cols": feature_cols,
        "label_map": {str(k): v for k, v in LABEL_MAP.items()},
        "class_multipliers": class_multipliers.tolist(),
    }
    config_path = os.path.join(OUTPUT_FOLDER, "config.json")
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=4)
    print(f"Saved: {config_path}")

    for filename in ["max_sequence_length.txt", "max_sequence_length"]:
        path = os.path.join(OUTPUT_FOLDER, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(str(SEQ_LENGTH))
        print(f"Saved: {path}")

    requirements_content = """tensorflow>=2.0
gradio>=4.0
scikit-learn
pandas
numpy
shap
matplotlib
"""
    requirements_path = os.path.join(OUTPUT_FOLDER, "requirements.txt")
    with open(requirements_path, "w", encoding="utf-8") as f:
        f.write(requirements_content)
    print(f"Saved: {requirements_path}")

    readme_content = """---
title: Nebraska Monthly Drought Prediction Scenario 7
emoji: chart_with_upwards_trend
colorFrom: green
colorTo: lime
sdk: gradio
sdk_version: 4.36.1
app_file: app.py
pinned: false
license: mit
---

# Monthly Drought Prediction Dashboard (Nebraska)

Model Scenario 7: Drought History Only (BiLSTM)

This Space runs a Bidirectional LSTM model trained to predict drought categories
(None, D0-D4) for 20 Nebraska counties using monthly drought history inputs and a
4-month forecast horizon.
"""
    readme_path = os.path.join(OUTPUT_FOLDER, "README.md")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(readme_content)
    print(f"Saved: {readme_path}")
    print("All monthly 4-month forecast artifacts saved successfully.")

    _ = y_test_enc


if __name__ == "__main__":
    main()
