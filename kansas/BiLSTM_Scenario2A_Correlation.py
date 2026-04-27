# Auto-generated from BiLSTM_Weekly_Kansas_Clean.ipynb
# Scenario 2A: Correlation-aware statistical feature selection (Top-20)
# Conversion: extracted code cells in original order with feature selection between cells 6 and 7.

# %% [code cell 1]
import os
import random
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.feature_selection import mutual_info_classif

from imblearn.over_sampling import RandomOverSampler, SMOTE, BorderlineSMOTE, ADASYN
from imblearn.combine import SMOTETomek, SMOTEENN

import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

print(f'TensorFlow version: {tf.__version__}')

# %% [code cell 2]
SEED = 42
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
KANSAS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'Integrated_weekly_KAN_20counties.csv')
OUTPUT_FOLDER = os.path.join(KANSAS_DIR, 'output_weekly_kansas_scenario2A')

# Scenario 2: Correlation-aware feature selection
CORRELATION_THRESHOLD = 0.9
TARGET_FEATURE_COUNT = 20

# Weekly setup
SEQ_LENGTH = 52
BATCH_SIZE = 64
EPOCHS = 120

# Temporal boundaries
TRAIN_END_DATE = '2019-12-31'
VAL_START_DATE = '2020-01-01'
VAL_END_DATE = '2021-12-31'
TEST_START_DATE = '2022-01-01'

# Targeted class weighting (boost None and D0)
USE_TARGETED_CLASS_BOOST = False
CLASS_WEIGHT_BOOST = {0: 1.2, 1: 1.2, 2: 1.0, 3: 1.0, 4: 1.0, 5: 1.0}

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

print('Config ready')
print(f'Data path: {DATA_PATH}')
print(f'Use targeted class boost: {USE_TARGETED_CLASS_BOOST}')
print(f'Class weight boost: {CLASS_WEIGHT_BOOST}')
print(f'Output folder: {OUTPUT_FOLDER}')
print(f'Scenario 2A: Correlation threshold = {CORRELATION_THRESHOLD}, Target feature count = {TARGET_FEATURE_COUNT}')

# %% [code cell 3]
df = pd.read_csv(DATA_PATH)
df['week_start'] = pd.to_datetime(df['week_start'])
df['ValidEnd'] = pd.to_datetime(df['ValidEnd'])
df = df.sort_values(['FIPS', 'week_start']).reset_index(drop=True)

print(f'Shape: {df.shape}')
print(f'Unique counties: {df["FIPS"].nunique()}')
print(f'Date range: {df["week_start"].min().date()} to {df["week_start"].max().date()}')
print(df.head())

# %% [code cell 4]
def decumulate_drought(row):
    pmf_d4 = row['D4']
    pmf_d3 = max(0.0, row['D3'] - row['D4'])
    pmf_d2 = max(0.0, row['D2'] - row['D3'])
    pmf_d1 = max(0.0, row['D1'] - row['D2'])
    pmf_d0 = max(0.0, row['D0'] - row['D1'])
    pmf_none = max(0.0, row['None'])
    return pd.Series([pmf_none, pmf_d0, pmf_d1, pmf_d2, pmf_d3, pmf_d4])

pmf_cols = ['PMF_None', 'PMF_D0', 'PMF_D1', 'PMF_D2', 'PMF_D3', 'PMF_D4']
df[pmf_cols] = df.apply(decumulate_drought, axis=1)
df['PMF_Sum'] = df[pmf_cols].sum(axis=1)
df['Label'] = df[pmf_cols].idxmax(axis=1).apply(lambda x: pmf_cols.index(x))
label_map = {0: 'None', 1: 'D0', 2: 'D1', 3: 'D2', 4: 'D3', 5: 'D4'}

print('PMF sum stats:')
print(df['PMF_Sum'].describe())

class_dist = df['Label'].value_counts().sort_index()
print('Class distribution (full data):')
for idx, cnt in class_dist.items():
    print(f'  {label_map[idx]:>4s}: {cnt}')

# %% [code cell 5]
base_weather = ['ALLSKY_SFC_SW_DWN', 'PRECTOTCORR', 'PS', 'RH2M', 'T2M', 'WS2M']
df_fe = df.copy().sort_values(['FIPS', 'week_start']).reset_index(drop=True)

for lag in [1, 2, 4, 8]:
    df_fe[f'PREC_lag{lag}'] = df_fe.groupby('FIPS')['PRECTOTCORR'].shift(lag)
    df_fe[f'T2M_lag{lag}'] = df_fe.groupby('FIPS')['T2M'].shift(lag)
    df_fe[f'RH2M_lag{lag}'] = df_fe.groupby('FIPS')['RH2M'].shift(lag)

for window in [4, 12]:
    df_fe[f'PREC_roll{window}_mean'] = df_fe.groupby('FIPS')['PRECTOTCORR'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
    df_fe[f'PREC_roll{window}_std'] = df_fe.groupby('FIPS')['PRECTOTCORR'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0.0))
    df_fe[f'T2M_roll{window}_mean'] = df_fe.groupby('FIPS')['T2M'].transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())

iso_week = df_fe['week_start'].dt.isocalendar().week.astype(int)
df_fe['week_sin'] = np.sin(2 * np.pi * iso_week / 52.0)
df_fe['week_cos'] = np.cos(2 * np.pi * iso_week / 52.0)

for col in ['None', 'D0', 'D1', 'D2', 'D3', 'D4']:
    df_fe[f'{col}_lag1'] = df_fe.groupby('FIPS')[col].shift(1)
    df_fe[f'{col}_lag2'] = df_fe.groupby('FIPS')[col].shift(2)

df_fe['drought_carryover_lag1'] = df_fe['D0_lag1'] + df_fe['D1_lag1'] + 0.5 * df_fe['D2_lag1']
df_fe['severe_carryover_lag1'] = df_fe['D3_lag1'] + df_fe['D4_lag1']
df_fe['heat_dry_stress'] = df_fe['T2M'] * (1.0 - df_fe['RH2M'] / 100.0)

feature_cols = [
    'ALLSKY_SFC_SW_DWN', 'PRECTOTCORR', 'PS', 'RH2M', 'T2M', 'WS2M',
    'PREC_lag1', 'PREC_lag2', 'PREC_lag4', 'PREC_lag8',
    'T2M_lag1', 'T2M_lag2', 'T2M_lag4', 'T2M_lag8',
    'RH2M_lag1', 'RH2M_lag2', 'RH2M_lag4', 'RH2M_lag8',
    'PREC_roll4_mean', 'PREC_roll4_std', 'PREC_roll12_mean', 'PREC_roll12_std',
    'T2M_roll4_mean', 'T2M_roll12_mean',
    'week_sin', 'week_cos',
    'None_lag1', 'D0_lag1', 'D1_lag1', 'D2_lag1', 'D3_lag1', 'D4_lag1',
    'None_lag2', 'D0_lag2', 'D1_lag2', 'D2_lag2', 'D3_lag2', 'D4_lag2',
    'drought_carryover_lag1', 'severe_carryover_lag1', 'heat_dry_stress'
]

before_drop = len(df_fe)
df_fe = df_fe.dropna(subset=feature_cols + ['Label']).reset_index(drop=True)
print(f'Rows before dropna: {before_drop:,}')
print(f'Rows after dropna:  {len(df_fe):,}')
print(f'Feature count (baseline): {len(feature_cols)}')

# %% [code cell 6]
train_df = df_fe[df_fe['week_start'] <= TRAIN_END_DATE].copy()
val_df = df_fe[(df_fe['week_start'] >= VAL_START_DATE) & (df_fe['week_start'] <= VAL_END_DATE)].copy()
test_df = df_fe[df_fe['week_start'] >= TEST_START_DATE].copy()

print(f'Train rows: {len(train_df):,}')
print(f'Val rows:   {len(val_df):,}')
print(f'Test rows:  {len(test_df):,}')

for split_name, split_df in [('Train', train_df), ('Val', val_df), ('Test', test_df)]:
    dist = split_df['Label'].value_counts().sort_index()
    print(f'\n{split_name} class distribution:')
    for idx in range(6):
        print(f'  {label_map[idx]:>4s}: {dist.get(idx, 0)}')

# %% [SCENARIO 2] Feature Selection: Correlation-aware mutual information ranking
print('\n' + '=' * 80)
print('SCENARIO 2A: Feature Selection (Correlation-aware Mutual Information, Top-20)')
print('=' * 80)

# Compute mutual information using train_df only
print('\n1. Computing mutual information on train set...')
mi_scores = mutual_info_classif(train_df[feature_cols], train_df['Label'], random_state=SEED)
mi_ranking = np.argsort(-mi_scores)

print(f'\nTop 20 features by mutual information:')
for rank, idx in enumerate(mi_ranking[:20], 1):
    feat_name = feature_cols[idx]
    mi_score = mi_scores[idx]
    print(f'  {rank:2d}. {feat_name:30s} MI={mi_score:.6f}')

# Compute correlation matrix using train_df only (unscaled)
print('\n2. Computing correlation matrix on train set...')
corr_matrix = train_df[feature_cols].corr().abs()

# Greedy pruning by MI rank
print(f'\n3. Greedy correlation pruning (threshold={CORRELATION_THRESHOLD})...')
selected_features = []
pruned_pairs = []

for idx in mi_ranking:
    feat_name = feature_cols[idx]
    
    # Check if this feature is highly correlated with any already-selected feature
    is_pruned = False
    for selected_feat_idx in [feature_cols.index(sf) for sf in selected_features]:
        if corr_matrix.loc[feat_name, feature_cols[selected_feat_idx]] > CORRELATION_THRESHOLD:
            is_pruned = True
            pruned_pairs.append((feat_name, feature_cols[selected_feat_idx], 
                               corr_matrix.loc[feat_name, feature_cols[selected_feat_idx]]))
            break
    
    if not is_pruned:
        selected_features.append(feat_name)
    
    if len(selected_features) >= TARGET_FEATURE_COUNT:
        break

if len(selected_features) < TARGET_FEATURE_COUNT:
    print(f'WARNING: Only {len(selected_features)} features selected (target was {TARGET_FEATURE_COUNT})')
else:
    selected_features = selected_features[:TARGET_FEATURE_COUNT]

print(f'\nPruned {len(pruned_pairs)} features due to multicollinearity:')
for feat_dropped, feat_kept, corr_val in pruned_pairs[:10]:
    print(f'  Dropped {feat_dropped:30s} (corr={corr_val:.4f} with {feat_kept})')
if len(pruned_pairs) > 10:
    print(f'  ... and {len(pruned_pairs) - 10} more')

feature_cols_corr = selected_features
print(f'\n4. Final selected features (count={len(feature_cols_corr)}):')
for i, feat in enumerate(feature_cols_corr, 1):
    print(f'  {i:2d}. {feat}')

# Replace feature_cols with selected features
feature_cols = feature_cols_corr

print('=' * 80)

# %% [code cell 7]
scaler = MinMaxScaler()
scaler.fit(train_df[feature_cols])
df_fe.loc[:, feature_cols] = scaler.transform(df_fe[feature_cols])

print('Scaling done using train-only fit.')

# %% [code cell 8]
def create_sequences_from_df(df_input, feature_columns, label_col, seq_length=52, id_col='FIPS', start_date=None, end_date=None):
    X, y = [], []
    for _, group in df_input.groupby(id_col):
        group = group.sort_values('week_start')
        feats = group[feature_columns].values
        labels = group[label_col].values
        dates = group['week_start'].values
        if len(group) < seq_length:
            continue
        for i in range(seq_length - 1, len(group)):
            target_date = pd.Timestamp(dates[i])
            if start_date is not None and target_date < pd.Timestamp(start_date):
                continue
            if end_date is not None and target_date > pd.Timestamp(end_date):
                continue
            X.append(feats[i - seq_length + 1:i + 1])
            y.append(labels[i])
    return np.array(X), np.array(y)

X_train_seq, y_train_seq = create_sequences_from_df(
    df_fe, feature_cols, 'Label', SEQ_LENGTH, start_date=None, end_date=TRAIN_END_DATE
)
X_val_seq, y_val_seq = create_sequences_from_df(
    df_fe, feature_cols, 'Label', SEQ_LENGTH, start_date=VAL_START_DATE, end_date=VAL_END_DATE
)
X_test_seq, y_test = create_sequences_from_df(
    df_fe, feature_cols, 'Label', SEQ_LENGTH, start_date=TEST_START_DATE, end_date=None
)

print(f'X_train_seq: {X_train_seq.shape}')
print(f'X_val_seq:   {X_val_seq.shape}')
print(f'X_test_seq:  {X_test_seq.shape}')
print(f'Train distribution: {Counter(y_train_seq)}')
print(f'Val distribution:   {Counter(y_val_seq)}')
print(f'Test distribution:  {Counter(y_test)}')

split_dates = {
    'train_max': train_df['week_start'].max(),
    'val_min': val_df['week_start'].min(),
    'val_max': val_df['week_start'].max(),
    'test_min': test_df['week_start'].min(),
}
print('\nSplit boundary check:')
print(split_dates)
if split_dates['train_max'] >= split_dates['val_min']:
    print('WARNING: train and val date ranges overlap.')
if split_dates['val_max'] >= split_dates['test_min']:
    print('WARNING: val and test date ranges overlap.')

# %% [code cell 9]
def get_balancer(name, seed):
    name = name.upper()
    if name == 'NONE':
        return None
    if name == 'ROS':
        return RandomOverSampler(random_state=seed)
    if name == 'SMOTE':
        return SMOTE(random_state=seed)
    if name == 'BSMOTE':
        return BorderlineSMOTE(random_state=seed)
    if name == 'ADASYN':
        return ADASYN(random_state=seed)
    if name == 'SMOTETOMEK':
        return SMOTETomek(random_state=seed)
    if name == 'SMOTEENN':
        return SMOTEENN(random_state=seed)
    raise ValueError(f'Unknown balancer: {name}')

n_steps, n_features = X_train_seq.shape[1], X_train_seq.shape[2]
num_classes = 6
y_val_enc = to_categorical(y_val_seq, num_classes=num_classes)
y_test_enc = to_categorical(y_test, num_classes=num_classes)

TRIAL_CONFIGS = [
    {
        'name': 'ros_focal_no_cw_96x48',
        'balancer': 'ROS',
        'use_class_weight': False,
        'focal_gamma': 1.5,
        'focal_alpha_mode': 'none',
        'manual_focal_alpha': None,
        'lstm_units': (96, 48),
        'dropout': 0.30,
        'dense_units': 64,
        'lr': 8e-4,
        'patience': 14,
    },
    {
        'name': 'none_focal_inv_cw_64x32',
        'balancer': 'NONE',
        'use_class_weight': True,
        'focal_gamma': 2.0,
        'focal_alpha_mode': 'inverse_train_seq',
        'manual_focal_alpha': None,
        'lstm_units': (64, 32),
        'dropout': 0.35,
        'dense_units': 64,
        'lr': 1e-3,
        'patience': 16,
    },
    {
        'name': 'ros_focal_inv_no_cw_128x64',
        'balancer': 'ROS',
        'use_class_weight': False,
        'focal_gamma': 2.0,
        'focal_alpha_mode': 'inverse_train_seq',
        'manual_focal_alpha': None,
        'lstm_units': (128, 64),
        'dropout': 0.28,
        'dense_units': 96,
        'lr': 6e-4,
        'patience': 14,
    },
]

def compute_focal_alpha(y_labels, n_classes, mode='inverse_train_seq', manual_alpha=None):
    if mode == 'none':
        return None
    if mode == 'manual':
        if manual_alpha is None or len(manual_alpha) != n_classes:
            raise ValueError('manual_alpha must be provided with length equal to n_classes.')
        alpha = np.array(manual_alpha, dtype=np.float32)
    elif mode == 'inverse_train_seq':
        counts = np.bincount(y_labels.astype(int), minlength=n_classes).astype(np.float32)
        inv = 1.0 / np.maximum(counts, 1.0)
        alpha = inv / inv.sum()
    else:
        raise ValueError(f'Unknown FOCAL_ALPHA_MODE: {mode}')
    return alpha.tolist()

def categorical_focal_loss(gamma=2.0, alpha=None):
    def focal_loss(y_true, y_pred):
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1.0 - eps)
        ce = -y_true * K.log(y_pred)
        fw = K.pow(1.0 - y_pred, gamma)
        loss = fw * ce
        if alpha is not None:
            loss = tf.constant(alpha, dtype=tf.float32) * loss
        return K.sum(loss, axis=-1)
    return focal_loss

def build_model(seq_length, n_features, n_classes, lstm_units=(64, 32), dropout=0.3, dense_units=64, lr=1e-3, focal_gamma=2.0, focal_alpha=None):
    u1, u2 = lstm_units
    model = Sequential([
        Input(shape=(seq_length, n_features)),
        Bidirectional(LSTM(u1, return_sequences=True, dropout=0.2, recurrent_dropout=0.1)),
        BatchNormalization(),
        Dropout(dropout),
        Bidirectional(LSTM(u2, dropout=0.2, recurrent_dropout=0.1)),
        BatchNormalization(),
        Dropout(dropout),
        Dense(dense_units, activation='relu'),
        Dropout(min(0.4, dropout)),
        Dense(n_classes, activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=categorical_focal_loss(gamma=focal_gamma, alpha=focal_alpha),
        metrics=['accuracy']
    )
    return model

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
        score = f1_score(val_true_label, val_pred_label, average='macro', zero_division=0)
        logs['val_macro_f1'] = score
        print(f' - val_macro_f1: {score:.4f}', end='')

def prepare_training_data(x_train, y_train, balancer_name, seed):
    balancer = get_balancer(balancer_name, seed)
    if balancer is None:
        return x_train, y_train
    n_samples_local, n_steps_local, n_features_local = x_train.shape
    x_train_flat = x_train.reshape(n_samples_local, n_steps_local * n_features_local)
    x_bal_flat, y_bal = balancer.fit_resample(x_train_flat, y_train)
    x_bal = x_bal_flat.reshape(-1, n_steps_local, n_features_local)
    return x_bal, y_bal

def compute_class_weights(y_labels):
    cw = compute_class_weight(class_weight='balanced', classes=np.arange(num_classes), y=y_labels)
    cw = cw / cw.mean()
    cw_dict = dict(enumerate(cw))
    if USE_TARGETED_CLASS_BOOST:
        for cls_idx, boost in CLASS_WEIGHT_BOOST.items():
            cw_dict[cls_idx] = cw_dict.get(cls_idx, 1.0) * float(boost)
        mean_cw = np.mean(list(cw_dict.values()))
        cw_dict = {k: v / mean_cw for k, v in cw_dict.items()}
    return cw_dict

trial_results = []
best_trial = None
best_val_macro_f1 = -1.0
best_model = None
best_history = None

print('\nStarting hyperparameter trials...')
for i, cfg in enumerate(TRIAL_CONFIGS, start=1):
    print('\n' + '=' * 80)
    print(f'Trial {i}/{len(TRIAL_CONFIGS)}: {cfg["name"]}')
    print(cfg)

    X_train_bal, y_train_bal = prepare_training_data(X_train_seq, y_train_seq, cfg['balancer'], SEED + i)
    y_train_enc = to_categorical(y_train_bal, num_classes=num_classes)

    focal_alpha = compute_focal_alpha(
        y_train_bal,
        n_classes=num_classes,
        mode=cfg['focal_alpha_mode'],
        manual_alpha=cfg['manual_focal_alpha'],
    )
    print('Focal alpha:', focal_alpha)
    print(f'Before balancing: {Counter(y_train_seq)}')
    print(f'After balancing:  {Counter(y_train_bal)}')

    model = build_model(
        SEQ_LENGTH,
        n_features,
        num_classes,
        lstm_units=cfg['lstm_units'],
        dropout=cfg['dropout'],
        dense_units=cfg['dense_units'],
        lr=cfg['lr'],
        focal_gamma=cfg['focal_gamma'],
        focal_alpha=focal_alpha,
    )

    trial_ckpt = f'{OUTPUT_FOLDER}/best_model_{cfg["name"]}.keras'
    callbacks = [
        MacroF1Callback(X_val_seq, y_val_enc),
        EarlyStopping(monitor='val_macro_f1', mode='max', patience=cfg['patience'], restore_best_weights=True, verbose=1),
        ModelCheckpoint(trial_ckpt, monitor='val_macro_f1', mode='max', save_best_only=True, verbose=0),
        ReduceLROnPlateau(monitor='val_macro_f1', mode='max', factor=0.5, patience=max(4, cfg['patience'] // 3), min_lr=1e-6, verbose=1),
    ]

    fit_kwargs = {}
    if cfg['use_class_weight']:
        class_weight_dict = compute_class_weights(y_train_bal)
        print('Class weights:', class_weight_dict)
        fit_kwargs['class_weight'] = class_weight_dict
    else:
        print('Class weights: not used')

    history = model.fit(
        X_train_bal, y_train_enc,
        validation_data=(X_val_seq, y_val_enc),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1,
        **fit_kwargs,
    )

    val_pred_prob = model.predict(X_val_seq, verbose=0)
    val_pred = np.argmax(val_pred_prob, axis=1)
    val_macro_f1 = f1_score(y_val_seq, val_pred, average='macro', zero_division=0)
    trial_results.append({'name': cfg['name'], 'val_macro_f1': float(val_macro_f1), 'balancer': cfg['balancer']})
    print(f'Trial {cfg["name"]} val macro-F1: {val_macro_f1:.4f}')

    if val_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = val_macro_f1
        best_trial = cfg
        best_model = model
        best_history = history

print('\nTrial leaderboard (by validation macro-F1):')
for row in sorted(trial_results, key=lambda x: x['val_macro_f1'], reverse=True):
    print(f'  {row["name"]}: {row["val_macro_f1"]:.4f} (balancer={row["balancer"]})')

print(f'\nBest trial: {best_trial["name"]} with val macro-F1 {best_val_macro_f1:.4f}')
best_model.save(f'{OUTPUT_FOLDER}/best_model.keras')

def tune_class_multipliers(y_true, y_prob, n_classes=6, n_iter=2500, seed=SEED):
    rng = np.random.default_rng(seed)
    best_m = np.ones(n_classes, dtype=np.float32)
    base_pred = np.argmax(y_prob, axis=1)
    best_score = f1_score(y_true, base_pred, average='macro', zero_division=0)

    for _ in range(n_iter):
        cand = np.exp(rng.normal(loc=0.0, scale=0.30, size=n_classes)).astype(np.float32)
        cand = np.clip(cand, 0.55, 1.8)
        pred = np.argmax(y_prob * cand, axis=1)
        score = f1_score(y_true, pred, average='macro', zero_division=0)
        if score > best_score:
            best_score = score
            best_m = cand

    return best_m, float(best_score)

val_best_prob = best_model.predict(X_val_seq, verbose=0)
class_multipliers, tuned_val_macro_f1 = tune_class_multipliers(y_val_seq, val_best_prob, n_classes=num_classes, n_iter=2500, seed=SEED)
raw_val_macro_f1 = f1_score(y_val_seq, np.argmax(val_best_prob, axis=1), average='macro', zero_division=0)
print('\nClass multiplier calibration (from validation only):')
print(f'  raw val macro-F1:   {raw_val_macro_f1:.4f}')
print(f'  tuned val macro-F1: {tuned_val_macro_f1:.4f}')
print(f'  multipliers: {class_multipliers}')

# %% [code cell 12]
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
axes[0].plot(best_history.history['loss'], label='Train')
axes[0].plot(best_history.history['val_loss'], label='Val')
axes[0].set_title('Loss')
axes[0].set_xlabel('Epoch')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(best_history.history['accuracy'], label='Train')
axes[1].plot(best_history.history['val_accuracy'], label='Val')
axes[1].set_title('Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/training_history.png', dpi=140)
plt.show()

# %% [code cell 13]
y_pred_prob = best_model.predict(X_test_seq, verbose=0)
y_pred_raw = np.argmax(y_pred_prob, axis=1)
y_pred = np.argmax(y_pred_prob * class_multipliers, axis=1)

present_classes = sorted(set(y_test) | set(y_pred))
target_names = [label_map[i] for i in present_classes]

report = classification_report(y_test, y_pred, labels=present_classes, target_names=target_names, digits=4, zero_division=0)
accuracy = accuracy_score(y_test, y_pred)
macro_f1 = f1_score(y_test, y_pred, average='macro', zero_division=0)
weighted_f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
macro_f1_raw = f1_score(y_test, y_pred_raw, average='macro', zero_division=0)
accuracy_raw = accuracy_score(y_test, y_pred_raw)

print('=' * 70)
print('CLASSIFICATION REPORT')
print('=' * 70)
print(report)
print('=' * 70)
print(f'Accuracy (raw):    {accuracy_raw:.4f}')
print(f'Macro F1 (raw):    {macro_f1_raw:.4f}')
print(f'Accuracy:    {accuracy:.4f}')
print(f'Macro F1:    {macro_f1:.4f}')
print(f'Weighted F1: {weighted_f1:.4f}')
print('=' * 70)

# %% [code cell 14]
num_classes = 6
cm = confusion_matrix(y_test, y_pred, labels=list(range(num_classes)))

fig, axes = plt.subplots(1, 2, figsize=(18, 7))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[label_map[i] for i in range(num_classes)],
            yticklabels=[label_map[i] for i in range(num_classes)],
            ax=axes[0])
axes[0].set_title('Confusion Matrix (Counts)')
axes[0].set_xlabel('Predicted')
axes[0].set_ylabel('Actual')

cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)
cm_norm = np.nan_to_num(cm_norm)
sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Blues',
            xticklabels=[label_map[i] for i in range(num_classes)],
            yticklabels=[label_map[i] for i in range(num_classes)],
            ax=axes[1])
axes[1].set_title('Confusion Matrix (Normalized)')
axes[1].set_xlabel('Predicted')
axes[1].set_ylabel('Actual')

plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/confusion_matrix.png', dpi=140)
plt.show()

per_class_f1 = f1_score(y_test, y_pred, labels=list(range(num_classes)), average=None, zero_division=0)
fig, ax = plt.subplots(figsize=(10, 5))
bars = ax.bar([label_map[i] for i in range(num_classes)], per_class_f1)
for bar, val in zip(bars, per_class_f1):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01, f'{val:.3f}', ha='center')
ax.axhline(macro_f1, color='red', linestyle='--', label=f'Macro F1: {macro_f1:.4f}')
ax.set_ylim(0, 1.05)
ax.set_title('Per-Class F1')
ax.grid(alpha=0.3, axis='y')
ax.legend()
plt.tight_layout()
plt.savefig(f'{OUTPUT_FOLDER}/per_class_f1.png', dpi=140)
plt.show()

# %% [code cell 15]
summary_path = f'{OUTPUT_FOLDER}/results_summary.txt'
with open(summary_path, 'w', encoding='utf-8') as f:
    f.write('BiLSTM Weekly Kansas - Scenario 2A: Correlation-aware Feature Selection (Top-20)\n')
    f.write(f'Best trial: {best_trial["name"]}\n')
    f.write(f'Best trial config: {best_trial}\n')
    f.write(f'Seq Length: {SEQ_LENGTH}\n')
    f.write(f'\n--- FEATURE SELECTION ---\n')
    f.write(f'Baseline feature count: 41\n')
    f.write(f'Correlation threshold: {CORRELATION_THRESHOLD}\n')
    f.write(f'Final selected feature count: {len(feature_cols_corr)}\n')
    f.write(f'Selected features: {feature_cols_corr}\n')
    f.write(f'\n--- RESULTS ---\n')
    f.write(f'Val Macro F1 (best trial): {best_val_macro_f1:.4f}\n')
    f.write(f'Val Macro F1 (raw/tuned): {raw_val_macro_f1:.4f} / {tuned_val_macro_f1:.4f}\n')
    f.write(f'Class multipliers: {class_multipliers.tolist()}\n')
    f.write(f'Accuracy (raw): {accuracy_raw:.4f}\n')
    f.write(f'Macro F1 (raw): {macro_f1_raw:.4f}\n')
    f.write(f'Accuracy: {accuracy:.4f}\n')
    f.write(f'Macro F1: {macro_f1:.4f}\n')
    f.write(f'Weighted F1: {weighted_f1:.4f}\n\n')
    f.write('Trial leaderboard:\n')
    for row in sorted(trial_results, key=lambda x: x['val_macro_f1'], reverse=True):
        f.write(f'  {row["name"]}: {row["val_macro_f1"]:.4f} (balancer={row["balancer"]})\n')
    f.write('\n')
    f.write('Per-class F1:\n')
    for i in range(num_classes):
        f.write(f'  {label_map[i]}: {per_class_f1[i]:.4f}\n')
    f.write('\nClassification Report:\n')
    f.write(report)

print(f'Results saved to {summary_path}')
