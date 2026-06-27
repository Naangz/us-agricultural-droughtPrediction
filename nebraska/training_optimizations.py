import os

import numpy as np
import pandas as pd
import tensorflow as tf


PMF_COLS = ['PMF_None', 'PMF_D0', 'PMF_D1', 'PMF_D2', 'PMF_D3', 'PMF_D4']


def configure_cpu_runtime(seed):
    logical_cpus = os.cpu_count() or 1
    intra_threads = min(10, logical_cpus)
    inter_threads = 2 if logical_cpus >= 8 else 1

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ.setdefault('TF_ENABLE_ONEDNN_OPTS', '1')
    os.environ.setdefault('OMP_NUM_THREADS', str(intra_threads))
    os.environ.setdefault('TF_NUM_INTRAOP_THREADS', str(intra_threads))
    os.environ.setdefault('TF_NUM_INTEROP_THREADS', str(inter_threads))

    try:
        tf.config.threading.set_intra_op_parallelism_threads(intra_threads)
        tf.config.threading.set_inter_op_parallelism_threads(inter_threads)
    except RuntimeError:
        # Threading config can no longer be changed after TF runtime initialization.
        pass

    return {
        'logical_cpus': logical_cpus,
        'intra_threads': intra_threads,
        'inter_threads': inter_threads,
    }


def add_drought_labels(df):
    d4 = df['D4'].to_numpy(dtype=np.float32, copy=False)
    d3 = np.maximum(0.0, df['D3'].to_numpy(dtype=np.float32, copy=False) - d4)
    d2 = np.maximum(0.0, df['D2'].to_numpy(dtype=np.float32, copy=False) - df['D3'].to_numpy(dtype=np.float32, copy=False))
    d1 = np.maximum(0.0, df['D1'].to_numpy(dtype=np.float32, copy=False) - df['D2'].to_numpy(dtype=np.float32, copy=False))
    d0 = np.maximum(0.0, df['D0'].to_numpy(dtype=np.float32, copy=False) - df['D1'].to_numpy(dtype=np.float32, copy=False))
    none = np.maximum(0.0, df['None'].to_numpy(dtype=np.float32, copy=False))

    pmf_values = np.column_stack([none, d0, d1, d2, d3, d4]).astype(np.float32, copy=False)

    df = df.copy()
    df[PMF_COLS] = pmf_values
    df['PMF_Sum'] = pmf_values.sum(axis=1, dtype=np.float32)
    df['Label'] = pmf_values.argmax(axis=1).astype(np.int8, copy=False)
    return df


def engineer_base_features(df):
    df_fe = df.copy().sort_values(['FIPS', 'week_start']).reset_index(drop=True)
    county_ids = df_fe['FIPS']
    grouped = df_fe.groupby('FIPS', sort=False)

    for lag in [1, 2, 4, 8]:
        df_fe[f'PREC_lag{lag}'] = grouped['PRECTOTCORR'].shift(lag)
        df_fe[f'T2M_lag{lag}'] = grouped['T2M'].shift(lag)
        df_fe[f'RH2M_lag{lag}'] = grouped['RH2M'].shift(lag)

    shifted_prec = grouped['PRECTOTCORR'].shift(1)
    shifted_t2m = grouped['T2M'].shift(1)
    prec_grouped = shifted_prec.groupby(county_ids, sort=False)
    t2m_grouped = shifted_t2m.groupby(county_ids, sort=False)

    for window in [4, 12]:
        df_fe[f'PREC_roll{window}_mean'] = (
            prec_grouped.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )
        df_fe[f'PREC_roll{window}_std'] = (
            prec_grouped.rolling(window, min_periods=1).std().reset_index(level=0, drop=True).fillna(0.0)
        )
        df_fe[f'T2M_roll{window}_mean'] = (
            t2m_grouped.rolling(window, min_periods=1).mean().reset_index(level=0, drop=True)
        )

    iso_week = df_fe['week_start'].dt.isocalendar().week.astype(np.int16)
    df_fe['week_sin'] = np.sin(2 * np.pi * iso_week / 52.0).astype(np.float32)
    df_fe['week_cos'] = np.cos(2 * np.pi * iso_week / 52.0).astype(np.float32)

    for col in ['None', 'D0', 'D1', 'D2', 'D3', 'D4']:
        df_fe[f'{col}_lag1'] = grouped[col].shift(1)
        df_fe[f'{col}_lag2'] = grouped[col].shift(2)

    df_fe['heat_dry_stress'] = (df_fe['T2M'] * (1.0 - df_fe['RH2M'] / 100.0)).astype(np.float32)
    return df_fe


def scale_features_inplace(df_fe, train_df, feature_cols, scaler):
    scaler.fit(train_df[feature_cols])
    scaled = scaler.transform(df_fe[feature_cols]).astype(np.float32, copy=False)
    df_fe.loc[:, feature_cols] = scaled
    return df_fe


def create_sequences_from_df_fast(
    df_input,
    feature_columns,
    label_col,
    seq_length=52,
    id_col='FIPS',
    start_date=None,
    end_date=None,
):
    x_parts = []
    y_parts = []
    start_dt = np.datetime64(start_date) if start_date is not None else None
    end_dt = np.datetime64(end_date) if end_date is not None else None

    for _, group in df_input.groupby(id_col, sort=False):
        group = group.sort_values('week_start')
        feats = np.ascontiguousarray(group[feature_columns].to_numpy(dtype=np.float32, copy=False))
        labels = group[label_col].to_numpy(dtype=np.int32, copy=False)
        dates = group['week_start'].to_numpy(dtype='datetime64[ns]')

        if len(group) < seq_length:
            continue

        windows = np.lib.stride_tricks.sliding_window_view(feats, window_shape=seq_length, axis=0)
        windows = np.moveaxis(windows, -1, 1)

        target_dates = dates[seq_length - 1:]
        valid_mask = np.ones(target_dates.shape[0], dtype=bool)
        if start_dt is not None:
            valid_mask &= target_dates >= start_dt
        if end_dt is not None:
            valid_mask &= target_dates <= end_dt
        if not valid_mask.any():
            continue

        x_parts.append(np.ascontiguousarray(windows[valid_mask], dtype=np.float32))
        y_parts.append(labels[seq_length - 1:][valid_mask])

    if not x_parts:
        return (
            np.empty((0, seq_length, len(feature_columns)), dtype=np.float32),
            np.empty((0,), dtype=np.int32),
        )

    return np.concatenate(x_parts, axis=0), np.concatenate(y_parts, axis=0)


def ensure_float32_3d(x):
    return np.ascontiguousarray(x, dtype=np.float32)


def batched_predict(model, x, batch_size):
    return model.predict(x, batch_size=batch_size, verbose=0)
