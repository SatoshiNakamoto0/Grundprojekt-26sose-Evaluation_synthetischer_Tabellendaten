import numpy as np
import pandas as pd
import math
import random

COLUMNS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
COLUMN_STATS = {'numeric': {'Pregnancies': {'mean': 3.78, 'std': 3.2728580781940426, 'min': 0.0, 'q05': 0.0, 'q25': 1.0, 'q50': 3.0, 'q75': 5.0, 'q95': 10.0, 'max': 14.0}, 'Glucose': {'mean': 124.37, 'std': 31.631836810403534, 'min': 77.0, 'q05': 82.95, 'q25': 100.0, 'q50': 115.0, 'q75': 141.0, 'q95': 187.05, 'max': 197.0}, 'BloodPressure': {'mean': 69.42, 'std': 16.17972805704719, 'min': 0.0, 'q05': 51.9, 'q25': 63.5, 'q50': 72.0, 'q75': 78.5, 'q95': 88.1, 'max': 98.0}, 'SkinThickness': {'mean': 22.38, 'std': 17.202778845291245, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 25.5, 'q75': 33.25, 'q95': 45.0, 'max': 99.0}, 'Insulin': {'mean': 87.55, 'std': 129.80418906953656, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 54.5, 'q75': 132.75, 'q95': 307.29999999999984, 'max': 744.0}, 'BMI': {'mean': 31.787999999999997, 'std': 6.762267075471065, 'min': 18.2, 'q05': 23.155, 'q25': 26.4, 'q50': 30.6, 'q75': 36.525, 'q95': 45.63, 'max': 52.9}, 'DiabetesPedigreeFunction': {'mean': 0.46009, 'std': 0.3113400101175562, 'min': 0.088, 'q05': 0.14679999999999999, 'q25': 0.23675, 'q50': 0.382, 'q75': 0.61775, 'q95': 0.9448999999999999, 'max': 2.329}, 'Age': {'mean': 33.42, 'std': 11.447427658648907, 'min': 21.0, 'q05': 21.0, 'q25': 24.0, 'q50': 29.0, 'q75': 41.0, 'q95': 58.05, 'max': 62.0}, 'Outcome': {'mean': 0.35, 'std': 0.4769696007084728, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 0.0, 'q75': 1.0, 'q95': 1.0, 'max': 1.0}}, 'categorical': {}, 'target_probs': {'values': [0, 1], 'probs': [0.6470588235294118, 0.35294117647058826]}}
NUMERIC_STATS = COLUMN_STATS.get('numeric', {})
CATEGORICAL_STATS = COLUMN_STATS.get('categorical', {})
TARGET_STATS = COLUMN_STATS.get('target_probs', {})
FLAT_COLUMN_STATS = {'Pregnancies': {'mean': 3.78, 'std': 3.2728580781940426, 'min': 0.0, 'q05': 0.0, 'q25': 1.0, 'q50': 3.0, 'q75': 5.0, 'q95': 10.0, 'max': 14.0}, 'Glucose': {'mean': 124.37, 'std': 31.631836810403534, 'min': 77.0, 'q05': 82.95, 'q25': 100.0, 'q50': 115.0, 'q75': 141.0, 'q95': 187.05, 'max': 197.0}, 'BloodPressure': {'mean': 69.42, 'std': 16.17972805704719, 'min': 0.0, 'q05': 51.9, 'q25': 63.5, 'q50': 72.0, 'q75': 78.5, 'q95': 88.1, 'max': 98.0}, 'SkinThickness': {'mean': 22.38, 'std': 17.202778845291245, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 25.5, 'q75': 33.25, 'q95': 45.0, 'max': 99.0}, 'Insulin': {'mean': 87.55, 'std': 129.80418906953656, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 54.5, 'q75': 132.75, 'q95': 307.29999999999984, 'max': 744.0}, 'BMI': {'mean': 31.787999999999997, 'std': 6.762267075471065, 'min': 18.2, 'q05': 23.155, 'q25': 26.4, 'q50': 30.6, 'q75': 36.525, 'q95': 45.63, 'max': 52.9}, 'DiabetesPedigreeFunction': {'mean': 0.46009, 'std': 0.3113400101175562, 'min': 0.088, 'q05': 0.14679999999999999, 'q25': 0.23675, 'q50': 0.382, 'q75': 0.61775, 'q95': 0.9448999999999999, 'max': 2.329}, 'Age': {'mean': 33.42, 'std': 11.447427658648907, 'min': 21.0, 'q05': 21.0, 'q25': 24.0, 'q50': 29.0, 'q75': 41.0, 'q95': 58.05, 'max': 62.0}, 'Outcome': {'values': [0, 1], 'probs': [0.6470588235294118, 0.35294117647058826]}}
ALLOWED_VALUES = {'Outcome': [0, 1]}
EDUCATION_TO_NUM = {'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16}
CODE_LLM_METADATA = {'dataset': 'pima', 'seed': 123, 'scaffold_version': 'Code-LLM-v2', 'generation_success': True, 'compile_success': True, 'runtime_success': True, 'validation_success': True, 'fallback_used': False, 'failure_stage': '', 'failure_reason': '', 'llm_attempts': 1}

def sample_categorical(rng, values, probs=None):
    if isinstance(values, dict):
        probs = values.get('probs', probs)
        values = values.get('values', [])
    if isinstance(values, str) or not hasattr(values, '__iter__'):
        values = [values]
    values = list(values) if values is not None else []
    if not values:
        return np.nan
    try:
        p = np.asarray(probs, dtype=float) if probs is not None else np.ones(len(values), dtype=float)
    except Exception:
        p = np.ones(len(values), dtype=float)
    if p.ndim != 1 or len(p) != len(values) or not np.all(np.isfinite(p)) or np.any(p < 0) or float(p.sum()) <= 0.0:
        p = np.ones(len(values), dtype=float)
    p = p / p.sum()
    return values[int(rng.choice(len(values), p=p))]

def sample_numeric_quantile_jitter(rng, stats, integer=False):
    anchors = np.array([stats['q05'], stats['q25'], stats['q50'], stats['q75'], stats['q95']], dtype=float)
    value = float(rng.choice(anchors))
    value += float(rng.normal(0.0, max((stats['q95'] - stats['q05']) / 12.0, 1e-6)))
    value = float(np.clip(value, stats['min'], stats['max']))
    return int(round(value)) if integer else value

def sample_truncated_normal(rng, mean, std=None, low=None, high=None, integer=False):
    if isinstance(mean, dict):
        stats = mean
        mean = stats.get('mean', stats.get('q50', 0.0))
        std = stats.get('std', std)
        if std is None:
            std = max((float(stats.get('q95', mean)) - float(stats.get('q05', mean))) / 4.0, 1e-6)
        low = stats.get('low', stats.get('min', stats.get('q05', low)))
        high = stats.get('high', stats.get('max', stats.get('q95', high)))
    mean = float(0.0 if mean is None or pd.isna(mean) else mean)
    std = float(1e-6 if std is None or pd.isna(std) else std)
    std = max(std, 1e-6)
    if low is None or pd.isna(low):
        low = mean - 4.0 * std
    if high is None or pd.isna(high):
        high = mean + 4.0 * std
    low, high = float(low), float(high)
    if low > high:
        low, high = high, low
    value = float(np.clip(rng.normal(mean, std), low, high))
    return int(round(value)) if integer else value


def generate_synthetic_data(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        rows.append({
            "Pregnancies": sample_numeric_quantile_jitter(rng, NUMERIC_STATS["Pregnancies"], True),
            "Glucose": sample_numeric_quantile_jitter(rng, NUMERIC_STATS["Glucose"], True),
            "BloodPressure": sample_numeric_quantile_jitter(rng, NUMERIC_STATS["BloodPressure"], True),
            "SkinThickness": sample_numeric_quantile_jitter(rng, NUMERIC_STATS["SkinThickness"], True),
            "Insulin": sample_numeric_quantile_jitter(rng, NUMERIC_STATS["Insulin"], True),
            "BMI": sample_numeric_quantile_jitter(rng, NUMERIC_STATS["BMI"], False),
            "DiabetesPedigreeFunction": sample_numeric_quantile_jitter(rng, NUMERIC_STATS["DiabetesPedigreeFunction"], False),
            "Age": sample_numeric_quantile_jitter(rng, NUMERIC_STATS["Age"], True),
            "Outcome": int(sample_categorical(rng, TARGET_STATS)),
        })
    return pd.DataFrame(rows[:n], columns=COLUMNS)
