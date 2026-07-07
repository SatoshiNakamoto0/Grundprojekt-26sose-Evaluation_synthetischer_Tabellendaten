import numpy as np
import pandas as pd
import math
import random

COLUMNS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
COLUMN_STATS = {'numeric': {'Pregnancies': {'mean': 3.83, 'std': 3.1813676304382055, 'min': 0.0, 'q05': 0.0, 'q25': 1.0, 'q50': 3.5, 'q75': 6.0, 'q95': 9.049999999999997, 'max': 13.0}, 'Glucose': {'mean': 125.2, 'std': 30.58332879200693, 'min': 68.0, 'q05': 81.0, 'q25': 102.75, 'q50': 121.5, 'q75': 143.75, 'q95': 184.14999999999998, 'max': 197.0}, 'BloodPressure': {'mean': 67.56, 'std': 20.948661055065067, 'min': 0.0, 'q05': 0.0, 'q25': 62.0, 'q50': 70.0, 'q75': 78.0, 'q95': 90.0, 'max': 108.0}, 'SkinThickness': {'mean': 19.61, 'std': 15.652408760315456, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 21.0, 'q75': 32.0, 'q95': 43.0, 'max': 60.0}, 'Insulin': {'mean': 84.56, 'std': 118.85169918852655, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 54.5, 'q75': 129.75, 'q95': 285.8499999999999, 'max': 680.0}, 'BMI': {'mean': 32.558, 'std': 7.7669193378069785, 'min': 0.0, 'q05': 23.38, 'q25': 26.6, 'q50': 32.4, 'q75': 38.425000000000004, 'q95': 45.01, 'max': 52.3}, 'DiabetesPedigreeFunction': {'mean': 0.46819999999999984, 'std': 0.35601151666764935, 'min': 0.089, 'q05': 0.15895, 'q25': 0.229, 'q50': 0.351, 'q75': 0.64825, 'q95': 1.1093499999999996, 'max': 2.288}, 'Age': {'mean': 33.84, 'std': 12.25293434243406, 'min': 21.0, 'q05': 21.0, 'q25': 24.0, 'q50': 28.5, 'q75': 41.25, 'q95': 62.05, 'max': 66.0}, 'Outcome': {'mean': 0.35, 'std': 0.4769696007084728, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 0.0, 'q75': 1.0, 'q95': 1.0, 'max': 1.0}}, 'categorical': {}, 'target_probs': {'values': [0, 1], 'probs': [0.6470588235294118, 0.35294117647058826]}}
NUMERIC_STATS = COLUMN_STATS.get('numeric', {})
CATEGORICAL_STATS = COLUMN_STATS.get('categorical', {})
TARGET_STATS = COLUMN_STATS.get('target_probs', {})
FLAT_COLUMN_STATS = {'Pregnancies': {'mean': 3.83, 'std': 3.1813676304382055, 'min': 0.0, 'q05': 0.0, 'q25': 1.0, 'q50': 3.5, 'q75': 6.0, 'q95': 9.049999999999997, 'max': 13.0}, 'Glucose': {'mean': 125.2, 'std': 30.58332879200693, 'min': 68.0, 'q05': 81.0, 'q25': 102.75, 'q50': 121.5, 'q75': 143.75, 'q95': 184.14999999999998, 'max': 197.0}, 'BloodPressure': {'mean': 67.56, 'std': 20.948661055065067, 'min': 0.0, 'q05': 0.0, 'q25': 62.0, 'q50': 70.0, 'q75': 78.0, 'q95': 90.0, 'max': 108.0}, 'SkinThickness': {'mean': 19.61, 'std': 15.652408760315456, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 21.0, 'q75': 32.0, 'q95': 43.0, 'max': 60.0}, 'Insulin': {'mean': 84.56, 'std': 118.85169918852655, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 54.5, 'q75': 129.75, 'q95': 285.8499999999999, 'max': 680.0}, 'BMI': {'mean': 32.558, 'std': 7.7669193378069785, 'min': 0.0, 'q05': 23.38, 'q25': 26.6, 'q50': 32.4, 'q75': 38.425000000000004, 'q95': 45.01, 'max': 52.3}, 'DiabetesPedigreeFunction': {'mean': 0.46819999999999984, 'std': 0.35601151666764935, 'min': 0.089, 'q05': 0.15895, 'q25': 0.229, 'q50': 0.351, 'q75': 0.64825, 'q95': 1.1093499999999996, 'max': 2.288}, 'Age': {'mean': 33.84, 'std': 12.25293434243406, 'min': 21.0, 'q05': 21.0, 'q25': 24.0, 'q50': 28.5, 'q75': 41.25, 'q95': 62.05, 'max': 66.0}, 'Outcome': {'values': [0, 1], 'probs': [0.6470588235294118, 0.35294117647058826]}}
ALLOWED_VALUES = {'Outcome': [0, 1]}
EDUCATION_TO_NUM = {'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16}
CODE_LLM_METADATA = {'dataset': 'pima', 'seed': 2025, 'scaffold_version': 'Code-LLM-v2', 'generation_success': True, 'compile_success': True, 'runtime_success': True, 'validation_success': True, 'fallback_used': False, 'failure_stage': '', 'failure_reason': '', 'llm_attempts': 1}

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
