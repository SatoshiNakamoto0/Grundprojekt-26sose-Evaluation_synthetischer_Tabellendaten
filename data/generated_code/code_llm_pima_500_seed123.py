import numpy as np
import pandas as pd
import math
import random

COLUMNS = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
COLUMN_STATS = {'numeric': {'Pregnancies': {'mean': 3.82, 'std': 3.423390132602476, 'min': 0.0, 'q05': 0.0, 'q25': 1.0, 'q50': 3.0, 'q75': 6.0, 'q95': 10.0, 'max': 17.0}, 'Glucose': {'mean': 122.192, 'std': 31.55108771500596, 'min': 0.0, 'q05': 80.95, 'q25': 100.0, 'q50': 118.0, 'q75': 141.0, 'q95': 181.0, 'max': 198.0}, 'BloodPressure': {'mean': 68.868, 'std': 19.655802603811424, 'min': 0.0, 'q05': 28.500000000000085, 'q25': 62.0, 'q50': 71.0, 'q75': 80.0, 'q95': 90.09999999999991, 'max': 110.0}, 'SkinThickness': {'mean': 20.078, 'std': 16.21209166023928, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 22.0, 'q75': 32.0, 'q95': 45.0, 'max': 99.0}, 'Insulin': {'mean': 80.656, 'std': 120.09924922329864, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 26.0, 'q75': 125.25, 'q95': 293.3499999999997, 'max': 846.0}, 'BMI': {'mean': 31.7146, 'std': 7.81995951651925, 'min': 0.0, 'q05': 21.195, 'q25': 26.75, 'q50': 32.0, 'q75': 36.3, 'q95': 43.504999999999995, 'max': 67.1}, 'DiabetesPedigreeFunction': {'mean': 0.45868600000000004, 'std': 0.32554995224081973, 'min': 0.078, 'q05': 0.141, 'q25': 0.235, 'q50': 0.354, 'q75': 0.605, 'q95': 1.036099999999998, 'max': 2.329}, 'Age': {'mean': 33.314, 'std': 12.047547634269804, 'min': 21.0, 'q05': 21.0, 'q25': 24.0, 'q50': 29.0, 'q75': 41.0, 'q95': 58.049999999999955, 'max': 81.0}, 'Outcome': {'mean': 0.348, 'std': 0.47633601585435464, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 0.0, 'q75': 1.0, 'q95': 1.0, 'max': 1.0}}, 'categorical': {}, 'target_probs': {'values': [0, 1], 'probs': [0.651394422310757, 0.34860557768924305]}}
NUMERIC_STATS = COLUMN_STATS.get('numeric', {})
CATEGORICAL_STATS = COLUMN_STATS.get('categorical', {})
TARGET_STATS = COLUMN_STATS.get('target_probs', {})
FLAT_COLUMN_STATS = {'Pregnancies': {'mean': 3.82, 'std': 3.423390132602476, 'min': 0.0, 'q05': 0.0, 'q25': 1.0, 'q50': 3.0, 'q75': 6.0, 'q95': 10.0, 'max': 17.0}, 'Glucose': {'mean': 122.192, 'std': 31.55108771500596, 'min': 0.0, 'q05': 80.95, 'q25': 100.0, 'q50': 118.0, 'q75': 141.0, 'q95': 181.0, 'max': 198.0}, 'BloodPressure': {'mean': 68.868, 'std': 19.655802603811424, 'min': 0.0, 'q05': 28.500000000000085, 'q25': 62.0, 'q50': 71.0, 'q75': 80.0, 'q95': 90.09999999999991, 'max': 110.0}, 'SkinThickness': {'mean': 20.078, 'std': 16.21209166023928, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 22.0, 'q75': 32.0, 'q95': 45.0, 'max': 99.0}, 'Insulin': {'mean': 80.656, 'std': 120.09924922329864, 'min': 0.0, 'q05': 0.0, 'q25': 0.0, 'q50': 26.0, 'q75': 125.25, 'q95': 293.3499999999997, 'max': 846.0}, 'BMI': {'mean': 31.7146, 'std': 7.81995951651925, 'min': 0.0, 'q05': 21.195, 'q25': 26.75, 'q50': 32.0, 'q75': 36.3, 'q95': 43.504999999999995, 'max': 67.1}, 'DiabetesPedigreeFunction': {'mean': 0.45868600000000004, 'std': 0.32554995224081973, 'min': 0.078, 'q05': 0.141, 'q25': 0.235, 'q50': 0.354, 'q75': 0.605, 'q95': 1.036099999999998, 'max': 2.329}, 'Age': {'mean': 33.314, 'std': 12.047547634269804, 'min': 21.0, 'q05': 21.0, 'q25': 24.0, 'q50': 29.0, 'q75': 41.0, 'q95': 58.049999999999955, 'max': 81.0}, 'Outcome': {'values': [0, 1], 'probs': [0.651394422310757, 0.34860557768924305]}}
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
