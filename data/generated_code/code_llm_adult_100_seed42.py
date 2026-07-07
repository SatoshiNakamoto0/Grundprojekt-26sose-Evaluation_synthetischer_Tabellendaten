import numpy as np
import pandas as pd
import math
import random

COLUMNS = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 'race', 'sex', 'hours-per-week', 'income']
COLUMN_STATS = {'numeric': {'age': {'mean': 37.33, 'std': 12.50764166419873, 'min': 17.0, 'q05': 19.95, 'q25': 27.75, 'q50': 36.5, 'q75': 44.25, 'q95': 62.0, 'max': 79.0}, 'education-num': {'mean': 10.11, 'std': 2.58416330753302, 'min': 3.0, 'q05': 5.0, 'q25': 9.0, 'q50': 10.0, 'q75': 13.0, 'q95': 14.0, 'max': 15.0}, 'hours-per-week': {'mean': 41.39, 'std': 10.170442468250828, 'min': 12.0, 'q05': 23.8, 'q25': 40.0, 'q50': 40.0, 'q75': 44.25, 'q95': 60.0, 'max': 70.0}}, 'categorical': {'workclass': {'values': ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'], 'probs': [0.04672897196261682, 0.07476635514018691, 0.719626168224299, 0.037383177570093455, 0.04672897196261682, 0.06542056074766354, 0.009345794392523364]}, 'education': {'values': ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'], 'probs': [0.034482758620689655, 0.02586206896551724, 0.008620689655172414, 0.008620689655172414, 0.017241379310344827, 0.034482758620689655, 0.034482758620689655, 0.034482758620689655, 0.04310344827586207, 0.19827586206896552, 0.008620689655172414, 0.31896551724137934, 0.0603448275862069, 0.008620689655172414, 0.017241379310344827, 0.14655172413793102]}, 'marital-status': {'values': ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], 'probs': [0.12149532710280374, 0.009345794392523364, 0.38317757009345793, 0.028037383177570093, 0.38317757009345793, 0.04672897196261682, 0.028037383177570093]}, 'occupation': {'values': ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'], 'probs': [0.15789473684210525, 0.008771929824561403, 0.11403508771929824, 0.08771929824561403, 0.02631578947368421, 0.06140350877192982, 0.12280701754385964, 0.08771929824561403, 0.008771929824561403, 0.09649122807017543, 0.02631578947368421, 0.07017543859649122, 0.043859649122807015, 0.08771929824561403]}, 'race': {'values': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'], 'probs': [0.009523809523809525, 0.02857142857142857, 0.08571428571428572, 0.01904761904761905, 0.8571428571428571]}, 'sex': {'values': ['Female', 'Male'], 'probs': [0.3137254901960784, 0.6862745098039216]}, 'income': {'values': ['<=50K', '>50K'], 'probs': [0.7450980392156863, 0.2549019607843137]}}, 'target_probs': {'values': ['<=50K', '>50K'], 'probs': [0.7450980392156863, 0.2549019607843137]}}
NUMERIC_STATS = COLUMN_STATS.get('numeric', {})
CATEGORICAL_STATS = COLUMN_STATS.get('categorical', {})
TARGET_STATS = COLUMN_STATS.get('target_probs', {})
FLAT_COLUMN_STATS = {'age': {'mean': 37.33, 'std': 12.50764166419873, 'min': 17.0, 'q05': 19.95, 'q25': 27.75, 'q50': 36.5, 'q75': 44.25, 'q95': 62.0, 'max': 79.0}, 'education-num': {'mean': 10.11, 'std': 2.58416330753302, 'min': 3.0, 'q05': 5.0, 'q25': 9.0, 'q50': 10.0, 'q75': 13.0, 'q95': 14.0, 'max': 15.0}, 'hours-per-week': {'mean': 41.39, 'std': 10.170442468250828, 'min': 12.0, 'q05': 23.8, 'q25': 40.0, 'q50': 40.0, 'q75': 44.25, 'q95': 60.0, 'max': 70.0}, 'workclass': {'values': ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'], 'probs': [0.04672897196261682, 0.07476635514018691, 0.719626168224299, 0.037383177570093455, 0.04672897196261682, 0.06542056074766354, 0.009345794392523364]}, 'education': {'values': ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'], 'probs': [0.034482758620689655, 0.02586206896551724, 0.008620689655172414, 0.008620689655172414, 0.017241379310344827, 0.034482758620689655, 0.034482758620689655, 0.034482758620689655, 0.04310344827586207, 0.19827586206896552, 0.008620689655172414, 0.31896551724137934, 0.0603448275862069, 0.008620689655172414, 0.017241379310344827, 0.14655172413793102]}, 'marital-status': {'values': ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], 'probs': [0.12149532710280374, 0.009345794392523364, 0.38317757009345793, 0.028037383177570093, 0.38317757009345793, 0.04672897196261682, 0.028037383177570093]}, 'occupation': {'values': ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'], 'probs': [0.15789473684210525, 0.008771929824561403, 0.11403508771929824, 0.08771929824561403, 0.02631578947368421, 0.06140350877192982, 0.12280701754385964, 0.08771929824561403, 0.008771929824561403, 0.09649122807017543, 0.02631578947368421, 0.07017543859649122, 0.043859649122807015, 0.08771929824561403]}, 'race': {'values': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'], 'probs': [0.009523809523809525, 0.02857142857142857, 0.08571428571428572, 0.01904761904761905, 0.8571428571428571]}, 'sex': {'values': ['Female', 'Male'], 'probs': [0.3137254901960784, 0.6862745098039216]}, 'income': {'values': ['<=50K', '>50K'], 'probs': [0.7450980392156863, 0.2549019607843137]}}
ALLOWED_VALUES = {'workclass': ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'], 'education': ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'], 'marital-status': ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], 'occupation': ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'], 'race': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'], 'sex': ['Female', 'Male'], 'income': ['<=50K', '>50K']}
EDUCATION_TO_NUM = {'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16}
CODE_LLM_METADATA = {'dataset': 'adult', 'seed': 42, 'scaffold_version': 'Code-LLM-v2', 'generation_success': True, 'compile_success': True, 'runtime_success': True, 'validation_success': True, 'fallback_used': False, 'failure_stage': '', 'failure_reason': '', 'llm_attempts': 1}

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
        education = sample_categorical(rng, CATEGORICAL_STATS["education"])
        rows.append({
            "age": sample_numeric_quantile_jitter(rng, NUMERIC_STATS["age"], integer=True),
            "workclass": sample_categorical(rng, CATEGORICAL_STATS["workclass"]),
            "education": education,
            "education-num": EDUCATION_TO_NUM.get(education, 9),
            "marital-status": sample_categorical(rng, CATEGORICAL_STATS["marital-status"]),
            "occupation": sample_categorical(rng, CATEGORICAL_STATS["occupation"]),
            "race": sample_categorical(rng, CATEGORICAL_STATS["race"]),
            "sex": sample_categorical(rng, CATEGORICAL_STATS["sex"]),
            "hours-per-week": sample_numeric_quantile_jitter(rng, NUMERIC_STATS["hours-per-week"], integer=True),
            "income": sample_categorical(rng, TARGET_STATS),
        })
    return pd.DataFrame(rows[:n], columns=COLUMNS)
