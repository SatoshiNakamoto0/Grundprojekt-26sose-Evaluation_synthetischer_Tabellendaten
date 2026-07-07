import numpy as np
import pandas as pd
import math
import random

COLUMNS = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 'race', 'sex', 'hours-per-week', 'income']
COLUMN_STATS = {'numeric': {'age': {'mean': 38.562, 'std': 13.721448757328798, 'min': 17.0, 'q05': 19.950000000000003, 'q25': 28.0, 'q50': 37.0, 'q75': 47.0, 'q95': 63.049999999999955, 'max': 90.0}, 'education-num': {'mean': 10.178, 'std': 2.5965970037724375, 'min': 1.0, 'q05': 6.0, 'q25': 9.0, 'q50': 10.0, 'q75': 13.0, 'q95': 14.0, 'max': 16.0}, 'hours-per-week': {'mean': 41.688, 'std': 12.82492323563771, 'min': 5.0, 'q05': 20.0, 'q25': 40.0, 'q50': 40.0, 'q75': 45.0, 'q95': 60.0, 'max': 99.0}}, 'categorical': {'workclass': {'values': ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'], 'probs': [0.03944773175542406, 0.0631163708086785, 0.7337278106508875, 0.03550295857988166, 0.08875739644970414, 0.03747534516765286, 0.0019723865877712033]}, 'education': {'values': ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'], 'probs': [0.029069767441860465, 0.031007751937984496, 0.009689922480620155, 0.003875968992248062, 0.015503875968992248, 0.01937984496124031, 0.013565891472868217, 0.027131782945736434, 0.040697674418604654, 0.16279069767441862, 0.015503875968992248, 0.3430232558139535, 0.06589147286821706, 0.003875968992248062, 0.023255813953488372, 0.19573643410852712]}, 'marital-status': {'values': ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], 'probs': [0.15384615384615385, 0.0039447731755424065, 0.45759368836291914, 0.01775147928994083, 0.3214990138067061, 0.021696252465483234, 0.023668639053254437]}, 'occupation': {'values': ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'], 'probs': [0.10311284046692606, 0.0019455252918287938, 0.122568093385214, 0.13618677042801555, 0.04085603112840467, 0.0622568093385214, 0.08365758754863813, 0.0933852140077821, 0.005836575875486381, 0.11089494163424124, 0.02529182879377432, 0.11867704280155641, 0.033073929961089495, 0.0622568093385214]}, 'race': {'values': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'], 'probs': [0.007920792079207921, 0.02574257425742574, 0.07722772277227723, 0.009900990099009901, 0.8792079207920792]}, 'sex': {'values': ['Female', 'Male'], 'probs': [0.3346613545816733, 0.6653386454183267]}, 'income': {'values': ['<=50K', '>50K'], 'probs': [0.750996015936255, 0.24900398406374502]}}, 'target_probs': {'values': ['<=50K', '>50K'], 'probs': [0.750996015936255, 0.24900398406374502]}}
NUMERIC_STATS = COLUMN_STATS.get('numeric', {})
CATEGORICAL_STATS = COLUMN_STATS.get('categorical', {})
TARGET_STATS = COLUMN_STATS.get('target_probs', {})
FLAT_COLUMN_STATS = {'age': {'mean': 38.562, 'std': 13.721448757328798, 'min': 17.0, 'q05': 19.950000000000003, 'q25': 28.0, 'q50': 37.0, 'q75': 47.0, 'q95': 63.049999999999955, 'max': 90.0}, 'education-num': {'mean': 10.178, 'std': 2.5965970037724375, 'min': 1.0, 'q05': 6.0, 'q25': 9.0, 'q50': 10.0, 'q75': 13.0, 'q95': 14.0, 'max': 16.0}, 'hours-per-week': {'mean': 41.688, 'std': 12.82492323563771, 'min': 5.0, 'q05': 20.0, 'q25': 40.0, 'q50': 40.0, 'q75': 45.0, 'q95': 60.0, 'max': 99.0}, 'workclass': {'values': ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'], 'probs': [0.03944773175542406, 0.0631163708086785, 0.7337278106508875, 0.03550295857988166, 0.08875739644970414, 0.03747534516765286, 0.0019723865877712033]}, 'education': {'values': ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'], 'probs': [0.029069767441860465, 0.031007751937984496, 0.009689922480620155, 0.003875968992248062, 0.015503875968992248, 0.01937984496124031, 0.013565891472868217, 0.027131782945736434, 0.040697674418604654, 0.16279069767441862, 0.015503875968992248, 0.3430232558139535, 0.06589147286821706, 0.003875968992248062, 0.023255813953488372, 0.19573643410852712]}, 'marital-status': {'values': ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], 'probs': [0.15384615384615385, 0.0039447731755424065, 0.45759368836291914, 0.01775147928994083, 0.3214990138067061, 0.021696252465483234, 0.023668639053254437]}, 'occupation': {'values': ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'], 'probs': [0.10311284046692606, 0.0019455252918287938, 0.122568093385214, 0.13618677042801555, 0.04085603112840467, 0.0622568093385214, 0.08365758754863813, 0.0933852140077821, 0.005836575875486381, 0.11089494163424124, 0.02529182879377432, 0.11867704280155641, 0.033073929961089495, 0.0622568093385214]}, 'race': {'values': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'], 'probs': [0.007920792079207921, 0.02574257425742574, 0.07722772277227723, 0.009900990099009901, 0.8792079207920792]}, 'sex': {'values': ['Female', 'Male'], 'probs': [0.3346613545816733, 0.6653386454183267]}, 'income': {'values': ['<=50K', '>50K'], 'probs': [0.750996015936255, 0.24900398406374502]}}
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
