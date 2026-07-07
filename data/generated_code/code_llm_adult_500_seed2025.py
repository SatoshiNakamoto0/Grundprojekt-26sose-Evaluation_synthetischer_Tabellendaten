import numpy as np
import pandas as pd
import math
import random

COLUMNS = ['age', 'workclass', 'education', 'education-num', 'marital-status', 'occupation', 'race', 'sex', 'hours-per-week', 'income']
COLUMN_STATS = {'numeric': {'age': {'mean': 37.682, 'std': 13.386891946975593, 'min': 17.0, 'q05': 19.0, 'q25': 27.0, 'q50': 36.0, 'q75': 47.0, 'q95': 60.0, 'max': 89.0}, 'education-num': {'mean': 10.024, 'std': 2.623628022414763, 'min': 2.0, 'q05': 5.0, 'q25': 9.0, 'q50': 10.0, 'q75': 13.0, 'q95': 14.0, 'max': 16.0}, 'hours-per-week': {'mean': 39.942, 'std': 11.170972920923226, 'min': 8.0, 'q05': 19.900000000000006, 'q25': 40.0, 'q50': 40.0, 'q75': 45.0, 'q95': 60.0, 'max': 88.0}}, 'categorical': {'workclass': {'values': ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'], 'probs': [0.03155818540433925, 0.07297830374753451, 0.7495069033530573, 0.029585798816568046, 0.07100591715976332, 0.04339250493096647, 0.0019723865877712033]}, 'education': {'values': ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'], 'probs': [0.031007751937984496, 0.03294573643410853, 0.011627906976744186, 0.005813953488372093, 0.01744186046511628, 0.023255813953488372, 0.023255813953488372, 0.027131782945736434, 0.03488372093023256, 0.17635658914728683, 0.009689922480620155, 0.3178294573643411, 0.04263565891472868, 0.001937984496124031, 0.027131782945736434, 0.21705426356589147]}, 'marital-status': {'values': ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], 'probs': [0.13609467455621302, 0.0019723865877712033, 0.4418145956607495, 0.011834319526627219, 0.32938856015779094, 0.047337278106508875, 0.03155818540433925]}, 'occupation': {'values': ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'], 'probs': [0.1264591439688716, 0.0019455252918287938, 0.122568093385214, 0.12840466926070038, 0.038910505836575876, 0.04474708171206226, 0.05642023346303502, 0.11478599221789883, 0.007782101167315175, 0.12062256809338522, 0.023346303501945526, 0.1264591439688716, 0.02529182879377432, 0.0622568093385214]}, 'race': {'values': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'], 'probs': [0.007920792079207921, 0.04356435643564356, 0.09900990099009901, 0.007920792079207921, 0.8415841584158416]}, 'sex': {'values': ['Female', 'Male'], 'probs': [0.31673306772908366, 0.6832669322709163]}, 'income': {'values': ['<=50K', '>50K'], 'probs': [0.750996015936255, 0.24900398406374502]}}, 'target_probs': {'values': ['<=50K', '>50K'], 'probs': [0.750996015936255, 0.24900398406374502]}}
NUMERIC_STATS = COLUMN_STATS.get('numeric', {})
CATEGORICAL_STATS = COLUMN_STATS.get('categorical', {})
TARGET_STATS = COLUMN_STATS.get('target_probs', {})
FLAT_COLUMN_STATS = {'age': {'mean': 37.682, 'std': 13.386891946975593, 'min': 17.0, 'q05': 19.0, 'q25': 27.0, 'q50': 36.0, 'q75': 47.0, 'q95': 60.0, 'max': 89.0}, 'education-num': {'mean': 10.024, 'std': 2.623628022414763, 'min': 2.0, 'q05': 5.0, 'q25': 9.0, 'q50': 10.0, 'q75': 13.0, 'q95': 14.0, 'max': 16.0}, 'hours-per-week': {'mean': 39.942, 'std': 11.170972920923226, 'min': 8.0, 'q05': 19.900000000000006, 'q25': 40.0, 'q50': 40.0, 'q75': 45.0, 'q95': 60.0, 'max': 88.0}, 'workclass': {'values': ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'], 'probs': [0.03155818540433925, 0.07297830374753451, 0.7495069033530573, 0.029585798816568046, 0.07100591715976332, 0.04339250493096647, 0.0019723865877712033]}, 'education': {'values': ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'], 'probs': [0.031007751937984496, 0.03294573643410853, 0.011627906976744186, 0.005813953488372093, 0.01744186046511628, 0.023255813953488372, 0.023255813953488372, 0.027131782945736434, 0.03488372093023256, 0.17635658914728683, 0.009689922480620155, 0.3178294573643411, 0.04263565891472868, 0.001937984496124031, 0.027131782945736434, 0.21705426356589147]}, 'marital-status': {'values': ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], 'probs': [0.13609467455621302, 0.0019723865877712033, 0.4418145956607495, 0.011834319526627219, 0.32938856015779094, 0.047337278106508875, 0.03155818540433925]}, 'occupation': {'values': ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'], 'probs': [0.1264591439688716, 0.0019455252918287938, 0.122568093385214, 0.12840466926070038, 0.038910505836575876, 0.04474708171206226, 0.05642023346303502, 0.11478599221789883, 0.007782101167315175, 0.12062256809338522, 0.023346303501945526, 0.1264591439688716, 0.02529182879377432, 0.0622568093385214]}, 'race': {'values': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'], 'probs': [0.007920792079207921, 0.04356435643564356, 0.09900990099009901, 0.007920792079207921, 0.8415841584158416]}, 'sex': {'values': ['Female', 'Male'], 'probs': [0.31673306772908366, 0.6832669322709163]}, 'income': {'values': ['<=50K', '>50K'], 'probs': [0.750996015936255, 0.24900398406374502]}}
ALLOWED_VALUES = {'workclass': ['Federal-gov', 'Local-gov', 'Private', 'Self-emp-inc', 'Self-emp-not-inc', 'State-gov', 'Without-pay'], 'education': ['10th', '11th', '12th', '1st-4th', '5th-6th', '7th-8th', '9th', 'Assoc-acdm', 'Assoc-voc', 'Bachelors', 'Doctorate', 'HS-grad', 'Masters', 'Preschool', 'Prof-school', 'Some-college'], 'marital-status': ['Divorced', 'Married-AF-spouse', 'Married-civ-spouse', 'Married-spouse-absent', 'Never-married', 'Separated', 'Widowed'], 'occupation': ['Adm-clerical', 'Armed-Forces', 'Craft-repair', 'Exec-managerial', 'Farming-fishing', 'Handlers-cleaners', 'Machine-op-inspct', 'Other-service', 'Priv-house-serv', 'Prof-specialty', 'Protective-serv', 'Sales', 'Tech-support', 'Transport-moving'], 'race': ['Amer-Indian-Eskimo', 'Asian-Pac-Islander', 'Black', 'Other', 'White'], 'sex': ['Female', 'Male'], 'income': ['<=50K', '>50K']}
EDUCATION_TO_NUM = {'Preschool': 1, '1st-4th': 2, '5th-6th': 3, '7th-8th': 4, '9th': 5, '10th': 6, '11th': 7, '12th': 8, 'HS-grad': 9, 'Some-college': 10, 'Assoc-voc': 11, 'Assoc-acdm': 12, 'Bachelors': 13, 'Masters': 14, 'Prof-school': 15, 'Doctorate': 16}
CODE_LLM_METADATA = {'dataset': 'adult', 'seed': 2025, 'scaffold_version': 'Code-LLM-v2', 'generation_success': True, 'compile_success': True, 'runtime_success': True, 'validation_success': True, 'fallback_used': False, 'failure_stage': '', 'failure_reason': '', 'llm_attempts': 1}

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
            "age": sample_numeric_quantile_jitter(rng, NUMERIC_STATS["age"], True),
            "workclass": sample_categorical(rng, CATEGORICAL_STATS["workclass"]),
            "education": education,
            "education-num": EDUCATION_TO_NUM.get(education, 9),
            "marital-status": sample_categorical(rng, CATEGORICAL_STATS["marital-status"]),
            "occupation": sample_categorical(rng, CATEGORICAL_STATS["occupation"]),
            "race": sample_categorical(rng, CATEGORICAL_STATS["race"]),
            "sex": sample_categorical(rng, CATEGORICAL_STATS["sex"]),
            "hours-per-week": sample_numeric_quantile_jitter(rng, NUMERIC_STATS["hours-per-week"], True),
            "income": sample_categorical(rng, TARGET_STATS),
        })
    return pd.DataFrame(rows[:n], columns=COLUMNS)
