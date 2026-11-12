"""Transformation déterministe commune à l'apprentissage et à l'API films."""
import numpy as np
import pandas as pd

RAW_FEATURES = ['release_date', 'original_language', 'overview']
NUMERIC = ['release_year', 'release_month', 'overview_words', 'overview_chars']
CATEGORICAL = ['original_language']

def movie_features(frame):
    frame = pd.DataFrame(frame).copy()
    dates = pd.to_datetime(frame['release_date'], errors='coerce')
    text = frame['overview'].fillna('').astype(str)
    return pd.DataFrame({
        'release_year': dates.dt.year.astype(float),
        'release_month': dates.dt.month.astype(float),
        'overview_words': np.log1p(text.str.split().str.len()),
        'overview_chars': np.log1p(text.str.len()),
        'original_language': frame['original_language'].fillna('unknown').astype(str),
        'overview': text,
    }, index=frame.index)
