"""Shared source-aware features for pooled research inference."""

import numpy as np
import pandas as pd

from pickplace_ai.features import GYEONGGI_RESEARCH_FEATURES, gyeonggi_four_quarter_features

UNIFIED_RESEARCH_FEATURES = (*GYEONGGI_RESEARCH_FEATURES, "source_gyeonggi")


def unified_research_feature_frame(features: pd.DataFrame, source: str) -> pd.DataFrame:
    """Keep the source domain explicit without converting its codes or sales units."""
    if source not in ("seoul", "gyeonggi"):
        raise ValueError("Unsupported source")
    if list(features.columns) != list(GYEONGGI_RESEARCH_FEATURES):
        raise ValueError("Unexpected research feature schema")
    result = features.copy()
    result["source_gyeonggi"] = float(source == "gyeonggi")
    return result[list(UNIFIED_RESEARCH_FEATURES)]


def unified_research_features(sales: np.ndarray, base_quarter: int, source: str) -> pd.DataFrame:
    """Shared source-aware unit-free features for training and local inference."""
    return unified_research_feature_frame(
        gyeonggi_four_quarter_features(sales, base_quarter), source
    )
