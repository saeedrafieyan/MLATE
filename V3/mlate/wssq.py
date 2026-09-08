from __future__ import annotations

import numpy as np

DEFAULT_PRINT_WEIGHT = 0.3
DEFAULT_CELL_WEIGHT = 0.7
DEFAULT_BLEND = 0.5

SLIDER_CELL_WEIGHTS = tuple(w / 100 for w in range(0, 101, 5))

PRINTABILITY_RANGE = (0.0, 3.0)
CELL_RESPONSE_RANGE = (1.0, 5.0)


def compute_wssq(
    printability,
    cell_response,
    print_weight: float = DEFAULT_PRINT_WEIGHT,
    cell_weight: float = DEFAULT_CELL_WEIGHT,
    blend: float = DEFAULT_BLEND,
):
    p = np.asarray(printability, dtype=float)
    c = np.asarray(cell_response, dtype=float)
    if np.any(~np.isfinite(p)) or np.any(~np.isfinite(c)):
        raise ValueError("WSSQ inputs must be finite")
    if np.any((p < 0) | (p > 3)) or np.any((c < 1) | (c > 5)):
        raise ValueError("WSSQ inputs are outside their label domains")
    if print_weight < 0 or cell_weight < 0 or print_weight + cell_weight <= 0:
        raise ValueError("WSSQ weights must be non-negative and not both zero")
    if not 0 <= blend <= 1:
        raise ValueError("blend must be within [0, 1]")
    total = print_weight + cell_weight
    wp, wc = print_weight / total, cell_weight / total
    norm_p = p / 3.0
    norm_c = (c - 1.0) / 4.0
    epsilon = np.finfo(float).eps
    harmonic = 1.0 / (wp / np.maximum(norm_p, epsilon)
                      + wc / np.maximum(norm_c, epsilon))
    multiplicative = (np.maximum(norm_p, epsilon) ** wp
                      * np.maximum(norm_c, epsilon) ** wc)
    score = 100.0 * (blend * harmonic + (1.0 - blend) * multiplicative)
    score = np.where(p == 0, 0.0, np.where(c <= 1, 100.0 * norm_p, score))
    score = np.clip(score, 0.0, 100.0)
    return float(score) if score.ndim == 0 else score


def components(printability, cell_response,
               print_weight: float = DEFAULT_PRINT_WEIGHT,
               cell_weight: float = DEFAULT_CELL_WEIGHT):
    p = np.asarray(printability, dtype=float)
    c = np.asarray(cell_response, dtype=float)
    total = print_weight + cell_weight
    wp, wc = print_weight / total, cell_weight / total
    norm_p, norm_c = p / 3.0, (c - 1.0) / 4.0
    eps = np.finfo(float).eps
    np_, nc_ = np.maximum(norm_p, eps), np.maximum(norm_c, eps)
    return {
        "harmonic": 100.0 / (wp / np_ + wc / nc_),
        "multiplicative": 100.0 * np_ ** wp * nc_ ** wc,
        "arithmetic": 100.0 * (wp * np_ + wc * nc_),
    }


def is_weight_invariant(printability, cell_response) -> np.ndarray:
    p = np.asarray(printability, dtype=float)
    c = np.asarray(cell_response, dtype=float)
    return (p == 0) | (c <= 1)
