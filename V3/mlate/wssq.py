"""
Weighted Synergistic Scaffold Quality (WSSQ)
============================================

The single score the optimiser maximises, combining Printability and Cell
Response. This module is the canonical implementation: the web application,
the Bayesian optimiser and the sensitivity analysis in `07_wssq/` all import
from here, so there is exactly one definition of the metric in the project.

The arithmetic is transcribed unchanged from the deployed V3 application
(`app/legacy/wssq.py`). Nothing here alters a published score; what is added is
the documentation of *why* the form is what it is, which the referees asked for,
and the constants named so the sensitivity analysis can vary them.

The construction
----------------
Both targets are first mapped onto [0, 1]:

    norm_p = p / 3          Printability 0-3
    norm_c = (c - 1) / 4    Cell Response 1-5

They are then combined by two weighted means, with weights wp and wc summing
to one:

    HWM  = 1 / (wp/norm_p + wc/norm_c)        weighted harmonic mean
    WMC  = norm_p**wp * norm_c**wc            weighted geometric mean
    WSSQ = 100 * (blend * HWM + (1 - blend) * WMC)

Why two conjunctive means rather than one arithmetic mean
---------------------------------------------------------
An arithmetic mean is *compensatory*: excellent printability offsets dead
cells, and a scaffold that fails one objective outright can still score well.
That is the wrong behaviour for a screening score whose purpose is to surface
candidates worth attempting in a laboratory.

Both means used here are *conjunctive* - each goes to zero when either
component goes to zero, so neither objective can be traded away entirely. They
differ in how hard they punish imbalance, and the weighted mean inequality

    HWM  <=  WMC  <=  arithmetic mean

orders them: the harmonic mean is the more severe of the two, and both are
bounded above by the arithmetic mean the metric deliberately avoids.

`blend` therefore is not an arbitrary mixing constant - it selects severity
inside a bounded family. blend = 1 is the most conservative scoring the family
permits, blend = 0 the most permissive, and because WSSQ is linear in `blend`
the whole family lies between the two endpoints. The shipped default of 0.5
sits at the midpoint. `07_wssq/sensitivity.py` reports how much the ranking of
candidate formulations actually moves across that range.

Two boundary rules
------------------
`p == 0` scores zero. An ink that does not extrude is not a scaffold, and no
biological performance redeems it.

`c <= 1` falls back to printability alone, scaled to [0, 100]. Cell Response 1
means *no cells were cultured*, not *cells did badly*; scoring an acellular
scaffold as though it had failed a biological test would be wrong, and the
platform is explicitly intended to serve acellular 3D-printed scaffolds as well
as bioprinted ones. Note the consequence, quantified in the sensitivity
analysis: for any candidate on this branch the weights cancel out entirely, so
its score is weight-invariant by construction.
"""

from __future__ import annotations

import numpy as np

# Shipped defaults - but the three are not on the same footing, and the
# distinction is what the two referee comments turn on.
#
#   print_weight / cell_weight   USER-CONTROLLED. The application exposes cell
#       weight as a sidebar slider (0-100% in steps of 5, default 70), with
#       print weight taking the remainder. 0.3/0.7 is where the slider starts,
#       not a fixed modelling choice, so the sensitivity analysis over these is
#       a map of the tool across its own input range rather than a defence of
#       one setting.
#
#   blend                        FIXED. Not exposed anywhere in the interface;
#       every user of the platform scores at 0.5. It is therefore the only
#       constant in the metric that has to be justified on its own, which is
#       precisely what referee 2 asked about equation 6.
DEFAULT_PRINT_WEIGHT = 0.3
DEFAULT_CELL_WEIGHT = 0.7
DEFAULT_BLEND = 0.5

# The slider's actual positions, so the analysis samples what users can
# really select rather than a continuum they cannot reach.
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
    """Canonical MLATE V3 WSSQ implementation shared by the deployed app."""
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
    """
    The two means before blending, on the 0-100 scale, plus the arithmetic
    mean the metric deliberately does not use.

    Exposed so the sensitivity analysis can verify the HWM <= WMC <= AM
    ordering on the actual data rather than asserting it from the inequality.
    Boundary rules are NOT applied here: this returns the raw aggregators.
    """
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
    """
    Candidates whose score cannot move when the weights or the blend change.

    Both boundary rules bypass the weighted means entirely: `p == 0` returns a
    constant zero, and `c <= 1` returns a function of printability alone. For
    those rows every weighting produces the same number, so they contribute no
    information to a weight sensitivity analysis and would dilute any rank
    correlation computed over the whole corpus if left unflagged.
    """
    p = np.asarray(printability, dtype=float)
    c = np.asarray(cell_response, dtype=float)
    return (p == 0) | (c <= 1)
