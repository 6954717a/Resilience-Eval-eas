"""Small, auditable primitives for paper Rebound Eq. (2)."""

from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np

from .recovery_features import REBOUND_FORMULA_VERSION


def regularized_mahalanobis(
    values: Sequence[float],
    covariance: Sequence[Sequence[float]],
    *,
    sample_count: int,
    regularizer: float,
    mean: Optional[Sequence[float]] = None,
    one_sided: bool = True,
    normalize_dimension: bool = True,
) -> Optional[float]:
    r"""Return the baseline-centred channel norm used by Rebound Eq. (2).

    Existing callers that omit ``mean`` remain source-compatible and use a
    zero centre.  Formal runtime calls pass ``StageBaseline.recovery_mean``.
    The one-sided delta measures excess burden only, while division by the
    vector dimension keeps otherwise identical channels comparable.

    The ridge is relative to the channel covariance scale:

    .. math::

        \Sigma_{reg} = \Sigma + \lambda
        \max(\operatorname{tr}(\Sigma)/k, 1)I.

    A machine-scale floor keeps a zero-rank covariance finite even when a
    diagnostic caller explicitly sets ``regularizer=0``.
    """

    vector = np.asarray(values, dtype=np.float64)
    matrix = np.asarray(covariance, dtype=np.float64)
    if vector.ndim != 1 or vector.size == 0:
        return None
    if sample_count < 1 or matrix.shape != (vector.size, vector.size):
        return None
    if not np.all(np.isfinite(vector)) or not np.all(np.isfinite(matrix)):
        return None

    if mean is None:
        centre = np.zeros(vector.size, dtype=np.float64)
    else:
        centre = np.asarray(mean, dtype=np.float64)
        if centre.shape != vector.shape or not np.all(np.isfinite(centre)):
            return None

    delta = vector - centre
    if one_sided:
        delta = np.maximum(delta, 0.0)

    # Covariances should be symmetric, but averaging avoids numerical noise
    # from serialized/merged StageBaselines changing the pseudo-inverse.
    matrix = 0.5 * (matrix + matrix.T)
    channel_scale = max(float(np.trace(matrix)) / float(vector.size), 1.0)
    ridge = max(float(regularizer), 1.0e-12) * channel_scale
    stabilized = matrix + ridge * np.eye(vector.size, dtype=np.float64)
    squared = float(delta.T @ np.linalg.pinv(stabilized) @ delta)
    if not math.isfinite(squared):
        return None
    if normalize_dimension:
        squared /= float(vector.size)
    return float(math.sqrt(max(squared, 0.0)))


def recovery_intensity(channel_magnitudes: Sequence[float]) -> float:
    """Combine aligned channel norms by Eq. (2)'s root mean square."""

    values = [float(value) for value in channel_magnitudes]
    if not values:
        return 0.0
    return float(math.sqrt(sum(value * value for value in values) / len(values)))


def bounded_recovery_cost_index(c_rec: Optional[float]) -> Optional[float]:
    """Map a finite non-negative raw recovery cost to ``[0, 100)``.

    This is a presentation/statistics index only.  The paper-facing raw
    ``C_rec`` remains unchanged and continues to drive GE and inference.
    """

    if c_rec is None:
        return None
    value = float(c_rec)
    if not math.isfinite(value) or value < 0.0:
        return None
    if value == 0.0:
        return 0.0
    # This form is stable for very large finite costs.  Guard against float
    # rounding reaching exactly 100 so the documented half-open range holds.
    index = 100.0 - 100.0 / (1.0 + value)
    return min(index, math.nextafter(100.0, 0.0))


__all__ = [
    "REBOUND_FORMULA_VERSION",
    "bounded_recovery_cost_index",
    "recovery_intensity",
    "regularized_mahalanobis",
]
