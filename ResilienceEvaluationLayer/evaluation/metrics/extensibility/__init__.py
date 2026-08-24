"""
Graceful Extensibility metrics package.

This package implements the post-processed GE pipeline:

* episode boundary evidence
* family/stress/evolve cells
* discrete stress-grid capacity summaries

`BoundarySweep` remains a declarative helper for constructing stress grids,
while `BoundaryCollector` performs the actual evidence -> cell -> capacity
aggregation on merged episode rows.
"""

from .boundary_sweep import BoundarySweep
from .boundary_margin import BoundaryMargin, BoundaryThresholds
from .boundary_collector import BoundaryCollector

__all__ = [
    "BoundaryMargin",
    "BoundaryThresholds",
    "BoundaryCollector",
    "BoundarySweep",
]
