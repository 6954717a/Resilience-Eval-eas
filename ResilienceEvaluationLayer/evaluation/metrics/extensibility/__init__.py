"""
Graceful Extensibility metrics package.

This package implements the post-processed GE pipeline:

* episode contract evidence
* family/stress/evolve cells
* discrete stress-grid capacity summaries

`BoundarySweep` remains a declarative helper for constructing stress grids,
while `MarginCollector` performs the actual evidence -> cell -> capacity
aggregation on merged episode rows.
"""

from .boundary_sweep import BoundarySweep
from .contract_margin import ContractMargin, MarginThresholds
from .margin_collector import MarginCollector

__all__ = [
    "ContractMargin",
    "MarginThresholds",
    "MarginCollector",
    "BoundarySweep",
]
