"""Terminology regression checks for the retired Margin/Contract API.

The behavioral suite lives in ``test_boundary_collector.py``.  Keeping this
small filename-level guard prevents old Contract/Margin modules from being
accidentally reintroduced while avoiding a second copy of the same tests.
"""

from pathlib import Path


def test_boundary_is_the_only_extensibility_implementation_name():
    module_dir = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "metrics"
        / "extensibility"
    )

    assert (module_dir / "boundary_collector.py").is_file()
    assert (module_dir / "boundary_margin.py").is_file()
    assert not (module_dir / "margin_collector.py").exists()
    assert not (module_dir / "contract_margin.py").exists()
