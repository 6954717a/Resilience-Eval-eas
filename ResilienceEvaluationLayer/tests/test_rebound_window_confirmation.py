from pathlib import Path
import sys
import types


_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if "habitat_llm" not in sys.modules:
    root_package = types.ModuleType("habitat_llm")
    root_package.__path__ = [str(_PACKAGE_ROOT)]
    sys.modules["habitat_llm"] = root_package
if "habitat_llm.evaluation" not in sys.modules:
    evaluation_package = types.ModuleType("habitat_llm.evaluation")
    evaluation_package.__path__ = [str(_PACKAGE_ROOT / "evaluation")]
    sys.modules["habitat_llm.evaluation"] = evaluation_package
if "habitat_llm.evaluation.metrics" not in sys.modules:
    metrics_package = types.ModuleType("habitat_llm.evaluation.metrics")
    metrics_package.__path__ = [str(_PACKAGE_ROOT / "evaluation" / "metrics")]
    sys.modules["habitat_llm.evaluation.metrics"] = metrics_package


from habitat_llm.evaluation.metrics.rebound.online_tracker import (
    OnlineReboundTracker,
)
from habitat_llm.evaluation.metrics.rebound.rebound_templates import ReboundEvent
from habitat_llm.evaluation.stage_baseline.stage_anchor import StageAnchor


def _event(step: int, kind: str, template: str, cause: str) -> ReboundEvent:
    return ReboundEvent(
        step=step,
        kind=kind,
        template=template,
        payload={"cause_id": cause},
    )


def test_heuristic_recovery_requires_persistent_confirmation() -> None:
    tracker = OnlineReboundTracker(
        templates=[], recovery_confirmation_steps=2
    )
    anchor = StageAnchor(family="placement_only", stage=0, modality="planning")

    tracker._apply_event(
        _event(1, "t_d", "R3", "R3"),
        anchor,
        0.0,
        observed_step=1,
    )
    candidate = _event(4, "t_r", "R3", "R3")
    tracker._apply_event(candidate, anchor, 0.5, observed_step=4)

    assert tracker._open_window is not None
    assert candidate.payload["recovery_pending_confirmation"] is True

    confirmed = tracker._advance_pending_recoveries(
        step=5, anchor=anchor, curr_q=0.5, stable=True
    )

    assert len(confirmed) == 1
    assert tracker._open_window is None
    window = tracker.closed_windows[0]
    assert window.t_r == 5
    assert window.cause_recovery_evidence_steps["R3"] == 4


def test_proposition_grounded_recovery_is_immediate() -> None:
    tracker = OnlineReboundTracker(
        templates=[], recovery_confirmation_steps=2
    )
    anchor = StageAnchor(family="placement_only", stage=0, modality="planning")

    tracker._apply_event(
        _event(1, "t_d", "T1", "T1:cup"),
        anchor,
        0.0,
        observed_step=1,
    )
    tracker._apply_event(
        _event(2, "t_r", "T1", "T1:cup"),
        anchor,
        1.0,
        observed_step=2,
    )

    assert tracker._open_window is None
    window = tracker.closed_windows[0]
    assert window.t_r == 2
    assert [step for step, *_rest in window.step_trace] == [2]


def test_same_step_immediate_recovery_trace_is_recorded_once() -> None:
    tracker = OnlineReboundTracker(templates=[])
    anchor = StageAnchor(family="placement_only", stage=0, modality="planning")

    tracker._apply_event(
        _event(1, "t_d", "T1", "T1:cup"),
        anchor,
        0.0,
        observed_step=1,
    )
    tracker._apply_event(
        _event(1, "t_r", "T1", "T1:cup"),
        anchor,
        1.0,
        observed_step=1,
    )

    assert tracker._open_window is None
    window = tracker.closed_windows[0]
    assert window.t_d == window.t_r == 1
    assert [step for step, *_rest in window.step_trace] == [1]


def test_terminal_td_anchor_uses_previous_nonterminal_anchor() -> None:
    tracker = OnlineReboundTracker(templates=[])
    nominal = StageAnchor(
        family="placement_only", stage=2, modality="manipulation"
    )
    terminal = StageAnchor(
        family="placement_only", stage=-1, modality="planning"
    )
    tracker._anchors.extend([nominal, terminal])

    resolved, resolution = tracker._anchor_for_step(2, terminal)

    assert resolved == nominal
    assert resolution == "previous_nonterminal"


def test_overlapping_causes_share_window_until_every_cause_recovers() -> None:
    tracker = OnlineReboundTracker(
        templates=[], recovery_confirmation_steps=2
    )
    anchor = StageAnchor(family="placement_only", stage=0, modality="planning")

    tracker._apply_event(
        _event(1, "t_d", "T1", "object_displaced"),
        anchor,
        0.0,
        observed_step=1,
    )
    tracker._apply_event(
        _event(2, "t_d", "T2", "wrong_receptacle"),
        anchor,
        0.0,
        observed_step=2,
    )

    assert tracker._open_window is not None
    assert tracker._open_window.t_d == 1
    assert set(tracker._open_window.active_causes) == {
        "object_displaced",
        "wrong_receptacle",
    }

    tracker._apply_event(
        _event(3, "t_r", "T1", "object_displaced"),
        anchor,
        0.5,
        observed_step=3,
    )
    assert tracker._open_window is not None
    assert set(tracker._open_window.active_causes) == {"wrong_receptacle"}
    assert tracker.closed_windows == []

    tracker._apply_event(
        _event(4, "t_r", "T2", "wrong_receptacle"),
        anchor,
        1.0,
        observed_step=4,
    )
    assert tracker._open_window is None
    assert len(tracker.closed_windows) == 1
    window = tracker.closed_windows[0]
    assert window.t_d == 1
    assert window.t_r == 4
    assert window.cause_recovery_steps == {
        "object_displaced": 3,
        "wrong_receptacle": 4,
    }


def test_backdated_disruption_uses_true_onset_and_backfills_trace() -> None:
    tracker = OnlineReboundTracker(templates=[])
    anchors = [
        StageAnchor(
            family="placement_only",
            stage=stage,
            modality="planning",
        )
        for stage in (0, 1, 1, 2, 2)
    ]
    tracker._anchors.extend(anchors)
    tracker._q_series.extend([0.0, 0.1, 0.2, 0.3, 0.4])

    tracker._apply_event(
        _event(2, "t_d", "R1", "navigation_stall"),
        anchors[-1],
        0.4,
        observed_step=5,
    )

    assert tracker._open_window is not None
    assert tracker._open_window.t_d == 2
    assert tracker._open_window.anchor_t_d == anchors[1]
    assert tracker._open_window.anchor_t_d_resolution == "exact"
    assert [step for step, _anchor, _q in tracker._open_window.stage_trace] == [
        2,
        3,
        4,
        5,
    ]
