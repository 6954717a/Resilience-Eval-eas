import ast
import importlib.util
from pathlib import Path
import sys

import pytest


def _load_tabular_module():
    module_path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "reporting"
        / "tabular.py"
    )
    module_name = "habitat_llm.evaluation.reporting.tabular_test_target"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _load_episode_rows_module():
    package_root = Path(__file__).resolve().parents[1]
    package_paths = {
        "habitat_llm": package_root,
        "habitat_llm.evaluation": package_root / "evaluation",
        "habitat_llm.evaluation.reporting": package_root / "evaluation" / "reporting",
    }
    for name, path in package_paths.items():
        if name not in sys.modules:
            package = type(sys)(name)
            package.__path__ = [str(path)]
            sys.modules[name] = package
    module_name = "habitat_llm.evaluation.reporting.episode_rows_test_target"
    module_path = package_root / "evaluation" / "reporting" / "episode_rows.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


_TABULAR = _load_tabular_module()
_EPISODE_ROWS = _load_episode_rows_module()
DuplicateColumnError = _TABULAR.DuplicateColumnError
TabularSchemaError = _TABULAR.TabularSchemaError
append_csv_rows = _TABULAR.append_csv_rows
read_csv_rows = _TABULAR.read_csv_rows
write_csv_rows_exact = _TABULAR.write_csv_rows_exact


def test_exact_write_is_atomic_when_value_serialization_fails(tmp_path: Path):
    output_path = tmp_path / "evidence.csv"
    write_csv_rows_exact(output_path, [{"metric": "old"}])
    original_bytes = output_path.read_bytes()

    class ExplodingValue:
        def __str__(self) -> str:
            raise RuntimeError("cannot serialize")

    with pytest.raises(RuntimeError, match="cannot serialize"):
        write_csv_rows_exact(output_path, [{"metric": ExplodingValue()}])

    assert output_path.read_bytes() == original_bytes
    assert not list(tmp_path.glob(f".{output_path.name}.*.tmp"))


def test_schema_aware_append_expands_without_dropping_existing_fields(
    tmp_path: Path,
):
    output_path = tmp_path / "evidence.csv"
    write_csv_rows_exact(output_path, [{"episode_id": "1", "score": 0.5}])

    append_csv_rows(
        output_path,
        [{"episode_id": "2", "diagnostic": "new"}],
        schema_policy="expand",
    )

    assert output_path.read_text(encoding="utf-8").splitlines()[0] == (
        "episode_id,score,diagnostic"
    )
    assert read_csv_rows(output_path) == [
        {"episode_id": "1", "score": "0.5", "diagnostic": ""},
        {"episode_id": "2", "score": "", "diagnostic": "new"},
    ]


def test_exact_append_rejects_schema_change_without_touching_file(tmp_path: Path):
    output_path = tmp_path / "evidence.csv"
    write_csv_rows_exact(output_path, [{"episode_id": "1"}])
    original_bytes = output_path.read_bytes()

    with pytest.raises(TabularSchemaError, match="absent from the declared schema"):
        append_csv_rows(
            output_path,
            [{"episode_id": "2", "new_metric": 1.0}],
            schema_policy="exact",
        )

    assert output_path.read_bytes() == original_bytes


def test_empty_exact_write_requires_explicit_stale_artifact_policy(tmp_path: Path):
    output_path = tmp_path / "evidence.csv"
    write_csv_rows_exact(output_path, [{"episode_id": "old"}])
    original_bytes = output_path.read_bytes()

    with pytest.raises(ValueError, match="empty CSV rows"):
        write_csv_rows_exact(output_path, [])
    assert output_path.read_bytes() == original_bytes

    write_csv_rows_exact(
        output_path,
        [],
        fieldnames=["episode_id", "score"],
        empty_policy="write_header",
    )
    assert output_path.read_text(encoding="utf-8").splitlines() == [
        "episode_id,score"
    ]

    write_csv_rows_exact(output_path, [], empty_policy="remove")
    assert not output_path.exists()


def test_overwrite_error_policy_preserves_existing_artifact(tmp_path: Path):
    output_path = tmp_path / "evidence.csv"
    write_csv_rows_exact(output_path, [{"episode_id": "old"}])

    with pytest.raises(FileExistsError):
        write_csv_rows_exact(
            output_path,
            [{"episode_id": "new"}],
            overwrite_policy="error",
        )

    assert read_csv_rows(output_path) == [{"episode_id": "old"}]


def test_duplicate_columns_are_rejected_in_declared_rows_and_existing_csv(
    tmp_path: Path,
):
    output_path = tmp_path / "evidence.csv"

    with pytest.raises(DuplicateColumnError, match="Duplicate CSV column"):
        write_csv_rows_exact(
            output_path,
            [{"episode_id": "1"}],
            fieldnames=["episode_id", "episode_id"],
        )

    with pytest.raises(DuplicateColumnError, match="duplicate normalized key"):
        write_csv_rows_exact(output_path, [{1: "numeric", "1": "string"}])

    output_path.write_text("episode_id,episode_id\n1,2\n", encoding="utf-8")
    with pytest.raises(DuplicateColumnError, match="Duplicate CSV column"):
        read_csv_rows(output_path)


def test_declared_schema_never_silently_discards_row_columns(tmp_path: Path):
    output_path = tmp_path / "evidence.csv"
    with pytest.raises(TabularSchemaError, match="absent from the declared schema"):
        write_csv_rows_exact(
            output_path,
            [{"episode_id": "1", "diagnostic": "retain-me"}],
            fieldnames=["episode_id"],
        )
    assert not output_path.exists()


def test_explicit_empty_schema_is_not_treated_as_schema_inference(tmp_path: Path):
    output_path = tmp_path / "evidence.csv"
    with pytest.raises(TabularSchemaError, match="absent from the declared schema"):
        write_csv_rows_exact(
            output_path,
            [{"episode_id": "1"}],
            fieldnames=[],
        )
    assert not output_path.exists()


def test_expand_append_honors_seed_schema_and_new_row_fields(tmp_path: Path):
    output_path = tmp_path / "evidence.csv"
    append_csv_rows(
        output_path,
        [{"episode_id": "1", "score": 0.5}],
        fieldnames=["episode_id"],
        schema_policy="expand",
    )
    assert output_path.read_text(encoding="utf-8").splitlines()[0] == (
        "episode_id,score"
    )


def test_main_mp_csv_helpers_delegate_to_atomic_tabular_layer():
    source_path = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "planner_demo_mp_new.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    functions = {
        node.name: node
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }

    expected_calls = {
        "write_to_csv": "append_csv_rows",
        "_write_rows_exact": "write_csv_rows_exact",
        "_read_csv_rows": "read_csv_rows",
    }
    for function_name, delegated_name in expected_calls.items():
        calls = {
            node.func.id
            for node in ast.walk(functions[function_name])
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        }
        assert delegated_name in calls
        assert "open" not in calls


@pytest.mark.parametrize(
    "relative_path",
    [
        "examples/planner_demo_mp_new.py",
        "examples/planner_demo_resilience.py",
        "examples/resilience_helpers.py",
        "examples/expq_helpers.py",
    ],
)
def test_resilience_orchestration_has_no_direct_csv_reader_or_writer(
    relative_path: str,
):
    source_path = Path(__file__).resolve().parents[1] / relative_path
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    direct_csv_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "csv"
        and node.func.attr in {"DictReader", "DictWriter"}
    ]
    assert direct_csv_calls == []


def test_episode_row_builder_preserves_flat_perturbation_audit_only():
    info = {
        "task_percent_complete": 0.5,
        "replanning_count": {0: 2},
        "perturbation_audit": {"prompt": {"realized": 0.5}},
        "perturbation_requested": 0.4,
        "perturbation_realized": 0.5,
        "perturbation_valid": True,
        "perturbation_components": '{"operation_count":1}',
    }

    keys = _EPISODE_ROWS.collect_episode_stats_keys(info)
    assert "replanning_count_0" in keys
    assert "perturbation_components" in keys
    assert "perturbation_audit" not in keys

    row = _EPISODE_ROWS.build_episode_csv_metrics(
        {"task_percent_complete": 0.5},
        info,
    )
    assert row["perturbation_valid"] is True
    assert row["perturbation_components"] == '{"operation_count":1}'
    assert "perturbation_audit" not in row


def test_episode_row_builder_materializes_canonical_proposition_fraction():
    info = {
        "auto_eval_proposition_tracker": {
            "proposition_satisfied_at": [0, -1, 3],
        }
    }

    keys = _EPISODE_ROWS.collect_episode_stats_keys(info)
    row = _EPISODE_ROWS.build_episode_csv_metrics({}, info)

    assert "proposition_satisfied_fraction" in keys
    assert info["proposition_satisfied_count"] == 2
    assert info["proposition_total"] == 3
    assert row["proposition_satisfied_fraction"] == pytest.approx(2.0 / 3.0)


def test_episode_row_builder_preserves_canonical_fraction_after_tracker_trim():
    info = {
        "auto_eval_proposition_tracker": {
            "proposition_satisfied_at": [0, -1],
        }
    }
    _EPISODE_ROWS.collect_episode_stats_keys(info)
    del info["auto_eval_proposition_tracker"]

    row = _EPISODE_ROWS.build_episode_csv_metrics({}, info)

    assert row["proposition_satisfied_fraction"] == pytest.approx(0.5)
    assert row["proposition_satisfied_fraction_source"] == (
        "auto_eval_proposition_tracker"
    )
