import importlib.util
import math
from pathlib import Path
import sys


def _load_bank_module():
    path = (
        Path(__file__).resolve().parents[1]
        / "evaluation"
        / "perturbation"
        / "bank.py"
    )
    name = "_test_frozen_perturbation_quality_bank"
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


BANK = _load_bank_module()


CANONICAL = (
    "Move the phone from the living room to the bedroom. "
    "Then place it next to the bed."
)
NATURAL_REWRITES = (
    "Take the phone from the living room into the bedroom, "
    "then put it next to the bed.",
    "After moving the phone from the living room into the bedroom, "
    "position it next to the bed.",
)


def test_v2_tokenizer_does_not_reward_punctuation_padding():
    padded = f"(((((( {CANONICAL} ))))))"

    assert BANK.token_levenshtein_distance(CANONICAL, padded) == 0
    quality = BANK.assess_rewrite_quality(
        CANONICAL,
        (padded, NATURAL_REWRITES[1]),
    )

    assert not quality["valid"]
    assert any("punctuation_padding" in value for value in quality["errors"])


def test_quality_contract_rejects_meta_language_and_modifier_stuffing():
    rewrites = (
        NATURAL_REWRITES[0] + " This wording preserves the requested task.",
        NATURAL_REWRITES[1]
        + " Carefully safely thoroughly methodically complete the request.",
    )

    quality = BANK.assess_rewrite_quality(CANONICAL, rewrites)

    assert not quality["valid"]
    assert any("meta_language" in value for value in quality["errors"])
    assert any("modifier_stuffing" in value for value in quality["errors"])


def test_quality_contract_accepts_natural_distinct_rewrites():
    quality = BANK.assess_rewrite_quality(CANONICAL, NATURAL_REWRITES)

    assert quality["valid"]
    assert quality["errors"] == []


def test_variant_edit_path_signature_is_lexical_and_deterministic():
    first = BANK.rewrite_edit_signature(CANONICAL, NATURAL_REWRITES[0])
    repeated = BANK.rewrite_edit_signature(CANONICAL, NATURAL_REWRITES[0])
    different = BANK.rewrite_edit_signature(CANONICAL, NATURAL_REWRITES[1])

    assert first == repeated
    assert first != different


def test_integer_quantized_dose_allows_small_non_semantic_fluctuation():
    budgets = (12, 12, 13, 16, 16)
    doses = BANK.cumulative_text_doses(budgets)
    maximum_error = max(
        abs(actual - target)
        for actual, target in zip(doses, BANK.STRESS_GRID)
    )
    ratio = max(budgets) / min(budgets)

    assert math.isclose(maximum_error, 0.06376811594202891)
    assert maximum_error <= BANK.DEFAULT_TEXT_DOSE_TOLERANCE
    assert ratio <= BANK.DEFAULT_UNIT_BUDGET_RATIO_LIMIT


def test_ratio_is_diagnostic_when_direct_cumulative_dose_is_aligned():
    budgets = (19, 20, 13, 16, 16)
    doses = BANK.cumulative_text_doses(budgets)
    maximum_error = max(
        abs(actual - target)
        for actual, target in zip(doses, BANK.STRESS_GRID)
    )

    assert max(budgets) / min(budgets) > BANK.DEFAULT_UNIT_BUDGET_RATIO_LIMIT
    assert maximum_error <= BANK.DEFAULT_TEXT_DOSE_TOLERANCE
