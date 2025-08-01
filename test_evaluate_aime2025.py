"""Tests for evaluate_aime2025 helper functions."""
from evaluate_aime2025 import extract_boxed_answer


def test_extract_boxed_answer_basic() -> None:
    """The function should grab the integer between \\boxed{ }."""
    txt = "Some reasoning here. Finally we get \\boxed{123}. End."
    assert extract_boxed_answer(txt) == "123"


def test_extract_boxed_answer_missing() -> None:
    """If no box exists, *None* should be returned."""
    txt = "No final box given."
    assert extract_boxed_answer(txt) is None 