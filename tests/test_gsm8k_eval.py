from eval.gsm8k_eval import (
    extract_gsm8k_prediction,
    format_gsm8k_prompt,
    grade_gsm8k_response,
    normalize_gsm8k_answer,
)


def test_normalize_gsm8k_answer_matches_lm_eval_regexes():
    assert normalize_gsm8k_answer("Janet sells eggs.\n#### 18") == "18"
    assert normalize_gsm8k_answer("$70,000.") == "70000"


def test_extract_gsm8k_prediction_strict_then_flexible():
    assert extract_gsm8k_prediction("work\n#### 18") == "18"
    flexible = extract_gsm8k_prediction("Therefore the total is 42.")
    assert flexible is not None
    assert normalize_gsm8k_answer(flexible) == "42"


def test_grade_gsm8k_response_uses_full_gold_answer():
    gold = "She makes 9 * 2 = $<<9*2=18>>18 every day.\n#### 18"
    assert grade_gsm8k_response(gold, "reasoning\n#### 18") is True
    assert grade_gsm8k_response(gold, "reasoning\n#### 19") is False


def test_format_gsm8k_prompt_includes_five_shot_prefix():
    prompt = format_gsm8k_prompt("How many apples?")
    assert prompt.count("Question:") == 6
    assert prompt.endswith("Question: How many apples?\nAnswer:")
