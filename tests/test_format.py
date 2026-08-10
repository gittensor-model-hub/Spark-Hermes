from teacher.format import to_sft_record


def test_to_sft_record_wraps_reasoning_in_think_tags():
    trajectory = {
        "prompt": "What is 2 + 2?",
        "response": "4",
        "reasoning": "  2 + 2 is basic addition, giving 4.  ",
        "system": "You are helpful.",
    }

    record = to_sft_record(trajectory)

    assert record["prompt"] == "What is 2 + 2?"
    assert record["response"] == "<think>\n2 + 2 is basic addition, giving 4.\n</think>\n\n4"
    assert record["system"] == "You are helpful."


def test_to_sft_record_falls_back_to_response_without_reasoning():
    trajectory = {"prompt": "What is 2 + 2?", "response": "4"}

    record = to_sft_record(trajectory)

    assert record["response"] == "4"
    assert record["system"] is None


def test_to_sft_record_falls_back_when_reasoning_is_empty_string():
    trajectory = {"prompt": "What is 2 + 2?", "response": "4", "reasoning": ""}

    record = to_sft_record(trajectory)

    assert record["response"] == "4"


def test_to_sft_record_falls_back_when_reasoning_is_whitespace_only():
    trajectory = {"prompt": "What is 2 + 2?", "response": "4", "reasoning": "   \n  "}

    record = to_sft_record(trajectory)

    assert record["response"] == "4"


def test_to_messages_record_falls_back_when_reasoning_is_whitespace_only():
    from teacher.format import to_messages_record

    trajectory = {"prompt": "What is 2 + 2?", "response": "4", "reasoning": "  \t "}

    record = to_messages_record(trajectory)

    assert record["messages"][-1]["content"] == "4"
    assert "<think>" not in record["messages"][-1]["content"]


def test_to_messages_record_drops_encrypted_reasoning_dump():
    import json

    from teacher.format import to_messages_record

    encrypted = json.dumps([{"type": "reasoning.encrypted", "data": "b64blob=="}])
    trajectory = {"prompt": "What is 2 + 2?", "response": "4", "reasoning": encrypted}

    record = to_messages_record(trajectory)

    assert record["messages"][-1]["content"] == "4"
    assert "<think>" not in record["messages"][-1]["content"]


def test_to_messages_record_uses_qwen3_chat_roles():
    from teacher.format import to_messages_record

    trajectory = {
        "prompt": "Write a Triton softmax kernel",
        "response": "def launch(...): ...",
        "reasoning": "Use row-wise max for stability.",
        "system": "You are a kernel expert.",
    }
    record = to_messages_record(trajectory)
    assert record["messages"][0] == {"role": "system", "content": "You are a kernel expert."}
    assert record["messages"][1]["role"] == "user"
    assert record["messages"][2]["role"] == "assistant"
    assert "<think>" in record["messages"][2]["content"]


def test_assistant_content_coerces_null_or_absent_response():
    # A null/absent response must not crash SFT conversion / mix aggregation (issue #212).
    from teacher.format import to_messages_record

    assert to_messages_record({"prompt": "p", "response": None})["messages"][-1]["content"] == ""
    assert to_messages_record({"prompt": "p"})["messages"][-1]["content"] == ""
