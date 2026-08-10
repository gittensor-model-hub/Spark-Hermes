"""Assistant text whose only source in the row is the system prompt."""

import json

from hermes.leakage import leakage, main


def _row(system: str, user: str, assistant: str, tool: str = "") -> dict:
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
    if tool:
        messages.append({"role": "tool", "tool_call_id": "c1", "content": tool})
    messages.append({"role": "assistant", "content": assistant})
    return {"messages": messages}


def test_a_sentinel_the_prompt_asked_for_and_the_user_did_not_is_leakage():
    """The measured case: 78 of 133 rows in the canonical export emit
    SPARKPROOF_TRITON_PASS because the system prompt says to, not because anyone asked."""
    report = leakage(
        [
            _row(
                'Always end with print("SPARKPROOF_TRITON_PASS") after tests pass.',
                "Write a kernel.",
                'print("SPARKPROOF_TRITON_PASS")',
            )
        ]
    )
    assert report.rate == 1.0
    assert "sparkproof_triton_pass" in report.affected[0].phrases


def test_the_same_sentinel_is_not_leakage_when_the_user_asked_for_it():
    """If the user asked, removing the system prompt changes nothing about this row.
    That is what makes this provenance rather than coincidence."""
    report = leakage(
        [
            _row(
                'Always end with print("SPARKPROOF_TRITON_PASS").',
                "Write a kernel and print SPARKPROOF_TRITON_PASS at the end.",
                'print("SPARKPROOF_TRITON_PASS")',
            )
        ]
    )
    assert report.affected == ()


def test_text_the_assistant_read_from_a_tool_is_not_leakage():
    """An agent echoing a filename it observed is not reciting the prompt."""
    report = leakage(
        [_row("Work on BUILD_TARGET files.", "Fix it.", "Patched BUILD_TARGET.", tool="BUILD_TARGET missing")]
    )
    assert report.affected == ()


def test_a_single_distinctive_token_counts_even_without_a_shared_phrase():
    """The three-word floor alone found 5.3% of the canonical export; the real figure is
    62%, because the leaked text is one token inside print(...) and never part of a
    three-word run shared with the prompt."""
    report = leakage([_row("Target Blackwell SM12x.", "Write it.", "Tuned for SM12x throughput.")])
    assert "sm12x" in report.affected[0].phrases


def test_ordinary_english_is_not_reported():
    """'the', 'you should', 'in the file' appear in every prompt and every answer;
    reporting them would bury the real findings."""
    report = leakage([_row("You should solve the task.", "Do the thing.", "I solved the task.")])
    assert report.affected == ()


def test_a_row_with_no_system_turn_is_skipped_not_counted_clean():
    report = leakage([{"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]}])
    assert report.rows == 1
    assert report.system_prompts == 0
    assert report.affected == ()


def test_tool_call_names_and_arguments_are_searched():
    """A tool name copied out of the prompt leaks exactly as a sentinel does."""
    rows = [
        {
            "messages": [
                {"role": "system", "content": "Use the deploy_canary tool."},
                {"role": "user", "content": "Ship it."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{"function": {"name": "deploy_canary", "arguments": "{}"}}],
                },
            ]
        }
    ]
    assert leakage(rows).rate == 1.0


def test_longer_findings_absorb_their_own_prefixes():
    """Reporting a phrase and its four sub-windows is the same finding five times."""
    report = leakage(
        [
            _row(
                "Always end with print SPARKPROOF_TRITON_PASS after tests pass.",
                "Go.",
                "always end with print SPARKPROOF_TRITON_PASS after tests pass",
            )
        ]
    )
    phrases = report.affected[0].phrases
    assert not any(a != b and a in b for a in phrases for b in phrases)


def test_the_report_states_what_it_cannot_see():
    """A clean report is not a clean corpus: behaviour caused by the prompt that left no
    textual trace is the majority of the risk and no string comparison reaches it."""
    record = leakage([_row("Be careful.", "Go.", "Done.")]).to_record()
    assert "does_not_measure" in record
    assert record["rows_affected"] == 0


def test_the_cli_reports_and_can_gate(tmp_path, capsys):
    source = tmp_path / "corpus.jsonl"
    source.write_text(
        json.dumps(_row("Always print SPARKPROOF_TRITON_PASS.", "Write it.", 'print("SPARKPROOF_TRITON_PASS")')) + "\n",
        encoding="utf-8",
    )
    out = tmp_path / "report.json"
    assert main(["--in", str(source), "--out", str(out)]) == 0
    assert json.loads(out.read_text())["rate"] == 1.0
    # Gating is opt-in: a number nobody has looked at yet should not start breaking
    # pipelines on the day it lands.
    assert main(["--in", str(source), "--max-rate", "0.5"]) == 1


def test_a_stripped_corpus_reports_unmeasurable_rather_than_clean():
    """The trap this exists to close. Strip the prompts, measure, and every row comes out
    clean at rate 0.0 -- because there is nothing left to trace text back to. The leakage
    did not go away, it went invisible, and it is now baked into rows that no longer say
    what caused them."""
    stripped = [
        {"messages": [{"role": "user", "content": "Go."}, {"role": "assistant", "content": "SPARKPROOF_TRITON_PASS"}]}
    ]
    report = leakage(stripped)
    assert report.rate == 0.0
    assert report.measurable is False
    assert report.to_record()["measurable"] is False


def test_a_genuinely_clean_corpus_is_measurable():
    report = leakage([_row("Be careful.", "Go.", "Done.")])
    assert report.rate == 0.0
    assert report.measurable is True


def test_the_cli_fails_on_an_unmeasurable_corpus(tmp_path):
    source = tmp_path / "corpus.jsonl"
    source.write_text(
        json.dumps({"messages": [{"role": "user", "content": "Go."}, {"role": "assistant", "content": "done"}]}) + "\n",
        encoding="utf-8",
    )
    assert main(["--in", str(source)]) == 1
