"""The Hermes wire format is upstream. A change to it is a defect, however good."""

import pytest

from hermes.conformance import DIALECTS, drift, pinned, render, template_path


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_the_rendered_system_prompt_matches_its_pin(dialect):
    """The whole point: editing hermes/protocol.py now requires editing a checked-in
    artifact in the same commit, which puts the diff in front of a reviewer instead of
    leaving it implicit in a green suite."""
    assert drift(dialect) == "", drift(dialect)


@pytest.mark.parametrize("dialect", sorted(DIALECTS))
def test_every_dialect_has_a_pin(dialect):
    assert template_path(dialect).is_file()


def test_drift_names_the_line_that_changed(monkeypatch):
    """A reviewer told 'the Hermes 4 system prompt changed' has to diff two strings by
    eye. Naming the line is the difference between a useful failure and an annoying one."""
    import hermes.conformance as conformance

    original = conformance.render
    monkeypatch.setattr(conformance, "render", lambda d: original(d).replace("You are Hermes", "You are Sparky", 1))
    message = conformance.drift("hermes-4")
    assert "drifted at line 1" in message
    assert "You are Sparky" in message


def test_a_missing_pin_is_reported_as_missing(monkeypatch, tmp_path):
    import hermes.conformance as conformance

    monkeypatch.setattr(conformance, "TEMPLATE_DIR", tmp_path)
    assert "no pinned template" in conformance.drift("hermes-4")


def test_the_official_preamble_is_present_verbatim():
    """Hermes 4's identity line is not ours to reword. If this ever needs changing, the
    change belongs upstream at Nous, not here."""
    assert pinned("hermes-4").startswith("You are Hermes, created by Nous Research.")


@pytest.mark.parametrize("dialect", ["hermes-3", "hermes-4"])
def test_each_dialect_pins_its_own_reasoning_tag(dialect):
    """Hermes 4 dropped <scratch_pad> entirely; it is not even a token any more. Emitting
    the wrong one trains a tag the tokenizer splits into pieces."""
    text = pinned(dialect)
    assert ("<scratch_pad>" in text) == (dialect == "hermes-3")


@pytest.mark.parametrize("dialect", ["hermes-3", "hermes-4"])
def test_the_pin_advertises_the_reference_tools(dialect):
    """A pin over an empty tool set would fix almost nothing about the envelope.

    Hermes only. A dialect whose definitions come from the serving template has no rendered prompt
    to advertise anything in -- the tools go with the request, so the reference set appears nowhere
    in the pinned artifact. `test_the_atem_pin_is_the_models_own_template` is the check for those.
    """
    text = pinned(dialect)
    assert "terminal" in text and "file_write" in text
    assert "<tools>" in text and "</tools>" in text
    assert "<tool_call>" in text


@pytest.mark.parametrize("dialect", sorted(d for d in DIALECTS if not DIALECTS[d].tools_in_prompt))
def test_the_pin_is_the_models_own_template_not_a_rendered_prompt(dialect):
    """What conditions the model here is upstream's file, and this repo writes no tool block at
    all. So the pinned artifact is that file, and the check is that every marker the parser
    depends on is still in it -- a renamed tag otherwise makes the parser return zero calls on
    every turn, which reads as a model that never calls tools rather than as a broken parser."""
    from hermes.conformance import ATEM_MARKERS, template_path

    assert template_path(dialect).suffix == ".jinja"
    text = pinned(dialect)
    assert "{%-" in text, "a jinja template, not a rendered prompt"
    for marker in ATEM_MARKERS:
        assert marker in text, marker


def test_the_marker_check_would_catch_a_renamed_tag(monkeypatch, tmp_path):
    """A guard nobody has seen fail. Renaming one tag in a copy of the pinned template must be
    reported, naming the marker."""
    import hermes.conformance as conformance

    original = conformance.pinned
    monkeypatch.setattr(
        conformance, "pinned", lambda d: original(d).replace('<atem:invoke name="', '<atem:call name="')
    )
    message = conformance.drift("atem")
    assert "no longer contains" in message and "atem:invoke" in message


def test_update_is_idempotent():
    """Running --update on an unchanged tree must not rewrite the files, or the pin
    becomes noise in every diff and stops being read."""
    from hermes.conformance import update

    assert update() == []
    assert all(drift(name) == "" for name in DIALECTS)


def test_render_is_deterministic():
    assert render("hermes-4") == render("hermes-4")
