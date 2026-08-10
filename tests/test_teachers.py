"""The pinned teacher registry: rights and reproducibility as executable facts."""

import json

import pytest

from hermes.router.manifest import RIGHTS_APPROVED, RIGHTS_DENIED
from hermes.teachers import (
    CLAUDE_FABLE_5,
    DEEPSEEK_V4_PRO,
    KIMI_K3,
    PIN_OPEN_WEIGHTS,
    QWEN_38_MAX,
    REGISTRY,
    TEACHER_FIELD_V1,
    Teacher,
    TeacherError,
    audit,
    check_field,
    get,
    kimi_k3_self_hosted,
    token_teachers,
    trainable,
)

# --- rights are a recorded decision, not a judgement this module makes ----------------


def test_the_field_is_qwen_and_deepseek_and_both_may_be_trained_on():
    assert [t.teacher_id for t in TEACHER_FIELD_V1] == ["qwen3.8-max", "deepseek-v4-pro"]
    assert [t.teacher_id for t in trainable()] == ["qwen3.8-max", "deepseek-v4-pro"]


def test_kimi_is_declared_but_out_of_the_field():
    """The gateway does not serve it. Kept as the record of what was evaluated."""
    assert KIMI_K3 not in TEACHER_FIELD_V1
    assert "kimi-k3" in REGISTRY


def test_an_approval_must_record_who_made_it():
    """An approval nobody signed is indistinguishable from an approval nobody made."""
    assert QWEN_38_MAX.rights_basis and KIMI_K3.rights_basis
    with pytest.raises(TeacherError, match="approval nobody signed"):
        Teacher(
            teacher_id="x",
            model="m",
            base_url="u",
            usage_shape="openai",
            pin="none",
            training_rights=RIGHTS_APPROVED,
        )


def test_fable_5_is_declared_but_outside_the_field():
    """Kept rather than deleted: the declaration is the record of why it is absent."""
    assert CLAUDE_FABLE_5.training_rights == RIGHTS_DENIED
    assert not CLAUDE_FABLE_5.may_train
    assert CLAUDE_FABLE_5 not in TEACHER_FIELD_V1


def test_adding_fable_5_back_stays_safe():
    """build_artifacts keeps a rights-denied teacher as router evidence, never as a row."""
    extended = (*TEACHER_FIELD_V1, CLAUDE_FABLE_5)
    assert CLAUDE_FABLE_5.teacher_id not in [t.teacher_id for t in trainable(extended)]
    assert audit(extended)["rights_denied"] == ["claude-fable-5"]


# --- rights and reproducibility are separate axes ---------------------------------------


def test_neither_hosted_teacher_in_the_field_can_be_pinned():
    """Both ids can shift under a corpus that was already generated."""
    assert not QWEN_38_MAX.reproducible and not DEEPSEEK_V4_PRO.reproducible
    assert audit()["fully_reproducible"] is False


def test_the_field_yields_training_data_but_is_not_reproducible():
    """The user settled rights; pinning is a different question and still open."""
    report = audit()
    assert report["yields_training_data"] is True
    assert report["not_reproducible"] == ["qwen3.8-max", "deepseek-v4-pro"]


def test_check_field_says_so_on_every_run_rather_than_once_in_a_docstring():
    problems = check_field(TEACHER_FIELD_V1)
    assert len(problems) == 1
    assert "cannot be pinned" in problems[0]


def test_fable_5_is_pinnable_yet_untrainable_which_is_why_the_axes_are_separate():
    assert CLAUDE_FABLE_5.reproducible and not CLAUDE_FABLE_5.may_train


# --- token distillation ------------------------------------------------------------------


def test_the_field_has_exactly_one_logprob_teacher():
    """This test previously asserted that neither returned logprobs, on a measurement that
    re-probing falsified. Qwen returns a content-aligned stream; DeepSeek was saturated (429)
    at every attempt on 2026-08-09 and stays False until it answers, because assuming
    capability is how the first wrong entry got written."""
    assert QWEN_38_MAX.logprobs
    assert not DEEPSEEK_V4_PRO.logprobs
    assert [t.teacher_id for t in token_teachers()] == ["qwen3.8-max"]


def test_token_distillation_needs_logprobs_and_rights_together():
    assert not CLAUDE_FABLE_5.may_distil_tokens  # pinnable, no logprobs, no rights
    assert not DEEPSEEK_V4_PRO.may_distil_tokens  # rights, no logprobs
    assert QWEN_38_MAX.may_distil_tokens  # rights and content-aligned logprobs


def test_a_logprob_teacher_can_still_be_unreproducible():
    """The two axes moved apart rather than together: qwen can now supply token-level
    targets and still cannot be regenerated from its identity, so a logit corpus from it is
    usable and unreplayable at the same time."""
    assert QWEN_38_MAX.may_distil_tokens
    assert not QWEN_38_MAX.reproducible


# --- the one reproducible configuration ----------------------------------------------------


def test_the_self_hosted_teacher_is_trainable_reproducible_and_token_capable():
    pinned = kimi_k3_self_hosted(revision="d34db33f")
    assert pinned.may_train and pinned.reproducible and pinned.may_distil_tokens


def test_it_cannot_be_had_without_choosing_a_pin():
    """A factory rather than a constant: shipping one would mean inventing a SHA."""
    with pytest.raises(TeacherError, match="needs the exact weights revision"):
        kimi_k3_self_hosted(revision="")


def test_an_open_weights_pin_without_a_revision_is_refused():
    """A repo name alone moves whenever the publisher pushes."""
    with pytest.raises(TeacherError, match="a repo name alone moves"):
        Teacher(
            teacher_id="x",
            model="m",
            base_url="u",
            usage_shape="openai",
            pin=PIN_OPEN_WEIGHTS,
            training_rights=RIGHTS_APPROVED,
            rights_basis="checked",
            weights_repo="org/repo",
        )


def test_a_pinned_field_has_no_problems_at_all():
    field_ = (kimi_k3_self_hosted(revision="abc123"), CLAUDE_FABLE_5)
    assert check_field(field_) == []


# --- malformed declarations -------------------------------------------------------------------


def test_an_unknown_pin_kind_is_refused():
    with pytest.raises(TeacherError, match="unknown pin kind"):
        Teacher(
            teacher_id="x", model="m", base_url="u", usage_shape="openai", pin="vibes", training_rights=RIGHTS_DENIED
        )


def test_an_unknown_usage_shape_is_refused():
    """The shape decides how cached tokens are read; guessing it mis-bills silently."""
    with pytest.raises(TeacherError, match="unknown usage shape"):
        Teacher(teacher_id="x", model="m", base_url="u", usage_shape="guess", pin="none", training_rights=RIGHTS_DENIED)


def test_a_single_teacher_is_not_a_tournament():
    assert any("not a comparison" in p for p in check_field((CLAUDE_FABLE_5,)))


def test_lookup_of_an_undeclared_teacher_is_refused():
    with pytest.raises(TeacherError, match="unknown teacher"):
        get("gpt-9")


def test_records_carry_the_rights_basis():
    record = json.loads(json.dumps([t.to_record() for t in TEACHER_FIELD_V1]))
    assert all(r["may_train"] for r in record)
    assert all(r["rights_basis"] for r in record)


# --- rights attach to an artefact, not to a model ---------------------------------------


def test_the_hosted_kimi_endpoint_may_not_be_trained_on():
    """It was RIGHTS_APPROVED on a review of the open-weights licence -- which governs the
    weights and not this endpoint. The hosted API is covered by Moonshot's service terms,
    and those bar building models that could compete with the service."""
    from hermes.teachers import get

    hosted = get("kimi-k3")
    assert hosted.base_url.startswith("https://api.moonshot.ai")
    assert not hosted.may_train


def test_the_self_hosted_kimi_weights_may_be_trained_on():
    from hermes.teachers import kimi_k3_self_hosted

    pinned = kimi_k3_self_hosted(revision="a" * 40)
    assert pinned.may_train
    assert pinned.reproducible


def test_the_two_kimi_entries_are_the_same_model_under_different_rights():
    """The exact confusion that produced the bug, now asserted so it cannot come back
    silently: same model string, two artefacts, two agreements, opposite answers."""
    from hermes.teachers import get, kimi_k3_self_hosted

    hosted = get("kimi-k3")
    pinned = kimi_k3_self_hosted(revision="a" * 40)
    assert hosted.model == pinned.model
    assert hosted.training_rights != pinned.training_rights


def test_a_rights_conflict_is_reported_rather_than_hidden():
    from hermes.teachers import REGISTRY, kimi_k3_self_hosted, rights_conflicts

    notes = rights_conflicts([*REGISTRY.values(), kimi_k3_self_hosted(revision="a" * 40)])
    assert any("kimi-k3" in note for note in notes)


def test_a_field_with_one_set_of_rights_per_model_reports_no_conflict():
    from hermes.teachers import TEACHER_FIELD_V1, rights_conflicts

    assert rights_conflicts(TEACHER_FIELD_V1) == []


def test_every_approved_teacher_records_what_was_reviewed():
    """`rights_basis` exists because 'approved' with no provenance is indistinguishable from
    'nobody checked'. A basis that does not say which document was read is the same failure
    one step later -- it is what let a weights licence stand in for an API's terms."""
    from hermes.teachers import REGISTRY

    for teacher in REGISTRY.values():
        if teacher.may_train:
            assert len(teacher.rights_basis) > 20, teacher.teacher_id


# --- logprobs are measured per endpoint, and a boolean is not enough ------------------------


def test_qwen_is_a_logit_teacher_measured_not_assumed():
    """Re-probed against the live gateway 2026-08-09: 9 logprob tokens for a 9-token answer,
    top_logprobs populated, and the stream reconstructs `content` exactly. The docs say the
    `-max` series has no logprobs and the previous entry recorded an empty object; both were
    wrong, so this is measurement pinned against the next person who reads the docs."""
    from hermes.teachers import LOGPROB_CONTENT, get

    qwen = get("qwen3.8-max")
    assert qwen.logprobs
    assert qwen.logprob_scope == LOGPROB_CONTENT
    assert qwen.may_distil_tokens


def test_the_hosted_kimi_endpoint_returns_no_logprobs():
    """Absent, not empty -- and the opposite of what its documentation claims."""
    from hermes.teachers import LOGPROB_NONE, get

    kimi = get("kimi-k3")
    assert not kimi.logprobs
    assert kimi.logprob_scope == LOGPROB_NONE


def test_declaring_logprobs_without_a_scope_is_refused():
    """The dangerous case must not be the one you get by saying nothing."""
    from hermes.teachers import PIN_NONE, Teacher, TeacherError

    with pytest.raises(TeacherError, match="declares logprobs with no scope"):
        Teacher(
            teacher_id="t",
            model="m",
            base_url="https://example.invalid/v1",
            usage_shape="openai",
            pin=PIN_NONE,
            training_rights="denied",
            logprobs=True,
        )


def test_a_scope_without_logprobs_is_refused():
    from hermes.teachers import LOGPROB_CONTENT, PIN_NONE, Teacher, TeacherError

    with pytest.raises(TeacherError, match="scope but no logprobs"):
        Teacher(
            teacher_id="t",
            model="m",
            base_url="https://example.invalid/v1",
            usage_shape="openai",
            pin=PIN_NONE,
            training_rights="denied",
            logprobs=False,
            logprob_scope=LOGPROB_CONTENT,
        )


def test_reasoning_aligned_logprobs_are_not_distillable():
    """kimi-k2.7-code returned 58 logprob tokens for a nine-token answer, reconstructing its
    reasoning rather than its content. That is a real distribution over real tokens -- just
    not the text the student is being taught to emit."""
    from hermes.teachers import LOGPROB_FULL_STREAM, PIN_NONE, Teacher

    reasoner = Teacher(
        teacher_id="reasoner",
        model="reasoner",
        base_url="https://example.invalid/v1",
        usage_shape="openai",
        pin=PIN_NONE,
        training_rights="approved",
        rights_basis="test fixture, not a real approval",
        logprobs=True,
        logprob_scope=LOGPROB_FULL_STREAM,
    )
    assert reasoner.logprobs
    assert not reasoner.may_distil_tokens


def test_a_field_mixing_logprob_scopes_is_flagged():
    """Both are well-formed logprobs.content arrays, so nothing downstream would fail."""
    from hermes.teachers import LOGPROB_FULL_STREAM, PIN_NONE, Teacher, check_field, get

    reasoner = Teacher(
        teacher_id="reasoner",
        model="reasoner",
        base_url="https://example.invalid/v1",
        usage_shape="openai",
        pin=PIN_NONE,
        training_rights="approved",
        rights_basis="test fixture, not a real approval",
        logprobs=True,
        logprob_scope=LOGPROB_FULL_STREAM,
    )
    problems = check_field((get("qwen3.8-max"), reasoner))
    assert any("different token streams" in p for p in problems)


def test_the_declared_field_now_has_a_token_distillation_teacher():
    """The measurement changed a conclusion: STATUS.md said self-hosting was the only path
    to logprobs. It is still the only path to *reproducible* ones."""
    from hermes.teachers import TEACHER_FIELD_V1, audit

    assert audit(TEACHER_FIELD_V1)["token_distillation"] == ["qwen3.8-max"]


def test_the_field_excludes_kimi_for_rights_and_logprobs_not_availability():
    """The gateway does serve `kimi-k3` -- that was measured, after it was dropped on the
    belief that it did not. It stays out on the two reasons that survived probing."""
    from hermes.teachers import TEACHER_FIELD_V1, get

    kimi = get("kimi-k3")
    assert kimi.teacher_id not in {t.teacher_id for t in TEACHER_FIELD_V1}
    assert not kimi.may_train
    assert not kimi.logprobs
