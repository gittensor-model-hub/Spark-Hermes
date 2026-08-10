"""Per-task salts: opening one spent task must not unseal every task still sealed."""

import pytest

from hermes.harness import HarnessError, derive_task_salt, salted_digest
from hermesbench.verify import redact_for_release

MASTER = "a-master-secret-long-enough"


def test_each_task_gets_a_different_salt():
    salts = {derive_task_salt(MASTER, f"task-{i}") for i in range(50)}
    assert len(salts) == 50


def test_derivation_is_deterministic():
    """Two runs must agree, or a commitment published today fails to verify tomorrow."""
    assert derive_task_salt(MASTER, "t") == derive_task_salt(MASTER, "t")


def test_a_revealed_task_salt_does_not_open_another_task():
    """The property the whole change exists for. A spent task's salt is published so anyone
    can confirm its withheld check was fixed in advance; that disclosure must not turn the
    remaining commitments back into the check-your-guess oracle salting exists to prevent."""
    body_b = "test -f b.txt"
    commitment_b = salted_digest(body_b, derive_task_salt(MASTER, "task-b"))

    revealed = derive_task_salt(MASTER, "task-a")  # task-a is spent, its salt is now public

    assert salted_digest(body_b, revealed) != commitment_b
    # And knowing task-a's salt does not let you derive task-b's, because the master is the key.
    assert revealed != derive_task_salt(MASTER, "task-b")


def test_a_commitment_does_not_verify_under_the_master_itself():
    """Guards the migration: if anything still commits under the bare master, every task
    shares a salt again and the first reveal unseals the corpus."""
    body = "test -f x.txt"
    commitment = salted_digest(body, derive_task_salt(MASTER, "t"))
    assert salted_digest(body, MASTER) != commitment


def test_the_task_id_is_length_prefixed_so_no_two_ids_collide():
    """Plain concatenation would let a crafted pair share a salt: `a` + `bc` and `ab` + `c`
    produce the same bytes. A shared salt between two tasks is the one thing this must not do."""
    assert derive_task_salt(MASTER, "a-bc") != derive_task_salt(MASTER, "ab-c")
    assert derive_task_salt(MASTER, "ab") != derive_task_salt(MASTER, "a") + "b"


def test_a_weak_master_is_refused():
    """Every per-task salt is derived from it, so its strength is every commitment's strength."""
    with pytest.raises(HarnessError, match="at least 16 characters"):
        derive_task_salt("short", "t")


def test_a_missing_task_id_is_refused():
    with pytest.raises(HarnessError, match="without a task id"):
        derive_task_salt(MASTER, "")


# --- the publishing path --------------------------------------------------------------------


def test_redact_for_release_commits_under_the_derived_salt():
    record = {"task_id": "t", "verify": "true", "hidden_verify": "test -f secret.txt"}
    public = redact_for_release(record, salt=MASTER)
    assert "hidden_verify" not in public
    assert public["metadata"]["hidden_verify_commitment"] == salted_digest(
        "test -f secret.txt", derive_task_salt(MASTER, "t")
    )


def test_two_tasks_with_identical_withheld_checks_get_different_commitments():
    """Under a shared salt these would be byte-identical, which leaks that two tasks grade the
    same way -- and lets one opened task confirm a guess about the other."""
    body = "pytest -q"
    a = redact_for_release({"task_id": "a", "hidden_verify": body}, salt=MASTER)
    b = redact_for_release({"task_id": "b", "hidden_verify": body}, salt=MASTER)
    assert a["metadata"]["hidden_verify_commitment"] != b["metadata"]["hidden_verify_commitment"]


def test_publishing_a_commitment_without_a_task_id_is_refused():
    """Falling back to the master would silently restore the shared-salt behaviour for that
    task, which is the failure mode this change removes."""
    with pytest.raises(ValueError, match="per-task salt is derived from it"):
        redact_for_release({"hidden_verify": "true"}, salt=MASTER)


def test_a_task_with_no_withheld_check_publishes_no_commitment():
    public = redact_for_release({"task_id": "t", "verify": "true"}, salt=MASTER)
    assert "hidden_verify_commitment" not in public["metadata"]
    assert public["metadata"]["has_hidden_tests"] is False


# --- the shipped corpus ---------------------------------------------------------------------


def test_every_shipped_commitment_is_a_distinct_value():
    """Cheap standing check that the corpus is not sharing salts. It cannot verify the
    commitments themselves -- that needs the withheld bodies, which no public clone holds --
    but identical commitments across tasks would be visible proof of a shared salt."""
    from hermesbench.tasks import load_suite

    commitments = [t.hidden_verify_commitment for t in load_suite("all") if t.hidden_verify_commitment]
    assert len(commitments) == 19
    assert len(set(commitments)) == len(commitments)
