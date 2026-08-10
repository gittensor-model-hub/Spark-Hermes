"""Task supply: mutate source projects into a suite, and split it so it can be trusted."""

import json
import sys

import pytest
import yaml

from hermesbench.supply import (
    DEV,
    SEALED_EVAL,
    SPLITS,
    TRAIN,
    SourceProject,
    SupplyError,
    build,
    lineage_digest,
    load_manifest,
    problems,
    split_for,
    write_suite,
)

SOURCE = """\
def clamp(value, low, high):
    if value < low:
        return low
    if value > high:
        return high
    return value


def total(items):
    running = 0
    for item in items:
        running = running + item
    return running
"""

TESTS = """\
from calc import clamp, total


def test_clamp():
    assert clamp(5, 0, 10) == 5
    assert clamp(-1, 0, 10) == 0
    assert clamp(99, 0, 10) == 10


def test_total():
    assert total([1, 2, 3]) == 6
    assert total([]) == 0
"""


def _project(tmp_path, name="calcs"):
    directory = tmp_path / name
    directory.mkdir(parents=True)
    (directory / "calc.py").write_text(SOURCE, encoding="utf-8")
    (directory / "test_calc.py").write_text(TESTS, encoding="utf-8")
    return SourceProject(
        name=name,
        project_dir=directory,
        target_files=("calc.py",),
        # sys.executable rather than a bare `python`: this box has an unrelated vendored
        # venv on PATH, and a verifier resolving to it measures a different tree.
        verify=f"{sys.executable} -m pytest -q",
        timeout_s=60,
    )


# --- lineage is content-addressed ------------------------------------------------------------


def test_the_same_source_has_one_lineage_wherever_it_lives():
    """Path-addressed lineage would call a vendored copy a different family and let the
    split straddle it -- the same leak wearing a different filename."""
    assert lineage_digest(SOURCE) == lineage_digest(SOURCE)


def test_different_sources_have_different_lineages():
    assert lineage_digest(SOURCE) != lineage_digest(SOURCE + "\n# changed\n")


# --- the split is drawn per lineage, and it is stable ------------------------------------------


def test_a_lineage_always_lands_in_the_same_split():
    """Otherwise a regenerated project reshuffles the suite, and a task the model was
    trained on last week is sealed evaluation this week."""
    lineage = lineage_digest(SOURCE)
    assert split_for(lineage) == split_for(lineage)


def test_every_split_is_one_of_the_declared_ones():
    for index in range(200):
        assert split_for(lineage_digest(f"source-{index}")) in SPLITS


def test_the_salt_redraws_the_assignment():
    """Drawing a fresh sealed set has to be possible, and reassigning everything is its
    honest cost rather than something to work around."""
    lineages = [lineage_digest(f"source-{i}") for i in range(60)]
    a = [split_for(x) for x in lineages]
    b = [split_for(x, salt="second-sealed-set") for x in lineages]
    assert a != b


def test_the_shares_are_roughly_honoured():
    counts = {name: 0 for name in SPLITS}
    for index in range(3000):
        counts[split_for(lineage_digest(f"source-{index}"))] += 1
    assert counts[TRAIN] > counts[DEV]
    assert counts[TRAIN] > counts[SEALED_EVAL]
    assert all(count > 0 for count in counts.values())


def test_shares_summing_to_zero_are_refused():
    with pytest.raises(SupplyError, match="sum to zero"):
        split_for("sha256:abc", shares={TRAIN: 0.0, DEV: 0.0, SEALED_EVAL: 0.0})


def test_a_split_can_be_forced_for_a_dedicated_run():
    assert split_for("sha256:abc", shares={TRAIN: 1.0, DEV: 0.0, SEALED_EVAL: 0.0}) == TRAIN


# --- generating from a real project -------------------------------------------------------------


def test_a_project_generates_tasks_that_the_verifier_actually_catches(tmp_path):
    """The invariant the whole generator rests on: a mutant is only emitted once it has been
    observed to FAIL. An equivalent mutant would be a task solved by doing nothing, and that
    success would flow into SFT data and the capability matrix."""
    records, report = build([_project(tmp_path)], workspace=tmp_path / "ws")
    assert report.emitted > 0
    assert records
    for record in records:
        assert record["verify"]
        assert record["setup"]
        assert record["metadata"]["lineage_digest"].startswith("sha256:")


def test_every_mutant_of_one_file_shares_a_lineage_and_a_split(tmp_path):
    """The property that makes a generated suite safe to benchmark on. Split per task and
    `flip_comparison-0003` trains while `flip_comparison-0007` -- same file, same function,
    one operator apart -- is sealed evaluation."""
    records, report = build([_project(tmp_path)], workspace=tmp_path / "ws")
    lineages = {r["metadata"]["lineage_digest"] for r in records}
    splits = {r["metadata"]["split"] for r in records}
    assert len(lineages) == 1
    assert len(splits) == 1
    assert len(set(report.lineages.values())) == 1


def test_the_split_is_also_a_tag_so_a_suite_can_be_filtered_on_it(tmp_path):
    """`load_suite` filters on tags; a split recorded only in metadata could not be selected
    without loading and re-filtering every task by hand."""
    records, _report = build([_project(tmp_path)], workspace=tmp_path / "ws")
    for record in records:
        assert record["metadata"]["split"] in record["tags"]


def test_generated_tasks_load_as_tasks(tmp_path):
    """A suite that writes files `load_suite` cannot read is a supply of nothing."""
    from hermesbench.tasks import load_suite

    records, _report = build([_project(tmp_path)], workspace=tmp_path / "ws")
    written = write_suite(records, tmp_path / "suite" / "gen")
    assert written == len(records)
    loaded = load_suite("gen", root=tmp_path / "suite")
    assert len(loaded) == written
    assert all(task.task_id for task in loaded)


def test_two_projects_holding_the_same_file_are_one_lineage(tmp_path):
    """Vendoring is how one file becomes two paths. Content addressing is what stops the
    split straddling them."""
    first = _project(tmp_path, name="alpha")
    second = _project(tmp_path, name="beta")
    _records, report = build([first, second], workspace=tmp_path / "ws")
    assert len(set(report.lineages.values())) == 1


def test_the_second_project_is_rejected_as_a_duplicate_not_silently_kept(tmp_path):
    """Same verifier, same environment, different directory. Keeping both would double the
    suite while measuring one thing twice, and the count would read as coverage."""
    _records, report = build(
        [_project(tmp_path, name="alpha"), _project(tmp_path, name="beta")], workspace=tmp_path / "ws"
    )
    assert report.rejected
    assert "execution_duplicate" in report.rejection_reasons


# --- refusals ----------------------------------------------------------------------------------


def test_no_projects_is_refused(tmp_path):
    with pytest.raises(SupplyError, match="nothing to mutate"):
        build([], workspace=tmp_path / "ws")


def test_a_missing_target_file_is_refused(tmp_path):
    project = _project(tmp_path)
    broken = SourceProject(
        name=project.name,
        project_dir=project.project_dir,
        target_files=("does_not_exist.py",),
        verify=project.verify,
    )
    with pytest.raises(SupplyError, match="no such target file"):
        build([broken], workspace=tmp_path / "ws")


def test_a_project_whose_verifier_already_fails_is_refused(tmp_path):
    """Generating find-the-bug tasks from a broken project produces tasks nobody can
    complete, and every one of them would look like a hard task rather than a broken one."""
    from hermesbench.mutation import MutationError

    project = _project(tmp_path)
    (project.project_dir / "calc.py").write_text("def clamp(*a):\n    raise RuntimeError\n", encoding="utf-8")
    with pytest.raises(MutationError, match="does not pass on the unmutated project"):
        build([project], workspace=tmp_path / "ws")


# --- the report says what happened ----------------------------------------------------------------


def test_an_empty_supply_is_reported_as_unfit():
    from hermesbench.supply import SupplyReport

    assert problems(SupplyReport()) == ["no task survived generation and dedup; the suite would be empty"]


def test_too_few_lineages_is_reported_even_when_many_tasks_were_generated():
    """The failure a task count cannot show: a thousand tasks from one file is one split
    holding everything and two holding nothing."""
    from hermesbench.supply import SupplyReport

    report = SupplyReport(
        emitted=1000,
        accepted=1000,
        per_split={TRAIN: 1000},
        lineages={f"t{i}": "sha256:one" for i in range(1000)},
    )
    found = problems(report)
    assert any("lineage" in problem for problem in found)


def test_a_healthy_supply_reports_no_problems():
    from hermesbench.supply import SupplyReport

    report = SupplyReport(
        emitted=40,
        accepted=36,
        per_split={TRAIN: 24, DEV: 6, SEALED_EVAL: 6},
        lineages={f"t{i}": f"sha256:{i % 9}" for i in range(36)},
    )
    assert problems(report) == []


# --- manifests --------------------------------------------------------------------------------------


def test_a_manifest_round_trips(tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "projects": [
                    {
                        "name": "calcs",
                        "project_dir": "calcs",
                        "target_files": ["calc.py"],
                        "verify": "python -m pytest -q",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    projects = load_manifest(manifest)
    assert len(projects) == 1
    # Relative paths resolve against the manifest so it can be moved with its projects.
    assert projects[0].project_dir == (tmp_path / "calcs").resolve()


def test_a_json_manifest_also_loads(tmp_path):
    manifest = tmp_path / "sources.json"
    manifest.write_text(
        json.dumps({"projects": [{"name": "c", "project_dir": "/tmp/c", "target_files": ["a.py"], "verify": "true"}]}),
        encoding="utf-8",
    )
    assert load_manifest(manifest)[0].name == "c"


def test_a_manifest_with_no_projects_key_is_refused(tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(yaml.safe_dump({"sources": []}), encoding="utf-8")
    with pytest.raises(SupplyError, match="top-level 'projects'"):
        load_manifest(manifest)


def test_an_incomplete_project_record_names_what_is_missing(tmp_path):
    manifest = tmp_path / "sources.yaml"
    manifest.write_text(yaml.safe_dump({"projects": [{"name": "c"}]}), encoding="utf-8")
    with pytest.raises(SupplyError, match="missing"):
        load_manifest(manifest)
