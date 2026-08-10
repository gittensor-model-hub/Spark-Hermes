from eval.fair_dataset_label import (
    apply_fair_label,
    fair_label_from_rows_selected,
    rows_selected_for_sha,
)


def test_rows_selected_for_sha():
    manifest = {
        "components": [
            {"trajectories_sha256": "a" * 64, "rows_selected": 42},
            {"trajectories_sha256": "b" * 64, "rows_selected": 7},
        ]
    }
    assert rows_selected_for_sha(manifest, "a" * 64) == 42
    assert rows_selected_for_sha(manifest, "c" * 64) is None


def test_fair_label_from_rows_selected_matches_size_bands():
    assert fair_label_from_rows_selected(150) == "dataset:xl"
    assert fair_label_from_rows_selected(25) == "dataset:xs"
    assert fair_label_from_rows_selected(24) == "dataset:none"


def test_apply_fair_label_downgrades_bundle_label():
    report = apply_fair_label(
        {"verified": True, "label": "dataset:xl", "rows_total": 159},
        rows_selected=25,
    )
    assert report["label"] == "dataset:xs"
    assert report["bundle_label"] == "dataset:xl"
    assert report["rows_selected"] == 25
    assert "fair label dataset:xs" in report["fair_label_note"]


def test_apply_fair_label_keeps_matching_bundle_label():
    report = apply_fair_label(
        {"verified": True, "label": "dataset:l", "rows_total": 100},
        rows_selected=100,
    )
    assert report["label"] == "dataset:l"
    assert "fair_label_note" not in report
