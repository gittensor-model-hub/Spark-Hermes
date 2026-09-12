"""Checkpoint completeness at the actual shared merge/parent identity boundary."""

import hashlib

import pytest
from release_support import checkpoint

from admin.artifacts import StageError, checkpoint_files, write_record
from admin.candidates import check_initial_parent


def indexed(root, names, *, kind="safetensors"):
    root.mkdir(parents=True, exist_ok=True)
    write_record(root / "config.json", {"model_type": "qwen3_5", "fixture_only": True})
    write_record(root / "tokenizer.json", {"fixture_only": True})
    write_record(root / "tokenizer_config.json", {"fixture_only": True})
    for name in names:
        (root / name).write_text("CPU fixture bytes " + name)
    name = "model.safetensors.index.json" if kind == "safetensors" else "pytorch_model.bin.index.json"
    write_record(root / name, {"weight_map": {f"layer.{i}": s for i, s in enumerate(names)}})
    return root / name


@pytest.mark.parametrize("kind", ["safetensors", "bin"])
def test_complete_single_and_numbered_layouts(tmp_path, kind):
    single = checkpoint(tmp_path / "single", "single")
    if kind == "bin":
        (single / "model.safetensors").rename(single / "pytorch_model.bin")
    assert len(checkpoint_files(single)) == 4
    prefix = "model" if kind == "safetensors" else "pytorch_model"
    names = [f"{prefix}-{i:05}-of-00002.{kind}" for i in (1, 2)]
    index = indexed(tmp_path / "indexed", names, kind=kind)
    files = checkpoint_files(index.parent)
    assert set(names) <= files.keys()
    for name in names:
        assert files[name] == hashlib.sha256((index.parent / name).read_bytes()).hexdigest()
    (index.parent / names[-1]).unlink()
    with pytest.raises(StageError, match="missing or unsafe"):
        checkpoint_files(index.parent)


def test_index_references_outside_old_glob_are_hashed(tmp_path):
    indexed(tmp_path, ["untracked.bin", "other.bin"], kind="bin")
    before = checkpoint_files(tmp_path)
    assert {"untracked.bin", "other.bin"} <= before.keys()
    (tmp_path / "untracked.bin").write_text("changed actual referenced bytes")
    assert checkpoint_files(tmp_path)["untracked.bin"] != before["untracked.bin"]


@pytest.mark.parametrize(
    "variant",
    [
        "partial",
        "unindexed_complete",
        "monolith_and_shard",
        "mixed",
        "unknown_index",
        "empty_map",
        "omitted",
        "missing_number",
        "mixed_totals",
        "duplicate_number",
        "zero_number",
        "mixed_numbered",
        "missing_file",
        "empty_file",
        "directory",
        "traversal",
        "absolute",
        "backslash",
        "wrong_suffix",
    ],
)
def test_incomplete_ambiguous_and_unsafe_layouts_refuse(tmp_path, variant):
    a, b = "model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"
    index = indexed(tmp_path, [a, b])
    if variant in {"partial", "unindexed_complete"}:
        index.unlink()
        if variant == "partial":
            (tmp_path / b).unlink()
    elif variant == "monolith_and_shard":
        (tmp_path / "model.safetensors").write_text("ambiguous complete alternate")
    elif variant == "mixed":
        write_record(tmp_path / "pytorch_model.bin.index.json", {"weight_map": {"x": "x.bin"}})
    elif variant == "unknown_index":
        index.rename(tmp_path / "unknown.index.json")
    elif variant == "empty_map":
        write_record(index, {"weight_map": {}})
    elif variant in {"omitted", "missing_number"}:
        write_record(index, {"weight_map": {"x": a}})
        if variant == "missing_number":
            (tmp_path / b).unlink()
    elif variant in {"mixed_totals", "duplicate_number", "zero_number", "mixed_numbered"}:
        name = {
            "mixed_totals": "model-00002-of-00003.safetensors",
            "duplicate_number": "model-1-of-00002.safetensors",
            "zero_number": "model-00000-of-00002.safetensors",
            "mixed_numbered": "extra.safetensors",
        }[variant]
        (tmp_path / b).rename(tmp_path / name)
        write_record(index, {"weight_map": {"x": a, "y": name}})
    elif variant in {"missing_file", "empty_file", "directory"}:
        (tmp_path / b).unlink()
        if variant == "empty_file":
            (tmp_path / b).touch()
        if variant == "directory":
            (tmp_path / b).mkdir()
    else:
        name = {
            "traversal": "../other.safetensors",
            "absolute": "/tmp/other.safetensors",
            "backslash": "nested\\other.safetensors",
            "wrong_suffix": "config.json",
        }[variant]
        write_record(index, {"weight_map": {"x": name}})
    with pytest.raises(StageError):
        checkpoint_files(tmp_path)


def test_pinned_local_hub_cache_keeps_blob_symlinks(tmp_path, monkeypatch):
    import huggingface_hub

    revision = "a" * 40
    snapshot = checkpoint(tmp_path / "snapshots" / revision, "pinned")
    blob = tmp_path / "blob"
    (snapshot / "model.safetensors").rename(blob)
    (snapshot / "model.safetensors").symlink_to(blob)
    calls = []

    def cached(repository, **kwargs):
        calls.append((repository, kwargs))
        return str(snapshot)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", cached)
    check_initial_parent(
        snapshot, {"base_model": "Qwen/Qwen3.5-4B", "revision": revision}, identity={"mode": "production"}
    )
    assert calls == [("Qwen/Qwen3.5-4B", {"revision": revision, "local_files_only": True})]
    assert checkpoint_files(snapshot)["model.safetensors"] == hashlib.sha256(blob.read_bytes()).hexdigest()


def test_index_digest_binds_the_snapshot_that_selected_shards(tmp_path, monkeypatch):
    import admin.artifacts as artifacts

    index = indexed(tmp_path, ["untracked.bin"], kind="bin")
    original = index.read_bytes()
    digest = artifacts.file_digest

    def change_index(path):
        if path.name == "untracked.bin":
            write_record(index, {"weight_map": {"new": "missing.bin"}})
        return digest(path)

    monkeypatch.setattr(artifacts, "file_digest", change_index)
    files = checkpoint_files(tmp_path)
    assert files[index.name] == hashlib.sha256(original).hexdigest()
    assert files[index.name] != hashlib.sha256(index.read_bytes()).hexdigest()
    with pytest.raises(StageError, match="missing or unsafe"):
        checkpoint_files(tmp_path)
