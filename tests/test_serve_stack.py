from eval.serve_stack import (
    DEFAULT_CUDA_TAG,
    SERVE_MAX_MODEL_LEN,
    VLLM_VERSION,
    pytorch_extra_index_url,
    resolve_vllm_executable,
    vllm_manylinux_platform,
    vllm_serve_argv,
    vllm_wheel_url,
)


def test_vllm_wheel_url_x86_64_cu129():
    url = vllm_wheel_url(version=VLLM_VERSION, cuda_tag="129", cpu_arch="x86_64")
    assert url.endswith("vllm-0.25.0+cu129-cp38-abi3-manylinux_2_28_x86_64.whl")


def test_vllm_wheel_url_aarch64_cu130():
    url = vllm_wheel_url(version=VLLM_VERSION, cuda_tag="130", cpu_arch="aarch64")
    assert "manylinux_2_35_aarch64" in url
    assert "+cu130-" in url


def test_pytorch_extra_index_defaults_to_cu129():
    assert pytorch_extra_index_url() == f"https://download.pytorch.org/whl/cu{DEFAULT_CUDA_TAG}"


def test_vllm_manylinux_platform_mapping():
    assert vllm_manylinux_platform(cuda_tag="129", cpu_arch="aarch64") == "manylinux_2_28_aarch64"
    assert vllm_manylinux_platform(cuda_tag="130", cpu_arch="aarch64") == "manylinux_2_35_aarch64"


def test_resolve_vllm_executable_prefers_serve_venv(tmp_path, monkeypatch):
    venv = tmp_path / "serve"
    bindir = venv / "bin"
    bindir.mkdir(parents=True)
    vllm_bin = bindir / "vllm"
    vllm_bin.write_text("#!/bin/sh\necho vllm\n", encoding="utf-8")
    vllm_bin.chmod(0o755)
    monkeypatch.setenv("SPARKDISTILL_SERVE_VENV", str(venv))
    monkeypatch.delenv("PATH", raising=False)
    assert resolve_vllm_executable() == str(vllm_bin)


def test_vllm_serve_argv_includes_eval_defaults(monkeypatch):
    monkeypatch.setenv("SPARKDISTILL_GPU_ARCHITECTURE", "hopper-h100")
    argv = vllm_serve_argv("/models/ckpt", port=8001, served_model_name="ckpt")
    assert argv[0] == resolve_vllm_executable()
    assert "--served-model-name" in argv
    assert "ckpt" in argv
    assert "--seed" in argv
    assert "--no-enable-prefix-caching" in argv
    assert "--dtype" in argv
    assert "bfloat16" in argv
    assert "--max-model-len" in argv
    # vLLM 0.25 removed this flag; passing it exits `vllm serve` 2 (issue #303).
    assert "--disable-log-requests" not in argv
    assert "--compilation-config" not in argv


def test_serve_max_model_len_fits_full_config_output_budget():
    # Regression for issue #303: the serve context must exceed the full config's
    # output budget with room for the prompt, or every full-level request 400s.
    #
    # 4096 was read from `tritonbench/configs/default.yaml` until the vendored tree moved
    # to the `archive/tritonbench` branch. Inlined rather than dropped: the bug this
    # guards is a served-model misconfiguration, not a TritonBench one, and it would come
    # back the moment someone lowered SERVE_MAX_MODEL_LEN to save memory.
    largest_known_output_budget = 4096
    assert SERVE_MAX_MODEL_LEN > largest_known_output_budget


def test_vllm_serve_argv_blackwell_disables_cudagraph(monkeypatch):
    monkeypatch.setenv("SPARKDISTILL_GPU_ARCHITECTURE", "blackwell")
    argv = vllm_serve_argv("/models/ckpt")
    assert "--compilation-config" in argv
    assert '{"cudagraph_mode": "NONE"}' in argv


def test_python_headers_available_default_reads_sysconfig(tmp_path, monkeypatch):
    import eval.serve_stack as ss

    inc = tmp_path / "include"
    inc.mkdir()
    monkeypatch.setattr("sysconfig.get_paths", lambda: {"include": str(inc)})
    assert ss.python_headers_available() is False
    (inc / "Python.h").write_text("", encoding="utf-8")
    assert ss.python_headers_available() is True


def test_python_headers_available_subprocess_returncode(monkeypatch):
    import eval.serve_stack as ss

    class _R:
        def __init__(self, rc):
            self.returncode = rc

    monkeypatch.setattr(ss.subprocess, "run", lambda *a, **k: _R(0))
    assert ss.python_headers_available("/some/python") is True
    monkeypatch.setattr(ss.subprocess, "run", lambda *a, **k: _R(1))
    assert ss.python_headers_available("/some/python") is False


def test_python_headers_available_missing_interpreter_is_false(monkeypatch):
    import eval.serve_stack as ss

    def _boom(*a, **k):
        raise OSError("no such executable")

    monkeypatch.setattr(ss.subprocess, "run", _boom)
    assert ss.python_headers_available("/nonexistent/python") is False
