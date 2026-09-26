"""`custom swe_test <node id>` — one predicate per test the bug breaks. Runs inside the grading container over the
agent's final tree. Everything it needs arrives in `/ep/withheld.json` under `assets`, never in the agent's boundary:
`swe/spec.json` (the broken tests, the passing tests a fix must keep, the files the bug lives in, the tracked
test-infrastructure files) and `swe/files/<path>` (the files holding the broken tests, as they were before SWE-smith
deleted those tests).

The tests import the code under repair, so that code runs with them. Grading is built so that code has as little to
work with as possible:

  * **only the bug's files are graded** — the tree the tests run on is the pristine tree the task image ships
    (`/opt/sh-pristine`) with the agent's versions of the files the bug touched laid over it. Nothing else the agent
    changed or added — tests, configuration, other modules, `.pth` files — reaches the run;
  * **those files are scanned structurally** — their syntax tree is compared with the pristine one, and any construct
    that reaches for the test runner, the import machinery, other processes or the interpreter's frames that the
    pristine file did not already have (`_pytest`, `sys.modules`, `os.fork`, `exec`, `__subclasses__`, a dynamic
    `getattr` on `sys` …) is reported as tamper, which disqualifies the episode (`harness_tamper`);
  * **the tests run as a separate user** (uid 2000, dropped with `setpriv`) that cannot read the answer key in `/ep`,
    cannot write where outcomes are kept, and whose every process — a daemon started with `setsid` included — is
    killed before the outcomes are read;
  * **sentinels** — two tests written at grade time, inside the hidden test module when there is one, under names and
    numbers chosen then: one must PASS and one must FAIL, or the run is not trusted at all;
  * **outcomes come from a witness plugin**, not from parsing stdout, so a printed line cannot stand in for a result,
    and a repository's own `addopts` cannot change the format.

A broken test counts only if it PASSES **and every kept test PASSES**: a fix that breaks what worked is not a fix,
and an outcome other than PASSED (an XFAIL raised from library code, a skip) is not a pass.
"""

import ast
import base64
import collections
import hashlib
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import tempfile
import time

TEST_INFRA = re.compile(
    r"(^|/)(tests?|testing)/|^tests?\.py$|^test_[^/]*\.py$|^[^/]*_tests?\.py$|(^|/)conftest\.py$|(^|/)(site|user)customize\.py$"
    r"|\.pth$|(^|/)(pytest\.ini|\.pytest\.ini|tox\.ini|setup\.cfg|pyproject\.toml|setup\.py|\.coveragerc|noxfile\.py)$"
)
PRISTINE = "/opt/sh-pristine"
TESTER = 2000  # the uid the suite runs as; the grader itself is root inside its container
PYTEST = (
    "source /opt/miniconda3/bin/activate >/dev/null 2>&1; conda activate testbed >/dev/null 2>&1; "
    'exec pytest --disable-warnings --color=no --tb=no -p no:cacheprovider -p no:xdist -p no:randomly -p sh_witness "$@"'
)
WITNESS = '''"""The validator's witness: every test's outcome, by node id, written where the runner reads it."""
import json, os
_seen = {}


def pytest_runtest_logreport(report):
    o = report.outcome  # passed | failed | skipped
    if report.when == "call":
        if hasattr(report, "wasxfail"):
            o = "xfail" if o == "skipped" else "xpass"
        _seen[report.nodeid] = o.upper()
    elif report.when == "setup" and o != "passed":
        _seen[report.nodeid] = "ERROR" if o == "failed" else "SKIPPED"
    elif report.when == "teardown" and o == "failed":
        _seen[report.nodeid] = "ERROR"


def pytest_sessionfinish(session, exitstatus):
    with open(os.environ["SH_WITNESS_OUT"], "w") as f:
        json.dump(_seen, f)
'''
STATUS = re.compile(r"^(\S+)\s+(PASSED|FAILED|ERROR|SKIPPED|XFAIL|XPASS)\b")  # what SWE-smith's parser reads
_RESULTS: dict = {}


def parse_verbose(output: str) -> dict:
    """{node id: status} from `pytest --verbose` — the index checks every node id against it; grading uses the witness."""
    return {m.group(1): m.group(2) for line in output.splitlines() if (m := STATUS.match(line))}


# ── the structural scan ──

# Modules no bug fix needs to start importing: the test runner, the import and introspection machinery, other
# processes, interpreter hooks.
DANGEROUS_MODULES = frozenset(
    {
        "_pytest", "pytest", "pluggy", "importlib", "imp", "inspect", "gc", "ctypes", "cffi", "subprocess",
        "multiprocessing", "signal", "atexit", "threading", "_thread", "marshal", "builtins", "code", "codeop",
        "runpy", "pty", "resource", "site", "sitecustomize", "usercustomize", "faulthandler", "tracemalloc",
    }
)  # fmt: skip
# Attributes that reach the same places, wherever they are read from.
DANGEROUS_ATTRS = frozenset(
    {
        "meta_path", "path_hooks", "path_importer_cache", "settrace", "setprofile", "_getframe", "addaudithook",
        "__import__", "__builtins__", "__subclasses__", "__globals__", "__code__", "__closure__", "f_back",
        "f_globals", "f_locals", "f_code", "tb_frame", "gi_frame", "cr_frame", "__loader__",
    }
)  # fmt: skip
# ... and those that are dangerous on the process modules only (`obj.fork()` on a repository's own object is not).
# What only lets the code sink its own run (`sys.exit`, `os._exit`, `os.kill`) is not listed: that costs nobody else.
PROCESS_ATTRS = frozenset(
    {
        "fork", "forkpty", "setsid", "setpgid", "setpgrp", "execv", "execve", "execvp", "execvpe", "execl", "execle",
        "execlp", "spawnv", "spawnve", "spawnl", "system", "popen", "posix_spawn", "modules",
    }
)  # fmt: skip
PROCESS_MODULES = frozenset({"os", "posix", "sys", "signal", "subprocess", "pty", "_thread", "threading"})
DANGEROUS_CALLS = frozenset({"exec", "eval", "compile", "__import__", "breakpoint"})
_WORD = re.compile(
    r"_?pytest|pluggy|sitecustomize|usercustomize|__import__|meta_path|path_hooks|__builtins__|setsid|fork"
)


def _folded(node) -> str | None:
    """A string literal, or a chain of them joined with `+` — what an obfuscated module name is built from."""
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
        left, right = _folded(node.left), _folded(node.right)
        return left + right if left is not None and right is not None else None
    if isinstance(node, ast.JoinedStr):
        return "".join(v.value for v in node.values if isinstance(v, ast.Constant) and isinstance(v.value, str))
    return None


def _docstrings(tree) -> set:
    docs = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Module, ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and n.body:
            first = n.body[0]
            if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
                docs.add(id(first.value))
    return docs


def constructs(source: str) -> collections.Counter:
    """How often each dangerous construct occurs in a module. Pure; a module that does not parse has none (its tests
    will fail on their own)."""
    found: collections.Counter = collections.Counter()
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError):
        return found
    docs = _docstrings(tree)
    parts = set()  # string pieces already counted as part of a larger `+` chain
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] in DANGEROUS_MODULES:
                    found[f"import {alias.name.split('.')[0]}"] += 1
        elif isinstance(node, ast.ImportFrom):
            base = (node.module or "").split(".")[0]
            if base in DANGEROUS_MODULES:
                found[f"import {base}"] += 1
            if base in PROCESS_MODULES:
                for alias in node.names:
                    if alias.name in PROCESS_ATTRS or alias.name in DANGEROUS_ATTRS:
                        found[f"from {base} import {alias.name}"] += 1
        elif isinstance(node, ast.Attribute):
            if node.attr in DANGEROUS_ATTRS:
                found[f".{node.attr}"] += 1
            elif node.attr in PROCESS_ATTRS and isinstance(node.value, ast.Name) and node.value.id in PROCESS_MODULES:
                found[f"{node.value.id}.{node.attr}"] += 1
            elif isinstance(node.value, ast.Name) and node.value.id in ("pytest", "_pytest", "pluggy"):
                found[f"{node.value.id}.{node.attr}"] += 1
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            name = node.func.id
            if name in DANGEROUS_CALLS:
                found[f"{name}()"] += 1
            elif (
                name == "vars"
                and node.args
                and isinstance(node.args[0], ast.Name)
                and (node.args[0].id in PROCESS_MODULES or node.args[0].id in DANGEROUS_MODULES)
            ):
                found[f"vars({node.args[0].id})"] += 1
            elif name in ("getattr", "setattr", "delattr", "hasattr") and len(node.args) >= 2:
                target, attr = node.args[0], node.args[1]
                on_module = isinstance(target, ast.Name) and (
                    target.id in PROCESS_MODULES or target.id in DANGEROUS_MODULES
                )
                literal = _folded(attr)
                if on_module and literal is None:
                    found[f"{name}({target.id}, <dynamic>)"] += 1
                elif literal is not None and (literal in DANGEROUS_ATTRS or (on_module and literal in PROCESS_ATTRS)):
                    found[f"{name}(…, {literal!r})"] += 1
        if isinstance(node, (ast.BinOp, ast.JoinedStr)) and id(node) not in parts:
            text = _folded(node)
            if text is not None:
                for child in ast.walk(node):
                    parts.add(id(child))
                if (m := _WORD.search(text)) is not None:
                    found[f"string {m.group(0)}"] += 1
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in parts:
            if id(node) not in docs and (m := _WORD.search(node.value)) is not None:
                found[f"string {m.group(0)}"] += 1
    return found


def _text(path: str) -> str | None:
    try:
        with open(path, "rb") as f:
            return f.read().decode(errors="replace")
    except OSError:
        return None


def tamper_scan(ws: str, pristine: str, files=None) -> list:
    """Constructs the agent's versions of `files` (default: every non-test `.py` file outside dot-directories) have
    more of than the pristine versions. [(file, construct)]; [] when clean."""
    if files is None:
        files = []
        for root, dirs, names in os.walk(ws):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d != "__pycache__" and not d.endswith(".egg-info")]
            for name in names:
                rel = os.path.relpath(os.path.join(root, name), ws)
                if name.endswith(".py") and not TEST_INFRA.search(rel):
                    files.append(rel)
    hits = []
    for rel in sorted(files):
        if not rel.endswith(".py"):
            continue
        now, was = _text(os.path.join(ws, rel)), _text(os.path.join(pristine, rel))
        if now is None or now == was:
            continue  # a new file is scanned against nothing: an `__init__.py` or a module shadowing `json` runs too
        before = constructs(was or "")
        for key, n in sorted(constructs(now).items()):
            if n > before.get(key, 0):
                hits.append((rel, key))
    return hits


# ── the tree, the sentinels, the run ──


def stray_files(ws: str, shipped: set) -> list:
    """Test-infrastructure files in the tree that the task did not ship (relative paths)."""
    out = []
    for root, dirs, files in os.walk(ws):
        dirs[:] = [d for d in dirs if d != ".git"]
        for name in files:
            rel = os.path.relpath(os.path.join(root, name), ws)
            if TEST_INFRA.search(rel) and rel not in shipped:
                out.append(rel)
    return sorted(out)


def _raw(path: str) -> bytes | None:
    try:
        with open(path, "rb") as f:
            return f.read()
    except OSError:
        return None


def graded_tree(ws: str, pristine: str, bug_files) -> list:
    """Make `ws` the pristine tree with the agent's versions of the bug's files laid over it. Returns the bug files
    whose agent version differs from the pristine one (a deleted file counts)."""
    kept = {rel: _raw(os.path.join(ws, rel)) for rel in bug_files}
    for name in os.listdir(ws):
        path = os.path.join(ws, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)
    shutil.copytree(pristine, ws, symlinks=True, dirs_exist_ok=True)
    changed = []
    for rel, data in kept.items():
        path = os.path.join(ws, rel)
        if data != _raw(path):
            changed.append(rel)
        if data is None:
            if os.path.lexists(path):
                os.remove(path)
        else:
            os.makedirs(os.path.dirname(path) or ws, exist_ok=True)
            with open(path, "wb") as f:
                f.write(data)
    return sorted(changed)


def verdicts(spec: dict, statuses: dict, valid: bool = True) -> dict:
    """{broken test: counts} — PASSED, and every kept test PASSED, in a run the sentinels vouch for. Pure."""
    if not valid:
        return {t: False for t in spec["f2p"]}
    kept = all(statuses.get(t) == "PASSED" for t in spec.get("p2p", []))
    return {t: statuses.get(t) == "PASSED" and kept for t in spec["f2p"]}


def _assets() -> dict:
    ep = os.environ.get("SH_EP", "/ep")
    try:
        with open(os.path.join(ep, "withheld.json")) as f:
            return json.load(f).get("assets", {})
    except (OSError, ValueError):
        return {}


def sentinel_prefix(ids) -> str:
    """The prefix this repository's test functions carry (`test_`, or `it_` where `python_functions` says so): a
    sentinel pytest does not collect is `not found`, and pytest then runs nothing at all. Pure."""
    for t in ids:
        name = t.split("[", 1)[0].split("::")[-1] if "::" in t else ""
        if "_" in name.strip("_"):
            return name.split("_")[0] + "_"
    return "test_"


def sentinel_dir(spec: dict, ws: str) -> str:
    """Beside the broken tests when they live in a Python test module; a data tree collected by a custom collector
    (pygments' example files) would swallow a new .py file, so then the nearest plain tests directory, or the root."""
    for t in spec["f2p"]:
        f = t.split("::")[0]
        if f.endswith(".py"):
            return os.path.dirname(f)
    for cand in ("tests", "test"):
        if os.path.isdir(os.path.join(ws, cand)):
            return cand
    return ""


def sentinel_names(module_source: str, prefix: str) -> tuple:
    """Two new function names in the module's own vocabulary — words drawn from its own test names — so the sentinels
    read like two more of its tests. Never a name the module already defines."""
    defined = set(re.findall(r"def (\w+)\(", module_source))
    words = sorted({w for n in defined for w in n.split("_")[1:] if w.isalpha() and len(w) > 2}) or [
        "case",
        "value",
        "result",
        "default",
    ]
    rng, names = secrets.SystemRandom(), []
    while len(names) < 2:
        picked = rng.sample(words, min(len(words), rng.choice((2, 3))))
        name = prefix + "_".join(picked) + f"_{rng.randrange(10, 99)}"
        if name not in defined and name not in names:
            names.append(name)
    return tuple(names)


def _digests(ws: str, rels) -> dict:
    out = {}
    for rel in rels:
        p = os.path.join(ws, rel)
        try:
            with open(p, "rb") as f:
                out[rel] = hashlib.sha256(f.read()).hexdigest()
        except OSError:
            out[rel] = None
    return out


def as_tester(cmd: list) -> list:
    """The suite drops to the tester's uid when the grader is root; outside a grader (a developer's machine, a unit
    test) it runs as whoever runs it."""
    if os.geteuid() != 0:
        return cmd
    return ["setpriv", f"--reuid={TESTER}", f"--regid={TESTER}", "--clear-groups", "--inh-caps=-all", *cmd]


def reap(uid: int) -> int:
    """Kill every process of `uid`, however it detached itself. Root only; returns how many were killed."""
    if os.geteuid() != 0:
        return 0
    killed = 0
    for _ in range(50):
        found = False
        for pid in os.listdir("/proc"):
            if not pid.isdigit():
                continue
            try:
                with open(f"/proc/{pid}/status") as f:
                    owner = next(int(line.split()[1]) for line in f if line.startswith("Uid:"))
            except (OSError, StopIteration, ValueError):
                continue
            if owner == uid:
                found = True
                try:
                    os.kill(int(pid), signal.SIGKILL)
                    killed += 1
                except (ProcessLookupError, PermissionError):
                    pass
        if not found:
            break
        time.sleep(0.1)
    return killed


def _chown_tree(path: str, uid: int) -> None:
    if os.geteuid() != 0:
        return
    os.lchown(path, uid, uid)
    for root, dirs, files in os.walk(path):
        for name in dirs + files:
            try:
                os.lchown(os.path.join(root, name), uid, uid)
            except OSError:
                pass


def run_suite(ws: str, spec: dict, files: dict) -> tuple:
    """Scan, build the graded tree, write the broken tests back with the sentinels, run the suite once as the tester
    through the witness, reap. Returns ({node id: status}, info) with info = graded (bug files the agent changed),
    removed (stray test infrastructure, when the task names no bug files), modified (shipped test files the suite
    itself changed), tamper (constructs added to graded files; None without a pristine copy), valid (the sentinels
    ran true)."""
    pristine = PRISTINE if os.path.isdir(PRISTINE) else None
    bug_files = spec.get("bug_files")
    removed: list = []
    graded: list = []
    if pristine and bug_files:
        tamper = tamper_scan(ws, pristine, bug_files)
        graded = graded_tree(ws, pristine, bug_files)
    else:  # a task sealed before bug files were recorded, or no pristine copy: grade the whole tree, scanned
        tamper = tamper_scan(ws, pristine) if pristine else None
        for root, dirs, _ in os.walk(ws):
            for d in list(dirs):
                if d == "__pycache__":
                    shutil.rmtree(os.path.join(root, d), ignore_errors=True)
                    dirs.remove(d)
        for rel in stray_files(ws, set(spec["infra"])):
            os.remove(os.path.join(ws, rel))
            removed.append(rel)
    for rel, data in files.items():
        path = os.path.join(ws, rel)
        os.makedirs(os.path.dirname(path) or ws, exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
    prefix = sentinel_prefix([*spec["f2p"], *spec["p2p"]])
    host = next((rel for rel in sorted(files) if rel.endswith(".py") and "conftest" not in rel), None)
    x, y = secrets.randbelow(900) + 100, secrets.randbelow(900) + 100
    bodies = f"    assert {x} + {y} == {x + y}\n", f"    assert {x} * {y} == {x * y + 1}\n"
    if host is not None:  # inside the hidden module: two more of its tests, as far as anything running can tell
        a, b = sentinel_names(files[host].decode(errors="replace"), prefix)
        sentinel, own_file = host, False
        with open(os.path.join(ws, host), "a") as f:
            f.write(f"\n\ndef {a}():\n{bodies[0]}\n\ndef {b}():\n{bodies[1]}")
    else:
        a, b = f"{prefix}{secrets.token_hex(5)}", f"{prefix}{secrets.token_hex(5)}"
        sentinel, own_file = os.path.join(sentinel_dir(spec, ws), f"test_{secrets.token_hex(4)}.py"), True
        with open(os.path.join(ws, sentinel), "w") as f:
            f.write(f"def {a}():\n{bodies[0]}\n\ndef {b}():\n{bodies[1]}")
    guarded = [rel for rel in spec["infra"] if rel not in files]
    before = _digests(ws, guarded)
    work = tempfile.mkdtemp(prefix="sh-swe-")
    with open(os.path.join(work, "sh_witness.py"), "w") as f:
        f.write(WITNESS)
    witness_out = os.path.join(work, "outcomes.json")
    _chown_tree(ws, TESTER)
    _chown_tree(work, TESTER)
    env = {
        "PATH": os.environ.get("PATH", "/usr/local/bin:/usr/bin:/bin"),
        "LANG": os.environ.get("LANG", "C.UTF-8"),
        "HOME": work,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": work,
        "SH_WITNESS_OUT": witness_out,
    }  # nothing of the grader's own environment: not where /ep is, not a token
    ids = [*spec["f2p"], *spec["p2p"], f"{sentinel}::{a}", f"{sentinel}::{b}"]
    with tempfile.TemporaryFile() as out:
        proc = subprocess.Popen(
            as_tester(["bash", "-c", PYTEST, "pytest", *ids]),
            cwd=ws,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=out,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            proc.wait(timeout=int(spec.get("timeout_s", 900)))
        except subprocess.TimeoutExpired:
            pass
        finally:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
            reap(TESTER)  # before a single outcome is read: nothing of the run may still be writing
    try:
        with open(witness_out) as f:
            statuses = json.load(f)
    except (OSError, ValueError):
        statuses = {}  # no witness: nothing ran, or something kept it from speaking — not a result either way
    # The sentinels were the grader's, never the tree's: take them out again. Leaving two randomly named tests in
    # the hidden module made the tree differ between two otherwise identical runs, and mint's "materialises
    # identically twice" gate rejected every instance whose broken tests live in a Python module.
    if own_file:
        os.remove(os.path.join(ws, sentinel))
    else:
        with open(os.path.join(ws, host), "wb") as f:
            f.write(files[host])
    after = _digests(ws, guarded)
    info = {
        "graded": graded,
        "removed": removed,
        "modified": sorted(rel for rel in guarded if before[rel] != after[rel]),
        "tamper": tamper,
        "valid": statuses.get(f"{sentinel}::{a}") == "PASSED" and statuses.get(f"{sentinel}::{b}") == "FAILED",
    }
    return statuses, info


def _run_tests(ws) -> dict:
    key = str(ws)
    if key in _RESULTS:
        return _RESULTS[key]
    assets = _assets()
    spec = json.loads(base64.b64decode(assets["swe/spec.json"]))
    files = {n[len("swe/files/") :]: base64.b64decode(b) for n, b in assets.items() if n.startswith("swe/files/")}
    statuses, info = run_suite(str(ws), spec, files)
    results = verdicts(spec, statuses, info["valid"] and not info["tamper"])
    _RESULTS[key] = results
    try:  # every broken test's outcome beside the grade (a fraction needs all of them), and why
        outdir = os.path.join(os.environ.get("SH_EP", "/ep"), "out")
        if os.path.isdir(outdir):
            with open(os.path.join(outdir, "tests.json"), "w") as f:
                json.dump(results, f)
            with open(os.path.join(outdir, "detail.json"), "w") as f:  # the grader keeps it beside the episode
                json.dump(
                    {
                        "broken": {t: statuses.get(t, "not run") for t in spec["f2p"]},
                        "kept_failed": [t for t in spec.get("p2p", []) if statuses.get(t) != "PASSED"],
                        "kept_total": len(spec.get("p2p", [])),
                        "graded_files": info["graded"],
                        "removed_infra": info["removed"],
                        "suite_modified": info["modified"],
                        "tamper": info["tamper"],
                        "valid": info["valid"],
                    },
                    f,
                )
    except OSError:
        pass
    return results


def swe_test(ws, node_id: str) -> bool:
    return bool(_run_tests(ws).get(node_id, False))


CHECKS = {"swe_test": swe_test}
