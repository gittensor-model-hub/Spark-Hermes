"""Every module with a `main()` can actually be run.

`validator/score.py` shipped without its `if __name__ == "__main__"` block, so
`python -m validator.score ...` imported the module, ran nothing, and **exited 0**. No output, no
traceback, no error -- the shape of success. It was found by using it in an end-to-end test and
noticing the scorecard file was absent, which is a slow way to learn that a command does nothing.

Enumerated from source rather than by invoking thirty-seven subprocesses, the same reasoning as
`validator.api.unscreened_handlers`: the failure mode is a module added later by someone who did
not hit this, and such a module passes every behavioural test that only exercises the commands
somebody remembered.
"""

import ast
from pathlib import Path

import pytest

PACKAGES = ("eval", "hermes", "hermesbench", "miner", "proof", "validator")


def _module_files() -> list[Path]:
    root = Path(__file__).resolve().parent.parent
    return sorted(p for pkg in PACKAGES for p in (root / pkg).rglob("*.py") if "__pycache__" not in p.parts)


def _defines_main(tree: ast.Module) -> bool:
    """A module-level `def main(...)`. Nested definitions do not make a module runnable."""
    return any(isinstance(node, ast.FunctionDef) and node.name == "main" for node in tree.body)


def _has_entry_point(tree: ast.Module) -> bool:
    """A module-level `if __name__ == "__main__":` guard.

    Matched on the AST rather than the text so a mention inside a docstring or a comment cannot
    satisfy it -- which is precisely how a guard test for a missing guard comes to pass.
    """
    for node in tree.body:
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (
            isinstance(test, ast.Compare)
            and isinstance(test.left, ast.Name)
            and test.left.id == "__name__"
            and any(isinstance(c, ast.Constant) and c.value == "__main__" for c in test.comparators)
        ):
            return True
    return False


CANDIDATES = [(p, ast.parse(p.read_text(encoding="utf-8"))) for p in _module_files()]
RUNNABLE = [(p, t) for p, t in CANDIDATES if _defines_main(t)]


def test_there_are_modules_to_check():
    """A guard whose corpus is empty passes for the wrong reason."""
    assert len(RUNNABLE) > 20, f"only found {len(RUNNABLE)} modules defining main()"


@pytest.mark.parametrize("path", [p for p, _ in RUNNABLE], ids=lambda p: p.as_posix())
def test_a_module_that_defines_main_can_be_invoked(path):
    """Without the guard, `python -m <module>` exits 0 having done nothing.

    That is worse than a crash: a caller checking the exit code sees success, and a caller reading
    stdout sees an empty result rather than an error. `validator.score` shipped this way and its
    scorecard silently never appeared.
    """
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert _has_entry_point(tree), (
        f'{path.as_posix()} defines main() with no `if __name__ == "__main__"` block, so '
        "`python -m` on it exits 0 without running anything"
    )


def test_the_guard_would_catch_a_module_missing_its_entry_point():
    """A guard nobody has seen fail is a guard nobody knows works."""
    missing = ast.parse("def main():\n    return 0\n")
    assert _defines_main(missing)
    assert not _has_entry_point(missing)

    present = ast.parse('def main():\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n')
    assert _has_entry_point(present)


def test_a_mention_in_a_docstring_does_not_satisfy_the_guard():
    """Matched on the AST, not the text. A grep-based version of this check is satisfied by the
    string appearing in a comment explaining that it is missing."""
    decoy = ast.parse('"""Remember to add if __name__ == \\"__main__\\" here."""\n\ndef main():\n    return 0\n')
    assert _defines_main(decoy)
    assert not _has_entry_point(decoy)
