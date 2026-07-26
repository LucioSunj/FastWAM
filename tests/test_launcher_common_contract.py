"""Contract tests for ``scripts/adaptive_gate/_common.sh`` root resolution.

The adaptive_gate launchers live in FastWAM but every analyzer tool
(``MANIFEST_TOOL``, ``DECISION_TOOL``, ...) lives in the outer workspace
repository. ``_common.sh`` therefore has to locate that outer repository before
it can source the shared launcher contract.

Historically it hard-coded ``${PROJECT_REPO_ROOT}/../scripts/adaptive_gate``,
which is only correct for the primary checkout. A linked worktree such as
``FastWAM-worktrees/<slug>`` has a parent directory that contains no ``scripts/``
at all, so all thirteen launchers died with an opaque ``No such file or
directory`` from bash itself.

These tests pin the three behaviours the resolution must have:

1. no ``WORKSPACE_ROOT`` and a primary-checkout-shaped layout -> unchanged
   behaviour, the parent directory is used;
2. ``WORKSPACE_ROOT`` set -> that value wins, so a linked worktree works;
3. neither -> fail closed with an actionable message and a non-zero status.

Everything runs against synthetic directory layouts and a stub outer
``_common.sh``. The tests deliberately do **not** require the real outer
repository to be present, so the FastWAM submodule stays testable standalone.
"""

from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_COMMON = REPO_ROOT / "scripts" / "adaptive_gate" / "_common.sh"


# Mirrors the parts of the outer ``scripts/adaptive_gate/_common.sh`` contract
# that FastWAM's ``_common.sh`` interacts with: it refuses to run without
# PROJECT_REPO_ROOT, and it derives WORKSPACE_ROOT from the FastWAM parent only
# when the caller has not already provided one. Keeping the `:-` default here is
# what lets test_workspace_root_wins_over_parent_derivation catch a regression
# where FastWAM's side forgets to set the variable before sourcing.
STUB_OUTER_COMMON = """#!/usr/bin/env bash
if [[ -z "${PROJECT_REPO_ROOT:-}" ]]; then
    echo "PROJECT_REPO_ROOT must be set before sourcing adaptive_gate/_common.sh" >&2
    return 2
fi
WORKSPACE_ROOT=${WORKSPACE_ROOT:-"$(cd "${PROJECT_REPO_ROOT}/.." && pwd)"}
STUB_OUTER_COMMON_WAS_SOURCED=1
"""

# Sources FastWAM's _common.sh exactly the way a launcher does (`set -euo
# pipefail` first, then source) and reports what was resolved.
PROBE = """#!/usr/bin/env bash
set -euo pipefail
source "$1/scripts/adaptive_gate/_common.sh"
echo "OUTER_SOURCED=${STUB_OUTER_COMMON_WAS_SOURCED:-0}"
echo "PROJECT_REPO_ROOT=${PROJECT_REPO_ROOT}"
echo "WORKSPACE_ROOT=${WORKSPACE_ROOT}"
echo "PWD_AFTER_SOURCE=${PWD}"
# `env` only lists exported names, so this proves the variable is inherited by
# child processes. Launchers spawn launchers, so plain assignment is not enough.
echo "EXPORTED=$(env | grep -c '^WORKSPACE_ROOT=')"
"""


def _write(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)
    return path


def _fastwam_at(root: Path) -> Path:
    """Place a copy of the real FastWAM _common.sh at ``root``."""
    _write(root / "scripts" / "adaptive_gate" / "_common.sh", REAL_COMMON.read_text())
    return root


def _outer_at(root: Path) -> Path:
    _write(root / "scripts" / "adaptive_gate" / "_common.sh", STUB_OUTER_COMMON)
    return root


def _run(probe: Path, fastwam_root: Path, *, workspace_root: str | None):
    env = dict(os.environ)
    env.pop("WORKSPACE_ROOT", None)
    if workspace_root is not None:
        env["WORKSPACE_ROOT"] = workspace_root
    return subprocess.run(
        ["bash", str(probe), str(fastwam_root)],
        capture_output=True,
        text=True,
        env=env,
    )


def _fields(stdout: str) -> dict[str, str]:
    out = {}
    for line in stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            out[key] = value
    return out


@pytest.fixture()
def probe(tmp_path: Path) -> Path:
    path = _write(tmp_path / "probe.sh", PROBE)
    path.chmod(0o755)
    return path


@pytest.fixture()
def primary_layout(tmp_path: Path) -> Path:
    """``<ws>/scripts`` + ``<ws>/FastWAM`` -- the primary checkout shape."""
    ws = tmp_path / "workspace"
    _outer_at(ws)
    return _fastwam_at(ws / "FastWAM")


@pytest.fixture()
def worktree_layout(tmp_path: Path) -> tuple[Path, Path]:
    """``<ws>`` plus a detached ``<parent>/<wt>`` whose parent has no scripts/.

    This is the ``FastWAM-worktrees/<slug>`` shape that used to break.
    """
    ws = _outer_at(tmp_path / "workspace")
    wt = _fastwam_at(tmp_path / "FastWAM-worktrees" / "stage2-example")
    assert not (wt.parent / "scripts").exists(), "fixture must not have a sibling scripts/"
    return ws, wt


def test_real_common_sh_exists():
    assert REAL_COMMON.is_file(), f"missing {REAL_COMMON}"


def test_primary_layout_derives_root_from_parent(probe, primary_layout):
    """Unset WORKSPACE_ROOT keeps the historical parent-directory derivation."""
    result = _run(probe, primary_layout, workspace_root=None)
    assert result.returncode == 0, result.stderr
    fields = _fields(result.stdout)
    assert fields["OUTER_SOURCED"] == "1"
    assert fields["WORKSPACE_ROOT"] == str(primary_layout.parent)
    assert fields["PROJECT_REPO_ROOT"] == str(primary_layout)
    # _common.sh cd's into the FastWAM root; launchers rely on relative paths.
    assert fields["PWD_AFTER_SOURCE"] == str(primary_layout)


def test_worktree_layout_without_workspace_root_fails_closed(probe, worktree_layout):
    """The regression that motivated this work order must stay fixed."""
    _, wt = worktree_layout
    result = _run(probe, wt, workspace_root=None)
    assert result.returncode != 0
    # The message has to name the path it wanted, what it derived, and the fix.
    assert "does not exist" in result.stderr
    assert str(wt.parent / "scripts" / "adaptive_gate" / "_common.sh") in result.stderr
    assert f"PROJECT_REPO_ROOT={wt}" in result.stderr
    assert "export WORKSPACE_ROOT=" in result.stderr
    # It must fail on our explicit check, not on bash's own source error.
    assert "No such file or directory" not in result.stderr


def test_worktree_layout_with_workspace_root_succeeds(probe, worktree_layout):
    ws, wt = worktree_layout
    result = _run(probe, wt, workspace_root=str(ws))
    assert result.returncode == 0, result.stderr
    fields = _fields(result.stdout)
    assert fields["OUTER_SOURCED"] == "1"
    assert fields["WORKSPACE_ROOT"] == str(ws)
    assert fields["PROJECT_REPO_ROOT"] == str(wt)
    assert fields["PWD_AFTER_SOURCE"] == str(wt)


def test_workspace_root_wins_over_parent_derivation(probe, primary_layout, tmp_path):
    """An explicit root must beat the parent even when the parent is valid.

    Without this, FastWAM's _common.sh could silently forget to set the variable
    and the outer stub's ``:-`` default would paper over the bug in the primary
    layout, leaving worktrees broken.
    """
    other = _outer_at(tmp_path / "elsewhere")
    result = _run(probe, primary_layout, workspace_root=str(other))
    assert result.returncode == 0, result.stderr
    fields = _fields(result.stdout)
    assert fields["WORKSPACE_ROOT"] == str(other)
    assert fields["WORKSPACE_ROOT"] != str(primary_layout.parent)


def test_derived_workspace_root_is_exported(probe, primary_layout):
    """A *derived* root must be exported, not just assigned.

    Passing WORKSPACE_ROOT in through the environment would make this vacuous:
    re-assigning an already-exported name keeps the export attribute, so the
    check would pass even without the ``export``. The derived path is the one
    where the distinction is observable, so it is the one pinned here.
    """
    result = _run(probe, primary_layout, workspace_root=None)
    assert result.returncode == 0, result.stderr
    assert _fields(result.stdout)["EXPORTED"] == "1"


def test_caller_assigned_root_reaches_child_launchers(tmp_path, worktree_layout):
    """The case where ``export`` is actually load-bearing.

    Launchers spawn launchers -- ``run_e1_sdr_learning_probe.sh`` ends with
    ``exec bash scripts/adaptive_gate/run_e1_sdr_formal_train.sh`` and
    ``run_e1_train_shared_pilots.sh`` spawns ``run_e1_train_shared.sh``. If a
    wrapper sets WORKSPACE_ROOT as a plain shell variable (not exported) and then
    sources ``_common.sh``, the child would inherit nothing and re-derive the root
    from its own parent directory, which is wrong inside a worktree. Exporting in
    ``_common.sh`` closes that hole regardless of how the caller set the value.
    """
    ws, wt = worktree_layout
    child = _write(
        tmp_path / "child.sh",
        "#!/usr/bin/env bash\necho CHILD_SEES=${WORKSPACE_ROOT:-<unset>}\n",
    )
    caller = _write(
        tmp_path / "caller.sh",
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        # Deliberately assigned without `export`.
        f"WORKSPACE_ROOT={shlex.quote(str(ws))}\n"
        f"source {shlex.quote(str(wt))}/scripts/adaptive_gate/_common.sh\n"
        f"bash {shlex.quote(str(child))}\n",
    )
    env = dict(os.environ)
    env.pop("WORKSPACE_ROOT", None)
    result = subprocess.run(
        ["bash", str(caller)], capture_output=True, text=True, env=env
    )
    assert result.returncode == 0, result.stderr
    assert f"CHILD_SEES={ws}" in result.stdout


def test_nonexistent_workspace_root_is_rejected(probe, worktree_layout, tmp_path):
    ws, wt = worktree_layout
    missing = tmp_path / "definitely-not-here"
    assert not missing.exists()
    result = _run(probe, wt, workspace_root=str(missing))
    assert result.returncode != 0
    assert "not a directory" in result.stderr
    assert str(missing) in result.stderr


def test_workspace_root_pointing_at_a_file_is_rejected(probe, worktree_layout, tmp_path):
    _, wt = worktree_layout
    not_a_dir = _write(tmp_path / "a-file", "")
    result = _run(probe, wt, workspace_root=str(not_a_dir))
    assert result.returncode != 0
    assert "not a directory" in result.stderr


def test_workspace_root_directory_without_scripts_is_rejected(probe, worktree_layout, tmp_path):
    """A readable directory that simply is not the outer repo must still fail."""
    _, wt = worktree_layout
    empty = tmp_path / "empty-dir"
    empty.mkdir()
    result = _run(probe, wt, workspace_root=str(empty))
    assert result.returncode != 0
    assert "does not exist" in result.stderr
    assert str(empty / "scripts" / "adaptive_gate" / "_common.sh") in result.stderr


def test_paths_containing_shell_metacharacters(probe, tmp_path):
    """The real workspace root ends in ``?``; quoting must hold throughout."""
    ws = _outer_at(tmp_path / "weird ? root")
    wt = _fastwam_at(tmp_path / "wt parent" / "worktree with space")
    result = _run(probe, wt, workspace_root=str(ws))
    assert result.returncode == 0, result.stderr
    fields = _fields(result.stdout)
    assert fields["WORKSPACE_ROOT"] == str(ws)
    assert fields["PROJECT_REPO_ROOT"] == str(wt)


def test_failure_uses_return_not_exit(tmp_path, worktree_layout):
    """``_common.sh`` is sourced, so it must ``return`` and let the caller decide.

    A bare ``exit`` would also end an interactive shell that sourced it, and
    would bypass any caller that wraps the source in a conditional.
    """
    _, wt = worktree_layout
    caller = _write(
        tmp_path / "caller.sh",
        "#!/usr/bin/env bash\n"
        "set -uo pipefail\n"
        f"if source {shlex.quote(str(wt))}/scripts/adaptive_gate/_common.sh; then\n"
        "    echo CALLER_SAW=success\n"
        "else\n"
        "    echo CALLER_SAW=failure_rc=$?\n"
        "fi\n"
        "echo CALLER_STILL_RUNNING=1\n",
    )
    env = dict(os.environ)
    env.pop("WORKSPACE_ROOT", None)
    result = subprocess.run(
        ["bash", str(caller)], capture_output=True, text=True, env=env
    )
    # The caller regained control instead of being terminated outright.
    assert "CALLER_STILL_RUNNING=1" in result.stdout
    assert "CALLER_SAW=failure_rc=2" in result.stdout


def test_no_source_before_the_guard(worktree_layout):
    """Static check: nothing may source the outer file ahead of the existence guard.

    Guards the specific shape of the fix, so a future edit cannot reintroduce the
    unguarded ``source "${PROJECT_REPO_ROOT}/../scripts/..."`` line.
    """
    text = REAL_COMMON.read_text()
    assert '"${PROJECT_REPO_ROOT}/../scripts/adaptive_gate/_common.sh"' not in text
    guard_at = text.index("does not exist")
    source_at = text.index("source \"${_ADAPTIVE_GATE_SHARED_COMMON}\"")
    assert guard_at < source_at, "the existence guard must precede the source"
