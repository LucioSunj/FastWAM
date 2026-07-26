"""`imageio` and `safetensors` must not be import-time requirements.

Both used to be module-scope imports that leaked into the whole package:

* ``fastwam/utils/__init__.py`` re-exports ``save_mp4`` from ``video_io``, which
  did ``import imageio`` at module scope. Because nearly every fastwam module
  reaches ``fastwam.utils.logging_config``, that made ``imageio`` a hard
  requirement for importing almost anything -- including
  ``fastwam.models.wan22.action_dit``.
* ``fastwam/models/wan22/helpers/__init__.py`` re-exports ``io``, which did
  ``from safetensors import safe_open`` at module scope, so every model class
  needed ``safetensors`` even when loading a ``.bin`` checkpoint or no
  checkpoint at all.

Both are now imported inside the one function that uses them, which is the
idiom the same file already uses for ``modelscope`` / ``huggingface_hub``
(``ModelConfig.download``).

The tests run each import in a **subprocess** with the dependency blocked via
``sys.modules[name] = None``. That is the only way to assert something about a
*cold* import, and it means these tests behave identically whether or not the
real packages happen to be installed on the machine running them.
"""

import ast
import os
import pathlib
import subprocess
import sys

import pytest


REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
VIDEO_IO = SRC_ROOT / "fastwam" / "utils" / "video_io.py"
HELPERS_IO = SRC_ROOT / "fastwam" / "models" / "wan22" / "helpers" / "io.py"


def _run_blocked(blocked, body):
    """Run `body` in a subprocess where importing `blocked` raises ImportError."""
    preamble = "import sys\n" + "".join(
        f"sys.modules[{name!r}] = None\n" for name in blocked
    )
    env = dict(os.environ)
    # Inherit however *this* process located fastwam (installed or src on path).
    env["PYTHONPATH"] = os.pathsep.join(
        [p for p in sys.path if p] + [str(SRC_ROOT)]
    )
    return subprocess.run(
        [sys.executable, "-c", preamble + body],
        capture_output=True,
        text=True,
        env=env,
        cwd=str(REPO_ROOT),
    )


# --------------------------------------------------------------------------- #
# 1. Cold import must succeed with the optional dependency absent.
# --------------------------------------------------------------------------- #

def test_utils_package_imports_without_imageio():
    proc = _run_blocked(
        ["imageio"],
        "import fastwam.utils as u\n"
        "assert callable(u.save_mp4), u.save_mp4\n"
        "assert callable(u.ensure_dir), u.ensure_dir\n"
        "print('OK')\n",
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_helpers_package_imports_without_safetensors():
    proc = _run_blocked(
        ["safetensors"],
        "import fastwam.models.wan22.helpers as h\n"
        "assert h.ModelConfig is not None\n"
        "assert callable(h.load_state_dict)\n"
        "assert callable(h.hash_model_file)\n"
        "print('OK')\n",
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


def test_action_dit_imports_with_both_dependencies_absent():
    """The regression this change exists to prevent.

    ``action_dit`` pulls ``fastwam.utils.logging_config`` and
    ``fastwam.models.wan22.helpers``; before the fix it therefore needed both
    ``imageio`` and ``safetensors`` even though it uses neither.
    """
    proc = _run_blocked(
        ["imageio", "safetensors"],
        "from fastwam.models.wan22.action_dit import ActionDiT\n"
        "from fastwam.models.wan22.mot import MoT\n"
        "assert ActionDiT is not None and MoT is not None\n"
        "print('OK')\n",
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# --------------------------------------------------------------------------- #
# 2. Calling the feature must fail loudly and actionably, never silently.
# --------------------------------------------------------------------------- #

def test_save_mp4_raises_actionable_error_without_imageio():
    proc = _run_blocked(
        ["imageio"],
        "from fastwam.utils.video_io import save_mp4\n"
        "try:\n"
        "    save_mp4([], 'unused.mp4')\n"
        "except ImportError as exc:\n"
        "    assert exc.__cause__ is not None, 'original ImportError was not chained'\n"
        "    print('MSG:' + str(exc))\n"
        "else:\n"
        "    raise AssertionError('save_mp4 did not raise with imageio absent')\n",
    )
    assert proc.returncode == 0, proc.stderr
    message = proc.stdout.split("MSG:", 1)[1]
    assert "imageio" in message
    assert "pip install" in message
    assert "save_mp4" in message


def test_load_state_dict_raises_actionable_error_without_safetensors():
    proc = _run_blocked(
        ["safetensors"],
        "from fastwam.models.wan22.helpers.io import load_state_dict\n"
        "try:\n"
        "    load_state_dict('nonexistent.safetensors')\n"
        "except ImportError as exc:\n"
        "    assert exc.__cause__ is not None, 'original ImportError was not chained'\n"
        "    print('MSG:' + str(exc))\n"
        "else:\n"
        "    raise AssertionError('load_state_dict did not raise with safetensors absent')\n",
    )
    assert proc.returncode == 0, proc.stderr
    message = proc.stdout.split("MSG:", 1)[1]
    assert "safetensors" in message
    assert "pip install" in message


def test_hash_model_file_also_raises_for_safetensors_input():
    """The second `safe_open` call site must be lazy too, not just the first."""
    proc = _run_blocked(
        ["safetensors"],
        "from fastwam.models.wan22.helpers.io import hash_model_file\n"
        "try:\n"
        "    hash_model_file('nonexistent.safetensors')\n"
        "except ImportError as exc:\n"
        "    print('MSG:' + str(exc))\n"
        "else:\n"
        "    raise AssertionError('hash_model_file did not raise with safetensors absent')\n",
    )
    assert proc.returncode == 0, proc.stderr
    assert "safetensors" in proc.stdout.split("MSG:", 1)[1]


def test_bin_checkpoint_path_never_touches_safetensors(tmp_path):
    """A `.bin`/`.pt` checkpoint must load with `safetensors` absent."""
    torch = pytest.importorskip("torch")
    payload = {"w": torch.zeros(2, 3)}
    ckpt = tmp_path / "weights.bin"
    torch.save(payload, ckpt)
    proc = _run_blocked(
        ["safetensors"],
        "import sys\n"
        "from fastwam.models.wan22.helpers.io import load_state_dict\n"
        f"sd = load_state_dict({str(ckpt)!r})\n"
        "assert list(sd) == ['w'], sd\n"
        "assert tuple(sd['w'].shape) == (2, 3)\n"
        "print('OK')\n",
    )
    assert proc.returncode == 0, proc.stderr
    assert "OK" in proc.stdout


# --------------------------------------------------------------------------- #
# 3. Static guard: stop the module-scope import from creeping back.
# --------------------------------------------------------------------------- #

def _module_level_imported_names(path):
    """Top-level (module-scope) imported root module names, ignoring nested ones."""
    tree = ast.parse(path.read_text())
    names = set()
    for node in tree.body:  # module scope only -- function bodies are not walked
        if isinstance(node, ast.Import):
            names.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module:
                names.add(node.module.split(".")[0])
    return names


@pytest.mark.parametrize(
    "path,forbidden",
    [
        (VIDEO_IO, "imageio"),
        (HELPERS_IO, "safetensors"),
    ],
    ids=["video_io/imageio", "helpers_io/safetensors"],
)
def test_optional_dependency_is_not_imported_at_module_scope(path, forbidden):
    assert path.exists(), path
    assert forbidden not in _module_level_imported_names(path), (
        f"{path.name} imports `{forbidden}` at module scope again. It must stay "
        "inside the function that uses it, otherwise importing fastwam.utils / "
        "fastwam.models.wan22 requires it once more."
    )


@pytest.mark.parametrize(
    "path,required",
    [
        (VIDEO_IO, "imageio"),
        (HELPERS_IO, "safetensors"),
    ],
    ids=["video_io/imageio", "helpers_io/safetensors"],
)
def test_optional_dependency_is_still_imported_somewhere(path, required):
    """Guard against the opposite mistake: deleting the import entirely."""
    assert required in path.read_text(), (
        f"{path.name} no longer references `{required}` at all -- the lazy "
        "import was removed rather than moved."
    )
