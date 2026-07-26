"""Contract tests for the W22 tap-verification harness.

Everything runs against a small fake module tree that mimics the exact
attribute structure `CANDIDATE_FASTWAM_TAPS` names (``vae``,
``video_expert.blocks.N``, ``action_expert.head``) — no Wan weights, no
simulator, no heavy deps. The harness must be trustworthy before it is ever
pointed at a real checkpoint; the real-model run stays NOT-RUN until assets
exist and is then a single CLI command.

W20's own fail-closed behaviour (path resolution) is asserted to *propagate*,
never re-implemented or weakened here.
"""

import json
import subprocess
import sys

import numpy as np  # noqa: F401 - matches the harness's dependency floor
import pytest

torch = pytest.importorskip("torch")
from torch import nn  # noqa: E402

from fastwam.diagnostics import CANDIDATE_FASTWAM_TAPS, DEFAULT_POOL_DIM, TapSpec  # noqa: E402
from fastwam.diagnostics.verify_taps import (  # noqa: E402
    TapVerificationError,
    TapVerificationReport,
    build_synthetic_sample,
    verify_taps,
)


# --------------------------------------------------------------------------- #
# Fake module tree mirroring the CANDIDATE tap module paths
# --------------------------------------------------------------------------- #

class _FakeVAE(nn.Module):
    """Returns a [B, 6, 1, 4, 4] latent through forward (hookable)."""

    def __init__(self, channels=6):
        super().__init__()
        self.lift = nn.Conv2d(3, channels, kernel_size=1)

    def forward(self, x):
        return self.lift(torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))).unsqueeze(2)


class _FakeBlock(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.lin = nn.Linear(dim, dim)

    def forward(self, x):
        return torch.tanh(self.lin(x))


class _FakeVideoExpert(nn.Module):
    def __init__(self, dim=12):
        super().__init__()
        self.blocks = nn.ModuleList([_FakeBlock(dim), _FakeBlock(dim)])


class _FakeActionExpert(nn.Module):
    def __init__(self, dim=12, action_dim=7):
        super().__init__()
        self.head = nn.Linear(dim, action_dim)


class _FakeFastWAM(nn.Module):
    """vae -> video_expert.blocks -> action_expert.head, all via __call__."""

    def __init__(self, dim=12):
        super().__init__()
        self.vae = _FakeVAE()
        self.video_expert = _FakeVideoExpert(dim)
        self.action_expert = _FakeActionExpert(dim)
        self.to_tokens = nn.Linear(6, dim)

    def forward(self, x):
        latent = self.vae(x)                                    # [B, 6, 1, 4, 4]
        tokens = self.to_tokens(latent.flatten(2).transpose(1, 2))  # [B, 16, dim]
        for block in self.video_expert.blocks:
            tokens = block(tokens)
        return self.action_expert.head(tokens)


class _FakeFastWAMSilentVAE(_FakeFastWAM):
    """Reaches the latent through a *method call*, exactly like the real
    ``self.vae.encode(...)`` path — the vae module exists and resolves, but its
    forward hook never fires."""

    def forward(self, x):
        latent = self.vae.lift(
            torch.nn.functional.adaptive_avg_pool2d(x, (4, 4))
        ).unsqueeze(2)  # bypasses self.vae.__call__
        tokens = self.to_tokens(latent.flatten(2).transpose(1, 2))
        for block in self.video_expert.blocks:
            tokens = block(tokens)
        return self.action_expert.head(tokens)


class _FakeFastWAMDoubleFire(_FakeFastWAM):
    """Runs the block stack twice per forward, like a two-step solver loop."""

    def forward(self, x):
        latent = self.vae(x)
        tokens = self.to_tokens(latent.flatten(2).transpose(1, 2))
        for _ in range(2):
            for block in self.video_expert.blocks:
                tokens = block(tokens)
        return self.action_expert.head(tokens)


def _plain_forward(model, sample):
    return model(sample)


def _sample_image(batch=2):
    torch.manual_seed(0)
    return torch.randn(batch, 3, 16, 32)


# --------------------------------------------------------------------------- #
# 1. All taps fire: the report is pinned
# --------------------------------------------------------------------------- #

def test_all_candidate_taps_fire_and_report_is_pinned():
    model = _FakeFastWAM().eval()
    report = verify_taps(
        model,
        CANDIDATE_FASTWAM_TAPS,
        forward_fn=_plain_forward,
        sample=_sample_image(),
        pool_output_dim=8,
    )
    assert isinstance(report, TapVerificationReport)
    assert set(report.taps) == {"vae_latent", "video_block_0", "action_readout"}

    vae = report.taps["vae_latent"]
    assert vae.fired == 1
    assert vae.shape == (2, 6, 1, 4, 4)
    assert vae.feature_dim == 1
    assert vae.pooled_feature_dim == 8
    assert vae.dtype == "torch.float32"
    assert vae.device == "cpu"
    assert vae.shape_varies is False

    block = report.taps["video_block_0"]
    assert block.fired == 1
    assert block.shape == (2, 16, 12)
    assert block.pooled_feature_dim == 8

    readout = report.taps["action_readout"]
    assert readout.fired == 1
    assert readout.site == "input"
    assert readout.shape == (2, 16, 12), (
        "the action head's *input* is the readout state, not its output"
    )
    assert readout.pooled_feature_dim == 8


def test_pooled_feature_dim_matches_probe_default():
    model = _FakeFastWAM().eval()
    report = verify_taps(
        model, CANDIDATE_FASTWAM_TAPS, forward_fn=_plain_forward, sample=_sample_image()
    )
    assert report.pool_output_dim == DEFAULT_POOL_DIM
    assert all(
        firing.pooled_feature_dim == DEFAULT_POOL_DIM
        for firing in report.taps.values()
    )


def test_forward_fn_is_called_exactly_once():
    model = _FakeFastWAM().eval()
    calls = {"n": 0}

    def counting_forward(m, s):
        calls["n"] += 1
        return m(s)

    verify_taps(
        model, CANDIDATE_FASTWAM_TAPS, forward_fn=counting_forward, sample=_sample_image()
    )
    assert calls["n"] == 1


# --------------------------------------------------------------------------- #
# 2. Fail-closed: a silent tap names itself and its module path
# --------------------------------------------------------------------------- #

def test_silent_tap_raises_and_names_exactly_that_tap():
    model = _FakeFastWAMSilentVAE().eval()
    with pytest.raises(TapVerificationError) as info:
        verify_taps(
            model, CANDIDATE_FASTWAM_TAPS, forward_fn=_plain_forward, sample=_sample_image()
        )
    message = str(info.value)
    assert "vae_latent (module_path='vae'" in message
    assert "never fired" in message
    # The taps that did fire must not be blamed.
    silent_part = message.split("Fired counts")[0]
    assert "video_block_0 (module_path" not in silent_part
    assert "action_readout (module_path" not in silent_part
    # The partial report still carries the evidence from the taps that fired.
    partial = info.value.report
    assert partial is not None
    assert set(partial.taps) == {"video_block_0", "action_readout"}
    assert partial.taps["video_block_0"].fired == 1


def test_hooks_are_cleaned_up_even_when_verification_fails():
    model = _FakeFastWAMSilentVAE().eval()
    with pytest.raises(TapVerificationError):
        verify_taps(
            model, CANDIDATE_FASTWAM_TAPS, forward_fn=_plain_forward, sample=_sample_image()
        )
    assert (
        sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
        == 0
    ), "verification failure stranded hooks on the model"


# --------------------------------------------------------------------------- #
# 3. Unknown tap: W20's fail-closed resolution propagates untouched
# --------------------------------------------------------------------------- #

def test_unknown_module_path_propagates_w20_error():
    model = _FakeFastWAM().eval()
    with pytest.raises(AttributeError) as info:
        verify_taps(
            model,
            [TapSpec("typo", "vidoe_expert.blocks.0")],
            forward_fn=_plain_forward,
            sample=_sample_image(),
        )
    message = str(info.value)
    assert "vidoe_expert" in message
    assert "video_expert" in message, "W20's error lists available children"


def test_method_tap_point_propagates_w20_type_error():
    model = _FakeFastWAM().eval()
    with pytest.raises(TypeError, match="not an nn.Module"):
        verify_taps(
            model,
            [TapSpec("bad", "vae.forward")],
            forward_fn=_plain_forward,
            sample=_sample_image(),
        )


# --------------------------------------------------------------------------- #
# 4. Multiple fires per forward are reported, not hidden
# --------------------------------------------------------------------------- #

def test_double_fire_is_reported_with_count_two():
    model = _FakeFastWAMDoubleFire().eval()
    report = verify_taps(
        model, CANDIDATE_FASTWAM_TAPS, forward_fn=_plain_forward, sample=_sample_image()
    )
    assert report.taps["video_block_0"].fired == 2
    assert report.taps["video_block_0"].shape_varies is False
    assert report.taps["vae_latent"].fired == 1
    assert report.taps["action_readout"].fired == 1


# --------------------------------------------------------------------------- #
# 5. Default forward: the model's action-inference path with a synthetic sample
# --------------------------------------------------------------------------- #

class _FakeFastWAMWithInferAction(_FakeFastWAM):
    proprio_dim = 8

    def __init__(self):
        super().__init__()
        self.seen_kwargs = None

    def infer_action(
        self,
        prompt=None,
        input_image=None,
        action_horizon=32,
        proprio=None,
        context=None,
        context_mask=None,
        num_inference_steps=2,
        seed=None,
    ):
        self.seen_kwargs = {
            "prompt": prompt,
            "input_image": input_image,
            "action_horizon": action_horizon,
            "proprio": proprio,
            "context": context,
            "context_mask": context_mask,
            "num_inference_steps": num_inference_steps,
            "seed": seed,
        }
        return {"action": super().forward(input_image)}


def test_default_forward_drives_infer_action_with_synthetic_sample():
    model = _FakeFastWAMWithInferAction().eval()
    report = verify_taps(model, CANDIDATE_FASTWAM_TAPS)  # no forward_fn, no sample
    assert set(report.taps) == {"vae_latent", "video_block_0", "action_readout"}
    assert all(firing.fired == 1 for firing in report.taps.values())

    seen = model.seen_kwargs
    assert seen is not None, "infer_action was never called"
    assert seen["input_image"].shape == (1, 3, 224, 448)
    assert seen["proprio"].shape == (1, 8), "proprio_dim must be read off the model"
    assert seen["context"].shape[-2] == 128
    assert seen["context_mask"].dtype == torch.bool
    assert seen["prompt"] is None


def test_unsupported_kwargs_are_filtered_by_signature():
    # infer_action above has no num_video_frames / rand_device; the builder
    # includes both, so reaching infer_action at all proves the filter worked.
    model = _FakeFastWAMWithInferAction().eval()
    sample = build_synthetic_sample(model, image_hw=(16, 32))
    assert "num_video_frames" in sample and "rand_device" in sample
    verify_taps(model, CANDIDATE_FASTWAM_TAPS, sample=sample)
    assert model.seen_kwargs["input_image"].shape == (1, 3, 16, 32)


def test_default_forward_without_infer_action_requires_a_sample():
    model = _FakeFastWAM().eval()
    with pytest.raises(ValueError, match="sample"):
        verify_taps(model, CANDIDATE_FASTWAM_TAPS)


def test_build_synthetic_sample_is_deterministic():
    model = _FakeFastWAMWithInferAction().eval()
    a = build_synthetic_sample(model, image_hw=(16, 32))
    b = build_synthetic_sample(model, image_hw=(16, 32))
    assert sorted(a) == sorted(b)
    for key in a:
        if isinstance(a[key], torch.Tensor):
            assert torch.equal(a[key], b[key]), key
        else:
            assert a[key] == b[key], key


# --------------------------------------------------------------------------- #
# 6. JSON report: deterministic and round-trippable
# --------------------------------------------------------------------------- #

def test_report_json_is_deterministic_and_roundtrips():
    dumps = lambda r: json.dumps(r.to_dict(), sort_keys=True)
    first = verify_taps(
        _FakeFastWAM().eval(), CANDIDATE_FASTWAM_TAPS,
        forward_fn=_plain_forward, sample=_sample_image(),
    )
    second = verify_taps(
        _FakeFastWAM().eval(), CANDIDATE_FASTWAM_TAPS,
        forward_fn=_plain_forward, sample=_sample_image(),
    )
    assert dumps(first) == dumps(second), "report must not depend on run identity"
    payload = json.loads(dumps(first))
    assert payload["taps"]["vae_latent"]["shape"] == [2, 6, 1, 4, 4]
    assert payload["taps"]["action_readout"]["site"] == "input"
    assert payload["pool_output_dim"] == DEFAULT_POOL_DIM
    assert json.loads(json.dumps(payload, sort_keys=True)) == payload


# --------------------------------------------------------------------------- #
# 7. CLI
# --------------------------------------------------------------------------- #

def test_cli_self_test_exits_zero_via_subprocess():
    proc = subprocess.run(
        [sys.executable, "-m", "fastwam.diagnostics.verify_taps", "--self-test"],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["self_test"] is True
    assert set(payload["taps"]) == {"vae_latent", "video_block_0", "action_readout"}
    assert all(tap["fired"] >= 1 for tap in payload["taps"].values())


def test_cli_self_test_writes_deterministic_report(tmp_path):
    out_a = tmp_path / "a.json"
    out_b = tmp_path / "b.json"
    for out in (out_a, out_b):
        proc = subprocess.run(
            [
                sys.executable, "-m", "fastwam.diagnostics.verify_taps",
                "--self-test", "--out", str(out),
            ],
            capture_output=True, text=True,
        )
        assert proc.returncode == 0, proc.stderr
    assert out_a.read_text() == out_b.read_text()
    assert "timestamp" not in out_a.read_text()


def test_cli_requires_task_and_ckpt_when_not_self_test():
    proc = subprocess.run(
        [sys.executable, "-m", "fastwam.diagnostics.verify_taps"],
        capture_output=True, text=True,
    )
    assert proc.returncode != 0
    assert "--task" in proc.stderr and "--ckpt" in proc.stderr


def test_module_imports_without_heavy_dependencies():
    """The real-model path must import hydra/runtime lazily: bare import of the
    module pulls in neither, so the harness loads in asset-free environments."""
    code = (
        "import sys\n"
        "import fastwam.diagnostics.verify_taps\n"
        "assert 'hydra' not in sys.modules, 'hydra imported eagerly'\n"
        "assert 'transformers' not in sys.modules, 'transformers imported eagerly'\n"
        "assert 'accelerate' not in sys.modules, 'accelerate imported eagerly'\n"
        "assert 'fastwam.runtime' not in sys.modules, 'runtime imported eagerly'\n"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert proc.returncode == 0, proc.stderr


# --------------------------------------------------------------------------- #
# 8. Read-only guarantee: verification must not perturb the model
# --------------------------------------------------------------------------- #

def test_verification_does_not_mutate_the_model():
    torch.manual_seed(0)
    model = _FakeFastWAM().eval()
    x = _sample_image()
    with torch.no_grad():
        before = model(x).clone()
    verify_taps(model, CANDIDATE_FASTWAM_TAPS, forward_fn=_plain_forward, sample=x)
    with torch.no_grad():
        after = model(x).clone()
    assert torch.equal(before, after), "verification changed the model's outputs"
    assert (
        sum(len(m._forward_hooks) + len(m._forward_pre_hooks) for m in model.modules())
        == 0
    ), "verification left hooks behind"
