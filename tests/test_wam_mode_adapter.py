"""Binary adaptive-gate adapter, feature cache and cost contract tests."""
from __future__ import annotations

import os
import ast
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from fastwam.adaptive_gate import (  # noqa: E402
    MODE_ORDER,
    NUM_MODES,
    EncodedWorldState,
    WAMMode,
    WAMModeAdapter,
    coerce_mode,
    default_cost_table,
    explicit_eval_branch,
    dual_regime_schedule_fingerprint,
    inference_solver_contract,
    inference_solver_fingerprint,
    load_cost_table,
    mode_from_index,
    mode_to_branch_steps,
    mode_to_index,
    normalize_cost_table,
    normalized_dual_regime_action_loss,
    save_cost_table,
    sha256_file,
    validate_cost_table,
    validate_action_only_attention_mode,
    validate_dataset_stats_fingerprint,
)


def _schedule_contract():
    return {
        "uncond_weight_schedule": [[0.0, 0.05], [1.0, 1.0]],
        "total_optimizer_steps": 100,
    }


STATS_SHA = "c" * 64


def _with_schedule(provenance):
    provenance = dict(provenance)
    contract = _schedule_contract()
    provenance["dual_regime_training_contract"] = contract
    provenance["schedule_fingerprint"] = dual_regime_schedule_fingerprint(contract)
    provenance["initialization_type"] = "standalone_idm"
    provenance["parent_checkpoint_id"] = "parent-id"
    provenance["parent_checkpoint_sha256"] = "a" * 64
    provenance["parent_config_sha256"] = "b" * 64
    provenance["parent_dataset_stats_sha256"] = provenance.get(
        "dataset_stats_fingerprint", "c" * 64
    )
    return provenance


def test_binary_mode_contract():
    assert NUM_MODES == 2
    assert MODE_ORDER == (WAMMode.UNCOND, WAMMode.IDM)
    for index, mode in enumerate(MODE_ORDER):
        assert mode_from_index(index) is mode
        assert mode_to_index(mode) == index
        assert coerce_mode(mode.value) is mode
        assert coerce_mode(index) is mode
    assert mode_to_branch_steps(WAMMode.UNCOND, inference_steps=20) == ("base", 20)
    assert mode_to_branch_steps(WAMMode.IDM, inference_steps=20) == ("idm", 20)
    with pytest.raises(ValueError):
        coerce_mode("latent")
    for invalid in (True, False, 0.9, 1.9, "0.9"):
        with pytest.raises(ValueError):
            coerce_mode(invalid)
    with pytest.raises(ValueError):
        mode_to_branch_steps(WAMMode.IDM, inference_steps=0)
    assert validate_action_only_attention_mode("first_frame_causal") == "first_frame_causal"
    assert validate_action_only_attention_mode("per_frame_causal") == "per_frame_causal"
    with pytest.raises(ValueError):
        validate_action_only_attention_mode("bidirectional")


def test_solver_fingerprint_binds_video_action_steps_and_shift():
    model = _StubModel()
    base = inference_solver_contract(
        model, video_inference_steps=20, action_inference_steps=20
    )
    shifted = inference_solver_contract(
        model,
        video_inference_steps=20,
        action_inference_steps=20,
        sigma_shift=7.0,
    )
    more_action = inference_solver_contract(
        model, video_inference_steps=20, action_inference_steps=21
    )
    fingerprints = {
        inference_solver_fingerprint(base),
        inference_solver_fingerprint(shifted),
        inference_solver_fingerprint(more_action),
    }
    assert len(fingerprints) == 3


def test_explicit_eval_branch_for_adaptive_and_vanilla_models():
    class Adaptive:
        adaptive_backbone_kind = "idm"

        def infer_action(self, *, force_branch=None):
            pass

        def infer_joint(self, *, force_branch=None):
            pass

    class Vanilla:
        def infer_action(self):
            pass

    class AdaptiveJoint(Adaptive):
        adaptive_backbone_kind = "joint"

    model = Adaptive()
    assert explicit_eval_branch(model, "infer_action", "base") == {"force_branch": "base"}
    assert explicit_eval_branch(model, "infer_joint", "idm", require_video=True) == {
        "force_branch": "idm"
    }
    with pytest.raises(ValueError, match="Video visualization"):
        explicit_eval_branch(model, "infer_joint", "base", require_video=True)
    with pytest.raises(ValueError, match="force_branch"):
        explicit_eval_branch(model, "infer_action", "joint")
    assert explicit_eval_branch(AdaptiveJoint(), "infer_action", "joint") == {
        "force_branch": "joint"
    }
    assert explicit_eval_branch(Vanilla(), "infer_action", "anything") == {}


def test_cost_tables_are_strict(tmp_path):
    assert default_cost_table(20) == {"uncond": 0.15, "idm": 1.0}
    assert normalize_cost_table({"uncond": 2.0, "idm": 10.0}) == pytest.approx(
        {"uncond": 0.2, "idm": 1.0}
    )
    for bad in (
        {"uncond": 0.2},
        {"uncond": 0.2, "idm": 0.9},
        {"uncond": 1.0, "idm": 1.0},
        {"uncond": float("nan"), "idm": 1.0},
    ):
        with pytest.raises(ValueError):
            validate_cost_table(bad)

    path = str(tmp_path / "cost.yaml")
    save_cost_table(
        path,
        normalized={"uncond": 0.2, "idm": 1.0},
        raw={
            "flops": {"uncond": 20.0, "idm": 100.0},
            "latency_ms": {"uncond": 3.0, "idm": 10.0},
        },
    )
    assert load_cost_table(path) == pytest.approx({"uncond": 0.2, "idm": 1.0})
    assert load_cost_table(path, source="latency") == pytest.approx(
        {"uncond": 0.3, "idm": 1.0}
    )
    assert load_cost_table(None) is None
    with pytest.raises(FileNotFoundError):
        load_cost_table(str(tmp_path / "missing.yaml"))
    with pytest.raises(ValueError):
        load_cost_table(path, source="energy")


def test_normalized_action_objective_preserves_scale():
    combined, idm, uncond = normalized_dual_regime_action_loss(
        torch.tensor(2.0), torch.tensor(4.0), 1.0
    )
    assert combined.item() == pytest.approx(3.0)
    assert combined.item() == pytest.approx(idm.item() + uncond.item())
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            normalized_dual_regime_action_loss(torch.tensor(1.0), torch.tensor(1.0), bad)


class _StubVAEModel:
    z_dim = 8


class _StubVAE:
    model = _StubVAEModel()


class _StubScheduler:
    shift = 5.0
    num_train_timesteps = 1000


class _StubModel:
    adaptive_regimes = ("uncond", "idm")
    adaptive_backbone_kind = "idm"

    def __init__(self):
        self.vae = _StubVAE()
        self.action_expert = type("A", (), {"action_dim": 7})()
        self.torch_dtype = torch.bfloat16
        self.device = torch.device("cpu")
        self.proprio_dim = None
        self.encode_calls = 0
        self.calls = []
        self.infer_video_scheduler = _StubScheduler()
        self.infer_action_scheduler = _StubScheduler()

    def _encode_input_image_latents_tensor(self, image):
        assert image.dtype == torch.bfloat16
        self.encode_calls += 1
        values = torch.arange(8 * 4 * 4, dtype=torch.float32).reshape(1, 8, 1, 4, 4)
        return values.to(torch.bfloat16)

    def infer_action(
        self,
        *,
        prompt,
        input_image,
        first_frame_latents,
        action_horizon,
        num_video_frames,
        proprio=None,
        context=None,
        context_mask=None,
        num_inference_steps=20,
        sigma_shift=None,
        seed=None,
        force_branch=None,
        return_routing_info=False,
    ):
        self.calls.append(
            {
                "branch": force_branch,
                "steps": num_inference_steps,
                "latents": first_frame_latents,
                "horizon": action_horizon,
            }
        )
        return {
            "action": torch.zeros(action_horizon, 7),
            "_routing": {"selected_branch": force_branch},
        }


def _adapter(model=None, **kwargs):
    kwargs.setdefault("allow_unloaded_model", True)
    kwargs.setdefault("context_len", 8)
    return WAMModeAdapter(
        model or _StubModel(),
        backbone_kind="idm",
        num_video_frames=9,
        generation_horizon=32,
        inference_steps=20,
        **kwargs,
    )


def test_adapter_rejects_joint_and_unverified_legacy_checkpoint():
    with pytest.raises(ValueError):
        WAMModeAdapter(
            _StubModel(), backbone_kind="joint", num_video_frames=9, generation_horizon=32
        )
    legacy = _StubModel()
    legacy._loaded_checkpoint_provenance = None
    with pytest.raises(ValueError):
        _adapter(legacy)
    with pytest.warns(RuntimeWarning):
        _adapter(legacy, allow_legacy_checkpoint=True)
    with pytest.raises(ValueError, match="loaded dual-regime checkpoint"):
        WAMModeAdapter(
            _StubModel(), backbone_kind="idm", num_video_frames=9,
            generation_horizon=32, context_len=8,
        )
    wrong_live = _StubModel()
    wrong_live.adaptive_backbone_kind = "joint"
    with pytest.raises(ValueError, match="Live model"):
        _adapter(wrong_live)


def test_adapter_validates_cost_profile_metadata_and_resolution(tmp_path):
    model = _StubModel()
    model._loaded_checkpoint_provenance = _with_schedule({
        "schema_version": 2,
        "adaptive_regimes": ["uncond", "idm"], "checkpoint_id": "abc", "task": "libero",
        "action_regime_weight_uncond": 1.0,
        "dual_regime_optimizer_steps": 12,
        "dataset_stats_fingerprint": STATS_SHA,
    })
    model._loaded_checkpoint_fingerprint = "abc"
    solver_fingerprint = inference_solver_fingerprint(
        inference_solver_contract(
            model,
            video_inference_steps=20,
            action_inference_steps=20,
        )
    )
    path = str(tmp_path / "profile.yaml")
    save_cost_table(
        path,
        normalized={"uncond": 0.2, "idm": 1.0},
        meta={
            "task": "libero",
            "backbone_kind": "idm",
            "ckpt_fingerprint": "abc",
            "inference_steps": 20,
            "solver_fingerprint": solver_fingerprint,
            "num_video_frames": 9,
            "action_horizon": 32,
            "height": 64,
            "width": 64,
            "context_len": 8,
            "model_dtype": "torch.bfloat16",
            "proprio_dim": None,
            "device_name": "cpu",
        },
    )
    adapter = _adapter(
        model, task="libero", cost_table_path=path,
        dataset_stats_fingerprint=STATS_SHA,
    )
    adapter.act(
        input_image=torch.rand(1, 3, 64, 64),
        mode=WAMMode.UNCOND,
        context=torch.randn(1, 8, 32),
        context_mask=torch.ones(1, 8, dtype=torch.bool),
    )
    bad_resolution_adapter = _adapter(
        model, task="libero", cost_table_path=path,
        dataset_stats_fingerprint=STATS_SHA,
    )
    with pytest.raises(ValueError, match="resolution"):
        bad_resolution_adapter.act(
            input_image=torch.rand(1, 3, 32, 64),
            mode=WAMMode.UNCOND,
            context=torch.randn(1, 8, 32),
            context_mask=torch.ones(1, 8, dtype=torch.bool),
        )

    with pytest.raises(ValueError, match="task"):
        _adapter(
            model, task="robotwin", cost_table_path=path,
            dataset_stats_fingerprint=STATS_SHA,
        )

    with pytest.raises(ValueError, match="generation_horizon override"):
        adapter.act(
            input_image=torch.rand(1, 3, 64, 64),
            mode=WAMMode.IDM,
            generation_horizon=16,
            context=torch.randn(1, 8, 32),
            context_mask=torch.ones(1, 8, dtype=torch.bool),
        )

    untrained = _StubModel()
    untrained._loaded_checkpoint_provenance = _with_schedule({
        "schema_version": 2, "checkpoint_id": "untrained",
        "adaptive_regimes": ["uncond", "idm"], "task": "libero",
        "action_regime_weight_uncond": 0.0,
        "dual_regime_optimizer_steps": 1,
        "dataset_stats_fingerprint": STATS_SHA,
    })
    with pytest.raises(ValueError, match="positive"):
        _adapter(untrained, task="libero")

    s0 = _StubModel()
    s0._loaded_checkpoint_provenance = _with_schedule({
        "schema_version": 2,
        "checkpoint_id": "s0",
        "adaptive_regimes": ["uncond", "idm"],
        "task": "libero",
        "action_regime_weight_uncond": 0.05,
        "dual_regime_optimizer_steps": 0,
        "dataset_stats_fingerprint": STATS_SHA,
    })
    with pytest.raises(ValueError, match="untrained S0"):
        _adapter(s0, task="libero")

    with pytest.raises(ValueError, match="Dataset-stats fingerprint"):
        _adapter(
            model,
            task="libero",
            dataset_stats_fingerprint="different-stats",
        )

    scratch = _StubModel()
    scratch._loaded_checkpoint_provenance = _with_schedule(
        {
            "schema_version": 2,
            "checkpoint_id": "scratch",
            "adaptive_regimes": ["uncond", "idm"],
            "task": "libero",
            "action_regime_weight_uncond": 1.0,
            "dual_regime_optimizer_steps": 12,
            "dataset_stats_fingerprint": STATS_SHA,
        }
    )
    scratch._loaded_checkpoint_provenance["initialization_type"] = "scratch"
    with pytest.raises(ValueError, match="standalone IDM"):
        _adapter(scratch, task="libero", dataset_stats_fingerprint=STATS_SHA)

    malformed = _StubModel()
    malformed._loaded_checkpoint_provenance = {
        "adaptive_regimes": ["uncond", "idm"],
        "task": "libero",
        "action_regime_weight_uncond": 1.0,
        "dataset_stats_fingerprint": STATS_SHA,
    }
    with pytest.raises(ValueError, match="schema_version"):
        _adapter(malformed, task="libero")
    with pytest.warns(RuntimeWarning, match="unsupported schema"):
        _adapter(malformed, task="libero", allow_legacy_checkpoint=True)


def test_adapter_validates_proprio_and_context_contracts():
    model = _StubModel()
    model.proprio_dim = 5
    adapter = _adapter(model)
    common = {
        "input_image": torch.rand(1, 3, 64, 64),
        "mode": WAMMode.UNCOND,
        "context": torch.randn(1, 8, 32),
        "context_mask": torch.ones(1, 8, dtype=torch.bool),
    }
    with pytest.raises(ValueError, match="proprio is required"):
        adapter.act(**common)
    with pytest.raises(ValueError, match="proprio last dim"):
        adapter.act(**common, proprio=torch.randn(1, 4))
    adapter.act(**common, proprio=torch.randn(1, 5))
    with pytest.raises(ValueError, match="context length"):
        adapter.act(
            input_image=common["input_image"], mode=WAMMode.UNCOND,
            proprio=torch.randn(1, 5), context=torch.randn(1, 7, 32),
            context_mask=torch.ones(1, 7, dtype=torch.bool),
        )


def test_checkpoint_provenance_roundtrip_sets_stable_fingerprint(tmp_path):
    pytest.importorskip("imageio", reason="FastWAM model utilities require imageio")
    from fastwam.models.wan22.fastwam import FastWAM

    class Tiny:
        adaptive_regimes = ("uncond", "idm")
        adaptive_backbone_kind = "idm"

        def __init__(self):
            self.mot = torch.nn.Linear(2, 2)
            self.action_expert = type("A", (), {"action_dim": 7})()
            self.vae = _StubVAE()
            self.proprio_dim = None
            self.proprio_encoder = None
            self.torch_dtype = torch.bfloat16

    path = tmp_path / "weights.pt"
    source = Tiny()
    with pytest.raises(ValueError, match="dataset_stats_fingerprint"):
        FastWAM.save_checkpoint(source, str(path), step=3)
    source.dataset_stats_fingerprint = "stats-sha256"
    source.action_regime_weight_uncond = 1.0
    source.dual_regime_optimizer_steps = 3
    source.dual_regime_training_contract = _schedule_contract()
    FastWAM.save_checkpoint(source, str(path), step=3)
    target = Tiny()
    payload = FastWAM.load_checkpoint(target, str(path))
    checkpoint_id = payload["fastwam_provenance"]["checkpoint_id"]
    assert target._loaded_checkpoint_fingerprint == checkpoint_id
    assert target._loaded_checkpoint_provenance["adaptive_regimes"] == ["uncond", "idm"]
    assert target._loaded_checkpoint_provenance["dataset_stats_fingerprint"] == "stats-sha256"
    assert target.dual_regime_optimizer_steps == 3

    malformed_payload = torch.load(path, weights_only=False)
    malformed_payload["fastwam_provenance"].pop("checkpoint_id")
    malformed_path = tmp_path / "malformed.pt"
    torch.save(malformed_payload, malformed_path)
    with pytest.raises(ValueError, match="Malformed FastWAM checkpoint provenance"):
        FastWAM.load_checkpoint(Tiny(), str(malformed_path))

    class VanillaTiny(Tiny):
        adaptive_regimes = ()
        adaptive_backbone_kind = None

    with pytest.raises(ValueError, match="adaptive regimes"):
        FastWAM.load_checkpoint(VanillaTiny(), str(path))


def test_adaptive_eval_stats_are_bound_to_checkpoint(tmp_path):
    stats_path = tmp_path / "dataset_stats.json"
    stats_path.write_text('{"action": {"mean": [0]}}')
    model = _StubModel()
    model._loaded_checkpoint_provenance = {
        "adaptive_regimes": ["uncond", "idm"],
        "dataset_stats_fingerprint": sha256_file(stats_path),
    }
    assert validate_dataset_stats_fingerprint(model, stats_path) == sha256_file(
        stats_path
    )
    stats_path.write_text('{"action": {"mean": [1]}}')
    with pytest.raises(ValueError, match="do not match"):
        validate_dataset_stats_fingerprint(model, stats_path)


def test_idm_action_only_uses_explicit_parent_dispatch_without_router_recursion(monkeypatch):
    pytest.importorskip("imageio", reason="FastWAM model utilities require imageio")
    from fastwam.models.wan22.fastwam_idm import FastWAMIDM

    captured = {}

    class Tiny:
        def infer_joint(self, **kwargs):
            raise AssertionError("dynamic dispatch would re-enter the adaptive router")

    def fake_parent_infer_joint(self, **kwargs):
        captured.update(kwargs)
        return {"action": torch.zeros(4, 7)}

    monkeypatch.setattr(FastWAMIDM, "infer_joint", fake_parent_infer_joint)

    cached = torch.randn(1, 8, 1, 2, 2)
    out = FastWAMIDM.infer_action(
        Tiny(),
        prompt=None,
        input_image=torch.rand(1, 3, 16, 16),
        action_horizon=4,
        num_video_frames=5,
        first_frame_latents=cached,
        context=torch.randn(1, 2, 8),
        context_mask=torch.ones(1, 2, dtype=torch.bool),
    )
    assert out["action"].shape == (4, 7)
    assert captured["decode_video"] is False
    assert captured["first_frame_latents"] is cached


def test_video_seed_is_independent_but_action_seed_stays_paired():
    pytest.importorskip("imageio", reason="FastWAM model utilities require imageio")
    from fastwam.models.wan22.fastwam import FastWAM

    assert FastWAM._video_seed(123) != 123
    assert FastWAM._video_seed(123) == FastWAM._video_seed(123)
    assert FastWAM._video_seed(None) is None


def test_idm_action_source_uses_explicit_parent_dispatch():
    source = Path(__file__).parents[1] / "src/fastwam/models/wan22/fastwam_idm.py"
    tree = ast.parse(source.read_text(encoding="utf-8"))
    infer_action = next(
        node for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "infer_action"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "infer_joint"
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "FastWAMIDM"
        for node in ast.walk(infer_action)
    )


def test_world_state_preserves_spatial_layout_and_dtype():
    adapter = _adapter()
    state = adapter.encode_world_state(torch.rand(1, 3, 64, 64))
    assert isinstance(state, EncodedWorldState)
    assert state.world_feat.shape == (adapter.world_feat_dim,) == (40,)
    assert state.world_feat.dtype == torch.float32
    assert state.first_frame_latents.dtype == torch.bfloat16
    # The first 4*C entries are a 2x2 spatial pool and therefore are not all equal.
    assert state.world_feat[:32].unique().numel() > 4


@pytest.mark.parametrize(
    "mode,branch,video_steps",
    [(WAMMode.UNCOND, "base", None), (WAMMode.IDM, "idm", 20)],
)
def test_act_encodes_once_and_reuses_exact_latent(mode, branch, video_steps):
    adapter = _adapter()
    image = torch.rand(1, 3, 64, 64)
    out = adapter.act(
        input_image=image,
        mode=mode,
        context=torch.randn(1, 8, 32),
        context_mask=torch.ones(1, 8, dtype=torch.bool),
    )
    assert adapter.model.encode_calls == 1
    assert adapter.model.calls[-1]["latents"] is not None
    assert adapter.model.calls[-1]["branch"] == branch
    assert adapter.model.calls[-1]["steps"] == 20
    assert out["aux"]["video_inference_steps"] == video_steps
    assert out["action_chunk"].shape == (32, 7)


def test_explicit_encoded_state_avoids_any_second_encode():
    adapter = _adapter()
    image = torch.rand(1, 3, 64, 64)
    state = adapter.encode_world_state(image)
    adapter.act(
        input_image=image,
        mode=WAMMode.IDM,
        encoded_state=state,
        context=torch.randn(1, 8, 32),
        context_mask=torch.ones(1, 8, dtype=torch.bool),
    )
    assert adapter.model.encode_calls == 1
    assert adapter.model.calls[-1]["latents"] is state.first_frame_latents


RUN_MODEL = os.environ.get("RUN_FASTWAM_MODEL_TESTS", "0") == "1"


@pytest.mark.skipif(not RUN_MODEL, reason="set RUN_FASTWAM_MODEL_TESTS=1 (Wan weights + GPU)")
class TestRealModel:
    @pytest.fixture(scope="class")
    def adapter(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA is unavailable")
        from hydra import compose, initialize_config_dir
        from hydra.utils import instantiate

        task = os.environ.get("FASTWAM_TEST_TASK", "libero_dual_regime_fused_2cam224_1e-4")
        configs_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "configs"))
        with initialize_config_dir(version_base="1.3", config_dir=configs_dir):
            cfg = compose(config_name="train", overrides=[f"task={task}"])
        model = instantiate(cfg.model, model_dtype=torch.bfloat16, device="cuda")
        model.eval().requires_grad_(False)
        return WAMModeAdapter(
            model,
            backbone_kind="idm",
            num_video_frames=9,
            generation_horizon=32,
            inference_steps=20,
            default_seed=0,
            context_len=128,
            allow_unloaded_model=True,
        )

    def _obs(self, adapter):
        proprio = None
        if adapter.model.proprio_dim is not None:
            proprio = torch.randn(1, int(adapter.model.proprio_dim))
        return {
            "input_image": torch.rand(1, 3, 224, 448),
            "context": torch.randn(1, 128, int(adapter.model.action_expert.text_dim)),
            "context_mask": torch.ones(1, 128, dtype=torch.bool),
            "proprio": proprio,
        }

    @pytest.mark.parametrize("mode", list(MODE_ORDER))
    def test_each_mode_runs(self, adapter, mode):
        out = adapter.act(mode=mode, **self._obs(adapter))
        assert out["action_chunk"].shape[0] == 32
