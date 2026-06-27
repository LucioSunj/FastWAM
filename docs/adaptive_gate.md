# Adaptive-prediction gate (fastwam side)

An RL-trained controller that, at each action-chunk step, picks how much
forward-prediction compute the frozen world-action model spends:

| mode | meaning | code path (frozen model) |
|---|---|---|
| **SKIP** | no future prediction (reactive); action conditions on current-context latent only | `force_branch="base"` → `FastWAM.infer_action` (first-frame, no video denoise) |
| **LATENT** | run video branch a few steps (`k_lo`), condition action on intermediate self-generated latent (no pixel decode) | `force_branch=<joint|idm>` → `infer_action(num_inference_steps=k_lo)` |
| **FULL** | full video schedule (`k_hi`), condition action on refined self-generated latent | `force_branch=<joint|idm>` → `infer_action(num_inference_steps=k_hi)` |

The gate is trained with RLinf (GRPO first, PPO later); **fastwam stays frozen**;
only the gate trains. This doc covers the **fastwam side** (Milestone 1). The gate
policy, env/rollout wiring, reward, and RL configs live in the RLinf repo.

## Backbone requirement (one interface, two backbones)

`WAMModeAdapter` needs a **dual-regime checkpoint** loaded into a routing model so
the SAME action expert is in-distribution for both SKIP and future conditioning:

- `backbone_kind="joint"` → a `MetricAdaptiveFastWAMJoint` checkpoint (routes base/joint)
- `backbone_kind="idm"`   → a `MetricAdaptiveFastWAM` checkpoint (routes base/idm)

Both are driven through the identical `WAMModeAdapter.act(...)` interface (decision
#2), used the same way in training and inference. A vanilla base `FastWAM`
checkpoint supports SKIP only (LATENT/FULL would be OOD) — train a dual-regime
checkpoint first (see `docs/metric_adaptive*.md`).

## API

```python
from fastwam.adaptive_gate import WAMModeAdapter, WAMMode

adapter = WAMModeAdapter(
    model,                       # frozen MetricAdaptiveFastWAMJoint / MetricAdaptiveFastWAM
    backbone_kind="joint",       # or "idm"
    num_video_frames=9, action_horizon=32,
    k_lo=4, k_hi=20,             # TODO: make runtime variables (decision #3)
    cost_table_path="configs/adaptive_gate/wam_cost.yaml",
)

world_feat = adapter.encode_world_feat(input_image)         # cheap, once, [z_dim]
mode = gate(world_feat, proprio)                            # RLinf policy decides
out = adapter.act(input_image=img, proprio=proprio,
                  context=ctx, context_mask=mask,
                  mode=mode, world_feat=world_feat)
# out = {action_chunk [Ta,a_dim], world_feat [z_dim], cost (FULL=1), aux{mode,branch,steps,routing}}
```

No-leakage: the conditioned future is the model's own self-generated latent (the
dual-regime model denoises from noise; only frame-0 is the encoded current image).
LATENT/FULL never decode pixels. `force_branch` bypasses the model's internal
PolicyEntropy probe so the gate is the sole decision-maker.

## Cost / reward unit (decision #6)

The reward's per-step compute penalty is `-lambda * cost(mode)`, `cost(FULL)=1`.
"Cost" = **relative FLOPs** of the mode (hardware-independent; the quantity the
gate economizes). The dominant variable is the video denoising loop
(SKIP=0, LATENT=`k_lo`, FULL=`k_hi` steps) on top of a fixed encode+action cost:

    total(mode) ≈ C_fixed + C_video_step · video_steps(mode);  cost = total/total(FULL)

- **Measured (preferred):** `scripts/profile_wam_modes.py` runs each mode once with
  a FLOP counter (and times latency), normalizes by FULL, writes a cost YAML.
- **Analytical fallback** (`default_cost_table`, used when no file): a single
  `fixed_fraction` knob; see `configs/adaptive_gate/wam_cost.yaml`.

Latency is recorded for reference but the reward defaults to FLOPs. A flat per-step
constant is intentionally avoided — it would erase LATENT's genuine saving over FULL.

## Files

- `src/fastwam/adaptive_gate/modes.py` — `WAMMode`, `MODE_ORDER`, mode→(branch,steps).
- `src/fastwam/adaptive_gate/cost.py` — cost table (default / load / save / normalize).
- `src/fastwam/adaptive_gate/wam_mode_adapter.py` — `WAMModeAdapter`.
- `scripts/profile_wam_modes.py` — FLOPs/latency profiler → cost YAML.
- `configs/adaptive_gate/wam_cost.yaml` — analytical default cost table.
- `tests/test_wam_mode_adapter.py` — modes/cost/dispatch (pure) + gated real-model tests.

## Run / verify (Milestone 1)

```bash
cd FastWAM
# pure-logic tests (anywhere with torch + pyyaml)
pytest tests/test_wam_mode_adapter.py -v -k "not TestRealModel"

# profile real per-mode FLOPs (needs Wan weights + GPU)
python scripts/profile_wam_modes.py --task libero_metric_adaptive_joint_2cam224_1e-4 \
  --backbone-kind joint --out configs/adaptive_gate/wam_cost_libero_joint.yaml

# real-model checks incl. "SKIP reproduces base-branch fastwam"
RUN_FASTWAM_MODEL_TESTS=1 FASTWAM_TEST_TASK=libero_metric_adaptive_joint_2cam224_1e-4 \
  pytest tests/test_wam_mode_adapter.py::TestRealModel -v
```

## Roadmap

- **M1 (this) — done:** `WAMModeAdapter` (3 modes, both backbones, one interface) + FLOPs profiler + cost; SKIP reproduces fastwam.
- **M2 (RLinf):** gate policy (3-way categorical MLP) registered as a custom RLinf policy; env/rollout wiring (LIBERO + RoboTwin) calling the adapter; multi-component reward (terminal success; `-lambda*cost`; optional dense agreement-with-FULL) with per-component logging; forced-mode smoke tests.
- **M3:** oracle-label generation + BC warm-start.
- **M4:** GRPO training (LIBERO then RoboTwin); collapse checks + per-setting (in-domain/OOD) mode-usage logging; then PPO config (TODO).
