# Fused Single-Pass Dual-Regime Training

An alternative implementation of the dual-regime action training that lives in
`MetricAdaptiveFastWAM` / `MetricAdaptiveFastWAMJoint`. It trains the same
objective (shared action expert in-distribution for BOTH the base regime and
the joint/idm regime, video loss exactly once) but in **one MoT forward per
step instead of two**, with **no copied parent training code**. The one
`MoT.forward` internally executes two reference-shaped attention groups per
layer so BF16 kernels see the same component shapes as the two-forward
reference. Grouped execution is opt-in; `attention_groups=None`, inference,
and KV-cache paths retain the original behavior.

## Why (critique of the two-forward implementation)

The inherited `training_loss` in the metric-adaptive classes:

1. Runs TWO sequential MoT forwards per step — the main-regime forward
   (`_joint_regime_losses` / `_idm_regime_losses`) plus a separate base-regime
   forward (`_base_regime_action_loss`) that re-processes the first-frame video
   tokens through all 30 blocks even though the main forward already contains
   them.
2. With `share_inputs=true`, the main-regime forward is a line-for-line copy of
   the parent's `training_loss` body ("keep in sync with the parent if it ever
   changes") — a silent-drift liability against the upstream fork. The
   `share_inputs=false` escape hatch removes the drift risk but pays a second
   VAE encode per step. Two code paths exist solely to manage that trade-off.
3. The routing plumbing AND the training helpers are duplicated near-verbatim
   across `fastwam_metric_adaptive.py` and `fastwam_metric_adaptive_joint.py`.

## Key observation

Under the adaptive configs, three facts already hold for the TRAINING forward:

- `video_attention_mask_mode="first_frame_causal"`: first-frame video rows
  attend ONLY first-frame columns inside the full video sequence.
- `fuse_vae_embedding_in_latents=true` + `seperated_timestep=true`: the first
  latent frame is the CLEAN input frame and gets timestep-0 token-wise
  modulation (`wan_video_dit.pre_dit` forces `token_timesteps[:, 0, :] = 0`).
- Frame-0 RoPE coordinates are the same whether the video holds 1 frame or many.

Base-branch inference (`FastWAM.infer_action` → `_predict_action_noise`) runs
the video expert on the clean first frame ALONE at timestep 0, with the tokens
attending only themselves. By the three facts above, the training forward's
first-frame tokens undergo the **layer-exact same computation** — same input
values, same visible attention set, same modulation, same RoPE, same
cross-attention. The training sequence therefore already CONTAINS a faithful
replica of the base branch's conditioning tokens; the second forward of the
inherited implementation recomputes something that is already present.

## Design

Append a second, independently-noised action draft to the logical mixed
sequence and give each draft its own visibility span in one block-structured
global attention mask. The mask remains the source of truth, but each layer
extracts two submatrices and runs exact reference component shapes instead of
sending all tokens through one dense BF16 kernel:

```
joint variant (FusedDualRegimeFastWAMJoint):
  [ noisy_video (S_v) | action_joint (S_a) | action_base (S_a) ]
    - video rows:      video->video mask only (never see any draft)
    - action_joint:    itself + ALL video columns          (FastWAMJoint mask)
    - action_base:     itself + first-frame columns ONLY   (FastWAM mask)

idm variant (FusedDualRegimeFastWAM):
  [ noisy_video (S_v) | cond_video (S_v) | action_idm (S_a) | action_base (S_a) ]
    - blocks internally masked as in FastWAMIDM teacher forcing
    - action_idm:      itself + cond_video block
    - action_base:     itself + cond block's first-frame columns (clean, t=0)
```

The per-layer execution plan is:

1. Main group: `[full main video | main action]`. IDM therefore uses
   `[noisy_video | cond_video | action_idm]`, exactly matching its reference
   forward.
2. Base group: `[clean cond first frame | action_base]`, exactly matching the
   standalone base reference shape.
3. The base video span is read-only. Its Q/K/V and SDPA rows are recomputed so
   the kernel shape stays reference-aligned, but its attention output,
   cross-attention, and FFN are discarded. The next layer reuses the main
   group updated first-frame hidden state.
4. Main and base action separately execute Q/K/V, output projection,
   cross-attention, and FFN. They are never projected as one merged matrix.

Every logical token has exactly one output writer. Validation rejects
out-of-bounds spans, duplicate or missing writers, duplicate experts inside a
group, and groups without an output. The read-only first-frame overlap still
participates in autograd through base-action K/V dependencies.


- Each draft carries its own noise draw and diffusion timestep: per-draft
  `t_mod [B,6,D]` is expanded to token-wise `[B,S,6,D]` and concatenated —
  `MoT._split_modulation` natively supports 4D modulation (the video expert
  already uses it every step). Drafts reuse identical RoPE positions, which is
  unambiguous because they never attend each other.
- Because video tokens never attend action tokens, appending drafts provably
  leaves every video-token output — and hence the video loss — unchanged.
- Video loss comes from the noisy block only (`pred_slice`), computed once.
- All randomness is isolated in `_sample_dual_regime_draws(inputs)`; the
  forward is deterministic given the draws, which is what makes the
  fused-vs-two-forward parity test possible.

### What is inherited vs. new

`FusedDualRegime*` subclass the existing `MetricAdaptive*` classes and override
**only `training_loss`** (plus private helpers). Inference routing
(`infer_action` / `infer_joint` / `infer` / metric / selector / `force_branch`),
`_action_loss_per_sample`, `_compute_video_loss_per_sample`, checkpoint
save/load format, and the `WAMModeAdapter` contract are byte-inherited —
`tests/test_dual_regime_fused.py::test_only_training_loss_is_overridden`
asserts this. Checkpoints are interchangeable with the metric-adaptive classes
(same `mot` / `proprio_encoder` payload).

The pure mask/merge math lives in `dual_regime_masks.py` with zero
fastwam-internal imports, so its unit tests run anywhere torch is installed.

## Cost accounting (LIBERO joint 2cam224: S_v=294, S_f=98, S_a=32)

| per training step | inherited (share_inputs=true) | grouped fused |
|---|---|---|
| MoT forwards | 2 | 1 |
| attention-group tokens, joint | (294+32) + (98+32) = 456 | same two shapes inside one forward |
| attention-group tokens, idm | (588+32) + (98+32) = 750 | same two shapes inside one forward |
| post/cross-attn/FFN writer tokens, joint | 456 | 294+64 = **358 (-21%)** |
| post/cross-attn/FFN writer tokens, idm | 750 | 588+64 = **652 (-13%)** |
| VAE encodes | 1 (2 if `share_inputs=false`) | 1, unconditionally |
| video `pre_dit` calls | 2 (joint) / 3 (idm) | 1 (joint) / 2 (idm) |
| copied parent training code | ~200 lines ("keep in sync") | none (parity is test-enforced instead) |

The grouped implementation deliberately recomputes base first-frame Q/K/V and
SDPA. This trades some of the old dense implementation speed for numerical
equivalence while still avoiding the base video 30-layer post/cross-attention
and FFN shadow path.

## Configs & usage

Model configs: `configs/model/fastwam_dual_regime_fused[_joint].yaml`
(identical to the metric-adaptive ones except `_target_` and the `train:`
block — `share_inputs` no longer exists; use
`action_regime_weight_uncond`, default 1.0).

Task configs: `{libero,robotwin}_dual_regime_fused[_joint]_*` mirror the
metric-adaptive task configs.

```bash
cd FastWAM
python scripts/precompute_text_embeds.py task=libero_dual_regime_fused_2cam224_1e-4
bash scripts/train_zero1.sh 8 task=libero_dual_regime_fused_2cam224_1e-4
```

Loss dictionaries report `loss_video`, the normalized main-regime contribution,
`loss_action_uncond`, and `loss_action_combined`.

For the adaptive gate (RLinf workstream), point the gate config's `wam.task`
at the fused IDM task and set `backbone_kind=idm`. Joint classes remain training
ablations but are not choices exposed by the adaptive gate.

## Constraints

- Requires `fuse_vae_embedding_in_latents=true`, `seperated_timestep=true`
  (same as the inherited dual-regime path) and a `video_attention_mask_mode`
  that isolates first-frame rows (`first_frame_causal` or `per_frame_causal`).
  Violations raise immediately with a pointed message — the exactness of the
  base-regime replica depends on them, and the guard checks the actual mask,
  not the mode string. Temporal `patch_size[0]` must also equal 1 and is checked.
- Model factories reject `action_regime_weight_uncond<=0`; a checkpoint exposed
  to the gate must actually have trained both regimes.

## Verification

```bash
# anywhere torch is installed (no weights/GPU): mask semantics + structure
pytest tests/test_dual_regime_fused.py -v

# on the training server (Wan2.2 weights + GPU): real step + numerical parity
RUN_FASTWAM_MODEL_TESTS=1 pytest tests/test_dual_regime_fused.py -v
```
Run `pytest tests/test_mot_attention_groups.py -q` for tiny FP32 output and
gradient parity, writer/read-only validation, checkpoint backward, and
`attention_groups=None` compatibility.


The heavy group includes `test_fused_matches_two_forward_reference`, which
replays identical noise/timestep draws through the fused forward and through
parent-style separate forwards and asserts per-regime predictions match at the
preregistered BF16 tolerances. The unchanged unseeded case is accompanied by
fixed-seed 0/1/2 regressions. The P0 launcher requires every real-model case to
appear in JUnit with zero failures, errors, and skips; source-file presence
alone cannot produce PASS.
