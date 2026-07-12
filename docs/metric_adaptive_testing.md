# Dual-regime verification

Run lightweight logic tests locally:

```bash
PYTHONPATH=src pytest -q \
  tests/test_metric_adaptive_routing.py \
  tests/test_metric_adaptive_training.py \
  tests/test_dual_regime_fused.py \
  tests/test_wam_mode_adapter.py \
  tests/test_gate_oracle.py \
  tests/test_trainer_forward_contract.py
```

Before training, run the opt-in real-model tests on a GPU host with Wan2.2
assets installed:

```bash
RUN_FASTWAM_MODEL_TESTS=1 FASTWAM_TEST_TASK=libero_dual_regime_fused_2cam224_1e-4 \
  PYTHONPATH=src pytest -q tests/test_dual_regime_fused.py tests/test_wam_mode_adapter.py
```

Once `RUN_FASTWAM_MODEL_TESTS=1` is set, construction or model failures are test
failures rather than broad skips. Only explicitly unavailable dependencies,
assets or CUDA may skip before construction begins.

Required checks before a long run:

1. Fused versus separate-forward IDM and UNCOND predictions agree within the
   documented BF16 tolerance when replaying identical draws.
2. Exactly one video loss and one fused MoT forward occur per step.
3. Every trainable MoT/proprio parameter receives a gradient.
4. The action objective equals
   `(L_idm + w_uncond*L_uncond)/(1+w_uncond)`; factories reject zero,
   negative and non-finite UNCOND weights.
5. Training enters through the accelerator-prepared model `forward`, preserving
   DDP/DeepSpeed reducer hooks.
6. `UNCOND` and `IDM` forced-mode action rollouts both complete; IDM action-only
   inference performs no VAE pixel decode.
7. The gate and inference consume the same cached BF16 first-frame latent, so
   there is one VAE encode per policy decision.
8. A measured cost file matches checkpoint id, task, IDM backbone, inference
   steps, video frames, generation horizon and input resolution.

During the first short GPU run, log `loss_video`, `loss_action_idm`,
`loss_action_uncond`, `loss_action_combined`, gradient norm and memory. With
`w_uncond=1`, each action regime contributes half of `lambda_action`; this keeps
the original action/video scale rather than doubling it.
