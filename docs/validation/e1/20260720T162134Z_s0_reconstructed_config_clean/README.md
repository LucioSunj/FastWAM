# E1-S0 Warm-Start Parity Validation

Status: **PASS under the user-confirmed legacy config reconstruction**

## Scope

- E0 was skipped by explicit user direction; no E0 PASS was created.
- This run performed only standalone FastWAM-IDM to fused dual-regime forced-IDM parity.
- No endpoint rollout, S-DR pilot, model training, or checkpoint save was started.

## Reconstructed E-I Input

The historical resolved config was not recovered. The user confirmed that the
checkpoint used the default `libero_idm_2cam224_1e-4` recipe, with only the
global batch reduced by half. The repository documents 8 processes x 16 samples
(global batch 128); the reconstructed topology is 4 processes x 16 samples
(global batch 64). With 277,713 transitions and 10 epochs this gives
`ceil(277713 / 64) * 10 = 43,400` steps, matching the checkpoint.

- Reconstruction manifest SHA256: `7475eea1a4cf0691f308bfc4fee2f76d0d9361d1b97c7e768e30bf5d95648937`
- Reconstructed config SHA256: `12be4e401aec1e613f8e0fae8ba3e2bb23102e539f585092d7b834355f5a18af`
- E-I checkpoint SHA256: `f6e29dd6638d19a9e60c87ab387f0cc8d5c75f1ab9262cacd8d269ea1ee43c9c`
- Dataset stats SHA256: `30f81ad7d5076e97323e3328bce003e01a04cb21327b5bacd21bb72846768638`

## Acceptance Result

- Source task: `libero_idm_2cam224_1e-4`
- Target task: `libero_dual_regime_fused_2cam224_1e-4`
- Sample index / seed: `0 / 0`
- Model dtype: BF16
- Video and action inference steps: `20 / 20`
- Scheduler shift: `5.0`
- Tolerances: `atol=5e-4`, `rtol=5e-3`
- Action shape: `[32, 7]`
- Maximum absolute error: `0.0`
- Worst normalized error: `0.0`
- Source and target action SHA256: `5e9f1ba1f322c4f7bcffcc17a66d0f479459c0407d8beafb74ed174a46526767`
- Parity result SHA256: `618380cd431f4e8571a90fe9294f8660211bd81ad28fea8b99af582e353b67c6`
- Contract decision: `PASS`
- Wall clock: `177.342 seconds`
- Peak monitored GPU memory: `15,865 MiB`
- GPU: `NVIDIA A800-SXM4-80GB`

All five repositories were recorded clean in `run_manifest.json`.

## Limitation

The legacy checkpoint contains no embedded FastWAM provenance metadata. Exact
state-schema import and zero-error numerical parity establish compatibility with
the reconstructed default config, but do not independently recover or prove the
missing historical config/stats lineage.

Two earlier orchestration attempts remain in the server-local experiment root
and are not part of this curated archive: one exited before execution because
`/usr/bin/time` was unavailable, and one stopped during dataset initialization
because the isolated worktree lacked its local dataset binding. Neither reached
model parity assertions.
