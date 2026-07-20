# P0 Real-Wan Fused Parity Validation

- Status: `FAIL`
- Validation evidence ID: `20260720T023208Z`
- Real launcher run ID: `p0_real_20260720T025557Z`
- GPU: NVIDIA A800-SXM4-80GB, driver 580.126.09
- Runtime: Python 3.10.20, PyTorch 2.7.1+cu128, BF16, CUDA device 0

## Repository revisions

| Repository | Revision |
|---|---|
| Outer | `1cd2407ddcbb6c87b0db0c395a7034154b41264a` |
| FastWAM | `55510a15bfdc052ad53d26a3d2bf7cbb5368a093` |
| RLinf | `aabebda3cd7200870d677f159165ebf564d92d89` |
| StarWAM | `f6c771fc3be0a9bc271ea4f1531d8ea35efb0ec7` |
| starVLA | `235f39929759164f1087204e610ce6c1252b22aa` |

The revisions matched the preregistered values. The worktree was not clean:
the outer repository had a tracked `.DS_Store` deletion and FastWAM had
pre-existing tracked and untracked changes. Validation proceeded only after
explicit user authorization. The pre- and post-P0 status captures are
byte-identical.

## Results

| Check | Result |
|---|---|
| `tests/test_dual_regime_p0.py -q` | 8 passed |
| Fused lightweight group | 17 passed, 8 expected real-model skips |
| Real-model fused group | 24 passed, 1 failed, 0 skipped |
| Attention-mask isolation | PASS |
| Trained-parameter gradient coverage | PASS |
| Single MoT forward and one video loss | PASS |
| Forced low/high branch inference | PASS |
| UNCOND/base fused-reference parity | FAIL, max absolute difference 0.09375 |
| IDM and video/loss parity | Not reached after the base assertion failed |
| Launcher exit | 1 |

The failing assertion used the committed tolerances unchanged:
`atol=5e-2, rtol=5e-2`. No test shape, mask, model configuration, dtype, or
acceptance threshold was modified. There was no CUDA OOM and no real-model
test was skipped.

The launcher stopped on the pytest failure before it could write
`decision.json` or `run_manifest.json`. Neither file was synthesized after the
failure. The actual launcher-resolved configuration is preserved as
`resolved_config.yaml`.

## Runtime evidence

- Pytest time: 342.86 seconds
- Launcher wall-clock time: 344.466 seconds
- Peak sampled GPU memory: 26055 MiB of 81920 MiB
- GPU sampling interval: 0.5 seconds
- Full stdout/stderr: `p0_real_stdout_stderr.log`
- GPU samples: `p0_gpu_monitor.csv`
- Artifact audit: `artifact_audit.txt`
- Evidence hashes: `SHA256SUMS.txt`

Do not proceed to E0 or E1-S0 on this result. Diagnose and fix the base
fused-reference discrepancy, then rerun P0 without changing its acceptance
contract.
