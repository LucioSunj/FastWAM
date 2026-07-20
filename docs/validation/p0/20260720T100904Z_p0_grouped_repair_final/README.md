# P0 Grouped-Attention Real-Wan Parity

- Status: PASS
- Run ID: 20260720T100904Z_p0_grouped_repair_final
- Decision SHA256: 37595f03dc8b3fb4e54621915d9dc5c1cdb8a0e2f954d63d5f23dfa54cb1bd28
- Test source SHA256: caf8a2e4180153e60f45488d97d75769c47df54540caf5bead2bb0e06a5e2b51
- GPU: NVIDIA A800-SXM4-80GB, 81920 MiB, driver 580.126.09
- Runtime: Python 3.10.20, PyTorch 2.7.1+cu128, CUDA 12.8, BF16

## Scope

This run validates one real Wan2.2/ActionDiT model load with synthetic
batch=1, frames=5, height=64, width=64, horizon=8 inputs. The grouped repair
keeps exactly one MoT.forward and one video loss, but forms reference-shaped
IDM and UNCOND attention groups inside that forward. It passed fused/reference
action, video, and loss parity; attention-mask isolation; trained-parameter
gradient coverage; forced branch inference; and checkpoint provenance.

This is not a training result. P0 did not run E0, E1-S0, endpoint rollout,
Plus-Full evaluation, or any optimizer step.

## Repository Revisions

| Repository | Revision | Dirty |
|---|---|---|
| Outer | aef7bc153bcffecdabd8f8b5e2ab4ab106898032 | false |
| FastWAM | 4cfc6acd6ddfef2cb18b2a829d77306351203448 | false |
| RLinf | aabebda3cd7200870d677f159165ebf564d92d89 | false |
| StarWAM | f6c771fc3be0a9bc271ea4f1531d8ea35efb0ec7 | false |
| starVLA | 235f39929759164f1087204e610ce6c1252b22aa | false |

## Acceptance Evidence

| Check | Result |
|---|---|
| Pytest process exit | 0 |
| JUnit totals | 28 tests, 0 failures, 0 errors, 0 skipped |
| Required real-model cases | all present |
| IDM fused/reference action parity | PASS |
| UNCOND fused/reference action parity | PASS |
| Video and loss parity | PASS |
| Attention-mask isolation | PASS |
| Trained-parameter gradient coverage | PASS |
| One MoT forward and one video loss | PASS |
| Launcher and strict P0 decision | PASS |

The acceptance inputs remained unchanged from the dense-fused validation:
atol=5e-2, rtol=5e-2, the synthetic shape above, BF16, the fused IDM task,
and the committed mask construction. Comparing test source at FastWAM
55510a15... and 4cfc6acd... confirms those tolerance and shape lines are
identical. The repair changed grouped execution semantics, not the acceptance
contract.

## Runtime

- Pytest: 72.58 seconds
- Launcher wall clock: 76 seconds
- Peak sampled GPU memory: 26081 MiB of 81920 MiB
- CUDA device count used: one (CUDA_VISIBLE_DEVICES=0)
- Model: Wan-AI/Wan2.2-TI2V-5B
- Model factory: create_fused_dual_regime_fastwam

The report-only microbenchmark in audit/p0_mode_benchmark.json records grouped
single-MoT, old dense single-MoT, and two-forward reference timings and memory
for the same small input. Performance is not a P0 acceptance criterion.

## Retry History

The first grouped run (20260720T094723Z_p0_grouped_repair) completed all parity
cases and reached 27 passes. Its final checkpoint-provenance test failed while
writing approximately 7.5 GB under /tmp; the traceback contains
PytorchStreamWriter failed writing file. The decision correctly recorded FAIL,
not a numerical failure.

The final run's only deliberate execution change was adding pytest --basetemp
under the experiment filesystem. The resolved model and data configuration is
otherwise identical; only Hydra's automatically generated output_dir timestamp
differs. No tolerance, shape, mask, dtype, solver, or model setting changed.

## Artifact Layout

- Final PASS evidence is at this directory's top level.
- Environment, model inventory, repository SHA records, dense operator trace,
  and report-only mode benchmark are under audit/.
- Minimal diagnostics from the first grouped disk failure are under
  first_grouped_disk_failure/.
- artifact_inventory.tsv records path, size, and SHA256.
- SHA256SUMS.txt verifies every archived file except itself.

P0 proves real-Wan parity only at the committed synthetic acceptance shape. It
does not prove production-training-shape memory or performance, S0 warm-start
parity, a shared endpoint gap, or Gate quality.
