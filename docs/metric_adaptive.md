# Metric-Adaptive FastWAM

This extension adds inference-time selection between the basic `fastwam`
inference behavior and the `fastwam_idm` inference behavior without modifying
the original model, training, or evaluation files.

## Design

The primary implementation uses inheritance:

- `MetricAdaptiveFastWAM` inherits from `FastWAMIDM`.
- The `idm` branch calls `FastWAMIDM.infer_*` on the current instance.
- The `base` branch explicitly calls `FastWAM.infer_*` on the current instance.
- A `Metric` computes one scalar score from input, model output, or repeated
  probe-policy samples.
- A `Selector` maps that score to either `base` or `idm`.
- Existing evaluation code still calls `load_checkpoint()`, `infer_action()`,
  `infer_joint()`, or `infer()` on one model object.

The default metric follows the policy-entropy signal used by DemoSpeedup:
conditioned on the current observation, it samples several action chunks from
the `base` branch, estimates the local action density with Gaussian KDE, and
uses the resulting conditional action entropy as the routing score.

By default, high entropy selects `idm` and low entropy selects `base`:

```text
policy_entropy >= threshold -> idm
policy_entropy < threshold  -> base
```

This treats entropy as policy uncertainty: when the sampled action chunks spread
out, the adaptive model chooses the IDM path; when the sampled actions are
concentrated, it uses the basic FastWAM path. The threshold is intentionally
configurable because KDE entropy scale depends on action normalization,
bandwidth, action dimension, and the number of samples.

Because this is a single inherited model instance, it loads one checkpoint and
uses the same underlying weights for both code paths. The difference is the
inference algorithm selected at runtime.

## Files

- `src/fastwam/routing/metrics.py`: metric and selector abstractions.
- `src/fastwam/models/wan22/fastwam_metric_adaptive.py`: inherited adaptive model subclass.
- `configs/model/fastwam_metric_adaptive.yaml`: adaptive model config.
- `configs/task/libero_metric_adaptive_2cam224_1e-4.yaml`: LIBERO task config.
- `configs/task/robotwin_metric_adaptive_3cam_384_1e-4.yaml`: RoboTwin task config.

## Checkpoint Loading

The inherited adaptive model uses the normal FastWAM checkpoint format. Pass a
single checkpoint exactly as before:

```bash
python experiments/libero/run_libero_manager.py \
  task=libero_metric_adaptive_2cam224_1e-4 \
  ckpt=/path/to/checkpoint.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json
```

## Changing The Metric

Tune the policy-entropy metric:

```bash
model.adaptive.metric.num_samples=8
model.adaptive.metric.bandwidth=silverman
model.adaptive.selector.threshold=-0.25
```

Use a cheaper proxy call for entropy probing while keeping the final selected
branch unchanged:

```bash
model.adaptive.metric.probe_overrides.num_inference_steps=8
```

Use only selected action dimensions for entropy:

```bash
model.adaptive.metric.action_dims='[0,1,2,3,4,5,6]'
```

Route from the base branch action output instead:

```bash
model.adaptive.metric._target_=fastwam.routing.metrics.ActionOutputMetric
model.adaptive.metric.statistic=mean_abs
model.adaptive.selector.threshold=0.10
model.adaptive.selector.mode=ge
```

Use an input-image threshold instead:

```bash
model.adaptive.metric._target_=fastwam.routing.metrics.InputImageStdMetric
model.adaptive.selector.threshold=0.35
```

Use an externally supplied scalar in custom code:

```python
out = model.infer_action(
    prompt=prompt,
    input_image=image,
    action_horizon=32,
    routing_metric_value=0.8,
)
```

and configure:

```bash
model.adaptive.metric._target_=fastwam.routing.metrics.ExternalValueMetric
model.adaptive.selector.threshold=0.5
```

## Changing The Decision Rule

The default selector uses `mode=ge`, meaning:

```text
metric >= threshold -> idm
metric < threshold  -> base
```

Supported modes are `ge`, `gt`, `le`, and `lt`.

## Debugging

Adaptive outputs include an extra `_routing` field:

```python
{
  "action": ...,
  "_routing": {
    "selected_branch": "idm",
    "metric_name": "base_policy_entropy",
    "metric_value": 0.18,
    "threshold": 0.0,
    "mode": "ge"
  }
}
```

Existing evaluation code ignores this extra field, but custom scripts can log
it. The last decision is also available as `model.last_routing_decision`.

For controlled ablations, force a branch from custom inference code:

```python
out = model.infer_action(
    prompt=prompt,
    input_image=image,
    action_horizon=32,
    num_video_frames=9,
    force_branch="base",
)
```
