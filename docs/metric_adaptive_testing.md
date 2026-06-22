# Metric-Adaptive FastWAM — 测试与调试文档

本文档给出 `MetricAdaptiveFastWAM`(双 regime 训练)的全部测试用例,以及在服务器上
从「能跑通」到「真正训练起来」的调试阶梯。

测试分三层(由轻到重):

| 层 | 文件 / 命令 | 依赖 | 目的 |
|---|---|---|---|
| Tier 0 | 静态检查命令 | 无(或仅 hydra) | 语法、import、config 可组装 |
| Tier 1 | `tests/test_metric_adaptive_routing.py` | torch | 路由 metric / selector 纯逻辑 |
| Tier 2 | `tests/test_metric_adaptive_training.py` | torch+hydra(单测)/ +Wan权重+GPU(集成) | 训练 loss、梯度、推理分支、checkpoint |
| Tier 3 | 真训练阶梯(本文 §6) | 全栈+数据+GPU | 2 步冒烟 → 过拟合单 batch → 正式训练 |

---

## 1. 快速开始

```bash
cd FastWAM
pip install pytest                       # 若未安装

# Tier 1(任何有 torch 的机器都能跑)
pytest tests/test_metric_adaptive_routing.py -v

# Tier 2 单测(需要包能 import:torch + hydra,不需要权重/GPU)
pytest tests/test_metric_adaptive_training.py -v -k "not TestDualRegime"

# Tier 2 集成(需要 Wan2.2 权重 + GPU)
RUN_FASTWAM_MODEL_TESTS=1 pytest tests/test_metric_adaptive_training.py -v
```

集成测试可用环境变量:
- `RUN_FASTWAM_MODEL_TESTS=1` — 打开重型集成测试(默认跳过)。
- `FASTWAM_TEST_TASK`(默认 `libero_metric_adaptive_2cam224_1e-4`)。
- `FASTWAM_TEST_DEVICE`(默认 `cuda`)。
- `FASTWAM_CONFIGS_DIR`(默认 `<repo>/configs`)。

---

## 2. Tier 0 — 静态检查

```bash
# 语法 / 字节码
python -m compileall src/fastwam/models/wan22/fastwam_metric_adaptive.py

# import 检查(会拉起 Wan 模块的 import,但不构造模型)
python -c "import fastwam.models.wan22.fastwam_metric_adaptive as m; print('import OK')"

# Hydra 干组装(只解析 config,不构造模型);确认 train: 块与 adaptive: 块都在
python -c "
from hydra import compose, initialize_config_dir; import os
with initialize_config_dir(version_base='1.3', config_dir=os.path.abspath('configs')):
    cfg = compose(config_name='train', overrides=['task=libero_metric_adaptive_2cam224_1e-4'])
print('model._target_ =', cfg.model._target_)
print('train =', cfg.model.train)
print('adaptive.selector =', cfg.model.adaptive.selector)
"
```

期望:`_target_` 指向 `create_metric_adaptive_fastwam`;`train` 含
`action_regime_weight_base` 与 `share_inputs`。

---

## 3. Tier 1 — 路由纯逻辑测试(`test_metric_adaptive_routing.py`)

不需要权重/GPU/数据。覆盖推理时的 metric 与 selector:

| 测试 | 验证点 |
|---|---|
| `test_threshold_selector_modes` | `ge/gt/le/lt` 四种比较、边界值(`value==threshold`)走向正确分支 |
| `test_threshold_selector_custom_branches` | 自定义 `low_branch/high_branch` 映射 |
| `test_threshold_selector_rejects_unknown_mode` | 非法 mode 抛 `ValueError` |
| `test_external_value_*` | `ExternalValueMetric` 读 kwargs / 用 default / required 缺失抛错 |
| `test_input_image_std_*` | `InputImageStdMetric` 等于张量 std;缺 `input_image` 抛错 |
| `test_action_output_mean_abs` / `_temporal_delta_mean_abs` | `ActionOutputMetric` 统计量数值正确 |
| `test_action_output_rejects_unknown_statistic` / `_missing_action_key` | 非法统计量 / 缺 `action` 键抛错 |
| `test_policy_entropy_runs_and_probes_num_samples_times` | KDE 熵有限、探测次数 == `num_samples` |
| `test_policy_entropy_requires_two_samples` | `num_samples<2` 抛错 |
| `test_policy_entropy_action_dims_subset` | `action_dims` 子集生效 |

> 这些用例的期望值已对照真实 `fastwam.routing.metrics` 实测通过。

---

## 4. Tier 2 单测 — import 级(`test_metric_adaptive_training.py`,Group 1)

只要包能 import(torch + hydra)就能跑,不需要权重/GPU:

| 测试 | 验证点 |
|---|---|
| `test_to_plain_dict_none_is_empty` / `_passthrough_mapping` | 工厂解析 `train` 块的辅助函数 |
| `test_train_knob_parsing_matches_factory_defaults` | `train=None` 时默认 `w_base=1.0`、`share_inputs=True`;显式值生效 |
| `test_action_loss_per_sample_no_pad` | 动作 loss 还原口径(== 50.0)与父类一致 |
| `test_action_loss_per_sample_respects_pad_mask` | `action_is_pad` 屏蔽 padded 步 |
| `test_action_loss_per_sample_all_pad_is_finite` | 全 padding 时 `clamp(min=1)` 防除零、结果有限 |

---

## 5. Tier 2 集成 — 真模型(`test_metric_adaptive_training.py`,Group 2,`RUN_FASTWAM_MODEL_TESTS=1`)

按 trainer 的方式 `instantiate(cfg.model, ...)` 构造真模型,用小合成 batch
(`T=5, H=W=64, B=1`,满足 `T%4==1`、`Ta%(T-1)==0`、`HW%16==0`)跑真前向。
若权重/GPU 不可用会自动 `skip`。

| 测试 | 验证点 | 对应设计约束 |
|---|---|---|
| `test_training_loss_runs_backward_and_keys` | `loss` 是标量、可微、有限;`loss_dict` 三键齐全;`backward()` 不报错 | 训练能跑 |
| `test_gradient_covers_all_trained_params` | `model.dit` 所有 `requires_grad` 参数都拿到梯度 | **grad-every-step**(分布式安全) |
| `test_video_loss_computed_exactly_once` | `_compute_video_loss_per_sample` 每步只被调用 1 次 | **视频损失不双计** |
| `test_w_base_zero_drops_base_action_term` | `w_base=0` 时 `loss_action_base` 项为 0 | base 项权重可控 |
| `test_share_inputs_false_delegated_path` | `share_inputs=false`(委托父类)路径也能跑通+反传 | 零漂移回退路径 |
| `test_force_branch_inference[base/idm]` | 两条推理分支都能产出正确形状动作,`_routing.selected_branch` 正确 | 训练后两分支都可用 |
| `test_checkpoint_format_unchanged` | payload 键仍为 `{mot, [proprio_encoder], step, torch_dtype}` | **checkpoint 格式不变** |

`test_gradient_covers_all_trained_params` 是「双 regime / grad-every-step」设计的
经验性裁判:它通过即说明当前实现每步覆盖所有被训练参数(见 §7)。

---

## 6. Tier 3 — 真训练阶梯(从跑通到训起来)

按顺序逐级放大,每一级先通过再进下一级。

**前置(一次性):**
```bash
# 文本 embedding 预计算(训练前必须,否则 dataset __getitem__ 会 FileNotFoundError)
python scripts/precompute_text_embeds.py task=libero_metric_adaptive_2cam224_1e-4
```

**6.1 单步反传冒烟(最快,定位模型/loss 问题)**
```bash
RUN_FASTWAM_MODEL_TESTS=1 pytest tests/test_metric_adaptive_training.py::TestDualRegimeTraining::test_training_loss_runs_backward_and_keys -v
```

**6.2 2 步真训练冒烟(打通 data → VAE → MoT → training_loss → optimizer → DeepSpeed)**
```bash
bash scripts/train_zero1.sh 1 \
  task=libero_metric_adaptive_2cam224_1e-4 \
  max_steps=2 batch_size=1 log_every=1 save_every=2 eval_every=100000
```
看日志里是否每步打印 `loss_video / loss_action_idm / loss_action_base` 三项,且都为有限值。

**6.3 过拟合单 batch(验证学习信号正确)**

把数据集截到 1~2 个样本(或临时让 sampler 反复取同一 batch),跑几百步,
确认 `loss_action_idm` 与 `loss_action_base` 都能稳定下降(理想趋近 0)。
这是判断「动作专家是否真在两种条件下都学到东西」的关键。

**6.4 小规模多卡 + 正式训练**
```bash
bash scripts/train_zero1.sh 8 task=libero_metric_adaptive_2cam224_1e-4
# 或 ZeRO-2:
bash scripts/train_zero2.sh 8 task=libero_metric_adaptive_2cam224_1e-4
```
多卡时重点确认:无 unused-parameter / reduction-mismatch 报错;三个 loss key
在所有 rank 上每步都存在(trainer 逐 key gather)。

**6.5 评测两分支**(训练若干步存 ckpt 后)
```bash
python experiments/libero/run_libero_manager.py \
  task=libero_metric_adaptive_2cam224_1e-4 \
  ckpt=/path/to/checkpoint.pt \
  EVALUATION.dataset_stats_path=/path/to/dataset_stats.json
```
对比 base 分支动作误差相对「仅 IDM 训练」基线是否下降——这是本次修正的最终目标指标。
也可在自定义脚本里用 `force_branch="base"` / `"idm"` 直接对照两分支。

---

## 7. 常见报错排查

| 现象 | 可能原因 | 处理 |
|---|---|---|
| `FileNotFoundError` 在 dataset 取数时 | 没预计算文本 embedding | 先跑 §6 前置的 `precompute_text_embeds.py` |
| `Base-regime action training requires fuse_vae_embedding_in_latents=true` | 配置里 `fuse_vae_embedding_in_latents` 非 true | adaptive 配置默认已是 true,勿改 |
| `Teacher-forcing requires token-wise t_mod` | `seperated_timestep` 或 `fuse_vae_embedding_in_latents` 非 true | 保持 adaptive 配置默认 |
| DDP 报 unused parameter | 改成了「base-only 步」(见 §7 说明) | 用默认双 regime;或保证每步都跑 idm 前向 |
| `loss_dict` key 在 rank 间不一致导致 gather 卡住 | 自定义了按 rank 不同的 loss 组成 | 保证每步三键恒定 |
| OOM | 双前向峰值激活偏高 | 降 `batch_size`;确认 `mot_checkpoint_mixed_attn=true`;或用「两次 backward」省显存(见 §8) |
| `TypeError: unexpected keyword 'train'` | 误把 `train` 透传进 `from_wan22_pretrained` | 工厂里 `train` 只解析为属性,勿透传(已实现) |

---

## 8. 附:每步两 forward vs 每步一边(实验方法)

详见对话中的分析。要点:

- **默认(both,每步两 forward)**:idm 前向**每步都跑**且带视频损失,因此**覆盖所有被训练参数**;
  base 前向很便宜(视频分支只有单帧)。低方差、分布式安全。
- **每步只走一边的风险**:base-only 步**没有视频损失、且丢弃视频输出**,会让 video expert
  的一小撮参数(只被丢弃视频输出消费的那部分,如最后一个 block 的输出路径)**该步拿不到梯度**
  → DDP/ZeRO 可能报 unused parameter,信号也更抖。
- **若显存紧张的折中(更安全)**:`idm 每步都跑 + base 以概率 p 跑`,而不是纯交替。
  idm 永远覆盖全部参数,跳过的只是廉价的 base 前向。
- **省显存技巧(保持 both)**:把一次 `loss_total.backward()` 拆成两次
  `loss_idm.backward()` 再 `loss_base.backward()`(梯度累加),可在构造 base 图前释放 idm 图。

如需把它做成可切换旋钮,在 `MetricAdaptiveFastWAM.training_loss` 顶部加:

```python
# 可选:base 前向以概率 p 执行(idm 永远执行,保证 grad 覆盖)
import torch
base_apply_prob = float(getattr(self, "train_base_apply_prob", 1.0))
run_base = base_apply_prob >= 1.0 or bool((torch.rand(()) < base_apply_prob).item())
# ...仅当 run_base 时才计算 loss_action_base 并加入 loss_total;否则该步 base 项为 0
```

多卡时让该决策跨 rank 一致(用全局 step 派生而非独立随机),避免各 rank 目标不一致。
对应可加测试:`p=0` 时 `loss_action_base==0` 且 `test_gradient_covers_all_trained_params` 仍通过
(因为 idm 前向覆盖全部参数)。
