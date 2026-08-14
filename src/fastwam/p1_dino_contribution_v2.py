"""Deterministic control plane for DINO-to-action contribution v2.

This module intentionally contains no model construction.  It owns the frozen
validation ledger, task-paired batch ordering, dynamic reader warm-up state,
and causal checkpoint eligibility so each contract can be tested on CPU.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch


DINO_CONTRIBUTION_V2_PROFILE = "dino_contribution_v2"
TASK_PAIRED_SAMPLER_SCHEMA = "fastwam-p1-task-paired-sampler-v2"
WARMUP_STATE_SCHEMA = "fastwam-p1-dino-dependency-warmup-v2"
CAUSAL_LEDGER_SCHEMA = "fastwam-p1-dino-causal-ledger-v2"
CAUSAL_SELECTOR_SCHEMA = "fastwam-p1-dino-causal-selector-v2"


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        temporary.write_bytes(_canonical_json(value) + b"\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _as_int(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.item())
    return int(value)


@dataclass(frozen=True)
class DatasetWindowIdentity:
    """Stable metadata for one concatenated LeRobot window."""

    dataset_index: int
    suite_index: int
    suite: str
    task_index: int
    episode_index: int
    frame_index: int
    namespace: str = "validation-seed42"
    normalized_proprio: tuple[float, ...] | None = None

    @property
    def task_key(self) -> tuple[int, int]:
        return (self.suite_index, self.task_index)

    @property
    def stable_sample_id(self) -> str:
        return (
            f"{self.namespace}:{self.dataset_index}:suite{self.suite_index}:"
            f"task{self.task_index}:episode{self.episode_index}:"
            f"frame{self.frame_index}"
        )


def extract_dataset_window_identities(
    dataset: Any,
    *,
    include_normalized_proprio: bool,
    namespace: str = "validation-seed42",
) -> list[DatasetWindowIdentity]:
    """Read window identity columns without decoding videos."""

    underlying = getattr(dataset, "dataset", dataset)
    robot = getattr(underlying, "lerobot_dataset", None)
    multi = getattr(robot, "multi_dataset", None)
    if multi is None or not getattr(multi, "_datasets", None):
        raise TypeError("Expected a SampleIdentityDataset over RobotVideoDataset.")

    state_scale = None
    state_offset = None
    if include_normalized_proprio:
        processor = getattr(robot, "processor", None)
        normalizer = getattr(processor, "normalizer", None)
        if normalizer is None:
            raise ValueError("Dataset processor has no installed state normalizer.")
        state_normalizers = normalizer.normalizers["state"]
        if tuple(state_normalizers) != ("default",):
            raise ValueError("Causal ledger expects one merged default state field.")
        state_scale = state_normalizers["default"].scale.float()
        state_offset = state_normalizers["default"].offset.float()

    result: list[DatasetWindowIdentity] = []
    offset = 0
    for suite_index, (suite_name, source) in enumerate(
        zip(multi.ds_names, multi._datasets, strict=True)
    ):
        columns = source.hf_dataset
        episodes = columns["episode_index"]
        frames = columns["frame_index"]
        tasks = columns["task_index"]
        states = columns["observation.state"] if include_normalized_proprio else None
        for local_index, (episode, frame, task) in enumerate(
            zip(episodes, frames, tasks, strict=True)
        ):
            normalized = None
            if states is not None:
                state = torch.as_tensor(states[local_index]).float()
                if state.shape != state_scale.shape:
                    raise ValueError("Ledger proprio shape differs from normalizer.")
                normalized_tensor = (state * state_scale + state_offset).clamp(
                    -5.0,
                    5.0,
                )
                normalized = tuple(float(value) for value in normalized_tensor)
            result.append(
                DatasetWindowIdentity(
                    dataset_index=offset + local_index,
                    suite_index=suite_index,
                    suite=Path(str(suite_name)).name,
                    task_index=_as_int(task),
                    episode_index=_as_int(episode),
                    frame_index=_as_int(frame),
                    namespace=str(namespace),
                    normalized_proprio=normalized,
                )
            )
        offset += int(source.num_frames)
    if len(result) != len(dataset):
        raise ValueError(
            f"Concatenated metadata length changed: {len(result)} != {len(dataset)}."
        )
    return result


def _shuffle(values: Sequence[Any], generator: torch.Generator) -> list[Any]:
    if not values:
        return []
    order = torch.randperm(len(values), generator=generator).tolist()
    return [values[index] for index in order]


class TaskPairedDistributedBatchSampler(torch.utils.data.Sampler[list[int]]):
    """Build deterministic local batches from same-task cross-episode cycles.

    The usable population exactly matches the old global ``drop_last`` count.
    Each usable index occurs once; every negative cycle has length two or three
    and maps each primary sample to memory from a different episode.
    """

    def __init__(
        self,
        *,
        task_keys: Sequence[tuple[int, int]],
        episode_indices: Sequence[int],
        batch_size: int,
        rank: int,
        world_size: int,
        seed: int,
    ) -> None:
        if len(task_keys) != len(episode_indices) or not task_keys:
            raise ValueError("Task/episode metadata must be non-empty and aligned.")
        if batch_size < 2 or batch_size % 2:
            raise ValueError("Task-paired local batch size must be even and >= 2.")
        if not 0 <= rank < world_size:
            raise ValueError("Invalid distributed rank/world size.")
        self.task_keys = tuple((int(a), int(b)) for a, b in task_keys)
        self.episode_indices = tuple(int(value) for value in episode_indices)
        self.batch_size = int(batch_size)
        self.rank = int(rank)
        self.world_size = int(world_size)
        self.seed = int(seed)
        self.epoch = 0
        self._cache: (
            tuple[list[list[int]], dict[tuple[int, ...], tuple[int, ...]]] | None
        ) = None

    @property
    def usable_size_global(self) -> int:
        quantum = self.world_size * self.batch_size
        return len(self.task_keys) // quantum * quantum

    @property
    def dropped_size_global(self) -> int:
        return len(self.task_keys) - self.usable_size_global

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError("Sampler epoch must be non-negative.")
        self.epoch = int(epoch)
        self._cache = None

    def _groups_for_task(
        self,
        indices: Sequence[int],
        generator: torch.Generator,
    ) -> tuple[list[list[int]], list[list[int]]]:
        by_episode: dict[int, list[int]] = defaultdict(list)
        for index in indices:
            by_episode[self.episode_indices[index]].append(index)
        if len(by_episode) < 2:
            raise ValueError(
                "Task-paired sampling requires multiple episodes per task."
            )
        for episode in tuple(by_episode):
            by_episode[episode] = _shuffle(by_episode[episode], generator)

        triples: list[list[int]] = []
        if len(indices) % 2:
            available = sorted(
                by_episode,
                key=lambda episode: (-len(by_episode[episode]), episode),
            )
            if len(available) < 3:
                raise ValueError("An odd task group needs three distinct episodes.")
            triple = [by_episode[episode].pop() for episode in available[:3]]
            triples.append(triple)

        heap = [
            (-len(values), episode) for episode, values in by_episode.items() if values
        ]
        heapq.heapify(heap)
        pairs: list[list[int]] = []
        while heap:
            if len(heap) < 2:
                raise ValueError(
                    "Task windows cannot be paired without a same-episode negative."
                )
            _, first_episode = heapq.heappop(heap)
            _, second_episode = heapq.heappop(heap)
            pairs.append(
                [by_episode[first_episode].pop(), by_episode[second_episode].pop()]
            )
            for episode in (first_episode, second_episode):
                if by_episode[episode]:
                    heapq.heappush(heap, (-len(by_episode[episode]), episode))
        return pairs, triples

    def _build(self) -> tuple[list[list[int]], dict[tuple[int, ...], tuple[int, ...]]]:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.seed + self.epoch)
        global_order = torch.randperm(
            len(self.task_keys),
            generator=generator,
        )[: self.usable_size_global]
        local = global_order[self.rank :: self.world_size].tolist()
        if len(local) % self.batch_size:
            raise AssertionError("Rank-local usable population is not batch aligned.")

        by_task: dict[tuple[int, int], list[int]] = defaultdict(list)
        for index in local:
            by_task[self.task_keys[index]].append(index)
        pairs: list[list[int]] = []
        triples: list[list[int]] = []
        for task_key in sorted(by_task):
            task_pairs, task_triples = self._groups_for_task(
                by_task[task_key],
                generator,
            )
            pairs.extend(task_pairs)
            triples.extend(task_triples)
        pairs = _shuffle(pairs, generator)
        triples = _shuffle(triples, generator)
        if len(triples) % 2:
            raise AssertionError("An even usable population requires even triples.")

        batch_count = len(local) // self.batch_size
        batches: list[list[int]] = []
        permutations: dict[tuple[int, ...], tuple[int, ...]] = {}
        pair_cursor = 0
        triple_cursor = 0
        remaining_triples = len(triples)
        for batch_index in range(batch_count):
            bins_left = batch_count - batch_index
            minimum_here = max(0, remaining_triples - 10 * (bins_left - 1))
            triple_count = min(10, remaining_triples)
            triple_count = max(minimum_here, triple_count)
            if triple_count % 2:
                triple_count -= 1
            pair_count = (self.batch_size - 3 * triple_count) // 2
            if pair_count < 0:
                raise AssertionError("Task-paired batch packing overflowed.")
            selected_groups = (
                triples[triple_cursor : triple_cursor + triple_count]
                + pairs[pair_cursor : pair_cursor + pair_count]
            )
            if len(selected_groups) != triple_count + pair_count:
                raise AssertionError("Task-paired groups did not fill all batches.")
            selected_groups = _shuffle(selected_groups, generator)
            batch: list[int] = []
            permutation: list[int] = []
            for group in selected_groups:
                start = len(batch)
                batch.extend(group)
                permutation.extend(
                    start + ((offset + 1) % len(group)) for offset in range(len(group))
                )
            if len(batch) != self.batch_size or len(permutation) != self.batch_size:
                raise AssertionError("Task-paired batch packing changed local size.")
            for source, target in enumerate(permutation):
                if self.task_keys[batch[source]] != self.task_keys[batch[target]]:
                    raise AssertionError("Task-paired permutation crossed tasks.")
                if (
                    self.episode_indices[batch[source]]
                    == self.episode_indices[batch[target]]
                ):
                    raise AssertionError("Task-paired permutation reused an episode.")
            batches.append(batch)
            permutations[tuple(batch)] = tuple(permutation)
            triple_cursor += triple_count
            pair_cursor += pair_count
            remaining_triples -= triple_count
        if pair_cursor != len(pairs) or triple_cursor != len(triples):
            raise AssertionError("Task-paired batch packing left unused groups.")
        flat = [index for batch in batches for index in batch]
        if len(flat) != len(set(flat)) or set(flat) != set(local):
            raise AssertionError("Task-paired sampler changed usable sample coverage.")
        return batches, permutations

    def _materialize(
        self,
    ) -> tuple[list[list[int]], dict[tuple[int, ...], tuple[int, ...]]]:
        if self._cache is None:
            self._cache = self._build()
        return self._cache

    def __iter__(self) -> Iterator[list[int]]:
        batches, _ = self._materialize()
        yield from (list(batch) for batch in batches)

    def __len__(self) -> int:
        return self.usable_size_global // self.world_size // self.batch_size

    def permutation_for_sample_identities(
        self,
        sample_identities: Sequence[str],
    ) -> torch.Tensor:
        indices = tuple(
            int(str(identity).rsplit(":", 1)[-1]) for identity in sample_identities
        )
        _, permutations = self._materialize()
        if indices not in permutations:
            raise ValueError(
                "Observed DataLoader batch differs from paired sampler order."
            )
        return torch.tensor(permutations[indices], dtype=torch.int64)

    def state_dict(self) -> dict[str, Any]:
        batches, permutations = self._materialize()
        order_payload = [
            {"indices": batch, "permutation": list(permutations[tuple(batch)])}
            for batch in batches
        ]
        return {
            "schema": TASK_PAIRED_SAMPLER_SCHEMA,
            "seed": self.seed,
            "epoch": self.epoch,
            "rank": self.rank,
            "world_size": self.world_size,
            "batch_size": self.batch_size,
            "dataset_size": len(self.task_keys),
            "usable_size_global": self.usable_size_global,
            "dropped_size_global": self.dropped_size_global,
            "order_sha256": _sha256_json(order_payload),
        }

    def validate_state_dict(self, payload: Mapping[str, Any]) -> None:
        if dict(payload) != self.state_dict():
            raise ValueError("Task-paired sampler resume state/order changed.")


@dataclass
class DependencyWarmupController:
    """Serializable dynamic reader-only warm-up and LoRA-ramp state."""

    min_steps: int = 256
    max_steps: int = 1024
    window_active_updates: int = 128
    dependency_gap_min: float = 0.10
    pass_rate_min: float = 0.70
    consecutive_pass_windows_required: int = 2
    lora_gradient_ramp_steps: int = 512
    phase: str = "reader_warmup"
    warmup_end_step: int | None = None
    active_update_count: int = 0
    window_correct_sum: float = 0.0
    window_negative_sum: float = 0.0
    window_pass_count: int = 0
    window_update_count: int = 0
    consecutive_pass_windows: int = 0
    recent_windows: list[dict[str, Any]] = field(default_factory=list)
    lora_ramp_progress: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.min_steps <= self.max_steps:
            raise ValueError("Warm-up min/max steps are invalid.")
        if self.window_active_updates <= 0:
            raise ValueError("Warm-up window size must be positive.")
        if self.min_steps < (
            self.window_active_updates * self.consecutive_pass_windows_required
        ):
            raise ValueError("Warm-up minimum cannot precede two complete windows.")
        if not 0.0 <= self.pass_rate_min <= 1.0:
            raise ValueError("Warm-up pass-rate threshold must lie in [0,1].")
        if self.dependency_gap_min <= 0.0 or self.lora_gradient_ramp_steps <= 0:
            raise ValueError("Warm-up gap/ramp values must be positive.")
        if self.phase not in {"reader_warmup", "joint", "no_go"}:
            raise ValueError(f"Unknown DINO contribution phase {self.phase!r}.")

    @property
    def reader_only(self) -> bool:
        return self.phase == "reader_warmup"

    @property
    def failed(self) -> bool:
        return self.phase == "no_go"

    @property
    def next_lora_gradient_multiplier(self) -> float:
        if self.phase != "joint":
            return 0.0
        return min(
            (self.lora_ramp_progress + 1) / float(self.lora_gradient_ramp_steps),
            1.0,
        )

    def observe(
        self,
        *,
        correct_loss: float,
        negative_loss: float,
        global_step_after_update: int,
    ) -> dict[str, Any] | None:
        if self.phase != "reader_warmup":
            raise ValueError("Dependency warm-up observations are reader-only.")
        if not all(
            math.isfinite(float(value)) for value in (correct_loss, negative_loss)
        ):
            raise ValueError("Warm-up losses must be finite.")
        if correct_loss <= 0.0:
            raise ValueError("Warm-up correct loss must be positive.")
        relative_gap = (negative_loss - correct_loss) / correct_loss
        self.active_update_count += 1
        self.window_correct_sum += float(correct_loss)
        self.window_negative_sum += float(negative_loss)
        self.window_pass_count += int(relative_gap >= self.dependency_gap_min)
        self.window_update_count += 1
        completed = None
        if self.window_update_count == self.window_active_updates:
            aggregate_gap = (
                self.window_negative_sum - self.window_correct_sum
            ) / self.window_correct_sum
            pass_rate = self.window_pass_count / self.window_update_count
            passed = bool(
                aggregate_gap >= self.dependency_gap_min
                and pass_rate >= self.pass_rate_min
            )
            completed = {
                "end_global_step": int(global_step_after_update),
                "active_updates": self.window_update_count,
                "correct_loss_mean": self.window_correct_sum / self.window_update_count,
                "negative_loss_mean": self.window_negative_sum
                / self.window_update_count,
                "relative_gap": aggregate_gap,
                "update_pass_rate": pass_rate,
                "passed": passed,
            }
            self.recent_windows = (self.recent_windows + [completed])[-2:]
            self.consecutive_pass_windows = (
                self.consecutive_pass_windows + 1 if passed else 0
            )
            self.window_correct_sum = 0.0
            self.window_negative_sum = 0.0
            self.window_pass_count = 0
            self.window_update_count = 0
            if (
                global_step_after_update >= self.min_steps
                and self.consecutive_pass_windows
                >= self.consecutive_pass_windows_required
            ):
                self.phase = "joint"
                self.warmup_end_step = int(global_step_after_update)
        if global_step_after_update >= self.max_steps and self.phase == "reader_warmup":
            self.phase = "no_go"
        return completed

    def record_joint_update(self) -> None:
        if self.phase != "joint":
            raise ValueError("LoRA ramp progress advances only in joint phase.")
        self.lora_ramp_progress = min(
            self.lora_ramp_progress + 1,
            self.lora_gradient_ramp_steps,
        )

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": WARMUP_STATE_SCHEMA,
            "config": {
                "min_steps": self.min_steps,
                "max_steps": self.max_steps,
                "window_active_updates": self.window_active_updates,
                "dependency_gap_min": self.dependency_gap_min,
                "pass_rate_min": self.pass_rate_min,
                "consecutive_pass_windows": self.consecutive_pass_windows_required,
                "lora_gradient_ramp_steps": self.lora_gradient_ramp_steps,
            },
            "phase": self.phase,
            "warmup_end_step": self.warmup_end_step,
            "active_update_count": self.active_update_count,
            "window": {
                "correct_sum": self.window_correct_sum,
                "negative_sum": self.window_negative_sum,
                "pass_count": self.window_pass_count,
                "update_count": self.window_update_count,
            },
            "consecutive_pass_windows": self.consecutive_pass_windows,
            "recent_windows": list(self.recent_windows),
            "lora_ramp_progress": self.lora_ramp_progress,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> DependencyWarmupController:
        if payload.get("schema") != WARMUP_STATE_SCHEMA:
            raise ValueError("Unsupported dependency warm-up state schema.")
        config = payload.get("config")
        window = payload.get("window")
        if not isinstance(config, Mapping) or not isinstance(window, Mapping):
            raise ValueError("Dependency warm-up state is incomplete.")
        result = cls(
            min_steps=int(config["min_steps"]),
            max_steps=int(config["max_steps"]),
            window_active_updates=int(config["window_active_updates"]),
            dependency_gap_min=float(config["dependency_gap_min"]),
            pass_rate_min=float(config["pass_rate_min"]),
            consecutive_pass_windows_required=int(config["consecutive_pass_windows"]),
            lora_gradient_ramp_steps=int(config["lora_gradient_ramp_steps"]),
            phase=str(payload["phase"]),
            warmup_end_step=(
                None
                if payload.get("warmup_end_step") is None
                else int(payload["warmup_end_step"])
            ),
            active_update_count=int(payload["active_update_count"]),
            window_correct_sum=float(window["correct_sum"]),
            window_negative_sum=float(window["negative_sum"]),
            window_pass_count=int(window["pass_count"]),
            window_update_count=int(window["update_count"]),
            consecutive_pass_windows=int(payload["consecutive_pass_windows"]),
            recent_windows=[dict(value) for value in payload["recent_windows"]],
            lora_ramp_progress=int(payload["lora_ramp_progress"]),
        )
        if result.state_dict() != dict(payload):
            raise ValueError("Dependency warm-up state failed canonical round-trip.")
        return result


@dataclass
class NegativeModeCycle:
    """Serializable dependency-negative cycle."""

    modes: tuple[str, ...] = ("task_paired", "task_paired", "task_paired", "off")
    update_count: int = 0

    def __post_init__(self) -> None:
        if not self.modes or any(
            mode not in {"task_paired", "off"} for mode in self.modes
        ):
            raise ValueError("V2 negative cycle supports only task_paired/off.")
        if self.update_count < 0:
            raise ValueError("Negative-cycle update count must be non-negative.")

    @property
    def offset(self) -> int:
        return self.update_count % len(self.modes)

    @property
    def current(self) -> str:
        return self.modes[self.offset]

    def advance(self) -> None:
        self.update_count += 1

    def state_dict(self) -> dict[str, Any]:
        return {
            "modes": list(self.modes),
            "dependency_update_count": self.update_count,
            "negative_cycle_offset": self.offset,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> NegativeModeCycle:
        result = cls(
            modes=tuple(str(value) for value in payload["modes"]),
            update_count=int(payload["dependency_update_count"]),
        )
        if int(payload["negative_cycle_offset"]) != result.offset:
            raise ValueError("Negative-cycle offset disagrees with update count.")
        return result


@dataclass(frozen=True)
class CausalSelectionThresholds:
    """Frozen checkpoint-eligibility thresholds."""

    validation_loss_max: float
    pose_mse_max: float
    gripper_mse_max: float
    hard_negative_gap_min: float = 0.10
    off_gap_min: float = 0.10
    positive_task_fraction_min: float = 0.70
    residual_hidden_p95_max: float = 0.20


@dataclass
class CausalCheckpointSelector:
    """Reject low-loss checkpoints that do not depend on DINO memory."""

    thresholds: CausalSelectionThresholds
    best: dict[str, Any] | None = None

    def assess(
        self,
        *,
        step: int,
        validation: Mapping[str, Any],
        causal: Mapping[str, Any],
        frozen_contract_unchanged: bool,
    ) -> dict[str, Any]:
        checks = {
            "validation_loss": float(validation["loss_action_bc"])
            <= self.thresholds.validation_loss_max,
            "pose_mse": float(validation["mse_pose"]) <= self.thresholds.pose_mse_max,
            "gripper_mse": float(validation["mse_gripper"])
            <= self.thresholds.gripper_mse_max,
            "hard_negative_gap": float(causal["task_paired_relative_gap"])
            >= self.thresholds.hard_negative_gap_min,
            "off_gap": float(causal["off_relative_gap"]) >= self.thresholds.off_gap_min,
            "positive_task_fraction": float(causal["positive_task_fraction"])
            >= self.thresholds.positive_task_fraction_min,
            "residual_hidden_p95": float(causal["residual_hidden_p95"])
            <= self.thresholds.residual_hidden_p95_max,
            "finite": bool(causal.get("finite", False)),
            "frozen_contract": bool(frozen_contract_unchanged),
        }
        eligible = all(checks.values())
        return {
            "step": int(step),
            "eligible": eligible,
            "checks": checks,
            "reasons": [name for name, passed in checks.items() if not passed],
            "validation": dict(validation),
            "causal": dict(causal),
        }

    def consider(self, assessment: Mapping[str, Any]) -> bool:
        if not bool(assessment["eligible"]):
            return False
        loss = float(assessment["validation"]["loss_action_bc"])
        if self.best is not None and loss >= float(
            self.best["validation"]["loss_action_bc"]
        ):
            return False
        self.best = dict(assessment)
        return True

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": CAUSAL_SELECTOR_SCHEMA,
            "thresholds": {
                key: float(value) for key, value in vars(self.thresholds).items()
            },
            "best": self.best,
        }

    @classmethod
    def from_state_dict(cls, payload: Mapping[str, Any]) -> CausalCheckpointSelector:
        if payload.get("schema") != CAUSAL_SELECTOR_SCHEMA:
            raise ValueError("Unsupported causal-selector state schema.")
        result = cls(
            thresholds=CausalSelectionThresholds(**dict(payload["thresholds"])),
            best=None if payload.get("best") is None else dict(payload["best"]),
        )
        if result.state_dict() != dict(payload):
            raise ValueError("Causal-selector state failed canonical round-trip.")
        return result


def _select_anchor_rows(
    rows: Sequence[DatasetWindowIdentity],
    *,
    anchors_per_task: int,
) -> list[DatasetWindowIdentity]:
    by_task: dict[tuple[int, int], list[DatasetWindowIdentity]] = defaultdict(list)
    for row in rows:
        by_task[row.task_key].append(row)
    if len(by_task) != 40:
        raise ValueError(
            f"Causal ledger requires exactly 40 tasks, got {len(by_task)}."
        )
    selected: list[DatasetWindowIdentity] = []
    for task_key in sorted(by_task):
        by_episode: dict[int, list[DatasetWindowIdentity]] = defaultdict(list)
        for row in by_task[task_key]:
            by_episode[row.episode_index].append(row)
        episodes = sorted(by_episode)
        visits = [slot % len(episodes) for slot in range(anchors_per_task)]
        total_visits = {index: visits.count(index) for index in set(visits)}
        seen: dict[int, int] = defaultdict(int)
        for episode_position in visits:
            episode = episodes[episode_position]
            candidates = sorted(
                by_episode[episode],
                key=lambda row: (row.frame_index, row.dataset_index),
            )
            visit = seen[episode_position]
            seen[episode_position] += 1
            quantile = (visit + 0.5) / total_visits[episode_position]
            candidate_position = min(
                int(math.floor(quantile * len(candidates))),
                len(candidates) - 1,
            )
            selected.append(candidates[candidate_position])
    return selected


def _rgb_sha256(sample: Mapping[str, Any]) -> str:
    pixels = sample.get("p1_camera_pixels")
    valid = sample.get("p1_camera_valid_mask")
    if not isinstance(pixels, torch.Tensor) or pixels.dtype != torch.uint8:
        raise ValueError("Ledger sample is missing uint8 P1 camera pixels.")
    if not isinstance(valid, torch.Tensor) or valid.shape != pixels.shape[:1]:
        raise ValueError("Ledger sample camera-valid mask is malformed.")
    digest = hashlib.sha256()
    digest.update(pixels.contiguous().numpy().tobytes())
    digest.update(valid.to(torch.uint8).contiguous().numpy().tobytes())
    return digest.hexdigest()


def build_frozen_causal_ledger(
    dataset: Any,
    path: str | os.PathLike[str],
    *,
    anchors_per_task: int = 16,
    negative_fallback_dataset: Any | None = None,
) -> dict[str, Any]:
    """Freeze 640 model-independent anchors and proprio-nearest negatives."""

    target = Path(path).expanduser().resolve()
    if target.exists():
        return load_frozen_causal_ledger(target)
    if anchors_per_task != 16:
        raise ValueError("The v2 causal ledger freezes exactly 16 anchors per task.")
    rows = extract_dataset_window_identities(
        dataset,
        include_normalized_proprio=True,
        namespace="validation-seed42",
    )
    fallback_rows = (
        []
        if negative_fallback_dataset is None
        else extract_dataset_window_identities(
            negative_fallback_dataset,
            include_normalized_proprio=True,
            namespace="train-seed42",
        )
    )
    anchors = _select_anchor_rows(rows, anchors_per_task=anchors_per_task)
    by_task: dict[tuple[int, int], list[DatasetWindowIdentity]] = defaultdict(list)
    for row in rows:
        by_task[row.task_key].append(row)
    fallback_by_task: dict[tuple[int, int], list[DatasetWindowIdentity]] = defaultdict(
        list
    )
    for row in fallback_rows:
        fallback_by_task[row.task_key].append(row)
    rgb_cache: dict[tuple[str, int], str] = {}

    def rgb_hash(split: str, index: int) -> str:
        key = (split, index)
        if key not in rgb_cache:
            source = dataset if split == "validation" else negative_fallback_dataset
            if source is None:
                raise ValueError(
                    "Training negative requested without a fallback dataset."
                )
            rgb_cache[key] = _rgb_sha256(source[index])
        return rgb_cache[key]

    entries = []
    for anchor in anchors:
        if anchor.normalized_proprio is None:
            raise AssertionError("Ledger anchor is missing normalized proprio.")
        anchor_state = torch.tensor(anchor.normalized_proprio)
        candidate_groups = []
        for split, source_rows in (
            ("validation", by_task[anchor.task_key]),
            ("train", fallback_by_task[anchor.task_key]),
        ):
            candidates = []
            for candidate in source_rows:
                if candidate.episode_index == anchor.episode_index:
                    continue
                candidate_state = torch.tensor(candidate.normalized_proprio)
                distance = float(
                    torch.linalg.vector_norm(anchor_state - candidate_state)
                )
                candidates.append((distance, candidate.stable_sample_id, candidate))
            candidates.sort(key=lambda value: (value[0], value[1]))
            candidate_groups.append((split, candidates))
        anchor_rgb = rgb_hash("validation", anchor.dataset_index)
        negative = None
        negative_rgb = None
        negative_split = None
        distance = None
        for split, candidates in candidate_groups:
            for candidate_distance, _, candidate in candidates:
                candidate_rgb = rgb_hash(split, candidate.dataset_index)
                if candidate_rgb != anchor_rgb:
                    negative = candidate
                    negative_rgb = candidate_rgb
                    negative_split = split
                    distance = candidate_distance
                    break
            if negative is not None:
                break
        if negative is None:
            raise ValueError(
                f"No RGB-distinct cross-episode negative for {anchor.stable_sample_id}."
            )
        entries.append(
            {
                "anchor_id": anchor.stable_sample_id,
                "anchor_split": "validation",
                "anchor_dataset_index": anchor.dataset_index,
                "negative_id": negative.stable_sample_id,
                "negative_split": negative_split,
                "negative_dataset_index": negative.dataset_index,
                "suite_index": anchor.suite_index,
                "suite": anchor.suite,
                "task_index": anchor.task_index,
                "anchor_episode_index": anchor.episode_index,
                "negative_episode_index": negative.episode_index,
                "anchor_frame_index": anchor.frame_index,
                "negative_frame_index": negative.frame_index,
                "normalized_proprio_l2": distance,
                "anchor_rgb_sha256": anchor_rgb,
                "negative_rgb_sha256": negative_rgb,
            }
        )
    payload: dict[str, Any] = {
        "schema": CAUSAL_LEDGER_SCHEMA,
        "selection": {
            "source": "173-episode validation split seed 42",
            "task_count": 40,
            "anchors_per_task": 16,
            "anchor_count": 640,
            "anchor_method": "round_robin_episode_then_uniform_frame_quantile",
            "negative_method": (
                "validation_then_train_fallback_same_suite_task_cross_episode_"
                "nearest_parent_normalized_proprio_l2"
            ),
            "tie_break": "stable_sample_id",
            "rgb_contract": "sha256(p1_camera_uint8 || camera_valid_mask)",
            "model_forward_before_write": False,
        },
        "validation_pool": {
            "episode_count": len(
                {(row.suite_index, row.episode_index) for row in rows}
            ),
            "window_count": len(rows),
        },
        "negative_fallback_pool": {
            "source": "training split seed 42",
            "episode_count": len(
                {(row.suite_index, row.episode_index) for row in fallback_rows}
            ),
            "window_count": len(fallback_rows),
            "used_anchor_count": sum(
                value["negative_split"] == "train" for value in entries
            ),
        },
        "anchors": entries,
    }
    payload["content_sha256"] = _sha256_json(payload)
    _atomic_json(target, payload)
    file_sha256 = hashlib.sha256(target.read_bytes()).hexdigest()
    _atomic_json(
        target.with_suffix(target.suffix + ".sha256.json"),
        {
            "ledger": str(target),
            "ledger_file_sha256": file_sha256,
            "content_sha256": payload["content_sha256"],
        },
    )
    return payload | {"ledger_file_sha256": file_sha256}


def load_frozen_causal_ledger(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Strictly load and verify a frozen causal ledger."""

    target = Path(path).expanduser().resolve()
    payload = json.loads(target.read_text(encoding="utf-8"))
    if payload.get("schema") != CAUSAL_LEDGER_SCHEMA:
        raise ValueError("Unsupported causal-ledger schema.")
    content_sha256 = payload.pop("content_sha256", None)
    if content_sha256 != _sha256_json(payload):
        raise ValueError("Causal-ledger content SHA256 mismatch.")
    payload["content_sha256"] = content_sha256
    anchors = payload.get("anchors")
    if not isinstance(anchors, list) or len(anchors) != 640:
        raise ValueError("Causal ledger must contain exactly 640 anchors.")
    tasks = {(value["suite_index"], value["task_index"]) for value in anchors}
    if len(tasks) != 40:
        raise ValueError("Causal ledger task coverage changed.")
    if any(
        value.get("anchor_split") != "validation"
        or value.get("negative_split") not in {"validation", "train"}
        or value["anchor_episode_index"] == value["negative_episode_index"]
        or value["anchor_rgb_sha256"] == value["negative_rgb_sha256"]
        for value in anchors
    ):
        raise ValueError("Causal ledger contains an invalid hard negative.")
    payload["ledger_file_sha256"] = hashlib.sha256(target.read_bytes()).hexdigest()
    return payload
