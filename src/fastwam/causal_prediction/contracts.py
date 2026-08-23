"""Versioned public contracts for same-state causal interventions."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import torch

CAUSAL_POLICY_CHECKPOINT_SCHEMA = "fastwam-shared-causal-policy-v1"
CAUSAL_POLICY_CHECKPOINT_SCHEMA_V2 = "fastwam-shared-causal-policy-v2"
UPLIFT_GATE_CHECKPOINT_SCHEMA = "causal-uplift-gate-v1"
UPLIFT_GATE_CHECKPOINT_SCHEMA_V2 = "causal-uplift-gate-v2"
PAIRED_INTERVENTION_RECORD_SCHEMA = "paired-intervention-record-v1"
PAIRED_INTERVENTION_RECORD_SCHEMA_V2 = "paired-intervention-record-v2"
CURRENT_ONLY_CAUSAL_POLICY_CHECKPOINT_SCHEMA = "fastwam-current-only-causal-policy-v1"
GATE_FEATURE_RECORD_SCHEMA = "causal-gate-feature-record-v1"
GATE_TRAINING_EXAMPLE_SCHEMA = "causal-gate-training-example-v1"


class CausalComputeMode(str, Enum):
    """Compute treatment applied to the current action chunk.

    ``G_NO_READ`` is an audit intervention, not a routable scientific expert.
    It pays the full C2 prediction cost while using the exact C0 action
    condition. ``C1_ONE_PASS`` is reserved until the pilot phase gate passes.
    """

    C0_CURRENT = "c0_current"
    C1_ONE_PASS = "c1_one_pass"
    C2_FULL = "c2_full"
    G_NO_READ = "g_no_read"

    @classmethod
    def parse(cls, value: CausalComputeMode | str) -> CausalComputeMode:
        """Return a validated compute mode."""

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as error:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown causal compute mode {value!r}; expected: {allowed}"
            ) from error

    @property
    def runs_future_prediction(self) -> bool:
        """Whether the intervention executes a future-video expert path."""

        return self in {
            CausalComputeMode.C1_ONE_PASS,
            CausalComputeMode.C2_FULL,
            CausalComputeMode.G_NO_READ,
        }

    @property
    def reads_future_condition(self) -> bool:
        """Whether ActionDiT may consume generated future-video K/V."""

        return self in {
            CausalComputeMode.C1_ONE_PASS,
            CausalComputeMode.C2_FULL,
        }

    @property
    def is_routable(self) -> bool:
        """Whether the mode is a policy choice rather than a negative control."""

        return self is not CausalComputeMode.G_NO_READ


class CausalControlKind(str, Enum):
    """Diagnostic intervention applied around a formal compute expert."""

    STANDARD = "standard"
    NO_READ = "no_read"
    GENERIC_MEDOID = "generic_medoid"
    REPEAT_CURRENT = "repeat_current"
    SHUFFLED_WRONG_STATE = "shuffled_wrong_state"
    TEMPORAL_SHIFT = "temporal_shift"
    INSTRUCTION_MISMATCH = "instruction_mismatch"
    GT_FUTURE_OFFLINE = "gt_future_offline"

    @classmethod
    def parse(cls, value: CausalControlKind | str) -> CausalControlKind:
        """Return a validated diagnostic control kind."""

        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as error:
            allowed = ", ".join(item.value for item in cls)
            raise ValueError(
                f"Unknown causal control kind {value!r}; expected: {allowed}"
            ) from error


class CausalDomain(str, Enum):
    """Supported data domains without implying scientific authorization."""

    CLEAN = "clean"
    PLUS = "plus"
    DEVELOPMENT_OOD = "development_ood"
    OFFLINE_DEMONSTRATION = "offline_demonstration"


class CausalSamplingStratum(str, Enum):
    """Frozen within-episode state-selection design."""

    UNIFORM = "uniform"
    CRITICALITY = "criticality"


class CausalPhase(str, Enum):
    """Primary mutually exclusive phase available at the decision state."""

    TRANSIT = "transit"
    PRE_CONTACT = "pre_contact"
    CONTACT = "contact"
    POST_CONTACT = "post_contact"
    SUBGOAL_TRANSITION = "subgoal_transition"


class CausalTerminationType(str, Enum):
    """Directly observed branch termination; no inferred failure semantics."""

    SUCCESS = "success"
    TIME_LIMIT = "time_limit"
    ENV_TERMINATION = "env_termination"
    UNKNOWN = "unknown"


_CRITICALITY_COMPONENTS = {
    "gripper_transition",
    "action_curvature",
    "contact_proximity",
    "predicate_transition",
    "action_precision",
}


def _finite_probability(value: float, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or not 0.0 < parsed <= 1.0:
        raise ValueError(f"`{name}` must lie in (0, 1].")
    return parsed


def _finite_nonnegative(value: float, *, name: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed < 0.0:
        raise ValueError(f"`{name}` must be finite and non-negative.")
    return parsed


def _json_value(value: Any) -> Any:
    """Convert nested contract values into JSON-compatible primitives."""

    if isinstance(value, Enum):
        return value.value
    if hasattr(value, "to_artifact"):
        return value.to_artifact()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    return value


@dataclass(frozen=True)
class CausalStateIdentityV2:
    """Collision-free task, episode, and chunk identity for v2 artifacts."""

    domain: CausalDomain
    suite: str
    local_task_id: int
    global_task_uid: str
    task_name: str
    clean_base_task_uid: str
    trial_id: int
    reset_id: int
    source_episode_id: str
    chunk_index: int
    policy_seed: int
    model_seed: int
    plus_factor: str | None = None
    plus_difficulty: str | None = None
    plus_variant: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "domain", CausalDomain(self.domain))
        required = (
            self.suite,
            self.global_task_uid,
            self.task_name,
            self.clean_base_task_uid,
            self.source_episode_id,
        )
        if any(not str(value) for value in required):
            raise ValueError("Causal v2 state identity strings must be non-empty.")
        counters = (
            self.local_task_id,
            self.trial_id,
            self.reset_id,
            self.chunk_index,
            self.policy_seed,
            self.model_seed,
        )
        if any(int(value) < 0 for value in counters):
            raise ValueError("Causal v2 state identity counters must be non-negative.")
        plus_metadata = (
            self.plus_factor,
            self.plus_difficulty,
            self.plus_variant,
        )
        if self.domain is CausalDomain.PLUS and any(
            not value for value in plus_metadata
        ):
            raise ValueError("Plus state identities require factor/difficulty/variant.")
        if self.domain is not CausalDomain.PLUS and any(
            value is not None for value in plus_metadata
        ):
            raise ValueError("Plus metadata is only valid in the Plus domain.")

    @property
    def snapshot_id(self) -> str:
        """Return the canonical suite-safe snapshot identity."""

        return (
            f"{self.domain.value}:{self.global_task_uid}:trial-{self.trial_id}:"
            f"episode-{self.source_episode_id}:chunk-{self.chunk_index}"
        )

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-compatible identity payload."""

        return _json_value(asdict(self))


@dataclass(frozen=True)
class CausalSamplingMetadataV2:
    """Frozen source-policy and within-episode inclusion design."""

    source_policy: str
    source_final_success: bool | None
    sampling_stratum: CausalSamplingStratum
    criticality_components: Mapping[str, float]
    criticality_percentiles: Mapping[str, float]
    criticality_score: float
    eligible_chunk_count: int
    conditional_selection_probability: float
    joint_inclusion_probability: float
    phase: CausalPhase
    failure_adjacent: bool = False
    recovery: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "sampling_stratum", CausalSamplingStratum(self.sampling_stratum)
        )
        object.__setattr__(self, "phase", CausalPhase(self.phase))
        if not self.source_policy:
            raise ValueError("Source policy must be non-empty.")
        if set(self.criticality_components) != _CRITICALITY_COMPONENTS:
            raise ValueError(
                "Criticality component names changed from the preregistration."
            )
        if set(self.criticality_percentiles) != _CRITICALITY_COMPONENTS:
            raise ValueError(
                "Criticality percentile names changed from the preregistration."
            )
        if any(
            not math.isfinite(float(value))
            for value in self.criticality_components.values()
        ):
            raise ValueError("Criticality components must be finite.")
        if any(
            not 0.0 <= float(value) <= 1.0
            for value in self.criticality_percentiles.values()
        ):
            raise ValueError("Criticality percentiles must lie in [0, 1].")
        _finite_nonnegative(self.criticality_score, name="criticality_score")
        if self.eligible_chunk_count < 1:
            raise ValueError("Sampling metadata requires a positive eligible count.")
        conditional = _finite_probability(
            self.conditional_selection_probability,
            name="conditional_selection_probability",
        )
        joint = _finite_probability(
            self.joint_inclusion_probability,
            name="joint_inclusion_probability",
        )
        if not math.isclose(joint, 0.5 * conditional, abs_tol=1e-12):
            raise ValueError(
                "Joint inclusion probability must equal 0.5 times the conditional "
                "within-stratum probability."
            )

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-compatible sampling payload."""

        return _json_value(asdict(self))


@dataclass(frozen=True)
class CausalInterventionSpecV2:
    """One formal expert plus an optional diagnostic transformation."""

    mode: CausalComputeMode
    control: CausalControlKind
    treatment_chunks: int
    continuation_mode: CausalComputeMode
    replicate: int
    action_seed: int
    video_seed: int | None
    donor_state_id: str | None = None
    donor_episode_id: str | None = None
    donor_instruction_id: str | None = None
    generic_proposal_count: int | None = None
    generic_medoid_index: int | None = None
    privileged_offline: bool = False

    def __post_init__(self) -> None:
        mode = CausalComputeMode.parse(self.mode)
        continuation = CausalComputeMode.parse(self.continuation_mode)
        control = CausalControlKind.parse(self.control)
        object.__setattr__(self, "mode", mode)
        object.__setattr__(self, "continuation_mode", continuation)
        object.__setattr__(self, "control", control)
        if mode is CausalComputeMode.G_NO_READ or not continuation.is_routable:
            raise ValueError("V2 specs use formal routable experts only.")
        if self.treatment_chunks not in {1, 2, 4}:
            raise ValueError("Treatment duration must be one, two, or four chunks.")
        if self.treatment_chunks != 1 and control is not CausalControlKind.STANDARD:
            raise ValueError("Mechanism controls are single-chunk diagnostics.")
        if min(self.replicate, self.action_seed) < 0:
            raise ValueError("Replicate and action seed must be non-negative.")
        if self.video_seed is not None and self.video_seed < 0:
            raise ValueError("Video seed must be non-negative when present.")
        if mode.runs_future_prediction and self.video_seed is None:
            raise ValueError("Future-prediction interventions require a video seed.")
        c2_controls = {
            CausalControlKind.NO_READ,
            CausalControlKind.REPEAT_CURRENT,
            CausalControlKind.SHUFFLED_WRONG_STATE,
            CausalControlKind.TEMPORAL_SHIFT,
            CausalControlKind.INSTRUCTION_MISMATCH,
            CausalControlKind.GT_FUTURE_OFFLINE,
        }
        if control in c2_controls and mode is not CausalComputeMode.C2_FULL:
            raise ValueError(f"Control {control.value} requires the C2 expert.")
        if control is CausalControlKind.GENERIC_MEDOID:
            if mode is not CausalComputeMode.C0_CURRENT:
                raise ValueError("Generic medoid compute uses only C0 proposals.")
            if self.generic_proposal_count is None or self.generic_proposal_count < 1:
                raise ValueError("Generic medoid requires a positive proposal count.")
            if self.generic_medoid_index is not None and not (
                0 <= self.generic_medoid_index < self.generic_proposal_count
            ):
                raise ValueError("Generic medoid index lies outside the proposal set.")
        elif (
            self.generic_proposal_count is not None
            or self.generic_medoid_index is not None
        ):
            raise ValueError("Generic proposal fields are exclusive to generic medoid.")
        donor_controls = {
            CausalControlKind.SHUFFLED_WRONG_STATE,
            CausalControlKind.TEMPORAL_SHIFT,
            CausalControlKind.INSTRUCTION_MISMATCH,
        }
        if control in donor_controls and not any(
            (self.donor_state_id, self.donor_episode_id, self.donor_instruction_id)
        ):
            raise ValueError(f"Control {control.value} requires donor identity.")
        if self.privileged_offline != (control is CausalControlKind.GT_FUTURE_OFFLINE):
            raise ValueError("Only GT-future diagnostics may be privileged offline.")

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-compatible intervention payload."""

        return _json_value(asdict(self))


@dataclass(frozen=True)
class CausalOutcomeV2:
    """Structured branch outcome without inferred failure labels."""

    predicate_before: tuple[bool, ...]
    predicate_after_treatment: tuple[bool, ...]
    predicate_terminal: tuple[bool, ...]
    progress_before: float
    progress_after_treatment: float
    progress_terminal: float
    final_success: bool
    final_return: float
    first_success_step: int | None
    completion_step: int
    termination_type: CausalTerminationType
    contact_events: Mapping[str, int]
    treatment_submitted_action_count: int
    continuation_submitted_action_count: int
    treatment_action_audit: Mapping[str, Any]
    continuation_action_audit: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "termination_type", CausalTerminationType(self.termination_type)
        )
        predicate_lengths = {
            len(self.predicate_before),
            len(self.predicate_after_treatment),
            len(self.predicate_terminal),
        }
        if len(predicate_lengths) != 1:
            raise ValueError("Predicate vectors must have identical lengths.")
        for name in (
            "progress_before",
            "progress_after_treatment",
            "progress_terminal",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"`{name}` must lie in [0, 1].")
        if not math.isfinite(float(self.final_return)):
            raise ValueError("Final return must be finite.")
        if self.completion_step < 0 or (
            self.first_success_step is not None and self.first_success_step < 0
        ):
            raise ValueError("Outcome step counters must be non-negative.")
        if (
            min(
                self.treatment_submitted_action_count,
                self.continuation_submitted_action_count,
            )
            < 0
        ):
            raise ValueError("Submitted action counts must be non-negative.")
        if any(int(value) < 0 for value in self.contact_events.values()):
            raise ValueError("Contact-event counts must be non-negative.")

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-compatible outcome payload."""

        return _json_value(asdict(self))


@dataclass(frozen=True)
class CausalCostV2:
    """Segmented critical-path cost and exact operation counts."""

    treatment_latency_ms: Mapping[str, float]
    continuation_latency_ms: Mapping[str, float]
    total_latency_ms: Mapping[str, float]
    treatment_calls: Mapping[str, int]
    continuation_calls: Mapping[str, int]
    episode_gpu_seconds: float

    def __post_init__(self) -> None:
        for label, latency in (
            ("treatment", self.treatment_latency_ms),
            ("continuation", self.continuation_latency_ms),
            ("total", self.total_latency_ms),
        ):
            if any(
                not math.isfinite(float(value)) or float(value) < 0
                for value in latency.values()
            ):
                raise ValueError(f"{label} latency must be finite and non-negative.")
        keys = set(self.treatment_latency_ms) | set(self.continuation_latency_ms)
        if set(self.total_latency_ms) != keys:
            raise ValueError("Total latency keys must equal the union of segment keys.")
        for key in keys:
            expected = float(self.treatment_latency_ms.get(key, 0.0)) + float(
                self.continuation_latency_ms.get(key, 0.0)
            )
            if not math.isclose(
                float(self.total_latency_ms.get(key, 0.0)),
                expected,
                rel_tol=1e-9,
                abs_tol=1e-9,
            ):
                raise ValueError(f"Total latency does not equal segment sum for {key}.")
        for counts in (self.treatment_calls, self.continuation_calls):
            if any(int(value) < 0 for value in counts.values()):
                raise ValueError("Operation counts must be non-negative.")
        _finite_nonnegative(self.episode_gpu_seconds, name="episode_gpu_seconds")

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-compatible cost payload."""

        return _json_value(asdict(self))


@dataclass(frozen=True)
class PairedInterventionRecordV2:
    """One v2 restored branch with typed identity, outcome, and cost."""

    identity: CausalStateIdentityV2
    sampling: CausalSamplingMetadataV2
    intervention: CausalInterventionSpecV2
    outcome: CausalOutcomeV2
    cost: CausalCostV2
    treatment_submitted_actions: tuple[tuple[float, ...], ...]
    continuation_submitted_actions: tuple[tuple[float, ...], ...]
    secondary_outcomes: Mapping[str, Any]

    def __post_init__(self) -> None:
        all_actions = (
            *self.treatment_submitted_actions,
            *self.continuation_submitted_actions,
        )
        if any(
            len(action) != 7 or any(not math.isfinite(float(value)) for value in action)
            for action in all_actions
        ):
            raise ValueError("Submitted LIBERO actions must be finite seven-vectors.")
        if len(self.treatment_submitted_actions) != (
            self.outcome.treatment_submitted_action_count
        ):
            raise ValueError("Treatment action count does not match the outcome.")
        if len(self.continuation_submitted_actions) != (
            self.outcome.continuation_submitted_action_count
        ):
            raise ValueError("Continuation action count does not match the outcome.")

    @property
    def record_key(self) -> str:
        """Return the unique resumable branch key."""

        spec = self.intervention
        return (
            f"{self.identity.snapshot_id}:{spec.mode.value}:{spec.control.value}:"
            f"k{spec.treatment_chunks}:r{spec.replicate}:"
            f"cont-{spec.continuation_mode.value}"
        )

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-compatible versioned artifact payload."""

        return {
            "schema": PAIRED_INTERVENTION_RECORD_SCHEMA_V2,
            "record_key": self.record_key,
            "identity": self.identity.to_artifact(),
            "sampling": self.sampling.to_artifact(),
            "intervention": self.intervention.to_artifact(),
            "outcome": self.outcome.to_artifact(),
            "cost": self.cost.to_artifact(),
            "treatment_submitted_actions": _json_value(
                self.treatment_submitted_actions
            ),
            "continuation_submitted_actions": _json_value(
                self.continuation_submitted_actions
            ),
            "secondary_outcomes": _json_value(self.secondary_outcomes),
        }


@dataclass(frozen=True)
class CausalGateFeatureRecordV1:
    """Index from one pre-treatment state to a detached tensor-shard row."""

    state: CausalStateIdentityV2
    tensor_shard: str
    tensor_row: int
    proposal_variant: str
    feature_names: tuple[str, ...]
    pre_prediction: bool = True

    def __post_init__(self) -> None:
        if not self.tensor_shard or self.tensor_row < 0:
            raise ValueError("Gate feature shard identity is invalid.")
        if self.proposal_variant not in {"one_proposal", "two_proposal"}:
            raise ValueError("Unknown Gate proposal variant.")
        if not self.pre_prediction:
            raise ValueError("Gate features must be captured before prediction.")
        allowed = {
            "current_video_kv",
            "language",
            "proprio",
            "history",
            "action_proposal",
            "proposal_disagreement",
            "remaining_budget",
            "previous_mode",
            "steps_to_go",
        }
        if not self.feature_names or not set(self.feature_names) <= allowed:
            raise ValueError("Gate feature names contain a non-preregistered signal.")

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-compatible Gate feature index row."""

        return {
            "schema": GATE_FEATURE_RECORD_SCHEMA,
            **_json_value(asdict(self)),
        }


@dataclass(frozen=True)
class ModeOutcomeSummaryV1:
    """Replicate-aggregated empirical label for one expert mode."""

    mode: CausalComputeMode
    success_mean: float
    success_variance: float
    predicate_progress_mean: float
    replicate_count: int

    def __post_init__(self) -> None:
        mode = CausalComputeMode.parse(self.mode)
        object.__setattr__(self, "mode", mode)
        if not mode.is_routable:
            raise ValueError("Gate labels may contain only routable experts.")
        if self.replicate_count < 1:
            raise ValueError("Gate labels require at least one replicate.")
        if not 0.0 <= float(self.success_mean) <= 1.0:
            raise ValueError("Success mean must lie in [0, 1].")
        if not 0.0 <= float(self.predicate_progress_mean) <= 1.0:
            raise ValueError("Predicate progress mean must lie in [0, 1].")
        _finite_nonnegative(self.success_variance, name="success_variance")


@dataclass(frozen=True)
class GateTrainingExampleV1:
    """Fold-aware state label joined to one pre-treatment feature row."""

    state: CausalStateIdentityV2
    feature_shard: str
    feature_row: int
    proposal_variant: str
    outcomes: tuple[ModeOutcomeSummaryV1, ...]
    empirical_uplift: Mapping[str, float]
    inclusion_weight: float
    fold: int
    split: str
    replicate_joint_outcomes: tuple[Mapping[str, bool], ...]

    def __post_init__(self) -> None:
        if self.feature_row < 0 or not self.feature_shard:
            raise ValueError("Gate example feature identity is invalid.")
        if self.fold not in range(5) or self.split not in {
            "train",
            "validation",
            "test",
        }:
            raise ValueError("Gate example fold/split is invalid.")
        modes = tuple(item.mode for item in self.outcomes)
        if not modes or len(set(modes)) != len(modes):
            raise ValueError("Gate example outcome modes must be unique.")
        if set(self.empirical_uplift) != {mode.value for mode in modes}:
            raise ValueError("Empirical uplift heads do not match outcome modes.")
        if (
            not math.isfinite(float(self.inclusion_weight))
            or self.inclusion_weight <= 0
        ):
            raise ValueError("Gate inclusion weight must be finite and positive.")

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-compatible Gate training example."""

        return {
            "schema": GATE_TRAINING_EXAMPLE_SCHEMA,
            **_json_value(asdict(self)),
        }


@dataclass(frozen=True)
class PairedInterventionRecordV1:
    """One branch result from a restored causal snapshot."""

    snapshot_id: str
    environment: str
    task_id: int
    trial_id: int
    reset_id: int
    chunk_index: int
    mode: CausalComputeMode
    replicate: int
    action_seed: int
    video_seed: int | None
    continuation: str
    source_policy: str
    inclusion_probability: float
    final_success: bool
    final_return: float
    remaining_steps: int
    submitted_action_count: int
    submitted_actions: tuple[tuple[float, ...], ...]
    latency_ms: Mapping[str, float]
    secondary_outcomes: Mapping[str, Any]

    def __post_init__(self) -> None:
        object.__setattr__(self, "mode", CausalComputeMode.parse(self.mode))
        if not self.snapshot_id:
            raise ValueError("`snapshot_id` must be non-empty.")
        if self.replicate < 0:
            raise ValueError("`replicate` must be non-negative.")
        if not 0.0 < self.inclusion_probability <= 1.0:
            raise ValueError("`inclusion_probability` must lie in (0, 1].")
        if self.remaining_steps < 0:
            raise ValueError("`remaining_steps` must be non-negative.")
        if self.submitted_action_count < 0:
            raise ValueError("`submitted_action_count` must be non-negative.")
        if len(self.submitted_actions) != self.submitted_action_count:
            raise ValueError("Submitted action rows must match their declared count.")
        if any(
            len(action) != 7 or any(not math.isfinite(float(value)) for value in action)
            for action in self.submitted_actions
        ):
            raise ValueError("Submitted LIBERO actions must be finite seven-vectors.")
        bad_latency = {
            key: value
            for key, value in self.latency_ms.items()
            if not isinstance(value, (int, float)) or value < 0
        }
        if bad_latency:
            raise ValueError(f"Latency components must be non-negative: {bad_latency}")

    def to_artifact(self) -> dict[str, Any]:
        """Return a JSON-compatible versioned artifact payload."""

        payload = asdict(self)
        payload["schema"] = PAIRED_INTERVENTION_RECORD_SCHEMA
        payload["mode"] = self.mode.value
        payload["latency_ms"] = dict(self.latency_ms)
        payload["secondary_outcomes"] = dict(self.secondary_outcomes)
        return payload


@dataclass(frozen=True)
class UpliftGateOutput:
    """Per-mode outcome, uplift, uncertainty, and measured-cost predictions."""

    modes: tuple[CausalComputeMode, ...]
    q_values: torch.Tensor
    uplift: torch.Tensor
    uncertainty: torch.Tensor
    normalized_cost: torch.Tensor

    def __post_init__(self) -> None:
        modes = tuple(CausalComputeMode.parse(mode) for mode in self.modes)
        if not modes or any(not mode.is_routable for mode in modes):
            raise ValueError("Gate output modes must be non-empty routable modes.")
        if len(set(modes)) != len(modes):
            raise ValueError("Gate output modes must be unique.")
        object.__setattr__(self, "modes", modes)
        tensors = (
            self.q_values,
            self.uplift,
            self.uncertainty,
            self.normalized_cost,
        )
        if any(not isinstance(value, torch.Tensor) for value in tensors):
            raise TypeError("All UpliftGateOutput numeric fields must be tensors.")
        if any(value.shape != self.q_values.shape for value in tensors[1:]):
            raise ValueError("All UpliftGateOutput tensors must have identical shapes.")
        if self.q_values.ndim != 2 or self.q_values.shape[-1] != len(modes):
            raise ValueError(
                "Gate tensors must have shape [batch, number_of_modes], got "
                f"{tuple(self.q_values.shape)} for {len(modes)} modes."
            )
        if not all(torch.isfinite(value).all() for value in tensors):
            raise ValueError("Gate outputs must be finite.")
        if (self.uncertainty < 0).any() or (self.normalized_cost < 0).any():
            raise ValueError(
                "Gate uncertainty and normalized cost must be non-negative."
            )

    def utilities(self, *, beta: float, cost_weight: float) -> torch.Tensor:
        """Return uncertainty- and cost-adjusted routing utilities."""

        if beta < 0 or cost_weight < 0:
            raise ValueError("`beta` and `cost_weight` must be non-negative.")
        return (
            self.q_values - beta * self.uncertainty - cost_weight * self.normalized_cost
        )
