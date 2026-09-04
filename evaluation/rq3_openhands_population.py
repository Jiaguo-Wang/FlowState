"""构造正式 RQ3 OpenHands population 的确定性采样与 epoch 分配。

本模块只包含纯 CPU 的确定性逻辑：

- dataset-level eligibility（n_turns、assistant turns、replay 输入长度上限）
- 固定 seed 的 SHA-256 会话排序
- 连续 4 会话组成互不重叠 workflow group，组内 A/B/C/D 标记
- Main 前 200 groups 与 reserved sensitivity 后 100 groups 的划分
- group ordinal 到 allocation round（2/3/4/5）的确定性轮转

本模块不读取 runtime、不执行任何 policy、不包含任何未来信息。
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Mapping, Sequence

import pyarrow.parquet as pq


# 与 evaluation/openhands_single_workflow_smoke.py 中冻结路径保持一致；
# collector 启动时会与正式常量做一致性断言。
DEFAULT_DATASET_PATH = Path(
    "/home/wjg/data/agentic_coding_trajectories/sessions.parquet"
)

PROTOCOL_VERSION = "rq3-openhands-v1"
POPULATION_SEED = 20260903
SOURCE_DATASET = "nebius-swe-rebench-openhands"
MIN_N_TURNS = 60
MAX_REPLAY_INPUT_TOKENS = 131_072
WORKFLOWS_PER_GROUP = 4
MAIN_GROUP_COUNT = 200
SENSITIVITY_GROUP_COUNT = 100
ALLOCATION_ROUNDS = (2, 3, 4, 5)
WORKFLOW_LABELS = ("A", "B", "C", "D")
# 最大 round 为 5，snapshot 还需要 round 6 的 pending 输入。
REQUIRED_ASSISTANT_TURNS = max(ALLOCATION_ROUNDS) + 1


@dataclass(frozen=True)
class SessionEligibility:
    """记录单个 session 的 dataset-level eligibility 判定与证据。"""

    session_id: str
    n_turns: int
    assistant_turns: int
    replay_input_tokens: tuple[int, ...]
    eligible: bool
    reason: str


@dataclass(frozen=True)
class WorkflowGroup:
    """保存一个互不重叠的 4-workflow group 的冻结组成。"""

    group_ordinal: int
    population_segment: str
    allocation_round: int
    session_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if len(self.session_ids) != WORKFLOWS_PER_GROUP:
            raise ValueError("workflow group 必须恰好包含 4 个 session")
        if len(set(self.session_ids)) != WORKFLOWS_PER_GROUP:
            raise ValueError("workflow group 内 session 不能重复")
        if self.population_segment not in ("main", "sensitivity", "unused"):
            raise ValueError("population_segment 必须是 main、sensitivity 或 unused")
        if self.allocation_round not in ALLOCATION_ROUNDS:
            raise ValueError("allocation_round 必须在 2/3/4/5 之中")

    @property
    def session_by_label(self) -> dict[str, str]:
        """返回按 hash 顺序冻结的 A/B/C/D 到 session 的映射。"""

        return {
            label: session_id
            for label, session_id in zip(WORKFLOW_LABELS, self.session_ids)
        }


def session_order_digest(
    session_id: str,
    *,
    seed: int = POPULATION_SEED,
    protocol_version: str = PROTOCOL_VERSION,
) -> str:
    """按冻结格式计算会话排序摘要。"""

    material = f"{protocol_version}|{seed}|{session_id}"
    return sha256(material.encode("utf-8")).hexdigest()


def order_sessions_by_digest(
    session_ids: Sequence[str],
    *,
    seed: int = POPULATION_SEED,
    protocol_version: str = PROTOCOL_VERSION,
) -> tuple[str, ...]:
    """按排序摘要升序返回全部 eligible session。"""

    return tuple(
        sorted(
            (str(session_id) for session_id in session_ids),
            key=lambda value: session_order_digest(
                value,
                seed=seed,
                protocol_version=protocol_version,
            ),
        )
    )


def allocation_round_for_ordinal(group_ordinal: int) -> int:
    """按确定性轮转把 group ordinal 映射到 allocation round。"""

    if group_ordinal < 0:
        raise ValueError("group_ordinal 必须是非负整数")
    return ALLOCATION_ROUNDS[group_ordinal % len(ALLOCATION_ROUNDS)]


def build_workflow_groups(
    ordered_session_ids: Sequence[str],
) -> tuple[WorkflowGroup, ...]:
    """按冻结顺序把有序会话切成互不重叠的 4-workflow groups。"""

    ordered = tuple(str(value) for value in ordered_session_ids)
    group_total = len(ordered) // WORKFLOWS_PER_GROUP
    groups = []
    for ordinal in range(group_total):
        members = ordered[
            ordinal
            * WORKFLOWS_PER_GROUP : (ordinal + 1)
            * WORKFLOWS_PER_GROUP
        ]
        if ordinal < MAIN_GROUP_COUNT:
            segment = "main"
        elif ordinal < MAIN_GROUP_COUNT + SENSITIVITY_GROUP_COUNT:
            segment = "sensitivity"
        else:
            segment = "unused"
        groups.append(
            WorkflowGroup(
                group_ordinal=ordinal,
                population_segment=segment,
                allocation_round=allocation_round_for_ordinal(ordinal),
                session_ids=tuple(members),
            )
        )
    return tuple(groups)


def designated_main_groups(
    groups: Sequence[WorkflowGroup],
) -> tuple[WorkflowGroup, ...]:
    """返回冻结的前 200 个 Main groups，不做任何补位。"""

    main = tuple(group for group in groups if group.population_segment == "main")
    if len(main) != MAIN_GROUP_COUNT:
        raise ValueError(
            f"Main population 必须是 {MAIN_GROUP_COUNT} 个 group，实际 {len(main)}"
        )
    if tuple(group.group_ordinal for group in main) != tuple(
        range(MAIN_GROUP_COUNT)
    ):
        raise ValueError("Main groups 必须严格是 ordinal 0..199")
    return main


def reserved_sensitivity_groups(
    groups: Sequence[WorkflowGroup],
) -> tuple[WorkflowGroup, ...]:
    """返回保留给 sensitivity 的 100 个 groups，本步骤禁止使用。"""

    return tuple(
        group for group in groups if group.population_segment == "sensitivity"
    )


def load_source_session_rows(
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> list[dict[str, object]]:
    """读取冻结数据子集的全部 session_id 与 n_turns。"""

    table = pq.read_table(
        dataset_path,
        filters=[("source_dataset", "=", SOURCE_DATASET)],
        columns=["session_id", "n_turns"],
    )
    rows = []
    for row in table.to_pylist():
        rows.append(
            {
                "session_id": str(row["session_id"]),
                "n_turns": int(row["n_turns"]),
            }
        )
    return rows


def load_session_messages(
    session_id: str,
    dataset_path: Path = DEFAULT_DATASET_PATH,
) -> list[dict[str, object]]:
    """读取单个 session 的原始 messages 列表。"""

    table = pq.read_table(
        dataset_path,
        filters=[("session_id", "=", session_id)],
        columns=["session_id", "messages_json"],
    )
    if table.num_rows != 1:
        raise RuntimeError(
            f"session 应唯一命中一行，实际为 {table.num_rows} 行"
        )
    raw_messages = json.loads(table.column("messages_json")[0].as_py())
    if not isinstance(raw_messages, list):
        raise TypeError("messages_json 反序列化后必须是列表")
    return raw_messages


def count_assistant_turns(raw_messages: Sequence[Mapping[str, object]]) -> int:
    """统计一个 session 的 assistant 消息数。"""

    return sum(
        1 for message in raw_messages if message.get("role") == "assistant"
    )


def replay_input_token_lengths(
    tokenizer: object,
    raw_messages: Sequence[Mapping[str, object]],
    *,
    normalize_message: object,
    template_input_ids: object,
    max_assistant_turn: int = REQUIRED_ASSISTANT_TURNS,
) -> tuple[tuple[int, ...], dict[str, object]]:
    """计算前 max_assistant_turn 个 assistant turn 的 replay 输入长度。

    只消费到第 max_assistant_turn 个 assistant 消息之前的 history；
    该 assistant 消息本身的输出内容不进入任何输入。
    """

    history: list[dict[str, object]] = []
    lengths: list[int] = []
    assistant_turn = 0
    raw_items_iterated = 0
    for raw_message in raw_messages:
        raw_items_iterated += 1
        if raw_message.get("role") == "assistant":
            assistant_turn += 1
            if assistant_turn <= max_assistant_turn:
                output = tokenizer.apply_chat_template(
                    list(history),
                    tokenize=True,
                    add_generation_prompt=True,
                )
                lengths.append(len(template_input_ids(output)))
            if assistant_turn == max_assistant_turn:
                break
        history.append(normalize_message(raw_message))
    audit = {
        "maximum_assistant_turn_consumed": assistant_turn,
        "raw_items_iterated": raw_items_iterated,
        "pending_turn_output_read": False,
        "beyond_boundary_message_consumed": False,
    }
    return tuple(lengths), audit


def evaluate_session_eligibility(
    session_id: str,
    n_turns: int,
    assistant_turns: int,
    replay_input_tokens: Sequence[int],
) -> SessionEligibility:
    """按冻结条件判定单个 session 的 dataset-level eligibility。"""

    lengths = tuple(int(value) for value in replay_input_tokens)
    reason = "eligible"
    eligible = True
    if n_turns < MIN_N_TURNS:
        eligible = False
        reason = "n_turns_below_60"
    elif assistant_turns < REQUIRED_ASSISTANT_TURNS:
        eligible = False
        reason = "assistant_turns_insufficient"
    elif len(lengths) < REQUIRED_ASSISTANT_TURNS:
        eligible = False
        reason = "replay_inputs_incomplete"
    elif any(length > MAX_REPLAY_INPUT_TOKENS for length in lengths):
        eligible = False
        reason = "replay_input_exceeds_131072"
    return SessionEligibility(
        session_id=str(session_id),
        n_turns=int(n_turns),
        assistant_turns=int(assistant_turns),
        replay_input_tokens=lengths,
        eligible=eligible,
        reason=reason,
    )


def k_sweep_for_candidate_count(candidate_count: int) -> tuple[int, ...]:
    """按冻结 K protocol 计算去重后的评价预算集合。"""

    if candidate_count <= 0:
        raise ValueError("candidate_count 必须是正整数")
    values = {
        max(1, int(candidate_count * 25 // 100)),
        max(1, int(candidate_count * 50 // 100)),
        max(1, int(candidate_count * 75 // 100)),
        2,
    }
    # 0.25/0.50/0.75 的 floor 语义与冻结 protocol 的 floor(r*|C|) 一致。
    sweep = tuple(sorted(values))
    if any(value >= candidate_count for value in sweep):
        raise ValueError("存在不小于 |C_t| 的 trivial K，不进入主比较")
    if any(value < 1 for value in sweep):
        raise ValueError("K 必须至少为 1")
    return sweep


def reference_budget_for_candidate_count(candidate_count: int) -> int:
    """返回写入 snapshot 的 canonical 参考预算（最大主相对预算）。"""

    if candidate_count < 8:
        raise ValueError("正式 snapshot 要求 |C_t| >= 8")
    return max(1, int(candidate_count * 75 // 100))
