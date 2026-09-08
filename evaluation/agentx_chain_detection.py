"""Lightweight, pure-stdlib+numpy chain detection for AgentX Weka traces.

This module mirrors the AIPerf ``weka_agent_chains.py`` and the relevant
helpers from ``weka_trace.py`` without importing pydantic or the AIPerf
package, so it can run inside the FlowState test virtual environment.

All credit for the algorithm belongs to the AIPerf project
(https://github.com/ai-dynamo/aiperf), Apache-2.0 licensed. The code here
is a mechanical port that replaces pydantic models with plain dataclasses
and dict-backed request views.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

_EPSILON_SECONDS = 1e-6
_TITLE_GEN_MAX_OUTPUT_TOKENS = 64

# Defaults copied from AIPerf Environment.DATASET.
DEFAULT_SEAM_MAX_GAP_SECONDS = 3600.0
DEFAULT_SEAM_MIN_OVERLAP_RATIO = 0.5
DEFAULT_AUX_MAX_REQUESTS = 1
DEFAULT_AUX_ISL_RATIO = 0.10
DEFAULT_AUX_ISL_FLOOR = 16384
DEFAULT_AUX_CROSS_MODEL = True
DEFAULT_AUX_REDUCTION_OSL_MAX = 4000
DEFAULT_AUX_REDUCTION_RATIO = 20.0
DEFAULT_WORKER_GROUP_MIN = 3


@dataclass(slots=True, frozen=True)
class _Req:
    """Minimal read-only view of a Weka normal/streaming request."""

    t: float
    req_type: str
    model: str
    input_length: int
    output_length: int
    hash_ids: list[int]
    api_time: float | None

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "_Req":
        api_time = value.get("api_time")
        if api_time is not None and not isinstance(api_time, (int, float)):
            api_time = None
        return cls(
            t=float(value["t"]),
            req_type=str(value.get("type", "n")),
            model=str(value["model"]),
            input_length=int(value.get("in", 0)),
            output_length=int(value.get("out", 0)),
            hash_ids=list(value.get("hash_ids", [])),
            api_time=float(api_time) if api_time is not None else None,
        )


IndexedRequest = tuple[int, _Req]


@dataclass(slots=True)
class ChainFork:
    """Where a chain split off another chain."""

    parent_chain: int | None
    fork_outer_idx: int | None
    depth: int
    fork_time: float


@dataclass(slots=True)
class AgentChain:
    """One detected agent: a time-ordered run of requests."""

    requests: list[IndexedRequest] = field(default_factory=list)
    fork: ChainFork | None = None
    spliced_into: int | None = None
    tail_outer_idx: int = -1
    tail_hash: np.ndarray = field(default_factory=lambda: np.empty(0, np.int64))
    tail_end: float = 0.0
    tail_model: str = ""


@dataclass(slots=True)
class ChainDetectionResult:
    """Output of :func:`detect_agent_chains`."""

    chains: list[AgentChain]
    main_index: int
    worker_indices: list[int]
    seams_merged: int
    unclassified_empty_hash: int


def _req_end(req: _Req) -> float:
    """Interval end in seconds; missing/non-finite durations become zero."""
    duration = (
        req.api_time
        if req.api_time is not None and math.isfinite(req.api_time)
        else 0.0
    )
    return req.t + max(duration, 0.0)


def _np_lcp(a: np.ndarray, b: np.ndarray) -> int:
    """Length of the longest common prefix of two int64 hash arrays."""
    n = min(a.shape[0], b.shape[0])
    if n == 0:
        return 0
    neq = a[:n] != b[:n]
    i = int(neq.argmax())
    return i if neq[i] else n


def _hash_list_lcp(a: list[int], b: list[int]) -> int:
    """Length of the longest common prefix of two hash-id lists."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def is_aux_chain(
    requests: list[_Req],
    main_peak_isl: int,
    *,
    max_requests: int = DEFAULT_AUX_MAX_REQUESTS,
    isl_ratio: float = DEFAULT_AUX_ISL_RATIO,
    isl_floor: int = DEFAULT_AUX_ISL_FLOOR,
    main_model: str | None = None,
    cross_model: bool = DEFAULT_AUX_CROSS_MODEL,
) -> bool:
    """Classify a detected worker chain as an auxiliary one-shot call.

    Mirrors AIPerf ``weka_agent_chains.is_aux_chain``.
    """
    if not requests or len(requests) > max_requests:
        return False
    first = requests[0]
    if cross_model and main_model is not None and first.model != main_model:
        return True
    threshold = max(isl_floor, int(isl_ratio * main_peak_isl))
    return first.input_length < threshold


def is_reduction_chain(
    requests: list[_Req],
    *,
    osl_max: int = DEFAULT_AUX_REDUCTION_OSL_MAX,
    ratio: float = DEFAULT_AUX_REDUCTION_RATIO,
    isl_floor: int = DEFAULT_AUX_ISL_FLOOR,
) -> bool:
    """Classify a single-request worker chain as an auxiliary reduction call.

    Mirrors AIPerf ``weka_agent_chains.is_reduction_chain``.
    """
    if osl_max <= 0 or len(requests) != 1:
        return False
    first = requests[0]
    osl = first.output_length or 0
    if osl <= 0 or osl >= osl_max:
        return False
    if first.input_length < isl_floor:
        return False
    return first.input_length > ratio * osl


_IntervalCand = tuple[float, float, int, int]


def _overlap_components(cands: list[_IntervalCand]) -> list[list[_IntervalCand]]:
    """Connected components of [t0, t1) interval overlap."""
    if not cands:
        return []
    order = sorted(range(len(cands)), key=lambda i: (cands[i][0], cands[i][2]))
    parent = list(range(len(cands)))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    active: list[tuple[float, int]] = []
    for i in order:
        t0 = cands[i][0]
        while active and active[0][0] <= t0:
            heapq.heappop(active)
        if active:
            parent[_find(i)] = _find(active[0][1])
        heapq.heappush(active, (cands[i][1], i))

    comps: dict[int, list[_IntervalCand]] = {}
    for i in range(len(cands)):
        comps.setdefault(_find(i), []).append(cands[i])
    return list(comps.values())


def worker_group_assignment(
    result: ChainDetectionResult,
    *,
    group_min: int = DEFAULT_WORKER_GROUP_MIN,
) -> dict[int, tuple[int, int]]:
    """Assign each worker-group member a (group, member) coordinate.

    Mirrors AIPerf ``weka_agent_chains.worker_group_assignment``.
    """
    if group_min <= 0:
        return {}

    buckets: dict[tuple[int, int], list[_IntervalCand]] = {}
    for ci in result.worker_indices:
        chain = result.chains[ci]
        fork = chain.fork
        if (
            fork is None
            or fork.depth <= 0
            or fork.parent_chain is None
            or fork.fork_outer_idx is None
            or not chain.requests
        ):
            continue
        t0 = chain.requests[0][1].t
        t1 = max(_req_end(req) for _, req in chain.requests)
        key = (fork.parent_chain, fork.fork_outer_idx)
        buckets.setdefault(key, []).append((t0, t1, chain.requests[0][0], ci))

    components: list[list[_IntervalCand]] = []
    for cands in buckets.values():
        components.extend(_overlap_components(cands))

    groups = [comp for comp in components if len(comp) >= group_min]
    groups.sort(key=lambda comp: min((c[0], c[2]) for c in comp))

    out: dict[int, tuple[int, int]] = {}
    for group, comp in enumerate(groups):
        comp.sort(key=lambda c: (c[0], c[2]))
        for member, (_, _, _, ci) in enumerate(comp):
            out[ci] = (group, member)
    return out


def worker_group_members(
    result: ChainDetectionResult, *, group_min: int = DEFAULT_WORKER_GROUP_MIN
) -> set[int]:
    """Worker chain indices belonging to a parallel fan-out group."""
    return set(worker_group_assignment(result, group_min=group_min))


@dataclass(slots=True)
class _Phase1State:
    """Working state for the greedy forward pass."""

    chains: list[AgentChain] = field(default_factory=list)
    chain_of_request: dict[int, int] = field(default_factory=dict)
    forks_by_tail: dict[int, list[int]] = field(default_factory=dict)
    req_by_outer: dict[int, _Req] = field(default_factory=dict)
    unclassified: int = 0

    def _append(self, chain_idx: int, outer_idx: int, req: _Req) -> None:
        c = self.chains[chain_idx]
        c.requests.append((outer_idx, req))
        self.chain_of_request[outer_idx] = chain_idx
        if req.hash_ids:
            c.tail_outer_idx = outer_idx
            c.tail_hash = np.asarray(req.hash_ids, dtype=np.int64)
            c.tail_end = _req_end(req)
            c.tail_model = req.model

    def classify(self, outer_idx: int, req: _Req) -> None:
        self.req_by_outer[outer_idx] = req
        if not req.hash_ids:
            self.unclassified += 1
            if not self.chains:
                self.chains.append(AgentChain())
            self.chains[0].requests.append((outer_idx, req))
            self.chain_of_request[outer_idx] = 0
            return

        h = np.asarray(req.hash_ids, dtype=np.int64)
        if not self.chains:
            self.chains.append(AgentChain())
            self._append(0, outer_idx, req)
            return
        target = _find_extension_target(self.chains, h, req.t, req.model)
        if target is not None:
            self._append(target, outer_idx, req)
            return

        parent, depth = _max_lcp_chain(self.chains, h)
        if parent is None and all(c.tail_hash.shape[0] == 0 for c in self.chains):
            self._append(0, outer_idx, req)
            return
        fork = ChainFork(
            parent_chain=parent,
            fork_outer_idx=(
                self.chains[parent].tail_outer_idx if parent is not None else None
            ),
            depth=depth,
            fork_time=req.t,
        )
        new_idx = len(self.chains)
        self.chains.append(AgentChain(fork=fork))
        self._append(new_idx, outer_idx, req)
        if fork.fork_outer_idx is not None and depth > 0:
            self.forks_by_tail.setdefault(fork.fork_outer_idx, []).append(new_idx)


def detect_agent_chains(
    normals: list[IndexedRequest],
    *,
    seam_max_gap_seconds: float = DEFAULT_SEAM_MAX_GAP_SECONDS,
    seam_min_overlap_ratio: float = DEFAULT_SEAM_MIN_OVERLAP_RATIO,
) -> ChainDetectionResult:
    """Partition retained requests into per-agent chains.

    Mirrors AIPerf ``weka_agent_chains.detect_agent_chains``.
    """
    if not normals:
        return ChainDetectionResult(
            chains=[],
            main_index=0,
            worker_indices=[],
            seams_merged=0,
            unclassified_empty_hash=0,
        )

    ordered = sorted(normals, key=lambda item: (item[1].t, item[0]))

    state = _Phase1State()
    for outer_idx, req in ordered:
        state.classify(outer_idx, req)
    chains = state.chains

    seams = _resolve_seams(
        chains,
        state.forks_by_tail,
        state.chain_of_request,
        state.req_by_outer,
        max_gap_seconds=seam_max_gap_seconds,
        min_overlap_ratio=seam_min_overlap_ratio,
    )

    alias = {
        i: c.spliced_into for i, c in enumerate(chains) if c.spliced_into is not None
    }

    def _resolve(i: int) -> int:
        while i in alias:
            i = alias[i]
        return i

    for c in chains:
        if (
            c.spliced_into is None
            and c.fork is not None
            and c.fork.parent_chain is not None
        ):
            c.fork.parent_chain = _resolve(c.fork.parent_chain)

    main_index = _resolve(state.chain_of_request[ordered[0][0]])
    workers = [
        i for i, c in enumerate(chains) if c.spliced_into is None and i != main_index
    ]
    workers.sort(key=lambda i: (chains[i].requests[0][1].t, chains[i].requests[0][0]))
    return ChainDetectionResult(
        chains=chains,
        main_index=main_index,
        worker_indices=workers,
        seams_merged=seams,
        unclassified_empty_hash=state.unclassified,
    )


def _find_extension_target(
    chains: list[AgentChain], h: np.ndarray, t: float, model: str
) -> int | None:
    """Chain whose tail is a complete prefix of ``h``, has ended by ``t``,
    and ran on the same ``model``. Deepest tail wins."""
    best: int | None = None
    best_len = -1
    hn = h.shape[0]
    for idx, c in enumerate(chains):
        tl = c.tail_hash.shape[0]
        if tl == 0 or tl > hn or tl <= best_len:
            continue
        if c.tail_model != model:
            continue
        if c.tail_end > t + _EPSILON_SECONDS:
            continue
        if c.tail_hash[tl - 1] != h[tl - 1]:
            continue
        if bool((h[:tl] == c.tail_hash).all()):
            best, best_len = idx, tl
    return best


def _max_lcp_chain(chains: list[AgentChain], h: np.ndarray) -> tuple[int | None, int]:
    """Chain tail with the deepest LCP against ``h``."""
    best_idx: int | None = None
    best_key = (0, 0)
    for idx, c in enumerate(chains):
        if c.tail_hash.shape[0] == 0:
            continue
        d = _np_lcp(c.tail_hash, h)
        if d == 0:
            continue
        key = (d, c.tail_hash.shape[0])
        if key > best_key:
            best_idx, best_key = idx, key
    return best_idx, best_key[0]


def _observed_group_prefix(result: ChainDetectionResult, members: list[int]) -> int:
    """LCP over the group members' first-request hash lists (0 if < 2)."""
    firsts = [
        np.asarray(result.chains[ci].requests[0][1].hash_ids, dtype=np.int64)
        for ci in members
    ]
    firsts = [f for f in firsts if f.shape[0] > 0]
    if len(firsts) < 2:
        return 0
    observed = firsts[0].shape[0]
    for other in firsts[1:]:
        observed = min(observed, _np_lcp(firsts[0][:observed], other))
    return observed


def compute_chain_prefix_blocks(
    result: ChainDetectionResult, *, declared_prefix_blocks: int
) -> dict[int, int]:
    """Effective setup-prefix block count per live chain."""
    live = [i for i, c in enumerate(result.chains) if c.spliced_into is None]
    if not live:
        return {}

    def _group_root(ci: int) -> int:
        while True:
            c = result.chains[ci]
            if c.fork is None or c.fork.parent_chain is None or c.fork.depth == 0:
                return ci
            ci = c.fork.parent_chain

    groups: dict[int, list[int]] = {}
    for ci in live:
        groups.setdefault(_group_root(ci), []).append(ci)

    prefixes: dict[int, int] = {}
    for members in groups.values():
        observed = _observed_group_prefix(result, members)
        for ci in members:
            if ci == result.main_index:
                prefixes[ci] = max(declared_prefix_blocks, observed)
            else:
                prefixes[ci] = observed
    return prefixes


def _last_hash_outer_idx(chain: AgentChain) -> int | None:
    """Outer index of the chain's last hash-bearing request."""
    for oi, req in reversed(chain.requests):
        if req.hash_ids:
            return oi
    return None


def _elect_continuation(
    chains: list[AgentChain],
    registered: list[int],
    t_req: _Req,
    *,
    max_gap_seconds: float,
    min_overlap_ratio: float,
) -> int | None:
    """Pick the seam continuation among forks registered on a dead tail."""
    t_end = _req_end(t_req)
    tail_blocks = len(t_req.hash_ids)

    def _seam_blocked(ci: int) -> bool:
        if tail_blocks == 0:
            return False
        gap = chains[ci].requests[0][1].t - t_end
        overlap = chains[ci].fork.depth / tail_blocks  # type: ignore[union-attr]
        return gap > max_gap_seconds and overlap < min_overlap_ratio

    candidates = [
        ci
        for ci in registered
        if chains[ci].fork is not None
        and chains[ci].fork.depth > 0
        and t_end <= chains[ci].requests[0][1].t + _EPSILON_SECONDS
        and chains[ci].requests[0][1].model == t_req.model
        and not _seam_blocked(ci)
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda ci: (
            chains[ci].fork.depth,
            -chains[ci].fork.fork_time,
            -ci,
        ),
    )


def _rekey_leftover_forks(
    *,
    chains: list[AgentChain],
    forks_by_tail: dict[int, list[int]],
    req_by_outer: dict[int, _Req],
    registered: list[int],
    elected: int,
    owner: int,
    new_tail_outer: int,
) -> bool:
    """Re-evaluate non-elected forks against the merged chain's new tail."""
    new_tail_hash = np.asarray(req_by_outer[new_tail_outer].hash_ids, np.int64)
    rekeyed = False
    for ci in registered:
        c = chains[ci]
        if ci == elected or c.spliced_into is not None or c.fork is None:
            continue
        d = _np_lcp(
            new_tail_hash,
            np.asarray(c.requests[0][1].hash_ids, dtype=np.int64),
        )
        if d <= 0:
            continue
        c.fork.fork_outer_idx = new_tail_outer
        c.fork.depth = d
        c.fork.parent_chain = owner
        forks_by_tail.setdefault(new_tail_outer, []).append(ci)
        rekeyed = True
    return rekeyed


def _resolve_seams(
    chains: list[AgentChain],
    forks_by_tail: dict[int, list[int]],
    chain_of_request: dict[int, int],
    req_by_outer: dict[int, _Req],
    *,
    max_gap_seconds: float,
    min_overlap_ratio: float,
) -> int:
    """Phase 2: splice join-seam continuations onto dead tails."""
    alias: dict[int, int] = {}

    def _resolve(i: int) -> int:
        while i in alias:
            i = alias[i]
        return i

    seams = 0
    keys = sorted(forks_by_tail)
    heapq.heapify(keys)
    processed: set[int] = set()
    while keys:
        fork_outer_idx = heapq.heappop(keys)
        if fork_outer_idx in processed:
            continue
        processed.add(fork_outer_idx)
        owner = _resolve(chain_of_request[fork_outer_idx])
        owner_chain = chains[owner]
        if _last_hash_outer_idx(owner_chain) != fork_outer_idx:
            continue
        registered = [
            ci
            for ci in forks_by_tail[fork_outer_idx]
            if chains[ci].spliced_into is None
        ]
        elected = _elect_continuation(
            chains,
            registered,
            req_by_outer[fork_outer_idx],
            max_gap_seconds=max_gap_seconds,
            min_overlap_ratio=min_overlap_ratio,
        )
        if elected is None:
            continue
        target = chains[elected]
        owner_chain.requests.extend(target.requests)
        for oi, _ in target.requests:
            chain_of_request[oi] = owner
        target.spliced_into = owner
        alias[elected] = owner
        seams += 1

        new_tail_outer = _last_hash_outer_idx(owner_chain)
        if new_tail_outer is None:
            continue
        if _rekey_leftover_forks(
            chains=chains,
            forks_by_tail=forks_by_tail,
            req_by_outer=req_by_outer,
            registered=registered,
            elected=elected,
            owner=owner,
            new_tail_outer=new_tail_outer,
        ):
            processed.discard(new_tail_outer)
            heapq.heappush(keys, new_tail_outer)
    return seams


def split_off_preamble(
    normals: list[IndexedRequest],
) -> tuple[list[IndexedRequest], list[IndexedRequest]]:
    """Pull a leading throwaway request off the front before chain detection.

    Mirrors AIPerf ``weka_trace._split_off_preamble``.
    """
    if len(normals) < 2:
        return [], normals
    ordered = sorted(normals, key=lambda item: (item[1].t, item[0]))
    outer_idx, req = ordered[0]
    if not req.hash_ids:
        return [], normals
    rest = ordered[1:]
    if any(
        _hash_list_lcp(req.hash_ids, other.hash_ids) > 0
        for _, other in rest
        if other.hash_ids
    ):
        return [], normals
    if req.output_length > _TITLE_GEN_MAX_OUTPUT_TOKENS:
        other_blocks: set[int] = set()
        for _, other in rest:
            other_blocks.update(other.hash_ids)
        if not other_blocks.isdisjoint(req.hash_ids):
            return [], normals
    return [(outer_idx, req)], sorted(rest, key=lambda item: item[0])
