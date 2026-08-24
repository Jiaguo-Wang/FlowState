"""基于 WP2 剖面数据估计恢复成本。"""

from bisect import bisect_left
import csv
import json
from pathlib import Path
from typing import Optional, Tuple, Union


_ARTIFACT_DIRECTORY = (
    Path(__file__).resolve().parents[1]
    / "motivation"
    / "artifacts"
    / "replay_cost_20260819"
)
_DEFAULT_PROFILE_PATH = _ARTIFACT_DIRECTORY / "replay_cost.csv"
_DEFAULT_FIT_PATH = _ARTIFACT_DIRECTORY / "fit_metrics.json"


class RecoveryCostModel:
    """使用实测剖面点和既有线性拟合估计额外恢复延迟。"""

    def __init__(
        self,
        profile_path: Optional[Union[str, Path]] = None,
        fit_metrics_path: Optional[Union[str, Path]] = None,
    ) -> None:
        profile = Path(profile_path) if profile_path is not None else _DEFAULT_PROFILE_PATH
        fit_metrics = (
            Path(fit_metrics_path)
            if fit_metrics_path is not None
            else _DEFAULT_FIT_PATH
        )

        self._profile_points = self._load_profile(profile)
        self._tokens = tuple(point[0] for point in self._profile_points)
        self._fit_intercept_ms, self._fit_slope_ms_per_token = self._load_fit(
            fit_metrics
        )

    @staticmethod
    def _load_profile(path: Path) -> Tuple[Tuple[int, float], ...]:
        """读取总延迟中位数，并转换为相对零回放基线的恢复成本。"""
        required_fields = {
            "expected_replay_tokens",
            "recovery_latency_median_ms",
        }
        total_latency_points = []

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            missing_fields = required_fields.difference(reader.fieldnames or ())
            if missing_fields:
                missing = ", ".join(sorted(missing_fields))
                raise ValueError(f"剖面文件缺少字段：{missing}")

            for row in reader:
                try:
                    replay_tokens = int(row["expected_replay_tokens"])
                    total_latency_ms = float(row["recovery_latency_median_ms"])
                except (TypeError, ValueError) as error:
                    raise ValueError(f"剖面数据格式无效：{row}") from error

                if replay_tokens < 0:
                    raise ValueError(
                        f"剖面中的 replay token 必须大于等于零：{replay_tokens}"
                    )
                total_latency_points.append((replay_tokens, total_latency_ms))

        if not total_latency_points:
            raise ValueError("剖面文件中没有可用数据点")

        total_latency_points.sort(key=lambda point: point[0])
        for previous, current in zip(
            total_latency_points, total_latency_points[1:]
        ):
            if previous[0] == current[0]:
                raise ValueError(f"剖面包含重复 replay token：{current[0]}")

        baseline_points = [point for point in total_latency_points if point[0] == 0]
        if not baseline_points:
            raise ValueError("剖面缺少零回放基线")
        baseline_ms = baseline_points[0][1]

        recovery_points = tuple(
            (replay_tokens, 0.0 if replay_tokens == 0 else total_ms - baseline_ms)
            for replay_tokens, total_ms in total_latency_points
        )
        RecoveryCostModel._validate_monotonic(recovery_points)
        return recovery_points

    @staticmethod
    def _validate_monotonic(points: Tuple[Tuple[int, float], ...]) -> None:
        """检查相邻恢复成本是否单调非减。"""
        for previous, current in zip(points, points[1:]):
            if current[1] < previous[1]:
                raise ValueError(
                    "恢复成本剖面不满足单调非减："
                    f"{previous[0]} token 为 {previous[1]:.6f} ms，"
                    f"{current[0]} token 为 {current[1]:.6f} ms"
                )

    @staticmethod
    def _load_fit(path: Path) -> Tuple[float, float]:
        """读取 WP2 已有的原始运行线性拟合参数。"""
        with path.open("r", encoding="utf-8") as handle:
            metrics = json.load(handle)

        raw_run_fit = metrics.get("raw_run_ols")
        if not isinstance(raw_run_fit, dict):
            raise ValueError("拟合指标缺少 raw_run_ols")

        try:
            intercept_ms = float(raw_run_fit["intercept_ms"])
            slope_ms_per_token = float(raw_run_fit["slope_ms_per_token"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("拟合指标缺少有效的截距或斜率") from error

        if slope_ms_per_token < 0:
            raise ValueError(
                f"拟合斜率必须大于等于零：{slope_ms_per_token}"
            )
        return intercept_ms, slope_ms_per_token

    def estimate(self, replay_tokens: int) -> float:
        """返回 replay_tokens 对应的额外恢复延迟，单位为毫秒。"""
        if replay_tokens < 0:
            raise ValueError("replay_tokens 必须大于等于零")
        if replay_tokens == 0:
            return 0.0

        maximum_profile_tokens = self._tokens[-1]
        if replay_tokens > maximum_profile_tokens:
            fitted_total_ms = (
                self._fit_intercept_ms
                + self._fit_slope_ms_per_token * replay_tokens
            )
            fitted_baseline_ms = self._fit_intercept_ms
            recovery_cost_ms = fitted_total_ms - fitted_baseline_ms
            if recovery_cost_ms < 0:
                raise ValueError(
                    f"线性外推得到负恢复成本：{recovery_cost_ms} ms"
                )
            return recovery_cost_ms

        upper_index = bisect_left(self._tokens, replay_tokens)
        upper_tokens, upper_cost_ms = self._profile_points[upper_index]
        if upper_tokens == replay_tokens:
            return upper_cost_ms

        lower_tokens, lower_cost_ms = self._profile_points[upper_index - 1]
        position = (replay_tokens - lower_tokens) / (upper_tokens - lower_tokens)
        return lower_cost_ms + position * (upper_cost_ms - lower_cost_ms)
