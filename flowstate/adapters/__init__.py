"""FlowState 的运行时适配器接口。"""

from .sglang import RuntimeCheckpointHandle, SGLangAdapter

__all__ = ["RuntimeCheckpointHandle", "SGLangAdapter"]
