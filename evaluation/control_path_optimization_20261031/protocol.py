"""无Runtime依赖的单请求证据门禁（四 Pending 版：按 rid 分派期望输入）。"""
def require(value, message):
    if not value:
        raise ValueError(message)


def admission_gate(baseline, current, received, expected):
    """Admission 时刻的必要在线检查（测量模式下仅 Token 一致性；全量比对移至请求后）。"""
    require(received == expected, 'Admission实际Token输入不一致')
