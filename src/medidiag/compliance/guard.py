"""合规管控模块占位。

阶段 5 实现 ComplianceGuard：
- 超范围医疗问题拒答
- 绝对化诊断措辞拦截（"确诊""保证治愈""无需就医"等）
- 强制风险提示插入（"仅供学习和工程演示，不构成医疗建议"）
- 不当输出过滤
- 合规命中必须写入 trace / event log

注意：这是技术层面合规管控，不是真正医疗合规认证。
"""

from __future__ import annotations


class ComplianceGuard:
    """合规管控器骨架。

    TODO[阶段5]: 实现超范围拒答、绝对化拦截、风险提示强制插入。
    """

    def __init__(self) -> None:
        raise NotImplementedError("ComplianceGuard 将在阶段 5 实现")
