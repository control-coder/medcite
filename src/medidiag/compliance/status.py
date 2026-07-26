"""合规状态取值契约。

此前 provider 各自写入自由字符串，trace 侧用 ``startswith("PASSED")`` 反推
是否命中。live provider 返回 ``PASS_WITH_DEMO_LIMITATION``（无 ``ED``），
因此每一次正常的实时复核都被标成合规命中；确定性 provider 返回
``PASSED_NON_DIAGNOSTIC_FIXTURE``，测试永远看不到这个问题。

这里把取值集合显式化：新增状态必须在本模块登记，并明确归入"命中"或"未命中"，
不能靠命名前缀碰运气。
"""

from __future__ import annotations

from enum import Enum


class ComplianceStatus(str, Enum):
    """复核阶段可写入的合规状态。"""

    PASSED_NON_DIAGNOSTIC_FIXTURE = "PASSED_NON_DIAGNOSTIC_FIXTURE"
    """确定性 fixture provider：非诊断性固定产物，未触发管控。"""

    PASS_WITH_DEMO_LIMITATION = "PASS_WITH_DEMO_LIMITATION"
    """live provider：未触发管控，但仅为受限演示复核，不是正式合规结论。"""

    BLOCKED = "BLOCKED"
    """`ComplianceGuard` 命中绝对化措辞或超范围问题，输出被拦截。"""


#: 视为合规命中的状态。命中会驱动升级并计入 trace 的 ``compliance_hit``。
COMPLIANCE_HIT_STATUSES: frozenset[str] = frozenset(
    {ComplianceStatus.BLOCKED.value}
)

#: 明确未命中的状态。二者的并集必须覆盖 ``ComplianceStatus`` 全部成员。
COMPLIANCE_CLEAR_STATUSES: frozenset[str] = frozenset(
    {
        ComplianceStatus.PASSED_NON_DIAGNOSTIC_FIXTURE.value,
        ComplianceStatus.PASS_WITH_DEMO_LIMITATION.value,
    }
)

assert COMPLIANCE_HIT_STATUSES | COMPLIANCE_CLEAR_STATUSES == {
    member.value for member in ComplianceStatus
}


def is_compliance_hit(status: object) -> bool:
    """判断一个已记录的合规状态是否算作命中。

    ``None`` 表示该事件没有复核产物，不是命中。未登记的状态按命中处理：
    宁可让一个未知状态进入人工视野，也不要因为它不在拦截集合里就被静默放行。
    """
    if status is None:
        return False
    value = str(status)
    if value in COMPLIANCE_CLEAR_STATUSES:
        return False
    return True
