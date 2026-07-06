"""Cohen's Kappa 计算测试。"""

from __future__ import annotations

import pytest

from eval.kappa import cohen_kappa, kappa_verdict


def test_perfect_agreement() -> None:
    """完全一致 → kappa = 1.0。"""
    ann_a = {"s1": "yes", "s2": "no", "s3": "yes"}
    ann_b = {"s1": "yes", "s2": "no", "s3": "yes"}
    kappa, po, pe, labels, _, n = cohen_kappa(ann_a, ann_b)
    assert kappa == 1.0
    assert po == 1.0
    assert n == 3
    assert set(labels) == {"yes", "no"}


def test_no_agreement() -> None:
    """完全不一致 → kappa < 0（比随机还差）。"""
    ann_a = {"s1": "yes", "s2": "no"}
    ann_b = {"s1": "no", "s2": "yes"}
    kappa, po, pe, _, _, n = cohen_kappa(ann_a, ann_b)
    assert kappa < 0
    assert po == 0.0
    assert n == 2


def test_partial_agreement() -> None:
    """部分一致 → 0 <= po <= 1。"""
    ann_a = {"s1": "yes", "s2": "yes", "s3": "no", "s4": "no"}
    ann_b = {"s1": "yes", "s2": "no", "s3": "no", "s4": "yes"}
    kappa, po, pe, _, _, n = cohen_kappa(ann_a, ann_b)
    assert 0 <= po <= 1
    assert n == 4
    # 2/4 一致
    assert po == 0.5


def test_random_agreement_kappa_near_zero() -> None:
    """随机标注 → kappa 接近 0。"""
    # 100 个样本，两人各随机标 yes/no，kappa 应接近 0
    ann_a = {f"s{i}": "yes" if i % 2 == 0 else "no" for i in range(100)}
    ann_b = {f"s{i}": "yes" if i % 3 == 0 else "no" for i in range(100)}
    kappa, po, pe, _, _, n = cohen_kappa(ann_a, ann_b)
    assert abs(kappa) < 0.3, f"kappa {kappa} should be near 0 for random"
    assert n == 100


def test_verdict_thresholds() -> None:
    """PLAN.md 阈值判定。"""
    assert "BELOW" in kappa_verdict(0.5)
    assert "DISCUSS" in kappa_verdict(0.7)
    assert "STABLE" in kappa_verdict(0.9)
    # 边界
    assert "BELOW" in kappa_verdict(0.59)
    assert "DISCUSS" in kappa_verdict(0.60)
    assert "DISCUSS" in kappa_verdict(0.79)
    assert "STABLE" in kappa_verdict(0.80)


def test_no_common_ids_raises() -> None:
    """无共同 sample_id → ValueError。"""
    with pytest.raises(ValueError, match="no common"):
        cohen_kappa({"s1": "yes"}, {"s2": "no"})


def test_three_class_kappa() -> None:
    """三类标注（yes/no/maybe）的 kappa 计算。"""
    ann_a = {"s1": "yes", "s2": "no", "s3": "maybe", "s4": "yes"}
    ann_b = {"s1": "yes", "s2": "no", "s3": "maybe", "s4": "no"}
    kappa, po, pe, labels, _, n = cohen_kappa(ann_a, ann_b)
    assert set(labels) == {"yes", "no", "maybe"}
    assert n == 4
    # 3/4 一致
    assert po == 0.75
    assert 0 < kappa < 1.0
