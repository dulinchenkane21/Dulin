# engine/core.py
# -*- coding: utf-8 -*-
"""
核心统计逻辑：
- Counters 数据结构
- 观测频率计算
- 平滑估计 (贝叶斯先验)
- Δ (估计 - 基线)
- 基线修正工具
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Tuple
import math


# =====================
# 数据结构
# =====================

@dataclass
class Counters:
    """记录观测到的样本数量"""
    first_act_n: int = 0      # 总的入局机会数（VPIP/PFR 计算分母）
    vpip_k: int = 0           # VPIP 次数
    pfr_k: int = 0            # PFR 次数
    ats_n: int = 0            # ATS 机会数（按钮/CO/SB 等位置）
    ats_k: int = 0            # ATS 次数
    threebet_n: int = 0       # 3bet 机会数
    threebet_k: int = 0       # 3bet 次数


# =====================
# 基础计算
# =====================

def compute_observed(c: Counters) -> Dict[str, float]:
    """根据 Counters 计算直接观测频率（百分比）。"""
    def pct(k, n):
        return (100.0 * k / n) if n > 0 else 0.0

    return {
        "vpip": pct(c.vpip_k, c.first_act_n),
        "pfr": pct(c.pfr_k, c.first_act_n),
        "ats": pct(c.ats_k, c.ats_n),
        "threebet": pct(c.threebet_k, c.threebet_n),
    }


def smooth_against_baseline(c: Counters, baseline: Dict[str, float], tau_cfg: Dict[str, float]) -> Dict[str, Dict[str, Any]]:
    """
    对观测值做平滑（Beta-Binomial 思路），避免小样本波动过大。
    baseline: 基线频率
    tau_cfg: 各指标先验强度
    返回 {"mean": {...}, "ci95": {...}}
    """
    est_mean = {}
    est_ci95 = {}

    def beta_posterior_mean(k, n, p0, tau):
        # k: 成功数, n: 总次数, p0: 先验均值, tau: 先验强度
        return (k + p0 * tau) / (n + tau)

    def beta_posterior_var(k, n, p0, tau):
        alpha = k + p0 * tau + 1
        beta = (n - k) + (1 - p0) * tau + 1
        s = alpha + beta
        return (alpha * beta) / (s * s * (s + 1))

    for key in ("vpip", "pfr", "ats", "threebet"):
        n = getattr(c, f"{key}_n", c.first_act_n if key in ("vpip", "pfr") else 0)
        k = getattr(c, f"{key}_k", 0)
        p0 = baseline.get(key, 0.0) / 100.0
        tau = tau_cfg.get(key, 50.0)

        mean = beta_posterior_mean(k, n, p0, tau)
        var = beta_posterior_var(k, n, p0, tau)
        sd = math.sqrt(var)

        est_mean[key] = mean * 100
        est_ci95[key] = (mean - 1.96 * sd) * 100, (mean + 1.96 * sd) * 100

    return {"mean": est_mean, "ci95": est_ci95}


def compute_delta(est_mean: Dict[str, float], baseline: Dict[str, float]) -> Dict[str, float]:
    """计算 Δ = 估计 - 基线"""
    return {k: est_mean.get(k, 0.0) - baseline.get(k, 0.0) for k in ("vpip", "pfr", "ats", "threebet")}


def apply_context(baseline: Dict[str, float], ctx_delta: Dict[str, float]) -> Dict[str, float]:
    """在基线上应用情境修正（百分点加和）。"""
    return {k: baseline.get(k, 0.0) + ctx_delta.get(k, 0.0) for k in ("vpip", "pfr", "ats", "threebet")}