# engine/app.py
# -*- coding: utf-8 -*-
"""
一条龙管道：
baseline(按bb插值) → 情境修正 → 统计/平滑 → Δ → 风格判定 → 策略建议(规则驱动)
"""
from __future__ import annotations

import json
import os
import argparse
from typing import Dict, Any, List

from engine.baseline_loader import load_baseline
from engine.core import (
    Counters, compute_observed, apply_context,
    smooth_against_baseline, compute_delta
)

# 路径（可通过环境变量覆盖）
BASELINE_ROOT = os.environ.get("BASELINE_ROOT", "baselines")
CONTEXT_PATH  = os.environ.get("CTX_PARAMS", "engine/engine_params_context.json")
EXPLOIT_PATH  = os.environ.get("EXP_PARAMS", "engine/engine_params_exploit.json")

TAU_CFG = {"vpip": 50.0, "pfr": 50.0, "ats": 50.0, "threebet": 30.0}

def _load_json_safe(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default

CTX_PARAMS = _load_json_safe(CONTEXT_PATH, {})
EXP_PARAMS = _load_json_safe(EXPLOIT_PATH, {"exploit_rules": {}})

def get_context_delta(stage: str | None = None,
                      field_size: int | None = None,
                      is_pko: bool | None = None,
                      blind_level_minutes: int | None = None) -> Dict[str, float]:
    """从 context 配置推导小幅百分点修正。"""
    delta = {"vpip": 0.0, "pfr": 0.0, "ats": 0.0, "threebet": 0.0}

    # 阶段
    stages = (CTX_PARAMS.get("stages") or {})
    if stage and stage in stages:
        s = stages[stage]
        for k in delta:
            if k in s:
                delta[k] += float(s[k])

    # 场次规模 → 风险溢价映射（示意）
    fs_cfg = (CTX_PARAMS.get("field_size") or {})
    if field_size is not None:
        if field_size <= 500 and "small_<=500" in fs_cfg:
            rp = float(fs_cfg["small_<=500"].get("risk_premium", 0.0))
            delta["vpip"] -= rp; delta["pfr"] -= rp
        elif field_size <= 2000 and "medium_<=2000" in fs_cfg:
            rp = float(fs_cfg["medium_<=2000"].get("risk_premium", 0.0))
            delta["vpip"] -= rp; delta["pfr"] -= rp
        elif "large_>2000" in fs_cfg:
            rp = float(fs_cfg["large_>2000"].get("risk_premium", 0.0))
            delta["vpip"] -= rp; delta["pfr"] -= rp

    # PKO
    if is_pko:
        pko_cfg = (CTX_PARAMS.get("pko") or {})
        if pko_cfg.get("enabled"):
            delta["ats"] += 0.6
            delta["threebet"] += 0.4

    # 盲注级别时长
    bl_cfg = (CTX_PARAMS.get("blind_level_minutes") or {})
    if blind_level_minutes is not None:
        if blind_level_minutes <= 8 and "<=8" in bl_cfg:
            for k in ("ats","threebet"):
                delta[k] += float(bl_cfg["<=8"].get(k, 0.0))
        elif 9 <= blind_level_minutes <= 12 and "9-12" in bl_cfg:
            for k in ("ats","threebet"):
                delta[k] += float(bl_cfg["9-12"].get(k, 0.0))
        elif blind_level_minutes >= 13 and ">=13" in bl_cfg:
            for k in ("ats","threebet"):
                delta[k] += float(bl_cfg[">=13"].get(k, 0.0))

    return delta

def judge_style(delta: Dict[str, float]) -> List[str]:
    """根据 Δ 打风格标签；可多标签。"""
    tags: List[str] = []
    dv, dp, da, d3 = delta.get("vpip",0.0), delta.get("pfr",0.0), delta.get("ats",0.0), delta.get("threebet",0.0)

    if dv >= 8 and dp >= 6: tags.append("loose-aggressive")
    elif dv >= 8 and dp < 2: tags.append("loose-passive")
    elif dv <= -8 and dp <= -6: tags.append("nitty-passive")
    elif dv <= -8 and dp > -2:  tags.append("nitty-aggressive-lite")

    if da >= 8: tags.append("over-stealer")
    elif da <= -6: tags.append("under-stealer")

    if d3 >= 3: tags.append("over-3bet")
    elif d3 <= -3: tags.append("under-3bet")

    if not tags: tags.append("balanced")
    return tags

def exploit_actions(delta: Dict[str, float], params: Dict[str, Any]) -> Dict[str, Any]:
    """基于 Δ 触发剥削规则（engine_params_exploit.json）。"""
    rules = params.get("exploit_rules") or {}
    outs: Dict[str, Any] = {"adjustments": {}, "advices": []}

    def pass_trigger(trig: Dict[str, Any]) -> bool:
        ok = True
        for k, v in (trig or {}).items():
            if k.endswith("_min"):
                key = k[:-4]; ok &= (delta.get(key, 0.0) >= float(v))
            elif k.endswith("_max"):
                key = k[:-4]; ok &= (delta.get(key, 0.0) <= float(v))
        return ok

    for name, rule in rules.items():
        if not pass_trigger(rule.get("trigger")):
            continue
        for k, v in (rule.get("adjustment") or {}).items():
            outs["adjustments"][k] = v
        if rule.get("advice"):
            outs["advices"].append(rule["advice"])

    return outs

def advise(table_type: str,
           fmt: str,
           hero_bb: float,
           counters: Counters,
           stage: str | None = None,
           field_size: int | None = None,
           blind_level_minutes: int | None = None) -> Dict[str, Any]:
    """主流程：返回 baseline / baseline_ctx / observed / estimate / delta / style / exploit。"""
    base_metrics, meta = load_baseline(table_type, fmt, hero_bb, baseline_root=BASELINE_ROOT)
    base = base_metrics.as_dict()

    ctx_delta = get_context_delta(stage=stage, field_size=field_size, is_pko=(fmt.lower()=="pko"),
                                  blind_level_minutes=blind_level_minutes)
    base_ctx = apply_context(base, ctx_delta)

    observed = compute_observed(counters)
    est = smooth_against_baseline(counters, base_ctx, TAU_CFG)
    delta = compute_delta(est["mean"], base_ctx)
    style_tags = judge_style(delta)
    exploit = exploit_actions(delta, EXP_PARAMS)

    return {
        "baseline": base,
        "baseline_ctx": base_ctx,
        "observed": observed,
        "estimate": est,
        "delta": delta,
        "style_tags": style_tags,
        "exploit": exploit,
        "meta": {
            "table": meta.table,
            "format": meta.format,
            "version": meta.version,
            "bb_used": hero_bb
        }
    }

# 作为模块或 CLI 均可
def _parse_counters_json(s: str) -> Counters:
    d = json.loads(s)
    return Counters(
        first_act_n=int(d.get("first_act_n", 0)),
        vpip_k=int(d.get("vpip_k", 0)),
        pfr_k=int(d.get("pfr_k", 0)),
        ats_n=int(d.get("ats_n", 0)),
        ats_k=int(d.get("ats_k", 0)),
        threebet_n=int(d.get("threebet_n", 0)),
        threebet_k=int(d.get("threebet_k", 0)),
    )

def main():
    ap = argparse.ArgumentParser(description="Poker preflop advisor (VPIP/PFR/ATS/3bet)")
    ap.add_argument("--table", required=True, choices=["6max", "8max"])
    ap.add_argument("--format", required=True, choices=["mtt", "pko"])
    ap.add_argument("--bb", required=True, type=float)
    ap.add_argument("--counters", required=False, default=None,
                    help='JSON: {"first_act_n":120,"vpip_k":34,"pfr_k":28,"ats_n":40,"ats_k":18,"threebet_n":36,"threebet_k":2}')
    ap.add_argument("--stage", required=False, default=None, choices=["early","mid","bubble","itm","ft"])
    ap.add_argument("--field-size", required=False, type=int, default=None)
    ap.add_argument("--blind-min", required=False, type=int, default=None)
    args = ap.parse_args()

    counters = _parse_counters_json(args.counters) if args.counters else Counters()
    out = advise(table_type=args.table, fmt=args.format, hero_bb=args.bb,
                 counters=counters, stage=args.stage, field_size=args.field_size,
                 blind_level_minutes=args.blind_min)
    print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()