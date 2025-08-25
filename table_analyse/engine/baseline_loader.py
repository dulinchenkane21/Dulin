# -*- coding: utf-8 -*-
"""
baseline_loader.py (ATS version)
--------------------------------
加载“按筹码深度”的基线（VPIP/PFR/ATS/3bet），提供：
- baselines/index.json 别名映射（可选）
- 线性插值（非离散bb）
- 极短筹吸附（≤6bb → 最小桶，一般为5bb）
- 约束校验：边界、单调（bb变浅→频率非增）、pfr<=vpip
- 兼容旧RFI基线：若无 ats 但有 rfi，将按锚点自动转换为 ats

目录建议：
/baselines/
  6max_mtt.json
  6max_pko.json
  8max_mtt.json
  8max_pko.json
  index.json           # 可选：别名映射
"""

from __future__ import annotations

import json
import os
import bisect
from dataclasses import dataclass
from typing import Dict, Tuple, Any, Optional, List


# ---------------------------
# 数据结构
# ---------------------------

Number = float

@dataclass(frozen=True)
class Metrics:
    vpip: Number
    pfr: Number
    ats: Number
    threebet: Number

    def as_dict(self) -> Dict[str, Number]:
        return {"vpip": self.vpip, "pfr": self.pfr, "ats": self.ats, "threebet": self.threebet}


@dataclass(frozen=True)
class Meta:
    table: str
    format: str
    version: str
    depth_buckets: List[str]
    precision: int = 1
    constraints: Optional[Dict[str, Any]] = None
    interpolation: Optional[Dict[str, Any]] = None
    notes: Optional[str] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "Meta":
        return Meta(
            table=d.get("table", ""),
            format=d.get("format", ""),
            version=d.get("version", ""),
            depth_buckets=d.get("depth_buckets") or [],
            precision=int(d.get("precision", 1)),
            constraints=d.get("constraints"),
            interpolation=d.get("interpolation"),
            notes=d.get("notes"),
        )


@dataclass(frozen=True)
class BaselineDoc:
    meta: Meta
    buckets: Dict[str, Metrics]  # key="100bb" value=Metrics


# ---------------------------
# 工具函数
# ---------------------------

def _parse_bucket_key(key: str) -> int:
    key = key.strip().lower().replace(" ", "")
    if not key.endswith("bb"):
        raise ValueError(f"非法桶名: {key}")
    return int(key[:-2])

def _sorted_bucket_keys(bucket_keys: List[str]) -> List[str]:
    return sorted(bucket_keys, key=lambda k: _parse_bucket_key(k))

def _round_metrics(m: Metrics, precision: int) -> Metrics:
    r = round
    return Metrics(
        vpip=r(m.vpip, precision),
        pfr=r(m.pfr, precision),
        ats=r(m.ats, precision),
        threebet=r(m.threebet, precision),
    )

def _lerp(a: Number, b: Number, t: Number) -> Number:
    return (1.0 - t) * a + t * b

def _lerp_metrics(a: Metrics, b: Metrics, t: Number, precision: int) -> Metrics:
    return _round_metrics(
        Metrics(
            vpip=_lerp(a.vpip, b.vpip, t),
            pfr=_lerp(a.pfr, b.pfr, t),
            ats=_lerp(a.ats, b.ats, t),
            threebet=_lerp(a.threebet, b.threebet, t),
        ),
        precision=precision,
    )


# ---------------------------
# 约束校验
# ---------------------------

def _validate_constraints(doc: BaselineDoc) -> None:
    """按 meta.constraints 校验；若失败抛 AssertionError。"""
    constraints = doc.meta.constraints or {}
    bounds = constraints.get("bounds_pct", [0, 100])
    lo, hi = float(bounds[0]), float(bounds[1])

    # 边界 + 关系：仅强制 pfr<=vpip（ATS 不与 VPIP 比较，因为分母不同）
    for k, m in doc.buckets.items():
        for name, val in m.as_dict().items():
            assert lo <= val <= hi, f"[{k}] {name}={val} 越界 {bounds}"
        assert m.pfr <= m.vpip + 1e-9, f"[{k}] 违反关系: pfr({m.pfr}) <= vpip({m.vpip})"

    # 单调性（bb 变小→数值非增）
    mono = constraints.get("monotonic_vs_depth", {})
    keys_sorted = _sorted_bucket_keys(list(doc.buckets.keys()))  # small→large
    def check_nonincreasing(metric_name: str):
        last = None
        for k in reversed(keys_sorted):  # 从大bb向小bb检查
            cur = getattr(doc.buckets[k], metric_name)
            if last is not None:
                assert cur <= last + 1e-9, f"单调性失败: {metric_name} 在 {k} 高于更大bb的值"
            last = cur

    for metric, rule in mono.items():
        if rule == "nonincreasing":
            check_nonincreasing(metric)


# ---------------------------
# 文件加载与缓存
# ---------------------------

class _Cache:
    def __init__(self):
        self.docs: Dict[str, BaselineDoc] = {}
        self.index: Optional[Dict[str, Any]] = None

_CACHE = _Cache()

def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _load_index(baseline_root: str) -> Dict[str, Any]:
    if _CACHE.index is not None:
        return _CACHE.index
    path = os.path.join(baseline_root, "index.json")
    if not os.path.exists(path):
        _CACHE.index = {"alias": {}}
    else:
        _CACHE.index = _load_json(path)
    return _CACHE.index

def _resolve_file_from_alias(table_type: str, fmt: str, baseline_root: str) -> str:
    idx = _load_index(baseline_root)
    alias = idx.get("alias", {})
    key = f"default_{table_type}_{fmt}"
    filename = alias.get(key)
    if filename:
        p = os.path.join(baseline_root, filename)
        if os.path.exists(p):
            return p
    # 回退约定命名
    fallback = os.path.join(baseline_root, f"{table_type}_{fmt}.json")
    if not os.path.exists(fallback):
        raise FileNotFoundError(
            f"未找到基线文件：别名 '{key}' 与回退 '{table_type}_{fmt}.json' 均失败。"
        )
    return fallback

# ---- ATS 兼容：若文件没有 ats 但有 rfi，按锚点把 rfi→ats ----
_ATS_ANCHORS = {
    ("6max", "mtt"): 42.0,   # 100bb 目标ATS锚点（CO/BTN/SB未入池开局率）
    ("6max", "pko"): 43.5,
    ("8max", "mtt"): 38.0,
    ("8max", "pko"): 39.5,
}

def _ensure_ats_in_bucket_dict(meta: Meta, raw_bucket: Dict[str, Any]) -> Dict[str, Any]:
    """
    输入一个原始桶字典（可能有 rfi/也可能已有 ats），输出确保含有 ats 的字典。
    若存在 rfi 且不存在 ats，则用锚点缩放生成 ats，并移除 rfi。
    """
    d = dict(raw_bucket)  # copy
    if "ats" in d:
        # 已有 ats，确保是 float
        d["ats"] = float(d["ats"])
        d.pop("rfi", None)
        return d

    if "ats" not in d and "rfi" in d:
        key = (meta.table.strip().lower(), meta.format.strip().lower())
        key = (key[0], key[1])
        # 读取 100bb 的 rfi 需要在外层处理；这里用一个临时标记，稍后统一缩放
        d["_rfi_tmp"] = float(d.pop("rfi"))
        return d

    # 既没有 ats 也没有 rfi，则默认 ats=0
    d.setdefault("ats", 0.0)
    return d

def _finalize_ats_scale(meta: Meta, buckets_raw: Dict[str, Dict[str, Any]]) -> None:
    """
    若存在 _rfi_tmp 字段，按锚点将其整体缩放为 ats，并删除临时字段。
    以 100bb 的 _rfi_tmp 作为基准比值。
    """
    # 找到 100bb rfi
    rfi_100 = None
    if "100bb" in buckets_raw and "_rfi_tmp" in buckets_raw["100bb"]:
        rfi_100 = float(buckets_raw["100bb"]["_rfi_tmp"])

    if rfi_100 and rfi_100 > 0:
        key = (meta.table.strip().lower(), meta.format.strip().lower())
        anchor = _ATS_ANCHORS.get((key[0], key[1]))
        if anchor is None:
            # 无锚点配置，则直接把 rfi 当做 ats 使用
            for k, d in buckets_raw.items():
                if "_rfi_tmp" in d:
                    d["ats"] = float(d["_rfi_tmp"])
                    d.pop("_rfi_tmp", None)
        else:
            scale = float(anchor) / float(rfi_100)
            for k, d in buckets_raw.items():
                if "_rfi_tmp" in d:
                    d["ats"] = round(float(d["_rfi_tmp"]) * scale, max(0, int(meta.precision or 1)))
                    d.pop("_rfi_tmp", None)
    else:
        # 没有 rfi 锚点：若有 _rfi_tmp 但缺100bb，退化为直接赋值
        for k, d in buckets_raw.items():
            if "_rfi_tmp" in d:
                d["ats"] = float(d["_rfi_tmp"])
                d.pop("_rfi_tmp", None)

def _load_baseline_doc(path: str) -> BaselineDoc:
    cached = _CACHE.docs.get(path)
    if cached is not None:
        return cached

    data = _load_json(path)
    if "meta" not in data or "gto_baseline" not in data:
        raise ValueError(f"文件格式错误：{path}，缺少 'meta' 或 'gto_baseline'")

    meta = Meta.from_dict(data["meta"])
    gto = data["gto_baseline"]

    # 先把每个桶的原始字典规整为含 ats（或待转换的 _rfi_tmp）
    tmp_buckets: Dict[str, Dict[str, Any]] = {}
    for k, v in gto.items():
        d = {
            "vpip": float(v["vpip"]),
            "pfr": float(v["pfr"]),
            "threebet": float(v["threebet"]),
        }
        # 处理 ats/rfi
        d = _ensure_ats_in_bucket_dict(meta, {**d, **v})
        tmp_buckets[k] = d

    # 若存在 rfi 临时值，按锚点统一缩放
    _finalize_ats_scale(meta, tmp_buckets)

    # 构造 Metrics
    buckets: Dict[str, Metrics] = {}
    for k, d in tmp_buckets.items():
        buckets[k] = Metrics(
            vpip=float(d["vpip"]),
            pfr=float(d["pfr"]),
            ats=float(d.get("ats", 0.0)),
            threebet=float(d["threebet"]),
        )

    doc = BaselineDoc(meta=meta, buckets=buckets)

    # 约束校验（若 meta.constraints 存在）
    if meta.constraints:
        _validate_constraints(doc)

    _CACHE.docs[path] = doc
    return doc


# ---------------------------
# 插值/映射
# ---------------------------

def bucketize(bb: Number, bucket_keys: List[str], snap_min_threshold: float = 6.0) -> Tuple[str, str, float]:
    """
    把实数bb映射到相邻两个桶并返回插值权重 t：
    - 若 bb <= snap_min_threshold（默认6）→ 直接吸附到最小桶（如 '5bb'），返回(lo=hi=min_bucket, t=0)
    - 否则，在相邻桶之间做线性插值：返回 (lo_bucket, hi_bucket, t ∈ [0,1])
    """
    keys_sorted = _sorted_bucket_keys(bucket_keys)  # small→large
    xs = [_parse_bucket_key(k) for k in keys_sorted]

    if bb <= snap_min_threshold:
        min_key = keys_sorted[0]
        return min_key, min_key, 0.0

    if bb <= xs[0]:
        return keys_sorted[0], keys_sorted[0], 0.0
    if bb >= xs[-1]:
        return keys_sorted[-1], keys_sorted[-1], 0.0

    i = bisect.bisect_left(xs, bb)
    lo_v, hi_v = xs[i - 1], xs[i]
    lo_k, hi_k = keys_sorted[i - 1], keys_sorted[i]
    t = (bb - lo_v) / float(hi_v - lo_v)
    return lo_k, hi_k, float(t)


def load_baseline(table_type: str, fmt: str, bb: Number, baseline_root: str = "baselines") -> Tuple[Metrics, Meta]:
    """
    载入对应的基线，并按bb做插值，返回最终可用的四项指标。
    用法：
        metrics, meta = load_baseline("6max", "mtt", 37)
        print(metrics.as_dict())  # {"vpip":..., "pfr":..., "ats":..., "threebet":...}
    """
    path = _resolve_file_from_alias(table_type.strip().lower(), fmt.strip().lower(), baseline_root)
    doc = _load_baseline_doc(path)

    # 插值配置
    snap_min = 6.0
    interp_cfg = doc.meta.interpolation or {}
    snap_rules = interp_cfg.get("snap_rules") or {}
    if "snap_to_min_when" in snap_rules:
        rule = str(snap_rules["snap_to_min_when"]).strip()
        if rule.startswith("<="):
            try:
                snap_min = float(rule[2:])
            except ValueError:
                pass

    # 桶列表
    if doc.meta.depth_buckets:
        depth_buckets = doc.meta.depth_buckets
    else:
        depth_buckets = _sorted_bucket_keys(list(doc.buckets.keys()))

    # 完整性检查
    for k in depth_buckets:
        if k not in doc.buckets:
            raise KeyError(f"缺少桶 '{k}' 的数值，请检查 {path}")

    # 插值
    lo_key, hi_key, t = bucketize(bb=float(bb), bucket_keys=depth_buckets, snap_min_threshold=snap_min)
    lo_m = doc.buckets[lo_key]
    hi_m = doc.buckets[hi_key]
    precision = max(0, int(doc.meta.precision or 1))

    if lo_key == hi_key:
        metrics = _round_metrics(lo_m, precision)
    else:
        metrics = _lerp_metrics(lo_m, hi_m, t, precision)

    return metrics, doc.meta


# ---------------------------
# CLI
# ---------------------------

def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Baseline Loader (VPIP/PFR/ATS/3bet by stack depth)")
    parser.add_argument("--table", required=True, choices=["6max", "8max"], help="table type")
    parser.add_argument("--format", required=True, choices=["mtt", "pko"], help="game format")
    parser.add_argument("--bb", required=True, type=float, help="hero (or target) stack in BB")
    parser.add_argument("--root", default="baselines", help="baseline root folder (default: ./baselines)")
    args = parser.parse_args()

    metrics, meta = load_baseline(args.table, args.format, args.bb, baseline_root=args.root)
    print(json.dumps({
        "input": {"table": args.table, "format": args.format, "bb": args.bb},
        "meta": {
            "table": meta.table,
            "format": meta.format,
            "version": meta.version,
            "precision": meta.precision,
            "notes": meta.notes
        },
        "metrics": metrics.as_dict()
    }, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    _main()