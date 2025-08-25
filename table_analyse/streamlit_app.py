# streamlit_app.py
# -*- coding: utf-8 -*-
import os, sys, json, statistics
from typing import Dict, Any, List, Tuple

# 让 Python 能找到 engine 包
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
from engine.app import advise
from engine.core import Counters

st.set_page_config(page_title="策略报告（极简版 / 含PKO赏金与买入）", layout="wide")
st.title("桌面录入 → 策略报告（极简版：一页文本）")

# ========= 赛事/基础输入 =========
game_type = st.selectbox("赛事类型", ["8max MTT", "8max PKO", "6max MTT", "6max PKO"])
def parse_game(s:str):
    table = "6max" if s.startswith("6max") else "8max"
    fmt   = "pko" if "PKO" in s.upper() else "mtt"
    return table, fmt
table_type, fmt = parse_game(game_type)

col1, col2 = st.columns(2)
with col1:
    total_players     = st.number_input("报名人数 N", min_value=2, max_value=500000, value=1270, step=10)
    remaining_players = st.number_input("剩余人数 R", min_value=2, max_value=500000, value=215, step=5)
    buy_in            = st.number_input("买入 (USD)", min_value=1.0, max_value=100000.0, value=55.0, step=1.0)
with col2:
    avg_bb     = st.number_input("全场平均筹码 (BB)", min_value=1.0, max_value=10000.0, value=59.0, step=0.5)
    blind_mins = st.number_input("盲注级别时长(分钟)", min_value=1, max_value=30, value=12, step=1)

# PKO 起始赏金
starting_bounty = 0.0
if fmt == "pko":
    starting_bounty = st.number_input("PKO 起始赏金 (USD)", min_value=0.0, max_value=100000.0, value=27.0, step=1.0)

# ========= 玩家录入（Hero + 其余） =========
st.subheader("玩家录入（Hero + 对手）")
default_players = 6 if table_type == "6max" else 8
n_players = st.number_input("桌上人数", min_value=2, max_value=9, value=default_players, step=1)

def player_block(title:str, default:Dict[str,float], key:str, pko:bool) -> Dict[str,float]:
    with st.expander(title, expanded=("Hero" in title or title.endswith("2"))):
        bb   = st.number_input("筹码(BB)", 1.0, 1000.0, default["bb"], 0.5, key=f"{key}_bb")
        vpip = st.number_input("VPIP%", 0.0, 100.0, default["vpip"], 1.0, key=f"{key}_vpip")
        pfr  = st.number_input("PFR%",  0.0, 100.0, default["pfr"],  1.0, key=f"{key}_pfr")
        ats  = st.number_input("ATS%",  0.0, 100.0, default["ats"],  1.0, key=f"{key}_ats")
        tb   = st.number_input("3bet%", 0.0, 100.0, default["tb"],   0.5, key=f"{key}_tb")
        bnty = st.number_input("当前赏金(USD)", 0.0, 100000.0,
                               default.get("bounty", 0.0 if not pko else starting_bounty),
                               1.0, key=f"{key}_bnty") if pko else 0.0
        return {"bb":bb,"vpip":vpip,"pfr":pfr,"ats":ats,"threebet":tb,"bounty":bnty}

players: List[Dict[str,Any]] = []
hero_defaults = {"bb":40.0,"vpip":0.0,"pfr":0.0,"ats":0.0,"tb":0.0,"bounty":starting_bounty}
players.append({"seat":"Hero","is_hero":True, **player_block("玩家1（Hero）", hero_defaults, "hero", fmt=="pko")})

villain_defaults = {"bb":30.0,"vpip":25.0,"pfr":18.0,"ats":30.0,"tb":6.0,"bounty":starting_bounty}
for i in range(2, int(n_players)+1):
    players.append({"seat": f"Player{i}", "is_hero": False,
                    **player_block(f"玩家{i}", villain_defaults, f"p{i}", fmt=="pko")})

# ========= 统计工具 =========
def pct_to_counters(p: Dict[str,float], N:int=200) -> Counters:
    c01 = lambda x: max(0.0, min(1.0, x/100.0))
    fa = atsn = tbn = N
    return Counters(
        first_act_n=fa,
        vpip_k=int(round(c01(p.get("vpip",0))*fa)),
        pfr_k=int(round(c01(p.get("pfr",0))*fa)),
        ats_n=atsn,
        ats_k=int(round(c01(p.get("ats",0))*atsn)),
        threebet_n=tbn,
        threebet_k=int(round(c01(p.get("threebet",0))*tbn))
    )

def stage_guess(R:int, N:int)->str:
    if R <= max(9, int(0.01*N)): return "ft"
    r = R/max(1,N)
    if r <= 0.15: return "bubble"
    if r <= 0.5:  return "mid"
    return "early"

# ========= Buy-in & Bounty 影响函数 =========
def buyin_exploit_weight(buy_in: float) -> float:
    """
    买入对剥削意愿的权重（+ 更敢剥削 / - 更保守），范围约 ±0.10。
    你可按自己池子再调。
    """
    if buy_in <= 20:   return +0.10
    if buy_in <= 109:  return +0.05
    if buy_in <= 530:  return +0.00
    if buy_in <= 1050: return -0.05
    return -0.10

def bounty_stats(players: List[Dict[str,Any]], starting_bounty: float) -> Dict[str, float]:
    """
    计算赏金覆盖率统计（不含 Hero）。
    返回：avg_cov, high_cov_ratio(>=1.5), n_high, n
    """
    covs = []
    for p in players:
        if p["seat"] == "Hero":  # 仅统计对手
            continue
        if starting_bounty and starting_bounty > 0:
            cov = float(p.get("bounty", 0.0)) / float(starting_bounty)
        else:
            cov = 1.0
        covs.append(cov)
    if not covs:
        return {"avg_cov": 1.0, "high_cov_ratio": 0.0, "n_high": 0, "n": 0}
    n_high = sum(1 for x in covs if x >= 1.5)
    return {
        "avg_cov": statistics.fmean(covs),
        "high_cov_ratio": n_high / len(covs),
        "n_high": n_high,
        "n": len(covs)
    }

def bounty_bias_for_tone(avg_cov: float) -> float:
    """
    将平均覆盖率映射为开局/Bluff 的“进攻偏置”。
    覆盖率>1 说明桌上赏金更诱人 → 稍微放宽（正值）。
    """
    return max(-0.03, min(+0.08, 0.05 * (avg_cov - 1.0)))  # [-0.03, +0.08] 之间

# ========= 位置开局建议（占位模型，可替换为逐位置GTO） =========
POSITION_ORDER_8 = ["UTG","UTG1","MP","HJ","CO","BTN","SB"]
POSITION_ORDER_6 = ["UTG","MP","HJ","CO","BTN","SB"]
POSITION_OPEN_BASE = {
    "8max": {"UTG":0.14,"UTG1":0.16,"MP":0.18,"HJ":0.22,"CO":0.30,"BTN":0.46,"SB":0.36},
    "6max": {"UTG":0.20,"MP":0.23,"HJ":0.27,"CO":0.35,"BTN":0.53,"SB":0.40},
}
def pos_micros(pos:str)->float:
    m = {"UTG":-0.05,"UTG1":-0.05,"MP":-0.04,"HJ":-0.03,"CO":-0.02,"BTN":-0.02,"SB":-0.01}
    return m.get(pos, 0.0)

# ========= 汇总逻辑 =========
def style_overview_lines(players, results, pko: bool, start_bnty: float):
    lines=[]
    for p, r in zip(players, results):
        tags = ", ".join(r.get("style_tags", [])) or "（无）"
        if pko:
            cov = (p.get("bounty", 0.0)/start_bnty) if start_bnty else 1.0
            lines.append(f"- {p['seat']}\t{p['bb']:.1f}BB\t赏金:{p.get('bounty',0.0):.2f}({cov:.2f}x)\t风格：{tags}")
        else:
            lines.append(f"- {p['seat']}\t{p['bb']:.1f}BB\t风格：{tags}")
    return lines

def avg_delta(results, hero_idx=0)->Dict[str,float]:
    acc = {"vpip":0.0,"pfr":0.0,"ats":0.0,"threebet":0.0}; n=0
    for i, r in enumerate(results):
        if i==hero_idx: continue
        d = r.get("delta", {})
        for k in acc: acc[k]+=float(d.get(k,0.0))
        n+=1
    return {k:(acc[k]/max(1,n)) for k in acc}

def base_table_tone(avg_d:Dict[str,float])->Tuple[str, Dict[str,float]]:
    """返回 tone_key 和两旋钮：open_adj、bluff_adj（负=收紧/减少；正=放宽/增加）"""
    dv, dp, da, d3 = avg_d["vpip"], avg_d["pfr"], avg_d["ats"], avg_d["threebet"]
    open_adj = 0.0; bluff_adj = 0.0; key="balanced"
    if dv >= 5 and dp >= 4:
        key="loose_agg"; open_adj=-0.07; bluff_adj=-0.09
    elif dv <= -5 and dp <= -4:
        key="tight_pas"; open_adj=+0.06; bluff_adj=+0.06
    elif da >= 5:
        key="over_steal"; open_adj=-0.05; bluff_adj=-0.07
    elif d3 >= 2.5:
        key="over_3bet"; open_adj=-0.04; bluff_adj=-0.06
    elif d3 <= -2.5:
        key="under_3bet"; open_adj=+0.04; bluff_adj=+0.04
    return key, {"open_adj":open_adj, "bluff_adj":bluff_adj}

def apply_pko_bias_to_tone(tone: Dict[str,float], bias: float) -> Dict[str,float]:
    """在 PKO 下把赏金偏置叠加到两旋钮上（正值=更激进）。"""
    return {"open_adj": tone["open_adj"] + bias, "bluff_adj": tone["bluff_adj"] + bias}

def hero_open_reco(table_type:str, hero_res:Dict[str,Any], tone:Dict[str,float],
                   pko_bias_for_late: float = 0.0) -> List[str]:
    order = POSITION_ORDER_8 if table_type=="8max" else POSITION_ORDER_6
    base_map = POSITION_OPEN_BASE["8max" if table_type=="8max" else "6max"]
    lines=[]
    for pos in order:
        gto = base_map[pos]
        pos_bias = pos_micros(pos)
        # PKO：晚位（CO/BTN/SB）再给一点积极 bias，以利抢赏
        if pos in ("CO","BTN","SB"):
            pos_bias += pko_bias_for_late
        adj = tone["open_adj"] + pos_bias
        gto_final = max(0.02, gto * (1.0 + adj))
        lines.append(f"- {pos}: 基线 {gto:>.2f}  →  调整 {gto_final:>.2f}（{'收紧' if adj<0 else '放宽'}；位置微调 {pos_micros(pos):+.02f}）")
    return lines

def exploit_summary(players, results, start_bnty: float) -> List[str]:
    lines=[]
    for p, r in zip(players, results):
        if p["seat"]=="Hero":  # 对手摘要不包含 Hero
            continue
        advs = (r.get("exploit") or {}).get("advices") or []
        # 赏金覆盖率
        cov = (p.get("bounty",0.0)/start_bnty) if (start_bnty and start_bnty>0) else 1.0
        cov_flag = "（带重赏）" if cov >= 1.5 else ""
        if advs:
            lines.append(f"- vs {p['seat']}: {advs[0]}{cov_flag}")
            continue
        tags = set(r.get("style_tags") or [])
        if "over-stealer" in tags:
            lines.append(f"- vs {p['seat']}: 可少量 3bet bluff；盯注更积极防守/反击{cov_flag}")
        elif "under-3bet" in tags:
            lines.append(f"- vs {p['seat']}: 扩大开局/偷盲；更多小注率持续施压{cov_flag}")
        elif "over-3bet" in tags:
            lines.append(f"- vs {p['seat']}: 收紧边缘 open；用 4bet bluff/价值对抗{cov_flag}")
        elif "loose-aggressive" in tags:
            lines.append(f"- vs {p['seat']}: 减少轻率 cbet；偏价值下注，控制底池{cov_flag}")
        elif "nitty-passive" in tags:
            lines.append(f"- vs {p['seat']}: 增加偷盲与隔离；多做薄价值{cov_flag}")
        else:
            lines.append(f"- vs {p['seat']}: 维持基线；待更多样本后再调整{cov_flag}")
    return lines

# ========= 生成报告 =========
def run_advise_for(p:Dict[str,Any], table_type, fmt, stage, field_size, blind_minutes)->Dict[str,Any]:
    counters = pct_to_counters(p)
    return advise(table_type=table_type, fmt=fmt, hero_bb=float(p["bb"]),
                  counters=counters, stage=stage, field_size=field_size, blind_level_minutes=blind_minutes)

if st.button("生成文本报告", type="primary"):
    stage = stage_guess(int(remaining_players), int(total_players))
    results: List[Dict[str,Any]] = [run_advise_for(p, table_type, fmt, stage, int(total_players), int(blind_mins)) for p in players]

    # 顶部概览
    itm_rem = max(1, int(0.15*total_players))                 # 假设 ITM≈15%
    icm_k = 1.0 + max(0.0, (itm_rem - remaining_players) / itm_rem) * 0.6
    hero_rel = players[0]["bb"]/max(1e-6, avg_bb)
    slot = "short" if hero_rel < 0.75 else ("big" if hero_rel > 1.25 else "avg")

    # PKO 赏金影响
    bnty = bounty_stats(players, starting_bounty) if fmt=="pko" else {"avg_cov":1.0,"high_cov_ratio":0.0,"n_high":0,"n":0}
    pko_bias = bounty_bias_for_tone(bnty["avg_cov"]) if fmt=="pko" else 0.0

    # 基调 & 旋钮（叠加 PKO 偏置）
    avg_d = avg_delta(results, hero_idx=0)
    tone_key, tone_base = base_table_tone(avg_d)
    tone = apply_pko_bias_to_tone(tone_base, pko_bias) if fmt=="pko" else tone_base

    # Buy-in 与 Bounty 汇总为 exploit 权重（仅展示，供你参考）
    w_buyin = buyin_exploit_weight(buy_in)
    w_bnty  = pko_bias  # 赏金越高→越激进，这里直接展示为权重
    exploit_weight_total = w_buyin + w_bnty

    # 风格概览
    style_lines = style_overview_lines(players, results, fmt=="pko", starting_bounty)

    # Hero 全位置开局建议（PKO: 晚位加一点积极因子）
    late_push = 0.01 * bnty["high_cov_ratio"] if fmt=="pko" else 0.0  # 桌上重赏占比越高，晚位越积极
    hero_open_lines = hero_open_reco(table_type, results[0], tone, pko_bias_for_late=late_push)

    # 剥削摘要（逐人，标注“带重赏”）
    exploit_lines = exploit_summary(players, results, starting_bounty if fmt=="pko" else 0.0)

    # ===== 组装文本 =====
    txt=[]
    txt.append(f"报名/剩余：{total_players}/{remaining_players} | 平均筹码：{avg_bb:.1f}BB")
    txt.append(f"ICM因子：{icm_k:.2f}（T=ITM阈值 {itm_rem}）")
    if fmt == "pko":
        txt.append(f"PKO 起始赏金：${starting_bounty:.2f} | 平均覆盖率：{bnty['avg_cov']:.2f}x | 重赏人数：{bnty['n_high']}/{bnty['n']}")
    txt.append(f"Exploit 权重（买入/赏金）：{exploit_weight_total:+.2f}  [buy-in {w_buyin:+.2f} / bounty {w_bnty:+.2f}]")
    txt.append(f"Hero: Hero  相对平均:{hero_rel:.2f}  档位：{slot}")
    txt.append("")
    txt.append("【对手风格概览】")
    txt.append(f"分析：tag数：{len([1 for r in results if 'style_tags' in r])}")
    txt.extend(style_lines)
    txt.append("")
    txt.append("【总体策略基调】（按筹码档位与 ICM 合成，PKO 含赏金偏置；不区分位置）")
    txt.append(f"- 开局幅度基调：{tone['open_adj']:+.02f}")
    txt.append(f"- Bluff 幅度基调：{tone['bluff_adj']:+.02f}")
    txt.append("")
    txt.append("【Hero开局建议（所有位置，含身后风格/赏金修正）】")
    txt.extend(hero_open_lines)
    txt.append("")
    txt.append("【Hero 对手剥削摘要】")
    txt.extend(exploit_lines)

    report = "\n".join(txt)
    st.code(report, language="text")

    st.download_button(
        "下载 report.txt",
        data=report.encode("utf-8"),
        file_name="report.txt",
        mime="text/plain"
    )