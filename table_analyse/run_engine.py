import json, argparse
from engine.core import StrategyEngine, TableState, Player

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_state(path) -> TableState:
    data = load_json(path)
    players = [Player(**p) for p in data["players"]]
    return TableState(players=players, **data["tournament"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--state", required=True, help="table state JSON")
    ap.add_argument("--params", default="config/engine_params.exploit_lite.json")
    ap.add_argument("--baseline", default="config/baseline_synthetic_mtt_8max.json")
    ap.add_argument("--out", default="report.txt")
    args = ap.parse_args()

    params   = load_json(args.params)
    baseline = load_json(args.baseline)
    state    = load_state(args.state)

    engine = StrategyEngine(params, baseline)
    report = engine.generate_report(state)

    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"[OK] wrote {args.out}")

if __name__ == "__main__":
    main()