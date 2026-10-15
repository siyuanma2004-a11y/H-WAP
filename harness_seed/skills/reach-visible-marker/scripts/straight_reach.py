"""Straight right-palm reach: JSON {start, target, reach_s=1.5, hold_s=1.0} -> {"keyframes": [...]}."""
import json
import sys

args = json.load(sys.stdin)
start, target = [float(v) for v in args["start"]], [float(v) for v in args["target"]]
reach = round(float(args.get("reach_s", 1.5)) / 0.002) * 0.002
hold = round(float(args.get("hold_s", 1.0)) / 0.002) * 0.002
if len(start) != 3 or len(target) != 3 or reach <= 0 or hold < 0 or reach + hold > 3:
    raise SystemExit("need 3-D start/target and reach_s + hold_s within 3 s")
times = [0.0, reach] + ([round(reach + hold, 6)] if hold > 0 else [])
points = [start, target, target]
print(json.dumps({"keyframes": [{"t": round(t, 6), "right_wrist": {"position": p, "quaternion": None}}
                                for t, p in zip(times, points)]}))
