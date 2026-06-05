import json
import math
import os
from collections import defaultdict
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSONS_DIR = os.path.join(SCRIPT_DIR, "dataset", "jsons")
OUT_DIR = os.path.join(SCRIPT_DIR, "tables")


def mean_sem(values):
    """Return (mean, SEM) for a list of numbers, ignoring None."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None, None
    n = len(vals)
    m = sum(vals) / n
    if n < 2:
        return m, 0.0
    variance = sum((x - m) ** 2 for x in vals) / (n - 1)
    sem = math.sqrt(variance / n)
    return m, sem


def fmt(m, s, decimals=2):
    """Format mean +/- SEM as a string."""
    if m is None:
        return "---"
    return f"{m:.{decimals}f} +/- {s:.{decimals}f}"


def fmt_sex(males, females):
    total = males + females
    if total == 0:
        return "---"
    m_pct = round(100 * males / total)
    f_pct = 100 - m_pct
    return f"{m_pct}:{f_pct}"


def main():
    # Load all participant JSONs
    all_recordings = []
    for fname in sorted(os.listdir(JSONS_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(JSONS_DIR, fname)) as f:
            data = json.load(f)

        demo = data["demographics"]
        for rec in data["recordings"]:
            all_recordings.append({
                "protocol": rec["protocol"],
                "group": demo["group"],
                "age": rec["age"],
                "sex": demo["sex"],  # 0=male, 1=female
                "iris": rec["iris"],
                "a_time": rec["features"]["a_time_ms"],
                "a_amp": rec["features"]["a_amp_uv"],
                "b_time": rec["features"]["b_time_ms"],
                "b_amp": rec["features"]["b_amp_uv"],
                "op_sum_time": rec["features"]["op_sum_time_ms"],
                "op_sum_amp": rec["features"]["op_sum_amp_uv"],
            })

    # Define protocol sets
    protocol_sets = {
        "9-step + 2-step": ["9_step", "2_step"],
        "LA3 ISCEV Standard": ["LA3"],
    }

    group_display = {
        "ASD": "ASD",
        "ASD+ADHD": "ASD+ADHD",
        "Control": "TD",
    }
    group_order = ["ASD", "ASD+ADHD", "Control"]

    rows = []
    for pset_name, pset_protocols in protocol_sets.items():
        for group in group_order:
            recs = [
                r for r in all_recordings
                if r["protocol"] in pset_protocols and r["group"] == group
            ]
            n = len(recs)
            if n == 0:
                continue

            age_m, age_s = mean_sem([r["age"] for r in recs])
            iris_m, iris_s = mean_sem([r["iris"] for r in recs])
            at_m, at_s = mean_sem([r["a_time"] for r in recs])
            aa_m, aa_s = mean_sem([r["a_amp"] for r in recs])
            bt_m, bt_s = mean_sem([r["b_time"] for r in recs])
            ba_m, ba_s = mean_sem([r["b_amp"] for r in recs])
            ot_m, ot_s = mean_sem([r["op_sum_time"] for r in recs])
            oa_m, oa_s = mean_sem([r["op_sum_amp"] for r in recs])

            males = sum(1 for r in recs if r["sex"] == 0)
            females = sum(1 for r in recs if r["sex"] == 1)

            rows.append({
                "Protocol set": pset_name,
                "Group": group_display[group],
                "N": n,
                "Age (yr)": fmt(age_m, age_s, 1),
                "Sex M:F %": fmt_sex(males, females),
                "Iris": fmt(iris_m, iris_s, 3),
                "a-time (ms)": fmt(at_m, at_s, 2),
                "a-amp (uV)": fmt(aa_m, aa_s, 2),
                "b-time (ms)": fmt(bt_m, bt_s, 2),
                "b-amp (uV)": fmt(ba_m, ba_s, 2),
                "OPs-time (ms)": fmt(ot_m, ot_s, 2),
                "OPs-amp (uV)": fmt(oa_m, oa_s, 2),
            })

    df = pd.DataFrame(rows)
    df = df.set_index(["Protocol set", "Group"])

    print(df.to_string())

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "statistics.csv")
    df.to_csv(csv_path)
    print(f"\nTable saved to {csv_path}")


if __name__ == "__main__":
    main()
