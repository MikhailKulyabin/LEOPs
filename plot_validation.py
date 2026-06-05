import json
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy import stats

# -- Configuration -------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(SCRIPT_DIR, "dataset", "jsons")
OUT_DIR = os.path.join(SCRIPT_DIR, "figures")

GROUP_STYLE = {
    "ASD":      {"color": "#E63946", "marker": "s", "label": "ASD"},
    "ASD+ADHD": {"color": "#F4A261", "marker": "D", "label": "ASD+ADHD"},
    "Control":  {"color": "#457B9D", "marker": "o", "label": "Control"},
}
GROUP_ORDER = ["Control", "ASD+ADHD", "ASD"]

PROTOCOL_CONFIG = {
    "9_step": {
        "flash_strengths": [12, 21, 35, 70, 113, 178, 251, 356, 446],
        "flash_log_cd": {
            12: -0.37, 21: -0.12, 35: 0.11, 70: 0.40, 113: 0.60,
            178: 0.80, 251: 0.95, 356: 1.11, 446: 1.20,
        },
        "title": "9-step protocol",
    },
    "2_step": {
        "flash_strengths": [113, 446],
        "flash_log_cd": {113: 0.60, 446: 1.20},
        "title": "2-step protocol",
    },
    "LA3": {
        "flash_strengths": [85],
        "flash_log_cd": {85: 0.48},
        "title": "LA3 protocol",
    },
}

PROTOCOL_ORDER = ["9_step", "2_step", "LA3"]
FIG_DPI = 300


# -- Load data -----------------------------------------------------------------

def load_participants():
    participants = []
    for fname in sorted(os.listdir(JSON_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(JSON_DIR, fname)) as fh:
            participants.append(json.load(fh))
    return participants


def bin_flash(flash_tds, canonical):
    return min(canonical, key=lambda b: abs(b - flash_tds))


# -- Photopic Hill figure ------------------------------------------------------

def make_photopic_hill(participants, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax_idx, protocol in enumerate(PROTOCOL_ORDER):
        ax = axes[ax_idx]
        cfg = PROTOCOL_CONFIG[protocol]
        flashes = cfg["flash_strengths"]
        log_cd_map = cfg["flash_log_cd"]

        # Collect b-wave amplitudes per group and flash
        data = defaultdict(lambda: defaultdict(list))
        for p in participants:
            group = p["demographics"]["group"]
            for rec in p["recordings"]:
                if rec["protocol"] != protocol:
                    continue
                feat = rec["features"]
                if feat["b_amp_uv"] is None:
                    continue
                flash = bin_flash(rec["stimulus"]["flash_tds"], flashes)
                data[group][flash].append(feat["b_amp_uv"])

        log_cd_vals = [log_cd_map[f] for f in flashes]

        for group in GROUP_ORDER:
            style = GROUP_STYLE[group]
            means = []
            sems = []
            for flash in flashes:
                vals = data[group].get(flash, [])
                if vals:
                    means.append(np.mean(vals))
                    sems.append(np.std(vals, ddof=1) / np.sqrt(len(vals)))
                else:
                    means.append(np.nan)
                    sems.append(np.nan)

            means = np.array(means)
            sems = np.array(sems)
            n_per = [len(data[group].get(f, [])) for f in flashes]
            avg_n = int(np.mean([n for n in n_per if n > 0])) if any(
                n > 0 for n in n_per) else 0

            ax.errorbar(log_cd_vals, means, yerr=sems,
                        color=style["color"], marker=style["marker"],
                        markersize=6, linewidth=1.8, capsize=3, capthick=1.2,
                        label=f"{style['label']} (n\u2248{avg_n})",
                        zorder=3)

        ax.set_xlabel("Flash strength (log cd\u00b7s\u00b7m$^{-2}$)",
                       fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("b-wave amplitude (\u00b5V)", fontsize=10)
        ax.set_title(cfg["title"], fontsize=11, fontweight="bold")
        ax.legend(loc="lower right" if protocol == "9_step" else "best",
                  fontsize=7.5, framealpha=0.9)
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)

        # Td.s labels on top x-axis
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(log_cd_vals)
        ax2.set_xticklabels([str(f) for f in flashes], fontsize=8)
        if ax_idx == 1:
            ax2.set_xlabel("Flash strength (Td\u00b7s)", fontsize=10)
        else:
            ax2.tick_params(labelsize=8)

        # Panel letter
        ax.text(-0.08, 1.18, f"({chr(ord('a') + ax_idx)})",
                transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

    # Unify y-axis limits across all panels
    all_ylims = [ax.get_ylim() for ax in axes]
    ymin = min(y[0] for y in all_ylims)
    ymax = max(y[1] for y in all_ylims)
    for ax in axes:
        ax.set_ylim(ymin, ymax)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# -- Inter-ocular scatter figure -----------------------------------------------

def make_interocular_scatter(participants, out_path):
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    for ax_idx, protocol in enumerate(PROTOCOL_ORDER):
        ax = axes[ax_idx]
        cfg = PROTOCOL_CONFIG[protocol]
        flashes = cfg["flash_strengths"]

        # Collect paired R/L b-wave amplitudes per participant per flash
        eye_data = defaultdict(lambda: {"R": [], "L": []})
        for p in participants:
            pid = p["participant_id"]
            for rec in p["recordings"]:
                if rec["protocol"] != protocol:
                    continue
                feat = rec["features"]
                eye = rec.get("test_eye", "")
                if feat["b_amp_uv"] is None:
                    continue
                flash = bin_flash(rec["stimulus"]["flash_tds"], flashes)
                if eye == "RightEye":
                    eye_data[(pid, flash)]["R"].append(feat["b_amp_uv"])
                elif eye == "LeftEye":
                    eye_data[(pid, flash)]["L"].append(feat["b_amp_uv"])

        # Build paired arrays
        right_vals = []
        left_vals = []
        for key, sides in eye_data.items():
            if sides["R"] and sides["L"]:
                right_vals.append(np.mean(sides["R"]))
                left_vals.append(np.mean(sides["L"]))

        right_arr = np.array(right_vals)
        left_arr = np.array(left_vals)

        if len(right_arr) < 3:
            ax.text(0.5, 0.5,
                    f"Insufficient paired\nobservations (n={len(right_arr)})",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=10)
            ax.set_title(cfg["title"], fontsize=11, fontweight="bold")
            ax.text(-0.08, 1.08, f"({chr(ord('a') + ax_idx)})",
                    transform=ax.transAxes, fontsize=14,
                    fontweight="bold", va="top")
            continue

        # Statistics
        r_val, r_p = stats.pearsonr(right_arr, left_arr)
        icc_diff = right_arr - left_arr

        ax.scatter(right_arr, left_arr, alpha=0.35, s=18, color="#457B9D",
                   edgecolors="none", zorder=3)

        # Identity line
        lims = [min(right_arr.min(), left_arr.min()) - 2,
                max(right_arr.max(), left_arr.max()) + 2]
        ax.plot(lims, lims, "--", color="grey", linewidth=1, zorder=2)

        # Regression line
        slope, intercept = np.polyfit(right_arr, left_arr, 1)
        x_fit = np.linspace(lims[0], lims[1], 100)
        ax.plot(x_fit, slope * x_fit + intercept, color="#E63946",
                linewidth=1.5, zorder=2, label="Regression")

        ax.set_xlabel("Right eye b-wave (\u00b5V)", fontsize=10)
        if ax_idx == 0:
            ax.set_ylabel("Left eye b-wave (\u00b5V)", fontsize=10)
        ax.set_title(cfg["title"], fontsize=11, fontweight="bold")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_aspect("equal")
        ax.tick_params(labelsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # Annotate with statistics
        ax.text(0.05, 0.95,
                f"n = {len(right_arr)}\n"
                f"r = {r_val:.3f}\n"
                f"\u0394 = {np.mean(icc_diff):.2f} \u00b1 "
                f"{np.std(icc_diff, ddof=1):.2f} \u00b5V",
                transform=ax.transAxes, fontsize=8,
                verticalalignment="top",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                          edgecolor="grey", alpha=0.9))

        # Panel letter
        ax.text(-0.08, 1.08, f"({chr(ord('a') + ax_idx)})",
                transform=ax.transAxes,
                fontsize=14, fontweight="bold", va="top")

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


# -- Main ----------------------------------------------------------------------

def main():
    print("Loading participant data...")
    participants = load_participants()
    print(f"Loaded {len(participants)} participants.\n")

    print("Generating validation figures...")
    make_photopic_hill(participants,
                       os.path.join(OUT_DIR, "figure_photopic_hill.pdf"))
    make_interocular_scatter(participants,
                             os.path.join(OUT_DIR, "figure_interocular.pdf"))

    print("\nDone.")


if __name__ == "__main__":
    main()
