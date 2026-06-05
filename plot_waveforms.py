import json
import os
import string
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# -- Configuration -------------------------------------------------------------

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
JSON_DIR = os.path.join(SCRIPT_DIR, "dataset", "jsons")
OUT_DIR = os.path.join(SCRIPT_DIR, "figures")

# Group colors and display names
GROUP_STYLE = {
    "ASD":      {"color": "#E63946", "label": "ASD"},
    "ASD+ADHD": {"color": "#F4A261", "label": "ASD+ADHD"},
    "Control":  {"color": "#457B9D", "label": "Control"},
}

# Order for plotting (Control drawn first / behind)
GROUP_ORDER = ["Control", "ASD+ADHD", "ASD"]

# Canonical flash strengths per protocol
FLASH_BINS = {
    "9_step": [12, 21, 35, 70, 113, 178, 251, 356, 446],
    "2_step": [113, 446],
    "LA3":    [85],
}

# Log cd.s.m^-2 equivalents for each flash Td.s
FLASH_LOG_CD = {
    12: -0.37, 21: -0.12, 35: 0.11, 70: 0.40, 113: 0.60,
    178: 0.80, 251: 0.95, 356: 1.11, 446: 1.20, 85: 0.48,
}

ALPHA_FILL = 0.18
LINE_WIDTH = 1.5
FIG_DPI = 300
SUBPLOT_LABEL_SIZE = 13
SUBPLOT_LABEL_WEIGHT = "bold"


def bin_flash(flash_tds, protocol):
    bins = FLASH_BINS.get(protocol, [])
    if not bins:
        return round(flash_tds)
    return min(bins, key=lambda b: abs(b - flash_tds))


def load_waveforms():
    data = defaultdict(list)
    for fname in sorted(os.listdir(JSON_DIR)):
        if not fname.endswith(".json"):
            continue
        with open(os.path.join(JSON_DIR, fname)) as fh:
            pdata = json.load(fh)

        group = pdata["demographics"]["group"]

        for rec in pdata["recordings"]:
            protocol = rec["protocol"]
            stimulus = rec.get("stimulus") or {}
            flash_tds = stimulus.get("flash_tds")
            if flash_tds is None:
                continue

            erg = rec.get("erg_waveform")
            if not erg or "time_ms" not in erg or "amplitude_uv" not in erg:
                continue

            t = np.array(erg["time_ms"], dtype=np.float64)
            a = np.array(erg["amplitude_uv"], dtype=np.float64)

            if np.any(np.isnan(t)) or np.any(np.isnan(a)):
                continue

            fb = bin_flash(flash_tds, protocol)
            data[(protocol, fb, group)].append((t, a))

    return data


def compute_mean_sd(waveforms):
    if not waveforms:
        return None, None, None, 0

    t_min = max(t[0] for t, _ in waveforms)
    t_max = min(t[-1] for t, _ in waveforms)
    if t_min >= t_max:
        return None, None, None, 0

    dt = np.median([t[1] - t[0] for t, _ in waveforms])
    common_t = np.arange(t_min, t_max, dt)

    interp_matrix = np.empty((len(waveforms), len(common_t)))
    for i, (t, a) in enumerate(waveforms):
        interp_matrix[i, :] = np.interp(common_t, t, a)

    mean = np.mean(interp_matrix, axis=0)
    sd = np.std(interp_matrix, axis=0, ddof=1)
    return common_t, mean, sd, len(waveforms)


def draw_subplot(ax, protocol, flash_bin, data, panel_letter,
                 show_xlabel=True, show_ylabel=True, show_legend=False):
    for group in GROUP_ORDER:
        key = (protocol, flash_bin, group)
        waveforms = data.get(key, [])
        if not waveforms:
            continue

        common_t, mean, sd, n = compute_mean_sd(waveforms)
        if common_t is None:
            continue

        style = GROUP_STYLE[group]
        label = f"{style['label']} (n={n})"
        ax.plot(common_t, mean, color=style["color"], linewidth=LINE_WIDTH,
                label=label, zorder=3)
        ax.fill_between(common_t, mean - sd, mean + sd,
                        color=style["color"], alpha=ALPHA_FILL, zorder=2)

    ax.axhline(0, color="grey", linewidth=0.4, linestyle="--", zorder=1)
    ax.axvline(0, color="grey", linewidth=0.4, linestyle="--", zorder=1)

    log_cd = FLASH_LOG_CD.get(flash_bin, "")
    if log_cd != "":
        title = f"{flash_bin} Td\u00b7s ({log_cd:+.2f} log cd\u00b7s\u00b7m$^{{-2}}$)"
    else:
        title = f"{flash_bin} Td\u00b7s"
    ax.set_title(title, fontsize=9, pad=4)

    if panel_letter is not None:
        ax.text(-0.02, 1.05, f"({panel_letter})", transform=ax.transAxes,
                fontsize=SUBPLOT_LABEL_SIZE, fontweight=SUBPLOT_LABEL_WEIGHT,
                verticalalignment="bottom", horizontalalignment="right")

    if show_xlabel:
        ax.set_xlabel("Time (ms)", fontsize=9)
    if show_ylabel:
        ax.set_ylabel("Amplitude (\u00b5V)", fontsize=9)

    ax.tick_params(labelsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if show_legend:
        ax.legend(loc="upper right", fontsize=7, framealpha=0.9,
                  handlelength=1.5)


def compute_ylim(protocol, data, margin=0.05):
    global_min = np.inf
    global_max = -np.inf

    for flash_bin in FLASH_BINS[protocol]:
        for group in GROUP_ORDER:
            key = (protocol, flash_bin, group)
            waveforms = data.get(key, [])
            if not waveforms:
                continue
            common_t, mean, sd, n = compute_mean_sd(waveforms)
            if common_t is None:
                continue
            lo = np.min(mean - sd)
            hi = np.max(mean + sd)
            global_min = min(global_min, lo)
            global_max = max(global_max, hi)

    span = global_max - global_min
    return global_min - margin * span, global_max + margin * span


def make_9step_figure(data, out_path):
    fig, axes = plt.subplots(3, 3, figsize=(10, 9))
    axes_flat = axes.flatten()
    letters = list(string.ascii_lowercase)
    ylim = compute_ylim("9_step", data)

    for idx, flash_bin in enumerate(FLASH_BINS["9_step"]):
        ax = axes_flat[idx]
        row, col = divmod(idx, 3)
        show_xlabel = (row == 2)
        show_ylabel = (col == 0)
        draw_subplot(ax, "9_step", flash_bin, data, letters[idx],
                     show_xlabel=show_xlabel, show_ylabel=show_ylabel,
                     show_legend=True)
        ax.set_ylim(ylim)

    fig.tight_layout(h_pad=2.5, w_pad=2.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_2step_figure(data, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    letters = list(string.ascii_lowercase)
    ylim = compute_ylim("2_step", data)

    for idx, flash_bin in enumerate(FLASH_BINS["2_step"]):
        ax = axes[idx]
        show_ylabel = (idx == 0)
        draw_subplot(ax, "2_step", flash_bin, data, letters[idx],
                     show_xlabel=True, show_ylabel=show_ylabel,
                     show_legend=True)
        ax.set_ylim(ylim)

    fig.tight_layout(w_pad=3.0)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def make_LA3_figure(data, out_path):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5))
    ylim = compute_ylim("LA3", data)

    draw_subplot(ax, "LA3", 85, data, None,
                 show_xlabel=True, show_ylabel=True, show_legend=True)
    ax.set_ylim(ylim)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path}")


def main():
    print("Loading waveforms from JSON files...")
    data = load_waveforms()

    total = sum(len(v) for v in data.values())
    combos = len(set(k[:2] for k in data.keys()))
    print(f"Loaded {total} waveforms across {combos} protocol x flash "
          f"combinations.\n")

    print("Generating ERG waveform figures...")
    make_9step_figure(data, os.path.join(OUT_DIR, "figure_9step.pdf"))
    make_2step_figure(data, os.path.join(OUT_DIR, "figure_2step.pdf"))
    make_LA3_figure(data, os.path.join(OUT_DIR, "figure_LA3.pdf"))

    print("\nDone. Figures saved to", OUT_DIR)


if __name__ == "__main__":
    main()
