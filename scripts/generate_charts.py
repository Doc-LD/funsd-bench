#!/usr/bin/env python3
"""
Generate brand-consistent visualizations for DocLD FUNSD benchmark results.
Uses DocLD brand palette (Rose, Violet, Amber, Emerald, Mauve) and Inter font.
"""

import json
import os
import subprocess
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
import matplotlib.font_manager as fm
import seaborn as sns
import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Brand tokens
# ---------------------------------------------------------------------------
# Feature accents (from UsageSparkline.tsx / globals.css)
ROSE    = "#e11d48"
VIOLET  = "#7c3aed"
AMBER   = "#d97706"
EMERALD = "#059669"
PINK    = "#ec4899"

# Mauve neutrals (oklch → approx hex)
MAUVE_50  = "#faf9fb"
MAUVE_100 = "#f5f3f7"
MAUVE_200 = "#ebe7ef"
MAUVE_300 = "#ddd6e3"
MAUVE_400 = "#b4acbd"
MAUVE_500 = "#877e93"
MAUVE_600 = "#6b6378"
MAUVE_700 = "#585063"
MAUVE_800 = "#3e3849"
MAUVE_900 = "#312c3b"
MAUVE_950 = "#1f1b25"

SLATE_950 = "#020617"
SLATE_700 = "#334155"
SLATE_500 = "#64748b"
SLATE_400 = "#94a3b8"
SLATE_300 = "#cbd5e1"
SLATE_200 = "#e2e8f0"
SLATE_100 = "#f1f5f9"

# Ordered accent palette for multi-series charts
ACCENT_PALETTE = [ROSE, VIOLET, AMBER, EMERALD]

# ---------------------------------------------------------------------------
# Font setup – use Inter if available, else fallback to system sans
# ---------------------------------------------------------------------------
def _setup_fonts():
    """Register Inter if installed and configure matplotlib defaults."""
    inter_paths = [
        p for p in fm.findSystemFonts()
        if "Inter" in os.path.basename(p) and p.endswith((".ttf", ".otf"))
    ]
    for p in inter_paths:
        try:
            fm.fontManager.addfont(p)
        except Exception:
            pass

    inter_available = any(
        f.name == "Inter" for f in fm.fontManager.ttflist
    )
    font_family = "Inter" if inter_available else "sans-serif"

    plt.rcParams.update({
        "font.family": font_family,
        "font.size": 11,
        "axes.titlesize": 14,
        "axes.titleweight": "bold",
        "axes.labelsize": 12,
        "axes.labelcolor": SLATE_700,
        "axes.edgecolor": MAUVE_200,
        "axes.linewidth": 0.8,
        "axes.facecolor": "#ffffff",
        "figure.facecolor": "#ffffff",
        "xtick.color": SLATE_500,
        "ytick.color": SLATE_500,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "legend.framealpha": 0.9,
        "legend.edgecolor": MAUVE_200,
        "grid.color": MAUVE_200,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.6,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "savefig.facecolor": "#ffffff",
    })

_setup_fonts()

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _brand_grid(ax, axis="y"):
    ax.set_axisbelow(True)
    if axis in ("y", "both"):
        ax.yaxis.grid(True, linewidth=0.4, color=MAUVE_200, alpha=0.7)
    if axis in ("x", "both"):
        ax.xaxis.grid(True, linewidth=0.4, color=MAUVE_200, alpha=0.7)
    ax.xaxis.grid(False) if axis == "y" else None
    ax.yaxis.grid(False) if axis == "x" else None
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)


def _watermark(fig, text="DocLD"):
    fig.text(
        0.99, 0.01, text,
        fontsize=9, color=MAUVE_400,
        ha="right", va="bottom", alpha=0.5,
        fontweight="bold",
    )


def load_results():
    script_dir = Path(__file__).parent
    results_dir = script_dir.parent / "results"
    with open(results_dir / "results.json") as f:
        results = json.load(f)
    with open(results_dir / "analysis.json") as f:
        analysis = json.load(f)
    return results, analysis


def create_output_dirs():
    script_dir = Path(__file__).parent
    charts_dir = script_dir.parent / "results" / "charts"
    public_dir = script_dir.parent.parent / "public" / "blog" / "images" / "docld-funsd"
    charts_dir.mkdir(parents=True, exist_ok=True)
    public_dir.mkdir(parents=True, exist_ok=True)
    return charts_dir, public_dir


def save_chart(fig, name, charts_dir, public_dir):
    fig.savefig(charts_dir / f"{name}.png")
    fig.savefig(public_dir / f"{name}.png")
    plt.close(fig)
    print(f"  ✓ {name}.png")


# ===================================================================
# Charts
# ===================================================================

def create_hero_banner(results, charts_dir, public_dir):
    docs = results["documents"]
    wm = results.get("word_match_accuracy", {})
    mean_acc = wm.get("mean", 0)
    median_acc = wm.get("median", 0)
    if not median_acc and docs:
        vals = [d.get("word_match_accuracy", 0) for d in docs]
        mean_acc = np.mean(vals) * 100
        median_acc = np.median(vals) * 100
    success_rate = results["summary"]["success_rate"]

    fig, ax = plt.subplots(figsize=(14, 5.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Subtle gradient bar across bottom
    gradient = np.linspace(0, 1, 256).reshape(1, -1)
    ax.imshow(
        gradient, aspect="auto", cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "brand", [MAUVE_100, VIOLET + "18"]
        ),
        extent=[0, 1, 0, 0.08], transform=ax.transAxes, zorder=0,
    )

    # Brand name
    ax.text(0.5, 0.94, "DocLD", fontsize=13, fontweight="bold",
            ha="center", va="top", color=MAUVE_500, transform=ax.transAxes)

    # Headline number
    ax.text(0.5, 0.68, f"{median_acc:.1f}%", fontsize=68, fontweight="bold",
            ha="center", va="center", color=VIOLET, transform=ax.transAxes)
    ax.text(0.5, 0.44, "Median Word-Match OCR Accuracy on FUNSD", fontsize=16,
            ha="center", va="center", color=SLATE_500, transform=ax.transAxes)

    # Stat pills
    stats = [
        (f"{success_rate:.0f}%", "Parse Success Rate", EMERALD),
        (f"{mean_acc:.1f}%", "Mean Accuracy", ROSE),
        ("50", "Test Documents", AMBER),
    ]
    pill_y = 0.17
    for i, (val, label, color) in enumerate(stats):
        x = 0.18 + i * 0.32
        ax.text(x, pill_y + 0.05, val, fontsize=22, fontweight="bold",
                ha="center", color=color, transform=ax.transAxes)
        ax.text(x, pill_y - 0.05, label, fontsize=10, ha="center",
                color=SLATE_400, transform=ax.transAxes)

    _watermark(fig)
    save_chart(fig, "hero-ocr-accuracy", charts_dir, public_dir)


def create_cer_histogram(results, charts_dir, public_dir):
    docs = results["documents"]
    vals = [d["cer"] * 100 for d in docs if d["cer"] < 2]
    mean_v, med_v = np.mean(vals), np.median(vals)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(vals, bins=20, color=ROSE, edgecolor="#ffffff", linewidth=0.8, alpha=0.85)
    ax.axvline(med_v, color=VIOLET, ls="--", lw=2, label=f"Median  {med_v:.1f}%")
    ax.axvline(mean_v, color=AMBER, ls=":", lw=2, label=f"Mean  {mean_v:.1f}%")
    ax.set_xlabel("Character Error Rate (%)")
    ax.set_ylabel("Documents")
    ax.set_title("CER Distribution — DocLD on FUNSD")
    ax.legend(frameon=True)
    _brand_grid(ax)
    _watermark(fig)
    save_chart(fig, "cer-histogram", charts_dir, public_dir)


def create_wer_histogram(results, charts_dir, public_dir):
    docs = results["documents"]
    vals = [d["wer"] * 100 for d in docs if d["wer"] < 2]
    mean_v, med_v = np.mean(vals), np.median(vals)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(vals, bins=20, color=VIOLET, edgecolor="#ffffff", linewidth=0.8, alpha=0.85)
    ax.axvline(med_v, color=ROSE, ls="--", lw=2, label=f"Median  {med_v:.1f}%")
    ax.axvline(mean_v, color=AMBER, ls=":", lw=2, label=f"Mean  {mean_v:.1f}%")
    ax.set_xlabel("Word Error Rate (%)")
    ax.set_ylabel("Documents")
    ax.set_title("WER Distribution — DocLD on FUNSD")
    ax.legend(frameon=True)
    _brand_grid(ax)
    _watermark(fig)
    save_chart(fig, "wer-histogram", charts_dir, public_dir)


def create_cer_wer_scatter(results, charts_dir, public_dir):
    docs = results["documents"]
    cer = np.array([d["cer"] * 100 for d in docs])
    wer = np.array([d["wer"] * 100 for d in docs])
    mask = cer < 150
    cer_f, wer_f = cer[mask], wer[mask]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.scatter(cer_f, wer_f, c=VIOLET, alpha=0.65, s=72, edgecolors="#ffffff", linewidths=0.8, zorder=3)
    z = np.polyfit(cer_f, wer_f, 1)
    x_ln = np.linspace(cer_f.min(), cer_f.max(), 100)
    ax.plot(x_ln, np.poly1d(z)(x_ln), color=ROSE, ls="--", lw=1.8, label="Trend", zorder=2)
    ax.set_xlabel("CER (%)")
    ax.set_ylabel("WER (%)")
    ax.set_title("CER vs WER per Document — DocLD")
    ax.legend(frameon=True)
    _brand_grid(ax, "both")
    _watermark(fig)
    save_chart(fig, "cer-wer-scatter", charts_dir, public_dir)


def create_error_rate_boxplot(results, charts_dir, public_dir):
    docs = results["documents"]
    cer = [d["cer"] * 100 for d in docs if d["cer"] < 2]
    wer = [d["wer"] * 100 for d in docs if d["wer"] < 2]

    fig, ax = plt.subplots(figsize=(7, 5.5))
    bp = ax.boxplot(
        [cer, wer], tick_labels=["CER", "WER"], patch_artist=True,
        widths=0.45, showfliers=True,
        flierprops=dict(marker="o", markerfacecolor=MAUVE_400, markersize=4, alpha=0.5),
        medianprops=dict(color="#ffffff", linewidth=2),
        whiskerprops=dict(color=MAUVE_400),
        capprops=dict(color=MAUVE_400),
    )
    for patch, color in zip(bp["boxes"], [ROSE, VIOLET]):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)
        patch.set_edgecolor("#ffffff")
    ax.set_ylabel("Error Rate (%)")
    ax.set_title("Error Rate Distribution — DocLD")
    _brand_grid(ax)
    _watermark(fig)
    save_chart(fig, "cer-wer-box", charts_dir, public_dir)


def create_duration_histogram(results, charts_dir, public_dir):
    docs = results["documents"]
    dur = [d["duration_ms"] / 1000 for d in docs]
    mean_d, med_d = np.mean(dur), np.median(dur)

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.hist(dur, bins=15, color=EMERALD, edgecolor="#ffffff", linewidth=0.8, alpha=0.85)
    ax.axvline(med_d, color=VIOLET, ls="--", lw=2, label=f"Median  {med_d:.1f}s")
    ax.axvline(mean_d, color=AMBER, ls=":", lw=2, label=f"Mean  {mean_d:.1f}s")
    ax.set_xlabel("Processing Time (seconds)")
    ax.set_ylabel("Documents")
    ax.set_title("Processing Time Distribution — DocLD")
    ax.legend(frameon=True)
    _brand_grid(ax)
    _watermark(fig)
    save_chart(fig, "duration-histogram", charts_dir, public_dir)


def create_entity_count_vs_cer(analysis, charts_dir, public_dir):
    sd = analysis["scatter_data"]
    ec = np.array([d["entity_count"] for d in sd])
    cer = np.array([d["cer"] * 100 for d in sd])
    mask = cer < 150
    ec_f, cer_f = ec[mask], cer[mask]

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(ec_f, cer_f, c=ROSE, alpha=0.65, s=72, edgecolors="#ffffff", linewidths=0.8, zorder=3)
    z = np.polyfit(ec_f, cer_f, 1)
    x_ln = np.linspace(ec_f.min(), ec_f.max(), 100)
    ax.plot(x_ln, np.poly1d(z)(x_ln), color=VIOLET, ls="--", lw=1.8, zorder=2)
    corr = analysis["correlations"]["entity_count_vs_cer"]
    ax.text(
        0.96, 0.96, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=11,
        va="top", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec=MAUVE_200, alpha=0.9),
    )
    ax.set_xlabel("Entity Count")
    ax.set_ylabel("CER (%)")
    ax.set_title("Document Complexity vs Error Rate — DocLD")
    _brand_grid(ax, "both")
    _watermark(fig)
    save_chart(fig, "entity-count-vs-cer", charts_dir, public_dir)


def create_entity_type_breakdown(results, charts_dir, public_dir):
    entity_types = ["Question", "Answer", "Header", "Other"]
    keys = ["question", "answer", "header", "other"]
    counts = {k: 0 for k in keys}
    for doc in results["documents"]:
        for k in keys:
            counts[k] += doc["entity_breakdown"].get(k, 0)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(entity_types))
    bars = ax.bar(x, [counts[k] for k in keys], color=ACCENT_PALETTE, edgecolor="#ffffff",
                  linewidth=0.8, width=0.55, zorder=3)
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 12, f"{int(h):,}",
                ha="center", va="bottom", fontsize=11, fontweight="bold", color=SLATE_700)
    ax.set_xticks(x)
    ax.set_xticklabels(entity_types)
    ax.set_ylabel("Total Count")
    ax.set_title("FUNSD Entity Type Distribution")
    _brand_grid(ax)
    _watermark(fig)
    save_chart(fig, "entity-type-breakdown", charts_dir, public_dir)


def create_cumulative_distribution(results, charts_dir, public_dir):
    docs = results["documents"]
    cer = sorted([d["cer"] * 100 for d in docs if d["cer"] < 2])
    cum = np.arange(1, len(cer) + 1) / len(cer) * 100

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.plot(cer, cum, color=VIOLET, lw=2.5, zorder=3)
    ax.fill_between(cer, cum, alpha=0.12, color=VIOLET)
    for thr in [25, 50, 75]:
        idx = np.searchsorted(cer, thr)
        if idx < len(cum):
            pct = cum[idx]
            ax.plot(thr, pct, "o", color=ROSE, markersize=7, zorder=4)
            ax.annotate(
                f"{pct:.0f}% < {thr}%", (thr, pct), textcoords="offset points",
                xytext=(10, -4), fontsize=9, color=SLATE_700,
            )
    ax.set_xlabel("CER (%)")
    ax.set_ylabel("Cumulative % of Documents")
    ax.set_title("CER Cumulative Distribution — DocLD")
    ax.set_xlim(0, max(cer) * 1.08)
    ax.set_ylim(0, 105)
    _brand_grid(ax, "both")
    _watermark(fig)
    save_chart(fig, "cer-cumulative", charts_dir, public_dir)


def create_per_document_heatmap(results, charts_dir, public_dir):
    docs = results["documents"]
    sorted_docs = sorted(docs, key=lambda d: d.get("word_match_accuracy", 0), reverse=True)[:30]
    names = [d["filename"][:15] for d in sorted_docs]
    accs = [d.get("word_match_accuracy", 0) * 100 for d in sorted_docs]

    fig, ax = plt.subplots(figsize=(11, 8))
    colors = []
    for a in accs:
        if a >= 95:
            colors.append(EMERALD)
        elif a >= 90:
            colors.append(VIOLET)
        elif a >= 80:
            colors.append(AMBER)
        else:
            colors.append(ROSE)

    bars = ax.barh(names, accs, color=colors, edgecolor="#ffffff", linewidth=0.6, height=0.7, zorder=3)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_width() + 0.8, bar.get_y() + bar.get_height() / 2,
                f"{acc:.1f}%", va="center", fontsize=9, color=SLATE_500)
    ax.set_xlabel("Word-Match Accuracy (%)")
    ax.set_title("Per-Document Accuracy (Top 30) — DocLD")
    ax.invert_yaxis()
    ax.set_xlim(0, 108)
    legend_el = [
        mpatches.Patch(fc=EMERALD, ec="#fff", label="≥ 95%"),
        mpatches.Patch(fc=VIOLET, ec="#fff", label="90–95%"),
        mpatches.Patch(fc=AMBER, ec="#fff", label="80–90%"),
        mpatches.Patch(fc=ROSE, ec="#fff", label="< 80%"),
    ]
    ax.legend(handles=legend_el, loc="lower right", frameon=True)
    _brand_grid(ax, "x")
    _watermark(fig)
    save_chart(fig, "per-document-accuracy", charts_dir, public_dir)


def create_performance_summary(results, analysis, charts_dir, public_dir):
    docs = results["documents"]
    cer = [d["cer"] * 100 for d in docs if d["cer"] < 2]
    wer = [d["wer"] * 100 for d in docs if d["wer"] < 2]
    wm = results.get("word_match_accuracy", {})
    med_wm = wm.get("median") or (np.median([d.get("word_match_accuracy", 0) for d in docs]) * 100)

    fig = plt.figure(figsize=(16, 11))
    gs = fig.add_gridspec(3, 3, hspace=0.38, wspace=0.32)

    # ── Panel 1: Key metrics ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    metrics = [
        ("Parse Success", f"{results['summary']['success_rate']:.0f}%", EMERALD),
        ("Word-Match Acc", f"{med_wm:.1f}%", VIOLET),
        ("Median WER", f"{np.median(wer):.1f}%", ROSE),
        ("Documents", str(len(docs)), AMBER),
    ]
    for i, (label, val, color) in enumerate(metrics):
        y = 0.82 - i * 0.22
        ax1.text(0.05, y, label, fontsize=11, color=SLATE_500, transform=ax1.transAxes)
        ax1.text(0.95, y, val, fontsize=14, fontweight="bold", color=color,
                 ha="right", transform=ax1.transAxes)
    ax1.set_title("Key Metrics", pad=12)

    # ── Panel 2: CER hist ──
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(cer, bins=15, color=ROSE, edgecolor="#fff", alpha=0.85)
    ax2.axvline(np.median(cer), color=VIOLET, ls="--", lw=1.5)
    ax2.set_xlabel("CER (%)", fontsize=9)
    ax2.set_title("CER Distribution", pad=10)
    _brand_grid(ax2)

    # ── Panel 3: WER hist ──
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.hist(wer, bins=15, color=VIOLET, edgecolor="#fff", alpha=0.85)
    ax3.axvline(np.median(wer), color=ROSE, ls="--", lw=1.5)
    ax3.set_xlabel("WER (%)", fontsize=9)
    ax3.set_title("WER Distribution", pad=10)
    _brand_grid(ax3)

    # ── Panel 4: CER vs WER scatter ──
    ax4 = fig.add_subplot(gs[1, 0])
    mask = np.array([d["cer"] for d in docs]) < 1.5
    cf = np.array([d["cer"] * 100 for d in docs])[mask]
    wf = np.array([d["wer"] * 100 for d in docs])[mask]
    ax4.scatter(cf, wf, c=VIOLET, alpha=0.55, s=36, edgecolors="#fff", linewidths=0.5)
    ax4.set_xlabel("CER (%)", fontsize=9)
    ax4.set_ylabel("WER (%)", fontsize=9)
    ax4.set_title("CER vs WER", pad=10)
    _brand_grid(ax4, "both")

    # ── Panel 5: Entity pie ──
    ax5 = fig.add_subplot(gs[1, 1])
    e_types = ["Question", "Answer", "Header", "Other"]
    e_keys = ["question", "answer", "header", "other"]
    counts = [sum(d["entity_breakdown"].get(k, 0) for d in docs) for k in e_keys]
    wedges, texts, autotexts = ax5.pie(
        counts, labels=e_types, colors=ACCENT_PALETTE,
        autopct="%1.0f%%", startangle=90, pctdistance=0.78,
        wedgeprops=dict(edgecolor="#ffffff", linewidth=1.5),
    )
    for t in autotexts:
        t.set_fontsize(9)
        t.set_color("#ffffff")
        t.set_fontweight("bold")
    ax5.set_title("Entity Types", pad=10)

    # ── Panel 6: Duration hist ──
    ax6 = fig.add_subplot(gs[1, 2])
    durations = [d["duration_ms"] / 1000 for d in docs]
    ax6.hist(durations, bins=15, color=EMERALD, edgecolor="#fff", alpha=0.85)
    ax6.set_xlabel("Time (s)", fontsize=9)
    ax6.set_title("Processing Time", pad=10)
    _brand_grid(ax6)

    # ── Panel 7: Top performers bar ──
    ax7 = fig.add_subplot(gs[2, :])
    top = sorted(docs, key=lambda d: d.get("word_match_accuracy", 0), reverse=True)[:15]
    names = [d["filename"][:14] for d in top]
    accs = [d.get("word_match_accuracy", 0) * 100 for d in top]
    bar_colors = [EMERALD if a >= 95 else VIOLET if a >= 90 else AMBER for a in accs]
    ax7.barh(names, accs, color=bar_colors, edgecolor="#fff", linewidth=0.5, height=0.65, zorder=3)
    ax7.invert_yaxis()
    ax7.set_xlim(0, 108)
    ax7.set_xlabel("Word-Match Accuracy (%)", fontsize=9)
    ax7.set_title("Top 15 Documents by Accuracy", pad=10)
    _brand_grid(ax7, "x")

    fig.suptitle("DocLD × FUNSD Benchmark — Performance Summary",
                 fontsize=17, fontweight="bold", color=MAUVE_900, y=0.995)
    _watermark(fig)
    save_chart(fig, "performance-summary", charts_dir, public_dir)


def create_word_count_vs_duration(analysis, charts_dir, public_dir):
    sd = analysis["scatter_data"]
    wc = np.array([d["word_count"] for d in sd])
    dur = np.array([d["duration_ms"] / 1000 for d in sd])

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.scatter(wc, dur, c=EMERALD, alpha=0.65, s=72, edgecolors="#ffffff", linewidths=0.8, zorder=3)
    z = np.polyfit(wc, dur, 1)
    x_ln = np.linspace(wc.min(), wc.max(), 100)
    ax.plot(x_ln, np.poly1d(z)(x_ln), color=ROSE, ls="--", lw=1.8, zorder=2)
    corr = analysis["correlations"]["word_count_vs_duration"]
    ax.text(
        0.96, 0.06, f"r = {corr:.3f}", transform=ax.transAxes, fontsize=11,
        va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="#ffffff", ec=MAUVE_200, alpha=0.9),
    )
    ax.set_xlabel("Word Count (Ground Truth)")
    ax.set_ylabel("Processing Time (s)")
    ax.set_title("Document Size vs Processing Time — DocLD")
    _brand_grid(ax, "both")
    _watermark(fig)
    save_chart(fig, "word-count-vs-duration", charts_dir, public_dir)


def create_accuracy_by_entity_type(results, charts_dir, public_dir):
    entity_labels = ["Question", "Answer", "Header", "Other"]
    entity_keys = ["question", "answer", "header", "other"]

    # per_entity_type_cer now stores word-match accuracy (0-1) per entity type
    avg_wm = {k: [] for k in entity_keys}
    for doc in results["documents"]:
        for k in entity_keys:
            v = doc["per_entity_type_cer"].get(k, 0)
            if v > 0:
                avg_wm[k].append(v * 100)

    accs = [np.mean(avg_wm[k]) if avg_wm[k] else 0 for k in entity_keys]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(entity_labels))
    bars = ax.bar(x, accs, color=ACCENT_PALETTE, edgecolor="#ffffff", linewidth=0.8,
                  width=0.55, zorder=3)
    for bar, a in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                f"{a:.1f}%", ha="center", fontsize=11, fontweight="bold", color=SLATE_700)
    ax.set_xticks(x)
    ax.set_xticklabels(entity_labels)
    ax.set_ylabel("Word-Match Accuracy (%)")
    ax.set_title("OCR Accuracy by Entity Type — DocLD")
    ax.set_ylim(0, 105)
    _brand_grid(ax)
    _watermark(fig)
    save_chart(fig, "accuracy-by-entity-type", charts_dir, public_dir)


# ===================================================================
def main():
    print("Loading results…")
    results, analysis = load_results()
    print("Creating directories…")
    charts_dir, public_dir = create_output_dirs()

    print("\nGenerating charts:")
    create_hero_banner(results, charts_dir, public_dir)
    create_cer_histogram(results, charts_dir, public_dir)
    create_wer_histogram(results, charts_dir, public_dir)
    create_cer_wer_scatter(results, charts_dir, public_dir)
    create_error_rate_boxplot(results, charts_dir, public_dir)
    create_duration_histogram(results, charts_dir, public_dir)
    create_entity_count_vs_cer(analysis, charts_dir, public_dir)
    create_entity_type_breakdown(results, charts_dir, public_dir)
    create_cumulative_distribution(results, charts_dir, public_dir)
    create_per_document_heatmap(results, charts_dir, public_dir)
    create_performance_summary(results, analysis, charts_dir, public_dir)
    create_word_count_vs_duration(analysis, charts_dir, public_dir)
    create_accuracy_by_entity_type(results, charts_dir, public_dir)

    print(f"\nDone — {charts_dir}")
    print(f"     — {public_dir}")


if __name__ == "__main__":
    main()
