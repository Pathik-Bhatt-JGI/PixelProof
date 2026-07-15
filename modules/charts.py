"""Dark-themed chart rendering for the statistical forensic signals,
matching the app's visual identity. Uses matplotlib's Agg backend.
"""
import io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

BG = "#060C12"
GREEN = "#00FFB3"
CYAN = "#00B8FF"
AMBER = "#FFD700"
GRID = "#123"
TEXT = "#7A95B0"


def _style_ax(ax, fig):
    fig.patch.set_facecolor(BG)
    ax.set_facecolor(BG)
    ax.tick_params(colors=TEXT, labelsize=8)
    for spine in ax.spines.values():
        spine.set_color("#0F2030")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)
    ax.title.set_color("#E8F0F8")
    ax.grid(True, color="#0F2030", linewidth=0.6, alpha=0.6)


def _fig_to_pil(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def benford_chart(observed, expected) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    _style_ax(ax, fig)
    digits = list(range(1, 10))
    width = 0.38
    ax.bar([d - width / 2 for d in digits], observed, width=width, color=GREEN, label="Observed", alpha=0.9)
    ax.bar([d + width / 2 for d in digits], expected, width=width, color=CYAN, label="Benford's Law", alpha=0.75)
    ax.set_xticks(digits)
    ax.set_xlabel("Leading digit")
    ax.set_ylabel("Frequency")
    ax.set_title("DCT Coefficient Leading-Digit Distribution")
    leg = ax.legend(facecolor=BG, edgecolor="#0F2030", labelcolor=TEXT, fontsize=8)
    return _fig_to_pil(fig)


def double_compression_chart(histogram, bin_edges) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    _style_ax(ax, fig)
    centers = [(bin_edges[i] + bin_edges[i + 1]) / 2 for i in range(len(bin_edges) - 1)]
    ax.plot(centers, histogram, color=GREEN, linewidth=1.3)
    ax.fill_between(centers, histogram, color=GREEN, alpha=0.15)
    ax.set_xlabel("DCT coefficient value")
    ax.set_ylabel("Frequency")
    ax.set_title("DCT(1,1) Coefficient Histogram")
    return _fig_to_pil(fig)


def chromatic_aberration_chart(samples) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    _style_ax(ax, fig)
    radii = [s["radius"] for s in samples]
    mags = [s["magnitude"] for s in samples]
    ax.scatter(radii, mags, color=CYAN, s=28, alpha=0.85, edgecolors=GREEN, linewidths=0.4)
    ax.set_xlabel("Normalized distance from centre")
    ax.set_ylabel("Channel misalignment (px)")
    ax.set_title("Chromatic Aberration vs. Radial Position")
    return _fig_to_pil(fig)
