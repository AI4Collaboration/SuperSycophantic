"""Generate vector plots, print the HTML poster, and export web images."""

from pathlib import Path
from io import StringIO
import shutil
import subprocess

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from PIL import Image
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics import renderSVG
from reportlab.graphics.shapes import Drawing

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
STEM = "SuperSycophantic-NeurIPS-2026-Sydney"
URL = "https://github.com/AI4Collaboration/SuperSycophantic"
OBJ, SUB, NEUTRAL = "#087F8C", "#C45D37", "#929C9F"
INK, GRID = "#25343B", "#C6D6DA"
TEXT, SMALL, EMPHASIS = 30, 28, 34

# Camera-ready context/controlled-intervention tables and a supplementary
# mode-separated Claude recount. Only aggregate counts are included.
MODELS = ["GPT-5.4", "GPT-5.4-mini", "GPT-5.4-nano", "Opus-4.5", "Sonnet-4.5",
          "Haiku-4.5", "Gemini-3.1-Flash-Lite", "Mistral-Medium-3.1", "Command-R"]
CONTEXT_OBJ_NEUTRAL = [65.5, 63.5, 48.5, 69, 63.5, 60, 63, 58, 31]
CONTEXT_OBJ_FRAMED = [61.2, 51.5, 31.8, 62, 56, 52.3, 57, 49.5, 12]
CONTEXT_SUB_FRAMED = [71.5, 84.5, 87.5, 68.2, 70.7, 63, 78.2, 69, 85.7]
CLAUDE_SUB = {
    "Static": {"Opus-4.5": ([177, 486, 460], [700, 700, 700]),
               "Sonnet-4.5": ([426, 641, 641], [700, 699, 700]),
               "Haiku-4.5": ([418, 664, 644], [700, 700, 700])},
    "Adaptive": {"Opus-4.5": ([371, 531, 449], [700, 700, 700]),
                 "Sonnet-4.5": ([585, 648, 610], [700, 699, 700]),
                 "Haiku-4.5": ([583, 644, 625], [700, 700, 700])},
}
OTHER_SUB = {
    "Static": ([701, 2728, 3300], [4200, 4200, 4199]),
    "Adaptive": ([1676, 3098, 3246], [4200, 4199, 4199]),
}
CERTAINTY = {"OBJ": ([153, 215, 352], 1055), "SUB": ([501, 714, 837], 900)}
RECHECK = {"OBJ": ([17, 8], 96), "SUB": ([34, 13], 120)}


def style():
    for weight in ["Regular", "Bold"]:
        matplotlib.font_manager.fontManager.addfont(ASSETS / "fonts" / f"Lato-{weight}.ttf")
    plt.rcParams.update({"font.family": "Lato", "font.size": TEXT,
        "svg.fonttype": "path", "svg.hashsalt": "supersycophantic-poster",
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.titleweight": "bold", "axes.titlesize": TEXT, "legend.frameon": False})


def save(fig, name, transparent=True):
    buffer = StringIO()
    fig.savefig(buffer, format="svg", transparent=transparent,
                facecolor="none" if transparent else "white", metadata={"Date": None})
    (ASSETS / name).write_text("\n".join(line.rstrip() for line in buffer.getvalue().splitlines()) + "\n", encoding="utf-8")
    plt.close(fig)


def context_plot():
    # Match the HTML slot in physical units so plot text retains its point size.
    fig, axes = plt.subplots(1, 2, figsize=(420 / 25.4, 258 / 25.4), sharey=True)
    fig.subplots_adjust(left=.32, right=.97, bottom=.11, top=.855, wspace=.27)
    y = np.arange(len(MODELS))
    for ax, before, after, color, title in zip(
            axes, [CONTEXT_OBJ_NEUTRAL, [50] * 9],
            [CONTEXT_OBJ_FRAMED, CONTEXT_SUB_FRAMED], [OBJ, SUB], ["OBJ", "SUB"]):
        ax.hlines(y, before, after, color=NEUTRAL, linewidth=2.6)
        ax.scatter(before, y, color=NEUTRAL, s=75, marker="o", zorder=3)
        ax.scatter(after, y, color=color, s=95, marker="D", zorder=4)
        ax.set_title(title, pad=18, color=color)
        ax.set_yticks(y, MODELS, fontsize=SMALL)
        ax.tick_params(axis="both", length=0, pad=7)
        ax.set_xlabel("Accuracy (%)" if title == "OBJ" else "User alignment (%)", fontsize=TEXT)
        ax.set_xlim((0, 80) if title == "OBJ" else (45, 95))
        ax.set_xticks([0, 40, 80] if title == "OBJ" else [50, 70, 90])
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", color=NEUTRAL, ls="", label="Neutral (OBJ) / 50% (SUB)"),
                        Line2D([], [], marker="D", color=INK, ls="", label="User framing")],
               loc="upper center", ncol=2, fontsize=SMALL, bbox_to_anchor=(.5, 1.0))
    save(fig, "context.svg")


def claude_plot():
    fig, axes = plt.subplots(2, 1, figsize=(410 / 25.4, 438 / 25.4))
    fig.subplots_adjust(left=.13, right=.94, bottom=.045, top=.87, hspace=.28)
    palette = ["#C8B3ED", "#8EE0E5", "#FFB18C"]
    for ax, (mode, records) in zip(axes, CLAUDE_SUB.items()):
        counts, ns = OTHER_SUB[mode]
        ax.plot(range(3), np.array(counts) / np.array(ns) * 100,
                color="white", marker="D", markersize=9, linewidth=3,
                linestyle=":", label="Other six (pooled)")
        for (name, (counts, ns)), color, marker, ls in zip(records.items(), palette, ["o", "s", "^"], ["-", "--", "-."]):
            values = np.array(counts) / np.array(ns) * 100
            ax.plot(range(3), values, color=color, marker=marker, markersize=10, linewidth=3.5, linestyle=ls, label=name)
            if name == "Opus-4.5" and mode == "Adaptive":
                for i in [1, 2]:
                    ax.annotate(f"{values[i]:.1f}%", (i, values[i]), xytext=(0, -33), textcoords="offset points",
                                ha="center", color=color, weight="bold", fontsize=TEXT,
                                bbox={"facecolor": "#163e4d", "edgecolor": "none", "pad": 1.5})
        ax.set(xticks=range(3), xticklabels=["Mild", "Moderate", "Strong"], ylim=(0, 105), xlim=(-.15, 2.15),
               yticks=[0, 50, 100])
        ax.tick_params(colors="white", labelsize=TEXT, length=0, pad=8)
        ax.set_title(mode, pad=20, fontsize=TEXT, color="white", loc="left")
        ax.spines["bottom"].set_color("#87A4AF")
        ax.grid(axis="y", color="#61808B", alpha=.55)
        ax.set_axisbelow(True)
    fig.supylabel("SUB switching (%)", x=.012, fontsize=TEXT, color="white")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.53, 1.01),
               ncol=2, fontsize=SMALL, columnspacing=1.3, labelcolor="white")
    save(fig, "claude-tone.svg", transparent=True)


def certainty_plot():
    fig, ax = plt.subplots(figsize=(420 / 25.4, 210 / 25.4))
    fig.subplots_adjust(left=.13, right=.97, bottom=.17, top=.80)
    for name, color, marker, ls in [("SUB", SUB, "s", "-"), ("OBJ", OBJ, "o", "--")]:
        counts, n = CERTAINTY[name]
        values = np.array(counts) / n * 100
        ax.plot(range(3), values, color=color, marker=marker, linestyle=ls, linewidth=3, markersize=11,
                label="SUB switch" if name == "SUB" else "OBJ correct to incorrect")
        for i in [0, 2]:
            v = values[i]
            ax.annotate(f"{v:.1f}%", (i, v), xytext=(0, 13), textcoords="offset points",
                        ha="center", fontsize=EMPHASIS, weight="bold", color=color)
    ax.set(ylim=(0, 108), xlim=(-.16, 2.16), ylabel="Outcome rate (%)", yticks=range(0, 101, 20),
           xticks=range(3), xticklabels=["Might be\nwrong", "Probably\nwrong", "Definitely\nwrong"])
    ax.legend(loc="upper left", bbox_to_anchor=(-.02, 1.33), ncol=1, fontsize=SMALL)
    ax.tick_params(labelsize=TEXT, length=0, pad=7)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    save(fig, "certainty.svg")


def recheck_plot():
    fig, axes = plt.subplots(1, 2, figsize=(420 / 25.4, 244 / 25.4), sharey=True)
    fig.subplots_adjust(left=.13, right=.97, bottom=.10, top=.92, wspace=.22)
    for ax, name, color in zip(axes, ["OBJ", "SUB"], [OBJ, SUB]):
        counts, n = RECHECK[name]
        values = np.array(counts) / n * 100
        ax.bar([0, 1], values, color=[NEUTRAL, color], width=.57)
        for i, count, v in zip([0, 1], counts, values):
            ax.text(i, v + 1.3, f"{v:.1f}%", ha="center", fontsize=EMPHASIS, weight="bold", color=color)
        ax.set(xticks=[0, 1], xticklabels=["Baseline", "+ Recheck"], ylim=(0, 36), yticks=[0, 10, 20, 30])
        ax.set_title("OBJ correct to incorrect" if name == "OBJ" else "SUB switching", fontsize=SMALL, color=color, pad=18)
        ax.tick_params(axis="both", length=0, pad=9)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Outcome rate (%)")
    save(fig, "recheck.svg")


def qr():
    code = QrCodeWidget(URL)
    x0, y0, x1, y1 = code.getBounds()
    drawing = Drawing(200, 200, transform=[200 / (x1 - x0), 0, 0, 200 / (y1 - y0), 0, 0])
    drawing.add(code)
    renderSVG.drawToFile(drawing, str(ASSETS / "code-qr.svg"))


def verify_and_export():
    with pymupdf.open(ROOT / f"{STEM}.pdf") as doc:
        assert len(doc) == 1, "Poster must be a single page"
        p = doc[0]
        assert "SuperSycophantic" in p.get_text(), "Poster content is missing"
        assert abs(p.rect.width * 2.54 / 72 - 140) < .04
        assert abs(p.rect.height * 2.54 / 72 - 90) < .04
        # Chromium rounds custom paper sizes; normalize the empty outer margin.
        p.set_mediabox(pymupdf.Rect(0, 0, 140 / 2.54 * 72, 90 / 2.54 * 72))
        for b in p.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    assert p.rect.contains(pymupdf.Rect(s["bbox"])), s["text"]
                    assert "\ufffd" not in s["text"], s
        for font in p.get_fonts(full=True):
            assert doc.extract_font(font[0])[3], font
            assert any(name in font[3] for name in ["Raleway", "Lato"]), f"Unexpected font: {font[3]}"
        assert any(x.get("uri") == URL for x in p.get_links())
        pdf_bytes = doc.tobytes(garbage=4, deflate=True)
        pix = p.get_pixmap(matrix=pymupdf.Matrix(2880 / p.rect.height, 2880 / p.rect.height), alpha=False)
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        im.save(ROOT / f"{STEM}.png", optimize=True)
    (ROOT / f"{STEM}.pdf").write_bytes(pdf_bytes)
    logo = Image.open(ASSETS / "supersycophantic-logo.png").convert("RGB")
    logo.thumbnail((320, 256), Image.Resampling.LANCZOS)
    logo.save(ROOT / "thumbnail.png", optimize=True)
    for name, width, height, limit in [(f"{STEM}.png", 5120, 2880, 10_000_000), ("thumbnail.png", 320, 256, 5_000_000)]:
        p = ROOT / name
        with Image.open(p) as im:
            assert im.mode == "RGB" and im.width <= width and im.height <= height
        assert p.stat().st_size < limit
    print("Verified: one-page 140 x 90 cm HTML PDF, embedded fonts, text bounds, code link, and web image limits.")


def main():
    style()
    for fn in [context_plot, claude_plot, certainty_plot, recheck_plot, qr]:
        fn()
    node = shutil.which("node")
    if not node:
        raise SystemExit("Install Node.js and the poster's npm dependencies before building.")
    subprocess.run([node, str(ROOT / "render_poster.cjs")], cwd=ROOT, check=True)
    verify_and_export()


if __name__ == "__main__":
    main()
