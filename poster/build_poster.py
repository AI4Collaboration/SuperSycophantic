"""Generate vector plots, compile the LaTeX poster, and export web images."""

from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from PIL import Image
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
STEM = "SuperSycophantic-NeurIPS-2026-Sydney"
URL = "https://github.com/AI4Collaboration/SuperSycophantic"
OBJ, SUB, NEUTRAL = "#007C87", "#B34B35", "#89949A"
INK, GRID = "#25343B", "#DCE2E5"

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
CERTAINTY = {"OBJ": ([153, 215, 352], 1055), "SUB": ([501, 714, 837], 900)}
RECHECK = {"OBJ": ([17, 8], 96), "SUB": ([34, 13], 120)}


def style():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18,
        "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
        "axes.spines.left": False, "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": INK, "ytick.color": INK,
        "axes.titleweight": "bold", "axes.titlesize": 20, "legend.frameon": False})


def save(fig, name):
    fig.savefig(ASSETS / name, facecolor="white")
    plt.close(fig)


def context_plot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.1), sharey=True)
    fig.subplots_adjust(left=.31, right=.97, bottom=.14, top=.84, wspace=.25)
    y = np.arange(len(MODELS))
    for ax, before, after, color, title in zip(
            axes, [CONTEXT_OBJ_NEUTRAL, [50] * 9],
            [CONTEXT_OBJ_FRAMED, CONTEXT_SUB_FRAMED], [OBJ, SUB], ["OBJ", "SUB"]):
        ax.hlines(y, before, after, color=NEUTRAL, linewidth=2.6)
        ax.scatter(before, y, color=NEUTRAL, s=75, marker="o", zorder=3)
        ax.scatter(after, y, color=color, s=95, marker="D", zorder=4)
        ax.set_title(title, pad=18, color=color)
        ax.set_yticks(y, MODELS, fontsize=16)
        ax.tick_params(axis="both", length=0, pad=7)
        ax.set_xlabel("Accuracy (%)" if title == "OBJ" else "User alignment (%)", fontsize=17)
        ax.set_xlim((0, 80) if title == "OBJ" else (45, 95))
        ax.set_xticks([0, 40, 80] if title == "OBJ" else [50, 70, 90])
        ax.grid(axis="x", color=GRID)
        ax.set_axisbelow(True)
    axes[0].invert_yaxis()
    from matplotlib.lines import Line2D
    fig.legend(handles=[Line2D([], [], marker="o", color=NEUTRAL, ls="", label="Neutral / balanced reference"),
                        Line2D([], [], marker="D", color=INK, ls="", label="User framing")],
               loc="upper center", ncol=2, fontsize=16, bbox_to_anchor=(.54, .99))
    save(fig, "context.pdf")


def claude_plot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 7.1), sharey=True)
    fig.subplots_adjust(left=.12, right=.97, bottom=.15, top=.79, wspace=.17)
    palette = ["#72528B", "#007C87", "#BC6B28"]
    for ax, (mode, records) in zip(axes, CLAUDE_SUB.items()):
        for (name, (counts, ns)), color, marker, ls in zip(records.items(), palette, ["o", "s", "^"], ["-", "--", "-."]):
            values = np.array(counts) / np.array(ns) * 100
            ax.plot(range(3), values, color=color, marker=marker, markersize=9, linewidth=2.6, linestyle=ls, label=name)
            if name == "Opus-4.5":
                for i in [1, 2]:
                    ax.annotate(f"{values[i]:.1f}%", (i, values[i]), xytext=(0, -27), textcoords="offset points",
                                ha="center", color=color, weight="bold", fontsize=18)
        ax.set(xticks=range(3), xticklabels=["Mild", "Moderate", "Strong"], ylim=(0, 100), xlim=(-.25, 2.25),
               yticks=range(0, 101, 20))
        ax.tick_params(axis="x", labelsize=16)
        ax.set_title(mode, pad=18, fontsize=19)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("SUB switching (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.55, .99), ncol=3, fontsize=16, columnspacing=1.0)
    save(fig, "claude-tone.pdf")


def certainty_plot():
    fig, ax = plt.subplots(figsize=(12, 6.1))
    fig.subplots_adjust(left=.13, right=.96, bottom=.19, top=.80)
    for name, color, marker, ls in [("SUB", SUB, "s", "-"), ("OBJ", OBJ, "o", "--")]:
        counts, n = CERTAINTY[name]
        values = np.array(counts) / n * 100
        ax.plot(range(3), values, color=color, marker=marker, linestyle=ls, linewidth=3, markersize=11,
                label="SUB switch" if name == "SUB" else "OBJ correct to incorrect")
        for i, v in enumerate(values):
            ax.annotate(f"{v:.1f}%", (i, v), xytext=(0, 13), textcoords="offset points",
                        ha="center", fontsize=21, weight="bold", color=color)
    ax.set(ylim=(0, 108), xlim=(-.16, 2.16), ylabel="Outcome rate (%)", yticks=range(0, 101, 20),
           xticks=range(3), xticklabels=["Might be\nwrong", "Probably\nwrong", "Definitely\nwrong"])
    ax.legend(loc="upper center", bbox_to_anchor=(.48, 1.28), ncol=2, fontsize=17)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    save(fig, "certainty.pdf")


def recheck_plot():
    fig, axes = plt.subplots(1, 2, figsize=(12, 6.1), sharey=True)
    fig.subplots_adjust(left=.13, right=.97, bottom=.20, top=.83, wspace=.22)
    for ax, name, color in zip(axes, ["OBJ", "SUB"], [OBJ, SUB]):
        counts, n = RECHECK[name]
        values = np.array(counts) / n * 100
        ax.bar([0, 1], values, color=[NEUTRAL, color], width=.57)
        for i, count, v in zip([0, 1], counts, values):
            ax.text(i, v + 1.3, f"{v:.1f}%", ha="center", fontsize=22, weight="bold", color=color)
            ax.text(i, 1.0, f"{count}/{n}", ha="center", va="bottom", fontsize=17, color="white", weight="bold")
        ax.set(xticks=[0, 1], xticklabels=["Baseline", "+ Recheck"], ylim=(0, 36), yticks=[0, 10, 20, 30])
        ax.set_title("OBJ correct to incorrect" if name == "OBJ" else "SUB switching", fontsize=18, color=color, pad=18)
        ax.tick_params(axis="both", length=0, pad=9)
        ax.grid(axis="y", color=GRID)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Outcome rate (%)")
    save(fig, "recheck.pdf")


def qr():
    code = QrCodeWidget(URL)
    x0, y0, x1, y1 = code.getBounds()
    pdfmetrics.registerFont(TTFont("PosterQR", matplotlib.font_manager.findfont("DejaVu Sans")))
    canvas = Canvas(str(ASSETS / "code-qr.pdf"), pagesize=(200, 200), initialFontName="PosterQR")
    canvas.scale(200 / (x1 - x0), 200 / (y1 - y0))
    canvas.setFillColorRGB(0, 0, 0)
    for rect in code.draw().contents:
        if rect.fillColor is not None:
            canvas.rect(rect.x, rect.y, rect.width, rect.height, stroke=0, fill=1)
    canvas.showPage()
    canvas.save()


def verify_and_export():
    with pymupdf.open(ROOT / f"{STEM}.pdf") as doc:
        assert len(doc) == 1, "Poster must be a single page"
        p = doc[0]
        assert "SuperSycophantic" in p.get_text(), "Poster content is missing"
        assert abs(p.rect.width * 2.54 / 72 - 140) < .01
        assert abs(p.rect.height * 2.54 / 72 - 90) < .01
        for b in p.get_text("dict")["blocks"]:
            for line in b.get("lines", []):
                for s in line["spans"]:
                    assert p.rect.contains(pymupdf.Rect(s["bbox"])), s["text"]
                    assert "\ufffd" not in s["text"], s
        for font in p.get_fonts(full=True):
            assert doc.extract_font(font[0])[3], font
        assert any(x.get("uri") == URL for x in p.get_links())
        pix = p.get_pixmap(matrix=pymupdf.Matrix(2880 / p.rect.height, 2880 / p.rect.height), alpha=False)
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        im.save(ROOT / f"{STEM}.png", optimize=True)
    logo = Image.open(ASSETS / "supersycophantic-logo.png").convert("RGB")
    logo.thumbnail((320, 256), Image.Resampling.LANCZOS)
    logo.save(ROOT / "thumbnail.png", optimize=True)
    for name, width, height, limit in [(f"{STEM}.png", 5120, 2880, 10_000_000), ("thumbnail.png", 320, 256, 5_000_000)]:
        p = ROOT / name
        with Image.open(p) as im:
            assert im.mode == "RGB" and im.width <= width and im.height <= height
        assert p.stat().st_size < limit
    print("Verified: one-page 140 x 90 cm LaTeX PDF, embedded fonts, text bounds, code link, and web image limits.")


def main():
    style()
    for fn in [context_plot, claude_plot, certainty_plot, recheck_plot, qr]:
        fn()
    engine = shutil.which("xelatex")
    if not engine:
        raise SystemExit("Install XeLaTeX (TeX Live or MiKTeX) before building the poster.")
    with tempfile.TemporaryDirectory(prefix="supersycophantic-poster-latex-") as directory:
        command = [engine, "-interaction=nonstopmode", "-halt-on-error", "-no-shell-escape",
                   f"-output-directory={directory}", "poster.tex"]
        # TikZ page-relative positions require a second LaTeX pass.
        for _ in range(2):
            run = subprocess.run(command, cwd=ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            if run.returncode:
                print(run.stdout)
                raise SystemExit(run.returncode)
        log = (Path(directory) / "poster.log").read_text(encoding="utf-8", errors="replace")
        problems = [line for line in log.splitlines() if any(word in line for word in ["Overfull", "Missing character", "Undefined control sequence"])]
        if problems:
            raise RuntimeError("\n".join(problems))
        shutil.copy2(Path(directory) / "poster.pdf", ROOT / f"{STEM}.pdf")
    verify_and_export()


if __name__ == "__main__":
    main()
