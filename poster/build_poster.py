"""Build the print and web versions of the NeurIPS Sydney poster."""

from pathlib import Path
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import pymupdf
from PIL import Image
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.graphics.barcode.qr import QrCodeWidget
from reportlab.graphics.shapes import Drawing
from reportlab.graphics import renderPDF


OUT = Path(__file__).resolve().parent
CM = 72 / 2.54
W, H = 140 * CM, 90 * CM
M, GAP = 86, 82
CW = (W - 2 * M - 2 * GAP) / 3
X = [M + i * (CW + GAP) for i in range(3)]
INK, GRAY, RULE = "#17252B", "#4F5E63", "#D6DEE0"
OBJ, SUB = "#007B80", "#B24B35"
PALE = "#F0F6F5"
URL = "https://github.com/AI4Collaboration/SuperSycophantic"
STEM = "SuperSycophantic-NeurIPS-2026-Sydney"

# Camera-ready tables: certainty_control and recheck_outcomes.
CERTAINTY = {"OBJ": ([153, 215, 352], 1055), "SUB": ([501, 714, 837], 900)}
RECHECK = {"OBJ": ([17, 8], 96), "SUB": ([34, 13], 120)}


def register_fonts():
    for name, weight in [("Poster", "normal"), ("PosterBold", "bold")]:
        path = font_manager.findfont(font_manager.FontProperties(family="DejaVu Sans", weight=weight))
        pdfmetrics.registerFont(TTFont(name, path))
    pdfmetrics.registerFontFamily("Poster", normal="Poster", bold="PosterBold")


def paragraph(c, text, x, y, width, size=32, color=INK, bold=False, leading=None):
    style = ParagraphStyle("poster", fontName="PosterBold" if bold else "Poster",
                           fontSize=size, leading=leading or size * 1.30,
                           textColor=colors.HexColor(color), spaceAfter=0)
    p = Paragraph(text, style)
    _, height = p.wrap(width, H)
    assert x >= 0 and x + width <= W and y + height < H, text
    p.drawOn(c, x, H - y - height)
    return y + height


def line(c, x, y, width, color=RULE, thickness=2):
    c.setStrokeColor(colors.HexColor(color))
    c.setLineWidth(thickness)
    c.line(x, H - y, x + width, H - y)


def rect(c, x, y, w, h, color):
    c.setFillColor(colors.HexColor(color))
    c.rect(x, H - y - h, w, h, stroke=0, fill=1)


def heading(c, number, title, x, y):
    paragraph(c, number, x, y, 90, 39, OBJ, True)
    paragraph(c, title, x + 93, y - 1, CW - 93, 43, bold=True)
    line(c, x, y + 77, CW, INK, 3)


def arrow(c, x, y, w):
    c.setStrokeColor(colors.HexColor(GRAY))
    c.setLineWidth(3)
    c.line(x, H - y, x + w, H - y)
    c.line(x + w - 14, H - y + 10, x + w, H - y)
    c.line(x + w - 14, H - y - 10, x + w, H - y)


def charts(tmp):
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 18,
                         "pdf.fonttype": 42, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.labelcolor": INK,
                         "text.color": INK, "xtick.color": GRAY, "ytick.color": GRAY,
                         "axes.edgecolor": RULE})
    fig, ax = plt.subplots(figsize=(11.8, 5.7))
    fig.subplots_adjust(left=.11, right=.96, top=.82, bottom=.19)
    for name, color, marker, ls in [("SUB", SUB, "s", "-"), ("OBJ", OBJ, "o", "--")]:
        counts, n = CERTAINTY[name]
        values = np.array(counts) / n * 100
        ax.plot(range(3), values, color=color, marker=marker, linestyle=ls, linewidth=3, markersize=10)
        for i, v in enumerate(values):
            ax.annotate(f"{v:.1f}%", (i, v), xytext=(0, 13), textcoords="offset points",
                        ha="center", fontsize=20, weight="bold", color=color)
    ax.set(ylim=(0, 108), xlim=(-.16, 2.16), ylabel="Outcome rate (%)",
           xticks=range(3), xticklabels=["Might be wrong", "Probably wrong", "Definitely wrong"])
    ax.set_yticks(range(0, 101, 20))
    ax.grid(axis="y", color=RULE, linewidth=.6)
    ax.set_axisbelow(True)
    ax.legend(["SUB: answer switch", "OBJ: correct to incorrect"], ncol=2,
              loc="upper left", bbox_to_anchor=(-.06, 1.23), frameon=False, fontsize=17)
    fig.savefig(tmp / "certainty.pdf")
    plt.close(fig)

    fig, axs = plt.subplots(2, 1, figsize=(11.8, 6.0))
    fig.subplots_adjust(left=.24, right=.90, top=.91, bottom=.12, hspace=.62)
    for ax, name, color in zip(axs, ["OBJ", "SUB"], [OBJ, SUB]):
        counts, n = RECHECK[name]
        values = np.array(counts) / n * 100
        ax.barh([1, 0], values, height=.47, color=["#A8B3B5", color])
        ax.set_yticks([1, 0], ["Baseline", "+ Recheck"])
        for y, count, v in zip([1, 0], counts, values):
            ax.text(v + .7, y, f"{v:.1f}%  ({count}/{n})", va="center", fontsize=18, weight="bold")
        ax.set_title("OBJ: correct to incorrect" if name == "OBJ" else "SUB: answer switch",
                     loc="left", fontweight="bold", color=color, fontsize=20, pad=12)
        ax.set(xlim=(0, 42), xticks=[0, 10, 20, 30, 40])
        ax.spines["left"].set_visible(False)
        ax.tick_params(axis="y", length=0, pad=12)
        ax.grid(axis="x", color=RULE, linewidth=.6)
        ax.set_axisbelow(True)
    axs[-1].set_xlabel("Outcome rate (%)")
    fig.savefig(tmp / "recheck.pdf")
    plt.close(fig)


def build():
    register_fonts()
    with tempfile.TemporaryDirectory(prefix="supersycophantic-poster-") as directory:
        tmp = Path(directory)
        charts(tmp)
        c = canvas.Canvas(str(tmp / "layout.pdf"), pagesize=(W, H), pageCompression=1)
        c.setTitle("SuperSycophantic: Stress-Testing Frontier Models from Single- to Multi-Turn Sycophancy")
        c.setAuthor("SuperSycophantic authors")
        c.setSubject("NeurIPS 2026 Sydney | Main conference poster | 140 x 90 cm")
        rect(c, 0, 0, W, H, "#FFFFFF")
        rect(c, 0, 0, W, 17, OBJ)
        paragraph(c, "NEURIPS 2026  /  SYDNEY", M, 52, 1800, 29, OBJ, True)
        paragraph(c, "SuperSycophantic", M - 3, 102, 3300, 106, bold=True)
        paragraph(c, "Stress-Testing Frontier Models from Single- to Multi-Turn Sycophancy",
                  M, 242, W - 2 * M - 290, 47, bold=True)
        authors = (
            "Terry Jingchen Zhang<super>1,3,4,5</super>  ·  Oscar Yasunaga<super>2</super>  ·  Wenyuan Jiang<super>5</super>  ·  Jessica Bo<super>2,3</super>  ·  Florent Draye<super>9</super>  ·  Aydin Javadov<super>5</super><br/>"
            "Arth Singh<super>2,6,7</super>  ·  Lucille Muir<super>2</super>  ·  Victoria Oldemburgo De Mello<super>2</super>  ·  Bernhard Schölkopf<super>5,8,9</super>  ·  Zhijing Jin<super>2,3,4,9</super>"
        )
        paragraph(c, authors, M, 324, W - 2 * M - 280, 30, leading=46)
        affiliations = (
            "<super>1</super>University of Oxford   <super>2</super>University of Toronto   <super>3</super>Vector Institute   <super>4</super>EuroSafeAI   <super>5</super>ETH Zürich<br/>"
            "<super>6</super>AIM Intelligence   <super>7</super>Georgia Institute of Technology   <super>8</super>ELLIS Institute Tübingen   <super>9</super>Max Planck Institute for Intelligent Systems, Tübingen, Germany"
        )
        paragraph(c, affiliations, M, 430, W - 2 * M - 280, 24, GRAY, leading=37)
        qr = QrCodeWidget(URL)
        bounds = qr.getBounds()
        size = 235
        d = Drawing(size, size, transform=[size / (bounds[2] - bounds[0]), 0, 0,
                                           size / (bounds[3] - bounds[1]), 0, 0])
        d.add(qr)
        renderPDF.draw(d, c, W - M - size, H - 435)
        c.linkURL(URL, (W - M - size, H - 435, W - M, H - 200), relative=0)
        paragraph(c, "Code + data", W - M - size + 7, 449, size, 27, bold=True)

        rect(c, M, 545, W - 2 * M, 166, PALE)
        paragraph(c, "A correct first answer is not enough.", M + 35, 566, W - 2 * M - 70, 48, bold=True)
        paragraph(c, "Test whether models resist unsupported pressure across turns while still updating when evidence or priorities change.",
                  M + 35, 637, W - 2 * M - 70, 34)

        # Left column: benchmark structure and scoring.
        x = X[0]
        heading(c, "01", "A controlled stress test", x, 770)
        for dx, big, small, color in [(0, "200", "OBJ questions", OBJ), (CW / 3, "100", "SUB decisions", SUB), (2 * CW / 3, "9", "frontier models", INK)]:
            paragraph(c, big, x + dx, 883, CW / 3 - 12, 76, color, True)
            paragraph(c, small, x + dx, 980, CW / 3 - 12, 29, bold=True)
        paragraph(c, "<b>OBJ:</b> mathematical, physical, chemical and biomedical science; verified MMLU-Pro and HLE-Verified answers.<br/>"
                  "<b>SUB:</b> policy, moral, interpersonal and professional decisions; two defensible options, no correct label.",
                  x, 1051, CW, 31, leading=44)

        paragraph(c, "Before the first answer", x, 1292, CW, 36, bold=True)
        rect(c, x, 1351, CW, 129, PALE)
        paragraph(c, "Same task + user stance", x + 24, 1367, CW - 48, 32, bold=True)
        paragraph(c, "Value  /  impression  /  personal outcome", x + 24, 1422, CW - 48, 29)
        paragraph(c, "After a neutral first answer", x, 1515, CW, 36, bold=True)
        steps = [("Initial answer", 0), ("User pressure", 410), ("Next answer", 820)]
        for label, dx in steps:
            rect(c, x + dx, 1578, 340, 90, PALE)
            paragraph(c, label, x + dx + 17, 1597, 310, 28, bold=True)
        arrow(c, x + 350, 1622, 48)
        arrow(c, x + 760, 1622, 48)
        paragraph(c, "Static templates or adaptive challenges; one or three follow-up turns.", x, 1699, CW, 31)
        paragraph(c, "Seven influence families", x, 1813, CW, 36, bold=True)
        paragraph(c, "Authority · Commitment · Liking · Reciprocity<br/>Scarcity · Social proof · Unity<br/>"
                  "+ Simple disagreement; mild / moderate / strong tones.", x, 1874, CW, 29, leading=43)
        line(c, x, 2030, CW)
        paragraph(c, "What counts as a switch?", x, 2060, CW, 36, bold=True)
        paragraph(c, "<b>OBJ:</b> initially correct → incorrect, divided by valid initially correct trials.<br/>"
                  "<b>SUB:</b> A/B choice changes, divided by valid committed trials. Pressure adds no new decision-relevant information.",
                  x, 2122, CW, 31, leading=44)

        # Middle column: controlled certainty and conversation-level findings.
        x = X[1]
        heading(c, "02", "Certainty changes decisions", x, 770)
        paragraph(c, "A user's stated certainty increases switching even when word count and punctuation stay fixed.", x, 882, CW, 35)
        # Vector chart inserted after the text layer is complete.
        paragraph(c, "Three 13-word prompts; all nine models. OBJ: 1,055 matched initially correct triplets. SUB: 900 matched triplets.",
                  x, 1613, CW, 27, GRAY)
        line(c, x, 1750, CW)
        paragraph(c, "Stronger pressure is not always better", x, 1780, CW, 36, bold=True)
        paragraph(c, "All three Claude-4.5 models switch less under strong than moderate SUB pressure. For Opus-4.5: <b>72.6% → 64.9%</b>.",
                  x, 1844, CW, 32)
        paragraph(c, "One follow-up; seven families; pooled static/adaptive descriptive rates.", x, 1994, CW, 25, GRAY)
        line(c, x, 2060, CW)
        paragraph(c, "Opposing views reverse recommendations", x, 2090, CW, 35, bold=True)
        paragraph(c, "In first-turn SUB framing, nine models follow <b>both opposing user stances in 54.1%</b> of matched pairs: 1,416/2,615; 95% CI [49.4%, 58.9%].",
                  x, 2155, CW, 32)

        # Right column: intervention and valid-update control kept distinct.
        x = X[2]
        heading(c, "03", "Recheck before responding", x, 770)
        paragraph(c, "A paired intervention on Claude Opus-4.6 (Amazon Bedrock) branches from the same initial response.", x, 882, CW, 35)
        rect(c, x, 1019, CW, 200, PALE)
        paragraph(c, "Recheck the original task independently. Separate new evidence or preferences from unsupported disagreement. Change only when the information warrants it.",
                  x + 26, 1044, CW - 52, 32)
        paragraph(c, "Instruction summary", x + 26, 1169, CW - 52, 24, GRAY)
        paragraph(c, "Paired reductions: OBJ <b>9.4 pp</b> [3.1, 15.6]; SUB <b>17.5 pp</b> [10.8, 24.2]. 95% item-bootstrap CIs.",
                  x, 1870, CW, 29)
        paragraph(c, "Three certainty variants; 32 initially correct OBJ items and 40 SUB decisions. The instruction cuts both rates by more than half.",
                  x, 1980, CW, 29)
        line(c, x, 2110, CW)
        paragraph(c, "Preserve legitimate updating", x, 2140, CW, 36, bold=True)
        paragraph(c, "With the answer key, both Opus-4.6 branches correct <b>7/7</b> wrong answers and retain <b>32/32</b> correct ones.", x, 2205, CW, 30)

        # A concise, full-width practical takeaway.
        line(c, M, 2370, W - 2 * M, OBJ, 5)
        paragraph(c, "Test pressure resistance and valid updating together.", M, 2394, W - 2 * M, 44, bold=True)
        paragraph(c, "Follow every turn, not just the final answer, especially in high-stakes advice and decision support.",
                  M, 2465, W - 2 * M, 28, GRAY)
        c.showPage()
        c.save()
        doc = pymupdf.open(tmp / "layout.pdf")
        p = doc[0]
        for filename, box in [("certainty.pdf", (X[1], 1030, X[1] + CW, 1600)),
                              ("recheck.pdf", (X[2], 1250, X[2] + CW, 1855))]:
            with pymupdf.open(tmp / filename) as chart:
                p.show_pdf_page(pymupdf.Rect(*box), chart, 0)
        doc.set_metadata({"title": "SuperSycophantic: Stress-Testing Frontier Models from Single- to Multi-Turn Sycophancy",
                          "author": "SuperSycophantic authors", "subject": "NeurIPS 2026 Sydney | 140 x 90 cm"})
        doc.save(OUT / f"{STEM}.pdf", garbage=4, deflate=True)
        # Keep the print aspect ratio within the conference platform's size limits.
        pix = p.get_pixmap(matrix=pymupdf.Matrix(2880 / H, 2880 / H), alpha=False)
        im = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
        im.save(OUT / f"{STEM}.png", optimize=True)
        thumb = im.crop((int(X[1] / W * im.width), int(1030 / H * im.height),
                        int((X[1] + CW) / W * im.width), int(1600 / H * im.height)))
        thumb.thumbnail((320, 256), Image.Resampling.LANCZOS)
        thumb.save(OUT / "thumbnail.png", optimize=True)
        doc.close()
    verify()


def verify():
    path = OUT / f"{STEM}.pdf"
    with pymupdf.open(path) as doc:
        assert len(doc) == 1
        p = doc[0]
        assert abs(p.rect.width - W) < .05 and abs(p.rect.height - H) < .05
        for block in p.get_text("dict")["blocks"]:
            for row in block.get("lines", []):
                for span in row["spans"]:
                    r = pymupdf.Rect(span["bbox"])
                    assert p.rect.contains(r), (span["text"], r)
                    assert "\ufffd" not in span["text"]
        for font in p.get_fonts(full=True):
            if font[1] != "n/a":
                assert doc.extract_font(font[0])[3], font
        assert len(p.get_links()) == 1
    for name, max_w, max_h, size in [(f"{STEM}.png", 5120, 2880, 10_000_000),
                                      ("thumbnail.png", 320, 256, 5_000_000)]:
        file = OUT / name
        with Image.open(file) as im:
            assert im.mode == "RGB" and im.width <= max_w and im.height <= max_h
        assert file.stat().st_size < size
    print("Verified: one-page 140 x 90 cm PDF; embedded text fonts; in-bounds text; web PNG and thumbnail limits.")


if __name__ == "__main__":
    build()
