import argparse
import json
import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]

MODEL_ORDER = [
    "openai/gpt-5.4",
    "openai/gpt-5.4-mini",
    "openai/gpt-5.4-nano",
    "anthropic/claude-opus-4.5",
    "anthropic/claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5",
    "google/gemini-3.1-flash-lite-preview",
    "mistralai/mistral-medium-3.1",
    "cohere/command-r-08-2024",
]

SHORT_LABELS = {
    "openai/gpt-5.4": "GPT-5.4",
    "openai/gpt-5.4-mini": "GPT-5.4\nmini",
    "openai/gpt-5.4-nano": "GPT-5.4\nnano",
    "anthropic/claude-opus-4.5": "Opus 4.5",
    "anthropic/claude-sonnet-4.5": "Sonnet 4.5",
    "anthropic/claude-haiku-4.5": "Haiku 4.5",
    "google/gemini-3.1-flash-lite-preview": "Gemini\nFlash",
    "mistralai/mistral-medium-3.1": "Mistral",
    "cohere/command-r-08-2024": "Command R",
}

COLUMN_LABELS = {
    "openai/gpt-5.4": "GPT-5.4",
    "openai/gpt-5.4-mini": "5.4\nmini",
    "openai/gpt-5.4-nano": "5.4\nnano",
    "anthropic/claude-opus-4.5": "Opus",
    "anthropic/claude-sonnet-4.5": "Sonnet",
    "anthropic/claude-haiku-4.5": "Haiku",
    "google/gemini-3.1-flash-lite-preview": "Gemini",
    "mistralai/mistral-medium-3.1": "Mistral",
    "cohere/command-r-08-2024": "Command\nR",
}

PROVIDER = {
    "openai/gpt-5.4": "openai",
    "openai/gpt-5.4-mini": "openai",
    "openai/gpt-5.4-nano": "openai",
    "anthropic/claude-opus-4.5": "anthropic",
    "anthropic/claude-sonnet-4.5": "anthropic",
    "anthropic/claude-haiku-4.5": "anthropic",
    "google/gemini-3.1-flash-lite-preview": "google",
    "mistralai/mistral-medium-3.1": "mistralai",
    "cohere/command-r-08-2024": "cohere",
}

LOGO_FILES = {
    "openai": REPO_ROOT / "images" / "logos" / "openai.png",
    "anthropic": REPO_ROOT / "images" / "logos" / "claude.ico",
    "google": REPO_ROOT / "images" / "logos" / "gemini.png",
    "mistralai": REPO_ROOT / "images" / "logos" / "mistral.png",
    "cohere": REPO_ROOT / "images" / "logos" / "cohere.png",
}

PROVIDER_COLORS = {
    "openai": "#6E59C9",
    "anthropic": "#C86A45",
    "google": "#3B78E7",
    "mistralai": "#D45A2A",
    "cohere": "#2F6659",
}

BADGE_PX = {
    "openai/gpt-5.4": 150,
    "openai/gpt-5.4-mini": 136,
    "openai/gpt-5.4-nano": 122,
    "anthropic/claude-opus-4.5": 148,
    "anthropic/claude-sonnet-4.5": 138,
    "anthropic/claude-haiku-4.5": 122,
    "google/gemini-3.1-flash-lite-preview": 126,
    "mistralai/mistral-medium-3.1": 132,
    "cohere/command-r-08-2024": 132,
}

LOGO_FILL = {
    "openai/gpt-5.4": 0.76,
    "openai/gpt-5.4-mini": 0.67,
    "openai/gpt-5.4-nano": 0.58,
    "anthropic/claude-opus-4.5": 0.76,
    "anthropic/claude-sonnet-4.5": 0.69,
    "anthropic/claude-haiku-4.5": 0.58,
    "google/gemini-3.1-flash-lite-preview": 0.60,
    "mistralai/mistral-medium-3.1": 0.68,
    "cohere/command-r-08-2024": 0.66,
}

# Offsets move large labels away from dense metric clusters. Thin hairlines keep
# the plotted metric anchor unambiguous while avoiding overlapping logos.
SCATTER_OFFSETS = {
    "openai/gpt-5.4": (-210, -10),
    "anthropic/claude-opus-4.5": (-210, 160),
    "anthropic/claude-sonnet-4.5": (158, 92),
    "anthropic/claude-haiku-4.5": (188, 70),
    "google/gemini-3.1-flash-lite-preview": (-20, -170),
    "mistralai/mistral-medium-3.1": (240, 10),
    "openai/gpt-5.4-mini": (210, -154),
    "openai/gpt-5.4-nano": (118, -58),
    "cohere/command-r-08-2024": (-102, -122),
}

INK = "#1D2733"
MUTED = "#596879"
GRID = "#DDE5ED"
PANEL_BG = "#F7FAFC"
WHITE = "#FFFFFF"


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    font_dir = Path("C:/Windows/Fonts")
    candidates = {
        "regular": ["arial.ttf", "segoeui.ttf"],
        "bold": ["arialbd.ttf", "segoeuib.ttf"],
        "italic": ["ariali.ttf", "segoeuii.ttf"],
    }[name]
    for candidate in candidates:
        path = font_dir / candidate
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


F_TITLE = font("bold", 64)
F_AXIS = font("bold", 48)
F_TICK = font("regular", 38)
F_MODEL = font("bold", 38)
F_HEAT_COL = font("regular", 41)
F_HEAT_ROW = font("bold", 45)
F_CELL = font("bold", 54)
F_SHIFT_TITLE = font("bold", 60)
F_SHIFT_SUB = font("regular", 36)
F_SHIFT_LABEL = font("bold", 36)
F_SHIFT_TICK = font("regular", 34)
F_SHIFT_LEGEND = font("regular", 34)


def trim_alpha(img: Image.Image) -> Image.Image:
    img = img.convert("RGBA")
    alpha = img.getchannel("A")
    bbox = alpha.getbbox()
    return img.crop(bbox) if bbox else img


def trim_white(img: Image.Image, padding: int = 28) -> Image.Image:
    rgb = img.convert("RGB")
    pix = rgb.load()
    width, height = rgb.size

    def nonwhite_pixel(x: int, y: int) -> bool:
        return pix[x, y] != (255, 255, 255)

    top = next((y for y in range(height) if any(nonwhite_pixel(x, y) for x in range(width))), 0)
    bottom = next((y for y in range(height - 1, -1, -1) if any(nonwhite_pixel(x, y) for x in range(width))), height - 1)
    left = next((x for x in range(width) if any(nonwhite_pixel(x, y) for y in range(height))), 0)
    right = next((x for x in range(width - 1, -1, -1) if any(nonwhite_pixel(x, y) for y in range(height))), width - 1)
    crop = (
        max(0, left - padding),
        max(0, top - padding),
        min(width, right + padding + 1),
        min(height, bottom + padding + 1),
    )
    return rgb.crop(crop)


def load_provider_logo(provider: str) -> Image.Image:
    path = LOGO_FILES[provider]
    img = Image.open(path)
    if provider == "anthropic" and getattr(img, "n_frames", 1) > 1:
        best = img.copy()
        for frame in range(img.n_frames):
            img.seek(frame)
            candidate = img.copy()
            if candidate.width * candidate.height > best.width * best.height:
                best = candidate
        img = best
    return trim_alpha(img)


def fit_image(img: Image.Image, max_side: int) -> Image.Image:
    ratio = min(max_side / img.width, max_side / img.height)
    size = (max(1, round(img.width * ratio)), max(1, round(img.height * ratio)))
    return img.resize(size, Image.Resampling.LANCZOS)


def make_badge(model: str) -> Image.Image:
    size = BADGE_PX[model]
    provider = PROVIDER[model]
    badge = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(badge)
    radius = max(16, size // 7)
    draw.rounded_rectangle(
        [3, 3, size - 4, size - 4],
        radius=radius,
        fill=WHITE,
        outline="#D7DFE8",
        width=3,
    )
    draw.rounded_rectangle(
        [6, 6, size - 7, size - 7],
        radius=max(12, radius - 4),
        outline=PROVIDER_COLORS[provider],
        width=3,
    )
    logo = load_provider_logo(provider)
    logo = fit_image(logo, round(size * LOGO_FILL[model]))
    badge.alpha_composite(logo, ((size - logo.width) // 2, (size - logo.height) // 2))
    return badge


def text_size(draw: ImageDraw.ImageDraw, text: str, fnt: ImageFont.FreeTypeFont) -> tuple[int, int]:
    lines = text.split("\n")
    widths = []
    heights = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        widths.append(bbox[2] - bbox[0])
        heights.append(bbox[3] - bbox[1])
    return max(widths) if widths else 0, sum(heights) + max(0, len(lines) - 1) * 6


def draw_centered_text(
    draw: ImageDraw.ImageDraw,
    center_x: int,
    y: int,
    text: str,
    fnt: ImageFont.FreeTypeFont,
    fill: str = INK,
) -> tuple[int, int, int, int]:
    lines = text.split("\n")
    line_heights = []
    widths = []
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=fnt)
        widths.append(bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    cursor = y
    for line, width, height in zip(lines, widths, line_heights):
        draw.text((center_x - width / 2, cursor), line, font=fnt, fill=fill)
        cursor += height + 6
    return (center_x - max(widths) // 2, y, center_x + max(widths) // 2, cursor)


def lerp_color(a: str, b: str, t: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, t))
    av = tuple(int(a[i : i + 2], 16) for i in (1, 3, 5))
    bv = tuple(int(b[i : i + 2], 16) for i in (1, 3, 5))
    return tuple(round(x + (y - x) * t) for x, y in zip(av, bv))


def rounded_cell(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], fill: tuple[int, int, int]) -> None:
    draw.rounded_rectangle(box, radius=17, fill=fill)


def metric_bundle(summary: dict) -> dict[str, dict[str, float]]:
    out = {}
    for model, row in summary["models"].items():
        out[model] = {
            "gt_truth_departure": row["gt"]["correct_to_incorrect_rate"] * 100,
            "ngt_user_lift": row["ngt"]["all"]["user_answer_agreement"]["lift"] * 100,
        }
    return out


def heatmap_values(summary: dict) -> list[tuple[str, str, list[float]]]:
    rows = [
        ("GT Belief", "gt", "value_relevant"),
        ("GT Identity", "gt", "impression_relevant"),
        ("GT Stake", "gt", "outcome_relevant"),
        ("NGT Belief", "ngt", "value_relevant"),
        ("NGT Identity", "ngt", "impression_relevant"),
        ("NGT Stake", "ngt", "outcome_relevant"),
    ]
    data = []
    for label, branch, cue in rows:
        vals = []
        for model in MODEL_ORDER:
            row = summary["models"][model]
            if branch == "gt":
                vals.append(row["gt"]["by_cue"][cue]["correct_to_incorrect_rate"] * 100)
            else:
                vals.append(row["ngt"]["by_cue"][cue]["user_answer_agreement"]["lift"] * 100)
        data.append((label, branch, vals))
    return data


def draw_axes(draw: ImageDraw.ImageDraw, rect: tuple[int, int, int, int]) -> None:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle([x0 - 34, y0 - 70, x1 + 30, y1 + 96], radius=28, fill=PANEL_BG)
    for x_tick in range(10, 80, 10):
        x = x0 + (x_tick - 8) / (76 - 8) * (x1 - x0)
        draw.line([(x, y0), (x, y1)], fill=GRID, width=2)
        label = f"{x_tick}%"
        bbox = draw.textbbox((0, 0), label, font=F_TICK)
        draw.text((x - (bbox[2] - bbox[0]) / 2, y1 + 22), label, font=F_TICK, fill=MUTED)
    for y_tick in range(10, 45, 5):
        y = y1 - (y_tick - 8) / (40 - 8) * (y1 - y0)
        draw.line([(x0, y), (x1, y)], fill=GRID, width=2)
        label = f"{y_tick}%"
        bbox = draw.textbbox((0, 0), label, font=F_TICK)
        draw.text((x0 - 26 - (bbox[2] - bbox[0]), y - 20), label, font=F_TICK, fill=MUTED)
    draw.line([(x0, y1), (x1, y1)], fill="#758294", width=4)
    draw.line([(x0, y0), (x0, y1)], fill="#758294", width=4)
    x_label = "GT Truth Departure (%)"
    bbox = draw.textbbox((0, 0), x_label, font=F_AXIS)
    draw.text((x0 + (x1 - x0 - (bbox[2] - bbox[0])) / 2, y1 + 74), x_label, font=F_AXIS, fill=INK)
    y_label = "NGT User-View Lift (%)"
    y_img = Image.new("RGBA", (560, 76), (0, 0, 0, 0))
    yd = ImageDraw.Draw(y_img)
    yd.text((0, 0), y_label, font=F_AXIS, fill=INK)
    y_img = y_img.rotate(90, expand=True)
    draw.bitmap((x0 - 168, y0 + (y1 - y0 - y_img.height) // 2), y_img, fill=None)


def draw_scatter(
    base: Image.Image,
    draw: ImageDraw.ImageDraw,
    summary: dict,
    rect: tuple[int, int, int, int],
) -> list[tuple[str, tuple[int, int, int, int]]]:
    x0, y0, x1, y1 = rect
    draw_axes(draw, rect)
    title = "Model-level Context Susceptibility"
    bbox = draw.textbbox((0, 0), title, font=F_TITLE)
    draw.text((x0 + (x1 - x0 - (bbox[2] - bbox[0])) / 2, y0 - 62), title, font=F_TITLE, fill=INK)
    metrics = metric_bundle(summary)
    bboxes = []

    def x_map(v: float) -> float:
        return x0 + (v - 8) / (76 - 8) * (x1 - x0)

    def y_map(v: float) -> float:
        return y1 - (v - 8) / (40 - 8) * (y1 - y0)

    for model in MODEL_ORDER:
        mx = x_map(metrics[model]["gt_truth_departure"])
        my = y_map(metrics[model]["ngt_user_lift"])
        dx, dy = SCATTER_OFFSETS[model]
        cx = int(max(x0 + 84, min(x1 - 84, mx + dx)))
        cy = int(max(y0 + 88, min(y1 - 50, my + dy)))
        badge = make_badge(model)
        label = SHORT_LABELS[model]
        label_w, label_h = text_size(draw, label, F_MODEL)
        total_w = max(badge.width, label_w)
        total_h = badge.height + 12 + label_h
        left = cx - total_w // 2
        top = cy - total_h // 2
        right = cx + math.ceil(total_w / 2)
        bottom = top + total_h

        if abs(cx - mx) > 24 or abs(cy - my) > 24:
            draw.line([(mx, my), (cx, top + badge.height / 2)], fill="#B5C0CE", width=2)

        base.alpha_composite(badge, (int(cx - badge.width / 2), int(top)))
        draw_centered_text(draw, cx, int(top + badge.height + 12), label, F_MODEL, fill=INK)
        bboxes.append((model, (int(left), int(top), int(right), int(bottom))))
    return bboxes


def draw_heatmap(
    draw: ImageDraw.ImageDraw,
    summary: dict,
    rect: tuple[int, int, int, int],
) -> list[tuple[str, tuple[int, int, int, int]]]:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle([x0 - 34, y0 - 88, x1 + 30, y1 + 52], radius=28, fill=PANEL_BG)
    title = "Framing Effect by Branch"
    bbox = draw.textbbox((0, 0), title, font=F_TITLE)
    draw.text((x0 + (x1 - x0 - (bbox[2] - bbox[0])) / 2, y0 - 70), title, font=F_TITLE, fill=INK)

    row_label_w = 250
    top_label_h = 110
    gap = 12
    n_cols = len(MODEL_ORDER)
    n_rows = 6
    cell_w = (x1 - x0 - row_label_w - (n_cols - 1) * gap) // n_cols
    cell_h = (y1 - y0 - top_label_h - (n_rows - 1) * gap) // n_rows
    grid_x = x0 + row_label_w
    grid_y = y0 + top_label_h

    bboxes = []
    for c, model in enumerate(MODEL_ORDER):
        label = COLUMN_LABELS[model]
        cx = grid_x + c * (cell_w + gap) + cell_w // 2
        draw_centered_text(draw, cx, y0 + 6, label, F_HEAT_COL, fill=INK)

    for r, (label, branch, vals) in enumerate(heatmap_values(summary)):
        y = grid_y + r * (cell_h + gap)
        row_color = "#A53228" if branch == "gt" else "#247A5B"
        row_bbox = draw.textbbox((0, 0), label, font=F_HEAT_ROW)
        draw.text((x0 + row_label_w - 26 - (row_bbox[2] - row_bbox[0]), y + cell_h / 2 - 28), label, font=F_HEAT_ROW, fill=row_color)
        max_val = 80 if branch == "gt" else 40
        for c, val in enumerate(vals):
            x = grid_x + c * (cell_w + gap)
            if val < 0:
                fill = lerp_color("#F8FAFF", "#9CB7E8", min(1.0, abs(val) / 10))
            elif branch == "gt":
                fill = lerp_color("#FFF4F0", "#CA3E32", val / max_val)
            else:
                fill = lerp_color("#EFFAF3", "#30B36B", val / max_val)
            rounded_cell(draw, (x, y, x + cell_w, y + cell_h), fill)
            label_text = f"{round(val):.0f}"
            tb = draw.textbbox((0, 0), label_text, font=F_CELL)
            tw, th = tb[2] - tb[0], tb[3] - tb[1]
            draw.text((x + (cell_w - tw) / 2, y + (cell_h - th) / 2 - 4), label_text, font=F_CELL, fill=INK)
            bboxes.append((f"{label}:{model}", (x, y, x + cell_w, y + cell_h)))
    return bboxes


def draw_shift_panel(
    draw: ImageDraw.ImageDraw,
    summary: dict,
    rect: tuple[int, int, int, int],
    title: str,
    branch: str,
    x_min: float,
    x_max: float,
) -> list[tuple[str, tuple[int, int, int, int]]]:
    x0, y0, x1, y1 = rect
    draw.rounded_rectangle([x0 - 18, y0 - 22, x1 + 18, y1 + 66], radius=26, fill=PANEL_BG)
    title_box = draw.textbbox((0, 0), title, font=F_SHIFT_TITLE)
    draw.text((x0 + (x1 - x0 - (title_box[2] - title_box[0])) / 2, y0 + 18), title, font=F_SHIFT_TITLE, fill=INK)

    plot_left = x0 + 230
    plot_right = x1 - 34
    plot_top = y0 + 96
    row_gap = (y1 - plot_top - 80) / (len(MODEL_ORDER) - 1)

    def x_map(v: float) -> float:
        return plot_left + (v - x_min) / (x_max - x_min) * (plot_right - plot_left)

    for tick in range(math.ceil(x_min / 10) * 10, math.floor(x_max / 10) * 10 + 1, 10):
        x = x_map(tick)
        draw.line([(x, plot_top - 24), (x, y1 - 18)], fill=GRID, width=2)
        label = f"{tick}%"
        box = draw.textbbox((0, 0), label, font=F_SHIFT_TICK)
        draw.text((x - (box[2] - box[0]) / 2, y1 + 20), label, font=F_SHIFT_TICK, fill=MUTED)

    boxes = []
    for i, model in enumerate(MODEL_ORDER):
        y = plot_top + i * row_gap
        label = SHORT_LABELS[model].replace("\n", " ")
        label_box = draw.textbbox((0, 0), label, font=F_SHIFT_LABEL)
        draw.text((x0 + 8, y - 20), label, font=F_SHIFT_LABEL, fill=INK)

        row = summary["models"][model]
        if branch == "gt":
            neutral = row["gt"]["neutral_accuracy_rate"] * 100
            framed = row["gt"]["framed_accuracy_rate"] * 100
            neutral_color = "#4C78A8"
            framed_color = "#CB4B47"
        else:
            neutral = row["ngt"]["all"]["user_answer_agreement"]["neutral_rate"] * 100
            framed = row["ngt"]["all"]["user_answer_agreement"]["framed_rate"] * 100
            neutral_color = "#4C78A8"
            framed_color = "#2BAE66"

        xn = x_map(neutral)
        xf = x_map(framed)
        draw.line([(xn, y), (xf, y)], fill="#9EAABD", width=8)
        direction = 1 if xf >= xn else -1
        arrow = [
            (xf, y),
            (xf - direction * 18, y - 11),
            (xf - direction * 18, y + 11),
        ]
        draw.polygon(arrow, fill="#9EAABD")
        r = 15
        draw.ellipse([xn - r, y - r, xn + r, y + r], fill=neutral_color, outline=WHITE, width=4)
        draw.ellipse([xf - r, y - r, xf + r, y + r], fill=framed_color, outline=WHITE, width=4)

        if branch == "gt":
            delta = framed - neutral
            delta_text = f"{delta:.0f}"
        else:
            delta = framed - neutral
            delta_text = f"+{delta:.0f}"
        delta_box = draw.textbbox((0, 0), delta_text, font=F_SHIFT_LABEL)
        tx = min(plot_right - (delta_box[2] - delta_box[0]), max(plot_left, xf + 26))
        if branch == "gt" and xf < xn:
            tx = max(plot_left, xf - 34 - (delta_box[2] - delta_box[0]))
        draw.text((tx, y - 20), delta_text, font=F_SHIFT_LABEL, fill=framed_color)
        boxes.append((f"{branch}:{model}", (x0, int(y - 24), x1, int(y + 24))))

    draw.line([(plot_left, y1 - 18), (plot_right, y1 - 18)], fill="#758294", width=4)
    return boxes


def save_shift_plot(summary: dict, out_png: Path, out_pdf: Path) -> None:
    width, height = 3300, 1190
    img = Image.new("RGBA", (width, height), WHITE)
    draw = ImageDraw.Draw(img)

    left_boxes = draw_shift_panel(
        draw,
        summary,
        (120, 58, 1575, 1026),
        "GT",
        "gt",
        0,
        75,
    )
    right_boxes = draw_shift_panel(
        draw,
        summary,
        (1715, 58, 3170, 1026),
        "NGT",
        "ngt",
        45,
        90,
    )
    assert_no_overlap("shift-left", left_boxes)
    assert_no_overlap("shift-right", right_boxes)

    legend_y = 1120
    legend_x = 1210
    draw.ellipse([legend_x, legend_y - 14, legend_x + 28, legend_y + 14], fill="#4C78A8", outline=WHITE, width=3)
    draw.text((legend_x + 42, legend_y - 20), "Neutral", font=F_SHIFT_LEGEND, fill=INK)
    legend_x += 230
    draw.ellipse([legend_x, legend_y - 14, legend_x + 28, legend_y + 14], fill="#CB4B47", outline=WHITE, width=3)
    draw.text((legend_x + 42, legend_y - 20), "GT framed", font=F_SHIFT_LEGEND, fill=INK)
    legend_x += 270
    draw.ellipse([legend_x, legend_y - 14, legend_x + 28, legend_y + 14], fill="#2BAE66", outline=WHITE, width=3)
    draw.text((legend_x + 42, legend_y - 20), "NGT framed", font=F_SHIFT_LEGEND, fill=INK)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    rgb = trim_white(img, padding=8)
    rgb.save(out_png, dpi=(360, 360), optimize=True)
    rgb.save(out_pdf, resolution=360)


def overlaps(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 8) -> bool:
    return not (a[2] + pad <= b[0] or b[2] + pad <= a[0] or a[3] + pad <= b[1] or b[3] + pad <= a[1])


def assert_no_overlap(name: str, boxes: list[tuple[str, tuple[int, int, int, int]]]) -> None:
    bad = []
    for i, (ai, ab) in enumerate(boxes):
        for bi, bb in boxes[i + 1 :]:
            if overlaps(ab, bb):
                bad.append((ai, bi))
    if bad:
        pairs = ", ".join(f"{a} vs {b}" for a, b in bad[:8])
        raise RuntimeError(f"{name} overlap check failed: {pairs}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--summary",
        default=REPO_ROOT / "Experimental" / "results" / "context_20260504_184050_context_main_summary.json",
        type=Path,
    )
    parser.add_argument("--out-png", default=REPO_ROOT / "images" / "results" / "context_results.png", type=Path)
    parser.add_argument("--out-pdf", default=REPO_ROOT / "images" / "results" / "context_results.pdf", type=Path)
    parser.add_argument(
        "--out-shift-png",
        default=REPO_ROOT / "images" / "results" / "context_neutral_shift.png",
        type=Path,
    )
    parser.add_argument(
        "--out-shift-pdf",
        default=REPO_ROOT / "images" / "results" / "context_neutral_shift.pdf",
        type=Path,
    )
    args = parser.parse_args()

    summary = json.loads(args.summary.read_text(encoding="utf-8"))
    missing = [model for model in MODEL_ORDER if model not in summary["models"]]
    if missing:
        raise RuntimeError(f"Summary missing models: {missing}")

    width, height = 3900, 1500
    img = Image.new("RGBA", (width, height), WHITE)
    draw = ImageDraw.Draw(img)
    scatter_boxes = draw_scatter(img, draw, summary, (300, 240, 1640, 1300))
    heat_boxes = draw_heatmap(draw, summary, (1785, 250, 3810, 1312))
    assert_no_overlap("scatter", scatter_boxes)
    assert_no_overlap("heatmap", heat_boxes)

    args.out_png.parent.mkdir(parents=True, exist_ok=True)
    rgb = trim_white(img, padding=26)
    rgb.save(args.out_png, dpi=(360, 360), optimize=True)
    rgb.save(args.out_pdf, resolution=360)
    save_shift_plot(summary, args.out_shift_png, args.out_shift_pdf)
    print(f"Wrote {args.out_png}")
    print(f"Wrote {args.out_pdf}")
    print(f"Wrote {args.out_shift_png}")
    print(f"Wrote {args.out_shift_pdf}")
    print(f"Overlap checks passed: scatter={len(scatter_boxes)} labels, heatmap={len(heat_boxes)} cells")


if __name__ == "__main__":
    main()
