#!/usr/bin/env python3
"""Draw the single-model results placeholder for the manuscript."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPO_ROOT = Path(__file__).resolve().parents[1]
CLAUDE = "anthropic/claude-sonnet-4.6"

ANCHORED_ADAPTIVE_PATH = REPO_ROOT / "Experimental/results/hle20_adaptive_trigger_gpt54mini_anchor_meta_nolabel_strong_unsupported_claude_20260428.jsonl"
REPETITION_PATH = REPO_ROOT / "Experimental/context_pilot/results/hle20_claude_temporal_repetition_strong3_20260428.jsonl"
ESCALATION_PATH = REPO_ROOT / "Experimental/context_pilot/results/hle20_claude_temporal_escalation_mild_moderate_strong_20260428.jsonl"
HETERO_PATH = REPO_ROOT / "Experimental/context_pilot/results/hle20_claude_temporal_hetero_strong3_20260428.jsonl"

PNG_OUT = REPO_ROOT / "images/result_trigger_temporal_claude_anchored_adaptive_placeholder.png"
PDF_OUT = REPO_ROOT / "images/result_trigger_temporal_claude_anchored_adaptive_placeholder.pdf"


def read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rate(rows: list[dict]) -> tuple[int, int, float, tuple[float, float]]:
    eligible = [
        row
        for row in rows
        if row.get("initial_answer") and row.get("pressure_target") and row.get("initial_answer") != row.get("pressure_target")
    ]
    n = len(eligible)
    switches = sum(bool(row.get("pressure_aligned_switch") or row.get("sycophantic_switch")) for row in eligible)
    pct = 100 * switches / n if n else 0.0
    lo, hi = wilson(switches, n)
    return switches, n, pct, (100 * lo, 100 * hi)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n) / denom
    return max(0.0, center - half), min(1.0, center + half)


def names_replacement_label(text: str) -> bool:
    return bool(
        re.search(
            r"\b(?:switch|change|correct|fix|go|give|choose|pick|land|lock|stick|move)\s+(?:back\s+)?(?:in\s+)?(?:to|with|on)?\s*(?:option|choice|answer|label)?\s*[AB]\b",
            text,
        )
        or re.search(r"\b(?:option|choice|answer|label)\s+[AB]\b", text)
        or re.search(r"\b[AB]\s+(?:is|looks|seems|feels)\s+(?:right|better|correct|cleaner|wrong)\b", text)
    )


def collect_bars() -> list[dict]:
    trigger_order = [
        ("simple_baseline", "Baseline"),
        ("authority", "Authority"),
        ("social_proof", "Social\nproof"),
        ("consistency", "Consistency"),
        ("reciprocity", "Reciprocity"),
        ("liking", "Liking"),
        ("scarcity", "Scarcity"),
        ("unity", "Unity"),
    ]
    colors = {
        "simple_baseline": "#8E9097",
        "authority": "#2F80ED",
        "social_proof": "#27AE60",
        "consistency": "#F2994A",
        "reciprocity": "#9B51E0",
        "liking": "#EB5757",
        "scarcity": "#D9A404",
        "unity": "#00A6A6",
    }
    bars: list[dict] = []
    adaptive = read_jsonl(ANCHORED_ADAPTIVE_PATH)
    for trigger, label in trigger_order:
        rows = [
            row
            for row in adaptive
            if row.get("model") == CLAUDE
            and row.get("tone") == "strong"
            and row.get("trigger") == trigger
            and not names_replacement_label(str(row.get("generated_trigger_text", "")))
        ]
        switches, n, pct, ci = rate(rows)
        bars.append(
            {
                "label": label,
                "group": "Single trigger",
                "switches": switches,
                "n": n,
                "pct": pct,
                "ci": ci,
                "color": colors[trigger],
            }
        )

    temporal_specs = [
        ("Repetition", REPETITION_PATH, "#4C78A8"),
        ("Escalation", ESCALATION_PATH, "#72B7B2"),
        ("Hetero\ncombo", HETERO_PATH, "#B279A2"),
    ]
    for label, path, color in temporal_specs:
        switches, n, pct, ci = rate([row for row in read_jsonl(path) if row.get("model") == CLAUDE])
        bars.append(
            {
                "label": label,
                "group": "Temporal",
                "switches": switches,
                "n": n,
                "pct": pct,
                "ci": ci,
                "color": color,
            }
        )

    return bars


def placeholder_bars(reference: list[dict]) -> list[dict]:
    bars: list[dict] = []
    for bar in reference:
        item = dict(bar)
        item["switches"] = None
        item["n"] = None
        item["pct"] = 0.0
        item["placeholder"] = True
        bars.append(item)
    return bars


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    name = "arialbd.ttf" if bold else "arial.ttf"
    return ImageFont.truetype(str(Path("C:/Windows/Fonts") / name), size=size)


def draw_centered(draw: ImageDraw.ImageDraw, xy: tuple[float, float], text: str, fnt: ImageFont.FreeTypeFont, fill: str) -> None:
    lines = text.split("\n")
    line_heights = [draw.textbbox((0, 0), line, font=fnt)[3] for line in lines]
    total_h = sum(line_heights) + (len(lines) - 1) * 7
    y = xy[1] - total_h / 2
    for line, h in zip(lines, line_heights):
        bbox = draw.textbbox((0, 0), line, font=fnt)
        draw.text((xy[0] - (bbox[2] - bbox[0]) / 2, y), line, font=fnt, fill=fill)
        y += h + 7


def main() -> None:
    claude_bars = collect_bars()
    model_clusters = [
        ("GPT-5.5", placeholder_bars(claude_bars)),
        ("Gemini-3.1-Pro", placeholder_bars(claude_bars)),
        ("Opus-4.7", claude_bars),
        ("DeepSeek-V4", placeholder_bars(claude_bars)),
        ("Kimi K-2.6", placeholder_bars(claude_bars)),
    ]
    bars = claude_bars
    width, height = 1500, 520
    margin_left, plot_right = 95, 1185
    legend_left = 1225
    plot_top, plot_bottom = 70, 408
    axis_bottom = plot_bottom
    max_y = 90

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)

    axis_font = font(22, True)
    tick_font = font(18)
    small_font = font(15)
    legend_font = font(17, True)
    legend_small = font(14)

    def ycoord(value: float) -> float:
        return axis_bottom - (value / max_y) * (plot_bottom - plot_top)

    for y in range(0, max_y + 1, 15):
        yy = ycoord(y)
        line_color = "#E7E9EE" if y else "#222222"
        draw.line((margin_left, yy, plot_right, yy), fill=line_color, width=2 if y else 3)
        bbox = draw.textbbox((0, 0), str(y), font=tick_font)
        draw.text((margin_left - 22 - (bbox[2] - bbox[0]), yy - 12), str(y), font=tick_font, fill="#333333")

    axis_label = "Pressure-aligned switch rate (%)"
    label_bbox = draw.textbbox((0, 0), axis_label, font=axis_font)
    label_img = Image.new("RGBA", (label_bbox[2] + 16, label_bbox[3] + 16), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_img)
    label_draw.text((8, 8), axis_label, font=axis_font, fill="#222222")
    rotated = label_img.rotate(90, expand=True)
    img.paste(rotated, (18, int((plot_top + plot_bottom) / 2 - rotated.height / 2)), rotated)

    gap_after = {7: 8, 10: 8}
    base_gap = 2
    bar_w = 9
    cluster_gap = 48
    cluster_width = len(bars) * bar_w + (len(bars) - 1) * base_gap + sum(gap_after.values())
    total_clusters_width = len(model_clusters) * cluster_width + (len(model_clusters) - 1) * cluster_gap
    x = margin_left + (plot_right - margin_left - total_clusters_width) / 2

    for model_name, cluster_bars in model_clusters:
        xs: list[float] = []
        local_x = x
        for i in range(len(cluster_bars)):
            xs.append(local_x + bar_w / 2)
            local_x += bar_w + base_gap + gap_after.get(i, 0)
        for i, bar in enumerate(cluster_bars):
            cx = xs[i]
            x0 = cx - bar_w / 2
            x1 = cx + bar_w / 2
            if bar.get("placeholder"):
                y0 = ycoord(4)
                draw.rectangle((x0, y0, x1, axis_bottom), fill="#D7DAE2")
            else:
                y0 = ycoord(bar["pct"])
                draw.rectangle((x0, y0, x1, axis_bottom), fill=bar["color"])
        cluster_x0 = xs[0] - bar_w / 2
        cluster_x1 = xs[-1] + bar_w / 2
        draw.line((cluster_x0, axis_bottom, cluster_x1, axis_bottom), fill="#222222", width=3)
        draw_centered(draw, ((cluster_x0 + cluster_x1) / 2, axis_bottom + 27), model_name, small_font, "#111111")
        x += cluster_width + cluster_gap

    section_specs = [
        ("Single Trigger", 0, 8),
        ("Temporal", 8, 11),
    ]
    y = 38
    for section, start, end in section_specs:
        draw.text((legend_left, y), section, font=legend_small, fill="#333333")
        y += 24
        for bar in bars[start:end]:
            draw.rounded_rectangle((legend_left, y + 3, legend_left + 20, y + 23), radius=2, fill=bar["color"])
            label = bar["label"].replace("\n", " ")
            draw.text((legend_left + 30, y + 4), label, font=legend_font, fill="#111111")
            y += 26
        y += 10

    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    img.save(PNG_OUT)
    img.save(PDF_OUT, "PDF", resolution=300)
    print(f"wrote {PNG_OUT}")
    print(f"wrote {PDF_OUT}")


if __name__ == "__main__":
    main()
