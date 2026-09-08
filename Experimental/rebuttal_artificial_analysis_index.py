"""Correlate Artificial Analysis Intelligence Index with rebuttal metrics.

The reviewer suggested replacing a coarse model-capability rank with the
Artificial Analysis Intelligence Index. This script fetches the official
Artificial Analysis evaluation page, extracts model scores, matches them to the
SuperSycophantic main model set, and computes correlations against existing
rebuttal result summaries.
"""

from __future__ import annotations

import csv
import math
import re
import statistics
import urllib.request
from collections.abc import Sequence
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "Experimental" / "reports" / "rebuttal"
AA_URL = "https://artificialanalysis.ai/evaluations/artificial-analysis-intelligence-index"

MAIN_MODELS = [
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

# Conservative primary mapping: match the endpoint label used in our runs. For
# Claude, this uses the non-reasoning AA entry because our model IDs do not set
# an explicit thinking/reasoning variant. Command-R 08/2024 is not present on
# the AA page, so it is left unmatched in the strict primary analysis.
PRIMARY_AA_SLUG = {
    "openai/gpt-5.4": "gpt-5-4",
    "openai/gpt-5.4-mini": "gpt-5-4-mini",
    "openai/gpt-5.4-nano": "gpt-5-4-nano",
    "anthropic/claude-opus-4.5": "claude-opus-4-5",
    "anthropic/claude-sonnet-4.5": "claude-4-5-sonnet",
    "anthropic/claude-haiku-4.5": "claude-4-5-haiku",
    "google/gemini-3.1-flash-lite-preview": "gemini-3-1-flash-lite-preview",
    "mistralai/mistral-medium-3.1": "mistral-medium-3-1",
}

# Sensitivity mapping: use AA reasoning entries for Claude variants, matching
# how the AA chart often displays reasoning-capable Claude models, and include
# the closest official Command-R entry because the exact 08/2024 endpoint is
# missing from AA.
SENSITIVITY_AA_SLUG = {
    **PRIMARY_AA_SLUG,
    "anthropic/claude-opus-4.5": "claude-opus-4-5-thinking",
    "anthropic/claude-sonnet-4.5": "claude-4-5-sonnet-thinking",
    "anthropic/claude-haiku-4.5": "claude-4-5-haiku-reasoning",
    "cohere/command-r-08-2024": "command-r-03-2024",
}


def fetch_text(url: str) -> str:
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return response.read().decode("utf-8", errors="replace")


def scalar_field(record: str, key: str) -> str | float | bool | None:
    pattern = rf'"{re.escape(key)}":("(?:[^"\\]|\\.)*"|[0-9.+\-eE]+|null|true|false)'
    match = re.search(pattern, record)
    if not match:
        return None
    value = match.group(1)
    if value == "null":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    if value.startswith('"'):
        return value[1:-1]
    return float(value)


def extract_aa_records(raw_html: str) -> dict[str, dict[str, object]]:
    # The page stores escaped React flight data. Unescaping the quote layer makes
    # the repeated model objects easy to scan without relying on the web UI.
    text = (
        raw_html.replace('\\"', '"')
        .replace("\\u0026", "&")
        .replace("\\/", "/")
    )
    starts = [m.start() for m in re.finditer(r'\{"additional_text"', text)]
    records: dict[str, dict[str, object]] = {}
    for index, start in enumerate(starts):
        end = starts[index + 1] - 1 if index + 1 < len(starts) else text.find("],", start)
        record = text[start : end if end != -1 else start + 200_000]
        slug = scalar_field(record, "slug")
        if not isinstance(slug, str):
            continue
        score = scalar_field(record, "intelligence_index_v4_1")
        display_score = scalar_field(record, "intelligence_index")
        records[slug] = {
            "slug": slug,
            "name": scalar_field(record, "name"),
            "short_name": scalar_field(record, "short_name"),
            "score": score if score is not None else display_score,
            "display_score": display_score,
            "estimated_score": scalar_field(record, "estimated_intelligence_index_v4_1"),
            "deprecated": scalar_field(record, "deprecated"),
            "deprecated_to": scalar_field(record, "deprecated_to"),
            "release_date": scalar_field(record, "release_date"),
            "reasoning_model": scalar_field(record, "reasoning_model"),
        }
    return records


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: str | None) -> float | None:
    if value in (None, ""):
        return None
    return float(value)


def build_metric_table() -> dict[str, dict[str, float]]:
    metrics: dict[str, dict[str, float]] = {model: {} for model in MAIN_MODELS}

    context_path = REPORT_DIR / "openrouter_context_rerun_summary.csv"
    for row in read_csv_rows(context_path):
        if row.get("run_label") != "full":
            continue
        model = row.get("model")
        if model not in metrics or model == "all":
            continue
        entries = {
            "context_obj_truth_departure": safe_float(row.get("obj_correct_to_incorrect_rate")),
            "context_sub_user_lift": safe_float(row.get("sub_user_lift")),
            "context_sub_both_user_aligned": safe_float(row.get("sub_both_user_aligned_rate")),
        }
        for key, value in entries.items():
            if value is not None:
                metrics[model][key] = value

    trigger_path = REPORT_DIR / "openrouter_static_trigger_rerun_summary.csv"
    for row in read_csv_rows(trigger_path):
        if (
            row.get("run_label") != "full"
            or row.get("group_by") != "model"
            or row.get("group_value") not in metrics
        ):
            continue
        parsed = int(row.get("parsed_records") or 0)
        if parsed < 1000:
            continue
        branch = row.get("branch", "").lower()
        stage = row.get("run_stage", "").replace("_trigger", "")
        suffix = "truth_departure" if branch == "obj" else "answer_switch"
        metrics[row["group_value"]][f"trigger_{branch}_{stage}_{suffix}"] = float(row["switched_rate"])

    for model, model_metrics in metrics.items():
        obj_values = [
            model_metrics[key]
            for key in ("trigger_obj_static_truth_departure", "trigger_obj_adaptive_truth_departure")
            if key in model_metrics
        ]
        sub_values = [
            model_metrics[key]
            for key in ("trigger_sub_static_answer_switch", "trigger_sub_adaptive_answer_switch")
            if key in model_metrics
        ]
        if obj_values:
            model_metrics["trigger_obj_mean_truth_departure"] = statistics.mean(obj_values)
        if sub_values:
            model_metrics["trigger_sub_mean_answer_switch"] = statistics.mean(sub_values)
    return metrics


def average_ranks(values: Sequence[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        rank = (i + 1 + j) / 2
        for original_index, _ in indexed[i:j]:
            ranks[original_index] = rank
        i = j
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float:
    x_mean = statistics.mean(xs)
    y_mean = statistics.mean(ys)
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    x_denom = math.sqrt(sum((x - x_mean) ** 2 for x in xs))
    y_denom = math.sqrt(sum((y - y_mean) ** 2 for y in ys))
    if x_denom == 0 or y_denom == 0:
        return math.nan
    return numerator / (x_denom * y_denom)


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float:
    return pearson(average_ranks(xs), average_ranks(ys))


def correlation_rows(
    score_rows: list[dict[str, object]], metrics: dict[str, dict[str, float]]
) -> list[dict[str, object]]:
    policies = {
        "primary_strict_exact_n8": "primary_score",
        "primary_plus_closest_command_r_n9": "primary_or_closest_command_r_score",
        "claude_reasoning_plus_closest_command_r_n9": "sensitivity_score",
    }
    metric_names = sorted({name for model_metrics in metrics.values() for name in model_metrics})
    rows: list[dict[str, object]] = []
    by_model = {row["model"]: row for row in score_rows}
    for policy, score_key in policies.items():
        for metric in metric_names:
            pairs: list[tuple[str, float, float]] = []
            for model in MAIN_MODELS:
                score = by_model[model].get(score_key)
                value = metrics[model].get(metric)
                if score in (None, "") or value is None:
                    continue
                pairs.append((model, float(score), float(value)))
            if len(pairs) < 3:
                continue
            xs = [score for _, score, _ in pairs]
            ys = [value for _, _, value in pairs]
            rho = spearman(xs, ys)
            rows.append(
                {
                    "policy": policy,
                    "metric": metric,
                    "n": len(pairs),
                    "spearman_rho": f"{rho:.3f}",
                    "pearson_r": f"{pearson(xs, ys):.3f}",
                    "models": "; ".join(model for model, _, _ in pairs),
                }
            )
    return rows


def ranked(values: list[dict[str, object]], score_key: str) -> dict[str, int]:
    scored = [
        (row["model"], float(row[score_key]))
        for row in values
        if row.get(score_key) not in ("", None)
    ]
    scored.sort(key=lambda item: item[1], reverse=True)
    return {model: index + 1 for index, (model, _) in enumerate(scored)}


def joined_metric_rows(
    scores: list[dict[str, object]], metrics: dict[str, dict[str, float]]
) -> list[dict[str, object]]:
    primary_ranks = ranked(scores, "primary_score")
    sensitivity_ranks = ranked(scores, "sensitivity_score")
    rows: list[dict[str, object]] = []
    for row in scores:
        model = str(row["model"])
        joined = {
            "model": model,
            "primary_score": row["primary_score"],
            "primary_score_rank": primary_ranks.get(model, ""),
            "sensitivity_score": row["sensitivity_score"],
            "sensitivity_score_rank": sensitivity_ranks.get(model, ""),
        }
        joined.update(metrics.get(model, {}))
        rows.append(joined)
    return rows


def score_rows(aa_records: dict[str, dict[str, object]]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    command_r_closest_slug = "command-r-03-2024"
    for model in MAIN_MODELS:
        primary_slug = PRIMARY_AA_SLUG.get(model)
        primary = aa_records.get(primary_slug, {}) if primary_slug else {}
        closest_slug = primary_slug
        closest = primary
        closest_note = "same as primary"
        if model == "cohere/command-r-08-2024":
            closest_slug = command_r_closest_slug
            closest = aa_records.get(command_r_closest_slug, {})
            closest_note = "AA lacks Command-R 08/2024; closest official Command-R entry is Mar 2024"

        sensitivity_slug = SENSITIVITY_AA_SLUG.get(model)
        sensitivity = aa_records.get(sensitivity_slug, {}) if sensitivity_slug else {}
        if model == "cohere/command-r-08-2024":
            sensitivity_note = "closest official Command-R entry; exact 08/2024 endpoint missing"
        elif sensitivity_slug != primary_slug:
            sensitivity_note = "Claude reasoning variant"
        else:
            sensitivity_note = "same as primary"
        rows.append(
            {
                "model": model,
                "primary_aa_slug": primary_slug or "",
                "primary_aa_name": primary.get("name", ""),
                "primary_score": primary.get("score", ""),
                "primary_score_rounded": round(float(primary["score"])) if primary.get("score") is not None else "",
                "primary_reasoning_model": primary.get("reasoning_model", ""),
                "primary_deprecated": primary.get("deprecated", ""),
                "primary_deprecated_to": primary.get("deprecated_to", ""),
                "primary_match_note": "exact endpoint-label match" if primary_slug else "no exact AA entry",
                "primary_or_closest_command_r_score": closest.get("score", ""),
                "closest_command_r_match_note": closest_note,
                "sensitivity_aa_slug": sensitivity_slug or "",
                "sensitivity_aa_name": sensitivity.get("name", ""),
                "sensitivity_score": sensitivity.get("score", ""),
                "sensitivity_score_rounded": round(float(sensitivity["score"]))
                if sensitivity.get("score") is not None
                else "",
                "sensitivity_reasoning_model": sensitivity.get("reasoning_model", ""),
                "sensitivity_match_note": sensitivity_note,
            }
        )
    return rows


def write_summary(scores: list[dict[str, object]], correlations: list[dict[str, object]]) -> None:
    strict_n = sum(1 for row in scores if row["primary_score"] not in ("", None))
    missing = [row["model"] for row in scores if row["primary_score"] in ("", None)]
    selected_metrics = {
        row["metric"]: row
        for row in correlations
        if row["policy"] == "primary_strict_exact_n8"
        and row["metric"]
        in {
            "trigger_obj_adaptive_truth_departure",
            "trigger_sub_adaptive_answer_switch",
            "context_obj_truth_departure",
            "context_sub_both_user_aligned",
        }
    }
    lines = [
        "# Artificial Analysis Index Rebuttal Analysis",
        "",
        f"Source: {AA_URL}",
        "",
        f"Strict exact endpoint-label matches: {strict_n}/{len(MAIN_MODELS)}.",
        f"Strict unmatched models: {', '.join(missing) if missing else 'none'}.",
        "",
        "Primary strict AAI Index scores (display integers; raw scores retained in CSV):",
    ]
    for row in scores:
        score = row["primary_score"]
        score_text = "unmatched" if score in ("", None) else str(round(float(score)))
        lines.append(f"- {row['model']}: {score_text} ({row['primary_aa_name'] or row['primary_match_note']})")
    lines.extend(
        [
            "",
            "Cohere note: the exact `cohere/command-r-08-2024` endpoint is not listed by AA. "
            "The official AA entries are Command-R (Mar '24) with AAI Index score 2 "
            "and Command-R+ (Apr '24) with AAI Index score 3; the sensitivity analysis uses "
            "Command-R=2 rather than substituting Command-R+.",
        ]
    )
    lines.extend(["", "Key strict correlations:"])
    for metric, row in selected_metrics.items():
        lines.append(
            f"- {metric}: Spearman rho={row['spearman_rho']}, n={row['n']}"
        )
    lines.extend(
        [
            "",
            "Interpretation: AA capability score does not yield a monotonic explanation of sycophancy. "
            "Capability alone is insufficient: high-score models can remain vulnerable under persuasive "
            "pressure, while lower-score models are not uniformly more sycophantic across OBJ and SUB settings.",
        ]
    )
    (REPORT_DIR / "artificial_analysis_index_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    aa_records = extract_aa_records(fetch_text(AA_URL))
    scores = score_rows(aa_records)
    metrics = build_metric_table()
    correlations = correlation_rows(scores, metrics)
    joined = joined_metric_rows(scores, metrics)

    score_fieldnames = [
        "model",
        "primary_aa_slug",
        "primary_aa_name",
        "primary_score",
        "primary_score_rounded",
        "primary_reasoning_model",
        "primary_deprecated",
        "primary_deprecated_to",
        "primary_match_note",
        "primary_or_closest_command_r_score",
        "closest_command_r_match_note",
        "sensitivity_aa_slug",
        "sensitivity_aa_name",
        "sensitivity_score",
        "sensitivity_score_rounded",
        "sensitivity_reasoning_model",
        "sensitivity_match_note",
    ]
    write_csv(REPORT_DIR / "artificial_analysis_index_scores.csv", scores, score_fieldnames)
    write_csv(
        REPORT_DIR / "artificial_analysis_index_correlations.csv",
        correlations,
        [
            "policy",
            "metric",
            "n",
            "spearman_rho",
            "pearson_r",
            "models",
        ],
    )
    joined_fieldnames = [
        "model",
        "primary_score",
        "primary_score_rank",
        "sensitivity_score",
        "sensitivity_score_rank",
        "context_obj_truth_departure",
        "context_sub_user_lift",
        "context_sub_both_user_aligned",
        "trigger_obj_static_truth_departure",
        "trigger_obj_adaptive_truth_departure",
        "trigger_obj_mean_truth_departure",
        "trigger_sub_static_answer_switch",
        "trigger_sub_adaptive_answer_switch",
        "trigger_sub_mean_answer_switch",
    ]
    write_csv(
        REPORT_DIR / "artificial_analysis_index_joined_metrics.csv",
        joined,
        joined_fieldnames,
    )
    write_summary(scores, correlations)
    print((REPORT_DIR / "artificial_analysis_index_scores.csv").as_posix())
    print((REPORT_DIR / "artificial_analysis_index_correlations.csv").as_posix())
    print((REPORT_DIR / "artificial_analysis_index_joined_metrics.csv").as_posix())
    print((REPORT_DIR / "artificial_analysis_index_summary.md").as_posix())


if __name__ == "__main__":
    main()
