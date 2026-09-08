"""Compute appendix-derived descriptive statistics.

This script intentionally works from manuscript appendix aggregate tables rather
than raw trial-level result files. The outputs are therefore descriptive and
aggregate-level only: Wilson intervals for reported rates, conservative
bounded intervals for confidence means, capability-rank correlations, and
clearly labeled aggregate difference checks.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import re
from pathlib import Path


MODEL_ORDER = [
    "GPT-5.4",
    "GPT-5.4-Mini",
    "GPT-5.4-Nano",
    "Opus-4.5",
    "Sonnet-4.5",
    "Haiku-4.5",
    "Gemini-3.1-Flash-Lite",
    "Mistral-Medium-3.1",
    "Command-R",
]

MODEL_LABEL_ALIASES = {
    "GPT-5.4-Mini": "GPT-5.4-Mini",
    "GPT-5.4-Nano": "GPT-5.4-Nano",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--appendix-tex", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parents[1] / "Experimental/reports/appendix_statistics")
    return parser.parse_args()


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def strip_tex(value: str) -> str:
    value = value.strip()
    value = value.replace("\\%", "%")
    value = value.replace("\\&", "&")
    value = value.replace("\\texttt{", "").replace("\\textbf{", "")
    value = value.replace("$^\\dagger$", "").replace("$", "")
    value = value.replace("{", "").replace("}", "")
    value = re.sub(r"\\[a-zA-Z]+", "", value)
    value = value.replace("~", " ")
    return re.sub(r"\s+", " ", value).strip()


def table_block(tex: str, label: str) -> str:
    label_pos = tex.find(f"\\label{{{label}}}")
    if label_pos < 0:
        raise ValueError(f"Cannot find appendix table label: {label}")
    begin = tex.rfind("\\begin{tabular}", 0, label_pos)
    if begin < 0:
        raise ValueError(f"Cannot find tabular start for: {label}")
    end = tex.find("\\end{tabular}", begin)
    if end < 0:
        raise ValueError(f"Cannot find tabular end for: {label}")
    return tex[begin:end]


def data_rows(block: str) -> list[list[str]]:
    rows: list[list[str]] = []
    for raw_line in block.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("\\") or "&" not in line:
            continue
        line = line.split("\\\\", 1)[0]
        fields = [strip_tex(part) for part in line.split("&")]
        if fields and fields[0].lower() in {"model", "framing type", "panel", "setting", "rank"}:
            continue
        rows.append(fields)
    return rows


def parse_rate(value: str) -> float:
    return float(strip_tex(value).replace("%", ""))


def pct_to_events(rate_pct: float, denom: int) -> int:
    return int(round(rate_pct / 100.0 * denom))


def wilson_interval(events: int, denom: int, z: float = 1.96) -> tuple[float, float]:
    if denom <= 0:
        return 0.0, 0.0
    p = events / denom
    denom_adj = 1 + z * z / denom
    center = (p + z * z / (2 * denom)) / denom_adj
    half = z * math.sqrt((p * (1 - p) / denom) + (z * z / (4 * denom * denom))) / denom_adj
    return 100 * max(0.0, center - half), 100 * min(1.0, center + half)


def independent_diff_check(
    name: str,
    rate_a: float,
    denom_a: int,
    rate_b: float,
    denom_b: int,
) -> dict:
    events_a = pct_to_events(rate_a, denom_a)
    events_b = pct_to_events(rate_b, denom_b)
    p_a = events_a / denom_a
    p_b = events_b / denom_b
    diff = p_a - p_b
    se = math.sqrt(p_a * (1 - p_a) / denom_a + p_b * (1 - p_b) / denom_b)
    if se:
        z = diff / se
        p_value = max(math.erfc(abs(z) / math.sqrt(2)), 1e-300)
        low = 100 * (diff - 1.96 * se)
        high = 100 * (diff + 1.96 * se)
    else:
        z = 0.0
        p_value = 1.0
        low = high = 100 * diff
    return {
        "contrast": name,
        "rate_a_pct": 100 * p_a,
        "denom_a": denom_a,
        "events_a_approx": events_a,
        "rate_b_pct": 100 * p_b,
        "denom_b": denom_b,
        "events_b_approx": events_b,
        "diff_a_minus_b_pp": 100 * diff,
        "ci95_low_pp": low,
        "ci95_high_pp": high,
        "z_approx": z,
        "p_value_approx": p_value,
        "method": "independent_two_proportion_normal_approximation",
        "interpretation_note": "aggregate-only sanity check; ignores matched cells and should not replace raw-record paired tests",
    }


def bounded_mean_interval(mean: float, n: int, low: float, high: float, alpha: float = 0.05) -> tuple[float, float]:
    if n <= 0:
        return mean, mean
    half = (high - low) * math.sqrt(math.log(2 / alpha) / (2 * n))
    return max(low, mean - half), min(high, mean + half)


def fisher_ci(r: float, n: int) -> tuple[float, float]:
    if n <= 3:
        return -1.0, 1.0
    r = max(-0.999999, min(0.999999, r))
    z = math.atanh(r)
    half = 1.96 / math.sqrt(n - 3)
    return math.tanh(z - half), math.tanh(z + half)


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda pair: pair[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i + 1
        while j < len(indexed) and indexed[j][1] == indexed[i][1]:
            j += 1
        avg = (i + 1 + j) / 2
        for k in range(i, j):
            ranks[indexed[k][0]] = avg
        i = j
    return ranks


def pearson(x: list[float], y: list[float]) -> float:
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    num = sum((a - mean_x) * (b - mean_y) for a, b in zip(x, y))
    den_x = math.sqrt(sum((a - mean_x) ** 2 for a in x))
    den_y = math.sqrt(sum((b - mean_y) ** 2 for b in y))
    return num / (den_x * den_y) if den_x and den_y else 0.0


def spearman_exact_permutation(x: list[float], y: list[float]) -> tuple[float, float]:
    rx = rank(x)
    ry = rank(y)
    observed = pearson(rx, ry)
    extreme = 0
    total = 0
    for perm in itertools.permutations(ry):
        total += 1
        if abs(pearson(rx, list(perm))) >= abs(observed) - 1e-12:
            extreme += 1
    return observed, extreme / total


def context_model_rows(tex: str) -> list[dict]:
    rows = []
    for fields in data_rows(table_block(tex, "tab:context_model_rates")):
        model, neutral, framed, correct_wrong, sub_conf, sub_lift = fields
        neutral_rate = parse_rate(neutral)
        framed_rate = parse_rate(framed)
        correct_wrong_rate = parse_rate(correct_wrong)
        sub_conf_rate = parse_rate(sub_conf)
        neutral_correct = pct_to_events(neutral_rate, 200)
        specs = [
            ("OBJ neutral accuracy", neutral_rate, 200, "exact_design_denominator"),
            ("OBJ framed accuracy", framed_rate, 600, "exact_design_denominator"),
            ("OBJ correct-to-wrong", correct_wrong_rate, neutral_correct * 3, "denominator_inferred_from_neutral_accuracy"),
            ("SUB conformity", sub_conf_rate, 600, "exact_design_denominator"),
            ("SUB lift", parse_rate(sub_lift), None, "descriptive_difference_from_neutral"),
        ]
        for metric, rate, denom, note in specs:
            if denom is None:
                events = low = high = ""
                method = "not_computed_from_appendix"
            else:
                events = pct_to_events(rate, denom)
                low, high = wilson_interval(events, denom)
                method = "wilson_score_interval_for_reported_rate"
            rows.append(
                {
                    "panel": "context_model",
                    "group": model,
                    "metric": metric,
                    "rate_pct": rate,
                    "events_approx": events,
                    "denom": "" if denom is None else denom,
                    "ci95_low_pct": low,
                    "ci95_high_pct": high,
                    "method": method,
                    "denominator_note": note,
                }
            )
    return rows


def context_cue_rows(tex: str, neutral_correct_total: int) -> list[dict]:
    rows = []
    for cue, obj_cw, sub_conf, sub_lift in data_rows(table_block(tex, "tab:context_cue_rates")):
        specs = [
            ("OBJ correct-to-wrong", parse_rate(obj_cw), neutral_correct_total, "denominator_inferred_from_model_neutral_accuracy"),
            ("SUB conformity", parse_rate(sub_conf), 1800, "exact_design_denominator"),
            ("SUB lift", parse_rate(sub_lift), None, "descriptive_difference_from_neutral"),
        ]
        for metric, rate, denom, note in specs:
            if denom is None:
                events = low = high = ""
                method = "not_computed_from_appendix"
            else:
                events = pct_to_events(rate, denom)
                low, high = wilson_interval(events, denom)
                method = "wilson_score_interval_for_reported_rate"
            rows.append(
                {
                    "panel": "context_cue",
                    "group": cue,
                    "metric": metric,
                    "rate_pct": rate,
                    "events_approx": events,
                    "denom": "" if denom is None else denom,
                    "ci95_low_pct": low,
                    "ci95_high_pct": high,
                    "method": method,
                    "denominator_note": note,
                }
            )
    return rows


def trigger_rows(tex: str) -> tuple[list[dict], list[dict]]:
    interval_rows = []
    diff_rows = []
    for model, obj_changed, obj_cw, sub_switch in data_rows(table_block(tex, "tab:trigger_model_rates")):
        for metric, rate, denom, note in [
            ("OBJ answer changed", parse_rate(obj_changed), 8400, "exact_design_denominator_for_cialdini_single_followup"),
            ("SUB switching", parse_rate(sub_switch), 4200, "exact_design_denominator_for_cialdini_single_followup"),
        ]:
            events = pct_to_events(rate, denom)
            low, high = wilson_interval(events, denom)
            interval_rows.append(
                {
                    "panel": "trigger_model_single_followup",
                    "group": model,
                    "metric": metric,
                    "rate_pct": rate,
                    "events_approx": events,
                    "denom": denom,
                    "ci95_low_pct": low,
                    "ci95_high_pct": high,
                    "method": "wilson_score_interval_for_reported_rate",
                    "denominator_note": note,
                }
            )
        interval_rows.append(
            {
                "panel": "trigger_model_single_followup",
                "group": model,
                "metric": "OBJ correct-to-wrong",
                "rate_pct": parse_rate(obj_cw),
                "events_approx": "",
                "denom": "",
                "ci95_low_pct": "",
                "ci95_high_pct": "",
                "method": "not_computed_from_appendix",
                "denominator_note": "initially-correct trigger denominator is not reported in appendix aggregate table",
            }
        )

    aggregate = {}
    for panel, split, r1, r2, r3 in data_rows(table_block(tex, "tab:trigger_aggregate_rates")):
        aggregate[panel] = [parse_rate(r1), parse_rate(r2), None if r3 == "n/a" else parse_rate(r3)]

    # SUB denominators are exactly inferable from the release grid for these aggregate rows.
    sub_tone_den = 9 * 100 * 7 * 2
    sub_mode_den = 9 * 100 * 7 * 3
    sub_temporal_denoms = {"single": sub_mode_den, "same_family": 9 * 100 * 7, "heterogeneous": 9 * 100 * 6}

    for tone, rate in zip(["mild", "moderate", "strong"], aggregate["SUB tone, all models"]):
        events = pct_to_events(rate, sub_tone_den)
        low, high = wilson_interval(events, sub_tone_den)
        interval_rows.append(
            {
                "panel": "trigger_aggregate",
                "group": f"SUB tone {tone}",
                "metric": "SUB switching",
                "rate_pct": rate,
                "events_approx": events,
                "denom": sub_tone_den,
                "ci95_low_pct": low,
                "ci95_high_pct": high,
                "method": "wilson_score_interval_for_reported_rate",
                "denominator_note": "exact_design_denominator",
            }
        )
    diff_rows.append(independent_diff_check("SUB tone moderate minus mild", aggregate["SUB tone, all models"][1], sub_tone_den, aggregate["SUB tone, all models"][0], sub_tone_den))
    diff_rows.append(independent_diff_check("SUB tone strong minus moderate", aggregate["SUB tone, all models"][2], sub_tone_den, aggregate["SUB tone, all models"][1], sub_tone_den))

    for mode, rate in zip(["static", "adaptive"], aggregate["SUB"][:2]):
        events = pct_to_events(rate, sub_mode_den)
        low, high = wilson_interval(events, sub_mode_den)
        interval_rows.append(
            {
                "panel": "trigger_aggregate",
                "group": f"SUB {mode}",
                "metric": "SUB switching",
                "rate_pct": rate,
                "events_approx": events,
                "denom": sub_mode_den,
                "ci95_low_pct": low,
                "ci95_high_pct": high,
                "method": "wilson_score_interval_for_reported_rate",
                "denominator_note": "exact_design_denominator",
            }
        )
    diff_rows.append(independent_diff_check("SUB adaptive minus static", aggregate["SUB"][1], sub_mode_den, aggregate["SUB"][0], sub_mode_den))

    for mode_key in ["SUB static", "SUB adaptive"]:
        rates = aggregate[mode_key]
        for stage, rate in zip(["single", "same_family", "heterogeneous"], rates):
            denom = sub_temporal_denoms[stage]
            events = pct_to_events(rate, denom)
            low, high = wilson_interval(events, denom)
            interval_rows.append(
                {
                    "panel": "trigger_temporal_aggregate",
                    "group": f"{mode_key} {stage}",
                    "metric": "SUB final switching",
                    "rate_pct": rate,
                    "events_approx": events,
                    "denom": denom,
                    "ci95_low_pct": low,
                    "ci95_high_pct": high,
                    "method": "wilson_score_interval_for_reported_rate",
                    "denominator_note": "exact_design_denominator",
                }
            )
        diff_rows.append(independent_diff_check(f"{mode_key} same-family minus single", rates[1], sub_temporal_denoms["same_family"], rates[0], sub_temporal_denoms["single"]))
        diff_rows.append(independent_diff_check(f"{mode_key} heterogeneous minus single", rates[2], sub_temporal_denoms["heterogeneous"], rates[0], sub_temporal_denoms["single"]))
        diff_rows.append(independent_diff_check(f"{mode_key} heterogeneous minus same-family", rates[2], sub_temporal_denoms["heterogeneous"], rates[1], sub_temporal_denoms["same_family"]))

    for panel in ["OBJ tone, all models", "OBJ", "OBJ static", "OBJ adaptive"]:
        if panel in aggregate:
            for idx, rate in enumerate(aggregate[panel]):
                if rate is None:
                    continue
                interval_rows.append(
                    {
                        "panel": "trigger_aggregate",
                        "group": f"{panel} rate_{idx + 1}",
                        "metric": "OBJ correct-to-wrong",
                        "rate_pct": rate,
                        "events_approx": "",
                        "denom": "",
                        "ci95_low_pct": "",
                        "ci95_high_pct": "",
                        "method": "not_computed_from_appendix",
                        "denominator_note": "initially-correct trigger denominator is not reported in appendix aggregate table",
                    }
                )
    return interval_rows, diff_rows


def confidence_rows(tex: str) -> list[dict]:
    rows = []
    for setting, final_state, initial, final, delta, n_text in data_rows(table_block(tex, "tab:trigger_confidence_rates")):
        n = int(n_text)
        initial_mean = parse_rate(initial)
        final_mean = parse_rate(final)
        delta_mean = parse_rate(delta)
        initial_low, initial_high = bounded_mean_interval(initial_mean, n, 1.0, 5.0)
        final_low, final_high = bounded_mean_interval(final_mean, n, 1.0, 5.0)
        delta_low, delta_high = bounded_mean_interval(delta_mean, n, -4.0, 4.0)
        rows.append(
            {
                "setting": setting,
                "final_state": final_state,
                "n": n,
                "initial_mean": initial_mean,
                "initial_ci95_low": initial_low,
                "initial_ci95_high": initial_high,
                "final_mean": final_mean,
                "final_ci95_low": final_low,
                "final_ci95_high": final_high,
                "delta_mean": delta_mean,
                "delta_ci95_low": delta_low,
                "delta_ci95_high": delta_high,
                "method": "hoeffding_bounded_interval_from_appendix_mean_and_n",
                "interpretation_note": "conservative descriptive interval; raw paired confidence scores are needed for tighter paired tests",
            }
        )
    return rows


def judge_rows(tex: str) -> list[dict]:
    rows = []
    try:
        block_rows = data_rows(table_block(tex, "tab:judge_agreement"))
    except ValueError:
        return rows
    pattern = re.compile(r"(Uncritical Agreement|Obsequiousness|Excitement)\s+([0-9.]+)")
    for panel, units_text, missing, factor_text, binary_text in block_rows:
        n = int(units_text.replace(",", ""))
        for label, value in pattern.findall(factor_text):
            r = float(value)
            low, high = fisher_ci(r, n)
            rows.append(
                {
                    "panel": panel,
                    "metric": label,
                    "agreement": "pearson_r",
                    "n": n,
                    "estimate": r,
                    "ci95_low": low,
                    "ci95_high": high,
                    "method": "fisher_z_interval",
                    "note": "computed from appendix aggregate Pearson r and paired-unit count",
                }
            )
        rows.append(
            {
                "panel": panel,
                "metric": "binary_label_range",
                "agreement": "cohen_kappa_and_exact_agreement_ranges",
                "n": n,
                "estimate": binary_text,
                "ci95_low": "",
                "ci95_high": "",
                "method": "not_computed_from_appendix",
                "note": "appendix reports only ranges; per-label confusion counts are needed for kappa intervals",
            }
        )
    return rows


def capability_rows(tex: str) -> list[dict]:
    capability = {}
    for rank_text, model, match, index in data_rows(table_block(tex, "tab:capability_rank")):
        capability[model] = float(index)

    context = {}
    for fields in data_rows(table_block(tex, "tab:context_model_rates")):
        model = fields[0]
        context[model] = {
            "context_OBJ_correct_to_wrong": parse_rate(fields[3]),
            "context_SUB_conformity": parse_rate(fields[4]),
            "context_SUB_lift": parse_rate(fields[5]),
        }
    trigger = {}
    for fields in data_rows(table_block(tex, "tab:trigger_model_rates")):
        model = fields[0]
        trigger[model] = {
            "trigger_OBJ_answer_changed": parse_rate(fields[1]),
            "trigger_OBJ_correct_to_wrong": parse_rate(fields[2]),
            "trigger_SUB_switching": parse_rate(fields[3]),
        }

    rows = []
    for metric in [
        "context_OBJ_correct_to_wrong",
        "context_SUB_conformity",
        "context_SUB_lift",
        "trigger_OBJ_answer_changed",
        "trigger_OBJ_correct_to_wrong",
        "trigger_SUB_switching",
    ]:
        models = [model for model in MODEL_ORDER if model in capability and (model in context or model in trigger)]
        x = [capability[model] for model in models]
        y = [(context.get(model) or trigger.get(model, {})).get(metric) for model in models]
        if any(value is None for value in y):
            y = [context.get(model, {}).get(metric, trigger.get(model, {}).get(metric)) for model in models]
        rho, p_value = spearman_exact_permutation(x, [float(value) for value in y])
        rows.append(
            {
                "metric": metric,
                "n_models": len(models),
                "spearman_rho": rho,
                "exact_permutation_p_two_sided": p_value,
                "method": "spearman_rank_correlation_exact_permutation",
                "interpretation_note": "diagnostic association over nine models; not a mechanistic or causal test",
            }
        )
    return rows


def write_summary(out_dir: Path, counts: dict[str, int]) -> None:
    text = [
        "# Appendix-Derived Statistical Outputs",
        "",
        "These files are computed only from manuscript appendix aggregate tables.",
        "They are suitable for descriptive uncertainty and diagnostic association checks.",
        "They do not replace raw-record matched tests once pass@1/pass@k records are available.",
        "",
        "Generated files:",
        f"- `appendix_context_rate_intervals.csv`: {counts['context']} rows",
        f"- `appendix_trigger_rate_intervals.csv`: {counts['trigger']} rows",
        f"- `appendix_aggregate_difference_checks.csv`: {counts['diff']} rows",
        f"- `appendix_confidence_bounded_intervals.csv`: {counts['confidence']} rows",
        f"- `appendix_capability_rank_correlations.csv`: {counts['capability']} rows",
        "",
        "Interpretation constraints:",
        "- Rows marked `not_computed_from_appendix` need raw initially-correct denominators.",
        "- Aggregate difference checks use independent-proportion approximations and ignore matched cells.",
        "- Confidence intervals use bounded Hoeffding intervals because appendix means do not include SDs.",
        "- Capability correlations use exact rank-permutation p-values over the nine-model panel.",
    ]
    (out_dir / "appendix_statistics_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    tex = args.appendix_tex.read_text(encoding="utf-8")

    context_rows = context_model_rows(tex)
    neutral_correct_total = sum(
        pct_to_events(row["rate_pct"], 200)
        for row in context_rows
        if row["panel"] == "context_model" and row["metric"] == "OBJ neutral accuracy"
    )
    context_rows.extend(context_cue_rows(tex, neutral_correct_total))
    trigger_interval_rows, diff_rows = trigger_rows(tex)
    conf_rows = confidence_rows(tex)
    judge_interval_rows = judge_rows(tex)
    cap_rows = capability_rows(tex)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.out_dir / "appendix_context_rate_intervals.csv", context_rows)
    write_csv(args.out_dir / "appendix_trigger_rate_intervals.csv", trigger_interval_rows)
    write_csv(args.out_dir / "appendix_aggregate_difference_checks.csv", diff_rows)
    write_csv(args.out_dir / "appendix_confidence_bounded_intervals.csv", conf_rows)
    write_csv(args.out_dir / "appendix_judge_agreement_intervals.csv", judge_interval_rows)
    write_csv(args.out_dir / "appendix_capability_rank_correlations.csv", cap_rows)
    write_csv(
        args.out_dir / "appendix_statistics_manifest.csv",
        [
            {
                "appendix_tex": str(args.appendix_tex),
                "source": "manuscript_appendix_aggregate_tables",
                "status": "ok",
                "note": "descriptive aggregate outputs only",
            }
        ],
    )
    write_summary(
        args.out_dir,
        {
            "context": len(context_rows),
            "trigger": len(trigger_interval_rows),
            "diff": len(diff_rows),
            "confidence": len(conf_rows),
            "judge": len(judge_interval_rows),
            "capability": len(cap_rows),
        },
    )
    print(f"Wrote appendix-derived statistics to {args.out_dir}")


if __name__ == "__main__":
    main()
