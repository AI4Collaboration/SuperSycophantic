#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PROJECT_7 = SCRIPT_DIR / "project-7-at-2026-05-06-00-06-dd1a3816.json"
DEFAULT_PROJECT_11 = SCRIPT_DIR / "project-11-at-2026-05-06-09-42-654c7fca.json"
DEFAULT_OUTPUT_JSON = SCRIPT_DIR / "label_studio_iaa_results.json"
DEFAULT_OUTPUT_MD = SCRIPT_DIR / "label_studio_iaa_summary.md"

ALIGNMENT_KEY_FIELDS = [
    "source_transcript_id",
    "transcript_id",
    "source_item_id",
    "branch",
    "model",
    "trigger_family",
    "context_condition",
    "tones",
]

BINARY_CATEGORIES = ["No", "Yes"]
ORDINAL_CATEGORIES = ["1", "2", "3", "4", "5"]


def stable_string(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    if value is None:
        return ""
    return str(value)


def alignment_key(task: dict[str, Any]) -> tuple[str, ...]:
    data = task.get("data", {})
    return tuple(stable_string(data.get(field)) for field in ALIGNMENT_KEY_FIELDS)


def display_key(key: tuple[str, ...]) -> dict[str, str]:
    return dict(zip(ALIGNMENT_KEY_FIELDS, key, strict=True))


def load_tasks(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected a list of tasks in {path}")
    return data


def choose_annotation(task: dict[str, Any]) -> dict[str, Any] | None:
    annotations = [
        annotation
        for annotation in task.get("annotations", [])
        if not annotation.get("was_cancelled") and annotation.get("result")
    ]
    if not annotations:
        return None
    return sorted(
        annotations,
        key=lambda item: (
            stable_string(item.get("updated_at")),
            stable_string(item.get("created_at")),
            stable_string(item.get("id")),
        ),
    )[-1]


def parse_annotation_values(annotation: dict[str, Any] | None) -> tuple[dict[str, str], list[dict[str, Any]]]:
    if annotation is None:
        return {}, []

    values: dict[str, str] = {}
    duplicate_fields: list[dict[str, Any]] = []

    for result in annotation.get("result", []):
        field = result.get("from_name")
        value = result.get("value", {})
        if not field:
            continue

        parsed_value: str | None = None
        choices = value.get("choices")
        if isinstance(choices, list) and len(choices) == 1:
            parsed_value = stable_string(choices[0])
        elif "rating" in value:
            parsed_value = stable_string(value.get("rating"))
        elif "number" in value:
            parsed_value = stable_string(value.get("number"))
        elif "text" in value:
            text = value.get("text")
            if isinstance(text, list) and len(text) == 1:
                parsed_value = stable_string(text[0])

        if parsed_value is None:
            continue
        if field in values:
            duplicate_fields.append(
                {
                    "field": field,
                    "first_value": values[field],
                    "second_value": parsed_value,
                }
            )
        values[field] = parsed_value

    return values, duplicate_fields


def index_export(path: Path, label: str) -> dict[str, Any]:
    tasks = load_tasks(path)
    records: dict[tuple[str, ...], dict[str, Any]] = {}
    duplicate_task_keys: list[dict[str, Any]] = []
    duplicate_annotation_fields: list[dict[str, Any]] = []
    empty_annotations: list[dict[str, Any]] = []
    annotations_per_task = Counter()
    annotator_ids = Counter()
    nonempty_annotation_count = 0

    for task in tasks:
        key = alignment_key(task)
        annotations = task.get("annotations", [])
        annotations_per_task[len(annotations)] += 1

        for annotation in annotations:
            annotator_ids[stable_string(annotation.get("completed_by"))] += 1
            if annotation.get("result"):
                nonempty_annotation_count += 1
            else:
                empty_annotations.append(
                    {
                        "export": label,
                        "task_id": task.get("id"),
                        "annotation_id": annotation.get("id"),
                        "completed_by": annotation.get("completed_by"),
                        "updated_at": annotation.get("updated_at"),
                        "key": display_key(key),
                    }
                )

        annotation = choose_annotation(task)
        values, duplicates = parse_annotation_values(annotation)
        for duplicate in duplicates:
            duplicate_annotation_fields.append(
                {
                    "export": label,
                    "task_id": task.get("id"),
                    "annotation_id": annotation.get("id") if annotation else None,
                    "key": display_key(key),
                    **duplicate,
                }
            )

        if key in records:
            duplicate_task_keys.append(
                {
                    "export": label,
                    "previous_task_id": records[key].get("task_id"),
                    "task_id": task.get("id"),
                    "key": display_key(key),
                }
            )

        records[key] = {
            "task_id": task.get("id"),
            "selected_annotation_id": annotation.get("id") if annotation else None,
            "completed_by": annotation.get("completed_by") if annotation else None,
            "updated_at": annotation.get("updated_at") if annotation else None,
            "values": values,
        }

    return {
        "path": str(path),
        "label": label,
        "task_count": len(tasks),
        "records": records,
        "annotations_per_task": dict(sorted(annotations_per_task.items())),
        "annotator_ids": dict(sorted(annotator_ids.items())),
        "nonempty_annotation_count": nonempty_annotation_count,
        "empty_annotations": empty_annotations,
        "duplicate_task_keys": duplicate_task_keys,
        "duplicate_annotation_fields": duplicate_annotation_fields,
    }


def parse_number(value: str) -> float | None:
    try:
        parsed = float(value)
    except ValueError:
        return None
    if math.isfinite(parsed):
        return parsed
    return None


def infer_field_type(values: list[str]) -> str:
    unique_values = set(values)
    if unique_values and unique_values.issubset(set(BINARY_CATEGORIES)):
        return "binary"
    if values and all(parse_number(value) is not None for value in values):
        return "continuous"
    return "unsupported"


def krippendorff_alpha_interval(units: list[list[float]]) -> float | None:
    usable_units = [unit for unit in units if len(unit) >= 2]
    values = [value for unit in usable_units for value in unit]
    total_values = len(values)
    if total_values < 2:
        return None

    observed = 0.0
    for unit in usable_units:
        unit_sum = 0.0
        for left in unit:
            for right in unit:
                unit_sum += (left - right) ** 2
        observed += unit_sum / (len(unit) - 1)
    observed /= total_values

    expected = 0.0
    for left in values:
        for right in values:
            expected += (left - right) ** 2
    expected /= total_values * (total_values - 1)

    if expected == 0:
        return None
    return 1.0 - observed / expected


def weighted_kappa(
    rater_a: list[str],
    rater_b: list[str],
    categories: list[str],
    weighting: str = "quadratic",
) -> float | None:
    if len(rater_a) != len(rater_b):
        raise ValueError("Rater vectors must have the same length")
    if not rater_a:
        return None

    index = {category: offset for offset, category in enumerate(categories)}
    width = max(len(categories) - 1, 1)

    def penalty(left: str, right: str) -> float:
        distance = abs(index[left] - index[right]) / width
        if weighting == "linear":
            return distance
        if weighting == "quadratic":
            return distance * distance
        raise ValueError(f"Unknown weighting: {weighting}")

    observed_counts = Counter(zip(rater_a, rater_b, strict=True))
    left_counts = Counter(rater_a)
    right_counts = Counter(rater_b)
    total = len(rater_a)

    observed_disagreement = 0.0
    expected_disagreement = 0.0
    for left in categories:
        for right in categories:
            observed_disagreement += penalty(left, right) * observed_counts[(left, right)] / total
            expected_disagreement += (
                penalty(left, right) * left_counts[left] * right_counts[right] / (total * total)
            )

    if expected_disagreement == 0:
        return None
    return 1.0 - observed_disagreement / expected_disagreement


def category_penalty(left: str, right: str, categories: list[str], weighting: str = "quadratic") -> float:
    index = {category: offset for offset, category in enumerate(categories)}
    width = max(len(categories) - 1, 1)
    distance = abs(index[left] - index[right]) / width
    if weighting == "linear":
        return distance
    if weighting == "quadratic":
        return distance * distance
    raise ValueError(f"Unknown weighting: {weighting}")


def category_agreement_weight(
    left: str,
    right: str,
    categories: list[str],
    weighting: str = "quadratic",
) -> float:
    return 1.0 - category_penalty(left, right, categories, weighting)


def gwet_ac1(rater_a: list[str], rater_b: list[str], categories: list[str]) -> float | None:
    if len(rater_a) != len(rater_b):
        raise ValueError("Rater vectors must have the same length")
    if not rater_a or len(categories) < 2:
        return None

    total = len(rater_a)
    observed_agreement = sum(
        1 for left, right in zip(rater_a, rater_b, strict=True) if left == right
    ) / total

    pooled_counts = Counter(rater_a)
    pooled_counts.update(rater_b)
    proportions = {category: pooled_counts[category] / (2 * total) for category in categories}
    chance_agreement = sum(
        proportions[category] * (1.0 - proportions[category])
        for category in categories
    ) / (len(categories) - 1)

    denominator = 1.0 - chance_agreement
    if denominator == 0:
        return None
    return (observed_agreement - chance_agreement) / denominator


def gwet_ac2(
    rater_a: list[str],
    rater_b: list[str],
    categories: list[str],
    weighting: str = "quadratic",
) -> float | None:
    if len(rater_a) != len(rater_b):
        raise ValueError("Rater vectors must have the same length")
    if not rater_a or len(categories) < 2:
        return None

    total = len(rater_a)
    observed_agreement = sum(
        category_agreement_weight(left, right, categories, weighting)
        for left, right in zip(rater_a, rater_b, strict=True)
    ) / total

    pooled_counts = Counter(rater_a)
    pooled_counts.update(rater_b)
    proportions = {category: pooled_counts[category] / (2 * total) for category in categories}
    chance_agreement = 0.0
    for left in categories:
        for right in categories:
            chance_agreement += (
                category_agreement_weight(left, right, categories, weighting)
                * proportions[left]
                * (1.0 - proportions[right])
            )
    chance_agreement /= len(categories) - 1

    denominator = 1.0 - chance_agreement
    if denominator == 0:
        return None
    return (observed_agreement - chance_agreement) / denominator


def pabak(exact_agreement: float | None) -> float | None:
    if exact_agreement is None:
        return None
    return 2.0 * exact_agreement - 1.0


def binary_pair_counts(rater_a: list[str], rater_b: list[str]) -> dict[str, int]:
    counts = Counter(zip(rater_a, rater_b, strict=True))
    return {
        "yes_yes": counts[("Yes", "Yes")],
        "yes_no": counts[("Yes", "No")],
        "no_yes": counts[("No", "Yes")],
        "no_no": counts[("No", "No")],
    }


def positive_agreement(rater_a: list[str], rater_b: list[str]) -> float | None:
    counts = binary_pair_counts(rater_a, rater_b)
    denominator = 2 * counts["yes_yes"] + counts["yes_no"] + counts["no_yes"]
    if denominator == 0:
        return None
    return 2 * counts["yes_yes"] / denominator


def negative_agreement(rater_a: list[str], rater_b: list[str]) -> float | None:
    counts = binary_pair_counts(rater_a, rater_b)
    denominator = 2 * counts["no_no"] + counts["yes_no"] + counts["no_yes"]
    if denominator == 0:
        return None
    return 2 * counts["no_no"] / denominator


def matthews_correlation_coefficient(rater_a: list[str], rater_b: list[str]) -> float | None:
    counts = binary_pair_counts(rater_a, rater_b)
    tp = counts["yes_yes"]
    fn = counts["yes_no"]
    fp = counts["no_yes"]
    tn = counts["no_no"]
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    if denominator == 0:
        return None
    return (tp * tn - fp * fn) / denominator


def within_tolerance(units: list[list[float]], tolerance: float) -> float | None:
    if not units:
        return None
    return sum(1 for left, right in units if abs(left - right) <= tolerance) / len(units)


def icc_absolute_agreement(units: list[list[float]]) -> float | None:
    complete_units = [unit for unit in units if len(unit) >= 2]
    if len(complete_units) < 2:
        return None

    n_subjects = len(complete_units)
    n_raters = len(complete_units[0])
    if n_raters < 2 or any(len(unit) != n_raters for unit in complete_units):
        return None

    grand_mean = sum(sum(unit) for unit in complete_units) / (n_subjects * n_raters)
    subject_means = [sum(unit) / n_raters for unit in complete_units]
    rater_means = [
        sum(unit[rater_index] for unit in complete_units) / n_subjects
        for rater_index in range(n_raters)
    ]

    ms_subject = (
        n_raters
        * sum((subject_mean - grand_mean) ** 2 for subject_mean in subject_means)
        / (n_subjects - 1)
    )
    ms_rater = (
        n_subjects
        * sum((rater_mean - grand_mean) ** 2 for rater_mean in rater_means)
        / (n_raters - 1)
    )
    residual_sum = 0.0
    for subject_index, unit in enumerate(complete_units):
        for rater_index, value in enumerate(unit):
            residual_sum += (
                value
                - subject_means[subject_index]
                - rater_means[rater_index]
                + grand_mean
            ) ** 2
    ms_error = residual_sum / ((n_subjects - 1) * (n_raters - 1))

    denominator = (
        ms_subject
        + (n_raters - 1) * ms_error
        + n_raters * (ms_rater - ms_error) / n_subjects
    )
    if denominator == 0:
        return None
    return (ms_subject - ms_error) / denominator


def rounded(value: float | None) -> float | None:
    if value is None:
        return None
    return round(value, 6)


def distribution(values: list[str]) -> dict[str, int]:
    return dict(sorted(Counter(values).items(), key=lambda item: item[0]))


def confusion(values_a: list[str], values_b: list[str], categories: list[str]) -> dict[str, dict[str, int]]:
    matrix: dict[str, dict[str, int]] = {
        left: {right: 0 for right in categories}
        for left in categories
    }
    for left, right in zip(values_a, values_b, strict=True):
        matrix[left][right] += 1
    return matrix


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def median(values: list[float]) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    midpoint = len(sorted_values) // 2
    if len(sorted_values) % 2:
        return sorted_values[midpoint]
    return (sorted_values[midpoint - 1] + sorted_values[midpoint]) / 2


def comparable_pairs(
    left_records: dict[tuple[str, ...], dict[str, Any]],
    right_records: dict[tuple[str, ...], dict[str, Any]],
    common_keys: list[tuple[str, ...]],
    field: str,
) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for key in common_keys:
        left_value = left_records[key]["values"].get(field)
        right_value = right_records[key]["values"].get(field)
        if left_value is None or right_value is None:
            continue
        pairs.append(
            {
                "key": key,
                "left_value": left_value,
                "right_value": right_value,
            }
        )
    return pairs


def build_results(project_7: Path, project_11: Path) -> dict[str, Any]:
    left = index_export(project_7, "project_7")
    right = index_export(project_11, "project_11")
    left_records = left["records"]
    right_records = right["records"]

    left_keys = set(left_records)
    right_keys = set(right_records)
    common_keys = sorted(left_keys & right_keys)

    all_fields = sorted(
        {
            field
            for key in common_keys
            for field in set(left_records[key]["values"]) | set(right_records[key]["values"])
        }
    )

    comparable_fields: dict[str, Any] = {}
    pooled_continuous_units: list[list[float]] = []
    pooled_continuous_left: list[str] = []
    pooled_continuous_right: list[str] = []
    pooled_binary_left: list[str] = []
    pooled_binary_right: list[str] = []
    missing_pairs: list[dict[str, Any]] = []

    for field in all_fields:
        pairs = comparable_pairs(left_records, right_records, common_keys, field)
        observed_values = [
            value
            for pair in pairs
            for value in (pair["left_value"], pair["right_value"])
        ]
        field_type = infer_field_type(observed_values)

        missing_count = len(common_keys) - len(pairs)
        if missing_count:
            missing_pairs.append(
                {
                    "field": field,
                    "missing_pair_count": missing_count,
                    "missing_keys": [
                        display_key(key)
                        for key in common_keys
                        if left_records[key]["values"].get(field) is None
                        or right_records[key]["values"].get(field) is None
                    ],
                }
            )

        left_values = [pair["left_value"] for pair in pairs]
        right_values = [pair["right_value"] for pair in pairs]
        exact_agreement = (
            sum(1 for a_value, b_value in zip(left_values, right_values, strict=True) if a_value == b_value)
            / len(pairs)
            if pairs
            else None
        )

        field_result: dict[str, Any] = {
            "field_type": field_type,
            "n_pairs": len(pairs),
            "missing_pair_count": missing_count,
            "exact_agreement": rounded(exact_agreement),
            "project_7_distribution": distribution(left_values),
            "project_11_distribution": distribution(right_values),
        }

        if field_type == "continuous":
            numeric_pairs = [
                (parse_number(pair["left_value"]), parse_number(pair["right_value"]))
                for pair in pairs
            ]
            units = [
                [left_value, right_value]
                for left_value, right_value in numeric_pairs
                if left_value is not None and right_value is not None
            ]
            absolute_differences = [abs(unit[0] - unit[1]) for unit in units]
            signed_differences = [unit[1] - unit[0] for unit in units]
            pooled_continuous_units.extend(units)
            pooled_continuous_left.extend(left_values)
            pooled_continuous_right.extend(right_values)

            field_result.update(
                {
                    "metric": "krippendorff_alpha_interval",
                    "krippendorff_alpha": rounded(krippendorff_alpha_interval(units)),
                    "gwet_ac2_quadratic": rounded(
                        gwet_ac2(left_values, right_values, ORDINAL_CATEGORIES)
                    ),
                    "ordinal_weighted_kappa_quadratic": rounded(
                        weighted_kappa(left_values, right_values, ORDINAL_CATEGORIES)
                    ),
                    "icc_a1_absolute_agreement": rounded(icc_absolute_agreement(units)),
                    "within_one_point_agreement": rounded(within_tolerance(units, 1.0)),
                    "mean_absolute_difference": rounded(mean(absolute_differences)),
                    "median_absolute_difference": rounded(median(absolute_differences)),
                    "mean_project_11_minus_project_7": rounded(mean(signed_differences)),
                }
            )
        elif field_type == "binary":
            pooled_binary_left.extend(left_values)
            pooled_binary_right.extend(right_values)
            field_result.update(
                {
                    "metric": "weighted_kappa_quadratic",
                    "weighted_kappa": rounded(weighted_kappa(left_values, right_values, BINARY_CATEGORIES)),
                    "gwet_ac1": rounded(gwet_ac1(left_values, right_values, BINARY_CATEGORIES)),
                    "pabak": rounded(pabak(exact_agreement)),
                    "matthews_correlation_coefficient": rounded(
                        matthews_correlation_coefficient(left_values, right_values)
                    ),
                    "positive_agreement": rounded(positive_agreement(left_values, right_values)),
                    "negative_agreement": rounded(negative_agreement(left_values, right_values)),
                    "confusion_project_7_rows_project_11_columns": confusion(
                        left_values,
                        right_values,
                        BINARY_CATEGORIES,
                    ),
                }
            )

        comparable_fields[field] = field_result

    continuous_alphas = [
        field["krippendorff_alpha"]
        for field in comparable_fields.values()
        if field.get("field_type") == "continuous" and field.get("krippendorff_alpha") is not None
    ]
    continuous_ac2s = [
        field["gwet_ac2_quadratic"]
        for field in comparable_fields.values()
        if field.get("field_type") == "continuous" and field.get("gwet_ac2_quadratic") is not None
    ]
    continuous_kappas = [
        field["ordinal_weighted_kappa_quadratic"]
        for field in comparable_fields.values()
        if field.get("field_type") == "continuous"
        and field.get("ordinal_weighted_kappa_quadratic") is not None
    ]
    continuous_iccs = [
        field["icc_a1_absolute_agreement"]
        for field in comparable_fields.values()
        if field.get("field_type") == "continuous"
        and field.get("icc_a1_absolute_agreement") is not None
    ]
    continuous_within_one = [
        field["within_one_point_agreement"]
        for field in comparable_fields.values()
        if field.get("field_type") == "continuous"
        and field.get("within_one_point_agreement") is not None
    ]
    binary_kappas = [
        field["weighted_kappa"]
        for field in comparable_fields.values()
        if field.get("field_type") == "binary" and field.get("weighted_kappa") is not None
    ]
    binary_ac1s = [
        field["gwet_ac1"]
        for field in comparable_fields.values()
        if field.get("field_type") == "binary" and field.get("gwet_ac1") is not None
    ]
    binary_pabaks = [
        field["pabak"]
        for field in comparable_fields.values()
        if field.get("field_type") == "binary" and field.get("pabak") is not None
    ]
    binary_mccs = [
        field["matthews_correlation_coefficient"]
        for field in comparable_fields.values()
        if field.get("field_type") == "binary"
        and field.get("matthews_correlation_coefficient") is not None
    ]
    binary_positive_agreements = [
        field["positive_agreement"]
        for field in comparable_fields.values()
        if field.get("field_type") == "binary" and field.get("positive_agreement") is not None
    ]
    binary_negative_agreements = [
        field["negative_agreement"]
        for field in comparable_fields.values()
        if field.get("field_type") == "binary" and field.get("negative_agreement") is not None
    ]

    pooled_continuous_exact = (
        sum(
            1
            for left_value, right_value in zip(
                pooled_continuous_left,
                pooled_continuous_right,
                strict=True,
            )
            if left_value == right_value
        )
        / len(pooled_continuous_left)
        if pooled_continuous_left
        else None
    )
    pooled_continuous_abs_diffs = [
        abs(unit[0] - unit[1])
        for unit in pooled_continuous_units
    ]
    pooled_binary_exact = (
        sum(
            1
            for left_value, right_value in zip(pooled_binary_left, pooled_binary_right, strict=True)
            if left_value == right_value
        )
        / len(pooled_binary_left)
        if pooled_binary_left
        else None
    )

    comparable_task_count = sum(
        1
        for key in common_keys
        if left_records[key]["values"] and right_records[key]["values"]
    )

    results = {
        "metadata": {
            "generated_at_utc": dt.datetime.now(dt.UTC).replace(microsecond=0).isoformat(),
            "script": Path(__file__).name,
            "source_files": [project_7.name, project_11.name],
            "alignment_key_fields": ALIGNMENT_KEY_FIELDS,
            "continuous_metric": "Krippendorff alpha with interval distance",
            "binary_metric": "Cohen weighted kappa with quadratic weights",
        },
        "inputs": {
            "project_7": {
                "path": Path(left["path"]).name,
                "task_count": left["task_count"],
                "annotations_per_task": left["annotations_per_task"],
                "annotator_ids": left["annotator_ids"],
                "nonempty_annotation_count": left["nonempty_annotation_count"],
            },
            "project_11": {
                "path": Path(right["path"]).name,
                "task_count": right["task_count"],
                "annotations_per_task": right["annotations_per_task"],
                "annotator_ids": right["annotator_ids"],
                "nonempty_annotation_count": right["nonempty_annotation_count"],
            },
        },
        "alignment": {
            "project_7_task_count": len(left_keys),
            "project_11_task_count": len(right_keys),
            "common_task_count": len(common_keys),
            "project_7_only_task_count": len(left_keys - right_keys),
            "project_11_only_task_count": len(right_keys - left_keys),
            "comparable_task_count_with_nonempty_annotations": comparable_task_count,
        },
        "pooled": {
            "continuous": {
                "n_label_pairs": len(pooled_continuous_units),
                "metric": "krippendorff_alpha_interval",
                "krippendorff_alpha": rounded(krippendorff_alpha_interval(pooled_continuous_units)),
                "macro_mean_krippendorff_alpha": rounded(mean(continuous_alphas)),
                "gwet_ac2_quadratic": rounded(
                    gwet_ac2(pooled_continuous_left, pooled_continuous_right, ORDINAL_CATEGORIES)
                ),
                "macro_mean_gwet_ac2_quadratic": rounded(mean(continuous_ac2s)),
                "ordinal_weighted_kappa_quadratic": rounded(
                    weighted_kappa(pooled_continuous_left, pooled_continuous_right, ORDINAL_CATEGORIES)
                ),
                "macro_mean_ordinal_weighted_kappa_quadratic": rounded(mean(continuous_kappas)),
                "icc_a1_absolute_agreement": rounded(icc_absolute_agreement(pooled_continuous_units)),
                "macro_mean_icc_a1_absolute_agreement": rounded(mean(continuous_iccs)),
                "exact_agreement": rounded(pooled_continuous_exact),
                "within_one_point_agreement": rounded(
                    within_tolerance(pooled_continuous_units, 1.0)
                ),
                "macro_mean_within_one_point_agreement": rounded(mean(continuous_within_one)),
                "mean_absolute_difference": rounded(mean(pooled_continuous_abs_diffs)),
            },
            "binary": {
                "n_label_pairs": len(pooled_binary_left),
                "metric": "weighted_kappa_quadratic",
                "weighted_kappa": rounded(
                    weighted_kappa(pooled_binary_left, pooled_binary_right, BINARY_CATEGORIES)
                ),
                "macro_mean_weighted_kappa": rounded(mean(binary_kappas)),
                "gwet_ac1": rounded(gwet_ac1(pooled_binary_left, pooled_binary_right, BINARY_CATEGORIES)),
                "macro_mean_gwet_ac1": rounded(mean(binary_ac1s)),
                "pabak": rounded(pabak(pooled_binary_exact)),
                "macro_mean_pabak": rounded(mean(binary_pabaks)),
                "matthews_correlation_coefficient": rounded(
                    matthews_correlation_coefficient(pooled_binary_left, pooled_binary_right)
                ),
                "macro_mean_matthews_correlation_coefficient": rounded(mean(binary_mccs)),
                "exact_agreement": rounded(pooled_binary_exact),
                "positive_agreement": rounded(
                    positive_agreement(pooled_binary_left, pooled_binary_right)
                ),
                "macro_mean_positive_agreement": rounded(mean(binary_positive_agreements)),
                "negative_agreement": rounded(
                    negative_agreement(pooled_binary_left, pooled_binary_right)
                ),
                "macro_mean_negative_agreement": rounded(mean(binary_negative_agreements)),
            },
        },
        "fields": comparable_fields,
        "data_quality": {
            "empty_annotations": left["empty_annotations"] + right["empty_annotations"],
            "duplicate_task_keys": left["duplicate_task_keys"] + right["duplicate_task_keys"],
            "duplicate_annotation_fields": left["duplicate_annotation_fields"]
            + right["duplicate_annotation_fields"],
            "missing_pairs": missing_pairs,
        },
    }
    return results


def format_metric(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.3f}"


def field_label(field: str) -> str:
    return field.replace("_", " ")


def write_summary(results: dict[str, Any], output_md: Path, output_json: Path) -> None:
    fields = results["fields"]
    continuous_fields = [
        (field, details)
        for field, details in fields.items()
        if details.get("field_type") == "continuous"
    ]
    binary_fields = [
        (field, details)
        for field, details in fields.items()
        if details.get("field_type") == "binary"
    ]

    lines: list[str] = []
    lines.append("# Label Studio IAA summary")
    lines.append("")
    lines.append("## Inputs")
    lines.append("")
    for input_name, input_details in results["inputs"].items():
        lines.append(f"{input_name}: `{Path(input_details['path']).name}`")
        lines.append(f"Tasks: {input_details['task_count']}")
        lines.append(f"Annotator ids: `{json.dumps(input_details['annotator_ids'], sort_keys=True)}`")
        lines.append("")

    alignment = results["alignment"]
    lines.append("## Alignment")
    lines.append("")
    lines.append(f"Common tasks: {alignment['common_task_count']}")
    lines.append(f"Comparable tasks with nonempty annotations in both exports: {alignment['comparable_task_count_with_nonempty_annotations']}")
    lines.append(f"Project 7 only tasks: {alignment['project_7_only_task_count']}")
    lines.append(f"Project 11 only tasks: {alignment['project_11_only_task_count']}")
    lines.append("")

    lines.append("## Methods")
    lines.append("")
    lines.append("Continuous 1 to 5 ratings were scored with Krippendorff alpha using interval distance.")
    lines.append("Continuous fields also include Gwet AC2, ordinal quadratic weighted kappa, ICC(A,1), exact agreement, within one point agreement, mean absolute difference, and average signed difference.")
    lines.append("Binary Yes or No fields were scored with Cohen weighted kappa using quadratic weights.")
    lines.append("Binary fields also include Gwet AC1, PABAK, MCC, positive agreement, and negative agreement.")
    lines.append("The pooled rows treat each task by field cell as one paired label.")
    lines.append("")

    pooled = results["pooled"]
    lines.append("## Pooled results")
    lines.append("")
    lines.append("Continuous pooled metrics:")
    lines.append("")
    lines.append("<table>")
    lines.append("<tr><th>Metric</th><th>N label pairs</th><th>Pooled value</th><th>Macro mean</th></tr>")
    lines.append(
        "<tr><td>Krippendorff alpha</td>"
        f"<td>{pooled['continuous']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['continuous']['krippendorff_alpha'])}</td>"
        f"<td>{format_metric(pooled['continuous']['macro_mean_krippendorff_alpha'])}</td></tr>"
    )
    lines.append(
        "<tr><td>Gwet AC2</td>"
        f"<td>{pooled['continuous']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['continuous']['gwet_ac2_quadratic'])}</td>"
        f"<td>{format_metric(pooled['continuous']['macro_mean_gwet_ac2_quadratic'])}</td></tr>"
    )
    lines.append(
        "<tr><td>Ordinal weighted kappa</td>"
        f"<td>{pooled['continuous']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['continuous']['ordinal_weighted_kappa_quadratic'])}</td>"
        f"<td>{format_metric(pooled['continuous']['macro_mean_ordinal_weighted_kappa_quadratic'])}</td></tr>"
    )
    lines.append(
        "<tr><td>ICC(A,1)</td>"
        f"<td>{pooled['continuous']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['continuous']['icc_a1_absolute_agreement'])}</td>"
        f"<td>{format_metric(pooled['continuous']['macro_mean_icc_a1_absolute_agreement'])}</td></tr>"
    )
    lines.append(
        "<tr><td>Exact agreement</td>"
        f"<td>{pooled['continuous']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['continuous']['exact_agreement'])}</td>"
        "<td>NA</td></tr>"
    )
    lines.append(
        "<tr><td>Within one point agreement</td>"
        f"<td>{pooled['continuous']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['continuous']['within_one_point_agreement'])}</td>"
        f"<td>{format_metric(pooled['continuous']['macro_mean_within_one_point_agreement'])}</td></tr>"
    )
    lines.append(
        "<tr><td>Mean absolute difference</td>"
        f"<td>{pooled['continuous']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['continuous']['mean_absolute_difference'])}</td>"
        "<td>NA</td></tr>"
    )
    lines.append("</table>")
    lines.append("")
    lines.append("Binary pooled metrics:")
    lines.append("")
    lines.append("<table>")
    lines.append("<tr><th>Metric</th><th>N label pairs</th><th>Pooled value</th><th>Macro mean</th></tr>")
    lines.append(
        "<tr><td>Weighted kappa</td>"
        f"<td>{pooled['binary']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['binary']['weighted_kappa'])}</td>"
        f"<td>{format_metric(pooled['binary']['macro_mean_weighted_kappa'])}</td></tr>"
    )
    lines.append(
        "<tr><td>Gwet AC1</td>"
        f"<td>{pooled['binary']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['binary']['gwet_ac1'])}</td>"
        f"<td>{format_metric(pooled['binary']['macro_mean_gwet_ac1'])}</td></tr>"
    )
    lines.append(
        "<tr><td>PABAK</td>"
        f"<td>{pooled['binary']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['binary']['pabak'])}</td>"
        f"<td>{format_metric(pooled['binary']['macro_mean_pabak'])}</td></tr>"
    )
    lines.append(
        "<tr><td>MCC</td>"
        f"<td>{pooled['binary']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['binary']['matthews_correlation_coefficient'])}</td>"
        f"<td>{format_metric(pooled['binary']['macro_mean_matthews_correlation_coefficient'])}</td></tr>"
    )
    lines.append(
        "<tr><td>Exact agreement</td>"
        f"<td>{pooled['binary']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['binary']['exact_agreement'])}</td>"
        "<td>NA</td></tr>"
    )
    lines.append(
        "<tr><td>Positive agreement</td>"
        f"<td>{pooled['binary']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['binary']['positive_agreement'])}</td>"
        f"<td>{format_metric(pooled['binary']['macro_mean_positive_agreement'])}</td></tr>"
    )
    lines.append(
        "<tr><td>Negative agreement</td>"
        f"<td>{pooled['binary']['n_label_pairs']}</td>"
        f"<td>{format_metric(pooled['binary']['negative_agreement'])}</td>"
        f"<td>{format_metric(pooled['binary']['macro_mean_negative_agreement'])}</td></tr>"
    )
    lines.append("</table>")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(f"Coverage is complete: {alignment['common_task_count']} common tasks and {alignment['comparable_task_count_with_nonempty_annotations']} paired nonempty annotations.")
    lines.append(f"Continuous ratings show stronger coarse agreement than exact agreement: pooled Gwet AC2 is {format_metric(pooled['continuous']['gwet_ac2_quadratic'])}, within one point agreement is {format_metric(pooled['continuous']['within_one_point_agreement'])}, and exact agreement is {format_metric(pooled['continuous']['exact_agreement'])}.")
    lines.append(f"Field level continuous reliability remains uneven: macro mean Krippendorff alpha is {format_metric(pooled['continuous']['macro_mean_krippendorff_alpha'])}, macro mean Gwet AC2 is {format_metric(pooled['continuous']['macro_mean_gwet_ac2_quadratic'])}, and macro mean ICC(A,1) is {format_metric(pooled['continuous']['macro_mean_icc_a1_absolute_agreement'])}.")
    lines.append(f"Binary labels have high raw agreement under class imbalance: exact agreement is {format_metric(pooled['binary']['exact_agreement'])}, Gwet AC1 is {format_metric(pooled['binary']['gwet_ac1'])}, and PABAK is {format_metric(pooled['binary']['pabak'])}.")
    lines.append(f"Minority class reliability is weaker: macro mean positive agreement is {format_metric(pooled['binary']['macro_mean_positive_agreement'])}, macro mean negative agreement is {format_metric(pooled['binary']['macro_mean_negative_agreement'])}, and macro mean MCC is {format_metric(pooled['binary']['macro_mean_matthews_correlation_coefficient'])}.")
    lines.append("Overall, the IAA supports complete audit coverage and usable coarse binary audit labels, with weaker evidence for fine grained field level ordinal claims.")
    lines.append("")

    lines.append("## Continuous fields")
    lines.append("")
    lines.append("<table>")
    lines.append("<tr><th>Field</th><th>N pairs</th><th>Alpha</th><th>Gwet AC2</th><th>Ordinal weighted kappa</th><th>ICC(A,1)</th><th>Exact</th><th>Within one point</th><th>Mean absolute difference</th><th>Project 11 minus Project 7</th></tr>")
    for field, details in continuous_fields:
        lines.append(
            f"<tr><td>{field_label(field)}</td>"
            f"<td>{details['n_pairs']}</td>"
            f"<td>{format_metric(details.get('krippendorff_alpha'))}</td>"
            f"<td>{format_metric(details.get('gwet_ac2_quadratic'))}</td>"
            f"<td>{format_metric(details.get('ordinal_weighted_kappa_quadratic'))}</td>"
            f"<td>{format_metric(details.get('icc_a1_absolute_agreement'))}</td>"
            f"<td>{format_metric(details.get('exact_agreement'))}</td>"
            f"<td>{format_metric(details.get('within_one_point_agreement'))}</td>"
            f"<td>{format_metric(details.get('mean_absolute_difference'))}</td>"
            f"<td>{format_metric(details.get('mean_project_11_minus_project_7'))}</td></tr>"
        )
    lines.append("</table>")
    lines.append("")

    lines.append("## Binary fields")
    lines.append("")
    lines.append("<table>")
    lines.append("<tr><th>Field</th><th>N pairs</th><th>Weighted kappa</th><th>Gwet AC1</th><th>PABAK</th><th>MCC</th><th>Exact</th><th>Positive agreement</th><th>Negative agreement</th><th>Project 7 distribution</th><th>Project 11 distribution</th></tr>")
    for field, details in binary_fields:
        lines.append(
            f"<tr><td>{field_label(field)}</td>"
            f"<td>{details['n_pairs']}</td>"
            f"<td>{format_metric(details.get('weighted_kappa'))}</td>"
            f"<td>{format_metric(details.get('gwet_ac1'))}</td>"
            f"<td>{format_metric(details.get('pabak'))}</td>"
            f"<td>{format_metric(details.get('matthews_correlation_coefficient'))}</td>"
            f"<td>{format_metric(details.get('exact_agreement'))}</td>"
            f"<td>{format_metric(details.get('positive_agreement'))}</td>"
            f"<td>{format_metric(details.get('negative_agreement'))}</td>"
            f"<td>{json.dumps(details['project_7_distribution'], sort_keys=True)}</td>"
            f"<td>{json.dumps(details['project_11_distribution'], sort_keys=True)}</td></tr>"
        )
    lines.append("</table>")
    lines.append("")

    quality = results["data_quality"]
    lines.append("## Data notes")
    lines.append("")
    lines.append(f"Empty annotations: {len(quality['empty_annotations'])}")
    lines.append(f"Duplicate task keys: {len(quality['duplicate_task_keys'])}")
    lines.append(f"Duplicate annotation fields: {len(quality['duplicate_annotation_fields'])}")
    lines.append(f"Fields with missing paired labels: {len(quality['missing_pairs'])}")
    if quality["empty_annotations"]:
        lines.append("")
        lines.append("Empty annotation details:")
        lines.append("")
        lines.append("<table>")
        lines.append("<tr><th>Export</th><th>Task id</th><th>Annotation id</th><th>Source transcript</th></tr>")
        for item in quality["empty_annotations"]:
            lines.append(
                f"<tr><td>{item['export']}</td>"
                f"<td>{item['task_id']}</td>"
                f"<td>{item['annotation_id']}</td>"
                f"<td>{item['key'].get('source_transcript_id')}</td></tr>"
            )
        lines.append("</table>")
    lines.append("")

    lines.append("## Artifacts")
    lines.append("")
    lines.append(f"JSON results: `{output_json.name}`")
    lines.append(f"Calculation script: `{Path(__file__).name}`")
    lines.append("")

    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute IAA for two Label Studio exports.")
    parser.add_argument("--project-7", type=Path, default=DEFAULT_PROJECT_7)
    parser.add_argument("--project-11", type=Path, default=DEFAULT_PROJECT_11)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = build_results(args.project_7, args.project_11)
    args.output_json.write_text(
        json.dumps(results, indent=2, ensure_ascii=True, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_summary(results, args.output_md, args.output_json)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
