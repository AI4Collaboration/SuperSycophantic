#!/usr/bin/env python3
"""Build the mixed GT source panel.

The output is the source JSONL consumed by build_supersycophantic_context_panels.py.
It keeps the target shape at 25 MMLU-Pro and 25 HLE-Verified items per GT
domain while applying the saturated-screening and agent-audit exclusions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


DATA_DIR = Path(__file__).resolve().parents[1]
MMLU_SELECTED_PATH = DATA_DIR / "mmlu_pro_saturated_gt_200.jsonl"
HLE_DATA_DIR = DATA_DIR / "hle_verified" / "data"
OUTPUT_PATH = DATA_DIR / "supersycophantic_mixed_gt_200_candidate.jsonl"

MMLU_PRO_DATASET = "TIGER-Lab/MMLU-Pro"
MMLU_PRO_SOURCE_URL = "https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro"
HLE_DATASET = "skylenage-ai/HLE-Verified"
HLE_SOURCE_URL = "https://huggingface.co/datasets/skylenage-ai/HLE-Verified"

DOMAINS = [
    "Mathematical Science",
    "Physical Science",
    "Chemical Science",
    "Biomedical Science",
]

MMLU_CATEGORY_BY_DOMAIN = {
    "Mathematical Science": {"math"},
    "Physical Science": {"physics"},
    "Chemical Science": {"chemistry"},
    "Biomedical Science": {"health"},
}

HLE_SUBJECTS_BY_DOMAIN = {
    "Mathematical Science": {
        "Mathematics",
        "Applied Mathematics",
    },
    "Physical Science": {
        "Acoustics",
        "Aerospace Engineering",
        "Atmospheric Physics",
        "Bioeletronics",
        "Biomedical Engineering",
        "Biophysics",
        "Calculus Engineering",
        "Chemical Engineering",
        "Classical Physics",
        "Civil Engineering",
        "Condensed Matter Physics",
        "Crystallography",
        "Earth Science",
        "Engineering Physics",
        "Physics",
        "Foremost Quantum: Particle Physics",
        "Geography",
        "Geology",
        "Geophyics/Geodynamics",
        "High Energy Physics And Nuclear Physics",
        "Astronomy",
        "Electrical Engineering",
        "Industrial Engineering",
        "Lighting Design",
        "Materials Science",
        "Mechanical Engineering",
        "Mathematical Physics: Odes",
        "Nanotechnology",
        "Nuclear Physics",
        "Nuclear Science",
        "Oceanography",
        "Photonics",
        "Process Optimization",
        "Quantum Computing",
        "Quantum And Classical Physics Combined",
        "Quantum Logic",
        "Quantum Physics",
        "Remote Sensing",
        "Robotics",
        "Wireless Communication",
    },
    # Chemical Science includes molecular and biochemical measurement questions,
    # but excludes ecology, general organismal biology, and clinical medicine.
    "Chemical Science": {
        "Chemistry",
        "Computational Chemistry",
        "Biochemistry",
    },
    "Biomedical Science": {
        "Medicine",
        "Pediatrics",
        "Orthopedics",
        "Anatomy",
        "Pathology",
        "Physical Medicine And Rehabilitation",
        "Physiology",
        "Microbiology",
        "Immunology",
        "Neuroscience",
        "Genomics",
        "Genetics",
        "Molecular Genetics",
        "Molecular Biology",
        "Biochemistry",
        "Bioinformatics",
        "Computational Biology",
    },
}

SUBSET_PRIORITY = {
    "Gold subset": 0,
    "Revision subset": 1,
}

# Source record IDs from Opus/adversarial review and human-agent audit.
EXCLUDED_SOURCE_RECORD_IDS = {
    # MMLU-Pro source rows.
    "8271",
    "8436",
    "9204",
    "6481",
    "6013",
    "8714",
    "4337",
    "3804",
    "4164",
    "7801",
    "9165",
    "8667",
    "6200",
    # Mathematical HLE.
    "67352e9911e5510fc618f619",
    "673668e658bad7ba89d4ad54",
    "671dc2145167d65f41c86c58",
    "670e92583011a5b80bfb6c60",
    "6725fe6b26992c47ce3a7ef5",
    "673b0fb5fa1e03dfc8513c37",
    "670f241acb7ead88385e0ca0",
    "672614260019e08d7f82b2d0",
    "671aee77ed3d54e87368bc9a",
    "66eadaf22b7de7dec0a046bd",
    # Physical HLE.
    "677da0a433769e54d305f23c",
    "671d97e729e7fde7166e4743",
    "671f1f0bb0b665acec70c3aa",
    "67225b0a9e5897be2aec5257",
    "6721ad631648dda151c2a7ec",
    "671eeb53c1a668a6c81e5993",
    "66ed5f1e85adbeda9f978022",
    "671f7f334db66145d9e41f1f",
    "671ee933019b32e00d827382",
    "671f0b0c7301fac39660e7a3",
    "67229b1f5a95bf7d096a6319",
    "67d317cab57b67a3417a4969",
    "673497b017a9687889930ac4",
    "671ee72aec85f81abd7a2f92",
    "6720a9feec461e4c6a4e2c3a",
    # Chemical HLE.
    "6723a613f747d32c6b0b65dc",
    "671bc0c855449c636f4bbd36",
    "670edc9dbddc0cfe673272c8",
    "66fcbff58d68a0bf3cafb696",
    "673ad4214ff8ec86c1487ed3",
    "672c033ff576aed47449d75f",
    "6720cd0acf47ec0733864dd8",
    "6730f3c006cd33fe46ca2dfe",
    "671d9e9d29e7fde7166e474d",
    "671d91bcad7fb0793a0e93bd",
    "6733fe294a614b25db9b204a",
    "671808958b88f01935b5825a",
    # Biomedical HLE.
    "67666a593e38774fc651f9f3",
    "66ef3de8b8a1ba6e0ba23498",
    "671f40348b7ca154256661c6",
    "6766662d37a605e5d7cb9ee8",
    "671f0c2578ab3005e439aeba",
    "671f2b0ee38f776acdad8aa1",
    "671f083dc8da11076ce9960e",
    "66faccfb44cb2f3b0e1be0ff",
    "67242f1f911674ab1b5d904b",
    "672179e4c8cc0ac89395e8d0",
    "67256e36e5e6cc87293fc3f0",
    # Second-pass GT content audit exclusions.
    "66f2dee46721a56e35d20300",
    "671fee7b3fdf9b50dcb7ca58",
    "671fec80cee768cca0b65e5a",
    "671faf8171f6aa7bd1e604cd",
    "671bd4fb69d17f19519341dc",
    "6734989917a9687889930ac9",
    "66f31140cdc5dafd297e9b5a",
    "671bea6afd01315eb4f1c376",
    "6731fa3bb08cf72258a21348",
    "671a947c9deaf77048dcc8b7",
    "6723fc382a5a297be25c2d79",
    # Third-pass replacement-row audit exclusions.
    "671ee48a3b4aac040ec0bd85",
    "671ef8426edc2afd69958982",
    "671f07b1d0f22fd6e50482e7",
    "671fadb2272218e5d336a33f",
    "6734d0fd45aa0757a21c4ddc",
    "671fbb0cc6abf8266c1892ca",
    "672de1afed6de72b75b8c7e6",
    "6734956467d2904eebed3a09",
    "672ddd9bff7bf1483f564046",
    "671eefbfb6d7145231fa28e4",
    "67155ca2675b57d8bbc9144d",
    "673a6a6c4c465c371379b670",
    "670010ff77983eff561814b2",
    "672403a54850f72c4c930d47",
    "672f0ac02ccf8890167533ec",
    "6738373cf2df805520bc86ce",
    "6722f2b2f3aeb8d4f9cf83ab",
    # Fourth-pass full-panel audit exclusions.
    "6738f778851b80b033aa8654",
    "66eb894e6feeaea27b557782",
    "672e09b50a85795d0ed2d36e",
    "67225f3cf135fd983a87bc1f",
    "6734968f832777944c775cc4",
    "67258c4124b290d28a9f8abe",
    "670013827794cc36bc974f27",
    "66ea29ab7131229007ccbd9d",
    "673a8c2e4e2e35b51a27fafd",
    "6736a9b0b949d548db8da965",
    "672eff1c72cfb81d78f82cdb",
    "66fd469e7a33ca69ccc69729",
    "66ea1dd348eb93c2aef1c735",
    "67257fe9be53ed439b973ff9",
    "66ec02c52ec65d6153428744",
    # Fifth-pass full-panel audit exclusions.
    "672f4434e9c13daba078d693",
    "671d6502b996cf9936d1afd0",
    "66eaa5ddc7a3252f0f3fe53f",
    "670f289fb671096a201efee4",
    "670d5ce6d57c80b4d4090cb4",
    "67383288f2df805520bc86b5",
    "67253c33ad98e94b47ea3477",
    "672416e85247738dfbb432fa",
    "672235a01e36ca7369b8b157",
}

MMLU_SLOT_REPLACEMENT_RECORD_IDS = {
    # Keep item ids stable while replacing source rows found non-unique under
    # trace audit.
    "7801": "9015",
    "9165": "9253",
    "8667": "8173",
    "6200": "6257",
}

HLE_SLOT_REPLACEMENT_RECORD_IDS_BY_DOMAIN = {
    # Keep HLE item ids stable while replacing rows found non-unique,
    # tolerance-ambiguous, or clinically overbroad in the full-panel audit.
    "Physical Science": {
        "672f4434e9c13daba078d693": "66f8cff8469c315e2c9ed2f6",
        "671d6502b996cf9936d1afd0": "676433a3921b0ce4e14c557f",
        "66eaa5ddc7a3252f0f3fe53f": "67390213fc9dc4f5102ad835",
    },
    "Chemical Science": {
        "670f289fb671096a201efee4": "67241711454d2eee0ceef767",
    },
    "Biomedical Science": {
        "670d5ce6d57c80b4d4090cb4": "66ee93ba02314f06feb186cf",
        "67383288f2df805520bc86b5": "66fe596c0eeb4521791d31ef",
        "67253c33ad98e94b47ea3477": "67018145e8c125b41caa9ee3",
        "672416e85247738dfbb432fa": "676727c0c393c4ff629cb82b",
    },
}

CHEMICAL_EXTRA_HLE_RECORD_IDS = {
    # HLE's pure chemistry MC supply is sparse after full-panel audit
    # exclusions. These manually screened rows are molecular/biochemical or
    # chemical-engineering questions rather than clinical biomedical items.
    "66ec02c52ec65d6153428744",
    "6724db7012a8d5ea6a619372",
    "66fe596c0eeb4521791d31ef",
    "66ff063787bfb80443d02df6",
    "67018145e8c125b41caa9ee3",
    "67241711454d2eee0ceef767",
    "67240e5786f4c71a70e0b499",
    "672500151d07f0962c8993d7",
}

PHYSICAL_EXTRA_HLE_RECORD_IDS = {
    # HLE's physics-native MC supply is sparse after full-panel audit
    # exclusions. Physical Science is allowed to draw from adjacent subjects
    # when the item itself is about a physical mechanism, measurement,
    # material process, or kinematic/geophysical inference.
    "67223944bb174709043a2641",
    "67204844452d0bd5546e11ef",
    "67217f97262eafa82562cc2b",
    "676433a3921b0ce4e14c557f",
}

BAD_TEXT_PATTERNS = [
    "\u9225",
    "\u63b3",
    "\ufffd",
    "FWDH",
    "I the cancer",
    "incresed",
    "dates dates",
]

ANSWER_CHOICES_RE = re.compile(r"(?ims)^Answer choices:\s*(.+)$")
CHOICE_RE = re.compile(r"(?m)^([A-Z])\.\s+(.+?)(?=\n[A-Z]\.\s+|\Z)", re.S)
SIMPLE_NUMERIC_ANSWER_RE = re.compile(
    r"^\s*(?P<prefix>[$]?\s*)"
    r"(?P<number>[+-]?\d+(?:\.\d+)?)"
    r"(?P<suffix>\s*(?:[A-Za-z°%μµ/().^-]+(?:\s+[A-Za-z°%μµ/().^-]+)*)?)"
    r"\s*[$]?\s*$"
)
SIMPLE_LATEX_FRAC_RE = re.compile(
    r"^\s*[$]?\s*\\frac\{(?P<num>[+-]?\d+)\}\{(?P<den>\d+)\}\s*[$]?\s*$",
    re.I,
)
SIMPLE_SLASH_FRAC_RE = re.compile(r"^\s*(?P<num>[+-]?\d+)\s*/\s*(?P<den>\d+)\s*$")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    text = "\n".join(json.dumps(compact_row(row), ensure_ascii=False) for row in rows)
    path.write_text(text + "\n", encoding="utf-8")


def compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in row.items()
        if value is not None and value != {} and value != []
    }


def stable_int(*parts: object) -> int:
    key = "::".join(str(part) for part in parts)
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:16], 16)


def domain_slug(domain: str) -> str:
    return domain.upper().replace("&", "AND").replace(" ", "_")


def public_mmlu_category(category: object) -> str | None:
    value = str(category or "").strip()
    if value == "health":
        return "biomedical_science"
    return value or None


def normalize_text(text: object) -> str:
    return re.sub(r"\s+", " ", str(text)).strip()


def duplicated_choice_text(choices: dict[str, str]) -> bool:
    normalized = [normalize_text(value).lower() for value in choices.values()]
    return len(normalized) != len(set(normalized))


def has_bad_text(text: object) -> bool:
    value = str(text)
    return any(pattern in value for pattern in BAD_TEXT_PATTERNS)


def parse_hle_choices(question: str) -> tuple[str, dict[str, str]]:
    match = ANSWER_CHOICES_RE.search(question)
    if not match:
        return question.strip(), {}
    stem = question[: match.start()].strip()
    choice_block = match.group(1).strip()
    choices = {
        label: normalize_text(text)
        for label, text in CHOICE_RE.findall(choice_block)
    }
    return stem, choices


def format_decimal_like(value: float, decimals: int) -> str:
    if decimals == 0:
        return str(int(round(value)))
    return f"{value:.{decimals}f}"


def synthetic_numeric_mc(answer_text: str, record_id: str) -> tuple[dict[str, str], str] | None:
    answer = str(answer_text).strip()
    offsets = [-2, -1, 0, 1, 2]
    values: list[str]
    correct_text: str

    frac_match = SIMPLE_LATEX_FRAC_RE.match(answer)
    if frac_match:
        numerator = int(frac_match.group("num"))
        denominator = frac_match.group("den")
        values = [rf"\frac{{{numerator + offset}}}{{{denominator}}}" for offset in offsets]
        correct_text = rf"\frac{{{numerator}}}{{{denominator}}}"
    else:
        slash_match = SIMPLE_SLASH_FRAC_RE.match(answer)
        if slash_match:
            numerator = int(slash_match.group("num"))
            denominator = slash_match.group("den")
            values = [f"{numerator + offset}/{denominator}" for offset in offsets]
            correct_text = f"{numerator}/{denominator}"
        else:
            numeric_match = SIMPLE_NUMERIC_ANSWER_RE.match(answer)
            if not numeric_match:
                return None
            number_text = numeric_match.group("number")
            suffix = numeric_match.group("suffix") or ""
            # Avoid converting symbolic formulas that only begin with a number.
            if "\\" in suffix or any(ch in suffix for ch in "{}=<>"):
                return None
            decimals = len(number_text.split(".", 1)[1]) if "." in number_text else 0
            base_value = float(number_text)
            if decimals:
                step = 10 ** (-decimals)
            else:
                step = 1
            values = [
                f"{format_decimal_like(base_value + offset * step, decimals)}{suffix}"
                for offset in offsets
            ]
            correct_text = f"{format_decimal_like(base_value, decimals)}{suffix}"

    if len(set(values)) != 5 or correct_text not in values:
        return None

    ordered = list(values)
    shift = stable_int("synthetic_hle_numeric_mc_v1", record_id, correct_text) % len(ordered)
    ordered = ordered[shift:] + ordered[:shift]
    choices = {label: value for label, value in zip("ABCDE", ordered)}
    correct_label = next(label for label, value in choices.items() if value == correct_text)
    return choices, correct_label


def hle_has_image(row: pd.Series) -> bool:
    try:
        payload = json.loads(row["json"])
    except Exception:
        return False
    return bool(
        payload.get("image")
        or payload.get("image_preview")
        or payload.get("rationale_image")
    )


def source_quote(question: str, label: str, choices: dict[str, str]) -> str:
    return f"Question: {question} Correct answer ({label}): {choices[label]}"


def mmlu_source_rows() -> list[dict[str, Any]]:
    return read_jsonl(MMLU_SELECTED_PATH)


def hle_source_rows() -> list[dict[str, Any]]:
    parquet_paths = [
        path
        for path in sorted(HLE_DATA_DIR.glob("*.parquet"))
        if path.name.startswith(("Gold_subset.", "Revision_subset."))
    ]
    frames = [pd.read_parquet(path) for path in parquet_paths]
    if not frames:
        raise FileNotFoundError(f"No HLE parquet files found under {HLE_DATA_DIR}")
    df = pd.concat(frames, ignore_index=True)
    rows = []
    slot_replacement_sources = {
        source
        for replacements in HLE_SLOT_REPLACEMENT_RECORD_IDS_BY_DOMAIN.values()
        for source in replacements
    }
    hard_exclusions = EXCLUDED_SOURCE_RECORD_IDS - slot_replacement_sources
    for _, row in df.iterrows():
        record_id = str(row["id"])
        if record_id in hard_exclusions:
            continue
        if hle_has_image(row):
            continue
        question, choices = parse_hle_choices(str(row["question"]))
        answer_text = str(row["answer"]).strip()
        choice_source = "source_native_mc"
        answer = answer_text.upper()
        if choices and answer not in choices:
            continue
        if not choices:
            synthetic = synthetic_numeric_mc(answer_text, record_id)
            if not synthetic:
                continue
            choices, answer = synthetic
            choice_source = "synthetic_numeric_mc_from_exact_answer"
        if duplicated_choice_text(choices):
            continue
        if has_bad_text(row["question"]):
            continue
        rows.append(
            {
                "record_id": record_id,
                "native_id": record_id,
                "hle_verified_subset": str(row["Verified_Classes"]),
                "hle_native_category": str(row["category"]),
                "hle_raw_subject": str(row["raw_subject"]),
                "hle_choice_source": choice_source,
                "hle_original_answer": answer_text,
                "synthetic_mc_generation_rule": (
                    "five_options_numeric_offsets_around_verified_answer"
                    if choice_source == "synthetic_numeric_mc_from_exact_answer"
                    else None
                ),
                "question": question,
                "choices": choices,
                "correct_answer": answer,
            }
        )
    return rows


def select_mmlu_for_domain(domain: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    categories = MMLU_CATEGORY_BY_DOMAIN[domain]
    candidates = []
    slot_replacement_sources = set(MMLU_SLOT_REPLACEMENT_RECORD_IDS)
    hard_exclusions = EXCLUDED_SOURCE_RECORD_IDS - slot_replacement_sources
    for row in rows:
        record_id = str(row.get("record_id") or row.get("source") or row.get("id"))
        category = str(row.get("mmlu_pro_category") or "")
        if record_id in hard_exclusions or category not in categories:
            continue
        if str(row.get("correct_answer", "")).upper() not in row.get("choices", {}):
            continue
        candidate = dict(row)
        candidate["_record_id"] = record_id
        candidates.append(candidate)
    candidates.sort(
        key=lambda row: (
            stable_int("mmlu_mixed_gt_v2", domain, row["_record_id"]),
            row["_record_id"],
        )
    )
    if len(candidates) < 25:
        raise ValueError(f"{domain} has only {len(candidates)} eligible MMLU-Pro candidates")
    by_record_id = {row["_record_id"]: row for row in candidates}
    selected = []
    used_record_ids: set[str] = set()
    for candidate in candidates:
        record_id = candidate["_record_id"]
        replacement_id = MMLU_SLOT_REPLACEMENT_RECORD_IDS.get(record_id)
        if replacement_id:
            candidate = by_record_id.get(replacement_id)
            if candidate is None:
                raise ValueError(
                    f"{domain} replacement {replacement_id} is not an eligible MMLU-Pro candidate"
                )
            record_id = replacement_id
        if record_id in used_record_ids:
            continue
        selected.append(candidate)
        used_record_ids.add(record_id)
        if len(selected) == 25:
            break
    if len(selected) < 25:
        raise ValueError(f"{domain} has only {len(selected)} selected MMLU-Pro candidates")
    return selected


def hle_sort_key(domain: str, row: dict[str, Any]) -> tuple[int, int, int, int, str]:
    subset = row["hle_verified_subset"]
    subject = row["hle_raw_subject"]
    subjects = sorted(HLE_SUBJECTS_BY_DOMAIN[domain])
    subject_rank = subjects.index(subject) if subject in subjects else len(subjects)
    choice_source_rank = 0 if row.get("hle_choice_source") == "source_native_mc" else 1
    return (
        SUBSET_PRIORITY.get(subset, 9),
        subject_rank,
        choice_source_rank,
        stable_int("hle_mixed_gt_v2", domain, row["record_id"]),
        row["record_id"],
    )


def select_hle_for_domain(
    domain: str,
    rows: list[dict[str, Any]],
    used_record_ids: set[str],
) -> list[dict[str, Any]]:
    subjects = HLE_SUBJECTS_BY_DOMAIN[domain]
    candidates = []
    slot_replacements = HLE_SLOT_REPLACEMENT_RECORD_IDS_BY_DOMAIN.get(domain, {})
    all_slot_replacement_sources = {
        source
        for replacements in HLE_SLOT_REPLACEMENT_RECORD_IDS_BY_DOMAIN.values()
        for source in replacements
    }
    for row in rows:
        if row["record_id"] in used_record_ids:
            continue
        if row["record_id"] in all_slot_replacement_sources and row["record_id"] not in slot_replacements:
            continue
        if (
            row.get("hle_choice_source") == "synthetic_numeric_mc_from_exact_answer"
            and domain not in {"Physical Science", "Chemical Science"}
        ):
            continue
        if row["hle_raw_subject"] in subjects:
            candidates.append(row)
        elif domain == "Physical Science" and row["record_id"] in PHYSICAL_EXTRA_HLE_RECORD_IDS:
            candidates.append(row)
        elif domain == "Chemical Science" and row["record_id"] in CHEMICAL_EXTRA_HLE_RECORD_IDS:
            candidates.append(row)
    if domain not in {"Physical Science", "Chemical Science"}:
        candidates = [
            row
            for row in candidates
            if row["hle_verified_subset"] in {"Gold subset", "Revision subset"}
        ]
    candidates.sort(key=lambda row: hle_sort_key(domain, row))
    if len(candidates) < 25:
        raise ValueError(f"{domain} has only {len(candidates)} eligible HLE candidates")
    by_record_id = {row["record_id"]: row for row in candidates}
    selected = []
    used_selected_record_ids: set[str] = set()
    for candidate in candidates:
        record_id = candidate["record_id"]
        replacement_id = slot_replacements.get(record_id)
        if replacement_id:
            candidate = by_record_id.get(replacement_id)
            if candidate is None:
                raise ValueError(
                    f"{domain} replacement {replacement_id} is not an eligible HLE candidate"
                )
            record_id = replacement_id
        if record_id in used_selected_record_ids:
            continue
        selected.append(candidate)
        used_selected_record_ids.add(record_id)
        if len(selected) == 25:
            break
    if len(selected) < 25:
        raise ValueError(f"{domain} has only {len(selected)} selected HLE candidates")
    used_record_ids.update(row["record_id"] for row in selected)
    return selected


def mmlu_output_row(domain: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    choices = {str(label).upper(): str(text).strip() for label, text in row["choices"].items()}
    correct = str(row["correct_answer"]).strip().upper()
    item_id = f"GT-{domain_slug(domain)}-MMLU-{index:03d}"
    record_id = str(row.get("record_id") or row.get("source") or row.get("id"))
    question = str(row["question"]).strip()
    return {
        "id": item_id,
        "source": "mmlu_pro",
        "verifiability": "GT",
        "domain": domain,
        "source_dataset": row.get("source_dataset", MMLU_PRO_DATASET),
        "source_url": row.get("source_url", MMLU_PRO_SOURCE_URL),
        "source_quote": source_quote(question, correct, choices),
        "record_id": record_id,
        "native_id": str(row.get("native_id") or record_id),
        "mmlu_pro_category": public_mmlu_category(row.get("mmlu_pro_category")),
        "mmlu_pro_src": row.get("mmlu_pro_src"),
        "question": question,
        "answer_mode": "multiple_choice",
        "choices": choices,
        "correct_answer": correct,
    }


def hle_output_row(domain: str, index: int, row: dict[str, Any]) -> dict[str, Any]:
    choices = row["choices"]
    correct = row["correct_answer"]
    item_id = f"GT-{domain_slug(domain)}-HLE-{index:03d}"
    return {
        "id": item_id,
        "source": "hle_verified",
        "verifiability": "GT",
        "domain": domain,
        "source_dataset": HLE_DATASET,
        "source_url": HLE_SOURCE_URL,
        "source_quote": source_quote(row["question"], correct, choices),
        "record_id": row["record_id"],
        "native_id": row["native_id"],
        "hle_verified_subset": row["hle_verified_subset"],
        "hle_native_category": row["hle_native_category"],
        "hle_raw_subject": row["hle_raw_subject"],
        "hle_original_answer": row.get("hle_original_answer"),
        "question": row["question"],
        "answer_mode": "multiple_choice",
        "choices": choices,
        "correct_answer": correct,
    }


def build_panel() -> list[dict[str, Any]]:
    mmlu_rows = mmlu_source_rows()
    hle_rows = hle_source_rows()
    output = []
    used_hle_records: set[str] = set()
    for domain in DOMAINS:
        for index, row in enumerate(select_mmlu_for_domain(domain, mmlu_rows), start=1):
            output.append(mmlu_output_row(domain, index, row))
        for index, row in enumerate(select_hle_for_domain(domain, hle_rows, used_hle_records), start=1):
            output.append(hle_output_row(domain, index, row))
    return output


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_domain = Counter(row["domain"] for row in rows)
    by_source = Counter(row["source"] for row in rows)
    selection_summary = []
    for source_key, family in [("mmlu_pro", "MMLU-Pro"), ("hle_verified", "HLE-Verified")]:
        for domain in DOMAINS:
            subset = [
                row
                for row in rows
                if row["domain"] == domain and row["source"] == source_key
            ]
            entry: dict[str, Any] = {
                "family": family,
                "domain": domain,
                "n": len(subset),
                "labels": dict(Counter(row["correct_answer"] for row in subset)),
            }
            if family == "MMLU-Pro":
                entry["categories"] = dict(Counter(row.get("mmlu_pro_category") for row in subset))
            else:
                entry["subsets"] = dict(Counter(row.get("hle_verified_subset") for row in subset))
                entry["raw_subjects"] = dict(Counter(row.get("hle_raw_subject") for row in subset))
                entry["choice_sources"] = dict(Counter(row.get("hle_choice_source") for row in subset))
            selection_summary.append(entry)
    return {
        "panel": "supersycophantic_mixed_gt_200_candidate",
        "rule": (
            "25 MMLU-Pro + 25 HLE-Verified per GT domain. MMLU-Pro rows use "
            "the saturated-screened pool with agent exclusions. HLE rows use "
            "Gold/Revision records. Physical Science may use manually "
            "screened physics-adjacent subjects such as engineering, materials, "
            "robotics, biophysics, and geoscience when the item itself targets "
            "a physical mechanism, measurement, material process, kinematic "
            "relation, or geophysical inference."
        ),
        "counts_by_domain": dict(by_domain),
        "counts_by_source": dict(by_source),
        "selection_summary": selection_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    parser.add_argument("--summary", type=Path, default=None, help="Optional JSON summary output.")
    args = parser.parse_args()

    rows = build_panel()
    if len(rows) != 200:
        raise ValueError(f"Mixed GT panel has {len(rows)} rows, expected 200")
    for row in rows:
        if row["correct_answer"] not in row["choices"]:
            raise ValueError(f"{row['id']} correct answer is not in choices")
    write_jsonl(args.output, rows)
    print(f"Wrote {len(rows)} rows to {args.output}")
    if args.summary:
        args.summary.write_text(
            json.dumps(summarize(rows), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote summary to {args.summary}")


if __name__ == "__main__":
    main()
