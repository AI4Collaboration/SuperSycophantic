import json
import re
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parent
OUTPUT = DATA_DIR / "hle_verified_math_text_50.jsonl"
TOTAL_ITEMS = 50


def hle_json(row):
    raw = row.get("json")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return {}
    return {}


def is_gold_valid_math_text(row, meta):
    return (
        "gold" in str(row.get("Verified_Classes", "")).lower()
        and str(row.get("problem_is_valid", "")).strip() == "1"
        and str(row.get("answer_is_valid", "")).strip() == "1"
        and str(row.get("category", "")).strip() == "Math"
        and not meta.get("image")
    )


def parse_hle_choices(question):
    parts = re.split(r"\n\s*Answer Choices:\s*\n", str(question), maxsplit=1, flags=re.IGNORECASE)
    if len(parts) != 2:
        raise ValueError("missing Answer Choices block")
    stem, choices_block = parts
    matches = list(
        re.finditer(
            r"(?ms)^\s*([A-Z])[\.\)]\s*(.*?)(?=^\s*[A-Z][\.\)]\s|\Z)",
            choices_block.strip(),
        )
    )
    choices = {match.group(1).strip().upper(): match.group(2).strip() for match in matches}
    if len(choices) < 2:
        raise ValueError("too few choices")
    return stem.strip(), choices


def clean_answer(answer):
    answer = str(answer).strip()
    if answer.startswith("$") and answer.endswith("$"):
        answer = answer[1:-1].strip()
    return answer


def is_simple_exact_answer(answer):
    answer = clean_answer(answer)
    if not answer or len(answer) > 20:
        return False
    without_macros = answer.replace(r"\frac", "").replace(r"\sqrt", "")
    if re.search(r"[A-Za-z]", without_macros):
        return False
    return bool(re.fullmatch(r"[0-9\s+\-*/^().,{}\\]+", without_macros))


def shift_number(match):
    token = match.group(0)
    if "." in token:
        value = float(token)
        shifted = value + 1 if value >= 0 else value - 1
        return f"{shifted:g}"
    value = int(token)
    shifted = value + 1 if value >= 0 else value - 1
    return str(shifted)


def exact_pressure_target(answer):
    answer = clean_answer(answer)
    candidate, count = re.subn(r"(?<![A-Za-z])-?\d+(?:\.\d+)?", shift_number, answer, count=1)
    if count and candidate != answer:
        return candidate
    return "1" if answer != "1" else "2"


def base_item(row, meta, question, answer_mode):
    return {
        "id": f"hle_verified-train-{row.get('id')}",
        "source_dataset": "skylenage/HLE-Verified",
        "source": "hle_verified_train",
        "record_id": row.get("id", ""),
        "verified_class": row.get("Verified_Classes", ""),
        "category": row.get("category", ""),
        "raw_subject": row.get("raw_subject", ""),
        "question": question,
        "answer_mode": answer_mode,
        "source_file": OUTPUT.name,
    }


def make_mc_item(row, meta):
    question = str(meta.get("question") or row["question"]).strip()
    stem, choices = parse_hle_choices(question)
    correct = str(meta.get("answer") or row["answer"]).strip().upper()
    if correct not in choices:
        raise ValueError(f"answer {correct!r} missing from choices")
    pressure = next(label for label in sorted(choices) if label != correct)
    item = base_item(row, meta, stem, "multiple_choice")
    item.update(
        {
            "choices": choices,
            "correct_answer": correct,
            "correct_answer_text": choices[correct],
            "pressure_target": pressure,
            "pressure_target_text": choices[pressure],
        }
    )
    return item


def make_exact_item(row, meta):
    question = str(meta.get("question") or row["question"]).strip()
    answer = clean_answer(meta.get("answer") or row["answer"])
    pressure = exact_pressure_target(answer)
    item = base_item(row, meta, question, "exact")
    item.update(
        {
            "choices": None,
            "correct_answer": answer,
            "correct_answer_text": answer,
            "pressure_target": pressure,
            "pressure_target_text": pressure,
        }
    )
    return item


def main():
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("The 'datasets' package is required to build HLE Math source data.") from exc

    mc_items = []
    exact_items = []
    dataset = load_dataset("skylenage/HLE-Verified", split="train", streaming=True)
    for row in dataset:
        meta = hle_json(row)
        if not is_gold_valid_math_text(row, meta):
            continue
        answer_type = str(meta.get("answer_type", "")).lower()
        try:
            if answer_type == "multiplechoice":
                mc_items.append(make_mc_item(row, meta))
            elif is_simple_exact_answer(meta.get("answer") or row.get("answer")):
                exact_items.append(make_exact_item(row, meta))
        except (KeyError, ValueError):
            continue
        if len(mc_items) >= TOTAL_ITEMS:
            break

    selected = mc_items[:TOTAL_ITEMS]
    if len(selected) < TOTAL_ITEMS:
        selected.extend(exact_items[: TOTAL_ITEMS - len(selected)])
    if len(selected) != TOTAL_ITEMS:
        raise SystemExit(f"Only found {len(selected)} usable HLE Math rows")

    OUTPUT.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in selected),
        encoding="utf-8",
    )
    print(
        f"wrote {len(selected)} rows to {OUTPUT} "
        f"(multiple_choice={sum(item['answer_mode'] == 'multiple_choice' for item in selected)}, "
        f"exact={sum(item['answer_mode'] == 'exact' for item in selected)})"
    )


if __name__ == "__main__":
    main()
