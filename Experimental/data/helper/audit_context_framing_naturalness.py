import argparse
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
DATA_DIR = ROOT / "Experimental" / "data"
GT_PATH = DATA_DIR / "supersycophantic_context_gt_200.json"
NGT_PATH = DATA_DIR / "supersycophantic_context_ngt_100.json"

GT_CUES = ["value_relevant", "impression_relevant", "outcome_relevant"]
NGT_CUES = GT_CUES
NGT_DIRECTIONS = ["A", "B"]

VAGUE_OR_OLD_PHRASES = [
    "evaluation i care about",
    "kind of reasoning",
    "people around me",
    "situations like this",
    "fallout around",
    "getting pulled",
    "feel drawn",
    "part of me is pulled",
    "course notes",
    "practice problems",
    "rule i remember",
    "shortcut i learned",
    "method i used",
    "work through the problem",
    "worked through the problem",
    "when i work through",
    "procedural protection during eviction cases",
    "people whose judgment",
    "my peers",
    "people in my field",
    "people on my team",
    "people i identify with",
    "colleagues",
    "people i respect",
    "people in my community",
    "usually defend",
    "usually handle",
    "usually respect",
    "responsible answer",
    "serious answer",
]

UNNATURAL_PATTERNS = [
    r"\bgetting\s+\w+ing\b",
    r"\bsacrificing\s+\w+ing\b",
    r"\bignoring\s+\w+ing\b",
    r"\bthe damage from ignoring\b",
    r"\bthe practical cost of sacrificing\b",
    r"\bthe correct answer to me seems like\b",
    r"\bdecision about a decision\b",
    r"\bwould hurt my course grade\b",
    r"\bsuper important exams\b",
    r"\bi am\b.*\bbecause i am\b",
    r"\bi keep\b.*\bbecause i keep\b",
    r"\bis the value i least want to compromise\b",
    r"\bhas concrete consequences, not just symbolic ones\b",
    r"\bi believe .+ is too important to discount\b",
    r"\bi believe .+ is the central value\b",
    r"\bfeels like the duty\b",
    r"\bi do not want to push .+ aside\b",
    r"\ba balanced answer may still leave .+ underserved\b",
    r"\bi do not want to overlook\b",
    r"\bi do not want to compromise\b",
    r"\bmy values put .+ first\b",
    r"\bthe costs of missing\b",
    r"\bthe choice about\b",
    r"\baffects who bears the cost\b",
]

GT_VALUE_PATTERN = re.compile(
    r"^I believe Option [A-Z] is correct\. .+\.$"
)
GT_IMPRESSION_TERMS = [
    "classmates",
    "study group",
    "students in my class",
    "in my class",
    "lab section",
    "labmates",
    "breaking ranks",
]
GT_OUTCOME_TERMS = ["exam", "graded quiz", "course grade", "homework answer", "grad school application"]

NGT_VALUE_TERMS = [
    "believe",
    "better call",
    "direction",
    "feels right",
    "fits my judgment",
    "matches my own judgment",
    "personal judgment",
    "personal view",
    "makes the most sense",
    "makes more sense",
    "matches what i think",
    "right",
    "trust more",
    "should stand by",
    "should take seriously",
    "should trust",
]
NGT_IMPRESSION_TERMS = [
    "group",
    "same page",
    "stick out",
    "toeing the party line",
    "out of step",
    "people like me",
    "circle expects",
    "friends to see me",
    "friends see me",
    "fit in with",
    "sound like me",
    "sounds like me",
    "breaking ranks",
    "kind of person",
    "who i am",
    "impression of myself",
    "stand behind",
    "see myself",
    "professional",
    "says about me",
    "recognizable as mine",
    "person i am trying to be",
    "person i try to be",
    "integrity",
    "identity",
    "group i feel part of",
    "line i usually take",
]
NGT_OUTCOME_TERMS = [
    "cost",
    "consequence",
    "consequences",
    "deal with",
    "downside",
    "happens to me",
    "fallout",
    "harm",
    "land on me",
    "live with",
    "mine to handle",
    "my own life",
    "outcome on the line",
    "personal consequences",
    "personal future",
    "personal stakes",
    "regret",
    "right for me",
    "reverse",
    "repair",
    "what follows",
    "bears",
    "hurt",
    "riding",
    "affected",
    "stakes",
]
NGT_OUTCOME_EXTERNAL_STAKE_TERMS = [
    "adult shelter",
    "community",
    "communities",
    "employee",
    "employees",
    "employer",
    "employers",
    "families",
    "household",
    "households",
    "patient",
    "patients",
    "people affected",
    "people involved",
    "resident",
    "residents",
    "rural",
    "shelter",
    "tenant",
    "tenants",
    "urban",
    "worker",
    "workers",
]
NGT_VALUE_STAKE_LEAK_TERMS = [
    "cost",
    "consequence",
    "fallout",
    "financial",
    "harm",
    "harmful",
    "injury",
    "life-safety",
    "safety",
    "stake",
    "stakes",
]


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def has_any(text, terms):
    lowered = text.lower()
    return any(term in lowered for term in terms)


def collect_common_text_issues(item_id, variant_name, text):
    issues = []
    lowered = text.lower()
    for phrase in VAGUE_OR_OLD_PHRASES:
        if phrase in lowered:
            issues.append(f"{item_id} {variant_name}: old/vague phrase {phrase!r}")
    for pattern in UNNATURAL_PATTERNS:
        if re.search(pattern, lowered):
            issues.append(f"{item_id} {variant_name}: unnatural pattern {pattern!r}: {text}")
    if " because " in lowered:
        issues.append(f"{item_id} {variant_name}: evidence-style because-clause: {text}")
    if len([part for part in re.split(r"\.\s+", text.strip()) if part]) < 2:
        issues.append(f"{item_id} {variant_name}: framing should use separate stance and cue sentences")
    if not text.endswith("."):
        issues.append(f"{item_id} {variant_name}: framing should end with a period")
    return issues


def audit_gt(items):
    issues = []
    counts = {cue: 0 for cue in GT_CUES}
    for item in items:
        item_id = item.get("id") or item.get("item_id") or "<missing-id>"
        variants = item.get("context_variants") or {}
        if list(item.get("context_variant_order") or []) != ["neutral", *GT_CUES]:
            issues.append(f"{item_id}: GT context_variant_order is not neutral/value/impression/outcome")
        for cue in GT_CUES:
            variant = variants.get(cue) or {}
            counts[cue] += 1
            text = str(variant.get("user_framing") or "")
            issues.extend(collect_common_text_issues(item_id, cue, text))
            if variant.get("cue_type") != cue:
                issues.append(f"{item_id} {cue}: cue_type mismatch")
            if variant.get("injected_answer_truth_relation") != "incorrect":
                issues.append(f"{item_id} {cue}: injected answer is not marked incorrect")
            if cue == "value_relevant":
                if not GT_VALUE_PATTERN.match(text):
                    issues.append(f"{item_id} {cue}: GT value should be direct belief wording: {text}")
                if has_any(text, GT_IMPRESSION_TERMS + GT_OUTCOME_TERMS):
                    issues.append(f"{item_id} {cue}: GT value leaks impression/outcome language: {text}")
            elif cue == "impression_relevant":
                if not has_any(text, GT_IMPRESSION_TERMS):
                    issues.append(f"{item_id} {cue}: GT impression lacks peer/class cue: {text}")
                if has_any(text, GT_OUTCOME_TERMS):
                    issues.append(f"{item_id} {cue}: GT impression leaks outcome language: {text}")
            elif cue == "outcome_relevant":
                if not has_any(text, GT_OUTCOME_TERMS):
                    issues.append(f"{item_id} {cue}: GT outcome lacks exam/grade cue: {text}")
                if has_any(text, GT_IMPRESSION_TERMS):
                    issues.append(f"{item_id} {cue}: GT outcome leaks peer/class cue: {text}")
    return issues, counts


def audit_ngt(items):
    issues = []
    counts = {cue: {"A": 0, "B": 0} for cue in NGT_CUES}
    expected_order = ["neutral"] + [f"{cue}_{direction}" for cue in NGT_CUES for direction in NGT_DIRECTIONS]
    for item in items:
        item_id = item.get("item_id") or item.get("id") or "<missing-id>"
        variants = item.get("context_variants") or {}
        if list(item.get("context_variant_order") or []) != expected_order:
            issues.append(f"{item_id}: NGT context_variant_order mismatch")
        for cue in NGT_CUES:
            for direction in NGT_DIRECTIONS:
                variant_name = f"{cue}_{direction}"
                variant = variants.get(variant_name) or {}
                counts[cue][direction] += 1
                text = str(variant.get("user_framing") or "")
                lowered = text.lower()
                issues.extend(collect_common_text_issues(item_id, variant_name, text))
                if variant.get("cue_type") != cue:
                    issues.append(f"{item_id} {variant_name}: cue_type mismatch")
                if variant.get("direction") != direction:
                    issues.append(f"{item_id} {variant_name}: direction mismatch")
                if f"option {direction.lower()}" not in lowered:
                    issues.append(f"{item_id} {variant_name}: missing directed Option {direction}")
                if cue == "value_relevant":
                    if not has_any(text, NGT_VALUE_TERMS):
                        issues.append(f"{item_id} {variant_name}: NGT value lacks value/belief cue: {text}")
                    if has_any(text, NGT_VALUE_STAKE_LEAK_TERMS):
                        issues.append(f"{item_id} {variant_name}: NGT value leaks stake wording: {text}")
                elif cue == "impression_relevant":
                    if not has_any(text, NGT_IMPRESSION_TERMS):
                        issues.append(f"{item_id} {variant_name}: NGT impression lacks social cue: {text}")
                elif cue == "outcome_relevant":
                    if not has_any(text, NGT_OUTCOME_TERMS):
                        issues.append(f"{item_id} {variant_name}: NGT outcome lacks consequence cue: {text}")
                    if has_any(text, NGT_OUTCOME_EXTERNAL_STAKE_TERMS):
                        issues.append(f"{item_id} {variant_name}: NGT outcome uses third-party stake: {text}")
    return issues, counts


def main():
    parser = argparse.ArgumentParser(description="Audit GT/NGT context framing naturalness and cue alignment.")
    parser.add_argument("--gt", type=Path, default=GT_PATH)
    parser.add_argument("--ngt", type=Path, default=NGT_PATH)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    gt = read_json(args.gt)
    ngt = read_json(args.ngt)
    gt_issues, gt_counts = audit_gt(gt)
    ngt_issues, ngt_counts = audit_ngt(ngt)
    issues = gt_issues + ngt_issues

    lines = [
        "# Context Framing Naturalness Audit",
        "",
        f"GT items: {len(gt)}",
        f"NGT items: {len(ngt)}",
        f"GT non-neutral framing cells: {sum(gt_counts.values())}",
        f"NGT non-neutral framing cells: {sum(sum(v.values()) for v in ngt_counts.values())}",
        "",
        "GT cue counts:",
    ]
    lines.extend(f"- {cue}: {count}" for cue, count in gt_counts.items())
    lines.append("")
    lines.append("NGT cue/direction counts:")
    for cue, by_direction in ngt_counts.items():
        lines.append(f"- {cue}: A={by_direction['A']}, B={by_direction['B']}")
    lines.append("")
    lines.append(f"Issues: {len(issues)}")
    lines.extend(f"- {issue}" for issue in issues)

    report_text = "\n".join(lines) + "\n"
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(report_text, encoding="utf-8")
    print(report_text, end="")
    if issues:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
