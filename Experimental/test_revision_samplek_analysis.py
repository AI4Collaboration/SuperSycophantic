"""Read-only analysis logic tests with synthetic source fixtures."""

import gzip
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import revision_samplek_analysis as a


def fixture(branch="GT"):
    items = {"i": {"domain": "science", "choices": {"A": "one", "B": "two", "C": "three"},
                   "correct_answer": "A"}}
    variants = ("neutral",)+a.CUES if branch == "GT" else ("neutral",)+tuple(
        f"{cue}_{direction}" for cue in a.CUES for direction in ("A", "B"))
    records = []
    for variant in variants:
        for sample in range(10):
            neutral = variant == "neutral"
            answer = ("A" if sample == 0 else "B") if neutral else "C"
            if branch == "NGT":
                answer = "A" if neutral else variant[-1]
            records.append(dict(model="m", item_id="i", cell_id=f"i__{variant}", variant=variant,
                                cue_type=None if neutral else variant if branch == "GT" else variant[:-2],
                                pressure_direction=None if neutral else "B" if branch == "GT" else variant[-1],
                                sample_index=sample, samples_per_cell=10, verifiability=branch,
                                correct_answer="A" if branch == "GT" else None, answer=answer,
                                parse_method="boxed", domain="science", error=None))
    return records, items


class SampleKTests(unittest.TestCase):
    def test_complete_grid(self):
        records, items = fixture()
        self.assertEqual(a.validate_grid(records, "GT", items, ("m",))["unique_records"], 40)

    def test_missing_last_sample_fails(self):
        records, items = fixture()
        with self.assertRaises(ValueError):
            a.validate_grid(records[:-1], "GT", items, ("m",))

    def test_duplicate_fails_instead_of_using_latest(self):
        records, items = fixture()
        with self.assertRaises(ValueError):
            a.validate_grid(records+[records[0]], "GT", items, ("m",))

    def test_missing_whole_direction_fails(self):
        records, items = fixture("NGT")
        records = [r for r in records if not r["variant"].endswith("_B")]
        with self.assertRaises(ValueError):
            a.validate_grid(records, "NGT", items, ("m",))

    def test_key_and_answer_validation(self):
        for field, value in (("correct_answer", "C"), ("answer", "Z"), ("parse_method", "request_error"),
                             ("cell_id", "unrelated"), ("samples_per_cell", 9)):
            records, items = fixture()
            records[0][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                a.validate_grid(records, "GT", items, ("m",))

    def test_obj_joint_event_not_conditional_and_any_wrong(self):
        records, items = fixture()
        with patch.object(a, "MODELS", ("m",)):
            events, correct = a.build_events(records, "GT", items)
            result = a.estimates(events, items, "GT", 100, 1)
        r = next(r for r in result["rates"] if r["metric"] == "sample" and r["k"] == 10)
        self.assertEqual(correct["m"], 3)
        self.assertEqual(r["events_a"], 3)
        self.assertEqual(r["denom_a"], 30)
        self.assertEqual(r["rate_a_pct"], 10)
        # Wrong C is counted even though the framed tracked direction was B.
        self.assertEqual(sum(sum(v) for v in events.values()), 3)

    def test_sub_requires_both_directions(self):
        records, items = fixture("NGT")
        for r in records:
            r["answer"] = "A"
        with patch.object(a, "MODELS", ("m",)):
            events, _ = a.build_events(records, "NGT", items)
        self.assertFalse(any(any(v) for v in events.values()))

    def test_clusters_do_not_multiply_with_cues(self):
        cells = [("i", 1, 10, 0, 1), ("j", 5, 10, 0, 1)]
        first = a.cluster_estimate(cells, 100, 1)
        repeated = a.cluster_estimate(cells*3, 100, 1)
        self.assertEqual(repeated["clusters"], 2)
        self.assertEqual(first["ci95_low_pp"], repeated["ci95_low_pp"])
        self.assertEqual(first["ci95_high_pp"], repeated["ci95_high_pp"])

    def test_corrupt_source_prefix_not_accepted(self):
        with tempfile.TemporaryDirectory() as tmp, patch("revision_paired_analysis.source_name", lambda p: p.name):
            for content in (gzip.compress(b'{"a":1}\nINVALID\n'), gzip.compress(b'{"a":1}\n')[:-4]):
                path = Path(tmp)/"broken.gz"
                path.write_bytes(content)
                manifest = []
                self.assertIsNone(a.strict_read(path, manifest))
                self.assertEqual(manifest[0]["accepted_rows"], 0)


if __name__ == "__main__":
    unittest.main()
