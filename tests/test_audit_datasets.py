from __future__ import annotations

import sys
import unittest
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from audit_datasets import audit  # noqa: E402


class DatasetAuditTests(unittest.TestCase):
    def test_committed_catalog_has_no_integrity_errors(self) -> None:
        report = audit()

        self.assertEqual(report["errors"], [])
        self.assertEqual(len(report["datasets"]), 10)
        self.assertEqual(report["total_cases"], 657)

    def test_known_exact_overlaps_remain_explicit(self) -> None:
        report = audit()
        overlap_sets = [
            {(dataset_id, case_id) for dataset_id, case_id in group}
            for group in report["exact_overlaps"]
        ]

        self.assertIn(
            {
                ("crt7_classic", "crt7_001"),
                ("crt_pilot", "bat_ball_original"),
            },
            overlap_sets,
        )
        self.assertIn(
            {
                ("crt7_classic", "crt7_002"),
                ("crt_pilot", "machines_widgets"),
            },
            overlap_sets,
        )


if __name__ == "__main__":
    unittest.main()
