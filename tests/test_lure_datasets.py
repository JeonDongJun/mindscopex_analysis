from __future__ import annotations

import unittest

from mindscopex_analysis import (
    HagendorffCRTItem,
    NatureCRTItem,
    available_lure_datasets,
    download_hagendorff_crt150_source,
    download_nature_crt150_source,
    load_all_lure_cases,
    load_lure_dataset,
    lure_dataset_cases,
    lure_dataset_catalog,
    lure_dataset_info,
)

_EXPECTED = {
    "crt2": 4,
    "crt_fresh_v1": 30,
    "crt_fresh_v2": 150,
    "crt7_classic": 7,
    "crt_pilot": 9,
    "goal_affordance_traps_v1": 240,
    "hagendorff_crt": 150,
    "hagendorff_semantic_illusion": 50,
    "verbal_crt": 10,
    "yax_crt_isomorph": 7,
}


class LureDatasetLoaderTests(unittest.TestCase):
    def test_available_datasets_match_expected_counts(self) -> None:
        self.assertEqual(set(available_lure_datasets()), set(_EXPECTED))
        for name, count in _EXPECTED.items():
            self.assertEqual(len(load_lure_dataset(name)), count, name)

    def test_logprob_margin_cases_have_scoreable_answer_format(self) -> None:
        for info in lure_dataset_catalog():
            if info.scoring != "logprob_margin":
                continue
            for case in load_lure_dataset(info.dataset_id):
                self.assertTrue(case.prompt.endswith("\nAnswer:"), case.case_id)
                self.assertTrue(case.correct_answer.startswith(" "), case.case_id)
                self.assertTrue(case.lure_answer.startswith(" "), case.case_id)
                self.assertNotEqual(
                    case.correct_answer.strip().casefold(),
                    case.lure_answer.strip().casefold(),
                    case.case_id,
                )

    def test_hagendorff_crt_embeds_matched_control(self) -> None:
        cases = load_lure_dataset("hagendorff_crt")
        self.assertTrue(all(case.control_prompt for case in cases))
        first = cases[0]
        self.assertEqual(first.case_id, "hagendorff_crt_difference_001")
        self.assertTrue(first.control_prompt.endswith("\nAnswer:"))
        self.assertNotEqual(first.prompt, first.control_prompt)

    def test_crt_fresh_v1_is_balanced_and_has_validated_controls(self) -> None:
        info = lure_dataset_info("crt_fresh_v1")
        self.assertEqual(
            info.family_counts,
            {"crt_difference": 10, "crt_growth": 10, "crt_rate": 10},
        )
        cases = load_lure_dataset("crt_fresh_v1")
        self.assertTrue(all(case.control_prompt for case in cases))
        self.assertTrue(all("validation=closed_form" in case.note for case in cases))
        self.assertTrue(all("control_answer_equals_lure=true" in case.note for case in cases))

    def test_crt_fresh_v2_is_balanced_and_preserves_pair_metadata(self) -> None:
        info = lure_dataset_info("crt_fresh_v2")
        self.assertEqual(
            info.family_counts,
            {"crt_difference": 50, "crt_growth": 50, "crt_rate": 50},
        )
        cases = load_lure_dataset("crt_fresh_v2")
        self.assertEqual(len({case.pair_id for case in cases}), 150)
        self.assertTrue(all(case.template_id for case in cases))
        self.assertTrue(all(case.condition == "hostile" for case in cases))
        self.assertTrue(all(case.control_prompt for case in cases))
        self.assertTrue(all("validation=closed_form" in case.note for case in cases))

    def test_semantic_illusions_are_premise_rejection(self) -> None:
        info = lure_dataset_info("hagendorff_semantic_illusion")
        self.assertEqual(info.scoring, "premise_rejection")
        cases = load_lure_dataset("hagendorff_semantic_illusion")
        self.assertEqual(cases[0].correct_answer, "")
        self.assertEqual(cases[0].lure_answer, "")
        self.assertIn("reference_answer:", cases[0].note)

    def test_goal_affordance_dataset_is_balanced_and_paired(self) -> None:
        info = lure_dataset_info("goal_affordance_traps_v1")
        self.assertEqual(info.scoring, "binary_choice")
        self.assertEqual(set(info.family_counts.values()), {40})
        cases = load_lure_dataset("goal_affordance_traps_v1")
        self.assertEqual(len({case.pair_id for case in cases}), 60)
        self.assertEqual(
            {case.condition for case in cases},
            {"counterfactual", "explicit", "hostile", "neutral"},
        )
        self.assertTrue(all(case.correct_answer.startswith(" ") for case in cases))
        self.assertTrue(all(case.lure_answer.startswith(" ") for case in cases))

    def test_all_case_ids_globally_unique(self) -> None:
        ids = [case.case_id for cases in load_all_lure_cases().values() for case in cases]
        self.assertEqual(len(ids), len(set(ids)))

    def test_lure_dataset_cases_limits_and_filters(self) -> None:
        smoke = lure_dataset_cases("hagendorff_crt", limit_per_family=3)
        self.assertEqual(len(smoke), 9)
        self.assertEqual(
            {case.family for case in smoke}, {"crt_difference", "crt_growth", "crt_rate"}
        )

        filtered = lure_dataset_cases("hagendorff_crt", families=("crt_rate",), limit_per_family=5)
        self.assertEqual(len(filtered), 5)
        self.assertTrue(all(case.family == "crt_rate" for case in filtered))

        with self.assertRaises(ValueError):
            lure_dataset_cases("hagendorff_crt", families=("nonexistent",))
        with self.assertRaises(ValueError):
            lure_dataset_cases("hagendorff_crt", limit_per_family=0)

    def test_unknown_dataset_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_lure_dataset("does_not_exist")

    def test_hagendorff_aliases_match_nature(self) -> None:
        self.assertIs(HagendorffCRTItem, NatureCRTItem)
        self.assertIs(download_hagendorff_crt150_source, download_nature_crt150_source)


if __name__ == "__main__":
    unittest.main()
