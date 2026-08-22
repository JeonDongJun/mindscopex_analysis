from __future__ import annotations

import unittest

from mindscopex_analysis.stats import bootstrap_ci, mean_or_none, paired_summary, sign_flip_p


class SignFlipTests(unittest.TestCase):
    def test_empty_reports_none_not_zero(self) -> None:
        # "no items" and "an effect of zero" must not read the same downstream.
        result = sign_flip_p([])
        self.assertEqual(result["n"], 0)
        self.assertIsNone(result["mean"])
        self.assertIsNone(result["p"])

    def test_all_same_sign_is_the_smallest_p_the_design_can_give(self) -> None:
        # With n items all of the same sign, the only sign assignment reaching the
        # observed |mean| is the observed one (and its mirror), so p = 2 / 2^n.
        values = [1.0] * 6
        result = sign_flip_p(values, draws=40000, seed=0)
        self.assertAlmostEqual(result["p"], 2 / 2**6, delta=0.005)

    def test_symmetric_data_is_not_significant(self) -> None:
        result = sign_flip_p([1.0, -1.0, 2.0, -2.0, 0.5, -0.5], draws=20000, seed=0)
        self.assertAlmostEqual(result["mean"], 0.0, places=9)
        self.assertGreater(result["p"], 0.9)

    def test_p_is_floored_at_the_resolution_of_the_draws(self) -> None:
        # The draws are sampled, not exhaustive. At n=25 the true p can be ~1e-8 while
        # only 20k draws are taken, so b/draws returns a literal 0.0 -- a claim no
        # randomisation test can support. (b+1)/(draws+1) floors it at the real limit.
        for values in ([5.0], [3.0, 4.0, 5.0], [100.0] * 25):
            result = sign_flip_p(values, draws=2000, seed=1)
            self.assertGreaterEqual(result["p"], 1 / 2001)
            self.assertGreater(result["p"], 0.0)

    def test_a_lone_outlier_still_reaches_significance(self) -> None:
        # Worth pinning because it is counter-intuitive: the sign-flip test is NOT
        # robust to one dominant item. 24 tiny concordant values plus one huge one is
        # a surprising SIGN pattern, so p is tiny even though the mean rests on a
        # single observation. The p-value alone cannot flag this -- see
        # PairedSummaryTests, where the bootstrap CI is what exposes it.
        values = [0.001] * 24 + [10.0]
        self.assertLess(sign_flip_p(values, draws=20000, seed=0)["p"], 0.01)

    def test_two_sided_is_sign_agnostic(self) -> None:
        values = [0.4, 0.9, -0.2, 1.1, 0.7]
        positive = sign_flip_p(values, draws=20000, seed=3)
        negative = sign_flip_p([-v for v in values], draws=20000, seed=3)
        self.assertAlmostEqual(positive["p"], negative["p"], places=9)
        self.assertAlmostEqual(positive["mean"], -negative["mean"], places=9)

    def test_deterministic_in_seed(self) -> None:
        values = [0.3, -0.1, 0.8, 0.2, -0.4, 0.9]
        self.assertEqual(
            sign_flip_p(values, draws=5000, seed=7), sign_flip_p(values, draws=5000, seed=7)
        )
        self.assertNotEqual(
            sign_flip_p(values, draws=5000, seed=7)["p"],
            sign_flip_p(values, draws=5000, seed=8)["p"],
        )


class BootstrapTests(unittest.TestCase):
    def test_empty_returns_a_degenerate_interval(self) -> None:
        self.assertEqual(bootstrap_ci([]), (0.0, 0.0))

    def test_interval_brackets_the_mean_and_is_ordered(self) -> None:
        values = [0.4, 0.9, 0.2, 1.1, 0.7, 0.5, 0.8]
        low, high = bootstrap_ci(values, draws=5000, seed=0)
        mean = sum(values) / len(values)
        self.assertLess(low, mean)
        self.assertLess(mean, high)

    def test_constant_data_gives_a_zero_width_interval(self) -> None:
        self.assertEqual(bootstrap_ci([2.0] * 8, draws=1000, seed=0), (2.0, 2.0))

    def test_wider_alpha_gives_a_narrower_interval(self) -> None:
        values = [0.1, 0.9, -0.3, 1.4, 0.2, 0.6]
        narrow = bootstrap_ci(values, draws=8000, seed=0, alpha=0.5)
        wide = bootstrap_ci(values, draws=8000, seed=0, alpha=0.05)
        self.assertGreater(narrow[0], wide[0])
        self.assertLess(narrow[1], wide[1])

    def test_top_index_stays_in_range(self) -> None:
        # alpha small enough that (1 - alpha/2) * draws rounds to draws itself.
        self.assertIsNotNone(bootstrap_ci([1.0, 2.0], draws=100, seed=0, alpha=0.001))

    def test_rejects_impossible_alpha(self) -> None:
        for alpha in (0.0, 1.0, -0.1, 2.0):
            with self.assertRaises(ValueError):
                bootstrap_ci([1.0, 2.0], alpha=alpha)

    def test_deterministic_in_seed(self) -> None:
        values = [0.3, -0.1, 0.8, 0.2]
        self.assertEqual(
            bootstrap_ci(values, draws=2000, seed=5), bootstrap_ci(values, draws=2000, seed=5)
        )


class PairedSummaryTests(unittest.TestCase):
    def test_the_ci_is_what_separates_a_broad_effect_from_an_outlier_driven_one(self) -> None:
        # This is the whole reason the three statistics are bundled. Both of these
        # look identical on p AND on n_positive; only the CI width tells them apart,
        # because resampling sometimes drops the single item the outlier case rests on.
        broad = paired_summary([0.2] * 20 + [-0.1] * 5, draws=8000, seed=0)
        outlier = paired_summary([0.001] * 24 + [10.0], draws=8000, seed=0)

        self.assertLess(broad["p"], 0.05)
        self.assertLess(outlier["p"], 0.05)  # p cannot tell them apart
        self.assertGreater(outlier["n_positive"], broad["n_positive"])  # nor can the sign count

        broad_width = (broad["ci_high"] - broad["ci_low"]) / abs(broad["mean"])
        outlier_width = (outlier["ci_high"] - outlier["ci_low"]) / abs(outlier["mean"])
        self.assertLess(broad_width, 1.0)
        self.assertGreater(outlier_width, 2.0)

    def test_empty_propagates_none_through_every_field(self) -> None:
        result = paired_summary([])
        self.assertEqual(result["n"], 0)
        for key in ("mean", "p", "ci_low", "ci_high"):
            self.assertIsNone(result[key], key)
        self.assertEqual(result["n_positive"], 0)

    def test_agrees_with_its_components(self) -> None:
        values = [0.4, -0.2, 0.9, 0.1, 0.6]
        summary = paired_summary(values, draws=3000, seed=2)
        self.assertEqual(summary["p"], sign_flip_p(values, draws=3000, seed=2)["p"])
        self.assertEqual(
            (summary["ci_low"], summary["ci_high"]), bootstrap_ci(values, draws=3000, seed=2)
        )


class MeanOrNoneTests(unittest.TestCase):
    def test_empty_is_none_not_zero(self) -> None:
        # Three separate review rounds found the same defect: an arm with no items
        # publishing a confident 0.0, which reads as "measured, no difference".
        self.assertIsNone(mean_or_none([]))

    def test_a_real_zero_mean_is_still_zero(self) -> None:
        # The distinction only works if a genuine zero survives.
        self.assertEqual(mean_or_none([1.0, -1.0]), 0.0)
        self.assertEqual(mean_or_none([0.0]), 0.0)

    def test_matches_the_plain_mean_when_there_is_data(self) -> None:
        values = [0.5, 1.5, 2.5]
        self.assertAlmostEqual(mean_or_none(values), sum(values) / len(values), places=9)


if __name__ == "__main__":
    unittest.main()
