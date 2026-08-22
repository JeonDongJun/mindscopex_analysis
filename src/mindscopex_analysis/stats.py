"""The randomization tests this study reports its numbers with.

Every headline statistic here is paired and per-item: ``margin_delta`` for one case,
``cue_effect`` for one case, the difference-in-differences for one case. n is around
25 and the distributions are heavy-tailed, so a t-test would be quoting a normal
approximation that the data does not support. These are the exact / resampling
alternatives instead.

They lived as private copies in three job files and three analysis scripts, which
meant the p-values the study reports were never covered by a single test. Everything
here is pure and deterministic in ``seed``.
"""

from __future__ import annotations

import random
import statistics
from collections.abc import Sequence
from typing import Any

import torch


def sign_flip_p(
    values: Sequence[float],
    *,
    draws: int = 20000,
    seed: int = 0,
) -> dict[str, Any]:
    """Two-sided paired sign-flip randomisation test on the mean of ``values``.

    Under the null the per-item value is symmetric about zero, so flipping signs at
    random gives the *exact* reference distribution for its mean -- no distributional
    assumption at all, which is what makes it the right test at n ~ 25.

    Returns ``{"n", "mean", "p"}``. An empty input returns ``mean``/``p`` as None
    rather than 0.0, because "no items" and "an effect of zero" are different claims
    and a downstream reader must not confuse them.

    The p-value is the Monte Carlo estimator ``(b + 1) / (draws + 1)``, not
    ``b / draws``. The draws are sampled, not exhaustive, so the plain ratio returns
    an exact **0.0** whenever no draw happens to reach the observation -- which is
    routine at n = 25, where the true p can be ~1e-8 while only 20k draws are taken.
    Reporting p = 0 from a randomisation test is an overstatement of what the test
    can possibly establish; the floor of 1/(draws+1) is the real resolution limit.
    """

    if not values:
        return {"n": 0, "mean": None, "p": None}
    tensor = torch.tensor([float(value) for value in values], dtype=torch.float64)
    observed = float(tensor.mean())
    generator = torch.Generator().manual_seed(int(seed))
    signs = torch.randint(
        0, 2, (int(draws), tensor.numel()), generator=generator, dtype=torch.float64
    )
    means = ((signs * 2 - 1) * tensor).mean(dim=1)
    # >= not >: ties belong to the null, and excluding them biases p low.
    at_least_as_extreme = int((means.abs() >= abs(observed)).sum())
    p = (at_least_as_extreme + 1) / (int(draws) + 1)
    return {"n": len(values), "mean": observed, "p": p}


def bootstrap_ci(
    values: Sequence[float],
    *,
    draws: int = 20000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean, deterministic in ``seed``.

    Percentile method rather than a normal interval, for the same reason as above.
    An empty input returns ``(0.0, 0.0)`` -- callers should check ``n`` separately.
    """

    if not values:
        return (0.0, 0.0)
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    population = [float(value) for value in values]
    rng = random.Random(int(seed))
    means = sorted(
        statistics.fmean(rng.choices(population, k=len(population))) for _ in range(int(draws))
    )
    low = means[int((alpha / 2) * draws)]
    high = means[min(int(draws) - 1, int((1 - alpha / 2) * draws))]
    return (low, high)


def paired_summary(
    values: Sequence[float],
    *,
    draws: int = 20000,
    seed: int = 0,
) -> dict[str, Any]:
    """``sign_flip_p`` plus a bootstrap CI plus the sign count, in one row.

    The three answer different questions and are cheap together: the p-value says
    whether the mean is distinguishable from zero, the CI says how big it could be,
    and ``n_positive`` says whether a handful of outliers are carrying it -- a mean
    that is significant on 14/25 positive items is a different finding from one that
    is significant on 24/25.
    """

    test = sign_flip_p(values, draws=draws, seed=seed)
    low, high = bootstrap_ci(values, draws=draws, seed=seed)
    return {
        **test,
        "ci_low": low if values else None,
        "ci_high": high if values else None,
        "n_positive": sum(1 for value in values if float(value) > 0.0),
    }


def mean_or_none(values: Sequence[float]) -> float | None:
    """Mean of ``values``, or None when there are none.

    Every job in this repo carries a private ``_mean`` that returns ``0.0`` for an
    empty sequence, which is right for accumulating a rate and wrong everywhere a
    number reaches an artifact. Three separate review rounds found the same defect:
    an arm with no items publishes a confident ``0.0``, and ``0.0`` in a gap or a
    ratio reads as "measured, and there is no difference" -- the exact opposite of
    "not measured". Use this at any site whose value is written to a CSV or JSON.
    """

    if not values:
        return None
    return statistics.fmean(float(value) for value in values)
