import random
from collections import Counter

import pytest

from terradelta.training.sampling_v21 import draw_batch, pools


def test_preserves_positive_balance_and_actively_draws_real_negatives():
    names = ["new_building", "tree_removal", "both", "negative", "real_negative"]
    groups = pools([{"category": name} for name in names])
    rng = random.Random(42)
    negatives = Counter()
    for _ in range(1000):
        counts = Counter(names[i] for i in draw_batch(groups, 8, rng))
        assert counts["new_building"] == counts["tree_removal"] == 2
        assert counts["both"] == 1
        negatives.update({k: counts[k] for k in ("negative", "real_negative")})
    assert 2.5 < negatives["real_negative"] / negatives["negative"] < 3.5


def test_unaudited_categories_are_rejected():
    with pytest.raises(ValueError, match="Unaudited"):
        pools([{"category": "real"}])


@pytest.mark.parametrize("ratio", [0, 1, 5, float("nan")])
def test_invalid_ratio_rejected(ratio):
    with pytest.raises(ValueError):
        draw_batch({}, 8, random.Random(1), ratio)
