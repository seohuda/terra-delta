from copy import deepcopy
import json
from pathlib import Path

import pytest

from terradelta.inference.v2_evaluation import promotion_gate


def baseline():
    return json.loads((Path(__file__).parents[1] / "docs/final-v2-selection.json").read_text())["metrics"]


def test_same_checkpoint_is_not_improvement():
    old = baseline()
    result = promotion_gate(deepcopy(old), old)
    assert not result["passed"]
    assert result["reasons"] == ["legacy: real no-change false positives did not decrease"]


def test_reduced_false_positives_with_preserved_positives_passes():
    old = baseline()
    new = deepcopy(old)
    new["legacy"]["no_change_fp_any"] -= 1
    assert promotion_gate(new, old)["passed"]


@pytest.mark.parametrize("dataset,class_name", [
    ("legacy", "new_building"), ("legacy", "tree_removal"),
    ("stress", "new_building"), ("stress", "tree_removal"),
])
def test_positive_recall_loss_blocks_negative_only_improvement(dataset, class_name):
    old = baseline()
    new = deepcopy(old)
    new["legacy"]["no_change_fp_any"] = 0
    new[dataset]["classes"][class_name]["presence_confusion"]["tp"] -= 1
    assert "positive recall regressed" in " ".join(promotion_gate(new, old)["reasons"])


def test_shapes_and_stress_score_are_required():
    old = baseline()
    new = deepcopy(old)
    new["legacy"]["no_change_fp_any"] = 0
    new["stress"]["score"] -= .01
    new["legacy"]["classes"]["tree_removal"]["shape_score"] *= .9
    result = promotion_gate(new, old)
    assert not result["passed"]
    assert len(result["reasons"]) == 2


def test_population_change_is_error():
    old = baseline()
    new = deepcopy(old)
    new["legacy"]["no_change_count"] -= 1
    with pytest.raises(ValueError, match="populations differ"):
        promotion_gate(new, old)
