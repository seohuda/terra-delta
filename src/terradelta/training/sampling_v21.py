"""Keep positive categories balanced and draw real/synthetic negatives at 3:1."""

import math


def pools(rows):
    groups = {name: [] for name in ("new_building", "tree_removal", "both", "negative", "real_negative")}
    for i, row in enumerate(rows):
        category = row.get("category")
        if category not in groups:
            raise ValueError(f"Unaudited training category: {category}")
        groups[category].append(i)
    if not all(groups.values()):
        raise ValueError("v2.1 needs both positive classes and both negative pools")
    return groups


def draw_batch(groups, batch_size, rng, real_negative_ratio=3.0):
    if isinstance(batch_size, bool) or batch_size < 8 or not isinstance(batch_size, int):
        raise ValueError("batch_size must be an integer >= 8")
    if not math.isfinite(real_negative_ratio) or not 2 <= real_negative_ratio <= 4:
        raise ValueError("real negative ratio must be in [2,4]")
    chosen = []
    for key, fraction in (("new_building", .25), ("tree_removal", .25), ("both", .125)):
        chosen.extend(rng.choice(groups[key]) for _ in range(max(1, int(batch_size * fraction))))
    while len(chosen) < batch_size:
        key = "real_negative" if rng.random() < real_negative_ratio / (1 + real_negative_ratio) else "negative"
        chosen.append(rng.choice(groups[key]))
    rng.shuffle(chosen)
    return chosen
