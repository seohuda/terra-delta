import numpy as np
import pytest

from terradelta.inference.alignment import align_pair


def test_alignment_identity_and_translation():
    import cv2
    rng = np.random.default_rng(1)
    pre = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    post = cv2.warpAffine(pre, np.float32([[1, 0, 2], [0, 1, -1]]), (64, 64), borderMode=cv2.BORDER_REFLECT_101)
    a, b = align_pair(pre, post)
    assert a is pre and b is post
    a, b = align_pair(pre, post, "phase")
    assert np.mean(np.abs(a[5:-5,5:-5].astype(float) - b[5:-5,5:-5])) < 20
    with pytest.raises(ValueError):
        align_pair(pre, post, "phase", max_shift=.1)
    a, _ = align_pair(pre, post, "phase", max_shift=.1, on_failure="identity")
    assert a is pre
