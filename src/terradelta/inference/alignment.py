"""Optional registration of pre to post. Default is no alignment."""
import numpy as np


def estimate_translation(pre, post):
    """Estimate pre-to-post pixel displacement without changing either image.

    Phase response is diagnostic, not a guarantee of correct registration.
    Flat images have no usable registration signal and are rejected.
    """
    import cv2
    if pre.shape != post.shape or pre.ndim != 3 or pre.shape[2] != 3:
        raise ValueError("Registration requires matching RGB arrays")
    a = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY).astype(np.float32)
    b = cv2.cvtColor(post, cv2.COLOR_RGB2GRAY).astype(np.float32)
    if not np.isfinite(a).all() or not np.isfinite(b).all() or min(a.std(), b.std()) < 1e-4:
        raise ValueError("Registration lacks finite nonconstant signal")
    window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
    (dx, dy), response = cv2.phaseCorrelate(a, b, window)
    if not np.isfinite([dx, dy, response]).all():
        raise ValueError("Registration returned nonfinite values")
    return {"dx": float(dx), "dy": float(dy), "magnitude": float(np.hypot(dx, dy)),
            "response": float(response), "reliable": bool(response >= .05)}


def align_pair(pre, post, method="none", max_shift=4.0, on_failure="raise"):
    if pre.shape != post.shape or pre.ndim != 3 or pre.shape[2] != 3:
        raise ValueError("Alignment requires matching RGB arrays")
    if method == "none":
        return pre, post
    if method not in {"phase", "ecc_translation", "ecc_affine"}:
        raise ValueError(f"Unknown alignment: {method}")
    if on_failure not in {"raise", "identity"}:
        raise ValueError("on_failure must be raise or identity")
    if not np.isfinite(max_shift) or max_shift < 0:
        raise ValueError("max_shift must be finite and nonnegative")
    import cv2
    try:
        a = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY).astype(np.float32)
        b = cv2.cvtColor(post, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if method == "phase":
            estimate = estimate_translation(pre, post)
            dx, dy = estimate["dx"], estimate["dy"]
            if not estimate["reliable"]:
                raise ValueError("Insufficient phase correlation response")
            warp = np.array([[1, 0, dx], [0, 1, dy]], np.float32)
            flags = cv2.INTER_LINEAR
        else:
            warp = np.eye(2, 3, dtype=np.float32)
            motion = cv2.MOTION_TRANSLATION if method == "ecc_translation" else cv2.MOTION_AFFINE
            _, warp = cv2.findTransformECC(b, a, warp, motion,
                (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 50, 1e-5))
            flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
        corners = np.array([[0,0,1],[pre.shape[1]-1,0,1],[0,pre.shape[0]-1,1],
                            [pre.shape[1]-1,pre.shape[0]-1,1]], np.float32)
        displacement = corners @ warp.T - corners[:, :2]
        if not np.isfinite(warp).all() or np.max(np.linalg.norm(displacement, axis=1)) > max_shift:
            raise ValueError("Registration displacement exceeds configured bound")
        aligned = cv2.warpAffine(pre, warp, (pre.shape[1], pre.shape[0]), flags=flags,
                                 borderMode=cv2.BORDER_REFLECT_101)
        return aligned, post
    except (cv2.error, ValueError):
        if on_failure == "identity":
            return pre, post
        raise
