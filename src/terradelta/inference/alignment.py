"""Optional registration of pre to post. Default is no alignment."""
import numpy as np


def align_pair(pre, post, method="none", max_shift=4.0, on_failure="raise"):
    if pre.shape != post.shape or pre.ndim != 3 or pre.shape[2] != 3:
        raise ValueError("Alignment requires matching RGB arrays")
    if method == "none":
        return pre, post
    if method not in {"phase", "ecc_translation", "ecc_affine"}:
        raise ValueError(f"Unknown alignment: {method}")
    if on_failure not in {"raise", "identity"}:
        raise ValueError("on_failure must be raise or identity")
    import cv2
    try:
        a = cv2.cvtColor(pre, cv2.COLOR_RGB2GRAY).astype(np.float32)
        b = cv2.cvtColor(post, cv2.COLOR_RGB2GRAY).astype(np.float32)
        if method == "phase":
            window = cv2.createHanningWindow((a.shape[1], a.shape[0]), cv2.CV_32F)
            (dx, dy), response = cv2.phaseCorrelate(a, b, window)
            if response < .05:
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
