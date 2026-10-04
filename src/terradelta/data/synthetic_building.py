"""Mock building-removal pairs from registered RGB imagery and footprints.

API: ``SyntheticBuildingGenerator(strategy='telea', seed=0).generate(post,
building_mask, donor_mask=None, seed=None)`` (also callable), or
``generate_synthetic_building_pair(post, building_mask, **options)``.
Inputs are numpy HWC RGB uint8 and HW bool/uint8 masks (nonzero = selected).
Outputs: independent ``pre``/``post`` RGB arrays, binary uint8
``new_building``/``tree_removal`` arrays, and JSON-compatible ``metadata``.
The footprint is the synthetic GT; feathering changes an explicitly recorded
outer halo. Every footprint pixel is fully replaced, avoiding roof ghosts.

Strategies are separate functions: ``telea_fill``, ``texture_copy_fill`` and
``patch_fill``; ``edge_feather`` performs the common blend. Texture copy uses
one nearby clean rectangle; patch fill selects small clean donor patches by
boundary similarity. Donor masks restrict copy strategies to suitable terrain.
Telea uses OpenCV's neighborhood reconstruction (donor_mask is unsupported).
These are simulation utilities, not proof of real historical construction.
No imagery acquisition, persistence, or dataset generation is performed here.
OpenCV API: https://docs.opencv.org/4.x/d7/d8b/group__photo__inpaint.html
"""

from dataclasses import dataclass

import cv2
import numpy as np


def _image(image: np.ndarray, name: str = "image") -> np.ndarray:
    if not isinstance(image, np.ndarray) or image.dtype != np.uint8:
        raise ValueError(f"{name} must be a numpy uint8 array")
    if image.ndim != 3 or image.shape[2] != 3 or min(image.shape[:2]) < 1:
        raise ValueError(f"{name} must have nonempty HWC RGB shape")
    return np.ascontiguousarray(image)


def _mask(mask: np.ndarray, shape: tuple[int, int], name: str = "mask") -> np.ndarray:
    if not isinstance(mask, np.ndarray) or mask.dtype not in (np.dtype(bool), np.dtype(np.uint8)):
        raise ValueError(f"{name} must be a numpy HW bool or uint8 array")
    if mask.shape != shape:
        raise ValueError(f"{name} must match the image HW shape; implicit resampling is forbidden")
    return np.ascontiguousarray(mask != 0)


def _seed(seed: int) -> int:
    if isinstance(seed, (bool, np.bool_)) or not isinstance(seed, (int, np.integer)) or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return int(seed)


def _uint8(values: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(values), 0, 255).astype(np.uint8)


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    if radius == 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    return cv2.dilate(mask.astype(np.uint8), kernel).astype(bool)


def _donors(mask: np.ndarray, donor_mask: np.ndarray | None) -> np.ndarray:
    allowed = ~mask
    if donor_mask is not None:
        allowed &= _mask(donor_mask, mask.shape, "donor_mask")
    if not allowed.any():
        raise ValueError("nonempty removal mask requires unmasked donor pixels")
    return allowed


def _patch_origins(allowed: np.ndarray, height: int, width: int) -> np.ndarray:
    # Integral images identify fully clean donor rectangles, including borders.
    integral = cv2.integral(allowed.astype(np.uint8), sdepth=cv2.CV_64F)
    sums = (integral[height:, width:] - integral[:-height, width:]
            - integral[height:, :-width] + integral[:-height, :-width])
    return np.argwhere(sums == height * width)


def telea_fill(image: np.ndarray, mask: np.ndarray, *, radius: float = 3.0) -> np.ndarray:
    """Return an inpainted copy using cv2.INPAINT_TELEA; no black placeholder."""
    image = _image(image)
    mask = _mask(mask, image.shape[:2])
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("radius must be finite and positive")
    if not mask.any():
        return image.copy()
    _donors(mask, None)
    return cv2.inpaint(image, mask.astype(np.uint8) * 255, float(radius), cv2.INPAINT_TELEA)


def texture_copy_fill(
    image: np.ndarray, mask: np.ndarray, *, donor_mask: np.ndarray | None = None, seed: int = 0
) -> np.ndarray:
    """Copy a nearby intact texture rectangle; reject if none fits the target bbox.

    No silent change of strategy: choose patch_fill explicitly for larger holes.
    Donor rectangles never overlap removed pixels or donor exclusions.
    """
    image = _image(image)
    mask = _mask(mask, image.shape[:2])
    rng = np.random.default_rng(_seed(seed))
    if not mask.any():
        return image.copy()
    allowed = _donors(mask, donor_mask)
    yy, xx = np.nonzero(mask)
    y0, x0 = int(yy.min()), int(xx.min())
    height, width = int(yy.max() - y0 + 1), int(xx.max() - x0 + 1)
    origins = _patch_origins(allowed, height, width)
    if len(origins) == 0:
        raise ValueError("no clean donor rectangle fits; use patch_fill or a smaller footprint")
    distances = ((origins - (y0, x0)) ** 2).sum(axis=1)
    nearest = origins[distances == distances.min()]
    dy, dx = nearest[int(rng.integers(len(nearest)))]
    result = image.copy()
    target = result[y0:y0 + height, x0:x0 + width]
    region_mask = mask[y0:y0 + height, x0:x0 + width]
    target[region_mask] = image[dy:dy + height, dx:dx + width][region_mask]
    return result


def patch_fill(
    image: np.ndarray, mask: np.ndarray, *, donor_mask: np.ndarray | None = None,
    patch_size: int = 7, max_candidates: int = 64, seed: int = 0,
) -> np.ndarray:
    """Fill masked tiles with clean patches, scoring observed boundary pixels.

    The effective patch size shrinks if necessary until a clean donor fits.
    Interior tiles prefer spatially nearby donors. Candidate subsampling and
    ties use a call-local RNG; no global random state is touched.
    """
    image = _image(image)
    mask = _mask(mask, image.shape[:2])
    rng = np.random.default_rng(_seed(seed))
    for name, value in (("patch_size", patch_size), ("max_candidates", max_candidates)):
        if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if not mask.any():
        return image.copy()
    allowed = _donors(mask, donor_mask)
    height, width = mask.shape
    size = min(int(patch_size), height, width)
    while size > 1 and len(_patch_origins(allowed, size, size)) == 0:
        size -= 1
    result = image.copy()
    cache: dict[tuple[int, int], np.ndarray] = {}
    for y in range(0, height, size):
        for x in range(0, width, size):
            th, tw = min(size, height - y), min(size, width - x)
            selected = mask[y:y + th, x:x + tw]
            if not selected.any():
                continue
            key = (th, tw)
            if key not in cache:
                cache[key] = _patch_origins(allowed, th, tw)
            origins = cache[key]
            if len(origins) > max_candidates:
                origins = origins[rng.choice(len(origins), max_candidates, replace=False)]
            target = image[y:y + th, x:x + tw].astype(np.float32)
            context = ~selected
            # Spatial penalty breaks interior ambiguity without consulting roof RGB.
            scores = ((origins - (y, x)) ** 2).sum(axis=1).astype(np.float64)
            scores /= max(height * height + width * width, 1)
            if context.any():
                for index, (dy, dx) in enumerate(origins):
                    donor = image[dy:dy + th, dx:dx + tw].astype(np.float32)
                    scores[index] += np.mean((donor[context] - target[context]) ** 2)
            winners = np.flatnonzero(scores == scores.min())
            dy, dx = origins[winners[int(rng.integers(len(winners)))]]
            result[y:y + th, x:x + tw][selected] = image[dy:dy + th, dx:dx + tw][selected]
    return result


def edge_feather(
    original: np.ndarray, filled: np.ndarray, footprint: np.ndarray, *, radius: int = 3
) -> np.ndarray:
    """Fully replace footprint and blend only its outer halo, preserving GT.

    ``filled`` must have been reconstructed over a dilated footprint if radius
    is positive. Pixels outside that halo are bit-identical to ``original``.
    """
    original, filled = _image(original), _image(filled, "filled")
    if filled.shape != original.shape:
        raise ValueError("filled and original image shapes must match")
    footprint = _mask(footprint, original.shape[:2], "footprint")
    if isinstance(radius, bool) or not isinstance(radius, (int, np.integer)) or radius < 0:
        raise ValueError("radius must be a nonnegative integer")
    if not footprint.any():
        return original.copy()
    if radius == 0:
        alpha = footprint.astype(np.float32)
    else:
        distance = cv2.distanceTransform((~footprint).astype(np.uint8), cv2.DIST_L2, 5)
        alpha = np.clip(1.0 - distance / (radius + 1.0), 0, 1)
        alpha[~_dilate(footprint, int(radius))] = 0
        alpha[footprint] = 1
    return _uint8(original * (1 - alpha[..., None]) + filled * alpha[..., None])


@dataclass(frozen=True)
class SyntheticBuildingGenerator:
    """Stateless seeded mock generator; the same inputs/seed reproduce a pair.

    Strategies: telea, texture_copy, patch_fill. ``seed`` overrides are per call.
    ``feather_radius=0`` disables feathering; ``inpaint_radius`` is Telea's
    neighborhood radius. Caller supplies registered/rasterized footprints.
    """

    strategy: str = "telea"
    seed: int = 0
    feather_radius: int = 3
    inpaint_radius: float = 3.0
    patch_size: int = 7
    max_candidates: int = 64

    def __post_init__(self) -> None:
        if self.strategy not in ("telea", "texture_copy", "patch_fill"):
            raise ValueError("strategy must be telea, texture_copy, or patch_fill")
        _seed(self.seed)
        if (isinstance(self.feather_radius, bool)
                or not isinstance(self.feather_radius, (int, np.integer)) or self.feather_radius < 0):
            raise ValueError("feather_radius must be a nonnegative integer")
        if not np.isfinite(self.inpaint_radius) or self.inpaint_radius <= 0:
            raise ValueError("inpaint_radius must be finite and positive")
        for name in ("patch_size", "max_candidates"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError(f"{name} must be a positive integer")

    def generate(
        self, post: np.ndarray, building_mask: np.ndarray, *,
        donor_mask: np.ndarray | None = None, seed: int | None = None,
    ) -> dict:
        post = _image(post, "post")
        footprint = _mask(building_mask, post.shape[:2], "building_mask")
        call_seed = _seed(self.seed if seed is None else seed)
        if donor_mask is not None:
            donor_mask = _mask(donor_mask, footprint.shape, "donor_mask")
            if self.strategy == "telea":
                raise ValueError("donor_mask is supported by copy strategies only")
        edit_mask = _dilate(footprint, int(self.feather_radius))
        if self.strategy == "telea":
            filled = telea_fill(post, edit_mask, radius=self.inpaint_radius)
        elif self.strategy == "texture_copy":
            filled = texture_copy_fill(post, edit_mask, donor_mask=donor_mask, seed=call_seed)
        else:
            filled = patch_fill(post, edit_mask, donor_mask=donor_mask, patch_size=self.patch_size,
                                max_candidates=self.max_candidates, seed=call_seed)
        pre = edge_feather(post, filled, footprint, radius=self.feather_radius)
        return {
            "pre": pre, "post": post.copy(), "new_building": footprint.astype(np.uint8),
            "tree_removal": np.zeros(footprint.shape, dtype=np.uint8),
            "metadata": {
                "source": "synthetic_building", "strategy": self.strategy, "seed": call_seed,
                "feather_radius": int(self.feather_radius), "footprint_pixels": int(footprint.sum()),
                "edit_pixels": int(edit_mask.sum()), "label_source": "supplied_building_footprint",
                "synthetic": True,
            },
        }

    __call__ = generate


def generate_synthetic_building_pair(post: np.ndarray, building_mask: np.ndarray, **options) -> dict:
    """Functional wrapper; donor_mask is a generate option, others constructor options."""
    donor_mask = options.pop("donor_mask", None)
    return SyntheticBuildingGenerator(**options).generate(post, building_mask, donor_mask=donor_mask)
