"""Seeded appearance-only no-change and hard-negative mock pairs.

API: ``NegativePairGenerator(seed=0).generate(image, kind='exact_no_change',
roi_mask=None, seed=None, post=None, region_pre=None, region_post=None,
season_pre=None, season_post=None, confirmed_no_change=False)`` (also callable),
or ``generate_negative_pair(image, kind, **options)``. HWC RGB uint8 images;
optional ROI mask is HW bool/uint8 (nonzero selected). All outputs contain
independent pre/post arrays, zero uint8 new_building/tree_removal labels,
valid_mask and metadata. RNG is reset per call from the supplied seed.

The nine requested types in ``NEGATIVE_TYPES`` are exact_no_change, brightness,
seasonal_color, shadows, vegetation_appearance, slight_shift, compression,
blur and temporary_objects. Extra types: roof_color, agricultural, road
(require explicit roi_mask) and same_region_season. Aliases: same_image,
shadow, small_shift, seasonal. Temporary objects simulate small removable
tarps/vehicles, not permanent construction. Roof/road/crop modifications
change color only, retaining structure. Shift uses reflected padding, with
invalid border pixels excluded by valid_mask.

same_region_season accepts a real ``post`` only with explicit matching region
and season identifiers AND confirmed_no_change=True. Geography/season alone
cannot establish negative GT. This confirmation must follow human review or
independent trusted annotations. Other types perturb a single known image;
never pass an unreviewed temporal pair to get automatic background labels.
"""

from dataclasses import dataclass

import cv2
import numpy as np

from .synthetic_building import _image, _mask, _seed, _uint8


NEGATIVE_TYPES = (
    "exact_no_change", "brightness", "seasonal_color", "shadows", "vegetation_appearance",
    "slight_shift", "compression", "blur", "temporary_objects",
)
EXTRA_NEGATIVE_TYPES = ("same_region_season", "roof_color", "agricultural", "road")
_ALIASES = {"same_image": "exact_no_change", "shadow": "shadows", "small_shift": "slight_shift",
            "seasonal": "seasonal_color"}


@dataclass(frozen=True)
class NegativePairGenerator:
    """No persistent random state; override seed to obtain a different mock pair.

    max_shift is measured in pixels. JPEG quality is an integer in [1,100].
    Real paired negatives require the explicit review gate described above.
    """

    seed: int = 0
    max_shift: int = 2
    jpeg_quality: int = 65

    def __post_init__(self) -> None:
        _seed(self.seed)
        if (isinstance(self.max_shift, bool) or not isinstance(self.max_shift, (int, np.integer))
                or self.max_shift < 1):
            raise ValueError("max_shift must be a positive integer")
        if (isinstance(self.jpeg_quality, bool) or not isinstance(self.jpeg_quality, (int, np.integer))
                or not 1 <= self.jpeg_quality <= 100):
            raise ValueError("jpeg_quality must be an integer in [1, 100]")

    def generate(
        self, image: np.ndarray, kind: str = "exact_no_change", *,
        roi_mask: np.ndarray | None = None, seed: int | None = None,
        post: np.ndarray | None = None,
        region_pre: str | None = None, region_post: str | None = None,
        season_pre: str | None = None, season_post: str | None = None,
        confirmed_no_change: bool = False,
    ) -> dict:
        image = _image(image)
        requested_kind = kind
        kind = _ALIASES.get(kind, kind)
        if kind not in NEGATIVE_TYPES + EXTRA_NEGATIVE_TYPES:
            raise ValueError(f"unknown negative kind {kind!r}")
        call_seed = _seed(self.seed if seed is None else seed)
        rng = np.random.default_rng(call_seed)
        height, width = image.shape[:2]
        roi = np.ones((height, width), dtype=bool)
        if roi_mask is not None:
            roi = _mask(roi_mask, (height, width), "roi_mask")
        valid = np.ones((height, width), dtype=np.uint8)
        result = image.copy()
        parameters: dict = {}
        if kind != "same_region_season" and post is not None:
            raise ValueError("post is allowed only for an explicitly reviewed same_region_season pair")
        if kind == "same_region_season":
            if roi_mask is not None:
                raise ValueError("roi_mask does not establish no-change for a real temporal pair")
            if post is None:
                raise ValueError("same_region_season requires a reviewed post image")
            if (not isinstance(region_pre, str) or not region_pre.strip() or region_pre != region_post
                    or not isinstance(season_pre, str) or not season_pre.strip() or season_pre != season_post):
                raise ValueError("real paired negatives require matching nonempty region and season identifiers")
            if confirmed_no_change is not True:
                raise ValueError("real paired negatives require explicit confirmed_no_change=True after review")
            result = _image(post, "post").copy()
            if result.shape != image.shape:
                raise ValueError("paired negative images must be registered and have identical shapes")
            parameters = {"region_id": region_pre, "season": season_pre, "confirmed_no_change": True}
        elif kind == "exact_no_change":
            pass
        elif kind == "brightness":
            gain = float(rng.uniform(0.7, 1.3))
            changed = _uint8(image.astype(np.float32) * gain)
            result[roi] = changed[roi]
            parameters = {"gain": gain}
        elif kind in ("seasonal_color", "vegetation_appearance", "roof_color", "agricultural", "road"):
            if kind in ("roof_color", "agricultural", "road") and roi_mask is None:
                raise ValueError(f"{kind} requires an explicit roi_mask for the existing feature")
            if kind == "vegetation_appearance" and roi_mask is None:
                rgb = image.astype(np.float32)
                roi = (rgb[..., 1] > rgb[..., 0] * 1.1) & (rgb[..., 1] > rgb[..., 2] * 1.1)
            gains = {
                "seasonal_color": (1.15, 0.9, 0.8), "vegetation_appearance": (1.1, 0.8, 0.9),
                "roof_color": (0.8, 1.0, 1.3), "agricultural": (1.25, 0.8, 0.9),
                "road": (0.85, 1.0, 1.15),
            }[kind]
            strength = float(rng.uniform(0.6, 1.0))
            channel_gains = 1 + (np.array(gains, dtype=np.float32) - 1) * strength
            changed = _uint8(image.astype(np.float32) * channel_gains)
            result[roi] = changed[roi]
            parameters = {"channel_gains": [float(value) for value in channel_gains],
                          "roi_pixels": int(roi.sum())}
        elif kind == "shadows":
            yy, xx = np.mgrid[:height, :width]
            cy, cx = float(rng.uniform(0, height)), float(rng.uniform(0, width))
            ry, rx = max(height * 0.3, 1), max(width * 0.35, 1)
            distance = ((yy - cy) / ry) ** 2 + ((xx - cx) / rx) ** 2
            darkness = float(rng.uniform(0.25, 0.5))
            alpha = np.exp(-distance / 2) * darkness * roi
            result = _uint8(image * (1 - alpha[..., None]))
            parameters = {"center_yx": [cy, cx], "darkness": darkness}
        elif kind == "slight_shift":
            if roi_mask is not None:
                raise ValueError("slight_shift operates on the whole image; roi_mask is unsupported")
            mx, my = min(int(self.max_shift), width // 2), min(int(self.max_shift), height // 2)
            shifts = [(dx, dy) for dy in range(-my, my + 1) for dx in range(-mx, mx + 1)
                      if dx != 0 or dy != 0]
            dx, dy = shifts[int(rng.integers(len(shifts)))] if shifts else (0, 0)
            matrix = np.array([[1, 0, dx], [0, 1, dy]], dtype=np.float32)
            result = cv2.warpAffine(image, matrix, (width, height), flags=cv2.INTER_NEAREST,
                                    borderMode=cv2.BORDER_REFLECT_101)
            if dx > 0:
                valid[:, :dx] = 0
            elif dx < 0:
                valid[:, dx:] = 0
            if dy > 0:
                valid[:dy] = 0
            elif dy < 0:
                valid[dy:] = 0
            parameters = {"dx": dx, "dy": dy, "padding": "reflect101", "training_scope": "valid_mask_only"}
        elif kind == "compression":
            encoded_ok, encoded = cv2.imencode(
                ".jpg", cv2.cvtColor(image, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, int(self.jpeg_quality)],
            )
            if not encoded_ok:
                raise RuntimeError("OpenCV JPEG encoding failed")
            decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
            if decoded is None:
                raise RuntimeError("OpenCV JPEG decoding failed")
            changed = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            result[roi] = changed[roi]
            parameters = {"jpeg_quality": int(self.jpeg_quality)}
        elif kind == "blur":
            sigma = float(rng.uniform(0.6, 1.3))
            changed = cv2.GaussianBlur(image, (3, 3), sigmaX=sigma, borderType=cv2.BORDER_REFLECT_101)
            result[roi] = changed[roi]
            parameters = {"kernel_size": 3, "sigma": sigma}
        elif kind == "temporary_objects":
            if roi_mask is None:
                roi = np.zeros((height, width), dtype=bool)
                th, tw = max(1, height // 16), max(1, width // 12)
                y, x = int(rng.integers(height - th + 1)), int(rng.integers(width - tw + 1))
                roi[y:y + th, x:x + tw] = True
            color = rng.integers(90, 231, size=3, dtype=np.uint8)
            result[roi] = color
            parameters = {"color_rgb": [int(value) for value in color], "temporary_pixels": int(roi.sum())}
        return {
            "pre": image.copy(), "post": result,
            "new_building": np.zeros((height, width), dtype=np.uint8),
            "tree_removal": np.zeros((height, width), dtype=np.uint8), "valid_mask": valid,
            "metadata": {
                "source": "negative_pair", "kind": kind, "requested_kind": requested_kind,
                "seed": call_seed, "synthetic": kind != "same_region_season",
                "label_source": "reviewed_no_change" if kind == "same_region_season" else "appearance_only_mock",
                "train_eligible": True, "parameters": parameters,
                "changed_pixels": int(np.any(image != result, axis=2).sum()),
            },
        }

    __call__ = generate


def generate_negative_pair(image: np.ndarray, kind: str = "exact_no_change", **options) -> dict:
    """Functional wrapper; seed/max_shift/jpeg_quality configure the generator.

    All remaining options are passed to generate (ROI or reviewed-pair metadata).
    """
    config = {key: options.pop(key) for key in ("seed", "max_shift", "jpeg_quality") if key in options}
    return NegativePairGenerator(**config).generate(image, kind, **options)
