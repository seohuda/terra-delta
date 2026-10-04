"""Independent binary morphology with NumPy only (no rasterio dependency)."""

from numbers import Integral

import numpy as np

from .polygons import _binary_mask


def _radius(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer radius")
    return int(value)


def _morph(mask, radius, dilate):
    if radius == 0 or not mask.size:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, radius, constant_values=False)
    output = np.zeros_like(mask) if dilate else np.ones_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            view = padded[dy:dy + height, dx:dx + width]
            if dilate:
                output |= view
            else:
                output &= view
    return output


def _components(mask, connectivity):
    labels = np.zeros(mask.shape, dtype=np.int32)
    sizes = [0]
    offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    height, width = mask.shape
    for y, x in zip(*np.nonzero(mask)):
        if labels[y, x]:
            continue
        label = len(sizes)
        labels[y, x] = label
        stack = [(y, x)]
        count = 0
        while stack:
            row, col = stack.pop()
            count += 1
            for dy, dx in offsets:
                ny, nx = row + dy, col + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not labels[ny, nx]:
                    labels[ny, nx] = label
                    stack.append((ny, nx))
        sizes.append(count)
    return labels, np.asarray(sizes)


def fill_holes(mask):
    """Fill only background components not connected to the image border (4-way)."""
    m = _binary_mask(mask)
    if not m.size:
        return m
    labels, _ = _components(~m, 4)
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    return m | ((labels > 0) & ~np.isin(labels, border))


def filter_components(mask, min_area=0, connectivity=4):
    """Remove connected foreground components below pixel-count ``min_area``."""
    m = _binary_mask(mask)
    if not np.isfinite(min_area) or min_area < 0:
        raise ValueError("component min_area must be finite and nonnegative")
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if min_area == 0 or not m.any():
        return m.copy()
    labels, sizes = _components(m, connectivity)
    keep = sizes >= min_area
    keep[0] = False
    return keep[labels]


def postprocess_mask(mask, config=None):
    """Opening → closing → dilation → erosion → hole fill → component filter.

    Operation values are integer radii of square ``(2*r+1)`` kernels; zero
    disables an operation. Outside-image pixels are always background.
    ``morphology`` may contain the operation settings within a class config.
    """
    settings = dict(config or {})
    morphology = dict(settings.get("morphology") or {})
    m = _binary_mask(mask).copy()
    for name in ("opening", "closing", "dilation", "erosion"):
        radius = _radius(morphology.get(name, settings.get(name, 0)), name)
        if name == "opening":
            m = _morph(_morph(m, radius, False), radius, True)
        elif name == "closing":
            m = _morph(_morph(m, radius, True), radius, False)
        else:
            m = _morph(m, radius, name == "dilation")
    if morphology.get("fill_holes", settings.get("fill_holes", False)):
        m = fill_holes(m)
    return filter_components(m, settings.get("cc_min_area", 0), settings.get("connectivity", 4))
