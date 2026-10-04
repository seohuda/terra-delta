"""Version-pinned 30m forest loss CANDIDATE MINING, never segmentation GT."""

from __future__ import annotations

import math

import numpy as np

from .common import Asset, head_size
from terradelta.utils.geo import validate_bounds

VERSION = "GFC-2024-v1.12"
SOURCE = "https://storage.googleapis.com/earthenginepartners-hansen/GFC-2024-v1.12/download.html"
BASE_URL = "https://storage.googleapis.com/earthenginepartners-hansen"
LAYERS = {"lossyear", "treecover2000", "datamask", "gain", "first", "last"}


def discover_hansen(*, bounds, layers=("lossyear",), inspect_sizes=False, max_tiles=8):
    bounds = validate_bounds(bounds, geographic=True)
    west, south, east, north = bounds
    if south < -60 or north > 80:
        raise ValueError("Hansen published grid coverage is 60S through 80N")
    if not set(layers) <= LAYERS or not layers:
        raise ValueError("Invalid Hansen layers")
    if len(set(layers)) != len(layers):
        raise ValueError("Duplicate layers")
    if max_tiles < 1 or max_tiles > 100:
        raise ValueError("max_tiles must be in [1,100]")
    tiles = []
    for top in range(int(math.ceil(north / 10)) * 10, int(math.floor(south / 10)) * 10, -10):
        for left in range(int(math.floor(west / 10)) * 10, int(math.ceil(east / 10)) * 10, 10):
            tiles.append((top, left))
    if len(tiles) > max_tiles:
        raise ValueError("Hansen AOI exceeds max_tiles; narrow bounds")
    assets = []
    for top, left in tiles:
        tile = f"{abs(top):02d}{'N' if top >= 0 else 'S'}_{abs(left):03d}{'E' if left >= 0 else 'W'}"
        for layer in layers:
            filename = f"Hansen_{VERSION}_{layer}_{tile}.tif"
            url = f"{BASE_URL}/{VERSION}/{filename}"
            assets.append(Asset(id=f"{layer}_{tile}", url=url, filename=filename, source="Hansen_GFC",
                                size_bytes=head_size(url) if inspect_sizes else None,
                                estimated_bytes=36000 * 36000 * (4 if layer in {"first", "last"} else 1),
                                region=tile, year=2024, resolution_m=30, license_status="cc_by_4.0",
                                purpose="candidate_mining_only"))
    return assets


def loss_candidates(lossyear, *, year_pre, year_post, treecover2000=None, min_cover=30,
                    datamask=None):
    """Boolean 30m candidate grid for (year_pre, year_post]; NO high-res labels.

    Acquisition year alone is coarse: review NAIP dates within the loss year.
    ``lossyear`` encoding is 1=2001 through 24=2024, 0=no loss.
    """
    if not 2000 <= year_pre < year_post <= 2024:
        raise ValueError("Year interval must lie within pinned 2000-2024 product")
    values = np.asarray(lossyear)
    if values.ndim != 2 or not np.issubdtype(values.dtype, np.integer) or np.any(values > 24) or np.any(values < 0):
        raise ValueError("Expected an integer Hansen lossyear grid with values 0..24")
    candidate = (values > year_pre - 2000) & (values <= year_post - 2000) & (values != 0)
    if treecover2000 is not None:
        cover = np.asarray(treecover2000)
        if cover.shape != values.shape or not 0 <= min_cover <= 100:
            raise ValueError("Invalid tree cover grid or threshold")
        candidate &= cover >= min_cover
    if datamask is not None:
        if np.asarray(datamask).shape != values.shape:
            raise ValueError("Datamask shape mismatch")
        candidate &= np.asarray(datamask) == 1
    return candidate
