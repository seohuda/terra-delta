"""USDA NAIP metadata through Microsoft's public Planetary Computer STAC API.

Source/API documentation:
https://planetarycomputer.microsoft.com/docs/reference/stac/
https://planetarycomputer.microsoft.com/docs/concepts/sas/
Dry-run does not sign or read raster assets unless HEAD sizes are requested.
"""

from __future__ import annotations

from urllib.parse import urlencode

from .common import Asset, get_json, head_size, safe_filename
from terradelta.utils.geo import validate_bounds

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
SIGN_API = "https://planetarycomputer.microsoft.com/api/sas/v1/sign"


def sign_url(url):
    """Transient SAS authorization; caller must not log/persist the signed URL."""
    return get_json(SIGN_API, params={"href": url})["href"]


def discover_naip(*, bounds, years, region="", state=None, max_items=20,
                  inspect_sizes=False, endpoint=STAC_API, request_json=None):
    """Bounded metadata discovery. Years use actual acquisition datetime.

    Pagination is bounded by max_items, and ``truncated`` is explicit. ``state``
    is the NAIP two-letter state code. An AOI is always mandatory; no national
    listing/download. ``size_bytes=None`` is honest when STAC lacks wire size.
    """
    request_json = request_json or get_json
    bounds = validate_bounds(bounds, geographic=True)
    years = sorted(set(int(year) for year in years))
    if not years or min(years) < 2003 or max_items < 1 or max_items > 1000:
        raise ValueError("Specify NAIP years >=2003 and max_items in [1,1000]")
    params = {"collections": "naip", "bbox": ",".join(map(str, bounds)),
              "datetime": f"{min(years)}-01-01T00:00:00Z/{max(years)}-12-31T23:59:59Z",
              "limit": min(max_items, 100)}
    url = endpoint.rstrip("/") + "/search?" + urlencode(params)
    assets, items, seen = [], [], set()
    truncated = False
    # Also bound pages when a sparse noncontiguous year/state filter matches few.
    for _ in range(20):
        response = request_json(url)
        page_items = response.get("features", [])
        for page_index, item in enumerate(page_items):
            if item.get("id") in seen:
                continue
            seen.add(item["id"])
            props = item["properties"]
            year = int((props.get("datetime") or props.get("start_datetime") or "0000")[:4])
            if year not in years or (state and str(props.get("naip:state", "")).lower() != state.lower()):
                continue
            image = item.get("assets", {}).get("image")
            if not image:
                continue
            shape = image.get("proj:shape", props.get("proj:shape"))
            estimated = int(shape[0]) * int(shape[1]) * 4 if shape else None
            size = image.get("file:size")
            if size is not None:
                size = int(size)
            if inspect_sizes:
                size = head_size(sign_url(image["href"]))
            gsd = props.get("gsd", props.get("naip:resolution"))
            assets.append(Asset(id=item["id"], url=image["href"], filename=safe_filename(item["id"]) + ".tif",
                                source="NAIP", size_bytes=size, estimated_bytes=estimated,
                                region=region or str(props.get("naip:state", "")), year=year,
                                resolution_m=float(gsd) if gsd else None, license_status="public_domain"))
            items.append(item)
            if len(assets) >= max_items:
                truncated = page_index + 1 < len(page_items) or any(link.get("rel") == "next" for link in response.get("links", []))
                break
        if len(assets) >= max_items:
            break
        next_link = next((link for link in response.get("links", []) if link.get("rel") == "next"), None)
        if not next_link:
            break
        if next_link.get("method", "GET").upper() != "GET":
            raise ValueError("STAC returned POST pagination for a GET search; use an explicit compatible endpoint")
        url = next_link["href"]
    else:
        truncated = True
    return {"assets": assets, "items": items, "bounds": bounds, "years": years, "truncated": truncated}


def estimate_aoi_bytes(bounds, *, target_resolution=1.0, bands="rgb", temporal_images=2):
    """Projected AOI raw tensor/storage estimate; no claim about compressed COGs."""
    from terradelta.utils.geo import metric_crs, project_bounds
    from .common import estimate_raster_bytes
    if bands not in {"rgb", "rgbnir", "nir"} or temporal_images < 1:
        raise ValueError("bands must be rgb/rgbnir/nir and temporal_images positive")
    projected = project_bounds(bounds, "EPSG:4326", metric_crs(bounds))
    count = {"rgb": 3, "rgbnir": 4, "nir": 1}[bands]
    return estimate_raster_bytes(projected, target_resolution, count) * temporal_images
