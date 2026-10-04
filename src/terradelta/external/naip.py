"""USDA NAIP metadata through Microsoft's public Planetary Computer STAC API.

Source/API documentation:
https://planetarycomputer.microsoft.com/docs/reference/stac/
https://planetarycomputer.microsoft.com/docs/concepts/sas/
Dry-run does not sign or read raster assets unless HEAD sizes are requested.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from itertools import combinations
import re
from threading import Lock
from urllib.parse import urlencode, urlsplit, urlunsplit

from .common import Asset, get_json, head_size, safe_filename
from terradelta.utils.geo import validate_bounds

STAC_API = "https://planetarycomputer.microsoft.com/api/stac/v1"
SAS_API = "https://planetarycomputer.microsoft.com/api/sas/v1"
_TOKEN_CACHE = {}
_TOKEN_LOCK = Lock()


def sign_url(url):
    """Use the provider's account/container token API, as its official SDK does.

    Tokens remain transient; callers must not log/persist the signed URL.
    A transient, expiry-aware account cache avoids provider rate limits. Nothing
    is persisted, and concurrent calls share one request per account.
    """
    parsed = urlsplit(url)
    account = re.fullmatch(r"([a-z0-9]+)\.blob\.core\.windows\.net", parsed.netloc)
    if parsed.scheme != "https" or not account or not parsed.path.startswith("/naip/") or parsed.query or parsed.fragment:
        raise ValueError("NAIP signing requires an unsigned HTTPS Azure /naip/ blob URL")
    with _TOKEN_LOCK:
        cached = _TOKEN_CACHE.get(account[1])
        if cached and cached[0] > datetime.now(timezone.utc) + timedelta(seconds=60):
            _, token = cached
        else:
            response = get_json(f"{SAS_API}/token/{account[1]}/naip")
            expiry = datetime.fromisoformat(response["msft:expiry"].replace("Z", "+00:00"))
            token = response["token"]
            if expiry.tzinfo is None or expiry <= datetime.now(timezone.utc) or not isinstance(token, str) or not token:
                raise ValueError("Provider returned an expired/invalid NAIP SAS token")
            _TOKEN_CACHE[account[1]] = (expiry, token)
    return urlunsplit(parsed._replace(query=token))


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


def discover_naip_years(*, bounds, region="", state=None, max_items=200,
                        endpoint=STAC_API, request_json=None):
    """List observed acquisition years and 2–4-year options from bounded metadata.

    A listed year only intersects the AOI. Candidate temporal pairs require an
    individual footprint covering the entire requested AOI in both years.
    Truncated discovery never establishes that missing years are unavailable.
    No signing, HEAD request, raster read or payload download is performed.
    """
    from shapely.geometry import box, shape
    result = discover_naip(bounds=bounds, years=range(2003, date.today().year + 1),
                           region=region, state=state, max_items=max_items,
                           inspect_sizes=False, endpoint=endpoint, request_json=request_json)
    aoi = box(*result["bounds"])
    groups = {}
    for asset, item in zip(result["assets"], result["items"]):
        properties = item["properties"]
        geometry = item.get("geometry")
        covers = bool(geometry and shape(geometry).covers(aoi))
        groups.setdefault(asset.year, []).append({"id": asset.id,
            "acquired": properties.get("datetime") or properties.get("start_datetime"),
            "resolution_m": asset.resolution_m, "covers_full_aoi": covers,
            "url": asset.url, "filename": asset.filename, "license_status": asset.license_status})
    years = sorted(groups)
    options = []
    for pre_year, post_year in combinations(years, 2):
        if 2 <= post_year - pre_year <= 4:
            pre = [item["id"] for item in groups[pre_year] if item["covers_full_aoi"]]
            post = [item["id"] for item in groups[post_year] if item["covers_full_aoi"]]
            if pre and post:
                options.append({"year_pre": pre_year, "year_post": post_year,
                                "gap_years": post_year - pre_year, "pre_asset_ids": pre, "post_asset_ids": post})
    return {"dry_run": True, "source": "NAIP", "region": region, "bounds": list(result["bounds"]),
            "observed_years": years, "object_count": len(result["assets"]), "truncated": result["truncated"],
            "years": [{"year": year, "object_count": len(groups[year]), "items": groups[year]} for year in years],
            "temporal_options": options, "estimated_download_bytes": None, "payload_downloaded_bytes": 0,
            "note": "Metadata discovery only; observed years are not an availability guarantee when truncated. No event or labels are inferred."}
