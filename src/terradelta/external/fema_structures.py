"""Bounded AOI USA Structures query and CRS-aware footprint rasterization.

An existing footprint is NOT evidence of a temporal new-building event. Use it
for synthetic deletion labels or manually reviewed high-resolution differences.
"""

from __future__ import annotations

import json
from pathlib import Path

from .common import get_json
from terradelta.utils.geo import rasterize_geometries, validate_bounds

FEATURE_LAYER = (
    "https://services2.arcgis.com/FiaPA4ga0iQKduv3/arcgis/rest/services/"
    "USA_Structures_View/FeatureServer/0"
)


def _aoi_params(bounds):
    bounds = validate_bounds(bounds, geographic=True)
    return {"where": "1=1", "geometry": ",".join(map(str, bounds)),
            "geometryType": "esriGeometryEnvelope", "inSR": 4326,
            "spatialRel": "esriSpatialRelIntersects"}


def discover_fema(*, bounds, endpoint=FEATURE_LAYER, request_json=None):
    """Only layer metadata and count; no polygon download during dry-run."""
    request_json = request_json or get_json
    metadata = request_json(endpoint, params={"f": "json"})
    count = request_json(endpoint + "/query", params={**_aoi_params(bounds), "f": "json", "returnCountOnly": "true"})["count"]
    return {"source": "FEMA_USA_Structures", "endpoint": endpoint, "bounds": list(bounds), "feature_count": count,
            "estimated_bytes": count * 1500, "exact_size_bytes": None,
            "size_note": "1500 bytes per feature is a planning heuristic; geometry size varies",
            "crs": "EPSG:4326", "license_status": "unknown",
            "license_note": "Public access does not establish rights; review item-specific license terms before training",
            "max_record_count": metadata.get("maxRecordCount", 2000),
            "object_id_field": metadata.get("objectIdField", "OBJECTID"),
            "supports_pagination": metadata.get("advancedQueryCapabilities", {}).get("supportsPagination", False)}


def fetch_fema(*, bounds, output=None, allow_download=False, max_features=1000,
               max_bytes=10_000_000, page_size=200, endpoint=FEATURE_LAYER, request_json=None):
    """Explicit download with count preflight, ordered pagination and hard limits.

    GeoJSON always requests outSR=4326; downstream rasterization reprojects it.
    Errors/truncation raise; a partial national dataset is never silently accepted.
    """
    if not allow_download:
        raise PermissionError("FEMA feature fetch requires explicit --download")
    if not 1 <= max_features <= 1_000_000 or max_bytes < 1 or page_size < 1:
        raise ValueError("Invalid feature/byte limits")
    request_json = request_json or get_json
    plan = discover_fema(bounds=bounds, endpoint=endpoint, request_json=request_json)
    if plan["feature_count"] > max_features:
        raise ValueError("FEMA AOI exceeds max_features; narrow bounds")
    if plan["feature_count"] > plan["max_record_count"] and not plan["supports_pagination"]:
        raise ValueError("Layer cannot paginate this AOI")
    features, seen, offset, consumed = [], set(), 0, 0
    while offset < plan["feature_count"]:
        params = {**_aoi_params(bounds), "f": "geojson", "outSR": 4326, "outFields": plan["object_id_field"],
                  "returnGeometry": "true", "orderByFields": plan["object_id_field"] + " ASC",
                  "resultOffset": offset, "resultRecordCount": min(page_size, plan["max_record_count"], max_features - offset)}
        response = request_json(endpoint + "/query", params=params, max_bytes=max_bytes - consumed)
        consumed += len(json.dumps(response).encode())
        if consumed > max_bytes:
            raise ValueError("FEMA features exceed max_bytes")
        page = response.get("features", [])
        if not page:
            raise ValueError("FEMA query truncated or service changed after discovery")
        for feature in page:
            identifier = feature.get("id", feature.get("properties", {}).get(plan["object_id_field"]))
            if identifier is None or identifier in seen:
                raise ValueError("FEMA pagination returned missing/duplicate feature ids")
            if not feature.get("geometry") or feature["geometry"]["type"] not in {"Polygon", "MultiPolygon"}:
                raise ValueError("FEMA returned a missing or non-polygon footprint")
            seen.add(identifier)
            features.append(feature)
        offset += len(page)
        if offset > max_features or offset > plan["feature_count"]:
            raise ValueError("FEMA feature count changed or exceeded limit")
    collection = {"type": "FeatureCollection", "features": features}
    encoded = json.dumps(collection).encode()
    if len(encoded) > max_bytes:
        raise ValueError("Serialized FEMA collection exceeds byte limit")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(encoded)
    return collection


def rasterize_buildings(features, *, transform, out_shape, raster_crs, feature_crs="EPSG:4326", all_touched=False):
    if isinstance(features, dict) and features.get("type") == "FeatureCollection":
        features = features["features"]
    return rasterize_geometries(features, out_shape=out_shape, transform=transform,
                               src_crs=feature_crs, dst_crs=raster_crs, all_touched=all_touched)
