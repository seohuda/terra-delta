"""Mock HTTP only: bounded plans, streaming failures and source/CRS semantics."""

import io
import json
from dataclasses import replace
from urllib.parse import parse_qs, urlsplit

import numpy as np
import pytest
from pyproj import Transformer
from rasterio.transform import from_origin

from terradelta.external import common, fema_structures, hansen, naip
from terradelta.external.common import Asset, DownloadGuard, download_assets, get_json, head_size, plan_summary
from terradelta.utils.geo import pixel_coordinates, rasterize_geometries, validate_bounds


@pytest.fixture(autouse=True)
def deny_live_http(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Tests must never perform live HTTP")
    monkeypatch.setattr(common, "urlopen", forbidden)


class Response(io.BytesIO):
    def __init__(self, body=b"", headers=None):
        super().__init__(body)
        self.headers = headers or {}
        self.read_sizes = []

    def read(self, size=-1):
        self.read_sizes.append(size)
        return super().read(size)


def asset(**options):
    return Asset(**{"id": "tile", "url": "https://example.invalid/tile.tif", "filename": "tile.tif",
                    "source": "NAIP", "size_bytes": 4, **options})


def test_metadata_requests_have_hard_body_cap_and_support_query_and_post(monkeypatch):
    calls = []
    def mock_open(request, timeout):
        calls.append((request, timeout))
        return Response(b'{"count": 2}')
    monkeypatch.setattr(common, "urlopen", mock_open)
    assert get_json("https://example.invalid/query?f=json", params={"returnCountOnly": "true"}, timeout=3) == {"count": 2}
    assert parse_qs(urlsplit(calls[0][0].full_url).query)["returnCountOnly"] == ["true"]
    assert calls[0][1] == 3
    get_json("https://example.invalid/search", payload={"limit": 1})
    assert json.loads(calls[1][0].data) == {"limit": 1}
    oversized = Response(b" " * 10)
    monkeypatch.setattr(common, "urlopen", lambda *a, **k: oversized)
    with pytest.raises(ValueError, match="byte limit"):
        get_json("https://example.invalid", max_bytes=4)
    assert oversized.read_sizes == [5]
    monkeypatch.setattr(common, "urlopen", lambda *a, **k: Response(b'{"error":{"code":400}}'))
    with pytest.raises(ValueError, match="Remote metadata error"):
        get_json("https://example.invalid")


def test_head_size_never_reads_body_or_falls_back_to_get(monkeypatch):
    calls = []
    response = Response(b"imagery-must-not-be-read", {"Content-Length": "42"})
    def mock_open(request, **kwargs):
        calls.append(request.get_method())
        return response
    monkeypatch.setattr(common, "urlopen", mock_open)
    assert head_size("https://example.invalid/image.tif") == 42
    assert calls == ["HEAD"] and response.read_sizes == []
    monkeypatch.setattr(common, "urlopen", lambda *a, **k: Response(headers={}))
    assert head_size("https://example.invalid/image.tif") is None


def test_asset_plan_strips_temporary_sas_and_keeps_unknown_sizes_explicit():
    planned = plan_summary([asset(url="https://example.invalid/tile.tif?sig=secret#token", size_bytes=None,
                                 estimated_bytes=100, region="md", year=2021, resolution_m=0.6)])
    assert planned["dry_run"] and planned["unknown_size_files"] == 1
    assert planned["known_download_bytes"] == 0 and planned["estimated_uncompressed_bytes"] == 100
    assert planned["assets"][0]["url"] == "https://example.invalid/tile.tif"
    assert "secret" not in json.dumps(planned)


@pytest.mark.parametrize("assets,guard,error", [
    ([asset()], DownloadGuard(), PermissionError),
    ([asset()], DownloadGuard(True, max_files=0), ValueError),
    ([asset(), asset(filename="other.tif")], DownloadGuard(True, max_files=1), ValueError),
    ([asset(size_bytes=None, estimated_bytes=1)], DownloadGuard(True), ValueError),
    ([asset(size_bytes=-1)], DownloadGuard(True), ValueError),
    ([asset(size_bytes=5)], DownloadGuard(True, max_file_bytes=4), ValueError),
    ([asset(), asset(filename="other.tif")], DownloadGuard(True, max_total_bytes=7), ValueError),
    ([asset(filename="../escape.tif")], DownloadGuard(True), ValueError),
    ([asset(), asset()], DownloadGuard(True), ValueError),
])
def test_download_preflight_rejects_before_http_or_directory_creation(tmp_path, assets, guard, error):
    output = tmp_path / "raw"
    with pytest.raises(error):
        download_assets(assets, output, guard=guard)
    assert not output.exists()


def test_streaming_success_uses_transient_url_and_no_clobber_publish(tmp_path, monkeypatch):
    requests = []
    def mock_open(request, **kwargs):
        requests.append(request)
        return Response(b"abcd", {"Content-Length": "4"})
    monkeypatch.setattr(common, "urlopen", mock_open)
    paths = download_assets([asset()], tmp_path, guard=DownloadGuard(True),
                            url_resolver=lambda url: url + "?sig=transient")
    assert paths == [tmp_path / "tile.tif"] and paths[0].read_bytes() == b"abcd"
    assert requests[0].full_url.endswith("?sig=transient")
    assert requests[0].get_header("Accept-encoding") == "identity"
    assert not list(tmp_path.glob("*.part"))
    with pytest.raises(FileExistsError):
        download_assets([asset()], tmp_path, guard=DownloadGuard(True))
    assert len(requests) == 1 and paths[0].read_bytes() == b"abcd"


@pytest.mark.parametrize("body,headers,match", [
    (b"abcde", {}, "exceeded planned"), (b"ab", {}, "Truncated"),
    (b"abcd", {"Content-Length": "5"}, "differs from planned"),
])
def test_streaming_size_drift_cleans_partial_and_never_publishes(tmp_path, monkeypatch, body, headers, match):
    response = Response(body, headers)
    monkeypatch.setattr(common, "urlopen", lambda *a, **k: response)
    with pytest.raises(ValueError, match=match):
        download_assets([asset()], tmp_path, guard=DownloadGuard(True, max_file_bytes=4, max_total_bytes=4))
    assert list(tmp_path.iterdir()) == []
    assert all(0 < size <= 5 for size in response.read_sizes)


def test_streaming_network_error_cleans_partial_and_batch_failure_preserves_completed_asset(tmp_path, monkeypatch):
    class Broken(Response):
        def read(self, size=-1):
            if self.tell():
                raise OSError("connection interrupted")
            return super().read(2)
    monkeypatch.setattr(common, "urlopen", lambda *a, **k: Broken(b"abcd"))
    with pytest.raises(OSError, match="interrupted"):
        download_assets([asset()], tmp_path, guard=DownloadGuard(True))
    assert list(tmp_path.iterdir()) == []
    responses = iter([Response(b"abcd"), Response(b"ab")])
    monkeypatch.setattr(common, "urlopen", lambda *a, **k: next(responses))
    with pytest.raises(ValueError, match="Truncated"):
        download_assets([asset(), asset(filename="other.tif")], tmp_path, guard=DownloadGuard(True, max_total_bytes=8))
    assert (tmp_path / "tile.tif").read_bytes() == b"abcd"
    assert not (tmp_path / "other.tif").exists() and not list(tmp_path.glob("*.part"))


def test_download_requires_https_even_when_enabled(tmp_path):
    with pytest.raises(ValueError, match="HTTPS"):
        download_assets([asset(url="http://example.invalid/tile.tif")], tmp_path, guard=DownloadGuard(True))
    assert not list(tmp_path.glob("*.tif"))


def stac_item(identifier, year=2021, state="md", **image_options):
    return {"id": identifier, "properties": {"datetime": f"{year}-06-17T16:00:00Z", "naip:state": state, "gsd": 0.6},
            "assets": {"image": {"href": f"https://example.invalid/{identifier}.tif", "proj:shape": [10, 20], **image_options}}}


def test_naip_discovery_filters_year_state_deduplicates_and_does_not_sign_or_head(monkeypatch):
    calls = []
    pages = iter([
        {"features": [stac_item("a"), stac_item("old", 2018), stac_item("wrong-state", state="va")],
         "links": [{"rel": "next", "href": "https://example.invalid/next"}]},
        {"features": [stac_item("a"), stac_item("b", 2023, **{"file:size": 17})]},
    ])
    def request(url):
        calls.append(url)
        return next(pages)
    def forbidden(*a, **k):
        raise AssertionError("dry-run must not sign or inspect imagery")
    monkeypatch.setattr(naip, "sign_url", forbidden)
    monkeypatch.setattr(naip, "head_size", forbidden)
    result = naip.discover_naip(bounds=(-77, 39, -76, 40), years=[2023, 2021], state="MD", request_json=request)
    assert [a.id for a in result["assets"]] == ["a", "b"]
    assert result["assets"][0].size_bytes is None and result["assets"][0].estimated_bytes == 800
    assert result["assets"][1].size_bytes == 17 and not result["truncated"]
    assert parse_qs(urlsplit(calls[0]).query)["collections"] == ["naip"]
    assert calls[1] == "https://example.invalid/next"


def test_naip_truncation_and_sparse_page_budget_are_explicit():
    next_link = [{"rel": "next", "href": "https://example.invalid/next"}]
    result = naip.discover_naip(bounds=(-77, 39, -76, 40), years=[2021], max_items=1,
                               request_json=lambda url: {"features": [stac_item("a")], "links": next_link})
    assert result["truncated"] and len(result["assets"]) == 1
    calls = []
    def sparse(url):
        calls.append(url)
        return {"features": [stac_item("old", 2018)], "links": next_link}
    result = naip.discover_naip(bounds=(-77, 39, -76, 40), years=[2021], request_json=sparse)
    assert result["truncated"] and not result["assets"] and len(calls) == 20
    with pytest.raises(ValueError, match="POST pagination"):
        naip.discover_naip(bounds=(-77, 39, -76, 40), years=[2021],
                           request_json=lambda url: {"features": [], "links": [{**next_link[0], "method": "POST"}]})


def test_naip_last_page_truncation_counts_items_within_current_page():
    """Regression: previous-page matches must not hide omitted final-page items."""
    pages = iter([
        {"features": [stac_item("a")], "links": [{"rel": "next", "href": "https://example.invalid/next"}]},
        {"features": [stac_item("b"), stac_item("c")]},
    ])
    result = naip.discover_naip(bounds=(-77, 39, -76, 40), years=[2021], max_items=2,
                               request_json=lambda url: next(pages))
    assert [a.id for a in result["assets"]] == ["a", "b"]
    assert result["truncated"]  # c is deliberately omitted by max_items.


def test_naip_head_inspection_signs_only_transiently(monkeypatch):
    seen = []
    monkeypatch.setattr(naip, "sign_url", lambda url: url + "?sig=transient")
    def size(url):
        seen.append(url)
        return 42
    monkeypatch.setattr(naip, "head_size", size)
    result = naip.discover_naip(bounds=(-77, 39, -76, 40), years=[2021], inspect_sizes=True,
                               request_json=lambda url: {"features": [stac_item("a")]})
    assert seen == ["https://example.invalid/a.tif?sig=transient"]
    assert result["assets"][0].size_bytes == 42 and "sig" not in result["assets"][0].url


def fema_request(count=2, pages=None, pagination=True):
    calls = []
    pages = iter(pages or [])
    def request(url, **kwargs):
        calls.append((url, kwargs))
        params = kwargs["params"]
        if not url.endswith("/query"):
            return {"objectIdField": "OBJECTID", "maxRecordCount": 1,
                    "advancedQueryCapabilities": {"supportsPagination": pagination}}
        if params.get("returnCountOnly") == "true":
            return {"count": count}
        return {"type": "FeatureCollection", "features": next(pages)}
    return request, calls


def footprint(identifier):
    return {"type": "Feature", "id": identifier, "properties": {"OBJECTID": identifier},
            "geometry": {"type": "Polygon", "coordinates": [[[-77, 39], [-76.9, 39], [-76.9, 39.1], [-77, 39]]]}}


def test_fema_dry_run_is_metadata_and_count_only():
    request, calls = fema_request()
    plan = fema_structures.discover_fema(bounds=(-77, 39, -76, 40), request_json=request)
    assert plan["license_status"] == "unknown"  # Public REST access cannot approve training rights.
    assert len(calls) == 2 and plan["feature_count"] == 2 and plan["exact_size_bytes"] is None
    assert calls[1][1]["params"]["returnCountOnly"] == "true"
    assert calls[1][1]["params"]["inSR"] == 4326
    with pytest.raises(PermissionError):
        fema_structures.fetch_fema(bounds=(-77, 39, -76, 40), request_json=request)
    assert len(calls) == 2


def test_fema_ordered_pages_request_wgs84_and_publish_without_overwrite(tmp_path):
    request, calls = fema_request(pages=[[footprint(1)], [footprint(2)]])
    output = tmp_path / "buildings.geojson"
    result = fema_structures.fetch_fema(bounds=(-77, 39, -76, 40), output=output, allow_download=True,
                                       max_features=2, max_bytes=4000, page_size=1, request_json=request)
    assert [f["id"] for f in result["features"]] == [1, 2]
    assert json.loads(output.read_text()) == result
    for offset, (_, kwargs) in enumerate(calls[2:]):
        params = kwargs["params"]
        assert params["resultOffset"] == offset and params["outSR"] == 4326
        assert params["orderByFields"] == "OBJECTID ASC" and params["returnGeometry"] == "true"
    assert calls[3][1]["max_bytes"] < calls[2][1]["max_bytes"]
    request, _ = fema_request(pages=[[footprint(1)], [footprint(2)]])
    with pytest.raises(FileExistsError):
        fema_structures.fetch_fema(bounds=(-77, 39, -76, 40), output=output, allow_download=True, request_json=request)
    assert json.loads(output.read_text()) == result


@pytest.mark.parametrize("pages,count,options,match", [
    ([], 3, {"max_features": 2}, "max_features"),
    ([], 2, {"pagination": False}, "cannot paginate"),
    ([[]], 2, {}, "truncated"),
    ([[footprint(1)], [footprint(1)]], 2, {}, "duplicate"),
    ([[{**footprint(1), "id": None, "properties": {}}]], 1, {}, "missing/duplicate"),
    ([[{**footprint(1), "geometry": {"type": "Point", "coordinates": [0, 0]}}]], 1, {}, "non-polygon"),
    ([[footprint(1), footprint(2)]], 1, {}, "count changed"),
    ([[footprint(1)]], 1, {"max_bytes": 10}, "max_bytes"),
])
def test_fema_incomplete_or_over_limit_collections_never_publish(tmp_path, pages, count, options, match):
    options = dict(options)
    request, _ = fema_request(count=count, pages=pages, pagination=options.pop("pagination", True))
    output = tmp_path / "must-not-exist.geojson"
    with pytest.raises(ValueError, match=match):
        fema_structures.fetch_fema(bounds=(-77, 39, -76, 40), output=output, allow_download=True,
                                   request_json=request, **options)
    assert not output.exists()


def test_crs_rasterization_preserves_holes_and_pixel_edge_coordinates():
    transform = from_origin(500000, 4300008, 1, 1)
    outer = [[500001, 4300007], [500007, 4300007], [500007, 4300001], [500001, 4300001], [500001, 4300007]]
    hole = [[500003, 4300005], [500005, 4300005], [500005, 4300003], [500003, 4300003], [500003, 4300005]]
    to_wgs84 = Transformer.from_crs("EPSG:32618", "EPSG:4326", always_xy=True)
    rings = [[list(to_wgs84.transform(x, y)) for x, y in ring] for ring in (outer, hole)]
    collection = {"type": "FeatureCollection", "features": [{"type": "Feature", "properties": {},
                   "geometry": {"type": "Polygon", "coordinates": rings}}]}
    mask = fema_structures.rasterize_buildings(collection, transform=transform, out_shape=(8, 8), raster_crs="EPSG:32618")
    expected = np.zeros((8, 8), np.uint8)
    expected[1:7, 1:7] = 1
    expected[3:5, 3:5] = 0
    assert np.array_equal(mask, expected)
    pixels = pixel_coordinates(rings[0][:2], transform, src_crs="EPSG:4326", raster_crs="EPSG:32618")
    np.testing.assert_allclose(pixels, [[1, 1], [7, 1]], atol=1e-6)
    empty = rasterize_geometries([], out_shape=(8, 8), transform=transform, src_crs="EPSG:4326", dst_crs="EPSG:32618")
    assert not empty.any()
    with pytest.raises(ValueError, match="pixel limit"):
        rasterize_geometries([], out_shape=(8, 8), transform=transform, src_crs="EPSG:4326", dst_crs="EPSG:32618", max_pixels=63)


def test_hansen_granule_conventions_limits_and_candidate_only_plan(monkeypatch):
    monkeypatch.setattr(hansen, "head_size", lambda url: pytest.fail("dry-run must not HEAD"))
    assets = hansen.discover_hansen(bounds=(-80, 30, -70, 40), layers=("lossyear", "first"), max_tiles=1)
    assert len(assets) == 2
    assert assets[0].filename == "Hansen_GFC-2024-v1.12_lossyear_40N_080W.tif"
    assert assets[0].url.endswith("/GFC-2024-v1.12/" + assets[0].filename)
    assert all(a.purpose == "candidate_mining_only" and a.size_bytes is None for a in assets)
    assert assets[1].estimated_bytes == assets[0].estimated_bytes * 4
    south = hansen.discover_hansen(bounds=(10, -20, 20, -10))
    assert south[0].region == "10S_010E"
    with pytest.raises(ValueError, match="max_tiles"):
        hansen.discover_hansen(bounds=(-81, 29, -69, 41), max_tiles=1)
    with pytest.raises(ValueError, match="Duplicate"):
        hansen.discover_hansen(bounds=(-80, 30, -70, 40), layers=("lossyear", "lossyear"))


def test_hansen_interval_excludes_pre_year_water_and_low_cover():
    years = np.array([[0, 20, 21, 22, 23, 24], [21, 22, 23, 24, 1, 19]], np.uint8)
    cover = np.full_like(years, 70)
    cover[0, 3] = 20
    datamask = np.ones_like(years)
    datamask[1, 2] = 2
    result = hansen.loss_candidates(years, year_pre=2020, year_post=2023, treecover2000=cover, datamask=datamask)
    expected = np.array([[0, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 0]], bool)
    assert result.dtype == bool and np.array_equal(result, expected)
    assert np.array_equal(years, [[0, 20, 21, 22, 23, 24], [21, 22, 23, 24, 1, 19]])
    for invalid in (years.astype(float), np.full_like(years, 25), years[..., None]):
        with pytest.raises(ValueError, match="integer Hansen"):
            hansen.loss_candidates(invalid, year_pre=2020, year_post=2023)
    with pytest.raises(ValueError, match="pinned"):
        hansen.loss_candidates(years, year_pre=2023, year_post=2025)


@pytest.mark.parametrize("bounds", [(1, 0, 0, 1), (-181, 0, -180, 1), (0, 0, 1, float("nan"))])
def test_invalid_aoi_rejected_before_discovery(bounds):
    with pytest.raises(ValueError):
        validate_bounds(bounds, geographic=True)


def test_estimates_describe_raw_storage_not_download_authorization():
    assert common.estimate_raster_bytes((0, 0, 10, 20), 1, 3) == 600
    rgb = naip.estimate_aoi_bytes((-76.61, 39.29, -76.60, 39.30), bands="rgb")
    nir = naip.estimate_aoi_bytes((-76.61, 39.29, -76.60, 39.30), bands="nir")
    assert rgb == 3 * nir and rgb > 0
    assert replace(asset(), size_bytes=None).estimated_bytes is None
