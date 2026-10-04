"""Temporal availability is observed metadata, never guessed years or downloaded imagery."""
import importlib.util
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from terradelta.external import naip


def item(year, *, identifier=None, bounds=(-1, -1, 2, 2), state="md", acquired=None):
    w, s, e, n = bounds
    return {"id": identifier or f"tile_{year}", "properties": {
        "datetime": acquired or f"{year}-06-01T00:00:00Z", "naip:state": state, "gsd": .6},
        "geometry": {"type": "Polygon", "coordinates": [[[w, s], [e, s], [e, n], [w, n], [w, s]]]},
        "assets": {"image": {"href": f"https://example.invalid/tile_{year}.tif", "proj:shape": [256, 256]}}}


def test_years_use_acquisition_dates_and_return_only_observed_full_aoi_two_to_four_year_options(monkeypatch):
    monkeypatch.setattr(naip, "sign_url", lambda *a: pytest.fail("year discovery must not sign"))
    monkeypatch.setattr(naip, "head_size", lambda *a: pytest.fail("year discovery must not HEAD"))
    urls = []
    def request(url):
        urls.append(url)
        return {"features": [item(2018), item(2020), item(2021), item(2023), item(2025, state="va")], "links": []}
    report = naip.discover_naip_years(bounds=(0, 0, 1, 1), state="md", request_json=request)
    assert report["observed_years"] == [2018, 2020, 2021, 2023]
    assert {(p["year_pre"], p["year_post"]) for p in report["temporal_options"]} == {(2018, 2020), (2018, 2021), (2020, 2023), (2021, 2023)}
    assert report["object_count"] == 4 and report["estimated_download_bytes"] is None
    assert report["payload_downloaded_bytes"] == 0
    params = parse_qs(urlsplit(urls[0]).query)
    assert params["collections"] == ["naip"]
    assert len(urls) == 1 and report["truncated"] is False


def test_intersection_does_not_establish_same_aoi_temporal_coverage():
    def request(url):
        return {"features": [item(2019), item(2021, bounds=(0, 0, .5, .5))]}
    report = naip.discover_naip_years(bounds=(0, 0, 1, 1), request_json=request)
    assert report["observed_years"] == [2019, 2021]
    assert report["temporal_options"] == []
    assert report["years"][1]["items"][0]["covers_full_aoi"] is False


def test_truncated_listing_is_explicit_and_not_missing_year_proof():
    report = naip.discover_naip_years(bounds=(0, 0, 1, 1), max_items=2,
        request_json=lambda url: {"features": [item(2019), item(2021), item(2023)]})
    assert report["truncated"] and report["object_count"] == 2


def test_pagination_and_dedup_preserve_actual_years():
    pages = iter([{"features": [item(2019)], "links": [{"rel": "next", "href": "https://example.invalid/next"}]},
                  {"features": [item(2019), item(2022)], "links": []}])
    report = naip.discover_naip_years(bounds=(0, 0, 1, 1), request_json=lambda url: next(pages))
    assert report["observed_years"] == [2019, 2022] and report["object_count"] == 2


@pytest.mark.parametrize("features", [[], [item(2020)]])
def test_empty_and_single_timestamp_never_yield_a_temporal_pair(features):
    report = naip.discover_naip_years(bounds=(0, 0, 1, 1), request_json=lambda url: {"features": features})
    assert report["temporal_options"] == []


def test_cli_list_years_is_metadata_only_and_accepts_remote_path_without_creating_it(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("naip_cli", Path(__file__).parents[1] / "scripts/download_naip.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "discover_naip_years", lambda **kwargs: {"observed_years": [2019, 2021], "payload_downloaded_bytes": 0})
    report = module.main(["--bounds", "0", "0", "1", "1", "--list-years", "--dry-run", "--output", str(tmp_path / "remote")])
    assert report["observed_years"] == [2019, 2021] and not (tmp_path / "remote").exists()
    for forbidden in (["--download"], ["--inspect-sizes"], ["--metadata-plan", "old.json"]):
        with pytest.raises(SystemExit):
            module.main(["--bounds", "0", "0", "1", "1", "--list-years", *forbidden])
    with pytest.raises(SystemExit):
        module.main(["--bounds", "0", "0", "1", "1", "--list-years", "--years", "2021"])
