"""Signing follows the official SDK account/container API without token persistence."""
import pytest
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

from terradelta.external import naip


@pytest.fixture(autouse=True)
def clear_token_cache():
    naip._TOKEN_CACHE.clear()
    yield
    naip._TOKEN_CACHE.clear()


def test_actual_asset_account_owns_token_request(monkeypatch):
    calls = []
    def get(url):
        calls.append(url)
        return {"msft:expiry": "2099-01-01T00:00:00Z", "token": "sp=r&sig=temporary"}
    monkeypatch.setattr(naip, "get_json", get)
    url = "https://naipeuwest.blob.core.windows.net/naip/v002/wa/source.tif"
    assert naip.sign_url(url) == url + "?sp=r&sig=temporary"
    assert calls == ["https://planetarycomputer.microsoft.com/api/sas/v1/token/naipeuwest/naip"]


def test_concurrent_account_requests_share_one_transient_token(monkeypatch):
    calls = []
    def get(url):
        calls.append(url)
        return {"msft:expiry": "2099-01-01T00:00:00Z", "token": "sig=temporary"}
    monkeypatch.setattr(naip, "get_json", get)
    urls = [f"https://naipeuwest.blob.core.windows.net/naip/{i}.tif" for i in range(12)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        signed = list(executor.map(naip.sign_url, urls))
    assert signed == [u + "?sig=temporary" for u in urls]
    assert len(calls) == 1


def test_nearly_expired_cache_is_refreshed(monkeypatch):
    naip._TOKEN_CACHE["naipeuwest"] = (datetime.now(timezone.utc) + timedelta(seconds=30), "sig=old")
    monkeypatch.setattr(naip, "get_json", lambda url: {"msft:expiry": "2099-01-01T00:00:00Z", "token": "sig=fresh"})
    assert naip.sign_url("https://naipeuwest.blob.core.windows.net/naip/a.tif").endswith("?sig=fresh")


@pytest.mark.parametrize("url", [
    "http://naipeuwest.blob.core.windows.net/naip/a.tif",
    "https://naipeuwest.blob.core.windows.net.evil.invalid/naip/a.tif",
    "https://naipeuwest.blob.core.windows.net/other/a.tif",
    "https://naipeuwest.blob.core.windows.net/naip/a.tif?sig=old",
    "https://naipeuwest.blob.core.windows.net/naip/a.tif#fragment",
])
def test_invalid_signing_target_rejected_before_request(url, monkeypatch):
    monkeypatch.setattr(naip, "get_json", lambda *a, **kw: pytest.fail("no request for invalid target"))
    with pytest.raises(ValueError, match="unsigned HTTPS"):
        naip.sign_url(url)


@pytest.mark.parametrize("response", [
    {"msft:expiry": "2000-01-01T00:00:00Z", "token": "expired"},
    {"msft:expiry": "2099-01-01T00:00:00", "token": "naive"},
    {"msft:expiry": "2099-01-01T00:00:00Z", "token": ""},
])
def test_expired_or_invalid_token_is_not_used(response, monkeypatch):
    monkeypatch.setattr(naip, "get_json", lambda *a, **kw: response)
    with pytest.raises(ValueError, match="expired/invalid"):
        naip.sign_url("https://naipeuwest.blob.core.windows.net/naip/a.tif")
