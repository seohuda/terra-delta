"""Signing follows the official SDK account/container API without token persistence."""
import pytest

from terradelta.external import naip


def test_actual_asset_account_owns_token_request(monkeypatch):
    calls = []
    def get(url):
        calls.append(url)
        return {"msft:expiry": "2099-01-01T00:00:00Z", "token": "sp=r&sig=temporary"}
    monkeypatch.setattr(naip, "get_json", get)
    url = "https://naipeuwest.blob.core.windows.net/naip/v002/wa/source.tif"
    assert naip.sign_url(url) == url + "?sp=r&sig=temporary"
    assert calls == ["https://planetarycomputer.microsoft.com/api/sas/v1/token/naipeuwest/naip"]


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
