"""Small HTTP metadata reads and atomic opt-in downloads with hard byte limits."""

from __future__ import annotations

import json
import math
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlencode, urlsplit, urlunsplit
from urllib.request import Request, urlopen


def public_url(url):
    """Remove temporary authorization/query data from persisted discovery records."""
    parsed = urlsplit(url)
    return urlunsplit((parsed.scheme, parsed.netloc, parsed.path, "", ""))


def get_json(url, *, params=None, payload=None, timeout=30, max_bytes=4_000_000):
    if params:
        url += ("&" if "?" in url else "?") + urlencode(params)
    data = json.dumps(payload).encode() if payload is not None else None
    request = Request(url, data=data, headers={"User-Agent": "TerraDelta/0.1 metadata",
                                             "Content-Type": "application/json"})
    with urlopen(request, timeout=timeout) as response:
        raw = response.read(max_bytes + 1)
    if len(raw) > max_bytes:
        raise ValueError("Metadata response exceeded byte limit")
    result = json.loads(raw)
    if isinstance(result, dict) and result.get("error"):
        raise ValueError(f"Remote metadata error: {result['error']}")
    return result


def head_size(url, *, timeout=30):
    """HEAD only; never falls back to a potentially expensive GET."""
    with urlopen(Request(url, method="HEAD"), timeout=timeout) as response:
        length = response.headers.get("Content-Length")
    return int(length) if length and length.isdigit() else None


def safe_filename(value):
    value = re.sub(r"[^A-Za-z0-9_.-]", "_", str(value))
    if value in {"", ".", ".."}:
        raise ValueError("Invalid asset id")
    return value


@dataclass(frozen=True)
class Asset:
    id: str
    url: str
    filename: str
    source: str
    size_bytes: int | None = None
    estimated_bytes: int | None = None
    region: str = ""
    year: int | None = None
    resolution_m: float | None = None
    license_status: str = "unknown"
    purpose: str = "imagery"

    def metadata(self):
        return {**asdict(self), "url": public_url(self.url)}


@dataclass(frozen=True)
class DownloadGuard:
    enabled: bool = False
    max_files: int = 2
    max_total_bytes: int = 100_000_000
    max_file_bytes: int = 50_000_000

    def check(self, assets):
        if not self.enabled:
            raise PermissionError("Downloads require explicit --download; dry-run is the default")
        if any(value <= 0 for value in (self.max_files, self.max_total_bytes, self.max_file_bytes)):
            raise ValueError("Download limits must be positive")
        if len(assets) > self.max_files:
            raise ValueError("Discovery exceeds max_files; narrow AOI/year or increase explicit bound")
        total = 0
        filenames = set()
        for asset in assets:
            if asset.filename != Path(asset.filename).name or asset.filename in {"", ".", ".."}:
                raise ValueError("Asset filename must be a safe basename")
            if asset.filename in filenames:
                raise ValueError("Duplicate output filenames")
            filenames.add(asset.filename)
            if asset.size_bytes is None:
                raise ValueError("Unknown exact download size; inspect HEAD metadata before downloading")
            if asset.size_bytes < 0 or asset.size_bytes > self.max_file_bytes:
                raise ValueError("Asset exceeds max_file_bytes")
            total += asset.size_bytes
        if total > self.max_total_bytes:
            raise ValueError("Assets exceed max_total_bytes")


def download_assets(assets, output_dir, *, guard=None, url_resolver=None, timeout=30):
    """No files/network calls before guard checks; .part files removed on failure.

    Size is required from server metadata, not an uncompressed estimate. Responses
    must match planned Content-Length and bytes read. Authorization URLs are used
    transiently through url_resolver and never written into manifests.
    """
    assets = list(assets)
    guard = guard or DownloadGuard()
    guard.check(assets)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths, total = [], 0
    for asset in assets:
        destination = output_dir / asset.filename
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite: {destination}")
        url = url_resolver(asset.url) if url_resolver else asset.url
        parsed = urlsplit(url)
        if parsed.scheme != "https":
            raise ValueError("Downloads require HTTPS")
        temporary = None
        try:
            with urlopen(Request(url, headers={"Accept-Encoding": "identity"}), timeout=timeout) as response:
                length = response.headers.get("Content-Length")
                if length and int(length) != asset.size_bytes:
                    raise ValueError("Content-Length differs from planned size; rediscover before downloading")
                with tempfile.NamedTemporaryFile(dir=output_dir, prefix=asset.filename + ".", suffix=".part", delete=False) as handle:
                    temporary = Path(handle.name)
                    received = 0
                    while True:
                        # Read at most one byte past the guard, including size drift.
                        limit = min(guard.max_file_bytes, asset.size_bytes, guard.max_total_bytes - total)
                        block = response.read(min(1_048_576, limit - received + 1))
                        if not block:
                            break
                        received += len(block)
                        if received > limit:
                            raise ValueError("Streaming download exceeded planned byte bound")
                        handle.write(block)
                    if received != asset.size_bytes:
                        raise ValueError("Truncated download")
                # Atomic no-clobber publish on the same filesystem.
                os.link(temporary, destination)
                total += received
                paths.append(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    return paths


def plan_summary(assets):
    assets = list(assets)
    known = [a.size_bytes for a in assets if a.size_bytes is not None]
    estimated = [a.estimated_bytes for a in assets if a.estimated_bytes is not None]
    return {"dry_run": True, "assets": [a.metadata() for a in assets], "files": len(assets),
            "known_download_bytes": sum(known), "unknown_size_files": len(assets) - len(known),
            "estimated_uncompressed_bytes": sum(estimated),
            "size_note": "Uncompressed estimates are not wire sizes or download authorization."}


def estimate_raster_bytes(bounds, resolution, bands=4, dtype_bytes=1):
    from terradelta.utils.geo import validate_bounds
    west, south, east, north = validate_bounds(bounds)
    if not math.isfinite(resolution) or resolution <= 0 or bands <= 0:
        raise ValueError("Resolution/bands must be positive")
    return math.ceil((east - west) / resolution) * math.ceil((north - south) / resolution) * bands * dtype_bytes
