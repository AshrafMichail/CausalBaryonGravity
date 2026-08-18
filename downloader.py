"""Checksum-enforced downloader for the allow-listed public inputs."""

from __future__ import annotations

import hashlib
from pathlib import Path
import shutil
import urllib.request

from .sources import SOURCES, Source

CHUNK_BYTES = 1024 * 1024
REQUEST_TIMEOUT_SECONDS = 180
USER_AGENT = "causal-baryon-review-validation/1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_sources(cases: list[str]) -> list[Source]:
    selected = set(cases)
    return [source for source in SOURCES if selected & set(source.cases)]


def download(source: Source, data_dir: Path, force: bool = False) -> None:
    destination = data_dir / source.path
    if destination.exists() and not force:
        actual = sha256(destination)
        if actual != source.sha256:
            raise ValueError(f"checksum mismatch for {destination}: {actual}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(
        source.url,
        headers={"User-Agent": USER_AGENT},
    )
    try:
        with urllib.request.urlopen(
            request, timeout=REQUEST_TIMEOUT_SECONDS
        ) as response:
            with partial.open("wb") as handle:
                shutil.copyfileobj(response, handle, length=CHUNK_BYTES)
        actual = sha256(partial)
        if actual != source.sha256:
            raise ValueError(
                f"checksum mismatch for {destination}: expected "
                f"{source.sha256}, got {actual}"
            )
        partial.replace(destination)
    finally:
        partial.unlink(missing_ok=True)


def download_inputs(
    data_dir: Path, cases: list[str], force: bool = False
) -> None:
    for source in selected_sources(cases):
        download(source, data_dir, force)
        print(f"verified {source.path}")


def verify_present_inputs(data_dir: Path, cases: list[str]) -> None:
    """Require and verify every allow-listed input for the selected cases."""
    for source in selected_sources(cases):
        path = data_dir / source.path
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != source.sha256:
            raise ValueError(
                f"checksum mismatch for {path}: expected "
                f"{source.sha256}, got {actual}"
            )
