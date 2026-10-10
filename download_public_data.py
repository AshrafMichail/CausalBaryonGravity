#!/usr/bin/env python3
"""Optionally download the public SPARC mass-model table with checksum pinning."""

from __future__ import annotations

import argparse
import hashlib
import urllib.request
from pathlib import Path

SPARC_URL = "https://astroweb.cwru.edu/SPARC/MassModels_Lelli2016c.mrt"
SPARC_SHA256 = "9108994b12cc401b94a1768beca61c53ec354779385c9c9cc571049f3043244c"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("public_data"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    destination = args.output_dir / "MassModels_Lelli2016c.mrt"
    temporary = destination.with_suffix(destination.suffix + ".part")
    with urllib.request.urlopen(SPARC_URL) as response:
        temporary.write_bytes(response.read())
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    if digest != SPARC_SHA256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"SPARC checksum mismatch: {digest}")
    temporary.replace(destination)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
