"""Verify historical byte hashes after LF/CRLF conversion on two named files.

The historical experiments recorded operating-system-specific raw bytes. These
two files are text; the approved transformation changes line endings only.
All other inputs retain ordinary byte-exact fingerprint checks.
"""

import hashlib
from pathlib import Path

import common


def _matches_line_endings(data, expected):
    lf = data.replace(b"\r\n", b"\n")
    candidates = (data, lf, lf.replace(b"\n", b"\r\n"))
    return any(hashlib.sha256(candidate).hexdigest() == expected for candidate in candidates)


def matches_historical_text(path, expected):
    """Compare only the shared split manifest or A's historical subset script."""
    source = Path(path).resolve()
    allowed = {(common.SPLITS_DIR / "manifest.json").resolve(),
               (common.ROOT / "learning_subsets.py").resolve()}
    if source not in allowed:
        raise ValueError(f"Line-ending compatibility not permitted for {source}")
    return _matches_line_endings(source.read_bytes(), expected)
