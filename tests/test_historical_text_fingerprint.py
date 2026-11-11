"""Restrict line-ending equivalence to the two historic text inputs."""

import hashlib
from pathlib import Path

import pytest

import common
from B_Traditional_ML.historical_text_fingerprint import (
    _matches_line_endings, matches_historical_text,
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def test_line_ending_equivalence_and_tampering():
    lf = b'{"seed": 67,\n"sample_count": 12000}\n'
    crlf = lf.replace(b"\n", b"\r\n")
    assert _matches_line_endings(lf, digest(crlf))
    assert _matches_line_endings(crlf, digest(lf))
    assert _matches_line_endings(lf, digest(lf))
    assert not _matches_line_endings(lf.replace(b"67", b"68"), digest(crlf))
    assert not _matches_line_endings(lf + b" ", digest(crlf))


def test_compatibility_only_accepts_two_named_sources():
    with pytest.raises(ValueError):
        matches_historical_text(common.ROOT / "common.py", "0" * 64)
    for path in (common.SPLITS_DIR / "manifest.json", common.ROOT / "learning_subsets.py"):
        assert matches_historical_text(path, digest(Path(path).read_bytes()))
