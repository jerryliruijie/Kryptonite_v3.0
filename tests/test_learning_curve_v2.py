from pathlib import Path

import pytest

from B_Traditional_ML import learning_curve_v2


def test_rejects_nonempty_unidentified_directory(tmp_path, monkeypatch):
    marker = tmp_path / "unrelated.txt"
    marker.write_text("keep", encoding="utf-8")
    called = []
    monkeypatch.setattr(learning_curve_v2.learning_curve, "run", called.append)
    with pytest.raises(FileExistsError, match="without manifest"):
        learning_curve_v2.run(tmp_path)
    assert not called
    assert marker.read_text(encoding="utf-8") == "keep"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["unrelated.txt"]


@pytest.mark.parametrize("with_manifest", [False, True])
def test_delegates_empty_or_manifest_directory(tmp_path, monkeypatch, with_manifest):
    if with_manifest:
        (tmp_path / "manifest.json").write_text("{}", encoding="utf-8")
    called = []
    monkeypatch.setattr(learning_curve_v2.learning_curve, "run", called.append)
    learning_curve_v2.run(tmp_path)
    assert called == [Path(tmp_path).resolve()]
