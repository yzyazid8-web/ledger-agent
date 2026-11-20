import io
import tarfile
from pathlib import Path

import pytest

from ledger_agent.snapshot import INCLUDED, SnapshotError, build, extract


def fake_data_dir(root: Path) -> Path:
    data = root / "data"
    for relative in INCLUDED:
        target = data / relative
        if "." in target.name:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(relative, encoding="utf-8")
        else:
            (target / "segment").mkdir(parents=True, exist_ok=True)
            (target / "segment" / "data.bin").write_bytes(b"x" * 10)
    # Fichiers à ne pas embarquer
    (data / "indexes" / "qdrant" / ".lock").write_text("lock", encoding="utf-8")
    (data / "raw").mkdir()
    (data / "raw" / "big.pdf").write_bytes(b"%PDF")
    return data


def test_build_then_extract_roundtrip(tmp_path: Path) -> None:
    data = fake_data_dir(tmp_path)

    lock = build(data, tmp_path / "out", "2026-11")
    extract(tmp_path / "out" / lock["asset"], tmp_path / "restored", lock["sha256"])

    assert lock["tag"] == "data-2026-11"
    restored = tmp_path / "restored"
    assert (restored / "processed" / "facts.duckdb").read_text(encoding="utf-8") == (
        "processed/facts.duckdb"
    )
    assert not (restored / "raw").exists()
    assert not (restored / "indexes" / "qdrant" / ".lock").exists()


def test_tampered_archive_is_refused(tmp_path: Path) -> None:
    lock = build(fake_data_dir(tmp_path), tmp_path / "out", "t")

    with pytest.raises(SnapshotError, match="Empreinte"):
        extract(tmp_path / "out" / lock["asset"], tmp_path / "restored", "0" * 64)


def test_build_requires_complete_data(tmp_path: Path) -> None:
    data = fake_data_dir(tmp_path)
    (data / "processed" / "facts.duckdb").unlink()

    with pytest.raises(SnapshotError, match="manquantes"):
        build(data, tmp_path / "out", "t")


def test_path_traversal_in_archive_is_blocked(tmp_path: Path) -> None:
    archive = tmp_path / "evil.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        payload = b"pwned"
        info = tarfile.TarInfo("../outside.txt")
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))

    from ledger_agent.snapshot import sha256_of

    with pytest.raises(tarfile.TarError):
        extract(archive, tmp_path / "dest", sha256_of(archive))
    assert not (tmp_path / "outside.txt").exists()
