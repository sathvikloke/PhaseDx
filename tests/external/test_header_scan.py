import importlib.util
import io
import hashlib
import json
from pathlib import Path

import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

spec = importlib.util.spec_from_file_location("header_scan", Path(__file__).resolve().parents[2] / "tools/scan_header_shard.py")
M = importlib.util.module_from_spec(spec)
spec.loader.exec_module(M)


def test_header_fields_without_pixel_decode():
    meta = FileMetaDataset()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    meta.MediaStorageSOPClassUID = CTImageStorage
    meta.MediaStorageSOPInstanceUID = "1.2.3.4"
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0"*128)
    ds.StudyInstanceUID, ds.SeriesInstanceUID, ds.SOPInstanceUID = "1.2.3", "1.2.3.1", "1.2.3.4"
    ds.Modality = "CT"
    ds.ImagePositionPatient, ds.ImageOrientationPatient = [0, 0, 5], [1, 0, 0, 0, 1, 0]
    ds.InstanceNumber = 2
    ds.Rows, ds.Columns = 2, 2
    ds.BitsAllocated, ds.BitsStored, ds.PixelRepresentation = 16, 12, 1
    ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
    # Deliberately invalid pixel payload: a header-only reader still succeeds.
    ds.PixelData = b"xx"
    b = io.BytesIO()
    ds.save_as(b, enforce_file_format=True)
    h = M.header(b.getvalue())
    assert h["ipp2"] == 5 and h["iop4"] == 1
    assert h["n_iop"] == 6 and h["n_ipp"] == 3
    assert h["transfer_syntax"] == str(ExplicitVRLittleEndian)
    assert h["has_rescale"] is True
    assert "PixelData" not in h


def test_unreadable_source_is_not_invented_header():
    with pytest.raises(Exception):
        M.header(b"not a DICOM file")


def plan_for(tmp_path, root, files):
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({
        "script_sha256": hashlib.sha256(Path(M.__file__).read_bytes()).hexdigest(),
        "image_root": str(root), "run_root": str(tmp_path), "deadline_utc": "2099-01-01T00:00:00Z",
        "files": [{"relpath": name, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                   "source_exam_id": "synthetic", "assignment": "roster_folder"} for name, data in files]
    }))
    return plan


def test_missing_drive_is_blocked_and_does_not_mark_cases_done(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "require_registration", lambda *a: None)
    root = tmp_path / "absent"
    plan = plan_for(tmp_path, root, [("fake.dcm", b"bad header")])
    out = tmp_path / "out"
    assert M.main(["--plan", str(plan), "--out", str(out)]) == 92
    summary = json.loads((out / "SUMMARY.json").read_text())
    assert summary["status"] == "blocked_io"
    assert summary["completed"] == summary["parse_failures"] == summary["identity_failures"] == 0
    assert not (out / "headers.jsonl").exists()
    # After reconnection the same locked source is read; its real malformed
    # header can then be reported, independently of the earlier I/O failure.
    root.mkdir()
    (root / "fake.dcm").write_bytes(b"bad header")
    assert M.main(["--plan", str(plan), "--out", str(out)]) == 0
    summary = json.loads((out / "SUMMARY.json").read_text())
    assert summary["completed"] == summary["parse_failures"] == 1
    assert summary["identity_failures"] == 0


def test_mid_scan_disconnect_preserves_progress_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "require_registration", lambda *a: None)
    root = tmp_path / "source"
    root.mkdir()
    (root / "a").write_bytes(b"first")
    plan = plan_for(tmp_path, root, [("a", b"first"), ("b", b"second")])
    out = tmp_path / "out"
    assert M.main(["--plan", str(plan), "--out", str(out)]) == 92
    s = json.loads((out / "SUMMARY.json").read_text())
    assert s["status"] == "blocked_io" and s["completed"] == 1
    assert s["io_failure"]["relpath"] == "b"
    (root / "b").write_bytes(b"second")
    assert M.main(["--plan", str(plan), "--out", str(out)]) == 0
    s = json.loads((out / "SUMMARY.json").read_text())
    assert s["completed"] == 2 and s["new_files"] == 1
    assert len((out / "headers.jsonl").read_text().splitlines()) == 2


def test_changed_source_cannot_be_complete_success(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "require_registration", lambda *a: None)
    root = tmp_path / "source"
    root.mkdir()
    (root / "a").write_bytes(b"changed")
    plan = plan_for(tmp_path, root, [("a", b"original")])
    out = tmp_path / "out"
    assert M.main(["--plan", str(plan), "--out", str(out)]) == 92
    s = json.loads((out / "SUMMARY.json").read_text())
    assert s["status"] == "identity_failure" and s["identity_failures"] == 1
    assert s["parse_failures"] == 0
