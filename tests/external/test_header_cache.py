import hashlib
import json

import pytest

from phasedx_external import header_cache as C


def fixture(tmp_path, status="complete", identity_failures=0):
    root = tmp_path / "image-root"
    src = {"relpath": "synthetic/a", "sha256": hashlib.sha256(b"a").hexdigest(),
           "size": 1, "source_exam_id": "synthetic", "assignment": "roster_folder"}
    plan, summary, records = [tmp_path / name for name in ("plan.json", "summary.json", "records.jsonl")]
    plan.write_text(json.dumps({"image_root": str(root), "files": [src]}))
    summary.write_text(json.dumps({"status": status, "identity_failures": identity_failures,
                                  "completed": 1, "plan_sha256": C.sha(plan)}))
    records.write_text(json.dumps(dict(src, readable=False, hash_verified=True,
                                      unreadable_reason="synthetic not DICOM")) + "\n")
    shard = {k: str(p) for k, p in zip(("plan", "summary", "records"), (plan, summary, records))}
    shard.update({k + "_sha256": C.sha(shard[k]) for k in ("plan", "summary", "records")})
    index = tmp_path / "index.json"
    index.write_text(json.dumps({"image_root": str(root), "expected_files": 1, "shards": [shard]}))
    return index, root, records


def test_authenticated_cache_usable_without_reopening_image_drive(tmp_path):
    index, root, _ = fixture(tmp_path)
    rows = C.load(index, root)
    assert len(rows) == 1 and not rows[0]["readable"] and not root.exists()


@pytest.mark.parametrize("status,failures", [("blocked_io", 0), ("complete", 1), ("checkpoint", 0)])
def test_io_or_partial_cache_never_becomes_cohort(tmp_path, status, failures):
    index, root, _ = fixture(tmp_path, status, failures)
    with pytest.raises(ValueError, match="partial|failures"):
        C.load(index, root)


def test_cache_mutation_and_other_source_rejected(tmp_path):
    index, root, records = fixture(tmp_path)
    with pytest.raises(ValueError, match="another image"):
        C.load(index, tmp_path / "other")
    records.write_text(records.read_text().replace("synthetic/a", "synthetic/b"))
    with pytest.raises(ValueError, match="identity changed"):
        C.load(index, root)
