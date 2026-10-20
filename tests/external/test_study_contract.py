import json
from pathlib import Path

import numpy as np
import pytest

from phasedx_external import protocol28 as P, scorer26, scorer29
from phasedx_external.smoke import prepare, rebind_manifest, run


def test_complete_point_only_rehearsal(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer29, "BOOT", lambda *a: pytest.fail("point-only invoked bootstrap"))
    result = run(tmp_path / "study")
    assert result["label"] == "SYNTHETIC_POINT_ONLY_NOT_CLINICAL"


def test_changed_cohort_rejected_before_labels_even_with_new_receipt(tmp_path):
    args = prepare(tmp_path / "study")
    m = Path(args[4])
    p = m / "candidates.csv"
    p.write_text(p.read_text().replace("FAKE_A,eligible", "FAKE_A,unresolved"))
    rebind_manifest(m)
    new = tmp_path / "new-receipt.json"
    P.make_receipt(str(new), args[1], args[3], args[4])
    args[2] = str(new)
    args[6] = "/nonexistent-label-file-that-must-not-be-opened"
    with pytest.raises(P.AdmissionError, match="manifest differs"):
        scorer29.evaluate(*args)


def test_decoder_identity_is_checked_before_run(tmp_path):
    args = prepare(tmp_path / "study")
    d = json.loads(Path(args[0]).read_text())
    e = json.loads(Path(args[1]).read_text())
    d["preprocessing"].update(decoder_policy="lossless_sv1_pylibjpeg_v1",
                               decoder_sha256="decoder-code", adapter_sha256="adapter-code",
                               decoder_packages={"pydicom": "3.0.2", "pylibjpeg-libjpeg": "2.3.0"})
    dp, ep = tmp_path / "new-design.json", tmp_path / "new-execution.json"
    P.make_design_lock(str(dp), d)
    P.make_execution_lock(str(ep), str(dp), args[4])
    actual = P.intended_identity("primary_image", d, e)
    actual["decoder_packages"]["pydicom"] = "changed"
    # Restore independent intended object; no shared mutable dict can mask the mutation.
    with pytest.raises(P.AdmissionError, match="decoder_packages"):
        P.start_run(str(tmp_path / "bad-run"), "primary_image", str(dp), str(ep), args[4], actual)
    assert not (tmp_path / "bad-run").exists()


def test_ordered_draw_retention_with_fixed_nonrandom_test_indices(monkeypatch):
    class FixedSampler:
        def __init__(self):
            self.rows = iter([["A", "A"], ["A", "B"], ["B", "B"]])

        def choice(self, *args, **kwargs):
            return np.array(next(self.rows))

    monkeypatch.setattr(np.random, "default_rng", lambda *args: FixedSampler())
    r = scorer26.cluster_bootstrap(np.array([1, 0]), np.array([2, 1]),
                                  np.array([1, 2]), np.array(["A", "B"]), 3, 0)
    assert r["draws"] == [None, 1., None]
    assert r["n_undefined_draws"] == 2
    assert r["lo"] == r["hi"] == 1


def test_export_excludes_case_keys_and_draws(tmp_path):
    from phasedx_external.commands import export
    root = tmp_path / "study"
    run(root)
    export(["--result", str(root / "RESULT.json"), "--out", str(tmp_path / "export")])
    text = (tmp_path / "export/aggregate_table.csv").read_text()
    assert "FAKE_" not in text
    assert "draws" not in text
    assert "synthetic_point_only" in text
