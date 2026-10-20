"""One tiny real-CLI integration, with handwritten model responses and no RNG."""
import json
from pathlib import Path

import numpy as np
import pytest
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import CTImageStorage, ExplicitVRLittleEndian

from phasedx_external import commands, protocol28 as P, prospective_guard
from phasedx_external import build_manifest28, predict_image28, predict_geometry28
from phasedx_external import image_adapter27 as IA, geometry_adapter27 as GA, bgeom
from phasedx_external.reference28 import RULE_ID
from phasedx_external.selection28 import COLS, CLIN
from phasedx_external.smoke import write_csv


def test_real_cli_chain_with_stubbed_models(tmp_path, monkeypatch):
    # Registration is tested separately through actual child processes. Here
    # parser, manifest, selection, predictor and scorer paths are exercised.
    monkeypatch.setattr(prospective_guard, "require_registration", lambda *a: None)
    root = tmp_path / "images"
    keys = ["FAKE_A", "FAKE_B", "FAKE_C", "FAKE_D", "FAKE_E", "FAKE_F"]
    selection = []
    for k, key in enumerate(keys, 1):
        folder = root / key
        folder.mkdir(parents=True)
        study, series = f"1.2.826.1.{k}", f"1.2.826.2.{k}"
        for j in range(2):
            meta = FileMetaDataset()
            meta.TransferSyntaxUID = ExplicitVRLittleEndian
            meta.MediaStorageSOPClassUID = CTImageStorage
            meta.MediaStorageSOPInstanceUID = f"1.2.826.3.{k}.{j}"
            ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0"*128)
            ds.StudyInstanceUID, ds.SeriesInstanceUID = study, series
            ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
            ds.Modality, ds.InstanceNumber = "CT", j+1
            ds.ImagePositionPatient, ds.ImageOrientationPatient = [0, 0, j*k], [1, 0, 0, 0, 1, 0]
            ds.Rows, ds.Columns = 2, 2
            ds.BitsAllocated, ds.BitsStored, ds.PixelRepresentation = 16, 16, 1
            ds.RescaleSlope, ds.RescaleIntercept = 1, -1024
            ds.PixelData = b"\0"*8
            ds.save_as(folder / f"{j}.dcm", enforce_file_format=True)
        selection.append(dict(source_exam_id=key, study_uid=study, series_uid=series,
                              **{c: "unknown" if key == "FAKE_F" else "true" for c in CLIN},
                              evidence_locator="FABRICATED", selector="fixture",
                              review_status="fabricated", reason="not clinical review"))
    roster, policy, review = [tmp_path / n for n in ("roster.csv", "policy.json", "review.csv")]
    write_csv(roster, ["source_exam_id"], [{"source_exam_id": k} for k in keys])
    policy.write_text('{"modality":"CT","decoder_policy":"uncompressed_v1"}')
    write_csv(review, COLS, selection)
    unselected, selected = tmp_path / "unselected", tmp_path / "selected"
    build_manifest28.main(["--roster", str(roster), "--root", str(root),
                           "--policy", str(policy), "--out", str(unselected)])
    assert P.verify_manifest_dir(str(unselected))
    commands.select(["--manifest", str(unselected), "--review", str(review),
                     "--images", str(root), "--out", str(selected)])
    fixture_ck = {}
    for role in P.ROLES_ALL:
        d = tmp_path / (role + "_weights")
        d.mkdir()
        (d / "best_model.pt").write_bytes(("FABRICATED_"+role).encode())
        fixture_ck[role] = d
    geometry_file = fixture_ck["geometry"] / "best_model.pt"
    design = {
        "protocol_id": "CLI_SYNTHETIC_NO_NEURAL_EXECUTION", "mode": "synthetic_point_only",
        "models": {role: {"model_id": mid, "checkpoint_sha256": P.sha_file(fixture_ck[role]/"best_model.pt")}
                   for role, mid in zip(P.ROLES_ALL, ("I1_seed42", "I1_seed123",
                       "G2_tree_transported_rank_reconstructed_frozen"))},
        "preprocessing": {"recipe_sha256": P.sha_file(Path(IA.__file__).parent/"a_ct_recipe_b2_v7_verbatim.py"),
                          "precision_path": IA.PRECISION_PATH_ID},
        "aggregation": "max_slice_logit",
        "representation": {"representation_id": "TRANSPORTED_PHYSICAL_RANK_FROZEN_RECONSTRUCTED_G2"},
        "label_rule_id": RULE_ID, "grouping_policy": "fallback_if_linkage_unavailable",
        "estimand": "examination-weighted paired AUROC difference",
    }
    dp, ep, ctl = tmp_path/"design.json", tmp_path/"execution.json", tmp_path/"control.json"
    P.make_design_lock(str(dp), design)
    commands.lock(["--design", str(dp), "--manifest", str(selected), "--out", str(ep)])
    ctl.write_text('{"deadline_utc":"2099-01-01T00:00:00+00:00"}')
    def fake_load(path, expected):
        assert P.sha_file(path) == expected
        return ("primary" if "primary_image" in path else "secondary"), {"fixture": True}
    def fake_score(model, files, *args, **kwargs):
        name = Path(files[0]["dicom_path"]).parent.name
        score = dict(zip(keys, [4, 3, 2, 1, None, None] if model == "primary"
                        else [4, None, 2, 1, 3, None]))[name]
        return {"score": float("nan") if score is None else score,
                "status": "failed" if score is None else "scored_complete",
                "n_listed": len(files), "n_scored": 0 if score is None else len(files),
                "n_rejected": len(files) if score is None else 0, "reasons": {}}
    monkeypatch.setattr(IA, "load_model", fake_load)
    monkeypatch.setattr(IA, "score_exam", fake_score)
    runs = {}
    for role in P.ROLES_ALL[:2]:
        rd = tmp_path/role
        with pytest.raises(SystemExit) as exit_:
            predict_image28.main(["--design-lock", str(dp), "--execution-lock", str(ep),
                "--role", role, "--manifest-dir", str(selected), "--checkpoint-dir", str(fixture_ck[role]),
                "--cycle-control", str(ctl), "--out", str(rd)])
        assert exit_.value.code == 0
        runs[role] = [str(rd), str(rd/"predictions.csv")]
    class HandwrittenGeometry:
        def decision_function(self, x):
            # The two-slice fabricated series have spans 1,2,...,6.
            return np.array([x[0, bgeom.ALL_COLS.index("span_median_mm")]])
    monkeypatch.setattr(GA, "load_model", lambda *a: {
        "imputer_medians": {k: 0. for k in bgeom.CONT_COLS}, "hgb": HandwrittenGeometry()})
    rd = tmp_path/"geometry"
    predict_geometry28.main(["--design-lock", str(dp), "--execution-lock", str(ep),
        "--manifest-dir", str(selected), "--model", str(geometry_file), "--out", str(rd)])
    runs["geometry"] = [str(rd), str(rd/"predictions.csv")]
    rp, receipt = tmp_path/"runs.json", tmp_path/"receipt.json"
    rp.write_text(json.dumps(runs))
    commands.receipt(["--execution-lock", str(ep), "--manifest", str(selected),
                      "--runs", str(rp), "--out", str(receipt)])
    schema, reads = tmp_path/"schema.json", tmp_path/"reads.csv"
    schema.write_text(json.dumps({
        "authenticated": True, "provenance": {"locator": "FABRICATED"},
        "exam_key_col": "name", "key_transform": "identity", "reader_cols": ["r1", "r2", "r3"],
        "n_readers": 3, "rule_id": RULE_ID, "missing_policy": "two_agreeing_valid_reads_else_missing"}))
    write_csv(reads, ["name", "r1", "r2", "r3"],
              [{"name": k, "r1": y, "r2": y, "r3": y} for k,y in zip(keys,[1,1,0,0,1,0])])
    args = ["--design", str(dp), "--execution-lock", str(ep), "--receipt", str(receipt),
            "--runs", str(rp), "--manifest", str(selected), "--schema", str(schema)]
    commands.reference(args+["--reads", str(reads), "--out", str(tmp_path/"reference")])
    result = tmp_path/"result.json"
    commands.score(args+["--reference", str(reads), "--out", str(result)])
    r = json.loads(result.read_text())
    assert r["primary"]["difference"] == 1
    assert r["eligible_labeled_denominator"] == 5
    assert r["flow_partition"]["unresolved_eligibility"] == 1
    assert r["secondary_seed123"]["difference"] == .5
    assert r["bootstrap"]["status"] == "not_applicable_point_only_mode"
    commands.flow(["--result", str(result), "--out", str(tmp_path/"flow.json")])
    commands.export(["--result", str(result), "--out", str(tmp_path/"export")])
    assert "FAKE_" not in (tmp_path/"export/aggregate_table.csv").read_text()
