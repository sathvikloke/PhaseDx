import importlib
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
PKG = ROOT / "src/phasedx_external"


def test_imports_have_no_execution(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    for path in PKG.glob("*.py"):
        if path.stem == "__main__":
            continue
        importlib.import_module("phasedx_external." + path.stem)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("cmd", ["manifest", "predict-image", "predict-geometry",
                                 "acquire", "extract", "select", "lock", "receipt", "reference", "score", "flow", "export", "smoke"])
def test_cli_help_is_read_only(cmd, tmp_path):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), PYTHONDONTWRITEBYTECODE="1")
    p = subprocess.run([sys.executable, "-m", "phasedx_external", cmd, "--help"],
                       cwd=tmp_path, env=env, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr
    assert "usage:" in p.stdout
    assert list(tmp_path.iterdir()) == []


def test_completion_bounds_preserve_full_denominator():
    from phasedx_external.scorer26 import completion_bounds
    y = np.array([1, 1, 0, 0])
    a = np.array([4., 3., 2., 1.])
    b = np.array([1., 2., 3., 4.])
    full = completion_bounds(y, a, b, np.ones(4, dtype=bool))
    assert full  # Exact schema assertions are in the historical eight-case fixture.
    partial = completion_bounds(y, np.array([4., np.nan, 2., 1.]), b, np.array([True, False, True, True]))
    assert partial != full


def test_geometry_rank_and_support_are_not_exclusions():
    from phasedx_external.geometry_adapter27 import series_row
    import pandas as pd
    data = dict(ipp0=[0, 0, 0], ipp1=[0, 0, 0], ipp2=[0, 5, 10],
                iop0=[1]*3, iop1=[0]*3, iop2=[0]*3, iop3=[0]*3,
                iop4=[1]*3, iop5=[0]*3, instance_number=[11, 12, 14])
    row, images = series_row("synthetic-series", "synthetic-exam", pd.DataFrame(data))
    assert row["orientation_valid"]
    assert row["instance_number_gaps"] == 1
    assert images.bin20.tolist() == [0, 10, 19]
    assert "out_of_dev_support:n_slices" in row["support_flags"]


def test_missing_clinical_qualification_is_not_selected():
    from phasedx_external.selection28 import select
    series = [{"series_key": "S", "technical_usable": True, "technical_reason": "",
               "span_mm": 100, "median_gap_mm": 5}]
    assert select("E", series, {})[1] == "clinical_unresolved"


def test_invalid_grouping_does_not_fall_back(tmp_path):
    from phasedx_external.scorer29 import load_person_map
    from phasedx_external.protocol28 import AdmissionError
    p = tmp_path / "people.csv"
    p.write_text("exam_key,person_key\nE,A\nE,B\n")
    with pytest.raises(AdmissionError, match="duplicate"):
        load_person_map(p)


def test_scoring_has_no_runtime_bootstrap_overrides(tmp_path):
    from phasedx_external.scorer29 import evaluate, ContractError
    p = tmp_path / "design.json"
    p.write_text('{"mode":"clinical_external"}')
    with pytest.raises(ContractError, match="overrides"):
        evaluate(str(p), "", "", {}, "", "", "", "", n_draws=2)
