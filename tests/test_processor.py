"""The synthetic fixture through the processor (no Docker), both input layouts."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from processor import aligned as A
from processor import main as M
from tests import make_fixture

FAST = {"CEBRA_ITERS": 250, "CEBRA_BATCH": 128, "GPFA_ITERS": 10, "GPFA_MAX_SEGMENTS": 60, "USE_GPU": "false",
        "THREADS": 2, "MAX_PLOT_POINTS": 3000}
RAT = "sub-rat1_ses-track1"
IBL = "sub-mouse1_ses-ibl1"


@pytest.fixture(scope="session")
def fixture_dir(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("fixture")
    make_fixture.write(out, "both", short=True)
    return out


def run(monkeypatch, input_dir: Path, output_dir: Path, **params) -> int:
    for k in M.PARAMETERS:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("INPUT_DIR", str(input_dir))
    monkeypatch.setenv("OUTPUT_DIR", str(output_dir))
    for k, v in {**FAST, **params}.items():
        monkeypatch.setenv(k, str(v))
    return M.main()


@pytest.fixture(scope="session")
def default_run(fixture_dir, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("out")
    mp = pytest.MonkeyPatch()
    try:
        assert run(mp, fixture_dir / "linked", out) == 0
    finally:
        mp.undo()
    return out / "neuro-embedding"


def test_outputs(default_run):
    for sid in (RAT, "sub-rat2_ses-track1", IBL):
        d = default_run / sid
        for f in (f"{sid}_embeddings.npz", f"{sid}_embedding_3d.png", f"{sid}_embedding_3d.html",
                  f"{sid}_explained_variance.png", f"{sid}_cebra_loss.png", f"{sid}_gpfa.png", "summary.json"):
            assert (d / f).is_file(), (sid, f)
        z = np.load(d / f"{sid}_embeddings.npz")
        n = z["t"].size
        for k in ("pca", "fa", "cebra_time", "cebra_behavior", "cebra_shuffled"):
            assert z[k].shape[0] == n and z[k].shape[1] >= 3, k
        assert z["cebra_behavior"].shape[1] == 3
        assert z["cebra_behavior_train_mask"].dtype == bool
        assert z["gpfa"].ndim == 3 and z["gpfa"].shape[1] == 3
    for f in ("embedding.csv", "consistency.csv", "consistency.png", "summary.json", "provenance.json"):
        assert (default_run / f).is_file(), f
    prov = json.loads((default_run / "provenance.json").read_text())
    assert prov["app"] == "neuro-embedding" and prov["version"] == "0.1.0" and prov["tools"]["cebra"]
    assert set(prov["sessions"]) == {RAT, "sub-rat2_ses-track1", IBL}
    summ = json.loads((default_run / "summary.json").read_text())
    for f in summ["figures"]:
        assert (default_run / f["path"]).is_file() and len(f["caption"]) > 40
    html = (default_run / RAT / f"{RAT}_embedding_3d.html").read_text()
    assert "plotly" in html and "cdn.plot.ly" not in html.split("<body>")[0][:2000]   # library embedded


def test_planted_structure(default_run):
    """Place cells: position is encoded, so CEBRA-Behavior beats the shuffled control."""
    for sid in (RAT, "sub-rat2_ses-track1"):
        s = json.loads((default_run / sid / "summary.json").read_text())
        assert s["behavior_variables"] == ["position"]
        assert s["cebra_behavior"]["goodness_of_fit_bits"] > s["cebra_shuffled"]["goodness_of_fit_bits"] + 0.2, sid
        assert s["knn_r2_test"]["cebra_behavior"] > 0.3, sid
        assert s["knn_r2_test"]["cebra_behavior"] > s["knn_r2_test"]["cebra_shuffled"] + 0.3, sid
        assert s["fa"]["n_factors_cv"] >= 2
    df = pd.read_csv(default_run / "embedding.csv")
    assert {"session_id", "method", "metric", "value", "subject_id"} <= set(df.columns)
    cons = pd.read_csv(default_run / "consistency.csv")
    beh = cons[cons["embedding"] == "cebra_behavior"]
    assert len(beh) == 2 and (beh["consistency_r2"] > 0.5).all()


def test_ibl_gpfa_on_trials(default_run):
    s = json.loads((default_run / IBL / "summary.json").read_text())
    assert s["gpfa"]["segments"] == "trials"
    assert s["behavior_variables"] == ["wheel_velocity"]


def test_flat_layout_and_methods(fixture_dir, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture_dir / "flat", tmp_path, METHODS="pca,fa") == 0
    d = tmp_path / "neuro-embedding" / RAT
    z = np.load(d / f"{RAT}_embeddings.npz")
    assert "pca" in z.files and "cebra_time" not in z.files
    assert (d / f"{RAT}_embedding_3d.png").is_file()


def test_bad_parameters_and_empty_input(fixture_dir, tmp_path, monkeypatch):
    assert run(monkeypatch, fixture_dir / "linked", tmp_path, METHODS="tsne") == 2
    assert run(monkeypatch, fixture_dir / "linked", tmp_path, TRAIN_FRACTION="1.5") == 2
    assert run(monkeypatch, fixture_dir / "linked", tmp_path, CEBRA_ARCH="nope") == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert run(monkeypatch, empty, tmp_path / "o") == 1


def test_reader_tolerates_variants(tmp_path):
    """neuron_meta as JSON, no manifest, Y given transposed."""
    d = tmp_path / "neuro-align" / "s1"
    d.mkdir(parents=True)
    t = np.arange(100) * 0.1
    X = np.random.default_rng(0).poisson(1, (100, 4)).astype(np.float32)
    meta = json.dumps([{"brain_area": r} for r in ("CA1", "CA1", "DG", "")])
    np.savez(d / "aligned.npz", t=t, X=X, Y=np.vstack([t, t]), behavior_names=np.array(["x", "y"]),
             neuron_meta=np.array(meta))
    f = A.discover(tmp_path)
    assert [s for s, _ in f.sessions] == ["s1"]
    s = A.load("s1", f.sessions[0][1])
    assert s.Y.shape == (100, 2) and s.regions == ["CA1", "CA1", "DG", "unknown"]
    assert s.neural_source == "units" and abs(s.bin_size - 0.1) < 1e-9
    r = A.rebin(s, 0.2)
    assert r.X.shape == (50, 4) and r.X.sum() == X.sum()
