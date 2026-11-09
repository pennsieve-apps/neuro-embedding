"""
neuro-embedding: low-dimensional population embeddings of neural activity.

Reads neuro-align's `aligned` output (one folder per session: aligned.npz, events.parquet,
spikes.npz, manifest.json) and writes, under $OUTPUT_DIR/neuro-embedding/:

  <session_id>/<session_id>_embeddings.npz   PCA, FA, CEBRA-Time, CEBRA-Behavior, shuffled CEBRA, GPFA
  <session_id>/<session_id>_embedding_3d.png the CEBRA figure: 3-D embeddings coloured by behaviour
  <session_id>/<session_id>_embedding_3d.html the same, interactive (plotly, offline)
  <session_id>/<session_id>_explained_variance.png, _cebra_loss.png, _gpfa.png
  <session_id>/summary.json                  per-session headline numbers
  embedding.csv                              session_id, method, metric, value (+ sheet columns)
  consistency.csv, consistency.png           CEBRA consistency between sessions (>= 2 sessions)
  summary.json, provenance.json

Sessions are processed one at a time; a failing session is logged and skipped.
"""
from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

from processor.common import (Heartbeat, app_version, peak_rss_mb, rel, setup_logging, tool_versions, write_csv,
                              write_json)

log = setup_logging("neuro-embedding")

APP = "neuro-embedding"
METHODS = ("pca", "fa", "gpfa", "cebra")
DEFAULTS = {
    "METHODS": "pca,fa,gpfa,cebra",
    "BEHAVIOR_VARS": "auto",
    "BIN_SIZE_S": "0",
    "SMOOTH_S": "0.05",
    "N_COMPONENTS": "10",
    "CV_FOLDS": "5",
    "GPFA_MAX_NEURONS": "150",
    "GPFA_BIN_S": "0.05",
    "GPFA_DIMS": "3",
    "GPFA_ITERS": "100",
    "GPFA_SEGMENT_S": "2.0",
    "GPFA_MAX_SEGMENTS": "150",
    "TRIAL_ALIGN": "auto",
    "TRIAL_WINDOW_S": "-0.2,1.0",
    "CEBRA_ITERS": "10000",
    "CEBRA_BATCH": "512",
    "CEBRA_DIMS": "3",
    "CEBRA_ARCH": "offset10-model",
    "CEBRA_LR": "0.0003",
    "CEBRA_TEMPERATURE": "1.0",
    "TRAIN_FRACTION": "0.8",
    "MAX_PLOT_POINTS": "20000",
    "SEED": "42",
    "THREADS": "0",
    "USE_GPU": "auto",
}
PARAMETERS = tuple(DEFAULTS)
ARCHS = ("offset10-model", "offset5-model", "offset1-model", "offset10-model-mse", "offset36-model")
CITATIONS = ["10.1038/s41586-023-06031-6", "10.1152/jn.90941.2008", "10.5281/zenodo.1186602"]


class ParameterError(ValueError):
    pass


def read_parameters(env=os.environ) -> dict:
    raw = {k: (env.get(k) if env.get(k) not in (None, "") else v) for k, v in DEFAULTS.items()}
    p: dict = {}
    meth = [m.strip().lower() for m in raw["METHODS"].split(",") if m.strip()]
    bad = [m for m in meth if m not in METHODS]
    if bad or not meth:
        raise ParameterError(f"METHODS must be a comma-separated subset of {', '.join(METHODS)}, got {raw['METHODS']!r}")
    p["METHODS"] = meth
    bv = raw["BEHAVIOR_VARS"].strip()
    p["BEHAVIOR_VARS"] = "auto" if bv.lower() == "auto" else [v.strip() for v in bv.split(",") if v.strip()]
    try:
        for k in ("N_COMPONENTS", "CV_FOLDS", "GPFA_MAX_NEURONS", "GPFA_DIMS", "GPFA_ITERS", "GPFA_MAX_SEGMENTS",
                  "CEBRA_ITERS", "CEBRA_BATCH", "CEBRA_DIMS", "MAX_PLOT_POINTS", "SEED", "THREADS"):
            p[k] = int(raw[k])
        for k in ("BIN_SIZE_S", "SMOOTH_S", "GPFA_BIN_S", "GPFA_SEGMENT_S", "CEBRA_LR", "CEBRA_TEMPERATURE",
                  "TRAIN_FRACTION"):
            p[k] = float(raw[k])
        w = [float(x) for x in raw["TRIAL_WINDOW_S"].split(",")]
    except ValueError as exc:
        raise ParameterError(f"bad numeric parameter: {exc}") from exc
    if len(w) != 2 or w[1] <= w[0]:
        raise ParameterError("TRIAL_WINDOW_S must be 'start,stop' seconds relative to TRIAL_ALIGN, stop > start")
    p["TRIAL_WINDOW_S"] = w
    p["TRIAL_ALIGN"] = raw["TRIAL_ALIGN"].strip()
    p["CEBRA_ARCH"] = raw["CEBRA_ARCH"].strip()
    if p["CEBRA_ARCH"] not in ARCHS:
        raise ParameterError(f"CEBRA_ARCH must be one of {', '.join(ARCHS)}")
    from processor.resources import DEVICES, normalise_use_gpu

    p["USE_GPU"] = normalise_use_gpu(raw["USE_GPU"])
    if p["USE_GPU"] not in DEVICES:
        raise ParameterError("USE_GPU must be auto, true or false")
    if p["CV_FOLDS"] < 2 or p["N_COMPONENTS"] < 1 or p["GPFA_DIMS"] < 1 or p["CEBRA_DIMS"] < 2:
        raise ParameterError("CV_FOLDS >= 2, N_COMPONENTS >= 1, GPFA_DIMS >= 1 and CEBRA_DIMS >= 2 required")
    if p["CEBRA_ITERS"] < 10 or p["CEBRA_BATCH"] < 16 or p["GPFA_ITERS"] < 1 or p["GPFA_MAX_NEURONS"] < 2:
        raise ParameterError("CEBRA_ITERS >= 10, CEBRA_BATCH >= 16, GPFA_ITERS >= 1, GPFA_MAX_NEURONS >= 2 required")
    if not 0.3 <= p["TRAIN_FRACTION"] < 1:
        raise ParameterError("TRAIN_FRACTION must be in [0.3, 1)")
    if min(p["BIN_SIZE_S"], p["SMOOTH_S"]) < 0 or p["GPFA_BIN_S"] <= 0 or p["GPFA_SEGMENT_S"] <= 0 or p["THREADS"] < 0:
        raise ParameterError("BIN_SIZE_S, SMOOTH_S, THREADS >= 0 and GPFA_BIN_S, GPFA_SEGMENT_S > 0 required")
    if p["MAX_PLOT_POINTS"] < 100:
        raise ParameterError("MAX_PLOT_POINTS must be >= 100")
    return p


def direction_of(y, bin_size: float):
    """+1 / -1 / 0 (still) from the smoothed derivative of a 1-D position."""
    import numpy as np
    from scipy.ndimage import gaussian_filter1d

    yy = np.where(np.isfinite(y), y, np.nanmedian(y))
    v = np.gradient(gaussian_filter1d(yy, max(1.0, 0.1 / bin_size)), bin_size)
    thr = 0.1 * np.nanpercentile(np.abs(v), 95)
    return np.where(v > thr, 1, np.where(v < -thr, -1, 0)).astype(np.int8)


def choose_labels(s, wanted, A):
    """The behaviour variables CEBRA-Behavior uses and the one figures are coloured by."""
    import numpy as np

    cols, why = A.continuous_variables(s, None if wanted == "auto" else wanted)
    names = [s.behavior_names[j] for j in cols]
    if wanted == "auto":
        pos = [j for j in cols if A.is_position(s.behavior_names[j], s.variables.get(s.behavior_names[j]))]
        if pos:
            cols = pos[:2]
            if len(cols) == 2:   # x and y of a linear track: keep the long axis (it then gets the direction)
                r = [float(np.nanpercentile(s.Y[:, j], 99) - np.nanpercentile(s.Y[:, j], 1)) for j in cols]
                if max(r) > 5 * max(min(r), 1e-12):
                    cols = [cols[int(np.argmax(r))]]
            names = [s.behavior_names[j] for j in cols]
        else:
            cols = cols[:3]
            names = names[:3]
    if not cols:
        return None
    Yc = s.Y[:, cols].astype(np.float64)
    direction = None
    if len(cols) == 1 and A.is_position(names[0], s.variables.get(names[0])):
        direction = direction_of(Yc[:, 0], s.bin_size)
    return {"cols": cols, "names": names, "Y": Yc, "direction": direction, "skipped": why}


def run_session(s, p, cfg, out_dir: Path, A, M, F, hb) -> dict:
    import numpy as np

    sid = s.session_id
    res: dict = {"session_id": sid, "n_bins": int(s.t.size), "n_neurons": int(s.X.shape[1]), "bin_s": s.bin_size,
                 "neural_source": s.neural_source, "duration_s": round(float(s.t[-1] - s.t[0] + s.bin_size), 2),
                 "epoch_fraction": round(float(s.epoch_mask.mean()), 3), **s.sheet}
    metrics: list[dict] = []
    figs: list[dict] = []
    arrays: dict = {"session_id": np.array(sid), "t": s.t, "epoch_mask": s.epoch_mask}
    timings: dict = {}

    def add(method, metric, value):
        metrics.append({"method": method, "metric": metric, "value": value})

    if s.X.shape[1] < 3:
        raise ValueError(f"only {s.X.shape[1]} neurons")
    epoch = s.epoch_mask.copy()
    if epoch.sum() < 200:
        raise ValueError(f"only {int(epoch.sum())} bins in the epoch")
    lab = choose_labels(s, p["BEHAVIOR_VARS"], A)
    if lab:
        valid = epoch & np.all(np.isfinite(lab["Y"]), axis=1)
        arrays["behavior"] = lab["Y"].astype(np.float32)
        arrays["behavior_names"] = np.array(lab["names"])
        res["behavior_variables"] = lab["names"]
        res["colour_variable"] = lab["names"][0]
        if lab["skipped"]:
            res["variables_skipped"] = lab["skipped"]
    else:
        valid = epoch
        res["behavior_variables"] = []
        log.warning("%s: no usable continuous behaviour variable; CEBRA-Behavior is skipped, figures coloured by time", sid)
    colour = lab["Y"][:, 0] if lab else s.t - s.t[0]
    colour_name = (lab["names"][0] + (f" ({s.unit(lab['names'][0])})" if s.unit(lab["names"][0]) else "")) if lab else "time (s)"
    direction = lab["direction"] if lab else None
    sid_dir = out_dir / sid
    sid_dir.mkdir(parents=True, exist_ok=True)

    # linear baselines on smoothed, z-scored activity
    Z = M.zscore(M.smooth(s.X if not s.is_counts else np.sqrt(s.X), p["SMOOTH_S"] / s.bin_size), epoch)
    panels = []
    if "pca" in p["METHODS"]:
        t0 = time.monotonic()
        pc = M.pca(Z, epoch, p["N_COMPONENTS"], p["SEED"])
        arrays["pca"] = pc["scores"]
        arrays["pca_explained_variance_ratio"] = pc["evr"]
        for k, v in pc["metrics"].items():
            add("pca", k, v)
        res["pca"] = pc["metrics"]
        panels.append(("PCA (top 3 PCs)", pc["scores"][:, :3], f"{pc['metrics']['var_explained_3pc'] * 100:.0f}% variance"))
        timings["pca"] = round(time.monotonic() - t0, 1)
    if "fa" in p["METHODS"]:
        t0 = time.monotonic()
        fa = M.factor_analysis(Z, epoch, p["N_COMPONENTS"], p["CV_FOLDS"], p["SEED"])
        arrays["fa"] = fa["scores"]
        for k, v in fa["metrics"].items():
            add("fa", k, v)
        res["fa"] = fa["metrics"]
        timings["fa"] = round(time.monotonic() - t0, 1)
        if "pca" in p["METHODS"]:
            figs.append({"path": F.explained_variance(sid, pc["evr"], fa["candidates"], fa["cv_loglik"],
                                                      fa["metrics"]["n_factors_cv"],
                                                      sid_dir / f"{sid}_explained_variance.png"),
                         "caption": f"{sid}: dimensionality. Left: variance of the smoothed, z-scored activity "
                                    f"explained by each principal component (bars) and cumulatively (line); "
                                    f"{pc['metrics']['n_pc_80'] or '>50'} PCs reach 80%. Right: factor analysis "
                                    f"held-out log-likelihood per number of factors (blocked cross-validation); "
                                    f"its peak ({fa['metrics']['n_factors_cv']} factors) estimates how many shared "
                                    f"dimensions the population uses. Few dimensions = strongly coordinated activity."})
    del Z

    # GPFA (spiking data)
    if "gpfa" in p["METHODS"]:
        t0 = time.monotonic()
        try:
            g = run_gpfa(s, p, A, M)
            if g:
                arrays["gpfa"] = g["trajectories"]
                arrays["gpfa_segments"] = np.asarray(g["segments"], np.float64)
                arrays["gpfa_neuron_ids"] = np.array([s.neuron_ids[j] for j in g["neurons"]])
                for k, v in g["metrics"].items():
                    if not isinstance(v, list):
                        add("gpfa", k, v)
                res["gpfa"] = g["metrics"]
                figs.append({"path": F.gpfa_trajectories(sid, g["trajectories"], g["groups"], g["group_name"],
                                                         p["GPFA_BIN_S"], g["offset"], sid_dir / f"{sid}_gpfa.png"),
                             "caption": f"{sid}: GPFA (Gaussian-process factor analysis) latent trajectories of "
                                        f"{g['metrics']['n_neurons']} neurons on {g['metrics']['n_segments']} "
                                        f"{'trials' if g['trials'] else 'segments of ' + str(p['GPFA_SEGMENT_S']) + ' s'}"
                                        + (f", averaged by {g['group_name']}" if g["groups"] is not None else "")
                                        + ". GPFA smooths single-trial activity with a learned timescale "
                                        f"({', '.join(str(x) for x in g['metrics']['timescales_ms'])} ms); "
                                        "trajectories that separate by condition show trial-by-trial population "
                                        "dynamics that differ between conditions."})
        except Exception as exc:
            log.warning("%s: GPFA failed (%s: %s); continuing", sid, type(exc).__name__, exc)
            res["gpfa"] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        timings["gpfa"] = round(time.monotonic() - t0, 1)

    # CEBRA
    if "cebra" in p["METHODS"]:
        t0 = time.monotonic()
        Xc = s.X.astype(np.float32) if s.is_counts else M.zscore(s.X, epoch)
        vidx = np.flatnonzero(valid)
        if vidx.size < 3 * p["CEBRA_BATCH"]:
            log.warning("%s: %d valid bins for batch size %d", sid, vidx.size, p["CEBRA_BATCH"])
        n_train = int(p["TRAIN_FRACTION"] * vidx.size)
        tr, te = vidx[:n_train], vidx[n_train:]
        train_mask = np.zeros(s.t.size, bool)
        train_mask[tr] = True
        rng = np.random.default_rng(p["SEED"])
        nb = 100
        time_fit = M.fit_cebra(Xc[valid], None, cfg, f"{sid} CEBRA-Time", hb)
        emb_time = time_fit["model"].transform(Xc).astype(np.float32)
        arrays["cebra_time"] = emb_time
        ts = M.cebra_scores(time_fit["model"], Xc[te], None, nb)
        add("cebra_time", "train_loss_final", float(time_fit["loss"][-100:].mean()))
        add("cebra_time", "goodness_of_fit_bits", ts["goodness_of_fit_bits"])
        curves = [("CEBRA-Time (no labels)", time_fit["loss"], ts["goodness_of_fit_bits"])]
        res["cebra_time"] = {"train_loss_final": float(time_fit["loss"][-100:].mean()), **ts, "seconds": time_fit["seconds"]}
        if lab:
            L = lab["Y"].copy()
            mu, sd = np.nanmean(L[valid], axis=0), np.nanstd(L[valid], axis=0)
            L = (L - mu) / np.where(sd > 0, sd, 1)
            if direction is not None:
                L = np.column_stack([L, (direction == 1).astype(float), (direction == -1).astype(float)])
            L = L.astype(np.float32)
            arrays["labels"] = L
            arrays["label_names"] = np.array(lab["names"] + (["moving_up", "moving_down"] if direction is not None else []))
            beh = M.fit_cebra(Xc[tr], L[tr], cfg, f"{sid} CEBRA-Behavior", hb)
            emb_beh = beh["model"].transform(Xc).astype(np.float32)
            Lsh = L[valid][rng.permutation(vidx.size)]
            shuf = M.fit_cebra(Xc[valid], Lsh, cfg, f"{sid} CEBRA-Behavior, shuffled labels", hb)
            emb_shuf = shuf["model"].transform(Xc).astype(np.float32)
            arrays.update(cebra_behavior=emb_beh, cebra_behavior_train_mask=train_mask, cebra_shuffled=emb_shuf)
            bs = M.cebra_scores(beh["model"], Xc[te], L[te], nb)
            ss = M.cebra_scores(shuf["model"], Xc[te], L[te], nb)
            shuf_train = M.cebra_scores(shuf["model"], Xc[valid], Lsh, nb)
            ytr, yte = lab["Y"][tr, 0], lab["Y"][te, 0]
            knn = {"cebra_behavior": M.knn_r2(emb_beh[tr], ytr, emb_beh[te], yte),
                   "cebra_shuffled": M.knn_r2(emb_shuf[tr], ytr, emb_shuf[te], yte),
                   "cebra_time": M.knn_r2(emb_time[tr], ytr, emb_time[te], yte)}
            if "pca" in arrays:
                knn["pca"] = M.knn_r2(arrays["pca"][tr, :3], ytr, arrays["pca"][te, :3], yte)
            for name, d, fit in (("cebra_behavior", bs, beh), ("cebra_shuffled", ss, shuf)):
                add(name, "train_loss_final", float(fit["loss"][-100:].mean()))
                add(name, "goodness_of_fit_bits", d["goodness_of_fit_bits"])
            add("cebra_shuffled", "goodness_of_fit_bits_shuffled_labels", shuf_train["goodness_of_fit_bits"])
            for name, v in knn.items():
                add(name, f"knn_r2_{lab['names'][0]}_test", v)
            res["cebra_behavior"] = {"labels": arrays["label_names"].tolist(), "train_fraction": p["TRAIN_FRACTION"],
                                     "train_loss_final": float(beh["loss"][-100:].mean()), **bs,
                                     "seconds": beh["seconds"]}
            res["cebra_shuffled"] = {"train_loss_final": float(shuf["loss"][-100:].mean()), **ss,
                                     "goodness_of_fit_bits_on_shuffled_labels": shuf_train["goodness_of_fit_bits"],
                                     "seconds": shuf["seconds"]}
            res["knn_r2_test"] = {"variable": lab["names"][0], **knn}
            curves = [("CEBRA-Behavior", beh["loss"], bs["goodness_of_fit_bits"]),
                      ("CEBRA-Behavior, shuffled labels", shuf["loss"], ss["goodness_of_fit_bits"])] + curves
            panels += [("CEBRA-Time", emb_time, "no labels"),
                       ("CEBRA-Behavior", emb_beh, f"labels: {', '.join(lab['names'])}"),
                       ("Shuffled labels", emb_shuf, "control")]
            html_panels = [("CEBRA-Behavior", emb_beh), ("CEBRA-Time", emb_time), ("Shuffled labels", emb_shuf)]
        else:
            panels.append(("CEBRA-Time", emb_time, "no labels"))
            html_panels = [("CEBRA-Time", emb_time)]
        timings["cebra"] = round(time.monotonic() - t0, 1)
        figs.append({"path": F.cebra_loss(sid, curves, sid_dir / f"{sid}_cebra_loss.png"),
                     "caption": f"{sid}: CEBRA training loss (InfoNCE). A model trained with the real behaviour labels "
                                "should reach a clearly lower loss (higher goodness of fit, GoF, in bits on held-out "
                                "bins) than the same model trained with shuffled labels; if the two curves overlap, "
                                "the neural activity carries little information about these variables."})
        try:
            html = F.embedding_html(sid, html_panels, colour, colour_name, sid_dir / f"{sid}_embedding_3d.html",
                                    p["MAX_PLOT_POINTS"], p["SEED"])
            res["interactive"] = html.name
        except Exception as exc:
            log.warning("%s: interactive figure failed: %s", sid, exc)
    if panels:
        valid_plot = np.flatnonzero(valid)
        pan = [(a, e[valid_plot], b) for a, e, b in panels]
        figs.insert(0, {"path": F.embedding_panels(sid, pan, colour[valid_plot], colour_name,
                                                   None if direction is None else direction[valid_plot],
                                                   sid_dir / f"{sid}_embedding_3d.png", p["MAX_PLOT_POINTS"], p["SEED"]),
                        "caption": embedding_caption(sid, lab, res, direction is not None)})
    np.savez_compressed(sid_dir / f"{sid}_embeddings.npz", **arrays)
    res["timings_s"] = timings
    res["figures"] = [{"path": f"{sid}/{Path(f['path']).name}", "caption": f["caption"]} for f in figs]
    res["metrics"] = metrics
    return res


def embedding_caption(sid, lab, res, has_dir) -> str:
    if not lab:
        return (f"{sid}: population activity embedded in 3-D (PCA and CEBRA-Time, which uses no behaviour labels), "
                "coloured by time in the session. Smooth trajectories show slowly evolving population states.")
    v = lab["names"][0]
    knn = res.get("knn_r2_test", {})
    nums = ""
    if knn:
        nums = (f" On held-out bins, {v} is decoded from the 3-D CEBRA-Behavior embedding with kNN R² = "
                f"{knn.get('cebra_behavior', float('nan')):.2f}, vs {knn.get('cebra_shuffled', float('nan')):.2f} "
                f"for the shuffled-label control.")
    d = (" Points are coloured by position, in blue while it increases and in orange while it decreases (the two "
         "running directions)." if has_dir else f" Points are coloured by {v}.")
    return (f"{sid}: each point is one time bin of population activity, placed in 3-D by PCA (linear), CEBRA-Time "
            f"(self-supervised, no labels), CEBRA-Behavior (trained with {', '.join(lab['names'])}) and CEBRA with "
            f"shuffled labels (the control: structure that appears here is not behavioural).{d} A smooth colour "
            f"gradient along a ring or line in CEBRA-Behavior, absent from the control, means the population encodes "
            f"the variable.{nums}")


def run_gpfa(s, p, A, M):
    import numpy as np

    if not s.is_counts:
        log.info("%s: GPFA skipped (needs spike counts; this session is %s)", s.session_id, s.neural_source)
        return None
    events = A.read_events(s)
    trials = A.trial_table(events)
    segs, groups, group_name, offset, is_trials = [], None, "", 0.0, False
    if trials is not None and len(trials) >= 10:
        col = A.align_column(trials, p["TRIAL_ALIGN"])
        al = trials[col].to_numpy(float)
        ok = np.flatnonzero(np.isfinite(al))
        w0, w1 = p["TRIAL_WINDOW_S"]
        segs = [(al[i] + w0, al[i] + w1) for i in ok]
        tv = A.trial_variables(trials)
        for name in ("choice", "stimulus_side", *tv.keys()):
            if name in tv:
                groups, group_name = tv[name][ok], name
                break
        offset, is_trials = w0, True
    else:
        ep = s.epoch_mask
        starts = np.flatnonzero(np.diff(np.r_[0, ep.astype(int)]) == 1)
        stops = np.flatnonzero(np.diff(np.r_[ep.astype(int), 0]) == -1)
        for a, b in zip(starts, stops):
            t_a, t_b = s.t[a] - s.bin_size / 2, s.t[b] + s.bin_size / 2
            k = int((t_b - t_a) // p["GPFA_SEGMENT_S"])
            segs += [(t_a + i * p["GPFA_SEGMENT_S"], t_a + (i + 1) * p["GPFA_SEGMENT_S"]) for i in range(k)]
    if len(segs) < 5:
        log.info("%s: GPFA skipped (only %d segments)", s.session_id, len(segs))
        return None
    if len(segs) > p["GPFA_MAX_SEGMENTS"]:
        pick = np.unique(np.linspace(0, len(segs) - 1, p["GPFA_MAX_SEGMENTS"]).round().astype(int))
        segs = [segs[i] for i in pick]
        if groups is not None:
            groups = groups[pick]
    spikes = A.read_spikes(s)
    if spikes is None:
        spikes = M.counts_to_spikes(s.t, s.X, s.bin_size)
    # neurons: the most active, at most GPFA_MAX_NEURONS, each with spikes in at least 10% of segments
    seg_counts = np.zeros((len(segs), len(spikes)))
    for j, st in enumerate(spikes):
        for i, (a, b) in enumerate(segs):
            seg_counts[i, j] = np.searchsorted(st, b) - np.searchsorted(st, a)
    active = (seg_counts > 0).mean(axis=0) >= 0.1
    order = np.argsort(-seg_counts.sum(axis=0))
    neurons = np.array([j for j in order if active[j]][: p["GPFA_MAX_NEURONS"]])
    if neurons.size < max(p["GPFA_DIMS"] + 1, 3):
        log.info("%s: GPFA skipped (only %d active neurons)", s.session_id, neurons.size)
        return None
    log.info("%s: GPFA on %d neurons x %d %s, %d EM iterations", s.session_id, neurons.size, len(segs),
             "trials" if is_trials else "segments", p["GPFA_ITERS"])
    g = M.gpfa(spikes, segs, neurons, p["GPFA_BIN_S"], p["GPFA_DIMS"], p["GPFA_ITERS"], p["SEED"])
    g.update(segments=segs, neurons=neurons, groups=groups, group_name=group_name, offset=offset, trials=is_trials)
    g["metrics"]["segments"] = "trials" if is_trials else f"{p['GPFA_SEGMENT_S']} s chunks of the epoch"
    return g


def main() -> int:
    started = datetime.now(timezone.utc)
    t_start = time.monotonic()
    input_dir, output_dir = os.environ.get("INPUT_DIR"), os.environ.get("OUTPUT_DIR")
    if not input_dir or not output_dir:
        log.error("INPUT_DIR and OUTPUT_DIR must be set")
        return 2
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    try:
        p = read_parameters()
    except ParameterError as exc:
        log.error("%s", exc)
        return 2
    if not input_dir.is_dir():
        log.error("INPUT_DIR %s is not a folder", input_dir)
        return 1
    for k, v in (("MPLCONFIGDIR", "/tmp/matplotlib"), ("NUMBA_CACHE_DIR", "/tmp/numba")):
        os.environ.setdefault(k, v)
    from processor.resources import choose_device, resolve_threads, set_thread_env

    threads, thread_src = resolve_threads(p["THREADS"])
    set_thread_env(threads)

    import numpy as np

    from processor import aligned as A
    from processor import figures as F
    from processor import methods as M

    found = A.discover(input_dir, "aligned")
    for name, reason in found.skipped.items():
        log.warning("%s: skipped, %s", name, reason)
    if not found.sessions:
        log.error("no neuro-align aligned.npz found under %s", input_dir)
        return 1
    device = {"accelerator": "cpu", "reason": "CEBRA not requested"}
    if "cebra" in p["METHODS"]:
        import torch

        torch.set_num_threads(threads)
        device = choose_device(p["USE_GPU"])
        log.info("%s", device["reason"])
    cfg = {"arch": p["CEBRA_ARCH"], "batch": p["CEBRA_BATCH"], "lr": p["CEBRA_LR"], "temperature": p["CEBRA_TEMPERATURE"],
           "dims": p["CEBRA_DIMS"], "iters": p["CEBRA_ITERS"], "seed": p["SEED"],
           "device": "cuda" if device["accelerator"] == "gpu" else "cpu"}
    log.info("%d sessions; methods %s; CEBRA %s, %d iterations, batch %d, on %s; %d threads (%s)", len(found.sessions),
             ",".join(p["METHODS"]), p["CEBRA_ARCH"], p["CEBRA_ITERS"], p["CEBRA_BATCH"], cfg["device"], threads, thread_src)

    out = output_dir / APP
    out.mkdir(parents=True, exist_ok=True)
    hb = Heartbeat(log, "progress", 30)
    results, statuses, rows, cons_inputs = {}, {}, [], {}
    figures_all: list[dict] = []
    warnings: list[str] = []
    for sid, path in found.sessions:
        t0 = time.monotonic()
        try:
            s = A.load(sid, path, found.sheet.get(sid))
            if p["BIN_SIZE_S"] > 0:
                s = A.rebin(s, p["BIN_SIZE_S"])
            r = run_session(s, p, cfg, out, A, M, F, hb)
            for m in r.pop("metrics"):
                rows.append({"session_id": sid, **{k: s.sheet.get(k) for k in A.SHEET_KEYS}, **m})
            write_json(out / sid / "summary.json", {"app": APP, "version": app_version(), "session_id": sid, **r})
            results[sid] = r
            figures_all += r["figures"]
            statuses[sid] = {"status": "ok", "seconds": round(time.monotonic() - t0, 1)}
            if "cebra" in p["METHODS"]:
                z = np.load(out / sid / f"{sid}_embeddings.npz")
                if "cebra_behavior" in z.files and "behavior" in z.files:
                    ok = s.epoch_mask & np.isfinite(z["behavior"][:, 0])
                    cons_inputs[sid] = {"var": str(z["behavior_names"][0]), "behavior": z["cebra_behavior"][ok],
                                        "time": z["cebra_time"][ok], "label": z["behavior"][ok, 0]}
            knn = r.get("knn_r2_test", {})
            log.info("%s: %d bins x %d neurons (%s, %.3g s bins); PCA 80%% at %s PCs; FA %s factors; GPFA %s; "
                     "CEBRA-Behavior GoF %s bits vs shuffled %s; kNN R2 %s %s vs shuffled %s; %.1fs", sid, s.t.size,
                     s.X.shape[1], s.neural_source, s.bin_size, r.get("pca", {}).get("n_pc_80"),
                     r.get("fa", {}).get("n_factors_cv"), "ok" if "n_segments" in r.get("gpfa", {}) else "-",
                     _f(r.get("cebra_behavior", {}).get("goodness_of_fit_bits")),
                     _f(r.get("cebra_shuffled", {}).get("goodness_of_fit_bits")), knn.get("variable", ""),
                     _f(knn.get("cebra_behavior")), _f(knn.get("cebra_shuffled")), time.monotonic() - t0)
            del s
        except Exception as exc:
            log.exception("%s: failed: %s", sid, exc)
            statuses[sid] = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                             "seconds": round(time.monotonic() - t0, 1)}
            warnings.append(f"{sid}: failed ({type(exc).__name__}: {exc})")
    if not results:
        log.error("every session failed")
        return 1

    # consistency across sessions that share the colouring variable
    cons = {}
    by_var: dict[str, list[str]] = {}
    for sid, c in cons_inputs.items():
        by_var.setdefault(c["var"], []).append(sid)
    cons_rows = []
    for var, sids in by_var.items():
        if len(sids) < 2:
            continue
        for kind in ("behavior", "time"):
            try:
                Mx, names = M.consistency([cons_inputs[x][kind] for x in sids], [cons_inputs[x]["label"] for x in sids], sids)
                off = Mx[np.isfinite(Mx)]
                cons[f"cebra_{kind}:{var}"] = {"sessions": names, "mean": float(off.mean()), "matrix": Mx.tolist()}
                for i, a in enumerate(names):
                    for j, b in enumerate(names):
                        if i != j:
                            cons_rows.append({"variable": var, "embedding": f"cebra_{kind}", "source": a, "target": b,
                                              "consistency_r2": Mx[i, j]})
                if kind == "behavior":
                    multi = sum(len(v) >= 2 for v in by_var.values()) > 1
                    fn = f"consistency_{A.sanitise(var)}.png" if multi else "consistency.png"
                    F.consistency_matrix(Mx, names, f"CEBRA-Behavior consistency across sessions ({var})", out / fn)
                    figures_all.append({"path": fn, "caption": (
                        f"Consistency of the CEBRA-Behavior embeddings between sessions (labels: {var}): R² of the "
                        f"best linear map from one session's embedding to another's, after binning both by {var} "
                        f"(CEBRA's consistency score). Mean {off.mean() * 100:.0f}% (CEBRA-Time: see summary.json). "
                        "High values mean different animals or days share the same latent geometry for this behaviour.")})
                log.info("consistency %s (%s): %d sessions, mean %.2f", kind, var, len(sids), off.mean())
            except Exception as exc:
                log.warning("consistency (%s, %s) failed: %s", kind, var, exc)
    if cons_rows:
        write_csv(out / "consistency.csv", cons_rows, ["variable", "embedding", "source", "target", "consistency_r2"])

    write_csv(out / "embedding.csv", rows, ["session_id", *A.SHEET_KEYS, "method", "metric", "value"])
    peak_vram = None
    if device.get("accelerator") == "gpu":
        import torch

        peak_vram = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    headline = {}
    for sid, r in results.items():
        headline[sid] = {k: r.get(k) for k in ("n_neurons", "n_bins", "bin_s", "neural_source", "behavior_variables",
                                                 *A.SHEET_KEYS) if r.get(k) is not None}
        headline[sid].update({
            "pca_n_pc_80": r.get("pca", {}).get("n_pc_80"), "pca_var_3pc": r.get("pca", {}).get("var_explained_3pc"),
            "fa_n_factors": r.get("fa", {}).get("n_factors_cv"),
            "cebra_behavior_gof_bits": r.get("cebra_behavior", {}).get("goodness_of_fit_bits"),
            "cebra_shuffled_gof_bits": r.get("cebra_shuffled", {}).get("goodness_of_fit_bits"),
            "cebra_time_gof_bits": r.get("cebra_time", {}).get("goodness_of_fit_bits"),
            "knn_r2_test": r.get("knn_r2_test"), "interactive": f"{sid}/{r['interactive']}" if r.get("interactive") else None})
    write_json(out / "summary.json", {
        "app": APP, "version": app_version(), "variant": os.environ.get("TORCH_FLAVOUR", "cpu") == "cuda" and "gpu" or "cpu",
        "sessions": headline, "consistency": cons, "figures": figures_all,
        "tables": ["embedding.csv"] + (["consistency.csv"] if cons_rows else []),
        "warnings": warnings, "n_ok": len(results), "n_failed": len(statuses) - len(results)})
    write_json(out / "provenance.json", {
        "app": APP, "version": app_version(), "variant": "gpu" if os.environ.get("TORCH_FLAVOUR") == "cuda" else "cpu",
        "tools": tool_versions(["cebra", "torch", "elephant", "neo", "quantities", "scikit-learn", "numpy", "scipy",
                                "pandas", "pyarrow", "matplotlib", "plotly"]),
        "parameters": p, "inputs": [rel(pth, input_dir) for _, pth in found.sessions],
        "started": started.isoformat().replace("+00:00", "Z"),
        "finished": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "wall_seconds": round(time.monotonic() - t_start, 1), "peak_rss_mb": peak_rss_mb(),
        "gpu": {"name": device.get("gpu"), "peak_vram_mb": peak_vram} if device.get("accelerator") == "gpu" else None,
        "device": device, "threads": threads, "sessions": statuses, "citations": CITATIONS})
    log.info("done: %d ok, %d failed, %.1fs, peak RSS %s MB", len(results), len(statuses) - len(results),
             time.monotonic() - t_start, peak_rss_mb())
    return 0


def _f(v):
    return "-" if v is None else f"{v:.2f}"
