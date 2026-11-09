"""
Population embeddings of one session: PCA and factor analysis (linear baselines), GPFA (elephant),
CEBRA-Time / CEBRA-Behavior / shuffled-label CEBRA (the nonlinear contrastive embeddings), and the
cross-session consistency of CEBRA embeddings.
"""
from __future__ import annotations

import logging
import time

import numpy as np

log = logging.getLogger("neuro-embedding")


# --- preprocessing -----------------------------------------------------------------------------

def smooth(X: np.ndarray, sigma_bins: float) -> np.ndarray:
    if sigma_bins <= 0:
        return X.astype(np.float32, copy=True)
    from scipy.ndimage import gaussian_filter1d

    return gaussian_filter1d(X.astype(np.float32), sigma_bins, axis=0, mode="nearest")


def zscore(X: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    ref = X[mask] if mask is not None else X
    mu = ref.mean(axis=0)
    sd = ref.std(axis=0)
    sd[sd <= 0] = 1.0
    return ((X - mu) / sd).astype(np.float32)


# --- PCA and factor analysis -------------------------------------------------------------------

def pca(Z: np.ndarray, mask: np.ndarray, n_keep: int, seed: int) -> dict:
    """PCA of smoothed z-scored activity on the epoch bins; scores for every bin."""
    from sklearn.decomposition import PCA

    n_comp = int(min(Z.shape[1], 50, mask.sum() - 1))
    p = PCA(n_components=n_comp, svd_solver="full", random_state=seed).fit(Z[mask])
    evr = p.explained_variance_ratio_
    cum = np.cumsum(evr)
    ev = p.explained_variance_
    pr = float(ev.sum() ** 2 / (ev ** 2).sum())  # participation ratio over the PCs kept
    scores = p.transform(Z)[:, : max(3, min(n_keep, n_comp))].astype(np.float32)

    def n_for(f):
        return int(np.searchsorted(cum, f) + 1) if cum[-1] >= f else None

    return {"scores": scores, "evr": evr.astype(np.float32),
            "metrics": {"var_explained_3pc": float(cum[min(2, cum.size - 1)]), "n_pc_50": n_for(0.5),
                        "n_pc_80": n_for(0.8), "n_pc_90": n_for(0.9), "participation_ratio": round(pr, 3),
                        "n_components_fitted": n_comp}}


FA_MAX_BINS = 20000


def factor_analysis(Z: np.ndarray, mask: np.ndarray, max_k: int, folds: int, seed: int) -> dict:
    """Factor analysis: blocked (contiguous) cross-validated log-likelihood per number of factors
    picks the dimensionality; the best model gives shared-variance fraction and factor scores."""
    from sklearn.decomposition import FactorAnalysis
    from sklearn.model_selection import KFold

    D = Z[mask]
    if D.shape[0] > FA_MAX_BINS:   # FA's cost grows with bins x neurons^2: every k-th bin is plenty
        D = D[:: int(np.ceil(D.shape[0] / FA_MAX_BINS))]
    cands = [k for k in (1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30) if k <= min(max_k, D.shape[1] - 1)]
    if not cands:
        cands = [1]
    kf = KFold(n_splits=max(2, folds), shuffle=False)
    # neurons (nearly) silent in one block (drift, units lost or found mid-session) make FA's noise
    # variance collapse and the held-out likelihood explode: keep neurons that vary in every block
    stds = np.array([D[te].std(axis=0) for _, te in kf.split(D)])
    keep = stds.min(axis=0) >= 0.05
    if keep.sum() >= 3:
        D = D[:, keep]
    cands = [k for k in cands if k <= D.shape[1] - 1] or [1]
    ll = []
    for k in cands:
        s = []
        for tr, te in kf.split(D):
            fa = FactorAnalysis(n_components=k, random_state=seed, max_iter=500).fit(D[tr])
            s.append(fa.score(D[te]))
        ll.append(float(np.mean(s)))
    best = cands[int(np.argmax(ll))]
    fa = FactorAnalysis(n_components=best, random_state=seed, max_iter=1000).fit(D)
    Zk = Z[:, keep] if keep.sum() >= 3 else Z
    comm = (fa.components_ ** 2).sum(axis=0)
    shared = comm / (comm + fa.noise_variance_)
    k3 = max(best, 3) if D.shape[1] > 3 else best
    fa3 = fa if k3 == best else FactorAnalysis(n_components=k3, random_state=seed, max_iter=1000).fit(D)
    return {"scores": fa3.transform(Zk).astype(np.float32), "candidates": cands, "cv_loglik": ll,
            "metrics": {"n_factors_cv": best, "shared_variance_fraction": float(np.mean(shared)),
                        "cv_loglik_best": float(max(ll)), "n_neurons_used": int(D.shape[1])}}


# --- GPFA --------------------------------------------------------------------------------------

def gpfa(spikes: list[np.ndarray], segments: list[tuple[float, float]], neuron_idx: np.ndarray, bin_s: float,
         x_dim: int, iters: int, seed: int) -> dict:
    """elephant GPFA on the given segments (trials, or chunks of continuous data) and neurons.
    Returns orthonormalised latent trajectories per segment."""
    import neo
    import quantities as pq
    from elephant.gpfa import GPFA

    np.random.seed(seed)
    trials = []
    for (a, b) in segments:
        sts = []
        for j in neuron_idx:
            st = spikes[j]
            s = st[(st >= a) & (st < b)]
            sts.append(neo.SpikeTrain(s * pq.s, t_start=a * pq.s, t_stop=b * pq.s))
        trials.append(sts)
    t0 = time.monotonic()
    g = GPFA(bin_size=bin_s * pq.s, x_dim=x_dim, em_max_iters=iters, freq_ll=5)
    g.fit(trials)
    lat = g.transform(trials, returned_data=["latent_variable_orth"])
    lls = g.fit_info.get("log_likelihoods", [])
    n_t = min(x.shape[1] for x in lat)
    traj = np.stack([x[:, :n_t] for x in lat]).astype(np.float32)   # segments x dims x time
    taus = np.asarray(g.params_estimated.get("gamma", []))
    tau_ms = (bin_s * 1000 / np.sqrt(taus)).tolist() if taus.size else []
    return {"trajectories": traj, "metrics": {
        "n_segments": len(segments), "n_neurons": int(len(neuron_idx)), "n_neurons_with_spikes": int(g.has_spikes_bool.sum()),
        "bin_s": bin_s, "x_dim": x_dim, "em_iters": len(g.fit_info.get("iteration_time", [])) or iters,
        "final_loglik": float(lls[-1]) if len(lls) else None, "timescales_ms": [round(x, 1) for x in tau_ms],
        "seconds": round(time.monotonic() - t0, 1)}}


def counts_to_spikes(t: np.ndarray, X: np.ndarray, bin_size: float) -> list[np.ndarray]:
    """Spike times at bin centres, repeated by count (when spikes.npz is missing)."""
    out = []
    for j in range(X.shape[1]):
        c = np.round(X[:, j]).astype(int)
        out.append(np.repeat(t, np.maximum(c, 0)))
    return out


# --- CEBRA -------------------------------------------------------------------------------------

def cebra_model(cfg: dict, conditional: str, iters: int | None = None):
    import cebra

    return cebra.CEBRA(model_architecture=cfg["arch"], batch_size=cfg["batch"], learning_rate=cfg["lr"],
                       temperature=cfg["temperature"], output_dimension=cfg["dims"],
                       max_iterations=iters or cfg["iters"], distance="cosine", conditional=conditional,
                       device=cfg["device"], time_offsets=10, verbose=False)


def fit_cebra(X: np.ndarray, labels: np.ndarray | None, cfg: dict, what: str, heartbeat=None) -> dict:
    """Fit one CEBRA model (labels None = CEBRA-Time) and return the model and its loss curve."""
    import torch

    torch.manual_seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    if cfg["device"].startswith("cuda"):
        torch.cuda.manual_seed_all(cfg["seed"])
    m = cebra_model(cfg, "time" if labels is None else "time_delta")
    t0 = time.monotonic()
    if heartbeat:
        heartbeat(f"{what}: training {cfg['iters']} iterations on {X.shape[0]} bins x {X.shape[1]} neurons", force=True)
    cb = None
    if heartbeat:
        def cb(step, solver):   # progress at least every 30 s (conventions: a run never looks hung)
            loss = solver.state_dict().get("loss", []) if hasattr(solver, "state_dict") else []
            last = f", loss {np.mean([float(x) for x in loss[-50:]]):.3f}" if len(loss) else ""
            heartbeat(f"{what}: iteration {step}/{cfg['iters']}{last}")
    kw = {"callback": cb, "callback_frequency": 25} if cb else {}
    if labels is None:
        m.fit(X, **kw)
    else:
        m.fit(X, labels, **kw)
    loss = np.asarray(m.state_dict_["loss"], dtype=np.float32)
    secs = time.monotonic() - t0
    if heartbeat:
        heartbeat(f"{what}: done in {secs:.0f}s, final loss {loss[-100:].mean():.3f}", force=True)
    return {"model": m, "loss": loss, "seconds": round(secs, 1)}


def cebra_scores(m, X: np.ndarray, labels: np.ndarray | None, num_batches: int) -> dict:
    """InfoNCE loss and goodness of fit (bits; 0 = chance, log2(batch size) = perfect) on X."""
    from cebra.integrations.sklearn.metrics import infonce_loss, infonce_to_goodness_of_fit

    if labels is None:
        loss = infonce_loss(m, X, num_batches=num_batches)
    else:
        loss = infonce_loss(m, X, labels, num_batches=num_batches)
    return {"infonce": float(loss), "goodness_of_fit_bits": float(infonce_to_goodness_of_fit(loss, m))}


def knn_r2(train_emb, train_y, test_emb, test_y, k: int = 25) -> float:
    """kNN (cosine) decoding of a continuous label from an embedding: R^2 on the test bins."""
    from sklearn.metrics import r2_score
    from sklearn.neighbors import KNeighborsRegressor

    if len(train_y) < k + 1 or len(test_y) < 3:
        return float("nan")
    knn = KNeighborsRegressor(n_neighbors=k, metric="cosine", weights="distance").fit(train_emb, train_y)
    return float(r2_score(test_y, knn.predict(test_emb)))


def consistency(embeddings: list[np.ndarray], labels: list[np.ndarray], ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """CEBRA's consistency_score between datasets (R^2 of a linear fit between label-binned
    embeddings, both directions), as an ids x ids matrix (NaN on the diagonal)."""
    from cebra.integrations.sklearn.metrics import consistency_score

    scores, pairs, idx = consistency_score(embeddings=embeddings, labels=labels, dataset_ids=ids, between="datasets")
    names = [str(i) for i in idx]
    M = np.full((len(names), len(names)), np.nan)
    pos = {n: i for i, n in enumerate(names)}
    for s, (a, b) in zip(np.ravel(scores), np.asarray(pairs).reshape(-1, 2)):
        M[pos[str(a)], pos[str(b)]] = s
    return M, names
