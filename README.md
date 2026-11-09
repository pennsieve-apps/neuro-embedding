# neuro-embedding

**Built on CEBRA and Elephant's GPFA, not developed by Pennsieve.** [CEBRA](https://github.com/AdaptiveMotorControlLab/CEBRA)
was developed by Steffen Schneider, Jin Hwa Lee and Mackenzie W. Mathis at the Mathis Laboratory of
Adaptive Intelligence, EPFL (Brain Mind Institute, Geneva) ([Schneider, Lee & Mathis, *Nature* 2023](https://doi.org/10.1038/s41586-023-06031-6)).
Gaussian-process factor analysis (GPFA) was developed by Byron Yu, John Cunningham, Gopal Santhanam,
Stephen Ryu, Krishna Shenoy and Maneesh Sahani at Stanford University and the Gatsby Computational
Neuroscience Unit, UCL ([Yu et al., *J Neurophysiol* 2009](https://doi.org/10.1152/jn.90941.2008));
this app runs the implementation in [Elephant](https://github.com/NeuralEnsemble/elephant), the
Electrophysiology Analysis Toolkit of NeuralEnsemble (Michael Denker, Sonja Grün and colleagues,
Jülich Research Centre; [doi:10.5281/zenodo.1186602](https://doi.org/10.5281/zenodo.1186602)), on
spike trains in [Neo](https://github.com/NeuralEnsemble/python-neo) (Garcia et al. 2014). PCA,
factor analysis and kNN come from [scikit-learn](https://scikit-learn.org). Pennsieve packages these
methods as a workflow step and did not develop them; please cite the original papers (below). The demo
data are from Andres Grosmark and György Buzsáki (NYU; DANDI:000044) and the International Brain
Laboratory (DANDI:000409).

The step after `neuro-align` in the **Neural Population Dynamics & Decoding** workflow:

- `neuro-align → {neuro-encoding, neuro-embedding, neuro-decoding} → neuro-report`

**For a neuroscientist.** A population of neurons can carry far fewer independent signals than it
has neurons: when a rat runs along a track, the activity of a hundred place cells traces out a
ring or a line in a space of a few dimensions. This app finds that low-dimensional structure in
each session and shows it in 3-D, coloured by what the animal was doing. It answers: how many
dimensions does the activity use (PCA, factor analysis)? How does the population state evolve
around a task event, for each choice and stimulus side (trial-averaged PCA trajectories; optionally
single trials with GPFA)? Is there a smooth map from behaviour to population activity, and is it the
same in different animals (CEBRA)? The headline figure is CEBRA's: the activity of each time bin
as a point in 3-D, coloured by position; a clean ring or line that the shuffled-label control does
not show means the population encodes position.

**For a computational analyst.** Per session: PCA and factor analysis (FA dimensionality by
blocked cross-validated log-likelihood) of Gaussian-smoothed, square-rooted, z-scored activity;
for trial-based sessions, PCA of the trial-averaged activity per choice × stimulus side; optionally
(`METHODS=...,gpfa`; off by default because it is slow on hundreds of units) elephant's GPFA on at
most `GPFA_MAX_NEURONS` neurons, on trials when the session has a trials table, else on
fixed-length segments of the epoch; CEBRA (`offset10-model`, cosine distance,
InfoNCE, 3-D output) trained three ways: CEBRA-Time (`conditional="time"`, no labels),
CEBRA-Behavior (`conditional="time_delta"` with the behaviour variables, trained on the first
`TRAIN_FRACTION` of the epoch's bins), and CEBRA-Behavior with shuffled labels (the control). The
held-out bins give the InfoNCE loss, the goodness of fit in bits (CEBRA's
`infonce_to_goodness_of_fit`: 0 = chance, log2(batch size) = perfect) and a kNN (cosine, k = 25)
decoding check of the first variable. With two or more sessions that share that variable, CEBRA's
`consistency_score` (R² of the best linear map between label-binned embeddings) compares them.

## Ports

| Port | Direction | Content |
|---|---|---|
| `aligned` | input (subfolder, required) | neuro-align's output: `neuro-align/<session_id>/aligned.npz`, `events.parquet`, `spikes.npz`, `manifest.json`. Found by name anywhere below the port (or `INPUT_DIR` when merged). |
| `embedding` | output | `neuro-embedding/...` (below) |

## Method

1. **Read** each session's `aligned.npz` (bins × neurons `X`: spike counts or, for calcium data, a
   rate or ΔF/F; behaviour `Y`; `epoch_mask`). Only bins in neuro-align's epoch (e.g. running) are
   used to fit; embeddings are written for every bin. Angles (head direction) become sine and cosine.
2. **Behaviour labels** (`BEHAVIOR_VARS=auto`): the position variables when there are any (a 1-D
   linear-track position also gets the running direction as two binary labels, as in the CEBRA
   paper's rat hippocampus model), else up to three continuous variables (wheel velocity, speeds,
   pupil); labels are z-scored. The first is the colour of the figures.
3. **PCA / FA**: activity (square-rooted counts) smoothed with a Gaussian of SD `SMOOTH_S`, z-scored
   over the epoch. PCA: variance explained, PCs for 50/80/90 %, participation ratio. FA: 1 to
   `N_COMPONENTS` factors, contiguous `CV_FOLDS`-fold held-out log-likelihood; the best number of
   factors and the mean shared-variance fraction (communality / total variance).
4. **Trial-averaged trajectories** (sessions with a trials table): the smoothed, z-scored activity
   in `TRIAL_WINDOW_S` around `TRIAL_ALIGN` averaged per condition (choice × stimulus side, from
   the trials table: IBL `mouse_wheel_choice` × `gabor_stimulus_side`, or `choice` ×
   `stimulus_side`), conditions with at least 3 trials; PCA of the condition averages, top 3 PCs.
5. **GPFA** (optional, `METHODS` must list `gpfa`; spiking data only): up to `GPFA_MAX_SEGMENTS` trials (window `TRIAL_WINDOW_S` around
   `TRIAL_ALIGN`, by default the stimulus onset) or `GPFA_SEGMENT_S`-long chunks of the epoch, the
   `GPFA_MAX_NEURONS` most active neurons that fire in at least 10 % of the segments,
   `GPFA_DIMS` latents, `GPFA_ITERS` EM iterations, bins of `GPFA_BIN_S`; orthonormalised latent
   trajectories, the learned timescales; the figure averages them by choice (or stimulus side).
6. **CEBRA**: as above; `CEBRA_ITERS` iterations, batch `CEBRA_BATCH`, learning rate `CEBRA_LR`,
   temperature `CEBRA_TEMPERATURE`, seeded. Spike counts are the input as in the paper; calcium
   traces are z-scored.
7. **Consistency** between sessions sharing the colour variable: CEBRA-Behavior and CEBRA-Time.

## Parameters

| Name | Default | What it does |
|---|---|---|
| `METHODS` | `pca,fa,cebra` | Which methods run (`pca` includes the trial-averaged trajectories; add `gpfa` for GPFA). |
| `BEHAVIOR_VARS` | `auto` | CEBRA-Behavior labels and figure colour (see Method 2), or a comma-separated list of neuro-align variable names. |
| `BIN_SIZE_S` | `0` | Re-bin neuro-align's bins (s); 0 keeps them (25 ms for units). |
| `SMOOTH_S` | `0.05` | Gaussian SD (s) before PCA / FA. |
| `N_COMPONENTS` | `10` | PCs saved; largest number of factors tried. |
| `CV_FOLDS` | `5` | Blocked folds of the FA cross-validation. |
| `GPFA_MAX_NEURONS` | `150` | Neuron cap of GPFA (the most active). |
| `GPFA_BIN_S` | `0.05` | GPFA bin (s). |
| `GPFA_DIMS` | `3` | GPFA latents. |
| `GPFA_ITERS` | `100` | GPFA EM iterations. |
| `GPFA_SEGMENT_S` | `2.0` | Segment length (s) without trials. |
| `GPFA_MAX_SEGMENTS` | `150` | Trials / segments cap (evenly spaced). |
| `TRIAL_ALIGN` | `auto` | Trials-table column GPFA trials are aligned to (auto: stimulus onset, else trial start). |
| `TRIAL_WINDOW_S` | `-0.2,1.0` | GPFA trial window (s) around `TRIAL_ALIGN`. |
| `CEBRA_ITERS` | `10000` (gpu), `3000` (cpu) | CEBRA iterations per model (3 models per session). |
| `CEBRA_BATCH` | `512` (gpu), `256` (cpu) | CEBRA batch size. |
| `CEBRA_DIMS` | `3` | Embedding dimensions. |
| `CEBRA_ARCH` | `offset10-model` | CEBRA model (receptive field of 10 bins). |
| `CEBRA_LR` | `0.0003` | Learning rate. |
| `CEBRA_TEMPERATURE` | `1.0` | InfoNCE temperature. |
| `TRAIN_FRACTION` | `0.8` | CEBRA-Behavior trains on this first fraction of the epoch; the rest is held out. |
| `MAX_PLOT_POINTS` | `20000` | Bins drawn in the 3-D figures (random, seeded). |
| `SEED` | `42` | numpy and torch seed. |
| `THREADS` | `0` | CPU threads; 0 = the container's CPU quota. |
| `USE_GPU` | `auto` | `auto` / `true` (GPU when torch sees one, else CPU with a logged message) / `false`. |

Exit codes: `0` ok, `1` no `aligned.npz` found or every session failed, `2` bad parameters. A
session that fails is logged and skipped; a failing GPFA is logged and the rest goes on.

## Outputs

`neuro-embedding/` in the `embedding` port:

- `<session_id>/<session_id>_embeddings.npz`: `t` (bin centres, s, session clock), `epoch_mask`,
  `pca` (bins × `N_COMPONENTS`), `pca_explained_variance_ratio`, `fa` (bins × factors),
  `cebra_time`, `cebra_behavior`, `cebra_behavior_train_mask` (bins CEBRA-Behavior was trained on),
  `cebra_shuffled` (bins × 3 each), `labels`, `label_names`, `behavior`, `behavior_names`, `gpfa`
  (segments × latents × time), `gpfa_segments` (start, stop in s), `gpfa_neuron_ids`,
  `pca_trial_trajectories` (conditions × time × 3), `pca_trial_time`, `pca_trial_conditions`,
  `pca_trial_counts`. No pickles.
- `<session_id>/<session_id>_embedding_3d.png`: PCA, CEBRA-Time, CEBRA-Behavior and the shuffled
  control in 3-D, coloured by the first behaviour variable (blue / orange for the two running
  directions of a linear track); `<session_id>_embedding_3d.html`: the same, interactive (plotly,
  with plotly.js embedded: opens offline).
- `<session_id>/<session_id>_explained_variance.png`, `_cebra_loss.png`, `_pca_trials.png` (trial
  sessions), `_gpfa.png` (with GPFA), `summary.json`.
- `embedding.csv`: `session_id, subject_id, condition, group, method, metric, value` (long format:
  `pca` var_explained_3pc, n_pc_50/80/90, participation_ratio; `pca_trials` var_explained_3pc;
  `fa` n_factors_cv,
  shared_variance_fraction; `gpfa` n_neurons, n_segments, final_loglik; `cebra_*`
  train_loss_final, goodness_of_fit_bits, `knn_r2_<variable>_test`).
- `consistency.csv` (`variable, embedding, source, target, consistency_r2`) and `consistency.png`,
  with two or more sessions.
- `summary.json` (headline numbers per session, consistency, the figures with captions in the
  order the report shows them), `provenance.json`.

## Runtime

RUNTIME_SECTION

## Variants

| Variant | Image | Compute | Defaults | Pick it when |
|---|---|---|---|---|
| `gpu` (default, "GPU") | CUDA 12.6 PyTorch 2.14.1 (NVIDIA libraries as wheels; driver 550, T4 or newer) | `gpu`, 4 vCPU / 16 GiB | `CEBRA_ITERS=10000`, `CEBRA_BATCH=512` (the CEBRA paper's rat hippocampus setting) | A GPU is available. |
| `cpu` ("CPU") | CPU-only PyTorch 2.14.1+cpu, no NVIDIA libraries | `standard`, 4 vCPU / 8 GiB | `CEBRA_ITERS=3000`, `CEBRA_BATCH=256` | No GPU; shapes are similar, losses a little higher. |

The gpu image falls back to CPU when no GPU is visible (`USE_GPU=auto: cuda not available, using
cpu`), checked here. CEBRA's VRAM use is small (a 4-layer 1-D convolutional network with 32 hidden
units, batches of 512 × 10 bins): well under 1 GB on a T4. On a bigger card nothing needs changing;
for many sessions, raising `CEBRA_BATCH` to 1024 or 2048 improves the loss estimate.

## Limitations

- An embedding is a description, not a mechanism: a ring in CEBRA-Behavior shows that position
  is encoded, not how. CEBRA-Behavior is trained with the labels, so its structure is partly
  imposed by them; that is why the shuffled-label control and the held-out goodness of fit are
  shown next to it. Judge CEBRA-Time (no labels) and PCA for structure the data have on their own.
- Goodness of fit and kNN R² are computed on the last `1 - TRAIN_FRACTION` of the epoch (one split,
  not cross-validated); neuro-decoding does the cross-validated decoding.
- Consistency needs sessions with the same colour variable in the same units; it compares
  geometry, not neuron identities.
- GPFA assumes spike counts (it is skipped for calcium data) and is slow above ~200 neurons,
  hence off by default and capped; its latents are only comparable within a session.
- On IBL data CEBRA embeddings decode the wheel about as well as a linear decoder and the choice
  near chance ([Andre et al. 2026, BrainWideBench](https://doi.org/10.48550/arXiv.2609.22064)): CEBRA is this app's showpiece
  for hippocampal position and cross-session consistency, not a better decoder; neuro-decoding
  reports kNN on CEBRA only as a comparison.
- The CPU variant trains fewer iterations; check the loss curve before comparing numbers between
  the variants.

## Licences

Apache-2.0 (our code). CEBRA ≥ 0.4 is Apache-2.0, but **EPFL holds patent WO2023143843A1 on the
method**; CEBRA's licence says non-academic users should contact EPFL's Technology Transfer Office
(see `THIRD_PARTY.md`). Everything else bundled is BSD/MIT/Apache; the gpu image contains NVIDIA's
redistributable CUDA libraries.

## Citations

Please cite these works if you use this app. Verified on Crossref/DataCite and PubMed; full
author lists in `CITATION.cff`.

- Schneider S, Lee JH, Mathis MW. Learnable latent embeddings for joint behavioural and neural
  analysis. *Nature* 617:360–368 (2023). doi:[10.1038/s41586-023-06031-6](https://doi.org/10.1038/s41586-023-06031-6), PMID: 37138088. (CEBRA)
- Yu BM, Cunningham JP, Santhanam G, Ryu SI, Shenoy KV, Sahani M. Gaussian-Process Factor Analysis
  for Low-Dimensional Single-Trial Analysis of Neural Population Activity. *Journal of
  Neurophysiology* 102:614–635 (2009). doi:[10.1152/jn.90941.2008](https://doi.org/10.1152/jn.90941.2008), PMID: 19357332. (GPFA)
- Elephant, the Electrophysiology Analysis Toolkit (Denker M, Kern M, Köhler C, Jos HA; Zenodo,
  version 1.2.1). doi:[10.5281/zenodo.1186602](https://doi.org/10.5281/zenodo.1186602); RRID:SCR_003833.
- Denker M, Yegenoglu A, Grün S. Collaborative HPC-enabled workflows on the HBP Collaboratory using
  the Elephant framework. Neuroinformatics 2018, P19. doi:[10.12751/incf.ni2018.0019](https://doi.org/10.12751/incf.ni2018.0019). (Elephant's citation request)
- Garcia S et al. Neo: an object model for handling electrophysiology data in multiple formats.
  *Frontiers in Neuroinformatics* 8:10 (2014). doi:[10.3389/fninf.2014.00010](https://doi.org/10.3389/fninf.2014.00010), PMID: 24600386.
- Paszke A et al. PyTorch: An Imperative Style, High-Performance Deep Learning Library (2019).
  doi:[10.48550/arXiv.1912.01703](https://doi.org/10.48550/arXiv.1912.01703).
- Andre A, Mahato SP, Arora V, et al. BrainWideBench: Benchmarking large-scale pretraining and
  across-animal transfer in multi-region neural recordings. arXiv (2026). doi:[10.48550/arXiv.2609.22064](https://doi.org/10.48550/arXiv.2609.22064).
- scikit-learn (Pedregosa et al., *JMLR* 12:2825–2830, 2011; no DOI): https://scikit-learn.org.
  plotly.py: https://github.com/plotly/plotly.py.
- Demo data: Grosmark AD, Buzsáki G. Diversity in neural firing dynamics supports both rigid and
  learned hippocampal sequences. *Science* 351:1440–1443 (2016). doi:[10.1126/science.aad1935](https://doi.org/10.1126/science.aad1935), PMID: 27013730;
  dataset DANDI:000044, doi:[10.48324/dandi.000044/0.261106.0236](https://doi.org/10.48324/dandi.000044/0.261106.0236).
- Demo data: International Brain Laboratory et al. A brain-wide map of neural activity during
  complex behaviour. *Nature* 645:177–191 (2025). doi:[10.1038/s41586-025-09235-0](https://doi.org/10.1038/s41586-025-09235-0), PMID: 40903598;
  dataset DANDI:000409 (IBL - Brain Wide Map), doi:[10.48324/dandi.000409/0.260309.1324](https://doi.org/10.48324/dandi.000409/0.260309.1324).

## Acknowledgements

CEBRA is developed by the Mathis Laboratory of Adaptive Intelligence at EPFL; Elephant and Neo by
the NeuralEnsemble community with support from the EU Human Brain Project / EBRAINS; scikit-learn,
PyTorch and plotly by their open-source communities. The demo data come from the Buzsáki lab (NYU,
CRCNS hc-11 / DANDI:000044) and the International Brain Laboratory (DANDI:000409), served by the
DANDI Archive, in the Neurodata Without Borders (NWB) format.
