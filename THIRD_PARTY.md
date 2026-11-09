# Third-party software

This app is Apache-2.0 (see `LICENSE`). Its image contains the following third-party components,
installed from PyPI at the pinned versions in `requirements.txt` and `requirements-{cpu,cuda}.txt`;
their licences apply to the image. Nothing here is GPL. Two images are built (`TORCH_FLAVOUR`):
the **cpu** variant has the CPU-only PyTorch wheel (`2.14.1+cpu`) and no NVIDIA software; in the
**gpu** variant the NVIDIA CUDA libraries that come with the CUDA build of PyTorch are proprietary
but redistributable under the NVIDIA licence terms shipped in each wheel. No model weights are
bundled: every model is trained on the user's data at run time.

| Component | What it does here | Authors / copyright | Licence | Source |
|---|---|---|---|---|
| CEBRA 0.6.1 | CEBRA-Time, CEBRA-Behavior and the shuffled control (`cebra.CEBRA`), InfoNCE / goodness of fit, `consistency_score`. Called as a library, not modified. | Steffen Schneider, Jin Hwa Lee, Mackenzie W. Mathis and contributors (Mathis Laboratory of Adaptive Intelligence, EPFL); copyright Mackenzie W. Mathis & Steffen Schneider | **Apache-2.0** for versions ≥ 0.4.0 (0.1.0–0.3.1 were non-commercial). **Patent notice from CEBRA's LICENSE.md:** "EPFL has filed a patent titled 'Dimensionality reduction of time-series data, and systems and devices that use the resultant embeddings' (https://patents.google.com/patent/WO2023143843A1) so if this does not work for your non-academic use case, please contact the TTO Office at EPFL." Apache-2.0 covers copyright; non-academic users should check the patent with EPFL. | https://github.com/AdaptiveMotorControlLab/CEBRA |
| literate-dataclasses 0.0.6 | Dependency of CEBRA. | Steffen Schneider | Apache-2.0 | https://pypi.org/project/literate-dataclasses/ |
| Elephant 1.2.1 | GPFA (`elephant.gpfa.GPFA`). | Elephant authors (NeuralEnsemble; Jülich Research Centre and partners, EBRAINS / Human Brain Project) | BSD-3-Clause | https://github.com/NeuralEnsemble/elephant |
| Neo 0.14.6, quantities 0.16.4 | Spike-train objects and units that Elephant's GPFA takes. | Neo authors (NeuralEnsemble); python-quantities developers | BSD-3-Clause | https://github.com/NeuralEnsemble/python-neo, https://github.com/python-quantities/python-quantities |
| PyTorch 2.14.1 (cpu: `+cpu`; gpu: `+cu126` with triton 3.8.0) | CEBRA's neural networks, on CPU or GPU. | PyTorch contributors (Meta Platforms, Inc. and others) | BSD-3-Clause style (torch's licence bundle); triton MIT | https://pytorch.org |
| nvidia-*-cu12, cuda-toolkit 12.6.3, cuda-bindings, cuda-pathfinder (versions in `requirements-cuda.txt`) | gpu variant only: CUDA runtime libraries. | NVIDIA Corporation | NVIDIA proprietary, redistributable (licence in each wheel); cuda-bindings/pathfinder Apache-2.0 | https://developer.nvidia.com/cuda-toolkit |
| scikit-learn 1.9.1 | PCA, factor analysis, kNN check. | The scikit-learn developers | BSD-3-Clause | https://scikit-learn.org |
| NumPy 2.5.4, SciPy 1.18.1 | Arrays, Gaussian smoothing. | NumPy and SciPy developers | BSD-3-Clause (NumPy also bundles 0BSD/MIT/Zlib/CC0 parts) | https://numpy.org, https://scipy.org |
| pandas 3.1.1, pyarrow 26.0.1 | events.parquet, tables. | pandas developers; Apache Software Foundation | BSD-3-Clause; Apache-2.0 | https://pandas.pydata.org, https://arrow.apache.org |
| Matplotlib 3.11.2, Pillow 13.0.0 | PNG figures. | Matplotlib Development Team; Jeffrey A. Clark and contributors | Matplotlib licence (PSF-based); MIT-CMU | https://matplotlib.org |
| plotly 7.1.0, narwhals 2.28.0 | Interactive 3-D HTML (plotly.js is embedded in each HTML file, so it works offline). | Plotly Technologies Inc.; narwhals developers | MIT | https://github.com/plotly/plotly.py |
| joblib, threadpoolctl, tqdm, requests, urllib3, certifi, charset-normalizer, idna, cloudpickle, filelock, fsspec, Jinja2, MarkupSafe, networkx, sympy, mpmath, setuptools, typing_extensions, packaging, contourpy, cycler, fonttools, kiwisolver, pyparsing, python-dateutil, tzdata | Dependencies of the above (versions in `requirements.txt`). | Their authors | BSD / MIT / Apache-2.0 / MPL-2.0 (certifi) / PSF-2.0 / MPL-2.0+MIT (tqdm) | PyPI |
| python:3.12.16-slim-bookworm | Base image (Debian 12 + CPython 3.12). | Python Software Foundation; Debian | PSF; Debian packages under their own free licences | https://hub.docker.com/_/python |

## Not bundled

- **Demo data** (not in the image or the repository): DANDI:000044 (Grosmark & Buzsáki 2016,
  CC-BY-4.0, doi:10.48324/dandi.000044/0.261106.0236) and DANDI:000409 (International Brain
  Laboratory Brain Wide Map, CC-BY-4.0, doi:10.48324/dandi.000409/0.260309.1324), read through
  neuro-align.
