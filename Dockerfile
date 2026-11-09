FROM python:3.12.16-slim-bookworm@sha256:623674a14ef93d052bb9b0de4dbf5f28eb578a35263838941cc8aa43213c1487

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /processor

# PyTorch build: cpu or cuda. app.yml's variants set it (gpu variant, the default: the CUDA 12.6 build for the
# platform's driver 550, which also runs on CPU with USE_GPU=auto; cpu variant: CPU-only torch from the PyTorch
# CPU index, no NVIDIA libraries). The default here is cpu for a builder that passes no build arguments.
ARG TORCH_FLAVOUR=cpu

# The complete pinned dependency set, no resolver (see requirements.txt).
COPY requirements.txt requirements-${TORCH_FLAVOUR}.txt /processor/
RUN pip install --no-deps --only-binary=:all: \
        -r /processor/requirements.txt -r /processor/requirements-${TORCH_FLAVOUR}.txt \
    && find /usr/local/lib/python3.12/site-packages -name '*.pyc' -delete \
    && find /usr/local/lib/python3.12/site-packages -path '*/torch/*' -prune -o -type d -name tests -prune -exec rm -rf {} + \
    && rm -rf /usr/local/lib/python3.12/site-packages/torch/include

COPY processor/ /processor/processor/
COPY VERSION /processor/VERSION

ENV PYTHONPATH="/processor" \
    PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    USER=app \
    LOGNAME=app \
    TORCH_FLAVOUR=${TORCH_FLAVOUR}

# Import the stack once at build time (fails the build on a broken install). Nothing is baked into /tmp or
# HOME: on Lambda only INPUT_DIR, OUTPUT_DIR, HOME and TMPDIR are writable and /tmp starts empty, so
# processor/entry.py points matplotlib, numba, torch and XDG caches below $TMPDIR at start-up. No model
# weights are needed (every model is fitted on the user's data).
RUN MPLCONFIGDIR=/dev/shm/mpl python -W ignore -c "import matplotlib.pyplot, sklearn, pyarrow, processor.main, cebra, elephant.gpfa, plotly.graph_objects"

# Runs as uid 1000 on ECS and as an arbitrary least-privileged user on Lambda; nothing depends on the uid
# (getpass finds USER in the environment).
RUN useradd --uid 1000 --no-create-home --home-dir /nonexistent --shell /usr/sbin/nologin app
USER 1000

ENTRYPOINT ["python3", "-m", "processor.entry"]
