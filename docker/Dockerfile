# linux/amd64 is not a preference -- it is a hard requirement.
#
# textworld 1.7.0 publishes only x86_64 wheels (macosx_10_13_x86_64 and
# manylinux x86_64), and its dependencies `jericho` and `fast-downward-textworld`
# ship as source-only C/C++ builds. ALFWorld's own README tells arm64 macOS users
# to fall back to `CONDA_SUBDIR=osx-64`. Pinning the platform here means the same
# image runs under Rosetta on the M3 for development and natively on the rented
# x86_64 GPU box, so there is exactly one artifact to reason about.
#
# Environment stepping costs milliseconds, so emulation overhead is irrelevant --
# all wall-clock in this project is LLM inference, which happens outside the
# container.
#
# Python 3.11, not 3.9. ALFWorld only requires >=3.9, but on 3.9 pip resolves
# spacy (a transitive TextWorld dependency) to a source tarball whose build
# requirements pull `thinc>=8.3.12`, which itself requires Python >=3.10 -- an
# unsatisfiable resolution. 3.11 gets prebuilt wheels for spacy/thinc and avoids
# the dead end, while staying old enough to keep setup.py-based packages
# (jericho, fast-downward-textworld) building cleanly.
FROM --platform=linux/amd64 python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    ALFWORLD_DATA=/data/alfworld

# build-essential + cmake are needed because fast-downward-textworld (the PDDL
# planner behind textworld[pddl]) and jericho compile from source.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        cmake \
        git \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Text-only ALFWorld: no `[full]`/`[vis]` extra, so ai2thor, torch and
# torchvision are never pulled in. We only need AlfredTWEnv.
RUN pip install "alfworld==0.4.2"

WORKDIR /app
COPY pyproject.toml /app/pyproject.toml
COPY src /app/src
RUN pip install -e ".[analysis,dev,tokenizer]"

COPY configs /app/configs
COPY scripts /app/scripts
COPY tests /app/tests

# Game files live on a mounted volume, not in the image -- they are large and
# are downloaded once with `alfworld-download`.
VOLUME ["/data"]

CMD ["python", "-c", "import alfworld, textworld; print('alfworld', alfworld.__version__ if hasattr(alfworld,'__version__') else 'ok', '| textworld', textworld.__version__)"]
