# The demo server as a container: the code, its dependencies (CPU-only PyTorch), the demo data built by
# notebook 13, the NASA POWER daily data and the text encoder, all baked in. See the README, "Hosting the demo".
#
# Before building, stage the data (not part of the repository) under deploy/:
#   deploy/data/demo/      <- $ASKMAP_DATA_ROOT/demo
#   deploy/data/power/     <- $ASKMAP_DATA_ROOT/power (the daily .nc files and manifest.json)
#   deploy/hf_cache/       <- the text encoder from $HF_HOME (models--openai--clip-vit-large-patch14)
# scripts/deploy_azure.sh does this.

FROM python:3.12-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, so code changes do not rebuild them. torch comes from PyTorch's CPU index (the PyPI
# wheel carries CUDA and is several times larger), so the CUDA helper packages the lock file lists for Linux
# are left out; every other package comes from PyPI at the locked version.
COPY pyproject.toml uv.lock README.md ./
RUN pip install uv \
    && uv export --no-dev --no-hashes --no-emit-project --format requirements-txt -o requirements.txt \
    && grep -v -E '^(nvidia-|cuda-|triton)' requirements.txt > requirements-cpu.txt \
    && pip install --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple \
         -r requirements-cpu.txt

# The data and the text encoder (the largest layers; they change rarely)
COPY deploy/hf_cache /app/hf_cache
COPY deploy/data /app/data

# The saved language-model replies for the page examples, so the deployed demo uses exactly the plans that
# were checked (a question that has no saved reply is asked afresh and saved in the container)
COPY outputs/llm/demo /app/outputs/llm/demo

# The code runs from /app/src, so the package finds the repository's configs/ next to it (its project root is
# two folders above askmap/config.py)
COPY src ./src
COPY configs ./configs

ENV PYTHONPATH=/app/src \
    ASKMAP_DATA_ROOT=/app/data \
    HF_HOME=/app/hf_cache \
    HF_HUB_OFFLINE=1 \
    ASKMAP_HOST=0.0.0.0

# HF_TOKEN must be provided at run time (a platform secret); it is never baked into the image.
EXPOSE 8765
CMD ["python", "-m", "askmap.demo.serve"]
