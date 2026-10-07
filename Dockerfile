########################################
# STAGE 1 — BUILDER
########################################
FROM python:3.10-slim AS builder

WORKDIR /install

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libfreetype6-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies in standard location.
#
# torch is installed FIRST, as the CPU-only build, by exact version. Plain PyPI torch pulls in ~19 NVIDIA/CUDA
# packages (several GB). When PyPI had a newer torch than the PyTorch CPU index, the old unpinned install picked it
# and produced a 6.2 GB image that exceeded Railway's 4 GB limit (2026-10-07). A "+cpu" version string can only match
# the CPU index, so this cannot silently drift.
# Every other package is locked to the versions in constraints.txt (the known-good 2.4 GB build) so builds are
# reproducible. To refresh the lock deliberately: rebuild without -c, test, then regenerate with `pip freeze`.
COPY requirements.txt constraints.txt ./
RUN pip install --upgrade pip \
 && pip install --no-cache-dir "torch==2.14.1+cpu" --extra-index-url https://download.pytorch.org/whl/cpu \
 && grep -v -i -E "^(--extra-index-url|torch)" requirements.txt > requirements.docker.txt \
 && pip install --no-cache-dir -r requirements.docker.txt -c constraints.txt \
 && if pip list 2>/dev/null | grep -i -E "^(nvidia|cuda|triton)"; then echo "ERROR: CUDA packages present - aborting build"; exit 1; fi

# Optional: cleanup pycache
RUN find /usr/local/lib/python3.10/site-packages -type d -name "__pycache__" -exec rm -rf {} + && \
    find /usr/local/lib/python3.10/site-packages -type f -name "*.pyc" -delete

########################################
# STAGE 2 — RUNTIME
########################################
FROM python:3.10-slim

WORKDIR /app

# Copy installed packages from builder (standard site-packages)
COPY --from=builder /usr/local /usr/local

# Copy app code
COPY . .

# Streamlit config: dynamic port fallback
RUN mkdir -p /app/.streamlit && \
    echo "[server]\n\
port = ${PORT:-8501}\n\
address = \"0.0.0.0\"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\n\
[browser]\ngatherUsageStats = false\n" \
    > /app/.streamlit/config.toml

# Expose port dynamically
EXPOSE ${PORT:-8501}

# Run the app
CMD ["streamlit", "run", "streamlit_app/app.py"]