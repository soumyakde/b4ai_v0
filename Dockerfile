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

# Copy and install Python dependencies in standard location
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

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