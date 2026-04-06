########################################
# STAGE 1 — BUILDER
########################################
FROM python:3.10-slim AS builder

WORKDIR /install

RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libfreetype6-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --prefix=/install --no-cache-dir -r requirements.txt && \
    find /install -type d -name "__pycache__" -exec rm -rf {} + && \
    find /install -type f -name "*.pyc" -delete

########################################
# STAGE 2 — RUNTIME (SMALL IMAGE)
########################################
FROM python:3.10-slim

WORKDIR /app

COPY --from=builder /install /usr/local

RUN rm -rf /usr/local/lib/python*/site-packages/*.dist-info \
           /usr/local/lib/python*/site-packages/*.egg-info

COPY . .

RUN mkdir -p /app/.streamlit && \
    echo '[server]\nport = 8501\naddress = "0.0.0.0"\nheadless = true\nenableCORS = false\nenableXsrfProtection = false\n\n[browser]\ngatherUsageStats = false\n' \
    > /app/.streamlit/config.toml

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/app.py"]