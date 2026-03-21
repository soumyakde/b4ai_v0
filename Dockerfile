# Dockerfile — Basics4AI
# Python 3.10-slim matches your conda environment
FROM python:3.10-slim

# System dependencies needed by some packages
# (sentence-transformers needs gcc, reportlab needs freetype)
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    libfreetype6-dev \
    libffi-dev \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies first (cached layer if requirements unchanged)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Streamlit config — disable telemetry, set port
RUN mkdir -p /app/.streamlit
RUN echo '\
[server]\n\
port = 8501\n\
address = "0.0.0.0"\n\
headless = true\n\
enableCORS = false\n\
enableXsrfProtection = false\n\
\n\
[browser]\n\
gatherUsageStats = false\n\
' > /app/.streamlit/config.toml

EXPOSE 8501

# Use launch.py if R is available, otherwise direct streamlit
CMD ["python", "-m", "streamlit", "run", "streamlit_app/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
