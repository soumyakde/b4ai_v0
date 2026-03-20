# run_b4ai.ps1
# B4AI Streamlit launcher — run this instead of `streamlit run` directly.
# Ensures R is on PATH before Streamlit starts, preventing rpy2 thread issues.

# Add R to PATH so rpy2 can find it without using 'sh'
$env:PATH = "C:\Program Files\R\R-4.5.2\bin\x64;" + $env:PATH
$env:R_HOME = "C:\Program Files\R\R-4.5.2"
$env:RPY2_CFFI_MODE = "ABI"

# Change to project root (adjust if needed)
Set-Location "C:\Users\soumy\b4ai_v0"

# Activate conda environment if not already active
# (comment out if you activate manually before running)
# conda activate b4ai_v0

Write-Host "Starting B4AI Streamlit app..."
Write-Host "Press Ctrl+C once to stop (may need 2 presses due to rpy2)"
Write-Host ""

# Run streamlit — use python -m to ensure correct interpreter
python -m streamlit run streamlit_app/app.py
