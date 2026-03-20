"""Diagnose why .env is not being found."""
import os
from pathlib import Path

print("=== .env diagnostic ===\n")

# 1. Where Python thinks it is running from
print(f"cwd: {os.getcwd()}")

# 2. Check .env at common locations
candidates = [
    Path(os.getcwd()) / ".env",
    Path(os.getcwd()).parent / ".env",
    Path(__file__).resolve().parent / ".env",
]
for c in candidates:
    print(f"  {'EXISTS' if c.exists() else 'missing'}  {c}")

# 3. Simulate the walk from llm_clients.py location
llm_path = Path(os.getcwd()) / "core" / "analytics" / "llm" / "llm_clients.py"
print(f"\nWalking up from: {llm_path}")
for parent in llm_path.parents:
    candidate = parent / ".env"
    print(f"  checking: {candidate}  → {'FOUND' if candidate.exists() else 'not found'}")
    if candidate.exists():
        break

# 4. Try loading it directly
env_direct = Path(os.getcwd()) / ".env"
if env_direct.exists():
    print(f"\n.env contents preview (first 3 lines, keys masked):")
    with open(env_direct) as f:
        for i, line in enumerate(f):
            if i >= 3:
                break
            key = line.split("=")[0].strip()
            has_val = "=" in line and len(line.split("=", 1)[1].strip()) > 0
            print(f"  {key}={'<set>' if has_val else '<EMPTY>'}")
    
    # Try loading with dotenv
    try:
        from dotenv import load_dotenv
        load_dotenv(env_direct, override=True)
        anthropic_set = bool(os.environ.get("ANTHROPIC_API_KEY"))
        gemini_set    = bool(os.environ.get("GEMINI_API_KEY"))
        openai_set    = bool(os.environ.get("OPENAI_API_KEY"))
        print(f"\nAfter load_dotenv:")
        print(f"  ANTHROPIC_API_KEY: {'set' if anthropic_set else 'MISSING'}")
        print(f"  GEMINI_API_KEY:    {'set' if gemini_set else 'MISSING'}")
        print(f"  OPENAI_API_KEY:    {'set' if openai_set else 'MISSING'}")
    except ImportError:
        print("\npython-dotenv not installed — run: pip install python-dotenv")
else:
    print(f"\n.env NOT FOUND at {env_direct}")
    print("Create it with your API keys.")
