"""
check_railway_services.py -- read-only health check of every public Railway address.

Prints UP (HTTP 200) / DOWN with the HTTP code for each project. Railway answers with
"HTTP 404 + header x-railway-fallback: true" when NO deployment is running behind an address
(the dashboard shows "Service is offline"), which is reported as OFFLINE (no running deployment).
It never changes anything.

Usage:  python scripts/check_railway_services.py
Exit code 0 = all up, 1 = at least one is not up.
"""
import sys
import urllib.request
import urllib.error

URLS = {
    "basics4ai-staging (production)": "https://basics4ai-staging-production.up.railway.app/",
    "basics4ai-staging (test)": "https://basics4ai-staging-test.up.railway.app/",
    "basics4ai-web": "https://basics4ai-web-production.up.railway.app/",
    "get2knourlearners": "https://get2knourlearners-production.up.railway.app/",
    "inferencing demo": "https://basics4ai-inferencing-demo-production.up.railway.app/",
    "maze demo": "https://basics4ai-maze-demo-production.up.railway.app/",
    "tactical planning demo": "https://basics4ai-tactical-planning-demo-production.up.railway.app/",
    "shopping demo": "https://basics4ai-shopping-demo-production.up.railway.app/",
    "dice probability demo": "https://basics4ai-dice-probability-demo-production.up.railway.app/",
    "sudoku demo": "https://basics4ai-sudoku-demo-production.up.railway.app/",
    "tic-tac-toe demo": "https://basics4ai-tictactoe-demo-production.up.railway.app/",
}


def probe(url):
    req = urllib.request.Request(url, headers={"User-Agent": "b4ai-health-check"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, r.headers.get("x-railway-fallback")
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("x-railway-fallback")
    except Exception as e:
        return None, str(e)[:60]


bad = 0
for name, url in URLS.items():
    code, extra = probe(url)
    if code == 200:
        state = "UP"
    elif code == 404 and extra == "true":
        state, bad = "OFFLINE (no running deployment)", bad + 1
    else:
        state, bad = f"DOWN ({code} {extra or ''})", bad + 1
    print(f"{name:34s} {state}")
print(f"\n{len(URLS) - bad} of {len(URLS)} up")
sys.exit(1 if bad else 0)
