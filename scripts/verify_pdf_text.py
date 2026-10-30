"""Guard for PDF text (defect D-3): characters the PDF font (Helvetica) cannot draw print as black squares.
Builds a PDF with such characters in the heading, body, table cells and caption, reads the text back,
and fails if any square remains; also checks that characters Helvetica CAN draw are left alone.
Run: set PYTHONUTF8=1, then python scripts/verify_pdf_text.py"""
import io, os, sys, warnings
warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import pandas as pd
from pypdf import PdfReader
import streamlit_app.dashboards.teacher_dashboard as td

fail = []
def check(name, ok, detail=""):
    print(("PASS  " if ok else "FAIL  ") + name + (("  -> " + detail) if detail else ""))
    if not ok: fail.append(name)

def pdf_text(sections):
    r = PdfReader(io.BytesIO(td._build_pdf(sections, "Guard")))
    return "\n".join(p.extract_text() for p in r.pages)

PROBLEM = {"U+2010 hyphen": "‐", "U+2011 non-breaking hyphen": "‑", "U+2012 figure dash": "‒",
           "U+2015 horizontal bar": "―", "U+2212 minus": "−", "U+00A0 no-break space": " ",
           "U+202F narrow no-break space": " ", "U+2009 thin space": " ", "U+200B zero-width space": "​",
           "U+2264 less-or-equal": "≤", "U+2265 greater-or-equal": "≥", "U+03BA kappa": "κ",
           "U+00AD soft hyphen": "­", "U+FEFF byte-order mark": "﻿", "U+2043 hyphen bullet": "⁃",
           "U+03C1 rho": "ρ", "U+2192 arrow": "→", "U+2713 check mark": "✓"}
for name, ch in PROBLEM.items():
    t = pdf_text([{"heading": f"H{ch}x", "body": f"body a{ch}b", "table": pd.DataFrame({f"c{ch}": [f"cell{ch}v"]}),
                   "caption": f"cap c{ch}d"}])
    check(f"{name}: no black square anywhere in the PDF", "■" not in t and "�" not in t)

# characters Helvetica draws correctly must be unchanged
OK = "– — ‘ ’ “ ” • … é × ° ± –"
t = pdf_text([{"heading": "Keep", "body": OK}])
check("characters the font can draw are left as they are", all(c in t for c in "–—‘’“”…é×°±"), t.split("\n")[4] if len(t.split("\n")) > 4 else "")

# the real ITA wording: 'non-verbal' with a non-breaking hyphen reads as 'non-verbal'
t = pdf_text([{"heading": "ITA", "body": "Students used non‑verbal cues and self‑regulation."}])
check("ITA wording keeps its hyphens as plain hyphens", "non-verbal" in t and "self-regulation" in t)

print("\n%d check(s) failed" % len(fail) if fail else "\nAll checks passed")
sys.exit(1 if fail else 0)
