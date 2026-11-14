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

# Greek letters / maths symbols come from the non-embedded Symbol font and show as black squares in some viewers:
# they are spelled out (seen 2026-11-14 in the Inferential Statistics PDF: alpha, chi-square, Delta, d-bar)
t = pdf_text([{"heading": "Greek", "body": "Friedman \u03c7\u00b2 = 5.0, \u03b1 = 0.05, \u0394 = 2, Bias d\u0304: 1.9",
               "table": pd.DataFrame({"Friedman \u03c7\u00b2": ["\u03b1/15"]}), "caption": "\u03b1 \u2248 .05, p \u2264 \u03b1"}])
check("Greek letters are spelled out (alpha, chi, Delta)", all(w in t for w in ("alpha", "chi", "Delta")), t[:200])
check("no Greek letter, d-bar or combining mark is left in the PDF text",
      not any("\u0300" <= c <= "\u036f" or "\u0370" <= c <= "\u03ff" for c in t))
check("d-bar reads 'mean d'", "mean d" in t)
check("<= and ~ are spelled out", "<=" in t and "~" in t)

# the real ITA wording: 'non-verbal' with a non-breaking hyphen reads as 'non-verbal'
t = pdf_text([{"heading": "ITA", "body": "Students used non‑verbal cues and self‑regulation."}])
check("ITA wording keeps its hyphens as plain hyphens", "non-verbal" in t and "self-regulation" in t)

print("\n%d check(s) failed" % len(fail) if fail else "\nAll checks passed")
sys.exit(1 if fail else 0)
