"""Page-count check: body (before 'Challenges, Limitations and Future Work') must be <= 8 pages; appendix <= 2 pages."""
import re
import shutil
import subprocess
import sys

pdf = sys.argv[1] if len(sys.argv) > 1 else "paper/main.pdf"
if not shutil.which("pdftotext") or not shutil.which("pdfinfo"):
    print("pdftotext/pdfinfo not found (install poppler); check pages manually: body <= 8 pages")
    sys.exit(0)
n = int(re.search(r"Pages:\s+(\d+)", subprocess.check_output(["pdfinfo", pdf]).decode()).group(1))
pages = [subprocess.check_output(["pdftotext", "-f", str(i), "-l", str(i), pdf, "-"]).decode() for i in range(1, n + 1)]
find = lambda pat: next((i + 1 for i, t in enumerate(pages) if re.search(pat, t)), None)
lim, app = find(r"Challenges, Limitations and Future Work"), find(r"\bAppendix\b")
print(f"total pages {n}; limitations section starts on page {lim}; appendix on page {app}")
bad = False
if lim and lim > 9:
    print("FAIL: body text exceeds 8 pages (limitations section must start on page <= 9)")
    bad = True
if app and n - app + 1 > 3:
    print("WARNING: appendix may exceed 2 pages")
sys.exit(1 if bad else 0)
