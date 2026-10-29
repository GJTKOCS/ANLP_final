PY ?= python3
.PHONY: env smoke test pipeline status report audit submission paper slides
env:
	$(PY) -m pip install -r requirements.txt && $(PY) -m pip install -e . --no-deps
test:
	$(PY) -m pytest -q --ignore=tests/test_smoke.py
smoke:
	$(PY) -m pytest -q
pipeline:
	covert pipeline run
status:
	covert status
report:
	covert eval report
audit:
	covert audit
submission:
	bash scripts/make_submission.sh
paper:
	cd paper && pdflatex -interaction=nonstopmode main >/dev/null && bibtex main >/dev/null; cd paper && pdflatex -interaction=nonstopmode main >/dev/null && pdflatex -interaction=nonstopmode main >/dev/null
	$(PY) scripts/check_pages.py paper/main.pdf
slides:
	cd slides && pdflatex -interaction=nonstopmode slides >/dev/null && pdflatex -interaction=nonstopmode slides >/dev/null
