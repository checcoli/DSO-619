PYTHON ?= python
BENCHMARKS ?= all

.PHONY: verify main-results service-level main-assets artifacts report report-only clean

verify:
	$(PYTHON) scripts/verify_report_claims.py

main-results:
	$(PYTHON) scripts/run_main_benchmarks.py --benchmarks $(BENCHMARKS)

service-level:
	$(PYTHON) scripts/service_level_experiment.py

main-assets:
	$(PYTHON) scripts/build_main_report_assets.py

artifacts:
	$(PYTHON) scripts/generate_report_artifacts.py --benchmarks $(BENCHMARKS)

report: artifacts report-only

report-only:
	mkdir -p reports/build
	cd reports && latexmk -pdf -interaction=nonstopmode -outdir=build decision_focused_kernel_report.tex
	cp reports/build/decision_focused_kernel_report.pdf reports/decision_focused_kernel_report.pdf

clean:
	rm -f reports/build/*
	find . -name '.DS_Store' -delete
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
