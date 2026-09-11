# Regenerate every number, table and figure from results/ on a clean checkout.
PY ?= .venv/bin/python
BUDGET ?= 20

.PHONY: figures aggregate primary tables exploratory smoke doctor

figures: primary
	$(PY) -m sdts.analysis.figures --all --budget $(BUDGET)
	$(PY) -m sdts.analysis.tables --budget $(BUDGET)

aggregate:
	$(PY) -m sdts.analysis.aggregate

primary: aggregate
	$(PY) -m sdts.analysis.primary --budget $(BUDGET)

tables: aggregate
	$(PY) -m sdts.analysis.tables --budget $(BUDGET)

exploratory: aggregate
	$(PY) -m sdts.analysis.exploratory.per_classifier

smoke:
	$(PY) -m sdts.runner.run_cell --smoke

doctor:
	$(PY) -m sdts.doctor
