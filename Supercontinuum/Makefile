PY ?= python

.PHONY: test test-all smoke checks extensions all clean

test:            ## fast tests (~30 s)
	$(PY) -m pytest -q -m "not slow" tests

test-all:        ## everything, including benchmark-scale tests (minutes)
	$(PY) -m pytest -q tests

smoke:           ## reduced run of the whole pipeline into results/small (~1 min)
	cd scripts && $(PY) run_checks.py --small --fresh && $(PY) run_extensions.py --small --fresh

checks:          ## checks table + figures 1-5
	cd scripts && $(PY) run_checks.py

extensions:      ## extension studies + figures 6-9 (needs `make checks` first)
	cd scripts && $(PY) run_extensions.py

all: checks extensions

clean:
	rm -f results/*.pkl
