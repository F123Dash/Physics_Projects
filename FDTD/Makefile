SHELL := /bin/bash

CXX := g++
PYTHON := python3

CXXFLAGS := -O3 -march=native -std=c++17 -fopenmp -Wall -Wextra

PY_MODULES := numpy pandas matplotlib
DATA_DIR := data_outputs
RESULTS_DIR := results_outputs

CPP_SOURCES := $(wildcard *.cpp)
EXE_TARGETS := $(patsubst %.cpp,%,$(CPP_SOURCES))

# Memory-bound kernels: 8 threads is the sweet spot on a 16-thread laptop CPU
export OMP_NUM_THREADS ?= 8

.PHONY: all check-tools python-deps dirs cpp validate plots clean

all: check-tools python-deps dirs cpp

check-tools:
	@command -v $(CXX) >/dev/null 2>&1 || { echo "Error: $(CXX) is not installed."; exit 1; }
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "Error: $(PYTHON) is not installed."; exit 1; }
	@echo "Tool check passed."

# Warn (don't fail) so the C++ build still works without the plotting stack
python-deps:
	@missing=""; \
	for m in $(PY_MODULES); do $(PYTHON) -c "import $$m" 2>/dev/null || missing="$$missing $$m"; done; \
	if [ -n "$$missing" ]; then \
		echo "Warning: missing Python modules:$$missing  (pip install$$missing)"; \
	else \
		echo "Python dependencies satisfied."; \
	fi

dirs:
	@mkdir -p $(DATA_DIR) $(RESULTS_DIR)

cpp: $(EXE_TARGETS)
	@echo "Built C++ executables in current FDTD directory."

3D cavity cpml_test: fdtd3d.hpp

%: %.cpp | dirs
	$(CXX) $(CXXFLAGS) $< -o $@

# Full 3-D validation suite (≈10 min on 8 threads), then the figures
validate: cavity cpml_test 3D | dirs
	./cavity
	./cpml_test
	./3D leak
	./3D mie 10
	./3D mie 30
	./3D mie 20
	./3D grin
	$(PYTHON) 3D.py

plots:
	$(PYTHON) 1D.py
	$(PYTHON) 2D.py
	$(PYTHON) 3D.py

clean:
	@rm -f $(EXE_TARGETS)
	@echo "Removed generated executables."
