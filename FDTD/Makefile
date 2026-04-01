SHELL := /bin/bash

CXX := g++
PYTHON := python3
PIP := $(PYTHON) -m pip

CXXFLAGS := -O3 -march=native -ffast-math -std=c++17 -Wall -Wextra

REQ_FILE := requirements.txt
DATA_DIR := data_outputs
RESULTS_DIR := results_outputs

CPP_SOURCES := $(wildcard *.cpp)
EXE_TARGETS := $(patsubst %.cpp,%,$(CPP_SOURCES))

.PHONY: all check-tools python-deps dirs cpp clean

all: check-tools python-deps dirs cpp

check-tools:
	@command -v $(CXX) >/dev/null 2>&1 || { echo "Error: $(CXX) is not installed."; exit 1; }
	@command -v $(PYTHON) >/dev/null 2>&1 || { echo "Error: $(PYTHON) is not installed."; exit 1; }
	@$(PYTHON) -m pip --version >/dev/null 2>&1 || { echo "Error: pip is not available for $(PYTHON)."; exit 1; }
	@echo "Tool check passed."

python-deps: $(REQ_FILE)
	@echo "Checking Python dependencies from $(REQ_FILE)..."
	@missing_pkgs=""; \
	for pkg in $$(grep -Ehv '^[[:space:]]*($$|#)' $(REQ_FILE) | sed -E 's/[[:space:]]+//g' | sed -E 's/[<>=!~].*$$//'); do \
		$(PYTHON) -m pip show "$$pkg" >/dev/null 2>&1 || missing_pkgs="$$missing_pkgs $$pkg"; \
	done; \
	if [ -n "$$missing_pkgs" ]; then \
		echo "Missing packages: $$missing_pkgs"; \
		echo "Installing from $(REQ_FILE)..."; \
		$(PIP) install -r $(REQ_FILE); \
	else \
		echo "Python dependencies already satisfied."; \
	fi

dirs:
	@mkdir -p $(DATA_DIR) $(RESULTS_DIR)
	@echo "Created/verified: $(DATA_DIR), $(RESULTS_DIR)"

cpp: $(EXE_TARGETS)
	@echo "Built C++ executables in current FDTD directory."


%: %.cpp | dirs
	$(CXX) $(CXXFLAGS) $< -o $@

clean:
	@rm -f $(EXE_TARGETS)
	@echo "Removed generated executables."
