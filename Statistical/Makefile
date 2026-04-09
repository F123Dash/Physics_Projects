# Makefile for 2D Ising Model CUDA Implementation
# Target: NVIDIA GeForce RTX 3050 (sm_86, Ampere)

NVCC = nvcc

# CUDA flags for RTX 3050 (sm_86)
CUDA_ARCH = -arch=sm_86
CUDA_OPT = -O3 --use_fast_math
CUDA_FLAGS = $(CUDA_ARCH) $(CUDA_OPT) -Xcompiler -Wall

# Source files
CUDA_SOURCES = main_cuda.cu ising_cuda.cu
CUDA_HEADERS = ising_cuda.cuh

# Targets
TARGET_CUDA = ising2d_cuda

# CPU target (for comparison)
CXX = g++
CXX_FLAGS = -O3 -Wall -std=c++17
CPU_SOURCES = main.cpp ising.cpp metropolis.cpp observables.cpp
CPU_HEADERS = ising.hpp
TARGET_CPU = ising2d_cpu

.PHONY: all clean cuda cpu

all: $(TARGET_CUDA)

cuda: $(TARGET_CUDA)

cpu: $(TARGET_CPU)

$(TARGET_CUDA): $(CUDA_SOURCES) $(CUDA_HEADERS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $(CUDA_SOURCES)

$(TARGET_CPU): $(CPU_SOURCES) $(CPU_HEADERS)
	$(CXX) $(CXX_FLAGS) -o $@ $(CPU_SOURCES)

# Build both for comparison
both: $(TARGET_CUDA) $(TARGET_CPU)

# Run quick test
test: $(TARGET_CUDA)
	./$(TARGET_CUDA) --sizes=32 --tmin=2.0 --tmax=2.5 --therm=1000 --meas=5000 --stride=10 --out=test_gpu.csv

# Run quick CPU test for comparison
test-cpu: $(TARGET_CPU)
	./$(TARGET_CPU) --sizes=32 --tmin=2.0 --tmax=2.5 --therm=1000 --meas=5000 --stride=10 --out=test_cpu.csv

# Performance benchmark
benchmark: $(TARGET_CUDA) $(TARGET_CPU)
	@echo "=== CPU Benchmark ==="
	time ./$(TARGET_CPU) --sizes=64 --tmin=2.2 --tmax=2.3 --therm=5000 --meas=20000 --stride=20 --out=/dev/null
	@echo ""
	@echo "=== GPU Benchmark ==="
	time ./$(TARGET_CUDA) --sizes=64 --tmin=2.2 --tmax=2.3 --therm=5000 --meas=20000 --stride=20 --out=/dev/null

clean:
	rm -f $(TARGET_CUDA) $(TARGET_CPU) test_gpu.csv test_cpu.csv
