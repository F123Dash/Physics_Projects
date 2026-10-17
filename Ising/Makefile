NVCC = nvcc

CUDA_ARCH = -arch=sm_86
CUDA_OPT = -O3 --use_fast_math
CUDA_FLAGS = $(CUDA_ARCH) $(CUDA_OPT) -Xcompiler -Wall

CUDA_SOURCES = main_cuda.cu ising_cuda.cu ising.cpp observables.cpp
CUDA_HEADERS = ising_cuda.cuh ising.hpp run_io.hpp

TARGET_CUDA = ising2d_cuda

CXX = g++
CXX_FLAGS = -O3 -Wall -std=c++17
CPU_SOURCES = main.cpp ising.cpp algorithms.cpp observables.cpp
CPU_HEADERS = ising.hpp run_io.hpp
TARGET_CPU = ising2d_cpu

WOLFF_SOURCES = wolff.cpp ising.cpp algorithms.cpp observables.cpp
TARGET_WOLFF = ising2d_wolff

.PHONY: all clean cuda cpu wolff check fss

all: $(TARGET_CUDA) $(TARGET_CPU) $(TARGET_WOLFF)

cuda: $(TARGET_CUDA)

cpu: $(TARGET_CPU)

wolff: $(TARGET_WOLFF)

$(TARGET_CUDA): $(CUDA_SOURCES) $(CUDA_HEADERS)
	$(NVCC) $(CUDA_FLAGS) -o $@ $(CUDA_SOURCES)

$(TARGET_CPU): $(CPU_SOURCES) $(CPU_HEADERS)
	$(CXX) $(CXX_FLAGS) -o $@ $(CPU_SOURCES)

$(TARGET_WOLFF): $(WOLFF_SOURCES) $(CPU_HEADERS)
	$(CXX) $(CXX_FLAGS) -o $@ $(WOLFF_SOURCES)

both: $(TARGET_CUDA) $(TARGET_CPU)

test: $(TARGET_CUDA)
	./$(TARGET_CUDA) --sizes=32 --tmin=2.0 --tmax=2.5 --therm=1000 --meas=5000 --stride=10 --out=test_gpu.csv

test-cpu: $(TARGET_CPU)
	./$(TARGET_CPU) --sizes=32 --tmin=2.0 --tmax=2.5 --therm=1000 --meas=5000 --stride=10 --out=test_cpu.csv

test-wolff: $(TARGET_WOLFF)
	./$(TARGET_WOLFF) --sizes=32 --tmin=2.0 --tmax=2.5 --therm=1000 --meas=5000 --stride=10 --out=test_wolff.csv

benchmark: $(TARGET_CUDA) $(TARGET_CPU)
	@echo "CPU Benchmark"
	time ./$(TARGET_CPU) --sizes=64 --tmin=2.2 --tmax=2.3 --therm=5000 --meas=20000 --stride=20 --out=/dev/null
	@echo ""
	@echo "GPU Benchmark"
	time ./$(TARGET_CUDA) --sizes=64 --tmin=2.2 --tmax=2.3 --therm=5000 --meas=20000 --stride=20 --out=/dev/null

# Compare every built simulator with exact enumeration of the 4x4 lattice.
check: $(TARGET_CPU) $(TARGET_WOLFF)
	python3 check_exact.py --binaries $(wildcard ./$(TARGET_CPU) ./$(TARGET_WOLFF) ./$(TARGET_CUDA))

# Fixed-T finite-size scaling: Wolff chains at the exact Tc, raw series, block bootstrap.
FSS_DIR = data_outputs/fss
FSS_TC = 2.269185314213022
fss: $(TARGET_WOLFF)
	mkdir -p $(FSS_DIR)/series
	./$(TARGET_WOLFF) --sizes=16,32,64,128,256 --tmin=$(FSS_TC) --tmax=$(FSS_TC) --no-adaptive \
		--therm=10000 --meas=60000 --stride=1 --seed=20261002 \
		--out=$(FSS_DIR)/fss_wolff.csv --series-dir=$(FSS_DIR)/series
	python3 fss_analysis.py --series-dir $(FSS_DIR)/series --metrics $(FSS_DIR)/fss_metrics.csv

clean:
	rm -f $(TARGET_CUDA) $(TARGET_CPU) $(TARGET_WOLFF) test_gpu.csv test_cpu.csv test_wolff.csv
