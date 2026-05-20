NVCC = nvcc

CUDA_ARCH = -arch=sm_86
CUDA_OPT = -O3 --use_fast_math
CUDA_FLAGS = $(CUDA_ARCH) $(CUDA_OPT) -Xcompiler -Wall

CUDA_SOURCES = main_cuda.cu ising_cuda.cu
CUDA_HEADERS = ising_cuda.cuh

TARGET_CUDA = cuda

CXX = g++
CXX_FLAGS = -O3 -Wall -std=c++17
CPU_SOURCES = main.cpp ising.cpp algorithms.cpp observables.cpp
CPU_HEADERS = ising.hpp
TARGET_CPU = cpu

WOLFF_SOURCES = wolff.cpp ising.cpp algorithms.cpp observables.cpp
TARGET_WOLFF = wolff

.PHONY: all clean cuda cpu wolff

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

clean:
	rm -f $(TARGET_CUDA) $(TARGET_CPU) $(TARGET_WOLFF) test_gpu.csv test_cpu.csv test_wolff.csv
