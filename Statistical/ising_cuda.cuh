#pragma once

#include <cstdint>
#include <curand_kernel.h>

// Configuration structure (mirrors CPU version)
struct SimulationConfigCuda {
    int* sizes;
    int num_sizes;
    double t_min;
    double t_max;
    double t_step;
    int thermal_sweeps;
    int measurement_sweeps;
    int sample_stride;
    bool adaptive_grid;
    uint64_t seed;
    const char* output_csv;
};

// Sample accumulator (mirrors CPU version)
struct SampleAccumulatorCuda {
    double sum_m;
    double sum_abs_m;
    double sum_e;
    double sum_m2;
    double sum_e2;
    double sum_m4;
    int count;

    __host__ void reset();
    __host__ void add(double m_per_spin, double e_per_spin);
};

// Averaged observables
struct AveragedObservablesCuda {
    double m;
    double abs_m;
    double e;
    double m2;
    double e2;
    double m4;
};

AveragedObservablesCuda finalize_cuda(const SampleAccumulatorCuda& acc);

// CUDA kernels
__global__ void init_rng_kernel(curandState* states, uint64_t seed, int N);

__global__ void init_spins_random_kernel(int8_t* spins, curandState* rng, int N);

__global__ void init_spins_ordered_kernel(int8_t* spins, int8_t value, int N);

// Checkerboard Metropolis kernel
// color: 0 = black sites (x+y even), 1 = white sites (x+y odd)
__global__ void metropolis_checkerboard_kernel(
    int8_t* spins,
    curandState* rng,
    int L,
    int color,
    float exp_dE4,
    float exp_dE8
);

// Observables computation with reduction
__global__ void compute_magnetization_kernel(
    const int8_t* spins,
    int* partial_sums,
    int N
);

__global__ void compute_energy_kernel(
    const int8_t* spins,
    int* partial_sums,
    int L
);

// Final reduction kernel
__global__ void reduce_sum_kernel(int* data, int* result, int N);

// CUDA Ising model class
class Ising2DCUDA {
public:
    Ising2DCUDA(int L, uint64_t seed);
    ~Ising2DCUDA();

    void initialize_random();
    void initialize_ordered(int8_t spin_value = 1);
    void set_temperature(double T);
    void sweep_metropolis();

    double magnetization_per_spin();
    double energy_per_spin();

private:
    int L_;
    int N_;

    // Device memory
    int8_t* d_spins_;
    curandState* d_rng_states_;

    // Precomputed Boltzmann factors
    float exp_dE4_;
    float exp_dE8_;

    // For reductions
    int* d_partial_mag_;
    int* d_partial_energy_;
    int* d_result_;
    int num_blocks_;

    // Cached observables
    int cached_magnetization_;
    int cached_energy_;
    bool observables_valid_;

    void compute_observables();
};

// Utility functions
void check_cuda_error(cudaError_t err, const char* msg);
void check_cuda_last_error(const char* msg);
