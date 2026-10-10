#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <curand_kernel.h>

__global__ void init_rng_kernel(curandState* states, uint64_t seed, int N);

__global__ void init_spins_ordered_kernel(int8_t* spins, int8_t value, int N);

// Checkerboard update of one sublattice. color: 0 = sites with x+y even, 1 = odd.
// Requires even L: with periodic boundaries and odd L, x = 0 and x = L-1 have the
// same color but are neighbours, so a parallel sublattice update would race.
__global__ void metropolis_checkerboard_kernel(
    int8_t* spins,
    curandState* rng,
    int L,
    int color,
    float exp_dE4,
    float exp_dE8
);

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

__global__ void reduce_sum_kernel(int* data, int* result, int N);

class Ising2DCUDA {
public:
    Ising2DCUDA(int L, uint64_t seed);  // throws std::invalid_argument for odd or non-positive L
    ~Ising2DCUDA();
    Ising2DCUDA(const Ising2DCUDA&) = delete;
    Ising2DCUDA& operator=(const Ising2DCUDA&) = delete;

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
    curandState* d_rng_states_;  // one state per site of a sublattice (N / 2)

    // Precomputed Boltzmann factors
    float exp_dE4_;
    float exp_dE8_;

    // For reductions
    int* d_partial_mag_;
    int* d_partial_energy_;
    int* d_result_;
    int num_blocks_;
    int num_blocks_energy_;

    // Cached observables
    int cached_magnetization_;
    int cached_energy_;
    bool observables_valid_;

    void compute_observables();
};

void check_cuda_error(cudaError_t err, const char* msg);
void check_cuda_last_error(const char* msg);
