#pragma once

#include <cstdint>
#include <random>
#include <vector>
#include <cuda_runtime.h>
#include <curand_kernel.h>

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

struct AveragedObservablesCuda {
    double m;
    double abs_m;
    double e;
    double m2;
    double e2;
    double m4;
};

AveragedObservablesCuda finalize_cuda(const SampleAccumulatorCuda& acc);

__global__ void init_rng_kernel(curandState* states, uint64_t seed, int N);

__global__ void init_spins_random_kernel(int8_t* spins, curandState* rng, int N);

__global__ void init_spins_ordered_kernel(int8_t* spins, int8_t value, int N);

// color: 0 = black sites (x+y even), 1 = white sites (x+y odd)
__global__ void metropolis_checkerboard_kernel(
    int8_t* spins,
    curandState* rng,
    int L,
    int color,
    float exp_dE4,
    float exp_dE8
);

__global__ void wolff_initialize_kernel(
    int8_t* spins,
    int* visited,
    int* queue,
    int* read_pos,
    int* write_pos,
    int* queue_capacity,
    curandState* rng,
    int N,
    int* seed_idx
);

__global__ void wolff_expand_kernel(
    int8_t* spins,
    int* visited,
    int* queue,
    int* read_pos,
    int* write_pos,
    int* queue_capacity,
    curandState* rng,
    int L,
    float wolff_prob
);

__global__ void wolff_flip_cluster_kernel(
    int8_t* spins,
    const int* visited,
    int N
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
    Ising2DCUDA(int L, uint64_t seed);
    ~Ising2DCUDA();

    void initialize_random();
    void initialize_ordered(int8_t spin_value = 1);
    void reseed(uint64_t seed);
    void set_temperature(double T);
    void sweep_metropolis();
    void sweep_metropolis(cudaStream_t stream);
    void sweep_wolff();

    double magnetization_per_spin();
    double energy_per_spin();

private:
    friend class ParallelTempering;

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
    int num_blocks_energy_;

    // For Wolff algorithm
    int* d_visited_;
    int* d_queue_;
    int* d_cluster_continue_;
    int* d_read_pos_;
    int* d_write_pos_;
    int* d_queue_capacity_;
    int* d_seed_idx_;
    curandState* d_rng_states_full_;

    // Cached observables
    int cached_magnetization_;
    int cached_energy_;
    bool observables_valid_;

    void compute_observables();
    void swap_spins_with(Ising2DCUDA& other);
};

class ParallelTempering {
public:
    ParallelTempering(
        int n_replicas,
        const std::vector<double>& temps,
        int L,
        uint64_t seed
    );
    ~ParallelTempering();

    void sweep_all(int n_sweeps);
    std::vector<double> attempt_swaps();
    double magnetization(int replica_idx);
    double energy(int replica_idx);
    std::vector<double> acceptance_rates() const;

    static std::vector<double> build_temperature_ladder(
        double t_min,
        double t_max,
        int n_replicas,
        double tc = 2.2692
    );

private:
    int n_replicas_;
    int L_;
    std::vector<double> temps_;
    std::vector<Ising2DCUDA*> replicas_;
    std::vector<cudaStream_t> streams_;

    std::vector<int> swap_attempts_;
    std::vector<int> swap_accepts_;
    std::vector<std::vector<int>> swap_history_;
    std::vector<int> swap_history_pos_;

    std::mt19937_64 swap_rng_;
    bool even_swap_pass_;
};

void check_cuda_error(cudaError_t err, const char* msg);
void check_cuda_last_error(const char* msg);
