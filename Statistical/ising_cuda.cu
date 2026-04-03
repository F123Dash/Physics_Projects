#include "ising_cuda.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>

// Block size for kernels
constexpr int BLOCK_SIZE = 256;

// Error checking utilities
void check_cuda_error(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

void check_cuda_last_error(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error: %s: %s\n", msg, cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}

// SampleAccumulatorCuda methods
__host__ void SampleAccumulatorCuda::reset() {
    sum_m = 0.0;
    sum_abs_m = 0.0;
    sum_e = 0.0;
    sum_m2 = 0.0;
    sum_e2 = 0.0;
    sum_m4 = 0.0;
    count = 0;
}

__host__ void SampleAccumulatorCuda::add(double m_per_spin, double e_per_spin) {
    sum_m += m_per_spin;
    sum_abs_m += fabs(m_per_spin);
    sum_e += e_per_spin;
    sum_m2 += m_per_spin * m_per_spin;
    sum_e2 += e_per_spin * e_per_spin;
    double m2 = m_per_spin * m_per_spin;
    sum_m4 += m2 * m2;
    ++count;
}

AveragedObservablesCuda finalize_cuda(const SampleAccumulatorCuda& acc) {
    AveragedObservablesCuda out;
    if (acc.count <= 0) {
        out.m = out.abs_m = out.e = out.m2 = out.e2 = out.m4 = 0.0;
        return out;
    }

    double inv = 1.0 / static_cast<double>(acc.count);
    out.m = acc.sum_m * inv;
    out.abs_m = acc.sum_abs_m * inv;
    out.e = acc.sum_e * inv;
    out.m2 = acc.sum_m2 * inv;
    out.e2 = acc.sum_e2 * inv;
    out.m4 = acc.sum_m4 * inv;
    return out;
}

// Kernel: Initialize cuRAND states
__global__ void init_rng_kernel(curandState* states, uint64_t seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

// Kernel: Initialize spins randomly
__global__ void init_spins_random_kernel(int8_t* spins, curandState* rng, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float r = curand_uniform(&rng[idx]);
        spins[idx] = (r < 0.5f) ? 1 : -1;
    }
}

// Kernel: Initialize spins to ordered state
__global__ void init_spins_ordered_kernel(int8_t* spins, int8_t value, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        spins[idx] = value;
    }
}

// Kernel: Checkerboard Metropolis update
// Each thread handles one site of the active color
// color: 0 = even sites (x+y even), 1 = odd sites (x+y odd)
__global__ void metropolis_checkerboard_kernel(
    int8_t* spins,
    curandState* rng,
    int L,
    int color,
    float exp_dE4,
    float exp_dE8
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int half_N = (L * L) / 2;

    if (tid >= half_N) return;

    // Map thread ID to lattice coordinates for checkerboard pattern
    // For a row y: if (y % 2 == color), sites start at x=0,2,4,...
    //              else sites start at x=1,3,5,...
    int row = tid / (L / 2);
    int col_half = tid % (L / 2);

    int y = row;
    int x;
    if ((y & 1) == color) {
        x = col_half * 2;       // Even columns: 0, 2, 4, ...
    } else {
        x = col_half * 2 + 1;   // Odd columns: 1, 3, 5, ...
    }

    int idx = y * L + x;

    // Get current spin
    int8_t s = spins[idx];

    // Get neighbor spins with periodic boundary conditions
    int xp = (x + 1 < L) ? x + 1 : 0;
    int xm = (x > 0) ? x - 1 : L - 1;
    int yp = (y + 1 < L) ? y + 1 : 0;
    int ym = (y > 0) ? y - 1 : L - 1;

    int nn_sum = spins[y * L + xp] +
                 spins[y * L + xm] +
                 spins[yp * L + x] +
                 spins[ym * L + x];

    // Compute energy change: dE = 2 * s * nn_sum
    int dE = 2 * s * nn_sum;

    // Metropolis acceptance
    bool accept = false;
    if (dE <= 0) {
        accept = true;
    } else {
        float r = curand_uniform(&rng[tid]);
        if (dE == 4) {
            accept = (r < exp_dE4);
        } else if (dE == 8) {
            accept = (r < exp_dE8);
        }
    }

    // Flip spin if accepted
    if (accept) {
        spins[idx] = -s;
    }
}

// Kernel: Compute partial sums of magnetization
__global__ void compute_magnetization_kernel(
    const int8_t* spins,
    int* partial_sums,
    int N
) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load into shared memory
    sdata[tid] = (idx < N) ? static_cast<int>(spins[idx]) : 0;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel: Compute partial sums of energy
// Energy = -J * sum over neighbors. We count each bond once (right and down).
__global__ void compute_energy_kernel(
    const int8_t* spins,
    int* partial_sums,
    int L
) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int N = L * L;

    int local_energy = 0;
    if (idx < N) {
        int x = idx % L;
        int y = idx / L;
        int8_t s = spins[idx];

        // Right neighbor (periodic)
        int xp = (x + 1 < L) ? x + 1 : 0;
        local_energy += -s * spins[y * L + xp];

        // Down neighbor (periodic)
        int yp = (y + 1 < L) ? y + 1 : 0;
        local_energy += -s * spins[yp * L + x];
    }

    sdata[tid] = local_energy;
    __syncthreads();

    // Reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// Kernel: Final reduction sum
__global__ void reduce_sum_kernel(int* data, int* result, int N) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? data[idx] : 0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        atomicAdd(result, sdata[0]);
    }
}

// Ising2DCUDA class implementation

Ising2DCUDA::Ising2DCUDA(int L, uint64_t seed)
    : L_(L), N_(L * L), d_spins_(nullptr), d_rng_states_(nullptr),
      exp_dE4_(1.0f), exp_dE8_(1.0f),
      d_partial_mag_(nullptr), d_partial_energy_(nullptr), d_result_(nullptr),
      cached_magnetization_(0), cached_energy_(0), observables_valid_(false)
{
    // Allocate device memory for spins
    check_cuda_error(
        cudaMalloc(&d_spins_, N_ * sizeof(int8_t)),
        "Allocating spins"
    );

    // Allocate cuRAND states (one per half of lattice for checkerboard)
    int half_N = N_ / 2;
    check_cuda_error(
        cudaMalloc(&d_rng_states_, half_N * sizeof(curandState)),
        "Allocating RNG states"
    );

    // Initialize RNG states
    int num_blocks_rng = (half_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_rng_kernel<<<num_blocks_rng, BLOCK_SIZE>>>(d_rng_states_, seed, half_N);
    check_cuda_last_error("init_rng_kernel");
    cudaDeviceSynchronize();

    // Allocate reduction buffers
    num_blocks_ = (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    check_cuda_error(
        cudaMalloc(&d_partial_mag_, num_blocks_ * sizeof(int)),
        "Allocating partial magnetization"
    );
    check_cuda_error(
        cudaMalloc(&d_partial_energy_, num_blocks_ * sizeof(int)),
        "Allocating partial energy"
    );
    check_cuda_error(
        cudaMalloc(&d_result_, sizeof(int)),
        "Allocating result"
    );
}

Ising2DCUDA::~Ising2DCUDA() {
    if (d_spins_) cudaFree(d_spins_);
    if (d_rng_states_) cudaFree(d_rng_states_);
    if (d_partial_mag_) cudaFree(d_partial_mag_);
    if (d_partial_energy_) cudaFree(d_partial_energy_);
    if (d_result_) cudaFree(d_result_);
}

void Ising2DCUDA::initialize_random() {
    int num_blocks = (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // We need full RNG states for initialization
    curandState* d_full_rng;
    check_cuda_error(
        cudaMalloc(&d_full_rng, N_ * sizeof(curandState)),
        "Allocating full RNG for init"
    );

    // Initialize full RNG with different offset
    init_rng_kernel<<<num_blocks, BLOCK_SIZE>>>(d_full_rng, 42ULL, N_);
    check_cuda_last_error("init full rng");

    init_spins_random_kernel<<<num_blocks, BLOCK_SIZE>>>(d_spins_, d_full_rng, N_);
    check_cuda_last_error("init_spins_random_kernel");
    cudaDeviceSynchronize();

    cudaFree(d_full_rng);
    observables_valid_ = false;
}

void Ising2DCUDA::initialize_ordered(int8_t spin_value) {
    int num_blocks = (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int8_t val = (spin_value >= 0) ? 1 : -1;
    init_spins_ordered_kernel<<<num_blocks, BLOCK_SIZE>>>(d_spins_, val, N_);
    check_cuda_last_error("init_spins_ordered_kernel");
    cudaDeviceSynchronize();
    observables_valid_ = false;
}

void Ising2DCUDA::set_temperature(double T) {
    double beta = 1.0 / T;
    exp_dE4_ = static_cast<float>(exp(-4.0 * beta));
    exp_dE8_ = static_cast<float>(exp(-8.0 * beta));
}

void Ising2DCUDA::sweep_metropolis() {
    int half_N = N_ / 2;
    int num_blocks = (half_N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Update black sites (color = 0)
    metropolis_checkerboard_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_spins_, d_rng_states_, L_, 0, exp_dE4_, exp_dE8_
    );

    // Update white sites (color = 1)
    metropolis_checkerboard_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_spins_, d_rng_states_, L_, 1, exp_dE4_, exp_dE8_
    );

    observables_valid_ = false;
}

void Ising2DCUDA::compute_observables() {
    if (observables_valid_) return;

    // Compute magnetization
    compute_magnetization_kernel<<<num_blocks_, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_spins_, d_partial_mag_, N_
    );

    // Zero the result
    cudaMemset(d_result_, 0, sizeof(int));

    // Final reduction for magnetization
    int reduce_blocks = (num_blocks_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_kernel<<<reduce_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_partial_mag_, d_result_, num_blocks_
    );

    cudaMemcpy(&cached_magnetization_, d_result_, sizeof(int), cudaMemcpyDeviceToHost);

    // Compute energy
    compute_energy_kernel<<<num_blocks_, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_spins_, d_partial_energy_, L_
    );

    // Zero the result
    cudaMemset(d_result_, 0, sizeof(int));

    // Final reduction for energy
    reduce_sum_kernel<<<reduce_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_partial_energy_, d_result_, num_blocks_
    );

    cudaMemcpy(&cached_energy_, d_result_, sizeof(int), cudaMemcpyDeviceToHost);

    cudaDeviceSynchronize();
    observables_valid_ = true;
}

double Ising2DCUDA::magnetization_per_spin() {
    compute_observables();
    return static_cast<double>(cached_magnetization_) / static_cast<double>(N_);
}

double Ising2DCUDA::energy_per_spin() {
    compute_observables();
    return static_cast<double>(cached_energy_) / static_cast<double>(N_);
}
