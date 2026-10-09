#include "ising_cuda.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <stdexcept>
#include <vector>

constexpr int BLOCK_SIZE = 256;
constexpr int ENERGY_BLOCK_X = 16;
constexpr int ENERGY_BLOCK_Y = 16;

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

__global__ void init_rng_kernel(curandState* states, uint64_t seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void init_spins_ordered_kernel(int8_t* spins, int8_t value, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        spins[idx] = value;
    }
}

__global__ void metropolis_checkerboard_kernel(
    int8_t* spins,
    curandState* rng,
    int L,
    int color,
    float exp_dE4,
    float exp_dE8
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= (L * L) / 2) return;
    const int half_L = L / 2;
    const int y = tid / half_L;
    const int col_half = tid - y * half_L;
    const int x = 2 * col_half + (((y & 1) == color) ? 0 : 1);
    int idx = y * L + x;
    int8_t s = spins[idx];
    int xp = (x + 1 < L) ? x + 1 : 0;
    int xm = (x > 0) ? x - 1 : L - 1;
    int yp = (y + 1 < L) ? y + 1 : 0;
    int ym = (y > 0) ? y - 1 : L - 1;
    int nn_sum = __ldg(&spins[y * L + xp]) +
                 __ldg(&spins[y * L + xm]) +
                 __ldg(&spins[yp * L + x]) +
                 __ldg(&spins[ym * L + x]);
    int dE = 2 * s * nn_sum;
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
    if (accept) {
        spins[idx] = -s;
    }
}
__global__ void compute_magnetization_kernel(
    const int8_t* spins,
    int* partial_sums,
    int N
) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? __ldg(&spins[idx]) : 0;
    __syncthreads();

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
__global__ void compute_energy_kernel(
    const int8_t* spins,
    int* partial_sums,
    int L
) {
    __shared__ int8_t tile[ENERGY_BLOCK_Y + 1][ENERGY_BLOCK_X + 1];
    __shared__ int sdata[ENERGY_BLOCK_X * ENERGY_BLOCK_Y];
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * ENERGY_BLOCK_X + tx;
    int y = blockIdx.y * ENERGY_BLOCK_Y + ty;
    int tid = ty * ENERGY_BLOCK_X + tx;
    int8_t s = 0;
    if (x < L && y < L) {
        s = spins[y * L + x];
    }
    tile[ty][tx] = s;
    if (tx == ENERGY_BLOCK_X - 1 && y < L) {
        int x_right = (x + 1 < L) ? x + 1 : 0;
        tile[ty][ENERGY_BLOCK_X] = __ldg(&spins[y * L + x_right]);
    }
    if (ty == ENERGY_BLOCK_Y - 1 && x < L) {
        int y_down = (y + 1 < L) ? y + 1 : 0;
        tile[ENERGY_BLOCK_Y][tx] = __ldg(&spins[y_down * L + x]);
    }
    __syncthreads();
    int local_energy = 0;
    if (x < L && y < L) {
        int8_t s_right = 0;
        int8_t s_down = 0;

        if (x + 1 < L) {
            s_right = (tx == ENERGY_BLOCK_X - 1) ? tile[ty][ENERGY_BLOCK_X] : tile[ty][tx + 1];
        } else {
            s_right = __ldg(&spins[y * L]);
        }

        if (y + 1 < L) {
            s_down = (ty == ENERGY_BLOCK_Y - 1) ? tile[ENERGY_BLOCK_Y][tx] : tile[ty + 1][tx];
        } else {
            s_down = __ldg(&spins[x]);
        }

        local_energy = -s * (s_right + s_down);
    }

    sdata[tid] = local_energy;
    __syncthreads();

    for (int stride = (ENERGY_BLOCK_X * ENERGY_BLOCK_Y) / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    if (tid == 0) {
        partial_sums[blockIdx.y * gridDim.x + blockIdx.x] = sdata[0];
    }
}

__global__ void reduce_sum_kernel(int* data, int* result, int N) {
    extern __shared__ int sdata[];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (idx < N) ? __ldg(&data[idx]) : 0;
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

Ising2DCUDA::Ising2DCUDA(int L, uint64_t seed)
        : L_(L), N_(L * L), d_spins_(nullptr), d_rng_states_(nullptr),
            exp_dE4_(1.0f), exp_dE8_(1.0f),
            d_partial_mag_(nullptr), d_partial_energy_(nullptr), d_result_(nullptr),
            cached_magnetization_(0), cached_energy_(0), observables_valid_(false)
{
    if (L < 2 || (L & 1) != 0) {
        throw std::invalid_argument("CUDA checkerboard update needs an even L >= 2, got L = " +
                                    std::to_string(L));
    }

    check_cuda_error(
        cudaMalloc(&d_spins_, N_ * sizeof(int8_t)),
        "Allocating spins"
    );

    const int half_N = N_ / 2;
    check_cuda_error(
        cudaMalloc(&d_rng_states_, half_N * sizeof(curandState)),
        "Allocating RNG states"
    );

    int num_blocks_rng = (half_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_rng_kernel<<<num_blocks_rng, BLOCK_SIZE>>>(d_rng_states_, seed, half_N);
    check_cuda_last_error("init_rng_kernel");

    num_blocks_ = (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int energy_grid_x = (L_ + ENERGY_BLOCK_X - 1) / ENERGY_BLOCK_X;
    int energy_grid_y = (L_ + ENERGY_BLOCK_Y - 1) / ENERGY_BLOCK_Y;
    num_blocks_energy_ = energy_grid_x * energy_grid_y;
    check_cuda_error(
        cudaMalloc(&d_partial_mag_, num_blocks_ * sizeof(int)),
        "Allocating partial magnetization"
    );
    check_cuda_error(
        cudaMalloc(&d_partial_energy_, num_blocks_energy_ * sizeof(int)),
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

void Ising2DCUDA::initialize_ordered(int8_t spin_value) {
    int num_blocks = (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int8_t val = (spin_value >= 0) ? 1 : -1;
    init_spins_ordered_kernel<<<num_blocks, BLOCK_SIZE>>>(d_spins_, val, N_);
    check_cuda_last_error("init_spins_ordered_kernel");
    observables_valid_ = false;
}

void Ising2DCUDA::set_temperature(double T) {
    double beta = 1.0 / T;
    exp_dE4_ = exp(-4.0 * beta);
    exp_dE8_ = exp(-8.0 * beta);
}

void Ising2DCUDA::sweep_metropolis() {
    const int num_blocks_color = (N_ / 2 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    metropolis_checkerboard_kernel<<<num_blocks_color, BLOCK_SIZE>>>(
        d_spins_, d_rng_states_, L_, 0, exp_dE4_, exp_dE8_
    );
    metropolis_checkerboard_kernel<<<num_blocks_color, BLOCK_SIZE>>>(
        d_spins_, d_rng_states_, L_, 1, exp_dE4_, exp_dE8_
    );
    observables_valid_ = false;
}

void Ising2DCUDA::compute_observables() {
    if (observables_valid_) return;

    compute_magnetization_kernel<<<num_blocks_, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_spins_, d_partial_mag_, N_
    );

    check_cuda_error(cudaMemset(d_result_, 0, sizeof(int)), "Clearing magnetization result");

    int reduce_blocks = (num_blocks_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_kernel<<<reduce_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_partial_mag_, d_result_, num_blocks_
    );

    check_cuda_error(
        cudaMemcpy(&cached_magnetization_, d_result_, sizeof(int), cudaMemcpyDeviceToHost),
        "Copying magnetization"
    );

    int energy_grid_x = (L_ + ENERGY_BLOCK_X - 1) / ENERGY_BLOCK_X;
    int energy_grid_y = (L_ + ENERGY_BLOCK_Y - 1) / ENERGY_BLOCK_Y;
    dim3 energy_grid(energy_grid_x, energy_grid_y);
    dim3 energy_block(ENERGY_BLOCK_X, ENERGY_BLOCK_Y);
    compute_energy_kernel<<<energy_grid, energy_block>>>(
        d_spins_, d_partial_energy_, L_
    );
    check_cuda_last_error("compute_energy_kernel");

    check_cuda_error(cudaMemset(d_result_, 0, sizeof(int)), "Clearing energy result");

    reduce_blocks = (num_blocks_energy_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_kernel<<<reduce_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_partial_energy_, d_result_, num_blocks_energy_
    );

    check_cuda_error(
        cudaMemcpy(&cached_energy_, d_result_, sizeof(int), cudaMemcpyDeviceToHost),
        "Copying energy"
    );
    observables_valid_ = true;
}

double Ising2DCUDA::magnetization_per_spin() {
    compute_observables();
    return static_cast<double>(cached_magnetization_) / N_;
}

double Ising2DCUDA::energy_per_spin() {
    compute_observables();
    return static_cast<double>(cached_energy_) / N_;
}
