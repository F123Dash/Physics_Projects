#include "ising_cuda.cuh"
#include <cstdio>
#include <cmath>
#include <cstdlib>

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

__global__ void init_rng_kernel(curandState* states, uint64_t seed, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void init_spins_random_kernel(int8_t* spins, curandState* rng, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        float r = curand_uniform(&rng[idx]);
        spins[idx] = (r < 0.5f) ? 1 : -1;
    }
}

__global__ void init_spins_ordered_kernel(int8_t* spins, int8_t value, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        spins[idx] = value;
    }
}

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
    int N = L * L;
    int color_count = (N + (color == 0 ? 1 : 0)) / 2;

    if (tid >= color_count) return;

    int x = 0;
    int y = 0;

    if ((L & 1) == 0) {
        // Even L: every row has L/2 sites of each color.
        int row = tid / (L / 2);
        int col_half = tid % (L / 2);
        y = row;
        if ((y & 1) == color) {
            x = col_half * 2;
        } else {
            x = col_half * 2 + 1;
        }
    } else {
        int a = (L + 1) / 2;
        int b = L / 2;        
        int full_pairs = L / 2;
        int full_pair_sites = full_pairs * L;
        if (tid < full_pair_sites) {
            int pair = tid / L;
            int rem = tid - pair * L;
            if (color == 0) {
                if (rem < a) {
                    y = 2 * pair;
                    x = rem * 2;
                } else {
                    y = 2 * pair + 1;
                    x = (rem - a) * 2 + 1;
                }
            } else {
                if (rem < b) {
                    y = 2 * pair;
                    x = rem * 2 + 1;
                } else {
                    y = 2 * pair + 1;
                    x = (rem - b) * 2;
                }
            }
        } else {
            y = L - 1;
            int row_count = ((y & 1) == color) ? a : b;
            int col_half = tid - full_pair_sites;
            if (col_half >= row_count) return;
            if ((y & 1) == color) {
                x = col_half * 2;
            } else {
                x = col_half * 2 + 1;
            }
        }
    }

    int idx = y * L + x;

    int8_t s = spins[idx];

    int xp = (x + 1 < L) ? x + 1 : 0;
    int xm = (x > 0) ? x - 1 : L - 1;
    int yp = (y + 1 < L) ? y + 1 : 0;
    int ym = (y > 0) ? y - 1 : L - 1;

    int nn_sum = spins[y * L + xp] +
                 spins[y * L + xm] +
                 spins[yp * L + x] +
                 spins[ym * L + x];

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

    sdata[tid] = (idx < N) ? static_cast<int>(spins[idx]) : 0;
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

// Energy = -J * sum over neighbors. We count each bond once (right and down).
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
        tile[ty][ENERGY_BLOCK_X] = spins[y * L + x_right];
    }
    if (ty == ENERGY_BLOCK_Y - 1 && x < L) {
        int y_down = (y + 1 < L) ? y + 1 : 0;
        tile[ENERGY_BLOCK_Y][tx] = spins[y_down * L + x];
    }

    __syncthreads();

    int local_energy = 0;
    if (x < L && y < L) {
        int8_t s_right = 0;
        int8_t s_down = 0;

        if (x + 1 < L) {
            s_right = (tx == ENERGY_BLOCK_X - 1) ? tile[ty][ENERGY_BLOCK_X] : tile[ty][tx + 1];
        } else {
            s_right = spins[y * L];
        }

        if (y + 1 < L) {
            s_down = (ty == ENERGY_BLOCK_Y - 1) ? tile[ENERGY_BLOCK_Y][tx] : tile[ty + 1][tx];
        } else {
            s_down = spins[x];
        }

        local_energy = -static_cast<int>(s) * (static_cast<int>(s_right) + static_cast<int>(s_down));
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
) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        for (int i = 0; i < N; ++i) {
            visited[i] = 0;
        }
        
        curandState local_state = rng[0];
        int idx = (int)(curand_uniform(&local_state) * N);
        rng[0] = local_state;
        
        *seed_idx = idx;
        visited[idx] = 1;
        queue[0] = idx;
        *read_pos = 0;
        *write_pos = 1;
        *queue_capacity = N;
    }
    __syncthreads();
}

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
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    int curr_read = *read_pos;
    int curr_write = *write_pos;
    int total_in_queue = curr_write - curr_read;
    
    if (tid >= total_in_queue || curr_read >= curr_write) {
        return;
    }
    
    int queue_idx = curr_read + tid;
    int site_idx = queue[queue_idx];
    
    int x = site_idx % L;
    int y = site_idx / L;
    int8_t seed_spin = spins[site_idx];
    
    int neighbors[4] = {
        y * L + ((x + 1 < L) ? x + 1 : 0),  // right
        y * L + ((x > 0) ? x - 1 : L - 1),  // left
        ((y + 1 < L) ? y + 1 : 0) * L + x,  // down
        ((y > 0) ? y - 1 : L - 1) * L + x   // up
    };
    
    for (int i = 0; i < 4; ++i) {
        int neighbor_idx = neighbors[i];
        
        if (!visited[neighbor_idx] && spins[neighbor_idx] == seed_spin) {
            int old_val = atomicCAS(&visited[neighbor_idx], 0, 1);
            
            if (old_val == 0) {
                curandState local_state = rng[threadIdx.x + blockIdx.x * blockDim.x];
                float r = curand_uniform(&local_state);
                rng[threadIdx.x + blockIdx.x * blockDim.x] = local_state;
                
                if (r < wolff_prob) {
                    int pos = atomicAdd(write_pos, 1);
                    if (pos < *queue_capacity) {
                        queue[pos] = neighbor_idx;
                    }
                }
            }
        }
    }
}

__global__ void wolff_flip_cluster_kernel(
    int8_t* spins,
    const int* visited,
    int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N && visited[idx]) {
        spins[idx] = -spins[idx];
    }
}

Ising2DCUDA::Ising2DCUDA(int L, uint64_t seed)
    : L_(L), N_(L * L), d_spins_(nullptr), d_rng_states_(nullptr),
      exp_dE4_(1.0f), exp_dE8_(1.0f),
      d_partial_mag_(nullptr), d_partial_energy_(nullptr), d_result_(nullptr),
    d_visited_(nullptr), d_queue_(nullptr), d_read_pos_(nullptr), d_write_pos_(nullptr),
      d_queue_capacity_(nullptr), d_seed_idx_(nullptr), d_rng_states_full_(nullptr),
    cached_magnetization_(0), cached_energy_(0), observables_valid_(false)
{
    check_cuda_error(
        cudaMalloc(&d_spins_, N_ * sizeof(int8_t)),
        "Allocating spins"
    );

    int half_N = N_ / 2;
    check_cuda_error(
        cudaMalloc(&d_rng_states_, half_N * sizeof(curandState)),
        "Allocating RNG states"
    );

    int num_blocks_rng = (half_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_rng_kernel<<<num_blocks_rng, BLOCK_SIZE>>>(d_rng_states_, seed, half_N);
    check_cuda_last_error("init_rng_kernel");
    cudaDeviceSynchronize();

    check_cuda_error(
        cudaMalloc(&d_rng_states_full_, N_ * sizeof(curandState)),
        "Allocating full RNG states for Wolff"
    );
    int num_blocks_full_rng = (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_rng_kernel<<<num_blocks_full_rng, BLOCK_SIZE>>>(d_rng_states_full_, seed + 1, N_);
    check_cuda_last_error("init_rng_kernel for full states");
    cudaDeviceSynchronize();

    check_cuda_error(
        cudaMalloc(&d_visited_, N_ * sizeof(int)),
        "Allocating visited array"
    );
    check_cuda_error(
        cudaMalloc(&d_queue_, N_ * sizeof(int)),
        "Allocating queue"
    );
    check_cuda_error(
        cudaMalloc(&d_read_pos_, sizeof(int)),
        "Allocating queue read position"
    );
    check_cuda_error(
        cudaMalloc(&d_write_pos_, sizeof(int)),
        "Allocating queue write position"
    );
    check_cuda_error(
        cudaMalloc(&d_queue_capacity_, sizeof(int)),
        "Allocating queue capacity"
    );
    check_cuda_error(
        cudaMalloc(&d_seed_idx_, sizeof(int)),
        "Allocating seed index"
    );

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
    if (d_rng_states_full_) cudaFree(d_rng_states_full_);
    if (d_partial_mag_) cudaFree(d_partial_mag_);
    if (d_partial_energy_) cudaFree(d_partial_energy_);
    if (d_result_) cudaFree(d_result_);
    if (d_visited_) cudaFree(d_visited_);
    if (d_queue_) cudaFree(d_queue_);
    if (d_read_pos_) cudaFree(d_read_pos_);
    if (d_write_pos_) cudaFree(d_write_pos_);
    if (d_queue_capacity_) cudaFree(d_queue_capacity_);
    if (d_seed_idx_) cudaFree(d_seed_idx_);
}

void Ising2DCUDA::initialize_random() {
    int num_blocks = (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE;

    curandState* d_full_rng;
    check_cuda_error(
        cudaMalloc(&d_full_rng, N_ * sizeof(curandState)),
        "Allocating full RNG for init"
    );

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

void Ising2DCUDA::reseed(uint64_t seed) {
    int half_N = N_ / 2;
    int num_blocks_rng = (half_N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_rng_kernel<<<num_blocks_rng, BLOCK_SIZE>>>(d_rng_states_, seed, half_N);
    check_cuda_last_error("init_rng_kernel reseed");

    int num_blocks_full_rng = (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_rng_kernel<<<num_blocks_full_rng, BLOCK_SIZE>>>(d_rng_states_full_, seed + 1, N_);
    check_cuda_last_error("init_rng_kernel full reseed");

    cudaDeviceSynchronize();
}

void Ising2DCUDA::set_temperature(double T) {
    double beta = 1.0 / T;
    exp_dE4_ = static_cast<float>(exp(-4.0 * beta));
    exp_dE8_ = static_cast<float>(exp(-8.0 * beta));
}

void Ising2DCUDA::sweep_metropolis() {
    int color0_count = (N_ + 1) / 2;
    int color1_count = N_ / 2;
    int num_blocks_color0 = (color0_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_color1 = (color1_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Update black sites (color = 0)
    metropolis_checkerboard_kernel<<<num_blocks_color0, BLOCK_SIZE>>>(
        d_spins_, d_rng_states_, L_, 0, exp_dE4_, exp_dE8_
    );

    // Update white sites (color = 1)
    metropolis_checkerboard_kernel<<<num_blocks_color1, BLOCK_SIZE>>>(
        d_spins_, d_rng_states_, L_, 1, exp_dE4_, exp_dE8_
    );

    observables_valid_ = false;
}

void Ising2DCUDA::sweep_wolff(){
    int num_blocks = (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    // Wolff probability: P = 1 - exp(-2*beta)
    // If exp_dE4 = exp(-4*beta), then exp(-2*beta) = sqrt(exp_dE4)
    // wolff_prob = 1 - sqrt(exp_dE4)
    float wolff_prob = 1.0f - sqrtf(exp_dE4_);
    
    int num_blocks_init = 1;
    wolff_initialize_kernel<<<num_blocks_init, BLOCK_SIZE>>>(
        d_spins_, d_visited_, d_queue_, d_read_pos_, d_write_pos_, d_queue_capacity_,
        d_rng_states_full_, N_, d_seed_idx_
    );
    check_cuda_last_error("wolff_initialize_kernel");
    
    int max_iterations = N_;  // Maximum iterations to prevent infinite loops
    int iteration = 0;
    
    while (iteration < max_iterations) {
        int h_read_pos = 0;
        int h_write_pos = 0;
        cudaMemcpy(&h_read_pos, d_read_pos_, sizeof(int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&h_write_pos, d_write_pos_, sizeof(int), cudaMemcpyDeviceToHost);

        int frontier_size = h_write_pos - h_read_pos;
        if (frontier_size <= 0) break;
        
        int frontier_blocks = (frontier_size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        wolff_expand_kernel<<<frontier_blocks, BLOCK_SIZE>>>(
            d_spins_, d_visited_, d_queue_, d_read_pos_, d_write_pos_,
            d_queue_capacity_, d_rng_states_full_, L_, wolff_prob
        );
        check_cuda_last_error("wolff_expand_kernel");

        cudaMemcpy(d_read_pos_, &h_write_pos, sizeof(int), cudaMemcpyHostToDevice);
        iteration++;
    }
    
    wolff_flip_cluster_kernel<<<num_blocks, BLOCK_SIZE>>>(
        d_spins_, d_visited_, N_
    );
    check_cuda_last_error("wolff_flip_cluster_kernel");
    
    cudaDeviceSynchronize();
    observables_valid_ = false;
}


void Ising2DCUDA::compute_observables() {
    if (observables_valid_) return;

    compute_magnetization_kernel<<<num_blocks_, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_spins_, d_partial_mag_, N_
    );

    cudaMemset(d_result_, 0, sizeof(int));

    int reduce_blocks = (num_blocks_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_kernel<<<reduce_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_partial_mag_, d_result_, num_blocks_
    );

    cudaMemcpy(&cached_magnetization_, d_result_, sizeof(int), cudaMemcpyDeviceToHost);

    int energy_grid_x = (L_ + ENERGY_BLOCK_X - 1) / ENERGY_BLOCK_X;
    int energy_grid_y = (L_ + ENERGY_BLOCK_Y - 1) / ENERGY_BLOCK_Y;
    dim3 energy_grid(energy_grid_x, energy_grid_y);
    dim3 energy_block(ENERGY_BLOCK_X, ENERGY_BLOCK_Y);
    compute_energy_kernel<<<energy_grid, energy_block>>>(
        d_spins_, d_partial_energy_, L_
    );
    check_cuda_last_error("compute_energy_kernel");

    cudaMemset(d_result_, 0, sizeof(int));

    reduce_blocks = (num_blocks_energy_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    reduce_sum_kernel<<<reduce_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(int)>>>(
        d_partial_energy_, d_result_, num_blocks_energy_
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
