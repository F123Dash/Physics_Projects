#include "ising_cuda.cuh"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
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
    sum_e2 += e_per_spin * e_per_spin;
    double m2 = m_per_spin * m_per_spin;
    sum_m2 += m2;
    sum_m4 += m2 * m2;
    ++count;
}

AveragedObservablesCuda finalize_cuda(const SampleAccumulatorCuda& acc) {
    AveragedObservablesCuda out;
    if (acc.count <= 0) {
        out.m = out.abs_m = out.e = out.m2 = out.e2 = out.m4 = 0.0;
        return out;
    }

    double inv = 1.0 / acc.count;
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
            d_rng_states_full_(nullptr), rng_ready_event_(nullptr),
            exp_dE4_(1.0f), exp_dE8_(1.0f),
            d_partial_mag_(nullptr), d_partial_energy_(nullptr), d_result_(nullptr),
            cached_magnetization_(0), cached_energy_(0), observables_valid_(false)
{
    check_cuda_error(
        cudaEventCreateWithFlags(&rng_ready_event_, cudaEventDisableTiming),
        "Creating RNG ready event"
    );

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

    check_cuda_error(
        cudaMalloc(&d_rng_states_full_, N_ * sizeof(curandState)),
        "Allocating full RNG states"
    );
    int num_blocks_full_rng = (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_rng_kernel<<<num_blocks_full_rng, BLOCK_SIZE>>>(d_rng_states_full_, seed + 1, N_);
    check_cuda_last_error("init_rng_kernel full states");
    check_cuda_error(
        cudaEventRecord(rng_ready_event_, 0),
        "Recording RNG ready event"
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
    if (rng_ready_event_) cudaEventDestroy(rng_ready_event_);
}

void Ising2DCUDA::initialize_spins_random() {
    int num_blocks = (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    init_spins_random_kernel<<<num_blocks, BLOCK_SIZE>>>(d_spins_, d_rng_states_full_, N_);
    check_cuda_last_error("init_spins_random_kernel");
    observables_valid_ = false;
}

void Ising2DCUDA::initialize_ordered(int8_t spin_value) {
    int num_blocks = (N_ + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int8_t val = (spin_value >= 0) ? 1 : -1;
    init_spins_ordered_kernel<<<num_blocks, BLOCK_SIZE>>>(d_spins_, val, N_);
    check_cuda_last_error("init_spins_ordered_kernel");
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
    check_cuda_error(
        cudaEventRecord(rng_ready_event_, 0),
        "Recording RNG ready event"
    );
}

void Ising2DCUDA::set_temperature(double T) {
    double beta = 1.0 / T;
    exp_dE4_ = exp(-4.0 * beta);
    exp_dE8_ = exp(-8.0 * beta);
}

void Ising2DCUDA::sweep_metropolis() {
    sweep_metropolis(0);
}

void Ising2DCUDA::sweep_metropolis(cudaStream_t stream) {
    check_cuda_error(
        cudaStreamWaitEvent(stream, rng_ready_event_, 0),
        "Waiting for RNG init event"
    );
    int color0_count = (N_ + 1) / 2;
    int color1_count = N_ / 2;
    int num_blocks_color0 = (color0_count + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int num_blocks_color1 = (color1_count + BLOCK_SIZE - 1) / BLOCK_SIZE;

    metropolis_checkerboard_kernel<<<num_blocks_color0, BLOCK_SIZE, 0, stream>>>(
        d_spins_, d_rng_states_, L_, 0, exp_dE4_, exp_dE8_
    );

    metropolis_checkerboard_kernel<<<num_blocks_color1, BLOCK_SIZE, 0, stream>>>(
        d_spins_, d_rng_states_, L_, 1, exp_dE4_, exp_dE8_
    );

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

void Ising2DCUDA::swap_spins_with(Ising2DCUDA& other) {
    std::swap(d_spins_, other.d_spins_);
    observables_valid_ = false;
    other.observables_valid_ = false;
}

ParallelTempering::ParallelTempering(
    int n_replicas,
    const std::vector<double>& temps,
    int L,
    uint64_t seed
)
    : n_replicas_(n_replicas),
      L_(L),
      temps_(temps),
      swap_rng_(seed),
      even_swap_pass_(true) {
    if (n_replicas_ <= 0 || static_cast<int>(temps_.size()) != n_replicas_) {
        throw std::runtime_error("ParallelTempering: temperature count must match n_replicas and be > 0.");
    }

    replicas_.reserve(static_cast<size_t>(n_replicas_));
    streams_.reserve(static_cast<size_t>(n_replicas_));

    for (int i = 0; i < n_replicas_; ++i) {
        replicas_.push_back(new Ising2DCUDA(L_, seed + static_cast<uint64_t>(i + 1)));
        replicas_[i]->set_temperature(temps_[i]);

        cudaStream_t stream;
        check_cuda_error(cudaStreamCreate(&stream), "Creating replica stream");
        streams_.push_back(stream);
    }

    const int pairs = n_replicas_ - 1;
    swap_attempts_.assign(static_cast<size_t>(pairs), 0);
    swap_accepts_.assign(static_cast<size_t>(pairs), 0);
    swap_history_.assign(static_cast<size_t>(pairs), std::vector<int>(100, 0));
    swap_history_pos_.assign(static_cast<size_t>(pairs), 0);
}

ParallelTempering::~ParallelTempering() {
    for (cudaStream_t stream : streams_) {
        cudaStreamDestroy(stream);
    }
    for (Ising2DCUDA* replica : replicas_) {
        delete replica;
    }
}

void ParallelTempering::sweep_all(int n_sweeps) {
    if (n_sweeps <= 0) return;

    for (int s = 0; s < n_sweeps; ++s) {
        for (int i = 0; i < n_replicas_; ++i) {
            replicas_[i]->sweep_metropolis(streams_[i]);
        }
    }

    for (cudaStream_t stream : streams_) {
        check_cuda_error(cudaStreamSynchronize(stream), "Synchronizing replica stream");
    }
}

std::vector<double> ParallelTempering::attempt_swaps() {
    const int pairs = n_replicas_ - 1;
    std::vector<double> rates(static_cast<size_t>(pairs), 0.0);
    if (pairs <= 0) return rates;

    for (cudaStream_t stream : streams_) {
        check_cuda_error(cudaStreamSynchronize(stream), "Synchronizing before swaps");
    }

    const double n_spins = static_cast<double>(L_) * L_;
    std::vector<double> energies(static_cast<size_t>(n_replicas_), 0.0);

    // Energies live on the device; swap acceptance needs host access, so we must copy them back.
    for (int i = 0; i < n_replicas_; ++i) {
        energies[i] = replicas_[i]->energy_per_spin() * n_spins;
    }

    std::uniform_real_distribution<double> unif01(0.0, 1.0);
    const int start = even_swap_pass_ ? 0 : 1;
    for (int i = start; i < n_replicas_ - 1; i += 2) {
        const int j = i + 1;
        const double beta_i = 1.0 / temps_[i];
        const double beta_j = 1.0 / temps_[j];
        const double delta = (beta_i - beta_j) * (energies[i] - energies[j]);
        const double accept_prob = (delta >= 0.0) ? 1.0 : std::exp(delta);
        const bool accepted = (unif01(swap_rng_) < accept_prob);

        const int pair_idx = i;
        const int history_idx = swap_history_pos_[pair_idx];
        if (swap_attempts_[pair_idx] < 100) {
            swap_attempts_[pair_idx] += 1;
        } else {
            swap_accepts_[pair_idx] -= swap_history_[pair_idx][history_idx];
        }
        swap_history_[pair_idx][history_idx] = accepted ? 1 : 0;
        swap_accepts_[pair_idx] += accepted ? 1 : 0;
        swap_history_pos_[pair_idx] = (history_idx + 1) % 100;

        if (accepted) {
            // Pointer swap avoids device-to-device data copies; spins move between temperatures at O(1) cost.
            replicas_[i]->swap_spins_with(*replicas_[j]);
        }
    }

    even_swap_pass_ = !even_swap_pass_;

    for (int p = 0; p < pairs; ++p) {
        if (swap_attempts_[p] > 0) {
            rates[p] = static_cast<double>(swap_accepts_[p]) / static_cast<double>(swap_attempts_[p]);
        }
    }

    return rates;
}

double ParallelTempering::magnetization(int replica_idx) {
    return replicas_[replica_idx]->magnetization_per_spin();
}

double ParallelTempering::energy(int replica_idx) {
    return replicas_[replica_idx]->energy_per_spin();
}

std::vector<double> ParallelTempering::acceptance_rates() const {
    const int pairs = n_replicas_ - 1;
    std::vector<double> rates(static_cast<size_t>(pairs), 0.0);
    for (int p = 0; p < pairs; ++p) {
        if (swap_attempts_[p] > 0) {
            rates[p] = static_cast<double>(swap_accepts_[p]) / static_cast<double>(swap_attempts_[p]);
        }
    }
    return rates;
}

std::vector<double> ParallelTempering::build_temperature_ladder(
    double t_min,
    double t_max,
    int n_replicas,
    double tc
) {
    std::vector<double> temps;
    if (n_replicas <= 0) return temps;

    if (t_max <= t_min || n_replicas == 1) {
        temps.assign(static_cast<size_t>(n_replicas), t_min);
        return temps;
    }

    const double dense_low = tc - 0.15;
    const double dense_high = tc + 0.15;

    if (t_min >= dense_low || t_max <= dense_high) {
        temps.resize(static_cast<size_t>(n_replicas));
        const double step = (t_max - t_min) / (n_replicas - 1);
        for (int i = 0; i < n_replicas; ++i) {
            temps[static_cast<size_t>(i)] = t_min + step * static_cast<double>(i);
        }
        return temps;
    }

    int n_dense = static_cast<int>(std::lround(0.4 * n_replicas));
    n_dense = std::max(1, std::min(n_dense, n_replicas));
    int n_outer = n_replicas - n_dense;
    int n_low = n_outer / 2;
    int n_high = n_outer - n_low;

    auto linspace = [](double a, double b, int n) {
        std::vector<double> v;
        if (n <= 0) return v;
        if (n == 1) {
            v.push_back(a);
            return v;
        }
        v.resize(static_cast<size_t>(n));
        const double step = (b - a) / (n - 1);
        for (int i = 0; i < n; ++i) {
            v[static_cast<size_t>(i)] = a + step * static_cast<double>(i);
        }
        return v;
    };

    const std::vector<double> low = linspace(t_min, dense_low, n_low);
    const std::vector<double> mid = linspace(dense_low, dense_high, n_dense);
    const std::vector<double> high = linspace(dense_high, t_max, n_high);

    temps.reserve(static_cast<size_t>(n_replicas));
    temps.insert(temps.end(), low.begin(), low.end());
    if (!mid.empty()) {
        temps.insert(temps.end(), mid.begin() + (low.empty() ? 0 : 1), mid.end());
    }
    if (!high.empty()) {
        temps.insert(temps.end(), high.begin() + (mid.empty() ? 0 : 1), high.end());
    }
    if (static_cast<int>(temps.size()) > n_replicas) {
        temps.resize(static_cast<size_t>(n_replicas));
    }
    return temps;
}
