#include "ising.hpp"
#include <cmath>
#include <queue>
#include <vector>

Ising2D::Ising2D(int L, std::mt19937_64& rng)
    : L_(L),
      N_(L * L),
      spins_(N_, 1),
      rng_(rng),
      unif01_(0.0, 1.0),
      site_dist_(0, L * L - 1) {}

int Ising2D::periodic(int i) const {
    if (i >= L_) return i - L_;
    if (i < 0) return i + L_;
    return i;
}

int Ising2D::idx(int x, int y) const {
    return y * L_ + x;
}

void Ising2D::initialize_random() {
    std::uniform_int_distribution<int> spin_dist(0, 1);
    for (int i = 0; i < N_; ++i) {
        spins_[i] = spin_dist(rng_) ? 1 : -1;
    }
    compute_total_energy_and_magnetization();
}

void Ising2D::initialize_ordered(int spin_value) {
    const int s = (spin_value >= 0) ? 1 : -1;
    for (int i = 0; i < N_; ++i) {
        spins_[i] = s;
    }
    compute_total_energy_and_magnetization();
}

void Ising2D::compute_total_energy_and_magnetization() {
    magnetization_total_ = 0;
    energy_total_ = 0.0;

    for (int y = 0; y < L_; ++y) {
        for (int x = 0; x < L_; ++x) {
            const int s = spins_[idx(x, y)];
            magnetization_total_ += s;

            // Count each bond once: +x and +y neighbors only.
            const int sx = spins_[idx(periodic(x + 1), y)];
            const int sy = spins_[idx(x, periodic(y + 1))];
            energy_total_ += -(s * (sx + sy));
        }
    }
}

void Ising2D::set_temperature(double T) {
    beta_ = 1.0 / T;  // kB = 1, J = 1
    boltzmann_dE4_ = std::exp(-4.0 * beta_);
    boltzmann_dE8_ = std::exp(-8.0 * beta_);
}

void Ising2D::sweep_metropolis() {
    for (int color = 0; color < 2; ++color) {
        for (int y = 0; y < L_; ++y) {
            const int parity = (y & 1) ^ color;
            for (int x = parity; x < L_; x += 2) {
                const int p = idx(x, y);
                const int s = spins_[p];
                const int nn_sum = spins_[idx(periodic(x + 1), y)] +
                                   spins_[idx(periodic(x - 1), y)] +
                                   spins_[idx(x, periodic(y + 1))] +
                                   spins_[idx(x, periodic(y - 1))];

                const int dE = 2 * s * nn_sum;

                bool accept = false;
                if (dE <= 0) {
                    accept = true;
                } else if (dE == 4) {
                    accept = (unif01_(rng_) < boltzmann_dE4_);
                } else if (dE == 8) {
                    accept = (unif01_(rng_) < boltzmann_dE8_);
                }

                if (accept) {
                    spins_[p] = -s;
                    magnetization_total_ += -2 * s;
                    energy_total_ += dE;
                }
            }
        }
    }
}

double Ising2D::magnetization_per_spin() const {
    return static_cast<double>(magnetization_total_) / N_;
}

double Ising2D::energy_per_spin() const {
    return energy_total_ / N_;
}

void Ising2D::sweep_wolff() {
    std::vector<bool> visited(N_, false);
    const int seed_idx = site_dist_(rng_);
    const int seed_spin = spins_[seed_idx];
    std::queue<int> cluster_queue;
    std::vector<int> cluster;
    cluster_queue.push(seed_idx);
    visited[seed_idx] = true;
    cluster.push_back(seed_idx);
    const double wolff_prob = 1.0 - std::exp(-2.0 * beta_);
    while (!cluster_queue.empty()) {
        const int current_idx = cluster_queue.front();
        cluster_queue.pop();
        
        const int x = current_idx % L_;
        const int y = current_idx / L_;
        
        const int neighbors[4] = {
            idx(periodic(x + 1), y),
            idx(periodic(x - 1), y),
            idx(x, periodic(y + 1)),
            idx(x, periodic(y - 1))
        };
        
        for (int i = 0; i < 4; ++i) {
            const int neighbor_idx = neighbors[i];
            if (!visited[neighbor_idx] && spins_[neighbor_idx] == seed_spin) {
                visited[neighbor_idx] = true;
                if (unif01_(rng_) < wolff_prob) {
                    cluster.push_back(neighbor_idx);
                    cluster_queue.push(neighbor_idx);
                }
            }
        }
    }
    
    std::vector<char> in_cluster(N_, 0);
    for (int site_idx : cluster) {
        in_cluster[site_idx] = 1;
    }

    double delta_energy = 0.0;
    for (int site_idx : cluster) {
        const int x = site_idx % L_;
        const int y = site_idx / L_;
        const int s = spins_[site_idx];
        const int neighbors[4] = {
            idx(periodic(x + 1), y),
            idx(periodic(x - 1), y),
            idx(x, periodic(y + 1)),
            idx(x, periodic(y - 1))
        };

        for (int i = 0; i < 4; ++i) {
            const int neighbor_idx = neighbors[i];
            if (!in_cluster[neighbor_idx]) {
                delta_energy += 2.0 * static_cast<double>(s * spins_[neighbor_idx]);
            }
        }
    }
    for (int site_idx : cluster) {
        const int old_spin = spins_[site_idx];
        spins_[site_idx] = -old_spin;
        magnetization_total_ += -2 * old_spin;
    }
    energy_total_ += delta_energy;
}
