#include "ising.hpp"

#include <cmath>

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
            energy_total_ += -static_cast<double>(s * (sx + sy));
        }
    }
}

void Ising2D::set_temperature(double T) {
    beta_ = 1.0 / T;  // kB = 1, J = 1
    boltzmann_dE4_ = std::exp(-4.0 * beta_);
    boltzmann_dE8_ = std::exp(-8.0 * beta_);
}

void Ising2D::sweep_metropolis() {
    for (int trial = 0; trial < N_; ++trial) {
        const int p = site_dist_(rng_);
        const int x = p % L_;
        const int y = p / L_;

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
            energy_total_ += static_cast<double>(dE);
        }
    }
}

double Ising2D::magnetization_per_spin() const {
    return static_cast<double>(magnetization_total_) / static_cast<double>(N_);
}

double Ising2D::energy_per_spin() const {
    return energy_total_ / static_cast<double>(N_);
}
