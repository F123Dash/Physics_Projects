#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

struct SimulationConfig {
    std::vector<int> sizes{32, 48, 64, 96, 128, 160, 192, 256};
    double t_min = 1.8;
    double t_max = 3.4;
    double t_step = 0.02;
    int thermal_sweeps = 30000;
    int measurement_sweeps = 200000;
    int sample_stride = 50;
    bool adaptive_grid = true;
    std::uint64_t seed = 123456789ULL;
    std::string output_csv = "./data_outputs/data.csv";
};

struct SampleAccumulator {
    double sum_m = 0.0;
    double sum_abs_m = 0.0;
    double sum_e = 0.0;
    double sum_m2 = 0.0;
    double sum_e2 = 0.0;
    double sum_m4 = 0.0;
    int count = 0;

    void add(double m_per_spin, double e_per_spin);
};

struct AveragedObservables {
    double m = 0.0;
    double abs_m = 0.0;
    double e = 0.0;
    double m2 = 0.0;
    double e2 = 0.0;
    double m4 = 0.0;
};

AveragedObservables finalize(const SampleAccumulator& acc);

std::vector<double> make_temperature_grid(double t_min, double t_max, double t_step, bool adaptive = false);

class Ising2D {
public:
    Ising2D(int L, std::mt19937_64& rng);

    void initialize_random();
    void initialize_ordered(int spin_value = 1);
    void set_temperature(double T);
    void sweep_metropolis();

    double magnetization_per_spin() const;
    double energy_per_spin() const;

private:
    int idx(int x, int y) const;
    int periodic(int i) const;
    void compute_total_energy_and_magnetization();

    int L_;
    int N_;
    std::vector<int> spins_;

    std::mt19937_64& rng_;
    std::uniform_real_distribution<double> unif01_;
    std::uniform_int_distribution<int> site_dist_;

    double beta_ = 1.0;
    double boltzmann_dE4_ = 1.0;
    double boltzmann_dE8_ = 1.0;

    long long magnetization_total_ = 0;
    double energy_total_ = 0.0;
};

SimulationConfig parse_args(int argc, char** argv);
std::vector<double> make_temperature_grid(double t_min, double t_max, double t_step);
