#pragma once

#include <cstdint>
#include <random>
#include <string>
#include <vector>

struct SimulationConfig {
    std::vector<int> sizes{32, 48, 64, 96, 128, 160, 192, 256,512,1024};
    double t_min = 1.8;
    double t_max = 3.4;
    double t_step = 0.02;
    int thermal_sweeps = 30000;
    int measurement_sweeps = 200000;
    int sample_stride = 50;
    bool adaptive_grid = true;
    bool append_mode = false;
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

struct AutocorrResult {
    double tau_int = 0.0;
    double tau_stderr = 0.0;
    int window = 0;
    bool converged = false;
};

AutocorrResult measure_autocorrelation(
    const std::vector<double>& timeseries,
    int max_lag = -1
);

struct RawSample {
    double m = 0.0;
    double abs_m = 0.0;
    double e = 0.0;
};

struct JackknifeObservables {
    double chi = 0.0;
    double chi_err = 0.0;
    double C = 0.0;
    double C_err = 0.0;
    double U = 0.0;
    double U_err = 0.0;
    double abs_m = 0.0;
    double abs_m_err = 0.0;
    double e = 0.0;
    double e_err = 0.0;
    int n_samples = 0;
};

JackknifeObservables jackknife_observables(
    const std::vector<RawSample>& samples,
    int L,
    double T
);

std::vector<double> make_temperature_grid(double t_min, double t_max, double t_step, bool adaptive = false);

class Ising2D {
public:
    Ising2D(int L, std::mt19937_64& rng);

    void initialize_random();
    void initialize_ordered(int spin_value = 1);
    void set_temperature(double T);
    void sweep_metropolis();
    void sweep_wolff();

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

int compute_adaptive_stride(
    Ising2D& model,
    int calibration_sweeps,
    int min_stride,
    int max_stride
);

SimulationConfig parse_args(int argc, char** argv);

std::vector<int> get_existing_sizes(const std::string& filename);
