#pragma once
// Shared output helpers for the CPU, Wolff and CUDA drivers.
// Header-only so the nvcc build can use it without linking the CPU sources.

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// n_samples: measurements in the average; tau_absM: integrated autocorrelation
// time of |m| in units of measurements (n_eff ~ n_samples / (2 tau_absM)).
inline const char* kCsvHeader = "T,L,M,absM,E,M2,E2,M4,n_samples,tau_absM,algo,seed";

using RunKey = std::pair<int, long long>;  // (L, round(T * 1e6))

inline RunKey make_run_key(int L, double T) {
    return {L, std::llround(T * 1e6)};
}

// Integrated autocorrelation time with Sokal's automatic window (W >= c * tau).
// Correlations are computed lazily so the cost is O(N * W), not O(N^2).
struct TauEstimate {
    double tau_int = 0.5;
    int window = 0;
    bool converged = false;
};

inline TauEstimate integrated_autocorr_time(const std::vector<double>& x, double c = 6.0) {
    TauEstimate out;
    const int N = static_cast<int>(x.size());
    if (N < 4) return out;

    double mean = 0.0;
    for (double v : x) mean += v;
    mean /= N;

    double c0 = 0.0;
    for (double v : x) c0 += (v - mean) * (v - mean);
    c0 /= N;
    if (c0 <= 0.0) return out;

    const int max_lag = N / 4;
    double tau = 0.5;
    for (int t = 1; t <= max_lag; ++t) {
        double acc = 0.0;
        for (int s = 0; s + t < N; ++s) acc += (x[s] - mean) * (x[s + t] - mean);
        tau += acc / (N - t) / c0;
        if (t >= c * tau) {
            out.tau_int = tau;
            out.window = t;
            out.converged = true;
            return out;
        }
    }
    out.tau_int = tau;
    out.window = max_lag;
    return out;
}

// Reads an existing data CSV, requiring the current header and well-formed rows.
inline std::set<RunKey> read_existing_keys(const std::string& path) {
    std::set<RunKey> keys;
    std::ifstream in(path);
    if (!in.is_open()) return keys;

    std::string line;
    if (!std::getline(in, line)) return keys;
    if (line != kCsvHeader) {
        throw std::runtime_error(
            "Refusing to append to " + path + ": header is\n  " + line +
            "\nexpected\n  " + kCsvHeader + "\nRegenerate the file or write to a new --out.");
    }

    const int n_cols = 12;
    int line_no = 1;
    while (std::getline(in, line)) {
        ++line_no;
        if (line.empty()) continue;
        std::vector<std::string> cells;
        std::stringstream ss(line);
        std::string cell;
        while (std::getline(ss, cell, ',')) cells.push_back(cell);
        if (static_cast<int>(cells.size()) != n_cols) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) + ": expected " +
                                     std::to_string(n_cols) + " columns, got " +
                                     std::to_string(cells.size()));
        }
        const RunKey key = make_run_key(std::stoi(cells[1]), std::stod(cells[0]));
        if (!keys.insert(key).second) {
            throw std::runtime_error(path + ":" + std::to_string(line_no) +
                                     ": duplicate (T, L) row already in file");
        }
    }
    return keys;
}

// Opens the output CSV. Overwrites unless append is set; when appending, any
// planned (T, L) already present is an error, so mixed runs cannot pile up.
inline std::ofstream open_output_csv(
    const std::string& path,
    bool append,
    const std::vector<int>& sizes,
    const std::vector<double>& temps
) {
    std::ifstream probe(path);
    const bool exists = probe.good() && probe.peek() != std::ifstream::traits_type::eof();
    probe.close();

    std::ofstream out;
    if (append && exists) {
        const std::set<RunKey> existing = read_existing_keys(path);
        for (int L : sizes) {
            for (double T : temps) {
                if (existing.count(make_run_key(L, T))) {
                    std::ostringstream msg;
                    msg << "Refusing to append: (T=" << std::fixed << std::setprecision(6) << T
                        << ", L=" << L << ") already in " << path;
                    throw std::runtime_error(msg.str());
                }
            }
        }
        out.open(path, std::ios::app);
    } else {
        out.open(path, std::ios::trunc);
        if (out) out << kCsvHeader << '\n';
    }
    if (!out) throw std::runtime_error("Failed to open output file: " + path);
    return out;
}

inline void write_csv_row(
    std::ostream& out,
    double T, int L,
    double m, double abs_m, double e, double m2, double e2, double m4,
    int n_samples, double tau_abs_m,
    const std::string& algo, std::uint64_t seed
) {
    out << std::setprecision(10) << std::defaultfloat
        << T << ',' << L << ','
        << m << ',' << abs_m << ',' << e << ','
        << m2 << ',' << e2 << ',' << m4 << ','
        << n_samples << ',' << tau_abs_m << ','
        << algo << ',' << seed << '\n';
    out.flush();
}

inline std::string series_path(const std::string& dir, const std::string& algo, int L, double T) {
    std::ostringstream name;
    name << dir << "/series_" << algo << "_L" << L << "_T"
         << std::fixed << std::setprecision(6) << T << ".csv";
    return name.str();
}

// Called before any simulation so a clash fails fast instead of mid-run.
inline void require_series_paths_free(
    const std::string& dir, const std::string& algo,
    const std::vector<int>& sizes, const std::vector<double>& temps
) {
    if (dir.empty()) return;
    for (int L : sizes) {
        for (double T : temps) {
            const std::string path = series_path(dir, algo, L, T);
            std::ifstream probe(path);
            if (probe.good()) {
                throw std::runtime_error("Refusing to overwrite existing series file " + path);
            }
        }
    }
    std::ofstream test(dir + "/.write_test");
    if (!test) throw std::runtime_error("Series directory not writable (does it exist?): " + dir);
    test.close();
    std::remove((dir + "/.write_test").c_str());
}

// Raw per-measurement time series, for block bootstrap and reweighting.
// One file per (algo, L, T); see require_series_paths_free.
inline void write_series(
    const std::string& dir,
    const std::string& algo, int L, double T, std::uint64_t seed, int stride,
    const std::vector<double>& m, const std::vector<double>& e
) {
    const std::string path = series_path(dir, algo, L, T);
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Failed to open series file " + path);
    out << "# T=" << std::setprecision(17) << T << " L=" << L << " algo=" << algo
        << " seed=" << seed << " stride=" << stride << '\n';
    out << "m,e\n" << std::setprecision(10);
    for (size_t i = 0; i < m.size(); ++i) out << m[i] << ',' << e[i] << '\n';
}

// Uniform grid, or (adaptive) 0.02 steps with fine_step steps in [2.1, 2.4).
// Points are computed as start + i * step and always clipped to [t_min, t_max].
inline std::vector<double> build_temperature_grid(double t_min, double t_max, double t_step, bool adaptive,
                                                  double fine_step = 0.005) {
    std::vector<double> temps;
    auto add_range = [&](double lo, double hi, double step, bool include_hi) {
        for (long i = 0;; ++i) {
            const double t = lo + i * step;
            if (include_hi ? t > hi + 1e-9 : t >= hi - 1e-9) break;
            if (t >= t_min - 1e-9 && t <= t_max + 1e-9) temps.push_back(t);
        }
    };
    if (!adaptive) {
        add_range(t_min, t_max, t_step, true);
        return temps;
    }
    add_range(t_min, 2.1, 0.02, false);
    add_range(2.1, 2.4, fine_step, false);
    add_range(std::max(2.4, t_min), t_max, 0.02, true);
    return temps;
}
