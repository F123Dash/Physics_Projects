#include "ising.hpp"
#include "run_io.hpp"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>

static bool starts_with(const std::string& s, const std::string& pref) {
    return s.rfind(pref, 0) == 0;
}

static std::vector<int> parse_sizes_csv(const std::string& csv) {
    std::vector<int> out;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (token.empty()) continue;
        out.push_back(std::stoi(token));
    }
    if (out.empty()) {
        throw std::runtime_error("--sizes requires at least one integer.");
    }
    return out;
}

static std::vector<int> get_existing_sizes(const std::string& filename) {
    // Validates the header and every row; throws on an old-format or malformed file.
    std::set<int> sizes_set;
    for (const RunKey& key : read_existing_keys(filename)) sizes_set.insert(key.first);
    return std::vector<int>(sizes_set.begin(), sizes_set.end());
}

SimulationConfig parse_args(int argc, char** argv) {
    SimulationConfig cfg;

    for (int i = 1; i < argc; ++i) {
        std::string arg(argv[i]);

        if (starts_with(arg, "--sizes=")) {
            cfg.sizes = parse_sizes_csv(arg.substr(8));
        } else if (starts_with(arg, "--tmin=")) {
            cfg.t_min = std::stod(arg.substr(7));
        } else if (starts_with(arg, "--tmax=")) {
            cfg.t_max = std::stod(arg.substr(7));
        } else if (starts_with(arg, "--dt=")) {
            cfg.t_step = std::stod(arg.substr(5));
        } else if (starts_with(arg, "--therm=")) {
            cfg.thermal_sweeps = std::stoi(arg.substr(8));
        } else if (starts_with(arg, "--meas=")) {
            cfg.measurement_sweeps = std::stoi(arg.substr(7));
        } else if (starts_with(arg, "--stride=")) {
            cfg.sample_stride = std::stoi(arg.substr(9));
        } else if (starts_with(arg, "--seed=")) {
            cfg.seed = std::stoull(arg.substr(7));
        } else if (starts_with(arg, "--out=")) {
            cfg.output_csv = arg.substr(6);
        } else if (starts_with(arg, "--append=")) {
            cfg.sizes = parse_sizes_csv(arg.substr(9));
            cfg.append_mode = true;
        } else if (starts_with(arg, "--fine-dt=")) {
            cfg.fine_step = std::stod(arg.substr(10));
        } else if (arg == "--no-adaptive") {
            cfg.adaptive_grid = false;
        } else if (starts_with(arg, "--series-dir=")) {
            cfg.series_dir = arg.substr(13);
        } else if (arg == "--append") {
            cfg.append_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: ./ising2d [options]\n"
                << "  --sizes=32,48,64,96,128,160,192,256\n"
                << "  --tmin=1.8 --tmax=3.4 --dt=0.02\n"
                << "  --therm=10000 --meas=50000 --stride=10\n"
                << "  --seed=123456789 --out=./data_outputs/data.csv\n"
                << "  --no-adaptive         Uniform T grid (use --tmin=--tmax for a single T)\n"
                << "  --fine-dt=0.005       Adaptive-grid step in [2.1, 2.4) (0.02 -> 15 points)\n"
                << "  --series-dir=DIR      Also write raw (m, e) time series per (T, L) to DIR\n"
                << "  --append              Append with default sizes (skip existing)\n"
                << "  --append=512,768      Append only specific sizes\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.t_step <= 0.0 || cfg.fine_step <= 0.0 || cfg.t_max < cfg.t_min) {
        throw std::runtime_error("Invalid temperature range parameters.");
    }
    if (cfg.thermal_sweeps < 0 || cfg.measurement_sweeps <= 0 || cfg.sample_stride <= 0) {
        throw std::runtime_error("Sweep and stride values must be positive (thermal can be zero).\n");
    }

    if (cfg.append_mode) {
        std::vector<int> existing = get_existing_sizes(cfg.output_csv);
        std::vector<int> new_sizes;
        for (int L : cfg.sizes) {
            bool found = false;
            for (int existing_L : existing) {
                if (L == existing_L) {
                    found = true;
                    break;
                }
            }
            if (!found) {
                new_sizes.push_back(L);
            }
        }
        cfg.sizes = new_sizes;
    }

    return cfg;
}
