#include "ising.hpp"

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

std::vector<int> get_existing_sizes(const std::string& filename) {
    std::set<int> sizes_set;
    std::ifstream file(filename);
    if (!file.is_open()) {
        // File does not exist or cannot be read, return empty
        return std::vector<int>();
    }
    
    std::string line;
    // Skip header
    if (!std::getline(file, line)) {
        return std::vector<int>();
    }
    
    // Read data lines and extract L values (column index 1)
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        int col = 0;
        while (std::getline(ss, cell, ',')) {
            if (col == 1) {
                try {
                    sizes_set.insert(std::stoi(cell));
                } catch (...) {
                    // Skip malformed lines
                }
                break;
            }
            col++;
        }
    }
    file.close();
    
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
            cfg.seed = static_cast<std::uint64_t>(std::stoull(arg.substr(7)));
        } else if (starts_with(arg, "--out=")) {
            cfg.output_csv = arg.substr(6);
        } else if (starts_with(arg, "--append=")) {
            cfg.sizes = parse_sizes_csv(arg.substr(9));
            cfg.append_mode = true;
        } else if (arg == "--append") {
            cfg.append_mode = true;
        } else if (arg == "--help" || arg == "-h") {
            std::cout
                << "Usage: ./ising2d [options]\n"
                << "  --sizes=32,48,64,96,128,160,192,256\n"
                << "  --tmin=1.8 --tmax=3.4 --dt=0.02\n"
                << "  --therm=10000 --meas=50000 --stride=10\n"
                << "  --seed=123456789 --out=./data_outputs/data.csv\n"
                << "  --append              Append with default sizes (skip existing)\n"
                << "  --append=512,768      Append only specific sizes\n";
            std::exit(0);
        } else {
            throw std::runtime_error("Unknown argument: " + arg);
        }
    }

    if (cfg.t_step <= 0.0 || cfg.t_max < cfg.t_min) {
        throw std::runtime_error("Invalid temperature range parameters.");
    }
    if (cfg.thermal_sweeps < 0 || cfg.measurement_sweeps <= 0 || cfg.sample_stride <= 0) {
        throw std::runtime_error("Sweep and stride values must be positive (thermal can be zero).\n");
    }

    // Filter out existing sizes if in append mode
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

std::vector<double> make_temperature_grid(double t_min, double t_max, double t_step) {
    std::vector<double> temps;
    for (double t = t_min; t <= t_max + 1e-12; t += t_step) {
        temps.push_back(t);
    }
    return temps;
}

static std::vector<double> make_uniform_grid(double t_min, double t_max, double t_step) {
    std::vector<double> temps;
    for (double t = t_min; t <= t_max + 1e-12; t += t_step) {
        temps.push_back(t);
    }
    return temps;
}

std::vector<double> make_temperature_grid(double t_min, double t_max, double t_step, bool adaptive) {
    if (!adaptive) {
        return make_uniform_grid(t_min, t_max, t_step);
    }

    // Adaptive grid: finer resolution near critical temperature
    // Region 1: 1.8 to 2.1, step 0.02
    // Region 2: 2.1 to 2.4, step 0.005 (critical region)
    // Region 3: 2.4 to 3.4, step 0.02
    std::vector<double> temps;

    // Region 1: coarse
    for (double t = t_min; t < 2.1 - 1e-9; t += 0.02) {
        temps.push_back(t);
    }

    // Region 2: fine (critical region)
    for (double t = 2.1; t < 2.4 - 1e-9; t += 0.005) {
        temps.push_back(t);
    }

    // Region 3: coarse
    for (double t = 2.4; t <= t_max + 1e-9; t += 0.02) {
        temps.push_back(t);
    }

    return temps;
}
