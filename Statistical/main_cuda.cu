#include "ising_cuda.cuh"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <set>

// Helper functions for argument parsing
static bool starts_with(const char* s, const char* pref) {
    return strncmp(s, pref, strlen(pref)) == 0;
}

std::vector<int> get_existing_sizes(const std::string& filename);

static std::vector<int> parse_sizes_csv(const char* csv) {
    std::vector<int> out;
    std::stringstream ss(csv);
    std::string token;
    while (std::getline(ss, token, ',')) {
        if (!token.empty()) {
            out.push_back(std::stoi(token));
        }
    }
    if (out.empty()) {
        fprintf(stderr, "Error: --sizes requires at least one integer.\n");
        exit(1);
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

struct Config {
    std::vector<int> sizes{32, 48, 64, 96, 128, 160, 192, 256};
    double t_min = 1.8;
    double t_max = 3.4;
    double t_step = 0.02;
    int thermal_sweeps = 30000;
    int measurement_sweeps = 200000;
    int sample_stride = 50;
    bool adaptive_grid = true;
    bool append_mode = false;
    uint64_t seed = 123456789ULL;
    std::string output_csv = "./data_outputs/data.csv";
};

Config parse_args(int argc, char** argv) {
    Config cfg;

    for (int i = 1; i < argc; ++i) {
        const char* arg = argv[i];

        if (starts_with(arg, "--sizes=")) {
            cfg.sizes = parse_sizes_csv(arg + 8);
        } else if (starts_with(arg, "--tmin=")) {
            cfg.t_min = atof(arg + 7);
        } else if (starts_with(arg, "--tmax=")) {
            cfg.t_max = atof(arg + 7);
        } else if (starts_with(arg, "--dt=")) {
            cfg.t_step = atof(arg + 5);
        } else if (starts_with(arg, "--therm=")) {
            cfg.thermal_sweeps = atoi(arg + 8);
        } else if (starts_with(arg, "--meas=")) {
            cfg.measurement_sweeps = atoi(arg + 7);
        } else if (starts_with(arg, "--stride=")) {
            cfg.sample_stride = atoi(arg + 9);
        } else if (starts_with(arg, "--seed=")) {
            cfg.seed = static_cast<uint64_t>(strtoull(arg + 7, nullptr, 10));
        } else if (starts_with(arg, "--out=")) {
            cfg.output_csv = arg + 6;
        } else if (starts_with(arg, "--no-adaptive")) {
            cfg.adaptive_grid = false;
        } else if (starts_with(arg, "--append=")) {
            cfg.sizes = parse_sizes_csv(arg + 9);
            cfg.append_mode = true;
        } else if (starts_with(arg, "--append")) {
            cfg.append_mode = true;
        } else if (strcmp(arg, "--help") == 0 || strcmp(arg, "-h") == 0) {
            printf("Usage: ./ising2d_cuda [options]\n");
            printf("  --sizes=32,48,64,96,128,160,192,256   Lattice sizes to simulate\n");
            printf("  --tmin=1.8             Minimum temperature\n");
            printf("  --tmax=3.4             Maximum temperature\n");
            printf("  --dt=0.02              Temperature step (non-adaptive)\n");
            printf("  --therm=30000          Thermalization sweeps\n");
            printf("  --meas=200000          Measurement sweeps\n");
            printf("  --stride=50            Sample every N sweeps\n");
            printf("  --seed=123456789       Random seed\n");
            printf("  --out=./data_outputs/data.csv   Output file\n");
            printf("  --no-adaptive          Use uniform temperature grid\n");
            printf("  --append               Append with default sizes (skip existing)\n");
            printf("  --append=512,768       Append only specific sizes\n");
            exit(0);
        } else {
            fprintf(stderr, "Unknown argument: %s\n", arg);
            exit(1);
        }
    }

    if (cfg.t_step <= 0.0 || cfg.t_max < cfg.t_min) {
        fprintf(stderr, "Error: Invalid temperature range parameters.\n");
        exit(1);
    }
    if (cfg.thermal_sweeps < 0 || cfg.measurement_sweeps <= 0 || cfg.sample_stride <= 0) {
        fprintf(stderr, "Error: Sweep and stride values must be positive.\n");
        exit(1);
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

std::vector<double> make_temperature_grid(double t_min, double t_max, double t_step, bool adaptive) {
    std::vector<double> temps;

    if (!adaptive) {
        for (double t = t_min; t <= t_max + 1e-12; t += t_step) {
            temps.push_back(t);
        }
        return temps;
    }

    // Adaptive grid: finer resolution near critical temperature Tc ~ 2.269
    // Region 1: coarse (t_min to 2.1)
    for (double t = t_min; t < 2.1 - 1e-9; t += 0.02) {
        temps.push_back(t);
    }

    // Region 2: fine (2.1 to 2.4, critical region)
    for (double t = 2.1; t < 2.4 - 1e-9; t += 0.005) {
        temps.push_back(t);
    }

    // Region 3: coarse (2.4 to t_max)
    for (double t = 2.4; t <= t_max + 1e-9; t += 0.02) {
        temps.push_back(t);
    }

    return temps;
}

void print_gpu_info() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("GPU: %s\n", prop.name);
    printf("  Compute capability: %d.%d\n", prop.major, prop.minor);
    printf("  Memory: %.1f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("  SM count: %d\n", prop.multiProcessorCount);
    printf("\n");
}

int main(int argc, char** argv) {
    Config cfg = parse_args(argc, argv);
    std::vector<double> temps = make_temperature_grid(
        cfg.t_min, cfg.t_max, cfg.t_step, cfg.adaptive_grid);

    // Check if file exists for header logic
    std::ifstream infile(cfg.output_csv);
    bool file_exists = infile.good();
    infile.close();

    // Print GPU info
    print_gpu_info();

    // Open output file in append or truncate mode
    std::ofstream out;
    if (cfg.append_mode && file_exists) {
        out.open(cfg.output_csv, std::ios_base::app);
    } else {
        out.open(cfg.output_csv);
    }
    
    if (!out) {
        fprintf(stderr, "Error: Failed to open output file: %s\n", cfg.output_csv.c_str());
        return 1;
    }

    // Only write header if file is new or not appending
    if (!cfg.append_mode || !file_exists) {
        out << "T,L,M,absM,E,M2,E2,M4\n";
    }
    out << std::fixed << std::setprecision(8);

    printf("2D Ising Metropolis simulation (CUDA)\n");
    if (cfg.append_mode && file_exists && cfg.sizes.empty()) {
        printf("  Append mode: All requested sizes already present in data file.\n");
        printf("  Nothing to simulate.\n");
        out.close();
        printf("Exiting.\n");
        return 0;
    }
    if (cfg.append_mode) {
        printf("  Append mode: Adding new sizes only\n");
    }
    printf("  sizes: ");
    for (size_t i = 0; i < cfg.sizes.size(); ++i) {
        printf("%d%c", cfg.sizes[i], (i + 1 == cfg.sizes.size()) ? '\n' : ',');
    }
    printf("  T range: [%.2f, %.2f] ", cfg.t_min, cfg.t_max);
    if (cfg.adaptive_grid) {
        printf("(adaptive grid, %zu points)\n", temps.size());
    } else {
        printf("step %.3f\n", cfg.t_step);
    }
    printf("  sweeps: therm=%d, meas=%d, stride=%d\n",
           cfg.thermal_sweeps, cfg.measurement_sweeps, cfg.sample_stride);
    printf("  seed: %lu\n", cfg.seed);

    // Main simulation loop
    for (int L : cfg.sizes) {
        printf("\nL=%d\n", L);

        for (double T : temps) {
            // Create model for this (L, T) point
            Ising2DCUDA model(L, cfg.seed + static_cast<uint64_t>(L * 1000 + static_cast<int>(T * 1000)));
            model.initialize_random();
            model.set_temperature(T);

            // Thermalization
            for (int s = 0; s < cfg.thermal_sweeps; ++s) {
                model.sweep_metropolis();
            }

            // Measurement
            SampleAccumulatorCuda acc;
            acc.reset();

            for (int s = 0; s < cfg.measurement_sweeps; ++s) {
                model.sweep_metropolis();
                if ((s + 1) % cfg.sample_stride == 0) {
                    acc.add(model.magnetization_per_spin(), model.energy_per_spin());
                }
            }

            AveragedObservablesCuda obs = finalize_cuda(acc);
            out << T << ',' << L << ','
                << obs.m << ',' << obs.abs_m << ',' << obs.e << ','
                << obs.m2 << ',' << obs.e2 << ',' << obs.m4 << '\n';

            printf("  T=%.3f  <|M|>=%.4f  <E>=%.4f  samples=%d\n",
                   T, obs.abs_m, obs.e, acc.count);
        }
    }

    out.close();
    printf("\nSaved: %s\n", cfg.output_csv.c_str());

    return 0;
}
