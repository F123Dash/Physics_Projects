#include "ising.hpp"
#include "ising_cuda.cuh"
#include "run_io.hpp"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

// Options are parsed by the shared CPU parser (ising.cpp); see --help.

static void print_gpu_info() {
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
    try {
        const SimulationConfig cfg = parse_args(argc, argv);
        const std::vector<double> temps = build_temperature_grid(
            cfg.t_min, cfg.t_max, cfg.t_step, cfg.adaptive_grid, cfg.fine_step);
        for (int L : cfg.sizes) {
            if (L < 2 || (L & 1) != 0) {
                throw std::invalid_argument("CUDA checkerboard update needs even L >= 2, got L = " +
                                            std::to_string(L));
            }
        }

        print_gpu_info();
        const std::string algo = "metropolis_cuda";

        printf("2D Ising simulation (CUDA)\n");
        if (cfg.append_mode && cfg.sizes.empty()) {
            printf("  Append mode: All requested sizes already present in data file.\n");
            printf("  Nothing to simulate.\n");
            return 0;
        }

        require_series_paths_free(cfg.series_dir, algo, cfg.sizes, temps);
        std::ofstream out = open_output_csv(cfg.output_csv, cfg.append_mode, cfg.sizes, temps);

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
        printf("  sweeps: therm(base)=%d (scaled by (L/32)^2.17), meas=%d, stride=%d (fixed)\n",
               cfg.thermal_sweeps, cfg.measurement_sweeps, cfg.sample_stride);
        printf("  seed: %llu\n", static_cast<unsigned long long>(cfg.seed));

        for (int L : cfg.sizes) {
            printf("\nL=%d\n", L);

            const double scale_ref_L = 32.0;
            const double scale_z = 2.17;
            double therm_scale = std::pow(static_cast<double>(L) / scale_ref_L, scale_z);
            int therm_sweeps = static_cast<int>(std::lround(cfg.thermal_sweeps * therm_scale));
            if (therm_sweeps < cfg.thermal_sweeps) {
                therm_sweeps = cfg.thermal_sweeps;
            }
            int stride = cfg.sample_stride;
            if (stride > cfg.measurement_sweeps) {
                stride = cfg.measurement_sweeps;
            }
            if (therm_sweeps != cfg.thermal_sweeps) {
                printf("  therm(L)=%d (scaled from %d)\n", therm_sweeps, cfg.thermal_sweeps);
            }

            Ising2DCUDA model(L, cfg.seed + L * 1000);

            for (double T : temps) {
                // Cold start: a hot start below Tc can freeze into a striped
                // two-domain state that Metropolis does not escape on large L.
                model.initialize_ordered(1);
                model.set_temperature(T);

                for (int s = 0; s < therm_sweeps; ++s) {
                    model.sweep_metropolis();
                }

                SampleAccumulator acc;
                std::vector<double> ms, es, abs_ms;

                for (int s = 0; s < cfg.measurement_sweeps; ++s) {
                    model.sweep_metropolis();
                    if ((s + 1) % stride == 0) {
                        const double m = model.magnetization_per_spin();
                        const double e = model.energy_per_spin();
                        acc.add(m, e);
                        ms.push_back(m);
                        es.push_back(e);
                        abs_ms.push_back(std::fabs(m));
                    }
                }

                const AveragedObservables obs = finalize(acc);
                const TauEstimate tau = integrated_autocorr_time(abs_ms);
                write_csv_row(out, T, L, obs.m, obs.abs_m, obs.e, obs.m2, obs.e2, obs.m4,
                              acc.count, tau.tau_int, algo, cfg.seed);
                if (!cfg.series_dir.empty()) {
                    write_series(cfg.series_dir, algo, L, T, cfg.seed, stride, ms, es);
                }

                printf("  T=%.4f  <|M|>=%.4f  <E>=%.4f  samples=%d  tau(|M|)=%.1f%s\n",
                       T, obs.abs_m, obs.e, acc.count, tau.tau_int,
                       tau.converged ? "" : " (window not converged)");
            }
        }

        printf("\nSaved: %s\n", cfg.output_csv.c_str());
        return 0;
    } catch (const std::exception& ex) {
        fprintf(stderr, "Error: %s\n", ex.what());
        return 1;
    }
}
