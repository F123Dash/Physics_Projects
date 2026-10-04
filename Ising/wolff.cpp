#include "ising.hpp"
#include "run_io.hpp"

#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <vector>

// Note: for this driver --therm, --meas and --stride count single Wolff cluster
// flips, not lattice sweeps.
int main(int argc, char** argv) {
    try {
        const SimulationConfig cfg = parse_args(argc, argv);
        const std::vector<double> temps = build_temperature_grid(
            cfg.t_min, cfg.t_max, cfg.t_step, cfg.adaptive_grid, cfg.fine_step);
        const std::string algo = "wolff";

        std::cerr << "Wolff Algorithm - 2D Ising Model\n";
        if (cfg.append_mode && cfg.sizes.empty()) {
            std::cerr << "Append mode: all requested sizes already present. Nothing to simulate.\n";
            return 0;
        }

        require_series_paths_free(cfg.series_dir, algo, cfg.sizes, temps);
        std::ofstream outfile = open_output_csv(cfg.output_csv, cfg.append_mode, cfg.sizes, temps);
        std::mt19937_64 rng(cfg.seed);

        std::cerr << "System sizes: ";
        for (int L : cfg.sizes) std::cerr << L << " ";
        std::cerr << "\n";
        std::cerr << "Temperatures: " << temps.size() << " in [" << cfg.t_min << ", " << cfg.t_max << "]"
                  << (cfg.adaptive_grid ? " (adaptive grid)" : "") << "\n";
        std::cerr << "Thermal cluster flips: " << cfg.thermal_sweeps << "\n";
        std::cerr << "Measurement cluster flips: " << cfg.measurement_sweeps << "\n";
        std::cerr << "Sample stride (cluster flips): " << cfg.sample_stride << "\n";
        std::cerr << "Seed: " << cfg.seed << "\n";
        std::cerr << "Output: " << cfg.output_csv << (cfg.append_mode ? " (append)" : "") << "\n\n";

        for (int L : cfg.sizes) {
            std::cerr << "System size L = " << L << "\n";

            for (double T : temps) {
                Ising2D ising(L, rng);
                ising.initialize_ordered(1);
                ising.set_temperature(T);

                for (int flip = 0; flip < cfg.thermal_sweeps; ++flip) {
                    ising.sweep_wolff();
                }

                SampleAccumulator acc;
                std::vector<double> ms, es, abs_ms;
                for (int flip = 0; flip < cfg.measurement_sweeps; ++flip) {
                    ising.sweep_wolff();
                    if ((flip + 1) % cfg.sample_stride == 0) {
                        const double m = ising.magnetization_per_spin();
                        const double e = ising.energy_per_spin();
                        acc.add(m, e);
                        ms.push_back(m);
                        es.push_back(e);
                        abs_ms.push_back(std::fabs(m));
                    }
                }

                const AveragedObservables obs = finalize(acc);
                const TauEstimate tau = integrated_autocorr_time(abs_ms);
                write_csv_row(outfile, T, L, obs.m, obs.abs_m, obs.e, obs.m2, obs.e2, obs.m4,
                              acc.count, tau.tau_int, algo, cfg.seed);
                if (!cfg.series_dir.empty()) {
                    write_series(cfg.series_dir, algo, L, T, cfg.seed, cfg.sample_stride, ms, es);
                }

                std::cerr << "  T = " << std::fixed << std::setprecision(4) << T
                          << ": |M| = " << obs.abs_m
                          << ", E = " << obs.e
                          << ", tau(|M|) = " << std::setprecision(1) << tau.tau_int
                          << (tau.converged ? "" : " (window not converged)") << "\n";
            }

            std::cerr << "\n";
        }

        std::cerr << "Simulation completed. Results saved to " << cfg.output_csv << "\n";
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
}
