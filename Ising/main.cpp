#include "ising.hpp"
#include "run_io.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        const SimulationConfig cfg = parse_args(argc, argv);
        const std::vector<double> temps = build_temperature_grid(
            cfg.t_min, cfg.t_max, cfg.t_step, cfg.adaptive_grid, cfg.fine_step);
        const std::string algo = "metropolis";

        std::cout << "2D Ising Metropolis simulation\n";
        if (cfg.append_mode && cfg.sizes.empty()) {
            std::cout << "  Append mode: All requested sizes already present in data file.\n";
            std::cout << "  Nothing to simulate.\n";
            return 0;
        }

        require_series_paths_free(cfg.series_dir, algo, cfg.sizes, temps);
        std::ofstream out = open_output_csv(cfg.output_csv, cfg.append_mode, cfg.sizes, temps);
        std::mt19937_64 rng(cfg.seed);

        if (cfg.append_mode) {
            std::cout << "  Append mode: Adding new sizes only\n";
        }
        std::cout << "  sizes: ";
        for (size_t i = 0; i < cfg.sizes.size(); ++i) {
            std::cout << cfg.sizes[i] << (i + 1 == cfg.sizes.size() ? '\n' : ',');
        }
        std::cout << "  T range: [" << cfg.t_min << ", " << cfg.t_max << "] ";
        if (cfg.adaptive_grid) {
            std::cout << "(adaptive grid, " << temps.size() << " points)\n";
        } else {
            std::cout << "step " << cfg.t_step << "\n";
        }
        std::cout << "  sweeps: therm(base)=" << cfg.thermal_sweeps
                  << " (scaled by (L/32)^2.17), meas=" << cfg.measurement_sweeps
                  << ", stride=" << cfg.sample_stride << " (fixed)\n"
                  << "  seed: " << cfg.seed << "\n";

        for (int L : cfg.sizes) {
            std::cout << "\nL=" << L << "\n";
            constexpr double scale_ref_L = 32.0;
            constexpr double scale_z = 2.17;
            double therm_scale = std::pow(L / scale_ref_L, scale_z);
            int therm_sweeps = std::lround(cfg.thermal_sweeps * therm_scale);
            therm_sweeps = std::max(cfg.thermal_sweeps, therm_sweeps);
            int stride = std::min(cfg.sample_stride, cfg.measurement_sweeps);
            if (therm_sweeps != cfg.thermal_sweeps) {
                std::cout << "  therm(L)=" << therm_sweeps << " (scaled from "
                          << cfg.thermal_sweeps << ")\n";
            }
            for (double T : temps) {
                Ising2D model(L, rng);
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

                std::cout << std::fixed << "  T=" << std::setprecision(3) << T
                          << "  <|M|>=" << std::setprecision(4) << obs.abs_m
                          << "  <E>=" << std::setprecision(4) << obs.e
                          << "  samples=" << acc.count
                          << "  tau(|M|)=" << std::setprecision(1) << tau.tau_int
                          << (tau.converged ? "" : " (window not converged)") << "\n"
                          << std::defaultfloat;
            }
        }

        std::cout << "\nSaved: " << cfg.output_csv << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
