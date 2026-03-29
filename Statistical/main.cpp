#include "ising.hpp"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    try {
        const SimulationConfig cfg = parse_args(argc, argv);
        const std::vector<double> temps = make_temperature_grid(
            cfg.t_min, cfg.t_max, cfg.t_step, cfg.adaptive_grid);

        std::mt19937_64 rng(cfg.seed);

        std::ofstream out(cfg.output_csv);
        if (!out) {
            throw std::runtime_error("Failed to open output file: " + cfg.output_csv);
        }

        out << "T,L,M,absM,E,M2,E2,M4\n";
        out << std::fixed << std::setprecision(8);

        std::cout << "2D Ising Metropolis simulation\n";
        std::cout << "  sizes: ";
        for (size_t i = 0; i < cfg.sizes.size(); ++i) {
            std::cout << cfg.sizes[i] << (i + 1 == cfg.sizes.size() ? '\n' : ',');
        }
        std::cout << "  T range: [" << cfg.t_min << ", " << cfg.t_max << "] ";
        if (cfg.adaptive_grid) {
            std::cout << "(adaptive grid)\n";
        } else {
            std::cout << "step " << cfg.t_step << "\n";
        }
        std::cout << "  sweeps: therm=" << cfg.thermal_sweeps
                  << ", meas=" << cfg.measurement_sweeps
                  << ", stride=" << cfg.sample_stride << "\n";

        for (int L : cfg.sizes) {
            std::cout << "\nL=" << L << "\n";
            for (double T : temps) {
                // CRITICAL FIX: Reinitialize state at each temperature
                Ising2D model(L, rng);
                model.initialize_random();
                model.set_temperature(T);

                for (int s = 0; s < cfg.thermal_sweeps; ++s) {
                    model.sweep_metropolis();
                }

                SampleAccumulator acc;
                for (int s = 0; s < cfg.measurement_sweeps; ++s) {
                    model.sweep_metropolis();
                    if ((s + 1) % cfg.sample_stride == 0) {
                        acc.add(model.magnetization_per_spin(), model.energy_per_spin());
                    }
                }

                const AveragedObservables obs = finalize(acc);
                out << T << ',' << L << ','
                    << obs.m << ',' << obs.abs_m << ',' << obs.e << ','
                    << obs.m2 << ',' << obs.e2 << ',' << obs.m4 << '\n';

                std::cout << "  T=" << std::setprecision(3) << T
                          << "  <|M|>=" << std::setprecision(4) << obs.abs_m
                          << "  <E>=" << std::setprecision(4) << obs.e
                          << "  samples=" << acc.count << "\n"
                          << std::setprecision(8);
            }
        }

        std::cout << "\nSaved: " << cfg.output_csv << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "Error: " << ex.what() << "\n";
        return 1;
    }
}
