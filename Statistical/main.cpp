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

        // Check if file exists for header logic
        std::ifstream infile(cfg.output_csv);
        bool file_exists = infile.good();
        infile.close();

        std::mt19937_64 rng(cfg.seed);

        // Open in append mode if configured, otherwise truncate
        std::ofstream out;
        if (cfg.append_mode && file_exists) {
            out.open(cfg.output_csv, std::ios_base::app);
        } else {
            out.open(cfg.output_csv);
        }
        
        if (!out) {
            throw std::runtime_error("Failed to open output file: " + cfg.output_csv);
        }

        // Only write header if file is new or not appending
        if (!cfg.append_mode || !file_exists) {
            out << "T,L,M,absM,E,M2,E2,M4\n";
        }
        out << std::fixed << std::setprecision(8);

        std::cout << "2D Ising Metropolis simulation\n";
        if (cfg.append_mode && file_exists && cfg.sizes.empty()) {
            std::cout << "  Append mode: All requested sizes already present in data file.\n";
            std::cout << "  Nothing to simulate.\n";
            out.close();
            std::cout << "Exiting.\n";
            return 0;
        }
        if (cfg.append_mode) {
            std::cout << "  Append mode: Adding new sizes only\n";
        }
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
