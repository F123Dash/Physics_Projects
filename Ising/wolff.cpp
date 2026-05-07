//Wolff Cluster Algorithm Implementation for 2D Ising Model

#include "ising.hpp"

#include <fstream>
#include <iostream>
#include <iomanip>
#include <random>
#include <vector>

int main(int argc, char** argv) {
    // Parse command-line arguments
    SimulationConfig cfg;
    try {
        cfg = parse_args(argc, argv);
    } catch (const std::exception& e) {
        std::cerr << "Error parsing arguments: " << e.what() << "\n";
        return 1;
    }

    // Seed the random number generator
    std::mt19937_64 rng(cfg.seed);

    // Get temperature grid
    std::vector<double> temps = make_temperature_grid(
        cfg.t_min, cfg.t_max, cfg.t_step, cfg.adaptive_grid
    );

    std::cerr << "Wolff Algorithm - 2D Ising Model\n";
    std::cerr << "System sizes: ";
    for (int L : cfg.sizes) std::cerr << L << " ";
    std::cerr << "\n";
    std::cerr << "Temperature range: [" << cfg.t_min << ", " << cfg.t_max << "], step = " << cfg.t_step << "\n";
    std::cerr << "Thermal sweeps: " << cfg.thermal_sweeps << "\n";
    std::cerr << "Measurement sweeps: " << cfg.measurement_sweeps << "\n";
    std::cerr << "Sample stride: " << cfg.sample_stride << "\n";
    std::cerr << "Output: " << cfg.output_csv << "\n\n";

    // Open output file
    std::ofstream outfile(cfg.output_csv, std::ios::app);
    if (!outfile.is_open()) {
        std::cerr << "Error: Cannot open output file " << cfg.output_csv << "\n";
        return 1;
    }

    // Write header if file is empty or new
    outfile.seekp(0, std::ios::end);
    if (outfile.tellp() == 0) {
        outfile << "T,L,M,|M|,E,M^2,E^2,M^4,U4\n";
    }

    // Simulation loop over all system sizes and temperatures
    for (int L : cfg.sizes) {
        std::cerr << "System size L = " << L << "\n";

        for (double T : temps) {
            // Create Ising2D system
            Ising2D ising(L, rng);
            ising.initialize_random();
            ising.set_temperature(T);

            // Thermalization with Wolff sweeps
            for (int sweep = 0; sweep < cfg.thermal_sweeps; ++sweep) {
                ising.sweep_wolff();
            }

            // Measurement phase
            SampleAccumulator acc;
            for (int sweep = 0; sweep < cfg.measurement_sweeps; ++sweep) {
                ising.sweep_wolff();

                // Sample every stride sweeps
                if (sweep % cfg.sample_stride == 0) {
                    double m = ising.magnetization_per_spin();
                    double e = ising.energy_per_spin();
                    acc.add(m, e);
                }
            }

            // Finalize statistics and write to file
            AveragedObservables obs = finalize(acc);
            
            // Calculate Binder cumulant U4 = 1 - <M^4> / (3 * <M^2>^2)
            double u4 = 1.0;
            if (obs.m2 > 1e-12) {
                u4 = 1.0 - obs.m4 / (3.0 * obs.m2 * obs.m2);
            }

            outfile << std::fixed << std::setprecision(6)
                    << T << ","
                    << L << ","
                    << obs.m << ","
                    << obs.abs_m << ","
                    << obs.e << ","
                    << obs.m2 << ","
                    << obs.e2 << ","
                    << obs.m4 << ","
                    << u4 << "\n";

            std::cerr << "  T = " << std::fixed << std::setprecision(3) << T
                      << ": |M| = " << std::setprecision(4) << obs.abs_m
                      << ", E = " << obs.e << "\n";
        }

        std::cerr << "\n";
    }

    outfile.close();
    std::cerr << "Simulation completed. Results saved to " << cfg.output_csv << "\n";
    return 0;
}
