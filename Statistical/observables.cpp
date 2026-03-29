#include "ising.hpp"

#include <stdexcept>

void SampleAccumulator::add(double m_per_spin, double e_per_spin) {
    sum_m += m_per_spin;
    sum_abs_m += (m_per_spin >= 0.0 ? m_per_spin : -m_per_spin);
    sum_e += e_per_spin;
    sum_m2 += m_per_spin * m_per_spin;
    sum_e2 += e_per_spin * e_per_spin;
    double m2 = m_per_spin * m_per_spin;
    sum_m4 += m2 * m2;
    ++count;
}

AveragedObservables finalize(const SampleAccumulator& acc) {
    if (acc.count <= 0) {
        throw std::runtime_error("No samples collected; increase measurement sweeps or decrease sample stride.");
    }

    const double inv = 1.0 / static_cast<double>(acc.count);
    AveragedObservables out;
    out.m = acc.sum_m * inv;
    out.abs_m = acc.sum_abs_m * inv;
    out.e = acc.sum_e * inv;
    out.m2 = acc.sum_m2 * inv;
    out.e2 = acc.sum_e2 * inv;
    out.m4 = acc.sum_m4 * inv;
    return out;
}
