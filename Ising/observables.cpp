#include "ising.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
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

    const double inv = 1.0 / acc.count;
    AveragedObservables out;
    out.m = acc.sum_m * inv;
    out.abs_m = acc.sum_abs_m * inv;
    out.e = acc.sum_e * inv;
    out.m2 = acc.sum_m2 * inv;
    out.e2 = acc.sum_e2 * inv;
    out.m4 = acc.sum_m4 * inv;
    return out;
}

AutocorrResult measure_autocorrelation(const std::vector<double>& timeseries, int max_lag) {
    AutocorrResult result;
    const int N = static_cast<int>(timeseries.size());
    if (N < 2) {
        result.tau_int = 0.5;
        result.tau_stderr = 0.0;
        result.window = 0;
        result.converged = false;
        return result;
    }

    if (max_lag < 0) {
        max_lag = N / 4;
    }
    max_lag = std::min(max_lag, N - 1);
    if (max_lag < 1) {
        result.tau_int = 0.5;
        result.window = max_lag;
        result.converged = false;
        result.tau_stderr = std::sqrt(2.0 * (2.0 * result.window + 1.0) / N) * result.tau_int;
        return result;
    }

    double mean = 0.0;
    for (double v : timeseries) {
        mean += v;
    }
    mean /= N;

    std::vector<double> C(max_lag + 1, 0.0);
    for (int t = 0; t <= max_lag; ++t) {
        double accum = 0.0;
        const int limit = N - t;
        for (int s = 0; s < limit; ++s) {
            accum += timeseries[s] * timeseries[s + t];
        }
        C[t] = accum / limit - mean * mean;
    }

    const double C0 = C[0];
    if (C0 <= 0.0) {
        result.tau_int = 0.5;
        result.window = 0;
        result.converged = false;
        result.tau_stderr = std::sqrt(2.0 / N) * result.tau_int;
        return result;
    }

    double tau_int = 0.5;
    int window = max_lag;
    bool converged = false;

    for (int W = 1; W <= max_lag; ++W) {
        double sum_corr = 0.0;
        for (int t = 1; t <= W; ++t) {
            sum_corr += C[t];
        }
        tau_int = 0.5 + sum_corr / C0;
        if (static_cast<double>(W) >= 6.0 * tau_int) {
            window = W;
            converged = true;
            break;
        }
    }

    if (!converged) {
        window = max_lag;
        double sum_corr = 0.0;
        for (int t = 1; t <= window; ++t) {
            sum_corr += C[t];
        }
        tau_int = 0.5 + sum_corr / C0;
    }

    result.tau_int = tau_int;
    result.window = window;
    result.converged = converged;
    result.tau_stderr = std::sqrt(2.0 * (2.0 * window + 1.0) / N) * tau_int;
    return result;
}

int compute_adaptive_stride(
    Ising2D& model,
    int calibration_sweeps,
    int min_stride,
    int max_stride
) {
    if (calibration_sweeps <= 0) {
        return std::max(1, min_stride);
    }

    std::vector<double> series;
    series.reserve(static_cast<size_t>(calibration_sweeps));
    for (int s = 0; s < calibration_sweeps; ++s) {
        model.sweep_metropolis();
        series.push_back(std::fabs(model.magnetization_per_spin()));
    }

    const AutocorrResult result = measure_autocorrelation(series, -1);
    int stride = static_cast<int>(std::floor(2.0 * result.tau_int));
    stride = std::max(stride, min_stride);
    stride = std::min(stride, max_stride);
    return std::max(1, stride);
}

JackknifeObservables jackknife_observables(
    const std::vector<RawSample>& samples,
    int L,
    double T
) {
    JackknifeObservables out;
    const int N = static_cast<int>(samples.size());
    assert(N >= 10 && "Jackknife requires at least 10 samples for stable error estimates.");
    if (N < 10) {
        throw std::runtime_error("Jackknife requires at least 10 samples for stable error estimates.");
    }

    const double invN = 1.0 / N;
    double sum_abs_m = 0.0;
    double sum_m2 = 0.0;
    double sum_m4 = 0.0;
    double sum_e = 0.0;
    double sum_e2 = 0.0;

    for (const auto& s : samples) {
        const double m2 = s.m * s.m;
        sum_abs_m += s.abs_m;
        sum_m2 += m2;
        sum_m4 += m2 * m2;
        sum_e += s.e;
        sum_e2 += s.e * s.e;
    }
    const double mean_abs_m = sum_abs_m * invN;
    const double mean_m2 = sum_m2 * invN;
    const double mean_m4 = sum_m4 * invN;
    const double mean_e = sum_e * invN;
    const double mean_e2 = sum_e2 * invN;
    const double n_spins = static_cast<double>(L) * L;
    const double chi_full = n_spins * (mean_m2 - mean_abs_m * mean_abs_m) / T;
    const double C_full = (mean_e2 - mean_e * mean_e) / (T * T);
    const double U_full = 1.0 - mean_m4 / (3.0 * mean_m2 * mean_m2);

    std::vector<double> chi_loo(N);
    std::vector<double> C_loo(N);
    std::vector<double> U_loo(N);
    std::vector<double> abs_m_loo(N);
    std::vector<double> e_loo(N);

    const double invNm1 = 1.0 / (N - 1);
    for (int k = 0; k < N; ++k) {
        const double m2_k = samples[k].m * samples[k].m;
        const double m4_k = m2_k * m2_k;
        const double e2_k = samples[k].e * samples[k].e;

        const double mean_abs_m_k = (mean_abs_m * N - samples[k].abs_m) * invNm1;
        const double mean_m2_k = (mean_m2 * N - m2_k) * invNm1;
        const double mean_m4_k = (mean_m4 * N - m4_k) * invNm1;
        const double mean_e_k = (mean_e * N - samples[k].e) * invNm1;
        const double mean_e2_k = (mean_e2 * N - e2_k) * invNm1;

        chi_loo[k] = n_spins * (mean_m2_k - mean_abs_m_k * mean_abs_m_k) / T;
        C_loo[k] = (mean_e2_k - mean_e_k * mean_e_k) / (T * T);
        U_loo[k] = 1.0 - mean_m4_k / (3.0 * mean_m2_k * mean_m2_k);
        abs_m_loo[k] = mean_abs_m_k;
        e_loo[k] = mean_e_k;
    }

    auto mean_of = [](const std::vector<double>& v) {
        double sum = 0.0;
        for (double x : v) {
            sum += x;
        }
        return sum / static_cast<double>(v.size());
    };

    const double mean_chi = mean_of(chi_loo);
    const double mean_C = mean_of(C_loo);
    const double mean_U = mean_of(U_loo);
    const double mean_abs_m_loo = mean_of(abs_m_loo);
    const double mean_e_loo = mean_of(e_loo);

    auto jackknife_var = [N](const std::vector<double>& v, double mean_v) {
        double accum = 0.0;
        for (double x : v) {
            const double diff = x - mean_v;
            accum += diff * diff;
        }
        return (static_cast<double>(N - 1) / N) * accum;
    };

    out.chi = N * chi_full - (N - 1) * mean_chi;
    out.C = N * C_full - (N - 1) * mean_C;
    out.U = N * U_full - (N - 1) * mean_U;
    out.abs_m = N * mean_abs_m - (N - 1) * mean_abs_m_loo;
    out.e = N * mean_e - (N - 1) * mean_e_loo;

    out.chi_err = std::sqrt(jackknife_var(chi_loo, mean_chi));
    out.C_err = std::sqrt(jackknife_var(C_loo, mean_C));
    out.U_err = std::sqrt(jackknife_var(U_loo, mean_U));
    out.abs_m_err = std::sqrt(jackknife_var(abs_m_loo, mean_abs_m_loo));
    out.e_err = std::sqrt(jackknife_var(e_loo, mean_e_loo));
    out.n_samples = N;
    return out;
}
