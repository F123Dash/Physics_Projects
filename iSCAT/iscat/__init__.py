"""Interferometric scattering (iSCAT) signal."""

from .scattering import (
    backscatter_x2_coefficient,
    clausius_mossotti,
    dipole_amplitude,
    mie_amplitudes,
    mie_backscatter_amplitude,
    mie_coefficients,
    mie_dipole_polarizability,
    mie_efficiencies,
    mie_nmax,
    polarizability,
    polarizability_radiative,
    protein_polarizability,
    rayleigh_cross_sections,
    rayleigh_sigma_sca,
    rayleigh_x2_coefficient,
)

from .imaging import (
    DipoleImager,
    best_focus,
    contrast,
    fresnel,
    fresnel_angular,
    momentum_transfer_sq,
    scalar_peak,
    scalar_psf,
    scalar_reference,
    vectorial_psf,
    vectorial_reference,
    vectorial_transfer,
    vectorial_window_signal,
    wrap,
)

from .noise import (
    camera_grid,
    contrast_noise,
    detection_limit,
    expected_counts,
    matched_filter,
    ratiometric_contrast,
    snr_single_pixel,
)

from .tracking import (
    ParticleModel,
    brownian_track,
    crlb,
    fisher_information,
    fit_msd,
    localise,
    msd,
)

from .roughness import (
    gaussian_psd,
    rough_surface,
    roughness_field,
    speckle_contrast_std,
)

from .photothermal import (
    absorption_cross_section,
    lens_form_factor,
    lens_polarizability,
    lockin_demodulate,
    photothermal_contrast,
    temperature,
    thermal_gamma,
)
