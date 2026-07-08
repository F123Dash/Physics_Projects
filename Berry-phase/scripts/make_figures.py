"""Generate figures/fig1-fig4 and figures/results.txt.

Usage: poetry run python Berry-phase/scripts/make_figures.py
"""

import pathlib
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from berry_phase import (
    BASE,
    aharonov_anandan_phase,
    berry_phase,
    berry_phase_cone,
    berry_phase_exact,
    chern_number,
    cone_states,
    connection,
    dark_states,
    discretisation_error,
    evolve_loops,
    field_strength_exact,
    holonomy,
    loop_frames,
    loop_hamiltonian,
    loop_holonomy,
    loop_pair,
    magnus4,
    nonadiabatic_slope,
    parallel_transport,
    ramp,
    rotating_frame_angle,
    rotating_frame_exact,
    small_loop_field,
    spin_geometric_phase,
    spin_geometric_phase_exact,
    spin_hamiltonian,
    spin_state,
    unitary_part,
    wilson_loop,
    wrap,
)
from berry_phase.spin import SIGMA

FIG = ROOT / "figures"
BLUE, ORANGE, GREEN = "#2a78d6", "#eb6834", "#1baf7a"
COLORS = [BLUE, ORANGE, GREEN]
GREY = "#9a9994"
SINGLE = (4.6, 3.3)
DOUBLE = (9.2, 3.5)

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.labelsize": 12,
    "axes.linewidth": 0.8,
    "legend.fontsize": 9,
    "legend.frameon": False,
    "legend.handlelength": 1.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "xtick.minor.size": 2,
    "ytick.minor.size": 2,
    "lines.linewidth": 1.6,
    "lines.markersize": 4.5,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

C0 = np.array([1.0, 0.0], dtype=complex)
T_SLOW = 200.0
results = []


def log(msg):
    print(msg)
    results.append(msg)


def panel(ax, label):
    ax.text(-0.01, 1.015, f"({label})", transform=ax.transAxes, va="bottom", ha="right", fontsize=12)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


def bloch(c):
    """Dark-subspace Bloch vector of components c with shape (..., 2)."""
    c = c / np.linalg.norm(c, axis=-1, keepdims=True)
    return np.real(np.einsum("...a,kab,...b->...k", np.conj(c), SIGMA, c))


def fig_berry_phase():
    rng = np.random.default_rng(0)
    thetas = [0.3, 0.9, 1.5, 2.2, 2.9]
    for th in thetas:
        errs = [wrap(berry_phase_cone(th, M) - berry_phase_exact(th)) for M in (20, 50, 400)]
        gauge = max(abs(wrap(berry_phase_cone(th, 400, rng=rng) - berry_phase_cone(th, 400))) for _ in range(10))
        u = cone_states(th, 400)
        rev = abs(wrap(berry_phase(u[::-1]) + berry_phase(u)))
        anti = abs(wrap(berry_phase_cone(th, 400, False) + berry_phase_cone(th, 400)))
        log(f"[berry] theta={th}: exact={berry_phase_exact(th):+.6f}, error M=20,50,400 = "
            f"{errs[0]:+.3e}, {errs[1]:+.3e}, {errs[2]:+.3e}; leading term at M=400 = "
            f"{discretisation_error(th, 400):+.3e}; random-gauge change {gauge:.1e}; "
            f"reversed loop |g_rev + g| = {rev:.1e}; |g_up + g_down| = {anti:.1e}")
    log(f"[chern] aligned: {chern_number(aligned=True):+.12f}, anti-aligned: {chern_number(aligned=False):+.12f}, "
        f"aligned with random gauge on a 6x8 grid: {chern_number(6, 8, rng=rng):+.12f}")

    th_line = np.linspace(0, np.pi, 300)
    th_pts = np.linspace(0.1, np.pi - 0.1, 16)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    for aligned, c, m, label in [(True, BLUE, "o", "aligned"), (False, ORANGE, "s", "anti-aligned")]:
        num = np.unwrap([berry_phase_cone(t, 400, aligned) for t in th_pts])
        num += 2 * np.pi * np.round((berry_phase_exact(th_pts[0], aligned) - num[0]) / (2 * np.pi))
        gr = np.array([berry_phase_cone(t, 400, aligned, rng=rng) for t in th_pts])
        gr = num + wrap(gr - num)
        a1.plot(th_line, berry_phase_exact(th_line, aligned) / np.pi, color=c, lw=1.2)
        a1.plot(th_pts, num / np.pi, m, color=c, mfc="white", mew=1.0, label=f"{label}, $M = 400$")
        a1.plot(th_pts, gr / np.pi, "x", color="k", ms=3.5, mew=0.8)
    a1.plot([], [], "x", color="k", ms=3.5, mew=0.8, label="random phase on each $u_j$")
    a1.text(1.75, -0.5, r"$-\pi(1-\cos\theta)$", color="k", ha="center")
    a1.text(1.75, 0.38, r"$+\pi(1-\cos\theta)$", color="k", ha="center")
    a1.axhline(0, color=GREY, lw=0.5, zorder=0)
    a1.set(xlabel=r"cone angle $\theta$", ylabel=r"Berry phase $\gamma/\pi$", xlim=(0, np.pi), ylim=(-2.15, 2.15))
    a1.set_xticks([0, np.pi / 4, np.pi / 2, 3 * np.pi / 4, np.pi], ["0", r"$\pi/4$", r"$\pi/2$", r"$3\pi/4$", r"$\pi$"])
    a1.legend(loc="upper left")

    Ms = np.unique(np.round(np.logspace(np.log10(8), 3.3, 22)).astype(int))
    for th, c, m in zip([0.2, 1.45, 0.9], COLORS, "o^s"):
        err = np.abs([wrap(berry_phase_cone(th, M) - berry_phase_exact(th)) for M in Ms])
        a2.loglog(Ms, err, m, color=c, mfc="white", mew=1.0)
        a2.loglog(Ms, np.abs(discretisation_error(th, Ms)), color=c, lw=1.1, label=rf"$\theta = {th}$")
    a2.axvline(400, color=GREY, ls=":", lw=0.9)
    a2.set(xlabel=r"loop points $M$", ylabel=r"$|\gamma_M - \gamma|$", xlim=(7, 2400), ylim=(3e-7, 0.3))
    a2.legend(loc="upper right", title=r"lines: $\frac{\pi^3}{3}\,\sin^2\!\theta\,|\cos\theta|\,/M^2$",
              title_fontsize=9)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig1_berry_phase.png")


def fig_nonadiabatic():
    th, Om = 0.9, 0.1
    T = 2 * np.pi / Om
    psi0 = spin_state(th, 0.0)
    exact = rotating_frame_exact(th, 1.0, Om, T, psi0)
    for steps in (1000, 2000, 4000):
        err = np.linalg.norm(magnus4(spin_hamiltonian(th, 1.0, Om), psi0, T, steps) - exact)
        log(f"[integrator] theta={th}, Omega/omega0={Om}, T=2pi/Omega, {steps} steps: |psi - psi_exact| = {err:.2e}")

    eps = np.logspace(-3, np.log10(0.5), 12)
    eps_line = np.logspace(-3, np.log10(0.5), 300)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    for t, c, m in zip([0.5, 0.9, 2.0], COLORS, "o^s"):
        g = np.array([spin_geometric_phase(t, e) for e in eps])
        err = np.abs(wrap(g - berry_phase_exact(t)))
        close = np.max(np.abs(wrap(g - spin_geometric_phase_exact(t, eps))))
        slope = np.polyfit(np.log(eps[:5]), np.log(err[:5]), 1)[0]
        ratio = wrap(g[0] - berry_phase_exact(t)) / eps[0]
        log(f"[nonadiabatic] theta={t}: log-log slope (eps<=0.007) = {slope:.4f}; "
            f"(gamma - gamma_B)/eps at eps=1e-3 = {ratio:.5f}, predicted -(pi/2) sin^2 theta = "
            f"{nonadiabatic_slope(t):.5f}; max |numerics - closed form| = {close:.1e}")
        a1.loglog(eps_line, np.abs(wrap(spin_geometric_phase_exact(t, eps_line) - berry_phase_exact(t))),
                  color=c, lw=1.1, label=rf"$\theta = {t}$")
        a1.loglog(eps_line, np.abs(nonadiabatic_slope(t)) * eps_line, color=c, ls="--", lw=0.9)
        a1.loglog(eps, err, m, color=c, mfc="white", mew=1.0)
    a1.set(xlabel=r"$\Omega/\omega_0$", ylabel=r"$|\gamma_{\rm ext} - \gamma_{\rm Berry}|$", xlim=(8e-4, 0.6),
           ylim=(2e-4, 2))
    a1.legend(loc="lower right", title="lines: exact; dashed: $\\frac{\\pi}{2}\\sin^2\\theta\\;\\Omega/\\omega_0$",
              title_fontsize=9)

    th = 0.9
    eps2 = np.linspace(0.02, 0.8, 14)
    g_b = np.array([spin_geometric_phase(th, e) for e in eps2])
    g_aa = []
    for e in eps2:
        T = 2 * np.pi / e
        H = spin_hamiltonian(th, 1.0, e)
        steps = int(np.ceil(80 / e))
        path = magnus4(H, spin_state(rotating_frame_angle(th, e), 0.0), T, steps, return_path=True)
        g_aa.append(aharonov_anandan_phase(H, np.linspace(0, T, steps + 1), path))
    g_aa = np.array(g_aa)
    aa_err = np.max(np.abs(wrap(g_aa - berry_phase_exact(rotating_frame_angle(th, eps2)))))
    log(f"[aharonov-anandan] theta={th}, Omega/omega0 in [0.02, 0.8]: max |gamma_AA + pi(1 - cos theta')| = {aa_err:.1e}")
    e_line = np.linspace(0, 0.82, 200)
    a2.axhline(berry_phase_exact(th), color=GREY, ls="--", lw=0.9)
    a2.text(0.45, berry_phase_exact(th) + 0.06, r"$-\pi(1-\cos\theta)$", fontsize=9)
    a2.plot(e_line, berry_phase_exact(rotating_frame_angle(th, e_line)), color=ORANGE, lw=1.1)
    a2.plot(eps2, g_aa, "s", color=ORANGE, mfc="white", mew=1.0,
            label=r"AA phase, start on $H_{\rm rot}$ axis; line $-\pi(1-\cos\theta')$")
    a2.plot(e_line[1:], spin_geometric_phase_exact(th, e_line[1:]), color=BLUE, lw=1.1)
    a2.plot(eps2, g_b, "o", color=BLUE, mfc="white", mew=1.0, label=r"$\arg\langle\psi_0|\psi_T\rangle + \omega_0 T/2$")
    a2.set(xlabel=r"$\Omega/\omega_0$", ylabel=r"geometric phase", xlim=(0, 0.82), ylim=(-4.0, -0.85))
    a2.legend(loc="lower left", fontsize=8.5)
    a2.text(0.97, 0.80, rf"$\theta = {th}$", transform=a2.transAxes, ha="right", va="top")
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig2_nonadiabatic.png")


def fig_wilson_vs_evolution():
    x = np.array([0.7, 0.4, 0.3])
    A = connection(BASE)
    log(f"[tripod] at BASE={BASE}: ||[A_phi, A_S]||_F = {np.linalg.norm(A[1] @ A[2] - A[2] @ A[1]):.6f}, "
        f"analytic sqrt(2) cos(theta) sin^2(theta) = {np.sqrt(2) * np.cos(BASE[0]) * np.sin(BASE[0]) ** 2:.6f}")
    for plane in [(0, 1), (0, 2), (1, 2)]:
        F = field_strength_exact(x[0])[plane]
        errs = []
        for r in (0.02, 0.01, 0.005):
            x0 = x.copy()
            x0[plane[0]] += r
            errs.append(np.linalg.norm(small_loop_field(x0, plane, r) - F))
        log(f"[small loop] plane {plane}, x={x.tolist()}: ||-i log U / area - F||_F at r=0.02,0.01,0.005 = "
            + ", ".join(f"{e:.2e}" for e in errs))

    _, b = loop_pair()
    for M in (250, 500, 1000, 2000):
        f = loop_frames(b, M)
        W = wilson_loop(f)
        U = holonomy(f)
        log(f"[wilson] loop b, M={M}: ||polar(W)^dag - U|| = {np.linalg.norm(unitary_part(W).conj().T - U):.2e}, "
            f"1 - min singular value of W = {1 - np.min(np.linalg.svd(W)[1]):.2e}, "
            f"U eigenphases = {np.round(np.sort(np.angle(np.linalg.eigvals(U))), 6).tolist()}")

    M = 2000
    frames = loop_frames(b, M)
    U = holonomy(frames)
    D0 = dark_states(*BASE)
    steps = int(40 * T_SLOW)
    path = magnus4(loop_hamiltonian(b, T_SLOW), D0 @ C0, T_SLOW, steps, return_path=True)
    t = np.linspace(0, T_SLOW, steps + 1)
    s = ramp(t / T_SLOW)
    D_t = dark_states(*np.moveaxis(b(s), -1, 0))
    c_t = np.einsum("kia,ki->ka", np.conj(D_t), path)
    pt = parallel_transport(frames)
    s_grid = np.arange(M + 1) / M
    D_grid = dark_states(*np.moveaxis(b(s_grid), -1, 0))
    c_pred = np.einsum("kia,kib,b->ka", np.conj(D_grid), pt, C0)
    fid = abs(np.vdot(U @ C0, D0.conj().T @ path[-1])) ** 2
    log(f"[evolution] loop b, T={T_SLOW}, smooth ramp: 1 - fidelity with U c0 = {1 - fid:.2e}, "
        f"leakage out of dark subspace = {1 - np.linalg.norm(D0.conj().T @ path[-1]) ** 2:.2e}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    sub = np.linspace(0, steps, 31).astype(int)
    for k, c, m, lab in zip(range(3), COLORS, "o^s", [r"$\langle\sigma_x\rangle$", r"$\langle\sigma_y\rangle$",
                                                      r"$\langle\sigma_z\rangle$"]):
        a1.plot(s_grid, bloch(c_pred)[:, k], color=c, lw=1.2, label=lab)
        a1.plot(s[sub], bloch(c_t[sub])[:, k], m, color=c, mfc="white", mew=1.0)
    a1.set(xlabel=r"loop parameter $s$", ylabel="dark-state Bloch vector", xlim=(0, 1), ylim=(-1.1, 1.5))
    a1.legend(loc="upper center", ncol=3, columnspacing=1.2,
              title="lines: parallel transport; symbols: 4-level evolution", title_fontsize=9)

    Ts = np.array([10, 20, 40, 80, 160, 320, 640])
    for smooth, c, m, lab in [(False, GREY, "o", "linear ramp"), (True, BLUE, "s", "smooth ramp")]:
        inf = np.array([1 - abs(np.vdot(U @ C0, D0.conj().T @ evolve_loops([b], T, D0 @ C0, smooth=smooth))) ** 2
                        for T in Ts])
        p = np.polyfit(np.log(Ts[3:]), np.log(inf[3:]), 1)[0]
        log(f"[ramp] {lab}: 1 - F at T = {Ts.tolist()}: " + ", ".join(f"{v:.2e}" for v in inf)
            + f"; fitted exponent (T >= 80) = {p:.2f}")
        a2.loglog(Ts, inf, m + "-", color=c, mfc="white", mew=1.0, lw=0.9, label=lab)
    Tg = Ts[Ts >= 80]
    a2.loglog(Tg, 3e-6 * (Tg / 640.0) ** -2, color="k", ls="--", lw=0.8)
    a2.loglog(Tg, 1.5e-9 * (Tg / 640.0) ** -4, color="k", ls=":", lw=0.9)
    a2.text(300, 3e-6 * (300 / 640) ** -2 / 4, r"$\propto T^{-2}$", fontsize=10, ha="center", va="top")
    a2.text(150, 1.5e-9 * (150 / 640) ** -4 / 4, r"$\propto T^{-4}$", fontsize=10, ha="center", va="top")
    a2.set(xlabel=r"loop time $T\,\Omega_0$", ylabel=r"$1 - |\langle U c_0 | c(T)\rangle|^2$", xlim=(8, 800))
    a2.legend(loc="lower left")
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig3_wilson_vs_evolution.png")


def fig_loop_order():
    D0 = dark_states(*BASE)
    a, b = loop_pair()
    Ua, Ub = loop_holonomy(a, 2000), loop_holonomy(b, 2000)
    wilson = abs(np.vdot(Ub @ Ua @ C0, Ua @ Ub @ C0))
    log(f"[order] ||[U_a, U_b]||_F = {np.linalg.norm(Ua @ Ub - Ub @ Ua):.4f}; "
        f"Wilson-loop overlap |<U_b U_a c0 | U_a U_b c0>| = {wilson:.4f}")
    ac, bc = loop_pair(abelian=True)
    Uac, Ubc = loop_holonomy(ac, 2000), loop_holonomy(bc, 2000)
    log(f"[order] Abelian control (S fixed): ||[U_a, U_b']||_F = {np.linalg.norm(Uac @ Ubc - Ubc @ Uac):.1e}, "
        f"overlap = {abs(np.vdot(Ubc @ Uac @ C0, Uac @ Ubc @ C0)):.12f}")

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE, gridspec_kw={"width_ratios": [1.25, 1]})
    steps = int(40 * T_SLOW)
    finals = []
    for (p, q), c, m, lab, pred in [((a, b), BLUE, "o", r"$a$ then $b$", Ub @ Ua @ C0),
                                    ((b, a), ORANGE, "s", r"$b$ then $a$", Ua @ Ub @ C0)]:
        p1 = magnus4(loop_hamiltonian(p, T_SLOW), D0 @ C0, T_SLOW, steps, return_path=True)
        p2 = magnus4(loop_hamiltonian(q, T_SLOW), p1[-1], T_SLOW, steps, return_path=True)
        traj = np.concatenate([p1, p2[1:]])
        t = np.linspace(0, 2, 2 * steps + 1)
        sz = bloch(np.einsum("ia,ki->ka", np.conj(D0), traj))[:, 2]
        a1.plot(t, sz, color=c, lw=1.2, label=lab)
        first = (Ua if p is a else Ub) @ C0
        a1.plot([1, 2], [bloch(first)[2], bloch(pred)[2]], m, color=c, ms=7, mfc="white", mew=1.4)
        finals.append(D0.conj().T @ traj[-1])
        log(f"[order] {lab.replace('$', '')}: 1 - fidelity with Wilson prediction = "
            f"{1 - abs(np.vdot(pred, finals[-1])) ** 2:.2e}")
    evo = abs(np.vdot(*finals))
    log(f"[order] time-evolution overlap of the two final states (T={T_SLOW} per loop) = {evo:.4f}")
    a1.axvline(1, ymax=0.76, color=GREY, lw=0.6, zorder=0)
    a1.text(1.97, 0.8, rf"$|\langle\psi_{{ab}}|\psi_{{ba}}\rangle| = {evo:.3f}$", ha="right", fontsize=10)
    a1.set(xlabel=r"$t/T$", ylabel=r"$\langle\sigma_z\rangle$ in the basis $D(x_0)$", xlim=(0, 2.06),
           ylim=(-1.1, 1.65))
    a1.legend(loc="upper center", ncol=2, title="symbols: Wilson-loop prediction", title_fontsize=9)

    lam = np.linspace(0.02, 1.0, 40)

    def overlap(scale, abelian):
        p, q = loop_pair(scale, abelian)
        up, uq = loop_holonomy(p, 600), loop_holonomy(q, 600)
        return abs(np.vdot(uq @ up @ C0, up @ uq @ C0))

    a2.plot(lam, [overlap(l, True) for l in lam], color=GREEN, lw=1.4, ls="--",
            label=r"control: both loops at fixed $S$")
    a2.plot(lam, [overlap(l, False) for l in lam], color=BLUE, lw=1.4, label=r"$b$ in the $(\varphi, S)$ plane")
    lam_pts = np.array([0.25, 0.5, 0.75, 1.0])
    evo_pts = []
    for l in lam_pts:
        p, q = loop_pair(l)
        fa = D0.conj().T @ evolve_loops([p, q], T_SLOW, D0 @ C0)
        fb = D0.conj().T @ evolve_loops([q, p], T_SLOW, D0 @ C0)
        evo_pts.append(abs(np.vdot(fa, fb)))
    a2.plot(lam_pts, evo_pts, "o", color=BLUE, mfc="white", mew=1.2, ms=6, label="4-level evolution")
    log("[order] overlap vs loop scale (Wilson, evolution): " + ", ".join(
        f"lambda={l}: {overlap(l, False):.4f}, {e:.4f}" for l, e in zip(lam_pts, evo_pts)))
    a2.set(xlabel=r"loop size $\lambda$ (radii $\times\lambda$)", ylabel=r"$|\langle\psi_{ab}|\psi_{ba}\rangle|$",
           xlim=(0, 1.02), ylim=(0, 1.08))
    a2.legend(loc="lower left")
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_loop_order.png")


def main():
    FIG.mkdir(exist_ok=True)
    fig_berry_phase()
    fig_nonadiabatic()
    fig_wilson_vs_evolution()
    fig_loop_order()
    (FIG / "results.txt").write_text("\n".join(results) + "\n")


if __name__ == "__main__":
    main()
