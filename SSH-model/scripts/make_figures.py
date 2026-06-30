"""Generate figures/fig1-fig6 and figures/results.txt.

Usage: poetry run python SSH-model/scripts/make_figures.py [--quick]
"""

import os

for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import argparse
import pathlib
import sys
from concurrent.futures import ProcessPoolExecutor

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ssh_model import (
    H_open,
    bands,
    critical_disorder,
    edge_splitting,
    edge_states,
    fit_xi,
    h_k,
    open_spectrum,
    random_hoppings,
    real_space_winding,
    splitting_prediction,
    winding_number,
    xi_exact,
    zak_phase,
)

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
    "errorbar.capsize": 2,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})

results = []


def log(msg):
    print(msg)
    results.append(msg)


def panel(ax, label):
    ax.text(-0.01, 1.015, f"({label})", transform=ax.transAxes, va="bottom", ha="right", fontsize=12)


def save(fig, name):
    fig.savefig(FIG / name)
    plt.close(fig)


def fig_bulk():
    k = np.linspace(-np.pi, np.pi, 801)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE, gridspec_kw={"width_ratios": [1.15, 1]})
    for c, v in zip(COLORS, [0.5, 1.0, 1.5]):
        nu = winding_number(v, 1.0)
        nu_text = "undefined" if np.isnan(nu) else f"{nu:.0f}"
        em, ep = bands(k, v, 1.0)
        a1.plot(k, ep, color=c, label=rf"$v={v}$, $\nu$ = {nu_text}")
        a1.plot(k, em, color=c)
        h = h_k(k, v, 1.0)
        a2.plot(h.real, h.imag, color=c)
        i = 250
        a2.annotate("", xy=(h.real[i + 1], h.imag[i + 1]), xytext=(h.real[i], h.imag[i]),
                    arrowprops={"arrowstyle": "-|>", "color": c, "lw": 1.2, "mutation_scale": 11})
        log(f"[bulk] v={v}, w=1: gap={2 * abs(v - 1):.3f}, winding={nu}")
    a1.set(xlabel=r"$k$", ylabel=r"$E(k)/w$", xlim=(-np.pi, np.pi), ylim=(-2.9, 2.9))
    a1.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi], [r"$-\pi$", r"$-\pi/2$", "0", r"$\pi/2$", r"$\pi$"])
    a1.legend(loc="center", bbox_to_anchor=(0.5, 0.5), handlelength=1.4)
    a2.axhline(0, color=GREY, lw=0.5, zorder=0)
    a2.axvline(0, color=GREY, lw=0.5, zorder=0)
    a2.plot(0, 0, "o", color="k", ms=5, zorder=5)
    a2.set(xlabel=r"$\mathrm{Re}\,h(k)$", ylabel=r"$\mathrm{Im}\,h(k)$",
           xlim=(-0.8, 2.8), ylim=(-1.3, 1.3), aspect="equal")
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig1_bulk.png")


def fig_open_spectrum(N=40):
    ratios = np.linspace(0.0, 2.0, 401)
    E = np.array([open_spectrum(N, r, 1.0)[0] for r in ratios])
    fig, ax = plt.subplots(figsize=SINGLE)
    ax.axvspan(0, 1, color=BLUE, alpha=0.06, lw=0)
    ax.plot(ratios, np.delete(E, [N - 1, N], axis=1), color=GREY, lw=0.5)
    ax.plot(ratios, E[:, N - 1], color=ORANGE, lw=1.8, label="two states closest to $E=0$")
    ax.plot(ratios, E[:, N], color=ORANGE, lw=1.8)
    ax.text(0.5, 2.6, r"$\nu = 1$", ha="center")
    ax.text(1.5, 2.6, r"$\nu = 0$", ha="center")
    ax.set(xlabel=r"$v/w$", ylabel=r"$E/w$", xlim=(0, 2), ylim=(-3.1, 3.1))
    ax.legend(loc="lower left")
    save(fig, "fig2_open_spectrum.png")
    for v in [0.5, 1.5]:
        log(f"[open] N=60, v={v}, w=1: min|E| = {edge_splitting(60, v, 1.0):.3e}")


def fig_edge_profile(N=40):
    psi_A, psi_B = edge_states(N, 0.5, 1.0)
    cells = np.arange(N)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    for psi, sub, c, m, label in [(psi_A, 0, BLUE, "o", r"left state (on A)"),
                                  (psi_B, 1, ORANGE, "s", r"right state (on B)")]:
        p = psi[sub::2] ** 2
        keep = p > 1e-4
        a1.vlines(cells[keep], 0, p[keep], color=c, lw=1)
        a1.plot(cells[keep], p[keep], m, color=c, label=label)
    a1.set(xlabel=r"cell $n$", ylabel=r"$|\psi(n)|^2$", xlim=(-1, N), ylim=(0, 0.82))
    a1.legend(loc="upper center")
    a1.text(0.5, 0.62, rf"$v/w = 0.5,\ N = {N}$", transform=a1.transAxes, ha="center")

    for c, m, (v, w) in zip(COLORS, "o^s", [(0.3, 1.0), (0.5, 1.0), (0.7, 1.0)]):
        xi, n, amp = fit_xi(100, v, w)
        xe = xi_exact(v, w)
        sel = amp > 1e-12
        a2.semilogy(n[sel], amp[sel], m, color=c, ms=3.5, mfc="none", mew=0.9)
        a2.semilogy(n[sel], np.exp(-n[sel] / xe), color=c, lw=1.2,
                    label=rf"$v/w={v / w:.1f}$:  $\xi={xi:.4f}$")
        log(f"[xi] v={v}, w={w}: fit={xi:.6f}, exact={xe:.6f}, rel.err={xi / xe - 1:+.2e}")
    a2.set(xlabel=r"cell $n$", ylabel=r"$|\psi_A(n)|\,/\,|\psi_A(0)|$", xlim=(-2, 80), ylim=(1e-12, 3))
    a2.legend(loc="upper right", title=r"lines: $e^{-n/\xi}$, $\xi = 1/\ln(w/v)$", title_fontsize=9)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig3_edge_profile.png")


def fig_splitting():
    Ns = np.arange(4, 41)
    pairs = [(0.5, 1.0), (0.7, 1.0), (1.0, 2.0)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    for c, m, (v, w) in zip(COLORS, "o^s", pairs):
        eps = np.array([edge_splitting(N, v, w) for N in Ns])
        ok = eps > 1e-14
        fit = ok & (Ns >= 6)
        slope, _ = np.polyfit(Ns[fit], np.log(eps[fit]), 1)
        label = rf"$v={v},\ w={w}$"
        a1.semilogy(Ns[ok], eps[ok], m, color=c, mfc="none", mew=0.9)
        a1.semilogy(Ns, splitting_prediction(Ns, v, w), color=c, lw=1.2, label=label)
        r = v / w
        a2.plot(Ns[ok], eps[ok] / r ** Ns[ok], m + "-", color=c, mfc="none", mew=0.9, lw=0.8, label=label)
        a2.axhline(w * (1 - r * r), color=c, ls="--", lw=0.9)
        pref = [edge_splitting(N, v, w) / r ** N for N in (10, 20, 30)]
        log(f"[split] v={v}, w={w}: slope={slope:.5f}, ln(v/w)={np.log(r):.5f}, "
            f"eps/(v/w)^N at N=10,20,30 = {np.round(pref, 5).tolist()}, "
            f"1-(v/w)^2={1 - r * r:.4f}, w(1-(v/w)^2)={w * (1 - r * r):.4f}")
    a1.set(xlabel=r"$N$ (cells)", ylabel=r"$\varepsilon = \min|E|$", xlim=(3, 41))
    a1.legend(loc="lower left", title=r"lines: $w(1-(v/w)^2)(v/w)^N$", title_fontsize=9)
    a2.set(xlabel=r"$N$ (cells)", ylabel=r"$\varepsilon\,/\,(v/w)^N$", xlim=(3, 41), ylim=(0.4, 1.65))
    a2.legend(loc="center right", bbox_to_anchor=(1.0, 0.58), title=r"dashed: $w(1-(v/w)^2)$",
              title_fontsize=9)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig4_splitting.png")


def fig_invariants():
    ratios = np.concatenate([np.linspace(0.02, 0.98, 25), np.linspace(1.02, 2.0, 25)])
    rng = np.random.default_rng(0)
    nu = np.array([winding_number(r, 1.0) for r in ratios])
    g = np.abs(np.array([zak_phase(r, 1.0) for r in ratios]))
    gr = np.abs(np.array([zak_phase(r, 1.0, random_gauge=rng) for r in ratios]))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE)
    for ax in (a1, a2):
        ax.axvspan(0, 1, color=BLUE, alpha=0.06, lw=0)
        ax.axvline(1.0, color=GREY, ls="--", lw=0.8)
    a1.plot(ratios, nu, "o", color=BLUE)
    a1.set(xlabel=r"$v/w$", ylabel=r"winding number $\nu$", xlim=(0, 2), ylim=(-0.15, 1.15))
    a1.set_yticks([0, 1])
    a2.plot(ratios, g, "o", color=BLUE, label="eigh gauge")
    a2.plot(ratios, gr, "x", color=ORANGE, ms=6, mew=1.2, label=r"random phase on each $u(k_j)$")
    a2.set(xlabel=r"$v/w$", ylabel=r"Zak phase $|\gamma|$", xlim=(0, 2), ylim=(-0.5, np.pi + 0.5))
    a2.set_yticks([0, np.pi / 2, np.pi], ["0", r"$\pi/2$", r"$\pi$"])
    a2.legend(loc="center right")
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig5_invariants.png")
    log(f"[winding] max |nu - round(nu)|: {np.max(np.abs(nu - np.round(nu))):.2e}")
    log(f"[zak] max |gamma - pi| for v<w: {np.max(np.abs(g[ratios < 1] - np.pi)):.2e}; "
        f"max |gamma| for v>w: {np.max(g[ratios > 1]):.2e}; "
        f"max gauge difference: {np.max(np.abs(g - gr)):.2e}")


def winding_samples(args):
    N, v, Wv, Ww, n, seed = args
    rng = np.random.default_rng(seed)
    out = np.empty(n)
    for i in range(n):
        vn, wn = random_hoppings(N, v, 1.0, Wv, Ww, rng)
        out[i] = real_space_winding(H_open(N, vn, wn), N)
    return out


def sample_points(points, realisations, seed, chunk=10):
    """points: list of (N, v, Wv, Ww). Returns (mean, standard error) per point."""
    n_chunks = -(-realisations // chunk)
    seeds = np.random.SeedSequence(seed).spawn(len(points) * n_chunks)
    tasks = [(*p, chunk, seeds[i * n_chunks + j]) for i, p in enumerate(points) for j in range(n_chunks)]
    with ProcessPoolExecutor() as pool:
        samples = list(pool.map(winding_samples, tasks))
    out = []
    for i in range(len(points)):
        s = np.concatenate(samples[i * n_chunks:(i + 1) * n_chunks])[:realisations]
        out.append((s.mean(), s.std(ddof=1) / np.sqrt(len(s))))
    return np.array(out)


def cached(name, compute):
    path = FIG / name
    if path.exists():
        return np.load(path)["res"]
    res = compute()
    np.savez(path, res=res)
    return res


def fig_disorder(realisations, N=200):
    Ws = np.arange(0.0, 8.01, 0.25)
    cases = [
        (0.5, 0.5, r"$v=0.5$, $W_w = W/2$"),
        (1.2, 0.5, r"$v=1.2$, $W_w = W/2$"),
        (0.5, 1.0, r"$v=0.5$, $W_w = W$"),
    ]
    points = [(N, v, W, ratio * W) for v, ratio, _ in cases for W in Ws]
    data = cached(f"disorder_N{N}_R{realisations}.npz",
                  lambda: sample_points(points, realisations, 12345)).reshape(len(cases), len(Ws), 2)

    sizes = np.array([50, 100, 200, 400, 800])
    fss_cases = [
        (0.5, 0.5, 3.75, BLUE, "o", BLUE, r"$W_w = W/2$, $W = 3.75 < W_c$"),
        (0.5, 0.5, 4.25, BLUE, "s", "white", r"$W_w = W/2$, $W = 4.25 > W_c$"),
        (0.5, 1.0, 3.0, GREEN, "D", GREEN, r"$W_w = W$, $W = 3$"),
        (0.5, 1.0, 5.0, GREEN, "^", "white", r"$W_w = W$, $W = 5$"),
    ]
    fss_points = [(n, v, W, ratio * W) for v, ratio, W, *_ in fss_cases for n in sizes]
    fss_R = max(realisations, 100)
    fss = cached(f"disorder_fss_R{fss_R}.npz",
                 lambda: sample_points(fss_points, fss_R, 54321)).reshape(len(fss_cases), len(sizes), 2)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=DOUBLE, gridspec_kw={"width_ratios": [1.35, 1]})
    for c, m, (v, ratio, label), res in zip(COLORS, "osD", cases, data):
        a1.errorbar(Ws, res[:, 0], yerr=res[:, 1], fmt=m + "-", color=c, ms=3.8, lw=1.1,
                    mfc="white", mew=1.0, label=label)
        Wc = critical_disorder(v, 1.0, ratio)
        for x in Wc:
            a1.axvline(x, color=c, ls="--", lw=0.8)
        log(f"[disorder] v={v}, Ww/Wv={ratio}: <ln|v|>=<ln|w|> at W = {np.round(Wc, 3).tolist()}")
        for W, (mu, err) in zip(Ws, res):
            log(f"    W={W:4.2f}  nu={mu:+.4f} +- {err:.4f}")
    a1.set(xlabel=r"disorder strength $W$", ylabel=r"real-space winding $\nu$",
           xlim=(-0.1, 8.1), ylim=(-0.06, 1.1))
    a1.legend(loc="upper right")
    a1.text(0.97, 0.62, rf"$N={N}$, central {N // 2} cells" "\n" rf"{realisations} realisations",
            transform=a1.transAxes, ha="right", fontsize=9)

    for (v, ratio, W, c, m, fill, label), res in zip(fss_cases, fss):
        a2.errorbar(sizes, res[:, 0], yerr=res[:, 1], fmt=m + "-", color=c, ms=4, lw=1.0,
                    mfc=fill, mew=1.0, label=label)
        log(f"[fss] v={v}, Ww/Wv={ratio}, W={W}: " +
            ", ".join(f"N={n}: {mu:.3f}+-{err:.3f}" for n, (mu, err) in zip(sizes, res)))
    a2.set_xscale("log")
    a2.set_xticks(sizes, [str(s) for s in sizes])
    a2.minorticks_off()
    a2.set(xlabel=r"$N$ (cells), central $N/2$ cells", ylabel=r"$\nu$", ylim=(-0.06, 1.42))
    a2.set_yticks(np.arange(0, 1.01, 0.2))
    a2.legend(loc="upper left", fontsize=8.5, ncol=2, columnspacing=1.0)
    panel(a1, "a")
    panel(a2, "b")
    fig.tight_layout(w_pad=2.0)
    save(fig, "fig6_disorder.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="20 disorder realisations instead of 100")
    args = parser.parse_args()
    FIG.mkdir(exist_ok=True)
    fig_bulk()
    fig_open_spectrum()
    fig_edge_profile()
    fig_splitting()
    fig_invariants()
    fig_disorder(20 if args.quick else 100)
    (FIG / "results.txt").write_text("\n".join(results) + "\n")


if __name__ == "__main__":
    main()
