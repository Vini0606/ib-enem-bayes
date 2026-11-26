"""Figuras do relatório (matplotlib puro, para render estático em PDF)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from ib_enem_bayes.config import FIGURES_DIR, PARAM_LABELS, PARAM_NAMES
from ib_enem_bayes.diagnostics import acf

# Paleta categórica fixa (ordem nunca muda): azul, laranja, verde-água, amarelo
C = ["#257de0", "#f47230", "#00b894", "#f2a600", "#6a4ce2"]
INK, INK2, GRID = "#191919", "#666666", "#e6e6e6"

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.size": 9,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "axes.titlesize": 10,
        "axes.titleweight": "bold",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "legend.frameon": False,
        "lines.linewidth": 1.5,
    }
)


def _save(fig, name: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path)
    plt.close(fig)
    return path


def _names(k: int) -> list[str]:
    return PARAM_NAMES[: k - 1] + ["sigma2"]


# --------------------------------------------------------------------------- #
# EDA
# --------------------------------------------------------------------------- #
def eda(df: pd.DataFrame, name: str = "eda") -> Path:
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.2))
    ax[0].hist(df["nota_mt"], bins=60, color=C[0], edgecolor="white", linewidth=0.3)
    ax[0].set(title="Distribuição de Y (nota MT)", xlabel="nota", ylabel="frequência")
    g = df.groupby("log2_renda_sm")["nota_mt"].agg(["mean", "count"]).reset_index()
    for k, (lab, sub) in enumerate(df.groupby("escola_privada")):
        gg = sub.groupby("log2_renda_sm")["nota_mt"].mean()
        ax[1].plot(
            gg.index, gg.values, "o-", ms=4, color=C[k], label=["Pública", "Privada"][int(lab)]
        )
    ax[1].set(
        title="Nota média por faixa de renda", xlabel="log$_2$(renda em SM)", ylabel="nota média"
    )
    ax[1].legend()
    data = [df.loc[df.mae_superior == v, "nota_mt"] for v in (0, 1)]
    bp = ax[2].boxplot(data, tick_labels=["Mãe s/ superior", "Mãe c/ superior"], patch_artist=True, widths=0.5,
                       medianprops={"color": INK}, flierprops={"markersize": 1, "alpha": 0.2})  # fmt: skip
    for patch, col in zip(bp["boxes"], C[:2]):
        patch.set_facecolor(col)
        patch.set_alpha(0.6)
    ax[2].set(title="Nota por escolaridade da mãe", ylabel="nota")
    del g
    return _save(fig, name)


# --------------------------------------------------------------------------- #
# Diagnóstico MCMC
# --------------------------------------------------------------------------- #
def traceplot(draws: np.ndarray, name: str, title: str, burnin: int | None = None) -> Path:
    k = draws.shape[-1]
    fig, axes = plt.subplots(k, 2, figsize=(11, 1.7 * k), gridspec_kw={"width_ratios": [3, 1]})
    for j, pname in enumerate(_names(k)):
        for c in range(draws.shape[0]):
            axes[j, 0].plot(
                draws[c, :, j], lw=0.4, color=C[c % len(C)], alpha=0.8, label=f"cadeia {c + 1}"
            )
            kde = stats.gaussian_kde(draws[c, (burnin or 0) :, j])
            xs = np.linspace(*np.percentile(draws[:, (burnin or 0) :, j], [0.5, 99.5]), 200)
            axes[j, 1].plot(xs, kde(xs), color=C[c % len(C)], lw=1.2)
        if burnin:
            axes[j, 0].axvspan(0, burnin, color=GRID, alpha=0.6, lw=0)
            axes[j, 0].text(burnin / 2, axes[j, 0].get_ylim()[1], "aquecimento", ha="center",
                            va="top", fontsize=7, color=INK2)  # fmt: skip
        axes[j, 0].set_ylabel(PARAM_LABELS[pname], rotation=0, ha="right", va="center")
        axes[j, 1].set_yticks([])
    axes[0, 0].set_title(f"Traceplots — {title}")
    axes[0, 1].set_title("Densidades por cadeia")
    axes[0, 0].legend(ncol=draws.shape[0], fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("iteração")
    return _save(fig, name)


def acf_plot(samplers: dict[str, np.ndarray], name: str, max_lag: int = 40) -> Path:
    k = next(iter(samplers.values())).shape[-1]
    fig, axes = plt.subplots(1, k, figsize=(2.3 * k, 2.6), sharey=True)
    for i, (sname, d) in enumerate(samplers.items()):
        for j, pname in enumerate(_names(k)):
            r = np.mean([acf(d[c, :, j], max_lag) for c in range(d.shape[0])], axis=0)
            axes[j].plot(
                np.arange(max_lag + 1) + 0.15 * i, r, "o-", ms=2.5, lw=1, color=C[i], label=sname
            )
            axes[j].set_title(PARAM_LABELS[pname], fontsize=9)
            axes[j].axhline(0, color=INK2, lw=0.6)
            axes[j].set_xlabel("defasagem")
    axes[0].set_ylabel("ACF (média das cadeias)")
    axes[-1].legend(fontsize=7)
    return _save(fig, name)


def exact_vs_mcmc(exact, samplers: dict[str, np.ndarray], name: str, title: str) -> Path:
    k = next(iter(samplers.values())).shape[-1]
    fig, axes = plt.subplots(1, k, figsize=(2.4 * k, 2.6))
    for j, pname in enumerate(_names(k)):
        d = exact.marginal(pname)
        lo, hi = d.ppf([0.0005, 0.9995])
        xs = np.linspace(lo, hi, 300)
        for i, (sname, dr) in enumerate(samplers.items()):
            kde = stats.gaussian_kde(dr[..., j].ravel())
            axes[j].plot(xs, kde(xs), color=C[i + 1], lw=1.2, ls="--", label=sname)
        axes[j].plot(xs, d.pdf(xs), color=INK, lw=1.6, label="Exata (analítica)")
        axes[j].set_title(PARAM_LABELS[pname], fontsize=9)
        axes[j].set_yticks([])
    axes[-1].legend(fontsize=7)
    fig.suptitle(title, fontsize=10, fontweight="bold", y=1.04)
    return _save(fig, name)


# --------------------------------------------------------------------------- #
# Priori x verossimilhança x posteriori
# --------------------------------------------------------------------------- #
def prior_likelihood_posterior(priors: dict, posts: dict, mle, name: str) -> Path:
    p = len(mle.beta_hat)
    fig, axes = plt.subplots(1, p, figsize=(2.6 * p, 2.7))
    for j in range(p):
        pname = PARAM_NAMES[j]
        se = np.sqrt(mle.cov[j, j])
        centers = [mle.beta_hat[j]] + [po.m[j] for po in posts.values()]
        lo, hi = min(centers) - 6 * se, max(centers) + 6 * se
        xs = np.linspace(lo, hi, 400)
        axes[j].plot(xs, stats.norm.pdf(xs, mle.beta_hat[j], se), color=INK2, lw=3.5, alpha=0.35,
                     label="Verossimilhança (normalizada)")  # fmt: skip
        for i, (lab, pr) in enumerate(priors.items()):
            if pr.is_proper:
                axes[j].plot(
                    xs,
                    pr.beta_marginal(j).pdf(xs),
                    color=C[i],
                    lw=1,
                    ls="--",
                    label=f"Priori — {lab}",
                )
            axes[j].plot(
                xs,
                posts[lab].beta_marginal(j).pdf(xs),
                color=C[i],
                lw=1.6,
                ls="-" if i else (0, (4, 2)),
                label=f"Posteriori — {lab}",
            )
        axes[j].set_title(PARAM_LABELS[pname], fontsize=9)
        axes[j].set_yticks([])
    axes[0].legend(fontsize=6.5, loc="upper left", bbox_to_anchor=(0, -0.18), ncol=5)
    return _save(fig, name)


def asymptotic(df: pd.DataFrame, name: str) -> Path:
    """df: colunas n, cenario, parametro, media, hpd_inf, hpd_sup (+ EMV)."""
    params = df["parametro"].unique()
    fig, axes = plt.subplots(2, len(params), figsize=(2.7 * len(params), 5), sharex=True)
    for j, pname in enumerate(params):
        sub = df[df.parametro == pname]
        for i, (cen, s) in enumerate(sub.groupby("cenario", sort=False)):
            # EMV e Bayes-vaga coincidem (resultado teórico): EMV largo por baixo, vaga tracejada por cima
            sty = {"lw": 3.2, "alpha": 0.55} if i == 0 else ({"ls": "--"} if i == 1 else {})
            axes[0, j].fill_between(
                s["n"], s["hpd_inf"], s["hpd_sup"], color=C[i], alpha=0.12, lw=0
            )
            axes[0, j].plot(s["n"], s["media"], marker="o", ms=3, color=C[i], label=cen, **sty)
            axes[1, j].plot(
                s["n"], s["hpd_sup"] - s["hpd_inf"], marker="o", ms=3, color=C[i], label=cen, **sty
            )
        axes[0, j].set_title(PARAM_LABELS[pname], fontsize=9)
        axes[1, j].set_xscale("log")
        axes[1, j].set_yscale("log")
        axes[1, j].set_xlabel("n (log)")
    axes[0, 0].set_ylabel("estimativa e intervalo 95%")
    axes[1, 0].set_ylabel("amplitude do intervalo (log)")
    axes[0, -1].legend(fontsize=7)
    fig.suptitle("Dominância da verossimilhança: as abordagens convergem quando n cresce",
                 fontsize=10, fontweight="bold")  # fmt: skip
    return _save(fig, name)


# --------------------------------------------------------------------------- #
# Predição
# --------------------------------------------------------------------------- #
def predictive(
    draws_by_scen: dict[str, dict[str, np.ndarray]],
    exact_by_scen: dict,
    x_new: dict,
    mle,
    name: str,
) -> Path:
    labels = list(x_new)
    fig, axes = plt.subplots(1, len(labels), figsize=(2.9 * len(labels), 2.8), sharey=True)
    from ib_enem_bayes.predictive import hpd

    for j, lab in enumerate(labels):
        xs = np.linspace(100, 1000, 400)
        for i, (scen, dd) in enumerate(draws_by_scen.items()):
            yn = dd[lab]
            axes[j].hist(yn, bins=70, density=True, color=C[i], alpha=0.25, edgecolor="none")
            d = exact_by_scen[scen].predictive(np.asarray(x_new[lab]))
            axes[j].plot(xs, d.pdf(xs), color=C[i], lw=1.4, label=f"{scen} (t exata)")
            lo, hi = hpd(yn)
            axes[j].plot(
                [lo, hi], [-0.0003 * (i + 1)] * 2, color=C[i], lw=3, solid_capstyle="butt"
            )
        yhat = float(np.asarray(x_new[lab]) @ mle.beta_hat)
        axes[j].axvline(yhat, color=INK, lw=1, ls=":", label="EMV $x'\\hat\\beta$")
        axes[j].set_title(lab, fontsize=8.5)
        axes[j].set_xlabel("$Y_{new}$")
        axes[j].set_yticks([])
    axes[0].legend(fontsize=6.5, loc="upper left")
    fig.suptitle(
        "Preditiva a posteriori (histograma = Monte Carlo; barra = HPD 95%)",
        fontsize=10,
        fontweight="bold",
        y=1.03,
    )
    return _save(fig, name)


def ppc(y: np.ndarray, y_rep: np.ndarray, name: str) -> Path:
    fig, ax = plt.subplots(figsize=(5.5, 3))
    xs = np.linspace(np.percentile(y, 0.1) - 50, np.percentile(y, 99.9) + 50, 300)
    for r in y_rep[:100]:
        ax.plot(xs, stats.gaussian_kde(r)(xs), color=C[0], alpha=0.06, lw=0.8)
    ax.plot([], [], color=C[0], label="$Y^{rep}$ (100 réplicas)")
    ax.plot(xs, stats.gaussian_kde(y)(xs), color=INK, lw=1.8, label="$Y$ observado")
    ax.set(title="Checagem preditiva a posteriori", xlabel="nota MT", yticks=[])
    ax.legend()
    return _save(fig, name)


def forest(table: pd.DataFrame, name: str) -> Path:
    params = table["parametro"].unique()
    metodos = table["metodo"].unique()
    fig, axes = plt.subplots(1, len(params), figsize=(2.6 * len(params), 2.4))
    for j, pn in enumerate(params):
        s = table[table.parametro == pn]
        for i, m in enumerate(metodos):
            r = s[s.metodo == m].iloc[0]
            axes[j].plot([r["inf"], r["sup"]], [i, i], color=C[i], lw=2.5, solid_capstyle="round")
            axes[j].plot(r["pontual"], i, "o", color=C[i], ms=5, mec="white", mew=1)
        axes[j].set_yticks(
            range(len(metodos)), metodos if j == 0 else [""] * len(metodos), fontsize=7
        )
        axes[j].invert_yaxis()
        axes[j].set_title(PARAM_LABELS[pn], fontsize=9)
    return _save(fig, name)
