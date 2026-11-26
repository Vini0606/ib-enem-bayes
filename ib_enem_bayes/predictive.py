r"""Etapa D — Distribuição preditiva a posteriori.

    p(Y_new | Y, X) = ∫_Θ p(Y_new | x_new, θ) p(θ | Y, X) dθ

Avaliação por Monte Carlo (método da composição): para s = 1..S,
    θ^(s) ~ p(θ | Y, X)                         (amostra MCMC ou exata)
    Y_new^(s) ~ N(x_new' β^(s), σ^2(s))          (verossimilhança no novo ponto)
Os Y_new^(s) são amostras da mistura acima — é a integral sendo calculada,
pois (1/S) Σ_s p(y | x_new, θ^(s)) → ∫ p(y | x_new, θ) p(θ|Y) dθ.

O resultado é comparado com a forma fechada t_{2a_n} (ver conjugate.NIG.predictive).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ib_enem_bayes.modeling.conjugate import NIG


def hpd(samples: np.ndarray, prob: float = 0.95) -> tuple[float, float]:
    """Intervalo HPD empírico: o menor intervalo que contém `prob` das amostras
    (Chen & Shao, 1999). Válido para distribuições unimodais."""
    x = np.sort(np.asarray(samples))
    n = len(x)
    k = int(np.floor(prob * n))
    widths = x[k:] - x[: n - k]
    i = int(np.argmin(widths))
    return float(x[i]), float(x[i + k])


def posterior_predictive_draws(
    theta: np.ndarray, x_new: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """theta: (S, p+1) amostras da posteriori; x_new: (p,). Retorna S amostras de Y_new."""
    beta, s2 = theta[:, :-1], theta[:, -1]
    mu = beta @ np.asarray(x_new, float)
    return mu + np.sqrt(s2) * rng.standard_normal(len(s2))


def predictive_table(
    theta: np.ndarray,
    x_new: dict[str, list[float]],
    rng: np.random.Generator,
    cenario: str,
    exact: NIG | None = None,
    prob: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, np.ndarray]]:
    rows, draws = [], {}
    for label, x in x_new.items():
        yn = posterior_predictive_draws(theta, x, rng)
        draws[label] = yn
        lo, hi = hpd(yn, prob)
        row = {
            "cenario": cenario,
            "perfil": label,
            "media_mc": yn.mean(),
            "mediana_mc": np.median(yn),
            "hpd_inf_mc": lo,
            "hpd_sup_mc": hi,
        }
        if exact is not None:
            d = exact.predictive(np.asarray(x, float))
            lo_e, hi_e = d.interval(prob)  # t simétrica: HPD = intervalo central
            row.update({"media_exata": d.mean(), "hpd_inf_exato": lo_e, "hpd_sup_exato": hi_e})
        rows.append(row)
    return pd.DataFrame(rows), draws


def posterior_predictive_check(
    theta: np.ndarray, X: np.ndarray, y: np.ndarray, rng, n_rep: int = 200
):
    """Réplicas Y_rep ~ p(Y_rep | Y) para checagem gráfica do ajuste (PPC)."""
    idx = rng.choice(len(theta), size=n_rep, replace=False)
    beta, s2 = theta[idx, :-1], theta[idx, -1]
    mu = beta @ X.T
    return mu + np.sqrt(s2)[:, None] * rng.standard_normal(mu.shape)
