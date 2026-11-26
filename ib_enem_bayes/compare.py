r"""Etapa E — Critérios de ajuste: frequentistas (AIC, BIC) vs. bayesianos (DIC, WAIC, LOO-CV).

AIC  = -2 l(θ_hat) + 2k                    BIC = -2 l(θ_hat) + k log n          (k = p + 1)

Com a matriz de log-verossimilhança pontual  ℓ_{si} = log p(y_i | θ^(s)):
    lppd   = Σ_i log( (1/S) Σ_s exp ℓ_{si} )
    p_WAIC = Σ_i Var_s(ℓ_{si})
    WAIC   = -2 (lppd - p_WAIC)                  (Watanabe, 2010; Gelman et al., 2014)
    LOO    = -2 elpd_loo, com PSIS (Vehtari, Gelman & Gabry, 2017) — via ArviZ
    DIC    = -2 log p(y | θ_bar) + 2 p_D,  p_D = 2[log p(y|θ_bar) - mean_s log p(y|θ^(s))]

Todos na escala de "deviance" (-2 × log), menor = melhor.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd
from scipy import special, stats

from ib_enem_bayes.diagnostics import to_datatree


def pointwise_loglik(
    draws: np.ndarray, y: np.ndarray, X: np.ndarray, max_draws: int = 4000
) -> np.ndarray:
    """(chains, draws', n) com draws' ≤ max_draws/chains (afinamento só para memória)."""
    c, d, _ = draws.shape
    step = max(1, (c * d) // max_draws)
    sub = draws[:, ::step]
    beta, s2 = sub[..., :-1], sub[..., -1]
    mu = beta @ X.T  # (c, d', n)
    return stats.norm.logpdf(y, loc=mu, scale=np.sqrt(s2)[..., None])


def waic(ll: np.ndarray) -> dict:
    ll2 = ll.reshape(-1, ll.shape[-1])
    S = ll2.shape[0]
    lppd_i = special.logsumexp(ll2, axis=0) - np.log(S)
    pw_i = ll2.var(axis=0, ddof=1)
    elpd_i = lppd_i - pw_i
    n = len(elpd_i)
    return {
        "elpd_waic": elpd_i.sum(),
        "p_waic": pw_i.sum(),
        "waic": -2 * elpd_i.sum(),
        "se_waic": 2 * np.sqrt(n * elpd_i.var(ddof=1)),
    }


def loo(ll: np.ndarray, y: np.ndarray) -> dict:
    dt = to_datatree(np.zeros(ll.shape[:2] + (2,)), log_lik=ll, y=y)
    res = az.loo(dt)
    k = np.asarray(res.pareto_k).ravel()
    return {
        "elpd_loo": float(res.elpd),
        "p_loo": float(res.p),
        "looic": -2 * float(res.elpd),
        "se_looic": 2 * float(res.se),
        "pareto_k_max": float(np.nanmax(k)),
    }


def dic(draws: np.ndarray, ll: np.ndarray) -> dict:
    """ll é calculada sobre as mesmas amostras; θ_bar = média a posteriori."""
    flat = draws.reshape(-1, draws.shape[-1])
    theta_bar = flat.mean(axis=0)
    return {"theta_bar": theta_bar, "mean_loglik": ll.reshape(-1, ll.shape[-1]).sum(axis=1).mean()}


def dic_value(theta_bar: np.ndarray, mean_loglik: float, y: np.ndarray, X: np.ndarray) -> dict:
    mu = X @ theta_bar[:-1]
    ll_bar = stats.norm.logpdf(y, mu, np.sqrt(theta_bar[-1])).sum()
    pD = 2 * (ll_bar - mean_loglik)
    return {"dic": -2 * ll_bar + 2 * pD, "p_dic": pD}


def information_criteria(draws: np.ndarray, y: np.ndarray, X: np.ndarray, cenario: str) -> dict:
    ll = pointwise_loglik(draws, y, X)
    w = waic(ll)
    lo = loo(ll, y)
    d = dic(draws, ll)
    dv = dic_value(d["theta_bar"], d["mean_loglik"], y, X)
    return {"cenario": cenario, **w, **lo, **dv}


def side_by_side(mle_table: pd.DataFrame, bayes_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Tabela lado a lado: EMV + IC 95% vs. média/mediana a posteriori + HPD 95%."""
    parts = [
        mle_table.rename(columns={"estimativa": "pontual", "ic_inf": "inf", "ic_sup": "sup"})[
            ["pontual", "inf", "sup"]
        ].assign(metodo="EMV + IC 95% (Wald)")
    ]
    for name, t in bayes_tables.items():
        parts.append(
            t.rename(columns={"media": "pontual", "hpd_inf": "inf", "hpd_sup": "sup"})[
                ["pontual", "mediana", "inf", "sup"]
            ].assign(metodo=f"{name}: média + HPD 95%")
        )
    out = pd.concat(parts).reset_index(names="parametro")
    out["amplitude"] = out["sup"] - out["inf"]
    return out[["parametro", "metodo", "pontual", "mediana", "inf", "sup", "amplitude"]]
