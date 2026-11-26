r"""Etapa C — Protocolo de diagnóstico de convergência.

Estatísticas formais:
* R-hat clássico de Gelman–Rubin (1992), implementado à mão:
      W = média das variâncias intra-cadeia,  B/N = variância das médias das cadeias
      V_hat = (N-1)/N W + B/N,                 R_hat = sqrt(V_hat / W)
* R-hat "rank-normalized split" e ESS bulk/tail (Vehtari et al., 2021) via ArviZ.
* ESS = N_total / (1 + 2 sum_k rho_k)  — a ACF entra diretamente aqui.

Critérios usados no relatório: R_hat < 1.01 e ESS_bulk, ESS_tail > 400 para todos os parâmetros.
"""

from __future__ import annotations

import arviz as az
import numpy as np
import pandas as pd

from ib_enem_bayes.config import PARAM_NAMES


def to_datatree(draws: np.ndarray, log_lik: np.ndarray | None = None, y: np.ndarray | None = None):
    """draws (chains, draws, p+1) -> DataTree do ArviZ (grupos posterior/log_likelihood)."""
    p = draws.shape[-1] - 1
    data = {
        "posterior": {
            **{PARAM_NAMES[j]: draws[..., j] for j in range(p)},
            "sigma2": draws[..., p],
        }
    }
    if log_lik is not None:
        data["log_likelihood"] = {"y": log_lik}
        data["observed_data"] = {"y": y}
    return az.from_dict(data, dims={"y": ["obs"]} if log_lik is not None else None)


def gelman_rubin_classic(x: np.ndarray) -> float:
    """x: (chains, draws) de um único parâmetro."""
    _, n = x.shape
    chain_means = x.mean(axis=1)
    W = x.var(axis=1, ddof=1).mean()
    B_over_n = chain_means.var(ddof=1)
    V_hat = (n - 1) / n * W + B_over_n
    return float(np.sqrt(V_hat / W))


def acf(x: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Autocorrelação amostral de uma cadeia (via FFT)."""
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    f = np.fft.rfft(x, n=2 * n)
    r = np.fft.irfft(f * np.conj(f))[:n]
    return r[: max_lag + 1] / r[0]


def convergence_table(
    draws: np.ndarray, sampler: str, acceptance: np.ndarray | None = None
) -> pd.DataFrame:
    dt = to_datatree(draws)
    rhat = az.rhat(dt)
    ess_b = az.ess(dt, method="bulk")
    ess_t = az.ess(dt, method="tail")
    mcse = az.mcse(dt)
    names = PARAM_NAMES[: draws.shape[-1] - 1] + ["sigma2"]
    rows = []
    for j, name in enumerate(names):
        rows.append(
            {
                "amostrador": sampler,
                "parametro": name,
                "rhat_gelman_rubin": gelman_rubin_classic(draws[..., j]),
                "rhat_rank": float(rhat[name].values),
                "ess_bulk": float(ess_b[name].values),
                "ess_tail": float(ess_t[name].values),
                "mcse_media": float(mcse[name].values),
                "acf_lag1": float(
                    np.mean([acf(draws[c, :, j], 1)[1] for c in range(draws.shape[0])])
                ),
                "taxa_aceitacao": float(np.mean(acceptance)) if acceptance is not None else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    out["convergiu"] = (
        (out["rhat_rank"] < 1.01) & (out["ess_bulk"] > 400) & (out["ess_tail"] > 400)
    )
    return out


def posterior_summary(draws: np.ndarray, sampler: str, prob: float = 0.95) -> pd.DataFrame:
    """Média, mediana, DP e HPD a partir das amostras MCMC."""
    from ib_enem_bayes.predictive import hpd

    flat = draws.reshape(-1, draws.shape[-1])
    names = PARAM_NAMES[: flat.shape[1] - 1] + ["sigma2"]
    rows = {}
    for j, name in enumerate(names):
        lo, hi = hpd(flat[:, j], prob)
        rows[name] = {
            "media": flat[:, j].mean(),
            "mediana": np.median(flat[:, j]),
            "dp": flat[:, j].std(ddof=1),
            "hpd_inf": lo,
            "hpd_sup": hi,
        }
    return pd.DataFrame(rows).T.assign(amostrador=sampler)
