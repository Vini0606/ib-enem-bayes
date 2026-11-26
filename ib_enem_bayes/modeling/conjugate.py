r"""Etapas B e D (parte analítica) — Modelo Normal–Gama (Normal–Gama-Inversa) conjugado.

Verossimilhança:
    p(Y | X, beta, sigma^2) = (2 pi sigma^2)^{-n/2} exp{ -(Y - X beta)'(Y - X beta) / (2 sigma^2) }

Priori conjugada NIG(m0, Lambda0, a0, b0)  [equivalente a Normal–Gama na precisão tau = 1/sigma^2]:
    beta | sigma^2 ~ N_p(m0, sigma^2 Lambda0^{-1})
    sigma^2        ~ GI(a0, b0)

Posteriori (Teorema de Bayes: p(theta|Y,X) ∝ p(Y|X,theta) p(theta)):
    Lambda_n = Lambda0 + X'X
    m_n      = Lambda_n^{-1} (Lambda0 m0 + X'Y)
    a_n      = a0 + n/2
    b_n      = b0 + (Y'Y + m0' Lambda0 m0 - m_n' Lambda_n m_n) / 2

Marginais a posteriori:
    sigma^2 | Y ~ GI(a_n, b_n)
    beta    | Y ~ t_{2 a_n}(m_n, (b_n/a_n) Lambda_n^{-1})          (t de Student multivariada)

Preditiva a posteriori (a integral de mistura tem forma fechada):
    p(Y_new | Y) = ∫ N(Y_new | x_new' beta, sigma^2) NIG(beta, sigma^2 | m_n, Lambda_n, a_n, b_n) d beta d sigma^2
                 = t_{2 a_n}( x_new' m_n , (b_n/a_n) (1 + x_new' Lambda_n^{-1} x_new) )

Priori de Jeffreys p(beta, sigma^2) ∝ 1/sigma^2 é o caso-limite Lambda0 = 0, a0 = -p/2, b0 = 0,
o que dá beta | Y ~ t_{n-p}(beta_hat, s^2 (X'X)^{-1}) — o mesmo centro do EMV.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import special, stats

from ib_enem_bayes.config import LITERATURE_PRIOR, PARAM_NAMES


@dataclass
class NIG:
    """Parâmetros de uma distribuição Normal–Gama-Inversa (priori ou posteriori)."""

    m: np.ndarray
    Lambda: np.ndarray  # matriz de PRECISÃO (a menos de sigma^2)
    a: float
    b: float
    name: str = ""

    @property
    def p(self) -> int:
        return len(self.m)

    @property
    def is_proper(self) -> bool:
        return self.a > 0 and self.b > 0 and np.all(np.linalg.eigvalsh(self.Lambda) > 0)

    # ------------------------------------------------------------------ #
    # Atualização bayesiana
    # ------------------------------------------------------------------ #
    def update(self, y: np.ndarray, X: np.ndarray, name: str | None = None) -> NIG:
        Ln = self.Lambda + X.T @ X
        mn = np.linalg.solve(Ln, self.Lambda @ self.m + X.T @ y)
        an = self.a + len(y) / 2
        bn = self.b + 0.5 * (y @ y + self.m @ self.Lambda @ self.m - mn @ Ln @ mn)
        return NIG(mn, Ln, an, bn, name or f"posteriori | {self.name}")

    # ------------------------------------------------------------------ #
    # Marginais
    # ------------------------------------------------------------------ #
    @property
    def cov_beta_scale(self) -> np.ndarray:
        """Matriz de escala da t marginal de beta: (b/a) Lambda^{-1}."""
        return (self.b / self.a) * np.linalg.inv(self.Lambda)

    @property
    def df(self) -> float:
        return 2 * self.a

    def beta_marginal(self, j: int) -> stats.rv_continuous:
        """beta_j | Y ~ t_{2a}(m_j, sqrt((b/a) [Lambda^{-1}]_jj))."""
        return stats.t(df=self.df, loc=self.m[j], scale=np.sqrt(self.cov_beta_scale[j, j]))

    def sigma2_marginal(self) -> stats.rv_continuous:
        return stats.invgamma(a=self.a, scale=self.b)

    def marginal(self, name: str) -> stats.rv_continuous:
        if name == "sigma2":
            return self.sigma2_marginal()
        return self.beta_marginal(PARAM_NAMES.index(name))

    def summary(self, level: float = 0.95) -> pd.DataFrame:
        """Média, mediana, DP e HPD exato. (t é simétrica => HPD = intervalo central;
        para a GI o HPD é obtido numericamente.)"""
        rows = {}
        for name in PARAM_NAMES[: self.p] + ["sigma2"]:
            d = self.marginal(name)
            lo, hi = hpd_unimodal(d, level) if name == "sigma2" else d.interval(level)
            rows[name] = {
                "media": d.mean(),
                "mediana": d.median(),
                "dp": d.std(),
                "hpd_inf": lo,
                "hpd_sup": hi,
            }
        return pd.DataFrame(rows).T

    # ------------------------------------------------------------------ #
    # Amostragem exata (Monte Carlo direto, sem cadeia de Markov)
    # ------------------------------------------------------------------ #
    def sample(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Retorna matriz (size, p+1) com colunas (beta_0..beta_{p-1}, sigma^2)."""
        s2 = stats.invgamma.rvs(a=self.a, scale=self.b, size=size, random_state=rng)
        L = np.linalg.cholesky(np.linalg.inv(self.Lambda))
        z = rng.standard_normal((size, self.p))
        beta = self.m + np.sqrt(s2)[:, None] * (z @ L.T)
        return np.column_stack([beta, s2])

    # ------------------------------------------------------------------ #
    # Preditiva a posteriori analítica
    # ------------------------------------------------------------------ #
    def predictive(self, x_new: np.ndarray) -> stats.rv_continuous:
        x_new = np.asarray(x_new, dtype=float)
        scale2 = (self.b / self.a) * (1 + x_new @ np.linalg.solve(self.Lambda, x_new))
        return stats.t(df=self.df, loc=x_new @ self.m, scale=np.sqrt(scale2))

    # ------------------------------------------------------------------ #
    # Evidência (verossimilhança marginal) — só para prioris próprias
    # ------------------------------------------------------------------ #
    def log_evidence(self, y: np.ndarray, X: np.ndarray) -> float:
        if not self.is_proper:
            return np.nan
        post = self.update(y, X)
        n = len(y)
        _, ld0 = np.linalg.slogdet(self.Lambda)
        _, ldn = np.linalg.slogdet(post.Lambda)
        return float(
            -n / 2 * np.log(2 * np.pi)
            + 0.5 * (ld0 - ldn)
            + self.a * np.log(self.b)
            - post.a * np.log(post.b)
            + special.gammaln(post.a)
            - special.gammaln(self.a)
        )

    def hyperparams(self) -> pd.DataFrame:
        """Tabela de hiperparâmetros para o relatório."""
        sd_beta = np.sqrt(np.diag(self.cov_beta_scale)) if self.is_proper else [np.inf] * self.p
        return pd.DataFrame(
            {"m0": self.m, "dp_marginal_beta": sd_beta},
            index=PARAM_NAMES[: self.p],
        ).assign(a0=self.a, b0=self.b)


def hpd_unimodal(dist: stats.rv_continuous, level: float = 0.95) -> tuple[float, float]:
    """HPD de densidade unimodal: menor intervalo [F^{-1}(u), F^{-1}(u+level)]."""
    from scipy.optimize import minimize_scalar

    res = minimize_scalar(
        lambda u: dist.ppf(u + level) - dist.ppf(u),
        bounds=(1e-9, 1 - level - 1e-9),
        method="bounded",
    )
    return float(dist.ppf(res.x)), float(dist.ppf(res.x + level))


# ---------------------------------------------------------------------- #
# Construtores de priori
# ---------------------------------------------------------------------- #
def jeffreys_prior(p: int) -> NIG:
    """Cenário (a): priori não informativa de Jeffreys, p(beta, sigma^2) ∝ 1/sigma^2 (imprópria)."""
    return NIG(np.zeros(p), np.zeros((p, p)), a=-p / 2, b=0.0, name="Jeffreys (vaga)")


def power_prior_from_previous(y_prev: np.ndarray, X_prev: np.ndarray, n0: int) -> NIG:
    """Cenário (b): priori informativa a partir da edição anterior do ENEM.

    1. Posteriori de Jeffreys com os dados do ano anterior:  NIG(m*, Λ*, a*, b*).
    2. "Power prior" com peso n0/n_prev:  a informação é rebaixada para equivaler a
       n0 observações (Ibrahim & Chen, 2000):
           m0 = m*,   Λ0 = (n0/n_prev) Λ*,   a0 = n0/2,   b0 = a0 · b*/a*.
    """
    p = X_prev.shape[1]
    post_prev = jeffreys_prior(p).update(y_prev, X_prev)
    w = n0 / len(y_prev)
    a0 = n0 / 2
    b0 = a0 * post_prev.b / post_prev.a
    return NIG(
        post_prev.m.copy(),
        w * post_prev.Lambda,
        a0,
        b0,
        name=f"Informativa (ENEM anterior, n0={n0})",
    )


def literature_prior(cfg: dict = LITERATURE_PRIOR) -> NIG:
    """Cenário (b) alternativo: elicitação direta (médias e DPs de beta + palpite de sigma).

    Escolhe-se a0 = n0/2 e b0 = a0·sigma_guess^2, e V0 = diag(sd0^2)/sigma_guess^2,
    de modo que DP marginal a priori de beta_j ≈ sd0_j.
    """
    m0 = np.asarray(cfg["m0"], float)
    sd0 = np.asarray(cfg["sd0"], float)
    a0 = cfg["n0"] / 2
    b0 = a0 * cfg["sigma_guess"] ** 2
    Lambda0 = np.diag(cfg["sigma_guess"] ** 2 / sd0**2)
    return NIG(m0, Lambda0, a0, b0, name="Informativa (literatura)")
