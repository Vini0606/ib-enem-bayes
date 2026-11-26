r"""Etapa A — Abordagem frequentista (Máxima Verossimilhança).

Modelo amostral:  Y | X, beta, sigma^2 ~ N_n(X beta, sigma^2 I_n)

Log-verossimilhança:
    l(beta, sigma^2) = -n/2 log(2 pi sigma^2) - (Y - X beta)'(Y - X beta) / (2 sigma^2)

EMV (forma fechada):
    beta_hat     = (X'X)^{-1} X'Y
    sigma2_hat   = RSS / n           (viesado; s^2 = RSS/(n-p) é o não viesado)

Informação de Fisher observada, avaliada no EMV (theta = (beta, sigma^2)):
    J(theta_hat) = [[ X'X / sigma2_hat ,          0            ],
                    [        0         , n / (2 sigma2_hat^2)  ]]

Erros-padrão: raiz da diagonal de J^{-1}. O código também calcula J por
diferenças finitas da log-verossimilhança, para conferir a forma fechada.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from ib_enem_bayes.config import PARAM_NAMES


def loglik(theta: np.ndarray, y: np.ndarray, X: np.ndarray) -> float:
    """Log-verossimilhança em theta = (beta_0..beta_{p-1}, sigma^2)."""
    beta, s2 = theta[:-1], theta[-1]
    r = y - X @ beta
    n = len(y)
    return -0.5 * n * np.log(2 * np.pi * s2) - 0.5 * (r @ r) / s2


def numerical_hessian(f, x: np.ndarray, rel_step: float = 1e-4) -> np.ndarray:
    """Hessiana por diferenças centrais (para conferir a informação de Fisher)."""
    k = len(x)
    H = np.zeros((k, k))
    h = rel_step * np.maximum(np.abs(x), 1.0)
    for i in range(k):
        for j in range(i, k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h[i], h[j]
            H[i, j] = (f(x + ei + ej) - f(x + ei - ej) - f(x - ei + ej) + f(x - ei - ej)) / (
                4 * h[i] * h[j]
            )
            H[j, i] = H[i, j]
    return H


@dataclass
class MLEResult:
    beta_hat: np.ndarray
    sigma2_hat: np.ndarray  # EMV (RSS/n)
    s2: float  # não viesado (RSS/(n-p))
    fisher_obs: np.ndarray
    fisher_numeric: np.ndarray
    cov: np.ndarray
    n: int
    p: int
    loglik_max: float
    rss: float  # SQR = soma dos quadrados dos resíduos
    tss: float  # SQT = soma dos quadrados em torno da média de Y
    XtX_inv: np.ndarray = field(repr=False)

    @property
    def theta_hat(self) -> np.ndarray:
        return np.append(self.beta_hat, self.sigma2_hat)

    @property
    def se(self) -> np.ndarray:
        return np.sqrt(np.diag(self.cov))

    @property
    def aic(self) -> float:
        return -2 * self.loglik_max + 2 * (self.p + 1)

    @property
    def bic(self) -> float:
        return -2 * self.loglik_max + np.log(self.n) * (self.p + 1)

    @property
    def r2(self) -> float:
        """Coeficiente de determinação R² = 1 - SQR/SQT (modelo com intercepto)."""
        return 1 - self.rss / self.tss

    @property
    def r2_adj(self) -> float:
        """R² ajustado = 1 - [SQR/(n-p)] / [SQT/(n-1)]."""
        return 1 - (self.rss / (self.n - self.p)) / (self.tss / (self.n - 1))

    def fit_table(self) -> pd.DataFrame:
        """Medidas globais de ajuste do modelo."""
        return pd.DataFrame(
            {
                "valor": [
                    self.n,
                    self.r2,
                    self.r2_adj,
                    np.sqrt(self.sigma2_hat),
                    np.sqrt(self.tss / (self.n - 1)),
                ]
            },
            index=["n", "r2", "r2_ajustado", "sigma_hat", "dp_y"],
        )

    def table(self, level: float = 0.95) -> pd.DataFrame:
        """Estimativas, EP (Fisher observada) e IC de Wald a 95%."""
        z = stats.norm.ppf(0.5 + level / 2)
        th, se = self.theta_hat, self.se
        return pd.DataFrame(
            {
                "estimativa": th,
                "ep_fisher": se,
                "ic_inf": th - z * se,
                "ic_sup": th + z * se,
            },
            index=PARAM_NAMES[: self.p] + ["sigma2"],
        )

    def predict(self, x_new: np.ndarray, level: float = 0.95) -> pd.DataFrame:
        """Predição pontual clássica x'beta_hat, IC da média e intervalo de predição.

        IC da média:        x'b ± t_{n-p} s sqrt(x'(X'X)^{-1}x)
        Int. de predição:   x'b ± t_{n-p} s sqrt(1 + x'(X'X)^{-1}x)
        """
        x_new = np.atleast_2d(x_new)
        t = stats.t.ppf(0.5 + level / 2, self.n - self.p)
        s = np.sqrt(self.s2)
        yhat = x_new @ self.beta_hat
        h = np.einsum("ij,jk,ik->i", x_new, self.XtX_inv, x_new)
        return pd.DataFrame(
            {
                "y_hat": yhat,
                "ic_media_inf": yhat - t * s * np.sqrt(h),
                "ic_media_sup": yhat + t * s * np.sqrt(h),
                "ip_inf": yhat - t * s * np.sqrt(1 + h),
                "ip_sup": yhat + t * s * np.sqrt(1 + h),
            }
        )

    def pointwise_loglik(self, y: np.ndarray, X: np.ndarray) -> np.ndarray:
        r = y - X @ self.beta_hat
        return stats.norm.logpdf(r, scale=np.sqrt(self.sigma2_hat))


def fit_mle(y: np.ndarray, X: np.ndarray) -> MLEResult:
    n, p = X.shape
    XtX = X.T @ X
    XtX_inv = np.linalg.inv(XtX)
    beta_hat = XtX_inv @ X.T @ y
    rss = float(np.sum((y - X @ beta_hat) ** 2))
    sigma2_hat = rss / n
    J = np.zeros((p + 1, p + 1))
    J[:p, :p] = XtX / sigma2_hat
    J[p, p] = n / (2 * sigma2_hat**2)
    theta = np.append(beta_hat, sigma2_hat)
    J_num = -numerical_hessian(lambda t: loglik(t, y, X), theta)
    return MLEResult(
        beta_hat=beta_hat,
        sigma2_hat=sigma2_hat,
        s2=rss / (n - p),
        fisher_obs=J,
        fisher_numeric=J_num,
        cov=np.linalg.inv(J),
        n=n,
        p=p,
        loglik_max=loglik(theta, y, X),
        rss=rss,
        tss=float(np.sum((y - y.mean()) ** 2)),
        XtX_inv=XtX_inv,
    )
