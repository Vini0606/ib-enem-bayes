r"""Etapa C — Amostradores MCMC implementados do zero.

Ambos trabalham com as estatísticas suficientes (X'X, X'Y, Y'Y), de modo que o custo
por iteração não depende de n.

1) Amostrador de Gibbs (condicionais completas da priori NIG / Jeffreys):
       beta    | sigma^2, Y ~ N_p( m_n , sigma^2 Lambda_n^{-1} )
       sigma^2 | beta,    Y ~ GI( a0 + (n+p)/2 ,
                                  b0 + [ (Y-X beta)'(Y-X beta) + (beta-m0)' Lambda0 (beta-m0) ] / 2 )

2) Metropolis–Hastings de passeio aleatório em phi = (beta, log sigma^2):
       proposta  phi* ~ N(phi^{(t-1)}, c^2 Sigma),   c = 2.38/sqrt(d)  (Roberts et al., 1997)
       aceita com prob. min{1, pi(phi*)/pi(phi^{(t-1)})}  (proposta simétrica => razão de Hastings = 1)
       pi(phi) inclui o jacobiano |d sigma^2 / d log sigma^2| = sigma^2.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from tqdm import trange

from ib_enem_bayes.modeling.conjugate import NIG


@dataclass
class SuffStats:
    XtX: np.ndarray
    Xty: np.ndarray
    yty: float
    n: int

    @classmethod
    def from_data(cls, y: np.ndarray, X: np.ndarray) -> SuffStats:
        return cls(X.T @ X, X.T @ y, float(y @ y), len(y))

    def rss(self, beta: np.ndarray) -> float:
        return self.yty - 2 * beta @ self.Xty + beta @ self.XtX @ beta


def overdispersed_inits(
    prior_post: NIG, n_chains: int, rng: np.random.Generator, spread: float = 10.0
) -> np.ndarray:
    """Pontos iniciais dispersos (essenciais para o R-hat de Gelman–Rubin ter sentido)."""
    sd = np.sqrt(np.diag(prior_post.cov_beta_scale))
    beta0 = prior_post.m + spread * sd * rng.standard_normal((n_chains, prior_post.p))
    s2_0 = (prior_post.b / prior_post.a) * np.exp(rng.uniform(-2, 2, n_chains))
    return np.column_stack([beta0, s2_0])


def log_posterior(beta: np.ndarray, s2: float, ss: SuffStats, prior: NIG) -> float:
    """log p(beta, sigma^2 | Y) até uma constante."""
    if s2 <= 0:
        return -np.inf
    p = len(beta)
    d = beta - prior.m
    loglik = -0.5 * ss.n * np.log(s2) - 0.5 * ss.rss(beta) / s2
    logprior = -(prior.a + 1 + p / 2) * np.log(s2) - (prior.b + 0.5 * d @ prior.Lambda @ d) / s2
    return loglik + logprior


def gibbs(
    y: np.ndarray,
    X: np.ndarray,
    prior: NIG,
    n_draws: int,
    n_burnin: int,
    n_chains: int,
    seed: int,
    thin: int = 1,
) -> np.ndarray:
    """Retorna draws com shape (chains, draws, p+1); colunas (beta..., sigma^2)."""
    rng = np.random.default_rng(seed)
    ss = SuffStats.from_data(y, X)
    p = X.shape[1]
    post = prior.update(y, X)  # Lambda_n e m_n não dependem de sigma^2
    L_chol = np.linalg.cholesky(np.linalg.inv(post.Lambda))
    shape = prior.a + (ss.n + p) / 2
    inits = overdispersed_inits(post, n_chains, rng)
    total = n_burnin + n_draws * thin
    out = np.empty((n_chains, n_draws, p + 1))
    for c in range(n_chains):
        s2 = inits[c, -1]
        k = 0
        for t in trange(total, desc=f"Gibbs cadeia {c + 1}", leave=False):
            beta = post.m + np.sqrt(s2) * (L_chol @ rng.standard_normal(p))
            d = beta - prior.m
            rate = prior.b + 0.5 * (ss.rss(beta) + d @ prior.Lambda @ d)
            s2 = rate / rng.gamma(shape)  # GI(shape, rate) = rate / Gama(shape, 1)
            if t >= n_burnin and (t - n_burnin) % thin == 0:
                out[c, k] = np.append(beta, s2)
                k += 1
    return out


def gibbs_with_burnin(y, X, prior, n_iter: int, n_chains: int, seed: int) -> np.ndarray:
    """Mesma cadeia de Gibbs, mas SEM descartar o aquecimento (para o traceplot inicial)."""
    return gibbs(y, X, prior, n_draws=n_iter, n_burnin=0, n_chains=n_chains, seed=seed)


def metropolis_hastings(
    y: np.ndarray,
    X: np.ndarray,
    prior: NIG,
    n_draws: int,
    n_burnin: int,
    n_chains: int,
    seed: int,
    thin: int = 1,
    scale: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Retorna (draws (chains, draws, p+1), taxa de aceitação por cadeia)."""
    rng = np.random.default_rng(seed)
    ss = SuffStats.from_data(y, X)
    p = X.shape[1]
    d = p + 1
    post = prior.update(y, X)
    # Covariância da proposta: aproximação de Laplace em phi = (beta, log sigma^2)
    Sigma = np.zeros((d, d))
    Sigma[:p, :p] = post.cov_beta_scale
    Sigma[p, p] = 1.0 / post.a  # Var(log sigma^2 | Y) ≈ 1/a_n
    c = scale if scale is not None else 2.38 / np.sqrt(d)
    Lprop = np.linalg.cholesky(c**2 * Sigma)

    def logpi(phi):
        return log_posterior(phi[:p], np.exp(phi[p]), ss, prior) + phi[p]  # + jacobiano

    inits = overdispersed_inits(post, n_chains, rng)
    total = n_burnin + n_draws * thin
    out = np.empty((n_chains, n_draws, d))
    acc = np.zeros(n_chains)
    for ch in range(n_chains):
        phi = np.append(inits[ch, :p], np.log(inits[ch, p]))
        lp = logpi(phi)
        k = 0
        for t in trange(total, desc=f"MH cadeia {ch + 1}", leave=False):
            prop = phi + Lprop @ rng.standard_normal(d)
            lp_prop = logpi(prop)
            if np.log(rng.random()) < lp_prop - lp:
                phi, lp = prop, lp_prop
                if t >= n_burnin:
                    acc[ch] += 1
            if t >= n_burnin and (t - n_burnin) % thin == 0:
                out[ch, k] = np.append(phi[:p], np.exp(phi[p]))
                k += 1
    return out, acc / (total - n_burnin)
