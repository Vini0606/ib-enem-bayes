r"""Etapa C — o mesmo modelo em um pacote estruturado (PyMC, amostrador NUTS/HMC).

Serve de "terceira opinião" ao Gibbs e ao Metropolis–Hastings implementados à mão.
(Stan via cmdstanpy é equivalente; o arquivo models/nig_regression.stan traz a versão Stan.)
"""

from __future__ import annotations

import numpy as np

from ib_enem_bayes.modeling.conjugate import NIG


def fit_pymc(
    y: np.ndarray,
    X: np.ndarray,
    prior: NIG,
    n_draws: int,
    n_burnin: int,
    n_chains: int,
    seed: int,
) -> np.ndarray:
    """Retorna draws (chains, draws, p+1) com colunas (beta..., sigma^2)."""
    import pymc as pm

    p = X.shape[1]
    with pm.Model():
        if prior.is_proper:
            sigma2 = pm.InverseGamma("sigma2", alpha=prior.a, beta=prior.b)
            cov0 = np.linalg.inv(prior.Lambda)
            beta = pm.MvNormal("beta", mu=prior.m, cov=sigma2 * cov0, shape=p)
        else:  # Jeffreys: p(beta, sigma^2) ∝ 1/sigma^2
            sigma2 = pm.HalfFlat("sigma2")
            beta = pm.Flat("beta", shape=p)
            pm.Potential("jeffreys", -pm.math.log(sigma2))
        pm.Normal("y", mu=pm.math.dot(X, beta), sigma=pm.math.sqrt(sigma2), observed=y)
        idata = pm.sample(
            draws=n_draws,
            tune=n_burnin,
            chains=n_chains,
            cores=1,
            random_seed=seed,
            progressbar=False,
            compute_convergence_checks=False,
        )
    post = idata.posterior if hasattr(idata, "posterior") else idata["posterior"]
    b = np.asarray(post["beta"])
    s2 = np.asarray(post["sigma2"])
    return np.concatenate([b, s2[..., None]], axis=-1)
