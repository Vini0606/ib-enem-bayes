"""Testes de correção: os amostradores devem reproduzir a posteriori analítica."""

import numpy as np
import pytest

from ib_enem_bayes.modeling.conjugate import NIG, jeffreys_prior, literature_prior
from ib_enem_bayes.modeling.frequentist import fit_mle
from ib_enem_bayes.modeling.samplers import gibbs, metropolis_hastings
from ib_enem_bayes.predictive import hpd


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(42)
    n = 3000
    X = np.column_stack([np.ones(n), rng.normal(1, 1.5, n), rng.random(n) < 0.2, rng.random(n) < 0.3]).astype(float)
    beta = np.array([478.0, 18.0, 44.0, 19.0])
    y = X @ beta + rng.normal(0, 94, n)
    return y, X, beta


def test_mle_recovers_truth_and_fisher_matches_numeric(data):
    y, X, beta = data
    m = fit_mle(y, X)
    assert np.all(np.abs(m.beta_hat - beta) < 4 * m.se[:4])
    np.testing.assert_allclose(m.fisher_obs, m.fisher_numeric, rtol=1e-3, atol=1e-8)


def test_jeffreys_posterior_centered_at_mle(data):
    y, X, _ = data
    post = jeffreys_prior(4).update(y, X)
    m = fit_mle(y, X)
    np.testing.assert_allclose(post.m, m.beta_hat, rtol=1e-10)
    assert post.a == pytest.approx((len(y) - 4) / 2)
    assert post.b / post.a == pytest.approx(m.s2)  # escala da t = s^2 (X'X)^{-1}


@pytest.mark.parametrize("prior", [jeffreys_prior(4), literature_prior()])
def test_gibbs_matches_exact(data, prior: NIG):
    y, X, _ = data
    exact = prior.update(y, X)
    d = gibbs(y, X, prior, 3000, 300, 2, seed=1).reshape(-1, 5)
    sd = np.sqrt(np.diag(exact.cov_beta_scale))
    assert np.all(np.abs(d[:, :4].mean(0) - exact.m) < 0.1 * sd)
    assert d[:, 4].mean() == pytest.approx(exact.sigma2_marginal().mean(), rel=0.01)


def test_mh_matches_exact(data):
    y, X, _ = data
    prior = jeffreys_prior(4)
    exact = prior.update(y, X)
    d = metropolis_hastings(y, X, prior, 8000, 1000, 2, seed=2)[0].reshape(-1, 5)
    sd = np.sqrt(np.diag(exact.cov_beta_scale))
    assert np.all(np.abs(d[:, :4].mean(0) - exact.m) < 0.25 * sd)


def test_hpd_normal():
    x = np.random.default_rng(0).normal(size=200_000)
    lo, hi = hpd(x)
    assert lo == pytest.approx(-1.96, abs=0.03) and hi == pytest.approx(1.96, abs=0.03)
