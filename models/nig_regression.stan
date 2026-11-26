// Regressão linear bayesiana com priori conjugada Normal-Gama-Inversa
// (versão Stan equivalente a ib_enem_bayes/modeling/pymc_model.py)
data {
  int<lower=1> n;
  int<lower=1> p;
  matrix[n, p] X;
  vector[n] y;
  vector[p] m0;
  cov_matrix[p] V0;        // = Lambda0^{-1}
  real<lower=0> a0;
  real<lower=0> b0;
  int<lower=0> n_new;
  matrix[n_new, p] X_new;
}
parameters {
  vector[p] beta;
  real<lower=0> sigma2;
}
model {
  sigma2 ~ inv_gamma(a0, b0);
  beta ~ multi_normal(m0, sigma2 * V0);
  y ~ normal(X * beta, sqrt(sigma2));
}
generated quantities {
  vector[n] log_lik;                 // para WAIC / LOO
  vector[n_new] y_new;               // preditiva a posteriori por composição
  for (i in 1:n) log_lik[i] = normal_lpdf(y[i] | X[i] * beta, sqrt(sigma2));
  for (j in 1:n_new) y_new[j] = normal_rng(X_new[j] * beta, sqrt(sigma2));
}
