"""Pipeline completo de estimação (Etapas A–E). Gera tabelas em reports/tables,
figuras em reports/figures e as cadeias em models/.

Uso:
    python -m ib_enem_bayes.modeling.train            # configuração do config.py
    python -m ib_enem_bayes.modeling.train --fast     # rodada rápida (teste)
"""

from __future__ import annotations

import json
import time

from loguru import logger
import numpy as np
import pandas as pd
import typer

from ib_enem_bayes import compare, diagnostics, plots, predictive
from ib_enem_bayes.config import (
    ASYMPTOTIC_NS,
    FEATURES,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    N0_PRIOR,
    N_BURNIN,
    N_CHAINS,
    N_DRAWS,
    PARAM_NAMES,
    RESPONSE,
    SAMPLE_N,
    SEED,
    TABLES_DIR,
    X_NEW,
    YEAR_MAIN,
    YEAR_PRIOR,
)
from ib_enem_bayes.features import processed_path, subsample
from ib_enem_bayes.modeling.conjugate import (
    NIG,
    jeffreys_prior,
    literature_prior,
    power_prior_from_previous,
)
from ib_enem_bayes.modeling.frequentist import fit_mle
from ib_enem_bayes.modeling.pymc_model import fit_pymc
from ib_enem_bayes.modeling.samplers import gibbs, metropolis_hastings

app = typer.Typer()


def _save_table(df: pd.DataFrame, name: str) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(TABLES_DIR / f"{name}.csv", index=True)


def build_priors(p: int) -> tuple[dict[str, NIG], str]:
    prev = processed_path(YEAR_PRIOR)
    priors = {"Vaga": jeffreys_prior(p)}
    if prev.exists():
        dprev = pd.read_parquet(prev)
        priors["Informativa"] = power_prior_from_previous(
            dprev[RESPONSE].to_numpy(), dprev[FEATURES].to_numpy(), N0_PRIOR
        )
        fonte = f"ENEM {YEAR_PRIOR} (n = {len(dprev):,}), power prior com n0 = {N0_PRIOR}"
    else:
        logger.warning(f"{prev} ausente: usando priori de literatura (config.LITERATURE_PRIOR).")
        priors["Informativa"] = literature_prior()
        fonte = "Elicitação/literatura (config.LITERATURE_PRIOR)"
    return priors, fonte


def acceptance_from_chain(draws: np.ndarray) -> np.ndarray:
    moved = np.any(np.diff(draws, axis=1) != 0, axis=-1)
    return moved.mean(axis=1)


def asymptotic_study(df_full: pd.DataFrame, priors: dict[str, NIG]) -> pd.DataFrame:
    perm = df_full.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    rows = []
    for n in ASYMPTOTIC_NS:
        sub = perm if n is None else perm.iloc[:n]
        y, X = sub[RESPONSE].to_numpy(), sub[FEATURES].to_numpy()
        nn = len(sub)
        mle = fit_mle(y, X)
        mt = mle.table()
        for j, pn in enumerate(PARAM_NAMES[: X.shape[1]]):
            rows.append({"n": nn, "cenario": "EMV (IC 95%)", "parametro": pn, "media": mt.iloc[j, 0],
                         "hpd_inf": mt.iloc[j, 2], "hpd_sup": mt.iloc[j, 3]})  # fmt: skip
        for lab, pr in priors.items():
            post = pr.update(y, X)
            for j, pn in enumerate(PARAM_NAMES[: X.shape[1]]):
                d = post.beta_marginal(j)
                lo, hi = d.interval(0.95)
                rows.append({"n": nn, "cenario": f"Bayes — {lab}", "parametro": pn, "media": d.mean(),
                             "hpd_inf": lo, "hpd_sup": hi})  # fmt: skip
        if n is None or n >= len(perm):
            break
    return pd.DataFrame(rows)


@app.command()
def main(
    fast: bool = typer.Option(False, help="Poucas iterações, para testar o pipeline."),
    skip_pymc: bool = typer.Option(False, help="Não roda o PyMC (NUTS)."),
) -> None:
    t0 = time.time()
    n_draws, n_burn = (600, 200) if fast else (N_DRAWS, N_BURNIN)
    rng = np.random.default_rng(SEED)
    simulated = (INTERIM_DATA_DIR / "SIMULATED").exists()
    if simulated:
        logger.warning("DADOS SIMULADOS em uso — resultados só servem para testar o pipeline.")

    # ------------------------------------------------------------------ dados
    df_full = pd.read_parquet(processed_path(YEAR_MAIN))
    df = subsample(df_full, SAMPLE_N, SEED)
    y, X = df[RESPONSE].to_numpy(), df[FEATURES].to_numpy()
    n, p = X.shape
    plots.eda(df_full.sample(min(len(df_full), 200_000), random_state=SEED))
    _save_table(df.describe().T, "descritivas_amostra")

    # ------------------------------------------------------------------ Etapa A
    mle = fit_mle(y, X)
    _save_table(mle.table(), "A_emv")
    _save_table(
        pd.DataFrame(mle.fisher_obs, index=PARAM_NAMES, columns=PARAM_NAMES), "A_fisher_analitica"
    )
    _save_table(
        pd.DataFrame(mle.fisher_numeric, index=PARAM_NAMES, columns=PARAM_NAMES),
        "A_fisher_numerica",
    )
    Xn = np.array(list(X_NEW.values()))
    _save_table(mle.predict(Xn).set_index(pd.Index(list(X_NEW), name="perfil")), "A_predicao_emv")

    # ------------------------------------------------------------------ Etapa B
    priors, prior_source = build_priors(p)
    _save_table(priors["Informativa"].hyperparams(), "B_hiperparametros_informativa")
    exact = {lab: pr.update(y, X, name=f"Posteriori — {lab}") for lab, pr in priors.items()}
    for lab, post in exact.items():
        _save_table(post.summary(), f"B_posteriori_exata_{lab.lower()}")
        _save_table(
            pd.DataFrame({"m_n": post.m}, index=PARAM_NAMES[:p]).assign(a_n=post.a, b_n=post.b),
            f"B_hiperparametros_posteriori_{lab.lower()}",
        )
    plots.prior_likelihood_posterior(priors, exact, mle, "B_priori_verossimilhanca_posteriori")

    # ------------------------------------------------------------------ Etapa C
    conv, summaries, all_draws = [], [], {}
    for k, (lab, pr) in enumerate(priors.items()):
        seed = SEED + 100 * k
        logger.info(f"[{lab}] Gibbs")
        g_full = gibbs(y, X, pr, n_draws + n_burn, 0, N_CHAINS, seed)
        logger.info(f"[{lab}] Metropolis–Hastings")
        mh_full, _ = metropolis_hastings(y, X, pr, n_draws + n_burn, 0, N_CHAINS, seed + 1)
        plots.traceplot(
            g_full, f"C_trace_gibbs_{lab.lower()}", f"Gibbs, priori {lab}", burnin=n_burn
        )
        plots.traceplot(
            mh_full,
            f"C_trace_mh_{lab.lower()}",
            f"Metropolis–Hastings, priori {lab}",
            burnin=n_burn,
        )
        samplers = {"Gibbs": g_full[:, n_burn:], "Metropolis–Hastings": mh_full[:, n_burn:]}
        if not skip_pymc:
            logger.info(f"[{lab}] PyMC (NUTS)")
            samplers["PyMC (NUTS)"] = fit_pymc(
                y, X, pr, n_draws // (2 if not fast else 1), n_burn, N_CHAINS, seed + 2
            )
        for sname, d in samplers.items():
            acc = acceptance_from_chain(d) if sname == "Metropolis–Hastings" else None
            conv.append(diagnostics.convergence_table(d, sname, acc).assign(cenario=lab))
            summaries.append(diagnostics.posterior_summary(d, sname).assign(cenario=lab))
            np.save(
                MODELS_DIR
                / f"draws_{sname.split()[0].lower().replace('–', '_')}_{lab.lower()}.npy",
                d,
            )
        plots.acf_plot(samplers, f"C_acf_{lab.lower()}")
        plots.exact_vs_mcmc(exact[lab], samplers, f"C_exata_vs_mcmc_{lab.lower()}",
                            f"Posteriori exata vs. MCMC — priori {lab}")  # fmt: skip
        all_draws[lab] = samplers
    conv_df = pd.concat(conv, ignore_index=True)
    _save_table(conv_df, "C_convergencia")
    _save_table(pd.concat(summaries), "C_posteriori_mcmc")

    # ------------------------------------------------------------------ Etapa D
    pred_tabs, pred_draws = [], {}
    for lab in priors:
        theta = all_draws[lab]["Gibbs"].reshape(-1, p + 1)
        tab, dd = predictive.predictive_table(theta, X_NEW, rng, lab, exact=exact[lab])
        pred_tabs.append(tab)
        pred_draws[lab] = dd
    _save_table(pd.concat(pred_tabs, ignore_index=True), "D_preditiva")
    plots.predictive(pred_draws, exact, X_NEW, mle, "D_preditiva")
    theta_v = all_draws["Vaga"]["Gibbs"].reshape(-1, p + 1)
    plots.ppc(y, predictive.posterior_predictive_check(theta_v, X, y, rng), "D_ppc")

    # ------------------------------------------------------------------ Etapa E
    gibbs_tabs = {
        lab: diagnostics.posterior_summary(all_draws[lab]["Gibbs"], "Gibbs").drop(
            columns="amostrador"
        )
        for lab in priors
    }
    sbs = compare.side_by_side(mle.table(), gibbs_tabs)
    _save_table(sbs, "E_lado_a_lado")
    plots.forest(sbs[sbs.parametro != "sigma2"], "E_forest")
    ic = [
        compare.information_criteria(all_draws[lab]["Gibbs"], y, X, f"Bayes — {lab}")
        for lab in priors
    ]
    ic.append({"cenario": "EMV", "aic": mle.aic, "bic": mle.bic, "loglik_max": mle.loglik_max})
    ic_df = pd.DataFrame(ic)
    _save_table(ic_df.set_index("cenario"), "E_criterios_informacao")
    log_ev = {lab: pr.log_evidence(y, X) for lab, pr in priors.items()}

    asy = asymptotic_study(df_full, priors)
    _save_table(asy, "E_assintotico")
    plots.asymptotic(asy, "E_assintotico")

    meta = {
        "simulado": simulated,
        "ano_principal": YEAR_MAIN,
        "n_base_completa": len(df_full),
        "n_amostra": int(n),
        "fonte_priori_informativa": prior_source,
        "n_cadeias": N_CHAINS,
        "n_draws": n_draws,
        "n_burnin": n_burn,
        "log_evidencia": log_ev,
        "todos_convergiram": bool(conv_df["convergiu"].all()),
        "tempo_s": round(time.time() - t0, 1),
    }
    (TABLES_DIR / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False, default=float)
    )
    logger.success(f"Pipeline concluído em {meta['tempo_s']} s.")


if __name__ == "__main__":
    app()
