"""Gera dados SIMULADOS no mesmo formato dos microdados limpos do INEP.

Serve apenas para testar o pipeline (testes automatizados e `make demo`) sem
baixar ~1 GB do INEP. Os parâmetros verdadeiros são conhecidos, o que permite
checar se os estimadores recuperam os valores corretos.

NUNCA use estes dados no relatório final: quando eles estão em uso, o arquivo
data/interim/SIMULATED existe e o relatório exibe um aviso em destaque.
"""

from __future__ import annotations

from loguru import logger
import numpy as np
import pandas as pd
import typer

from ib_enem_bayes.config import INTERIM_DATA_DIR, Q006_SM, SEED, YEAR_MAIN, YEAR_PRIOR

app = typer.Typer(help="Dados simulados para teste do pipeline.")

TRUE_PARAMS = {
    YEAR_PRIOR: {"beta": [472.0, 19.0, 48.0, 17.0], "sigma": 96.0},
    YEAR_MAIN: {"beta": [478.0, 18.0, 44.0, 19.0], "sigma": 94.0},
}

SIM_FLAG = INTERIM_DATA_DIR / "SIMULATED"


def simulate_raw(n: int, year: int, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed + year)
    letters = np.array(list(Q006_SM))
    # distribuição de renda assimétrica (muitas famílias até 2 SM)
    w = np.exp(-0.32 * np.arange(len(letters)))
    w[0] *= 0.4
    q006 = rng.choice(letters, size=n, p=w / w.sum())
    log2r = np.log2(np.vectorize(Q006_SM.get)(q006))
    p_priv = 1 / (1 + np.exp(-(-2.2 + 0.75 * log2r)))
    priv = rng.random(n) < p_priv
    p_mae = 1 / (1 + np.exp(-(-1.8 + 0.6 * log2r + 0.5 * priv)))
    mae_sup = rng.random(n) < p_mae
    q002 = np.where(mae_sup, rng.choice(["F", "G"], n, p=[0.75, 0.25]),
                    rng.choice(list("ABCDE"), n, p=[0.05, 0.2, 0.15, 0.2, 0.4]))  # fmt: skip
    b, s = TRUE_PARAMS[year]["beta"], TRUE_PARAMS[year]["sigma"]
    mu = b[0] + b[1] * log2r + b[2] * priv + b[3] * mae_sup
    nota = mu + rng.normal(0, s, n)
    return pd.DataFrame(
        {
            "NU_NOTA_MT": nota.round(1),
            "TP_ESCOLA": np.where(priv, 3, 2).astype(np.int8),
            "Q002": q002,
            "Q006": q006,
            "ano": year,
        }
    )


@app.command()
def main(n: int = 200_000) -> None:
    INTERIM_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for year in (YEAR_PRIOR, YEAR_MAIN):
        df = simulate_raw(n, year)
        df.to_parquet(INTERIM_DATA_DIR / f"enem_{year}.parquet", index=False)
        logger.warning(f"[SIMULADO] {year}: {n:,} linhas")
    SIM_FLAG.write_text("Dados simulados - não usar no relatório final.\n")


if __name__ == "__main__":
    app()
