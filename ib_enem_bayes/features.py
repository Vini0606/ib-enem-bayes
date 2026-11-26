"""Construção da variável resposta Y e da matriz de delineamento X.

Modelo:  Y_i = beta_0 + beta_1 * log2(renda_i / 1 SM) + beta_2 * privada_i
               + beta_3 * mae_superior_i + eps_i,   eps_i ~ N(0, sigma^2)

Interpretação:
    beta_0: nota esperada de aluno de escola pública, renda familiar de 1 SM,
            mãe sem ensino superior;
    beta_1: variação esperada na nota quando a renda familiar DOBRA;
    beta_2: diferença esperada privada - pública (mantidos os demais fixos);
    beta_3: diferença esperada mãe com superior - sem superior.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
import typer

from ib_enem_bayes.config import (
    FEATURES,
    INTERIM_DATA_DIR,
    PROCESSED_DATA_DIR,
    Q002_SUPERIOR,
    Q006_SM,
    RESPONSE,
    YEAR_MAIN,
    YEAR_PRIOR,
)

app = typer.Typer(help="Gera a matriz de delineamento.")


def processed_path(year: int) -> Path:
    return PROCESSED_DATA_DIR / f"enem_{year}_modelo.parquet"


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Converte as colunas brutas do INEP em (Y, X)."""
    out = pd.DataFrame(index=df.index)
    out[RESPONSE] = df["NU_NOTA_MT"].astype(float)
    out["intercepto"] = 1.0
    renda_sm = df["Q006"].map(Q006_SM).astype(float)
    out["log2_renda_sm"] = np.log2(renda_sm)
    out["escola_privada"] = (df["TP_ESCOLA"] == 3).astype(float)
    out["mae_superior"] = df["Q002"].isin(Q002_SUPERIOR).astype(float)
    return out.dropna().reset_index(drop=True)


def load_xy(year: int = YEAR_MAIN) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Retorna (y, X, df) prontos para os modelos."""
    df = pd.read_parquet(processed_path(year))
    return df[RESPONSE].to_numpy(), df[FEATURES].to_numpy(), df


def subsample(df: pd.DataFrame, n: int | None, seed: int) -> pd.DataFrame:
    if n is None or n >= len(df):
        return df
    return df.sample(n=n, random_state=seed).reset_index(drop=True)


@app.command()
def main(years: list[int] = typer.Option([YEAR_PRIOR, YEAR_MAIN])) -> None:  # noqa: B008
    for year in years:
        src = INTERIM_DATA_DIR / f"enem_{year}.parquet"
        if not src.exists():
            logger.warning(f"{src} não encontrado; pulando {year}.")
            continue
        feats = build_features(pd.read_parquet(src))
        dest = processed_path(year)
        dest.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(dest, index=False)
        logger.success(f"{year}: {len(feats):,} linhas -> {dest}")


if __name__ == "__main__":
    app()
