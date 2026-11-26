"""Aquisição e limpeza dos microdados do ENEM (INEP).

Uso:
    python -m ib_enem_bayes.dataset download --year 2023
    python -m ib_enem_bayes.dataset process  --year 2023
    python -m ib_enem_bayes.dataset all            # 2022 e 2023

Se o download automático falhar (o site do INEP costuma ser instável), baixe o
zip manualmente em https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem
e salve como data/raw/microdados_enem_<ano>.zip — o passo `process` lê o zip direto.
"""

from __future__ import annotations

import io
from pathlib import Path
import shutil
import urllib.request
import zipfile

from loguru import logger
import numpy as np
import pandas as pd
import typer

from ib_enem_bayes.config import (
    ENEM_COLUMNS,
    INEP_URL,
    INTERIM_DATA_DIR,
    RAW_DATA_DIR,
    YEAR_MAIN,
    YEAR_PRIOR,
)

app = typer.Typer(help="Download e limpeza dos microdados do ENEM.")


def raw_zip_path(year: int) -> Path:
    return RAW_DATA_DIR / f"microdados_enem_{year}.zip"


def interim_path(year: int) -> Path:
    return INTERIM_DATA_DIR / f"enem_{year}.parquet"


@app.command()
def download(year: int = YEAR_MAIN, force: bool = False) -> Path:
    """Baixa o zip de microdados do INEP para data/raw/."""
    dest = raw_zip_path(year)
    if dest.exists() and not force:
        logger.info(f"{dest.name} já existe; pulando download.")
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = INEP_URL.format(year=year)
    logger.info(f"Baixando {url} (~600 MB)...")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    # O certificado do INEP às vezes vem com cadeia incompleta; se der erro de
    # SSL, baixe manualmente pelo navegador (ver docstring do módulo).
    with urllib.request.urlopen(req) as r, open(dest, "wb") as f:
        shutil.copyfileobj(r, f)
    logger.success(f"Salvo em {dest}")
    return dest


def _find_csv(zf: zipfile.ZipFile, year: int) -> str:
    target = f"MICRODADOS_ENEM_{year}.csv".lower()
    for name in zf.namelist():
        if name.lower().endswith(target):
            return name
    raise FileNotFoundError(f"{target} não encontrado dentro do zip.")


def read_enem(year: int, chunksize: int = 500_000) -> pd.DataFrame:
    """Lê apenas as colunas necessárias do CSV dentro do zip (em blocos)."""
    src = raw_zip_path(year)
    if not src.exists():
        raise FileNotFoundError(
            f"{src} não existe. Rode `python -m ib_enem_bayes.dataset download --year {year}` "
            "ou coloque o zip manualmente em data/raw/."
        )
    parts = []
    with zipfile.ZipFile(src) as zf:
        csv_name = _find_csv(zf, year)
        logger.info(f"Lendo {csv_name}")
        with zf.open(csv_name) as fh:
            reader = pd.read_csv(
                io.TextIOWrapper(fh, encoding="latin-1"),
                sep=";",
                usecols=ENEM_COLUMNS,
                dtype={"Q002": "string", "Q006": "string"},
                chunksize=chunksize,
            )
            for chunk in reader:
                parts.append(clean(chunk))
    df = pd.concat(parts, ignore_index=True)
    df["ano"] = year
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Regras de inclusão (documentadas no relatório, seção A.1).

    1. Presente na prova de Matemática (TP_PRESENCA_MT == 1) e nota > 0;
    2. Tipo de escola declarado (pública = 2 ou privada = 3);
    3. Escolaridade da mãe conhecida (Q002 != 'H');
    4. Renda familiar informada (Q006 não nulo).
    """
    out = df.loc[
        (df["TP_PRESENCA_MT"] == 1)
        & (df["NU_NOTA_MT"] > 0)
        & (df["TP_ESCOLA"].isin([2, 3]))
        & (df["Q002"].notna())
        & (df["Q002"] != "H")
        & (df["Q006"].notna()),
        ["NU_NOTA_MT", "TP_ESCOLA", "Q002", "Q006"],
    ].copy()
    out["NU_NOTA_MT"] = out["NU_NOTA_MT"].astype(np.float64)
    out["TP_ESCOLA"] = out["TP_ESCOLA"].astype(np.int8)
    return out


@app.command()
def process(year: int = YEAR_MAIN) -> Path:
    """Gera data/interim/enem_<ano>.parquet com as linhas válidas."""
    df = read_enem(year)
    dest = interim_path(year)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    (INTERIM_DATA_DIR / "SIMULATED").unlink(missing_ok=True)  # dados reais substituem a simulação
    logger.success(f"{len(df):,} linhas válidas salvas em {dest}")
    return dest


@app.command("all")
def all_years() -> None:
    """Baixa e processa o ano principal e o ano da priori."""
    for year in (YEAR_PRIOR, YEAR_MAIN):
        try:
            download(year)
            process(year)
        except Exception as exc:
            logger.error(f"Falha no ano {year}: {exc}")
            if year == YEAR_MAIN:
                raise


if __name__ == "__main__":
    app()
