"""Utilitários compartilhados pelo relatório e pelos slides (carregam as saídas do pipeline)."""

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TAB = ROOT / "reports" / "tables"
FIG = "figures"  # relativo a reports/

META = json.loads((TAB / "metadata.json").read_text())

NOMES = {
    "beta_0": r"$\beta_0$",
    "beta_1": r"$\beta_1$",
    "beta_2": r"$\beta_2$",
    "beta_3": r"$\beta_3$",
    "sigma2": r"$\sigma^2$",
}


def tab(name: str, **kw) -> pd.DataFrame:
    return pd.read_csv(TAB / f"{name}.csv", index_col=kw.pop("index_col", 0), **kw)


def md(df: pd.DataFrame, floatfmt=".2f", index=True) -> str:
    df = df.rename(index=NOMES)
    for col in ("parametro", "Par."):
        if col in df.columns:
            df = df.assign(**{col: df[col].map(lambda s: NOMES.get(s, s))})
    return df.to_markdown(floatfmt=floatfmt, index=index)


def fmt(x, nd=2) -> str:
    """Número no padrão brasileiro (vírgula decimal)."""
    s = f"{x:,.{nd}f}"
    return s.replace(",", "X").replace(".", ",").replace("X", ".")
