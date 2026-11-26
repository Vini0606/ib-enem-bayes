"""Predição para um novo aluno a partir das cadeias salvas (Etapa D, linha de comando).

Exemplo:
    python -m ib_enem_bayes.modeling.predict --renda-sm 3 --privada --mae-superior
"""

from __future__ import annotations

import numpy as np
import typer

from ib_enem_bayes.config import MODELS_DIR, SEED
from ib_enem_bayes.predictive import hpd, posterior_predictive_draws

app = typer.Typer()


@app.command()
def main(
    renda_sm: float = typer.Option(1.0, help="Renda familiar em salários mínimos."),
    privada: bool = typer.Option(False, help="Escola privada?"),
    mae_superior: bool = typer.Option(False, help="Mãe com ensino superior?"),
    cenario: str = typer.Option("vaga", help="vaga | informativa"),
) -> None:
    theta = np.load(MODELS_DIR / f"draws_gibbs_{cenario}.npy")
    theta = theta.reshape(-1, theta.shape[-1])
    x = np.array([1.0, np.log2(renda_sm), float(privada), float(mae_superior)])
    yn = posterior_predictive_draws(theta, x, np.random.default_rng(SEED))
    lo, hi = hpd(yn)
    typer.echo(
        f"E[Y_new|Y] = {yn.mean():.1f}  mediana = {np.median(yn):.1f}  HPD95% = [{lo:.1f}, {hi:.1f}]"
    )


if __name__ == "__main__":
    app()
