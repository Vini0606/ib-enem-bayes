"""Configuração central do projeto: caminhos, sementes, hiperparâmetros e cenários de predição.

Tudo o que é "escolha do analista" (tamanho da amostra, prioris, perfis de X_new)
fica aqui, para que o relatório cite exatamente os mesmos valores usados no código.
"""

from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

load_dotenv()

# --------------------------------------------------------------------------- #
# Caminhos (padrão Cookiecutter Data Science v2)
# --------------------------------------------------------------------------- #
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.debug(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"

# --------------------------------------------------------------------------- #
# Dados
# --------------------------------------------------------------------------- #
# Ano principal da análise e ano usado para construir a priori informativa.
# (A partir de 2024 o INEP separou questionário e notas em arquivos não
#  vinculáveis; por isso a análise usa 2023 e a priori usa 2022.)
YEAR_MAIN = 2023
YEAR_PRIOR = 2022

INEP_URL = "https://download.inep.gov.br/microdados/microdados_enem_{year}.zip"

# Colunas lidas dos microdados (o arquivo completo tem ~ 2 GB).
ENEM_COLUMNS = [
    "TP_PRESENCA_MT",  # 1 = presente na prova de Matemática
    "NU_NOTA_MT",  # Y: nota de Matemática e suas Tecnologias (escala TRI)
    "TP_ESCOLA",  # 1 = não respondeu, 2 = pública, 3 = privada
    "Q002",  # escolaridade da mãe (A..H)
    "Q006",  # renda mensal familiar em faixas (A..Q)
]

# Q006: ponto médio de cada faixa, em salários mínimos (SM).
# Usar SM (e não R$) torna as edições 2022 e 2023 comparáveis.
# A = "nenhuma renda" recebe 0.25 SM para permitir o log.
# Q = "acima de 20 SM" recebe 25 SM (convenção; documentar no relatório).
Q006_SM = {
    "A": 0.25, "B": 0.5, "C": 1.25, "D": 1.75, "E": 2.25, "F": 2.75,
    "G": 3.5, "H": 4.5, "I": 5.5, "J": 6.5, "K": 7.5, "L": 8.5,
    "M": 9.5, "N": 11.0, "O": 13.5, "P": 17.5, "Q": 25.0,
}  # fmt: skip

# Q002: F = superior completo, G = pós-graduação, H = "não sei" (excluído).
Q002_SUPERIOR = {"F", "G"}

# --------------------------------------------------------------------------- #
# Modelo
# --------------------------------------------------------------------------- #
RESPONSE = "nota_mt"
# Ordem das colunas da matriz de delineamento X (intercepto incluído).
FEATURES = ["intercepto", "log2_renda_sm", "escola_privada", "mae_superior"]
PARAM_NAMES = [f"beta_{i}" for i in range(len(FEATURES))] + ["sigma2"]
PARAM_LABELS = {
    "beta_0": r"$\beta_0$ (intercepto)",
    "beta_1": r"$\beta_1$ (log$_2$ renda)",
    "beta_2": r"$\beta_2$ (escola privada)",
    "beta_3": r"$\beta_3$ (mãe c/ superior)",
    "sigma2": r"$\sigma^2$",
}

SEED = 20261126
# Tamanho da amostra de análise (MCMC, WAIC, LOO). A posteriori analítica é
# calculada também para tamanhos crescentes (até a base completa) no estudo
# de dominância da verossimilhança.
SAMPLE_N = 5_000
ASYMPTOTIC_NS = [25, 50, 100, 250, 500, 1_000, 5_000, 25_000, 100_000, None]  # None = base toda

# MCMC
N_CHAINS = 4
N_DRAWS = 5_000
N_BURNIN = 1_000
THIN = 1

# --------------------------------------------------------------------------- #
# Prioris
# --------------------------------------------------------------------------- #
# Cenário (b) informativo — "power prior" a partir da edição anterior do ENEM:
# a posteriori (sob priori vaga) do ano YEAR_PRIOR é rebaixada para valer o
# equivalente a N0_PRIOR observações. Isso evita que uma base de milhões de
# linhas do ano anterior "engula" o ano atual e torna o peso da priori explícito.
N0_PRIOR = 500

# Fallback caso a base de YEAR_PRIOR não esteja disponível: elicitação manual.
# >>> SUBSTITUA pelos valores e referências da literatura que você citar. <<<
LITERATURE_PRIOR = {
    "m0": [480.0, 20.0, 50.0, 20.0],  # médias a priori de beta
    "sd0": [40.0, 10.0, 25.0, 15.0],  # desvios-padrão a priori de beta
    "sigma_guess": 95.0,  # palpite para sigma (pontos)
    "n0": N0_PRIOR,  # peso equivalente em observações
}

# --------------------------------------------------------------------------- #
# Perfis para predição (X_new). log2_renda_sm = 0 => 1 SM; 2 => 4 SM; etc.
# --------------------------------------------------------------------------- #
X_NEW = {
    "Pública, 1 SM, mãe s/ sup.": [1.0, 0.0, 0.0, 0.0],
    "Pública, 4 SM, mãe c/ sup.": [1.0, 2.0, 0.0, 1.0],
    "Privada, 4 SM, mãe s/ sup.": [1.0, 2.0, 1.0, 0.0],
    "Privada, 16 SM, mãe c/ sup.": [1.0, 4.0, 1.0, 1.0],
}

try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True, level="INFO")
except ModuleNotFoundError:
    pass
