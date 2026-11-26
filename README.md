# ib-enem-bayes

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Trabalho final — Inferência Bayesiana.** Modelagem e predição bayesiana vs. abordagem
frequentista para a **nota de Matemática do ENEM**, com regressão linear e priori conjugada
**Normal–Gama(-Inversa)**.

$$
Y_i = \beta_0 + \beta_1 \log_2(\text{renda}_i/\text{SM}) + \beta_2\,\text{privada}_i + \beta_3\,\text{mãe\_superior}_i + \varepsilon_i,
\qquad \varepsilon_i \sim \mathcal N(0, \sigma^2)
$$

| Etapa do enunciado | Onde está |
|---|---|
| **A** — EMV, EP pela Fisher observada, IC 95%, predição clássica | `modeling/frequentist.py` |
| **B** — verossimilhança, priori NIG conjugada, cenário vago (Jeffreys) e informativo (power prior do ENEM anterior) | `modeling/conjugate.py` |
| **C** — Gibbs e Metropolis–Hastings do zero, PyMC/NUTS (e `.stan`), traceplots, densidades, ACF, R̂, ESS, exata × MCMC | `modeling/samplers.py`, `modeling/pymc_model.py`, `diagnostics.py` |
| **D** — preditiva a posteriori (forma fechada t e Monte Carlo por composição), HPD 95% | `predictive.py` |
| **E** — EMV × Bayes lado a lado, AIC/BIC × DIC/WAIC/LOO, estudo assintótico, IC × HPD | `compare.py`, `reports/relatorio.qmd` |

## Como reproduzir

```bash
make create_environment          # ou: python -m venv .venv && source .venv/bin/activate
make requirements                # instala dependências + o pacote em modo editável
make data                        # baixa e limpa os microdados ENEM 2022 e 2023 (~1,2 GB)
make train                       # Etapas A–E: tabelas, figuras e cadeias (~1 min)
make test                        # amostradores vs. posteriori analítica
make report                      # reports/relatorio.pdf e reports/slides.pdf (Quarto + LaTeX)
```

Para renderizar é preciso ter o [Quarto](https://quarto.org) e uma distribuição LaTeX com XeLaTeX
(TeX Live/MiKTeX ou `quarto install tinytex`).

**Se o download do INEP falhar** (é comum o site ficar instável ou dar erro de certificado), baixe
os zips pelo navegador em
<https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem>, salve como
`data/raw/microdados_enem_2022.zip` e `data/raw/microdados_enem_2023.zip` e rode `make data` de novo.
Se só o ano de 2023 estiver disponível, a priori informativa cai automaticamente para a elicitação
em `config.LITERATURE_PRIOR`. Nesse caso, preencha esses valores com referências.

**Teste sem baixar nada:** `make demo` gera dados *simulados* com o mesmo formato e roda tudo.
Relatório e slides saem com um aviso de "DADOS SIMULADOS"; **não entregue essa versão**.

Todas as escolhas do analista (tamanho da amostra, número de cadeias, `n0` da priori, perfis de
predição) estão em `ib_enem_bayes/config.py`.

## Organização do projeto

```
├── LICENSE
├── Makefile               <- make data | train | test | report | demo
├── README.md
├── data
│   ├── external
│   ├── interim            <- microdados filtrados (parquet)
│   ├── processed          <- Y e matriz de delineamento X
│   └── raw                <- zips originais do INEP (imutáveis)
├── docs                   <- mkdocs
├── models                 <- cadeias MCMC (.npy) e nig_regression.stan
├── notebooks              <- exploração interativa (numeração CCDS: 1.0-iniciais-descricao)
├── pyproject.toml
├── references             <- dicionário de dados do INEP, artigos das prioris
├── reports
│   ├── relatorio.qmd      <- relatório técnico (Quarto -> PDF)
│   ├── slides.qmd         <- apresentação (Quarto -> Beamer PDF)
│   ├── _common.py         <- leitura das tabelas geradas
│   ├── figures            <- figuras geradas por `make train`
│   └── tables             <- tabelas CSV + metadata.json
├── requirements.txt
├── tests                  <- amostradores vs. solução analítica
└── ib_enem_bayes
    ├── config.py          <- caminhos, sementes, hiperparâmetros, perfis X_new
    ├── dataset.py         <- download e limpeza (INEP)
    ├── features.py        <- construção de Y e X
    ├── simulate.py        <- dados sintéticos (somente teste)
    ├── modeling
    │   ├── frequentist.py <- Etapa A
    │   ├── conjugate.py   <- Etapa B + preditiva analítica
    │   ├── samplers.py    <- Gibbs e Metropolis–Hastings
    │   ├── pymc_model.py  <- NUTS (PyMC)
    │   ├── train.py       <- pipeline A–E
    │   └── predict.py     <- predição via linha de comando
    ├── diagnostics.py     <- R̂ (clássico e rank), ESS, ACF
    ├── predictive.py      <- preditiva a posteriori + HPD
    ├── compare.py         <- AIC, BIC, DIC, WAIC, LOO
    └── plots.py
```

Predição para um novo aluno, depois de `make train`:

```bash
python -m ib_enem_bayes.modeling.predict --renda-sm 3 --privada --mae-superior --cenario informativa
```
