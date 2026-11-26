# ib-enem-bayes

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

**Trabalho final — Inferência Bayesiana.** Modelagem e predição bayesiana vs. abordagem
frequentista para a **nota de Matemática do ENEM**, com regressão linear e priori conjugada
**Normal–Gama(-Inversa)**.

## Modelo

---

```math
Y_i = \beta_0 + \beta_1 \log_2\!\left(\frac{\text{renda}_i}{\text{SM}}\right) + \beta_2\,\text{privada}_i + \beta_3\,\text{mãe superior}_i + \varepsilon_i
```

<br>

```math
\varepsilon_i \sim \mathcal{N}(0, \sigma^2) \ \text{independentes}, \qquad i = 1, \dots, n
```

---

em que $Y_i$ é a nota de Matemática do participante $i$; $\text{renda}_i/\text{SM}$, a renda
familiar em salários mínimos; $\text{privada}_i$ e $\text{mãe superior}_i$, indicadoras de escola
privada e de mãe com ensino superior.

## Etapas do enunciado

| Etapa do enunciado | Onde está |
|---|---|
| **A** — EMV, EP pela Fisher observada, IC 95%, predição clássica | `modeling/frequentist.py` |
| **B** — verossimilhança, priori NIG conjugada, cenário vago (Jeffreys) e informativo (power prior do ENEM anterior) | `modeling/conjugate.py` |
| **C** — Gibbs e Metropolis–Hastings do zero, PyMC/NUTS (e `.stan`), traceplots, densidades, ACF, R̂, ESS, exata × MCMC | `modeling/samplers.py`, `modeling/pymc_model.py`, `diagnostics.py` |
| **D** — preditiva a posteriori (forma fechada t e Monte Carlo por composição), HPD 95% | `predictive.py` |
| **E** — EMV × Bayes lado a lado, AIC/BIC × DIC/WAIC/LOO, estudo assintótico, IC × HPD | `compare.py`, `reports/relatorio.qmd` |

## Entregáveis

- **[`reports/relatorio.pdf`](reports/relatorio.pdf)** é a nota técnica completa: resumo, introdução
  com o mapa Etapas A–E → seções, dados, métodos (equações numeradas), resultados, discussão,
  conclusões, referências ABNT e apêndices (diagnósticos completos, uso do pacote e código-fonte).
- **[`reports/slides.pdf`](reports/slides.pdf)** é a apresentação em linguagem de produto (títulos
  que já trazem a conclusão, números em destaque), seguida de um **apêndice técnico** com as fórmulas
  e os diagnósticos.

## Principais resultados

> Retrato da rodada com os microdados reais (ENEM 2023, priori do ENEM 2022). Os valores oficiais
> estão em `reports/tables/` e são lidos pelo relatório e pelos slides. Se o pipeline for rodado de
> novo, confira lá.

Base: 1.008.903 participantes válidos; análise MCMC numa amostra aleatória de n = 5.000.

| Parâmetro | Leitura | EMV (IC 95%) | Bayes, priori informativa (HPD 95%) |
|---|---|---|---|
| β₀ | nota esperada: escola pública, 1 SM, mãe sem superior | 500,0 (496,4; 503,6) | 500,6 (497,2; 504,0) |
| β₁ | variação quando a renda familiar **dobra** | +24,8 (22,3; 27,2) | +24,6 (22,4; 26,9) |
| β₂ | escola privada vs. pública | +63,7 (55,3; 72,2) | +62,0 (54,0; 70,0) |
| β₃ | mãe com ensino superior | +19,3 (11,7; 26,9) | +19,3 (12,2; 26,5) |

**Como interpretar:**

- **Etapa A (EMV):** os três fatores associam-se a notas maiores, mantidos os demais. São
  associações, não efeitos causais. O modelo explica cerca de **26%** da variância
  (R² = 0,255; σ̂ ≈ 105 pontos). Por isso ele estima com precisão a **média de um grupo** (IC de
  ±3,6 pontos), mas prevê mal **um aluno** (intervalo de predição de ±207 pontos).
- **Etapa B (posteriori exata):** sob Jeffreys, os números coincidem com o EMV, mas agora vale a
  leitura direta "há 95% de probabilidade de β₁ estar entre 22,3 e 27,2". A priori informativa
  (2022, peso de n₀ = 500, cerca de 9% da informação) só muda visivelmente β₂: em 2022 a vantagem da
  escola privada era de 43,8 pontos, e a posteriori fica em 62,0.
- **Etapa C (MCMC):** os três algoritmos convergiram (R̂ ≤ 1,008; ESS ≥ 1.062) e reproduzem a
  posteriori exata, com diferença máxima de 0,04 DP. O Gibbs é o mais eficiente (ESS ≈ 20 mil em
  20 mil iterações); o MH aceita 28,5% das propostas e tem ESS de cerca de 1,2 mil.
- **Etapa D (preditiva):** um aluno de escola pública, renda de 1 SM e mãe sem superior tem nota
  prevista de 500, com 95% de probabilidade de ficar entre 293 e 707. A checagem preditiva mostra a
  principal limitação: a Normal não reproduz a assimetria nem o piso da escala TRI (mínimo de 328).
- **Etapa E (comparação):** AIC (60.782,5), DIC, WAIC (60.782,6) e LOOIC praticamente coincidem, e
  o k de Pareto máximo é 0,14 (< 0,7). Com n grande, os paradigmas convergem nos números
  (Bernstein–von Mises) e diferem na interpretação: o IC fala do procedimento em amostras
  repetidas, e o HPD é uma probabilidade sobre o parâmetro, dados os dados.

## Onde conferir cada resultado

Nenhum número do relatório ou dos slides é digitado à mão: todos vêm dos arquivos abaixo, gerados
por `make train` (ou `python -m ib_enem_bayes.modeling.train`).

| O que conferir | No relatório | Arquivo |
|---|---|---|
| Dados reais, tamanho da base, fonte da priori, convergência geral | Seção 2 | `reports/tables/metadata.json` (`simulado: false`) |
| Estatísticas descritivas | Seção 2.3, Figura 1 | `descritivas_amostra.csv`, `figures/eda.png` |
| EMV, EP de Fisher, IC 95% | Seção 4.1 | `A_emv.csv`, `A_fisher_analitica.csv`, `A_fisher_numerica.csv` |
| R², σ̂, predição clássica | Seção 4.1 | `A_ajuste.csv`, `A_predicao_emv.csv` |
| Priori informativa e posteriori exata | Seção 4.2 | `B_hiperparametros_informativa.csv`, `B_posteriori_exata_*.csv` |
| Convergência (R̂, ESS, ACF) e exata × MCMC | Seção 4.3, Apêndice A | `C_convergencia.csv`, `C_posteriori_mcmc.csv`, `figures/C_*.png` |
| Preditiva, HPD e checagem preditiva | Seção 4.4 | `D_preditiva.csv`, `figures/D_preditiva.png`, `figures/D_ppc.png` |
| EMV × Bayes, critérios de informação | Seções 4.5 e 4.6 | `E_lado_a_lado.csv`, `E_criterios_informacao.csv` |
| Estudo assintótico | Seção 4.7 | `E_assintotico.csv`, `figures/E_assintotico.png` |
| IC × HPD, limitações | Seção 5 | — |

Para verificar que os algoritmos estão corretos, `python -m pytest tests` compara Gibbs e MH com a
posteriori analítica e o EMV/R² com o `statsmodels`. Para verificar a reprodutibilidade, rode o
pipeline de novo: com as sementes de `config.py`, as tabelas saem idênticas.

## Como reproduzir

O ambiente é gerenciado com o [uv](https://docs.astral.sh/uv/). Para renderizar é preciso também
o [Quarto](https://quarto.org) e uma distribuição LaTeX com XeLaTeX (MiKTeX, TeX Live ou
`quarto install tinytex`). Os blocos de código do PDF usam a fonte DejaVu Sans Mono, distribuída
com o TeX.

### Com `make` (Linux, macOS ou Windows com GNU Make)

O Makefile chama o uv em todos os alvos (`uv run --no-project`, que usa o `.venv` da pasta), então
**não é preciso ativar o ambiente**, nem no Windows.

```bash
make create_environment   # uv venv .venv
make requirements         # uv pip install -r requirements.txt (dependências + pacote editável)
make data                 # lê/baixa e limpa os microdados ENEM 2022 e 2023 (~1,2 GB)
make train                # Etapas A–E: tabelas, figuras e cadeias (~1,5 min)
make test                 # amostradores vs. posteriori analítica
make report               # reports/relatorio.pdf e reports/slides.pdf (Quarto + LaTeX)
make lint                 # ruff
make help                 # lista todos os alvos
```

### Sem `make`

Ambiente:

```bash
uv venv .venv                            # Python >= 3.11
uv pip install -r requirements.txt       # dependências + o pacote em modo editável (-e .)
```

Ative o ambiente antes dos passos seguintes:

```bash
source .venv/bin/activate                # Linux/macOS
.venv\Scripts\Activate.ps1               # Windows (PowerShell)
```

Pipeline (Windows, PowerShell, com o ambiente ativado):

```powershell
python -m ib_enem_bayes.dataset all
python -m ib_enem_bayes.features
python -m ib_enem_bayes.modeling.train
python -m pytest tests
ruff check ib_enem_bayes
$env:QUARTO_PYTHON = (Resolve-Path .venv\Scripts\python.exe).Path   # Quarto usa o Python do venv
cd reports; quarto render relatorio.qmd --to pdf; quarto render slides.qmd; cd ..
```

Sem ativar o ambiente, chame o Python dele direto: `.venv\Scripts\python -m ib_enem_bayes.modeling.train`
(ou `.venv/bin/python` no Linux/macOS). Para incluir um pacote novo, adicione-o a
`requirements.txt` e rode `uv pip install -r requirements.txt` de novo.

**Se o download do INEP falhar** (é comum o site ficar instável ou dar erro de certificado), baixe
os zips pelo navegador em
<https://www.gov.br/inep/pt-br/acesso-a-informacao/dados-abertos/microdados/enem>, salve como
`data/raw/microdados_enem_2022.zip` e `data/raw/microdados_enem_2023.zip` e rode `make data`
(ou `python -m ib_enem_bayes.dataset all`) de novo.
Se só o ano de 2023 estiver disponível, a priori informativa cai automaticamente para a elicitação
em `config.LITERATURE_PRIOR`. Nesse caso, preencha esses valores com referências.

**Teste sem baixar nada:** `make demo` (ou `python -m ib_enem_bayes.simulate` seguido dos passos de
features, train e report) gera dados *simulados* com o mesmo formato e roda tudo.
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
│   ├── references.bib     <- referências bibliográficas do relatório
│   ├── abnt.csl           <- estilo de citação ABNT
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
