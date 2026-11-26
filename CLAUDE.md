# CLAUDE.md — ib-enem-bayes

Trabalho final da disciplina **Inferência Bayesiana** (CIA044, IESB). Autor: Vinicius de Paula Ribeiro.
Responda sempre em **português do Brasil**.

## O que é o projeto

Regressão linear para a nota de Matemática do ENEM 2023 (`NU_NOTA_MT`) com
`log2(renda em SM)`, `escola_privada` e `mae_superior`, comparando:

- **Etapa A** — EMV, EP pela informação de Fisher observada, IC 95% (`modeling/frequentist.py`)
- **Etapa B** — priori conjugada Normal–Gama-Inversa: (a) Jeffreys, (b) power prior do ENEM 2022 com `n0`
  (`modeling/conjugate.py`)
- **Etapa C** — Gibbs e Metropolis–Hastings do zero + PyMC/NUTS; R̂, ESS, ACF, traceplots; exata × MCMC
  (`modeling/samplers.py`, `modeling/pymc_model.py`, `diagnostics.py`)
- **Etapa D** — preditiva a posteriori (t analítica + composição Monte Carlo), HPD 95% (`predictive.py`)
- **Etapa E** — EMV × Bayes, AIC/BIC × DIC/WAIC/LOO, estudo assintótico (`compare.py`)

O enunciado exige notação científica: parâmetros em letras gregas (β, σ², θ), estimadores com circunflexo.

## Ambiente (uv)

Use **somente o uv** para gerenciar o ambiente; nada de `pip`, `venv` ou conda direto.

```bash
uv venv .venv                          # cria o ambiente (sem pip dentro)
uv pip install -r requirements.txt     # dependências + pacote editável (-e .)
```

- Novas dependências vão em `requirements.txt`; depois rode `uv pip install -r requirements.txt`.
- `make requirements` **não funciona** num venv do uv (chama `python -m pip`); use o comando acima.
- O ambiente de desenvolvimento é Windows sem `make`. Chame o Python do venv direto
  (`.venv\Scripts\python -m ...`) em vez de depender de ativação, que não persiste entre comandos.
- Para ferramentas avulsas, fora do projeto, use `uv run --no-project --with <pacote> ...`
  em vez de instalar no `.venv`.

## Comandos

```bash
make data           # lê data/raw/microdados_enem_{2022,2023}.zip -> data/interim -> data/processed
make train          # pipeline A–E -> reports/tables/*.csv, reports/figures/*.png, models/*.npy
make test           # pytest: amostradores vs. posteriori analítica
make report         # quarto: reports/relatorio.pdf e reports/slides.pdf
make demo           # tudo com dados SIMULADOS (só para testar; nunca entregar)
```

Sem `make` (Windows, PowerShell, a partir da raiz):

```powershell
.venv\Scripts\python -m ib_enem_bayes.dataset all
.venv\Scripts\python -m ib_enem_bayes.features
.venv\Scripts\python -m ib_enem_bayes.modeling.train      # ~1,5 min
.venv\Scripts\python -m pytest tests
.venv\Scripts\ruff check ib_enem_bayes
$env:QUARTO_PYTHON = (Resolve-Path .venv\Scripts\python.exe).Path
$env:PATH = (Resolve-Path .venv\Scripts).Path + ";" + $env:PATH
cd reports; quarto render relatorio.qmd --to pdf; quarto render slides.qmd
```

Notas de renderização:
- Sem `QUARTO_PYTHON` apontando para o venv, o Quarto usa outro Python e não acha o pacote.
- Abra arquivos de texto sempre com `encoding="utf-8"`: o padrão do Windows (cp1252) quebra.
- A Latin Modern não tem grego nem ∫, ∝, ≈ fora do modo matemático. No texto corrido do `.qmd`,
  escreva em LaTeX (`$\beta$`, `$\approx$`). O código usa DejaVu Sans Mono (`monofont` no YAML).
  No relatório, não use `newunicodechar` para isso: ele quebra em linhas longas de código com o
  `breakanywhere` do fvextra. Nos slides, que não têm blocos de código, ele mapeia ≈ e −.

Configurações do analista (amostra, cadeias, n0, perfis de predição): `ib_enem_bayes/config.py`.

## Regras

- `data/raw/` é imutável; nunca editar os zips do INEP.
- Se `data/interim/SIMULATED` existir, os resultados são sintéticos: avise e não interprete como reais.
- Os números do relatório vêm de `reports/tables/` via `reports/_common.py`; não digite números à mão no `.qmd`.
- Após mudar código em `ib_enem_bayes/`, rode `make test` e `ruff check ib_enem_bayes`.

## Roteiro de análise dos resultados reais

Depois de `make train`, verifique e reporte, nesta ordem:

1. `reports/tables/metadata.json`: `simulado` deve ser `false`; anote `n_base_completa`, `fonte_priori_informativa`
   e `todos_convergiram`.
2. **Dados**: `descritivas_amostra.csv` e `figures/eda.png` — distribuição de Y, proporções das binárias, faixas de renda.
3. **Etapa A**: sinais e magnitudes em `A_emv.csv`; `A_fisher_analitica` ≈ `A_fisher_numerica`.
4. **Etapa B**: `B_hiperparametros_informativa.csv` é plausível? Compare `B_posteriori_exata_*` com o EMV.
5. **Etapa C**: em `C_convergencia.csv`, todos com `rhat_rank < 1.01` e `ess_bulk`, `ess_tail > 400`. Se o MH
   falhar, aumente `N_DRAWS` em `config.py`. Abra os traceplots e as figuras exata × MCMC.
6. **Etapa D**: `D_preditiva.csv` — HPD de Monte Carlo ≈ HPD exato; olhe `figures/D_ppc.png` e descreva os
   desajustes (assimetria, piso da escala TRI).
7. **Etapa E**: `E_lado_a_lado.csv`, `E_criterios_informacao.csv` (pareto_k_max < 0,7?) e `figures/E_assintotico.png`.
8. Releia **todo** o texto interpretativo de `reports/relatorio.qmd` e `reports/slides.qmd`, sobretudo
   Introdução, interpretação do EMV, Etapa D e Conclusões. Algumas frases foram escritas antes dos
   resultados reais: confirme cada afirmação contra as tabelas e corrija o que não se sustentar.
9. `make report`, abra os PDFs e confira: tabelas sem estouro de margem, figuras legíveis, sem aviso de simulado.
10. Faça commit dos resultados finais. Tabelas, figuras e PDFs já são versionados; os dados e as
    cadeias `models/*.npy` continuam ignorados.
