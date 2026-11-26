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

## Comandos

```bash
make requirements   # instala dependências e o pacote (-e .)
make data           # lê data/raw/microdados_enem_{2022,2023}.zip -> data/interim -> data/processed
make train          # pipeline A–E -> reports/tables/*.csv, reports/figures/*.png, models/*.npy
make test           # pytest: amostradores vs. posteriori analítica
make report         # quarto: reports/relatorio.pdf e reports/slides.pdf
make demo           # tudo com dados SIMULADOS (só para testar; nunca entregar)
```

Sem `make` (Windows): `python -m ib_enem_bayes.dataset all`, `python -m ib_enem_bayes.features`,
`python -m ib_enem_bayes.modeling.train`, `python -m pytest tests`,
`cd reports && quarto render relatorio.qmd --to pdf && quarto render slides.qmd`.

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
10. Remova as 3 últimas linhas do `.gitignore` (tabelas, figuras e PDFs) e faça commit dos resultados finais.
