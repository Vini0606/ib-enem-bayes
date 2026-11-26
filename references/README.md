# Referências

- Dicionário de dados do ENEM: vem dentro do zip do INEP (`DICIONÁRIO/Dicionário_Microdados_Enem_<ano>.xlsx`).
  Variáveis usadas: `NU_NOTA_MT`, `TP_PRESENCA_MT`, `TP_ESCOLA`, `Q002` (escolaridade da mãe) e `Q006` (renda familiar).
- Faixas de `Q006` convertidas para salários mínimos em `ib_enem_bayes/config.py::Q006_SM`
  (SM 2022 = R$ 1.212; SM 2023 = R$ 1.320).
- Ibrahim & Chen (2000), *Power prior distributions for regression models* — base do cenário de priori informativa.
- Material da disciplina (CIA044, Profª Simone de Araújo Góes Assis, IESB): apostila de Inferência Bayesiana (cap. 3, prioris conjugadas; cap. 8, Gibbs) e Módulos 03, 05 e 06.
