#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ib-enem-bayes
PYTHON_VERSION = 3.13

# Tudo roda no .venv do projeto via uv. `uv run --no-project` usa o .venv da pasta
# (sem criar uv.lock) e põe o Python dele no PATH, o que vale também para o Quarto.
UV_RUN = uv run --no-project
PYTHON_INTERPRETER = $(UV_RUN) python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Cria o ambiente virtual .venv com o uv
.PHONY: create_environment
create_environment:
	uv venv .venv --python $(PYTHON_VERSION)
	@echo ">>> Ambiente criado em .venv. Rode: make requirements"


## Instala as dependências e o pacote (modo editável) com o uv
.PHONY: requirements
requirements:
	uv pip install -r requirements.txt


## Apaga arquivos Python compilados (fora do .venv)
.PHONY: clean
clean:
	$(PYTHON_INTERPRETER) -c "import pathlib, shutil; ps = [p for p in pathlib.Path('.').rglob('*') if '.venv' not in p.parts]; [p.unlink() for p in ps if p.suffix in ('.pyc', '.pyo')]; [shutil.rmtree(p) for p in ps if p.name == '__pycache__' and p.exists()]"


## Lint com ruff (use `make format` para formatar)
.PHONY: lint
lint:
	$(UV_RUN) ruff format --check
	$(UV_RUN) ruff check

## Formata o código com ruff
.PHONY: format
format:
	$(UV_RUN) ruff check --fix
	$(UV_RUN) ruff format



## Roda os testes
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest tests


#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Lê/baixa e limpa os microdados do ENEM (2022 = priori, 2023 = análise)
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m ib_enem_bayes.dataset all
	$(PYTHON_INTERPRETER) -m ib_enem_bayes.features

## Gera dados SIMULADOS (apenas para testar o pipeline sem baixar o INEP)
.PHONY: data_sim
data_sim:
	$(PYTHON_INTERPRETER) -m ib_enem_bayes.simulate
	$(PYTHON_INTERPRETER) -m ib_enem_bayes.features

## Estima todos os modelos (Etapas A-E): tabelas, figuras e cadeias
.PHONY: train
train:
	$(PYTHON_INTERPRETER) -m ib_enem_bayes.modeling.train

## Rodada rápida do pipeline (poucas iterações)
.PHONY: train_fast
train_fast:
	$(PYTHON_INTERPRETER) -m ib_enem_bayes.modeling.train --fast

## Renderiza o relatório (PDF via LaTeX) e os slides (Quarto), com o Python do .venv
.PHONY: report
report:
	cd reports && $(UV_RUN) quarto render relatorio.qmd --to pdf
	cd reports && $(UV_RUN) quarto render slides.qmd

## Pipeline completo: dados -> modelos -> testes -> relatório
.PHONY: all
all: data train test report

## Demonstração ponta a ponta com dados simulados
.PHONY: demo
demo: data_sim train test report

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
