#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = ib-enem-bayes
PYTHON_VERSION = 3.11
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -r requirements.txt
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format



## Run tests
.PHONY: test
test:
	python -m pytest tests


## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	@bash -c "if [ ! -z `which virtualenvwrapper.sh` ]; then source `which virtualenvwrapper.sh`; mkvirtualenv $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); else mkvirtualenv.bat $(PROJECT_NAME) --python=$(PYTHON_INTERPRETER); fi"
	@echo ">>> New virtualenv created. Activate with:\nworkon $(PROJECT_NAME)"
	



#################################################################################
# PROJECT RULES                                                                 #
#################################################################################


## Baixa e limpa os microdados do ENEM (2022 = priori, 2023 = análise)
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

## Renderiza o relatório (PDF via LaTeX) e os slides (Quarto)
.PHONY: report
report:
	cd reports && quarto render relatorio.qmd --to pdf
	cd reports && quarto render slides.qmd

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
