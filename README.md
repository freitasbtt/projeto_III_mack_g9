# MODELO DE RECOMENDAÇÃO PARA REDISTRIBUIÇÃO DE ALIMENTOS EXCEDENTES A PESSOAS EM VULNERABILIDADE SOCIAL

Este projeto visa a criação de dados sintéticos baseados em preferências latentes para aplicar uma metodologia de sistema de recomendação baseado em K vizinhos mais próximos (KNN) .

**Índice**
- Visão geral
- Instalação
- Execução
- Configuração
- Saídas
- Estrutura do projeto
- Licença e contato
- Notas Rápidas

**Visão geral**
- Objetivo: reproduzir o pipeline de recomendação em módulos Python reutilizáveis e permitir buscas de hiperparâmetros via grid search.

**Instalação**
- Recomendado: criar ambiente Conda via `environment.yml` ou instalar via `requirements.txt`.

Conda (recomendado):
```bash
conda env create -f environment.yml
conda activate recsys
```

Pip (alternativa):
```bash
python -m venv .venv
.venv\Scripts\Activate.ps1   # PowerShell
pip install -r requirements.txt
```

**Execução**
- Pipeline completo:
```bash
python main.py --config config.json --quiet
```
- Grid search:
```bash
python gridsearch.py --config gridsearch_config.json --out grid_results.csv
```

A execução do `main.py` gera arquivos de dados para cada função macro.
A execução do `gridsearch.py` não gera dados intermediários e tem como foco apenas as métricas.

**Configuração**
- `config.json`: parâmetros para `main.py` (P, seeds, n_holdout_per_user, etc.).
- `gridsearch_config.json`: listas de valores para varrer no grid (ex.: `n_holdout_per_user`, `K_rec`, `limiar_sim_util`).

**Saídas**
- Cada arquivo python na pasta `app`, se executado individualmente, armazena os dados na pasta `dados_sinteticos/`. O parâmetro `salvar_saida=False` interrompe esse comportamento, mantendo a saída limitada ao `return` dafunção.
- O resultado do `gridsearch.py` permanece explícito na raiz do projeto como `grid_result.csv`.

**Estrutura do projeto**
- `app/` — módulos `nb00_...` a `nb05_...` (funções que recebem DataFrames e retornam resultados).
- `main.py` — orquestra 00→05.
- `main_grid.py` — orquestra 01→05 (aceita dados em memória).
- `gridsearch.py` — runner do grid; gera CSV com métricas por combinação.
- `grid_result.csv` — resultado do `gridsearch.py`, contendo os hiperparâmetros e as métricas como colunas.
- `config.json`, `gridsearch_config.json` — exemplos de configuração.
- `datasets/` — entradas CSV (deprecated).
- `dados_sinteticos/` — saídas das funções (nomes incluem timestamps).
- `scripts/` — notebooks originais (ensaios; não necessários para execução automatizada).

**Licença e contato**
- Este projeto está licenciado sob a Licença MIT — veja o texto completo em LICENSE.

**Notas rápidas**
- `scripts/` contém notebooks experimentais; podem ser removidos se desejar um repositório limpo.
- Em grid search, opções de salvar intermediários já são desativadas por padrão (uso interno de `salvar_saida=False`).
- O arquivo


