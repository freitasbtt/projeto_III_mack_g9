import json
import itertools
import argparse
import pandas as pd
from pprint import pprint
from app.nb00_geracao_dados_simulados import run_00_geracao_dados_simulados
from main_grid import run_pipeline_grid


def load_config(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def expand_grid(grid_dict):
    keys = list(grid_dict.keys())
    values = [grid_dict[k] for k in keys]
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo))


def main():
    parser = argparse.ArgumentParser(description='Run grid search over parameters for steps 01-05.')
    parser.add_argument('--config', type=str, default='gridsearch_config.json', help='Path to gridsearch config JSON.')
    parser.add_argument('--out', type=str, default='gridsearch_results.csv', help='CSV output path for results.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    nb00_cfg = cfg.get('nb00', {})
    grid = cfg.get('grid', {})

    print('Gerando dados sintéticos (nb00) uma única vez...')
    out00 = run_00_geracao_dados_simulados(nb00_cfg, salvar_saida=False)
    produtos = out00['produtos'][0]
    interacoes = out00['itens'][0]
    print('Dados gerados.')

    results = []
    for params in expand_grid(grid):
        run_cfg = params.copy()
        # ensure common defaults
        run_cfg.setdefault('seed', nb00_cfg.get('seed', 8))
        run_cfg.setdefault('limiar_sim_util', cfg.get('limiar_sim_util', 0.01))
        run_cfg.setdefault('K_rec', cfg.get('K_rec', 10))

        print('Executando combinação:', params)
        out = run_pipeline_grid(run_cfg, produtos_df=produtos, interacoes_df=interacoes, salvar_saida=False)

        metrics = out['out05']['metrics'][0] if isinstance(out['out05']['metrics'][0], dict) else out['out05']['metrics'][0]

        # extrai tamanhos de treino/teste (número de interações)
        try:
            interacoes_treino_df = out['out01']['interacoes_treino'][0]
            interacoes_teste_df = out['out01']['interacoes_teste'][0]
            n_interacoes_treino = int(len(interacoes_treino_df)) if interacoes_treino_df is not None else None
            n_interacoes_teste = int(len(interacoes_teste_df)) if interacoes_teste_df is not None else None
        except Exception:
            n_interacoes_treino = None
            n_interacoes_teste = None

        row = {}
        row.update(params)
        # add metrics
        row.update({
            'precision_at_K': metrics.get('precision_at_K'),
            'recall_at_K': metrics.get('recall_at_K'),
            'mrr_at_K': metrics.get('mrr_at_K'),
            'n_users_evaluated': metrics.get('n_users_evaluated'),
            'n_interacoes_treino': n_interacoes_treino,
            'n_interacoes_teste': n_interacoes_teste,
        })
        results.append(row)

    df = pd.DataFrame(results)
    df.to_csv(args.out, index=False)
    print(f'Grid search finalizado. Resultados salvos em {args.out}')


if __name__ == '__main__':
    main()
