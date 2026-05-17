import json
import argparse
from pprint import pprint
import os
import pandas as pd

from app.nb01_data_split import run_01_data_split
from app.nb02_scores import run_02_scores
from app.nb03_matriz_distancia import run_03_matriz_distancia
from app.nb04_avaliacao import run_04_avaliacao
from app.nb05_recomendacao_e_performance import run_05_recomendacao_e_performance


DEFAULT_CONFIG = {
    'seed': 8,
    'n_holdout_per_user': 3,
    'K_rec': 10,
    'limiar_sim_util': 0.01,
    # caminhos padrão dos conjuntos de dados (podem ser sobrescritos por config ou CLI)
    'interacoes_path': 'datasets/interacoes_v3.csv',
    'produtos_path': 'datasets/produtos_v2.csv',
    'matriz_path': 'datasets/matriz_produto_usuario.csv',
    'similaridade_path': None,
}


def load_config(path=None):
    cfg = DEFAULT_CONFIG.copy()
    if path:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg.update(data)
    return cfg


def read_table(path):
    if path is None:
        return None
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)


def run_pipeline_grid(config, produtos_path=None, interacoes_path=None, matriz_path=None, similaridade_path=None,
                     produtos_df=None, interacoes_df=None, matriz_df=None, similaridade_df=None, salvar_saida: bool = True):
    # carregar entradas (prefere DataFrames em memória quando fornecidos)
    interacoes_full = interacoes_df if interacoes_df is not None else read_table(interacoes_path or config.get('interacoes_path'))
    produtos = produtos_df if produtos_df is not None else read_table(produtos_path or config.get('produtos_path'))
    matriz_produto_usuario = matriz_df
    if matriz_produto_usuario is None and (matriz_path or config.get('matriz_path')):
        mp = read_table(matriz_path or config.get('matriz_path'))
        if mp is not None:
            matriz_produto_usuario = mp

    # 01: divisão
    out01 = run_01_data_split(interacoes_full, seed=config.get('seed', 8), n_holdout_per_user=config.get('n_holdout_per_user', 1), salvar_saida=salvar_saida)
    interacoes_treino = out01['interacoes_treino'][0]
    interacoes_teste = out01['interacoes_teste'][0]

    # 02: calcula scores se a matriz não for fornecida
    if matriz_produto_usuario is None:
        out02 = run_02_scores(produtos, interacoes_treino, salvar_saida=salvar_saida)
        matriz_produto_usuario = out02['matriz_produto_usuario'][0]
    else:
        out02 = {'matriz_produto_usuario': (matriz_produto_usuario, matriz_path or config.get('matriz_path'))}

    # 03: similaridade
    if similaridade_df is not None:
        df_similaridade = similaridade_df
    elif similaridade_path or config.get('similaridade_path'):
        df_similaridade = read_table(similaridade_path or config.get('similaridade_path'))
    else:
        out03 = run_03_matriz_distancia(matriz_produto_usuario, salvar_saida=salvar_saida)
        df_similaridade = out03['similaridade_item_item'][0]

    # 04: avaliação
    out04 = run_04_avaliacao(matriz_produto_usuario, df_similaridade, limiar_sim_util=config.get('limiar_sim_util', 0.01), salvar_saida=salvar_saida)

    # 05: recomendações
    out05 = run_05_recomendacao_e_performance(df_similaridade, interacoes_treino, interacoes_teste, K=config.get('K_rec', 10), salvar_saida=salvar_saida)

    result = {
        'out01': out01,
        'out02': out02,
        'out03': {'similaridade_item_item': (df_similaridade, similaridade_path or config.get('similaridade_path'))},
        'out04': out04,
        'out05': out05,
        'config_used': config,
    }
    return result


def main():
    parser = argparse.ArgumentParser(description='Run grid-search pipeline (01→05) using existing data files.')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file with parameters.')
    parser.add_argument('--produtos', type=str, default=None, help='Path to produtos file (csv/parquet).')
    parser.add_argument('--interacoes', type=str, default=None, help='Path to interacoes file (csv/parquet).')
    parser.add_argument('--matriz', type=str, default=None, help='Path to precomputed matriz produto-usuario (csv/parquet).')
    parser.add_argument('--similaridade', type=str, default=None, help='Path to precomputed similaridade item-item (csv/parquet).')
    parser.add_argument('--quiet', action='store_true', help='Suppress printing outputs.')
    args = parser.parse_args()

    cfg = load_config(args.config)
    if not args.quiet:
        print('Configuração:')
        pprint(cfg)

    results = run_pipeline_grid(cfg, produtos_path=args.produtos, interacoes_path=args.interacoes, matriz_path=args.matriz, similaridade_path=args.similaridade)

    if not args.quiet:
        print('\nPipeline finalizado. Resumo (out05 e configuração):')
        pprint({k: v for k, v in results.items() if k == 'out05' or k == 'config_used'})
    else:
        print('Execução de grade concluída; saídas salvas em dados_sinteticos/dados_tratatos (se aplicável).')


if __name__ == '__main__':
    main()
