import json
import argparse
from pprint import pprint

from app.nb00_geracao_dados_simulados import run_00_geracao_dados_simulados
from app.nb01_data_split import run_01_data_split
from app.nb02_scores import run_02_scores
from app.nb03_matriz_distancia import run_03_matriz_distancia
from app.nb04_avaliacao import run_04_avaliacao
from app.nb05_recomendacao_e_performance import run_05_recomendacao_e_performance


DEFAULT_CONFIG = {
    'seed': 8,
    'n_usuarios': 1000,
    'n_produtos': 100,
    'n_categorias': 10,
    'n_marcas': 20,
    'min_renda': 95,
    'max_renda': 5000,
    'max_moradores': 10,
    'tamanho_espaco': 10000,
    'n_elegiveis': 200,
    'n_ofertas': 10000,
    'n_copias': 4,
    'mlt_ofr': 5,
    # pesos de preferência latente para nb00
    'P': 2,
    'C': 5,
    'M': 5,
    'E': 1,
    # número de produtos em holdout por usuário para teste (run_01_data_split)
    'n_holdout_per_user': 3,
    # K para vizinhos / recomendações
    'K_knn': 3,
    'K_rec': 10,
    'limiar_sim_util': 0.01,
}


def load_config(path=None):
    cfg = DEFAULT_CONFIG.copy()
    if path:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        cfg.update(data)
    return cfg


def run_pipeline(config):
    # 00: gerar dados sintéticos
    print('Executando etapa 00: geração de dados sintéticos')
    out00 = run_00_geracao_dados_simulados(config)
    usuarios, produtos, marcas, categorias, itens = (
        out00['usuarios'][0],
        out00['produtos'][0],
        out00['marcas'][0],
        out00['categorias'][0],
        out00['itens'][0],
    )

    # 01: divisão treino/teste
    print('Executando etapa 01: divisão treino/teste')
    out01 = run_01_data_split(itens, seed=config.get('seed', 8), n_holdout_per_user=config.get('n_holdout_per_user', 1))
    interacoes_treino = out01['interacoes_treino'][0]
    interacoes_teste = out01['interacoes_teste'][0]

    # 02: scores (matriz produto-usuario)
    print('Executando etapa 02: cálculo de scores (matriz produto-usuario)')
    out02 = run_02_scores(produtos, interacoes_treino)
    matriz_produto_usuario = out02['matriz_produto_usuario'][0]

    # 03: matriz de distância / similaridade
    print('Executando etapa 03: cálculo da similaridade item-item')
    out03 = run_03_matriz_distancia(matriz_produto_usuario)
    similaridade_item_item = out03['similaridade_item_item'][0]

    # 04: avaliação / diagnósticos
    print('Executando etapa 04: avaliação e diagnósticos')
    out04 = run_04_avaliacao(matriz_produto_usuario, similaridade_item_item, limiar_sim_util=config.get('limiar_sim_util', 0.01))

    # 05: recomendação e performance
    print('Executando etapa 05: geração de recomendações e métricas')
    out05 = run_05_recomendacao_e_performance(similaridade_item_item, interacoes_treino, interacoes_teste, K=config.get('K_rec', 10))

    # Final output is out05
    result = {
        'out00': {k: (v[0].shape if hasattr(v[0], 'shape') else None, v[1]) for k, v in out00.items()},
        'out01': {k: (v[0].shape if hasattr(v[0], 'shape') else None, v[1]) for k, v in out01.items()},
        'out02': {k: (v[0].shape if hasattr(v[0], 'shape') else None, v[1]) for k, v in out02.items()},
        'out03': {k: (v[0].shape if hasattr(v[0], 'shape') else None, v[1]) for k, v in out03.items()},
        'out04': out04,
        'out05': out05,
        'config_used': config,
    }

    return result


def main():
    parser = argparse.ArgumentParser(description='Run recommendation pipeline (00→05).')
    parser.add_argument('--config', type=str, default=None, help='Path to JSON config file with parameters.')
    parser.add_argument('--quiet', action='store_true', help='If set, do not print recommendations to stdout (they are still saved).')
    args = parser.parse_args()

    cfg = load_config(args.config)
    print('Configuração:')
    pprint(cfg)

    results = run_pipeline(cfg)

    print('\nPipeline finalizado.')
    if not args.quiet:
        print('Resumo dos resultados finais (out05 e configuração):')
        pprint({k: v for k, v in results.items() if k.startswith('out05') or k == 'config_used'})
    else:
        print('Recomendações salvas em dados_sinteticos/dados_tratatos (saída suprimida).')


if __name__ == '__main__':
    main()
