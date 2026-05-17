import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from .utils import save_csv


def run_03_matriz_distancia(matriz_scores, round_decimals: int = 6, salvar_saida: bool = True):
    """Calcula a similaridade cosseno item-item a partir da `matriz_produto_usuario`.

    Retorna o DataFrame de similaridade e o caminho do CSV salvo (se `salvar_saida=True`).
    """
    ids_produtos = matriz_scores['id_produto'].values
    X_scores = matriz_scores.drop(columns='id_produto').to_numpy(dtype=float)
    similaridade_cosseno_produtos = cosine_similarity(X_scores)
    similaridade_item_item = pd.DataFrame(similaridade_cosseno_produtos, index=ids_produtos, columns=ids_produtos)
    similaridade_item_item = similaridade_item_item.round(round_decimals)

    path = None
    if salvar_saida:
        path = save_csv(similaridade_item_item, 'dados_sinteticos/dados_tratatos/similaridade_item_item')
    return {'similaridade_item_item': (similaridade_item_item, path)}
