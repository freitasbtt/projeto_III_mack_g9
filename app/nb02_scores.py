import numpy as np
import pandas as pd
from .utils import save_csv


def run_02_scores(produtos, interacoes, salvar_saida: bool = True):
    """Calcula a `matriz_produto_usuario` (scores) a partir de `produtos` e `interacoes`.

    Retorna o DataFrame resultante e o caminho do CSV salvo (se `salvar_saida=True`).
    """
    ids_produtos = np.sort(produtos['id_produto'].unique())
    relacionamentos_usuario = {}

    for id_usuario, df_user in interacoes.groupby('id_usuario'):
        df_user = df_user.copy()
        media_dist = df_user['deslocamento'].mean()
        desvio_dist = df_user['deslocamento'].std()
        distancia_maxima = media_dist + 2 * desvio_dist

        df_user['distancia_tratada'] = df_user['deslocamento'].clip(upper=distancia_maxima)

        min_dist = df_user['distancia_tratada'].min()
        max_dist = df_user['distancia_tratada'].max()

        if pd.isna(max_dist) or pd.isna(min_dist):
            df_user['distancia_normalizada'] = 0.0
        elif max_dist == min_dist:
            df_user['distancia_normalizada'] = 0.0
        else:
            df_user['distancia_normalizada'] = (df_user['distancia_tratada'] - min_dist) / (max_dist - min_dist)

        df_user['score'] = df_user['distancia_normalizada'] + 1

        df_user = (
            pd.DataFrame({'id_produto': ids_produtos})
            .merge(df_user.groupby('id_produto', as_index=False)['score'].sum(), on='id_produto', how='left')
        )

        df_user['score'] = df_user['score'].fillna(0)
        df_user = df_user.sort_values('id_produto').reset_index(drop=True)

        relacionamentos_usuario[id_usuario] = df_user

    matriz_produto_usuario = pd.concat(
        [
            df_user.set_index('id_produto')['score'].rename(id_usuario)
            for id_usuario, df_user in relacionamentos_usuario.items()
        ],
        axis=1,
    ).sort_index()

    matriz_produto_usuario = matriz_produto_usuario.reset_index()

    path = None
    if salvar_saida:
        path = save_csv(matriz_produto_usuario, 'dados_sinteticos/dados_tratatos/matriz_produto_usuario', index=False)
    return {'matriz_produto_usuario': (matriz_produto_usuario, path)}
