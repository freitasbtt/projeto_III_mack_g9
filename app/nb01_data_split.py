import numpy as np
import pandas as pd
from .utils import save_parquet


def run_01_data_split(interacoes, seed: int = 8, n_holdout_per_user: int = 1, salvar_saida: bool = True):
    """Divide as interações em treino/teste removendo `n_holdout_per_user` produtos distintos por usuário quando possível.

    Se um usuário não tiver produtos distintos suficientes para remover `n_holdout_per_user` itens (mantendo ao menos
    um no treino), as interações desse usuário permanecem inteiras no conjunto de treino.

    Retorna um dicionário com DataFrames e caminhos salvos (se `salvar_saida=True`).
    """
    np.random.seed(seed)

    produtos_por_usuario = interacoes.groupby('id_usuario')['id_produto'].nunique()
    # um usuário deve ter pelo menos n_holdout_per_user + 1 produtos distintos para manter >=1 no treino
    usuarios_avaliaveis = produtos_por_usuario[produtos_por_usuario >= (n_holdout_per_user + 1)].index

    usuarios_nao_avaliaveis = produtos_por_usuario[produtos_por_usuario < (n_holdout_per_user + 1)].index

    usuarios_por_produto = interacoes.groupby('id_produto')['id_usuario'].nunique()
    produtos_sorteaveis = set(usuarios_por_produto[usuarios_por_produto >= 2].index)

    linhas_treino = []
    linhas_teste = []

    for id_usuario, grupo in interacoes.groupby('id_usuario'):
        if id_usuario not in usuarios_avaliaveis:
            linhas_treino.append(grupo)
            continue

        produtos_distintos = grupo['id_produto'].unique()
        candidatos_teste = [p for p in produtos_distintos if p in produtos_sorteaveis]

        # se não houver candidatos suficientes para formar o holdout, manter o usuário inteiro no treino
        if not candidatos_teste or len(candidatos_teste) < n_holdout_per_user:
            linhas_treino.append(grupo)
            continue

        produtos_teste = list(np.random.choice(candidatos_teste, size=n_holdout_per_user, replace=False))
        mask_teste = grupo['id_produto'].isin(produtos_teste)
        linhas_teste.append(grupo[mask_teste])
        linhas_treino.append(grupo[~mask_teste])

    interacoes_treino = pd.concat(linhas_treino).reset_index(drop=True)
    interacoes_teste = pd.concat(linhas_teste).reset_index(drop=True)

    # verificações básicas (lança erro se violado)
    usuarios_treino = set(interacoes_treino['id_usuario'].unique())
    usuarios_teste = set(interacoes_teste['id_usuario'].unique())
    if not usuarios_teste.issubset(usuarios_treino):
        raise AssertionError('Há usuários no teste sem histórico no treino!')

    pares_treino = set(zip(interacoes_treino['id_usuario'], interacoes_treino['id_produto']))
    pares_teste = set(zip(interacoes_teste['id_usuario'], interacoes_teste['id_produto']))
    if not pares_treino.isdisjoint(pares_teste):
        raise AssertionError('Há vazamento de pares usuário-produto!')

    treino_path = teste_path = None
    if salvar_saida:
        treino_path = save_parquet(interacoes_treino, 'dados_sinteticos/interacoes_treino')
        teste_path = save_parquet(interacoes_teste, 'dados_sinteticos/interacoes_teste')

    return {
        'interacoes_treino': (interacoes_treino, treino_path),
        'interacoes_teste': (interacoes_teste, teste_path),
    }
