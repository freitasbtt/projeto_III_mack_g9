import numpy as np
import pandas as pd
from .utils import save_csv


def run_05_recomendacao_e_performance(df_similaridade, df_interacoes_treino, df_interacoes_teste, K: int = 10, salvar_saida: bool = True):
    """Gera recomendações e calcula Precision@K, Recall@K e MRR@K.

    Retorna um dicionário com as recomendações, métricas e caminhos salvos (se `salvar_saida=True`).
    """
    ids_produtos_arr = df_similaridade.index.values
    usuarios_para_recomendar = df_interacoes_teste['id_usuario'].unique()

    recomendacoes = {}

    for id_usuario in usuarios_para_recomendar:
        adquiridos = set(df_interacoes_treino[df_interacoes_treino['id_usuario'] == id_usuario]['id_produto'])
        scores_candidatos = pd.Series(0.0, index=ids_produtos_arr)

        for id_produto in adquiridos:
            if id_produto in df_similaridade.index:
                scores_candidatos += df_similaridade.loc[id_produto]

        scores_candidatos = scores_candidatos.drop(index=list(adquiridos), errors='ignore')
        top_k = scores_candidatos.nlargest(K).index.tolist()
        recomendacoes[id_usuario] = top_k

    # Métricas
    relevantes_teste = df_interacoes_teste.groupby('id_usuario')['id_produto'].apply(set).to_dict()

    precisoes = []
    recalls = []
    reciprocal_ranks = []

    for id_usuario, top_k in recomendacoes.items():
        relevantes = relevantes_teste.get(id_usuario, set())
        if not relevantes:
            continue
        acertos = len(set(top_k) & relevantes)
        precisoes.append(acertos / K)
        recalls.append(acertos / len(relevantes))

        rr = 0.0
        for rank, id_produto in enumerate(top_k, start=1):
            if id_produto in relevantes:
                rr = 1 / rank
                break
        reciprocal_ranks.append(rr)

    metrics = {
        'K': K,
        'precision_at_K': float(np.mean(precisoes)) if precisoes else float('nan'),
        'recall_at_K': float(np.mean(recalls)) if recalls else float('nan'),
        'mrr_at_K': float(np.mean(reciprocal_ranks)) if reciprocal_ranks else float('nan'),
        'n_users_evaluated': len(precisoes),
    }

    # Salva recomendações e métricas
    rec_df = pd.DataFrame([(u, rec) for u, rec in recomendacoes.items()], columns=['id_usuario', 'top_k'])
    rec_path = metrics_path = None
    if salvar_saida:
        rec_path = save_csv(rec_df, 'dados_sinteticos/dados_tratatos/recomendacoes')
        metrics_df = pd.DataFrame([metrics])
        metrics_path = save_csv(metrics_df, 'dados_sinteticos/dados_tratatos/recomendacoes_metrics')

    return {
        'recomendacoes': (recomendacoes, rec_path),
        'metrics': (metrics, metrics_path),
    }
