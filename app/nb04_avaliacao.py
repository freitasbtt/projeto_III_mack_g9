import numpy as np
import pandas as pd
from .utils import save_csv


def run_04_avaliacao(score_produto_usuario, df_similaridade, limiar_sim_util: float = 0.01, salvar_saida: bool = True):
    """Calcula diagnósticos e métricas simples de avaliação a partir de scores e similaridade.

    Retorna um dicionário com diagnósticos, DataFrame de resumo e caminho salvo (se `salvar_saida=True`).
    """
    count_zeros = pd.DataFrame()
    count_zeros['count_zeros'] = (df_similaridade == 0).sum(axis=1) / len(df_similaridade.columns)

    X = score_produto_usuario.drop(columns='id_produto').to_numpy(dtype=float)
    n_produtos, n_usuarios = X.shape

    compradores_por_produto = (X != 0).sum(axis=1)
    produtos_por_usuario = (X != 0).sum(axis=0)

    matriz_binaria = (X != 0).astype(int)
    cooc_item_item = matriz_binaria @ matriz_binaria.T

    mask_fora_diagonal = ~np.eye(n_produtos, dtype=bool)
    cooc_sem_diagonal = cooc_item_item[mask_fora_diagonal]

    ids_produtos = score_produto_usuario['id_produto'].values

    # arrays derivados da similaridade
    matriz_similaridade = df_similaridade.values
    similaridade_sem_diagonal = matriz_similaridade.copy()
    np.fill_diagonal(similaridade_sem_diagonal, np.nan)
    sim_fora_diagonal = similaridade_sem_diagonal[~np.isnan(similaridade_sem_diagonal)]

    # produtos sem vizinhos uteis
    if cooc_sem_diagonal.size > 0:
        serie_cooc = pd.Series(cooc_sem_diagonal)
    else:
        serie_cooc = pd.Series([])

    matriz_binaria = (X != 0).astype(int)
    cooc_upper = (matriz_binaria @ matriz_binaria.T)[np.triu_indices(n_produtos, k=1)]
    total_pares_produtos = len(cooc_upper)

    proporcao_produtos_ate_2_compradores = (compradores_por_produto <= 2).mean() if n_produtos > 0 else np.nan
    proporcao_pares_cooc_zero = (cooc_upper == 0).mean() if total_pares_produtos > 0 else np.nan
    proporcao_pares_cooc_menor_5 = (cooc_upper < 5).mean() if total_pares_produtos > 0 else np.nan
    proporcao_sim_baixa = (sim_fora_diagonal <= 0.01).mean() if sim_fora_diagonal.size > 0 else np.nan

    produtos_sem_vizinhos_uteis = pd.DataFrame({'id_produto': ids_produtos})
    if not df_similaridade.empty:
        max_sim_por_produto = df_similaridade.groupby(df_similaridade.index).max().max(axis=1).rename('max_similaridade_vizinho')
        produtos_sem_vizinhos_uteis = produtos_sem_vizinhos_uteis.merge(max_sim_por_produto, left_on='id_produto', right_index=True, how='left')
        produtos_sem_vizinhos_uteis['max_similaridade_vizinho'] = produtos_sem_vizinhos_uteis['max_similaridade_vizinho'].fillna(0)
        produtos_sem_vizinhos_uteis = produtos_sem_vizinhos_uteis[produtos_sem_vizinhos_uteis['max_similaridade_vizinho'] <= limiar_sim_util].copy()
        produtos_sem_vizinhos_uteis['motivo'] = 'todas as similaridades sao zero ou muito proximas de zero'

    diagnosticos = []
    if not np.isnan(proporcao_produtos_ate_2_compradores) and proporcao_produtos_ate_2_compradores >= 0.5:
        diagnosticos.append('Baixa robustez: a maioria dos produtos tem poucos usuarios compradores.')
    if not np.isnan(proporcao_pares_cooc_zero) and proporcao_pares_cooc_zero >= 0.5:
        diagnosticos.append('Coocorrencia fraca: muitos pares de produtos nunca aparecem juntos.')
    if not np.isnan(proporcao_pares_cooc_menor_5) and proporcao_pares_cooc_menor_5 >= 0.8:
        diagnosticos.append('Coocorrencia limitada: a maior parte dos pares tem menos de 5 usuarios em comum.')
    if not np.isnan(proporcao_sim_baixa) and proporcao_sim_baixa >= 0.8:
        diagnosticos.append('Baixo poder discriminativo: as similaridades entre itens sao quase sempre proximas de zero.')

    if len(diagnosticos) == 0:
        diagnosticos.append('Diagnostico intermediario: a matriz pode ser utilizavel, mas a qualidade da vizinhanca item-item depende de ajustes de filtro e volume de interacoes.')

    diagnostico_final = ' '.join(diagnosticos)

    summary = {
        'proporcao_produtos_ate_2_compradores': proporcao_produtos_ate_2_compradores,
        'proporcao_pares_cooc_zero': proporcao_pares_cooc_zero,
        'proporcao_pares_cooc_menor_5': proporcao_pares_cooc_menor_5,
        'proporcao_sim_baixa': proporcao_sim_baixa,
        'diagnostico_final': diagnostico_final,
    }

    df_summary = pd.DataFrame([summary])
    path = None
    if salvar_saida:
        path = save_csv(df_summary, 'dados_sinteticos/dados_tratatos/avaliacao_diagnosticos')

    return {
        'diagnostico': diagnostico_final,
        'summary_df': (df_summary, path),
        'produtos_sem_vizinhos_uteis': produtos_sem_vizinhos_uteis,
    }
