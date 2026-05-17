import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from .utils import save_parquet


def run_00_geracao_dados_simulados(config=None, salvar_saida: bool = True):
    """Gera conjuntos de dados sintéticos. Retorna um dicionário com DataFrames e caminhos salvos.

    A função não lê arquivos externos. Aceita um dicionário de `config` opcional para sobrescrever valores padrão.
    """
    # defaults
    cfg = dict(
        seed=8,
        n_usuarios=1000,
        n_produtos=100,
        n_categorias=10,
        n_marcas=20,
        min_renda=95,
        max_renda=5000,
        max_moradores=10,
        tamanho_espaco=10000,
        n_elegiveis=200,
        n_ofertas=10000,
        n_copias=4,
        mlt_ofr=5,
        P=2, C=5, M=5, E=1,
    )
    if config:
        cfg.update(config)

    np.random.seed(cfg['seed'])

    def amostra_padrao(n):
        return np.random.uniform(0.001, 0.999, n)

    def amostra_espacial(n):
        return np.random.uniform(-cfg['tamanho_espaco'], cfg['tamanho_espaco'], n)

    def pref_socioeco(valor, media, sigma):
        z = (valor - media) / sigma
        pref = np.exp(-(z ** 2) / 2)
        return 0.001 + pref * 0.998

    usuarios = pd.DataFrame({
        'id_usuario': [f'U_{i}' for i in range(cfg['n_usuarios'])],
        'renda_per_capita': np.round(np.random.uniform(cfg['min_renda'], cfg['max_renda'], cfg['n_usuarios']), 2),
        'qtd_moradores': np.random.randint(1, cfg['max_moradores'] + 1, cfg['n_usuarios']),
        'recebe_beneficio': np.random.randint(0, 2, cfg['n_usuarios']),
        'latitude': amostra_espacial(cfg['n_usuarios']),
        'longitude': amostra_espacial(cfg['n_usuarios']),
        'unidade_deslocamento': amostra_padrao(cfg['n_usuarios'])
    })

    sigma_renda = np.std(usuarios['renda_per_capita'], ddof=1)
    sigma_moradores = np.std(usuarios['qtd_moradores'], ddof=1)

    categorias = pd.DataFrame({
        'id_categoria': [f'C_{i}' for i in range(cfg['n_categorias'])],
        'renda_per_capita': np.round(np.random.uniform(cfg['min_renda'], cfg['max_renda'], cfg['n_categorias']), 2),
        'qtd_moradores': np.random.randint(1, cfg['max_moradores'] + 1, cfg['n_categorias']),
        'recebe_beneficio_0': amostra_padrao(cfg['n_categorias']),
        'recebe_beneficio_1': amostra_padrao(cfg['n_categorias'])
    })

    marcas = pd.DataFrame({
        'id_marca': [f'M_{i}' for i in range(cfg['n_marcas'])],
        'renda_per_capita': np.round(np.random.uniform(cfg['min_renda'], cfg['max_renda'], cfg['n_marcas']), 2),
        'qtd_moradores': np.random.randint(1, cfg['max_moradores'] + 1, cfg['n_marcas']),
        'recebe_beneficio_0': amostra_padrao(cfg['n_marcas']),
        'recebe_beneficio_1': amostra_padrao(cfg['n_marcas'])
    })

    combinacoes = list(itertools.product(marcas['id_marca'], categorias['id_categoria']))
    np.random.shuffle(combinacoes)
    combinacoes = combinacoes[:cfg['n_produtos']]

    produtos = pd.DataFrame({
        'id_produto': [f'P_{i}' for i in range(cfg['n_produtos'])],
        'id_marca': [i[0] for i in combinacoes],
        'id_categoria': [i[1] for i in combinacoes]
    })

    np.random.seed(cfg['seed'])
    soma_ofertas = cfg['n_ofertas'] // cfg['n_copias']
    oferta_aleatoria = np.random.choice(range(1, soma_ofertas // cfg['mlt_ofr']), cfg['n_produtos'] - 1, replace=False) * cfg['mlt_ofr']
    oferta_cortes = np.sort(oferta_aleatoria)
    oferta = np.diff(np.concatenate([[0], oferta_cortes, [soma_ofertas]]))
    produtos['oferta'] = oferta

    # Preferências latentes
    def calcula_pref_linha(usuario, df_ref, sigma_renda, sigma_moradores):
        pref_renda = pref_socioeco(usuario['renda_per_capita'], df_ref['renda_per_capita'], sigma_renda)
        pref_moradores = pref_socioeco(usuario['qtd_moradores'], df_ref['qtd_moradores'], sigma_renda)
        col_beneficio = 'recebe_beneficio_0' if usuario['recebe_beneficio'] == 0 else 'recebe_beneficio_1'
        pref_beneficio = df_ref[col_beneficio]
        return np.mean([pref_renda, pref_moradores, pref_beneficio])

    def adiciona_pref_referente(id_ref, df_usuarios, df_ref, sigma_renda, sigma_moradores):
        df_usuarios[id_ref] = df_usuarios.apply(
            calcula_pref_linha,
            axis=1,
            df_ref=df_ref,
            sigma_renda=sigma_renda,
            sigma_moradores=sigma_moradores,
        )

    for _, marca in marcas.iterrows():
        adiciona_pref_referente(marca['id_marca'], usuarios, marca, sigma_renda, sigma_moradores)

    for _, categoria in categorias.iterrows():
        adiciona_pref_referente(categoria['id_categoria'], usuarios, categoria, sigma_renda, sigma_moradores)

    for _, produto in produtos.iterrows():
        usuarios[produto['id_produto']] = amostra_padrao(cfg['n_usuarios'])

    # Simulação de ofertas
    registros = []
    with tqdm(total=produtos['oferta'].sum(), desc="Progresso Total") as pbar:
        for _, produto in produtos.iterrows():
            for _round in range(int(produto['oferta'])):
                lat_oferta = amostra_espacial(1)[0]
                lon_oferta = amostra_espacial(1)[0]

                def dist_usuario_oferta(usuario, lat_oferta=lat_oferta, lon_oferta=lon_oferta):
                    return np.abs(usuario['latitude'] - lat_oferta) + np.abs(usuario['longitude'] - lon_oferta)

                def pr_usuario_oferta(usuario, id_produto=produto['id_produto'], id_marca=produto['id_marca'], id_categoria=produto['id_categoria']):
                    e = amostra_padrao(1)[0]
                    return (cfg['P'] * usuario[id_produto] + cfg['C'] * usuario[id_marca] + cfg['M'] * usuario[id_categoria] + cfg['E'] * e) / (cfg['P'] + cfg['C'] + cfg['M'] + cfg['E'])

                usuarios['pref'] = usuarios.apply(pr_usuario_oferta, axis=1)
                usuarios['dist'] = usuarios.apply(dist_usuario_oferta, axis=1)

                elegiveis = usuarios.nsmallest(cfg['n_elegiveis'], 'dist')
                adquirentes = elegiveis.nlargest(4, 'pref')

                for _, adquirente in adquirentes.iterrows():
                    registros.append({
                        'id_produto': produto['id_produto'],
                        'id_usuario': adquirente['id_usuario'],
                        'deslocamento': adquirente['dist']
                    })

                pbar.update(1)

    itens = pd.DataFrame(registros)

    # Salvamento (com timestamp) — opcional
    usuarios_path = produtos_path = marcas_path = categorias_path = itens_path = None
    if salvar_saida:
        usuarios_path = save_parquet(usuarios, 'dados_sinteticos/usuarios', engine='pyarrow', compression='snappy')
        produtos_path = save_parquet(produtos, 'dados_sinteticos/produtos', engine='pyarrow', compression='snappy')
        marcas_path = save_parquet(marcas, 'dados_sinteticos/marcas', engine='pyarrow', compression='snappy')
        categorias_path = save_parquet(categorias, 'dados_sinteticos/categorias', engine='pyarrow', compression='snappy')
        itens_path = save_parquet(itens, 'dados_sinteticos/itens', engine='pyarrow', compression='snappy')

    return {
        'usuarios': (usuarios, usuarios_path),
        'produtos': (produtos, produtos_path),
        'marcas': (marcas, marcas_path),
        'categorias': (categorias, categorias_path),
        'itens': (itens, itens_path),
    }
