import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

# Configurações de semente para reprodutibilidade
np.random.seed(42)
random.seed(42)

# --- 1. Tabela de Usuários ---
num_usuarios = 500
cidades = ["São Paulo", "Guarulhos", "Campinas", "São Bernardo do Campo", "Santo André", "Osasco", "Mogi das Cruzes", "Diadema"]

rendas = []
for _ in range(num_usuarios):
    r = random.random()
    if r < 0.4: # 40% em pobreza extrema (até R$ 218)
        renda = np.random.uniform(50, 218)
    elif r < 0.8: # 40% em pobreza (R$ 219 a R$ 706)
        renda = np.random.uniform(219, 706)
    else: # 20% baixa renda (R$ 707 a R$ 1412)
        renda = np.random.uniform(707, 1412)
    rendas.append(round(renda, 2))

usuarios = {
    "id_usuario": range(1, num_usuarios + 1),
    "renda_per_capita": rendas,
    "recebe_beneficio": [True if r < 706 else np.random.choice([True, False], p=[0.3, 0.7]) for r in rendas],
    "qtd_moradores": np.random.choice([1, 2, 3, 4, 5, 6, 7, 8], num_usuarios, p=[0.1, 0.2, 0.25, 0.2, 0.1, 0.08, 0.04, 0.03]),
    "cidade": np.random.choice(cidades, num_usuarios)
}
df_usuarios = pd.DataFrame(usuarios)

# --- 2. Tabela de Produtos ---
produtos_data = [
    (1, "Arroz 5kg", "Grãos", 25.00, 365), (2, "Feijão 1kg", "Grãos", 8.50, 180),
    (3, "Macarrão 500g", "Massas", 4.00, 365), (4, "Leite Integral 1L", "Laticínios", 5.50, 120),
    (5, "Óleo de Soja 900ml", "Óleos", 7.00, 365), (6, "Açúcar 1kg", "Mercearia", 4.50, 365),
    (7, "Café 500g", "Mercearia", 18.00, 180), (8, "Maçã (kg)", "Frutas", 9.00, 15),
    (9, "Banana (kg)", "Frutas", 6.00, 7), (10, "Batata (kg)", "Legumes", 5.00, 20),
    (11, "Cebola (kg)", "Legumes", 4.50, 30), (12, "Ovos (Dúzia)", "Proteína", 12.00, 30),
    (13, "Frango (kg)", "Proteína", 15.00, 90), (14, "Pão Francês (unidade)", "Padaria", 0.50, 2),
    (15, "Iogurte 170g", "Laticínios", 3.00, 30), (16, "Farinha de Trigo 1kg", "Mercearia", 5.50, 180),
    (17, "Cenoura (kg)", "Legumes", 6.00, 20), (18, "Tomate (kg)", "Legumes", 8.00, 10),
    (19, "Sardinha em Lata", "Proteína", 4.50, 730), (20, "Bolacha Salgada", "Mercearia", 3.50, 180)
]
df_produtos = pd.DataFrame(produtos_data, columns=["id_produto", "nome", "categoria", "preco_medio", "perecibilidade_dias"])

# --- 3. Tabela de Estabelecimentos ---
num_estabelecimentos = 50
estabelecimentos = {
    "id_estabelecimento": range(1, num_estabelecimentos + 1),
    "tipo_comercio": np.random.choice(["Supermercado", "Atacado", "Padaria", "Hortifruti", "Restaurante"], num_estabelecimentos),
    "cidade": np.random.choice(cidades, num_estabelecimentos),
    "volume_estimado": np.random.uniform(50, 1000, num_estabelecimentos).round(2)
}
df_estabelecimentos = pd.DataFrame(estabelecimentos)

# --- 4. Tabela Usuário e Produto (Interações com Distância) ---
num_interacoes = 5000
interacoes = []
start_date = datetime(2026, 1, 1)

# Para garantir que as cidades sejam mapeadas corretamente para o cálculo da distância
user_city_map = df_usuarios.set_index("id_usuario")["cidade"].to_dict()
est_city_map = df_estabelecimentos.set_index("id_estabelecimento")["cidade"].to_dict()

for i in range(num_interacoes):
    user_id = random.randint(1, num_usuarios)
    prod_id = random.randint(1, 20)
    
    # Filtra estabelecimentos na mesma cidade do usuário (regra de negócio lógica)
    user_city = user_city_map[user_id]
    est_na_cidade = df_estabelecimentos[df_estabelecimentos["cidade"] == user_city]["id_estabelecimento"].values
    
    if est_na_cidade.size > 0:
        est_id = int(np.random.choice(est_na_cidade))
        distancia_km = round(random.uniform(0.5, 5.0), 2)
    else:
        # Se não houver estabelecimento na mesma cidade, escolhe um aleatório e atribui distância maior
        est_id = random.randint(1, num_estabelecimentos)
        distancia_km = round(random.uniform(5.0, 25.0), 2)
        
    quantidade = round(random.uniform(0.5, 5.0), 2)
    days_delta = random.randint(0, 75)
    timestamp = start_date + timedelta(days=days_delta, hours=random.randint(8, 20))
    
    interacoes.append([i+1, user_id, prod_id, est_id, quantidade, distancia_km, timestamp])

df_interacoes = pd.DataFrame(interacoes, columns=["id_usuario_produto", "id_usuario", "id_produto", "id_estabelecimento", "quantidade", "distancia_km", "timestamp"])

# --- Salvar os arquivos CSV ---
df_usuarios.to_csv("usuarios.csv", index=False)
df_produtos.to_csv("produtos.csv", index=False)
df_estabelecimentos.to_csv("estabelecimentos.csv", index=False)
df_interacoes.to_csv("interacoes.csv", index=False)

print("Arquivos CSV gerados com sucesso: usuarios.csv, produtos.csv, estabelecimentos.csv, interacoes.csv")
