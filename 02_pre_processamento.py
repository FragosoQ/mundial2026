# 02_pre_processamento.py
import pandas as pd
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
import os

# 1. Carregamento
if not os.path.exists('dados_base_agregados.csv'):
    print("ERRO: 'dados_base_agregados.csv' não encontrado.")
    exit()

df = pd.read_csv('dados_base_agregados.csv')

# ==============================================================================
# INJEÇÃO E TRATAMENTO DOS NOVOS DADOS DE JOGOS (FADIGA CRÓNICA)
# ==============================================================================
if os.path.exists('jogos_selecoes.csv'):
    print("✓ Ficheiro 'jogos_selecoes.csv' detetado. A integrar no pipeline...")
    df_jogos = pd.read_csv('jogos_selecoes.csv')

    # Preencher visualmente os grupos vazios causados pela formatação do Excel
    df_jogos['Seleção'] = df_jogos['Seleção'].ffill()

    # Função interna robusta para converter os intervalos de texto em números puros
    def texto_para_num(texto):
        if pd.isna(texto):
            return 0.0

        # Converter para string, remover espaços e passar para minúsculas
        texto = str(texto).strip().lower()

        # 1. Ignorar o conteúdo dentro de parênteses (Ex: "34 (apertura + clausura)" vira "34")
        if '(' in texto:
            texto = texto.split('(')[0].strip()

        if 'anfitrião' in texto or 'anfitreão' in texto or texto == '':
            return 0.0

        if '~' in texto:
            return float(texto.replace('~', ''))

        # 2. Tratar intervalos reais de jogos (Ex: "6 a 8" -> Média: 7.0)
        # Usamos espaços ao redor do ' a ' para garantir que não quebra palavras
        if ' a ' in texto:
            partes = texto.split(' a ')
            try:
                return (float(partes[0].strip()) + float(partes[1].strip())) / 2
            except:
                pass
        elif 'a' in texto and not texto.isalpha():
            # Fallback caso esteja colado (Ex: "6a8")
            partes = texto.split('a')
            try:
                return (float(partes[0].strip()) + float(partes[1].strip())) / 2
            except:
                pass

        # 3. Conversão direta final
        try:
            return float(texto)
        except:
            return 0.0

    # Aplica a limpeza nas colunas de desgaste
    df_jogos['jogos_qualificacao_clean'] = df_jogos['Jogos na Qualificação (Pela Seleção)*'].apply(
        texto_para_num)
    df_jogos['jogos_liga_local_clean'] = df_jogos['Jogos na Principal Liga Local (Clubes)'].apply(
        texto_para_num)

    # Isolar apenas as colunas limpas para o merge
    df_jogos_limpo = df_jogos[[
        'Seleção', 'jogos_qualificacao_clean', 'jogos_liga_local_clean']]

    # Cruzar com os dados agregados originais através da Nacionalidade
    df = pd.merge(df, df_jogos_limpo, left_on='nacionalidade',
                  right_on='Seleção', how='left')

    # Limpar coluna duplicada do merge e preencher eventuais falhas com a média
    if 'Seleção' in df.columns:
        df = df.drop(columns=['Seleção'])
    df['jogos_qualificacao_clean'] = df['jogos_qualificacao_clean'].fillna(
        df['jogos_qualificacao_clean'].median())
    df['jogos_liga_local_clean'] = df['jogos_liga_local_clean'].fillna(
        df['jogos_liga_local_clean'].median())

    # --- ENGENHARIA DE FEATURES AVANÇADA ---
    df['indice_desgaste_previo'] = df['jogos_qualificacao_clean'] + \
        df['jogos_liga_local_clean']

    if 'altitude_media' in df.columns:
        df['stresse_viagem_vulnerabilidade'] = df['jogos_qualificacao_clean'] * \
            df['altitude_media']

    if 'dias_descanso_media' in df.columns and 'risco_biomecanico' in df.columns:
        df['racio_desgaste_descanso'] = df['indice_desgaste_previo'] / \
            (df['dias_descanso_media'] + 0.1)

print(
    f"✓ Base de dados expandida. Colunas atuais para treino: {list(df.columns)}")
# ==============================================================================

# 2. Separação X e y (Removida a duplicação)
# 'dias_parado_max' é o construtor do target — NÃO pode ser feature (data leakage)
cols_drop = ['id_jogador', 'epoca', 'risco_lesao',
             'dias_parado_max', 'nome_jogador', 'nacionalidade']
cols_drop = [c for c in cols_drop if c in df.columns]

X = df.drop(columns=cols_drop)
y = df['risco_lesao']

# 3. Tratamento de Nulos ANTES do Encoding
imputer_num = SimpleImputer(strategy='median')
num_cols = X.select_dtypes(include=[np.number]).columns
X[num_cols] = imputer_num.fit_transform(X[num_cols])

imputer_cat = SimpleImputer(strategy='constant', fill_value='Desconhecido')
cat_cols = X.select_dtypes(exclude=[np.number]).columns
X[cat_cols] = imputer_cat.fit_transform(X[cat_cols])

# 4. Encoding apenas da posição
if 'posicao' in X.columns:
    X = pd.get_dummies(X, columns=['posicao'], drop_first=True)

# 5. REMOÇÃO de nacionalidade (evita "death by one-hot" e overfitting)
nacionalidade_cols = [c for c in X.columns if c.startswith('nacionalidade_')]
if nacionalidade_cols:
    print(f"A remover {len(nacionalidade_cols)} colunas de nacionalidade.")
    X = X.drop(columns=nacionalidade_cols)

# 6. Seleção de Features (ADICIONADAS AS NOVAS COLUNAS DE DESGASTE AQUI)
colunas_numericas_relevantes = [
    'idade_atual',
    'dias_descanso_media',
    'altitude_media',
    'jogos_fase_grupos',
    'carga_lesao_historica',
    'num_lesoes',
    'idade_x_jogos',
    'idade_x_altitude',
    'descanso_relativo',
    'risco_biomecanico',
    # --- Injeções das novas variáveis do ficheiro jogos_selecoes ---
    'jogos_qualificacao_clean',
    'jogos_liga_local_clean',
    'indice_desgaste_previo',
    'stresse_viagem_vulnerabilidade',
    'racio_desgaste_descanso'
]

# Filtrar apenas colunas que realmente existem
colunas_numericas_relevantes = [
    c for c in colunas_numericas_relevantes if c in X.columns]

colunas_posicao = [c for c in X.columns if c.startswith('posicao_')]
colunas_finais = colunas_numericas_relevantes + colunas_posicao

# Verifica existência
missing = [c for c in colunas_finais if c not in X.columns]
if missing:
    print(f"AVISO: Colunas ignoradas por não existirem: {missing}")
    colunas_finais = [c for c in colunas_finais if c in X.columns]

X = X[colunas_finais].copy()

# 7. Divisão estratificada
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=42)

# 8. Escalonamento (numéricas contínuas; exclui ordinais/inteiros)
colunas_escalonar = [c for c in colunas_numericas_relevantes
                     if c in X.columns and c not in ['risco_biomecanico', 'num_lesoes']]
if colunas_escalonar:
    scaler = StandardScaler()
    X_train[colunas_escalonar] = scaler.fit_transform(
        X_train[colunas_escalonar])
    X_test[colunas_escalonar] = scaler.transform(X_test[colunas_escalonar])
else:
    scaler = None

# 9. Serialização
for nome, obj in [('X_train', X_train), ('X_test', X_test),
                  ('y_train', y_train), ('y_test', y_test), ('scaler', scaler)]:
    with open(f'{nome}.pkl', 'wb') as f:
        pickle.dump(obj, f)

print("--- Pré-processamento concluído ---")
print(f"Features finais ({len(X.columns)}): {list(X.columns)}")
print(f"Treino: {X_train.shape}, Teste: {X_test.shape}")
