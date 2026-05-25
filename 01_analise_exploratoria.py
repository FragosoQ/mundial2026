"""
01_analise_exploratoria.py
Análise Exploratória e Engenharia de Features
Pipeline de Machine Learning para Predição de Risco de Lesão - Mundial 2026
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURAÇÃO DE CAMINHOS - CORREÇÃO: Lê da raiz do projeto
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# Se os ficheiros estiverem na subpasta 'dados', use:
# DATA_DIR = os.path.join(BASE_DIR, 'dados')
# Se os ficheiros estiverem na RAIZ (mesma pasta do script):
DATA_DIR = BASE_DIR
OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("="*60)
print("[01] ANÁLISE EXPLORATÓRIA E ENGENHARIA DE FEATURES")
print("="*60)
print(f"Diretório de dados: {DATA_DIR}")

# ============================================================
# 1. CARREGAMENTO DOS DADOS
# ============================================================
print("\n[1/6] Carregamento dos dados...")

# Verificar ficheiros disponíveis
ficheiros_esperados = [
    'lesoes_jogadores_mundial_REAL.csv',
    'jogadores_copa_2026.csv',
    'calendario_selecoes.csv',
    'sedes_mundial.csv'
]

for f in ficheiros_esperados:
    caminho = os.path.join(DATA_DIR, f)
    if os.path.exists(caminho):
        print(f"  ✓ {f} encontrado")
    else:
        print(f"  ✗ {f} NÃO ENCONTRADO em: {caminho}")

# Ficheiro principal de lesões
df_lesoes = pd.read_csv(os.path.join(
    DATA_DIR, 'lesoes_jogadores_mundial_REAL.csv'))
print(
    f"\n✓ Ficheiro de lesões: {df_lesoes.shape[0]} registos, {df_lesoes.shape[1]} colunas")
print(f"  Colunas: {list(df_lesoes.columns)}")

# Ficheiro de jogadores da Copa 2026 (com BOM UTF-8)
df_jogadores = pd.read_csv(
    os.path.join(DATA_DIR, 'jogadores_copa_2026.csv'),
    encoding='utf-8-sig'
)
print(f"✓ Ficheiro de jogadores Copa 2026: {df_jogadores.shape[0]} registos")

# Ficheiro de calendário das seleções (separador ;)
try:
    df_calendario = pd.read_csv(
        os.path.join(DATA_DIR, 'calendario_selecoes.csv'),
        sep=';',
        encoding='utf-8'
    )
except UnicodeDecodeError:
    df_calendario = pd.read_csv(
        os.path.join(DATA_DIR, 'calendario_selecoes.csv'),
        sep=';',
        encoding='latin-1'
    )
print(f"✓ Ficheiro de calendário: {df_calendario.shape[0]} seleções")

# Ficheiro de sedes (separador ;)
try:
    df_sedes = pd.read_csv(
        os.path.join(DATA_DIR, 'sedes_mundial.csv'),
        sep=';',
        encoding='utf-8'
    )
except UnicodeDecodeError:
    df_sedes = pd.read_csv(
        os.path.join(DATA_DIR, 'sedes_mundial.csv'),
        sep=';',
        encoding='latin-1'
    )
print(f"✓ Ficheiro de sedes: {df_sedes.shape[0]} estádios")

# ============================================================
# 2. TRATAMENTO DE DADOS
# ============================================================
print("\n[2/6] Tratamento e limpeza dos dados...")

# Converter datas
df_lesoes['data_inicio'] = pd.to_datetime(
    df_lesoes['data_inicio'], errors='coerce')
df_lesoes['data_fim'] = pd.to_datetime(df_lesoes['data_fim'], errors='coerce')

# Calcular duração real se não existir (em dias)
df_lesoes['duracao_real'] = (
    df_lesoes['data_fim'] - df_lesoes['data_inicio']).dt.days

# Preencher dias_parado onde está NA com a duração real
df_lesoes['dias_parado'] = df_lesoes['dias_parado'].fillna(
    df_lesoes['duracao_real'])

# Preencher jogos_perdidos onde está NA com estimativa (dias/7 aproximadamente)
df_lesoes['jogos_perdidos'] = df_lesoes['jogos_perdidos'].fillna(
    (df_lesoes['dias_parado'] / 7).round()
)

# Extrair ano da época (formato "19/20" -> 2019)


def extrair_ano_epoca(epoca_str):
    try:
        ano_inicio = int(epoca_str.split('/')[0])
        if ano_inicio >= 50:  # Anos 50-99 = 1950-1999
            return 1900 + ano_inicio
        else:  # Anos 00-49 = 2000-2049
            return 2000 + ano_inicio
    except:
        return np.nan


df_lesoes['ano_epoca'] = df_lesoes['epoca'].apply(extrair_ano_epoca)

print(f"✓ Datas convertidas e valores NA tratados")
print(
    f"  Período: {df_lesoes['ano_epoca'].min():.0f} - {df_lesoes['ano_epoca'].max():.0f}")
print(f"  Valores NA em dias_parado: {df_lesoes['dias_parado'].isna().sum()}")
print(
    f"  Valores NA em jogos_perdidos: {df_lesoes['jogos_perdidos'].isna().sum()}")

# ============================================================
# 3. AGREGAÇÃO POR JOGADOR E ÉPOCA
# ============================================================
print("\n[3/6] Agregação por jogador e época...")

# CORREÇÃO PRINCIPAL: Incluir colunas categóricas no agg com 'first'
# para manter informação do jogador após o groupby

df_agregado = df_lesoes.groupby(['id_jogador', 'epoca']).agg({
    'dias_parado': ['sum', 'mean', 'max', 'count'],
    'jogos_perdidos': ['sum', 'mean'],
    'idade_atual': 'first',
    'posicao': 'first',
    'nacionalidade': 'first',
    'liga': 'first',
    'competicao': 'first'
}).reset_index()

# Aplanar nomes das colunas multi-index
df_agregado.columns = [
    'id_jogador', 'epoca',
    'total_dias_parado', 'media_dias_parado', 'max_dias_parado', 'num_lesoes',
    'total_jogos_perdidos', 'media_jogos_perdidos',
    'idade_atual', 'posicao', 'nacionalidade', 'liga', 'competicao'
]

print(f"✓ Agregação concluída: {df_agregado.shape[0]} registos")
print(f"  Colunas: {list(df_agregado.columns)}")

# ============================================================
# 4. FEATURE ENGINEERING
# ============================================================
print("\n[4/6] Engenharia de features...")

# Ordenar por jogador e época para calcular histórico
df_agregado = df_agregado.sort_values(['id_jogador', 'epoca'])

# Features de histórico de lesões (cumulativas)
df_agregado['lesoes_acumuladas'] = df_agregado.groupby('id_jogador')[
    'num_lesoes'].cumsum()
df_agregado['dias_parado_acumulados'] = df_agregado.groupby(
    'id_jogador')['total_dias_parado'].cumsum()

# Média móvel de lesões (últimas 3 épocas)
df_agregado['media_lesoes_3epocas'] = df_agregado.groupby('id_jogador')['num_lesoes'].transform(
    lambda x: x.rolling(window=3, min_periods=1).mean()
)

# Taxa de severidade (dias parado por jogo perdido)
df_agregado['taxa_severidade'] = df_agregado['total_dias_parado'] / (
    df_agregado['total_jogos_perdidos'] + 1
)

# Categorizar idade


def categorizar_idade(idade):
    if pd.isna(idade):
        return 'Desconhecido'
    elif idade < 23:
        return 'Jovem'
    elif idade < 30:
        return 'Experiente'
    else:
        return 'Veterano'


df_agregado['faixa_etaria'] = df_agregado['idade_atual'].apply(
    categorizar_idade)

# Risco por posição (média de lesões na posição)
posicao_risco = df_agregado.groupby('posicao')['num_lesoes'].mean().to_dict()
df_agregado['risco_posicao'] = df_agregado['posicao'].map(posicao_risco)

# Risco por liga
liga_risco = df_agregado.groupby('liga')['num_lesoes'].mean().to_dict()
df_agregado['risco_liga'] = df_agregado['liga'].map(liga_risco)

print(f"✓ Features criadas:")
print(f"  - lesoes_acumuladas, dias_parado_acumulados")
print(f"  - media_lesoes_3epocas, taxa_severidade")
print(f"  - faixa_etaria, risco_posicao, risco_liga")

# ============================================================
# 5. TARGET VARIABLE (RISCO DE LESÃO)
# ============================================================
print("\n[5/6] Definição da variável target...")

# Definir risco como alta probabilidade de lesão na próxima época
# Usamos percentil 75 como threshold para "alto risco"
threshold_risco = df_agregado['num_lesoes'].quantile(0.75)
df_agregado['risco_lesao'] = (
    df_agregado['num_lesoes'] >= threshold_risco).astype(int)

print(f"✓ Threshold de risco definido: {threshold_risco:.2f} lesões/época")
print(
    f"  Classe 0 (Baixo risco): {(df_agregado['risco_lesao']==0).sum()} ({(1-df_agregado['risco_lesao'].mean())*100:.1f}%)")
print(
    f"  Classe 1 (Alto risco):  {df_agregado['risco_lesao'].sum()} ({df_agregado['risco_lesao'].mean()*100:.1f}%)")

# ============================================================
# 6. MERGE COM DADOS DA COPA 2026
# ============================================================
print("\n[6/6] Integração com dados da Copa 2026...")

# Mapeamento de nacionalidades (inglês -> português)
mapeamento_paises = {
    'England': 'Inglaterra',
    'Spain': 'Espanha',
    'Germany': 'Alemanha',
    'France': 'França',
    'Brazil': 'Brasil',
    'Argentina': 'Argentina',
    'Portugal': 'Portugal',
    'Belgium': 'Bélgica',
    'Netherlands': 'Holanda',
    'Italy': 'Itália',
    'Croatia': 'Croácia',
    'Uruguay': 'Uruguai',
    'Colombia': 'Colômbia',
    'Mexico': 'México',
    'USA': 'Estados Unidos',
    'Korea, South': 'Coreia do Sul',
    'Japan': 'Japão',
    'Senegal': 'Senegal',
    'Morocco': 'Marrocos',
    'Tunisia': 'Tunísia',
    'Egypt': 'Egito',
    'Iran': 'Irã',
    'Saudi Arabia': 'Arábia Saudita',
    'Australia': 'Austrália',
    'Ecuador': 'Equador',
    'Poland': 'Polónia',
    'Serbia': 'Sérvia',
    'Switzerland': 'Suíça',
    'Wales': 'País de Gales',
    'Scotland': 'Escócia',
    'Denmark': 'Dinamarca',
    'Sweden': 'Suécia',
    'Norway': 'Noruega',
    'Austria': 'Áustria',
    'Czech Republic': 'República Tcheca',
    'Ukraine': 'Ucrânia',
    'Russia': 'Rússia',
    'Turkey': 'Turquia',
    'Greece': 'Grécia',
    'Nigeria': 'Nigéria',
    'Ghana': 'Gana',
    'Cameroon': 'Camarões',
    'Ivory Coast': 'Costa do Marfim',
    'Algeria': 'Argélia',
    'DR Congo': 'RD Congo',
    'Mali': 'Mali',
    'Burkina Faso': 'Burquina Faso',
    'South Africa': 'África do Sul',
    'Canada': 'Canadá',
    'Costa Rica': 'Costa Rica',
    'Panama': 'Panamá',
    'Honduras': 'Honduras',
    'Jamaica': 'Jamaica',
    'New Zealand': 'Nova Zelândia',
    'Paraguay': 'Paraguai',
    'Chile': 'Chile',
    'Peru': 'Peru',
    'Venezuela': 'Venezuela',
    'Bolivia': 'Bolívia',
    'Haiti': 'Haiti',
    'Curacao': 'Curaçao',
    'Cape Verde': 'Cabo Verde',
    'Iraq': 'Iraque',
    'Uzbekistan': 'Uzbequistão'
}

# Tentar fazer merge com calendário
# Primeiro normalizar nomes no calendário
df_calendario['nacionalidade_merge'] = df_calendario['nacionalidade'].str.strip()

# Normalizar nomes no dataset de lesões
df_agregado['nacionalidade_merge'] = df_agregado['nacionalidade'].str.strip()

# Tentar merge direto primeiro
df_merged = df_agregado.merge(
    df_calendario[['nacionalidade', 'dias_descanso_media',
                   'jogos_fase_grupos', 'altitude_media']],
    left_on='nacionalidade_merge',
    right_on='nacionalidade',
    how='left'
)

# Verificar quantos fizeram match
match_count = df_merged['dias_descanso_media'].notna().sum()
print(
    f"✓ Merge com calendário: {match_count}/{df_merged.shape[0]} registos com match ({match_count/df_merged.shape[0]*100:.1f}%)")

# Se match for baixo, tentar com mapeamento
if match_count < df_merged.shape[0] * 0.5:
    print("  ⚠ Match baixo, tentando com mapeamento de nomes...")
    df_agregado['nacionalidade_mapped'] = df_agregado['nacionalidade_merge'].replace(
        mapeamento_paises)

    df_merged = df_agregado.merge(
        df_calendario[['nacionalidade', 'dias_descanso_media',
                       'jogos_fase_grupos', 'altitude_media']],
        left_on='nacionalidade_mapped',
        right_on='nacionalidade',
        how='left'
    )
    match_count = df_merged['dias_descanso_media'].notna().sum()
    print(
        f"  ✓ Após mapeamento: {match_count}/{df_merged.shape[0]} registos com match ({match_count/df_merged.shape[0]*100:.1f}%)")

# Preencher valores NA do calendário com médias
df_merged['dias_descanso_media'] = df_merged['dias_descanso_media'].fillna(
    df_merged['dias_descanso_media'].mean()
)
df_merged['jogos_fase_grupos'] = df_merged['jogos_fase_grupos'].fillna(3)
df_merged['altitude_media'] = df_merged['altitude_media'].fillna(
    df_merged['altitude_media'].median()
)

# ============================================================
# 7. EXPORTAÇÃO
# ============================================================
print("\n[Exportação] Salvando dados processados...")

# Dataset principal para modelagem
cols_modelagem = [
    'id_jogador', 'epoca', 'posicao', 'nacionalidade',
    'idade_atual', 'faixa_etaria',
    'total_dias_parado', 'media_dias_parado', 'max_dias_parado',
    'num_lesoes', 'total_jogos_perdidos', 'media_jogos_perdidos',
    'lesoes_acumuladas', 'dias_parado_acumulados',
    'media_lesoes_3epocas', 'taxa_severidade',
    'risco_posicao', 'risco_liga',
    'dias_descanso_media', 'jogos_fase_grupos', 'altitude_media',
    'risco_lesao'
]

# Apenas colunas que existem
cols_existentes = [c for c in cols_modelagem if c in df_merged.columns]
df_modelagem = df_merged[cols_existentes].copy()

# Salvar CSV
df_modelagem.to_csv(os.path.join(
    OUTPUT_DIR, 'dados_modelagem.csv'), index=False)
print(
    f"✓ dados_modelagem.csv salvo ({df_modelagem.shape[0]} linhas, {df_modelagem.shape[1]} colunas)")

# Salvar também o agregado completo
df_agregado.to_csv(os.path.join(
    OUTPUT_DIR, 'dados_agregados.csv'), index=False)
print(f"✓ dados_agregados.csv salvo")

# Salvar estatísticas descritivas
estatisticas = df_modelagem.describe()
estatisticas.to_csv(os.path.join(OUTPUT_DIR, 'estatisticas_descritivas.csv'))
print(f"✓ estatisticas_descritivas.csv salvo")

# Relatório de qualidade
print("\n" + "="*60)
print("RELATÓRIO DE QUALIDADE DOS DADOS")
print("="*60)
print(f"Total de jogadores únicos: {df_agregado['id_jogador'].nunique()}")
print(f"Total de épocas: {df_agregado['epoca'].nunique()}")
print(f"Posições: {df_agregado['posicao'].nunique()} distintas")
print(f"  - {', '.join(df_agregado['posicao'].unique()[:5])}...")
print(f"Nacionalidades: {df_agregado['nacionalidade'].nunique()} distintas")
print(f"Valores NA no dataset final: {df_modelagem.isna().sum().sum()}")
print(f"\nDistribuição do target (risco_lesao):")
print(df_modelagem['risco_lesao'].value_counts())
print(f"\nEstatísticas das features numéricas:")
print(df_modelagem[['total_dias_parado', 'num_lesoes',
      'idade_atual']].describe().round(2))

print("\n" + "="*60)
print("✓ ANÁLISE EXPLORATÓRIA CONCLUÍDA COM SUCESSO!")
print("="*60)
