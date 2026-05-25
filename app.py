import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ==============================================================================
# 1. CONFIGURAÇÃO DA PÁGINA & ESTILOS
# ==============================================================================
st.set_page_config(
    page_title="PrevLesão 2026 - Sistema de Previsão de Risco",
    page_icon="⚽",
    layout="wide"
)

st.title("⚽ Sistema de Previsão de Risco de Lesão - Mundial 2026")
st.markdown("""
Esta plataforma utiliza os modelos preditivos treinados para avaliar o risco clínico
de lesões de jogadores com base em fatores biomecânicos, histórico e no contexto geográfico da sua seleção.
""")

# ==============================================================================
# 2. CARREGAMENTO DOS ARTEFATOS DO PIPELINE (CACHE)
# ==============================================================================


@st.cache_resource
def carregar_artefatos():
    try:
        # Identificar o melhor modelo a partir da tabela de métricas
        if os.path.exists('tabela_metricas.csv'):
            try:
                df_metricas = pd.read_csv(
                    'tabela_metricas.csv', encoding='utf-8')
            except Exception:
                df_metricas = pd.read_csv(
                    'tabela_metricas.csv', encoding='latin-1')
            melhor_modelo_nome = df_metricas.iloc[0]['Modelo']
            melhor_threshold = float(df_metricas.iloc[0]['Threshold_Otimo'])
        else:
            melhor_modelo_nome = 'Random Forest'
            melhor_threshold = 0.35

        # Carregar dicionário de modelos
        if os.path.exists('modelos_e_previsoes.pkl'):
            with open('modelos_e_previsoes.pkl', 'rb') as f:
                resultados = pickle.load(f)
            modelo_vencedor = resultados[melhor_modelo_nome]['modelo']
        else:
            modelo_vencedor = None

        # Carregar o scaler e colunas
        scaler = None
        if os.path.exists('scaler.pkl'):
            with open('scaler.pkl', 'rb') as f:
                scaler = pickle.load(f)

        colunas_treino = []
        if os.path.exists('colunas_treino.pkl'):
            with open('colunas_treino.pkl', 'rb') as f:
                colunas_treino = pickle.load(f)

        return {
            'modelo': modelo_vencedor,
            'nome': melhor_modelo_nome,
            'threshold': melhor_threshold,
            'scaler': scaler,
            'colunas': colunas_treino
        }
    except Exception as e:
        st.error(f"Erro ao carregar ficheiros de modelagem: {e}")
        return None


pipeline = carregar_artefatos()

# ==============================================================================
# 3. CARREGAMENTO SEGURO E EM CACHE DOS DATASETS DE APOIO
# ==============================================================================


@st.cache_data
def carregar_dados_apoio():
    # Dataset Histórico de Lesões
    df_hist = None
    if os.path.exists('dados_base_agregados.csv'):
        try:
            df_hist = pd.read_csv('dados_base_agregados.csv', encoding='utf-8')
        except Exception:
            df_hist = pd.read_csv(
                'dados_base_agregados.csv', encoding='latin-1')

    # Dataset de Logística Geográfica das Seleções
    df_cal = pd.DataFrame(columns=[
                          'nacionalidade', 'dias_descanso_media', 'jogos_fase_grupos', 'altitude_media'])
    if os.path.exists('calendario_selecoes.csv'):
        try:
            df_cal = pd.read_csv('calendario_selecoes.csv',
                                 sep=';', encoding='utf-8')
        except Exception:
            try:
                df_cal = pd.read_csv(
                    'calendario_selecoes.csv', sep=';', encoding='latin-1')
            except Exception:
                try:
                    df_cal = pd.read_csv(
                        'calendario_selecoes.csv', sep=',', encoding='utf-8')
                except Exception as e:
                    st.error(
                        f"Erro crítico ao processar o arquivo de calendário: {e}")

    return df_hist, df_cal


df_historico, df_calendario = carregar_dados_apoio()

# ==============================================================================
# 4. CRIAÇÃO DAS LISTAS DE SELEÇÃO PURAS (SEM COMBINAÇÕES OU NULOS)
# ==============================================================================
lista_paises_limpos = []
lista_posicoes = ["Central Midfield", "Attacking Midfield", "Centre-Back", "Centre-Forward",
                  "Left Winger", "Right Winger", "Left-Back", "Right-Back", "Defensive Midfield"]

if df_historico is not None:
    paises_brutos = df_historico['nacionalidade'].dropna().astype(str)
    paises_separados = paises_brutos.str.split(',').explode().str.strip()
    lista_paises_limpos = sorted(paises_separados.unique())
else:
    lista_paises_limpos = ["Alemanha", "Argentina",
                           "Brasil", "Colômbia", "França", "Portugal"]

if not df_calendario.empty:
    df_calendario['nacionalidade'] = df_calendario['nacionalidade'].astype(
        str).str.strip()

# ==============================================================================
# 5. ESTRUTURA DE ABAS PRINCIPAIS
# ==============================================================================
abas = st.tabs(
    ["🔮 Previsão Individual", "📋 Convocados & Logística", "📊 Métricas do Modelo"])

# ==============================================================================
# ABA 1: PREVISÃO INDIVIDUAL
# ==============================================================================
with abas[0]:
    st.header("🔮 Simulação de Risco por Jogador")
    st.markdown(
        "Selecione a Posição e País. As métricas clínicas complexas serão inferidas a partir do padrão histórico.")

    col1, col2 = st.columns(2)

    with col1:
        nome_jogador = st.text_input("Nome do Atleta", "Jogador Exemplo")
        posicao_escolhida = st.selectbox("Posição em Campo", lista_posicoes)
        pais_jogador = st.selectbox("Nacionalidade Única", lista_paises_limpos)

    # --------------------------------------------------------------------------
    # PROCESSAMENTO BACKGROUND: CAPTURA INTELIGENTE DE MÉTRICAS HISTÓRICAS
    # --------------------------------------------------------------------------
    if df_historico is not None:
        filtro_historico = df_historico[
            (df_historico['posicao'] == posicao_escolhida) &
            (df_historico['nacionalidade'].dropna().astype(
                str).str.contains(pais_jogador, regex=False))
        ]

        if filtro_historico.empty:
            filtro_historico = df_historico[df_historico['posicao']
                                            == posicao_escolhida]

        risco_biomecanico_auto = float(filtro_historico['risco_biomecanico'].mean(
        ) if not filtro_historico.empty else 3.0)
        carga_historica_auto = float(filtro_historico['carga_lesao_historica'].mean(
        ) if not filtro_historico.empty else 40.0)
        num_lesoes_auto = int(np.round(
            filtro_historico['num_lesoes'].mean())) if not filtro_historico.empty else 1
    else:
        risco_biomecanico_auto, carga_historica_auto, num_lesoes_auto = 3.0, 40.0, 1

    idade_auto = 26.5

    dados_logistica = df_calendario[df_calendario['nacionalidade'].str.lower(
    ) == pais_jogador.lower()]
    if not dados_logistica.empty:
        dias_descanso = float(dados_logistica['dias_descanso_media'].values[0])
        altitude = float(dados_logistica['altitude_media'].values[0])
        jogos_fase = float(dados_logistica['jogos_fase_grupos'].values[0])
    else:
        dias_descanso, altitude, jogos_fase = 5.0, 12.0, 3.0

    idade_x_jogos = idade_auto * jogos_fase
    idade_x_altitude = idade_auto * altitude
    descanso_relativo = dias_descanso / (idade_auto + 1)

    with col2:
        st.subheader("📊 Parâmetros Vinculados (Background)")
        st.info(
            f"Dados históricos calculados para **{posicao_escolhida}** associados a **{pais_jogador}**:")

        m1, m2, m3 = st.columns(3)
        m1.metric(label="⚠️ Risco Biomecânico",
                  value=f"{risco_biomecanico_auto:.2f} / 5")
        m2.metric(label="📈 Carga Histórica",
                  value=f"{carga_historica_auto:.1f} dias")
        m3.metric(label="🔄 Média de Lesões", value=f"{num_lesoes_auto}")

    st.write("---")

    # --------------------------------------------------------------------------
    # EXECUÇÃO DA PREDIÇÃO
    # --------------------------------------------------------------------------
    if st.button("Calcular Risco de Lesão Grave 🎯", use_container_width=True):
        if pipeline is not None and pipeline['modelo'] is not None:

            colunas_treino = pipeline['colunas']

            # ── GUARDA DEFENSIVO ──────────────────────────────────────────────
            if not colunas_treino:
                st.error(
                    "❌ 'colunas_treino.pkl' está vazio ou não foi carregado corretamente. "
                    "Verifica se o ficheiro existe e foi gerado pelo pipeline de treino."
                )
                st.stop()

            # ── CONSTRUÇÃO DO INPUT ───────────────────────────────────────────
            dados_entrada = {
                'carga_lesao_historica': [float(carga_historica_auto)],
                'num_lesoes':            [int(num_lesoes_auto)],
                'dias_descanso_media':   [float(dias_descanso)],
                'jogos_fase_grupos':     [float(jogos_fase)],
                'altitude_media':        [float(altitude)],
                'idade_x_jogos':         [float(idade_x_jogos)],
                'idade_x_altitude':      [float(idade_x_altitude)],
                'descanso_relativo':     [float(descanso_relativo)],
                'risco_biomecanico':     [float(risco_biomecanico_auto)],
            }

            # One-hot encoding das posições (igual ao treino)
            for pos in lista_posicoes:
                dados_entrada[f"posicao_{pos}"] = [
                    1 if pos == posicao_escolhida else 0]

            df_input = pd.DataFrame(dados_entrada)

            # Adicionar colunas em falta (presentes no treino mas não no input)
            for col in colunas_treino:
                if col not in df_input.columns:
                    df_input[col] = 0

            # Reordenar EXATAMENTE como no treino
            df_input = df_input[colunas_treino]

            # ── FORÇAR DTYPES NUMÉRICOS ───────────────────────────────────────
            # Resolve o ValueError: np.result_type(*dtypes_orig) com lista vazia
            df_input = df_input.astype(float)

            # ── SCALER (só se existir e tiver colunas válidas) ────────────────
            scaler = pipeline['scaler']
            if scaler is not None and hasattr(scaler, "feature_names_in_"):
                # Interseção entre colunas do scaler e colunas do input
                colunas_escalonar = [
                    c for c in scaler.feature_names_in_ if c in df_input.columns
                ]
                if colunas_escalonar:
                    df_input[colunas_escalonar] = scaler.transform(
                        df_input[colunas_escalonar]
                    )

            # ── DEBUG (expander colapsado — remove em produção) ───────────────
            with st.expander("🔍 Debug — Input ao Modelo", expanded=False):
                st.write(f"**Shape:** {df_input.shape}")
                st.write(
                    f"**Dtypes únicos:** {list(df_input.dtypes.unique())}")
                st.dataframe(df_input)

            # ── PREDIÇÃO ──────────────────────────────────────────────────────
            modelo = pipeline['modelo']
            thresh = pipeline['threshold']

            try:
                probabilidade = modelo.predict_proba(df_input)[0][1]
            except Exception as e:
                st.error(f"❌ Erro na predição: {e}")
                st.stop()

            resultado_final = 1 if probabilidade >= thresh else 0

            st.write("---")

            if resultado_final == 1:
                st.error(
                    f"🚨 **ALTO RISCO CLÍNICO DETETADO!** Probabilidade de "
                    f"**{probabilidade * 100:.1f}%** (Threshold Crítico: {thresh * 100:.0f}%)"
                )
                st.markdown(
                    f"**Recomendação Técnica:** O atleta `{nome_jogador}` deve ser monitorizado "
                    f"ou poupado devido ao desgaste acumulado esperado para a posição "
                    f"na altitude de {altitude:.0f}m."
                )
            else:
                st.success(
                    f"✅ **Atleta Seguro / Estável.** Probabilidade de lesão grave de apenas "
                    f"**{probabilidade * 100:.1f}%**."
                )

        else:
            st.warning(
                "⚠️ O Pipeline ou ficheiro de modelo binário (.pkl) não foi encontrado. "
                "Certifica-te que os ficheiros 'modelos_e_previsoes.pkl' e 'tabela_metricas.csv' "
                "estão na raiz do projeto."
            )

# ==============================================================================
# ABA 2: CONVOCADOS & LOGÍSTICA
# ==============================================================================
with abas[1]:
    st.header("📋 Consulta de Jogadores Convocados por País")

    if os.path.exists('jogadores_copa_2026.csv'):
        try:
            df_convocados = pd.read_csv(
                'jogadores_copa_2026.csv', encoding='utf-8')
        except Exception:
            df_convocados = pd.read_csv(
                'jogadores_copa_2026.csv', encoding='latin-1')

        pais_escolhido = st.selectbox(
            "Selecione o País para Filtrar a Convocatória", lista_paises_limpos, key="aba2_pais")

        jogadores_filtrados = df_convocados[df_convocados['Nacionalidade'].astype(
            str).str.lower() == pais_escolhido.lower()]
        st.subheader(f"Lista de Convocados - {pais_escolhido}")

        if not df_calendario.empty:
            dados_pais = df_calendario[df_calendario['nacionalidade'].str.lower(
            ) == pais_escolhido.lower()]
            if not dados_pais.empty:
                st.write(
                    f"ℹ️ **Contexto Logístico:** Média de descanso de "
                    f"{dados_pais['dias_descanso_media'].values[0]} dias a uma altitude média de "
                    f"{dados_pais['altitude_media'].values[0]}m."
                )

        if jogadores_filtrados.empty:
            st.info(f"Nenhum jogador encontrado para **{pais_escolhido}**.")
        else:
            st.dataframe(jogadores_filtrados, use_container_width=True)
    else:
        st.warning(
            "O ficheiro 'jogadores_copa_2026.csv' não foi detetado na raiz do projeto.")

# ==============================================================================
# ABA 3: MÉTRICAS DE DESEMPENHO DO MODELO
# ==============================================================================
with abas[2]:
    st.header("📊 Performance e Validação Clínica do Modelo")

    if os.path.exists('tabela_metricas.csv'):
        try:
            df_m = pd.read_csv('tabela_metricas.csv', encoding='utf-8')
        except Exception:
            df_m = pd.read_csv('tabela_metricas.csv', encoding='latin-1')
        st.dataframe(df_m, use_container_width=True)
    else:
        st.info("A tabela comparativa de métricas não foi localizada na raiz.")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        if os.path.exists('05_curva_roc.png'):
            st.image('05_curva_roc.png', caption='Curva ROC Comparativa')
        else:
            st.info("Gráfico ROC não encontrado.")
    with col_g2:
        if os.path.exists('06_precision_recall.png'):
            st.image('06_precision_recall.png',
                     caption='Curva Precision-Recall')
        else:
            st.info("Gráfico Precision-Recall não encontrado.")
