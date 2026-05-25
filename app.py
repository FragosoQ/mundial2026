import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

# ==============================================================================
# CONFIGURAÇÃO DA PÁGINA
# ==============================================================================
st.set_page_config(
    page_title="PrevLesão 2026 — Mundial",
    page_icon="⚽",
    layout="wide"
)

st.markdown("""
<style>
    [data-testid="stAppViewContainer"] { background: #0f1117; }
    [data-testid="stSidebar"] { background: #161b27; }

    .bloco-risco-alto {
        background: linear-gradient(135deg, #3d0000 0%, #1a0000 100%);
        border: 1px solid #ff4444;
        border-left: 5px solid #ff4444;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .bloco-risco-moderado {
        background: linear-gradient(135deg, #3d2600 0%, #1a1000 100%);
        border: 1px solid #ffaa00;
        border-left: 5px solid #ffaa00;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .bloco-risco-baixo {
        background: linear-gradient(135deg, #003d1a 0%, #001a0a 100%);
        border: 1px solid #44ff88;
        border-left: 5px solid #44ff88;
        border-radius: 12px;
        padding: 2rem;
        text-align: center;
        margin: 1rem 0;
    }
    .insight-box {
        background: #1e2535;
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin: 0.4rem 0;
        border-left: 3px solid #4f9cf9;
    }
    .tag-lesao {
        display: inline-block;
        background: #2d3748;
        border-radius: 6px;
        padding: 3px 10px;
        margin: 2px;
        font-size: 0.85rem;
        color: #cbd5e0;
    }
    h1 { color: #f0f4ff !important; }
    .stTabs [data-baseweb="tab"] { font-weight: 600; font-size: 0.95rem; }
</style>
""", unsafe_allow_html=True)

# ==============================================================================
# MAPEAMENTOS
# ==============================================================================

# Nome PT (convocados) → Nome EN (lesões)
MAPA_PAISES = {
    "Alemanha":        "Germany",
    "Argentina":       "Argentina",
    "Brasil":          "Brazil",
    "Bélgica":         "Belgium",
    "Cabo Verde":      "Cape Verde",
    "Colômbia":        "Colombia",
    "Coreia do Sul":   "Korea",
    "Costa do Marfim": "Cote d'Ivoire",
    "Croácia":         "Croatia",
    "Curaçao":         "Curacao",
    "Egito":           "Egypt",
    "Escócia":         "Scotland",
    "Espanha":         "Spain",
    "França":          "France",
    "Haiti":           "Haiti",
    "Inglaterra":      "England",
    "Iraque":          "Iraq",
    "Irã":             "Iran",
    "Japão":           "Japan",
    "México":          "Mexico",
    "Noruega":         "Norway",
    "Nova Zelândia":   "New Zealand",
    "Paraguai":        "Paraguay",
    "Portugal":        "Portugal",
    "RD Congo":        "DR Congo",
    "República Tcheca":"Czech Republic",
    "Senegal":         "Senegal",
    "Suécia":          "Sweden",
    "Tunísia":         "Tunisia",
    "Áustria":         "Austria",
}

# Risco biomecânico calculado dos dados reais (escala 1–6, baseado em dias_parado médio)
RISCO_BIO_POR_POSICAO = {
    "Goalkeeper":          6.0,
    "Centre-Forward":      5.0,
    "Right-Back":          4.9,
    "Left-Back":           4.5,
    "Left Midfield":       4.4,
    "Central Midfield":    4.2,
    "Right Winger":        4.2,
    "Centre-Back":         4.0,
    "Defensive Midfield":  4.0,
    "Right Midfield":      3.8,
    "Left Winger":         3.7,
    "Attacking Midfield":  3.0,
    "Second Striker":      1.0,
}

# Contexto logístico por país (de calendario_selecoes.csv)
LOGISTICA = {
    "Portugal":       {"descanso": 5.0, "altitude": 12.0,   "jogos": 3},
    "RD Congo":       {"descanso": 5.0, "altitude": 12.0,   "jogos": 3},
    "Colômbia":       {"descanso": 5.0, "altitude": 1.0,    "jogos": 3},
    "México":         {"descanso": 5.5, "altitude": 2240.0, "jogos": 3},
    "Brasil":         {"descanso": 5.2, "altitude": 12.0,   "jogos": 3},
    "Alemanha":       {"descanso": 5.0, "altitude": 12.0,   "jogos": 3},
    "França":         {"descanso": 5.0, "altitude": 65.0,   "jogos": 3},
    "Argentina":      {"descanso": 5.0, "altitude": 265.0,  "jogos": 3},
    "Inglaterra":     {"descanso": 5.0, "altitude": 130.0,  "jogos": 3},
}
DEFAULT_LOG = {"descanso": 5.0, "altitude": 12.0, "jogos": 3}

POSICOES_EN = [
    "Attacking Midfield", "Central Midfield", "Centre-Back", "Centre-Forward",
    "Defensive Midfield", "Goalkeeper", "Left Midfield", "Left Winger",
    "Left-Back", "Right Midfield", "Right Winger", "Right-Back", "Second Striker",
]

# ==============================================================================
# CARREGAMENTO DE DADOS (CACHE)
# ==============================================================================

@st.cache_data
def carregar_lesoes():
    for enc in ["utf-8", "latin-1"]:
        try:
            df = pd.read_csv("lesoes_jogadores_mundial_REAL.csv",
                             encoding=enc, on_bad_lines="skip")
            df["lesao_grave"] = (df["dias_parado"] > 21).astype(int)
            df["lesao_moderada"] = (df["dias_parado"].between(8, 21)).astype(int)
            return df
        except Exception:
            pass
    return None

@st.cache_data
def carregar_convocados():
    for enc in ["utf-8", "latin-1"]:
        try:
            return pd.read_csv("jogadores_copa_2026.csv", encoding=enc)
        except Exception:
            pass
    return None

@st.cache_resource
def carregar_pipeline():
    resultado = {"modelo": None, "nome": "—", "threshold": 0.35,
                 "scaler": None, "colunas": []}
    for enc in ["utf-8", "latin-1"]:
        try:
            df_m = pd.read_csv("tabela_metricas.csv", encoding=enc)
            resultado["nome"] = df_m.iloc[0]["Modelo"]
            resultado["threshold"] = float(df_m.iloc[0]["Threshold_Otimo"])
            break
        except Exception:
            pass
    if os.path.exists("modelos_e_previsoes.pkl"):
        try:
            with open("modelos_e_previsoes.pkl", "rb") as f:
                pkls = pickle.load(f)
            resultado["modelo"] = pkls[resultado["nome"]]["modelo"]
        except Exception:
            pass
    if os.path.exists("scaler.pkl"):
        try:
            with open("scaler.pkl", "rb") as f:
                resultado["scaler"] = pickle.load(f)
        except Exception:
            pass
    if os.path.exists("colunas_treino.pkl"):
        try:
            with open("colunas_treino.pkl", "rb") as f:
                resultado["colunas"] = pickle.load(f)
        except Exception:
            pass
    return resultado

df_lesoes    = carregar_lesoes()
df_convocados = carregar_convocados()
pipeline     = carregar_pipeline()

# ==============================================================================
# FUNÇÕES DE ANÁLISE
# ==============================================================================

def obter_subset(pais_pt: str, posicao: str, idade: int, janela: int = 5) -> tuple:
    """
    Filtra o dataset de lesões reais por país (via contains), posição e faixa etária.
    Aumenta a janela progressivamente se não houver dados suficientes.
    Devolve (DataFrame, origem) onde origem é 'pais' ou 'global'.
    """
    if df_lesoes is None:
        return pd.DataFrame(), "global"

    pais_en = MAPA_PAISES.get(pais_pt, pais_pt)
    mask_pos  = df_lesoes["posicao"] == posicao
    mask_pais = df_lesoes["nacionalidade"].str.contains(pais_en, regex=False, na=False)

    for j in [janela, janela + 3, janela + 6, 99]:
        mask_idade = df_lesoes["idade_atual"].between(idade - j, idade + j)
        subset = df_lesoes[mask_pais & mask_pos & mask_idade]
        if len(subset) >= 5:
            return subset, "pais"
        if j == 99:
            break

    # Sem dados de país: só posição + idade (média global da posição)
    for j in [janela, janela + 3, 99]:
        mask_idade = df_lesoes["idade_atual"].between(idade - j, idade + j)
        subset = df_lesoes[mask_pos & mask_idade]
        if len(subset) >= 10:
            return subset, "global"

    return df_lesoes[mask_pos], "global"


def calcular_risco(pais_pt: str, posicao: str, idade: int) -> dict:
    """
    Calcula probabilidade de lesão grave e métricas a partir dos dados reais.
    Incorpora impacto de altitude, dias de descanso e número de jogos.
    """
    subset, origem_dados = obter_subset(pais_pt, posicao, idade)
    log = LOGISTICA.get(pais_pt, DEFAULT_LOG)

    n_registos = len(subset)
    if n_registos == 0:
        prob_base = 0.35
        dias_medio = 25.0
        jogos_perdidos_med = 4.0
        top_lesoes = []
        pct_grave = 0.35
    else:
        pct_grave       = subset["lesao_grave"].mean()
        pct_moderada    = subset["lesao_moderada"].mean()
        dias_medio      = subset["dias_parado"].mean()
        jogos_perdidos_med = subset["jogos_perdidos"].mean() if subset["jogos_perdidos"].notna().any() else 4.0
        top_lesoes      = subset["lesao"].value_counts().head(5).index.tolist()
        prob_base       = pct_grave

    # ── Ajustes contextuais ──────────────────────────────────────────────────
    # Altitude: acima de 500m aumenta risco muscular
    fator_altitude = 0.0
    if log["altitude"] > 2000:
        fator_altitude = 0.08
    elif log["altitude"] > 1000:
        fator_altitude = 0.05
    elif log["altitude"] > 500:
        fator_altitude = 0.02

    # Descanso: menos de 4 dias entre jogos aumenta risco
    fator_descanso = 0.0
    if log["descanso"] < 4.0:
        fator_descanso = 0.06
    elif log["descanso"] < 5.0:
        fator_descanso = 0.02

    # Carga de jogos
    fator_jogos = (log["jogos"] - 3) * 0.02  # base = 3 jogos

    # Idade: pico de risco a partir dos 30 (fisiologicamente correcto)
    # <22 ainda em maturação | 22-29 pico físico | 30-33 recuperação mais lenta | >33 vulnerabilidade tendinosa
    if idade < 22:
        fator_idade = -0.03
    elif idade <= 29:
        fator_idade = 0.0
    elif idade <= 33:
        fator_idade = 0.04
    else:
        fator_idade = 0.07

    prob_final = prob_base + fator_altitude + fator_descanso + fator_jogos + fator_idade
    prob_final = float(min(max(prob_final, 0.05), 0.95))

    risco_bio = RISCO_BIO_POR_POSICAO.get(posicao, 3.0)

    return {
        "probabilidade":     prob_final,
        "prob_base":         float(prob_base),
        "n_registos":        n_registos,
        "dias_medio":        float(round(dias_medio, 1)),
        "jogos_perdidos":    float(round(jogos_perdidos_med, 1)),
        "top_lesoes":        top_lesoes,
        "risco_bio":         risco_bio,
        "fator_altitude":    round(fator_altitude * 100, 1),
        "fator_descanso":    round(fator_descanso * 100, 1),
        "fator_jogos":       round(fator_jogos * 100, 1),
        "fator_idade":       round(fator_idade * 100, 1),
        "altitude":          log["altitude"],
        "descanso":          log["descanso"],
        "jogos":             log["jogos"],
        "pais_en":           MAPA_PAISES.get(pais_pt, pais_pt),
        "origem_dados":      origem_dados,
    }


def prever_com_modelo(pais_pt: str, posicao: str, idade: float, resultado_heur: dict):
    """
    Tenta usar o modelo .pkl treinado. Devolve None se não estiver disponível.
    """
    if pipeline["modelo"] is None or not pipeline["colunas"]:
        return None

    log = LOGISTICA.get(pais_pt, DEFAULT_LOG)
    risco_bio = resultado_heur["risco_bio"]
    carga     = resultado_heur["dias_medio"]
    n_lesoes  = max(1, round(resultado_heur["jogos_perdidos"] / 5))

    idade_x_jogos    = idade * log["jogos"]
    idade_x_altitude = idade * log["altitude"]
    descanso_rel     = log["descanso"] / (idade + 1)

    dados = {
        "carga_lesao_historica": [carga],
        "num_lesoes":            [n_lesoes],
        "dias_descanso_media":   [log["descanso"]],
        "jogos_fase_grupos":     [float(log["jogos"])],
        "altitude_media":        [log["altitude"]],
        "idade_x_jogos":         [idade_x_jogos],
        "idade_x_altitude":      [idade_x_altitude],
        "descanso_relativo":     [descanso_rel],
        "risco_biomecanico":     [risco_bio],
    }
    for pos in POSICOES_EN:
        dados[f"posicao_{pos}"] = [1 if pos == posicao else 0]

    df_in = pd.DataFrame(dados)
    for col in pipeline["colunas"]:
        if col not in df_in.columns:
            df_in[col] = 0
    df_in = df_in[pipeline["colunas"]].astype(float)

    sc = pipeline["scaler"]
    if sc is not None and hasattr(sc, "feature_names_in_"):
        cols_sc = [c for c in sc.feature_names_in_ if c in df_in.columns]
        if cols_sc:
            df_in[cols_sc] = sc.transform(df_in[cols_sc])

    try:
        prob = pipeline["modelo"].predict_proba(df_in)[0][1]
        return float(prob)
    except Exception:
        return None

# ==============================================================================
# TÍTULO
# ==============================================================================
st.title("⚽ PrevLesão 2026 — Risco de Lesão no Mundial")
st.caption(
    "Análise baseada em **12.948 registos reais** de lesões no futebol profissional. "
    "Seleciona a seleção, posição e idade para obter a probabilidade e os factores de risco."
)

# ==============================================================================
# ABAS
# ==============================================================================
aba_prev, aba_conv, aba_metricas = st.tabs(
    ["🔮 Previsão de Risco", "📋 Convocados", "📊 Modelo & Métricas"]
)

# ==============================================================================
# ABA 1 — PREVISÃO
# ==============================================================================
with aba_prev:

    # ── Inputs ────────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([1.2, 1.2, 0.8], gap="medium")

    with col_a:
        pais_escolhido = st.selectbox(
            "🌍 Seleção Nacional",
            sorted(MAPA_PAISES.keys()),
            help="País convocado para o Mundial 2026"
        )

    with col_b:
        posicao_escolhida = st.selectbox(
            "🧤 Posição em Campo",
            POSICOES_EN,
            help="Posição tática do jogador"
        )

    with col_c:
        idade_escolhida = st.number_input(
            "🎂 Idade",
            min_value=17, max_value=42, value=25, step=1,
            help="Idade actual do jogador"
        )

    st.divider()

    # ── Cálculo base (dados históricos) ──────────────────────────────────────
    res = calcular_risco(pais_escolhido, posicao_escolhida, int(idade_escolhida))

    # ── Slider de altitude (simulação manual) ────────────────────────────────
    altitude_default = int(LOGISTICA.get(pais_escolhido, DEFAULT_LOG)["altitude"])
    st.markdown("**🏔️ Simular impacto da altitude**")
    st.caption(
        "A altitude abaixo corresponde à sede real do país seleccionado. "
        "Ajusta para explorar cenários hipotéticos e ver o impacto directo na probabilidade."
    )
    col_sl, col_ref = st.columns([3, 1], gap="medium")
    with col_sl:
        altitude_manual = st.slider(
            "Altitude (metros)",
            min_value=0, max_value=3500, step=50,
            value=altitude_default,
            format="%d m",
            label_visibility="collapsed"
        )
    with col_ref:
        referencia = ""
        if altitude_manual == 0:
            referencia = "🌊 Nível do mar"
        elif altitude_manual <= 200:
            referencia = "🏙️ Lisboa / Madrid"
        elif altitude_manual <= 600:
            referencia = "🏙️ Cidade do Cabo"
        elif altitude_manual <= 1200:
            referencia = "🏙️ Nairobi"
        elif altitude_manual <= 1800:
            referencia = "🏙️ Joanesburgo"
        elif altitude_manual <= 2300:
            referencia = "🏙️ Cidade do México"
        elif altitude_manual <= 2800:
            referencia = "🏔️ Bogotá"
        else:
            referencia = "🏔️ La Paz"
        st.markdown(f"<div style='padding-top:8px; color:#aaa; font-size:0.85rem'>{referencia}</div>",
                    unsafe_allow_html=True)

    # Recalcular fator de altitude com o valor do slider
    if altitude_manual > 2000:
        fator_alt_manual = 0.08
    elif altitude_manual > 1000:
        fator_alt_manual = 0.05
    elif altitude_manual > 500:
        fator_alt_manual = 0.02
    else:
        fator_alt_manual = 0.0

    # Substituir fator de altitude no resultado
    delta_altitude = fator_alt_manual - (res["fator_altitude"] / 100)
    prob_ajustada  = float(min(max(res["probabilidade"] + delta_altitude, 0.05), 0.95))
    res = dict(res)
    res["altitude"]       = altitude_manual
    res["fator_altitude"] = round(fator_alt_manual * 100, 1)
    res["probabilidade"]  = prob_ajustada

    st.divider()

    # Tentar modelo treinado; fallback nos dados reais
    prob_modelo = prever_com_modelo(pais_escolhido, posicao_escolhida, idade_escolhida, res)
    if prob_modelo is not None:
        prob_final = prob_modelo
        fonte_prob = f"Modelo ML ({pipeline['nome']})"
    else:
        prob_final = res["probabilidade"]
        fonte_prob = "Dados históricos reais"

    # Classificar risco
    if prob_final >= 0.50:
        nivel_risco, cor_risco, classe_bloco = "ALTO", "#ff4444", "bloco-risco-alto"
        emoji_risco = "🚨"
    elif prob_final >= 0.33:
        nivel_risco, cor_risco, classe_bloco = "MODERADO", "#ffaa00", "bloco-risco-moderado"
        emoji_risco = "⚠️"
    else:
        nivel_risco, cor_risco, classe_bloco = "BAIXO", "#44ff88", "bloco-risco-baixo"
        emoji_risco = "✅"

    # ── Layout de resultado ───────────────────────────────────────────────────
    col_res, col_ctx = st.columns([1, 1.4], gap="large")

    with col_res:
        st.markdown(f"""
        <div class="{classe_bloco}">
            <div style="font-size:0.9rem; color:#aaa; margin-bottom:0.3rem">
                {emoji_risco} RISCO DE LESÃO GRAVE
            </div>
            <div style="font-size:3.5rem; font-weight:900; color:{cor_risco}; line-height:1.1">
                {prob_final*100:.1f}%
            </div>
            <div style="font-size:1.1rem; color:{cor_risco}; font-weight:600; margin-top:0.3rem">
                {nivel_risco}
            </div>
            <div style="font-size:0.78rem; color:#888; margin-top:0.8rem">
                {fonte_prob} · {res['n_registos']} registos {"de " + pais_escolhido if res['origem_dados'] == 'pais' else "(média global da posição)"}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Barra de risco visual
        barra_pct = int(prob_final * 100)
        cor_barra = cor_risco
        st.markdown(f"""
        <div style="background:#2d3748; border-radius:8px; height:14px; margin:0.5rem 0; overflow:hidden">
            <div style="background:{cor_barra}; width:{barra_pct}%; height:100%;
                        border-radius:8px; transition:width 0.4s ease"></div>
        </div>
        <div style="text-align:right; font-size:0.8rem; color:#888">{barra_pct}% probabilidade</div>
        """, unsafe_allow_html=True)

        # Fonte dos dados e aviso de fallback
        if res["origem_dados"] == "global":
            st.markdown(f"""
            <div style="background:#2a1f00; border:1px solid #f59b0a; border-radius:8px;
                        padding:0.6rem 1rem; margin:0.5rem 0; font-size:0.85rem; color:#f59b0a">
                ⚠️ <b>Sem dados históricos para {pais_escolhido} nesta posição</b> —
                probabilidade calculada com base na média global da posição
                ({res['n_registos']} registos de todos os países).
            </div>
            """, unsafe_allow_html=True)

        # Métricas rápidas
        st.markdown("**📊 Dados históricos da selecção + posição:**")
        mc1, mc2 = st.columns(2)
        mc1.metric("⏱️ Dias parado médio", f"{res['dias_medio']} dias")
        mc2.metric("🎮 Jogos perdidos médio", f"{res['jogos_perdidos']:.1f}")

    with col_ctx:
        st.subheader("📋 Factores de Risco Detalhados")

        # Risco Biomecânico da posição
        bio = res["risco_bio"]
        bio_cor = "#ff4444" if bio >= 5 else "#ffaa00" if bio >= 3.5 else "#44ff88"
        st.markdown(f"""
        <div class="insight-box">
            <b>⚙️ Risco Biomecânico da Posição ({posicao_escolhida})</b><br>
            <span style="font-size:1.6rem; font-weight:800; color:{bio_cor}">{bio:.1f}</span>
            <span style="color:#888"> / 6.0</span>
            <div style="background:#2d3748; border-radius:4px; height:8px; margin-top:6px">
                <div style="background:{bio_cor}; width:{bio/6*100:.0f}%; height:100%; border-radius:4px"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Contexto Logístico
        st.markdown("**🗺️ Contexto Logístico — " + pais_escolhido + "**")
        cl1, cl2, cl3 = st.columns(3)
        cl1.metric("🏔️ Altitude", f"{res['altitude']:.0f} m")
        cl2.metric("😴 Descanso", f"{res['descanso']:.1f} dias")
        cl3.metric("🎮 Jogos fase", str(res["jogos"]))

        # Decomposição dos ajustes
        st.markdown("**📈 Impacto dos factores contextuais na probabilidade:**")

        fatores = [
            ("📊 Base histórica (pos + país + idade)", res["prob_base"] * 100, "#4f9cf9"),
            ("🏔️ Altitude", res["fator_altitude"], "#9b87f5"),
            ("😴 Dias de descanso", res["fator_descanso"], "#f59b0a"),
            ("🎮 Nº de jogos na fase", res["fator_jogos"], "#10b981"),
            ("🎂 Idade (" + str(int(idade_escolhida)) + " anos)", res["fator_idade"], "#f43f5e"),
        ]

        for label, valor, cor in fatores:
            sinal = "+" if valor >= 0 else ""
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center;
                        padding:6px 10px; background:#1a2030; border-radius:6px; margin:3px 0">
                <span style="font-size:0.88rem">{label}</span>
                <span style="font-weight:700; color:{cor}">{sinal}{valor:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Tipos de lesão mais frequentes nesta posição/contexto
        if res["top_lesoes"]:
            st.markdown("**🩹 Lesões mais frequentes (posição + perfil):**")
            tags = "".join(f'<span class="tag-lesao">{l}</span>' for l in res["top_lesoes"])
            st.markdown(f'<div style="margin-top:4px">{tags}</div>', unsafe_allow_html=True)

    st.divider()

    # ── Insight narrativo ─────────────────────────────────────────────────────
    st.subheader("💡 Análise do Preparador Físico")

    alt  = res["altitude"]
    desc = res["descanso"]
    bio  = res["risco_bio"]
    dias = res["dias_medio"]
    idade = int(idade_escolhida)

    # ── Dicionários de conhecimento clínico por posição ────────────────────────
    GRUPOS_MUSCULARES = {
        "Goalkeeper":          "isquiotibiais, ombros e coluna lombar",
        "Centre-Back":         "isquiotibiais, joelhos (ligamento cruzado) e adutores",
        "Left-Back":           "isquiotibiais, adutores e gémeos",
        "Right-Back":          "isquiotibiais, adutores e gémeos",
        "Defensive Midfield":  "isquiotibiais, joelhos e tornozelos",
        "Central Midfield":    "isquiotibiais, gémeos e tornozelos",
        "Left Midfield":       "isquiotibiais, adutores e gémeos",
        "Right Midfield":      "isquiotibiais, adutores e gémeos",
        "Attacking Midfield":  "isquiotibiais, quadricípites e tornozelos",
        "Left Winger":         "isquiotibiais, adutores, gémeos e tornozelos",
        "Right Winger":        "isquiotibiais, adutores, gémeos e tornozelos",
        "Centre-Forward":      "isquiotibiais, quadricípites, joelhos e adutores",
        "Second Striker":      "isquiotibiais, quadricípites e tornozelos",
    }

    MECANISMO_LESAO = {
        "Goalkeeper":          "esforços explosivos de curta duração, mergulhos e colisões — o contacto com o solo e a extensão rápida dos membros superiores são os principais vectores de lesão",
        "Centre-Back":         "duelos aéreos, travagens bruscas e acelerações curtas repetidas — o ligamento cruzado anterior é o principal alvo em situações de mudança de direcção",
        "Left-Back":           "sprints repetidos ao longo da linha lateral, combinados com cruzamentos em extensão — a sobrecarga dos adutores é o principal risco",
        "Right-Back":          "sprints repetidos ao longo da linha lateral, combinados com cruzamentos em extensão — a sobrecarga dos adutores é o principal risco",
        "Defensive Midfield":  "volume elevado de corrida (12–14km por jogo), duelos físicos constantes e acções defensivas em travagem — a fadiga acumulada é o principal factor",
        "Central Midfield":    "volume total de corrida mais alto em campo (frequentemente >13km), com variações constantes de ritmo — a sobrecarga crónica dos isquiotibiais é o risco dominante",
        "Left Midfield":       "combinação de volume de corrida elevado com arranques explosivos nas transições — perfil de fadiga muscular acumulada ao longo do torneio",
        "Right Midfield":      "combinação de volume de corrida elevado com arranques explosivos nas transições — perfil de fadiga muscular acumulada ao longo do torneio",
        "Attacking Midfield":  "acelerações explosivas em espaços reduzidos, mudanças de direcção rápidas e receção de bola sob pressão — o padrão de lesão é tipicamente muscular agudo",
        "Left Winger":         "sprints máximos repetidos (velocidades de pico acima de 30km/h), arranques e paragens abruptas — o mecanismo de lesão é quase sempre muscular por sobrecarga de alta intensidade",
        "Right Winger":        "sprints máximos repetidos (velocidades de pico acima de 30km/h), arranques e paragens abruptas — o mecanismo de lesão é quase sempre muscular por sobrecarga de alta intensidade",
        "Centre-Forward":      "acelerações explosivas, duelos físicos frontais e remates de alta potência — o pico de tensão nos isquiotibiais no momento do remate é o principal vector de ruptura",
        "Second Striker":      "movimentos de ruptura entre linhas, receção em profundidade e finalização — perfil de lesão misto entre explosividade e contacto",
    }

    PROTOCOLO_RISCO_ALTO = {
        "Goalkeeper":          "monitorização diária da mobilidade lombar e dos ombros; reduzir o volume de treino de remates antes dos jogos; protocolo de aquecimento específico para membros superiores",
        "Centre-Back":         "reforço preventivo do quadricípite e dos estabilizadores do joelho (protocolo Nordic Hamstring); evitar treinos de alta intensidade nas 48h anteriores ao jogo",
        "Left-Back":           "trabalho específico de adutores (protocolo Copenhagen) 3x/semana; gestão cuidadosa do volume de sprints nos treinos",
        "Right-Back":          "trabalho específico de adutores (protocolo Copenhagen) 3x/semana; gestão cuidadosa do volume de sprints nos treinos",
        "Defensive Midfield":  "monitorização da carga GPS (distância total e sprints de alta intensidade); recuperação activa obrigatória no dia seguinte ao jogo",
        "Central Midfield":    "protocolo Nordic Hamstring obrigatório; limitar o volume de treino de alta intensidade a <20min nas sessões entre jogos",
        "Left Midfield":       "monitorização GPS rigorosa; dias de treino reduzido quando a distância total superar 11km no jogo anterior",
        "Right Midfield":      "monitorização GPS rigorosa; dias de treino reduzido quando a distância total superar 11km no jogo anterior",
        "Attacking Midfield":  "aquecimento prolongado (mínimo 20min) com ênfase em activação de isquiotibiais e tornozelos; crioterapia pós-jogo como rotina",
        "Left Winger":         "protocolo Nordic Hamstring 2x/semana; limitar sprints máximos nos treinos após jogo; monitorização de rigidez muscular por palpação diária",
        "Right Winger":        "protocolo Nordic Hamstring 2x/semana; limitar sprints máximos nos treinos após jogo; monitorização de rigidez muscular por palpação diária",
        "Centre-Forward":      "protocolo Nordic Hamstring obrigatório; reduzir volume de remates nos treinos pré-jogo; vigilância específica ao padrão de corrida (assimetrias > 10% são sinal de alerta)",
        "Second Striker":      "trabalho de mobilidade de tornozelos e activação de quadricípites no aquecimento; atenção especial a sinais de fadiga nas 72h pós-jogo",
    }

    PROTOCOLO_RISCO_MODERADO = {
        "Goalkeeper":          "aquecimento específico para membros superiores e coluna; protocolo de fortalecimento isométrico dos ombros entre jogos",
        "Centre-Back":         "trabalho de estabilidade do joelho 2x/semana; monitorização de dor ou rigidez após duelos aéreos",
        "Left-Back":           "protocolo Copenhagen de adutores 2x/semana; gestão do volume de sprints laterais",
        "Right-Back":          "protocolo Copenhagen de adutores 2x/semana; gestão do volume de sprints laterais",
        "Defensive Midfield":  "recuperação activa no dia seguinte ao jogo (natação ou ciclismo de baixa intensidade); monitorização subjectiva de fadiga",
        "Central Midfield":    "protocolo Nordic Hamstring preventivo 2x/semana; atenção a sinais de rigidez nos isquiotibiais",
        "Left Midfield":       "monitorização GPS; incluir trabalho de mobilidade de tornozelos no aquecimento",
        "Right Midfield":      "monitorização GPS; incluir trabalho de mobilidade de tornozelos no aquecimento",
        "Attacking Midfield":  "aquecimento dinâmico focado em isquiotibiais e tornozelos; crioterapia preventiva após treinos de alta intensidade",
        "Left Winger":         "Nordic Hamstring 2x/semana; palpação diária dos isquiotibiais para detecção precoce de rigidez",
        "Right Winger":        "Nordic Hamstring 2x/semana; palpação diária dos isquiotibiais para detecção precoce de rigidez",
        "Centre-Forward":      "Nordic Hamstring preventivo; reduzir volume de remates de potência máxima nos treinos",
        "Second Striker":      "mobilidade de tornozelos e activação de quadricípites no aquecimento; monitorização de fadiga muscular",
    }

    PROTOCOLO_RISCO_BAIXO = {
        "Goalkeeper":          "manutenção do protocolo standard; atenção aos ombros em sessões de remates intensas",
        "Centre-Back":         "manutenção do protocolo standard; sem restrições específicas",
        "Left-Back":           "manutenção standard; atenção ao volume de cruzamentos nos treinos",
        "Right-Back":          "manutenção standard; atenção ao volume de cruzamentos nos treinos",
        "Defensive Midfield":  "manutenção standard; monitorização de fadiga após jogos intensos",
        "Central Midfield":    "manutenção standard; Nordic Hamstring como prevenção geral",
        "Left Midfield":       "manutenção standard",
        "Right Midfield":      "manutenção standard",
        "Attacking Midfield":  "manutenção standard; aquecimento dinâmico adequado antes de sessões técnicas",
        "Left Winger":         "manutenção standard; Nordic Hamstring preventivo",
        "Right Winger":        "manutenção standard; Nordic Hamstring preventivo",
        "Centre-Forward":      "manutenção standard; atenção ao padrão de remate em treinos de finalização",
        "Second Striker":      "manutenção standard",
    }

    grupos    = GRUPOS_MUSCULARES.get(posicao_escolhida, "isquiotibiais e articulações dos membros inferiores")
    mecanismo = MECANISMO_LESAO.get(posicao_escolhida, "esforços de alta intensidade e mudanças de direcção")

    if prob_final >= 0.50:
        protocolo = PROTOCOLO_RISCO_ALTO.get(posicao_escolhida, "monitorização intensiva e redução de carga")
        cor_alerta = "#ff4444"
        emoji_alerta = "🚨"
        nivel_texto = "ALTO RISCO — Intervenção Preventiva Prioritária"
    elif prob_final >= 0.33:
        protocolo = PROTOCOLO_RISCO_MODERADO.get(posicao_escolhida, "protocolos preventivos standard reforçados")
        cor_alerta = "#ffaa00"
        emoji_alerta = "⚠️"
        nivel_texto = "RISCO MODERADO — Monitorização Activa Recomendada"
    else:
        protocolo = PROTOCOLO_RISCO_BAIXO.get(posicao_escolhida, "manutenção do protocolo standard")
        cor_alerta = "#44ff88"
        emoji_alerta = "✅"
        nivel_texto = "RISCO CONTROLADO — Manutenção do Protocolo Standard"

    # ── Construir blocos do insight ───────────────────────────────────────────

    # Bloco 1: Avaliação biomecânica
    bio_texto = (
        f"Com um risco biomecânico de **{bio:.1f}/6**, a posição **{posicao_escolhida}** "
        f"exige esforços de **{mecanismo}**. "
        f"Os grupos musculares mais expostos são os **{grupos}**. "
        f"Historicamente, jogadores nesta posição ficam parados em média **{dias:.0f} dias** por lesão grave."
    )

    # Bloco 2: Perfil etário
    if idade < 22:
        idade_texto = (
            f"Aos **{idade} anos**, o jogador está numa fase de maturação muscular e tendinosa ainda incompleta. "
            "O risco de lesões por sobreuso e stress de crescimento é relevante, "
            "especialmente em torneios com jogos consecutivos em poucos dias."
        )
    elif idade <= 27:
        idade_texto = (
            f"Com **{idade} anos**, o jogador está no pico da capacidade de recuperação muscular. "
            "A aptidão física está no seu máximo, mas a intensidade competitiva do Mundial "
            "pode criar picos de fadiga acumulada que não devem ser subestimados."
        )
    elif idade <= 32:
        idade_texto = (
            f"Aos **{idade} anos**, o jogador entra na fase em que a recuperação começa a ser mais lenta "
            "e a fadiga acumulada tem maior impacto. "
            "A gestão da carga entre jogos é crítica — o risco de lesão muscular aumenta "
            "significativamente quando o intervalo de recuperação é inferior a 5 dias."
        )
    else:
        idade_texto = (
            f"Com **{idade} anos**, o perfil de recuperação muscular está claramente comprometido. "
            "A elasticidade tendinosa reduz-se, a regeneração muscular é mais lenta "
            "e a probabilidade de lesões recorrentes aumenta. "
            "A gestão de carga é o factor mais determinante neste perfil etário."
        )

    # Bloco 3: Contexto logístico
    ctx_partes = []
    if alt > 2000:
        ctx_partes.append(
            f"A altitude de **{alt:.0f}m** é o factor ambiental mais crítico deste perfil. "
            "Acima dos 2.000m, a pressão parcial de oxigénio reduz-se em ~20%, "
            "aumentando o stress cardiorrespiratório, a fadiga muscular e a susceptibilidade a lesões. "
            "Requer aclimatação mínima de 72h antes do primeiro jogo e hidratação intensificada."
        )
    elif alt > 1000:
        ctx_partes.append(
            f"A altitude de **{alt:.0f}m** representa um esforço adicional mensurável — "
            "o consumo de oxigénio aumenta cerca de 8–12%, o que se traduz numa fadiga muscular acumulada superior ao habitual."
        )
    elif alt > 300:
        ctx_partes.append(
            f"A altitude de **{alt:.0f}m** tem impacto fisiológico ligeiro mas não negligenciável "
            "em jogadores habitualmente treinados ao nível do mar."
        )

    if desc < 4:
        ctx_partes.append(
            f"Com **{desc:.1f} dias de descanso** médio entre jogos, o tempo de regeneração muscular "
            "é claramente insuficiente. A literatura desportiva indica que são necessários pelo menos "
            "5–6 dias para recuperação completa de fibras musculares após esforço máximo. "
            "Abaixo desse limiar, o risco de lesão muscular quase duplica."
        )
    elif desc < 5.5:
        ctx_partes.append(
            f"Os **{desc:.1f} dias de descanso** entre jogos estão no limite aceitável. "
            "Protocolo de recuperação activa (crioterapia, piscina, compressão) deve ser rigoroso."
        )

    ctx_texto = " ".join(ctx_partes) if ctx_partes else None

    # Bloco 4: Recomendação do preparador
    recomendacao_texto = (
        f"**Recomendação:** {protocolo}."
    )

    # ── Renderizar ────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div style="border:1px solid {cor_alerta}; border-radius:10px; padding:0.6rem 1rem;
                margin-bottom:1rem; background: rgba(0,0,0,0.2)">
        <span style="color:{cor_alerta}; font-weight:700; font-size:0.95rem">
            {emoji_alerta} {nivel_texto}
        </span>
    </div>
    """, unsafe_allow_html=True)

    blocos = [
        ("⚙️ Avaliação Biomecânica", bio_texto),
        ("🎂 Perfil Etário", idade_texto),
    ]
    if ctx_texto:
        blocos.append(("🗺️ Contexto Logístico", ctx_texto))
    blocos.append(("📋 Protocolo Recomendado", recomendacao_texto))

    for titulo, conteudo in blocos:
        st.markdown(f"""
        <div class="insight-box" style="margin-bottom:0.6rem">
            <div style="font-size:0.8rem; color:#888; margin-bottom:4px">{titulo}</div>
            <div style="font-size:0.93rem; line-height:1.6">{conteudo}</div>
        </div>
        """, unsafe_allow_html=True)

    if res["n_registos"] < 10:
        st.caption(
            f"⚠️ Análise baseada em {res['n_registos']} registos para esta combinação específica "
            "— interpretação com cautela."
        )

# ==============================================================================
# ABA 2 — CONVOCADOS
# ==============================================================================
with aba_conv:
    st.header("📋 Convocados por Seleção — Mundial 2026")

    if df_convocados is not None:
        pais_aba2 = st.selectbox("Seleciona a Seleção", sorted(MAPA_PAISES.keys()), key="aba2")
        log2 = LOGISTICA.get(pais_aba2, DEFAULT_LOG)

        c1, c2, c3 = st.columns(3)
        c1.metric("🏔️ Altitude média", f"{log2['altitude']:.0f} m")
        c2.metric("😴 Descanso médio", f"{log2['descanso']:.1f} dias")
        c3.metric("🎮 Jogos fase grupos", str(log2["jogos"]))

        if log2["altitude"] > 1500:
            st.warning(f"⛰️ Altitude elevada ({log2['altitude']:.0f}m) — risco adicional de lesões musculares.")

        st.divider()

        filt = df_convocados[
            df_convocados["Nacionalidade"].str.lower() == pais_aba2.lower()
        ].reset_index(drop=True)

        if filt.empty:
            st.info(f"Nenhum convocado encontrado para **{pais_aba2}**.")
        else:
            st.subheader(f"{pais_aba2} — {len(filt)} convocados")
            st.dataframe(filt, use_container_width=True)
    else:
        st.warning("`jogadores_copa_2026.csv` não encontrado.")

# ==============================================================================
# ABA 3 — MODELO & MÉTRICAS
# ==============================================================================
with aba_metricas:
    st.header("📊 Modelo & Métricas de Validação")

    if pipeline["modelo"] is not None:
        st.success(f"✅ Modelo activo: **{pipeline['nome']}** · Threshold: {pipeline['threshold']*100:.0f}%")
    else:
        st.info(
            "📁 Ficheiros `.pkl` não encontrados — a app usa análise estatística directa "
            "sobre os dados históricos reais (**12.948 registos**)."
        )

    if os.path.exists("tabela_metricas.csv"):
        for enc in ["utf-8", "latin-1"]:
            try:
                df_mt = pd.read_csv("tabela_metricas.csv", encoding=enc)
                st.subheader("Comparação de Modelos")
                st.dataframe(df_mt, use_container_width=True)
                break
            except Exception:
                pass

    cg1, cg2 = st.columns(2)
    with cg1:
        if os.path.exists("05_curva_roc.png"):
            st.image("05_curva_roc.png", caption="Curva ROC")
        else:
            st.info("`05_curva_roc.png` não encontrado.")
    with cg2:
        if os.path.exists("06_precision_recall.png"):
            st.image("06_precision_recall.png", caption="Precision-Recall")
        else:
            st.info("`06_precision_recall.png` não encontrado.")

    st.divider()
    st.subheader("🔬 Sobre a Metodologia")
    st.markdown("""
    **Probabilidade base** — calculada directamente a partir da taxa de lesões graves (>21 dias)
    nos registos históricos reais filtrados por seleção + posição + faixa etária (±5 anos).

    **Ajustes contextuais aplicados:**
    | Factor | Impacto |
    |---|---|
    | Altitude > 2000m | +8% |
    | Altitude 1000–2000m | +5% |
    | Altitude 500–1000m | +2% |
    | Descanso < 4 dias | +6% |
    | Descanso < 5 dias | +2% |
    | Idade 28–32 anos | +4% |
    | Idade > 32 anos | +7% |
    | Idade < 22 anos | −3% |

    **Risco Biomecânico** — calculado a partir da média de dias parado por posição,
    normalizado na escala 1–6 com base nos 12.948 registos do dataset.
    """)
