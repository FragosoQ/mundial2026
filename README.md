# ⚽ PrevLesão 2026 — Previsão de Risco de Lesão no Mundial

> Modelo preditivo de risco de lesão para jogadores do Mundial 2026, construído com dados reais de 12.948 registos de lesões no futebol profissional.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://LINK_DA_TUA_APP.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.11-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## 🎯 O que faz

Seleciona uma **seleção nacional**, uma **posição em campo** e uma **idade** — e obtém:

- 📊 **Probabilidade de lesão grave** (>21 dias de paragem) baseada em dados históricos reais
- ⚙️ **Risco biomecânico** da posição (escala 1–6, calculada dos dados)
- 🗺️ **Impacto contextual** de altitude das sedes, dias de descanso e número de jogos
- 🩹 **Lesões mais frequentes** para aquele perfil de jogador
- 💡 **Insight narrativo** explicando os factores dominantes

---

## 📸 Demo

![Demo da App](demo.gif)

---

## 🗂️ Estrutura do Projeto

```
├── app.py                              # App Streamlit principal
├── requirements.txt                    # Dependências Python
│
├── 01_analise_exploratoria.py          # EDA do dataset
├── 02_pre_processamento.py             # Limpeza e feature engineering
├── 03_treino_modelos.py                # Treino (Random Forest, KNN, Naïve Bayes)
├── 04_avaliacao_metricas.py            # Métricas e threshold otimizado
├── 05_matriz_confusao.py               # Matrizes de confusão
├── 06_curvas_graficas.py               # Curvas ROC e Precision-Recall
├── 07_gerar_relatorio.py               # Relatório final
├── lab_orquestrador.py                 # Orquestrador do pipeline completo
│
├── lesoes_jogadores_mundial_REAL.csv   # Dataset principal (12.948 registos)
├── dados_base_agregados.csv            # Features agregadas por jogador/época
├── jogadores_copa_2026.csv             # 942 convocados de 30 seleções
├── calendario_selecoes.csv             # Contexto logístico das seleções
├── jogos_selecoes.csv                  # Desgaste por seleção
└── sedes_mundial.csv                   # Altitudes e estádios
```

---

## 📊 Dados

| Dataset | Registos | Descrição |
|---|---|---|
| `lesoes_jogadores_mundial_REAL.csv` | 12.948 | Lesões reais no futebol profissional |
| `dados_base_agregados.csv` | 7.245 | Features por jogador/época |
| `jogadores_copa_2026.csv` | 942 | Convocados das 30 seleções |
| `calendario_selecoes.csv` | 11 | Dados logísticos das sedes |

**Variáveis principais:**
`posicao` · `nacionalidade` · `lesao` · `dias_parado` · `jogos_perdidos` · `idade_atual`

---

## 🧠 Metodologia

### Probabilidade base
Calculada directamente a partir da taxa de lesões graves (>21 dias) nos registos históricos filtrados por **seleção + posição + faixa etária (±5 anos)**.

### Ajustes contextuais

| Factor | Condição | Impacto |
|---|---|---|
| 🏔️ Altitude | > 2.000m | +8% |
| 🏔️ Altitude | 1.000–2.000m | +5% |
| 🏔️ Altitude | 500–1.000m | +2% |
| 😴 Descanso | < 4 dias | +6% |
| 😴 Descanso | < 5 dias | +2% |
| 🎂 Idade | 28–32 anos | +4% |
| 🎂 Idade | > 32 anos | +7% |
| 🎂 Idade | < 22 anos | −3% |

### Risco Biomecânico por Posição

| Posição | Risco (1–6) |
|---|---|
| Goalkeeper | 6.0 |
| Centre-Forward | 5.0 |
| Right-Back / Left-Back | 4.5–4.9 |
| Central / Defensive Midfield | 4.0–4.2 |
| Wingers | 3.7–4.2 |
| Attacking Midfield | 3.0 |
| Second Striker | 1.0 |

> Calculado a partir da média de dias parado por posição, normalizado na escala 1–6.

---

## 🚀 Correr localmente

```bash
# 1. Clonar o repositório
git clone https://github.com/SEU_USERNAME/mundial2026-prevlesao.git
cd mundial2026-prevlesao

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Correr a app
streamlit run app.py
```

A app abre automaticamente em `http://localhost:8501`

---

## ☁️ Deploy (Streamlit Cloud)

1. Faz fork deste repositório
2. Vai a [share.streamlit.io](https://share.streamlit.io)
3. Liga a tua conta GitHub
4. Seleciona este repositório → `app.py` → **Deploy**

---

## 🔧 Pipeline de ML (opcional)

Se quiseres treinar os modelos localmente e gerar os ficheiros `.pkl`:

```bash
python lab_orquestrador.py
```

Isto corre o pipeline completo e gera `modelos_e_previsoes.pkl`, `scaler.pkl` e `colunas_treino.pkl`.
Sem estes ficheiros, a app funciona em **modo estatístico** (análise directa dos dados históricos).

---

## ⚠️ Limitações conhecidas

- O calendário logístico (`calendario_selecoes.csv`) cobre apenas 11 das 30 seleções — as restantes usam valores por defeito
- Alguns países têm poucos registos históricos; nesses casos a janela etária alarga automaticamente
- O modelo foi treinado com dados até 2024 — não inclui lesões da época 2024/25

---

## 📄 Licença

MIT License — podes usar, modificar e distribuir livremente com atribuição.

---

*Construído com Python · Pandas · Scikit-learn · Streamlit*
