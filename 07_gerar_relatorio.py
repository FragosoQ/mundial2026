# 07_gerar_relatorio.py
import pandas as pd
import pickle
import os
from datetime import datetime

# Carregar dados
metricas = pd.read_csv('tabela_metricas.csv')

with open('cv_scores.pkl', 'rb') as f:
    cv_scores = pickle.load(f)

# Criar relatório em Markdown
relatorio = f"""# Relatório de Modelagem Preditiva - Risco de Lesão
## Copa do Mundo 2026

**Data de geração:** {datetime.now().strftime('%Y-%m-%d %H:%M')}

---

## 1. Resumo Executivo

Este relatório apresenta os resultados da modelagem preditiva para **risco de lesão grave (>14 dias)** em jogadores de futebol convocados para a Copa do Mundo 2026.

### Objetivo Clínico
- **Target:** Lesão grave individual (>14 dias de paragem)
- **Métrica Principal:** Score Clínico (Recall ponderado 2:1 vs Precision)
- **Justificativa:** Falsos Negativos (lesão omitida) são mais graves que Falsos Positivos (alarme falso)

---

## 2. Dados e Preparação

### Fontes de Dados
- **jogadores_copa_2026.csv:** Lista completa de jogadores convocados (incluindo saudáveis)
- **lesoes_jogadores_mundial_REAL.csv:** Histórico de lesões por jogador
- **calendario_selecoes.csv:** Dados agregados por seleção (dias de descanso, jogos, altitude)

### Engenharia de Features
- `idade_x_jogos`: Interação idade × carga de jogos
- `idade_x_altitude`: Interação idade × altitude
- `descanso_relativo`: Dias de descanso / jogos na fase de grupos
- `risco_biomecanico`: Risco fisiológico por posição (1-6 escala)
- `carga_lesao_historica`: Soma histórica de dias parado

### Pré-processamento
- Nacionalidade removida (evita overfitting por cardinalidade excessiva)
- Encoding One-Hot apenas para posição
- Escalonamento StandardScaler em features contínuas
- Divisão estratificada 80/20

---

## 3. Modelos e Performance

### Validação Cruzada (5-fold Stratified)
"""

for nome, scores in cv_scores.items():
    relatorio += f"- **{nome}: ** AUC = {scores['auc_mean']: .3f}(±{scores['auc_std']: .3f}) | Recall = {scores['recall_mean']: .3f}"

relatorio += """
### Métricas no Conjunto de Teste (Threshold Otimizado por F2-Score)

| Modelo | Threshold | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | Score Clínico |
|---|---|---|---|---|---|---|---|---|
"""

for _, row in metricas.iterrows():
    relatorio += f"| {row['Modelo']} | {row['Threshold_Otimo']} | {row['Accuracy']: .4f} | {row['Precision']: .4f} | {row['Recall']: .4f} | {row['F1-Score']: .4f} | {row['F2-Score']: .4f} | {row['ROC-AUC']: .4f} | {row['Score_Clinico']: .4f} |"

relatorio += f"""

**Melhor Modelo:** {metricas.iloc[0]['Modelo']} (Score Clínico: {metricas.iloc[0]['Score_Clinico']:.4f})

---

## 4. Interpretação Clínica

### Threshold Otimizado
O threshold de decisão foi ajustado para **{metricas.iloc[0]['Threshold_Otimo']}** (vs 0.50 padrão) de forma a:
- Maximizar a deteção de risco (Recall)
- Aceitar um aumento controlado de alarmes falsos (Precision)

### Matriz de Confusão
Ver imagem: `04_matriz_confusao_melhor_modelo.png`

---

## 5. Limitações e Viés Conhecido

1. **Dados de lesões:** O modelo depende da qualidade e completude do registo histórico de lesões. Lesões não reportadas ou sub-notificadas introduzem viés.

2. **Variáveis agregadas:** `dias_descanso_media`, `jogos_fase_grupos` e `altitude_media` são agregados ao nível da seleção, não individuais. Não capturam carga de treino específica do jogador.

3. **Idade estimada:** Caso `idade_atual` não esteja disponível no ficheiro de lesões, é usada uma estimativa. Recomenda-se validar com data de nascimento oficial.

4. **Nacionalidade removida:** Removida do modelo final devido a cardinalidade excessiva e risco de overfitting. Poderia ser reintroduzida agrupada por confederação/continente.

5. **População:** O modelo foi treinado sobre jogadores convocados para a Copa 2026. A generalização para outras competições ou níveis de jogo não é garantida.

6. **Target:** A predição distingue "lesão grave vs leve/nenhuma", não o risco absoluto de um atleta saudável se lesionar pela primeira vez.

---

## 6. Recomendações Operacionais

1. **Monitorização contínua:** Atualizar o modelo com dados de lesões à medida que a competição avança.
2. **Threshold dinâmico:** Ajustar o threshold consoante a fase da competição (fase de grupos vs eliminatórias).
3. **Features adicionais:** Integrar dados de GPS, carga interna (RPE), e histórico médico individual se disponíveis.
4. **Interpretabilidade:** Usar SHAP values para explicar previsões individuais aos staff médico.

---

*Relatório gerado automaticamente pelo pipeline ML Mundial 2026.*
"""

with open('relatorio_final.md', 'w', encoding='utf-8') as f:
    f.write(relatorio)

print("Relatório gerado: relatorio_final.md")

# Também gerar versão HTML simples
html = relatorio.replace('# ', '<h1>').replace('## ', '<h2>').replace('### ', '<h3>')
# Substitui quebras de linha por <br> (o motivo do teu erro "br" anterior)
html = html.replace('\n', '<br>') 
html = f"<html><body>{html}</body></html>"
with open('relatorio_final.html', 'w', encoding='utf-8') as f:
    f.write(html)

print("Relatório HTML gerado: relatorio_final.html")