# 04_avaliacao_metricas.py
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, fbeta_score)
import os

# 1. Carregamento
if not os.path.exists('modelos_e_previsoes.pkl') or not os.path.exists('y_test.pkl'):
    print("ERRO: Ficheiros de resultados não encontrados.")
    exit()

with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)
with open('modelos_e_previsoes.pkl', 'rb') as f:
    resultados = pickle.load(f)

metricas_lista = []
thresholds_otimizados = {}

# 2. Otimização de Threshold com F2-Score (beta=2 prioriza Recall)
print("--- Otimização de Threshold (F2-Score) ---")
for nome, res in resultados.items():
    y_prob = res['probabilidades']

    thresholds = np.arange(0.05, 0.95, 0.05)
    melhor_f2, melhor_thresh = 0, 0.5

    for t in thresholds:
        y_pred_t = (y_prob >= t).astype(int)
        f2 = fbeta_score(y_test, y_pred_t, beta=2, zero_division=0)
        if f2 > melhor_f2:
            melhor_f2 = f2
            melhor_thresh = t

    thresholds_otimizados[nome] = melhor_thresh

    # Métricas no threshold otimizado
    y_pred_opt = (y_prob >= melhor_thresh).astype(int)

    metricas_lista.append({
        'Modelo': nome,
        'Threshold_Otimo': round(melhor_thresh, 2),
        'Accuracy': accuracy_score(y_test, y_pred_opt),
        'Precision': precision_score(y_test, y_pred_opt, zero_division=0),
        'Recall': recall_score(y_test, y_pred_opt, zero_division=0),
        'F1-Score': f1_score(y_test, y_pred_opt, zero_division=0),
        'F2-Score': melhor_f2,
        'ROC-AUC': roc_auc_score(y_test, y_prob)
    })

    print(f"{nome}: Threshold ótimo = {melhor_thresh:.2f} | F2 = {melhor_f2:.4f}")

# 3. Score Clínico customizado: Recall pesa 2x mais que Precision
df_metricas = pd.DataFrame(metricas_lista)
df_metricas['Score_Clinico'] = (
    2 * df_metricas['Recall'] + df_metricas['Precision']) / 3
df_metricas = df_metricas.sort_values(
    by='Score_Clinico', ascending=False).round(4)

# 4. Exportação
df_metricas.to_csv('tabela_metricas.csv', index=False)

with open('thresholds_otimizados.pkl', 'wb') as f:
    pickle.dump(thresholds_otimizados, f)

with open('tabela_metricas.md', 'w', encoding='utf-8') as f:
    f.write("| Modelo | Threshold | Accuracy | Precision | Recall | F1 | F2 | ROC-AUC | Score Clínico |             ")
    f.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
    for _, row in df_metricas.iterrows():
        f.write(f"| {row['Modelo']} | {row['Threshold_Otimo']} | {row['Accuracy']} | "
                f"{row['Precision']} | {row['Recall']} | {row['F1-Score']} | "
                f"{row['F2-Score']} | {row['ROC-AUC']} | {row['Score_Clinico']} |                 ")

# 5. Apresentação
print("--- Tabela Comparativa(Ordenada por Score Clínico) - --")
print(df_metricas[['Modelo', 'Threshold_Otimo', 'Recall', 'Precision',
                   'F2-Score', 'ROC-AUC', 'Score_Clinico']].to_string(index=False))

print("--- Análise Crítica - --")
print("O Score Clínico pondera o Recall (sensibilidade) em 2:1 face à Precision.")
print("Falso Negativo = lesão omitida (risco para atleta).")
print("Falso Positivo = alarme falso (custo operacional aceitável).")
