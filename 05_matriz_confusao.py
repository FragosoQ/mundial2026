# 05_matriz_confusao.py
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import os

# 1. Carregar métricas e identificar vencedor (por Score Clínico)
df_metricas = pd.read_csv('tabela_metricas.csv')
melhor_modelo_nome = df_metricas.iloc[0]['Modelo']
melhor_threshold = df_metricas.iloc[0]['Threshold_Otimo']
print(
    f"Modelo vencedor: {melhor_modelo_nome} (Threshold ótimo: {melhor_threshold})")

# 2. Carregar previsões e dados reais
with open('modelos_e_previsoes.pkl', 'rb') as f:
    resultados = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

y_prob = resultados[melhor_modelo_nome]['probabilidades']

# Aplica o threshold otimizado em vez de 0.5
y_pred = (y_prob >= melhor_threshold).astype(int)

# 3. Matriz de Confusão com labels=[0,1] (garante sempre 2x2, mesmo com previsões unimodais)
cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
tn, fp, fn, tp = cm.ravel()

total = tn + fp + fn + tp
fnr = (fn / (fn + tp)) * 100 if (fn + tp) > 0 else 0
fpr = (fp / (fp + tn)) * 100 if (fp + tn) > 0 else 0

print(f"- -- Matriz de Confusão(Threshold={melhor_threshold}) - --")
print(f"TN={tn}, FP={fp}, FN={fn}, TP={tp}")
print(f"Taxa Falsos Negativos (risco omitido): {fnr:.2f}%")
print(f"Taxa Falsos Positivos (alarme falso): {fpr:.2f}%")

# 4. Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Estável (0)', 'Alto Risco (1)'],
            yticklabels=['Estável (0)', 'Alto Risco (1)'])
plt.ylabel('Estado Clínico Real')
plt.xlabel('Previsão do Modelo')
plt.title(f'Matriz de Confusão: {melhor_modelo_nome}\n(Threshold={melhor_threshold})')
plt.tight_layout()
plt.savefig('04_matriz_confusao_melhor_modelo.png')
print("Imagem '04_matriz_confusao_melhor_modelo.png' guardada.")

# 5. Análise de Risco Operacional
print("--- Análise de Risco Operacional - --")
print(
    f"Com threshold={melhor_threshold}, aceitamos {fpr:.1f}% de alarmes falsos")
print(f"para reduzir riscos omitidos a {fnr:.1f}%.")
if fnr > 20:
    print("ALERTA: Falsos Negativos >20%. Considerar baixar threshold ou rever features.")