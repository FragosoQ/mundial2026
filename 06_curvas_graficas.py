# 06_curvas_graficas.py
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import os

# Carregar dados
with open('modelos_e_previsoes.pkl', 'rb') as f:
    resultados = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# 1. Curva ROC
plt.figure(figsize=(10, 8))
for nome, res in resultados.items():
    y_prob = res['probabilidades']
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{nome} (AUC = {roc_auc:.3f})')

plt.plot([0, 1], [0, 1], 'k--', label='Aleatório')
plt.xlabel('Taxa Falsos Positivos')
plt.ylabel('Taxa Verdadeiros Positivos')
plt.title('Curva ROC - Comparativo de Modelos')
plt.legend(loc='lower right')
plt.grid(True)
plt.tight_layout()
plt.savefig('05_curva_roc.png')
plt.close()

# 2. Curva Precision-Recall
plt.figure(figsize=(10, 8))
for nome, res in resultados.items():
    y_prob = res['probabilidades']
    precision, recall, _ = precision_recall_curve(y_test, y_prob)
    plt.plot(recall, precision, label=f'{nome}')

plt.xlabel('Recall (Sensibilidade)')
plt.ylabel('Precision')
plt.title('Curva Precision-Recall')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('06_precision_recall.png')
plt.close()

# 3. Importância das Features (Random Forest)
rf_model = resultados.get('Random Forest', {}).get('modelo')
if rf_model and hasattr(rf_model, 'feature_importances_'):
    importances = rf_model.feature_importances_
    features = pd.read_pickle('X_train.pkl').columns

    feat_imp = pd.DataFrame({'Feature': features, 'Importance': importances})
    feat_imp = feat_imp.sort_values('Importance', ascending=True).tail(15)

    plt.figure(figsize=(10, 8))
    plt.barh(feat_imp['Feature'], feat_imp['Importance'])
    plt.xlabel('Importância')
    plt.title('Top 15 Features - Random Forest')
    plt.tight_layout()
    plt.savefig('07_importancia_features.png')
    plt.close()
    print("Gráfico de importância das features guardado.")

print("Curvas e gráficos gerados com sucesso.")
