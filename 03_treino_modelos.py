# 03_treino_modelos.py
import pickle
import time
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold, cross_validate
import os

# 1. Carregamento
arquivos = ['X_train.pkl', 'X_test.pkl', 'y_train.pkl', 'y_test.pkl']
if not all(os.path.exists(f) for f in arquivos):
    print("ERRO: Ficheiros de treino/teste não encontrados. Execute o script 02.")
    exit()

with open('X_train.pkl', 'rb') as f:
    X_train = pickle.load(f)
with open('X_test.pkl', 'rb') as f:
    X_test = pickle.load(f)
with open('y_train.pkl', 'rb') as f:
    y_train = pickle.load(f)
with open('y_test.pkl', 'rb') as f:
    y_test = pickle.load(f)

# 2. Modelos com tratamento de desbalanceamento
modelos = {
    'Regressão Logística': LogisticRegression(
        max_iter=2000, random_state=42, class_weight='balanced', n_jobs=-1),
    'Random Forest': RandomForestClassifier(
        n_estimators=300, random_state=42, class_weight='balanced',
        max_depth=12, min_samples_leaf=5, n_jobs=-1),
    'KNN': KNeighborsClassifier(n_neighbors=5),
    'Naïve Bayes': GaussianNB()
}

resultados = {}
cv_scores = {}

# 3. Validação Cruzada + Treino Final
print("--- Treino com Validação Cruzada (5-fold Stratified) ---")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

for nome, modelo in modelos.items():
    inicio = time.time()

    # CV apenas para modelos que suportam (LR e RF com class_weight)
    if nome in ['Regressão Logística', 'Random Forest']:
        scores = cross_validate(
            modelo, X_train, y_train, cv=cv,
            scoring=['roc_auc', 'f1', 'recall'],
            return_train_score=False, n_jobs=-1
        )
        cv_scores[nome] = {
            'auc_mean': scores['test_roc_auc'].mean(),
            'auc_std': scores['test_roc_auc'].std(),
            'recall_mean': scores['test_recall'].mean()
        }
        print(f"{nome} | CV AUC: {cv_scores[nome]['auc_mean']:.3f} "
              f"(+/- {cv_scores[nome]['auc_std']:.3f}) | "
              f"CV Recall: {cv_scores[nome]['recall_mean']:.3f}")

    # Treino final no conjunto completo de treino
    modelo.fit(X_train, y_train)
    fim = time.time()

    predicoes = modelo.predict(X_test)
    probabilidades = modelo.predict_proba(X_test)[:, 1]

    resultados[nome] = {
        'modelo': modelo,
        'predicoes': predicoes,
        'probabilidades': probabilidades
    }

    print(f"Modelo '{nome}' treinado em {fim - inicio:.2f}s.")

# 4. Exportação
with open('modelos_e_previsoes.pkl', 'wb') as f:
    pickle.dump(resultados, f)

with open('cv_scores.pkl', 'wb') as f:
    pickle.dump(cv_scores, f)

print("Ficheiros gerados: modelos_e_previsoes.pkl, cv_scores.pkl")
