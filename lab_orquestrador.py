# lab_orquestrador.py
import subprocess
import sys
import os

print("--- Sistema de Orquestração: Mundial 2026 ---")
print("Pipeline de Machine Learning para Predição de Risco de Lesão\n")

scripts = [
    ("01_analise_exploratoria.py", "Análise Exploratória e Engenharia de Features"),
    ("02_pre_processamento.py", "Pré-processamento e Divisão Treino/Teste"),
    ("03_treino_modelos.py", "Treino de Modelos com Validação Cruzada"),
    ("04_avaliacao_metricas.py", "Avaliação de Métricas e Otimização de Threshold"),
    ("05_matriz_confusao.py", "Matriz de Confusão do Melhor Modelo"),
    ("06_curvas_graficas.py", "Curvas ROC e Precision-Recall"),
    ("07_gerar_relatorio.py", "Geração do Relatório Final")
]

# Verificar dependências antes de iniciar
print("[Verificação] A verificar ficheiros de dados...")
dependencias = ["lesoes_jogadores_mundial_REAL.csv",
                "calendario_selecoes.csv",
                "jogadores_copa_2026.csv"]

faltantes = [f for f in dependencias if not os.path.exists(f)]
if faltantes:
    print(f"ERRO: Ficheiros em falta: {faltantes}")
    print("Por favor, coloque os ficheiros na pasta de execução.")
    sys.exit(1)

print("✓ Todos os ficheiros encontrados.\n")

resposta = input(
    "Deseja iniciar a execução automatizada de todo o pipeline? (Sim/Não): ")
if resposta.strip().lower() not in ['sim', 's', 'yes', 'y']:
    print("Execução cancelada pelo utilizador.")
    sys.exit(0)

for script, descricao in scripts:
    print(f"\n{'='*60}")
    print(f"[Executando] {script} - {descricao}")
    print(f"{'='*60}")

    try:
        result = subprocess.run(
            [sys.executable, script], check=True, capture_output=False)
        print(f"✓ {script} concluído com sucesso.")
    except subprocess.CalledProcessError as e:
        print(f"✗ ERRO na execução de {script}.")
        print(f"Código de retorno: {e.returncode}")

        opcao = input("\n[1] Tentar de novo, [2] Saltar, [3] Abortar? ")
        if opcao == '1':
            try:
                result = subprocess.run([sys.executable, script], check=True)
                print(f"✓ {script} concluído na segunda tentativa.")
            except:
                print(f"✗ Falha persistente em {script}. Abortando.")
                sys.exit(1)
        elif opcao == '2':
            print(f"Saltando {script}...")
            continue
        else:
            print("Pipeline abortado.")
            sys.exit(1)

print("\n" + "="*60)
print("PIPELINE CONCLUÍDO COM SUCESSO!")
print("="*60)
print("\nFicheiros gerados:")
print("  - dados_base_agregados.csv (dataset processado)")
print("  - *.pkl (modelos e objetos serializados)")
print("  - *.png (visualizações)")
print("  - tabela_metricas.csv (comparativo de modelos)")
print("  - relatorio_final.md / .html (relatório final)")
