"""
Compara los resultados de evaluate_cnn.py y evaluate_rf.py.
Corre AMBOS scripts primero -- este solo lee sus resumen_*.json.

Genera una tabla comparativa (CSV) y una gráfica de barras (PNG) con
accuracy y F1-macro de cada modelo. Cubre el criterio de bono:
"comparación entre dos arquitecturas distintas" (+5 pts).
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

RESUMEN_CNN = r"Vision\MiniProyecto1\Modelo_MNIST\Output\resumen_cnn.json"
RESUMEN_RF = r"C:\Users\gaboo\OneDrive\Documentos\Coding\Vision\resultados_rf\resumen_rf.json"
OUTPUT_DIR = r"C:\Users\gaboo\OneDrive\Documentos\Coding\Vision\resultados_comparacion"


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    with open(RESUMEN_CNN, encoding="utf-8") as f:
        cnn = json.load(f)
    with open(RESUMEN_RF, encoding="utf-8") as f:
        rf = json.load(f)

    tabla = pd.DataFrame([cnn, rf]).set_index("modelo")
    print(tabla[["accuracy", "macro_f1", "macro_precision", "macro_recall", "n_muestras_test"]])
    tabla.to_csv(os.path.join(OUTPUT_DIR, "tabla_comparativa.csv"))

    fig, ax = plt.subplots(figsize=(8, 5))
    metricas = ["accuracy", "macro_precision", "macro_recall", "macro_f1"]
    x = range(len(metricas))
    ancho = 0.35

    ax.bar([i - ancho / 2 for i in x], [cnn[m] for m in metricas], ancho, label=cnn["modelo"])
    ax.bar([i + ancho / 2 for i in x], [rf[m] for m in metricas], ancho, label=rf["modelo"])
    ax.set_xticks(list(x))
    ax.set_xticklabels(["Accuracy", "Precision\n(macro)", "Recall\n(macro)", "F1\n(macro)"])
    ax.set_ylim(0, 1.0)
    ax.set_title("Comparación CNN vs Random Forest + MediaPipe")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "comparacion_modelos.png"), dpi=150)
    plt.close()

    print(f"\nComparación guardada en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()