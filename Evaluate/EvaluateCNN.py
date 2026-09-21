"""
Evaluación cuantitativa del modelo CNN entrenado sobre Sign Language MNIST.

Genera:
  - accuracy global
  - reporte de clasificación (precision / recall / F1 por clase) -> CSV y TXT
  - matriz de confusión -> PNG
  - resumen JSON (usado por compare_models.py)

Requiere: tensorflow, pandas, numpy, scikit-learn, matplotlib, seaborn
"""

import os
import json
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ==========================================
# CONFIGURACIÓN — AJUSTA ESTAS RUTAS
# ==========================================
PATH_CNN = r"C:\Users\gaboo\OneDrive\Documentos\Coding\models\best_sign_model.keras"
PATH_TEST_CSV = r"Vision\MiniProyecto1\Modelo_MNIST\Dataset\sign_mnist_test.csv"
OUTPUT_DIR = r"Vision\MiniProyecto1\Modelo_MNIST\Output"

# Mapeo oficial Sign Language MNIST (25 clases, 0-24; 'J'=9 no está en el dataset)
MAPEO_MNIST = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: 'J(N/A)', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}


def cargar_datos(path_csv):
    """Carga el CSV de test de Sign Language MNIST: columna 'label' + 784 píxeles."""
    df = pd.read_csv(path_csv)
    y = df["label"].values
    X = df.drop(columns=["label"]).values.astype("float32") / 255.0
    X = X.reshape(-1, 28, 28, 1)
    return X, y


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Cargando modelo CNN...")
    model = tf.keras.models.load_model(PATH_CNN)

    print("Cargando datos de prueba...")
    X_test, y_test = cargar_datos(PATH_TEST_CSV)
    print(f"  -> {X_test.shape[0]} muestras de prueba")

    print("Generando predicciones...")
    probs = model.predict(X_test, verbose=0)
    y_pred = np.argmax(probs, axis=1)

    # Etiquetas presentes realmente en el test set (evita filas vacías en el reporte)
    etiquetas_presentes = sorted(set(y_test) | set(y_pred))
    nombres = [MAPEO_MNIST.get(i, str(i)) for i in etiquetas_presentes]

    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy global: {acc * 100:.2f}%")

    reporte_dict = classification_report(
        y_test, y_pred, labels=etiquetas_presentes, target_names=nombres,
        output_dict=True, zero_division=0
    )
    reporte_txt = classification_report(
        y_test, y_pred, labels=etiquetas_presentes, target_names=nombres,
        zero_division=0
    )
    print("\n" + reporte_txt)

    # Guardar reporte
    pd.DataFrame(reporte_dict).transpose().to_csv(
        os.path.join(OUTPUT_DIR, "classification_report_cnn.csv")
    )
    with open(os.path.join(OUTPUT_DIR, "classification_report_cnn.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy global: {acc * 100:.2f}%\n\n")
        f.write(reporte_txt)

    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred, labels=etiquetas_presentes)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=nombres, yticklabels=nombres, cbar=True)
    plt.title(f"Matriz de Confusión — CNN (Sign Language MNIST)\nAccuracy: {acc * 100:.2f}%")
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "matriz_confusion_cnn.png"), dpi=150)
    plt.close()

    # Resumen JSON para compare_models.py
    resumen = {
        "modelo": "CNN (Sign Language MNIST)",
        "accuracy": acc,
        "macro_f1": reporte_dict["macro avg"]["f1-score"],
        "macro_precision": reporte_dict["macro avg"]["precision"],
        "macro_recall": reporte_dict["macro avg"]["recall"],
        "n_muestras_test": int(X_test.shape[0]),
    }
    with open(os.path.join(OUTPUT_DIR, "resumen_cnn.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()