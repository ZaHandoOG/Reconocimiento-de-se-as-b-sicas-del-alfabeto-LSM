"""
Evaluación cuantitativa del modelo Random Forest entrenado sobre landmarks
de MediaPipe (126 features: 2 manos x 21 puntos x 3 coords).

Reproduce EXACTAMENTE el split usado en el entrenamiento original
(train_test_split con test_size=0.2, random_state=42, stratify=y sobre
dataset_lsm.csv), así que el "test" aquí es el mismo holdout que nunca
vio el modelo durante el fit — no una aproximación.

Genera:
  - accuracy global
  - reporte de clasificación (precision / recall / F1 por clase) -> CSV y TXT
  - matriz de confusión -> PNG
  - resumen JSON (usado por compare_models.py)

Requiere: scikit-learn, pandas, numpy, matplotlib, seaborn
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURACIÓN — AJUSTA ESTAS RUTAS
# ==========================================
PATH_RF = r"C:\Users\gaboo\OneDrive\Documentos\Coding\Vision\MiniProyecto1\Modelo MediaPipe\modelo_lsm_mediapipe.pkl"
PATH_LANDMARKS_CSV = r"C:\Users\gaboo\Downloads\dataset_lsm.csv"
OUTPUT_DIR = r"C:\Users\gaboo\OneDrive\Documentos\Coding\Vision\resultados_rf"

# Mismos parámetros de split que en el script de entrenamiento original
TEST_SIZE = 0.2
RANDOM_STATE = 42


def cargar_xy(path_csv):
    """Misma lógica que el script de entrenamiento: columna 0 = etiqueta
    (string), columnas 1: = 126 coordenadas de landmarks. Descarta filas
    con NaN igual que en el entrenamiento."""
    df = pd.read_csv(path_csv)
    filas_originales = len(df)
    
    # Limpieza de nulos y reseteo de índice
    df = df.dropna().reset_index(drop=True)
    if len(df) != filas_originales:
        print(f"  Filas descartadas con NaN: {filas_originales - len(df)}")
        
    X = np.ascontiguousarray(df.iloc[:, 1:].to_numpy(), dtype=np.float32)
    y = np.ascontiguousarray(df.iloc[:, 0].astype(str).str.strip().to_numpy()).ravel()
    
    return X, y


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Cargando modelo Random Forest...")
    with open(PATH_RF, "rb") as f:
        model = pickle.load(f)

    print("Cargando dataset de landmarks y reproduciendo el split original...")
    X, y = cargar_xy(PATH_LANDMARKS_CSV)
    
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"  -> {X_test.shape[0]} muestras de prueba (holdout real), {X_test.shape[1]} features")

    print("Generando predicciones...")
    y_pred = model.predict(X_test)

    clases = sorted(set(y_test) | set(y_pred))
    acc = accuracy_score(y_test, y_pred)
    print(f"\nAccuracy global: {acc * 100:.2f}%")

    reporte_dict = classification_report(
        y_test, y_pred, labels=clases, output_dict=True, zero_division=0
    )
    reporte_txt = classification_report(
        y_test, y_pred, labels=clases, zero_division=0
    )
    print("\n" + reporte_txt)

    pd.DataFrame(reporte_dict).transpose().to_csv(
        os.path.join(OUTPUT_DIR, "classification_report_rf.csv")
    )
    with open(os.path.join(OUTPUT_DIR, "classification_report_rf.txt"), "w", encoding="utf-8") as f:
        f.write(f"Accuracy global: {acc * 100:.2f}%\n\n")
        f.write(reporte_txt)

    cm = confusion_matrix(y_test, y_pred, labels=clases)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens",
                xticklabels=clases, yticklabels=clases, cbar=True)
    plt.title(f"Matriz de Confusión — Random Forest (MediaPipe)\nAccuracy: {acc * 100:.2f}%")
    plt.xlabel("Predicción")
    plt.ylabel("Etiqueta real")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "matriz_confusion_rf.png"), dpi=150)
    plt.close()

    resumen = {
        "modelo": "Random Forest (MediaPipe landmarks)",
        "accuracy": float(acc),
        "macro_f1": float(reporte_dict["macro avg"]["f1-score"]),
        "macro_precision": float(reporte_dict["macro avg"]["precision"]),
        "macro_recall": float(reporte_dict["macro avg"]["recall"]),
        "n_muestras_test": int(X_test.shape[0]),
    }
    with open(os.path.join(OUTPUT_DIR, "resumen_rf.json"), "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"\nResultados guardados en: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()