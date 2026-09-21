import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Cargar el dataset
print("[INFO] Cargando dataset_lsm.csv...")
df = pd.read_csv(r"C:\Users\gaboo\Downloads\dataset_lsm.csv")

# 2. Limpieza de filas vacías (elimina la fila 496 con NaN)
filas_originales = len(df)
df = df.dropna()
print(f"[INFO] Filas descartadas con NaN: {filas_originales - len(df)}")
print(f"[INFO] Muestras listas para entrenar: {len(df)}")

# 3. Separar X (126 coordenadas) e y (etiqueta)
X = df.iloc[:, 1:].values.astype('float32')
y = df.iloc[:, 0].astype(str).values

# 4. Partición estratificada (80% entrenamiento, 20% prueba)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 5. Entrenar el clasificador Random Forest
print("[INFO] Entrenando clasificador...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# 6. Evaluación formal
y_pred = clf.predict(X_test)
exactitud = accuracy_score(y_test, y_pred)
print(f"\n==========================================")
print(f">> EXACTITUD EN TEST: {exactitud * 100:.2f}% <<")
print(f"==========================================\n")
print(classification_report(y_test, y_pred))

# 7. Guardar el modelo en formato .pkl
output_model = "modelo_lsm_mediapipe.pkl"
with open(output_model, "wb") as f:
    pickle.dump(clf, f)

print(f"[EXITO] Modelo guardado como '{output_model}'.")