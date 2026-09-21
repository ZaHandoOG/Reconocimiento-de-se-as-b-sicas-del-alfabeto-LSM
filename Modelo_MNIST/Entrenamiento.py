import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# ==========================================
# 1. PREPROCESAMIENTO DE DATOS 
# ==========================================
print("--- [1/4] Cargando datos ---")
train_path = 'Vision\MiniProyecto1\Modelo_MNIST\Dataset\sign_mnist_train.csv'
test_path = 'Vision\MiniProyecto1\Modelo_MNIST\Dataset\sign_mnist_test.csv'
train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

# Separar etiquetas y píxeles
y_train_full = train_df['label'].values
X_train_full = train_df.drop('label', axis=1).values

y_test = test_df['label'].values
X_test = test_df.drop('label', axis=1).values

# Normalización Min-Max (0 a 1) y Reshape a tensor 4D: (N, 28, 28, 1)
X_train_full = X_train_full.reshape(-1, 28, 28, 1).astype('float32') / 255.0
X_test = X_test.reshape(-1, 28, 28, 1).astype('float32') / 255.0

# Split de Validación (85% entrenamiento, 15% validación)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.15, random_state=42, stratify=y_train_full
)

print(f"Muestras Train: {X_train.shape[0]}")
print(f"Muestras Validation: {X_val.shape[0]}")
print(f"Muestras Test: {X_test.shape[0]}")

# Data Augmentation ligero para robustez ante la webcam
datagen = ImageDataGenerator(
    rotation_range=10,
    zoom_range=0.1,
    width_shift_range=0.1,
    height_shift_range=0.1
)
datagen.fit(X_train)

# ==========================================
# 2. ARQUITECTURA DE LA CNN (Rúbrica 2)
# ==========================================
print("\n--- [2/4] Construyendo arquitectura CNN ---")
model = Sequential([
    # Bloque Convolucional 1: Extracción de características de bajo nivel (bordes, gradientes)
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.2),

    # Bloque Convolucional 2: Características intermedias (formas de dedos, contornos de mano)
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.25),

    # Bloque Convolucional 3: Patrones espaciales combinados
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2, 2)),
    Dropout(0.3),

    # Clasificador Denso
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.4),
    Dense(25, activation='softmax')  # 25 salidas para manejar índices 0 a 24
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ==========================================
# 3. ENTRENAMIENTO Y CALLBACKS (Rúbrica 3)
# ==========================================
print("\n--- [3/4] Entrenando modelo ---")
os.makedirs("models", exist_ok=True)
os.makedirs("docs", exist_ok=True)

callbacks = [
    # Detiene si no mejora en 5 épocas y restaura los mejores pesos
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    # Guarda el mejor modelo automáticamente
    ModelCheckpoint(filepath='models/best_sign_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    # Reduce el learning rate si entra en meseta
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1, min_lr=1e-5)
]

batch_size = 64
epochs = 25

history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    epochs=epochs,
    validation_data=(X_val, y_val),
    callbacks=callbacks
)

# ==========================================
# 4. EVALUACIÓN CUANTITATIVA (Rúbrica 4)
# ==========================================
print("\n--- [4/4] Evaluando en conjunto Test independiente ---")
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n>> Exactitud final en Test (Accuracy): {test_acc * 100:.2f}% <<")

# Predicciones
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)

# Mapeo a letras del alfabeto para el reporte
letras = [chr(i) for i in range(65, 91)]  # A-Z
letras_presentes = [letras[i] for i in sorted(np.unique(y_test))]

print("\n--- Reporte de Clasificación (Precision / Recall / F1-Score) ---")
print(classification_report(y_test, y_pred, target_names=letras_presentes))

# Matriz de Confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(14, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=letras_presentes, yticklabels=letras_presentes)
plt.title(f"Matriz de Confusión - Sign Language MNIST (Test Acc: {test_acc*100:.1f}%)")
plt.xlabel("Clase Predicha")
plt.ylabel("Clase Real")
plt.tight_layout()
plt.savefig("Vision\MiniProyecto1\Modelo_MNIST\Output/confusion_matrix.png", dpi=300)
print("-> Matriz de confusión guardada en 'Vision\MiniProyecto1\Modelo_MNIST\Output/confusion_matrix.png'")

# Gráficas de pérdida y exactitud
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Acc')
plt.plot(history.history['val_accuracy'], label='Val Acc')
plt.title('Evolución de Accuracy')
plt.xlabel('Época')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Evolución de Loss')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("Vision\MiniProyecto1\Modelo_MNIST\Output/training_history.png", dpi=300)
print("-> Gráfica de entrenamiento guardada en 'docs/training_history.png'")