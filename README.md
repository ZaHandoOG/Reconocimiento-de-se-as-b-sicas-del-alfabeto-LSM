# 🤟 Reconocimiento del alfabeto de la Lengua de Señas Mexicana (LSM)

Proyecto de Visión por Computadora (Universidad Modelo) que reconoce las letras del alfabeto en Lengua de Señas Mexicana. Ha pasado por tres etapas: (1) una comparación entre una CNN sobre imágenes 28x28 y un Random Forest sobre puntos de la mano (MediaPipe); (2) captura automática de letras con salida a texto y voz en español; y (3) un detector YOLO entrenado con un dataset de LSM de Roboflow, que es la versión más reciente.

---

## 🧭 Etapas del proyecto

| Etapa | Carpeta | Qué hace |
| :--- | :--- | :--- |
| Parcial 1 | `Modelo_MNIST/`, `Modelo MediaPipe/`, `Evaluate/` | CNN (Sign Language MNIST) contra MediaPipe + Random Forest |
| Avance 1 | `Modelo MediaPipe/Camra_En_Vivo.py` | Captura automática de letras, texto y voz en español |
| Parcial 2 (actual) | `YOLO/` | Detector YOLOv8 entrenado con el dataset de Roboflow (27 clases: a-z y rr) |
| Reportes | `Reportes/` | Reportes entregados en PDF y Word |

### 🟢 YOLO (versión más reciente)

* **Modelo:** YOLOv8n (Ultralytics), 50 épocas, imágenes de 640 px, con aumentación de datos.
* **Dataset:** [Lengua de Señas Mexicana (Roboflow)](https://universe.roboflow.com/datasetbyte/lengua-de-senas-mexicana). El dataset **no se sube** al repo (son miles de imágenes con personas); se descarga en formato YOLOv8 y se coloca en `YOLO/Lengua de Senas Mexicana.v3i.yolov8/` con las carpetas `train/`, `valid/` y `test/`.
* **Resultado en validación:** mAP@50 de aproximadamente 0.71 (mejor época: la 48). Las curvas y la matriz de confusión están en `YOLO/.../modelo_lsm_v1-2/`, y los pesos entrenados en `weights/best.pt`.
* **Reentrenar:** `python train.py` desde la carpeta del dataset. Ajusta las rutas absolutas y `device` dentro del script (usa `device="cpu"` si no tienes GPU NVIDIA).
* **Cámara en vivo:** `python run.py`. En el script hay que apuntar `model_path` a `weights/best.pt`; la voz usa pyttsx3 con una voz en español instalada en Windows.

> Nota: los mAP de YOLO (detección) no son directamente comparables con la exactitud de la CNN o del Random Forest (clasificación), así que la comparación con ellos es cualitativa.

---

## 📊 Parcial 1: Enfoques Comparados (CNN vs Random Forest)

El sistema evalúa dos metodologías distintas para el reconocimiento gestual:

| Característica | Enfoque 1: CNN (Deep Learning) | Enfoque 2: MediaPipe + Random Forest |
| :--- | :--- | :--- |
| **Dataset de Origen** | Sign Language MNIST (27k train / 7k test) | Dataset customizado (`dataset_lsm.csv`) |
| **Entrada del Modelo** | Matriz de píxeles en escala de grises ($28 \times 28 \times 1$) | Vector normalizado de 126 variables espaciales |
| **Representación** | Espacio denso de píxeles | Coordenadas $(x, y, z)$ relativas a la muñeca |
| **Clasificador** | Red Convolucional (Conv2D + MaxPool + Dropout) | Random Forest Classifier (100 estimadores) |
| **Invarianza al Entorno**| Sensible al fondo, iluminación y textura | Invariante a luz y fondo; enfocado en geometría |
| **Inferencia** | Requiere encuadre en Región de Interés (ROI) | Libre posicionamiento en el encuadre de la cámara |

---

## 🛠️ Tecnologías Utilizadas

* **Lenguaje:** Python 3.11
* **Deep Learning & ML:** TensorFlow 2.16.2, Scikit-Learn, Ultralytics YOLOv8
* **Visión Artificial:** OpenCV 4.11, MediaPipe 0.10.14
* **Interfaces de Despliegue:** Streamlit, OpenCV (cámara en vivo), pyttsx3 (voz)
* **Procesamiento de Datos:** NumPy, Pandas

---

## 🚀 Instalación y Entorno

1. **Clonar el repositorio:**
   ```bash
   git clone https://github.com/ZaHandoOG/Reconocimiento-de-se-as-b-sicas-del-alfabeto-LSM.git
   cd Reconocimiento-de-se-as-b-sicas-del-alfabeto-LSM
   ```

2. **Crear y activar un entorno virtual (recomendado):**
   ```bash
   python -m venv venv
   # En Windows:
   .\venv\Scripts\activate
   ```

3. **Instalar dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
   > *Nota de compatibilidad:* Se fuerza la instalación de `protobuf==4.25.3` para garantizar la interoperabilidad entre TensorFlow 2.16.2 y MediaPipe 0.10.14.

---

## 🖥️ Ejecución

### Streamlit (CNN y Random Forest)
Panel de control, umbral de aceptación ajustable, vista previa de ROI ($28 \times 28$) y visualización en tiempo real:
```bash
streamlit run App_Streamlit.py
```

### Cámara en vivo con MediaPipe + Random Forest
```bash
python "Modelo MediaPipe/Camra_En_Vivo.py"
```

### Cámara en vivo con YOLO
```bash
python "YOLO/Lengua de Senas Mexicana.v3i.yolov8/run.py"
```

### Datos que no están en el repositorio
* **Sign Language MNIST** (`sign_mnist_train.csv`, 81 MB): descárgalo de [Kaggle](https://www.kaggle.com/datasets/datamunge/sign-language-mnist) y colócalo en `Modelo_MNIST/Dataset/`.
* **Dataset de YOLO:** ver la sección de YOLO.

---

## 🔬 Consideraciones Técnicas de Implementación

1. **Mapeo de Clases en Sign Language MNIST:**  
   El dataset original excluye las letras 'J' (índice 9) y 'Z' (índice 25) debido a su naturaleza dinámica. La capa de salida de la CNN maneja 25 neuronas activas; la interfaz implementa un mapeo seguro para evitar desbordamientos de índice al predecir la letra 'Y' (índice 24).

2. **Corrección de Modo Espejo en MediaPipe:**  
   Al aplicar `cv2.flip(frame, 1)` para un espejo natural del usuario, las etiquetas de lateralidad se invierten. El script incorpora una reasignación lógica para asegurar que los 63 landmarks de la mano derecha física se ubiquen consistentemente en las posiciones 64 a 126 del vector de características.

3. **Filtrado Temporal:**  
   Ambos despliegues integran un buffer rodante (`deque`) que calcula la moda estadística de las últimas predicciones, estabilizando la salida en pantalla y eliminando el parpadeo de falsos positivos entre cuadros consecutivos.
