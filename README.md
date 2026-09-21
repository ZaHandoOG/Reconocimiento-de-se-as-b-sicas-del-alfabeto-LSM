# 🤟 Sistema de Reconocimiento de Lengua de Señas (LSM / ASL)

Sistema de visión artificial para clasificación e inferencia en tiempo real de señas estáticas del alfabeto dactilológico. El proyecto implementa y compara dos enfoques arquitectónicos: una Red Neuronal Convolucional (CNN) sobre imágenes 28x28 y un ensamble de Random Forest sobre coordenadas articulares tridimensionales (MediaPipe Hands), complementado con despliegues interactivos en Streamlit y Gradio.

---

## 📊 Arquitectura y Enfoques Comparados

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
* **Deep Learning & ML:** TensorFlow 2.16.2, Scikit-Learn
* **Visión Artificial:** OpenCV 4.11, MediaPipe 0.10.14
* **Interfaces de Despliegue:** Streamlit, Gradio
* **Procesamiento de Datos:** NumPy, Pandas

---

## 🚀 Instalación y Entorno

1. **Clonar el repositorio:**
   ```bash
   git clone [https://github.com/TU_USUARIO/TU_REPOSITORIO.git](https://github.com/TU_USUARIO/TU_REPOSITORIO.git)
   cd TU_REPOSITORIO
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

## 🖥️ Ejecución de los Despliegues

El proyecto incluye dos interfaces gráficas completas con soporte para transmisión de cámara web en vivo:

### Opción A: Despliegue en Streamlit
Incluye panel de control, umbral de aceptación ajustable, vista previa de ROI ($28 \times 28$) y visualización en tiempo real:
```bash
streamlit run App.py
```

### Opción B: Despliegue en Gradio
Permite streaming directo y modo de evaluación sobre imágenes estáticas con diagnóstico de probabilidades Top-3:
```bash
python AppGradio.py
```

---

## 🔬 Consideraciones Técnicas de Implementación

1. **Mapeo de Clases en Sign Language MNIST:**  
   El dataset original excluye las letras 'J' (índice 9) y 'Z' (índice 25) debido a su naturaleza dinámica. La capa de salida de la CNN maneja 25 neuronas activas; la interfaz implementa un mapeo seguro para evitar desbordamientos de índice al predecir la letra 'Y' (índice 24).

2. **Corrección de Modo Espejo en MediaPipe:**  
   Al aplicar `cv2.flip(frame, 1)` para un espejo natural del usuario, las etiquetas de lateralidad se invierten. El script incorpora una reasignación lógica para asegurar que los 63 landmarks de la mano derecha física se ubiquen consistentemente en las posiciones 64 a 126 del vector de características.

3. **Filtrado Temporal:**  
   Ambos despliegues integran un buffer rodante (`deque`) que calcula la moda estadística de las últimas predicciones, estabilizando la salida en pantalla y eliminando el parpadeo de falsos positivos entre cuadros consecutivos.
