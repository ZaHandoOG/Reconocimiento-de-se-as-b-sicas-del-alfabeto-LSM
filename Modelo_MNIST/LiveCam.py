import cv2
import numpy as np
import tensorflow as tf
import os

# ==========================================
# 1. CARGA DEL MODELO ENTRENADO
# ==========================================
model_path = "models/best_sign_model.keras"

if not os.path.exists(model_path):
    print(f"[ERROR] No se encontro el archivo del modelo en: {model_path}")
    print("Asegurate de haber corrido exitosamente '01_train_and_evaluate.py' primero.")
    exit()

print("[INFO] Cargando modelo convolucional...")
model = tf.keras.models.load_model(model_path)
print("[INFO] Modelo cargado con exito.")

# Alfabeto: del 0 al 24 (sin la 'J' en el indice 9)
ALFABETO = [chr(i) for i in range(65, 91)]  # A-Z

# ==========================================
# 2. INICIALIZACIÓN DE LA CÁMARA
# ==========================================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("[ERROR] No se pudo acceder a la camara web.")
    print("Verifica que ninguna otra app la este usando y que los permisos esten activos.")
    exit()

# Dimensiones del recuadro ROI (Region of Interest)
ROI_TOP = 100
ROI_BOTTOM = 350
ROI_RIGHT = 350
ROI_LEFT = 600

UMBRAL_CONFIANZA = 0.70  # 70% minimo de certeza para aceptar la prediccion

print("\n=== TRADUCTOR DE SEÑAS EN VIVO INICIADO ===")
print("Coloca tu mano dentro del recuadro verde.")
print("Presiona la tecla 'q' o 'ESC' para salir.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Fallo al capturar fotograma de la camara.")
        break

    # Espejo horizontal para que el movimiento sea intuitivo
    frame = cv2.flip(frame, 1)

    # Extraer la subimagen de la Region de Interes (ROI)
    roi = frame[ROI_TOP:ROI_BOTTOM, ROI_RIGHT:ROI_LEFT]

    # Preprocesamiento de la imagen para la CNN:
    # 1. Escala de grises
    gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # 2. Redimensionar a 28x28 píxeles
    resized_roi = cv2.resize(gray_roi, (28, 28), interpolation=cv2.INTER_AREA)
    # 3. Normalizar rango a [0.0, 1.0]
    normalized_roi = resized_roi.astype('float32') / 255.0
    # 4. Ajustar forma del tensor a (1, 28, 28, 1)
    input_tensor = normalized_roi.reshape(1, 28, 28, 1)

    # Inferencia con la CNN
    prediccion = model.predict(input_tensor, verbose=0)[0]
    clase_id = np.argmax(prediccion)
    confianza = float(prediccion[clase_id])

    # Letra correspondiente
    letra_predicha = ALFABETO[clase_id]

    # Feedback visual condicional según nivel de confianza
    if confianza >= UMBRAL_CONFIANZA:
        texto_resultado = f"Letra: {letra_predicha}"
        texto_confianza = f"Certeza: {confianza * 100:.1f}%"
        color_borde = (0, 255, 0)  # Verde
    else:
        texto_resultado = "Esperando sena..."
        texto_confianza = f"Certeza: {confianza * 100:.1f}%"
        color_borde = (0, 165, 255)  # Naranja

    # Dibujar el marco del ROI en pantalla
    cv2.rectangle(frame, (ROI_RIGHT, ROI_TOP), (ROI_LEFT, ROI_BOTTOM), color_borde, 2)
    cv2.putText(frame, "Pon tu mano aqui", (ROI_RIGHT, ROI_TOP - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_borde, 2)

    # Panel informativo superior
    cv2.rectangle(frame, (10, 10), (320, 110), (30, 30, 30), -1)
    cv2.putText(frame, texto_resultado, (20, 45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, texto_confianza, (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    # Barra de progreso visual de confianza
    ancho_barra = int(280 * confianza)
    cv2.rectangle(frame, (20, 95), (20 + ancho_barra, 102), color_borde, -1)

    # Miniatura en la esquina para ver qué recibe el modelo en 28x28
    preview_28x28 = cv2.resize(resized_roi, (84, 84), interpolation=cv2.INTER_NEAREST)
    preview_bgr = cv2.cvtColor(preview_28x28, cv2.COLOR_GRAY2BGR)
    frame[10:94, frame.shape[1] - 94:frame.shape[1] - 10] = preview_bgr
    cv2.putText(frame, "Input 28x28", (frame.shape[1] - 94, 105),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    # Mostrar la ventana en vivo
    cv2.imshow("Traductor de Lenguaje de Senas - ASL", frame)

    # Salida con 'q' o tecla ESC (27)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or key == 27:
        break

# Liberar recursos de la camara
cap.release()
cv2.destroyAllWindows()
print("[INFO] Programa finalizado correctamente.")