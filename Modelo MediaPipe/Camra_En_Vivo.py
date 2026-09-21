import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
from collections import deque, Counter

# ==========================================
# 1. CARGA DEL MODELO RANDOM FOREST
# ==========================================
model_path = "Vision\MiniProyecto1\Modelo MediaPipe\modelo_lsm_mediapipe.pkl"

if not os.path.exists(model_path):
    print(f"[ERROR] No se encontro el archivo '{model_path}'.")
    print("Asegurate de correr 'Entrenar_MediaPipe.py' primero.")
    exit()

with open(model_path, "rb") as f:
    clf = pickle.load(f)

print("[INFO] Modelo Random Forest cargado con exito.")

# ==========================================
# 2. INICIALIZACIÓN DE MEDIAPIPE Y CÁMARA
# ==========================================
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Configurado para detectar hasta 2 manos simultáneas
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)

# Buffer de 7 frames para estabilizar el texto en pantalla
buffer_predicciones = deque(maxlen=7)
UMBRAL_PROBABILIDAD = 0.55

print("\n=== TRADUCTOR LSM CON MEDIAPIPE EN VIVO ===")
print("- Muestra 1 o 2 manos frente a la camara.")
print("- Presiona 'q' o ESC para salir.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Efecto espejo horizontal
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Conversión BGR a RGB para MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(rgb_frame)

    letra_predicha = "..."
    confianza = 0.0

    if resultados.multi_hand_landmarks:
        # Dibujar landmarks y conexiones de todas las manos detectadas
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        # Extraer vector de características de 126 valores
        vector_126 = []

        # Procesar hasta 2 manos
        for i in range(2):
            if i < len(resultados.multi_hand_landmarks):
                hand_lms = resultados.multi_hand_landmarks[i]
                
                # Coordenadas base de la muñeca (Landmark 0)
                base_x = hand_lms.landmark[0].x
                base_y = hand_lms.landmark[0].y
                base_z = hand_lms.landmark[0].z

                # 21 puntos normalizados respecto a la muñeca
                mano_features = []
                for lm in hand_lms.landmark:
                    mano_features.append(lm.x - base_x)
                    mano_features.append(lm.y - base_y)
                    mano_features.append(lm.z - base_z)

                vector_126.extend(mano_features)
            else:
                # Si no hay segunda mano, rellenar con 63 ceros
                vector_126.extend([0.0] * 63)

        # Inferencia con Random Forest
        vector_entrada = np.array(vector_126, dtype=np.float32).reshape(1, -1)
        probabilidades = clf.predict_proba(vector_entrada)[0]
        max_idx = np.argmax(probabilidades)
        confianza = float(probabilidades[max_idx])

        if confianza >= UMBRAL_PROBABILIDAD:
            buffer_predicciones.append(clf.classes_[max_idx])
        else:
            buffer_predicciones.append("...")
    else:
        buffer_predicciones.append("...")

    # Suavizado por moda en el buffer
    if len(buffer_predicciones) > 0:
        conteo = Counter(buffer_predicciones)
        letra_estable, votos = conteo.most_common(1)[0]
    else:
        letra_estable = "..."

    # ==========================================
    # 3. INTERFAZ GRÁFICA EN PANTALLA
    # ==========================================
    cv2.rectangle(frame, (15, 15), (320, 115), (25, 25, 25), -1)

    if letra_estable != "..." and votos >= 4:
        texto_letra = f"Letra: {letra_estable}"
        color_estado = (0, 255, 0)
    else:
        texto_letra = "Detectando..."
        color_estado = (0, 165, 255)

    cv2.putText(frame, texto_letra, (30, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_estado, 3)
    cv2.putText(frame, f"Confianza: {confianza * 100:.1f}%", (30, 95),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

    # Barra dinámica de certeza
    ancho_barra = int(270 * min(confianza, 1.0))
    cv2.rectangle(frame, (25, 105), (25 + ancho_barra, 110), color_estado, -1)

    cv2.imshow("Traductor LSM - MediaPipe (126 Puntos)", frame)

    tecla = cv2.waitKey(1) & 0xFF
    if tecla in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()