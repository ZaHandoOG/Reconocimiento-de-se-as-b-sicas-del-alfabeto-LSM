import cv2
import mediapipe as mp
import numpy as np
import pickle
import os
import pyttsx3
import threading
import time
from collections import deque, Counter

# ==========================================
# 0. INICIALIZACIÓN DE ARCHIVO Y MOTOR DE VOZ
# ==========================================
ruta_archivo = "traduccion.txt"
esta_hablando = False

def hablar(texto):
    global esta_hablando
    try:
        motor = pyttsx3.init()
        motor.setProperty('rate', 150)
        
        # Buscador de voz en español mejorado
        voces = motor.getProperty('voices')
        for voz in voces:
            nombre = voz.name.lower()
            id_voz = voz.id.lower()
            # Busca etiquetas o nombres comunes de voces en español de Windows
            if 'spanish' in nombre or 'es' in id_voz or 'sabina' in nombre or 'helena' in nombre or 'mexico' in nombre:
                motor.setProperty('voice', voz.id)
                break
                
        motor.say(texto)
        motor.runAndWait()
    except Exception as e:
        print(f"[ERROR DE VOZ] {e}")
    finally:
        esta_hablando = False 

def leer_texto_en_voz_alta():
    global esta_hablando
    if esta_hablando:
        print("[VOZ] Ya estoy hablando, por favor espera...")
        return
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as archivo:
            texto = archivo.read()
            if texto.strip():
                print(f"[VOZ] Leyendo: {texto}")
                esta_hablando = True
                hilo_voz = threading.Thread(target=hablar, args=(texto,))
                hilo_voz.start()
            else:
                print("[VOZ] El archivo está vacío.")
    except FileNotFoundError:
        print("[VOZ] El archivo aún no existe.")

def guardar_caracter(caracter):
    with open(ruta_archivo, "a", encoding="utf-8") as archivo:
        archivo.write(caracter)

def limpiar_archivo():
    with open(ruta_archivo, "w", encoding="utf-8") as archivo:
        archivo.write("")
    print("[INFO] Archivo de texto limpiado.")

def obtener_texto_actual():
    try:
        with open(ruta_archivo, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""

limpiar_archivo()

# ==========================================
# 1. CARGA DEL MODELO RANDOM FOREST
# ==========================================
model_path = r"MiniProyecto1\Modelo MediaPipe\modelo_lsm_mediapipe.pkl"

if not os.path.exists(model_path):
    print(f"[ERROR] No se encontro el archivo '{model_path}'.")
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

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6,
    min_tracking_confidence=0.5
)

cap = cv2.VideoCapture(0)
buffer_predicciones = deque(maxlen=7)
UMBRAL_PROBABILIDAD = 0.55

# --- VARIABLES PARA AUTO-CAPTURA ---
tiempo_requerido = 2.0  # Segundos que debes mantener la seña
tiempo_inicio_letra = 0
letra_actual_tracking = "..."
letra_ya_capturada = False

print("\n=== TRADUCTOR LSM CON AUTO-CAPTURA ===")
print("- Mantén la seña por 2 segundos para escribirla automáticamente.")
print("- Presiona 'espacio' para añadir un ESPACIO entre palabras.")
print("- Presiona 'r' para REPRODUCIR el texto.")
print("- Presiona 'l' para LIMPIAR el texto.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    resultados = hands.process(rgb_frame)

    confianza = 0.0

    if resultados.multi_hand_landmarks:
        for hand_landmarks in resultados.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

        vector_126 = []
        for i in range(2):
            if i < len(resultados.multi_hand_landmarks):
                hand_lms = resultados.multi_hand_landmarks[i]
                base_x = hand_lms.landmark[0].x
                base_y = hand_lms.landmark[0].y
                base_z = hand_lms.landmark[0].z

                mano_features = []
                for lm in hand_lms.landmark:
                    mano_features.append(lm.x - base_x)
                    mano_features.append(lm.y - base_y)
                    mano_features.append(lm.z - base_z)
                vector_126.extend(mano_features)
            else:
                vector_126.extend([0.0] * 63)

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

    if len(buffer_predicciones) > 0:
        conteo = Counter(buffer_predicciones)
        letra_estable, votos = conteo.most_common(1)[0]
    else:
        letra_estable = "..."

    # ==========================================
    # 3. LÓGICA DE AUTO-CAPTURA
    # ==========================================
    progreso = 0.0
    if letra_estable != "..." and votos >= 4:
        if letra_estable == letra_actual_tracking:
            if not letra_ya_capturada:
                tiempo_transcurrido = time.time() - tiempo_inicio_letra
                progreso = min(tiempo_transcurrido / tiempo_requerido, 1.0)
                
                # Si se cumple el tiempo, guardamos la letra
                if tiempo_transcurrido >= tiempo_requerido:
                    guardar_caracter(letra_estable)
                    print(f"[AUTO-CAPTURA] {letra_estable}")
                    letra_ya_capturada = True # Evita que se guarde 100 veces
        else:
            # Si cambiaste de letra, reiniciamos el temporizador
            letra_actual_tracking = letra_estable
            tiempo_inicio_letra = time.time()
            letra_ya_capturada = False
    else:
        letra_actual_tracking = "..."
        letra_ya_capturada = False

    # ==========================================
    # 4. INTERFAZ GRÁFICA EN PANTALLA
    # ==========================================
    cv2.rectangle(frame, (15, 15), (320, 135), (25, 25, 25), -1)

    if letra_estable != "..." and votos >= 4:
        texto_letra = f"Letra: {letra_estable}"
        color_estado = (0, 255, 0) if letra_ya_capturada else (255, 255, 0)
    else:
        texto_letra = "Detectando..."
        color_estado = (0, 165, 255)

    cv2.putText(frame, texto_letra, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_estado, 3)
    cv2.putText(frame, f"Confianza: {confianza * 100:.1f}%", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

    # Barra dinámica de certeza
    ancho_barra = int(270 * min(confianza, 1.0))
    cv2.rectangle(frame, (25, 105), (25 + ancho_barra, 110), color_estado, -1)

    # Barra de progreso para Auto-Captura (Amarilla)
    if progreso > 0 and not letra_ya_capturada:
        ancho_progreso = int(270 * progreso)
        cv2.rectangle(frame, (25, 120), (25 + ancho_progreso, 125), (0, 255, 255), -1)

    # Mostrar texto en vivo abajo
    texto_acumulado = obtener_texto_actual()
    cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Texto: {texto_acumulado}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    cv2.putText(frame, "ESPACIO:Espacio | R:Leer | L:Limpiar", (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Traductor LSM - Auto-Captura", frame)

    # ==========================================
    # 5. CONTROLES DE TECLADO
    # ==========================================
    tecla = cv2.waitKey(1) & 0xFF
    
    if tecla == ord(' '):  # Barra espaciadora
        guardar_caracter(" ")
        print("[ESPACIO AÑADIDO]")
    elif tecla == ord('r'):
        leer_texto_en_voz_alta()
    elif tecla == ord('l'):
        limpiar_archivo()
    elif tecla in (ord('q'), 27):
        break

cap.release()
cv2.destroyAllWindows()