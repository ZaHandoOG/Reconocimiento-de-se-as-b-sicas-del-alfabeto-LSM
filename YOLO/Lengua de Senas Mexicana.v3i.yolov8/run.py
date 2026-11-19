import cv2
import numpy as np
import os
import pyttsx3
import threading
import time
from collections import deque, Counter
from ultralytics import YOLO

# ==========================================
# 0. INICIALIZACIÓN DE ARCHIVO Y MOTOR DE VOZ
# ==========================================
ruta_archivo = "traduccion.txt"
esta_hablando = False

def hablar(texto):
    global esta_hablando
    try:
        motor = pyttsx3.init()
        motor.setProperty('rate', 140)  # Velocidad ajustada para que suene más natural
        
        # Asignar directamente el ID de Sabina
        id_voz_espanol = r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_ES-MX_SABINA_11.0"
        motor.setProperty('voice', id_voz_espanol)
                
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
# 1. CARGA DEL MODELO YOLO
# ==========================================
# Entrenado en roboflox
model_path = r"C:\Users\gaboo\Downloads\weights.pt" 
# Entrenado en mi pc
#model_path = r"C:\Users\gaboo\OneDrive\Documentos\Coding\Vision\MiniProyecto1\YOLO\Lengua de Senas Mexicana.v3i.yolov8\modelo_lsm_v1-2\weights\best.pt"    

if not os.path.exists(model_path):
    print(f"[ERROR] No se encontró el modelo en '{model_path}'.")
    exit()

print("[INFO] Cargando modelo YOLO...")
modelo = YOLO(model_path)
print("[INFO] Modelo YOLO cargado con éxito.")

# ==========================================
# 2. INICIALIZACIÓN DE CÁMARA Y VARIABLES
# ==========================================
cap = cv2.VideoCapture(0)
buffer_predicciones = deque(maxlen=7)
UMBRAL_CONFIANZA = 0.55

# --- VARIABLES PARA AUTO-CAPTURA ---
tiempo_requerido = 0.5  # Segundos que debes mantener la seña
tiempo_inicio_letra = 0
letra_actual_tracking = "..."
letra_ya_capturada = False

print("\n=== TRADUCTOR LSM YOLO CON AUTO-CAPTURA ===")
print("- Mantén la seña por 2 segundos para escribirla automáticamente.")
print("- Presiona 'espacio' para añadir un ESPACIO entre palabras.")
print("- Presiona 'r' para REPRODUCIR el texto.")
print("- Presiona 'l' para LIMPIAR el texto.\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Efecto espejo
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Inferencia de YOLO
    resultados = modelo.predict(frame, verbose=False, conf=UMBRAL_CONFIANZA)
    
    confianza = 0.0
    nombre_letra = "..."
    caja_coords = None

    # Si YOLO detecta al menos un objeto (mano)
    if len(resultados[0].boxes) > 0:
        # Tomar la primera caja detectada (la de mayor confianza)
        caja = resultados[0].boxes[0]
        confianza = float(caja.conf[0])
        clase_id = int(caja.cls[0])
        nombre_letra = modelo.names[clase_id]
        
        # Coordenadas [x1, y1, x2, y2]
        caja_coords = list(map(int, caja.xyxy[0]))
        buffer_predicciones.append(nombre_letra)
    else:
        buffer_predicciones.append("...")

    # Filtro estabilizador
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
                    letra_ya_capturada = True 
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
    
    # Determinar el color base según el estado de la captura
    if letra_estable != "..." and votos >= 4:
        texto_letra = f"Letra: {letra_estable}"
        color_estado = (0, 255, 0) if letra_ya_capturada else (255, 255, 0) # Verde o Cyan
    else:
        texto_letra = "Detectando..."
        color_estado = (0, 165, 255) # Naranja

    # Dibujar Bounding Box de YOLO si hay detección
    if caja_coords is not None:
        x1, y1, x2, y2 = caja_coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), color_estado, 2)
        cv2.putText(frame, f"{nombre_letra} {confianza*100:.0f}%", (x1, max(20, y1 - 10)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color_estado, 2)

    # Panel superior de información
    cv2.rectangle(frame, (15, 15), (320, 135), (25, 25, 25), -1)
    cv2.putText(frame, texto_letra, (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, color_estado, 3)
    cv2.putText(frame, f"Confianza: {confianza * 100:.1f}%", (30, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (220, 220, 220), 1)

    # Barra dinámica de certeza
    ancho_barra = int(270 * min(confianza, 1.0))
    cv2.rectangle(frame, (25, 105), (25 + ancho_barra, 110), color_estado, -1)

    # Barra de progreso para Auto-Captura (Amarilla)
    if progreso > 0 and not letra_ya_capturada:
        ancho_progreso = int(270 * progreso)
        cv2.rectangle(frame, (25, 120), (25 + ancho_progreso, 125), (0, 255, 255), -1)

    # Mostrar texto en vivo abajo (Panel inferior)
    texto_acumulado = obtener_texto_actual()
    cv2.rectangle(frame, (0, h - 60), (w, h), (0, 0, 0), -1)
    cv2.putText(frame, f"Texto: {texto_acumulado}", (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    cv2.putText(frame, "ESPACIO:Espacio | R:Leer | L:Limpiar | Q:Salir", (20, h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    cv2.imshow("Traductor LSM YOLO - Auto-Captura", frame)

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