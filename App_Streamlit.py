import streamlit as st
import cv2
import numpy as np
import pickle
import os
import tensorflow as tf
from collections import deque, Counter

# ==========================================
# 1. CONFIGURACIÓN VISUAL DE STREAMLIT
# ==========================================
st.set_page_config(
    page_title="LSM Vision · Sistema de Reconocimiento",
    page_icon="🤟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        display: flex; align-items: center; gap: 14px;
        padding: 4px 0 18px 0;
        border-bottom: 1px solid #2a2e37;
        margin-bottom: 22px;
    }
    .main-header .emoji { font-size: 2.4rem; }
    .main-header h1 { margin: 0; font-size: 1.6rem; font-weight: 800; color: #ffffff; }
    .main-header p { margin: 0; color: #8b93a1; font-size: 0.92rem; }

    .metric-box {
        background: linear-gradient(145deg, #1c2029, #171a21);
        border-radius: 14px; padding: 18px 20px; margin-bottom: 14px;
        border: 1px solid #2a2e37; border-left: 5px solid #00d26a;
        box-shadow: 0 4px 14px rgba(0,0,0,0.25);
    }
    .metric-box.waiting { border-left-color: #ffb020; }
    .metric-title {
        font-size: 0.78rem; letter-spacing: 0.08em; text-transform: uppercase;
        color: #8b93a1; margin-bottom: 4px;
    }
    .metric-val { font-size: 3rem; font-weight: 800; color: #ffffff; line-height: 1.1; }

    .badge {
        display: inline-block; padding: 3px 10px; border-radius: 20px;
        font-size: 0.75rem; font-weight: 600; margin-right: 6px;
    }
    .badge-ok { background: rgba(0,210,106,0.15); color: #00d26a; }
    .badge-warn { background: rgba(255,176,32,0.15); color: #ffb020; }
    .badge-off { background: rgba(255,80,80,0.15); color: #ff5050; }

    .top-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 8px 12px; background: #1c2029; border-radius: 8px;
        margin-bottom: 6px; border: 1px solid #262a33;
    }
    .top-row .letra { font-weight: 700; color: #fff; font-size: 1.05rem; }
    .top-row .pct { color: #8b93a1; font-size: 0.88rem; }

    .info-card {
        background: #1c2029; border-radius: 10px; padding: 12px 16px;
        border: 1px solid #262a33; color: #c3c9d3; font-size: 0.9rem; margin-bottom: 14px;
    }

    div[data-testid="stCameraInput"] > label,
    div[data-testid="stFileUploader"] > label { font-weight: 600; color: #c3c9d3; }
</style>
""", unsafe_allow_html=True)

# Mapeo oficial Sign Language MNIST (25 clases posibles, del 0 al 24; 'J' no está en el dataset)
MAPEO_MNIST = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I',
    9: '[J N/A]', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P',
    16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y'
}

# ==========================================
# 2. RUTAS Y CARGA DE MODELOS
# ==========================================
PATH_CNN = r"C:\Users\gaboo\OneDrive\Documentos\Coding\models\best_sign_model.keras"
PATH_RF = r"C:\Users\gaboo\OneDrive\Documentos\Coding\Vision\MiniProyecto1\Modelo MediaPipe\modelo_lsm_mediapipe.pkl"


@st.cache_resource
def load_cnn_model():
    if os.path.exists(PATH_CNN):
        return tf.keras.models.load_model(PATH_CNN)
    return None


@st.cache_resource
def load_rf_model():
    if os.path.exists(PATH_RF):
        with open(PATH_RF, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_resource
def get_mediapipe():
    """Devuelve los módulos de MediaPipe junto con dos detectores:
    uno para video (tracking continuo) y otro para imágenes fijas."""
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands_video = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.5
    )
    hands_static = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    return mp_hands, mp_draw, mp_styles, hands_video, hands_static


cnn_model = load_cnn_model()
rf_model = load_rf_model()

# ==========================================
# 3. FUNCIONES DE PREDICCIÓN (compartidas entre modos)
# ==========================================

def predecir_cnn(frame, umbral, roi_size=220):
    """Recorta el centro del frame, lo pasa por la CNN 28x28 y devuelve
    un diccionario con letra, confianza, top-3 y el frame anotado."""
    h, w = frame.shape[:2]
    x1 = max(0, (w - roi_size) // 2)
    y1 = max(0, (h - roi_size) // 2)
    x2 = min(w, x1 + roi_size)
    y2 = min(h, y1 + roi_size)

    roi = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
    norm = resized.astype("float32") / 255.0
    tensor_in = norm.reshape(1, 28, 28, 1)

    preds = cnn_model.predict(tensor_in, verbose=0)[0]
    max_idx = int(np.argmax(preds))
    conf = float(preds[max_idx])
    letra = MAPEO_MNIST.get(max_idx, "?") if conf >= umbral else None

    top_idx = np.argsort(preds)[-3:][::-1]
    top3 = [(MAPEO_MNIST.get(int(i), "?"), float(preds[i])) for i in top_idx]

    frame_out = frame.copy()
    color = (0, 255, 100) if conf >= umbral else (0, 180, 255)
    cv2.rectangle(frame_out, (x1, y1), (x2, y2), color, 2)
    cv2.putText(frame_out, "Coloca tu mano aqui", (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return {
        "letra": letra, "conf": conf, "top3": top3,
        "frame": frame_out, "detalle": "Región central 28x28 px",
        "roi_preview": resized
    }


def predecir_rf(frame, hands_obj, mp_hands, mp_draw, mp_styles, umbral):
    """Extrae 126 landmarks (dos manos) con MediaPipe y predice con el
    Random Forest. Devuelve letra, confianza, top-3 y el frame anotado."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands_obj.process(rgb)
    frame_out = frame.copy()

    letra, conf, detalle, top3 = None, 0.0, "No se detectaron manos", []

    if res.multi_hand_landmarks and res.multi_handedness:
        vector_izq = [0.0] * 63
        vector_der = [0.0] * 63
        lados = []

        for lms, handedness in zip(res.multi_hand_landmarks, res.multi_handedness):
            mp_draw.draw_landmarks(
                frame_out, lms, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )
            bx, by, bz = lms.landmark[0].x, lms.landmark[0].y, lms.landmark[0].z
            features = []
            for lm in lms.landmark:
                features.extend([lm.x - bx, lm.y - by, lm.z - bz])

            if handedness.classification[0].label == 'Right':
                vector_der = features
                lados.append("Derecha")
            else:
                vector_izq = features
                lados.append("Izquierda")

        vec_126 = np.array([vector_izq + vector_der], dtype=np.float32)
        prob = rf_model.predict_proba(vec_126)[0]
        max_idx = int(np.argmax(prob))
        conf = float(prob[max_idx])
        letra = rf_model.classes_[max_idx] if conf >= umbral else None
        detalle = "Mano(s) detectada(s): " + ", ".join(lados)

        top_idx = np.argsort(prob)[-3:][::-1]
        top3 = [(rf_model.classes_[i], float(prob[i])) for i in top_idx]

    return {"letra": letra, "conf": conf, "top3": top3, "frame": frame_out, "detalle": detalle}


def ejecutar_prediccion(frame, modelo_seleccionado, umbral, mp_objs):
    if "CNN" in modelo_seleccionado:
        return predecir_cnn(frame, umbral)
    mp_hands, mp_draw, mp_styles, hands_video, hands_static = mp_objs
    hands_obj = hands_video if st.session_state.get("modo_actual") == "live" else hands_static
    return predecir_rf(frame, hands_obj, mp_hands, mp_draw, mp_styles, umbral)


# ==========================================
# 4. RENDER DEL PANEL DE MÉTRICAS
# ==========================================

def render_metricas(resultado, placeholders):
    letra = resultado["letra"]
    conf = resultado["conf"]
    top3 = resultado["top3"]
    detalle = resultado["detalle"]

    box_class = "metric-box" if letra else "metric-box waiting"
    display_letra = letra if letra else "..."
    placeholders["letra"].markdown(f"""
    <div class="{box_class}">
        <div class="metric-title">Letra identificada</div>
        <div class="metric-val">{display_letra}</div>
    </div>
    """, unsafe_allow_html=True)

    placeholders["conf"].progress(min(conf, 1.0), text=f"Certeza del modelo: {conf * 100:.1f}%")

    if top3:
        filas = ""
        for l, p in top3:
            filas += f"""<div class="top-row"><span class="letra">{l}</span>
            <span class="pct">{p * 100:.1f}%</span></div>"""
        placeholders["top3"].markdown(filas, unsafe_allow_html=True)
    else:
        placeholders["top3"].empty()

    placeholders["detalle"].markdown(f'<div class="info-card">{detalle}</div>', unsafe_allow_html=True)


# ==========================================
# 5. BARRA LATERAL
# ==========================================
with st.sidebar:
    st.header("⚙️ Panel de Control")
    modelo_seleccionado = st.selectbox(
        "Modelo a evaluar:",
        ["MediaPipe + Random Forest (126 Features)", "CNN (Sign Language MNIST 28x28)"]
    )

    st.markdown("---")
    st.subheader("Estado de modelos")
    cnn_badge = '<span class="badge badge-ok">✅ Listo</span>' if cnn_model is not None else '<span class="badge badge-off">❌ No encontrado</span>'
    rf_badge = '<span class="badge badge-ok">✅ Listo</span>' if rf_model is not None else '<span class="badge badge-off">❌ No encontrado</span>'
    st.markdown(f"CNN Keras: {cnn_badge}", unsafe_allow_html=True)
    st.markdown(f"Random Forest: {rf_badge}", unsafe_allow_html=True)

    st.markdown("---")
    umbral_conf = st.slider("Umbral de aceptación (%)", min_value=30, max_value=90, value=50, step=5) / 100.0
    st.caption("Filtra predicciones con certeza menor a este umbral para evitar falsos positivos.")

# ==========================================
# 6. ENCABEZADO
# ==========================================
st.markdown("""
<div class="main-header">
    <div class="emoji">🤟</div>
    <div>
        <h1>Traductor de Lengua de Señas (LSM)</h1>
        <p>Plataforma interactiva para desplegar y comparar modelos en vivo, por foto o por imagen subida.</p>
    </div>
</div>
""", unsafe_allow_html=True)

modo = st.radio(
    "Modo de entrada",
    ["🎥 Cámara en vivo", "📸 Tomar foto", "🖼️ Subir imagen"],
    horizontal=True,
    label_visibility="collapsed"
)

col_visual, col_metrics = st.columns([2, 1])

with col_metrics:
    st.subheader("Panel de predicción")
    placeholders = {
        "letra": st.empty(),
        "conf": st.empty(),
        "top3": st.empty(),
        "detalle": st.empty(),
    }

modelo_listo = (cnn_model is not None) if "CNN" in modelo_seleccionado else (rf_model is not None)
if not modelo_listo:
    st.error("El modelo seleccionado no está disponible en la ruta especificada. Revisa PATH_CNN / PATH_RF.")
    st.stop()

mp_objs = get_mediapipe() if "MediaPipe" in modelo_seleccionado else None

# ==========================================
# 7A. MODO: CÁMARA EN VIVO
# ==========================================
if modo == "🎥 Cámara en vivo":
    st.session_state["modo_actual"] = "live"
    with col_visual:
        st.subheader("Transmisión de video")
        activo = st.toggle("Encender cámara web", value=False)
        frame_window = st.empty()

    if activo:
        cap = cv2.VideoCapture(0)
        historial = deque(maxlen=7)

        while activo and cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.error("No se pudo acceder al dispositivo de captura.")
                break

            frame = cv2.flip(frame, 1)
            resultado = ejecutar_prediccion(frame, modelo_seleccionado, umbral_conf, mp_objs)
            historial.append(resultado["letra"] if resultado["letra"] else "...")

            moda_letra, votos = Counter(historial).most_common(1)[0]
            resultado_estable = dict(resultado)
            resultado_estable["letra"] = moda_letra if (moda_letra != "..." and votos >= 4) else None

            render_metricas(resultado_estable, placeholders)
            frame_window.image(cv2.cvtColor(resultado["frame"], cv2.COLOR_BGR2RGB),
                                channels="RGB", use_container_width=True)

        cap.release()
    else:
        with col_visual:
            st.info("Activa el interruptor para iniciar la transmisión en vivo.")

# ==========================================
# 7B. MODO: TOMAR FOTO
# ==========================================
elif modo == "📸 Tomar foto":
    st.session_state["modo_actual"] = "static"
    with col_visual:
        st.subheader("Captura una foto")
        foto = st.camera_input("Coloca tu mano frente a la cámara y captura")

    if foto is not None:
        file_bytes = np.asarray(bytearray(foto.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        resultado = ejecutar_prediccion(frame, modelo_seleccionado, umbral_conf, mp_objs)
        render_metricas(resultado, placeholders)
        with col_visual:
            st.image(cv2.cvtColor(resultado["frame"], cv2.COLOR_BGR2RGB),
                      caption="Resultado anotado", use_container_width=True)
    else:
        with col_visual:
            st.info("Toma una foto para ver la predicción.")

# ==========================================
# 7C. MODO: SUBIR IMAGEN
# ==========================================
else:
    st.session_state["modo_actual"] = "static"
    with col_visual:
        st.subheader("Sube una imagen")
        archivo = st.file_uploader("Formatos aceptados: JPG, JPEG, PNG", type=["jpg", "jpeg", "png"])

    if archivo is not None:
        file_bytes = np.asarray(bytearray(archivo.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        resultado = ejecutar_prediccion(frame, modelo_seleccionado, umbral_conf, mp_objs)
        render_metricas(resultado, placeholders)
        with col_visual:
            st.image(cv2.cvtColor(resultado["frame"], cv2.COLOR_BGR2RGB),
                      caption="Resultado anotado", use_container_width=True)
    else:
        with col_visual:
            st.info("Sube una imagen para ver la predicción.")