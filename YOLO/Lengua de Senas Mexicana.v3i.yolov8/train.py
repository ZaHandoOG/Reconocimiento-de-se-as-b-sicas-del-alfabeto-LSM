from ultralytics import YOLO

# 1. Cargar el modelo base pre-entrenado 
# ('n' es la versión nano, ideal para pruebas rápidas y PCs estándar)
model = YOLO("yolov8n.pt") 

# 2. Iniciar el entrenamiento
if __name__ == '__main__':
    results = model.train(
        # Pon la ruta ABSOLUTA o RELATIVA correcta hacia tu archivo data.yaml
        data="C:\\Users\\gaboo\\OneDrive\\Documentos\\Coding\\Vision\\MiniProyecto1\\YOLO\\Lengua de Senas Mexicana.v3i.yolov8\\data.yaml", 
        
        # Parámetros de entrenamiento
        epochs=50,       # Número de vueltas completas al dataset (ajústalo según veas resultados)
        imgsz=640,       # Tamaño al que YOLO redimensionará las imágenes (640 es el estándar)
        batch=16,        # Imágenes procesadas a la vez (bájalo a 8 o 4 si Visual Studio te marca error de memoria RAM/VRAM)
        
        # Usa '0' si tienes tarjeta gráfica NVIDIA (CUDA) configurada, si no, cambia a 'cpu'
        device=0,        
        
        # Nombre de la carpeta donde se guardarán tus pesos finales (best.pt)
        name="C:\\Users\\gaboo\\OneDrive\\Documentos\\Coding\\Vision\\MiniProyecto1\\YOLO\\Lengua de Senas Mexicana.v3i.yolov8\\modelo_lsm_v1" 
    )