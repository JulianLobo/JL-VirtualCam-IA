import cv2
import numpy as np
import pyvirtualcam
import time
from ultralytics import YOLO

# Índice de tu cámara
CAMARA_INDEX = 1

# Cargar el modelo de segmentación de YOLOv8
model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(CAMARA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = 30

# Variable inicial para el desenfoque (debe ser IMPAR)
blur_level = 55  

print("Transmitiendo desenfoque filtrado a la Cámara Virtual...")
print("-----------------------------------------------------")
print("CONTROLES EN LA VENTANA DE VISTA PREVIA:")
print("  [ + ] / [ = ] : Aumentar desenfoque")
print("  [ - ]         : Disminuir desenfoque")
print("  [ Q ]         : Salir del programa")
print("-----------------------------------------------------")

with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue

        # Aplicar desenfoque dinámico usando blur_level
        blurred_frame = cv2.GaussianBlur(frame, (blur_level, blur_level), 0)

        # Inferencia detectando personas (clase 0) optimizada a 320px
        results = model(frame, classes=[0], imgsz=320, verbose=False)

        mask_3d = np.zeros((height, width, 3), dtype=np.float32)

        if results and len(results[0]) > 0 and results[0].masks is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            masks = results[0].masks.data.cpu().numpy()
            
            # FILTRO: Seleccionar la persona con el área más grande
            areas = [w * h for x, y, w, h in boxes]
            max_idx = np.argmax(areas)
            
            best_mask = cv2.resize(masks[max_idx], (width, height))
            best_mask = cv2.GaussianBlur(best_mask, (15, 15), 0)
            mask_3d = np.repeat(best_mask[:, :, np.newaxis], 3, axis=2)

        final_output = (frame * mask_3d + blurred_frame * (1.0 - mask_3d)).astype(np.uint8)

        # Dibujar el nivel actual en la pantalla para guiarnos
        cv2.putText(final_output, f"Desenfoque: {blur_level}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cam.send(final_output)

        # Mostrar ventana interactiva
        cv2.imshow("JL-VirtualCam-IA (Controles)", final_output)

        # Capturar teclas presionadas
        key = cv2.waitKey(1) & 0xFF
        
        # Subir desenfoque con '+' o '=' (Aumenta de 10 en 10, máx 151)
        if key == ord('+') or key == ord('='):
            blur_level = min(151, blur_level + 10)
            
        # Bajar desenfoque con '-' (Baja de 10 en 10, mín 3)
        elif key == ord('-'):
            blur_level = max(3, blur_level - 10)
            
        # Salir presionando 'q'
        elif key == ord('q'):
            break

        elapsed_time = time.time() - start_time
        sleep_time = max(0, (1.0 / fps) - elapsed_time)
        time.sleep(sleep_time)

cap.release()
cv2.destroyAllWindows()