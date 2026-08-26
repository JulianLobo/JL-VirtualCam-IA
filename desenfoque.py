import cv2
import numpy as np
import pyvirtualcam
import time
from ultralytics import YOLO

# Configuración de cámara
CAMARA_INDEX = 1

# Cargar modelo YOLOv8
model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(CAMARA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = 30

# Nivel de desenfoque inicial (Porcentaje de 0 a 100)
blur_percent = 50  
show_preview = True  # Estado de la vista previa

print("Transmitiendo desenfoque a la Cámara Virtual...")
print("-----------------------------------------------------")
print("CONTROLES EN LA VENTANA:")
print("  [ + ] / [ - ] : Aumentar desenfoque (+10%)")
print("  [ - ]         : Disminuir desenfoque (-10%)")
print("  [ Q ]         : Alternar (Ocultar / Mostrar) vista previa")
print("  [ Ctrl + C ]  : Detener script por completo desde la consola")
print("-----------------------------------------------------")

NOMBRE_VENTANA = "JL-VirtualCam-IA (Controles)"
cv2.namedWindow(NOMBRE_VENTANA, cv2.WINDOW_AUTOSIZE)

# Crear una imagen negra ligera para cuando la vista previa esté en modo "Ahorro"
black_frame = np.zeros((150, 400, 3), dtype=np.uint8)
cv2.putText(black_frame, "Modo Ahorro Activo", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
cv2.putText(black_frame, "Presiona 'Q' para mostrar", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue

        # Convertir el porcentaje (0 a 100) a un Kernel impar de OpenCV (1 a 151)
        kernel_size = int(1 + (blur_percent / 100.0) * 150)
        if kernel_size % 2 == 0:
            kernel_size += 1

        # Aplicar desenfoque
        if blur_percent == 0:
            blurred_frame = frame.copy()
        else:
            blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        # Inferencia con IA
        results = model(frame, classes=[0], imgsz=320, verbose=False)
        mask_3d = np.zeros((height, width, 3), dtype=np.float32)

        if results and len(results[0]) > 0 and results[0].masks is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            masks = results[0].masks.data.cpu().numpy()
            
            areas = [w * h for x, y, w, h in boxes]
            max_idx = np.argmax(areas)
            
            best_mask = cv2.resize(masks[max_idx], (width, height))
            best_mask = cv2.GaussianBlur(best_mask, (15, 15), 0)
            mask_3d = np.repeat(best_mask[:, :, np.newaxis], 3, axis=2)

        final_output = (frame * mask_3d + blurred_frame * (1.0 - mask_3d)).astype(np.uint8)

        # Enviar siempre el cuadro procesado a la Cámara Virtual de OBS
        cam.send(final_output)

        # GESTIÓN DE LA VENTANA Y TECLAS
        if show_preview:
            preview_frame = final_output.copy()
            cv2.putText(preview_frame, f"Desenfoque: {blur_percent}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(NOMBRE_VENTANA, preview_frame)
        else:
            # En modo ahorro muestra un recuadro pequeño negro sin procesar renderizado pesado
            cv2.imshow(NOMBRE_VENTANA, black_frame)

        key = cv2.waitKey(1) & 0xFF

        # Controles
        if key == ord('+') or key == ord('='):
            blur_percent = min(100, blur_percent + 10)
        elif key == ord('-'):
            blur_percent = max(0, blur_percent - 10)
        elif key == ord('q'):
            # Alternar estado al presionar Q
            show_preview = not show_preview

        # Sincronización de FPS
        elapsed_time = time.time() - start_time
        sleep_time = max(0, (1.0 / fps) - elapsed_time)
        time.sleep(sleep_time)

cap.release()
cv2.destroyAllWindows()