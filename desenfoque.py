import cv2
import numpy as np
import pyvirtualcam
import time
import keyboard  # Captura teclas globales
from ultralytics import YOLO

CAMARA_INDEX = 1
model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(CAMARA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = 30

blur_percent = 50  
show_preview = True  

print("Transmitiendo desenfoque a la Cámara Virtual...")
print("-----------------------------------------------------")
print("CONTROLES GLOBALES (Funcionan en cualquier ventana):")
print("  [ + ] : Aumentar desenfoque (+10%)")
print("  [ - ] : Disminuir desenfoque (-10%)")
print("  [ Q ] : Abrir / Cerrar ventana de vista previa")
print("-----------------------------------------------------")

NOMBRE_VENTANA = "JL-VirtualCam-IA (Controles)"

# Variables para evitar repetición rápida al mantener presionada la tecla
last_key_time = 0

with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue

        # Convertir porcentaje a tamaño de kernel impar
        kernel_size = int(1 + (blur_percent / 100.0) * 150)
        if kernel_size % 2 == 0:
            kernel_size += 1

        if blur_percent == 0:
            blurred_frame = frame.copy()
        else:
            blurred_frame = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)

        # IA Inferencia
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

        cam.send(final_output)

        # DETECCIÓN DE TECLAS GLOBALES (Antirrebote de 0.2 segundos)
        current_time = time.time()
        if current_time - last_key_time > 0.2:
            if keyboard.is_pressed('+') or keyboard.is_pressed('='):
                blur_percent = min(100, blur_percent + 10)
                print(f"Desenfoque: {blur_percent}%")
                last_key_time = current_time
            elif keyboard.is_pressed('-'):
                blur_percent = max(0, blur_percent - 10)
                print(f"Desenfoque: {blur_percent}%")
                last_key_time = current_time
            elif keyboard.is_pressed('q'):
                show_preview = not show_preview
                if not show_preview:
                    cv2.destroyAllWindows()
                last_key_time = current_time

        # GESTIÓN DE VENTANA
        if show_preview:
            preview_frame = final_output.copy()
            cv2.putText(preview_frame, f"Desenfoque: {blur_percent}%", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow(NOMBRE_VENTANA, preview_frame)
            cv2.waitKey(1)

        elapsed_time = time.time() - start_time
        sleep_time = max(0, (1.0 / fps) - elapsed_time)
        time.sleep(sleep_time)

cap.release()
cv2.destroyAllWindows()