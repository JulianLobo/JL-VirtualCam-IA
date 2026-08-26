import cv2
import numpy as np
import pyvirtualcam
import time
from ultralytics import YOLO

CAMARA_INDEX = 1

# Cargar modelo de segmentación de YOLOv8
model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(CAMARA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = 30

print("Transmitiendo desenfoque filtrado a la Cámara Virtual...")
print("Control + C para detener Cámara Virtual...")

with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    while cap.isOpened():
        start_time = time.time()
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue

        blurred_frame = cv2.GaussianBlur(frame, (55, 55), 0)

        # Inferencia detectando personas (clase 0)
        results = model(frame, classes=[0], imgsz=320, verbose=False)

        mask_3d = np.zeros((height, width, 3), dtype=np.float32)

        if results and len(results[0]) > 0 and results[0].masks is not None:
            boxes = results[0].boxes.xywh.cpu().numpy()
            masks = results[0].masks.data.cpu().numpy()
            
            # FILTRO: Seleccionar únicamente la persona con el área de caja más grande (tú en primer plano)
            areas = [w * h for x, y, w, h in boxes]
            max_idx = np.argmax(areas)
            
            # Usar solo la máscara del sujeto principal
            best_mask = cv2.resize(masks[max_idx], (width, height))
            best_mask = cv2.GaussianBlur(best_mask, (15, 15), 0)
            mask_3d = np.repeat(best_mask[:, :, np.newaxis], 3, axis=2)

        final_output = (frame * mask_3d + blurred_frame * (1.0 - mask_3d)).astype(np.uint8)

        cam.send(final_output)

        elapsed_time = time.time() - start_time
        sleep_time = max(0, (1.0 / fps) - elapsed_time)
        time.sleep(sleep_time)

cap.release()