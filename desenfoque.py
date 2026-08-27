import os
import cv2
import numpy as np
import pyvirtualcam
import time
import keyboard
import torch
from ultralytics import YOLO
from pygrabber.dshow_graph import FilterGraph

def obtener_camaras_con_nombres():
    """Obtiene los nombres reales de las cámaras disponibles en Windows."""
    try:
        graph = FilterGraph()
        return graph.get_input_devices()
    except Exception:
        return []

def seleccionar_camara():
    """Muestra un menú claro con el nombre real de cada cámara."""
    nombres_camaras = obtener_camaras_con_nombres()
    
    if not nombres_camaras:
        print("Buscando cámaras conectadas al sistema...")
        camaras_validas = []
        for index in range(5):
            cap_test = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap_test.isOpened():
                ret, _ = cap_test.read()
                if ret:
                    camaras_validas.append((index, f"Cámara en índice {index}"))
                cap_test.release()
        
        if not camaras_validas:
            print("❌ Error: No se detectó ninguna cámara conectada.")
            input("Presiona Enter para salir...")
            exit()
        return camaras_validas[0][0], camaras_validas[0][1]

    if len(nombres_camaras) == 1:
        print(f"✔️ Cámara detectada: {nombres_camaras[0]}. Seleccionada automáticamente.\n")
        time.sleep(1)
        return 0, nombres_camaras[0]

    print("\n=====================================================")
    print("        CÁMARAS DETECTADAS EN EL SISTEMA             ")
    print("=====================================================")
    for idx, nombre in enumerate(nombres_camaras):
        print(f"  [ {idx} ] -> {nombre}")
    print("=====================================================")
    
    while True:
        try:
            opcion = int(input("Selecciona el número de la cámara que deseas usar: "))
            if 0 <= opcion < len(nombres_camaras):
                return opcion, nombres_camaras[opcion]
            else:
                print("Opción no válida. Elige un número de la lista.")
        except ValueError:
            print("Por favor, ingresa un número válido.")

# Selección de cámara
CAMARA_INDEX, NOMBRE_CAMARA = seleccionar_camara()

# Configuración de hardware y Cargar modelo YOLOv8
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = YOLO("yolov8n-seg.pt").to(device)

cap = cv2.VideoCapture(CAMARA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = 30

blur_percent = 50  
show_preview = True  
last_key_time = 0

def mostrar_interfaz_consola(nivel, vista_previa, nombre_camara):
    """Limpia la terminal y muestra un panel interactivo ordenado."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    bloques = int(nivel / 10)
    barra = "█" * bloques + "░" * (10 - bloques)
    estado_vista = "ACTIVADA" if vista_previa else "OCULTA (Ahorro de recursos)"
    
    print("=====================================================")
    print("        JL-VirtualCam-IA | PANEL DE CONTROL          ")
    print("=====================================================")
    print(f" CÁMARA ACTIVA:      {nombre_camara}")
    print(f" DISPOSITIVO IA:     {device.upper()}")
    print(f" NIVEL DE DESENFOQUE: [{barra}] {nivel}%")
    print(f" VISTA PREVIA:       {estado_vista}")
    print("-----------------------------------------------------")
    print(" CONTROLES GLOBALES:")
    print("   [ + ] / [ - ] : Aumentar desenfoque (+10%)")
    print("   [ - ]         : Disminuir desenfoque (-10%)")
    print("   [ Q ]         : Mostrar / Ocultar Vista Previa")
    print("   [ Ctrl + C ]  : Detener programa")
    print("=====================================================")

def aplicar_diseno_emergente(frame, nivel_blur, nombre_cam, fps_real):
    """Aplica una capa de interfaz gráfica estilizada (HUD) sobre el video."""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # 1. Borde exterior
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (235, 206, 135), 2)

    # 2. Header Superior Oscuro
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
    
    # 3. Tarjeta Inferior de Estado
    cv2.rectangle(overlay, (10, h - 65), (w - 10, h - 10), (25, 25, 25), -1)

    # Aplicar transparencia (Blending)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # 4. Texto Header: Título + Estado Rec
    cv2.putText(frame, "JL-VirtualCam IA", (15, 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.circle(frame, (w - 100, 20), 5, (0, 0, 255), -1)
    cv2.putText(frame, "EN VIVO", (w - 88, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # 5. Texto Tarjeta Inferior
    cv2.putText(frame, f"Blur: {nivel_blur}%", (25, h - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 127), 2, cv2.LINE_AA)
    
    cv2.putText(frame, f"FPS: {fps_real}", (160, h - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cam_corta = nombre_cam if len(nombre_cam) < 35 else nombre_cam[:32] + "..."
    cv2.putText(frame, f"Camara: {cam_corta}", (25, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return frame

mostrar_interfaz_consola(blur_percent, show_preview, NOMBRE_CAMARA)
NOMBRE_VENTANA = "JL-VirtualCam-IA (Vista Previa)"

# Variables para medición de FPS y control de la máscara
fps_count = 0
fps_mostrar = 30
last_fps_time = time.time()

frame_count = 0
last_mask_3d = np.zeros((height, width, 3), dtype=np.float32)

try:
    with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue

            frame_count += 1

            # --- OPTIMIZACIÓN 1: DESENFOQUE LIGERO (DOWNSCALING) ---
            if blur_percent == 0:
                blurred_frame = frame.copy()
            else:
                small_w, small_h = width // 4, height // 4
                small_frame = cv2.resize(frame, (small_w, small_h), interpolation=cv2.INTER_NEAREST)
                
                ksize = int(1 + (blur_percent / 100.0) * 30)
                if ksize % 2 == 0:
                    ksize += 1
                
                blurred_small = cv2.GaussianBlur(small_frame, (ksize, ksize), 0)
                blurred_frame = cv2.resize(blurred_small, (width, height), interpolation=cv2.INTER_LINEAR)

            # --- OPTIMIZACIÓN 2: FRAME SKIPPING E INFERENCIA REDUCIDA ---
            # Ejecuta la IA solo 1 de cada 2 cuadros para duplicar la fluidez
            if frame_count % 2 == 0 or frame_count == 1:
                results = model(frame, classes=[0], imgsz=256, verbose=False)

                if results and len(results[0]) > 0 and results[0].masks is not None:
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    masks = results[0].masks.data.cpu().numpy()
                    areas = [w * h for x, y, w, h in boxes]
                    max_idx = np.argmax(areas)
                    
                    best_mask = cv2.resize(masks[max_idx], (width, height))
                    best_mask = cv2.GaussianBlur(best_mask, (15, 15), 0)
                    last_mask_3d = np.repeat(best_mask[:, :, np.newaxis], 3, axis=2)

            mask_3d = last_mask_3d
            final_output = (frame * mask_3d + blurred_frame * (1.0 - mask_3d)).astype(np.uint8)

            # Transmitir el cuadro limpio a la Cámara Virtual
            cam.send(final_output)

            # Medidor de FPS
            fps_count += 1
            if time.time() - last_fps_time >= 1.0:
                fps_mostrar = fps_count
                fps_count = 0
                last_fps_time = time.time()

            # CONTROL DE TECLAS
            current_time = time.time()
            if current_time - last_key_time > 0.2:
                hubo_cambio = False
                
                if keyboard.is_pressed('+') or keyboard.is_pressed('='):
                    if blur_percent < 100:
                        blur_percent += 10
                        hubo_cambio = True
                    last_key_time = current_time
                elif keyboard.is_pressed('-'):
                    if blur_percent > 0:
                        blur_percent -= 10
                        hubo_cambio = True
                    last_key_time = current_time
                elif keyboard.is_pressed('q'):
                    show_preview = not show_preview
                    if not show_preview:
                        cv2.destroyAllWindows()
                    hubo_cambio = True
                    last_key_time = current_time

                if hubo_cambio:
                    mostrar_interfaz_consola(blur_percent, show_preview, NOMBRE_CAMARA)

            # MOSTRAR VISTA PREVIA
            if show_preview:
                preview_frame = final_output.copy()
                preview_frame = aplicar_diseno_emergente(preview_frame, blur_percent, NOMBRE_CAMARA, fps_mostrar)
                cv2.imshow(NOMBRE_VENTANA, preview_frame)
                cv2.waitKey(1)

            # Control de FPS adaptativo (sin demoras innecesarias)
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1.0 / fps) - elapsed_time)
            if sleep_time > 0.005:
                time.sleep(sleep_time)

except KeyboardInterrupt:
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=====================================================")
    print("        Programa JL-VirtualCam-IA finalizado.          ")
    print("=====================================================")

finally:
    cap.release()
    cv2.destroyAllWindows()