import os
import cv2
import numpy as np
import pyvirtualcam
import time
import keyboard
from ultralytics import YOLO
from pygrabber.dshow_graph import FilterGraph

def obtener_camaras_con_nombres():
    """Obtiene los nombres reales de las cámaras disponibles en Windows."""
    try:
        graph = FilterGraph()
        dispositivos = graph.get_input_devices()
        return dispositivos
    except Exception:
        # Respaldo por si ocurre algún fallo con DirectShow
        return []

def seleccionar_camara():
    """Muestra un menú claro con el nombre real de cada cámara."""
    nombres_camaras = obtener_camaras_con_nombres()
    
    # Si por alguna razón no detectó nombres, probamos con OpenCV básico
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

    # Si solo hay 1 cámara detectada
    if len(nombres_camaras) == 1:
        print(f"✔️ Cámara detectada: {nombres_camaras[0]}. Seleccionada automáticamente.\n")
        time.sleep(1)
        return 0, nombres_camaras[0]

    # Si hay múltiples cámaras (webcam, USB, OBS Virtual Cam, etc.)
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

# Selección dinámica de la cámara con nombre real
CAMARA_INDEX, NOMBRE_CAMARA = seleccionar_camara()

# Cargar modelo YOLOv8
model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(CAMARA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = 30

blur_percent = 50  
show_preview = True  
last_key_time = 0

def mostrar_interfaz(nivel, vista_previa, nombre_camara):
    """Limpia la terminal y muestra un panel interactivo ordenado."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    bloques = int(nivel / 10)
    barra = "█" * bloques + "░" * (10 - bloques)
    
    estado_vista = "ACTIVADA" if vista_previa else "OCULTA (Ahorro de recursos)"
    
    print("=====================================================")
    print("        JL-VirtualCam-IA | PANEL DE CONTROL          ")
    print("=====================================================")
    print(f" CÁMARA ACTIVA:      {nombre_camara}")
    print(f" NIVEL DE DESENFOQUE: [{barra}] {nivel}%")
    print(f" VISTA PREVIA:       {estado_vista}")
    print("-----------------------------------------------------")
    print(" CONTROLES GLOBALES:")
    print("   [ + ] / [ - ] : Aumentar desenfoque (+10%)")
    print("   [ - ]         : Disminuir desenfoque (-10%)")
    print("   [ Q ]         : Mostrar / Ocultar Vista Previa")
    print("   [ Ctrl + C ]  : Detener programa")
    print("=====================================================")

# Imprimir la interfaz inicial
mostrar_interfaz(blur_percent, show_preview, NOMBRE_CAMARA)

NOMBRE_VENTANA = "JL-VirtualCam-IA (Controles)"

try:
    with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue

            # Convertir porcentaje a kernel impar (1 a 151)
            kernel_size = int(1 + (blur_percent / 100.0) * 150)
            if kernel_size % 2 == 0:
                kernel_size += 1

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

            # Transmitir a la Cámara Virtual
            cam.send(final_output)

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
                    mostrar_interfaz(blur_percent, show_preview, NOMBRE_CAMARA)

            # VISTA PREVIA
            if show_preview:
                preview_frame = final_output.copy()
                cv2.putText(preview_frame, f"Desenfoque: {blur_percent}%", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.imshow(NOMBRE_VENTANA, preview_frame)
                cv2.waitKey(1)

            # Control de FPS
            elapsed_time = time.time() - start_time
            sleep_time = max(0, (1.0 / fps) - elapsed_time)
            time.sleep(sleep_time)

except KeyboardInterrupt:
    os.system('cls' if os.name == 'nt' else 'clear')
    print("=====================================================")
    print("      Programa JL-VirtualCam-IA finalizado.          ")
    print("=====================================================")

finally:
    cap.release()
    cv2.destroyAllWindows()