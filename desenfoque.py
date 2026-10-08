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

def seleccionar_modo_inicial():
    """Permite al usuario elegir qué efecto aplicar al iniciar."""
    print("\n=====================================================")
    print("        SELECCIONA EL EFECTO DE FONDO                ")
    print("=====================================================")
    print("  [ 1 ] -> Desenfoque de Fondo (0% a 100%)")
    print("  [ 2 ] -> Imagen de Fondo Virtual (fondo.jpg)")
    print("=====================================================")
    
    while True:
        opcion = input("Selecciona una opción (1 o 2): ").strip()
        if opcion == '1':
            return 'blur'
        elif opcion == '2':
            return 'image'
        else:
            print("Opción no válida. Ingresa 1 o 2.")

# Configuración Inicial
CAMARA_INDEX, NOMBRE_CAMARA = seleccionar_camara()
modo_actual = seleccionar_modo_inicial()

# Cargar modelo YOLOv8
model = YOLO("yolov8n-seg.pt")

cap = cv2.VideoCapture(CAMARA_INDEX, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = 30

# Cargar imagen de fondo si existe
RUTA_FONDO = "fondo.jpg"
if os.path.exists(RUTA_FONDO):
    bg_image = cv2.imread(RUTA_FONDO)
    bg_image = cv2.resize(bg_image, (width, height))
else:
    bg_image = np.zeros((height, width, 3), dtype=np.uint8)

blur_percent = 50  
show_preview = True  
last_key_time = 0

def mostrar_interfaz_consola(modo, nivel_blur, vista_previa, nombre_camara):
    """Limpia la terminal y muestra el panel de control activo."""
    os.system('cls' if os.name == 'nt' else 'clear')
    
    estado_vista = "ACTIVADA" if vista_previa else "OCULTA (Ahorro de recursos)"
    texto_modo = "DESENFOQUE DE FONDO" if modo == 'blur' else "IMAGEN DE FONDO VIRTUAL"
    
    print("=====================================================")
    print("        JL-VirtualCam-IA | PANEL DE CONTROL          ")
    print("=====================================================")
    print(f" CÁMARA ACTIVA:      {nombre_camara}")
    print(f" MODO ACTIVO:        {texto_modo}")
    
    if modo == 'blur':
        bloques = int(nivel_blur / 10)
        barra = "█" * bloques + "░" * (10 - bloques)
        print(f" NIVEL DE DESENFOQUE: [{barra}] {nivel_blur}%")
    else:
        print(f" IMAGEN ARCHIVO:     {RUTA_FONDO if os.path.exists(RUTA_FONDO) else 'Sin archivo (Fondo Negro)'}")
        
    print(f" VISTA PREVIA:       {estado_vista}")
    print("-----------------------------------------------------")
    print(" CONTROLES GLOBALES:")
    print("   [ M ]         : Cambiar entre Desenfoque e Imagen")
    if modo == 'blur':
        print("   [ + ] / [ = ] : Aumentar desenfoque (+10%)")
        print("   [ - ]         : Disminuir desenfoque (-10%)")
    print("   [ Q ]         : Mostrar / Ocultar Vista Previa")
    print("   [ Ctrl + C ]  : Detener programa")
    print("=====================================================")

def aplicar_diseno_emergente(frame, modo, nivel_blur, nombre_cam, fps_real):
    """Aplica la capa de interfaz gráfica (HUD) sobre la vista previa."""
    h, w, _ = frame.shape
    overlay = frame.copy()

    # Borde exterior Neón
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), (235, 206, 135), 2)

    # Header Superior y Tarjeta Inferior
    cv2.rectangle(overlay, (0, 0), (w, 40), (20, 20, 20), -1)
    cv2.rectangle(overlay, (10, h - 65), (w - 10, h - 10), (25, 25, 25), -1)

    # Aplicar transparencia
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Header: Título y REC
    cv2.putText(frame, "JL-VirtualCam IA", (15, 26), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    
    cv2.circle(frame, (w - 100, 20), 5, (0, 0, 255), -1)
    cv2.putText(frame, "EN VIVO", (w - 88, 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)

    # Información Tarjeta Inferior
    info_modo = f"Blur: {nivel_blur}%" if modo == 'blur' else "Fondo: Imagen"
    cv2.putText(frame, info_modo, (25, h - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 127), 2, cv2.LINE_AA)
    
    cv2.putText(frame, f"FPS: {fps_real}", (170, h - 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    cam_corta = nombre_cam if len(nombre_cam) < 35 else nombre_cam[:32] + "..."
    cv2.putText(frame, f"Camara: {cam_corta}", (25, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)

    return frame

mostrar_interfaz_consola(modo_actual, blur_percent, show_preview, NOMBRE_CAMARA)
NOMBRE_VENTANA = "JL-VirtualCam-IA (Vista Previa)"

fps_count = 0
fps_mostrar = 30
last_fps_time = time.time()

try:
    with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
        while cap.isOpened():
            start_time = time.time()
            ret, frame = cap.read()
            
            if not ret or frame is None:
                continue

            # Inferencia con IA para segmentación
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

            # Generar fondo según el modo elegido
            if modo_actual == 'blur':
                kernel_size = int(1 + (blur_percent / 100.0) * 150)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                
                if blur_percent == 0:
                    fondo_procesado = frame.copy()
                else:
                    fondo_procesado = cv2.GaussianBlur(frame, (kernel_size, kernel_size), 0)
            else:
                fondo_procesado = bg_image

            # Combinar persona con el fondo seleccionado
            final_output = (frame * mask_3d + fondo_procesado * (1.0 - mask_3d)).astype(np.uint8)

            # Transmitir a OBS
            cam.send(final_output)

            # Medidor FPS
            fps_count += 1
            if time.time() - last_fps_time >= 1.0:
                fps_mostrar = fps_count
                fps_count = 0
                last_fps_time = time.time()

            # CONTROL DE TECLAS
            current_time = time.time()
            if current_time - last_key_time > 0.2:
                hubo_cambio = False
                
                # Tecla M: Alternar entre Desenfoque e Imagen
                if keyboard.is_pressed('m'):
                    modo_actual = 'image' if modo_actual == 'blur' else 'blur'
                    hubo_cambio = True
                    last_key_time = current_time
                elif modo_actual == 'blur' and (keyboard.is_pressed('+') or keyboard.is_pressed('=')):
                    if blur_percent < 100:
                        blur_percent += 10
                        hubo_cambio = True
                    last_key_time = current_time
                elif modo_actual == 'blur' and keyboard.is_pressed('-'):
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
                    mostrar_interfaz_consola(modo_actual, blur_percent, show_preview, NOMBRE_CAMARA)

            # VISTA PREVIA
            if show_preview:
                preview_frame = final_output.copy()
                preview_frame = aplicar_diseno_emergente(preview_frame, modo_actual, blur_percent, NOMBRE_CAMARA, fps_mostrar)
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