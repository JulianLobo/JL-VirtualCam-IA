# JL-VirtualCam-IA 🚀

Filtro inteligente de desenfoque de fondo en tiempo real para OBS Studio utilizando Inteligencia Artificial (YOLOv8), OpenCV y PyVirtualCam. Detecta y prioriza automáticamente al sujeto principal.

---

## 🌟 Características

* **Segmentación por IA:** Utiliza el modelo ultraligero `YOLOv8n-seg` para detectar la silueta humana completa (cuerpo, cabello, ropa y accesorios).
* **Filtro inteligente de sujeto:** Calcula el área ocupada y prioriza automáticamente a la persona en primer plano (ideal para entornos compartidos u oficinas).
* **Integración con OBS:** Emite directamente a la **OBS Virtual Camera** sin requerir configuraciones complejas dentro del software de transmisión.
* **Control de FPS:** Optimizado a 30 FPS para no saturar el rendimiento del procesador ni colapsar OBS.
* **Incluye ejecutable `.bat` para usuarios sin experiencia en programación.

---

## 🛠️ Requisitos Previos

* Python 3.10 o superior.
* OBS Studio instalado (con la función de Cámara Virtual habilitada).
* Webcam física conectada al sistema.

---

## 📦 Instalación

1. Clona este repositorio o descarga los archivos:
   ```bash
   git clone [https://github.com/JulianLobo/JL-VirtualCam-IA.git](https://github.com/JulianLobo/JL-VirtualCam-IA.git)
   cd JL-VirtualCam-IA
2. Instala las dependencias necesarias:
  pip install -r requirements.txt

4. 🚀 Uso

    Asegúrate de tener OBS Studio cerrado (para permitir que Python tome el control de la webcam).
   Ejecuta el script principal en Powershell:
   python desenfoque.py

6. 📄 Licencia

Este proyecto está bajo la Licencia MIT - consulta el archivo LICENSE para más detalles.

Desarrollado por Julián Lobo
