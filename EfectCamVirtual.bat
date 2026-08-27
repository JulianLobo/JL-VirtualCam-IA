@echo off
title JL-VirtualCam-IA - Modulo de Desenfoque
echo ===================================================
echo Iniciando script de desenfoque... Por favor espere.
echo ===================================================
echo.

:: Navegar a la carpeta del proyecto
cd /d "C:\Users\soporte7\Documents\lobo\Documento\GITHUB\JL-VirtualCam-IA"

:: Ejecutar el script usando el ejecutor 'py' para evitar bloqueos
py desenfoque.py

:: Si ocurre un error, mantendrá la ventana abierta para ver el mensaje
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo Ocurrio un error al ejecutar el programa.
    pause
)