@echo off
title JL-VirtualCam-IA - Modulo de Desenfoque
cls

echo ===================================================
echo   Verificando librerias del sistema...
echo ===================================================
pip install -r requirements.txt >nul 2>&1

cls
echo ===================================================
echo      Iniciando JL-VirtualCam-IA...
echo ===================================================
echo.

python desenfoque.py
pause