@echo off
title JL-VirtualCam-IA - Modulo de Desenfoque
echo ===================================================
echo Iniciando script de desenfoque... Por favor espere.
echo ===================================================
echo.

:: Se posiciona en la misma carpeta donde esta guardado el archivo .bat
cd /d "%~dp0"

:: 1. Intenta ejecutar con 'python'
python desenfoque.py 2>nul
if %ERRORLEVEL% EQU 0 goto FIN

:: 2. Si falla, intenta ejecutar con 'py' (lanzador de Windows)
py desenfoque.py 2>nul
if %ERRORLEVEL% EQU 0 goto FIN

:: 3. Si ambos fallan, busca 'python.exe' en la ruta habitual del usuario actual
if exist "%LocalAppData%\Programs\Python\Python*\python.exe" (
    for /d %%I in ("%LocalAppData%\Programs\Python\Python*") do (
        "%%I\python.exe" desenfoque.py
        if %ERRORLEVEL% EQU 0 goto FIN
    )
)

:: Mensaje de error solo si ningun metodo funciono
echo.
echo [ERROR] No se pudo encontrar una instalacion activa de Python.
echo Por favor asegurese de tener Python instalado y agregado al PATH.
echo.
pause

:FIN