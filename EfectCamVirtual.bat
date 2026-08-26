@echo off
title Lanzador de JL-VirtualCam-IA
echo Instalando/Verificando dependencias necesarias...
pip install -r requirements.txt
echo.
echo Iniciando Filtro de Camara Virtual...
python desenfoque.py
pause