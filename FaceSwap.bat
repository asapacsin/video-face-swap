@echo off
chcp 65001 >nul
start "" "%~dp0.venv-gpu\Scripts\pythonw.exe" -m face_replace.gui
