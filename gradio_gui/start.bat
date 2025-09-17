@echo off
chcp 65001 > nul
title 动漫角色对话提取系统 - Gradio GUI

echo.
echo 🎭 动漫角色对话提取系统 - Gradio版
echo ================================
echo.

cd /d "%~dp0"

python start.py

echo.
echo 按任意键退出...
pause > nul