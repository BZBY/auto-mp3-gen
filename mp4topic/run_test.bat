@echo off
echo 🎬 视频关键帧提取器 - 一键测试
echo ====================================
echo.

echo 检查Python环境...
python --version
if %errorlevel% neq 0 (
    echo ❌ Python未安装或不在PATH中
    pause
    exit /b 1
)

echo.
echo 🚀 开始运行测试...
echo.

python quick_test.py

echo.
echo 测试完成，按任意键退出...
pause > nul