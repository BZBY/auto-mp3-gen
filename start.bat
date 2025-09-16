@echo off
echo ================================
echo 动漫角色对话提取系统
echo ================================
echo.
echo 正在启动系统...
echo.

REM 检查Python是否安装
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Python，请先安装Python 3.8+
    pause
    exit /b 1
)

REM 检查Node.js是否安装
node --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Node.js，请先安装Node.js 16+
    pause
    exit /b 1
)

REM 检查yarn是否安装
yarn --version >nul 2>&1
if %errorlevel% neq 0 (
    echo 错误: 未找到Yarn，请先安装Yarn
    echo 可以使用: npm install -g yarn
    pause
    exit /b 1
)

echo 正在检查依赖...

REM 检查后端依赖
if not exist "backend\venv" (
    echo 创建Python虚拟环境...
    cd backend
    python -m venv venv
    call venv\Scripts\activate.bat
    pip install -r requirements.txt
    cd ..
)

REM 检查前端依赖
if not exist "frontend\node_modules" (
    echo 安装前端依赖...
    cd frontend
    yarn install
    cd ..
)

echo.
echo 依赖检查完成，正在启动服务...
echo.
echo 前端地址: http://localhost:3000
echo 后端地址: http://localhost:5000
echo.
echo 按 Ctrl+C 停止服务
echo.

REM 启动服务
yarn dev

pause
