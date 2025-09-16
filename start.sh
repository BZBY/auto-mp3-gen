#!/bin/bash

echo "================================"
echo "动漫角色对话提取系统"
echo "================================"
echo ""
echo "正在启动系统..."
echo ""

# 检查Python是否安装
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到Python，请先安装Python 3.8+"
    exit 1
fi

# 检查Node.js是否安装
if ! command -v node &> /dev/null; then
    echo "错误: 未找到Node.js，请先安装Node.js 16+"
    exit 1
fi

# 检查yarn是否安装
if ! command -v yarn &> /dev/null; then
    echo "错误: 未找到Yarn，请先安装Yarn"
    echo "可以使用: npm install -g yarn"
    exit 1
fi

echo "正在检查依赖..."

# 检查后端依赖
if [ ! -d "backend/venv" ]; then
    echo "创建Python虚拟环境..."
    cd backend
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    cd ..
fi

# 检查前端依赖
if [ ! -d "frontend/node_modules" ]; then
    echo "安装前端依赖..."
    cd frontend
    yarn install
    cd ..
fi

echo ""
echo "依赖检查完成，正在启动服务..."
echo ""
echo "前端地址: http://localhost:3000"
echo "后端地址: http://localhost:5000"
echo ""
echo "按 Ctrl+C 停止服务"
echo ""

# 启动服务
yarn dev
