#!/usr/bin/env python3
"""
动漫角色对话提取系统启动脚本
"""

import os
import sys
import subprocess
import time
from pathlib import Path

def check_requirements():
    """检查系统要求"""
    print("🔍 检查系统要求...")
    
    # 检查Python版本
    if sys.version_info < (3, 8):
        print("❌ Python版本过低，需要Python 3.8+")
        return False
    
    # 检查Node.js
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ 未找到Node.js，请先安装Node.js 16+")
            return False
    except FileNotFoundError:
        print("❌ 未找到Node.js，请先安装Node.js 16+")
        return False
    
    # 检查yarn
    try:
        result = subprocess.run(['yarn', '--version'], capture_output=True, text=True)
        if result.returncode != 0:
            print("❌ 未找到Yarn，请运行: npm install -g yarn")
            return False
    except FileNotFoundError:
        print("❌ 未找到Yarn，请运行: npm install -g yarn")
        return False
    
    print("✅ 系统要求检查通过")
    return True

def install_dependencies():
    """安装依赖"""
    print("📦 检查和安装依赖...")
    
    # 检查后端依赖
    backend_deps = Path("backend/requirements.txt")
    if backend_deps.exists():
        print("  安装后端依赖...")
        try:
            subprocess.run([
                sys.executable, "-m", "pip", "install", "-r", str(backend_deps)
            ], check=True, cwd="backend")
            print("  ✅ 后端依赖安装完成")
        except subprocess.CalledProcessError:
            print("  ⚠️ 后端依赖安装失败，请手动运行: pip install -r backend/requirements.txt")
    
    # 检查前端依赖
    frontend_deps = Path("frontend/node_modules")
    if not frontend_deps.exists():
        print("  安装前端依赖...")
        try:
            subprocess.run(["yarn", "install"], check=True, cwd="frontend")
            print("  ✅ 前端依赖安装完成")
        except subprocess.CalledProcessError:
            print("  ⚠️ 前端依赖安装失败，请手动运行: cd frontend && yarn install")

def start_services():
    """启动服务"""
    print("🚀 启动系统...")
    print("=" * 60)
    print("动漫角色对话提取系统")
    print("前端地址: http://localhost:3000")
    print("后端地址: http://localhost:5000")
    print("=" * 60)
    print("按 Ctrl+C 停止服务")
    print()
    
    try:
        # 使用concurrently同时启动前后端
        subprocess.run(["yarn", "dev"], check=True)
    except subprocess.CalledProcessError:
        print("❌ 启动失败")
        print("请尝试手动启动:")
        print("  终端1: cd backend && python app.py")
        print("  终端2: cd frontend && yarn start")
    except KeyboardInterrupt:
        print("\n👋 服务已停止")

def main():
    """主函数"""
    print("🎭 动漫角色对话提取系统")
    print("=" * 60)
    
    if not check_requirements():
        print("\n请安装必要的软件后重试")
        sys.exit(1)
    
    install_dependencies()
    
    print("\n🎉 准备就绪！")
    input("按回车键启动系统...")
    
    start_services()

if __name__ == "__main__":
    main()
