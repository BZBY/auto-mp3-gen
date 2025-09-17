#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动漫角色对话提取系统 - 启动脚本
一键启动Gradio GUI界面
"""

import os
import sys
import subprocess
from pathlib import Path

def check_python_version():
    """检查Python版本"""
    if sys.version_info < (3, 8):
        print("❌ 错误: 需要Python 3.8或更高版本")
        print(f"当前版本: {sys.version}")
        return False
    return True

def install_dependencies():
    """安装依赖包"""
    print("📦 检查并安装依赖包...")
    requirements_file = Path(__file__).parent / "requirements.txt"

    try:
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", "-r", str(requirements_file)
        ])
        print("✅ 依赖包安装完成")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ 依赖包安装失败: {e}")
        return False

def main():
    """主启动函数"""
    print("🎭 动漫角色对话提取系统 - Gradio版")
    print("=" * 50)

    # 检查Python版本
    if not check_python_version():
        input("按Enter键退出...")
        return

    # 检查是否需要安装依赖
    try:
        import gradio
        print(f"✅ Gradio已安装 (版本: {gradio.__version__})")
    except ImportError:
        print("📦 Gradio未安装，正在安装依赖包...")
        if not install_dependencies():
            input("按Enter键退出...")
            return

    # 启动应用
    print("🚀 启动Gradio界面...")
    print("💻 访问地址: http://127.0.0.1:28000")
    print("⚠️ 关闭此窗口将停止服务")
    print("-" * 50)

    # 导入并运行主应用
    try:
        from app_ux_enhanced import main as run_app
        run_app()
    except KeyboardInterrupt:
        print("\n👋 服务已停止")
    except Exception as e:
        print(f"❌ 启动失败: {e}")
        import traceback
        traceback.print_exc()
        input("按Enter键退出...")

if __name__ == "__main__":
    main()