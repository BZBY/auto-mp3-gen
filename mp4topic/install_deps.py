#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
依赖安装脚本
自动安装视频关键帧提取器所需的所有依赖包
"""

import subprocess
import sys
import os


def run_command(command, description):
    """运行命令并显示结果"""
    print(f"\n🔧 {description}...")
    print(f"执行命令: {command}")

    try:
        result = subprocess.run(command, shell=True, check=True,
                              capture_output=True, text=True)
        print("✅ 成功")
        if result.stdout:
            print(result.stdout[-200:])  # 显示最后200个字符
        return True
    except subprocess.CalledProcessError as e:
        print("❌ 失败")
        print(f"错误信息: {e.stderr}")
        return False


def main():
    """主安装函数"""
    print("🎬 视频关键帧提取器 - 依赖安装脚本")
    print("=" * 50)

    # 检查Python版本
    python_version = sys.version_info
    print(f"Python版本: {python_version.major}.{python_version.minor}.{python_version.micro}")

    if python_version < (3, 7):
        print("❌ 需要Python 3.7或更高版本")
        return False

    # 升级pip
    if not run_command("python -m pip install --upgrade pip", "升级pip"):
        print("⚠️ pip升级失败，继续安装其他包...")

    # 基础科学计算包
    packages = [
        ("numpy>=1.21.0", "NumPy数值计算库"),
        ("opencv-python>=4.5.0", "OpenCV图像处理库"),
        ("pillow>=8.0.0", "PIL图像处理库"),
        ("scipy>=1.7.0", "SciPy科学计算库"),
        ("scikit-learn>=1.0.0", "机器学习库"),
        ("pandas>=1.3.0", "数据处理库"),
        ("tqdm>=4.62.0", "进度条库"),
    ]

    print("\n📦 安装基础依赖包...")
    failed_packages = []

    for package, description in packages:
        if not run_command(f"pip install {package}", f"安装{description}"):
            failed_packages.append(package)

    # PyTorch (重要)
    print("\n🔥 安装PyTorch...")
    if not run_command("pip install torch torchvision", "安装PyTorch和TorchVision"):
        print("❌ PyTorch安装失败，这是必需的依赖")
        failed_packages.append("torch")

    # CLIP (重要)
    print("\n🧠 安装CLIP模型...")
    if not run_command("pip install git+https://github.com/openai/CLIP.git", "安装CLIP模型"):
        print("❌ CLIP安装失败，这是必需的依赖")
        failed_packages.append("clip")

    # TransNetV2 PyTorch版本 (可选但推荐)
    print("\n🎬 安装TransNetV2 PyTorch版本...")
    if not run_command("pip install transnetv2-pytorch", "安装TransNetV2 PyTorch版本"):
        print("⚠️ TransNetV2安装失败，将使用备用镜头检测算法")

    # 显示安装结果
    print("\n" + "=" * 50)
    print("📊 安装结果汇总")
    print("=" * 50)

    if not failed_packages:
        print("✅ 所有依赖包安装成功！")
        print("\n🎉 您现在可以运行:")
        print("  python quick_test.py")
        print("  或者")
        print("  python main.py 1.mp4")

    else:
        print(f"❌ {len(failed_packages)} 个包安装失败:")
        for package in failed_packages:
            print(f"  - {package}")

        print("\n💡 解决建议:")
        print("  1. 检查网络连接")
        print("  2. 尝试使用国内镜像:")
        print("     pip install -i https://pypi.tuna.tsinghua.edu.cn/simple package_name")
        print("  3. 如果是Windows，可能需要安装Visual C++构建工具")

    return len(failed_packages) == 0


if __name__ == "__main__":
    try:
        success = main()

        if success:
            print("\n🎉 安装完成！现在可以测试系统了。")
        else:
            print("\n⚠️ 部分依赖安装失败，但可能仍然可以运行基本功能。")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断安装")
    except Exception as e:
        print(f"\n❌ 安装过程中出错: {e}")

    # Windows下暂停
    if os.name == 'nt':
        input("\n按Enter键退出...")