#!/bin/bash

echo "🎭 动漫角色对话提取系统 - Gradio版"
echo "================================"
echo

# 获取脚本目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
cd "$SCRIPT_DIR"

# 运行启动脚本
python3 start.py