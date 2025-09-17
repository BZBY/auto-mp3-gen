#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio GUI 配置文件
"""

import os
from pathlib import Path

class GradioConfig:
    """Gradio GUI配置类"""

    # 基础配置
    APP_TITLE = "🎭 动漫角色对话提取系统"
    APP_DESCRIPTION = "基于Gradio 5.x构建的现代化Web界面"

    # 服务器配置
    HOST = "127.0.0.1"
    PORT = 28000

    # 文件配置
    MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024  # 2GB
    TEMP_DIR = Path.cwd() / "temp"
    OUTPUT_DIR = Path.cwd() / "output"

    # 支持的文件格式
    SUPPORTED_VIDEO_FORMATS = [".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".m4v"]
    SUPPORTED_SUBTITLE_FORMATS = [".srt", ".ass", ".ssa", ".vtt"]

    # Gradio主题配置
    THEME_CONFIG = {
        "primary_hue": "blue",
        "secondary_hue": "slate",
        "neutral_hue": "slate",
    }

    # CSS样式
    CUSTOM_CSS = """
    .gradio-container {
        font-family: 'Microsoft YaHei', 'Segoe UI', system-ui, sans-serif;
        max-width: 1200px;
        margin: 0 auto;
    }

    .gr-button {
        transition: all 0.2s ease;
        border-radius: 8px;
    }

    .gr-button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }

    .gr-button-primary {
        background: linear-gradient(135deg, #3b82f6 0%, #1d4ed8 100%);
        border: none;
    }

    .status-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.5rem;
        margin: 1rem 0;
    }

    .success-message {
        border-left: 4px solid #10b981;
        background: #f0fdf4;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .error-message {
        border-left: 4px solid #ef4444;
        background: #fef2f2;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .info-message {
        border-left: 4px solid #3b82f6;
        background: #eff6ff;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .gr-tab-nav {
        background: #f1f5f9;
        border-radius: 12px;
        padding: 4px;
    }

    .gr-tab-nav button {
        border-radius: 8px;
        transition: all 0.2s ease;
    }

    .gr-file {
        border: 2px dashed #cbd5e1;
        border-radius: 12px;
        transition: all 0.2s ease;
    }

    .gr-file:hover {
        border-color: #3b82f6;
        background: #f8fafc;
    }
    """

    @classmethod
    def init_directories(cls):
        """初始化必要的目录"""
        cls.TEMP_DIR.mkdir(exist_ok=True)
        cls.OUTPUT_DIR.mkdir(exist_ok=True)

    @classmethod
    def get_gradio_config(cls):
        """获取Gradio启动配置"""
        return {
            "server_name": cls.HOST,
            "server_port": cls.PORT,
            "share": False,
            "inbrowser": True,
            "show_error": True,
            "favicon_path": None,
            "ssr_mode": True,  # Gradio 5.x SSR特性
            "show_api": False,
            "quiet": False
        }