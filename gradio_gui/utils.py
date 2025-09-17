#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Gradio GUI 工具函数
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FileHandler:
    """文件处理工具类"""

    @staticmethod
    def create_temp_directory() -> str:
        """创建临时目录"""
        return tempfile.mkdtemp()

    @staticmethod
    def cleanup_temp_directory(temp_dir: str) -> bool:
        """清理临时目录"""
        try:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
                logger.info(f"已清理临时目录: {temp_dir}")
                return True
        except Exception as e:
            logger.error(f"清理临时目录失败: {e}")
        return False

    @staticmethod
    def copy_uploaded_file(uploaded_file, target_dir: str, new_name: str = None) -> str:
        """复制上传的文件到目标目录"""
        if new_name is None:
            new_name = f"file{Path(uploaded_file.name).suffix}"

        target_path = os.path.join(target_dir, new_name)
        shutil.copy2(uploaded_file.name, target_path)
        return target_path

    @staticmethod
    def collect_output_files(output_dir: str) -> List[str]:
        """收集输出目录中的所有文件"""
        files = []
        if os.path.exists(output_dir):
            for root, dirs, filenames in os.walk(output_dir):
                for filename in filenames:
                    file_path = os.path.join(root, filename)
                    files.append(file_path)
        return files

class StatusFormatter:
    """状态信息格式化工具类"""

    @staticmethod
    def format_success_message(title: str, details: Dict) -> str:
        """格式化成功消息"""
        msg = f"## ✅ {title}\n\n"

        if 'speakers_count' in details:
            msg += f"### 📊 处理结果:\n"
            msg += f"- **识别出说话人**: {details['speakers_count']} 个\n"
            msg += f"- **总对话片段**: {details['total_segments']} 个\n"
            msg += f"- **输出目录**: `{details['output_directory']}`\n\n"

        if 'export_info' in details:
            msg += f"### 💾 生成的文件:\n"
            export_info = details['export_info']
            if isinstance(export_info, dict):
                msg += f"- **导出信息**: {export_info.get('export_file', '未知')}\n"
            msg += f"- **摘要文件**: {details.get('summary_file', '未知')}\n\n"

        msg += "> 🎉 所有文件已准备就绪，可以在结果页面下载！"
        return msg

    @staticmethod
    def format_error_message(title: str, error: str, details: str = None) -> str:
        """格式化错误消息"""
        msg = f"## ❌ {title}\n\n"
        msg += f"**错误信息**: {error}\n\n"

        if details:
            msg += f"### 详细错误信息:\n```\n{details}\n```"

        return msg

    @staticmethod
    def format_system_check_result(status: Dict) -> str:
        """格式化系统检查结果"""
        report = "## 🔍 系统环境检查结果\n\n"

        for component, info in status.items():
            if isinstance(info, dict):
                status_icon = "✅" if info.get('available', False) else "❌"
                report += f"{status_icon} **{component}**: {info.get('message', 'Unknown')}\n"
            else:
                report += f"ℹ️ **{component}**: {info}\n"

        return report

class ProgressTracker:
    """进度跟踪工具类"""

    def __init__(self, total_steps: int = 6):
        self.total_steps = total_steps
        self.current_step = 0
        self.messages = []

    def update(self, step: int, message: str):
        """更新进度"""
        self.current_step = step
        self.messages.append(message)

    def get_progress_ratio(self) -> float:
        """获取进度比例"""
        return self.current_step / self.total_steps

    def get_progress_percentage(self) -> int:
        """获取进度百分比"""
        return int(self.get_progress_ratio() * 100)

    def get_latest_message(self) -> str:
        """获取最新消息"""
        return self.messages[-1] if self.messages else ""

class ConfigValidator:
    """配置验证工具类"""

    @staticmethod
    def validate_file_format(file_path: str, allowed_extensions: List[str]) -> Tuple[bool, str]:
        """验证文件格式"""
        if not file_path:
            return False, "文件路径为空"

        ext = Path(file_path).suffix.lower()
        if ext not in allowed_extensions:
            return False, f"不支持的文件格式: {ext}"

        return True, "文件格式验证通过"

    @staticmethod
    def validate_cluster_count(n_clusters: int) -> Tuple[bool, str]:
        """验证聚类数量"""
        if n_clusters < 0:
            return False, "聚类数量不能为负数"
        if n_clusters > 20:
            return False, "聚类数量过大，建议不超过20"

        return True, "聚类数量验证通过"

    @staticmethod
    def validate_uvr5_path(uvr5_path: str) -> Tuple[bool, str]:
        """验证UVR5路径"""
        if not uvr5_path or not uvr5_path.strip():
            return True, "UVR5路径为空，将跳过人声分离"

        if not os.path.exists(uvr5_path):
            return False, f"UVR5路径不存在: {uvr5_path}"

        return True, "UVR5路径验证通过"