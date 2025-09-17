#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动漫角色对话提取系统 - 用户体验至上版本
专注于极致的用户体验设计
"""

import gradio as gr
import os
import sys
import traceback
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import threading
import time

# 添加backend路径到系统路径
backend_path = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_path))

try:
    from core.main_processor import MainProcessor
    from config import Config
    from utils.system_check import SystemChecker
except ImportError as e:
    print(f"导入后端模块失败: {e}")
    print("请确保backend目录存在且包含必要的模块")
    sys.exit(1)

# 全局状态
app_state = {
    'current_step': 0,
    'files_uploaded': False,
    'video_validated': False,
    'subtitle_validated': False,
    'processing': False,
    'result': None
}

def validate_file_realtime(file, file_type="video"):
    """实时文件验证 - 上传时立即反馈"""
    if file is None:
        return "📋 请选择文件...", "⚪"

    try:
        file_path = Path(file.name)
        file_size = os.path.getsize(file.name) / (1024 * 1024)  # MB

        if file_type == "video":
            allowed_exts = Config.ALLOWED_VIDEO_EXTENSIONS
            type_name = "视频"
        else:
            allowed_exts = Config.ALLOWED_SUBTITLE_EXTENSIONS
            type_name = "字幕"

        ext = file_path.suffix.lower()

        if ext not in allowed_exts:
            return f"❌ 不支持的{type_name}格式: {ext}", "🔴"

        # 更新全局状态
        if file_type == "video":
            app_state['video_validated'] = True
        else:
            app_state['subtitle_validated'] = True

        app_state['files_uploaded'] = app_state['video_validated'] and app_state['subtitle_validated']

        return f"✅ {type_name}文件验证通过\n📁 文件名: {file_path.name}\n📊 大小: {file_size:.1f} MB", "🟢"

    except Exception as e:
        return f"❌ 文件验证失败: {str(e)}", "🔴"

def update_step_indicator(current_step):
    """更新步骤指示器"""
    steps = ["📁 上传文件", "⚙️ 配置参数", "🚀 开始处理", "📊 查看结果"]

    html = "<div style='display: flex; justify-content: space-between; margin: 20px 0;'>"

    for i, step in enumerate(steps):
        if i < current_step:
            color = "#10b981"  # 已完成 - 绿色
            icon = "✅"
        elif i == current_step:
            color = "#3b82f6"  # 当前步骤 - 蓝色
            icon = "▶️"
        else:
            color = "#6b7280"  # 未开始 - 灰色
            icon = "⚪"

        html += f"""
        <div style='text-align: center; flex: 1;'>
            <div style='font-size: 24px; margin-bottom: 8px;'>{icon}</div>
            <div style='color: {color}; font-weight: bold; font-size: 14px;'>{step}</div>
        </div>
        """

        if i < len(steps) - 1:
            line_color = "#10b981" if i < current_step else "#e5e7eb"
            html += f"<div style='flex: 0.5; height: 2px; background: {line_color}; margin: 20px 0; align-self: center;'></div>"

    html += "</div>"
    return html

def smart_help_system(context="welcome"):
    """智能帮助系统"""
    help_texts = {
        "welcome": """
        ## 🎯 使用指南

        ### 第一次使用？跟着这个步骤：
        1. **🔍 系统检查** - 确保环境正常（推荐先做）
        2. **📁 上传文件** - 视频+字幕文件
        3. **⚙️ 配置参数** - 说话人数量等
        4. **🚀 开始处理** - 一键启动
        5. **📥 下载结果** - 获取分离后的音频

        ### 💡 小贴士：
        - 支持拖拽上传文件
        - 文件会实时验证格式
        - 处理进度实时显示
        """,

        "upload": """
        ## 📁 文件上传帮助

        ### 视频文件要求：
        - **格式**: MP4, AVI, MKV, MOV, WMV, FLV, M4V
        - **大小**: 建议不超过2GB
        - **质量**: 音频清晰度越高效果越好

        ### 字幕文件要求：
        - **格式**: SRT, ASS, SSA, VTT
        - **时间轴**: 必须与视频准确对应
        - **编码**: 建议使用UTF-8编码

        ### ⚠️ 常见问题：
        - 字幕时间不准确会影响分割效果
        - 多角色对话重叠时效果可能不佳
        """,

        "config": """
        ## ⚙️ 参数配置帮助

        ### 说话人数量：
        - **自动检测 (推荐)**: 设置为0
        - **手动指定**: 根据实际角色数量设置
        - **建议范围**: 2-8个角色效果最佳

        ### UVR5路径：
        - **什么是UVR5**: 专业人声分离工具
        - **是否必需**: 不必需，但强烈推荐
        - **下载地址**: https://github.com/Anjok07/ultimatevocalremovergui
        - **效果提升**: 可显著提高语音识别准确性
        """,

        "processing": """
        ## 🚀 处理过程说明

        ### 处理步骤 (大约需要5-15分钟)：
        1. **音频提取** - 从视频中提取音轨
        2. **字幕解析** - 分析字幕时间轴
        3. **音频切分** - 按字幕时间切分音频
        4. **人声分离** - UVR5处理 (如果配置)
        5. **特征提取** - AI分析说话人特征
        6. **聚类分析** - 自动分组不同说话人

        ### 📊 进度说明：
        - 实时显示当前步骤
        - 可以随时查看详细日志
        - 处理时间取决于视频长度
        """
    }

    return help_texts.get(context, help_texts["welcome"])

def enhanced_progress_tracker(progress, message, step_details=None):
    """增强的进度追踪"""
    # 创建可视化进度条
    progress_html = f"""
    <div style='margin: 20px 0;'>
        <div style='display: flex; justify-content: space-between; margin-bottom: 8px;'>
            <span style='font-weight: bold;'>{message}</span>
            <span>{int(progress * 100)}%</span>
        </div>
        <div style='background: #e5e7eb; border-radius: 10px; height: 8px; overflow: hidden;'>
            <div style='background: linear-gradient(90deg, #3b82f6, #1d4ed8); height: 100%; width: {progress * 100}%; transition: width 0.3s ease;'></div>
        </div>
    </div>
    """

    if step_details:
        progress_html += f"<div style='margin-top: 10px; font-size: 14px; color: #6b7280;'>{step_details}</div>"

    return progress_html

def user_friendly_error(error_msg, suggested_solutions=None):
    """用户友好的错误处理"""
    friendly_errors = {
        "FileNotFoundError": {
            "title": "文件未找到",
            "message": "上传的文件可能已被移动或删除",
            "solutions": ["重新上传文件", "检查文件是否完整", "尝试使用其他文件"]
        },
        "ImportError": {
            "title": "缺少必要组件",
            "message": "系统缺少某些必要的处理组件",
            "solutions": ["运行系统检查", "重新安装依赖", "检查Python环境"]
        },
        "ValueError": {
            "title": "参数设置错误",
            "message": "某些参数设置不正确",
            "solutions": ["检查文件格式", "调整参数配置", "查看使用说明"]
        }
    }

    # 尝试匹配错误类型
    error_type = type(error_msg).__name__ if hasattr(error_msg, '__name__') else "未知错误"

    for err_key, err_info in friendly_errors.items():
        if err_key in str(error_msg):
            title = err_info["title"]
            message = err_info["message"]
            solutions = err_info["solutions"]
            break
    else:
        title = "处理过程出错"
        message = str(error_msg)
        solutions = suggested_solutions or ["重试操作", "检查输入文件", "查看详细错误信息"]

    html = f"""
    <div style='border-left: 4px solid #ef4444; background: #fef2f2; padding: 20px; border-radius: 8px; margin: 20px 0;'>
        <h3 style='margin: 0 0 10px 0; color: #dc2626;'>❌ {title}</h3>
        <p style='margin: 0 0 15px 0; color: #374151;'>{message}</p>
        <div style='background: white; padding: 15px; border-radius: 6px; border: 1px solid #fca5a5;'>
            <h4 style='margin: 0 0 10px 0; color: #dc2626;'>💡 建议解决方案：</h4>
            <ul style='margin: 0; padding-left: 20px;'>
    """

    for solution in solutions:
        html += f"<li style='margin: 5px 0; color: #374151;'>{solution}</li>"

    html += """
            </ul>
        </div>
    </div>
    """

    return html

def process_with_enhanced_ux(video_file, subtitle_file, n_clusters, uvr5_path, progress=gr.Progress()):
    """增强用户体验的处理函数"""
    try:
        app_state['processing'] = True

        # 步骤1: 验证和准备
        progress(0.05, desc="🔍 验证文件和参数...")
        time.sleep(0.5)  # 让用户看到进度开始

        if not video_file or not subtitle_file:
            return user_friendly_error("请确保已上传视频和字幕文件",
                                     ["检查文件是否正确上传", "刷新页面重试"])

        # 步骤2: 文件准备
        progress(0.1, desc="📁 准备处理文件...")
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, f"video{Path(video_file.name).suffix}")
        subtitle_path = os.path.join(temp_dir, f"subtitle{Path(subtitle_file.name).suffix}")

        shutil.copy2(video_file.name, video_path)
        shutil.copy2(subtitle_file.name, subtitle_path)

        # 步骤3: 初始化处理器
        progress(0.15, desc="⚙️ 初始化AI处理器...")
        uvr5_path = uvr5_path.strip() if uvr5_path else None
        processor = MainProcessor(uvr5_path)

        # 增强的进度回调
        def enhanced_progress_callback(percent, msg):
            # 添加更详细的步骤说明
            step_details = {
                "正在提取音轨": "从视频文件中分离音频轨道，这可能需要几分钟...",
                "正在解析字幕": "分析字幕文件的时间轴信息...",
                "正在切分音频": "根据字幕时间精确切分音频片段...",
                "正在进行人声分离": "使用UVR5技术分离人声和背景音乐...",
                "正在提取说话人特征": "使用AI分析每个音频片段的说话人特征...",
                "正在进行说话人聚类": "智能识别和分组不同的说话人..."
            }

            detail = step_details.get(msg, "正在处理，请稍候...")
            progress(percent / 100, desc=f"{msg} - {detail}")

        processor.set_progress_callback(enhanced_progress_callback)

        # 步骤4: 开始主要处理
        progress(0.2, desc="🚀 开始智能分析处理...")
        result = processor.process(video_path, subtitle_path, n_clusters if n_clusters > 0 else None)

        if result['success']:
            # 收集输出文件
            output_files = []
            output_dir = result.get('output_directory', '')
            if output_dir and os.path.exists(output_dir):
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        output_files.append(file_path)

            progress(1.0, desc="✅ 处理完成！")

            # 创建成功结果展示
            success_html = f"""
            <div style='border-left: 4px solid #10b981; background: #f0fdf4; padding: 20px; border-radius: 8px; margin: 20px 0;'>
                <h2 style='margin: 0 0 15px 0; color: #065f46;'>🎉 处理成功完成！</h2>

                <div style='background: white; padding: 15px; border-radius: 6px; margin: 15px 0; border: 1px solid #a7f3d0;'>
                    <h3 style='margin: 0 0 10px 0; color: #065f46;'>📊 处理结果统计</h3>
                    <div style='display: grid; grid-template-columns: 1fr 1fr; gap: 15px;'>
                        <div>
                            <strong>🗣️ 识别说话人数量:</strong><br>
                            <span style='font-size: 24px; color: #10b981;'>{result.get('speakers_count', 0)} 个</span>
                        </div>
                        <div>
                            <strong>🎵 对话片段总数:</strong><br>
                            <span style='font-size: 24px; color: #10b981;'>{result.get('total_segments', 0)} 个</span>
                        </div>
                    </div>
                </div>

                <div style='background: #f8fafc; padding: 15px; border-radius: 6px; margin: 15px 0;'>
                    <h4 style='margin: 0 0 10px 0; color: #065f46;'>📁 生成的文件</h4>
                    <ul style='margin: 0; padding-left: 20px;'>
                        <li>各说话人的音频文件 (WAV格式)</li>
                        <li>对应的字幕文本文件 (TXT格式)</li>
                        <li>详细的处理报告和统计信息</li>
                    </ul>
                </div>

                <div style='margin-top: 20px; padding: 15px; background: #eff6ff; border-radius: 6px; border: 1px solid #93c5fd;'>
                    <h4 style='margin: 0 0 10px 0; color: #1e40af;'>📥 下一步操作</h4>
                    <p style='margin: 0; color: #374151;'>
                        请切换到 <strong>"📥 处理结果"</strong> 标签页下载您的文件。
                        所有音频文件已按说话人自动分类，可以直接使用！
                    </p>
                </div>
            </div>
            """

            app_state['result'] = result
            return success_html, result, output_files, gr.update(selected=4)  # 自动切换到结果页

        else:
            error_msg = result.get('message', '未知错误')
            return user_friendly_error(f"处理失败: {error_msg}"), {}, [], gr.update()

    except Exception as e:
        return user_friendly_error(e, ["检查文件完整性", "重新上传文件", "查看系统检查结果"]), {}, [], gr.update()

    finally:
        app_state['processing'] = False
        # 清理临时文件
        try:
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass

def create_ux_enhanced_interface():
    """创建用户体验至上的界面"""

    # 更精美的主题
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="indigo",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Noto Sans"), "ui-sans-serif", "system-ui", "sans-serif"]
    ).set(
        body_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        body_text_color="#1e293b",
        button_primary_background_fill="linear-gradient(135deg, #667eea 0%, #764ba2 100%)",
        button_primary_text_color="white",
        block_background_fill="rgba(255, 255, 255, 0.95)",
        input_background_fill="rgba(255, 255, 255, 0.9)",
    )

    with gr.Blocks(
        title="🎭 动漫角色对话提取系统 - 用户体验至上版",
        theme=theme,
        fill_height=True,
        css="""
        .gradio-container {
            font-family: 'Noto Sans', ui-sans-serif, system-ui, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }

        .gr-button {
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            border-radius: 12px;
            font-weight: 600;
            text-transform: none;
        }

        .gr-button:hover {
            transform: translateY(-2px);
            box-shadow: 0 10px 25px rgba(0, 0, 0, 0.2);
        }

        .gr-button-primary {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            border: none;
            box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
        }

        .gr-file {
            border: 2px dashed #cbd5e1;
            border-radius: 16px;
            transition: all 0.3s ease;
            background: rgba(255, 255, 255, 0.9);
        }

        .gr-file:hover {
            border-color: #667eea;
            background: rgba(255, 255, 255, 1);
            transform: translateY(-1px);
            box-shadow: 0 8px 25px rgba(0, 0, 0, 0.1);
        }

        .gr-tab-nav {
            background: rgba(255, 255, 255, 0.9);
            border-radius: 16px;
            padding: 6px;
            backdrop-filter: blur(10px);
        }

        .gr-tab-nav button {
            border-radius: 12px;
            transition: all 0.3s ease;
            font-weight: 500;
        }

        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }
        """
    ) as demo:

        # 欢迎标题
        gr.HTML("""
        <div style='text-align: center; padding: 30px 20px; background: rgba(255, 255, 255, 0.95); border-radius: 20px; margin: 20px 0; backdrop-filter: blur(10px);'>
            <h1 style='margin: 0; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.5em; font-weight: 800;'>
                🎭 动漫角色对话提取系统
            </h1>
            <p style='margin: 10px 0 0 0; color: #64748b; font-size: 1.2em; font-weight: 500;'>
                用户体验至上版 | AI智能分离 | 一键完成
            </p>
        </div>
        """)

        # 步骤指示器
        step_indicator = gr.HTML(update_step_indicator(0))

        with gr.Tabs(selected=0) as tabs:

            # Tab 0: 欢迎和系统检查
            with gr.Tab("🏠 开始使用", id="welcome"):
                with gr.Row():
                    with gr.Column(scale=2):
                        welcome_help = gr.Markdown(smart_help_system("welcome"))

                        system_check_btn = gr.Button(
                            "🔍 一键系统检查",
                            variant="primary",
                            size="lg"
                        )

                        system_status = gr.HTML("")

                    with gr.Column(scale=1):
                        gr.HTML("""
                        <div style='background: rgba(255, 255, 255, 0.9); padding: 25px; border-radius: 16px; text-align: center;'>
                            <h3 style='margin: 0 0 15px 0; color: #374151;'>🎯 快速导航</h3>
                            <div style='margin: 15px 0;'>
                                <div style='background: #f0f9ff; padding: 12px; border-radius: 8px; margin: 8px 0;'>
                                    <strong>🔍 系统检查</strong><br>
                                    <small>检查环境依赖</small>
                                </div>
                                <div style='background: #fef3c7; padding: 12px; border-radius: 8px; margin: 8px 0;'>
                                    <strong>📁 文件上传</strong><br>
                                    <small>上传视频和字幕</small>
                                </div>
                                <div style='background: #f3e8ff; padding: 12px; border-radius: 8px; margin: 8px 0;'>
                                    <strong>⚙️ 参数配置</strong><br>
                                    <small>设置处理参数</small>
                                </div>
                                <div style='background: #dcfce7; padding: 12px; border-radius: 8px; margin: 8px 0;'>
                                    <strong>🚀 开始处理</strong><br>
                                    <small>一键智能分析</small>
                                </div>
                            </div>
                        </div>
                        """)

            # Tab 1: 文件上传
            with gr.Tab("📁 文件上传", id="upload"):
                with gr.Row():
                    with gr.Column():
                        gr.Markdown("### 📹 视频文件")
                        video_input = gr.File(
                            label="拖拽或点击上传视频文件",
                            file_types=[".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".m4v"],
                            file_count="single",
                            height=150
                        )
                        video_status = gr.HTML("📋 请选择视频文件...")

                    with gr.Column():
                        gr.Markdown("### 📝 字幕文件")
                        subtitle_input = gr.File(
                            label="拖拽或点击上传字幕文件",
                            file_types=[".srt", ".ass", ".ssa", ".vtt"],
                            file_count="single",
                            height=150
                        )
                        subtitle_status = gr.HTML("📋 请选择字幕文件...")

                upload_help = gr.Markdown(smart_help_system("upload"))

                next_step_btn = gr.Button(
                    "➡️ 下一步：配置参数",
                    variant="primary",
                    size="lg",
                    visible=False
                )

            # Tab 2: 参数配置
            with gr.Tab("⚙️ 参数配置", id="config"):
                config_help = gr.Markdown(smart_help_system("config"))

                with gr.Row():
                    with gr.Column():
                        n_clusters = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=0,
                            step=1,
                            label="🗣️ 说话人数量 (0=自动检测)",
                            info="AI会自动分析最优的说话人数量",
                            interactive=True
                        )

                    with gr.Column():
                        uvr5_path = gr.Textbox(
                            label="🎵 UVR5路径 (可选但推荐)",
                            placeholder="例如: C:/UVR5/UVR5.exe",
                            info="配置后可显著提高语音分离质量",
                            lines=1
                        )

                start_btn = gr.Button(
                    "🚀 开始智能处理",
                    variant="primary",
                    size="lg"
                )

            # Tab 3: 处理过程
            with gr.Tab("🚀 处理过程", id="process"):
                processing_help = gr.Markdown(smart_help_system("processing"))

                status_output = gr.HTML("等待开始处理...")

                with gr.Row():
                    reset_btn = gr.Button("🔄 重置", variant="secondary")
                    cancel_btn = gr.Button("⏹️ 取消", variant="stop", visible=False)

            # Tab 4: 结果下载
            with gr.Tab("📥 处理结果", id="results"):
                gr.Markdown("### 🎉 恭喜！处理已完成")

                result_json = gr.JSON(label="📊 详细结果", visible=False)

                download_files = gr.File(
                    label="📁 下载处理后的文件",
                    file_count="multiple",
                    interactive=False,
                    height=200
                )

                gr.HTML("""
                <div style='background: rgba(255, 255, 255, 0.9); padding: 25px; border-radius: 16px; margin: 20px 0;'>
                    <h3 style='margin: 0 0 15px 0; color: #374151;'>📋 文件说明</h3>
                    <ul style='margin: 0; padding-left: 20px; color: #4b5563;'>
                        <li><strong>speaker_XX.wav</strong> - 各说话人的音频文件</li>
                        <li><strong>speaker_XX.txt</strong> - 对应的文字内容</li>
                        <li><strong>summary.txt</strong> - 处理摘要报告</li>
                        <li><strong>export_info.json</strong> - 详细的导出信息</li>
                    </ul>
                </div>
                """)

        # 状态变量
        result_state = gr.State(None)

        # 事件绑定
        def check_system():
            try:
                checker = SystemChecker()
                status = checker.check_all()

                html = "<div style='background: rgba(255, 255, 255, 0.95); padding: 20px; border-radius: 12px;'>"
                html += "<h3 style='margin: 0 0 15px 0; color: #374151;'>🔍 系统检查结果</h3>"

                all_good = True
                for component, info in status.items():
                    if isinstance(info, dict):
                        available = info.get('available', False)
                        icon = "✅" if available else "❌"
                        color = "#10b981" if available else "#ef4444"
                        if not available:
                            all_good = False
                    else:
                        icon = "ℹ️"
                        color = "#3b82f6"

                    html += f"<div style='margin: 10px 0; padding: 10px; background: {'#f0fdf4' if icon == '✅' else '#fef2f2' if icon == '❌' else '#eff6ff'}; border-radius: 8px;'>"
                    html += f"<span style='color: {color}; font-weight: bold;'>{icon} {component}:</span> "
                    html += f"<span style='color: #374151;'>{info.get('message', info) if isinstance(info, dict) else info}</span>"
                    html += "</div>"

                if all_good:
                    html += "<div style='margin-top: 20px; padding: 15px; background: #dcfce7; border-radius: 8px; text-align: center;'>"
                    html += "<span style='color: #166534; font-weight: bold;'>🎉 系统环境完美！可以开始使用了</span>"
                    html += "</div>"

                html += "</div>"

                return html

            except Exception as e:
                return user_friendly_error(e)

        # 文件验证事件
        video_input.upload(
            fn=lambda f: validate_file_realtime(f, "video"),
            inputs=video_input,
            outputs=video_status
        )

        subtitle_input.upload(
            fn=lambda f: validate_file_realtime(f, "subtitle"),
            inputs=subtitle_input,
            outputs=subtitle_status
        )

        # 系统检查
        system_check_btn.click(
            fn=check_system,
            outputs=system_status
        )

        # 处理逻辑
        start_btn.click(
            fn=process_with_enhanced_ux,
            inputs=[video_input, subtitle_input, n_clusters, uvr5_path],
            outputs=[status_output, result_state, download_files, tabs],
            show_progress="full"
        )

        # 结果显示
        result_state.change(
            fn=lambda x: gr.update(visible=x is not None, value=x),
            inputs=result_state,
            outputs=result_json
        )

    return demo

def main():
    """主函数"""
    print("🎭 动漫角色对话提取系统 (用户体验至上版)")
    print("=" * 60)
    print("✨ 专注极致用户体验")
    print("🎨 智能引导和反馈")
    print("⚡ 实时状态更新")
    print("💡 用户友好的错误处理")
    print("=" * 60)

    try:
        Config.init_folders()
        print("✅ 配置目录初始化完成")
    except Exception as e:
        print(f"❌ 配置目录初始化失败: {e}")

    demo = create_ux_enhanced_interface()

    demo.launch(
        server_name="127.0.0.1",
        server_port=28000,
        share=False,
        inbrowser=True,
        show_error=True,
        favicon_path=None,
        ssr_mode=True,
        show_api=False,
        quiet=False
    )

if __name__ == "__main__":
    main()