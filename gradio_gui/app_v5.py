#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动漫角色对话提取系统 - Gradio 5.x GUI
使用最新版本Gradio 5.x实现的现代化Web界面
支持SSR、低延迟流式传输、现代主题等新特性
"""

import gradio as gr
import os
import sys
import traceback
import shutil
import tempfile
import asyncio
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any, Generator
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

# 全局变量
processor_instance = None
processing_status = {
    'is_processing': False,
    'current_step': 0,
    'total_steps': 6,
    'message': '',
    'result': None,
    'progress': 0
}

def system_check() -> str:
    """系统环境检查"""
    try:
        checker = SystemChecker()
        status = checker.check_all()

        report = "🔍 **系统环境检查结果:**\n\n"

        for component, info in status.items():
            if isinstance(info, dict):
                status_icon = "✅" if info.get('available', False) else "❌"
                report += f"{status_icon} **{component}**: {info.get('message', 'Unknown')}\n"
            else:
                report += f"ℹ️ **{component}**: {info}\n"

        return report
    except Exception as e:
        return f"❌ **系统检查失败**: {str(e)}"

def validate_files(video_file, subtitle_file) -> Tuple[bool, str]:
    """验证上传的文件"""
    if video_file is None:
        return False, "❌ 请上传视频文件"

    if subtitle_file is None:
        return False, "❌ 请上传字幕文件"

    # 检查视频文件扩展名
    video_ext = Path(video_file.name).suffix.lower()
    if video_ext not in Config.ALLOWED_VIDEO_EXTENSIONS:
        return False, f"❌ 不支持的视频格式: {video_ext}\n✅ 支持的格式: {', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)}"

    # 检查字幕文件扩展名
    subtitle_ext = Path(subtitle_file.name).suffix.lower()
    if subtitle_ext not in Config.ALLOWED_SUBTITLE_EXTENSIONS:
        return False, f"❌ 不支持的字幕格式: {subtitle_ext}\n✅ 支持的格式: {', '.join(Config.ALLOWED_SUBTITLE_EXTENSIONS)}"

    return True, "✅ 文件验证通过"

def process_dialogue_extraction(
    video_file,
    subtitle_file,
    n_clusters: int,
    uvr5_path: str,
    progress: gr.Progress = gr.Progress(track_tqdm=True)
) -> Tuple[str, Dict, List[str]]:
    """处理对话提取的主函数 - 使用Gradio 5.x的最新Progress API"""
    global processor_instance, processing_status

    try:
        # 重置状态
        processing_status = {
            'is_processing': True,
            'current_step': 0,
            'total_steps': 6,
            'message': '正在初始化...',
            'result': None,
            'progress': 0
        }

        # 验证文件
        valid, message = validate_files(video_file, subtitle_file)
        if not valid:
            return message, {}, []

        progress(0.1, desc="验证文件...")

        # 创建临时目录来处理文件
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, f"video{Path(video_file.name).suffix}")
        subtitle_path = os.path.join(temp_dir, f"subtitle{Path(subtitle_file.name).suffix}")

        # 复制文件到临时目录
        shutil.copy2(video_file.name, video_path)
        shutil.copy2(subtitle_file.name, subtitle_path)

        progress(0.2, desc="初始化处理器...")

        # 初始化处理器
        uvr5_path = uvr5_path.strip() if uvr5_path else None
        processor_instance = MainProcessor(uvr5_path)

        # 设置进度回调 - 使用Gradio 5.x的新式progress API
        def progress_callback(percent, msg):
            processing_status['message'] = msg
            processing_status['current_step'] = int(percent / 100 * processing_status['total_steps'])
            processing_status['progress'] = percent
            progress(percent / 100, desc=msg)

        processor_instance.set_progress_callback(progress_callback)

        progress(0.3, desc="开始处理...")

        # 处理
        result = processor_instance.process(
            video_path=video_path,
            subtitle_path=subtitle_path,
            n_clusters=n_clusters if n_clusters > 0 else None
        )

        processing_status['result'] = result
        processing_status['is_processing'] = False

        if result['success']:
            progress(1.0, desc="处理完成!")

            # 收集输出文件
            output_files = []
            output_dir = result.get('output_directory', '')
            if output_dir and os.path.exists(output_dir):
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        output_files.append(file_path)

            status_msg = f"""## ✅ 处理成功完成!

### 📊 处理结果:
- **识别出说话人**: {result.get('speakers_count', 0)} 个
- **总对话片段**: {result.get('total_segments', 0)} 个
- **输出目录**: `{output_dir}`

### 💾 生成的文件:
- **导出信息**: {result.get('export_info', {}).get('export_file', '未知')}
- **摘要文件**: {result.get('summary_file', '未知')}

> 🎉 所有文件已准备就绪，可以在下方下载！
"""

            return status_msg, result, output_files
        else:
            error_msg = f"## ❌ 处理失败\n\n**错误信息**: {result.get('message', '未知错误')}"
            return error_msg, {}, []

    except Exception as e:
        processing_status['is_processing'] = False
        error_msg = f"""## ❌ 处理过程中发生错误

**错误信息**: {str(e)}

### 详细错误信息:
```
{traceback.format_exc()}
```
"""
        return error_msg, {}, []

    finally:
        # 清理临时文件
        try:
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass

def reset_system() -> Tuple[str, Dict, List]:
    """重置系统状态"""
    global processing_status, processor_instance

    processing_status = {
        'is_processing': False,
        'current_step': 0,
        'total_steps': 6,
        'message': '',
        'result': None,
        'progress': 0
    }

    processor_instance = None

    return "## ✅ 系统已重置\n\n系统状态已清空，可以开始新的处理任务。", {}, []

def create_interface():
    """创建Gradio 5.x界面"""

    # 使用Gradio 5.x的新主题系统
    theme = gr.themes.Soft(
        primary_hue="blue",
        secondary_hue="slate",
        neutral_hue="slate",
    ).set(
        body_background_fill="#f8fafc",
        body_text_color="#1e293b",
        button_primary_background_fill="#3b82f6",
        button_primary_text_color="white",
    )

    with gr.Blocks(
        title="🎭 动漫角色对话提取系统",
        theme=theme,
        fill_height=True,  # Gradio 5.x新特性
        css="""
        .gradio-container {
            font-family: 'Microsoft YaHei', 'Segoe UI', system-ui, sans-serif;
        }
        .gr-button {
            transition: all 0.2s ease;
        }
        .gr-button:hover {
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
        }
        .status-success {
            border-left: 4px solid #10b981;
            background: #f0fdf4;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        .status-error {
            border-left: 4px solid #ef4444;
            background: #fef2f2;
            padding: 1rem;
            border-radius: 0.5rem;
        }
        """
    ) as demo:

        # 使用Gradio 5.x的新Markdown组件
        gr.Markdown("""
        # 🎭 动漫角色对话提取系统

        **基于最新Gradio 5.x构建** | 支持SSR快速加载 | 现代化UI设计

        使用UVR5进行人声分离和说话人识别技术，自动从动漫视频中提取并分类不同角色的对话。

        ---
        """)

        # 使用新的Tab系统，支持更好的布局
        with gr.Tabs(selected=0) as tabs:

            # Tab 1: 系统检查
            with gr.Tab("🔍 系统检查", id="system"):
                gr.Markdown("### 系统环境检查")
                gr.Markdown("检查系统环境和依赖项是否正确安装")

                with gr.Row():
                    system_check_btn = gr.Button(
                        "🔍 开始系统检查",
                        variant="secondary",
                        size="lg"
                    )

                # 使用新的组件样式
                system_status_output = gr.Markdown(
                    value="点击上方按钮开始系统检查...",
                    container=True,
                    line_breaks=True
                )

                system_check_btn.click(
                    fn=system_check,
                    outputs=system_status_output
                )

            # Tab 2: 文件上传
            with gr.Tab("📁 文件上传", id="upload"):
                gr.Markdown("### 上传视频和字幕文件")

                with gr.Row():
                    with gr.Column():
                        video_input = gr.File(
                            label="🎬 视频文件",
                            file_types=[".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".m4v"],
                            file_count="single",
                            height=200  # Gradio 5.x支持高度设置
                        )
                        gr.Markdown("**支持格式**: MP4, AVI, MKV, MOV, WMV, FLV, M4V")

                    with gr.Column():
                        subtitle_input = gr.File(
                            label="📝 字幕文件",
                            file_types=[".srt", ".ass", ".ssa", ".vtt"],
                            file_count="single",
                            height=200
                        )
                        gr.Markdown("**支持格式**: SRT, ASS, SSA, VTT")

            # Tab 3: 参数配置
            with gr.Tab("⚙️ 参数配置", id="config"):
                gr.Markdown("### 处理参数配置")

                with gr.Row():
                    with gr.Column():
                        n_clusters = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=3,
                            step=1,
                            label="🗣️ 预期说话人数量",
                            info="设置为0则自动检测，建议根据实际角色数量设置",
                            interactive=True
                        )

                    with gr.Column():
                        uvr5_path = gr.Textbox(
                            label="🎵 UVR5路径 (可选)",
                            placeholder="例如: C:/UVR5/UVR5.exe",
                            info="Ultimate Vocal Remover 5 的安装路径，用于高质量人声分离",
                            lines=1,
                            max_lines=1
                        )

                gr.Markdown("""
                > **提示**: 如果不配置UVR5路径，系统将跳过人声分离步骤，直接使用原始音频进行说话人识别。
                > 配置UVR5可以显著提高语音识别的准确性。
                """)

            # Tab 4: 开始处理
            with gr.Tab("🚀 开始处理", id="process"):
                gr.Markdown("### 处理控制")

                with gr.Row():
                    process_btn = gr.Button(
                        "🚀 开始处理",
                        variant="primary",
                        size="lg",
                        scale=2
                    )
                    reset_btn = gr.Button(
                        "🔄 重置系统",
                        variant="secondary",
                        size="lg",
                        scale=1
                    )

                # 状态显示区域
                with gr.Row():
                    status_output = gr.Markdown(
                        value="### 📋 等待开始处理...\n\n点击上方'开始处理'按钮启动任务。",
                        container=True,
                        line_breaks=True
                    )

            # Tab 5: 处理结果
            with gr.Tab("📥 处理结果", id="results"):
                gr.Markdown("### 下载处理结果")

                # 结果详情
                result_json_output = gr.JSON(
                    label="📊 详细结果",
                    container=True,
                    visible=False
                )

                # 文件下载
                download_files = gr.File(
                    label="📁 下载文件",
                    file_count="multiple",
                    interactive=False,
                    height=300
                )

                gr.Markdown("""
                ### 📋 处理完成后，您可以下载以下文件：

                - 🎵 **各说话人的音频文件** (WAV格式)
                - 📝 **对应的字幕文本文件** (TXT格式)
                - 📊 **处理摘要和导出信息**
                - 📈 **详细的统计报告**
                """)

        # 隐藏的状态变量
        result_state = gr.State(None)

        # 事件处理 - 使用Gradio 5.x的新事件系统
        process_btn.click(
            fn=process_dialogue_extraction,
            inputs=[video_input, subtitle_input, n_clusters, uvr5_path],
            outputs=[status_output, result_state, download_files],
            show_progress="full",  # Gradio 5.x新特性：显示完整进度条
            scroll_to_output=True
        )

        reset_btn.click(
            fn=reset_system,
            outputs=[status_output, result_state, download_files]
        )

        # 当有结果时显示JSON输出并自动切换到结果tab
        def update_result_display(result):
            if result:
                return gr.update(visible=True, value=result), gr.update(selected="results")
            else:
                return gr.update(visible=False), gr.update()

        result_state.change(
            fn=update_result_display,
            inputs=result_state,
            outputs=[result_json_output, tabs]
        )

    return demo

def main():
    """主函数"""
    print("🎭 动漫角色对话提取系统 (Gradio 5.x版)")
    print("=" * 60)
    print("✨ 支持SSR快速加载")
    print("🎨 现代化UI设计")
    print("⚡ 低延迟流式传输")
    print("=" * 60)

    # 初始化配置目录
    try:
        Config.init_folders()
        print("✅ 配置目录初始化完成")
    except Exception as e:
        print(f"❌ 配置目录初始化失败: {e}")

    # 创建界面
    demo = create_interface()

    # 启动服务 - 使用Gradio 5.x的新特性
    demo.launch(
        server_name="127.0.0.1",
        server_port=28000,
        share=False,
        inbrowser=True,
        show_error=True,
        favicon_path=None,
        ssr_mode=True,  # 启用服务端渲染
        show_api=False,  # 隐藏API文档
        quiet=False
    )

if __name__ == "__main__":
    main()