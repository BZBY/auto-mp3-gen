#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
动漫角色对话提取系统 - Gradio GUI
使用最新版本Gradio实现的Web界面
"""

import gradio as gr
import os
import sys
import traceback
import shutil
import tempfile
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

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
    'result': None
}

def system_check() -> str:
    """系统环境检查"""
    try:
        checker = SystemChecker()
        status = checker.check_all()

        report = "🔍 系统环境检查结果:\n\n"

        for component, info in status.items():
            if isinstance(info, dict):
                status_icon = "✅" if info.get('available', False) else "❌"
                report += f"{status_icon} {component}: {info.get('message', 'Unknown')}\n"
            else:
                report += f"ℹ️ {component}: {info}\n"

        return report
    except Exception as e:
        return f"❌ 系统检查失败: {str(e)}"

def validate_files(video_file, subtitle_file) -> Tuple[bool, str]:
    """验证上传的文件"""
    if video_file is None:
        return False, "请上传视频文件"

    if subtitle_file is None:
        return False, "请上传字幕文件"

    # 检查视频文件扩展名
    video_ext = Path(video_file.name).suffix.lower()
    if video_ext not in Config.ALLOWED_VIDEO_EXTENSIONS:
        return False, f"不支持的视频格式: {video_ext}\n支持的格式: {', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)}"

    # 检查字幕文件扩展名
    subtitle_ext = Path(subtitle_file.name).suffix.lower()
    if subtitle_ext not in Config.ALLOWED_SUBTITLE_EXTENSIONS:
        return False, f"不支持的字幕格式: {subtitle_ext}\n支持的格式: {', '.join(Config.ALLOWED_SUBTITLE_EXTENSIONS)}"

    return True, "文件验证通过"

def process_dialogue_extraction(
    video_file,
    subtitle_file,
    n_clusters: int,
    uvr5_path: str,
    progress=gr.Progress()
) -> Tuple[str, str, Optional[Dict], List[str]]:
    """处理对话提取的主函数"""
    global processor_instance, processing_status

    try:
        # 重置状态
        processing_status = {
            'is_processing': True,
            'current_step': 0,
            'total_steps': 6,
            'message': '正在初始化...',
            'result': None
        }

        # 验证文件
        valid, message = validate_files(video_file, subtitle_file)
        if not valid:
            return message, "", None, []

        progress(0.1, "验证文件...")

        # 创建临时目录来处理文件
        temp_dir = tempfile.mkdtemp()
        video_path = os.path.join(temp_dir, f"video{Path(video_file.name).suffix}")
        subtitle_path = os.path.join(temp_dir, f"subtitle{Path(subtitle_file.name).suffix}")

        # 复制文件到临时目录
        shutil.copy2(video_file.name, video_path)
        shutil.copy2(subtitle_file.name, subtitle_path)

        progress(0.2, "初始化处理器...")

        # 初始化处理器
        uvr5_path = uvr5_path.strip() if uvr5_path else None
        processor_instance = MainProcessor(uvr5_path)

        # 设置进度回调
        def progress_callback(percent, msg):
            processing_status['message'] = msg
            processing_status['current_step'] = int(percent / 100 * processing_status['total_steps'])
            progress(percent / 100, msg)

        processor_instance.set_progress_callback(progress_callback)

        progress(0.3, "开始处理...")

        # 处理
        result = processor_instance.process(
            video_path=video_path,
            subtitle_path=subtitle_path,
            n_clusters=n_clusters if n_clusters > 0 else None
        )

        processing_status['result'] = result
        processing_status['is_processing'] = False

        if result['success']:
            progress(1.0, "处理完成!")

            # 收集输出文件
            output_files = []
            output_dir = result.get('output_directory', '')
            if output_dir and os.path.exists(output_dir):
                for root, dirs, files in os.walk(output_dir):
                    for file in files:
                        file_path = os.path.join(root, file)
                        output_files.append(file_path)

            status_msg = f"""
✅ 处理成功完成!

📊 处理结果:
• 识别出 {result.get('speakers_count', 0)} 个说话人
• 总共 {result.get('total_segments', 0)} 个对话片段
• 输出目录: {output_dir}

💾 生成的文件:
• 导出信息: {result.get('export_info', {}).get('export_file', '未知')}
• 摘要文件: {result.get('summary_file', '未知')}
"""

            return status_msg, str(result), result, output_files
        else:
            error_msg = f"❌ 处理失败: {result.get('message', '未知错误')}"
            return error_msg, str(result), None, []

    except Exception as e:
        processing_status['is_processing'] = False
        error_msg = f"❌ 处理过程中发生错误: {str(e)}\n\n详细错误信息:\n{traceback.format_exc()}"
        return error_msg, "", None, []

    finally:
        # 清理临时文件
        try:
            if 'temp_dir' in locals() and os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
        except:
            pass

def reset_system() -> Tuple[str, str, None, List]:
    """重置系统状态"""
    global processing_status, processor_instance

    processing_status = {
        'is_processing': False,
        'current_step': 0,
        'total_steps': 6,
        'message': '',
        'result': None
    }

    processor_instance = None

    return "✅ 系统已重置", "", None, []

def create_interface():
    """创建Gradio界面"""

    # 自定义CSS样式
    custom_css = """
    .gradio-container {
        font-family: 'Microsoft YaHei', 'Segoe UI', Arial, sans-serif;
    }
    .status-box {
        background-color: #f8f9fa;
        border: 1px solid #dee2e6;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    """

    with gr.Blocks(
        title="🎭 动漫角色对话提取系统",
        theme=gr.themes.Soft(),
        css=custom_css
    ) as demo:

        # 标题和说明
        gr.Markdown("""
        # 🎭 动漫角色对话提取系统

        基于字幕的动漫角色对话提取系统，使用UVR5进行人声分离和说话人识别技术，
        自动从动漫视频中提取并分类不同角色的对话。

        ---
        """)

        with gr.Tabs():

            # Tab 1: 系统检查
            with gr.Tab("🔍 系统检查"):
                gr.Markdown("### 系统环境检查")
                gr.Markdown("检查系统环境和依赖项是否正确安装")

                system_check_btn = gr.Button("开始系统检查", variant="secondary")
                system_status_output = gr.Textbox(
                    label="系统状态",
                    lines=15,
                    interactive=False,
                    placeholder="点击'开始系统检查'查看系统状态..."
                )

                system_check_btn.click(
                    fn=system_check,
                    outputs=system_status_output
                )

            # Tab 2: 文件上传
            with gr.Tab("📁 文件上传"):
                gr.Markdown("### 上传视频和字幕文件")

                with gr.Row():
                    with gr.Column():
                        video_input = gr.File(
                            label="🎬 视频文件",
                            file_types=[".mp4", ".avi", ".mkv", ".mov", ".wmv", ".flv", ".m4v"],
                            file_count="single"
                        )
                        gr.Markdown("**支持格式:** MP4, AVI, MKV, MOV, WMV, FLV, M4V")

                    with gr.Column():
                        subtitle_input = gr.File(
                            label="📝 字幕文件",
                            file_types=[".srt", ".ass", ".ssa", ".vtt"],
                            file_count="single"
                        )
                        gr.Markdown("**支持格式:** SRT, ASS, SSA, VTT")

            # Tab 3: 参数配置
            with gr.Tab("⚙️ 参数配置"):
                gr.Markdown("### 处理参数配置")

                with gr.Row():
                    with gr.Column():
                        n_clusters = gr.Slider(
                            minimum=0,
                            maximum=10,
                            value=3,
                            step=1,
                            label="🗣️ 预期说话人数量",
                            info="设置为0则自动检测，建议根据实际角色数量设置"
                        )

                    with gr.Column():
                        uvr5_path = gr.Textbox(
                            label="🎵 UVR5路径 (可选)",
                            placeholder="例如: C:/UVR5/UVR5.exe (留空则跳过人声分离)",
                            info="Ultimate Vocal Remover 5 的安装路径，用于高质量人声分离"
                        )

            # Tab 4: 开始处理
            with gr.Tab("🚀 开始处理"):
                gr.Markdown("### 处理控制")

                with gr.Row():
                    process_btn = gr.Button(
                        "🚀 开始处理",
                        variant="primary",
                        size="lg"
                    )
                    reset_btn = gr.Button(
                        "🔄 重置系统",
                        variant="secondary"
                    )

                with gr.Row():
                    with gr.Column():
                        status_output = gr.Textbox(
                            label="📋 处理状态",
                            lines=15,
                            interactive=False,
                            placeholder="点击'开始处理'开始任务..."
                        )

                    with gr.Column():
                        result_json_output = gr.JSON(
                            label="📊 详细结果",
                            visible=False
                        )

            # Tab 5: 处理结果
            with gr.Tab("📥 处理结果"):
                gr.Markdown("### 下载处理结果")

                download_files = gr.File(
                    label="📁 下载文件",
                    file_count="multiple",
                    interactive=False
                )

                gr.Markdown("""
                **处理完成后，您可以下载以下文件：**
                - 各说话人的音频文件 (WAV格式)
                - 对应的字幕文本文件 (TXT格式)
                - 处理摘要和导出信息
                """)

        # 隐藏的状态变量
        result_state = gr.State(None)

        # 事件处理
        process_btn.click(
            fn=process_dialogue_extraction,
            inputs=[video_input, subtitle_input, n_clusters, uvr5_path],
            outputs=[status_output, result_json_output, result_state, download_files],
            show_progress=True
        )

        reset_btn.click(
            fn=reset_system,
            outputs=[status_output, result_json_output, result_state, download_files]
        )

        # 当有结果时显示JSON输出
        result_state.change(
            fn=lambda x: gr.update(visible=x is not None),
            inputs=result_state,
            outputs=result_json_output
        )

    return demo

def main():
    """主函数"""
    print("🎭 启动动漫角色对话提取系统 (Gradio版)")
    print("=" * 60)

    # 初始化配置目录
    try:
        Config.init_folders()
        print("✅ 配置目录初始化完成")
    except Exception as e:
        print(f"❌ 配置目录初始化失败: {e}")

    # 创建界面
    demo = create_interface()

    # 启动服务
    demo.launch(
        server_name="127.0.0.1",
        server_port=28000,
        share=False,
        inbrowser=True,
        show_error=True,
        favicon_path=None
    )

if __name__ == "__main__":
    main()