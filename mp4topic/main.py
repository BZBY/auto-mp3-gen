#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
视频关键帧提取器 - 主程序入口
简单易用的命令行接口
"""

import sys
import os
import argparse
from pathlib import Path

# 设置CUDA确定性算法环境变量（消除PyTorch警告）
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video_keyframe_extractor import KeyFrameExtractor, Config
from video_keyframe_extractor.core.config import ConfigManager


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="视频关键帧提取器 - 使用AI技术自动提取视频中的关键帧",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python main.py video.mp4                           # 基本使用，输出到默认目录
  python main.py video.mp4 -o output/                # 指定输出目录
  python main.py video.mp4 -p quality                # 使用质量预设
  python main.py video.mp4 -c config.json            # 使用配置文件
  python main.py video1.mp4 video2.mp4 -b            # 批量处理多个视频

预设配置:
  fast      - 快速模式，较低质量但处理速度快
  balanced  - 平衡模式，质量和速度的平衡（默认）
  quality   - 质量模式，更高质量但处理较慢
  detailed  - 详细模式，最高质量但处理最慢
  original  - 原图模式，保持原始分辨率，适合训练数据
        """
    )

    # 输入参数
    parser.add_argument(
        "videos",
        nargs="+",
        help="输入视频文件路径（可以指定多个）"
    )

    # 输出参数
    parser.add_argument(
        "-o", "--output",
        default="keyframes_output",
        help="输出目录路径 (默认: keyframes_output)"
    )

    # 配置参数
    parser.add_argument(
        "-p", "--preset",
        choices=["fast", "balanced", "quality", "detailed", "original"],
        default="balanced",
        help="预设配置 (默认: balanced)"
    )

    parser.add_argument(
        "-c", "--config",
        help="配置文件路径 (JSON格式)"
    )

    # 处理参数
    parser.add_argument(
        "-b", "--batch",
        action="store_true",
        help="批量处理模式，为每个视频创建独立的输出目录"
    )

    parser.add_argument(
        "--target-fps",
        type=int,
        help="目标采样帧率"
    )

    parser.add_argument(
        "--max-keyframes",
        type=int,
        help="每个镜头的最大关键帧数"
    )

    parser.add_argument(
        "--similarity-threshold",
        type=float,
        help="相似度去重阈值 (0-1)"
    )

    parser.add_argument(
        "--no-transnet",
        action="store_true",
        help="禁用TransNetV2镜头检测"
    )

    # 输出参数
    parser.add_argument(
        "--resolution",
        help="输出图像分辨率，格式: WxH (例如: 512x512)"
    )

    parser.add_argument(
        "--quality",
        type=int,
        help="JPEG图像质量 (1-100)"
    )

    parser.add_argument(
        "--preserve-original-resolution",
        action="store_true",
        help="保持原始分辨率（适合训练数据）"
    )

    parser.add_argument(
        "--max-resolution",
        help="最大分辨率限制，格式: WxH (例如: 1920x1080)"
    )

    # 模型参数
    parser.add_argument(
        "--models-dir",
        help="自定义模型存储目录（默认为项目目录下的models文件夹）"
    )

    # 其他参数
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="显示详细日志"
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="启用调试模式"
    )

    parser.add_argument(
        "--info",
        action="store_true",
        help="显示视频信息后退出"
    )

    return parser.parse_args()


def create_config_from_args(args):
    """根据命令行参数创建配置"""
    # 从预设开始
    if args.config:
        # 从配置文件加载
        config = Config.from_file(args.config)
    else:
        # 使用预设配置
        config_manager = ConfigManager()
        config = config_manager.create_preset_config(args.preset)

    # 应用命令行覆盖
    updates = {}

    if args.target_fps:
        updates["target_fps"] = args.target_fps

    if args.max_keyframes:
        updates["max_keyframes_per_shot"] = args.max_keyframes

    if args.similarity_threshold:
        updates["similarity_threshold"] = args.similarity_threshold

    if args.no_transnet:
        updates["use_transnet"] = False

    if args.resolution:
        try:
            w, h = args.resolution.split('x')
            updates["output_resolution"] = (int(w), int(h))
        except ValueError:
            print(f"❌ 无效的分辨率格式: {args.resolution}")
            sys.exit(1)

    if args.quality:
        updates["image_quality"] = args.quality

    if args.preserve_original_resolution:
        updates["preserve_original_resolution"] = True

    if args.max_resolution:
        try:
            w, h = args.max_resolution.split('x')
            updates["max_resolution"] = (int(w), int(h))
        except ValueError:
            print(f"❌ 无效的最大分辨率格式: {args.max_resolution}")
            sys.exit(1)

    if args.verbose:
        updates["verbose"] = True

    if args.debug:
        updates["verbose"] = True
        updates["log_level"] = "DEBUG"

    if args.models_dir:
        updates["models_dir"] = args.models_dir

    # 应用更新
    if updates:
        config.update(**updates)

    return config


def show_video_info(video_path):
    """显示视频信息"""
    try:
        import cv2

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"❌ 无法打开视频: {video_path}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0

        cap.release()

        print(f"📹 视频信息: {video_path}")
        print(f"  分辨率: {width} x {height}")
        print(f"  帧率: {fps:.2f} FPS")
        print(f"  总帧数: {frame_count}")
        print(f"  时长: {duration:.2f} 秒")
        print()

    except Exception as e:
        print(f"❌ 获取视频信息失败: {e}")


def process_single_video(video_path, output_dir, config):
    """处理单个视频"""
    print(f"🎬 处理视频: {video_path}")

    try:
        with KeyFrameExtractor(config) as extractor:
            results = extractor.extract_keyframes(video_path, output_dir)

            if results:
                print(f"✅ 成功提取 {len(results)} 个关键帧")
                print(f"📁 结果保存在: {output_dir}")

                # 显示统计信息
                shots_covered = len(set(r['shot_id'] for r in results))
                avg_confidence = sum(r['confidence'] for r in results) / len(results)

                print(f"📊 统计信息:")
                print(f"  覆盖镜头数: {shots_covered}")
                print(f"  平均置信度: {avg_confidence:.3f}")

                return True
            else:
                print("❌ 未提取到关键帧")
                return False

    except Exception as e:
        print(f"❌ 处理失败: {e}")
        return False


def process_batch_videos(video_paths, output_base_dir, config):
    """批量处理多个视频"""
    print(f"🚀 批量处理 {len(video_paths)} 个视频")

    successful = 0
    failed = 0

    try:
        with KeyFrameExtractor(config) as extractor:
            results = extractor.extract_keyframes_batch(video_paths, output_base_dir)

            for video_path, video_results in results.items():
                if video_results:
                    print(f"✅ {Path(video_path).name}: {len(video_results)} 个关键帧")
                    successful += 1
                else:
                    print(f"❌ {Path(video_path).name}: 处理失败")
                    failed += 1

    except Exception as e:
        print(f"❌ 批量处理失败: {e}")
        return False

    print(f"\n📊 批量处理完成:")
    print(f"  成功: {successful} 个视频")
    print(f"  失败: {failed} 个视频")

    return successful > 0


def main():
    """主函数"""
    print("🎬 视频关键帧提取器 v2.0.0")
    print("=" * 50)

    # 解析参数
    args = parse_arguments()

    # 验证输入文件
    valid_videos = []
    for video_path in args.videos:
        if Path(video_path).exists():
            valid_videos.append(video_path)
        else:
            print(f"⚠️ 视频文件不存在: {video_path}")

    if not valid_videos:
        print("❌ 没有有效的视频文件")
        sys.exit(1)

    # 显示视频信息模式
    if args.info:
        for video_path in valid_videos:
            show_video_info(video_path)
        return

    # 创建配置
    try:
        config = create_config_from_args(args)
        print(f"⚙️ 使用配置: {args.preset if not args.config else Path(args.config).name}")
        if args.verbose:
            print(f"  目标帧率: {config.target_fps}")
            print(f"  相似度阈值: {config.similarity_threshold}")
            print(f"  输出分辨率: {config.output_resolution}")
            print()
    except Exception as e:
        print(f"❌ 配置创建失败: {e}")
        sys.exit(1)

    # 处理视频
    success = False

    if args.batch or len(valid_videos) > 1:
        # 批量处理模式
        success = process_batch_videos(valid_videos, args.output, config)
    else:
        # 单个视频处理
        video_path = valid_videos[0]
        output_dir = args.output

        # 为单个视频创建专门的输出目录
        if not args.batch:
            video_name = Path(video_path).stem
            output_dir = Path(args.output) / f"{video_name}_keyframes"

        success = process_single_video(video_path, str(output_dir), config)

    # 退出状态
    if success:
        print("\n🎉 处理完成！")
        sys.exit(0)
    else:
        print("\n❌ 处理失败")
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 程序异常: {e}")
        sys.exit(1)