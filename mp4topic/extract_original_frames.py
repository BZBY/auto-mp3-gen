#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
原图质量关键帧提取脚本
专门用于提取训练数据的高质量关键帧
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


def create_original_quality_config():
    """创建原图质量配置"""
    config = Config(
        # 保持原始分辨率
        preserve_original_resolution=True,
        max_resolution=(1920, 1080),  # 限制最大尺寸防止内存问题
        
        # 高质量设置
        image_quality=100,  # 无损JPEG质量
        
        # 采样策略：随机秒采样，增加多样性
        sampling_strategy="random_seconds",  # 随机秒采样
        random_sample_rate=4,               # 每秒4个候选帧
        random_select_count=3,              # 每秒从所有帧中随机选3个
        
        # 启用所有检测方法
        use_transnet=True,
        
        # 使用更高精度的CLIP模型 - 针对训练数据优化
        clip_model_name="ViT-L/14",  # 最佳特征质量，适合训练数据
        
        # 聚类设置 - 回滚到保守配置并逐步优化
        cluster_eps=0.15,             # 回滚聚类距离
        max_keyframes_per_shot=30,    # 每镜头关键帧数
        similarity_threshold=0.92,    # 适中的相似度阈值，平衡去重和多样性
        motion_threshold=1.5,         # 适中的运动阈值
        
        # GPU性能优化
        gpu_batch_size=128,           # 大批次提高GPU利用率
        use_mixed_precision=True,     # 混合精度加速
        gpu_memory_fraction=0.9,      # 使用更多GPU内存
        
        # 特征提取设置
        skip_motion_features=False,   # 启用光流计算（CPU处理，但有助于质量）
        skip_color_features=True,     # 跳过颜色特征（影响较小）
        
        # 输出设置
        save_metadata=True,
        save_csv=True,
        verbose=True,
    )
    return config


def main():
    parser = argparse.ArgumentParser(
        description="原图质量关键帧提取器 - 专为训练数据设计",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python extract_original_frames.py                    # 自动使用1.mp4
  python extract_original_frames.py video.mp4          # 指定视频文件
  python extract_original_frames.py video.mp4 -o training_data/
  python extract_original_frames.py video.mp4 --max-frames 10
  python extract_original_frames.py *.mp4 --batch
        """
    )
    
    parser.add_argument(
        "videos",
        nargs="*",  # 改为可选参数
        help="输入视频文件路径（默认使用1.mp4）"
    )
    
    parser.add_argument(
        "-o", "--output",
        default="original_keyframes",
        help="输出目录 (默认: original_keyframes)"
    )
    
    parser.add_argument(
        "--max-frames",
        type=int,
        help="每个镜头最大关键帧数"
    )
    
    parser.add_argument(
        "--fps",
        type=int,
        help="目标采样帧率（仅在fps模式下有效）"
    )
    
    parser.add_argument(
        "--frames-per-second",
        type=int,
        help="每秒采样帧数 (默认: 1)"
    )
    
    parser.add_argument(
        "--fps-ratio",
        type=float,
        help="帧率比例 (默认: 3.0，即原帧率/3)"
    )
    
    parser.add_argument(
        "--random-sample-rate",
        type=int,
        help="随机采样：每秒候选帧数 (默认: 4)"
    )
    
    parser.add_argument(
        "--random-select-count", 
        type=int,
        help="随机采样：从候选帧中选择的数量 (默认: 3)"
    )
    
    parser.add_argument(
        "--sampling-fps-mode",
        action="store_true",
        help="使用传统的FPS采样模式"
    )
    
    parser.add_argument(
        "--max-resolution",
        help="最大分辨率限制 (格式: 1920x1080)"
    )
    
    parser.add_argument(
        "--batch",
        action="store_true",
        help="批量处理模式"
    )
    
    parser.add_argument(
        "--gpu-batch-size",
        type=int,
        help="GPU批处理大小 (默认: 128)"
    )
    
    parser.add_argument(
        "--no-mixed-precision",
        action="store_true",
        help="禁用混合精度加速"
    )
    
    # 镜头检测参数
    parser.add_argument(
        "--min-shot-duration",
        type=int,
        help="最小镜头长度（帧数，默认30）"
    )
    
    parser.add_argument(
        "--max-shot-duration", 
        type=int,
        help="最大镜头长度（帧数，默认3000）"
    )
    
    parser.add_argument(
        "--disable-shot-filtering",
        action="store_true",
        help="禁用镜头过滤（保留所有检测到的镜头）"
    )
    
    parser.add_argument(
        "--use-histogram-detection",
        action="store_true",
        help="使用直方图检测而不是TransNetV2"
    )
    
    parser.add_argument(
        "--skip-motion-features",
        action="store_true",
        help="跳过光流计算以加速处理（会影响效果）"
    )
    
    parser.add_argument(
        "--fast-mode",
        action="store_true",
        help="快速模式：跳过光流和颜色特征（速度优先）"
    )
    
    args = parser.parse_args()
    
    print("🎬 原图质量关键帧提取器")
    print("=" * 50)
    print("⚡ 特性：保持原始分辨率 + GPU加速 + 无损质量")
    print()
    
    # 处理视频文件参数
    video_files = args.videos
    if not video_files:
        # 如果没有指定视频文件，尝试使用默认的1.mp4
        default_video = "1.mp4"
        if Path(default_video).exists():
            video_files = [default_video]
            print(f"📹 使用默认视频文件: {default_video}")
        else:
            print("❌ 未指定视频文件，且默认的1.mp4不存在")
            print("请使用: python extract_original_frames.py <视频文件路径>")
            sys.exit(1)
    
    # 验证输入文件
    valid_videos = []
    for video_path in video_files:
        if Path(video_path).exists():
            valid_videos.append(video_path)
        else:
            print(f"⚠️ 视频文件不存在: {video_path}")
    
    if not valid_videos:
        print("❌ 没有有效的视频文件")
        sys.exit(1)
    
    # 创建原图质量配置
    config = create_original_quality_config()
    
    # 应用命令行参数
    if args.max_frames:
        config.max_keyframes_per_shot = args.max_frames
    
    if args.fps:
        config.target_fps = args.fps
        
    if args.frames_per_second:
        config.frames_per_second = args.frames_per_second
        config.sampling_strategy = "seconds"
        
    if args.fps_ratio:
        config.fps_ratio = args.fps_ratio
    
    if args.random_sample_rate:
        config.random_sample_rate = args.random_sample_rate
        
    if args.random_select_count:
        config.random_select_count = args.random_select_count
        
    if args.sampling_fps_mode:
        config.sampling_strategy = "fps"
        
    if args.max_resolution:
        try:
            w, h = args.max_resolution.split('x')
            config.max_resolution = (int(w), int(h))
        except ValueError:
            print(f"❌ 无效的分辨率格式: {args.max_resolution}")
            sys.exit(1)
    
    if args.gpu_batch_size:
        config.gpu_batch_size = args.gpu_batch_size
        
    if args.no_mixed_precision:
        config.use_mixed_precision = False
        
    if args.skip_motion_features:
        config.skip_motion_features = True
        
    if args.fast_mode:
        config.skip_motion_features = True
        config.skip_color_features = True
        print("🚀 快速模式已启用（跳过运动和颜色特征）")
    
    # 镜头检测参数
    if args.min_shot_duration:
        config.min_shot_duration = args.min_shot_duration
        
    if args.max_shot_duration:
        config.max_shot_duration = args.max_shot_duration
        
    if args.disable_shot_filtering:
        config.enable_shot_filtering = False
        
    if args.use_histogram_detection:
        config.use_transnet = False
    
    print(f"⚙️ 配置信息:")
    print(f"  保持原始分辨率: ✅")
    print(f"  最大分辨率限制: {config.max_resolution}")
    print(f"  图像质量: {config.image_quality}%")
    
    if config.sampling_strategy == "seconds":
        print(f"  采样策略: 按秒采样 ({config.frames_per_second} 帧/秒)")
    elif config.sampling_strategy == "fps_ratio":
        print(f"  采样策略: 按帧率比例 (原帧率/{config.fps_ratio})")
    elif config.sampling_strategy == "random_seconds":
        print(f"  采样策略: 随机秒采样 (每秒从{config.random_sample_rate}个候选中随机选{config.random_select_count}个)")
    else:
        print(f"  采样策略: 按帧率采样 ({config.target_fps} FPS)")
    
    print(f"  每镜头最大关键帧: {config.max_keyframes_per_shot}")
    print(f"  GPU批处理大小: {config.gpu_batch_size}")
    print(f"  混合精度加速: {'✅' if config.use_mixed_precision else '❌'}")
    print(f"  光流计算: {'✅' if not config.skip_motion_features else '❌ (跳过以加速)'}")
    print(f"  颜色特征: {'✅' if not config.skip_color_features else '❌ (跳过以加速)'}")
    
    print(f"🎬 镜头检测设置:")
    print(f"  检测方法: {'TransNetV2 (AI)' if config.use_transnet else '直方图差异'}")
    print(f"  最小镜头长度: {config.min_shot_duration}帧")
    print(f"  最大镜头长度: {config.max_shot_duration}帧") 
    print(f"  镜头过滤: {'✅' if config.enable_shot_filtering else '❌ (保留所有)'}")
    print()
    
    try:
        with KeyFrameExtractor(config) as extractor:
            if args.batch or len(valid_videos) > 1:
                # 批量处理
                print(f"🚀 批量处理 {len(valid_videos)} 个视频")
                results = extractor.extract_keyframes_batch(valid_videos, args.output)
                
                successful = sum(1 for video_results in results.values() if video_results)
                failed = len(results) - successful
                
                print(f"\n📊 批量处理完成:")
                print(f"  成功: {successful} 个视频")
                print(f"  失败: {failed} 个视频")
                
            else:
                # 单个视频处理
                video_path = valid_videos[0]
                video_name = Path(video_path).stem
                output_dir = Path(args.output) / f"{video_name}_original_keyframes"
                
                print(f"🎬 处理视频: {video_path}")
                results = extractor.extract_keyframes(video_path, str(output_dir))
                
                if results:
                    print(f"✅ 成功提取 {len(results)} 个原图质量关键帧")
                    print(f"📁 结果保存在: {output_dir}")
                    
                    # 显示统计信息
                    shots_covered = len(set(r['shot_id'] for r in results))
                    avg_confidence = sum(r['confidence'] for r in results) / len(results)
                    
                    print(f"📊 统计信息:")
                    print(f"  覆盖镜头数: {shots_covered}")
                    print(f"  平均置信度: {avg_confidence:.3f}")
                    print(f"  图像质量: 100% (无损)")
                else:
                    print("❌ 未提取到关键帧")
                    sys.exit(1)
        
        print("\n🎉 处理完成！关键帧已保存为原图质量")
        
    except Exception as e:
        print(f"\n❌ 处理失败: {e}")
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
