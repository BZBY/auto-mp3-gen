#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速测试脚本
直接运行即可测试视频关键帧提取器的所有功能
"""

import sys
import os
import time
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))


def main():
    """快速测试主函数"""
    print("🎬 视频关键帧提取器 - 快速测试")
    print("=" * 50)

    # 检查测试视频
    video_path = "1.mp4"
    if not os.path.exists(video_path):
        print("❌ 测试视频文件 1.mp4 不存在")
        print("请确保 1.mp4 文件在项目根目录下")
        return

    print(f"✅ 找到测试视频: {video_path}")

    try:
        # 导入模块
        print("\n📦 导入模块...")
        from video_keyframe_extractor import KeyFrameExtractor, Config
        print("✅ 模块导入成功")

        # 创建配置（使用快速模式）
        print("\n⚙️ 创建配置...")
        config = Config(
            target_fps=5,           # 低帧率，快速测试
            use_transnet=True,      # 使用PyTorch版本的TransNetV2
            max_keyframes_per_shot=3,
            similarity_threshold=0.85,
            verbose=True
        )
        print("✅ 配置创建成功（包含TransNetV2 PyTorch版本）")

        # 设置输出目录
        output_dir = "test_output"
        print(f"\n📁 输出目录: {output_dir}")

        # 开始提取
        print("\n🚀 开始关键帧提取...")
        start_time = time.time()

        with KeyFrameExtractor(config) as extractor:
            # 显示模型信息
            model_info = extractor.get_model_info()
            print(f"📊 模型状态:")
            print(f"  CLIP模型: {'✅ 已加载' if model_info['clip']['loaded'] else '❌ 未加载'}")
            print(f"  设备: {model_info['clip']['device']}")
            print(f"  TransNetV2 PyTorch: {'✅ 可用并已加载' if model_info['transnet']['available'] and model_info['transnet']['loaded'] else '❌ 不可用'}")

            if model_info['transnet']['available'] and model_info['transnet']['loaded']:
                print(f"  镜头检测: 使用TransNetV2 PyTorch版本")
            else:
                print(f"  镜头检测: 将使用备用算法（直方图差异）")

            # 提取关键帧
            results = extractor.extract_keyframes(video_path, output_dir)

        end_time = time.time()
        processing_time = end_time - start_time

        # 显示结果
        print("\n🎉 提取完成！")
        print(f"⏱️ 处理时间: {processing_time:.2f} 秒")

        if results:
            print(f"📊 提取结果:")
            print(f"  关键帧数量: {len(results)}")
            print(f"  输出目录: {output_dir}")

            # 显示前几个关键帧信息
            print(f"\n📋 关键帧详情:")
            for i, result in enumerate(results[:5]):
                print(f"  {i+1}. {result['filename']}")
                print(f"     时间戳: {result['timestamp']:.2f}s")
                print(f"     置信度: {result['confidence']:.3f}")
                print(f"     镜头ID: {result['shot_id']}")

            if len(results) > 5:
                print(f"  ... 还有 {len(results) - 5} 个关键帧")

            # 检查输出文件
            output_path = Path(output_dir)
            if output_path.exists():
                image_files = list(output_path.glob("*.jpg"))
                metadata_files = list(output_path.glob("*.json")) + list(output_path.glob("*.csv")) + list(output_path.glob("*.txt"))

                print(f"\n📂 输出文件:")
                print(f"  图像文件: {len(image_files)} 个")
                print(f"  元数据文件: {len(metadata_files)} 个")

                # 显示一些文件名
                if image_files:
                    print(f"  示例图像: {image_files[0].name}")
                if metadata_files:
                    print(f"  元数据: {[f.name for f in metadata_files]}")

            print(f"\n✅ 测试成功完成！")
            print(f"📁 查看结果: {output_dir}")

        else:
            print("❌ 未提取到关键帧")

    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        print("请检查是否安装了所有依赖:")
        print("  pip install -r requirements.txt")
        print("  pip install git+https://github.com/openai/CLIP.git")

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        print(f"错误类型: {type(e).__name__}")

        # 提供一些常见问题的解决建议
        if "CLIP" in str(e):
            print("\n💡 可能的解决方案:")
            print("  1. 安装CLIP: pip install git+https://github.com/openai/CLIP.git")
            print("  2. 安装torch: pip install torch torchvision")

        elif "cv2" in str(e) or "opencv" in str(e):
            print("\n💡 可能的解决方案:")
            print("  1. 安装OpenCV: pip install opencv-python")

        elif "sklearn" in str(e):
            print("\n💡 可能的解决方案:")
            print("  1. 安装scikit-learn: pip install scikit-learn")

        elif "transnetv2" in str(e).lower():
            print("\n💡 可能的解决方案:")
            print("  1. 安装TransNetV2 PyTorch版本: pip install transnetv2-pytorch")
            print("  2. 或者禁用TransNetV2: 在配置中设置 use_transnet=False")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断测试")
    except Exception as e:
        print(f"\n❌ 脚本运行异常: {e}")

    print("\n" + "=" * 50)
    print("测试完成")

    # 在Windows下暂停，让用户看到结果
    if os.name == 'nt':
        input("\n按Enter键退出...")