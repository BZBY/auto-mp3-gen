#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理工具
用于管理视频关键帧提取器的模型文件
"""

import sys
import argparse
from pathlib import Path

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video_keyframe_extractor.utils.model_paths import ModelPathManager, show_model_info


def show_info(args):
    """显示模型信息"""
    print("🤖 视频关键帧提取器 - 模型管理工具")
    print("=" * 60)

    if args.models_dir:
        path_manager = ModelPathManager(args.models_dir)
        print(f"使用自定义模型目录: {args.models_dir}")
    else:
        path_manager = ModelPathManager()
        print("使用默认项目模型目录")

    print()

    # 显示路径信息
    info = path_manager.get_model_info()
    print(f"📁 模型根目录: {info['models_root']}")
    print()

    print("📊 各模型目录状态:")
    print("-" * 50)

    for name, path in info['paths'].items():
        size_mb = info['sizes'][name]['size_mb']
        exists = "✅" if Path(path).exists() else "❌"
        print(f"  {exists} {name:<12}: {size_mb:>8.1f} MB - {path}")

    total_size = sum(s['size_mb'] for s in info['sizes'].values())
    print("-" * 50)
    print(f"📦 总大小: {total_size:.1f} MB ({total_size/1024:.2f} GB)")

    # 检查模型文件
    print("\n🔍 模型文件检查:")
    status = path_manager.check_model_files()

    for model_type, model_status in status.items():
        if model_status['exists'] and model_status['size_mb'] > 0:
            file_count = len([f for f in model_status['files'] if Path(f).is_file()])
            print(f"  ✅ {model_type}: {file_count} 个文件, {model_status['size_mb']:.1f} MB")
        else:
            print(f"  ❌ {model_type}: 未下载")


def clean_cache(args):
    """清理模型缓存"""
    if args.models_dir:
        path_manager = ModelPathManager(args.models_dir)
    else:
        path_manager = ModelPathManager()

    if not args.confirm:
        print("⚠️ 这将删除模型缓存文件，请使用 --confirm 确认操作")
        return

    print("🧹 清理模型缓存...")

    if args.type:
        # 清理特定类型
        success = path_manager.clean_cache(args.type, confirm=True)
        if success:
            print(f"✅ 已清理 {args.type} 缓存")
        else:
            print(f"❌ 清理 {args.type} 缓存失败")
    else:
        # 清理所有缓存
        success = path_manager.clean_cache(confirm=True)
        if success:
            print("✅ 已清理所有模型缓存")
        else:
            print("❌ 清理缓存失败")


def setup_custom_dir(args):
    """设置自定义模型目录"""
    if not args.directory:
        print("❌ 请指定模型目录路径")
        return

    custom_dir = Path(args.directory)

    # 创建目录
    try:
        custom_dir.mkdir(parents=True, exist_ok=True)
        print(f"📁 创建模型目录: {custom_dir}")
    except Exception as e:
        print(f"❌ 创建目录失败: {e}")
        return

    # 初始化模型路径管理器
    path_manager = ModelPathManager(str(custom_dir))

    print("✅ 自定义模型目录设置完成")
    print(f"模型将下载到: {path_manager.get_models_root()}")

    # 导出配置
    config_path = custom_dir / "model_paths.json"
    path_manager.export_config(str(config_path))
    print(f"配置已保存到: {config_path}")


def download_models(args):
    """预下载模型"""
    if args.models_dir:
        path_manager = ModelPathManager(args.models_dir)
    else:
        path_manager = ModelPathManager()

    # 设置环境变量
    path_manager.setup_environment_variables()

    print("📥 开始下载模型...")
    print(f"下载目录: {path_manager.get_models_root()}")

    try:
        # 下载CLIP模型
        if args.clip or args.all:
            print("\n🤖 下载CLIP模型...")
            try:
                import clip
                clip_path = path_manager.get_path('clip')
                model, preprocess = clip.load("ViT-B/32", download_root=str(clip_path))
                print("✅ CLIP模型下载完成")
                del model, preprocess  # 释放内存
            except Exception as e:
                print(f"❌ CLIP模型下载失败: {e}")

        # 下载TransNetV2模型
        if args.transnet or args.all:
            print("\n🎬 下载TransNetV2模型...")
            try:
                import os
                os.environ['TFHUB_CACHE_DIR'] = str(path_manager.get_path('tfhub'))

                from transnetv2 import TransNetV2
                model = TransNetV2()
                print("✅ TransNetV2模型下载完成")
                del model  # 释放内存
            except ImportError:
                print("❌ TransNetV2未安装，请先安装: pip install transnetv2")
            except Exception as e:
                print(f"❌ TransNetV2模型下载失败: {e}")

        print("\n🎉 模型下载完成！")

    except Exception as e:
        print(f"❌ 下载过程中出错: {e}")


def export_config(args):
    """导出配置"""
    if args.models_dir:
        path_manager = ModelPathManager(args.models_dir)
    else:
        path_manager = ModelPathManager()

    output_path = args.output or "model_config.json"
    path_manager.export_config(output_path)
    print(f"✅ 配置已导出到: {output_path}")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="视频关键帧提取器 - 模型管理工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python model_manager.py info                           # 显示模型信息
  python model_manager.py clean --confirm                # 清理所有缓存
  python model_manager.py clean --type clip --confirm    # 清理CLIP缓存
  python model_manager.py setup --directory /path/to/models  # 设置自定义目录
  python model_manager.py download --all                 # 下载所有模型
  python model_manager.py download --clip                # 仅下载CLIP模型
  python model_manager.py export --output config.json   # 导出配置
        """
    )

    # 全局参数
    parser.add_argument(
        "--models-dir",
        help="自定义模型目录路径"
    )

    # 子命令
    subparsers = parser.add_subparsers(dest="command", help="可用命令")

    # info命令
    info_parser = subparsers.add_parser("info", help="显示模型信息")

    # clean命令
    clean_parser = subparsers.add_parser("clean", help="清理模型缓存")
    clean_parser.add_argument(
        "--type",
        choices=["torch", "clip", "huggingface", "transformers", "transnetv2", "tfhub"],
        help="要清理的模型类型"
    )
    clean_parser.add_argument(
        "--confirm",
        action="store_true",
        help="确认删除操作"
    )

    # setup命令
    setup_parser = subparsers.add_parser("setup", help="设置自定义模型目录")
    setup_parser.add_argument(
        "--directory",
        required=True,
        help="自定义模型目录路径"
    )

    # download命令
    download_parser = subparsers.add_parser("download", help="预下载模型")
    download_parser.add_argument(
        "--all",
        action="store_true",
        help="下载所有模型"
    )
    download_parser.add_argument(
        "--clip",
        action="store_true",
        help="下载CLIP模型"
    )
    download_parser.add_argument(
        "--transnet",
        action="store_true",
        help="下载TransNetV2模型"
    )

    # export命令
    export_parser = subparsers.add_parser("export", help="导出配置")
    export_parser.add_argument(
        "--output",
        help="输出配置文件路径"
    )

    return parser.parse_args()


def main():
    """主函数"""
    args = parse_arguments()

    if not args.command:
        print("请指定命令，使用 --help 查看帮助")
        return

    try:
        if args.command == "info":
            show_info(args)
        elif args.command == "clean":
            clean_cache(args)
        elif args.command == "setup":
            setup_custom_dir(args)
        elif args.command == "download":
            download_models(args)
        elif args.command == "export":
            export_config(args)
        else:
            print(f"❌ 未知命令: {args.command}")

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断操作")
    except Exception as e:
        print(f"❌ 执行失败: {e}")


if __name__ == "__main__":
    main()