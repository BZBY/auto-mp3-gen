#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型路径管理工具
集中管理所有模型的下载和缓存路径
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class ModelPathManager:
    """模型路径管理器"""

    def __init__(self, custom_models_dir: Optional[str] = None):
        """
        初始化模型路径管理器

        Args:
            custom_models_dir: 自定义模型目录路径，如果为None则使用项目目录
        """
        self.custom_models_dir = custom_models_dir
        self._setup_paths()

    def _setup_paths(self):
        """设置所有模型路径"""
        # 获取模型根目录
        if self.custom_models_dir:
            self.models_root = Path(self.custom_models_dir)
        else:
            # 默认使用项目目录下的models文件夹
            current_file = Path(__file__)
            project_root = current_file.parent.parent.parent.parent
            self.models_root = project_root / "models"

        # 确保根目录存在
        self.models_root.mkdir(parents=True, exist_ok=True)

        # 定义各个模型的缓存目录
        self.paths = {
            "torch": self.models_root / "torch",
            "clip": self.models_root / "clip",
            "huggingface": self.models_root / "huggingface",
            "transformers": self.models_root / "transformers",
            "transnetv2": self.models_root / "transnetv2",
            "tensorflow": self.models_root / "tensorflow",
            "tfhub": self.models_root / "tfhub"
        }

        # 创建所有目录
        for path_name, path_obj in self.paths.items():
            path_obj.mkdir(parents=True, exist_ok=True)

        logger.info(f"模型路径已设置到: {self.models_root}")

    def get_models_root(self) -> Path:
        """获取模型根目录"""
        return self.models_root

    def get_path(self, model_type: str) -> Path:
        """
        获取特定模型类型的路径

        Args:
            model_type: 模型类型 (torch, clip, huggingface, etc.)

        Returns:
            模型路径
        """
        if model_type not in self.paths:
            raise ValueError(f"Unknown model type: {model_type}")
        return self.paths[model_type]

    def setup_environment_variables(self):
        """设置环境变量"""
        env_vars = {
            'TORCH_HOME': str(self.paths['torch']),
            'HF_HOME': str(self.paths['huggingface']),
            'TRANSFORMERS_CACHE': str(self.paths['transformers']),
            'TFHUB_CACHE_DIR': str(self.paths['tfhub']),
            'TF_CPP_MIN_LOG_LEVEL': '2',  # 减少TensorFlow日志
        }

        for var_name, var_value in env_vars.items():
            os.environ[var_name] = var_value

        logger.info("环境变量已设置完成")

    def get_model_info(self) -> Dict:
        """
        获取模型目录信息

        Returns:
            包含路径和大小信息的字典
        """
        info = {
            "models_root": str(self.models_root),
            "paths": {name: str(path) for name, path in self.paths.items()},
            "sizes": {}
        }

        # 计算各目录大小
        for name, path in self.paths.items():
            if path.exists():
                size = self._calculate_directory_size(path)
                info["sizes"][name] = {
                    "size_mb": size / (1024 * 1024),
                    "size_gb": size / (1024 * 1024 * 1024)
                }
            else:
                info["sizes"][name] = {"size_mb": 0, "size_gb": 0}

        return info

    def _calculate_directory_size(self, directory: Path) -> int:
        """计算目录大小（字节）"""
        total_size = 0
        try:
            for file_path in directory.rglob("*"):
                if file_path.is_file():
                    total_size += file_path.stat().st_size
        except Exception as e:
            logger.warning(f"计算目录大小失败 {directory}: {e}")
        return total_size

    def clean_cache(self, model_type: Optional[str] = None,
                   confirm: bool = False) -> bool:
        """
        清理模型缓存

        Args:
            model_type: 要清理的模型类型，None表示全部
            confirm: 是否确认删除

        Returns:
            是否成功清理
        """
        if not confirm:
            logger.warning("清理缓存需要设置 confirm=True")
            return False

        try:
            import shutil

            if model_type:
                # 清理特定类型
                if model_type not in self.paths:
                    logger.error(f"未知的模型类型: {model_type}")
                    return False

                path_to_clean = self.paths[model_type]
                if path_to_clean.exists():
                    shutil.rmtree(path_to_clean)
                    path_to_clean.mkdir(parents=True, exist_ok=True)
                    logger.info(f"已清理 {model_type} 缓存: {path_to_clean}")
                else:
                    logger.info(f"{model_type} 缓存目录不存在")
            else:
                # 清理所有缓存
                if self.models_root.exists():
                    shutil.rmtree(self.models_root)
                    self._setup_paths()  # 重新创建目录
                    logger.info(f"已清理所有模型缓存: {self.models_root}")

            return True

        except Exception as e:
            logger.error(f"清理缓存失败: {e}")
            return False

    def check_model_files(self) -> Dict:
        """
        检查模型文件状态

        Returns:
            各模型文件的状态信息
        """
        status = {}

        # 检查CLIP模型
        clip_dir = self.paths['clip']
        status['clip'] = {
            'exists': clip_dir.exists(),
            'files': list(clip_dir.glob("*.pt")) if clip_dir.exists() else [],
            'size_mb': self._calculate_directory_size(clip_dir) / (1024 * 1024)
        }

        # 检查TransNetV2模型
        transnet_dir = self.paths['transnetv2']
        status['transnetv2'] = {
            'exists': transnet_dir.exists(),
            'files': list(transnet_dir.rglob("*")) if transnet_dir.exists() else [],
            'size_mb': self._calculate_directory_size(transnet_dir) / (1024 * 1024)
        }

        # 检查TensorFlow Hub缓存
        tfhub_dir = self.paths['tfhub']
        status['tfhub'] = {
            'exists': tfhub_dir.exists(),
            'files': list(tfhub_dir.rglob("*")) if tfhub_dir.exists() else [],
            'size_mb': self._calculate_directory_size(tfhub_dir) / (1024 * 1024)
        }

        return status

    def export_config(self, config_path: str):
        """
        导出路径配置到文件

        Args:
            config_path: 配置文件路径
        """
        import json

        config = {
            "models_root": str(self.models_root),
            "paths": {name: str(path) for name, path in self.paths.items()},
            "info": self.get_model_info()
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=2, ensure_ascii=False)

        logger.info(f"路径配置已导出到: {config_path}")

    @classmethod
    def from_config_file(cls, config_path: str) -> 'ModelPathManager':
        """
        从配置文件创建管理器

        Args:
            config_path: 配置文件路径

        Returns:
            模型路径管理器实例
        """
        import json

        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        models_root = config.get('models_root')
        return cls(custom_models_dir=models_root)


# 全局实例
_global_path_manager = None


def get_global_path_manager(custom_models_dir: Optional[str] = None) -> ModelPathManager:
    """
    获取全局路径管理器实例

    Args:
        custom_models_dir: 自定义模型目录

    Returns:
        模型路径管理器实例
    """
    global _global_path_manager

    if _global_path_manager is None:
        _global_path_manager = ModelPathManager(custom_models_dir)
        _global_path_manager.setup_environment_variables()

    return _global_path_manager


def setup_model_paths(custom_models_dir: Optional[str] = None):
    """
    快速设置模型路径

    Args:
        custom_models_dir: 自定义模型目录路径
    """
    path_manager = get_global_path_manager(custom_models_dir)
    logger.info(f"模型路径已设置: {path_manager.get_models_root()}")


def get_model_path(model_type: str) -> Path:
    """
    获取特定模型的路径

    Args:
        model_type: 模型类型

    Returns:
        模型路径
    """
    path_manager = get_global_path_manager()
    return path_manager.get_path(model_type)


def show_model_info():
    """显示模型信息"""
    path_manager = get_global_path_manager()
    info = path_manager.get_model_info()

    print("=" * 60)
    print("模型路径信息")
    print("=" * 60)
    print(f"模型根目录: {info['models_root']}")
    print()

    print("各模型目录:")
    for name, path in info['paths'].items():
        size_mb = info['sizes'][name]['size_mb']
        print(f"  {name:<12}: {path} ({size_mb:.1f} MB)")

    total_size = sum(s['size_mb'] for s in info['sizes'].values())
    print(f"\n总大小: {total_size:.1f} MB ({total_size/1024:.2f} GB)")
    print("=" * 60)