#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
配置管理器
负责管理所有的配置参数和默认值
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass, field
import json
from pathlib import Path


@dataclass
class Config:
    """
    配置类 - 管理所有配置参数
    """
    # 视频处理参数
    target_fps: int = 10
    min_frame_interval: float = 0.5
    output_resolution: tuple = (512, 512)
    
    # 采样策略选项
    sampling_strategy: str = "fps"     # "fps": 按帧率采样, "seconds": 按秒采样, "fps_ratio": 按帧率比例, "random_seconds": 随机秒采样
    frames_per_second: int = 1         # 按秒采样时每秒取几帧
    fps_ratio: float = 3.0             # 按帧率比例采样：原帧率/比例
    random_sample_rate: int = 4        # 随机采样：每秒候选帧数（已废弃，保留兼容性）
    random_select_count: int = 3       # 随机采样：每秒从所有帧中选择的数量
    preserve_original_resolution: bool = False  # 新增：是否保持原始分辨率
    max_resolution: tuple = (1920, 1080)  # 新增：最大分辨率限制

    # 相似度和阈值参数
    similarity_threshold: float = 0.9
    motion_threshold: float = 2.0
    shot_boundary_threshold: float = 0.5
    
    # 镜头检测参数
    min_shot_duration: int = 10          # 最小镜头长度（帧数）
    max_shot_duration: int = 3000        # 最大镜头长度（帧数）
    shot_merge_threshold: float = 0.8    # 镜头合并阈值
    enable_shot_filtering: bool = True   # 启用镜头过滤

    # 聚类参数
    cluster_eps: float = 0.15
    cluster_min_samples: int = 2
    max_keyframes_per_shot: int = 30

    # 模型相关
    use_transnet: bool = True
    clip_model_name: str = "ViT-B/32"
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    # GPU优化选项
    gpu_batch_size: int = 0  # 0=自动调整，>0=固定大小
    use_mixed_precision: bool = True  # 使用混合精度加速
    gpu_memory_fraction: float = 0.8  # GPU内存使用比例
    
    # 特征提取选项
    skip_motion_features: bool = False  # 跳过光流计算（加速处理）
    skip_color_features: bool = False   # 跳过颜色特征（加速处理）

    # 输出相关
    save_metadata: bool = True
    save_csv: bool = True
    image_quality: int = 95

    # 调试和日志
    verbose: bool = True
    log_level: str = "INFO"

    # 模型路径设置
    models_dir: Optional[str] = None  # 自定义模型目录，None表示使用项目目录

    # 其他选项
    extra_options: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """初始化后的验证和处理"""
        self._validate_config()

    def _validate_config(self):
        """验证配置参数的有效性"""
        if self.target_fps <= 0:
            raise ValueError("target_fps must be positive")

        if not 0 < self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")

        if self.cluster_eps <= 0:
            raise ValueError("cluster_eps must be positive")

        if self.max_keyframes_per_shot <= 0:
            raise ValueError("max_keyframes_per_shot must be positive")

        if self.image_quality < 1 or self.image_quality > 100:
            raise ValueError("image_quality must be between 1 and 100")

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'Config':
        """从字典创建配置"""
        return cls(**config_dict)

    @classmethod
    def from_file(cls, config_path: str) -> 'Config':
        """从文件加载配置"""
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(path, 'r', encoding='utf-8') as f:
            if path.suffix.lower() == '.json':
                config_dict = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")

        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = {}
        for key, value in self.__dict__.items():
            if not key.startswith('_'):
                result[key] = value
        return result

    def save_to_file(self, config_path: str):
        """保存配置到文件"""
        path = Path(config_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = self.to_dict()

        with open(path, 'w', encoding='utf-8') as f:
            if path.suffix.lower() == '.json':
                json.dump(config_dict, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported config file format: {path.suffix}")

    def update(self, **kwargs):
        """更新配置参数"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.extra_options[key] = value

        self._validate_config()

    def get_device(self) -> str:
        """获取实际使用的设备"""
        if self.device == "auto":
            try:
                import torch
                return "cuda" if torch.cuda.is_available() else "cpu"
            except ImportError:
                return "cpu"
        return self.device

    def copy(self) -> 'Config':
        """创建配置的深拷贝"""
        return Config.from_dict(self.to_dict())


class ConfigManager:
    """
    配置管理器 - 管理配置的加载、保存和验证
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化配置管理器

        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or Config()

    @classmethod
    def create_default(cls) -> 'ConfigManager':
        """创建使用默认配置的管理器"""
        return cls(Config())

    @classmethod
    def create_from_file(cls, config_path: str) -> 'ConfigManager':
        """从文件创建配置管理器"""
        config = Config.from_file(config_path)
        return cls(config)

    @classmethod
    def create_from_dict(cls, config_dict: Dict[str, Any]) -> 'ConfigManager':
        """从字典创建配置管理器"""
        config = Config.from_dict(config_dict)
        return cls(config)

    def get_config(self) -> Config:
        """获取配置对象"""
        return self.config

    def update_config(self, **kwargs):
        """更新配置"""
        self.config.update(**kwargs)

    def save_config(self, config_path: str):
        """保存配置到文件"""
        self.config.save_to_file(config_path)

    def create_preset_config(self, preset_name: str) -> Config:
        """
        创建预设配置

        Args:
            preset_name: 预设名称

        Returns:
            预设配置对象
        """
        presets = {
            "fast": {
                "target_fps": 5,
                "use_transnet": False,
                "cluster_eps": 0.2,
                "max_keyframes_per_shot": 3,
                "similarity_threshold": 0.85,
            },
            "balanced": {
                "target_fps": 10,
                "use_transnet": True,
                "cluster_eps": 0.15,
                "max_keyframes_per_shot": 5,
                "similarity_threshold": 0.9,
            },
            "quality": {
                "target_fps": 15,
                "use_transnet": True,
                "cluster_eps": 0.12,
                "max_keyframes_per_shot": 8,
                "similarity_threshold": 0.95,
                "output_resolution": (768, 768),
            },
            "detailed": {
                "target_fps": 20,
                "use_transnet": True,
                "cluster_eps": 0.1,
                "max_keyframes_per_shot": 10,
                "similarity_threshold": 0.97,
                "output_resolution": (1024, 1024),
            },
            "original": {
                "target_fps": 10,
                "use_transnet": True,
                "cluster_eps": 0.15,
                "max_keyframes_per_shot": 5,
                "similarity_threshold": 0.9,
                "preserve_original_resolution": True,
                "max_resolution": (1920, 1080),
                "image_quality": 100,
            }
        }

        if preset_name not in presets:
            available = ", ".join(presets.keys())
            raise ValueError(f"Unknown preset '{preset_name}'. Available: {available}")

        # 创建基础配置并更新预设参数
        config = Config()
        config.update(**presets[preset_name])

        return config

    def validate_for_video(self, video_path: str) -> bool:
        """
        验证配置是否适合指定的视频

        Args:
            video_path: 视频文件路径

        Returns:
            是否通过验证
        """
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return False

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            # 检查目标帧率是否合理
            if self.config.target_fps > fps:
                return False

            # 检查视频长度是否适合当前配置
            estimated_frames = duration * self.config.target_fps
            if estimated_frames > 10000:  # 太多帧可能导致内存问题
                return False

            return True

        except Exception:
            return False

    def optimize_for_video(self, video_path: str) -> Config:
        """
        为指定视频优化配置

        Args:
            video_path: 视频文件路径

        Returns:
            优化后的配置
        """
        try:
            import cv2

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            duration = frame_count / fps if fps > 0 else 0

            cap.release()

            # 创建优化配置
            optimized_config = self.config.copy()

            # 根据视频长度调整目标帧率
            if duration > 3600:  # 超过1小时
                optimized_config.target_fps = min(5, fps // 4)
            elif duration > 1800:  # 超过30分钟
                optimized_config.target_fps = min(8, fps // 3)
            else:
                optimized_config.target_fps = min(12, fps // 2)

            # 根据视频分辨率调整输出分辨率
            if width * height > 1920 * 1080:  # 高分辨率视频
                optimized_config.output_resolution = (768, 768)
            elif width * height < 640 * 480:  # 低分辨率视频
                optimized_config.output_resolution = (256, 256)

            # 根据视频长度调整关键帧数量
            if duration > 1800:  # 长视频减少关键帧
                optimized_config.max_keyframes_per_shot = max(3,
                    optimized_config.max_keyframes_per_shot - 2)

            return optimized_config

        except Exception:
            return self.config.copy()