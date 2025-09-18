#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
模型管理器
负责加载和管理CLIP和TransNetV2模型
"""

import logging
import os
from typing import Optional, Tuple, Any, Dict, List
import torch
import numpy as np
from pathlib import Path

from ..core.config import Config
from ..utils.model_paths import get_global_path_manager, get_model_path

logger = logging.getLogger(__name__)


class CLIPModelManager:
    """CLIP模型管理器"""

    def __init__(self, config: Config):
        """
        初始化CLIP模型管理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.device = config.get_device()
        self.model = None
        self.preprocess = None
        self._is_loaded = False

        # 获取路径管理器
        self.path_manager = get_global_path_manager(config.models_dir)

    def load_model(self) -> bool:
        """
        加载CLIP模型

        Returns:
            是否加载成功
        """
        try:
            import clip

            logger.info(f"Loading CLIP model: {self.config.clip_model_name}")

            # 获取CLIP模型下载路径
            clip_cache_dir = self.path_manager.get_path('clip')

            # 使用自定义下载路径加载CLIP模型
            self.model, self.preprocess = clip.load(
                self.config.clip_model_name,
                device=self.device,
                download_root=str(clip_cache_dir)
            )

            # 设置为评估模式
            self.model.eval()
            self._is_loaded = True

            logger.info(f"CLIP model loaded successfully on device: {self.device}")
            return True

        except ImportError as e:
            logger.error(f"CLIP library not found: {e}")
            logger.error("Please install CLIP: pip install git+https://github.com/openai/CLIP.git")
            return False

        except Exception as e:
            logger.error(f"Failed to load CLIP model: {e}")
            return False

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded

    def encode_image(self, image: torch.Tensor) -> Optional[np.ndarray]:
        """
        编码图像为特征向量

        Args:
            image: 预处理后的图像张量

        Returns:
            特征向量，如果失败返回None
        """
        if not self._is_loaded:
            logger.error("CLIP model not loaded")
            return None

        try:
            with torch.no_grad():
                features = self.model.encode_image(image)
                # 转换为numpy并归一化
                features = features.cpu().numpy()
                features = features / np.linalg.norm(features, axis=1, keepdims=True)
                return features

        except Exception as e:
            logger.error(f"Failed to encode image: {e}")
            return None

    def preprocess_image(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """
        预处理图像

        Args:
            image: BGR格式的图像数组

        Returns:
            预处理后的张量，如果失败返回None
        """
        if not self._is_loaded:
            logger.error("CLIP model not loaded")
            return None

        try:
            import cv2
            from PIL import Image

            # BGR转RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # 转换为PIL Image
            pil_image = Image.fromarray(image_rgb)
            # 预处理
            processed = self.preprocess(pil_image).unsqueeze(0).to(self.device)
            return processed

        except Exception as e:
            logger.error(f"Failed to preprocess image: {e}")
            return None

    def unload_model(self):
        """卸载模型释放内存"""
        if self._is_loaded:
            del self.model
            del self.preprocess
            self.model = None
            self.preprocess = None
            self._is_loaded = False

            # 清理GPU缓存
            if self.device == "cuda":
                torch.cuda.empty_cache()

            logger.info("CLIP model unloaded")


class TransNetV2ModelManager:
    """TransNetV2模型管理器"""

    def __init__(self, config: Config):
        """
        初始化TransNetV2模型管理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.model = None
        self._is_loaded = False
        self._available = self._check_availability()

        # 获取路径管理器
        self.path_manager = get_global_path_manager(config.models_dir)

    def _check_availability(self) -> bool:
        """检查TransNetV2是否可用"""
        try:
            # 尝试导入PyTorch版本的TransNetV2
            from transnetv2_pytorch import TransNetV2
            return True
        except ImportError:
            logger.warning("TransNetV2 PyTorch not available. Install with: pip install transnetv2-pytorch")
            return False

    def is_available(self) -> bool:
        """检查TransNetV2是否可用"""
        return self._available

    def load_model(self) -> bool:
        """
        加载TransNetV2模型

        Returns:
            是否加载成功
        """
        if not self._available:
            logger.error("TransNetV2 not available")
            return False

        try:
            from transnetv2_pytorch import TransNetV2

            logger.info("Loading TransNetV2 PyTorch model...")

            # 初始化PyTorch版本的TransNetV2
            # 使用auto设备选择，让模型自动选择最佳设备
            self.model = TransNetV2(device='auto')
            self.model.eval()  # 设置为评估模式

            self._is_loaded = True

            logger.info("TransNetV2 PyTorch model loaded successfully")
            return True

        except Exception as e:
            logger.error(f"Failed to load TransNetV2 model: {e}")
            return False

    def is_loaded(self) -> bool:
        """检查模型是否已加载"""
        return self._is_loaded

    def predict_video(self, video_path: str) -> Optional[Dict[str, Any]]:
        """
        预测视频的镜头边界（PyTorch版本）

        Args:
            video_path: 视频文件路径

        Returns:
            包含场景信息的字典，如果失败返回None
        """
        if not self._is_loaded:
            logger.error("TransNetV2 model not loaded")
            return None

        try:
            import torch

            logger.info(f"Analyzing video with TransNetV2 PyTorch: {video_path}")

            with torch.no_grad():
                # 使用PyTorch版本的detect_scenes方法
                scenes = self.model.detect_scenes(video_path)

                # 构建与原版API兼容的返回格式
                result = {
                    'scenes': scenes,
                    'total_scenes': len(scenes),
                    'video_path': video_path
                }

            logger.info(f"TransNetV2 analysis completed, found {len(scenes)} scenes")
            return result

        except Exception as e:
            logger.error(f"Failed to predict video with TransNetV2: {e}")
            return None

    def get_scenes_from_result(self, result: Dict[str, Any]) -> Optional[List[Tuple[int, int]]]:
        """
        从PyTorch版本的结果中提取场景边界

        Args:
            result: predict_video返回的结果字典

        Returns:
            场景边界列表 [(start, end), ...]，如果失败返回None
        """
        if not result or 'scenes' not in result:
            logger.error("Invalid result format")
            return None

        try:
            scenes = result['scenes']
            # 转换为原版API格式的(start, end)元组列表
            scene_boundaries = []

            for scene in scenes:
                # PyTorch版本返回的场景格式可能包含start_time, end_time等
                if isinstance(scene, dict):
                    # 假设场景包含start_frame和end_frame字段
                    start_frame = scene.get('start_frame', 0)
                    end_frame = scene.get('end_frame', 0)
                    scene_boundaries.append((start_frame, end_frame))
                elif isinstance(scene, (list, tuple)) and len(scene) >= 2:
                    # 如果是元组或列表格式
                    scene_boundaries.append((scene[0], scene[1]))

            return scene_boundaries

        except Exception as e:
            logger.error(f"Failed to extract scenes from result: {e}")
            return None

    def unload_model(self):
        """卸载模型释放内存"""
        if self._is_loaded:
            del self.model
            self.model = None
            self._is_loaded = False
            logger.info("TransNetV2 model unloaded")


class ModelManager:
    """
    统一模型管理器
    管理所有模型的加载、卸载和使用
    """

    def __init__(self, config: Config):
        """
        初始化模型管理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.clip_manager = CLIPModelManager(config)
        self.transnet_manager = TransNetV2ModelManager(config)

    def initialize_models(self) -> bool:
        """
        初始化所有需要的模型

        Returns:
            是否初始化成功
        """
        success = True

        # 加载CLIP模型
        if not self.clip_manager.load_model():
            logger.error("Failed to load CLIP model")
            success = False

        # 加载TransNetV2模型（如果配置启用且可用）
        if self.config.use_transnet:
            if self.transnet_manager.is_available():
                if not self.transnet_manager.load_model():
                    logger.warning("Failed to load TransNetV2, will use fallback method")
            else:
                logger.warning("TransNetV2 not available, will use fallback method")

        return success

    def get_clip_manager(self) -> CLIPModelManager:
        """获取CLIP模型管理器"""
        return self.clip_manager

    def get_transnet_manager(self) -> TransNetV2ModelManager:
        """获取TransNetV2模型管理器"""
        return self.transnet_manager

    def is_ready(self) -> bool:
        """检查模型是否准备就绪"""
        # 至少需要CLIP模型加载成功
        return self.clip_manager.is_loaded()

    def get_model_info(self) -> dict:
        """
        获取模型信息

        Returns:
            包含模型状态的字典
        """
        return {
            "clip": {
                "loaded": self.clip_manager.is_loaded(),
                "model_name": self.config.clip_model_name,
                "device": self.config.get_device()
            },
            "transnet": {
                "available": self.transnet_manager.is_available(),
                "loaded": self.transnet_manager.is_loaded(),
                "enabled": self.config.use_transnet
            }
        }

    def cleanup(self):
        """清理所有模型"""
        logger.info("Cleaning up models...")
        self.clip_manager.unload_model()
        self.transnet_manager.unload_model()
        logger.info("Model cleanup completed")

    def __enter__(self):
        """上下文管理器入口"""
        self.initialize_models()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()