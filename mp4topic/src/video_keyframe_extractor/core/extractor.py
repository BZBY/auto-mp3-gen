#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键帧提取器主类
整合所有功能模块，提供统一的接口
"""

import logging
import time
from typing import List, Dict, Optional
from pathlib import Path

from .config import Config
from ..models.model_manager import ModelManager
from ..processors.shot_detector import ShotDetector
from ..processors.feature_extractor import FeatureExtractor
from ..processors.keyframe_selector import KeyFrameSelector
from ..utils.output_manager import OutputManager

logger = logging.getLogger(__name__)


class KeyFrameExtractor:
    """
    视频关键帧提取器主类

    统一接口，整合所有功能模块：
    1. 配置管理
    2. 模型管理
    3. 镜头检测
    4. 特征提取
    5. 关键帧选择
    6. 结果输出
    """

    def __init__(self, config: Optional[Config] = None):
        """
        初始化关键帧提取器

        Args:
            config: 配置对象，如果为None则使用默认配置
        """
        self.config = config or Config()
        self._setup_logging()

        # 初始化各个组件
        self.model_manager = None
        self.shot_detector = None
        self.feature_extractor = None
        self.keyframe_selector = None
        self.output_manager = None

        self._initialized = False

        logger.info("KeyFrameExtractor initialized")

    def _setup_logging(self):
        """设置日志"""
        if self.config.verbose:
            log_level = getattr(logging, self.config.log_level.upper(), logging.INFO)
            logging.basicConfig(
                level=log_level,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )

    def initialize(self) -> bool:
        """
        初始化所有组件

        Returns:
            是否初始化成功
        """
        try:
            logger.info("Initializing KeyFrameExtractor components...")

            # 1. 初始化模型管理器
            self.model_manager = ModelManager(self.config)
            if not self.model_manager.initialize_models():
                logger.error("Failed to initialize models")
                return False

            # 2. 初始化镜头检测器
            self.shot_detector = ShotDetector(
                self.config,
                self.model_manager.get_transnet_manager()
            )

            # 3. 初始化特征提取器
            self.feature_extractor = FeatureExtractor(
                self.config,
                self.model_manager.get_clip_manager()
            )

            # 4. 初始化关键帧选择器
            self.keyframe_selector = KeyFrameSelector(self.config)

            # 5. 初始化输出管理器
            self.output_manager = OutputManager(self.config)

            self._initialized = True
            logger.info("All components initialized successfully")

            # 打印模型信息
            model_info = self.model_manager.get_model_info()
            logger.info(f"Model status: {model_info}")

            return True

        except Exception as e:
            logger.error(f"Initialization failed: {e}")
            return False

    def extract_keyframes(self, video_path: str, output_dir: str) -> List[Dict]:
        """
        提取视频关键帧的主要方法

        Args:
            video_path: 视频文件路径
            output_dir: 输出目录

        Returns:
            关键帧信息列表
        """
        if not self._initialized:
            logger.error("Extractor not initialized. Call initialize() first.")
            return []

        start_time = time.time()

        try:
            logger.info(f"Starting keyframe extraction for: {video_path}")

            # 验证输入
            if not self._validate_input(video_path, output_dir):
                return []

            # 步骤1: 镜头边界检测
            logger.info("Step 1: Shot boundary detection")
            shots = self.shot_detector.detect_shots(video_path)
            if not shots:
                logger.error("No shots detected")
                return []

            logger.info(f"Detected {len(shots)} shots")

            # 步骤2: 特征提取
            logger.info("Step 2: Feature extraction")
            feature_data = self.feature_extractor.extract_all_features(video_path)
            if not feature_data["frames"]:
                logger.error("No features extracted")
                return []

            logger.info(f"Extracted features for {len(feature_data['frames'])} frames")

            # 步骤3: 关键帧选择
            logger.info("Step 3: Keyframe selection")
            keyframes = self.keyframe_selector.select_keyframes(
                shots, feature_data["features"], feature_data["frame_info"]
            )
            if not keyframes:
                logger.error("No keyframes selected")
                return []

            logger.info(f"Selected {len(keyframes)} keyframes")

            # 步骤4: 保存结果
            logger.info("Step 4: Saving results")
            processing_info = self._create_processing_info(shots, feature_data, keyframes, start_time)

            results = self.output_manager.save_results(
                keyframes,
                feature_data["frames"],
                feature_data["frame_info"],
                output_dir,
                video_path,
                processing_info
            )

            if not results:
                logger.error("Failed to save results")
                return []

            # 生成最终报告
            end_time = time.time()
            self._log_final_report(results, start_time, end_time)

            # 转换为字典格式返回
            return [result.to_dict() for result in results]

        except Exception as e:
            logger.error(f"Keyframe extraction failed: {e}")
            return []

    def extract_keyframes_batch(self, video_paths: List[str],
                              output_base_dir: str) -> Dict[str, List[Dict]]:
        """
        批量提取多个视频的关键帧

        Args:
            video_paths: 视频文件路径列表
            output_base_dir: 输出基础目录

        Returns:
            {video_path: 关键帧信息列表} 的字典
        """
        if not self._initialized:
            logger.error("Extractor not initialized. Call initialize() first.")
            return {}

        results = {}

        logger.info(f"Starting batch extraction for {len(video_paths)} videos")

        for i, video_path in enumerate(video_paths):
            try:
                logger.info(f"Processing video {i+1}/{len(video_paths)}: {video_path}")

                # 为每个视频创建独立的输出目录
                output_dir = self.output_manager.create_output_directory(
                    output_base_dir, video_path
                )

                # 提取关键帧
                video_results = self.extract_keyframes(video_path, output_dir)
                results[video_path] = video_results

                logger.info(f"Completed video {i+1}: {len(video_results)} keyframes")

            except Exception as e:
                logger.error(f"Failed to process {video_path}: {e}")
                results[video_path] = []

        logger.info(f"Batch extraction completed: {len(results)} videos processed")
        return results

    def _validate_input(self, video_path: str, output_dir: str) -> bool:
        """验证输入参数"""
        # 检查视频文件
        if not Path(video_path).exists():
            logger.error(f"Video file not found: {video_path}")
            return False

        # 检查输出目录
        if not self.output_manager.validate_output_directory(output_dir):
            logger.error(f"Invalid output directory: {output_dir}")
            return False

        return True

    def _create_processing_info(self, shots, feature_data, keyframes, start_time) -> Dict:
        """创建处理信息"""
        processing_time = time.time() - start_time

        return {
            "processing_time_seconds": processing_time,
            "total_shots": len(shots),
            "total_frames_processed": len(feature_data["frames"]),
            "total_keyframes": len(keyframes),
            "avg_keyframes_per_shot": len(keyframes) / len(shots) if shots else 0,
            "shot_detection_methods": list(set(shot.method for shot in shots)),
            "feature_extraction_summary": self.feature_extractor.get_feature_summary(feature_data),
            "keyframe_selection_stats": self.keyframe_selector.get_selection_statistics(keyframes),
            "config_used": self.config.to_dict()
        }

    def _log_final_report(self, results, start_time: float, end_time: float):
        """记录最终报告"""
        processing_time = end_time - start_time

        logger.info("=" * 60)
        logger.info("KEYFRAME EXTRACTION COMPLETED")
        logger.info("=" * 60)
        logger.info(f"Total keyframes extracted: {len(results)}")
        logger.info(f"Processing time: {processing_time:.2f} seconds")

        if results:
            # 输出摘要
            summary = self.output_manager.get_output_summary(results)
            logger.info(f"Total file size: {summary.get('total_size_mb', 0):.2f} MB")
            logger.info(f"Shots covered: {summary.get('shots_covered', 0)}")
            logger.info(f"Average confidence: {summary.get('confidence_stats', {}).get('mean', 0):.3f}")

            output_dir = Path(results[0].filepath).parent
            logger.info(f"Results saved to: {output_dir}")

        logger.info("=" * 60)

    def get_config(self) -> Config:
        """获取当前配置"""
        return self.config

    def update_config(self, **kwargs):
        """更新配置"""
        self.config.update(**kwargs)
        logger.info("Configuration updated")

    def get_model_info(self) -> Dict:
        """获取模型信息"""
        if self.model_manager:
            return self.model_manager.get_model_info()
        return {}

    def cleanup(self):
        """清理资源"""
        if self.model_manager:
            self.model_manager.cleanup()

        self._initialized = False
        logger.info("KeyFrameExtractor cleaned up")

    def __enter__(self):
        """上下文管理器入口"""
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.cleanup()


# 便捷函数
def extract_keyframes_simple(video_path: str, output_dir: str,
                           config: Optional[Dict] = None) -> List[Dict]:
    """
    简化的关键帧提取函数

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        config: 配置字典

    Returns:
        关键帧信息列表
    """
    # 创建配置
    if config:
        cfg = Config.from_dict(config)
    else:
        cfg = Config()

    # 提取关键帧
    with KeyFrameExtractor(cfg) as extractor:
        return extractor.extract_keyframes(video_path, output_dir)


def extract_keyframes_with_preset(video_path: str, output_dir: str,
                                preset: str = "balanced") -> List[Dict]:
    """
    使用预设配置提取关键帧

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        preset: 预设名称 ("fast", "balanced", "quality", "detailed")

    Returns:
        关键帧信息列表
    """
    from .config import ConfigManager

    config_manager = ConfigManager()
    config = config_manager.create_preset_config(preset)

    with KeyFrameExtractor(config) as extractor:
        return extractor.extract_keyframes(video_path, output_dir)