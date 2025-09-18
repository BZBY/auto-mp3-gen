#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
镜头边界检测器
负责检测视频中的镜头边界，支持TransNetV2和备用方法
"""

import logging
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

from ..core.config import Config
from ..models.model_manager import TransNetV2ModelManager

logger = logging.getLogger(__name__)


class ShotBoundary:
    """镜头边界数据类"""

    def __init__(self, start_frame: int, end_frame: int,
                 confidence: float = 1.0, method: str = "unknown"):
        """
        初始化镜头边界

        Args:
            start_frame: 开始帧索引
            end_frame: 结束帧索引
            confidence: 置信度 (0-1)
            method: 检测方法
        """
        self.start_frame = start_frame
        self.end_frame = end_frame
        self.confidence = confidence
        self.method = method
        self.duration_frames = end_frame - start_frame

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "start_frame": self.start_frame,
            "end_frame": self.end_frame,
            "duration_frames": self.duration_frames,
            "confidence": self.confidence,
            "method": self.method
        }

    def __repr__(self):
        return f"ShotBoundary({self.start_frame}-{self.end_frame}, conf={self.confidence:.3f})"


class TransNetV2ShotDetector:
    """基于TransNetV2的镜头检测器"""

    def __init__(self, config: Config, transnet_manager: TransNetV2ModelManager):
        """
        初始化TransNetV2检测器

        Args:
            config: 配置对象
            transnet_manager: TransNetV2模型管理器
        """
        self.config = config
        self.transnet_manager = transnet_manager

    def detect_shots(self, video_path: str) -> List[ShotBoundary]:
        """
        使用TransNetV2检测镜头边界

        Args:
            video_path: 视频文件路径

        Returns:
            镜头边界列表
        """
        if not self.transnet_manager.is_loaded():
            logger.error("TransNetV2 model not loaded")
            return []

        try:
            # 使用TransNetV2 PyTorch版本预测
            result = self.transnet_manager.predict_video(video_path)
            if result is None:
                logger.error("TransNetV2 prediction failed")
                return []

            # 从结果中提取场景边界
            scenes = self.transnet_manager.get_scenes_from_result(result)
            if scenes is None:
                logger.error("Failed to extract scenes from TransNetV2 result")
                return []

            # 转换为ShotBoundary对象
            shot_boundaries = []
            for i, (start, end) in enumerate(scenes):
                # PyTorch版本使用固定置信度，因为没有单帧预测
                confidence = 0.9  # 默认高置信度，因为TransNetV2是预训练模型

                shot_boundary = ShotBoundary(
                    start_frame=int(start),
                    end_frame=int(end),
                    confidence=confidence,
                    method="TransNetV2-PyTorch"
                )
                shot_boundaries.append(shot_boundary)

            logger.info(f"TransNetV2 PyTorch detected {len(shot_boundaries)} shots")
            return shot_boundaries

        except Exception as e:
            logger.error(f"TransNetV2 shot detection failed: {e}")
            return []


class HistogramShotDetector:
    """基于直方图差异的镜头检测器（备用方法）"""

    def __init__(self, config: Config):
        """
        初始化直方图检测器

        Args:
            config: 配置对象
        """
        self.config = config

    def detect_shots(self, video_path: str) -> List[ShotBoundary]:
        """
        使用直方图差异检测镜头边界

        Args:
            video_path: 视频文件路径

        Returns:
            镜头边界列表
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            logger.info(f"Analyzing {total_frames} frames for shot boundaries")

            # 计算直方图差异
            hist_diffs = self._calculate_histogram_differences(cap, total_frames)
            cap.release()

            if not hist_diffs:
                logger.error("Failed to calculate histogram differences")
                return []

            # 检测镜头边界
            shot_boundaries = self._detect_boundaries_from_diffs(hist_diffs, total_frames)

            logger.info(f"Histogram method detected {len(shot_boundaries)} shots")
            return shot_boundaries

        except Exception as e:
            logger.error(f"Histogram shot detection failed: {e}")
            return []

    def _calculate_histogram_differences(self, cap: cv2.VideoCapture,
                                       total_frames: int) -> List[float]:
        """计算帧间直方图差异"""
        hist_diffs = []
        prev_hist = None

        # 采样参数
        sample_rate = max(1, total_frames // 5000)  # 最多分析5000帧

        pbar = tqdm(range(0, total_frames, sample_rate), desc="计算直方图差异")

        for frame_idx in pbar:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            # 调整图像大小以加速计算
            frame = cv2.resize(frame, (320, 240))

            # 计算HSV直方图
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            hist = cv2.calcHist([hsv], [0, 1, 2], None,
                              [50, 60, 60], [0, 180, 0, 256, 0, 256])

            # 归一化直方图
            hist = cv2.normalize(hist, hist).flatten()

            if prev_hist is not None:
                # 计算相关性（越小差异越大）
                correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)
                # 转换为差异度（越大差异越大）
                diff = 1 - correlation
                hist_diffs.append((frame_idx, diff))

            prev_hist = hist

        return hist_diffs

    def _detect_boundaries_from_diffs(self, hist_diffs: List[Tuple[int, float]],
                                    total_frames: int) -> List[ShotBoundary]:
        """从直方图差异中检测边界"""
        if not hist_diffs:
            return []

        # 提取差异值
        frame_indices, diff_values = zip(*hist_diffs)
        diff_values = np.array(diff_values)

        # 动态阈值计算
        mean_diff = np.mean(diff_values)
        std_diff = np.std(diff_values)

        # 使用均值加上若干标准差作为阈值
        threshold = mean_diff + 2.5 * std_diff

        # 如果阈值太高，使用百分位数
        if threshold > np.percentile(diff_values, 98):
            threshold = np.percentile(diff_values, 95)

        # 检测峰值
        shot_boundaries_frames = [0]  # 视频开始

        for i, (frame_idx, diff) in enumerate(hist_diffs):
            if diff > threshold:
                # 避免太接近的边界
                if not shot_boundaries_frames or frame_idx - shot_boundaries_frames[-1] > 30:
                    shot_boundaries_frames.append(frame_idx)

        shot_boundaries_frames.append(total_frames)  # 视频结束

        # 创建ShotBoundary对象
        shot_boundaries = []
        for i in range(len(shot_boundaries_frames) - 1):
            start = shot_boundaries_frames[i]
            end = shot_boundaries_frames[i + 1]

            # 计算该段的平均差异作为置信度参考
            segment_diffs = [diff for frame_idx, diff in hist_diffs
                           if start <= frame_idx < end]
            confidence = 1.0 - (np.mean(segment_diffs) if segment_diffs else 0.5)
            confidence = max(0.1, min(1.0, confidence))  # 限制在合理范围

            shot_boundary = ShotBoundary(
                start_frame=start,
                end_frame=end,
                confidence=confidence,
                method="Histogram"
            )
            shot_boundaries.append(shot_boundary)

        return shot_boundaries


class OpticalFlowShotDetector:
    """基于光流的镜头检测器（补充方法）"""

    def __init__(self, config: Config):
        """
        初始化光流检测器

        Args:
            config: 配置对象
        """
        self.config = config

    def detect_shots(self, video_path: str) -> List[ShotBoundary]:
        """
        使用光流检测镜头边界

        Args:
            video_path: 视频文件路径

        Returns:
            镜头边界列表
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return []

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # 计算光流特征
            motion_scores = self._calculate_motion_scores(cap, total_frames)
            cap.release()

            if not motion_scores:
                logger.error("Failed to calculate motion scores")
                return []

            # 检测镜头边界
            shot_boundaries = self._detect_boundaries_from_motion(motion_scores, total_frames)

            logger.info(f"Optical flow method detected {len(shot_boundaries)} shots")
            return shot_boundaries

        except Exception as e:
            logger.error(f"Optical flow shot detection failed: {e}")
            return []

    def _calculate_motion_scores(self, cap: cv2.VideoCapture,
                               total_frames: int) -> List[Tuple[int, float]]:
        """计算光流运动分数"""
        motion_scores = []
        prev_gray = None

        # 采样参数
        sample_rate = max(1, total_frames // 3000)  # 最多分析3000帧

        pbar = tqdm(range(0, total_frames, sample_rate), desc="计算光流")

        for frame_idx in pbar:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()

            if not ret:
                break

            # 调整大小并转换为灰度
            frame = cv2.resize(frame, (320, 240))
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            if prev_gray is not None:
                # 计算光流
                flow = cv2.calcOpticalFlowFarneback(
                    prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
                )

                # 计算运动强度
                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                motion_score = np.mean(magnitude)
                motion_scores.append((frame_idx, motion_score))

            prev_gray = gray

        return motion_scores

    def _detect_boundaries_from_motion(self, motion_scores: List[Tuple[int, float]],
                                     total_frames: int) -> List[ShotBoundary]:
        """从运动分数检测边界"""
        if not motion_scores:
            return []

        frame_indices, scores = zip(*motion_scores)
        scores = np.array(scores)

        # 寻找运动的急剧变化
        # 计算运动分数的一阶差分
        motion_changes = np.abs(np.diff(scores))

        # 阈值为变化的95%分位数
        threshold = np.percentile(motion_changes, 95)

        # 检测边界
        shot_boundaries_frames = [0]

        for i, change in enumerate(motion_changes):
            if change > threshold:
                frame_idx = frame_indices[i + 1]
                # 避免太接近的边界
                if frame_idx - shot_boundaries_frames[-1] > 30:
                    shot_boundaries_frames.append(frame_idx)

        shot_boundaries_frames.append(total_frames)

        # 创建ShotBoundary对象
        shot_boundaries = []
        for i in range(len(shot_boundaries_frames) - 1):
            start = shot_boundaries_frames[i]
            end = shot_boundaries_frames[i + 1]

            shot_boundary = ShotBoundary(
                start_frame=start,
                end_frame=end,
                confidence=0.7,  # 光流方法的固定置信度
                method="OpticalFlow"
            )
            shot_boundaries.append(shot_boundary)

        return shot_boundaries


class ShotDetector:
    """
    统一镜头检测器
    根据配置和模型可用性选择最佳检测方法
    """

    def __init__(self, config: Config, transnet_manager: Optional[TransNetV2ModelManager] = None):
        """
        初始化镜头检测器

        Args:
            config: 配置对象
            transnet_manager: TransNetV2模型管理器（可选）
        """
        self.config = config
        self.transnet_detector = None
        self.histogram_detector = HistogramShotDetector(config)
        self.optical_flow_detector = OpticalFlowShotDetector(config)

        # 初始化TransNetV2检测器（如果可用）
        if (transnet_manager and
            transnet_manager.is_available() and
            config.use_transnet):
            self.transnet_detector = TransNetV2ShotDetector(config, transnet_manager)

    def detect_shots(self, video_path: str) -> List[ShotBoundary]:
        """
        检测视频中的镜头边界

        Args:
            video_path: 视频文件路径

        Returns:
            镜头边界列表
        """
        logger.info(f"Starting shot detection for: {video_path}")

        # 方法优先级：TransNetV2 > Histogram > OpticalFlow
        methods = []

        if self.transnet_detector:
            methods.append(("TransNetV2", self.transnet_detector))

        methods.extend([
            ("Histogram", self.histogram_detector),
            ("OpticalFlow", self.optical_flow_detector)
        ])

        # 尝试各种方法
        for method_name, detector in methods:
            try:
                logger.info(f"Trying shot detection with {method_name}")
                shot_boundaries = detector.detect_shots(video_path)

                if shot_boundaries:
                    logger.info(f"Shot detection successful with {method_name}: "
                              f"{len(shot_boundaries)} shots found")
                    return self._post_process_shots(shot_boundaries, video_path)
                else:
                    logger.warning(f"{method_name} returned no shots")

            except Exception as e:
                logger.error(f"Shot detection with {method_name} failed: {e}")
                continue

        # 如果所有方法都失败，创建单个镜头
        logger.warning("All shot detection methods failed, creating single shot")
        return self._create_fallback_shot(video_path)

    def _post_process_shots(self, shot_boundaries: List[ShotBoundary],
                          video_path: str) -> List[ShotBoundary]:
        """
        后处理镜头边界：过滤、合并、优化
        
        Args:
            shot_boundaries: 原始镜头边界列表
            video_path: 视频路径
            
        Returns:
            处理后的镜头边界列表
        """
        """后处理镜头边界"""
        if not shot_boundaries:
            return shot_boundaries

        # 获取视频总帧数
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # 使用配置参数过滤镜头
        filtered_shots = []
        
        if self.config.enable_shot_filtering:
            for shot in shot_boundaries:
                # 检查镜头长度
                if (shot.duration_frames >= self.config.min_shot_duration and 
                    shot.duration_frames <= self.config.max_shot_duration):
                    filtered_shots.append(shot)
                else:
                    logger.debug(f"Filtered shot (duration={shot.duration_frames}): {shot}")
        else:
            filtered_shots = shot_boundaries.copy()

        # 如果过滤后没有镜头，保留最长的几个
        if not filtered_shots:
            shot_boundaries.sort(key=lambda x: x.duration_frames, reverse=True)
            filtered_shots = shot_boundaries[:max(1, len(shot_boundaries) // 2)]

        # 确保覆盖整个视频
        if filtered_shots:
            # 调整第一个镜头的开始帧
            filtered_shots[0].start_frame = 0
            # 调整最后一个镜头的结束帧
            filtered_shots[-1].end_frame = total_frames

        logger.info(f"Post-processing: {len(shot_boundaries)} -> {len(filtered_shots)} shots")
        return filtered_shots

    def _create_fallback_shot(self, video_path: str) -> List[ShotBoundary]:
        """创建备用的单个镜头"""
        try:
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()

            return [ShotBoundary(
                start_frame=0,
                end_frame=total_frames,
                confidence=0.5,
                method="Fallback"
            )]

        except Exception as e:
            logger.error(f"Failed to create fallback shot: {e}")
            return []

    def get_available_methods(self) -> List[str]:
        """获取可用的检测方法"""
        methods = ["Histogram", "OpticalFlow"]
        if self.transnet_detector:
            methods.insert(0, "TransNetV2")
        return methods