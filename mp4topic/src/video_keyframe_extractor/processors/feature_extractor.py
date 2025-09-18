#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
特征提取器
负责提取视频帧的CLIP语义特征和光流运动特征
"""

import logging
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm
import torch

from ..core.config import Config
from ..models.model_manager import CLIPModelManager

logger = logging.getLogger(__name__)


class FrameInfo:
    """帧信息数据类"""

    def __init__(self, sampled_idx: int, original_idx: int,
                 timestamp: float, fps: float):
        """
        初始化帧信息

        Args:
            sampled_idx: 采样后的帧索引
            original_idx: 原始帧索引
            timestamp: 时间戳（秒）
            fps: 视频帧率
        """
        self.sampled_idx = sampled_idx
        self.original_idx = original_idx
        self.timestamp = timestamp
        self.fps = fps

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "sampled_idx": self.sampled_idx,
            "original_idx": self.original_idx,
            "timestamp": self.timestamp,
            "fps": self.fps
        }


class VideoFrameExtractor:
    """视频帧提取器"""

    def __init__(self, config: Config):
        """
        初始化帧提取器

        Args:
            config: 配置对象
        """
        self.config = config
        self.device = config.get_device()
        self.use_gpu_resize = self.device == "cuda" and self._check_gpu_opencv()

    def extract_frames(self, video_path: str) -> Tuple[List[np.ndarray], List[FrameInfo]]:
        """
        从视频中提取帧

        Args:
            video_path: 视频文件路径

        Returns:
            (帧列表, 帧信息列表)
        """
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                logger.error(f"Cannot open video: {video_path}")
                return [], []

            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            logger.info(f"Video info: {total_frames} frames, {fps:.2f} FPS, {duration:.2f}s")

            # 根据采样策略计算间隔
            if self.config.sampling_strategy == "seconds":
                # 按秒采样：每秒取指定帧数
                frame_interval = max(1, int(fps / self.config.frames_per_second))
                estimated_frames = int(duration * self.config.frames_per_second)
                logger.info(f"Using seconds-based sampling: {self.config.frames_per_second} frame(s) per second")
                logger.info(f"Note: Will also ensure each shot has at least one frame")
            elif self.config.sampling_strategy == "fps_ratio":
                # 按帧率比例采样：原帧率/比例
                target_fps = fps / self.config.fps_ratio
                frame_interval = max(1, int(fps / target_fps))
                estimated_frames = total_frames // frame_interval
                logger.info(f"Using FPS-ratio sampling: {fps:.1f}FPS / {self.config.fps_ratio} = {target_fps:.1f}FPS")
            elif self.config.sampling_strategy == "random_seconds":
                # 随机秒采样：每秒随机选择指定数量的帧
                frame_interval = 1  # 我们会在后面处理随机选择
                estimated_frames = int(duration * self.config.random_select_count)
                logger.info(f"Using random seconds sampling: {self.config.random_select_count} random frames from all frames per second")
            else:
                # 按帧率采样（原方法）
                frame_interval = max(1, int(fps / self.config.target_fps))
                estimated_frames = total_frames // frame_interval
                logger.info(f"Using FPS-based sampling: target {self.config.target_fps} FPS")

            logger.info(f"Extracting frames with interval {frame_interval}, "
                       f"estimated {estimated_frames} frames")

            frames = []
            frame_info = []
            frame_idx = 0

            # 随机采样需要特殊处理
            if self.config.sampling_strategy == "random_seconds":
                return self._extract_frames_random_seconds(cap, fps, total_frames, duration)

            pbar = tqdm(total=estimated_frames, desc="提取帧")

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # 按间隔采样
                if frame_idx % frame_interval == 0:
                    # 根据配置决定是否调整分辨率
                    if self.config.preserve_original_resolution:
                        # 保持原始分辨率，但限制最大尺寸
                        processed_frame = self._process_frame_with_max_size(frame)
                    else:
                        # 调整到指定分辨率
                        if self.use_gpu_resize:
                            processed_frame = self._resize_with_gpu(frame, self.config.output_resolution)
                        else:
                            processed_frame = cv2.resize(frame, self.config.output_resolution)
                    
                    frames.append(processed_frame)

                    info = FrameInfo(
                        sampled_idx=len(frames) - 1,
                        original_idx=frame_idx,
                        timestamp=frame_idx / fps,
                        fps=fps
                    )
                    frame_info.append(info)
                    pbar.update(1)

                frame_idx += 1

            cap.release()
            pbar.close()

            logger.info(f"Extracted {len(frames)} frames from video")
            return frames, frame_info

        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            return [], []

    def _extract_frames_random_seconds(self, cap: cv2.VideoCapture, fps: float, 
                                     total_frames: int, duration: float) -> Tuple[List[np.ndarray], List[FrameInfo]]:
        """随机秒采样：每秒从所有帧中直接随机选择指定数量的帧"""
        import random
        
        frames = []
        frame_info = []
        
        # 按秒处理
        total_seconds = int(duration)
        estimated_frames = total_seconds * self.config.random_select_count
        
        pbar = tqdm(total=estimated_frames, desc=f"随机采样(每秒{self.config.random_select_count}帧,从所有帧)")
        
        try:
            for second in range(total_seconds):
                # 当前秒的帧范围
                start_frame = int(second * fps)
                end_frame = min(int((second + 1) * fps), total_frames)
                
                # 获取当前秒的所有帧索引
                all_frames_in_second = list(range(start_frame, end_frame))
                
                # 从所有帧中直接随机选择
                if all_frames_in_second:
                    select_count = min(self.config.random_select_count, len(all_frames_in_second))
                    selected_frames = random.sample(all_frames_in_second, select_count)
                    selected_frames.sort()  # 保持时间顺序
                    
                    # 提取选中的帧
                    for frame_idx in selected_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                        ret, frame = cap.read()
                        
                        if ret:
                            # 处理帧
                            if self.config.preserve_original_resolution:
                                processed_frame = self._process_frame_with_max_size(frame)
                            else:
                                if self.use_gpu_resize:
                                    processed_frame = self._resize_with_gpu(frame, self.config.output_resolution)
                                else:
                                    processed_frame = cv2.resize(frame, self.config.output_resolution)
                            
                            frames.append(processed_frame)
                            
                            # 创建帧信息
                            timestamp = frame_idx / fps if fps > 0 else 0
                            info = FrameInfo(
                                sampled_idx=len(frames) - 1,
                                original_idx=frame_idx,
                                timestamp=timestamp,
                                fps=fps
                            )
                            frame_info.append(info)
                            pbar.update(1)
            
            pbar.close()
            logger.info(f"Random sampling completed: extracted {len(frames)} frames")
            
        except Exception as e:
            logger.error(f"Random frame extraction failed: {e}")
            pbar.close()
        
        return frames, frame_info

    def _process_frame_with_max_size(self, frame: np.ndarray) -> np.ndarray:
        """
        处理帧，保持原始分辨率但限制最大尺寸
        
        Args:
            frame: 原始帧
            
        Returns:
            处理后的帧
        """
        height, width = frame.shape[:2]
        max_width, max_height = self.config.max_resolution
        
        # 如果帧尺寸超过最大限制，按比例缩放
        if width > max_width or height > max_height:
            # 计算缩放比例
            scale_w = max_width / width
            scale_h = max_height / height
            scale = min(scale_w, scale_h)
            
            # 计算新尺寸
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            logger.debug(f"Resizing frame from {width}x{height} to {new_width}x{new_height}")
            if self.use_gpu_resize:
                return self._resize_with_gpu(frame, (new_width, new_height))
            else:
                return cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
        
        # 如果尺寸在限制范围内，直接返回原帧
        return frame.copy()

    def _check_gpu_opencv(self) -> bool:
        """检查OpenCV是否支持GPU加速"""
        try:
            # 检查是否编译了CUDA支持
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            return False

    def _resize_with_gpu(self, frame: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
        """使用GPU加速的图像缩放"""
        try:
            # 上传到GPU
            gpu_frame = cv2.cuda_GpuMat()
            gpu_frame.upload(frame)
            
            # GPU缩放
            gpu_resized = cv2.cuda.resize(gpu_frame, target_size, interpolation=cv2.INTER_LANCZOS4)
            
            # 下载回CPU
            result = gpu_resized.download()
            return result
        except Exception as e:
            logger.warning(f"GPU resize failed, falling back to CPU: {e}")
            return cv2.resize(frame, target_size, interpolation=cv2.INTER_LANCZOS4)


class CLIPFeatureExtractor:
    """CLIP特征提取器"""

    def __init__(self, config: Config, clip_manager: CLIPModelManager):
        """
        初始化CLIP特征提取器

        Args:
            config: 配置对象
            clip_manager: CLIP模型管理器
        """
        self.config = config
        self.clip_manager = clip_manager

    def extract_features(self, frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """
        提取帧的CLIP特征

        Args:
            frames: 帧列表

        Returns:
            特征矩阵 (n_frames, feature_dim)，失败返回None
        """
        if not self.clip_manager.is_loaded():
            logger.error("CLIP model not loaded")
            return None

        if not frames:
            logger.warning("No frames to extract features from")
            return np.zeros((0, 512))  # 返回空的特征矩阵

        try:
            features = []
            # 动态批处理大小：根据配置和硬件调整
            if self.config.gpu_batch_size > 0:
                batch_size = self.config.gpu_batch_size
            elif self.config.get_device() == "cuda":
                # GPU自动调整：根据帧数和内存优化
                batch_size = min(256, max(64, len(frames) // 2))
            else:
                batch_size = 16  # CPU: 较小批次
                
            logger.info(f"Extracting CLIP features for {len(frames)} frames (batch_size={batch_size})")

            for i in tqdm(range(0, len(frames), batch_size), desc="提取CLIP特征"):
                batch_frames = frames[i:i + batch_size]
                batch_features = self._extract_batch_features(batch_frames)

                if batch_features is not None:
                    features.append(batch_features)
                else:
                    logger.warning(f"Failed to extract features for batch {i//batch_size}")
                    # 使用零特征作为填充
                    features.append(np.zeros((len(batch_frames), 512)))

            if features:
                all_features = np.vstack(features)
                logger.info(f"CLIP feature extraction completed: {all_features.shape}")
                return all_features
            else:
                logger.error("No features extracted")
                return None

        except Exception as e:
            logger.error(f"CLIP feature extraction failed: {e}")
            return None

    def _extract_batch_features(self, batch_frames: List[np.ndarray]) -> Optional[np.ndarray]:
        """提取一批帧的特征"""
        try:
            # 优化：批量预处理，减少CPU-GPU传输次数
            processed_images = []
            for frame in batch_frames:
                processed = self.clip_manager.preprocess_image(frame)
                if processed is not None:
                    processed_images.append(processed)

            if not processed_images:
                return None

            # 批量处理：一次性传输到GPU
            batch_tensor = torch.cat(processed_images, dim=0)

            # GPU特征提取：使用混合精度加速
            with torch.no_grad():
                if self.config.get_device() == "cuda":
                    # GPU: 使用混合精度加速（如果启用）
                    if self.config.use_mixed_precision:
                        try:
                            batch_tensor = batch_tensor.half()
                            features = self.clip_manager.model.encode_image(batch_tensor)
                            features = features.float()  # 转回float32
                        except:
                            # 如果不支持半精度，使用float32
                            features = self.clip_manager.model.encode_image(batch_tensor)
                    else:
                        features = self.clip_manager.model.encode_image(batch_tensor)
                else:
                    features = self.clip_manager.model.encode_image(batch_tensor)
                
                # 在GPU上进行归一化，减少数据传输
                features = features / torch.norm(features, dim=1, keepdim=True)
                features = features.cpu().numpy()

                return features

        except Exception as e:
            logger.error(f"Batch feature extraction failed: {e}")
            return None


class MotionFeatureExtractor:
    """运动特征提取器（基于光流）"""

    def __init__(self, config: Config):
        """
        初始化运动特征提取器

        Args:
            config: 配置对象
        """
        self.config = config
        self.device = config.get_device()
        self.use_gpu_resize = self.device == "cuda" and self._check_gpu_opencv()

    def _check_gpu_opencv(self) -> bool:
        """检查OpenCV是否支持GPU加速"""
        try:
            return cv2.cuda.getCudaEnabledDeviceCount() > 0
        except:
            return False

    def extract_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        提取帧间运动特征

        Args:
            frames: 帧列表

        Returns:
            运动特征向量 (n_frames,)
        """
        if len(frames) < 2:
            logger.warning("Need at least 2 frames for motion feature extraction")
            return np.zeros(len(frames))

        try:
            logger.info(f"Extracting motion features for {len(frames)} frames")
            
            # 使用CPU多线程光流计算
            logger.info("Using CPU optical flow with multi-threading")

            # 批量转换为灰度图（提高效率）
            gray_frames = []
            logger.info("Converting frames to grayscale...")
            for frame in tqdm(frames, desc="转换灰度"):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                gray_frames.append(gray)

            # 计算光流 - 使用CPU并行处理
            motion_features = [0.0]  # 第一帧运动为0
            motion_features.extend(self._calculate_optical_flow_batch_cpu(gray_frames))

            motion_array = np.array(motion_features)
            logger.info(f"Motion feature extraction completed: {motion_array.shape}")

            return motion_array

        except Exception as e:
            logger.error(f"Motion feature extraction failed: {e}")
            return np.zeros(len(frames))

    def _calculate_optical_flow(self, prev_gray: np.ndarray,
                              curr_gray: np.ndarray) -> float:
        """计算两帧之间的光流运动强度"""
        try:
            # 计算密集光流
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, curr_gray, None,
                pyr_scale=0.5,
                levels=3,
                winsize=15,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0
            )

            # 计算运动幅度
            magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

            # 返回平均运动强度
            return float(np.mean(magnitude))

        except Exception as e:
            logger.error(f"Optical flow calculation failed: {e}")
            return 0.0

    def _calculate_optical_flow_batch_cpu(self, gray_frames: List[np.ndarray]) -> List[float]:
        """CPU批量光流计算 - 多线程并行"""
        import multiprocessing as mp
        from concurrent.futures import ThreadPoolExecutor, as_completed
        
        def calculate_flow_pair(args):
            """计算单对帧的光流"""
            i, prev_gray, curr_gray = args
            return i, self._calculate_optical_flow(prev_gray, curr_gray)
        
        motion_scores = [0.0] * (len(gray_frames) - 1)
        
        # 准备任务数据
        tasks = []
        for i in range(1, len(gray_frames)):
            tasks.append((i-1, gray_frames[i-1], gray_frames[i]))
        
        # 使用80%的CPU核心数，避免系统卡顿
        cpu_count = mp.cpu_count()
        max_workers = max(1, int(cpu_count * 0.8))  # 使用80%的核心
        max_workers = min(max_workers, len(tasks))  # 不超过任务数
        logger.info(f"Using {max_workers}/{cpu_count} threads (80%) for optical flow calculation")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_task = {executor.submit(calculate_flow_pair, task): task for task in tasks}
            
            # 创建进度条
            pbar = tqdm(total=len(tasks), desc=f"计算光流(CPU-{max_workers}/{cpu_count}线程)")
            
            # 收集结果
            for future in as_completed(future_to_task):
                try:
                    idx, score = future.result()
                    motion_scores[idx] = score
                    pbar.update(1)
                except Exception as e:
                    logger.warning(f"Optical flow calculation failed for frame pair: {e}")
                    pbar.update(1)
            
            pbar.close()
        
        return motion_scores


class ColorFeatureExtractor:
    """颜色特征提取器"""

    def __init__(self, config: Config):
        """
        初始化颜色特征提取器

        Args:
            config: 配置对象
        """
        self.config = config

    def extract_features(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        提取颜色直方图特征

        Args:
            frames: 帧列表

        Returns:
            颜色特征矩阵 (n_frames, feature_dim)
        """
        if not frames:
            return np.zeros((0, 64))  # 64维颜色特征

        try:
            color_features = []
            logger.info(f"Extracting color features for {len(frames)} frames")

            for frame in tqdm(frames, desc="提取颜色特征"):
                # 转换为HSV颜色空间
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

                # 计算HSV直方图
                hist_h = cv2.calcHist([hsv], [0], None, [16], [0, 180])
                hist_s = cv2.calcHist([hsv], [1], None, [16], [0, 256])
                hist_v = cv2.calcHist([hsv], [2], None, [16], [0, 256])

                # 归一化并合并
                hist_h = cv2.normalize(hist_h, hist_h).flatten()
                hist_s = cv2.normalize(hist_s, hist_s).flatten()
                hist_v = cv2.normalize(hist_v, hist_v).flatten()

                # 连接所有直方图
                color_feature = np.concatenate([hist_h, hist_s, hist_v])
                color_features.append(color_feature)

            color_array = np.array(color_features)
            logger.info(f"Color feature extraction completed: {color_array.shape}")

            return color_array

        except Exception as e:
            logger.error(f"Color feature extraction failed: {e}")
            return np.zeros((len(frames), 64))


class FeatureExtractor:
    """
    统一特征提取器
    整合所有特征提取功能
    """

    def __init__(self, config: Config, clip_manager: Optional[CLIPModelManager] = None):
        """
        初始化特征提取器

        Args:
            config: 配置对象
            clip_manager: CLIP模型管理器（可选）
        """
        self.config = config
        self.frame_extractor = VideoFrameExtractor(config)
        self.clip_extractor = CLIPFeatureExtractor(config, clip_manager) if clip_manager else None
        self.motion_extractor = MotionFeatureExtractor(config)
        self.color_extractor = ColorFeatureExtractor(config)

    def extract_all_features(self, video_path: str) -> Dict:
        """
        提取视频的所有特征

        Args:
            video_path: 视频文件路径

        Returns:
            包含所有特征的字典
        """
        logger.info(f"Starting feature extraction for: {video_path}")

        # 步骤1: 提取帧
        frames, frame_info = self.frame_extractor.extract_frames(video_path)

        if not frames:
            logger.error("No frames extracted")
            return self._create_empty_features()

        # 步骤2: 提取CLIP特征
        clip_features = None
        if self.clip_extractor:
            clip_features = self.clip_extractor.extract_features(frames)

        # 如果CLIP特征提取失败，使用零特征
        if clip_features is None:
            logger.warning("Using zero CLIP features")
            clip_features = np.zeros((len(frames), 512))

        # 步骤3: 提取运动特征（可选）
        if self.config.skip_motion_features:
            logger.info("Skipping motion feature extraction (disabled in config)")
            motion_features = np.zeros(len(frames))
        else:
            motion_features = self.motion_extractor.extract_features(frames)

        # 步骤4: 提取颜色特征（可选）
        if self.config.skip_color_features:
            logger.info("Skipping color feature extraction (disabled in config)")
            color_features = np.zeros((len(frames), 64))
        else:
            color_features = self.color_extractor.extract_features(frames)

        # 构建结果
        result = {
            "frames": frames,
            "frame_info": frame_info,
            "features": {
                "clip": clip_features,
                "motion": motion_features,
                "color": color_features
            },
            "metadata": {
                "num_frames": len(frames),
                "video_path": video_path,
                "feature_dims": {
                    "clip": clip_features.shape[1] if clip_features.ndim > 1 else 0,
                    "motion": 1,
                    "color": color_features.shape[1] if color_features.ndim > 1 else 0
                }
            }
        }

        logger.info("Feature extraction completed successfully")
        return result

    def _create_empty_features(self) -> Dict:
        """创建空的特征字典"""
        return {
            "frames": [],
            "frame_info": [],
            "features": {
                "clip": np.zeros((0, 512)),
                "motion": np.zeros(0),
                "color": np.zeros((0, 64))
            },
            "metadata": {
                "num_frames": 0,
                "video_path": "",
                "feature_dims": {
                    "clip": 512,
                    "motion": 1,
                    "color": 64
                }
            }
        }

    def get_feature_summary(self, features: Dict) -> Dict:
        """
        获取特征摘要统计

        Args:
            features: 特征字典

        Returns:
            特征摘要
        """
        if not features or "features" not in features:
            return {}

        try:
            clip_features = features["features"]["clip"]
            motion_features = features["features"]["motion"]
            color_features = features["features"]["color"]

            summary = {
                "num_frames": features["metadata"]["num_frames"],
                "clip_features": {
                    "shape": clip_features.shape,
                    "mean_norm": float(np.mean(np.linalg.norm(clip_features, axis=1)))
                        if clip_features.size > 0 else 0.0
                },
                "motion_features": {
                    "shape": motion_features.shape,
                    "mean": float(np.mean(motion_features)) if motion_features.size > 0 else 0.0,
                    "std": float(np.std(motion_features)) if motion_features.size > 0 else 0.0,
                    "max": float(np.max(motion_features)) if motion_features.size > 0 else 0.0
                },
                "color_features": {
                    "shape": color_features.shape,
                    "mean": float(np.mean(color_features)) if color_features.size > 0 else 0.0
                }
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to create feature summary: {e}")
            return {}