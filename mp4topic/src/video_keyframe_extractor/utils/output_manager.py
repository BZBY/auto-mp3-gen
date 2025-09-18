#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
输出管理器
负责保存关键帧图像和元数据
"""

import logging
import cv2
import numpy as np
import pandas as pd
import json
from typing import List, Dict, Optional
from pathlib import Path
from datetime import datetime

from ..core.config import Config
from ..processors.keyframe_selector import KeyFrame
from ..processors.feature_extractor import FrameInfo

logger = logging.getLogger(__name__)


class OutputResult:
    """输出结果数据类"""

    def __init__(self, filename: str, filepath: str, keyframe: KeyFrame,
                 frame_info: FrameInfo, metadata: Optional[Dict] = None):
        """
        初始化输出结果

        Args:
            filename: 文件名
            filepath: 完整文件路径
            keyframe: 关键帧信息
            frame_info: 帧信息
            metadata: 额外元数据
        """
        self.filename = filename
        self.filepath = filepath
        self.keyframe = keyframe
        self.frame_info = frame_info
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        """转换为字典"""
        result = {
            "filename": self.filename,
            "filepath": self.filepath,
            "frame_idx": self.keyframe.frame_idx,
            "original_frame_idx": self.frame_info.original_idx,
            "timestamp": self.frame_info.timestamp,
            "shot_id": self.keyframe.shot_id,
            "cluster_id": self.keyframe.cluster_id,
            "confidence": self.keyframe.confidence,
            "selection_method": self.keyframe.selection_method,
            "fps": self.frame_info.fps
        }
        result.update(self.metadata)
        return result


class ImageSaver:
    """图像保存器"""

    def __init__(self, config: Config):
        """
        初始化图像保存器

        Args:
            config: 配置对象
        """
        self.config = config

    def save_keyframes(self, keyframes: List[KeyFrame], frames: List[np.ndarray],
                      frame_info: List[FrameInfo], output_dir: str,
                      video_path: str) -> List[OutputResult]:
        """
        保存关键帧图像

        Args:
            keyframes: 关键帧列表
            frames: 帧图像列表
            frame_info: 帧信息列表
            output_dir: 输出目录
            video_path: 视频路径

        Returns:
            输出结果列表
        """
        if not keyframes:
            logger.warning("No keyframes to save")
            return []

        try:
            # 创建输出目录
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)

            video_name = Path(video_path).stem
            results = []

            logger.info(f"Saving {len(keyframes)} keyframes to {output_dir}")

            for i, keyframe in enumerate(keyframes):
                try:
                    result = self._save_single_keyframe(
                        keyframe, frames, frame_info, output_path, video_name, i
                    )
                    if result:
                        results.append(result)

                except Exception as e:
                    logger.error(f"Failed to save keyframe {i}: {e}")
                    continue

            logger.info(f"Successfully saved {len(results)} keyframes")
            return results

        except Exception as e:
            logger.error(f"Keyframe saving failed: {e}")
            return []

    def _save_single_keyframe(self, keyframe: KeyFrame, frames: List[np.ndarray],
                            frame_info: List[FrameInfo], output_path: Path,
                            video_name: str, keyframe_idx: int) -> Optional[OutputResult]:
        """保存单个关键帧"""
        try:
            # 获取帧索引和信息
            frame_idx = keyframe.frame_idx
            if frame_idx >= len(frames) or frame_idx >= len(frame_info):
                logger.error(f"Frame index {frame_idx} out of range")
                return None

            frame = frames[frame_idx]
            info = frame_info[frame_idx]

            # 生成文件名
            filename = self._generate_filename(
                video_name, keyframe, info, keyframe_idx
            )

            # 完整路径
            filepath = output_path / filename

            # 设置JPEG质量
            jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, self.config.image_quality]

            # 保存图像
            success = cv2.imwrite(str(filepath), frame, jpeg_params)

            if not success:
                logger.error(f"Failed to write image: {filepath}")
                return None

            # 创建输出结果
            result = OutputResult(
                filename=filename,
                filepath=str(filepath),
                keyframe=keyframe,
                frame_info=info,
                metadata={
                    "keyframe_idx": keyframe_idx,
                    "image_quality": self.config.image_quality,
                    "selection_metadata": keyframe.metadata
                }
            )

            return result

        except Exception as e:
            logger.error(f"Failed to save single keyframe: {e}")
            return None

    def _generate_filename(self, video_name: str, keyframe: KeyFrame,
                         frame_info: FrameInfo, keyframe_idx: int) -> str:
        """生成文件名"""
        timestamp_str = f"{frame_info.timestamp:.2f}s"
        filename = (f"{video_name}_"
                   f"shot{keyframe.shot_id:02d}_"
                   f"keyframe_{keyframe_idx:04d}_"
                   f"frame_{frame_info.original_idx:06d}_"
                   f"{timestamp_str}.jpg")

        # 清理文件名中的非法字符
        filename = "".join(c for c in filename if c.isalnum() or c in "._-")
        return filename


class MetadataSaver:
    """元数据保存器"""

    def __init__(self, config: Config):
        """
        初始化元数据保存器

        Args:
            config: 配置对象
        """
        self.config = config

    def save_metadata(self, results: List[OutputResult], output_dir: str,
                     video_path: str, processing_info: Optional[Dict] = None):
        """
        保存元数据

        Args:
            results: 输出结果列表
            output_dir: 输出目录
            video_path: 视频路径
            processing_info: 处理信息
        """
        if not results:
            logger.warning("No results to save metadata for")
            return

        try:
            output_path = Path(output_dir)
            video_name = Path(video_path).stem

            # 保存JSON元数据
            if self.config.save_metadata:
                self._save_json_metadata(results, output_path, video_name,
                                       video_path, processing_info)

            # 保存CSV元数据
            if self.config.save_csv:
                self._save_csv_metadata(results, output_path, video_name)

            # 保存文本摘要
            self._save_text_summary(results, output_path, video_name,
                                  video_path, processing_info)

        except Exception as e:
            logger.error(f"Metadata saving failed: {e}")

    def _save_json_metadata(self, results: List[OutputResult], output_path: Path,
                          video_name: str, video_path: str,
                          processing_info: Optional[Dict]):
        """保存JSON格式元数据"""
        try:
            metadata = {
                "video_info": {
                    "path": video_path,
                    "name": video_name
                },
                "processing_info": processing_info or {},
                "extraction_time": datetime.now().isoformat(),
                "config": self.config.to_dict(),
                "total_keyframes": len(results),
                "keyframes": [result.to_dict() for result in results]
            }

            json_path = output_path / f"{video_name}_metadata.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)

            logger.info(f"JSON metadata saved to: {json_path}")

        except Exception as e:
            logger.error(f"Failed to save JSON metadata: {e}")

    def _save_csv_metadata(self, results: List[OutputResult], output_path: Path,
                         video_name: str):
        """保存CSV格式元数据"""
        try:
            # 转换为DataFrame
            data = [result.to_dict() for result in results]
            df = pd.DataFrame(data)

            # 保存CSV
            csv_path = output_path / f"{video_name}_keyframes.csv"
            df.to_csv(csv_path, index=False, encoding='utf-8')

            logger.info(f"CSV metadata saved to: {csv_path}")

        except Exception as e:
            logger.error(f"Failed to save CSV metadata: {e}")

    def _save_text_summary(self, results: List[OutputResult], output_path: Path,
                         video_name: str, video_path: str,
                         processing_info: Optional[Dict]):
        """保存文本摘要"""
        try:
            summary_path = output_path / f"{video_name}_summary.txt"

            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write("=" * 60 + "\n")
                f.write("视频关键帧提取摘要\n")
                f.write("=" * 60 + "\n\n")

                f.write(f"视频文件: {video_path}\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"总关键帧数: {len(results)}\n\n")

                if processing_info:
                    f.write("处理信息:\n")
                    for key, value in processing_info.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")

                # 按镜头分组
                shots = {}
                for result in results:
                    shot_id = result.keyframe.shot_id
                    if shot_id not in shots:
                        shots[shot_id] = []
                    shots[shot_id].append(result)

                f.write("按镜头分组:\n")
                f.write("-" * 40 + "\n")

                for shot_id in sorted(shots.keys()):
                    shot_results = shots[shot_id]
                    f.write(f"\n镜头 {shot_id} ({len(shot_results)} 个关键帧):\n")

                    for result in shot_results:
                        f.write(f"  {result.filename}\n")
                        f.write(f"    时间戳: {result.frame_info.timestamp:.2f}s\n")
                        f.write(f"    置信度: {result.keyframe.confidence:.3f}\n")
                        f.write(f"    方法: {result.keyframe.selection_method}\n")
                        f.write(f"    聚类ID: {result.keyframe.cluster_id}\n\n")

            logger.info(f"Text summary saved to: {summary_path}")

        except Exception as e:
            logger.error(f"Failed to save text summary: {e}")


class OutputManager:
    """
    统一输出管理器
    整合图像保存和元数据保存功能
    """

    def __init__(self, config: Config):
        """
        初始化输出管理器

        Args:
            config: 配置对象
        """
        self.config = config
        self.image_saver = ImageSaver(config)
        self.metadata_saver = MetadataSaver(config)

    def save_results(self, keyframes: List[KeyFrame], frames: List[np.ndarray],
                    frame_info: List[FrameInfo], output_dir: str, video_path: str,
                    processing_info: Optional[Dict] = None) -> List[OutputResult]:
        """
        保存所有结果

        Args:
            keyframes: 关键帧列表
            frames: 帧图像列表
            frame_info: 帧信息列表
            output_dir: 输出目录
            video_path: 视频路径
            processing_info: 处理信息

        Returns:
            输出结果列表
        """
        logger.info(f"Starting to save results to: {output_dir}")

        try:
            # 保存图像
            results = self.image_saver.save_keyframes(
                keyframes, frames, frame_info, output_dir, video_path
            )

            if not results:
                logger.error("No images were saved successfully")
                return []

            # 保存元数据
            self.metadata_saver.save_metadata(
                results, output_dir, video_path, processing_info
            )

            logger.info(f"All results saved successfully: {len(results)} files")
            return results

        except Exception as e:
            logger.error(f"Failed to save results: {e}")
            return []

    def create_output_directory(self, base_dir: str, video_path: str) -> str:
        """
        创建输出目录

        Args:
            base_dir: 基础目录
            video_path: 视频路径

        Returns:
            创建的输出目录路径
        """
        try:
            video_name = Path(video_path).stem
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

            output_dir = Path(base_dir) / f"{video_name}_keyframes_{timestamp}"
            output_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"Created output directory: {output_dir}")
            return str(output_dir)

        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            # 备用目录
            fallback_dir = Path(base_dir) / "keyframes_output"
            fallback_dir.mkdir(parents=True, exist_ok=True)
            return str(fallback_dir)

    def validate_output_directory(self, output_dir: str) -> bool:
        """
        验证输出目录是否可用

        Args:
            output_dir: 输出目录路径

        Returns:
            是否可用
        """
        try:
            output_path = Path(output_dir)

            # 创建目录（如果不存在）
            output_path.mkdir(parents=True, exist_ok=True)

            # 测试写权限
            test_file = output_path / "test_write.tmp"
            test_file.write_text("test")
            test_file.unlink()

            return True

        except Exception as e:
            logger.error(f"Output directory validation failed: {e}")
            return False

    def get_output_summary(self, results: List[OutputResult]) -> Dict:
        """
        获取输出摘要统计

        Args:
            results: 输出结果列表

        Returns:
            摘要统计信息
        """
        if not results:
            return {}

        try:
            # 统计信息
            total_files = len(results)
            total_size = sum(Path(result.filepath).stat().st_size
                           for result in results if Path(result.filepath).exists())

            # 按镜头统计
            shot_stats = {}
            for result in results:
                shot_id = result.keyframe.shot_id
                shot_stats[shot_id] = shot_stats.get(shot_id, 0) + 1

            # 按方法统计
            method_stats = {}
            for result in results:
                method = result.keyframe.selection_method
                method_stats[method] = method_stats.get(method, 0) + 1

            # 置信度统计
            confidences = [result.keyframe.confidence for result in results]

            summary = {
                "total_files": total_files,
                "total_size_mb": total_size / (1024 * 1024),
                "shots_covered": len(shot_stats),
                "avg_keyframes_per_shot": total_files / len(shot_stats) if shot_stats else 0,
                "confidence_stats": {
                    "mean": np.mean(confidences),
                    "min": np.min(confidences),
                    "max": np.max(confidences),
                    "std": np.std(confidences)
                },
                "method_distribution": method_stats,
                "shot_distribution": shot_stats
            }

            return summary

        except Exception as e:
            logger.error(f"Failed to create output summary: {e}")
            return {}