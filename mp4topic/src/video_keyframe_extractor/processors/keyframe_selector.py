#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
关键帧选择器
负责从特征中选择最具代表性的关键帧
"""

import logging
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
from scipy import signal

from ..core.config import Config
from .shot_detector import ShotBoundary
from .feature_extractor import FrameInfo

logger = logging.getLogger(__name__)


class KeyFrame:
    """关键帧数据类"""

    def __init__(self, frame_idx: int, shot_id: int, confidence: float,
                 cluster_id: int = -1, selection_method: str = "unknown",
                 metadata: Optional[Dict] = None):
        """
        初始化关键帧

        Args:
            frame_idx: 帧索引（在采样后的帧列表中）
            shot_id: 所属镜头ID
            confidence: 置信度 (0-1)
            cluster_id: 聚类ID
            selection_method: 选择方法
            metadata: 额外元数据
        """
        self.frame_idx = frame_idx
        self.shot_id = shot_id
        self.confidence = confidence
        self.cluster_id = cluster_id
        self.selection_method = selection_method
        self.metadata = metadata or {}

    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "frame_idx": self.frame_idx,
            "shot_id": self.shot_id,
            "confidence": self.confidence,
            "cluster_id": self.cluster_id,
            "selection_method": self.selection_method,
            "metadata": self.metadata
        }

    def __repr__(self):
        return f"KeyFrame(idx={self.frame_idx}, shot={self.shot_id}, conf={self.confidence:.3f})"


class MotionAnalyzer:
    """运动分析器"""

    def __init__(self, config: Config):
        """
        初始化运动分析器

        Args:
            config: 配置对象
        """
        self.config = config

    def detect_motion_peaks(self, motion_features: np.ndarray,
                          frame_indices: Optional[List[int]] = None) -> List[int]:
        """
        检测运动峰值

        Args:
            motion_features: 运动特征向量
            frame_indices: 对应的帧索引列表

        Returns:
            峰值位置的索引列表
        """
        if len(motion_features) < 3:
            return list(range(len(motion_features)))

        try:
            # 平滑运动特征 - 修复窗口大小计算
            data_length = len(motion_features)
            # 确保窗口大小为奇数且小于数据长度
            max_window = data_length if data_length % 2 == 1 else data_length - 1
            window_length = min(5, max_window)
            # 窗口大小至少为3（savgol_filter最小要求）
            window_length = max(3, window_length)
            
            # 如果数据太少，跳过滤波
            if data_length < 3:
                smoothed = motion_features.copy()
            else:
                smoothed = signal.savgol_filter(motion_features,
                                              window_length=window_length,
                                              polyorder=min(2, window_length-1))

            # 寻找峰值
            peaks, properties = signal.find_peaks(
                smoothed,
                height=np.percentile(smoothed, 70),  # 高度阈值
                distance=max(3, len(motion_features) // 20)  # 最小间距
            )

            # 如果没有找到峰值，使用较低的阈值重试
            if len(peaks) == 0:
                peaks, _ = signal.find_peaks(
                    smoothed,
                    height=np.percentile(smoothed, 50),
                    distance=max(2, len(motion_features) // 30)
                )

            return peaks.tolist()

        except Exception as e:
            logger.error(f"Motion peak detection failed: {e}")
            # 备用方法：简单阈值检测
            threshold = np.percentile(motion_features, 75)
            peaks = []
            for i in range(1, len(motion_features) - 1):
                if (motion_features[i] > motion_features[i-1] and
                    motion_features[i] > motion_features[i+1] and
                    motion_features[i] > threshold):
                    peaks.append(i)
            return peaks

    def calculate_motion_score(self, motion_features: np.ndarray,
                             window_size: int = 5) -> np.ndarray:
        """
        计算运动分数（考虑时间窗口内的运动变化）

        Args:
            motion_features: 运动特征向量
            window_size: 时间窗口大小

        Returns:
            运动分数向量
        """
        if len(motion_features) == 0:
            return np.array([])

        try:
            scores = np.zeros_like(motion_features)

            for i in range(len(motion_features)):
                start_idx = max(0, i - window_size // 2)
                end_idx = min(len(motion_features), i + window_size // 2 + 1)

                window_motion = motion_features[start_idx:end_idx]

                # 计算窗口内的运动统计
                mean_motion = np.mean(window_motion)
                std_motion = np.std(window_motion)
                current_motion = motion_features[i]

                # 综合分数：当前运动强度 + 运动变化程度
                score = current_motion + std_motion
                scores[i] = score

            return scores

        except Exception as e:
            logger.error(f"Motion score calculation failed: {e}")
            return motion_features.copy()


class ClusteringSelector:
    """基于聚类的关键帧选择器"""

    def __init__(self, config: Config):
        """
        初始化聚类选择器

        Args:
            config: 配置对象
        """
        self.config = config

    def select_keyframes_by_clustering(self, frame_indices: List[int],
                                     clip_features: np.ndarray,
                                     shot_id: int) -> List[KeyFrame]:
        """
        使用聚类方法选择关键帧

        Args:
            frame_indices: 候选帧索引列表
            clip_features: 对应的CLIP特征
            shot_id: 镜头ID

        Returns:
            选中的关键帧列表
        """
        if len(frame_indices) < 2:
            if frame_indices:
                return [KeyFrame(
                    frame_idx=frame_indices[0],
                    shot_id=shot_id,
                    confidence=0.8,
                    cluster_id=0,
                    selection_method="single"
                )]
            return []

        try:
            # 获取对应的特征
            features = clip_features[frame_indices]
            
            # 确保特征数值稳定性
            features = np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=-1.0)
            
            # 归一化特征以避免数值问题
            norms = np.linalg.norm(features, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)  # 避免除零
            features = features / norms

            # 计算相似度矩阵
            similarity_matrix = cosine_similarity(features)
            # 确保相似度矩阵数值稳定
            similarity_matrix = np.clip(similarity_matrix, -1.0, 1.0)
            distance_matrix = 1 - similarity_matrix

            # 动态调整聚类参数
            eps = self._calculate_adaptive_eps(distance_matrix)
            min_samples = max(1, min(2, len(frame_indices) // 3))

            # 确保距离矩阵的最终稳定性
            distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=0.0)
            distance_matrix = np.clip(distance_matrix, 0.0, 2.0)

            # DBSCAN聚类，添加额外的错误检查
            try:
                clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
                cluster_labels = clustering.fit_predict(distance_matrix)
                
                # 检查聚类结果是否有效
                if np.any(np.isnan(cluster_labels)) or np.any(np.isinf(cluster_labels)):
                    raise ValueError("Clustering produced invalid labels")
                    
            except Exception as cluster_error:
                logger.warning(f"DBSCAN clustering failed: {cluster_error}, using simple clustering")
                # 备用聚类：基于相似度阈值的简单聚类
                cluster_labels = self._simple_threshold_clustering(similarity_matrix)

            # 从每个聚类中选择代表帧
            keyframes = self._select_cluster_representatives(
                frame_indices, features, cluster_labels, shot_id
            )

            return keyframes

        except Exception as e:
            logger.error(f"Clustering-based selection failed: {e}")
            # 备用策略：均匀选择，但确保至少选择一些帧
            uniform_frames = self._uniform_selection(frame_indices, shot_id)
            if not uniform_frames and frame_indices:
                # 如果均匀选择也失败，至少返回第一帧
                return [KeyFrame(
                    frame_idx=frame_indices[0],
                    shot_id=shot_id,
                    confidence=0.5,
                    cluster_id=0,
                    selection_method="fallback"
                )]
            return uniform_frames

    def _calculate_adaptive_eps(self, distance_matrix: np.ndarray) -> float:
        """计算自适应的eps参数"""
        try:
            # 确保距离矩阵数值稳定
            distance_matrix = np.nan_to_num(distance_matrix, nan=1.0, posinf=1.0, neginf=0.0)
            distance_matrix = np.clip(distance_matrix, 0.0, 2.0)
            
            # 计算每个点到其最近邻的距离（排除自己）
            distance_copy = distance_matrix.copy()
            np.fill_diagonal(distance_copy, 2.0)  # 用最大值替代无穷大
            min_distances = np.min(distance_copy, axis=1)

            # 使用距离的中位数作为基础eps，但要确保数值稳定
            min_distances = min_distances[min_distances < 2.0]  # 排除对角线值
            if len(min_distances) > 0:
                base_eps = np.median(min_distances)
            else:
                base_eps = self.config.cluster_eps

            # 根据配置调整，确保在合理范围内
            eps = max(0.05, min(0.3, float(base_eps) * 1.5))

            return eps

        except Exception as e:
            logger.error(f"Adaptive eps calculation failed: {e}")
            return self.config.cluster_eps

    def _simple_threshold_clustering(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """简单的基于阈值的聚类备用方案"""
        n_samples = similarity_matrix.shape[0]
        cluster_labels = np.full(n_samples, -1, dtype=int)  # -1 表示噪声点
        cluster_id = 0
        
        for i in range(n_samples):
            if cluster_labels[i] != -1:  # 已经被分配到聚类
                continue
                
            # 找到与当前点相似的所有点
            similar_points = np.where(similarity_matrix[i] > 0.8)[0]  # 使用固定阈值
            
            if len(similar_points) >= 2:  # 至少包含自己和另一个点
                cluster_labels[similar_points] = cluster_id
                cluster_id += 1
            else:
                cluster_labels[i] = -1  # 标记为噪声点
                
        return cluster_labels

    def _select_cluster_representatives(self, frame_indices: List[int],
                                     features: np.ndarray,
                                     cluster_labels: np.ndarray,
                                     shot_id: int) -> List[KeyFrame]:
        """从每个聚类中选择代表帧"""
        keyframes = []
        unique_clusters = set(cluster_labels)

        for cluster_id in unique_clusters:
            cluster_mask = cluster_labels == cluster_id
            cluster_indices = [frame_indices[i] for i, mask in enumerate(cluster_mask) if mask]
            cluster_features = features[cluster_mask]

            if cluster_id == -1:  # 噪声点
                # 噪声点单独处理，置信度较低
                for idx in cluster_indices:
                    keyframes.append(KeyFrame(
                        frame_idx=idx,
                        shot_id=shot_id,
                        confidence=0.6,
                        cluster_id=cluster_id,
                        selection_method="noise_point"
                    ))
            else:
                # 选择距离聚类中心最近的帧
                cluster_center = np.mean(cluster_features, axis=0)
                distances = [np.linalg.norm(feat - cluster_center)
                           for feat in cluster_features]

                best_idx = np.argmin(distances)
                representative_frame = cluster_indices[best_idx]

                # 计算置信度
                max_dist = max(distances) if len(distances) > 1 else 1.0
                confidence = 1.0 - (distances[best_idx] / (max_dist + 1e-8))
                confidence = max(0.3, min(1.0, confidence))

                keyframes.append(KeyFrame(
                    frame_idx=representative_frame,
                    shot_id=shot_id,
                    confidence=confidence,
                    cluster_id=cluster_id,
                    selection_method="cluster_center"
                ))

        return keyframes

    def _uniform_selection(self, frame_indices: List[int], shot_id: int) -> List[KeyFrame]:
        """均匀选择备用策略"""
        if not frame_indices:
            return []

        # 最多选择3个帧
        max_frames = min(3, len(frame_indices))
        selected_indices = np.linspace(0, len(frame_indices) - 1, max_frames, dtype=int)

        keyframes = []
        for i, idx in enumerate(selected_indices):
            keyframes.append(KeyFrame(
                frame_idx=frame_indices[idx],
                shot_id=shot_id,
                confidence=0.7,
                cluster_id=i,
                selection_method="uniform"
            ))

        return keyframes


class ShotKeyFrameSelector:
    """单个镜头的关键帧选择器"""

    def __init__(self, config: Config):
        """
        初始化镜头关键帧选择器

        Args:
            config: 配置对象
        """
        self.config = config
        self.motion_analyzer = MotionAnalyzer(config)
        self.clustering_selector = ClusteringSelector(config)

    def select_shot_keyframes(self, shot: ShotBoundary, features: Dict,
                            frame_info: List[FrameInfo]) -> List[KeyFrame]:
        """
        为单个镜头选择关键帧

        Args:
            shot: 镜头边界信息
            features: 特征字典
            frame_info: 帧信息列表

        Returns:
            选中的关键帧列表
        """
        try:
            # 找到属于当前镜头的帧
            shot_frame_indices = self._find_shot_frames(shot, frame_info)

            if len(shot_frame_indices) < 1:
                logger.warning(f"No sampled frames found for shot {shot.start_frame}-{shot.end_frame}")
                # 保底策略：为没有采样帧的镜头找到最接近的帧
                closest_frame_idx = self._find_closest_frame(shot, frame_info)
                if closest_frame_idx is not None:
                    logger.info(f"Using closest frame {closest_frame_idx} for shot {shot.start_frame}-{shot.end_frame}")
                    return [KeyFrame(
                        frame_idx=closest_frame_idx,
                        shot_id=shot.start_frame // 100,
                        confidence=0.3,  # 低置信度，表示这是保底选择
                        cluster_id=0,
                        selection_method="closest_fallback"
                    )]
                else:
                    logger.error(f"Could not find any frame for shot {shot.start_frame}-{shot.end_frame}")
                    return []

            # 候选帧选择
            candidates = self._select_candidates(
                shot_frame_indices, features, frame_info
            )

            if not candidates:
                logger.warning(f"No candidates found for shot, using first available frame")
                # 保底策略：如果没有候选帧，使用第一个可用帧
                candidates = [shot_frame_indices[0]]

            # 使用聚类方法选择最终关键帧
            keyframes = self.clustering_selector.select_keyframes_by_clustering(
                candidates, features["clip"], shot.start_frame // 100  # 简单的shot_id
            )

            # 限制关键帧数量
            max_frames = min(self.config.max_keyframes_per_shot,
                           max(1, shot.duration_frames // 30))

            if len(keyframes) > max_frames:
                # 按置信度排序并选择top-k
                keyframes.sort(key=lambda x: x.confidence, reverse=True)
                keyframes = keyframes[:max_frames]

            # 按帧索引排序
            keyframes.sort(key=lambda x: x.frame_idx)

            return keyframes

        except Exception as e:
            logger.error(f"Shot keyframe selection failed: {e}")
            return []

    def _find_shot_frames(self, shot: ShotBoundary,
                         frame_info: List[FrameInfo]) -> List[int]:
        """找到属于镜头的帧索引"""
        shot_frames = []

        for i, info in enumerate(frame_info):
            if shot.start_frame <= info.original_idx <= shot.end_frame:
                shot_frames.append(i)

        # 添加调试日志
        if len(shot_frames) == 0:
            logger.debug(f"No sampled frames found for shot {shot.start_frame}-{shot.end_frame} "
                        f"(duration: {shot.duration_frames} frames)")
        else:
            logger.debug(f"Found {len(shot_frames)} sampled frames for shot "
                        f"{shot.start_frame}-{shot.end_frame}")

        return shot_frames

    def _find_closest_frame(self, shot: ShotBoundary, frame_info: List[FrameInfo]) -> Optional[int]:
        """为没有采样帧的镜头找到最接近的帧"""
        if not frame_info:
            return None
            
        shot_center = (shot.start_frame + shot.end_frame) // 2
        closest_idx = None
        min_distance = float('inf')
        
        for i, info in enumerate(frame_info):
            # 计算到镜头中心的距离
            distance = abs(info.original_idx - shot_center)
            if distance < min_distance:
                min_distance = distance
                closest_idx = i
        
        return closest_idx

    def _select_candidates(self, shot_frame_indices: List[int],
                         features: Dict, frame_info: List[FrameInfo]) -> List[int]:
        """选择候选关键帧"""
        if len(shot_frame_indices) < 2:
            return shot_frame_indices

        try:
            candidates = set()

            # 1. 添加镜头边界帧
            candidates.add(shot_frame_indices[0])   # 开始
            candidates.add(shot_frame_indices[-1])  # 结束

            # 2. 添加运动峰值帧
            if len(shot_frame_indices) > 2:
                shot_motion = features["motion"][shot_frame_indices]
                motion_peaks = self.motion_analyzer.detect_motion_peaks(shot_motion)

                for peak_idx in motion_peaks:
                    if peak_idx < len(shot_frame_indices):
                        candidates.add(shot_frame_indices[peak_idx])

            # 3. 添加时间分割点
            if len(shot_frame_indices) > 5:
                # 三等分点
                third_points = [
                    len(shot_frame_indices) // 3,
                    2 * len(shot_frame_indices) // 3
                ]
                for point in third_points:
                    if 0 <= point < len(shot_frame_indices):
                        candidates.add(shot_frame_indices[point])

            # 4. 如果候选帧太少，添加更多
            if len(candidates) < 3 and len(shot_frame_indices) > 3:
                # 五等分
                step = len(shot_frame_indices) // 5
                for i in range(1, 5):
                    idx = i * step
                    if 0 <= idx < len(shot_frame_indices):
                        candidates.add(shot_frame_indices[idx])

            # 临时禁用新增检测逻辑，排查问题
            # if len(shot_frame_indices) > 3:
            #     semantic_candidates = self._detect_semantic_changes(
            #         shot_frame_indices, features["clip"]
            #     )
            #     candidates.update(semantic_candidates)

            # if len(shot_frame_indices) > 4:
            #     motion_candidates = self._detect_motion_pattern_changes(
            #         shot_frame_indices, features["motion"]
            #     )
            #     candidates.update(motion_candidates)

            return sorted(list(candidates))

        except Exception as e:
            logger.error(f"Candidate selection failed: {e}")
            # 备用：更密集的均匀采样
            step = max(1, len(shot_frame_indices) // 8)
            return shot_frame_indices[::step]
    
    def _detect_semantic_changes(self, frame_indices: List[int], 
                               clip_features: np.ndarray) -> List[int]:
        """检测语义内容变化的候选帧"""
        candidates = []
        
        try:
            # 计算相邻帧的CLIP特征相似度
            for i in range(1, len(frame_indices)):
                prev_idx = frame_indices[i-1]
                curr_idx = frame_indices[i]
                
                # 计算余弦相似度
                prev_feat = clip_features[prev_idx]
                curr_feat = clip_features[curr_idx]
                
                similarity = np.dot(prev_feat, curr_feat) / (
                    np.linalg.norm(prev_feat) * np.linalg.norm(curr_feat)
                )
                
                # 如果相似度低于阈值，说明有显著变化
                if similarity < 0.85:  # 比全局阈值更宽松
                    candidates.append(curr_idx)
                    
        except Exception as e:
            logger.warning(f"Semantic change detection failed: {e}")
            
        return candidates
    
    def _detect_motion_pattern_changes(self, frame_indices: List[int],
                                     motion_features: np.ndarray) -> List[int]:
        """检测运动模式变化的候选帧"""
        candidates = []
        
        try:
            if len(frame_indices) < 4:
                return candidates
                
            # 计算运动强度的变化率
            motion_values = motion_features[frame_indices]
            
            # 寻找运动强度变化的转折点
            for i in range(2, len(motion_values) - 2):
                # 计算局部梯度变化
                prev_grad = motion_values[i] - motion_values[i-1]
                curr_grad = motion_values[i+1] - motion_values[i]
                
                # 检测梯度变化（运动模式转变）
                if abs(prev_grad - curr_grad) > 0.5:
                    candidates.append(frame_indices[i])
                    
                # 检测运动强度峰值和谷值
                if (motion_values[i] > motion_values[i-1] and 
                    motion_values[i] > motion_values[i+1] and 
                    motion_values[i] > 0.8):  # 运动峰值
                    candidates.append(frame_indices[i])
                elif (motion_values[i] < motion_values[i-1] and 
                      motion_values[i] < motion_values[i+1] and
                      motion_values[i-1] > 1.0):  # 运动结束点
                    candidates.append(frame_indices[i])
                    
        except Exception as e:
            logger.warning(f"Motion pattern change detection failed: {e}")
            
        return candidates


class KeyFrameSelector:
    """
    统一关键帧选择器
    整合所有选择策略
    """

    def __init__(self, config: Config):
        """
        初始化关键帧选择器

        Args:
            config: 配置对象
        """
        self.config = config
        self.shot_selector = ShotKeyFrameSelector(config)

    def select_keyframes(self, shots: List[ShotBoundary], features: Dict,
                        frame_info: List[FrameInfo]) -> List[KeyFrame]:
        """
        选择视频的关键帧

        Args:
            shots: 镜头边界列表
            features: 特征字典
            frame_info: 帧信息列表

        Returns:
            最终的关键帧列表
        """
        logger.info(f"Starting keyframe selection for {len(shots)} shots")

        all_keyframes = []

        # 为每个镜头选择关键帧
        for shot_idx, shot in enumerate(shots):
            logger.debug(f"Processing shot {shot_idx}: {shot}")

            shot_keyframes = self.shot_selector.select_shot_keyframes(
                shot, features, frame_info
            )

            # 更新shot_id
            for kf in shot_keyframes:
                kf.shot_id = shot_idx

            all_keyframes.extend(shot_keyframes)

        logger.info(f"Selected {len(all_keyframes)} keyframes from all shots")

        # 统计每个镜头的关键帧数量
        shot_counts = {}
        for kf in all_keyframes:
            shot_counts[kf.shot_id] = shot_counts.get(kf.shot_id, 0) + 1
        
        shots_with_keyframes = len(shot_counts)
        shots_without_keyframes = len(shots) - shots_with_keyframes
        
        if shots_without_keyframes > 0:
            logger.warning(f"{shots_without_keyframes} shots have no keyframes out of {len(shots)} total shots")
        else:
            logger.info(f"All {len(shots)} shots have keyframes")

        # 跳过全局去重，直接返回所有关键帧
        logger.info(f"Skipping global deduplication, returning all {len(all_keyframes)} keyframes")
        
        # 按帧索引排序
        all_keyframes.sort(key=lambda x: x.frame_idx)

        return all_keyframes

    def get_selection_statistics(self, keyframes: List[KeyFrame]) -> Dict:
        """
        获取选择统计信息

        Args:
            keyframes: 关键帧列表

        Returns:
            统计信息字典
        """
        if not keyframes:
            return {}

        try:
            stats = {
                "total_keyframes": len(keyframes),
                "shots_covered": len(set(kf.shot_id for kf in keyframes)),
                "avg_confidence": np.mean([kf.confidence for kf in keyframes]),
                "methods_used": {},
                "clusters_found": len(set(kf.cluster_id for kf in keyframes
                                       if kf.cluster_id >= 0))
            }

            # 统计各种选择方法的使用情况
            for kf in keyframes:
                method = kf.selection_method
                stats["methods_used"][method] = stats["methods_used"].get(method, 0) + 1

            return stats

        except Exception as e:
            logger.error(f"Failed to calculate selection statistics: {e}")
            return {}