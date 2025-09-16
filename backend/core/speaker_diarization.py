import numpy as np
import librosa
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_similarity
import os
from typing import List, Dict, Any, Tuple
import warnings
warnings.filterwarnings("ignore")

class SpeakerDiarization:
    """说话人分离与聚类"""
    
    def __init__(self):
        print("初始化说话人识别模块...")
        # 使用简单的音频特征提取，避免复杂的深度学习模型依赖
        self.feature_extractor = SimpleAudioFeatureExtractor()
    
    def extract_speaker_embeddings(self, vocal_segments: List[Dict[str, Any]]) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """提取每个片段的说话人特征"""
        try:
            print("正在提取说话人特征...")
            embeddings = []
            valid_segments = []
            
            for i, segment in enumerate(vocal_segments):
                try:
                    # 提取音频特征
                    embedding = self.feature_extractor.extract_features(segment['vocal_file'])
                    if embedding is not None:
                        embeddings.append(embedding)
                        valid_segments.append(segment)
                    
                    # 显示进度
                    if (i + 1) % 10 == 0:
                        print(f"已提取 {i + 1}/{len(vocal_segments)} 个特征")
                        
                except Exception as e:
                    print(f"处理 {segment['vocal_file']} 时出错: {e}")
                    continue
            
            if len(embeddings) == 0:
                print("警告：未能提取到任何有效特征")
                return np.array([]), []
            
            embeddings_array = np.array(embeddings)
            print(f"特征提取完成，共 {len(valid_segments)} 个有效片段")
            return embeddings_array, valid_segments
            
        except Exception as e:
            print(f"提取说话人特征时出错: {e}")
            return np.array([]), []
    
    def cluster_speakers(self, embeddings: np.ndarray, n_clusters: int = None) -> np.ndarray:
        """聚类说话人"""
        try:
            if len(embeddings) == 0:
                return np.array([])
            
            print("正在进行说话人聚类...")
            
            if n_clusters is None:
                # 自动确定聚类数量
                n_clusters = min(8, max(2, len(embeddings) // 15))
            
            print(f"聚类数量: {n_clusters}")
            
            # 使用层次聚类
            clustering = AgglomerativeClustering(
                n_clusters=n_clusters,
                metric='cosine',
                linkage='average'
            )
            
            speaker_labels = clustering.fit_predict(embeddings)
            
            # 统计每个说话人的片段数量
            unique_labels, counts = np.unique(speaker_labels, return_counts=True)
            print(f"聚类完成，说话人分布: {dict(zip(unique_labels, counts))}")
            
            return speaker_labels
            
        except Exception as e:
            print(f"说话人聚类时出错: {e}")
            # 返回单一标签
            return np.zeros(len(embeddings), dtype=int)
    
    def organize_by_speakers(self, vocal_segments: List[Dict[str, Any]], speaker_labels: np.ndarray) -> Dict[str, List[Dict[str, Any]]]:
        """按说话人组织音频片段"""
        try:
            print("正在按说话人组织片段...")
            speakers = {}
            
            for segment, label in zip(vocal_segments, speaker_labels):
                speaker_id = f"speaker_{label:02d}"
                
                if speaker_id not in speakers:
                    speakers[speaker_id] = []
                
                speakers[speaker_id].append(segment)
            
            # 按时间排序每个说话人的片段
            for speaker_id in speakers:
                speakers[speaker_id].sort(key=lambda x: x['start'])
            
            print(f"组织完成，共识别出 {len(speakers)} 个说话人")
            for speaker_id, segments in speakers.items():
                print(f"  {speaker_id}: {len(segments)} 个片段")
            
            return speakers
            
        except Exception as e:
            print(f"组织说话人片段时出错: {e}")
            # 返回单一说话人
            return {"speaker_00": vocal_segments}


class SimpleAudioFeatureExtractor:
    """简单的音频特征提取器"""
    
    def __init__(self):
        self.sample_rate = 22050
        self.n_mfcc = 13
        self.n_fft = 2048
        self.hop_length = 512
    
    def extract_features(self, audio_file: str) -> np.ndarray:
        """提取音频特征"""
        try:
            if not os.path.exists(audio_file):
                return None
            
            # 加载音频
            y, sr = librosa.load(audio_file, sr=self.sample_rate)
            
            # 如果音频太短，返回None
            if len(y) < self.sample_rate * 0.5:  # 少于0.5秒
                return None
            
            # 提取MFCC特征
            mfcc = librosa.feature.mfcc(
                y=y, 
                sr=sr, 
                n_mfcc=self.n_mfcc,
                n_fft=self.n_fft,
                hop_length=self.hop_length
            )
            
            # 计算统计特征
            mfcc_mean = np.mean(mfcc, axis=1)
            mfcc_std = np.std(mfcc, axis=1)
            mfcc_delta = np.mean(librosa.feature.delta(mfcc), axis=1)
            
            # 提取谱质心
            spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)
            spectral_centroid_mean = np.mean(spectral_centroids)
            
            # 提取谱带宽
            spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
            spectral_bandwidth_mean = np.mean(spectral_bandwidth)
            
            # 提取过零率
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
            zcr_mean = np.mean(zero_crossing_rate)
            
            # 组合特征
            features = np.concatenate([
                mfcc_mean,
                mfcc_std,
                mfcc_delta,
                [spectral_centroid_mean, spectral_bandwidth_mean, zcr_mean]
            ])
            
            return features
            
        except Exception as e:
            print(f"提取音频特征时出错: {e}")
            return None


