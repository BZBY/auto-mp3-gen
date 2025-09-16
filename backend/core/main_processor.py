import os
import traceback
from typing import Dict, Any, Callable
from .subtitle_splitter import SubtitleBasedAudioSplitter
from .uvr5_processor import UVR5BatchProcessor
from .speaker_diarization import SpeakerDiarization
from .dialogue_extractor import DialogueExtractor

class MainProcessor:
    """主处理程序"""
    
    def __init__(self, uvr5_path: str = None):
        self.uvr5_path = uvr5_path
        self.progress_callback = None
        self.current_step = 0
        self.total_steps = 6
        
    def set_progress_callback(self, callback: Callable[[int, str], None]):
        """设置进度回调函数"""
        self.progress_callback = callback
    
    def _update_progress(self, message: str):
        """更新进度"""
        self.current_step += 1
        if self.progress_callback:
            progress = int((self.current_step / self.total_steps) * 100)
            self.progress_callback(progress, message)
        print(f"[{self.current_step}/{self.total_steps}] {message}")
    
    def process(self, video_path: str, subtitle_path: str, n_clusters: int = None) -> Dict[str, Any]:
        """主处理流程"""
        try:
            print("=" * 60)
            print("开始处理动漫角色对话提取")
            print("=" * 60)
            
            self.current_step = 0
            
            # 验证输入文件
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频文件不存在: {video_path}")
            if not os.path.exists(subtitle_path):
                raise FileNotFoundError(f"字幕文件不存在: {subtitle_path}")
            
            # 1. 字幕驱动的音频分割
            self._update_progress("正在提取音轨和解析字幕...")
            splitter = SubtitleBasedAudioSplitter(video_path, subtitle_path)
            audio_path = splitter.extract_audio()
            segments = splitter.parse_subtitles()
            
            if not segments:
                raise ValueError("未能解析到有效的字幕片段")
            
            self._update_progress("正在按字幕切分音频...")
            segment_files = splitter.split_audio_by_subtitles(audio_path, segments)
            
            if not segment_files:
                raise ValueError("未能成功切分音频片段")
            
            # 2. 批量UVR5处理
            self._update_progress("正在进行人声分离处理...")
            uvr5_processor = UVR5BatchProcessor(self.uvr5_path)
            vocal_segments = uvr5_processor.process_segments(segment_files)
            
            if not vocal_segments:
                raise ValueError("人声分离处理失败")
            
            # 3. 说话人分离
            self._update_progress("正在提取说话人特征...")
            diarizer = SpeakerDiarization()
            embeddings, valid_segments = diarizer.extract_speaker_embeddings(vocal_segments)
            
            if len(valid_segments) == 0:
                raise ValueError("未能提取到有效的说话人特征")
            
            self._update_progress("正在进行说话人聚类...")
            speaker_labels = diarizer.cluster_speakers(embeddings, n_clusters)
            speakers_data = diarizer.organize_by_speakers(valid_segments, speaker_labels)
            
            # 4. 导出结果
            self._update_progress("正在导出处理结果...")
            extractor = DialogueExtractor()
            export_info = extractor.export_speaker_dialogues(speakers_data)
            summary_file = extractor.generate_summary(speakers_data)
            
            # 5. 清理临时文件
            self._update_progress("正在清理临时文件...")
            self._cleanup_temp_files(splitter, uvr5_processor, extractor)
            
            # 返回处理结果
            result = {
                'success': True,
                'message': '处理完成！',
                'export_info': export_info,
                'summary_file': summary_file,
                'speakers_count': len(speakers_data),
                'total_segments': sum(len(segments) for segments in speakers_data.values()),
                'output_directory': extractor.output_dir
            }
            
            print("=" * 60)
            print("处理完成！")
            print(f"识别出 {result['speakers_count']} 个说话人")
            print(f"总共 {result['total_segments']} 个对话片段")
            print(f"结果保存在: {result['output_directory']}")
            print("=" * 60)
            
            return result
            
        except Exception as e:
            error_msg = f"处理过程中出错: {str(e)}"
            print(f"错误: {error_msg}")
            print("错误详情:")
            traceback.print_exc()
            
            # 尝试清理临时文件
            try:
                self._cleanup_temp_files()
            except:
                pass
            
            return {
                'success': False,
                'message': error_msg,
                'error_details': traceback.format_exc()
            }
    
    def _cleanup_temp_files(self, splitter=None, uvr5_processor=None, extractor=None):
        """清理临时文件"""
        try:
            if splitter:
                splitter.cleanup_temp_files()
            if uvr5_processor:
                uvr5_processor.cleanup_segments()
            if extractor:
                extractor.cleanup_temp_files()
        except Exception as e:
            print(f"清理临时文件时出错: {e}")
    
    def get_progress(self) -> Dict[str, Any]:
        """获取当前进度"""
        progress = int((self.current_step / self.total_steps) * 100)
        return {
            'current_step': self.current_step,
            'total_steps': self.total_steps,
            'progress': progress
        }


