import pysrt
import subprocess
from moviepy.editor import VideoFileClip
import os
import time
from typing import List, Dict, Any

class SubtitleBasedAudioSplitter:
    """基于字幕的音频分割器"""
    
    def __init__(self, video_path: str, subtitle_path: str):
        self.video_path = video_path
        self.subtitle_path = subtitle_path
        
    def extract_audio(self) -> str:
        """提取完整音轨"""
        try:
            print("正在提取音轨...")
            video = VideoFileClip(self.video_path)
            audio_path = "temp_full_audio.wav"
            video.audio.write_audiofile(audio_path, verbose=False, logger=None)
            video.close()  # 释放资源
            print(f"音轨提取完成: {audio_path}")
            return audio_path
        except Exception as e:
            print(f"提取音轨时出错: {e}")
            raise
    
    def parse_subtitles(self) -> List[Dict[str, Any]]:
        """解析字幕，支持双语"""
        try:
            print("正在解析字幕...")
            # 尝试不同编码
            encodings = ['utf-8', 'gbk', 'gb2312', 'utf-8-sig']
            subs = None
            
            for encoding in encodings:
                try:
                    subs = pysrt.open(self.subtitle_path, encoding=encoding)
                    break
                except UnicodeDecodeError:
                    continue
            
            if subs is None:
                raise ValueError("无法解析字幕文件，请检查编码格式")
            
            segments = []
            for sub in subs:
                start_time = sub.start.to_time()
                end_time = sub.end.to_time()
                text = sub.text.replace('\n', ' ').strip()
                
                # 过滤掉空字幕
                if text:
                    segments.append({
                        'start': self._time_to_seconds(start_time),
                        'end': self._time_to_seconds(end_time),
                        'text': text,
                        'index': sub.index
                    })
            
            print(f"字幕解析完成，共 {len(segments)} 个片段")
            return segments
        except Exception as e:
            print(f"解析字幕时出错: {e}")
            raise
    
    def _time_to_seconds(self, time_obj) -> float:
        """将时间对象转换为秒数"""
        return time_obj.hour * 3600 + time_obj.minute * 60 + time_obj.second + time_obj.microsecond / 1000000
    
    def split_audio_by_subtitles(self, audio_path: str, segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """按字幕时间轴切分音频"""
        try:
            print("正在按字幕切分音频...")
            segment_files = []
            
            # 确保输出目录存在
            os.makedirs("segments", exist_ok=True)
            
            for i, segment in enumerate(segments):
                start_time = segment['start']
                duration = segment['end'] - segment['start']
                
                # 添加缓冲时间，避免切断
                start_buffer = max(0, start_time - 0.1)  # 前缓冲100ms
                end_buffer = duration + 0.2  # 后缓冲200ms
                
                output_file = f"segments/segment_{i:04d}.wav"
                
                # 使用FFmpeg精确切分
                cmd = [
                    'ffmpeg', '-i', audio_path,
                    '-ss', str(start_buffer),
                    '-t', str(end_buffer),
                    '-acodec', 'pcm_s16le',
                    '-ar', '44100',
                    '-ac', '1',  # 转换为单声道
                    output_file, '-y'
                ]
                
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode == 0:
                        segment_files.append({
                            'file': output_file,
                            'text': segment['text'],
                            'index': segment['index'],
                            'start': start_time,
                            'end': segment['end']
                        })
                    else:
                        print(f"切分片段 {i} 时出错: {result.stderr}")
                except Exception as e:
                    print(f"处理片段 {i} 时出错: {e}")
                    continue
                
                # 显示进度
                if (i + 1) % 10 == 0:
                    print(f"已处理 {i + 1}/{len(segments)} 个片段")
            
            print(f"音频切分完成，成功处理 {len(segment_files)} 个片段")
            return segment_files
        except Exception as e:
            print(f"切分音频时出错: {e}")
            raise
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            if os.path.exists("temp_full_audio.wav"):
                os.remove("temp_full_audio.wav")
                print("临时音频文件已清理")
        except Exception as e:
            print(f"清理临时文件时出错: {e}")


