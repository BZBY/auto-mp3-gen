import os
import shutil
import json
from typing import Dict, List, Any
from datetime import datetime

class DialogueExtractor:
    """对话提取和导出器"""
    
    def __init__(self, output_dir: str = "output"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def export_speaker_dialogues(self, speakers_data: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        """导出每个说话人的对话"""
        try:
            print("正在导出说话人对话...")
            export_info = {
                'timestamp': datetime.now().isoformat(),
                'speakers': {},
                'total_segments': 0,
                'total_duration': 0
            }
            
            for speaker_id, segments in speakers_data.items():
                speaker_dir = os.path.join(self.output_dir, speaker_id)
                os.makedirs(speaker_dir, exist_ok=True)
                
                speaker_info = {
                    'segment_count': len(segments),
                    'files': [],
                    'total_duration': 0,
                    'subtitles': []
                }
                
                # 处理每个音频片段
                for i, segment in enumerate(segments):
                    try:
                        # 复制音频文件
                        audio_filename = f"{speaker_id}_{i:03d}.wav"
                        audio_file = os.path.join(speaker_dir, audio_filename)
                        
                        if os.path.exists(segment['vocal_file']):
                            shutil.copy2(segment['vocal_file'], audio_file)
                            
                            # 保存对应字幕信息
                            subtitle_filename = f"{speaker_id}_{i:03d}.txt"
                            subtitle_file = os.path.join(speaker_dir, subtitle_filename)
                            
                            duration = segment['end'] - segment['start']
                            
                            with open(subtitle_file, 'w', encoding='utf-8') as f:
                                f.write(f"片段: {i + 1}\n")
                                f.write(f"时间: {self._format_time(segment['start'])} - {self._format_time(segment['end'])}\n")
                                f.write(f"时长: {duration:.2f}秒\n")
                                f.write(f"字幕: {segment['text']}\n")
                                f.write(f"原始索引: {segment['index']}\n")
                            
                            # 记录文件信息
                            speaker_info['files'].append({
                                'audio': audio_filename,
                                'subtitle': subtitle_filename,
                                'text': segment['text'],
                                'start_time': segment['start'],
                                'end_time': segment['end'],
                                'duration': duration
                            })
                            
                            speaker_info['total_duration'] += duration
                            speaker_info['subtitles'].append(segment['text'])
                            
                        else:
                            print(f"警告：音频文件不存在 {segment['vocal_file']}")
                            
                    except Exception as e:
                        print(f"处理 {speaker_id} 的片段 {i} 时出错: {e}")
                        continue
                
                export_info['speakers'][speaker_id] = speaker_info
                export_info['total_segments'] += speaker_info['segment_count']
                export_info['total_duration'] += speaker_info['total_duration']
                
                print(f"  {speaker_id}: {len(speaker_info['files'])} 个文件")
            
            # 保存导出信息
            info_file = os.path.join(self.output_dir, "export_info.json")
            with open(info_file, 'w', encoding='utf-8') as f:
                json.dump(export_info, f, ensure_ascii=False, indent=2)
            
            print(f"对话导出完成，结果保存在 {self.output_dir}")
            return export_info
            
        except Exception as e:
            print(f"导出说话人对话时出错: {e}")
            raise
    
    def generate_summary(self, speakers_data: Dict[str, List[Dict[str, Any]]]) -> str:
        """生成处理摘要"""
        try:
            summary_file = os.path.join(self.output_dir, "summary.txt")
            
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write("动漫角色对话提取结果\n")
                f.write("=" * 50 + "\n\n")
                f.write(f"处理时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
                
                total_segments = 0
                total_duration = 0
                
                for speaker_id, segments in speakers_data.items():
                    speaker_duration = sum(seg['end'] - seg['start'] for seg in segments)
                    total_segments += len(segments)
                    total_duration += speaker_duration
                    
                    f.write(f"{speaker_id}:\n")
                    f.write(f"  片段数量: {len(segments)}\n")
                    f.write(f"  总时长: {self._format_time(speaker_duration)}\n")
                    f.write(f"  平均片段长度: {speaker_duration/len(segments):.2f}秒\n")
                    
                    # 显示前几个字幕作为示例
                    f.write("  示例字幕:\n")
                    for i, segment in enumerate(segments[:3]):
                        f.write(f"    {i+1}. {segment['text'][:50]}{'...' if len(segment['text']) > 50 else ''}\n")
                    
                    if len(segments) > 3:
                        f.write(f"    ... 还有 {len(segments) - 3} 个片段\n")
                    
                    f.write("\n")
                
                f.write("-" * 50 + "\n")
                f.write(f"总计:\n")
                f.write(f"  说话人数量: {len(speakers_data)}\n")
                f.write(f"  总片段数量: {total_segments}\n")
                f.write(f"  总时长: {self._format_time(total_duration)}\n")
                f.write(f"  平均每个说话人: {total_segments/len(speakers_data):.1f} 个片段\n")
            
            print(f"摘要文件已生成: {summary_file}")
            return summary_file
            
        except Exception as e:
            print(f"生成摘要时出错: {e}")
            return ""
    
    def _format_time(self, seconds: float) -> str:
        """格式化时间显示"""
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:05.2f}"
    
    def cleanup_temp_files(self):
        """清理临时文件"""
        try:
            temp_dirs = ["vocals", "segments"]
            for temp_dir in temp_dirs:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print(f"临时目录 {temp_dir} 已清理")
            
            if os.path.exists("temp_full_audio.wav"):
                os.remove("temp_full_audio.wav")
                print("临时音频文件已清理")
                
        except Exception as e:
            print(f"清理临时文件时出错: {e}")
    
    def get_export_status(self) -> Dict[str, Any]:
        """获取导出状态"""
        try:
            info_file = os.path.join(self.output_dir, "export_info.json")
            if os.path.exists(info_file):
                with open(info_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"获取导出状态时出错: {e}")
            return {}


