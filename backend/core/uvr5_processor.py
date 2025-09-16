import subprocess
import os
import shutil
from typing import List, Dict, Any

class UVR5BatchProcessor:
    """UVR5批量处理器"""
    
    def __init__(self, uvr5_path: str = None, model_name: str = "HP2_all_vocals"):
        self.uvr5_path = uvr5_path or self._find_uvr5_path()
        self.model_name = model_name
        
    def _find_uvr5_path(self) -> str:
        """自动查找UVR5安装路径"""
        possible_paths = [
            "C:/Program Files/Ultimate Vocal Remover",
            "C:/Program Files (x86)/Ultimate Vocal Remover",
            "./Ultimate-Vocal-Remover-5_v5",
            "../Ultimate-Vocal-Remover-5_v5"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # 如果找不到，返回默认路径（用户需要手动设置）
        return "./uvr5"
    
    def process_segments(self, segment_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """批量处理音频片段，只保留人声"""
        try:
            print("开始UVR5人声分离处理...")
            vocal_segments = []
            
            # 创建输出目录
            input_dir = "segments/"
            output_dir = "vocals/"
            os.makedirs(output_dir, exist_ok=True)
            
            # 检查UVR5是否可用
            if not self._check_uvr5_available():
                print("UVR5不可用，使用简单的音频复制作为替代")
                return self._fallback_process(segment_files)
            
            # 批量处理每个音频文件
            processed_count = 0
            for segment in segment_files:
                try:
                    vocal_file = self._process_single_file(segment['file'])
                    if vocal_file and os.path.exists(vocal_file):
                        vocal_segments.append({
                            **segment,
                            'vocal_file': vocal_file
                        })
                        processed_count += 1
                        
                        # 显示进度
                        if processed_count % 5 == 0:
                            print(f"已处理 {processed_count}/{len(segment_files)} 个音频片段")
                            
                except Exception as e:
                    print(f"处理文件 {segment['file']} 时出错: {e}")
                    continue
            
            print(f"UVR5处理完成，成功处理 {len(vocal_segments)} 个片段")
            return vocal_segments
            
        except Exception as e:
            print(f"UVR5批量处理时出错: {e}")
            # 降级处理
            return self._fallback_process(segment_files)
    
    def _check_uvr5_available(self) -> bool:
        """检查UVR5是否可用"""
        try:
            # 检查UVR5目录是否存在
            if not os.path.exists(self.uvr5_path):
                return False
            
            # 检查必要文件
            required_files = ["inference.py", "models"]
            for file in required_files:
                if not os.path.exists(os.path.join(self.uvr5_path, file)):
                    return False
            
            return True
        except:
            return False
    
    def _process_single_file(self, input_file: str) -> str:
        """处理单个音频文件"""
        try:
            filename = os.path.basename(input_file)
            output_file = os.path.join("vocals", f"vocal_{filename}")
            
            # UVR5命令行调用
            cmd = [
                'python', 
                os.path.join(self.uvr5_path, 'inference.py'),
                '--input', input_file,
                '--output', 'vocals/',
                '--model_name', self.model_name,
                '--format', 'wav'
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                # UVR5通常会生成带前缀的文件名
                possible_outputs = [
                    output_file,
                    os.path.join("vocals", f"vocals_{filename}"),
                    os.path.join("vocals", f"{self.model_name}_{filename}")
                ]
                
                for possible_file in possible_outputs:
                    if os.path.exists(possible_file):
                        return possible_file
            
            return None
            
        except Exception as e:
            print(f"处理单个文件时出错: {e}")
            return None
    
    def _fallback_process(self, segment_files: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """降级处理：直接复制原始音频文件"""
        print("使用降级处理模式（跳过人声分离）")
        vocal_segments = []
        
        os.makedirs("vocals", exist_ok=True)
        
        for segment in segment_files:
            try:
                filename = os.path.basename(segment['file'])
                vocal_file = os.path.join("vocals", f"vocal_{filename}")
                
                # 直接复制文件
                shutil.copy2(segment['file'], vocal_file)
                
                vocal_segments.append({
                    **segment,
                    'vocal_file': vocal_file
                })
                
            except Exception as e:
                print(f"复制文件 {segment['file']} 时出错: {e}")
                continue
        
        return vocal_segments
    
    def cleanup_segments(self):
        """清理分割的音频片段"""
        try:
            if os.path.exists("segments"):
                shutil.rmtree("segments")
                print("音频片段文件已清理")
        except Exception as e:
            print(f"清理音频片段时出错: {e}")


