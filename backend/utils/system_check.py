import os
import sys
import subprocess
import importlib
from typing import Dict, List, Tuple

class SystemChecker:
    """系统环境检查器"""
    
    def __init__(self):
        self.results = {
            'python': False,
            'ffmpeg': False,
            'dependencies': [],
            'uvr5': False,
            'disk_space': 0,
            'errors': [],
            'warnings': []
        }
    
    def check_all(self) -> Dict:
        """执行所有检查"""
        print("🔍 正在检查系统环境...")
        
        self.check_python()
        self.check_ffmpeg()
        self.check_dependencies()
        self.check_uvr5()
        self.check_disk_space()
        
        return self.results
    
    def check_python(self):
        """检查Python版本"""
        try:
            version = sys.version_info
            if version.major >= 3 and version.minor >= 8:
                self.results['python'] = True
                print(f"✅ Python {version.major}.{version.minor}.{version.micro}")
            else:
                self.results['errors'].append(f"Python版本过低: {version.major}.{version.minor}")
                print(f"❌ Python版本过低: {version.major}.{version.minor} (需要3.8+)")
        except Exception as e:
            self.results['errors'].append(f"Python检查失败: {e}")
            print(f"❌ Python检查失败: {e}")
    
    def check_ffmpeg(self):
        """检查FFmpeg"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                self.results['ffmpeg'] = True
                # 提取版本信息
                version_line = result.stdout.split('\n')[0]
                print(f"✅ {version_line}")
            else:
                self.results['errors'].append("FFmpeg未正确安装")
                print("❌ FFmpeg未正确安装")
        except subprocess.TimeoutExpired:
            self.results['errors'].append("FFmpeg检查超时")
            print("❌ FFmpeg检查超时")
        except FileNotFoundError:
            self.results['errors'].append("未找到FFmpeg，请先安装")
            print("❌ 未找到FFmpeg，请先安装")
        except Exception as e:
            self.results['errors'].append(f"FFmpeg检查失败: {e}")
            print(f"❌ FFmpeg检查失败: {e}")
    
    def check_dependencies(self):
        """检查Python依赖包"""
        required_packages = [
            'flask', 'flask_cors', 'moviepy', 'pysrt', 
            'librosa', 'numpy', 'sklearn', 'torch', 'torchaudio'
        ]
        
        for package in required_packages:
            try:
                if package == 'sklearn':
                    importlib.import_module('sklearn')
                else:
                    importlib.import_module(package)
                self.results['dependencies'].append(package)
                print(f"✅ {package}")
            except ImportError:
                self.results['warnings'].append(f"缺少依赖包: {package}")
                print(f"⚠️ 缺少依赖包: {package}")
    
    def check_uvr5(self):
        """检查UVR5安装"""
        possible_paths = [
            "C:/Program Files/Ultimate Vocal Remover",
            "C:/Program Files (x86)/Ultimate Vocal Remover",
            "./Ultimate-Vocal-Remover-5_v5",
            "../Ultimate-Vocal-Remover-5_v5"
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                # 检查关键文件
                inference_file = os.path.join(path, "inference.py")
                models_dir = os.path.join(path, "models")
                
                if os.path.exists(inference_file) and os.path.exists(models_dir):
                    self.results['uvr5'] = True
                    print(f"✅ UVR5 发现在: {path}")
                    return
        
        self.results['warnings'].append("未找到UVR5安装，将跳过人声分离")
        print("⚠️ 未找到UVR5安装，将跳过人声分离")
    
    def check_disk_space(self):
        """检查磁盘空间"""
        try:
            import shutil
            total, used, free = shutil.disk_usage(".")
            free_gb = free // (1024**3)
            self.results['disk_space'] = free_gb
            
            if free_gb < 2:
                self.results['warnings'].append(f"磁盘空间不足: {free_gb}GB")
                print(f"⚠️ 磁盘空间不足: {free_gb}GB (建议至少2GB)")
            else:
                print(f"✅ 可用磁盘空间: {free_gb}GB")
                
        except Exception as e:
            self.results['warnings'].append(f"磁盘空间检查失败: {e}")
            print(f"⚠️ 磁盘空间检查失败: {e}")
    
    def print_summary(self):
        """打印检查摘要"""
        print("\n" + "="*50)
        print("📋 系统环境检查摘要")
        print("="*50)
        
        print(f"Python: {'✅' if self.results['python'] else '❌'}")
        print(f"FFmpeg: {'✅' if self.results['ffmpeg'] else '❌'}")
        print(f"依赖包: {len(self.results['dependencies'])}/9 已安装")
        print(f"UVR5: {'✅' if self.results['uvr5'] else '⚠️'}")
        print(f"磁盘空间: {self.results['disk_space']}GB")
        
        if self.results['errors']:
            print(f"\n❌ 错误 ({len(self.results['errors'])}):")
            for error in self.results['errors']:
                print(f"  - {error}")
        
        if self.results['warnings']:
            print(f"\n⚠️ 警告 ({len(self.results['warnings'])}):")
            for warning in self.results['warnings']:
                print(f"  - {warning}")
        
        if not self.results['errors']:
            print("\n🎉 系统环境检查通过！可以开始使用了。")
        else:
            print("\n🚨 请解决上述错误后重试。")
        
        print("="*50)


def main():
    """主函数"""
    checker = SystemChecker()
    results = checker.check_all()
    checker.print_summary()
    
    return len(results['errors']) == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
