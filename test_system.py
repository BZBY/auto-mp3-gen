#!/usr/bin/env python3
"""
系统功能测试脚本
用于验证动漫角色对话提取系统的基本功能
"""

import requests
import json
import time
import os
from pathlib import Path

API_BASE = "http://localhost:5000/api"

def test_health_check():
    """测试健康检查接口"""
    print("🔍 测试健康检查接口...")
    try:
        response = requests.get(f"{API_BASE}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 健康检查通过: {data['message']}")
            return True
        else:
            print(f"❌ 健康检查失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 健康检查异常: {e}")
        return False

def test_config_api():
    """测试配置接口"""
    print("🔍 测试配置接口...")
    try:
        response = requests.get(f"{API_BASE}/config", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 配置获取成功:")
            print(f"   - 最大文件大小: {data.get('max_file_size', 0) // (1024**3)}GB")
            print(f"   - 视频格式: {', '.join(data.get('supported_video_formats', []))}")
            print(f"   - 字幕格式: {', '.join(data.get('supported_subtitle_formats', []))}")
            
            # 显示系统状态
            system_status = data.get('system_status', {})
            print(f"   - Python: {'✅' if system_status.get('python') else '❌'}")
            print(f"   - FFmpeg: {'✅' if system_status.get('ffmpeg') else '❌'}")
            print(f"   - UVR5: {'✅' if system_status.get('uvr5') else '⚠️'}")
            print(f"   - 依赖包: {len(system_status.get('dependencies', []))}/9")
            
            return True
        else:
            print(f"❌ 配置获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 配置获取异常: {e}")
        return False

def test_reset_api():
    """测试重置接口"""
    print("🔍 测试重置接口...")
    try:
        response = requests.post(f"{API_BASE}/reset", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 重置成功: {data['message']}")
            return True
        else:
            print(f"❌ 重置失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 重置异常: {e}")
        return False

def test_progress_api():
    """测试进度接口"""
    print("🔍 测试进度接口...")
    try:
        response = requests.get(f"{API_BASE}/progress", timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ 进度获取成功:")
            print(f"   - 处理中: {data.get('is_processing', False)}")
            print(f"   - 进度: {data.get('progress', 0)}%")
            print(f"   - 消息: {data.get('message', 'N/A')}")
            return True
        else:
            print(f"❌ 进度获取失败: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ 进度获取异常: {e}")
        return False

def test_backend_imports():
    """测试后端模块导入"""
    print("🔍 测试后端模块导入...")
    try:
        # 切换到backend目录
        import sys
        backend_path = os.path.join(os.getcwd(), 'backend')
        if backend_path not in sys.path:
            sys.path.insert(0, backend_path)
        
        # 测试核心模块导入
        from core.subtitle_splitter import SubtitleBasedAudioSplitter
        from core.uvr5_processor import UVR5BatchProcessor
        from core.speaker_diarization import SpeakerDiarization
        from core.dialogue_extractor import DialogueExtractor
        from core.main_processor import MainProcessor
        from config import Config
        
        print("✅ 所有后端模块导入成功")
        return True
    except ImportError as e:
        print(f"❌ 模块导入失败: {e}")
        return False
    except Exception as e:
        print(f"❌ 模块导入异常: {e}")
        return False

def create_test_files():
    """创建测试用的示例文件"""
    print("🔍 创建测试文件...")
    
    # 创建示例字幕文件
    srt_content = """1
00:00:01,000 --> 00:00:03,000
这是第一句对话

2
00:00:04,000 --> 00:00:06,000
这是第二句对话

3
00:00:07,000 --> 00:00:09,000
这是第三句对话
"""
    
    test_dir = Path("test_files")
    test_dir.mkdir(exist_ok=True)
    
    srt_file = test_dir / "test.srt"
    with open(srt_file, 'w', encoding='utf-8') as f:
        f.write(srt_content)
    
    print(f"✅ 测试字幕文件已创建: {srt_file}")
    
    # 提示用户准备视频文件
    print("ℹ️ 请手动准备一个测试视频文件 (test_files/test.mp4) 进行完整测试")
    
    return str(srt_file)

def run_all_tests():
    """运行所有测试"""
    print("🚀 开始系统功能测试")
    print("=" * 60)
    
    tests = [
        ("后端模块导入", test_backend_imports),
        ("健康检查接口", test_health_check),
        ("配置接口", test_config_api),
        ("进度接口", test_progress_api),
        ("重置接口", test_reset_api),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 40)
        success = test_func()
        results.append((test_name, success))
        time.sleep(1)
    
    # 创建测试文件
    print(f"\n📋 创建测试文件")
    print("-" * 40)
    srt_file = create_test_files()
    
    # 总结
    print("\n" + "=" * 60)
    print("📊 测试结果总结")
    print("=" * 60)
    
    passed = 0
    for test_name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"{test_name}: {status}")
        if success:
            passed += 1
    
    print(f"\n总计: {passed}/{len(results)} 项测试通过")
    
    if passed == len(results):
        print("🎉 所有测试通过！系统已准备就绪。")
        print("\n💡 下一步:")
        print("1. 启动系统: python start.py 或 yarn dev")
        print("2. 打开浏览器访问: http://localhost:3000")
        print("3. 上传视频和字幕文件开始使用")
    else:
        print("⚠️ 部分测试失败，请检查系统配置。")
    
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
