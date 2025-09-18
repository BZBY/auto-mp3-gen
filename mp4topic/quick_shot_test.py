#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速镜头参数测试脚本
测试几个关键参数的效果
"""

import sys
import subprocess
from pathlib import Path


def run_test(name: str, description: str, args: str):
    """运行单个测试"""
    print(f"\n🧪 {name}: {description}")
    print(f"⚙️ 参数: {args}")
    print("-" * 50)
    
    cmd = f"python extract_original_frames.py 1.mp4 -o quick_test_{name} {args}"
    print(f"🚀 执行: {cmd}")
    
    try:
        # Windows编码问题修复
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True, 
                              encoding='utf-8', errors='ignore')
        
        # 提取关键信息
        output_lines = result.stdout.split('\n')
        keyframe_count = 0
        shot_count = 0
        confidence = 0
        
        for line in output_lines:
            if "成功提取" in line and "关键帧" in line:
                print(f"✅ {line.strip()}")
                # 提取数字
                import re
                match = re.search(r'(\d+)\s*个.*关键帧', line)
                if match:
                    keyframe_count = int(match.group(1))
            elif "覆盖镜头数" in line:
                print(f"📊 {line.strip()}")
                match = re.search(r'(\d+)', line)
                if match:
                    shot_count = int(match.group(1))
            elif "平均置信度" in line:
                print(f"🎯 {line.strip()}")
                match = re.search(r'(\d+\.\d+)', line)
                if match:
                    confidence = float(match.group(1))
        
        # 显示汇总
        if keyframe_count > 0:
            print(f"📈 汇总: {keyframe_count}个关键帧, {shot_count}个镜头, 置信度{confidence}")
        
        if result.returncode != 0:
            print(f"❌ 错误: {result.stderr}")
            
    except Exception as e:
        print(f"❌ 执行失败: {e}")
        # 备用方案：直接运行不捕获输出
        print("🔄 尝试直接执行...")
        try:
            subprocess.run(cmd, shell=True)
            print("✅ 执行完成，请手动查看结果")
        except:
            print("❌ 直接执行也失败了")


def main():
    """主函数"""
    print("🚀 快速镜头参数测试")
    print("=" * 60)
    
    # 测试配置列表
    tests = [
        {
            "name": "baseline",
            "description": "基线测试(当前默认配置)",
            "args": ""
        },
        {
            "name": "short_shots",
            "description": "短镜头敏感(最小15帧)",
            "args": "--min-shot-duration 15"
        },
        {
            "name": "long_shots",
            "description": "长镜头偏好(最小60帧)",
            "args": "--min-shot-duration 60"
        },
        {
            "name": "more_frames",
            "description": "更多关键帧(每镜头最多20帧)",
            "args": "--max-frames 20"
        },
        {
            "name": "less_similar",
            "description": "宽松去重(相似度0.85)",
            "args": "--similarity-threshold 0.85"
        },
        {
            "name": "histogram",
            "description": "直方图检测方法",
            "args": "--use-histogram-detection"
        },
        {
            "name": "no_filter",
            "description": "不过滤镜头(保留所有)",
            "args": "--disable-shot-filtering"
        },
        {
            "name": "optimized",
            "description": "优化配置组合",
            "args": "--min-shot-duration 25 --max-frames 12 --similarity-threshold 0.88"
        }
    ]
    
    # 运行所有测试
    results_summary = []
    
    for i, test in enumerate(tests, 1):
        print(f"\n{'='*60}")
        print(f"测试 {i}/{len(tests)}")
        run_test(test["name"], test["description"], test["args"])
        
        # 简单的结果收集（你可以手动记录）
        print(f"📝 请记录 {test['name']} 的结果...")
        input("按回车继续下一个测试...")
    
    print(f"\n🎉 所有快速测试完成！")
    print(f"📁 结果保存在各个 quick_test_* 目录中")
    print(f"\n💡 建议:")
    print(f"1. 比较各个测试的关键帧数量")
    print(f"2. 查看关键帧的视觉质量")
    print(f"3. 选择最适合你需求的配置")


if __name__ == "__main__":
    main()
