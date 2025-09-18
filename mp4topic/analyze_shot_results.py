#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
镜头测试结果分析脚本
分析不同配置的测试结果
"""

import os
import json
import csv
from pathlib import Path
from typing import List, Dict, Any


def analyze_directory(test_dir: Path) -> Dict[str, Any]:
    """分析单个测试目录的结果"""
    result = {
        "name": test_dir.name,
        "keyframes_count": 0,
        "shots_covered": 0,
        "avg_confidence": 0,
        "file_size_mb": 0,
        "metadata": None
    }
    
    try:
        # 统计关键帧图片数量
        jpg_files = list(test_dir.glob("*.jpg"))
        result["keyframes_count"] = len(jpg_files)
        
        # 计算总文件大小
        total_size = sum(f.stat().st_size for f in jpg_files)
        result["file_size_mb"] = round(total_size / 1024 / 1024, 2)
        
        # 读取元数据
        metadata_file = test_dir / "metadata.json"
        if metadata_file.exists():
            with open(metadata_file, 'r', encoding='utf-8') as f:
                metadata = json.load(f)
                result["metadata"] = metadata
                
                # 提取统计信息
                if "keyframes" in metadata:
                    keyframes = metadata["keyframes"]
                    if keyframes:
                        result["shots_covered"] = len(set(kf.get("shot_id", 0) for kf in keyframes))
                        confidences = [kf.get("confidence", 0) for kf in keyframes]
                        result["avg_confidence"] = round(sum(confidences) / len(confidences), 3)
        
        # 读取CSV文件
        csv_file = test_dir / "keyframes.csv"
        if csv_file.exists() and not result["shots_covered"]:
            with open(csv_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                if rows:
                    result["shots_covered"] = len(set(row.get("shot_id", "0") for row in rows))
                    if not result["avg_confidence"]:
                        confidences = [float(row.get("confidence", 0)) for row in rows if row.get("confidence")]
                        if confidences:
                            result["avg_confidence"] = round(sum(confidences) / len(confidences), 3)
    
    except Exception as e:
        result["error"] = str(e)
    
    return result


def generate_comparison_report(results: List[Dict[str, Any]], output_file: str):
    """生成对比报告"""
    
    # 过滤有效结果
    valid_results = [r for r in results if r["keyframes_count"] > 0]
    valid_results.sort(key=lambda x: x["keyframes_count"], reverse=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 镜头参数测试结果分析\n\n")
        f.write(f"**分析时间**: {Path().absolute()}\n\n")
        
        # 结果对比表
        f.write("## 📊 结果对比\n\n")
        f.write("| 测试名称 | 关键帧数 | 镜头覆盖 | 平均置信度 | 文件大小(MB) | 效率评分* |\n")
        f.write("|---------|----------|----------|------------|-------------|----------|\n")
        
        for result in valid_results:
            # 计算效率评分 (关键帧数 × 置信度 / 处理时间的近似)
            efficiency = result["keyframes_count"] * result["avg_confidence"] if result["avg_confidence"] else 0
            
            f.write(f"| {result['name']} | {result['keyframes_count']} | "
                   f"{result['shots_covered']} | {result['avg_confidence']} | "
                   f"{result['file_size_mb']} | {efficiency:.1f} |\n")
        
        f.write("\n*效率评分 = 关键帧数 × 平均置信度\n\n")
        
        # 最佳配置推荐
        if valid_results:
            best_count = valid_results[0]  # 关键帧数最多
            best_confidence = max(valid_results, key=lambda x: x["avg_confidence"])
            best_efficiency = max(valid_results, key=lambda x: x["keyframes_count"] * x["avg_confidence"])
            
            f.write("## 🏆 推荐配置\n\n")
            f.write(f"### 关键帧数量最多\n")
            f.write(f"- **{best_count['name']}**: {best_count['keyframes_count']}个关键帧\n\n")
            
            f.write(f"### 置信度最高\n") 
            f.write(f"- **{best_confidence['name']}**: 平均置信度{best_confidence['avg_confidence']}\n\n")
            
            f.write(f"### 综合效率最佳\n")
            f.write(f"- **{best_efficiency['name']}**: 效率评分{best_efficiency['keyframes_count'] * best_efficiency['avg_confidence']:.1f}\n\n")
        
        # 参数影响分析
        f.write("## 🔍 参数影响分析\n\n")
        
        # 分析最小镜头长度影响
        min_shot_results = [r for r in valid_results if 'short' in r['name'] or 'long' in r['name'] or 'baseline' in r['name']]
        if len(min_shot_results) >= 2:
            f.write("### 镜头长度设置影响\n")
            for r in min_shot_results:
                f.write(f"- **{r['name']}**: {r['keyframes_count']}个关键帧, {r['shots_covered']}个镜头\n")
            f.write("\n")
        
        # 分析相似度阈值影响
        similarity_results = [r for r in valid_results if 'similar' in r['name'] or 'baseline' in r['name']]
        if len(similarity_results) >= 2:
            f.write("### 相似度阈值影响\n")
            for r in similarity_results:
                f.write(f"- **{r['name']}**: {r['keyframes_count']}个关键帧, 置信度{r['avg_confidence']}\n")
            f.write("\n")
        
        # 建议
        f.write("## 💡 使用建议\n\n")
        f.write("1. **追求数量**: 使用关键帧数最多的配置\n")
        f.write("2. **追求质量**: 使用平均置信度最高的配置\n") 
        f.write("3. **平衡考虑**: 使用综合效率最佳的配置\n")
        f.write("4. **存储考虑**: 注意文件大小，选择适合的配置\n\n")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="分析镜头测试结果")
    parser.add_argument("--test-dir", default=".", help="测试结果目录 (默认: 当前目录)")
    parser.add_argument("--output", default="shot_analysis_report.md", help="输出报告文件")
    
    args = parser.parse_args()
    
    test_base_dir = Path(args.test_dir)
    
    # 查找所有测试目录
    test_dirs = []
    for item in test_base_dir.iterdir():
        if item.is_dir() and (item.name.startswith("quick_test_") or item.name.startswith("shot_test_")):
            test_dirs.append(item)
    
    if not test_dirs:
        print("❌ 没有找到测试结果目录")
        print("💡 请确保运行了 quick_shot_test.py 或 batch_shot_test.py")
        return
    
    print(f"📁 找到 {len(test_dirs)} 个测试目录")
    
    # 分析每个目录
    results = []
    for test_dir in test_dirs:
        print(f"🔍 分析: {test_dir.name}")
        result = analyze_directory(test_dir)
        results.append(result)
        print(f"   关键帧: {result['keyframes_count']}, 镜头: {result['shots_covered']}")
    
    # 生成报告
    generate_comparison_report(results, args.output)
    
    # 显示简要结果
    valid_results = [r for r in results if r["keyframes_count"] > 0]
    if valid_results:
        valid_results.sort(key=lambda x: x["keyframes_count"], reverse=True)
        
        print(f"\n📊 测试结果排名 (按关键帧数量):")
        for i, result in enumerate(valid_results[:5], 1):
            print(f"{i}. {result['name']}: {result['keyframes_count']}个关键帧, "
                  f"{result['shots_covered']}个镜头, 置信度{result['avg_confidence']}")
    
    print(f"\n📝 详细报告已生成: {args.output}")


if __name__ == "__main__":
    main()
