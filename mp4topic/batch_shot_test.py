#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
批量镜头参数测试脚本
使用控制变量法测试不同镜头检测参数的效果
"""

import sys
import os
import json
import time
from pathlib import Path
from typing import Dict, List, Any
import subprocess

# 添加src目录到Python路径
sys.path.insert(0, str(Path(__file__).parent / "src"))

from video_keyframe_extractor import KeyFrameExtractor, Config


class BatchShotTester:
    """批量镜头参数测试器"""
    
    def __init__(self, video_path: str, output_base_dir: str = "shot_test_results"):
        self.video_path = video_path
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(exist_ok=True)
        self.results = []
        
    def create_base_config(self) -> Config:
        """创建基础配置"""
        return Config(
            # 保持原始分辨率
            preserve_original_resolution=True,
            max_resolution=(1920, 1080),
            image_quality=100,
            
            # 采样策略：固定使用随机采样
            sampling_strategy="random_seconds",
            random_sample_rate=4,
            random_select_count=3,
            
            # 启用所有检测方法
            use_transnet=True,
            
            # GPU优化
            gpu_batch_size=64,  # 减小批次大小，加快测试
            use_mixed_precision=True,
            gpu_memory_fraction=0.8,
            
            # 特征提取：快速模式
            skip_motion_features=True,  # 跳过光流加速测试
            skip_color_features=True,
            
            # 基础聚类设置
            cluster_eps=0.15,
            similarity_threshold=0.92,
            
            # 输出设置
            save_metadata=True,
            save_csv=True,
            verbose=False,  # 减少日志输出
        )
    
    def get_test_configs(self) -> List[Dict[str, Any]]:
        """定义测试配置列表"""
        test_configs = [
            # 测试1: 最小镜头长度对比
            {
                "name": "min_shot_15",
                "description": "最小镜头长度15帧(0.6秒)",
                "params": {"min_shot_duration": 15, "max_keyframes_per_shot": 8}
            },
            {
                "name": "min_shot_30", 
                "description": "最小镜头长度30帧(1.25秒) - 当前默认",
                "params": {"min_shot_duration": 30, "max_keyframes_per_shot": 8}
            },
            {
                "name": "min_shot_60",
                "description": "最小镜头长度60帧(2.5秒)",
                "params": {"min_shot_duration": 60, "max_keyframes_per_shot": 8}
            },
            {
                "name": "min_shot_120",
                "description": "最小镜头长度120帧(5秒)",
                "params": {"min_shot_duration": 120, "max_keyframes_per_shot": 8}
            },
            
            # 测试2: 每镜头关键帧数对比
            {
                "name": "max_frames_5",
                "description": "每镜头最多5个关键帧",
                "params": {"min_shot_duration": 30, "max_keyframes_per_shot": 5}
            },
            {
                "name": "max_frames_10",
                "description": "每镜头最多10个关键帧",
                "params": {"min_shot_duration": 30, "max_keyframes_per_shot": 10}
            },
            {
                "name": "max_frames_15",
                "description": "每镜头最多15个关键帧",
                "params": {"min_shot_duration": 30, "max_keyframes_per_shot": 15}
            },
            {
                "name": "max_frames_20",
                "description": "每镜头最多20个关键帧",
                "params": {"min_shot_duration": 30, "max_keyframes_per_shot": 20}
            },
            
            # 测试3: 相似度阈值对比
            {
                "name": "similarity_85",
                "description": "相似度阈值0.85(宽松去重)",
                "params": {"min_shot_duration": 30, "max_keyframes_per_shot": 10, "similarity_threshold": 0.85}
            },
            {
                "name": "similarity_90",
                "description": "相似度阈值0.90",
                "params": {"min_shot_duration": 30, "max_keyframes_per_shot": 10, "similarity_threshold": 0.90}
            },
            {
                "name": "similarity_95",
                "description": "相似度阈值0.95(严格去重)",
                "params": {"min_shot_duration": 30, "max_keyframes_per_shot": 10, "similarity_threshold": 0.95}
            },
            
            # 测试4: 检测方法对比
            {
                "name": "transnet_method",
                "description": "TransNetV2检测方法",
                "params": {"min_shot_duration": 30, "max_keyframes_per_shot": 10, "use_transnet": True}
            },
            {
                "name": "histogram_method",
                "description": "直方图检测方法",
                "params": {"min_shot_duration": 30, "max_keyframes_per_shot": 10, "use_transnet": False}
            },
            
            # 测试5: 组合优化测试
            {
                "name": "optimized_short",
                "description": "优化配置-短镜头敏感",
                "params": {"min_shot_duration": 20, "max_keyframes_per_shot": 12, "similarity_threshold": 0.88}
            },
            {
                "name": "optimized_long", 
                "description": "优化配置-长镜头偏好",
                "params": {"min_shot_duration": 45, "max_keyframes_per_shot": 15, "similarity_threshold": 0.90}
            },
        ]
        
        return test_configs
    
    def run_single_test(self, test_config: Dict[str, Any]) -> Dict[str, Any]:
        """运行单个测试"""
        print(f"\n🧪 测试: {test_config['name']}")
        print(f"📝 描述: {test_config['description']}")
        print(f"⚙️ 参数: {test_config['params']}")
        
        # 创建配置
        config = self.create_base_config()
        for param, value in test_config['params'].items():
            setattr(config, param, value)
        
        # 创建输出目录
        test_output_dir = self.output_base_dir / test_config['name']
        test_output_dir.mkdir(exist_ok=True)
        
        # 记录开始时间
        start_time = time.time()
        
        try:
            # 运行提取
            with KeyFrameExtractor(config) as extractor:
                results = extractor.extract_keyframes(self.video_path, str(test_output_dir))
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 收集结果统计
            result_stats = {
                "name": test_config['name'],
                "description": test_config['description'],
                "params": test_config['params'],
                "success": True,
                "keyframes_count": len(results) if results else 0,
                "processing_time": round(processing_time, 2),
                "shots_covered": len(set(r['shot_id'] for r in results)) if results else 0,
                "avg_confidence": round(sum(r['confidence'] for r in results) / len(results), 3) if results else 0,
                "output_dir": str(test_output_dir),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            
            print(f"✅ 成功: {result_stats['keyframes_count']}个关键帧, {result_stats['shots_covered']}个镜头, {processing_time:.1f}秒")
            
        except Exception as e:
            result_stats = {
                "name": test_config['name'],
                "description": test_config['description'],
                "params": test_config['params'],
                "success": False,
                "error": str(e),
                "processing_time": round(time.time() - start_time, 2),
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            print(f"❌ 失败: {e}")
        
        return result_stats
    
    def run_all_tests(self):
        """运行所有测试"""
        print(f"🚀 开始批量镜头参数测试")
        print(f"📹 视频: {self.video_path}")
        print(f"📁 输出目录: {self.output_base_dir}")
        
        test_configs = self.get_test_configs()
        print(f"📊 总共 {len(test_configs)} 个测试配置")
        
        # 运行所有测试
        for i, test_config in enumerate(test_configs, 1):
            print(f"\n{'='*50}")
            print(f"进度: {i}/{len(test_configs)}")
            
            result = self.run_single_test(test_config)
            self.results.append(result)
        
        # 保存结果
        self.save_results()
        self.generate_report()
        
        print(f"\n🎉 所有测试完成！")
        print(f"📊 结果报告: {self.output_base_dir / 'test_report.md'}")
    
    def save_results(self):
        """保存测试结果为JSON"""
        results_file = self.output_base_dir / "test_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        print(f"💾 结果已保存: {results_file}")
    
    def generate_report(self):
        """生成测试报告"""
        report_file = self.output_base_dir / "test_report.md"
        
        # 分析结果
        successful_tests = [r for r in self.results if r.get('success', False)]
        failed_tests = [r for r in self.results if not r.get('success', False)]
        
        # 排序：按关键帧数量
        successful_tests.sort(key=lambda x: x.get('keyframes_count', 0), reverse=True)
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("# 镜头参数测试报告\n\n")
            f.write(f"**测试视频**: {self.video_path}\n")
            f.write(f"**测试时间**: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**总测试数**: {len(self.results)}\n")
            f.write(f"**成功**: {len(successful_tests)}, **失败**: {len(failed_tests)}\n\n")
            
            # 成功测试结果表格
            if successful_tests:
                f.write("## 📊 测试结果对比\n\n")
                f.write("| 测试名称 | 描述 | 关键帧数 | 镜头数 | 平均置信度 | 处理时间(秒) |\n")
                f.write("|---------|------|----------|--------|------------|-------------|\n")
                
                for result in successful_tests:
                    f.write(f"| {result['name']} | {result['description']} | "
                           f"{result['keyframes_count']} | {result['shots_covered']} | "
                           f"{result['avg_confidence']} | {result['processing_time']} |\n")
            
            # 参数分析
            f.write("\n## 🔍 参数影响分析\n\n")
            
            # 最小镜头长度影响
            min_shot_tests = [r for r in successful_tests if 'min_shot' in r['name']]
            if min_shot_tests:
                f.write("### 最小镜头长度影响\n")
                for test in min_shot_tests:
                    min_duration = test['params'].get('min_shot_duration', 'unknown')
                    f.write(f"- **{min_duration}帧**: {test['keyframes_count']}个关键帧, "
                           f"{test['shots_covered']}个镜头\n")
                f.write("\n")
            
            # 每镜头关键帧数影响
            max_frames_tests = [r for r in successful_tests if 'max_frames' in r['name']]
            if max_frames_tests:
                f.write("### 每镜头最大关键帧数影响\n")
                for test in max_frames_tests:
                    max_frames = test['params'].get('max_keyframes_per_shot', 'unknown')
                    f.write(f"- **{max_frames}帧/镜头**: {test['keyframes_count']}个关键帧, "
                           f"平均置信度{test['avg_confidence']}\n")
                f.write("\n")
            
            # 推荐配置
            if successful_tests:
                best_test = successful_tests[0]  # 关键帧数最多的
                f.write("## 🏆 推荐配置\n\n")
                f.write(f"**最佳测试**: {best_test['name']}\n")
                f.write(f"**描述**: {best_test['description']}\n")
                f.write(f"**参数**: \n")
                for param, value in best_test['params'].items():
                    f.write(f"- `{param}`: {value}\n")
                f.write(f"\n**结果**: {best_test['keyframes_count']}个关键帧, "
                       f"{best_test['shots_covered']}个镜头, "
                       f"处理时间{best_test['processing_time']}秒\n")
            
            # 失败测试
            if failed_tests:
                f.write("\n## ❌ 失败的测试\n\n")
                for test in failed_tests:
                    f.write(f"- **{test['name']}**: {test.get('error', '未知错误')}\n")
        
        print(f"📝 报告已生成: {report_file}")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="批量镜头参数测试")
    parser.add_argument("video", help="测试视频文件路径")
    parser.add_argument("-o", "--output", default="shot_test_results", 
                       help="输出目录 (默认: shot_test_results)")
    
    args = parser.parse_args()
    
    # 检查视频文件
    if not Path(args.video).exists():
        print(f"❌ 视频文件不存在: {args.video}")
        sys.exit(1)
    
    # 运行批量测试
    tester = BatchShotTester(args.video, args.output)
    tester.run_all_tests()


if __name__ == "__main__":
    main()
