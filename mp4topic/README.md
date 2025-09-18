# 视频关键帧提取器 (Video KeyFrame Extractor)

一个先进的模块化视频关键帧提取工具，集成了TransNetV2镜头检测、CLIP语义特征提取、光流运动分析和智能聚类算法。

## ✨ 主要特性

- 🎯 **智能镜头检测**: 支持TransNetV2神经网络和多种备用算法
- 🧠 **CLIP语义特征**: 使用OpenAI CLIP模型提取视觉语义特征
- 🔄 **光流运动分析**: 基于密集光流的运动特征提取
- 📊 **自适应聚类**: DBSCAN聚类算法自动选择代表性关键帧
- 🔧 **模块化设计**: 松耦合的组件设计，易于扩展和定制
- ⚙️ **灵活配置**: 丰富的配置选项和预设模式
- 📁 **完整输出**: 图像、元数据、CSV和详细报告

## 🚀 快速开始

### 安装依赖

```bash
# 基础依赖
pip install -r requirements.txt

# 安装CLIP模型
pip install git+https://github.com/openai/CLIP.git

# 可选：安装TransNetV2（推荐，用于更好的镜头检测）
pip install transnetv2 tensorflow
```

### 模型存储配置

默认情况下，所有模型文件将下载到项目目录下的 `models` 文件夹中，避免占用C盘空间：

```
项目目录/
├── models/                    # 模型存储目录（自动创建）
│   ├── clip/                 # CLIP模型文件
│   ├── transnetv2/          # TransNetV2模型文件
│   ├── torch/               # PyTorch模型缓存
│   └── tfhub/               # TensorFlow Hub缓存
├── src/
└── main.py
```

#### 自定义模型存储位置

如果需要自定义模型存储位置：

```python
# 方法1：通过配置设置
config = Config(models_dir="/path/to/your/models")

# 方法2：通过命令行参数
python main.py video.mp4 --models-dir /path/to/your/models

# 方法3：使用模型管理工具
python model_manager.py setup --directory /path/to/your/models
```

### 基本使用

```python
from video_keyframe_extractor import KeyFrameExtractor

# 使用默认配置
extractor = KeyFrameExtractor()
extractor.initialize()

# 提取关键帧
results = extractor.extract_keyframes("video.mp4", "output_dir")

# 清理资源
extractor.cleanup()

print(f"提取了 {len(results)} 个关键帧")
```

### 使用上下文管理器（推荐）

```python
from video_keyframe_extractor import KeyFrameExtractor

with KeyFrameExtractor() as extractor:
    results = extractor.extract_keyframes("video.mp4", "output_dir")
    print(f"提取了 {len(results)} 个关键帧")
```

## 📋 配置选项

### 预设配置

```python
from video_keyframe_extractor.core.config import ConfigManager

config_manager = ConfigManager()

# 可用预设：fast, balanced, quality, detailed
config = config_manager.create_preset_config("balanced")

with KeyFrameExtractor(config) as extractor:
    results = extractor.extract_keyframes("video.mp4", "output")
```

### 自定义配置

```python
from video_keyframe_extractor import Config

config = Config(
    target_fps=15,                    # 目标采样帧率
    similarity_threshold=0.95,        # 相似度去重阈值
    max_keyframes_per_shot=5,         # 每个镜头最大关键帧数
    use_transnet=True,                # 使用TransNetV2
    output_resolution=(768, 768),     # 输出图像分辨率
    image_quality=95                  # JPEG质量
)
```

### 配置文件

```python
# 保存配置
config.save_to_file("my_config.json")

# 加载配置
config = Config.from_file("my_config.json")
```

## 🛠️ 模型管理工具

项目提供了专门的模型管理工具，方便管理模型文件：

### 查看模型信息

```bash
# 查看模型状态和存储信息
python model_manager.py info

# 使用自定义模型目录
python model_manager.py info --models-dir /path/to/models
```

### 预下载模型

```bash
# 下载所有模型
python model_manager.py download --all

# 仅下载CLIP模型
python model_manager.py download --clip

# 仅下载TransNetV2模型
python model_manager.py download --transnet
```

### 清理模型缓存

```bash
# 清理所有模型缓存
python model_manager.py clean --confirm

# 清理特定模型缓存
python model_manager.py clean --type clip --confirm
```

### 设置自定义模型目录

```bash
# 设置自定义模型存储目录
python model_manager.py setup --directory /path/to/your/models
```

## 🔧 高级功能

### 批量处理

```python
video_paths = ["video1.mp4", "video2.mp4", "video3.mp4"]

with KeyFrameExtractor() as extractor:
    results = extractor.extract_keyframes_batch(video_paths, "batch_output")

    for video_path, video_results in results.items():
        print(f"{video_path}: {len(video_results)} 关键帧")
```

### 视频自适应配置

```python
from video_keyframe_extractor.core.config import ConfigManager

config_manager = ConfigManager()

# 根据视频特性自动优化配置
optimized_config = config_manager.optimize_for_video("video.mp4")

with KeyFrameExtractor(optimized_config) as extractor:
    results = extractor.extract_keyframes("video.mp4", "output")
```

### 特征分析

```python
with KeyFrameExtractor() as extractor:
    # 仅提取特征而不保存关键帧
    feature_data = extractor.feature_extractor.extract_all_features("video.mp4")

    # 获取特征摘要
    summary = extractor.feature_extractor.get_feature_summary(feature_data)
    print(f"运动特征统计: {summary['motion_features']}")
```

## 📁 输出格式

### 文件结构

```
output_dir/
├── video_shot00_keyframe_0001_frame_000123_1.50s.jpg
├── video_shot00_keyframe_0002_frame_000456_4.20s.jpg
├── video_shot01_keyframe_0003_frame_000789_7.80s.jpg
├── video_metadata.json          # 详细元数据
├── video_keyframes.csv          # CSV格式数据
└── video_summary.txt            # 文本摘要报告
```

### 元数据格式

```json
{
    "video_info": {
        "path": "video.mp4",
        "name": "video"
    },
    "total_keyframes": 15,
    "keyframes": [
        {
            "filename": "video_shot00_keyframe_0001.jpg",
            "frame_idx": 123,
            "timestamp": 1.50,
            "shot_id": 0,
            "confidence": 0.95,
            "selection_method": "cluster_center"
        }
    ]
}
```

## 🎛️ 配置参数说明

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `target_fps` | int | 10 | 目标采样帧率 |
| `similarity_threshold` | float | 0.9 | 相似度去重阈值 (0-1) |
| `max_keyframes_per_shot` | int | 5 | 每个镜头最大关键帧数 |
| `use_transnet` | bool | True | 是否使用TransNetV2 |
| `cluster_eps` | float | 0.15 | DBSCAN聚类参数 |
| `output_resolution` | tuple | (512, 512) | 输出图像分辨率 |
| `image_quality` | int | 95 | JPEG图像质量 (1-100) |

## 📖 示例代码

完整的使用示例请参考：

- [`examples/basic_usage.py`](examples/basic_usage.py) - 基础使用示例
- [`examples/advanced_usage.py`](examples/advanced_usage.py) - 高级功能示例

## 🏗️ 架构设计

```
video_keyframe_extractor/
├── core/                    # 核心模块
│   ├── config.py           # 配置管理
│   └── extractor.py        # 主提取器
├── models/                  # 模型管理
│   └── model_manager.py    # CLIP和TransNetV2管理
├── processors/              # 处理器模块
│   ├── shot_detector.py    # 镜头检测
│   ├── feature_extractor.py# 特征提取
│   └── keyframe_selector.py# 关键帧选择
└── utils/                   # 工具模块
    └── output_manager.py   # 输出管理
```

## 🔬 算法原理

### 1. 镜头检测
- **TransNetV2**: 深度学习网络，专门用于镜头边界检测
- **直方图差异**: 基于HSV颜色直方图的传统方法
- **光流分析**: 基于运动变化的检测方法

### 2. 特征提取
- **CLIP特征**: 512维语义特征向量，捕捉视觉内容的语义信息
- **运动特征**: 基于密集光流的运动强度分析
- **颜色特征**: HSV颜色直方图特征

### 3. 关键帧选择
- **候选帧选择**: 运动峰值、镜头边界、时间分割点
- **DBSCAN聚类**: 基于CLIP特征的自适应聚类
- **代表帧选择**: 选择最接近聚类中心的帧
- **全局去重**: 基于相似度阈值的重复帧过滤

## ⚡ 性能优化

### 内存使用
- 批量处理CLIP特征提取
- 分段加载大视频文件
- 及时释放不需要的数据

### 处理速度
- 多种采样策略减少处理帧数
- 并行化特征计算
- 预设配置快速启动

### 精度平衡
- 动态调整聚类参数
- 自适应阈值计算
- 多层次特征融合

## 🐛 故障排除

### 常见问题

1. **CLIP模型加载失败**
   ```bash
   pip install git+https://github.com/openai/CLIP.git
   ```

2. **TransNetV2不可用**
   ```bash
   pip install transnetv2 tensorflow
   ```

3. **内存不足**
   - 降低`target_fps`
   - 减小`output_resolution`
   - 使用"fast"预设配置

4. **GPU支持**
   - 确保安装了CUDA版本的PyTorch
   - 设置`device="cuda"`

### 日志调试

```python
config = Config(
    verbose=True,
    log_level="DEBUG"
)
```

## 🤝 贡献指南

欢迎贡献代码！请遵循以下步骤：

1. Fork 项目
2. 创建特性分支
3. 提交更改
4. 推送到分支
5. 创建 Pull Request

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🙏 致谢

- [OpenAI CLIP](https://github.com/openai/CLIP) - 语义特征提取
- [TransNetV2](https://github.com/soCzech/TransNetV2) - 镜头边界检测
- [OpenCV](https://opencv.org/) - 图像处理
- [scikit-learn](https://scikit-learn.org/) - 机器学习算法

## 📧 联系方式

如有问题或建议，请提交 Issue 或联系维护者。

---

**享受智能的视频关键帧提取！** 🎬✨