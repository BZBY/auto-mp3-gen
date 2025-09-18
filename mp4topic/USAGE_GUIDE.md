# 关键帧提取器使用指南

## 问题解决方案

### 1. 图片压缩问题 ✅ 已解决

**问题：** 输出的关键帧被压缩到512x512，无法用于训练

**解决方案：**
- 添加了 `preserve_original_resolution` 配置选项
- 新增 `original` 预设模式
- 提供专用的 `extract_original_frames.py` 脚本

### 2. GPU加速 ⚡ 已优化

**问题：** 处理速度慢，希望使用GPU加速

**解决方案：**
- CLIP特征提取：已支持GPU加速
- 图像缩放：新增GPU加速支持（需要OpenCV CUDA版本）
- 光流计算：新增GPU加速支持（需要OpenCV CUDA版本）

## 使用方法

### 方法1：使用原图模式预设

```bash
# 使用原图预设（保持原始分辨率）
python main.py video.mp4 -p original

# 批量处理多个视频
python main.py *.mp4 -p original -b
```

### 方法2：使用专用脚本（推荐用于训练数据）

```bash
# 提取原图质量关键帧
python extract_original_frames.py video.mp4

# 指定输出目录
python extract_original_frames.py video.mp4 -o training_data/

# 批量处理
python extract_original_frames.py *.mp4 --batch

# 限制每镜头关键帧数
python extract_original_frames.py video.mp4 --max-frames 3
```

### 方法3：自定义配置

```bash
# 手动指定参数
python main.py video.mp4 \
  --preserve-original-resolution \
  --max-resolution 1920x1080 \
  --quality 100 \
  --target-fps 10
```

## 预设配置对比

| 预设模式 | 分辨率 | 图像质量 | 用途 | GPU加速 |
|---------|--------|----------|------|---------|
| fast | 512x512 | 95% | 快速预览 | ✅ |
| balanced | 512x512 | 95% | 一般使用 | ✅ |
| quality | 768x768 | 95% | 高质量 | ✅ |
| detailed | 1024x1024 | 95% | 最高质量 | ✅ |
| **original** | **原始尺寸** | **100%** | **训练数据** | ✅ |

## GPU加速说明

### 自动GPU检测
- 系统会自动检测CUDA设备
- CLIP模型会自动使用GPU（如果可用）
- 图像处理会尝试使用GPU加速

### GPU加速要求
1. **NVIDIA GPU** with CUDA support
2. **PyTorch with CUDA**：`pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118`
3. **OpenCV with CUDA**（可选，用于图像处理加速）：
   ```bash
   # 如果需要GPU图像处理加速
   pip uninstall opencv-python opencv-python-headless
   pip install opencv-contrib-python
   ```

### 检查GPU状态
```python
import torch
print("CUDA可用:", torch.cuda.is_available())
print("GPU设备数:", torch.cuda.device_count())

import cv2
print("OpenCV CUDA设备数:", cv2.cuda.getCudaEnabledDeviceCount())
```

## 性能对比

### 处理速度提升
- **CPU模式**: ~2-5 FPS
- **GPU模式**: ~10-20 FPS（取决于GPU性能）

### 图像质量对比
- **默认模式**: 512x512, 95%质量
- **原图模式**: 保持原始分辨率, 100%质量

## 配置文件示例

创建 `training_config.json`：

```json
{
  "preserve_original_resolution": true,
  "max_resolution": [1920, 1080],
  "image_quality": 100,
  "target_fps": 10,
  "use_transnet": true,
  "max_keyframes_per_shot": 5,
  "similarity_threshold": 0.9,
  "device": "auto",
  "verbose": true
}
```

使用配置文件：
```bash
python main.py video.mp4 -c training_config.json
```

## 常见问题

### Q: 如何确保输出原图质量？
A: 使用以下任一方法：
- `python main.py video.mp4 -p original`
- `python extract_original_frames.py video.mp4`
- 手动设置 `--preserve-original-resolution --quality 100`

### Q: GPU加速不生效？
A: 检查：
1. CUDA是否正确安装
2. PyTorch是否支持CUDA
3. OpenCV是否编译了CUDA支持

### Q: 内存不足？
A: 调整以下参数：
- 降低 `target_fps`
- 设置 `max_resolution`
- 减少 `max_keyframes_per_shot`

### Q: 如何定位原始帧？
A: 查看输出的元数据文件：
- CSV文件包含原始帧索引和时间戳
- JSON文件包含完整的帧信息
- 文件名包含帧索引和时间戳信息

## 输出文件说明

每次处理会生成：
- **图像文件**: `视频名_shot镜头号_keyframe关键帧序号_frame原始帧号_时间戳.jpg`
- **CSV文件**: 包含所有关键帧的详细信息
- **JSON文件**: 完整的元数据和配置信息
- **摘要文件**: 处理结果的文本摘要

示例文件名：
```
video_shot01_keyframe_0001_frame_000150_1.25s.jpg
```
表示：视频第1个镜头的第1个关键帧，对应原始第150帧，时间戳1.25秒
