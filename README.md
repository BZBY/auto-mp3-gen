# 🎭 动漫角色对话提取系统

基于字幕的动漫角色对话提取系统，使用UVR5进行人声分离和说话人识别技术，自动从动漫视频中提取并分类不同角色的对话。

## ✨ 功能特点

- 🎯 **精确分割**: 基于字幕时间轴精确切分，避免盲切造成的对话截断
- 🎵 **人声分离**: 集成UVR5进行高质量人声分离
- 🗣️ **说话人识别**: 使用机器学习算法自动识别和分类不同说话人
- 📝 **字幕同步**: 音频片段与字幕内容完美对应
- 🌐 **现代界面**: 基于React + Ant Design的美观用户界面
- 🚀 **批量处理**: 支持大文件和长视频的高效处理

## 🛠️ 技术栈

### 后端
- **Python 3.8+**
- **Flask** - Web API框架
- **MoviePy** - 视频处理
- **librosa** - 音频分析
- **scikit-learn** - 机器学习
- **pysrt** - 字幕解析

### 前端
- **React 18** + **TypeScript**
- **Ant Design** - UI组件库
- **Axios** - HTTP客户端

## 📋 系统要求

### 必需
- Python 3.8 或更高版本
- Node.js 16 或更高版本
- FFmpeg（用于音频处理）

### 可选
- Ultimate Vocal Remover 5 (UVR5) - 用于高质量人声分离

## 🚀 快速开始

### 1. 克隆项目
```bash
git clone <repository-url>
cd auto-mp3-gen
```

### 2. 安装依赖

#### 后端依赖
```bash
cd backend
pip install -r requirements.txt
```

#### 前端依赖
```bash
cd frontend
yarn install
# 或使用 npm install
```

#### 一键安装（推荐）
```bash
yarn install-all
```

### 3. 启动系统

#### 开发模式（同时启动前后端）
```bash
yarn dev
```

#### 分别启动
```bash
# 启动后端 (终端1)
yarn server

# 启动前端 (终端2)
yarn client
```

### 4. 访问系统
打开浏览器访问：http://localhost:3000

## 📖 使用说明

### 1. 准备文件
- **视频文件**: 支持 MP4, AVI, MKV, MOV, WMV, FLV, M4V 格式
- **字幕文件**: 支持 SRT, ASS, SSA, VTT 格式
- 确保视频和字幕文件的时间轴对应

### 2. 上传和配置
1. 在界面中上传视频和字幕文件
2. 配置处理参数：
   - **说话人数量**: 预期的角色数量（可选，系统会自动检测）
   - **UVR5路径**: Ultimate Vocal Remover 5 的安装路径（可选）

### 3. 开始处理
点击"开始处理"按钮，系统将自动执行以下步骤：
1. 提取音轨和解析字幕
2. 按字幕时间轴切分音频
3. 人声分离处理（如果配置了UVR5）
4. 提取说话人特征
5. 说话人聚类分析
6. 导出分类结果

### 4. 查看结果
处理完成后，可以：
- 查看每个说话人的对话统计
- 预览对话内容
- 下载分类后的音频文件
- 下载对应的字幕文件

## 📁 输出结构

```
output/
├── speaker_00/           # 说话人0的文件
│   ├── speaker_00_001.wav
│   ├── speaker_00_001.txt
│   └── ...
├── speaker_01/           # 说话人1的文件
│   ├── speaker_01_001.wav
│   ├── speaker_01_001.txt
│   └── ...
├── export_info.json      # 导出信息
└── summary.txt          # 处理摘要
```

## ⚙️ 高级配置

### UVR5集成
为了获得最佳的人声分离效果，建议安装Ultimate Vocal Remover 5：

1. 从[官方网站](https://github.com/Anjok07/ultimatevocalremovergui)下载UVR5
2. 安装到系统中
3. 在界面中配置UVR5的安装路径

### 性能优化
- 对于长视频，建议先截取测试片段验证效果
- 说话人数量设置合理可以提高聚类准确性
- 确保有足够的磁盘空间存储临时文件

## 🔧 开发

### 项目结构
```
auto-mp3-gen/
├── backend/              # 后端代码
│   ├── core/            # 核心处理模块
│   ├── app.py           # Flask应用
│   └── requirements.txt # Python依赖
├── frontend/            # 前端代码
│   ├── src/
│   │   ├── components/  # React组件
│   │   └── App.tsx      # 主应用
│   └── package.json     # Node.js依赖
└── package.json         # 项目配置
```

### API接口
- `POST /api/upload` - 文件上传
- `POST /api/process` - 开始处理
- `GET /api/progress` - 获取进度
- `GET /api/result` - 获取结果
- `GET /api/download/<path>` - 文件下载

## 🐛 故障排除

### 常见问题

1. **FFmpeg未安装**
   ```bash
   # Windows (使用chocolatey)
   choco install ffmpeg
   
   # macOS (使用homebrew)
   brew install ffmpeg
   
   # Linux (Ubuntu/Debian)
   sudo apt install ffmpeg
   ```

2. **Python依赖安装失败**
   ```bash
   # 使用国内镜像源
   pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
   ```

3. **前端启动失败**
   ```bash
   # 清除缓存重新安装
   cd frontend
   rm -rf node_modules package-lock.json
   yarn install
   ```

4. **处理过程中断**
   - 检查磁盘空间是否充足
   - 确认视频文件和字幕文件完整性
   - 查看后端日志获取详细错误信息

### 性能问题
- 长视频处理时间较长，这是正常现象
- 如果内存不足，可以尝试处理较短的视频片段
- 确保UVR5路径配置正确以获得最佳效果

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交Issue和Pull Request！

## 📞 支持

如有问题，请通过以下方式联系：
- 提交Issue到项目仓库
- 查看文档和FAQ
- 参与社区讨论

---

**享受从动漫中提取角色对话的乐趣！** 🎬✨