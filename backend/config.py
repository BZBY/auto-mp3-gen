import os

class Config:
    """系统配置类"""
    
    # 基础配置
    DEBUG = os.environ.get('DEBUG', 'True').lower() == 'true'
    HOST = os.environ.get('HOST', '0.0.0.0')
    PORT = int(os.environ.get('PORT', 5000))
    
    # 文件上传配置
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER', 'uploads')
    MAX_CONTENT_LENGTH = int(os.environ.get('MAX_CONTENT_LENGTH', 2 * 1024 * 1024 * 1024))  # 2GB
    
    # 支持的文件格式
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mkv', 'mov', 'wmv', 'flv', 'm4v'}
    ALLOWED_SUBTITLE_EXTENSIONS = {'srt', 'ass', 'ssa', 'vtt'}
    
    # UVR5配置
    DEFAULT_UVR5_PATHS = [
        "C:/Program Files/Ultimate Vocal Remover",
        "C:/Program Files (x86)/Ultimate Vocal Remover",
        "./Ultimate-Vocal-Remover-5_v5",
        "../Ultimate-Vocal-Remover-5_v5",
        "/usr/local/bin/uvr5",
        "/opt/uvr5"
    ]
    
    # 处理配置
    DEFAULT_N_CLUSTERS = None  # 自动检测
    MIN_CLUSTERS = 2
    MAX_CLUSTERS = 10
    
    # 音频处理配置
    SAMPLE_RATE = 22050
    N_MFCC = 13
    N_FFT = 2048
    HOP_LENGTH = 512
    
    # 输出配置
    OUTPUT_FOLDER = os.environ.get('OUTPUT_FOLDER', 'output')
    TEMP_FOLDERS = ['segments', 'vocals', 'uploads']
    
    @classmethod
    def init_folders(cls):
        """初始化必要的文件夹"""
        folders = [cls.UPLOAD_FOLDER, cls.OUTPUT_FOLDER] + cls.TEMP_FOLDERS
        for folder in folders:
            os.makedirs(folder, exist_ok=True)
    
    @classmethod
    def get_config_info(cls):
        """获取配置信息"""
        return {
            'max_file_size': cls.MAX_CONTENT_LENGTH,
            'supported_video_formats': list(cls.ALLOWED_VIDEO_EXTENSIONS),
            'supported_subtitle_formats': list(cls.ALLOWED_SUBTITLE_EXTENSIONS),
            'upload_folder': cls.UPLOAD_FOLDER,
            'output_folder': cls.OUTPUT_FOLDER,
            'sample_rate': cls.SAMPLE_RATE,
            'debug': cls.DEBUG
        }
