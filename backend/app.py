from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os
import json
import threading
import time
from werkzeug.utils import secure_filename
from core.main_processor import MainProcessor
from config import Config
from utils.system_check import SystemChecker

app = Flask(__name__)
CORS(app)  # 允许跨域请求

# 加载配置
app.config['UPLOAD_FOLDER'] = Config.UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = Config.MAX_CONTENT_LENGTH

# 初始化文件夹
Config.init_folders()

# 系统环境检查
print("正在进行系统环境检查...")
checker = SystemChecker()
system_status = checker.check_all()
checker.print_summary()

# 全局变量存储处理状态
processing_status = {
    'is_processing': False,
    'progress': 0,
    'message': '',
    'result': None,
    'error': None
}

def allowed_file(filename, allowed_extensions):
    """检查文件扩展名是否允许"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in allowed_extensions

def progress_callback(progress, message):
    """进度回调函数"""
    global processing_status
    processing_status['progress'] = progress
    processing_status['message'] = message

@app.route('/api/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'ok',
        'message': '动漫角色对话提取系统运行正常'
    })

@app.route('/api/upload', methods=['POST'])
def upload_files():
    """文件上传接口"""
    try:
        # 检查是否正在处理
        if processing_status['is_processing']:
            return jsonify({
                'success': False,
                'message': '系统正在处理其他任务，请稍后再试'
            }), 400
        
        # 检查文件
        if 'video' not in request.files or 'subtitle' not in request.files:
            return jsonify({
                'success': False,
                'message': '请同时上传视频文件和字幕文件'
            }), 400
        
        video_file = request.files['video']
        subtitle_file = request.files['subtitle']
        
        if video_file.filename == '' or subtitle_file.filename == '':
            return jsonify({
                'success': False,
                'message': '请选择有效的文件'
            }), 400
        
        # 验证文件类型
        if not allowed_file(video_file.filename, Config.ALLOWED_VIDEO_EXTENSIONS):
            return jsonify({
                'success': False,
                'message': f'不支持的视频格式，支持的格式: {", ".join(Config.ALLOWED_VIDEO_EXTENSIONS)}'
            }), 400
        
        if not allowed_file(subtitle_file.filename, Config.ALLOWED_SUBTITLE_EXTENSIONS):
            return jsonify({
                'success': False,
                'message': f'不支持的字幕格式，支持的格式: {", ".join(Config.ALLOWED_SUBTITLE_EXTENSIONS)}'
            }), 400
        
        # 保存文件
        video_filename = secure_filename(video_file.filename)
        subtitle_filename = secure_filename(subtitle_file.filename)
        
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video_filename)
        subtitle_path = os.path.join(app.config['UPLOAD_FOLDER'], subtitle_filename)
        
        video_file.save(video_path)
        subtitle_file.save(subtitle_path)
        
        # 获取其他参数
        n_clusters = request.form.get('n_clusters')
        if n_clusters:
            try:
                n_clusters = int(n_clusters)
            except ValueError:
                n_clusters = None
        
        uvr5_path = request.form.get('uvr5_path', '')
        
        return jsonify({
            'success': True,
            'message': '文件上传成功',
            'video_path': video_path,
            'subtitle_path': subtitle_path,
            'n_clusters': n_clusters,
            'uvr5_path': uvr5_path if uvr5_path else None
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'文件上传失败: {str(e)}'
        }), 500

@app.route('/api/process', methods=['POST'])
def start_processing():
    """开始处理接口"""
    try:
        # 检查是否正在处理
        if processing_status['is_processing']:
            return jsonify({
                'success': False,
                'message': '系统正在处理其他任务，请稍后再试'
            }), 400
        
        data = request.get_json()
        video_path = data.get('video_path')
        subtitle_path = data.get('subtitle_path')
        n_clusters = data.get('n_clusters')
        uvr5_path = data.get('uvr5_path')
        
        if not video_path or not subtitle_path:
            return jsonify({
                'success': False,
                'message': '缺少必要的文件路径参数'
            }), 400
        
        # 验证文件存在
        if not os.path.exists(video_path):
            return jsonify({
                'success': False,
                'message': f'视频文件不存在: {video_path}'
            }), 400
        
        if not os.path.exists(subtitle_path):
            return jsonify({
                'success': False,
                'message': f'字幕文件不存在: {subtitle_path}'
            }), 400
        
        # 重置状态
        global processing_status
        processing_status = {
            'is_processing': True,
            'progress': 0,
            'message': '正在初始化...',
            'result': None,
            'error': None
        }
        
        # 在后台线程中处理
        def process_in_background():
            try:
                processor = MainProcessor(uvr5_path)
                processor.set_progress_callback(progress_callback)
                result = processor.process(video_path, subtitle_path, n_clusters)
                
                processing_status['result'] = result
                processing_status['is_processing'] = False
                processing_status['progress'] = 100
                
                if result['success']:
                    processing_status['message'] = '处理完成！'
                else:
                    processing_status['error'] = result['message']
                    processing_status['message'] = '处理失败'
                    
            except Exception as e:
                processing_status['error'] = str(e)
                processing_status['is_processing'] = False
                processing_status['message'] = f'处理失败: {str(e)}'
        
        # 启动后台处理线程
        thread = threading.Thread(target=process_in_background)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'success': True,
            'message': '处理已开始，请通过进度接口查看处理状态'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'启动处理失败: {str(e)}'
        }), 500

@app.route('/api/progress', methods=['GET'])
def get_progress():
    """获取处理进度接口"""
    return jsonify({
        'is_processing': processing_status['is_processing'],
        'progress': processing_status['progress'],
        'message': processing_status['message'],
        'has_result': processing_status['result'] is not None,
        'has_error': processing_status['error'] is not None
    })

@app.route('/api/result', methods=['GET'])
def get_result():
    """获取处理结果接口"""
    if processing_status['result']:
        return jsonify(processing_status['result'])
    elif processing_status['error']:
        return jsonify({
            'success': False,
            'message': processing_status['error']
        }), 500
    else:
        return jsonify({
            'success': False,
            'message': '暂无处理结果'
        }), 404

@app.route('/api/download/<path:filename>', methods=['GET'])
def download_file(filename):
    """文件下载接口"""
    try:
        # 安全检查：只允许下载output目录下的文件
        if not filename.startswith('output/'):
            return jsonify({
                'success': False,
                'message': '无效的文件路径'
            }), 400
        
        if not os.path.exists(filename):
            return jsonify({
                'success': False,
                'message': '文件不存在'
            }), 404
        
        return send_file(filename, as_attachment=True)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'下载失败: {str(e)}'
        }), 500

@app.route('/api/reset', methods=['POST'])
def reset_system():
    """重置系统状态接口"""
    try:
        global processing_status
        processing_status = {
            'is_processing': False,
            'progress': 0,
            'message': '',
            'result': None,
            'error': None
        }
        
        return jsonify({
            'success': True,
            'message': '系统状态已重置'
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'message': f'重置失败: {str(e)}'
        }), 500

@app.route('/api/config', methods=['GET'])
def get_config():
    """获取系统配置接口"""
    config_info = Config.get_config_info()
    config_info['system_status'] = system_status
    return jsonify(config_info)

if __name__ == '__main__':
    print("启动动漫角色对话提取系统...")
    print(f"支持的视频格式: {', '.join(Config.ALLOWED_VIDEO_EXTENSIONS)}")
    print(f"支持的字幕格式: {', '.join(Config.ALLOWED_SUBTITLE_EXTENSIONS)}")
    print(f"最大文件大小: {Config.MAX_CONTENT_LENGTH // (1024*1024*1024)}GB")
    print("=" * 60)
    
    app.run(debug=Config.DEBUG, host=Config.HOST, port=Config.PORT)


