import React, { useState } from 'react';
import { Upload, Button, Form, InputNumber, Input, Card, message, Space, Typography, Divider } from 'antd';
import { InboxOutlined, UploadOutlined, SettingOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';

const { Dragger } = Upload;
const { Title, Text } = Typography;

interface FileUploadProps {
  onSuccess: () => void;
}

const FileUpload: React.FC<FileUploadProps> = ({ onSuccess }) => {
  const [form] = Form.useForm();
  const [videoFile, setVideoFile] = useState<File | null>(null);
  const [subtitleFile, setSubtitleFile] = useState<File | null>(null);
  const [uploading, setUploading] = useState(false);

  const videoUploadProps: UploadProps = {
    name: 'video',
    multiple: false,
    accept: '.mp4,.avi,.mkv,.mov,.wmv,.flv,.m4v',
    beforeUpload: (file) => {
      const isVideo = file.type.startsWith('video/') || 
        ['.mp4', '.avi', '.mkv', '.mov', '.wmv', '.flv', '.m4v'].some(ext => 
          file.name.toLowerCase().endsWith(ext)
        );
      
      if (!isVideo) {
        message.error('请选择视频文件！');
        return false;
      }
      
      const isLt2G = file.size / 1024 / 1024 / 1024 < 2;
      if (!isLt2G) {
        message.error('视频文件大小不能超过2GB！');
        return false;
      }
      
      setVideoFile(file);
      return false; // 阻止自动上传
    },
    onRemove: () => {
      setVideoFile(null);
    },
    fileList: videoFile ? [
      {
        uid: '-1',
        name: videoFile.name,
        status: 'done',
        size: videoFile.size,
      }
    ] : [],
  };

  const subtitleUploadProps: UploadProps = {
    name: 'subtitle',
    multiple: false,
    accept: '.srt,.ass,.ssa,.vtt',
    beforeUpload: (file) => {
      const isSubtitle = ['.srt', '.ass', '.ssa', '.vtt'].some(ext => 
        file.name.toLowerCase().endsWith(ext)
      );
      
      if (!isSubtitle) {
        message.error('请选择字幕文件（.srt, .ass, .ssa, .vtt）！');
        return false;
      }
      
      const isLt10M = file.size / 1024 / 1024 < 10;
      if (!isLt10M) {
        message.error('字幕文件大小不能超过10MB！');
        return false;
      }
      
      setSubtitleFile(file);
      return false; // 阻止自动上传
    },
    onRemove: () => {
      setSubtitleFile(null);
    },
    fileList: subtitleFile ? [
      {
        uid: '-1',
        name: subtitleFile.name,
        status: 'done',
        size: subtitleFile.size,
      }
    ] : [],
  };

  const handleSubmit = async (values: any) => {
    if (!videoFile || !subtitleFile) {
      message.error('请选择视频文件和字幕文件！');
      return;
    }

    setUploading(true);

    try {
      // 上传文件
      const formData = new FormData();
      formData.append('video', videoFile);
      formData.append('subtitle', subtitleFile);
      
      if (values.n_clusters) {
        formData.append('n_clusters', values.n_clusters.toString());
      }
      
      if (values.uvr5_path) {
        formData.append('uvr5_path', values.uvr5_path);
      }

      const uploadResponse = await fetch('http://localhost:5000/api/upload', {
        method: 'POST',
        body: formData,
      });

      const uploadResult = await uploadResponse.json();

      if (!uploadResult.success) {
        throw new Error(uploadResult.message);
      }

      // 开始处理
      const processResponse = await fetch('http://localhost:5000/api/process', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          video_path: uploadResult.video_path,
          subtitle_path: uploadResult.subtitle_path,
          n_clusters: uploadResult.n_clusters,
          uvr5_path: uploadResult.uvr5_path,
        }),
      });

      const processResult = await processResponse.json();

      if (!processResult.success) {
        throw new Error(processResult.message);
      }

      onSuccess();
    } catch (error: any) {
      message.error(error.message || '上传失败');
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="upload-area">
      <Title level={3} style={{ textAlign: 'center', marginBottom: 32 }}>
        上传文件
      </Title>

      <Form
        form={form}
        layout="vertical"
        onFinish={handleSubmit}
        initialValues={{
          n_clusters: 4,
        }}
      >
        <Space direction="vertical" size="large" style={{ width: '100%' }}>
          {/* 视频文件上传 */}
          <Card title="📹 视频文件" size="small">
            <Dragger {...videoUploadProps} className="upload-dragger">
              <p className="ant-upload-drag-icon">
                <InboxOutlined className="upload-icon" />
              </p>
              <p className="upload-text">
                点击或拖拽视频文件到此区域上传
              </p>
              <p className="upload-hint">
                支持格式：MP4, AVI, MKV, MOV, WMV, FLV, M4V（最大2GB）
              </p>
            </Dragger>
          </Card>

          {/* 字幕文件上传 */}
          <Card title="📝 字幕文件" size="small">
            <Dragger {...subtitleUploadProps} className="upload-dragger">
              <p className="ant-upload-drag-icon">
                <InboxOutlined className="upload-icon" />
              </p>
              <p className="upload-text">
                点击或拖拽字幕文件到此区域上传
              </p>
              <p className="upload-hint">
                支持格式：SRT, ASS, SSA, VTT（最大10MB）
              </p>
            </Dragger>
          </Card>

          {/* 高级配置 */}
          <Card 
            title={
              <span>
                <SettingOutlined /> 高级配置
              </span>
            } 
            size="small"
            className="config-form"
          >
            <div className="form-section">
              <Form.Item
                name="n_clusters"
                label="说话人数量"
                tooltip="预期的说话人数量，留空则自动检测"
              >
                <InputNumber
                  min={2}
                  max={10}
                  placeholder="自动检测"
                  style={{ width: '100%' }}
                />
              </Form.Item>

              <Form.Item
                name="uvr5_path"
                label="UVR5路径"
                tooltip="Ultimate Vocal Remover 5 的安装路径（可选）"
              >
                <Input
                  placeholder="例如: C:/Program Files/Ultimate Vocal Remover"
                  style={{ width: '100%' }}
                />
              </Form.Item>

              <Text type="secondary" style={{ fontSize: 12 }}>
                💡 提示：如果未安装UVR5，系统将跳过人声分离步骤
              </Text>
            </div>
          </Card>

          <Divider />

          {/* 提交按钮 */}
          <div style={{ textAlign: 'center' }}>
            <Button
              type="primary"
              size="large"
              icon={<UploadOutlined />}
              onClick={() => form.submit()}
              loading={uploading}
              disabled={!videoFile || !subtitleFile}
              style={{
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                border: 'none',
                borderRadius: 8,
                height: 48,
                fontSize: 16,
                fontWeight: 600,
                minWidth: 200,
              }}
            >
              {uploading ? '上传中...' : '开始处理'}
            </Button>
          </div>

          {/* 使用说明 */}
          <Card size="small" style={{ background: 'rgba(102, 126, 234, 0.05)' }}>
            <Title level={5}>📋 使用说明</Title>
            <ul style={{ margin: 0, paddingLeft: 20 }}>
              <li>请确保视频和字幕文件时间轴对应</li>
              <li>支持双语字幕，系统会自动处理</li>
              <li>处理时间取决于视频长度和说话人数量</li>
              <li>建议先安装UVR5以获得更好的人声分离效果</li>
            </ul>
          </Card>
        </Space>
      </Form>
    </div>
  );
};

export default FileUpload;

