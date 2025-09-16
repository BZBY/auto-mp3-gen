import React from 'react';
import { Progress, Card, Typography, Button, Space, Timeline, Tag, Spin } from 'antd';
import { ReloadOutlined, StopOutlined } from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;

interface ProcessStatus {
  is_processing: boolean;
  progress: number;
  message: string;
  has_result: boolean;
  has_error: boolean;
}

interface ProcessingProgressProps {
  status: ProcessStatus;
  onReset: () => void;
}

const ProcessingProgress: React.FC<ProcessingProgressProps> = ({ status, onReset }) => {
  const getProgressColor = () => {
    if (status.has_error) return '#ff4d4f';
    if (status.progress === 100) return '#52c41a';
    return '#1890ff';
  };

  const getProgressStatus = () => {
    if (status.has_error) return 'exception';
    if (status.progress === 100) return 'success';
    return 'active';
  };

  const processingSteps = [
    { title: '提取音轨', description: '从视频中提取完整音轨' },
    { title: '解析字幕', description: '分析字幕时间轴和文本内容' },
    { title: '音频切分', description: '按字幕时间轴精确切分音频' },
    { title: '人声分离', description: '使用UVR5分离人声和背景音' },
    { title: '特征提取', description: '提取每个片段的说话人特征' },
    { title: '说话人聚类', description: '识别和分组不同的说话人' },
    { title: '导出结果', description: '生成分类后的音频文件' },
  ];

  const getCurrentStepIndex = () => {
    const progress = status.progress;
    if (progress < 15) return 0;
    if (progress < 30) return 1;
    if (progress < 45) return 2;
    if (progress < 60) return 3;
    if (progress < 75) return 4;
    if (progress < 90) return 5;
    return 6;
  };

  const currentStepIndex = getCurrentStepIndex();

  return (
    <div className="progress-container">
      <Title level={3} style={{ textAlign: 'center', marginBottom: 32 }}>
        🚀 正在处理中...
      </Title>

      {/* 主进度条 */}
      <Card className="progress-card" style={{ marginBottom: 24 }}>
        <div style={{ textAlign: 'center', marginBottom: 24 }}>
          <Progress
            type="circle"
            percent={status.progress}
            size={120}
            strokeColor={getProgressColor()}
            status={getProgressStatus()}
            className="progress-circle"
          />
        </div>

        <div style={{ textAlign: 'center' }}>
          <Title level={4} style={{ margin: 0, color: getProgressColor() }}>
            {status.progress}%
          </Title>
          <Paragraph className="progress-message">
            {status.message || '正在初始化...'}
          </Paragraph>
        </div>

        {status.is_processing && (
          <div style={{ textAlign: 'center', marginTop: 16 }}>
            <Spin size="small" />
            <Text style={{ marginLeft: 8 }}>系统正在努力处理中，请耐心等待...</Text>
          </div>
        )}
      </Card>

      {/* 处理步骤时间线 */}
      <Card title="📋 处理步骤" className="progress-details">
        <Timeline mode="left">
          {processingSteps.map((step, index) => {
            let color = '#d9d9d9';
            let icon = null;
            
            if (index < currentStepIndex) {
              color = '#52c41a';
            } else if (index === currentStepIndex) {
              color = '#1890ff';
              icon = <Spin size="small" />;
            }

            return (
              <Timeline.Item 
                key={index} 
                color={color}
                dot={icon}
              >
                <div>
                  <Text strong style={{ color: index <= currentStepIndex ? '#000' : '#999' }}>
                    {step.title}
                  </Text>
                  <br />
                  <Text type="secondary" style={{ fontSize: 12 }}>
                    {step.description}
                  </Text>
                  {index === currentStepIndex && (
                    <Tag color="processing" style={{ marginLeft: 8 }}>
                      进行中
                    </Tag>
                  )}
                  {index < currentStepIndex && (
                    <Tag color="success" style={{ marginLeft: 8 }}>
                      已完成
                    </Tag>
                  )}
                </div>
              </Timeline.Item>
            );
          })}
        </Timeline>
      </Card>

      {/* 处理提示 */}
      <Card style={{ marginTop: 24, background: 'rgba(24, 144, 255, 0.05)' }}>
        <Title level={5}>💡 处理提示</Title>
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          <li>整个处理过程可能需要几分钟到几十分钟，具体时间取决于视频长度</li>
          <li>人声分离是最耗时的步骤，请耐心等待</li>
          <li>处理过程中请不要关闭浏览器或刷新页面</li>
          <li>如果长时间无响应，可以尝试重置系统</li>
        </ul>
      </Card>

      {/* 控制按钮 */}
      <div style={{ textAlign: 'center', marginTop: 32 }}>
        <Space>
          <Button 
            icon={<ReloadOutlined />}
            onClick={onReset}
            disabled={status.is_processing}
          >
            重置系统
          </Button>
          
          {status.has_error && (
            <Button 
              type="primary" 
              danger
              icon={<StopOutlined />}
              onClick={onReset}
            >
              停止处理
            </Button>
          )}
        </Space>
      </div>

      {/* 错误信息 */}
      {status.has_error && (
        <Card 
          style={{ 
            marginTop: 24, 
            borderColor: '#ff4d4f',
            background: 'rgba(255, 77, 79, 0.05)' 
          }}
        >
          <Title level={5} style={{ color: '#ff4d4f' }}>
            ❌ 处理失败
          </Title>
          <Text type="danger">
            {status.message}
          </Text>
          <div style={{ marginTop: 16 }}>
            <Button type="primary" danger onClick={onReset}>
              重新开始
            </Button>
          </div>
        </Card>
      )}
    </div>
  );
};

export default ProcessingProgress;

