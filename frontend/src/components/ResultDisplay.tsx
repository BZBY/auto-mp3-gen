import React, { useState } from 'react';
import { Card, Typography, Button, Space, Statistic, Row, Col, List, Tag, Collapse, message, Modal } from 'antd';
import { DownloadOutlined, ReloadOutlined, SoundOutlined, FileTextOutlined, FolderOpenOutlined } from '@ant-design/icons';

const { Title, Text, Paragraph } = Typography;
const { Panel } = Collapse;

interface ProcessResult {
  success: boolean;
  message: string;
  export_info?: {
    timestamp: string;
    speakers: { [key: string]: SpeakerInfo };
    total_segments: number;
    total_duration: number;
  };
  speakers_count?: number;
  total_segments?: number;
  output_directory?: string;
}

interface SpeakerInfo {
  segment_count: number;
  files: Array<{
    audio: string;
    subtitle: string;
    text: string;
    start_time: number;
    end_time: number;
    duration: number;
  }>;
  total_duration: number;
  subtitles: string[];
}

interface ResultDisplayProps {
  result: ProcessResult;
  onReset: () => void;
}

const ResultDisplay: React.FC<ResultDisplayProps> = ({ result, onReset }) => {
  const [selectedSpeaker, setSelectedSpeaker] = useState<string | null>(null);
  const [isModalVisible, setIsModalVisible] = useState(false);

  if (!result.success) {
    return (
      <div style={{ textAlign: 'center', padding: 40 }}>
        <Title level={3} type="danger">
          ❌ 处理失败
        </Title>
        <Paragraph>
          {result.message}
        </Paragraph>
        <Button type="primary" onClick={onReset}>
          重新开始
        </Button>
      </div>
    );
  }

  const exportInfo = result.export_info;
  const speakers = exportInfo?.speakers || {};

  const formatDuration = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${remainingSeconds.toString().padStart(2, '0')}`;
  };

  const formatTime = (seconds: number): string => {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = (seconds % 60).toFixed(2);
    return `${minutes}:${remainingSeconds.padStart(5, '0')}`;
  };

  const handleDownload = async (filePath: string) => {
    try {
      const response = await fetch(`http://localhost:5000/api/download/${filePath}`);
      if (response.ok) {
        const blob = await response.blob();
        const url = window.URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = filePath.split('/').pop() || 'download';
        a.click();
        window.URL.revokeObjectURL(url);
        message.success('下载成功');
      } else {
        throw new Error('下载失败');
      }
    } catch (error) {
      message.error('下载失败');
    }
  };

  const showSpeakerDetails = (speakerId: string) => {
    setSelectedSpeaker(speakerId);
    setIsModalVisible(true);
  };

  const getSpeakerColor = (index: number): string => {
    const colors = ['#1890ff', '#52c41a', '#faad14', '#f5222d', '#722ed1', '#13c2c2', '#eb2f96', '#fa8c16'];
    return colors[index % colors.length];
  };

  return (
    <div className="result-container">
      <Title level={3} style={{ textAlign: 'center', marginBottom: 32 }}>
        🎉 处理完成！
      </Title>

      {/* 结果摘要 */}
      <Card className="result-summary">
        <Title level={4} style={{ color: 'white', marginBottom: 16 }}>
          📊 处理结果摘要
        </Title>
        
        <Row gutter={16} className="summary-stats">
          <Col xs={12} sm={6}>
            <div className="stat-item">
              <span className="stat-number">{result.speakers_count || 0}</span>
              <div className="stat-label">说话人数量</div>
            </div>
          </Col>
          <Col xs={12} sm={6}>
            <div className="stat-item">
              <span className="stat-number">{result.total_segments || 0}</span>
              <div className="stat-label">对话片段</div>
            </div>
          </Col>
          <Col xs={12} sm={6}>
            <div className="stat-item">
              <span className="stat-number">
                {exportInfo?.total_duration ? formatDuration(exportInfo.total_duration) : '0:00'}
              </span>
              <div className="stat-label">总时长</div>
            </div>
          </Col>
          <Col xs={12} sm={6}>
            <div className="stat-item">
              <span className="stat-number">
                {exportInfo?.timestamp ? new Date(exportInfo.timestamp).toLocaleTimeString() : '--'}
              </span>
              <div className="stat-label">处理时间</div>
            </div>
          </Col>
        </Row>
      </Card>

      {/* 说话人列表 */}
      <div className="speakers-grid">
        {Object.entries(speakers).map(([speakerId, speakerInfo], index) => (
          <Card
            key={speakerId}
            className="speaker-card"
            hoverable
            onClick={() => showSpeakerDetails(speakerId)}
          >
            <div className="speaker-header" style={{ background: getSpeakerColor(index) }}>
              <Title level={5} style={{ color: 'white', margin: 0 }}>
                🎭 {speakerId.replace('_', ' ').toUpperCase()}
              </Title>
            </div>
            
            <div className="speaker-content">
              <Row gutter={16}>
                <Col span={12}>
                  <Statistic 
                    title="片段数量" 
                    value={speakerInfo.segment_count}
                    prefix={<SoundOutlined />}
                  />
                </Col>
                <Col span={12}>
                  <Statistic 
                    title="总时长" 
                    value={formatDuration(speakerInfo.total_duration)}
                    prefix="⏱️"
                  />
                </Col>
              </Row>

              <div className="subtitle-list">
                <Text strong>对话预览：</Text>
                {speakerInfo.subtitles.slice(0, 3).map((subtitle, idx) => (
                  <div key={idx} className="subtitle-item">
                    {subtitle.length > 50 ? subtitle.substring(0, 50) + '...' : subtitle}
                  </div>
                ))}
                {speakerInfo.subtitles.length > 3 && (
                  <div className="subtitle-item" style={{ fontStyle: 'italic', opacity: 0.7 }}>
                    还有 {speakerInfo.subtitles.length - 3} 个对话...
                  </div>
                )}
              </div>

              <div style={{ marginTop: 12, textAlign: 'center' }}>
                <Button type="primary" size="small" style={{ background: getSpeakerColor(index) }}>
                  查看详情
                </Button>
              </div>
            </div>
          </Card>
        ))}
      </div>

      {/* 操作按钮 */}
      <div className="action-buttons">
        <Space wrap>
          <Button 
            type="primary" 
            icon={<FolderOpenOutlined />}
            size="large"
            onClick={() => {
              if (result.output_directory) {
                // 在新窗口中打开文件夹（仅限本地环境）
                message.info(`结果保存在: ${result.output_directory}`);
              }
            }}
          >
            打开结果文件夹
          </Button>
          
          <Button 
            icon={<DownloadOutlined />}
            size="large"
            onClick={() => handleDownload('output/summary.txt')}
          >
            下载摘要文件
          </Button>
          
          <Button 
            icon={<ReloadOutlined />}
            size="large"
            onClick={onReset}
          >
            处理新文件
          </Button>
        </Space>
      </div>

      {/* 说话人详情模态框 */}
      <Modal
        title={`🎭 ${selectedSpeaker?.replace('_', ' ').toUpperCase()} 详细信息`}
        open={isModalVisible}
        onCancel={() => setIsModalVisible(false)}
        width={800}
        footer={[
          <Button key="close" onClick={() => setIsModalVisible(false)}>
            关闭
          </Button>
        ]}
      >
        {selectedSpeaker && speakers[selectedSpeaker] && (
          <div>
            <Row gutter={16} style={{ marginBottom: 16 }}>
              <Col span={8}>
                <Statistic title="片段数量" value={speakers[selectedSpeaker].segment_count} />
              </Col>
              <Col span={8}>
                <Statistic title="总时长" value={formatDuration(speakers[selectedSpeaker].total_duration)} />
              </Col>
              <Col span={8}>
                <Statistic 
                  title="平均片段长度" 
                  value={`${(speakers[selectedSpeaker].total_duration / speakers[selectedSpeaker].segment_count).toFixed(1)}s`}
                />
              </Col>
            </Row>

            <Collapse>
              <Panel header="📝 所有对话片段" key="1">
                <List
                  dataSource={speakers[selectedSpeaker].files}
                  renderItem={(file, index) => (
                    <List.Item
                      actions={[
                        <Button 
                          size="small" 
                          icon={<DownloadOutlined />}
                          onClick={() => handleDownload(`output/${selectedSpeaker}/${file.audio}`)}
                        >
                          音频
                        </Button>,
                        <Button 
                          size="small" 
                          icon={<FileTextOutlined />}
                          onClick={() => handleDownload(`output/${selectedSpeaker}/${file.subtitle}`)}
                        >
                          字幕
                        </Button>
                      ]}
                    >
                      <List.Item.Meta
                        title={
                          <Space>
                            <Tag color="blue">#{index + 1}</Tag>
                            <Text>{formatTime(file.start_time)} - {formatTime(file.end_time)}</Text>
                            <Tag>{file.duration.toFixed(1)}s</Tag>
                          </Space>
                        }
                        description={file.text}
                      />
                    </List.Item>
                  )}
                  pagination={{
                    pageSize: 5,
                    size: 'small'
                  }}
                />
              </Panel>
            </Collapse>
          </div>
        )}
      </Modal>

      {/* 使用提示 */}
      <Card style={{ marginTop: 24, background: 'rgba(82, 196, 26, 0.05)' }}>
        <Title level={5}>✅ 处理完成提示</Title>
        <ul style={{ margin: 0, paddingLeft: 20 }}>
          <li>所有音频文件已按说话人分类保存</li>
          <li>每个音频片段都有对应的字幕文件</li>
          <li>点击说话人卡片可查看详细信息和下载文件</li>
          <li>建议保存处理结果，以备后续使用</li>
        </ul>
      </Card>
    </div>
  );
};

export default ResultDisplay;

