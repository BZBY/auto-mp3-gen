import React, { useState, useEffect } from 'react';
import { Layout, Typography, Card, Steps, message, Spin } from 'antd';
import FileUpload from './components/FileUpload';
import ProcessingProgress from './components/ProcessingProgress';
import ResultDisplay from './components/ResultDisplay';
import './App.css';

const { Header, Content, Footer } = Layout;
const { Title, Text } = Typography;
const { Step } = Steps;

interface ProcessResult {
  success: boolean;
  message: string;
  export_info?: any;
  speakers_count?: number;
  total_segments?: number;
  output_directory?: string;
}

interface ProcessStatus {
  is_processing: boolean;
  progress: number;
  message: string;
  has_result: boolean;
  has_error: boolean;
}

const App: React.FC = () => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isProcessing, setIsProcessing] = useState(false);
  const [processStatus, setProcessStatus] = useState<ProcessStatus>({
    is_processing: false,
    progress: 0,
    message: '',
    has_result: false,
    has_error: false
  });
  const [result, setResult] = useState<ProcessResult | null>(null);

  // 轮询获取处理进度
  useEffect(() => {
    let interval: NodeJS.Timeout;
    
    if (isProcessing) {
      interval = setInterval(async () => {
        try {
          const response = await fetch('http://localhost:5000/api/progress');
          const status: ProcessStatus = await response.json();
          setProcessStatus(status);
          
          if (!status.is_processing && (status.has_result || status.has_error)) {
            setIsProcessing(false);
            
            if (status.has_result) {
              // 获取处理结果
              const resultResponse = await fetch('http://localhost:5000/api/result');
              const resultData: ProcessResult = await resultResponse.json();
              setResult(resultData);
              setCurrentStep(2);
              
              if (resultData.success) {
                message.success('处理完成！');
              } else {
                message.error(resultData.message);
              }
            } else if (status.has_error) {
              message.error('处理失败：' + status.message);
              setCurrentStep(0);
            }
          }
        } catch (error) {
          console.error('获取进度失败:', error);
        }
      }, 1000);
    }
    
    return () => {
      if (interval) {
        clearInterval(interval);
      }
    };
  }, [isProcessing]);

  const handleUploadSuccess = () => {
    setCurrentStep(1);
    setIsProcessing(true);
    setResult(null);
    message.info('文件上传成功，开始处理...');
  };

  const handleReset = async () => {
    try {
      await fetch('http://localhost:5000/api/reset', {
        method: 'POST'
      });
      
      setCurrentStep(0);
      setIsProcessing(false);
      setResult(null);
      setProcessStatus({
        is_processing: false,
        progress: 0,
        message: '',
        has_result: false,
        has_error: false
      });
      
      message.info('系统已重置');
    } catch (error) {
      message.error('重置失败');
    }
  };

  const steps = [
    {
      title: '上传文件',
      description: '选择视频和字幕文件'
    },
    {
      title: '处理中',
      description: '提取角色对话'
    },
    {
      title: '完成',
      description: '查看结果'
    }
  ];

  return (
    <Layout className="app-layout">
      <Header className="app-header">
        <Title level={2} style={{ color: 'white', margin: 0 }}>
          🎭 动漫角色对话提取系统
        </Title>
      </Header>
      
      <Content className="app-content">
        <div className="content-container">
          <Card className="main-card">
            <div className="steps-container">
              <Steps current={currentStep} items={steps} />
            </div>
            
            <div className="content-area">
              {currentStep === 0 && (
                <FileUpload onSuccess={handleUploadSuccess} />
              )}
              
              {currentStep === 1 && (
                <ProcessingProgress 
                  status={processStatus}
                  onReset={handleReset}
                />
              )}
              
              {currentStep === 2 && result && (
                <ResultDisplay 
                  result={result}
                  onReset={handleReset}
                />
              )}
            </div>
          </Card>
        </div>
      </Content>
      
      <Footer className="app-footer">
        <Text type="secondary">
          基于字幕的动漫角色对话提取系统 - 使用UVR5进行人声分离和说话人识别
        </Text>
      </Footer>
    </Layout>
  );
};

export default App;