import React, { useState, useEffect, useRef } from 'react';

interface Step {
  description: string;
  data: any;
  highlightedElements?: number[];
  codeLine?: number;
}

interface AlgorithmVisualizerProps {
  algorithm: string;
  steps: Step[];
  speed?: number;
  showControls?: boolean;
  onStepChange?: (step: number) => void;
  customRenderer?: (data: any, highlighted: number[]) => React.ReactNode;
}

export const AlgorithmVisualizer: React.FC<AlgorithmVisualizerProps> = ({
  algorithm,
  steps,
  speed = 1000,
  showControls = true,
  onStepChange,
  customRenderer
}) => {
  const [currentStep, setCurrentStep] = useState(0);
  const [isPlaying, setIsPlaying] = useState(false);
  const intervalRef = useRef<NodeJS.Timeout>();

  useEffect(() => {
    if (isPlaying) {
      intervalRef.current = setInterval(() => {
        setCurrentStep(prev => {
          const next = prev + 1;
          if (next >= steps.length) {
            setIsPlaying(false);
            return 0;
          }
          return next;
        });
      }, speed);
    }

    return () => {
      if (intervalRef.current) {
        clearInterval(intervalRef.current);
      }
    };
  }, [isPlaying, speed, steps.length]);

  useEffect(() => {
    onStepChange?.(currentStep);
  }, [currentStep, onStepChange]);

  const play = () => setIsPlaying(true);
  const pause = () => setIsPlaying(false);
  const reset = () => {
    setIsPlaying(false);
    setCurrentStep(0);
  };
  const stepForward = () => {
    if (currentStep < steps.length - 1) {
      setCurrentStep(prev => prev + 1);
    }
  };
  const stepBackward = () => {
    if (currentStep > 0) {
      setCurrentStep(prev => prev - 1);
    }
  };

  const currentStepData = steps[currentStep];

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="mb-4">
        <h3 className="text-lg font-semibold mb-2">{algorithm}</h3>
        <div className="bg-gray-100 rounded p-3 mb-4">
          <p className="text-sm text-gray-700">{currentStepData.description}</p>
        </div>
      </div>

      {/* 可视化区域 */}
      <div className="mb-6 min-h-[300px] bg-gray-50 rounded p-4">
        {customRenderer ? (
          customRenderer(currentStepData.data, currentStepData.highlightedElements || [])
        ) : (
          <div className="flex items-center justify-center h-full text-gray-500">
            <p>默认渲染器 - 请提供 customRenderer</p>
          </div>
        )}
      </div>

      {/* 步骤指示器 */}
      <div className="mb-4">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm text-gray-600">
            步骤 {currentStep + 1} / {steps.length}
          </span>
          <span className="text-sm text-gray-600">
            进度: {Math.round((currentStep + 1) / steps.length * 100)}%
          </span>
        </div>
        <div className="w-full bg-gray-200 rounded-full h-2">
          <div
            className="bg-blue-600 h-2 rounded-full transition-all duration-300"
            style={{ width: `${(currentStep + 1) / steps.length * 100}%` }}
          />
        </div>
      </div>

      {/* 控制按钮 */}
      {showControls && (
        <div className="flex justify-center space-x-2">
          <button
            onClick={reset}
            className="px-3 py-1 bg-gray-500 text-white rounded hover:bg-gray-600"
          >
            重置
          </button>
          <button
            onClick={stepBackward}
            disabled={currentStep === 0}
            className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300"
          >
            上一步
          </button>
          <button
            onClick={isPlaying ? pause : play}
            className="px-3 py-1 bg-green-500 text-white rounded hover:bg-green-600"
          >
            {isPlaying ? '暂停' : '播放'}
          </button>
          <button
            onClick={stepForward}
            disabled={currentStep === steps.length - 1}
            className="px-3 py-1 bg-blue-500 text-white rounded hover:bg-blue-600 disabled:bg-gray-300"
          >
            下一步
          </button>
        </div>
      )}

      {/* 速度控制 */}
      <div className="mt-4 flex items-center justify-center space-x-4">
        <span className="text-sm text-gray-600">速度:</span>
        <input
          type="range"
          min="100"
          max="2000"
          step="100"
          value={speed}
          onChange={(e) => speed !== parseInt(e.target.value) && speed !== 0}
          className="w-32"
        />
        <span className="text-sm text-gray-600">{speed}ms</span>
      </div>
    </div>
  );
};