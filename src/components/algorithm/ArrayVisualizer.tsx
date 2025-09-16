import React from 'react';

interface ArrayVisualizerProps {
  array: number[];
  highlightedIndices?: number[];
  comparedIndices?: [number, number];
  title?: string;
  maxHeight?: number;
}

export const ArrayVisualizer: React.FC<ArrayVisualizerProps> = ({
  array,
  highlightedIndices = [],
  comparedIndices,
  title = '数组可视化',
  maxHeight = 300
}) => {
  const maxValue = Math.max(...array);
  const minValue = Math.min(...array);
  const range = maxValue - minValue || 1;

  return (
    <div className="w-full">
      {title && (
        <h4 className="text-center font-medium mb-4 text-gray-700">{title}</h4>
      )}
      <div className="flex items-end justify-center space-x-1 h-[200px]">
        {array.map((value, index) => {
          const height = ((value - minValue) / range) * maxHeight;
          const isHighlighted = highlightedIndices.includes(index);
          const isCompared = comparedIndices?.includes(index);

          let bgColor = 'bg-blue-500';
          if (isHighlighted) bgColor = 'bg-red-500';
          if (isCompared) bgColor = 'bg-yellow-500';

          return (
            <div key={index} className="flex flex-col items-center">
              <div
                className={`${bgColor} transition-all duration-300 rounded-t-sm relative`}
                style={{
                  width: '20px',
                  height: `${height}px`,
                  minHeight: '4px'
                }}
              >
                <div className="absolute -top-6 left-1/2 transform -translate-x-1/2 text-xs font-medium">
                  {value}
                </div>
              </div>
              <div className="text-xs text-gray-600 mt-1">{index}</div>
            </div>
          );
        })}
      </div>
      <div className="mt-4 flex justify-center space-x-6 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-blue-500"></div>
          <span>正常</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-yellow-500"></div>
          <span>比较中</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-red-500"></div>
          <span>已选中</span>
        </div>
      </div>
    </div>
  );
};