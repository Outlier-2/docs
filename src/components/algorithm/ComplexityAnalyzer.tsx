import React, { useState, useEffect } from 'react';
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend
);

interface ComplexityData {
  n: number;
  time: number;
}

interface ComplexityAnalyzerProps {
  title: string;
  algorithm: string;
  theoreticalComplexity: string;
  data: ComplexityData[];
  color?: string;
}

export const ComplexityAnalyzer: React.FC<ComplexityAnalyzerProps> = ({
  title,
  algorithm,
  theoreticalComplexity,
  data,
  color = 'rgb(59, 130, 246)'
}) => {
  const [theoreticalData, setTheoreticalData] = useState<number[]>([]);

  useEffect(() => {
    // 计算理论值用于对比
    const theoretical = data.map(point => {
      switch (theoreticalComplexity) {
        case 'O(n)':
          return point.n;
        case 'O(n log n)':
          return point.n * Math.log2(point.n);
        case 'O(n²)':
          return point.n * point.n;
        case 'O(log n)':
          return Math.log2(point.n);
        case 'O(2ⁿ)':
          return Math.pow(2, point.n / 10); // 缩放以便显示
        default:
          return point.n;
      }
    });

    // 归一化理论数据
    const maxActual = Math.max(...data.map(d => d.time));
    const maxTheoretical = Math.max(...theoretical);
    const normalized = theoretical.map(t => (t / maxTheoretical) * maxActual);

    setTheoreticalData(normalized);
  }, [data, theoreticalComplexity]);

  const chartData = {
    labels: data.map(d => d.n),
    datasets: [
      {
        label: `${algorithm} (实际)`,
        data: data.map(d => d.time),
        borderColor: color,
        backgroundColor: color.replace('rgb', 'rgba').replace(')', ', 0.1)'),
        tension: 0.1,
      },
      {
        label: `${theoreticalComplexity} (理论)`,
        data: theoreticalData,
        borderColor: 'rgb(239, 68, 68)',
        backgroundColor: 'rgba(239, 68, 68, 0.1)',
        borderDash: [5, 5],
        tension: 0.1,
      }
    ]
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: 'top' as const,
      },
      title: {
        display: true,
        text: title,
      },
    },
    scales: {
      x: {
        title: {
          display: true,
          text: '输入大小 (n)'
        }
      },
      y: {
        title: {
          display: true,
          text: '执行时间 (秒)'
        }
      }
    }
  };

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <div className="mb-4">
        <h3 className="text-lg font-semibold">{title}</h3>
        <p className="text-sm text-gray-600">
          算法: {algorithm} | 理论复杂度: {theoreticalComplexity}
        </p>
      </div>

      <div className="h-[400px]">
        <Line data={chartData} options={options} />
      </div>

      {/* 统计信息 */}
      <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
        <div className="bg-gray-50 p-3 rounded">
          <h4 className="font-medium mb-2">测试结果</h4>
          <p>最大输入: {Math.max(...data.map(d => d.n))}</p>
          <p>最大时间: {Math.max(...data.map(d => d.time)).toFixed(4)}s</p>
          <p>平均时间: {(data.reduce((sum, d) => sum + d.time, 0) / data.length).toFixed(4)}s</p>
        </div>
        <div className="bg-gray-50 p-3 rounded">
          <h4 className="font-medium mb-2">复杂度分析</h4>
          <p>理论复杂度: {theoreticalComplexity}</p>
          <p>数据点数: {data.length}</p>
          <p>趋势匹配: 实际运行时间与理论曲线的对比</p>
        </div>
      </div>
    </div>
  );
};