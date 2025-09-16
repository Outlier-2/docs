import React, { useState } from 'react';

interface ExerciseOption {
  id: string;
  text: string;
  correct: boolean;
  explanation?: string;
}

interface ExerciseBlockProps {
  title: string;
  question: string;
  type: 'multiple-choice' | 'code' | 'text';
  options?: ExerciseOption[];
  hint?: string;
  solution?: string;
  difficulty: 'easy' | 'medium' | 'hard';
}

export const ExerciseBlock: React.FC<ExerciseBlockProps> = ({
  title,
  question,
  type,
  options,
  hint,
  solution,
  difficulty
}) => {
  const [selectedAnswer, setSelectedAnswer] = useState<string | null>(null);
  const [showHint, setShowHint] = useState(false);
  const [showSolution, setShowSolution] = useState(false);
  const [userCode, setUserCode] = useState('');

  const getDifficultyColor = () => {
    switch (difficulty) {
      case 'easy': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'hard': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getDifficultyText = () => {
    switch (difficulty) {
      case 'easy': return '简单';
      case 'medium': return '中等';
      case 'hard': return '困难';
      default: return '未知';
    }
  };

  const checkAnswer = () => {
    if (type === 'multiple-choice' && selectedAnswer) {
      const correctOption = options?.find(opt => opt.id === selectedAnswer);
      return correctOption?.correct || false;
    }
    return null;
  };

  const isAnswerCorrect = checkAnswer();

  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold">{title}</h3>
        <span className={`px-2 py-1 rounded text-sm font-medium ${getDifficultyColor()}`}>
          {getDifficultyText()}
        </span>
      </div>

      <div className="mb-4">
        <p className="text-gray-700 mb-4">{question}</p>

        {type === 'multiple-choice' && options && (
          <div className="space-y-2">
            {options.map((option) => (
              <button
                key={option.id}
                onClick={() => setSelectedAnswer(option.id)}
                className={`w-full text-left p-3 rounded border transition-colors ${
                  selectedAnswer === option.id
                    ? isAnswerCorrect
                      ? 'border-green-500 bg-green-50'
                      : 'border-red-500 bg-red-50'
                    : 'border-gray-300 hover:border-gray-400'
                }`}
              >
                <span className="font-medium">{option.id}. </span>
                {option.text}
              </button>
            ))}
          </div>
        )}

        {type === 'code' && (
          <div className="mb-4">
            <textarea
              value={userCode}
              onChange={(e) => setUserCode(e.target.value)}
              placeholder="在这里编写你的代码..."
              className="w-full h-32 p-3 border border-gray-300 rounded font-mono text-sm"
            />
          </div>
        )}
      </div>

      {/* 结果反馈 */}
      {selectedAnswer && type === 'multiple-choice' && (
        <div className={`p-4 rounded mb-4 ${
          isAnswerCorrect ? 'bg-green-100 border border-green-300' : 'bg-red-100 border border-red-300'
        }`}>
          <p className={`font-medium ${isAnswerCorrect ? 'text-green-800' : 'text-red-800'}`}>
            {isAnswerCorrect ? '✓ 回答正确！' : '✗ 回答错误'}
          </p>
          {options?.find(opt => opt.id === selectedAnswer)?.explanation && (
            <p className="text-sm mt-2 text-gray-700">
              {options.find(opt => opt.id === selectedAnswer)?.explanation}
            </p>
          )}
        </div>
      )}

      {/* 辅助按钮 */}
      <div className="flex flex-wrap gap-2">
        {hint && (
          <button
            onClick={() => setShowHint(!showHint)}
            className="px-4 py-2 bg-blue-500 text-white rounded hover:bg-blue-600"
          >
            {showHint ? '隐藏提示' : '显示提示'}
          </button>
        )}

        {solution && (
          <button
            onClick={() => setShowSolution(!showSolution)}
            className="px-4 py-2 bg-purple-500 text-white rounded hover:bg-purple-600"
          >
            {showSolution ? '隐藏解答' : '显示解答'}
          </button>
        )}

        {type === 'code' && (
          <button
            onClick={() => {/* 运行代码逻辑 */}}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            运行代码
          </button>
        )}
      </div>

      {/* 提示内容 */}
      {showHint && hint && (
        <div className="mt-4 p-4 bg-blue-50 border border-blue-200 rounded">
          <h4 className="font-medium text-blue-800 mb-2">💡 提示</h4>
          <p className="text-sm text-blue-700">{hint}</p>
        </div>
      )}

      {/* 解答内容 */}
      {showSolution && solution && (
        <div className="mt-4 p-4 bg-purple-50 border border-purple-200 rounded">
          <h4 className="font-medium text-purple-800 mb-2">📝 解答</h4>
          <div className="text-sm text-purple-700">
            {solution}
          </div>
        </div>
      )}
    </div>
  );
};