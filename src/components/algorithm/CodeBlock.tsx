import React, { useState, useEffect, useRef } from 'react';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';

interface CodeBlockProps {
  code: string;
  language?: string;
  editable?: boolean;
  onCodeChange?: (code: string) => void;
  showLineNumbers?: boolean;
  highlightedLines?: number[];
}

export const CodeBlock: React.FC<CodeBlockProps> = ({
  code,
  language = 'python',
  editable = false,
  onCodeChange,
  showLineNumbers = true,
  highlightedLines = []
}) => {
  const [internalCode, setInternalCode] = useState(code);
  const [isEditing, setIsEditing] = useState(false);

  const lineProps = (lineNumber: number) => {
    if (highlightedLines.includes(lineNumber)) {
      return {
        style: {
          backgroundColor: 'rgba(255, 255, 0, 0.1)',
          display: 'block',
          margin: '0 -1em',
          padding: '0 1em',
          borderLeft: '3px solid #f1c40f'
        }
      };
    }
    return {};
  };

  if (editable) {
    return (
      <div className="relative">
        <button
          onClick={() => setIsEditing(!isEditing)}
          className="absolute top-2 right-2 bg-blue-500 text-white px-2 py-1 rounded text-sm z-10"
        >
          {isEditing ? '预览' : '编辑'}
        </button>

        {isEditing ? (
          <textarea
            value={internalCode}
            onChange={(e) => {
              setInternalCode(e.target.value);
              onCodeChange?.(e.target.value);
            }}
            className="w-full h-64 font-mono text-sm p-4 bg-gray-900 text-white rounded"
            style={{ tabSize: 4 }}
          />
        ) : (
          <SyntaxHighlighter
            language={language}
            style={vscDarkPlus}
            showLineNumbers={showLineNumbers}
            wrapLines={true}
            lineProps={lineProps}
            customStyle={{
              margin: 0,
              borderRadius: '0.5rem',
              fontSize: '0.875rem'
            }}
          >
            {internalCode}
          </SyntaxHighlighter>
        )}
      </div>
    );
  }

  return (
    <SyntaxHighlighter
      language={language}
      style={vscDarkPlus}
      showLineNumbers={showLineNumbers}
      wrapLines={true}
      lineProps={lineProps}
      customStyle={{
        margin: 0,
        borderRadius: '0.5rem',
        fontSize: '0.875rem'
      }}
    >
      {code}
    </SyntaxHighlighter>
  );
};