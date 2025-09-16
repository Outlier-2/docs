import React from 'react';

interface TreeNode {
  value: number | string;
  left?: TreeNode;
  right?: TreeNode;
  highlighted?: boolean;
  id: string;
}

interface TreeVisualizerProps {
  tree: TreeNode | null;
  title?: string;
  width?: number;
  height?: number;
}

interface NodePosition {
  x: number;
  y: number;
}

export const TreeVisualizer: React.FC<TreeVisualizerProps> = ({
  tree,
  title = '树结构可视化',
  width = 600,
  height = 400
}) => {
  const calculatePositions = (node: TreeNode | null, x: number, y: number, spacing: number): Map<string, NodePosition> => {
    const positions = new Map<string, NodePosition>();

    if (!node) return positions;

    positions.set(node.id, { x, y });

    if (node.left) {
      const leftPositions = calculatePositions(node.left, x - spacing, y + 80, spacing / 2);
      leftPositions.forEach((pos, id) => positions.set(id, pos));
    }

    if (node.right) {
      const rightPositions = calculatePositions(node.right, x + spacing, y + 80, spacing / 2);
      rightPositions.forEach((pos, id) => positions.set(id, pos));
    }

    return positions;
  };

  const renderEdges = (node: TreeNode | null, positions: Map<string, NodePosition>): JSX.Element[] => {
    const edges: JSX.Element[] = [];

    if (!node) return edges;

    const nodePos = positions.get(node.id);
    if (!nodePos) return edges;

    if (node.left) {
      const leftPos = positions.get(node.left.id);
      if (leftPos) {
        edges.push(
          <line
            key={`edge-${node.id}-${node.left.id}`}
            x1={nodePos.x}
            y1={nodePos.y}
            x2={leftPos.x}
            y2={leftPos.y}
            stroke="#94a3b8"
            strokeWidth="2"
          />
        );
        edges.push(...renderEdges(node.left, positions));
      }
    }

    if (node.right) {
      const rightPos = positions.get(node.right.id);
      if (rightPos) {
        edges.push(
          <line
            key={`edge-${node.id}-${node.right.id}`}
            x1={nodePos.x}
            y1={nodePos.y}
            x2={rightPos.x}
            y2={rightPos.y}
            stroke="#94a3b8"
            strokeWidth="2"
          />
        );
        edges.push(...renderEdges(node.right, positions));
      }
    }

    return edges;
  };

  const renderNodes = (node: TreeNode | null, positions: Map<string, NodePosition>): JSX.Element[] => {
    if (!node) return [];

    const nodePos = positions.get(node.id);
    if (!nodePos) return [];

    const nodes = [
      <g key={`node-${node.id}`}>
        <circle
          cx={nodePos.x}
          cy={nodePos.y}
          r="20"
          fill={node.highlighted ? "#ef4444" : "#3b82f6"}
          stroke="#1e40af"
          strokeWidth="2"
        />
        <text
          x={nodePos.x}
          y={nodePos.y + 5}
          textAnchor="middle"
          fill="white"
          fontSize="14"
          fontWeight="bold"
        >
          {node.value}
        </text>
      </g>
    ];

    if (node.left) {
      nodes.push(...renderNodes(node.left, positions));
    }
    if (node.right) {
      nodes.push(...renderNodes(node.right, positions));
    }

    return nodes;
  };

  if (!tree) {
    return (
      <div className="flex items-center justify-center h-64 bg-gray-50 rounded-lg">
        <p className="text-gray-500">树为空</p>
      </div>
    );
  }

  const positions = calculatePositions(tree, width / 2, 40, width / 4);

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      <h3 className="text-lg font-semibold mb-4 text-center">{title}</h3>
      <svg width={width} height={height} className="border border-gray-200 rounded">
        {renderEdges(tree, positions)}
        {renderNodes(tree, positions)}
      </svg>
    </div>
  );
};