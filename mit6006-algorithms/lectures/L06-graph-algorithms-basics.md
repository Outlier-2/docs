# L06 - 图算法基础

## 学习目标
- 掌握图的基本概念和表示方法
- 理解图的遍历算法（BFS、DFS）
- 学会实现拓扑排序
- 掌握连通性检测算法

## 图的基本概念

### 术语定义
- **顶点(Vertex)**: 图中的节点
- **边(Edge)**: 连接顶点的线
- **度(Degree)**: 顶点连接的边数
- **路径(Path)**: 顶点间的连接序列
- **环(Cycle)**: 起点和终点相同的路径
- **连通图**: 任意两顶点间都有路径
- **权重图**: 边有权重的图

### 图的分类
- **有向图**: 边有方向
- **无向图**: 边无方向
- **加权图**: 边有权重
- **非加权图**: 边无权重

## Python实现

### 1. 图的表示方法
```python
from collections import defaultdict, deque
from typing import List, Dict, Set, Tuple, Optional
import heapq

class Graph:
    """
    图的基本实现
    支持有向和无向图
    """

    def __init__(self, directed=False):
        self.directed = directed
        self.adjacency_list = defaultdict(list)
        self.vertices = set()

    def add_vertex(self, vertex):
        """添加顶点"""
        self.vertices.add(vertex)

    def add_edge(self, u, v, weight=None):
        """添加边"""
        self.add_vertex(u)
        self.add_vertex(v)

        if weight is not None:
            self.adjacency_list[u].append((v, weight))
        else:
            self.adjacency_list[u].append(v)

        if not self.directed:
            if weight is not None:
                self.adjacency_list[v].append((u, weight))
            else:
                self.adjacency_list[v].append(u)

    def remove_vertex(self, vertex):
        """删除顶点"""
        if vertex in self.vertices:
            self.vertices.remove(vertex)
            # 删除所有与该顶点相关的边
            del self.adjacency_list[vertex]
            for neighbors in self.adjacency_list.values():
                # 处理带权重和不带权重的情况
                new_neighbors = []
                for neighbor in neighbors:
                    if isinstance(neighbor, tuple):
                        if neighbor[0] != vertex:
                            new_neighbors.append(neighbor)
                    else:
                        if neighbor != vertex:
                            new_neighbors.append(neighbor)
                neighbors[:] = new_neighbors

    def remove_edge(self, u, v):
        """删除边"""
        if u in self.adjacency_list:
            new_neighbors = []
            for neighbor in self.adjacency_list[u]:
                if isinstance(neighbor, tuple):
                    if neighbor[0] != v:
                        new_neighbors.append(neighbor)
                else:
                    if neighbor != v:
                        new_neighbors.append(neighbor)
            self.adjacency_list[u] = new_neighbors

        if not self.directed and v in self.adjacency_list:
            new_neighbors = []
            for neighbor in self.adjacency_list[v]:
                if isinstance(neighbor, tuple):
                    if neighbor[0] != u:
                        new_neighbors.append(neighbor)
                else:
                    if neighbor != u:
                        new_neighbors.append(neighbor)
            self.adjacency_list[v] = new_neighbors

    def get_vertices(self) -> Set:
        """获取所有顶点"""
        return self.vertices

    def get_edges(self) -> List[Tuple]:
        """获取所有边"""
        edges = []
        for u, neighbors in self.adjacency_list.items():
            for neighbor in neighbors:
                if isinstance(neighbor, tuple):
                    v, weight = neighbor
                    edges.append((u, v, weight))
                else:
                    edges.append((u, neighbor))
        return edges

    def get_neighbors(self, vertex) -> List:
        """获取顶点的邻居"""
        return self.adjacency_list.get(vertex, [])

    def get_degree(self, vertex) -> int:
        """获取顶点的度数"""
        return len(self.get_neighbors(vertex))

    def __str__(self):
        result = []
        for vertex in self.vertices:
            neighbors = self.get_neighbors(vertex)
            result.append(f"{vertex}: {neighbors}")
        return "\n".join(result)
```

### 2. 邻接矩阵实现
```python
class AdjacencyMatrixGraph:
    """
    邻接矩阵实现的图
    适合稠密图
    """

    def __init__(self, vertices=None, directed=False):
        self.directed = directed
        if vertices:
            self.vertices = list(vertices)
            self.vertex_index = {v: i for i, v in enumerate(vertices)}
            self.matrix = [[0] * len(vertices) for _ in range(len(vertices))]
        else:
            self.vertices = []
            self.vertex_index = {}
            self.matrix = []

    def add_vertex(self, vertex):
        """添加顶点"""
        if vertex not in self.vertex_index:
            self.vertices.append(vertex)
            self.vertex_index[vertex] = len(self.vertices) - 1
            # 扩展矩阵
            for row in self.matrix:
                row.append(0)
            self.matrix.append([0] * len(self.vertices))

    def add_edge(self, u, v, weight=1):
        """添加边"""
        if u not in self.vertex_index:
            self.add_vertex(u)
        if v not in self.vertex_index:
            self.add_vertex(v)

        u_idx = self.vertex_index[u]
        v_idx = self.vertex_index[v]

        self.matrix[u_idx][v_idx] = weight
        if not self.directed:
            self.matrix[v_idx][u_idx] = weight

    def remove_edge(self, u, v):
        """删除边"""
        if u in self.vertex_index and v in self.vertex_index:
            u_idx = self.vertex_index[u]
            v_idx = self.vertex_index[v]
            self.matrix[u_idx][v_idx] = 0
            if not self.directed:
                self.matrix[v_idx][u_idx] = 0

    def get_weight(self, u, v):
        """获取边的权重"""
        if u in self.vertex_index and v in self.vertex_index:
            u_idx = self.vertex_index[u]
            v_idx = self.vertex_index[v]
            return self.matrix[u_idx][v_idx]
        return 0

    def get_vertices(self) -> List:
        """获取所有顶点"""
        return self.vertices

    def get_edges(self) -> List[Tuple]:
        """获取所有边"""
        edges = []
        n = len(self.vertices)
        for i in range(n):
            for j in range(n if self.directed else i + 1, n):
                if self.matrix[i][j] != 0:
                    edges.append((self.vertices[i], self.vertices[j], self.matrix[i][j]))
        return edges
```

### 3. 广度优先搜索 (BFS)
```python
def bfs(graph: Graph, start: str, end: str = None) -> Dict:
    """
    广度优先搜索
    返回最短路径和距离
    """
    if start not in graph.vertices:
        return {"path": [], "distance": -1}

    visited = set()
    queue = deque([start])
    distance = {start: 0}
    parent = {start: None}

    while queue:
        current = queue.popleft()

        if current == end:
            break

        if current not in visited:
            visited.add(current)

            for neighbor in graph.get_neighbors(current):
                neighbor_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor

                if neighbor_vertex not in visited and neighbor_vertex not in distance:
                    distance[neighbor_vertex] = distance[current] + 1
                    parent[neighbor_vertex] = current
                    queue.append(neighbor_vertex)

    # 重构路径
    path = []
    if end is not None and end in distance:
        current = end
        while current is not None:
            path.append(current)
            current = parent[current]
        path.reverse()

    return {
        "path": path,
        "distance": distance.get(end, -1) if end else -1,
        "visited": visited
    }

def bfs_connected_components(graph: Graph) -> List[Set]:
    """找出所有连通分量"""
    visited = set()
    components = []

    for vertex in graph.vertices:
        if vertex not in visited:
            # 对未访问的顶点进行BFS
            component = set()
            queue = deque([vertex])

            while queue:
                current = queue.popleft()
                if current not in visited:
                    visited.add(current)
                    component.add(current)

                    for neighbor in graph.get_neighbors(current):
                        neighbor_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
                        if neighbor_vertex not in visited:
                            queue.append(neighbor_vertex)

            components.append(component)

    return components
```

### 4. 深度优先搜索 (DFS)
```python
def dfs(graph: Graph, start: str, end: str = None) -> Dict:
    """
    深度优先搜索
    使用递归实现
    """
    visited = set()
    path = []
    found = False

    def dfs_recursive(current, target=None):
        nonlocal found
        visited.add(current)
        path.append(current)

        if target and current == target:
            found = True
            return

        for neighbor in graph.get_neighbors(current):
            neighbor_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor

            if neighbor_vertex not in visited and not found:
                dfs_recursive(neighbor_vertex, target)
                if found:
                    return

        if not found:
            path.pop()

    dfs_recursive(start, end)

    return {
        "path": path if found or not end else [],
        "visited": visited,
        "found": found
    }

def dfs_iterative(graph: Graph, start: str, end: str = None) -> Dict:
    """
    深度优先搜索
    使用栈实现
    """
    visited = set()
    stack = [(start, [start])]  # (当前顶点, 路径)
    found = False
    final_path = []

    while stack:
        current, path = stack.pop()

        if current == end:
            found = True
            final_path = path
            break

        if current not in visited:
            visited.add(current)

            # 按字母顺序压入栈，保证顺序一致性
            neighbors = sorted(graph.get_neighbors(current), reverse=True)
            for neighbor in neighbors:
                neighbor_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
                if neighbor_vertex not in visited:
                    new_path = path + [neighbor_vertex]
                    stack.append((neighbor_vertex, new_path))

    return {
        "path": final_path,
        "visited": visited,
        "found": found
    }

def dfs_detect_cycles(graph: Graph) -> bool:
    """检测图中是否有环"""
    visited = set()
    rec_stack = set()

    def dfs_cycle(vertex):
        visited.add(vertex)
        rec_stack.add(vertex)

        for neighbor in graph.get_neighbors(vertex):
            neighbor_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor

            if neighbor_vertex not in visited:
                if dfs_cycle(neighbor_vertex):
                    return True
            elif neighbor_vertex in rec_stack:
                return True

        rec_stack.remove(vertex)
        return False

    for vertex in graph.vertices:
        if vertex not in visited:
            if dfs_cycle(vertex):
                return True

    return False
```

### 5. 拓扑排序
```python
def topological_sort(graph: Graph) -> List[str]:
    """
    拓扑排序 (Kahn算法)
    适用于有向无环图(DAG)
    """
    if not graph.directed:
        raise ValueError("拓扑排序仅适用于有向图")

    # 计算入度
    in_degree = {vertex: 0 for vertex in graph.vertices}
    for u in graph.vertices:
        for neighbor in graph.get_neighbors(u):
            neighbor_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
            in_degree[neighbor_vertex] += 1

    # 将入度为0的顶点加入队列
    queue = deque([vertex for vertex in graph.vertices if in_degree[vertex] == 0])
    result = []

    while queue:
        current = queue.popleft()
        result.append(current)

        # 减少邻居的入度
        for neighbor in graph.get_neighbors(current):
            neighbor_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
            in_degree[neighbor_vertex] -= 1
            if in_degree[neighbor_vertex] == 0:
                queue.append(neighbor_vertex)

    # 检查是否有环
    if len(result) != len(graph.vertices):
        raise ValueError("图中存在环，无法进行拓扑排序")

    return result

def topological_sort_dfs(graph: Graph) -> List[str]:
    """
    拓扑排序 (DFS实现)
    """
    if not graph.directed:
        raise ValueError("拓扑排序仅适用于有向图")

    visited = set()
    temp_visited = set()
    result = []

    def dfs_topological(vertex):
        visited.add(vertex)
        temp_visited.add(vertex)

        for neighbor in graph.get_neighbors(vertex):
            neighbor_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor

            if neighbor_vertex not in visited:
                dfs_topological(neighbor_vertex)
            elif neighbor_vertex in temp_visited:
                raise ValueError("图中存在环，无法进行拓扑排序")

        temp_visited.remove(vertex)
        result.append(vertex)

    for vertex in graph.vertices:
        if vertex not in visited:
            dfs_topological(vertex)

    return result[::-1]  # 反转得到拓扑顺序
```

### 6. 强连通分量
```python
def kosaraju_algorithm(graph: Graph) -> List[Set[str]]:
    """
    Kosaraju算法找强连通分量
    适用于有向图
    """
    if not graph.directed:
        raise ValueError("强连通分量仅适用于有向图")

    # 第一次DFS，按完成时间排序
    visited = set()
    stack = []

    def dfs_first(vertex):
        visited.add(vertex)
        for neighbor in graph.get_neighbors(vertex):
            neighbor_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
            if neighbor_vertex not in visited:
                dfs_first(neighbor_vertex)
        stack.append(vertex)

    for vertex in graph.vertices:
        if vertex not in visited:
            dfs_first(vertex)

    # 构建转置图
    reversed_graph = Graph(directed=True)
    for u, v, _ in graph.get_edges():
        reversed_graph.add_edge(v, u)

    # 第二次DFS，找强连通分量
    visited.clear()
    scc = []

    def dfs_second(vertex, component):
        visited.add(vertex)
        component.add(vertex)
        for neighbor in reversed_graph.get_neighbors(vertex):
            neighbor_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
            if neighbor_vertex not in visited:
                dfs_second(neighbor_vertex, component)

    while stack:
        vertex = stack.pop()
        if vertex not in visited:
            component = set()
            dfs_second(vertex, component)
            scc.append(component)

    return scc
```

## 性能测试与可视化

```python
import matplotlib.pyplot as plt
import networkx as nx
import time
import random

def create_sample_graph():
    """创建示例图"""
    g = Graph(directed=False)

    # 添加顶点
    vertices = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
    for v in vertices:
        g.add_vertex(v)

    # 添加边
    edges = [
        ('A', 'B'), ('A', 'C'), ('B', 'D'), ('B', 'E'),
        ('C', 'F'), ('D', 'G'), ('E', 'F'), ('F', 'H'),
        ('G', 'H')
    ]

    for u, v in edges:
        g.add_edge(u, v)

    return g

def create_directed_graph():
    """创建有向图"""
    g = Graph(directed=True)

    vertices = ['A', 'B', 'C', 'D', 'E', 'F']
    for v in vertices:
        g.add_vertex(v)

    edges = [
        ('A', 'B'), ('A', 'C'), ('B', 'D'), ('C', 'D'),
        ('D', 'E'), ('E', 'F'), ('C', 'F')
    ]

    for u, v in edges:
        g.add_edge(u, v)

    return g

def visualize_graph(graph: Graph):
    """可视化图"""
    G = nx.Graph()

    # 添加顶点
    for vertex in graph.vertices:
        G.add_node(vertex)

    # 添加边
    for edge in graph.get_edges():
        if len(edge) == 2:
            u, v = edge
            G.add_edge(u, v)
        else:
            u, v, weight = edge
            G.add_edge(u, v, weight=weight)

    # 绘制图
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=500, font_size=12, font_weight='bold')

    # 如果有权重，显示权重
    if graph.get_edges() and len(graph.get_edges()[0]) == 3:
        edge_labels = {(u, v): weight for u, v, weight in graph.get_edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels)

    plt.title("Graph Visualization")
    plt.axis('off')
    plt.show()

def benchmark_graph_algorithms():
    """图算法性能测试"""
    sizes = [100, 500, 1000, 2000]

    for size in sizes:
        print(f"\n=== 测试大小: {size} ===")

        # 创建随机图
        g = Graph()
        vertices = [f"V{i}" for i in range(size)]
        for v in vertices:
            g.add_vertex(v)

        # 随机添加边
        edge_count = size * 2
        for _ in range(edge_count):
            u = random.choice(vertices)
            v = random.choice(vertices)
            if u != v:
                g.add_edge(u, v)

        # 测试BFS
        start_time = time.time()
        bfs(g, vertices[0])
        bfs_time = time.time() - start_time

        # 测试DFS
        start_time = time.time()
        dfs(g, vertices[0])
        dfs_time = time.time() - start_time

        # 测试连通分量
        start_time = time.time()
        components = bfs_connected_components(g)
        components_time = time.time() - start_time

        print(f"BFS: {bfs_time:.6f}s")
        print(f"DFS: {dfs_time:.6f}s")
        print(f"连通分量: {components_time:.6f}s")
        print(f"连通分量数量: {len(components)}")
```

## 应用场景

### 1. 社交网络
- **朋友推荐**: 找出共同朋友
- **社区发现**: 识别社交群体
- **影响力分析**: 找出关键人物

```python
class SocialNetwork:
    """社交网络分析"""
    def __init__(self):
        self.graph = Graph()

    def add_user(self, user_id):
        """添加用户"""
        self.graph.add_vertex(user_id)

    def add_friendship(self, user1, user2):
        """添加好友关系"""
        self.graph.add_edge(user1, user2)

    def find_mutual_friends(self, user1, user2):
        """找出共同好友"""
        friends1 = set(self.graph.get_neighbors(user1))
        friends2 = set(self.graph.get_neighbors(user2))
        return friends1 & friends2

    def suggest_friends(self, user_id):
        """推荐好友"""
        friends = set(self.graph.get_neighbors(user_id))
        suggestions = {}

        for friend in friends:
            for friend_of_friend in self.graph.get_neighbors(friend):
                if friend_of_friend != user_id and friend_of_friend not in friends:
                    suggestions[friend_of_friend] = suggestions.get(friend_of_friend, 0) + 1

        return sorted(suggestions.items(), key=lambda x: x[1], reverse=True)
```

### 2. 路径规划
- **最短路径**: GPS导航
- **路线优化**: 物流配送
- **网络路由**: 数据包传输

```python
class PathFinder:
    """路径查找器"""
    def __init__(self):
        self.graph = Graph(directed=True)

    def add_road(self, from_loc, to_loc, distance):
        """添加道路"""
        self.graph.add_edge(from_loc, to_loc, distance)

    def find_shortest_path(self, start, end):
        """查找最短路径"""
        return bfs(self.graph, start, end)

    def find_all_routes(self, start, end):
        """查找所有路线"""
        return dfs(self.graph, start, end)
```

### 3. 任务调度
- **依赖关系**: 项目管理
- **执行顺序**: 编译系统
- **资源分配**: 操作系统

```python
class TaskScheduler:
    """任务调度器"""
    def __init__(self):
        self.graph = Graph(directed=True)

    def add_task(self, task_id):
        """添加任务"""
        self.graph.add_vertex(task_id)

    def add_dependency(self, task, prerequisite):
        """添加依赖关系"""
        self.graph.add_edge(prerequisite, task)

    def get_execution_order(self):
        """获取执行顺序"""
        return topological_sort(self.graph)

    def check_cycles(self):
        """检查是否有循环依赖"""
        return dfs_detect_cycles(self.graph)
```

### 4. 网络分析
- **网页排名**: PageRank算法
- **网络拓扑**: 网络结构分析
- **流量分析**: 网络流量预测

```python
class NetworkAnalyzer:
    """网络分析器"""
    def __init__(self):
        self.graph = Graph(directed=True)

    def add_node(self, node_id):
        """添加节点"""
        self.graph.add_vertex(node_id)

    def add_connection(self, from_node, to_node):
        """添加连接"""
        self.graph.add_edge(from_node, to_node)

    def find_critical_nodes(self):
        """找出关键节点"""
        nodes = []
        for node in self.graph.vertices:
            # 临时删除节点
            temp_graph = Graph(directed=True)
            for v in self.graph.vertices:
                if v != node:
                    temp_graph.add_vertex(v)

            for u, v in self.graph.get_edges():
                if u != node and v[0] != node:
                    temp_graph.add_edge(u, v[0])

            # 检查连通性
            components = bfs_connected_components(temp_graph)
            if len(components) > 1:
                nodes.append(node)

        return nodes
```

## 高级技巧与优化

### 1. 双向BFS
```python
def bidirectional_bfs(graph: Graph, start: str, end: str) -> Dict:
    """双向BFS，用于大规模图的最短路径查找"""
    if start == end:
        return {"path": [start], "distance": 0}

    # 正向搜索
    forward_visited = {start: [start]}
    forward_queue = deque([start])

    # 反向搜索
    backward_visited = {end: [end]}
    backward_queue = deque([end])

    while forward_queue and backward_queue:
        # 正向搜索一步
        forward_level_size = len(forward_queue)
        for _ in range(forward_level_size):
            current = forward_queue.popleft()

            if current in backward_visited:
                # 找到路径
                forward_path = forward_visited[current]
                backward_path = backward_visited[current][::-1]
                return {
                    "path": forward_path + backward_path[1:],
                    "distance": len(forward_path) + len(backward_path) - 1
                }

            for neighbor in graph.get_neighbors(current):
                neighbor_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
                if neighbor_vertex not in forward_visited:
                    forward_visited[neighbor_vertex] = forward_visited[current] + [neighbor_vertex]
                    forward_queue.append(neighbor_vertex)

        # 反向搜索一步
        backward_level_size = len(backward_queue)
        for _ in range(backward_level_size):
            current = backward_queue.popleft()

            if current in forward_visited:
                # 找到路径
                forward_path = forward_visited[current]
                backward_path = backward_visited[current][::-1]
                return {
                    "path": forward_path + backward_path[1:],
                    "distance": len(forward_path) + len(backward_path) - 1
                }

            for neighbor in graph.get_neighbors(current):
                neighbor_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
                if neighbor_vertex not in backward_visited:
                    backward_visited[neighbor_vertex] = backward_visited[current] + [neighbor_vertex]
                    backward_queue.append(neighbor_vertex)

    return {"path": [], "distance": -1}
```

### 2. 并行BFS
```python
import concurrent.futures

def parallel_bfs(graph: Graph, start: str) -> Dict:
    """并行BFS"""
    visited = set([start])
    queue = deque([start])
    result = {start: 0}

    def process_neighbors(neighbors):
        new_nodes = []
        for neighbor in neighbors:
            neighbor_vertex = neighbor[0] if isinstance(neighbor, tuple) else neighbor
            if neighbor_vertex not in visited:
                new_nodes.append(neighbor_vertex)
        return new_nodes

    while queue:
        level_size = len(queue)
        current_level = []

        for _ in range(level_size):
            current = queue.popleft()
            current_level.append(current)

        # 并行处理当前层的所有节点
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for node in current_level:
                neighbors = graph.get_neighbors(node)
                future = executor.submit(process_neighbors, neighbors)
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                new_nodes = future.result()
                for new_node in new_nodes:
                    visited.add(new_node)
                    result[new_node] = result[current_level[0]] + 1
                    queue.append(new_node)

    return result
```

### 3. 缓存优化的图遍历
```python
class CacheOptimizedGraph:
    """缓存优化的图结构"""
    def __init__(self):
        self.vertices = []
        self.adjacency_list = []
        self.vertex_index = {}

    def add_vertex(self, vertex):
        """添加顶点"""
        if vertex not in self.vertex_index:
            index = len(self.vertices)
            self.vertices.append(vertex)
            self.vertex_index[vertex] = index
            self.adjacency_list.append([])

    def add_edge(self, u, v, weight=None):
        """添加边"""
        if u not in self.vertex_index:
            self.add_vertex(u)
        if v not in self.vertex_index:
            self.add_vertex(v)

        u_idx = self.vertex_index[u]
        v_idx = self.vertex_index[v]

        if weight is not None:
            self.adjacency_list[u_idx].append((v_idx, weight))
        else:
            self.adjacency_list[u_idx].append(v_idx)

    def bfs_cache_optimized(self, start):
        """缓存优化的BFS"""
        if start not in self.vertex_index:
            return []

        start_idx = self.vertex_index[start]
        visited = [False] * len(self.vertices)
        queue = deque([start_idx])
        visited[start_idx] = True
        result = []

        while queue:
            current_idx = queue.popleft()
            result.append(self.vertices[current_idx])

            # 批量处理邻居，提高缓存命中率
            neighbors = self.adjacency_list[current_idx]
            for neighbor in neighbors:
                neighbor_idx = neighbor[0] if isinstance(neighbor, tuple) else neighbor
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    queue.append(neighbor_idx)

        return result
```

## 练习

1. 实现一个支持撤销/重做的图编辑器
2. 创建一个检测图中所有环的算法
3. 实现一个图的欧拉路径检测算法
4. 设计一个支持增量更新的最短路径算法
5. 实现一个图的同构检测算法
6. 创建一个支持多种查询的图数据库
7. 实现一个动态图的连通性检测算法
8. 设计一个支持分布式处理的图计算框架