# L07 - 最短路径算法

## 学习目标
- 掌握单源最短路径算法（Dijkstra、Bellman-Ford）
- 理解多源最短路径算法（Floyd-Warshall）
- 学会处理负权边和负权环
- 能够选择合适的最短路径算法解决实际问题

## 最短路径问题分类

### 单源最短路径
- **Dijkstra算法**: 适用于非负权图
- **Bellman-Ford算法**: 适用于有负权边的图
- **SPFA算法**: Bellman-Ford的优化版本

### 多源最短路径
- **Floyd-Warshall算法**: 动态规划求解所有点对最短路径
- **重复调用Dijkstra**: 适用于非负权图

### 特殊图的最短路径
- **无权图**: BFS即可
- **DAG**: 拓扑排序 + 动态规划

## Python实现

### 1. Dijkstra算法
```python
import heapq
from typing import Dict, List, Tuple, Optional
import math

class WeightedGraph:
    """加权图"""
    def __init__(self, directed=False):
        self.directed = directed
        self.adjacency_list = {}
        self.vertices = set()

    def add_vertex(self, vertex):
        """添加顶点"""
        self.vertices.add(vertex)
        if vertex not in self.adjacency_list:
            self.adjacency_list[vertex] = []

    def add_edge(self, u, v, weight):
        """添加边"""
        self.add_vertex(u)
        self.add_vertex(v)
        self.adjacency_list[u].append((v, weight))
        if not self.directed:
            self.adjacency_list[v].append((u, weight))

    def get_weight(self, u, v):
        """获取边权重"""
        for neighbor, weight in self.adjacency_list.get(u, []):
            if neighbor == v:
                return weight
        return float('inf')

    def get_neighbors(self, vertex):
        """获取邻居"""
        return self.adjacency_list.get(vertex, [])

    def __str__(self):
        result = []
        for vertex in self.vertices:
            neighbors = self.get_neighbors(vertex)
            result.append(f"{vertex}: {neighbors}")
        return "\n".join(result)

def dijkstra(graph: WeightedGraph, start: str) -> Dict[str, Tuple[float, List[str]]]:
    """
    Dijkstra算法求单源最短路径
    时间复杂度: O((V + E) log V)
    空间复杂度: O(V)
    适用于非负权图
    """
    # 初始化距离和前驱节点
    distances = {vertex: float('inf') for vertex in graph.vertices}
    distances[start] = 0
    predecessors = {vertex: None for vertex in graph.vertices}
    visited = set()

    # 优先队列 (距离, 顶点)
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_vertex = heapq.heappop(priority_queue)

        # 如果已经找到更短路径，跳过
        if current_vertex in visited:
            continue

        visited.add(current_vertex)

        # 更新邻居的距离
        for neighbor, weight in graph.get_neighbors(current_vertex):
            if neighbor in visited:
                continue

            new_distance = current_distance + weight

            if new_distance < distances[neighbor]:
                distances[neighbor] = new_distance
                predecessors[neighbor] = current_vertex
                heapq.heappush(priority_queue, (new_distance, neighbor))

    # 重构路径
    paths = {}
    for vertex in graph.vertices:
        if distances[vertex] == float('inf'):
            paths[vertex] = []
        else:
            path = []
            current = vertex
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()
            paths[vertex] = path

    return {vertex: (distances[vertex], paths[vertex]) for vertex in graph.vertices}

def dijkstra_with_heap_optimization(graph: WeightedGraph, start: str) -> Dict[str, float]:
    """
    优化的Dijkstra算法
    使用斐波那契堆的近似实现
    """
    distances = {vertex: float('inf') for vertex in graph.vertices}
    distances[start] = 0
    visited = set()

    # 使用堆优化
    heap = [(0, start)]

    while heap:
        current_distance, current_vertex = heapq.heappop(heap)

        if current_vertex in visited:
            continue

        visited.add(current_vertex)

        for neighbor, weight in graph.get_neighbors(current_vertex):
            if neighbor not in visited:
                new_distance = current_distance + weight
                if new_distance < distances[neighbor]:
                    distances[neighbor] = new_distance
                    heapq.heappush(heap, (new_distance, neighbor))

    return distances

def a_star_search(graph: WeightedGraph, start: str, goal: str, heuristic: Dict[str, float]) -> Tuple[float, List[str]]:
    """
    A*搜索算法
    结合Dijkstra和启发式搜索
    """
    # 优先队列 (f_score, 顶点)
    open_set = [(0, start)]
    came_from = {}

    # g_score: 从起点到当前节点的实际代价
    g_score = {vertex: float('inf') for vertex in graph.vertices}
    g_score[start] = 0

    # f_score: g_score + 启发式函数
    f_score = {vertex: float('inf') for vertex in graph.vertices}
    f_score[start] = heuristic.get(start, 0)

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == goal:
            # 重构路径
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return g_score[goal], path

        for neighbor, weight in graph.get_neighbors(current):
            tentative_g_score = g_score[current] + weight

            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic.get(neighbor, 0)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return float('inf'), []
```

### 2. Bellman-Ford算法
```python
def bellman_ford(graph: WeightedGraph, start: str) -> Dict[str, Tuple[float, List[str]]]:
    """
    Bellman-Ford算法求单源最短路径
    时间复杂度: O(V * E)
    空间复杂度: O(V)
    可以处理负权边，检测负权环
    """
    distances = {vertex: float('inf') for vertex in graph.vertices}
    distances[start] = 0
    predecessors = {vertex: None for vertex in graph.vertices}

    # 松弛操作，执行V-1次
    for _ in range(len(graph.vertices) - 1):
        for u in graph.vertices:
            for v, weight in graph.get_neighbors(u):
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u

    # 检测负权环
    for u in graph.vertices:
        for v, weight in graph.get_neighbors(u):
            if distances[u] + weight < distances[v]:
                raise ValueError("图中存在负权环")

    # 重构路径
    paths = {}
    for vertex in graph.vertices:
        if distances[vertex] == float('inf'):
            paths[vertex] = []
        else:
            path = []
            current = vertex
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()
            paths[vertex] = path

    return {vertex: (distances[vertex], paths[vertex]) for vertex in graph.vertices}

def spfa(graph: WeightedGraph, start: str) -> Dict[str, Tuple[float, List[str]]]:
    """
    SPFA (Shortest Path Faster Algorithm)
    Bellman-Ford的优化版本
    """
    distances = {vertex: float('inf') for vertex in graph.vertices}
    distances[start] = 0
    predecessors = {vertex: None for vertex in graph.vertices}
    in_queue = {vertex: False for vertex in graph.vertices}
    queue = [start]
    in_queue[start] = True

    # 记录每个顶点的入队次数，用于检测负权环
    count = {vertex: 0 for vertex in graph.vertices}
    count[start] = 1

    while queue:
        current = queue.pop(0)
        in_queue[current] = False

        for neighbor, weight in graph.get_neighbors(current):
            if distances[current] + weight < distances[neighbor]:
                distances[neighbor] = distances[current] + weight
                predecessors[neighbor] = current

                if not in_queue[neighbor]:
                    count[neighbor] += 1
                    if count[neighbor] > len(graph.vertices):
                        raise ValueError("图中存在负权环")

                    queue.append(neighbor)
                    in_queue[neighbor] = True

    # 重构路径
    paths = {}
    for vertex in graph.vertices:
        if distances[vertex] == float('inf'):
            paths[vertex] = []
        else:
            path = []
            current = vertex
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()
            paths[vertex] = path

    return {vertex: (distances[vertex], paths[vertex]) for vertex in graph.vertices}
```

### 3. Floyd-Warshall算法
```python
def floyd_warshall(graph: WeightedGraph) -> Dict[str, Dict[str, Tuple[float, List[str]]]]:
    """
    Floyd-Warshall算法求所有点对最短路径
    时间复杂度: O(V³)
    空间复杂度: O(V²)
    """
    vertices = list(graph.vertices)
    n = len(vertices)
    vertex_index = {v: i for i, v in enumerate(vertices)}

    # 初始化距离矩阵和路径矩阵
    dist = [[float('inf')] * n for _ in range(n)]
    next_vertex = [[None] * n for _ in range(n)]

    # 对角线设为0
    for i in range(n):
        dist[i][i] = 0

    # 填充初始距离
    for u in graph.vertices:
        for v, weight in graph.get_neighbors(u):
            i = vertex_index[u]
            j = vertex_index[v]
            dist[i][j] = weight
            next_vertex[i][j] = j

    # 动态规划
    for k in range(n):
        for i in range(n):
            for j in range(n):
                if dist[i][j] > dist[i][k] + dist[k][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]
                    next_vertex[i][j] = next_vertex[i][k]

    # 检测负权环
    for i in range(n):
        if dist[i][i] < 0:
            raise ValueError("图中存在负权环")

    # 重构路径
    result = {}
    for u in vertices:
        result[u] = {}
        for v in vertices:
            i = vertex_index[u]
            j = vertex_index[v]

            if dist[i][j] == float('inf'):
                result[u][v] = (float('inf'), [])
            else:
                path = []
                if next_vertex[i][j] is not None:
                    current = i
                    while current is not None and current != j:
                        path.append(vertices[current])
                        current = next_vertex[current][j] if next_vertex[current][j] is not None else j
                    path.append(vertices[j])
                result[u][v] = (dist[i][j], path)

    return result

def johnson_algorithm(graph: WeightedGraph) -> Dict[str, Dict[str, float]]:
    """
    Johnson算法求所有点对最短路径
    结合Bellman-Ford和Dijkstra，适用于稀疏图
    时间复杂度: O(V² log V + V * E)
    """
    # 添加新的顶点s
    new_vertex = "s"
    temp_graph = WeightedGraph(directed=True)

    # 复制原图
    for vertex in graph.vertices:
        temp_graph.add_vertex(vertex)
    for u in graph.vertices:
        for v, weight in graph.get_neighbors(u):
            temp_graph.add_edge(u, v, weight)

    # 添加新顶点s和到所有顶点的边
    temp_graph.add_vertex(new_vertex)
    for vertex in graph.vertices:
        temp_graph.add_edge(new_vertex, vertex, 0)

    # 使用Bellman-Ford计算h函数
    try:
        bellman_result = bellman_ford(temp_graph, new_vertex)
        h = {vertex: bellman_result[vertex][0] for vertex in graph.vertices}
    except ValueError:
        raise ValueError("图中存在负权环")

    # 重新计算边权重
    reweighted_graph = WeightedGraph(directed=True)
    for vertex in graph.vertices:
        reweighted_graph.add_vertex(vertex)

    for u in graph.vertices:
        for v, weight in graph.get_neighbors(u):
            new_weight = weight + h[u] - h[v]
            reweighted_graph.add_edge(u, v, new_weight)

    # 对每个顶点运行Dijkstra
    all_distances = {}
    for vertex in graph.vertices:
        distances = dijkstra_with_heap_optimization(reweighted_graph, vertex)
        # 恢复原始权重
        for v in distances:
            if distances[v] != float('inf'):
                distances[v] = distances[v] - h[vertex] + h[v]
        all_distances[vertex] = distances

    return all_distances
```

### 4. DAG最短路径
```python
def dag_shortest_path(graph: WeightedGraph, start: str) -> Dict[str, Tuple[float, List[str]]]:
    """
    DAG上的最短路径算法
    使用拓扑排序 + 动态规划
    时间复杂度: O(V + E)
    """
    # 拓扑排序
    def topological_sort():
        in_degree = {vertex: 0 for vertex in graph.vertices}
        for u in graph.vertices:
            for v, _ in graph.get_neighbors(u):
                in_degree[v] += 1

        queue = [vertex for vertex in graph.vertices if in_degree[vertex] == 0]
        result = []

        while queue:
            current = queue.pop(0)
            result.append(current)

            for neighbor, _ in graph.get_neighbors(current):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if len(result) != len(graph.vertices):
            raise ValueError("图中存在环，不是DAG")

        return result

    topo_order = topological_sort()

    # 初始化距离和前驱
    distances = {vertex: float('inf') for vertex in graph.vertices}
    distances[start] = 0
    predecessors = {vertex: None for vertex in graph.vertices}

    # 按拓扑顺序处理顶点
    for u in topo_order:
        if distances[u] != float('inf'):
            for v, weight in graph.get_neighbors(u):
                if distances[u] + weight < distances[v]:
                    distances[v] = distances[u] + weight
                    predecessors[v] = u

    # 重构路径
    paths = {}
    for vertex in graph.vertices:
        if distances[vertex] == float('inf'):
            paths[vertex] = []
        else:
            path = []
            current = vertex
            while current is not None:
                path.append(current)
                current = predecessors[current]
            path.reverse()
            paths[vertex] = path

    return {vertex: (distances[vertex], paths[vertex]) for vertex in graph.vertices}
```

## 性能测试与比较

```python
import time
import random
import matplotlib.pyplot as plt

def create_random_weighted_graph(num_vertices, num_edges, weight_range=(1, 100), negative_weights=False):
    """创建随机加权图"""
    graph = WeightedGraph(directed=True)

    # 添加顶点
    vertices = [f"V{i}" for i in range(num_vertices)]
    for vertex in vertices:
        graph.add_vertex(vertex)

    # 添加边
    edges = set()
    while len(edges) < num_edges:
        u = random.choice(vertices)
        v = random.choice(vertices)
        if u != v and (u, v) not in edges:
            if negative_weights:
                weight = random.randint(-weight_range[1], weight_range[1])
            else:
                weight = random.randint(weight_range[0], weight_range[1])
            graph.add_edge(u, v, weight)
            edges.add((u, v))

    return graph

def benchmark_shortest_path_algorithms():
    """最短路径算法性能测试"""
    sizes = [(50, 200), (100, 500), (200, 1000), (500, 2000)]

    for num_vertices, num_edges in sizes:
        print(f"\n=== 图大小: {num_vertices} 顶点, {num_edges} 边 ===")

        # 创建非负权图
        graph = create_random_weighted_graph(num_vertices, num_edges)
        start_vertex = list(graph.vertices)[0]

        # 测试Dijkstra
        start_time = time.time()
        try:
            dijkstra_result = dijkstra(graph, start_vertex)
            dijkstra_time = time.time() - start_time
            print(f"Dijkstra: {dijkstra_time:.6f}s")
        except Exception as e:
            print(f"Dijkstra: 失败 - {e}")

        # 测试SPFA
        start_time = time.time()
        try:
            spfa_result = spfa(graph, start_vertex)
            spfa_time = time.time() - start_time
            print(f"SPFA: {spfa_time:.6f}s")
        except Exception as e:
            print(f"SPFA: 失败 - {e}")

        # 测试Floyd-Warshall (仅在小图上)
        if num_vertices <= 100:
            start_time = time.time()
            try:
                floyd_result = floyd_warshall(graph)
                floyd_time = time.time() - start_time
                print(f"Floyd-Warshall: {floyd_time:.6f}s")
            except Exception as e:
                print(f"Floyd-Warshall: 失败 - {e}")

def visualize_shortest_path(graph: WeightedGraph, start: str, algorithm='dijkstra'):
    """可视化最短路径"""
    import networkx as nx
    import matplotlib.pyplot as plt

    # 创建NetworkX图
    G = nx.DiGraph()

    # 添加顶点
    for vertex in graph.vertices:
        G.add_node(vertex)

    # 添加边
    for u in graph.vertices:
        for v, weight in graph.get_neighbors(u):
            G.add_edge(u, v, weight=weight)

    # 计算最短路径
    if algorithm == 'dijkstra':
        result = dijkstra(graph, start)
    elif algorithm == 'bellman_ford':
        result = bellman_ford(graph, start)
    else:
        raise ValueError("不支持的算法")

    # 绘制图
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))

    # 绘制所有边
    nx.draw_networkx_edges(G, pos, edge_color='gray', width=1)

    # 绘制最短路径树
    for vertex in graph.vertices:
        distance, path = result[vertex]
        if path and len(path) > 1:
            path_edges = [(path[i], path[i+1]) for i in range(len(path)-1)]
            nx.draw_networkx_edges(G, pos, edgelist=path_edges,
                                 edge_color='red', width=2, alpha=0.7)

    # 绘制顶点和标签
    nx.draw_networkx_nodes(G, pos, node_color='lightblue', node_size=500)
    nx.draw_networkx_labels(G, pos)

    # 添加边权重标签
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    plt.title(f"最短路径可视化 ({algorithm})")
    plt.axis('off')
    plt.show()
```

## 应用场景

### 1. 网络路由
- **OSPF协议**: 使用Dijkstra算法
- **BGP协议**: 路径向量协议
- **内容分发网络**: 最优节点选择

```python
class NetworkRouter:
    """网络路由器"""
    def __init__(self):
        self.network_graph = WeightedGraph(directed=True)
        self.routing_table = {}

    def add_link(self, router1, router2, latency):
        """添加网络链接"""
        self.network_graph.add_edge(router1, router2, latency)
        self.network_graph.add_edge(router2, router1, latency)

    def update_routing_table(self, source_router):
        """更新路由表"""
        self.routing_table[source_router] = dijkstra(self.network_graph, source_router)

    def get_next_hop(self, source, destination):
        """获取下一跳"""
        if source not in self.routing_table:
            self.update_routing_table(source)

        _, path = self.routing_table[source].get(destination, (float('inf'), []))
        if len(path) > 1:
            return path[1]  # 返回下一跳路由器
        return None
```

### 2. 地图导航
- **GPS系统**: 最短时间/距离路径
- **实时导航**: 考虑交通状况
- **多目标优化**: 时间、费用、舒适度

```python
class NavigationSystem:
    """导航系统"""
    def __init__(self):
        self.road_network = WeightedGraph(directed=True)
        self.traffic_conditions = {}

    def add_road(self, from_location, to_location, distance, normal_time):
        """添加道路"""
        self.road_network.add_edge(from_location, to_location, normal_time)
        # 反向道路可能不同（单行道等）
        self.road_network.add_edge(to_location, from_location, normal_time)

    def update_traffic(self, from_location, to_location, delay_factor):
        """更新交通状况"""
        self.traffic_conditions[(from_location, to_location)] = delay_factor

    def find_fastest_route(self, start, end):
        """查找最快路线"""
        # 创建带交通状况的临时图
        temp_graph = WeightedGraph(directed=True)

        for u in self.road_network.vertices:
            temp_graph.add_vertex(u)

        for u in self.road_network.vertices:
            for v, base_time in self.road_network.get_neighbors(u):
                delay_factor = self.traffic_conditions.get((u, v), 1.0)
                adjusted_time = base_time * delay_factor
                temp_graph.add_edge(u, v, adjusted_time)

        return dijkstra(temp_graph, start)[end]

    def find_scenic_route(self, start, end, scenic_roads):
        """查找风景路线"""
        # 将风景路线的权重降低，使其更可能被选择
        temp_graph = WeightedGraph(directed=True)

        for u in self.road_network.vertices:
            temp_graph.add_vertex(u)

        for u in self.road_network.vertices:
            for v, weight in self.road_network.get_neighbors(u):
                if (u, v) in scenic_roads:
                    adjusted_weight = weight * 0.8  # 风景路线权重降低20%
                else:
                    adjusted_weight = weight
                temp_graph.add_edge(u, v, adjusted_weight)

        return dijkstra(temp_graph, start)[end]
```

### 3. 项目管理
- **关键路径法**: 项目进度优化
- **资源分配**: 最优资源分配
- **依赖分析**: 任务依赖关系

```python
class ProjectManager:
    """项目管理器"""
    def __init__(self):
        self.task_graph = WeightedGraph(directed=True)
        self.task_durations = {}

    def add_task(self, task_id, duration):
        """添加任务"""
        self.task_graph.add_vertex(task_id)
        self.task_durations[task_id] = duration

    def add_dependency(self, task, prerequisite):
        """添加任务依赖"""
        self.task_graph.add_edge(prerequisite, task, self.task_durations[prerequisite])

    def find_critical_path(self):
        """查找关键路径"""
        try:
            # 使用Floyd-Warshall找到最长路径（关键路径）
            all_paths = floyd_warshall(self.task_graph)

            # 找到开始和结束任务
            start_tasks = [v for v in self.task_graph.vertices
                          if all(all_paths[u][v][0] == float('inf') for u in self.task_graph.vertices if u != v)]
            end_tasks = [v for v in self.task_graph.vertices
                        if all(all_paths[v][u][0] == float('inf') for u in self.task_graph.vertices if u != v)]

            if not start_tasks or not end_tasks:
                return [], 0

            start = start_tasks[0]
            end = end_tasks[0]

            distance, path = all_paths[start][end]
            return path, distance
        except ValueError:
            return [], 0

    def calculate_earliest_start(self):
        """计算最早开始时间"""
        # 使用拓扑排序
        in_degree = {v: 0 for v in self.task_graph.vertices}
        for u in self.task_graph.vertices:
            for v, _ in self.task_graph.get_neighbors(u):
                in_degree[v] += 1

        queue = [v for v in self.task_graph.vertices if in_degree[v] == 0]
        earliest_start = {v: 0 for v in self.task_graph.vertices}

        while queue:
            current = queue.pop(0)
            for neighbor, duration in self.task_graph.get_neighbors(current):
                earliest_start[neighbor] = max(earliest_start[neighbor],
                                            earliest_start[current] + duration)
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        return earliest_start
```

### 4. 金融网络
- **风险传播**: 金融风险传播路径
- **资金流动**: 最优资金路径
- **信用评估**: 信用风险评估

```python
class FinancialNetwork:
    """金融网络分析"""
    def __init__(self):
        self.graph = WeightedGraph(directed=True)
        self.risk_weights = {}

    def add_institution(self, institution_id):
        """添加金融机构"""
        self.graph.add_vertex(institution_id)

    def add_exposure(self, from_institution, to_institution, amount):
        """添加风险敞口"""
        risk_weight = amount / 1000000  # 风险权重
        self.graph.add_edge(from_institution, to_institution, risk_weight)
        self.risk_weights[(from_institution, to_institution)] = amount

    def calculate_systemic_risk(self, stressed_institution):
        """计算系统性风险"""
        # 使用Bellman-Ford计算风险传播
        try:
            risk_scores = bellman_ford(self.graph, stressed_institution)
            return risk_scores
        except ValueError:
            # 存在风险循环放大
            return None

    def find_risk_paths(self, source, threshold=0.1):
        """查找风险传播路径"""
        # 使用Dijkstra找到风险传播路径
        distances, paths = dijkstra(self.graph, source)

        risk_paths = []
        for target in self.graph.vertices:
            if target != source and distances[target] < threshold:
                risk_paths.append((target, distances[target], paths[target]))

        return sorted(risk_paths, key=lambda x: x[1], reverse=True)
```

## 高级技巧与优化

### 1. 双向Dijkstra
```python
def bidirectional_dijkstra(graph: WeightedGraph, start: str, goal: str) -> Tuple[float, List[str]]:
    """双向Dijkstra算法"""
    if start == goal:
        return 0, [start]

    # 正向搜索
    forward_dist = {vertex: float('inf') for vertex in graph.vertices}
    forward_dist[start] = 0
    forward_prev = {vertex: None for vertex in graph.vertices}
    forward_queue = [(0, start)]
    forward_visited = set()

    # 反向搜索
    reverse_dist = {vertex: float('inf') for vertex in graph.vertices}
    reverse_dist[goal] = 0
    reverse_prev = {vertex: None for vertex in graph.vertices}
    reverse_queue = [(0, goal)]
    reverse_visited = set()

    best_distance = float('inf')
    meeting_point = None

    while forward_queue and reverse_queue:
        # 正向搜索一步
        if forward_queue:
            current_dist, current = heapq.heappop(forward_queue)
            if current in forward_visited:
                continue

            forward_visited.add(current)

            # 检查是否与反向搜索相遇
            if current in reverse_visited:
                total_distance = current_dist + reverse_dist[current]
                if total_distance < best_distance:
                    best_distance = total_distance
                    meeting_point = current

            for neighbor, weight in graph.get_neighbors(current):
                if neighbor not in forward_visited:
                    new_dist = current_dist + weight
                    if new_dist < forward_dist[neighbor]:
                        forward_dist[neighbor] = new_dist
                        forward_prev[neighbor] = current
                        heapq.heappush(forward_queue, (new_dist, neighbor))

        # 反向搜索一步
        if reverse_queue:
            current_dist, current = heapq.heappop(reverse_queue)
            if current in reverse_visited:
                continue

            reverse_visited.add(current)

            # 检查是否与正向搜索相遇
            if current in forward_visited:
                total_distance = forward_dist[current] + current_dist
                if total_distance < best_distance:
                    best_distance = total_distance
                    meeting_point = current

            # 反向搜索需要反向图
            for vertex in graph.vertices:
                for neighbor, weight in graph.get_neighbors(vertex):
                    if neighbor == current and vertex not in reverse_visited:
                        new_dist = current_dist + weight
                        if new_dist < reverse_dist[vertex]:
                            reverse_dist[vertex] = new_dist
                            reverse_prev[vertex] = current
                            heapq.heappush(reverse_queue, (new_dist, vertex))

    # 重构路径
    if meeting_point is None:
        return float('inf'), []

    # 正向路径
    forward_path = []
    current = meeting_point
    while current is not None:
        forward_path.append(current)
        current = forward_prev[current]
    forward_path.reverse()

    # 反向路径
    reverse_path = []
    current = reverse_prev[meeting_point]
    while current is not None:
        reverse_path.append(current)
        current = reverse_prev[current]

    full_path = forward_path + reverse_path

    return best_distance, full_path
```

### 2. 并行最短路径算法
```python
import concurrent.futures

def parallel_dijkstra(graph: WeightedGraph, start: str, num_threads=4) -> Dict[str, float]:
    """并行Dijkstra算法"""
    distances = {vertex: float('inf') for vertex in graph.vertices}
    distances[start] = 0
    visited = set()

    def process_vertex(vertex):
        """处理顶点的邻居"""
        results = []
        for neighbor, weight in graph.get_neighbors(vertex):
            if neighbor not in visited:
                new_distance = distances[vertex] + weight
                if new_distance < distances[neighbor]:
                    results.append((neighbor, new_distance))
        return results

    with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
        # 初始化优先队列
        heap = [(0, start)]

        while heap:
            current_distance, current = heapq.heappop(heap)

            if current in visited:
                continue

            visited.add(current)

            # 并行处理当前顶点的邻居
            future = executor.submit(process_vertex, current)
            updates = future.result()

            for neighbor, new_distance in updates:
                distances[neighbor] = new_distance
                heapq.heappush(heap, (new_distance, neighbor))

    return distances
```

### 3. 动态最短路径
```python
class DynamicShortestPath:
    """动态最短路径算法"""
    def __init__(self, graph: WeightedGraph):
        self.graph = graph
        self.distances = {}
        self.paths = {}
        self.source = None

    def initialize(self, source: str):
        """初始化最短路径"""
        self.source = source
        result = dijkstra(self.graph, source)
        self.distances = {v: result[v][0] for v in result}
        self.paths = {v: result[v][1] for v in result}

    def update_edge_weight(self, u: str, v: str, old_weight: float, new_weight: float):
        """更新边权重"""
        # 找到原图中的边
        for i, (neighbor, weight) in enumerate(self.graph.adjacency_list[u]):
            if neighbor == v and weight == old_weight:
                self.graph.adjacency_list[u][i] = (v, new_weight)
                break

        # 重新计算受影响的路径
        if self.source is not None:
            if new_weight < old_weight:
                # 权重减小，可能需要更新
                self._update_decrease(u, v, old_weight, new_weight)
            else:
                # 权重增加，可能需要重新计算
                self._update_increase(u, v, old_weight, new_weight)

    def _update_decrease(self, u: str, v: str, old_weight: float, new_weight: float):
        """处理权重减少的情况"""
        if self.distances[u] + new_weight < self.distances[v]:
            self.distances[v] = self.distances[u] + new_weight
            self.paths[v] = self.paths[u] + [v]

            # 递归更新受影响的顶点
            for neighbor, weight in self.graph.get_neighbors(v):
                if self.distances[v] + weight < self.distances[neighbor]:
                    self.distances[neighbor] = self.distances[v] + weight
                    self.paths[neighbor] = self.paths[v] + [neighbor]

    def _update_increase(self, u: str, v: str, old_weight: float, new_weight: float):
        """处理权重增加的情况"""
        # 检查是否使用了这条边
        if v in self.paths[v] and u in self.paths[v]:
            # 重新计算从源到v的最短路径
            temp_result = dijkstra(self.graph, self.source)
            self.distances[v] = temp_result[v][0]
            self.paths[v] = temp_result[v][1]
```

## 练习

1. 实现一个支持约束条件的最短路径算法
2. 创建一个多目标最短路径算法（时间+费用）
3. 实现一个K最短路径算法
4. 设计一个支持实时交通状况的导航系统
5. 实现一个基于最短路径的物流配送优化算法
6. 创建一个社交网络中的影响力传播路径算法
7. 实现一个支持负权边的最短路径算法的并行版本
8. 设计一个动态更新的最短路径算法用于实时网络路由