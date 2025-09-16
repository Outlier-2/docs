# L08 - 最小生成树

## 学习目标
- 掌握最小生成树的基本概念和性质
- 理解Kruskal和Prim算法的原理
- 学会处理带权无向图的最小生成树问题
- 能够应用最小生成树解决实际问题

## 最小生成树基础

### 基本概念
- **生成树**: 包含图中所有顶点的无环子图
- **最小生成树**: 权重之和最小的生成树
- **切割**: 将顶点分成两个不交子集
- **安全边**: 连接两个不同子集的最小权重边

### 性质
- 一个连通图的最小生成树权重是唯一的
- 最小生成树可能有多个，但总权重相同
- 最小生成树包含V-1条边（V为顶点数）
- 最小生成树是无环的

## Python实现

### 1. 基础图类
```python
from typing import List, Dict, Tuple, Set
import heapq
from collections import defaultdict

class UndirectedWeightedGraph:
    """无向加权图"""
    def __init__(self):
        self.vertices = set()
        self.edges = []
        self.adjacency_list = defaultdict(list)

    def add_vertex(self, vertex):
        """添加顶点"""
        self.vertices.add(vertex)

    def add_edge(self, u, v, weight):
        """添加边"""
        self.add_vertex(u)
        self.add_vertex(v)
        self.edges.append((u, v, weight))
        self.adjacency_list[u].append((v, weight))
        self.adjacency_list[v].append((u, weight))

    def remove_edge(self, u, v, weight):
        """删除边"""
        self.edges.remove((u, v, weight))
        self.adjacency_list[u].remove((v, weight))
        self.adjacency_list[v].remove((u, weight))

    def get_vertices(self) -> Set:
        """获取所有顶点"""
        return self.vertices

    def get_edges(self) -> List[Tuple]:
        """获取所有边"""
        return self.edges

    def get_neighbors(self, vertex) -> List[Tuple]:
        """获取邻居"""
        return self.adjacency_list.get(vertex, [])

    def get_total_weight(self) -> float:
        """计算总权重"""
        return sum(weight for _, _, weight in self.edges) // 2  # 无向图每条边被记录两次

    def is_connected(self) -> bool:
        """检查图是否连通"""
        if not self.vertices:
            return True

        visited = set()
        stack = [next(iter(self.vertices))]

        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                for neighbor, _ in self.get_neighbors(current):
                    if neighbor not in visited:
                        stack.append(neighbor)

        return len(visited) == len(self.vertices)

    def __str__(self):
        result = []
        for vertex in self.vertices:
            neighbors = self.get_neighbors(vertex)
            result.append(f"{vertex}: {neighbors}")
        return "\n".join(result)
```

### 2. Kruskal算法
```python
class UnionFind:
    """并查集数据结构"""
    def __init__(self, vertices):
        self.parent = {vertex: vertex for vertex in vertices}
        self.rank = {vertex: 0 for vertex in vertices}

    def find(self, vertex):
        """查找根节点（路径压缩）"""
        if self.parent[vertex] != vertex:
            self.parent[vertex] = self.find(self.parent[vertex])
        return self.parent[vertex]

    def union(self, vertex1, vertex2):
        """合并两个集合（按秩合并）"""
        root1 = self.find(vertex1)
        root2 = self.find(vertex2)

        if root1 == root2:
            return False  # 已经在同一集合

        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1

        return True

def kruskal_mst(graph: UndirectedWeightedGraph) -> Tuple[List[Tuple], float]:
    """
    Kruskal算法求最小生成树
    时间复杂度: O(E log E) = O(E log V)
    空间复杂度: O(V)
    """
    if not graph.is_connected():
        raise ValueError("图不连通，无法生成最小生成树")

    # 按权重排序边
    edges = sorted(graph.get_edges(), key=lambda x: x[2])
    union_find = UnionFind(graph.get_vertices())
    mst_edges = []
    total_weight = 0

    for u, v, weight in edges:
        if union_find.find(u) != union_find.find(v):
            union_find.union(u, v)
            mst_edges.append((u, v, weight))
            total_weight += weight

            if len(mst_edges) == len(graph.get_vertices()) - 1:
                break

    return mst_edges, total_weight

def kruskal_with_path_compression(graph: UndirectedWeightedGraph) -> Tuple[List[Tuple], float]:
    """带路径压缩的Kruskal算法"""
    if not graph.is_connected():
        raise ValueError("图不连通，无法生成最小生成树")

    edges = sorted(graph.get_edges(), key=lambda x: x[2])
    parent = {vertex: vertex for vertex in graph.get_vertices()}
    mst_edges = []
    total_weight = 0

    def find(vertex):
        """带路径压缩的查找"""
        if parent[vertex] != vertex:
            parent[vertex] = find(parent[vertex])
        return parent[vertex]

    for u, v, weight in edges:
        root_u = find(u)
        root_v = find(v)

        if root_u != root_v:
            parent[root_u] = root_v
            mst_edges.append((u, v, weight))
            total_weight += weight

            if len(mst_edges) == len(graph.get_vertices()) - 1:
                break

    return mst_edges, total_weight
```

### 3. Prim算法
```python
def prim_mst(graph: UndirectedWeightedGraph, start_vertex=None) -> Tuple[List[Tuple], float]:
    """
    Prim算法求最小生成树
    时间复杂度: O(E log V) 使用优先队列
    空间复杂度: O(V)
    """
    if not graph.is_connected():
        raise ValueError("图不连通，无法生成最小生成树")

    vertices = graph.get_vertices()
    if not vertices:
        return [], 0

    # 选择起始顶点
    if start_vertex is None:
        start_vertex = next(iter(vertices))

    # 初始化
    mst_edges = []
    total_weight = 0
    visited = set()
    min_heap = []

    # 添加起始顶点的所有边
    visited.add(start_vertex)
    for neighbor, weight in graph.get_neighbors(start_vertex):
        heapq.heappush(min_heap, (weight, start_vertex, neighbor))

    while min_heap and len(visited) < len(vertices):
        weight, u, v = heapq.heappop(min_heap)

        if v not in visited:
            visited.add(v)
            mst_edges.append((u, v, weight))
            total_weight += weight

            # 添加新顶点的边
            for neighbor, edge_weight in graph.get_neighbors(v):
                if neighbor not in visited:
                    heapq.heappush(min_heap, (edge_weight, v, neighbor))

    return mst_edges, total_weight

def prim_lazy_mst(graph: UndirectedWeightedGraph, start_vertex=None) -> Tuple[List[Tuple], float]:
    """
    延迟Prim算法（避免重复边）
    """
    if not graph.is_connected():
        raise ValueError("图不连通，无法生成最小生成树")

    vertices = graph.get_vertices()
    if not vertices:
        return [], 0

    if start_vertex is None:
        start_vertex = next(iter(vertices))

    mst_edges = []
    total_weight = 0
    visited = set()
    min_heap = []

    visited.add(start_vertex)

    def add_edges(vertex):
        """添加顶点的所有边到堆中"""
        for neighbor, weight in graph.get_neighbors(vertex):
            if neighbor not in visited:
                heapq.heappush(min_heap, (weight, vertex, neighbor))

    add_edges(start_vertex)

    while min_heap and len(visited) < len(vertices):
        while min_heap:
            weight, u, v = heapq.heappop(min_heap)
            if v not in visited:
                break
        else:
            break

        visited.add(v)
        mst_edges.append((u, v, weight))
        total_weight += weight
        add_edges(v)

    return mst_edges, total_weight
```

### 4. Borůvka算法
```python
def boruvka_mst(graph: UndirectedWeightedGraph) -> Tuple[List[Tuple], float]:
    """
    Borůvka算法求最小生成树
    时间复杂度: O(E log V)
    适合并行化
    """
    if not graph.is_connected():
        raise ValueError("图不连通，无法生成最小生成树")

    vertices = graph.get_vertices()
    mst_edges = []
    total_weight = 0

    # 初始化每个顶点为独立组件
    component = {vertex: vertex for vertex in vertices}
    component_size = {vertex: 1 for vertex in vertices}

    def find(vertex):
        """查找组件"""
        while component[vertex] != vertex:
            component[vertex] = component[component[vertex]]
            vertex = component[vertex]
        return vertex

    def union(vertex1, vertex2):
        """合并组件"""
        root1 = find(vertex1)
        root2 = find(vertex2)

        if root1 == root2:
            return False

        if component_size[root1] < component_size[root2]:
            component[root1] = root2
            component_size[root2] += component_size[root1]
        else:
            component[root2] = root1
            component_size[root1] += component_size[root2]

        return True

    # 最多进行log V轮
    while len(mst_edges) < len(vertices) - 1:
        # 为每个组件找到最小出边
        min_edges = {}

        for u, v, weight in graph.get_edges():
            root_u = find(u)
            root_v = find(v)

            if root_u != root_v:
                if root_u not in min_edges or weight < min_edges[root_u][2]:
                    min_edges[root_u] = (u, v, weight)

                if root_v not in min_edges or weight < min_edges[root_v][2]:
                    min_edges[root_v] = (v, u, weight)

        # 添加最小边到MST
        for component_root, edge in min_edges.items():
            u, v, weight = edge
            if find(u) != find(v):
                union(u, v)
                mst_edges.append(edge)
                total_weight += weight

    return mst_edges, total_weight
```

### 5. 次小生成树
```python
def find_max_edge_on_path(graph: UndirectedWeightedGraph, mst_edges: List[Tuple], u: str, v: str) -> float:
    """在MST中查找u到v路径上的最大边权重"""
    # 构建MST的邻接表
    mst_graph = defaultdict(list)
    for edge_u, edge_v, weight in mst_edges:
        mst_graph[edge_u].append((edge_v, weight))
        mst_graph[edge_v].append((edge_u, weight))

    # BFS查找路径
    parent = {}
    max_weight = {}
    queue = [u]
    parent[u] = None
    max_weight[u] = 0

    while queue:
        current = queue.pop(0)
        if current == v:
            break

        for neighbor, weight in mst_graph[current]:
            if neighbor not in parent:
                parent[neighbor] = current
                max_weight[neighbor] = max(max_weight[current], weight)
                queue.append(neighbor)

    return max_weight.get(v, 0)

def second_best_mst(graph: UndirectedWeightedGraph) -> Tuple[List[Tuple], float]:
    """
    次小生成树算法
    时间复杂度: O(E * V)
    """
    # 首先计算最小生成树
    mst_edges, mst_weight = kruskal_mst(graph)

    best_second_weight = float('inf')
    best_second_edges = []

    # 尝试替换每条非MST边
    non_mst_edges = [edge for edge in graph.get_edges() if edge not in mst_edges]

    for new_u, new_v, new_weight in non_mst_edges:
        # 在MST中查找new_u到new_v路径上的最大边
        max_edge_weight = find_max_edge_on_path(graph, mst_edges, new_u, new_v)

        # 计算新树的总权重
        new_weight_total = mst_weight - max_edge_weight + new_weight

        if new_weight_total < best_second_weight:
            best_second_weight = new_weight_total

            # 构建新的边集
            temp_edges = mst_edges.copy()
            # 移除最大边
            for i, (u, v, weight) in enumerate(temp_edges):
                if weight == max_edge_weight:
                    del temp_edges[i]
                    break
            # 添加新边
            temp_edges.append((new_u, new_v, new_weight))
            best_second_edges = temp_edges

    return best_second_edges, best_second_weight
```

### 6. 最小生成树验证
```python
def validate_mst(graph: UndirectedWeightedGraph, mst_edges: List[Tuple]) -> Tuple[bool, str]:
    """
    验证最小生成树
    返回 (是否有效, 错误信息)
    """
    vertices = graph.get_vertices()

    # 检查边数
    if len(mst_edges) != len(vertices) - 1:
        return False, f"边数不正确，期望 {len(vertices) - 1}，实际 {len(mst_edges)}"

    # 检查是否包含所有顶点
    mst_vertices = set()
    for u, v, _ in mst_edges:
        mst_vertices.add(u)
        mst_vertices.add(v)

    if mst_vertices != vertices:
        missing = vertices - mst_vertices
        return False, f"缺少顶点: {missing}"

    # 检查连通性
    mst_graph = defaultdict(list)
    for u, v, weight in mst_edges:
        mst_graph[u].append(v)
        mst_graph[v].append(u)

    visited = set()
    stack = [next(iter(mst_vertices))]

    while stack:
        current = stack.pop()
        if current not in visited:
            visited.add(current)
            for neighbor in mst_graph[current]:
                if neighbor not in visited:
                    stack.append(neighbor)

    if len(visited) != len(mst_vertices):
        return False, "生成树不连通"

    # 检查无环
    parent = {}
    for u, v, _ in mst_edges:
        if u in parent and v in parent:
            # 检查是否形成环
            def has_cycle(vertex, target, visited_set):
                if vertex == target:
                    return True
                visited_set.add(vertex)
                for neighbor in mst_graph[vertex]:
                    if neighbor not in visited_set and has_cycle(neighbor, target, visited_set):
                        return True
                return False

            if has_cycle(u, v, set()):
                return False, "生成树包含环"

    return True, "验证通过"

def calculate_mst_weight(mst_edges: List[Tuple]) -> float:
    """计算生成树总权重"""
    return sum(weight for _, _, weight in mst_edges)
```

## 性能测试与可视化

```python
import time
import random
import matplotlib.pyplot as plt
import networkx as nx

def create_random_weighted_graph(num_vertices, num_edges, weight_range=(1, 100)):
    """创建随机加权无向图"""
    graph = UndirectedWeightedGraph()

    # 添加顶点
    vertices = [f"V{i}" for i in range(num_vertices)]
    for vertex in vertices:
        graph.add_vertex(vertex)

    # 添加边
    edges = set()
    while len(edges) < num_edges:
        u = random.choice(vertices)
        v = random.choice(vertices)
        if u != v and (u, v) not in edges and (v, u) not in edges:
            weight = random.randint(weight_range[0], weight_range[1])
            graph.add_edge(u, v, weight)
            edges.add((u, v))

    return graph

def benchmark_mst_algorithms():
    """最小生成树算法性能测试"""
    sizes = [(50, 100), (100, 300), (200, 600), (500, 1500)]

    for num_vertices, num_edges in sizes:
        print(f"\n=== 图大小: {num_vertices} 顶点, {num_edges} 边 ===")

        graph = create_random_weighted_graph(num_vertices, num_edges)

        # 测试Kruskal算法
        start_time = time.time()
        try:
            kruskal_edges, kruskal_weight = kruskal_mst(graph)
            kruskal_time = time.time() - start_time
            print(f"Kruskal: {kruskal_time:.6f}s, 权重: {kruskal_weight}")
        except Exception as e:
            print(f"Kruskal: 失败 - {e}")

        # 测试Prim算法
        start_time = time.time()
        try:
            prim_edges, prim_weight = prim_mst(graph)
            prim_time = time.time() - start_time
            print(f"Prim: {prim_time:.6f}s, 权重: {prim_weight}")
        except Exception as e:
            print(f"Prim: 失败 - {e}")

        # 测试Borůvka算法
        start_time = time.time()
        try:
            boruvka_edges, boruvka_weight = boruvka_mst(graph)
            boruvka_time = time.time() - start_time
            print(f"Borůvka: {boruvka_time:.6f}s, 权重: {boruvka_weight}")
        except Exception as e:
            print(f"Borůvka: 失败 - {e}")

def visualize_mst(graph: UndirectedWeightedGraph, mst_edges: List[Tuple], title="最小生成树"):
    """可视化最小生成树"""
    G = nx.Graph()

    # 添加顶点
    for vertex in graph.get_vertices():
        G.add_node(vertex)

    # 添加所有边
    for u, v, weight in graph.get_edges():
        G.add_edge(u, v, weight=weight)

    # 设置MST边的颜色
    mst_set = set((u, v) for u, v, _ in mst_edges)
    edge_colors = ['red' if (u, v) in mst_set or (v, u) in mst_set else 'gray'
                   for u, v in G.edges()]

    # 设置边的宽度
    edge_widths = [3 if (u, v) in mst_set or (v, u) in mst_set else 1
                   for u, v in G.edges()]

    # 绘制图
    plt.figure(figsize=(12, 8))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, with_labels=True, node_color='lightblue',
            node_size=500, font_size=10, font_weight='bold',
            edge_color=edge_colors, width=edge_widths)

    # 添加权重标签
    edge_labels = {(u, v): d['weight'] for u, v, d in G.edges(data=True)}
    nx.draw_networkx_edge_labels(G, pos, edge_labels)

    plt.title(title)
    plt.axis('off')
    plt.show()
```

## 应用场景

### 1. 网络设计
- **通信网络**: 最小成本连接所有节点
- **电力网络**: 优化电网布线
- **交通网络**: 道路建设优化

```python
class NetworkDesigner:
    """网络设计器"""
    def __init__(self):
        self.network_graph = UndirectedWeightedGraph()
        self.cities = {}

    def add_city(self, city_id, name, x_coord, y_coord):
        """添加城市"""
        self.cities[city_id] = {'name': name, 'x': x_coord, 'y': y_coord}
        self.network_graph.add_vertex(city_id)

    def calculate_distance(self, city1_id, city2_id):
        """计算城市间距离"""
        city1 = self.cities[city1_id]
        city2 = self.cities[city2_id]
        return ((city1['x'] - city2['x'])**2 + (city1['y'] - city2['y'])**2)**0.5

    def add_connection(self, city1_id, city2_id, cost_per_unit):
        """添加连接"""
        distance = self.calculate_distance(city1_id, city2_id)
        total_cost = distance * cost_per_unit
        self.network_graph.add_edge(city1_id, city2_id, total_cost)

    def design_optimal_network(self):
        """设计最优网络"""
        mst_edges, total_cost = kruskal_mst(self.network_graph)

        network_design = []
        for u, v, cost in mst_edges:
            network_design.append({
                'from': self.cities[u]['name'],
                'to': self.cities[v]['name'],
                'cost': cost,
                'distance': self.calculate_distance(u, v)
            })

        return {
            'total_cost': total_cost,
            'connections': network_design,
            'cities_connected': len(self.cities)
        }

    def find_backup_routes(self):
        """查找备用路由（次小生成树）"""
        try:
            second_edges, second_cost = second_best_mst(self.network_graph)
            return second_cost, second_edges
        except:
            return None, []
```

### 2. 聚类分析
- **图像分割**: 最小生成树用于图像分割
- **数据聚类**: 基于距离的聚类算法
- **层次聚类**: 构建层次聚类树

```python
class MSTCluster:
    """基于MST的聚类"""
    def __init__(self, data_points):
        self.data_points = data_points
        self.graph = self._build_similarity_graph()

    def _build_similarity_graph(self):
        """构建相似度图"""
        graph = UndirectedWeightedGraph()

        # 添加顶点
        for i, point in enumerate(self.data_points):
            graph.add_vertex(f"P{i}")

        # 添加边（使用欧氏距离）
        for i in range(len(self.data_points)):
            for j in range(i + 1, len(self.data_points)):
                distance = self._calculate_distance(self.data_points[i], self.data_points[j])
                graph.add_edge(f"P{i}", f"P{j}", distance)

        return graph

    def _calculate_distance(self, point1, point2):
        """计算两点间距离"""
        return sum((a - b)**2 for a, b in zip(point1, point2))**0.5

    def cluster(self, n_clusters):
        """聚类到指定数量"""
        # 构建MST
        mst_edges, _ = kruskal_mst(self.graph)

        # 按权重排序边
        sorted_edges = sorted(mst_edges, key=lambda x: x[2], reverse=True)

        # 移除权重最大的n_clusters-1条边
        for i in range(n_clusters - 1):
            if i < len(sorted_edges):
                u, v, weight = sorted_edges[i]
                self.graph.remove_edge(u, v, weight)

        # 找到连通分量
        return self._find_connected_components()

    def _find_connected_components(self):
        """查找连通分量"""
        visited = set()
        components = []

        def dfs(vertex, component):
            visited.add(vertex)
            component.append(vertex)
            for neighbor, _ in self.graph.get_neighbors(vertex):
                if neighbor not in visited:
                    dfs(neighbor, component)

        for vertex in self.graph.get_vertices():
            if vertex not in visited:
                component = []
                dfs(vertex, component)
                components.append(component)

        return components
```

### 3. 电路设计
- **PCB布线**: 最小化导线长度
- **VLSI设计**: 优化芯片内部连接
- **电路板布局**: 减少交叉和长度

```python
class CircuitDesigner:
    """电路设计器"""
    def __init__(self):
        self.component_graph = UndirectedWeightedGraph()
        self.components = {}

    def add_component(self, comp_id, name, x, y):
        """添加组件"""
        self.components[comp_id] = {'name': name, 'x': x, 'y': y}
        self.component_graph.add_vertex(comp_id)

    def add_required_connection(self, comp1_id, comp2_id):
        """添加必需连接"""
        comp1 = self.components[comp1_id]
        comp2 = self.components[comp2_id]
        distance = ((comp1['x'] - comp2['x'])**2 + (comp1['y'] - comp2['y'])**2)**0.5
        self.component_graph.add_edge(comp1_id, comp2_id, distance)

    def design_minimal_circuit(self):
        """设计最小化电路"""
        mst_edges, total_length = prim_mst(self.component_graph)

        circuit_design = []
        for u, v, length in mst_edges:
            circuit_design.append({
                'from': self.components[u]['name'],
                'to': self.components[v]['name'],
                'length': length,
                'from_pos': (self.components[u]['x'], self.components[u]['y']),
                'to_pos': (self.components[v]['x'], self.components[v]['y'])
            })

        return {
            'total_length': total_length,
            'connections': circuit_design
        }

    def calculate_wire_efficiency(self):
        """计算布线效率"""
        total_possible_length = sum(
            ((self.components[u]['x'] - self.components[v]['x'])**2 +
             (self.components[u]['y'] - self.components[v]['y'])**2)**0.5
            for u, v, _ in self.component_graph.get_edges()
        ) / 2

        mst_length = prim_mst(self.component_graph)[1]

        efficiency = (total_possible_length - mst_length) / total_possible_length * 100
        return efficiency
```

### 4. 生物信息学
- **系统发育树**: 构建物种进化树
- **蛋白质结构**: 分析蛋白质折叠
- **基因网络**: 基因相互作用网络

```python
class PhylogeneticTree:
    """系统发育树构建"""
    def __init__(self, species_data):
        self.species_data = species_data
        self.distance_graph = self._build_distance_graph()

    def _build_distance_graph(self):
        """构建距离图"""
        graph = UndirectedWeightedGraph()

        species_names = list(self.species_data.keys())
        for name in species_names:
            graph.add_vertex(name)

        # 计算物种间距离（基于序列差异）
        for i, species1 in enumerate(species_names):
            for j, species2 in enumerate(species_names[i+1:], i+1):
                distance = self._calculate_genetic_distance(
                    self.species_data[species1],
                    self.species_data[species2]
                )
                graph.add_edge(species1, species2, distance)

        return graph

    def _calculate_genetic_distance(self, seq1, seq2):
        """计算遗传距离"""
        if len(seq1) != len(seq2):
            return float('inf')

        differences = sum(1 for a, b in zip(seq1, seq2) if a != b)
        return differences / len(seq1)

    def build_phylogenetic_tree(self):
        """构建系统发育树"""
        mst_edges, total_distance = kruskal_mst(self.distance_graph)

        # 按距离排序边，构建层次树
        tree_edges = sorted(mst_edges, key=lambda x: x[2])

        tree = {}
        for u, v, distance in tree_edges:
            tree[f"{u}-{v}"] = {
                'species': [u, v],
                'distance': distance,
                'time': distance * 1000000  # 假设的进化时间
            }

        return {
            'tree': tree,
            'total_evolutionary_distance': total_distance,
            'species_count': len(self.species_data)
        }
```

## 高级技巧与优化

### 1. 并行最小生成树
```python
import concurrent.futures

class ParallelMST:
    """并行最小生成树"""
    def __init__(self, graph: UndirectedWeightedGraph):
        self.graph = graph

    def parallel_kruskal(self, num_threads=4):
        """并行Kruskal算法"""
        edges = self.graph.get_edges()
        vertices = self.graph.get_vertices()

        # 分配边到不同线程
        edges_per_thread = len(edges) // num_threads
        edge_chunks = [edges[i:i + edges_per_thread]
                      for i in range(0, len(edges), edges_per_thread)]

        def process_chunk(chunk):
            """处理边的分块"""
            union_find = UnionFind(vertices)
            local_edges = []
            for u, v, weight in chunk:
                if union_find.find(u) != union_find.find(v):
                    union_find.union(u, v)
                    local_edges.append((u, v, weight))
            return local_edges

        # 并行处理
        with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = [executor.submit(process_chunk, chunk) for chunk in edge_chunks]
            results = [future.result() for future in concurrent.futures.as_completed(futures)]

        # 合并结果
        all_edges = [edge for chunk in results for edge in chunk]
        final_union_find = UnionFind(vertices)
        mst_edges = []
        total_weight = 0

        for u, v, weight in all_edges:
            if final_union_find.find(u) != final_union_find.find(v):
                final_union_find.union(u, v)
                mst_edges.append((u, v, weight))
                total_weight += weight

        return mst_edges, total_weight

    def parallel_prim(self, num_threads=4):
        """并行Prim算法"""
        vertices = self.graph.get_vertices()
        if not vertices:
            return [], 0

        start_vertex = next(iter(vertices))
        mst_edges = []
        total_weight = 0
        visited = set([start_vertex])

        def process_vertex(vertex):
            """处理顶点的边"""
            edges = []
            for neighbor, weight in self.graph.get_neighbors(vertex):
                if neighbor not in visited:
                    edges.append((weight, vertex, neighbor))
            return edges

        while len(visited) < len(vertices):
            # 并行处理所有已访问顶点的边
            with concurrent.futures.ThreadPoolExecutor(max_workers=num_threads) as executor:
                futures = [executor.submit(process_vertex, vertex) for vertex in visited]
                all_edges = []
                for future in concurrent.futures.as_completed(futures):
                    all_edges.extend(future.result())

            if not all_edges:
                break

            # 找到最小权重边
            min_edge = min(all_edges)
            weight, u, v = min_edge

            if v not in visited:
                visited.add(v)
                mst_edges.append((u, v, weight))
                total_weight += weight

        return mst_edges, total_weight
```

### 2. 动态最小生成树
```python
class DynamicMST:
    """动态最小生成树"""
    def __init__(self, graph: UndirectedWeightedGraph):
        self.graph = graph
        self.mst_edges = []
        self.total_weight = 0
        self.edge_weight_map = {}

    def initialize(self):
        """初始化MST"""
        self.mst_edges, self.total_weight = kruskal_mst(self.graph)
        self._build_edge_weight_map()

    def _build_edge_weight_map(self):
        """构建边权重映射"""
        self.edge_weight_map = {(u, v): weight for u, v, weight in self.graph.get_edges()}

    def add_edge(self, u, v, weight):
        """添加边"""
        self.graph.add_edge(u, v, weight)
        self.edge_weight_map[(u, v)] = weight
        self.edge_weight_map[(v, u)] = weight

        # 如果新边能改进MST，则更新
        if self._can_improve_mst(u, v, weight):
            self._update_mst_with_new_edge(u, v, weight)

    def remove_edge(self, u, v, weight):
        """删除边"""
        self.graph.remove_edge(u, v, weight)
        del self.edge_weight_map[(u, v)]
        del self.edge_weight_map[(v, u)]

        # 如果删除的是MST边，需要重新计算
        if (u, v, weight) in self.mst_edges or (v, u, weight) in self.mst_edges:
            self._recalculate_mst()

    def update_edge_weight(self, u, v, old_weight, new_weight):
        """更新边权重"""
        self.graph.remove_edge(u, v, old_weight)
        self.graph.add_edge(u, v, new_weight)
        self.edge_weight_map[(u, v)] = new_weight
        self.edge_weight_map[(v, u)] = new_weight

        if (u, v, old_weight) in self.mst_edges or (v, u, old_weight) in self.mst_edges:
            if new_weight > old_weight:
                self._recalculate_mst()
            else:
                # 权重减少，保持原MST
                pass
        else:
            if new_weight < old_weight and self._can_improve_mst(u, v, new_weight):
                self._update_mst_with_new_edge(u, v, new_weight)

    def _can_improve_mst(self, u, v, weight):
        """检查新边是否能改进MST"""
        if not self.mst_edges:
            return True

        # 在MST中查找u到v路径上的最大边
        max_weight = find_max_edge_on_path(self.graph, self.mst_edges, u, v)
        return weight < max_weight

    def _update_mst_with_new_edge(self, u, v, weight):
        """用新边更新MST"""
        # 找到并移除路径上的最大边
        max_weight = find_max_edge_on_path(self.graph, self.mst_edges, u, v)

        for i, (edge_u, edge_v, edge_weight) in enumerate(self.mst_edges):
            if edge_weight == max_weight:
                del self.mst_edges[i]
                break

        # 添加新边
        self.mst_edges.append((u, v, weight))
        self.total_weight = self.total_weight - max_weight + weight

    def _recalculate_mst(self):
        """重新计算MST"""
        self.mst_edges, self.total_weight = kruskal_mst(self.graph)
```

### 3. 近似最小生成树
```python
class ApproximateMST:
    """近似最小生成树"""
    def __init__(self, graph: UndirectedWeightedGraph, epsilon=0.1):
        self.graph = graph
        self.epsilon = epsilon

    def approximate_mst(self):
        """近似最小生成树算法"""
        vertices = self.graph.get_vertices()
        if not vertices:
            return [], 0

        # 构建轻量生成树
        light_edges = self._find_light_edges()
        light_tree_edges, light_weight = self._build_spanning_tree(light_edges)

        # 改进近似
        improved_edges = self._improve_approximation(light_tree_edges)
        improved_weight = sum(weight for _, _, weight in improved_edges)

        return improved_edges, improved_weight

    def _find_light_edges(self):
        """查找轻量边"""
        edges = self.graph.get_edges()
        light_edges = []

        # 按权重排序
        sorted_edges = sorted(edges, key=lambda x: x[2])

        # 选择每个顶点的轻量边
        vertex_min_edges = {}

        for u, v, weight in sorted_edges:
            if u not in vertex_min_edges:
                vertex_min_edges[u] = (u, v, weight)
            if v not in vertex_min_edges:
                vertex_min_edges[v] = (v, u, weight)

        light_edges.extend(vertex_min_edges.values())
        return light_edges

    def _build_spanning_tree(self, edges):
        """构建生成树"""
        if not edges:
            return [], 0

        # 使用Kruskal算法
        vertices = self.graph.get_vertices()
        union_find = UnionFind(vertices)
        tree_edges = []
        total_weight = 0

        for u, v, weight in edges:
            if union_find.find(u) != union_find.find(v):
                union_find.union(u, v)
                tree_edges.append((u, v, weight))
                total_weight += weight

        # 如果不连通，添加剩余边
        remaining_edges = [edge for edge in self.graph.get_edges() if edge not in edges]
        for u, v, weight in sorted(remaining_edges, key=lambda x: x[2]):
            if union_find.find(u) != union_find.find(v):
                union_find.union(u, v)
                tree_edges.append((u, v, weight))
                total_weight += weight

        return tree_edges, total_weight

    def _improve_approximation(self, initial_edges):
        """改进近似结果"""
        # 随机交换边以改进解
        improved_edges = initial_edges.copy()
        current_weight = sum(weight for _, _, weight in improved_edges)

        for _ in range(100):  # 最多尝试100次
            # 随机选择一条边进行替换
            if len(improved_edges) > 1:
                idx_to_remove = random.randint(0, len(improved_edges) - 1)
                removed_edge = improved_edges[idx_to_remove]

                # 构建不包含该边的图
                temp_graph = UndirectedWeightedGraph()
                for vertex in self.graph.get_vertices():
                    temp_graph.add_vertex(vertex)

                for edge in self.graph.get_edges():
                    if edge != removed_edge and edge[::-1] != removed_edge:
                        temp_graph.add_edge(*edge)

                # 尝试找到新的边
                try:
                    new_edges, new_weight = self._find_replacement_edge(temp_graph, removed_edge)
                    if new_weight < removed_edge[2]:
                        improved_edges[idx_to_remove] = new_edges[0]
                except:
                    continue

        return improved_edges

    def _find_replacement_edge(self, graph, removed_edge):
        """查找替换边"""
        # 移除removed_edge后，找到连接两个分量的最小边
        u, v, _ = removed_edge

        # 找到两个连通分量
        component1 = self._find_component(graph, u)
        component2 = self._find_component(graph, v)

        # 找到连接两个分量的最小边
        min_edge = None
        min_weight = float('inf')

        for edge in self.graph.get_edges():
            if (edge[0] in component1 and edge[1] in component2) or \
               (edge[1] in component1 and edge[0] in component2):
                if edge[2] < min_weight:
                    min_weight = edge[2]
                    min_edge = edge

        if min_edge:
            return [min_edge], min_weight
        else:
            return [removed_edge], removed_edge[2]

    def _find_component(self, graph, start_vertex):
        """查找连通分量"""
        visited = set()
        stack = [start_vertex]

        while stack:
            current = stack.pop()
            if current not in visited:
                visited.add(current)
                for neighbor, _ in graph.get_neighbors(current):
                    if neighbor not in visited:
                        stack.append(neighbor)

        return visited
```

## 练习

1. 实现一个支持删除边的动态最小生成树算法
2. 创建一个基于最小生成树的图像分割算法
3. 实现一个并行版本的Borůvka算法
4. 设计一个带约束条件的最小生成树算法
5. 实现一个分布式最小生成树算法
6. 创建一个最小生成树的增量式更新算法
7. 实现一个基于最小生成树的社区发现算法
8. 设计一个处理大规模图的流式最小生成树算法