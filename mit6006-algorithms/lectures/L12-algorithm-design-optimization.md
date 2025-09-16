# L12 - 算法设计与优化

## 学习目标
- 掌握算法设计的基本范式
- 理解算法优化的各种技巧
- 学会分析算法的性能瓶颈
- 能够设计高效的问题解决方案

## 算法设计范式

### 1. 分治算法 (Divide and Conquer)
- **原理**: 将问题分解为子问题，递归解决，合并结果
- **适用**: 具有最优子结构的问题
- **复杂度**: 通常为 O(n log n)

### 2. 贪心算法 (Greedy)
- **原理**: 每步选择局部最优解
- **适用**: 具有贪心选择性质的问题
- **验证**: 需要证明贪心选择的正确性

### 3. 动态规划 (Dynamic Programming)
- **原理**: 存储子问题解，避免重复计算
- **适用**: 重叠子问题和最优子结构
- **优化**: 空间优化、状态压缩

### 4. 回溯算法 (Backtracking)
- **原理**: 系统地搜索解空间
- **适用**: 组合优化、约束满足问题
- **优化**: 剪枝、启发式搜索

### 5. 分支限界 (Branch and Bound)
- **原理**: 搜索最优解，通过界限函数剪枝
- **适用**: 组合优化问题
- **优化**: 松弛、启发式下界

## Python实现

### 1. 分治算法示例

#### 1.1 快速幂算法
```python
def fast_pow(base: float, exponent: int) -> float:
    """
    快速幂算法
    时间复杂度: O(log n)
    空间复杂度: O(log n)
    """
    if exponent == 0:
        return 1
    if exponent == 1:
        return base

    half = fast_pow(base, exponent // 2)
    if exponent % 2 == 0:
        return half * half
    else:
        return half * half * base

def fast_pow_iterative(base: float, exponent: int) -> float:
    """
    迭代快速幂算法
    空间复杂度: O(1)
    """
    result = 1
    current_base = base

    while exponent > 0:
        if exponent % 2 == 1:
            result *= current_base
        current_base *= current_base
        exponent //= 2

    return result

def matrix_power(matrix, power):
    """
    矩阵快速幂
    用于线性递推的优化
    """
    n = len(matrix)
    result = [[1 if i == j else 0 for j in range(n)] for i in range(n)]  # 单位矩阵

    while power > 0:
        if power % 2 == 1:
            result = matrix_multiply(result, matrix)
        matrix = matrix_multiply(matrix, matrix)
        power //= 2

    return result

def matrix_multiply(a, b):
    """矩阵乘法"""
    n = len(a)
    m = len(b[0])
    p = len(b)
    result = [[0] * m for _ in range(n)]

    for i in range(n):
        for j in range(m):
            for k in range(p):
                result[i][j] += a[i][k] * b[k][j]

    return result
```

#### 1.2 查找第k大元素
```python
import random

def quick_select(arr: list, k: int) -> int:
    """
    快速选择算法
    时间复杂度: O(n) 平均
    空间复杂度: O(log n)
    """
    if len(arr) == 1:
        return arr[0]

    pivot = random.choice(arr)
    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x > pivot]
    equal = [x for x in arr if x == pivot]

    if k <= len(left):
        return quick_select(left, k)
    elif k <= len(left) + len(equal):
        return equal[0]
    else:
        return quick_select(right, k - len(left) - len(equal))

def median_of_medians(arr: list) -> int:
    """
    中位数的中位数算法
    时间复杂度: O(n) 最坏情况
    """
    if len(arr) <= 5:
        return sorted(arr)[len(arr) // 2]

    # 分成5个元素一组
    groups = [arr[i:i+5] for i in range(0, len(arr), 5)]
    medians = [median_of_medians(group) for group in groups]
    pivot = median_of_medians(medians)

    left = [x for x in arr if x < pivot]
    right = [x for x in arr if x > pivot]

    if len(left) > len(arr) // 2:
        return median_of_medians(left)
    elif len(arr) - len(right) > len(arr) // 2:
        return median_of_medians(right)
    else:
        return pivot
```

#### 1.3 最近点对问题
```python
import math

def closest_pair(points: list) -> tuple:
    """
    最近点对问题
    时间复杂度: O(n log n)
    """
    def distance(p1, p2):
        return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

    def closest_pair_recursive(points_sorted_x, points_sorted_y):
        n = len(points_sorted_x)
        if n <= 3:
            # 暴力法
            min_dist = float('inf')
            closest_pair = None
            for i in range(n):
                for j in range(i + 1, n):
                    dist = distance(points_sorted_x[i], points_sorted_x[j])
                    if dist < min_dist:
                        min_dist = dist
                        closest_pair = (points_sorted_x[i], points_sorted_x[j])
            return closest_pair, min_dist

        mid = n // 2
        mid_point = points_sorted_x[mid]

        # 递归求解左右两侧
        left_x = points_sorted_x[:mid]
        right_x = points_sorted_x[mid:]

        mid_y = mid_point[1]
        left_y = [p for p in points_sorted_y if p[0] < mid_point[0]]
        right_y = [p for p in points_sorted_y if p[0] >= mid_point[0]]

        left_pair, left_dist = closest_pair_recursive(left_x, left_y)
        right_pair, right_dist = closest_pair_recursive(right_x, right_y)

        if left_dist < right_dist:
            min_pair, min_dist = left_pair, left_dist
        else:
            min_pair, min_dist = right_pair, right_dist

        # 检查跨越中线的点对
        strip = [p for p in points_sorted_y if abs(p[0] - mid_point[0]) < min_dist]

        for i in range(len(strip)):
            for j in range(i + 1, min(i + 7, len(strip))):
                if strip[j][1] - strip[i][1] >= min_dist:
                    break
                dist = distance(strip[i], strip[j])
                if dist < min_dist:
                    min_pair, min_dist = (strip[i], strip[j]), dist

        return min_pair, min_dist

    points_sorted_x = sorted(points, key=lambda p: p[0])
    points_sorted_y = sorted(points, key=lambda p: p[1])
    return closest_pair_recursive(points_sorted_x, points_sorted_y)
```

### 2. 贪心算法示例

#### 2.1 活动选择问题
```python
def activity_selection(activities: list) -> list:
    """
    活动选择问题
    时间复杂度: O(n log n)
    """
    # 按结束时间排序
    activities.sort(key=lambda x: x[1])
    selected = [activities[0]]
    last_end = activities[0][1]

    for i in range(1, len(activities)):
        if activities[i][0] >= last_end:
            selected.append(activities[i])
            last_end = activities[i][1]

    return selected

def weighted_activity_selection(activities: list) -> list:
    """
    带权重的活动选择问题
    需要使用动态规划
    """
    if not activities:
        return []

    # 按结束时间排序
    activities.sort(key=lambda x: x[1])
    n = len(activities)

    # dp[i] = 前i个活动的最大权重
    dp = [0] * (n + 1)
    selected = [[] for _ in range(n + 1)]

    for i in range(1, n + 1):
        # 不选择第i个活动
        dp[i] = dp[i-1]
        selected[i] = selected[i-1][:]

        # 找到最后一个与第i个活动兼容的活动
        j = i - 1
        while j > 0 and activities[j-1][1] > activities[i-1][0]:
            j -= 1

        # 选择第i个活动
        if j >= 0:
            if dp[i] < dp[j] + activities[i-1][2]:
                dp[i] = dp[j] + activities[i-1][2]
                selected[i] = selected[j][:] + [activities[i-1]]

    return selected[n]
```

#### 2.2 哈夫曼编码
```python
import heapq
from collections import defaultdict

class HuffmanNode:
    """哈夫曼树节点"""
    def __init__(self, char=None, freq=0, left=None, right=None):
        self.char = char
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def huffman_encoding(text: str) -> dict:
    """
    哈夫曼编码
    时间复杂度: O(n log n)
    """
    if not text:
        return {}

    # 统计字符频率
    freq_map = defaultdict(int)
    for char in text:
        freq_map[char] += 1

    # 构建优先队列
    heap = []
    for char, freq in freq_map.items():
        heapq.heappush(heap, HuffmanNode(char=char, freq=freq))

    # 构建哈夫曼树
    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        merged = HuffmanNode(freq=left.freq + right.freq, left=left, right=right)
        heapq.heappush(heap, merged)

    # 生成编码
    encoding_map = {}

    def generate_codes(node, code=""):
        if node.char is not None:
            encoding_map[node.char] = code
        else:
            generate_codes(node.left, code + "0")
            generate_codes(node.right, code + "1")

    if heap:
        generate_codes(heap[0])

    return encoding_map

def huffman_compress(text: str) -> tuple:
    """
    哈夫曼压缩
    """
    encoding_map = huffman_encoding(text)

    if not encoding_map:
        return "", {}

    # 编码文本
    encoded_bits = ""
    for char in text:
        encoded_bits += encoding_map[char]

    # 补零到字节对齐
    padding = (8 - len(encoded_bits) % 8) % 8
    encoded_bits += "0" * padding

    # 转换为字节
    encoded_bytes = bytearray()
    for i in range(0, len(encoded_bits), 8):
        byte = int(encoded_bits[i:i+8], 2)
        encoded_bytes.append(byte)

    return bytes(encoded_bytes), {'encoding_map': encoding_map, 'padding': padding}
```

#### 2.3 最小生成树
```python
class DisjointSet:
    """并查集"""
    def __init__(self, elements):
        self.parent = {element: element for element in elements}
        self.rank = {element: 0 for element in elements}

    def find(self, element):
        if self.parent[element] != element:
            self.parent[element] = self.find(self.parent[element])
        return self.parent[element]

    def union(self, element1, element2):
        root1 = self.find(element1)
        root2 = self.find(element2)

        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1

def kruskal_mst(graph):
    """
    Kruskal最小生成树算法
    贪心算法的典型应用
    """
    edges = []
    vertices = set()

    for u in graph:
        for v, weight in graph[u].items():
            edges.append((weight, u, v))
            vertices.add(u)
            vertices.add(v)

    edges.sort()  # 按权重排序
    ds = DisjointSet(vertices)
    mst = []

    for weight, u, v in edges:
        if ds.find(u) != ds.find(v):
            ds.union(u, v)
            mst.append((u, v, weight))

    return mst
```

### 3. 回溯算法示例

#### 3.1 N皇后问题
```python
def solve_n_queens(n: int) -> list:
    """
    N皇后问题
    时间复杂度: O(n!)
    """
    def is_safe(board, row, col):
        # 检查同一列
        for i in range(row):
            if board[i] == col:
                return False

        # 检查左对角线
        for i in range(row):
            if board[i] - i == col - row:
                return False

        # 检查右对角线
        for i in range(row):
            if board[i] + i == col + row:
                return False

        return True

    def backtrack(row, board):
        if row == n:
            solutions.append(board[:])
            return

        for col in range(n):
            if is_safe(board, row, col):
                board[row] = col
                backtrack(row + 1, board)

    solutions = []
    board = [-1] * n
    backtrack(0, board)
    return solutions

def solve_n_queens_optimized(n: int) -> list:
    """
    优化的N皇后问题
    使用位运算
    """
    def backtrack(row, cols, diag1, diag2):
        if row == n:
            solutions.append([])
            return

        available = ~(cols | diag1 | diag2) & ((1 << n) - 1)

        while available:
            col = available & -available
            available ^= col

            # 将col转换为列索引
            col_idx = bin(col).count('0') - 1

            solutions[-1].append(col_idx)

            backtrack(row + 1,
                      cols | col,
                      (diag1 | col) << 1,
                      (diag2 | col) >> 1)

    solutions = []
    backtrack(0, 0, 0, 0)
    return solutions
```

#### 3.2 数独求解
```python
def solve_sudoku(board):
    """
    数独求解
    """
    def is_valid(row, col, num):
        # 检查行
        for j in range(9):
            if board[row][j] == num:
                return False

        # 检查列
        for i in range(9):
            if board[i][col] == num:
                return False

        # 检查3x3方格
        start_row, start_col = 3 * (row // 3), 3 * (col // 3)
        for i in range(start_row, start_row + 3):
            for j in range(start_col, start_col + 3):
                if board[i][j] == num:
                    return False

        return True

    def find_empty():
        for i in range(9):
            for j in range(9):
                if board[i][j] == 0:
                    return (i, j)
        return None

    def backtrack():
        empty = find_empty()
        if not empty:
            return True

        row, col = empty

        for num in range(1, 10):
            if is_valid(row, col, num):
                board[row][col] = num

                if backtrack():
                    return True

                board[row][col] = 0

        return False

    backtrack()
    return board
```

#### 3.3 子集和问题
```python
def subset_sum(numbers: list, target: int) -> list:
    """
    子集和问题
    """
    def backtrack(start, target, path):
        if target == 0:
            solutions.append(path[:])
            return

        for i in range(start, len(numbers)):
            if numbers[i] > target:
                continue

            path.append(numbers[i])
            backtrack(i + 1, target - numbers[i], path)
            path.pop()

    solutions = []
    backtrack(0, target, [])
    return solutions

def subset_sum_with_memoization(numbers: list, target: int) -> list:
    """
    带记忆化的子集和问题
    """
    memo = {}

    def backtrack(start, target):
        if target == 0:
            return [[]]
        if target < 0 or start >= len(numbers):
            return []

        if (start, target) in memo:
            return memo[(start, target)]

        # 不包含当前数字
        result = backtrack(start + 1, target)

        # 包含当前数字
        for subset in backtrack(start + 1, target - numbers[start]):
            result.append([numbers[start]] + subset)

        memo[(start, target)] = result
        return result

    return backtrack(0, target)
```

### 4. 分支限界示例

#### 4.1 旅行商问题
```python
import math
import heapq

def tsp_branch_and_bound(distances):
    """
    旅行商问题的分支限界算法
    """
    n = len(distances)

    def calculate_lower_bound(path, visited):
        """计算下界"""
        if len(path) == n:
            return 0

        # 已经访问的路径长度
        current_cost = sum(distances[path[i]][path[i+1]] for i in range(len(path)-1))

        # 估计剩余路径的最小长度
        min_outgoing = []
        for i in range(n):
            if i not in visited:
                min_dist = min(distances[i][j] for j in range(n) if j != i)
                min_outgoing.append(min_dist)

        return current_cost + sum(min_outgoing)

    def branch_and_bound():
        best_solution = None
        best_cost = float('inf')

        # 优先队列：(下界, 路径, 已访问)
        heap = [(0, [0], {0})]

        while heap:
            lower_bound, path, visited = heapq.heappop(heap)

            if lower_bound >= best_cost:
                continue

            if len(path) == n:
                # 完成路径，回到起点
                total_cost = (sum(distances[path[i]][path[i+1]] for i in range(n-1)) +
                             distances[path[-1]][path[0]])
                if total_cost < best_cost:
                    best_cost = total_cost
                    best_solution = path
                continue

            # 扩展所有可能的下一个城市
            current = path[-1]
            for next_city in range(n):
                if next_city not in visited:
                    new_path = path + [next_city]
                    new_visited = visited | {next_city}
                    new_lower_bound = calculate_lower_bound(new_path, new_visited)

                    if new_lower_bound < best_cost:
                        heapq.heappush(heap, (new_lower_bound, new_path, new_visited))

        return best_solution, best_cost

    return branch_and_bound()
```

#### 4.2 0-1背包问题
```python
def knapsack_branch_and_bound(weights, values, capacity):
    """
    0-1背包问题的分支限界算法
    """
    n = len(weights)

    class Node:
        def __init__(self, level, value, weight, bound):
            self.level = level
            self.value = value
            self.weight = weight
            self.bound = bound

        def __lt__(self, other):
            return self.bound > other.bound  # 最大堆

    def calculate_bound(node):
        """计算节点的上界"""
        if node.weight >= capacity:
            return 0

        # 已选物品的价值
        bound = node.value

        # 剩余容量
        remaining_weight = capacity - node.weight

        # 按价值密度排序剩余物品
        j = node.level + 1
        while j < n and weights[j] <= remaining_weight:
            bound += values[j]
            remaining_weight -= weights[j]
            j += 1

        # 添加最后一个物品的分数
        if j < n:
            bound += values[j] * (remaining_weight / weights[j])

        return bound

    # 按价值密度排序
    items = list(zip(weights, values))
    items.sort(key=lambda x: x[1]/x[0], reverse=True)
    weights, values = zip(*items)

    max_value = 0
    heap = []

    # 创建根节点
    root = Node(-1, 0, 0, 0)
    root.bound = calculate_bound(root)
    heapq.heappush(heap, root)

    while heap:
        node = heapq.heappop(heap)

        if node.bound <= max_value:
            continue

        if node.level == n - 1:
            max_value = max(max_value, node.value)
            continue

        # 考虑下一个物品
        next_level = node.level + 1

        # 包含下一个物品
        if node.weight + weights[next_level] <= capacity:
            child = Node(next_level,
                        node.value + values[next_level],
                        node.weight + weights[next_level],
                        0)
            child.bound = calculate_bound(child)
            if child.bound > max_value:
                heapq.heappush(heap, child)
                max_value = max(max_value, child.value)

        # 不包含下一个物品
        child = Node(next_level, node.value, node.weight, 0)
        child.bound = calculate_bound(child)
        if child.bound > max_value:
            heapq.heappush(heap, child)

    return max_value
```

### 5. 启发式算法示例

#### 5.1 A*算法
```python
import heapq
import math

def a_star_search(graph, start, goal, heuristic):
    """
    A*搜索算法
    """
    def reconstruct_path(came_from, current):
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path

    open_set = [(0, start)]
    came_from = {}
    g_score = {start: 0}
    f_score = {start: heuristic(start, goal)}

    while open_set:
        current_f, current = heapq.heappop(open_set)

        if current == goal:
            return reconstruct_path(came_from, current)

        for neighbor in graph.get(current, []):
            tentative_g_score = g_score[current] + graph[current][neighbor]

            if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # 没有找到路径

def euclidean_distance(point1, point2):
    """欧几里得距离启发式"""
    return math.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def manhattan_distance(point1, point2):
    """曼哈顿距离启发式"""
    return abs(point1[0] - point2[0]) + abs(point1[1] - point2[1])
```

#### 5.2 遗传算法
```python
import random

class GeneticAlgorithm:
    """遗传算法"""
    def __init__(self, population_size, chromosome_length,
                 mutation_rate=0.01, crossover_rate=0.7):
        self.population_size = population_size
        self.chromosome_length = chromosome_length
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate

    def initialize_population(self):
        """初始化种群"""
        return [[random.randint(0, 1) for _ in range(self.chromosome_length)]
                for _ in range(self.population_size)]

    def fitness_function(self, chromosome):
        """适应度函数（示例：最大化1的数量）"""
        return sum(chromosome)

    def selection(self, population, fitness_scores):
        """选择操作"""
        # 轮盘赌选择
        total_fitness = sum(fitness_scores)
        if total_fitness == 0:
            return random.choice(population)

        pick = random.uniform(0, total_fitness)
        current = 0
        for individual, fitness in zip(population, fitness_scores):
            current += fitness
            if current > pick:
                return individual
        return population[-1]

    def crossover(self, parent1, parent2):
        """交叉操作"""
        if random.random() > self.crossover_rate:
            return parent1, parent2

        crossover_point = random.randint(1, self.chromosome_length - 1)
        child1 = parent1[:crossover_point] + parent2[crossover_point:]
        child2 = parent2[:crossover_point] + parent1[crossover_point:]
        return child1, child2

    def mutation(self, chromosome):
        """变异操作"""
        for i in range(len(chromosome)):
            if random.random() < self.mutation_rate:
                chromosome[i] = 1 - chromosome[i]
        return chromosome

    def evolve(self, generations):
        """进化过程"""
        population = self.initialize_population()

        for generation in range(generations):
            # 计算适应度
            fitness_scores = [self.fitness_function(individual)
                            for individual in population]

            # 找到最优个体
            best_fitness = max(fitness_scores)
            best_individual = population[fitness_scores.index(best_fitness)]

            print(f"Generation {generation}: Best Fitness = {best_fitness}")

            # 创建新一代
            new_population = []

            # 保留最优个体（精英主义）
            new_population.append(best_individual)

            while len(new_population) < self.population_size:
                # 选择
                parent1 = self.selection(population, fitness_scores)
                parent2 = self.selection(population, fitness_scores)

                # 交叉
                child1, child2 = self.crossover(parent1, parent2)

                # 变异
                child1 = self.mutation(child1)
                child2 = self.mutation(child2)

                new_population.extend([child1, child2])

            population = new_population[:self.population_size]

        return best_individual
```

## 算法优化技巧

### 1. 时间复杂度优化

#### 1.1 预计算和缓存
```python
from functools import lru_cache
import math

class PrecomputedValues:
    """预计算值优化"""
    def __init__(self, max_n):
        self.max_n = max_n
        self.factorials = self._precompute_factorials(max_n)
        self.primes = self._sieve_of_eratosthenes(max_n)

    def _precompute_factorials(self, n):
        """预计算阶乘"""
        factorials = [1] * (n + 1)
        for i in range(1, n + 1):
            factorials[i] = factorials[i-1] * i
        return factorials

    def _sieve_of_eratosthenes(self, n):
        """埃拉托斯特尼筛法"""
        sieve = [True] * (n + 1)
        sieve[0] = sieve[1] = False
        for i in range(2, int(math.sqrt(n)) + 1):
            if sieve[i]:
                sieve[i*i : n+1 : i] = [False] * len(sieve[i*i : n+1 : i])
        return [i for i, is_prime in enumerate(sieve) if is_prime]

    def factorial(self, n):
        """获取阶乘"""
        if n > self.max_n:
            return math.factorial(n)
        return self.factorials[n]

    def is_prime(self, n):
        """检查素数"""
        if n > self.max_n:
            return self._is_prime_naive(n)
        return n in self.primes

    def _is_prime_naive(self, n):
        """朴素素数检查"""
        if n < 2:
            return False
        for i in range(2, int(math.sqrt(n)) + 1):
            if n % i == 0:
                return False
        return True
```

#### 1.2 算法选择优化
```python
def optimized_sort(arr):
    """优化排序算法选择"""
    n = len(arr)

    if n <= 1:
        return arr

    # 小数组使用插入排序
    if n <= 20:
        return insertion_sort(arr)

    # 大数组使用快速排序
    if n <= 1000:
        return quick_sort(arr)

    # 超大数组使用归并排序
    return merge_sort(arr)

def insertion_sort(arr):
    """插入排序"""
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1
        arr[j + 1] = key
    return arr

def quick_sort(arr):
    """快速排序"""
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort(left) + middle + quick_sort(right)

def merge_sort(arr):
    """归并排序"""
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    """合并两个有序数组"""
    result = []
    i = j = 0

    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1

    result.extend(left[i:])
    result.extend(right[j:])
    return result
```

### 2. 空间复杂度优化

#### 2.1 原地算法
```python
def in_place_algorithm_optimization():
    """原地算法优化示例"""

    def reverse_array_in_place(arr):
        """原地反转数组"""
        left, right = 0, len(arr) - 1
        while left < right:
            arr[left], arr[right] = arr[right], arr[left]
            left += 1
            right -= 1
        return arr

    def matrix_transpose_in_place(matrix):
        """原地矩阵转置"""
        n = len(matrix)
        for i in range(n):
            for j in range(i + 1, n):
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        return matrix

    def remove_duplicates_in_place(nums):
        """原地删除重复元素"""
        if not nums:
            return 0

        write_index = 1
        for read_index in range(1, len(nums)):
            if nums[read_index] != nums[read_index - 1]:
                nums[write_index] = nums[read_index]
                write_index += 1

        return write_index
```

#### 2.2 位运算优化
```python
class BitwiseOptimization:
    """位运算优化"""
    def __init__(self):
        pass

    def is_power_of_two(self, n):
        """检查是否为2的幂"""
        return n > 0 and (n & (n - 1)) == 0

    def count_set_bits(self, n):
        """计算设置位的数量"""
        count = 0
        while n:
            count += 1
            n &= n - 1
        return count

    def swap_numbers(self, a, b):
        """不使用临时变量交换数字"""
        a = a ^ b
        b = a ^ b
        a = a ^ b
        return a, b

    def absolute_value(self, n):
        """绝对值（不使用分支）"""
        mask = n >> (32 - 1)  # 假设32位整数
        return (n + mask) ^ mask

    def min_max(self, a, b):
        """最小最大值（不使用分支）"""
        min_val = b + ((a - b) & ((a - b) >> (32 - 1)))
        max_val = a - ((a - b) & ((a - b) >> (32 - 1)))
        return min_val, max_val
```

### 3. 并行化优化

#### 3.1 多线程优化
```python
import concurrent.futures
import threading
import time

class ParallelOptimizer:
    """并行优化"""
    def __init__(self, max_workers=4):
        self.max_workers = max_workers

    def parallel_map(self, func, data):
        """并行map操作"""
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(func, data))
        return results

    def parallel_sum(self, numbers):
        """并行求和"""
        def chunk_sum(chunk):
            return sum(chunk)

        chunk_size = len(numbers) // self.max_workers
        chunks = [numbers[i:i + chunk_size] for i in range(0, len(numbers), chunk_size)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            chunk_sums = list(executor.map(chunk_sum, chunks))

        return sum(chunk_sums)

    def matrix_multiply_parallel(self, A, B):
        """并行矩阵乘法"""
        def multiply_element(i, j):
            return sum(A[i][k] * B[k][j] for k in range(len(B)))

        rows_A, cols_B = len(A), len(B[0])
        result = [[0] * cols_B for _ in range(rows_A)]

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(rows_A):
                for j in range(cols_B):
                    future = executor.submit(multiply_element, i, j)
                    futures.append(((i, j), future))

            for (i, j), future in futures:
                result[i][j] = future.result()

        return result
```

#### 3.2 缓存优化
```python
class CacheOptimizedAlgorithm:
    """缓存优化算法"""
    def __init__(self):
        self.block_size = 64  # 假设缓存行大小为64字节

    def cache_friendly_matrix_multiply(self, A, B):
        """缓存友好的矩阵乘法"""
        n = len(A)
        C = [[0] * n for _ in range(n)]

        # 分块优化
        for i in range(0, n, self.block_size):
            for j in range(0, n, self.block_size):
                for k in range(0, n, self.block_size):
                    # 处理块
                    for ii in range(i, min(i + self.block_size, n)):
                        for jj in range(j, min(j + self.block_size, n)):
                            for kk in range(k, min(k + self.block_size, n)):
                                C[ii][jj] += A[ii][kk] * B[kk][jj]

        return C

    def prefetch_optimized_search(self, arr, target):
        """预取优化的搜索"""
        n = len(arr)
        i = 0

        while i < n:
            # 预取下一个缓存行
            if i + self.block_size < n:
                _ = arr[i + self.block_size]  # 触发预取

            # 处理当前缓存行
            for j in range(i, min(i + self.block_size, n)):
                if arr[j] == target:
                    return j

            i += self.block_size

        return -1
```

### 4. 算法正确性验证

#### 4.1 测试框架
```python
import unittest
import random

class AlgorithmValidator:
    """算法验证器"""
    def __init__(self):
        self.test_cases = []

    def add_test_case(self, input_data, expected_output):
        """添加测试用例"""
        self.test_cases.append((input_data, expected_output))

    def validate_algorithm(self, algorithm, description=""):
        """验证算法正确性"""
        print(f"验证算法: {description}")
        passed = 0
        failed = 0

        for i, (input_data, expected) in enumerate(self.test_cases):
            try:
                result = algorithm(*input_data) if isinstance(input_data, tuple) else algorithm(input_data)
                if result == expected:
                    passed += 1
                    print(f"测试用例 {i+1}: 通过")
                else:
                    failed += 1
                    print(f"测试用例 {i+1}: 失败")
                    print(f"  输入: {input_data}")
                    print(f"  期望: {expected}")
                    print(f"  实际: {result}")
            except Exception as e:
                failed += 1
                print(f"测试用例 {i+1}: 异常 - {e}")

        print(f"通过: {passed}, 失败: {failed}")
        return failed == 0

    def generate_random_test_cases(self, generator, count=100):
        """生成随机测试用例"""
        for _ in range(count):
            input_data, expected_output = generator()
            self.add_test_case(input_data, expected_output)

    def stress_test(self, algorithm, max_size=1000):
        """压力测试"""
        print("开始压力测试...")
        sizes = [10, 100, 1000, max_size]

        for size in sizes:
            test_data = [random.randint(1, 1000) for _ in range(size)]
            start_time = time.time()
            try:
                result = algorithm(test_data)
                end_time = time.time()
                print(f"大小 {size}: {end_time - start_time:.6f}s")
            except Exception as e:
                print(f"大小 {size}: 异常 - {e}")

def create_algorithm_benchmark():
    """创建算法基准测试"""
    validator = AlgorithmValidator()

    # 快速排序测试
    def quick_sort_test():
        def is_sorted(arr):
            return all(arr[i] <= arr[i+1] for i in range(len(arr)-1))

        def generate_test_case():
            data = [random.randint(1, 1000) for _ in range(100)]
            return (data,), sorted(data)

        return generate_test_case

    validator.generate_random_test_cases(quick_sort_test(), 50)
    validator.validate_algorithm(quick_sort, "快速排序")

    return validator
```

## 应用场景

### 1. 机器学习
- **特征工程**: 数据预处理和特征选择
- **模型训练**: 梯度下降、优化算法
- **模型评估**: 交叉验证、性能指标

```python
class MachineLearningOptimizer:
    """机器学习算法优化器"""
    def __init__(self):
        pass

    def gradient_descent_optimized(self, X, y, learning_rate=0.01, epochs=1000):
        """优化的梯度下降算法"""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0

        for epoch in range(epochs):
            # 向量化计算
            predictions = np.dot(X, weights) + bias
            errors = predictions - y

            # 计算梯度
            dw = (1/n_samples) * np.dot(X.T, errors)
            db = (1/n_samples) * np.sum(errors)

            # 更新参数
            weights -= learning_rate * dw
            bias -= learning_rate * db

            # 早停
            if np.linalg.norm(dw) < 1e-6:
                break

        return weights, bias

    def stochastic_gradient_descent(self, X, y, learning_rate=0.01, epochs=100):
        """随机梯度下降"""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0

        for epoch in range(epochs):
            # 随机打乱数据
            indices = np.random.permutation(n_samples)

            for i in indices:
                prediction = np.dot(X[i], weights) + bias
                error = prediction - y[i]

                # 更新参数
                weights -= learning_rate * error * X[i]
                bias -= learning_rate * error

        return weights, bias

    def mini_batch_gradient_descent(self, X, y, learning_rate=0.01, epochs=100, batch_size=32):
        """小批量梯度下降"""
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0

        for epoch in range(epochs):
            # 创建小批量
            indices = np.random.permutation(n_samples)

            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]

                predictions = np.dot(X_batch, weights) + bias
                errors = predictions - y_batch

                dw = (1/len(batch_indices)) * np.dot(X_batch.T, errors)
                db = (1/len(batch_indices)) * np.sum(errors)

                weights -= learning_rate * dw
                bias -= learning_rate * db

        return weights, bias
```

### 2. 图像处理
- **滤波算法**: 高斯滤波、中值滤波
- **边缘检测**: Sobel、Canny边缘检测
- **图像分割**: 区域生长、分水岭算法

```python
class ImageProcessingAlgorithms:
    """图像处理算法"""
    def __init__(self):
        pass

    def gaussian_filter_optimized(self, image, sigma=1.0):
        """优化的高斯滤波"""
        # 生成高斯核
        kernel_size = int(6 * sigma) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1

        def gaussian_kernel(size, sigma):
            kernel = np.zeros((size, size))
            center = size // 2

            for i in range(size):
                for j in range(size):
                    x, y = i - center, j - center
                    kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            kernel /= 2 * np.pi * sigma**2
            kernel /= np.sum(kernel)
            return kernel

        kernel = gaussian_kernel(kernel_size, sigma)

        # 应用滤波
        return self._apply_convolution(image, kernel)

    def _apply_convolution(self, image, kernel):
        """应用卷积（优化版）"""
        height, width = image.shape
        k_height, k_width = kernel.shape
        output = np.zeros_like(image)

        # 分离卷积优化
        kernel_1d = np.sum(kernel, axis=0)
        kernel_1d = kernel_1d / np.sum(kernel_1d)

        # 水平卷积
        temp = np.zeros_like(image)
        for i in range(height):
            for j in range(width):
                for k in range(k_width):
                    if 0 <= j - k//2 < width:
                        temp[i, j] += image[i, j - k//2] * kernel_1d[k]

        # 垂直卷积
        kernel_1d = np.sum(kernel, axis=1)
        kernel_1d = kernel_1d / np.sum(kernel_1d)

        for i in range(height):
            for j in range(width):
                for k in range(k_height):
                    if 0 <= i - k//2 < height:
                        output[i, j] += temp[i - k//2, j] * kernel_1d[k]

        return output

    def canny_edge_detection(self, image, low_threshold=50, high_threshold=150):
        """Canny边缘检测"""
        # 1. 高斯滤波
        blurred = self.gaussian_filter_optimized(image)

        # 2. 计算梯度
        grad_x = np.gradient(blurred, axis=1)
        grad_y = np.gradient(blurred, axis=0)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)
        angle = np.arctan2(grad_y, grad_x)

        # 3. 非极大值抑制
        suppressed = self._non_max_suppression(magnitude, angle)

        # 4. 双阈值检测
        edges = self._double_threshold(suppressed, low_threshold, high_threshold)

        return edges

    def _non_max_suppression(self, magnitude, angle):
        """非极大值抑制"""
        height, width = magnitude.shape
        suppressed = np.zeros_like(magnitude)

        # 将角度转换为0°, 45°, 90°, 135°
        angle_deg = np.degrees(angle) % 180
        angle_deg[angle_deg < 22.5] = 0
        angle_deg[angle_deg >= 157.5] = 0
        angle_deg[(angle_deg >= 22.5) & (angle_deg < 67.5)] = 45
        angle_deg[(angle_deg >= 67.5) & (angle_deg < 112.5)] = 90
        angle_deg[(angle_deg >= 112.5) & (angle_deg < 157.5)] = 135

        for i in range(1, height - 1):
            for j in range(1, width - 1):
                current_angle = angle_deg[i, j]
                current_magnitude = magnitude[i, j]

                if current_angle == 0:
                    neighbors = [magnitude[i, j-1], magnitude[i, j+1]]
                elif current_angle == 45:
                    neighbors = [magnitude[i-1, j+1], magnitude[i+1, j-1]]
                elif current_angle == 90:
                    neighbors = [magnitude[i-1, j], magnitude[i+1, j]]
                else:  # 135
                    neighbors = [magnitude[i-1, j-1], magnitude[i+1, j+1]

                if current_magnitude >= max(neighbors):
                    suppressed[i, j] = current_magnitude

        return suppressed

    def _double_threshold(self, image, low_threshold, high_threshold):
        """双阈值检测"""
        strong = 255
        weak = 50

        strong_edges = (image >= high_threshold)
        weak_edges = (image >= low_threshold) & (image < high_threshold)

        # 连接边缘
        result = np.zeros_like(image)
        result[strong_edges] = strong
        result[weak_edges] = weak

        # 连接弱边缘到强边缘
        for i in range(1, result.shape[0] - 1):
            for j in range(1, result.shape[1] - 1):
                if result[i, j] == weak:
                    if (result[i-1:i+2, j-1:j+2] == strong).any():
                        result[i, j] = strong
                    else:
                        result[i, j] = 0

        return result
```

### 3. 自然语言处理
- **文本预处理**: 分词、词干提取
- **特征提取**: TF-IDF、Word2Vec
- **序列建模**: RNN、Transformer

```python
class TextProcessingAlgorithms:
    """文本处理算法"""
    def __init__(self):
        self.stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}

    def tokenize_optimized(self, text):
        """优化的分词算法"""
        # 使用正则表达式优化
        import re
        tokens = re.findall(r'\b\w+\b', text.lower())
        return [token for token in tokens if token not in self.stop_words]

    def calculate_tf_idf(self, documents):
        """计算TF-IDF"""
        from collections import defaultdict
        import math

        # 计算词频
        tf = []
        df = defaultdict(int)

        for doc in documents:
            term_freq = defaultdict(int)
            tokens = self.tokenize_optimized(doc)

            for token in tokens:
                term_freq[token] += 1

            tf.append(term_freq)

            # 更新文档频率
            for token in set(tokens):
                df[token] += 1

        # 计算IDF
        idf = {}
        num_docs = len(documents)
        for term, freq in df.items():
            idf[term] = math.log(num_docs / freq)

        # 计算TF-IDF
        tf_idf = []
        for doc_tf in tf:
            doc_tf_idf = {}
            for term, freq in doc_tf.items():
                doc_tf_idf[term] = freq * idf[term]
            tf_idf.append(doc_tf_idf)

        return tf_idf

    def levenshtein_distance_optimized(self, s1, s2):
        """优化的编辑距离算法"""
        if len(s1) < len(s2):
            return self.levenshtein_distance_optimized(s2, s1)

        if len(s2) == 0:
            return len(s1)

        # 使用两行而不是整个矩阵
        previous_row = list(range(len(s2) + 1))
        current_row = [0] * (len(s2) + 1)

        for i, c1 in enumerate(s1):
            current_row[0] = i + 1
            for j, c2 in enumerate(s2):
                # 计算编辑距离
                if c1 == c2:
                    current_row[j + 1] = previous_row[j]
                else:
                    current_row[j + 1] = min(
                        previous_row[j + 1] + 1,      # 删除
                        current_row[j] + 1,          # 插入
                        previous_row[j] + 1           # 替换
                    )

            previous_row = current_row[:]

        return previous_row[-1]
```

### 4. 游戏开发
- **路径查找**: A*、Dijkstra算法
- **碰撞检测**: 空间划分、包围盒
- **AI行为**: 有限状态机、行为树

```python
class GameAlgorithms:
    """游戏算法"""
    def __init__(self):
        pass

    def a_star_with_heuristics(self, grid, start, goal, heuristic='manhattan'):
        """A*算法与启发式"""
        def manhattan_distance(p1, p2):
            return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

        def euclidean_distance(p1, p2):
            return math.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        def diagonal_distance(p1, p2):
            return max(abs(p1[0] - p2[0]), abs(p1[1] - p2[1]))

        heuristics = {
            'manhattan': manhattan_distance,
            'euclidean': euclidean_distance,
            'diagonal': diagonal_distance
        }

        h_func = heuristics.get(heuristic, manhattan_distance)

        return a_star_search(grid, start, goal, h_func)

    def jump_point_search(self, grid, start, goal):
        """跳点搜索算法"""
        def identify_successors(current):
            successors = []
            neighbors = self._get_neighbors(current, grid)

            for neighbor in neighbors:
                direction = (neighbor[0] - current[0], neighbor[1] - current[1])

                # 检查是否为强制邻居
                if self._is_forced_neighbor(current, neighbor, direction, grid):
                    successors.append(neighbor)
                else:
                    # 跳点搜索
                    jump_point = self._jump(current, direction, grid)
                    if jump_point:
                        successors.append(jump_point)

            return successors

        def _jump(current, direction, grid):
            """跳跃搜索"""
            next_pos = (current[0] + direction[0], current[1] + direction[1])

            # 检查边界和障碍物
            if (not self._is_valid_position(next_pos, grid) or
                grid[next_pos[1]][next_pos[0]] == 1):
                return None

            # 如果到达目标
            if next_pos == goal:
                return next_pos

            # 检查强制邻居
            if self._has_forced_neighbors(next_pos, direction, grid):
                return next_pos

            # 继续跳跃
            return _jump(next_pos, direction, grid)

        # 实现细节...
        return []

    def _get_neighbors(self, pos, grid):
        """获取邻居位置"""
        neighbors = []
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            new_x, new_y = pos[0] + dx, pos[1] + dy
            if self._is_valid_position((new_x, new_y), grid):
                neighbors.append((new_x, new_y))
        return neighbors

    def _is_valid_position(self, pos, grid):
        """检查位置是否有效"""
        x, y = pos
        return (0 <= x < len(grid[0]) and 0 <= y < len(grid[0]))

    def _is_forced_neighbor(self, current, neighbor, direction, grid):
        """检查是否为强制邻居"""
        # 实现强制邻居检查逻辑
        return False

    def _has_forced_neighbors(self, pos, direction, grid):
        """检查是否有强制邻居"""
        # 实现强制邻居检查逻辑
        return False
```

## 练习

1. 实现一个支持多种启发式函数的A*算法
2. 创建一个并行化的快速排序算法
3. 实现一个自适应的遗传算法
4. 设计一个缓存友好的矩阵乘法算法
5. 实现一个支持大规模数据的外部排序算法
6. 创建一个基于分支限界的作业调度算法
7. 实现一个支持在线学习的梯度下降算法
8. 设计一个多目标优化的帕累托前沿算法