# L01 - 算法复杂度与渐进分析

## 学习目标
- 理解渐进符号 (O, Ω, Θ) 的含义和使用
- 掌握算法复杂度分析方法
- 学会使用主定理分析递归算法
- 能够对算法进行时间和空间复杂度分析

## 渐进符号

### 大O符号 (Upper Bound)
**定义**: f(n) = O(g(n)) 如果存在正数c和n₀，使得对所有n ≥ n₀，有 f(n) ≤ c·g(n)

**常见复杂度**:
- O(1) - 常数时间
- O(log n) - 对数时间
- O(n) - 线性时间
- O(n log n) - 线性对数时间
- O(n²) - 平方时间
- O(2ⁿ) - 指数时间

### Omega符号 (Lower Bound)
**定义**: f(n) = Ω(g(n)) 如果存在正数c和n₀，使得对所有n ≥ n₀，有 f(n) ≥ c·g(n)

### Theta符号 (Tight Bound)
**定义**: f(n) = Θ(g(n)) 如果 f(n) = O(g(n)) 且 f(n) = Ω(g(n))

## Python实现示例

```python
import time
import random
import matplotlib.pyplot as plt
from typing import List, Callable

def measure_time(func: Callable, n: int) -> float:
    """测量函数执行时间"""
    arr = list(range(n))
    random.shuffle(arr)

    start_time = time.time()
    func(arr)
    end_time = time.time()

    return end_time - start_time

def plot_complexity(funcs: List[Callable], names: List[str], max_n: int = 10000):
    """绘制复杂度对比图"""
    n_values = list(range(100, max_n, 500))

    plt.figure(figsize=(12, 8))

    for func, name in zip(funcs, names):
        times = [measure_time(func, n) for n in n_values]
        plt.plot(n_values, times, label=name, marker='o')

    plt.xlabel('输入大小 n')
    plt.ylabel('执行时间 (秒)')
    plt.title('算法复杂度对比')
    plt.legend()
    plt.grid(True)
    plt.show()
```

## 常见算法复杂度分析

### 1. 线性搜索 - O(n)
```python
def linear_search(arr: List[int], target: int) -> bool:
    """
    线性搜索算法
    时间复杂度: O(n)
    空间复杂度: O(1)
    """
    for element in arr:
        if element == target:
            return True
    return False
```

### 2. 二分搜索 - O(log n)
```python
def binary_search(arr: List[int], target: int) -> bool:
    """
    二分搜索算法
    时间复杂度: O(log n)
    空间复杂度: O(1)
    前提: 数组已排序
    """
    left, right = 0, len(arr) - 1

    while left <= right:
        mid = (left + right) // 2
        if arr[mid] == target:
            return True
        elif arr[mid] < target:
            left = mid + 1
        else:
            right = mid - 1

    return False
```

### 3. 冒泡排序 - O(n²)
```python
def bubble_sort(arr: List[int]) -> List[int]:
    """
    冒泡排序算法
    时间复杂度: O(n²)
    空间复杂度: O(1)
    """
    n = len(arr)
    for i in range(n):
        for j in range(0, n - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
    return arr
```

## 主定理 (Master Theorem)

**主定理用于分析分治算法的递归关系式**: T(n) = aT(n/b) + f(n)

**三种情况**:
1. 如果 f(n) = O(n^log_b(a-ε))，则 T(n) = Θ(n^log_b(a))
2. 如果 f(n) = Θ(n^log_b(a))，则 T(n) = Θ(n^log_b(a) · log n)
3. 如果 f(n) = Ω(n^log_b(a+ε)) 且 af(n/b) ≤ cf(n)，则 T(n) = Θ(f(n))

### 示例：归并排序
T(n) = 2T(n/2) + O(n)
这里 a=2, b=2, f(n)=O(n)
log_b(a) = log₂(2) = 1
f(n) = Θ(n^1)，符合第二种情况
因此 T(n) = Θ(n log n)

```python
def merge_sort(arr: List[int]) -> List[int]:
    """
    归并排序算法
    递归关系: T(n) = 2T(n/2) + O(n)
    时间复杂度: Θ(n log n)
    空间复杂度: O(n)
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left: List[int], right: List[int]) -> List[int]:
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

## 复杂度分析技巧

### 1. 循环分析
```python
def analyze_loops(n: int):
    # O(n) - 单层循环
    for i in range(n):
        print(i)

    # O(n²) - 嵌套循环
    for i in range(n):
        for j in range(n):
            print(i, j)

    # O(n log n) - 外层n，内层log n
    for i in range(n):
        j = 1
        while j < n:
            j *= 2
            print(i, j)
```

### 2. 递归分析
```python
def fibonacci_recursive(n: int) -> int:
    """
    递归斐波那契数列
    递归关系: T(n) = T(n-1) + T(n-2) + O(1)
    时间复杂度: O(2ⁿ)
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_iterative(n: int) -> int:
    """
    迭代斐波那契数列
    时间复杂度: O(n)
    空间复杂度: O(1)
    """
    if n <= 1:
        return n

    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b

    return b
```

## 练习

1. 分析以下算法的时间复杂度：
```python
def mystery_function(n: int):
    count = 0
    for i in range(n):
        for j in range(i, n):
            count += 1
    return count
```

2. 使用主定理分析以下递归关系：
   - T(n) = 3T(n/2) + n²
   - T(n) = 4T(n/2) + n
   - T(n) = T(n/2) + 1

3. 实现一个算法并分析其复杂度：
   - 找出数组中第k大的元素
   - 检查字符串是否为回文
   - 计算数组的所有子集

## 应用场景

### 数据库索引设计
- B树索引：O(log n) 查找复杂度
- 哈希索引：O(1) 平均查找复杂度

### 网络路由算法
- Dijkstra算法：O(V²) 或 O(E + V log V)
- 路由表查找：O(1) 使用哈希表

### 机器学习算法
- K-means聚类：O(n·k·d·i)
- 决策树训练：O(n·d·log n)

通过理解算法复杂度，我们能够：
1. 预测算法在大数据集上的性能
2. 在多种算法方案中做出最优选择
3. 识别和优化代码中的性能瓶颈
4. 设计可扩展的系统架构