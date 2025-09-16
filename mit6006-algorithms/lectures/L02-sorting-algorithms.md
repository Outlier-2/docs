# L02 - 排序算法

## 学习目标
- 掌握常见排序算法的原理和实现
- 理解不同排序算法的时空复杂度
- 学会根据场景选择合适的排序算法
- 实现稳定的排序算法

## 排序算法分类

### 比较排序
- **插入排序**：适合小规模或基本有序数据
- **归并排序**：稳定，时间复杂度Θ(n log n)
- **快速排序**：平均性能最好，但最坏情况O(n²)
- **堆排序**：保证O(n log n)，原地排序

### 非比较排序
- **计数排序**：适合小范围整数
- **基数排序**：适合固定长度字符串或数字
- **桶排序**：适合均匀分布的数据

## Python实现

### 1. 插入排序 (Insertion Sort)
```python
def insertion_sort(arr: list) -> list:
    """
    插入排序算法
    时间复杂度: O(n²) 最坏和平均情况, O(n) 最好情况
    空间复杂度: O(1) 原地排序
    稳定性: 稳定
    适用场景: 小规模数据或基本有序的数据
    """
    for i in range(1, len(arr)):
        key = arr[i]
        j = i - 1

        # 将大于key的元素向后移动
        while j >= 0 and arr[j] > key:
            arr[j + 1] = arr[j]
            j -= 1

        arr[j + 1] = key

    return arr
```

### 2. 归并排序 (Merge Sort)
```python
def merge_sort(arr: list) -> list:
    """
    归并排序算法
    时间复杂度: Θ(n log n) 所有情况
    空间复杂度: O(n) 需要额外空间
    稳定性: 稳定
    适用场景: 大规模数据，外部排序
    """
    if len(arr) <= 1:
        return arr

    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])

    return merge(left, right)

def merge(left: list, right: list) -> list:
    """合并两个已排序的数组"""
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

### 3. 快速排序 (Quick Sort)
```python
def quick_sort(arr: list) -> list:
    """
    快速排序算法
    时间复杂度: O(n log n) 平均情况, O(n²) 最坏情况
    空间复杂度: O(log n) 递归调用栈
    稳定性: 不稳定
    适用场景: 平均性能最好，内存排序
    """
    if len(arr) <= 1:
        return arr

    # 随机选择pivot以避免最坏情况
    pivot_idx = random.randint(0, len(arr) - 1)
    pivot = arr[pivot_idx]

    # 三路划分
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]

    return quick_sort(left) + middle + quick_sort(right)

def quick_sort_inplace(arr: list, low: int = 0, high: int = None) -> None:
    """
    原地快速排序
    空间复杂度: O(1) 不考虑递归栈
    """
    if high is None:
        high = len(arr) - 1

    def partition(arr: list, low: int, high: int) -> int:
        """Lomuto划分方案"""
        pivot = arr[high]
        i = low - 1

        for j in range(low, high):
            if arr[j] <= pivot:
                i += 1
                arr[i], arr[j] = arr[j], arr[i]

        arr[i + 1], arr[high] = arr[high], arr[i + 1]
        return i + 1

    if low < high:
        pi = partition(arr, low, high)
        quick_sort_inplace(arr, low, pi - 1)
        quick_sort_inplace(arr, pi + 1, high)
```

### 4. 堆排序 (Heap Sort)
```python
def heap_sort(arr: list) -> list:
    """
    堆排序算法
    时间复杂度: O(n log n) 所有情况
    空间复杂度: O(1) 原地排序
    稳定性: 不稳定
    适用场景: 需要保证最坏情况的性能
    """
    def heapify(arr: list, n: int, i: int):
        """维护堆性质"""
        largest = i
        left = 2 * i + 1
        right = 2 * i + 2

        if left < n and arr[left] > arr[largest]:
            largest = left

        if right < n and arr[right] > arr[largest]:
            largest = right

        if largest != i:
            arr[i], arr[largest] = arr[largest], arr[i]
            heapify(arr, n, largest)

    n = len(arr)

    # 构建最大堆
    for i in range(n // 2 - 1, -1, -1):
        heapify(arr, n, i)

    # 逐个提取元素
    for i in range(n - 1, 0, -1):
        arr[0], arr[i] = arr[i], arr[0]
        heapify(arr, i, 0)

    return arr
```

### 5. 计数排序 (Counting Sort)
```python
def counting_sort(arr: list, max_val: int = None) -> list:
    """
    计数排序算法
    时间复杂度: O(n + k) k是数据范围
    空间复杂度: O(n + k)
    稳定性: 稳定
    适用场景: 小范围整数排序
    """
    if not arr:
        return arr

    if max_val is None:
        max_val = max(arr)

    count = [0] * (max_val + 1)
    output = [0] * len(arr)

    # 统计每个元素的出现次数
    for num in arr:
        count[num] += 1

    # 计算累积位置
    for i in range(1, len(count)):
        count[i] += count[i - 1]

    # 构建输出数组
    for num in reversed(arr):
        output[count[num] - 1] = num
        count[num] -= 1

    return output
```

### 6. 基数排序 (Radix Sort)
```python
def radix_sort(arr: list) -> list:
    """
    基数排序算法
    时间复杂度: O(d(n + k)) d是数字位数，k是基数
    空间复杂度: O(n + k)
    稳定性: 稳定
    适用场景: 固定长度的数字或字符串
    """
    if not arr:
        return arr

    # 找到最大值确定位数
    max_val = max(abs(x) for x in arr)
    max_digits = len(str(max_val))

    # 处理负数：将数组分为负数和非负数
    negatives = [x for x in arr if x < 0]
    positives = [x for x in arr if x >= 0]

    # 对负数取绝对值排序后反转
    if negatives:
        negatives = [-x for x in negatives]
        for digit in range(max_digits):
            negatives = counting_sort_by_digit(negatives, digit)
        negatives = [-x for x in reversed(negatives)]

    # 对正数排序
    for digit in range(max_digits):
        positives = counting_sort_by_digit(positives, digit)

    return negatives + positives

def counting_sort_by_digit(arr: list, digit: int) -> list:
    """按特定位数进行计数排序"""
    count = [0] * 10
    output = [0] * len(arr)

    for num in arr:
        d = (num // (10 ** digit)) % 10
        count[d] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    for num in reversed(arr):
        d = (num // (10 ** digit)) % 10
        output[count[d] - 1] = num
        count[d] -= 1

    return output
```

## 性能测试与可视化

```python
import time
import random
import matplotlib.pyplot as plt

def test_sorting_algorithms():
    """测试各种排序算法的性能"""
    algorithms = {
        'Insertion Sort': insertion_sort,
        'Merge Sort': merge_sort,
        'Quick Sort': quick_sort,
        'Heap Sort': heap_sort,
        'Counting Sort': lambda x: counting_sort(x, max(x) if x else 0),
        'Radix Sort': radix_sort,
        'Built-in Sort': sorted
    }

    sizes = [100, 500, 1000, 2000, 5000]
    results = {name: [] for name in algorithms}

    for size in sizes:
        # 生成随机数组
        arr = [random.randint(1, 1000) for _ in range(size)]

        for name, func in algorithms.items():
            # 复制数组以避免修改原数组
            arr_copy = arr.copy()

            start_time = time.time()
            func(arr_copy)
            end_time = time.time()

            results[name].append(end_time - start_time)

    # 绘制性能对比图
    plt.figure(figsize=(12, 8))
    for name, times in results.items():
        plt.plot(sizes, times, marker='o', label=name)

    plt.xlabel('数组大小')
    plt.ylabel('执行时间 (秒)')
    plt.title('排序算法性能对比')
    plt.legend()
    plt.grid(True)
    plt.xscale('log')
    plt.yscale('log')
    plt.show()
```

## 算法选择指南

### 根据数据规模选择
- **小规模数据 (n < 50)**: 插入排序
- **中等规模数据 (50 < n < 1000)**: 快速排序
- **大规模数据 (n > 1000)**: 归并排序、堆排序

### 根据数据特征选择
- **基本有序**: 插入排序、冒泡排序
- **随机分布**: 快速排序
- **大量重复**: 三路快速排序
- **小范围整数**: 计数排序、基数排序

### 根据应用需求选择
- **需要稳定性**: 归并排序、插入排序
- **内存受限**: 堆排序、原地快速排序
- **外部排序**: 归并排序
- **最坏情况保证**: 堆排序

## 应用场景

### 数据库系统
- **索引构建**: 使用快速排序或归并排序
- **结果排序**: 根据数据量和内存选择合适算法

### 大数据处理
- **外部排序**: 归并排序处理无法装入内存的数据
- **分布式排序**: 结合MapReduce的排序框架

### 实时系统
- **流数据排序**: 插入排序维护已排序流
- **优先级队列**: 堆排序实现优先级队列

### 游戏开发
- **排行榜排序**: 快速排序或归并排序
- **碰撞检测**: 空间划分排序提高检测效率

## 优化技巧

### 1. 混合排序算法
```python
def hybrid_sort(arr: list, threshold: int = 50) -> list:
    """混合排序：小数组用插入排序，大数组用快速排序"""
    if len(arr) <= threshold:
        return insertion_sort(arr)
    else:
        return quick_sort(arr)
```

### 2. 并行排序
```python
import concurrent.futures

def parallel_merge_sort(arr: list) -> list:
    """并行归并排序"""
    if len(arr) <= 1:
        return arr

    if len(arr) <= 1000:  # 小数组不并行
        return merge_sort(arr)

    mid = len(arr) // 2

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        left_future = executor.submit(merge_sort, arr[:mid])
        right_future = executor.submit(merge_sort, arr[mid:])

        left = left_future.result()
        right = right_future.result()

    return merge(left, right)
```

### 3. 缓存友好优化
```python
def cache_friendly_merge_sort(arr: list) -> list:
    """缓存友好的归并排序"""
    # 使用小块大小来优化缓存使用
    block_size = 1024  # 根据CPU缓存大小调整

    if len(arr) <= block_size:
        return insertion_sort(arr)

    # 先对小块排序，然后合并
    blocks = []
    for i in range(0, len(arr), block_size):
        block = arr[i:i + block_size]
        blocks.append(insertion_sort(block))

    # 逐步合并块
    while len(blocks) > 1:
        new_blocks = []
        for i in range(0, len(blocks), 2):
            if i + 1 < len(blocks):
                new_blocks.append(merge(blocks[i], blocks[i + 1]))
            else:
                new_blocks.append(blocks[i])
        blocks = new_blocks

    return blocks[0] if blocks else []
```

## 练习

1. 实现一个稳定的快速排序算法
2. 比较不同排序算法在特定数据分布上的性能
3. 实现一个支持自定义比较函数的通用排序接口
4. 优化归并排序以减少内存使用
5. 实现双向冒泡排序（鸡尾酒排序）算法