# L11 - 高级数据结构

## 学习目标
- 掌握高级数据结构的设计原理
- 理解各种数据结构的时空复杂度
- 学会根据应用场景选择合适的数据结构
- 能够实现复杂的数据结构操作

## 高级数据结构概述

### 设计原则
- **时间复杂度权衡**: 在时间和空间之间找到平衡
- **操作优化**: 针对特定操作进行优化
- **数据局部性**: 提高缓存命中率
- **并发友好**: 支持多线程访问

### 应用场景
- **数据库索引**: B树、LSM树
- **内存管理**: 跳表、布隆过滤器
- **实时系统**: 线段树、树状数组
- **分布式系统**: 一致性哈希、分布式哈希表

## Python实现

### 1. 并查集 (Union-Find)
```python
class UnionFind:
    """
    并查集数据结构
    时间复杂度:
    - 查找: O(α(n)) 近似常数
    - 合并: O(α(n)) 近似常数
    - 连通: O(α(n)) 近似常数
    空间复杂度: O(n)
    """

    def __init__(self, elements=None):
        self.parent = {}
        self.rank = {}
        self.size = {}

        if elements:
            for element in elements:
                self.parent[element] = element
                self.rank[element] = 0
                self.size[element] = 1

    def find(self, element):
        """查找根节点（路径压缩）"""
        if self.parent[element] != element:
            self.parent[element] = self.find(self.parent[element])
        return self.parent[element]

    def union(self, element1, element2):
        """合并两个集合（按秩合并）"""
        root1 = self.find(element1)
        root2 = self.find(element2)

        if root1 == root2:
            return False

        if self.rank[root1] < self.rank[root2]:
            self.parent[root1] = root2
            self.size[root2] += self.size[root1]
        elif self.rank[root1] > self.rank[root2]:
            self.parent[root2] = root1
            self.size[root1] += self.size[root2]
        else:
            self.parent[root2] = root1
            self.rank[root1] += 1
            self.size[root1] += self.size[root2]

        return True

    def connected(self, element1, element2):
        """检查两个元素是否连通"""
        return self.find(element1) == self.find(element2)

    def get_size(self, element):
        """获取元素所在集合的大小"""
        root = self.find(element)
        return self.size[root]

    def get_components(self):
        """获取所有连通分量"""
        components = {}
        for element in self.parent:
            root = self.find(element)
            if root not in components:
                components[root] = []
            components[root].append(element)
        return components

    def add_element(self, element):
        """添加新元素"""
        if element not in self.parent:
            self.parent[element] = element
            self.rank[element] = 0
            self.size[element] = 1

    def __str__(self):
        components = self.get_components()
        result = []
        for root, elements in components.items():
            result.append(f"{root}: {elements}")
        return "\n".join(result)
```

### 2. 线段树 (Segment Tree)
```python
class SegmentTree:
    """
    线段树数据结构
    支持区间查询和区间更新
    时间复杂度:
    - 构建: O(n)
    - 查询: O(log n)
    - 更新: O(log n)
    空间复杂度: O(n)
    """

    def __init__(self, data, func=min, default=float('inf')):
        """
        初始化线段树
        func: 区间合并函数（min, max, sum等）
        default: 默认值
        """
        self.n = len(data)
        self.func = func
        self.default = default
        self.size = 1
        while self.size < self.n:
            self.size *= 2

        self.tree = [self.default] * (2 * self.size)

        # 构建线段树
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.func(self.tree[2*i], self.tree[2*i+1])

    def update(self, index, value):
        """更新单个元素"""
        index += self.size
        self.tree[index] = value
        index //= 2

        while index >= 1:
            self.tree[index] = self.func(self.tree[2*index], self.tree[2*index+1])
            index //= 2

    def query(self, left, right):
        """查询区间[left, right]"""
        left += self.size
        right += self.size

        result = self.default

        while left <= right:
            if left % 2 == 1:
                result = self.func(result, self.tree[left])
                left += 1
            if right % 2 == 0:
                result = self.func(result, self.tree[right])
                right -= 1
            left //= 2
            right //= 2

        return result

    def range_update(self, left, right, value):
        """区间更新（需要配合Lazy Propagation）"""
        self._range_update(1, 0, self.size - 1, left, right, value)

    def _range_update(self, node, node_left, node_right, update_left, update_right, value):
        """区间更新辅助函数"""
        if update_right < node_left or node_right < update_left:
            return

        if update_left <= node_left and node_right <= update_right:
            self.tree[node] = value
            return

        mid = (node_left + node_right) // 2
        self._range_update(2*node, node_left, mid, update_left, update_right, value)
        self._range_update(2*node+1, mid+1, node_right, update_left, update_right, value)
        self.tree[node] = self.func(self.tree[2*node], self.tree[2*node+1])

class LazySegmentTree:
    """
    带延迟标记的线段树
    支持高效的区间更新
    """

    def __init__(self, data, func=sum, lazy_func=lambda x, y: x + y, default=0):
        self.n = len(data)
        self.func = func
        self.lazy_func = lazy_func
        self.default = default
        self.size = 1
        while self.size < self.n:
            self.size *= 2

        self.tree = [default] * (2 * self.size)
        self.lazy = [0] * (2 * self.size)

        # 构建线段树
        for i in range(self.n):
            self.tree[self.size + i] = data[i]
        for i in range(self.size - 1, 0, -1):
            self.tree[i] = self.func(self.tree[2*i], self.tree[2*i+1])

    def _push(self, node, node_left, node_right):
        """下推延迟标记"""
        if self.lazy[node] != 0:
            self.tree[node] = self.lazy_func(self.tree[node], self.lazy[node])

            if node_left != node_right:
                self.lazy[2*node] = self.lazy_func(self.lazy[2*node], self.lazy[node])
                self.lazy[2*node+1] = self.lazy_func(self.lazy[2*node+1], self.lazy[node])

            self.lazy[node] = 0

    def _update_range(self, node, node_left, node_right, update_left, update_right, value):
        """区间更新"""
        self._push(node, node_left, node_right)

        if update_right < node_left or node_right < update_left:
            return

        if update_left <= node_left and node_right <= update_right:
            self.lazy[node] = value
            self._push(node, node_left, node_right)
            return

        mid = (node_left + node_right) // 2
        self._update_range(2*node, node_left, mid, update_left, update_right, value)
        self._update_range(2*node+1, mid+1, node_right, update_left, update_right, value)
        self.tree[node] = self.func(self.tree[2*node], self.tree[2*node+1])

    def _query_range(self, node, node_left, node_right, query_left, query_right):
        """区间查询"""
        self._push(node, node_left, node_right)

        if query_right < node_left or node_right < query_left:
            return self.default

        if query_left <= node_left and node_right <= query_right:
            return self.tree[node]

        mid = (node_left + node_right) // 2
        left_result = self._query_range(2*node, node_left, mid, query_left, query_right)
        right_result = self._query_range(2*node+1, mid+1, node_right, query_left, query_right)
        return self.func(left_result, right_result)

    def range_update(self, left, right, value):
        """区间更新"""
        self._update_range(1, 0, self.size - 1, left, right, value)

    def range_query(self, left, right):
        """区间查询"""
        return self._query_range(1, 0, self.size - 1, left, right)
```

### 3. 树状数组 (Binary Indexed Tree)
```python
class FenwickTree:
    """
    树状数组（Binary Indexed Tree）
    支持单点更新和前缀查询
    时间复杂度:
    - 更新: O(log n)
    - 查询: O(log n)
    空间复杂度: O(n)
    """

    def __init__(self, size):
        self.size = size
        self.tree = [0] * (size + 1)

    def update(self, index, delta):
        """更新index位置的值增加delta"""
        while index <= self.size:
            self.tree[index] += delta
            index += index & -index

    def query(self, index):
        """查询前缀和[1, index]"""
        result = 0
        while index > 0:
            result += self.tree[index]
            index -= index & -index
        return result

    def range_query(self, left, right):
        """查询区间和[left, right]"""
        return self.query(right) - self.query(left - 1)

class FenwickTree2D:
    """
    二维树状数组
    支持二维区间查询和更新
    """

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.tree = [[0] * (cols + 1) for _ in range(rows + 1)]

    def update(self, row, col, delta):
        """更新(row, col)位置的值"""
        i = row
        while i <= self.rows:
            j = col
            while j <= self.cols:
                self.tree[i][j] += delta
                j += j & -j
            i += i & -i

    def query(self, row, col):
        """查询[1, row] × [1, col]的和"""
        result = 0
        i = row
        while i > 0:
            j = col
            while j > 0:
                result += self.tree[i][j]
                j -= j & -j
            i -= i & -i
        return result

    def range_query(self, row1, col1, row2, col2):
        """查询[row1, row2] × [col1, col2]的和"""
        return (self.query(row2, col2) - self.query(row1-1, col2) -
                self.query(row2, col1-1) + self.query(row1-1, col1-1))
```

### 4. 跳表 (Skip List)
```python
import random

class SkipListNode:
    """跳表节点"""
    def __init__(self, value, level):
        self.value = value
        self.forward = [None] * (level + 1)

class SkipList:
    """
    跳表数据结构
    时间复杂度:
    - 查找: O(log n) 平均
    - 插入: O(log n) 平均
    - 删除: O(log n) 平均
    空间复杂度: O(n log n)
    """

    def __init__(self, max_level=16, p=0.5):
        self.max_level = max_level
        self.p = p
        self.level = 0
        self.head = SkipListNode(None, max_level)

    def _random_level(self):
        """随机生成层数"""
        level = 0
        while random.random() < self.p and level < self.max_level:
            level += 1
        return level

    def search(self, value):
        """搜索值"""
        current = self.head

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]

        current = current.forward[0]
        return current and current.value == value

    def insert(self, value):
        """插入值"""
        update = [None] * (self.max_level + 1)
        current = self.head

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if not current or current.value != value:
            new_level = self._random_level()

            if new_level > self.level:
                for i in range(self.level + 1, new_level + 1):
                    update[i] = self.head
                self.level = new_level

            new_node = SkipListNode(value, new_level)

            for i in range(new_level + 1):
                new_node.forward[i] = update[i].forward[i]
                update[i].forward[i] = new_node

    def delete(self, value):
        """删除值"""
        update = [None] * (self.max_level + 1)
        current = self.head

        for i in range(self.level, -1, -1):
            while current.forward[i] and current.forward[i].value < value:
                current = current.forward[i]
            update[i] = current

        current = current.forward[0]

        if current and current.value == value:
            for i in range(self.level + 1):
                if update[i].forward[i] != current:
                    break
                update[i].forward[i] = current.forward[i]

            while self.level > 0 and self.head.forward[self.level] is None:
                self.level -= 1

    def __str__(self):
        """打印跳表结构"""
        result = []
        for i in range(self.level, -1, -1):
            level_nodes = []
            current = self.head.forward[i]
            while current:
                level_nodes.append(str(current.value))
                current = current.forward[0]
            result.append(f"Level {i}: {' -> '.join(level_nodes)}")
        return "\n".join(result)
```

### 5. 布隆过滤器 (Bloom Filter)
```python
import mmh3
import math
from bitarray import bitarray

class BloomFilter:
    """
    布隆过滤器
    概率性数据结构，用于判断元素是否存在
    时间复杂度:
    - 插入: O(k) k为哈希函数数量
    - 查询: O(k)
    空间复杂度: O(m) m为位数组大小
    """

    def __init__(self, capacity, error_rate=0.01):
        """
        capacity: 预期元素数量
        error_rate: 期望的误判率
        """
        self.capacity = capacity
        self.error_rate = error_rate

        # 计算最优参数
        self.size = self._calculate_size(capacity, error_rate)
        self.hash_count = self._calculate_hash_count(self.size, capacity)

        self.bit_array = bitarray(self.size)
        self.bit_array.setall(False)

        self.count = 0

    def _calculate_size(self, capacity, error_rate):
        """计算位数组大小"""
        return int(- (capacity * math.log(error_rate)) / (math.log(2) ** 2))

    def _calculate_hash_count(self, size, capacity):
        """计算哈希函数数量"""
        return int((size / capacity) * math.log(2))

    def _get_hashes(self, item):
        """获取多个哈希值"""
        hashes = []
        for i in range(self.hash_count):
            # 使用不同的种子值
            hash_val = mmh3.hash(str(item), i) % self.size
            hashes.append(hash_val)
        return hashes

    def add(self, item):
        """添加元素"""
        if self.count >= self.capacity:
            raise Exception("Bloom filter is full")

        for hash_val in self._get_hashes(item):
            self.bit_array[hash_val] = True

        self.count += 1

    def contains(self, item):
        """检查元素是否存在"""
        for hash_val in self._get_hashes(item):
            if not self.bit_array[hash_val]:
                return False
        return True

    def __contains__(self, item):
        return self.contains(item)

    def __len__(self):
        return self.count

    def fill_ratio(self):
        """计算填充比例"""
        return self.bit_array.count() / self.size

class CountingBloomFilter:
    """
    计数布隆过滤器
    支持删除操作
    """

    def __init__(self, capacity, error_rate=0.01):
        self.capacity = capacity
        self.error_rate = error_rate
        self.size = self._calculate_size(capacity, error_rate)
        self.hash_count = self._calculate_hash_count(self.size, capacity)

        # 使用计数器数组
        self.counters = [0] * self.size

    def _calculate_size(self, capacity, error_rate):
        return int(- (capacity * math.log(error_rate)) / (math.log(2) ** 2))

    def _calculate_hash_count(self, size, capacity):
        return int((size / capacity) * math.log(2))

    def _get_hashes(self, item):
        hashes = []
        for i in range(self.hash_count):
            hash_val = mmh3.hash(str(item), i) % self.size
            hashes.append(hash_val)
        return hashes

    def add(self, item):
        """添加元素"""
        for hash_val in self._get_hashes(item):
            self.counters[hash_val] += 1

    def contains(self, item):
        """检查元素是否存在"""
        for hash_val in self._get_hashes(item):
            if self.counters[hash_val] == 0:
                return False
        return True

    def remove(self, item):
        """删除元素"""
        if not self.contains(item):
            return False

        for hash_val in self._get_hashes(item):
            self.counters[hash_val] -= 1

        return True
```

### 6. LRU缓存
```python
from collections import OrderedDict

class LRUCache:
    """
    LRU (Least Recently Used) 缓存
    时间复杂度:
    - 获取: O(1)
    - 设置: O(1)
    空间复杂度: O(capacity)
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = OrderedDict()
        self.access_order = OrderedDict()

    def get(self, key):
        """获取值"""
        if key not in self.cache:
            return None

        # 更新访问顺序
        self.access_order.move_to_end(key)
        return self.cache[key]

    def put(self, key, value):
        """设置值"""
        if key in self.cache:
            # 更新值和访问顺序
            self.cache[key] = value
            self.access_order.move_to_end(key)
        else:
            if len(self.cache) >= self.capacity:
                # 删除最久未使用的项
                lru_key = next(iter(self.access_order))
                del self.cache[lru_key]
                del self.access_order[lru_key]

            self.cache[key] = value
            self.access_order[key] = True

    def __str__(self):
        return f"LRUCache(capacity={self.capacity}, items={list(self.cache.keys())})"

class LFUCache:
    """
    LFU (Least Frequently Used) 缓存
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.cache = {}  # key -> (value, frequency)
        self.freq_map = {}  # frequency -> OrderedDict
        self.min_frequency = 0

    def get(self, key):
        """获取值"""
        if key not in self.cache:
            return None

        value, freq = self.cache[key]
        self._update_frequency(key, value, freq + 1)
        return value

    def put(self, key, value):
        """设置值"""
        if key in self.cache:
            _, freq = self.cache[key]
            self._update_frequency(key, value, freq + 1)
        else:
            if len(self.cache) >= self.capacity:
                # 删除频率最低的项
                if self.min_frequency in self.freq_map:
                    lru_key = next(iter(self.freq_map[self.min_frequency]))
                    del self.cache[lru_key]
                    del self.freq_map[self.min_frequency][lru_key]
                    if not self.freq_map[self.min_frequency]:
                        del self.freq_map[self.min_frequency]

            self.cache[key] = (value, 1)
            if 1 not in self.freq_map:
                self.freq_map[1] = OrderedDict()
            self.freq_map[1][key] = True
            self.min_frequency = 1

    def _update_frequency(self, key, value, new_freq):
        """更新频率"""
        old_freq = self.cache[key][1]
        del self.freq_map[old_freq][key]
        if not self.freq_map[old_freq]:
            del self.freq_map[old_freq]
            if old_freq == self.min_frequency:
                self.min_frequency += 1

        self.cache[key] = (value, new_freq)
        if new_freq not in self.freq_map:
            self.freq_map[new_freq] = OrderedDict()
        self.freq_map[new_freq][key] = True
```

### 7. 字典树 (Trie) 的变种
```python
class CompressedTrieNode:
    """压缩字典树节点"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.value = None

class CompressedTrie:
    """
    压缩字典树（前缀树）
    节省空间，提高查询效率
    """

    def __init__(self):
        self.root = CompressedTrieNode()

    def insert(self, word, value=None):
        """插入单词"""
        node = self.root

        while word:
            # 查找匹配的子节点
            matched_child = None
            matched_prefix = ""
            matched_child_key = None

            for child_key, child_node in node.children.items():
                # 计算公共前缀
                common_prefix = self._common_prefix(word, child_key)
                if common_prefix:
                    matched_child = child_node
                    matched_prefix = common_prefix
                    matched_child_key = child_key
                    break

            if matched_child:
                if matched_prefix == matched_child_key:
                    # 完全匹配，继续向下
                    node = matched_child
                    word = word[len(matched_prefix):]
                else:
                    # 部分匹配，需要分裂节点
                    new_node = CompressedTrieNode()
                    remaining_key = matched_child_key[len(matched_prefix):]

                    # 重建节点关系
                    del node.children[matched_child_key]
                    node.children[matched_prefix] = new_node
                    new_node.children[remaining_key] = matched_child

                    node = new_node
                    word = word[len(matched_prefix):]
            else:
                # 没有匹配，直接添加
                node.children[word] = CompressedTrieNode()
                node = node.children[word]
                word = ""

        node.is_end = True
        node.value = value

    def _common_prefix(self, str1, str2):
        """计算两个字符串的公共前缀"""
        min_len = min(len(str1), len(str2))
        for i in range(min_len):
            if str1[i] != str2[i]:
                return str1[:i]
        return str1[:min_len]

    def search(self, word):
        """搜索单词"""
        node = self.root

        while word and node:
            matched_child = None
            matched_prefix = ""

            for child_key, child_node in node.children.items():
                common_prefix = self._common_prefix(word, child_key)
                if common_prefix:
                    matched_child = child_node
                    matched_prefix = common_prefix
                    break

            if matched_child:
                if matched_prefix == child_key:
                    node = matched_child
                    word = word[len(matched_prefix):]
                else:
                    # 部分匹配，但单词不完整
                    return None
            else:
                return None

        return node.value if node and node.is_end else None

class SuffixTree:
    """
    后缀树（简化版）
    用于字符串搜索和模式匹配
    """

    def __init__(self, text):
        self.text = text
        self.root = {}
        self.build_suffix_tree()

    def build_suffix_tree(self):
        """构建后缀树"""
        n = len(self.text)

        for i in range(n):
            suffix = self.text[i:]
            current = self.root

            for char in suffix:
                if char not in current:
                    current[char] = {}
                current = current[char]

    def search(self, pattern):
        """搜索模式"""
        current = self.root

        for char in pattern:
            if char not in current:
                return False
            current = current[char]

        return True

    def find_all_occurrences(self, pattern):
        """查找所有出现位置"""
        occurrences = []
        n = len(self.text)
        m = len(pattern)

        for i in range(n - m + 1):
            if self.text[i:i+m] == pattern:
                occurrences.append(i)

        return occurrences
```

### 8. 堆的高级变种
```python
import heapq

class MinMaxHeap:
    """
    最小-最大堆
    支持同时获取最小值和最大值
    """

    def __init__(self):
        self.min_heap = []
        self.max_heap = []
        self.entry_map = {}  # 值到堆节点的映射
        self.counter = 0  # 用于处理重复值

    def push(self, value):
        """添加值"""
        entry = [value, self.counter]
        self.counter += 1

        # 添加到最小堆
        heapq.heappush(self.min_heap, entry)
        # 添加到最大堆（使用负值）
        max_entry = [-value, entry[1]]
        heapq.heappush(self.max_heap, max_entry)

        # 更新映射
        self.entry_map[entry[1]] = {'min': entry, 'max': max_entry}

    def get_min(self):
        """获取最小值"""
        while self.min_heap:
            value, counter = self.min_heap[0]
            if self._is_valid(counter, 'min'):
                return value
            heapq.heappop(self.min_heap)
        return None

    def get_max(self):
        """获取最大值"""
        while self.max_heap:
            value, counter = self.max_heap[0]
            if self._is_valid(counter, 'max'):
                return -value
            heapq.heappop(self.max_heap)
        return None

    def pop_min(self):
        """弹出最小值"""
        while self.min_heap:
            value, counter = heapq.heappop(self.min_heap)
            if self._is_valid(counter, 'min'):
                # 标记为已删除
                self.entry_map[counter]['min'] = None
                return value
        return None

    def pop_max(self):
        """弹出最大值"""
        while self.max_heap:
            value, counter = heapq.heappop(self.max_heap)
            if self._is_valid(counter, 'max'):
                # 标记为已删除
                self.entry_map[counter]['max'] = None
                return -value
        return None

    def _is_valid(self, counter, heap_type):
        """检查堆节点是否有效"""
        return (counter in self.entry_map and
                self.entry_map[counter][heap_type] is not None)

class FibonacciHeap:
    """
    斐波那契堆
    支持高效的合并操作
    """

    class Node:
        def __init__(self, value):
            self.value = value
            self.degree = 0
            self.parent = None
            self.child = None
            self.left = self
            self.right = self
            self.marked = False

    def __init__(self):
        self.min = None
        self.count = 0
        self.root_list = self.Node(None)  # 哨兵节点

    def insert(self, value):
        """插入值"""
        new_node = self.Node(value)
        self._insert_node(new_node)
        self.count += 1
        return new_node

    def _insert_node(self, node):
        """插入节点到根列表"""
        if self.min is None:
            self.min = node
        else:
            node.right = self.root_list.right
            node.left = self.root_list
            self.root_list.right.left = node
            self.root_list.right = node

            if node.value < self.min.value:
                self.min = node

    def get_min(self):
        """获取最小值"""
        return self.min.value if self.min else None

    def extract_min(self):
        """提取最小值"""
        if self.min is None:
            return None

        min_node = self.min
        self.count -= 1

        # 将最小节点的子节点添加到根列表
        if min_node.child:
            child = min_node.child
            while True:
                next_child = child.right
                self._insert_node(child)
                child.parent = None
                if child == min_node.child:
                    break
                child = next_child

        # 从根列表中移除最小节点
        min_node.left.right = min_node.right
        min_node.right.left = min_node.left

        if min_node == min_node.right:
            self.min = None
        else:
            self.min = min_node.right
            self._consolidate()

        return min_node.value

    def _consolidate(self):
        """合并相同度数的树"""
        max_degree = int(math.log(self.count) * 2) + 1
        degree_table = [None] * max_degree

        current = self.root_list.right
        nodes_to_process = []

        while current != self.root_list:
            nodes_to_process.append(current)
            current = current.right

        for node in nodes_to_process:
            degree = node.degree
            while degree_table[degree] is not None:
                other = degree_table[degree]
                if node.value > other.value:
                    node, other = other, node
                self._link(other, node)
                degree_table[degree] = None
                degree += 1
            degree_table[degree] = node

        self.min = None
        for node in degree_table:
            if node is not None:
                if self.min is None or node.value < self.min.value:
                    self.min = node

    def _link(self, child, parent):
        """链接两个节点"""
        # 从根列表中移除child
        child.left.right = child.right
        child.right.left = child.left

        child.parent = parent
        if parent.child is None:
            parent.child = child
            child.right = child
            child.left = child
        else:
            child.right = parent.child.right
            child.left = parent.child
            parent.child.right.left = child
            parent.child.right = child

        parent.degree += 1
        child.marked = False
```

## 性能测试与比较

```python
import time
import random

def benchmark_advanced_structures():
    """高级数据结构性能测试"""
    # 测试并查集
    print("=== 并查集测试 ===")
    uf = UnionFind(range(10000))
    start_time = time.time()
    for i in range(10000):
        uf.union(random.randint(0, 9999), random.randint(0, 9999))
    union_time = time.time() - start_time
    print(f"10000次合并操作: {union_time:.6f}s")

    # 测试线段树
    print("\n=== 线段树测试 ===")
    data = [random.randint(1, 1000) for _ in range(10000)]
    st = SegmentTree(data, func=sum)
    start_time = time.time()
    for i in range(1000):
        st.query(random.randint(0, 9999), random.randint(0, 9999))
    query_time = time.time() - start_time
    print(f"1000次区间查询: {query_time:.6f}s")

    # 测试跳表
    print("\n=== 跳表测试 ===")
    sl = SkipList()
    keys = list(range(10000))
    random.shuffle(keys)
    start_time = time.time()
    for key in keys:
        sl.insert(key)
    insert_time = time.time() - start_time
    print(f"10000次插入: {insert_time:.6f}s")

    start_time = time.time()
    for i in range(1000):
        sl.search(random.randint(0, 9999))
    search_time = time.time() - start_time
    print(f"1000次搜索: {search_time:.6f}s")

    # 测试布隆过滤器
    print("\n=== 布隆过滤器测试 ===")
    bf = BloomFilter(10000, 0.01)
    test_items = [f"item_{i}" for i in range(10000)]
    start_time = time.time()
    for item in test_items:
        bf.add(item)
    insert_time = time.time() - start_time
    print(f"10000次插入: {insert_time:.6f}s")

    start_time = time.time()
    for i in range(1000):
        bf.contains(f"item_{random.randint(0, 9999)}")
    search_time = time.time() - start_time
    print(f"1000次查询: {search_time:.6f}s")
```

## 应用场景

### 1. 数据库系统
- **索引结构**: B树、B+树、LSM树
- **连接算法**: 哈希连接、归并连接
- **缓存管理**: LRU、LFU缓存策略

```python
class DatabaseIndex:
    """数据库索引实现"""
    def __init__(self):
        self.primary_index = SkipList()  # 主键索引
        self.secondary_indexes = {}      # 次级索引
        self.cache = LRUCache(1000)      # 查询缓存

    def add_record(self, record_id, record_data):
        """添加记录"""
        # 更新主索引
        self.primary_index.insert(record_id, record_data)

        # 更新次级索引
        for field_name, field_value in record_data.items():
            if field_name not in self.secondary_indexes:
                self.secondary_indexes[field_name] = {}
            if field_value not in self.secondary_indexes[field_name]:
                self.secondary_indexes[field_name][field_value] = []
            self.secondary_indexes[field_name][field_value].append(record_id)

    def query_by_primary(self, record_id):
        """根据主键查询"""
        # 检查缓存
        cache_key = f"primary_{record_id}"
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            return cached_result

        # 查询主索引
        result = self.primary_index.search(record_id)
        self.cache.put(cache_key, result)
        return result

    def query_by_field(self, field_name, field_value):
        """根据字段查询"""
        if field_name not in self.secondary_indexes:
            return []

        record_ids = self.secondary_indexes[field_name].get(field_value, [])
        results = []

        for record_id in record_ids:
            record = self.query_by_primary(record_id)
            if record:
                results.append(record)

        return results
```

### 2. 操作系统
- **内存管理**: 伙伴系统、slab分配器
- **文件系统**: B树、extents
- **进程调度**: 红黑树、完全公平调度器

```python
class MemoryManager:
    """内存管理器"""
    def __init__(self, total_memory):
        self.total_memory = total_memory
        self.free_blocks = FenwickTree(total_memory)  # 空闲块管理
        self.allocated_blocks = {}  # 已分配块
        self.buddy_system = [2**i for i in range(0, 21)]  # 伙伴系统

    def allocate(self, size):
        """分配内存块"""
        # 找到合适的块大小
        block_size = self._find_block_size(size)
        block_address = self._find_free_block(block_size)

        if block_address is None:
            raise MemoryError("Out of memory")

        self.allocated_blocks[block_address] = block_size
        return block_address

    def deallocate(self, address):
        """释放内存块"""
        if address not in self.allocated_blocks:
            return

        size = self.allocated_blocks[address]
        del self.allocated_blocks[address]

        # 合并伙伴块
        self._merge_buddies(address, size)

    def _find_block_size(self, size):
        """找到合适的块大小"""
        for i in range(len(self.buddy_system)):
            if self.buddy_system[i] >= size:
                return self.buddy_system[i]
        return None

    def _find_free_block(self, size):
        """查找空闲块"""
        # 简化实现，实际需要更复杂的算法
        return random.randint(0, self.total_memory - size)

    def _merge_buddies(self, address, size):
        """合并伙伴块"""
        # 简化实现
        pass
```

### 3. 网络路由
- **路由表**: 前缀树（Trie）
- **负载均衡**: 一致性哈希
- **连接管理**: LRU缓存

```python
class NetworkRouter:
    """网络路由器"""
    def __init__(self):
        self.routing_table = CompressedTrie()  # IP路由表
        self.connection_cache = LRUCache(10000)  # 连接缓存
        self.load_balancer = ConsistentHash()   # 负载均衡

    def add_route(self, ip_prefix, next_hop):
        """添加路由"""
        self.routing_table.insert(ip_prefix, next_hop)

    def route_packet(self, destination_ip):
        """路由数据包"""
        # 查找路由
        next_hop = self.routing_table.search(destination_ip)
        if next_hop is None:
            return None

        # 更新连接缓存
        cache_key = f"{destination_ip}_{next_hop}"
        self.connection_cache.put(cache_key, time.time())

        return next_hop

    def load_balance(self, request_key):
        """负载均衡"""
        server = self.load_balancer.get_node(request_key)
        return server

class ConsistentHash:
    """一致性哈希"""
    def __init__(self, virtual_nodes=100):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []

    def add_node(self, node):
        """添加节点"""
        for i in range(self.virtual_nodes):
            virtual_node = f"{node}#{i}"
            hash_val = self._hash(virtual_node)
            self.ring[hash_val] = node
            self.sorted_keys.append(hash_val)

        self.sorted_keys.sort()

    def remove_node(self, node):
        """删除节点"""
        for i in range(self.virtual_nodes):
            virtual_node = f"{node}#{i}"
            hash_val = self._hash(virtual_node)
            if hash_val in self.ring:
                del self.ring[hash_val]
                index = bisect.bisect_left(self.sorted_keys, hash_val)
                if index < len(self.sorted_keys) and self.sorted_keys[index] == hash_val:
                    self.sorted_keys.pop(index)

    def get_node(self, key):
        """获取节点"""
        if not self.ring:
            return None

        hash_val = self._hash(key)
        index = bisect.bisect_right(self.sorted_keys, hash_val)
        if index == len(self.sorted_keys):
            index = 0

        return self.ring[self.sorted_keys[index]]

    def _hash(self, key):
        """哈希函数"""
        return int(hashlib.md5(str(key).encode()).hexdigest(), 16)
```

### 4. 游戏开发
- **空间索引**: 四叉树、八叉树
- **AI路径查找**: A*算法、跳点搜索
- **物理引擎**: 空间划分、碰撞检测

```python
class QuadTree:
    """四叉树空间索引"""
    def __init__(self, bounds, max_objects=10, max_levels=5, level=0):
        self.bounds = bounds  # (x, y, width, height)
        self.max_objects = max_objects
        self.max_levels = max_levels
        self.level = level
        self.objects = []
        self.nodes = []

    def insert(self, obj, obj_bounds):
        """插入对象"""
        if not self._contains(obj_bounds):
            return False

        if len(self.objects) < self.max_objects and self.level == self.max_levels:
            self.objects.append((obj, obj_bounds))
            return True

        if len(self.nodes) == 0:
            self._split()

        for node in self.nodes:
            if node.insert(obj, obj_bounds):
                return True

        self.objects.append((obj, obj_bounds))
        return True

    def retrieve(self, search_bounds):
        """检索对象"""
        result = []
        if not self._intersects(search_bounds):
            return result

        for obj, obj_bounds in self.objects:
            if self._intersects_bounds(obj_bounds, search_bounds):
                result.append(obj)

        for node in self.nodes:
            result.extend(node.retrieve(search_bounds))

        return result

    def _split(self):
        """分裂节点"""
        sub_width = self.bounds[2] / 2
        sub_height = self.bounds[3] / 2
        x, y = self.bounds[0], self.bounds[1]

        self.nodes = [
            QuadTree((x, y, sub_width, sub_height), self.max_objects,
                    self.max_levels, self.level + 1),
            QuadTree((x + sub_width, y, sub_width, sub_height), self.max_objects,
                    self.max_levels, self.level + 1),
            QuadTree((x, y + sub_height, sub_width, sub_height), self.max_objects,
                    self.max_levels, self.level + 1),
            QuadTree((x + sub_width, y + sub_height, sub_width, sub_height),
                    self.max_objects, self.max_levels, self.level + 1)
        ]

    def _contains(self, bounds):
        """检查边界是否包含在节点内"""
        x, y, w, h = bounds
        node_x, node_y, node_w, node_h = self.bounds
        return (x >= node_x and x + w <= node_x + node_w and
                y >= node_y and y + h <= node_y + node_h)

    def _intersects(self, bounds):
        """检查边界是否与节点相交"""
        x, y, w, h = bounds
        node_x, node_y, node_w, node_h = self.bounds
        return not (x > node_x + node_w or x + w < node_x or
                    y > node_y + node_h or y + h < node_y)

    def _intersects_bounds(self, bounds1, bounds2):
        """检查两个边界是否相交"""
        x1, y1, w1, h1 = bounds1
        x2, y2, w2, h2 = bounds2
        return not (x1 > x2 + w2 or x1 + w1 < x2 or
                    y1 > y2 + h2 or y1 + h1 < y2)
```

## 高级技巧与优化

### 1. 内存池管理
```python
class MemoryPool:
    """内存池管理器"""
    def __init__(self, block_size, pool_size):
        self.block_size = block_size
        self.pool_size = pool_size
        self.free_blocks = []
        self.allocated_blocks = set()
        self._initialize_pool()

    def _initialize_pool(self):
        """初始化内存池"""
        memory = bytearray(block_size * pool_size)
        for i in range(pool_size):
            block_address = i * block_size
            self.free_blocks.append(memory[block_address:block_address + block_size])

    def allocate(self):
        """分配内存块"""
        if not self.free_blocks:
            raise MemoryError("Memory pool exhausted")

        block = self.free_blocks.pop()
        self.allocated_blocks.add(id(block))
        return block

    def deallocate(self, block):
        """释放内存块"""
        if id(block) not in self.allocated_blocks:
            raise ValueError("Block not allocated")

        self.allocated_blocks.remove(id(block))
        self.free_blocks.append(block)
```

### 2. 锁无关数据结构
```python
import threading

class LockFreeStack:
    """锁无关栈"""
    def __init__(self):
        self.head = None
        self.lock = threading.Lock()

    def push(self, value):
        """入栈"""
        with self.lock:
            new_node = self.Node(value)
            new_node.next = self.head
            self.head = new_node

    def pop(self):
        """出栈"""
        with self.lock:
            if self.head is None:
                return None

            value = self.head.value
            self.head = self.head.next
            return value

    class Node:
        def __init__(self, value):
            self.value = value
            self.next = None
```

### 3. 缓存友好的设计
```python
class CacheFriendlyArray:
    """缓存友好的数组"""
    def __init__(self, size):
        self.size = size
        self.data = [0] * size
        self.block_size = 64  # 缓存行大小

    def block_access(self, start, end):
        """块访问"""
        # 对齐到块边界
        aligned_start = (start // self.block_size) * self.block_size
        aligned_end = ((end // self.block_size) + 1) * self.block_size

        # 批量访问
        for i in range(aligned_start, min(aligned_end, self.size)):
            # 处理数据
            pass

    def prefetch_data(self, index):
        """预取数据"""
        if index + self.block_size < self.size:
            # 预取下一个块
            next_block = self.data[index + self.block_size:
                                     index + 2 * self.block_size]
```

## 练习

1. 实现一个支持并发访问的跳表
2. 创建一个支持范围查询的布隆过滤器变种
3. 实现一个自适应的LRU缓存策略
4. 设计一个支持增量更新的线段树
5. 实现一个分布式的一致性哈希环
6. 创建一个支持事务的内存数据库
7. 实现一个基于跳表的分布式索引
8. 设计一个支持版本控制的数据结构