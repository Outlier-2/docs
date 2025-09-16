# L05 - 哈希表

## 学习目标
- 理解哈希表的基本原理和工作机制
- 掌握哈希函数的设计原则
- 学会处理哈希冲突的各种方法
- 实现高效的哈希表操作

## 哈希表基础

### 核心概念
- **哈希函数**: 将键映射到数组索引的函数
- **冲突**: 不同键映射到相同索引的现象
- **负载因子**: 已存储元素数量与表大小的比值
- **再哈希**: 当负载因子过高时扩大表大小的过程

### 时间复杂度
- **平均情况**: O(1) 插入、删除、查找
- **最坏情况**: O(n) 当所有键都冲突时
- **空间复杂度**: O(n)

## Python实现

### 1. 基础哈希表
```python
class HashTable:
    """
    基础哈希表实现（使用链地址法）
    时间复杂度:
    - 插入: O(1) 平均, O(n) 最坏
    - 查找: O(1) 平均, O(n) 最坏
    - 删除: O(1) 平均, O(n) 最坏
    空间复杂度: O(n)
    """

    def __init__(self, capacity=10):
        self.capacity = capacity
        self.size = 0
        self.table = [[] for _ in range(capacity)]
        self.load_factor_threshold = 0.7

    def _hash(self, key):
        """简单的哈希函数"""
        return hash(key) % self.capacity

    def _resize(self, new_capacity):
        """调整哈希表大小"""
        old_table = self.table
        self.capacity = new_capacity
        self.table = [[] for _ in range(new_capacity)]
        self.size = 0

        # 重新插入所有元素
        for bucket in old_table:
            for key, value in bucket:
                self.insert(key, value)

    def insert(self, key, value):
        """插入键值对"""
        if self.size / self.capacity > self.load_factor_threshold:
            self._resize(2 * self.capacity)

        index = self._hash(key)

        # 检查键是否已存在
        for i, (existing_key, existing_value) in enumerate(self.table[index]):
            if existing_key == key:
                self.table[index][i] = (key, value)
                return

        # 插入新键值对
        self.table[index].append((key, value))
        self.size += 1

    def get(self, key):
        """获取值"""
        index = self._hash(key)
        for existing_key, value in self.table[index]:
            if existing_key == key:
                return value
        raise KeyError(f"Key '{key}' not found")

    def delete(self, key):
        """删除键值对"""
        index = self._hash(key)
        for i, (existing_key, value) in enumerate(self.table[index]):
            if existing_key == key:
                self.table[index].pop(i)
                self.size -= 1
                return
        raise KeyError(f"Key '{key}' not found")

    def contains(self, key):
        """检查键是否存在"""
        try:
            self.get(key)
            return True
        except KeyError:
            return False

    def __len__(self):
        return self.size

    def __str__(self):
        items = []
        for bucket in self.table:
            for key, value in bucket:
                items.append(f"{key}: {value}")
        return "{" + ", ".join(items) + "}"

    def __contains__(self, key):
        return self.contains(key)

    def __getitem__(self, key):
        return self.get(key)

    def __setitem__(self, key, value):
        self.insert(key, value)

    def __delitem__(self, key):
        self.delete(key)
```

### 2. 开放寻址法哈希表
```python
class OpenAddressingHashTable:
    """
    开放寻址法哈希表
    使用线性探测解决冲突
    """

    def __init__(self, capacity=10):
        self.capacity = capacity
        self.size = 0
        self.table = [None] * capacity
        self.load_factor_threshold = 0.7
        self.DELETED = object()  # 特殊标记表示已删除

    def _hash(self, key):
        """基础哈希函数"""
        return hash(key) % self.capacity

    def _probe(self, key, attempt):
        """线性探测函数"""
        return (self._hash(key) + attempt) % self.capacity

    def _resize(self, new_capacity):
        """调整哈希表大小"""
        old_table = self.table
        self.capacity = new_capacity
        self.table = [None] * new_capacity
        self.size = 0

        # 重新插入所有元素
        for item in old_table:
            if item and item != self.DELETED:
                key, value = item
                self.insert(key, value)

    def insert(self, key, value):
        """插入键值对"""
        if self.size / self.capacity > self.load_factor_threshold:
            self._resize(2 * self.capacity)

        for attempt in range(self.capacity):
            index = self._probe(key, attempt)

            if self.table[index] is None or self.table[index] == self.DELETED:
                self.table[index] = (key, value)
                self.size += 1
                return
            elif self.table[index][0] == key:
                self.table[index] = (key, value)
                return

        raise Exception("Hash table is full")

    def get(self, key):
        """获取值"""
        for attempt in range(self.capacity):
            index = self._probe(key, attempt)

            if self.table[index] is None:
                raise KeyError(f"Key '{key}' not found")
            elif self.table[index] != self.DELETED and self.table[index][0] == key:
                return self.table[index][1]

        raise KeyError(f"Key '{key}' not found")

    def delete(self, key):
        """删除键值对"""
        for attempt in range(self.capacity):
            index = self._probe(key, attempt)

            if self.table[index] is None:
                raise KeyError(f"Key '{key}' not found")
            elif self.table[index] != self.DELETED and self.table[index][0] == key:
                self.table[index] = self.DELETED
                self.size -= 1
                return

        raise KeyError(f"Key '{key}' not found")

    def contains(self, key):
        """检查键是否存在"""
        try:
            self.get(key)
            return True
        except KeyError:
            return False
```

### 3. 双重哈希表
```python
class DoubleHashTable:
    """
    双重哈希表
    使用双重哈希函数减少聚集
    """

    def __init__(self, capacity=10):
        self.capacity = capacity
        self.size = 0
        self.table = [None] * capacity
        self.load_factor_threshold = 0.7
        self.DELETED = object()

    def _hash1(self, key):
        """第一个哈希函数"""
        return hash(key) % self.capacity

    def _hash2(self, key):
        """第二个哈希函数"""
        # 确保第二个哈希函数的结果与capacity互质
        hash_val = hash(key)
        return 1 + (hash_val % (self.capacity - 1))

    def _probe(self, key, attempt):
        """双重哈希探测函数"""
        return (self._hash1(key) + attempt * self._hash2(key)) % self.capacity

    def _resize(self, new_capacity):
        """调整哈希表大小"""
        old_table = self.table
        self.capacity = new_capacity
        self.table = [None] * new_capacity
        self.size = 0

        for item in old_table:
            if item and item != self.DELETED:
                key, value = item
                self.insert(key, value)

    def insert(self, key, value):
        """插入键值对"""
        if self.size / self.capacity > self.load_factor_threshold:
            self._resize(self._next_prime(2 * self.capacity))

        for attempt in range(self.capacity):
            index = self._probe(key, attempt)

            if self.table[index] is None or self.table[index] == self.DELETED:
                self.table[index] = (key, value)
                self.size += 1
                return
            elif self.table[index][0] == key:
                self.table[index] = (key, value)
                return

        raise Exception("Hash table is full")

    def get(self, key):
        """获取值"""
        for attempt in range(self.capacity):
            index = self._probe(key, attempt)

            if self.table[index] is None:
                raise KeyError(f"Key '{key}' not found")
            elif self.table[index] != self.DELETED and self.table[index][0] == key:
                return self.table[index][1]

        raise KeyError(f"Key '{key}' not found")

    def _next_prime(self, n):
        """找到下一个质数"""
        def is_prime(num):
            if num < 2:
                return False
            for i in range(2, int(num ** 0.5) + 1):
                if num % i == 0:
                    return False
            return True

        while not is_prime(n):
            n += 1
        return n
```

### 4. 布隆过滤器
```python
import mmh3
from bitarray import bitarray

class BloomFilter:
    """
    布隆过滤器
    概率性数据结构，用于判断元素是否在集合中
    """

    def __init__(self, capacity, error_rate=0.01):
        self.capacity = capacity
        self.error_rate = error_rate
        self.bit_array = bitarray(self._calculate_size())
        self.bit_array.setall(False)
        self.hash_count = self._calculate_hash_count()

    def _calculate_size(self):
        """计算位数组大小"""
        return int(- (self.capacity * math.log(self.error_rate)) / (math.log(2) ** 2))

    def _calculate_hash_count(self):
        """计算哈希函数数量"""
        return int((self._calculate_size() / self.capacity) * math.log(2))

    def _get_hashes(self, item):
        """获取多个哈希值"""
        hashes = []
        for i in range(self.hash_count):
            # 使用不同的种子值
            hash_val = mmh3.hash(str(item), i) % len(self.bit_array)
            hashes.append(hash_val)
        return hashes

    def add(self, item):
        """添加元素"""
        for hash_val in self._get_hashes(item):
            self.bit_array[hash_val] = True

    def contains(self, item):
        """检查元素是否存在"""
        for hash_val in self._get_hashes(item):
            if not self.bit_array[hash_val]:
                return False
        return True

    def __contains__(self, item):
        return self.contains(item)
```

### 5. 一致性哈希
```python
import hashlib
import bisect

class ConsistentHash:
    """
    一致性哈希
    用于分布式系统中的负载均衡
    """

    def __init__(self, virtual_nodes=100):
        self.virtual_nodes = virtual_nodes
        self.ring = {}
        self.sorted_keys = []

    def _hash(self, key):
        """一致性哈希函数"""
        return int(hashlib.md5(str(key).encode()).hexdigest(), 16)

    def add_node(self, node):
        """添加节点"""
        for i in range(self.virtual_nodes):
            virtual_node = f"{node}#{i}"
            hash_val = self._hash(virtual_node)
            self.ring[hash_val] = node
            bisect.insort(self.sorted_keys, hash_val)

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
        """获取键对应的节点"""
        if not self.ring:
            return None

        hash_val = self._hash(key)
        index = bisect.bisect_right(self.sorted_keys, hash_val)

        if index == len(self.sorted_keys):
            index = 0

        return self.ring[self.sorted_keys[index]]
```

## 哈希函数设计

### 1. 乘法哈希
```python
def multiplication_hash(key, table_size):
    """
    乘法哈希函数
    适合整数键
    """
    A = 0.6180339887  # 黄金分割比
    return int(table_size * ((key * A) % 1))
```

### 2. 多项式滚动哈希
```python
def polynomial_rolling_hash(s, base=131, mod=10**9 + 7):
    """
    多项式滚动哈希
    适合字符串键
    """
    hash_val = 0
    for char in s:
        hash_val = (hash_val * base + ord(char)) % mod
    return hash_val
```

### 3. MurmurHash
```python
def murmur_hash(key, seed=0):
    """
    MurmurHash 2.0 简化实现
    分布式均匀，速度快
    """
    m = 0x5bd1e995
    r = 24

    # 将key转换为整数
    if isinstance(key, str):
        key = int.from_bytes(key.encode(), 'big')
    elif isinstance(key, (bytes, bytearray)):
        key = int.from_bytes(key, 'big')

    key = (key * m) ^ seed
    key = (key ^ (key >> r)) * m
    key = (key ^ (key >> r)) * m
    key = key ^ (key >> r)

    return key
```

## 性能测试与比较

```python
import time
import random
import string

def benchmark_hash_tables():
    """哈希表性能测试"""
    sizes = [1000, 5000, 10000, 20000]

    for size in sizes:
        print(f"\n=== 测试大小: {size} ===")

        # 生成测试数据
        keys = [random.randint(1, 1000000) for _ in range(size)]
        values = [random.random() for _ in range(size)]

        # 测试链地址法
        chain_table = HashTable(size)
        start_time = time.time()
        for key, value in zip(keys, values):
            chain_table.insert(key, value)
        chain_insert_time = time.time() - start_time

        # 测试开放寻址法
        open_table = OpenAddressingHashTable(size)
        start_time = time.time()
        for key, value in zip(keys, values):
            open_table.insert(key, value)
        open_insert_time = time.time() - start_time

        # 测试双重哈希
        double_table = DoubleHashTable(size)
        start_time = time.time()
        for key, value in zip(keys, values):
            double_table.insert(key, value)
        double_insert_time = time.time() - start_time

        # Python内置dict
        py_dict = {}
        start_time = time.time()
        for key, value in zip(keys, values):
            py_dict[key] = value
        py_dict_time = time.time() - start_time

        print(f"链地址法插入: {chain_insert_time:.6f}s")
        print(f"开放寻址法插入: {open_insert_time:.6f}s")
        print(f"双重哈希插入: {double_insert_time:.6f}s")
        print(f"Python dict插入: {py_dict_time:.6f}s")

        # 测试查找性能
        search_keys = random.sample(keys, 1000)

        start_time = time.time()
        for key in search_keys:
            chain_table.get(key)
        chain_search_time = time.time() - start_time

        start_time = time.time()
        for key in search_keys:
            open_table.get(key)
        open_search_time = time.time() - start_time

        start_time = time.time()
        for key in search_keys:
            double_table.get(key)
        double_search_time = time.time() - start_time

        start_time = time.time()
        for key in search_keys:
            py_dict[key]
        py_dict_search_time = time.time() - start_time

        print(f"链地址法查找: {chain_search_time:.6f}s")
        print(f"开放寻址法查找: {open_search_time:.6f}s")
        print(f"双重哈希查找: {double_search_time:.6f}s")
        print(f"Python dict查找: {py_dict_search_time:.6f}s")
```

## 应用场景

### 1. 数据库索引
- **哈希索引**: 快速精确匹配
- **内存表**: 高性能内存数据库
- **缓存系统**: Redis、Memcached

### 2. 编译器
- **符号表**: 变量和函数名查找
- **常量表**: 常量值快速查找
- **语法分析**: 关键字识别

### 3. 网络应用
- **路由表**: IP路由查找
- **会话管理**: HTTP会话存储
- **负载均衡**: 一致性哈希分配请求

### 4. 大数据处理
- **数据去重**: 快速判断重复数据
- **频率统计**: 统计元素出现频率
- **数据分片**: 分布式数据存储

## 高级技巧与优化

### 1. 线程安全哈希表
```python
import threading

class ThreadSafeHashTable:
    """线程安全的哈希表"""
    def __init__(self, capacity=10):
        self.table = [threading.RLock() for _ in range(capacity)]
        self.data = [[] for _ in range(capacity)]
        self.capacity = capacity
        self.size = 0

    def _hash(self, key):
        return hash(key) % self.capacity

    def insert(self, key, value):
        index = self._hash(key)
        with self.table[index]:
            for i, (k, v) in enumerate(self.data[index]):
                if k == key:
                    self.data[index][i] = (key, value)
                    return
            self.data[index].append((key, value))
            self.size += 1

    def get(self, key):
        index = self._hash(key)
        with self.table[index]:
            for k, v in self.data[index]:
                if k == key:
                    return v
            raise KeyError(f"Key '{key}' not found")
```

### 2. 持久化哈希表
```python
import json
import os

class PersistentHashTable:
    """持久化哈希表"""
    def __init__(self, filename="hash_table.json"):
        self.filename = filename
        self.data = self._load()

    def _load(self):
        """从文件加载数据"""
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                return json.load(f)
        return {}

    def _save(self):
        """保存数据到文件"""
        with open(self.filename, 'w') as f:
            json.dump(self.data, f)

    def insert(self, key, value):
        self.data[key] = value
        self._save()

    def get(self, key):
        if key in self.data:
            return self.data[key]
        raise KeyError(f"Key '{key}' not found")

    def delete(self, key):
        if key in self.data:
            del self.data[key]
            self._save()
        else:
            raise KeyError(f"Key '{key}' not found")
```

### 3. 缓存友好的哈希表
```python
class CacheFriendlyHashTable:
    """缓存友好的哈希表"""
    def __init__(self, capacity=1024):
        self.capacity = capacity
        self.size = 0
        self.buckets = [[] for _ in range(capacity)]
        self.bucket_size = 8  # 每个桶的大小，适合缓存行

    def _hash(self, key):
        return hash(key) % self.capacity

    def insert(self, key, value):
        index = self._hash(key)
        bucket = self.buckets[index]

        for i, (k, v) in enumerate(bucket):
            if k == key:
                bucket[i] = (key, value)
                return

        if len(bucket) < self.bucket_size:
            bucket.append((key, value))
            self.size += 1
        else:
            # 桶已满，扩容
            self._resize()
            self.insert(key, value)

    def _resize(self):
        """扩容哈希表"""
        new_capacity = self.capacity * 2
        new_buckets = [[] for _ in range(new_capacity)]

        for bucket in self.buckets:
            for key, value in bucket:
                index = hash(key) % new_capacity
                new_buckets[index].append((key, value))

        self.capacity = new_capacity
        self.buckets = new_buckets
```

## 练习

1. 实现一个支持LRU缓存的哈希表
2. 创建一个支持范围查询的哈希表变体
3. 实现一个分布式哈希表（DHT）
4. 设计一个支持事务的哈希表
5. 实现一个布谷鸟哈希表（Cuckoo Hashing）
6. 创建一个支持并发访问的高性能哈希表
7. 实现一个支持多种数据类型的哈希表
8. 设计一个支持数据压缩的哈希表