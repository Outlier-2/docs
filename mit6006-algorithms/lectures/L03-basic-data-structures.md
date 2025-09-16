# L03 - 数据结构基础

## 学习目标
- 掌握基本数据结构的原理和实现
- 理解不同数据结构的时空复杂度
- 学会根据应用场景选择合适的数据结构
- 实现高效的数据结构操作

## 基本数据结构分类

### 线性结构
- **数组**: 连续内存存储，随机访问O(1)
- **链表**: 动态内存分配，插入删除O(1)
- **栈**: LIFO（后进先出）
- **队列**: FIFO（先进先出）

### 非线性结构
- **树**: 层次结构，搜索效率高
- **图**: 网络结构，关系复杂

## Python实现

### 1. 动态数组 (Dynamic Array)
```python
class DynamicArray:
    """
    动态数组实现
    时间复杂度:
    - 访问: O(1)
    - 插入(末尾): 分摊O(1)
    - 插入(中间): O(n)
    - 删除: O(n)
    空间复杂度: O(n)
    """

    def __init__(self):
        self.capacity = 1  # 初始容量
        self.size = 0      # 当前大小
        self.array = [None] * self.capacity

    def __len__(self) -> int:
        """返回数组大小"""
        return self.size

    def __getitem__(self, index: int):
        """随机访问"""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")
        return self.array[index]

    def __setitem__(self, index: int, value):
        """设置元素"""
        if index < 0 or index >= self.size:
            raise IndexError("Index out of range")
        self.array[index] = value

    def resize(self, new_capacity: int):
        """调整数组容量"""
        new_array = [None] * new_capacity
        for i in range(self.size):
            new_array[i] = self.array[i]
        self.array = new_array
        self.capacity = new_capacity

    def append(self, value):
        """在末尾添加元素"""
        if self.size == self.capacity:
            self.resize(2 * self.capacity)  # 双倍扩容

        self.array[self.size] = value
        self.size += 1

    def insert(self, index: int, value):
        """在指定位置插入元素"""
        if index < 0 or index > self.size:
            raise IndexError("Index out of range")

        if self.size == self.capacity:
            self.resize(2 * self.capacity)

        # 移动元素
        for i in range(self.size, index, -1):
            self.array[i] = self.array[i - 1]

        self.array[index] = value
        self.size += 1

    def remove(self, value):
        """删除指定元素"""
        index = -1
        for i in range(self.size):
            if self.array[i] == value:
                index = i
                break

        if index == -1:
            raise ValueError("Value not found")

        # 移动元素
        for i in range(index, self.size - 1):
            self.array[i] = self.array[i + 1]

        self.size -= 1

        # 缩容
        if self.size <= self.capacity // 4 and self.capacity > 1:
            self.resize(self.capacity // 2)

    def pop(self) -> object:
        """删除并返回末尾元素"""
        if self.size == 0:
            raise IndexError("Empty array")

        value = self.array[self.size - 1]
        self.size -= 1

        # 缩容
        if self.size <= self.capacity // 4 and self.capacity > 1:
            self.resize(self.capacity // 2)

        return value

    def __str__(self) -> str:
        return f"DynamicArray({[self.array[i] for i in range(self.size)]})"
```

### 2. 链表 (Linked List)
```python
class ListNode:
    """链表节点"""
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class LinkedList:
    """
    单链表实现
    时间复杂度:
    - 访问: O(n)
    - 插入(头部): O(1)
    - 插入(尾部): O(n) 或 O(1) 有尾指针
    - 删除: O(n)
    空间复杂度: O(n)
    """

    def __init__(self):
        self.head = None
        self.tail = None
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def is_empty(self) -> bool:
        return self.size == 0

    def append(self, val):
        """在尾部添加元素"""
        new_node = ListNode(val)

        if self.is_empty():
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

        self.size += 1

    def prepend(self, val):
        """在头部添加元素"""
        new_node = ListNode(val)

        if self.is_empty():
            self.head = new_node
            self.tail = new_node
        else:
            new_node.next = self.head
            self.head = new_node

        self.size += 1

    def insert(self, index: int, val):
        """在指定位置插入元素"""
        if index < 0 or index > self.size:
            raise IndexError("Index out of range")

        if index == 0:
            self.prepend(val)
        elif index == self.size:
            self.append(val)
        else:
            new_node = ListNode(val)
            current = self.head

            # 找到插入位置的前一个节点
            for _ in range(index - 1):
                current = current.next

            new_node.next = current.next
            current.next = new_node
            self.size += 1

    def remove(self, val):
        """删除指定元素"""
        if self.is_empty():
            raise ValueError("Empty list")

        if self.head.val == val:
            self.head = self.head.next
            if self.head is None:
                self.tail = None
            self.size -= 1
            return

        current = self.head
        while current.next and current.next.val != val:
            current = current.next

        if current.next is None:
            raise ValueError("Value not found")

        current.next = current.next.next
        if current.next is None:
            self.tail = current

        self.size -= 1

    def find(self, val) -> int:
        """查找元素位置"""
        current = self.head
        index = 0

        while current:
            if current.val == val:
                return index
            current = current.next
            index += 1

        return -1

    def __str__(self) -> str:
        values = []
        current = self.head
        while current:
            values.append(str(current.val))
            current = current.next
        return " -> ".join(values)
```

### 3. 栈 (Stack)
```python
class Stack:
    """
    栈实现 (LIFO)
    时间复杂度:
    - push: O(1)
    - pop: O(1)
    - peek: O(1)
    空间复杂度: O(n)
    """

    def __init__(self):
        self.items = []

    def push(self, item):
        """入栈"""
        self.items.append(item)

    def pop(self):
        """出栈"""
        if self.is_empty():
            raise IndexError("Empty stack")
        return self.items.pop()

    def peek(self):
        """查看栈顶元素"""
        if self.is_empty():
            raise IndexError("Empty stack")
        return self.items[-1]

    def is_empty(self) -> bool:
        """检查是否为空"""
        return len(self.items) == 0

    def size(self) -> int:
        """返回栈大小"""
        return len(self.items)

    def __str__(self) -> str:
        return f"Stack({self.items})"
```

### 4. 队列 (Queue)
```python
class Queue:
    """
    队列实现 (FIFO)
    时间复杂度:
    - enqueue: O(1)
    - dequeue: O(1)
    空间复杂度: O(n)
    """

    def __init__(self):
        self.items = []

    def enqueue(self, item):
        """入队"""
        self.items.append(item)

    def dequeue(self):
        """出队"""
        if self.is_empty():
            raise IndexError("Empty queue")
        return self.items.pop(0)

    def front(self):
        """查看队首元素"""
        if self.is_empty():
            raise IndexError("Empty queue")
        return self.items[0]

    def is_empty(self) -> bool:
        """检查是否为空"""
        return len(self.items) == 0

    def size(self) -> int:
        """返回队列大小"""
        return len(self.items)

    def __str__(self) -> str:
        return f"Queue({self.items})"
```

### 5. 双端队列 (Deque)
```python
class Deque:
    """
    双端队列实现
    时间复杂度:
    - append_left: O(1)
    - append_right: O(1)
    - pop_left: O(1)
    - pop_right: O(1)
    空间复杂度: O(n)
    """

    def __init__(self):
        self.items = []

    def append_left(self, item):
        """从左侧添加"""
        self.items.insert(0, item)

    def append_right(self, item):
        """从右侧添加"""
        self.items.append(item)

    def pop_left(self):
        """从左侧弹出"""
        if self.is_empty():
            raise IndexError("Empty deque")
        return self.items.pop(0)

    def pop_right(self):
        """从右侧弹出"""
        if self.is_empty():
            raise IndexError("Empty deque")
        return self.items.pop()

    def peek_left(self):
        """查看左侧元素"""
        if self.is_empty():
            raise IndexError("Empty deque")
        return self.items[0]

    def peek_right(self):
        """查看右侧元素"""
        if self.is_empty():
            raise IndexError("Empty deque")
        return self.items[-1]

    def is_empty(self) -> bool:
        """检查是否为空"""
        return len(self.items) == 0

    def size(self) -> int:
        """返回双端队列大小"""
        return len(self.items)

    def __str__(self) -> str:
        return f"Deque({self.items})"
```

## 性能测试与比较

```python
import time
import random

def benchmark_data_structures():
    """性能基准测试"""
    sizes = [1000, 5000, 10000, 20000]

    for size in sizes:
        print(f"\n=== 测试大小: {size} ===")

        # 测试动态数组
        arr = DynamicArray()
        start_time = time.time()
        for i in range(size):
            arr.append(i)
        array_append_time = time.time() - start_time

        # 测试链表
        linked_list = LinkedList()
        start_time = time.time()
        for i in range(size):
            linked_list.append(i)
        linked_list_append_time = time.time() - start_time

        # 测试栈
        stack = Stack()
        start_time = time.time()
        for i in range(size):
            stack.push(i)
        stack_push_time = time.time() - start_time

        # 测试队列
        queue = Queue()
        start_time = time.time()
        for i in range(size):
            queue.enqueue(i)
        queue_enqueue_time = time.time() - start_time

        print(f"动态数组 append: {array_append_time:.6f}s")
        print(f"链表 append: {linked_list_append_time:.6f}s")
        print(f"栈 push: {stack_push_time:.6f}s")
        print(f"队列 enqueue: {queue_enqueue_time:.6f}s")

        # 测试随机访问性能
        indices = [random.randint(0, size-1) for _ in range(1000)]

        # 动态数组随机访问
        start_time = time.time()
        for idx in indices:
            _ = arr[idx]
        array_access_time = time.time() - start_time

        # 链表随机访问
        start_time = time.time()
        for idx in indices:
            current = linked_list.head
            for _ in range(idx):
                current = current.next
            _ = current.val
        linked_list_access_time = time.time() - start_time

        print(f"动态数组 随机访问: {array_access_time:.6f}s")
        print(f"链表 随机访问: {linked_list_access_time:.6f}s")
```

## 应用场景

### 1. 数组应用场景
- **图像处理**: 像素矩阵存储
- **数值计算**: 矩阵运算
- **缓存实现**: 固定大小的缓存
- **数据库索引**: B树的节点存储

```python
class ImageProcessor:
    """图像处理器 - 使用数组存储像素"""
    def __init__(self, width: int, height: int):
        self.width = width
        self.height = height
        self.pixels = [[0 for _ in range(width)] for _ in range(height)]

    def set_pixel(self, x: int, y: int, color: int):
        """设置像素值"""
        if 0 <= x < self.width and 0 <= y < self.height:
            self.pixels[y][x] = color

    def get_pixel(self, x: int, y: int) -> int:
        """获取像素值"""
        if 0 <= x < self.width and 0 <= y < self.height:
            return self.pixels[y][x]
        return 0
```

### 2. 链表应用场景
- **文本编辑器**: 行数据存储
- **音乐播放器**: 播放列表管理
- **浏览器历史**: 历史记录管理
- **内存管理**: 动态内存分配

```python
class TextEditor:
    """文本编辑器 - 使用链表存储文本行"""
    def __init__(self):
        self.lines = LinkedList()
        self.current_line = 0

    def insert_line(self, line: str, position: int = None):
        """插入文本行"""
        if position is None:
            position = self.current_line
        self.lines.insert(position, line)

    def delete_line(self, position: int = None):
        """删除文本行"""
        if position is None:
            position = self.current_line
        if position < self.lines.size:
            self.lines.remove_at(position)

    def get_line(self, position: int) -> str:
        """获取文本行"""
        current = self.lines.head
        for _ in range(position):
            current = current.next
        return current.val if current else ""
```

### 3. 栈应用场景
- **函数调用栈**: 递归调用管理
- **表达式求值**: 中缀转后缀
- **撤销操作**: 编辑器撤销功能
- **深度优先搜索**: DFS算法实现

```python
class ExpressionEvaluator:
    """表达式求值器 - 使用栈实现"""
    def __init__(self):
        self.operators = Stack()
        self.operands = Stack()

    def precedence(self, op: str) -> int:
        """返回运算符优先级"""
        if op in '+-':
            return 1
        if op in '*/':
            return 2
        return 0

    def apply_operator(self):
        """应用运算符"""
        operator = self.operators.pop()
        b = self.operands.pop()
        a = self.operands.pop()

        if operator == '+':
            result = a + b
        elif operator == '-':
            result = a - b
        elif operator == '*':
            result = a * b
        elif operator == '/':
            result = a / b

        self.operands.push(result)

    def evaluate(self, expression: str) -> float:
        """计算表达式值"""
        for char in expression:
            if char.isdigit():
                self.operands.push(int(char))
            elif char in '+-*/':
                while (not self.operators.is_empty() and
                       self.precedence(self.operators.peek()) >= self.precedence(char)):
                    self.apply_operator()
                self.operators.push(char)

        while not self.operators.is_empty():
            self.apply_operator()

        return self.operands.pop()
```

### 4. 队列应用场景
- **任务调度**: 操作系统进程调度
- **打印队列**: 打印任务管理
- **消息队列**: 系统间通信
- **广度优先搜索**: BFS算法实现

```python
class TaskScheduler:
    """任务调度器 - 使用队列实现"""
    def __init__(self):
        self.task_queue = Queue()
        self.current_task = None

    def add_task(self, task: str, priority: int = 0):
        """添加任务"""
        self.task_queue.enqueue((task, priority))

    def execute_next_task(self):
        """执行下一个任务"""
        if not self.task_queue.is_empty():
            self.current_task = self.task_queue.dequeue()
            task, priority = self.current_task
            print(f"执行任务: {task} (优先级: {priority})")
            return task
        return None

    def get_pending_tasks(self) -> int:
        """获取待处理任务数量"""
        return self.task_queue.size()
```

## 高级技巧与优化

### 1. 内存池管理
```python
class MemoryPool:
    """内存池 - 预分配内存提高性能"""
    def __init__(self, block_size: int, pool_size: int):
        self.block_size = block_size
        self.pool_size = pool_size
        self.free_blocks = Stack()

        # 预分配内存块
        for _ in range(pool_size):
            block = bytearray(block_size)
            self.free_blocks.push(block)

    def allocate(self) -> bytearray:
        """分配内存块"""
        if self.free_blocks.is_empty():
            raise MemoryError("Memory pool exhausted")
        return self.free_blocks.pop()

    def deallocate(self, block: bytearray):
        """释放内存块"""
        self.free_blocks.push(block)
```

### 2. 循环队列
```python
class CircularQueue:
    """循环队列 - 固定大小的高效队列"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.front = self.rear = -1
        self.size = 0

    def is_empty(self) -> bool:
        return self.size == 0

    def is_full(self) -> bool:
        return self.size == self.capacity

    def enqueue(self, item):
        """入队"""
        if self.is_full():
            raise IndexError("Queue is full")

        if self.is_empty():
            self.front = self.rear = 0
        else:
            self.rear = (self.rear + 1) % self.capacity

        self.queue[self.rear] = item
        self.size += 1

    def dequeue(self):
        """出队"""
        if self.is_empty():
            raise IndexError("Queue is empty")

        item = self.queue[self.front]
        self.front = (self.front + 1) % self.capacity
        self.size -= 1

        if self.is_empty():
            self.front = self.rear = -1

        return item
```

### 3. 缓存友好的数据结构
```python
class CacheFriendlyArray:
    """缓存友好的数组结构"""
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.size = 0
        self.data = [None] * capacity

    def batch_insert(self, items: list):
        """批量插入以提高缓存命中率"""
        for item in items:
            if self.size < self.capacity:
                self.data[self.size] = item
                self.size += 1

    def batch_access(self, indices: list) -> list:
        """批量访问以提高缓存命中率"""
        return [self.data[i] for i in indices if i < self.size]
```

## 练习

1. 实现一个双向链表及其常用操作
2. 创建一个支持最小值操作的栈（Min Stack）
3. 实现一个用两个栈模拟的队列
4. 设计一个LRU缓存，使用双向链表和哈希表
5. 实现一个跳表（Skip List）数据结构
6. 创建一个循环双端队列（Circular Deque）
7. 实现一个支持任意位置插入删除的动态数组
8. 设计一个内存高效的字符串连接器（String Builder）