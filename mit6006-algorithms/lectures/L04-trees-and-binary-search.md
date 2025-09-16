# L04 - 树与二叉搜索树

## 学习目标
- 掌握二叉树和二叉搜索树的基本概念
- 理解平衡树的重要性和实现原理
- 学会实现树的各种遍历算法
- 掌握树的插入、删除和查找操作

## 树的基本概念

### 术语定义
- **根节点**: 树的顶端节点
- **父节点**: 直接连接到子节点的节点
- **子节点**: 直接连接到父节点的节点
- **叶子节点**: 没有子节点的节点
- **高度**: 从根到叶子节点的最长路径
- **深度**: 从根到当前节点的路径长度
- **度数**: 节点拥有的子节点数量

## Python实现

### 1. 二叉树基础
```python
class TreeNode:
    """二叉树节点"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinaryTree:
    """
    二叉树实现
    时间复杂度:
    - 插入: O(h) h为树高度
    - 查找: O(h)
    - 删除: O(h)
    空间复杂度: O(n)
    """

    def __init__(self):
        self.root = None
        self.size = 0

    def insert(self, val):
        """插入节点（层次遍历插入）"""
        new_node = TreeNode(val)

        if not self.root:
            self.root = new_node
        else:
            queue = [self.root]
            while queue:
                current = queue.pop(0)

                if not current.left:
                    current.left = new_node
                    break
                elif not current.right:
                    current.right = new_node
                    break
                else:
                    queue.append(current.left)
                    queue.append(current.right)

        self.size += 1

    def inorder_traversal(self, node=None):
        """中序遍历"""
        if node is None:
            node = self.root

        result = []
        if node:
            result.extend(self.inorder_traversal(node.left))
            result.append(node.val)
            result.extend(self.inorder_traversal(node.right))
        return result

    def preorder_traversal(self, node=None):
        """前序遍历"""
        if node is None:
            node = self.root

        result = []
        if node:
            result.append(node.val)
            result.extend(self.preorder_traversal(node.left))
            result.extend(self.preorder_traversal(node.right))
        return result

    def postorder_traversal(self, node=None):
        """后序遍历"""
        if node is None:
            node = self.root

        result = []
        if node:
            result.extend(self.postorder_traversal(node.left))
            result.extend(self.postorder_traversal(node.right))
            result.append(node.val)
        return result

    def level_order_traversal(self):
        """层次遍历"""
        if not self.root:
            return []

        result = []
        queue = [self.root]

        while queue:
            level_size = len(queue)
            current_level = []

            for _ in range(level_size):
                current = queue.pop(0)
                current_level.append(current.val)

                if current.left:
                    queue.append(current.left)
                if current.right:
                    queue.append(current.right)

            result.append(current_level)

        return result

    def height(self, node=None):
        """计算树的高度"""
        if node is None:
            node = self.root

        if not node:
            return 0

        left_height = self.height(node.left)
        right_height = self.height(node.right)

        return max(left_height, right_height) + 1

    def is_balanced(self, node=None):
        """检查树是否平衡"""
        def check_balance(node):
            if not node:
                return 0, True

            left_height, left_balanced = check_balance(node.left)
            right_height, right_balanced = check_balance(node.right)

            current_height = max(left_height, right_height) + 1
            is_balanced = (left_balanced and right_balanced and
                          abs(left_height - right_height) <= 1)

            return current_height, is_balanced

        _, balanced = check_balance(node if node else self.root)
        return balanced
```

### 2. 二叉搜索树 (BST)
```python
class BSTNode:
    """二叉搜索树节点"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

class BinarySearchTree:
    """
    二叉搜索树实现
    时间复杂度:
    - 插入: O(h) 平均O(log n)，最坏O(n)
    - 查找: O(h) 平均O(log n)，最坏O(n)
    - 删除: O(h) 平均O(log n)，最坏O(n)
    空间复杂度: O(n)
    """

    def __init__(self):
        self.root = None
        self.size = 0

    def insert(self, val):
        """插入节点"""
        def insert_node(node, val):
            if not node:
                return BSTNode(val)

            if val < node.val:
                node.left = insert_node(node.left, val)
            elif val > node.val:
                node.right = insert_node(node.right, val)

            return node

        self.root = insert_node(self.root, val)
        self.size += 1

    def search(self, val):
        """查找节点"""
        def search_node(node, val):
            if not node or node.val == val:
                return node

            if val < node.val:
                return search_node(node.left, val)
            else:
                return search_node(node.right, val)

        return search_node(self.root, val)

    def delete(self, val):
        """删除节点"""
        def delete_node(node, val):
            if not node:
                return None

            if val < node.val:
                node.left = delete_node(node.left, val)
            elif val > node.val:
                node.right = delete_node(node.right, val)
            else:
                # 找到要删除的节点
                if not node.left:
                    return node.right
                elif not node.right:
                    return node.left
                else:
                    # 节点有两个子节点，找到右子树的最小值
                    min_right = self._find_min(node.right)
                    node.val = min_right.val
                    node.right = delete_node(node.right, min_right.val)

            return node

        self.root = delete_node(self.root, val)
        self.size -= 1

    def _find_min(self, node):
        """找到子树的最小值节点"""
        while node.left:
            node = node.left
        return node

    def _find_max(self, node):
        """找到子树的最大值节点"""
        while node.right:
            node = node.right
        return node

    def find_min(self):
        """查找最小值"""
        if not self.root:
            return None
        return self._find_min(self.root).val

    def find_max(self):
        """查找最大值"""
        if not self.root:
            return None
        return self._find_max(self.root).val

    def successor(self, val):
        """查找后继节点"""
        node = self.search(val)
        if not node:
            return None

        # 如果有右子树，后继是右子树的最小值
        if node.right:
            return self._find_min(node.right).val

        # 否则向上查找
        successor = None
        current = self.root

        while current:
            if val < current.val:
                successor = current
                current = current.left
            elif val > current.val:
                current = current.right
            else:
                break

        return successor.val if successor else None

    def predecessor(self, val):
        """查找前驱节点"""
        node = self.search(val)
        if not node:
            return None

        # 如果有左子树，前驱是左子树的最大值
        if node.left:
            return self._find_max(node.left).val

        # 否则向上查找
        predecessor = None
        current = self.root

        while current:
            if val > current.val:
                predecessor = current
                current = current.right
            elif val < current.val:
                current = current.left
            else:
                break

        return predecessor.val if predecessor else None

    def validate_bst(self):
        """验证是否为有效的二叉搜索树"""
        def validate(node, min_val=float('-inf'), max_val=float('inf')):
            if not node:
                return True

            if node.val <= min_val or node.val >= max_val:
                return False

            return (validate(node.left, min_val, node.val) and
                    validate(node.right, node.val, max_val))

        return validate(self.root)
```

### 3. AVL树 (平衡二叉搜索树)
```python
class AVLNode:
    """AVL树节点"""
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right
        self.height = 1  # 节点高度

class AVLTree:
    """
    AVL树实现
    时间复杂度:
    - 插入: O(log n)
    - 查找: O(log n)
    - 删除: O(log n)
    空间复杂度: O(n)
    """

    def __init__(self):
        self.root = None
        self.size = 0

    def _height(self, node):
        """获取节点高度"""
        if not node:
            return 0
        return node.height

    def _update_height(self, node):
        """更新节点高度"""
        if node:
            node.height = max(self._height(node.left), self._height(node.right)) + 1

    def _balance_factor(self, node):
        """计算平衡因子"""
        if not node:
            return 0
        return self._height(node.left) - self._height(node.right)

    def _left_rotate(self, z):
        """左旋"""
        y = z.right
        T2 = y.left

        # 旋转
        y.left = z
        z.right = T2

        # 更新高度
        self._update_height(z)
        self._update_height(y)

        return y

    def _right_rotate(self, z):
        """右旋"""
        y = z.left
        T3 = y.right

        # 旋转
        y.right = z
        z.left = T3

        # 更新高度
        self._update_height(z)
        self._update_height(y)

        return y

    def _balance(self, node):
        """平衡节点"""
        if not node:
            return node

        # 更新高度
        self._update_height(node)

        # 获取平衡因子
        balance = self._balance_factor(node)

        # 左左情况
        if balance > 1 and self._balance_factor(node.left) >= 0:
            return self._right_rotate(node)

        # 右右情况
        if balance < -1 and self._balance_factor(node.right) <= 0:
            return self._left_rotate(node)

        # 左右情况
        if balance > 1 and self._balance_factor(node.left) < 0:
            node.left = self._left_rotate(node.left)
            return self._right_rotate(node)

        # 右左情况
        if balance < -1 and self._balance_factor(node.right) > 0:
            node.right = self._right_rotate(node.right)
            return self._left_rotate(node)

        return node

    def insert(self, val):
        """插入节点"""
        def insert_node(node, val):
            if not node:
                return AVLNode(val)

            if val < node.val:
                node.left = insert_node(node.left, val)
            elif val > node.val:
                node.right = insert_node(node.right, val)
            else:
                return node  # 重复值不插入

            return self._balance(node)

        self.root = insert_node(self.root, val)
        self.size += 1

    def delete(self, val):
        """删除节点"""
        def delete_node(node, val):
            if not node:
                return None

            if val < node.val:
                node.left = delete_node(node.left, val)
            elif val > node.val:
                node.right = delete_node(node.right, val)
            else:
                # 找到要删除的节点
                if not node.left:
                    return node.right
                elif not node.right:
                    return node.left
                else:
                    # 节点有两个子节点
                    min_right = self._find_min(node.right)
                    node.val = min_right.val
                    node.right = delete_node(node.right, min_right.val)

            return self._balance(node)

        self.root = delete_node(self.root, val)
        self.size -= 1

    def _find_min(self, node):
        """找到子树的最小值节点"""
        while node and node.left:
            node = node.left
        return node

    def search(self, val):
        """查找节点"""
        def search_node(node, val):
            if not node or node.val == val:
                return node

            if val < node.val:
                return search_node(node.left, val)
            else:
                return search_node(node.right, val)

        return search_node(self.root, val)
```

### 4. 红黑树
```python
class Color:
    RED = 'RED'
    BLACK = 'BLACK'

class RBNode:
    """红黑树节点"""
    def __init__(self, val=0, color=Color.RED, left=None, right=None, parent=None):
        self.val = val
        self.color = color
        self.left = left
        self.right = right
        self.parent = parent

class RedBlackTree:
    """
    红黑树实现
    时间复杂度:
    - 插入: O(log n)
    - 查找: O(log n)
    - 删除: O(log n)
    空间复杂度: O(n)
    """

    def __init__(self):
        self.NIL = RBNode(0, Color.BLACK)  # NIL节点
        self.root = self.NIL
        self.size = 0

    def insert(self, val):
        """插入节点"""
        def insert_node(node, parent, val):
            if node == self.NIL:
                new_node = RBNode(val, Color.RED, self.NIL, self.NIL, parent)
                if not parent:
                    self.root = new_node
                elif val < parent.val:
                    parent.left = new_node
                else:
                    parent.right = new_node
                return new_node

            if val < node.val:
                return insert_node(node.left, node, val)
            elif val > node.val:
                return insert_node(node.right, node, val)
            else:
                return None  # 重复值不插入

        new_node = insert_node(self.root, None, val)
        if new_node:
            self.size += 1
            self._fix_insert(new_node)

    def _fix_insert(self, node):
        """修复插入后的红黑树性质"""
        while node != self.root and node.parent.color == Color.RED:
            if node.parent == node.parent.parent.left:
                uncle = node.parent.parent.right

                # 情况1：叔叔是红色
                if uncle.color == Color.RED:
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    # 情况2：叔叔是黑色，节点是右孩子
                    if node == node.parent.right:
                        node = node.parent
                        self._left_rotate(node)

                    # 情况3：叔叔是黑色，节点是左孩子
                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._right_rotate(node.parent.parent)
            else:
                # 对称情况
                uncle = node.parent.parent.left

                if uncle.color == Color.RED:
                    node.parent.color = Color.BLACK
                    uncle.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    node = node.parent.parent
                else:
                    if node == node.parent.left:
                        node = node.parent
                        self._right_rotate(node)

                    node.parent.color = Color.BLACK
                    node.parent.parent.color = Color.RED
                    self._left_rotate(node.parent.parent)

        self.root.color = Color.BLACK

    def _left_rotate(self, x):
        """左旋"""
        y = x.right
        x.right = y.left

        if y.left != self.NIL:
            y.left.parent = x

        y.parent = x.parent

        if x.parent is None:
            self.root = y
        elif x == x.parent.left:
            x.parent.left = y
        else:
            x.parent.right = y

        y.left = x
        x.parent = y

    def _right_rotate(self, y):
        """右旋"""
        x = y.left
        y.left = x.right

        if x.right != self.NIL:
            x.right.parent = y

        x.parent = y.parent

        if y.parent is None:
            self.root = x
        elif y == y.parent.right:
            y.parent.right = x
        else:
            y.parent.left = x

        x.right = y
        y.parent = x

    def search(self, val):
        """查找节点"""
        def search_node(node, val):
            if node == self.NIL or node.val == val:
                return node

            if val < node.val:
                return search_node(node.left, val)
            else:
                return search_node(node.right, val)

        result = search_node(self.root, val)
        return result if result != self.NIL else None
```

### 5. B树
```python
class BTreeNode:
    """B树节点"""
    def __init__(self, leaf=False):
        self.leaf = leaf
        self.keys = []
        self.children = []

class BTree:
    """
    B树实现
    时间复杂度:
    - 插入: O(log n)
    - 查找: O(log n)
    - 删除: O(log n)
    空间复杂度: O(n)
    """

    def __init__(self, t):
        """
        初始化B树
        t: 最小度数，每个节点至少有t-1个key，最多有2t-1个key
        """
        self.root = BTreeNode(leaf=True)
        self.t = t  # 最小度数

    def search(self, key):
        """查找key"""
        return self._search(self.root, key)

    def _search(self, node, key):
        """在节点中查找key"""
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            return True
        elif node.leaf:
            return False
        else:
            return self._search(node.children[i], key)

    def insert(self, key):
        """插入key"""
        root = self.root
        if len(root.keys) == 2 * self.t - 1:
            # 根节点已满，创建新根节点
            new_root = BTreeNode()
            new_root.children.append(self.root)
            self.root = new_root
            self._split_child(new_root, 0)
            self._insert_nonfull(new_root, key)
        else:
            self._insert_nonfull(root, key)

    def _split_child(self, parent, index):
        """分裂子节点"""
        t = self.t
        full_node = parent.children[index]
        new_node = BTreeNode(leaf=full_node.leaf)

        # 将full_node的keys和children分裂
        parent.keys.insert(index, full_node.keys[t - 1])
        parent.children.insert(index + 1, new_node)

        new_node.keys = full_node.keys[t:]
        full_node.keys = full_node.keys[:t - 1]

        if not full_node.leaf:
            new_node.children = full_node.children[t:]
            full_node.children = full_node.children[:t]

    def _insert_nonfull(self, node, key):
        """向非满节点插入key"""
        i = len(node.keys) - 1

        if node.leaf:
            # 叶子节点，直接插入
            node.keys.append(0)
            while i >= 0 and key < node.keys[i]:
                node.keys[i + 1] = node.keys[i]
                i -= 1
            node.keys[i + 1] = key
        else:
            # 内部节点，找到合适的子节点
            while i >= 0 and key < node.keys[i]:
                i -= 1
            i += 1

            if len(node.children[i].keys) == 2 * self.t - 1:
                # 子节点已满，分裂
                self._split_child(node, i)
                if key > node.keys[i]:
                    i += 1

            self._insert_nonfull(node.children[i], key)

    def delete(self, key):
        """删除key"""
        self._delete(self.root, key)

    def _delete(self, node, key):
        """在节点中删除key"""
        t = self.t
        i = 0
        while i < len(node.keys) and key > node.keys[i]:
            i += 1

        if i < len(node.keys) and key == node.keys[i]:
            # 找到key
            if node.leaf:
                # 叶子节点，直接删除
                node.keys.pop(i)
            else:
                # 内部节点
                if len(node.children[i].keys) >= t:
                    # 左子节点有足够的keys
                    predecessor = self._get_predecessor(node.children[i])
                    node.keys[i] = predecessor
                    self._delete(node.children[i], predecessor)
                elif len(node.children[i + 1].keys) >= t:
                    # 右子节点有足够的keys
                    successor = self._get_successor(node.children[i + 1])
                    node.keys[i] = successor
                    self._delete(node.children[i + 1], successor)
                else:
                    # 合并子节点
                    self._merge_children(node, i)
                    self._delete(node.children[i], key)
        else:
            # 未找到key
            if node.leaf:
                return  # key不存在

            # 确保子节点有足够的keys
            if i < len(node.children) and len(node.children[i].keys) == t - 1:
                self._fix_child(node, i)

            # 递归删除
            if i == len(node.keys):
                self._delete(node.children[i - 1], key)
            else:
                self._delete(node.children[i], key)

    def _get_predecessor(self, node):
        """获取前驱节点"""
        while not node.leaf:
            node = node.children[-1]
        return node.keys[-1]

    def _get_successor(self, node):
        """获取后继节点"""
        while not node.leaf:
            node = node.children[0]
        return node.keys[0]

    def _merge_children(self, node, i):
        """合并子节点"""
        t = self.t
        left_child = node.children[i]
        right_child = node.children[i + 1]

        # 将父节点的key下移
        left_child.keys.append(node.keys[i])

        # 合并右子节点的keys和children
        left_child.keys.extend(right_child.keys)
        left_child.children.extend(right_child.children)

        # 删除父节点的key和右子节点
        node.keys.pop(i)
        node.children.pop(i + 1)

    def _fix_child(self, node, i):
        """修复子节点"""
        t = self.t

        # 尝试从左兄弟节点借key
        if i > 0 and len(node.children[i - 1].keys) >= t:
            self._borrow_from_left(node, i)
        # 尝试从右兄弟节点借key
        elif i < len(node.keys) and len(node.children[i + 1].keys) >= t:
            self._borrow_from_right(node, i)
        # 合并兄弟节点
        else:
            if i < len(node.keys):
                self._merge_children(node, i)
            else:
                self._merge_children(node, i - 1)

    def _borrow_from_left(self, node, i):
        """从左兄弟节点借key"""
        child = node.children[i]
        left_sibling = node.children[i - 1]

        # 将父节点的key下移
        child.keys.insert(0, node.keys[i - 1])
        node.keys[i - 1] = left_sibling.keys.pop()

        # 移动children
        if not left_sibling.leaf:
            child.children.insert(0, left_sibling.children.pop())

    def _borrow_from_right(self, node, i):
        """从右兄弟节点借key"""
        child = node.children[i]
        right_sibling = node.children[i + 1]

        # 将父节点的key下移
        child.keys.append(node.keys[i])
        node.keys[i] = right_sibling.keys.pop(0)

        # 移动children
        if not right_sibling.leaf:
            child.children.append(right_sibling.children.pop(0))

    def inorder_traversal(self):
        """中序遍历"""
        result = []
        self._inorder(self.root, result)
        return result

    def _inorder(self, node, result):
        """中序遍历辅助函数"""
        if node:
            for i in range(len(node.keys)):
                if not node.leaf:
                    self._inorder(node.children[i], result)
                result.append(node.keys[i])
            if not node.leaf:
                self._inorder(node.children[-1], result)
```

## 性能测试与比较

```python
import time
import random
import matplotlib.pyplot as plt

def benchmark_trees():
    """树结构性能测试"""
    sizes = [1000, 5000, 10000, 20000]

    for size in sizes:
        print(f"\n=== 测试大小: {size} ===")

        # 生成随机数据
        data = [random.randint(1, 100000) for _ in range(size)]

        # 测试二叉搜索树
        bst = BinarySearchTree()
        start_time = time.time()
        for val in data:
            bst.insert(val)
        bst_insert_time = time.time() - start_time

        # 测试AVL树
        avl = AVLTree()
        start_time = time.time()
        for val in data:
            avl.insert(val)
        avl_insert_time = time.time() - start_time

        # 测试红黑树
        rbt = RedBlackTree()
        start_time = time.time()
        for val in data:
            rbt.insert(val)
        rbt_insert_time = time.time() - start_time

        # 测试B树
        bt = BTree(3)  # t=3
        start_time = time.time()
        for val in data:
            bt.insert(val)
        bt_insert_time = time.time() - start_time

        print(f"BST 插入: {bst_insert_time:.6f}s")
        print(f"AVL 插入: {avl_insert_time:.6f}s")
        print(f"红黑树 插入: {rbt_insert_time:.6f}s")
        print(f"B树 插入: {bt_insert_time:.6f}s")

        # 测试查找性能
        search_keys = [random.choice(data) for _ in range(1000)]

        start_time = time.time()
        for key in search_keys:
            bst.search(key)
        bst_search_time = time.time() - start_time

        start_time = time.time()
        for key in search_keys:
            avl.search(key)
        avl_search_time = time.time() - start_time

        start_time = time.time()
        for key in search_keys:
            rbt.search(key)
        rbt_search_time = time.time() - start_time

        start_time = time.time()
        for key in search_keys:
            bt.search(key)
        bt_search_time = time.time() - start_time

        print(f"BST 查找: {bst_search_time:.6f}s")
        print(f"AVL 查找: {avl_search_time:.6f}s")
        print(f"红黑树 查找: {rbt_search_time:.6f}s")
        print(f"B树 查找: {bt_search_time:.6f}s")

        # 检查树的高度
        print(f"BST 高度: {bst.height()}")
        print(f"AVL 是否平衡: {avl.is_balanced()}")
```

## 应用场景

### 1. 数据库索引
- **B+树**: 数据库索引的标准实现
- **B树**: 文件系统和数据库索引
- **哈希索引**: 内存数据库快速查找

### 2. 编译器实现
- **语法树**: 抽象语法树（AST）
- **符号表**: 变量和函数的快速查找
- **优化树**: 代码优化决策树

### 3. 网络路由
- **路由表**: IP路由的快速查找
- **前缀树**: IP地址的前缀匹配
- **平衡树**: 路由信息的动态维护

### 4. 游戏开发
- **场景图**: 游戏对象层次结构
- **空间划分**: 四叉树、八叉树
- **行为树**: AI决策系统

## 高级技巧与优化

### 1. 线程安全的树
```python
import threading

class ThreadSafeBST:
    """线程安全的二叉搜索树"""
    def __init__(self):
        self.root = None
        self.lock = threading.RLock()

    def insert(self, val):
        """线程安全的插入"""
        with self.lock:
            def insert_node(node, val):
                if not node:
                    return BSTNode(val)
                if val < node.val:
                    node.left = insert_node(node.left, val)
                elif val > node.val:
                    node.right = insert_node(node.right, val)
                return node

            self.root = insert_node(self.root, val)

    def search(self, val):
        """线程安全的查找"""
        with self.lock:
            def search_node(node, val):
                if not node or node.val == val:
                    return node
                if val < node.val:
                    return search_node(node.left, val)
                return search_node(node.right, val)

            return search_node(self.root, val)
```

### 2. 持久化树
```python
import pickle
import os

class PersistentBST:
    """持久化的二叉搜索树"""
    def __init__(self, filename="bst_data.pkl"):
        self.filename = filename
        self.root = self._load()

    def _load(self):
        """从文件加载树"""
        if os.path.exists(self.filename):
            with open(self.filename, 'rb') as f:
                return pickle.load(f)
        return None

    def _save(self):
        """保存树到文件"""
        with open(self.filename, 'wb') as f:
            pickle.dump(self.root, f)

    def insert(self, val):
        """插入并持久化"""
        def insert_node(node, val):
            if not node:
                return BSTNode(val)
            if val < node.val:
                node.left = insert_node(node.left, val)
            elif val > node.val:
                node.right = insert_node(node.right, val)
            return node

        self.root = insert_node(self.root, val)
        self._save()
```

### 3. 内存优化
```python
class CompactBST:
    """内存优化的二叉搜索树"""
    def __init__(self):
        self.root = None
        self.node_pool = []  # 节点池

    def _create_node(self, val):
        """从节点池创建节点"""
        if self.node_pool:
            node = self.node_pool.pop()
            node.val = val
            node.left = node.right = None
            return node
        return BSTNode(val)

    def _free_node(self, node):
        """释放节点到池"""
        if node:
            self.node_pool.append(node)

    def delete(self, val):
        """删除并回收内存"""
        def delete_node(node, val):
            if not node:
                return None

            if val < node.val:
                node.left = delete_node(node.left, val)
            elif val > node.val:
                node.right = delete_node(node.right, val)
            else:
                if not node.left:
                    temp = node.right
                    self._free_node(node)
                    return temp
                elif not node.right:
                    temp = node.left
                    self._free_node(node)
                    return temp
                else:
                    min_right = self._find_min(node.right)
                    node.val = min_right.val
                    node.right = delete_node(node.right, min_right.val)

            return node

        self.root = delete_node(self.root, val)
```

## 练习

1. 实现一个支持重复值的二叉搜索树
2. 创建一个自平衡的二叉搜索树（Splay Tree）
3. 实现一个支持范围查询的树结构
4. 设计一个支持最近邻搜索的k-d树
5. 实现一个线程安全的AVL树
6. 创建一个支持序列化和反序列化的树
7. 实现一个支持懒惰删除的树结构
8. 设计一个支持统计功能的增强树结构