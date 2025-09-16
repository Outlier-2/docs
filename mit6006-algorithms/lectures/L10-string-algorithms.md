# L10 - 字符串算法

## 学习目标
- 掌握字符串匹配的基本算法
- 理解字符串前缀处理的数据结构
- 学会处理复杂的字符串操作问题
- 能够应用字符串算法解决实际问题

## 字符串算法基础

### 基本概念
- **模式匹配**: 在文本中查找模式字符串
- **前缀函数**: 字符串的最长公共前后缀
- **后缀数组**: 字符串所有后缀的排序数组
- **后缀自动机**: 字符串的有限状态自动机

### 算法分类
- **精确匹配**: KMP、Boyer-Moore、Rabin-Karp
- **模糊匹配**: 编辑距离、正则表达式
- **多模式匹配**: AC自动机、后缀树
- **压缩算法**: LZW、哈夫曼编码

## Python实现

### 1. 基础字符串匹配

#### 1.1 朴素字符串匹配
```python
def naive_string_matching(text: str, pattern: str) -> list:
    """
    朴素字符串匹配算法
    时间复杂度: O(n * m)
    空间复杂度: O(1)
    """
    n = len(text)
    m = len(pattern)
    matches = []

    for i in range(n - m + 1):
        if text[i:i + m] == pattern:
            matches.append(i)

    return matches

def naive_string_matching_optimized(text: str, pattern: str) -> list:
    """
    优化的朴素字符串匹配（提前终止）
    """
    n = len(text)
    m = len(pattern)
    matches = []

    for i in range(n - m + 1):
        match = True
        for j in range(m):
            if text[i + j] != pattern[j]:
                match = False
                break
        if match:
            matches.append(i)

    return matches
```

#### 1.2 KMP算法
```python
def compute_lps_array(pattern: str) -> list:
    """
    计算最长公共前后缀数组
    时间复杂度: O(m)
    空间复杂度: O(m)
    """
    m = len(pattern)
    lps = [0] * m
    length = 0  # 当前最长公共前后缀长度
    i = 1

    while i < m:
        if pattern[i] == pattern[length]:
            length += 1
            lps[i] = length
            i += 1
        else:
            if length != 0:
                length = lps[length - 1]
            else:
                lps[i] = 0
                i += 1

    return lps

def kmp_search(text: str, pattern: str) -> list:
    """
    KMP字符串匹配算法
    时间复杂度: O(n + m)
    空间复杂度: O(m)
    """
    n = len(text)
    m = len(pattern)
    matches = []

    if m == 0:
        return []

    lps = compute_lps_array(pattern)
    i = 0  # text的索引
    j = 0  # pattern的索引

    while i < n:
        if pattern[j] == text[i]:
            i += 1
            j += 1

            if j == m:
                matches.append(i - j)
                j = lps[j - 1]
        else:
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1

    return matches
```

#### 1.3 Boyer-Moore算法
```python
def bad_character_heuristic(pattern: str) -> dict:
    """
    坏字符启发式函数
    """
    bad_char = {}
    m = len(pattern)

    for i in range(m - 1):
        bad_char[pattern[i]] = m - i - 1

    return bad_char

def good_suffix_heuristic(pattern: str) -> list:
    """
    好后缀启发式函数
    """
    m = len(pattern)
    good_suffix = [0] * (m + 1)

    # 情况1：pattern中存在匹配的后缀
    suffix = compute_lps_array(pattern[::-1])[::-1]

    # 情况2：pattern的前缀匹配后缀的一部分
    for i in range(m):
        good_suffix[i] = m - suffix[i] if suffix[i] > 0 else m

    return good_suffix

def boyer_moore_search(text: str, pattern: str) -> list:
    """
    Boyer-Moore字符串匹配算法
    时间复杂度: O(n/m) 最佳情况, O(n*m) 最坏情况
    空间复杂度: O(m + |Σ|)
    """
    n = len(text)
    m = len(pattern)
    matches = []

    if m == 0:
        return []

    bad_char = bad_character_heuristic(pattern)
    good_suffix = good_suffix_heuristic(pattern)

    i = 0  # text的起始位置
    while i <= n - m:
        j = m - 1  # pattern的起始位置（从后往前）

        while j >= 0 and pattern[j] == text[i + j]:
            j -= 1

        if j < 0:
            matches.append(i)
            # 完全匹配，移动pattern
            i += good_suffix[0]
        else:
            # 根据坏字符和好后缀规则移动
            bad_char_shift = bad_char.get(text[i + j], m)
            good_suffix_shift = good_suffix[j + 1]
            i += max(bad_char_shift, good_suffix_shift)

    return matches
```

#### 1.4 Rabin-Karp算法
```python
def rabin_karp_search(text: str, pattern: str, prime: int = 101) -> list:
    """
    Rabin-Karp字符串匹配算法
    时间复杂度: O(n + m) 平均情况
    空间复杂度: O(1)
    """
    n = len(text)
    m = len(pattern)
    matches = []

    if m == 0 or n < m:
        return []

    # 计算散列函数
    d = 256  # 字符集大小
    h = pow(d, m - 1, prime)
    p = 0  # pattern的散列值
    t = 0  # text窗口的散列值

    # 计算pattern和第一个text窗口的散列值
    for i in range(m):
        p = (d * p + ord(pattern[i])) % prime
        t = (d * t + ord(text[i])) % prime

    for i in range(n - m + 1):
        # 检查散列值是否匹配
        if p == t:
            # 确认字符是否匹配
            if text[i:i + m] == pattern:
                matches.append(i)

        # 计算下一个窗口的散列值
        if i < n - m:
            t = (d * (t - ord(text[i]) * h) + ord(text[i + m])) % prime
            if t < 0:
                t += prime

    return matches
```

### 2. 高级字符串数据结构

#### 2.1 后缀数组
```python
def build_suffix_array(s: str) -> list:
    """
    构建后缀数组
    时间复杂度: O(n log n)
    空间复杂度: O(n)
    """
    n = len(s)
    suffixes = []

    # 创建后缀数组
    for i in range(n):
        suffixes.append((s[i:], i))

    # 排序后缀
    suffixes.sort()

    # 提取起始索引
    return [suffix[1] for suffix in suffixes]

def build_suffix_array_optimized(s: str) -> list:
    """
    优化的后缀数组构建
    """
    n = len(s)
    rank = [ord(c) for c in s]
    k = 1

    def get_rank(i):
        return rank[i] if i < n else -1

    suffix_array = list(range(n))

    while k < n:
        # 排序依据：(rank[i], rank[i+k])
        suffix_array.sort(key=lambda i: (get_rank(i), get_rank(i + k)))

        # 重新计算rank
        new_rank = [0] * n
        new_rank[suffix_array[0]] = 0
        r = 0

        for i in range(1, n):
            r += (get_rank(suffix_array[i]) != get_rank(suffix_array[i-1]) or
                  get_rank(suffix_array[i] + k) != get_rank(suffix_array[i-1] + k))
            new_rank[suffix_array[i]] = r

        rank = new_rank
        k *= 2

    return suffix_array

def build_lcp_array(s: str, suffix_array: list) -> list:
    """
    构建最长公共前缀数组
    """
    n = len(s)
    rank = [0] * n
    lcp = [0] * n

    # 构建rank数组
    for i, suffix_idx in enumerate(suffix_array):
        rank[suffix_idx] = i

    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = suffix_array[rank[i] - 1]

            # 计算公共前缀
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1

            lcp[rank[i]] = h

            if h > 0:
                h -= 1

    return lcp
```

#### 2.2 前缀树 (Trie)
```python
class TrieNode:
    """前缀树节点"""
    def __init__(self):
        self.children = {}
        self.is_end = False
        self.count = 0

class Trie:
    """前缀树实现"""
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        """插入单词"""
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True
        node.count += 1

    def search(self, word: str) -> bool:
        """搜索单词"""
        node = self.root
        for char in word:
            if char not in node.children:
                return False
            node = node.children[char]
        return node.is_end

    def starts_with(self, prefix: str) -> bool:
        """检查前缀"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return False
            node = node.children[char]
        return True

    def delete(self, word: str):
        """删除单词"""
        def _delete_helper(node: TrieNode, word: str, index: int) -> bool:
            if index == len(word):
                if node.is_end:
                    node.is_end = False
                    node.count = 0
                    return len(node.children) == 0
                return False

            char = word[index]
            if char not in node.children:
                return False

            should_delete = _delete_helper(node.children[char], word, index + 1)

            if should_delete:
                del node.children[char]
                return len(node.children) == 0 and not node.is_end

            return False

        _delete_helper(self.root, word, 0)

    def get_all_words_with_prefix(self, prefix: str) -> list:
        """获取所有以prefix开头的单词"""
        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        words = []

        def _dfs(node: TrieNode, current_word: str):
            if node.is_end:
                words.append(current_word)

            for char, child in node.children.items():
                _dfs(child, current_word + char)

        _dfs(node, prefix)
        return words

    def get_word_count(self, word: str) -> int:
        """获取单词出现次数"""
        node = self.root
        for char in word:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.count if node.is_end else 0
```

#### 2.3 后缀树
```python
class SuffixTreeNode:
    """后缀树节点"""
    def __init__(self):
        self.children = {}
        self.suffix_link = None
        self.start = None
        self.end = None
        self.suffix_index = -1

class SuffixTree:
    """后缀树实现"""
    def __init__(self, text: str):
        self.text = text
        self.root = SuffixTreeNode()
        self.root.suffix_link = self.root
        self.active_node = self.root
        self.active_edge = -1
        self.active_length = 0
        self.remaining = 0
        self.end = -1
        self.root.end = -1
        self.build_suffix_tree()

    def edge_length(self, node: SuffixTreeNode) -> int:
        """计算边长度"""
        if node == self.root:
            return 0
        return node.end - node.start + 1

    def walk_down(self, node: SuffixTreeNode) -> bool:
        """沿着边向下走"""
        length = self.edge_length(node)
        if self.active_length >= length:
            self.active_edge += length
            self.active_length -= length
            self.active_node = node
            return True
        return False

    def extend_suffix_tree(self, pos: int):
        """扩展后缀树"""
        self.end = pos
        self.remaining += 1
        last_new_node = None

        while self.remaining > 0:
            if self.active_length == 0:
                self.active_edge = pos

            if self.text[self.active_edge] not in self.active_node.children:
                # 创建叶子节点
                leaf = SuffixTreeNode()
                leaf.start = pos
                leaf.end = self.end
                leaf.suffix_index = pos
                self.active_node.children[self.text[self.active_edge]] = leaf

                if last_new_node is not None:
                    last_new_node.suffix_link = self.active_node
                    last_new_node = None
            else:
                next_node = self.active_node.children[self.text[self.active_edge]]

                if self.walk_down(next_node):
                    continue

                if self.text[next_node.start + self.active_length] == self.text[pos]:
                    if last_new_node is not None and self.active_node != self.root:
                        last_new_node.suffix_link = self.active_node
                        last_new_node = None

                    self.active_length += 1
                    break

                # 创建内部节点
                split = SuffixTreeNode()
                split.start = next_node.start
                split.end = next_node.start + self.active_length - 1

                self.active_node.children[self.text[self.active_edge]] = split

                # 更新next_node
                next_node.start += self.active_length
                split.children[self.text[next_node.start]] = next_node

                # 创建叶子节点
                leaf = SuffixTreeNode()
                leaf.start = pos
                leaf.end = self.end
                leaf.suffix_index = pos
                split.children[self.text[pos]] = leaf

                if last_new_node is not None:
                    last_new_node.suffix_link = split

                last_new_node = split

            self.remaining -= 1

            if self.active_node == self.root and self.active_length > 0:
                self.active_length -= 1
                self.active_edge = pos - self.remaining + 1
            elif self.active_node != self.root:
                self.active_node = self.active_node.suffix_link

    def build_suffix_tree(self):
        """构建后缀树"""
        n = len(self.text)
        for i in range(n):
            self.extend_suffix_tree(i)

    def search(self, pattern: str) -> list:
        """搜索模式"""
        node = self.root
        i = 0

        while i < len(pattern):
            if pattern[i] not in node.children:
                return []

            node = node.children[pattern[i]]
            edge_start = node.start
            edge_end = node.end if node.end != -1 else len(self.text) - 1

            # 沿着边比较
            j = 0
            while i < len(pattern) and j <= edge_end - edge_start:
                if pattern[i] != self.text[edge_start + j]:
                    return []
                i += 1
                j += 1

            if i < len(pattern) and j > edge_end - edge_start:
                # 需要继续向下
                continue
            elif i == len(pattern):
                # 找到匹配，收集所有叶子节点
                matches = []
                self._collect_leaf_indices(node, matches)
                return matches
            else:
                return []

        return []

    def _collect_leaf_indices(self, node: SuffixTreeNode, matches: list):
        """收集叶子节点的后缀索引"""
        if node.suffix_index != -1:
            matches.append(node.suffix_index)
        else:
            for child in node.children.values():
                self._collect_leaf_indices(child, matches)
```

### 3. AC自动机

```python
class ACNode:
    """AC自动机节点"""
    def __init__(self):
        self.children = {}
        self.fail = None
        self.output = set()
        self.pattern_id = -1

class AhoCorasick:
    """Aho-Corasick多模式匹配算法"""
    def __init__(self):
        self.root = ACNode()

    def add_pattern(self, pattern: str, pattern_id: int = None):
        """添加模式"""
        node = self.root
        for char in pattern:
            if char not in node.children:
                node.children[char] = ACNode()
            node = node.children[char]

        if pattern_id is None:
            node.pattern_id = len(node.output)
        else:
            node.pattern_id = pattern_id
        node.output.add(node.pattern_id)

    def build_fail_links(self):
        """构建失败指针"""
        queue = [self.root]
        self.root.fail = self.root

        while queue:
            current = queue.pop(0)

            for char, child in current.children.items():
                if current == self.root:
                    child.fail = self.root
                else:
                    fail_node = current.fail
                    while fail_node != self.root and char not in fail_node.children:
                        fail_node = fail_node.fail

                    if char in fail_node.children:
                        child.fail = fail_node.children[char]
                    else:
                        child.fail = self.root

                # 继承output
                child.output.update(child.fail.output)
                queue.append(child)

    def search(self, text: str) -> dict:
        """在文本中搜索所有模式"""
        result = {}
        current = self.root

        for i, char in enumerate(text):
            while current != self.root and char not in current.children:
                current = current.fail

            if char in current.children:
                current = current.children[char]
            else:
                current = self.root

            for pattern_id in current.output:
                if pattern_id not in result:
                    result[pattern_id] = []
                result[pattern_id].append(i)

        return result
```

### 4. 字符串压缩算法

#### 4.1 LZW压缩
```python
def lzw_compress(text: str) -> list:
    """
    LZW压缩算法
    """
    dictionary = {chr(i): i for i in range(256)}
    next_code = 256
    compressed = []
    current_code = ""

    for char in text:
        current_code += char
        if current_code not in dictionary:
            dictionary[current_code[len(current_code)-1]] = next_code
            next_code += 1
            compressed.append(dictionary[current_code[:-1]])
            current_code = char

    if current_code:
        compressed.append(dictionary[current_code])

    return compressed

def lzw_decompress(compressed: list) -> str:
    """
    LZW解压缩算法
    """
    dictionary = {i: chr(i) for i in range(256)}
    next_code = 256
    decompressed = ""
    prev_code = compressed[0]
    decompressed += dictionary[prev_code]

    for code in compressed[1:]:
        if code in dictionary:
            current_string = dictionary[code]
        elif code == next_code:
            current_string = dictionary[prev_code] + dictionary[prev_code][0]
        else:
            raise ValueError("Invalid compressed data")

        decompressed += current_string

        dictionary[next_code] = dictionary[prev_code] + current_string[0]
        next_code += 1
        prev_code = code

    return decompressed
```

#### 4.2 游程编码
```python
def run_length_encode(text: str) -> str:
    """
    游程编码
    """
    if not text:
        return ""

    encoded = []
    count = 1

    for i in range(1, len(text)):
        if text[i] == text[i-1]:
            count += 1
        else:
            encoded.append(f"{count}{text[i-1]}")
            count = 1

    encoded.append(f"{count}{text[-1]}")
    return "".join(encoded)

def run_length_decode(encoded: str) -> str:
    """
    游程解码
    """
    if not encoded:
        return ""

    decoded = []
    i = 0

    while i < len(encoded):
        count_str = ""
        while i < len(encoded) and encoded[i].isdigit():
            count_str += encoded[i]
            i += 1

        if count_str and i < len(encoded):
            count = int(count_str)
            char = encoded[i]
            decoded.append(char * count)
            i += 1

    return "".join(decoded)
```

### 5. 字符串相似度

#### 5.1 编辑距离
```python
def levenshtein_distance(s1: str, s2: str) -> int:
    """
    Levenshtein距离（编辑距离）
    """
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    previous_row = list(range(len(s2) + 1))

    for i, c1 in enumerate(s1):
        current_row = [i + 1]

        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        previous_row = current_row

    return previous_row[-1]

def damerau_levenshtein_distance(s1: str, s2: str) -> int:
    """
    Damerau-Levenshtein距离（包含相邻字符交换）
    """
    if len(s1) < len(s2):
        return damerau_levenshtein_distance(s2, s1)

    if len(s2) == 0:
        return len(s1)

    # 动态规划表
    dp = [[0] * (len(s2) + 1) for _ in range(len(s1) + 1)]

    # 初始化
    for i in range(len(s1) + 1):
        dp[i][0] = i
    for j in range(len(s2) + 1):
        dp[0][j] = j

    for i in range(1, len(s1) + 1):
        for j in range(1, len(s2) + 1):
            cost = 0 if s1[i-1] == s2[j-1] else 1
            dp[i][j] = min(
                dp[i-1][j] + 1,     # 删除
                dp[i][j-1] + 1,     # 插入
                dp[i-1][j-1] + cost  # 替换
            )

            # 检查相邻字符交换
            if i > 1 and j > 1 and s1[i-1] == s2[j-2] and s1[i-2] == s2[j-1]:
                dp[i][j] = min(dp[i][j], dp[i-2][j-2] + cost)

    return dp[len(s1)][len(s2)]
```

#### 5.2 最长公共子序列
```python
def longest_common_subsequence(s1: str, s2: str) -> str:
    """
    最长公共子序列
    """
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # 重构LCS
    lcs = []
    i, j = m, n
    while i > 0 and j > 0:
        if s1[i-1] == s2[j-1]:
            lcs.append(s1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    lcs.reverse()
    return "".join(lcs)
```

## 性能测试与比较

```python
import time
import random
import string

def generate_random_string(length: int) -> str:
    """生成随机字符串"""
    return ''.join(random.choices(string.ascii_lowercase, k=length))

def benchmark_string_algorithms():
    """字符串算法性能测试"""
    # 生成测试数据
    text_length = 1000000
    pattern_length = 100
    text = generate_random_string(text_length)
    pattern = generate_random_string(pattern_length)

    print(f"文本长度: {text_length}, 模式长度: {pattern_length}")

    # 测试朴素匹配
    start_time = time.time()
    naive_matches = naive_string_matching(text, pattern)
    naive_time = time.time() - start_time
    print(f"朴素匹配: {naive_time:.6f}s, 匹配数: {len(naive_matches)}")

    # 测试KMP
    start_time = time.time()
    kmp_matches = kmp_search(text, pattern)
    kmp_time = time.time() - start_time
    print(f"KMP匹配: {kmp_time:.6f}s, 匹配数: {len(kmp_matches)}")

    # 测试Boyer-Moore
    start_time = time.time()
    bm_matches = boyer_moore_search(text, pattern)
    bm_time = time.time() - start_time
    print(f"Boyer-Moore匹配: {bm_time:.6f}s, 匹配数: {len(bm_matches)}")

    # 测试Rabin-Karp
    start_time = time.time()
    rk_matches = rabin_karp_search(text, pattern)
    rk_time = time.time() - start_time
    print(f"Rabin-Karp匹配: {rk_time:.6f}s, 匹配数: {len(rk_matches)}")

def benchmark_string_structures():
    """字符串数据结构性能测试"""
    # 测试前缀树
    words = [generate_random_string(random.randint(3, 10)) for _ in range(10000)]
    prefix_words = [word[:random.randint(1, len(word))] for word in words[:1000]]

    trie = Trie()
    start_time = time.time()
    for word in words:
        trie.insert(word)
    trie_build_time = time.time() - start_time
    print(f"前缀树构建: {trie_build_time:.6f}s")

    start_time = time.time()
    for prefix in prefix_words:
        trie.get_all_words_with_prefix(prefix)
    trie_search_time = time.time() - start_time
    print(f"前缀树搜索: {trie_search_time:.6f}s")

    # 测试AC自动机
    patterns = words[:100]
    ac = AhoCorasick()
    start_time = time.time()
    for i, pattern in enumerate(patterns):
        ac.add_pattern(pattern, i)
    ac.build_fail_links()
    ac_build_time = time.time() - start_time
    print(f"AC自动机构建: {ac_build_time:.6f}s")

    test_text = generate_random_string(10000)
    start_time = time.time()
    ac.search(test_text)
    ac_search_time = time.time() - start_time
    print(f"AC自动机搜索: {ac_search_time:.6f}s")
```

## 应用场景

### 1. 生物信息学
- **DNA序列比对**: 寻找相似基因序列
- **蛋白质结构分析**: 识别蛋白质模式
- **基因组组装**: 从短序列重建完整基因组

```python
class DNASequenceAnalyzer:
    """DNA序列分析器"""
    def __init__(self):
        self.trie = Trie()

    def add_gene_sequence(self, sequence_id: str, sequence: str):
        """添加基因序列"""
        self.trie.insert(sequence)

    def find_similar_sequences(self, query: str, max_distance: int = 2) -> list:
        """查找相似序列"""
        similar = []
        words = self.trie.get_all_words_with_prefix(query[:3])  # 使用前缀缩小范围

        for word in words:
            distance = levenshtein_distance(query, word)
            if distance <= max_distance:
                similar.append((word, distance))

        return sorted(similar, key=lambda x: x[1])

    def find_repeated_patterns(self, sequence: str) -> list:
        """查找重复模式"""
        patterns = []
        n = len(sequence)

        for length in range(2, min(10, n//2 + 1)):
            pattern_counts = {}
            for i in range(n - length + 1):
                pattern = sequence[i:i+length]
                pattern_counts[pattern] = pattern_counts.get(pattern, 0) + 1

            for pattern, count in pattern_counts.items():
                if count >= 3:  # 至少出现3次
                    patterns.append((pattern, count))

        return sorted(patterns, key=lambda x: x[1], reverse=True)
```

### 2. 搜索引擎
- **全文检索**: 快速查找包含关键词的文档
- **拼写检查**: 自动纠正用户拼写错误
- **自动补全**: 实时提供搜索建议

```python
class SearchEngine:
    """搜索引擎"""
    def __init__(self):
        self.inverted_index = {}
        self.trie = Trie()
        self.documents = {}

    def add_document(self, doc_id: int, content: str):
        """添加文档"""
        self.documents[doc_id] = content
        words = content.lower().split()

        # 构建倒排索引
        for word in words:
            if word not in self.inverted_index:
                self.inverted_index[word] = set()
            self.inverted_index[word].add(doc_id)

        # 构建前缀树用于自动补全
        for word in set(words):
            self.trie.insert(word)

    def search(self, query: str) -> list:
        """搜索文档"""
        query_words = query.lower().split()
        if not query_words:
            return []

        # 找到包含所有查询词的文档
        result_docs = None
        for word in query_words:
            if word in self.inverted_index:
                if result_docs is None:
                    result_docs = self.inverted_index[word].copy()
                else:
                    result_docs.intersection_update(self.inverted_index[word])
            else:
                return []

        if result_docs is None:
            return []

        # 按相关性排序
        ranked_docs = []
        for doc_id in result_docs:
            score = self._calculate_relevance(doc_id, query_words)
            ranked_docs.append((doc_id, score))

        return sorted(ranked_docs, key=lambda x: x[1], reverse=True)

    def _calculate_relevance(self, doc_id: int, query_words: list) -> float:
        """计算相关性分数"""
        content = self.documents[doc_id].lower()
        score = 0

        for word in query_words:
            # 词频
            frequency = content.count(word)
            score += frequency

        return score

    def autocomplete(self, prefix: str, max_suggestions: int = 5) -> list:
        """自动补全"""
        suggestions = self.trie.get_all_words_with_prefix(prefix.lower())
        return suggestions[:max_suggestions]

    def spell_check(self, word: str) -> list:
        """拼写检查"""
        suggestions = []
        for dict_word in self.inverted_index.keys():
            distance = levenshtein_distance(word, dict_word)
            if distance <= 2:
                suggestions.append((dict_word, distance))

        return sorted(suggestions, key=lambda x: x[1])[:5]
```

### 3. 数据压缩
- **文件压缩**: 减少存储空间
- **网络传输**: 提高传输效率
- **数据库压缩**: 优化数据库性能

```python
class DataCompressor:
    """数据压缩器"""
    def __init__(self):
        self.compression_methods = {
            'lzw': self._lzw_compress,
            'rle': self._rle_compress
        }

    def compress(self, data: str, method: str = 'lzw') -> bytes:
        """压缩数据"""
        if method not in self.compression_methods:
            raise ValueError(f"不支持的压缩方法: {method}")

        compressed_data = self.compression_methods[method](data)
        return self._encode_compressed_data(compressed_data, method)

    def decompress(self, compressed_data: bytes) -> str:
        """解压缩数据"""
        method, data = self._decode_compressed_data(compressed_data)

        if method == 'lzw':
            return self._lzw_decompress(data)
        elif method == 'rle':
            return self._rle_decompress(data)
        else:
            raise ValueError(f"不支持的解压缩方法: {method}")

    def _lzw_compress(self, data: str) -> list:
        """LZW压缩"""
        return lzw_compress(data)

    def _lzw_decompress(self, compressed_data: list) -> str:
        """LZW解压缩"""
        return lzw_decompress(compressed_data)

    def _rle_compress(self, data: str) -> str:
        """游程编码压缩"""
        return run_length_encode(data)

    def _rle_decompress(self, encoded_data: str) -> str:
        """游程编码解压缩"""
        return run_length_decode(encoded_data)

    def _encode_compressed_data(self, data, method: str) -> bytes:
        """编码压缩数据"""
        if method == 'lzw':
            # LZW: 方法标识 + 数据
            method_byte = b'L'
            data_bytes = b','.join(str(x).encode() for x in data)
            return method_byte + data_bytes
        else:
            # RLE: 方法标识 + 数据
            method_byte = b'R'
            return method_byte + data.encode()

    def _decode_compressed_data(self, compressed_data: bytes) -> tuple:
        """解码压缩数据"""
        method_byte = compressed_data[0:1]
        data = compressed_data[1:]

        if method_byte == b'L':
            # 解析LZW数据
            compressed_list = [int(x) for x in data.split(b',')]
            return 'lzw', compressed_list
        else:
            # RLE数据
            return 'rle', data.decode()

    def benchmark_compression(self, text: str) -> dict:
        """基准测试压缩效果"""
        results = {}

        for method in self.compression_methods.keys():
            try:
                compressed = self.compress(text, method)
                decompressed = self.decompress(compressed)

                compression_ratio = len(compressed) / len(text.encode())
                is_correct = decompressed == text

                results[method] = {
                    'compression_ratio': compression_ratio,
                    'is_correct': is_correct,
                    'original_size': len(text.encode()),
                    'compressed_size': len(compressed)
                }
            except Exception as e:
                results[method] = {'error': str(e)}

        return results
```

### 4. 网络安全
- **入侵检测**: 识别恶意模式
- **病毒扫描**: 检测病毒特征码
- **数据过滤**: 过滤敏感信息

```python
class SecurityScanner:
    """安全扫描器"""
    def __init__(self):
        self.virus_signatures = AhoCorasick()
        self.malicious_patterns = []
        self.sensitive_keywords = []

    def add_virus_signature(self, signature: str, virus_id: int):
        """添加病毒特征码"""
        self.virus_signatures.add_pattern(signature, virus_id)

    def scan_file(self, file_content: str) -> dict:
        """扫描文件"""
        scan_results = {
            'viruses_detected': [],
            'malicious_patterns': [],
            'sensitive_keywords': []
        }

        # 病毒扫描
        virus_matches = self.virus_signatures.search(file_content)
        for virus_id, positions in virus_matches.items():
            scan_results['viruses_detected'].append({
                'virus_id': virus_id,
                'positions': positions
            })

        # 恶意模式检测
        for pattern in self.malicious_patterns:
            matches = kmp_search(file_content, pattern)
            if matches:
                scan_results['malicious_patterns'].append({
                    'pattern': pattern,
                    'positions': matches
                })

        # 敏感关键词检测
        for keyword in self.sensitive_keywords:
            matches = boyer_moore_search(file_content, keyword)
            if matches:
                scan_results['sensitive_keywords'].append({
                    'keyword': keyword,
                    'positions': matches
                })

        return scan_results

    def add_malicious_pattern(self, pattern: str):
        """添加恶意模式"""
        self.malicious_patterns.append(pattern)

    def add_sensitive_keyword(self, keyword: str):
        """添加敏感关键词"""
        self.sensitive_keywords.append(keyword)

    def build_scanner(self):
        """构建扫描器"""
        self.virus_signatures.build_fail_links()

    def is_file_clean(self, file_content: str) -> bool:
        """检查文件是否干净"""
        scan_results = self.scan_file(file_content)
        return (not scan_results['viruses_detected'] and
                not scan_results['malicious_patterns'] and
                not scan_results['sensitive_keywords'])
```

## 高级技巧与优化

### 1. 并行字符串匹配
```python
import concurrent.futures

class ParallelStringMatcher:
    """并行字符串匹配器"""
    def __init__(self, num_threads=4):
        self.num_threads = num_threads

    def parallel_kmp_search(self, text: str, pattern: str) -> list:
        """并行KMP搜索"""
        n = len(text)
        m = len(pattern)
        chunk_size = n // self.num_threads

        def search_chunk(start, end):
            return kmp_search(text[start:end], pattern)

        all_matches = []

        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_threads) as executor:
            futures = []
            for i in range(self.num_threads):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < self.num_threads - 1 else n
                futures.append(executor.submit(search_chunk, start, end))

            for future in concurrent.futures.as_completed(futures):
                chunk_matches = future.result()
                # 调整匹配位置到全文坐标
                chunk_index = futures.index(future)
                offset = chunk_index * chunk_size
                adjusted_matches = [pos + offset for pos in chunk_matches]
                all_matches.extend(adjusted_matches)

        return sorted(all_matches)
```

### 2. 内存优化的字符串处理
```python
class MemoryEfficientStringProcessor:
    """内存高效的字符串处理器"""
    def __init__(self):
        self.string_pool = {}

    def intern_string(self, s: str) -> str:
        """字符串驻留"""
        if s not in self.string_pool:
            self.string_pool[s] = s
        return self.string_pool[s]

    def process_large_file(self, filename: str, chunk_size=8192):
        """处理大文件"""
        with open(filename, 'r', encoding='utf-8') as file:
            while True:
                chunk = file.read(chunk_size)
                if not chunk:
                    break
                # 处理chunk
                yield self.intern_string(chunk)

    def memory_efficient_concatenation(self, strings: list) -> str:
        """内存高效的字符串连接"""
        # 计算总长度
        total_length = sum(len(s) for s in strings)
        # 预分配内存
        result = bytearray(total_length)
        # 逐个复制
        pos = 0
        for s in strings:
            result[pos:pos+len(s)] = s.encode('utf-8')
            pos += len(s)
        return result.decode('utf-8')
```

### 3. 缓存优化的字符串算法
```python
class CacheOptimizedStringMatcher:
    """缓存优化的字符串匹配器"""
    def __init__(self):
        self.block_size = 64  # 缓存行大小

    def cache_aware_search(self, text: str, pattern: str) -> list:
        """缓存感知的搜索"""
        n = len(text)
        m = len(pattern)
        matches = []

        # 按块处理以提高缓存命中率
        for i in range(0, n - m + 1, self.block_size):
            block_end = min(i + self.block_size + m, n)
            block_matches = []

            # 在块内搜索
            for j in range(i, min(block_end - m + 1, n - m + 1)):
                if text[j:j + m] == pattern:
                    block_matches.append(j)

            matches.extend(block_matches)

        return matches
```

## 练习

1. 实现一个支持Unicode字符串的前缀树
2. 创建一个并行版本的后缀数组构建算法
3. 实现一个基于后缀数组的模式匹配算法
4. 设计一个支持增量更新的AC自动机
5. 实现一个压缩率更高的字符串压缩算法
6. 创建一个支持模糊搜索的搜索引擎
7. 实现一个基于字符串相似度的聚类算法
8. 设计一个处理大规模文本的流式字符串处理框架