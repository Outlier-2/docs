# L09 - 动态规划

## 学习目标
- 掌握动态规划的基本原理和思想
- 学会识别和解决动态规划问题
- 理解状态转移方程的设计方法
- 能够应用动态规划解决复杂优化问题

## 动态规划基础

### 核心概念
- **最优子结构**: 问题的最优解包含子问题的最优解
- **重叠子问题**: 重复计算相同的子问题
- **状态**: 描述问题当前状况的变量组合
- **状态转移**: 从一个状态到另一个状态的转换规则
- **边界条件**: 问题的初始或终止条件

### 解决步骤
1. **定义状态**: 确定描述问题的状态变量
2. **状态转移方程**: 找出状态之间的关系
3. **边界条件**: 确定初始状态和终止状态
4. **计算顺序**: 确定状态的计算顺序
5. **空间优化**: 尽可能减少空间复杂度

## Python实现

### 1. 基础动态规划问题

#### 1.1 斐波那契数列
```python
def fibonacci_recursive(n: int) -> int:
    """
    递归斐波那契数列
    时间复杂度: O(2^n)
    空间复杂度: O(n)
    """
    if n <= 1:
        return n
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

def fibonacci_memoization(n: int, memo=None) -> int:
    """
    记忆化递归斐波那契数列
    时间复杂度: O(n)
    空间复杂度: O(n)
    """
    if memo is None:
        memo = {}
    if n in memo:
        return memo[n]
    if n <= 1:
        return n

    memo[n] = fibonacci_memoization(n-1, memo) + fibonacci_memoization(n-2, memo)
    return memo[n]

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

def fibonacci_matrix(n: int) -> int:
    """
    矩阵快速幂斐波那契数列
    时间复杂度: O(log n)
    空间复杂度: O(1)
    """
    def matrix_multiply(a, b):
        return [
            [a[0][0] * b[0][0] + a[0][1] * b[1][0],
             a[0][0] * b[0][1] + a[0][1] * b[1][1]
        ], [
            a[1][0] * b[0][0] + a[1][1] * b[1][0],
             a[1][0] * b[0][1] + a[1][1] * b[1][1]
        ]

    def matrix_pow(mat, power):
        result = [[1, 0], [0, 1]]  # 单位矩阵
        while power > 0:
            if power % 2 == 1:
                result = matrix_multiply(result, mat)
            mat = matrix_multiply(mat, mat)
            power //= 2
        return result

    if n <= 1:
        return n

    mat = [[1, 1], [1, 0]]
    result = matrix_pow(mat, n - 1)
    return result[0][0]
```

#### 1.2 爬楼梯问题
```python
def climb_stairs(n: int) -> int:
    """
    爬楼梯问题：每次可以爬1或2个台阶
    状态: dp[i] = 爬到第i阶的方法数
    状态转移: dp[i] = dp[i-1] + dp[i-2]
    """
    if n <= 2:
        return n

    dp = [0] * (n + 1)
    dp[0] = 1  # 地面
    dp[1] = 1
    dp[2] = 2

    for i in range(3, n + 1):
        dp[i] = dp[i-1] + dp[i-2]

    return dp[n]

def climb_stairs_optimized(n: int) -> int:
    """
    空间优化的爬楼梯问题
    """
    if n <= 2:
        return n

    a, b = 1, 2
    for _ in range(3, n + 1):
        a, b = b, a + b

    return b

def climb_stairs_general(n: int, steps: list) -> int:
    """
    通用爬楼梯问题：每次可以爬steps中的任意步数
    """
    if n < 0:
        return 0
    if n == 0:
        return 1

    dp = [0] * (n + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        for step in steps:
            if i >= step:
                dp[i] += dp[i - step]

    return dp[n]
```

### 2. 背包问题

#### 2.1 0-1背包问题
```python
def zero_one_knapsack(weights: list, values: list, capacity: int) -> int:
    """
    0-1背包问题
    时间复杂度: O(n * capacity)
    空间复杂度: O(n * capacity)
    """
    n = len(weights)
    if n == 0 or capacity == 0:
        return 0

    # dp[i][j] = 前i个物品，容量为j时的最大价值
    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, capacity + 1):
            if weights[i-1] <= j:
                dp[i][j] = max(dp[i-1][j],
                             dp[i-1][j - weights[i-1]] + values[i-1])
            else:
                dp[i][j] = dp[i-1][j]

    return dp[n][capacity]

def zero_one_knapsack_optimized(weights: list, values: list, capacity: int) -> int:
    """
    空间优化的0-1背包问题
    空间复杂度: O(capacity)
    """
    n = len(weights)
    if n == 0 or capacity == 0:
        return 0

    dp = [0] * (capacity + 1)

    for i in range(n):
        for j in range(capacity, weights[i] - 1, -1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])

    return dp[capacity]

def zero_one_knapsack_with_items(weights: list, values: list, capacity: int) -> tuple:
    """
    返回物品选择的0-1背包问题
    """
    n = len(weights)
    if n == 0 or capacity == 0:
        return 0, []

    dp = [[0] * (capacity + 1) for _ in range(n + 1)]

    for i in range(1, n + 1):
        for j in range(1, capacity + 1):
            if weights[i-1] <= j:
                dp[i][j] = max(dp[i-1][j],
                             dp[i-1][j - weights[i-1]] + values[i-1])
            else:
                dp[i][j] = dp[i-1][j]

    # 重构选择的物品
    selected_items = []
    total_value = dp[n][capacity]
    remaining_capacity = capacity

    for i in range(n, 0, -1):
        if dp[i][remaining_capacity] != dp[i-1][remaining_capacity]:
            selected_items.append(i-1)
            remaining_capacity -= weights[i-1]

    selected_items.reverse()
    return total_value, selected_items
```

#### 2.2 完全背包问题
```python
def unbounded_knapsack(weights: list, values: list, capacity: int) -> int:
    """
    完全背包问题：物品可以重复选择
    时间复杂度: O(n * capacity)
    空间复杂度: O(capacity)
    """
    n = len(weights)
    if n == 0 or capacity == 0:
        return 0

    dp = [0] * (capacity + 1)

    for i in range(n):
        for j in range(weights[i], capacity + 1):
            dp[j] = max(dp[j], dp[j - weights[i]] + values[i])

    return dp[capacity]

def unbounded_knapsack_with_counts(weights: list, values: list, capacity: int) -> tuple:
    """
    返回物品数量的完全背包问题
    """
    n = len(weights)
    if n == 0 or capacity == 0:
        return 0, []

    dp = [0] * (capacity + 1)
    count = [0] * (capacity + 1)

    for j in range(1, capacity + 1):
        for i in range(n):
            if weights[i] <= j:
                if dp[j] < dp[j - weights[i]] + values[i]:
                    dp[j] = dp[j - weights[i]] + values[i]
                    count[j] = count[j - weights[i]] + 1

    # 重构物品选择
    items = []
    remaining_capacity = capacity
    while remaining_capacity > 0:
        for i in range(n):
            if weights[i] <= remaining_capacity:
                if dp[remaining_capacity] == dp[remaining_capacity - weights[i]] + values[i]:
                    items.append(i)
                    remaining_capacity -= weights[i]
                    break

    return dp[capacity], items
```

#### 2.3 多重背包问题
```python
def multiple_knapsack(weights: list, values: list, counts: list, capacity: int) -> int:
    """
    多重背包问题：每个物品有数量限制
    """
    n = len(weights)
    if n == 0 or capacity == 0:
        return 0

    dp = [0] * (capacity + 1)

    for i in range(n):
        # 将多重背包转换为0-1背包
        num = min(counts[i], capacity // weights[i])
        k = 1
        while k <= num:
            current_weight = k * weights[i]
            current_value = k * values[i]

            for j in range(capacity, current_weight - 1, -1):
                dp[j] = max(dp[j], dp[j - current_weight] + current_value)

            num -= k
            k *= 2

        if num > 0:
            current_weight = num * weights[i]
            current_value = num * values[i]
            for j in range(capacity, current_weight - 1, -1):
                dp[j] = max(dp[j], dp[j - current_weight] + current_value)

    return dp[capacity]
```

### 3. 字符串动态规划

#### 3.1 最长公共子序列 (LCS)
```python
def longest_common_subsequence(text1: str, text2: str) -> int:
    """
    最长公共子序列长度
    时间复杂度: O(m * n)
    空间复杂度: O(m * n)
    """
    m, n = len(text1), len(text2)

    # dp[i][j] = text1前i个字符和text2前j个字符的LCS长度
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    return dp[m][n]

def longest_common_subsequence_with_sequence(text1: str, text2: str) -> tuple:
    """
    返回最长公共子序列
    """
    m, n = len(text1), len(text2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])

    # 重构LCS
    lcs = []
    i, j = m, n

    while i > 0 and j > 0:
        if text1[i-1] == text2[j-1]:
            lcs.append(text1[i-1])
            i -= 1
            j -= 1
        elif dp[i-1][j] > dp[i][j-1]:
            i -= 1
        else:
            j -= 1

    lcs.reverse()
    return dp[m][n], ''.join(lcs)

def longest_common_subsequence_optimized(text1: str, text2: str) -> int:
    """
    空间优化的LCS
    空间复杂度: O(min(m, n))
    """
    if len(text1) < len(text2):
        text1, text2 = text2, text1

    m, n = len(text1), len(text2)
    prev = [0] * (n + 1)
    curr = [0] * (n + 1)

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i-1] == text2[j-1]:
                curr[j] = prev[j-1] + 1
            else:
                curr[j] = max(prev[j], curr[j-1])
        prev, curr = curr, prev

    return prev[n]
```

#### 3.2 最长递增子序列 (LIS)
```python
def longest_increasing_subsequence(nums: list) -> int:
    """
    最长递增子序列长度
    时间复杂度: O(n²)
    空间复杂度: O(n)
    """
    if not nums:
        return 0

    n = len(nums)
    dp = [1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j]:
                dp[i] = max(dp[i], dp[j] + 1)

    return max(dp)

def longest_increasing_subsequence_optimized(nums: list) -> int:
    """
    优化的LIS：使用二分查找
    时间复杂度: O(n log n)
    空间复杂度: O(n)
    """
    if not nums:
        return 0

    tails = []
    for num in nums:
        # 使用二分查找找到插入位置
        left, right = 0, len(tails)
        while left < right:
            mid = (left + right) // 2
            if tails[mid] < num:
                left = mid + 1
            else:
                right = mid

        if left == len(tails):
            tails.append(num)
        else:
            tails[left] = num

    return len(tails)

def longest_increasing_subsequence_with_sequence(nums: list) -> tuple:
    """
    返回最长递增子序列
    """
    if not nums:
        return 0, []

    n = len(nums)
    dp = [1] * n
    prev = [-1] * n

    for i in range(1, n):
        for j in range(i):
            if nums[i] > nums[j] and dp[i] < dp[j] + 1:
                dp[i] = dp[j] + 1
                prev[i] = j

    max_length = max(dp)
    max_index = dp.index(max_length)

    # 重构LIS
    lis = []
    current = max_index
    while current != -1:
        lis.append(nums[current])
        current = prev[current]

    lis.reverse()
    return max_length, lis
```

#### 3.3 编辑距离
```python
def min_edit_distance(word1: str, word2: str) -> int:
    """
    最小编辑距离（Levenshtein距离）
    时间复杂度: O(m * n)
    空间复杂度: O(m * n)
    """
    m, n = len(word1), len(word2)

    # dp[i][j] = word1前i个字符转换为word2前j个字符的最小编辑距离
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # 初始化
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + 1,    # 删除
                    dp[i][j-1] + 1,    # 插入
                    dp[i-1][j-1] + 1   # 替换
                )

    return dp[m][n]

def min_edit_distance_with_operations(word1: str, word2: str) -> tuple:
    """
    返回编辑操作的编辑距离
    """
    m, n = len(word1), len(word2)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    operations = [[None] * (n + 1) for _ in range(m + 1)]

    for i in range(m + 1):
        dp[i][0] = i
        operations[i][0] = 'delete' * i
    for j in range(n + 1):
        dp[0][j] = j
        operations[0][j] = 'insert' * j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i-1] == word2[j-1]:
                dp[i][j] = dp[i-1][j-1]
                operations[i][j] = operations[i-1][j-1]
            else:
                delete_cost = dp[i-1][j] + 1
                insert_cost = dp[i][j-1] + 1
                replace_cost = dp[i-1][j-1] + 1

                if delete_cost <= insert_cost and delete_cost <= replace_cost:
                    dp[i][j] = delete_cost
                    operations[i][j] = operations[i-1][j] + ' delete'
                elif insert_cost <= delete_cost and insert_cost <= replace_cost:
                    dp[i][j] = insert_cost
                    operations[i][j] = operations[i][j-1] + ' insert'
                else:
                    dp[i][j] = replace_cost
                    operations[i][j] = operations[i-1][j-1] + f' replace {word1[i-1]} with {word2[j-1]}'

    return dp[m][n], operations[m][n].strip()
```

### 4. 区间动态规划

#### 4.1 矩阵链乘法
```python
def matrix_chain_order(dimensions: list) -> tuple:
    """
    矩阵链乘法最优括号化
    时间复杂度: O(n³)
    空间复杂度: O(n²)
    """
    n = len(dimensions) - 1  # 矩阵数量

    # dp[i][j] = 矩阵i到矩阵j的最小乘法次数
    dp = [[0] * n for _ in range(n)]
    # split[i][j] = 最优分割点
    split = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):  # 链长度
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')

            for k in range(i, j):
                cost = (dp[i][k] + dp[k+1][j] +
                       dimensions[i] * dimensions[k+1] * dimensions[j+1])

                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k

    # 重构最优括号化
    def construct_optimal_parenthesis(i, j):
        if i == j:
            return f"A{i+1}"
        else:
            k = split[i][j]
            left = construct_optimal_parenthesis(i, k)
            right = construct_optimal_parenthesis(k+1, j)
            return f"({left} × {right})"

    optimal_order = construct_optimal_parenthesis(0, n-1)

    return dp[0][n-1], optimal_order

def matrix_chain_multiplication(matrices: list, dimensions: list) -> tuple:
    """
    实际执行矩阵链乘法
    """
    min_multiplications, optimal_order = matrix_chain_order(dimensions)

    def multiply_matrices(i, j):
        if i == j:
            return matrices[i]
        else:
            k = split[i][j]
            left = multiply_matrices(i, k)
            right = multiply_matrices(k+1, j)
            return multiply_matrices_helper(left, right)

    def multiply_matrices_helper(A, B):
        """辅助函数：矩阵乘法"""
        if isinstance(A, list) and isinstance(B, list):
            # A和B都是矩阵
            m = len(A)
            p = len(A[0])
            n = len(B[0])
            result = [[0] * n for _ in range(m)]

            for i in range(m):
                for j in range(n):
                    for k in range(p):
                        result[i][j] += A[i][k] * B[k][j]

            return result
        else:
            # 简化处理：返回矩阵维度
            return (A[0], B[1]) if isinstance(A, tuple) and isinstance(B, tuple) else A

    # 需要先计算split矩阵
    n = len(dimensions) - 1
    dp = [[0] * n for _ in range(n)]
    global split
    split = [[0] * n for _ in range(n)]

    for length in range(2, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            dp[i][j] = float('inf')

            for k in range(i, j):
                cost = (dp[i][k] + dp[k+1][j] +
                       dimensions[i] * dimensions[k+1] * dimensions[j+1])

                if cost < dp[i][j]:
                    dp[i][j] = cost
                    split[i][j] = k

    return min_multiplications, optimal_order
```

#### 4.2 最长回文子串
```python
def longest_palindromic_substring(s: str) -> str:
    """
    最长回文子串
    时间复杂度: O(n²)
    空间复杂度: O(n²)
    """
    if not s:
        return ""

    n = len(s)
    # dp[i][j] = s[i:j+1]是否为回文
    dp = [[False] * n for _ in range(n)]
    start = 0
    max_length = 1

    # 所有长度为1的子串都是回文
    for i in range(n):
        dp[i][i] = True

    # 检查长度为2的子串
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_length = 2

    # 检查长度大于2的子串
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1

            if s[i] == s[j] and dp[i + 1][j - 1]:
                dp[i][j] = True
                if length > max_length:
                    start = i
                    max_length = length

    return s[start:start + max_length]

def longest_palindromic_substring_optimized(s: str) -> str:
    """
    空间优化的最长回文子串
    空间复杂度: O(1)
    """
    if not s:
        return ""

    def expand_around_center(left, right):
        """从中心扩展"""
        while left >= 0 and right < len(s) and s[left] == s[right]:
            left -= 1
            right += 1
        return right - left - 1

    start = 0
    end = 0

    for i in range(len(s)):
        len1 = expand_around_center(i, i)      # 奇数长度
        len2 = expand_around_center(i, i + 1)  # 偶数长度
        max_len = max(len1, len2)

        if max_len > end - start:
            start = i - (max_len - 1) // 2
            end = i + max_len // 2

    return s[start:end + 1]
```

### 5. 股票买卖问题

#### 5.1 买卖股票的最佳时机
```python
def max_profit_one_transaction(prices: list) -> int:
    """
    买卖股票的最佳时机（只允许一次交易）
    时间复杂度: O(n)
    空间复杂度: O(1)
    """
    if not prices:
        return 0

    min_price = prices[0]
    max_profit = 0

    for price in prices[1:]:
        max_profit = max(max_profit, price - min_price)
        min_price = min(min_price, price)

    return max_profit

def max_profit_unlimited_transactions(prices: list) -> int:
    """
    买卖股票的最佳时机（允许无限次交易）
    时间复杂度: O(n)
    空间复杂度: O(1)
    """
    if not prices:
        return 0

    max_profit = 0

    for i in range(1, len(prices)):
        if prices[i] > prices[i-1]:
            max_profit += prices[i] - prices[i-1]

    return max_profit

def max_profit_k_transactions(prices: list, k: int) -> int:
    """
    买卖股票的最佳时机（最多k次交易）
    时间复杂度: O(n * k)
    空间复杂度: O(n * k)
    """
    if not prices or k == 0:
        return 0

    n = len(prices)

    # 如果k >= n/2，相当于无限次交易
    if k >= n // 2:
        return max_profit_unlimited_transactions(prices)

    # dp[i][j] = 最多i次交易，在第j天的最大利润
    dp = [[0] * n for _ in range(k + 1)]

    for i in range(1, k + 1):
        max_diff = -prices[0]
        for j in range(1, n):
            dp[i][j] = max(dp[i][j-1], prices[j] + max_diff)
            max_diff = max(max_diff, dp[i-1][j-1] - prices[j])

    return dp[k][n-1]

def max_profit_with_cooldown(prices: list) -> int:
    """
    买卖股票的最佳时机（包含冷却期）
    时间复杂度: O(n)
    空间复杂度: O(1)
    """
    if not prices:
        return 0

    # hold[i] = 第i天持有股票的最大收益
    # sold[i] = 第i天卖出股票的最大收益
    # rest[i] = 第i天休息的最大收益
    hold = sold = rest = -float('inf')

    for price in prices:
        prev_hold = hold
        prev_sold = sold
        prev_rest = rest

        # 当前持有：继续保持持有 或 之前休息今天买入
        hold = max(prev_hold, prev_rest - price)
        # 当前卖出：之前持有今天卖出
        sold = prev_hold + price
        # 当前休息：之前休息 或 之前卖出
        rest = max(prev_rest, prev_sold)

    return max(sold, rest)
```

### 6. 高级动态规划

#### 6.1 掷骰子问题
```python
def number_of_dice_rolls(n: int, k: int, target: int) -> int:
    """
    n个骰子，每个骰子k个面，求掷出target的方法数
    时间复杂度: O(n * target * k)
    空间复杂度: O(n * target)
    """
    MOD = 10**9 + 7

    # dp[i][j] = 使用i个骰子，掷出j的方法数
    dp = [[0] * (target + 1) for _ in range(n + 1)]
    dp[0][0] = 1

    for i in range(1, n + 1):
        for j in range(1, target + 1):
            for face in range(1, min(k, j) + 1):
                dp[i][j] = (dp[i][j] + dp[i-1][j-face]) % MOD

    return dp[n][target]

def number_of_dice_rolls_optimized(n: int, k: int, target: int) -> int:
    """
    空间优化的掷骰子问题
    空间复杂度: O(target)
    """
    MOD = 10**9 + 7

    if target < n or target > n * k:
        return 0

    dp = [0] * (target + 1)
    dp[0] = 1

    for i in range(1, n + 1):
        new_dp = [0] * (target + 1)
        for j in range(1, target + 1):
            for face in range(1, min(k, j) + 1):
                new_dp[j] = (new_dp[j] + dp[j-face]) % MOD
        dp = new_dp

    return dp[target]
```

#### 6.2 戳气球问题
```python
def max_coins(nums: list) -> int:
    """
    戳气球问题
    时间复杂度: O(n³)
    空间复杂度: O(n²)
    """
    n = len(nums)
    # 添加边界
    nums = [1] + nums + [1]
    new_n = n + 2

    # dp[i][j] = 戳破(i,j)开区间内所有气球的最大硬币数
    dp = [[0] * new_n for _ in range(new_n)]

    for length in range(2, new_n):  # 区间长度
        for left in range(new_n - length):
            right = left + length

            for k in range(left + 1, right):
                # k是区间内最后戳破的气球
                coins = (nums[left] * nums[k] * nums[right] +
                        dp[left][k] + dp[k][right])
                dp[left][right] = max(dp[left][right], coins)

    return dp[0][new_n - 1]
```

#### 6.3 正则表达式匹配
```python
def is_match(s: str, p: str) -> bool:
    """
    正则表达式匹配（支持 '.' 和 '*'）
    时间复杂度: O(m * n)
    空间复杂度: O(m * n)
    """
    m, n = len(s), len(p)

    # dp[i][j] = s前i个字符和p前j个字符是否匹配
    dp = [[False] * (n + 1) for _ in range(m + 1)]

    # 空字符串匹配空字符串
    dp[0][0] = True

    # 处理模式开头的'*'（可以匹配0次）
    for j in range(1, n + 1):
        if p[j-1] == '*':
            dp[0][j] = dp[0][j-2]

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if p[j-1] == '.' or p[j-1] == s[i-1]:
                dp[i][j] = dp[i-1][j-1]
            elif p[j-1] == '*':
                dp[i][j] = dp[i][j-2]  # 匹配0次
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    dp[i][j] = dp[i][j] or dp[i-1][j]  # 匹配1次或多次

    return dp[m][n]

def is_match_optimized(s: str, p: str) -> bool:
    """
    空间优化的正则表达式匹配
    空间复杂度: O(n)
    """
    m, n = len(s), len(p)

    prev = [False] * (n + 1)
    curr = [False] * (n + 1)

    prev[0] = True

    for j in range(1, n + 1):
        if p[j-1] == '*':
            prev[j] = prev[j-2]

    for i in range(1, m + 1):
        curr[0] = False
        for j in range(1, n + 1):
            if p[j-1] == '.' or p[j-1] == s[i-1]:
                curr[j] = prev[j-1]
            elif p[j-1] == '*':
                curr[j] = curr[j-2]
                if p[j-2] == '.' or p[j-2] == s[i-1]:
                    curr[j] = curr[j] or prev[j]
        prev, curr = curr, prev

    return prev[n]
```

## 动态规划模式总结

### 1. 线性DP
- **特征**: 问题可以按线性顺序解决
- **状态**: dp[i] 表示前i个元素的最优解
- **例子**: 斐波那契数列、爬楼梯、最长递增子序列

### 2. 背包DP
- **特征**: 在有限容量下选择物品
- **状态**: dp[i][j] 表示前i个物品在容量j下的最优解
- **例子**: 0-1背包、完全背包、多重背包

### 3. 区间DP
- **特征**: 问题涉及区间操作
- **状态**: dp[i][j] 表示区间[i,j]的最优解
- **例子**: 矩阵链乘法、最长回文子串、戳气球

### 4. 树形DP
- **特征**: 问题在树结构上定义
- **状态**: dp[u] 表示以u为根的子树的最优解
- **例子**: 树的最大独立集、树的直径

### 5. 状态压缩DP
- **特征**: 状态可以用二进制表示
- **状态**: dp[mask] 表示状态mask下的最优解
- **例子**: 旅行商问题、集合覆盖问题

## 性能优化技巧

### 1. 空间优化
```python
def space_optimization_example():
    """空间优化示例：滚动数组"""
    # 原始版本
    dp = [[0] * n for _ in range(m)]

    # 优化版本
    dp = [[0] * n for _ in range(2)]
    for i in range(m):
        curr = i % 2
        prev = 1 - curr
        for j in range(n):
            dp[curr][j] = dp[prev][j] + ...  # 使用前一行
```

### 2. 前缀和优化
```python
def prefix_sum_optimization():
    """前缀和优化"""
    # 计算前缀和
    prefix = [0]
    for num in nums:
        prefix.append(prefix[-1] + num)

    # 快速计算区间和
    def range_sum(l, r):
        return prefix[r+1] - prefix[l]
```

### 3. 斜率优化
```python
def convex_hull_trick_optimization():
    """斜率优化（凸包优化）"""
    # 用于优化形如 dp[i] = min(dp[j] + cost(i,j)) 的转移
    # 适用于 cost(i,j) 满足四边形不等式的情况
    pass
```

## 应用场景

### 1. 资源分配
- **项目投资**: 在预算约束下最大化收益
- **任务调度**: 优化任务执行顺序
- **库存管理**: 最优库存策略

### 2. 序列分析
- **生物信息学**: DNA序列比对
- **自然语言处理**: 语义分析、机器翻译
- **信号处理**: 模式识别

### 3. 路径规划
- **机器人导航**: 最优路径规划
- **游戏AI**: 最优策略计算
- **物流优化**: 配送路径优化

### 4. 金融建模
- **投资组合**: 最优资产配置
- **风险评估**: 风险最小化
- **期权定价**: 金融衍生品定价

## 练习

1. 实现一个解决旅行商问题的动态规划算法
2. 创建一个解决编辑距离问题的并行算法
3. 实现一个解决最长公共子串问题的动态规划算法
4. 设计一个解决背包问题的分支限界算法
5. 实现一个解决矩阵链乘法的并行算法
6. 创建一个解决正则表达式匹配的动态规划算法
7. 实现一个解决戳气球问题的记忆化搜索算法
8. 设计一个解决股票买卖问题的通用框架