---

---
1.

![[image 15.png]]

2.

![[image 16.png]]

![[image 17.png]]

![[image 18.png]]

![[image 19.png]]

![[image 20.png]]

3、attention

$$
O_{new} = \underbrace{O_{old} \cdot \left( \frac{d \cdot e^{m - m_{new}}}{d_{new}} \right)}_{\text{对旧结果的分子、分母统一修正}} + \underbrace{V_{tile} \cdot \left( \frac{e^{S_{tile} - m_{new}}}{d_{new}} \right)}_{\text{新块的贡献}}
$$

import numpy as np

def online_softmax_attention(Q_dot_K_row, V_row):
"""
模拟 FlashAttention 中对一行 attention 的 online softmax 计算

```plain text
Parameters
----------
Q_dot_K_row : np.ndarray
    这一行所有 attention score（Q·K^T），shape: [N]

V_row : np.ndarray
    对应的 value 向量矩阵，shape: [N, D]

Returns
-------
np.ndarray
    attention 输出向量，shape: [D]
"""

n = len(Q_dot_K_row)
dim = V_row.shape[1]

m = -float("inf")      # 当前最大值 (running max)
d = 0.0                # 当前分母 (sum of exp)
O = np.zeros(dim)      # 当前加权累加结果 (分子部分)

for i in range(n):
    x_curr = Q_dot_K_row[i]   # 当前 score
    v_curr = V_row[i]         # 当前 value 向量

    m_old = m

    # 1️⃣ 更新最大值
    m = max(m_old, x_curr)

    # 2️⃣ 计算缩放因子
    alpha = np.exp(m_old - m)   # 旧数据的缩放系数
    beta = np.exp(x_curr - m)   # 新数据的权重

    # 3️⃣ 更新分母
    d = d * alpha + beta

    # 4️⃣ 更新分子（分子 × V 的累加）
    O = O * alpha + v_curr * beta

# 5️⃣ 最终归一化
return O / d
```

# =========================

# 测试 / 验证

# =========================

scores = np.array([0.1, 0.8, 0.3])

values = np.array([
[1, 0],
[0, 1],
[1, 1]
])

result = online_softmax_attention(scores, values)

print("Online Attention 结果:", result)

import numpy as np

def online_softmax_attention(Q_dot_K_row, V_row):
"""
模拟 FlashAttention 中对一行 attention 的 online softmax 计算

```plain text
Parameters
----------
Q_dot_K_row : np.ndarray
    这一行所有 attention score（Q·K^T），shape: [N]

V_row : np.ndarray
    对应的 value 向量矩阵，shape: [N, D]

Returns
-------
np.ndarray
    attention 输出向量，shape: [D]
"""

n = len(Q_dot_K_row)
dim = V_row.shape[1]

m = -float("inf")      # 当前最大值 (running max)
d = 0.0                # 当前分母 (sum of exp)
O = np.zeros(dim)      # 当前加权累加结果 (分子部分)

for i in range(n):
    x_curr = Q_dot_K_row[i]   # 当前 score
    v_curr = V_row[i]         # 当前 value 向量

    m_old = m

    # 1️⃣ 更新最大值
    m = max(m_old, x_curr)

    # 2️⃣ 计算缩放因子
    alpha = np.exp(m_old - m)   # 旧数据的缩放系数
    beta = np.exp(x_curr - m)   # 新数据的权重

    # 3️⃣ 更新分母
    d = d * alpha + beta

    # 4️⃣ 更新分子（分子 × V 的累加）
    O = O * alpha + v_curr * beta

# 5️⃣ 最终归一化
return O / d
```

# =========================

# 测试 / 验证

# =========================

scores = np.array([0.1, 0.8, 0.3])

values = np.array([
[1, 0],
[0, 1],
[1, 1]
])

result = online_softmax_attention(scores, values)

print("Online Attention 结果:", result)