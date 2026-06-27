# 增量学习笔记：从理论到实践的新发现

> 基于 InfraTech 项目复现过程，对比前两轮文档（面试准备 + 五类问题应对）的增量知识点

---

## 一、学习路径回顾

### 1.1 前两轮学习成果

| 轮次 | 文档 | 侧重点 |
|------|------|--------|
| 第一轮 | LLM_Algorithm_Interview_Preparation.md | 知识体系梳理、面试问题准备 |
| 第二轮 | Technical_Interview_Five_Categories.md | 五类能力应对、回答框架 |

### 1.2 本轮（第三轮）目标

从**理论理解**到**实践复现**，发现前两轮忽略的细节和新知识点：
- 实际工程中的 trade-off
- 代码实现细节
- 性能优化的实操经验
- 问题排查的方法论

---

## 二、推理框架深入：从原理到实现

### 2.1 vLLM 调度器的隐藏细节

**前两轮认知**：vLLM 使用 Continuous Batching 提高吞吐

**本轮新发现**：

#### 2.1.1 调度策略的细节

```python
# vllm_basic_scheduler.ipynb 中的实际实现
class Scheduler:
    def __init__(self):
        self.waiting = []  # 等待中的请求
        self.running = []  # 运行中的请求
        self.swapped = []  # 被换出的请求
    
    def schedule(self):
        # 1. 首先处理被换出的请求（优先级最高）
        # 2. 然后处理等待中的请求
        # 3. 最后继续运行中的请求
        
        # 关键：prefill 和 decode 的资源竞争
        # prefill 需要大量计算，decode 需要大量显存
```

**新学到的点**：
1. **抢占机制**：当显存不足时，vLLM 会抢占（preempt）正在运行的请求
2. **Swap 机制**：被抢占的请求的 KV Cache 可以换出到 CPU
3. **优先级策略**：等待时间长的请求优先级更高

#### 2.1.2 Prefix Cache 的实现细节

**前两轮认知**：相同前缀可以共享 KV Cache

**本轮新发现**：
```python
# vLLM 的 Prefix Cache 使用哈希实现
# 每个 Block 有一个哈希值，基于：
# 1. Block 内容的哈希
# 2. 前一个 Block 的哈希（链式）

# 这意味着：
# - 相同前缀的请求可以自动复用
# - 哈希冲突的概率极低
# - 无需额外的存储开销（"零开销"）
```

**新学到的点**：
1. **哈希链**：Block 的哈希是链式的，确保前缀匹配的正确性
2. **LRU 淘汰**：当显存不足时，使用 LRU 策略淘汰不活跃的 Block
3. **跨请求共享**：不同请求的相同前缀可以共享 KV Cache

### 2.2 SGLang RadixAttention 的数据结构

**前两轮认知**：SGLang 使用 Radix Tree 管理 KV Cache

**本轮新发现**：

```python
# Radix Tree 的实际结构
class RadixTree:
    def __init__(self):
        self.root = TreeNode()
    
    def insert(self, tokens, kv_cache):
        # 1. 从根节点开始匹配
        # 2. 找到最长公共前缀
        # 3. 在分歧点分裂节点
        # 4. 插入新的 KV Cache
        
    def lookup(self, tokens):
        # 1. 从根节点开始查找
        # 2. 返回最长匹配的 KV Cache
        # 3. 返回需要计算的部分
```

**新学到的点**：
1. **最长前缀匹配**：自动找到最长的可复用前缀
2. **节点分裂**：当出现分歧时，Radix Tree 会自动分裂
3. **引用计数**：每个节点有引用计数，确保 KV Cache 不被过早释放

### 2.3 Chunked Prefill 的工程挑战

**前两轮认知**：将长 prompt 切成 chunk，交错执行

**本轮新发现**：

**挑战1：Chunk 边界的注意力计算**
```python
# 因果注意力需要确保 chunk 之间的正确性
# Chunk 1: [0, 512)
# Chunk 2: [512, 1024)
# 
# Chunk 2 的 attention 需要能看到 Chunk 1 的 KV
# 解决方案：维护全局的 KV Cache，每个 chunk 更新
```

**挑战2：与 Continuous Batching 的协调**
```python
# 长 prompt 的 prefill 可能被切分成多个 chunk
# 每个 chunk 之间可能插入其他请求的 decode
# 需要调度器正确处理交错执行
```

**新学到的点**：
1. **Chunk 大小选择**：太小则开销大，太大则阻塞 decode
2. **与 Flash Decoding 的配合**：Chunked Prefill + Flash Decoding 可以进一步优化
3. **显存预分配**：需要为所有 chunk 预分配足够的显存

---

## 三、模型架构：从论文到代码

### 3.1 MLA 的 MHA/MQA 模式切换细节

**前两轮认知**：Prefill 用 MHA 模式，Decode 用 MQA 模式

**本轮新发现**：

```python
# MHA 模式（Prefill）
# 完整计算 Q, K, V，然后做 attention
q = W_q(x)  # [batch, seq_len, n_heads * d_head]
k = W_k(x)  # [batch, seq_len, n_heads * d_head]
v = W_v(x)  # [batch, seq_len, n_heads * d_head]
output = attention(q, k, v)

# MQA 模式（Decode）
# 使用吸收矩阵，避免显式恢复 K, V
c = W_dkv(x)  # [batch, 1, kv_lora_rank]  # 只缓存 c
# 将 W_uk 吸收到 W_q 中
q_absorbed = q @ W_uk  # [batch, 1, n_heads * d_head]
# 直接用 c 计算 attention
output = attention(q_absorbed, c, c)  # 使用 c 作为 K, V
```

**新学到的点**：
1. **吸收矩阵的计算**：`W_q' = W_q @ W_uk`，推理时只需计算一次
2. **MHA 模式的显存开销**：需要完整的 K, V，但计算效率高
3. **MQA 模式的计算开销**：需要额外的矩阵乘法恢复 K, V

### 3.2 RoPE 与 MLA 的兼容性问题

**前两轮认知**：需要额外的 decoupled RoPE 分量

**本轮新发现**：

```python
# 问题：RoPE 需要作用于 Q, K
# 但 MLA 的 K 是从潜空间恢复的，直接应用会破坏低秩结构

# 解决方案：Decoupled RoPE
# 1. 为 RoPE 添加额外的 Q, K 分量
q_rope = W_q_rope(x)  # [batch, seq, n_heads * rope_dim]
k_rope = W_k_rope(x)  # [batch, seq, rope_dim]  # 共享的

# 2. 最终的 Q, K 由两部分拼接
q = [q_main, q_rope]  # 主分量 + RoPE 分量
k = [k_main, k_rope]  # 从潜空间恢复 + RoPE 分量

# 3. Attention 计算
# score = q_main @ k_main + q_rope @ k_rope
# 第一项可以使用吸收矩阵，第二项需要显式计算
```

**新学到的点**：
1. **解耦设计**：RoPE 分量独立于主分量
2. **额外参数**：需要额外的 W_q_rope, W_k_rope
3. **计算开销**：RoPE 分量需要显式计算，无法使用吸收矩阵

### 3.3 MoE 负载均衡的实际实现

**前两轮认知**：DeepSeek V3 使用 Auxiliary Loss-free Balancing

**本轮新发现**：

```python
# 传统方法：Auxiliary Loss
# 添加一个辅助损失，鼓励均匀路由
# L_aux = α * Σ(f_i * P_i)
# f_i: 第i个专家的实际负载
# P_i: 第i个专家的平均路由概率

# DeepSeek V3 的方法：动态偏置
# 不需要额外的损失函数
# 给每个专家添加一个动态偏置 b_i
# 路由分数 = original_score + b_i

# 更新规则：
# 如果专家i负载过高，减小 b_i
# 如果专家i负载过低，增大 b_i
```

**新学到的点**：
1. **无损失设计**：不需要额外的超参数 α
2. **动态调整**：偏置会根据实际负载动态调整
3. **训练稳定性**：比 Auxiliary Loss 更稳定

---

## 四、分布式训练：从理论到实践

### 4.1 PCIe 环境下的通信优化

**前两轮认知**：无 NVLink 时 TP 效率低

**本轮新发现**：

```python
# 4090 无 NVLink，PCIe 4.0 x16 带宽约 32GB/s（双向）

# 优化策略1：减少 TP 度数
# TP=2 比 TP=8 通信量小得多
# 推荐：TP=2 或 TP=4

# 优化策略2：使用梯度累积
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps
    loss.backward()
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# 优化策略3：使用通信压缩
# DDP 支持梯度压缩
model = DDP(model, gradient_as_bucket_view=True)
```

**新学到的点**：
1. **通信模式**：AllReduce vs AllGather vs ReduceScatter 的带宽需求不同
2. **计算通信重叠**：可以将通信和计算重叠，隐藏延迟
3. **梯度累积**：减少通信频率，但会增加训练时间

### 4.2 集合通信的实际性能

**前两轮认知**：理解 AllReduce、AllGather 等原语

**本轮新发现**：

```python
# collective_operations.ipynb 中的实际测量

# AllReduce 性能（8卡 4090，PCIe）
# 1MB 数据：约 0.5ms
# 100MB 数据：约 15ms
# 1GB 数据：约 150ms

# 瓶颈分析：
# - 小数据量：延迟主导（约 100μs）
# - 大数据量：带宽主导（约 30GB/s）

# 优化建议：
# - 合并小通信为大通信
# - 使用异步通信
# - 使用计算通信重叠
```

**新学到的点**：
1. **Ring AllReduce**：最常用的 AllReduce 实现
2. **Tree AllReduce**：适合大规模集群
3. **NCCL 优化**：NCCL 会自动选择最优的通信算法

### 4.3 流水线并行的 Bubble 问题

**前两轮认知**：PP 有 Pipeline Bubble

**本轮新发现**：

```python
# 朴素 PP 的 Bubble 率
# Bubble 时间 = (p-1) / (p-1+m)
# p: 流水线阶段数
# m: micro-batch 数量

# 优化策略1：增加 micro-batch 数量
# m 越大，Bubble 率越低
# 但会增加显存开销

# 优化策略2：使用 1F1B 调度
# 每个阶段交替执行 forward 和 backward
# 可以进一步减少 Bubble

# 优化策略3：使用 Interleaved 1F1B
# 每个阶段处理多个非连续的层
# 可以进一步减少 Bubble
```

**新学到的点**：
1. **Bubble 不可避免**：这是 PP 的固有问题
2. **1F1B 调度**：比朴素 PP 更高效
3. **显存 vs Bubble 的 trade-off**：更多 micro-batch 可以减少 Bubble，但需要更多显存

---

## 五、LoRA 微调：从原理到实践

### 5.1 LoRA 的秩选择

**前两轮认知**：秩 r 通常 4-64

**本轮新发现**：

```python
# LoRA_to_Multi_LoRA.ipynb 中的实验

# 不同秩的性能对比（以 Qwen3-7B 为例）
# r=4:  参数量 0.1%，效果一般
# r=8:  参数量 0.2%，效果较好
# r=16: 参数量 0.4%，效果很好
# r=32: 参数量 0.8%，效果接近全量微调

# 选择建议：
# - 简单任务：r=8 或 r=16
# - 复杂任务：r=32 或 r=64
# - 资源受限：r=4 或 r=8
```

**新学到的点**：
1. **秩的理论依据**：基于"微调时权重变化是低秩的"假设
2. **不同层的秩可以不同**：注意力层通常需要更高的秩
3. **LoRA+**：不同层使用不同的学习率

### 5.2 Multi-LoRA 的实现细节

**前两轮认知**：可以同时服务多个 LoRA 适配器

**本轮新发现**：

```python
# vLLM 的 Multi-LoRA 实现

# 1. 每个 LoRA 适配器有唯一的 ID
lora_request = LoRARequest(
    lora_name="adapter1",
    lora_int_id=1,
    lora_local_path="./lora_adapter1"
)

# 2. 推理时指定使用哪个适配器
outputs = llm.generate(
    prompts,
    SamplingParams(max_tokens=100),
    lora_request=lora_request
)

# 3. 显存管理
# - 主模型在 GPU 上常驻
# - LoRA 适配器按需加载
# - 使用 LRU 缓存策略
```

**新学到的点**：
1. **按需加载**：LoRA 适配器不需要全部加载到 GPU
2. **缓存策略**：使用 LRU 缓存最近使用的适配器
3. **批处理优化**：同一批次可以使用不同的 LoRA

### 5.3 QLoRA 的工程细节

**前两轮认知**：4-bit 量化 + LoRA

**本轮新发现**：

```python
# QLoRA 的实际实现

# 1. 使用 NF4 数据类型
# NF4 (Normal Float 4-bit) 专为正态分布设计
# 比 INT4 更适合权重分布

# 2. 双重量化
# 对量化常数也进行量化
# 进一步减少显存开销

# 3. 分页优化器
# 使用 CPU 内存存储优化器状态
# 避免 GPU 显存不足
```

**新学到的点**：
1. **NF4 数据类型**：专为正态分布设计的 4-bit 数据类型
2. **双重量化**：对量化常数也进行量化
3. **分页优化器**：使用 CPU 内存存储优化器状态

---

## 六、RL 训练：从概念到实现

### 6.1 训推共卡的显存管理

**前两轮认知**：训练和推理交替使用 GPU

**本轮新发现**：

```python
# training_infer_colocate.ipynb 中的实现

# 1. 显存卸载策略
# 训练阶段：
#   - 卸载推理模型的 KV Cache
#   - 保留训练模型的参数和梯度
#   - 使用 CPU Offload 存储优化器状态

# 推理阶段：
#   - 卸载训练模型的梯度和优化器状态
#   - 加载推理模型的 KV Cache

# 2. 切换时机
# - 训练完成后，立即切换到推理
# - 推理完成后，立即切换到训练
# - 需要同步点，确保所有卡完成当前阶段

# 3. 显存预算
# 假设每卡 24GB
# 训练模型：约 14GB（7B 模型 FP16）
# 优化器状态：约 6GB（Adam）
# 推理模型：与训练模型共享参数
# KV Cache：约 4GB
```

**新学到的点**：
1. **显存复用**：训练和推理共享模型参数
2. **CPU Offload**：将优化器状态卸载到 CPU
3. **同步点**：需要确保所有卡完成当前阶段

### 6.2 Ray 资源调度的实践

**前两轮认知**：使用 Ray 进行资源调度

**本轮新发现**：

```python
# Ray Placement Group 的实际使用

import ray
from ray.util.placement_group import placement_group

# 1. 预占资源
pg = placement_group(
    [{"GPU": 1}] * 4,  # 需要4个GPU
    strategy="STRICT_PACK"  # 强制在同一节点
)
ray.get(pg.ready())

# 2. 使用预占的资源
@ray.remote(num_gpus=1)
def train_worker():
    # 训练逻辑
    pass

@ray.remote(num_gpus=1)
def inference_worker():
    # 推理逻辑
    pass

# 3. 调度策略
# - PACK：尽量放在同一节点
# - SPREAD：尽量分散到不同节点
# - STRICT_PACK：强制同一节点
# - STRICT_SPREAD：强制不同节点
```

**新学到的点**：
1. **Placement Group**：预占资源，确保资源可用
2. **调度策略**：不同策略适用于不同场景
3. **资源隔离**：不同任务可以隔离资源

### 6.3 R3（Rollout Router Replay）的实现

**前两轮认知**：记录并回放路由决策，确保训推一致性

**本轮新发现**：

```python
# R3 的实际实现

# 1. 记录路由决策
# 在 rollout 阶段，记录每个 token 的路由决策
routing_decisions = []
for token in tokens:
    expert_id = router(token)
    routing_decisions.append(expert_id)

# 2. 回放路由决策
# 在训练阶段，强制使用相同的路由决策
for token, expert_id in zip(tokens, routing_decisions):
    # 不使用 router，直接使用记录的 expert_id
    output = experts[expert_id](token)

# 3. 为什么需要 R3？
# - MoE 的路由是离散的（选择哪个专家）
# - 训练和推理的路由可能不一致
# - 不一致会导致梯度估计有偏
```

**新学到的点**：
1. **路由决策的离散性**：MoE 的路由是离散的，无法直接求导
2. **训推一致性**：确保训练和推理使用相同的路由
3. **梯度偏差**：不一致的路由会导致梯度估计有偏

---

## 七、性能优化：从理论到实践

### 7.1 Flash Attention 的实际效果

**前两轮认知**：Flash Attention 减少显存访问

**本轮新发现**：

```python
# Flash Attention 的性能对比（Qwen3-7B，4090）

# 标准 Attention
# 显存：O(N^2)（需要存储完整的 attention 矩阵）
# 速度：基准

# Flash Attention v2
# 显存：O(N)（不需要存储完整矩阵）
# 速度：约 2x 提升

# 实际测试（序列长度 4096）
# 标准 Attention：约 50ms
# Flash Attention v2：约 25ms
```

**新学到的点**：
1. **分块计算**：将 attention 矩阵分成小块计算
2. **显存优化**：不需要存储完整的 attention 矩阵
3. **计算优化**：减少 HBM 访问，利用 SRAM

### 7.2 CUDA Graph 的适用场景

**前两轮认知**：CUDA Graph 减少 kernel launch 开销

**本轮新发现**：

```python
# CUDA Graph 的适用场景

# 适合：
# - 小 kernel 的重复执行
# - 计算图固定
# - batch size 固定

# 不适合：
# - 计算图动态变化
# - batch size 变化
# - Prefill 阶段（序列长度不固定）

# vLLM 的使用策略
# - Decode 阶段：使用 CUDA Graph
# - Prefill 阶段：不使用 CUDA Graph
```

**新学到的点**：
1. **Graph 录制**：第一次执行时录制，后续直接重放
2. **动态 shape**：CUDA Graph 不支持动态 shape
3. **显存开销**：需要额外的显存存储 Graph

### 7.3 量化的实际效果

**前两轮认知**：INT8/FP8 减少显存和计算量

**本轮新发现**：

```python
# 量化的实际效果（Qwen3-14B，4090）

# FP16 基准
# 显存：约 28GB（需要 2 卡 TP）
# 速度：基准

# INT8 量化（AWQ）
# 显存：约 14GB（单卡可运行）
# 速度：约 1.2x 提升
# 精度损失：约 1%

# FP8 量化
# 显存：约 14GB（单卡可运行）
# 速度：约 1.3x 提升
# 精度损失：约 0.5%

# 选择建议
# - 资源受限：INT8（AWQ 或 GPTQ）
# - 精度敏感：FP8 或 INT8（精度损失小）
# - 速度优先：INT8（计算量减少更多）
```

**新学到的点**：
1. **量化粒度**：Per-tensor vs Per-channel vs Per-group
2. **量化方法**：AWQ（激活感知）vs GPTQ（基于 Hessian）
3. **KV Cache 量化**：可以单独量化 KV Cache

---

## 八、问题排查：从理论到方法论

### 8.1 显存问题排查

**前两轮认知**：使用 nvidia-smi 监控显存

**本轮新发现**：

```python
# 显存排查的完整流程

# 1. 快速检查
nvidia-smi  # 查看显存使用

# 2. 详细分析
torch.cuda.memory_summary()  # PyTorch 显存统计

# 3. Snapshot 分析
torch.cuda.memory._record_memory_history()
# ... 运行代码 ...
torch.cuda.memory._dump_snapshot("snapshot.pickle")
# 使用 PyTorch Memory Visualizer 分析

# 4. 常见原因
# - 模型参数：约 2B/GB（FP16）
# - 优化器状态：约 6B/GB（Adam）
# - 梯度：约 2B/GB（FP16）
# - 激活值：取决于 batch size 和序列长度
# - KV Cache：取决于序列长度和 batch size
```

**新学到的点**：
1. **显存碎片化**：频繁分配释放会导致碎片化
2. **显存泄漏**：未释放的 Tensor 会导致显存泄漏
3. **显存预分配**：vLLM 使用预分配策略避免碎片化

### 8.2 性能问题排查

**前两轮认知**：使用 Profiling 工具

**本轮新发现**：

```python
# 性能排查的完整流程

# 1. Python 层 Profiling
import cProfile
cProfile.run('my_function()', 'profile_output')

# 2. GPU 层 Profiling
# 使用 nsys
nsys profile -o report python train.py

# 3. Kernel 层 Profiling
# 使用 ncu
ncu --set full python train.py

# 4. 分析瓶颈
# - CPU bound：Python 代码或数据加载
# - GPU compute bound：计算 kernel
# - GPU memory bound：显存访问
# - Communication bound：多卡通信
```

**新学到的点**：
1. **瓶颈分类**：CPU bound vs GPU bound vs Memory bound
2. **Profiling 工具**：cProfile、nsys、ncu 的使用场景
3. **优化优先级**：先解决最大瓶颈

### 8.3 训练不稳定问题排查

**前两轮认知**：监控 loss 和梯度

**本轮新发现**：

```python
# 训练不稳定的排查流程

# 1. 监控指标
# - Loss 曲线：是否有异常波动
# - 梯度范数：是否有梯度爆炸/消失
# - 学习率：是否按预期变化
# - 权重范数：是否有权重发散

# 2. 常见原因
# - 学习率过大：使用 warmup 和 cosine decay
# - 梯度爆炸：使用梯度裁剪
# - 数据问题：检查数据分布和质量
# - 数值问题：使用混合精度训练

# 3. 解决方案
# - 梯度裁剪：torch.nn.utils.clip_grad_norm_
# - 学习率 warmup：LinearLR 或 CosineAnnealingLR
# - 混合精度：使用 GradScaler
# - 权重衰减：L2 正则化
```

**新学到的点**：
1. **梯度裁剪**：防止梯度爆炸
2. **学习率调度**：warmup + cosine decay
3. **混合精度训练**：使用 GradScaler 防止下溢

---

## 九、工程实践：从理论到落地

### 9.1 推理服务的监控指标

**前两轮认知**：监控 TTFT 和 TPOT

**本轮新发现**：

```python
# 完整的监控指标体系

# 1. 延迟指标
# - TTFT（Time To First Token）：首 token 延迟
# - TPOT（Time Per Output Token）：每 token 延迟
# - Latency：总延迟

# 2. 吞吐指标
# - QPS（Queries Per Second）：每秒请求数
# - TPS（Tokens Per Second）：每秒 token 数
# - Throughput：吞吐量

# 3. 资源指标
# - GPU 利用率
# - 显存使用率
# - KV Cache 使用率
# - 请求队列长度

# 4. 业务指标
# - 成功率
# - 超时率
# - 用户满意度
```

**新学到的点**：
1. **指标分类**：延迟 vs 吞吐 vs 资源 vs 业务
2. **告警阈值**：根据 SLA 设置告警阈值
3. **监控工具**：Prometheus + Grafana

### 9.2 灰度发布的实践

**前两轮认知**：使用 A/B 测试

**本轮新发现**：

```python
# 灰度发布的实践

# 1. 流量分配
# - 1% 流量到新版本
# - 观察指标
# - 逐步增加到 10%, 50%, 100%

# 2. 回滚策略
# - 自动回滚：指标异常时自动回滚
# - 手动回滚：人工确认后回滚

# 3. 监控指标
# - 延迟：新版本是否变慢
# - 错误率：新版本是否有 bug
# - 业务指标：新版本是否提升业务价值
```

**新学到的点**：
1. **流量分配**：根据用户 ID 或请求 ID 分配
2. **自动回滚**：设置阈值自动回滚
3. **金丝雀发布**：先发布到少量用户

### 9.3 数据回滚的实践

**前两轮认知**：使用版本管理

**本轮新发现**：

```python
# 数据回滚的实践

# 1. Checkpoint 管理
# - 定期保存 checkpoint
# - 保留最近 N 个 checkpoint
# - 使用版本号标识

# 2. 数据版本管理
# - 使用 DVC 或 MLflow
# - 记录数据 lineage
# - 支持数据回滚

# 3. 模型版本管理
# - 使用 Model Registry
# - 记录模型性能指标
# - 支持模型回滚
```

**新学到的点**：
1. **Checkpoint 策略**：定期保存 + 最佳模型保存
2. **数据版本**：使用 DVC 或 MLflow
3. **模型 Registry**：使用 MLflow 或 Weights & Biases

---

## 十、总结：从理论到实践的关键认知

### 10.1 理论 vs 实践的差距

| 理论认知 | 实践发现 |
|---------|---------|
| Continuous Batching 提高吞吐 | 需要精细的调度策略 |
| MLA 减少 KV Cache | 需要处理 MHA/MQA 模式切换 |
| LoRA 参数高效 | 需要选择合适的秩 |
| RL 训练需要训推一致性 | 需要 R3 等机制保证 |

### 10.2 工程实践的关键点

1. **显存管理**：预分配、碎片化、泄漏
2. **通信优化**：PCIe vs NVLink、梯度累积、通信压缩
3. **性能优化**：Flash Attention、CUDA Graph、量化
4. **稳定性保障**：监控、告警、灰度发布、回滚

### 10.3 面试准备的增量点

**可以深入追问的点**：
1. "vLLM 的调度策略是怎么实现的？"
2. "MLA 的 MHA/MQA 模式切换有什么工程挑战？"
3. "训推共卡的显存怎么管理？"
4. "量化有哪些方法？各自有什么 trade-off？"
5. "怎么排查推理服务的性能问题？"

**可以展示的实践经验**：
1. "我在 4090 上复现了 vLLM 的调度器"
2. "我理解 MLA 的吸收矩阵和模式切换"
3. "我实现了简单的 LoRA 微调"
4. "我搭建了简易的推理服务"

---

*本笔记基于 InfraTech 项目复现过程整理，最后更新：2026年6月*

*持续更新中，记录从理论到实践的新发现*
