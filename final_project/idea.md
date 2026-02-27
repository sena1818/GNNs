
# Proposal 2: Solving TSP via Diffusion Models

## 1. Scientific Problem and Research Value

**Core Problem**: Can discrete combinatorial optimization (NP-hard TSP) be reformulated as continuous generative modeling to achieve gradient-based learning with cross-scale generalization?

**Research Value**: TSP is a classical problem in combinatorial optimization with applications in logistics, chip design, and robotic navigation. Recent integration of diffusion models and GNNs provides a new paradigm, transforming discrete search space (N! permutations) into continuous probability modeling to generate near-optimal solutions (< 2% gap) in < 1 second. This study aims to: (1) Explore whether lightweight GNN architectures (GAT, GCN) maintain solution quality while significantly reducing computational cost; (2) Validate cross-scale generalization—whether training on small instances (TSP-50) enables solving larger problems (TSP-100/200); (3) Explore diverse solution strategies through probabilistic generation for emergency planning (e.g., road closures in logistics), combined with traditional 2-opt refinement.
## 2. Literature Review

**Core Papers**:
- **T2T-CO** (Chen et al., NeurIPS 2024): Anisotropic GNN + diffusion achieving 0.3% optimality gap on TSP-100 (SOTA), but high computational cost with Graph Transformer
- **DIFUSCO** (Sun et al., NeurIPS 2023): First TSP diffusion framework with complex discrete decoding
- **LKH Algorithm** (Helsgaun 2017): Near-optimal but slow (hours for TSP-10000), no learning capability
- **Attention Model** (Kool et al., ICLR 2019): RL end-to-end with poor generalization (TSP-50→100 gap > 10%)

**Our Extensions**: 
(1) Lightweight architecture ablation comparing Graph Transformer vs. GAT vs. GCN for efficiency-performance tradeoffs; 
(2) Cross-scale generalization testing (TSP-50 training → TSP-20/100/200 testing, target: TSP-100 gap < 2%); 
(3) Multi-modal diffusion generating 10 diverse high-quality paths; 
(4) Hybrid solver combining diffusion initialization with 2-opt local refinement.

## 3. Methodology and Feasibility

**Technical Roadmap**: 
(1) **Infrastructure setup**: Configure PyTorch Geometric and PyConcorde, generate multi-scale TSP instances (TSP-20: 1000, TSP-50: 5000) with LKH-computed optimal solutions; 
(2) **Core model implementation**: Anisotropic graph convolution encoder (4 layers) with separate weight matrices for in/out edges, U-Net diffusion decoder predicting denoising trajectory for adjacency matrix A ∈ {0,1}^(N×N), forward noising q(A_t|A_0) and reverse denoising with direct x₀ prediction; 
(3) **Extension experiments**: Architecture ablation (Graph Transformer/GAT/GCN), cross-scale generalization (TSP-50 training → TSP-20/100/200 testing), decoding strategy optimization (beam search k=10, greedy, temperature sampling), multi-modal generation (10-choose-1 strategy), hybrid solver (diffusion + 2-opt); 
(4) **Evaluation**: Quantitative metrics (optimality gap, inference time, success rate) and visualization (diffusion process GIF, edge probability heatmap, path comparison).

**Feasibility**: 
(1) T2T-CO provides complete implementation framework; 
(2) Self-generated data eliminates external dependencies, LKH solver produces deterministic labels quickly; 
(3) Memory requirements (~10GB) within Google Colab T4 capacity; 
(4) Fast inference (TSP-100 < 1 second) enables rapid experimentation; 
(5) NeurIPS 2024 paper provides detailed implementation and hyperparameters.

## 4. Data Source

**Self-Generated TSP Instances**: Sample N city coordinates uniformly from [0,1]², compute optimal paths via LKH solver (exact for N ≤ 1000), convert to adjacency matrix A[i,j]=1 (if edge in path), augment via rotation/reflection (×8 per instance).
**Scale and Quantity**: TSP-20: 1000 instances (debugging), TSP-50: 5000 instances (main training set), TSP-100/200: 1000/500 instances (generalization testing).

## 5. Computational Resources

**Hardware**: Google Colab free tier (T4 GPU)
**Estimated Time**: Data generation 5h (LKH-dominated), TSP-20/50 training 20h, architecture ablation 20h (3 variants × 5h), hybrid solver experiments 5h, evaluation and visualization 5h. 
**Total GPU time**: ~50h.

## 6. Potential Risks and Mitigation

**Main Risks**: 
(1) **Discrete decoding difficulty**: Decoding from continuous probability matrix to valid TSP path may fail. Mitigation: Use T2T-CO validated beam search, fallback to greedy decoding if needed; 
(2) **Cross-scale generalization degradation**: Model may underperform on large instances (100/200 nodes). Mitigation: Mixed-scale training; even if suboptimal, analyze failure modes as research findings; 
(3) **GNN training instability**: Deep GNNs risk over-smoothing or gradient vanishing. Mitigation: Exponential moving average (EMA) for weights, learning rate warmup; 
(4) **Resource constraints**: GPU memory/time limitations. Mitigation: Reduce batch size (32→16) with gradient accumulation, prioritize TSP-50 core experiments, simplify ablation if necessary.

# 提案3：扩散模型求解组合优化问题 (TSP)

> **基于**: NeurIPS 2023/2024 最新论文
> **方向**: AI for Math - 扩散模型 + 图神经网络 + 组合优化
> **核心创新**: 将"找最优路径"转化为"生成图像"

---

## 🎯 **项目概述**

**项目标题**: "Diffusion Models for Combinatorial Optimization: Learning to Solve TSP via Generative Modeling"

**核心思路**:
将旅行商问题（TSP）的求解过程转化为**图像生成**任务。把最优路径表示为一个N×N的热力图（Heatmap），使用扩散模型学习从噪声中"画出"最优连接路径。

**为什么新颖？**
- ✅ **跨领域创新**: 扩散模型（CV）+ 图神经网络（GNNs）+ 组合优化（OR）
- ✅ **2023-2024最新**: NeurIPS 2023/2024热门方向 "AI for Math"
- ✅ **与课程完美契合**: GNNs + 生成模型
- ✅ **可视化效果震撼**: 能生成扩散过程的动画GIF

---

## 📚 **核心论文**

### **主论文1: DIFUSCO**
**Diffusion Model for Combinatorial Optimization**
- **会议**: NeurIPS 2023
- **论文**: https://arxiv.org/abs/2303.18138
- **代码**: ✅ 有官方PyTorch实现
- **核心创新**: 首次将扩散模型系统化应用到TSP等离散优化问题

### **主论文2: T2T-CO (Fast T2T)**
**Graph Generation via Diffusion for TSP**
- **会议**: NeurIPS 2024
- **代码**: https://github.com/yongchao97/t2t-co
- **核心创新**:
  - Anisotropic GNN（各向异性图神经网络）
  - 加速的Beam Search解码
  - 在TSP-100上达到**0.3% Optimality Gap**

### **对比论文: LKH算法**
- 传统组合优化经典算法
- 用途: 作为Baseline对比

---

## 🔬 **核心原理详解**

### **1. 核心直觉：把"找路径"变成"画图"**

**传统方法** (LKH算法):
- 在不断尝试交换城市连接的顺序
- 离散搜索空间，组合爆炸

**扩散模型方法** (DIFUSCO/T2T):
- 把最优路径看作一张**热力图**（Heatmap）
- 在N×N网格上：
  - 如果城市i和城市j相连 → 格子(i,j) = 1（黑色）
  - 否则 → 格子(i,j) = 0（白色）

**扩散过程**:
```
Forward (加噪):
  清晰路径图 + 高斯噪声 → 灰蒙蒙的噪声图

Reverse (去噪):
  噪声图 + 城市坐标信息 → 清晰的连接路径
```

**妙处**: 模型不需要显式计算距离，而是像"看图说话"一样，直接"画出"哪里应该有路。

---

### **2. 技术架构拆解**

```
输入: N个城市的2D坐标 {(x1, y1), (x2, y2), ..., (xN, yN)}
     ↓
[模块A] 图编码器 (Graph Encoder)
  - 网络: Anisotropic GCN 或 Graph Transformer
  - 作用: 提取城市之间的几何特征
  - 输出: 每个城市的Embedding向量
     ↓
[模块B] 扩散去噪器 (Diffusion Decoder)
  - 输入: 图Embedding + 当前带噪声的连接矩阵 X_t
  - 网络: U-Net风格的GNN
  - 输出: 预测噪声 ε 或直接预测原图 X_0
  - 结果: 概率热力图（越亮的地方表示应该有连接）
     ↓
[模块C] 离散解码 (Discrete Decoding)
  - 问题: 扩散输出是概率(0.8, 0.1, 0.9...)，需要离散的0或1
  - 方法: Beam Search在概率热力图上走出合法闭环
  - 输出: 最终的TSP路径
```

---

## 💻 **项目实施计划** (180小时)

### **阶段1: 复现与理解** (Week 1-3, 70h)

**任务清单**:
- [ ] **环境配置** (10h)
  - 克隆T2T-CO或DIFUSCO代码仓库
  - 配置PyTorch + PyTorch Geometric环境
  - 安装TSP求解器（Concorde/LKH，用于生成Ground Truth）

- [ ] **数据生成** (15h)
  - 生成TSP-20数据集（1000个实例）
  - 用LKH求解器生成最优解作为标签
  - 将路径转换为邻接矩阵（热力图格式）

- [ ] **模型复现** (30h)
  - 实现图编码器（Graph Transformer或GAT）
  - 实现扩散去噪模块（U-Net GNN）
  - 训练在TSP-20上，观察Loss下降

- [ ] **可视化调试** (15h)
  - 可视化扩散过程（t=1000 → t=0）
  - 画出预测路径 vs 最优路径
  - 验证模型是否学到合理连接

**交付物**:
- 能在TSP-20上训练的完整代码
- 扩散过程可视化GIF
- 初步的Optimality Gap报告

---

### **阶段2: 扩展与改进** (Week 4-6, 70h)

**核心改进方向**（选择其中2-3个）:

#### **改进1: 轻量化架构实验** (25h)
- **动机**: 原论文用Graph Transformer很重，训练慢
- **实验**:
  - Baseline: Graph Transformer（原论文）
  - 变体1: GAT (Graph Attention Network)
  - 变体2: GCN (简单图卷积)
- **评估**: 性能 vs 训练速度权衡
- **创新点**: 消融实验，找到最优架构

#### **改进2: 泛化性测试** (25h)
- **动机**: 模型在训练规模上泛化能力未知
- **实验**:
  - 用TSP-50训练
  - 测试在TSP-20, TSP-50, TSP-100上的表现
  - 分析：模型能否"举一反三"？
- **创新点**: 首次系统测试扩散模型在TSP上的泛化能力

#### **改进3: 解码策略优化** (20h)
- **动机**: Beam Search计算慢
- **实验**:
  - Baseline: Beam Search (k=10)
  - 变体1: Greedy Search（更快）
  - 变体2: Sampling（随机采样多条路径）
- **评估**: 解的质量 vs 解码速度
- **创新点**: 实时TSP求解（适合动态场景）

**交付物**:
- 2-3个改进实验的完整结果
- 消融实验表格
- 泛化性能曲线图

---

### **阶段3: 评估与可视化** (Week 7, 40h)

**任务清单**:
- [ ] **核心指标计算** (15h)
  - **Optimality Gap**: (模型解 - 最优解) / 最优解
    - 目标: < 1% (TSP-50), < 2% (TSP-100)
  - **求解时间**: 对比LKH算法
  - **成功率**: 有效解的比例（闭环、访问所有城市）

- [ ] **震撼可视化** (15h) 🎬
  - **扩散过程GIF**:
    - t=1000: 杂乱噪声
    - t=500: 隐约轮廓
    - t=0: 清晰路径
  - **多实例对比图**:
    - 同一个TSP实例：LKH解 vs 扩散模型解
  - **热力图可视化**:
    - 概率分布（哪些边概率高）

- [ ] **项目报告撰写** (10h)
  - 引言: TSP + 扩散模型背景
  - 方法: 架构详解
  - 实验: 改进结果 + 可视化
  - 结论: 优势与局限

**交付物**:
- 15-20页项目报告（PDF）
- 展示PPT（10分钟演讲）
- 完整代码 + README

---

## 📊 **预期成果**

### **定量指标**
- **TSP-20**: Optimality Gap < 0.5%，求解时间 < 0.1s
- **TSP-50**: Optimality Gap < 1%，求解时间 < 1s
- **TSP-100**: Optimality Gap < 2%，求解时间 < 5s

### **定性成果**
- ✅ 扩散过程可视化动画（最震撼）
- ✅ 消融实验表格（轻量化架构）
- ✅ 泛化性能分析（跨规模测试）
- ✅ 完整的开源代码

---

## 🎯 **与课程结合**

| 课程内容 | 项目对应 | 结合度 |
|---------|---------|--------|
| **GNNs课程名称** | 图神经网络编码器 | ⭐⭐⭐⭐⭐ |
| **VAE (T1)** | 生成模型思想（隐空间） | ⭐⭐⭐⭐ |
| **Normalizing Flows (sheet_3)** | 连续空间变换（对比） | ⭐⭐⭐ |
| **密度估计 (sheet_1)** | 概率热力图 | ⭐⭐⭐⭐ |

**核心亮点**:
- ✅ **与GNNs课程名完美契合**（图生成 + 图神经网络）
- ✅ 扩散模型（课程核心技术）
- ✅ 组合优化（拓展应用领域）

---

## 🔥 **为什么选这个项目？**

### **1. 技术创新性** ⭐⭐⭐⭐⭐
- 扩散模型 + 图神经网络 + 组合优化的**三重交叉**
- NeurIPS 2023/2024最火的"AI for Math"方向
- 原论文本身就是顶会Oral/Spotlight

### **2. 可视化效果震撼** ⭐⭐⭐⭐⭐
- 扩散过程动画GIF（从噪声到路径）
- 路径对比图（模型 vs 最优解）
- 热力图可视化（概率分布）
- **做Presentation效果极佳**

### **3. 科学价值高** ⭐⭐⭐⭐⭐
- 运筹学（OR）+ 深度学习（DL）交叉前沿
- TSP是NP-hard问题，有实际应用（物流、芯片设计）
- 扩散模型在离散优化的新范式

### **4. 代码成熟度高** ⭐⭐⭐⭐⭐
- T2T-CO有完整PyTorch实现（2024年更新）
- DIFUSCO有官方代码
- PyTorch Geometric生态成熟

### **5. 发表潜力** ⭐⭐⭐⭐
- **会议**: NeurIPS/ICML AI4Math Workshop
- **期刊**: Operations Research, Mathematical Programming
- **创新点**: 轻量化架构、泛化性分析

---

## 🚀 **数据与资源**

### **数据集**
- **自己生成**: 用代码随机生成TSP实例
- **Ground Truth**: 用Concorde/LKH求解器生成最优解
- **规模**: TSP-20（调试）→ TSP-50（训练）→ TSP-100（测试）

### **代码资源**
- **T2T-CO**: https://github.com/yongchao97/t2t-co ✅
- **DIFUSCO**: https://github.com/Edward-Sun/DIFUSCO ✅
- **TSP求解器**:
  - Python: `concorde` 库
  - 或: LKH求解器（C++）

### **GPU需求**
- **训练**: Colab T4即可（TSP-50约10小时）
- **推理**: CPU可运行（单个实例 < 1秒）

---

## 📋 **技术难点与解决方案**

### **难点1: 离散解码**
- **问题**: 扩散输出是连续概率，需要离散化为合法路径
- **解决**:
  - 使用Beam Search（T2T论文方法）
  - 或Greedy Search（更快）
  - 添加合法性约束（必须是闭环）

### **难点2: 大规模TSP泛化**
- **问题**: TSP-50训练的模型在TSP-100上可能失效
- **解决**:
  - 数据增强（训练时混合不同规模）
  - 或使用Position Encoding（标记节点位置）

### **难点3: 训练不稳定**
- **问题**: 扩散模型训练可能震荡
- **解决**:
  - 使用Exponential Moving Average (EMA)
  - 调整学习率（Warmup + Cosine Decay）

---

## 🎨 **可视化示例**

### **扩散过程动画**
```
Frame 1 (t=1000): [噪声满满的灰图]
Frame 2 (t=750):  [隐约看到一些线条]
Frame 3 (t=500):  [轮廓逐渐清晰]
Frame 4 (t=250):  [几乎看清路径]
Frame 5 (t=0):    [清晰的城市连线]
```

### **路径对比图**
```
左图: LKH最优解（蓝色路径）
右图: 扩散模型解（红色路径）
下方: Optimality Gap = 0.8%
```

### **热力图可视化**
```
N×N矩阵，颜色越深表示连接概率越高
对角线为0（城市不能自己连自己）
对称矩阵（无向图）
```

---

## 💡 **创新点总结**

1. ✅ **首次系统测试轻量化架构**（GAT vs Transformer）
2. ✅ **首次分析扩散TSP的跨规模泛化能力**
3. ✅ **实时TSP求解**（Greedy解码 < 0.1秒）
4. ✅ **震撼的可视化**（扩散动画GIF）

---

## 📖 **关键文献清单**

### **必读论文**
1. ⭐⭐⭐⭐⭐ **DIFUSCO** (NeurIPS 2023)
   - https://arxiv.org/abs/2303.18138
   - 重点: Section 3（扩散TSP架构）

2. ⭐⭐⭐⭐⭐ **T2T-CO** (NeurIPS 2024)
   - https://github.com/yongchao97/t2t-co
   - 重点: Anisotropic GNN + Beam Search

3. ⭐⭐⭐⭐ **Denoising Diffusion Probabilistic Models** (NeurIPS 2020)
   - https://arxiv.org/abs/2006.11239
   - 扩散模型基础理论

### **选读论文**
4. ⭐⭐⭐ **LKH算法** (传统OR方法)
   - 理解传统TSP求解器

5. ⭐⭐⭐ **Graph Attention Networks** (ICLR 2018)
   - GAT架构（轻量化实验）

---

## 🏆 **最终推荐理由**

这个项目是**扩散模型方向**中最独特的一个，因为：

1. **极客指数最高** 🤓
   - 把"找路径"变成"画图"，思路新颖
   - 有图有真相，可视化震撼

2. **技术深度强** 🔬
   - 扩散模型 + GNN + 组合优化三重交叉
   - 涉及离散优化、概率建模、图表示学习

3. **与课程完美契合** 🎯
   - GNNs课程名（图生成）
   - 扩散模型（生成模型核心）
   - 密度估计（概率热力图）

4. **发表潜力高** 📄
   - AI4Math是NeurIPS热门Workshop
   - 轻量化、泛化性都是未充分探索的方向

5. **实用价值** 💼
   - TSP应用广泛（物流、芯片、机器人）
   - 实时求解（< 1秒）有商业价值

---

## 📌 **与其他提案对比**

| 维度 | 提案3 (扩散TSP) | 提案1 (PMDM药物) | 提案2 (物理约束电池) |
|------|----------------|-----------------|-------------------|
| **新颖度** | ⭐⭐⭐⭐⭐ AI4Math前沿 | ⭐⭐⭐⭐ Nature论文 | ⭐⭐⭐⭐⭐ Spotlight |
| **可视化** | ⭐⭐⭐⭐⭐ 动画GIF | ⭐⭐⭐ 分子结构 | ⭐⭐⭐⭐ 3D电极 |
| **课程契合** | ⭐⭐⭐⭐⭐ GNNs完美 | ⭐⭐⭐⭐ 3D生成 | ⭐⭐⭐⭐ 物理约束 |
| **代码成熟** | ⭐⭐⭐⭐⭐ 2024最新 | ⭐⭐⭐⭐ 有官方 | ⭐⭐⭐ 预计有 |
| **极客指数** | ⭐⭐⭐⭐⭐ 最高 | ⭐⭐⭐⭐ 高 | ⭐⭐⭐⭐ 高 |

---

## 🚀 **下一步行动**

如果选择这个提案，我将为您准备：

1. ✅ **1页项目提案说明**（提交给老师）
2. ✅ **详细代码框架**（基于T2T-CO改编）
3. ✅ **数据生成脚本**（TSP实例 + LKH求解）
4. ✅ **可视化工具**（扩散动画GIF生成器）
5. ✅ **论文阅读笔记**（DIFUSCO + T2T-CO核心要点）

---

**这个项目的最大亮点**:
> "能把复杂的数学问题（TSP）用生成模型'画'出来，既有理论深度，又有视觉冲击力。非常适合做课程展示和后续发表！"


