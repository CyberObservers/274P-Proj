# CS 274P 期末 Project 提案集

> 课程：Neural Networks (Pierre F. Baldi)
> 团队：5 人（3 老手 + 1 中等 + 1 新手）
> 算力：2–4 张 A6000
> 剩余时间：**2–3 周**
> 工具：Claude Code / Codex 等 vibe coding

## 总体推荐排序

| # | 课题 | Baldi 契合度 | 风险 | 出彩潜力 | 适合新手参与度 | 状态 |
|---|---|---|---|---|---|---|
| **主推** | **PIFT (HEP 表格 DL)** | ⭐⭐⭐⭐⭐ | 中 | **高** | 良 | 主推方案 |
| 备选 1 | 蛋白 PEFT (ESM-2) | ⭐⭐⭐⭐⭐ | 中 | 中-高 | 中 | 备选 |
| 备选 2 | Dropout 现代变体 | ⭐⭐⭐⭐⭐ | 低 | 中 | 高 | 兜底 |
| 备选 3 | SAE on small LLM | ⭐⭐⭐⭐ | 中 | 高 | 中 | ambitious |
| 备选 4 | Speculative Decoding | ⭐⭐ | 中 | 中 | 低 | 工程偏重 |
| 备选 5 | **Grokking 动力学研究** | ⭐⭐⭐⭐ | 低 | 中-高 | 高 | **纯 CS** |
| 备选 6 | **Lottery Ticket on Transformer** | ⭐⭐⭐ | 低 | 中 | 高 | **纯 CS** |
| 备选 7 | **优化器隐式偏置 study** | ⭐⭐⭐ | 低 | 中 | 高 | **纯 CS** |

---

# 主推方案：PIFT

## Physics-Informed Feature Tokenization for Tabular Deep Learning: Bridging Domain Knowledge and Neural Architecture on High-Energy Physics Benchmarks

---

## 1. 项目概述

### 1.1 核心问题

在表格数据（tabular data）上，深度学习模型至今无法全面超越 XGBoost 等树模型。但高能物理（HEP）数据具有独特的物理对称性和守恒律，这类**领域先验知识**在现有表格深度学习架构中几乎被完全忽略。

### 1.2 核心想法

我们提出 **Physics-Informed Feature Tokenizer (PIFT)**——一个可插拔的特征嵌入模块，将物理领域知识（洛伦兹不变量、守恒量、粒子对称性）编码为网络的 inductive bias，嵌入到 FT-Transformer 等架构中。

**核心假设**：在具有物理结构的表格数据上，融入领域知识的深度学习模型可以超越通用表格模型和传统树模型，**尤其是在只给低层运动学特征（low-level features）的设置下**。

### 1.3 Headline Target（量化目标）

> 在 **HIGGS low-level 21 features** 设置下，PIFT 将 FT-Transformer 的 AUC 从 baseline 的 ~0.85 推到 **≥ 0.87**，逼近 Baldi 2014 用 high-level features (28 features) 时的 0.88。
> 等价表述：**PIFT 让网络从原始低层特征自动"发现"物理学家手工构造的高层特征**。

### 1.4 为什么 Baldi 教授会感兴趣

- HIGGS / SUSY / HEPMASS 三个数据集**都是他本人团队发布的**经典 benchmark（Baldi 2014, 2016）
- 物理知识 + 深度学习的融合是他实验室的核心研究方向（SPANet、Parameterized Networks）
- 我们的对称性感知模块直接对话他 lab 的 SPANet（Shmakov et al., 2024）
- 我们的实验设置（low-level vs high-level）直接对应他 Nature Communications 2014 论文的关键 ablation
- 提出新模块而非仅仅跑实验，体现了架构创新的能力

---

## 2. 实验设计

### 2.1 数据集选择（3 个主要 + 2 个对照）

| 数据集 | 来源 | 采样量 | 特征数 | 特点 | 用途 |
|---|---|---|---|---|---|
| HIGGS | UCI / Baldi et al., 2014 | **1M** (从 11M) | 28 (21 low + 7 high) | 高能物理信号/背景分类 | 主实验 |
| SUSY | UCI / Baldi et al., 2014 | **1M** (从 5M) | 18 (8 low + 10 high) | 超对称粒子检测 | 主实验 |
| HEPMASS | UCI / Baldi et al., 2016 | **500K** (从 10.5M) | 28 | 参数化粒子质量分类 | 主实验 |
| Forest Cover | UCI | 581K | 54 | 非物理通用表格数据 | **对照组**（确保不退化） |
| Adult Income | UCI | 48K | 14 | 经典小型表格基准 | **对照组**（确保不退化） |

> **采样策略调整**：原 v1 写 50 万，但 Baldi 2014 的 baseline 是在更大数据上跑的。把 HIGGS / SUSY 提到 100 万，HEPMASS 维持 50 万，可以让我们的 baseline AUC 与他论文 ±0.005 内对齐，避免他第一眼觉得 baseline 跑歪。

**为什么选这些数据集？**
- HIGGS / SUSY / HEPMASS 都由 Baldi 团队发布，具有**已知的物理对称性**
- Forest Cover / Adult 作为**不含物理结构**的对照组，验证 PIFT 不会在通用表格数据上产生负面影响

### 2.2 对比模型（5 个，砍掉 TabNet）

| 模型 | 类别 | 选择理由 |
|---|---|---|
| XGBoost | 树模型 | 表格数据的标准 SOTA baseline |
| MLP (调优) | DL baseline | Gorishniy 2021 确认其为被低估的 baseline |
| ResNet-like | DL baseline | Gorishniy 2021 提出的强 baseline |
| FT-Transformer | Transformer 表格 SOTA | 当前表格 DL 的 SOTA 代表 |
| **FT-Transformer + PIFT（我们的）** | 改进模型 | 融入物理先验的 FT-Transformer |

> **砍 TabNet 的理由**：(1) 复现坑多；(2) Gorishniy 2021 已经表明它弱于 FT-T；(3) 评审不会关心。把节省的时间投到 PIFT 消融上回报更高。

### 2.3 关键实验设置：low-level vs high-level（v2 新增）

直接对应 Baldi 2014 表 1，设置三档输入：

| Setup | HIGGS 输入 | SUSY 输入 |
|---|---|---|
| **Low-level only** | 前 21 列原始运动学量 | 前 8 列原始运动学量 |
| **High-level only** | 后 7 列物理工程特征 | 后 10 列物理工程特征 |
| **All features** | 全部 28 列 | 全部 18 列 |

**Low-level only 是我们的主战场** — 此设置下 PIFT 的"显式构造不变量"模块注入的是真正的新信息，预期 gap 最大。

### 2.4 Data Efficiency 实验（v2 新增）

为每个模型在 [10K, 50K, 100K, 500K, 1M] 训练集大小上画 learning curve。即使最终 AUC 持平，"**PIFT 在小数据下显著领先**"也是 Baldi 喜欢的 narrative（物理先验等价于额外训练数据）。

### 2.5 评估指标

- **AUC-ROC**（主要指标，与 Baldi 2014 一致）
- Accuracy
- 收敛速度（达到 95% 最终 AUC 所需 epoch）
- 训练 / 推理 wall-clock time
- 特征消融分析

---

## 3. 核心创新：PIFT 模块设计

### 3.1 动机

FT-Transformer 的 Feature Tokenizer 对每个特征做独立的线性嵌入：

```
e_j = W_j · x_j + b_j   (每个特征一个独立的嵌入层)
```

这完全忽略了特征之间的**物理关联**。例如在 HIGGS 数据中：
- 粒子的 (pT, eta, phi) 三个特征共同描述一个四动量
- 不变质量 m_jj, m_bb 等是物理守恒量
- 同类型粒子（jet_1 vs jet_2）应当 permutation equivariant

### 3.2 PIFT 架构

PIFT 在标准 Feature Tokenizer 之前增加三个子模块：

```
原始特征 x
  │
  ├─── (a) 物理分组嵌入 (Physics Group Embedding)
  │       将属于同一物理对象的特征分组处理
  │       例如：(lepton_pT, lepton_eta, lepton_phi) → 一个 group token
  │
  ├─── (b) 不变量增强 (Invariant Augmentation)  ← 仅在 low-level setup 下注入新信息
  │       显式计算已知的洛伦兹不变量作为额外特征
  │       例如：从四动量计算不变质量 √(E² - px² - py² - pz²)
  │
  └─── (c) 对称性感知位置编码 (Symmetry-Aware PE)
          同类型粒子（jet_1 vs jet_2）共享 type embedding
          → permutation equivariant，呼应 SPANet (Shmakov 2024) 的对称性思想
  │
  ▼
增强后的 token 序列 → 标准 Transformer Encoder → 分类头
```

### 3.3 设计细节

#### (a) Physics Group Embedding

```python
class PhysicsGroupEmbedding(nn.Module):
    def __init__(self, group_config):
        # group_config: {"lepton": [0,1,2], "jet1": [3,4,5,6], ...}
        self.group_mlps = nn.ModuleDict({
            name: nn.Linear(len(indices), d_model)
            for name, indices in group_config.items()
        })

    def forward(self, x):
        tokens = []
        for name, mlp in self.group_mlps.items():
            group_features = x[:, self.group_config[name]]
            tokens.append(mlp(group_features))
        return torch.stack(tokens, dim=1)  # (B, num_groups, d_model)
```

#### (b) Invariant Augmentation

```python
def compute_invariants(x):
    # 不变质量、ΔR、横动量之和等
    # 注意：在 high-level setup 下这些信息已存在；在 low-level setup 下是真正的新信息注入
    invariants = []
    invariants.append(compute_invariant_mass(x))
    invariants.append(compute_delta_R(x))
    return torch.cat([x, torch.stack(invariants, -1)], dim=-1)
```

> **设计警惕**：HIGGS 后 7 列就是物理学家手工构造的不变量。如果只在 all-features setup 下做 (b)，可能被质疑信息冗余。**主战场放在 low-level only**，让 (b) 真正注入新信息。

#### (c) Symmetry-Aware Positional Encoding

```python
# 同类型粒子（如 4 个 jets）共享可学习的 type embedding
type_embedding = nn.Embedding(num_particle_types, d_model)
# jet_1, jet_2, jet_3, jet_4 → 同一个 type_id
# → 整个网络对 jet 编号的随机置换严格不变 (permutation equivariant)
# 这与 Shmakov et al. SPANet 的 attention 对称性思想一致
```

### 3.4 关键假设与验证

| 假设 | 验证方式 |
|---|---|
| 物理分组优于逐特征嵌入 | 消融：PIFT vs 标准 FT |
| 不变量增强在 low-level 设置下尤其有效 | 三档输入设置的对比 |
| 对称性编码减少过拟合 | jet 排列随机化实验 |
| PIFT 让小数据训练更高效 | Learning curve [10K → 1M] |
| PIFT 在非物理数据上不退化 | Forest Cover / Adult 上的对比 |

---

## 4. 分工建议（5 人，匹配团队结构）

| 角色 | 负责内容 | 适合谁 | 工作量 |
|---|---|---|---|
| 成员 A — 数据 & Baseline | 数据预处理管线、XGBoost / MLP / ResNet 实验（用 rtdl repo 直接跑） | **新手** | 中 |
| 成员 B — FT-Transformer | 复现 FT-T、调参、low/high/all 三档设置 | 中等 | 中 |
| 成员 C — PIFT 模块开发 | 实现 PIFT 三个子模块、集成到 FT-T | **老手 1** | 高 |
| 成员 D — 实验 & 可视化 | 消融、learning curve、attention 可视化、统计检验 | 中等 / 老手 2 | 中-高 |
| 成员 E — 论文 & 理论 | 物理背景、相关工作、报告撰写、对接 SPANet 等文献 | 老手 3 | 中 |

> **新手 → A**：rtdl repo 拿来即用，门槛低，成果可见
> **核心创新 → 老手 1**：PIFT 是命脉，必须由最强的人主导
> **报告 / 故事 → 老手 3**：Baldi 看的是 narrative，最后一公里要稳

---

## 5. 时间线（压缩到 3 周）

| 时间 | 里程碑 | 交付物 |
|---|---|---|
| Day 1–2 | 数据下载 + 环境搭建 | 5 个数据集本地、PyTorch / rtdl 环境跑通 |
| Day 3–5 | Baseline 全部跑通 | XGBoost / MLP / ResNet / FT-T 在 HIGGS 上的初步 AUC，与 Baldi 2014 对齐 |
| Day 6–10 | PIFT 实现 + HIGGS 主实验 | PIFT 三模块代码、HIGGS low-level 上达成 headline target |
| Day 11–14 | 完整实验 + 消融 + Learning curve | 全数据集结果表、消融表、learning curve 图 |
| Day 15–18 | 对照组 + 收尾 | Forest Cover / Adult 上的"无退化"验证、attention 可视化 |
| Day 19–21 | 报告 + 答辩准备 | 最终报告 + 幻灯片 |

> **关键 checkpoint**：第 5 天结束 baseline 必须复现完成（AUC 与 Baldi 2014 误差 < 0.005），否则立即报警。

---

## 6. 计算资源估计

- HIGGS 1M / SUSY 1M / HEPMASS 500K，每个模型单跑：A6000 上 30–90 分钟（AMP + d_model=128）
- 5 模型 × 5 数据集 × 3 setup × 3 seeds ≈ 200 runs × 45 min ≈ **150 GPU·hour**
- Learning curve 额外 ~50 GPU·h
- **总计 ~200 GPU·h，4 张 A6000 约 2.5 天满负荷**，预算充裕

**单 GPU 优化策略**：
- 混合精度训练（AMP）加速 2–3×
- FT-T 用 3–4 层、d_model=128（而非论文 192）
- 大数据集预先 tokenize 落盘，避免重复处理

---

## 7. 预期结论与故事线（三档 fallback）

**🟢 最佳情况**：PIFT 在 HEP 数据集（尤其 low-level setup）显著超越 FT-T 和 XGBoost，对照组持平。
→ 结论：**领域知识融入网络架构是表格 DL 超越树模型的关键路径**。
→ 标题："PIFT: Physics Priors Close the Gap Between Deep Learning and XGBoost on Tabular HEP Data"

**🟡 次佳情况**：PIFT 在小训练集上优势明显，大数据量时差距缩小。
→ 结论：**物理先验等价于额外训练数据，data efficiency 故事**。
→ 标题："Physics Priors as Free Training Data: A Data Efficiency Study"

**🔴 保底情况**：整体提升有限。
→ 通过消融分析哪些子模块有效；讨论为什么通用 Transformer 自注意力已隐式学到部分物理结构。
→ 这是有价值的 negative result，仍能产出完整报告。

---

## 8. 风险评估

| 风险 | 概率 | 应对 |
|---|---|---|
| FT-T 在 all-features 上已饱和，PIFT gain < 1% | 中 | **主战场切到 low-level only**，gap 更大 |
| Baseline 复现 AUC 偏低 | 中 | Day 5 强制 checkpoint；不行就加大采样到 2M |
| PIFT (b) 信息冗余被质疑 | 中 | low-level setup 化解 |
| 新手卡住数据 pipeline | 中 | 用 rtdl 现成 dataloader，老手 1 第 1 周做 1 次 pair coding |
| 物理分组 config 错误 | 低 | 直接照抄 Baldi 2014 论文附录的 feature 描述 |

---

## 9. Limitation（写报告时主动列出）

- 仅在 HEP 数据上验证，未触及其他科学领域（chem / bio）
- PIFT 的物理 group config 需人工指定，未做自动发现
- 没有 scaling 到 Transformer-XL 量级 / 预训练 setup
- 对照组只有两个，不能完全排除 PIFT 在某些非物理表格上的负面影响
- Invariant augmentation 列表是手工挑选的，未穷尽所有洛伦兹不变量

---

## 10. 技术栈与资源

**框架**：PyTorch + XGBoost + 混合精度 (AMP)

**参考代码库**：
- FT-Transformer: `github.com/yandex-research/rtdl-revisiting-models`
- HIGGS: `archive.ics.uci.edu/dataset/280/higgs`
- SUSY: `archive.ics.uci.edu/dataset/279/susy`
- HEPMASS: `archive.ics.uci.edu/dataset/347/hepmass`

---

## 11. 核心参考文献

1. Baldi, P., Sadowski, P., & Whiteson, D. (2014). *Searching for Exotic Particles in High-Energy Physics with Deep Learning*. Nature Communications 5:4308.
2. Baldi, P. et al. (2016). *Parameterized Neural Networks for High-Energy Physics*. European Physical Journal C.
3. Gorishniy, Y. et al. (2021). *Revisiting Deep Learning Models for Tabular Data*. NeurIPS 2021.
4. Grinsztajn, L. et al. (2022). *Why Do Tree-Based Models Still Outperform Deep Learning on Tabular Data?* NeurIPS 2022.
5. Shwartz-Ziv, R. & Armon, A. (2022). *Tabular Data: Deep Learning is Not All You Need*. Information Fusion 81.
6. Shmakov, A. et al. (2024). *Reconstruction of Unstable Heavy Particles Using Deep Symmetry-Preserving Attention Networks*. Nature Communications Physics.
7. Arik, S. & Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning*. AAAI 2021.

---

## v2 相对 v1 的修改总结

1. ⏱️ 时间线从 5–6 周压缩到 **3 周（21 天）**
2. ✂️ Baseline 砍掉 TabNet（保留 5 个模型）
3. 🎯 **主战场切到 HIGGS low-level 21 features**，让 PIFT 注入真正新信息而非冗余
4. 🔢 加入 **headline target**：HIGGS low-level AUC ≥ 0.87
5. 📈 加入 **Data Efficiency 实验**（learning curve [10K → 1M]）作为备线故事
6. 📊 数据采样从 50 万提到 HIGGS/SUSY 1M、HEPMASS 50 万，对齐 Baldi 2014 baseline
7. 🔗 (c) Symmetry-Aware PE 显式 link 到 Baldi lab 的 SPANet
8. 👥 分工明确到角色对应人员等级（新手 → A, 老手 → C/E）

---

# 备选方案 1：蛋白质语言模型的高效 PEFT

### 任务与解决的问题
ESM-2 (650M / 3B) 在下游任务（二级结构、接触图、稳定性预测、亚细胞定位）上的全参 finetune 成本高。我们要找一种**结构感知的 PEFT 方法**，在远低于 full finetune 的参数量下达到甚至超过 LoRA。

### 现有代码来源
- HuggingFace `facebook/esm2_t33_650M_UR50D`
- PEER benchmark (`github.com/DeepGraphLearning/PEER_Benchmark`)
- `peft` 库（LoRA / IA3 / DoRA / Prefix）
- TAPE benchmark (`github.com/songlab-cal/tape`)

### 现有 baseline
- Full finetune
- LoRA (r=8, 16)
- IA3
- Linear probe（freeze backbone）
- DoRA (2024)

### 我们的优化方向与原理
**Structure-prior LoRA**：LoRA 的 BA 分解中，A 矩阵不随机初始化，而是用 ESM-2 自带的 contact prediction head 输出的接触概率构造一个低秩"结构投影"。原理：蛋白任务的有效自由度沿着接触结构方向，把 LoRA 的更新方向先验地对齐到这个流形上，等于注入了归纳偏置。

或备选：**残基类型条件 LoRA**（mixture of LoRA over 20 amino acid types），原理类似 MoE，但选择函数显式由序列决定。

### 预期效果
在 PEER 的 4 个任务上，平均比同参数量 LoRA 提升 1–2%，尤其在数据量小的任务（<5k 样本）上更明显。

### 实验设置
- 4 个任务：Secondary Structure、Contact、Stability、Subcellular Localization
- 每个任务：full FT / LoRA / DoRA / Ours，3 seeds
- 报指标：accuracy / Spearman / F1（按任务）+ trainable params %

### 计算资源估计
- ESM-2-650M + LoRA 单任务 finetune：A6000 上 1–3 小时
- 4 任务 × 4 方法 × 3 seeds = 48 runs × 2h ≈ **100 GPU·hour**
- 4 张 A6000 → 1.5 天跑完

### 讲故事的切入点

**A. Baldi 自己的领域**：Baldi 是蛋白质二级结构预测的奠基人之一（SSpro），把现代 PLM 的高效适配做到他熟悉的任务上 — 从 SSpro / ProteinNet 历史接到 ESM-2，"小数据 + 强先验"作为永恒主线。

**B. Inductive bias for science**：通用 PEFT 假设任务是黑箱，但科学任务有明确的物理 / 结构先验，把先验注入 PEFT 的低秩子空间是未被开发的方向。

### 工作量估计
- 老手 1：PEER pipeline + LoRA baseline（4 天）
- 老手 2：structure-prior LoRA 实现（5 天）
- 老手 3：实验跑 + 分析（贯穿）
- 中等：消融 + 可视化（学到的 LoRA 方向 vs contact map 对齐度）
- 新手：4 个任务的 data loader 整理、表格汇总、跑 baseline

### 风险评估
- ✅ 任务定义清晰
- ⚠️ structure prior 可能在大数据任务上无效甚至有害（要诚实报告）
- ⚠️ ESM-2 3B 在 A6000 上要 LoRA 才能装下，650M 是稳妥选择

### Limitation
- 只测 4 个 PEER 任务，未覆盖结构预测整体 pipeline
- Contact prior 来自 ESM-2 自身，对其他 PLM 是否迁移未验证
- 没有跑 3B / 15B 量级

---

# 备选方案 2：Dropout 的现代变体研究（向 Baldi 致敬款，最稳）

### 任务与解决的问题
Baldi & Sadowski 2013 给出了 dropout 的几何 / 集成解释。今天 Transformer 时代主流是 attention dropout + DropPath。问题：**这些 dropout 变体的最优 schedule、最优层位置、与 LayerNorm / residual 的相互作用，并没有 Baldi 当年那么干净的研究**。我们做一次系统化实证 study + 提出一个简单改动。

### 现有代码来源
- timm（ViT、ConvNeXt 全家桶）
- nanoGPT
- 自己实现 dropout 容易

### 现有 baseline
- 标准 attention dropout
- DropPath / Stochastic Depth
- R-Drop（ICML 2021）
- LayerDrop
- DropKey (CVPR 2023)

### 我们的优化方向
**Curriculum DropPath**：训练前期 drop_rate 小，后期单调增大；并对深层位置 drop 概率更大。原理：早期网络需要充分信息流形成基础特征，后期需要更强正则；与 Baldi "dropout 作为自适应正则" 的解释相容。

或：**Layerwise R-Drop on ViT** — 对每一层 token 表示对两次 dropout 后的输出做一致性正则。

### 预期效果
- ImageNet-100 / CIFAR-100 上 ViT-Tiny / Small +0.5–1.5% top-1
- 给出"drop rate × depth × epoch"的 3D 热力图

### 实验设置
- 数据：CIFAR-100（必跑）+ ImageNet-100（如果时间够）
- 模型：ViT-Tiny / Small，DeiT 训练 recipe
- 5 种 dropout 变体 × 3 seeds
- 同时报：训练曲线、验证曲线、loss landscape sharpness（SAM-style）

### 计算资源估计
- ViT-S CIFAR-100 200 epoch：A6000 上 ~3 小时
- 5 × 3 = 15 runs × 3h = 45 GPU·h（CIFAR）+ ImageNet-100 关键 3 配置 36 GPU·h
- **总计 ~80–120 GPU·h，安全**

### 讲故事的切入点
**A. 直接对话 Baldi（杀手锏）**：Baldi & Sadowski 2013 把 dropout 解释为对几何平均的近似集成。十年后 Transformer 时代，我们重做这个研究 — 报告第一页就引 Baldi 2013，全程把他论文当 reference frame。

**B. 实用 + 干净**：一行代码改动，所有 ViT 训练 +1%。drop-in PyTorch module + GitHub 链接。

### 工作量估计
- 老手 1：训练 pipeline + baseline 复现（3 天）
- 老手 2：dropout 变体实现（3 天）
- 老手 3：sharpness 分析 + 理论 connection（贯穿）
- 中等：所有 ablation 实验
- 新手：CIFAR-100 数据 + 训练日志整理 + 画图（最适合新手，代码工整）

### 风险评估
- ✅ 风险最低，跑不出来概率几乎为零
- ⚠️ 提升幅度可能小（<0.5%），靠 narrative 救
- ✅ 与 Baldi 课题契合度最高

### Limitation
- 只在视觉 + 小规模验证
- 提升 marginal，可能在不同 recipe 下消失
- 没有严格理论证明

---

# 备选方案 3：在小型 LLM 上改进 Sparse Autoencoder（ambitious 选项）

### 任务与解决的问题
对开源小模型（GPT-2 small / Pythia-160M / Gemma-2-2B）的中间层激活训练 SAE，**把多义的 neuron 激活分解成可解释的稀疏特征**。当前 SAE 存在 **dead features、feature absorption、重构-稀疏性 Pareto 不优**。

### 现有代码来源
- EleutherAI `sae`、Anthropic 公开 recipe、`SAELens`、TransformerLens

### 现有 baseline
- Vanilla ReLU SAE / Top-K SAE / JumpReLU SAE / Gated SAE — 都有公开 checkpoint 和 metric

### 我们的优化方向
1. **Matryoshka SAE 改进版**：嵌套不同 K 的 Top-K，强迫前 k 个 latent 学粗粒度，后面学细粒度 — 解决 feature absorption
2. **激活函数 / 初始化改进**：JumpReLU threshold per-feature learnable + 稀疏正则的 schedule — 救 dead features

### 预期效果
- 同 L0 下 loss recovered +1–3%
- Dead feature 比例从 ~10–30% 降到 <5%
- 给出 1–2 个 case study（Neuronpedia 风格）

### 实验设置
- Pythia-160M / GPT-2 small，layer 6/8 residual stream
- 训练 token：~100M–500M（OpenWebText 子集）
- 评测：L0、reconstruction MSE、cross-entropy loss recovered、dead %、auto-interp（GPT-4o）

### 计算资源估计
- 单 SAE 训练 A6000 上 4–8h
- 4 张 A6000 一晚跑 8–16 个 config
- **总卡时 ~200 GPU·h**（auto-interp API 费 ~$50）

### 讲故事的切入点
**A. 与 Baldi 的 AE 理论对话**：Baldi & Hornik 1989 → 现代 overcomplete SAE 改变了优化 landscape，我们利用其几何性质。

**B. mech interp 工程瓶颈**：Anthropic ~30% feature is dead，我们系统研究 dead feature 起源。

**C. Pareto 前沿叙事（最稳）**：所有 baseline 一条线，我们一条线，肉眼可见碾压。

### 工作量估计
- 老手 1：SAE pipeline 复现 + 改造（5 天）
- 老手 2：评测套件 + auto-interp（4 天）
- 老手 3：ablation matrix（贯穿）
- 中等：可视化 dashboard + case study
- 新手：数据准备、激活抓取脚本、图表整理

### 风险评估
- ✅ baseline 复现风险低，SAELens 文档全
- ⚠️ 改动可能 marginal，准备 2 个改动同时尝试
- ⚠️ Auto-interp API 费用 ~$50

### Limitation
- 只在小模型验证，未必 scale 到 70B
- Auto-interp 用 LLM 打分有偏差
- 未触及 cross-layer SAE / transcoder
- "单语义"判定本身缺乏 ground truth

---

# 备选方案 4：Speculative Decoding 的 draft 改进（工程偏重，最弱契合度）

### 任务与解决的问题
LLM 推理慢。Speculative decoding 用小 draft model 提议 token、大 target 验证，可加速 2–3×。瓶颈：**draft 接受率不够高（30–60%）**。

### 现有代码来源
- vLLM 原生支持、EAGLE / EAGLE-2、Medusa

### 现有 baseline
- 普通 autoregressive / Vanilla SD / Medusa / EAGLE-2 (SOTA)

### 我们的优化方向
**Context-adaptive draft length**：根据当前 token 的熵动态调整 draft 长度（高熵 → 短 draft）。原理：高熵区域 token 不可预测，长 draft 浪费算力。

### 预期效果
MT-Bench / HumanEval 上比 EAGLE-2 再加 5–15% wall-clock 加速。

### 计算资源估计
- 7B target + 小 draft，A6000 单卡能跑
- 不需要训练，只评测吞吐
- **<50 GPU·h**

### 讲故事
Efficiency story 不太打动 Baldi（他偏理论 / 科学），如果组里有人想做工程方向才考虑。**列为最低优先级备选。**

### 风险
- ⚠️ EAGLE-2 已非常优化，再榨 5% 不容易
- ⚠️ 与 Baldi 研究品味不对齐

---

# 备选方案 5：Grokking 动力学研究（纯 CS · 强推）

### 任务与解决的问题
**Grokking** = 网络在训练 loss 降到 0 后很久才 generalize（test acc 突然跳升）。Power et al. 2022 发现于模算术任务，引发大量后续工作（Nanda 2023 mech-interp、Liu 2023 representation learning 视角、Thilak 2024 LU mechanism）。**问题**：grokking 的 phase transition 触发条件、临界 weight decay、与 representation 形成时点的关系，仍未被系统刻画。

### 现有代码来源
- Nanda 的 `grokking` repo（`github.com/neelnanda-io/Grokking`），不到 500 行
- Power 原始论文代码

### 现有 baseline
- Power 2022 原始 grokking 曲线
- Nanda 2023 mechanistic 解释（Fourier features in modular addition）
- 各种 weight decay / data fraction sweep

### 我们的优化方向
两选一：
1. **触发器 study**：系统扫描 (data fraction × weight decay × init scale × LR)，找到 grokking 是否发生的"相图"；用 weight norm trajectory + Fourier basis alignment 作为顺序指标，预测 grokking 何时发生
2. **加速 grokking**：测试若干启发式（动态 weight decay schedule、representation regularizer、auxiliary loss）能否把 grokking step 减少 5–10×

### 预期效果
- 一张干净的 (wd, data_frac) 相图，标注 grokking / no-grokking / fast-grokking 区域
- 给出 1 个早期可观测信号（如 weight norm peak 时间点）能预测 grokking step 的相关性 r > 0.7

### 实验设置
- 模型：2-layer 128-dim Transformer（与 Nanda 一致）
- 任务：模 113 加法 / 乘法 / 复合任务
- 每个 config 训练到 50K step
- 100+ configs × 3 seeds，单卡能并行很多

### 计算资源估计
- 单 run 在 A6000 上 5–15 分钟（小模型）
- 300 runs × 10 min ≈ **50 GPU·h，预算极宽松**

### 讲故事
**A. 与 Baldi 的过参数化 / shallow fallacy 论文呼应**：他 2024 年那篇关注 over-parameterization；grokking 正是过参数化下的一种独特 generalization 现象。
**B. 干净的科学问题**：phase transition 在 ML 里少见，相图是物理学家熟悉的语言（Baldi 是物理背景）。
**C. Pareto-style narrative**：相图 + 早期信号 = 完整故事，几乎不可能写不出来。

### 工作量估计
- 老手 1：训练框架 + 相图 sweep（4 天）
- 老手 2：mech-interp 探针、Fourier 分析（5 天）
- 老手 3：理论分析与文献对接（贯穿）
- 中等：扩展任务（乘法 / 多步算术）
- 新手：sweep 调参、画图、整理超参表（**最适合新手**——代码极小、循环极快）

### 风险评估
- ✅ 模型小、迭代快，几乎不可能跑不出结果
- ⚠️ 相图本身是已知现象，需要找到独特角度（早期信号 / 加速）
- ✅ 与 Baldi 课题契合度比想象高（理论 + 过参数化 + 几何）

### Limitation
- 只在模算术 toy 任务，未在大规模任务验证
- 早期信号可能任务特异
- 没有严格理论证明，纯实证

---

# 备选方案 6：Lottery Ticket Hypothesis on Small Transformers（纯 CS · 稳）

### 任务与解决的问题
Frankle & Carbin 2019 的 **Lottery Ticket Hypothesis (LTH)**：稠密网络包含一个稀疏子网络，独立训练即可达到原始性能。在 CNN 上验证充分，**在现代 Transformer（GPT-2 / ViT）上仍存在不少争议**（Chen 2020 部分验证、Liu 2023 反例）。

### 现有代码来源
- OpenLTH (`github.com/facebookresearch/open_lth`)
- nanoGPT 用于 GPT-2 实验
- timm 用于 ViT

### 现有 baseline
- Magnitude pruning + rewind to init / iterative magnitude pruning (IMP)
- Random pruning (lower bound)
- One-shot vs iterative pruning

### 我们的优化方向
**Structured ticket discovery**：在 Transformer 中按 attention head / FFN neuron 结构化剪枝，而非 unstructured weight masking。原理：unstructured 在 Transformer 上效果不稳定，结构化更接近实际可加速的稀疏性。
+ 可加：**早期 ticket prediction**（Frankle 2020 GraSP / SynFlow） — 只用前 5% 训练就预测 ticket。

### 预期效果
- 在 GPT-2 small (WikiText-103) 或 ViT-Tiny (CIFAR-100) 上，找到 50–70% 稀疏的 ticket，性能损失 <1%
- 给出 head-level / FFN-level 重要性热力图

### 计算资源估计
- IMP 需要多轮训练（~10 轮 × 单轮 4h）≈ 40 GPU·h per setting
- 4 settings × 3 seeds ≈ **150 GPU·h**

### 讲故事
**A. 与 Baldi over-parameterization 论文直接对话**：他 2024 论文核心问题就是"为什么 over-param 还能 generalize"，LTH 是答案的一种 — 实际有效的子网络远小于看起来的。
**B. 实用价值**：稀疏 Transformer = 推理加速。
**C. 实证 study**：Transformer LTH 文献矛盾，我们做一次 controlled comparison。

### 工作量估计
- 老手 1：IMP pipeline + GPT-2 / ViT 实验（5 天）
- 老手 2：结构化剪枝实现 + 早期 ticket（4 天）
- 老手 3：分析 + 理论对接
- 中等：对比 random / SynFlow / GraSP baseline
- 新手：稀疏度 sweep、画图、压缩率表

### 风险评估
- ✅ baseline 复现简单
- ⚠️ IMP 多轮训练时间累积，要早开始
- ⚠️ 现代 Transformer 上 LTH 可能本身效果有限（这本身就是结论）

### Limitation
- 只在小模型小数据集，未触及 7B 级别
- 没有真实 inference 加速测量
- 结构化稀疏的硬件友好性未实测

---

# 备选方案 7：优化器的隐式偏置实证 Study（纯 CS · 简单清爽）

### 任务与解决的问题
SGD / Adam / AdamW / Lion / Sophia 的 **implicit bias** — 即使最终训练 loss 相同，不同优化器找到的解在 sharpness、范数、generalization gap 上系统不同。已有理论（Smith 2021、Kaddour 2023）但**在 Transformer 上的系统实证比较仍稀缺**。

### 现有代码来源
- Lion (`github.com/google/automl/tree/master/lion`)
- Sophia (`github.com/Liuhong99/Sophia`)
- nanoGPT、timm

### 现有 baseline
- SGD-momentum / Adam / AdamW / Lion / Sophia
- 都有公开实现

### 我们的优化方向
不发明新优化器，做一个**控制良好的对比研究**：
- 固定 final train loss（用 LR schedule 拉到同一点）
- 测量：sharpness（top eigenvalue of Hessian via power iteration / SAM perturbation）、weight norm、effective rank、generalization gap
- 在 ViT-Tiny CIFAR-100 + GPT-2-small TinyStories 上各跑一次

可选加：**优化器 interpolation** — 训练后用 (Adam_solution → SGD_solution) 沿直线插值，画 loss landscape 一维切片。

### 预期效果
- 一张表：5 优化器 × 4 metric，揭示哪种优化器倾向于 flat minima、哪种小 weight norm 等
- 一组 mode connectivity 曲线

### 计算资源估计
- ViT-Tiny + 5 optim × 3 seeds = 15 runs × 3h = 45 GPU·h
- GPT-2-small TinyStories + 5 optim × 3 seeds = 15 runs × 4h = 60 GPU·h
- Sharpness 测量额外 ~20 GPU·h
- **总计 ~120 GPU·h**

### 讲故事
**A. Baldi 偏 theory taste 的菜**：implicit bias 是当代深度学习理论核心问题之一。
**B. 跨 vision 和 language 的横向对比**：很多研究只看 vision，我们多一个领域。
**C. 实用 takeaway**：给一句"Lion 找到的解 sharpness 比 AdamW 大 X%"这样的可引用结论。

### 工作量估计
- 老手 1：训练 pipeline（3 天）
- 老手 2：sharpness / Hessian 测量（4 天）
- 老手 3：mode connectivity（贯穿）
- 中等：跑 sweep、表格整理
- 新手：每个优化器的超参 sweep、画图（**适合新手**）

### 风险评估
- ✅ 风险极低，每个组件都是成熟工具
- ⚠️ 结论可能"如预期"，narrative 要立得住
- ✅ 即使无 surprise，systematic comparison 本身就有引用价值

### Limitation
- 只在小模型小数据
- Hessian 估计是 top eigenvalue，不是完整谱
- 没有 scaling law 视角

