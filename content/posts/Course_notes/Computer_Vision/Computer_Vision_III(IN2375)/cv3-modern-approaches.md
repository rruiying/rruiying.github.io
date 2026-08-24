---
title: "Computer Vision III (IN2375) — Notes 4: Modern Approaches"
date: 2026-08-23
tags: [Deep learning, Computer Vision, TUM, transformers, self-supervised learning, semi-supervised learning]
summary: Fourth notes for CV3 (IN2375, Detection, Segmentation and Tracking) at TUM — transformers from attention to ViT, Swin, DETR and Mask2Former; self-supervised learning from pretext tasks through contrastive (SimCLR, MoCo) and non-contrastive methods (DINO, MAE) with their downstream applications; and semi-supervised learning from its core assumptions to self-training, SAM, and consistency regularisation.
---
This chapter covers three modern threads that reshape everything from the
previous notes. First, **transformers**: the correlation layer we met in video
object segmentation already compares features against features, which is
exactly what **attention** generalises. From there we develop
**self-attention** and **positional encoding**, scale up to the **ViT** and
**Swin Transformer** backbones, and let transformers reshape the tasks
themselves — **DETR** casts detection as set prediction, **MaskFormer** and
**Mask2Former** unify segmentation as mask classification. Second,
**unsupervised (self-supervised) learning** in three waves: **pretext tasks**
(rotation, jigsaw puzzles, colorization, SSL on videos), **contrastive
learning** (SimCLR, MoCo), and **non-contrastive methods** (DINO and its
successors, masked autoencoders) — plus how SSL models are evaluated and what
they enable downstream, from semantic segmentation to segmentation from motion
cues and the contrastive random walk. Third, **semi-supervised learning**:
starting from the semi-supervised loss and its three assumptions (smoothness,
low density, manifold), we organise the field along two taxonomies — from
unsupervised preprocessing and **self-training** (OnAVOS, SAM and its
successors, pseudo-labels) to **intrinsically semi-supervised** methods
(entropy minimisation, virtual adversarial training), learning from synthetic
data with **domain alignment**, and **consistency regularisation**.

## 1. Transformers

### 1.1 Motivation: From correlation layers to attention

CNN encode的是**local context**：每一层卷积只看kernel覆盖的邻域，receptive field
随着网络深度单调增长。这带来几个问题：

1. 想要long-range的context aggregation就必须堆深度，而**深度越大越难训练**
2. receptive field是由每层的kernel size**设计死的**——这是一种inductive bias
   （认为图片信息集中在局部邻域），网络只能关注设计好的区域
3. 我们希望**在同一层内就有显式的non-local feature interactions**

那么：可不可以让网络**自己学习它的receptive field**？

**Recap：卷积其实就是矩阵乘法（im2col）**——把每个local patch拉平成列，
和拉平的filter做dot product：

![Convolution as matrix multiplication (im2col)](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/im2col.png)

而在video object segmentation里我们已经见过**correlation layer**：把两个feature map
拉平，两两做dot product得到相似度矩阵 $s_{ij} = f_i^\top f_j$——注意这是
**fixed operation，没有可学习的权重**：

![Correlation layer: dot products measure feature similarity](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/correlation.png)

把这两件事放在一起：卷积是"feature和**固定filter**做dot product"，correlation是
"feature和**feature**做dot product"。如果我们直接用correlation的方式让每个位置去
"查询"所有其他位置，就得到了最朴素的self-attention。

但把这种simple self-attention用到text上就暴露了问题。比如句子
*"this restaurant was not too terrible"*：

![Simple self-attention on text](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/text.png)

"not"直接影响"terrible"的含义，我们希望dot product $x_{not}^\top x_{terrible}$ 很大
——**但simple self-attention里没有任何可学习的参数，这种关系学不到**
（Problem: How do I learn this?）。解决的直觉：给每个token配上**可学习的线性映射**，
把同一个token投影成不同的"角色"再做dot product——这就引出了Q、K、V。

### 1.2 Attention and self-attention

**Attention的定义**。给定：

- query矩阵 $Q \in \mathbb{R}^{S \times n}$：当前元素（比如当前这个词）
- key矩阵 $K \in \mathbb{R}^{T \times n}$：被比较的元素
- value矩阵 $V \in \mathbb{R}^{T \times m}$：比较的权重对应取出的内容

$$\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{n}}\right)V$$

**Quiz: 为什么要除以scaling factor $\sqrt{n}$？**

假设 $q_i, k_i \sim \mathcal{N}(0,1)$，那么 $q^\top k = \sum_{i=1}^{n} q_i k_i$ 的
方差约为 $n$，标准差约 $\sqrt{n}$——n越大，进softmax的数值越大，softmax饱和后
梯度消失（vanishing gradient）。除以 $\sqrt{n}$ 把量级拉回来，所以叫
**scaled dot-product attention**。

**Quiz: 输出的维度是多少？**

$\mathbb{R}^{S \times m}$——"for every query we fetch the corresponding value"：
S个query，每个query取回一个m维的value加权和。

**Self-attention**就是Q、K、V全部来自同一个输入。输入 $X \in \mathbb{R}^{T \times d}$
（T个d维token），定义4个可学习的线性映射
$W^Q, W^K \in \mathbb{R}^{d \times n}$、$W^V \in \mathbb{R}^{d \times m}$、
$W^O \in \mathbb{R}^{m \times d}$：

$$Q := XW^Q, \quad K := XW^K, \quad V := XW^V$$

$$Y := \text{Attention}(Q, K, V)\, W^O \in \mathbb{R}^{T \times d}$$

输入 $[T \times d]$、输出 $[T \times d]$，可以像卷积层一样**堆叠成深层网络**。
完整的计算流程（对第三个元素算attention value）：

![Self-attention step by step](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/self-attention-steps.png)

**复杂度**：需要算 $T \times T$ 的pairwise相似度矩阵，所以memory是 $O(T^2)$、
runtime是 $O(T^2 n)$——**对token数量是二次的**，这个伏笔在Swin那里回收。

![Attention complexity and the idea of multiple heads](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/attention-complexity-multihead.png)

**Multi-head attention**：一个softmax就是一个attention head。把feature维度切给
Z个head、各自做attention再concatenate（过 $W^O$ 投影回d维）：

![Multi-head attention](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/multi-head-attention.png)

- 条件：Q、K、V的feature维度要能被head数整除
- Z个head让一个query最多取回Z种不同的value，**建模更复杂的token间关系**
- 复杂度不增加——实际wall time反而更快（并行）

**Normalisation**：两个标准改进（[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)）：
**residual connection**（把原始token feature加回去）+
**layer normalisation**（[Ba et al., 2016](https://arxiv.org/abs/1607.06450)，
每个token feature对自己的均值方差归一化）：

$$Y := \text{LayerNorm}(X + \text{MHA}(X))$$

但还剩一个根本问题：**整套计算里没有位置信息**。self-attention对token是
permutation-equivariant的——把输入顺序打乱，每个query取回的value完全不变。
"猫咬狗"和"狗咬猫"在它眼里一样，这显然不行。

### 1.3 Positional encoding

在[原始论文](https://arxiv.org/abs/1706.03762)里，解决办法是给每个token index
配一个**唯一的位置向量（PE矩阵的一行）**，和input embedding**逐元素相加**：

![Positional encoding](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/positional-encoding.png)

- PE引入了spatial affinity的概念（句子里两个词的距离、图片里两个patch的距离），
  **打破了permutation invariance**
- 可以是**学出来的**（positional embedding）或**固定的**（positional encoding），
  绝对或相对位置都行。原论文用固定的sinusoidal编码：

$$PE_{(pos, i)} = \begin{cases} \sin\!\left(pos / 10000^{i/d_{model}}\right) & i \text{ even} \\ \cos\!\left(pos / 10000^{(i-1)/d_{model}}\right) & i \text{ odd} \end{cases}$$

- 上图右侧热力图：**不同维度对应不同频率**（低维快、高维慢），编码绝对位置
- PE和token同维度（$1 \times d_{model}$），所以可以直接相加

**Self-attention recap**：

- 非常versatile：任意token数；head数、feature维度都是design choice
- 计算量对token数**二次增长**（pairwise相似度矩阵）
- 本身permutation-equivariant，绝对/相对位置靠**PE**补上

### 1.4 ViT

那么transformer能不能搬到视觉上？
[Dosovitskiy et al., 2020](https://arxiv.org/abs/2010.11929) 的
**ViT (Vision Transformer)**给出了肯定答案：在classification上和CNN有竞争力，
思路是**把图片切成patch序列，几乎不改NLP的Transformer**：

![ViT architecture](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/vit-architecture.png)

1. 把图片切成固定大小的patch
2. 每个patch过一个全连接层linearly embed成token
3. 额外接一个**可学习的[class] token**
4. 加上positional embedding
5. 喂给标准Transformer encoder
6. **最后只取[class] token的输出**，过一个MLP head做分类

**实验结论**：

- ViT只有在**超大数据集**（JFT，300M张图）上预训练才好——因为它
  **没有CNN的inductive bias**：locality（self-attention是全局的）、
  2D邻域结构（位置关系全靠数据学）、translation invariance
- 但意义重大：**language和vision从此用同一套计算框架**

### 1.5 Swin Transformer

**ViT的问题**（回顾1.2的伏笔）：

![Why ViT scales badly](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/swin-motivation.png)

- self-attention复杂度 $O((HW)^2 C)$——**对图片分辨率二次爆炸**
- 所有层的token数量不变；而CNN是逐层降分辨率的（省计算、扩receptive field），
  ViT享受不到这个好处

[Swin Transformer](https://arxiv.org/abs/2103.14030)（**S**hifted **Win**dow）
用三个设计解决，把复杂度降到**线性**：

**① Window attention（解决二次复杂度）**：只在固定大小 $M \times M$ 的
local window内做self-attention。每个window内是 $O(M^4 C)$，全图
$O(HW \cdot M^2 C)$——M是常数，所以总复杂度 $O(HWC)$，**对分辨率线性**。

**② Shifted windows（解决window割裂全局context）**：naive的固定window会让
context不再是全局的——相邻window之间永远不交流。解决：**相邻两层交替偏移
window的划分**，让上一层不同window里的feature在下一层进到同一个window：

![Shifted windows](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/swin-shifted-windows.png)

这样任意分辨率下feature都能逐层聚合全局context。

**③ Patch merging（引入层级/hierarchy）**：像CNN一样逐stage降分辨率——把
$2 \times 2$ 的C维patch拼接成4C维，再线性投影到2C维：分辨率减半、通道翻倍，
得到 $\frac{H}{4} \times \frac{W}{4} \times C \to \frac{H}{32} \times \frac{W}{32} \times 8C$
的**层级式backbone**，输出和ResNet等标准backbone兼容，可以直接接
detection/segmentation的下游头：

![Patch merging hierarchy](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/swin-patch-merging.png)

**结果**：

![Swin results on ImageNet](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/swin-results.png)

- 比ViT和不少CNN更快更准（还用着更低的输入分辨率），**不需要大数据集预训练**
- ImageNet-1K到22K的scalability更好

**Summary**：Swin把CNN的inductive bias（locality、hierarchy）和Transformer调和到了
一起；classification / detection / semantic segmentation全面SOTA；线性复杂度；
证明了Transformer是全能的视觉backbone。ICCV '21 best paper。

### 1.6 DETR

Transformer能不能直接做detection？关键观察：**object detection本质是set
prediction**——我们不关心bounding box的输出顺序。而Transformer恰好擅长处理
set。那就直接把detection建模成set prediction！
（[DETR, Carion et al., 2020](https://arxiv.org/abs/2005.12872)）

![DETR overview](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/detr-overview.png)

- CNN backbone学出2D feature（local feature embeddings）
- Transformer**并行**预测所有bounding box
- 训练时用**Hungarian matching**把prediction和ground truth一一对应
- **不需要NMS**——空类或低置信度的box直接丢掉即可

**A closer look**：

![DETR: a closer look](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/detr-closer-look.png)

- feature拉平、加positional encoding后进Transformer encoder（**self-attention**）
- 向decoder输入**object queries**（可学习的positional encoding），decoder对
  encoder输出做**cross-attention**
- 每个query的输出embedding过共享的FFN，预测一个detection
  （class logits + 归一化的box坐标）或"no object"类

![DETR transformer architecture](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/detr-architecture.png)

架构和[Vaswani et al., 2017](https://arxiv.org/abs/1706.03762)非常接近。

**Loss的问题**：预测是无序的set，必须先用**Hungarian matching**建立prediction和
GT的一一对应，才能算loss。box regression部分还有额外的坑——常用的L1距离
（对center/width/height）**惩罚和IoU脱节**：L1相同的预测，IoU可以差很多：

![L1 loss does not reflect IoU](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/detr-bbox-loss-problem.png)

那直接用IoU做loss呢？也不行——**两个box完全不重叠时IoU恒为0**，无论错得多
离谱梯度都是0（vanishing gradient）。解决：**GIoU**
（[Rezatofighi et al., 2019](https://arxiv.org/abs/1902.09630)）：

$$GIoU = IoU - \frac{|C \setminus (A \cup B)|}{|C|}$$

其中C是同时包住prediction和GT的最小convex hull。不重叠时GIoU由包络框里的
"空隙"决定（可以到-1），**预测越远惩罚越大**，梯度不再消失：

![GIoU](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/detr-giou.png)

**结果**：encoder的self-attention就能把instance分开：

![DETR qualitative results](/images/blog/Course_notes/Computer_Vision/Computer_Vision_III/cv3-note4/detr-qualitative.png)

**Summary**：架构（相对）简单、检测准确、不需要NMS，小改动就能做panoptic
segmentation。**问题**：计算和显存开销大（尤其显存）、收敛慢训练久——后续
[Deformable DETR](https://arxiv.org/abs/2010.04159)（ICLR 2021）解决了这两点。

## 2. Unsupervised (self-supervised) learning

### 2.1 Pretext tasks

*(rotation, jigsaw puzzle, colorization, SSL on videos …)*

### 2.2 Contrastive learning

*(SimCLR, MoCo …)*

### 2.3 Non-contrastive learning

*(DINO, DINOv2, DINOv3, masked autoencoders …)*

### 2.4 Evaluating SSL models

*(linear probing, fine-tuning, k-NN …)*

### 2.5 Downstream applications

*(semantic segmentation, self-supervision in videos, segmentation from motion
cues, contrastive random walk …)*

### 2.6 Additional reading

*(…)*

## 3. Semi-supervised learning

### 3.1 The semi-supervised loss

*(…)*

### 3.2 Three assumptions

*(smoothness, low-density, manifold …)*

### 3.3 Two taxonomies

*(…)*

### 3.4 Unsupervised preprocessing

*(…)*

### 3.5 Self-training

*(OnAVOS; SAM, SAM 2, SAM 3; self-training with pseudo-labels …)*

### 3.6 Intrinsically semi-supervised methods

*(entropy minimisation, virtual adversarial training …)*

### 3.7 Learning from synthetic data

*(domain alignment …)*

### 3.8 Consistency regularisation

*(…)*

## References

**Transformers**

- Vaswani et al. [Attention Is All You Need](https://arxiv.org/abs/1706.03762). NeurIPS 2017.
- Ba et al. [Layer Normalization](https://arxiv.org/abs/1607.06450). arXiv 2016.
- Dosovitskiy et al. [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929). ICLR 2021.
- Liu et al. [Swin Transformer](https://arxiv.org/abs/2103.14030). ICCV 2021.
- Carion et al. [End-to-End Object Detection with Transformers (DETR)](https://arxiv.org/abs/2005.12872). ECCV 2020.
- Rezatofighi et al. [Generalized Intersection over Union (GIoU)](https://arxiv.org/abs/1902.09630). CVPR 2019.
- Zhu et al. [Deformable DETR](https://arxiv.org/abs/2010.04159). ICLR 2021.
- Cheng et al. [Per-Pixel Classification is Not All You Need (MaskFormer)](https://arxiv.org/abs/2107.06278). NeurIPS 2021.
- Cheng et al. [Masked-attention Mask Transformer (Mask2Former)](https://arxiv.org/abs/2112.01527). CVPR 2022.

**Self-supervised learning**

- Gidaris et al. [Unsupervised Representation Learning by Predicting Image Rotations](https://arxiv.org/abs/1803.07728). ICLR 2018.
- Noroozi & Favaro. [Solving Jigsaw Puzzles](https://arxiv.org/abs/1603.09246). ECCV 2016.
- Zhang et al. [Colorful Image Colorization](https://arxiv.org/abs/1603.08511). ECCV 2016.
- Chen et al. [A Simple Framework for Contrastive Learning (SimCLR)](https://arxiv.org/abs/2002.05709). ICML 2020.
- He et al. [Momentum Contrast (MoCo)](https://arxiv.org/abs/1911.05722). CVPR 2020.
- Caron et al. [Emerging Properties in Self-Supervised Vision Transformers (DINO)](https://arxiv.org/abs/2104.14294). ICCV 2021.
- Oquab et al. [DINOv2](https://arxiv.org/abs/2304.07193). arXiv 2023.
- Siméoni et al. [DINOv3](https://arxiv.org/abs/2508.10104). arXiv 2025.
- He et al. [Masked Autoencoders Are Scalable Vision Learners (MAE)](https://arxiv.org/abs/2111.06377). CVPR 2022.
- Jabri et al. [Space-Time Correspondence as a Contrastive Random Walk](https://arxiv.org/abs/2006.14613). NeurIPS 2020.

**Semi-supervised learning**

- Kirillov et al. [Segment Anything (SAM)](https://arxiv.org/abs/2304.02643). ICCV 2023.
- Ravi et al. [SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714). arXiv 2024.
- Miyato et al. [Virtual Adversarial Training](https://arxiv.org/abs/1704.03976). TPAMI 2018.
- van Engelen & Hoos. [A Survey on Semi-Supervised Learning](https://link.springer.com/article/10.1007/s10994-019-05855-6). Machine Learning 2020.
