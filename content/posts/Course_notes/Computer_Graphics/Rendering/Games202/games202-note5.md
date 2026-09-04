---
title: "GAMES202 — Real-Time High Quality Rendering Note5: Real-Time Ray Tracing and Industrial Solutions"
date: 2026-09-04
tags: [Computer Graphics, real-time rendering, ray tracing, denoising, GAMES202]
summary: GAMES202 note 5 (Lectures 12–14) — real-time ray tracing at 1 SPP and why denoising is the key, temporal accumulation with motion vectors and its failure cases, spatial filtering (Gaussian, bilateral, joint bilateral, separable and à-trous filters, outlier clamping), SVGF and RAE, and industrial solutions such as TAA, DLSS, deferred/tiled/clustered shading, LoD and hybrid GI like Lumen.
---

> 对应 Lecture 12–14，课程的最后一块。Volumetric / scattering 材质本课不讲
> （依赖太多，留给离线渲染课）。

## 1. Real-Time Ray Tracing（RTRT）

### 1.1 RTRT 正在发生

实时工业界说 "Ray tracing is the future and ever will be." 2018 年 NVIDIA 发布
GeForce RTX（Turing 架构），打开了 2500 亿美元的市场。RTX 能做的：光追的阴影、
反射与高光、AO、全局光照。

**RTX 真正做到的是什么？** 10 Giga rays per second == 对实时应用来说
**1 sample per pixel**。

![10 Giga rays per second](/images/blog/Course_notes/Computer_Graphics/Games202/note5/rtx-1spp.jpg)

**1 SPP path tracing** = 1 次光栅化（primary，替代 primary ray）+ 1 条光线（primary
visibility，到光源的 shadow ray）+ 1 条光线（secondary bounce）+ 1 条光线
（secondary visibility）。

![1 SPP path tracing](/images/blog/Course_notes/Computer_Graphics/Games202/note5/one-spp-path-tracing.jpg)

1 SPP 的结果**极其噪**。**关键技术是 denoising**。

![1 SPP is extremely noisy](/images/blog/Course_notes/Computer_Graphics/Games202/note5/noisy-1spp.jpg)

### 1.2 去噪的目标

在 1 SPP 下：**质量**（不过度模糊、无 artifact、保留所有细节）和**速度**（去噪一帧
少于 2 ms）。听起来是 mission impossible。学术界的方法都做不到：sheared filtering
系列（SF、AAF、FSF、MAAF）、其他离线滤波（IPP、BM3D、APR）、深度学习系列
（CNN、autoencoder）。

![Denoising goals](/images/blog/Course_notes/Computer_Graphics/Games202/note5/denoising-goals.jpg)

### 1.3 工业解法：Temporal

三个最重要的想法：**Temporal！Temporal！！Temporal！！！**

- 假设**上一帧已经去噪好了**，复用它
- 用 **motion vector** 找到当前像素在上一帧的位置
- 本质上是**增加了 SPP**（把之前所有帧的样本累积起来）

![Temporal](/images/blog/Course_notes/Computer_Graphics/Games202/note5/temporal.jpg)

**G-Buffers**：渲染过程中**免费**得到的辅助信息，通常是每像素的深度、法线、世界
坐标、albedo 等。注意它们只有屏幕空间的信息。

![The G-buffers](/images/blog/Course_notes/Computer_Graphics/Games202/note5/g-buffers.jpg)

**Back projection**：当前帧 i 的像素 x，在上一帧 i−1 的哪里？

1. 如果 G-buffer 里有世界坐标 $s$，直接取；否则 $s = M^{-1} V^{-1} P^{-1} E^{-1} x$
   （$E$ 是 viewport 变换，仍需要 z 值）
2. 物体的运动 $T$ 是已知的（$s' \xrightarrow{T} s$），所以 $s' = T^{-1} s$
3. 把上一帧的世界坐标投影到上一帧的屏幕：$x' = P' V' M' s'$

![Back projection](/images/blog/Course_notes/Computer_Graphics/Games202/note5/back-projection.jpg)

**Temporal accumulation / denoising**。记 $\tilde{C}$ 为未滤波的，$\bar{C}$ 为
滤波后的：

$$\bar{C}^{(i)} = Filter[\tilde{C}^{(i)}], \qquad C^{(i)} = \alpha \bar{C}^{(i)} + (1 - \alpha) C^{(i - 1)}$$

$\alpha = 0.1 \sim 0.2$，即这一帧 **80% 到 90% 的贡献来自之前的帧**。

![Temporal accumulation](/images/blog/Course_notes/Computer_Graphics/Games202/note5/temporal-accumulation.jpg)
![1 SPP RTGI](/images/blog/Course_notes/Computer_Graphics/Games202/note5/temporal-noisy.jpg)
![1 SPP RTGI with denoising](/images/blog/Course_notes/Computer_Graphics/Games202/note5/temporal-denoised.jpg)

### 1.4 Temporal 的失败情况

时序信息不总是可用：

- **Case 1：切换场景**（burn-in period，前几帧没有历史）
- **Case 2：在走廊里倒着走**（屏幕空间问题：新出现的东西不在上一帧的屏幕里）
- **Case 3：突然出现的背景**（disocclusion，遮挡关系变化，back projection 找到的
  上一帧位置是另一个物体）

![Failure: switching scenes](/images/blog/Course_notes/Computer_Graphics/Games202/note5/failure-scene-switch.jpg)
![Failure: walking backwards](/images/blog/Course_notes/Computer_Graphics/Games202/note5/failure-walking-backwards.jpg)
![Failure: disocclusion](/images/blog/Course_notes/Computer_Graphics/Games202/note5/failure-disocclusion.jpg)

**忽略时序失败会怎样？** 当然不正确，artifact 是 **lagging（拖影、ghosting）**。

![Lagging](/images/blog/Course_notes/Computer_Graphics/Games202/note5/lagging.jpg)

**调整**：

- **Clamping**：把上一帧的值 $C^{(i - 1)}$ **clamp 到当前帧的附近**再混合
- **Detection**：用 object ID 等检测时序失败，调整 $\alpha$（二值或连续），可能需要
  加强 / 放大空间滤波
- 问题：**重新引入噪声**（本质上是在噪声和拖影之间权衡）

![Adjustments to temporal failure](/images/blog/Course_notes/Computer_Graphics/Games202/note5/adjustments.jpg)

**更多的时序失败发生在 shading 上**：栅栏场景里光源在背后移动，几何没动但阴影动了，
**阴影的 motion vector 是什么？** 移动的椅子，glossy 地板上反射像的 motion vector
是什么？几何的 motion vector 对 shading 不适用，会出现 detached / lagging 的阴影和
反射。

![Motion vector of shadows](/images/blog/Course_notes/Computer_Graphics/Games202/note5/shadow-motion-vector.jpg)
![Motion vector of reflections](/images/blog/Course_notes/Computer_Graphics/Games202/note5/reflection-motion-vector.jpg)

**说明**：Temporal accumulation 的灵感来自 TAA，两者非常相似，时序复用本质上是提高
采样率。进一步缓解时序失败的研究：Eurographics 论文 "Temporally Reliable Motion
Vectors for Real-time Ray Tracing"（专门为阴影、glossy 反射、遮挡设计 motion vector）。

## 2. 空间滤波的实现

### 2.1 Gaussian filter

要（低通）滤波一张图去掉高频噪声。输入噪声图 $\tilde{C}$ 和滤波核 $K$（可以逐像素
不同），输出 $\bar{C}$。以像素 i 为中心的高斯滤波，邻域内的每个像素 j 按距离贡献：

```text
For each pixel i
    sum_of_weights = sum_of_weighted_values = 0.0
    For each pixel j around i
        Calculate the weight w_ij = G(|i - j|, sigma)
        sum_of_weighted_values += w_ij * C_input[j]
        sum_of_weights += w_ij
    C_output[i] = sum_of_weighted_values / sum_of_weights
```

注意：记录权重之和做**归一化**；检查权重和是否为零（对其他核）；颜色可以多通道。

![Gaussian filtering](/images/blog/Course_notes/Computer_Graphics/Games202/note5/gaussian-filter.jpg)

### 2.2 Bilateral filtering

高斯滤波的问题：**把边界也模糊了**，而边界是我们想保留的高频。观察：边界 ↔ 颜色
剧烈变化。思路：**如果 j 的颜色和 i 差太多，就让 j 贡献得少**，给核加一个控制项：

$$w(i, j, k, l) = \exp\left(-\frac{(i - k)^2 + (j - l)^2}{2\sigma_d^2} - \frac{\|I(i, j) - I(k, l)\|^2}{2\sigma_r^2}\right)$$

效果不错，但对噪声很大的图（噪点本身就是颜色剧变）不够。

![Bilateral filtering](/images/blog/Course_notes/Computer_Graphics/Games202/note5/bilateral.jpg)

### 2.3 Joint bilateral filtering

高斯用 1 个度量（距离），双边用 2 个（位置距离和颜色距离）。**能用更多的"特征"
来引导滤波吗？** 可以，这就是 **cross / joint bilateral filtering**，特别适合给
path tracing 的结果去噪。

![Joint bilateral filtering](/images/blog/Course_notes/Computer_Graphics/Games202/note5/joint-bilateral.jpg)

渲染里的独特优势：有大量免费的特征，即 **G-buffers**（法线、深度、位置、object ID
等，主要是几何的）。更好的是，**G-buffer 是没有噪声的**，因为它们和多次弹射无关。

![Joint bilateral with G-buffers](/images/blog/Course_notes/Computer_Graphics/Games202/note5/joint-bilateral-gbuffers.jpg)

说明：度量本身不需要归一化，滤波过程会归一化；高斯不是唯一选择，任何随"距离"
递减的函数都行（指数、clamped cosine 等）。

![Notes on joint bilateral filtering](/images/blog/Course_notes/Computer_Graphics/Games202/note5/joint-bilateral-notes.jpg)

例子：考虑深度、法线、颜色。A 和 B 之间的边界靠**深度**区分，B 和 C 之间靠**法线**，
D 和 E 之间靠**颜色**。

![Joint bilateral example](/images/blog/Course_notes/Computer_Graphics/Games202/note5/joint-bilateral-example.jpg)

### 2.4 实现大滤波器

每个像素要遍历 N×N 邻域：小滤波器（7×7）没问题，大滤波器（64×64）代价太大。两种
方案：

**方案 1：分离的 pass。** 2D 高斯拆成一个水平 pass（1×N）和一个垂直 pass（N×1），
查询次数从 $N^2$ 降到 $N + N$。为什么可以？因为 **2D 高斯核是可分离的**：
$G_{2D}(x, y) = G_{1D}(x) \cdot G_{1D}(y)$，而滤波是卷积，
$\iint F(x_0, y_0) G_{2D}(x_0 - x, y_0 - y) \, dx \, dy = \int \left(\int F(x_0, y_0) G_{1D}(x_0 - x) \, dx\right) G_{1D}(y_0 - y) \, dy$。
所以分离 pass 要求核可分离，**理论上双边滤波不能分离实现**（实践中也有人硬这么做）。

![Separate passes](/images/blog/Course_notes/Computer_Graphics/Games202/note5/separate-passes.jpg)
![Separable kernel](/images/blog/Course_notes/Computer_Graphics/Games202/note5/separable-kernel.jpg)

**方案 2：逐渐增大的尺寸。** 用递增的尺寸滤波多次，具体是 **à-trous wavelet**：
多个 pass，每个都是 5×5 的滤波，但**样本间隔按 $2^i$ 增长**（64×64 的滤波只需
5 个 pass 各 25 次查询）。

![À-trous wavelet](/images/blog/Course_notes/Computer_Graphics/Games202/note5/a-trous.jpg)

更深的理解：为什么要增大尺寸？**更大的滤波器 == 去掉更低的频率**。为什么跳过样本
是安全的？**采样 == 重复频谱**，第一个 pass 已经去掉了高频，第二个 pass 采样稀疏
造成的频谱重复不会和剩余的低频重叠，所以不会走样。

![Frequency view of à-trous](/images/blog/Course_notes/Computer_Graphics/Games202/note5/a-trous-frequency.jpg)

以上滤波方法也可以用于 PCSS、SSR 等的去噪。

### 2.5 Outlier removal 与 temporal clamping

滤波不是万能的：有时滤波后仍然噪甚至一块一块的，主要是**极亮的像素（outliers、
fireflies）**造成的（一个亮点被摊到整个滤波范围）。思路：**在滤波前去掉 outlier**。

**Outlier detection**：对每个像素看它的 7×7 邻域，算均值和方差，值在
$[\mu - k\sigma, \mu + k\sigma]$ 之外的就是 outlier。**Outlier removal**：把范围外
的值 **clamp** 到这个范围，注意不是扔掉（置零）。

![Outlier detection and clamping](/images/blog/Course_notes/Computer_Graphics/Games202/note5/outlier-clamping.jpg)

**Temporal clamping**：直接用上一帧的颜色会 ghosting，因为 $C^{(i - 1)}$ 可能和
$\bar{C}^{(i)}$ 差很远。在时序复用里，把 $C^{(i - 1)}$ **clamp 到 $\bar{C}^{(i)}$
的附近**：

$$C^{(i)} = \alpha \bar{C}^{(i)} + (1 - \alpha) \, clamp(C^{(i - 1)}, \mu - k\sigma, \mu + k\sigma)$$

$\mu, \sigma$ 来自当前帧的邻域。这是噪声和拖影之间的权衡；注意是把上一帧 clamp
向当前帧，不是反过来。

![Temporal clamping](/images/blog/Course_notes/Computer_Graphics/Games202/note5/temporal-clamping.jpg)

## 3. RTRT 专用的滤波方法

### 3.1 SVGF

**Spatiotemporal Variance-Guided Filtering**（Schied et al. 2017）：和基本的
时空去噪方案非常相似，但加了**方差分析**和一些技巧。

![SVGF](/images/blog/Course_notes/Computer_Graphics/Games202/note5/svgf.jpg)

三个因子引导 joint bilateral filtering：

**深度**：

$$w_z = \exp\left(-\frac{|z(p) - z(q)|}{\sigma_z |\nabla z(p) \cdot (p - q)| + \epsilon}\right)$$

理解：A 和 B 在同一平面上、颜色相近，应该互相贡献，但它们的深度差很大（平面斜着）！
所以用**相对于切平面的深度差**（深度梯度沿 p 到 q 方向的投影作为分母）。

![SVGF depth](/images/blog/Course_notes/Computer_Graphics/Games202/note5/svgf-depth.jpg)

**法线**：

$$w_n = \max(0, n(p) \cdot n(q))^{\sigma_n}$$

回忆，不一定要是高斯。注意有 normal map 时用**宏观法线**。

![SVGF normal](/images/blog/Course_notes/Computer_Graphics/Games202/note5/svgf-normal.jpg)

**亮度**（灰度）：

$$w_l = \exp\left(-\frac{|l_i(p) - l_i(q)|}{\sigma_l \sqrt{g_{3 \times 3}(Var(l_i(p)))} + \epsilon}\right)$$

噪声大的地方颜色差不可信，所以用**方差**归一化：方差在 7×7 邻域内空间计算，也用
motion vector 在时间上平均，使用前再做一次 3×3 的空间滤波。

![SVGF luminance](/images/blog/Course_notes/Computer_Graphics/Games202/note5/svgf-luminance.jpg)

结果很干净，但失败情况是 **ghosting**（比如光源移动导致阴影移动时，时序累积不知道
阴影动了）。

![SVGF failure cases](/images/blog/Course_notes/Computer_Graphics/Games202/note5/svgf-failure.jpg)

### 3.2 RAE

**Recurrent AutoEncoder**（Chaitanya et al. 2017）："Interactive Reconstruction of
Monte Carlo Image Sequences using a Recurrent Denoising AutoEncoder"。一个做去噪
（noisy → clean）的后处理网络，借助 G-buffers，**网络自动做时序累积**。

架构：**AutoEncoder（或 U-Net）**结构，skip / residual connections 让训练更快更好；
**recurrent block** 累积（并逐渐遗忘）之前帧的信息。

![RAE](/images/blog/Course_notes/Computer_Graphics/Games202/note5/rae.jpg)
![RAE architecture](/images/blog/Course_notes/Computer_Graphics/Games202/note5/rae-architecture.jpg)

**对比**：

| | 质量 | Artifact | 性能 | 可解释性 | 发表 |
|---|---|---|---|---|---|
| SVGF | 干净 | Ghosting | 快 | 有 | HPG |
| RAE（刚发明时） | 过度模糊 | Ghosting | 慢 | 无 | SIGGRAPH |

![SVGF vs RAE](/images/blog/Course_notes/Computer_Graphics/Games202/note5/svgf-vs-rae.jpg)

## 4. 工业实践（仍从科学角度看）

### 4.1 Temporal Anti-Aliasing（TAA）

为什么走样？光栅化时每像素采样不够，终极方案是更多采样。**TAA 把采样分布 / 复用到
多帧上**（每帧在像素内用不同的抖动偏移），和 RTRT 里的时序复用几乎完全一样，包括
motion vector 和 clamping。

![Temporal anti-aliasing](/images/blog/Course_notes/Computer_Graphics/Games202/note5/taa.jpg)

**关于反走样的说明**：

- **MSAA vs SSAA**：SSAA 直接以更大分辨率渲染再下采样，是终极方案但昂贵；MSAA
  是性能上的改进：**同一个图元只 shade 一次**，而且在像素间复用采样点（边界上的
  采样点被相邻像素共享）
- **图像空间的反走样**：历史 FXAA → MLAA（morphological AA）→ **SMAA**（enhanced
  subpixel morphological AA），是目前最好的图像空间方案：找到锯齿的边，把它还原成
  矢量边，按覆盖面积混合
- **G-buffer 永远不该做反走样**（法线、深度混合后没有意义）

![MSAA vs SSAA](/images/blog/Course_notes/Computer_Graphics/Games202/note5/msaa-vs-ssaa.jpg)
![SMAA](/images/blog/Course_notes/Computer_Graphics/Games202/note5/smaa.jpg)

### 4.2 Temporal super resolution：DLSS

**Super resolution / super sampling**：字面意思是提高分辨率。来源 1（DLSS 1.0）：
凭空猜；来源 2（**DLSS 2.0**）：**来自时序信息**。DLSS 2.0 的关键思想是又一个 TAA
类的应用：**时序复用采样来提高分辨率**。

![Temporal super resolution](/images/blog/Course_notes/Computer_Graphics/Games202/note5/temporal-super-resolution.jpg)

主要问题：**时序失败时 clamping 不再是选项**，因为每个更小的像素都需要一个清晰的值
（clamp 得到的是模糊的值）。所以关键是**比 clamping 更聪明地使用时序信息**，这就是
网络在做的事。

![DLSS 2.0](/images/blog/Course_notes/Computer_Graphics/Games202/note5/dlss2.jpg)

实际问题：如果 DLSS 自己每帧跑 30 ms 就已经死了，网络推理的性能优化是保密的。
对应的方案：AMD 的 FidelityFX Super Resolution，Facebook 的 Neural Supersampling
for Real-time Rendering（Xiao et al.）。

### 4.3 Deferred / tiled / clustered shading

**Deferred shading** 最初是为了节省 shading 时间。光栅化的流程是三角形 → fragment
→ 深度测试 → shade → 像素，每个 fragment 都要 shade，复杂度 $O(\#fragment \times \#light)$。
**关键观察：大部分 fragment 在最终图像里看不到**（深度测试 / 遮挡）。能只 shade
可见的 fragment 吗？

**修改光栅化过程：把场景光栅化两次。** Pass 1 不 shade，只更新深度缓冲；pass 2
一样但只有深度相等的 fragment 通过（所以只 shade 可见的）。隐含假设是光栅化场景比
shade 所有看不见的 fragment 快得多（通常成立）。复杂度 → $O(\#vis.frag \times \#light)$。
问题：**难做反走样**（几何信息在 G-buffer 里已经离散了），但几乎被 TAA 完全解决。

![Deferred shading](/images/blog/Course_notes/Computer_Graphics/Games202/note5/deferred-shading.jpg)
![Deferred shading: two passes](/images/blog/Course_notes/Computer_Graphics/Games202/note5/deferred-shading-2.jpg)

**Tiled shading**：把屏幕分成 32×32 的 tile 分别 shade。关键观察：**不是所有光源都
能照到某个 tile**，主要因为距离平方衰减（给光源设影响范围）。复杂度 →
$O(\#vis.frag \times avg\ \#light\ per\ tile)$。

![Tiled shading](/images/blog/Course_notes/Computer_Graphics/Games202/note5/tiled-shading.jpg)

**Clustered shading**：每个 tile 再按深度分段，本质上是把视锥分成 3D 网格。关键
观察：每个 tile 的深度范围可能很大，很多光源被判定可能照到这个 tile，但实际只照到
很小的深度范围。复杂度 → $O(\#vis.frag \times avg\ \#light\ per\ cluster)$。

![Clustered shading](/images/blog/Course_notes/Computer_Graphics/Games202/note5/clustered-shading.jpg)

### 4.4 Level of Detail（LoD）

LoD 非常重要，回忆纹理的 MIPMAP：选对细节层级可以节省计算。RTR 工业界常把多层
细节叫 "**cascaded**"，例子：**cascaded shadow maps**（近处高分辨率、远处低分辨率
的 shadow map）、cascaded LPV。

![Level of detail](/images/blog/Course_notes/Computer_Graphics/Games202/note5/lod.jpg)

**关键挑战：不同层级之间的过渡**，通常在边界附近需要重叠和混合。另一个例子：
**geometric LoD**，预先生成一组不同三角形数的简化模型，按到相机的距离选（或模型
的一部分，使得没有三角形比一个像素大）。Popping artifact？交给 TAA！**这就是 UE5
的 Nanite**（当然 Nanite 远不止这些：不同地方不同层级的裂缝怎么办、动态加载调度
怎么最好地利用缓存和带宽、用三角形还是 geometry texture 表示几何、裁剪和剔除……）。

![Geometric LoD](/images/blog/Course_notes/Computer_Graphics/Games202/note5/geometric-lod.jpg)

### 4.5 全局光照的方案

从本课可以看到：**没有一个 GI 方案对所有情况都完美，除了 RTRT**，但当前一代完全用
RTRT 仍太贵，所以工业界倾向于**混合方案**。比如：用 **SSR** 做粗略的 GI 近似；SSR
失败时（屏幕外、被遮挡）切换到更复杂的光线追踪，硬件或软件的。

![GI solutions](/images/blog/Course_notes/Computer_Graphics/Games202/note5/gi-solutions.jpg)

- **软件光线追踪**：近处单个物体用高质量 SDF，整个场景用低质量 SDF；有强方向光 /
  点光时用 RSM；在 3D 网格里存 irradiance 的 probes（Dynamic Diffuse GI，DDGI）
- **硬件光线追踪**：不必用原始几何，用低模 proxy；probes（RTXGI）

把这些混合起来就是 **UE5 的 Lumen**。

![Lumen](/images/blog/Course_notes/Computer_Graphics/Games202/note5/lumen.jpg)

### 4.6 结语

**什么是有趣的？** 任何需要思考的东西，所以放弃思考 == 自杀。**实现比理论不重要吗？**
从不，但工程技能必须在工程中获得。

课程四块：real-time shadows / environment lighting、real-time global illumination、
real-time shading / materials、real-time ray tracing。没覆盖的话题还有很多：给 SDF
贴纹理、透明材质与 order-independent transparency、粒子渲染、后处理（景深、运动
模糊）、随机种子与蓝噪声、foveated rendering、基于 probe 的 GI、ReSTIR、Neural
Radiance Caching、many-light 理论与 light cuts、participating media 与 SSSSS、
头发外观……

![Uncovered topics](/images/blog/Course_notes/Computer_Graphics/Games202/note5/uncovered-topics.jpg)
