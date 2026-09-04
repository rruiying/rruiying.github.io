---
title: "GAMES202 — Real-Time High Quality Rendering Note3: Real-Time Global Illumination"
date: 2026-09-04
tags: [Computer Graphics, real-time rendering, global illumination, GAMES202]
summary: GAMES202 note 3 (Lectures 7–9) — one-bounce indirect illumination in real time, in 3D with reflective shadow maps, light propagation volumes and voxel GI, and in screen space with SSAO, SSDO and screen space reflections.
---

> 对应 Lecture 7 的后半、Lecture 8 和 Lecture 9。实时全局光照分两类：**3D 空间**的
> 方法（RSM、LPV、VXGI）和**屏幕空间**的方法（SSAO、SSDO、SSR）。

## 1. 引言

全局光照很重要但很复杂。在 RTR 里，人们追求**一次弹射间接光照（one bounce indirect
illumination）**的简单快速方案。从 GAMES101 知道：**任何被直接照亮的表面都会再次
成为光源**（secondary light source）。

![Directly lit surfaces act as lights](/images/blog/Course_notes/Computer_Graphics/Games202/note3/surfaces-as-lights.jpg)

要给任意点 p 加上间接光照，需要回答两个问题：

- **Q1：哪些表面片被直接照亮？**（提示：哪个技术能告诉你这个？）
- **Q2：每个表面片对 p 的贡献是多少？**然后把所有表面片的贡献加起来（提示：每个
  表面片像一个面光源）

![Key observations](/images/blog/Course_notes/Computer_Graphics/Games202/note3/key-observations.jpg)

## 2. Reflective Shadow Maps（RSM）

### 2.1 Q1

**经典的 shadow map 完美解决 Q1**：shadow map 上每个像素就是一个小表面片。每个像素
的出射 radiance 是确切知道的，但只知道**朝向相机（光源）方向**的。**假设：所有反射物
都是 diffuse 的**，出射 radiance 向所有方向均匀。

![RSM: Q1](/images/blog/Course_notes/Computer_Graphics/Games202/note3/rsm-q1.jpg)

### 2.2 Q2

每个表面片对 p 的贡献是对该片覆盖的立体角的积分，可以转成对面积的积分：

$$L_o(\mathbf{p}, \omega_o) = \int_{\Omega_{patch}} L_i(\mathbf{p}, \omega_i) V(\mathbf{p}, \omega_i) f_r(\mathbf{p}, \omega_i, \omega_o) \cos\theta_i \, d\omega_i
= \int_{A_{patch}} L_i(\mathbf{q} \to \mathbf{p}) V(\mathbf{p}, \omega_i) f_r(\mathbf{p}, \mathbf{q} \to \mathbf{p}, \omega_o) \frac{\cos\theta_p \cos\theta_q}{\|\mathbf{q} - \mathbf{p}\|^2} \, dA$$

对 diffuse 反射片：$f_r = \rho / \pi$，$L_i = f_r \cdot \Phi / dA$（$\Phi$ 是该片
接收的 flux）。于是每个像素 q 对 p 的贡献

$$E_p(\mathbf{x}, \mathbf{n}) = \Phi_p \frac{\max\{0, \langle \mathbf{n}_p, \mathbf{x} - \mathbf{x}_p \rangle\} \max\{0, \langle \mathbf{n}, \mathbf{x}_p - \mathbf{x} \rangle\}}{\|\mathbf{x} - \mathbf{x}_p\|^4}$$

用 flux 而不是 radiance 的好处是不用管 shadow map 上每个像素代表的实际面积。

![RSM: Q2](/images/blog/Course_notes/Computer_Graphics/Games202/note3/rsm-q2.jpg)

### 2.3 哪些像素有贡献，以及加速

不是 RSM 里所有像素都能贡献：**可见性**（仍然难处理，RSM 直接忽略）、**朝向**
（背对 p 的不贡献）、**距离**（太远的忽略）。

![Not all pixels contribute](/images/blog/Course_notes/Computer_Graphics/Games202/note3/rsm-not-all-pixels.jpg)

理论上 shadow map 的所有像素都可能贡献。能减少吗？提示：PCSS 的 step 1 和 3。
**采样**：把 p 投影到 shadow map 上，在它周围的一个区域内采样（比如 400 个点），
离得近的采得密，远的采得稀，用权重补偿。

![Acceleration by sampling](/images/blog/Course_notes/Computer_Graphics/Games202/note3/rsm-acceleration.jpg)

RSM 要记录：**深度、世界坐标、法线、flux** 等。常用于游戏里的**手电筒**（Gears of
War 4、Uncharted 4、The Last of Us）。

**Pros**：容易实现。**Cons**：性能随光源数线性增长；间接光照没有可见性检查；假设
很多（diffuse 反射物、把深度当距离等）；采样率 / 质量的权衡。

![RSM pros and cons](/images/blog/Course_notes/Computer_Graphics/Games202/note3/rsm-pros-cons.jpg)

## 3. Light Propagation Volumes（LPV）

CryEngine 3 首先引入，快而且质量不错。

- **关键问题**：在任意 shading point 查询来自任意方向的 radiance
- **关键思想**：**radiance 沿直线传播且不变**
- **关键方案**：用一个 **3D 网格**把 radiance 从直接被照亮的表面**传播**到任何地方

![LPV idea](/images/blog/Course_notes/Computer_Graphics/Games202/note3/lpv-idea.jpg)

四步：

1. **Generation**：找直接被照亮的表面，用 RSM 就够了；可以用一个精简的 diffuse
   表面片集合（virtual light sources）
2. **Injection**：预先把场景分成 3D 网格；对每个格子找到其中的 virtual lights，把
   它们的方向 radiance 分布加起来，**投影到前 2 阶 SH（共 4 个系数）**
3. **Propagation**：每个格子从它的 6 个面收集接收到的 radiance，加起来再用 SH 表示；
   重复传播几次直到 volume 稳定
4. **Rendering**：对任意 shading point 找到它所在的格子，取出格子里（来自所有方向
   的）入射 radiance，shading

![LPV steps](/images/blog/Course_notes/Computer_Graphics/Games202/note3/lpv-steps.jpg)
![LPV injection](/images/blog/Course_notes/Computer_Graphics/Games202/note3/lpv-injection.jpg)
![LPV propagation](/images/blog/Course_notes/Computer_Graphics/Games202/note3/lpv-propagation.jpg)

**问题：light leaking**。格子比几何粗时，一个格子内的 radiance 被认为处处相同，
薄墙背面的点会被墙前面的光照亮。

![LPV light leaking](/images/blog/Course_notes/Computer_Graphics/Games202/note3/lpv-light-leaking.jpg)

## 4. Voxel Global Illumination（VXGI）

仍是两 pass 算法，与 RSM 的两个主要区别：

- 直接照亮的像素 → **（层次化的）体素**
- 在 RSM 上采样 → **在 3D 里追踪反射的锥体（cone tracing）**（RSM 的采样是不准的）

![VXGI](/images/blog/Course_notes/Computer_Graphics/Games202/note3/vxgi.jpg)

先把整个场景**体素化**并建立层次结构（类似 octree）。

- **Pass 1 从光源**：在每个体素里存**入射光的分布和法线的分布**，向上更新层次
- **Pass 2 从相机**：对 glossy 表面，向反射方向追踪 **1 个锥体**，锥体随距离变粗，
  按锥体的大小查询层次结构的相应层级（体素的法线分布 + 入射分布可以算出该体素向
  锥体方向的出射 radiance）；对 diffuse 表面，追踪**若干个（比如 8 个）锥体**

![VXGI pass 1](/images/blog/Course_notes/Computer_Graphics/Games202/note3/vxgi-pass1.jpg)
![VXGI pass 2](/images/blog/Course_notes/Computer_Graphics/Games202/note3/vxgi-pass2.jpg)
![VXGI diffuse](/images/blog/Course_notes/Computer_Graphics/Games202/note3/vxgi-diffuse.jpg)

结果相当好，接近光线追踪。代价是体素化和层次结构的开销，场景动态时要重新体素化。

![VXGI results](/images/blog/Course_notes/Computer_Graphics/Games202/note3/vxgi-results.jpg)

## 5. Screen Space Ambient Occlusion（SSAO）

### 5.1 什么是 screen space

**只使用"屏幕"上的信息**，换句话说，是对已有渲染结果的**后处理**。

![GI in screen space](/images/blog/Course_notes/Computer_Graphics/Games202/note3/screen-space.jpg)

### 5.2 为什么要 AO

Crytek 再次首先引入。**实现便宜，但大大增强了相对位置的感觉**（接触处的暗角）。

![Why AO](/images/blog/Course_notes/Computer_Graphics/Games202/note3/why-ao.jpg)

SSAO 是全局光照在屏幕空间的近似。三个关键思想：

1. 我们不知道入射的间接光照，**假设它是常数**（对所有 shading point、所有方向）。
   听起来很熟悉：Blinn-Phong 的 ambient 项
2. **考虑不同 shading point 上不同的可见性**（向所有方向）。Ambient 项是常数，
   AO 给它乘上一个逐点的可见性
3. 假设 **diffuse 材质**

![SSAO key ideas](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssao-idea.jpg)

### 5.3 理论

一切从渲染方程开始，再次使用 RTR 的近似，**把可见性拆出来**：

$$L_o^{indir}(\mathbf{p}, \omega_o) \approx \underbrace{\frac{\int_{\Omega^+} V(\mathbf{p}, \omega_i) \cos\theta_i \, d\omega_i}{\int_{\Omega^+} \cos\theta_i \, d\omega_i}}_{k_A\text{: cosine 加权的平均可见性}}
\cdot \underbrace{\int_{\Omega^+} L_i^{indir}(\mathbf{p}, \omega_i) f_r(\mathbf{p}, \omega_i, \omega_o) \cos\theta_i \, d\omega_i}_{= L_i^{indir}(\mathbf{p}) \cdot \frac{\rho}{\pi} \cdot \pi = L_i^{indir}(\mathbf{p}) \cdot \rho}$$

![SSAO theory](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssao-theory.jpg)

**更深的理解 1**：这个近似等于 $\int_\Omega \bar{f}(x) g(x) \, dx$，其中 $\bar{f}$ 是
$f$ 在 $g$ 的 support 上的平均。在 AO 里近似是**准确**的，因为 $g = L \cdot f_r$ 是
常数。

**更深的理解 2：为什么可以把 cosine 项和 $d\omega_i$ 放在一起？** 因为
**projected solid angle** $dx_\perp = \cos\theta_i \, d\omega_i$：把单位半球投影到
单位圆盘，投影立体角的积分就是单位圆盘的面积 $\pi$。

![Projected solid angle](/images/blog/Course_notes/Computer_Graphics/Games202/note3/projected-solid-angle.jpg)

其实有一个简单得多的理解：入射光均匀（$L_i$ 常数）、diffuse BRDF（$f_r = \rho / \pi$
常数），两者直接从积分里拿出来：

$$L_o(\mathbf{p}, \omega_o) = \frac{\rho}{\pi} \cdot L_i(\mathbf{p}) \cdot \int_{\Omega^+} V(\mathbf{p}, \omega_i) \cos\theta_i \, d\omega_i$$

### 5.4 怎么在屏幕空间算 $k_A$

问题变成：**在 shading point 周围的半球里，有多少方向被挡住了**。SSAO（Crytek 2007）
的做法：

1. 以 p 为中心、在一个**球**内随机取若干采样点（不用法线，因为当时 deferred
   shading 还不普及，屏幕空间拿不到法线）
2. 对每个采样点，和深度缓冲比较：如果采样点的深度**比深度缓冲里记录的深**，就认为
   它在几何内部，被挡住了
3. 被挡住的比例超过一半时才开始变暗（因为球的下半部分总是在物体内部）

![SSAO sampling](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssao-sampling.jpg)

问题：这个判断只是**近似**，采样点在深度缓冲里被挡住并不代表从 p 看过去它真的被挡住
（比如悬空的物体会在下方产生错误的暗影）；只用一半的采样点是浪费；一面墙前的点会
被墙"挡住"约一半采样点，造成不正确的变暗。用**法线**（现在 deferred shading 里
有 normal buffer）只在**半球**里采样并按 cosine 加权，就是 **HBAO（Horizon-Based
AO）**一类的方法。

![SSAO false occlusion](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssao-false-occlusion.jpg)
![SSAO with normals](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssao-normal.jpg)
![HBAO](/images/blog/Course_notes/Computer_Graphics/Games202/note3/hbao.jpg)

采样数和性能的权衡：每点采样少（比如 16 个）会有噪声，然后**做一次模糊**
（edge-aware 的），和后面实时光线追踪的去噪思路一样。

![Noise and blur](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssao-blur.jpg)

## 6. Screen Space Directional Occlusion（SSDO）

SSAO 的改进：**考虑（更）真实的间接光照**。关键思想：为什么要假设间接光是均匀的？
**间接光的一些信息已经知道了**（听起来像 RSM）：SSDO 利用**已经渲染出来的直接光照**，
不是来自 RSM 而是来自相机。

![SSDO idea](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssdo-idea.jpg)

和 path tracing 非常像：在 shading point p 发一条随机光线，**不碰到障碍物就是直接
光照，碰到了就是间接光照**（障碍物那一点的直接光照贡献过来）。

![SSDO is like path tracing](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssdo-path-tracing.jpg)

**和 SSAO 的对比**：AO 是"（假设有）间接光 + 被挡住的方向没有间接光"；DO 是
"没被挡住的方向没有间接光 + 被挡住的方向有间接光"（和 path tracing 一样）。
AO 和 DO 的假设正好相反：AO 假设间接光来自远处，DO 假设间接光来自近处。

![AO vs DO](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ao-vs-do.jpg)

分别考虑未遮挡和遮挡的方向：

$$L_o^{dir}(\mathbf{p}, \omega_o) = \int_{\Omega^+, V = 1} L_i^{dir}(\mathbf{p}, \omega_i) f_r \cos\theta_i \, d\omega_i, \qquad
L_o^{indir}(\mathbf{p}, \omega_o) = \int_{\Omega^+, V = 0} L_i^{indir}(\mathbf{p}, \omega_i) f_r \cos\theta_i \, d\omega_i$$

来自一个像素（表面片）的间接光照用 RSM 里的公式。判断遮挡的方式和 HBAO 类似：在局部
半球里测试采样点的深度。

![SSDO equations](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssdo-equations.jpg)

SSDO 的质量更接近离线渲染。**问题**：仍然只是**短距离**的 GI；可见性的判断仍是近似
（采样点在深度缓冲里被挡住不等于从 p 看被挡住）；**屏幕空间的通病：看不见的表面
的信息丢失了**（比如从侧面被照亮但正面看不到的墙）。

![SSDO issues](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssdo-issues.jpg)

## 7. Screen Space Reflection（SSR）

### 7.1 什么是 SSR

仍是在 RTR 里引入全局光照的一种方式：**做光线追踪，但不需要 3D 图元**（三角形等），
把屏幕空间的深度图当作场景（一层壳）。两个基本任务：

- **Intersection**：任意光线与场景求交
- **Shading**：相交像素对 shading point 的贡献

![Screen space reflection](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssr.jpg)

### 7.2 求交：ray marching

最基本的做法：从 shading point 沿反射方向**逐步前进**（linear ray march），每一步
比较当前点的深度和深度缓冲里的值，比缓冲里的深就说明打到了。步长小则慢，步长大则
会漏掉薄物体。

![Ray marching](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssr-ray-marching.jpg)
![Linear ray march](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssr-linear-march.jpg)

**Depth mipmap（Hi-Z）**：对深度图建 mipmap，但**不是取平均而是取最小值**（min-Z
pyramid）。为什么？和 3D 里的层次结构（BVH、KD-tree）非常像，能更快地**拒绝一大片
不相交的区域**。min 操作保证了**保守**的逻辑：如果光线连一个大节点（该区域最浅的
深度）都碰不到，就不可能碰到它的任何子节点。

![Depth mipmap](/images/blog/Course_notes/Computer_Graphics/Games202/note3/depth-mipmap.jpg)
![Why depth mipmap](/images/blog/Course_notes/Computer_Graphics/Games202/note3/why-depth-mipmap.jpg)

**Hierarchical tracing**（无栈的 min-Z pyramid 遍历）：

```text
mip = 0;
while (level > -1)
    step through current cell;
    if (above Z plane) ++level;   // 没碰到，放大步长
    if (below Z plane) --level;   // 碰到了，缩小步长
```

在当前层级走一格，如果没碰到就升一级（步子更大），碰到了就降一级细查，直到第 0 层。

![Hierarchical tracing](/images/blog/Course_notes/Computer_Graphics/Games202/note3/hierarchical-tracing.jpg)

### 7.3 Shading

**和 path tracing 完全没有区别**，只是又假设反射物 / 二次光源是 diffuse 的：

$$L_o(\mathbf{p}, \omega_o) = \int_{\Omega^+} L_i(\mathbf{p}, \omega_i) f_r(\mathbf{p}, \omega_i, \omega_o) \cos\theta_i \, d\omega_i, \qquad L_i = L_o(\mathbf{q}, \mathbf{q} \to \mathbf{p})$$

对镜面反射只需要一条光线，对 glossy 按 BRDF 采样多条。**两个问题**：它引入了距离
平方衰减吗？没有，因为这是对立体角积分，不是对面积；它处理 shading point 和二次
光源之间的遮挡吗？处理了，因为光线追踪本身就是在求最近交点。

![Shading using SSR](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssr-shading.jpg)

SSR 能做到的效果：锐利和模糊的反射、**contact hardening**（离反射面近的地方反射
清晰、远的地方模糊）、**specular elongation**（高光沿反射方向拉长）、逐像素的
roughness 和法线。

![SSR requirements](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssr-requirements.jpg)

**改进**：BRDF importance sampling；**hit point reuse across neighbors**（邻居像素
打到的点也可以拿来当自己的样本，类似 photon mapping 的思路，一条光线当多条用）；
prefiltered samples，按各自的 BRDF 加权（先模糊再采样，减少噪声）。

![SSR improvements](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssr-improvements.jpg)

### 7.4 总结

**Pros**：glossy 和镜面反射性能快、质量好；没有亮斑和遮挡问题。**Cons**：diffuse
情形效率不高（要追踪的方向太多）；**屏幕外的信息缺失**（反射里看不到屏幕外的东西，
边缘处反射会截断，常用淡出处理）。

![Summary of SSR](/images/blog/Course_notes/Computer_Graphics/Games202/note3/ssr-summary.jpg)
