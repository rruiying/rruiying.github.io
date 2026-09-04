---
title: "GAMES202 — Real-Time High Quality Rendering Note4: Real-Time Physically-Based Materials"
date: 2026-09-04
tags: [Computer Graphics, real-time rendering, materials, PBR, GAMES202]
summary: GAMES202 note 4 (Lectures 10–11) — microfacet BRDFs in real time (Beckmann, GGX, GTR, Smith shadowing-masking), the Kulla-Conty approximation for multiple bounces, Disney's principled BRDF, linearly transformed cosines for polygonal lights, and non-photorealistic rendering.
---

> 对应 Lecture 10–11。主题是实时渲染里的"PBR"材质，以及 NPR。

## 1. PBR 与 RTR 里的 PBR 材质

**Physically-Based Rendering（PBR）**：渲染里的一切都应该是基于物理的，包括材质、
光照、相机、光传输等，不只是材质，但通常说 PBR 指的就是材质。

RTR 社区在 PBR 材质上远远落后于离线社区，**RTR 里的 "PB" 通常并不真的 physically
based**：

- **表面**：主要就是 **microfacet models**（用错了所以不算 PBR）和 **Disney
  principled BRDF**（对美术友好但仍不是 PBR）
- **体积**：主要是快速近似的 single scattering 和 multiple scattering（云、头发、
  皮肤等）
- 通常没有太多新理论，但有大量实现上的 hack；**性能（速度）仍是关键考虑因素**

![PBR materials in RTR](/images/blog/Course_notes/Computer_Graphics/Games202/note4/pbr-in-rtr.jpg)

## 2. Microfacet BRDF

$$f(\mathbf{i}, \mathbf{o}) = \frac{F(\mathbf{i}, \mathbf{h}) \, G(\mathbf{i}, \mathbf{o}, \mathbf{h}) \, D(\mathbf{h})}{4 (\mathbf{n} \cdot \mathbf{i})(\mathbf{n} \cdot \mathbf{o})}$$

### 2.1 Fresnel term

反射率依赖入射角（和偏振），掠射角时反射增强。Dielectric（$\eta = 1.5$）垂直入射时
很低；conductor 一直很高。准确公式要考虑偏振；近似用 **Schlick**：
$R(\theta) = R_0 + (1 - R_0)(1 - \cos\theta)^5$，$R_0 = \left(\frac{n_1 - n_2}{n_1 + n_2}\right)^2$。

### 2.2 Normal Distribution Function（NDF）

关键是微表面法线的分布：集中 ↔ glossy，分散 ↔ diffuse。注意 NDF 和统计里的正态
分布无关。有各种模型：Beckmann、GGX 等，以及细节模型（Yan 2014, 2016, 2018……）。

**Beckmann NDF**：类似高斯，但定义在 **slope space**（$\tan\theta_h$）上，所以永远
不会有超过 90° 的法线：

$$D(\mathbf{h}) = \frac{e^{-\tan^2\theta_h / \alpha^2}}{\pi \alpha^2 \cos^4\theta_h}$$

$\alpha$ 是表面的 roughness（越小越像镜面），$\theta_h$ 是半程向量与法线的夹角。

![Beckmann NDF](/images/blog/Course_notes/Computer_Graphics/Games202/note4/beckmann.jpg)

**GGX（Trowbridge-Reitz）**（Walter et al. 2007）：典型特征是**长尾（long tail）**。
高光中心之外衰减得慢，所以高光周围有一圈光晕，看起来更真实。

![GGX NDF](/images/blog/Course_notes/Computer_Graphics/Games202/note4/ggx.jpg)
![Beckmann vs GGX](/images/blog/Course_notes/Computer_Graphics/Games202/note4/beckmann-vs-ggx.jpg)

**GTR（Generalized Trowbridge-Reitz）**（Brent Burley, WDAS）：扩展 GGX，用一个参数
$\gamma$ 控制尾巴，可以更长。

![GTR](/images/blog/Course_notes/Computer_Graphics/Games202/note4/gtr.jpg)

### 2.3 Shadowing-masking term

也叫 geometry term $G$，考虑微表面的**自遮挡**：shadowing 对光，masking 对眼。
主要在**掠射角**附近提供变暗。

![Shadowing-masking term](/images/blog/Course_notes/Computer_Graphics/Games202/note4/shadowing-masking.jpg)

**为什么重要？** 假设没有 $G$，入射或出射在掠射角时分母 $(\mathbf{n} \cdot \mathbf{i})(\mathbf{n} \cdot \mathbf{o})$
趋于 0，BRDF **可以任意亮**，物体边缘会出现一圈亮边。

![Why the G term matters](/images/blog/Course_notes/Computer_Graphics/Games202/note4/why-g-term.jpg)

常用的是 **Smith shadowing-masking term**：把 shadowing 和 masking **解耦**，
$G(\mathbf{i}, \mathbf{o}, \mathbf{m}) \approx G_1(\mathbf{i}, \mathbf{m}) G_1(\mathbf{o}, \mathbf{m})$。

![Smith shadowing-masking](/images/blog/Course_notes/Computer_Graphics/Games202/note4/smith.jpg)

### 2.4 多次弹射与 Kulla-Conty 近似

**能量丢失！** 尤其在 roughness 高时明显（为什么？因为粗糙表面上更多光线被遮挡，
而被遮挡的光实际上会在微表面间继续弹射，模型直接把它丢了）。白炉测试里粗糙的球
越来越暗。

![Missing energy](/images/blog/Course_notes/Computer_Graphics/Games202/note4/missing-energy.jpg)

把丢失的能量加回来？准确方法存在（Heitz et al. 2016，微表面间的随机游走），但对 RTR
太慢。基本思想：**被遮挡 == 下一次弹射发生**。

![Multiple bounces](/images/blog/Course_notes/Computer_Graphics/Games202/note4/multiple-bounces.jpg)

**Kulla-Conty approximation**（Imageworks, 2017）。先定义一个出射 2D BRDF lobe 的
总能量（albedo）：

$$E(\mu_o) = \int_0^{2\pi} \int_0^1 f(\mu_o, \mu_i, \phi) \mu_i \, d\mu_i \, d\phi, \qquad \mu = \sin\theta$$

关键思想：**设计一个额外的 lobe，它的积分恰好是 $1 - E(\mu_o)$**。出射 lobe 对不同
的入射方向可以不同；考虑互易性，它应该是 $c(1 - E(\mu_i))(1 - E(\mu_o))$ 的形式。

![Kulla-Conty key idea](/images/blog/Course_notes/Computer_Graphics/Games202/note4/kulla-conty-idea.jpg)

解出常数 $c$：

$$f_{ms}(\mu_o, \mu_i) = \frac{(1 - E(\mu_o))(1 - E(\mu_i))}{\pi (1 - E_{avg})}, \qquad
E_{avg} = 2 \int_0^1 E(\mu) \mu \, d\mu$$

验证：把 $f_{ms}$ 对 $\mu_i$ 积分确实得到 $1 - E(\mu_o)$。

![Kulla-Conty lobe](/images/blog/Course_notes/Computer_Graphics/Games202/note4/kulla-conty-lobe.jpg)

但 $E(\mu)$ 和 $E_{avg}$ 都不是解析的。我们已经知道该怎么办：像 split sum 一样
**预计算 / 打表**。$E(\mu)$ 的参数是 roughness 和 $\mu$（2D 表），$E_{avg}$ 只有
roughness（1D 表）。

![Precomputed tables](/images/blog/Course_notes/Computer_Graphics/Games202/note4/kulla-conty-tables.jpg)
![Kulla-Conty results](/images/blog/Course_notes/Computer_Graphics/Games202/note4/kulla-conty-results.jpg)

**BRDF 有颜色怎么办？** 颜色 == 吸收 == 能量损失（这是应该的），所以只需要算总的
能量损失。定义**平均 Fresnel**（反射了多少能量，只是一个数）：

$$F_{avg} = \frac{\int_0^1 F(\mu) \mu \, d\mu}{\int_0^1 \mu \, d\mu} = 2 \int_0^1 F(\mu) \mu \, d\mu$$

回忆 $E_{avg}$ 是你能看到的能量（即不再参与后续弹射的部分）。于是各部分能量（颜色）
的比例：

- 直接看到：$F_{avg} E_{avg}$
- 弹一次后看到：$F_{avg}(1 - E_{avg}) \cdot F_{avg} E_{avg}$
- 弹 $k$ 次后看到：$F_{avg}^k (1 - E_{avg})^k \cdot F_{avg} E_{avg}$

全部加起来（等比级数）得到颜色项

$$\frac{F_{avg} E_{avg}}{1 - F_{avg}(1 - E_{avg})}$$

直接乘在无颜色的额外 BRDF 上。

![Average Fresnel](/images/blog/Course_notes/Computer_Graphics/Games202/note4/average-fresnel.jpg)
![The color term](/images/blog/Course_notes/Computer_Graphics/Games202/note4/color-term.jpg)

**一个不可取的 hack**：把 microfacet BRDF 和一个 diffuse lobe **直接相加**。在计算
机视觉的材质识别里普遍使用，但**完全错误**：物理上不对，不守恒能量（Kulla-Conty
修正了这一点，也可以用别的方式修正）。

![An undesirable hack](/images/blog/Course_notes/Computer_Graphics/Games202/note4/undesirable-hack.jpg)

## 3. Disney's principled BRDF

**为什么需要它？** 没有一个基于物理的材质能表示所有真实材质（比如大多数 microfacet
模型缺少 diffuse 项）；基于物理的材质**对美术不友好**（比如"复折射率 $n - ik$"）。
设计目标：**art directable**，不一定物理正确，但在实时渲染里仍被叫做 PBR。

![Why Disney's principled BRDF](/images/blog/Course_notes/Computer_Graphics/Games202/note4/disney-motivation.jpg)

**"Principled" 是什么意思？** BRDF 按几条重要原则设计：

- 用**直观**而不是物理的参数
- 参数**尽量少**
- 参数在合理范围内**取值 0 到 1**
- 在有意义的地方允许参数超出合理范围
- 所有参数组合都应尽可能鲁棒和合理

![What is principled](/images/blog/Course_notes/Computer_Graphics/Games202/note4/principled.jpg)

参数（每一行是一个参数从 0 到 1 的效果）：subsurface、metallic、specular、
specularTint、roughness、anisotropic、sheen、sheenTint、clearcoat、clearcoatGloss。

![Effects of individual parameters](/images/blog/Course_notes/Computer_Graphics/Games202/note4/disney-parameters.jpg)

**Pros**：容易理解和控制；一个模型覆盖很大范围的材质；有开源实现。**Cons**：不是
基于物理的（但这是大问题吗？学术界 vs 工业界）；参数空间巨大。

![Pros and cons](/images/blog/Course_notes/Computer_Graphics/Games202/note4/disney-pros-cons.jpg)

## 4. Linearly Transformed Cosines（LTC）

解决 **microfacet 模型（主要是 GGX，其他也行）在多边形光源下的 shading**，不考虑
阴影。

![LTC](/images/blog/Course_notes/Computer_Graphics/Games202/note4/ltc.jpg)

**关键思想**：给定观察方向，任何出射的 2D BRDF lobe 都可以通过一个线性变换变成一个
**cosine lobe**；光源的形状也可以跟着一起变换；**在 cosine lobe 上对变换后的（多边形）
光源积分是解析的**。

![LTC key idea](/images/blog/Course_notes/Computer_Graphics/Games202/note4/ltc-idea.jpg)

观察：用 $M^{-1}$，BRDF → cosine；方向 $\omega_i \to \omega_i'$；积分域（多边形）
$P \to P'$。

![LTC observations](/images/blog/Course_notes/Computer_Graphics/Games202/note4/ltc-observations.jpg)

**方法**：一个简单的变量替换 $\omega_i = \frac{M \omega_i'}{\|M \omega_i'\|}$，

$$L(\omega_o) = L_i \cdot \int_P F(\omega_i) \, d\omega_i
= L_i \cdot \int_{P'} \cos(\omega_i') \, d\frac{M \omega_i'}{\|M \omega_i'\|}
= L_i \cdot \int_{P'} \cos(\omega_i') \, J \, d\omega_i'$$

最后一步是解析的（cosine 在球面多边形上的积分有闭式解，Lambert 1760）。假设光源
radiance 均匀（$L_i$ 提到积分外）。$M$ 按 (roughness, 观察角) 预先拟合并打表。

![LTC approach](/images/blog/Course_notes/Computer_Graphics/Games202/note4/ltc-approach.jpg)
![LTC results](/images/blog/Course_notes/Computer_Graphics/Games202/note4/ltc-results.jpg)

## 5. Non-Photorealistic Rendering（NPR）

**NPR == stylization**；在实时渲染里，NPR == **快速可靠的**风格化。

- **Photorealistic rendering** 的目标是和照片无法区分，关注光照、阴影、材质等
- **NPR** 的目标是产生艺术化的外观

NPR 的特点：**从 photorealistic rendering 出发**；利用**抽象**；**强化重要部分**。
应用：艺术、可视化、说明、教育、娱乐（Atelier Ryza、进击的巨人）。

![Characteristics of NPR](/images/blog/Course_notes/Computer_Graphics/Games202/note4/npr-characteristics.jpg)

### 5.1 什么是风格

从异度神剑 2 的画面能总结出：**粗轮廓**（其实是 outlines）、**色块**、**表面上的
笔触**。

![What are styles](/images/blog/Course_notes/Computer_Graphics/Games202/note4/styles.jpg)

### 5.2 Outline rendering

Outlines 不只是 contour：**[B]oundary / border edge**（只属于一个面的边）、
**[C]rease**（折痕，两面夹角大）、**[M]aterial edge**（材质分界）、**[S]ilhouette
edge**（轮廓，一面朝前一面朝后）。

![Outlines](/images/blog/Course_notes/Computer_Graphics/Games202/note4/outlines.jpg)

三类做法：

- **Shading**：把 shading normal 与观察方向垂直的表面区域**变暗**。问题：边的粗细
  随曲率变化
- **Geometry：backface fattening**。正常渲染正面，把背面**变胖**再渲染一次，胖出来
  的部分就是轮廓线。扩展：沿顶点法线变胖
- **Image**：在图像上做边缘检测，通常用 **Sobel** 算子；可以在不同的信息上做
  （颜色、深度、法线）

![Outline by shading](/images/blog/Course_notes/Computer_Graphics/Games202/note4/outline-shading.jpg)
![Backface fattening](/images/blog/Course_notes/Computer_Graphics/Games202/note4/backface-fattening.jpg)
![Sobel edge detection](/images/blog/Course_notes/Computer_Graphics/Games202/note4/sobel.jpg)

### 5.3 色块

两种方式：**hard shading**，对 shading 做阈值化；**posterization**，对最终图像颜色
做阈值化。不一定是二值的，可以**量化**成几级。不同的成分可以用不同的风格（比如
diffuse 用色块、specular 保留）。

![Color blocks](/images/blog/Course_notes/Computer_Graphics/Games202/note4/color-blocks.jpg)
![Quantization](/images/blog/Course_notes/Computer_Graphics/Games202/note4/quantization.jpg)

### 5.4 笔触

有时不要色块而要模仿素描。思路：用**预先生成的笔触纹理**替代逐点的 shading。
问题：密度？连续性？**Tonal Art Maps（TAMs）**：不同密度的笔触，每个密度有自己的
MIPMAP，这样远近看密度一致，而且密的纹理包含疏的纹理的笔触以保证连续。

![Strokes](/images/blog/Course_notes/Computer_Graphics/Games202/note4/strokes.jpg)
![Tonal art maps](/images/blog/Course_notes/Computer_Graphics/Games202/note4/tonal-art-maps.jpg)

### 5.5 一些说明

NPR 是艺术驱动的，但需要把美术的需求"翻译"成渲染上的洞见（比如什么是"边"）。
沟通很重要，有时每个角色甚至每个部件都不同。**关键观察**：photorealistic 模型在 NPR
里也非常重要（比如布料的写实模型可以直接风格化）。
