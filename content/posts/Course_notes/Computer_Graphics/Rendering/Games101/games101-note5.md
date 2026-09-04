---
title: "GAMES101 — Modern Computer Graphics Note5: Advanced Rendering, Cameras and Color"
date: 2026-09-03
tags: [Computer Graphics, rendering, GAMES101]
summary: GAMES101 note 5 (Lectures 18–20) — advanced light transport (BDPT, MLT, photon mapping, VCM, instant radiosity), advanced appearance (participating media, hair and fur, BSSRDF, cloth, glinty details, procedural noise), cameras and lenses with the thin-lens model and depth of field, light fields, and the physics and perception of color.
---

> 对应 Lecture 18–20。Lecture 18 是高层次的 FYI 概览，Lecture 19、20 是独立话题。

## 1. Advanced light transport

### 1.1 Biased vs unbiased

**Unbiased** 的 Monte Carlo 方法没有系统误差：无论用多少样本，估计量的期望总是正确
值。否则就是 **biased**。一个特例是 **consistent**：样本数趋于无穷时期望收敛到正确值。

![Biased vs unbiased](/images/blog/Course_notes/Computer_Graphics/Games101/note5/biased-vs-unbiased.jpg)

- **Unbiased**：bidirectional path tracing（BDPT）、Metropolis light transport（MLT）
- **Biased**：photon mapping、vertex connection and merging（VCM）
- **Instant radiosity**（VPL / many-light methods）

### 1.2 BDPT

一条路径连接相机和光源。BDPT **同时从相机和光源发出子路径**，再把两端连起来。
适合**光源侧的传输很复杂**的场景（比如光源藏在灯罩里，靠间接光照亮）。实现难，
而且相当慢。

![Bidirectional path tracing](/images/blog/Course_notes/Computer_Graphics/Games101/note5/bdpt.jpg)

### 1.3 MLT

Markov Chain Monte Carlo（MCMC）的应用：按某个 PDF 从当前样本跳到下一个样本。
**关键思想：对已有路径做局部扰动得到新路径。** 很擅长局部探索困难的光路（比如
SDS 路径，水下的焦散），且是 unbiased 的。

缺点：收敛速度难以估计；不保证每个像素收敛速度相同，所以结果通常"脏"，动画会
闪烁，一般不用于渲染动画。

![Metropolis light transport](/images/blog/Course_notes/Computer_Graphics/Games101/note5/mlt.jpg)

### 1.4 Photon mapping

Biased 的两阶段方法，非常擅长 **Specular-Diffuse-Specular（SDS）路径**和**焦散
（caustics）**。

- **Stage 1 photon tracing**：从光源发射光子，在场景里弹射，打到漫反射表面就记录下来
- **Stage 2 photon collection（final gathering）**：从相机发射子路径，弹射到漫反射
  表面为止

![Photon mapping](/images/blog/Course_notes/Computer_Graphics/Games101/note5/photon-mapping.jpg)

计算用**局部密度估计**：光子多的区域更亮。对每个 shading point 找最近的 $N$ 个
光子，取它们覆盖的面积 $\Delta A$，密度是 $N / \Delta A$。

![Density estimation](/images/blog/Course_notes/Computer_Graphics/Games101/note5/density-estimation.jpg)

**为什么 biased？** 因为 $dN / dA \ne \Delta N / \Delta A$。$N$ 小则噪，$N$ 大则糊。
但在极限意义下：发射的光子越多，同样 $N$ 个光子覆盖的 $\Delta A$ 越小，越接近
$dA$，所以是 **biased but consistent**。图形学里一个好记的理解：**biased == blurry，
consistent == 样本无穷多时不模糊**。为什么不用固定半径的范围搜索？那样发再多光子
$\Delta A$ 也不变，就不 consistent 了。

![Why photon mapping is biased](/images/blog/Course_notes/Computer_Graphics/Games101/note5/photon-mapping-bias.jpg)

### 1.5 VCM 与 instant radiosity

**Vertex Connection and Merging** 结合 BDPT 和 photon mapping：BDPT 里两端不能
连接但足够近的子路径不要浪费，用 photon mapping 的方式把附近的"光子"**合并**。

![Vertex connection and merging](/images/blog/Course_notes/Computer_Graphics/Games101/note5/vcm.jpg)

**Instant Radiosity（IR）**，也叫 many-light 方法：**被照亮的表面可以当作光源**。
发射光源子路径，把每条子路径的终点当作 Virtual Point Light（VPL），然后用这些 VPL
正常渲染场景。快，漫反射场景效果通常不错。缺点：VPL 离 shading point 很近时会出现
亮斑（$1/r^2$ 爆炸）；处理不了 glossy 材质。

![Instant radiosity](/images/blog/Course_notes/Computer_Graphics/Games101/note5/instant-radiosity.jpg)

## 2. Advanced appearance modeling

- **Non-surface models**：participating media、hair / fur / fiber（BCSDF）、
  granular material
- **Surface models**：translucent material（BSSRDF）、cloth、detailed material
  （non-statistical BRDF）
- **Procedural appearance**

![Advanced appearance modeling](/images/blog/Course_notes/Computer_Graphics/Games101/note5/advanced-appearance.jpg)

### 2.1 Participating media

雾、云。光在介质中传播时的任意一点都可能被（部分）**吸收**和**散射**。用 **phase
function** 描述介质中某点散射的角度分布（类似表面的 BRDF）。

渲染：随机选一个方向弹射，随机选一个直行的距离，在每个"shading point"连到光源。
应用：Big Hero 6、Assassin's Creed 的雾效。

![Participating media](/images/blog/Course_notes/Computer_Graphics/Games101/note5/participating-media.jpg)
![Phase function](/images/blog/Course_notes/Computer_Graphics/Games101/note5/phase-function.jpg)
![Rendering participating media](/images/blog/Course_notes/Computer_Graphics/Games101/note5/participating-media-rendering.jpg)

### 2.2 Hair and fur

**Kajiya-Kay model**：早期模型，把头发当圆柱，效果一般。**Marschner model**：把
头发当**玻璃圆柱**，三种光的作用：**R**（反射）、**TT**（透射进去再透射出来）、
**TRT**（透射、内部反射、透射）。cuticle 是表皮，cortex 吸收光。

![Marschner model](/images/blog/Course_notes/Computer_Graphics/Games101/note5/marschner-model.jpg)

**动物毛发**用人的头发模型渲染不对：不够漫射也不够饱和。区别在**medulla（髓质）**：
人的头发髓质很小，动物毛发的髓质大且结构复杂、会散射光。

![Human hair vs animal fur](/images/blog/Course_notes/Computer_Graphics/Games101/note5/hair-vs-fur.jpg)

**Double cylinder model**（Yan et al. 2015, 2017）：cortex 吸收 + medulla 散射，
在 R、TT、TRT 之外增加 **TT^s**、**TRT^s**（经过髓质散射的）lobe。渲染 60 万根
毛的仓鼠每帧 36.9 分钟。用于《猩球崛起 3》《狮子王》。

![Double cylinder model](/images/blog/Course_notes/Computer_Graphics/Games101/note5/double-cylinder.jpg)
![Double cylinder lobes](/images/blog/Course_notes/Computer_Graphics/Games101/note5/double-cylinder-lobes.jpg)

### 2.3 Granular material

沙子、糖、盐这种颗粒材质。可以避免显式建模每个颗粒吗？可以，用 procedural 定义
（Meng et al. 2015）。应用：Pixar 的 Piper。

### 2.4 Translucent material：BSSRDF

玉、水母、皮肤。**Subsurface scattering**：光从一点进入、从另一点出射，这**违反了
BRDF 的基本假设**。

**BSSRDF** 是 BRDF 的推广：一点的出射 radiance 来自另一点的入射 irradiance，
$S(\mathbf{x}_i, \omega_i, \mathbf{x}_o, \omega_o)$。渲染方程也推广为对表面所有点和
所有方向的积分：

$$L(\mathbf{x}_o, \omega_o) = \int_A \int_{H^2} S(\mathbf{x}_i, \omega_i, \mathbf{x}_o, \omega_o) L_i(\mathbf{x}_i, \omega_i) \cos\theta_i \, d\omega_i \, dA$$

![Subsurface scattering](/images/blog/Course_notes/Computer_Graphics/Games101/note5/subsurface-scattering.jpg)
![BSSRDF](/images/blog/Course_notes/Computer_Graphics/Games101/note5/bssrdf.jpg)

**Dipole approximation**（Jensen et al. 2001）：用表面上下各一个点光源来近似光的
扩散。BSSRDF 让大理石、皮肤看起来通透，BRDF 做不到。

![Dipole approximation](/images/blog/Course_notes/Computer_Graphics/Games101/note5/dipole.jpg)
![BRDF vs BSSRDF](/images/blog/Course_notes/Computer_Graphics/Games101/note5/brdf-vs-bssrdf.jpg)

### 2.5 Cloth

布是**一堆扭在一起的纤维**：纤维扭成股（ply），股再扭成线（yarn），线织成
（woven）或编成（knitted）布。三种渲染方式：

1. **当作表面**：给定编织图案算整体行为，用 BRDF 渲染。局限是天鹅绒这类
   各向异性、有体积感的布做不好
2. **当作 participating media**：从单根纤维的性质和分布得到散射参数
3. **渲染每根纤维**：最真实也最贵

![Cloth](/images/blog/Course_notes/Computer_Graphics/Games101/note5/cloth.jpg)
![Cloth as participating media](/images/blog/Course_notes/Computer_Graphics/Games101/note5/cloth-as-media.jpg)

### 2.6 Detailed appearance

渲染的车和老鼠为什么不真实？真实世界更复杂：车漆有划痕、金属有闪光。
Microfacet BRDF 用**统计**的法线分布（NDF），而实际的 NDF 有细节。

![Statistical NDF vs actual NDF](/images/blog/Course_notes/Computer_Graphics/Games101/note5/statistical-vs-actual-ndf.jpg)

用超高分辨率的 normal map（约 200K × 200K）定义细节：各向同性噪声、拉丝、金属
薄片……但渲染太难：路径要经过凹凸的镜面表面精确打到光源，几乎不可能采到，一个
像素要收敛需要 21 天以上。

**解法：一个像素上的 BRDF（P-NDF）**。把一个像素覆盖的表面 patch $P$ 内的法线分布
统计出来，它有尖锐的特征，正是闪光（glints）的来源。不同的 normal map 给出不同形状的
P-NDF（拉丝金属、椭球凸起、海浪）。应用：Rise of the Tomb Raider。

![P-NDF](/images/blog/Course_notes/Computer_Graphics/Games101/note5/p-ndf.jpg)

**最近的趋势：波动光学。** 细节尺度接近光的波长时几何光学不再适用，CD、金属薄膜、
手机屏幕的彩色条纹都是衍射。划痕金属在波动光学下的 BRDF 与几何光学的差别很大
（Yan et al. 2018）。

![Wave optics](/images/blog/Course_notes/Computer_Graphics/Games101/note5/wave-optics.jpg)

### 2.7 Procedural appearance

不用纹理也能定义细节：**实时计算一个噪声函数**。3D 噪声给出内部结构（切开或打碎
也有纹理），对噪声做阈值化得到二值噪声。复杂的噪声函数非常强大（Perlin noise、
Worley noise 等），可以做木纹、大理石、地形、水面。

![Procedural noise](/images/blog/Course_notes/Computer_Graphics/Games101/note5/procedural-noise.jpg)

## 3. Cameras and lenses

**Imaging = synthesis + capture**。前面讲的都是合成，相机是捕获。

### 3.1 相机的组成

- **Pinhole 或 lens** 在传感器上成像
- **Shutter** 让传感器精确曝光一段时间
- **Sensor** 在曝光期间累积 irradiance

为什么不能没有镜头？每个传感器点会积分来自物体所有点的光，所有像素值都差不多，
即传感器记录的是 irradiance（不过 computational imaging 有这方面的研究）。

![Why not sensors without lenses](/images/blog/Course_notes/Computer_Graphics/Games101/note5/why-lenses.jpg)

Pinhole camera 历史悠久（墨子、亚里士多德、沈括……）。针孔相机没有景深，全部清晰。

### 3.2 视场角（FOV）

**焦距对 FOV 的影响**：传感器高 $h$，焦距 $f$，

$$FOV = 2 \arctan\left(\frac{h}{2f}\right)$$

传感器固定时，**焦距越短视场角越大**。

![Effect of focal length on FOV](/images/blog/Course_notes/Computer_Graphics/Games101/note5/focal-length-fov.jpg)

习惯上用 35mm 胶片（36 × 24 mm）上的镜头焦距来指代视场角：17mm 是广角（104°），
50mm 是"标准"镜头（47°），200mm 是长焦（12°）。手机的 "28mm 等效焦距"用的就是
这个约定。**传感器越大 FOV 越大**；要在小传感器上保持 FOV，就按比例缩短焦距。

![Focal length examples](/images/blog/Course_notes/Computer_Graphics/Games101/note5/focal-length-examples.jpg)
![Maintain FOV on smaller sensor](/images/blog/Course_notes/Computer_Graphics/Games101/note5/maintain-fov.jpg)

### 3.3 曝光

$$H = T \times E, \qquad \text{Exposure} = \text{time} \times \text{irradiance}$$

- **Exposure time T**：由快门控制
- **Irradiance E**：落在传感器单位面积的功率，由光圈和焦距控制

三个控制量：

- **Aperture size**：调 f-stop 开关光圈（模仿瞳孔）
- **Shutter speed**：传感器积分光的时长
- **ISO gain**：传感器值到数字值之间的放大（模拟和/或数字）。胶片是用颗粒换灵敏度，
  数字是用噪声换灵敏度。线性：ISO 200 需要的光是 ISO 100 的一半

![Exposure](/images/blog/Course_notes/Computer_Graphics/Games101/note5/exposure.jpg)
![ISO gain](/images/blog/Course_notes/Computer_Graphics/Games101/note5/iso.jpg)

**F-number（F-stop）**：写作 FN 或 F/N。非正式理解是光圈直径的倒数，数字越小
光圈越大、越亮。

![F-number](/images/blog/Course_notes/Computer_Graphics/Games101/note5/f-number.jpg)

**快门的副作用**：**motion blur**（手抖、物体运动，快门时间翻倍则运动模糊翻倍。
不总是坏事，想想反走样）；**rolling shutter**（照片的不同部分在不同时刻拍的，快速
运动的物体会变形）。

**等效曝光**：F1.4 + 1/500 s 和 F2.0 + 1/250 s 等价（光圈每一档面积减半，快门时间
翻倍）。摄影师要在**景深**和**运动模糊**之间取舍。高速摄影用极快的快门配大光圈或
高 ISO；长曝光相反。

![Constant exposure](/images/blog/Course_notes/Computer_Graphics/Games101/note5/constant-exposure.jpg)

### 3.4 薄透镜近似

真实镜头非常复杂，而且有像差（球面透镜不能把光线汇聚到一点）。**理想薄透镜**：

1. 所有平行入射的光线经过透镜后穿过**焦点**
2. 所有穿过焦点的光线经过透镜后平行
3. 焦距可以任意改变（现实中变焦镜头确实如此）

![Ideal thin lens](/images/blog/Course_notes/Computer_Graphics/Games101/note5/thin-lens-focal-point.jpg)

**薄透镜方程**（Gaussian thin lens equation）：物距 $z_o$、像距 $z_i$、焦距 $f$，

$$\frac{1}{f} = \frac{1}{z_i} + \frac{1}{z_o}$$

用 **Gauss 光线作图**推导：平行光线、主光线（过中心）、焦点光线，由相似三角形
$\frac{h_o}{h_i} = \frac{z_o - f}{f} = \frac{f}{z_i - f}$，得
$(z_o - f)(z_i - f) = f^2$（Newtonian form），化简即得上式。

![Thin lens equation](/images/blog/Course_notes/Computer_Graphics/Games101/note5/thin-lens-equation.jpg)
![Gauss ray construction](/images/blog/Course_notes/Computer_Graphics/Games101/note5/gauss-construction.jpg)

### 3.5 Defocus blur 与 depth of field

**Circle of Confusion（CoC）**：不在对焦平面上的物体，其像不在传感器平面上，在传感器
上形成一个圆斑。由相似三角形：

$$\frac{C}{A} = \frac{|z_s - z_i|}{z_i}$$

$A$ 是光圈直径，$z_s$ 是该物体的像距，$z_i$ 是传感器位置。**CoC 正比于光圈大小**。

![Circle of confusion](/images/blog/Course_notes/Computer_Graphics/Games101/note5/circle-of-confusion.jpg)

**F-number 的正式定义**：焦距除以光圈直径，$N = f / D$。常见 f-stop：1.4, 2, 2.8, 4,
5.6, 8, 11, 16, 22, 32（每档面积减半）。f/2 的写法反映了 $A = f / N$。于是

$$C = A \frac{|z_s - z_i|}{z_i} = \frac{f}{N} \frac{|z_s - z_i|}{z_i}$$

**CoC 与 F-number 成反比**：光圈越大（N 越小），背景越虚。

![F-number definition](/images/blog/Course_notes/Computer_Graphics/Games101/note5/f-number-definition.jpg)
![CoC vs F-stop](/images/blog/Course_notes/Computer_Graphics/Games101/note5/coc-vs-fstop.jpg)

**用光线追踪模拟薄透镜**：

- 设置：选传感器大小、焦距、光圈；选感兴趣的物距 $z_o$；由薄透镜方程算出传感器
  位置 $z_i$
- 渲染：对传感器上每个像素 $x'$，在透镜平面上随机采样点 $x''$；$x'$ 经过透镜中心的
  虚拟光线交对焦平面于 $x'''$，穿过透镜的光线一定也打到 $x'''$（因为它在焦平面上）；
  估计光线 $x'' \to x'''$ 的 radiance

![Ray tracing for defocus blur](/images/blog/Course_notes/Computer_Graphics/Games101/note5/ray-tracing-thin-lens.jpg)

**Depth of field**：图像中以可接受的清晰度呈现的物体深度范围。把 CoC 设为最终观看
条件下看起来仍清晰的最大允许模糊斑，对应的场景深度范围就是景深。光圈越小、焦距
越短、对焦距离越远，景深越大。

$$DOF = D_F - D_N, \qquad D_F = \frac{D_S f^2}{f^2 - N C (D_S - f)}, \qquad D_N = \frac{D_S f^2}{f^2 + N C (D_S - f)}$$

![Depth of field](/images/blog/Course_notes/Computer_Graphics/Games101/note5/depth-of-field.jpg)
![Depth of field formulas](/images/blog/Course_notes/Computer_Graphics/Games101/note5/dof-formulas.jpg)

## 4. Light field / lumigraph

### 4.1 Plenoptic function

我们能看到的所有东西的集合是什么？**全光函数**（Adelson & Bergen）。从一个静止的人
开始参数化他能看到的一切：

- 灰度快照 $P(\theta, \phi)$：从一个视点、一个时刻、在可见光谱上平均的光强
- 彩色快照 $P(\theta, \phi, \lambda)$
- 电影 $P(\theta, \phi, \lambda, t)$
- 全息电影 $P(\theta, \phi, \lambda, t, V_x, V_y, V_z)$：任意视点、任意时刻、任意波长

这个 7D 函数能重建任何视角、任何时刻、任何位置、任何波长的视图，**包含了每一张
照片、每一部电影、任何人看到过的一切**，完整捕获了我们的视觉现实。

![The plenoptic function](/images/blog/Course_notes/Computer_Graphics/Games101/note5/plenoptic-function.jpg)
![Plenoptic function summary](/images/blog/Course_notes/Computer_Graphics/Games101/note5/plenoptic-summary.jpg)

### 4.2 光场

先不管时间和颜色：5D $P(\theta, \phi, V_x, V_y, V_z)$，3D 位置 + 2D 方向，就是一条
**光线**。假设光在真空中沿直线不变（non-dispersive medium），一条光线上各点等价，
降到 **4D**：2D 位置 + 2D 方向。

![Ray reuse](/images/blog/Course_notes/Computer_Graphics/Games101/note5/ray-reuse.jpg)

只需要一个包住物体的凸曲面（plenoptic surface）：记录穿过这个面的每条光线的
radiance，就能合成任何外部视角。这就是 **lumigraph / light field**。

**两平面参数化**：光线由两个平面上的交点 $(s, t)$ 和 $(u, v)$ 定义。固定 $(s, t)$
让 $(u, v)$ 变化就是一张图像（从 $(s, t)$ 看过去的图）；固定 $(u, v)$ 让 $(s, t)$
变化是从不同位置看同一个点。

![Two-plane parameterization](/images/blog/Course_notes/Computer_Graphics/Games101/note5/two-plane.jpg)
![An image in the lumigraph](/images/blog/Course_notes/Computer_Graphics/Games101/note5/lumigraph-image.jpg)

采集：Stanford camera array；**integral imaging**（Lippmann 1908，苍蝇眼的 lenslet
阵列）在空间分辨率和角度分辨率之间做固定的权衡。

![Integral imaging](/images/blog/Course_notes/Computer_Graphics/Games101/note5/integral-imaging.jpg)

### 4.3 光场相机

Lytro（Ren Ng 创立）用微透镜设计。**每个像素（irradiance）现在存成一块像素
（radiance）**：一个微透镜把来自不同方向的光分开记录在它后面的一块传感器上。

![Light field camera](/images/blog/Course_notes/Computer_Graphics/Games101/note5/light-field-camera.jpg)

怎么得到"普通"照片？总是取每块的底部像素，就是从一个方向看；取中间的、顶部的，
本质上是**在移动相机**。**Computational refocusing** 同理：虚拟地改变焦距，按新的
对焦选择相应的光线方向重新积分。这些都能做是因为**光场包含了一切**。

问题：**空间分辨率不足**（同一块传感器要同时记录空间和方向信息）、**成本高**
（微透镜设计复杂）。图形学就是取舍。

![Refocusing](/images/blog/Course_notes/Computer_Graphics/Games101/note5/refocusing.jpg)
![Problems of light field cameras](/images/blog/Course_notes/Computer_Graphics/Games101/note5/light-field-problems.jpg)

## 5. Color and perception

### 5.1 光的物理基础

牛顿用棱镜把阳光分成彩虹，分出来的光不能再被第二个棱镜细分。可见光是波长约 400
到 700 nm 的电磁辐射。**Spectral Power Distribution（SPD，谱功率密度）**：各波长
的光的量，单位是辐射量 / nm，常用相对单位。日光的 SPD 随时间变化（蓝天、日盘不同）。
SPD 是**线性**的：两个光源叠加，SPD 相加。

![Spectral power distribution](/images/blog/Course_notes/Computer_Graphics/Games101/note5/spd.jpg)

**什么是颜色？** 颜色是**人的感知现象**，不是光的普遍属性。不同波长的光本身不是
"颜色"。

![What is color](/images/blog/Course_notes/Computer_Graphics/Games101/note5/what-is-color.jpg)

### 5.2 生物基础

视网膜上的感光细胞：**Rods**（视杆细胞）约 1.2 亿个，弱光（scotopic）下的主要
感受器，只感知灰度；**Cones**（视锥细胞）约 600 到 700 万个，正常光照（photopic）
下工作，**三种类型**有不同的光谱敏感度，提供颜色感觉。

![Rods and cones](/images/blog/Course_notes/Computer_Graphics/Games101/note5/rods-cones.jpg)

三种视锥 **S、M、L** 分别在短、中、长波长有峰值响应。三种细胞的比例因人而异，差别
很大。

![Spectral response of cone cells](/images/blog/Course_notes/Computer_Graphics/Games101/note5/cone-response.jpg)

### 5.3 三刺激理论与同色异谱

人眼**不测量每个波长**，大脑收到的只有三个响应值 $(S, M, L)$：

$$S = \int r_S(\lambda) s(\lambda) \, d\lambda, \quad M = \int r_M(\lambda) s(\lambda) \, d\lambda, \quad L = \int r_L(\lambda) s(\lambda) \, d\lambda$$

**Metamers（同色异谱）**：两个不同的光谱（无穷维）投影到同一个 $(S, M, L)$（三维）
响应，对人来说颜色一样。这对**颜色再现至关重要**：显示器不需要重现真实场景的完整
光谱，只用三种颜色的像素就能让你感知到同样的颜色。

![Metamers](/images/blog/Course_notes/Computer_Graphics/Games101/note5/metamers.jpg)

### 5.4 颜色匹配

**Additive color**：给定一组原色光 $s_R(\lambda), s_G(\lambda), s_B(\lambda)$，调整
亮度相加 $R s_R + G s_G + B s_B$，颜色就由标量 $(R, G, B)$ 描述。

![Additive color](/images/blog/Course_notes/Computer_Graphics/Games101/note5/additive-color.jpg)

**颜色匹配实验**：调三种原色的量去匹配一个测试颜色。有时怎么调都匹配不上，就把某
原色**加到测试颜色那一边**，相当于用了**负的量**。

![Negative amount of a primary](/images/blog/Course_notes/Computer_Graphics/Games101/note5/negative-primary.jpg)

**CIE RGB**：原色和测试光都是单色光。**颜色匹配函数** $\bar{r}(\lambda), \bar{g}(\lambda), \bar{b}(\lambda)$
记录匹配每个波长的单色光需要多少原色（注意不是响应曲线也不是光谱，而且 $\bar{r}$
有负值）。对任意光谱 $s$：

$$R_{CIE\,RGB} = \int s(\lambda) \bar{r}(\lambda) \, d\lambda, \quad
G_{CIE\,RGB} = \int s(\lambda) \bar{g}(\lambda) \, d\lambda, \quad
B_{CIE\,RGB} = \int s(\lambda) \bar{b}(\lambda) \, d\lambda$$

![CIE RGB color matching functions](/images/blog/Course_notes/Computer_Graphics/Games101/note5/cie-rgb-matching.jpg)
![Color reproduction with matching functions](/images/blog/Course_notes/Computer_Graphics/Games101/note5/color-reproduction.jpg)

### 5.5 颜色空间

**sRGB**：把某个特定显示器的 RGB 标准化，其他设备通过校准模拟它。广泛使用，
但 **gamut（色域）**有限。

**CIE XYZ**：一组**假想的**标准原色 X、Y、Z，对应的匹配函数**严格为正**，能张成
所有可见颜色，$Y$ 是 luminance（与颜色无关的亮度）。

![CIE XYZ](/images/blog/Course_notes/Computer_Graphics/Games101/note5/cie-xyz.jpg)

分离亮度和色度：

$$x = \frac{X}{X + Y + Z}, \quad y = \frac{Y}{X + Y + Z}, \quad z = \frac{Z}{X + Y + Z}, \quad x + y + z = 1$$

只需记录 $(x, y)$，加上亮度 $Y$。**CIE chromaticity diagram**：边界曲线是
**spectral locus**（纯单色光），内部是混合的、不那么纯的颜色，白色在中心 $(1/3, 1/3)$。

![Luminance and chromaticity](/images/blog/Course_notes/Computer_Graphics/Games101/note5/luminance-chromaticity.jpg)
![CIE chromaticity diagram](/images/blog/Course_notes/Computer_Graphics/Games101/note5/chromaticity-diagram.jpg)

**Gamut** 是一组原色能生成的色度集合，即在色度图上覆盖的区域。不同颜色空间色域
不同，sRGB 只覆盖一小块。

![Gamut](/images/blog/Course_notes/Computer_Graphics/Games101/note5/gamut.jpg)

**感知组织的颜色空间**：

- **HSV**（Hue-Saturation-Value）：轴对应艺术上的颜色特征，取色器里常用。
  Hue 是颜色的"种类"（主波长），saturation 是"彩度"（纯度），lightness / value
  是整体亮度
- **CIELAB（L\*a\*b\*）**：追求感知均匀。$L^*$ 是亮度，$a^*$ 是红绿对立轴，
  $b^*$ 是蓝黄对立轴。**Opponent color theory** 有神经学基础：大脑早期用白黑、
  红绿、黄蓝三个轴编码颜色。证据：可以有浅绿、深绿、黄绿、蓝绿，但没有"红绿"；
  以及 afterimage（残像）现象

![HSV](/images/blog/Course_notes/Computer_Graphics/Games101/note5/hsv.jpg)
![CIELAB](/images/blog/Course_notes/Computer_Graphics/Games101/note5/cielab.jpg)

**一切都是相对的**：颜色感知会适应，同样的颜色在不同背景下看起来不同（各种错觉）。

**CMYK**：**减色**模型，混得越多越暗。Cyan、Magenta、Yellow 加 Key（黑），印刷
常用。为什么 CMY 混出黑还要 K？因为黑墨更便宜。

![CMYK](/images/blog/Course_notes/Computer_Graphics/Games101/note5/cmyk.jpg)
