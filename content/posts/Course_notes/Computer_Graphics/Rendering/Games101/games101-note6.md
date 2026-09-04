---
title: "GAMES101 — Modern Computer Graphics Note6: Animation and Simulation"
date: 2026-09-03
tags: [Computer Graphics, animation, physics simulation, GAMES101]
summary: GAMES101 note 6 (Lectures 21–22) — keyframe animation, mass-spring systems and particle systems, forward and inverse kinematics, rigging and motion capture, then single-particle simulation with explicit, midpoint, implicit and Verlet integrators, and a glimpse of rigid-body and fluid simulation (PBF, Eulerian vs Lagrangian, MPM).
---

> 对应 Lecture 21–22，课程的最后一块。物理仿真的深入内容在 GAMES103 的笔记里。

## 1. 什么是动画

**Animation：bring things to life.** 它是一种沟通工具，美学问题常常比技术问题更重要。
从建模的角度看，动画是**建模的扩展：把场景模型表示成时间的函数**。输出是一串图像，
连续播放时产生运动感：电影 24 fps，视频一般 30 fps，VR 90 fps。

历史：伊朗 3200 BCE 的陶碗上的连续画、1831 年的 phenakistoscope、1878 年 Muybridge
的奔马照片（第一部"电影"，原本是科学工具）、1937 年迪士尼《白雪公主》（第一部手绘
长片）、1963 年 Sutherland 的 Sketchpad（第一个数字计算机生成的动画）、1972 年
Catmull 的 Computer Animated Faces、1993 年《侏罗纪公园》、1995 年《玩具总动员》
（第一部 CG 长片）。

## 2. Keyframe animation

动画师（lead animator）画**关键帧**，助手（人或计算机）画中间帧，叫 **tweening**。
把每一帧看作一个参数向量，对每个参数在关键帧之间插值。线性插值通常不够好，回忆
样条：用样条做光滑、可控的插值。

![Keyframe animation](/images/blog/Course_notes/Computer_Graphics/Games101/note6/keyframe-animation.jpg)
![Keyframe interpolation](/images/blog/Course_notes/Computer_Graphics/Games101/note6/keyframe-interpolation.jpg)

## 3. 物理仿真

牛顿定律 $F = ma$。**Physically based animation** 用数值仿真生成物体的运动，比如
抛体运动、布料、流体。

![Newton's law](/images/blog/Course_notes/Computer_Graphics/Games101/note6/newtons-law.jpg)

### 3.1 Mass-spring system

绳子、头发、布都可以用质点弹簧系统建模。

**理想弹簧**：两个质点 $a, b$，

$$\mathbf{f}_{a \to b} = k_s (\mathbf{b} - \mathbf{a}), \qquad \mathbf{f}_{b \to a} = -\mathbf{f}_{a \to b}$$

力把两点拉到一起，大小正比于位移（Hooke's law），$k_s$ 是弹簧系数（stiffness）。
问题：这个弹簧想要长度为零。

![A simple spring](/images/blog/Course_notes/Computer_Graphics/Games101/note6/simple-spring.jpg)

**非零 rest length**：

$$\mathbf{f}_{a \to b} = k_s \frac{\mathbf{b} - \mathbf{a}}{\|\mathbf{b} - \mathbf{a}\|} (\|\mathbf{b} - \mathbf{a}\| - l)$$

$l$ 是 rest length。问题：会永远振荡。

![Non-zero length spring](/images/blog/Course_notes/Computer_Graphics/Games101/note6/nonzero-length-spring.jpg)

**引入能量损失**。记号：$\dot{\mathbf{x}}$ 是速度，$\ddot{\mathbf{x}}$ 是加速度。
简单的运动阻尼 $\mathbf{f} = -k_d \dot{\mathbf{b}}$，像粘性阻力，减慢速度方向的运动，
$k_d$ 是阻尼系数。问题：**减慢所有运动**，生锈的弹簧振荡应该变慢，但它掉到地上也
应该变慢吗？

![Motion damping](/images/blog/Course_notes/Computer_Graphics/Games101/note6/motion-damping.jpg)

**弹簧的内部阻尼**：只阻尼内部的、弹簧驱动的运动。

$$\mathbf{f}_b = -k_d \frac{\mathbf{b} - \mathbf{a}}{\|\mathbf{b} - \mathbf{a}\|} \cdot (\dot{\mathbf{b}} - \dot{\mathbf{a}}) \cdot \frac{\mathbf{b} - \mathbf{a}}{\|\mathbf{b} - \mathbf{a}\|}$$

即把相对速度投影到 $a$ 到 $b$ 的方向上（标量），再乘以这个方向。这样只对**弹簧长度
的变化**施加粘性阻力，不会减慢整个系统的运动（整体平移或旋转）。这只是一种特定
类型的阻尼。

![Internal damping for spring](/images/blog/Course_notes/Computer_Graphics/Games101/note6/internal-damping.jpg)

**用弹簧搭结构**：行为由连接方式决定。只连相邻点的网格不抗**剪切**（shearing）也不抗
**面外弯曲**；加对角线抗剪切但有各向异性偏差；两条对角线都加则偏差小，仍不抗弯曲；
再加**跳一个点的弹簧**（红色，应该弱得多）才能抗面外弯曲。它们的行为就是它们看起来
的样子。更精确的方法是 FEM（有限元）。

![Structures from springs](/images/blog/Course_notes/Computer_Graphics/Games101/note6/structures-from-springs.jpg)

### 3.2 Particle systems

把动态系统建模为**大量粒子的集合**，每个粒子的运动由一组物理（或非物理）的力定义。
图形学和游戏里的流行技术：容易理解和实现；可扩展，粒子少则快，多则复杂。挑战：
可能需要很多粒子（流体）；可能需要加速结构（找最近的粒子）。

每一帧：

1. [如需要] 创建新粒子
2. 计算每个粒子受的力
3. 更新每个粒子的位置和速度
4. [如需要] 移除死掉的粒子
5. 渲染粒子

![Particle systems](/images/blog/Course_notes/Computer_Graphics/Games101/note6/particle-systems.jpg)
![Particle system animation loop](/images/blog/Course_notes/Computer_Graphics/Games101/note6/particle-animation-loop.jpg)

力的种类：吸引和排斥（引力、电磁力、弹簧、推进……）、阻尼（摩擦、空气阻力、
粘性……）、碰撞（墙、容器、固定物体、动态物体、角色身体……）。万有引力
$F_g = G m_1 m_2 / d^2$ 可以做星系仿真。

**Flocking**（Craig Reynolds 的 boids）：每只鸟是一个粒子，受三个简单的力：向邻居
中心的吸引、对个别邻居的排斥、与邻居平均轨迹的对齐。数值仿真大量粒子，**涌现出
复杂行为**（鱼群、蜂群也一样）。其他例子：分子动力学、人群。

![Simulated flocking](/images/blog/Course_notes/Computer_Graphics/Games101/note6/flocking.jpg)

## 4. Kinematics

### 4.1 Forward kinematics

**Articulated skeleton**：拓扑（什么连着什么）、关节的几何关系、树结构。关节类型：
pin（1D 旋转）、ball（2D 旋转）、prismatic（平移）。

![Forward kinematics](/images/blog/Course_notes/Computer_Graphics/Games101/note6/forward-kinematics.jpg)

例子：2D 的两段手臂。动画师给出角度 $\theta_1, \theta_2$，计算机算出末端执行器
（end effector）的位置 $p$：

$$p = (l_1 \cos\theta_1 + l_2 \cos(\theta_1 + \theta_2),\ l_1 \sin\theta_1 + l_2 \sin(\theta_1 + \theta_2))$$

动画就是角度参数随时间的函数。

![FK: angles as functions of time](/images/blog/Course_notes/Computer_Graphics/Games101/note6/fk-angles-over-time.jpg)

优点：直接控制方便，实现简单。缺点：动画可能不符合物理，艺术家很耗时。

### 4.2 Inverse kinematics

反过来：动画师给出末端执行器的位置，计算机求满足约束的关节角度。两段手臂可以解析
求解，但一般情况很难：

- **配置空间里有多个解**（可能分离也可能连通）
- **解可能不存在**（够不到）

![Inverse kinematics](/images/blog/Course_notes/Computer_Graphics/Games101/note6/inverse-kinematics.jpg)
![IK: multiple solutions](/images/blog/Course_notes/Computer_Graphics/Games101/note6/ik-multiple-solutions.jpg)

一般 N 连杆 IK 的**数值解**：选一个初始配置；定义误差度量（比如目标与当前位置的
距离平方）；计算误差关于配置的梯度；梯度下降（或牛顿法等优化方法）。
Style-based IK（Grochow et al.）从数据里学习姿态的先验。

![Numerical IK](/images/blog/Course_notes/Computer_Graphics/Games101/note6/numerical-ik.jpg)

## 5. Rigging 与 motion capture

**Rigging**：角色上的一组高层控制，可以更快、更直观地修改姿态、形变、表情等。像
木偶上的线，捕获角色所有有意义的变化，每个角色都不同。创建代价高：手工，需要艺术
和技术双重训练。

![Rigging](/images/blog/Course_notes/Computer_Graphics/Games101/note6/rigging.jpg)

**Blend shapes**：不用骨架，直接在曲面之间插值。比如建一组面部表情，最简单的方案是
顶点位置的线性组合，用样条控制权重随时间的变化。

![Blend shapes](/images/blog/Course_notes/Computer_Graphics/Games101/note6/blend-shapes.jpg)

**Motion capture**：数据驱动的动画方法。记录真实世界的表演，从数据中提取姿态随
时间的函数。优点：能快速捕获大量真实数据，真实感高。缺点：设备复杂昂贵；捕获的
动画可能不符合艺术需求，需要修改。设备：**optical**（反光标记点 + 红外相机三角化，
8 台以上、240 Hz，遮挡是难点）、**magnetic**（感应磁场推断位置和方向，有线）、
**mechanical**（直接测关节角，限制运动）。

![Motion capture](/images/blog/Course_notes/Computer_Graphics/Games101/note6/motion-capture.jpg)
![Motion capture equipment](/images/blog/Course_notes/Computer_Graphics/Games101/note6/mocap-equipment.jpg)
![Optical motion capture](/images/blog/Course_notes/Computer_Graphics/Games101/note6/optical-mocap.jpg)

面部动画的挑战：**uncanny valley（恐怖谷）**。人工角色越接近真人，情绪反应反而变
负面，直到真实到足够令人信服为止。

![Uncanny valley](/images/blog/Course_notes/Computer_Graphics/Games101/note6/uncanny-valley.jpg)

**Production pipeline**：story → storyboard → modeling → rigging → layout →
animation → simulation → lighting → rendering → compositing。

![The production pipeline](/images/blog/Course_notes/Computer_Graphics/Games101/note6/production-pipeline.jpg)

## 6. 单粒子仿真

### 6.1 ODE 与 Euler 方法

先研究单个粒子。假设粒子的运动由一个**速度矢量场** $\mathbf{v}(\mathbf{x}, t)$
决定，随时间计算位置需要解一阶常微分方程（ODE）：

$$\frac{d\mathbf{x}}{dt} = \dot{\mathbf{x}} = \mathbf{v}(\mathbf{x}, t)$$

"一阶"指只取一阶导数，"常"指没有偏导数。给定初始位置 $\mathbf{x}_0$，用前向数值
积分求解，即**初值问题**：从起点出发，跟着矢量走。

![ODE](/images/blog/Course_notes/Computer_Graphics/Games101/note6/ode.jpg)

**Euler's method**（forward / explicit Euler）：

$$\mathbf{x}^{t + \Delta t} = \mathbf{x}^t + \Delta t \, \dot{\mathbf{x}}^t, \qquad
\dot{\mathbf{x}}^{t + \Delta t} = \dot{\mathbf{x}}^t + \Delta t \, \ddot{\mathbf{x}}^t$$

简单、迭代、常用，但**很不准确**，而且**经常不稳定**。

![Euler's method](/images/blog/Course_notes/Computer_Graphics/Games101/note6/euler-method.jpg)

数值积分的误差会**累积**，Euler 特别差：步长越大偏离真实路径越远。在一个圆形速度场
里，每一步都沿切线走，会不断向外螺旋。

![Euler errors](/images/blog/Course_notes/Computer_Graphics/Games101/note6/euler-errors.jpg)
![Instability of the Euler method](/images/blog/Course_notes/Computer_Graphics/Games101/note6/euler-instability.jpg)

两个问题：**误差**（每步的误差累积，精度随仿真进行而下降，但图形学里精度可能不
关键）和**不稳定性**（误差复合导致仿真发散，即使底层系统不发散。稳定性是仿真的
根本问题，不能忽视，比如游戏里车子突然飞上天）。

![Errors and instability](/images/blog/Course_notes/Computer_Graphics/Games101/note6/errors-and-instability.jpg)

### 6.2 对抗不稳定性

- **Midpoint method / modified Euler**：平均起点和终点的速度
- **Adaptive step size**：比较一步和两个半步，递归直到误差可接受
- **Implicit methods**：用下一时刻的速度（难）
- **Position-based / Verlet integration**：时间步之后约束粒子的位置和速度

![Combating instability](/images/blog/Course_notes/Computer_Graphics/Games101/note6/combating-instability.jpg)

**Midpoint method**：先做一个 Euler 步（a），在这一步的中点算导数（b），用中点的
导数更新位置（c）：

$$\mathbf{x}_{mid} = \mathbf{x}(t) + \frac{\Delta t}{2} \mathbf{v}(\mathbf{x}(t), t), \qquad
\mathbf{x}(t + \Delta t) = \mathbf{x}(t) + \Delta t \, \mathbf{v}(\mathbf{x}_{mid}, t)$$

![Midpoint method](/images/blog/Course_notes/Computer_Graphics/Games101/note6/midpoint-method.jpg)

**Modified Euler**：平均起点和终点的速度，结果更好：

$$\dot{\mathbf{x}}^{t + \Delta t} = \dot{\mathbf{x}}^t + \Delta t \, \ddot{\mathbf{x}}^t, \qquad
\mathbf{x}^{t + \Delta t} = \mathbf{x}^t + \frac{\Delta t}{2}(\dot{\mathbf{x}}^t + \dot{\mathbf{x}}^{t + \Delta t})
= \mathbf{x}^t + \Delta t \, \dot{\mathbf{x}}^t + \frac{(\Delta t)^2}{2} \ddot{\mathbf{x}}^t$$

![Modified Euler](/images/blog/Course_notes/Computer_Graphics/Games101/note6/modified-euler.jpg)

**Adaptive step size**：基于误差估计选步长。重复直到误差低于阈值：算一个步长 $T$ 的
Euler 步得 $\mathbf{x}_T$，算两个 $T/2$ 的步得 $\mathbf{x}_{T/2}$，误差是
$\|\mathbf{x}_T - \mathbf{x}_{T/2}\|$，太大就减小步长重试。很实用，但可能需要非常小
的步。

![Adaptive step size](/images/blog/Course_notes/Computer_Graphics/Games101/note6/adaptive-step-size.jpg)

**Implicit Euler**（backward method）：用**未来**的导数更新当前步：

$$\mathbf{x}^{t + \Delta t} = \mathbf{x}^t + \Delta t \, \dot{\mathbf{x}}^{t + \Delta t}, \qquad
\dot{\mathbf{x}}^{t + \Delta t} = \dot{\mathbf{x}}^t + \Delta t \, \ddot{\mathbf{x}}^{t + \Delta t}$$

要对 $\mathbf{x}^{t + \Delta t}$ 和 $\dot{\mathbf{x}}^{t + \Delta t}$ 解非线性方程，用
求根算法（牛顿法）。**稳定性好得多**。

![Implicit Euler](/images/blog/Course_notes/Computer_Graphics/Games101/note6/implicit-euler.jpg)

**怎么量化"稳定性"？** 用 local truncation error（每步）和 total accumulated error
（整体），看的不是绝对值而是关于步长 $h = \Delta t$ 的阶。Implicit Euler 是一阶的：
局部截断误差 $O(h^2)$，全局截断误差 $O(h)$，即步长减半，误差也减半。

![Stability and order](/images/blog/Course_notes/Computer_Graphics/Games101/note6/stability-order.jpg)

**Runge-Kutta 家族**：解 ODE 的一族高级方法，特别擅长非线性，四阶版本 **RK4** 最
常用：

$$\mathbf{y}_{n + 1} = \mathbf{y}_n + \frac{h}{6}(k_1 + 2k_2 + 2k_3 + k_4), \quad
k_1 = f(t_n, \mathbf{y}_n), \ k_2 = f\left(t_n + \tfrac{h}{2}, \mathbf{y}_n + \tfrac{h}{2} k_1\right), \
k_3 = f\left(t_n + \tfrac{h}{2}, \mathbf{y}_n + \tfrac{h}{2} k_2\right), \ k_4 = f(t_n + h, \mathbf{y}_n + h k_3)$$

![Runge-Kutta](/images/blog/Course_notes/Computer_Graphics/Games101/note6/runge-kutta.jpg)

**Position-based / Verlet integration**：modified Euler 前向步之后，**约束粒子的位置**
防止发散和不稳定，再用约束后的位置算速度。两者都会耗散能量、趋于稳定。快而简单，
但不是物理的，耗散能量（误差）。详见作业 8。

![Verlet integration](/images/blog/Course_notes/Computer_Graphics/Games101/note6/verlet.jpg)

## 7. 刚体与流体仿真

**Rigid body simulation**：简单情况和模拟粒子类似，只是多考虑几个属性（位置、
旋转、线速度、角速度……）。

**流体：一个简单的 position-based 方法**（Position Based Fluids）。关键思想：假设
水由小的刚体球组成；假设水不可压缩（密度恒定）；所以只要某处密度变了，就通过改变
粒子位置来"修正"。需要知道密度关于每个粒子位置的梯度，更新就是梯度下降。

![Position based fluids](/images/blog/Course_notes/Computer_Graphics/Games101/note6/position-based-fluids.jpg)

**Eulerian vs Lagrangian**：模拟大量物质的两种视角。**Lagrangian**（质点法）跟着
每个粒子走；**Eulerian**（网格法）在固定的网格上看物质流过。

![Eulerian vs Lagrangian](/images/blog/Course_notes/Computer_Graphics/Games101/note6/eulerian-vs-lagrangian.jpg)

**Material Point Method（MPM）**：混合两种视角。Lagrangian 的粒子携带材质属性；
Eulerian 的网格做数值积分。粒子把属性传给网格，网格做更新，再插值回粒子。
《冰雪奇缘》的雪就是用 MPM 做的。

![Material point method](/images/blog/Course_notes/Computer_Graphics/Games101/note6/mpm.jpg)

## 8. 课程结束

四大块：rasterization、geometry、light transport、animation / simulation。后续课程：
GAMES201（胡渊鸣，高级物理引擎实战，Taichi）、GAMES202（实时高质量渲染）、
高级图像合成（advanced light transport、appearance modeling）。
