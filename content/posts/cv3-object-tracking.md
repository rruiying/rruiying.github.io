---
title: "Computer Vision III — Notes 2: Object Tracking"
date: 2026-08-23
categories: [Course notes/Computer Vision/Computer Vision III]
tags: [Deep learning, Computer Vision, TUM, object tracking, MOT]
summary: Second notes for CV3 at TUM — object tracking from Bayesian filtering to single-object online trackers (GOTURN, MDNet), multi-object association via motion models, Tracktor and ReID, graph-based MOT with message passing networks, and how tracking is evaluated.
---
As in the detection notes, we start from the motivation. We first frame
tracking probabilistically — **Bayesian tracking** and **graphical models** —
then look at single-object **online tracking** with GOTURN and MDNet. Moving to
multiple objects, tracking becomes an association problem: the first approach
combines a **motion model** with **bipartite matching**; the second,
**Tracktor**, turns the detector itself into a tracker; **metric learning**
and **re-identification (ReID)** supply the appearance cues. We then move to
**graph-based MOT** — cost-flow networks, learning the costs, and full graph
optimization — culminating in **message passing networks** and MOT with MPNs.
We close with how multi-object tracking is **evaluated**.

## 1. Motivation

1. 建造一个世界的**动态模型**， 理解where/what objects 和 object 如何 move
2. **预测**未来的运动（e.g. 行人是否穿越马路）
3. 仅凭外观无法辨别物体时辅助物体检测（to facilitate object detection when
appearance alone is insufficient）

我们的目标是在给出 t帧时的observation, 建立目标检测与时序关联模型，找到他们与在t + n帧的观测结果的对应关系

挑战有： Fast motions Changing appearance
•Changing object pose
•Dynamic backgrounds
•Occlusions
•Poor image quality 等等

一个简单的想法是 tracking by detection, detect the object in every frame 例如 Eigenfaces(el. turk, matthew A,.... "face recognition using eigenfaces")。 如果场景中只有一个人，且该人始终保持正面朝向，我们可以采用基于滑动窗口的模板匹配方法（例如使用人脸的 PCA 模型，但我们不希望搜索整幅图像，因为这种做法效率低下，所以可以利用上一帧中的位置信息来限定搜索范围。
但是问题是Many objects may be present, and the detector may misfire 我们不知道哪一个detection是哪一个
idea: We could use the motion prior.


## 2. Bayesian tracking and graphical models

直觉是 Illustration
23
[Michael Black]
Goal: Estimate car position at each time instant (say, of the white car).
Observations: Image sequence and known background.
Background extraction
24
• Idea: Background is almost constant across all frames.
Hence, it is well represented by a low-rank approx
Example:
• Video Resolution: 60 pixels by 80 pixels, 113 seconds, 10fps
• We unroll frames, resulting in vectors of size 1 x 4800
• We stack them into a matrix of size 11300 x 4800
Video Matrix
SVD
Low Rank
Illustration
25
[Michael Black]
•Perform background subtraction.
•Obtain binary map of possible cars.
•But which one is the one we want to track?
Bayesian tracking
26
[Michael Black]
observations: images
system state: car position (𝑥, 𝑦)
Likelihood:
Is there a car?
Prior:
Where was the car in
the previous frame?
Posterior:
Bayesian update
Notation
27
[Michael Black]
• 𝑥𝑘 ∈ ℝ𝑛: internal state at 𝑘-th frame
(hidden random variable, e.g., position of the object in the image).
𝑋𝑘 = [𝑥1, 𝑥2, . . . , 𝑥𝑘]𝑇 history up to time step 𝑘.
• 𝑧𝑘 ∈ ℝ𝑚: measurement at 𝑘-th frame
(observable random variable, e.g. the given image).
𝑍𝑘 = [𝑧1, 𝑧2, . . . , 𝑧𝑘]𝑇 history up to time step 𝑘

我们的目标是 Goal
28
[Michael Black]
Estimating posterior probability 𝑝(𝑥𝑘 ∣ 𝑍𝑘)
How?
One idea: recursion
𝑝(𝑥𝑘−1 ∣ 𝑍𝑘−1) → 𝑝(𝑥𝑘 ∣ 𝑍𝑘)
How to realise recursion?
What assumptions are necessary?

因此我们建立贝叶斯图模型， 
*(recursive Bayesian filtering, prediction / update, Kalman filter, graphical-model view …)*

## 3. Single-object online tracking

### 3.1 GOTURN

*(…)*

### 3.2 MDNet

*(…)*

## 4. Multi-object tracking as association

*(what breaks when there are many objects: data association, occlusion, identity switches …)*

### 4.1 Motion models and bipartite matching

*(Hungarian algorithm …)*

### 4.2 Tracktor

*(…)*

### 4.3 Metric learning and ReID

*(…)*

## 5. Graph-based MOT

### 5.1 Cost-flow networks

*(min-cost flow formulation …)*

### 5.2 Learning the costs

*(…)*

### 5.3 Graph optimization

*(…)*

### 5.4 Message passing networks

*(…)*

### 5.5 MOT with message passing networks

*(…)*

## 6. MOT evaluation

*(MOTA, IDF1, HOTA, identity switches …)*
