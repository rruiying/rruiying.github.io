---
title: "Computer Vision III (IN2375) — Notes 3: Image Segmentation"
date: 2026-08-23
tags: [Deep learning, Computer Vision, TUM, segmentation, video object segmentation]
summary: Third notes for CV3 (IN2375, Detection, Segmentation and Tracking) at TUM — segmentation from superpixels, clustering and normalized cuts through CRFs to FCNs, instance segmentation with Mask R-CNN and PointRend, panoptic segmentation and its evaluation, and video object segmentation from optical flow to pixel-wise retrieval.
---
> 本文笔记中英混杂，因为期末考试纯英文方便记忆，解释部分用中文方便理解

We again build up from classical methods to deep learning. Segmentation begins
as a grouping problem: **superpixels** and **clustering** — k-means, spectral
clustering, mean shift — leading to **normalized cuts**, and then to
**energy-based models** such as CRFs. Deep learning enters with the **fully
convolutional network (FCN)** and **1×1 convolutions**; the difficulties of
dense pixel-level prediction are addressed by **dilated convolutions** and
**upsampling**. From semantic segmentation we move to **instance
segmentation** with **Mask R-CNN**, whose RoI rounding problem motivates
coordinate-based networks such as **PointRend**. **Panoptic segmentation**
unifies the two tasks — Panoptic FPN, Panoptic FCN, and how panoptic quality
is evaluated. Finally, **video object segmentation (VOS)**: one-shot
(semi-supervised) versus zero-shot (unsupervised) settings; motion-based VOS
built on **optical flow** (FlowNet, correlation layers, SegFlow);
appearance-only VOS (OSVOS, OnAVOS, MaskTrack); and metric-based approaches
that cast VOS as **pixel-wise retrieval**.

## 1. Segmentation as grouping

### 1.1 Superpixels

*(…)*

### 1.2 Clustering: k-means, spectral clustering, mean shift

*(…)*

### 1.3 Normalized cuts

*(…)*

## 2. Energy-based models: CRFs

*(…)*

## 3. Deep semantic segmentation

### 3.1 FCN and 1×1 convolutions

*(…)*

### 3.2 The trouble with pixel-level prediction

*(resolution loss from pooling / striding …)*

### 3.3 Dilated convolutions and upsampling

*(transposed convolutions, skip connections …)*

## 4. Instance segmentation

### 4.1 Mask R-CNN

*(…)*

### 4.2 The RoI rounding problem

*(…)*

### 4.3 Coordinate-based networks: PointRend

*(…)*

## 5. Panoptic segmentation

### 5.1 Panoptic FPN

*(…)*

### 5.2 Panoptic FCN

*(…)*

### 5.3 Evaluating panoptic segmentation

*(panoptic quality …)*

## 6. Video object segmentation

### 6.1 One-shot (semi-supervised) vs zero-shot (unsupervised) VOS

*(…)*

### 6.2 Motion-based VOS: optical flow

*(FlowNet, correlation layer, SegFlow …)*

### 6.3 Appearance-only VOS

*(OSVOS, OnAVOS, MaskTrack …)*

### 6.4 Metric-based approaches: pixel-wise retrieval

*(…)*

## References

- Shi & Malik. [Normalized Cuts and Image Segmentation](https://ieeexplore.ieee.org/document/868688). TPAMI 2000.
- Comaniciu & Meer. [Mean Shift: A Robust Approach Toward Feature Space Analysis](https://ieeexplore.ieee.org/document/1000236). TPAMI 2002.
- Krähenbühl & Koltun. [Efficient Inference in Fully Connected CRFs](https://arxiv.org/abs/1210.5644). NeurIPS 2011.
- Long et al. [Fully Convolutional Networks for Semantic Segmentation (FCN)](https://arxiv.org/abs/1411.4038). CVPR 2015.
- Yu & Koltun. [Multi-Scale Context Aggregation by Dilated Convolutions](https://arxiv.org/abs/1511.07122). ICLR 2016.
- He et al. [Mask R-CNN](https://arxiv.org/abs/1703.06870). ICCV 2017.
- Kirillov et al. [PointRend: Image Segmentation as Rendering](https://arxiv.org/abs/1912.08193). CVPR 2020.
- Kirillov et al. [Panoptic Segmentation](https://arxiv.org/abs/1801.00868). CVPR 2019.
- Kirillov et al. [Panoptic Feature Pyramid Networks (Panoptic FPN)](https://arxiv.org/abs/1901.02446). CVPR 2019.
- Li et al. [Fully Convolutional Networks for Panoptic Segmentation (Panoptic FCN)](https://arxiv.org/abs/2012.00720). CVPR 2021.
- Dosovitskiy et al. [FlowNet: Learning Optical Flow with Convolutional Networks](https://arxiv.org/abs/1504.06852). ICCV 2015.
- Cheng et al. [SegFlow: Joint Learning for Video Object Segmentation and Optical Flow](https://arxiv.org/abs/1709.06750). ICCV 2017.
- Caelles et al. [One-Shot Video Object Segmentation (OSVOS)](https://arxiv.org/abs/1611.05198). CVPR 2017.
- Voigtlaender & Leibe. [Online Adaptation of Convolutional Neural Networks for VOS (OnAVOS)](https://arxiv.org/abs/1706.09364). BMVC 2017.
- Perazzi et al. [Learning Video Object Segmentation from Static Images (MaskTrack)](https://arxiv.org/abs/1612.02646). CVPR 2017.
- Jabri et al. [Space-Time Correspondence as a Contrastive Random Walk](https://arxiv.org/abs/2006.14613). NeurIPS 2020.
