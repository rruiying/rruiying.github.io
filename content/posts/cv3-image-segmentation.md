---
title: "Computer Vision III — Notes 3: Image Segmentation"
date: 2026-08-23
categories: [Course notes, Computer Vision]
tags: [Deep learning, Computer Vision, TUM, segmentation, video object segmentation]
summary: Third notes for CV3 at TUM — segmentation from superpixels, clustering and normalized cuts through CRFs to FCNs, instance segmentation with Mask R-CNN and PointRend, panoptic segmentation and its evaluation, and video object segmentation from optical flow to pixel-wise retrieval.
---
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
