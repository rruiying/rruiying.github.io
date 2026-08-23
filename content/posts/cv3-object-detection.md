---
title: "Computer Vision III — Notes 1: Object Detection"
date: 2026-08-20
categories: [Course notes, Computer Vision]
tags: [Deep learning, Computer Vision, TUM, object detection]
summary: First notes for CV3 (Detection, Segmentation and Tracking) at TUM — object detection from classical single-stage pipelines to the R-CNN family and modern one-stage detectors, plus how detection is evaluated.
---
These notes deviate from the order of the lecture slides and start from the
motivation instead. We first look at **how detection is evaluated**, then follow
the classical single-stage pipeline — from template matching to feature-based
detectors such as the Viola–Jones detector, HOG features and non-maximum
suppression (NMS) — and its first deep-learning incarnation, **OverFeat**. We
then study **object proposals** (selective search, edge boxes) and the
two-stage **R-CNN family**: R-CNN, SPPNet and Fast R-CNN; to make proposal
generation learnable, the **region proposal network (RPN)** takes us to
Faster R-CNN. Finally we return to modern **one-stage** methods — YOLO, SSD,
focal loss and RetinaNet — and close with keypoint-based detection
(CornerNet, CenterNet), sequential detection, and spatial transformer
networks.

## 1. Evaluation: how do we measure a detector?

*(IoU, precision / recall, AP and mAP, confidence thresholds …)*

## 2. Classical single-stage detection

### 2.1 Template matching

*(…)*

### 2.2 Feature-based detection: Viola–Jones

*(Haar features, integral image, cascade …)*

### 2.3 HOG features and sliding windows

*(…)*

### 2.4 Non-maximum suppression (NMS)

*(…)*

## 3. First deep one-stage detector: OverFeat

*(…)*

## 4. Object proposals

### 4.1 Selective search

*(…)*

### 4.2 Edge boxes

*(…)*

## 5. The two-stage R-CNN family

### 5.1 R-CNN

*(…)*

### 5.2 SPPNet

*(…)*

### 5.3 Fast R-CNN

*(…)*

### 5.4 RPN and Faster R-CNN

*(…)*

## 6. Modern one-stage detectors

### 6.1 YOLO

*(…)*

### 6.2 SSD

*(…)*

### 6.3 Focal loss and RetinaNet

*(…)*

## 7. Beyond boxes

### 7.1 Keypoint-based detection: CornerNet, CenterNet

*(…)*

### 7.2 Sequential detection

*(…)*

## 8. Spatial transformers

*(…)*
