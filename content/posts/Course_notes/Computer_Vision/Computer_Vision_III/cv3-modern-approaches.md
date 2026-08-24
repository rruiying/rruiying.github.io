---
title: "Computer Vision III — Notes 4: Modern Approaches"
date: 2026-08-23
tags: [Deep learning, Computer Vision, TUM, transformers, self-supervised learning, semi-supervised learning]
summary: Fourth notes for CV3 at TUM — transformers from attention to ViT, Swin, DETR and Mask2Former; self-supervised learning from pretext tasks through contrastive (SimCLR, MoCo) and non-contrastive methods (DINO, MAE) with their downstream applications; and semi-supervised learning from its core assumptions to self-training, SAM, and consistency regularisation.
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

### 1.1 From correlation layers to attention

*(…)*

### 1.2 Attention and self-attention

*(queries / keys / values, multi-head attention …)*

### 1.3 Positional encoding

*(…)*

### 1.4 ViT

*(…)*

### 1.5 Swin Transformer

*(shifted windows, hierarchical features …)*

### 1.6 DETR

*(set prediction, bipartite matching loss, object queries …)*

### 1.7 MaskFormer and Mask2Former

*(mask classification, masked attention …)*

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
