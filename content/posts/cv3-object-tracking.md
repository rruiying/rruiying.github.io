---
title: "Computer Vision III — Notes 2: Object Tracking"
date: 2026-08-23
categories: [Course notes, Computer Vision]
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

*(…)*

## 2. Bayesian tracking and graphical models

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
