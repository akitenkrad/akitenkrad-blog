---
draft: false
title: "arXiv @ 2023.10.21"
date: 2023-10-21
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.21"
    identifier: arxiv_20231021
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (27)](#cscv-27)
- [cs.LG (30)](#cslg-30)
- [cs.AI (5)](#csai-5)
- [cs.CL (59)](#cscl-59)
- [cs.RO (4)](#csro-4)
- [eess.SP (1)](#eesssp-1)
- [cs.HC (6)](#cshc-6)
- [cond-mat.mtrl-sci (1)](#cond-matmtrl-sci-1)
- [eess.SY (1)](#eesssy-1)
- [stat.ML (3)](#statml-3)
- [cs.DC (2)](#csdc-2)
- [eess.IV (4)](#eessiv-4)
- [cs.CR (4)](#cscr-4)
- [cs.SE (2)](#csse-2)
- [cs.DL (2)](#csdl-2)
- [cs.NE (2)](#csne-2)
- [cs.MM (1)](#csmm-1)
- [cs.LO (1)](#cslo-1)
- [q-fin.PR (1)](#q-finpr-1)
- [eess.AS (1)](#eessas-1)
- [cs.IR (1)](#csir-1)
- [q-bio.NC (1)](#q-bionc-1)
- [cs.SD (1)](#cssd-1)

## cs.CV (27)



### (1/160) A Car Model Identification System for Streamlining the Automobile Sales Process (Said Togru et al., 2023)

{{<citation>}}

Said Togru, Marco Moldovan. (2023)  
**A Car Model Identification System for Streamlining the Automobile Sales Process**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.13198v2)  

---


**ABSTRACT**  
This project presents an automated solution for the efficient identification of car models and makes from images, aimed at streamlining the vehicle listing process on online car-selling platforms. Through a thorough exploration encompassing various efficient network architectures including Convolutional Neural Networks (CNNs), Vision Transformers (ViTs), and hybrid models, we achieved a notable accuracy of 81.97% employing the EfficientNet (V2 b2) architecture. To refine performance, a combination of strategies, including data augmentation, fine-tuning pretrained models, and extensive hyperparameter tuning, were applied. The trained model offers the potential for automating information extraction, promising enhanced user experiences across car-selling websites.

{{</citation>}}


### (2/160) Breaking through Deterministic Barriers: Randomized Pruning Mask Generation and Selection (Jianwei Li et al., 2023)

{{<citation>}}

Jianwei Li, Weizhi Gao, Qi Lei, Dongkuan Xu. (2023)  
**Breaking through Deterministic Barriers: Randomized Pruning Mask Generation and Selection**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GLUE, Pruning  
[Paper Link](http://arxiv.org/abs/2310.13183v1)  

---


**ABSTRACT**  
It is widely acknowledged that large and sparse models have higher accuracy than small and dense models under the same model size constraints. This motivates us to train a large model and then remove its redundant neurons or weights by pruning. Most existing works pruned the networks in a deterministic way, the performance of which solely depends on a single pruning criterion and thus lacks variety. Instead, in this paper, we propose a model pruning strategy that first generates several pruning masks in a designed random way. Subsequently, along with an effective mask-selection rule, the optimal mask is chosen from the pool of mask candidates. To further enhance efficiency, we introduce an early mask evaluation strategy, mitigating the overhead associated with training multiple masks. Our extensive experiments demonstrate that this approach achieves state-of-the-art performance across eight datasets from GLUE, particularly excelling at high levels of sparsity.

{{</citation>}}


### (3/160) LeTFuser: Light-weight End-to-end Transformer-Based Sensor Fusion for Autonomous Driving with Multi-Task Learning (Pedram Agand et al., 2023)

{{<citation>}}

Pedram Agand, Mohammad Mahdavian, Manolis Savva, Mo Chen. (2023)  
**LeTFuser: Light-weight End-to-end Transformer-Based Sensor Fusion for Autonomous Driving with Multi-Task Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.13135v1)  

---


**ABSTRACT**  
In end-to-end autonomous driving, the utilization of existing sensor fusion techniques for imitation learning proves inadequate in challenging situations that involve numerous dynamic agents. To address this issue, we introduce LeTFuser, a transformer-based algorithm for fusing multiple RGB-D camera representations. To perform perception and control tasks simultaneously, we utilize multi-task learning. Our model comprises of two modules, the first being the perception module that is responsible for encoding the observation data obtained from the RGB-D cameras. It carries out tasks such as semantic segmentation, semantic depth cloud mapping (SDC), and traffic light state recognition. Our approach employs the Convolutional vision Transformer (CvT) \cite{wu2021cvt} to better extract and fuse features from multiple RGB cameras due to local and global feature extraction capability of convolution and transformer modules, respectively. Following this, the control module undertakes the decoding of the encoded characteristics together with supplementary data, comprising a rough simulator for static and dynamic environments, as well as various measurements, in order to anticipate the waypoints associated with a latent feature space. We use two methods to process these outputs and generate the vehicular controls (e.g. steering, throttle, and brake) levels. The first method uses a PID algorithm to follow the waypoints on the fly, whereas the second one directly predicts the control policy using the measurement features and environmental state. We evaluate the model and conduct a comparative analysis with recent models on the CARLA simulator using various scenarios, ranging from normal to adversarial conditions, to simulate real-world scenarios. Our code is available at \url{https://github.com/pagand/e2etransfuser/tree/cvpr-w} to facilitate future studies.

{{</citation>}}


### (4/160) RSAdapter: Adapting Multimodal Models for Remote Sensing Visual Question Answering (Yuduo Wang et al., 2023)

{{<citation>}}

Yuduo Wang, Pedram Ghamisi. (2023)  
**RSAdapter: Adapting Multimodal Models for Remote Sensing Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Image Captioning, QA, Question Answering, Text Generation  
[Paper Link](http://arxiv.org/abs/2310.13120v1)  

---


**ABSTRACT**  
In recent years, with the rapid advancement of transformer models, transformer-based multimodal architectures have found wide application in various downstream tasks, including but not limited to Image Captioning, Visual Question Answering (VQA), and Image-Text Generation. However, contemporary approaches to Remote Sensing (RS) VQA often involve resource-intensive techniques, such as full fine-tuning of large models or the extraction of image-text features from pre-trained multimodal models, followed by modality fusion using decoders. These approaches demand significant computational resources and time, and a considerable number of trainable parameters are introduced. To address these challenges, we introduce a novel method known as RSAdapter, which prioritizes runtime and parameter efficiency. RSAdapter comprises two key components: the Parallel Adapter and an additional linear transformation layer inserted after each fully connected (FC) layer within the Adapter. This approach not only improves adaptation to pre-trained multimodal models but also allows the parameters of the linear transformation layer to be integrated into the preceding FC layers during inference, reducing inference costs. To demonstrate the effectiveness of RSAdapter, we conduct an extensive series of experiments using three distinct RS-VQA datasets and achieve state-of-the-art results on all three datasets. The code for RSAdapter will be available online at https://github.com/Y-D-Wang/RSAdapter.

{{</citation>}}


### (5/160) AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting Multiple Experts for Video Deepfake Detection (Ammarah Hashmi et al., 2023)

{{<citation>}}

Ammarah Hashmi, Sahibzada Adil Shahzad, Chia-Wen Lin, Yu Tsao, Hsin-Min Wang. (2023)  
**AVTENet: Audio-Visual Transformer-based Ensemble Network Exploiting Multiple Experts for Video Deepfake Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs-SD, cs.CV, eess-AS  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2310.13103v1)  

---


**ABSTRACT**  
Forged content shared widely on social media platforms is a major social problem that requires increased regulation and poses new challenges to the research community. The recent proliferation of hyper-realistic deepfake videos has drawn attention to the threat of audio and visual forgeries. Most previous work on detecting AI-generated fake videos only utilizes visual modality or audio modality. While there are some methods in the literature that exploit audio and visual modalities to detect forged videos, they have not been comprehensively evaluated on multi-modal datasets of deepfake videos involving acoustic and visual manipulations. Moreover, these existing methods are mostly based on CNN and suffer from low detection accuracy. Inspired by the recent success of Transformer in various fields, to address the challenges posed by deepfake technology, in this paper, we propose an Audio-Visual Transformer-based Ensemble Network (AVTENet) framework that considers both acoustic manipulation and visual manipulation to achieve effective video forgery detection. Specifically, the proposed model integrates several purely transformer-based variants that capture video, audio, and audio-visual salient cues to reach a consensus in prediction. For evaluation, we use the recently released benchmark multi-modal audio-video FakeAVCeleb dataset. For a detailed analysis, we evaluate AVTENet, its variants, and several existing methods on multiple test sets of the FakeAVCeleb dataset. Experimental results show that our best model outperforms all existing methods and achieves state-of-the-art performance on Testset-I and Testset-II of the FakeAVCeleb dataset.

{{</citation>}}


### (6/160) HumanTOMATO: Text-aligned Whole-body Motion Generation (Shunlin Lu et al., 2023)

{{<citation>}}

Shunlin Lu, Ling-Hao Chen, Ailing Zeng, Jing Lin, Ruimao Zhang, Lei Zhang, Heung-Yeung Shum. (2023)  
**HumanTOMATO: Text-aligned Whole-body Motion Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.12978v1)  

---


**ABSTRACT**  
This work targets a novel text-driven whole-body motion generation task, which takes a given textual description as input and aims at generating high-quality, diverse, and coherent facial expressions, hand gestures, and body motions simultaneously. Previous works on text-driven motion generation tasks mainly have two limitations: they ignore the key role of fine-grained hand and face controlling in vivid whole-body motion generation, and lack a good alignment between text and motion. To address such limitations, we propose a Text-aligned whOle-body Motion generATiOn framework, named HumanTOMATO, which is the first attempt to our knowledge towards applicable holistic motion generation in this research area. To tackle this challenging task, our solution includes two key designs: (1) a Holistic Hierarchical VQ-VAE (aka H$^2$VQ) and a Hierarchical-GPT for fine-grained body and hand motion reconstruction and generation with two structured codebooks; and (2) a pre-trained text-motion-alignment model to help generated motion align with the input textual description explicitly. Comprehensive experiments verify that our model has significant advantages in both the quality of generated motions and their alignment with text.

{{</citation>}}


### (7/160) FSD: Fast Self-Supervised Single RGB-D to Categorical 3D Objects (Mayank Lunayach et al., 2023)

{{<citation>}}

Mayank Lunayach, Sergey Zakharov, Dian Chen, Rares Ambrus, Zsolt Kira, Muhammad Zubair Irshad. (2023)  
**FSD: Fast Self-Supervised Single RGB-D to Categorical 3D Objects**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.12974v1)  

---


**ABSTRACT**  
In this work, we address the challenging task of 3D object recognition without the reliance on real-world 3D labeled data. Our goal is to predict the 3D shape, size, and 6D pose of objects within a single RGB-D image, operating at the category level and eliminating the need for CAD models during inference. While existing self-supervised methods have made strides in this field, they often suffer from inefficiencies arising from non-end-to-end processing, reliance on separate models for different object categories, and slow surface extraction during the training of implicit reconstruction models; thus hindering both the speed and real-world applicability of the 3D recognition process. Our proposed method leverages a multi-stage training pipeline, designed to efficiently transfer synthetic performance to the real-world domain. This approach is achieved through a combination of 2D and 3D supervised losses during the synthetic domain training, followed by the incorporation of 2D supervised and 3D self-supervised losses on real-world data in two additional learning stages. By adopting this comprehensive strategy, our method successfully overcomes the aforementioned limitations and outperforms existing self-supervised 6D pose and size estimation baselines on the NOCS test-set with a 16.4% absolute improvement in mAP for 6D pose estimation while running in near real-time at 5 Hz.

{{</citation>}}


### (8/160) Frozen Transformers in Language Models Are Effective Visual Encoder Layers (Ziqi Pang et al., 2023)

{{<citation>}}

Ziqi Pang, Ziyang Xie, Yunze Man, Yu-Xiong Wang. (2023)  
**Frozen Transformers in Language Models Are Effective Visual Encoder Layers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: LLaMA, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12973v1)  

---


**ABSTRACT**  
This paper reveals that large language models (LLMs), despite being trained solely on textual data, are surprisingly strong encoders for purely visual tasks in the absence of language. Even more intriguingly, this can be achieved by a simple yet previously overlooked strategy -- employing a frozen transformer block from pre-trained LLMs as a constituent encoder layer to directly process visual tokens. Our work pushes the boundaries of leveraging LLMs for computer vision tasks, significantly departing from conventional practices that typically necessitate a multi-modal vision-language setup with associated language prompts, inputs, or outputs. We demonstrate that our approach consistently enhances performance across a diverse range of tasks, encompassing pure 2D and 3D visual recognition tasks (e.g., image and point cloud classification), temporal modeling tasks (e.g., action recognition), non-semantic tasks (e.g., motion forecasting), and multi-modal tasks (e.g., 2D/3D visual question answering and image-text retrieval). Such improvements are a general phenomenon, applicable to various types of LLMs (e.g., LLaMA and OPT) and different LLM transformer blocks. We additionally propose the information filtering hypothesis to explain the effectiveness of pre-trained LLMs in visual encoding -- the pre-trained LLM transformer blocks discern informative visual tokens and further amplify their effect. This hypothesis is empirically supported by the observation that the feature activation, after training with LLM transformer blocks, exhibits a stronger focus on relevant regions. We hope that our work inspires new perspectives on utilizing LLMs and deepening our understanding of their underlying mechanisms. Code is available at https://github.com/ziqipang/LM4VisualEncoding.

{{</citation>}}


### (9/160) Real-Time Motion Prediction via Heterogeneous Polyline Transformer with Relative Pose Encoding (Zhejun Zhang et al., 2023)

{{<citation>}}

Zhejun Zhang, Alexander Liniger, Christos Sakaridis, Fisher Yu, Luc Van Gool. (2023)  
**Real-Time Motion Prediction via Heterogeneous Polyline Transformer with Relative Pose Encoding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12970v1)  

---


**ABSTRACT**  
The real-world deployment of an autonomous driving system requires its components to run on-board and in real-time, including the motion prediction module that predicts the future trajectories of surrounding traffic participants. Existing agent-centric methods have demonstrated outstanding performance on public benchmarks. However, they suffer from high computational overhead and poor scalability as the number of agents to be predicted increases. To address this problem, we introduce the K-nearest neighbor attention with relative pose encoding (KNARPE), a novel attention mechanism allowing the pairwise-relative representation to be used by Transformers. Then, based on KNARPE we present the Heterogeneous Polyline Transformer with Relative pose encoding (HPTR), a hierarchical framework enabling asynchronous token update during the online inference. By sharing contexts among agents and reusing the unchanged contexts, our approach is as efficient as scene-centric methods, while performing on par with state-of-the-art agent-centric methods. Experiments on Waymo and Argoverse-2 datasets show that HPTR achieves superior performance among end-to-end methods that do not apply expensive post-processing or model ensembling. The code is available at https://github.com/zhejz/HPTR.

{{</citation>}}


### (10/160) CLAIR: Evaluating Image Captions with Large Language Models (David Chan et al., 2023)

{{<citation>}}

David Chan, Suzanne Petryk, Joseph E. Gonzalez, Trevor Darrell, John Canny. (2023)  
**CLAIR: Evaluating Image Captions with Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12971v1)  

---


**ABSTRACT**  
The evaluation of machine-generated image captions poses an interesting yet persistent challenge. Effective evaluation measures must consider numerous dimensions of similarity, including semantic relevance, visual structure, object interactions, caption diversity, and specificity. Existing highly-engineered measures attempt to capture specific aspects, but fall short in providing a holistic score that aligns closely with human judgments. Here, we propose CLAIR, a novel method that leverages the zero-shot language modeling capabilities of large language models (LLMs) to evaluate candidate captions. In our evaluations, CLAIR demonstrates a stronger correlation with human judgments of caption quality compared to existing measures. Notably, on Flickr8K-Expert, CLAIR achieves relative correlation improvements over SPICE of 39.6% and over image-augmented methods such as RefCLIP-S of 18.3%. Moreover, CLAIR provides noisily interpretable results by allowing the language model to identify the underlying reasoning behind its assigned score. Code is available at https://davidmchan.github.io/clair/

{{</citation>}}


### (11/160) 3D-GPT: Procedural 3D Modeling with Large Language Models (Chunyi Sun et al., 2023)

{{<citation>}}

Chunyi Sun, Junlin Han, Weijian Deng, Xinlong Wang, Zishan Qin, Stephen Gould. (2023)  
**3D-GPT: Procedural 3D Modeling with Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12945v1)  

---


**ABSTRACT**  
In the pursuit of efficient automated content creation, procedural generation, leveraging modifiable parameters and rule-based systems, emerges as a promising approach. Nonetheless, it could be a demanding endeavor, given its intricate nature necessitating a deep understanding of rules, algorithms, and parameters. To reduce workload, we introduce 3D-GPT, a framework utilizing large language models~(LLMs) for instruction-driven 3D modeling. 3D-GPT positions LLMs as proficient problem solvers, dissecting the procedural 3D modeling tasks into accessible segments and appointing the apt agent for each task. 3D-GPT integrates three core agents: the task dispatch agent, the conceptualization agent, and the modeling agent. They collaboratively achieve two objectives. First, it enhances concise initial scene descriptions, evolving them into detailed forms while dynamically adapting the text based on subsequent instructions. Second, it integrates procedural generation, extracting parameter values from enriched text to effortlessly interface with 3D software for asset creation. Our empirical investigations confirm that 3D-GPT not only interprets and executes instructions, delivering reliable results but also collaborates effectively with human designers. Furthermore, it seamlessly integrates with Blender, unlocking expanded manipulation possibilities. Our work highlights the potential of LLMs in 3D modeling, offering a basic framework for future advancements in scene generation and animation.

{{</citation>}}


### (12/160) Unsupervised Object Localization in the Era of Self-Supervised ViTs: A Survey (Oriane Siméoni et al., 2023)

{{<citation>}}

Oriane Siméoni, Éloi Zablocki, Spyros Gidaris, Gilles Puy, Patrick Pérez. (2023)  
**Unsupervised Object Localization in the Era of Self-Supervised ViTs: A Survey**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.12904v1)  

---


**ABSTRACT**  
The recent enthusiasm for open-world vision systems show the high interest of the community to perform perception tasks outside of the closed-vocabulary benchmark setups which have been so popular until now. Being able to discover objects in images/videos without knowing in advance what objects populate the dataset is an exciting prospect. But how to find objects without knowing anything about them? Recent works show that it is possible to perform class-agnostic unsupervised object localization by exploiting self-supervised pre-trained features. We propose here a survey of unsupervised object localization methods that discover objects in images without requiring any manual annotation in the era of self-supervised ViTs. We gather links of discussed methods in the repository https://github.com/valeoai/Awesome-Unsupervised-Object-Localization.

{{</citation>}}


### (13/160) Neural Degradation Representation Learning for All-In-One Image Restoration (Mingde Yao et al., 2023)

{{<citation>}}

Mingde Yao, Ruikang Xu, Yuanshen Guan, Jie Huang, Zhiwei Xiong. (2023)  
**Neural Degradation Representation Learning for All-In-One Image Restoration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.12848v1)  

---


**ABSTRACT**  
Existing methods have demonstrated effective performance on a single degradation type. In practical applications, however, the degradation is often unknown, and the mismatch between the model and the degradation will result in a severe performance drop. In this paper, we propose an all-in-one image restoration network that tackles multiple degradations. Due to the heterogeneous nature of different types of degradations, it is difficult to process multiple degradations in a single network. To this end, we propose to learn a neural degradation representation (NDR) that captures the underlying characteristics of various degradations. The learned NDR decomposes different types of degradations adaptively, similar to a neural dictionary that represents basic degradation components. Subsequently, we develop a degradation query module and a degradation injection module to effectively recognize and utilize the specific degradation based on NDR, enabling the all-in-one restoration ability for multiple degradations. Moreover, we propose a bidirectional optimization strategy to effectively drive NDR to learn the degradation representation by optimizing the degradation and restoration processes alternately. Comprehensive experiments on representative types of degradations (including noise, haze, rain, and downsampling) demonstrate the effectiveness and generalization capability of our method.

{{</citation>}}


### (14/160) 2D-3D Interlaced Transformer for Point Cloud Segmentation with Scene-Level Supervision (Cheng-Kun Yang et al., 2023)

{{<citation>}}

Cheng-Kun Yang, Min-Hung Chen, Yung-Yu Chuang, Yen-Yu Lin. (2023)  
**2D-3D Interlaced Transformer for Point Cloud Segmentation with Scene-Level Supervision**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.12817v1)  

---


**ABSTRACT**  
We present a Multimodal Interlaced Transformer (MIT) that jointly considers 2D and 3D data for weakly supervised point cloud segmentation. Research studies have shown that 2D and 3D features are complementary for point cloud segmentation. However, existing methods require extra 2D annotations to achieve 2D-3D information fusion. Considering the high annotation cost of point clouds, effective 2D and 3D feature fusion based on weakly supervised learning is in great demand. To this end, we propose a transformer model with two encoders and one decoder for weakly supervised point cloud segmentation using only scene-level class tags. Specifically, the two encoders compute the self-attended features for 3D point clouds and 2D multi-view images, respectively. The decoder implements interlaced 2D-3D cross-attention and carries out implicit 2D and 3D feature fusion. We alternately switch the roles of queries and key-value pairs in the decoder layers. It turns out that the 2D and 3D features are iteratively enriched by each other. Experiments show that it performs favorably against existing weakly supervised point cloud segmentation methods by a large margin on the S3DIS and ScanNet benchmarks. The project page will be available at https://jimmy15923.github.io/mit_web/.

{{</citation>}}


### (15/160) Anomaly Heterogeneity Learning for Open-set Supervised Anomaly Detection (Jiawen Zhu et al., 2023)

{{<citation>}}

Jiawen Zhu, Choubo Ding, Yu Tian, Guansong Pang. (2023)  
**Anomaly Heterogeneity Learning for Open-set Supervised Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.12790v2)  

---


**ABSTRACT**  
Open-set supervised anomaly detection (OSAD) - a recently emerging anomaly detection area - aims at utilizing a few samples of anomaly classes seen during training to detect unseen anomalies (i.e., samples from open-set anomaly classes), while effectively identifying the seen anomalies. Benefiting from the prior knowledge illustrated by the seen anomalies, current OSAD methods can often largely reduce false positive errors. However, these methods treat the anomaly examples as from a homogeneous distribution, rendering them less effective in generalizing to unseen anomalies that can be drawn from any distribution. In this paper, we propose to learn heterogeneous anomaly distributions using the limited anomaly examples to address this issue. To this end, we introduce a novel approach, namely Anomaly Heterogeneity Learning (AHL), that simulates a diverse set of heterogeneous (seen and unseen) anomaly distributions and then utilizes them to learn a unified heterogeneous abnormality model. Further, AHL is a generic framework that existing OSAD models can plug and play for enhancing their abnormality modeling. Extensive experiments on nine real-world anomaly detection datasets show that AHL can 1) substantially enhance different state-of-the-art (SOTA) OSAD models in detecting both seen and unseen anomalies, achieving new SOTA performance on a large set of datasets, and 2) effectively generalize to unseen anomalies in new target domains.

{{</citation>}}


### (16/160) DT/MARS-CycleGAN: Improved Object Detection for MARS Phenotyping Robot (David Liu et al., 2023)

{{<citation>}}

David Liu, Zhengkun Li, Zihao Wu, Changying Li. (2023)  
**DT/MARS-CycleGAN: Improved Object Detection for MARS Phenotyping Robot**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.12787v2)  

---


**ABSTRACT**  
Robotic crop phenotyping has emerged as a key technology to assess crops' morphological and physiological traits at scale. These phenotypical measurements are essential for developing new crop varieties with the aim of increasing productivity and dealing with environmental challenges such as climate change. However, developing and deploying crop phenotyping robots face many challenges such as complex and variable crop shapes that complicate robotic object detection, dynamic and unstructured environments that baffle robotic control, and real-time computing and managing big data that challenge robotic hardware/software. This work specifically tackles the first challenge by proposing a novel Digital-Twin(DT)MARS-CycleGAN model for image augmentation to improve our Modular Agricultural Robotic System (MARS)'s crop object detection from complex and variable backgrounds. Our core idea is that in addition to the cycle consistency losses in the CycleGAN model, we designed and enforced a new DT-MARS loss in the deep learning model to penalize the inconsistency between real crop images captured by MARS and synthesized images sensed by DT MARS. Therefore, the generated synthesized crop images closely mimic real images in terms of realism, and they are employed to fine-tune object detectors such as YOLOv8. Extensive experiments demonstrated that our new DT/MARS-CycleGAN framework significantly boosts our MARS' crop object/row detector's performance, contributing to the field of robotic crop phenotyping.

{{</citation>}}


### (17/160) Minimalist and High-Performance Semantic Segmentation with Plain Vision Transformers (Yuanduo Hong et al., 2023)

{{<citation>}}

Yuanduo Hong, Jue Wang, Weichao Sun, Huihui Pan. (2023)  
**Minimalist and High-Performance Semantic Segmentation with Plain Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12755v1)  

---


**ABSTRACT**  
In the wake of Masked Image Modeling (MIM), a diverse range of plain, non-hierarchical Vision Transformer (ViT) models have been pre-trained with extensive datasets, offering new paradigms and significant potential for semantic segmentation. Current state-of-the-art systems incorporate numerous inductive biases and employ cumbersome decoders. Building upon the original motivations of plain ViTs, which are simplicity and generality, we explore high-performance `minimalist' systems to this end. Our primary purpose is to provide simple and efficient baselines for practical semantic segmentation with plain ViTs. Specifically, we first explore the feasibility and methodology for achieving high-performance semantic segmentation using the last feature map. As a result, we introduce the PlainSeg, a model comprising only three 3$\times$3 convolutions in addition to the transformer layers (either encoder or decoder). In this process, we offer insights into two underlying principles: (i) high-resolution features are crucial to high performance in spite of employing simple up-sampling techniques and (ii) the slim transformer decoder requires a much larger learning rate than the wide transformer decoder. On this basis, we further present the PlainSeg-Hier, which allows for the utilization of hierarchical features. Extensive experiments on four popular benchmarks demonstrate the high performance and efficiency of our methods. They can also serve as powerful tools for assessing the transfer ability of base models in semantic segmentation. Code is available at \url{https://github.com/ydhongHIT/PlainSeg}.

{{</citation>}}


### (18/160) Recoverable Privacy-Preserving Image Classification through Noise-like Adversarial Examples (Jun Liu et al., 2023)

{{<citation>}}

Jun Liu, Jiantao Zhou, Jinyu Tian, Weiwei Sun. (2023)  
**Recoverable Privacy-Preserving Image Classification through Noise-like Adversarial Examples**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2310.12707v1)  

---


**ABSTRACT**  
With the increasing prevalence of cloud computing platforms, ensuring data privacy during the cloud-based image related services such as classification has become crucial. In this study, we propose a novel privacypreserving image classification scheme that enables the direct application of classifiers trained in the plaintext domain to classify encrypted images, without the need of retraining a dedicated classifier. Moreover, encrypted images can be decrypted back into their original form with high fidelity (recoverable) using a secret key. Specifically, our proposed scheme involves utilizing a feature extractor and an encoder to mask the plaintext image through a newly designed Noise-like Adversarial Example (NAE). Such an NAE not only introduces a noise-like visual appearance to the encrypted image but also compels the target classifier to predict the ciphertext as the same label as the original plaintext image. At the decoding phase, we adopt a Symmetric Residual Learning (SRL) framework for restoring the plaintext image with minimal degradation. Extensive experiments demonstrate that 1) the classification accuracy of the classifier trained in the plaintext domain remains the same in both the ciphertext and plaintext domains; 2) the encrypted images can be recovered into their original form with an average PSNR of up to 51+ dB for the SVHN dataset and 48+ dB for the VGGFace2 dataset; 3) our system exhibits satisfactory generalization capability on the encryption, decryption and classification tasks across datasets that are different from the training one; and 4) a high-level of security is achieved against three potential threat models. The code is available at https://github.com/csjunjun/RIC.git.

{{</citation>}}


### (19/160) Exploiting Low-confidence Pseudo-labels for Source-free Object Detection (Zhihong Chen et al., 2023)

{{<citation>}}

Zhihong Chen, Zilei Wang, Yixin Zhang. (2023)  
**Exploiting Low-confidence Pseudo-labels for Source-free Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, Object Detection  
[Paper Link](http://arxiv.org/abs/2310.12705v1)  

---


**ABSTRACT**  
Source-free object detection (SFOD) aims to adapt a source-trained detector to an unlabeled target domain without access to the labeled source data. Current SFOD methods utilize a threshold-based pseudo-label approach in the adaptation phase, which is typically limited to high-confidence pseudo-labels and results in a loss of information. To address this issue, we propose a new approach to take full advantage of pseudo-labels by introducing high and low confidence thresholds. Specifically, the pseudo-labels with confidence scores above the high threshold are used conventionally, while those between the low and high thresholds are exploited using the Low-confidence Pseudo-labels Utilization (LPU) module. The LPU module consists of Proposal Soft Training (PST) and Local Spatial Contrastive Learning (LSCL). PST generates soft labels of proposals for soft training, which can mitigate the label mismatch problem. LSCL exploits the local spatial relationship of proposals to improve the model's ability to differentiate between spatially adjacent proposals, thereby optimizing representational features further. Combining the two components overcomes the challenges faced by traditional methods in utilizing low-confidence pseudo-labels. Extensive experiments on five cross-domain object detection benchmarks demonstrate that our proposed method outperforms the previous SFOD methods, achieving state-of-the-art performance.

{{</citation>}}


### (20/160) Representation Learning via Consistent Assignment of Views over Random Partitions (Thalles Silva et al., 2023)

{{<citation>}}

Thalles Silva, Adín Ramírez Rivera. (2023)  
**Representation Learning via Consistent Assignment of Views over Random Partitions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.12692v1)  

---


**ABSTRACT**  
We present Consistent Assignment of Views over Random Partitions (CARP), a self-supervised clustering method for representation learning of visual features. CARP learns prototypes in an end-to-end online fashion using gradient descent without additional non-differentiable modules to solve the cluster assignment problem. CARP optimizes a new pretext task based on random partitions of prototypes that regularizes the model and enforces consistency between views' assignments. Additionally, our method improves training stability and prevents collapsed solutions in joint-embedding training. Through an extensive evaluation, we demonstrate that CARP's representations are suitable for learning downstream tasks. We evaluate CARP's representations capabilities in 17 datasets across many standard protocols, including linear evaluation, few-shot classification, k-NN, k-means, image retrieval, and copy detection. We compare CARP performance to 11 existing self-supervised methods. We extensively ablate our method and demonstrate that our proposed random partition pretext task improves the quality of the learned representations by devising multiple random classification tasks. In transfer learning tasks, CARP achieves the best performance on average against many SSL methods trained for a longer time.

{{</citation>}}


### (21/160) Heart Disease Detection using Vision-Based Transformer Models from ECG Images (Zeynep Hilal Kilimci et al., 2023)

{{<citation>}}

Zeynep Hilal Kilimci, Mustafa Yalcin, Ayhan Kucukmanisa, Amit Kumar Mishra. (2023)  
**Heart Disease Detection using Vision-Based Transformer Models from ECG Images**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Google, Microsoft, Transformer  
[Paper Link](http://arxiv.org/abs/2310.12630v1)  

---


**ABSTRACT**  
Heart disease, also known as cardiovascular disease, is a prevalent and critical medical condition characterized by the impairment of the heart and blood vessels, leading to various complications such as coronary artery disease, heart failure, and myocardial infarction. The timely and accurate detection of heart disease is of paramount importance in clinical practice. Early identification of individuals at risk enables proactive interventions, preventive measures, and personalized treatment strategies to mitigate the progression of the disease and reduce adverse outcomes. In recent years, the field of heart disease detection has witnessed notable advancements due to the integration of sophisticated technologies and computational approaches. These include machine learning algorithms, data mining techniques, and predictive modeling frameworks that leverage vast amounts of clinical and physiological data to improve diagnostic accuracy and risk stratification. In this work, we propose to detect heart disease from ECG images using cutting-edge technologies, namely vision transformer models. These models are Google-Vit, Microsoft-Beit, and Swin-Tiny. To the best of our knowledge, this is the initial endeavor concentrating on the detection of heart diseases through image-based ECG data by employing cuttingedge technologies namely, transformer models. To demonstrate the contribution of the proposed framework, the performance of vision transformer models are compared with state-of-the-art studies. Experiment results show that the proposed framework exhibits remarkable classification results.

{{</citation>}}


### (22/160) Cross-attention Spatio-temporal Context Transformer for Semantic Segmentation of Historical Maps (Sidi Wu et al., 2023)

{{<citation>}}

Sidi Wu, Yizi Chen, Konrad Schindler, Lorenz Hurni. (2023)  
**Cross-attention Spatio-temporal Context Transformer for Semantic Segmentation of Historical Maps**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation, Transformer  
[Paper Link](http://arxiv.org/abs/2310.12616v1)  

---


**ABSTRACT**  
Historical maps provide useful spatio-temporal information on the Earth's surface before modern earth observation techniques came into being. To extract information from maps, neural networks, which gain wide popularity in recent years, have replaced hand-crafted map processing methods and tedious manual labor. However, aleatoric uncertainty, known as data-dependent uncertainty, inherent in the drawing/scanning/fading defects of the original map sheets and inadequate contexts when cropping maps into small tiles considering the memory limits of the training process, challenges the model to make correct predictions. As aleatoric uncertainty cannot be reduced even with more training data collected, we argue that complementary spatio-temporal contexts can be helpful. To achieve this, we propose a U-Net-based network that fuses spatio-temporal features with cross-attention transformers (U-SpaTem), aggregating information at a larger spatial range as well as through a temporal sequence of images. Our model achieves a better performance than other state-or-art models that use either temporal or spatial contexts. Compared with pure vision transformers, our model is more lightweight and effective. To the best of our knowledge, leveraging both spatial and temporal contexts have been rarely explored before in the segmentation task. Even though our application is on segmenting historical maps, we believe that the method can be transferred into other fields with similar problems like temporal sequences of satellite images. Our code is freely accessible at https://github.com/chenyizi086/wu.2023.sigspatial.git.

{{</citation>}}


### (23/160) Diverse Diffusion: Enhancing Image Diversity in Text-to-Image Generation (Mariia Zameshina et al., 2023)

{{<citation>}}

Mariia Zameshina, Olivier Teytaud, Laurent Najman. (2023)  
**Diverse Diffusion: Enhancing Image Diversity in Text-to-Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.12583v1)  

---


**ABSTRACT**  
Latent diffusion models excel at producing high-quality images from text. Yet, concerns appear about the lack of diversity in the generated imagery. To tackle this, we introduce Diverse Diffusion, a method for boosting image diversity beyond gender and ethnicity, spanning into richer realms, including color diversity.Diverse Diffusion is a general unsupervised technique that can be applied to existing text-to-image models. Our approach focuses on finding vectors in the Stable Diffusion latent space that are distant from each other. We generate multiple vectors in the latent space until we find a set of vectors that meets the desired distance requirements and the required batch size.To evaluate the effectiveness of our diversity methods, we conduct experiments examining various characteristics, including color diversity, LPIPS metric, and ethnicity/gender representation in images featuring humans.The results of our experiments emphasize the significance of diversity in generating realistic and varied images, offering valuable insights for improving text-to-image models. Through the enhancement of image diversity, our approach contributes to the creation of more inclusive and representative AI-generated art.

{{</citation>}}


### (24/160) Weakly-Supervised Semantic Segmentation with Image-Level Labels: from Traditional Models to Foundation Models (Zhaozheng Chen et al., 2023)

{{<citation>}}

Zhaozheng Chen, Qianru Sun. (2023)  
**Weakly-Supervised Semantic Segmentation with Image-Level Labels: from Traditional Models to Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.13026v1)  

---


**ABSTRACT**  
The rapid development of deep learning has driven significant progress in the field of image semantic segmentation - a fundamental task in computer vision. Semantic segmentation algorithms often depend on the availability of pixel-level labels (i.e., masks of objects), which are expensive, time-consuming, and labor-intensive. Weakly-supervised semantic segmentation (WSSS) is an effective solution to avoid such labeling. It utilizes only partial or incomplete annotations and provides a cost-effective alternative to fully-supervised semantic segmentation. In this paper, we focus on the WSSS with image-level labels, which is the most challenging form of WSSS. Our work has two parts. First, we conduct a comprehensive survey on traditional methods, primarily focusing on those presented at premier research conferences. We categorize them into four groups based on where their methods operate: pixel-wise, image-wise, cross-image, and external data. Second, we investigate the applicability of visual foundation models, such as the Segment Anything Model (SAM), in the context of WSSS. We scrutinize SAM in two intriguing scenarios: text prompting and zero-shot learning. We provide insights into the potential and challenges associated with deploying visual foundational models for WSSS, facilitating future developments in this exciting research area.

{{</citation>}}


### (25/160) WeedCLR: Weed Contrastive Learning through Visual Representations with Class-Optimized Loss in Long-Tailed Datasets (Alzayat Saleh et al., 2023)

{{<citation>}}

Alzayat Saleh, Alex Olsen, Jake Wood, Bronson Philippa, Mostafa Rahimi Azghadi. (2023)  
**WeedCLR: Weed Contrastive Learning through Visual Representations with Class-Optimized Loss in Long-Tailed Datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.12465v1)  

---


**ABSTRACT**  
Image classification is a crucial task in modern weed management and crop intervention technologies. However, the limited size, diversity, and balance of existing weed datasets hinder the development of deep learning models for generalizable weed identification. In addition, the expensive labelling requirements of mainstream fully-supervised weed classifiers make them cost- and time-prohibitive to deploy widely, for new weed species, and in site-specific weed management. This paper proposes a novel method for Weed Contrastive Learning through visual Representations (WeedCLR), that uses class-optimized loss with Von Neumann Entropy of deep representation for weed classification in long-tailed datasets. WeedCLR leverages self-supervised learning to learn rich and robust visual features without any labels and applies a class-optimized loss function to address the class imbalance problem in long-tailed datasets. WeedCLR is evaluated on two public weed datasets: CottonWeedID15, containing 15 weed species, and DeepWeeds, containing 8 weed species. WeedCLR achieves an average accuracy improvement of 4.3\% on CottonWeedID15 and 5.6\% on DeepWeeds over previous methods. It also demonstrates better generalization ability and robustness to different environmental conditions than existing methods without the need for expensive and time-consuming human annotations. These significant improvements make WeedCLR an effective tool for weed classification in long-tailed datasets and allows for more rapid and widespread deployment of site-specific weed management and crop intervention technologies.

{{</citation>}}


### (26/160) Not Just Learning from Others but Relying on Yourself: A New Perspective on Few-Shot Segmentation in Remote Sensing (Hanbo Bi et al., 2023)

{{<citation>}}

Hanbo Bi, Yingchao Feng, Zhiyuan Yan, Yongqiang Mao, Wenhui Diao, Hongqi Wang, Xian Sun. (2023)  
**Not Just Learning from Others but Relying on Yourself: A New Perspective on Few-Shot Segmentation in Remote Sensing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Few-Shot  
[Paper Link](http://arxiv.org/abs/2310.12452v1)  

---


**ABSTRACT**  
Few-shot segmentation (FSS) is proposed to segment unknown class targets with just a few annotated samples. Most current FSS methods follow the paradigm of mining the semantics from the support images to guide the query image segmentation. However, such a pattern of `learning from others' struggles to handle the extreme intra-class variation, preventing FSS from being directly generalized to remote sensing scenes. To bridge the gap of intra-class variance, we develop a Dual-Mining network named DMNet for cross-image mining and self-mining, meaning that it no longer focuses solely on support images but pays more attention to the query image itself. Specifically, we propose a Class-public Region Mining (CPRM) module to effectively suppress irrelevant feature pollution by capturing the common semantics between the support-query image pair. The Class-specific Region Mining (CSRM) module is then proposed to continuously mine the class-specific semantics of the query image itself in a `filtering' and `purifying' manner. In addition, to prevent the co-existence of multiple classes in remote sensing scenes from exacerbating the collapse of FSS generalization, we also propose a new Known-class Meta Suppressor (KMS) module to suppress the activation of known-class objects in the sample. Extensive experiments on the iSAID and LoveDA remote sensing datasets have demonstrated that our method sets the state-of-the-art with a minimum number of model parameters. Significantly, our model with the backbone of Resnet-50 achieves the mIoU of 49.58% and 51.34% on iSAID under 1-shot and 5-shot settings, outperforming the state-of-the-art method by 1.8% and 1.12%, respectively. The code is publicly available at https://github.com/HanboBizl/DMNet.

{{</citation>}}


### (27/160) DocXChain: A Powerful Open-Source Toolchain for Document Parsing and Beyond (Cong Yao, 2023)

{{<citation>}}

Cong Yao. (2023)  
**DocXChain: A Powerful Open-Source Toolchain for Document Parsing and Beyond**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.12430v1)  

---


**ABSTRACT**  
In this report, we introduce DocXChain, a powerful open-source toolchain for document parsing, which is designed and developed to automatically convert the rich information embodied in unstructured documents, such as text, tables and charts, into structured representations that are readable and manipulable by machines. Specifically, basic capabilities, including text detection, text recognition, table structure recognition and layout analysis, are provided. Upon these basic capabilities, we also build a set of fully functional pipelines for document parsing, i.e., general text reading, table parsing, and document structurization, to drive various applications related to documents in real-world scenarios. Moreover, DocXChain is concise, modularized and flexible, such that it can be readily integrated with existing tools, libraries or models (such as LangChain and ChatGPT), to construct more powerful systems that can accomplish more complicated and challenging tasks. The code of DocXChain is publicly available at:~\url{https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/Applications/DocXChain}

{{</citation>}}


## cs.LG (30)



### (28/160) Heterogeneous Graph Neural Networks for Data-driven Traffic Assignment (Tong Liu et al., 2023)

{{<citation>}}

Tong Liu, Hadi Meidani. (2023)  
**Heterogeneous Graph Neural Networks for Data-driven Traffic Assignment**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.13193v1)  

---


**ABSTRACT**  
The traffic assignment problem is one of the significant components of traffic flow analysis for which various solution approaches have been proposed. However, deploying these approaches for large-scale networks poses significant challenges. In this paper, we leverage the power of heterogeneous graph neural networks to propose a novel data-driven approach for traffic assignment and traffic flow learning. The proposed model is capable of capturing spatial traffic patterns across different links, yielding highly accurate results. We present numerical experiments on urban transportation networks and show that the proposed heterogeneous graph neural network model outperforms other conventional neural network models in terms of convergence rate, training loss, and prediction accuracy. Notably, the proposed heterogeneous graph neural network model can also be generalized to different network topologies. This approach offers a promising solution for complex traffic flow analysis and prediction, enhancing our understanding and management of a wide range of transportation systems.

{{</citation>}}


### (29/160) Graph Neural Networks with polynomial activations have limited expressivity (Sammy Khalife, 2023)

{{<citation>}}

Sammy Khalife. (2023)  
**Graph Neural Networks with polynomial activations have limited expressivity**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.13139v1)  

---


**ABSTRACT**  
The expressivity of Graph Neural Networks (GNNs) can be entirely characterized by appropriate fragments of the first order logic. Namely, any query of the two variable fragment of graded modal logic (GC2) interpreted over labelled graphs can be expressed using a GNN whose size depends only on the depth of the query. As pointed out by [Barcelo & Al., 2020, Grohe, 2021 ], this description holds for a family of activation functions, leaving the possibibility for a hierarchy of logics expressible by GNNs depending on the chosen activation function. In this article, we show that such hierarchy indeed exists by proving that GC2 queries cannot be expressed by GNNs with polynomial activation functions. This implies a separation between polynomial and popular non polynomial activations (such as ReLUs, sigmoid and hyperbolic tan and others) and answers an open question formulated by [Grohe, 2021].

{{</citation>}}


### (30/160) Understanding Addition in Transformers (Philip Quirke et al., 2023)

{{<citation>}}

Philip Quirke, Fazl Barez. (2023)  
**Understanding Addition in Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.13121v2)  

---


**ABSTRACT**  
Understanding the inner workings of machine learning models like Transformers is vital for their safe and ethical use. This paper presents an in-depth analysis of a one-layer Transformer model trained for integer addition. We reveal that the model divides the task into parallel, digit-specific streams and employs distinct algorithms for different digit positions. Our study also finds that the model starts calculations late but executes them rapidly. A rare use case with high loss is identified and explained. Overall, the model's algorithm is explained in detail. These findings are validated through rigorous testing and mathematical modeling, contributing to the broader works in Mechanistic Interpretability, AI safety, and alignment. Our approach opens the door for analyzing more complex tasks and multi-layer Transformer models.

{{</citation>}}


### (31/160) Semi-Supervised Learning of Dynamical Systems with Neural Ordinary Differential Equations: A Teacher-Student Model Approach (Yu Wang et al., 2023)

{{<citation>}}

Yu Wang, Yuxuan Yin, Karthik Somayaji Nanjangud Suryanarayana, Jan Drgona, Malachi Schram, Mahantesh Halappanavar, Frank Liu, Peng Li. (2023)  
**Semi-Supervised Learning of Dynamical Systems with Neural Ordinary Differential Equations: A Teacher-Student Model Approach**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.13110v1)  

---


**ABSTRACT**  
Modeling dynamical systems is crucial for a wide range of tasks, but it remains challenging due to complex nonlinear dynamics, limited observations, or lack of prior knowledge. Recently, data-driven approaches such as Neural Ordinary Differential Equations (NODE) have shown promising results by leveraging the expressive power of neural networks to model unknown dynamics. However, these approaches often suffer from limited labeled training data, leading to poor generalization and suboptimal predictions. On the other hand, semi-supervised algorithms can utilize abundant unlabeled data and have demonstrated good performance in classification and regression tasks. We propose TS-NODE, the first semi-supervised approach to modeling dynamical systems with NODE. TS-NODE explores cheaply generated synthetic pseudo rollouts to broaden exploration in the state space and to tackle the challenges brought by lack of ground-truth system data under a teacher-student model. TS-NODE employs an unified optimization framework that corrects the teacher model based on the student's feedback while mitigating the potential false system dynamics present in pseudo rollouts. TS-NODE demonstrates significant performance improvements over a baseline Neural ODE model on multiple dynamical system modeling tasks.

{{</citation>}}


### (32/160) SRAI: Towards Standardization of Geospatial AI (Piotr Gramacki et al., 2023)

{{<citation>}}

Piotr Gramacki, Kacper Leśniara, Kamil Raczycki, Szymon Woźniak, Marcin Przymus, Piotr Szymański. (2023)  
**SRAI: Towards Standardization of Geospatial AI**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13098v2)  

---


**ABSTRACT**  
Spatial Representations for Artificial Intelligence (srai) is a Python library for working with geospatial data. The library can download geospatial data, split a given area into micro-regions using multiple algorithms and train an embedding model using various architectures. It includes baseline models as well as more complex methods from published works. Those capabilities make it possible to use srai in a complete pipeline for geospatial task solving. The proposed library is the first step to standardize the geospatial AI domain toolset. It is fully open-source and published under Apache 2.0 licence.

{{</citation>}}


### (33/160) Unsupervised Representation Learning to Aid Semi-Supervised Meta Learning (Atik Faysal et al., 2023)

{{<citation>}}

Atik Faysal, Mohammad Rostami, Huaxia Wang, Avimanyu Sahoo, Ryan Antle. (2023)  
**Unsupervised Representation Learning to Aid Semi-Supervised Meta Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2310.13085v1)  

---


**ABSTRACT**  
Few-shot learning or meta-learning leverages the data scarcity problem in machine learning. Traditionally, training data requires a multitude of samples and labeling for supervised learning. To address this issue, we propose a one-shot unsupervised meta-learning to learn the latent representation of the training samples. We use augmented samples as the query set during the training phase of the unsupervised meta-learning. A temperature-scaled cross-entropy loss is used in the inner loop of meta-learning to prevent overfitting during unsupervised learning. The learned parameters from this step are applied to the targeted supervised meta-learning in a transfer-learning fashion for initialization and fast adaptation with improved accuracy. The proposed method is model agnostic and can aid any meta-learning model to improve accuracy. We use model agnostic meta-learning (MAML) and relation network (RN) on Omniglot and mini-Imagenet datasets to demonstrate the performance of the proposed method. Furthermore, a meta-learning model with the proposed initialization can achieve satisfactory accuracy with significantly fewer training samples.

{{</citation>}}


### (34/160) Robust multimodal models have outlier features and encode more concepts (Jonathan Crabbé et al., 2023)

{{<citation>}}

Jonathan Crabbé, Pau Rodríguez, Vaishaal Shankar, Luca Zappella, Arno Blaas. (2023)  
**Robust multimodal models have outlier features and encode more concepts**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13040v1)  

---


**ABSTRACT**  
What distinguishes robust models from non-robust ones? This question has gained traction with the appearance of large-scale multimodal models, such as CLIP. These models have demonstrated unprecedented robustness with respect to natural distribution shifts. While it has been shown that such differences in robustness can be traced back to differences in training data, so far it is not known what that translates to in terms of what the model has learned. In this work, we bridge this gap by probing the representation spaces of 12 robust multimodal models with various backbones (ResNets and ViTs) and pretraining sets (OpenAI, LAION-400M, LAION-2B, YFCC15M, CC12M and DataComp). We find two signatures of robustness in the representation spaces of these models: (1) Robust models exhibit outlier features characterized by their activations, with some being several orders of magnitude above average. These outlier features induce privileged directions in the model's representation space. We demonstrate that these privileged directions explain most of the predictive power of the model by pruning up to $80 \%$ of the least important representation space directions without negative impacts on model accuracy and robustness; (2) Robust models encode substantially more concepts in their representation space. While this superposition of concepts allows robust models to store much information, it also results in highly polysemantic features, which makes their interpretation challenging. We discuss how these insights pave the way for future research in various fields, such as model pruning and mechanistic interpretability.

{{</citation>}}


### (35/160) Does Your Model Think Like an Engineer? Explainable AI for Bearing Fault Detection with Deep Learning (Thomas Decker et al., 2023)

{{<citation>}}

Thomas Decker, Michael Lebacher, Volker Tresp. (2023)  
**Does Your Model Think Like an Engineer? Explainable AI for Bearing Fault Detection with Deep Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.12967v1)  

---


**ABSTRACT**  
Deep Learning has already been successfully applied to analyze industrial sensor data in a variety of relevant use cases. However, the opaque nature of many well-performing methods poses a major obstacle for real-world deployment. Explainable AI (XAI) and especially feature attribution techniques promise to enable insights about how such models form their decision. But the plain application of such methods often fails to provide truly informative and problem-specific insights to domain experts. In this work, we focus on the specific task of detecting faults in rolling element bearings from vibration signals. We propose a novel and domain-specific feature attribution framework that allows us to evaluate how well the underlying logic of a model corresponds with expert reasoning. Utilizing the framework we are able to validate the trustworthiness and to successfully anticipate the generalization ability of different well-performing deep learning models. Our methodology demonstrates how signal processing tools can effectively be used to enhance Explainable AI techniques and acts as a template for similar problems.

{{</citation>}}


### (36/160) Eureka-Moments in Transformers: Multi-Step Tasks Reveal Softmax Induced Optimization Problems (David T. Hoffmann et al., 2023)

{{<citation>}}

David T. Hoffmann, Simon Schrodi, Nadine Behrmann, Volker Fischer, Thomas Brox. (2023)  
**Eureka-Moments in Transformers: Multi-Step Tasks Reveal Softmax Induced Optimization Problems**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12956v1)  

---


**ABSTRACT**  
In this work, we study rapid, step-wise improvements of the loss in transformers when being confronted with multi-step decision tasks. We found that transformers struggle to learn the intermediate tasks, whereas CNNs have no such issue on the tasks we studied. When transformers learn the intermediate task, they do this rapidly and unexpectedly after both training and validation loss saturated for hundreds of epochs. We call these rapid improvements Eureka-moments, since the transformer appears to suddenly learn a previously incomprehensible task. Similar leaps in performance have become known as Grokking. In contrast to Grokking, for Eureka-moments, both the validation and the training loss saturate before rapidly improving. We trace the problem back to the Softmax function in the self-attention block of transformers and show ways to alleviate the problem. These fixes improve training speed. The improved models reach 95% of the baseline model in just 20% of training steps while having a much higher likelihood to learn the intermediate task, lead to higher final accuracy and are more robust to hyper-parameters.

{{</citation>}}


### (37/160) Towards Robust Offline Reinforcement Learning under Diverse Data Corruption (Rui Yang et al., 2023)

{{<citation>}}

Rui Yang, Han Zhong, Jiawei Xu, Amy Zhang, Chongjie Zhang, Lei Han, Tong Zhang. (2023)  
**Towards Robust Offline Reinforcement Learning under Diverse Data Corruption**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.12955v1)  

---


**ABSTRACT**  
Offline reinforcement learning (RL) presents a promising approach for learning reinforced policies from offline datasets without the need for costly or unsafe interactions with the environment. However, datasets collected by humans in real-world environments are often noisy and may even be maliciously corrupted, which can significantly degrade the performance of offline RL. In this work, we first investigate the performance of current offline RL algorithms under comprehensive data corruption, including states, actions, rewards, and dynamics. Our extensive experiments reveal that implicit Q-learning (IQL) demonstrates remarkable resilience to data corruption among various offline RL algorithms. Furthermore, we conduct both empirical and theoretical analyses to understand IQL's robust performance, identifying its supervised policy learning scheme as the key factor. Despite its relative robustness, IQL still suffers from heavy-tail targets of Q functions under dynamics corruption. To tackle this challenge, we draw inspiration from robust statistics to employ the Huber loss to handle the heavy-tailedness and utilize quantile estimators to balance penalization for corrupted data and learning stability. By incorporating these simple yet effective modifications into IQL, we propose a more robust offline RL approach named Robust IQL (RIQL). Extensive experiments demonstrate that RIQL exhibits highly robust performance when subjected to diverse data corruption scenarios.

{{</citation>}}


### (38/160) The Foundation Model Transparency Index (Rishi Bommasani et al., 2023)

{{<citation>}}

Rishi Bommasani, Kevin Klyman, Shayne Longpre, Sayash Kapoor, Nestor Maslej, Betty Xiong, Daniel Zhang, Percy Liang. (2023)  
**The Foundation Model Transparency Index**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, GPT, GPT-4, Google, PaLM  
[Paper Link](http://arxiv.org/abs/2310.12941v1)  

---


**ABSTRACT**  
Foundation models have rapidly permeated society, catalyzing a wave of generative AI applications spanning enterprise and consumer-facing contexts. While the societal impact of foundation models is growing, transparency is on the decline, mirroring the opacity that has plagued past digital technologies (e.g. social media). Reversing this trend is essential: transparency is a vital precondition for public accountability, scientific innovation, and effective governance. To assess the transparency of the foundation model ecosystem and help improve transparency over time, we introduce the Foundation Model Transparency Index. The Foundation Model Transparency Index specifies 100 fine-grained indicators that comprehensively codify transparency for foundation models, spanning the upstream resources used to build a foundation model (e.g data, labor, compute), details about the model itself (e.g. size, capabilities, risks), and the downstream use (e.g. distribution channels, usage policies, affected geographies). We score 10 major foundation model developers (e.g. OpenAI, Google, Meta) against the 100 indicators to assess their transparency. To facilitate and standardize assessment, we score developers in relation to their practices for their flagship foundation model (e.g. GPT-4 for OpenAI, PaLM 2 for Google, Llama 2 for Meta). We present 10 top-level findings about the foundation model ecosystem: for example, no developer currently discloses significant information about the downstream impact of its flagship model, such as the number of users, affected market sectors, or how users can seek redress for harm. Overall, the Foundation Model Transparency Index establishes the level of transparency today to drive progress on foundation model governance via industry standards and regulatory intervention.

{{</citation>}}


### (39/160) Probabilistic Modeling of Human Teams to Infer False Beliefs (Paulo Soares et al., 2023)

{{<citation>}}

Paulo Soares, Adarsh Pyarelal, Kobus Barnard. (2023)  
**Probabilistic Modeling of Human Teams to Infer False Beliefs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.12929v1)  

---


**ABSTRACT**  
We develop a probabilistic graphical model (PGM) for artificially intelligent (AI) agents to infer human beliefs during a simulated urban search and rescue (USAR) scenario executed in a Minecraft environment with a team of three players. The PGM approach makes observable states and actions explicit, as well as beliefs and intentions grounded by evidence about what players see and do over time. This approach also supports inferring the effect of interventions, which are vital if AI agents are to assist human teams. The experiment incorporates manipulations of players' knowledge, and the virtual Minecraft-based testbed provides access to several streams of information, including the objects in the players' field of view. The participants are equipped with a set of marker blocks that can be placed near room entrances to signal the presence or absence of victims in the rooms to their teammates. In each team, one of the members is given a different legend for the markers than the other two, which may mislead them about the state of the rooms; that is, they will hold a false belief. We extend previous works in this field by introducing ToMCAT, an AI agent that can reason about individual and shared mental states. We find that the players' behaviors are affected by what they see in their in-game field of view, their beliefs about the meaning of the markers, and their beliefs about which meaning the team decided to adopt. In addition, we show that ToMCAT's beliefs are consistent with the players' actions and that it can infer false beliefs with accuracy significantly better than chance and comparable to inferences made by human observers.

{{</citation>}}


### (40/160) Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning (Juan Rocamonde et al., 2023)

{{<citation>}}

Juan Rocamonde, Victoriano Montesinos, Elvis Nava, Ethan Perez, David Lindner. (2023)  
**Vision-Language Models are Zero-Shot Reward Models for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Language Model, Reinforcement Learning, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.12921v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) requires either manually specifying a reward function, which is often infeasible, or learning a reward model from a large amount of human feedback, which is often very expensive. We study a more sample-efficient alternative: using pretrained vision-language models (VLMs) as zero-shot reward models (RMs) to specify tasks via natural language. We propose a natural and general approach to using VLMs as reward models, which we call VLM-RMs. We use VLM-RMs based on CLIP to train a MuJoCo humanoid to learn complex tasks without a manually specified reward function, such as kneeling, doing the splits, and sitting in a lotus position. For each of these tasks, we only provide a single sentence text prompt describing the desired task with minimal prompt engineering. We provide videos of the trained agents at: https://sites.google.com/view/vlm-rm. We can improve performance by providing a second ``baseline'' prompt and projecting out parts of the CLIP embedding space irrelevant to distinguish between goal and baseline. Further, we find a strong scaling effect for VLM-RMs: larger VLMs trained with more compute and data are better reward models. The failure modes of VLM-RMs we encountered are all related to known capability limitations of current VLMs, such as limited spatial reasoning ability or visually unrealistic environments that are far off-distribution for the VLM. We find that VLM-RMs are remarkably robust as long as the VLM is large enough. This suggests that future VLMs will become more and more useful reward models for a wide range of RL applications.

{{</citation>}}


### (41/160) Causal-structure Driven Augmentations for Text OOD Generalization (Amir Feder et al., 2023)

{{<citation>}}

Amir Feder, Yoav Wald, Claudia Shi, Suchi Saria, David Blei. (2023)  
**Causal-structure Driven Augmentations for Text OOD Generalization**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.12803v1)  

---


**ABSTRACT**  
The reliance of text classifiers on spurious correlations can lead to poor generalization at deployment, raising concerns about their use in safety-critical domains such as healthcare. In this work, we propose to use counterfactual data augmentation, guided by knowledge of the causal structure of the data, to simulate interventions on spurious features and to learn more robust text classifiers. We show that this strategy is appropriate in prediction problems where the label is spuriously correlated with an attribute. Under the assumptions of such problems, we discuss the favorable sample complexity of counterfactual data augmentation, compared to importance re-weighting. Pragmatically, we match examples using auxiliary data, based on diff-in-diff methodology, and use a large language model (LLM) to represent a conditional probability of text. Through extensive experimentation on learning caregiver-invariant predictors of clinical diagnoses from medical narratives and on semi-synthetic data, we demonstrate that our method for simulating interventions improves out-of-distribution (OOD) accuracy compared to baseline invariant learning algorithms.

{{</citation>}}


### (42/160) Exploring Graph Neural Networks for Indian Legal Judgment Prediction (Mann Khatri et al., 2023)

{{<citation>}}

Mann Khatri, Mirza Yusuf, Yaman Kumar, Rajiv Ratn Shah, Ponnurangam Kumaraguru. (2023)  
**Exploring Graph Neural Networks for Indian Legal Judgment Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks, Legal  
[Paper Link](http://arxiv.org/abs/2310.12800v1)  

---


**ABSTRACT**  
The burdensome impact of a skewed judges-to-cases ratio on the judicial system manifests in an overwhelming backlog of pending cases alongside an ongoing influx of new ones. To tackle this issue and expedite the judicial process, the proposition of an automated system capable of suggesting case outcomes based on factual evidence and precedent from past cases gains significance. This research paper centres on developing a graph neural network-based model to address the Legal Judgment Prediction (LJP) problem, recognizing the intrinsic graph structure of judicial cases and making it a binary node classification problem. We explored various embeddings as model features, while nodes such as time nodes and judicial acts were added and pruned to evaluate the model's performance. The study is done while considering the ethical dimension of fairness in these predictions, considering gender and name biases. A link prediction task is also conducted to assess the model's proficiency in anticipating connections between two specified nodes. By harnessing the capabilities of graph neural networks and incorporating fairness analyses, this research aims to contribute insights towards streamlining the adjudication process, enhancing judicial efficiency, and fostering a more equitable legal landscape, ultimately alleviating the strain imposed by mounting case backlogs. Our best-performing model with XLNet pre-trained embeddings as its features gives the macro F1 score of 75% for the LJP task. For link prediction, the same set of features is the best performing giving ROC of more than 80%

{{</citation>}}


### (43/160) Agri-GNN: A Novel Genotypic-Topological Graph Neural Network Framework Built on GraphSAGE for Optimized Yield Prediction (Aditya Gupta et al., 2023)

{{<citation>}}

Aditya Gupta, Asheesh Singh. (2023)  
**Agri-GNN: A Novel Genotypic-Topological Graph Neural Network Framework Built on GraphSAGE for Optimized Yield Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.13037v1)  

---


**ABSTRACT**  
Agriculture, as the cornerstone of human civilization, constantly seeks to integrate technology for enhanced productivity and sustainability. This paper introduces $\textit{Agri-GNN}$, a novel Genotypic-Topological Graph Neural Network Framework tailored to capture the intricate spatial and genotypic interactions of crops, paving the way for optimized predictions of harvest yields. $\textit{Agri-GNN}$ constructs a Graph $\mathcal{G}$ that considers farming plots as nodes, and then methodically constructs edges between nodes based on spatial and genotypic similarity, allowing for the aggregation of node information through a genotypic-topological filter. Graph Neural Networks (GNN), by design, consider the relationships between data points, enabling them to efficiently model the interconnected agricultural ecosystem. By harnessing the power of GNNs, $\textit{Agri-GNN}$ encapsulates both local and global information from plants, considering their inherent connections based on spatial proximity and shared genotypes, allowing stronger predictions to be made than traditional Machine Learning architectures. $\textit{Agri-GNN}$ is built from the GraphSAGE architecture, because of its optimal calibration with large graphs, like those of farming plots and breeding experiments. $\textit{Agri-GNN}$ experiments, conducted on a comprehensive dataset of vegetation indices, time, genotype information, and location data, demonstrate that $\textit{Agri-GNN}$ achieves an $R^2 = .876$ in yield predictions for farming fields in Iowa. The results show significant improvement over the baselines and other work in the field. $\textit{Agri-GNN}$ represents a blueprint for using advanced graph-based neural architectures to predict crop yield, providing significant improvements over baselines in the field.

{{</citation>}}


### (44/160) TabuLa: Harnessing Language Models for Tabular Data Synthesis (Zilong Zhao et al., 2023)

{{<citation>}}

Zilong Zhao, Robert Birke, Lydia Chen. (2023)  
**TabuLa: Harnessing Language Models for Tabular Data Synthesis**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.12746v1)  

---


**ABSTRACT**  
Given the ubiquitous use of tabular data in industries and the growing concerns in data privacy and security, tabular data synthesis emerges as a critical research area. The recent state-of-the-art methods show that large language models (LLMs) can be adopted to generate realistic tabular data. As LLMs pre-process tabular data as full text, they have the advantage of avoiding the curse of dimensionality associated with one-hot encoding high-dimensional data. However, their long training time and limited re-usability on new tasks prevent them from replacing exiting tabular generative models. In this paper, we propose Tabula, a tabular data synthesizer based on the language model structure. Through Tabula, we demonstrate the inherent limitation of employing pre-trained language models designed for natural language processing (NLP) in the context of tabular data synthesis. Our investigation delves into the development of a dedicated foundational model tailored specifically for tabular data synthesis. Additionally, we propose a token sequence compression strategy to significantly reduce training time while preserving the quality of synthetic data. Extensive experiments on six datasets demonstrate that using a language model structure without loading the well-trained model weights yields a better starting model for tabular data synthesis. Moreover, the Tabula model, previously trained on other tabular data, serves as an excellent foundation model for new tabular data synthesis tasks. Additionally, the token sequence compression method substantially reduces the model's training time. Results show that Tabula averagely reduces 46.2% training time per epoch comparing to current LLMs-based state-of-the-art algorithm and consistently achieves even higher synthetic data utility.

{{</citation>}}


### (45/160) Learn from the Past: A Proxy based Adversarial Defense Framework to Boost Robustness (Yaohua Liu et al., 2023)

{{<citation>}}

Yaohua Liu, Jiaxin Gao, Zhu Liu, Xianghao Jiao, Xin Fan, Risheng Liu. (2023)  
**Learn from the Past: A Proxy based Adversarial Defense Framework to Boost Robustness**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2310.12713v1)  

---


**ABSTRACT**  
In light of the vulnerability of deep learning models to adversarial samples and the ensuing security issues, a range of methods, including Adversarial Training (AT) as a prominent representative, aimed at enhancing model robustness against various adversarial attacks, have seen rapid development. However, existing methods essentially assist the current state of target model to defend against parameter-oriented adversarial attacks with explicit or implicit computation burdens, which also suffers from unstable convergence behavior due to inconsistency of optimization trajectories. Diverging from previous work, this paper reconsiders the update rule of target model and corresponding deficiency to defend based on its current state. By introducing the historical state of the target model as a proxy, which is endowed with much prior information for defense, we formulate a two-stage update rule, resulting in a general adversarial defense framework, which we refer to as `LAST' ({\bf L}earn from the P{\bf ast}). Besides, we devise a Self Distillation (SD) based defense objective to constrain the update process of the proxy model without the introduction of larger teacher models. Experimentally, we demonstrate consistent and significant performance enhancements by refining a series of single-step and multi-step AT methods (e.g., up to $\bf 9.2\%$ and $\bf 20.5\%$ improvement of Robust Accuracy (RA) on CIFAR10 and CIFAR100 datasets, respectively) across various datasets, backbones and attack modalities, and validate its ability to enhance training stability and ameliorate catastrophic overfitting issues meanwhile.

{{</citation>}}


### (46/160) On the Optimization and Generalization of Multi-head Attention (Puneesh Deora et al., 2023)

{{<citation>}}

Puneesh Deora, Rouzbeh Ghaderi, Hossein Taheri, Christos Thrampoulidis. (2023)  
**On the Optimization and Generalization of Multi-head Attention**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC, stat-ML  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.12680v1)  

---


**ABSTRACT**  
The training and generalization dynamics of the Transformer's core mechanism, namely the Attention mechanism, remain under-explored. Besides, existing analyses primarily focus on single-head attention. Inspired by the demonstrated benefits of overparameterization when training fully-connected networks, we investigate the potential optimization and generalization advantages of using multiple attention heads. Towards this goal, we derive convergence and generalization guarantees for gradient-descent training of a single-layer multi-head self-attention model, under a suitable realizability condition on the data. We then establish primitive conditions on the initialization that ensure realizability holds. Finally, we demonstrate that these conditions are satisfied for a simple tokenized-mixture model. We expect the analysis can be extended to various data-model and architecture variations.

{{</citation>}}


### (47/160) Neural networks for insurance pricing with frequency and severity data: a benchmark study from data preprocessing to technical tariff (Freek Holvoet et al., 2023)

{{<citation>}}

Freek Holvoet, Katrien Antonio, Roel Henckaerts. (2023)  
**Neural networks for insurance pricing with frequency and severity data: a benchmark study from data preprocessing to technical tariff**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-fin-RM  
Keywords: GLM  
[Paper Link](http://arxiv.org/abs/2310.12671v1)  

---


**ABSTRACT**  
Insurers usually turn to generalized linear models for modelling claim frequency and severity data. Due to their success in other fields, machine learning techniques are gaining popularity within the actuarial toolbox. Our paper contributes to the literature on frequency-severity insurance pricing with machine learning via deep learning structures. We present a benchmark study on four insurance data sets with frequency and severity targets in the presence of multiple types of input features. We compare in detail the performance of: a generalized linear model on binned input data, a gradient-boosted tree model, a feed-forward neural network (FFNN), and the combined actuarial neural network (CANN). Our CANNs combine a baseline prediction established with a GLM and GBM, respectively, with a neural network correction. We explain the data preprocessing steps with specific focus on the multiple types of input features typically present in tabular insurance data sets, such as postal codes, numeric and categorical covariates. Autoencoders are used to embed the categorical variables into the neural network and we explore their potential advantages in a frequency-severity setting. Finally, we construct global surrogate models for the neural nets' frequency and severity models. These surrogates enable the translation of the essential insights captured by the FFNNs or CANNs to GLMs. As such, a technical tariff table results that can easily be deployed in practice.

{{</citation>}}


### (48/160) WeaveNet for Approximating Two-sided Matching Problems (Shusaku Sone et al., 2023)

{{<citation>}}

Shusaku Sone, Jiaxin Ma, Atsushi Hashimoto, Naoya Chiba, Yoshitaka Ushiku. (2023)  
**WeaveNet for Approximating Two-sided Matching Problems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.12515v1)  

---


**ABSTRACT**  
Matching, a task to optimally assign limited resources under constraints, is a fundamental technology for society. The task potentially has various objectives, conditions, and constraints; however, the efficient neural network architecture for matching is underexplored. This paper proposes a novel graph neural network (GNN), \textit{WeaveNet}, designed for bipartite graphs. Since a bipartite graph is generally dense, general GNN architectures lose node-wise information by over-smoothing when deeply stacked. Such a phenomenon is undesirable for solving matching problems. WeaveNet avoids it by preserving edge-wise information while passing messages densely to reach a better solution. To evaluate the model, we approximated one of the \textit{strongly NP-hard} problems, \textit{fair stable matching}. Despite its inherent difficulties and the network's general purpose design, our model reached a comparative performance with state-of-the-art algorithms specially designed for stable matching for small numbers of agents.

{{</citation>}}


### (49/160) SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation (Chongyu Fan et al., 2023)

{{<citation>}}

Chongyu Fan, Jiancheng Liu, Yihua Zhang, Dennis Wei, Eric Wong, Sijia Liu. (2023)  
**SalUn: Empowering Machine Unlearning via Gradient-based Weight Saliency in Both Image Classification and Generation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Image Classification  
[Paper Link](http://arxiv.org/abs/2310.12508v1)  

---


**ABSTRACT**  
With evolving data regulations, machine unlearning (MU) has become an important tool for fostering trust and safety in today's AI models. However, existing MU methods focusing on data and/or weight perspectives often grapple with limitations in unlearning accuracy, stability, and cross-domain applicability. To address these challenges, we introduce the concept of 'weight saliency' in MU, drawing parallels with input saliency in model explanation. This innovation directs MU's attention toward specific model weights rather than the entire model, improving effectiveness and efficiency. The resultant method that we call saliency unlearning (SalUn) narrows the performance gap with 'exact' unlearning (model retraining from scratch after removing the forgetting dataset). To the best of our knowledge, SalUn is the first principled MU approach adaptable enough to effectively erase the influence of forgetting data, classes, or concepts in both image classification and generation. For example, SalUn yields a stability advantage in high-variance random data forgetting, e.g., with a 0.2% gap compared to exact unlearning on the CIFAR-10 dataset. Moreover, in preventing conditional diffusion models from generating harmful images, SalUn achieves nearly 100% unlearning accuracy, outperforming current state-of-the-art baselines like Erased Stable Diffusion and Forget-Me-Not.

{{</citation>}}


### (50/160) SDGym: Low-Code Reinforcement Learning Environments using System Dynamics Models (Emmanuel Klu et al., 2023)

{{<citation>}}

Emmanuel Klu, Sameer Sethi, DJ Passey, Donald Martin Jr. (2023)  
**SDGym: Low-Code Reinforcement Learning Environments using System Dynamics Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.12494v1)  

---


**ABSTRACT**  
Understanding the long-term impact of algorithmic interventions on society is vital to achieving responsible AI. Traditional evaluation strategies often fall short due to the complex, adaptive and dynamic nature of society. While reinforcement learning (RL) can be a powerful approach for optimizing decisions in dynamic settings, the difficulty of realistic environment design remains a barrier to building robust agents that perform well in practical settings. To address this issue we tap into the field of system dynamics (SD) as a complementary method that incorporates collaborative simulation model specification practices. We introduce SDGym, a low-code library built on the OpenAI Gym framework which enables the generation of custom RL environments based on SD simulation models. Through a feasibility study we validate that well specified, rich RL environments can be generated from preexisting SD models and a few lines of configuration code. We demonstrate the capabilities of the SDGym environment using an SD model of the electric vehicle adoption problem. We compare two SD simulators, PySD and BPTK-Py for parity, and train a D4PG agent using the Acme framework to showcase learning and environment interaction. Our preliminary findings underscore the dual potential of SD to improve RL environment design and for RL to improve dynamic policy discovery within SD models. By open-sourcing SDGym, the intent is to galvanize further research and promote adoption across the SD and RL communities, thereby catalyzing collaboration in this emerging interdisciplinary space.

{{</citation>}}


### (51/160) Improved Operator Learning by Orthogonal Attention (Zipeng Xiao et al., 2023)

{{<citation>}}

Zipeng Xiao, Zhongkai Hao, Bokai Lin, Zhijie Deng, Hang Su. (2023)  
**Improved Operator Learning by Orthogonal Attention**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.12487v2)  

---


**ABSTRACT**  
Neural operators, as an efficient surrogate model for learning the solutions of PDEs, have received extensive attention in the field of scientific machine learning. Among them, attention-based neural operators have become one of the mainstreams in related research. However, existing approaches overfit the limited training data due to the considerable number of parameters in the attention mechanism. To address this, we develop an orthogonal attention based on the eigendecomposition of the kernel integral operator and the neural approximation of eigenfunctions. The orthogonalization naturally poses a proper regularization effect on the resulting neural operator, which aids in resisting overfitting and boosting generalization. Experiments on six standard neural operator benchmark datasets comprising both regular and irregular geometries show that our method can outperform competing baselines with decent margins.

{{</citation>}}


### (52/160) Unmasking Transformers: A Theoretical Approach to Data Recovery via Attention Weights (Yichuan Deng et al., 2023)

{{<citation>}}

Yichuan Deng, Zhao Song, Shenghao Xie, Chiwun Yang. (2023)  
**Unmasking Transformers: A Theoretical Approach to Data Recovery via Attention Weights**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12462v1)  

---


**ABSTRACT**  
In the realm of deep learning, transformers have emerged as a dominant architecture, particularly in natural language processing tasks. However, with their widespread adoption, concerns regarding the security and privacy of the data processed by these models have arisen. In this paper, we address a pivotal question: Can the data fed into transformers be recovered using their attention weights and outputs? We introduce a theoretical framework to tackle this problem. Specifically, we present an algorithm that aims to recover the input data $X \in \mathbb{R}^{d \times n}$ from given attention weights $W = QK^\top \in \mathbb{R}^{d \times d}$ and output $B \in \mathbb{R}^{n \times n}$ by minimizing the loss function $L(X)$. This loss function captures the discrepancy between the expected output and the actual output of the transformer. Our findings have significant implications for the Localized Layer-wise Mechanism (LLM), suggesting potential vulnerabilities in the model's design from a security and privacy perspective. This work underscores the importance of understanding and safeguarding the internal workings of transformers to ensure the confidentiality of processed data.

{{</citation>}}


### (53/160) MuseGNN: Interpretable and Convergent Graph Neural Network Layers at Scale (Haitian Jiang et al., 2023)

{{<citation>}}

Haitian Jiang, Renjie Liu, Xiao Yan, Zhenkun Cai, Minjie Wang, David Wipf. (2023)  
**MuseGNN: Interpretable and Convergent Graph Neural Network Layers at Scale**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2310.12457v1)  

---


**ABSTRACT**  
Among the many variants of graph neural network (GNN) architectures capable of modeling data with cross-instance relations, an important subclass involves layers designed such that the forward pass iteratively reduces a graph-regularized energy function of interest. In this way, node embeddings produced at the output layer dually serve as both predictive features for solving downstream tasks (e.g., node classification) and energy function minimizers that inherit desirable inductive biases and interpretability. However, scaling GNN architectures constructed in this way remains challenging, in part because the convergence of the forward pass may involve models with considerable depth. To tackle this limitation, we propose a sampling-based energy function and scalable GNN layers that iteratively reduce it, guided by convergence guarantees in certain settings. We also instantiate a full GNN architecture based on these designs, and the model achieves competitive accuracy and scalability when applied to the largest publicly-available node classification benchmark exceeding 1TB in size.

{{</citation>}}


### (54/160) MTS-LOF: Medical Time-Series Representation Learning via Occlusion-Invariant Features (Huayu Li et al., 2023)

{{<citation>}}

Huayu Li, Ana S. Carreon-Rascon, Xiwen Chen, Geng Yuan, Ao Li. (2023)  
**MTS-LOF: Medical Time-Series Representation Learning via Occlusion-Invariant Features**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2310.12451v1)  

---


**ABSTRACT**  
Medical time series data are indispensable in healthcare, providing critical insights for disease diagnosis, treatment planning, and patient management. The exponential growth in data complexity, driven by advanced sensor technologies, has presented challenges related to data labeling. Self-supervised learning (SSL) has emerged as a transformative approach to address these challenges, eliminating the need for extensive human annotation. In this study, we introduce a novel framework for Medical Time Series Representation Learning, known as MTS-LOF. MTS-LOF leverages the strengths of contrastive learning and Masked Autoencoder (MAE) methods, offering a unique approach to representation learning for medical time series data. By combining these techniques, MTS-LOF enhances the potential of healthcare applications by providing more sophisticated, context-rich representations. Additionally, MTS-LOF employs a multi-masking strategy to facilitate occlusion-invariant feature learning. This approach allows the model to create multiple views of the data by masking portions of it. By minimizing the discrepancy between the representations of these masked patches and the fully visible patches, MTS-LOF learns to capture rich contextual information within medical time series datasets. The results of experiments conducted on diverse medical time series datasets demonstrate the superiority of MTS-LOF over other methods. These findings hold promise for significantly enhancing healthcare applications by improving representation learning. Furthermore, our work delves into the integration of joint-embedding SSL and MAE techniques, shedding light on the intricate interplay between temporal and structural dependencies in healthcare data. This understanding is crucial, as it allows us to grasp the complexities of healthcare data analysis.

{{</citation>}}


### (55/160) CAT: Closed-loop Adversarial Training for Safe End-to-End Driving (Linrui Zhang et al., 2023)

{{<citation>}}

Linrui Zhang, Zhenghao Peng, Quanyi Li, Bolei Zhou. (2023)  
**CAT: Closed-loop Adversarial Training for Safe End-to-End Driving**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2310.12432v1)  

---


**ABSTRACT**  
Driving safety is a top priority for autonomous vehicles. Orthogonal to prior work handling accident-prone traffic events by algorithm designs at the policy level, we investigate a Closed-loop Adversarial Training (CAT) framework for safe end-to-end driving in this paper through the lens of environment augmentation. CAT aims to continuously improve the safety of driving agents by training the agent on safety-critical scenarios that are dynamically generated over time. A novel resampling technique is developed to turn log-replay real-world driving scenarios into safety-critical ones via probabilistic factorization, where the adversarial traffic generation is modeled as the multiplication of standard motion prediction sub-problems. Consequently, CAT can launch more efficient physical attacks compared to existing safety-critical scenario generation methods and yields a significantly less computational cost in the iterative learning pipeline. We incorporate CAT into the MetaDrive simulator and validate our approach on hundreds of driving scenarios imported from real-world driving datasets. Experimental results demonstrate that CAT can effectively generate adversarial scenarios countering the agent being trained. After training, the agent can achieve superior driving safety in both log-replay and safety-critical traffic scenarios on the held-out test set. Code and data are available at https://metadriverse.github.io/cat.

{{</citation>}}


### (56/160) Detecting and Mitigating Algorithmic Bias in Binary Classification using Causal Modeling (Wendy Hui et al., 2023)

{{<citation>}}

Wendy Hui, Wai Kwong Lau. (2023)  
**Detecting and Mitigating Algorithmic Bias in Binary Classification using Causal Modeling**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.12421v1)  

---


**ABSTRACT**  
This paper proposes the use of causal modeling to detect and mitigate algorithmic bias. We provide a brief description of causal modeling and a general overview of our approach. We then use the Adult dataset, which is available for download from the UC Irvine Machine Learning Repository, to develop (1) a prediction model, which is treated as a black box, and (2) a causal model for bias mitigation. In this paper, we focus on gender bias and the problem of binary classification. We show that gender bias in the prediction model is statistically significant at the 0.05 level. We demonstrate the effectiveness of the causal model in mitigating gender bias by cross-validation. Furthermore, we show that the overall classification accuracy is improved slightly. Our novel approach is intuitive, easy-to-use, and can be implemented using existing statistical software tools such as "lavaan" in R. Hence, it enhances explainability and promotes trust.

{{</citation>}}


### (57/160) Cooperative Minibatching in Graph Neural Networks (Muhammed Fatih Balin et al., 2023)

{{<citation>}}

Muhammed Fatih Balin, Dominique LaSalle, Ümit V. Çatalyürek. (2023)  
**Cooperative Minibatching in Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.12403v2)  

---


**ABSTRACT**  
Significant computational resources are required to train Graph Neural Networks (GNNs) at a large scale, and the process is highly data-intensive. One of the most effective ways to reduce resource requirements is minibatch training coupled with graph sampling. GNNs have the unique property that items in a minibatch have overlapping data. However, the commonly implemented Independent Minibatching approach assigns each Processing Element (PE) its own minibatch to process, leading to duplicated computations and input data access across PEs. This amplifies the Neighborhood Explosion Phenomenon (NEP), which is the main bottleneck limiting scaling. To reduce the effects of NEP in the multi-PE setting, we propose a new approach called Cooperative Minibatching. Our approach capitalizes on the fact that the size of the sampled subgraph is a concave function of the batch size, leading to significant reductions in the amount of work per seed vertex as batch sizes increase. Hence, it is favorable for processors equipped with a fast interconnect to work on a large minibatch together as a single larger processor, instead of working on separate smaller minibatches, even though global batch size is identical. We also show how to take advantage of the same phenomenon in serial execution by generating dependent consecutive minibatches. Our experimental evaluations show up to 4x bandwidth savings for fetching vertex embeddings, by simply increasing this dependency without harming model convergence. Combining our proposed approaches, we achieve up to 64% speedup over Independent Minibatching on single-node multi-GPU systems.

{{</citation>}}


## cs.AI (5)



### (58/160) The opaque law of artificial intelligence (Vincenzo Calderonio, 2023)

{{<citation>}}

Vincenzo Calderonio. (2023)  
**The opaque law of artificial intelligence**  

---
Primary Category: cs.AI  
Categories: F-0; I-2; J-4; K-4; K-5, cs-AI, cs.AI  
Keywords: AI, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2310.13192v1)  

---


**ABSTRACT**  
The purpose of this paper is to analyse the opacity of algorithms, contextualized in the open debate on responsibility for artificial intelligence causation; with an experimental approach by which, applying the proposed conversational methodology of the Turing Test, we expect to evaluate the performance of one of the best existing NLP model of generative AI (Chat-GPT) to see how far it can go right now and how the shape of a legal regulation of it could be. The analysis of the problem will be supported by a comment of Italian classical law categories such as causality, intent and fault to understand the problem of the usage of AI, focusing in particular on the human-machine interaction. On the computer science side, for a technical point of view of the logic used to craft these algorithms, in the second chapter will be proposed a practical interrogation of Chat-GPT aimed at finding some critical points of the functioning of AI. The end of the paper will concentrate on some existing legal solutions which can be applied to the problem, plus a brief description of the approach proposed by EU Artificial Intelligence act.

{{</citation>}}


### (59/160) Safe RLHF: Safe Reinforcement Learning from Human Feedback (Josef Dai et al., 2023)

{{<citation>}}

Josef Dai, Xuehai Pan, Ruiyang Sun, Jiaming Ji, Xinbo Xu, Mickel Liu, Yizhou Wang, Yaodong Yang. (2023)  
**Safe RLHF: Safe Reinforcement Learning from Human Feedback**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.12773v1)  

---


**ABSTRACT**  
With the development of large language models (LLMs), striking a balance between the performance and safety of AI systems has never been more critical. However, the inherent tension between the objectives of helpfulness and harmlessness presents a significant challenge during LLM training. To address this issue, we propose Safe Reinforcement Learning from Human Feedback (Safe RLHF), a novel algorithm for human value alignment. Safe RLHF explicitly decouples human preferences regarding helpfulness and harmlessness, effectively avoiding the crowdworkers' confusion about the tension and allowing us to train separate reward and cost models. We formalize the safety concern of LLMs as an optimization task of maximizing the reward function while satisfying specified cost constraints. Leveraging the Lagrangian method to solve this constrained problem, Safe RLHF dynamically adjusts the balance between the two objectives during fine-tuning. Through a three-round fine-tuning using Safe RLHF, we demonstrate a superior ability to mitigate harmful responses while enhancing model performance compared to existing value-aligned algorithms. Experimentally, we fine-tuned the Alpaca-7B using Safe RLHF and aligned it with collected human preferences, significantly improving its helpfulness and harmlessness according to human evaluations.

{{</citation>}}


### (60/160) PSYCHIC: A Neuro-Symbolic Framework for Knowledge Graph Question-Answering Grounding (Hanna Abi Akl, 2023)

{{<citation>}}

Hanna Abi Akl. (2023)  
**PSYCHIC: A Neuro-Symbolic Framework for Knowledge Graph Question-Answering Grounding**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Knowledge Graph, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.12638v1)  

---


**ABSTRACT**  
The Scholarly Question Answering over Linked Data (Scholarly QALD) at The International Semantic Web Conference (ISWC) 2023 challenge presents two sub-tasks to tackle question answering (QA) over knowledge graphs (KGs). We answer the KGQA over DBLP (DBLP-QUAD) task by proposing a neuro-symbolic (NS) framework based on PSYCHIC, an extractive QA model capable of identifying the query and entities related to a KG question. Our system achieved a F1 score of 00.18% on question answering and came in third place for entity linking (EL) with a score of 71.00%.

{{</citation>}}


### (61/160) Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark (Jiaming Ji et al., 2023)

{{<citation>}}

Jiaming Ji, Borong Zhang, Jiayi Zhou, Xuehai Pan, Weidong Huang, Ruiyang Sun, Yiran Geng, Yifan Zhong, Juntao Dai, Yaodong Yang. (2023)  
**Safety-Gymnasium: A Unified Safe Reinforcement Learning Benchmark**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.12567v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) systems possess significant potential to drive societal progress. However, their deployment often faces obstacles due to substantial safety concerns. Safe reinforcement learning (SafeRL) emerges as a solution to optimize policies while simultaneously adhering to multiple constraints, thereby addressing the challenge of integrating reinforcement learning in safety-critical scenarios. In this paper, we present an environment suite called Safety-Gymnasium, which encompasses safety-critical tasks in both single and multi-agent scenarios, accepting vector and vision-only input. Additionally, we offer a library of algorithms named Safe Policy Optimization (SafePO), comprising 16 state-of-the-art SafeRL algorithms. This comprehensive library can serve as a validation tool for the research community. By introducing this benchmark, we aim to facilitate the evaluation and comparison of safety performance, thus fostering the development of reinforcement learning for safer, more reliable, and responsible real-world applications. The website of this project can be accessed at https://sites.google.com/view/safety-gymnasium.

{{</citation>}}


### (62/160) GPT-4 Doesn't Know It's Wrong: An Analysis of Iterative Prompting for Reasoning Problems (Kaya Stechly et al., 2023)

{{<citation>}}

Kaya Stechly, Matthew Marquez, Subbarao Kambhampati. (2023)  
**GPT-4 Doesn't Know It's Wrong: An Analysis of Iterative Prompting for Reasoning Problems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.12397v1)  

---


**ABSTRACT**  
There has been considerable divergence of opinion on the reasoning abilities of Large Language Models (LLMs). While the initial optimism that reasoning might emerge automatically with scale has been tempered thanks to a slew of counterexamples, a wide spread belief in their iterative self-critique capabilities persists. In this paper, we set out to systematically investigate the effectiveness of iterative prompting of LLMs in the context of Graph Coloring, a canonical NP-complete reasoning problem that is related to propositional satisfiability as well as practical problems like scheduling and allocation. We present a principled empirical study of the performance of GPT4 in solving graph coloring instances or verifying the correctness of candidate colorings. In iterative modes, we experiment with the model critiquing its own answers and an external correct reasoner verifying proposed solutions. In both cases, we analyze whether the content of the criticisms actually affects bottom line performance. The study seems to indicate that (i) LLMs are bad at solving graph coloring instances (ii) they are no better at verifying a solution--and thus are not effective in iterative modes with LLMs critiquing LLM-generated solutions (iii) the correctness and content of the criticisms--whether by LLMs or external solvers--seems largely irrelevant to the performance of iterative prompting. We show that the observed increase in effectiveness is largely due to the correct solution being fortuitously present in the top-k completions of the prompt (and being recognized as such by an external verifier). Our results thus call into question claims about the self-critiquing capabilities of state of the art LLMs.

{{</citation>}}


## cs.CL (59)



### (63/160) Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models (Jianwei Li et al., 2023)

{{<citation>}}

Jianwei Li, Qi Lei, Wei Cheng, Dongkuan Xu. (2023)  
**Towards Robust Pruning: An Adaptive Knowledge-Retention Pruning Strategy for Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model, Pruning  
[Paper Link](http://arxiv.org/abs/2310.13191v1)  

---


**ABSTRACT**  
The pruning objective has recently extended beyond accuracy and sparsity to robustness in language models. Despite this, existing methods struggle to enhance robustness against adversarial attacks when continually increasing model sparsity and require a retraining process. As humans step into the era of large language models, these issues become increasingly prominent. This paper proposes that the robustness of language models is proportional to the extent of pre-trained knowledge they encompass. Accordingly, we introduce a post-training pruning strategy designed to faithfully replicate the embedding space and feature space of dense language models, aiming to conserve more pre-trained knowledge during the pruning process. In this setup, each layer's reconstruction error not only originates from itself but also includes cumulative error from preceding layers, followed by an adaptive rectification. Compared to other state-of-art baselines, our approach demonstrates a superior balance between accuracy, sparsity, robustness, and pruning cost with BERT on datasets SST2, IMDB, and AGNews, marking a significant stride towards robust pruning in language models.

{{</citation>}}


### (64/160) Fast and Accurate Factual Inconsistency Detection Over Long Documents (Barrett Martin Lattimer et al., 2023)

{{<citation>}}

Barrett Martin Lattimer, Patrick Chen, Xinyuan Zhang, Yi Yang. (2023)  
**Fast and Accurate Factual Inconsistency Detection Over Long Documents**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Generative AI, NLI, Natural Language Inference  
[Paper Link](http://arxiv.org/abs/2310.13189v2)  

---


**ABSTRACT**  
Generative AI models exhibit remarkable potential; however, hallucinations across various tasks present a significant challenge, particularly for longer inputs that current approaches struggle to address effectively. We introduce SCALE (Source Chunking Approach for Large-scale inconsistency Evaluation), a task-agnostic model for detecting factual inconsistencies using a novel chunking strategy. Specifically, SCALE is a Natural Language Inference (NLI) based model that uses large text chunks to condition over long texts. This approach achieves state-of-the-art performance in factual inconsistency detection for diverse tasks and long inputs. Additionally, we leverage the chunking mechanism and employ a novel algorithm to explain SCALE's decisions through relevant source sentence retrieval. Our evaluations reveal that SCALE outperforms existing methods on both standard benchmarks and a new long-form dialogue dataset ScreenEval we constructed. Moreover, SCALE surpasses competitive systems in efficiency and model explanation evaluations. We have released our code and data publicly to GitHub.

{{</citation>}}


### (65/160) CLIFT: Analysing Natural Distribution Shift on Question Answering Models in Clinical Domain (Ankit Pal, 2023)

{{<citation>}}

Ankit Pal. (2023)  
**CLIFT: Analysing Natural Distribution Shift on Question Answering Models in Clinical Domain**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-DC, cs-LG, cs.CL  
Keywords: Clinical, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.13146v1)  

---


**ABSTRACT**  
This paper introduces a new testbed CLIFT (Clinical Shift) for the clinical domain Question-answering task. The testbed includes 7.5k high-quality question answering samples to provide a diverse and reliable benchmark. We performed a comprehensive experimental study and evaluated several QA deep-learning models under the proposed testbed. Despite impressive results on the original test set, the performance degrades when applied to new test sets, which shows the distribution shift. Our findings emphasize the need for and the potential for increasing the robustness of clinical domain models under distributional shifts. The testbed offers one way to track progress in that direction. It also highlights the necessity of adopting evaluation metrics that consider robustness to natural distribution shifts. We plan to expand the corpus by adding more samples and model results. The full paper and the updated benchmark are available at github.com/openlifescience-ai/clift

{{</citation>}}


### (66/160) Better to Ask in English: Cross-Lingual Evaluation of Large Language Models for Healthcare Queries (Yiqiao Jin et al., 2023)

{{<citation>}}

Yiqiao Jin, Mohit Chandra, Gaurav Verma, Yibo Hu, Munmun De Choudhury, Srijan Kumar. (2023)  
**Better to Ask in English: Cross-Lingual Evaluation of Large Language Models for Healthcare Queries**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.13132v2)  

---


**ABSTRACT**  
Large language models (LLMs) are transforming the ways the general public accesses and consumes information. Their influence is particularly pronounced in pivotal sectors like healthcare, where lay individuals are increasingly appropriating LLMs as conversational agents for everyday queries. While LLMs demonstrate impressive language understanding and generation proficiencies, concerns regarding their safety remain paramount in these high-stake domains. Moreover, the development of LLMs is disproportionately focused on English. It remains unclear how these LLMs perform in the context of non-English languages, a gap that is critical for ensuring equity in the real-world use of these systems.This paper provides a framework to investigate the effectiveness of LLMs as multi-lingual dialogue systems for healthcare queries. Our empirically-derived framework XlingEval focuses on three fundamental criteria for evaluating LLM responses to naturalistic human-authored health-related questions: correctness, consistency, and verifiability. Through extensive experiments on four major global languages, including English, Spanish, Chinese, and Hindi, spanning three expert-annotated large health Q&A datasets, and through an amalgamation of algorithmic and human-evaluation strategies, we found a pronounced disparity in LLM responses across these languages, indicating a need for enhanced cross-lingual capabilities. We further propose XlingHealth, a cross-lingual benchmark for examining the multilingual capabilities of LLMs in the healthcare context. Our findings underscore the pressing need to bolster the cross-lingual capacities of these models, and to provide an equitable information ecosystem accessible to all.

{{</citation>}}


### (67/160) Auto-Instruct: Automatic Instruction Generation and Ranking for Black-Box Language Models (Zhihan Zhang et al., 2023)

{{<citation>}}

Zhihan Zhang, Shuohang Wang, Wenhao Yu, Yichong Xu, Dan Iter, Qingkai Zeng, Yang Liu, Chenguang Zhu, Meng Jiang. (2023)  
**Auto-Instruct: Automatic Instruction Generation and Ranking for Black-Box Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.13127v1)  

---


**ABSTRACT**  
Large language models (LLMs) can perform a wide range of tasks by following natural language instructions, without the necessity of task-specific fine-tuning. Unfortunately, the performance of LLMs is greatly influenced by the quality of these instructions, and manually writing effective instructions for each task is a laborious and subjective process. In this paper, we introduce Auto-Instruct, a novel method to automatically improve the quality of instructions provided to LLMs. Our method leverages the inherent generative ability of LLMs to produce diverse candidate instructions for a given task, and then ranks them using a scoring model trained on a variety of 575 existing NLP tasks. In experiments on 118 out-of-domain tasks, Auto-Instruct surpasses both human-written instructions and existing baselines of LLM-generated instructions. Furthermore, our method exhibits notable generalizability even with other LLMs that are not incorporated into its training process.

{{</citation>}}


### (68/160) Do Language Models Learn about Legal Entity Types during Pretraining? (Claire Barale et al., 2023)

{{<citation>}}

Claire Barale, Michael Rovatsos, Nehal Bhuta. (2023)  
**Do Language Models Learn about Legal Entity Types during Pretraining?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model, Legal, NLP, QA  
[Paper Link](http://arxiv.org/abs/2310.13092v1)  

---


**ABSTRACT**  
Language Models (LMs) have proven their ability to acquire diverse linguistic knowledge during the pretraining phase, potentially serving as a valuable source of incidental supervision for downstream tasks. However, there has been limited research conducted on the retrieval of domain-specific knowledge, and specifically legal knowledge. We propose to explore the task of Entity Typing, serving as a proxy for evaluating legal knowledge as an essential aspect of text comprehension, and a foundational task to numerous downstream legal NLP applications. Through systematic evaluation and analysis and two types of prompting (cloze sentences and QA-based templates) and to clarify the nature of these acquired cues, we compare diverse types and lengths of entities both general and domain-specific entities, semantics or syntax signals, and different LM pretraining corpus (generic and legal-oriented) and architectures (encoder BERT-based and decoder-only with Llama2). We show that (1) Llama2 performs well on certain entities and exhibits potential for substantial improvement with optimized prompt templates, (2) law-oriented LMs show inconsistent performance, possibly due to variations in their training corpus, (3) LMs demonstrate the ability to type entities even in the case of multi-token entities, (4) all models struggle with entities belonging to sub-domains of the law (5) Llama2 appears to frequently overlook syntactic cues, a shortcoming less present in BERT-based architectures.

{{</citation>}}


### (69/160) From Multilingual Complexity to Emotional Clarity: Leveraging Commonsense to Unveil Emotions in Code-Mixed Dialogues (Shivani Kumar et al., 2023)

{{<citation>}}

Shivani Kumar, Ramaneswaran S, Md Shad Akhtar, Tanmoy Chakraborty. (2023)  
**From Multilingual Complexity to Emotional Clarity: Leveraging Commonsense to Unveil Emotions in Code-Mixed Dialogues**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Emotion Recognition, Multilingual, NLP  
[Paper Link](http://arxiv.org/abs/2310.13080v1)  

---


**ABSTRACT**  
Understanding emotions during conversation is a fundamental aspect of human communication, driving NLP research for Emotion Recognition in Conversation (ERC). While considerable research has focused on discerning emotions of individual speakers in monolingual dialogues, understanding the emotional dynamics in code-mixed conversations has received relatively less attention. This motivates our undertaking of ERC for code-mixed conversations in this study. Recognizing that emotional intelligence encompasses a comprehension of worldly knowledge, we propose an innovative approach that integrates commonsense information with dialogue context to facilitate a deeper understanding of emotions. To achieve this, we devise an efficient pipeline that extracts relevant commonsense from existing knowledge graphs based on the code-mixed input. Subsequently, we develop an advanced fusion technique that seamlessly combines the acquired commonsense information with the dialogue representation obtained from a dedicated dialogue understanding module. Our comprehensive experimentation showcases the substantial performance improvement obtained through the systematic incorporation of commonsense in ERC. Both quantitative assessments and qualitative analyses further corroborate the validity of our hypothesis, reaffirming the pivotal role of commonsense integration in enhancing ERC.

{{</citation>}}


### (70/160) GARI: Graph Attention for Relative Isomorphism of Arabic Word Embeddings (Muhammad Asif Ali et al., 2023)

{{<citation>}}

Muhammad Asif Ali, Maha Alshmrani, Jianbin Qin, Yan Hu, Di Wang. (2023)  
**GARI: Graph Attention for Relative Isomorphism of Arabic Word Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Embedding, NLP, Word Embedding  
[Paper Link](http://arxiv.org/abs/2310.13068v1)  

---


**ABSTRACT**  
Bilingual Lexical Induction (BLI) is a core challenge in NLP, it relies on the relative isomorphism of individual embedding spaces. Existing attempts aimed at controlling the relative isomorphism of different embedding spaces fail to incorporate the impact of semantically related words in the model training objective. To address this, we propose GARI that combines the distributional training objectives with multiple isomorphism losses guided by the graph attention network. GARI considers the impact of semantical variations of words in order to define the relative isomorphism of the embedding spaces. Experimental evaluation using the Arabic language data set shows that GARI outperforms the existing research by improving the average P@1 by a relative score of up to 40.95% and 76.80% for in-domain and domain mismatch settings respectively. We release the codes for GARI at https://github.com/asif6827/GARI.

{{</citation>}}


### (71/160) AutoMix: Automatically Mixing Language Models (Aman Madaan et al., 2023)

{{<citation>}}

Aman Madaan, Pranjal Aggarwal, Ankit Anand, Srividya Pranavi Potharaju, Swaroop Mishra, Pei Zhou, Aditya Gupta, Dheeraj Rajagopal, Karthik Kappaganthu, Yiming Yang, Shyam Upadhyay, Mausam, Manaal Faruqui. (2023)  
**AutoMix: Automatically Mixing Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12963v1)  

---


**ABSTRACT**  
Large language models (LLMs) are now available in various sizes and configurations from cloud API providers. While this diversity offers a broad spectrum of choices, effectively leveraging the options to optimize computational cost and performance remains challenging. In this work, we present AutoMix, an approach that strategically routes queries to larger LMs, based on the approximate correctness of outputs from a smaller LM. Central to AutoMix is a few-shot self-verification mechanism, which estimates the reliability of its own outputs without requiring training. Given that verifications can be noisy, we employ a meta verifier in AutoMix to refine the accuracy of these assessments. Our experiments using LLAMA2-13/70B, on five context-grounded reasoning datasets demonstrate that AutoMix surpasses established baselines, improving the incremental benefit per cost by up to 89%. Our code and data are available at https://github.com/automix-llm/automix.

{{</citation>}}


### (72/160) An Emulator for Fine-Tuning Large Language Models using Small Language Models (Eric Mitchell et al., 2023)

{{<citation>}}

Eric Mitchell, Rafael Rafailov, Archit Sharma, Chelsea Finn, Christopher D. Manning. (2023)  
**An Emulator for Fine-Tuning Large Language Models using Small Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Falcon, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12962v1)  

---


**ABSTRACT**  
Widely used language models (LMs) are typically built by scaling up a two-stage training pipeline: a pre-training stage that uses a very large, diverse dataset of text and a fine-tuning (sometimes, 'alignment') stage that uses targeted examples or other specifications of desired behaviors. While it has been hypothesized that knowledge and skills come from pre-training, and fine-tuning mostly filters this knowledge and skillset, this intuition has not been extensively tested. To aid in doing so, we introduce a novel technique for decoupling the knowledge and skills gained in these two stages, enabling a direct answer to the question, "What would happen if we combined the knowledge learned by a large model during pre-training with the knowledge learned by a small model during fine-tuning (or vice versa)?" Using an RL-based framework derived from recent developments in learning from human preferences, we introduce emulated fine-tuning (EFT), a principled and practical method for sampling from a distribution that approximates (or 'emulates') the result of pre-training and fine-tuning at different scales. Our experiments with EFT show that scaling up fine-tuning tends to improve helpfulness, while scaling up pre-training tends to improve factuality. Beyond decoupling scale, we show that EFT enables test-time adjustment of competing behavioral traits like helpfulness and harmlessness without additional training. Finally, a special case of emulated fine-tuning, which we call LM up-scaling, avoids resource-intensive fine-tuning of large pre-trained models by ensembling them with small fine-tuned models, essentially emulating the result of fine-tuning the large pre-trained model. Up-scaling consistently improves helpfulness and factuality of instruction-following models in the Llama, Llama-2, and Falcon families, without additional hyperparameters or training.

{{</citation>}}


### (73/160) SEGO: Sequential Subgoal Optimization for Mathematical Problem-Solving (Xueliang Zhao et al., 2023)

{{<citation>}}

Xueliang Zhao, Xinting Huang, Wei Bi, Lingpeng Kong. (2023)  
**SEGO: Sequential Subgoal Optimization for Mathematical Problem-Solving**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12960v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have driven substantial progress in artificial intelligence in recent years, exhibiting impressive capabilities across a wide range of tasks, including mathematical problem-solving. Inspired by the success of subgoal-based methods, we propose a novel framework called \textbf{SE}quential sub\textbf{G}oal \textbf{O}ptimization (SEGO) to enhance LLMs' ability to solve mathematical problems. By establishing a connection between the subgoal breakdown process and the probability of solving problems, SEGO aims to identify better subgoals with theoretical guarantees. Addressing the challenge of identifying suitable subgoals in a large solution space, our framework generates problem-specific subgoals and adjusts them according to carefully designed criteria. Incorporating these optimized subgoals into the policy model training leads to significant improvements in problem-solving performance. We validate SEGO's efficacy through experiments on two benchmarks, GSM8K and MATH, where our approach outperforms existing methods, highlighting the potential of SEGO in AI-driven mathematical problem-solving.   Data and code associated with this paper will be available at https://github.com/zhaoxlpku/SEGO

{{</citation>}}


### (74/160) On the Representational Capacity of Recurrent Neural Language Models (Franz Nowak et al., 2023)

{{<citation>}}

Franz Nowak, Anej Svete, Li Du, Ryan Cotterell. (2023)  
**On the Representational Capacity of Recurrent Neural Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12942v2)  

---


**ABSTRACT**  
This work investigates the computational expressivity of language models (LMs) based on recurrent neural networks (RNNs). Siegelmann and Sontag (1992) famously showed that RNNs with rational weights and hidden states and unbounded computation time are Turing complete. However, LMs define weightings over strings in addition to just (unweighted) language membership and the analysis of the computational power of RNN LMs (RLMs) should reflect this. We extend the Turing completeness result to the probabilistic case, showing how a rationally weighted RLM with unbounded computation time can simulate any probabilistic Turing machine (PTM). Since, in practice, RLMs work in real-time, processing a symbol at every time step, we treat the above result as an upper bound on the expressivity of RLMs. We also provide a lower bound by showing that under the restriction to real-time computation, such models can simulate deterministic real-time rational PTMs.

{{</citation>}}


### (75/160) A Predictive Factor Analysis of Social Biases and Task-Performance in Pretrained Masked Language Models (Yi Zhou et al., 2023)

{{<citation>}}

Yi Zhou, Jose Camacho-Collados, Danushka Bollegala. (2023)  
**A Predictive Factor Analysis of Social Biases and Task-Performance in Pretrained Masked Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12936v2)  

---


**ABSTRACT**  
Various types of social biases have been reported with pretrained Masked Language Models (MLMs) in prior work. However, multiple underlying factors are associated with an MLM such as its model size, size of the training data, training objectives, the domain from which pretraining data is sampled, tokenization, and languages present in the pretrained corpora, to name a few. It remains unclear as to which of those factors influence social biases that are learned by MLMs. To study the relationship between model factors and the social biases learned by an MLM, as well as the downstream task performance of the model, we conduct a comprehensive study over 39 pretrained MLMs covering different model sizes, training objectives, tokenization methods, training data domains and languages. Our results shed light on important factors often neglected in prior literature, such as tokenization or model objectives.

{{</citation>}}


### (76/160) Experimental Narratives: A Comparison of Human Crowdsourced Storytelling and AI Storytelling (Nina Begus, 2023)

{{<citation>}}

Nina Begus. (2023)  
**Experimental Narratives: A Comparison of Human Crowdsourced Storytelling and AI Storytelling**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.12902v1)  

---


**ABSTRACT**  
The paper proposes a framework that combines behavioral and computational experiments employing fictional prompts as a novel tool for investigating cultural artifacts and social biases in storytelling both by humans and generative AI. The study analyzes 250 stories authored by crowdworkers in June 2019 and 80 stories generated by GPT-3.5 and GPT-4 in March 2023 by merging methods from narratology and inferential statistics. Both crowdworkers and large language models responded to identical prompts about creating and falling in love with an artificial human. The proposed experimental paradigm allows a direct comparison between human and LLM-generated storytelling. Responses to the Pygmalionesque prompts confirm the pervasive presence of the Pygmalion myth in the collective imaginary of both humans and large language models. All solicited narratives present a scientific or technological pursuit. The analysis reveals that narratives from GPT-3.5 and particularly GPT-4 are more more progressive in terms of gender roles and sexuality than those written by humans. While AI narratives can occasionally provide innovative plot twists, they offer less imaginative scenarios and rhetoric than human-authored texts. The proposed framework argues that fiction can be used as a window into human and AI-based collective imaginary and social dimensions.

{{</citation>}}


### (77/160) A Systematic Study of Performance Disparities in Multilingual Task-Oriented Dialogue Systems (Songbo Hu et al., 2023)

{{<citation>}}

Songbo Hu, Han Zhou, Moy Yuan, Milan Gritta, Guchun Zhang, Ignacio Iacobacci, Anna Korhonen, Ivan Vulić. (2023)  
**A Systematic Study of Performance Disparities in Multilingual Task-Oriented Dialogue Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Multilingual, NLP  
[Paper Link](http://arxiv.org/abs/2310.12892v1)  

---


**ABSTRACT**  
Achieving robust language technologies that can perform well across the world's many languages is a central goal of multilingual NLP. In this work, we take stock of and empirically analyse task performance disparities that exist between multilingual task-oriented dialogue (ToD) systems. We first define new quantitative measures of absolute and relative equivalence in system performance, capturing disparities across languages and within individual languages. Through a series of controlled experiments, we demonstrate that performance disparities depend on a number of factors: the nature of the ToD task at hand, the underlying pretrained language model, the target language, and the amount of ToD annotated data. We empirically prove the existence of the adaptation and intrinsic biases in current ToD systems: e.g., ToD systems trained for Arabic or Turkish using annotated ToD data fully parallel to English ToD data still exhibit diminished ToD task performance. Beyond providing a series of insights into the performance disparities of ToD systems in different languages, our analyses offer practical tips on how to approach ToD data collection and system development for new languages.

{{</citation>}}


### (78/160) StoryAnalogy: Deriving Story-level Analogies from Large Language Models to Unlock Analogical Understanding (Cheng Jiayang et al., 2023)

{{<citation>}}

Cheng Jiayang, Lin Qiu, Tsz Ho Chan, Tianqing Fang, Weiqi Wang, Chunkit Chan, Dongyu Ru, Qipeng Guo, Hongming Zhang, Yangqiu Song, Yue Zhang, Zheng Zhang. (2023)  
**StoryAnalogy: Deriving Story-level Analogies from Large Language Models to Unlock Analogical Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2310.12874v2)  

---


**ABSTRACT**  
Analogy-making between narratives is crucial for human reasoning. In this paper, we evaluate the ability to identify and generate analogies by constructing a first-of-its-kind large-scale story-level analogy corpus, \textsc{StoryAnalogy}, which contains 24K story pairs from diverse domains with human annotations on two similarities from the extended Structure-Mapping Theory. We design a set of tests on \textsc{StoryAnalogy}, presenting the first evaluation of story-level analogy identification and generation. Interestingly, we find that the analogy identification tasks are incredibly difficult not only for sentence embedding models but also for the recent large language models (LLMs) such as ChatGPT and LLaMa. ChatGPT, for example, only achieved around 30% accuracy in multiple-choice questions (compared to over 85% accuracy for humans). Furthermore, we observe that the data in \textsc{StoryAnalogy} can improve the quality of analogy generation in LLMs, where a fine-tuned FlanT5-xxl model achieves comparable performance to zero-shot ChatGPT.

{{</citation>}}


### (79/160) The Locality and Symmetry of Positional Encodings (Lihu Chen et al., 2023)

{{<citation>}}

Lihu Chen, Gaël Varoquaux, Fabian M. Suchanek. (2023)  
**The Locality and Symmetry of Positional Encodings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12864v1)  

---


**ABSTRACT**  
Positional Encodings (PEs) are used to inject word-order information into transformer-based language models. While they can significantly enhance the quality of sentence representations, their specific contribution to language models is not fully understood, especially given recent findings that various positional encodings are insensitive to word order. In this work, we conduct a systematic study of positional encodings in \textbf{Bidirectional Masked Language Models} (BERT-style) , which complements existing work in three aspects: (1) We uncover the core function of PEs by identifying two common properties, Locality and Symmetry; (2) We show that the two properties are closely correlated with the performances of downstream tasks; (3) We quantify the weakness of current PEs by introducing two new probing tasks, on which current PEs perform poorly. We believe that these results are the basis for developing better PEs for transformer-based language models. The code is available at \faGithub~ \url{https://github.com/tigerchen52/locality\_symmetry}

{{</citation>}}


### (80/160) Probing LLMs for hate speech detection: strengths and vulnerabilities (Sarthak Roy et al., 2023)

{{<citation>}}

Sarthak Roy, Ashish Harshavardhan, Animesh Mukherjee, Punyajoy Saha. (2023)  
**Probing LLMs for hate speech detection: strengths and vulnerabilities**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: GPT, GPT-3.5, T5  
[Paper Link](http://arxiv.org/abs/2310.12860v1)  

---


**ABSTRACT**  
Recently efforts have been made by social media platforms as well as researchers to detect hateful or toxic language using large language models. However, none of these works aim to use explanation, additional context and victim community information in the detection process. We utilise different prompt variation, input information and evaluate large language models in zero shot setting (without adding any in-context examples). We select three large language models (GPT-3.5, text-davinci and Flan-T5) and three datasets - HateXplain, implicit hate and ToxicSpans. We find that on average including the target information in the pipeline improves the model performance substantially (~20-30%) over the baseline across the datasets. There is also a considerable effect of adding the rationales/explanations into the pipeline (~10-20%) over the baseline across the datasets. In addition, we further provide a typology of the error cases where these large language models fail to (i) classify and (ii) explain the reason for the decisions they take. Such vulnerable points automatically constitute 'jailbreak' prompts for these models and industry scale safeguard techniques need to be developed to make the models robust against such prompts.

{{</citation>}}


### (81/160) Knowledge-Augmented Language Model Verification (Jinheon Baek et al., 2023)

{{<citation>}}

Jinheon Baek, Soyeong Jeong, Minki Kang, Jong C. Park, Sung Ju Hwang. (2023)  
**Knowledge-Augmented Language Model Verification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12836v1)  

---


**ABSTRACT**  
Recent Language Models (LMs) have shown impressive capabilities in generating texts with the knowledge internalized in parameters. Yet, LMs often generate the factually incorrect responses to the given queries, since their knowledge may be inaccurate, incomplete, and outdated. To address this problem, previous works propose to augment LMs with the knowledge retrieved from an external knowledge source. However, such approaches often show suboptimal text generation performance due to two reasons: 1) the model may fail to retrieve the knowledge relevant to the given query, or 2) the model may not faithfully reflect the retrieved knowledge in the generated text. To overcome these, we propose to verify the output and the knowledge of the knowledge-augmented LMs with a separate verifier, which is a small LM that is trained to detect those two types of errors through instruction-finetuning. Then, when the verifier recognizes an error, we can rectify it by either retrieving new knowledge or generating new text. Further, we use an ensemble of the outputs from different instructions with a single verifier to enhance the reliability of the verification processes. We validate the effectiveness of the proposed verification steps on multiple question answering benchmarks, whose results show that the proposed verifier effectively identifies retrieval and generation errors, allowing LMs to provide more factually correct outputs. Our code is available at https://github.com/JinheonBaek/KALMV.

{{</citation>}}


### (82/160) AgentTuning: Enabling Generalized Agent Abilities for LLMs (Aohan Zeng et al., 2023)

{{<citation>}}

Aohan Zeng, Mingdao Liu, Rui Lu, Bowen Wang, Xiao Liu, Yuxiao Dong, Jie Tang. (2023)  
**AgentTuning: Enabling Generalized Agent Abilities for LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.12823v2)  

---


**ABSTRACT**  
Open large language models (LLMs) with great performance in various tasks have significantly advanced the development of LLMs. However, they are far inferior to commercial models such as ChatGPT and GPT-4 when acting as agents to tackle complex tasks in the real world. These agent tasks employ LLMs as the central controller responsible for planning, memorization, and tool utilization, necessitating both fine-grained prompting methods and robust LLMs to achieve satisfactory performance. Though many prompting methods have been proposed to complete particular agent tasks, there is lack of research focusing on improving the agent capabilities of LLMs themselves without compromising their general abilities. In this work, we present AgentTuning, a simple and general method to enhance the agent abilities of LLMs while maintaining their general LLM capabilities. We construct AgentInstruct, a lightweight instruction-tuning dataset containing high-quality interaction trajectories. We employ a hybrid instruction-tuning strategy by combining AgentInstruct with open-source instructions from general domains. AgentTuning is used to instruction-tune the Llama 2 series, resulting in AgentLM. Our evaluations show that AgentTuning enables LLMs' agent capabilities without compromising general abilities. The AgentLM-70B is comparable to GPT-3.5-turbo on unseen agent tasks, demonstrating generalized agent capabilities. We open source the AgentInstruct and AgentLM-7B, 13B, and 70B models at https://github.com/THUDM/AgentTuning, serving open and powerful alternatives to commercial LLMs for agent tasks.

{{</citation>}}


### (83/160) GestureGPT: Zero-shot Interactive Gesture Understanding and Grounding with Large Language Model Agents (Xin Zeng et al., 2023)

{{<citation>}}

Xin Zeng, Xiaoyu Wang, Tengxiang Zhang, Chun Yu, Shengdong Zhao, Yiqiang Chen. (2023)  
**GestureGPT: Zero-shot Interactive Gesture Understanding and Grounding with Large Language Model Agents**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12821v2)  

---


**ABSTRACT**  
Current gesture recognition systems primarily focus on identifying gestures within a predefined set, leaving a gap in connecting these gestures to interactive GUI elements or system functions (e.g., linking a 'thumb-up' gesture to a 'like' button). We introduce GestureGPT, a novel zero-shot gesture understanding and grounding framework leveraging large language models (LLMs). Gesture descriptions are formulated based on hand landmark coordinates from gesture videos and fed into our dual-agent dialogue system. A gesture agent deciphers these descriptions and queries about the interaction context (e.g., interface, history, gaze data), which a context agent organizes and provides. Following iterative exchanges, the gesture agent discerns user intent, grounding it to an interactive function. We validated the gesture description module using public first-view and third-view gesture datasets and tested the whole system in two real-world settings: video streaming and smart home IoT control. The highest zero-shot Top-5 grounding accuracies are 80.11% for video streaming and 90.78% for smart home tasks, showing potential of the new gesture understanding paradigm.

{{</citation>}}


### (84/160) Boosting Inference Efficiency: Unleashing the Power of Parameter-Shared Pre-trained Language Models (Weize Chen et al., 2023)

{{<citation>}}

Weize Chen, Xiaoyue Xu, Xu Han, Yankai Lin, Ruobing Xie, Zhiyuan Liu, Maosong Sun, Jie Zhou. (2023)  
**Boosting Inference Efficiency: Unleashing the Power of Parameter-Shared Pre-trained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12818v1)  

---


**ABSTRACT**  
Parameter-shared pre-trained language models (PLMs) have emerged as a successful approach in resource-constrained environments, enabling substantial reductions in model storage and memory costs without significant performance compromise. However, it is important to note that parameter sharing does not alleviate computational burdens associated with inference, thus impeding its practicality in situations characterized by limited stringent latency requirements or computational resources. Building upon neural ordinary differential equations (ODEs), we introduce a straightforward technique to enhance the inference efficiency of parameter-shared PLMs. Additionally, we propose a simple pre-training technique that leads to fully or partially shared models capable of achieving even greater inference acceleration. The experimental results demonstrate the effectiveness of our methods on both autoregressive and autoencoding PLMs, providing novel insights into more efficient utilization of parameter-shared models in resource-constrained settings.

{{</citation>}}


### (85/160) MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter (Zhiyuan Liu et al., 2023)

{{<citation>}}

Zhiyuan Liu, Sihang Li, Yanchen Luo, Hao Fei, Yixin Cao, Kenji Kawaguchi, Xiang Wang, Tat-Seng Chua. (2023)  
**MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-MM, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12798v1)  

---


**ABSTRACT**  
Language Models (LMs) have demonstrated impressive molecule understanding ability on various 1D text-related tasks. However, they inherently lack 2D graph perception - a critical ability of human professionals in comprehending molecules' topological structures. To bridge this gap, we propose MolCA: Molecular Graph-Language Modeling with Cross-Modal Projector and Uni-Modal Adapter. MolCA enables an LM (e.g., Galactica) to understand both text- and graph-based molecular contents via the cross-modal projector. Specifically, the cross-modal projector is implemented as a Q-Former to connect a graph encoder's representation space and an LM's text space. Further, MolCA employs a uni-modal adapter (i.e., LoRA) for the LM's efficient adaptation to downstream tasks. Unlike previous studies that couple an LM with a graph encoder via cross-modal contrastive learning, MolCA retains the LM's ability of open-ended text generation and augments it with 2D graph information. To showcase its effectiveness, we extensively benchmark MolCA on tasks of molecule captioning, IUPAC name prediction, and molecule-text retrieval, on which MolCA significantly outperforms the baselines. Our codes and checkpoints can be found at https://github.com/acharkq/MolCA.

{{</citation>}}


### (86/160) Are Structural Concepts Universal in Transformer Language Models? Towards Interpretable Cross-Lingual Generalization (Ningyu Xu et al., 2023)

{{<citation>}}

Ningyu Xu, Qi Zhang, Jingting Ye, Menghan Zhang, Xuanjing Huang. (2023)  
**Are Structural Concepts Universal in Transformer Language Models? Towards Interpretable Cross-Lingual Generalization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.12794v1)  

---


**ABSTRACT**  
Large language models (LLMs) have exhibited considerable cross-lingual generalization abilities, whereby they implicitly transfer knowledge across languages. However, the transfer is not equally successful for all languages, especially for low-resource ones, which poses an ongoing challenge. It is unclear whether we have reached the limits of implicit cross-lingual generalization and if explicit knowledge transfer is viable. In this paper, we investigate the potential for explicitly aligning conceptual correspondence between languages to enhance cross-lingual generalization. Using the syntactic aspect of language as a testbed, our analyses of 43 languages reveal a high degree of alignability among the spaces of structural concepts within each language for both encoder-only and decoder-only LLMs. We then propose a meta-learning-based method to learn to align conceptual spaces of different languages, which facilitates zero-shot and few-shot generalization in concept classification and also offers insights into the cross-lingual in-context learning phenomenon. Experiments on syntactic analysis tasks show that our approach achieves competitive results with state-of-the-art methods and narrows the performance gap between languages, particularly benefiting those with limited resources.

{{</citation>}}


### (87/160) Label-Aware Automatic Verbalizer for Few-Shot Text Classification (Thanakorn Thaminkaew et al., 2023)

{{<citation>}}

Thanakorn Thaminkaew, Piyawat Lertvittayakumjorn, Peerapon Vateekul. (2023)  
**Label-Aware Automatic Verbalizer for Few-Shot Text Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Few-Shot, Text Classification  
[Paper Link](http://arxiv.org/abs/2310.12778v1)  

---


**ABSTRACT**  
Prompt-based learning has shown its effectiveness in few-shot text classification. One important factor in its success is a verbalizer, which translates output from a language model into a predicted class. Notably, the simplest and widely acknowledged verbalizer employs manual labels to represent the classes. However, manual selection does not guarantee the optimality of the selected words when conditioned on the chosen language model. Therefore, we propose Label-Aware Automatic Verbalizer (LAAV), effectively augmenting the manual labels to achieve better few-shot classification results. Specifically, we use the manual labels along with the conjunction "and" to induce the model to generate more effective words for the verbalizer. The experimental results on five datasets across five languages demonstrate that LAAV significantly outperforms existing verbalizers. Furthermore, our analysis reveals that LAAV suggests more relevant words compared to similar approaches, especially in mid-to-low resource languages.

{{</citation>}}


### (88/160) Survival of the Most Influential Prompts: Efficient Black-Box Prompt Search via Clustering and Pruning (Han Zhou et al., 2023)

{{<citation>}}

Han Zhou, Xingchen Wan, Ivan Vulić, Anna Korhonen. (2023)  
**Survival of the Most Influential Prompts: Efficient Black-Box Prompt Search via Clustering and Pruning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2310.12774v1)  

---


**ABSTRACT**  
Prompt-based learning has been an effective paradigm for large pretrained language models (LLM), enabling few-shot or even zero-shot learning. Black-box prompt search has received growing interest recently for its distinctive properties of gradient-free optimization, proven particularly useful and powerful for model-as-a-service usage. However, the discrete nature and the complexity of combinatorial optimization hinder the efficiency of modern black-box approaches. Despite extensive research on search algorithms, the crucial aspect of search space design and optimization has been largely overlooked. In this paper, we first conduct a sensitivity analysis by prompting LLM, revealing that only a small number of tokens exert a disproportionate amount of influence on LLM predictions. Leveraging this insight, we propose the Clustering and Pruning for Efficient Black-box Prompt Search (ClaPS), a simple black-box search method that first clusters and prunes the search space to focus exclusively on influential prompt tokens. By employing even simple search methods within the pruned search space, ClaPS achieves state-of-the-art performance across various tasks and LLMs, surpassing the performance of complex approaches while significantly reducing search costs. Our findings underscore the critical role of search space design and optimization in enhancing both the usefulness and the efficiency of black-box prompt-based learning.

{{</citation>}}


### (89/160) Transformer-based Entity Legal Form Classification (Alexander Arimond et al., 2023)

{{<citation>}}

Alexander Arimond, Mauro Molteni, Dominik Jany, Zornitsa Manolova, Damian Borth, Andreas G. F. Hoepner. (2023)  
**Transformer-based Entity Legal Form Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, Legal, Transformer  
[Paper Link](http://arxiv.org/abs/2310.12766v1)  

---


**ABSTRACT**  
We propose the application of Transformer-based language models for classifying entity legal forms from raw legal entity names. Specifically, we employ various BERT variants and compare their performance against multiple traditional baselines. Our evaluation encompasses a substantial subset of freely available Legal Entity Identifier (LEI) data, comprising over 1.1 million legal entities from 30 different legal jurisdictions. The ground truth labels for classification per jurisdiction are taken from the Entity Legal Form (ELF) code standard (ISO 20275). Our findings demonstrate that pre-trained BERT variants outperform traditional text classification approaches in terms of F1 score, while also performing comparably well in the Macro F1 Score. Moreover, the validity of our proposal is supported by the outcome of third-party expert reviews conducted in ten selected jurisdictions. This study highlights the significant potential of Transformer-based models in advancing data standardization and data integration. The presented approaches can greatly benefit financial institutions, corporations, governments and other organizations in assessing business relationships, understanding risk exposure, and promoting effective governance.

{{</citation>}}


### (90/160) Character-level Chinese Backpack Language Models (Hao Sun et al., 2023)

{{<citation>}}

Hao Sun, John Hewitt. (2023)  
**Character-level Chinese Backpack Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.12751v1)  

---


**ABSTRACT**  
The Backpack is a Transformer alternative shown to improve interpretability in English language modeling by decomposing predictions into a weighted sum of token sense components. However, Backpacks' reliance on token-defined meaning raises questions as to their potential for languages other than English, a language for which subword tokenization provides a reasonable approximation for lexical items. In this work, we train, evaluate, interpret, and control Backpack language models in character-tokenized Chinese, in which words are often composed of many characters. We find that our (134M parameter) Chinese Backpack language model performs comparably to a (104M parameter) Transformer, and learns rich character-level meanings that log-additively compose to form word meanings. In SimLex-style lexical semantic evaluations, simple averages of Backpack character senses outperform input embeddings from a Transformer. We find that complex multi-character meanings are often formed by using the same per-character sense weights consistently across context. Exploring interpretability-through control, we show that we can localize a source of gender bias in our Backpacks to specific character senses and intervene to reduce the bias.

{{</citation>}}


### (91/160) Quality-Diversity through AI Feedback (Herbie Bradley et al., 2023)

{{<citation>}}

Herbie Bradley, Andrew Dai, Hannah Teufel, Jenny Zhang, Koen Oostermeijer, Marco Bellagente, Jeff Clune, Kenneth Stanley, Grégory Schott, Joel Lehman. (2023)  
**Quality-Diversity through AI Feedback**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-NE, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13032v1)  

---


**ABSTRACT**  
In many text-generation problems, users may prefer not only a single response, but a diverse range of high-quality outputs from which to choose. Quality-diversity (QD) search algorithms aim at such outcomes, by continually improving and diversifying a population of candidates. However, the applicability of QD to qualitative domains, like creative writing, has been limited by the difficulty of algorithmically specifying measures of quality and diversity. Interestingly, recent developments in language models (LMs) have enabled guiding search through AI feedback, wherein LMs are prompted in natural language to evaluate qualitative aspects of text. Leveraging this development, we introduce Quality-Diversity through AI Feedback (QDAIF), wherein an evolutionary algorithm applies LMs to both generate variation and evaluate the quality and diversity of candidate text. When assessed on creative writing domains, QDAIF covers more of a specified search space with high-quality samples than do non-QD controls. Further, human evaluation of QDAIF-generated creative texts validates reasonable agreement between AI and human evaluation. Our results thus highlight the potential of AI feedback to guide open-ended search for creative and original solutions, providing a recipe that seemingly generalizes to many domains and modalities. In this way, QDAIF is a step towards AI systems that can independently search, diversify, evaluate, and improve, which are among the core skills underlying human society's capacity for innovation.

{{</citation>}}


### (92/160) Is ChatGPT a Financial Expert? Evaluating Language Models on Financial Natural Language Processing (Yue Guo et al., 2023)

{{<citation>}}

Yue Guo, Zian Xu, Yi Yang. (2023)  
**Is ChatGPT a Financial Expert? Evaluating Language Models on Financial Natural Language Processing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Financial, GPT, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.12664v1)  

---


**ABSTRACT**  
The emergence of Large Language Models (LLMs), such as ChatGPT, has revolutionized general natural language preprocessing (NLP) tasks. However, their expertise in the financial domain lacks a comprehensive evaluation. To assess the ability of LLMs to solve financial NLP tasks, we present FinLMEval, a framework for Financial Language Model Evaluation, comprising nine datasets designed to evaluate the performance of language models. This study compares the performance of encoder-only language models and the decoder-only language models. Our findings reveal that while some decoder-only LLMs demonstrate notable performance across most financial tasks via zero-shot prompting, they generally lag behind the fine-tuned expert models, especially when dealing with proprietary datasets. We hope this study provides foundation evaluations for continuing efforts to build more advanced LLMs in the financial domain.

{{</citation>}}


### (93/160) A Use Case: Reformulating Query Rewriting as a Statistical Machine Translation Problem (Abdullah Can Algan et al., 2023)

{{<citation>}}

Abdullah Can Algan, Emre Yürekli, Aykut Çayır. (2023)  
**A Use Case: Reformulating Query Rewriting as a Statistical Machine Translation Problem**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Machine Translation, NLP  
[Paper Link](http://arxiv.org/abs/2310.13031v1)  

---


**ABSTRACT**  
One of the most important challenges for modern search engines is to retrieve relevant web content based on user queries. In order to achieve this challenge, search engines have a module to rewrite user queries. That is why modern web search engines utilize some statistical and neural models used in the natural language processing domain. Statistical machine translation is a well-known NLP method among them. The paper proposes a query rewriting pipeline based on a monolingual machine translation model that learns to rewrite Arabic user search queries. This paper also describes preprocessing steps to create a mapping between user queries and web page titles.

{{</citation>}}


### (94/160) Towards Real-World Streaming Speech Translation for Code-Switched Speech (Belen Alastruey et al., 2023)

{{<citation>}}

Belen Alastruey, Matthias Sperber, Christian Gollan, Dominic Telaar, Tim Ng, Aashish Agarwal. (2023)  
**Towards Real-World Streaming Speech Translation for Code-Switched Speech**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.12648v2)  

---


**ABSTRACT**  
Code-switching (CS), i.e. mixing different languages in a single sentence, is a common phenomenon in communication and can be challenging in many Natural Language Processing (NLP) settings. Previous studies on CS speech have shown promising results for end-to-end speech translation (ST), but have been limited to offline scenarios and to translation to one of the languages present in the source (\textit{monolingual transcription}).   In this paper, we focus on two essential yet unexplored areas for real-world CS speech translation: streaming settings, and translation to a third language (i.e., a language not included in the source). To this end, we extend the Fisher and Miami test and validation datasets to include new targets in Spanish and German. Using this data, we train a model for both offline and streaming ST and we establish baseline results for the two settings mentioned earlier.

{{</citation>}}


### (95/160) Non-Autoregressive Sentence Ordering (Yi Bin et al., 2023)

{{<citation>}}

Yi Bin, Wenhao Shi, Bin Ji, Jipeng Zhang, Yujuan Ding, Yang Yang. (2023)  
**Non-Autoregressive Sentence Ordering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.12640v1)  

---


**ABSTRACT**  
Existing sentence ordering approaches generally employ encoder-decoder frameworks with the pointer net to recover the coherence by recurrently predicting each sentence step-by-step. Such an autoregressive manner only leverages unilateral dependencies during decoding and cannot fully explore the semantic dependency between sentences for ordering. To overcome these limitations, in this paper, we propose a novel Non-Autoregressive Ordering Network, dubbed \textit{NAON}, which explores bilateral dependencies between sentences and predicts the sentence for each position in parallel. We claim that the non-autoregressive manner is not just applicable but also particularly suitable to the sentence ordering task because of two peculiar characteristics of the task: 1) each generation target is in deterministic length, and 2) the sentences and positions should match exclusively. Furthermore, to address the repetition issue of the naive non-autoregressive Transformer, we introduce an exclusive loss to constrain the exclusiveness between positions and sentences. To verify the effectiveness of the proposed model, we conduct extensive experiments on several common-used datasets and the experimental results show that our method outperforms all the autoregressive approaches and yields competitive performance compared with the state-of-the-arts. The codes are available at: \url{https://github.com/steven640pixel/nonautoregressive-sentence-ordering}.

{{</citation>}}


### (96/160) Predict the Future from the Past? On the Temporal Data Distribution Shift in Financial Sentiment Classifications (Yue Guo et al., 2023)

{{<citation>}}

Yue Guo, Chenxi Hu, Yi Yang. (2023)  
**Predict the Future from the Past? On the Temporal Data Distribution Shift in Financial Sentiment Classifications**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2310.12620v1)  

---


**ABSTRACT**  
Temporal data distribution shift is prevalent in the financial text. How can a financial sentiment analysis system be trained in a volatile market environment that can accurately infer sentiment and be robust to temporal data distribution shifts? In this paper, we conduct an empirical study on the financial sentiment analysis system under temporal data distribution shifts using a real-world financial social media dataset that spans three years. We find that the fine-tuned models suffer from general performance degradation in the presence of temporal distribution shifts. Furthermore, motivated by the unique temporal nature of the financial text, we propose a novel method that combines out-of-distribution detection with time series modeling for temporal financial sentiment analysis. Experimental results show that the proposed method enhances the model's capability to adapt to evolving temporal shifts in a volatile financial market.

{{</citation>}}


### (97/160) Identifying and Adapting Transformer-Components Responsible for Gender Bias in an English Language Model (Abhijith Chintam et al., 2023)

{{<citation>}}

Abhijith Chintam, Rahel Beloch, Willem Zuidema, Michael Hanna, Oskar van der Wal. (2023)  
**Identifying and Adapting Transformer-Components Responsible for Gender Bias in an English Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias, GPT, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.12611v1)  

---


**ABSTRACT**  
Language models (LMs) exhibit and amplify many types of undesirable biases learned from the training data, including gender bias. However, we lack tools for effectively and efficiently changing this behavior without hurting general language modeling performance. In this paper, we study three methods for identifying causal relations between LM components and particular output: causal mediation analysis, automated circuit discovery and our novel, efficient method called DiffMask+ based on differential masking. We apply the methods to GPT-2 small and the problem of gender bias, and use the discovered sets of components to perform parameter-efficient fine-tuning for bias mitigation. Our results show significant overlap in the identified components (despite huge differences in the computational requirements of the methods) as well as success in mitigating gender bias, with less damage to general language modeling compared to full model fine-tuning. However, our work also underscores the difficulty of defining and measuring bias, and the sensitivity of causal discovery procedures to dataset choice. We hope our work can contribute to more attention for dataset development, and lead to more effective mitigation strategies for other types of bias.

{{</citation>}}


### (98/160) Time-Aware Representation Learning for Time-Sensitive Question Answering (Jungbin Son et al., 2023)

{{<citation>}}

Jungbin Son, Alice Oh. (2023)  
**Time-Aware Representation Learning for Time-Sensitive Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: QA, Question Answering, Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.12585v1)  

---


**ABSTRACT**  
Time is one of the crucial factors in real-world question answering (QA) problems. However, language models have difficulty understanding the relationships between time specifiers, such as 'after' and 'before', and numbers, since existing QA datasets do not include sufficient time expressions. To address this issue, we propose a Time-Context aware Question Answering (TCQA) framework. We suggest a Time-Context dependent Span Extraction (TCSE) task, and build a time-context dependent data generation framework for model training. Moreover, we present a metric to evaluate the time awareness of the QA model using TCSE. The TCSE task consists of a question and four sentence candidates classified as correct or incorrect based on time and context. The model is trained to extract the answer span from the sentence that is both correct in time and context. The model trained with TCQA outperforms baseline models up to 8.5 of the F1-score in the TimeQA dataset. Our dataset and code are available at https://github.com/sonjbin/TCQA

{{</citation>}}


### (99/160) Pretraining Language Models with Text-Attributed Heterogeneous Graphs (Tao Zou et al., 2023)

{{<citation>}}

Tao Zou, Le Yu, Yifei Huang, Leilei Sun, Bowen Du. (2023)  
**Pretraining Language Models with Text-Attributed Heterogeneous Graphs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12580v2)  

---


**ABSTRACT**  
In many real-world scenarios (e.g., academic networks, social platforms), different types of entities are not only associated with texts but also connected by various relationships, which can be abstracted as Text-Attributed Heterogeneous Graphs (TAHGs). Current pretraining tasks for Language Models (LMs) primarily focus on separately learning the textual information of each entity and overlook the crucial aspect of capturing topological connections among entities in TAHGs. In this paper, we present a new pretraining framework for LMs that explicitly considers the topological and heterogeneous information in TAHGs. Firstly, we define a context graph as neighborhoods of a target node within specific orders and propose a topology-aware pretraining task to predict nodes involved in the context graph by jointly optimizing an LM and an auxiliary heterogeneous graph neural network. Secondly, based on the observation that some nodes are text-rich while others have little text, we devise a text augmentation strategy to enrich textless nodes with their neighbors' texts for handling the imbalance issue. We conduct link prediction and node classification tasks on three datasets from various domains. Experimental results demonstrate the superiority of our approach over existing methods and the rationality of each design. Our code is available at https://github.com/Hope-Rita/THLM.

{{</citation>}}


### (100/160) Multilingual estimation of political-party positioning: From label aggregation to long-input Transformers (Dmitry Nikolaev et al., 2023)

{{<citation>}}

Dmitry Nikolaev, Tanise Ceron, Sebastian Padó. (2023)  
**Multilingual estimation of political-party positioning: From label aggregation to long-input Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12575v1)  

---


**ABSTRACT**  
Scaling analysis is a technique in computational political science that assigns a political actor (e.g. politician or party) a score on a predefined scale based on a (typically long) body of text (e.g. a parliamentary speech or an election manifesto). For example, political scientists have often used the left--right scale to systematically analyse political landscapes of different countries. NLP methods for automatic scaling analysis can find broad application provided they (i) are able to deal with long texts and (ii) work robustly across domains and languages. In this work, we implement and compare two approaches to automatic scaling analysis of political-party manifestos: label aggregation, a pipeline strategy relying on annotations of individual statements from the manifestos, and long-input-Transformer-based models, which compute scaling values directly from raw text. We carry out the analysis of the Comparative Manifestos Project dataset across 41 countries and 27 languages and find that the task can be efficiently solved by state-of-the-art models, with label aggregation producing the best results.

{{</citation>}}


### (101/160) Large Language Models Help Humans Verify Truthfulness -- Except When They Are Convincingly Wrong (Chenglei Si et al., 2023)

{{<citation>}}

Chenglei Si, Navita Goyal, Sherry Tongshuang Wu, Chen Zhao, Shi Feng, Hal Daumé III, Jordan Boyd-Graber. (2023)  
**Large Language Models Help Humans Verify Truthfulness -- Except When They Are Convincingly Wrong**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12558v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are increasingly used for accessing information on the web. Their truthfulness and factuality are thus of great interest. To help users make the right decisions about the information they're getting, LLMs should not only provide but also help users fact-check information. In this paper, we conduct experiments with 80 crowdworkers in total to compare language models with search engines (information retrieval systems) at facilitating fact-checking by human users. We prompt LLMs to validate a given claim and provide corresponding explanations. Users reading LLM explanations are significantly more efficient than using search engines with similar accuracy. However, they tend to over-rely the LLMs when the explanation is wrong. To reduce over-reliance on LLMs, we ask LLMs to provide contrastive information - explain both why the claim is true and false, and then we present both sides of the explanation to users. This contrastive explanation mitigates users' over-reliance on LLMs, but cannot significantly outperform search engines. However, showing both search engine results and LLM explanations offers no complementary benefits as compared to search engines alone. Taken together, natural language explanations by LLMs may not be a reliable replacement for reading the retrieved passages yet, especially in high-stakes settings where over-relying on wrong AI explanations could lead to critical consequences.

{{</citation>}}


### (102/160) DepWiGNN: A Depth-wise Graph Neural Network for Multi-hop Spatial Reasoning in Text (Shuaiyi Li et al., 2023)

{{<citation>}}

Shuaiyi Li, Yang Deng, Wai Lam. (2023)  
**DepWiGNN: A Depth-wise Graph Neural Network for Multi-hop Spatial Reasoning in Text**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GNN, Graph Neural Network, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.12557v1)  

---


**ABSTRACT**  
Spatial reasoning in text plays a crucial role in various real-world applications. Existing approaches for spatial reasoning typically infer spatial relations from pure text, which overlook the gap between natural language and symbolic structures. Graph neural networks (GNNs) have showcased exceptional proficiency in inducing and aggregating symbolic structures. However, classical GNNs face challenges in handling multi-hop spatial reasoning due to the over-smoothing issue, \textit{i.e.}, the performance decreases substantially as the number of graph layers increases. To cope with these challenges, we propose a novel \textbf{Dep}th-\textbf{Wi}se \textbf{G}raph \textbf{N}eural \textbf{N}etwork (\textbf{DepWiGNN}). Specifically, we design a novel node memory scheme and aggregate the information over the depth dimension instead of the breadth dimension of the graph, which empowers the ability to collect long dependencies without stacking multiple layers. Experimental results on two challenging multi-hop spatial reasoning datasets show that DepWiGNN outperforms existing spatial reasoning methods. The comparisons with the other three GNNs further demonstrate its superiority in capturing long dependency in the graph.

{{</citation>}}


### (103/160) Reliable Academic Conference Question Answering: A Study Based on Large Language Model (Zhiwei Huang et al., 2023)

{{<citation>}}

Zhiwei Huang, Long Jin, Junjie Wang, Mingchen Tu, Yin Hua, Zhiqiang Liu, Jiawei Meng, Huajun Chen, Wen Zhang. (2023)  
**Reliable Academic Conference Question Answering: A Study Based on Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, NLP, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.13028v1)  

---


**ABSTRACT**  
The rapid growth of computer science has led to a proliferation of research presented at academic conferences, fostering global scholarly communication. Researchers consistently seek accurate, current information about these events at all stages. This data surge necessitates an intelligent question-answering system to efficiently address researchers' queries and ensure awareness of the latest advancements. The information of conferences is usually published on their official website, organized in a semi-structured way with a lot of text. To address this need, we have developed the ConferenceQA dataset for 7 diverse academic conferences with human annotations. Firstly, we employ a combination of manual and automated methods to organize academic conference data in a semi-structured JSON format. Subsequently, we annotate nearly 100 question-answer pairs for each conference. Each pair is classified into four different dimensions. To ensure the reliability of the data, we manually annotate the source of each answer. In light of recent advancements, Large Language Models (LLMs) have demonstrated impressive performance in various NLP tasks. They have demonstrated impressive capabilities in information-seeking question answering after instruction fine-tuning, and as such, we present our conference QA study based on LLM. Due to hallucination and outdated knowledge of LLMs, we adopt retrieval based methods to enhance LLMs' question-answering abilities. We have proposed a structure-aware retrieval method, specifically designed to leverage inherent structural information during the retrieval process. Empirical validation on the ConferenceQA dataset has demonstrated the effectiveness of this method. The dataset and code are readily accessible on https://github.com/zjukg/ConferenceQA.

{{</citation>}}


### (104/160) Product Attribute Value Extraction using Large Language Models (Alexander Brinkmann et al., 2023)

{{<citation>}}

Alexander Brinkmann, Roee Shraga, Christian Bizer. (2023)  
**Product Attribute Value Extraction using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12537v1)  

---


**ABSTRACT**  
E-commerce applications such as faceted product search or product comparison are based on structured product descriptions like attribute/value pairs. The vendors on e-commerce platforms do not provide structured product descriptions but describe offers using titles or descriptions. To process such offers, it is necessary to extract attribute/value pairs from textual product attributes. State-of-the-art attribute/value extraction techniques rely on pre-trained language models (PLMs), such as BERT. Two major drawbacks of these models for attribute/value extraction are that (i) the models require significant amounts of task-specific training data and (ii) the fine-tuned models face challenges in generalizing to attribute values not included in the training data. This paper explores the potential of large language models (LLMs) as a training data-efficient and robust alternative to PLM-based attribute/value extraction methods. We consider hosted LLMs, such as GPT-3.5 and GPT-4, as well as open-source LLMs based on Llama2. We evaluate the models in a zero-shot scenario and in a scenario where task-specific training data is available. In the zero-shot scenario, we compare various prompt designs for representing information about the target attributes of the extraction. In the scenario with training data, we investigate (i) the provision of example attribute values, (ii) the selection of in-context demonstrations, and (iii) the fine-tuning of GPT-3.5. Our experiments show that GPT-4 achieves an average F1-score of 85% on the two evaluation datasets while the best PLM-based techniques perform on average 5% worse using the same amount of training data. GPT-4 achieves a 10% higher F1-score than the best open-source LLM. The fine-tuned GPT-3.5 model reaches a similar performance as GPT-4 while being significantly more cost-efficient.

{{</citation>}}


### (105/160) ICU: Conquering Language Barriers in Vision-and-Language Modeling by Dividing the Tasks into Image Captioning and Language Understanding (Guojun Wu, 2023)

{{<citation>}}

Guojun Wu. (2023)  
**ICU: Conquering Language Barriers in Vision-and-Language Modeling by Dividing the Tasks into Image Captioning and Language Understanding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLUE, Image Captioning, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12531v2)  

---


**ABSTRACT**  
Most multilingual vision-and-language (V&L) research aims to accomplish multilingual and multimodal capabilities within one model. However, the scarcity of multilingual captions for images has hindered the development. To overcome this obstacle, we propose ICU, Image Caption Understanding, which divides a V&L task into two stages: a V&L model performs image captioning in English, and a multilingual language model (mLM), in turn, takes the caption as the alt text and performs crosslingual language understanding. The burden of multilingual processing is lifted off V&L model and placed on mLM. Since the multilingual text data is relatively of higher abundance and quality, ICU can facilitate the conquering of language barriers for V&L models. In experiments on two tasks across 9 languages in the IGLUE benchmark, we show that ICU can achieve new state-of-the-art results for five languages, and comparable results for the rest.

{{</citation>}}


### (106/160) Named Entity Recognition for Monitoring Plant Health Threats in Tweets: a ChouBERT Approach (Shufan Jiang et al., 2023)

{{<citation>}}

Shufan Jiang, Rafael Angarita, Stéphane Cormier, Francis Rousseaux. (2023)  
**Named Entity Recognition for Monitoring Plant Health Threats in Tweets: a ChouBERT Approach**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, Named Entity Recognition, Twitter  
[Paper Link](http://arxiv.org/abs/2310.12522v1)  

---


**ABSTRACT**  
An important application scenario of precision agriculture is detecting and measuring crop health threats using sensors and data analysis techniques. However, the textual data are still under-explored among the existing solutions due to the lack of labelled data and fine-grained semantic resources. Recent research suggests that the increasing connectivity of farmers and the emergence of online farming communities make social media like Twitter a participatory platform for detecting unfamiliar plant health events if we can extract essential information from unstructured textual data. ChouBERT is a French pre-trained language model that can identify Tweets concerning observations of plant health issues with generalizability on unseen natural hazards. This paper tackles the lack of labelled data by further studying ChouBERT's know-how on token-level annotation tasks over small labeled sets.

{{</citation>}}


### (107/160) Lost in Translation: When GPT-4V(ision) Can't See Eye to Eye with Text. A Vision-Language-Consistency Analysis of VLLMs and Beyond (Xiang Zhang et al., 2023)

{{<citation>}}

Xiang Zhang, Senyu Li, Zijun Wu, Ning Shi. (2023)  
**Lost in Translation: When GPT-4V(ision) Can't See Eye to Eye with Text. A Vision-Language-Consistency Analysis of VLLMs and Beyond**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12520v1)  

---


**ABSTRACT**  
Recent advancements in multimodal techniques open exciting possibilities for models excelling in diverse tasks involving text, audio, and image processing. Models like GPT-4V, blending computer vision and language modeling, excel in complex text and image tasks. Numerous prior research endeavors have diligently examined the performance of these Vision Large Language Models (VLLMs) across tasks like object detection, image captioning and others. However, these analyses often focus on evaluating the performance of each modality in isolation, lacking insights into their cross-modal interactions. Specifically, questions concerning whether these vision-language models execute vision and language tasks consistently or independently have remained unanswered. In this study, we draw inspiration from recent investigations into multilingualism and conduct a comprehensive analysis of model's cross-modal interactions. We introduce a systematic framework that quantifies the capability disparities between different modalities in the multi-modal setting and provide a set of datasets designed for these evaluations. Our findings reveal that models like GPT-4V tend to perform consistently modalities when the tasks are relatively simple. However, the trustworthiness of results derived from the vision modality diminishes as the tasks become more challenging. Expanding on our findings, we introduce "Vision Description Prompting," a method that effectively improves performance in challenging vision-related tasks.

{{</citation>}}


### (108/160) Automatic Hallucination Assessment for Aligned Large Language Models via Transferable Adversarial Attacks (Xiaodong Yu et al., 2023)

{{<citation>}}

Xiaodong Yu, Hao Cheng, Xiaodong Liu, Dan Roth, Jianfeng Gao. (2023)  
**Automatic Hallucination Assessment for Aligned Large Language Models via Transferable Adversarial Attacks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Adversarial Attack, ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12516v1)  

---


**ABSTRACT**  
Although remarkable progress has been achieved in preventing large language model (LLM) hallucinations using instruction tuning and retrieval augmentation, it remains challenging to measure the reliability of LLMs using human-crafted evaluation data which is not available for many tasks and domains and could suffer from data leakage. Inspired by adversarial machine learning, this paper aims to develop a method of automatically generating evaluation data by appropriately modifying existing data on which LLMs behave faithfully. Specifically, this paper presents AutoDebug, an LLM-based framework to use prompting chaining to generate transferable adversarial attacks in the form of question-answering examples. We seek to understand the extent to which these examples trigger the hallucination behaviors of LLMs.   We implement AutoDebug using ChatGPT and evaluate the resulting two variants of a popular open-domain question-answering dataset, Natural Questions (NQ), on a collection of open-source and proprietary LLMs under various prompting settings. Our generated evaluation data is human-readable and, as we show, humans can answer these modified questions well. Nevertheless, we observe pronounced accuracy drops across multiple LLMs including GPT-4. Our experimental results show that LLMs are likely to hallucinate in two categories of question-answering scenarios where (1) there are conflicts between knowledge given in the prompt and their parametric knowledge, or (2) the knowledge expressed in the prompt is complex. Finally, we find that the adversarial examples generated by our method are transferable across all considered LLMs. The examples generated by a small model can be used to debug a much larger model, making our approach cost-effective.

{{</citation>}}


### (109/160) Towards Anytime Fine-tuning: Continually Pre-trained Language Models with Hypernetwork Prompt (Gangwei Jiang et al., 2023)

{{<citation>}}

Gangwei Jiang, Caigao Jiang, Siqiao Xue, James Y. Zhang, Jun Zhou, Defu Lian, Ying Wei. (2023)  
**Towards Anytime Fine-tuning: Continually Pre-trained Language Models with Hypernetwork Prompt**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.13024v1)  

---


**ABSTRACT**  
Continual pre-training has been urgent for adapting a pre-trained model to a multitude of domains and tasks in the fast-evolving world. In practice, a continually pre-trained model is expected to demonstrate not only greater capacity when fine-tuned on pre-trained domains but also a non-decreasing performance on unseen ones. In this work, we first investigate such anytime fine-tuning effectiveness of existing continual pre-training approaches, concluding with unanimously decreased performance on unseen domains. To this end, we propose a prompt-guided continual pre-training method, where we train a hypernetwork to generate domain-specific prompts by both agreement and disagreement losses. The agreement loss maximally preserves the generalization of a pre-trained model to new domains, and the disagreement one guards the exclusiveness of the generated hidden states for each domain. Remarkably, prompts by the hypernetwork alleviate the domain identity when fine-tuning and promote knowledge transfer across domains. Our method achieved improvements of 3.57% and 3.4% on two real-world datasets (including domain shift and temporal shift), respectively, demonstrating its efficacy.

{{</citation>}}


### (110/160) GraphGPT: Graph Instruction Tuning for Large Language Models (Jiabin Tang et al., 2023)

{{<citation>}}

Jiabin Tang, Yuhao Yang, Wei Wei, Lei Shi, Lixin Su, Suqi Cheng, Dawei Yin, Chao Huang. (2023)  
**GraphGPT: Graph Instruction Tuning for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GNN, GPT, Graph Neural Network, Graph Neural Networks, Language Model  
[Paper Link](http://arxiv.org/abs/2310.13023v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have advanced graph structure understanding via recursive information exchange and aggregation among graph nodes. To improve model robustness, self-supervised learning (SSL) has emerged as a promising approach for data augmentation. However, existing methods for generating pre-trained graph embeddings often rely on fine-tuning with specific downstream task labels, which limits their usability in scenarios where labeled data is scarce or unavailable. To address this, our research focuses on advancing the generalization capabilities of graph models in challenging zero-shot learning scenarios. Inspired by the success of large language models (LLMs), we aim to develop a graph-oriented LLM that can achieve high generalization across diverse downstream datasets and tasks, even without any information available from the downstream graph data. In this work, we present the GraphGPT framework that aligns LLMs with graph structural knowledge with a graph instruction tuning paradigm. Our framework incorporates a text-graph grounding component to establish a connection between textual information and graph structures. Additionally, we propose a dual-stage instruction tuning paradigm, accompanied by a lightweight graph-text alignment projector. This paradigm explores self-supervised graph structural signals and task-specific graph instructions, to guide LLMs in understanding complex graph structures and improving their adaptability across different downstream tasks. Our framework is evaluated on supervised and zero-shot graph learning tasks, demonstrating superior generalization and outperforming state-of-the-art baselines.

{{</citation>}}


### (111/160) Attack Prompt Generation for Red Teaming and Defending Large Language Models (Boyi Deng et al., 2023)

{{<citation>}}

Boyi Deng, Wenjie Wang, Fuli Feng, Yang Deng, Qifan Wang, Xiangnan He. (2023)  
**Attack Prompt Generation for Red Teaming and Defending Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12505v1)  

---


**ABSTRACT**  
Large language models (LLMs) are susceptible to red teaming attacks, which can induce LLMs to generate harmful content. Previous research constructs attack prompts via manual or automatic methods, which have their own limitations on construction cost and quality. To address these issues, we propose an integrated approach that combines manual and automatic methods to economically generate high-quality attack prompts. Specifically, considering the impressive capabilities of newly emerged LLMs, we propose an attack framework to instruct LLMs to mimic human-generated prompts through in-context learning. Furthermore, we propose a defense framework that fine-tunes victim LLMs through iterative interactions with the attack framework to enhance their safety against red teaming attacks. Extensive experiments on different LLMs validate the effectiveness of our proposed attack and defense frameworks. Additionally, we release a series of attack prompts datasets named SAP with varying sizes, facilitating the safety evaluation and enhancement of more LLMs. Our code and dataset is available on https://github.com/Aatrox103/SAP .

{{</citation>}}


### (112/160) Co$^2$PT: Mitigating Bias in Pre-trained Language Models through Counterfactual Contrastive Prompt Tuning (Xiangjue Dong et al., 2023)

{{<citation>}}

Xiangjue Dong, Ziwei Zhu, Zhuoer Wang, Maria Teleki, James Caverlee. (2023)  
**Co$^2$PT: Mitigating Bias in Pre-trained Language Models through Counterfactual Contrastive Prompt Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12490v1)  

---


**ABSTRACT**  
Pre-trained Language Models are widely used in many important real-world applications. However, recent studies show that these models can encode social biases from large pre-training corpora and even amplify biases in downstream applications. To address this challenge, we propose Co$^2$PT, an efficient and effective debias-while-prompt tuning method for mitigating biases via counterfactual contrastive prompt tuning on downstream tasks. Our experiments conducted on three extrinsic bias benchmarks demonstrate the effectiveness of Co$^2$PT on bias mitigation during the prompt tuning process and its adaptability to existing upstream debiased language models. These findings indicate the strength of Co$^2$PT and provide promising avenues for further enhancement in bias mitigation on downstream tasks.

{{</citation>}}


### (113/160) MedAI Dialog Corpus (MEDIC): Zero-Shot Classification of Doctor and AI Responses in Health Consultations (Olumide E. Ojo et al., 2023)

{{<citation>}}

Olumide E. Ojo, Olaronke O. Adebanji, Alexander Gelbukh, Hiram Calvo, Anna Feldman. (2023)  
**MedAI Dialog Corpus (MEDIC): Zero-Shot Classification of Doctor and AI Responses in Health Consultations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Dialog, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.12489v2)  

---


**ABSTRACT**  
Zero-shot classification enables text to be classified into classes not seen during training. In this research, we investigate the effectiveness of pre-trained language models to accurately classify responses from Doctors and AI in health consultations through zero-shot learning. Our study aims to determine whether these models can effectively detect if a text originates from human or AI models without specific corpus training. We collect responses from doctors to patient inquiries about their health and pose the same question/response to AI models. While zero-shot language models show a good understanding of language in general, they have limitations in classifying doctor and AI responses in healthcare consultations. This research lays the groundwork for further research into this field of medical text classification, informing the development of more effective approaches to accurately classify doctor-generated and AI-generated text in health consultations.

{{</citation>}}


### (114/160) Not All Countries Celebrate Thanksgiving: On the Cultural Dominance in Large Language Models (Wenxuan Wang et al., 2023)

{{<citation>}}

Wenxuan Wang, Wenxiang Jiao, Jingyuan Huang, Ruyi Dai, Jen-tse Huang, Zhaopeng Tu, Michael R. Lyu. (2023)  
**Not All Countries Celebrate Thanksgiving: On the Cultural Dominance in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12481v1)  

---


**ABSTRACT**  
In this paper, we identify a cultural dominance issue within large language models (LLMs) due to the predominant use of English data in model training (e.g. ChatGPT). LLMs often provide inappropriate English-culture-related answers that are not relevant to the expected culture when users ask in non-English languages. To systematically evaluate the cultural dominance issue, we build a benchmark that consists of both concrete (e.g. holidays and songs) and abstract (e.g. values and opinions) cultural objects. Empirical results show that the representative GPT models suffer from the culture dominance problem, where GPT-4 is the most affected while text-davinci-003 suffers the least from this problem. Our study emphasizes the need for critical examination of cultural dominance and ethical consideration in their development and deployment. We show two straightforward methods in model development (i.e. pretraining on more diverse data) and deployment (e.g. culture-aware prompting) can significantly mitigate the cultural dominance issue in LLMs.

{{</citation>}}


### (115/160) Contrastive Learning for Inference in Dialogue (Etsuko Ishii et al., 2023)

{{<citation>}}

Etsuko Ishii, Yan Xu, Bryan Wilie, Ziwei Ji, Holy Lovenia, Willy Chung, Pascale Fung. (2023)  
**Contrastive Learning for Inference in Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2310.12467v1)  

---


**ABSTRACT**  
Inference, especially those derived from inductive processes, is a crucial component in our conversation to complement the information implicitly or explicitly conveyed by a speaker. While recent large language models show remarkable advances in inference tasks, their performance in inductive reasoning, where not all information is present in the context, is far behind deductive reasoning. In this paper, we analyze the behavior of the models based on the task difficulty defined by the semantic information gap -- which distinguishes inductive and deductive reasoning (Johnson-Laird, 1988, 1993). Our analysis reveals that the disparity in information between dialogue contexts and desired inferences poses a significant challenge to the inductive inference process. To mitigate this information gap, we investigate a contrastive learning approach by feeding negative samples. Our experiments suggest negative samples help models understand what is wrong and improve their inference generations.

{{</citation>}}


### (116/160) Rethinking the Construction of Effective Metrics for Understanding the Mechanisms of Pretrained Language Models (You Li et al., 2023)

{{<citation>}}

You Li, Jinhui Yin, Yuming Lin. (2023)  
**Rethinking the Construction of Effective Metrics for Understanding the Mechanisms of Pretrained Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model, Pretrained Language Models  
[Paper Link](http://arxiv.org/abs/2310.12454v1)  

---


**ABSTRACT**  
Pretrained language models are expected to effectively map input text to a set of vectors while preserving the inherent relationships within the text. Consequently, designing a white-box model to compute metrics that reflect the presence of specific internal relations in these vectors has become a common approach for post-hoc interpretability analysis of pretrained language models. However, achieving interpretability in white-box models and ensuring the rigor of metric computation becomes challenging when the source model lacks inherent interpretability. Therefore, in this paper, we discuss striking a balance in this trade-off and propose a novel line to constructing metrics for understanding the mechanisms of pretrained language models. We have specifically designed a family of metrics along this line of investigation, and the model used to compute these metrics is referred to as the tree topological probe. We conducted measurements on BERT-large by using these metrics. Based on the experimental results, we propose a speculation regarding the working mechanism of BERT-like pretrained language models, as well as a strategy for enhancing fine-tuning performance by leveraging the topological probe to improve specific submodules.

{{</citation>}}


### (117/160) Efficient Long-Range Transformers: You Need to Attend More, but Not Necessarily at Every Layer (Qingru Zhang et al., 2023)

{{<citation>}}

Qingru Zhang, Dhananjay Ram, Cole Hawkins, Sheng Zha, Tuo Zhao. (2023)  
**Efficient Long-Range Transformers: You Need to Attend More, but Not Necessarily at Every Layer**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12442v1)  

---


**ABSTRACT**  
Pretrained transformer models have demonstrated remarkable performance across various natural language processing tasks. These models leverage the attention mechanism to capture long- and short-range dependencies in the sequence. However, the (full) attention mechanism incurs high computational cost - quadratic in the sequence length, which is not affordable in tasks with long sequences, e.g., inputs with 8k tokens. Although sparse attention can be used to improve computational efficiency, as suggested in existing work, it has limited modeling capacity and often fails to capture complicated dependencies in long sequences. To tackle this challenge, we propose MASFormer, an easy-to-implement transformer variant with Mixed Attention Spans. Specifically, MASFormer is equipped with full attention to capture long-range dependencies, but only at a small number of layers. For the remaining layers, MASformer only employs sparse attention to capture short-range dependencies. Our experiments on natural language modeling and generation tasks show that a decoder-only MASFormer model of 1.3B parameters can achieve competitive performance to vanilla transformers with full attention while significantly reducing computational cost (up to 75%). Additionally, we investigate the effectiveness of continual training with long sequence data and how sequence length impacts downstream generation performance, which may be of independent interest.

{{</citation>}}


### (118/160) PoisonPrompt: Backdoor Attack on Prompt-based Large Language Models (Hongwei Yao et al., 2023)

{{<citation>}}

Hongwei Yao, Jian Lou, Zhan Qin. (2023)  
**PoisonPrompt: Backdoor Attack on Prompt-based Large Language Models**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12439v1)  

---


**ABSTRACT**  
Prompts have significantly improved the performance of pretrained Large Language Models (LLMs) on various downstream tasks recently, making them increasingly indispensable for a diverse range of LLM application scenarios. However, the backdoor vulnerability, a serious security threat that can maliciously alter the victim model's normal predictions, has not been sufficiently explored for prompt-based LLMs. In this paper, we present POISONPROMPT, a novel backdoor attack capable of successfully compromising both hard and soft prompt-based LLMs. We evaluate the effectiveness, fidelity, and robustness of POISONPROMPT through extensive experiments on three popular prompt methods, using six datasets and three widely used LLMs. Our findings highlight the potential security threats posed by backdoor attacks on prompt-based LLMs and emphasize the need for further research in this area.

{{</citation>}}


### (119/160) MAF: Multi-Aspect Feedback for Improving Reasoning in Large Language Models (Deepak Nathani et al., 2023)

{{<citation>}}

Deepak Nathani, David Wang, Liangming Pan, William Yang Wang. (2023)  
**MAF: Multi-Aspect Feedback for Improving Reasoning in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.12426v1)  

---


**ABSTRACT**  
Language Models (LMs) have shown impressive performance in various natural language tasks. However, when it comes to natural language reasoning, LMs still face challenges such as hallucination, generating incorrect intermediate reasoning steps, and making mathematical errors. Recent research has focused on enhancing LMs through self-improvement using feedback. Nevertheless, existing approaches relying on a single generic feedback source fail to address the diverse error types found in LM-generated reasoning chains. In this work, we propose Multi-Aspect Feedback, an iterative refinement framework that integrates multiple feedback modules, including frozen LMs and external tools, each focusing on a specific error category. Our experimental results demonstrate the efficacy of our approach to addressing several errors in the LM-generated reasoning chain and thus improving the overall performance of an LM in several reasoning tasks. We see a relative improvement of up to 20% in Mathematical Reasoning and up to 18% in Logical Entailment.

{{</citation>}}


### (120/160) The Shifted and The Overlooked: A Task-oriented Investigation of User-GPT Interactions (Siru Ouyang et al., 2023)

{{<citation>}}

Siru Ouyang, Shuohang Wang, Yang Liu, Ming Zhong, Yizhu Jiao, Dan Iter, Reid Pryzant, Chenguang Zhu, Heng Ji, Jiawei Han. (2023)  
**The Shifted and The Overlooked: A Task-oriented Investigation of User-GPT Interactions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.12418v1)  

---


**ABSTRACT**  
Recent progress in Large Language Models (LLMs) has produced models that exhibit remarkable performance across a variety of NLP tasks. However, it remains unclear whether the existing focus of NLP research accurately captures the genuine requirements of human users. This paper provides a comprehensive analysis of the divergence between current NLP research and the needs of real-world NLP applications via a large-scale collection of user-GPT conversations. We analyze a large-scale collection of real user queries to GPT. We compare these queries against existing NLP benchmark tasks and identify a significant gap between the tasks that users frequently request from LLMs and the tasks that are commonly studied in academic research. For example, we find that tasks such as ``design'' and ``planning'' are prevalent in user interactions but are largely neglected or different from traditional NLP benchmarks. We investigate these overlooked tasks, dissect the practical challenges they pose, and provide insights toward a roadmap to make LLMs better aligned with user needs.

{{</citation>}}


### (121/160) FinEntity: Entity-level Sentiment Classification for Financial Texts (Yixuan Tang et al., 2023)

{{<citation>}}

Yixuan Tang, Yi Yang, Allen H Huang, Andy Tam, Justin Z Tang. (2023)  
**FinEntity: Entity-level Sentiment Classification for Financial Texts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, ChatGPT, Financial, GPT  
[Paper Link](http://arxiv.org/abs/2310.12406v1)  

---


**ABSTRACT**  
In the financial domain, conducting entity-level sentiment analysis is crucial for accurately assessing the sentiment directed toward a specific financial entity. To our knowledge, no publicly available dataset currently exists for this purpose. In this work, we introduce an entity-level sentiment classification dataset, called \textbf{FinEntity}, that annotates financial entity spans and their sentiment (positive, neutral, and negative) in financial news. We document the dataset construction process in the paper. Additionally, we benchmark several pre-trained models (BERT, FinBERT, etc.) and ChatGPT on entity-level sentiment classification. In a case study, we demonstrate the practical utility of using FinEntity in monitoring cryptocurrency markets. The data and code of FinEntity is available at \url{https://github.com/yixuantt/FinEntity}

{{</citation>}}


## cs.RO (4)



### (122/160) Enhancing Multi-Drone Coordination for Filming Group Behaviours in Dynamic Environments (Aditya Rauniyar et al., 2023)

{{<citation>}}

Aditya Rauniyar, Jiaoyang Li, Sebastian Scherer. (2023)  
**Enhancing Multi-Drone Coordination for Filming Group Behaviours in Dynamic Environments**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: AI, Drone  
[Paper Link](http://arxiv.org/abs/2310.13184v1)  

---


**ABSTRACT**  
Multi-Agent Path Finding (MAPF) is a fundamental problem in robotics and AI, with numerous applications in real-world scenarios. One such scenario is filming scenes with multiple actors, where the goal is to capture the scene from multiple angles simultaneously. Here, we present a formation-based filming directive of task assignment followed by a Conflict-Based MAPF algorithm for efficient path planning of multiple agents to achieve filming objectives while avoiding collisions. We propose an extension to the standard MAPF formulation to accommodate actor-specific requirements and constraints. Our approach incorporates Conflict-Based Search, a widely used heuristic search technique for solving MAPF problems. We demonstrate the effectiveness of our approach through experiments on various MAPF scenarios in a simulated environment. The proposed algorithm enables the efficient online task assignment of formation-based filming to capture dynamic scenes, making it suitable for various filming and coverage applications.

{{</citation>}}


### (123/160) Creative Robot Tool Use with Large Language Models (Mengdi Xu et al., 2023)

{{<citation>}}

Mengdi Xu, Peide Huang, Wenhao Yu, Shiqi Liu, Xilun Zhang, Yaru Niu, Tingnan Zhang, Fei Xia, Jie Tan, Ding Zhao. (2023)  
**Creative Robot Tool Use with Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.13065v1)  

---


**ABSTRACT**  
Tool use is a hallmark of advanced intelligence, exemplified in both animal behavior and robotic capabilities. This paper investigates the feasibility of imbuing robots with the ability to creatively use tools in tasks that involve implicit physical constraints and long-term planning. Leveraging Large Language Models (LLMs), we develop RoboTool, a system that accepts natural language instructions and outputs executable code for controlling robots in both simulated and real-world environments. RoboTool incorporates four pivotal components: (i) an "Analyzer" that interprets natural language to discern key task-related concepts, (ii) a "Planner" that generates comprehensive strategies based on the language input and key concepts, (iii) a "Calculator" that computes parameters for each skill, and (iv) a "Coder" that translates these plans into executable Python code. Our results show that RoboTool can not only comprehend explicit or implicit physical constraints and environmental factors but also demonstrate creative tool use. Unlike traditional Task and Motion Planning (TAMP) methods that rely on explicit optimization, our LLM-based system offers a more flexible, efficient, and user-friendly solution for complex robotics tasks. Through extensive experiments, we validate that RoboTool is proficient in handling tasks that would otherwise be infeasible without the creative use of tools, thereby expanding the capabilities of robotic systems. Demos are available on our project page: https://creative-robotool.github.io/.

{{</citation>}}


### (124/160) CCIL: Continuity-based Data Augmentation for Corrective Imitation Learning (Liyiming Ke et al., 2023)

{{<citation>}}

Liyiming Ke, Yunchu Zhang, Abhay Deshpande, Siddhartha Srinivasa, Abhishek Gupta. (2023)  
**CCIL: Continuity-based Data Augmentation for Corrective Imitation Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.12972v1)  

---


**ABSTRACT**  
We present a new technique to enhance the robustness of imitation learning methods by generating corrective data to account for compounding errors and disturbances. While existing methods rely on interactive expert labeling, additional offline datasets, or domain-specific invariances, our approach requires minimal additional assumptions beyond access to expert data. The key insight is to leverage local continuity in the environment dynamics to generate corrective labels. Our method first constructs a dynamics model from the expert demonstration, encouraging local Lipschitz continuity in the learned model. In locally continuous regions, this model allows us to generate corrective labels within the neighborhood of the demonstrations but beyond the actual set of states and actions in the dataset. Training on this augmented data enhances the agent's ability to recover from perturbations and deal with compounding errors. We demonstrate the effectiveness of our generated labels through experiments in a variety of robotics domains in simulation that have distinct forms of continuity and discontinuity, including classic control problems, drone flying, navigation with high-dimensional sensor observations, legged locomotion, and tabletop manipulation.

{{</citation>}}


### (125/160) Eureka: Human-Level Reward Design via Coding Large Language Models (Yecheng Jason Ma et al., 2023)

{{<citation>}}

Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang, Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, Anima Anandkumar. (2023)  
**Eureka: Human-Level Reward Design via Coding Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12931v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have excelled as high-level semantic planners for sequential decision-making tasks. However, harnessing them to learn complex low-level manipulation tasks, such as dexterous pen spinning, remains an open problem. We bridge this fundamental gap and present Eureka, a human-level reward design algorithm powered by LLMs. Eureka exploits the remarkable zero-shot generation, code-writing, and in-context improvement capabilities of state-of-the-art LLMs, such as GPT-4, to perform evolutionary optimization over reward code. The resulting rewards can then be used to acquire complex skills via reinforcement learning. Without any task-specific prompting or pre-defined reward templates, Eureka generates reward functions that outperform expert human-engineered rewards. In a diverse suite of 29 open-source RL environments that include 10 distinct robot morphologies, Eureka outperforms human experts on 83% of the tasks, leading to an average normalized improvement of 52%. The generality of Eureka also enables a new gradient-free in-context learning approach to reinforcement learning from human feedback (RLHF), readily incorporating human inputs to improve the quality and the safety of the generated rewards without model updating. Finally, using Eureka rewards in a curriculum learning setting, we demonstrate for the first time, a simulated Shadow Hand capable of performing pen spinning tricks, adeptly manipulating a pen in circles at rapid speed.

{{</citation>}}


## eess.SP (1)



### (126/160) Active Sensing for Localization with Reconfigurable Intelligent Surface (Zhongze Zhang et al., 2023)

{{<citation>}}

Zhongze Zhang, Tao Jiang, Wei Yu. (2023)  
**Active Sensing for Localization with Reconfigurable Intelligent Surface**  

---
Primary Category: eess.SP  
Categories: cs-IT, eess-SP, eess.SP, math-IT  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.13160v1)  

---


**ABSTRACT**  
This paper addresses an uplink localization problem in which the base station (BS) aims to locate a remote user with the aid of reconfigurable intelligent surface (RIS). This paper proposes a strategy in which the user transmits pilots over multiple time frames, and the BS adaptively adjusts the RIS reflection coefficients based on the observations already received so far in order to produce an accurate estimate of the user location at the end. This is a challenging active sensing problem for which finding an optimal solution involves a search through a complicated functional space whose dimension increases with the number of measurements. In this paper, we show that the long short-term memory (LSTM) network can be used to exploit the latent temporal correlation between measurements to automatically construct scalable information vectors (called hidden state) based on the measurements. Subsequently, the state vector can be mapped to the RIS configuration for the next time frame in a codebook-free fashion via a deep neural network (DNN). After all the measurements have been received, a final DNN can be used to map the LSTM cell state to the estimated user equipment (UE) position. Numerical result shows that the proposed active RIS design results in lower localization error as compared to existing active and nonactive methods. The proposed solution produces interpretable results and is generalizable to early stopping in the sequence of sensing stages.

{{</citation>}}


## cs.HC (6)



### (127/160) Understanding Generative AI in Art: An Interview Study with Artists on G-AI from an HCI Perspective (Jingyu Shi et al., 2023)

{{<citation>}}

Jingyu Shi, Rahul Jain, Runlin Duan, Karthik Ramani. (2023)  
**Understanding Generative AI in Art: An Interview Study with Artists on G-AI from an HCI Perspective**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.13149v1)  

---


**ABSTRACT**  
The emergence of Generative Artificial Intelligence (G-AI) has changed the landscape of creative arts with its power to compose novel artwork and thus brought ethical concerns. Despite the efforts by prior works to address these concerns from technical and societal perspectives, there exists little discussion on this topic from an HCI point of view, considering the artists as human factors. We sought to investigate the impact of G-AI on artists, understanding the relationship between artists and G-AI, in order to motivate the underlying HCI research. We conducted semi-structured interviews with artists ($N=25$) from diverse artistic disciplines involved with G-AI in their artistic creation. We found (1) a dilemma among the artists, (2) a disparity in the understanding of G-AI between the artists and the AI developers(3) a tendency to oppose G-AI among the artists. We discuss the future opportunities of HCI research to tackle the problems identified from the interviews.

{{</citation>}}


### (128/160) Gender Biases in Error Mitigation by Voice Assistants (Amama Mahmood et al., 2023)

{{<citation>}}

Amama Mahmood, Chien-Ming Huang. (2023)  
**Gender Biases in Error Mitigation by Voice Assistants**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2310.13074v1)  

---


**ABSTRACT**  
Commercial voice assistants are largely feminized and associated with stereotypically feminine traits such as warmth and submissiveness. As these assistants continue to be adopted for everyday uses, it is imperative to understand how the portrayed gender shapes the voice assistant's ability to mitigate errors, which are still common in voice interactions. We report a study (N=40) that examined the effects of voice gender (feminine, ambiguous, masculine), error mitigation strategies (apology, compensation) and participant's gender on people's interaction behavior and perceptions of the assistant. Our results show that AI assistants that apologized appeared warmer than those offered compensation. Moreover, male participants preferred apologetic feminine assistants over apologetic masculine ones. Furthermore, male participants interrupted AI assistants regardless of perceived gender more frequently than female participants when errors occurred. Our results suggest that the perceived gender of a voice assistant biases user behavior, especially for male users, and that an ambiguous voice has the potential to reduce biases associated with gender-specific traits.

{{</citation>}}


### (129/160) Structured Generation and Exploration of Design Space with Large Language Models for Human-AI Co-Creation (Sangho Suh et al., 2023)

{{<citation>}}

Sangho Suh, Meng Chen, Bryan Min, Toby Jia-Jun Li, Haijun Xia. (2023)  
**Structured Generation and Exploration of Design Space with Large Language Models for Human-AI Co-Creation**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12953v2)  

---


**ABSTRACT**  
Thanks to their generative capabilities, large language models (LLMs) have become an invaluable tool for creative processes. These models have the capacity to produce hundreds and thousands of visual and textual outputs, offering abundant inspiration for creative endeavors. But are we harnessing their full potential? We argue that current interaction paradigms fall short, guiding users towards rapid convergence on a limited set of ideas, rather than empowering them to explore the vast latent design space in generative models. To address this limitation, we propose a framework that facilitates the structured generation of design space in which users can seamlessly explore, evaluate, and synthesize a multitude of responses. We demonstrate the feasibility and usefulness of this framework through the design and development of an interactive system, Luminate, and a user study with 8 professional writers. Our work advances how we interact with LLMs for creative tasks, introducing a way to harness the creative potential of LLMs.

{{</citation>}}


### (130/160) Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots (Xavier Puig et al., 2023)

{{<citation>}}

Xavier Puig, Eric Undersander, Andrew Szot, Mikael Dallaire Cote, Tsung-Yen Yang, Ruslan Partsey, Ruta Desai, Alexander William Clegg, Michal Hlavac, So Yeon Min, Vladimír Vondruš, Theophile Gervet, Vincent-Pierre Berges, John M. Turner, Oleksandr Maksymets, Zsolt Kira, Mrinal Kalakrishnan, Jitendra Malik, Devendra Singh Chaplot, Unnat Jain, Dhruv Batra, Akshara Rai, Roozbeh Mottaghi. (2023)  
**Habitat 3.0: A Co-Habitat for Humans, Avatars and Robots**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CV, cs-GR, cs-HC, cs-MA, cs-RO, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13724v1)  

---


**ABSTRACT**  
We present Habitat 3.0: a simulation platform for studying collaborative human-robot tasks in home environments. Habitat 3.0 offers contributions across three dimensions: (1) Accurate humanoid simulation: addressing challenges in modeling complex deformable bodies and diversity in appearance and motion, all while ensuring high simulation speed. (2) Human-in-the-loop infrastructure: enabling real human interaction with simulated robots via mouse/keyboard or a VR interface, facilitating evaluation of robot policies with human input. (3) Collaborative tasks: studying two collaborative tasks, Social Navigation and Social Rearrangement. Social Navigation investigates a robot's ability to locate and follow humanoid avatars in unseen environments, whereas Social Rearrangement addresses collaboration between a humanoid and robot while rearranging a scene. These contributions allow us to study end-to-end learned and heuristic baselines for human-robot collaboration in-depth, as well as evaluate them with humans in the loop. Our experiments demonstrate that learned robot policies lead to efficient task completion when collaborating with unseen humanoid agents and human partners that might exhibit behaviors that the robot has not seen before. Additionally, we observe emergent behaviors during collaborative task execution, such as the robot yielding space when obstructing a humanoid agent, thereby allowing the effective completion of the task by the humanoid agent. Furthermore, our experiments using the human-in-the-loop tool demonstrate that our automated evaluation with humanoids can provide an indication of the relative ordering of different policies when evaluated with real human collaborators. Habitat 3.0 unlocks interesting new features in simulators for Embodied AI, and we hope it paves the way for a new frontier of embodied human-AI interaction capabilities.

{{</citation>}}


### (131/160) Spatial and Temporal Attention-based emotion estimation on HRI-AVC dataset (Karthik Subramanian et al., 2023)

{{<citation>}}

Karthik Subramanian, Saurav Singh, Justin Namba, Jamison Heard, Christopher Kanan, Ferat Sahin. (2023)  
**Spatial and Temporal Attention-based emotion estimation on HRI-AVC dataset**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.12887v1)  

---


**ABSTRACT**  
Many attempts have been made at estimating discrete emotions (calmness, anxiety, boredom, surprise, anger) and continuous emotional measures commonly used in psychology, namely `valence' (The pleasantness of the emotion being displayed) and `arousal' (The intensity of the emotion being displayed). Existing methods to estimate arousal and valence rely on learning from data sets, where an expert annotator labels every image frame. Access to an expert annotator is not always possible, and the annotation can also be tedious. Hence it is more practical to obtain self-reported arousal and valence values directly from the human in a real-time Human-Robot collaborative setting. Hence this paper provides an emotion data set (HRI-AVC) obtained while conducting a human-robot interaction (HRI) task. The self-reported pair of labels in this data set is associated with a set of image frames. This paper also proposes a spatial and temporal attention-based network to estimate arousal and valence from this set of image frames. The results show that an attention-based network can estimate valence and arousal on the HRI-AVC data set even when Arousal and Valence values are unavailable per frame.

{{</citation>}}


### (132/160) Affective Conversational Agents: Understanding Expectations and Personal Influences (Javier Hernandez et al., 2023)

{{<citation>}}

Javier Hernandez, Jina Suh, Judith Amores, Kael Rowan, Gonzalo Ramos, Mary Czerwinski. (2023)  
**Affective Conversational Agents: Understanding Expectations and Personal Influences**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.12459v1)  

---


**ABSTRACT**  
The rise of AI conversational agents has broadened opportunities to enhance human capabilities across various domains. As these agents become more prevalent, it is crucial to investigate the impact of different affective abilities on their performance and user experience. In this study, we surveyed 745 respondents to understand the expectations and preferences regarding affective skills in various applications. Specifically, we assessed preferences concerning AI agents that can perceive, respond to, and simulate emotions across 32 distinct scenarios. Our results indicate a preference for scenarios that involve human interaction, emotional support, and creative tasks, with influences from factors such as emotional reappraisal and personality traits. Overall, the desired affective skills in AI agents depend largely on the application's context and nature, emphasizing the need for adaptability and context-awareness in the design of affective AI conversational agents.

{{</citation>}}


## cond-mat.mtrl-sci (1)



### (133/160) Approaches for Uncertainty Quantification of AI-predicted Material Properties: A Comparison (Francesca Tavazza et al., 2023)

{{<citation>}}

Francesca Tavazza, Kamal Choudhary, Brian DeCost. (2023)  
**Approaches for Uncertainty Quantification of AI-predicted Material Properties: A Comparison**  

---
Primary Category: cond-mat.mtrl-sci  
Categories: cond-mat-mtrl-sci, cond-mat.mtrl-sci, cs-LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13136v1)  

---


**ABSTRACT**  
The development of large databases of material properties, together with the availability of powerful computers, has allowed machine learning (ML) modeling to become a widely used tool for predicting material performances. While confidence intervals are commonly reported for such ML models, prediction intervals, i.e., the uncertainty on each prediction, are not as frequently available. Here, we investigate three easy-to-implement approaches to determine such individual uncertainty, comparing them across ten ML quantities spanning energetics, mechanical, electronic, optical, and spectral properties. Specifically, we focused on the Quantile approach, the direct machine learning of the prediction intervals and Ensemble methods.

{{</citation>}}


## eess.SY (1)



### (134/160) Deep Reinforcement Learning-based Intelligent Traffic Signal Controls with Optimized CO2 emissions (Pedram Agand et al., 2023)

{{<citation>}}

Pedram Agand, Alexey Iskrov, Mo Chen. (2023)  
**Deep Reinforcement Learning-based Intelligent Traffic Signal Controls with Optimized CO2 emissions**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.13129v2)  

---


**ABSTRACT**  
Nowadays, transportation networks face the challenge of sub-optimal control policies that can have adverse effects on human health, the environment, and contribute to traffic congestion. Increased levels of air pollution and extended commute times caused by traffic bottlenecks make intersection traffic signal controllers a crucial component of modern transportation infrastructure. Despite several adaptive traffic signal controllers in literature, limited research has been conducted on their comparative performance. Furthermore, despite carbon dioxide (CO2) emissions' significance as a global issue, the literature has paid limited attention to this area. In this report, we propose EcoLight, a reward shaping scheme for reinforcement learning algorithms that not only reduces CO2 emissions but also achieves competitive results in metrics such as travel time. We compare the performance of tabular Q-Learning, DQN, SARSA, and A2C algorithms using metrics such as travel time, CO2 emissions, waiting time, and stopped time. Our evaluation considers multiple scenarios that encompass a range of road users (trucks, buses, cars) with varying pollution levels.

{{</citation>}}


## stat.ML (3)



### (135/160) Sequence Length Independent Norm-Based Generalization Bounds for Transformers (Jacob Trauger et al., 2023)

{{<citation>}}

Jacob Trauger, Ambuj Tewari. (2023)  
**Sequence Length Independent Norm-Based Generalization Bounds for Transformers**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.13088v1)  

---


**ABSTRACT**  
This paper provides norm-based generalization bounds for the Transformer architecture that do not depend on the input sequence length. We employ a covering number based approach to prove our bounds. We use three novel covering number bounds for the function class of bounded linear transformations to upper bound the Rademacher complexity of the Transformer. Furthermore, we show this generalization bound applies to the common Transformer training technique of masking and then predicting the masked word. We also run a simulated study on a sparse majority data set that empirically validates our theoretical findings.

{{</citation>}}


### (136/160) On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers (Illia Horenko, 2023)

{{<citation>}}

Illia Horenko. (2023)  
**On existence, uniqueness and scalability of adversarial robustness measures for AI classifiers**  

---
Primary Category: stat.ML  
Categories: 68T01 (Primary), 68T99, 68Q32, 86A22, 92C50 (Secondary), cs-AI, cs-LG, stat-ML, stat.ML  
Keywords: AI, GLM  
[Paper Link](http://arxiv.org/abs/2310.14421v1)  

---


**ABSTRACT**  
Simply-verifiable mathematical conditions for existence, uniqueness and explicit analytical computation of minimal adversarial paths (MAP) and minimal adversarial distances (MAD) for (locally) uniquely-invertible classifiers, for generalized linear models (GLM), and for entropic AI (EAI) are formulated and proven. Practical computation of MAP and MAD, their comparison and interpretations for various classes of AI tools (for neuronal networks, boosted random forests, GLM and EAI) are demonstrated on the common synthetic benchmarks: on a double Swiss roll spiral and its extensions, as well as on the two biomedical data problems (for the health insurance claim predictions, and for the heart attack lethality classification). On biomedical applications it is demonstrated how MAP provides unique minimal patient-specific risk-mitigating interventions in the predefined subsets of accessible control variables.

{{</citation>}}


### (137/160) Neural Likelihood Approximation for Integer Valued Time Series Data (Luke O'Loughlin et al., 2023)

{{<citation>}}

Luke O'Loughlin, John Maclean, Andrew Black. (2023)  
**Neural Likelihood Approximation for Integer Valued Time Series Data**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2310.12544v1)  

---


**ABSTRACT**  
Stochastic processes defined on integer valued state spaces are popular within the physical and biological sciences. These models are necessary for capturing the dynamics of small systems where the individual nature of the populations cannot be ignored and stochastic effects are important. The inference of the parameters of such models, from time series data, is difficult due to intractability of the likelihood; current methods, based on simulations of the underlying model, can be so computationally expensive as to be prohibitive. In this paper we construct a neural likelihood approximation for integer valued time series data using causal convolutions, which allows us to evaluate the likelihood of the whole time series in parallel. We demonstrate our method by performing inference on a number of ecological and epidemiological models, showing that we can accurately approximate the true posterior while achieving significant computational speed ups in situations where current methods struggle.

{{</citation>}}


## cs.DC (2)



### (138/160) End-to-End Delay Minimization based on Joint Optimization of DNN Partitioning and Resource Allocation for Cooperative Edge Inference (Xinrui Ye et al., 2023)

{{<citation>}}

Xinrui Ye, Yanzan Sun, Dingzhu Wen, Guanjin Pan, Shunqing Zhang. (2023)  
**End-to-End Delay Minimization based on Joint Optimization of DNN Partitioning and Resource Allocation for Cooperative Edge Inference**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.12937v1)  

---


**ABSTRACT**  
Cooperative inference in Mobile Edge Computing (MEC), achieved by deploying partitioned Deep Neural Network (DNN) models between resource-constrained user equipments (UEs) and edge servers (ESs), has emerged as a promising paradigm. Firstly, we consider scenarios of continuous Artificial Intelligence (AI) task arrivals, like the object detection for video streams, and utilize a serial queuing model for the accurate evaluation of End-to-End (E2E) delay in cooperative edge inference. Secondly, to enhance the long-term performance of inference systems, we formulate a multi-slot stochastic E2E delay optimization problem that jointly considers model partitioning and multi-dimensional resource allocation. Finally, to solve this problem, we introduce a Lyapunov-guided Multi-Dimensional Optimization algorithm (LyMDO) that decouples the original problem into per-slot deterministic problems, where Deep Reinforcement Learning (DRL) and convex optimization are used for joint optimization of partitioning decisions and complementary resource allocation. Simulation results show that our approach effectively improves E2E delay while balancing long-term resource constraints.

{{</citation>}}


### (139/160) Reliable and Efficient In-Memory Fault Tolerance of Large Language Model Pretraining (Yuxin Wang et al., 2023)

{{<citation>}}

Yuxin Wang, Shaohuai Shi, Xin He, Zhenheng Tang, Xinglin Pan, Yang Zheng, Xiaoyu Wu, Amelie Chi Zhou, Bingsheng He, Xiaowen Chu. (2023)  
**Reliable and Efficient In-Memory Fault Tolerance of Large Language Model Pretraining**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-PF, cs.DC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12670v1)  

---


**ABSTRACT**  
Extensive system scales (i.e. thousands of GPU/TPUs) and prolonged training periods (i.e. months of pretraining) significantly escalate the probability of failures when training large language models (LLMs). Thus, efficient and reliable fault-tolerance methods are in urgent need. Checkpointing is the primary fault-tolerance method to periodically save parameter snapshots from GPU memory to disks via CPU memory. In this paper, we identify the frequency of existing checkpoint-based fault-tolerance being significantly limited by the storage I/O overheads, which results in hefty re-training costs on restarting from the nearest checkpoint. In response to this gap, we introduce an in-memory fault-tolerance framework for large-scale LLM pretraining. The framework boosts the efficiency and reliability of fault tolerance from three aspects: (1) Reduced Data Transfer and I/O: By asynchronously caching parameters, i.e., sharded model parameters, optimizer states, and RNG states, to CPU volatile memory, Our framework significantly reduces communication costs and bypasses checkpoint I/O. (2) Enhanced System Reliability: Our framework enhances parameter protection with a two-layer hierarchy: snapshot management processes (SMPs) safeguard against software failures, together with Erasure Coding (EC) protecting against node failures. This double-layered protection greatly improves the survival probability of the parameters compared to existing checkpointing methods. (3) Improved Snapshotting Frequency: Our framework achieves more frequent snapshotting compared with asynchronous checkpointing optimizations under the same saving time budget, which improves the fault tolerance efficiency. Empirical results demonstrate that Our framework minimizes the overhead of fault tolerance of LLM pretraining by effectively leveraging redundant CPU resources.

{{</citation>}}


## eess.IV (4)



### (140/160) Perceptual Assessment and Optimization of High Dynamic Range Image Rendering (Peibei Cao et al., 2023)

{{<citation>}}

Peibei Cao, Rafal K. Mantiuk, Kede Ma. (2023)  
**Perceptual Assessment and Optimization of High Dynamic Range Image Rendering**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.12877v2)  

---


**ABSTRACT**  
High dynamic range (HDR) imaging has gained increasing popularity for its ability to faithfully reproduce the luminance levels in natural scenes. Accordingly, HDR image quality assessment (IQA) is crucial but has been superficially treated. The majority of existing IQA models are developed for and calibrated against low dynamic range (LDR) images, which have been shown to be poorly correlated with human perception of HDR image quality. In this work, we propose a family of HDR IQA models by transferring the recent advances in LDR IQA. The key step in our approach is to specify a simple inverse display model that decomposes an HDR image to a set of LDR images with different exposures, which will be assessed by existing LDR quality models. The local quality scores of each exposure are then aggregated with the help of a simple well-exposedness measure into a global quality score for each exposure, which will be further weighted across exposures to obtain the overall quality score. When assessing LDR images, the proposed HDR quality models reduce gracefully to the original LDR ones with the same performance. Experiments on four human-rated HDR image datasets demonstrate that our HDR quality models are consistently better than existing IQA methods, including the HDR-VDP family. Moreover, we demonstrate their strengths in perceptual optimization of HDR novel view synthesis.

{{</citation>}}


### (141/160) Predicting Ovarian Cancer Treatment Response in Histopathology using Hierarchical Vision Transformers and Multiple Instance Learning (Jack Breen et al., 2023)

{{<citation>}}

Jack Breen, Katie Allen, Kieran Zucker, Geoff Hall, Nishant Ravikumar, Nicolas M. Orsi. (2023)  
**Predicting Ovarian Cancer Treatment Response in Histopathology using Hierarchical Vision Transformers and Multiple Instance Learning**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.12866v1)  

---


**ABSTRACT**  
For many patients, current ovarian cancer treatments offer limited clinical benefit. For some therapies, it is not possible to predict patients' responses, potentially exposing them to the adverse effects of treatment without any therapeutic benefit. As part of the automated prediction of treatment effectiveness in ovarian cancer using histopathological images (ATEC23) challenge, we evaluated the effectiveness of deep learning to predict whether a course of treatment including the antiangiogenic drug bevacizumab could contribute to remission or prevent disease progression for at least 6 months in a set of 282 histopathology whole slide images (WSIs) from 78 ovarian cancer patients. Our approach used a pretrained Hierarchical Image Pyramid Transformer (HIPT) to extract region-level features and an attention-based multiple instance learning (ABMIL) model to aggregate features and classify whole slides. The optimal HIPT-ABMIL model had an internal balanced accuracy of 60.2% +- 2.9% and an AUC of 0.646 +- 0.033. Histopathology-specific model pretraining was found to be beneficial to classification performance, though hierarchical transformers were not, with a ResNet feature extractor achieving similar performance. Due to the dataset being small and highly heterogeneous, performance was variable across 5-fold cross-validation folds, and there were some extreme differences between validation and test set performance within folds. The model did not generalise well to tissue microarrays, with accuracy worse than random chance. It is not yet clear whether ovarian cancer WSIs contain information that can be used to accurately predict treatment response, with further validation using larger, higher-quality datasets required.

{{</citation>}}


### (142/160) A reproducible 3D convolutional neural network with dual attention module (3D-DAM) for Alzheimer's disease classification (Gia Minh Hoang et al., 2023)

{{<citation>}}

Gia Minh Hoang, Youngjoo Lee, Jae Gwan Kim. (2023)  
**A reproducible 3D convolutional neural network with dual attention module (3D-DAM) for Alzheimer's disease classification**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.12574v1)  

---


**ABSTRACT**  
Alzheimer's disease is one of the most common types of neurodegenerative disease, characterized by the accumulation of amyloid-beta plaque and tau tangles. Recently, deep learning approaches have shown promise in Alzheimer's disease diagnosis. In this study, we propose a reproducible model that utilizes a 3D convolutional neural network with a dual attention module for Alzheimer's disease classification. We trained the model in the ADNI database and verified the generalizability of our method in two independent datasets (AIBL and OASIS1). Our method achieved state-of-the-art classification performance, with an accuracy of 91.94% for MCI progression classification and 96.30% for Alzheimer's disease classification on the ADNI dataset. Furthermore, the model demonstrated good generalizability, achieving an accuracy of 86.37% on the AIBL dataset and 83.42% on the OASIS1 dataset. These results indicate that our proposed approach has competitive performance and generalizability when compared to recent studies in the field.

{{</citation>}}


### (143/160) DA-TransUNet: Integrating Spatial and Channel Dual Attention with Transformer U-Net for Medical Image Segmentation (Guanqun Sun et al., 2023)

{{<citation>}}

Guanqun Sun, Yizhi Pan, Weikun Kong, Zichang Xu, Jianhua Ma, Teeradaj Racharak, Le-Minh Nguyen, Junyi Xin. (2023)  
**DA-TransUNet: Integrating Spatial and Channel Dual Attention with Transformer U-Net for Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-GR, cs-LG, eess-IV, eess.IV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2310.12570v1)  

---


**ABSTRACT**  
Great progress has been made in automatic medical image segmentation due to powerful deep representation learning. The influence of transformer has led to research into its variants, and large-scale replacement of traditional CNN modules. However, such trend often overlooks the intrinsic feature extraction capabilities of the transformer and potential refinements to both the model and the transformer module through minor adjustments. This study proposes a novel deep medical image segmentation framework, called DA-TransUNet, aiming to introduce the Transformer and dual attention block into the encoder and decoder of the traditional U-shaped architecture. Unlike prior transformer-based solutions, our DA-TransUNet utilizes attention mechanism of transformer and multifaceted feature extraction of DA-Block, which can efficiently combine global, local, and multi-scale features to enhance medical image segmentation. Meanwhile, experimental results show that a dual attention block is added before the Transformer layer to facilitate feature extraction in the U-net structure. Furthermore, incorporating dual attention blocks in skip connections can enhance feature transfer to the decoder, thereby improving image segmentation performance. Experimental results across various benchmark of medical image segmentation reveal that DA-TransUNet significantly outperforms the state-of-the-art methods. The codes and parameters of our model will be publicly available at https://github.com/SUN-1024/DA-TransUnet.

{{</citation>}}


## cs.CR (4)



### (144/160) Prompt Injection Attacks and Defenses in LLM-Integrated Applications (Yupei Liu et al., 2023)

{{<citation>}}

Yupei Liu, Yuqi Jia, Runpeng Geng, Jinyuan Jia, Neil Zhenqiang Gong. (2023)  
**Prompt Injection Attacks and Defenses in LLM-Integrated Applications**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12815v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) are increasingly deployed as the backend for a variety of real-world applications called LLM-Integrated Applications. Multiple recent works showed that LLM-Integrated Applications are vulnerable to prompt injection attacks, in which an attacker injects malicious instruction/data into the input of those applications such that they produce results as the attacker desires. However, existing works are limited to case studies. As a result, the literature lacks a systematic understanding of prompt injection attacks and their defenses. We aim to bridge the gap in this work. In particular, we propose a general framework to formalize prompt injection attacks. Existing attacks, which are discussed in research papers and blog posts, are special cases in our framework. Our framework enables us to design a new attack by combining existing attacks. Moreover, we also propose a framework to systematize defenses against prompt injection attacks. Using our frameworks, we conduct a systematic evaluation on prompt injection attacks and their defenses with 10 LLMs and 7 tasks. We hope our frameworks can inspire future research in this field. Our code is available at https://github.com/liu00222/Open-Prompt-Injection.

{{</citation>}}


### (145/160) RANDGENER: Distributed Randomness Beacon from Verifiable Delay Function (Arup Mondal et al., 2023)

{{<citation>}}

Arup Mondal, Ruthu Hulikal Rooparaghunath, Debayan Gupta. (2023)  
**RANDGENER: Distributed Randomness Beacon from Verifiable Delay Function**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2310.12693v1)  

---


**ABSTRACT**  
Buoyed by the excitement around secure decentralized applications, the last few decades have seen numerous constructions of distributed randomness beacons (DRB) along with use cases; however, a secure DRB (in many variations) remains an open problem. We further note that it is natural to want some kind of reward for participants who spend time and energy evaluating the randomness beacon value -- this is already common in distributed protocols.   In this work, we present RandGener, a novel $n$-party commit-reveal-recover (or collaborative) DRB protocol with a novel reward and penalty mechanism along with a set of realistic guarantees. We design our protocol using trapdoor watermarkable verifiable delay functions in the RSA group setting (without requiring a trusted dealer or distributed key generation).

{{</citation>}}


### (146/160) SecurityNet: Assessing Machine Learning Vulnerabilities on Public Models (Boyang Zhang et al., 2023)

{{<citation>}}

Boyang Zhang, Zheng Li, Ziqing Yang, Xinlei He, Michael Backes, Mario Fritz, Yang Zhang. (2023)  
**SecurityNet: Assessing Machine Learning Vulnerabilities on Public Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.12665v1)  

---


**ABSTRACT**  
While advanced machine learning (ML) models are deployed in numerous real-world applications, previous works demonstrate these models have security and privacy vulnerabilities. Various empirical research has been done in this field. However, most of the experiments are performed on target ML models trained by the security researchers themselves. Due to the high computational resource requirement for training advanced models with complex architectures, researchers generally choose to train a few target models using relatively simple architectures on typical experiment datasets. We argue that to understand ML models' vulnerabilities comprehensively, experiments should be performed on a large set of models trained with various purposes (not just the purpose of evaluating ML attacks and defenses). To this end, we propose using publicly available models with weights from the Internet (public models) for evaluating attacks and defenses on ML models. We establish a database, namely SecurityNet, containing 910 annotated image classification models. We then analyze the effectiveness of several representative attacks/defenses, including model stealing attacks, membership inference attacks, and backdoor detection on these public models. Our evaluation empirically shows the performance of these attacks/defenses can vary significantly on public models compared to self-trained models. We share SecurityNet with the research community. and advocate researchers to perform experiments on public models to better demonstrate their proposed methods' effectiveness in the future.

{{</citation>}}


### (147/160) Privacy Preserving Large Language Models: ChatGPT Case Study Based Vision and Framework (Imdad Ullah et al., 2023)

{{<citation>}}

Imdad Ullah, Najm Hassan, Sukhpal Singh Gill, Basem Suleiman, Tariq Ahamed Ahanger, Zawar Shah, Junaid Qadir, Salil S. Kanhere. (2023)  
**Privacy Preserving Large Language Models: ChatGPT Case Study Based Vision and Framework**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, ChatGPT, GPT, Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.12523v1)  

---


**ABSTRACT**  
The generative Artificial Intelligence (AI) tools based on Large Language Models (LLMs) use billions of parameters to extensively analyse large datasets and extract critical private information such as, context, specific details, identifying information etc. This have raised serious threats to user privacy and reluctance to use such tools. This article proposes the conceptual model called PrivChatGPT, a privacy-preserving model for LLMs that consists of two main components i.e., preserving user privacy during the data curation/pre-processing together with preserving private context and the private training process for large-scale data. To demonstrate its applicability, we show how a private mechanism could be integrated into the existing model for training LLMs to protect user privacy; specifically, we employed differential privacy and private training using Reinforcement Learning (RL). We measure the privacy loss and evaluate the measure of uncertainty or randomness once differential privacy is applied. It further recursively evaluates the level of privacy guarantees and the measure of uncertainty of public database and resources, during each update when new information is added for training purposes. To critically evaluate the use of differential privacy for private LLMs, we hypothetically compared other mechanisms e..g, Blockchain, private information retrieval, randomisation, for various performance measures such as the model performance and accuracy, computational complexity, privacy vs. utility etc. We conclude that differential privacy, randomisation, and obfuscation can impact utility and performance of trained models, conversely, the use of ToR, Blockchain, and PIR may introduce additional computational complexity and high training latency. We believe that the proposed model could be used as a benchmark for proposing privacy preserving LLMs for generative AI tools.

{{</citation>}}


## cs.SE (2)



### (148/160) Patch-CLIP: A Patch-Text Pre-Trained Model (Xunzhu Tang et al., 2023)

{{<citation>}}

Xunzhu Tang, Zhenghan Chen, Saad Ezzini, Haoye Tian, Jacques Klein, Tegawende F. Bissyande. (2023)  
**Patch-CLIP: A Patch-Text Pre-Trained Model**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BLEU, Pre-Trained Model  
[Paper Link](http://arxiv.org/abs/2310.12753v1)  

---


**ABSTRACT**  
In recent years, patch representation learning has emerged as a necessary research direction for exploiting the capabilities of machine learning in software generation. These representations have driven significant performance enhancements across a variety of tasks involving code changes. While the progress is undeniable, a common limitation among existing models is their specialization: they predominantly excel in either predictive tasks, such as security patch classification, or in generative tasks such as patch description generation. This dichotomy is further exacerbated by a prevalent dependency on potentially noisy data sources. Specifically, many models utilize patches integrated with Abstract Syntax Trees (AST) that, unfortunately, may contain parsing inaccuracies, thus acting as a suboptimal source of supervision. In response to these challenges, we introduce PATCH-CLIP, a novel pre-training framework for patches and natural language text. PATCH-CLIP deploys a triple-loss training strategy for 1) patch-description contrastive learning, which enables to separate patches and descriptions in the embedding space, 2) patch-description matching, which ensures that each patch is associated to its description in the embedding space, and 3) patch-description generation, which ensures that the patch embedding is effective for generation. These losses are implemented for joint learning to achieve good performance in both predictive and generative tasks involving patches. Empirical evaluations focusing on patch description generation, demonstrate that PATCH-CLIP sets new state of the art performance, consistently outperforming the state-of-the-art in metrics like BLEU, ROUGE-L, METEOR, and Recall.

{{</citation>}}


### (149/160) Automated Repair of Declarative Software Specifications in the Era of Large Language Models (Md Rashedul Hasan et al., 2023)

{{<citation>}}

Md Rashedul Hasan, Jiawei Li, Iftekhar Ahmed, Hamid Bagheri. (2023)  
**Automated Repair of Declarative Software Specifications in the Era of Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-LG, cs-SE, cs.SE  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.12425v1)  

---


**ABSTRACT**  
The growing adoption of declarative software specification languages, coupled with their inherent difficulty in debugging, has underscored the need for effective and automated repair techniques applicable to such languages. Researchers have recently explored various methods to automatically repair declarative software specifications, such as template-based repair, feedback-driven iterative repair, and bounded exhaustive approaches. The latest developments in large language models provide new opportunities for the automatic repair of declarative specifications. In this study, we assess the effectiveness of utilizing OpenAI's ChatGPT to repair software specifications written in the Alloy declarative language. Unlike imperative languages, specifications in Alloy are not executed but rather translated into logical formulas and evaluated using backend constraint solvers to identify specification instances and counterexamples to assertions. Our evaluation focuses on ChatGPT's ability to improve the correctness and completeness of Alloy declarative specifications through automatic repairs. We analyze the results produced by ChatGPT and compare them with those of leading automatic Alloy repair methods. Our study revealed that while ChatGPT falls short in comparison to existing techniques, it was able to successfully repair bugs that no other technique could address. Our analysis also identified errors in ChatGPT's generated repairs, including improper operator usage, type errors, higher-order logic misuse, and relational arity mismatches. Additionally, we observed instances of hallucinations in ChatGPT-generated repairs and inconsistency in its results. Our study provides valuable insights for software practitioners, researchers, and tool builders considering ChatGPT for declarative specification repairs.

{{</citation>}}


## cs.DL (2)



### (150/160) The Botization of Science? Large-scale study of the presence and impact of Twitter bots in science dissemination (Wenceslao Arroyo-Machado et al., 2023)

{{<citation>}}

Wenceslao Arroyo-Machado, Enrique Herrera-Viedma, Daniel Torres-Salinas. (2023)  
**The Botization of Science? Large-scale study of the presence and impact of Twitter bots in science dissemination**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs-SI, cs.DL, physics-soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.12741v1)  

---


**ABSTRACT**  
Twitter bots are a controversial element of the platform, and their negative impact is well known. In the field of scientific communication, they have been perceived in a more positive light, and the accounts that serve as feeds alerting about scientific publications are quite common. However, despite being aware of the presence of bots in the dissemination of science, no large-scale estimations have been made nor has it been evaluated if they can truly interfere with altmetrics. Analyzing a dataset of 3,744,231 papers published between 2017 and 2021 and their associated 51,230,936 Twitter mentions, our goal was to determine the volume of publications mentioned by bots and whether they skew altmetrics indicators. Using the BotometerLite API, we categorized Twitter accounts based on their likelihood of being bots. The results showed that 11,073 accounts (0.23% of total users) exhibited automated behavior, contributing to 4.72% of all mentions. A significant bias was observed in the activity of bots. Their presence was particularly pronounced in disciplines such as Mathematics, Physics, and Space Sciences, with some specialties even exceeding 70% of the tweets. However, these are extreme cases, and the impact of this activity on altmetrics varies by speciality, with minimal influence in Arts & Humanities and Social Sciences. This research emphasizes the importance of distinguishing between specialties and disciplines when using Twitter as an altmetric.

{{</citation>}}


### (151/160) Metadata for Scientific Experiment Reporting: A Case Study in Metal-Organic Frameworks (Xintong Zhao et al., 2023)

{{<citation>}}

Xintong Zhao, Kyle Langlois, Jacob Furst, Scott McClellan, Xiaohua Hu, Yuan An, Diego A. Gómez-Gualdrón, Fernando J. Uribe-Romo, Jane Greenberg. (2023)  
**Metadata for Scientific Experiment Reporting: A Case Study in Metal-Organic Frameworks**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs.DL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.12417v1)  

---


**ABSTRACT**  
Research methods and procedures are core aspects of the research process. Metadata focused on these components is critical to supporting the FAIR principles, particularly reproducibility. The research reported on in this paper presents a methodological framework for metadata documentation supporting the reproducibility of research producing Metal Organic Frameworks (MOFs). The MOF case study involved natural language processing to extract key synthesis experiment information from a corpus of research literature. Following, a classification activity was performed by domain experts to identify entity-relation pairs. Results include: 1) a research framework for metadata design, 2) a metadata schema that includes nine entities and two relationships for reporting MOF synthesis experiments, and 3) a growing database of MOF synthesis reports structured by our metadata scheme. The metadata schema is intended to support discovery and reproducibility of metal-organic framework research and the FAIR principles. The paper provides background information, identifies the research goals and objectives, research design, results, a discussion, and the conclusion.

{{</citation>}}


## cs.NE (2)



### (152/160) LASER: Linear Compression in Wireless Distributed Optimization (Ashok Vardhan Makkuva et al., 2023)

{{<citation>}}

Ashok Vardhan Makkuva, Marco Bondaschi, Thijs Vogels, Martin Jaggi, Hyeji Kim, Michael C. Gastpar. (2023)  
**LASER: Linear Compression in Wireless Distributed Optimization**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-IT, cs-LG, cs-NE, cs.NE, math-IT  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.13033v1)  

---


**ABSTRACT**  
Data-parallel SGD is the de facto algorithm for distributed optimization, especially for large scale machine learning. Despite its merits, communication bottleneck is one of its persistent issues. Most compression schemes to alleviate this either assume noiseless communication links, or fail to achieve good performance on practical tasks. In this paper, we close this gap and introduce LASER: LineAr CompreSsion in WirEless DistRibuted Optimization. LASER capitalizes on the inherent low-rank structure of gradients and transmits them efficiently over the noisy channels. Whilst enjoying theoretical guarantees similar to those of the classical SGD, LASER shows consistent gains over baselines on a variety of practical benchmarks. In particular, it outperforms the state-of-the-art compression schemes on challenging computer vision and GPT language modeling tasks. On the latter, we obtain $50$-$64 \%$ improvement in perplexity over our baselines for noisy channels.

{{</citation>}}


### (153/160) Large Language Model for Multi-objective Evolutionary Optimization (Fei Liu et al., 2023)

{{<citation>}}

Fei Liu, Xi Lin, Zhenkun Wang, Shunyu Yao, Xialiang Tong, Mingxuan Yuan, Qingfu Zhang. (2023)  
**Large Language Model for Multi-objective Evolutionary Optimization**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-CL, cs-ET, cs-NE, cs.NE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12541v1)  

---


**ABSTRACT**  
Multiobjective evolutionary algorithms (MOEAs) are major methods for solving multiobjective optimization problems (MOPs). Many MOEAs have been proposed in the past decades, of which the operators need carefully handcrafted design with domain knowledge. Recently, some attempts have been made to replace the manually designed operators in MOEAs with learning-based operators (e.g., neural network models). However, much effort is still required for designing and training such models, and the learned operators might not generalize well to solve new problems. To tackle the above challenges, this work investigates a novel approach that leverages the powerful large language model (LLM) to design MOEA operators. With proper prompt engineering, we successfully let a general LLM serve as a black-box search operator for decomposition-based MOEA (MOEA/D) in a zero-shot manner. In addition, by learning from the LLM behavior, we further design an explicit white-box operator with randomness and propose a new version of decomposition-based MOEA, termed MOEA/D-LO. Experimental studies on different test benchmarks show that our proposed method can achieve competitive performance with widely used MOEAs. It is also promising to see the operator only learned from a few instances can have robust generalization performance on unseen problems with quite different patterns and settings. The results reveal the potential benefits of using pre-trained LLMs in the design of MOEAs.

{{</citation>}}


## cs.MM (1)



### (154/160) Generating Robust Adversarial Examples against Online Social Networks (OSNs) (Jun Liu et al., 2023)

{{<citation>}}

Jun Liu, Jiantao Zhou, Haiwei Wu, Weiwei Sun, Jinyu Tian. (2023)  
**Generating Robust Adversarial Examples against Online Social Networks (OSNs)**  

---
Primary Category: cs.MM  
Categories: cs-CV, cs-MM, cs.MM  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2310.12708v1)  

---


**ABSTRACT**  
Online Social Networks (OSNs) have blossomed into prevailing transmission channels for images in the modern era. Adversarial examples (AEs) deliberately designed to mislead deep neural networks (DNNs) are found to be fragile against the inevitable lossy operations conducted by OSNs. As a result, the AEs would lose their attack capabilities after being transmitted over OSNs. In this work, we aim to design a new framework for generating robust AEs that can survive the OSN transmission; namely, the AEs before and after the OSN transmission both possess strong attack capabilities. To this end, we first propose a differentiable network termed SImulated OSN (SIO) to simulate the various operations conducted by an OSN. Specifically, the SIO network consists of two modules: 1) a differentiable JPEG layer for approximating the ubiquitous JPEG compression and 2) an encoder-decoder subnetwork for mimicking the remaining operations. Based upon the SIO network, we then formulate an optimization framework to generate robust AEs by enforcing model outputs with and without passing through the SIO to be both misled. Extensive experiments conducted over Facebook, WeChat and QQ demonstrate that our attack methods produce more robust AEs than existing approaches, especially under small distortion constraints; the performance gain in terms of Attack Success Rate (ASR) could be more than 60%. Furthermore, we build a public dataset containing more than 10,000 pairs of AEs processed by Facebook, WeChat or QQ, facilitating future research in the robust AEs generation. The dataset and code are available at https://github.com/csjunjun/RobustOSNAttack.git.

{{</citation>}}


## cs.LO (1)



### (155/160) Embedding Pure Type Systems in the lambda-Pi-calculus modulo (Denis Cousineau et al., 2023)

{{<citation>}}

Denis Cousineau, Gilles Dowek. (2023)  
**Embedding Pure Type Systems in the lambda-Pi-calculus modulo**  

---
Primary Category: cs.LO  
Categories: cs-LO, cs.LO  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.12540v1)  

---


**ABSTRACT**  
The lambda-Pi-calculus allows to express proofs of minimal predicate logic. It can be extended, in a very simple way, by adding computation rules. This leads to the lambda-Pi-calculus modulo. We show in this paper that this simple extension is surprisingly expressive and, in particular, that all functional Pure Type Systems, such as the system F, or the Calculus of Constructions, can be embedded in it. And, moreover, that this embedding is conservative under termination hypothesis.

{{</citation>}}


## q-fin.PR (1)



### (156/160) American Option Pricing using Self-Attention GRU and Shapley Value Interpretation (Yanhui Shen, 2023)

{{<citation>}}

Yanhui Shen. (2023)  
**American Option Pricing using Self-Attention GRU and Shapley Value Interpretation**  

---
Primary Category: q-fin.PR  
Categories: cs-LG, q-fin-PR, q-fin.PR  
Keywords: Attention, LSTM, Self-Attention  
[Paper Link](http://arxiv.org/abs/2310.12500v1)  

---


**ABSTRACT**  
Options, serving as a crucial financial instrument, are used by investors to manage and mitigate their investment risks within the securities market. Precisely predicting the present price of an option enables investors to make informed and efficient decisions. In this paper, we propose a machine learning method for forecasting the prices of SPY (ETF) option based on gated recurrent unit (GRU) and self-attention mechanism. We first partitioned the raw dataset into 15 subsets according to moneyness and days to maturity criteria. For each subset, we matched the corresponding U.S. government bond rates and Implied Volatility Indices. This segmentation allows for a more insightful exploration of the impacts of risk-free rates and underlying volatility on option pricing. Next, we built four different machine learning models, including multilayer perceptron (MLP), long short-term memory (LSTM), self-attention LSTM, and self-attention GRU in comparison to the traditional binomial model. The empirical result shows that self-attention GRU with historical data outperforms other models due to its ability to capture complex temporal dependencies and leverage the contextual information embedded in the historical data. Finally, in order to unveil the "black box" of artificial intelligence, we employed the SHapley Additive exPlanations (SHAP) method to interpret and analyze the prediction results of the self-attention GRU model with historical data. This provides insights into the significance and contributions of different input features on the pricing of American-style options.

{{</citation>}}


## eess.AS (1)



### (157/160) An Exploration of In-Context Learning for Speech Language Model (Ming-Hao Hsu et al., 2023)

{{<citation>}}

Ming-Hao Hsu, Kai-Wei Chang, Shang-Wen Li, Hung-yi Lee. (2023)  
**An Exploration of In-Context Learning for Speech Language Model**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-CL, eess-AS, eess.AS  
Keywords: GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.12477v1)  

---


**ABSTRACT**  
Ever since the development of GPT-3 in the natural language processing (NLP) field, in-context learning (ICL) has played an important role in utilizing large language models (LLMs). By presenting the LM utterance-label demonstrations at the input, the LM can accomplish few-shot learning without relying on gradient descent or requiring explicit modification of its parameters. This enables the LM to learn and adapt in a black-box manner. Despite the success of ICL in NLP, little work is exploring the possibility of ICL in speech processing. This study proposes the first exploration of ICL with a speech LM without text supervision. We first show that the current speech LM does not have the ICL capability. With the proposed warmup training, the speech LM can, therefore, perform ICL on unseen tasks. In this work, we verify the feasibility of ICL for speech LM on speech classification tasks.

{{</citation>}}


## cs.IR (1)



### (158/160) Know Where to Go: Make LLM a Relevant, Responsible, and Trustworthy Searcher (Xiang Shi et al., 2023)

{{<citation>}}

Xiang Shi, Jiawei Liu, Yinpeng Liu, Qikai Cheng, Wei Lu. (2023)  
**Know Where to Go: Make LLM a Relevant, Responsible, and Trustworthy Searcher**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.12443v1)  

---


**ABSTRACT**  
The advent of Large Language Models (LLMs) has shown the potential to improve relevance and provide direct answers in web searches. However, challenges arise in validating the reliability of generated results and the credibility of contributing sources, due to the limitations of traditional information retrieval algorithms and the LLM hallucination problem. Aiming to create a "PageRank" for the LLM era, we strive to transform LLM into a relevant, responsible, and trustworthy searcher. We propose a novel generative retrieval framework leveraging the knowledge of LLMs to foster a direct link between queries and online sources. This framework consists of three core modules: Generator, Validator, and Optimizer, each focusing on generating trustworthy online sources, verifying source reliability, and refining unreliable sources, respectively. Extensive experiments and evaluations highlight our method's superior relevance, responsibility, and trustfulness against various SOTA methods.

{{</citation>}}


## q-bio.NC (1)



### (159/160) AI for Mathematics: A Cognitive Science Perspective (Cedegao E. Zhang et al., 2023)

{{<citation>}}

Cedegao E. Zhang, Katherine M. Collins, Adrian Weller, Joshua B. Tenenbaum. (2023)  
**AI for Mathematics: A Cognitive Science Perspective**  

---
Primary Category: q-bio.NC  
Categories: cs-AI, q-bio-NC, q-bio.NC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.13021v1)  

---


**ABSTRACT**  
Mathematics is one of the most powerful conceptual systems developed and used by the human species. Dreams of automated mathematicians have a storied history in artificial intelligence (AI). Rapid progress in AI, particularly propelled by advances in large language models (LLMs), has sparked renewed, widespread interest in building such systems. In this work, we reflect on these goals from a \textit{cognitive science} perspective. We call attention to several classical and ongoing research directions from cognitive science, which we believe are valuable for AI practitioners to consider when seeking to build truly human (or superhuman)-level mathematical systems. We close with open discussions and questions that we believe necessitate a multi-disciplinary perspective -- cognitive scientists working in tandem with AI researchers and mathematicians -- as we move toward better mathematical AI systems which not only help us push the frontier of the mathematics, but also offer glimpses into how we as humans are even capable of such great cognitive feats.

{{</citation>}}


## cs.SD (1)



### (160/160) Loop Copilot: Conducting AI Ensembles for Music Generation and Iterative Editing (Yixiao Zhang et al., 2023)

{{<citation>}}

Yixiao Zhang, Akira Maezawa, Gus Xia, Kazuhiko Yamamoto, Simon Dixon. (2023)  
**Loop Copilot: Conducting AI Ensembles for Music Generation and Iterative Editing**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-HC, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.12404v1)  

---


**ABSTRACT**  
Creating music is iterative, requiring varied methods at each stage. However, existing AI music systems fall short in orchestrating multiple subsystems for diverse needs. To address this gap, we introduce Loop Copilot, a novel system that enables users to generate and iteratively refine music through an interactive, multi-round dialogue interface. The system uses a large language model to interpret user intentions and select appropriate AI models for task execution. Each backend model is specialized for a specific task, and their outputs are aggregated to meet the user's requirements. To ensure musical coherence, essential attributes are maintained in a centralized table. We evaluate the effectiveness of the proposed system through semi-structured interviews and questionnaires, highlighting its utility not only in facilitating music creation but also its potential for broader applications.

{{</citation>}}
