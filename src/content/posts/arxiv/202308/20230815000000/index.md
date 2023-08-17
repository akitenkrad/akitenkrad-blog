---
draft: false
title: "arXiv @ 2023.08.15"
date: 2023-08-15
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.15"
    identifier: arxiv_20230815
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (12)](#cscv-12)
- [cs.LG (10)](#cslg-10)
- [cs.CL (6)](#cscl-6)
- [cs.IR (1)](#csir-1)
- [cs.CR (2)](#cscr-2)
- [cs.RO (2)](#csro-2)
- [cs.DC (1)](#csdc-1)
- [cs.SE (1)](#csse-1)

## cs.CV (12)



### (1/35) Improving Face Recognition from Caption Supervision with Multi-Granular Contextual Feature Aggregation (Md Mahedi Hasan et al., 2023)

{{<citation>}}

Md Mahedi Hasan, Nasser Nasrabadi. (2023)  
**Improving Face Recognition from Caption Supervision with Multi-Granular Contextual Feature Aggregation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.06866v1)  

---


**ABSTRACT**  
We introduce caption-guided face recognition (CGFR) as a new framework to improve the performance of commercial-off-the-shelf (COTS) face recognition (FR) systems. In contrast to combining soft biometrics (eg., facial marks, gender, and age) with face images, in this work, we use facial descriptions provided by face examiners as a piece of auxiliary information. However, due to the heterogeneity of the modalities, improving the performance by directly fusing the textual and facial features is very challenging, as both lie in different embedding spaces. In this paper, we propose a contextual feature aggregation module (CFAM) that addresses this issue by effectively exploiting the fine-grained word-region interaction and global image-caption association. Specifically, CFAM adopts a self-attention and a cross-attention scheme for improving the intra-modality and inter-modality relationship between the image and textual features, respectively. Additionally, we design a textual feature refinement module (TFRM) that refines the textual features of the pre-trained BERT encoder by updating the contextual embeddings. This module enhances the discriminative power of textual features with a cross-modal projection loss and realigns the word and caption embeddings with visual features by incorporating a visual-semantic alignment loss. We implemented the proposed CGFR framework on two face recognition models (ArcFace and AdaFace) and evaluated its performance on the Multi-Modal CelebA-HQ dataset. Our framework significantly improves the performance of ArcFace in both 1:1 verification and 1:N identification protocol.

{{</citation>}}


### (2/35) Manifold DivideMix: A Semi-Supervised Contrastive Learning Framework for Severe Label Noise (Fahimeh Fooladgar et al., 2023)

{{<citation>}}

Fahimeh Fooladgar, Minh Nguyen Nhat To, Parvin Mousavi, Purang Abolmaesumi. (2023)  
**Manifold DivideMix: A Semi-Supervised Contrastive Learning Framework for Severe Label Noise**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.06861v1)  

---


**ABSTRACT**  
Deep neural networks have proven to be highly effective when large amounts of data with clean labels are available. However, their performance degrades when training data contains noisy labels, leading to poor generalization on the test set. Real-world datasets contain noisy label samples that either have similar visual semantics to other classes (in-distribution) or have no semantic relevance to any class (out-of-distribution) in the dataset. Most state-of-the-art methods leverage ID labeled noisy samples as unlabeled data for semi-supervised learning, but OOD labeled noisy samples cannot be used in this way because they do not belong to any class within the dataset. Hence, in this paper, we propose incorporating the information from all the training data by leveraging the benefits of self-supervised training. Our method aims to extract a meaningful and generalizable embedding space for each sample regardless of its label. Then, we employ a simple yet effective K-nearest neighbor method to remove portions of out-of-distribution samples. By discarding these samples, we propose an iterative "Manifold DivideMix" algorithm to find clean and noisy samples, and train our model in a semi-supervised way. In addition, we propose "MixEMatch", a new algorithm for the semi-supervised step that involves mixup augmentation at the input and final hidden representations of the model. This will extract better representations by interpolating both in the input and manifold spaces. Extensive experiments on multiple synthetic-noise image benchmarks and real-world web-crawled datasets demonstrate the effectiveness of our proposed framework. Code is available at https://github.com/Fahim-F/ManifoldDivideMix.

{{</citation>}}


### (3/35) Modified Topological Image Preprocessing for Skin Lesion Classifications (Hong Cheng et al., 2023)

{{<citation>}}

Hong Cheng, Rebekah Leamons, Ahmad Al Shami. (2023)  
**Modified Topological Image Preprocessing for Skin Lesion Classifications**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.06796v1)  

---


**ABSTRACT**  
This paper proposes a modified Topological Data Analysis model for skin images preprocessing and enhancements. The skin lesion dataset HAM10000 used with the intention of identifying the important objects in relevant regions of the images. In order to evaluate both the original dataset and the preprocessed dataset, Deep Convolutional Neural Network and Vision Transformer models were utilized to train both models. After training, the experimental results demonstrate that the images preprocessed using the Modified Topological Data Analysis consistently perform better.

{{</citation>}}


### (4/35) Shrinking Class Space for Enhanced Certainty in Semi-Supervised Learning (Lihe Yang et al., 2023)

{{<citation>}}

Lihe Yang, Zhen Zhao, Lei Qi, Yu Qiao, Yinghuan Shi, Hengshuang Zhao. (2023)  
**Shrinking Class Space for Enhanced Certainty in Semi-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.06777v1)  

---


**ABSTRACT**  
Semi-supervised learning is attracting blooming attention, due to its success in combining unlabeled data. To mitigate potentially incorrect pseudo labels, recent frameworks mostly set a fixed confidence threshold to discard uncertain samples. This practice ensures high-quality pseudo labels, but incurs a relatively low utilization of the whole unlabeled set. In this work, our key insight is that these uncertain samples can be turned into certain ones, as long as the confusion classes for the top-1 class are detected and removed. Invoked by this, we propose a novel method dubbed ShrinkMatch to learn uncertain samples. For each uncertain sample, it adaptively seeks a shrunk class space, which merely contains the original top-1 class, as well as remaining less likely classes. Since the confusion ones are removed in this space, the re-calculated top-1 confidence can satisfy the pre-defined threshold. We then impose a consistency regularization between a pair of strongly and weakly augmented samples in the shrunk space to strive for discriminative representations. Furthermore, considering the varied reliability among uncertain samples and the gradually improved model during training, we correspondingly design two reweighting principles for our uncertain loss. Our method exhibits impressive performance on widely adopted benchmarks. Code is available at https://github.com/LiheYoung/ShrinkMatch.

{{</citation>}}


### (5/35) Influence Function Based Second-Order Channel Pruning-Evaluating True Loss Changes For Pruning Is Possible Without Retraining (Hongrong Cheng et al., 2023)

{{<citation>}}

Hongrong Cheng, Miao Zhang, Javen Qinfeng Shi. (2023)  
**Influence Function Based Second-Order Channel Pruning-Evaluating True Loss Changes For Pruning Is Possible Without Retraining**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2308.06755v1)  

---


**ABSTRACT**  
A challenge of channel pruning is designing efficient and effective criteria to select channels to prune. A widely used criterion is minimal performance degeneration. To accurately evaluate the truth performance degeneration requires retraining the survived weights to convergence, which is prohibitively slow. Hence existing pruning methods use previous weights (without retraining) to evaluate the performance degeneration. However, we observe the loss changes differ significantly with and without retraining. It motivates us to develop a technique to evaluate true loss changes without retraining, with which channels to prune can be selected more reliably and confidently. We first derive a closed-form estimator of the true loss change per pruning mask change, using influence functions without retraining. Influence function which is from robust statistics reveals the impacts of a training sample on the model's prediction and is repurposed by us to assess impacts on true loss changes. We then show how to assess the importance of all channels simultaneously and develop a novel global channel pruning algorithm accordingly. We conduct extensive experiments to verify the effectiveness of the proposed algorithm. To the best of our knowledge, we are the first that shows evaluating true loss changes for pruning without retraining is possible. This finding will open up opportunities for a series of new paradigms to emerge that differ from existing pruning methods. The code is available at https://github.com/hrcheng1066/IFSO.

{{</citation>}}


### (6/35) Target before Shooting: Accurate Anomaly Detection and Localization under One Millisecond via Cascade Patch Retrieval (Hanxi Li et al., 2023)

{{<citation>}}

Hanxi Li, Jianfei Hu, Bo Li, Hao Chen, Yongbin Zheng, Chunhua Shen. (2023)  
**Target before Shooting: Accurate Anomaly Detection and Localization under One Millisecond via Cascade Patch Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.06748v1)  

---


**ABSTRACT**  
In this work, by re-examining the "matching" nature of Anomaly Detection (AD), we propose a new AD framework that simultaneously enjoys new records of AD accuracy and dramatically high running speed. In this framework, the anomaly detection problem is solved via a cascade patch retrieval procedure that retrieves the nearest neighbors for each test image patch in a coarse-to-fine fashion. Given a test sample, the top-K most similar training images are first selected based on a robust histogram matching process. Secondly, the nearest neighbor of each test patch is retrieved over the similar geometrical locations on those "global nearest neighbors", by using a carefully trained local metric. Finally, the anomaly score of each test image patch is calculated based on the distance to its "local nearest neighbor" and the "non-background" probability. The proposed method is termed "Cascade Patch Retrieval" (CPR) in this work. Different from the conventional patch-matching-based AD algorithms, CPR selects proper "targets" (reference images and locations) before "shooting" (patch-matching). On the well-acknowledged MVTec AD, BTAD and MVTec-3D AD datasets, the proposed algorithm consistently outperforms all the comparing SOTA methods by remarkable margins, measured by various AD metrics. Furthermore, CPR is extremely efficient. It runs at the speed of 113 FPS with the standard setting while its simplified version only requires less than 1 ms to process an image at the cost of a trivial accuracy drop. The code of CPR is available at https://github.com/flyinghu123/CPR.

{{</citation>}}


### (7/35) Free-ATM: Exploring Unsupervised Learning on Diffusion-Generated Images with Free Attention Masks (David Junhao Zhang et al., 2023)

{{<citation>}}

David Junhao Zhang, Mutian Xu, Chuhui Xue, Wenqing Zhang, Xiaoguang Han, Song Bai, Mike Zheng Shou. (2023)  
**Free-ATM: Exploring Unsupervised Learning on Diffusion-Generated Images with Free Attention Masks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2308.06739v1)  

---


**ABSTRACT**  
Despite the rapid advancement of unsupervised learning in visual representation, it requires training on large-scale datasets that demand costly data collection, and pose additional challenges due to concerns regarding data privacy. Recently, synthetic images generated by text-to-image diffusion models, have shown great potential for benefiting image recognition. Although promising, there has been inadequate exploration dedicated to unsupervised learning on diffusion-generated images. To address this, we start by uncovering that diffusion models' cross-attention layers inherently provide annotation-free attention masks aligned with corresponding text inputs on generated images. We then investigate the problems of three prevalent unsupervised learning techniques ( i.e., contrastive learning, masked modeling, and vision-language pretraining) and introduce customized solutions by fully exploiting the aforementioned free attention masks. Our approach is validated through extensive experiments that show consistent improvements in baseline models across various downstream tasks, including image classification, detection, segmentation, and image-text retrieval. By utilizing our method, it is possible to close the performance gap between unsupervised pretraining on synthetic data and real-world scenarios.

{{</citation>}}


### (8/35) Compositional Feature Augmentation for Unbiased Scene Graph Generation (Lin Li et al., 2023)

{{<citation>}}

Lin Li, Guikun Chen, Jun Xiao, Yi Yang, Chunping Wang, Long Chen. (2023)  
**Compositional Feature Augmentation for Unbiased Scene Graph Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.06712v1)  

---


**ABSTRACT**  
Scene Graph Generation (SGG) aims to detect all the visual relation triplets <sub, pred, obj> in a given image. With the emergence of various advanced techniques for better utilizing both the intrinsic and extrinsic information in each relation triplet, SGG has achieved great progress over the recent years. However, due to the ubiquitous long-tailed predicate distributions, today's SGG models are still easily biased to the head predicates. Currently, the most prevalent debiasing solutions for SGG are re-balancing methods, e.g., changing the distributions of original training samples. In this paper, we argue that all existing re-balancing strategies fail to increase the diversity of the relation triplet features of each predicate, which is critical for robust SGG. To this end, we propose a novel Compositional Feature Augmentation (CFA) strategy, which is the first unbiased SGG work to mitigate the bias issue from the perspective of increasing the diversity of triplet features. Specifically, we first decompose each relation triplet feature into two components: intrinsic feature and extrinsic feature, which correspond to the intrinsic characteristics and extrinsic contexts of a relation triplet, respectively. Then, we design two different feature augmentation modules to enrich the feature diversity of original relation triplets by replacing or mixing up either their intrinsic or extrinsic features from other samples. Due to its model-agnostic nature, CFA can be seamlessly incorporated into various SGG frameworks. Extensive ablations have shown that CFA achieves a new state-of-the-art performance on the trade-off between different metrics.

{{</citation>}}


### (9/35) Isomer: Isomerous Transformer for Zero-shot Video Object Segmentation (Yichen Yuan et al., 2023)

{{<citation>}}

Yichen Yuan, Yifan Wang, Lijun Wang, Xiaoqi Zhao, Huchuan Lu, Yu Wang, Weibo Su, Lei Zhang. (2023)  
**Isomer: Isomerous Transformer for Zero-shot Video Object Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.06693v1)  

---


**ABSTRACT**  
Recent leading zero-shot video object segmentation (ZVOS) works devote to integrating appearance and motion information by elaborately designing feature fusion modules and identically applying them in multiple feature stages. Our preliminary experiments show that with the strong long-range dependency modeling capacity of Transformer, simply concatenating the two modality features and feeding them to vanilla Transformers for feature fusion can distinctly benefit the performance but at a cost of heavy computation. Through further empirical analysis, we find that attention dependencies learned in Transformer in different stages exhibit completely different properties: global query-independent dependency in the low-level stages and semantic-specific dependency in the high-level stages. Motivated by the observations, we propose two Transformer variants: i) Context-Sharing Transformer (CST) that learns the global-shared contextual information within image frames with a lightweight computation. ii) Semantic Gathering-Scattering Transformer (SGST) that models the semantic correlation separately for the foreground and background and reduces the computation cost with a soft token merging mechanism. We apply CST and SGST for low-level and high-level feature fusions, respectively, formulating a level-isomerous Transformer framework for ZVOS task. Compared with the baseline that uses vanilla Transformers for multi-stage fusion, ours significantly increase the speed by 13 times and achieves new state-of-the-art ZVOS performance. Code is available at https://github.com/DLUT-yyc/Isomer.

{{</citation>}}


### (10/35) SimMatchV2: Semi-Supervised Learning with Graph Consistency (Mingkai Zheng et al., 2023)

{{<citation>}}

Mingkai Zheng, Shan You, Lang Huang, Chen Luo, Fei Wang, Chen Qian, Chang Xu. (2023)  
**SimMatchV2: Semi-Supervised Learning with Graph Consistency**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: ImageNet, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.06692v1)  

---


**ABSTRACT**  
Semi-Supervised image classification is one of the most fundamental problem in computer vision, which significantly reduces the need for human labor. In this paper, we introduce a new semi-supervised learning algorithm - SimMatchV2, which formulates various consistency regularizations between labeled and unlabeled data from the graph perspective. In SimMatchV2, we regard the augmented view of a sample as a node, which consists of a label and its corresponding representation. Different nodes are connected with the edges, which are measured by the similarity of the node representations. Inspired by the message passing and node classification in graph theory, we propose four types of consistencies, namely 1) node-node consistency, 2) node-edge consistency, 3) edge-edge consistency, and 4) edge-node consistency. We also uncover that a simple feature normalization can reduce the gaps of the feature norm between different augmented views, significantly improving the performance of SimMatchV2. Our SimMatchV2 has been validated on multiple semi-supervised learning benchmarks. Notably, with ResNet-50 as our backbone and 300 epochs of training, SimMatchV2 achieves 71.9\% and 76.2\% Top-1 Accuracy with 1\% and 10\% labeled examples on ImageNet, which significantly outperforms the previous methods and achieves state-of-the-art performance. Code and pre-trained models are available at \href{https://github.com/mingkai-zheng/SimMatchV2}{https://github.com/mingkai-zheng/SimMatchV2}.

{{</citation>}}


### (11/35) Estimator Meets Equilibrium Perspective: A Rectified Straight Through Estimator for Binary Neural Networks Training (Xiao-Ming Wu et al., 2023)

{{<citation>}}

Xiao-Ming Wu, Dian Zheng, Zuhao Liu, Wei-Shi Zheng. (2023)  
**Estimator Meets Equilibrium Perspective: A Rectified Straight Through Estimator for Binary Neural Networks Training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.06689v1)  

---


**ABSTRACT**  
Binarization of neural networks is a dominant paradigm in neural networks compression. The pioneering work BinaryConnect uses Straight Through Estimator (STE) to mimic the gradients of the sign function, but it also causes the crucial inconsistency problem. Most of the previous methods design different estimators instead of STE to mitigate it. However, they ignore the fact that when reducing the estimating error, the gradient stability will decrease concomitantly. These highly divergent gradients will harm the model training and increase the risk of gradient vanishing and gradient exploding. To fully take the gradient stability into consideration, we present a new perspective to the BNNs training, regarding it as the equilibrium between the estimating error and the gradient stability. In this view, we firstly design two indicators to quantitatively demonstrate the equilibrium phenomenon. In addition, in order to balance the estimating error and the gradient stability well, we revise the original straight through estimator and propose a power function based estimator, Rectified Straight Through Estimator (ReSTE for short). Comparing to other estimators, ReSTE is rational and capable of flexibly balancing the estimating error with the gradient stability. Extensive experiments on CIFAR-10 and ImageNet datasets show that ReSTE has excellent performance and surpasses the state-of-the-art methods without any auxiliary modules or losses.

{{</citation>}}


### (12/35) Unsupervised Adaptation of Polyp Segmentation Models via Coarse-to-Fine Self-Supervision (Jiexiang Wang et al., 2023)

{{<citation>}}

Jiexiang Wang, Chaoqi Chen. (2023)  
**Unsupervised Adaptation of Polyp Segmentation Models via Coarse-to-Fine Self-Supervision**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.06665v1)  

---


**ABSTRACT**  
Unsupervised Domain Adaptation~(UDA) has attracted a surge of interest over the past decade but is difficult to be used in real-world applications. Considering the privacy-preservation issues and security concerns, in this work, we study a practical problem of Source-Free Domain Adaptation (SFDA), which eliminates the reliance on annotated source data. Current SFDA methods focus on extracting domain knowledge from the source-trained model but neglects the intrinsic structure of the target domain. Moreover, they typically utilize pseudo labels for self-training in the target domain, but suffer from the notorious error accumulation problem. To address these issues, we propose a new SFDA framework, called Region-to-Pixel Adaptation Network~(RPANet), which learns the region-level and pixel-level discriminative representations through coarse-to-fine self-supervision. The proposed RPANet consists of two modules, Foreground-aware Contrastive Learning (FCL) and Confidence-Calibrated Pseudo-Labeling (CCPL), which explicitly address the key challenges of ``how to distinguish'' and ``how to refine''. To be specific, FCL introduces a supervised contrastive learning paradigm in the region level to contrast different region centroids across different target images, which efficiently involves all pseudo labels while robust to noisy samples. CCPL designs a novel fusion strategy to reduce the overconfidence problem of pseudo labels by fusing two different target predictions without introducing any additional network modules. Extensive experiments on three cross-domain polyp segmentation tasks reveal that RPANet significantly outperforms state-of-the-art SFDA and UDA methods without access to source data, revealing the potential of SFDA in medical applications.

{{</citation>}}


## cs.LG (10)



### (13/35) Effect of Choosing Loss Function when Using T-batching for Representation Learning on Dynamic Networks (Erfan Loghmani et al., 2023)

{{<citation>}}

Erfan Loghmani, MohammadAmin Fazli. (2023)  
**Effect of Choosing Loss Function when Using T-batching for Representation Learning on Dynamic Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.06862v1)  

---


**ABSTRACT**  
Representation learning methods have revolutionized machine learning on networks by converting discrete network structures into continuous domains. However, dynamic networks that evolve over time pose new challenges. To address this, dynamic representation learning methods have gained attention, offering benefits like reduced learning time and improved accuracy by utilizing temporal information.   T-batching is a valuable technique for training dynamic network models that reduces training time while preserving vital conditions for accurate modeling. However, we have identified a limitation in the training loss function used with t-batching. Through mathematical analysis, we propose two alternative loss functions that overcome these issues, resulting in enhanced training performance.   We extensively evaluate the proposed loss functions on synthetic and real-world dynamic networks. The results consistently demonstrate superior performance compared to the original loss function. Notably, in a real-world network characterized by diverse user interaction histories, the proposed loss functions achieved more than 26.9% enhancement in Mean Reciprocal Rank (MRR) and more than 11.8% improvement in Recall@10. These findings underscore the efficacy of the proposed loss functions in dynamic network modeling.

{{</citation>}}


### (14/35) Generalizing Topological Graph Neural Networks with Paths (Quang Truong et al., 2023)

{{<citation>}}

Quang Truong, Peter Chin. (2023)  
**Generalizing Topological Graph Neural Networks with Paths**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.06838v1)  

---


**ABSTRACT**  
While Graph Neural Networks (GNNs) have made significant strides in diverse areas, they are hindered by a theoretical constraint known as the 1-Weisfeiler-Lehmann test. Even though latest advancements in higher-order GNNs can overcome this boundary, they typically center around certain graph components like cliques or cycles. However, our investigation goes a different route. We put emphasis on paths, which are inherent in every graph. We are able to construct a more general topological perspective and form a bridge to certain established theories about other topological domains. Interestingly, without any assumptions on graph sub-structures, our approach surpasses earlier techniques in this field, achieving state-of-the-art performance on several benchmarks.

{{</citation>}}


### (15/35) SAILOR: Structural Augmentation Based Tail Node Representation Learning (Jie Liao et al., 2023)

{{<citation>}}

Jie Liao, Jintang Li, Liang Chen, Bingzhe Wu, Yatao Bian, Zibin Zheng. (2023)  
**SAILOR: Structural Augmentation Based Tail Node Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Augmentation, GNN, Graph Neural Network, Graph Neural Networks, Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.06801v2)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have achieved state-of-the-art performance in representation learning for graphs recently. However, the effectiveness of GNNs, which capitalize on the key operation of message propagation, highly depends on the quality of the topology structure. Most of the graphs in real-world scenarios follow a long-tailed distribution on their node degrees, that is, a vast majority of the nodes in the graph are tail nodes with only a few connected edges. GNNs produce inferior node representations for tail nodes since they lack structural information. In the pursuit of promoting the expressiveness of GNNs for tail nodes, we explore how the deficiency of structural information deteriorates the performance of tail nodes and propose a general Structural Augmentation based taIL nOde Representation learning framework, dubbed as SAILOR, which can jointly learn to augment the graph structure and extract more informative representations for tail nodes. Extensive experiments on public benchmark datasets demonstrate that SAILOR can significantly improve the tail node representations and outperform the state-of-the-art baselines.

{{</citation>}}


### (16/35) Neural Networks at a Fraction with Pruned Quaternions (Sahel Mohammad Iqbal et al., 2023)

{{<citation>}}

Sahel Mohammad Iqbal, Subhankar Mishra. (2023)  
**Neural Networks at a Fraction with Pruned Quaternions**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2308.06780v1)  

---


**ABSTRACT**  
Contemporary state-of-the-art neural networks have increasingly large numbers of parameters, which prevents their deployment on devices with limited computational power. Pruning is one technique to remove unnecessary weights and reduce resource requirements for training and inference. In addition, for ML tasks where the input data is multi-dimensional, using higher-dimensional data embeddings such as complex numbers or quaternions has been shown to reduce the parameter count while maintaining accuracy. In this work, we conduct pruning on real and quaternion-valued implementations of different architectures on classification tasks. We find that for some architectures, at very high sparsity levels, quaternion models provide higher accuracies than their real counterparts. For example, at the task of image classification on CIFAR-10 using Conv-4, at $3\%$ of the number of parameters as the original model, the pruned quaternion version outperforms the pruned real by more than $10\%$. Experiments on various network architectures and datasets show that for deployment in extremely resource-constrained environments, a sparse quaternion network might be a better candidate than a real sparse model of similar architecture.

{{</citation>}}


### (17/35) A Survey on Deep Neural Network Pruning-Taxonomy, Comparison, Analysis, and Recommendations (Hongrong Cheng et al., 2023)

{{<citation>}}

Hongrong Cheng, Miao Zhang, Javen Qinfeng Shi. (2023)  
**A Survey on Deep Neural Network Pruning-Taxonomy, Comparison, Analysis, and Recommendations**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2308.06767v1)  

---


**ABSTRACT**  
Modern deep neural networks, particularly recent large language models, come with massive model sizes that require significant computational and storage resources. To enable the deployment of modern models on resource-constrained environments and accelerate inference time, researchers have increasingly explored pruning techniques as a popular research direction in neural network compression. However, there is a dearth of up-to-date comprehensive review papers on pruning. To address this issue, in this survey, we provide a comprehensive review of existing research works on deep neural network pruning in a taxonomy of 1) universal/specific speedup, 2) when to prune, 3) how to prune, and 4) fusion of pruning and other compression techniques. We then provide a thorough comparative analysis of seven pairs of contrast settings for pruning (e.g., unstructured/structured) and explore emerging topics, including post-training pruning, different levels of supervision for pruning, and broader applications (e.g., adversarial robustness) to shed light on the commonalities and differences of existing methods and lay the foundation for further method development. To facilitate future research, we build a curated collection of datasets, networks, and evaluations on different applications. Finally, we provide some valuable recommendations on selecting pruning methods and prospect promising research directions. We build a repository at https://github.com/hrcheng1066/awesome-pruning.

{{</citation>}}


### (18/35) Heterogeneous Multi-Agent Reinforcement Learning via Mirror Descent Policy Optimization (Mohammad Mehdi Nasiri et al., 2023)

{{<citation>}}

Mohammad Mehdi Nasiri, Mansoor Rezghi. (2023)  
**Heterogeneous Multi-Agent Reinforcement Learning via Mirror Descent Policy Optimization**  

---
Primary Category: cs.LG  
Categories: 68T, I-2, cs-AI, cs-LG, cs-MA, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.06741v1)  

---


**ABSTRACT**  
This paper presents an extension of the Mirror Descent method to overcome challenges in cooperative Multi-Agent Reinforcement Learning (MARL) settings, where agents have varying abilities and individual policies. The proposed Heterogeneous-Agent Mirror Descent Policy Optimization (HAMDPO) algorithm utilizes the multi-agent advantage decomposition lemma to enable efficient policy updates for each agent while ensuring overall performance improvements. By iteratively updating agent policies through an approximate solution of the trust-region problem, HAMDPO guarantees stability and improves performance. Moreover, the HAMDPO algorithm is capable of handling both continuous and discrete action spaces for heterogeneous agents in various MARL problems. We evaluate HAMDPO on Multi-Agent MuJoCo and StarCraftII tasks, demonstrating its superiority over state-of-the-art algorithms such as HATRPO and HAPPO. These results suggest that HAMDPO is a promising approach for solving cooperative MARL problems and could potentially be extended to address other challenging problems in the field of MARL.

{{</citation>}}


### (19/35) Learning on Graphs with Out-of-Distribution Nodes (Yu Song et al., 2023)

{{<citation>}}

Yu Song, Donglin Wang. (2023)  
**Learning on Graphs with Out-of-Distribution Nodes**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, GNN, Graph Attention Network, Graph Neural Network, Graph Neural Networks, NLP  
[Paper Link](http://arxiv.org/abs/2308.06714v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) are state-of-the-art models for performing prediction tasks on graphs. While existing GNNs have shown great performance on various tasks related to graphs, little attention has been paid to the scenario where out-of-distribution (OOD) nodes exist in the graph during training and inference. Borrowing the concept from CV and NLP, we define OOD nodes as nodes with labels unseen from the training set. Since a lot of networks are automatically constructed by programs, real-world graphs are often noisy and may contain nodes from unknown distributions. In this work, we define the problem of graph learning with out-of-distribution nodes. Specifically, we aim to accomplish two tasks: 1) detect nodes which do not belong to the known distribution and 2) classify the remaining nodes to be one of the known classes. We demonstrate that the connection patterns in graphs are informative for outlier detection, and propose Out-of-Distribution Graph Attention Network (OODGAT), a novel GNN model which explicitly models the interaction between different kinds of nodes and separate inliers from outliers during feature propagation. Extensive experiments show that OODGAT outperforms existing outlier detection methods by a large margin, while being better or comparable in terms of in-distribution classification.

{{</citation>}}


### (20/35) Separable Gaussian Neural Networks: Structure, Analysis, and Function Approximations (Siyuan Xing et al., 2023)

{{<citation>}}

Siyuan Xing, Jianqiao Sun. (2023)  
**Separable Gaussian Neural Networks: Structure, Analysis, and Function Approximations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2308.06679v1)  

---


**ABSTRACT**  
The Gaussian-radial-basis function neural network (GRBFNN) has been a popular choice for interpolation and classification. However, it is computationally intensive when the dimension of the input vector is high. To address this issue, we propose a new feedforward network - Separable Gaussian Neural Network (SGNN) by taking advantage of the separable property of Gaussian functions, which splits input data into multiple columns and sequentially feeds them into parallel layers formed by uni-variate Gaussian functions. This structure reduces the number of neurons from O(N^d) of GRBFNN to O(dN), which exponentially improves the computational speed of SGNN and makes it scale linearly as the input dimension increases. In addition, SGNN can preserve the dominant subspace of the Hessian matrix of GRBFNN in gradient descent training, leading to a similar level of accuracy to GRBFNN. It is experimentally demonstrated that SGNN can achieve 100 times speedup with a similar level of accuracy over GRBFNN on tri-variate function approximations. The SGNN also has better trainability and is more tuning-friendly than DNNs with RuLU and Sigmoid functions. For approximating functions with complex geometry, SGNN can lead to three orders of magnitude more accurate results than a RuLU-DNN with twice the number of layers and the number of neurons per layer.

{{</citation>}}


### (21/35) Foundation Models in Smart Agriculture: Basics, Opportunities, and Challenges (Jiajia Li et al., 2023)

{{<citation>}}

Jiajia Li, Mingle Xu, Lirong Xiang, Dong Chen, Weichao Zhuang, Xunyuan Yin, Zhaojian Li. (2023)  
**Foundation Models in Smart Agriculture: Basics, Opportunities, and Challenges**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.06668v1)  

---


**ABSTRACT**  
The past decade has witnessed the rapid development of ML and DL methodologies in agricultural systems, showcased by great successes in variety of agricultural applications. However, these conventional ML/DL models have certain limitations: They heavily rely on large, costly-to-acquire labeled datasets for training, require specialized expertise for development and maintenance, and are mostly tailored for specific tasks, thus lacking generalizability. Recently, foundation models have demonstrated remarkable successes in language and vision tasks across various domains. These models are trained on a vast amount of data from multiple domains and modalities. Once trained, they can accomplish versatile tasks with just minor fine-tuning and minimal task-specific labeled data. Despite their proven effectiveness and huge potential, there has been little exploration of applying FMs to agriculture fields. Therefore, this study aims to explore the potential of FMs in the field of smart agriculture. In particular, we present conceptual tools and technical background to facilitate the understanding of the problem space and uncover new research directions in this field. To this end, we first review recent FMs in the general computer science domain and categorize them into four categories: language FMs, vision FMs, multimodal FMs, and reinforcement learning FMs. Subsequently, we outline the process of developing agriculture FMs and discuss their potential applications in smart agriculture. We also discuss the unique challenges associated with developing AFMs, including model training, validation, and deployment. Through this study, we contribute to the advancement of AI in agriculture by introducing AFMs as a promising paradigm that can significantly mitigate the reliance on extensive labeled datasets and enhance the efficiency, effectiveness, and generalization of agricultural AI systems.

{{</citation>}}


### (22/35) ALGAN: Time Series Anomaly Detection with Adjusted-LSTM GAN (Md Abul Bashar et al., 2023)

{{<citation>}}

Md Abul Bashar, Richi Nayak. (2023)  
**ALGAN: Time Series Anomaly Detection with Adjusted-LSTM GAN**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, LSTM, Time Series  
[Paper Link](http://arxiv.org/abs/2308.06663v1)  

---


**ABSTRACT**  
Anomaly detection in time series data, to identify points that deviate from normal behaviour, is a common problem in various domains such as manufacturing, medical imaging, and cybersecurity. Recently, Generative Adversarial Networks (GANs) are shown to be effective in detecting anomalies in time series data. The neural network architecture of GANs (i.e. Generator and Discriminator) can significantly improve anomaly detection accuracy. In this paper, we propose a new GAN model, named Adjusted-LSTM GAN (ALGAN), which adjusts the output of an LSTM network for improved anomaly detection in both univariate and multivariate time series data in an unsupervised setting. We evaluate the performance of ALGAN on 46 real-world univariate time series datasets and a large multivariate dataset that spans multiple domains. Our experiments demonstrate that ALGAN outperforms traditional, neural network-based, and other GAN-based methods for anomaly detection in time series data.

{{</citation>}}


## cs.CL (6)



### (23/35) Diagnostic Reasoning Prompts Reveal the Potential for Large Language Model Interpretability in Medicine (Thomas Savage et al., 2023)

{{<citation>}}

Thomas Savage, Ashwin Nayak, Robert Gallo, Ekanath Rangan, Jonathan H Chen. (2023)  
**Diagnostic Reasoning Prompts Reveal the Potential for Large Language Model Interpretability in Medicine**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: GPT, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.06834v1)  

---


**ABSTRACT**  
One of the major barriers to using large language models (LLMs) in medicine is the perception they use uninterpretable methods to make clinical decisions that are inherently different from the cognitive processes of clinicians. In this manuscript we develop novel diagnostic reasoning prompts to study whether LLMs can perform clinical reasoning to accurately form a diagnosis. We find that GPT4 can be prompted to mimic the common clinical reasoning processes of clinicians without sacrificing diagnostic accuracy. This is significant because an LLM that can use clinical reasoning to provide an interpretable rationale offers physicians a means to evaluate whether LLMs can be trusted for patient care. Novel prompting methods have the potential to expose the black box of LLMs, bringing them one step closer to safe and effective use in medicine.

{{</citation>}}


### (24/35) An Ensemble Approach to Question Classification: Integrating Electra Transformer, GloVe, and LSTM (Sanad Aburass et al., 2023)

{{<citation>}}

Sanad Aburass, Osama Dorgham. (2023)  
**An Ensemble Approach to Question Classification: Integrating Electra Transformer, GloVe, and LSTM**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2308.06828v1)  

---


**ABSTRACT**  
This paper introduces a novel ensemble approach for question classification using state-of-the-art models -- Electra, GloVe, and LSTM. The proposed model is trained and evaluated on the TREC dataset, a well-established benchmark for question classification tasks. The ensemble model combines the strengths of Electra, a transformer-based model for language understanding, GloVe, a global vectors for word representation, and LSTM, a recurrent neural network variant, providing a robust and efficient solution for question classification. Extensive experiments were carried out to compare the performance of the proposed ensemble approach with other cutting-edge models, such as BERT, RoBERTa, and DistilBERT. Our results demonstrate that the ensemble model outperforms these models across all evaluation metrics, achieving an accuracy of 0.8 on the test set. These findings underscore the effectiveness of the ensemble approach in enhancing the performance of question classification tasks, and invite further exploration of ensemble methods in natural language processing.

{{</citation>}}


### (25/35) Faithful to Whom? Questioning Interpretability Measures in NLP (Evan Crothers et al., 2023)

{{<citation>}}

Evan Crothers, Herna Viktor, Nathalie Japkowicz. (2023)  
**Faithful to Whom? Questioning Interpretability Measures in NLP**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2308.06795v1)  

---


**ABSTRACT**  
A common approach to quantifying model interpretability is to calculate faithfulness metrics based on iteratively masking input tokens and measuring how much the predicted label changes as a result. However, we show that such metrics are generally not suitable for comparing the interpretability of different neural text classifiers as the response to masked inputs is highly model-specific. We demonstrate that iterative masking can produce large variation in faithfulness scores between comparable models, and show that masked samples are frequently outside the distribution seen during training. We further investigate the impact of adversarial attacks and adversarial training on faithfulness scores, and demonstrate the relevance of faithfulness measures for analyzing feature salience in text adversarial attacks. Our findings provide new insights into the limitations of current faithfulness metrics and key considerations to utilize them appropriately.

{{</citation>}}


### (26/35) Token-Scaled Logit Distillation for Ternary Weight Generative Language Models (Minsoo Kim et al., 2023)

{{<citation>}}

Minsoo Kim, Sihwa Lee, Janghwan Lee, Sukjin Hong, Du-Seong Chang, Wonyong Sung, Jungwook Choi. (2023)  
**Token-Scaled Logit Distillation for Ternary Weight Generative Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLM, Language Model, QA, Quantization  
[Paper Link](http://arxiv.org/abs/2308.06744v1)  

---


**ABSTRACT**  
Generative Language Models (GLMs) have shown impressive performance in tasks such as text generation, understanding, and reasoning. However, the large model size poses challenges for practical deployment. To solve this problem, Quantization-Aware Training (QAT) has become increasingly popular. However, current QAT methods for generative models have resulted in a noticeable loss of accuracy. To counteract this issue, we propose a novel knowledge distillation method specifically designed for GLMs. Our method, called token-scaled logit distillation, prevents overfitting and provides superior learning from the teacher model and ground truth. This research marks the first evaluation of ternary weight quantization-aware training of large-scale GLMs with less than 1.0 degradation in perplexity and no loss of accuracy in a reasoning task.

{{</citation>}}


### (27/35) Transforming Sentiment Analysis in the Financial Domain with ChatGPT (Georgios Fatouros et al., 2023)

{{<citation>}}

Georgios Fatouros, John Soldatos, Kalliopi Kouroumali, Georgios Makridis, Dimosthenis Kyriazis. (2023)  
**Transforming Sentiment Analysis in the Financial Domain with ChatGPT**  

---
Primary Category: cs.CL  
Categories: 68T01, 68T50, 91B28, 91B30, cs-AI, cs-CE, cs-CL, cs-IR, cs.CL  
Keywords: BERT, ChatGPT, Financial, GPT, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2308.07935v1)  

---


**ABSTRACT**  
Financial sentiment analysis plays a crucial role in decoding market trends and guiding strategic trading decisions. Despite the deployment of advanced deep learning techniques and language models to refine sentiment analysis in finance, this study breaks new ground by investigating the potential of large language models, particularly ChatGPT 3.5, in financial sentiment analysis, with a strong emphasis on the foreign exchange market (forex). Employing a zero-shot prompting approach, we examine multiple ChatGPT prompts on a meticulously curated dataset of forex-related news headlines, measuring performance using metrics such as precision, recall, f1-score, and Mean Absolute Error (MAE) of the sentiment class. Additionally, we probe the correlation between predicted sentiment and market returns as an additional evaluation approach. ChatGPT, compared to FinBERT, a well-established sentiment analysis model for financial texts, exhibited approximately 35\% enhanced performance in sentiment classification and a 36\% higher correlation with market returns. By underlining the significance of prompt engineering, particularly in zero-shot contexts, this study spotlights ChatGPT's potential to substantially boost sentiment analysis in financial applications. By sharing the utilized dataset, our intention is to stimulate further research and advancements in the field of financial services.

{{</citation>}}


### (28/35) MACO: A Modality Adversarial and Contrastive Framework for Modality-missing Multi-modal Knowledge Graph Completion (Yichi Zhang et al., 2023)

{{<citation>}}

Yichi Zhang, Zhuo Chen, Wen Zhang. (2023)  
**MACO: A Modality Adversarial and Contrastive Framework for Modality-missing Multi-modal Knowledge Graph Completion**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-MM, cs.CL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.06696v1)  

---


**ABSTRACT**  
Recent years have seen significant advancements in multi-modal knowledge graph completion (MMKGC). MMKGC enhances knowledge graph completion (KGC) by integrating multi-modal entity information, thereby facilitating the discovery of unobserved triples in the large-scale knowledge graphs (KGs). Nevertheless, existing methods emphasize the design of elegant KGC models to facilitate modality interaction, neglecting the real-life problem of missing modalities in KGs. The missing modality information impedes modal interaction, consequently undermining the model's performance. In this paper, we propose a modality adversarial and contrastive framework (MACO) to solve the modality-missing problem in MMKGC. MACO trains a generator and discriminator adversarially to generate missing modality features that can be incorporated into the MMKGC model. Meanwhile, we design a cross-modal contrastive loss to improve the performance of the generator. Experiments on public benchmarks with further explorations demonstrate that MACO could achieve state-of-the-art results and serve as a versatile framework to bolster various MMKGC models. Our code and benchmark data are available at https://github.com/zjukg/MACO.

{{</citation>}}


## cs.IR (1)



### (29/35) InTune: Reinforcement Learning-based Data Pipeline Optimization for Deep Recommendation Models (Kabir Nagrecha et al., 2023)

{{<citation>}}

Kabir Nagrecha, Lingyi Liu, Pablo Delgado, Prasanna Padmanabhan. (2023)  
**InTune: Reinforcement Learning-based Data Pipeline Optimization for Deep Recommendation Models**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-DC, cs-IR, cs-LG, cs-PF, cs.IR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.08500v1)  

---


**ABSTRACT**  
Deep learning-based recommender models (DLRMs) have become an essential component of many modern recommender systems. Several companies are now building large compute clusters reserved only for DLRM training, driving new interest in cost- and time- saving optimizations. The systems challenges faced in this setting are unique; while typical deep learning training jobs are dominated by model execution, the most important factor in DLRM training performance is often online data ingestion.   In this paper, we explore the unique characteristics of this data ingestion problem and provide insights into DLRM training pipeline bottlenecks and challenges. We study real-world DLRM data processing pipelines taken from our compute cluster at Netflix to observe the performance impacts of online ingestion and to identify shortfalls in existing pipeline optimizers. We find that current tooling either yields sub-optimal performance, frequent crashes, or else requires impractical cluster re-organization to adopt. Our studies lead us to design and build a new solution for data pipeline optimization, InTune.   InTune employs a reinforcement learning (RL) agent to learn how to distribute the CPU resources of a trainer machine across a DLRM data pipeline to more effectively parallelize data loading and improve throughput. Our experiments show that InTune can build an optimized data pipeline configuration within only a few minutes, and can easily be integrated into existing training workflows. By exploiting the responsiveness and adaptability of RL, InTune achieves higher online data ingestion rates than existing optimizers, thus reducing idle times in model execution and increasing efficiency. We apply InTune to our real-world cluster, and find that it increases data ingestion throughput by as much as 2.29X versus state-of-the-art data pipeline optimizers while also improving both CPU & GPU utilization.

{{</citation>}}


## cs.CR (2)



### (30/35) SoK: Realistic Adversarial Attacks and Defenses for Intelligent Network Intrusion Detection (João Vitorino et al., 2023)

{{<citation>}}

João Vitorino, Isabel Praça, Eva Maia. (2023)  
**SoK: Realistic Adversarial Attacks and Defenses for Intelligent Network Intrusion Detection**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs-NI, cs.CR  
Keywords: Adversarial Attack, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2308.06819v1)  

---


**ABSTRACT**  
Machine Learning (ML) can be incredibly valuable to automate anomaly detection and cyber-attack classification, improving the way that Network Intrusion Detection (NID) is performed. However, despite the benefits of ML models, they are highly susceptible to adversarial cyber-attack examples specifically crafted to exploit them. A wide range of adversarial attacks have been created and researchers have worked on various defense strategies to safeguard ML models, but most were not intended for the specific constraints of a communication network and its communication protocols, so they may lead to unrealistic examples in the NID domain. This Systematization of Knowledge (SoK) consolidates and summarizes the state-of-the-art adversarial learning approaches that can generate realistic examples and could be used in real ML development and deployment scenarios with real network traffic flows. This SoK also describes the open challenges regarding the use of adversarial ML in the NID domain, defines the fundamental properties that are required for an adversarial example to be realistic, and provides guidelines for researchers to ensure that their future experiments are adequate for a real communication network.

{{</citation>}}


### (31/35) Impact of Orientation on the Bias of SRAM-Based PUFs (Zain Ul Abideen et al., 2023)

{{<citation>}}

Zain Ul Abideen, Rui Wang, Tiago Diadami Perez, Geert-Jan Schrijen, Samuel Pagliarini. (2023)  
**Impact of Orientation on the Bias of SRAM-Based PUFs**  

---
Primary Category: cs.CR  
Categories: cs-AR, cs-CR, cs.CR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.06730v1)  

---


**ABSTRACT**  
This paper investigates the impact of memory orientation on the bias pattern of SRAM-based PUFs. We designed and fabricated a 65nm CMOS chip that contains eleven SRAM macros that exercise different memory- and chip-level parameters. At the memory level, several parameters passed to the SRAM compiler are considered, including the number of addresses, the number of words, the aspect ratio, and the chosen bitcell. Chip-level decisions are considered during the floorplan, including the location and rotation of each SRAM macro in the testchip. In this study, we conduct a comprehensive analysis of different memory orientations and their effect on the biasing direction. Physical measurements performed on 50 fabricated chips revealed that specific memory orientations, namely R270 and MY90, exhibit a distinct negative biasing direction compared to other orientations. Importantly, this biasing direction remains consistent regardless of memory type, column mux ratio, memory size, or the utilization of SRAMs with different bitcells. Overall, this study highlights the significance of careful physical implementation and memory orientation selection in designing SRAM-based PUFs. Our findings can guide designers in the selection of SRAM memories with properties that make for better PUFs that potentially require less error correction effort to compensate for instability.

{{</citation>}}


## cs.RO (2)



### (32/35) Ground Manipulator Primitive Tasks to Executable Actions using Large Language Models (Yue Cao et al., 2023)

{{<citation>}}

Yue Cao, C. S. George Lee. (2023)  
**Ground Manipulator Primitive Tasks to Executable Actions using Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.06810v1)  

---


**ABSTRACT**  
Layered architectures have been widely used in robot systems. The majority of them implement planning and execution functions in separate layers. However, there still lacks a straightforward way to transit high-level tasks in the planning layer to the low-level motor commands in the execution layer. In order to tackle this challenge, we propose a novel approach to ground the manipulator primitive tasks to robot low-level actions using large language models (LLMs). We designed a program-like prompt based on the task frame formalism. In this way, we enable LLMs to generate position/force set-points for hybrid control. Evaluations over several state-of-the-art LLMs are provided.

{{</citation>}}


### (33/35) 3D Scene Graph Prediction on Point Clouds Using Knowledge Graphs (Yiding Qiu et al., 2023)

{{<citation>}}

Yiding Qiu, Henrik I. Christensen. (2023)  
**3D Scene Graph Prediction on Point Clouds Using Knowledge Graphs**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2308.06719v1)  

---


**ABSTRACT**  
3D scene graph prediction is a task that aims to concurrently predict object classes and their relationships within a 3D environment. As these environments are primarily designed by and for humans, incorporating commonsense knowledge regarding objects and their relationships can significantly constrain and enhance the prediction of the scene graph. In this paper, we investigate the application of commonsense knowledge graphs for 3D scene graph prediction on point clouds of indoor scenes. Through experiments conducted on a real-world indoor dataset, we demonstrate that integrating external commonsense knowledge via the message-passing method leads to a 15.0 % improvement in scene graph prediction accuracy with external knowledge and $7.96\%$ with internal knowledge when compared to state-of-the-art algorithms. We also tested in the real world with 10 frames per second for scene graph generation to show the usage of the model in a more realistic robotics setting.

{{</citation>}}


## cs.DC (1)



### (34/35) A Dynamic Distributed Scheduler for Computing on the Edge (Fei Hu et al., 2023)

{{<citation>}}

Fei Hu, Kunal Mehta, Shivakant Mishra, Mohammad AlMutawa. (2023)  
**A Dynamic Distributed Scheduler for Computing on the Edge**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.06806v1)  

---


**ABSTRACT**  
Edge computing has become a promising computing paradigm for building IoT (Internet of Things) applications, particularly for applications with specific constraints such as latency or privacy requirements. Due to resource constraints at the edge, it is important to efficiently utilize all available computing resources to satisfy these constraints. A key challenge in utilizing these computing resources is the scheduling of different computing tasks in a dynamically varying, highly hybrid computing environment. This paper described the design, implementation, and evaluation of a distributed scheduler for the edge that constantly monitors the current state of the computing infrastructure and dynamically schedules various computing tasks to ensure that all application constraints are met. This scheduler has been extensively evaluated with real-world AI applications under different scenarios and demonstrates that it outperforms current scheduling approaches in satisfying various application constraints.

{{</citation>}}


## cs.SE (1)



### (35/35) PentestGPT: An LLM-empowered Automatic Penetration Testing Tool (Gelei Deng et al., 2023)

{{<citation>}}

Gelei Deng, Yi Liu, Víctor Mayoral-Vilches, Peng Liu, Yuekang Li, Yuan Xu, Tianwei Zhang, Yang Liu, Martin Pinzger, Stefan Rass. (2023)  
**PentestGPT: An LLM-empowered Automatic Penetration Testing Tool**  

---
Primary Category: cs.SE  
Categories: cs-CR, cs-SE, cs.SE  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.06782v1)  

---


**ABSTRACT**  
Penetration testing, a crucial industrial practice for ensuring system security, has traditionally resisted automation due to the extensive expertise required by human professionals. Large Language Models (LLMs) have shown significant advancements in various domains, and their emergent abilities suggest their potential to revolutionize industries. In this research, we evaluate the performance of LLMs on real-world penetration testing tasks using a robust benchmark created from test machines with platforms. Our findings reveal that while LLMs demonstrate proficiency in specific sub-tasks within the penetration testing process, such as using testing tools, interpreting outputs, and proposing subsequent actions, they also encounter difficulties maintaining an integrated understanding of the overall testing scenario.   In response to these insights, we introduce PentestGPT, an LLM-empowered automatic penetration testing tool that leverages the abundant domain knowledge inherent in LLMs. PentestGPT is meticulously designed with three self-interacting modules, each addressing individual sub-tasks of penetration testing, to mitigate the challenges related to context loss. Our evaluation shows that PentestGPT not only outperforms LLMs with a task-completion increase of 228.6\% compared to the \gptthree model among the benchmark targets but also proves effective in tackling real-world penetration testing challenges. Having been open-sourced on GitHub, PentestGPT has garnered over 4,700 stars and fostered active community engagement, attesting to its value and impact in both the academic and industrial spheres.

{{</citation>}}
