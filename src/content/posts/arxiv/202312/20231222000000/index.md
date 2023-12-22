---
draft: false
title: "arXiv @ 2023.12.22"
date: 2023-12-22
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.22"
    identifier: arxiv_20231222
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.DS (1)](#csds-1)
- [cs.AI (4)](#csai-4)
- [math.OC (1)](#mathoc-1)
- [cs.CV (31)](#cscv-31)
- [cs.DB (1)](#csdb-1)
- [cs.DL (1)](#csdl-1)
- [cs.CL (23)](#cscl-23)
- [cs.RO (5)](#csro-5)
- [cs.LG (21)](#cslg-21)
- [eess.IV (5)](#eessiv-5)
- [cs.GT (1)](#csgt-1)
- [cs.SE (5)](#csse-5)
- [cs.CE (1)](#csce-1)
- [physics.chem-ph (1)](#physicschem-ph-1)
- [cs.CY (3)](#cscy-3)
- [cs.CR (4)](#cscr-4)
- [cs.PL (2)](#cspl-2)
- [cs.HC (3)](#cshc-3)
- [eess.AS (4)](#eessas-4)
- [cs.AR (1)](#csar-1)
- [physics.soc-ph (1)](#physicssoc-ph-1)
- [quant-ph (1)](#quant-ph-1)
- [cs.IR (2)](#csir-2)
- [cs.IT (1)](#csit-1)
- [cs.DC (1)](#csdc-1)

## cs.DS (1)



### (1/124) Dimension-Accuracy Tradeoffs in Contrastive Embeddings for Triplets, Terminals & Top-k Nearest Neighbors (Vaggos Chatziafratis et al., 2023)

{{<citation>}}

Vaggos Chatziafratis, Piotr Indyk. (2023)  
**Dimension-Accuracy Tradeoffs in Contrastive Embeddings for Triplets, Terminals & Top-k Nearest Neighbors**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs.DS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.13490v1)  

---


**ABSTRACT**  
Metric embeddings traditionally study how to map $n$ items to a target metric space such that distance lengths are not heavily distorted; but what if we only care to preserve the relative order of the distances (and not their length)? In this paper, we are motivated by the following basic question: given triplet comparisons of the form ``item $i$ is closer to item $j$ than to item $k$,'' can we find low-dimensional Euclidean representations for the $n$ items that respect those distance comparisons? Such order-preserving embeddings naturally arise in important applications and have been studied since the 1950s, under the name of ordinal or non-metric embeddings. Our main results are:   1. Nearly-Tight Bounds on Triplet Dimension: We introduce the natural concept of triplet dimension of a dataset, and surprisingly, we show that in order for an ordinal embedding to be triplet-preserving, its dimension needs to grow as $\frac n2$ in the worst case. This is optimal (up to constant) as $n-1$ dimensions always suffice.   2. Tradeoffs for Dimension vs (Ordinal) Relaxation: We then relax the requirement that every triplet should be exactly preserved and present almost tight lower bounds for the maximum ratio between distances whose relative order was inverted by the embedding; this ratio is known as (ordinal) relaxation in the literature and serves as a counterpart to (metric) distortion.   3. New Bounds on Terminal and Top-$k$-NNs Embeddings: Going beyond triplets, we then study two well-motivated scenarios where we care about preserving specific sets of distances (not necessarily triplets). The first scenario is Terminal Ordinal Embeddings and the second scenario is top-$k$-NNs Ordinal Embeddings.   To the best of our knowledge, these are some of the first tradeoffs on triplet-preserving ordinal embeddings and the first study of Terminal and Top-$k$-NNs Ordinal Embeddings.

{{</citation>}}


## cs.AI (4)



### (2/124) Understanding and Estimating Domain Complexity Across Domains (Katarina Doctor et al., 2023)

{{<citation>}}

Katarina Doctor, Mayank Kejriwal, Lawrence Holder, Eric Kildebeck, Emma Resmini, Christopher Pereyda, Robert J. Steininger, Daniel V. Olivença. (2023)  
**Understanding and Estimating Domain Complexity Across Domains**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.13487v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) systems, trained in controlled environments, often struggle in real-world complexities. We propose a general framework for estimating domain complexity across diverse environments, like open-world learning and real-world applications. This framework distinguishes between intrinsic complexity (inherent to the domain) and extrinsic complexity (dependent on the AI agent). By analyzing dimensionality, sparsity, and diversity within these categories, we offer a comprehensive view of domain challenges. This approach enables quantitative predictions of AI difficulty during environment transitions, avoids bias in novel situations, and helps navigate the vast search spaces of open-world domains.

{{</citation>}}


### (3/124) Adversarial Markov Games: On Adaptive Decision-Based Attacks and Defenses (Ilias Tsingenopoulos et al., 2023)

{{<citation>}}

Ilias Tsingenopoulos, Vera Rimmer, Davy Preuveneers, Fabio Pierazzi, Lorenzo Cavallaro, Wouter Joosen. (2023)  
**Adversarial Markov Games: On Adaptive Decision-Based Attacks and Defenses**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CR, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.13435v1)  

---


**ABSTRACT**  
Despite considerable efforts on making them robust, real-world ML-based systems remain vulnerable to decision based attacks, as definitive proofs of their operational robustness have so far proven intractable. The canonical approach in robustness evaluation calls for adaptive attacks, that is with complete knowledge of the defense and tailored to bypass it. In this study, we introduce a more expansive notion of being adaptive and show how attacks but also defenses can benefit by it and by learning from each other through interaction. We propose and evaluate a framework for adaptively optimizing black-box attacks and defenses against each other through the competitive game they form. To reliably measure robustness, it is important to evaluate against realistic and worst-case attacks. We thus augment both attacks and the evasive arsenal at their disposal through adaptive control, and observe that the same can be done for defenses, before we evaluate them first apart and then jointly under a multi-agent perspective. We demonstrate that active defenses, which control how the system responds, are a necessary complement to model hardening when facing decision-based attacks; then how these defenses can be circumvented by adaptive attacks, only to finally elicit active and adaptive defenses. We validate our observations through a wide theoretical and empirical investigation to confirm that AI-enabled adversaries pose a considerable threat to black-box ML-based systems, rekindling the proverbial arms race where defenses have to be AI-enabled too. Succinctly, we address the challenges posed by adaptive adversaries and develop adaptive defenses, thereby laying out effective strategies in ensuring the robustness of ML-based systems deployed in the real-world.

{{</citation>}}


### (4/124) Concept-based Explainable Artificial Intelligence: A Survey (Eleonora Poeta et al., 2023)

{{<citation>}}

Eleonora Poeta, Gabriele Ciravegna, Eliana Pastor, Tania Cerquitelli, Elena Baralis. (2023)  
**Concept-based Explainable Artificial Intelligence: A Survey**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12936v1)  

---


**ABSTRACT**  
The field of explainable artificial intelligence emerged in response to the growing need for more transparent and reliable models. However, using raw features to provide explanations has been disputed in several works lately, advocating for more user-understandable explanations. To address this issue, a wide range of papers proposing Concept-based eXplainable Artificial Intelligence (C-XAI) methods have arisen in recent years. Nevertheless, a unified categorization and precise field definition are still missing. This paper fills the gap by offering a thorough review of C-XAI approaches. We define and identify different concepts and explanation types. We provide a taxonomy identifying nine categories and propose guidelines for selecting a suitable category based on the development context. Additionally, we report common evaluation strategies including metrics, human evaluations and dataset employed, aiming to assist the development of future methods. We believe this survey will serve researchers, practitioners, and domain experts in comprehending and advancing this innovative field.

{{</citation>}}


### (5/124) Towards Machines that Trust: AI Agents Learn to Trust in the Trust Game (Ardavan S. Nobandegani et al., 2023)

{{<citation>}}

Ardavan S. Nobandegani, Irina Rish, Thomas R. Shultz. (2023)  
**Towards Machines that Trust: AI Agents Learn to Trust in the Trust Game**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, q-bio-NC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12868v1)  

---


**ABSTRACT**  
Widely considered a cornerstone of human morality, trust shapes many aspects of human social interactions. In this work, we present a theoretical analysis of the $\textit{trust game}$, the canonical task for studying trust in behavioral and brain sciences, along with simulation results supporting our analysis. Specifically, leveraging reinforcement learning (RL) to train our AI agents, we systematically investigate learning trust under various parameterizations of this task. Our theoretical analysis, corroborated by the simulations results presented, provides a mathematical basis for the emergence of trust in the trust game.

{{</citation>}}


## math.OC (1)



### (6/124) Task Planning for Multiple Item Insertion using ADMM (Gavin Zheng, 2023)

{{<citation>}}

Gavin Zheng. (2023)  
**Task Planning for Multiple Item Insertion using ADMM**  

---
Primary Category: math.OC  
Categories: cs-RO, math-OC, math.OC  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.13472v1)  

---


**ABSTRACT**  
Mixed-integer nonlinear programmings (MINLPs) are powerful formulation tools for task planning. However, it suffers from long solving time especially for large scale problems. In this work, we first formulate the task planning problem for item stowing into a mixed-integer nonlinear programming problem, then solve it using Alternative Direction Method of Multipliers (ADMM). ADMM separates the complete formulation into a nonlinear programming problem and mixed-integer programming problem, then iterate between them to solve the original problem. We show that our ADMM converges better than non-warm-started nonlinear complementary formulation. Our proposed methods are demonstrated on hardware as a high level planner to insert books into the bookshelf.

{{</citation>}}


## cs.CV (31)



### (7/124) MGAug: Multimodal Geometric Augmentation in Latent Spaces of Image Deformations (Tonmoy Hossain et al., 2023)

{{<citation>}}

Tonmoy Hossain, Jian Wang, Miaomiao Zhang. (2023)  
**MGAug: Multimodal Geometric Augmentation in Latent Spaces of Image Deformations**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.13440v1)  

---


**ABSTRACT**  
Geometric transformations have been widely used to augment the size of training images. Existing methods often assume a unimodal distribution of the underlying transformations between images, which limits their power when data with multimodal distributions occur. In this paper, we propose a novel model, Multimodal Geometric Augmentation (MGAug), that for the first time generates augmenting transformations in a multimodal latent space of geometric deformations. To achieve this, we first develop a deep network that embeds the learning of latent geometric spaces of diffeomorphic transformations (a.k.a. diffeomorphisms) in a variational autoencoder (VAE). A mixture of multivariate Gaussians is formulated in the tangent space of diffeomorphisms and serves as a prior to approximate the hidden distribution of image transformations. We then augment the original training dataset by deforming images using randomly sampled transformations from the learned multimodal latent space of VAE. To validate the efficiency of our model, we jointly learn the augmentation strategy with two distinct domain-specific tasks: multi-class classification on 2D synthetic datasets and segmentation on real 3D brain magnetic resonance images (MRIs). We also compare MGAug with state-of-the-art transformation-based image augmentation algorithms. Experimental results show that our proposed approach outperforms all baselines by significantly improved prediction accuracy. Our code is publicly available at https://github.com/tonmoy-hossain/MGAug.

{{</citation>}}


### (8/124) EPNet: An Efficient Pyramid Network for Enhanced Single-Image Super-Resolution with Reduced Computational Requirements (Xin Xu et al., 2023)

{{<citation>}}

Xin Xu, Jinman Park, Paul Fieguth. (2023)  
**EPNet: An Efficient Pyramid Network for Enhanced Single-Image Super-Resolution with Reduced Computational Requirements**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.13396v1)  

---


**ABSTRACT**  
Single-image super-resolution (SISR) has seen significant advancements through the integration of deep learning. However, the substantial computational and memory requirements of existing methods often limit their practical application. This paper introduces a new Efficient Pyramid Network (EPNet) that harmoniously merges an Edge Split Pyramid Module (ESPM) with a Panoramic Feature Extraction Module (PFEM) to overcome the limitations of existing methods, particularly in terms of computational efficiency. The ESPM applies a pyramid-based channel separation strategy, boosting feature extraction while maintaining computational efficiency. The PFEM, a novel fusion of CNN and Transformer structures, enables the concurrent extraction of local and global features, thereby providing a panoramic view of the image landscape. Our architecture integrates the PFEM in a manner that facilitates the streamlined exchange of feature information and allows for the further refinement of image texture details. Experimental results indicate that our model outperforms existing state-of-the-art methods in image resolution quality, while considerably decreasing computational and memory costs. This research contributes to the ongoing evolution of efficient and practical SISR methodologies, bearing broader implications for the field of computer vision.

{{</citation>}}


### (9/124) Zero-Shot Metric Depth with a Field-of-View Conditioned Diffusion Model (Saurabh Saxena et al., 2023)

{{<citation>}}

Saurabh Saxena, Junhwa Hur, Charles Herrmann, Deqing Sun, David J. Fleet. (2023)  
**Zero-Shot Metric Depth with a Field-of-View Conditioned Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.13252v1)  

---


**ABSTRACT**  
While methods for monocular depth estimation have made significant strides on standard benchmarks, zero-shot metric depth estimation remains unsolved. Challenges include the joint modeling of indoor and outdoor scenes, which often exhibit significantly different distributions of RGB and depth, and the depth-scale ambiguity due to unknown camera intrinsics. Recent work has proposed specialized multi-head architectures for jointly modeling indoor and outdoor scenes. In contrast, we advocate a generic, task-agnostic diffusion model, with several advancements such as log-scale depth parameterization to enable joint modeling of indoor and outdoor scenes, conditioning on the field-of-view (FOV) to handle scale ambiguity and synthetically augmenting FOV during training to generalize beyond the limited camera intrinsics in training datasets. Furthermore, by employing a more diverse training mixture than is common, and an efficient diffusion parameterization, our method, DMD (Diffusion for Metric Depth) achieves a 25\% reduction in relative error (REL) on zero-shot indoor and 33\% reduction on zero-shot outdoor datasets over the current SOTA using only a small number of denoising steps. For an overview see https://diffusion-vision.github.io/dmd

{{</citation>}}


### (10/124) StableKD: Breaking Inter-block Optimization Entanglement for Stable Knowledge Distillation (Shiu-hong Kao et al., 2023)

{{<citation>}}

Shiu-hong Kao, Jierun Chen, S. H. Gary Chan. (2023)  
**StableKD: Breaking Inter-block Optimization Entanglement for Stable Knowledge Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2312.13223v1)  

---


**ABSTRACT**  
Knowledge distillation (KD) has been recognized as an effective tool to compress and accelerate models. However, current KD approaches generally suffer from an accuracy drop and/or an excruciatingly long distillation process. In this paper, we tackle the issue by first providing a new insight into a phenomenon that we call the Inter-Block Optimization Entanglement (IBOE), which makes the conventional end-to-end KD approaches unstable with noisy gradients. We then propose StableKD, a novel KD framework that breaks the IBOE and achieves more stable optimization. StableKD distinguishes itself through two operations: Decomposition and Recomposition, where the former divides a pair of teacher and student networks into several blocks for separate distillation, and the latter progressively merges them back, evolving towards end-to-end distillation. We conduct extensive experiments on CIFAR100, Imagewoof, and ImageNet datasets with various teacher-student pairs. Compared to other KD approaches, our simple yet effective StableKD greatly boosts the model accuracy by 1% ~ 18%, speeds up the convergence up to 10 times, and outperforms them with only 40% of the training data.

{{</citation>}}


### (11/124) ASSISTGUI: Task-Oriented Desktop Graphical User Interface Automation (Difei Gao et al., 2023)

{{<citation>}}

Difei Gao, Lei Ji, Zechen Bai, Mingyu Ouyang, Peiran Li, Dongxing Mao, Qinchen Wu, Weichen Zhang, Peiyi Wang, Xiangwu Guo, Hengxu Wang, Luowei Zhou, Mike Zheng Shou. (2023)  
**ASSISTGUI: Task-Oriented Desktop Graphical User Interface Automation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.13108v1)  

---


**ABSTRACT**  
Graphical User Interface (GUI) automation holds significant promise for assisting users with complex tasks, thereby boosting human productivity. Existing works leveraging Large Language Model (LLM) or LLM-based AI agents have shown capabilities in automating tasks on Android and Web platforms. However, these tasks are primarily aimed at simple device usage and entertainment operations. This paper presents a novel benchmark, AssistGUI, to evaluate whether models are capable of manipulating the mouse and keyboard on the Windows platform in response to user-requested tasks. We carefully collected a set of 100 tasks from nine widely-used software applications, such as, After Effects and MS Word, each accompanied by the necessary project files for better evaluation. Moreover, we propose an advanced Actor-Critic Embodied Agent framework, which incorporates a sophisticated GUI parser driven by an LLM-agent and an enhanced reasoning mechanism adept at handling lengthy procedural tasks. Our experimental results reveal that our GUI Parser and Reasoning mechanism outshine existing methods in performance. Nevertheless, the potential remains substantial, with the best model attaining only a 46% success rate on our benchmark. We conclude with a thorough analysis of the current methods' limitations, setting the stage for future breakthroughs in this domain.

{{</citation>}}


### (12/124) Optimizing Ego Vehicle Trajectory Prediction: The Graph Enhancement Approach (Sushil Sharma et al., 2023)

{{<citation>}}

Sushil Sharma, Aryan Singh, Ganesh Sistu, Mark Halton, Ciarán Eising. (2023)  
**Optimizing Ego Vehicle Trajectory Prediction: The Graph Enhancement Approach**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.13104v1)  

---


**ABSTRACT**  
Predicting the trajectory of an ego vehicle is a critical component of autonomous driving systems. Current state-of-the-art methods typically rely on Deep Neural Networks (DNNs) and sequential models to process front-view images for future trajectory prediction. However, these approaches often struggle with perspective issues affecting object features in the scene. To address this, we advocate for the use of Bird's Eye View (BEV) perspectives, which offer unique advantages in capturing spatial relationships and object homogeneity. In our work, we leverage Graph Neural Networks (GNNs) and positional encoding to represent objects in a BEV, achieving competitive performance compared to traditional DNN-based methods. While the BEV-based approach loses some detailed information inherent to front-view images, we balance this by enriching the BEV data by representing it as a graph where relationships between the objects in a scene are captured effectively.

{{</citation>}}


### (13/124) SEER-ZSL: Semantic Encoder-Enhanced Representations for Generalized Zero-Shot Learning (William Heyden et al., 2023)

{{<citation>}}

William Heyden, Habib Ullah, M. Salman Siddiqui, Fadi Al Machot. (2023)  
**SEER-ZSL: Semantic Encoder-Enhanced Representations for Generalized Zero-Shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.13100v1)  

---


**ABSTRACT**  
Generalized Zero-Shot Learning (GZSL) recognizes unseen classes by transferring knowledge from the seen classes, depending on the inherent interactions between visual and semantic data. However, the discrepancy between well-prepared training data and unpredictable real-world test scenarios remains a significant challenge. This paper introduces a dual strategy to address the generalization gap. Firstly, we incorporate semantic information through an innovative encoder. This encoder effectively integrates class-specific semantic information by targeting the performance disparity, enhancing the produced features to enrich the semantic space for class-specific attributes. Secondly, we refine our generative capabilities using a novel compositional loss function. This approach generates discriminative classes, effectively classifying both seen and unseen classes. In addition, we extend the exploitation of the learned latent space by utilizing controlled semantic inputs, ensuring the robustness of the model in varying environments. This approach yields a model that outperforms the state-of-the-art models in terms of both generalization and diverse settings, notably without requiring hyperparameter tuning or domain-specific adaptations. We also propose a set of novel evaluation metrics to provide a more detailed assessment of the reliability and reproducibility of the results. The complete code is made available on https://github.com/william-heyden/SEER-ZeroShotLearning/.

{{</citation>}}


### (14/124) MoSAR: Monocular Semi-Supervised Model for Avatar Reconstruction using Differentiable Shading (Abdallah Dib et al., 2023)

{{<citation>}}

Abdallah Dib, Luiz Gustavo Hafemann, Emeline Got, Trevor Anderson, Amin Fadaeinejad, Rafael M. O. Cruz, Marc-Andre Carbonneau. (2023)  
**MoSAR: Monocular Semi-Supervised Model for Avatar Reconstruction using Differentiable Shading**  

---
Primary Category: cs.CV  
Categories: 68T45 (Primary) 68T07, 68T01 (Secondary), I-2-10; I-4; I-3-3; I-5, cs-CV, cs-GR, cs-LG, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.13091v1)  

---


**ABSTRACT**  
Reconstructing an avatar from a portrait image has many applications in multimedia, but remains a challenging research problem. Extracting reflectance maps and geometry from one image is ill-posed: recovering geometry is a one-to-many mapping problem and reflectance and light are difficult to disentangle. Accurate geometry and reflectance can be captured under the controlled conditions of a light stage, but it is costly to acquire large datasets in this fashion. Moreover, training solely with this type of data leads to poor generalization with in-the-wild images. This motivates the introduction of MoSAR, a method for 3D avatar generation from monocular images. We propose a semi-supervised training scheme that improves generalization by learning from both light stage and in-the-wild datasets. This is achieved using a novel differentiable shading formulation. We show that our approach effectively disentangles the intrinsic face parameters, producing relightable avatars. As a result, MoSAR estimates a richer set of skin reflectance maps, and generates more realistic avatars than existing state-of-the-art methods. We also introduce a new dataset, named FFHQ-UV-Intrinsics, the first public dataset providing intrisic face attributes at scale (diffuse, specular, ambient occlusion and translucency maps) for a total of 10k subjects. The project website and the dataset are available on the following link: https://ubisoftlaforge.github.io/character/mosar

{{</citation>}}


### (15/124) Perception Test 2023: A Summary of the First Challenge And Outcome (Joseph Heyward et al., 2023)

{{<citation>}}

Joseph Heyward, João Carreira, Dima Damen, Andrew Zisserman, Viorica Pătrăucean. (2023)  
**Perception Test 2023: A Summary of the First Challenge And Outcome**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.13090v1)  

---


**ABSTRACT**  
The First Perception Test challenge was held as a half-day workshop alongside the IEEE/CVF International Conference on Computer Vision (ICCV) 2023, with the goal of benchmarking state-of-the-art video models on the recently proposed Perception Test benchmark. The challenge had six tracks covering low-level and high-level tasks, with both a language and non-language interface, across video, audio, and text modalities, and covering: object tracking, point tracking, temporal action localisation, temporal sound localisation, multiple-choice video question-answering, and grounded video question-answering. We summarise in this report the task descriptions, metrics, baselines, and results.

{{</citation>}}


### (16/124) Point Deformable Network with Enhanced Normal Embedding for Point Cloud Analysis (Xingyilang Yin et al., 2023)

{{<citation>}}

Xingyilang Yin, Xi Yang, Liangchen Liu, Nannan Wang, Xinbo Gao. (2023)  
**Point Deformable Network with Enhanced Normal Embedding for Point Cloud Analysis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.13071v1)  

---


**ABSTRACT**  
Recently MLP-based methods have shown strong performance in point cloud analysis. Simple MLP architectures are able to learn geometric features in local point groups yet fail to model long-range dependencies directly. In this paper, we propose Point Deformable Network (PDNet), a concise MLP-based network that can capture long-range relations with strong representation ability. Specifically, we put forward Point Deformable Aggregation Module (PDAM) to improve representation capability in both long-range dependency and adaptive aggregation among points. For each query point, PDAM aggregates information from deformable reference points rather than points in limited local areas. The deformable reference points are generated data-dependent, and we initialize them according to the input point positions. Additional offsets and modulation scalars are learned on the whole point features, which shift the deformable reference points to the regions of interest. We also suggest estimating the normal vector for point clouds and applying Enhanced Normal Embedding (ENE) to the geometric extractors to improve the representation ability of single-point. Extensive experiments and ablation studies on various benchmarks demonstrate the effectiveness and superiority of our PDNet.

{{</citation>}}


### (17/124) PPEA-Depth: Progressive Parameter-Efficient Adaptation for Self-Supervised Monocular Depth Estimation (Yue-Jiang Dong et al., 2023)

{{<citation>}}

Yue-Jiang Dong, Yuan-Chen Guo, Ying-Tian Liu, Fang-Lue Zhang, Song-Hai Zhang. (2023)  
**PPEA-Depth: Progressive Parameter-Efficient Adaptation for Self-Supervised Monocular Depth Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.13066v1)  

---


**ABSTRACT**  
Self-supervised monocular depth estimation is of significant importance with applications spanning across autonomous driving and robotics. However, the reliance on self-supervision introduces a strong static-scene assumption, thereby posing challenges in achieving optimal performance in dynamic scenes, which are prevalent in most real-world situations. To address these issues, we propose PPEA-Depth, a Progressive Parameter-Efficient Adaptation approach to transfer a pre-trained image model for self-supervised depth estimation. The training comprises two sequential stages: an initial phase trained on a dataset primarily composed of static scenes, succeeded by an expansion to more intricate datasets involving dynamic scenes. To facilitate this process, we design compact encoder and decoder adapters to enable parameter-efficient tuning, allowing the network to adapt effectively. They not only uphold generalized patterns from pre-trained image models but also retain knowledge gained from the preceding phase into the subsequent one. Extensive experiments demonstrate that PPEA-Depth achieves state-of-the-art performance on KITTI, CityScapes and DDAD datasets.

{{</citation>}}


### (18/124) Quantifying Bias in Text-to-Image Generative Models (Jordan Vice et al., 2023)

{{<citation>}}

Jordan Vice, Naveed Akhtar, Richard Hartley, Ajmal Mian. (2023)  
**Quantifying Bias in Text-to-Image Generative Models**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.13053v1)  

---


**ABSTRACT**  
Bias in text-to-image (T2I) models can propagate unfair social representations and may be used to aggressively market ideas or push controversial agendas. Existing T2I model bias evaluation methods only focus on social biases. We look beyond that and instead propose an evaluation methodology to quantify general biases in T2I generative models, without any preconceived notions. We assess four state-of-the-art T2I models and compare their baseline bias characteristics to their respective variants (two for each), where certain biases have been intentionally induced. We propose three evaluation metrics to assess model biases including: (i) Distribution bias, (ii) Jaccard hallucination and (iii) Generative miss-rate. We conduct two evaluation studies, modelling biases under general, and task-oriented conditions, using a marketing scenario as the domain for the latter. We also quantify social biases to compare our findings to related works. Finally, our methodology is transferred to evaluate captioned-image datasets and measure their bias. Our approach is objective, domain-agnostic and consistently measures different forms of T2I model biases. We have developed a web application and practical implementation of what has been proposed in this work, which is at https://huggingface.co/spaces/JVice/try-before-you-bias. A video series with demonstrations is available at https://www.youtube.com/channel/UCk-0xyUyT0MSd_hkp4jQt1Q

{{</citation>}}


### (19/124) DiffPortrait3D: Controllable Diffusion for Zero-Shot Portrait View Synthesis (Yuming Gu et al., 2023)

{{<citation>}}

Yuming Gu, Hongyi Xu, You Xie, Guoxian Song, Yichun Shi, Di Chang, Jing Yang, Linjie Luo. (2023)  
**DiffPortrait3D: Controllable Diffusion for Zero-Shot Portrait View Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.13016v2)  

---


**ABSTRACT**  
We present DiffPortrait3D, a conditional diffusion model that is capable of synthesizing 3D-consistent photo-realistic novel views from as few as a single in-the-wild portrait. Specifically, given a single RGB input, we aim to synthesize plausible but consistent facial details rendered from novel camera views with retained both identity and facial expression. In lieu of time-consuming optimization and fine-tuning, our zero-shot method generalizes well to arbitrary face portraits with unposed camera views, extreme facial expressions, and diverse artistic depictions. At its core, we leverage the generative prior of 2D diffusion models pre-trained on large-scale image datasets as our rendering backbone, while the denoising is guided with disentangled attentive control of appearance and camera pose. To achieve this, we first inject the appearance context from the reference image into the self-attention layers of the frozen UNets. The rendering view is then manipulated with a novel conditional control module that interprets the camera pose by watching a condition image of a crossed subject from the same view. Furthermore, we insert a trainable cross-view attention module to enhance view consistency, which is further strengthened with a novel 3D-aware noise generation process during inference. We demonstrate state-of-the-art results both qualitatively and quantitatively on our challenging in-the-wild and multi-view benchmarks.

{{</citation>}}


### (20/124) D3Former: Jointly Learning Repeatable Dense Detectors and Feature-enhanced Descriptors via Saliency-guided Transformer (Junjie Gao et al., 2023)

{{<citation>}}

Junjie Gao, Pengfei Wang, Qiujie Dong, Qiong Zeng, Shiqing Xin, Caiming Zhang. (2023)  
**D3Former: Jointly Learning Repeatable Dense Detectors and Feature-enhanced Descriptors via Saliency-guided Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.12970v1)  

---


**ABSTRACT**  
Establishing accurate and representative matches is a crucial step in addressing the point cloud registration problem. A commonly employed approach involves detecting keypoints with salient geometric features and subsequently mapping these keypoints from one frame of the point cloud to another. However, methods within this category are hampered by the repeatability of the sampled keypoints. In this paper, we introduce a saliency-guided trans\textbf{former}, referred to as \textit{D3Former}, which entails the joint learning of repeatable \textbf{D}ense \textbf{D}etectors and feature-enhanced \textbf{D}escriptors. The model comprises a Feature Enhancement Descriptor Learning (FEDL) module and a Repetitive Keypoints Detector Learning (RKDL) module. The FEDL module utilizes a region attention mechanism to enhance feature distinctiveness, while the RKDL module focuses on detecting repeatable keypoints to enhance matching capabilities. Extensive experimental results on challenging indoor and outdoor benchmarks demonstrate that our proposed method consistently outperforms state-of-the-art point cloud matching methods. Notably, tests on 3DLoMatch, even with a low overlap ratio, show that our method consistently outperforms recently published approaches such as RoReg and RoITr. For instance, with the number of extracted keypoints reduced to 250, the registration recall scores for RoReg, RoITr, and our method are 64.3\%, 73.6\%, and 76.5\%, respectively.

{{</citation>}}


### (21/124) Sign Language Production with Latent Motion Transformer (Pan Xie et al., 2023)

{{<citation>}}

Pan Xie, Taiyi Peng, Yao Du, Qipeng Zhang. (2023)  
**Sign Language Production with Latent Motion Transformer**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.12917v1)  

---


**ABSTRACT**  
Sign Language Production (SLP) is the tough task of turning sign language into sign videos. The main goal of SLP is to create these videos using a sign gloss. In this research, we've developed a new method to make high-quality sign videos without using human poses as a middle step. Our model works in two main parts: first, it learns from a generator and the video's hidden features, and next, it uses another model to understand the order of these hidden features. To make this method even better for sign videos, we make several significant improvements. (i) In the first stage, we take an improved 3D VQ-GAN to learn downsampled latent representations. (ii) In the second stage, we introduce sequence-to-sequence attention to better leverage conditional information. (iii) The separated two-stage training discards the realistic visual semantic of the latent codes in the second stage. To endow the latent sequences semantic information, we extend the token-level autoregressive latent codes learning with perceptual loss and reconstruction loss for the prior model with visual perception. Compared with previous state-of-the-art approaches, our model performs consistently better on two word-level sign language datasets, i.e., WLASL and NMFs-CSL.

{{</citation>}}


### (22/124) Produce Once, Utilize Twice for Anomaly Detection (Shuyuan Wang et al., 2023)

{{<citation>}}

Shuyuan Wang, Qi Li, Huiyuan Luo, Chengkan Lv, Zhengtao Zhang. (2023)  
**Produce Once, Utilize Twice for Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection, Embedding  
[Paper Link](http://arxiv.org/abs/2312.12913v1)  

---


**ABSTRACT**  
Visual anomaly detection aims at classifying and locating the regions that deviate from the normal appearance. Embedding-based methods and reconstruction-based methods are two main approaches for this task. However, they are either not efficient or not precise enough for the industrial detection. To deal with this problem, we derive POUTA (Produce Once Utilize Twice for Anomaly detection), which improves both the accuracy and efficiency by reusing the discriminant information potential in the reconstructive network. We observe that the encoder and decoder representations of the reconstructive network are able to stand for the features of the original and reconstructed image respectively. And the discrepancies between the symmetric reconstructive representations provides roughly accurate anomaly information. To refine this information, a coarse-to-fine process is proposed in POUTA, which calibrates the semantics of each discriminative layer by the high-level representations and supervision loss. Equipped with the above modules, POUTA is endowed with the ability to provide a more precise anomaly location than the prior arts. Besides, the representation reusage also enables to exclude the feature extraction process in the discriminative network, which reduces the parameters and improves the efficiency. Extensive experiments show that, POUTA is superior or comparable to the prior methods with even less cost. Furthermore, POUTA also achieves better performance than the state-of-the-art few-shot anomaly detection methods without any special design, showing that POUTA has strong ability to learn representations inherent in the training data.

{{</citation>}}


### (23/124) Integration and Performance Analysis of Artificial Intelligence and Computer Vision Based on Deep Learning Algorithms (Bo Liu et al., 2023)

{{<citation>}}

Bo Liu, Liqiang Yu, Chang Che, Qunwei Lin, Hao Hu, Xinyu Zhao. (2023)  
**Integration and Performance Analysis of Artificial Intelligence and Computer Vision Based on Deep Learning Algorithms**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.12872v1)  

---


**ABSTRACT**  
This paper focuses on the analysis of the application effectiveness of the integration of deep learning and computer vision technologies. Deep learning achieves a historic breakthrough by constructing hierarchical neural networks, enabling end-to-end feature learning and semantic understanding of images. The successful experiences in the field of computer vision provide strong support for training deep learning algorithms. The tight integration of these two fields has given rise to a new generation of advanced computer vision systems, significantly surpassing traditional methods in tasks such as machine vision image classification and object detection. In this paper, typical image classification cases are combined to analyze the superior performance of deep neural network models while also pointing out their limitations in generalization and interpretability, proposing directions for future improvements. Overall, the efficient integration and development trend of deep learning with massive visual data will continue to drive technological breakthroughs and application expansion in the field of computer vision, making it possible to build truly intelligent machine vision systems. This deepening fusion paradigm will powerfully promote unprecedented tasks and functions in computer vision, providing stronger development momentum for related disciplines and industries.

{{</citation>}}


### (24/124) The Audio-Visual Conversational Graph: From an Egocentric-Exocentric Perspective (Wenqi Jia et al., 2023)

{{<citation>}}

Wenqi Jia, Miao Liu, Hao Jiang, Ishwarya Ananthabhotla, James M. Rehg, Vamsi Krishna Ithapu, Ruohan Gao. (2023)  
**The Audio-Visual Conversational Graph: From an Egocentric-Exocentric Perspective**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.12870v1)  

---


**ABSTRACT**  
In recent years, the thriving development of research related to egocentric videos has provided a unique perspective for the study of conversational interactions, where both visual and audio signals play a crucial role. While most prior work focus on learning about behaviors that directly involve the camera wearer, we introduce the Ego-Exocentric Conversational Graph Prediction problem, marking the first attempt to infer exocentric conversational interactions from egocentric videos. We propose a unified multi-modal, multi-task framework -- Audio-Visual Conversational Attention (Av-CONV), for the joint prediction of conversation behaviors -- speaking and listening -- for both the camera wearer as well as all other social partners present in the egocentric video. Specifically, we customize the self-attention mechanism to model the representations across-time, across-subjects, and across-modalities. To validate our method, we conduct experiments on a challenging egocentric video dataset that includes first-person perspective, multi-speaker, and multi-conversation scenarios. Our results demonstrate the superior performance of our method compared to a series of baselines. We also present detailed ablation studies to assess the contribution of each component in our model. Project page: https://vjwq.github.io/AV-CONV/.

{{</citation>}}


### (25/124) RadEdit: stress-testing biomedical vision models via diffusion image editing (Fernando Pérez-García et al., 2023)

{{<citation>}}

Fernando Pérez-García, Sam Bond-Taylor, Pedro P. Sanchez, Boris van Breugel, Daniel C. Castro, Harshita Sharma, Valentina Salvatelli, Maria T. A. Wetscherek, Hannah Richardson, Matthew P. Lungren, Aditya Nori, Javier Alvarez-Valle, Ozan Oktay, Maximilian Ilse. (2023)  
**RadEdit: stress-testing biomedical vision models via diffusion image editing**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12865v1)  

---


**ABSTRACT**  
Biomedical imaging datasets are often small and biased, meaning that real-world performance of predictive models can be substantially lower than expected from internal testing. This work proposes using generative image editing to simulate dataset shifts and diagnose failure modes of biomedical vision models; this can be used in advance of deployment to assess readiness, potentially reducing cost and patient harm. Existing editing methods can produce undesirable changes, with spurious correlations learned due to the co-occurrence of disease and treatment interventions, limiting practical applicability. To address this, we train a text-to-image diffusion model on multiple chest X-ray datasets and introduce a new editing method RadEdit that uses multiple masks, if present, to constrain changes and ensure consistency in the edited images. We consider three types of dataset shifts: acquisition shift, manifestation shift, and population shift, and demonstrate that our approach can diagnose failures and quantify model robustness without additional data collection, complementing more qualitative tools for explainable AI.

{{</citation>}}


### (26/124) Object-aware Adaptive-Positivity Learning for Audio-Visual Question Answering (Zhangbin Li et al., 2023)

{{<citation>}}

Zhangbin Li, Dan Guo, Jinxing Zhou, Jing Zhang, Meng Wang. (2023)  
**Object-aware Adaptive-Positivity Learning for Audio-Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2312.12816v1)  

---


**ABSTRACT**  
This paper focuses on the Audio-Visual Question Answering (AVQA) task that aims to answer questions derived from untrimmed audible videos. To generate accurate answers, an AVQA model is expected to find the most informative audio-visual clues relevant to the given questions. In this paper, we propose to explicitly consider fine-grained visual objects in video frames (object-level clues) and explore the multi-modal relations(i.e., the object, audio, and question) in terms of feature interaction and model optimization. For the former, we present an end-to-end object-oriented network that adopts a question-conditioned clue discovery module to concentrate audio/visual modalities on respective keywords of the question and designs a modality-conditioned clue collection module to highlight closely associated audio segments or visual objects. For model optimization, we propose an object-aware adaptive-positivity learning strategy that selects the highly semantic-matched multi-modal pair as positivity. Specifically, we design two object-aware contrastive loss functions to identify the highly relevant question-object pairs and audio-object pairs, respectively. These selected pairs are constrained to have larger similarity values than the mismatched pairs. The positivity-selecting process is adaptive as the positivity pairs selected in each video frame may be different. These two object-aware objectives help the model understand which objects are exactly relevant to the question and which are making sounds. Extensive experiments on the MUSIC-AVQA dataset demonstrate the proposed method is effective in finding favorable audio-visual clues and also achieves new state-of-the-art question-answering performance.

{{</citation>}}


### (27/124) Mutual-modality Adversarial Attack with Semantic Perturbation (Jingwen Ye et al., 2023)

{{<citation>}}

Jingwen Ye, Ruonan Yu, Songhua Liu, Xinchao Wang. (2023)  
**Mutual-modality Adversarial Attack with Semantic Perturbation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2312.12768v1)  

---


**ABSTRACT**  
Adversarial attacks constitute a notable threat to machine learning systems, given their potential to induce erroneous predictions and classifications. However, within real-world contexts, the essential specifics of the deployed model are frequently treated as a black box, consequently mitigating the vulnerability to such attacks. Thus, enhancing the transferability of the adversarial samples has become a crucial area of research, which heavily relies on selecting appropriate surrogate models. To address this challenge, we propose a novel approach that generates adversarial attacks in a mutual-modality optimization scheme. Our approach is accomplished by leveraging the pre-trained CLIP model. Firstly, we conduct a visual attack on the clean image that causes semantic perturbations on the aligned embedding space with the other textual modality. Then, we apply the corresponding defense on the textual modality by updating the prompts, which forces the re-matching on the perturbed embedding space. Finally, to enhance the attack transferability, we utilize the iterative training strategy on the visual attack and the textual defense, where the two processes optimize from each other. We evaluate our approach on several benchmark datasets and demonstrate that our mutual-modal attack strategy can effectively produce high-transferable attacks, which are stable regardless of the target networks. Our approach outperforms state-of-the-art attack methods and can be readily deployed as a plug-and-play solution.

{{</citation>}}


### (28/124) AMD:Anatomical Motion Diffusion with Interpretable Motion Decomposition and Fusion (Beibei Jing et al., 2023)

{{<citation>}}

Beibei Jing, Youjia Zhang, Zikai Song, Junqing Yu, Wei Yang. (2023)  
**AMD:Anatomical Motion Diffusion with Interpretable Motion Decomposition and Fusion**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.12763v2)  

---


**ABSTRACT**  
Generating realistic human motion sequences from text descriptions is a challenging task that requires capturing the rich expressiveness of both natural language and human motion.Recent advances in diffusion models have enabled significant progress in human motion synthesis.However, existing methods struggle to handle text inputs that describe complex or long motions.In this paper, we propose the Adaptable Motion Diffusion (AMD) model, which leverages a Large Language Model (LLM) to parse the input text into a sequence of concise and interpretable anatomical scripts that correspond to the target motion.This process exploits the LLM's ability to provide anatomical guidance for complex motion synthesis.We then devise a two-branch fusion scheme that balances the influence of the input text and the anatomical scripts on the inverse diffusion process, which adaptively ensures the semantic fidelity and diversity of the synthesized motion.Our method can effectively handle texts with complex or long motion descriptions, where existing methods often fail. Experiments on datasets with relatively more complex motions, such as CLCD1 and CLCD2, demonstrate that our AMD significantly outperforms existing state-of-the-art models.

{{</citation>}}


### (29/124) Spectral Prompt Tuning:Unveiling Unseen Classes for Zero-Shot Semantic Segmentation (Wenhao Xu et al., 2023)

{{<citation>}}

Wenhao Xu, Rongtao Xu, Changwei Wang, Shibiao Xu, Li Guo, Man Zhang, Xiaopeng Zhang. (2023)  
**Spectral Prompt Tuning:Unveiling Unseen Classes for Zero-Shot Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Semantic Segmentation, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.12754v1)  

---


**ABSTRACT**  
Recently, CLIP has found practical utility in the domain of pixel-level zero-shot segmentation tasks. The present landscape features two-stage methodologies beset by issues such as intricate pipelines and elevated computational costs. While current one-stage approaches alleviate these concerns and incorporate Visual Prompt Training (VPT) to uphold CLIP's generalization capacity, they still fall short in fully harnessing CLIP's potential for pixel-level unseen class demarcation and precise pixel predictions. To further stimulate CLIP's zero-shot dense prediction capability, we propose SPT-SEG, a one-stage approach that improves CLIP's adaptability from image to pixel. Specifically, we initially introduce Spectral Prompt Tuning (SPT), incorporating spectral prompts into the CLIP visual encoder's shallow layers to capture structural intricacies of images, thereby enhancing comprehension of unseen classes. Subsequently, we introduce the Spectral Guided Decoder (SGD), utilizing both high and low-frequency information to steer the network's spatial focus towards more prominent classification features, enabling precise pixel-level prediction outcomes. Through extensive experiments on two public datasets, we demonstrate the superiority of our method over state-of-the-art approaches, performing well across all classes and particularly excelling in handling unseen classes. Code is available at:https://github.com/clearxu/SPT.

{{</citation>}}


### (30/124) Cached Transformers: Improving Transformers with Differentiable Memory Cache (Zhaoyang Zhang et al., 2023)

{{<citation>}}

Zhaoyang Zhang, Wenqi Shao, Yixiao Ge, Xiaogang Wang, Jinwei Gu, Ping Luo. (2023)  
**Cached Transformers: Improving Transformers with Differentiable Memory Cache**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.12742v1)  

---


**ABSTRACT**  
This work introduces a new Transformer model called Cached Transformer, which uses Gated Recurrent Cached (GRC) attention to extend the self-attention mechanism with a differentiable memory cache of tokens. GRC attention enables attending to both past and current tokens, increasing the receptive field of attention and allowing for exploring long-range dependencies. By utilizing a recurrent gating unit to continuously update the cache, our model achieves significant advancements in \textbf{six} language and vision tasks, including language modeling, machine translation, ListOPs, image classification, object detection, and instance segmentation. Furthermore, our approach surpasses previous memory-based techniques in tasks such as language modeling and displays the ability to be applied to a broader range of situations.

{{</citation>}}


### (31/124) MetaSegNet: Metadata-collaborative Vision-Language Representation Learning for Semantic Segmentation of Remote Sensing Images (Libo Wang et al., 2023)

{{<citation>}}

Libo Wang, Sijun Dong, Ying Chen, Xiaoliang Meng, Shenghui Fang. (2023)  
**MetaSegNet: Metadata-collaborative Vision-Language Representation Learning for Semantic Segmentation of Remote Sensing Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, ChatGPT, GPT, Representation Learning, Semantic Segmentation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.12735v1)  

---


**ABSTRACT**  
Semantic segmentation of remote sensing images plays a vital role in a wide range of Earth Observation (EO) applications, such as land use land cover mapping, environment monitoring, and sustainable development. Driven by rapid developments in Artificial Intelligence (AI), deep learning (DL) has emerged as the mainstream tool for semantic segmentation and achieved many breakthroughs in the field of remote sensing. However, the existing DL-based methods mainly focus on unimodal visual data while ignoring the rich multimodal information involved in the real world, usually demonstrating weak reliability and generlization. Inspired by the success of Vision Transformers and large language models, we propose a novel metadata-collaborative multimodal segmentation network (MetaSegNet) that applies vision-language representation learning for semantic segmentation of remote sensing images. Unlike the common model structure that only uses unimodal visual data, we extract the key characteristic (i.e. the climate zone) from freely available remote sensing image metadata and transfer it into knowledge-based text prompts via the generic ChatGPT. Then, we construct an image encoder, a text encoder and a crossmodal attention fusion subnetwork to extract the image and text feature and apply image-text interaction. Benefiting from such a design, the proposed MetaSegNet demonstrates superior generalization and achieves competitive accuracy with state-of-the-art semantic segmentation methods on the large-scale OpenEarthMap dataset (68.6% mIoU) and Potsdam dataset (93.3% mean F1 score) as well as LoveDA dataset (52.2% mIoU).

{{</citation>}}


### (32/124) A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models (Julio Silva-Rodriguez et al., 2023)

{{<citation>}}

Julio Silva-Rodriguez, Sina Hajimiri, Ismail Ben Ayed, Jose Dolz. (2023)  
**A Closer Look at the Few-Shot Adaptation of Large Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Language Model  
[Paper Link](http://arxiv.org/abs/2312.12730v1)  

---


**ABSTRACT**  
Efficient transfer learning (ETL) is receiving increasing attention to adapt large pre-trained language-vision models on downstream tasks with a few labeled samples. While significant progress has been made, we reveal that state-of-the-art ETL approaches exhibit strong performance only in narrowly-defined experimental setups, and with a careful adjustment of hyperparameters based on a large corpus of labeled samples. In particular, we make two interesting, and surprising empirical observations. First, to outperform a simple Linear Probing baseline, these methods require to optimize their hyper-parameters on each target task. And second, they typically underperform -- sometimes dramatically -- standard zero-shot predictions in the presence of distributional drifts. Motivated by the unrealistic assumptions made in the existing literature, i.e., access to a large validation set and case-specific grid-search for optimal hyperparameters, we propose a novel approach that meets the requirements of real-world scenarios. More concretely, we introduce a CLass-Adaptive linear Probe (CLAP) objective, whose balancing term is optimized via an adaptation of the general Augmented Lagrangian method tailored to this context. We comprehensively evaluate CLAP on a broad span of datasets and scenarios, demonstrating that it consistently outperforms SoTA approaches, while yet being a much more efficient alternative.

{{</citation>}}


### (33/124) Multi-Clue Reasoning with Memory Augmentation for Knowledge-based Visual Question Answering (Chengxiang Yin et al., 2023)

{{<citation>}}

Chengxiang Yin, Zhengping Che, Kun Wu, Zhiyuan Xu, Jian Tang. (2023)  
**Multi-Clue Reasoning with Memory Augmentation for Knowledge-based Visual Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, QA, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.12723v1)  

---


**ABSTRACT**  
Visual Question Answering (VQA) has emerged as one of the most challenging tasks in artificial intelligence due to its multi-modal nature. However, most existing VQA methods are incapable of handling Knowledge-based Visual Question Answering (KB-VQA), which requires external knowledge beyond visible contents to answer questions about a given image. To address this issue, we propose a novel framework that endows the model with capabilities of answering more general questions, and achieves a better exploitation of external knowledge through generating Multiple Clues for Reasoning with Memory Neural Networks (MCR-MemNN). Specifically, a well-defined detector is adopted to predict image-question related relation phrases, each of which delivers two complementary clues to retrieve the supporting facts from external knowledge base (KB), which are further encoded into a continuous embedding space using a content-addressable memory. Afterwards, mutual interactions between visual-semantic representation and the supporting facts stored in memory are captured to distill the most relevant information in three modalities (i.e., image, question, and KB). Finally, the optimal answer is predicted by choosing the supporting fact with the highest score. We conduct extensive experiments on two widely-used benchmarks. The experimental results well justify the effectiveness of MCR-MemNN, as well as its superiority over other KB-VQA methods.

{{</citation>}}


### (34/124) Fine-Grained Knowledge Selection and Restoration for Non-Exemplar Class Incremental Learning (Jiang-Tian Zhai et al., 2023)

{{<citation>}}

Jiang-Tian Zhai, Xialei Liu, Lu Yu, Ming-Ming Cheng. (2023)  
**Fine-Grained Knowledge Selection and Restoration for Non-Exemplar Class Incremental Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.12722v1)  

---


**ABSTRACT**  
Non-exemplar class incremental learning aims to learn both the new and old tasks without accessing any training data from the past. This strict restriction enlarges the difficulty of alleviating catastrophic forgetting since all techniques can only be applied to current task data. Considering this challenge, we propose a novel framework of fine-grained knowledge selection and restoration. The conventional knowledge distillation-based methods place too strict constraints on the network parameters and features to prevent forgetting, which limits the training of new tasks. To loose this constraint, we proposed a novel fine-grained selective patch-level distillation to adaptively balance plasticity and stability. Some task-agnostic patches can be used to preserve the decision boundary of the old task. While some patches containing the important foreground are favorable for learning the new task.   Moreover, we employ a task-agnostic mechanism to generate more realistic prototypes of old tasks with the current task sample for reducing classifier bias for fine-grained knowledge restoration. Extensive experiments on CIFAR100, TinyImageNet and ImageNet-Subset demonstrate the effectiveness of our method. Code is available at https://github.com/scok30/vit-cil.

{{</citation>}}


### (35/124) Cross-Modal Reasoning with Event Correlation for Video Question Answering (Chengxiang Yin et al., 2023)

{{<citation>}}

Chengxiang Yin, Zhengping Che, Kun Wu, Zhiyuan Xu, Qinru Qiu, Jian Tang. (2023)  
**Cross-Modal Reasoning with Event Correlation for Video Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, QA, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.12721v1)  

---


**ABSTRACT**  
Video Question Answering (VideoQA) is a very attractive and challenging research direction aiming to understand complex semantics of heterogeneous data from two domains, i.e., the spatio-temporal video content and the word sequence in question. Although various attention mechanisms have been utilized to manage contextualized representations by modeling intra- and inter-modal relationships of the two modalities, one limitation of the predominant VideoQA methods is the lack of reasoning with event correlation, that is, sensing and analyzing relationships among abundant and informative events contained in the video. In this paper, we introduce the dense caption modality as a new auxiliary and distill event-correlated information from it to infer the correct answer. To this end, we propose a novel end-to-end trainable model, Event-Correlated Graph Neural Networks (EC-GNNs), to perform cross-modal reasoning over information from the three modalities (i.e., caption, video, and question). Besides the exploitation of a brand new modality, we employ cross-modal reasoning modules for explicitly modeling inter-modal relationships and aggregating relevant information across different modalities, and we propose a question-guided self-adaptive multi-modal fusion module to collect the question-oriented and event-correlated evidence through multi-step reasoning. We evaluate our model on two widely-used benchmark datasets and conduct an ablation study to justify the effectiveness of each proposed component.

{{</citation>}}


### (36/124) AdvST: Revisiting Data Augmentations for Single Domain Generalization (Guangtao Zheng et al., 2023)

{{<citation>}}

Guangtao Zheng, Mengdi Huai, Aidong Zhang. (2023)  
**AdvST: Revisiting Data Augmentations for Single Domain Generalization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.12720v1)  

---


**ABSTRACT**  
Single domain generalization (SDG) aims to train a robust model against unknown target domain shifts using data from a single source domain. Data augmentation has been proven an effective approach to SDG. However, the utility of standard augmentations, such as translate, or invert, has not been fully exploited in SDG; practically, these augmentations are used as a part of a data preprocessing procedure. Although it is intuitive to use many such augmentations to boost the robustness of a model to out-of-distribution domain shifts, we lack a principled approach to harvest the benefit brought from multiple these augmentations. Here, we conceptualize standard data augmentations with learnable parameters as semantics transformations that can manipulate certain semantics of a sample, such as the geometry or color of an image. Then, we propose Adversarial learning with Semantics Transformations (AdvST) that augments the source domain data with semantics transformations and learns a robust model with the augmented data. We theoretically show that AdvST essentially optimizes a distributionally robust optimization objective defined on a set of semantics distributions induced by the parameters of semantics transformations. We demonstrate that AdvST can produce samples that expand the coverage on target domain data. Compared with the state-of-the-art methods, AdvST, despite being a simple method, is surprisingly competitive and achieves the best average SDG performance on the Digits, PACS, and DomainNet datasets. Our code is available at https://github.com/gtzheng/AdvST.

{{</citation>}}


### (37/124) BloomVQA: Assessing Hierarchical Multi-modal Comprehension (Yunye Gong et al., 2023)

{{<citation>}}

Yunye Gong, Robik Shrestha, Jared Claypoole, Michael Cogswell, Arijit Ray, Christopher Kanan, Ajay Divakaran. (2023)  
**BloomVQA: Assessing Hierarchical Multi-modal Comprehension**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.12716v1)  

---


**ABSTRACT**  
We propose a novel VQA dataset, based on picture stories designed for educating young children, that aims to facilitate comprehensive evaluation and characterization of vision-language models on comprehension tasks. Unlike current VQA datasets that often focus on fact-based memorization and simple reasoning tasks without principled scientific grounding, we collect data containing tasks reflecting different levels of comprehension and underlying cognitive processes, as laid out in Bloom's Taxonomy, a classic framework widely adopted in education research. The proposed BloomVQA dataset can be mapped to a hierarchical graph-based representation of visual stories, enabling automatic data augmentation and novel measures characterizing model consistency across the underlying taxonomy. We demonstrate graded evaluation and reliability analysis based on our proposed consistency metrics on state-of-the-art vision-language models. Our results suggest that, while current models achieve the most gain on low-level comprehension tasks, they generally fall short on high-level tasks requiring more advanced comprehension and cognitive skills, as 38.0% drop in VQA accuracy is observed comparing lowest and highest level tasks. Furthermore, current models show consistency patterns misaligned with human comprehension in various scenarios, suggesting emergent structures of model behaviors.

{{</citation>}}


## cs.DB (1)



### (38/124) R2D2: Reducing Redundancy and Duplication in Data Lakes (Raunak Shah et al., 2023)

{{<citation>}}

Raunak Shah, Koyel Mukherjee, Atharv Tyagi, Sai Keerthana Karnam, Dhruv Joshi, Shivam Bhosale, Subrata Mitra. (2023)  
**R2D2: Reducing Redundancy and Duplication in Data Lakes**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: AWS, Azure  
[Paper Link](http://arxiv.org/abs/2312.13427v1)  

---


**ABSTRACT**  
Enterprise data lakes often suffer from substantial amounts of duplicate and redundant data, with data volumes ranging from terabytes to petabytes. This leads to both increased storage costs and unnecessarily high maintenance costs for these datasets. In this work, we focus on identifying and reducing redundancy in enterprise data lakes by addressing the problem of 'dataset containment'. To the best of our knowledge, this is one of the first works that addresses table-level containment at a large scale.   We propose R2D2: a three-step hierarchical pipeline that efficiently identifies almost all instances of containment by progressively reducing the search space in the data lake. It first builds (i) a schema containment graph, followed by (ii) statistical min-max pruning, and finally, (iii) content level pruning. We further propose minimizing the total storage and access costs by optimally identifying redundant datasets that can be deleted (and reconstructed on demand) while respecting latency constraints.   We implement our system on Azure Databricks clusters using Apache Spark for enterprise data stored in ADLS Gen2, and on AWS clusters for open-source data. In contrast to existing modified baselines that are inaccurate or take several days to run, our pipeline can process an enterprise customer data lake at the TB scale in approximately 5 hours with high accuracy. We present theoretical results as well as extensive empirical validation on both enterprise (scale of TBs) and open-source datasets (scale of MBs - GBs), which showcase the effectiveness of our pipeline.

{{</citation>}}


## cs.DL (1)



### (39/124) VADIS -- a VAriable Detection, Interlinking and Summarization system (Yavuz Selim Kartal et al., 2023)

{{<citation>}}

Yavuz Selim Kartal, Muhammad Ahsan Shahid, Sotaro Takeshita, Tornike Tsereteli, Andrea Zielinski, Benjamin Zapilko, Philipp Mayr. (2023)  
**VADIS -- a VAriable Detection, Interlinking and Summarization system**  

---
Primary Category: cs.DL  
Categories: cs-CL, cs-DL, cs-IR, cs.DL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2312.13423v1)  

---


**ABSTRACT**  
The VADIS system addresses the demand of providing enhanced information access in the domain of the social sciences. This is achieved by allowing users to search and use survey variables in context of their underlying research data and scholarly publications which have been interlinked with each other.

{{</citation>}}


## cs.CL (23)



### (40/124) Time is Encoded in the Weights of Finetuned Language Models (Kai Nylund et al., 2023)

{{<citation>}}

Kai Nylund, Suchin Gururangan, Noah A. Smith. (2023)  
**Time is Encoded in the Weights of Finetuned Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.13401v1)  

---


**ABSTRACT**  
We present time vectors, a simple tool to customize language models to new time periods. Time vectors are created by finetuning a language model on data from a single time (e.g., a year or month), and then subtracting the weights of the original pretrained model. This vector specifies a direction in weight space that, as our experiments show, improves performance on text from that time period. Time vectors specialized to adjacent time periods appear to be positioned closer together in a manifold. Using this structure, we interpolate between time vectors to induce new models that perform better on intervening and future time periods, without any additional training. We demonstrate the consistency of our findings across different tasks, domains, model sizes, and time scales. Our results suggest that time is encoded in the weight space of finetuned models.

{{</citation>}}


### (41/124) DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines (Arnav Singhvi et al., 2023)

{{<citation>}}

Arnav Singhvi, Manish Shetty, Shangyin Tan, Christopher Potts, Koushik Sen, Matei Zaharia, Omar Khattab. (2023)  
**DSPy Assertions: Computational Constraints for Self-Refining Language Model Pipelines**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-PL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2312.13382v1)  

---


**ABSTRACT**  
Chaining language model (LM) calls as composable modules is fueling a new powerful way of programming. However, ensuring that LMs adhere to important constraints remains a key challenge, one often addressed with heuristic "prompt engineering". We introduce LM Assertions, a new programming construct for expressing computational constraints that LMs should satisfy. We integrate our constructs into the recent DSPy programming model for LMs, and present new strategies that allow DSPy to compile programs with arbitrary LM Assertions into systems that are more reliable and more accurate. In DSPy, LM Assertions can be integrated at compile time, via automatic prompt optimization, and/or at inference time, via automatic selfrefinement and backtracking. We report on two early case studies for complex question answering (QA), in which the LM program must iteratively retrieve information in multiple hops and synthesize a long-form answer with citations. We find that LM Assertions improve not only compliance with imposed rules and guidelines but also enhance downstream task performance, delivering intrinsic and extrinsic gains up to 35.7% and 13.3%, respectively. Our reference implementation of LM Assertions is integrated into DSPy at https://github.com/stanfordnlp/dspy

{{</citation>}}


### (42/124) dIR -- Discrete Information Retrieval: Conversational Search over Unstructured (and Structured) Data with Large Language Models (Pablo M. Rodriguez Bertorello et al., 2023)

{{<citation>}}

Pablo M. Rodriguez Bertorello, Jean Rodmond Junior Laguerre. (2023)  
**dIR -- Discrete Information Retrieval: Conversational Search over Unstructured (and Structured) Data with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-DB, cs-IR, cs-LG, cs.CL  
Keywords: Information Retrieval, Language Model  
[Paper Link](http://arxiv.org/abs/2312.13264v1)  

---


**ABSTRACT**  
Data is stored in both structured and unstructured form. Querying both, to power natural language conversations, is a challenge. This paper introduces dIR, Discrete Information Retrieval, providing a unified interface to query both free text and structured knowledge. Specifically, a Large Language Model (LLM) transforms text into expressive representation. After the text is extracted into columnar form, it can then be queried via a text-to-SQL Semantic Parser, with an LLM converting natural language into SQL. Where desired, such conversation may be effected by a multi-step reasoning conversational agent. We validate our approach via a proprietary question/answer data set, concluding that dIR makes a whole new class of queries on free text possible when compared to traditionally fine-tuned dense-embedding-model-based Information Retrieval (IR) and SQL-based Knowledge Bases (KB). For sufficiently complex queries, dIR can succeed where no other method stands a chance.

{{</citation>}}


### (43/124) DSFormer: Effective Compression of Text-Transformers by Dense-Sparse Weight Factorization (Rahul Chand et al., 2023)

{{<citation>}}

Rahul Chand, Yashoteja Prabhu, Pratyush Kumar. (2023)  
**DSFormer: Effective Compression of Text-Transformers by Dense-Sparse Weight Factorization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.13211v1)  

---


**ABSTRACT**  
With the tremendous success of large transformer models in natural language understanding, down-sizing them for cost-effective deployments has become critical. Recent studies have explored the low-rank weight factorization techniques which are efficient to train, and apply out-of-the-box to any transformer architecture. Unfortunately, the low-rank assumption tends to be over-restrictive and hinders the expressiveness of the compressed model. This paper proposes, DSFormer, a simple alternative factorization scheme which expresses a target weight matrix as the product of a small dense and a semi-structured sparse matrix. The resulting approximation is more faithful to the weight distribution in transformers and therefore achieves a stronger efficiency-accuracy trade-off. Another concern with existing factorizers is their dependence on a task-unaware initialization step which degrades the accuracy of the resulting model. DSFormer addresses this issue through a novel Straight-Through Factorizer (STF) algorithm that jointly learns all the weight factorizations to directly maximize the final task accuracy. Extensive experiments on multiple natural language understanding benchmarks demonstrate that DSFormer obtains up to 40% better compression than the state-of-the-art low-rank factorizers, leading semi-structured sparsity baselines and popular knowledge distillation approaches. Our approach is also orthogonal to mainstream compressors and offers up to 50% additional compression when added to popular distilled, layer-shared and quantized transformers. We empirically evaluate the benefits of STF over conventional optimization practices.

{{</citation>}}


### (44/124) LlaMaVAE: Guiding Large Language Model Generation via Continuous Latent Sentence Spaces (Yingji Zhang et al., 2023)

{{<citation>}}

Yingji Zhang, Danilo S. Carvalho, Ian Pratt-Hartmann, André Freitas. (2023)  
**LlaMaVAE: Guiding Large Language Model Generation via Continuous Latent Sentence Spaces**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, T5  
[Paper Link](http://arxiv.org/abs/2312.13208v1)  

---


**ABSTRACT**  
Deep generative neural networks, such as Variational AutoEncoders (VAEs), offer an opportunity to better understand and control language models from the perspective of sentence-level latent spaces. To combine the controllability of VAE latent spaces with the state-of-the-art performance of recent large language models (LLMs), we present in this work LlaMaVAE, which combines expressive encoder and decoder models (sentenceT5 and LlaMA) with a VAE architecture, aiming to provide better text generation control to LLMs. In addition, to conditionally guide the VAE generation, we investigate a new approach based on flow-based invertible neural networks (INNs) named Invertible CVAE. Experimental results reveal that LlaMaVAE can outperform the previous state-of-the-art VAE language model, Optimus, across various tasks, including language modelling, semantic textual similarity and definition modelling. Qualitative analysis on interpolation and traversal experiments also indicates an increased degree of semantic clustering and geometric consistency, which enables better generation control.

{{</citation>}}


### (45/124) HCDIR: End-to-end Hate Context Detection, and Intensity Reduction model for online comments (Neeraj Kumar Singh et al., 2023)

{{<citation>}}

Neeraj Kumar Singh, Koyel Ghosh, Joy Mahapatra, Utpal Garain, Apurbalal Senapati. (2023)  
**HCDIR: End-to-end Hate Context Detection, and Intensity Reduction model for online comments**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.13193v1)  

---


**ABSTRACT**  
Warning: This paper contains examples of the language that some people may find offensive.   Detecting and reducing hateful, abusive, offensive comments is a critical and challenging task on social media. Moreover, few studies aim to mitigate the intensity of hate speech. While studies have shown that context-level semantics are crucial for detecting hateful comments, most of this research focuses on English due to the ample datasets available. In contrast, low-resource languages, like Indian languages, remain under-researched because of limited datasets. Contrary to hate speech detection, hate intensity reduction remains unexplored in high-resource and low-resource languages. In this paper, we propose a novel end-to-end model, HCDIR, for Hate Context Detection, and Hate Intensity Reduction in social media posts. First, we fine-tuned several pre-trained language models to detect hateful comments to ascertain the best-performing hateful comments detection model. Then, we identified the contextual hateful words. Identification of such hateful words is justified through the state-of-the-art explainable learning model, i.e., Integrated Gradient (IG). Lastly, the Masked Language Modeling (MLM) model has been employed to capture domain-specific nuances to reduce hate intensity. We masked the 50\% hateful words of the comments identified as hateful and predicted the alternative words for these masked terms to generate convincing sentences. An optimal replacement for the original hate comments from the feasible sentences is preferred. Extensive experiments have been conducted on several recent datasets using automatic metric-based evaluation (BERTScore) and thorough human evaluation. To enhance the faithfulness in human evaluation, we arranged a group of three human annotators with varied expertise.

{{</citation>}}


### (46/124) Contextual Code Switching for Machine Translation using Language Models (Arshad Kaji et al., 2023)

{{<citation>}}

Arshad Kaji, Manan Shah. (2023)  
**Contextual Code Switching for Machine Translation using Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2312.13179v1)  

---


**ABSTRACT**  
Large language models (LLMs) have exerted a considerable impact on diverse language-related tasks in recent years. Their demonstrated state-of-the-art performance is achieved through methodologies such as zero-shot or few-shot prompting. These models undergo training on extensive datasets that encompass segments of the Internet and subsequently undergo fine-tuning tailored to specific tasks. Notably, they exhibit proficiency in tasks such as translation, summarization, question answering, and creative writing, even in the absence of explicit training for those particular tasks. While they have shown substantial improvement in the multilingual tasks their performance in the code switching, especially for machine translation remains relatively uncharted. In this paper, we present an extensive study on the code switching task specifically for the machine translation task comparing multiple LLMs. Our results indicate that despite the LLMs having promising results in the certain tasks, the models with relatively lesser complexity outperform the multilingual large language models in the machine translation task. We posit that the efficacy of multilingual large language models in contextual code switching is constrained by their training methodologies. In contrast, relatively smaller models, when trained and fine-tuned on bespoke datasets, may yield superior results in comparison to the majority of multilingual models.

{{</citation>}}


### (47/124) Exploring Multimodal Large Language Models for Radiology Report Error-checking (Jinge Wu et al., 2023)

{{<citation>}}

Jinge Wu, Yunsoo Kim, Eva C. Keller, Jamie Chow, Adam P. Levine, Nikolas Pontikos, Zina Ibrahim, Paul Taylor, Michelle C. Williams, Honghan Wu. (2023)  
**Exploring Multimodal Large Language Models for Radiology Report Error-checking**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.13103v1)  

---


**ABSTRACT**  
This paper proposes one of the first clinical applications of multimodal large language models (LLMs) as an assistant for radiologists to check errors in their reports. We created an evaluation dataset from two real-world radiology datasets (MIMIC-CXR and IU-Xray), with 1,000 subsampled reports each. A subset of original reports was modified to contain synthetic errors by introducing various type of mistakes. The evaluation contained two difficulty levels: SIMPLE for binary error-checking and COMPLEX for identifying error types. LLaVA (Large Language and Visual Assistant) variant models, including our instruction-tuned model, were used for the evaluation. Additionally, a domain expert evaluation was conducted on a small test set. At the SIMPLE level, the LLaVA v1.5 model outperformed other publicly available models. Instruction tuning significantly enhanced performance by 47.4% and 25.4% on MIMIC-CXR and IU-Xray data, respectively. The model also surpassed the domain experts accuracy in the MIMIC-CXR dataset by 1.67%. Notably, among the subsets (N=21) of the test set where a clinician did not achieve the correct conclusion, the LLaVA ensemble mode correctly identified 71.4% of these cases. This study marks a promising step toward utilizing multi-modal LLMs to enhance diagnostic accuracy in radiology. The ensemble model demonstrated comparable performance to clinicians, even capturing errors overlooked by humans. Nevertheless, future work is needed to improve the model ability to identify the types of inconsistency.

{{</citation>}}


### (48/124) In Generative AI we Trust: Can Chatbots Effectively Verify Political Information? (Elizaveta Kuznetsova et al., 2023)

{{<citation>}}

Elizaveta Kuznetsova, Mykola Makhortykh, Victoria Vziatysheva, Martha Stolze, Ani Baghumyan, Aleksandra Urman. (2023)  
**In Generative AI we Trust: Can Chatbots Effectively Verify Political Information?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs.CL  
Keywords: AI, ChatGPT, GPT, Generative AI, Microsoft  
[Paper Link](http://arxiv.org/abs/2312.13096v1)  

---


**ABSTRACT**  
This article presents a comparative analysis of the ability of two large language model (LLM)-based chatbots, ChatGPT and Bing Chat, recently rebranded to Microsoft Copilot, to detect veracity of political information. We use AI auditing methodology to investigate how chatbots evaluate true, false, and borderline statements on five topics: COVID-19, Russian aggression against Ukraine, the Holocaust, climate change, and LGBTQ+ related debates. We compare how the chatbots perform in high- and low-resource languages by using prompts in English, Russian, and Ukrainian. Furthermore, we explore the ability of chatbots to evaluate statements according to political communication concepts of disinformation, misinformation, and conspiracy theory, using definition-oriented prompts. We also systematically test how such evaluations are influenced by source bias which we model by attributing specific claims to various political and social actors. The results show high performance of ChatGPT for the baseline veracity evaluation task, with 72 percent of the cases evaluated correctly on average across languages without pre-training. Bing Chat performed worse with a 67 percent accuracy. We observe significant disparities in how chatbots evaluate prompts in high- and low-resource languages and how they adapt their evaluations to political communication concepts with ChatGPT providing more nuanced outputs than Bing Chat. Finally, we find that for some veracity detection-related tasks, the performance of chatbots varied depending on the topic of the statement or the source to which it is attributed. These findings highlight the potential of LLM-based chatbots in tackling different forms of false information in online environments, but also points to the substantial variation in terms of how such potential is realized due to specific factors, such as language of the prompt or the topic.

{{</citation>}}


### (49/124) Retrieval-augmented Multilingual Knowledge Editing (Weixuan Wang et al., 2023)

{{<citation>}}

Weixuan Wang, Barry Haddow, Alexandra Birch. (2023)  
**Retrieval-augmented Multilingual Knowledge Editing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Multilingual  
[Paper Link](http://arxiv.org/abs/2312.13040v1)  

---


**ABSTRACT**  
Knowledge represented in Large Language Models (LLMs) is quite often incorrect and can also become obsolete over time. Updating knowledge via fine-tuning is computationally resource-hungry and not reliable, and so knowledge editing (KE) has developed as an effective and economical alternative to inject new knowledge or to fix factual errors in LLMs. Although there has been considerable interest in this area, current KE research exclusively focuses on the monolingual setting, typically in English. However, what happens if the new knowledge is supplied in one language, but we would like to query the LLM in a different language? To address the problem of multilingual knowledge editing, we propose Retrieval-augmented Multilingual Knowledge Editor (ReMaKE) to update new knowledge in LLMs. ReMaKE can perform model-agnostic knowledge editing in multilingual settings. ReMaKE concatenates the new knowledge retrieved from a multilingual knowledge base with prompts. Our experimental results show that ReMaKE outperforms baseline knowledge editing methods by a significant margin and is the first KE method to work in a multilingual setting. We provide our multilingual knowledge editing dataset (MzsRE) in 12 languages, which along with code, and additional project information is available at https://github.com/Vicky-Wil/ReMaKE.

{{</citation>}}


### (50/124) AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and Optimisation (Dong Huang et al., 2023)

{{<citation>}}

Dong Huang, Qingwen Bu, Jie M. Zhang, Michael Luck, Heming Cui. (2023)  
**AgentCoder: Multi-Agent-based Code Generation with Iterative Testing and Optimisation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, NLP  
[Paper Link](http://arxiv.org/abs/2312.13010v1)  

---


**ABSTRACT**  
The advancement of natural language processing (NLP) has been significantly boosted by the development of transformer-based large language models (LLMs). These models have revolutionized NLP tasks, particularly in code generation, aiding developers in creating software with enhanced efficiency. Despite their advancements, challenges in balancing code snippet generation with effective test case generation and execution persist. To address these issues, this paper introduces Multi-Agent Assistant Code Generation (AgentCoder), a novel solution comprising a multi-agent framework with specialized agents: the programmer agent, the test designer agent, and the test executor agent. During the coding procedure, the programmer agent will focus on the code generation and refinement based on the test executor agent's feedback. The test designer agent will generate test cases for the generated code, and the test executor agent will run the code with the test cases and write the feedback to the programmer. This collaborative system ensures robust code generation, surpassing the limitations of single-agent models and traditional methodologies. Our extensive experiments on 9 code generation models and 12 enhancement approaches showcase AgentCoder's superior performance over existing code generation models and prompt engineering techniques across various benchmarks. For example, AgentCoder achieves 77.4% and 89.1% pass@1 in HumanEval-ET and MBPP-ET with GPT-3.5, while SOTA baselines obtain only 69.5% and 63.0%.

{{</citation>}}


### (51/124) Machine Mindset: An MBTI Exploration of Large Language Models (Jiaxi Cui et al., 2023)

{{<citation>}}

Jiaxi Cui, Liuzhenghao Lv, Jing Wen, Jing Tang, YongHong Tian, Li Yuan. (2023)  
**Machine Mindset: An MBTI Exploration of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.12999v2)  

---


**ABSTRACT**  
We present a novel approach for integrating Myers-Briggs Type Indicator (MBTI) personality traits into large language models (LLMs), addressing the challenges of personality consistency in personalized AI. Our method, "Machine Mindset," involves a two-phase fine-tuning and Direct Preference Optimization (DPO) to embed MBTI traits into LLMs. This approach ensures that models internalize these traits, offering a stable and consistent personality profile. We demonstrate the effectiveness of our models across various domains, showing alignment between model performance and their respective MBTI traits. The paper highlights significant contributions in the development of personality datasets and a new training methodology for personality integration in LLMs, enhancing the potential for personalized AI applications. We also open-sourced our model and part of the data at \url{https://github.com/PKU-YuanGroup/Machine-Mindset}.

{{</citation>}}


### (52/124) Assaying on the Robustness of Zero-Shot Machine-Generated Text Detectors (Yi-Fan Zhang et al., 2023)

{{<citation>}}

Yi-Fan Zhang, Zhang Zhang, Liang Wang, Tieniu Tan, Rong Jin. (2023)  
**Assaying on the Robustness of Zero-Shot Machine-Generated Text Detectors**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model, Natural Language Generation, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.12918v2)  

---


**ABSTRACT**  
To combat the potential misuse of Natural Language Generation (NLG) technology, a variety of algorithms have been developed for the detection of AI-generated texts. Traditionally, this task is treated as a binary classification problem. Although supervised learning has demonstrated promising results, acquiring labeled data for detection purposes poses real-world challenges and the risk of overfitting. In an effort to address these issues, we delve into the realm of zero-shot machine-generated text detection. Existing zero-shot detectors, typically designed for specific tasks or topics, often assume uniform testing scenarios, limiting their practicality. In our research, we explore various advanced Large Language Models (LLMs) and their specialized variants, contributing to this field in several ways. In empirical studies, we uncover a significant correlation between topics and detection performance. Secondly, we delve into the influence of topic shifts on zero-shot detectors. These investigations shed light on the adaptability and robustness of these detection methods across diverse topics. The code is available at \url{https://github.com/yfzhang114/robustness-detection}.

{{</citation>}}


### (53/124) CORECODE: A Common Sense Annotated Dialogue Dataset with Benchmark Tasks for Chinese Large Language Models (Dan Shi et al., 2023)

{{<citation>}}

Dan Shi, Chaobin You, Jiantao Huang, Taihao Li, Deyi Xiong. (2023)  
**CORECODE: A Common Sense Annotated Dialogue Dataset with Benchmark Tasks for Chinese Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, Dialogue, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.12853v1)  

---


**ABSTRACT**  
As an indispensable ingredient of intelligence, commonsense reasoning is crucial for large language models (LLMs) in real-world scenarios. In this paper, we propose CORECODE, a dataset that contains abundant commonsense knowledge manually annotated on dyadic dialogues, to evaluate the commonsense reasoning and commonsense conflict detection capabilities of Chinese LLMs. We categorize commonsense knowledge in everyday conversations into three dimensions: entity, event, and social interaction. For easy and consistent annotation, we standardize the form of commonsense knowledge annotation in open-domain dialogues as "domain: slot = value". A total of 9 domains and 37 slots are defined to capture diverse commonsense knowledge. With these pre-defined domains and slots, we collect 76,787 commonsense knowledge annotations from 19,700 dialogues through crowdsourcing. To evaluate and enhance the commonsense reasoning capability for LLMs on the curated dataset, we establish a series of dialogue-level reasoning and detection tasks, including commonsense knowledge filling, commonsense knowledge generation, commonsense conflict phrase detection, domain identification, slot identification, and event causal inference. A wide variety of existing open-source Chinese LLMs are evaluated with these tasks on our dataset. Experimental results demonstrate that these models are not competent to predict CORECODE's plentiful reasoning content, and even ChatGPT could only achieve 0.275 and 0.084 accuracy on the domain identification and slot identification tasks under the zero-shot setting. We release the data and codes of CORECODE at https://github.com/danshi777/CORECODE to promote commonsense reasoning evaluation and study of LLMs in the context of daily conversations.

{{</citation>}}


### (54/124) Language Resources for Dutch Large Language Modelling (Bram Vanroy, 2023)

{{<citation>}}

Bram Vanroy. (2023)  
**Language Resources for Dutch Large Language Modelling**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.12852v1)  

---


**ABSTRACT**  
Despite the rapid expansion of types of large language models, there remains a notable gap in models specifically designed for the Dutch language. This gap is not only a shortage in terms of pretrained Dutch models but also in terms of data, and benchmarks and leaderboards. This work provides a small step to improve the situation. First, we introduce two fine-tuned variants of the Llama 2 13B model. We first fine-tuned Llama 2 using Dutch-specific web-crawled data and subsequently refined this model further on multiple synthetic instruction and chat datasets. These datasets as well as the model weights are made available. In addition, we provide a leaderboard to keep track of the performance of (Dutch) models on a number of generation tasks, and we include results of a number of state-of-the-art models, including our own. Finally we provide a critical conclusion on what we believe is needed to push forward Dutch language models and the whole eco-system around the models.

{{</citation>}}


### (55/124) Turning Dust into Gold: Distilling Complex Reasoning Capabilities from LLMs by Leveraging Negative Data (Yiwei Li et al., 2023)

{{<citation>}}

Yiwei Li, Peiwen Yuan, Shaoxiong Feng, Boyuan Pan, Bin Sun, Xinglin Wang, Heda Wang, Kan Li. (2023)  
**Turning Dust into Gold: Distilling Complex Reasoning Capabilities from LLMs by Leveraging Negative Data**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.12832v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have performed well on various reasoning tasks, but their inaccessibility and numerous parameters hinder wide application in practice. One promising way is distilling the reasoning ability from LLMs to small models by the generated chain-of-thought reasoning paths. In some cases, however, LLMs may produce incorrect reasoning chains, especially when facing complex mathematical problems. Previous studies only transfer knowledge from positive samples and drop the synthesized data with wrong answers. In this work, we illustrate the merit of negative data and propose a model specialization framework to distill LLMs with negative samples besides positive ones. The framework consists of three progressive steps, covering from training to inference stages, to absorb knowledge from negative data. We conduct extensive experiments across arithmetic reasoning tasks to demonstrate the role of negative data in distillation from LLM.

{{</citation>}}


### (56/124) Enhancing Consistency in Multimodal Dialogue System Using LLM with Dialogue Scenario (Hiroki Onozeki et al., 2023)

{{<citation>}}

Hiroki Onozeki, Zhiyang Qi, Kazuma Akiyama, Ryutaro Asahara, Takumasa Kaneko, Michimasa Inaba. (2023)  
**Enhancing Consistency in Multimodal Dialogue System Using LLM with Dialogue Scenario**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2312.12808v1)  

---


**ABSTRACT**  
This paper describes our dialogue system submitted to Dialogue Robot Competition 2023. The system's task is to help a user at a travel agency decide on a plan for visiting two sightseeing spots in Kyoto City that satisfy the user. Our dialogue system is flexible and stable and responds to user requirements by controlling dialogue flow according to dialogue scenarios. We also improved user satisfaction by introducing motion and speech control based on system utterances and user situations. In the preliminary round, our system was ranked fifth in the impression evaluation and sixth in the plan evaluation among all 12 teams.

{{</citation>}}


### (57/124) MedBench: A Large-Scale Chinese Benchmark for Evaluating Medical Large Language Models (Yan Cai et al., 2023)

{{<citation>}}

Yan Cai, Linlin Wang, Ye Wang, Gerard de Melo, Ya Zhang, Yanfeng Wang, Liang He. (2023)  
**MedBench: A Large-Scale Chinese Benchmark for Evaluating Medical Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.12806v1)  

---


**ABSTRACT**  
The emergence of various medical large language models (LLMs) in the medical domain has highlighted the need for unified evaluation standards, as manual evaluation of LLMs proves to be time-consuming and labor-intensive. To address this issue, we introduce MedBench, a comprehensive benchmark for the Chinese medical domain, comprising 40,041 questions sourced from authentic examination exercises and medical reports of diverse branches of medicine. In particular, this benchmark is composed of four key components: the Chinese Medical Licensing Examination, the Resident Standardization Training Examination, the Doctor In-Charge Qualification Examination, and real-world clinic cases encompassing examinations, diagnoses, and treatments. MedBench replicates the educational progression and clinical practice experiences of doctors in Mainland China, thereby establishing itself as a credible benchmark for assessing the mastery of knowledge and reasoning abilities in medical language learning models. We perform extensive experiments and conduct an in-depth analysis from diverse perspectives, which culminate in the following findings: (1) Chinese medical LLMs underperform on this benchmark, highlighting the need for significant advances in clinical knowledge and diagnostic precision. (2) Several general-domain LLMs surprisingly possess considerable medical knowledge. These findings elucidate both the capabilities and limitations of LLMs within the context of MedBench, with the ultimate goal of aiding the medical research community.

{{</citation>}}


### (58/124) Fine-tuning Large Language Models for Adaptive Machine Translation (Yasmin Moslem et al., 2023)

{{<citation>}}

Yasmin Moslem, Rejwanul Haque, Andy Way. (2023)  
**Fine-tuning Large Language Models for Adaptive Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2312.12740v1)  

---


**ABSTRACT**  
This paper presents the outcomes of fine-tuning Mistral 7B, a general-purpose large language model (LLM), for adaptive machine translation (MT). The fine-tuning process involves utilising a combination of zero-shot and one-shot translation prompts within the medical domain. The primary objective is to enhance real-time adaptive MT capabilities of Mistral 7B, enabling it to adapt translations to the required domain at inference time. The results, particularly for Spanish-to-English MT, showcase the efficacy of the fine-tuned model, demonstrating quality improvements in both zero-shot and one-shot translation scenarios, surpassing Mistral 7B's baseline performance. Notably, the fine-tuned Mistral outperforms ChatGPT "gpt-3.5-turbo" in zero-shot translation while achieving comparable one-shot translation quality. Moreover, the zero-shot translation of the fine-tuned Mistral matches NLLB 3.3B's performance, and its one-shot translation quality surpasses that of NLLB 3.3B. These findings emphasise the significance of fine-tuning efficient LLMs like Mistral 7B to yield high-quality zero-shot translations comparable to task-oriented models like NLLB 3.3B. Additionally, the adaptive gains achieved in one-shot translation are comparable to those of commercial LLMs such as ChatGPT. Our experiments demonstrate that, with a relatively small dataset of 20,000 segments that incorporate a mix of zero-shot and one-shot prompts, fine-tuning significantly enhances Mistral's in-context learning ability, especially for real-time adaptive MT.

{{</citation>}}


### (59/124) Learning and Forgetting Unsafe Examples in Large Language Models (Jiachen Zhao et al., 2023)

{{<citation>}}

Jiachen Zhao, Zhun Deng, David Madras, James Zou, Mengye Ren. (2023)  
**Learning and Forgetting Unsafe Examples in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.12736v1)  

---


**ABSTRACT**  
As the number of large language models (LLMs) released to the public grows, there is a pressing need to understand the safety implications associated with these models learning from third-party custom finetuning data. We explore the behavior of LLMs finetuned on noisy custom data containing unsafe content, represented by datasets that contain biases, toxicity, and harmfulness, finding that while aligned LLMs can readily learn this unsafe content, they also tend to forget it more significantly than other examples when subsequently finetuned on safer content. Drawing inspiration from the discrepancies in forgetting, we introduce the "ForgetFilter" algorithm, which filters unsafe data based on how strong the model's forgetting signal is for that data. We demonstrate that the ForgetFilter algorithm ensures safety in customized finetuning without compromising downstream task performance, unlike sequential safety finetuning. ForgetFilter outperforms alternative strategies like replay and moral self-correction in curbing LLMs' ability to assimilate unsafe content during custom finetuning, e.g. 75% lower than not applying any safety measures and 62% lower than using self-correction in toxicity score.

{{</citation>}}


### (60/124) Response Enhanced Semi-Supervised Dialogue Query Generation (Jianheng Huang et al., 2023)

{{<citation>}}

Jianheng Huang, Ante Wang, Linfeng Gao, Linfeng Song, Jinsong Su. (2023)  
**Response Enhanced Semi-Supervised Dialogue Query Generation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, Dialogue, GPT, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2312.12713v1)  

---


**ABSTRACT**  
Leveraging vast and continually updated knowledge from the Internet has been considered an important ability for a dialogue system. Therefore, the dialogue query generation task is proposed for generating search queries from dialogue histories, which will be submitted to a search engine for retrieving relevant websites on the Internet. In this regard, previous efforts were devoted to collecting conversations with annotated queries and training a query producer (QP) via standard supervised learning. However, these studies still face the challenges of data scarcity and domain adaptation. To address these issues, in this paper, we propose a semi-supervised learning framework -- SemiDQG, to improve model performance with unlabeled conversations. Based on the observation that the search query is typically related to the topic of dialogue response, we train a response-augmented query producer (RA) to provide rich and effective training signals for QP. We first apply a similarity-based query selection strategy to select high-quality RA-generated pseudo queries, which are used to construct pseudo instances for training QP and RA. Then, we adopt the REINFORCE algorithm to further enhance QP, with RA-provided rewards as fine-grained training signals. Experimental results and in-depth analysis of three benchmarks show the effectiveness of our framework in cross-domain and low-resource scenarios. Particularly, SemiDQG significantly surpasses ChatGPT and competitive baselines. Our code is available at \url{https://github.com/DeepLearnXMU/SemiDQG}.

{{</citation>}}


### (61/124) Turning English-centric LLMs Into Polyglots: How Much Multilinguality Is Needed? (Tannon Kew et al., 2023)

{{<citation>}}

Tannon Kew, Florian Schottmann, Rico Sennrich. (2023)  
**Turning English-centric LLMs Into Polyglots: How Much Multilinguality Is Needed?**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual, NLP  
[Paper Link](http://arxiv.org/abs/2312.12683v1)  

---


**ABSTRACT**  
The vast majority of today's large language models are English-centric, having been pretrained predominantly on English text. Yet, in order to meet user expectations, models need to be able to respond appropriately in multiple languages once deployed in downstream applications. Given limited exposure to other languages during pretraining, cross-lingual transfer is important for achieving decent performance in non-English settings. In this work, we investigate just how much multilinguality is required during finetuning to elicit strong cross-lingual generalisation across a range of tasks and target languages. We find that, compared to English-only finetuning, multilingual instruction tuning with as few as three languages significantly improves a model's cross-lingual transfer abilities on generative tasks that assume input/output language agreement, while being of less importance for highly structured tasks. Our code and data is available at https://github.com/ZurichNLP/multilingual-instruction-tuning.

{{</citation>}}


### (62/124) Mini-GPTs: Efficient Large Language Models through Contextual Pruning (Tim Valicenti et al., 2023)

{{<citation>}}

Tim Valicenti, Justice Vidal, Ritik Patnaik. (2023)  
**Mini-GPTs: Efficient Large Language Models through Contextual Pruning**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: AI, GPT, Language Model, Pruning  
[Paper Link](http://arxiv.org/abs/2312.12682v1)  

---


**ABSTRACT**  
In AI research, the optimization of Large Language Models (LLMs) remains a significant challenge, crucial for advancing the field's practical applications and sustainability. Building upon the foundational work of Professor Song Han's lab at MIT, this paper introduces a novel approach in developing Mini-GPTs via contextual pruning. Our methodology strategically prunes the computational architecture of traditional LLMs, like Phi-1.5, focusing on retaining core functionalities while drastically reducing model sizes. We employ the technique across diverse and complex datasets, including US law, Medical Q&A, Skyrim dialogue, English-Taiwanese translation, and Economics articles. The results underscore the efficiency and effectiveness of contextual pruning, not merely as a theoretical concept but as a practical tool in developing domain-specific, resource-efficient LLMs. Contextual pruning is a promising method for building domain-specific LLMs, and this research is a building block towards future development with more hardware compute, refined fine-tuning, and quantization.

{{</citation>}}


## cs.RO (5)



### (63/124) ORBSLAM3-Enhanced Autonomous Toy Drones: Pioneering Indoor Exploration (Murad Tukan et al., 2023)

{{<citation>}}

Murad Tukan, Fares Fares, Yotam Grufinkle, Ido Talmor, Loay Mualem, Vladimir Braverman, Dan Feldman. (2023)  
**ORBSLAM3-Enhanced Autonomous Toy Drones: Pioneering Indoor Exploration**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2312.13385v1)  

---


**ABSTRACT**  
Navigating toy drones through uncharted GPS-denied indoor spaces poses significant difficulties due to their reliance on GPS for location determination. In such circumstances, the necessity for achieving proper navigation is a primary concern. In response to this formidable challenge, we introduce a real-time autonomous indoor exploration system tailored for drones equipped with a monocular \emph{RGB} camera.   Our system utilizes \emph{ORB-SLAM3}, a state-of-the-art vision feature-based SLAM, to handle both the localization of toy drones and the mapping of unmapped indoor terrains. Aside from the practicability of \emph{ORB-SLAM3}, the generated maps are represented as sparse point clouds, making them prone to the presence of outlier data. To address this challenge, we propose an outlier removal algorithm with provable guarantees. Furthermore, our system incorporates a novel exit detection algorithm, ensuring continuous exploration by the toy drone throughout the unfamiliar indoor environment. We also transform the sparse point to ensure proper path planning using existing path planners.   To validate the efficacy and efficiency of our proposed system, we conducted offline and real-time experiments on the autonomous exploration of indoor spaces. The results from these endeavors demonstrate the effectiveness of our methods.

{{</citation>}}


### (64/124) Interactive Visual Task Learning for Robots (Weiwei Gu et al., 2023)

{{<citation>}}

Weiwei Gu, Anant Sah, Nakul Gopalan. (2023)  
**Interactive Visual Task Learning for Robots**  

---
Primary Category: cs.RO  
Categories: cs-CL, cs-CV, cs-RO, cs.RO  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2312.13219v1)  

---


**ABSTRACT**  
We present a framework for robots to learn novel visual concepts and tasks via in-situ linguistic interactions with human users. Previous approaches have either used large pre-trained visual models to infer novel objects zero-shot, or added novel concepts along with their attributes and representations to a concept hierarchy. We extend the approaches that focus on learning visual concept hierarchies by enabling them to learn novel concepts and solve unseen robotics tasks with them. To enable a visual concept learner to solve robotics tasks one-shot, we developed two distinct techniques. Firstly, we propose a novel approach, Hi-Viscont(HIerarchical VISual CONcept learner for Task), which augments information of a novel concept to its parent nodes within a concept hierarchy. This information propagation allows all concepts in a hierarchy to update as novel concepts are taught in a continual learning setting. Secondly, we represent a visual task as a scene graph with language annotations, allowing us to create novel permutations of a demonstrated task zero-shot in-situ. We present two sets of results. Firstly, we compare Hi-Viscont with the baseline model (FALCON) on visual question answering(VQA) in three domains. While being comparable to the baseline model on leaf level concepts, Hi-Viscont achieves an improvement of over 9% on non-leaf concepts on average. We compare our model's performance against the baseline FALCON model. Our framework achieves 33% improvements in success rate metric, and 19% improvements in the object level accuracy compared to the baseline model. With both of these results we demonstrate the ability of our model to learn tasks and concepts in a continual learning setting on the robot.

{{</citation>}}


### (65/124) Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation (Hongtao Wu et al., 2023)

{{<citation>}}

Hongtao Wu, Ya Jing, Chilam Cheang, Guangzeng Chen, Jiafeng Xu, Xinghang Li, Minghuan Liu, Hang Li, Tao Kong. (2023)  
**Unleashing Large-Scale Video Generative Pre-training for Visual Robot Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2312.13139v2)  

---


**ABSTRACT**  
Generative pre-trained models have demonstrated remarkable effectiveness in language and vision domains by learning useful representations. In this paper, we extend the scope of this effectiveness by showing that visual robot manipulation can significantly benefit from large-scale video generative pre-training. We introduce GR-1, a straightforward GPT-style model designed for multi-task language-conditioned visual robot manipulation. GR-1 takes as inputs a language instruction, a sequence of observation images, and a sequence of robot states. It predicts robot actions as well as future images in an end-to-end manner. Thanks to a flexible design, GR-1 can be seamlessly finetuned on robot data after pre-trained on a large-scale video dataset. We perform extensive experiments on the challenging CALVIN benchmark and a real robot. On CALVIN benchmark, our method outperforms state-of-the-art baseline methods and improves the success rate from 88.9% to 94.9%. In the setting of zero-shot unseen scene generalization, GR-1 improves the success rate from 53.3% to 85.4%. In real robot experiments, GR-1 also outperforms baseline methods and shows strong potentials in generalization to unseen scenes and objects. We provide inaugural evidence that a unified GPT-style transformer, augmented with large-scale video generative pre-training, exhibits remarkable generalization to multi-task visual robot manipulation. Project page: https://GR1-Manipulation.github.io

{{</citation>}}


### (66/124) Multi-sensory Anti-collision Design for Autonomous Nano-swarm Exploration (Mahyar Pourjabar et al., 2023)

{{<citation>}}

Mahyar Pourjabar, Manuele Rusci, Luca Bompani, Lorenzo Lamberti, Vlad Niculescu, Daniele Palossi, Luca Benini. (2023)  
**Multi-sensory Anti-collision Design for Autonomous Nano-swarm Exploration**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2312.13086v1)  

---


**ABSTRACT**  
This work presents a multi-sensory anti-collision system design to achieve robust autonomous exploration capabilities for a swarm of 10 cm-side nano-drones operating on object detection missions. We combine lightweight single-beam laser ranging to avoid proximity collisions with a long-range vision-based obstacle avoidance deep learning model (i.e., PULP-Dronet) and an ultra-wide-band (UWB) based ranging module to prevent intra-swarm collisions. An in-field study shows that our multisensory approach can prevent collisions with static obstacles, improving the mission success rate from 20% to 80% in cluttered environments w.r.t. a State-of-the-Art (SoA) baseline. At the same time, the UWB-based sub-system shows a 92.8% success rate in preventing collisions between drones of a four-agent fleet within a safety distance of 65 cm. On a SoA robotic platform extended by a GAP8 multi-core processor, the PULP-Dronet runs interleaved with an objected detection task, which constraints its execution at 1.6 frame/s. This throughput is sufficient for avoiding obstacles with a probability of about 40% but shows a need for more capable processors for the next-generation nano-drone swarms.

{{</citation>}}


### (67/124) Safe Multi-Agent Reinforcement Learning for Formation Control without Individual Reference Targets (Murad Dawood et al., 2023)

{{<citation>}}

Murad Dawood, Sicong Pan, Nils Dengler, Siqi Zhou, Angela P. Schoellig, Maren Bennewitz. (2023)  
**Safe Multi-Agent Reinforcement Learning for Formation Control without Individual Reference Targets**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.12861v1)  

---


**ABSTRACT**  
In recent years, formation control of unmanned vehicles has received considerable interest, driven by the progress in autonomous systems and the imperative for multiple vehicles to carry out diverse missions. In this paper, we address the problem of behavior-based formation control of mobile robots, where we use safe multi-agent reinforcement learning~(MARL) to ensure the safety of the robots by eliminating all collisions during training and execution. To ensure safety, we implemented distributed model predictive control safety filters to override unsafe actions. We focus on achieving behavior-based formation without having individual reference targets for the robots, and instead use targets for the centroid of the formation. This formulation facilitates the deployment of formation control on real robots and improves the scalability of our approach to more robots. The task cannot be addressed through optimization-based controllers without specific individual reference targets for the robots and information about the relative locations of each robot to the others. That is why, for our formulation we use MARL to train the robots. Moreover, in order to account for the interactions between the agents, we use attention-based critics to improve the training process. We train the agents in simulation and later on demonstrate the resulting behavior of our approach on real Turtlebot robots. We show that despite the agents having very limited information, we can still safely achieve the desired behavior.

{{</citation>}}


## cs.LG (21)



### (68/124) Transparency and Privacy: The Role of Explainable AI and Federated Learning in Financial Fraud Detection (Tomisin Awosika et al., 2023)

{{<citation>}}

Tomisin Awosika, Raj Mani Shukla, Bernardi Pranggono. (2023)  
**Transparency and Privacy: The Role of Explainable AI and Federated Learning in Financial Fraud Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: AI, Financial, Fraud Detection  
[Paper Link](http://arxiv.org/abs/2312.13334v1)  

---


**ABSTRACT**  
Fraudulent transactions and how to detect them remain a significant problem for financial institutions around the world. The need for advanced fraud detection systems to safeguard assets and maintain customer trust is paramount for financial institutions, but some factors make the development of effective and efficient fraud detection systems a challenge. One of such factors is the fact that fraudulent transactions are rare and that many transaction datasets are imbalanced; that is, there are fewer significant samples of fraudulent transactions than legitimate ones. This data imbalance can affect the performance or reliability of the fraud detection model. Moreover, due to the data privacy laws that all financial institutions are subject to follow, sharing customer data to facilitate a higher-performing centralized model is impossible. Furthermore, the fraud detection technique should be transparent so that it does not affect the user experience. Hence, this research introduces a novel approach using Federated Learning (FL) and Explainable AI (XAI) to address these challenges. FL enables financial institutions to collaboratively train a model to detect fraudulent transactions without directly sharing customer data, thereby preserving data privacy and confidentiality. Meanwhile, the integration of XAI ensures that the predictions made by the model can be understood and interpreted by human experts, adding a layer of transparency and trust to the system. Experimental results, based on realistic transaction datasets, reveal that the FL-based fraud detection system consistently demonstrates high performance metrics. This study grounds FL's potential as an effective and privacy-preserving tool in the fight against fraud.

{{</citation>}}


### (69/124) Enhancing Neural Training via a Correlated Dynamics Model (Jonathan Brokman et al., 2023)

{{<citation>}}

Jonathan Brokman, Roy Betser, Rotem Turjeman, Tom Berkov, Ido Cohen, Guy Gilboa. (2023)  
**Enhancing Neural Training via a Correlated Dynamics Model**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-DS  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.13247v1)  

---


**ABSTRACT**  
As neural networks grow in scale, their training becomes both computationally demanding and rich in dynamics. Amidst the flourishing interest in these training dynamics, we present a novel observation: Parameters during training exhibit intrinsic correlations over time. Capitalizing on this, we introduce Correlation Mode Decomposition (CMD). This algorithm clusters the parameter space into groups, termed modes, that display synchronized behavior across epochs. This enables CMD to efficiently represent the training dynamics of complex networks, like ResNets and Transformers, using only a few modes. Moreover, test set generalization is enhanced. We introduce an efficient CMD variant, designed to run concurrently with training. Our experiments indicate that CMD surpasses the state-of-the-art method for compactly modeled dynamics on image classification. Our modeling can improve training efficiency and lower communication overhead, as shown by our preliminary experiments in the context of federated learning.

{{</citation>}}


### (70/124) Diffusion Models With Learned Adaptive Noise (Subham Sekhar Sahoo et al., 2023)

{{<citation>}}

Subham Sekhar Sahoo, Aaron Gokaslan, Chris De Sa, Volodymyr Kuleshov. (2023)  
**Diffusion Models With Learned Adaptive Noise**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.13236v1)  

---


**ABSTRACT**  
Diffusion models have gained traction as powerful algorithms for synthesizing high-quality images. Central to these algorithms is the diffusion process, which maps data to noise according to equations inspired by thermodynamics and can significantly impact performance. A widely held assumption is that the ELBO objective of a diffusion model is invariant to the noise process (Kingma et al.,2021). In this work, we dispel this assumption -- we propose multivariate learned adaptive noise (MuLAN), a learned diffusion process that applies Gaussian noise at different rates across an image. Our method consists of three components -- a multivariate noise schedule, instance-conditional diffusion, and auxiliary variables -- which ensure that the learning objective is no longer invariant to the choice of the noise schedule as in previous works. Our work is grounded in Bayesian inference and casts the learned diffusion process as an approximate variational posterior that yields a tighter lower bound on marginal likelihood. Empirically, MuLAN sets a new state-of-the-art in density estimation on CIFAR-10 and ImageNet compared to classical diffusion. Code is available at https://github.com/s-sahoo/MuLAN

{{</citation>}}


### (71/124) FiFAR: A Fraud Detection Dataset for Learning to Defer (Jean V. Alves et al., 2023)

{{<citation>}}

Jean V. Alves, Diogo Leitão, Sérgio Jesus, Marco O. P. Sampaio, Pedro Saleiro, Mário A. T. Figueiredo, Pedro Bizarro. (2023)  
**FiFAR: A Fraud Detection Dataset for Learning to Defer**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Financial, Fraud Detection  
[Paper Link](http://arxiv.org/abs/2312.13218v1)  

---


**ABSTRACT**  
Public dataset limitations have significantly hindered the development and benchmarking of learning to defer (L2D) algorithms, which aim to optimally combine human and AI capabilities in hybrid decision-making systems. In such systems, human availability and domain-specific concerns introduce difficulties, while obtaining human predictions for training and evaluation is costly. Financial fraud detection is a high-stakes setting where algorithms and human experts often work in tandem; however, there are no publicly available datasets for L2D concerning this important application of human-AI teaming. To fill this gap in L2D research, we introduce the Financial Fraud Alert Review Dataset (FiFAR), a synthetic bank account fraud detection dataset, containing the predictions of a team of 50 highly complex and varied synthetic fraud analysts, with varied bias and feature dependence. We also provide a realistic definition of human work capacity constraints, an aspect of L2D systems that is often overlooked, allowing for extensive testing of assignment systems under real-world conditions. We use our dataset to develop a capacity-aware L2D method and rejection learning approach under realistic data availability conditions, and benchmark these baselines under an array of 300 distinct testing scenarios. We believe that this dataset will serve as a pivotal instrument in facilitating a systematic, rigorous, reproducible, and transparent evaluation and comparison of L2D methods, thereby fostering the development of more synergistic human-AI collaboration in decision-making systems. The public dataset and detailed synthetic expert information are available at: https://github.com/feedzai/fifar-dataset

{{</citation>}}


### (72/124) In-Context Reinforcement Learning for Variable Action Spaces (Viacheslav Sinii et al., 2023)

{{<citation>}}

Viacheslav Sinii, Alexander Nikulin, Vladislav Kurenkov, Ilya Zisman, Sergey Kolesnikov. (2023)  
**In-Context Reinforcement Learning for Variable Action Spaces**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.13327v1)  

---


**ABSTRACT**  
Recent work has shown that supervised pre-training on learning histories of RL algorithms results in a model that captures the learning process and is able to improve in-context on novel tasks through interactions with an environment. Despite the progress in this area, there is still a gap in the existing literature, particularly in the in-context generalization to new action spaces. While existing methods show high performance on new tasks created by different reward distributions, their architectural design and training process are not suited for the introduction of new actions during evaluation. We aim to bridge this gap by developing an architecture and training methodology specifically for the task of generalizing to new action spaces. Inspired by Headless LLM, we remove the dependence on the number of actions by directly predicting the action embeddings. Furthermore, we use random embeddings to force the semantic inference of actions from context and to prepare for the new unseen embeddings during test time. Using multi-armed bandit environments with a variable number of arms, we show that our model achieves the performance of the data generation algorithm without requiring retraining for each new environment.

{{</citation>}}


### (73/124) LRS: Enhancing Adversarial Transferability through Lipschitz Regularized Surrogate (Tao Wu et al., 2023)

{{<citation>}}

Tao Wu, Tie Luo, Donald C. Wunsch. (2023)  
**LRS: Enhancing Adversarial Transferability through Lipschitz Regularized Surrogate**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.13118v1)  

---


**ABSTRACT**  
The transferability of adversarial examples is of central importance to transfer-based black-box adversarial attacks. Previous works for generating transferable adversarial examples focus on attacking \emph{given} pretrained surrogate models while the connections between surrogate models and adversarial trasferability have been overlooked. In this paper, we propose {\em Lipschitz Regularized Surrogate} (LRS) for transfer-based black-box attacks, a novel approach that transforms surrogate models towards favorable adversarial transferability. Using such transformed surrogate models, any existing transfer-based black-box attack can run without any change, yet achieving much better performance. Specifically, we impose Lipschitz regularization on the loss landscape of surrogate models to enable a smoother and more controlled optimization process for generating more transferable adversarial examples. In addition, this paper also sheds light on the connection between the inner properties of surrogate models and adversarial transferability, where three factors are identified: smaller local Lipschitz constant, smoother loss landscape, and stronger adversarial robustness. We evaluate our proposed LRS approach by attacking state-of-the-art standard deep neural networks and defense models. The results demonstrate significant improvement on the attack success rates and transferability. Our code is available at https://github.com/TrustAIoT/LRS.

{{</citation>}}


### (74/124) Pre-training of Molecular GNNs as Conditional Boltzmann Generator (Daiki Koge et al., 2023)

{{<citation>}}

Daiki Koge, Naoaki Ono, Shigehiko Kanaya. (2023)  
**Pre-training of Molecular GNNs as Conditional Boltzmann Generator**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, physics-chem-ph, q-bio-BM  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.13110v1)  

---


**ABSTRACT**  
Learning representations of molecular structures using deep learning is a fundamental problem in molecular property prediction tasks. Molecules inherently exist in the real world as three-dimensional structures; furthermore, they are not static but in continuous motion in the 3D Euclidean space, forming a potential energy surface. Therefore, it is desirable to generate multiple conformations in advance and extract molecular representations using a 4D-QSAR model that incorporates multiple conformations. However, this approach is impractical for drug and material discovery tasks because of the computational cost of obtaining multiple conformations. To address this issue, we propose a pre-training method for molecular GNNs using an existing dataset of molecular conformations to generate a latent vector universal to multiple conformations from a 2D molecular graph. Our method, called Boltzmann GNN, is formulated by maximizing the conditional marginal likelihood of a conditional generative model for conformations generation. We show that our model has a better prediction performance for molecular properties than existing pre-training methods using molecular graphs and three-dimensional molecular structures.

{{</citation>}}


### (75/124) AutoXPCR: Automated Multi-Objective Model Selection for Time Series Forecasting (Raphael Fischer et al., 2023)

{{<citation>}}

Raphael Fischer, Amal Saadallah. (2023)  
**AutoXPCR: Automated Multi-Objective Model Selection for Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.13038v1)  

---


**ABSTRACT**  
Automated machine learning (AutoML) streamlines the creation of ML models. While most methods select the "best" model based on predictive quality, it's crucial to acknowledge other aspects, such as interpretability and resource consumption. This holds particular importance in the context of deep neural networks (DNNs), as these models are often perceived as computationally intensive black boxes. In the challenging domain of time series forecasting, DNNs achieve stunning results, but specialized approaches for automatically selecting models are scarce. In this paper, we propose AutoXPCR - a novel method for automated and explainable multi-objective model selection. Our approach leverages meta-learning to estimate any model's performance along PCR criteria, which encompass (P)redictive error, (C)omplexity, and (R)esource demand. Explainability is addressed on multiple levels, as our interactive framework can prioritize less complex models and provide by-product explanations of recommendations. We demonstrate practical feasibility by deploying AutoXPCR on over 1000 configurations across 114 data sets from various domains. Our method clearly outperforms other model selection approaches - on average, it only requires 20% of computation costs for recommending models with 90% of the best-possible quality.

{{</citation>}}


### (76/124) NodeMixup: Tackling Under-Reaching for Graph Neural Networks (Weigang Lu et al., 2023)

{{<citation>}}

Weigang Lu, Ziyu Guan, Wei Zhao, Yaming Yang, Long Jin. (2023)  
**NodeMixup: Tackling Under-Reaching for Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.13032v2)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have become mainstream methods for solving the semi-supervised node classification problem. However, due to the uneven location distribution of labeled nodes in the graph, labeled nodes are only accessible to a small portion of unlabeled nodes, leading to the \emph{under-reaching} issue. In this study, we firstly reveal under-reaching by conducting an empirical investigation on various well-known graphs. Then, we demonstrate that under-reaching results in unsatisfactory distribution alignment between labeled and unlabeled nodes through systematic experimental analysis, significantly degrading GNNs' performance. To tackle under-reaching for GNNs, we propose an architecture-agnostic method dubbed NodeMixup. The fundamental idea is to (1) increase the reachability of labeled nodes by labeled-unlabeled pairs mixup, (2) leverage graph structures via fusing the neighbor connections of intra-class node pairs to improve performance gains of mixup, and (3) use neighbor label distribution similarity incorporating node degrees to determine sampling weights for node mixup. Extensive experiments demonstrate the efficacy of NodeMixup in assisting GNNs in handling under-reaching. The source code is available at \url{https://github.com/WeigangLu/NodeMixup}.

{{</citation>}}


### (77/124) Benchmarking and Analyzing In-context Learning, Fine-tuning and Supervised Learning for Biomedical Knowledge Curation: a focused study on chemical entities of biological interest (Emily Groves et al., 2023)

{{<citation>}}

Emily Groves, Minhong Wang, Yusuf Abdulle, Holger Kunz, Jason Hoelscher-Obermaier, Ronin Wu, Honghan Wu. (2023)  
**Benchmarking and Analyzing In-context Learning, Fine-tuning and Supervised Learning for Biomedical Knowledge Curation: a focused study on chemical entities of biological interest**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG, q-bio-QM  
Keywords: BERT, GPT, GPT-3.5, GPT-4, NLP  
[Paper Link](http://arxiv.org/abs/2312.12989v1)  

---


**ABSTRACT**  
Automated knowledge curation for biomedical ontologies is key to ensure that they remain comprehensive, high-quality and up-to-date. In the era of foundational language models, this study compares and analyzes three NLP paradigms for curation tasks: in-context learning (ICL), fine-tuning (FT), and supervised learning (ML). Using the Chemical Entities of Biological Interest (ChEBI) database as a model ontology, three curation tasks were devised. For ICL, three prompting strategies were employed with GPT-4, GPT-3.5, BioGPT. PubmedBERT was chosen for the FT paradigm. For ML, six embedding models were utilized for training Random Forest and Long-Short Term Memory models. Five setups were designed to assess ML and FT model performance across different data availability scenarios.Datasets for curation tasks included: task 1 (620,386), task 2 (611,430), and task 3 (617,381), maintaining a 50:50 positive versus negative ratio. For ICL models, GPT-4 achieved best accuracy scores of 0.916, 0.766 and 0.874 for tasks 1-3 respectively. In a direct comparison, ML (trained on ~260,000 triples) outperformed ICL in accuracy across all tasks. (accuracy differences: +.11, +.22 and +.17). Fine-tuned PubmedBERT performed similarly to leading ML models in tasks 1 & 2 (F1 differences: -.014 and +.002), but worse in task 3 (-.048). Simulations revealed performance declines in both ML and FT models with smaller and higher imbalanced training data. where ICL (particularly GPT-4) excelled in tasks 1 & 3. GPT-4 excelled in tasks 1 and 3 with less than 6,000 triples, surpassing ML/FT. ICL underperformed ML/FT in task 2.ICL-augmented foundation models can be good assistants for knowledge curation with correct prompting, however, not making ML and FT paradigms obsolete. The latter two require task-specific data to beat ICL. In such cases, ML relies on small pretrained embeddings, minimizing computational demands.

{{</citation>}}


### (78/124) Class Conditional Time Series Generation with Structured Noise Space GAN (Hamidreza Gholamrezaei et al., 2023)

{{<citation>}}

Hamidreza Gholamrezaei, Alireza Koochali, Andreas Dengel, Sheraz Ahmed. (2023)  
**Class Conditional Time Series Generation with Structured Noise Space GAN**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.12946v1)  

---


**ABSTRACT**  
This paper introduces Structured Noise Space GAN (SNS-GAN), a novel approach in the field of generative modeling specifically tailored for class-conditional generation in both image and time series data. It addresses the challenge of effectively integrating class labels into generative models without requiring structural modifications to the network. The SNS-GAN method embeds class conditions within the generator's noise space, simplifying the training process and enhancing model versatility. The model's efficacy is demonstrated through qualitative validations in the image domain and superior performance in time series generation compared to baseline models. This research opens new avenues for the application of GANs in various domains, including but not limited to time series and image data generation.

{{</citation>}}


### (79/124) Rule-Extraction Methods From Feedforward Neural Networks: A Systematic Literature Review (Sara El Mekkaoui et al., 2023)

{{<citation>}}

Sara El Mekkaoui, Loubna Benabbou, Abdelaziz Berrado. (2023)  
**Rule-Extraction Methods From Feedforward Neural Networks: A Systematic Literature Review**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12878v1)  

---


**ABSTRACT**  
Motivated by the interpretability question in ML models as a crucial element for the successful deployment of AI systems, this paper focuses on rule extraction as a means for neural networks interpretability. Through a systematic literature review, different approaches for extracting rules from feedforward neural networks, an important block in deep learning models, are identified and explored. The findings reveal a range of methods developed for over two decades, mostly suitable for shallow neural networks, with recent developments to meet deep learning models' challenges. Rules offer a transparent and intuitive means of explaining neural networks, making this study a comprehensive introduction for researchers interested in the field. While the study specifically addresses feedforward networks with supervised learning and crisp rules, future work can extend to other network types, machine learning methods, and fuzzy rule extraction.

{{</citation>}}


### (80/124) Causal Discovery under Identifiable Heteroscedastic Noise Model (Naiyu Yin et al., 2023)

{{<citation>}}

Naiyu Yin, Tian Gao, Yue Yu, Qiang Ji. (2023)  
**Causal Discovery under Identifiable Heteroscedastic Noise Model**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ME  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12844v1)  

---


**ABSTRACT**  
Capturing the underlying structural causal relations represented by Directed Acyclic Graphs (DAGs) has been a fundamental task in various AI disciplines. Causal DAG learning via the continuous optimization framework has recently achieved promising performance in terms of both accuracy and efficiency. However, most methods make strong assumptions of homoscedastic noise, i.e., exogenous noises have equal variances across variables, observations, or even both. The noises in real data usually violate both assumptions due to the biases introduced by different data collection processes. To address the issue of heteroscedastic noise, we introduce relaxed and implementable sufficient conditions, proving the identifiability of a general class of SEM subject to these conditions. Based on the identifiable general SEM, we propose a novel formulation for DAG learning that accounts for the variation in noise variance across variables and observations. We then propose an effective two-phase iterative DAG learning algorithm to address the increasing optimization difficulties and to learn a causal DAG from data with heteroscedastic variable noise under varying variance. We show significant empirical gains of the proposed approaches over state-of-the-art methods on both synthetic data and real data.

{{</citation>}}


### (81/124) FedA3I: Annotation Quality-Aware Aggregation for Federated Medical Image Segmentation Against Heterogeneous Annotation Noise (Nannan Wu et al., 2023)

{{<citation>}}

Nannan Wu, Zhaobin Sun, Zengqiang Yan, Li Yu. (2023)  
**FedA3I: Annotation Quality-Aware Aggregation for Federated Medical Image Segmentation Against Heterogeneous Annotation Noise**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12838v1)  

---


**ABSTRACT**  
Federated learning (FL) has emerged as a promising paradigm for training segmentation models on decentralized medical data, owing to its privacy-preserving property. However, existing research overlooks the prevalent annotation noise encountered in real-world medical datasets, which limits the performance ceilings of FL. In this paper, we, for the first time, identify and tackle this problem. For problem formulation, we propose a contour evolution for modeling non-independent and identically distributed (Non-IID) noise across pixels within each client and then extend it to the case of multi-source data to form a heterogeneous noise model (\textit{i.e.}, Non-IID annotation noise across clients). For robust learning from annotations with such two-level Non-IID noise, we emphasize the importance of data quality in model aggregation, allowing high-quality clients to have a greater impact on FL. To achieve this, we propose \textbf{Fed}erated learning with \textbf{A}nnotation qu\textbf{A}lity-aware \textbf{A}ggregat\textbf{I}on, named \textbf{FedA$^3$I}, by introducing a quality factor based on client-wise noise estimation. Specifically, noise estimation at each client is accomplished through the Gaussian mixture model and then incorporated into model aggregation in a layer-wise manner to up-weight high-quality clients. Extensive experiments on two real-world medical image segmentation datasets demonstrate the superior performance of FedA$^3$I against the state-of-the-art approaches in dealing with cross-client annotation noise. The code is available at \color{blue}{https://github.com/wnn2000/FedAAAI}.

{{</citation>}}


### (82/124) Near-Optimal Resilient Aggregation Rules for Distributed Learning Using 1-Center and 1-Mean Clustering with Outliers (Yuhao Yi et al., 2023)

{{<citation>}}

Yuhao Yi, Ronghui You, Hong Liu, Changxin Liu, Yuan Wang, Jiancheng Lv. (2023)  
**Near-Optimal Resilient Aggregation Rules for Distributed Learning Using 1-Center and 1-Mean Clustering with Outliers**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12835v1)  

---


**ABSTRACT**  
Byzantine machine learning has garnered considerable attention in light of the unpredictable faults that can occur in large-scale distributed learning systems. The key to secure resilience against Byzantine machines in distributed learning is resilient aggregation mechanisms. Although abundant resilient aggregation rules have been proposed, they are designed in ad-hoc manners, imposing extra barriers on comparing, analyzing, and improving the rules across performance criteria. This paper studies near-optimal aggregation rules using clustering in the presence of outliers. Our outlier-robust clustering approach utilizes geometric properties of the update vectors provided by workers. Our analysis show that constant approximations to the 1-center and 1-mean clustering problems with outliers provide near-optimal resilient aggregators for metric-based criteria, which have been proven to be crucial in the homogeneous and heterogeneous cases respectively. In addition, we discuss two contradicting types of attacks under which no single aggregation rule is guaranteed to improve upon the naive average. Based on the discussion, we propose a two-phase resilient aggregation framework. We run experiments for image classification using a non-convex loss function. The proposed algorithms outperform previously known aggregation rules by a large margin with both homogeneous and heterogeneous data distributions among non-faulty workers. Code and appendix are available at https://github.com/jerry907/AAAI24-RASHB.

{{</citation>}}


### (83/124) Unlocking Deep Learning: A BP-Free Approach for Parallel Block-Wise Training of Neural Networks (Anzhe Cheng et al., 2023)

{{<citation>}}

Anzhe Cheng, Zhenkun Wang, Chenzhong Yin, Mingxi Cheng, Heng Ping, Xiongye Xiao, Shahin Nazarian, Paul Bogdan. (2023)  
**Unlocking Deep Learning: A BP-Free Approach for Parallel Block-Wise Training of Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-IV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.13311v1)  

---


**ABSTRACT**  
Backpropagation (BP) has been a successful optimization technique for deep learning models. However, its limitations, such as backward- and update-locking, and its biological implausibility, hinder the concurrent updating of layers and do not mimic the local learning processes observed in the human brain. To address these issues, recent research has suggested using local error signals to asynchronously train network blocks. However, this approach often involves extensive trial-and-error iterations to determine the best configuration for local training. This includes decisions on how to decouple network blocks and which auxiliary networks to use for each block. In our work, we introduce a novel BP-free approach: a block-wise BP-free (BWBPF) neural network that leverages local error signals to optimize distinct sub-neural networks separately, where the global loss is only responsible for updating the output layer. The local error signals used in the BP-free model can be computed in parallel, enabling a potential speed-up in the weight update process through parallel implementation. Our experimental results consistently show that this approach can identify transferable decoupled architectures for VGG and ResNet variations, outperforming models trained with end-to-end backpropagation and other state-of-the-art block-wise learning techniques on datasets such as CIFAR-10 and Tiny-ImageNet. The code is released at https://github.com/Belis0811/BWBPF.

{{</citation>}}


### (84/124) Fast Cell Library Characterization for Design Technology Co-Optimization Based on Graph Neural Networks (Tianliang Ma et al., 2023)

{{<citation>}}

Tianliang Ma, Zhihui Deng, Xuguang Sun, Leilai Shao. (2023)  
**Fast Cell Library Characterization for Design Technology Co-Optimization Based on Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.12784v1)  

---


**ABSTRACT**  
Design technology co-optimization (DTCO) plays a critical role in achieving optimal power, performance, and area (PPA) for advanced semiconductor process development. Cell library characterization is essential in DTCO flow, but traditional methods are time-consuming and costly. To overcome these challenges, we propose a graph neural network (GNN)-based machine learning model for rapid and accurate cell library characterization. Our model incorporates cell structures and demonstrates high prediction accuracy across various process-voltage-temperature (PVT) corners and technology parameters. Validation with 512 unseen technology corners and over one million test data points shows accurate predictions of delay, power, and input pin capacitance for 33 types of cells, with a mean absolute percentage error (MAPE) $\le$ 0.95% and a speed-up of 100X compared with SPICE simulations. Additionally, we investigate system-level metrics such as worst negative slack (WNS), leakage power, and dynamic power using predictions obtained from the GNN-based model on unseen corners. Our model achieves precise predictions, with absolute error $\le$3.0 ps for WNS, percentage errors $\le$0.60% for leakage power, and $\le$0.99% for dynamic power, when compared to golden reference. With the developed model, we further proposed a fine-grained drive strength interpolation methodology to enhance PPA for small-to-medium-scale designs, resulting in an approximate 1-3% improvement.

{{</citation>}}


### (85/124) ALMANACS: A Simulatability Benchmark for Language Model Explainability (Edmund Mills et al., 2023)

{{<citation>}}

Edmund Mills, Shiye Su, Stuart Russell, Scott Emmons. (2023)  
**ALMANACS: A Simulatability Benchmark for Language Model Explainability**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG, stat-ML  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.12747v1)  

---


**ABSTRACT**  
How do we measure the efficacy of language model explainability methods? While many explainability methods have been developed, they are typically evaluated on bespoke tasks, preventing an apples-to-apples comparison. To help fill this gap, we present ALMANACS, a language model explainability benchmark. ALMANACS scores explainability methods on simulatability, i.e., how well the explanations improve behavior prediction on new inputs. The ALMANACS scenarios span twelve safety-relevant topics such as ethical reasoning and advanced AI behaviors; they have idiosyncratic premises to invoke model-specific behavior; and they have a train-test distributional shift to encourage faithful explanations. By using another language model to predict behavior based on the explanations, ALMANACS is a fully automated benchmark. We use ALMANACS to evaluate counterfactuals, rationalizations, attention, and Integrated Gradients explanations. Our results are sobering: when averaged across all topics, no explanation method outperforms the explanation-free control. We conclude that despite modest successes in prior work, developing an explanation method that aids simulatability in ALMANACS remains an open challenge.

{{</citation>}}


### (86/124) Locally Optimal Fixed-Budget Best Arm Identification in Two-Armed Gaussian Bandits with Unknown Variances (Masahiro Kato, 2023)

{{<citation>}}

Masahiro Kato. (2023)  
**Locally Optimal Fixed-Budget Best Arm Identification in Two-Armed Gaussian Bandits with Unknown Variances**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, econ-EM, math-ST, stat-ME, stat-ML, stat-TH  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12741v1)  

---


**ABSTRACT**  
We address the problem of best arm identification (BAI) with a fixed budget for two-armed Gaussian bandits. In BAI, given multiple arms, we aim to find the best arm, an arm with the highest expected reward, through an adaptive experiment. Kaufmann et al. (2016) develops a lower bound for the probability of misidentifying the best arm. They also propose a strategy, assuming that the variances of rewards are known, and show that it is asymptotically optimal in the sense that its probability of misidentification matches the lower bound as the budget approaches infinity. However, an asymptotically optimal strategy is unknown when the variances are unknown. For this open issue, we propose a strategy that estimates variances during an adaptive experiment and draws arms with a ratio of the estimated standard deviations. We refer to this strategy as the Neyman Allocation (NA)-Augmented Inverse Probability weighting (AIPW) strategy. We then demonstrate that this strategy is asymptotically optimal by showing that its probability of misidentification matches the lower bound when the budget approaches infinity, and the gap between the expected rewards of two arms approaches zero (small-gap regime). Our results suggest that under the worst-case scenario characterized by the small-gap regime, our strategy, which employs estimated variance, is asymptotically optimal even when the variances are unknown.

{{</citation>}}


### (87/124) Robustly Improving Bandit Algorithms with Confounded and Selection Biased Offline Data: A Causal Approach (Wen Huang et al., 2023)

{{<citation>}}

Wen Huang, Xintao Wu. (2023)  
**Robustly Improving Bandit Algorithms with Confounded and Selection Biased Offline Data: A Causal Approach**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2312.12731v1)  

---


**ABSTRACT**  
This paper studies bandit problems where an agent has access to offline data that might be utilized to potentially improve the estimation of each arm's reward distribution. A major obstacle in this setting is the existence of compound biases from the observational data. Ignoring these biases and blindly fitting a model with the biased data could even negatively affect the online learning phase. In this work, we formulate this problem from a causal perspective. First, we categorize the biases into confounding bias and selection bias based on the causal structure they imply. Next, we extract the causal bound for each arm that is robust towards compound biases from biased observational data. The derived bounds contain the ground truth mean reward and can effectively guide the bandit agent to learn a nearly-optimal decision policy. We also conduct regret analysis in both contextual and non-contextual bandit settings and show that prior causal bounds could help consistently reduce the asymptotic regret.

{{</citation>}}


### (88/124) Towards Efficient Verification of Quantized Neural Networks (Pei Huang et al., 2023)

{{<citation>}}

Pei Huang, Haoze Wu, Yuting Yang, Ieva Daukantas, Min Wu, Yedi Zhang, Clark Barrett. (2023)  
**Towards Efficient Verification of Quantized Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-LO, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2312.12679v1)  

---


**ABSTRACT**  
Quantization replaces floating point arithmetic with integer arithmetic in deep neural network models, providing more efficient on-device inference with less power and memory. In this work, we propose a framework for formally verifying properties of quantized neural networks. Our baseline technique is based on integer linear programming which guarantees both soundness and completeness. We then show how efficiency can be improved by utilizing gradient-based heuristic search methods and also bound-propagation techniques. We evaluate our approach on perception networks quantized with PyTorch. Our results show that we can verify quantized networks with better scalability and efficiency than the previous state of the art.

{{</citation>}}


## eess.IV (5)



### (89/124) Responsible Deep Learning for Software as a Medical Device (Pratik Shah et al., 2023)

{{<citation>}}

Pratik Shah, Jenna Lester, Jana G Deflino, Vinay Pai. (2023)  
**Responsible Deep Learning for Software as a Medical Device**  

---
Primary Category: eess.IV  
Categories: I-2; K-4-1; J-3; I-4, cs-CY, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.13333v1)  

---


**ABSTRACT**  
Tools, models and statistical methods for signal processing and medical image analysis and training deep learning models to create research prototypes for eventual clinical applications are of special interest to the biomedical imaging community. But material and optical properties of biological tissues are complex and not easily captured by imaging devices. Added complexity can be introduced by datasets with underrepresentation of medical images from races and ethnicities for deep learning, and limited knowledge about the regulatory framework needed for commercialization and safety of emerging Artificial Intelligence (AI) and Machine Learning (ML) technologies for medical image analysis. This extended version of the workshop paper presented at the special session of the 2022 IEEE 19th International Symposium on Biomedical Imaging, describes strategy and opportunities by University of California professors engaged in machine learning (section I) and clinical research (section II), the Office of Science and Engineering Laboratories (OSEL) section III, and officials at the US FDA in Center for Devices & Radiological Health (CDRH) section IV. Performance evaluations of AI/ML models of skin (RGB), tissue biopsy (digital pathology), and lungs and kidneys (Magnetic Resonance, X-ray, Computed Tomography) medical images for regulatory evaluations and real-world deployment are discussed.

{{</citation>}}


### (90/124) Pixel-to-Abundance Translation: Conditional Generative Adversarial Networks Based on Patch Transformer for Hyperspectral Unmixing (Li Wang et al., 2023)

{{<citation>}}

Li Wang, Xiaohua Zhang, Longfei Li, Hongyun Meng, Xianghai Cao. (2023)  
**Pixel-to-Abundance Translation: Conditional Generative Adversarial Networks Based on Patch Transformer for Hyperspectral Unmixing**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.13127v1)  

---


**ABSTRACT**  
Spectral unmixing is a significant challenge in hyperspectral image processing. Existing unmixing methods utilize prior knowledge about the abundance distribution to solve the regularization optimization problem, where the difficulty lies in choosing appropriate prior knowledge and solving the complex regularization optimization problem. To solve these problems, we propose a hyperspectral conditional generative adversarial network (HyperGAN) method as a generic unmixing framework, based on the following assumption: the unmixing process from pixel to abundance can be regarded as a transformation of two modalities with an internal specific relationship. The proposed HyperGAN is composed of a generator and discriminator, the former completes the modal conversion from mixed hyperspectral pixel patch to the abundance of corresponding endmember of the central pixel and the latter is used to distinguish whether the distribution and structure of generated abundance are the same as the true ones. We propose hyperspectral image (HSI) Patch Transformer as the main component of the generator, which utilize adaptive attention score to capture the internal pixels correlation of the HSI patch and leverage the spatial-spectral information in a fine-grained way to achieve optimization of the unmixing process. Experiments on synthetic data and real hyperspectral data achieve impressive results compared to state-of-the-art competitors.

{{</citation>}}


### (91/124) In2SET: Intra-Inter Similarity Exploiting Transformer for Dual-Camera Compressive Hyperspectral Imaging (Xin Wang et al., 2023)

{{<citation>}}

Xin Wang, Lizhi Wang, Xiangtian Ma, Maoqing Zhang, Lin Zhu, Hua Huang. (2023)  
**In2SET: Intra-Inter Similarity Exploiting Transformer for Dual-Camera Compressive Hyperspectral Imaging**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.13319v1)  

---


**ABSTRACT**  
Dual-Camera Compressed Hyperspectral Imaging (DCCHI) offers the capability to reconstruct 3D Hyperspectral Image (HSI) by fusing compressive and Panchromatic (PAN) image, which has shown great potential for snapshot hyperspectral imaging in practice. In this paper, we introduce a novel DCCHI reconstruction network, the Intra-Inter Similarity Exploiting Transformer (In2SET). Our key insight is to make full use of the PAN image to assist the reconstruction. To this end, we propose using the intra-similarity within the PAN image as a proxy for approximating the intra-similarity in the original HSI, thereby offering an enhanced content prior for more accurate HSI reconstruction. Furthermore, we aim to align the features from the underlying HSI with those of the PAN image, maintaining semantic consistency and introducing new contextual information for the reconstruction process. By integrating In2SET into a PAN-guided unrolling framework, our method substantially enhances the spatial-spectral fidelity and detail of the reconstructed images, providing a more comprehensive and accurate depiction of the scene. Extensive experiments conducted on both real and simulated datasets demonstrate that our approach consistently outperforms existing state-of-the-art methods in terms of reconstruction quality and computational complexity. Code will be released.

{{</citation>}}


### (92/124) Multi-task Learning To Improve Semantic Segmentation Of CBCT Scans Using Image Reconstruction (Maximilian Ernst Tschuchnig et al., 2023)

{{<citation>}}

Maximilian Ernst Tschuchnig, Julia Coste-Marin, Philipp Steininger, Michael Gadermayr. (2023)  
**Multi-task Learning To Improve Semantic Segmentation Of CBCT Scans Using Image Reconstruction**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2312.12990v1)  

---


**ABSTRACT**  
Semantic segmentation is a crucial task in medical image processing, essential for segmenting organs or lesions such as tumors. In this study we aim to improve automated segmentation in CBCTs through multi-task learning. To evaluate effects on different volume qualities, a CBCT dataset is synthesised from the CT Liver Tumor Segmentation Benchmark (LiTS) dataset. To improve segmentation, two approaches are investigated. First, we perform multi-task learning to add morphology based regularization through a volume reconstruction task. Second, we use this reconstruction task to reconstruct the best quality CBCT (most similar to the original CT), facilitating denoising effects. We explore both holistic and patch-based approaches. Our findings reveal that, especially using a patch-based approach, multi-task learning improves segmentation in most cases and that these results can further be improved by our denoising approach.

{{</citation>}}


### (93/124) Learning Exhaustive Correlation for Spectral Super-Resolution: Where Unified Spatial-Spectral Attention Meets Mutual Linear Dependence (Hongyuan Wang et al., 2023)

{{<citation>}}

Hongyuan Wang, Lizhi Wang, Jiang Xu, Chang Chen, Xue Hu, Fenglong Song, Youliang Yan. (2023)  
**Learning Exhaustive Correlation for Spectral Super-Resolution: Where Unified Spatial-Spectral Attention Meets Mutual Linear Dependence**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.12833v1)  

---


**ABSTRACT**  
Spectral super-resolution from the easily obtainable RGB image to hyperspectral image (HSI) has drawn increasing interest in the field of computational photography. The crucial aspect of spectral super-resolution lies in exploiting the correlation within HSIs. However, two types of bottlenecks in existing Transformers limit performance improvement and practical applications. First, existing Transformers often separately emphasize either spatial-wise or spectral-wise correlation, disrupting the 3D features of HSI and hindering the exploitation of unified spatial-spectral correlation. Second, the existing self-attention mechanism learns the correlation between pairs of tokens and captures the full-rank correlation matrix, leading to its inability to establish mutual linear dependence among multiple tokens. To address these issues, we propose a novel Exhaustive Correlation Transformer (ECT) for spectral super-resolution. First, we propose a Spectral-wise Discontinuous 3D (SD3D) splitting strategy, which models unified spatial-spectral correlation by simultaneously utilizing spatial-wise continuous splitting and spectral-wise discontinuous splitting. Second, we propose a Dynamic Low-Rank Mapping (DLRM) model, which captures mutual linear dependence among multiple tokens through a dynamically calculated low-rank dependence map. By integrating unified spatial-spectral attention with mutual linear dependence, our ECT can establish exhaustive correlation within HSI. The experimental results on both simulated and real data indicate that our method achieves state-of-the-art performance. Codes and pretrained models will be available later.

{{</citation>}}


## cs.GT (1)



### (94/124) Learning Best Response Policies in Dynamic Auctions via Deep Reinforcement Learning (Vinzenz Thoma et al., 2023)

{{<citation>}}

Vinzenz Thoma, Michael Curry, Niao He, Sven Seuken. (2023)  
**Learning Best Response Policies in Dynamic Auctions via Deep Reinforcement Learning**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs.GT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.13232v1)  

---


**ABSTRACT**  
Many real-world auctions are dynamic processes, in which bidders interact and report information over multiple rounds with the auctioneer. The sequential decision making aspect paired with imperfect information renders analyzing the incentive properties of such auctions much more challenging than in the static case. It is clear that bidders often have incentives for manipulation, but the full scope of such strategies is not well-understood. We aim to develop a tool for better understanding the incentive properties in dynamic auctions by using reinforcement learning to learn the optimal strategic behavior for an auction participant. We frame the decision problem as a Markov Decision Process, show its relation to multi-task reinforcement learning and use a soft actor-critic algorithm with experience relabeling to best-respond against several known analytical equilibria as well as to find profitable deviations against exploitable bidder strategies.

{{</citation>}}


## cs.SE (5)



### (95/124) Automated DevOps Pipeline Generation for Code Repositories using Large Language Models (Deep Mehta et al., 2023)

{{<citation>}}

Deep Mehta, Kartik Rawool, Subodh Gujar, Bowen Xu. (2023)  
**Automated DevOps Pipeline Generation for Code Repositories using Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, BLEU, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.13225v1)  

---


**ABSTRACT**  
Automating software development processes through the orchestration of GitHub Action workflows has revolutionized the efficiency and agility of software delivery pipelines. This paper presents a detailed investigation into the use of Large Language Models (LLMs) specifically, GPT 3.5 and GPT 4 to generate and evaluate GitHub Action workflows for DevOps tasks. Our methodology involves data collection from public GitHub repositories, prompt engineering for LLM utilization, and evaluation metrics encompassing exact match scores, BLEU scores, and a novel DevOps Aware score. The research scrutinizes the proficiency of GPT 3.5 and GPT 4 in generating GitHub workflows, while assessing the influence of various prompt elements in constructing the most efficient pipeline. Results indicate substantial advancements in GPT 4, particularly in DevOps awareness and syntax correctness. The research introduces a GitHub App built on Probot, empowering users to automate workflow generation within GitHub ecosystem. This study contributes insights into the evolving landscape of AI-driven automation in DevOps practices.

{{</citation>}}


### (96/124) A Novel Approach for Rapid Development Based on ChatGPT and Prompt Engineering (Youjia Li et al., 2023)

{{<citation>}}

Youjia Li, Jianjun Shi, Zheng Zhang. (2023)  
**A Novel Approach for Rapid Development Based on ChatGPT and Prompt Engineering**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BLEU, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.13115v2)  

---


**ABSTRACT**  
Code generation stands as a powerful technique in modern software development, improving development efficiency, reducing errors, and fostering standardization and consistency. Recently, ChatGPT has exhibited immense potential in automatic code generation. However, existing researches on code generation lack guidance for practical software development process. In this study, we utilized ChatGPT to develop a web-based code generation platform consisting of key components: User Interface, Prompt Builder and Backend Service. Specifically, Prompt Builder dynamically generated comprehensive prompts to enhance model generation performance. We conducted experiments on 2 datasets, evaluating the generated code through 8 widely used metrics.The results demonstrate that (1) Our Prompt Builder is effective, resulting in a 65.06% improvement in EM, a 38.45% improvement in BLEU, a 15.70% improvement in CodeBLEU, and a 50.64% improvement in Pass@1. (2) In real development scenarios, 98.5% of test cases can be validated through manual validation, highlighting the genuine assistance provided by the ChatGPT-based code generation approach.

{{</citation>}}


### (97/124) Exploring ChatGPT for Toxicity Detection in GitHub (Shyamal Mishra et al., 2023)

{{<citation>}}

Shyamal Mishra, Preetha Chatterjee. (2023)  
**Exploring ChatGPT for Toxicity Detection in GitHub**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.13105v1)  

---


**ABSTRACT**  
Fostering a collaborative and inclusive environment is crucial for the sustained progress of open source development. However, the prevalence of negative discourse, often manifested as toxic comments, poses significant challenges to developer well-being and productivity. To identify such negativity in project communications, especially within large projects, automated toxicity detection models are necessary. To train these models effectively, we need large software engineering-specific toxicity datasets. However, such datasets are limited in availability and often exhibit imbalance (e.g., only 6 in 1000 GitHub issues are toxic), posing challenges for training effective toxicity detection models. To address this problem, we explore a zero-shot LLM (ChatGPT) that is pre-trained on massive datasets but without being fine-tuned specifically for the task of detecting toxicity in software-related text. Our preliminary evaluation indicates that ChatGPT shows promise in detecting toxicity in GitHub, and warrants further investigation. We experimented with various prompts, including those designed for justifying model outputs, thereby enhancing model interpretability and paving the way for potential integration of ChatGPT-enabled toxicity detection into developer communication channels.

{{</citation>}}


### (98/124) Selecting Source Code Generation Tools Based on Bandit Algorithms (Ryoto Shima et al., 2023)

{{<citation>}}

Ryoto Shima, Masateru Tsunoda, Yukasa Murakami, Akito Monden, Amjed Tahir, Kwabena Ebo Bennin, Koji Toda, Keitaro Nakasai. (2023)  
**Selecting Source Code Generation Tools Based on Bandit Algorithms**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.12813v1)  

---


**ABSTRACT**  
Background: Recently, code generation tools such as ChatGPT have drawn attention to their performance. Generally, a prior analysis of their performance is needed to select new code-generation tools from a list of candidates. Without such analysis, there is a higher risk of selecting an ineffective tool, negatively affecting software development productivity. Additionally, conducting prior analysis of new code generation tools takes time and effort. Aim: To use a new code generation tool without prior analysis but with low risk, we propose to evaluate the new tools during software development (i.e., online optimization). Method: We apply the bandit algorithm (BA) approach to help select the best code-generation tool among candidates. Developers evaluate whether the result of the tool is correct or not. When code generation and evaluation are repeated, the evaluation results are saved. We utilize the stored evaluation results to select the best tool based on the BA approach. Our preliminary analysis evaluated five code generation tools with 164 code generation cases using BA. Result: The BA approach selected ChatGPT as the best tool as the evaluation proceeded, and during the evaluation, the average accuracy by the BA approach outperformed the second-best performing tool. Our results reveal the feasibility and effectiveness of BA in assisting the selection of best-performing code generation tools.

{{</citation>}}


### (99/124) CodeLL: A Lifelong Learning Dataset to Support the Co-Evolution of Data and Language Models of Code (Martin Weyssow et al., 2023)

{{<citation>}}

Martin Weyssow, Claudio Di Sipio, Davide Di Ruscio, Houari Sahraoui. (2023)  
**CodeLL: A Lifelong Learning Dataset to Support the Co-Evolution of Data and Language Models of Code**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.12492v1)  

---


**ABSTRACT**  
Motivated by recent work on lifelong learning applications for language models (LMs) of code, we introduce CodeLL, a lifelong learning dataset focused on code changes. Our contribution addresses a notable research gap marked by the absence of a long-term temporal dimension in existing code change datasets, limiting their suitability in lifelong learning scenarios. In contrast, our dataset aims to comprehensively capture code changes across the entire release history of open-source software repositories. In this work, we introduce an initial version of CodeLL, comprising 71 machine-learning-based projects mined from Software Heritage. This dataset enables the extraction and in-depth analysis of code changes spanning 2,483 releases at both the method and API levels. CodeLL enables researchers studying the behaviour of LMs in lifelong fine-tuning settings for learning code changes. Additionally, the dataset can help studying data distribution shifts within software repositories and the evolution of API usages over time.

{{</citation>}}


## cs.CE (1)



### (100/124) AccidentGPT: Accident analysis and prevention from V2X Environmental Perception with Multi-modal Large Model (Lening Wang et al., 2023)

{{<citation>}}

Lening Wang, Han Jiang, Pinlong Cai, Daocheng Fu, Tianqi Wang, Zhiyong Cui, Yilong Ren, Haiyang Yu, Xuesong Wang, Yinhai Wang. (2023)  
**AccidentGPT: Accident analysis and prevention from V2X Environmental Perception with Multi-modal Large Model**  

---
Primary Category: cs.CE  
Categories: cs-AI, cs-CE, cs.CE  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2312.13156v1)  

---


**ABSTRACT**  
Traffic accidents, being a significant contributor to both human casualties and property damage, have long been a focal point of research for many scholars in the field of traffic safety. However, previous studies, whether focusing on static environmental assessments or dynamic driving analyses, as well as pre-accident predictions or post-accident rule analyses, have typically been conducted in isolation. There has been a lack of an effective framework for developing a comprehensive understanding and application of traffic safety. To address this gap, this paper introduces AccidentGPT, a comprehensive accident analysis and prevention multi-modal large model. AccidentGPT establishes a multi-modal information interaction framework grounded in multi-sensor perception, thereby enabling a holistic approach to accident analysis and prevention in the field of traffic safety. Specifically, our capabilities can be categorized as follows: for autonomous driving vehicles, we provide comprehensive environmental perception and understanding to control the vehicle and avoid collisions. For human-driven vehicles, we offer proactive long-range safety warnings and blind-spot alerts while also providing safety driving recommendations and behavioral norms through human-machine dialogue and interaction. Additionally, for traffic police and management agencies, our framework supports intelligent and real-time analysis of traffic safety, encompassing pedestrian, vehicles, roads, and the environment through collaborative perception from multiple vehicles and road testing devices. The system is also capable of providing a thorough analysis of accident causes and liability after vehicle collisions. Our framework stands as the first large model to integrate comprehensive scene understanding into traffic safety studies.

{{</citation>}}


## physics.chem-ph (1)



### (101/124) Molecular Hypergraph Neural Networks (Junwu Chen et al., 2023)

{{<citation>}}

Junwu Chen, Philippe Schwaller. (2023)  
**Molecular Hypergraph Neural Networks**  

---
Primary Category: physics.chem-ph  
Categories: cs-LG, physics-chem-ph, physics.chem-ph  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2312.13136v2)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have demonstrated promising performance across various chemistry-related tasks. However, conventional graphs only model the pairwise connectivity in molecules, failing to adequately represent higher-order connections like multi-center bonds and conjugated structures. To tackle this challenge, we introduce molecular hypergraphs and propose Molecular Hypergraph Neural Networks (MHNN) to predict the optoelectronic properties of organic semiconductors, where hyperedges represent conjugated structures. A general algorithm is designed for irregular high-order connections, which can efficiently operate on molecular hypergraphs with hyperedges of various orders. The results show that MHNN outperforms all baseline models on most tasks of OPV, OCELOTv1 and PCQM4Mv2 datasets. Notably, MHNN achieves this without any 3D geometric information, surpassing the baseline model that utilizes atom positions. Moreover, MHNN achieves better performance than pretrained GNNs under limited training data, underscoring its excellent data efficiency. This work provides a new strategy for more general molecular representations and property prediction tasks related to high-order connections.

{{</citation>}}


## cs.CY (3)



### (102/124) Generative agents in the streets: Exploring the use of Large Language Models (LLMs) in collecting urban perceptions (Deepank Verma et al., 2023)

{{<citation>}}

Deepank Verma, Olaf Mumm, Vanessa Miriam Carlow. (2023)  
**Generative agents in the streets: Exploring the use of Large Language Models (LLMs) in collecting urban perceptions**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2312.13126v1)  

---


**ABSTRACT**  
Evaluating the surroundings to gain understanding, frame perspectives, and anticipate behavioral reactions is an inherent human trait. However, these continuous encounters are diverse and complex, posing challenges to their study and experimentation. Researchers have been able to isolate environmental features and study their effect on human perception and behavior. However, the research attempts to replicate and study human behaviors with proxies, such as by integrating virtual mediums and interviews, have been inconsistent. Large language models (LLMs) have recently been unveiled as capable of contextual understanding and semantic reasoning. These models have been trained on large amounts of text and have evolved to mimic believable human behavior. This study explores the current advancements in Generative agents powered by LLMs with the help of perceptual experiments. The experiment employs Generative agents to interact with the urban environments using street view images to plan their journey toward specific goals. The agents are given virtual personalities, which make them distinguishable. They are also provided a memory database to store their thoughts and essential visual information and retrieve it when needed to plan their movement. Since LLMs do not possess embodiment, nor have access to the visual realm, and lack a sense of motion or direction, we designed movement and visual modules that help agents gain an overall understanding of surroundings. The agents are further employed to rate the surroundings they encounter based on their perceived sense of safety and liveliness. As these agents store details in their memory, we query the findings to get details regarding their thought processes. Overall, this study experiments with current AI developments and their potential in simulated human behavior in urban environments.

{{</citation>}}


### (103/124) Survey on Multi-Document Summarization: Systematic Literature Review (Uswa Ihsan et al., 2023)

{{<citation>}}

Uswa Ihsan, Humaira Ashraf, NZ Jhanjhi. (2023)  
**Survey on Multi-Document Summarization: Systematic Literature Review**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2312.12915v1)  

---


**ABSTRACT**  
In this era of information technology, abundant information is available on the internet in the form of web pages and documents on any given topic. Finding the most relevant and informative content out of these huge number of documents, without spending several hours of reading has become a very challenging task. Various methods of multi-document summarization have been developed to overcome this problem. The multi-document summarization methods try to produce high-quality summaries of documents with low redundancy. This study conducts a systematic literature review of existing methods for multi-document summarization methods and provides an in-depth analysis of performance achieved by these methods. The findings of the study show that more effective methods are still required for getting higher accuracy of these methods. The study also identifies some open challenges that can gain the attention of future researchers of this domain.

{{</citation>}}


### (104/124) Human-Centred Learning Analytics and AI in Education: a Systematic Literature Review (Riordan Alfredo et al., 2023)

{{<citation>}}

Riordan Alfredo, Vanessa Echeverria, Yueqiao Jin, Lixiang Yan, Zachari Swiecki, Dragan Gašević, Roberto Martinez-Maldonado. (2023)  
**Human-Centred Learning Analytics and AI in Education: a Systematic Literature Review**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-HC, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12751v1)  

---


**ABSTRACT**  
The rapid expansion of Learning Analytics (LA) and Artificial Intelligence in Education (AIED) offers new scalable, data-intensive systems but also raises concerns about data privacy and agency. Excluding stakeholders -- like students and teachers -- from the design process can potentially lead to mistrust and inadequately aligned tools. Despite a shift towards human-centred design in recent LA and AIED research, there remain gaps in our understanding of the importance of human control, safety, reliability, and trustworthiness in the design and implementation of these systems. We conducted a systematic literature review to explore these concerns and gaps. We analysed 108 papers to provide insights about i) the current state of human-centred LA/AIED research; ii) the extent to which educational stakeholders have contributed to the design process of human-centred LA/AIED systems; iii) the current balance between human control and computer automation of such systems; and iv) the extent to which safety, reliability and trustworthiness have been considered in the literature. Results indicate some consideration of human control in LA/AIED system design, but limited end-user involvement in actual design. Based on these findings, we recommend: 1) carefully balancing stakeholders' involvement in designing and deploying LA/AIED systems throughout all design phases, 2) actively involving target end-users, especially students, to delineate the balance between human control and automation, and 3) exploring safety, reliability, and trustworthiness as principles in future human-centred LA/AIED systems.

{{</citation>}}


## cs.CR (4)



### (105/124) Prometheus: Infrastructure Security Posture Analysis with AI-generated Attack Graphs (Xin Jin et al., 2023)

{{<citation>}}

Xin Jin, Charalampos Katsis, Fan Sang, Jiahao Sun, Elisa Bertino, Ramana Rao Kompella, Ashish Kundu. (2023)  
**Prometheus: Infrastructure Security Posture Analysis with AI-generated Attack Graphs**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs-LG, cs.CR  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2312.13119v1)  

---


**ABSTRACT**  
The rampant occurrence of cybersecurity breaches imposes substantial limitations on the progress of network infrastructures, leading to compromised data, financial losses, potential harm to individuals, and disruptions in essential services. The current security landscape demands the urgent development of a holistic security assessment solution that encompasses vulnerability analysis and investigates the potential exploitation of these vulnerabilities as attack paths. In this paper, we propose Prometheus, an advanced system designed to provide a detailed analysis of the security posture of computing infrastructures. Using user-provided information, such as device details and software versions, Prometheus performs a comprehensive security assessment. This assessment includes identifying associated vulnerabilities and constructing potential attack graphs that adversaries can exploit. Furthermore, Prometheus evaluates the exploitability of these attack paths and quantifies the overall security posture through a scoring mechanism. The system takes a holistic approach by analyzing security layers encompassing hardware, system, network, and cryptography. Furthermore, Prometheus delves into the interconnections between these layers, exploring how vulnerabilities in one layer can be leveraged to exploit vulnerabilities in others. In this paper, we present the end-to-end pipeline implemented in Prometheus, showcasing the systematic approach adopted for conducting this thorough security analysis.

{{</citation>}}


### (106/124) Advancing SQL Injection Detection for High-Speed Data Centers: A Novel Approach Using Cascaded NLP (Kasim Tasdemir et al., 2023)

{{<citation>}}

Kasim Tasdemir, Rafiullah Khan, Fahad Siddiqui, Sakir Sezer, Fatih Kurugollu, Sena Busra Yengec-Tasdemir, Alperen Bolat. (2023)  
**Advancing SQL Injection Detection for High-Speed Data Centers: A Novel Approach Using Cascaded NLP**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: BERT, NLP  
[Paper Link](http://arxiv.org/abs/2312.13041v1)  

---


**ABSTRACT**  
Detecting SQL Injection (SQLi) attacks is crucial for web-based data center security, but it is challenging to balance accuracy and computational efficiency, especially in high-speed networks. Traditional methods struggle with this balance, while NLP-based approaches, although accurate, are computationally intensive.   We introduce a novel cascade SQLi detection method, blending classical and transformer-based NLP models, achieving a 99.86% detection accuracy with significantly lower computational demands-20 times faster than using transformer-based models alone. Our approach is tested in a realistic setting and compared with 35 other methods, including Machine Learning-based and transformer models like BERT, on a dataset of over 30,000 SQL sentences.   Our results show that this hybrid method effectively detects SQLi in high-traffic environments, offering efficient and accurate protection against SQLi vulnerabilities with computational efficiency. The code is available at https://github.com/gdrlab/cascaded-sqli-detection .

{{</citation>}}


### (107/124) Symbolic Security Verification of Mesh Commissioning Protocol in Thread (extended version) (Pankaj Upadhyay et al., 2023)

{{<citation>}}

Pankaj Upadhyay, Subodh Sharma, Guangdong Bai. (2023)  
**Symbolic Security Verification of Mesh Commissioning Protocol in Thread (extended version)**  

---
Primary Category: cs.CR  
Categories: 68Q60, I-6-5, cs-CR, cs-SC, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.12958v1)  

---


**ABSTRACT**  
The Thread protocol (or simply Thread ) is a popular networking protocol for the Internet of Things (IoT). It allows seamless integration of a set of applications and protocols, hence reducing the risk of incompatibility among different applications or user protocols. Thread has been deployed in many popular smart home products by the majority of IoT manufacturers, such as Apple TV, Apple HomePod mini, eero 6, Nest Hub, and Nest Wifi. Despite a few empirical analyses on the security of Thread, there is still a lack of formal analysis on this infrastructure of the booming IoT ecosystem. In this work, we performed a formal symbolic analysis of the security properties of Thread. Our main focus is on MeshCoP (Mesh Commissioning Protocol), the main subprotocol in Thread for secure authentication and commissioning of new, untrusted devices inside an existing Thread network. This case study presents the challenges and proposed solutions in modeling MeshCoP. We use ProVerif, a symbolic verification tool of {\pi}-calculus models, for verifying the security properties of MeshCoP.

{{</citation>}}


### (108/124) Secure Authentication Mechanism for Cluster based Vehicular Adhoc Network (VANET): A Survey (Rabia Nasir et al., 2023)

{{<citation>}}

Rabia Nasir, Humaira Ashraf, NZ Jhanjhi. (2023)  
**Secure Authentication Mechanism for Cluster based Vehicular Adhoc Network (VANET): A Survey**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12925v1)  

---


**ABSTRACT**  
Vehicular Ad Hoc Networks (VANETs) play a crucial role in Intelligent Transportation Systems (ITS) by facilitating communication between vehicles and infrastructure. This communication aims to enhance road safety, improve traffic efficiency, and enhance passenger comfort. The secure and reliable exchange of information is paramount to ensure the integrity and confidentiality of data, while the authentication of vehicles and messages is essential to prevent unauthorized access and malicious activities. This survey paper presents a comprehensive analysis of existing authentication mechanisms proposed for cluster-based VANETs. The strengths, weaknesses, and suitability of these mechanisms for various scenarios are carefully examined. Additionally, the integration of secure key management techniques is discussed to enhance the overall authentication process. Cluster-based VANETs are formed by dividing the network into smaller groups or clusters, with designated cluster heads comprising one or more vehicles. Furthermore, this paper identifies gaps in the existing literature through an exploration of previous surveys. Several schemes based on different methods are critically evaluated, considering factors such as throughput, detection rate, security, packet delivery ratio, and end-to-end delay. To provide optimal solutions for authentication in cluster-based VANETs, this paper highlights AI- and ML-based routing-based schemes. These approaches leverage artificial intelligence and machine learning techniques to enhance authentication within the cluster-based VANET network. Finally, this paper explores the open research challenges that exist in the realm of authentication for cluster-based Vehicular Adhoc Networks, shedding light on areas that require further investigation and development.

{{</citation>}}


## cs.PL (2)



### (109/124) Domain-Specific Code Language Models: Unraveling the Potential for HPC Codes and Tasks (Tal Kadosh et al., 2023)

{{<citation>}}

Tal Kadosh, Niranjan Hasabnis, Vy A. Vo, Nadav Schneider, Neva Krien, Mihai Capota, Abdul Wasay, Nesreen Ahmed, Ted Willke, Guy Tamir, Yuval Pinter, Timothy Mattson, Gal Oren. (2023)  
**Domain-Specific Code Language Models: Unraveling the Potential for HPC Codes and Tasks**  

---
Primary Category: cs.PL  
Categories: cs-AI, cs-LG, cs-PL, cs-SE, cs.PL  
Keywords: AI, BLEU, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.13322v1)  

---


**ABSTRACT**  
With easier access to powerful compute resources, there is a growing trend in AI for software development to develop larger language models (LLMs) to address a variety of programming tasks. Even LLMs applied to tasks from the high-performance computing (HPC) domain are huge in size and demand expensive compute resources for training. This is partly because these LLMs for HPC tasks are obtained by finetuning existing LLMs that support several natural and/or programming languages. We found this design choice confusing - why do we need large LMs trained on natural languages and programming languages unrelated to HPC for HPC-specific tasks?   In this line of work, we aim to question choices made by existing LLMs by developing smaller LMs for specific domains - we call them domain-specific LMs. Specifically, we start off with HPC as a domain and build an HPC-specific LM, named MonoCoder, that is orders of magnitude smaller than existing LMs but delivers similar, if not better performance, on non-HPC and HPC tasks. Specifically, we pre-trained MonoCoder on an HPC-specific dataset (named HPCorpus) of C and C++ programs mined from GitHub. We evaluated the performance of MonoCoder against conventional multi-lingual LLMs. Results demonstrate that MonoCoder, although much smaller than existing LMs, achieves similar results on normalized-perplexity tests and much better ones in CodeBLEU competence for high-performance and parallel code generations. Furthermore, fine-tuning the base model for the specific task of parallel code generation (OpenMP parallel for pragmas) demonstrates outstanding results compared to GPT, especially when local misleading semantics are removed by our novel pre-processor Tokompiler, showcasing the ability of domain-specific models to assist in HPC-relevant tasks.

{{</citation>}}


### (110/124) Lampr: Boosting the Effectiveness of Language-Generic Program Reduction via Large Language Models (Mengxiao Zhang et al., 2023)

{{<citation>}}

Mengxiao Zhang, Yongqiang Tian, Zhenyang Xu, Yiwen Dong, Shin Hwei Tan, Chengnian Sun. (2023)  
**Lampr: Boosting the Effectiveness of Language-Generic Program Reduction via Large Language Models**  

---
Primary Category: cs.PL  
Categories: cs-PL, cs-SE, cs.PL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.13064v1)  

---


**ABSTRACT**  
Program reduction is a prevalent technique to facilitate compilers' debugging by automatically minimizing bug-triggering programs. Existing program reduction techniques are either generic across languages (e.g., Perses and Vulcan) or specifically customized for one certain language by employing language-specific features, like C-Reduce. However, striking the balance between generality across multiple programming languages and specificity to individual languages in program reduction is yet to be explored.   This paper proposes Lampr, the first technique utilizing LLMs to perform language-specific program reduction for multiple languages. The core insight is to utilize both the language-generic syntax level program reduction (e.g., Perses) and the language-specific semantic level program transformations learned by LLMs. Alternately, language-generic program reducers efficiently reduce programs into 1-tree-minimality, which is small enough to be manageable for LLMs; LLMs effectively transform programs via the learned semantics to expose new reduction opportunities for the language-generic program reducers to further reduce the programs.   Our extensive evaluation on 50 benchmarks across three languages (C, Rust, and JavaScript) has highlighted Lampr's practicality and superiority over Vulcan, the state-of-the-art language-generic program reducer. For effectiveness, Lampr surpasses Vulcan by producing 24.93\%, 4.47\%, and 11.71\% smaller programs on benchmarks in C, Rust and JavaScript. Moreover, Lampr and Vulcan have demonstrated their potential to complement each other. By using Vulcan on Lampr's output for C programs, we achieve program sizes comparable to those reduced by C-Reduce. For efficiency, Lampr takes 10.77\%, 34.88\%, 36.96\% less time than Vulcan to finish all benchmarks in C, Rust and JavaScript, separately.

{{</citation>}}


## cs.HC (3)



### (111/124) Explainable artificial intelligence approaches for brain-computer interfaces: a review and design space (Param Rajpura et al., 2023)

{{<citation>}}

Param Rajpura, Hubert Cecotti, Yogesh Kumar Meena. (2023)  
**Explainable artificial intelligence approaches for brain-computer interfaces: a review and design space**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-LG, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.13033v1)  

---


**ABSTRACT**  
This review paper provides an integrated perspective of Explainable Artificial Intelligence techniques applied to Brain-Computer Interfaces. BCIs use predictive models to interpret brain signals for various high-stake applications. However, achieving explainability in these complex models is challenging as it compromises accuracy. The field of XAI has emerged to address the need for explainability across various stakeholders, but there is a lack of an integrated perspective in XAI for BCI (XAI4BCI) literature. It is necessary to differentiate key concepts like explainability, interpretability, and understanding in this context and formulate a comprehensive framework. To understand the need of XAI for BCI, we pose six key research questions for a systematic review and meta-analysis, encompassing its purposes, applications, usability, and technical feasibility. We employ the PRISMA methodology -- preferred reporting items for systematic reviews and meta-analyses to review (n=1246) and analyze (n=84) studies published in 2015 and onwards for key insights. The results highlight that current research primarily focuses on interpretability for developers and researchers, aiming to justify outcomes and enhance model performance. We discuss the unique approaches, advantages, and limitations of XAI4BCI from the literature. We draw insights from philosophy, psychology, and social sciences. We propose a design space for XAI4BCI, considering the evolving need to visualize and investigate predictive model outcomes customised for various stakeholders in the BCI development and deployment lifecycle. This paper is the first to focus solely on reviewing XAI4BCI research articles. This systematic review and meta-analysis findings with the proposed design space prompt important discussions on establishing standards for BCI explanations, highlighting current limitations, and guiding the future of XAI in BCI.

{{</citation>}}


### (112/124) Android dialogue system for customer service using prompt-based topic control and compliments generation (Miyama Tamotsu et al., 2023)

{{<citation>}}

Miyama Tamotsu, Okada Shogo. (2023)  
**Android dialogue system for customer service using prompt-based topic control and compliments generation**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: ChatGPT, Dialog, Dialogue, GPT  
[Paper Link](http://arxiv.org/abs/2312.12924v1)  

---


**ABSTRACT**  
This paper describes a dialogue system developed for the Dialogue Robot Competition 2023 that achieves topic control for trip planning by inserting text into prompts using the ChatGPT-API. We built a system that is capable of generating compliments for the user based on recognition of the user's appearance and creating travel plans by extracting the knowledge about the user's preference from the history of the user's utterances. Complements and planning based on preference are the elements required to maintain the quality of customer service. A preliminary round was held at a travel agency's actual store, where real customers experienced and evaluated the system. This system was evaluated first in the preliminary round and participated in the final round. The results of the preliminary round showed the effectiveness of the proposed system.

{{</citation>}}


### (113/124) 3D-CLMI: A Motor Imagery EEG Classification Model via Fusion of 3D-CNN and LSTM with Attention (Shiwei Cheng et al., 2023)

{{<citation>}}

Shiwei Cheng, Yuejiang Hao. (2023)  
**3D-CLMI: A Motor Imagery EEG Classification Model via Fusion of 3D-CNN and LSTM with Attention**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs-LG, cs.HC, eess-SP  
Keywords: Attention, LSTM  
[Paper Link](http://arxiv.org/abs/2312.12744v1)  

---


**ABSTRACT**  
Due to the limitations in the accuracy and robustness of current electroencephalogram (EEG) classification algorithms, applying motor imagery (MI) for practical Brain-Computer Interface (BCI) applications remains challenging. This paper proposed a model that combined a three-dimensional convolutional neural network (CNN) with a long short-term memory (LSTM) network with attention to classify MI-EEG signals. This model combined MI-EEG signals from different channels into three-dimensional features and extracted spatial features through convolution operations with multiple three-dimensional convolutional kernels of different scales. At the same time, to ensure the integrity of the extracted MI-EEG signal temporal features, the LSTM network was directly trained on the preprocessed raw signal. Finally, the features obtained from these two networks were combined and used for classification. Experimental results showed that this model achieved a classification accuracy of 92.7% and an F1-score of 0.91 on the public dataset BCI Competition IV dataset 2a, which were both higher than the state-of-the-art models in the field of MI tasks. Additionally, 12 participants were invited to complete a four-class MI task in our lab, and experiments on the collected dataset showed that the 3D-CLMI model also maintained the highest classification accuracy and F1-score. The model greatly improved the classification accuracy of users' motor imagery intentions, giving brain-computer interfaces better application prospects in emerging fields such as autonomous vehicles and medical rehabilitation.

{{</citation>}}


## eess.AS (4)



### (114/124) FusDom: Combining In-Domain and Out-of-Domain Knowledge for Continuous Self-Supervised Learning (Ashish Seth et al., 2023)

{{<citation>}}

Ashish Seth, Sreyan Ghosh, S. Umesh, Dinesh Manocha. (2023)  
**FusDom: Combining In-Domain and Out-of-Domain Knowledge for Continuous Self-Supervised Learning**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.13026v1)  

---


**ABSTRACT**  
Continued pre-training (CP) offers multiple advantages, like target domain adaptation and the potential to exploit the continuous stream of unlabeled data available online. However, continued pre-training on out-of-domain distributions often leads to catastrophic forgetting of previously acquired knowledge, leading to sub-optimal ASR performance. This paper presents FusDom, a simple and novel methodology for SSL-based continued pre-training. FusDom learns speech representations that are robust and adaptive yet not forgetful of concepts seen in the past. Instead of solving the SSL pre-text task on the output representations of a single model, FusDom leverages two identical pre-trained SSL models, a teacher and a student, with a modified pre-training head to solve the CP SSL pre-text task. This head employs a cross-attention mechanism between the representations of both models while only the student receives gradient updates and the teacher does not. Finally, the student is fine-tuned for ASR. In practice, FusDom outperforms all our baselines across settings significantly, with WER improvements in the range of 0.2 WER - 7.3 WER in the target domain while retaining the performance in the earlier domain.

{{</citation>}}


### (115/124) CST-former: Transformer with Channel-Spectro-Temporal Attention for Sound Event Localization and Detection (Yusun Shul et al., 2023)

{{<citation>}}

Yusun Shul, Jung-Woo Choi. (2023)  
**CST-former: Transformer with Channel-Spectro-Temporal Attention for Sound Event Localization and Detection**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2312.12821v1)  

---


**ABSTRACT**  
Sound event localization and detection (SELD) is a task for the classification of sound events and the localization of direction of arrival (DoA) utilizing multichannel acoustic signals. Prior studies employ spectral and channel information as the embedding for temporal attention. However, this usage limits the deep neural network from extracting meaningful features from the spectral or spatial domains. Therefore, our investigation in this paper presents a novel framework termed the Channel-Spectro-Temporal Transformer (CST-former) that bolsters SELD performance through the independent application of attention mechanisms to distinct domains. The CST-former architecture employs distinct attention mechanisms to independently process channel, spectral, and temporal information. In addition, we propose an unfolded local embedding (ULE) technique for channel attention (CA) to generate informative embedding vectors including local spectral and temporal information. Empirical validation through experimentation on the 2022 and 2023 DCASE Challenge task3 datasets affirms the efficacy of employing attention mechanisms separated across each domain and the benefit of ULE, in enhancing SELD performance.

{{</citation>}}


### (116/124) Stable Distillation: Regularizing Continued Pre-training for Low-Resource Automatic Speech Recognition (Ashish Seth et al., 2023)

{{<citation>}}

Ashish Seth, Sreyan Ghosh, S. Umesh, Dinesh Manocha. (2023)  
**Stable Distillation: Regularizing Continued Pre-training for Low-Resource Automatic Speech Recognition**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Low-Resource, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2312.12783v1)  

---


**ABSTRACT**  
Continued self-supervised (SSL) pre-training for adapting existing SSL models to the target domain has shown to be extremely effective for low-resource Automatic Speech Recognition (ASR). This paper proposes Stable Distillation, a simple and novel approach for SSL-based continued pre-training that boosts ASR performance in the target domain where both labeled and unlabeled data are limited. Stable Distillation employs self-distillation as regularization for continued pre-training, alleviating the over-fitting issue, a common problem continued pre-training faces when the source and target domains differ. Specifically, first, we perform vanilla continued pre-training on an initial SSL pre-trained model on the target domain ASR dataset and call it the teacher. Next, we take the same initial pre-trained model as a student to perform continued pre-training while enforcing its hidden representations to be close to that of the teacher (via MSE loss). This student is then used for downstream ASR fine-tuning on the target dataset. In practice, Stable Distillation outperforms all our baselines by 0.8 - 7 WER when evaluated in various experimental settings.

{{</citation>}}


### (117/124) Lattice Rescoring Based on Large Ensemble of Complementary Neural Language Models (Atsunori Ogawa et al., 2023)

{{<citation>}}

Atsunori Ogawa, Naohiro Tawara, Marc Delcroix, Shoko Araki. (2023)  
**Lattice Rescoring Based on Large Ensemble of Complementary Neural Language Models**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2312.12764v1)  

---


**ABSTRACT**  
We investigate the effectiveness of using a large ensemble of advanced neural language models (NLMs) for lattice rescoring on automatic speech recognition (ASR) hypotheses. Previous studies have reported the effectiveness of combining a small number of NLMs. In contrast, in this study, we combine up to eight NLMs, i.e., forward/backward long short-term memory/Transformer-LMs that are trained with two different random initialization seeds. We combine these NLMs through iterative lattice generation. Since these NLMs work complementarily with each other, by combining them one by one at each rescoring iteration, language scores attached to given lattice arcs can be gradually refined. Consequently, errors of the ASR hypotheses can be gradually reduced. We also investigate the effectiveness of carrying over contextual information (previous rescoring results) across a lattice sequence of a long speech such as a lecture speech. In experiments using a lecture speech corpus, by combining the eight NLMs and using context carry-over, we obtained a 24.4% relative word error rate reduction from the ASR 1-best baseline. For further comparison, we performed simultaneous (i.e., non-iterative) NLM combination and 100-best rescoring using the large ensemble of NLMs, which confirmed the advantage of lattice rescoring with iterative NLM combination.

{{</citation>}}


## cs.AR (1)



### (118/124) Accelerator-driven Data Arrangement to Minimize Transformers Run-time on Multi-core Architectures (Alireza Amirshahi et al., 2023)

{{<citation>}}

Alireza Amirshahi, Giovanni Ansaloni, David Atienza. (2023)  
**Accelerator-driven Data Arrangement to Minimize Transformers Run-time on Multi-core Architectures**  

---
Primary Category: cs.AR  
Categories: cs-AI, cs-AR, cs.AR  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.13000v1)  

---


**ABSTRACT**  
The increasing complexity of transformer models in artificial intelligence expands their computational costs, memory usage, and energy consumption. Hardware acceleration tackles the ensuing challenges by designing processors and accelerators tailored for transformer models, supporting their computation hotspots with high efficiency. However, memory bandwidth can hinder improvements in hardware accelerators. Against this backdrop, in this paper we propose a novel memory arrangement strategy, governed by the hardware accelerator's kernel size, which effectively minimizes off-chip data access. This arrangement is particularly beneficial for end-to-end transformer model inference, where most of the computation is based on general matrix multiplication (GEMM) operations. Additionally, we address the overhead of non-GEMM operations in transformer models within the scope of this memory data arrangement. Our study explores the implementation and effectiveness of the proposed accelerator-driven data arrangement approach in both single- and multi-core systems. Our evaluation demonstrates that our approach can achieve up to a 2.8x speed increase when executing inferences employing state-of-the-art transformers.

{{</citation>}}


## physics.soc-ph (1)



### (119/124) Big Tech influence over AI research revisited: memetic analysis of attribution of ideas to affiliation (Stanisław Giziński et al., 2023)

{{<citation>}}

Stanisław Giziński, Paulina Kaczyńska, Hubert Ruczyński, Emilia Wiśnios, Bartosz Pieliński, Przemysław Biecek, Julian Sienkiewicz. (2023)  
**Big Tech influence over AI research revisited: memetic analysis of attribution of ideas to affiliation**  

---
Primary Category: physics.soc-ph  
Categories: cs-CL, cs-SI, physics-soc-ph, physics.soc-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12881v1)  

---


**ABSTRACT**  
There exists a growing discourse around the domination of Big Tech on the landscape of artificial intelligence (AI) research, yet our comprehension of this phenomenon remains cursory. This paper aims to broaden and deepen our understanding of Big Tech's reach and power within AI research. It highlights the dominance not merely in terms of sheer publication volume but rather in the propagation of new ideas or \textit{memes}. Current studies often oversimplify the concept of influence to the share of affiliations in academic papers, typically sourced from limited databases such as arXiv or specific academic conferences.   The main goal of this paper is to unravel the specific nuances of such influence, determining which AI ideas are predominantly driven by Big Tech entities. By employing network and memetic analysis on AI-oriented paper abstracts and their citation network, we are able to grasp a deeper insight into this phenomenon. By utilizing two databases: OpenAlex and S2ORC, we are able to perform such analysis on a much bigger scale than previous attempts.   Our findings suggest, that while Big Tech-affiliated papers are disproportionately more cited in some areas, the most cited papers are those affiliated with both Big Tech and Academia. Focusing on the most contagious memes, their attribution to specific affiliation groups (Big Tech, Academia, mixed affiliation) seems to be equally distributed between those three groups. This suggests that the notion of Big Tech domination over AI research is oversimplified in the discourse.   Ultimately, this more nuanced understanding of Big Tech's and Academia's influence could inform a more symbiotic alliance between these stakeholders which would better serve the dual goals of societal welfare and the scientific integrity of AI research.

{{</citation>}}


## quant-ph (1)



### (120/124) Quantum Annealing for Computer Vision Minimization Problems (Shahrokh Heidari et al., 2023)

{{<citation>}}

Shahrokh Heidari, Michael J. Dinneen, Patrice Delmas. (2023)  
**Quantum Annealing for Computer Vision Minimization Problems**  

---
Primary Category: quant-ph  
Categories: cs-CV, quant-ph, quant-ph  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.12848v1)  

---


**ABSTRACT**  
Computer Vision (CV) labelling algorithms play a pivotal role in the domain of low-level vision. For decades, it has been known that these problems can be elegantly formulated as discrete energy minimization problems derived from probabilistic graphical models (such as Markov Random Fields). Despite recent advances in inference algorithms (such as graph-cut and message-passing algorithms), the resulting energy minimization problems are generally viewed as intractable. The emergence of quantum computations, which offer the potential for faster solutions to certain problems than classical methods, has led to an increased interest in utilizing quantum properties to overcome intractable problems. Recently, there has also been a growing interest in Quantum Computer Vision (QCV), with the hope of providing a credible alternative or assistant to deep learning solutions in the field. This study investigates a new Quantum Annealing based inference algorithm for CV discrete energy minimization problems. Our contribution is focused on Stereo Matching as a significant CV labeling problem. As a proof of concept, we also use a hybrid quantum-classical solver provided by D-Wave System to compare our results with the best classical inference algorithms in the literature.

{{</citation>}}


## cs.IR (2)



### (121/124) Parallel Ranking of Ads and Creatives in Real-Time Advertising Systems (Zhiguang Yang et al., 2023)

{{<citation>}}

Zhiguang Yang, Lu Wang, Chun Gan, Liufang Sang, Haoran Wang, Wenlong Chen, Jie He, Changping Peng, Zhangang Lin, Jingping Shao. (2023)  
**Parallel Ranking of Ads and Creatives in Real-Time Advertising Systems**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.12750v1)  

---


**ABSTRACT**  
"Creativity is the heart and soul of advertising services". Effective creatives can create a win-win scenario: advertisers can reach target users and achieve marketing objectives more effectively, users can more quickly find products of interest, and platforms can generate more advertising revenue. With the advent of AI-Generated Content, advertisers now can produce vast amounts of creative content at a minimal cost. The current challenge lies in how advertising systems can select the most pertinent creative in real-time for each user personally. Existing methods typically perform serial ranking of ads or creatives, limiting the creative module in terms of both effectiveness and efficiency. In this paper, we propose for the first time a novel architecture for online parallel estimation of ads and creatives ranking, as well as the corresponding offline joint optimization model. The online architecture enables sophisticated personalized creative modeling while reducing overall latency. The offline joint model for CTR estimation allows mutual awareness and collaborative optimization between ads and creatives. Additionally, we optimize the offline evaluation metrics for the implicit feedback sorting task involved in ad creative ranking. We conduct extensive experiments to compare ours with two state-of-the-art approaches. The results demonstrate the effectiveness of our approach in both offline evaluations and real-world advertising platforms online in terms of response time, CTR, and CPM.

{{</citation>}}


### (122/124) Lookahead: An Inference Acceleration Framework for Large Language Model with Lossless Generation Accuracy (Yao Zhao et al., 2023)

{{<citation>}}

Yao Zhao, Zhitian Xie, Chenyi Zhuang, Jinjie Gu. (2023)  
**Lookahead: An Inference Acceleration Framework for Large Language Model with Lossless Generation Accuracy**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs-LG, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.12728v1)  

---


**ABSTRACT**  
As Large Language Models (LLMs) have made significant advancements across various tasks, such as question answering, translation, text summarization, and dialogue systems, the need for accuracy in information becomes crucial, especially for serious financial products serving billions of users like Alipay. To address this, Alipay has developed a Retrieval-Augmented Generation (RAG) system that grounds LLMs on the most accurate and up-to-date information. However, for a real-world product serving millions of users, the inference speed of LLMs becomes a critical factor compared to a mere experimental model.   Hence, this paper presents a generic framework for accelerating the inference process, resulting in a substantial increase in speed and cost reduction for our RAG system, with lossless generation accuracy. In the traditional inference process, each token is generated sequentially by the LLM, leading to a time consumption proportional to the number of generated tokens. To enhance this process, our framework, named \textit{lookahead}, introduces a \textit{multi-branch} strategy. Instead of generating a single token at a time, we propose a \textit{Trie-based Retrieval} (TR) process that enables the generation of multiple branches simultaneously, each of which is a sequence of tokens. Subsequently, for each branch, a \textit{Verification and Accept} (VA) process is performed to identify the longest correct sub-sequence as the final output. Our strategy offers two distinct advantages: (1) it guarantees absolute correctness of the output, avoiding any approximation algorithms, and (2) the worst-case performance of our approach is equivalent to the conventional process. We conduct extensive experiments to demonstrate the significant improvements achieved by applying our inference acceleration framework.

{{</citation>}}


## cs.IT (1)



### (123/124) DoDo-Code: a Deep Levenshtein Distance Embedding-based Code for IDS Channel and DNA Storage (Alan J. X. Guo et al., 2023)

{{<citation>}}

Alan J. X. Guo, Sihan Sun, Xiang Wei, Mengyi Wei, Xin Chen. (2023)  
**DoDo-Code: a Deep Levenshtein Distance Embedding-based Code for IDS Channel and DNA Storage**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs-LG, cs.IT, math-IT  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.12717v1)  

---


**ABSTRACT**  
Recently, DNA storage has emerged as a promising data storage solution, offering significant advantages in storage density, maintenance cost efficiency, and parallel replication capability. Mathematically, the DNA storage pipeline can be viewed as an insertion, deletion, and substitution (IDS) channel. Because of the mathematical terra incognita of the Levenshtein distance, designing an IDS-correcting code is still a challenge. In this paper, we propose an innovative approach that utilizes deep Levenshtein distance embedding to bypass these mathematical challenges. By representing the Levenshtein distance between two sequences as a conventional distance between their corresponding embedding vectors, the inherent structural property of Levenshtein distance is revealed in the friendly embedding space. Leveraging this embedding space, we introduce the DoDo-Code, an IDS-correcting code that incorporates deep embedding of Levenshtein distance, deep embedding-based codeword search, and deep embedding-based segment correcting. To address the requirements of DNA storage, we also present a preliminary algorithm for long sequence decoding. As far as we know, the DoDo-Code is the first IDS-correcting code designed using plausible deep learning methodologies, potentially paving the way for a new direction in error-correcting code research. It is also the first IDS code that exhibits characteristics of being `optimal' in terms of redundancy, significantly outperforming the mainstream IDS-correcting codes of the Varshamov-Tenengolts code family in code rate.

{{</citation>}}


## cs.DC (1)



### (124/124) Optimizing Distributed Training on Frontier for Large Language Models (Sajal Dash et al., 2023)

{{<citation>}}

Sajal Dash, Isaac Lyngaas, Junqi Yin, Xiao Wang, Romain Egele, Guojing Cong, Feiyi Wang, Prasanna Balaprakash. (2023)  
**Optimizing Distributed Training on Frontier for Large Language Models**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs.DC  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.12705v1)  

---


**ABSTRACT**  
Large language models (LLM) are showing tremendous success as foundation models, and many downstream applications benefit from fine-tuning. Prior works on loss scaling have demonstrated that the larger LLMs perform better than their smaller counterparts. However, training LLMs with billions of parameters requires considerable computational resources; to train a one trillion GPT-style model on 20 trillion tokens, we need to perform 120 million exaflops. Frontier is the world's first and fastest exascale supercomputer for open science and is equipped with 75264 MI250X GPUs. This work explores efficient distributed strategies such as tensor parallelism, pipeline parallelism, and sharded data parallelism to train a trillion-parameter model on the Frontier exascale supercomputer. We analyze these distributed training techniques and associated parameters individually to decide which techniques to use and what associated parameters to select for a particular technique. We perform hyperparameter tuning on these techniques to understand their complex interplay. Combined with these two tuning efforts, we have found optimal strategies to train three models of size 22B, 175B, and 1T parameters with $38.38\%$ , $36.14\%$ , and $31.96\%$ achieved throughput. For training the 175B parameter model and 1T model, we have achieved $100\%$ weak scaling efficiency and $89\%$ and $87\%$ strong scaling efficiency, respectively. Our work presents a set of strategies for distributed training of LLMs through experimental findings and hyperparameter tuning.

{{</citation>}}
