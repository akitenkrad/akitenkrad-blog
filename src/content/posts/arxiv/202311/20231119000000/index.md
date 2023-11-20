---
draft: false
title: "arXiv @ 2023.11.19"
date: 2023-11-19
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.19"
    identifier: arxiv_20231119
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (11)](#cscv-11)
- [cs.LG (13)](#cslg-13)
- [cs.CL (15)](#cscl-15)
- [cs.DS (1)](#csds-1)
- [cs.HC (3)](#cshc-3)
- [cs.RO (2)](#csro-2)
- [physics.flu-dyn (1)](#physicsflu-dyn-1)
- [cs.AI (1)](#csai-1)
- [cs.PL (1)](#cspl-1)
- [eess.IV (4)](#eessiv-4)
- [cs.CR (2)](#cscr-2)
- [cs.DC (1)](#csdc-1)
- [cs.NI (1)](#csni-1)
- [cs.SE (2)](#csse-2)
- [cs.IR (1)](#csir-1)
- [cs.AR (1)](#csar-1)

## cs.CV (11)



### (1/60) Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning (Rohit Girdhar et al., 2023)

{{<citation>}}

Rohit Girdhar, Mannat Singh, Andrew Brown, Quentin Duval, Samaneh Azadi, Sai Saketh Rambhatla, Akbar Shah, Xi Yin, Devi Parikh, Ishan Misra. (2023)  
**Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-GR, cs-LG, cs-MM, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2311.10709v1)  

---


**ABSTRACT**  
We present Emu Video, a text-to-video generation model that factorizes the generation into two steps: first generating an image conditioned on the text, and then generating a video conditioned on the text and the generated image. We identify critical design decisions--adjusted noise schedules for diffusion, and multi-stage training--that enable us to directly generate high quality and high resolution videos, without requiring a deep cascade of models as in prior work. In human evaluations, our generated videos are strongly preferred in quality compared to all prior work--81% vs. Google's Imagen Video, 90% vs. Nvidia's PYOCO, and 96% vs. Meta's Make-A-Video. Our model outperforms commercial solutions such as RunwayML's Gen2 and Pika Labs. Finally, our factorizing approach naturally lends itself to animating images based on a user's text prompt, where our generations are preferred 96% over prior work.

{{</citation>}}


### (2/60) SpACNN-LDVAE: Spatial Attention Convolutional Latent Dirichlet Variational Autoencoder for Hyperspectral Pixel Unmixing (Soham Chitnis et al., 2023)

{{<citation>}}

Soham Chitnis, Kiran Mantripragada, Faisal Z. Qureshi. (2023)  
**SpACNN-LDVAE: Spatial Attention Convolutional Latent Dirichlet Variational Autoencoder for Hyperspectral Pixel Unmixing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.10701v1)  

---


**ABSTRACT**  
The Hyperspectral Unxming problem is to find the pure spectral signal of the underlying materials (endmembers) and their proportions (abundances). The proposed method builds upon the recently proposed method, Latent Dirichlet Variational Autoencoder (LDVAE). It assumes that abundances can be encoded as Dirichlet Distributions while mixed pixels and endmembers are represented by Multivariate Normal Distributions. However, LDVAE does not leverage spatial information present in an HSI; we propose an Isotropic CNN encoder with spatial attention to solve the hyperspectral unmixing problem. We evaluated our model on Samson, Hydice Urban, Cuprite, and OnTech-HSI-Syn-21 datasets. Our model also leverages the transfer learning paradigm for Cuprite Dataset, where we train the model on synthetic data and evaluate it on real-world data. We are able to observe the improvement in the results for the endmember extraction and abundance estimation by incorporating the spatial information. Code can be found at https://github.com/faisalqureshi/cnn-ldvae

{{</citation>}}


### (3/60) 3D-TexSeg: Unsupervised Segmentation of 3D Texture using Mutual Transformer Learning (Iyyakutti Iyappan Ganapathi et al., 2023)

{{<citation>}}

Iyyakutti Iyappan Ganapathi, Fayaz Ali, Sajid Javed, Syed Sadaf Ali, Naoufel Werghi. (2023)  
**3D-TexSeg: Unsupervised Segmentation of 3D Texture using Mutual Transformer Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.10651v1)  

---


**ABSTRACT**  
Analysis of the 3D Texture is indispensable for various tasks, such as retrieval, segmentation, classification, and inspection of sculptures, knitted fabrics, and biological tissues. A 3D texture is a locally repeated surface variation independent of the surface's overall shape and can be determined using the local neighborhood and its characteristics. Existing techniques typically employ computer vision techniques that analyze a 3D mesh globally, derive features, and then utilize the obtained features for retrieval or classification. Several traditional and learning-based methods exist in the literature, however, only a few are on 3D texture, and nothing yet, to the best of our knowledge, on the unsupervised schemes. This paper presents an original framework for the unsupervised segmentation of the 3D texture on the mesh manifold. We approach this problem as binary surface segmentation, partitioning the mesh surface into textured and non-textured regions without prior annotation. We devise a mutual transformer-based system comprising a label generator and a cleaner. The two models take geometric image representations of the surface mesh facets and label them as texture or non-texture across an iterative mutual learning scheme. Extensive experiments on three publicly available datasets with diverse texture patterns demonstrate that the proposed framework outperforms standard and SOTA unsupervised techniques and competes reasonably with supervised methods.

{{</citation>}}


### (4/60) FOCAL: A Cost-Aware Video Dataset for Active Learning (Kiran Kokilepersaud et al., 2023)

{{<citation>}}

Kiran Kokilepersaud, Yash-Yee Logan, Ryan Benkert, Chen Zhou, Mohit Prabhushankar, Ghassan AlRegib, Enrique Corona, Kunjan Singh, Mostafa Parchami. (2023)  
**FOCAL: A Cost-Aware Video Dataset for Active Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2311.10591v1)  

---


**ABSTRACT**  
In this paper, we introduce the FOCAL (Ford-OLIVES Collaboration on Active Learning) dataset which enables the study of the impact of annotation-cost within a video active learning setting. Annotation-cost refers to the time it takes an annotator to label and quality-assure a given video sequence. A practical motivation for active learning research is to minimize annotation-cost by selectively labeling informative samples that will maximize performance within a given budget constraint. However, previous work in video active learning lacks real-time annotation labels for accurately assessing cost minimization and instead operates under the assumption that annotation-cost scales linearly with the amount of data to annotate. This assumption does not take into account a variety of real-world confounding factors that contribute to a nonlinear cost such as the effect of an assistive labeling tool and the variety of interactions within a scene such as occluded objects, weather, and motion of objects. FOCAL addresses this discrepancy by providing real annotation-cost labels for 126 video sequences across 69 unique city scenes with a variety of weather, lighting, and seasonal conditions. We also introduce a set of conformal active learning algorithms that take advantage of the sequential structure of video data in order to achieve a better trade-off between annotation-cost and performance while also reducing floating point operations (FLOPS) overhead by at least 77.67%. We show how these approaches better reflect how annotations on videos are done in practice through a sequence selection framework. We further demonstrate the advantage of these approaches by introducing two performance-cost metrics and show that the best conformal active learning method is cheaper than the best traditional active learning method by 113 hours.

{{</citation>}}


### (5/60) SSB: Simple but Strong Baseline for Boosting Performance of Open-Set Semi-Supervised Learning (Yue Fan et al., 2023)

{{<citation>}}

Yue Fan, Anna Kukleva, Dengxin Dai, Bernt Schiele. (2023)  
**SSB: Simple but Strong Baseline for Boosting Performance of Open-Set Semi-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2311.10572v1)  

---


**ABSTRACT**  
Semi-supervised learning (SSL) methods effectively leverage unlabeled data to improve model generalization. However, SSL models often underperform in open-set scenarios, where unlabeled data contain outliers from novel categories that do not appear in the labeled set. In this paper, we study the challenging and realistic open-set SSL setting, where the goal is to both correctly classify inliers and to detect outliers. Intuitively, the inlier classifier should be trained on inlier data only. However, we find that inlier classification performance can be largely improved by incorporating high-confidence pseudo-labeled data, regardless of whether they are inliers or outliers. Also, we propose to utilize non-linear transformations to separate the features used for inlier classification and outlier detection in the multi-task learning framework, preventing adverse effects between them. Additionally, we introduce pseudo-negative mining, which further boosts outlier detection performance. The three ingredients lead to what we call Simple but Strong Baseline (SSB) for open-set SSL. In experiments, SSB greatly improves both inlier classification and outlier detection performance, outperforming existing methods by a large margin. Our code will be released at https://github.com/YUE-FAN/SSB.

{{</citation>}}


### (6/60) Enhancing Object Coherence in Layout-to-Image Synthesis (Yibin Wang et al., 2023)

{{<citation>}}

Yibin Wang, Weizhong Zhang, Jianwei Zheng, Cheng Jin. (2023)  
**Enhancing Object Coherence in Layout-to-Image Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.10522v1)  

---


**ABSTRACT**  
Layout-to-image synthesis is an emerging technique in conditional image generation. It aims to generate complex scenes, where users require fine control over the layout of the objects in a scene. However, it remains challenging to control the object coherence, including semantic coherence (e.g., the cat looks at the flowers or not) and physical coherence (e.g., the hand and the racket should not be misaligned). In this paper, we propose a novel diffusion model with effective global semantic fusion (GSF) and self-similarity feature enhancement modules to guide the object coherence for this task. For semantic coherence, we argue that the image caption contains rich information for defining the semantic relationship within the objects in the images. Instead of simply employing cross-attention between captions and generated images, which addresses the highly relevant layout restriction and semantic coherence separately and thus leads to unsatisfying results shown in our experiments, we develop GSF to fuse the supervision from the layout restriction and semantic coherence requirement and exploit it to guide the image synthesis process. Moreover, to improve the physical coherence, we develop a Self-similarity Coherence Attention (SCA) module to explicitly integrate local contextual physical coherence into each pixel's generation process. Specifically, we adopt a self-similarity map to encode the coherence restrictions and employ it to extract coherent features from text embedding. Through visualization of our self-similarity map, we explore the essence of SCA, revealing that its effectiveness is not only in capturing reliable physical coherence patterns but also in enhancing complex texture generation. Extensive experiments demonstrate the superiority of our proposed method in both image generation quality and controllability.

{{</citation>}}


### (7/60) DUA-DA: Distillation-based Unbiased Alignment for Domain Adaptive Object Detection (Yongchao Feng et al., 2023)

{{<citation>}}

Yongchao Feng, Shiwei Li, Yingjie Gao, Ziyue Huang, Yanan Zhang, Qingjie Liu, Yunhong Wang. (2023)  
**DUA-DA: Distillation-based Unbiased Alignment for Domain Adaptive Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.10437v1)  

---


**ABSTRACT**  
Though feature-alignment based Domain Adaptive Object Detection (DAOD) have achieved remarkable progress, they ignore the source bias issue, i.e. the aligned features are more favorable towards the source domain, leading to a sub-optimal adaptation. Furthermore, the presence of domain shift between the source and target domains exacerbates the problem of inconsistent classification and localization in general detection pipelines. To overcome these challenges, we propose a novel Distillation-based Unbiased Alignment (DUA) framework for DAOD, which can distill the source features towards a more balanced position via a pre-trained teacher model during the training process, alleviating the problem of source bias effectively. In addition, we design a Target-Relevant Object Localization Network (TROLN), which can mine target-related knowledge to produce two classification-free metrics (IoU and centerness). Accordingly, we implement a Domain-aware Consistency Enhancing (DCE) strategy that utilizes these two metrics to further refine classification confidences, achieving a harmonization between classification and localization in cross-domain scenarios. Extensive experiments have been conducted to manifest the effectiveness of this method, which consistently improves the strong baseline by large margins, outperforming existing alignment-based works.

{{</citation>}}


### (8/60) Breaking Temporal Consistency: Generating Video Universal Adversarial Perturbations Using Image Models (Hee-Seon Kim et al., 2023)

{{<citation>}}

Hee-Seon Kim, Minji Son, Minbeom Kim, Myung-Joon Kwon, Changick Kim. (2023)  
**Breaking Temporal Consistency: Generating Video Universal Adversarial Perturbations Using Image Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2311.10366v1)  

---


**ABSTRACT**  
As video analysis using deep learning models becomes more widespread, the vulnerability of such models to adversarial attacks is becoming a pressing concern. In particular, Universal Adversarial Perturbation (UAP) poses a significant threat, as a single perturbation can mislead deep learning models on entire datasets. We propose a novel video UAP using image data and image model. This enables us to take advantage of the rich image data and image model-based studies available for video applications. However, there is a challenge that image models are limited in their ability to analyze the temporal aspects of videos, which is crucial for a successful video attack. To address this challenge, we introduce the Breaking Temporal Consistency (BTC) method, which is the first attempt to incorporate temporal information into video attacks using image models. We aim to generate adversarial videos that have opposite patterns to the original. Specifically, BTC-UAP minimizes the feature similarity between neighboring frames in videos. Our approach is simple but effective at attacking unseen video models. Additionally, it is applicable to videos of varying lengths and invariant to temporal shifts. Our approach surpasses existing methods in terms of effectiveness on various datasets, including ImageNet, UCF-101, and Kinetics-400.

{{</citation>}}


### (9/60) Enhancing Student Engagement in Online Learning through Facial Expression Analysis and Complex Emotion Recognition using Deep Learning (Rekha R Nair et al., 2023)

{{<citation>}}

Rekha R Nair, Tina Babu, Pavithra K. (2023)  
**Enhancing Student Engagement in Online Learning through Facial Expression Analysis and Complex Emotion Recognition using Deep Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2311.10343v1)  

---


**ABSTRACT**  
In response to the COVID-19 pandemic, traditional physical classrooms have transitioned to online environments, necessitating effective strategies to ensure sustained student engagement. A significant challenge in online teaching is the absence of real-time feedback from teachers on students learning progress. This paper introduces a novel approach employing deep learning techniques based on facial expressions to assess students engagement levels during online learning sessions. Human emotions cannot be adequately conveyed by a student using only the basic emotions, including anger, disgust, fear, joy, sadness, surprise, and neutrality. To address this challenge, proposed a generation of four complex emotions such as confusion, satisfaction, disappointment, and frustration by combining the basic emotions. These complex emotions are often experienced simultaneously by students during the learning session. To depict these emotions dynamically,utilized a continuous stream of image frames instead of discrete images. The proposed work utilized a Convolutional Neural Network (CNN) model to categorize the fundamental emotional states of learners accurately. The proposed CNN model demonstrates strong performance, achieving a 95% accuracy in precise categorization of learner emotions.

{{</citation>}}


### (10/60) Shifting to Machine Supervision: Annotation-Efficient Semi and Self-Supervised Learning for Automatic Medical Image Segmentation and Classification (Pranav Singh et al., 2023)

{{<citation>}}

Pranav Singh, Raviteja Chukkapalli, Shravan Chaudhari, Luoyao Chen, Mei Chen, Jinqian Pan, Craig Smuda, Jacopo Cirrone. (2023)  
**Shifting to Machine Supervision: Annotation-Efficient Semi and Self-Supervised Learning for Automatic Medical Image Segmentation and Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2311.10319v1)  

---


**ABSTRACT**  
Advancements in clinical treatment and research are limited by supervised learning techniques that rely on large amounts of annotated data, an expensive task requiring many hours of clinical specialists' time. In this paper, we propose using self-supervised and semi-supervised learning. These techniques perform an auxiliary task that is label-free, scaling up machine-supervision is easier compared with fully-supervised techniques. This paper proposes S4MI (Self-Supervision and Semi-Supervision for Medical Imaging), our pipeline to leverage advances in self and semi-supervision learning. We benchmark them on three medical imaging datasets to analyze their efficacy for classification and segmentation. This advancement in self-supervised learning with 10% annotation performed better than 100% annotation for the classification of most datasets. The semi-supervised approach yielded favorable outcomes for segmentation, outperforming the fully-supervised approach by using 50% fewer labels in all three datasets.

{{</citation>}}


### (11/60) SSASS: Semi-Supervised Approach for Stenosis Segmentation (In Kyu Lee et al., 2023)

{{<citation>}}

In Kyu Lee, Junsup Shin, Yong-Hee Lee, Jonghoe Ku, Hyun-Woo Kim. (2023)  
**SSASS: Semi-Supervised Approach for Stenosis Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2311.10281v1)  

---


**ABSTRACT**  
Coronary artery stenosis is a critical health risk, and its precise identification in Coronary Angiography (CAG) can significantly aid medical practitioners in accurately evaluating the severity of a patient's condition. The complexity of coronary artery structures combined with the inherent noise in X-ray images poses a considerable challenge to this task. To tackle these obstacles, we introduce a semi-supervised approach for cardiovascular stenosis segmentation. Our strategy begins with data augmentation, specifically tailored to replicate the structural characteristics of coronary arteries. We then apply a pseudo-label-based semi-supervised learning technique that leverages the data generated through our augmentation process. Impressively, our approach demonstrated an exceptional performance in the Automatic Region-based Coronary Artery Disease diagnostics using x-ray angiography imagEs (ARCADE) Stenosis Detection Algorithm challenge by utilizing a single model instead of relying on an ensemble of multiple models. This success emphasizes our method's capability and efficiency in providing an automated solution for accurately assessing stenosis severity from medical imaging data.

{{</citation>}}


## cs.LG (13)



### (12/60) Multimodal Representation Learning by Alternating Unimodal Adaptation (Xiaohui Zhang et al., 2023)

{{<citation>}}

Xiaohui Zhang, Jaehong Yoon, Mohit Bansal, Huaxiu Yao. (2023)  
**Multimodal Representation Learning by Alternating Unimodal Adaptation**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.10707v1)  

---


**ABSTRACT**  
Multimodal learning, which integrates data from diverse sensory modes, plays a pivotal role in artificial intelligence. However, existing multimodal learning methods often struggle with challenges where some modalities appear more dominant than others during multimodal learning, resulting in suboptimal performance. To address this challenge, we propose MLA (Multimodal Learning with Alternating Unimodal Adaptation). MLA reframes the conventional joint multimodal learning process by transforming it into an alternating unimodal learning process, thereby minimizing interference between modalities. Simultaneously, it captures cross-modal interactions through a shared head, which undergoes continuous optimization across different modalities. This optimization process is controlled by a gradient modification mechanism to prevent the shared head from losing previously acquired information. During the inference phase, MLA utilizes a test-time uncertainty-based model fusion mechanism to integrate multimodal information. Extensive experiments are conducted on five diverse datasets, encompassing scenarios with complete modalities and scenarios with missing modalities. These experiments demonstrate the superiority of MLA over competing prior approaches.

{{</citation>}}


### (13/60) Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks (Benjamin Feuer et al., 2023)

{{<citation>}}

Benjamin Feuer, Chinmay Hegde, Niv Cohen. (2023)  
**Scaling TabPFN: Sketching and Feature Selection for Tabular Prior-Data Fitted Networks**  

---
Primary Category: cs.LG  
Categories: cs-DB, cs-LG, cs.LG  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2311.10609v1)  

---


**ABSTRACT**  
Tabular classification has traditionally relied on supervised algorithms, which estimate the parameters of a prediction model using its training data. Recently, Prior-Data Fitted Networks (PFNs) such as TabPFN have successfully learned to classify tabular data in-context: the model parameters are designed to classify new samples based on labelled training samples given after the model training. While such models show great promise, their applicability to real-world data remains limited due to the computational scale needed. Here we study the following question: given a pre-trained PFN for tabular data, what is the best way to summarize the labelled training samples before feeding them to the model? We conduct an initial investigation of sketching and feature-selection methods for TabPFN, and note certain key differences between it and conventionally fitted tabular models.

{{</citation>}}


### (14/60) EduGym: An Environment Suite for Reinforcement Learning Education (Thomas M. Moerland et al., 2023)

{{<citation>}}

Thomas M. Moerland, Matthias Müller-Brockhausen, Zhao Yang, Andrius Bernatavicius, Koen Ponse, Tom Kouwenhoven, Andreas Sauter, Michiel van der Meer, Bram Renting, Aske Plaat. (2023)  
**EduGym: An Environment Suite for Reinforcement Learning Education**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.10590v1)  

---


**ABSTRACT**  
Due to the empirical success of reinforcement learning, an increasing number of students study the subject. However, from our practical teaching experience, we see students entering the field (bachelor, master and early PhD) often struggle. On the one hand, textbooks and (online) lectures provide the fundamentals, but students find it hard to translate between equations and code. On the other hand, public codebases do provide practical examples, but the implemented algorithms tend to be complex, and the underlying test environments contain multiple reinforcement learning challenges at once. Although this is realistic from a research perspective, it often hinders educational conceptual understanding. To solve this issue we introduce EduGym, a set of educational reinforcement learning environments and associated interactive notebooks tailored for education. Each EduGym environment is specifically designed to illustrate a certain aspect/challenge of reinforcement learning (e.g., exploration, partial observability, stochasticity, etc.), while the associated interactive notebook explains the challenge and its possible solution approaches, connecting equations and code in a single document. An evaluation among RL students and researchers shows 86% of them think EduGym is a useful tool for reinforcement learning education. All notebooks are available from https://sites.google.com/view/edu-gym/home, while the full software package can be installed from https://github.com/RLG-Leiden/edugym.

{{</citation>}}


### (15/60) Graph Neural Networks for Pressure Estimation in Water Distribution Systems (Huy Truong et al., 2023)

{{<citation>}}

Huy Truong, Andrés Tello, Alexander Lazovik, Victoria Degeler. (2023)  
**Graph Neural Networks for Pressure Estimation in Water Distribution Systems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.10579v1)  

---


**ABSTRACT**  
Pressure and flow estimation in Water Distribution Networks (WDN) allows water management companies to optimize their control operations. For many years, mathematical simulation tools have been the most common approach to reconstructing an estimate of the WDN hydraulics. However, pure physics-based simulations involve several challenges, e.g. partially observable data, high uncertainty, and extensive manual configuration. Thus, data-driven approaches have gained traction to overcome such limitations. In this work, we combine physics-based modeling and Graph Neural Networks (GNN), a data-driven approach, to address the pressure estimation problem. First, we propose a new data generation method using a mathematical simulation but not considering temporal patterns and including some control parameters that remain untouched in previous works; this contributes to a more diverse training data. Second, our training strategy relies on random sensor placement making our GNN-based estimation model robust to unexpected sensor location changes. Third, a realistic evaluation protocol considers real temporal patterns and additionally injects the uncertainties intrinsic to real-world scenarios. Finally, a multi-graph pre-training strategy allows the model to be reused for pressure estimation in unseen target WDNs. Our GNN-based model estimates the pressure of a large-scale WDN in The Netherlands with a MAE of 1.94mH$_2$O and a MAPE of 7%, surpassing the performance of previous studies. Likewise, it outperformed previous approaches on other WDN benchmarks, showing a reduction of absolute error up to approximately 52% in the best cases.

{{</citation>}}


### (16/60) Regions are Who Walk Them: a Large Pre-trained Spatiotemporal Model Based on Human Mobility for Ubiquitous Urban Sensing (Ruixing Zhang et al., 2023)

{{<citation>}}

Ruixing Zhang, Liangzhe Han, Leilei Sun, Yunqi Liu, Jibin Wang, Weifeng Lv. (2023)  
**Regions are Who Walk Them: a Large Pre-trained Spatiotemporal Model Based on Human Mobility for Ubiquitous Urban Sensing**  

---
Primary Category: cs.LG  
Categories: 68T30, cs-AI, cs-LG, cs.LG  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.10471v1)  

---


**ABSTRACT**  
User profiling and region analysis are two tasks of significant commercial value. However, in practical applications, modeling different features typically involves four main steps: data preparation, data processing, model establishment, evaluation, and optimization. This process is time-consuming and labor-intensive. Repeating this workflow for each feature results in abundant development time for tasks and a reduced overall volume of task development. Indeed, human mobility data contains a wealth of information. Several successful cases suggest that conducting in-depth analysis of population movement data could potentially yield meaningful profiles about users and areas. Nonetheless, most related works have not thoroughly utilized the semantic information within human mobility data and trained on a fixed number of the regions. To tap into the rich information within population movement, based on the perspective that Regions Are Who walk them, we propose a large spatiotemporal model based on trajectories (RAW). It possesses the following characteristics: 1) Tailored for trajectory data, introducing a GPT-like structure with a parameter count of up to 1B; 2) Introducing a spatiotemporal fine-tuning module, interpreting trajectories as collection of users to derive arbitrary region embedding. This framework allows rapid task development based on the large spatiotemporal model. We conducted extensive experiments to validate the effectiveness of our proposed large spatiotemporal model. It's evident that our proposed method, relying solely on human mobility data without additional features, exhibits a certain level of relevance in user profiling and region analysis. Moreover, our model showcases promising predictive capabilities in trajectory generation tasks based on the current state, offering the potential for further innovative work utilizing this large spatiotemporal model.

{{</citation>}}


### (17/60) Using Cooperative Game Theory to Prune Neural Networks (Mauricio Diaz-Ortiz Jr et al., 2023)

{{<citation>}}

Mauricio Diaz-Ortiz Jr, Benjamin Kempinski, Daphne Cornelisse, Yoram Bachrach, Tal Kachman. (2023)  
**Using Cooperative Game Theory to Prune Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CE, cs-GT, cs-LG, cs-MA, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2311.10468v1)  

---


**ABSTRACT**  
We show how solution concepts from cooperative game theory can be used to tackle the problem of pruning neural networks.   The ever-growing size of deep neural networks (DNNs) increases their performance, but also their computational requirements. We introduce a method called Game Theory Assisted Pruning (GTAP), which reduces the neural network's size while preserving its predictive accuracy. GTAP is based on eliminating neurons in the network based on an estimation of their joint impact on the prediction quality through game theoretic solutions. Specifically, we use a power index akin to the Shapley value or Banzhaf index, tailored using a procedure similar to Dropout (commonly used to tackle overfitting problems in machine learning).   Empirical evaluation of both feedforward networks and convolutional neural networks shows that this method outperforms existing approaches in the achieved tradeoff between the number of parameters and model accuracy.

{{</citation>}}


### (18/60) Maintenance Techniques for Anomaly Detection AIOps Solutions (Lorena Poenaru-Olaru et al., 2023)

{{<citation>}}

Lorena Poenaru-Olaru, Natalia Karpova, Luis Cruz, Jan Rellermeyer, Arie van Deursen. (2023)  
**Maintenance Techniques for Anomaly Detection AIOps Solutions**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SE, cs.LG  
Keywords: AI, Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2311.10421v1)  

---


**ABSTRACT**  
Anomaly detection techniques are essential in automating the monitoring of IT systems and operations. These techniques imply that machine learning algorithms are trained on operational data corresponding to a specific period of time and that they are continuously evaluated on newly emerging data. Operational data is constantly changing over time, which affects the performance of deployed anomaly detection models. Therefore, continuous model maintenance is required to preserve the performance of anomaly detectors over time. In this work, we analyze two different anomaly detection model maintenance techniques in terms of the model update frequency, namely blind model retraining and informed model retraining. We further investigate the effects of updating the model by retraining it on all the available data (full-history approach) and on only the newest data (sliding window approach). Moreover, we investigate whether a data change monitoring tool is capable of determining when the anomaly detection model needs to be updated through retraining.

{{</citation>}}


### (19/60) Few-shot Message-Enhanced Contrastive Learning for Graph Anomaly Detection (Fan Xu et al., 2023)

{{<citation>}}

Fan Xu, Nan Wang, Xuezhi Wen, Meiqi Gao, Chaoqun Guo, Xibin Zhao. (2023)  
**Few-shot Message-Enhanced Contrastive Learning for Graph Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, Contrastive Learning, GNN  
[Paper Link](http://arxiv.org/abs/2311.10370v1)  

---


**ABSTRACT**  
Graph anomaly detection plays a crucial role in identifying exceptional instances in graph data that deviate significantly from the majority. It has gained substantial attention in various domains of information security, including network intrusion, financial fraud, and malicious comments, et al. Existing methods are primarily developed in an unsupervised manner due to the challenge in obtaining labeled data. For lack of guidance from prior knowledge in unsupervised manner, the identified anomalies may prove to be data noise or individual data instances. In real-world scenarios, a limited batch of labeled anomalies can be captured, making it crucial to investigate the few-shot problem in graph anomaly detection. Taking advantage of this potential, we propose a novel few-shot Graph Anomaly Detection model called FMGAD (Few-shot Message-Enhanced Contrastive-based Graph Anomaly Detector). FMGAD leverages a self-supervised contrastive learning strategy within and across views to capture intrinsic and transferable structural representations. Furthermore, we propose the Deep-GNN message-enhanced reconstruction module, which extensively exploits the few-shot label information and enables long-range propagation to disseminate supervision signals to deeper unlabeled nodes. This module in turn assists in the training of self-supervised contrastive learning. Comprehensive experimental results on six real-world datasets demonstrate that FMGAD can achieve better performance than other state-of-the-art methods, regardless of artificially injected anomalies or domain-organic anomalies.

{{</citation>}}


### (20/60) Federated Knowledge Graph Completion via Latent Embedding Sharing and Tensor Factorization (Maolin Wang et al., 2023)

{{<citation>}}

Maolin Wang, Dun Zeng, Zenglin Xu, Ruocheng Guo, Xiangyu Zhao. (2023)  
**Federated Knowledge Graph Completion via Latent Embedding Sharing and Tensor Factorization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2311.10341v1)  

---


**ABSTRACT**  
Knowledge graphs (KGs), which consist of triples, are inherently incomplete and always require completion procedure to predict missing triples. In real-world scenarios, KGs are distributed across clients, complicating completion tasks due to privacy restrictions. Many frameworks have been proposed to address the issue of federated knowledge graph completion. However, the existing frameworks, including FedE, FedR, and FEKG, have certain limitations. = FedE poses a risk of information leakage, FedR's optimization efficacy diminishes when there is minimal overlap among relations, and FKGE suffers from computational costs and mode collapse issues. To address these issues, we propose a novel method, i.e., Federated Latent Embedding Sharing Tensor factorization (FLEST), which is a novel approach using federated tensor factorization for KG completion. FLEST decompose the embedding matrix and enables sharing of latent dictionary embeddings to lower privacy risks. Empirical results demonstrate FLEST's effectiveness and efficiency, offering a balanced solution between performance and privacy. FLEST expands the application of federated tensor factorization in KG completion tasks.

{{</citation>}}


### (21/60) Imagination-augmented Hierarchical Reinforcement Learning for Safe and Interactive Autonomous Driving in Urban Environments (Sang-Hyun Lee et al., 2023)

{{<citation>}}

Sang-Hyun Lee, Yoonjae Jung, Seung-Woo Seo. (2023)  
**Imagination-augmented Hierarchical Reinforcement Learning for Safe and Interactive Autonomous Driving in Urban Environments**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.10309v1)  

---


**ABSTRACT**  
Hierarchical reinforcement learning (HRL) has led to remarkable achievements in diverse fields. However, existing HRL algorithms still cannot be applied to real-world navigation tasks. These tasks require an agent to perform safety-aware behaviors and interact with surrounding objects in dynamic environments. In addition, an agent in these tasks should perform consistent and structured exploration as they are long-horizon and have complex structures with diverse objects and task-specific rules. Designing HRL agents that can handle these challenges in real-world navigation tasks is an open problem. In this paper, we propose imagination-augmented HRL (IAHRL), a new and general navigation algorithm that allows an agent to learn safe and interactive behaviors in real-world navigation tasks. Our key idea is to train a hierarchical agent in which a high-level policy infers interactions by interpreting behaviors imagined with low-level policies. Specifically, the high-level policy is designed with a permutation-invariant attention mechanism to determine which low-level policy generates the most interactive behavior, and the low-level policies are implemented with an optimization-based behavior planner to generate safe and structured behaviors following task-specific rules. To evaluate our algorithm, we introduce five complex urban driving tasks, which are among the most challenging real-world navigation tasks. The experimental results indicate that our hierarchical agent performs safety-aware behaviors and properly interacts with surrounding vehicles, achieving higher success rates and lower average episode steps than baselines in urban driving tasks.

{{</citation>}}


### (22/60) Hierarchical Pruning of Deep Ensembles with Focal Diversity (Yanzhao Wu et al., 2023)

{{<citation>}}

Yanzhao Wu, Ka-Ho Chow, Wenqi Wei, Ling Liu. (2023)  
**Hierarchical Pruning of Deep Ensembles with Focal Diversity**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2311.10293v1)  

---


**ABSTRACT**  
Deep neural network ensembles combine the wisdom of multiple deep neural networks to improve the generalizability and robustness over individual networks. It has gained increasing popularity to study deep ensemble techniques in the deep learning community. Some mission-critical applications utilize a large number of deep neural networks to form deep ensembles to achieve desired accuracy and resilience, which introduces high time and space costs for ensemble execution. However, it still remains a critical challenge whether a small subset of the entire deep ensemble can achieve the same or better generalizability and how to effectively identify these small deep ensembles for improving the space and time efficiency of ensemble execution. This paper presents a novel deep ensemble pruning approach, which can efficiently identify smaller deep ensembles and provide higher ensemble accuracy than the entire deep ensemble of a large number of member networks. Our hierarchical ensemble pruning approach (HQ) leverages three novel ensemble pruning techniques. First, we show that the focal diversity metrics can accurately capture the complementary capacity of the member networks of an ensemble, which can guide ensemble pruning. Second, we design a focal diversity based hierarchical pruning approach, which will iteratively find high quality deep ensembles with low cost and high accuracy. Third, we develop a focal diversity consensus method to integrate multiple focal diversity metrics to refine ensemble pruning results, where smaller deep ensembles can be effectively identified to offer high accuracy, high robustness and high efficiency. Evaluated using popular benchmark datasets, we demonstrate that the proposed hierarchical ensemble pruning approach can effectively identify high quality deep ensembles with better generalizability while being more time and space efficient in ensemble decision making.

{{</citation>}}


### (23/60) FREE: The Foundational Semantic Recognition for Modeling Environmental Ecosystems (Shiyuan Luo et al., 2023)

{{<citation>}}

Shiyuan Luo, Juntong Ni, Shengyu Chen, Runlong Yu, Yiqun Xie, Licheng Liu, Zhenong Jin, Huaxiu Yao, Xiaowei Jia. (2023)  
**FREE: The Foundational Semantic Recognition for Modeling Environmental Ecosystems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-PE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.10255v1)  

---


**ABSTRACT**  
Modeling environmental ecosystems is critical for the sustainability of our planet, but is extremely challenging due to the complex underlying processes driven by interactions amongst a large number of physical variables. As many variables are difficult to measure at large scales, existing works often utilize a combination of observable features and locally available measurements or modeled values as input to build models for a specific study region and time period. This raises a fundamental question in advancing the modeling of environmental ecosystems: how to build a general framework for modeling the complex relationships amongst various environmental data over space and time? In this paper, we introduce a new framework, FREE, which maps available environmental data into a text space and then converts the traditional predictive modeling task in environmental science to the semantic recognition problem. The proposed FREE framework leverages recent advances in Large Language Models (LLMs) to supplement the original input features with natural language descriptions. This facilitates capturing the data semantics and also allows harnessing the irregularities of input features. When used for long-term prediction, FREE has the flexibility to incorporate newly collected observations to enhance future prediction. The efficacy of FREE is evaluated in the context of two societally important real-world applications, predicting stream water temperature in the Delaware River Basin and predicting annual corn yield in Illinois and Iowa. Beyond the superior predictive performance over multiple baseline methods, FREE is shown to be more data- and computation-efficient as it can be pre-trained on simulated data generated by physics-based models.

{{</citation>}}


### (24/60) Advancements in Generative AI: A Comprehensive Review of GANs, GPT, Autoencoders, Diffusion Model, and Transformers (Staphord Bengesi et al., 2023)

{{<citation>}}

Staphord Bengesi, Hoda El-Sayed, Md Kamruzzaman Sarker, Yao Houkpati, John Irungu, Timothy Oladunni. (2023)  
**Advancements in Generative AI: A Comprehensive Review of GANs, GPT, Autoencoders, Diffusion Model, and Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, ChatGPT, GPT, GPT-4, Generative AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.10242v1)  

---


**ABSTRACT**  
The launch of ChatGPT has garnered global attention, marking a significant milestone in the field of Generative Artificial Intelligence. While Generative AI has been in effect for the past decade, the introduction of ChatGPT has ignited a new wave of research and innovation in the AI domain. This surge in interest has led to the development and release of numerous cutting-edge tools, such as Bard, Stable Diffusion, DALL-E, Make-A-Video, Runway ML, and Jukebox, among others. These tools exhibit remarkable capabilities, encompassing tasks ranging from text generation and music composition, image creation, video production, code generation, and even scientific work. They are built upon various state-of-the-art models, including Stable Diffusion, transformer models like GPT-3 (recent GPT-4), variational autoencoders, and generative adversarial networks. This advancement in Generative AI presents a wealth of exciting opportunities and, simultaneously, unprecedented challenges. Throughout this paper, we have explored these state-of-the-art models, the diverse array of tasks they can accomplish, the challenges they pose, and the promising future of Generative Artificial Intelligence.

{{</citation>}}


## cs.CL (15)



### (25/60) Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2 (Hamish Ivison et al., 2023)

{{<citation>}}

Hamish Ivison, Yizhong Wang, Valentina Pyatkin, Nathan Lambert, Matthew Peters, Pradeep Dasigi, Joel Jang, David Wadden, Noah A. Smith, Iz Beltagy, Hannaneh Hajishirzi. (2023)  
**Camels in a Changing Climate: Enhancing LM Adaptation with Tulu 2**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2311.10702v1)  

---


**ABSTRACT**  
Since the release of T\"ULU [Wang et al., 2023b], open resources for instruction tuning have developed quickly, from better base models to new finetuning techniques. We test and incorporate a number of these advances into T\"ULU, resulting in T\"ULU 2, a suite of improved T\"ULU models for advancing the understanding and best practices of adapting pretrained language models to downstream tasks and user preferences. Concretely, we release: (1) T\"ULU-V2-mix, an improved collection of high-quality instruction datasets; (2) T\"ULU 2, LLAMA-2 models finetuned on the V2 mixture; (3) T\"ULU 2+DPO, T\"ULU 2 models trained with direct preference optimization (DPO), including the largest DPO-trained model to date (T\"ULU 2+DPO 70B); (4) CODE T\"ULU 2, CODE LLAMA models finetuned on our V2 mix that outperform CODE LLAMA and its instruction-tuned variant, CODE LLAMA-Instruct. Our evaluation from multiple perspectives shows that the T\"ULU 2 suite achieves state-of-the-art performance among open models and matches or exceeds the performance of GPT-3.5-turbo-0301 on several benchmarks. We release all the checkpoints, data, training and evaluation code to facilitate future open efforts on adapting large language models.

{{</citation>}}


### (26/60) PEFT-MedAware: Large Language Model for Medical Awareness (Keivalya Pandya, 2023)

{{<citation>}}

Keivalya Pandya. (2023)  
**PEFT-MedAware: Large Language Model for Medical Awareness**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: AI, Falcon, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2311.10697v1)  

---


**ABSTRACT**  
Chat models are capable of answering a wide range of questions, however, the accuracy of their responses is highly uncertain. In this research, we propose a specialized PEFT-MedAware model where we utilize parameter-efficient fine-tuning (PEFT) to enhance the Falcon-1b large language model on specialized MedQuAD data consisting of 16,407 medical QA pairs, leveraging only 0.44% of its trainable parameters to enhance computational efficiency. The paper adopts data preprocessing and PEFT to optimize model performance, complemented by a BitsAndBytesConfig for efficient transformer training. The resulting model was capable of outperforming other LLMs in medical question-answering tasks in specific domains with greater accuracy utilizing limited computational resources making it suitable for deployment in resource-constrained environments. We propose further improvements through expanded datasets, larger models, and feedback mechanisms for sustained medical relevancy. Our work highlights the efficiency gains and specialized capabilities of PEFT in medical AI, outpacing standard models in precision without extensive resource demands. The proposed model and data are released for research purposes only.

{{</citation>}}


### (27/60) Rethinking Attention: Exploring Shallow Feed-Forward Neural Networks as an Alternative to Attention Layers in Transformers (Vukasin Bozic et al., 2023)

{{<citation>}}

Vukasin Bozic, Danilo Dordevic, Daniele Coppola, Joseph Thommes. (2023)  
**Rethinking Attention: Exploring Shallow Feed-Forward Neural Networks as an Alternative to Attention Layers in Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Attention, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.10642v1)  

---


**ABSTRACT**  
This work presents an analysis of the effectiveness of using standard shallow feed-forward networks to mimic the behavior of the attention mechanism in the original Transformer model, a state-of-the-art architecture for sequence-to-sequence tasks. We substitute key elements of the attention mechanism in the Transformer with simple feed-forward networks, trained using the original components via knowledge distillation. Our experiments, conducted on the IWSLT2017 dataset, reveal the capacity of these "attentionless Transformers" to rival the performance of the original architecture. Through rigorous ablation studies, and experimenting with various replacement network types and sizes, we offer insights that support the viability of our approach. This not only sheds light on the adaptability of shallow feed-forward networks in emulating attention mechanisms but also underscores their potential to streamline complex architectures for sequence-to-sequence tasks.

{{</citation>}}


### (28/60) A Self-enhancement Approach for Domain-specific Chatbot Training via Knowledge Mining and Digest (Ruohong Zhang et al., 2023)

{{<citation>}}

Ruohong Zhang, Luyu Gao, Chen Zheng, Zhen Fan, Guokun Lai, Zheng Zhang, Fangzhou Ai, Yiming Yang, Hongxia Yang. (2023)  
**A Self-enhancement Approach for Domain-specific Chatbot Training via Knowledge Mining and Digest**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2311.10614v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), despite their great power in language generation, often encounter challenges when dealing with intricate and knowledge-demanding queries in specific domains. This paper introduces a novel approach to enhance LLMs by effectively extracting the relevant knowledge from domain-specific textual sources, and the adaptive training of a chatbot with domain-specific inquiries. Our two-step approach starts from training a knowledge miner, namely LLMiner, which autonomously extracts Question-Answer pairs from relevant documents through a chain-of-thought reasoning process. Subsequently, we blend the mined QA pairs with a conversational dataset to fine-tune the LLM as a chatbot, thereby enriching its domain-specific expertise and conversational capabilities. We also developed a new evaluation benchmark which comprises four domain-specific text corpora and associated human-crafted QA pairs for testing. Our model shows remarkable performance improvement over generally aligned LLM and surpasses domain-adapted models directly fine-tuned on domain corpus. In particular, LLMiner achieves this with minimal human intervention, requiring only 600 seed instances, thereby providing a pathway towards self-improvement of LLMs through model-synthesized training data.

{{</citation>}}


### (29/60) Hashing it Out: Predicting Unhealthy Conversations on Twitter (Steven Leung et al., 2023)

{{<citation>}}

Steven Leung, Filippos Papapolyzos. (2023)  
**Hashing it Out: Predicting Unhealthy Conversations on Twitter**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, BERT, LSTM, Twitter  
[Paper Link](http://arxiv.org/abs/2311.10596v1)  

---


**ABSTRACT**  
Personal attacks in the context of social media conversations often lead to fast-paced derailment, leading to even more harmful exchanges being made. State-of-the-art systems for the detection of such conversational derailment often make use of deep learning approaches for prediction purposes. In this paper, we show that an Attention-based BERT architecture, pre-trained on a large Twitter corpus and fine-tuned on our task, is efficient and effective in making such predictions. This model shows clear advantages in performance to the existing LSTM model we use as a baseline. Additionally, we show that this impressive performance can be attained through fine-tuning on a relatively small, novel dataset, particularly after mitigating overfitting issues through synthetic oversampling techniques. By introducing the first transformer based model for forecasting conversational events on Twitter, this work lays the foundation for a practical tool to encourage better interactions on one of the most ubiquitous social media platforms.

{{</citation>}}


### (30/60) Detection of Offensive and Threatening Online Content in a Low Resource Language (Fatima Muhammad Adam et al., 2023)

{{<citation>}}

Fatima Muhammad Adam, Abubakar Yakubu Zandam, Isa Inuwa-Dutse. (2023)  
**Detection of Offensive and Threatening Online Content in a Low Resource Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Google, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.10541v1)  

---


**ABSTRACT**  
Hausa is a major Chadic language, spoken by over 100 million people in Africa. However, from a computational linguistic perspective, it is considered a low-resource language, with limited resources to support Natural Language Processing (NLP) tasks. Online platforms often facilitate social interactions that can lead to the use of offensive and threatening language, which can go undetected due to the lack of detection systems designed for Hausa. This study aimed to address this issue by (1) conducting two user studies (n=308) to investigate cyberbullying-related issues, (2) collecting and annotating the first set of offensive and threatening datasets to support relevant downstream tasks in Hausa, (3) developing a detection system to flag offensive and threatening content, and (4) evaluating the detection system and the efficacy of the Google-based translation engine in detecting offensive and threatening terms in Hausa. We found that offensive and threatening content is quite common, particularly when discussing religion and politics. Our detection system was able to detect more than 70% of offensive and threatening content, although many of these were mistranslated by Google's translation engine. We attribute this to the subtle relationship between offensive and threatening content and idiomatic expressions in the Hausa language. We recommend that diverse stakeholders participate in understanding local conventions and demographics in order to develop a more effective detection system. These insights are essential for implementing targeted moderation strategies to create a safe and inclusive online environment.

{{</citation>}}


### (31/60) Sinhala-English Word Embedding Alignment: Introducing Datasets and Benchmark for a Low Resource Language (Kasun Wickramasinghe et al., 2023)

{{<citation>}}

Kasun Wickramasinghe, Nisansa de Silva. (2023)  
**Sinhala-English Word Embedding Alignment: Introducing Datasets and Benchmark for a Low Resource Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, NLP, Natural Language Processing, Word Embedding  
[Paper Link](http://arxiv.org/abs/2311.10436v1)  

---


**ABSTRACT**  
Since their inception, embeddings have become a primary ingredient in many flavours of Natural Language Processing (NLP) tasks supplanting earlier types of representation. Even though multilingual embeddings have been used for the increasing number of multilingual tasks, due to the scarcity of parallel training data, low-resource languages such as Sinhala, tend to focus more on monolingual embeddings. Then when it comes to the aforementioned multi-lingual tasks, it is challenging to utilize these monolingual embeddings given that even if the embedding spaces have a similar geometric arrangement due to an identical training process, the embeddings of the languages considered are not aligned. This is solved by the embedding alignment task. Even in this, high-resource language pairs are in the limelight while low-resource languages such as Sinhala which is in dire need of help seem to have fallen by the wayside. In this paper, we try to align Sinhala and English word embedding spaces based on available alignment techniques and introduce a benchmark for Sinhala language embedding alignment. In addition to that, to facilitate the supervised alignment, as an intermediate task, we also introduce Sinhala-English alignment datasets. These datasets serve as our anchor datasets for supervised word embedding alignment. Even though we do not obtain results comparable to the high-resource languages such as French, German, or Chinese, we believe our work lays the groundwork for more specialized alignment between English and Sinhala embeddings.

{{</citation>}}


### (32/60) Causal Graph in Language Model Rediscovers Cortical Hierarchy in Human Narrative Processing (Zhengqi He et al., 2023)

{{<citation>}}

Zhengqi He, Taro Toyoizumi. (2023)  
**Causal Graph in Language Model Rediscovers Cortical Hierarchy in Human Narrative Processing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.10431v1)  

---


**ABSTRACT**  
Understanding how humans process natural language has long been a vital research direction. The field of natural language processing (NLP) has recently experienced a surge in the development of powerful language models. These models have proven to be invaluable tools for studying another complex system known to process human language: the brain. Previous studies have demonstrated that the features of language models can be mapped to fMRI brain activity. This raises the question: is there a commonality between information processing in language models and the human brain? To estimate information flow patterns in a language model, we examined the causal relationships between different layers. Drawing inspiration from the workspace framework for consciousness, we hypothesized that features integrating more information would more accurately predict higher hierarchical brain activity. To validate this hypothesis, we classified language model features into two categories based on causal network measures: 'low in-degree' and 'high in-degree'. We subsequently compared the brain prediction accuracy maps for these two groups. Our results reveal that the difference in prediction accuracy follows a hierarchical pattern, consistent with the cortical hierarchy map revealed by activity time constants. This finding suggests a parallel between how language models and the human brain process linguistic information.

{{</citation>}}


### (33/60) Bias A-head? Analyzing Bias in Transformer-Based Language Model Attention Heads (Yi Yang et al., 2023)

{{<citation>}}

Yi Yang, Hanyu Duan, Ahmed Abbasi, John P. Lalor, Kar Yan Tam. (2023)  
**Bias A-head? Analyzing Bias in Transformer-Based Language Model Attention Heads**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, BERT, Bias, GPT, Language Model, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2311.10395v1)  

---


**ABSTRACT**  
Transformer-based pretrained large language models (PLM) such as BERT and GPT have achieved remarkable success in NLP tasks. However, PLMs are prone to encoding stereotypical biases. Although a burgeoning literature has emerged on stereotypical bias mitigation in PLMs, such as work on debiasing gender and racial stereotyping, how such biases manifest and behave internally within PLMs remains largely unknown. Understanding the internal stereotyping mechanisms may allow better assessment of model fairness and guide the development of effective mitigation strategies. In this work, we focus on attention heads, a major component of the Transformer architecture, and propose a bias analysis framework to explore and identify a small set of biased heads that are found to contribute to a PLM's stereotypical bias. We conduct extensive experiments to validate the existence of these biased heads and to better understand how they behave. We investigate gender and racial bias in the English language in two types of Transformer-based PLMs: the encoder-based BERT model and the decoder-based autoregressive GPT model. Overall, the results shed light on understanding the bias behavior in pretrained language models.

{{</citation>}}


### (34/60) FOAL: Fine-grained Contrastive Learning for Cross-domain Aspect Sentiment Triplet Extraction (Ting Xu et al., 2023)

{{<citation>}}

Ting Xu, Zhen Wu, Huiyun Yang, Xinyu Dai. (2023)  
**FOAL: Fine-grained Contrastive Learning for Cross-domain Aspect Sentiment Triplet Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.10373v1)  

---


**ABSTRACT**  
Aspect Sentiment Triplet Extraction (ASTE) has achieved promising results while relying on sufficient annotation data in a specific domain. However, it is infeasible to annotate data for each individual domain. We propose to explore ASTE in the cross-domain setting, which transfers knowledge from a resource-rich source domain to a resource-poor target domain, thereby alleviating the reliance on labeled data in the target domain. To effectively transfer the knowledge across domains and extract the sentiment triplets accurately, we propose a method named Fine-grained cOntrAstive Learning (FOAL) to reduce the domain discrepancy and preserve the discriminability of each category. Experiments on six transfer pairs show that FOAL achieves 6% performance gains and reduces the domain discrepancy significantly compared with strong baselines. Our code will be publicly available once accepted.

{{</citation>}}


### (35/60) Exploring the Relationship between In-Context Learning and Instruction Tuning (Hanyu Duan et al., 2023)

{{<citation>}}

Hanyu Duan, Yixuan Tang, Yi Yang, Ahmed Abbasi, Kar Yan Tam. (2023)  
**Exploring the Relationship between In-Context Learning and Instruction Tuning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2311.10367v1)  

---


**ABSTRACT**  
In-Context Learning (ICL) and Instruction Tuning (IT) are two primary paradigms of adopting Large Language Models (LLMs) to downstream applications. However, they are significantly different. In ICL, a set of demonstrations are provided at inference time but the LLM's parameters are not updated. In IT, a set of demonstrations are used to tune LLM's parameters in training time but no demonstrations are used at inference time. Although a growing body of literature has explored ICL and IT, studies on these topics have largely been conducted in isolation, leading to a disconnect between these two paradigms. In this work, we explore the relationship between ICL and IT by examining how the hidden states of LLMs change in these two paradigms. Through carefully designed experiments conducted with LLaMA-2 (7B and 13B), we find that ICL is implicit IT. In other words, ICL changes an LLM's hidden states as if the demonstrations were used to instructionally tune the model. Furthermore, the convergence between ICL and IT is largely contingent upon several factors related to the provided demonstrations. Overall, this work offers a unique perspective to explore the connection between ICL and IT and sheds light on understanding the behaviors of LLM.

{{</citation>}}


### (36/60) Complementary Advantages of ChatGPTs and Human Readers in Reasoning: Evidence from English Text Reading Comprehension (Tongquan Zhou et al., 2023)

{{<citation>}}

Tongquan Zhou, Yao Zhang, Siyi Cao, Yulu Li, Tao Wang. (2023)  
**Complementary Advantages of ChatGPTs and Human Readers in Reasoning: Evidence from English Text Reading Comprehension**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.10344v1)  

---


**ABSTRACT**  
ChatGPT has shown its great power in text processing, including its reasoning ability from text reading. However, there has not been any direct comparison between human readers and ChatGPT in reasoning ability related to text reading. This study was undertaken to investigate how ChatGPTs (i.e., ChatGPT and ChatGPT Plus) and Chinese senior school students as ESL learners exhibited their reasoning ability from English narrative texts. Additionally, we compared the two ChatGPTs in the reasoning performances when commands were updated elaborately. The whole study was composed of three reasoning tests: Test 1 for commonsense inference, Test 2 for emotional inference, and Test 3 for causal inference. The results showed that in Test 1, the students outdid the two ChatGPT versions in local-culture-related inferences but performed worse than the chatbots in daily-life inferences. In Test 2, ChatGPT Plus excelled whereas ChatGPT lagged behind in accuracy. In association with both accuracy and frequency of correct responses, the students were inferior to the two chatbots. Compared with ChatGPTs' better performance in positive emotions, the students showed their superiority in inferring negative emotions. In Test 3, the students demonstrated better logical analysis, outdoing both chatbots. In updating command condition, ChatGPT Plus displayed good causal reasoning ability while ChatGPT kept unchanged. Our study reveals that human readers and ChatGPTs have their respective advantages and disadvantages in drawing inferences from text reading comprehension, unlocking a complementary relationship in text-based reasoning.

{{</citation>}}


### (37/60) Prompt Pool based Class-Incremental Continual Learning for Dialog State Tracking (Hong Liu et al., 2023)

{{<citation>}}

Hong Liu, Yucheng Cai, Yuan Zhou, Zhijian Ou, Yi Huang, Junlan Feng. (2023)  
**Prompt Pool based Class-Incremental Continual Learning for Dialog State Tracking**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog  
[Paper Link](http://arxiv.org/abs/2311.10271v1)  

---


**ABSTRACT**  
Continual learning is crucial for dialog state tracking (DST) in dialog systems, since requirements from users for new functionalities are often encountered. However, most of existing continual learning methods for DST require task identities during testing, which is a severe limit in real-world applications. In this paper, we aim to address continual learning of DST in the class-incremental scenario (namely the task identity is unknown in testing). Inspired by the recently emerging prompt tuning method that performs well on dialog systems, we propose to use the prompt pool method, where we maintain a pool of key-value paired prompts and select prompts from the pool according to the distance between the dialog history and the prompt keys. The proposed method can automatically identify tasks and select appropriate prompts during testing. We conduct experiments on Schema-Guided Dialog dataset (SGD) and another dataset collected from a real-world dialog application. Experiment results show that the prompt pool method achieves much higher joint goal accuracy than the baseline. After combining with a rehearsal buffer, the model performance can be further improved.

{{</citation>}}


### (38/60) Energy and Carbon Considerations of Fine-Tuning BERT (Xiaorong Wang et al., 2023)

{{<citation>}}

Xiaorong Wang, Clara Na, Emma Strubell, Sorelle Friedler, Sasha Luccioni. (2023)  
**Energy and Carbon Considerations of Fine-Tuning BERT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, NLP  
[Paper Link](http://arxiv.org/abs/2311.10267v1)  

---


**ABSTRACT**  
Despite the popularity of the `pre-train then fine-tune' paradigm in the NLP community, existing work quantifying energy costs and associated carbon emissions has largely focused on language model pre-training. Although a single pre-training run draws substantially more energy than fine-tuning, fine-tuning is performed more frequently by many more individual actors, and thus must be accounted for when considering the energy and carbon footprint of NLP. In order to better characterize the role of fine-tuning in the landscape of energy and carbon emissions in NLP, we perform a careful empirical study of the computational costs of fine-tuning across tasks, datasets, hardware infrastructure and measurement modalities. Our experimental results allow us to place fine-tuning energy and carbon costs into perspective with respect to pre-training and inference, and outline recommendations to NLP researchers and practitioners who wish to improve their fine-tuning energy efficiency.

{{</citation>}}


### (39/60) Diagnosing and Debiasing Corpus-Based Political Bias and Insults in GPT2 (Ambri Ma et al., 2023)

{{<citation>}}

Ambri Ma, Arnav Kumar, Brett Zeligson. (2023)  
**Diagnosing and Debiasing Corpus-Based Political Bias and Insults in GPT2**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Bias, GPT  
[Paper Link](http://arxiv.org/abs/2311.10266v1)  

---


**ABSTRACT**  
The training of large language models (LLMs) on extensive, unfiltered corpora sourced from the internet is a common and advantageous practice. Consequently, LLMs have learned and inadvertently reproduced various types of biases, including violent, offensive, and toxic language. However, recent research shows that generative pretrained transformer (GPT) language models can recognize their own biases and detect toxicity in generated content, a process referred to as self-diagnosis. In response, researchers have developed a decoding algorithm that allows LLMs to self-debias, or reduce their likelihood of generating harmful text. This study investigates the efficacy of the diagnosing-debiasing approach in mitigating two additional types of biases: insults and political bias. These biases are often used interchangeably in discourse, despite exhibiting potentially dissimilar semantic and syntactic properties. We aim to contribute to the ongoing effort of investigating the ethical and social implications of human-AI interaction.

{{</citation>}}


## cs.DS (1)



### (40/60) Optimal Embedding Dimension for Sparse Subspace Embeddings (Shabarish Chenakkod et al., 2023)

{{<citation>}}

Shabarish Chenakkod, Michał Dereziński, Xiaoyu Dong, Mark Rudelson. (2023)  
**Optimal Embedding Dimension for Sparse Subspace Embeddings**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs-LG, cs-NA, cs.DS, math-NA, stat-ML  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2311.10680v1)  

---


**ABSTRACT**  
A random $m\times n$ matrix $S$ is an oblivious subspace embedding (OSE) with parameters $\epsilon>0$, $\delta\in(0,1/3)$ and $d\leq m\leq n$, if for any $d$-dimensional subspace $W\subseteq R^n$,   $P\big(\,\forall_{x\in W}\ (1+\epsilon)^{-1}\|x\|\leq\|Sx\|\leq (1+\epsilon)\|x\|\,\big)\geq 1-\delta.$   It is known that the embedding dimension of an OSE must satisfy $m\geq d$, and for any $\theta > 0$, a Gaussian embedding matrix with $m\geq (1+\theta) d$ is an OSE with $\epsilon = O_\theta(1)$. However, such optimal embedding dimension is not known for other embeddings. Of particular interest are sparse OSEs, having $s\ll m$ non-zeros per column, with applications to problems such as least squares regression and low-rank approximation.   We show that, given any $\theta > 0$, an $m\times n$ random matrix $S$ with $m\geq (1+\theta)d$ consisting of randomly sparsified $\pm1/\sqrt s$ entries and having $s= O(\log^4(d))$ non-zeros per column, is an oblivious subspace embedding with $\epsilon = O_{\theta}(1)$. Our result addresses the main open question posed by Nelson and Nguyen (FOCS 2013), who conjectured that sparse OSEs can achieve $m=O(d)$ embedding dimension, and it improves on $m=O(d\log(d))$ shown by Cohen (SODA 2016). We use this to construct the first oblivious subspace embedding with $O(d)$ embedding dimension that can be applied faster than current matrix multiplication time, and to obtain an optimal single-pass algorithm for least squares regression. We further extend our results to construct even sparser non-oblivious embeddings, leading to the first subspace embedding with low distortion $\epsilon=o(1)$ and optimal embedding dimension $m=O(d/\epsilon^2)$ that can be applied in current matrix multiplication time.

{{</citation>}}


## cs.HC (3)



### (41/60) What Lies Beneath? Exploring the Impact of Underlying AI Model Updates in AI-Infused Systems (Vikram Mohanty et al., 2023)

{{<citation>}}

Vikram Mohanty, Jude Lim, Kurt Luther. (2023)  
**What Lies Beneath? Exploring the Impact of Underlying AI Model Updates in AI-Infused Systems**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.10652v1)  

---


**ABSTRACT**  
As AI models evolve, understanding the influence of underlying models on user experience and performance in AI-infused systems becomes critical, particularly while transitioning between different model versions. We studied the influence of model change by conducting two complementary studies in the context of AI-based facial recognition for historical person identification tasks. First, we ran an online experiment where crowd workers interacted with two different facial recognition models: an older version and a recently updated, developer-certified more accurate model. Second, we studied a real-world deployment of these models on a popular historical photo platform through a diary study with 10 users. Our findings sheds light on models affecting human-AI team performance, users' abilities to differentiate between different models, the folk theories they develop, and how these theories influence their preferences. Drawing from these insights, we discuss design implications for updating models in AI-infused systems.

{{</citation>}}


### (42/60) Chatbots as social companions: How people perceive consciousness, human likeness, and social health benefits in machines (Rose Guingrich et al., 2023)

{{<citation>}}

Rose Guingrich, Michael S. A. Graziano. (2023)  
**Chatbots as social companions: How people perceive consciousness, human likeness, and social health benefits in machines**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.10599v1)  

---


**ABSTRACT**  
As artificial intelligence (AI) becomes more widespread, one question that arises is how human-AI interaction might impact human-human interaction. Chatbots, for example, are increasingly used as social companions, but little is known about how their use impacts human relationships. A common hypothesis is that these companion bots are detrimental to social health by harming or replacing human interaction. To understand how companion bots impact social health, we studied people who used companion bots and people who did not. Contrary to expectations, companion bot users indicated that these relationships were beneficial to their social health, whereas nonusers viewed them as harmful. Another common assumption is that people perceive conscious, humanlike AI as disturbing and threatening. Among both users and nonusers, however, we found the opposite: perceiving companion bots as more conscious and humanlike correlated with more positive opinions and better social health benefits. Humanlike bots may aid social health by supplying reliable and safe interactions, without necessarily harming human relationships.

{{</citation>}}


### (43/60) Designing and Evaluating an Adaptive Virtual Reality System using EEG Frequencies to Balance Internal and External Attention States (Francesco Chiossi et al., 2023)

{{<citation>}}

Francesco Chiossi, Changkun Ou, Carolina Gerhardt, Felix Putze, Sven Mayer. (2023)  
**Designing and Evaluating an Adaptive Virtual Reality System using EEG Frequencies to Balance Internal and External Attention States**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2311.10447v1)  

---


**ABSTRACT**  
Virtual reality finds various applications in productivity, entertainment, and training scenarios requiring working memory and attentional resources. Working memory relies on prioritizing relevant information and suppressing irrelevant information through internal attention, which is fundamental for successful task performance and training. Today, virtual reality systems do not account for the impact of working memory loads resulting in over or under-stimulation. In this work, we designed an adaptive system based on EEG correlates of external and internal attention to support working memory task performance. Here, participants engaged in a visual working memory N-Back task, and we adapted the visual complexity of distracting surrounding elements. Our study first demonstrated the feasibility of EEG frontal theta and parietal alpha frequency bands for dynamic visual complexity adjustments. Second, our adaptive system showed improved task performance and diminished perceived workload compared to a reverse adaptation. Our results show the effectiveness of the proposed adaptive system, allowing for the optimization of distracting elements in high-demanding conditions. Adaptive systems based on alpha and theta frequency bands allow for the regulation of attentional and executive resources to keep users engaged in a task without resulting in cognitive overload.

{{</citation>}}


## cs.RO (2)



### (44/60) TacFR-Gripper: A Reconfigurable Fin Ray-Based Compliant Robotic Gripper with Tactile Skin for In-Hand Manipulation (Qingzheng Cong et al., 2023)

{{<citation>}}

Qingzheng Cong, Wen Fan, Dandan Zhang. (2023)  
**TacFR-Gripper: A Reconfigurable Fin Ray-Based Compliant Robotic Gripper with Tactile Skin for In-Hand Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2311.10611v1)  

---


**ABSTRACT**  
This paper introduces the TacFR-Gripper, a reconfigurable Fin Ray-based soft and compliant robotic gripper equipped with tactile skin, which can be used for dexterous in-hand manipulation tasks. This gripper can adaptively grasp objects of diverse shapes and stiffness levels. An array of Force Sensitive Resistor (FSR) sensors is embedded within the robotic finger to serve as the tactile skin, enabling the robot to perceive contact information during manipulation. We provide theoretical analysis for gripper design, including kinematic analysis, workspace analysis, and finite element analysis to identify the relationship between the gripper's load and its deformation. Moreover, we implemented a Graph Neural Network (GNN)-based tactile perception approach to enable reliable grasping without accidental slip or excessive force.   Three physical experiments were conducted to quantify the performance of the TacFR-Gripper. These experiments aimed to i) assess the grasp success rate across various everyday objects through different configurations, ii) verify the effectiveness of tactile skin with the GNN algorithm in grasping, iii) evaluate the gripper's in-hand manipulation capabilities for object pose control. The experimental results indicate that the TacFR-Gripper can grasp a wide range of complex-shaped objects with a high success rate and deliver dexterous in-hand manipulation. Additionally, the integration of tactile skin with the GNN algorithm enhances grasp stability by incorporating tactile feedback during manipulations. For more details of this project, please view our website: https://sites.google.com/view/tacfr-gripper/homepage.

{{</citation>}}


### (45/60) From 'Thumbs Up' to '10 out of 10': Reconsidering Scalar Feedback in Interactive Reinforcement Learning (Hang Yu et al., 2023)

{{<citation>}}

Hang Yu, Reuben M. Aronson, Katherine H. Allen, Elaine Schaertl Short. (2023)  
**From 'Thumbs Up' to '10 out of 10': Reconsidering Scalar Feedback in Interactive Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-HC, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.10284v1)  

---


**ABSTRACT**  
Learning from human feedback is an effective way to improve robotic learning in exploration-heavy tasks. Compared to the wide application of binary human feedback, scalar human feedback has been used less because it is believed to be noisy and unstable. In this paper, we compare scalar and binary feedback, and demonstrate that scalar feedback benefits learning when properly handled. We collected binary or scalar feedback respectively from two groups of crowdworkers on a robot task. We found that when considering how consistently a participant labeled the same data, scalar feedback led to less consistency than binary feedback; however, the difference vanishes if small mismatches are allowed. Additionally, scalar and binary feedback show no significant differences in their correlations with key Reinforcement Learning targets. We then introduce Stabilizing TEacher Assessment DYnamics (STEADY) to improve learning from scalar feedback. Based on the idea that scalar feedback is muti-distributional, STEADY re-constructs underlying positive and negative feedback distributions and re-scales scalar feedback based on feedback statistics. We show that models trained with \textit{scalar feedback + STEADY } outperform baselines, including binary feedback and raw scalar feedback, in a robot reaching task with non-expert human feedback. Our results show that both binary feedback and scalar feedback are dynamic, and scalar feedback is a promising signal for use in interactive Reinforcement Learning.

{{</citation>}}


## physics.flu-dyn (1)



### (46/60) RONAALP: Reduced-Order Nonlinear Approximation with Active Learning Procedure (Clément Scherding et al., 2023)

{{<citation>}}

Clément Scherding, Georgios Rigas, Denis Sipp, Peter J Schmid, Taraneh Sayadi. (2023)  
**RONAALP: Reduced-Order Nonlinear Approximation with Active Learning Procedure**  

---
Primary Category: physics.flu-dyn  
Categories: cs-LG, physics-flu-dyn, physics.flu-dyn  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2311.10550v1)  

---


**ABSTRACT**  
Many engineering applications rely on the evaluation of expensive, non-linear high-dimensional functions. In this paper, we propose the RONAALP algorithm (Reduced Order Nonlinear Approximation with Active Learning Procedure) to incrementally learn a fast and accurate reduced-order surrogate model of a target function on-the-fly as the application progresses. First, the combination of nonlinear auto-encoder, community clustering and radial basis function networks allows to learn an efficient and compact surrogate model with limited training data. Secondly, the active learning procedure overcome any extrapolation issue when evaluating the surrogate model outside of its initial training range during the online stage. This results in generalizable, fast and accurate reduced-order models of high-dimensional functions. The method is demonstrated on three direct numerical simulations of hypersonic flows in chemical nonequilibrium. Accurate simulations of these flows rely on detailed thermochemical gas models that dramatically increase the cost of such calculations. Using RONAALP to learn a reduced-order thermodynamic model surrogate on-the-fly, the cost of such simulation was reduced by up to 75% while maintaining an error of less than 10% on relevant quantities of interest.

{{</citation>}}


## cs.AI (1)



### (47/60) Testing Language Model Agents Safely in the Wild (Silen Naihin et al., 2023)

{{<citation>}}

Silen Naihin, David Atkinson, Marc Green, Merwane Hamadi, Craig Swift, Douglas Schonholtz, Adam Tauman Kalai, David Bau. (2023)  
**Testing Language Model Agents Safely in the Wild**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.10538v1)  

---


**ABSTRACT**  
A prerequisite for safe autonomy-in-the-wild is safe testing-in-the-wild. Yet real-world autonomous tests face several unique safety challenges, both due to the possibility of causing harm during a test, as well as the risk of encountering new unsafe agent behavior through interactions with real-world and potentially malicious actors. We propose a framework for conducting safe autonomous agent tests on the open internet: agent actions are audited by a context-sensitive monitor that enforces a stringent safety boundary to stop an unsafe test, with suspect behavior ranked and logged to be examined by humans. We a design a basic safety monitor that is flexible enough to monitor existing LLM agents, and, using an adversarial simulated agent, we measure its ability to identify and stop unsafe situations. Then we apply the safety monitor on a battery of real-world tests of AutoGPT, and we identify several limitations and challenges that will face the creation of safe in-the-wild tests as autonomous agents grow more capable.

{{</citation>}}


## cs.PL (1)



### (48/60) Towards General Loop Invariant Generation via Coordinating Symbolic Execution and Large Language Models (Chang Liu et al., 2023)

{{<citation>}}

Chang Liu, Xiwei Wu, Yuan Feng, Qinxiang Cao, Junchi Yan. (2023)  
**Towards General Loop Invariant Generation via Coordinating Symbolic Execution and Large Language Models**  

---
Primary Category: cs.PL  
Categories: cs-PL, cs.PL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.10483v1)  

---


**ABSTRACT**  
Loop invariants, essential for program verification, are challenging to auto-generate especially for programs incorporating complex memory manipulations. Existing approaches for generating loop invariants rely on fixed sets or templates, hampering adaptability to real-world programs. Recent efforts have explored machine learning for loop invariant generation, but the lack of labeled data and the need for efficient generation are still troublesome. We consider the advent of the large language model (LLM) presents a promising solution, which can analyze the separation logic assertions after symbolic execution to infer loop invariants. To overcome the data scarcity issue, we propose a self-supervised learning paradigm to fine-tune LLM, using the split-and-reassembly of predicates to create an auxiliary task and generate rich synthetic data for offline training. Meanwhile, the proposed interactive system between LLM and traditional verification tools provides an efficient online querying process for unseen programs. Our framework can readily extend to new data structures or multi-loop programs since our framework only needs the definitions of different separation logic predicates, aiming to bridge the gap between existing capabilities and requirements of loop invariant generation in practical scenarios. Experiments across diverse memory-manipulated programs have demonstrated the performance of our proposed method compared to the baselines with respect to efficiency and effectiveness.

{{</citation>}}


## eess.IV (4)



### (49/60) End-to-end autoencoding architecture for the simultaneous generation of medical images and corresponding segmentation masks (Aghiles Kebaili et al., 2023)

{{<citation>}}

Aghiles Kebaili, Jérôme Lapuyade-Lahorgue, Pierre Vera, Su Ruan. (2023)  
**End-to-end autoencoding architecture for the simultaneous generation of medical images and corresponding segmentation masks**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.10472v1)  

---


**ABSTRACT**  
Despite the increasing use of deep learning in medical image segmentation, acquiring sufficient training data remains a challenge in the medical field. In response, data augmentation techniques have been proposed; however, the generation of diverse and realistic medical images and their corresponding masks remains a difficult task, especially when working with insufficient training sets. To address these limitations, we present an end-to-end architecture based on the Hamiltonian Variational Autoencoder (HVAE). This approach yields an improved posterior distribution approximation compared to traditional Variational Autoencoders (VAE), resulting in higher image generation quality. Our method outperforms generative adversarial architectures under data-scarce conditions, showcasing enhancements in image quality and precise tumor mask synthesis. We conduct experiments on two publicly available datasets, MICCAI's Brain Tumor Segmentation Challenge (BRATS), and Head and Neck Tumor Segmentation Challenge (HECKTOR), demonstrating the effectiveness of our method on different medical imaging modalities.

{{</citation>}}


### (50/60) Pseudo Label-Guided Data Fusion and Output Consistency for Semi-Supervised Medical Image Segmentation (Tao Wang et al., 2023)

{{<citation>}}

Tao Wang, Yuanbin Chen, Xinlin Zhang, Yuanbo Zhou, Junlin Lan, Bizhe Bai, Tao Tan, Min Du, Qinquan Gao, Tong Tong. (2023)  
**Pseudo Label-Guided Data Fusion and Output Consistency for Semi-Supervised Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2311.10349v1)  

---


**ABSTRACT**  
Supervised learning algorithms based on Convolutional Neural Networks have become the benchmark for medical image segmentation tasks, but their effectiveness heavily relies on a large amount of labeled data. However, annotating medical image datasets is a laborious and time-consuming process. Inspired by semi-supervised algorithms that use both labeled and unlabeled data for training, we propose the PLGDF framework, which builds upon the mean teacher network for segmenting medical images with less annotation. We propose a novel pseudo-label utilization scheme, which combines labeled and unlabeled data to augment the dataset effectively. Additionally, we enforce the consistency between different scales in the decoder module of the segmentation network and propose a loss function suitable for evaluating the consistency. Moreover, we incorporate a sharpening operation on the predicted results, further enhancing the accuracy of the segmentation.   Extensive experiments on three publicly available datasets demonstrate that the PLGDF framework can largely improve performance by incorporating the unlabeled data. Meanwhile, our framework yields superior performance compared to six state-of-the-art semi-supervised learning methods. The codes of this study are available at https://github.com/ortonwang/PLGDF.

{{</citation>}}


### (51/60) MPSeg : Multi-Phase strategy for coronary artery Segmentation (Jonghoe Ku et al., 2023)

{{<citation>}}

Jonghoe Ku, Yong-Hee Lee, Junsup Shin, In Kyu Lee, Hyun-Woo Kim. (2023)  
**MPSeg : Multi-Phase strategy for coronary artery Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.10306v1)  

---


**ABSTRACT**  
Accurate segmentation of coronary arteries is a pivotal process in assessing cardiovascular diseases. However, the intricate structure of the cardiovascular system presents significant challenges for automatic segmentation, especially when utilizing methodologies like the SYNTAX Score, which relies extensively on detailed structural information for precise risk stratification. To address these difficulties and cater to this need, we present MPSeg, an innovative multi-phase strategy designed for coronary artery segmentation. Our approach specifically accommodates these structural complexities and adheres to the principles of the SYNTAX Score. Initially, our method segregates vessels into two categories based on their unique morphological characteristics: Left Coronary Artery (LCA) and Right Coronary Artery (RCA). Specialized ensemble models are then deployed for each category to execute the challenging segmentation task. Due to LCA's higher complexity over RCA, a refinement model is utilized to scrutinize and correct initial class predictions on segmented areas. Notably, our approach demonstrated exceptional effectiveness when evaluated in the Automatic Region-based Coronary Artery Disease diagnostics using x-ray angiography imagEs (ARCADE) Segmentation Detection Algorithm challenge at MICCAI 2023.

{{</citation>}}


### (52/60) Semi-supervised ViT knowledge distillation network with style transfer normalization for colorectal liver metastases survival prediction (Mohamed El Amine Elforaici et al., 2023)

{{<citation>}}

Mohamed El Amine Elforaici, Emmanuel Montagnon, Francisco Perdigon Romero, William Trung Le, Feryel Azzi, Dominique Trudel, Bich Nguyen, Simon Turcotte, An Tang, Samuel Kadoury. (2023)  
**Semi-supervised ViT knowledge distillation network with style transfer normalization for colorectal liver metastases survival prediction**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.10305v1)  

---


**ABSTRACT**  
Colorectal liver metastases (CLM) significantly impact colon cancer patients, influencing survival based on systemic chemotherapy response. Traditional methods like tumor grading scores (e.g., tumor regression grade - TRG) for prognosis suffer from subjectivity, time constraints, and expertise demands. Current machine learning approaches often focus on radiological data, yet the relevance of histological images for survival predictions, capturing intricate tumor microenvironment characteristics, is gaining recognition. To address these limitations, we propose an end-to-end approach for automated prognosis prediction using histology slides stained with H&E and HPS. We first employ a Generative Adversarial Network (GAN) for slide normalization to reduce staining variations and improve the overall quality of the images that are used as input to our prediction pipeline. We propose a semi-supervised model to perform tissue classification from sparse annotations, producing feature maps. We use an attention-based approach that weighs the importance of different slide regions in producing the final classification results. We exploit the extracted features for the metastatic nodules and surrounding tissue to train a prognosis model. In parallel, we train a vision Transformer (ViT) in a knowledge distillation framework to replicate and enhance the performance of the prognosis prediction. In our evaluation on a clinical dataset of 258 patients, our approach demonstrates superior performance with c-indexes of 0.804 (0.014) for OS and 0.733 (0.014) for TTR. Achieving 86.9% to 90.3% accuracy in predicting TRG dichotomization and 78.5% to 82.1% accuracy for the 3-class TRG classification task, our approach outperforms comparative methods. Our proposed pipeline can provide automated prognosis for pathologists and oncologists, and can greatly promote precision medicine progress in managing CLM patients.

{{</citation>}}


## cs.CR (2)



### (53/60) A Novel VAPT Algorithm: Enhancing Web Application Security Trough OWASP top 10 Optimization (Rui Ventura et al., 2023)

{{<citation>}}

Rui Ventura, Daniel Jose Franco, Omar Khasro Akram. (2023)  
**A Novel VAPT Algorithm: Enhancing Web Application Security Trough OWASP top 10 Optimization**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.10450v1)  

---


**ABSTRACT**  
This research study is built upon cybersecurity audits and investigates the optimization of an Open Web Application Security Project (OWASP) Top 10 algorithm for Web Applications (WA) security audits using Vulnerability Assessment and Penetration Testing (VAPT) processes. The study places particular emphasis on enhancing the VAPT process by optimizing the OWASP algorithm. To achieve this, the research utilizes desk documents to gain knowledge of WA cybersecurity audits and their associated tools. It also delves into archives to explore VAPT processes and identify techniques, methods, and tools for VAPT automation. Furthermore, the research proposes a prototype optimization that streamlines the two steps of VAPT using the OWASP Top 10 algorithm through an experimental procedure. The results are obtained within a virtual environment, which employs black box testing methods as the primary means of data acquisition and analysis. In this experimental setting, the OWASP algorithm demonstrates an impressive level of precision, achieving a precision rate exceeding 90%. It effectively covers all researched vulnerabilities, thus justifying its optimization. This research contributes significantly to the enhancement of the OWASP algorithm and benefits the offensive security community. It plays a crucial role in ensuring compliance processes for professionals and analysts in the security and software development fields.

{{</citation>}}


### (54/60) Towards Stronger Blockchains: Security Against Front-Running Attacks (Anshuman Misra et al., 2023)

{{<citation>}}

Anshuman Misra, Ajay D. Kshemkalyani. (2023)  
**Towards Stronger Blockchains: Security Against Front-Running Attacks**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-DC, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.10253v1)  

---


**ABSTRACT**  
Blockchains add transactions to a distributed shared ledger by arriving at consensus on sets of transactions contained in blocks. This provides a total ordering on a set of global transactions. However, total ordering is not enough to satisfy application semantics under the Byzantine fault model. This is due to the fact that malicious miners and clients can collaborate to add their own transactions ahead of correct clients' transactions in order to gain application level and financial advantages. These attacks fall under the umbrella of front-running attacks. Therefore, total ordering is not strong enough to preserve application semantics. In this paper, we propose causality preserving total order as a solution to this problem. The resulting Blockchains will be stronger than traditional consensus based blockchains and will provide enhanced security ensuring correct application semantics in a Byzantine setting.

{{</citation>}}


## cs.DC (1)



### (55/60) DynaPipe: Optimizing Multi-task Training through Dynamic Pipelines (Chenyu Jiang et al., 2023)

{{<citation>}}

Chenyu Jiang, Zhen Jia, Shuai Zheng, Yida Wang, Chuan Wu. (2023)  
**DynaPipe: Optimizing Multi-task Training through Dynamic Pipelines**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-LG, cs.DC  
Keywords: GPT, T5  
[Paper Link](http://arxiv.org/abs/2311.10418v1)  

---


**ABSTRACT**  
Multi-task model training has been adopted to enable a single deep neural network model (often a large language model) to handle multiple tasks (e.g., question answering and text summarization). Multi-task training commonly receives input sequences of highly different lengths due to the diverse contexts of different tasks. Padding (to the same sequence length) or packing (short examples into long sequences of the same length) is usually adopted to prepare input samples for model training, which is nonetheless not space or computation efficient. This paper proposes a dynamic micro-batching approach to tackle sequence length variation and enable efficient multi-task model training. We advocate pipeline-parallel training of the large model with variable-length micro-batches, each of which potentially comprises a different number of samples. We optimize micro-batch construction using a dynamic programming-based approach, and handle micro-batch execution time variation through dynamic pipeline and communication scheduling, enabling highly efficient pipeline training. Extensive evaluation on the FLANv2 dataset demonstrates up to 4.39x higher training throughput when training T5, and 3.25x when training GPT, as compared with packing-based baselines. DynaPipe's source code is publicly available at https://github.com/awslabs/optimizing-multitask-training-through-dynamic-pipelines.

{{</citation>}}


## cs.NI (1)



### (56/60) Decentralized Energy Marketplace via NFTs and AI-based Agents (Rasoul Nikbakht et al., 2023)

{{<citation>}}

Rasoul Nikbakht, Farhana Javed, Farhad Rezazadeh, Nikolaos Bartzoudis, Josep Mangues-Bafalluy. (2023)  
**Decentralized Energy Marketplace via NFTs and AI-based Agents**  

---
Primary Category: cs.NI  
Categories: cs-LG, cs-NI, cs.NI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.10406v1)  

---


**ABSTRACT**  
The paper introduces an advanced Decentralized Energy Marketplace (DEM) integrating blockchain technology and artificial intelligence to manage energy exchanges among smart homes with energy storage systems. The proposed framework uses Non-Fungible Tokens (NFTs) to represent unique energy profiles in a transparent and secure trading environment. Leveraging Federated Deep Reinforcement Learning (FDRL), the system promotes collaborative and adaptive energy management strategies, maintaining user privacy. A notable innovation is the use of smart contracts, ensuring high efficiency and integrity in energy transactions. Extensive evaluations demonstrate the system's scalability and the effectiveness of the FDRL method in optimizing energy distribution. This research significantly contributes to developing sophisticated decentralized smart grid infrastructures. Our approach broadens potential blockchain and AI applications in sustainable energy systems and addresses incentive alignment and transparency challenges in traditional energy trading mechanisms. The implementation of this paper is publicly accessible at \url{https://github.com/RasoulNik/DEM}.

{{</citation>}}


## cs.SE (2)



### (57/60) Automatic Smart Contract Comment Generation via Large Language Models and In-Context Learning (Junjie Zhao et al., 2023)

{{<citation>}}

Junjie Zhao, Xiang Chen, Guang Yang, Yiheng Shen. (2023)  
**Automatic Smart Contract Comment Generation via Large Language Models and In-Context Learning**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.10388v1)  

---


**ABSTRACT**  
The previous smart contract code comment (SCC) generation approaches can be divided into two categories: fine-tuning paradigm-based approaches and information retrieval-based approaches. However, for the fine-tuning paradigm-based approaches, the performance may be limited by the quality of the gathered dataset for the downstream task and they may have knowledge-forgetting issues. While for the information retrieval-based approaches, it is difficult for them to generate high-quality comments if similar code does not exist in the historical repository. Therefore we want to utilize the domain knowledge related to SCC generation in large language models (LLMs) to alleviate the disadvantages of these two types of approaches. In this study, we propose an approach SCCLLM based on LLMs and in-context learning. Specifically, in the demonstration selection phase, SCCLLM retrieves the top-k code snippets from the historical corpus by considering syntax, semantics, and lexical information. In the in-context learning phase, SCCLLM utilizes the retrieved code snippets as demonstrations, which can help to utilize the related knowledge for this task. We select a large corpus from a smart contract community Etherscan.io as our experimental subject. Extensive experimental results show the effectiveness of SCCLLM when compared with baselines in automatic evaluation and human evaluation.

{{</citation>}}


### (58/60) A Survey of Large Language Models for Code: Evolution, Benchmarking, and Future Trends (Zibin Zheng et al., 2023)

{{<citation>}}

Zibin Zheng, Kaiwen Ning, Yanlin Wang, Jingwen Zhang, Dewu Zheng, Mingxi Ye, Jiachi Chen. (2023)  
**A Survey of Large Language Models for Code: Evolution, Benchmarking, and Future Trends**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.10372v1)  

---


**ABSTRACT**  
General large language models (LLMs), represented by ChatGPT, have demonstrated significant potential in tasks such as code generation in software engineering. This has led to the development of specialized LLMs for software engineering, known as Code LLMs. A considerable portion of Code LLMs is derived from general LLMs through model fine-tuning. As a result, Code LLMs are often updated frequently and their performance can be influenced by the base LLMs. However, there is currently a lack of systematic investigation into Code LLMs and their performance. In this study, we conduct a comprehensive survey and analysis of the types of Code LLMs and their differences in performance compared to general LLMs. We aim to address three questions: (1) What LLMs are specifically designed for software engineering tasks, and what is the relationship between these Code LLMs? (2) Do Code LLMs really outperform general LLMs in software engineering tasks? (3) Which LLMs are more proficient in different software engineering tasks? To answer these questions, we first collect relevant literature and work from five major databases and open-source communities, resulting in 134 works for analysis. Next, we categorize the Code LLMs based on their publishers and examine their relationships with general LLMs and among themselves. Furthermore, we investigate the performance differences between general LLMs and Code LLMs in various software engineering tasks to demonstrate the impact of base models and Code LLMs. Finally, we comprehensively maintained the performance of LLMs across multiple mainstream benchmarks to identify the best-performing LLMs for each software engineering task. Our research not only assists developers of Code LLMs in choosing base models for the development of more advanced LLMs but also provides insights for practitioners to better understand key improvement directions for Code LLMs.

{{</citation>}}


## cs.IR (1)



### (59/60) A Comparative Analysis of Retrievability and PageRank Measures (Aman Sinha et al., 2023)

{{<citation>}}

Aman Sinha, Priyanshu Raj Mall, Dwaipayan Roy. (2023)  
**A Comparative Analysis of Retrievability and PageRank Measures**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2311.10348v1)  

---


**ABSTRACT**  
The accessibility of documents within a collection holds a pivotal role in Information Retrieval, signifying the ease of locating specific content in a collection of documents. This accessibility can be achieved via two distinct avenues. The first is through some retrieval model using a keyword or other feature-based search, and the other is where a document can be navigated using links associated with them, if available. Metrics such as PageRank, Hub, and Authority illuminate the pathways through which documents can be discovered within the network of content while the concept of Retrievability is used to quantify the ease with which a document can be found by a retrieval model. In this paper, we compare these two perspectives, PageRank and retrievability, as they quantify the importance and discoverability of content in a corpus. Through empirical experimentation on benchmark datasets, we demonstrate a subtle similarity between retrievability and PageRank particularly distinguishable for larger datasets.

{{</citation>}}


## cs.AR (1)



### (60/60) Improving FSM State Enumeration Performance for Hardware Security with RECUT and REFSM-SAT (Jim Geist et al., 2023)

{{<citation>}}

Jim Geist, Travis Meade, Shaojie Zhang, Yier Jin. (2023)  
**Improving FSM State Enumeration Performance for Hardware Security with RECUT and REFSM-SAT**  

---
Primary Category: cs.AR  
Categories: 68U07, cs-AR, cs.AR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2311.10273v1)  

---


**ABSTRACT**  
Finite state machines (FSM's) are implemented with sequential circuits and are used to orchestrate the operation of hardware designs. Sequential obfuscation schemes aimed at preventing IP theft often operate by augmenting a design's FSM post-synthesis. Many such schemes are based on the ability to recover the FSM's topology from the synthesized design. In this paper, we present two tools which can improve the performance of topology extraction: RECUT, which extracts the FSM implementation from a netlist, and REFSM-SAT, which solves topology enumeration as a series of SAT problems. In some cases, these tools can improve performance significantly over current methods, attaining up to a 99\% decrease in runtime.

{{</citation>}}
