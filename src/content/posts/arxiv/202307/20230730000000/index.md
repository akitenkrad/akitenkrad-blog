---
draft: false
title: "arXiv @ 2023.07.30"
date: 2023-07-30
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.30"
    identifier: arxiv_20230730
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (17)](#cscv-17)
- [cs.CL (18)](#cscl-18)
- [cs.LG (6)](#cslg-6)
- [eess.IV (2)](#eessiv-2)
- [cs.SI (3)](#cssi-3)
- [cs.RO (2)](#csro-2)
- [cs.MA (1)](#csma-1)
- [cs.AR (1)](#csar-1)
- [cs.SD (1)](#cssd-1)
- [cs.HC (1)](#cshc-1)
- [cs.MM (1)](#csmm-1)
- [cs.IT (2)](#csit-2)
- [eess.SY (1)](#eesssy-1)
- [cs.SE (1)](#csse-1)
- [cs.NE (1)](#csne-1)
- [eess.AS (1)](#eessas-1)

## cs.CV (17)



### (1/59) Semi-Supervised Object Detection in the Open World (Garvita Allabadi et al., 2023)

{{<citation>}}

Garvita Allabadi, Ana Lucic, Peter Pao-Huang, Yu-Xiong Wang, Vikram Adve. (2023)  
**Semi-Supervised Object Detection in the Open World**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2307.15710v1)  

---


**ABSTRACT**  
Existing approaches for semi-supervised object detection assume a fixed set of classes present in training and unlabeled datasets, i.e., in-distribution (ID) data. The performance of these techniques significantly degrades when these techniques are deployed in the open-world, due to the fact that the unlabeled and test data may contain objects that were not seen during training, i.e., out-of-distribution (OOD) data. The two key questions that we explore in this paper are: can we detect these OOD samples and if so, can we learn from them? With these considerations in mind, we propose the Open World Semi-supervised Detection framework (OWSSD) that effectively detects OOD data along with a semi-supervised learning pipeline that learns from both ID and OOD data. We introduce an ensemble based OOD detector consisting of lightweight auto-encoder networks trained only on ID data. Through extensive evalulation, we demonstrate that our method performs competitively against state-of-the-art OOD detection algorithms and also significantly boosts the semi-supervised learning performance in open-world scenarios.

{{</citation>}}


### (2/59) MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking (Ruopeng Gao et al., 2023)

{{<citation>}}

Ruopeng Gao, Limin Wang. (2023)  
**MeMOTR: Long-Term Memory-Augmented Transformer for Multi-Object Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.15700v1)  

---


**ABSTRACT**  
As a video task, Multi-Object Tracking (MOT) is expected to capture temporal information of targets effectively. Unfortunately, most existing methods only explicitly exploit the object features between adjacent frames, while lacking the capacity to model long-term temporal information. In this paper, we propose MeMOTR, a long-term memory-augmented Transformer for multi-object tracking. Our method is able to make the same object's track embedding more stable and distinguishable by leveraging long-term memory injection with a customized memory-attention layer. This significantly improves the target association ability of our model. Experimental results on DanceTrack show that MeMOTR impressively surpasses the state-of-the-art method by 7.9\% and 13.0\% on HOTA and AssA metrics, respectively. Furthermore, our model also outperforms other Transformer-based methods on association performance on MOT17 and generalizes well on BDD100K. Code is available at \href{https://github.com/MCG-NJU/MeMOTR}{https://github.com/MCG-NJU/MeMOTR}.

{{</citation>}}


### (3/59) TrackAgent: 6D Object Tracking via Reinforcement Learning (Konstantin Röhrl et al., 2023)

{{<citation>}}

Konstantin Röhrl, Dominik Bauer, Timothy Patten, Markus Vincze. (2023)  
**TrackAgent: 6D Object Tracking via Reinforcement Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.15671v1)  

---


**ABSTRACT**  
Tracking an object's 6D pose, while either the object itself or the observing camera is moving, is important for many robotics and augmented reality applications. While exploiting temporal priors eases this problem, object-specific knowledge is required to recover when tracking is lost. Under the tight time constraints of the tracking task, RGB(D)-based methods are often conceptionally complex or rely on heuristic motion models. In comparison, we propose to simplify object tracking to a reinforced point cloud (depth only) alignment task. This allows us to train a streamlined approach from scratch with limited amounts of sparse 3D point clouds, compared to the large datasets of diverse RGBD sequences required in previous works. We incorporate temporal frame-to-frame registration with object-based recovery by frame-to-model refinement using a reinforcement learning (RL) agent that jointly solves for both objectives. We also show that the RL agent's uncertainty and a rendering-based mask propagation are effective reinitialization triggers.

{{</citation>}}


### (4/59) OAFuser: Towards Omni-Aperture Fusion for Light Field Semantic Segmentation of Road Scenes (Fei Teng et al., 2023)

{{<citation>}}

Fei Teng, Jiaming Zhang, Kunyu Peng, Kailun Yang, Yaonan Wang, Rainer Stiefelhagen. (2023)  
**OAFuser: Towards Omni-Aperture Fusion for Light Field Semantic Segmentation of Road Scenes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV, eess-IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.15588v1)  

---


**ABSTRACT**  
Light field cameras can provide rich angular and spatial information to enhance image semantic segmentation for scene understanding in the field of autonomous driving. However, the extensive angular information of light field cameras contains a large amount of redundant data, which is overwhelming for the limited hardware resource of intelligent vehicles. Besides, inappropriate compression leads to information corruption and data loss. To excavate representative information, we propose an Omni-Aperture Fusion model (OAFuser), which leverages dense context from the central view and discovers the angular information from sub-aperture images to generate a semantically-consistent result. To avoid feature loss during network propagation and simultaneously streamline the redundant information from the light field camera, we present a simple yet very effective Sub-Aperture Fusion Module (SAFM) to embed sub-aperture images into angular features without any additional memory cost. Furthermore, to address the mismatched spatial information across viewpoints, we present Center Angular Rectification Module (CARM) realized feature resorting and prevent feature occlusion caused by asymmetric information. Our proposed OAFuser achieves state-of-the-art performance on the UrbanLF-Real and -Syn datasets and sets a new record of 84.93% in mIoU on the UrbanLF-Real Extended dataset, with a gain of +4.53%. The source code of OAFuser will be made publicly available at https://github.com/FeiBryantkit/OAFuser.

{{</citation>}}


### (5/59) Point Clouds Are Specialized Images: A Knowledge Transfer Approach for 3D Understanding (Jiachen Kang et al., 2023)

{{<citation>}}

Jiachen Kang, Wenjing Jia, Xiangjian He, Kin Man Lam. (2023)  
**Point Clouds Are Specialized Images: A Knowledge Transfer Approach for 3D Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.15569v1)  

---


**ABSTRACT**  
Self-supervised representation learning (SSRL) has gained increasing attention in point cloud understanding, in addressing the challenges posed by 3D data scarcity and high annotation costs. This paper presents PCExpert, a novel SSRL approach that reinterprets point clouds as "specialized images". This conceptual shift allows PCExpert to leverage knowledge derived from large-scale image modality in a more direct and deeper manner, via extensively sharing the parameters with a pre-trained image encoder in a multi-way Transformer architecture. The parameter sharing strategy, combined with a novel pretext task for pre-training, i.e., transformation estimation, empowers PCExpert to outperform the state of the arts in a variety of tasks, with a remarkable reduction in the number of trainable parameters. Notably, PCExpert's performance under LINEAR fine-tuning (e.g., yielding a 90.02% overall accuracy on ScanObjectNN) has already approached the results obtained with FULL model fine-tuning (92.66%), demonstrating its effective and robust representation capability.

{{</citation>}}


### (6/59) Panoptic Scene Graph Generation with Semantics-prototype Learning (Li Li et al., 2023)

{{<citation>}}

Li Li, Wei Ji, Yiming Wu, Mengze Li, You Qin, Lina Wei, Roger Zimmermann. (2023)  
**Panoptic Scene Graph Generation with Semantics-prototype Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.15567v1)  

---


**ABSTRACT**  
Panoptic Scene Graph Generation (PSG) parses objects and predicts their relationships (predicate) to connect human language and visual scenes. However, different language preferences of annotators and semantic overlaps between predicates lead to biased predicate annotations in the dataset, i.e. different predicates for same object pairs. Biased predicate annotations make PSG models struggle in constructing a clear decision plane among predicates, which greatly hinders the real application of PSG models. To address the intrinsic bias above, we propose a novel framework named ADTrans to adaptively transfer biased predicate annotations to informative and unified ones. To promise consistency and accuracy during the transfer process, we propose to measure the invariance of representations in each predicate class, and learn unbiased prototypes of predicates with different intensities. Meanwhile, we continuously measure the distribution changes between each presentation and its prototype, and constantly screen potential biased data. Finally, with the unbiased predicate-prototype representation embedding space, biased annotations are easily identified. Experiments show that ADTrans significantly improves the performance of benchmark models, achieving a new state-of-the-art performance, and shows great generalization and effectiveness on multiple datasets.

{{</citation>}}


### (7/59) Few-shot Image Classification based on Gradual Machine Learning (Na Chen et al., 2023)

{{<citation>}}

Na Chen, Xianming Kuang, Feiyu Liu, Kehao Wang, Qun Chen. (2023)  
**Few-shot Image Classification based on Gradual Machine Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2307.15524v1)  

---


**ABSTRACT**  
Few-shot image classification aims to accurately classify unlabeled images using only a few labeled samples. The state-of-the-art solutions are built by deep learning, which focuses on designing increasingly complex deep backbones. Unfortunately, the task remains very challenging due to the difficulty of transferring the knowledge learned in training classes to new ones. In this paper, we propose a novel approach based on the non-i.i.d paradigm of gradual machine learning (GML). It begins with only a few labeled observations, and then gradually labels target images in the increasing order of hardness by iterative factor inference in a factor graph. Specifically, our proposed solution extracts indicative feature representations by deep backbones, and then constructs both unary and binary factors based on the extracted features to facilitate gradual learning. The unary factors are constructed based on class center distance in an embedding space, while the binary factors are constructed based on k-nearest neighborhood. We have empirically validated the performance of the proposed approach on benchmark datasets by a comparative study. Our extensive experiments demonstrate that the proposed approach can improve the SOTA performance by 1-5% in terms of accuracy. More notably, it is more robust than the existing deep models in that its performance can consistently improve as the size of query set increases while the performance of deep models remains essentially flat or even becomes worse.

{{</citation>}}


### (8/59) Cross-Modal Concept Learning and Inference for Vision-Language Models (Yi Zhang et al., 2023)

{{<citation>}}

Yi Zhang, Ce Zhang, Yushun Tang, Zhihai He. (2023)  
**Cross-Modal Concept Learning and Inference for Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.15460v1)  

---


**ABSTRACT**  
Large-scale pre-trained Vision-Language Models (VLMs), such as CLIP, establish the correlation between texts and images, achieving remarkable success on various downstream tasks with fine-tuning. In existing fine-tuning methods, the class-specific text description is matched against the whole image. We recognize that this whole image matching is not effective since images from the same class often contain a set of different semantic objects, and an object further consists of a set of semantic parts or concepts. Individual semantic parts or concepts may appear in image samples from different classes. To address this issue, in this paper, we develop a new method called cross-model concept learning and inference (CCLI). Using the powerful text-image correlation capability of CLIP, our method automatically learns a large set of distinctive visual concepts from images using a set of semantic text concepts. Based on these visual concepts, we construct a discriminative representation of images and learn a concept inference network to perform downstream image classification tasks, such as few-shot learning and domain generalization. Extensive experimental results demonstrate that our CCLI method is able to improve the performance upon the current state-of-the-art methods by large margins, for example, by up to 8.0% improvement on few-shot learning and by up to 1.3% for domain generalization.

{{</citation>}}


### (9/59) Uncertainty-aware Unsupervised Multi-Object Tracking (Kai Liu et al., 2023)

{{<citation>}}

Kai Liu, Sheng Jin, Zhihang Fu, Ze Chen, Rongxin Jiang, Jieping Ye. (2023)  
**Uncertainty-aware Unsupervised Multi-Object Tracking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2307.15409v1)  

---


**ABSTRACT**  
Without manually annotated identities, unsupervised multi-object trackers are inferior to learning reliable feature embeddings. It causes the similarity-based inter-frame association stage also be error-prone, where an uncertainty problem arises. The frame-by-frame accumulated uncertainty prevents trackers from learning the consistent feature embedding against time variation. To avoid this uncertainty problem, recent self-supervised techniques are adopted, whereas they failed to capture temporal relations. The interframe uncertainty still exists. In fact, this paper argues that though the uncertainty problem is inevitable, it is possible to leverage the uncertainty itself to improve the learned consistency in turn. Specifically, an uncertainty-based metric is developed to verify and rectify the risky associations. The resulting accurate pseudo-tracklets boost learning the feature consistency. And accurate tracklets can incorporate temporal information into spatial transformation. This paper proposes a tracklet-guided augmentation strategy to simulate tracklets' motion, which adopts a hierarchical uncertainty-based sampling mechanism for hard sample mining. The ultimate unsupervised MOT framework, namely U2MOT, is proven effective on MOT-Challenges and VisDrone-MOT benchmark. U2MOT achieves a SOTA performance among the published supervised and unsupervised trackers.

{{</citation>}}


### (10/59) Prompt Guided Transformer for Multi-Task Dense Prediction (Yuxiang Lu et al., 2023)

{{<citation>}}

Yuxiang Lu, Shalayiding Sirejiding, Yue Ding, Chunlin Wang, Hongtao Lu. (2023)  
**Prompt Guided Transformer for Multi-Task Dense Prediction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.15362v1)  

---


**ABSTRACT**  
Task-conditional architecture offers advantage in parameter efficiency but falls short in performance compared to state-of-the-art multi-decoder methods. How to trade off performance and model parameters is an important and difficult problem. In this paper, we introduce a simple and lightweight task-conditional model called Prompt Guided Transformer (PGT) to optimize this challenge. Our approach designs a Prompt-conditioned Transformer block, which incorporates task-specific prompts in the self-attention mechanism to achieve global dependency modeling and parameter-efficient feature adaptation across multiple tasks. This block is integrated into both the shared encoder and decoder, enhancing the capture of intra- and inter-task features. Moreover, we design a lightweight decoder to further reduce parameter usage, which accounts for only 2.7% of the total model parameters. Extensive experiments on two multi-task dense prediction benchmarks, PASCAL-Context and NYUD-v2, demonstrate that our approach achieves state-of-the-art results among task-conditional methods while using fewer parameters, and maintains a significant balance between performance and parameter size.

{{</citation>}}


### (11/59) Dynamic PlenOctree for Adaptive Sampling Refinement in Explicit NeRF (Haotian Bai et al., 2023)

{{<citation>}}

Haotian Bai, Yiqi Lin, Yize Chen, Lin Wang. (2023)  
**Dynamic PlenOctree for Adaptive Sampling Refinement in Explicit NeRF**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2307.15333v1)  

---


**ABSTRACT**  
The explicit neural radiance field (NeRF) has gained considerable interest for its efficient training and fast inference capabilities, making it a promising direction such as virtual reality and gaming. In particular, PlenOctree (POT)[1], an explicit hierarchical multi-scale octree representation, has emerged as a structural and influential framework. However, POT's fixed structure for direct optimization is sub-optimal as the scene complexity evolves continuously with updates to cached color and density, necessitating refining the sampling distribution to capture signal complexity accordingly. To address this issue, we propose the dynamic PlenOctree DOT, which adaptively refines the sample distribution to adjust to changing scene complexity. Specifically, DOT proposes a concise yet novel hierarchical feature fusion strategy during the iterative rendering process. Firstly, it identifies the regions of interest through training signals to ensure adaptive and efficient refinement. Next, rather than directly filtering out valueless nodes, DOT introduces the sampling and pruning operations for octrees to aggregate features, enabling rapid parameter learning. Compared with POT, our DOT outperforms it by enhancing visual quality, reducing over $55.15$/$68.84\%$ parameters, and providing 1.7/1.9 times FPS for NeRF-synthetic and Tanks $\&$ Temples, respectively. Project homepage:https://vlislab22.github.io/DOT.   [1] Yu, Alex, et al. "Plenoctrees for real-time rendering of neural radiance fields." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

{{</citation>}}


### (12/59) TaskExpert: Dynamically Assembling Multi-Task Representations with Memorial Mixture-of-Experts (Hanrong Ye et al., 2023)

{{<citation>}}

Hanrong Ye, Dan Xu. (2023)  
**TaskExpert: Dynamically Assembling Multi-Task Representations with Memorial Mixture-of-Experts**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.15324v1)  

---


**ABSTRACT**  
Learning discriminative task-specific features simultaneously for multiple distinct tasks is a fundamental problem in multi-task learning. Recent state-of-the-art models consider directly decoding task-specific features from one shared task-generic feature (e.g., feature from a backbone layer), and utilize carefully designed decoders to produce multi-task features. However, as the input feature is fully shared and each task decoder also shares decoding parameters for different input samples, it leads to a static feature decoding process, producing less discriminative task-specific representations. To tackle this limitation, we propose TaskExpert, a novel multi-task mixture-of-experts model that enables learning multiple representative task-generic feature spaces and decoding task-specific features in a dynamic manner. Specifically, TaskExpert introduces a set of expert networks to decompose the backbone feature into several representative task-generic features. Then, the task-specific features are decoded by using dynamic task-specific gating networks operating on the decomposed task-generic features. Furthermore, to establish long-range modeling of the task-specific representations from different layers of TaskExpert, we design a multi-task feature memory that updates at each layer and acts as an additional feature expert for dynamic task-specific feature decoding. Extensive experiments demonstrate that our TaskExpert clearly outperforms previous best-performing methods on all 9 metrics of two competitive multi-task learning benchmarks for visual scene understanding (i.e., PASCAL-Context and NYUD-v2). Codes and models will be made publicly available at https://github.com/prismformore/Multi-Task-Transformer

{{</citation>}}


### (13/59) DocDeshadower: Frequency-aware Transformer for Document Shadow Removal (Shenghong Luo et al., 2023)

{{<citation>}}

Shenghong Luo, Ruifeng Xu, Xuhang Chen, Zinuo Li, Chi-Man Pun, Shuqiang Wang. (2023)  
**DocDeshadower: Frequency-aware Transformer for Document Shadow Removal**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2307.15318v1)  

---


**ABSTRACT**  
The presence of shadows significantly impacts the visual quality of scanned documents. However, the existing traditional techniques and deep learning methods used for shadow removal have several limitations. These methods either rely heavily on heuristics, resulting in suboptimal performance, or require large datasets to learn shadow-related features. In this study, we propose the DocDeshadower, a multi-frequency Transformer-based model built on Laplacian Pyramid. DocDeshadower is designed to remove shadows at different frequencies in a coarse-to-fine manner. To achieve this, we decompose the shadow image into different frequency bands using Laplacian Pyramid. In addition, we introduce two novel components to this model: the Attention-Aggregation Network and the Gated Multi-scale Fusion Transformer. The Attention-Aggregation Network is designed to remove shadows in the low-frequency part of the image, whereas the Gated Multi-scale Fusion Transformer refines the entire image at a global scale with its large perceptive field. Our extensive experiments demonstrate that DocDeshadower outperforms the current state-of-the-art methods in both qualitative and quantitative terms.

{{</citation>}}


### (14/59) DiffKendall: A Novel Approach for Few-Shot Learning with Differentiable Kendall's Rank Correlation (Kaipeng Zheng et al., 2023)

{{<citation>}}

Kaipeng Zheng, Huishuai Zhang, Weiran Huang. (2023)  
**DiffKendall: A Novel Approach for Few-Shot Learning with Differentiable Kendall's Rank Correlation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-MM, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2307.15317v1)  

---


**ABSTRACT**  
Few-shot learning aims to adapt models trained on the base dataset to novel tasks where the categories are not seen by the model before. This often leads to a relatively uniform distribution of feature values across channels on novel classes, posing challenges in determining channel importance for novel tasks. Standard few-shot learning methods employ geometric similarity metrics such as cosine similarity and negative Euclidean distance to gauge the semantic relatedness between two features. However, features with high geometric similarities may carry distinct semantics, especially in the context of few-shot learning. In this paper, we demonstrate that the importance ranking of feature channels is a more reliable indicator for few-shot learning than geometric similarity metrics. We observe that replacing the geometric similarity metric with Kendall's rank correlation only during inference is able to improve the performance of few-shot learning across a wide range of datasets with different domains. Furthermore, we propose a carefully designed differentiable loss for meta-training to address the non-differentiability issue of Kendall's rank correlation. Extensive experiments demonstrate that the proposed rank-correlation-based approach substantially enhances few-shot learning performance.

{{</citation>}}


### (15/59) RSGPT: A Remote Sensing Vision Language Model and Benchmark (Yuan Hu et al., 2023)

{{<citation>}}

Yuan Hu, Jianlong Yuan, Congcong Wen, Xiaonan Lu, Xiang Li. (2023)  
**RSGPT: A Remote Sensing Vision Language Model and Benchmark**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4, Image Captioning, Language Model  
[Paper Link](http://arxiv.org/abs/2307.15266v1)  

---


**ABSTRACT**  
The emergence of large-scale large language models, with GPT-4 as a prominent example, has significantly propelled the rapid advancement of artificial general intelligence and sparked the revolution of Artificial Intelligence 2.0. In the realm of remote sensing (RS), there is a growing interest in developing large vision language models (VLMs) specifically tailored for data analysis in this domain. However, current research predominantly revolves around visual recognition tasks, lacking comprehensive, large-scale image-text datasets that are aligned and suitable for training large VLMs, which poses significant challenges to effectively training such models for RS applications. In computer vision, recent research has demonstrated that fine-tuning large vision language models on small-scale, high-quality datasets can yield impressive performance in visual and language understanding. These results are comparable to state-of-the-art VLMs trained from scratch on massive amounts of data, such as GPT-4. Inspired by this captivating idea, in this work, we build a high-quality Remote Sensing Image Captioning dataset (RSICap) that facilitates the development of large VLMs in the RS field. Unlike previous RS datasets that either employ model-generated captions or short descriptions, RSICap comprises 2,585 human-annotated captions with rich and high-quality information. This dataset offers detailed descriptions for each image, encompassing scene descriptions (e.g., residential area, airport, or farmland) as well as object information (e.g., color, shape, quantity, absolute position, etc). To facilitate the evaluation of VLMs in the field of RS, we also provide a benchmark evaluation dataset called RSIEval. This dataset consists of human-annotated captions and visual question-answer pairs, allowing for a comprehensive assessment of VLMs in the context of RS.

{{</citation>}}


### (16/59) Multiple Instance Learning Framework with Masked Hard Instance Mining for Whole Slide Image Classification (Wenhao Tang et al., 2023)

{{<citation>}}

Wenhao Tang, Sheng Huang, Xiaoxian Zhang, Fengtao Zhou, Yi Zhang, Bo Liu. (2023)  
**Multiple Instance Learning Framework with Masked Hard Instance Mining for Whole Slide Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2307.15254v1)  

---


**ABSTRACT**  
The whole slide image (WSI) classification is often formulated as a multiple instance learning (MIL) problem. Since the positive tissue is only a small fraction of the gigapixel WSI,existing MIL methods intuitively focus on identifying salient instances via attention mechanisms. However, this leads to a bias towards easy-to-classify instances while neglecting hard-to-classify instances.Some literature has revealed that hard examples are beneficial for modeling a discriminative boundary accurately.By applying such an idea at the instance level,we elaborate a novel MIL framework with masked hard instance mining (MHIM-MIL), which uses a Siamese structure (Teacher-Student) with a consistency constraint to explore the potential hard instances. With several instance masking strategies based on attention scores, MHIM-MIL employs a momentum teacher to implicitly mine hard instances for training the student model, which can be any attention-based MIL model.This counter-intuitive strategy essentially enables the student to learn a better discriminating boundary.Moreover, the student is used to update the teacher with an exponential moving average (EMA), which in turn identifies new hard instances for subsequent training iterations and stabilizes the optimization.Experimental results on the CAMELYON-16 and TCGA Lung Cancer datasets demonstrate that MHIM-MIL outperforms other latest methods in terms of performance and training cost. The code is available at:https://github.com/DearCaat/MHIM-MIL.

{{</citation>}}


### (17/59) A Solution to Co-occurrence Bias: Attributes Disentanglement via Mutual Information Minimization for Pedestrian Attribute Recognition (Yibo Zhou et al., 2023)

{{<citation>}}

Yibo Zhou, Hai-Miao Hu, Jinzuo Yu, Zhenbo Xu, Weiqing Lu, Yuran Cao. (2023)  
**A Solution to Co-occurrence Bias: Attributes Disentanglement via Mutual Information Minimization for Pedestrian Attribute Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.15252v1)  

---


**ABSTRACT**  
Recent studies on pedestrian attribute recognition progress with either explicit or implicit modeling of the co-occurrence among attributes. Considering that this known a prior is highly variable and unforeseeable regarding the specific scenarios, we show that current methods can actually suffer in generalizing such fitted attributes interdependencies onto scenes or identities off the dataset distribution, resulting in the underlined bias of attributes co-occurrence. To render models robust in realistic scenes, we propose the attributes-disentangled feature learning to ensure the recognition of an attribute not inferring on the existence of others, and which is sequentially formulated as a problem of mutual information minimization. Rooting from it, practical strategies are devised to efficiently decouple attributes, which substantially improve the baseline and establish state-of-the-art performance on realistic datasets like PETAzs and RAPzs. Code is released on https://github.com/SDret/A-Solution-to-Co-occurence-Bias-in-Pedestrian-Attribute-Recognition.

{{</citation>}}


## cs.CL (18)



### (18/59) Uncertainty in Natural Language Generation: From Theory to Applications (Joris Baan et al., 2023)

{{<citation>}}

Joris Baan, Nico Daheim, Evgenia Ilia, Dennis Ulmer, Haau-Sing Li, Raquel Fernández, Barbara Plank, Rico Sennrich, Chrysoula Zerva, Wilker Aziz. (2023)  
**Uncertainty in Natural Language Generation: From Theory to Applications**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Natural Language Generation  
[Paper Link](http://arxiv.org/abs/2307.15703v1)  

---


**ABSTRACT**  
Recent advances of powerful Language Models have allowed Natural Language Generation (NLG) to emerge as an important technology that can not only perform traditional tasks like summarisation or translation, but also serve as a natural language interface to a variety of applications. As such, it is crucial that NLG systems are trustworthy and reliable, for example by indicating when they are likely to be wrong; and supporting multiple views, backgrounds and writing styles -- reflecting diverse human sub-populations. In this paper, we argue that a principled treatment of uncertainty can assist in creating systems and evaluation protocols better aligned with these goals. We first present the fundamental theory, frameworks and vocabulary required to represent uncertainty. We then characterise the main sources of uncertainty in NLG from a linguistic perspective, and propose a two-dimensional taxonomy that is more informative and faithful than the popular aleatoric/epistemic dichotomy. Finally, we move from theory to applications and highlight exciting research directions that exploit uncertainty to power decoding, controllable generation, self-assessment, selective answering, active learning and more.

{{</citation>}}


### (19/59) 'What are you referring to?' Evaluating the Ability of Multi-Modal Dialogue Models to Process Clarificational Exchanges (Javier Chiyah-Garcia et al., 2023)

{{<citation>}}

Javier Chiyah-Garcia, Alessandro Suglia, Arash Eshghi, Helen Hastie. (2023)  
**'What are you referring to?' Evaluating the Ability of Multi-Modal Dialogue Models to Process Clarificational Exchanges**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2307.15554v1)  

---


**ABSTRACT**  
Referential ambiguities arise in dialogue when a referring expression does not uniquely identify the intended referent for the addressee. Addressees usually detect such ambiguities immediately and work with the speaker to repair it using meta-communicative, Clarificational Exchanges (CE): a Clarification Request (CR) and a response. Here, we argue that the ability to generate and respond to CRs imposes specific constraints on the architecture and objective functions of multi-modal, visually grounded dialogue models. We use the SIMMC 2.0 dataset to evaluate the ability of different state-of-the-art model architectures to process CEs, with a metric that probes the contextual updates that arise from them in the model. We find that language-based models are able to encode simple multi-modal semantic information and process some CEs, excelling with those related to the dialogue history, whilst multi-modal models can use additional learning objectives to obtain disentangled object representations, which become crucial to handle complex referential ambiguities across modalities overall.

{{</citation>}}


### (20/59) The Road to Quality is Paved with Good Revisions: A Detailed Evaluation Methodology for Revision Policies in Incremental Sequence Labelling (Brielen Madureira et al., 2023)

{{<citation>}}

Brielen Madureira, Patrick Kahardipraja, David Schlangen. (2023)  
**The Road to Quality is Paved with Good Revisions: A Detailed Evaluation Methodology for Revision Policies in Incremental Sequence Labelling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.15508v1)  

---


**ABSTRACT**  
Incremental dialogue model components produce a sequence of output prefixes based on incoming input. Mistakes can occur due to local ambiguities or to wrong hypotheses, making the ability to revise past outputs a desirable property that can be governed by a policy. In this work, we formalise and characterise edits and revisions in incremental sequence labelling and propose metrics to evaluate revision policies. We then apply our methodology to profile the incremental behaviour of three Transformer-based encoders in various tasks, paving the road for better revision policies.

{{</citation>}}


### (21/59) Exploring Format Consistency for Instruction Tuning (Shihao Liang et al., 2023)

{{<citation>}}

Shihao Liang, Kunlun Zhu, Runchu Tian, Yujia Qin, Huadong Wang, Xin Cong, Zhiyuan Liu, Xiaojiang Liu, Maosong Sun. (2023)  
**Exploring Format Consistency for Instruction Tuning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.15504v1)  

---


**ABSTRACT**  
Instruction tuning has emerged as a promising approach to enhancing large language models in following human instructions. It is shown that increasing the diversity and number of instructions in the training data can consistently enhance generalization performance, which facilitates a recent endeavor to collect various instructions and integrate existing instruction tuning datasets into larger collections. However, different users have their unique ways of expressing instructions, and there often exist variations across different datasets in the instruction styles and formats, i.e., format inconsistency. In this work, we study how format inconsistency may impact the performance of instruction tuning. We propose a framework called "Unified Instruction Tuning" (UIT), which calls OpenAI APIs for automatic format transfer among different instruction tuning datasets. We show that UIT successfully improves the generalization performance on unseen instructions, which highlights the importance of format consistency for instruction tuning. To make the UIT framework more practical, we further propose a novel perplexity-based denoising method to reduce the noise of automatic format transfer. We also train a smaller offline model that achieves comparable format transfer capability than OpenAI APIs to reduce costs in practice.

{{</citation>}}


### (22/59) ETHER: Aligning Emergent Communication for Hindsight Experience Replay (Kevin Denamganaï et al., 2023)

{{<citation>}}

Kevin Denamganaï, Daniel Hernandez, Ozan Vardal, Sondess Missaoui, James Alfred Walker. (2023)  
**ETHER: Aligning Emergent Communication for Hindsight Experience Replay**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-HC, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.15494v1)  

---


**ABSTRACT**  
Natural language instruction following is paramount to enable collaboration between artificial agents and human beings. Natural language-conditioned reinforcement learning (RL) agents have shown how natural languages' properties, such as compositionality, can provide a strong inductive bias to learn complex policies. Previous architectures like HIGhER combine the benefit of language-conditioning with Hindsight Experience Replay (HER) to deal with sparse rewards environments. Yet, like HER, HIGhER relies on an oracle predicate function to provide a feedback signal highlighting which linguistic description is valid for which state. This reliance on an oracle limits its application. Additionally, HIGhER only leverages the linguistic information contained in successful RL trajectories, thus hurting its final performance and data-efficiency. Without early successful trajectories, HIGhER is no better than DQN upon which it is built. In this paper, we propose the Emergent Textual Hindsight Experience Replay (ETHER) agent, which builds on HIGhER and addresses both of its limitations by means of (i) a discriminative visual referential game, commonly studied in the subfield of Emergent Communication (EC), used here as an unsupervised auxiliary task and (ii) a semantic grounding scheme to align the emergent language with the natural language of the instruction-following benchmark. We show that the referential game's agents make an artificial language emerge that is aligned with the natural-like language used to describe goals in the BabyAI benchmark and that it is expressive enough so as to also describe unsuccessful RL trajectories and thus provide feedback to the RL agent to leverage the linguistic, structured information contained in all trajectories. Our work shows that EC is a viable unsupervised auxiliary task for RL and provides missing pieces to make HER more widely applicable.

{{</citation>}}


### (23/59) Trie-NLG: Trie Context Augmentation to Improve Personalized Query Auto-Completion for Short and Unseen Prefixes (Kaushal Kumar Maurya et al., 2023)

{{<citation>}}

Kaushal Kumar Maurya, Maunendra Sankar Desarkar, Manish Gupta, Puneet Agrawal. (2023)  
**Trie-NLG: Trie Context Augmentation to Improve Personalized Query Auto-Completion for Short and Unseen Prefixes**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Natural Language Generation, QA  
[Paper Link](http://arxiv.org/abs/2307.15455v1)  

---


**ABSTRACT**  
Query auto-completion (QAC) aims at suggesting plausible completions for a given query prefix. Traditionally, QAC systems have leveraged tries curated from historical query logs to suggest most popular completions. In this context, there are two specific scenarios that are difficult to handle for any QAC system: short prefixes (which are inherently ambiguous) and unseen prefixes. Recently, personalized Natural Language Generation (NLG) models have been proposed to leverage previous session queries as context for addressing these two challenges. However, such NLG models suffer from two drawbacks: (1) some of the previous session queries could be noisy and irrelevant to the user intent for the current prefix, and (2) NLG models cannot directly incorporate historical query popularity. This motivates us to propose a novel NLG model for QAC, Trie-NLG, which jointly leverages popularity signals from trie and personalization signals from previous session queries. We train the Trie-NLG model by augmenting the prefix with rich context comprising of recent session queries and top trie completions. This simple modeling approach overcomes the limitations of trie-based and NLG-based approaches and leads to state-of-the-art performance. We evaluate the Trie-NLG model using two large QAC datasets. On average, our model achieves huge ~57% and ~14% boost in MRR over the popular trie-based lookup and the strong BART-based baseline methods, respectively. We make our code publicly available.

{{</citation>}}


### (24/59) CFN-ESA: A Cross-Modal Fusion Network with Emotion-Shift Awareness for Dialogue Emotion Recognition (Jiang Li et al., 2023)

{{<citation>}}

Jiang Li, Yingjian Liu, Xiaoping Wang, Zhigang Zeng. (2023)  
**CFN-ESA: A Cross-Modal Fusion Network with Emotion-Shift Awareness for Dialogue Emotion Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2307.15432v1)  

---


**ABSTRACT**  
Multimodal Emotion Recognition in Conversation (ERC) has garnered growing attention from research communities in various fields. In this paper, we propose a cross-modal fusion network with emotion-shift awareness (CFN-ESA) for ERC. Extant approaches employ each modality equally without distinguishing the amount of emotional information, rendering it hard to adequately extract complementary and associative information from multimodal data. To cope with this problem, in CFN-ESA, textual modalities are treated as the primary source of emotional information, while visual and acoustic modalities are taken as the secondary sources. Besides, most multimodal ERC models ignore emotion-shift information and overfocus on contextual information, leading to the failure of emotion recognition under emotion-shift scenario. We elaborate an emotion-shift module to address this challenge. CFN-ESA mainly consists of the unimodal encoder (RUME), cross-modal encoder (ACME), and emotion-shift module (LESM). RUME is applied to extract conversation-level contextual emotional cues while pulling together the data distributions between modalities; ACME is utilized to perform multimodal interaction centered on textual modality; LESM is used to model emotion shift and capture related information, thereby guide the learning of the main task. Experimental results demonstrate that CFN-ESA can effectively promote performance for ERC and remarkably outperform the state-of-the-art models.

{{</citation>}}


### (25/59) A Critical Review of Large Language Models: Sensitivity, Bias, and the Path Toward Specialized AI (Arash Hajikhani et al., 2023)

{{<citation>}}

Arash Hajikhani, Carolyn Cole. (2023)  
**A Critical Review of Large Language Models: Sensitivity, Bias, and the Path Toward Specialized AI**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Bias, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2307.15425v1)  

---


**ABSTRACT**  
This paper examines the comparative effectiveness of a specialized compiled language model and a general-purpose model like OpenAI's GPT-3.5 in detecting SDGs within text data. It presents a critical review of Large Language Models (LLMs), addressing challenges related to bias and sensitivity. The necessity of specialized training for precise, unbiased analysis is underlined. A case study using a company descriptions dataset offers insight into the differences between the GPT-3.5 and the specialized SDG detection model. While GPT-3.5 boasts broader coverage, it may identify SDGs with limited relevance to the companies' activities. In contrast, the specialized model zeroes in on highly pertinent SDGs. The importance of thoughtful model selection is emphasized, taking into account task requirements, cost, complexity, and transparency. Despite the versatility of LLMs, the use of specialized models is suggested for tasks demanding precision and accuracy. The study concludes by encouraging further research to find a balance between the capabilities of LLMs and the need for domain-specific expertise and interpretability.

{{</citation>}}


### (26/59) Towards a Fully Unsupervised Framework for Intent Induction in Customer Support Dialogues (Rita Costa et al., 2023)

{{<citation>}}

Rita Costa, Bruno Martins, Sérgio Viana, Luisa Coheur. (2023)  
**Towards a Fully Unsupervised Framework for Intent Induction in Customer Support Dialogues**  

---
Primary Category: cs.CL  
Categories: I-2; I-7, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2307.15410v1)  

---


**ABSTRACT**  
State of the art models in intent induction require annotated datasets. However, annotating dialogues is time-consuming, laborious and expensive. In this work, we propose a completely unsupervised framework for intent induction within a dialogue. In addition, we show how pre-processing the dialogue corpora can improve results. Finally, we show how to extract the dialogue flows of intentions by investigating the most common sequences. Although we test our work in the MultiWOZ dataset, the fact that this framework requires no prior knowledge make it applicable to any possible use case, making it very relevant to real world customer support applications across industry.

{{</citation>}}


### (27/59) Multilingual Tourist Assistance using ChatGPT: Comparing Capabilities in Hindi, Telugu, and Kannada (Sanjana Kolar et al., 2023)

{{<citation>}}

Sanjana Kolar, Rohit Kumar. (2023)  
**Multilingual Tourist Assistance using ChatGPT: Comparing Capabilities in Hindi, Telugu, and Kannada**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, BLEU, ChatGPT, GPT, Multilingual  
[Paper Link](http://arxiv.org/abs/2307.15376v1)  

---


**ABSTRACT**  
This research investigates the effectiveness of ChatGPT, an AI language model by OpenAI, in translating English into Hindi, Telugu, and Kannada languages, aimed at assisting tourists in India's linguistically diverse environment. To measure the translation quality, a test set of 50 questions from diverse fields such as general knowledge, food, and travel was used. These were assessed by five volunteers for accuracy and fluency, and the scores were subsequently converted into a BLEU score. The BLEU score evaluates the closeness of a machine-generated translation to a human translation, with a higher score indicating better translation quality. The Hindi translations outperformed others, showcasing superior accuracy and fluency, whereas Telugu translations lagged behind. Human evaluators rated both the accuracy and fluency of translations, offering a comprehensive perspective on the language model's performance.

{{</citation>}}


### (28/59) Med-HALT: Medical Domain Hallucination Test for Large Language Models (Logesh Kumar Umapathi et al., 2023)

{{<citation>}}

Logesh Kumar Umapathi, Ankit Pal, Malaikannan Sankarasubbu. (2023)  
**Med-HALT: Medical Domain Hallucination Test for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs-LO, cs.CL, stat-ML  
Keywords: Falcon, GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2307.15343v1)  

---


**ABSTRACT**  
This research paper focuses on the challenges posed by hallucinations in large language models (LLMs), particularly in the context of the medical domain. Hallucination, wherein these models generate plausible yet unverified or incorrect information, can have serious consequences in healthcare applications. We propose a new benchmark and dataset, Med-HALT (Medical Domain Hallucination Test), designed specifically to evaluate and reduce hallucinations. Med-HALT provides a diverse multinational dataset derived from medical examinations across various countries and includes multiple innovative testing modalities. Med-HALT includes two categories of tests reasoning and memory-based hallucination tests, designed to assess LLMs's problem-solving and information retrieval abilities.   Our study evaluated leading LLMs, including Text Davinci, GPT-3.5, LlaMa-2, MPT, and Falcon, revealing significant differences in their performance. The paper provides detailed insights into the dataset, promoting transparency and reproducibility. Through this work, we aim to contribute to the development of safer and more reliable language models in healthcare. Our benchmark can be found at medhalt.github.io

{{</citation>}}


### (29/59) Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding (Xuefei Ning et al., 2023)

{{<citation>}}

Xuefei Ning, Zinan Lin, Zixuan Zhou, Huazhong Yang, Yu Wang. (2023)  
**Skeleton-of-Thought: Large Language Models Can Do Parallel Decoding**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.15337v1)  

---


**ABSTRACT**  
This work aims at decreasing the end-to-end generation latency of large language models (LLMs). One of the major causes of the high generation latency is the sequential decoding approach adopted by almost all state-of-the-art LLMs. In this work, motivated by the thinking and writing process of humans, we propose "Skeleton-of-Thought" (SoT), which guides LLMs to first generate the skeleton of the answer, and then conducts parallel API calls or batched decoding to complete the contents of each skeleton point in parallel. Not only does SoT provide considerable speed-up (up to 2.39x across 11 different LLMs), but it can also potentially improve the answer quality on several question categories in terms of diversity and relevance. SoT is an initial attempt at data-centric optimization for efficiency, and reveal the potential of pushing LLMs to think more like a human for answer quality.

{{</citation>}}


### (30/59) BARTPhoBEiT: Pre-trained Sequence-to-Sequence and Image Transformers Models for Vietnamese Visual Question Answering (Khiem Vinh Tran et al., 2023)

{{<citation>}}

Khiem Vinh Tran, Kiet Van Nguyen, Ngan Luu Thuy Nguyen. (2023)  
**BARTPhoBEiT: Pre-trained Sequence-to-Sequence and Image Transformers Models for Vietnamese Visual Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: NLP, QA, Question Answering, Sequence-to-Sequence, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.15335v1)  

---


**ABSTRACT**  
Visual Question Answering (VQA) is an intricate and demanding task that integrates natural language processing (NLP) and computer vision (CV), capturing the interest of researchers. The English language, renowned for its wealth of resources, has witnessed notable advancements in both datasets and models designed for VQA. However, there is a lack of models that target specific countries such as Vietnam. To address this limitation, we introduce a transformer-based Vietnamese model named BARTPhoBEiT. This model includes pre-trained Sequence-to-Sequence and bidirectional encoder representation from Image Transformers in Vietnamese and evaluates Vietnamese VQA datasets. Experimental results demonstrate that our proposed model outperforms the strong baseline and improves the state-of-the-art in six metrics: Accuracy, Precision, Recall, F1-score, WUPS 0.0, and WUPS 0.9.

{{</citation>}}


### (31/59) Tutorials on Stance Detection using Pre-trained Language Models: Fine-tuning BERT and Prompting Large Language Models (Yun-Shiuan Chuang, 2023)

{{<citation>}}

Yun-Shiuan Chuang. (2023)  
**Tutorials on Stance Detection using Pre-trained Language Models: Fine-tuning BERT and Prompting Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, ChatGPT, GPT, Language Model, Stance Detection, T5, Twitter  
[Paper Link](http://arxiv.org/abs/2307.15331v1)  

---


**ABSTRACT**  
This paper presents two self-contained tutorials on stance detection in Twitter data using BERT fine-tuning and prompting large language models (LLMs). The first tutorial explains BERT architecture and tokenization, guiding users through training, tuning, and evaluating standard and domain-specific BERT models with HuggingFace transformers. The second focuses on constructing prompts and few-shot examples to elicit stances from ChatGPT and open-source FLAN-T5 without fine-tuning. Various prompting strategies are implemented and evaluated using confusion matrices and macro F1 scores. The tutorials provide code, visualizations, and insights revealing the strengths of few-shot ChatGPT and FLAN-T5 which outperform fine-tuned BERTs. By covering both model fine-tuning and prompting-based techniques in an accessible, hands-on manner, these tutorials enable learners to gain applied experience with cutting-edge methods for stance detection.

{{</citation>}}


### (32/59) TrafficSafetyGPT: Tuning a Pre-trained Large Language Model to a Domain-Specific Expert in Transportation Safety (Ou Zheng et al., 2023)

{{<citation>}}

Ou Zheng, Mohamed Abdel-Aty, Dongdong Wang, Chenzhu Wang, Shengxuan Ding. (2023)  
**TrafficSafetyGPT: Tuning a Pre-trained Large Language Model to a Domain-Specific Expert in Transportation Safety**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2307.15311v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown remarkable effectiveness in various general-domain natural language processing (NLP) tasks. However, their performance in transportation safety domain tasks has been suboptimal, primarily attributed to the requirement for specialized transportation safety expertise in generating accurate responses [1]. To address this challenge, we introduce TrafficSafetyGPT, a novel LLAMA-based model, which has undergone supervised fine-tuning using TrafficSafety-2K dataset which has human labels from government produced guiding books and ChatGPT-generated instruction-output pairs. Our proposed TrafficSafetyGPT model and TrafficSafety-2K train dataset are accessible at https://github.com/ozheng1993/TrafficSafetyGPT.

{{</citation>}}


### (33/59) WC-SBERT: Zero-Shot Text Classification via SBERT with Self-Training for Wikipedia Categories (Te-Yu Chi et al., 2023)

{{<citation>}}

Te-Yu Chi, Yu-Meng Tang, Chia-Wen Lu, Qiu-Xia Zhang, Jyh-Shing Roger Jang. (2023)  
**WC-SBERT: Zero-Shot Text Classification via SBERT with Self-Training for Wikipedia Categories**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, NLP, Text Classification, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.15293v1)  

---


**ABSTRACT**  
Our research focuses on solving the zero-shot text classification problem in NLP, with a particular emphasis on innovative self-training strategies. To achieve this objective, we propose a novel self-training strategy that uses labels rather than text for training, significantly reducing the model's training time. Specifically, we use categories from Wikipedia as our training set and leverage the SBERT pre-trained model to establish positive correlations between pairs of categories within the same text, facilitating associative training. For new test datasets, we have improved the original self-training approach, eliminating the need for prior training and testing data from each target dataset. Instead, we adopt Wikipedia as a unified training dataset to better approximate the zero-shot scenario. This modification allows for rapid fine-tuning and inference across different datasets, greatly reducing the time required for self-training. Our experimental results demonstrate that this method can adapt the model to the target dataset within minutes. Compared to other BERT-based transformer models, our approach significantly reduces the amount of training data by training only on labels, not the actual text, and greatly improves training efficiency by utilizing a unified training set. Additionally, our method achieves state-of-the-art results on both the Yahoo Topic and AG News datasets.

{{</citation>}}


### (34/59) ChatHome: Development and Evaluation of a Domain-Specific Language Model for Home Renovation (Cheng Wen et al., 2023)

{{<citation>}}

Cheng Wen, Xianghui Sun, Shuaijiang Zhao, Xiaoquan Fang, Liangyu Chen, Wei Zou. (2023)  
**ChatHome: Development and Evaluation of a Domain-Specific Language Model for Home Renovation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.15290v1)  

---


**ABSTRACT**  
This paper presents the development and evaluation of ChatHome, a domain-specific language model (DSLM) designed for the intricate field of home renovation. Considering the proven competencies of large language models (LLMs) like GPT-4 and the escalating fascination with home renovation, this study endeavors to reconcile these aspects by generating a dedicated model that can yield high-fidelity, precise outputs relevant to the home renovation arena. ChatHome's novelty rests on its methodology, fusing domain-adaptive pretraining and instruction-tuning over an extensive dataset. This dataset includes professional articles, standard documents, and web content pertinent to home renovation. This dual-pronged strategy is designed to ensure that our model can assimilate comprehensive domain knowledge and effectively address user inquiries. Via thorough experimentation on diverse datasets, both universal and domain-specific, including the freshly introduced "EvalHome" domain dataset, we substantiate that ChatHome not only amplifies domain-specific functionalities but also preserves its versatility.

{{</citation>}}


### (35/59) Multilingual Lexical Simplification via Paraphrase Generation (Kang Liu et al., 2023)

{{<citation>}}

Kang Liu, Jipeng Qiang, Yun Li, Yunhao Yuan, Yi Zhu, Kaixun Hua. (2023)  
**Multilingual Lexical Simplification via Paraphrase Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, GPT, Multilingual  
[Paper Link](http://arxiv.org/abs/2307.15286v1)  

---


**ABSTRACT**  
Lexical simplification (LS) methods based on pretrained language models have made remarkable progress, generating potential substitutes for a complex word through analysis of its contextual surroundings. However, these methods require separate pretrained models for different languages and disregard the preservation of sentence meaning. In this paper, we propose a novel multilingual LS method via paraphrase generation, as paraphrases provide diversity in word selection while preserving the sentence's meaning. We regard paraphrasing as a zero-shot translation task within multilingual neural machine translation that supports hundreds of languages. After feeding the input sentence into the encoder of paraphrase modeling, we generate the substitutes based on a novel decoding strategy that concentrates solely on the lexical variations of the complex word. Experimental results demonstrate that our approach surpasses BERT-based methods and zero-shot GPT3-based method significantly on English, Spanish, and Portuguese.

{{</citation>}}


## cs.LG (6)



### (36/59) Benchmarking Offline Reinforcement Learning on Real-Robot Hardware (Nico Gürtler et al., 2023)

{{<citation>}}

Nico Gürtler, Sebastian Blaes, Pavel Kolev, Felix Widmaier, Manuel Wüthrich, Stefan Bauer, Bernhard Schölkopf, Georg Martius. (2023)  
**Benchmarking Offline Reinforcement Learning on Real-Robot Hardware**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.15690v1)  

---


**ABSTRACT**  
Learning policies from previously recorded data is a promising direction for real-world robotics tasks, as online learning is often infeasible. Dexterous manipulation in particular remains an open problem in its general form. The combination of offline reinforcement learning with large diverse datasets, however, has the potential to lead to a breakthrough in this challenging domain analogously to the rapid progress made in supervised learning in recent years. To coordinate the efforts of the research community toward tackling this problem, we propose a benchmark including: i) a large collection of data for offline learning from a dexterous manipulation platform on two tasks, obtained with capable RL agents trained in simulation; ii) the option to execute learned policies on a real-world robotic system and a simulation for efficient debugging. We evaluate prominent open-sourced offline reinforcement learning algorithms on the datasets and provide a reproducible experimental setup for offline reinforcement learning on real systems.

{{</citation>}}


### (37/59) Dynamic Analysis and an Eigen Initializer for Recurrent Neural Networks (Ran Dou et al., 2023)

{{<citation>}}

Ran Dou, Jose Principe. (2023)  
**Dynamic Analysis and an Eigen Initializer for Recurrent Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.15679v1)  

---


**ABSTRACT**  
In recurrent neural networks, learning long-term dependency is the main difficulty due to the vanishing and exploding gradient problem. Many researchers are dedicated to solving this issue and they proposed many algorithms. Although these algorithms have achieved great success, understanding how the information decays remains an open problem. In this paper, we study the dynamics of the hidden state in recurrent neural networks. We propose a new perspective to analyze the hidden state space based on an eigen decomposition of the weight matrix. We start the analysis by linear state space model and explain the function of preserving information in activation functions. We provide an explanation for long-term dependency based on the eigen analysis. We also point out the different behavior of eigenvalues for regression tasks and classification tasks. From the observations on well-trained recurrent neural networks, we proposed a new initialization method for recurrent neural networks, which improves consistently performance. It can be applied to vanilla-RNN, LSTM, and GRU. We test on many datasets, such as Tomita Grammars, pixel-by-pixel MNIST datasets, and machine translation datasets (Multi30k). It outperforms the Xavier initializer and kaiming initializer as well as other RNN-only initializers like IRNN and sp-RNN in several tasks.

{{</citation>}}


### (38/59) Case Studies of Causal Discovery from IT Monitoring Time Series (Ali Aït-Bachir et al., 2023)

{{<citation>}}

Ali Aït-Bachir, Charles K. Assaad, Christophe de Bignicourt, Emilie Devijver, Simon Ferreira, Eric Gaussier, Hosein Mohanna, Lei Zan. (2023)  
**Case Studies of Causal Discovery from IT Monitoring Time Series**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-AP, stat-ME  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.15678v1)  

---


**ABSTRACT**  
Information technology (IT) systems are vital for modern businesses, handling data storage, communication, and process automation. Monitoring these systems is crucial for their proper functioning and efficiency, as it allows collecting extensive observational time series data for analysis. The interest in causal discovery is growing in IT monitoring systems as knowing causal relations between different components of the IT system helps in reducing downtime, enhancing system performance and identifying root causes of anomalies and incidents. It also allows proactive prediction of future issues through historical data analysis. Despite its potential benefits, applying causal discovery algorithms on IT monitoring data poses challenges, due to the complexity of the data. For instance, IT monitoring data often contains misaligned time series, sleeping time series, timestamp errors and missing values. This paper presents case studies on applying causal discovery algorithms to different IT monitoring datasets, highlighting benefits and ongoing challenges.

{{</citation>}}


### (39/59) Robust Distortion-free Watermarks for Language Models (Rohith Kuditipudi et al., 2023)

{{<citation>}}

Rohith Kuditipudi, John Thickstun, Tatsunori Hashimoto, Percy Liang. (2023)  
**Robust Distortion-free Watermarks for Language Models**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2307.15593v1)  

---


**ABSTRACT**  
We propose a methodology for planting watermarks in text from an autoregressive language model that are robust to perturbations without changing the distribution over text up to a certain maximum generation budget. We generate watermarked text by mapping a sequence of random numbers -- which we compute using a randomized watermark key -- to a sample from the language model. To detect watermarked text, any party who knows the key can align the text to the random number sequence. We instantiate our watermark methodology with two sampling schemes: inverse transform sampling and exponential minimum sampling. We apply these watermarks to three language models -- OPT-1.3B, LLaMA-7B and Alpaca-7B -- to experimentally validate their statistical power and robustness to various paraphrasing attacks. Notably, for both the OPT-1.3B and LLaMA-7B models, we find we can reliably detect watermarked text ($p \leq 0.01$) from $35$ tokens even after corrupting between $40$-$50$\% of the tokens via random edits (i.e., substitutions, insertions or deletions). For the Alpaca-7B model, we conduct a case study on the feasibility of watermarking responses to typical user instructions. Due to the lower entropy of the responses, detection is more difficult: around $25\%$ of the responses -- whose median length is around $100$ tokens -- are detectable with $p \leq 0.01$, and the watermark is also less robust to certain automated paraphrasing attacks we implement.

{{</citation>}}


### (40/59) Co-attention Graph Pooling for Efficient Pairwise Graph Interaction Learning (Junhyun Lee et al., 2023)

{{<citation>}}

Junhyun Lee, Bumsoo Kim, Minji Jeon, Jaewoo Kang. (2023)  
**Co-attention Graph Pooling for Efficient Pairwise Graph Interaction Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.15377v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have proven to be effective in processing and learning from graph-structured data. However, previous works mainly focused on understanding single graph inputs while many real-world applications require pair-wise analysis for graph-structured data (e.g., scene graph matching, code searching, and drug-drug interaction prediction). To this end, recent works have shifted their focus to learning the interaction between pairs of graphs. Despite their improved performance, these works were still limited in that the interactions were considered at the node-level, resulting in high computational costs and suboptimal performance. To address this issue, we propose a novel and efficient graph-level approach for extracting interaction representations using co-attention in graph pooling. Our method, Co-Attention Graph Pooling (CAGPool), exhibits competitive performance relative to existing methods in both classification and regression tasks using real-world datasets, while maintaining lower computational complexity.

{{</citation>}}


### (41/59) Toward Transparent Sequence Models with Model-Based Tree Markov Model (Chan Hsu et al., 2023)

{{<citation>}}

Chan Hsu, Wei-Chun Huang, Jun-Ting Wu, Chih-Yuan Li, Yihuang Kang. (2023)  
**Toward Transparent Sequence Models with Model-Based Tree Markov Model**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2307.15367v1)  

---


**ABSTRACT**  
In this study, we address the interpretability issue in complex, black-box Machine Learning models applied to sequence data. We introduce the Model-Based tree Hidden Semi-Markov Model (MOB-HSMM), an inherently interpretable model aimed at detecting high mortality risk events and discovering hidden patterns associated with the mortality risk in Intensive Care Units (ICU). This model leverages knowledge distilled from Deep Neural Networks (DNN) to enhance predictive performance while offering clear explanations. Our experimental results indicate the improved performance of Model-Based trees (MOB trees) via employing LSTM for learning sequential patterns, which are then transferred to MOB trees. Integrating MOB trees with the Hidden Semi-Markov Model (HSMM) in the MOB-HSMM enables uncovering potential and explainable sequences using available information.

{{</citation>}}


## eess.IV (2)



### (42/59) Scale-aware Test-time Click Adaptation for Pulmonary Nodule and Mass Segmentation (Zhihao Li et al., 2023)

{{<citation>}}

Zhihao Li, Jiancheng Yang, Yongchao Xu, Li Zhang, Wenhui Dong, Bo Du. (2023)  
**Scale-aware Test-time Click Adaptation for Pulmonary Nodule and Mass Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.15645v1)  

---


**ABSTRACT**  
Pulmonary nodules and masses are crucial imaging features in lung cancer screening that require careful management in clinical diagnosis. Despite the success of deep learning-based medical image segmentation, the robust performance on various sizes of lesions of nodule and mass is still challenging. In this paper, we propose a multi-scale neural network with scale-aware test-time adaptation to address this challenge. Specifically, we introduce an adaptive Scale-aware Test-time Click Adaptation method based on effortlessly obtainable lesion clicks as test-time cues to enhance segmentation performance, particularly for large lesions. The proposed method can be seamlessly integrated into existing networks. Extensive experiments on both open-source and in-house datasets consistently demonstrate the effectiveness of the proposed method over some CNN and Transformer-based segmentation methods. Our code is available at https://github.com/SplinterLi/SaTTCA

{{</citation>}}


### (43/59) ERCPMP: An Endoscopic Image and Video Dataset for Colorectal Polyps Morphology and Pathology (Mojgan Forootan et al., 2023)

{{<citation>}}

Mojgan Forootan, Mohsen Rajabnia, Ahmad R Mafi, Hamed Azhdari Tehrani, Erfan Ghadirzadeh, Mahziar Setayeshfar, Zahra Ghaffari, Mohammad Tashakoripour, Mohammad Reza Zali, Hamidreza Bolhasani. (2023)  
**ERCPMP: An Endoscopic Image and Video Dataset for Colorectal Polyps Morphology and Pathology**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.15444v1)  

---


**ABSTRACT**  
In the recent years, artificial intelligence (AI) and its leading subtypes, machine learning (ML) and deep learning (DL) and their applications are spreading very fast in various aspects such as medicine. Today the most important challenge of developing accurate algorithms for medical prediction, detection, diagnosis, treatment and prognosis is data. ERCPMP is an Endoscopic Image and Video Dataset for Recognition of Colorectal Polyps Morphology and Pathology. This dataset contains demographic, morphological and pathological data, endoscopic images and videos of 191 patients with colorectal polyps. Morphological data is included based on the latest international gastroenterology classification references such as Paris, Pit and JNET classification. Pathological data includes the diagnosis of the polyps including Tubular, Villous, Tubulovillous, Hyperplastic, Serrated, Inflammatory and Adenocarcinoma with Dysplasia Grade & Differentiation. The current version of this dataset is published and available on Elsevier Mendeley Dataverse and since it is under development, the latest version is accessible via: https://databiox.com.

{{</citation>}}


## cs.SI (3)



### (44/59) Trends and Topics: Characterizing Echo Chambers' Topological Stability and In-group Attitudes (Erica Cau et al., 2023)

{{<citation>}}

Erica Cau, Virginia Morini, Giulio Rossetti. (2023)  
**Trends and Topics: Characterizing Echo Chambers' Topological Stability and In-group Attitudes**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2307.15610v1)  

---


**ABSTRACT**  
Social Network sites are fertile ground for several polluting phenomena affecting online and offline spaces. Among these phenomena are included echo chambers, closed systems in which the opinions expressed by the people inside are exacerbated for the effect of the repetition, while opposite views are actively excluded. This paper offers a framework to explore, in a platform-independent manner, the topological changes through time of echo chambers, while considering the content posted by users and the attitude conveyed in discussing specific controversial issues.   The proposed framework consists of four steps: (i) data collection and annotation of users' ideology regarding a controversial topic, (ii) construction of a dynamic network of interactions, (iii) ECs extraction and analysis of their dynamics, and (iv) topic extraction and valence analysis. The paper then enhances the formalization of the framework by conducting a case study on Reddit threads about sociopolitical issues (gun control, American politics, and minorities discrimination) during the first two years and a half of Donald Trump's presidency.   The results unveil that users often stay inside echo chambers over time. Furthermore, in the analyzed discussions, the focus is on controversies related to right-wing parties and specific events in American and Canadian politics. The analysis of the attitude conveyed in the discussions shows a slight inclination toward a more negative or neutral attitude when discussing particularly sensitive issues, such as fascism, school shootings, or police violence.

{{</citation>}}


### (45/59) The Role of the IRA in Twitter during the 2016 US Presidential Election: Unveiling Amplification and Influence of Suspended Accounts (Matteo Serafino et al., 2023)

{{<citation>}}

Matteo Serafino, Zhenkun Zhou, Jose S. Andrade, Jr., Alexandre Bovet, Hernan A. Makse. (2023)  
**The Role of the IRA in Twitter during the 2016 US Presidential Election: Unveiling Amplification and Influence of Suspended Accounts**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-data-an, physics-soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.15365v1)  

---


**ABSTRACT**  
The impact of the social media campaign conducted by the Internet Research Agency (IRA) during the 2016 U.S. presidential election continues to be a topic of ongoing debate. While it is widely acknowledged that the objective of this campaign was to support Donald Trump, the true extent of its influence on Twitter users remains uncertain. Previous research has primarily focused on analyzing the interactions between IRA users and the broader Twitter community to assess the campaign's impact. In this study, we propose an alternative perspective that suggests the existing approach may underestimate the true extent of the IRA campaign. Our analysis uncovers the presence of a notable group of suspended Twitter users, whose size surpasses the IRA user group size by a factor of 60. These suspended users exhibit close interactions with IRA accounts, suggesting potential collaboration or coordination. Notably, our findings reveal the significant role played by these previously unnoticed accounts in amplifying the impact of the IRA campaign, surpassing even the reach of the IRA accounts themselves by a factor of 10. In contrast to previous findings, our study reveals that the combined efforts of the Internet Research Agency (IRA) and the identified group of suspended Twitter accounts had a significant influence on individuals categorized as undecided or weak supporters, probably with the intention of swaying their opinions.

{{</citation>}}


### (46/59) BOURNE: Bootstrapped Self-supervised Learning Framework for Unified Graph Anomaly Detection (Jie Liu et al., 2023)

{{<citation>}}

Jie Liu, Mengting He, Xuequn Shang, Jieming Shi, Bin Cui, Hongzhi Yin. (2023)  
**BOURNE: Bootstrapped Self-supervised Learning Framework for Unified Graph Anomaly Detection**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.15244v1)  

---


**ABSTRACT**  
Graph anomaly detection (GAD) has gained increasing attention in recent years due to its critical application in a wide range of domains, such as social networks, financial risk management, and traffic analysis. Existing GAD methods can be categorized into node and edge anomaly detection models based on the type of graph objects being detected. However, these methods typically treat node and edge anomalies as separate tasks, overlooking their associations and frequent co-occurrences in real-world graphs. As a result, they fail to leverage the complementary information provided by node and edge anomalies for mutual detection. Additionally, state-of-the-art GAD methods, such as CoLA and SL-GAD, heavily rely on negative pair sampling in contrastive learning, which incurs high computational costs, hindering their scalability to large graphs. To address these limitations, we propose a novel unified graph anomaly detection framework based on bootstrapped self-supervised learning (named BOURNE). We extract a subgraph (graph view) centered on each target node as node context and transform it into a dual hypergraph (hypergraph view) as edge context. These views are encoded using graph and hypergraph neural networks to capture the representations of nodes, edges, and their associated contexts. By swapping the context embeddings between nodes and edges and measuring the agreement in the embedding space, we enable the mutual detection of node and edge anomalies. Furthermore, we adopt a bootstrapped training strategy that eliminates the need for negative sampling, enabling BOURNE to handle large graphs efficiently. Extensive experiments conducted on six benchmark datasets demonstrate the superior effectiveness and efficiency of BOURNE in detecting both node and edge anomalies.

{{</citation>}}


## cs.RO (2)



### (47/59) Learning to Open Doors with an Aerial Manipulator (Eugenio Cuniato et al., 2023)

{{<citation>}}

Eugenio Cuniato, Ismail Geles, Weixuan Zhang, Olov Andersson, Marco Tognon, Roland Siegwart. (2023)  
**Learning to Open Doors with an Aerial Manipulator**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.15581v1)  

---


**ABSTRACT**  
The field of aerial manipulation has seen rapid advances, transitioning from push-and-slide tasks to interaction with articulated objects. So far, when more complex actions are performed, the motion trajectory is usually handcrafted or a result of online optimization methods like Model Predictive Control (MPC) or Model Predictive Path Integral (MPPI) control. However, these methods rely on heuristics or model simplifications to efficiently run on onboard hardware, producing results in acceptable amounts of time. Moreover, they can be sensitive to disturbances and differences between the real environment and its simulated counterpart. In this work, we propose a Reinforcement Learning (RL) approach to learn motion behaviors for a manipulation task while producing policies that are robust to disturbances and modeling errors. Specifically, we train a policy to perform a door-opening task with an Omnidirectional Micro Aerial Vehicle (OMAV). The policy is trained in a physics simulator and experiments are presented both in simulation and running onboard the real platform, investigating the simulation to real world transfer. We compare our method against a state-of-the-art MPPI solution, showing a considerable increase in robustness and speed.

{{</citation>}}


### (48/59) Does Unpredictability Influence Driving Behavior? (Sepehr Samavi et al., 2023)

{{<citation>}}

Sepehr Samavi, Florian Shkurti, Angela Schoellig. (2023)  
**Does Unpredictability Influence Driving Behavior?**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.15287v1)  

---


**ABSTRACT**  
In this paper we investigate the effect of the unpredictability of surrounding cars on an ego-car performing a driving maneuver. We use Maximum Entropy Inverse Reinforcement Learning to model reward functions for an ego-car conducting a lane change in a highway setting. We define a new feature based on the unpredictability of surrounding cars and use it in the reward function. We learn two reward functions from human data: a baseline and one that incorporates our defined unpredictability feature, then compare their performance with a quantitative and qualitative evaluation. Our evaluation demonstrates that incorporating the unpredictability feature leads to a better fit of human-generated test data. These results encourage further investigation of the effect of unpredictability on driving behavior.

{{</citation>}}


## cs.MA (1)



### (49/59) Learning to Collaborate by Grouping: a Consensus-oriented Strategy for Multi-agent Reinforcement Learning (Jingqing Ruan et al., 2023)

{{<citation>}}

Jingqing Ruan, Xiaotian Hao, Dong Li, Hangyu Mao. (2023)  
**Learning to Collaborate by Grouping: a Consensus-oriented Strategy for Multi-agent Reinforcement Learning**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: Google, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.15530v1)  

---


**ABSTRACT**  
Multi-agent systems require effective coordination between groups and individuals to achieve common goals. However, current multi-agent reinforcement learning (MARL) methods primarily focus on improving individual policies and do not adequately address group-level policies, which leads to weak cooperation. To address this issue, we propose a novel Consensus-oriented Strategy (CoS) that emphasizes group and individual policies simultaneously. Specifically, CoS comprises two main components: (a) the vector quantized group consensus module, which extracts discrete latent embeddings that represent the stable and discriminative group consensus, and (b) the group consensus-oriented strategy, which integrates the group policy using a hypernet and the individual policies using the group consensus, thereby promoting coordination at both the group and individual levels. Through empirical experiments on cooperative navigation tasks with both discrete and continuous spaces, as well as Google research football, we demonstrate that CoS outperforms state-of-the-art MARL algorithms and achieves better collaboration, thus providing a promising solution for achieving effective coordination in multi-agent systems.

{{</citation>}}


## cs.AR (1)



### (50/59) Fast Prototyping Next-Generation Accelerators for New ML Models using MASE: ML Accelerator System Exploration (Jianyi Cheng et al., 2023)

{{<citation>}}

Jianyi Cheng, Cheng Zhang, Zhewen Yu, Alex Montgomerie-Corcoran, Can Xiao, Christos-Savvas Bouganis, Yiren Zhao. (2023)  
**Fast Prototyping Next-Generation Accelerators for New ML Models using MASE: ML Accelerator System Exploration**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs.AR  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2307.15517v1)  

---


**ABSTRACT**  
Machine learning (ML) accelerators have been studied and used extensively to compute ML models with high performance and low power. However, designing such accelerators normally takes a long time and requires significant effort. Unfortunately, the pace of development of ML software models is much faster than the accelerator design cycle, leading to frequent and drastic modifications in the model architecture, thus rendering many accelerators obsolete. Existing design tools and frameworks can provide quick accelerator prototyping, but only for a limited range of models that can fit into a single hardware device, such as an FPGA. Furthermore, with the emergence of large language models, such as GPT-3, there is an increased need for hardware prototyping of these large models within a many-accelerator system to ensure the hardware can scale with the ever-growing model sizes. In this paper, we propose an efficient and scalable approach for exploring accelerator systems to compute large ML models. We developed a tool named MASE that can directly map large ML models onto an efficient streaming accelerator system. Over a set of ML models, we show that MASE can achieve better energy efficiency to GPUs when computing inference for recent transformer models. Our tool will open-sourced upon publication.

{{</citation>}}


## cs.SD (1)



### (51/59) Minimally-Supervised Speech Synthesis with Conditional Diffusion Model and Language Model: A Comparative Study of Semantic Coding (Chunyu Qiang et al., 2023)

{{<citation>}}

Chunyu Qiang, Hao Li, Hao Ni, He Qu, Ruibo Fu, Tao Wang, Longbiao Wang, Jianwu Dang. (2023)  
**Minimally-Supervised Speech Synthesis with Conditional Diffusion Model and Language Model: A Comparative Study of Semantic Coding**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.15484v1)  

---


**ABSTRACT**  
Recently, there has been a growing interest in text-to-speech (TTS) methods that can be trained with minimal supervision by combining two types of discrete speech representations and using two sequence-to-sequence tasks to decouple TTS. To address the challenges associated with high dimensionality and waveform distortion in discrete representations, we propose Diff-LM-Speech, which models semantic embeddings into mel-spectrogram based on diffusion models and introduces a prompt encoder structure based on variational autoencoders and prosody bottlenecks to improve prompt representation capabilities. Autoregressive language models often suffer from missing and repeated words, while non-autoregressive frameworks face expression averaging problems due to duration prediction models. To address these issues, we propose Tetra-Diff-Speech, which designs a duration diffusion model to achieve diverse prosodic expressions. While we expect the information content of semantic coding to be between that of text and acoustic coding, existing models extract semantic coding with a lot of redundant information and dimensionality explosion. To verify that semantic coding is not necessary, we propose Tri-Diff-Speech. Experimental results show that our proposed methods outperform baseline methods. We provide a website with audio samples.

{{</citation>}}


## cs.HC (1)



### (52/59) From OECD to India: Exploring cross-cultural differences in perceived trust, responsibility and reliance of AI and human experts (Vishakha Agrawal et al., 2023)

{{<citation>}}

Vishakha Agrawal, Serhiy Kandul, Markus Kneer, Markus Christen. (2023)  
**From OECD to India: Exploring cross-cultural differences in perceived trust, responsibility and reliance of AI and human experts**  

---
Primary Category: cs.HC  
Categories: cs-CY, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.15452v1)  

---


**ABSTRACT**  
AI is getting more involved in tasks formerly exclusively assigned to humans. Most of research on perceptions and social acceptability of AI in these areas is mainly restricted to the Western world. In this study, we compare trust, perceived responsibility, and reliance of AI and human experts across OECD and Indian sample. We find that OECD participants consider humans to be less capable but more morally trustworthy and more responsible than AI. In contrast, Indian participants trust humans more than AI but assign equal responsibility for both types of experts. We discuss implications of the observed differences for algorithmic ethics and human-computer interaction.

{{</citation>}}


## cs.MM (1)



### (53/59) Improving Social Media Popularity Prediction with Multiple Post Dependencies (Zhizhen Zhang et al., 2023)

{{<citation>}}

Zhizhen Zhang, Xiaohui Xie, Mengyu Yang, Ye Tian, Yong Jiang, Yong Cui. (2023)  
**Improving Social Media Popularity Prediction with Multiple Post Dependencies**  

---
Primary Category: cs.MM  
Categories: cs-AI, cs-CL, cs-MM, cs.MM  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2307.15413v1)  

---


**ABSTRACT**  
Social Media Popularity Prediction has drawn a lot of attention because of its profound impact on many different applications, such as recommendation systems and multimedia advertising. Despite recent efforts to leverage the content of social media posts to improve prediction accuracy, many existing models fail to fully exploit the multiple dependencies between posts, which are important to comprehensively extract content information from posts. To tackle this problem, we propose a novel prediction framework named Dependency-aware Sequence Network (DSN) that exploits both intra- and inter-post dependencies. For intra-post dependency, DSN adopts a multimodal feature extractor with an efficient fine-tuning strategy to obtain task-specific representations from images and textual information of posts. For inter-post dependency, DSN uses a hierarchical information propagation method to learn category representations that could better describe the difference between posts. DSN also exploits recurrent networks with a series of gating layers for more flexible local temporal processing abilities and multi-head attention for long-term dependencies. The experimental results on the Social Media Popularity Dataset demonstrate the superiority of our method compared to existing state-of-the-art models.

{{</citation>}}


## cs.IT (2)



### (54/59) Deep Reinforcement Learning Based Intelligent Reflecting Surface Optimization for TDD MultiUser MIMO Systems (Fengyu Zhao et al., 2023)

{{<citation>}}

Fengyu Zhao, Wen Chen, Ziwei Liu, Jun Li, Qingqing Wu. (2023)  
**Deep Reinforcement Learning Based Intelligent Reflecting Surface Optimization for TDD MultiUser MIMO Systems**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.15393v1)  

---


**ABSTRACT**  
In this letter, we investigate the discrete phase shift design of the intelligent reflecting surface (IRS) in a time division duplexing (TDD) multi-user multiple input multiple output (MIMO) system.We modify the design of deep reinforcement learning (DRL) scheme so that we can maximizing the average downlink data transmission rate free from the sub-channel channel state information (CSI). Based on the characteristics of the model, we modify the proximal policy optimization (PPO) algorithm and integrate gated recurrent unit (GRU) to tackle the non-convex optimization problem. Simulation results show that the performance of the proposed PPO-GRU surpasses the benchmarks in terms of performance, convergence speed, and training stability.

{{</citation>}}


### (55/59) Efficient Multiuser AI Downloading via Reusable Knowledge Broadcasting (Hai Wu et al., 2023)

{{<citation>}}

Hai Wu, Qunsong Zeng, Kaibin Huang. (2023)  
**Efficient Multiuser AI Downloading via Reusable Knowledge Broadcasting**  

---
Primary Category: cs.IT  
Categories: cs-AI, cs-IT, cs.IT, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.15316v1)  

---


**ABSTRACT**  
For the 6G mobile networks, in-situ model downloading has emerged as an important use case to enable real-time adaptive artificial intelligence on edge devices. However, the simultaneous downloading of diverse and high-dimensional models to multiple devices over wireless links presents a significant communication bottleneck. To overcome the bottleneck, we propose the framework of model broadcasting and assembling (MBA), which represents the first attempt on leveraging reusable knowledge, referring to shared parameters among tasks, to enable parameter broadcasting to reduce communication overhead. The MBA framework comprises two key components. The first, the MBA protocol, defines the system operations including parameter selection from a model library, power control for broadcasting, and model assembling at devices. The second component is the joint design of parameter-selection-and-power-control (PS-PC), which provides guarantees on devices' model performance and minimizes the downloading latency. The corresponding optimization problem is simplified by decomposition into the sequential PS and PC sub-problems without compromising its optimality. The PS sub-problem is solved efficiently by designing two efficient algorithms. On one hand, the low-complexity algorithm of greedy parameter selection features the construction of candidate model sets and a selection metric, both of which are designed under the criterion of maximum reusable knowledge among tasks. On the other hand, the optimal tree-search algorithm gains its efficiency via the proposed construction of a compact binary tree pruned using model architecture constraints and an intelligent branch-and-bound search. Given optimal PS, the optimal PC policy is derived in closed form. Extensive experiments demonstrate the substantial reduction in downloading latency achieved by the proposed MBA compared to traditional model downloading.

{{</citation>}}


## eess.SY (1)



### (56/59) Leveraging Optical Communication Fiber and AI for Distributed Water Pipe Leak Detection (Huan Wu et al., 2023)

{{<citation>}}

Huan Wu, Huan-Feng Duan, Wallace W. L. Lai, Kun Zhu, Xin Cheng, Hao Yin, Bin Zhou, Chun-Cheung Lai, Chao Lu, Xiaoli Ding. (2023)  
**Leveraging Optical Communication Fiber and AI for Distributed Water Pipe Leak Detection**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.15374v1)  

---


**ABSTRACT**  
Detecting leaks in water networks is a costly challenge. This article introduces a practical solution: the integration of optical network with water networks for efficient leak detection. Our approach uses a fiber-optic cable to measure vibrations, enabling accurate leak identification and localization by an intelligent algorithm. We also propose a method to access leak severity for prioritized repairs. Our solution detects even small leaks with flow rates as low as 0.027 L/s. It offers a cost-effective way to improve leak detection, enhance water management, and increase operational efficiency.

{{</citation>}}


## cs.SE (1)



### (57/59) Private-Library-Oriented Code Generation with Large Language Models (Daoguang Zan et al., 2023)

{{<citation>}}

Daoguang Zan, Bei Chen, Yongshun Gong, Junzhi Cao, Fengji Zhang, Bingchao Wu, Bei Guan, Yilong Yin, Yongji Wang. (2023)  
**Private-Library-Oriented Code Generation with Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.15370v1)  

---


**ABSTRACT**  
Large language models (LLMs), such as Codex and GPT-4, have recently showcased their remarkable code generation abilities, facilitating a significant boost in coding efficiency. This paper will delve into utilizing LLMs for code generation in private libraries, as they are widely employed in everyday programming. Despite their remarkable capabilities, generating such private APIs poses a formidable conundrum for LLMs, as they inherently lack exposure to these private libraries during pre-training. To address this challenge, we propose a novel framework that emulates the process of programmers writing private code. This framework comprises two modules: APIFinder first retrieves potentially useful APIs from API documentation; and APICoder then leverages these retrieved APIs to generate private code. Specifically, APIFinder employs vector retrieval techniques and allows user involvement in the retrieval process. For APICoder, it can directly utilize off-the-shelf code generation models. To further cultivate explicit proficiency in invoking APIs from prompts, we continuously pre-train a reinforced version of APICoder, named CodeGenAPI. Our goal is to train the above two modules on vast public libraries, enabling generalization to private ones. Meanwhile, we create four private library benchmarks, including TorchDataEval, TorchDataComplexEval, MonkeyEval, and BeatNumEval, and meticulously handcraft test cases for each benchmark to support comprehensive evaluations. Numerous experiments on the four benchmarks consistently affirm the effectiveness of our approach. Furthermore, deeper analysis is also conducted to glean additional insights.

{{</citation>}}


## cs.NE (1)



### (58/59) Differential Evolution Algorithm based Hyper-Parameters Selection of Transformer Neural Network Model for Load Forecasting (Anuvab Sen et al., 2023)

{{<citation>}}

Anuvab Sen, Arul Rhik Mazumder, Udayon Sen. (2023)  
**Differential Evolution Algorithm based Hyper-Parameters Selection of Transformer Neural Network Model for Load Forecasting**  

---
Primary Category: cs.NE  
Categories: cs-LG, cs-NE, cs.NE  
Keywords: Attention, LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2307.15299v1)  

---


**ABSTRACT**  
Accurate load forecasting plays a vital role in numerous sectors, but accurately capturing the complex dynamics of dynamic power systems remains a challenge for traditional statistical models. For these reasons, time-series models (ARIMA) and deep-learning models (ANN, LSTM, GRU, etc.) are commonly deployed and often experience higher success. In this paper, we analyze the efficacy of the recently developed Transformer-based Neural Network model in Load forecasting. Transformer models have the potential to improve Load forecasting because of their ability to learn long-range dependencies derived from their Attention Mechanism. We apply several metaheuristics namely Differential Evolution to find the optimal hyperparameters of the Transformer-based Neural Network to produce accurate forecasts. Differential Evolution provides scalable, robust, global solutions to non-differentiable, multi-objective, or constrained optimization problems. Our work compares the proposed Transformer based Neural Network model integrated with different metaheuristic algorithms by their performance in Load forecasting based on numerical metrics such as Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE). Our findings demonstrate the potential of metaheuristic-enhanced Transformer-based Neural Network models in Load forecasting accuracy and provide optimal hyperparameters for each model.

{{</citation>}}


## eess.AS (1)



### (59/59) PCNN: A Lightweight Parallel Conformer Neural Network for Efficient Monaural Speech Enhancement (Xinmeng Xu et al., 2023)

{{<citation>}}

Xinmeng Xu, Weiping Tu, Yuhong Yang. (2023)  
**PCNN: A Lightweight Parallel Conformer Neural Network for Efficient Monaural Speech Enhancement**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.15251v1)  

---


**ABSTRACT**  
Convolutional neural networks (CNN) and Transformer have wildly succeeded in multimedia applications. However, more effort needs to be made to harmonize these two architectures effectively to satisfy speech enhancement. This paper aims to unify these two architectures and presents a Parallel Conformer for speech enhancement. In particular, the CNN and the self-attention (SA) in the Transformer are fully exploited for local format patterns and global structure representations. Based on the small receptive field size of CNN and the high computational complexity of SA, we specially designed a multi-branch dilated convolution (MBDC) and a self-channel-time-frequency attention (Self-CTFA) module. MBDC contains three convolutional layers with different dilation rates for the feature from local to non-local processing. Experimental results show that our method performs better than state-of-the-art methods in most evaluation criteria while maintaining the lowest model parameters.

{{</citation>}}
