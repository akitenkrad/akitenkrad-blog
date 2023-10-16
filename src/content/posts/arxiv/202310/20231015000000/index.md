---
draft: false
title: "arXiv @ 2023.10.15"
date: 2023-10-15
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.15"
    identifier: arxiv_20231015
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (17)](#cscv-17)
- [cs.RO (6)](#csro-6)
- [cs.CR (4)](#cscr-4)
- [cs.CL (33)](#cscl-33)
- [cs.LG (19)](#cslg-19)
- [cs.AI (15)](#csai-15)
- [cs.IR (1)](#csir-1)
- [cs.MM (1)](#csmm-1)
- [cs.GT (1)](#csgt-1)
- [stat.ML (1)](#statml-1)
- [eess.IV (2)](#eessiv-2)
- [cs.NI (2)](#csni-2)
- [cs.SI (3)](#cssi-3)
- [cs.DC (1)](#csdc-1)
- [cs.SD (3)](#cssd-3)
- [cs.MA (1)](#csma-1)
- [cs.SE (2)](#csse-2)
- [cs.HC (1)](#cshc-1)
- [stat.ME (1)](#statme-1)

## cs.CV (17)



### (1/114) Vision-by-Language for Training-Free Compositional Image Retrieval (Shyamgopal Karthik et al., 2023)

{{<citation>}}

Shyamgopal Karthik, Karsten Roth, Massimiliano Mancini, Zeynep Akata. (2023)  
**Vision-by-Language for Training-Free Compositional Image Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.09291v1)  

---


**ABSTRACT**  
Given an image and a target modification (e.g an image of the Eiffel tower and the text "without people and at night-time"), Compositional Image Retrieval (CIR) aims to retrieve the relevant target image in a database. While supervised approaches rely on annotating triplets that is costly (i.e. query image, textual modification, and target image), recent research sidesteps this need by using large-scale vision-language models (VLMs), performing Zero-Shot CIR (ZS-CIR). However, state-of-the-art approaches in ZS-CIR still require training task-specific, customized models over large amounts of image-text pairs. In this work, we propose to tackle CIR in a training-free manner via our Compositional Image Retrieval through Vision-by-Language (CIReVL), a simple, yet human-understandable and scalable pipeline that effectively recombines large-scale VLMs with large language models (LLMs). By captioning the reference image using a pre-trained generative VLM and asking a LLM to recompose the caption based on the textual target modification for subsequent retrieval via e.g. CLIP, we achieve modular language reasoning. In four ZS-CIR benchmarks, we find competitive, in-part state-of-the-art performance - improving over supervised methods. Moreover, the modularity of CIReVL offers simple scalability without re-training, allowing us to both investigate scaling laws and bottlenecks for ZS-CIR while easily scaling up to in parts more than double of previously reported results. Finally, we show that CIReVL makes CIR human-understandable by composing image and text in a modular fashion in the language domain, thereby making it intervenable, allowing to post-hoc re-align failure cases. Code will be released upon acceptance.

{{</citation>}}


### (2/114) SAIR: Learning Semantic-aware Implicit Representation (Canyu Zhang et al., 2023)

{{<citation>}}

Canyu Zhang, Xiaoguang Li, Qing Guo, Song Wang. (2023)  
**SAIR: Learning Semantic-aware Implicit Representation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.09285v1)  

---


**ABSTRACT**  
Implicit representation of an image can map arbitrary coordinates in the continuous domain to their corresponding color values, presenting a powerful capability for image reconstruction. Nevertheless, existing implicit representation approaches only focus on building continuous appearance mapping, ignoring the continuities of the semantic information across pixels. As a result, they can hardly achieve desired reconstruction results when the semantic information within input images is corrupted, for example, a large region misses. To address the issue, we propose to learn semantic-aware implicit representation (SAIR), that is, we make the implicit representation of each pixel rely on both its appearance and semantic information (\eg, which object does the pixel belong to). To this end, we propose a framework with two modules: (1) building a semantic implicit representation (SIR) for a corrupted image whose large regions miss. Given an arbitrary coordinate in the continuous domain, we can obtain its respective text-aligned embedding indicating the object the pixel belongs. (2) building an appearance implicit representation (AIR) based on the SIR. Given an arbitrary coordinate in the continuous domain, we can reconstruct its color whether or not the pixel is missed in the input. We validate the novel semantic-aware implicit representation method on the image inpainting task, and the extensive experiments demonstrate that our method surpasses state-of-the-art approaches by a significant margin.

{{</citation>}}


### (3/114) Transformer-based Multimodal Change Detection with Multitask Consistency Constraints (Biyuan Liu et al., 2023)

{{<citation>}}

Biyuan Liu, Huaixin Chen, Kun Li, Michael Ying Yang. (2023)  
**Transformer-based Multimodal Change Detection with Multitask Consistency Constraints**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.09276v1)  

---


**ABSTRACT**  
Change detection plays a fundamental role in Earth observation for analyzing temporal iterations over time. However, recent studies have largely neglected the utilization of multimodal data that presents significant practical and technical advantages compared to single-modal approaches. This research focuses on leveraging digital surface model (DSM) data and aerial images captured at different times for detecting change beyond 2D. We observe that the current change detection methods struggle with the multitask conflicts between semantic and height change detection tasks. To address this challenge, we propose an efficient Transformer-based network that learns shared representation between cross-dimensional inputs through cross-attention. It adopts a consistency constraint to establish the multimodal relationship, which involves obtaining pseudo change through height change thresholding and minimizing the difference between semantic and pseudo change within their overlapping regions. A DSM-to-image multimodal dataset encompassing three cities in the Netherlands was constructed. It lays a new foundation for beyond-2D change detection from cross-dimensional inputs. Compared to five state-of-the-art change detection methods, our model demonstrates consistent multitask superiority in terms of semantic and height change detection. Furthermore, the consistency strategy can be seamlessly adapted to the other methods, yielding promising improvements.

{{</citation>}}


### (4/114) Hypernymy Understanding Evaluation of Text-to-Image Models via WordNet Hierarchy (Anton Baryshnikov et al., 2023)

{{<citation>}}

Anton Baryshnikov, Max Ryabinin. (2023)  
**Hypernymy Understanding Evaluation of Text-to-Image Models via WordNet Hierarchy**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-LG, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.09247v1)  

---


**ABSTRACT**  
Text-to-image synthesis has recently attracted widespread attention due to rapidly improving quality and numerous practical applications. However, the language understanding capabilities of text-to-image models are still poorly understood, which makes it difficult to reason about prompt formulations that a given model would understand well. In this work, we measure the capability of popular text-to-image models to understand $\textit{hypernymy}$, or the "is-a" relation between words. We design two automatic metrics based on the WordNet semantic hierarchy and existing image classifiers pretrained on ImageNet. These metrics both enable broad quantitative comparison of linguistic capabilities for text-to-image models and offer a way of finding fine-grained qualitative differences, such as words that are unknown to models and thus are difficult for them to draw. We comprehensively evaluate popular text-to-image models, including GLIDE, Latent Diffusion, and Stable Diffusion, showing how our metrics can provide a better understanding of the individual strengths and weaknesses of these models.

{{</citation>}}


### (5/114) PaLI-3 Vision Language Models: Smaller, Faster, Stronger (Xi Chen et al., 2023)

{{<citation>}}

Xi Chen, Xiao Wang, Lucas Beyer, Alexander Kolesnikov, Jialin Wu, Paul Voigtlaender, Basil Mustafa, Sebastian Goodman, Ibrahim Alabdulmohsin, Piotr Padlewski, Daniel Salz, Xi Xiong, Daniel Vlasic, Filip Pavetic, Keran Rong, Tianli Yu, Daniel Keysers, Xiaohua Zhai, Radu Soricut. (2023)  
**PaLI-3 Vision Language Models: Smaller, Faster, Stronger**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.09199v1)  

---


**ABSTRACT**  
This paper presents PaLI-3, a smaller, faster, and stronger vision language model (VLM) that compares favorably to similar models that are 10x larger. As part of arriving at this strong performance, we compare Vision Transformer (ViT) models pretrained using classification objectives to contrastively (SigLIP) pretrained ones. We find that, while slightly underperforming on standard image classification benchmarks, SigLIP-based PaLI shows superior performance across various multimodal benchmarks, especially on localization and visually-situated text understanding. We scale the SigLIP image encoder up to 2 billion parameters, and achieves a new state-of-the-art on multilingual cross-modal retrieval. We hope that PaLI-3, at only 5B parameters, rekindles research on fundamental pieces of complex VLMs, and could fuel a new generation of scaled-up models.

{{</citation>}}


### (6/114) mnmDTW: An extension to Dynamic Time Warping for Camera-based Movement Error Localization (Sebastian Dill et al., 2023)

{{<citation>}}

Sebastian Dill, Maurice Rohr. (2023)  
**mnmDTW: An extension to Dynamic Time Warping for Camera-based Movement Error Localization**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2310.09170v1)  

---


**ABSTRACT**  
In this proof of concept, we use Computer Vision (CV) methods to extract pose information out of exercise videos. We then employ a modified version of Dynamic Time Warping (DTW) to calculate the deviation from a gold standard execution of the exercise. Specifically, we calculate the distance between each body part individually to get a more precise measure for exercise accuracy. We can show that exercise mistakes are clearly visible, identifiable and localizable through this metric.

{{</citation>}}


### (7/114) Equirectangular image construction method for standard CNNs for Semantic Segmentation (Haoqian Chen et al., 2023)

{{<citation>}}

Haoqian Chen, Jian Liu, Minghe Li, Kaiwen Jiang, Ziheng Xu, Rencheng Sun, Yi Sui. (2023)  
**Equirectangular image construction method for standard CNNs for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.09122v1)  

---


**ABSTRACT**  
360{\deg} spherical images have advantages of wide view field, and are typically projected on a planar plane for processing, which is known as equirectangular image. The object shape in equirectangular images can be distorted and lack translation invariance. In addition, there are few publicly dataset of equirectangular images with labels, which presents a challenge for standard CNNs models to process equirectangular images effectively. To tackle this problem, we propose a methodology for converting a perspective image into equirectangular image. The inverse transformation of the spherical center projection and the equidistant cylindrical projection are employed. This enables the standard CNNs to learn the distortion features at different positions in the equirectangular image and thereby gain the ability to semantically the equirectangular image. The parameter, {\phi}, which determines the projection position of the perspective image, has been analyzed using various datasets and models, such as UNet, UNet++, SegNet, PSPNet, and DeepLab v3+. The experiments demonstrate that an optimal value of {\phi} for effective semantic segmentation of equirectangular images is 6{\pi}/16 for standard CNNs. Compared with the other three types of methods (supervised learning, unsupervised learning and data augmentation), the method proposed in this paper has the best average IoU value of 43.76%. This value is 23.85%, 10.7% and 17.23% higher than those of other three methods, respectively.

{{</citation>}}


### (8/114) Timestamp-supervised Wearable-based Activity Segmentation and Recognition with Contrastive Learning and Order-Preserving Optimal Transport (Songpengcheng Xia et al., 2023)

{{<citation>}}

Songpengcheng Xia, Lei Chu, Ling Pei, Jiarui Yang, Wenxian Yu, Robert C. Qiu. (2023)  
**Timestamp-supervised Wearable-based Activity Segmentation and Recognition with Contrastive Learning and Order-Preserving Optimal Transport**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.09114v1)  

---


**ABSTRACT**  
Human activity recognition (HAR) with wearables is one of the serviceable technologies in ubiquitous and mobile computing applications. The sliding-window scheme is widely adopted while suffering from the multi-class windows problem. As a result, there is a growing focus on joint segmentation and recognition with deep-learning methods, aiming at simultaneously dealing with HAR and time-series segmentation issues. However, obtaining the full activity annotations of wearable data sequences is resource-intensive or time-consuming, while unsupervised methods yield poor performance. To address these challenges, we propose a novel method for joint activity segmentation and recognition with timestamp supervision, in which only a single annotated sample is needed in each activity segment. However, the limited information of sparse annotations exacerbates the gap between recognition and segmentation tasks, leading to sub-optimal model performance. Therefore, the prototypes are estimated by class-activation maps to form a sample-to-prototype contrast module for well-structured embeddings. Moreover, with the optimal transport theory, our approach generates the sample-level pseudo-labels that take advantage of unlabeled data between timestamp annotations for further performance improvement. Comprehensive experiments on four public HAR datasets demonstrate that our model trained with timestamp supervision is superior to the state-of-the-art weakly-supervised methods and achieves comparable performance to the fully-supervised approaches.

{{</citation>}}


### (9/114) A Spatial-Temporal Dual-Mode Mixed Flow Network for Panoramic Video Salient Object Detection (Xiaolei Chen et al., 2023)

{{<citation>}}

Xiaolei Chen, Pengcheng Zhang, Zelong Du, Ishfaq Ahmad. (2023)  
**A Spatial-Temporal Dual-Mode Mixed Flow Network for Panoramic Video Salient Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Object Detection  
[Paper Link](http://arxiv.org/abs/2310.09016v1)  

---


**ABSTRACT**  
Salient object detection (SOD) in panoramic video is still in the initial exploration stage. The indirect application of 2D video SOD method to the detection of salient objects in panoramic video has many unmet challenges, such as low detection accuracy, high model complexity, and poor generalization performance. To overcome these hurdles, we design an Inter-Layer Attention (ILA) module, an Inter-Layer weight (ILW) module, and a Bi-Modal Attention (BMA) module. Based on these modules, we propose a Spatial-Temporal Dual-Mode Mixed Flow Network (STDMMF-Net) that exploits the spatial flow of panoramic video and the corresponding optical flow for SOD. First, the ILA module calculates the attention between adjacent level features of consecutive frames of panoramic video to improve the accuracy of extracting salient object features from the spatial flow. Then, the ILW module quantifies the salient object information contained in the features of each level to improve the fusion efficiency of the features of each level in the mixed flow. Finally, the BMA module improves the detection accuracy of STDMMF-Net. A large number of subjective and objective experimental results testify that the proposed method demonstrates better detection accuracy than the state-of-the-art (SOTA) methods. Moreover, the comprehensive performance of the proposed method is better in terms of memory required for model inference, testing time, complexity, and generalization performance.

{{</citation>}}


### (10/114) VCL Challenges 2023 at ICCV 2023 Technical Report: Bi-level Adaptation Method for Test-time Adaptive Object Detection (Chenyu Lin et al., 2023)

{{<citation>}}

Chenyu Lin, Yusheng He, Zhengqing Zang, Chenwei Tang, Tao Wang, Jiancheng Lv. (2023)  
**VCL Challenges 2023 at ICCV 2023 Technical Report: Bi-level Adaptation Method for Test-time Adaptive Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.08986v1)  

---


**ABSTRACT**  
This report outlines our team's participation in VCL Challenges B Continual Test_time Adaptation, focusing on the technical details of our approach. Our primary focus is Testtime Adaptation using bi_level adaptations, encompassing image_level and detector_level adaptations. At the image level, we employ adjustable parameterbased image filters, while at the detector level, we leverage adjustable parameterbased mean teacher modules. Ultimately, through the utilization of these bi_level adaptations, we have achieved a remarkable 38.3% mAP on the target domain of the test set within VCL Challenges B. It is worth noting that the minimal drop in mAP, is mearly 4.2%, and the overall performance is 32.5% mAP.

{{</citation>}}


### (11/114) UniParser: Multi-Human Parsing with Unified Correlation Representation Learning (Jiaming Chu et al., 2023)

{{<citation>}}

Jiaming Chu, Lei Jin, Junliang Xing, Jian Zhao. (2023)  
**UniParser: Multi-Human Parsing with Unified Correlation Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2310.08984v1)  

---


**ABSTRACT**  
Multi-human parsing is an image segmentation task necessitating both instance-level and fine-grained category-level information. However, prior research has typically processed these two types of information through separate branches and distinct output formats, leading to inefficient and redundant frameworks. This paper introduces UniParser, which integrates instance-level and category-level representations in three key aspects: 1) we propose a unified correlation representation learning approach, allowing our network to learn instance and category features within the cosine space; 2) we unify the form of outputs of each modules as pixel-level segmentation results while supervising instance and category features using a homogeneous label accompanied by an auxiliary loss; and 3) we design a joint optimization procedure to fuse instance and category representations. By virtual of unifying instance-level and category-level output, UniParser circumvents manually designed post-processing techniques and surpasses state-of-the-art methods, achieving 49.3% AP on MHPv2.0 and 60.4% AP on CIHP. We will release our source code, pretrained models, and online demos to facilitate future studies.

{{</citation>}}


### (12/114) Federated Class-Incremental Learning with Prompting (Jiale Liu et al., 2023)

{{<citation>}}

Jiale Liu, Yu-Wei Zhan, Chong-Yu Zhang, Xin Luo, Zhen-Duo Chen, Yinwei Wei, Xin-Shun Xu. (2023)  
**Federated Class-Incremental Learning with Prompting**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.08948v1)  

---


**ABSTRACT**  
As Web technology continues to develop, it has become increasingly common to use data stored on different clients. At the same time, federated learning has received widespread attention due to its ability to protect data privacy when let models learn from data which is distributed across various clients. However, most existing works assume that the client's data are fixed. In real-world scenarios, such an assumption is most likely not true as data may be continuously generated and new classes may also appear. To this end, we focus on the practical and challenging federated class-incremental learning (FCIL) problem. For FCIL, the local and global models may suffer from catastrophic forgetting on old classes caused by the arrival of new classes and the data distributions of clients are non-independent and identically distributed (non-iid).   In this paper, we propose a novel method called Federated Class-Incremental Learning with PrompTing (FCILPT). Given the privacy and limited memory, FCILPT does not use a rehearsal-based buffer to keep exemplars of old data. We choose to use prompts to ease the catastrophic forgetting of the old classes. Specifically, we encode the task-relevant and task-irrelevant knowledge into prompts, preserving the old and new knowledge of the local clients and solving the problem of catastrophic forgetting. We first sort the task information in the prompt pool in the local clients to align the task information on different clients before global aggregation. It ensures that the same task's knowledge are fully integrated, solving the problem of non-iid caused by the lack of classes among different clients in the same incremental task. Experiments on CIFAR-100, Mini-ImageNet, and Tiny-ImageNet demonstrate that FCILPT achieves significant accuracy improvements over the state-of-the-art methods.

{{</citation>}}


### (13/114) Towards Interpretable Controllability in Object-Centric Learning (Jinwoo Kim et al., 2023)

{{<citation>}}

Jinwoo Kim, Janghyuk Choi, Jaehyun Kang, Changyeon Lee, Ho-Jin Choi, Seon Joo Kim. (2023)  
**Towards Interpretable Controllability in Object-Centric Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Augmentation  
[Paper Link](http://arxiv.org/abs/2310.08929v1)  

---


**ABSTRACT**  
The binding problem in artificial neural networks is actively explored with the goal of achieving human-level recognition skills through the comprehension of the world in terms of symbol-like entities. Especially in the field of computer vision, object-centric learning (OCL) is extensively researched to better understand complex scenes by acquiring object representations or slots. While recent studies in OCL have made strides with complex images or videos, the interpretability and interactivity over object representation remain largely uncharted, still holding promise in the field of OCL. In this paper, we introduce a novel method, Slot Attention with Image Augmentation (SlotAug), to explore the possibility of learning interpretable controllability over slots in a self-supervised manner by utilizing an image augmentation strategy. We also devise the concept of sustainability in controllable slots by introducing iterative and reversible controls over slots with two proposed submethods: Auxiliary Identity Manipulation and Slot Consistency Loss. Extensive empirical studies and theoretical validation confirm the effectiveness of our approach, offering a novel capability for interpretable and sustainable control of object representations. Code will be available soon.

{{</citation>}}


### (14/114) Rank-DETR for High Quality Object Detection (Yifan Pu et al., 2023)

{{<citation>}}

Yifan Pu, Weicong Liang, Yiduo Hao, Yuhui Yuan, Yukang Yang, Chao Zhang, Han Hu, Gao Huang. (2023)  
**Rank-DETR for High Quality Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.08854v1)  

---


**ABSTRACT**  
Modern detection transformers (DETRs) use a set of object queries to predict a list of bounding boxes, sort them by their classification confidence scores, and select the top-ranked predictions as the final detection results for the given input image. A highly performant object detector requires accurate ranking for the bounding box predictions. For DETR-based detectors, the top-ranked bounding boxes suffer from less accurate localization quality due to the misalignment between classification scores and localization accuracy, thus impeding the construction of high-quality detectors. In this work, we introduce a simple and highly performant DETR-based object detector by proposing a series of rank-oriented designs, combinedly called Rank-DETR. Our key contributions include: (i) a rank-oriented architecture design that can prompt positive predictions and suppress the negative ones to ensure lower false positive rates, as well as (ii) a rank-oriented loss function and matching cost design that prioritizes predictions of more accurate localization accuracy during ranking to boost the AP under high IoU thresholds. We apply our method to improve the recent SOTA methods (e.g., H-DETR and DINO-DETR) and report strong COCO object detection results when using different backbones such as ResNet-$50$, Swin-T, and Swin-L, demonstrating the effectiveness of our approach. Code is available at \url{https://github.com/LeapLabTHU/Rank-DETR}.

{{</citation>}}


### (15/114) Revisiting Multi-modal 3D Semantic Segmentation in Real-world Autonomous Driving (Feng Jiang et al., 2023)

{{<citation>}}

Feng Jiang, Chaoping Tu, Gang Zhang, Jun Li, Hanqing Huang, Junyu Lin, Di Feng, Jian Pu. (2023)  
**Revisiting Multi-modal 3D Semantic Segmentation in Real-world Autonomous Driving**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.08826v1)  

---


**ABSTRACT**  
LiDAR and camera are two critical sensors for multi-modal 3D semantic segmentation and are supposed to be fused efficiently and robustly to promise safety in various real-world scenarios. However, existing multi-modal methods face two key challenges: 1) difficulty with efficient deployment and real-time execution; and 2) drastic performance degradation under weak calibration between LiDAR and cameras. To address these challenges, we propose CPGNet-LCF, a new multi-modal fusion framework extending the LiDAR-only CPGNet. CPGNet-LCF solves the first challenge by inheriting the easy deployment and real-time capabilities of CPGNet. For the second challenge, we introduce a novel weak calibration knowledge distillation strategy during training to improve the robustness against the weak calibration. CPGNet-LCF achieves state-of-the-art performance on the nuScenes and SemanticKITTI benchmarks. Remarkably, it can be easily deployed to run in 20ms per frame on a single Tesla V100 GPU using TensorRT TF16 mode. Furthermore, we benchmark performance over four weak calibration levels, demonstrating the robustness of our proposed approach.

{{</citation>}}


### (16/114) From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models (Dongsheng Jiang et al., 2023)

{{<citation>}}

Dongsheng Jiang, Yuchen Liu, Songlin Liu, Xiaopeng Zhang, Jin Li, Hongkai Xiong, Qi Tian. (2023)  
**From CLIP to DINO: Visual Encoders Shout in Multi-modal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.08825v1)  

---


**ABSTRACT**  
Multi-modal Large Language Models (MLLMs) have made significant strides in expanding the capabilities of Large Language Models (LLMs) through the incorporation of visual perception interfaces. Despite the emergence of exciting applications and the availability of diverse instruction tuning data, existing approaches often rely on CLIP or its variants as the visual branch, and merely extract features from the deep layers. However, these methods lack a comprehensive analysis of the visual encoders in MLLMs. In this paper, we conduct an extensive investigation into the effectiveness of different vision encoders within MLLMs. Our findings reveal that the shallow layer features of CLIP offer particular advantages for fine-grained tasks such as grounding and region understanding. Surprisingly, the vision-only model DINO, which is not pretrained with text-image alignment, demonstrates promising performance as a visual branch within MLLMs. By simply equipping it with an MLP layer for alignment, DINO surpasses CLIP in fine-grained related perception tasks. Building upon these observations, we propose a simple yet effective feature merging strategy, named COMM, that integrates CLIP and DINO with Multi-level features Merging, to enhance the visual capabilities of MLLMs. We evaluate COMM through comprehensive experiments on a wide range of benchmarks, including image captioning, visual question answering, visual grounding, and object hallucination. Experimental results demonstrate the superior performance of COMM compared to existing methods, showcasing its enhanced visual capabilities within MLLMs. Code will be made available at https://github.com/YuchenLiu98/COMM.

{{</citation>}}


### (17/114) Incremental Object Detection with CLIP (Yupeng He et al., 2023)

{{<citation>}}

Yupeng He, Ziyue Huang, Qingjie Liu, Yunhong Wang. (2023)  
**Incremental Object Detection with CLIP**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.08815v1)  

---


**ABSTRACT**  
In the incremental detection task, unlike the incremental classification task, data ambiguity exists due to the possibility of an image having different labeled bounding boxes in multiple continuous learning stages. This phenomenon often impairs the model's ability to learn new classes. However, the forward compatibility of the model is less considered in existing work, which hinders the model's suitability for incremental learning. To overcome this obstacle, we propose to use a language-visual model such as CLIP to generate text feature embeddings for different class sets, which enhances the feature space globally. We then employ the broad classes to replace the unavailable novel classes in the early learning stage to simulate the actual incremental scenario. Finally, we use the CLIP image encoder to identify potential objects in the proposals, which are classified into the background by the model. We modify the background labels of those proposals to known classes and add the boxes to the training set to alleviate the problem of data ambiguity. We evaluate our approach on various incremental learning settings on the PASCAL VOC 2007 dataset, and our approach outperforms state-of-the-art methods, particularly for the new classes.

{{</citation>}}


## cs.RO (6)



### (18/114) An Unbiased Look at Datasets for Visuo-Motor Pre-Training (Sudeep Dasari et al., 2023)

{{<citation>}}

Sudeep Dasari, Mohan Kumar Srirama, Unnat Jain, Abhinav Gupta. (2023)  
**An Unbiased Look at Datasets for Visuo-Motor Pre-Training**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.09289v1)  

---


**ABSTRACT**  
Visual representation learning hold great promise for robotics, but is severely hampered by the scarcity and homogeneity of robotics datasets. Recent works address this problem by pre-training visual representations on large-scale but out-of-domain data (e.g., videos of egocentric interactions) and then transferring them to target robotics tasks. While the field is heavily focused on developing better pre-training algorithms, we find that dataset choice is just as important to this paradigm's success. After all, the representation can only learn the structures or priors present in the pre-training dataset. To this end, we flip the focus on algorithms, and instead conduct a dataset centric analysis of robotic pre-training. Our findings call into question some common wisdom in the field. We observe that traditional vision datasets (like ImageNet, Kinetics and 100 Days of Hands) are surprisingly competitive options for visuo-motor representation learning, and that the pre-training dataset's image distribution matters more than its size. Finally, we show that common simulation benchmarks are not a reliable proxy for real world performance and that simple regularization strategies can dramatically improve real world policy learning. https://data4robotics.github.io

{{</citation>}}


### (19/114) Interactive Navigation in Environments with Traversable Obstacles Using Large Language and Vision-Language Models (Zhen Zhang et al., 2023)

{{<citation>}}

Zhen Zhang, Anran Lin, Chun Wai Wong, Xiangyu Chu, Qi Dou, K. W. Samuel Au. (2023)  
**Interactive Navigation in Environments with Traversable Obstacles Using Large Language and Vision-Language Models**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08873v1)  

---


**ABSTRACT**  
This paper proposes an interactive navigation framework by using large language and vision-language models, allowing robots to navigate in environments with traversable obstacles. We utilize the large language model (GPT-3.5) and the open-set Vision-language Model (Grounding DINO) to create an action-aware costmap to perform effective path planning without fine-tuning. With the large models, we can achieve an end-to-end system from textual instructions like "Can you pass through the curtains to deliver medicines to me?", to bounding boxes (e.g., curtains) with action-aware attributes. They can be used to segment LiDAR point clouds into two parts: traversable and untraversable parts, and then an action-aware costmap is constructed for generating a feasible path. The pre-trained large models have great generalization ability and do not require additional annotated data for training, allowing fast deployment in the interactive navigation tasks. We choose to use multiple traversable objects such as curtains and grasses for verification by instructing the robot to traverse them. Besides, traversing curtains in a medical scenario was tested. All experimental results demonstrated the proposed framework's effectiveness and adaptability to diverse environments.

{{</citation>}}


### (20/114) Open X-Embodiment: Robotic Learning Datasets and RT-X Models (Abhishek Padalkar et al., 2023)

{{<citation>}}

Abhishek Padalkar, Acorn Pooley, Ajinkya Jain, Alex Bewley, Alex Herzog, Alex Irpan, Alexander Khazatsky, Anant Rai, Anikait Singh, Anthony Brohan, Antonin Raffin, Ayzaan Wahid, Ben Burgess-Limerick, Beomjoon Kim, Bernhard Schölkopf, Brian Ichter, Cewu Lu, Charles Xu, Chelsea Finn, Chenfeng Xu, Cheng Chi, Chenguang Huang, Christine Chan, Chuer Pan, Chuyuan Fu, Coline Devin, Danny Driess, Deepak Pathak, Dhruv Shah, Dieter Büchler, Dmitry Kalashnikov, Dorsa Sadigh, Edward Johns, Federico Ceola, Fei Xia, Freek Stulp, Gaoyue Zhou, Gaurav S. Sukhatme, Gautam Salhotra, Ge Yan, Giulio Schiavi, Gregory Kahn, Hao Su, Hao-Shu Fang, Haochen Shi, Heni Ben Amor, Henrik I Christensen, Hiroki Furuta, Homer Walke, Hongjie Fang, Igor Mordatch, Ilija Radosavovic, Isabel Leal, Jacky Liang, Jad Abou-Chakra, Jaehyung Kim, Jan Peters, Jan Schneider, Jasmine Hsu, Jeannette Bohg, Jeffrey Bingham, Jiajun Wu, Jialin Wu, Jianlan Luo, Jiayuan Gu, Jie Tan, Jihoon Oh, Jitendra Malik, Jonathan Tompson, Jonathan Yang, Joseph J. Lim, João Silvério, Junhyek Han, Kanishka Rao, Karl Pertsch, Karol Hausman, Keegan Go, Keerthana Gopalakrishnan, Ken Goldberg, Kendra Byrne, Kenneth Oslund, Kento Kawaharazuka, Kevin Zhang, Krishan Rana, Krishnan Srinivasan, Lawrence Yunliang Chen, Lerrel Pinto, Liam Tan, Lionel Ott, Lisa Lee, Masayoshi Tomizuka, Maximilian Du, Michael Ahn, Mingtong Zhang, Mingyu Ding, Mohan Kumar Srirama, Mohit Sharma, Moo Jin Kim, Naoaki Kanazawa, Nicklas Hansen, Nicolas Heess, Nikhil J Joshi, Niko Suenderhauf, Norman Di Palo, Nur Muhammad Mahi Shafiullah, Oier Mees, Oliver Kroemer, Pannag R Sanketi, Paul Wohlhart, Peng Xu, Pierre Sermanet, Priya Sundaresan, Quan Vuong, Rafael Rafailov, Ran Tian, Ria Doshi, Roberto Martín-Martín, Russell Mendonca, Rutav Shah, Ryan Hoque, Ryan Julian, Samuel Bustamante, Sean Kirmani, Sergey Levine, Sherry Moore, Shikhar Bahl, Shivin Dass, Shubham Sonawani, Shuran Song, Sichun Xu, Siddhant Haldar, Simeon Adebola, Simon Guist, Soroush Nasiriany, Stefan Schaal, Stefan Welker, Stephen Tian, Sudeep Dasari, Suneel Belkhale, Takayuki Osa, Tatsuya Harada, Tatsuya Matsushima, Ted Xiao, Tianhe Yu, Tianli Ding, Todor Davchev, Tony Z. Zhao, Travis Armstrong, Trevor Darrell, Vidhi Jain, Vincent Vanhoucke, Wei Zhan, Wenxuan Zhou, Wolfram Burgard, Xi Chen, Xiaolong Wang, Xinghao Zhu, Xuanlin Li, Yao Lu, Yevgen Chebotar, Yifan Zhou, Yifeng Zhu, Ying Xu, Yixuan Wang, Yonatan Bisk, Yoonyoung Cho, Youngwoon Lee, Yuchen Cui, Yueh-Hua Wu, Yujin Tang, Yuke Zhu, Yunzhu Li, Yusuke Iwasawa, Yutaka Matsuo, Zhuo Xu, Zichen Jeff Cui. (2023)  
**Open X-Embodiment: Robotic Learning Datasets and RT-X Models**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Computer Vision, NLP  
[Paper Link](http://arxiv.org/abs/2310.08864v1)  

---


**ABSTRACT**  
Large, high-capacity models trained on diverse datasets have shown remarkable successes on efficiently tackling downstream applications. In domains from NLP to Computer Vision, this has led to a consolidation of pretrained models, with general pretrained backbones serving as a starting point for many applications. Can such a consolidation happen in robotics? Conventionally, robotic learning methods train a separate model for every application, every robot, and even every environment. Can we instead train generalist X-robot policy that can be adapted efficiently to new robots, tasks, and environments? In this paper, we provide datasets in standardized data formats and models to make it possible to explore this possibility in the context of robotic manipulation, alongside experimental results that provide an example of effective X-robot policies. We assemble a dataset from 22 different robots collected through a collaboration between 21 institutions, demonstrating 527 skills (160266 tasks). We show that a high-capacity model trained on this data, which we call RT-X, exhibits positive transfer and improves the capabilities of multiple robots by leveraging experience from other platforms.

{{</citation>}}


### (21/114) A Framework for Few-Shot Policy Transfer through Observation Mapping and Behavior Cloning (Yash Shukla et al., 2023)

{{<citation>}}

Yash Shukla, Bharat Kesari, Shivam Goel, Robert Wright, Jivko Sinapov. (2023)  
**A Framework for Few-Shot Policy Transfer through Observation Mapping and Behavior Cloning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Few-Shot, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.08836v1)  

---


**ABSTRACT**  
Despite recent progress in Reinforcement Learning for robotics applications, many tasks remain prohibitively difficult to solve because of the expensive interaction cost. Transfer learning helps reduce the training time in the target domain by transferring knowledge learned in a source domain. Sim2Real transfer helps transfer knowledge from a simulated robotic domain to a physical target domain. Knowledge transfer reduces the time required to train a task in the physical world, where the cost of interactions is high. However, most existing approaches assume exact correspondence in the task structure and the physical properties of the two domains. This work proposes a framework for Few-Shot Policy Transfer between two domains through Observation Mapping and Behavior Cloning. We use Generative Adversarial Networks (GANs) along with a cycle-consistency loss to map the observations between the source and target domains and later use this learned mapping to clone the successful source task behavior policy to the target domain. We observe successful behavior policy transfer with limited target task interactions and in cases where the source and target task are semantically dissimilar.

{{</citation>}}


### (22/114) Urban Drone Navigation: Autoencoder Learning Fusion for Aerodynamics (Jiaohao Wu et al., 2023)

{{<citation>}}

Jiaohao Wu, Yang Ye, Jing Du. (2023)  
**Urban Drone Navigation: Autoencoder Learning Fusion for Aerodynamics**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2310.08830v1)  

---


**ABSTRACT**  
Drones are vital for urban emergency search and rescue (SAR) due to the challenges of navigating dynamic environments with obstacles like buildings and wind. This paper presents a method that combines multi-objective reinforcement learning (MORL) with a convolutional autoencoder to improve drone navigation in urban SAR. The approach uses MORL to achieve multiple goals and the autoencoder for cost-effective wind simulations. By utilizing imagery data of urban layouts, the drone can autonomously make navigation decisions, optimize paths, and counteract wind effects without traditional sensors. Tested on a New York City model, this method enhances drone SAR operations in complex urban settings.

{{</citation>}}


### (23/114) DexCatch: Learning to Catch Arbitrary Objects with Dexterous Hands (Fengbo Lan et al., 2023)

{{<citation>}}

Fengbo Lan, Shengjie Wang, Yunzhe Zhang, Haotian Xu, Oluwatosin Oseni, Yang Gao, Tao Zhang. (2023)  
**DexCatch: Learning to Catch Arbitrary Objects with Dexterous Hands**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.08809v1)  

---


**ABSTRACT**  
Achieving human-like dexterous manipulation remains a crucial area of research in robotics. Current research focuses on improving the success rate of pick-and-place tasks. Compared with pick-and-place, throw-catching behavior has the potential to increase picking speed without transporting objects to their destination. However, dynamic dexterous manipulation poses a major challenge for stable control due to a large number of dynamic contacts. In this paper, we propose a Stability-Constrained Reinforcement Learning (SCRL) algorithm to learn to catch diverse objects with dexterous hands. The SCRL algorithm outperforms baselines by a large margin, and the learned policies show strong zero-shot transfer performance on unseen objects. Remarkably, even though the object in a hand facing sideward is extremely unstable due to the lack of support from the palm, our method can still achieve a high level of success in the most challenging task. Video demonstrations of learned behaviors and the code can be found on the supplementary website.

{{</citation>}}


## cs.CR (4)



### (24/114) User Inference Attacks on Large Language Models (Nikhil Kandpal et al., 2023)

{{<citation>}}

Nikhil Kandpal, Krishna Pillutla, Alina Oprea, Peter Kairouz, Christopher A. Choquette-Choo, Zheng Xu. (2023)  
**User Inference Attacks on Large Language Models**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs-LG, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.09266v1)  

---


**ABSTRACT**  
Fine-tuning is a common and effective method for tailoring large language models (LLMs) to specialized tasks and applications. In this paper, we study the privacy implications of fine-tuning LLMs on user data. To this end, we define a realistic threat model, called user inference, wherein an attacker infers whether or not a user's data was used for fine-tuning. We implement attacks for this threat model that require only a small set of samples from a user (possibly different from the samples used for training) and black-box access to the fine-tuned LLM. We find that LLMs are susceptible to user inference attacks across a variety of fine-tuning datasets, at times with near perfect attack success rates. Further, we investigate which properties make users vulnerable to user inference, finding that outlier users (i.e. those with data distributions sufficiently different from other users) and users who contribute large quantities of data are most susceptible to attack. Finally, we explore several heuristics for mitigating privacy attacks. We find that interventions in the training algorithm, such as batch or per-example gradient clipping and early stopping fail to prevent user inference. However, limiting the number of fine-tuning samples from a single user can reduce attack effectiveness, albeit at the cost of reducing the total amount of fine-tuning data.

{{</citation>}}


### (25/114) Tikuna: An Ethereum Blockchain Network Security Monitoring System (Andres Gomez Ramirez et al., 2023)

{{<citation>}}

Andres Gomez Ramirez, Loui Al Sardy, Francis Gomez Ramirez. (2023)  
**Tikuna: An Ethereum Blockchain Network Security Monitoring System**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-DC, cs.CR  
Keywords: LSTM, Network Security, Security  
[Paper Link](http://arxiv.org/abs/2310.09193v1)  

---


**ABSTRACT**  
Blockchain security is becoming increasingly relevant in today's cyberspace as it extends its influence in many industries. This paper focuses on protecting the lowest level layer in the blockchain, particularly the P2P network that allows the nodes to communicate and share information. The P2P network layer may be vulnerable to several families of attacks, such as Distributed Denial of Service (DDoS), eclipse attacks, or Sybil attacks. This layer is prone to threats inherited from traditional P2P networks, and it must be analyzed and understood by collecting data and extracting insights from the network behavior to reduce those risks. We introduce Tikuna, an open-source tool for monitoring and detecting potential attacks on the Ethereum blockchain P2P network, at an early stage. Tikuna employs an unsupervised Long Short-Term Memory (LSTM) method based on Recurrent Neural Network (RNN) to detect attacks and alert users. Empirical results indicate that the proposed approach significantly improves detection performance, with the ability to detect and classify attacks, including eclipse attacks, Covert Flash attacks, and others that target the Ethereum blockchain P2P network layer, with high accuracy. Our research findings demonstrate that Tikuna is a valuable security tool for assisting operators to efficiently monitor and safeguard the status of Ethereum validators and the wider P2P network

{{</citation>}}


### (26/114) Privacy-Preserving Encrypted Low-Dose CT Denoising (Ziyuan Yang et al., 2023)

{{<citation>}}

Ziyuan Yang, Huijie Huangfu, Maosong Ran, Zhiwen Wang, Hui Yu, Yi Zhang. (2023)  
**Privacy-Preserving Encrypted Low-Dose CT Denoising**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2310.09101v1)  

---


**ABSTRACT**  
Deep learning (DL) has made significant advancements in tomographic imaging, particularly in low-dose computed tomography (LDCT) denoising. A recent trend involves servers training powerful models with large amounts of self-collected private data and providing application programming interfaces (APIs) for users, such as Chat-GPT. To avoid model leakage, users are required to upload their data to the server model, but this way raises public concerns about the potential risk of privacy disclosure, especially for medical data. Hence, to alleviate related concerns, in this paper, we propose to directly denoise LDCT in the encrypted domain to achieve privacy-preserving cloud services without exposing private data to the server. To this end, we employ homomorphic encryption to encrypt private LDCT data, which is then transferred to the server model trained with plaintext LDCT for further denoising. However, since traditional operations, such as convolution and linear transformation, in DL methods cannot be directly used in the encrypted domain, we transform the fundamental mathematic operations in the plaintext domain into the operations in the encrypted domain. In addition, we present two interactive frameworks for linear and nonlinear models in this paper, both of which can achieve lossless operating. In this way, the proposed methods can achieve two merits, the data privacy is well protected and the server model is free from the risk of model leakage. Moreover, we provide theoretical proof to validate the lossless property of our framework. Finally, experiments were conducted to demonstrate that the transferred contents are well protected and cannot be reconstructed. The code will be released once the paper is accepted.

{{</citation>}}


### (27/114) Log Anomaly Detection on EuXFEL Nodes (Antonin Sulc et al., 2023)

{{<citation>}}

Antonin Sulc, Annika Eichler, Tim Wilksen. (2023)  
**Log Anomaly Detection on EuXFEL Nodes**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.08951v1)  

---


**ABSTRACT**  
This article introduces a method to detect anomalies in the log data generated by control system nodes at the European XFEL accelerator. The primary aim of this proposed method is to provide operators a comprehensive understanding of the availability, status, and problems specific to each node. This information is vital for ensuring the smooth operation. The sequential nature of logs and the absence of a rich text corpus that is specific to our nodes poses significant limitations for traditional and learning-based approaches for anomaly detection. To overcome this limitation, we propose a method that uses word embedding and models individual nodes as a sequence of these vectors that commonly co-occur, using a Hidden Markov Model (HMM). We score individual log entries by computing a probability ratio between the probability of the full log sequence including the new entry and the probability of just the previous log entries, without the new entry. This ratio indicates how probable the sequence becomes when the new entry is added. The proposed approach can detect anomalies by scoring and ranking log entries from EuXFEL nodes where entries that receive high scores are potential anomalies that do not fit the routine of the node. This method provides a warning system to alert operators about these irregular log events that may indicate issues.

{{</citation>}}


## cs.CL (33)



### (28/114) PromptRE: Weakly-Supervised Document-Level Relation Extraction via Prompting-Based Data Programming (Chufan Gao et al., 2023)

{{<citation>}}

Chufan Gao, Xulin Fan, Jimeng Sun, Xuan Wang. (2023)  
**PromptRE: Weakly-Supervised Document-Level Relation Extraction via Prompting-Based Data Programming**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Relation Extraction  
[Paper Link](http://arxiv.org/abs/2310.09265v1)  

---


**ABSTRACT**  
Relation extraction aims to classify the relationships between two entities into pre-defined categories. While previous research has mainly focused on sentence-level relation extraction, recent studies have expanded the scope to document-level relation extraction. Traditional relation extraction methods heavily rely on human-annotated training data, which is time-consuming and labor-intensive. To mitigate the need for manual annotation, recent weakly-supervised approaches have been developed for sentence-level relation extraction while limited work has been done on document-level relation extraction. Weakly-supervised document-level relation extraction faces significant challenges due to an imbalanced number "no relation" instances and the failure of directly probing pretrained large language models for document relation extraction. To address these challenges, we propose PromptRE, a novel weakly-supervised document-level relation extraction method that combines prompting-based techniques with data programming. Furthermore, PromptRE incorporates the label distribution and entity types as prior knowledge to improve the performance. By leveraging the strengths of both prompting and data programming, PromptRE achieves improved performance in relation classification and effectively handles the "no relation" problem. Experimental results on ReDocRED, a benchmark dataset for document-level relation extraction, demonstrate the superiority of PromptRE over baseline approaches.

{{</citation>}}


### (29/114) Table-GPT: Table-tuned GPT for Diverse Table Tasks (Peng Li et al., 2023)

{{<citation>}}

Peng Li, Yeye He, Dror Yashar, Weiwei Cui, Song Ge, Haidong Zhang, Danielle Rifinski Fainman, Dongmei Zhang, Surajit Chaudhuri. (2023)  
**Table-GPT: Table-tuned GPT for Diverse Table Tasks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-DB, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2310.09263v1)  

---


**ABSTRACT**  
Language models, such as GPT-3.5 and ChatGPT, demonstrate remarkable abilities to follow diverse human instructions and perform a wide range of tasks. However, when probing language models using a range of basic table-understanding tasks, we observe that today's language models are still sub-optimal in many table-related tasks, likely because they are pre-trained predominantly on \emph{one-dimensional} natural-language texts, whereas relational tables are \emph{two-dimensional} objects.   In this work, we propose a new "\emph{table-tuning}" paradigm, where we continue to train/fine-tune language models like GPT-3.5 and ChatGPT, using diverse table-tasks synthesized from real tables as training data, with the goal of enhancing language models' ability to understand tables and perform table tasks. We show that our resulting Table-GPT models demonstrate (1) better \emph{table-understanding} capabilities, by consistently outperforming the vanilla GPT-3.5 and ChatGPT, on a wide-range of table tasks, including holdout unseen tasks, and (2) strong \emph{generalizability}, in its ability to respond to diverse human instructions to perform new table-tasks, in a manner similar to GPT-3.5 and ChatGPT.

{{</citation>}}


### (30/114) Precedent-Enhanced Legal Judgment Prediction with LLM and Domain-Model Collaboration (Yiquan Wu et al., 2023)

{{<citation>}}

Yiquan Wu, Siying Zhou, Yifei Liu, Weiming Lu, Xiaozhong Liu, Yating Zhang, Changlong Sun, Fei Wu, Kun Kuang. (2023)  
**Precedent-Enhanced Legal Judgment Prediction with LLM and Domain-Model Collaboration**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Legal  
[Paper Link](http://arxiv.org/abs/2310.09241v1)  

---


**ABSTRACT**  
Legal Judgment Prediction (LJP) has become an increasingly crucial task in Legal AI, i.e., predicting the judgment of the case in terms of case fact description. Precedents are the previous legal cases with similar facts, which are the basis for the judgment of the subsequent case in national legal systems. Thus, it is worthwhile to explore the utilization of precedents in the LJP. Recent advances in deep learning have enabled a variety of techniques to be used to solve the LJP task. These can be broken down into two categories: large language models (LLMs) and domain-specific models. LLMs are capable of interpreting and generating complex natural language, while domain models are efficient in learning task-specific information. In this paper, we propose the precedent-enhanced LJP framework (PLJP), a system that leverages the strength of both LLM and domain models in the context of precedents. Specifically, the domain models are designed to provide candidate labels and find the proper precedents efficiently, and the large models will make the final prediction with an in-context precedents comprehension. Experiments on the real-world dataset demonstrate the effectiveness of our PLJP. Moreover, our work shows a promising direction for LLM and domain-model collaboration that can be generalized to other vertical domains.

{{</citation>}}


### (31/114) BanglaNLP at BLP-2023 Task 2: Benchmarking different Transformer Models for Sentiment Analysis of Bangla Social Media Posts (Saumajit Saha et al., 2023)

{{<citation>}}

Saumajit Saha, Albert Nanda. (2023)  
**BanglaNLP at BLP-2023 Task 2: Benchmarking different Transformer Models for Sentiment Analysis of Bangla Social Media Posts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Sentiment Analysis, Social Media, Transformer  
[Paper Link](http://arxiv.org/abs/2310.09238v1)  

---


**ABSTRACT**  
Bangla is the 7th most widely spoken language globally, with a staggering 234 million native speakers primarily hailing from India and Bangladesh. This morphologically rich language boasts a rich literary tradition, encompassing diverse dialects and language-specific challenges. Despite its linguistic richness and history, Bangla remains categorized as a low-resource language within the natural language processing (NLP) and speech community. This paper presents our submission to Task 2 (Sentiment Analysis of Bangla Social Media Posts) of the BLP Workshop. We experiment with various Transformer-based architectures to solve this task. Our quantitative results show that transfer learning really helps in better learning of the models in this low-resource language scenario. This becomes evident when we further finetune a model which has already been finetuned on twitter data for sentiment analysis task and that finetuned model performs the best among all other models. We also perform a detailed error analysis where we find some instances where ground truth labels need to be relooked at. We obtain a micro-F1 of 67.02\% on the test set and our performance in this shared task is ranked at 21 in the leaderboard.

{{</citation>}}


### (32/114) Automated Claim Matching with Large Language Models: Empowering Fact-Checkers in the Fight Against Misinformation (Eun Cheol Choi et al., 2023)

{{<citation>}}

Eun Cheol Choi, Emilio Ferrara. (2023)  
**Automated Claim Matching with Large Language Models: Empowering Fact-Checkers in the Fight Against Misinformation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-HC, cs.CL  
Keywords: Augmentation, GPT, GPT-4, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2310.09223v1)  

---


**ABSTRACT**  
In today's digital era, the rapid spread of misinformation poses threats to public well-being and societal trust. As online misinformation proliferates, manual verification by fact checkers becomes increasingly challenging. We introduce FACT-GPT (Fact-checking Augmentation with Claim matching Task-oriented Generative Pre-trained Transformer), a framework designed to automate the claim matching phase of fact-checking using Large Language Models (LLMs). This framework identifies new social media content that either supports or contradicts claims previously debunked by fact-checkers. Our approach employs GPT-4 to generate a labeled dataset consisting of simulated social media posts. This data set serves as a training ground for fine-tuning more specialized LLMs. We evaluated FACT-GPT on an extensive dataset of social media content related to public health. The results indicate that our fine-tuned LLMs rival the performance of larger pre-trained LLMs in claim matching tasks, aligning closely with human annotations. This study achieves three key milestones: it provides an automated framework for enhanced fact-checking; demonstrates the potential of LLMs to complement human expertise; offers public resources, including datasets and models, to further research and applications in the fact-checking domain.

{{</citation>}}


### (33/114) 'Kelly is a Warm Person, Joseph is a Role Model': Gender Biases in LLM-Generated Reference Letters (Yixin Wan et al., 2023)

{{<citation>}}

Yixin Wan, George Pu, Jiao Sun, Aparna Garimella, Kai-Wei Chang, Nanyun Peng. (2023)  
**'Kelly is a Warm Person, Joseph is a Role Model': Gender Biases in LLM-Generated Reference Letters**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.09219v1)  

---


**ABSTRACT**  
As generative language models advance, users have started to utilize Large Language Models (LLMs) to assist in writing various types of content, including professional documents such as recommendation letters. Despite their convenience, these applications introduce unprecedented fairness concerns. As generated reference letters might be directly utilized by users in professional or academic scenarios, they have the potential to cause direct social harms, such as lowering success rates for female applicants. Therefore, it is imminent and necessary to comprehensively study fairness issues and associated harms in such real-world use cases for future mitigation and monitoring. In this paper, we critically examine gender bias in LLM-generated reference letters. Inspired by findings in social science, we design evaluation methods to manifest gender biases in LLM-generated letters through 2 dimensions: biases in language style and biases in lexical content. Furthermore, we investigate the extent of bias propagation by separately analyze bias amplification in model-hallucinated contents, which we define to be the hallucination bias of model-generated documents. Through benchmarking evaluation on 4 popular LLMs, including ChatGPT, Alpaca, Vicuna and StableLM, our study reveals significant gender biases in LLM-generated recommendation letters. Our findings further point towards the importance and imminence to recognize biases in LLM-generated professional documents.

{{</citation>}}


### (34/114) Explore-Instruct: Enhancing Domain-Specific Instruction Coverage through Active Exploration (Fanqi Wan et al., 2023)

{{<citation>}}

Fanqi Wan, Xinting Huang, Tao Yang, Xiaojun Quan, Wei Bi, Shuming Shi. (2023)  
**Explore-Instruct: Enhancing Domain-Specific Instruction Coverage through Active Exploration**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.09168v1)  

---


**ABSTRACT**  
Instruction-tuning can be substantially optimized through enhanced diversity, resulting in models capable of handling a broader spectrum of tasks. However, existing data employed for such tuning often exhibit an inadequate coverage of individual domains, limiting the scope for nuanced comprehension and interactions within these areas. To address this deficiency, we propose Explore-Instruct, a novel approach to enhance the data coverage to be used in domain-specific instruction-tuning through active exploration via Large Language Models (LLMs). Built upon representative domain use cases, Explore-Instruct explores a multitude of variations or possibilities by implementing a search algorithm to obtain diversified and domain-focused instruction-tuning data. Our data-centric analysis validates the effectiveness of this proposed approach in improving domain-specific instruction coverage. Moreover, our model's performance demonstrates considerable advancements over multiple baselines, including those utilizing domain-specific data enhancement. Our findings offer a promising opportunity to improve instruction coverage, especially in domain-specific contexts, thereby advancing the development of adaptable language models. Our code, model weights, and data are public at \url{https://github.com/fanqiwan/Explore-Instruct}.

{{</citation>}}


### (35/114) Developing a Natural Language Understanding Model to Characterize Cable News Bias (Seth P. Benson et al., 2023)

{{<citation>}}

Seth P. Benson, Iain J. Cruickshank. (2023)  
**Developing a Natural Language Understanding Model to Characterize Cable News Bias**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Named Entity Recognition, Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2310.09166v1)  

---


**ABSTRACT**  
Media bias has been extensively studied by both social and computational sciences. However, current work still has a large reliance on human input and subjective assessment to label biases. This is especially true for cable news research. To address these issues, we develop an unsupervised machine learning method to characterize the bias of cable news programs without any human input. This method relies on the analysis of what topics are mentioned through Named Entity Recognition and how those topics are discussed through Stance Analysis in order to cluster programs with similar biases together. Applying our method to 2020 cable news transcripts, we find that program clusters are consistent over time and roughly correspond to the cable news network of the program. This method reveals the potential for future tools to objectively assess media bias and characterize unfamiliar media environments.

{{</citation>}}


### (36/114) BibRank: Automatic Keyphrase Extraction Platform Using~Metadata (Abdelrhman Eldallal et al., 2023)

{{<citation>}}

Abdelrhman Eldallal, Eduard Barbu. (2023)  
**BibRank: Automatic Keyphrase Extraction Platform Using~Metadata**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Keyphrase Extraction  
[Paper Link](http://arxiv.org/abs/2310.09151v1)  

---


**ABSTRACT**  
Automatic Keyphrase Extraction involves identifying essential phrases in a document. These keyphrases are crucial in various tasks such as document classification, clustering, recommendation, indexing, searching, summarization, and text simplification. This paper introduces a platform that integrates keyphrase datasets and facilitates the evaluation of keyphrase extraction algorithms. The platform includes BibRank, an automatic keyphrase extraction algorithm that leverages a rich dataset obtained by parsing bibliographic data in BibTeX format. BibRank combines innovative weighting techniques with positional, statistical, and word co-occurrence information to extract keyphrases from documents. The platform proves valuable for researchers and developers seeking to enhance their keyphrase extraction algorithms and advance the field of natural language processing.

{{</citation>}}


### (37/114) PuoBERTa: Training and evaluation of a curated language model for Setswana (Vukosi Marivate et al., 2023)

{{<citation>}}

Vukosi Marivate, Moseli Mots'Oehli, Valencia Wagner, Richard Lastrucci, Isheanesu Dzingirai. (2023)  
**PuoBERTa: Training and evaluation of a curated language model for Setswana**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NER, NLP  
[Paper Link](http://arxiv.org/abs/2310.09141v1)  

---


**ABSTRACT**  
Natural language processing (NLP) has made significant progress for well-resourced languages such as English but lagged behind for low-resource languages like Setswana. This paper addresses this gap by presenting PuoBERTa, a customised masked language model trained specifically for Setswana. We cover how we collected, curated, and prepared diverse monolingual texts to generate a high-quality corpus for PuoBERTa's training. Building upon previous efforts in creating monolingual resources for Setswana, we evaluated PuoBERTa across several NLP tasks, including part-of-speech (POS) tagging, named entity recognition (NER), and news categorisation. Additionally, we introduced a new Setswana news categorisation dataset and provided the initial benchmarks using PuoBERTa. Our work demonstrates the efficacy of PuoBERTa in fostering NLP capabilities for understudied languages like Setswana and paves the way for future research directions.

{{</citation>}}


### (38/114) A Frustratingly Easy Plug-and-Play Detection-and-Reasoning Module for Chinese Spelling Check (Haojing Huang et al., 2023)

{{<citation>}}

Haojing Huang, Jingheng Ye, Qingyu Zhou, Yinghui Li, Yangning Li, Feng Zhou, Hai-Tao Zheng. (2023)  
**A Frustratingly Easy Plug-and-Play Detection-and-Reasoning Module for Chinese Spelling Check**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.09119v1)  

---


**ABSTRACT**  
In recent years, Chinese Spelling Check (CSC) has been greatly improved by designing task-specific pre-training methods or introducing auxiliary tasks, which mostly solve this task in an end-to-end fashion. In this paper, we propose to decompose the CSC workflow into detection, reasoning, and searching subtasks so that the rich external knowledge about the Chinese language can be leveraged more directly and efficiently. Specifically, we design a plug-and-play detection-and-reasoning module that is compatible with existing SOTA non-autoregressive CSC models to further boost their performance. We find that the detection-and-reasoning module trained for one model can also benefit other models. We also study the primary interpretability provided by the task decomposition. Extensive experiments and detailed analyses demonstrate the effectiveness and competitiveness of the proposed module.

{{</citation>}}


### (39/114) GLoRE: Evaluating Logical Reasoning of Large Language Models (Hanmeng liu et al., 2023)

{{<citation>}}

Hanmeng liu, Zhiyang Teng, Ruoxi Ning, Jian Liu, Qiji Zhou, Yue Zhang. (2023)  
**GLoRE: Evaluating Logical Reasoning of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.09107v1)  

---


**ABSTRACT**  
Recently, large language models (LLMs), including notable models such as GPT-4 and burgeoning community models, have showcased significant general language understanding abilities. However, there has been a scarcity of attempts to assess the logical reasoning capacities of these LLMs, an essential facet of natural language understanding. To encourage further investigation in this area, we introduce GLoRE, a meticulously assembled General Logical Reasoning Evaluation benchmark comprised of 12 datasets that span three different types of tasks. Our experimental results show that compared to the performance of human and supervised fine-tuning, the logical reasoning capabilities of open LLM models necessitate additional improvement; ChatGPT and GPT-4 show a strong capability of logical reasoning, with GPT-4 surpassing ChatGPT by a large margin. We propose a self-consistency probing method to enhance the accuracy of ChatGPT and a fine-tuned method to boost the performance of an open LLM. We release the datasets and evaluation programs to facilitate future research.

{{</citation>}}


### (40/114) Qilin-Med: Multi-stage Knowledge Injection Advanced Medical Large Language Model (Qichen Ye et al., 2023)

{{<citation>}}

Qichen Ye, Junling Liu, Dading Chong, Peilin Zhou, Yining Hua, Andrew Liu. (2023)  
**Qilin-Med: Multi-stage Knowledge Injection Advanced Medical Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, Language Model  
[Paper Link](http://arxiv.org/abs/2310.09089v1)  

---


**ABSTRACT**  
Integrating large language models (LLMs) into healthcare presents potential but faces challenges. Directly pre-training LLMs for domains like medicine is resource-heavy and sometimes unfeasible. Sole reliance on Supervised Fine-tuning (SFT) can result in overconfident predictions and may not tap into domain specific insights. Addressing these challenges, we present a multi-stage training method combining Domain-specific Continued Pre-training (DCPT), SFT, and Direct Preference Optimization (DPO). A notable contribution of our study is the introduction of a 3Gb Chinese Medicine (ChiMed) dataset, encompassing medical question answering, plain texts, knowledge graphs, and dialogues, segmented into three training stages. The medical LLM trained with our pipeline, Qilin-Med, exhibits significant performance boosts. In the CPT and SFT phases, it achieves 38.4% and 40.0% accuracy on the CMExam, surpassing Baichuan-7B's 33.5%. In the DPO phase, on the Huatuo-26M test set, it scores 16.66 in BLEU-1 and 27.44 in ROUGE1, outperforming the SFT's 12.69 and 24.21. This highlights the strength of our training approach in refining LLMs for medical applications.

{{</citation>}}


### (41/114) KCTS: Knowledge-Constrained Tree Search Decoding with Token-Level Hallucination Detection (Sehyun Choi et al., 2023)

{{<citation>}}

Sehyun Choi, Tianqing Fang, Zhaowei Wang, Yangqiu Song. (2023)  
**KCTS: Knowledge-Constrained Tree Search Decoding with Token-Level Hallucination Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.09044v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable human-level natural language generation capabilities. However, their potential to generate misinformation, often called the hallucination problem, poses a significant risk to their deployment. A common approach to address this issue is to retrieve relevant knowledge and fine-tune the LLM with the knowledge in its input. Unfortunately, this method incurs high training costs and may cause catastrophic forgetting for multi-tasking models. To overcome these limitations, we propose a knowledge-constrained decoding method called KCTS (Knowledge-Constrained Tree Search), which guides a frozen LM to generate text aligned with the reference knowledge at each decoding step using a knowledge classifier score and MCTS (Monte-Carlo Tree Search). To adapt the sequence-level knowledge classifier to token-level guidance, we also propose a novel token-level hallucination detection method called RIPA (Reward Inflection Point Approximation). Our empirical results on knowledge-grounded dialogue and abstractive summarization demonstrate the strength of KCTS as a plug-and-play, model-agnostic decoding method that can effectively reduce hallucinations in natural language generation.

{{</citation>}}


### (42/114) Dont Add, dont Miss: Effective Content Preserving Generation from Pre-Selected Text Spans (Aviv Slobodkin et al., 2023)

{{<citation>}}

Aviv Slobodkin, Avi Caciularu, Eran Hirsch, Ido Dagan. (2023)  
**Dont Add, dont Miss: Effective Content Preserving Generation from Pre-Selected Text Spans**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.09017v1)  

---


**ABSTRACT**  
The recently introduced Controlled Text Reduction (CTR) task isolates the text generation step within typical summarization-style tasks. It does so by challenging models to generate coherent text conforming to pre-selected content within the input text ("highlights").   This framing enables increased modularity in summarization-like tasks, allowing to couple a single CTR model with various content-selection setups and modules.   However, there are currently no reliable CTR models, while the performance of the existing baseline for the task is mediocre, falling short of practical utility.   Here, we address this gap by introducing a high-quality, open-source CTR model that tackles two prior key limitations: inadequate enforcement of the content-preservation constraint, and suboptimal silver training data.   Addressing these, we amplify the content-preservation constraint in both training, via RL, and inference, via a controlled decoding strategy.   Further, we substantially improve the silver training data quality via GPT-4 distillation.   Overall, pairing the distilled dataset with the highlight-adherence strategies yields marked gains over the current baseline, of up to 30 ROUGE-L points, providing a reliable CTR model for downstream use.

{{</citation>}}


### (43/114) ChatKBQA: A Generate-then-Retrieve Framework for Knowledge Base Question Answering with Fine-tuned Large Language Models (Haoran Luo et al., 2023)

{{<citation>}}

Haoran Luo, Haihong E, Zichen Tang, Shiyao Peng, Yikai Guo, Wentai Zhang, Chenghao Ma, Guanting Dong, Meina Song, Wei Lin. (2023)  
**ChatKBQA: A Generate-then-Retrieve Framework for Knowledge Base Question Answering with Fine-tuned Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GLM, Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.08975v1)  

---


**ABSTRACT**  
Knowledge Base Question Answering (KBQA) aims to derive answers to natural language questions over large-scale knowledge bases (KBs), which are generally divided into two research components: knowledge retrieval and semantic parsing. However, three core challenges remain, including inefficient knowledge retrieval, retrieval errors adversely affecting semantic parsing, and the complexity of previous KBQA methods. In the era of large language models (LLMs), we introduce ChatKBQA, a novel generate-then-retrieve KBQA framework built on fine-tuning open-source LLMs such as Llama-2, ChatGLM2 and Baichuan2. ChatKBQA proposes generating the logical form with fine-tuned LLMs first, then retrieving and replacing entities and relations through an unsupervised retrieval method, which improves both generation and retrieval more straightforwardly. Experimental results reveal that ChatKBQA achieves new state-of-the-art performance on standard KBQA datasets, WebQSP, and ComplexWebQuestions (CWQ). This work also provides a new paradigm for combining LLMs with knowledge graphs (KGs) for interpretable and knowledge-required question answering. Our code is publicly available.

{{</citation>}}


### (44/114) Towards Example-Based NMT with Multi-Levenshtein Transformers (Maxime Bouthors et al., 2023)

{{<citation>}}

Maxime Bouthors, Josep Crego, François Yvon. (2023)  
**Towards Example-Based NMT with Multi-Levenshtein Transformers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Machine Translation, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.08967v1)  

---


**ABSTRACT**  
Retrieval-Augmented Machine Translation (RAMT) is attracting growing attention. This is because RAMT not only improves translation metrics, but is also assumed to implement some form of domain adaptation. In this contribution, we study another salient trait of RAMT, its ability to make translation decisions more transparent by allowing users to go back to examples that contributed to these decisions.   For this, we propose a novel architecture aiming to increase this transparency. This model adapts a retrieval-augmented version of the Levenshtein Transformer and makes it amenable to simultaneously edit multiple fuzzy matches found in memory. We discuss how to perform training and inference in this model, based on multi-way alignment algorithms and imitation learning. Our experiments show that editing several examples positively impacts translation scores, notably increasing the number of target spans that are copied from existing instances.

{{</citation>}}


### (45/114) xDial-Eval: A Multilingual Open-Domain Dialogue Evaluation Benchmark (Chen Zhang et al., 2023)

{{<citation>}}

Chen Zhang, Luis Fernando D'Haro, Chengguang Tang, Ke Shi, Guohua Tang, Haizhou Li. (2023)  
**xDial-Eval: A Multilingual Open-Domain Dialogue Evaluation Benchmark**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, BERT, ChatGPT, Dialog, Dialogue, GPT, Multilingual  
[Paper Link](http://arxiv.org/abs/2310.08958v1)  

---


**ABSTRACT**  
Recent advancements in reference-free learned metrics for open-domain dialogue evaluation have been driven by the progress in pre-trained language models and the availability of dialogue data with high-quality human annotations. However, current studies predominantly concentrate on English dialogues, and the generalization of these metrics to other languages has not been fully examined. This is largely due to the absence of a multilingual dialogue evaluation benchmark. To address the issue, we introduce xDial-Eval, built on top of open-source English dialogue evaluation datasets. xDial-Eval includes 12 turn-level and 6 dialogue-level English datasets, comprising 14930 annotated turns and 8691 annotated dialogues respectively. The English dialogue data are extended to nine other languages with commercial machine translation systems. On xDial-Eval, we conduct comprehensive analyses of previous BERT-based metrics and the recently-emerged large language models. Lastly, we establish strong self-supervised and multilingual baselines. In terms of average Pearson correlations over all datasets and languages, the best baseline outperforms OpenAI's ChatGPT by absolute improvements of 6.5% and 4.6% at the turn and dialogue levels respectively, albeit with much fewer parameters. The data and code are publicly available at https://github.com/e0397123/xDial-Eval.

{{</citation>}}


### (46/114) CAMELL: Confidence-based Acquisition Model for Efficient Self-supervised Active Learning with Label Validation (Carel van Niekerk et al., 2023)

{{<citation>}}

Carel van Niekerk, Christian Geishauser, Michael Heck, Shutong Feng, Hsien-chin Lin, Nurul Lubis, Benjamin Ruppik, Renato Vukovic, Milica Gašić. (2023)  
**CAMELL: Confidence-based Acquisition Model for Efficient Self-supervised Active Learning with Label Validation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2310.08944v1)  

---


**ABSTRACT**  
Supervised neural approaches are hindered by their dependence on large, meticulously annotated datasets, a requirement that is particularly cumbersome for sequential tasks. The quality of annotations tends to deteriorate with the transition from expert-based to crowd-sourced labelling. To address these challenges, we present \textbf{CAMELL} (Confidence-based Acquisition Model for Efficient self-supervised active Learning with Label validation), a pool-based active learning framework tailored for sequential multi-output problems. CAMELL possesses three core features: (1) it requires expert annotators to label only a fraction of a chosen sequence, (2) it facilitates self-supervision for the remainder of the sequence, and (3) it employs a label validation mechanism to prevent erroneous labels from contaminating the dataset and harming model performance. We evaluate CAMELL on sequential tasks, with a special emphasis on dialogue belief tracking, a task plagued by the constraints of limited and noisy datasets. Our experiments demonstrate that CAMELL outperforms the baselines in terms of efficiency. Furthermore, the data corrections suggested by our method contribute to an overall improvement in the quality of the resulting datasets.

{{</citation>}}


### (47/114) Multi-level Adaptive Contrastive Learning for Knowledge Internalization in Dialogue Generation (Chenxu Yang et al., 2023)

{{<citation>}}

Chenxu Yang, Zheng Lin, Lanrui Wang, Chong Tian, Liang Pang, Jiangnan Li, Yanan Cao, Weiping Wang. (2023)  
**Multi-level Adaptive Contrastive Learning for Knowledge Internalization in Dialogue Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Contrastive Learning, Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2310.08943v1)  

---


**ABSTRACT**  
Knowledge-grounded dialogue generation aims to mitigate the issue of text degeneration by incorporating external knowledge to supplement the context. However, the model often fails to internalize this information into responses in a human-like manner. Instead, it simply inserts segments of the provided knowledge into generic responses. As a result, the generated responses tend to be tedious, incoherent, and in lack of interactivity which means the degeneration problem is still unsolved. In this work, we first find that such copying-style degeneration is primarily due to the weak likelihood objective, which allows the model to "cheat" the objective by merely duplicating knowledge segments in a superficial pattern matching based on overlap. To overcome this challenge, we then propose a Multi-level Adaptive Contrastive Learning (MACL) framework that dynamically samples negative examples and subsequently penalizes degeneration behaviors at both the token-level and sequence-level. Extensive experiments on the WoW dataset demonstrate the effectiveness of our approach across various pre-trained models.

{{</citation>}}


### (48/114) Towards Informative Few-Shot Prompt with Maximum Information Gain for In-Context Learning (Hongfu Liu et al., 2023)

{{<citation>}}

Hongfu Liu, Ye Wang. (2023)  
**Towards Informative Few-Shot Prompt with Maximum Information Gain for In-Context Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2310.08923v1)  

---


**ABSTRACT**  
Large Language models (LLMs) possess the capability to engage In-context Learning (ICL) by leveraging a few demonstrations pertaining to a new downstream task as conditions. However, this particular learning paradigm suffers from high instability stemming from substantial variances induced by factors such as the input distribution of selected examples, their ordering, and prompt formats. In this work, we demonstrate that even when all these factors are held constant, the random selection of examples still results in high variance. Consequently, we aim to explore the informative ability of data examples by quantifying the Information Gain (IG) obtained in prediction after observing a given example candidate. Then we propose to sample those with maximum IG. Additionally, we identify the presence of template bias, which can lead to unfair evaluations of IG during the sampling process. To mitigate this bias, we introduce Calibration Before Sampling strategy. The experimental results illustrate that our proposed method can yield an average relative improvement of 14.3% across six classification tasks using three LLMs.

{{</citation>}}


### (49/114) Human-in-the-loop Machine Translation with Large Language Model (Xinyi Yang et al., 2023)

{{<citation>}}

Xinyi Yang, Runzhe Zhan, Derek F. Wong, Junchao Wu, Lidia S. Chao. (2023)  
**Human-in-the-loop Machine Translation with Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model, Machine Translation, NLP  
[Paper Link](http://arxiv.org/abs/2310.08908v1)  

---


**ABSTRACT**  
The large language model (LLM) has garnered significant attention due to its in-context learning mechanisms and emergent capabilities. The research community has conducted several pilot studies to apply LLMs to machine translation tasks and evaluate their performance from diverse perspectives. However, previous research has primarily focused on the LLM itself and has not explored human intervention in the inference process of LLM. The characteristics of LLM, such as in-context learning and prompt engineering, closely mirror human cognitive abilities in language tasks, offering an intuitive solution for human-in-the-loop generation. In this study, we propose a human-in-the-loop pipeline that guides LLMs to produce customized outputs with revision instructions. The pipeline initiates by prompting the LLM to produce a draft translation, followed by the utilization of automatic retrieval or human feedback as supervision signals to enhance the LLM's translation through in-context learning. The human-machine interactions generated in this pipeline are also stored in an external database to expand the in-context retrieval database, enabling us to leverage human supervision in an offline setting. We evaluate the proposed pipeline using GPT-3.5-turbo API on five domain-specific benchmarks for German-English translation. The results demonstrate the effectiveness of the pipeline in tailoring in-domain translations and improving translation performance compared to direct translation. Additionally, we discuss the results from the following perspectives: 1) the effectiveness of different in-context retrieval methods; 2) the construction of a retrieval database under low-resource scenarios; 3) the observed domains differences; 4) the quantitative analysis of linguistic statistics; and 5) the qualitative analysis of translation cases. The code and data are available at https://github.com/NLP2CT/HIL-MT/.

{{</citation>}}


### (50/114) SeqXGPT: Sentence-Level AI-Generated Text Detection (Pengyu Wang et al., 2023)

{{<citation>}}

Pengyu Wang, Linyang Li, Ke Ren, Botian Jiang, Dong Zhang, Xipeng Qiu. (2023)  
**SeqXGPT: Sentence-Level AI-Generated Text Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2310.08903v1)  

---


**ABSTRACT**  
Widely applied large language models (LLMs) can generate human-like content, raising concerns about the abuse of LLMs. Therefore, it is important to build strong AI-generated text (AIGT) detectors. Current works only consider document-level AIGT detection, therefore, in this paper, we first introduce a sentence-level detection challenge by synthesizing a dataset that contains documents that are polished with LLMs, that is, the documents contain sentences written by humans and sentences modified by LLMs. Then we propose \textbf{Seq}uence \textbf{X} (Check) \textbf{GPT}, a novel method that utilizes log probability lists from white-box LLMs as features for sentence-level AIGT detection. These features are composed like \textit{waves} in speech processing and cannot be studied by LLMs. Therefore, we build SeqXGPT based on convolution and self-attention networks. We test it in both sentence and document-level detection challenges. Experimental results show that previous methods struggle in solving sentence-level AIGT detection, while our method not only significantly surpasses baseline methods in both sentence and document-level detection challenges but also exhibits strong generalization capabilities.

{{</citation>}}


### (51/114) Exploration with Principles for Diverse AI Supervision (Hao Liu et al., 2023)

{{<citation>}}

Hao Liu, Matei Zaharia, Pieter Abbeel. (2023)  
**Exploration with Principles for Diverse AI Supervision**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2310.08899v1)  

---


**ABSTRACT**  
Training large transformers using next-token prediction has given rise to groundbreaking advancements in AI. While this generative AI approach has produced impressive results, it heavily leans on human supervision. Even state-of-the-art AI models like ChatGPT depend on fine-tuning through human demonstrations, demanding extensive human input and domain expertise. This strong reliance on human oversight poses a significant hurdle to the advancement of AI innovation. To address this limitation, we propose a novel paradigm termed Exploratory AI (EAI) aimed at autonomously generating high-quality training data. Drawing inspiration from unsupervised reinforcement learning (RL) pretraining, EAI achieves exploration within the natural language space. We accomplish this by harnessing large language models to assess the novelty of generated content. Our approach employs two key components: an actor that generates novel content following exploration principles and a critic that evaluates the generated content, offering critiques to guide the actor. Empirical evaluations demonstrate that EAI significantly boosts model performance on complex reasoning tasks, addressing the limitations of human-intensive supervision.

{{</citation>}}


### (52/114) PerturbScore: Connecting Discrete and Continuous Perturbations in NLP (Linyang Li et al., 2023)

{{<citation>}}

Linyang Li, Ke Ren, Yunfan Shao, Pengyu Wang, Xipeng Qiu. (2023)  
**PerturbScore: Connecting Discrete and Continuous Perturbations in NLP**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.08889v1)  

---


**ABSTRACT**  
With the rapid development of neural network applications in NLP, model robustness problem is gaining more attention. Different from computer vision, the discrete nature of texts makes it more challenging to explore robustness in NLP. Therefore, in this paper, we aim to connect discrete perturbations with continuous perturbations, therefore we can use such connections as a bridge to help understand discrete perturbations in NLP models. Specifically, we first explore how to connect and measure the correlation between discrete perturbations and continuous perturbations. Then we design a regression task as a PerturbScore to learn the correlation automatically. Through experimental results, we find that we can build a connection between discrete and continuous perturbations and use the proposed PerturbScore to learn such correlation, surpassing previous methods used in discrete perturbation measuring. Further, the proposed PerturbScore can be well generalized to different datasets, perturbation methods, indicating that we can use it as a powerful tool to study model robustness in NLP.

{{</citation>}}


### (53/114) InstructTODS: Large Language Models for End-to-End Task-Oriented Dialogue Systems (Willy Chung et al., 2023)

{{<citation>}}

Willy Chung, Samuel Cahyawijaya, Bryan Wilie, Holy Lovenia, Pascale Fung. (2023)  
**InstructTODS: Large Language Models for End-to-End Task-Oriented Dialogue Systems**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.08885v1)  

---


**ABSTRACT**  
Large language models (LLMs) have been used for diverse tasks in natural language processing (NLP), yet remain under-explored for task-oriented dialogue systems (TODS), especially for end-to-end TODS. We present InstructTODS, a novel off-the-shelf framework for zero-shot end-to-end task-oriented dialogue systems that can adapt to diverse domains without fine-tuning. By leveraging LLMs, InstructTODS generates a proxy belief state that seamlessly translates user intentions into dynamic queries for efficient interaction with any KB. Our extensive experiments demonstrate that InstructTODS achieves comparable performance to fully fine-tuned TODS in guiding dialogues to successful completion without prior knowledge or task-specific data. Furthermore, a rigorous human evaluation of end-to-end TODS shows that InstructTODS produces dialogue responses that notably outperform both the gold responses and the state-of-the-art TODS in terms of helpfulness, informativeness, and humanness. Moreover, the effectiveness of LLMs in TODS is further supported by our comprehensive evaluations on TODS subtasks: dialogue state tracking, intent classification, and response generation. Code and implementations could be found here https://github.com/WillyHC22/InstructTODS/

{{</citation>}}


### (54/114) Retrieval-Generation Alignment for End-to-End Task-Oriented Dialogue System (Weizhou Shen et al., 2023)

{{<citation>}}

Weizhou Shen, Yingqi Gao, Canbin Huang, Fanqi Wan, Xiaojun Quan, Wei Bi. (2023)  
**Retrieval-Generation Alignment for End-to-End Task-Oriented Dialogue System**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, Dialog, Dialogue, GPT, T5  
[Paper Link](http://arxiv.org/abs/2310.08877v1)  

---


**ABSTRACT**  
Developing an efficient retriever to retrieve knowledge from a large-scale knowledge base (KB) is critical for task-oriented dialogue systems to effectively handle localized and specialized tasks. However, widely used generative models such as T5 and ChatGPT often struggle to differentiate subtle differences among the retrieved KB records when generating responses, resulting in suboptimal quality of generated responses. In this paper, we propose the application of maximal marginal likelihood to train a perceptive retriever by utilizing signals from response generation for supervision. In addition, our approach goes beyond considering solely retrieved entities and incorporates various meta knowledge to guide the generator, thus improving the utilization of knowledge. We evaluate our approach on three task-oriented dialogue datasets using T5 and ChatGPT as the backbone models. The results demonstrate that when combined with meta knowledge, the response generator can effectively leverage high-quality knowledge records from the retriever and enhance the quality of generated responses. The codes and models of this paper are available at https://github.com/shenwzh3/MK-TOD.

{{</citation>}}


### (55/114) Guiding AMR Parsing with Reverse Graph Linearization (Bofei Gao et al., 2023)

{{<citation>}}

Bofei Gao, Liang Chen, Peiyi Wang, Zhifang Sui, Baobao Chang. (2023)  
**Guiding AMR Parsing with Reverse Graph Linearization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Abstract Meaning Representation  
[Paper Link](http://arxiv.org/abs/2310.08860v1)  

---


**ABSTRACT**  
Abstract Meaning Representation (AMR) parsing aims to extract an abstract semantic graph from a given sentence. The sequence-to-sequence approaches, which linearize the semantic graph into a sequence of nodes and edges and generate the linearized graph directly, have achieved good performance. However, we observed that these approaches suffer from structure loss accumulation during the decoding process, leading to a much lower F1-score for nodes and edges decoded later compared to those decoded earlier. To address this issue, we propose a novel Reverse Graph Linearization (RGL) enhanced framework. RGL defines both default and reverse linearization orders of an AMR graph, where most structures at the back part of the default order appear at the front part of the reversed order and vice versa. RGL incorporates the reversed linearization to the original AMR parser through a two-pass self-distillation mechanism, which guides the model when generating the default linearizations. Our analysis shows that our proposed method significantly mitigates the problem of structure loss accumulation, outperforming the previously best AMR parsing model by 0.8 and 0.5 Smatch scores on the AMR 2.0 and AMR 3.0 dataset, respectively. The code are available at https://github.com/pkunlp-icler/AMR_reverse_graph_linearization.

{{</citation>}}


### (56/114) Large Language Models as Source Planner for Personalized Knowledge-grounded Dialogue (Hongru Wang et al., 2023)

{{<citation>}}

Hongru Wang, Minda Hu, Yang Deng, Rui Wang, Fei Mi, Weichao Wang, Yasheng Wang, Wai-Chung Kwan, Irwin King, Kam-Fai Wong. (2023)  
**Large Language Models as Source Planner for Personalized Knowledge-grounded Dialogue**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08840v1)  

---


**ABSTRACT**  
Open-domain dialogue system usually requires different sources of knowledge to generate more informative and evidential responses. However, existing knowledge-grounded dialogue systems either focus on a single knowledge source or overlook the dependency between multiple sources of knowledge, which may result in generating inconsistent or even paradoxical responses. To incorporate multiple knowledge sources and dependencies between them, we propose SAFARI, a novel framework that leverages the exceptional capabilities of large language models (LLMs) in planning, understanding, and incorporating under both supervised and unsupervised settings. Specifically, SAFARI decouples the knowledge grounding into multiple sources and response generation, which allows easy extension to various knowledge sources including the possibility of not using any sources. To study the problem, we construct a personalized knowledge-grounded dialogue dataset \textit{\textbf{K}nowledge \textbf{B}ehind \textbf{P}ersona}~(\textbf{KBP}), which is the first to consider the dependency between persona and implicit knowledge. Experimental results on the KBP dataset demonstrate that the SAFARI framework can effectively produce persona-consistent and knowledge-enhanced responses.

{{</citation>}}


### (57/114) A Comparative Analysis of Task-Agnostic Distillation Methods for Compressing Transformer Language Models (Takuma Udagawa et al., 2023)

{{<citation>}}

Takuma Udagawa, Aashka Trivedi, Michele Merler, Bishwaranjan Bhattacharjee. (2023)  
**A Comparative Analysis of Task-Agnostic Distillation Methods for Compressing Transformer Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, Language Model, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2310.08797v1)  

---


**ABSTRACT**  
Large language models have become a vital component in modern NLP, achieving state of the art performance in a variety of tasks. However, they are often inefficient for real-world deployment due to their expensive inference costs. Knowledge distillation is a promising technique to improve their efficiency while retaining most of their effectiveness. In this paper, we reproduce, compare and analyze several representative methods for task-agnostic (general-purpose) distillation of Transformer language models. Our target of study includes Output Distribution (OD) transfer, Hidden State (HS) transfer with various layer mapping strategies, and Multi-Head Attention (MHA) transfer based on MiniLMv2. Through our extensive experiments, we study the effectiveness of each method for various student architectures in both monolingual (English) and multilingual settings. Overall, we show that MHA transfer based on MiniLMv2 is generally the best option for distillation and explain the potential reasons behind its success. Moreover, we show that HS transfer remains as a competitive baseline, especially under a sophisticated layer mapping strategy, while OD transfer consistently lags behind other approaches. Findings from this study helped us deploy efficient yet effective student models for latency-critical applications.

{{</citation>}}


### (58/114) End-to-end Story Plot Generator (Hanlin Zhu et al., 2023)

{{<citation>}}

Hanlin Zhu, Andrew Cohen, Danqing Wang, Kevin Yang, Xiaomeng Yang, Jiantao Jiao, Yuandong Tian. (2023)  
**End-to-end Story Plot Generator**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, LLaMA  
[Paper Link](http://arxiv.org/abs/2310.08796v1)  

---


**ABSTRACT**  
Story plots, while short, carry most of the essential information of a full story that may contain tens of thousands of words. We study the problem of automatic generation of story plots, which includes story premise, character descriptions, plot outlines, etc. To generate a single engaging plot, existing plot generators (e.g., DOC (Yang et al., 2022a)) require hundreds to thousands of calls to LLMs (e.g., OpenAI API) in the planning stage of the story plot, which is costly and takes at least several minutes. Moreover, the hard-wired nature of the method makes the pipeline non-differentiable, blocking fast specialization and personalization of the plot generator. In this paper, we propose three models, $\texttt{OpenPlot}$, $\texttt{E2EPlot}$ and $\texttt{RLPlot}$, to address these challenges. $\texttt{OpenPlot}$ replaces expensive OpenAI API calls with LLaMA2 (Touvron et al., 2023) calls via careful prompt designs, which leads to inexpensive generation of high-quality training datasets of story plots. We then train an end-to-end story plot generator, $\texttt{E2EPlot}$, by supervised fine-tuning (SFT) using approximately 13000 story plots generated by $\texttt{OpenPlot}$. $\texttt{E2EPlot}$ generates story plots of comparable quality to $\texttt{OpenPlot}$, and is > 10$\times$ faster (1k tokens in only 30 seconds on average). Finally, we obtain $\texttt{RLPlot}$ that is further fine-tuned with RLHF on several different reward models for different aspects of story quality, which yields 60.0$\%$ winning rate against $\texttt{E2EPlot}$ along the aspect of suspense and surprise.

{{</citation>}}


### (59/114) Mitigating Bias for Question Answering Models by Tracking Bias Influence (Mingyu Derek Ma et al., 2023)

{{<citation>}}

Mingyu Derek Ma, Jiun-Yu Kao, Arpit Gupta, Yu-Hsiang Lin, Wenbo Zhao, Tagyoung Chung, Wei Wang, Kai-Wei Chang, Nanyun Peng. (2023)  
**Mitigating Bias for Question Answering Models by Tracking Bias Influence**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: Bias, NLP, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.08795v1)  

---


**ABSTRACT**  
Models of various NLP tasks have been shown to exhibit stereotypes, and the bias in the question answering (QA) models is especially harmful as the output answers might be directly consumed by the end users. There have been datasets to evaluate bias in QA models, while bias mitigation technique for the QA models is still under-explored. In this work, we propose BMBI, an approach to mitigate the bias of multiple-choice QA models. Based on the intuition that a model would lean to be more biased if it learns from a biased example, we measure the bias level of a query instance by observing its influence on another instance. If the influenced instance is more biased, we derive that the query instance is biased. We then use the bias level detected as an optimization objective to form a multi-task learning setting in addition to the original QA task. We further introduce a new bias evaluation metric to quantify bias in a comprehensive and sensitive way. We show that our method could be applied to multiple QA formulations across multiple bias categories. It can significantly reduce the bias level in all 9 bias categories in the BBQ dataset while maintaining comparable QA accuracy.

{{</citation>}}


### (60/114) 'Im not Racist but...': Discovering Bias in the Internal Knowledge of Large Language Models (Abel Salinas et al., 2023)

{{<citation>}}

Abel Salinas, Louis Penafiel, Robert McCormack, Fred Morstatter. (2023)  
**'Im not Racist but...': Discovering Bias in the Internal Knowledge of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08780v1)  

---


**ABSTRACT**  
Large language models (LLMs) have garnered significant attention for their remarkable performance in a continuously expanding set of natural language processing tasks. However, these models have been shown to harbor inherent societal biases, or stereotypes, which can adversely affect their performance in their many downstream applications. In this paper, we introduce a novel, purely prompt-based approach to uncover hidden stereotypes within any arbitrary LLM. Our approach dynamically generates a knowledge representation of internal stereotypes, enabling the identification of biases encoded within the LLM's internal knowledge. By illuminating the biases present in LLMs and offering a systematic methodology for their analysis, our work contributes to advancing transparency and promoting fairness in natural language processing systems.

{{</citation>}}


## cs.LG (19)



### (61/114) Towards End-to-end 4-Bit Inference on Generative Large Language Models (Saleh Ashkboos et al., 2023)

{{<citation>}}

Saleh Ashkboos, Ilia Markov, Elias Frantar, Tingxuan Zhong, Xincheng Wang, Jie Ren, Torsten Hoefler, Dan Alistarh. (2023)  
**Towards End-to-end 4-Bit Inference on Generative Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.09259v1)  

---


**ABSTRACT**  
We show that the majority of the inference computations for large generative models such as LLaMA and OPT can be performed with both weights and activations being cast to 4 bits, in a way that leads to practical speedups while at the same time maintaining good accuracy. We achieve this via a hybrid quantization strategy called QUIK, which compresses most of the weights and activations to 4-bit, while keeping some outlier weights and activations in higher-precision. Crucially, our scheme is designed with computational efficiency in mind: we provide GPU kernels with highly-efficient layer-wise runtimes, which lead to practical end-to-end throughput improvements of up to 3.1x relative to FP16 execution. Code and models are provided at https://github.com/IST-DASLab/QUIK.

{{</citation>}}


### (62/114) It's an Alignment, Not a Trade-off: Revisiting Bias and Variance in Deep Models (Lin Chen et al., 2023)

{{<citation>}}

Lin Chen, Michal Lukasik, Wittawat Jitkrittum, Chong You, Sanjiv Kumar. (2023)  
**It's an Alignment, Not a Trade-off: Revisiting Bias and Variance in Deep Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.09250v1)  

---


**ABSTRACT**  
Classical wisdom in machine learning holds that the generalization error can be decomposed into bias and variance, and these two terms exhibit a \emph{trade-off}. However, in this paper, we show that for an ensemble of deep learning based classification models, bias and variance are \emph{aligned} at a sample level, where squared bias is approximately \emph{equal} to variance for correctly classified sample points. We present empirical evidence confirming this phenomenon in a variety of deep learning models and datasets. Moreover, we study this phenomenon from two theoretical perspectives: calibration and neural collapse. We first show theoretically that under the assumption that the models are well calibrated, we can observe the bias-variance alignment. Second, starting from the picture provided by the neural collapse theory, we show an approximate correlation between bias and variance.

{{</citation>}}


### (63/114) Graph Condensation via Eigenbasis Matching (Yang Liu et al., 2023)

{{<citation>}}

Yang Liu, Deyu Bo, Chuan Shi. (2023)  
**Graph Condensation via Eigenbasis Matching**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2310.09202v1)  

---


**ABSTRACT**  
The increasing amount of graph data places requirements on the efficiency and scalability of graph neural networks (GNNs), despite their effectiveness in various graph-related applications. Recently, the emerging graph condensation (GC) sheds light on reducing the computational cost of GNNs from a data perspective. It aims to replace the real large graph with a significantly smaller synthetic graph so that GNNs trained on both graphs exhibit comparable performance. However, our empirical investigation reveals that existing GC methods suffer from poor generalization, i.e., different GNNs trained on the same synthetic graph have obvious performance gaps. What factors hinder the generalization of GC and how can we mitigate it? To answer this question, we commence with a detailed analysis and observe that GNNs will inject spectrum bias into the synthetic graph, resulting in a distribution shift. To tackle this issue, we propose eigenbasis matching for spectrum-free graph condensation, named GCEM, which has two key steps: First, GCEM matches the eigenbasis of the real and synthetic graphs, rather than the graph structure, which eliminates the spectrum bias of GNNs. Subsequently, GCEM leverages the spectrum of the real graph and the synthetic eigenbasis to construct the synthetic graph, thereby preserving the essential structural information. We theoretically demonstrate that the synthetic graph generated by GCEM maintains the spectral similarity, i.e., total variation, of the real graph. Extensive experiments conducted on five graph datasets verify that GCEM not only achieves state-of-the-art performance over baselines but also significantly narrows the performance gaps between different GNNs.

{{</citation>}}


### (64/114) Goodhart's Law in Reinforcement Learning (Jacek Karwowski et al., 2023)

{{<citation>}}

Jacek Karwowski, Oliver Hayman, Xingjian Bai, Klaus Kiendlhofer, Charlie Griffin, Joar Skalse. (2023)  
**Goodhart's Law in Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.09144v1)  

---


**ABSTRACT**  
Implementing a reward function that perfectly captures a complex task in the real world is impractical. As a result, it is often appropriate to think of the reward function as a proxy for the true objective rather than as its definition. We study this phenomenon through the lens of Goodhart's law, which predicts that increasing optimisation of an imperfect proxy beyond some critical point decreases performance on the true objective. First, we propose a way to quantify the magnitude of this effect and show empirically that optimising an imperfect proxy reward often leads to the behaviour predicted by Goodhart's law for a wide range of environments and reward functions. We then provide a geometric explanation for why Goodhart's law occurs in Markov decision processes. We use these theoretical insights to propose an optimal early stopping method that provably avoids the aforementioned pitfall and derive theoretical regret bounds for this method. Moreover, we derive a training method that maximises worst-case reward, for the setting where there is uncertainty about the true reward function. Finally, we evaluate our early stopping method experimentally. Our results support a foundation for a theoretically-principled study of reinforcement learning under reward misspecification.

{{</citation>}}


### (65/114) DSG: An End-to-End Document Structure Generator (Johannes Rausch et al., 2023)

{{<citation>}}

Johannes Rausch, Gentiana Rashiti, Maxim Gusev, Ce Zhang, Stefan Feuerriegel. (2023)  
**DSG: An End-to-End Document Structure Generator**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2310.09118v1)  

---


**ABSTRACT**  
Information in industry, research, and the public sector is widely stored as rendered documents (e.g., PDF files, scans). Hence, to enable downstream tasks, systems are needed that map rendered documents onto a structured hierarchical format. However, existing systems for this task are limited by heuristics and are not end-to-end trainable. In this work, we introduce the Document Structure Generator (DSG), a novel system for document parsing that is fully end-to-end trainable. DSG combines a deep neural network for parsing (i) entities in documents (e.g., figures, text blocks, headers, etc.) and (ii) relations that capture the sequence and nested structure between entities. Unlike existing systems that rely on heuristics, our DSG is trained end-to-end, making it effective and flexible for real-world applications. We further contribute a new, large-scale dataset called E-Periodica comprising real-world magazines with complex document structures for evaluation. Our results demonstrate that our DSG outperforms commercial OCR tools and, on top of that, achieves state-of-the-art performance. To the best of our knowledge, our DSG system is the first end-to-end trainable system for hierarchical document parsing.

{{</citation>}}


### (66/114) Insightful analysis of historical sources at scales beyond human capabilities using unsupervised Machine Learning and XAI (Oliver Eberle et al., 2023)

{{<citation>}}

Oliver Eberle, Jochen Büttner, Hassan El-Hajj, Grégoire Montavon, Klaus-Robert Müller, Matteo Valleriani. (2023)  
**Insightful analysis of historical sources at scales beyond human capabilities using unsupervised Machine Learning and XAI**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-DL, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.09091v1)  

---


**ABSTRACT**  
Historical materials are abundant. Yet, piecing together how human knowledge has evolved and spread both diachronically and synchronically remains a challenge that can so far only be very selectively addressed. The vast volume of materials precludes comprehensive studies, given the restricted number of human specialists. However, as large amounts of historical materials are now available in digital form there is a promising opportunity for AI-assisted historical analysis. In this work, we take a pivotal step towards analyzing vast historical corpora by employing innovative machine learning (ML) techniques, enabling in-depth historical insights on a grand scale. Our study centers on the evolution of knowledge within the `Sacrobosco Collection' -- a digitized collection of 359 early modern printed editions of textbooks on astronomy used at European universities between 1472 and 1650 -- roughly 76,000 pages, many of which contain astronomic, computational tables. An ML based analysis of these tables helps to unveil important facets of the spatio-temporal evolution of knowledge and innovation in the field of mathematical astronomy in the period, as taught at European universities.

{{</citation>}}


### (67/114) Optimal Scheduling of Electric Vehicle Charging with Deep Reinforcement Learning considering End Users Flexibility (Christoforos Menos-Aikateriniadis et al., 2023)

{{<citation>}}

Christoforos Menos-Aikateriniadis, Stavros Sykiotis, Pavlos S. Georgilakis. (2023)  
**Optimal Scheduling of Electric Vehicle Charging with Deep Reinforcement Learning considering End Users Flexibility**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.09040v1)  

---


**ABSTRACT**  
The rapid growth of decentralized energy resources and especially Electric Vehicles (EV), that are expected to increase sharply over the next decade, will put further stress on existing power distribution networks, increasing the need for higher system reliability and flexibility. In an attempt to avoid unnecessary network investments and to increase the controllability over distribution networks, network operators develop demand response (DR) programs that incentivize end users to shift their consumption in return for financial or other benefits. Artificial intelligence (AI) methods are in the research forefront for residential load scheduling applications, mainly due to their high accuracy, high computational speed and lower dependence on the physical characteristics of the models under development. The aim of this work is to identify households' EV cost-reducing charging policy under a Time-of-Use tariff scheme, with the use of Deep Reinforcement Learning, and more specifically Deep Q-Networks (DQN). A novel end users flexibility potential reward is inferred from historical data analysis, where households with solar power generation have been used to train and test the designed algorithm. The suggested DQN EV charging policy can lead to more than 20% of savings in end users electricity bills.

{{</citation>}}


### (68/114) Subspace Adaptation Prior for Few-Shot Learning (Mike Huisman et al., 2023)

{{<citation>}}

Mike Huisman, Aske Plaat, Jan N. van Rijn. (2023)  
**Subspace Adaptation Prior for Few-Shot Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2310.09028v1)  

---


**ABSTRACT**  
Gradient-based meta-learning techniques aim to distill useful prior knowledge from a set of training tasks such that new tasks can be learned more efficiently with gradient descent. While these methods have achieved successes in various scenarios, they commonly adapt all parameters of trainable layers when learning new tasks. This neglects potentially more efficient learning strategies for a given task distribution and may be susceptible to overfitting, especially in few-shot learning where tasks must be learned from a limited number of examples. To address these issues, we propose Subspace Adaptation Prior (SAP), a novel gradient-based meta-learning algorithm that jointly learns good initialization parameters (prior knowledge) and layer-wise parameter subspaces in the form of operation subsets that should be adaptable. In this way, SAP can learn which operation subsets to adjust with gradient descent based on the underlying task distribution, simultaneously decreasing the risk of overfitting when learning new tasks. We demonstrate that this ability is helpful as SAP yields superior or competitive performance in few-shot image classification settings (gains between 0.1% and 3.9% in accuracy). Analysis of the learned subspaces demonstrates that low-dimensional operations often yield high activation strengths, indicating that they may be important for achieving good few-shot learning performance. For reproducibility purposes, we publish all our research code publicly.

{{</citation>}}


### (69/114) Federated Meta-Learning for Few-Shot Fault Diagnosis with Representation Encoding (Jixuan Cui et al., 2023)

{{<citation>}}

Jixuan Cui, Jun Li, Zhen Mei, Kang Wei, Sha Wei, Ming Ding, Wen Chen, Song Guo. (2023)  
**Federated Meta-Learning for Few-Shot Fault Diagnosis with Representation Encoding**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2310.09002v1)  

---


**ABSTRACT**  
Deep learning-based fault diagnosis (FD) approaches require a large amount of training data, which are difficult to obtain since they are located across different entities. Federated learning (FL) enables multiple clients to collaboratively train a shared model with data privacy guaranteed. However, the domain discrepancy and data scarcity problems among clients deteriorate the performance of the global FL model. To tackle these issues, we propose a novel framework called representation encoding-based federated meta-learning (REFML) for few-shot FD. First, a novel training strategy based on representation encoding and meta-learning is developed. It harnesses the inherent heterogeneity among training clients, effectively transforming it into an advantage for out-of-distribution generalization on unseen working conditions or equipment types. Additionally, an adaptive interpolation method that calculates the optimal combination of local and global models as the initialization of local training is proposed. This helps to further utilize local information to mitigate the negative effects of domain discrepancy. As a result, high diagnostic accuracy can be achieved on unseen working conditions or equipment types with limited training data. Compared with the state-of-the-art methods, such as FedProx, the proposed REFML framework achieves an increase in accuracy by 2.17%-6.50% when tested on unseen working conditions of the same equipment type and 13.44%-18.33% when tested on totally unseen equipment types, respectively.

{{</citation>}}


### (70/114) LLaMA Rider: Spurring Large Language Models to Explore the Open World (Yicheng Feng et al., 2023)

{{<citation>}}

Yicheng Feng, Yuxuan Wang, Jiazheng Liu, Sipeng Zheng, Zongqing Lu. (2023)  
**LLaMA Rider: Spurring Large Language Models to Explore the Open World**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08922v1)  

---


**ABSTRACT**  
Recently, various studies have leveraged Large Language Models (LLMs) to help decision-making and planning in environments, and try to align the LLMs' knowledge with the world conditions. Nonetheless, the capacity of LLMs to continuously acquire environmental knowledge and adapt in an open world remains uncertain. In this paper, we propose an approach to spur LLMs to explore the open world, gather experiences, and learn to improve their task-solving capabilities. In this approach, a multi-round feedback-revision mechanism is utilized to encourage LLMs to actively select appropriate revision actions guided by feedback information from the environment. This facilitates exploration and enhances the model's performance. Besides, we integrate sub-task relabeling to assist LLMs in maintaining consistency in sub-task planning and help the model learn the combinatorial nature between tasks, enabling it to complete a wider range of tasks through training based on the acquired exploration experiences. By evaluation in Minecraft, an open-ended sandbox world, we demonstrate that our approach LLaMA-Rider enhances the efficiency of the LLM in exploring the environment, and effectively improves the LLM's ability to accomplish more tasks through fine-tuning with merely 1.3k instances of collected data, showing minimal training costs compared to the baseline using reinforcement learning.

{{</citation>}}


### (71/114) Embarrassingly Simple Text Watermarks (Ryoma Sato et al., 2023)

{{<citation>}}

Ryoma Sato, Yuki Takezawa, Han Bao, Kenta Niwa, Makoto Yamada. (2023)  
**Embarrassingly Simple Text Watermarks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: BLEU, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08920v1)  

---


**ABSTRACT**  
We propose Easymark, a family of embarrassingly simple yet effective watermarks. Text watermarking is becoming increasingly important with the advent of Large Language Models (LLM). LLMs can generate texts that cannot be distinguished from human-written texts. This is a serious problem for the credibility of the text. Easymark is a simple yet effective solution to this problem. Easymark can inject a watermark without changing the meaning of the text at all while a validator can detect if a text was generated from a system that adopted Easymark or not with high credibility. Easymark is extremely easy to implement so that it only requires a few lines of code. Easymark does not require access to LLMs, so it can be implemented on the user-side when the LLM providers do not offer watermarked LLMs. In spite of its simplicity, it achieves higher detection accuracy and BLEU scores than the state-of-the-art text watermarking methods. We also prove the impossibility theorem of perfect watermarking, which is valuable in its own right. This theorem shows that no matter how sophisticated a watermark is, a malicious user could remove it from the text, which motivate us to use a simple watermark such as Easymark. We carry out experiments with LLM-generated texts and confirm that Easymark can be detected reliably without any degradation of BLEU and perplexity, and outperform state-of-the-art watermarks in terms of both quality and reliability.

{{</citation>}}


### (72/114) Relation-aware Ensemble Learning for Knowledge Graph Embedding (Ling Yue et al., 2023)

{{<citation>}}

Ling Yue, Yongqi Zhang, Quanming Yao, Yong Li, Xian Wu, Ziheng Zhang, Zhenxi Lin, Yefeng Zheng. (2023)  
**Relation-aware Ensemble Learning for Knowledge Graph Embedding**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2310.08917v1)  

---


**ABSTRACT**  
Knowledge graph (KG) embedding is a fundamental task in natural language processing, and various methods have been proposed to explore semantic patterns in distinctive ways. In this paper, we propose to learn an ensemble by leveraging existing methods in a relation-aware manner. However, exploring these semantics using relation-aware ensemble leads to a much larger search space than general ensemble methods. To address this issue, we propose a divide-search-combine algorithm RelEns-DSC that searches the relation-wise ensemble weights independently. This algorithm has the same computation cost as general ensemble methods but with much better performance. Experimental results on benchmark datasets demonstrate the effectiveness of the proposed method in efficiently searching relation-aware ensemble weights and achieving state-of-the-art embedding performance. The code is public at https://github.com/LARS-research/RelEns.

{{</citation>}}


### (73/114) Adaptivity and Modularity for Efficient Generalization Over Task Complexity (Samira Abnar et al., 2023)

{{<citation>}}

Samira Abnar, Omid Saremi, Laurent Dinh, Shantel Wilson, Miguel Angel Bautista, Chen Huang, Vimal Thilak, Etai Littwin, Jiatao Gu, Josh Susskind, Samy Bengio. (2023)  
**Adaptivity and Modularity for Efficient Generalization Over Task Complexity**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.08866v1)  

---


**ABSTRACT**  
Can transformers generalize efficiently on problems that require dealing with examples with different levels of difficulty? We introduce a new task tailored to assess generalization over different complexities and present results that indicate that standard transformers face challenges in solving these tasks. These tasks are variations of pointer value retrieval previously introduced by Zhang et al. (2021). We investigate how the use of a mechanism for adaptive and modular computation in transformers facilitates the learning of tasks that demand generalization over the number of sequential computation steps (i.e., the depth of the computation graph). Based on our observations, we propose a transformer-based architecture called Hyper-UT, which combines dynamic function generation from hyper networks with adaptive depth from Universal Transformers. This model demonstrates higher accuracy and a fairer allocation of computational resources when generalizing to higher numbers of computation steps. We conclude that mechanisms for adaptive depth and modularity complement each other in improving efficient generalization concerning example complexity. Additionally, to emphasize the broad applicability of our findings, we illustrate that in a standard image recognition task, Hyper- UT's performance matches that of a ViT model but with considerably reduced computational demands (achieving over 70\% average savings by effectively using fewer layers).

{{</citation>}}


### (74/114) In-Context Learning for Few-Shot Molecular Property Prediction (Christopher Fifty et al., 2023)

{{<citation>}}

Christopher Fifty, Jure Leskovec, Sebastian Thrun. (2023)  
**In-Context Learning for Few-Shot Molecular Property Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Few-Shot, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08863v1)  

---


**ABSTRACT**  
In-context learning has become an important approach for few-shot learning in Large Language Models because of its ability to rapidly adapt to new tasks without fine-tuning model parameters. However, it is restricted to applications in natural language and inapplicable to other domains. In this paper, we adapt the concepts underpinning in-context learning to develop a new algorithm for few-shot molecular property prediction. Our approach learns to predict molecular properties from a context of (molecule, property measurement) pairs and rapidly adapts to new properties without fine-tuning. On the FS-Mol and BACE molecular property prediction benchmarks, we find this method surpasses the performance of recent meta-learning algorithms at small support sizes and is competitive with the best methods at large support sizes.

{{</citation>}}


### (75/114) Overcoming Recency Bias of Normalization Statistics in Continual Learning: Balance and Adaptation (Yilin Lyu et al., 2023)

{{<citation>}}

Yilin Lyu, Liyuan Wang, Xingxing Zhang, Zicheng Sun, Hang Su, Jun Zhu, Liping Jing. (2023)  
**Overcoming Recency Bias of Normalization Statistics in Continual Learning: Balance and Adaptation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias, ImageNet  
[Paper Link](http://arxiv.org/abs/2310.08855v1)  

---


**ABSTRACT**  
Continual learning entails learning a sequence of tasks and balancing their knowledge appropriately. With limited access to old training samples, much of the current work in deep neural networks has focused on overcoming catastrophic forgetting of old tasks in gradient-based optimization. However, the normalization layers provide an exception, as they are updated interdependently by the gradient and statistics of currently observed training samples, which require specialized strategies to mitigate recency bias. In this work, we focus on the most popular Batch Normalization (BN) and provide an in-depth theoretical analysis of its sub-optimality in continual learning. Our analysis demonstrates the dilemma between balance and adaptation of BN statistics for incremental tasks, which potentially affects training stability and generalization. Targeting on these particular challenges, we propose Adaptive Balance of BN (AdaB$^2$N), which incorporates appropriately a Bayesian-based strategy to adapt task-wise contributions and a modified momentum to balance BN statistics, corresponding to the training and testing stages. By implementing BN in a continual learning fashion, our approach achieves significant performance gains across a wide range of benchmarks, particularly for the challenging yet realistic online scenarios (e.g., up to 7.68%, 6.86% and 4.26% on Split CIFAR-10, Split CIFAR-100 and Split Mini-ImageNet, respectively). Our code is available at https://github.com/lvyilin/AdaB2N.

{{</citation>}}


### (76/114) Semi-Supervised End-To-End Contrastive Learning For Time Series Classification (Huili Cai et al., 2023)

{{<citation>}}

Huili Cai, Xiang Zhang, Xiaofeng Liu. (2023)  
**Semi-Supervised End-To-End Contrastive Learning For Time Series Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning, Semi-Supervised, Time Series  
[Paper Link](http://arxiv.org/abs/2310.08848v1)  

---


**ABSTRACT**  
Time series classification is a critical task in various domains, such as finance, healthcare, and sensor data analysis. Unsupervised contrastive learning has garnered significant interest in learning effective representations from time series data with limited labels. The prevalent approach in existing contrastive learning methods consists of two separate stages: pre-training the encoder on unlabeled datasets and fine-tuning the well-trained model on a small-scale labeled dataset. However, such two-stage approaches suffer from several shortcomings, such as the inability of unsupervised pre-training contrastive loss to directly affect downstream fine-tuning classifiers, and the lack of exploiting the classification loss which is guided by valuable ground truth. In this paper, we propose an end-to-end model called SLOTS (Semi-supervised Learning fOr Time clasSification). SLOTS receives semi-labeled datasets, comprising a large number of unlabeled samples and a small proportion of labeled samples, and maps them to an embedding space through an encoder. We calculate not only the unsupervised contrastive loss but also measure the supervised contrastive loss on the samples with ground truth. The learned embeddings are fed into a classifier, and the classification loss is calculated using the available true labels. The unsupervised, supervised contrastive losses and classification loss are jointly used to optimize the encoder and classifier. We evaluate SLOTS by comparing it with ten state-of-the-art methods across five datasets. The results demonstrate that SLOTS is a simple yet effective framework. When compared to the two-stage framework, our end-to-end SLOTS utilizes the same input data, consumes a similar computational cost, but delivers significantly improved performance. We release code and datasets at https://anonymous.4open.science/r/SLOTS-242E.

{{</citation>}}


### (77/114) Distance-rank Aware Sequential Reward Learning for Inverse Reinforcement Learning with Sub-optimal Demonstrations (Lu Li et al., 2023)

{{<citation>}}

Lu Li, Yuxin Pan, Ruobing Chen, Jie Liu, Zilin Wang, Yu Liu, Zhiheng Li. (2023)  
**Distance-rank Aware Sequential Reward Learning for Inverse Reinforcement Learning with Sub-optimal Demonstrations**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2310.08823v1)  

---


**ABSTRACT**  
Inverse reinforcement learning (IRL) aims to explicitly infer an underlying reward function based on collected expert demonstrations. Considering that obtaining expert demonstrations can be costly, the focus of current IRL techniques is on learning a better-than-demonstrator policy using a reward function derived from sub-optimal demonstrations. However, existing IRL algorithms primarily tackle the challenge of trajectory ranking ambiguity when learning the reward function. They overlook the crucial role of considering the degree of difference between trajectories in terms of their returns, which is essential for further removing reward ambiguity. Additionally, it is important to note that the reward of a single transition is heavily influenced by the context information within the trajectory. To address these issues, we introduce the Distance-rank Aware Sequential Reward Learning (DRASRL) framework. Unlike existing approaches, DRASRL takes into account both the ranking of trajectories and the degrees of dissimilarity between them to collaboratively eliminate reward ambiguity when learning a sequence of contextually informed reward signals. Specifically, we leverage the distance between policies, from which the trajectories are generated, as a measure to quantify the degree of differences between traces. This distance-aware information is then used to infer embeddings in the representation space for reward learning, employing the contrastive learning technique. Meanwhile, we integrate the pairwise ranking loss function to incorporate ranking information into the latent features. Moreover, we resort to the Transformer architecture to capture the contextual dependencies within the trajectories in the latent space, leading to more accurate reward estimation. Through extensive experimentation, our DRASRL framework demonstrates significant performance improvements over previous SOTA methods.

{{</citation>}}


### (78/114) DDMT: Denoising Diffusion Mask Transformer Models for Multivariate Time Series Anomaly Detection (Chaocheng Yang et al., 2023)

{{<citation>}}

Chaocheng Yang, Tingyin Wang, Xuanhui Yan. (2023)  
**DDMT: Denoising Diffusion Mask Transformer Models for Multivariate Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2310.08800v1)  

---


**ABSTRACT**  
Anomaly detection in multivariate time series has emerged as a crucial challenge in time series research, with significant research implications in various fields such as fraud detection, fault diagnosis, and system state estimation. Reconstruction-based models have shown promising potential in recent years for detecting anomalies in time series data. However, due to the rapid increase in data scale and dimensionality, the issues of noise and Weak Identity Mapping (WIM) during time series reconstruction have become increasingly pronounced. To address this, we introduce a novel Adaptive Dynamic Neighbor Mask (ADNM) mechanism and integrate it with the Transformer and Denoising Diffusion Model, creating a new framework for multivariate time series anomaly detection, named Denoising Diffusion Mask Transformer (DDMT). The ADNM module is introduced to mitigate information leakage between input and output features during data reconstruction, thereby alleviating the problem of WIM during reconstruction. The Denoising Diffusion Transformer (DDT) employs the Transformer as an internal neural network structure for Denoising Diffusion Model. It learns the stepwise generation process of time series data to model the probability distribution of the data, capturing normal data patterns and progressively restoring time series data by removing noise, resulting in a clear recovery of anomalies. To the best of our knowledge, this is the first model that combines Denoising Diffusion Model and the Transformer for multivariate time series anomaly detection. Experimental evaluations were conducted on five publicly available multivariate time series anomaly detection datasets. The results demonstrate that the model effectively identifies anomalies in time series data, achieving state-of-the-art performance in anomaly detection.

{{</citation>}}


### (79/114) Selectivity Drives Productivity: Efficient Dataset Pruning for Enhanced Transfer Learning (Yihua Zhang et al., 2023)

{{<citation>}}

Yihua Zhang, Yimeng Zhang, Aochuan Chen, Jinghan Jia, Jiancheng Liu, Gaowen Liu, Mingyi Hong, Shiyu Chang, Sijia Liu. (2023)  
**Selectivity Drives Productivity: Efficient Dataset Pruning for Enhanced Transfer Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2310.08782v1)  

---


**ABSTRACT**  
Massive data is often considered essential for deep learning applications, but it also incurs significant computational and infrastructural costs. Therefore, dataset pruning (DP) has emerged as an effective way to improve data efficiency by identifying and removing redundant training samples without sacrificing performance. In this work, we aim to address the problem of DP for transfer learning, i.e., how to prune a source dataset for improved pretraining efficiency and lossless finetuning accuracy on downstream target tasks. To our best knowledge, the problem of DP for transfer learning remains open, as previous studies have primarily addressed DP and transfer learning as separate problems. By contrast, we establish a unified viewpoint to integrate DP with transfer learning and find that existing DP methods are not suitable for the transfer learning paradigm. We then propose two new DP methods, label mapping and feature mapping, for supervised and self-supervised pretraining settings respectively, by revisiting the DP problem through the lens of source-target domain mapping. Furthermore, we demonstrate the effectiveness of our approach on numerous transfer learning tasks. We show that source data classes can be pruned by up to 40% ~ 80% without sacrificing downstream performance, resulting in a significant 2 ~ 5 times speed-up during the pretraining stage. Besides, our proposal exhibits broad applicability and can improve other computationally intensive transfer learning techniques, such as adversarial pretraining. Codes are available at https://github.com/OPTML-Group/DP4TL.

{{</citation>}}


## cs.AI (15)



### (80/114) Augmented Computational Design: Methodical Application of Artificial Intelligence in Generative Design (Pirouz Nourian et al., 2023)

{{<citation>}}

Pirouz Nourian, Shervin Azadi, Roy Uijtendaal, Nan Bai. (2023)  
**Augmented Computational Design: Methodical Application of Artificial Intelligence in Generative Design**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CE, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.09243v1)  

---


**ABSTRACT**  
This chapter presents methodological reflections on the necessity and utility of artificial intelligence in generative design. Specifically, the chapter discusses how generative design processes can be augmented by AI to deliver in terms of a few outcomes of interest or performance indicators while dealing with hundreds or thousands of small decisions. The core of the performance-based generative design paradigm is about making statistical or simulation-driven associations between these choices and consequences for mapping and navigating such a complex decision space. This chapter will discuss promising directions in Artificial Intelligence for augmenting decision-making processes in architectural design for mapping and navigating complex design spaces.

{{</citation>}}


### (81/114) Evaluating Machine Perception of Indigeneity: An Analysis of ChatGPT's Perceptions of Indigenous Roles in Diverse Scenarios (Cecilia Delgado Solorzano et al., 2023)

{{<citation>}}

Cecilia Delgado Solorzano, Carlos Toxtli Hernandez. (2023)  
**Evaluating Machine Perception of Indigeneity: An Analysis of ChatGPT's Perceptions of Indigenous Roles in Diverse Scenarios**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2310.09237v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), like ChatGPT, are fundamentally tools trained on vast data, reflecting diverse societal impressions. This paper aims to investigate LLMs' self-perceived bias concerning indigeneity when simulating scenarios of indigenous people performing various roles. Through generating and analyzing multiple scenarios, this work offers a unique perspective on how technology perceives and potentially amplifies societal biases related to indigeneity in social computing. The findings offer insights into the broader implications of indigeneity in critical computing.

{{</citation>}}


### (82/114) Multinational AGI Consortium (MAGIC): A Proposal for International Coordination on AI (Jason Hausenloy et al., 2023)

{{<citation>}}

Jason Hausenloy, Andrea Miotti, Claire Dennis. (2023)  
**Multinational AGI Consortium (MAGIC): A Proposal for International Coordination on AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.09217v1)  

---


**ABSTRACT**  
This paper proposes a Multinational Artificial General Intelligence Consortium (MAGIC) to mitigate existential risks from advanced artificial intelligence (AI). MAGIC would be the only institution in the world permitted to develop advanced AI, enforced through a global moratorium by its signatory members on all other advanced AI development. MAGIC would be exclusive, safety-focused, highly secure, and collectively supported by member states, with benefits distributed equitably among signatories. MAGIC would allow narrow AI models to flourish while significantly reducing the possibility of misaligned, rogue, breakout, or runaway outcomes of general-purpose systems. We do not address the political feasibility of implementing a moratorium or address the specific legislative strategies and rules needed to enforce a ban on high-capacity AGI training runs. Instead, we propose one positive vision of the future, where MAGIC, as a global governance regime, can lay the groundwork for long-term, safe regulation of advanced AI.

{{</citation>}}


### (83/114) Learning To Teach Large Language Models Logical Reasoning (Meiqi Chen et al., 2023)

{{<citation>}}

Meiqi Chen, Yubo Ma, Kaitao Song, Yixin Cao, Yan Zhang, Dongsheng Li. (2023)  
**Learning To Teach Large Language Models Logical Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.09158v1)  

---


**ABSTRACT**  
Large language models (LLMs) have gained enormous attention from both academia and industry, due to their exceptional ability in language generation and extremely powerful generalization. However, current LLMs still output unreliable content in practical reasoning tasks due to their inherent issues (e.g., hallucination). To better disentangle this problem, in this paper, we conduct an in-depth investigation to systematically explore the capability of LLMs in logical reasoning. More in detail, we first investigate the deficiency of LLMs in logical reasoning on different tasks, including event relation extraction and deductive reasoning. Our study demonstrates that LLMs are not good reasoners in solving tasks with rigorous reasoning and will produce counterfactual answers, which require us to iteratively refine. Therefore, we comprehensively explore different strategies to endow LLMs with logical reasoning ability, and thus enable them to generate more logically consistent answers across different scenarios. Based on our approach, we also contribute a synthesized dataset (LLM-LR) involving multi-hop reasoning for evaluation and pre-training. Extensive quantitative and qualitative analyses on different tasks also validate the effectiveness and necessity of teaching LLMs with logic and provide insights for solving practical tasks with LLMs in future work.

{{</citation>}}


### (84/114) Lincoln AI Computing Survey (LAICS) Update (Albert Reuther et al., 2023)

{{<citation>}}

Albert Reuther, Peter Michaleas, Michael Jones, Vijay Gadepally, Siddharth Samsi, Jeremy Kepner. (2023)  
**Lincoln AI Computing Survey (LAICS) Update**  

---
Primary Category: cs.AI  
Categories: C-1-4; C-4, cs-AI, cs-DC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.09145v1)  

---


**ABSTRACT**  
This paper is an update of the survey of AI accelerators and processors from past four years, which is now called the Lincoln AI Computing Survey - LAICS (pronounced "lace"). As in past years, this paper collects and summarizes the current commercial accelerators that have been publicly announced with peak performance and peak power consumption numbers. The performance and power values are plotted on a scatter graph, and a number of dimensions and observations from the trends on this plot are again discussed and analyzed. Market segments are highlighted on the scatter plot, and zoomed plots of each segment are also included. Finally, a brief description of each of the new accelerators that have been added in the survey this year is included.

{{</citation>}}


### (85/114) HierarchicalContrast: A Coarse-to-Fine Contrastive Learning Framework for Cross-Domain Zero-Shot Slot Filling (Junwen Zhang et al., 2023)

{{<citation>}}

Junwen Zhang, Yin Zhang. (2023)  
**HierarchicalContrast: A Coarse-to-Fine Contrastive Learning Framework for Cross-Domain Zero-Shot Slot Filling**  

---
Primary Category: cs.AI  
Categories: I-2-7, cs-AI, cs-CL, cs.AI  
Keywords: Contrastive Learning, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.09135v1)  

---


**ABSTRACT**  
In task-oriented dialogue scenarios, cross-domain zero-shot slot filling plays a vital role in leveraging source domain knowledge to learn a model with high generalization ability in unknown target domain where annotated data is unavailable. However, the existing state-of-the-art zero-shot slot filling methods have limited generalization ability in target domain, they only show effective knowledge transfer on seen slots and perform poorly on unseen slots. To alleviate this issue, we present a novel Hierarchical Contrastive Learning Framework (HiCL) for zero-shot slot filling. Specifically, we propose a coarse- to fine-grained contrastive learning based on Gaussian-distributed embedding to learn the generalized deep semantic relations between utterance-tokens, by optimizing inter- and intra-token distribution distance. This encourages HiCL to generalize to the slot types unseen at training phase. Furthermore, we present a new iterative label set semantics inference method to unbiasedly and separately evaluate the performance of unseen slot types which entangled with their counterparts (i.e., seen slot types) in the previous zero-shot slot filling evaluation methods. The extensive empirical experiments on four datasets demonstrate that the proposed method achieves comparable or even better performance than the current state-of-the-art zero-shot slot filling approaches.

{{</citation>}}


### (86/114) Split-and-Denoise: Protect large language model inference with local differential privacy (Peihua Mai et al., 2023)

{{<citation>}}

Peihua Mai, Ran Yan, Zhe Huang, Youjia Yang, Yan Pang. (2023)  
**Split-and-Denoise: Protect large language model inference with local differential privacy**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CR, cs.AI  
Keywords: Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2310.09130v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) shows powerful capability in natural language understanding by capturing hidden semantics in vector space. This process enriches the value of the text embeddings for various downstream tasks, thereby fostering the Embedding-as-a-Service (EaaS) business model. However, the direct transmission of text to servers poses a largely unaddressed risk of privacy leakage. To mitigate this issue, we introduce Split-N-Denoise (SnD), an innovative framework that split the model to execute the token embedding layer on the client side at minimal computational cost. This allows the client to introduce noise prior to transmitting the embeddings to the server, and subsequently receive and denoise the perturbed output embeddings for downstream tasks. Our approach is designed for the inference stage of LLMs and requires no modifications to the model parameters. Extensive experiments demonstrate SnD's effectiveness in optimizing the privacy-utility tradeoff across various LLM architectures and diverse downstream tasks. The results reveal a significant performance improvement under the same privacy budget compared to the baseline, offering clients a privacy-preserving solution for local privacy protection.

{{</citation>}}


### (87/114) SAI: Solving AI Tasks with Systematic Artificial Intelligence in Communication Network (Lei Yao et al., 2023)

{{<citation>}}

Lei Yao, Yong Zhang, Zilong Yan, Jialu Tian. (2023)  
**SAI: Solving AI Tasks with Systematic Artificial Intelligence in Communication Network**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.09049v1)  

---


**ABSTRACT**  
In the rapid development of artificial intelligence, solving complex AI tasks is a crucial technology in intelligent mobile networks. Despite the good performance of specialized AI models in intelligent mobile networks, they are unable to handle complicated AI tasks. To address this challenge, we propose Systematic Artificial Intelligence (SAI), which is a framework designed to solve AI tasks by leveraging Large Language Models (LLMs) and JSON-format intent-based input to connect self-designed model library and database. Specifically, we first design a multi-input component, which simultaneously integrates Large Language Models (LLMs) and JSON-format intent-based inputs to fulfill the diverse intent requirements of different users. In addition, we introduce a model library module based on model cards which employ model cards to pairwise match between different modules for model composition. Model cards contain the corresponding model's name and the required performance metrics. Then when receiving user network requirements, we execute each subtask for multiple selected model combinations and provide output based on the execution results and LLM feedback. By leveraging the language capabilities of LLMs and the abundant AI models in the model library, SAI can complete numerous complex AI tasks in the communication network, achieving impressive results in network optimization, resource allocation, and other challenging tasks.

{{</citation>}}


### (88/114) CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules (Hung Le et al., 2023)

{{<citation>}}

Hung Le, Hailin Chen, Amrita Saha, Akash Gokul, Doyen Sahoo, Shafiq Joty. (2023)  
**CodeChain: Towards Modular Code Generation Through Chain of Self-revisions with Representative Sub-modules**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-PL, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08992v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have already become quite proficient at solving simpler programming tasks like those in HumanEval or MBPP benchmarks. However, solving more complex and competitive programming tasks is still quite challenging for these models - possibly due to their tendency to generate solutions as monolithic code blocks instead of decomposing them into logical sub-tasks and sub-modules. On the other hand, experienced programmers instinctively write modularized code with abstraction for solving complex tasks, often reusing previously developed modules. To address this gap, we propose CodeChain, a novel framework for inference that elicits modularized code generation through a chain of self-revisions, each being guided by some representative sub-modules generated in previous iterations. Concretely, CodeChain first instructs the LLM to generate modularized codes through chain-of-thought prompting. Then it applies a chain of self-revisions by iterating the two steps: 1) extracting and clustering the generated sub-modules and selecting the cluster representatives as the more generic and re-usable implementations, and 2) augmenting the original chain-of-thought prompt with these selected module-implementations and instructing the LLM to re-generate new modularized solutions. We find that by naturally encouraging the LLM to reuse the previously developed and verified sub-modules, CodeChain can significantly boost both modularity as well as correctness of the generated solutions, achieving relative pass@1 improvements of 35% on APPS and 76% on CodeContests. It is shown to be effective on both OpenAI LLMs as well as open-sourced LLMs like WizardCoder. We also conduct comprehensive ablation studies with different methods of prompting, number of clusters, model sizes, program qualities, etc., to provide useful insights that underpin CodeChain's success.

{{</citation>}}


### (89/114) Multi-Purpose NLP Chatbot : Design, Methodology & Conclusion (Shivom Aggarwal et al., 2023)

{{<citation>}}

Shivom Aggarwal, Shourya Mehra, Pritha Mitra. (2023)  
**Multi-Purpose NLP Chatbot : Design, Methodology & Conclusion**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2310.08977v1)  

---


**ABSTRACT**  
With a major focus on its history, difficulties, and promise, this research paper provides a thorough analysis of the chatbot technology environment as it exists today. It provides a very flexible chatbot system that makes use of reinforcement learning strategies to improve user interactions and conversational experiences. Additionally, this system makes use of sentiment analysis and natural language processing to determine user moods. The chatbot is a valuable tool across many fields thanks to its amazing characteristics, which include voice-to-voice conversation, multilingual support [12], advising skills, offline functioning, and quick help features. The complexity of chatbot technology development is also explored in this study, along with the causes that have propelled these developments and their far-reaching effects on a range of sectors. According to the study, three crucial elements are crucial: 1) Even without explicit profile information, the chatbot system is built to adeptly understand unique consumer preferences and fluctuating satisfaction levels. With the use of this capacity, user interactions are made to meet their wants and preferences. 2) Using a complex method that interlaces Multiview voice chat information, the chatbot may precisely simulate users' actual experiences. This aids in developing more genuine and interesting discussions. 3) The study presents an original method for improving the black-box deep learning models' capacity for prediction. This improvement is made possible by introducing dynamic satisfaction measurements that are theory-driven, which leads to more precise forecasts of consumer reaction.

{{</citation>}}


### (90/114) Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs (Yuxin Zhang et al., 2023)

{{<citation>}}

Yuxin Zhang, Lirui Zhao, Mingbao Lin, Yunyun Sun, Yiwu Yao, Xingjia Han, Jared Tanner, Shiwei Liu, Rongrong Ji. (2023)  
**Dynamic Sparse No Training: Training-Free Fine-tuning for Sparse LLMs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2310.08915v1)  

---


**ABSTRACT**  
The ever-increasing large language models (LLMs), though opening a potential path for the upcoming artificial general intelligence, sadly drops a daunting obstacle on the way towards their on-device deployment. As one of the most well-established pre-LLMs approaches in reducing model complexity, network pruning appears to lag behind in the era of LLMs, due mostly to its costly fine-tuning (or re-training) necessity under the massive volumes of model parameter and training data. To close this industry-academia gap, we introduce Dynamic Sparse No Training (DSnoT), a training-free fine-tuning approach that slightly updates sparse LLMs without the expensive backpropagation and any weight updates. Inspired by the Dynamic Sparse Training, DSnoT minimizes the reconstruction error between the dense and sparse LLMs, in the fashion of performing iterative weight pruning-and-growing on top of sparse LLMs. To accomplish this purpose, DSnoT particularly takes into account the anticipated reduction in reconstruction error for pruning and growing, as well as the variance w.r.t. different input data for growing each weight. This practice can be executed efficiently in linear time since its obviates the need of backpropagation for fine-tuning LLMs. Extensive experiments on LLaMA-V1/V2, Vicuna, and OPT across various benchmarks demonstrate the effectiveness of DSnoT in enhancing the performance of sparse LLMs, especially at high sparsity levels. For instance, DSnoT is able to outperform the state-of-the-art Wanda by 26.79 perplexity at 70% sparsity with LLaMA-7B. Our paper offers fresh insights into how to fine-tune sparse LLMs in an efficient training-free manner and open new venues to scale the great potential of sparsity to LLMs. Codes are available at https://github.com/zxyxmu/DSnoT.

{{</citation>}}


### (91/114) Path To Gain Functional Transparency In Artificial Intelligence With Meaningful Explainability (Md. Tanzib Hosain et al., 2023)

{{<citation>}}

Md. Tanzib Hosain, Mehedi Hasan Anik, Sadman Rafi, Rana Tabassum, Khaleque Insia, Md. Mehrab Siddiky. (2023)  
**Path To Gain Functional Transparency In Artificial Intelligence With Meaningful Explainability**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08849v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) is rapidly integrating into various aspects of our daily lives, influencing decision-making processes in areas such as targeted advertising and matchmaking algorithms. As AI systems become increasingly sophisticated, ensuring their transparency and explainability becomes crucial. Functional transparency is a fundamental aspect of algorithmic decision-making systems, allowing stakeholders to comprehend the inner workings of these systems and enabling them to evaluate their fairness and accuracy. However, achieving functional transparency poses significant challenges that need to be addressed. In this paper, we propose a design for user-centered compliant-by-design transparency in transparent systems. We emphasize that the development of transparent and explainable AI systems is a complex and multidisciplinary endeavor, necessitating collaboration among researchers from diverse fields such as computer science, artificial intelligence, ethics, law, and social science. By providing a comprehensive understanding of the challenges associated with transparency in AI systems and proposing a user-centered design framework, we aim to facilitate the development of AI systems that are accountable, trustworthy, and aligned with societal values.

{{</citation>}}


### (92/114) A Case-Based Persistent Memory for a Large Language Model (Ian Watson, 2023)

{{<citation>}}

Ian Watson. (2023)  
**A Case-Based Persistent Memory for a Large Language Model**  

---
Primary Category: cs.AI  
Categories: I-2-0, cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08842v1)  

---


**ABSTRACT**  
Case-based reasoning (CBR) as a methodology for problem-solving can use any appropriate computational technique. This position paper argues that CBR researchers have somewhat overlooked recent developments in deep learning and large language models (LLMs). The underlying technical developments that have enabled the recent breakthroughs in AI have strong synergies with CBR and could be used to provide a persistent memory for LLMs to make progress towards Artificial General Intelligence.

{{</citation>}}


### (93/114) Leveraging Optimal Transport for Enhanced Offline Reinforcement Learning in Surgical Robotic Environments (Maryam Zare et al., 2023)

{{<citation>}}

Maryam Zare, Parham M. Kebria, Abbas Khosravi. (2023)  
**Leveraging Optimal Transport for Enhanced Offline Reinforcement Learning in Surgical Robotic Environments**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-RO, cs.AI, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.08841v1)  

---


**ABSTRACT**  
Most Reinforcement Learning (RL) methods are traditionally studied in an active learning setting, where agents directly interact with their environments, observe action outcomes, and learn through trial and error. However, allowing partially trained agents to interact with real physical systems poses significant challenges, including high costs, safety risks, and the need for constant supervision. Offline RL addresses these cost and safety concerns by leveraging existing datasets and reducing the need for resource-intensive real-time interactions. Nevertheless, a substantial challenge lies in the demand for these datasets to be meticulously annotated with rewards. In this paper, we introduce Optimal Transport Reward (OTR) labelling, an innovative algorithm designed to assign rewards to offline trajectories, using a small number of high-quality expert demonstrations. The core principle of OTR involves employing Optimal Transport (OT) to calculate an optimal alignment between an unlabeled trajectory from the dataset and an expert demonstration. This alignment yields a similarity measure that is effectively interpreted as a reward signal. An offline RL algorithm can then utilize these reward signals to learn a policy. This approach circumvents the need for handcrafted rewards, unlocking the potential to harness vast datasets for policy learning. Leveraging the SurRoL simulation platform tailored for surgical robot learning, we generate datasets and employ them to train policies using the OTR algorithm. By demonstrating the efficacy of OTR in a different domain, we emphasize its versatility and its potential to expedite RL deployment across a wide range of fields.

{{</citation>}}


### (94/114) Advancing Perception in Artificial Intelligence through Principles of Cognitive Science (Palaash Agrawal et al., 2023)

{{<citation>}}

Palaash Agrawal, Cheston Tan, Heena Rathore. (2023)  
**Advancing Perception in Artificial Intelligence through Principles of Cognitive Science**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08803v1)  

---


**ABSTRACT**  
Although artificial intelligence (AI) has achieved many feats at a rapid pace, there still exist open problems and fundamental shortcomings related to performance and resource efficiency. Since AI researchers benchmark a significant proportion of performance standards through human intelligence, cognitive sciences-inspired AI is a promising domain of research. Studying cognitive science can provide a fresh perspective to building fundamental blocks in AI research, which can lead to improved performance and efficiency. In this review paper, we focus on the cognitive functions of perception, which is the process of taking signals from one's surroundings as input, and processing them to understand the environment. Particularly, we study and compare its various processes through the lens of both cognitive sciences and AI. Through this study, we review all current major theories from various sub-disciplines of cognitive science (specifically neuroscience, psychology and linguistics), and draw parallels with theories and techniques from current practices in AI. We, hence, present a detailed collection of methods in AI for researchers to build AI systems inspired by cognitive science. Further, through the process of reviewing the state of cognitive-inspired AI, we point out many gaps in the current state of AI (with respect to the performance of the human brain), and hence present potential directions for researchers to develop better perception systems in AI.

{{</citation>}}


## cs.IR (1)



### (95/114) ClickPrompt: CTR Models are Strong Prompt Generators for Adapting Language Models to CTR Prediction (Jianghao Lin et al., 2023)

{{<citation>}}

Jianghao Lin, Bo Chen, Hangyu Wang, Yunjia Xi, Yanru Qu, Xinyi Dai, Kangning Zhang, Ruiming Tang, Yong Yu, Weinan Zhang. (2023)  
**ClickPrompt: CTR Models are Strong Prompt Generators for Adapting Language Models to CTR Prediction**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.09234v1)  

---


**ABSTRACT**  
Click-through rate (CTR) prediction has become increasingly indispensable for various Internet applications. Traditional CTR models convert the multi-field categorical data into ID features via one-hot encoding, and extract the collaborative signals among features. Such a paradigm suffers from the problem of semantic information loss. Another line of research explores the potential of pretrained language models (PLMs) for CTR prediction by converting input data into textual sentences through hard prompt templates. Although semantic signals are preserved, they generally fail to capture the collaborative information (e.g., feature interactions, pure ID features), not to mention the unacceptable inference overhead brought by the huge model size. In this paper, we aim to model both the semantic knowledge and collaborative knowledge for accurate CTR estimation, and meanwhile address the inference inefficiency issue. To benefit from both worlds and close their gaps, we propose a novel model-agnostic framework (i.e., ClickPrompt), where we incorporate CTR models to generate interaction-aware soft prompts for PLMs. We design a prompt-augmented masked language modeling (PA-MLM) pretraining task, where PLM has to recover the masked tokens based on the language context, as well as the soft prompts generated by CTR model. The collaborative and semantic knowledge from ID and textual features would be explicitly aligned and interacted via the prompt interface. Then, we can either tune the CTR model with PLM for superior performance, or solely tune the CTR model without PLM for inference efficiency. Experiments on four real-world datasets validate the effectiveness of ClickPrompt compared with existing baselines.

{{</citation>}}


## cs.MM (1)



### (96/114) Exploring Sparse Spatial Relation in Graph Inference for Text-Based VQA (Sheng Zhou et al., 2023)

{{<citation>}}

Sheng Zhou, Dan Guo, Jia Li, Xun Yang, Meng Wang. (2023)  
**Exploring Sparse Spatial Relation in Graph Inference for Text-Based VQA**  

---
Primary Category: cs.MM  
Categories: cs-CV, cs-MM, cs.MM  
Keywords: OCR, QA  
[Paper Link](http://arxiv.org/abs/2310.09147v1)  

---


**ABSTRACT**  
Text-based visual question answering (TextVQA) faces the significant challenge of avoiding redundant relational inference. To be specific, a large number of detected objects and optical character recognition (OCR) tokens result in rich visual relationships. Existing works take all visual relationships into account for answer prediction. However, there are three observations: (1) a single subject in the images can be easily detected as multiple objects with distinct bounding boxes (considered repetitive objects). The associations between these repetitive objects are superfluous for answer reasoning; (2) two spatially distant OCR tokens detected in the image frequently have weak semantic dependencies for answer reasoning; and (3) the co-existence of nearby objects and tokens may be indicative of important visual cues for predicting answers. Rather than utilizing all of them for answer prediction, we make an effort to identify the most important connections or eliminate redundant ones. We propose a sparse spatial graph network (SSGN) that introduces a spatially aware relation pruning technique to this task. As spatial factors for relation measurement, we employ spatial distance, geometric dimension, overlap area, and DIoU for spatially aware pruning. We consider three visual relationships for graph learning: object-object, OCR-OCR tokens, and object-OCR token relationships. SSGN is a progressive graph learning architecture that verifies the pivotal relations in the correlated object-token sparse graph, and then in the respective object-based sparse graph and token-based sparse graph. Experiment results on TextVQA and ST-VQA datasets demonstrate that SSGN achieves promising performances. And some visualization results further demonstrate the interpretability of our method.

{{</citation>}}


## cs.GT (1)



### (97/114) The Consensus Game: Language Model Generation via Equilibrium Search (Athul Paul Jacob et al., 2023)

{{<citation>}}

Athul Paul Jacob, Yikang Shen, Gabriele Farina, Jacob Andreas. (2023)  
**The Consensus Game: Language Model Generation via Equilibrium Search**  

---
Primary Category: cs.GT  
Categories: cs-AI, cs-CL, cs-GT, cs-LG, cs.GT  
Keywords: LLaMA, Language Model, NER, PaLM  
[Paper Link](http://arxiv.org/abs/2310.09139v1)  

---


**ABSTRACT**  
When applied to question answering and other text generation tasks, language models (LMs) may be queried generatively (by sampling answers from their output distribution) or discriminatively (by using them to score or rank a set of candidate outputs). These procedures sometimes yield very different predictions. How do we reconcile mutually incompatible scoring procedures to obtain coherent LM predictions? We introduce a new, a training-free, game-theoretic procedure for language model decoding. Our approach casts language model decoding as a regularized imperfect-information sequential signaling game - which we term the CONSENSUS GAME - in which a GENERATOR seeks to communicate an abstract correctness parameter using natural language sentences to a DISCRIMINATOR. We develop computational procedures for finding approximate equilibria of this game, resulting in a decoding algorithm we call EQUILIBRIUM-RANKING. Applied to a large number of tasks (including reading comprehension, commonsense reasoning, mathematical problem-solving, and dialog), EQUILIBRIUM-RANKING consistently, and sometimes substantially, improves performance over existing LM decoding procedures - on multiple benchmarks, we observe that applying EQUILIBRIUM-RANKING to LLaMA-7B outperforms the much larger LLaMA-65B and PaLM-540B models. These results highlight the promise of game-theoretic tools for addressing fundamental challenges of truthfulness and consistency in LMs.

{{</citation>}}


## stat.ML (1)



### (98/114) Automatic Music Playlist Generation via Simulation-based Reinforcement Learning (Federico Tomasi et al., 2023)

{{<citation>}}

Federico Tomasi, Joseph Cauteruccio, Surya Kanoria, Kamil Ciosek, Matteo Rinaldi, Zhenwen Dai. (2023)  
**Automatic Music Playlist Generation via Simulation-based Reinforcement Learning**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.09123v1)  

---


**ABSTRACT**  
Personalization of playlists is a common feature in music streaming services, but conventional techniques, such as collaborative filtering, rely on explicit assumptions regarding content quality to learn how to make recommendations. Such assumptions often result in misalignment between offline model objectives and online user satisfaction metrics. In this paper, we present a reinforcement learning framework that solves for such limitations by directly optimizing for user satisfaction metrics via the use of a simulated playlist-generation environment. Using this simulator we develop and train a modified Deep Q-Network, the action head DQN (AH-DQN), in a manner that addresses the challenges imposed by the large state and action space of our RL formulation. The resulting policy is capable of making recommendations from large and dynamic sets of candidate items with the expectation of maximizing consumption metrics. We analyze and evaluate agents offline via simulations that use environment models trained on both public and proprietary streaming datasets. We show how these agents lead to better user-satisfaction metrics compared to baseline methods during online A/B tests. Finally, we demonstrate that performance assessments produced from our simulator are strongly correlated with observed online metric results.

{{</citation>}}


## eess.IV (2)



### (99/114) Faster 3D cardiac CT segmentation with Vision Transformers (Lee Jollans et al., 2023)

{{<citation>}}

Lee Jollans, Mariana Bustamante, Lilian Henriksson, Anders Persson, Tino Ebbers. (2023)  
**Faster 3D cardiac CT segmentation with Vision Transformers**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.09099v1)  

---


**ABSTRACT**  
Accurate segmentation of the heart is essential for personalized blood flow simulations and surgical intervention planning. A recent advancement in image recognition is the Vision Transformer (ViT), which expands the field of view to encompass a greater portion of the global image context. We adapted ViT for three-dimensional volume inputs. Cardiac computed tomography (CT) volumes from 39 patients, featuring up to 20 timepoints representing the complete cardiac cycle, were utilized. Our network incorporates a modified ResNet50 block as well as a ViT block and employs cascade upsampling with skip connections. Despite its increased model complexity, our hybrid Transformer-Residual U-Net framework, termed TRUNet, converges in significantly less time than residual U-Net while providing comparable or superior segmentations of the left ventricle, left atrium, left atrial appendage, ascending aorta, and pulmonary veins. TRUNet offers more precise vessel boundary segmentation and better captures the heart's overall anatomical structure compared to residual U-Net, as confirmed by the absence of extraneous clusters of missegmented voxels. In terms of both performance and training speed, TRUNet exceeded U-Net, a commonly used segmentation architecture, making it a promising tool for 3D semantic segmentation tasks in medical imaging. The code for TRUNet is available at github.com/ljollans/TRUNet.

{{</citation>}}


### (100/114) Two-Stage Deep Learning Framework for Quality Assessment of Left Atrial Late Gadolinium Enhanced MRI Images (K M Arefeen Sultan et al., 2023)

{{<citation>}}

K M Arefeen Sultan, Benjamin Orkild, Alan Morris, Eugene Kholmovski, Erik Bieging, Eugene Kwan, Ravi Ranjan, Ed DiBella, Shireen Elhabian. (2023)  
**Two-Stage Deep Learning Framework for Quality Assessment of Left Atrial Late Gadolinium Enhanced MRI Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.08805v1)  

---


**ABSTRACT**  
Accurate assessment of left atrial fibrosis in patients with atrial fibrillation relies on high-quality 3D late gadolinium enhancement (LGE) MRI images. However, obtaining such images is challenging due to patient motion, changing breathing patterns, or sub-optimal choice of pulse sequence parameters. Automated assessment of LGE-MRI image diagnostic quality is clinically significant as it would enhance diagnostic accuracy, improve efficiency, ensure standardization, and contributes to better patient outcomes by providing reliable and high-quality LGE-MRI scans for fibrosis quantification and treatment planning. To address this, we propose a two-stage deep-learning approach for automated LGE-MRI image diagnostic quality assessment. The method includes a left atrium detector to focus on relevant regions and a deep network to evaluate diagnostic quality. We explore two training strategies, multi-task learning, and pretraining using contrastive learning, to overcome limited annotated data in medical imaging. Contrastive Learning result shows about $4\%$, and $9\%$ improvement in F1-Score and Specificity compared to Multi-Task learning when there's limited data.

{{</citation>}}


## cs.NI (2)



### (101/114) DNFS-VNE: Deep Neuro-Fuzzy System-Driven Virtual Network Embedding Algorithm (Ailing Xiao et al., 2023)

{{<citation>}}

Ailing Xiao, Ning Chen, Sheng Wu, Shigen Shen, Weiping Ding, Peiying Zhang. (2023)  
**DNFS-VNE: Deep Neuro-Fuzzy System-Driven Virtual Network Embedding Algorithm**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2310.09078v1)  

---


**ABSTRACT**  
By decoupling substrate resources, network virtualization (NV) is a promising solution for meeting diverse demands and ensuring differentiated quality of service (QoS). In particular, virtual network embedding (VNE) is a critical enabling technology that enhances the flexibility and scalability of network deployment by addressing the coupling of Internet processes and services. However, in the existing works, the black-box nature of deep neural networks (DNNs) limits the analysis, development, and improvement of systems. In recent times, interpretable deep learning (DL) represented by deep neuro-fuzzy systems (DNFS) combined with fuzzy inference has shown promising interpretability to further exploit the hidden value in the data. Motivated by this, we propose a DNFS-based VNE algorithm that aims to provide an interpretable NV scheme. Specifically, data-driven convolutional neural networks (CNNs) are used as fuzzy implication operators to compute the embedding probabilities of candidate substrate nodes through entailment operations. And, the identified fuzzy rule patterns are cached into the weights by forward computation and gradient back-propagation (BP). In addition, the fuzzy rule base is constructed based on Mamdani-type linguistic rules using linguistic labels. Finally, the effectiveness of evaluation indicators and fuzzy rules is verified by experiments.

{{</citation>}}


### (102/114) Generative AI-driven Semantic Communication Framework for NextG Wireless Network (Avi Deb Raha et al., 2023)

{{<citation>}}

Avi Deb Raha, Md. Shirajum Munir, Apurba Adhikary, Yu Qiao, Choong Seon Hong. (2023)  
**Generative AI-driven Semantic Communication Framework for NextG Wireless Network**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2310.09021v1)  

---


**ABSTRACT**  
This work designs a novel semantic communication (SemCom) framework for the next-generation wireless network to tackle the challenges of unnecessary transmission of vast amounts that cause high bandwidth consumption, more latency, and experience with bad quality of services (QoS). In particular, these challenges hinder applications like intelligent transportation systems (ITS), metaverse, mixed reality, and the Internet of Everything, where real-time and efficient data transmission is paramount. Therefore, to reduce communication overhead and maintain the QoS of emerging applications such as metaverse, ITS, and digital twin creation, this work proposes a novel semantic communication framework. First, an intelligent semantic transmitter is designed to capture the meaningful information (e.g., the rode-side image in ITS) by designing a domain-specific Mobile Segment Anything Model (MSAM)-based mechanism to reduce the potential communication traffic while QoS remains intact. Second, the concept of generative AI is introduced for building the SemCom to reconstruct and denoise the received semantic data frame at the receiver end. In particular, the Generative Adversarial Network (GAN) mechanism is designed to maintain a superior quality reconstruction under different signal-to-noise (SNR) channel conditions. Finally, we have tested and evaluated the proposed semantic communication (SemCom) framework with the real-world 6G scenario of ITS; in particular, the base station equipped with an RGB camera and a mmWave phased array. Experimental results demonstrate the efficacy of the proposed SemCom framework by achieving high-quality reconstruction across various SNR channel conditions, resulting in 93.45% data reduction in communication.

{{</citation>}}


## cs.SI (3)



### (103/114) Bots, Elections, and Controversies: Twitter Insights from Brazil's Polarised Elections (Diogo Pacheco, 2023)

{{<citation>}}

Diogo Pacheco. (2023)  
**Bots, Elections, and Controversies: Twitter Insights from Brazil's Polarised Elections**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2310.09051v1)  

---


**ABSTRACT**  
From 2018 to 2023, Brazil experienced its most fiercely contested elections in history, resulting in the election of far-right candidate Jair Bolsonaro followed by the left-wing, Lula da Silva. This period was marked by a murder attempt, a coup attempt, the pandemic, and a plethora of conspiracy theories and controversies. This paper analyses 437 million tweets originating from 13 million accounts associated with Brazilian politics during these two presidential election cycles. We focus on accounts' behavioural patterns. We noted a quasi-monotonic escalation in bot engagement, marked by notable surges both during COVID-19 and in the aftermath of the 2022 election. The data revealed a strong correlation between bot engagement and the number of replies during a single day ($r=0.66$, $p<0.01$). Furthermore, we identified a range of suspicious activities, including an unusually high number of accounts being created on the same day, with some days witnessing over 20,000 new accounts and super-prolific accounts generating close to 100,000 tweets. Lastly, we uncovered a sprawling network of accounts sharing Twitter handles, with a select few managing to utilise more than 100 distinct handles. This work can be instrumental in dismantling coordinated campaigns and offer valuable insights for the enhancement of bot detection algorithms.

{{</citation>}}


### (104/114) Community Membership Hiding as Counterfactual Graph Search via Deep Reinforcement Learning (Andrea Bernini et al., 2023)

{{<citation>}}

Andrea Bernini, Fabrizio Silvestri, Gabriele Tolomei. (2023)  
**Community Membership Hiding as Counterfactual Graph Search via Deep Reinforcement Learning**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-LG, cs-SI, cs.SI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.08909v1)  

---


**ABSTRACT**  
Community detection techniques are useful tools for social media platforms to discover tightly connected groups of users who share common interests. However, this functionality often comes at the expense of potentially exposing individuals to privacy breaches by inadvertently revealing their tastes or preferences. Therefore, some users may wish to safeguard their anonymity and opt out of community detection for various reasons, such as affiliation with political or religious organizations.   In this study, we address the challenge of community membership hiding, which involves strategically altering the structural properties of a network graph to prevent one or more nodes from being identified by a given community detection algorithm. We tackle this problem by formulating it as a constrained counterfactual graph objective, and we solve it via deep reinforcement learning. We validate the effectiveness of our method through two distinct tasks: node and community deception. Extensive experiments show that our approach overall outperforms existing baselines in both tasks.

{{</citation>}}


### (105/114) Impact of Stricter Content Moderation on Parler's Users' Discourse (Nihal Kumarswamy et al., 2023)

{{<citation>}}

Nihal Kumarswamy, Mohit Singhal, Shirin Nilizadeh. (2023)  
**Impact of Stricter Content Moderation on Parler's Users' Discourse**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-SI, cs.SI  
Keywords: Amazon, Google, QA  
[Paper Link](http://arxiv.org/abs/2310.08844v1)  

---


**ABSTRACT**  
Social media platforms employ various content moderation techniques to remove harmful, offensive, and hate speech content. The moderation level varies across platforms; even over time, it can evolve in a platform. For example, Parler, a fringe social media platform popular among conservative users, was known to have the least restrictive moderation policies, claiming to have open discussion spaces for their users. However, after linking the 2021 US Capitol Riots and the activity of some groups on Parler, such as QAnon and Proud Boys, on January 12, 2021, Parler was removed from the Apple and Google App Store and suspended from Amazon Cloud hosting service. Parler would have to modify their moderation policies to return to these online stores. After a month of downtime, Parler was back online with a new set of user guidelines, which reflected stricter content moderation, especially regarding the \emph{hate speech} policy.   In this paper, we studied the moderation changes performed by Parler and their effect on the toxicity of its content. We collected a large longitudinal Parler dataset with 17M parleys from 432K active users from February 2021 to January 2022, after its return to the Internet and App Store. To the best of our knowledge, this is the first study investigating the effectiveness of content moderation techniques using data-driven approaches and also the first Parler dataset after its brief hiatus. Our quasi-experimental time series analysis indicates that after the change in Parler's moderation, the severe forms of toxicity (above a threshold of 0.5) immediately decreased and sustained. In contrast, the trend did not change for less severe threats and insults (a threshold between 0.5 - 0.7). Finally, we found an increase in the factuality of the news sites being shared, as well as a decrease in the number of conspiracy or pseudoscience sources being shared.

{{</citation>}}


## cs.DC (1)



### (106/114) μ-DDRL: A QoS-Aware Distributed Deep Reinforcement Learning Technique for Service Offloading in Fog computing Environments (Mohammad Goudarzi et al., 2023)

{{<citation>}}

Mohammad Goudarzi, Maria A. Rodriguez, Majid Sarvi, Rajkumar Buyya. (2023)  
**μ-DDRL: A QoS-Aware Distributed Deep Reinforcement Learning Technique for Service Offloading in Fog computing Environments**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.09003v1)  

---


**ABSTRACT**  
Fog and Edge computing extend cloud services to the proximity of end users, allowing many Internet of Things (IoT) use cases, particularly latency-critical applications. Smart devices, such as traffic and surveillance cameras, often do not have sufficient resources to process computation-intensive and latency-critical services. Hence, the constituent parts of services can be offloaded to nearby Edge/Fog resources for processing and storage. However, making offloading decisions for complex services in highly stochastic and dynamic environments is an important, yet difficult task. Recently, Deep Reinforcement Learning (DRL) has been used in many complex service offloading problems; however, existing techniques are most suitable for centralized environments, and their convergence to the best-suitable solutions is slow. In addition, constituent parts of services often have predefined data dependencies and quality of service constraints, which further intensify the complexity of service offloading. To solve these issues, we propose a distributed DRL technique following the actor-critic architecture based on Asynchronous Proximal Policy Optimization (APPO) to achieve efficient and diverse distributed experience trajectory generation. Also, we employ PPO clipping and V-trace techniques for off-policy correction for faster convergence to the most suitable service offloading solutions. The results obtained demonstrate that our technique converges quickly, offers high scalability and adaptability, and outperforms its counterparts by improving the execution time of heterogeneous services.

{{</citation>}}


## cs.SD (3)



### (107/114) Transformer-based Autoencoder with ID Constraint for Unsupervised Anomalous Sound Detection (Jian Guan et al., 2023)

{{<citation>}}

Jian Guan, Youde Liu, Qiuqiang Kong, Feiyang Xiao, Qiaoxi Zhu, Jiantong Tian, Wenwu Wang. (2023)  
**Transformer-based Autoencoder with ID Constraint for Unsupervised Anomalous Sound Detection**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.08950v1)  

---


**ABSTRACT**  
Unsupervised anomalous sound detection (ASD) aims to detect unknown anomalous sounds of devices when only normal sound data is available. The autoencoder (AE) and self-supervised learning based methods are two mainstream methods. However, the AE-based methods could be limited as the feature learned from normal sounds can also fit with anomalous sounds, reducing the ability of the model in detecting anomalies from sound. The self-supervised methods are not always stable and perform differently, even for machines of the same type. In addition, the anomalous sound may be short-lived, making it even harder to distinguish from normal sound. This paper proposes an ID constrained Transformer-based autoencoder (IDC-TransAE) architecture with weighted anomaly score computation for unsupervised ASD. Machine ID is employed to constrain the latent space of the Transformer-based autoencoder (TransAE) by introducing a simple ID classifier to learn the difference in the distribution for the same machine type and enhance the ability of the model in distinguishing anomalous sound. Moreover, weighted anomaly score computation is introduced to highlight the anomaly scores of anomalous events that only appear for a short time. Experiments performed on DCASE 2020 Challenge Task2 development dataset demonstrate the effectiveness and superiority of our proposed method.

{{</citation>}}


### (108/114) Differential Evolution Algorithm based Hyper-Parameters Selection of Convolutional Neural Network for Speech Command Recognition (Sandipan Dhar et al., 2023)

{{<citation>}}

Sandipan Dhar, Anuvab Sen, Aritra Bandyopadhyay, Nanda Dulal Jana, Arjun Ghosh, Zahra Sarayloo. (2023)  
**Differential Evolution Algorithm based Hyper-Parameters Selection of Convolutional Neural Network for Speech Command Recognition**  

---
Primary Category: cs.SD  
Categories: cs-NE, cs-SD, cs.SD, eess-AS  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2310.08914v1)  

---


**ABSTRACT**  
Speech Command Recognition (SCR), which deals with identification of short uttered speech commands, is crucial for various applications, including IoT devices and assistive technology. Despite the promise shown by Convolutional Neural Networks (CNNs) in SCR tasks, their efficacy relies heavily on hyper-parameter selection, which is typically laborious and time-consuming when done manually. This paper introduces a hyper-parameter selection method for CNNs based on the Differential Evolution (DE) algorithm, aiming to enhance performance in SCR tasks. Training and testing with the Google Speech Command (GSC) dataset, the proposed approach showed effectiveness in classifying speech commands. Moreover, a comparative analysis with Genetic Algorithm based selections and other deep CNN (DCNN) models highlighted the efficiency of the proposed DE algorithm in hyper-parameter selection for CNNs in SCR tasks.

{{</citation>}}


### (109/114) Learning to Behave Like Clean Speech: Dual-Branch Knowledge Distillation for Noise-Robust Fake Audio Detection (Cunhang Fan et al., 2023)

{{<citation>}}

Cunhang Fan, Mingming Ding, Jianhua Tao, Ruibo Fu, Jiangyan Yi, Zhengqi Wen, Zhao Lv. (2023)  
**Learning to Behave Like Clean Speech: Dual-Branch Knowledge Distillation for Noise-Robust Fake Audio Detection**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.08869v1)  

---


**ABSTRACT**  
Most research in fake audio detection (FAD) focuses on improving performance on standard noise-free datasets. However, in actual situations, there is usually noise interference, which will cause significant performance degradation in FAD systems. To improve the noise robustness, we propose a dual-branch knowledge distillation fake audio detection (DKDFAD) method. Specifically, a parallel data flow of the clean teacher branch and the noisy student branch is designed, and interactive fusion and response-based teacher-student paradigms are proposed to guide the training of noisy data from the data distribution and decision-making perspectives. In the noise branch, speech enhancement is first introduced for denoising, which reduces the interference of strong noise. The proposed interactive fusion combines denoising features and noise features to reduce the impact of speech distortion and seek consistency with the data distribution of clean branch. The teacher-student paradigm maps the student's decision space to the teacher's decision space, making noisy speech behave as clean. In addition, a joint training method is used to optimize the two branches to achieve global optimality. Experimental results based on multiple datasets show that the proposed method performs well in noisy environments and maintains performance in cross-dataset experiments.

{{</citation>}}


## cs.MA (1)



### (110/114) Welfare Diplomacy: Benchmarking Language Model Cooperation (Gabriel Mukobi et al., 2023)

{{<citation>}}

Gabriel Mukobi, Hannah Erlebach, Niklas Lauffer, Lewis Hammond, Alan Chan, Jesse Clifton. (2023)  
**Welfare Diplomacy: Benchmarking Language Model Cooperation**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-CL, cs-MA, cs.MA  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2310.08901v1)  

---


**ABSTRACT**  
The growing capabilities and increasingly widespread deployment of AI systems necessitate robust benchmarks for measuring their cooperative capabilities. Unfortunately, most multi-agent benchmarks are either zero-sum or purely cooperative, providing limited opportunities for such measurements. We introduce a general-sum variant of the zero-sum board game Diplomacy -- called Welfare Diplomacy -- in which players must balance investing in military conquest and domestic welfare. We argue that Welfare Diplomacy facilitates both a clearer assessment of and stronger training incentives for cooperative capabilities. Our contributions are: (1) proposing the Welfare Diplomacy rules and implementing them via an open-source Diplomacy engine; (2) constructing baseline agents using zero-shot prompted language models; and (3) conducting experiments where we find that baselines using state-of-the-art models attain high social welfare but are exploitable. Our work aims to promote societal safety by aiding researchers in developing and assessing multi-agent AI systems. Code to evaluate Welfare Diplomacy and reproduce our experiments is available at https://github.com/mukobi/welfare-diplomacy.

{{</citation>}}


## cs.SE (2)



### (111/114) A Critical Review of Large Language Model on Software Engineering: An Example from ChatGPT and Automated Program Repair (Quanjun Zhang et al., 2023)

{{<citation>}}

Quanjun Zhang, Tongke Zhang, Juan Zhai, Chunrong Fang, Bowen Yu, Weisong Sun, Zhenyu Chen. (2023)  
**A Critical Review of Large Language Model on Software Engineering: An Example from ChatGPT and Automated Program Repair**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2310.08879v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have been gaining increasing attention and demonstrated promising performance across a variety of Software Engineering (SE) tasks, such as Automated Program Repair (APR), code summarization, and code completion. For example, ChatGPT, the latest black-box LLM, has been investigated by numerous recent research studies and has shown impressive performance in various tasks. However, there exists a potential risk of data leakage since these LLMs are usually close-sourced with unknown specific training details, e.g., pre-training datasets.   In this paper, we seek to review the bug-fixing capabilities of ChatGPT on a clean APR benchmark with different research objectives. We first introduce {\benchmark}, a new benchmark with buggy and the corresponding fixed programs from competitive programming problems starting from 2023, after the training cutoff point of ChatGPT. The results on {\benchmark} show that ChatGPT is able to fix 109 out of 151 buggy programs using the basic prompt within 35 independent rounds, outperforming state-of-the-art LLMs CodeT5 and PLBART by 27.5\% and 62.4\% prediction accuracy. We also investigate the impact of three types of prompts, i.e., problem description, error feedback, and bug localization, leading to additional 34 fixed bugs. Besides, we provide additional discussion from the interactive nature of ChatGPT to illustrate the capacity of a dialog-based repair workflow with 9 additional fixed bugs. Inspired by the findings, we further pinpoint various challenges and opportunities for advanced SE study equipped with such LLMs (e.g.,~ChatGPT) in the near future. More importantly, our work calls for more research on the reevaluation of the achievements obtained by existing black-box LLMs across various SE tasks, not limited to ChatGPT on APR.

{{</citation>}}


### (112/114) Static Code Analysis in the AI Era: An In-depth Exploration of the Concept, Function, and Potential of Intelligent Code Analysis Agents (Gang Fan et al., 2023)

{{<citation>}}

Gang Fan, Xiaoheng Xie, Xunjin Zheng, Yinan Liang, Peng Di. (2023)  
**Static Code Analysis in the AI Era: An In-depth Exploration of the Concept, Function, and Potential of Intelligent Code Analysis Agents**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2310.08837v1)  

---


**ABSTRACT**  
The escalating complexity of software systems and accelerating development cycles pose a significant challenge in managing code errors and implementing business logic. Traditional techniques, while cornerstone for software quality assurance, exhibit limitations in handling intricate business logic and extensive codebases. To address these challenges, we introduce the Intelligent Code Analysis Agent (ICAA), a novel concept combining AI models, engineering process designs, and traditional non-AI components. The ICAA employs the capabilities of large language models (LLMs) such as GPT-3 or GPT-4 to automatically detect and diagnose code errors and business logic inconsistencies. In our exploration of this concept, we observed a substantial improvement in bug detection accuracy, reducing the false-positive rate to 66\% from the baseline's 85\%, and a promising recall rate of 60.8\%. However, the token consumption cost associated with LLMs, particularly the average cost for analyzing each line of code, remains a significant consideration for widespread adoption. Despite this challenge, our findings suggest that the ICAA holds considerable potential to revolutionize software quality assurance, significantly enhancing the efficiency and accuracy of bug detection in the software development process. We hope this pioneering work will inspire further research and innovation in this field, focusing on refining the ICAA concept and exploring ways to mitigate the associated costs.

{{</citation>}}


## cs.HC (1)



### (113/114) Confounding-Robust Policy Improvement with Human-AI Teams (Ruijiang Gao et al., 2023)

{{<citation>}}

Ruijiang Gao, Mingzhang Yin. (2023)  
**Confounding-Robust Policy Improvement with Human-AI Teams**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.08824v1)  

---


**ABSTRACT**  
Human-AI collaboration has the potential to transform various domains by leveraging the complementary strengths of human experts and Artificial Intelligence (AI) systems. However, unobserved confounding can undermine the effectiveness of this collaboration, leading to biased and unreliable outcomes. In this paper, we propose a novel solution to address unobserved confounding in human-AI collaboration by employing the marginal sensitivity model (MSM). Our approach combines domain expertise with AI-driven statistical modeling to account for potential confounders that may otherwise remain hidden. We present a deferral collaboration framework for incorporating the MSM into policy learning from observational data, enabling the system to control for the influence of unobserved confounding factors. In addition, we propose a personalized deferral collaboration system to leverage the diverse expertise of different human decision-makers. By adjusting for potential biases, our proposed solution enhances the robustness and reliability of collaborative outcomes. The empirical and theoretical analyses demonstrate the efficacy of our approach in mitigating unobserved confounding and improving the overall performance of human-AI collaborations.

{{</citation>}}


## stat.ME (1)



### (114/114) A Nonlinear Method for time series forecasting using VMD-GARCH-LSTM model (Zhengtao Gui et al., 2023)

{{<citation>}}

Zhengtao Gui, Haoyuan Li, Sijie Xu, Yu Chen. (2023)  
**A Nonlinear Method for time series forecasting using VMD-GARCH-LSTM model**  

---
Primary Category: stat.ME  
Categories: cs-LG, stat-ME, stat.ME  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2310.08812v1)  

---


**ABSTRACT**  
Time series forecasting represents a significant and challenging task across various fields. Recently, methods based on mode decomposition have dominated the forecasting of complex time series because of the advantages of capturing local characteristics and extracting intrinsic modes from data. Unfortunately, most models fail to capture the implied volatilities that contain significant information. To enhance the forecasting of current, rapidly evolving, and volatile time series, we propose a novel decomposition-ensemble paradigm, the VMD-LSTM-GARCH model. The Variational Mode Decomposition algorithm is employed to decompose the time series into K sub-modes. Subsequently, the GARCH model extracts the volatility information from these sub-modes, which serve as the input for the LSTM. The numerical and volatility information of each sub-mode is utilized to train a Long Short-Term Memory network. This network predicts the sub-mode, and then we aggregate the predictions from all sub-modes to produce the output. By integrating econometric and artificial intelligence methods, and taking into account both the numerical and volatility information of the time series, our proposed model demonstrates superior performance in time series forecasting, as evidenced by the significant decrease in MSE, RMSE, and MAPE in our comparative experimental results.

{{</citation>}}
