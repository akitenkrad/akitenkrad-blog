---
draft: false
title: "arXiv @ 2023.12.12"
date: 2023-12-12
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.12.12"
    identifier: arxiv_20231212
    parent: 202312_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (19)](#cscv-19)
- [cs.AI (12)](#csai-12)
- [cs.LG (16)](#cslg-16)
- [eess.SP (1)](#eesssp-1)
- [cs.HC (1)](#cshc-1)
- [cs.CL (8)](#cscl-8)
- [cs.CR (3)](#cscr-3)
- [cs.SD (1)](#cssd-1)
- [cs.RO (4)](#csro-4)
- [cs.NE (1)](#csne-1)
- [eess.IV (1)](#eessiv-1)
- [cs.CY (1)](#cscy-1)
- [quant-ph (1)](#quant-ph-1)
- [eess.SY (2)](#eesssy-2)
- [cs.IR (1)](#csir-1)
- [cs.SE (1)](#csse-1)
- [cs.SI (1)](#cssi-1)
- [cs.MM (1)](#csmm-1)

## cs.CV (19)



### (1/75) Correcting Diffusion Generation through Resampling (Yujian Liu et al., 2023)

{{<citation>}}

Yujian Liu, Yang Zhang, Tommi Jaakkola, Shiyu Chang. (2023)  
**Correcting Diffusion Generation through Resampling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2312.06038v1)  

---


**ABSTRACT**  
Despite diffusion models' superior capabilities in modeling complex distributions, there are still non-trivial distributional discrepancies between generated and ground-truth images, which has resulted in several notable problems in image generation, including missing object errors in text-to-image generation and low image quality. Existing methods that attempt to address these problems mostly do not tend to address the fundamental cause behind these problems, which is the distributional discrepancies, and hence achieve sub-optimal results. In this paper, we propose a particle filtering framework that can effectively address both problems by explicitly reducing the distributional discrepancies. Specifically, our method relies on a set of external guidance, including a small set of real images and a pre-trained object detector, to gauge the distribution gap, and then design the resampling weight accordingly to correct the gap. Experiments show that our methods can effectively correct missing object errors and improve image quality in various image generation tasks. Notably, our method outperforms the existing strongest baseline by 5% in object occurrence and 1.0 in FID on MS-COCO. Our code is publicly available at https://github.com/UCSB-NLP-Chang/diffusion_resampling.git.

{{</citation>}}


### (2/75) GenDepth: Generalizing Monocular Depth Estimation for Arbitrary Camera Parameters via Ground Plane Embedding (Karlo Koledić et al., 2023)

{{<citation>}}

Karlo Koledić, Luka Petrović, Ivan Petrović, Ivan Marković. (2023)  
**GenDepth: Generalizing Monocular Depth Estimation for Arbitrary Camera Parameters via Ground Plane Embedding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.06021v1)  

---


**ABSTRACT**  
Learning-based monocular depth estimation leverages geometric priors present in the training data to enable metric depth perception from a single image, a traditionally ill-posed problem. However, these priors are often specific to a particular domain, leading to limited generalization performance on unseen data. Apart from the well studied environmental domain gap, monocular depth estimation is also sensitive to the domain gap induced by varying camera parameters, an aspect that is often overlooked in current state-of-the-art approaches. This issue is particularly evident in autonomous driving scenarios, where datasets are typically collected with a single vehicle-camera setup, leading to a bias in the training data due to a fixed perspective geometry. In this paper, we challenge this trend and introduce GenDepth, a novel model capable of performing metric depth estimation for arbitrary vehicle-camera setups. To address the lack of data with sufficiently diverse camera parameters, we first create a bespoke synthetic dataset collected with different vehicle-camera systems. Then, we design GenDepth to simultaneously optimize two objectives: (i) equivariance to the camera parameter variations on synthetic data, (ii) transferring the learned equivariance to real-world environmental features using a single real-world dataset with a fixed vehicle-camera system. To achieve this, we propose a novel embedding of camera parameters as the ground plane depth and present a novel architecture that integrates these embeddings with adversarial domain alignment. We validate GenDepth on several autonomous driving datasets, demonstrating its state-of-the-art generalization capability for different vehicle-camera systems.

{{</citation>}}


### (3/75) FM-G-CAM: A Holistic Approach for Explainable AI in Computer Vision (Ravidu Suien Rammuni Silva et al., 2023)

{{<citation>}}

Ravidu Suien Rammuni Silva, Jordan J. Bird. (2023)  
**FM-G-CAM: A Holistic Approach for Explainable AI in Computer Vision**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI, Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.05975v1)  

---


**ABSTRACT**  
Explainability is an aspect of modern AI that is vital for impact and usability in the real world. The main objective of this paper is to emphasise the need to understand the predictions of Computer Vision models, specifically Convolutional Neural Network (CNN) based models. Existing methods of explaining CNN predictions are mostly based on Gradient-weighted Class Activation Maps (Grad-CAM) and solely focus on a single target class. We show that from the point of the target class selection, we make an assumption on the prediction process, hence neglecting a large portion of the predictor CNN model's thinking process. In this paper, we present an exhaustive methodology called Fused Multi-class Gradient-weighted Class Activation Map (FM-G-CAM) that considers multiple top predicted classes, which provides a holistic explanation of the predictor CNN's thinking rationale. We also provide a detailed and comprehensive mathematical and algorithmic description of our method. Furthermore, along with a concise comparison of existing methods, we compare FM-G-CAM with Grad-CAM, highlighting its benefits through real-world practical use cases. Finally, we present an open-source Python library with FM-G-CAM implementation to conveniently generate saliency maps for CNN-based model predictions.

{{</citation>}}


### (4/75) Activating Frequency and ViT for 3D Point Cloud Quality Assessment without Reference (Oussama Messai et al., 2023)

{{<citation>}}

Oussama Messai, Abdelouahid Bentamou, Abbass Zein-Eddine, Yann Gavet. (2023)  
**Activating Frequency and ViT for 3D Point Cloud Quality Assessment without Reference**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-MM, cs.CV, eess-IV  
Keywords: QA, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2312.05972v1)  

---


**ABSTRACT**  
Deep learning-based quality assessments have significantly enhanced perceptual multimedia quality assessment, however it is still in the early stages for 3D visual data such as 3D point clouds (PCs). Due to the high volume of 3D-PCs, such quantities are frequently compressed for transmission and viewing, which may affect perceived quality. Therefore, we propose no-reference quality metric of a given 3D-PC. Comparing to existing methods that mostly focus on geometry or color aspects, we propose integrating frequency magnitudes as indicator of spatial degradation patterns caused by the compression. To map the input attributes to quality score, we use a light-weight hybrid deep model; combined of Deformable Convolutional Network (DCN) and Vision Transformers (ViT). Experiments are carried out on ICIP20 [1], PointXR [2] dataset, and a new big dataset called BASICS [3]. The results show that our approach outperforms state-of-the-art NR-PCQA measures and even some FR-PCQA on PointXR. The implementation code can be found at: https://github.com/o-messai/3D-PCQA

{{</citation>}}


### (5/75) Jumpstarting Surgical Computer Vision (Deepak Alapatt et al., 2023)

{{<citation>}}

Deepak Alapatt, Aditya Murali, Vinkle Srivastav, Pietro Mascagni, AI4SafeChole Consortium, Nicolas Padoy. (2023)  
**Jumpstarting Surgical Computer Vision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.05968v1)  

---


**ABSTRACT**  
Purpose: General consensus amongst researchers and industry points to a lack of large, representative annotated datasets as the biggest obstacle to progress in the field of surgical data science. Self-supervised learning represents a solution to part of this problem, removing the reliance on annotations. However, the robustness of current self-supervised learning methods to domain shifts remains unclear, limiting our understanding of its utility for leveraging diverse sources of surgical data. Methods: In this work, we employ self-supervised learning to flexibly leverage diverse surgical datasets, thereby learning taskagnostic representations that can be used for various surgical downstream tasks. Based on this approach, to elucidate the impact of pre-training on downstream task performance, we explore 22 different pre-training dataset combinations by modulating three variables: source hospital, type of surgical procedure, and pre-training scale (number of videos). We then finetune the resulting model initializations on three diverse downstream tasks: namely, phase recognition and critical view of safety in laparoscopic cholecystectomy and phase recognition in laparoscopic hysterectomy. Results: Controlled experimentation highlights sizable boosts in performance across various tasks, datasets, and labeling budgets. However, this performance is intricately linked to the composition of the pre-training dataset, robustly proven through several study stages. Conclusion: The composition of pre-training datasets can severely affect the effectiveness of SSL methods for various downstream tasks and should critically inform future data collection efforts to scale the application of SSL methodologies.   Keywords: Self-Supervised Learning, Transfer Learning, Surgical Computer Vision, Endoscopic Videos, Critical View of Safety, Phase Recognition

{{</citation>}}


### (6/75) Aikyam: A Video Conferencing Utility for Deaf and Dumb (Kshitij Deshpande et al., 2023)

{{<citation>}}

Kshitij Deshpande, Varad Mashalkar, Kaustubh Mhaisekar, Amaan Naikwadi, Archana Ghotkar. (2023)  
**Aikyam: A Video Conferencing Utility for Deaf and Dumb**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.05962v1)  

---


**ABSTRACT**  
With the advent of the pandemic, the use of video conferencing platforms as a means of communication has greatly increased and with it, so have the remote opportunities. The deaf and dumb have traditionally faced several issues in communication, but now the effect is felt more severely. This paper proposes an all-encompassing video conferencing utility that can be used with existing video conferencing platforms to address these issues. Appropriate semantically correct sentences are generated from the signer's gestures which would be interpreted by the system. Along with an audio to emit this sentence, the user's feed is also used to annotate the sentence. This can be viewed by all participants, thus aiding smooth communication with all parties involved. This utility utilizes a simple LSTM model for classification of gestures. The sentences are constructed by a t5 based model. In order to achieve the required data flow, a virtual camera is used.

{{</citation>}}


### (7/75) ASH: Animatable Gaussian Splats for Efficient and Photoreal Human Rendering (Haokai Pang et al., 2023)

{{<citation>}}

Haokai Pang, Heming Zhu, Adam Kortylewski, Christian Theobalt, Marc Habermann. (2023)  
**ASH: Animatable Gaussian Splats for Efficient and Photoreal Human Rendering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2312.05941v1)  

---


**ABSTRACT**  
Real-time rendering of photorealistic and controllable human avatars stands as a cornerstone in Computer Vision and Graphics. While recent advances in neural implicit rendering have unlocked unprecedented photorealism for digital avatars, real-time performance has mostly been demonstrated for static scenes only. To address this, we propose ASH, an animatable Gaussian splatting approach for photorealistic rendering of dynamic humans in real-time. We parameterize the clothed human as animatable 3D Gaussians, which can be efficiently splatted into image space to generate the final rendering. However, naively learning the Gaussian parameters in 3D space poses a severe challenge in terms of compute. Instead, we attach the Gaussians onto a deformable character model, and learn their parameters in 2D texture space, which allows leveraging efficient 2D convolutional architectures that easily scale with the required number of Gaussians. We benchmark ASH with competing methods on pose-controllable avatars, demonstrating that our method outperforms existing real-time methods by a large margin and shows comparable or even better results than offline methods.

{{</citation>}}


### (8/75) AM-RADIO: Agglomerative Model -- Reduce All Domains Into One (Mike Ranzinger et al., 2023)

{{<citation>}}

Mike Ranzinger, Greg Heinrich, Jan Kautz, Pavlo Molchanov. (2023)  
**AM-RADIO: Agglomerative Model -- Reduce All Domains Into One**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2312.06709v1)  

---


**ABSTRACT**  
A handful of visual foundation models (VFMs) have recently emerged as the backbones for numerous downstream tasks. VFMs like CLIP, DINOv2, SAM are trained with distinct objectives, exhibiting unique characteristics for various downstream tasks. We find that despite their conceptual differences, these models can be effectively merged into a unified model through multi-teacher distillation. We name this approach AM-RADIO (Agglomerative Model -- Reduce All Domains Into One). This integrative approach not only surpasses the performance of individual teacher models but also amalgamates their distinctive features, such as zero-shot vision-language comprehension, detailed pixel-level understanding, and open vocabulary segmentation capabilities. In pursuit of the most hardware-efficient backbone, we evaluated numerous architectures in our multi-teacher distillation pipeline using the same training recipe. This led to the development of a novel architecture (E-RADIO) that exceeds the performance of its predecessors and is at least 7x faster than the teacher models. Our comprehensive benchmarking process covers downstream tasks including ImageNet classification, ADE20k semantic segmentation, COCO object detection and LLaVa-1.5 framework.   Code: https://github.com/NVlabs/RADIO

{{</citation>}}


### (9/75) AesFA: An Aesthetic Feature-Aware Arbitrary Neural Style Transfer (Joonwoo Kwon et al., 2023)

{{<citation>}}

Joonwoo Kwon, Sooyoung Kim, Yuewei Lin, Shinjae Yoo, Jiook Cha. (2023)  
**AesFA: An Aesthetic Feature-Aware Arbitrary Neural Style Transfer**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2312.05928v1)  

---


**ABSTRACT**  
Neural style transfer (NST) has evolved significantly in recent years. Yet, despite its rapid progress and advancement, existing NST methods either struggle to transfer aesthetic information from a style effectively or suffer from high computational costs and inefficiencies in feature disentanglement due to using pre-trained models. This work proposes a lightweight but effective model, AesFA -- Aesthetic Feature-Aware NST. The primary idea is to decompose the image via its frequencies to better disentangle aesthetic styles from the reference image while training the entire model in an end-to-end manner to exclude pre-trained models at inference completely. To improve the network's ability to extract more distinct representations and further enhance the stylization quality, this work introduces a new aesthetic feature: contrastive loss. Extensive experiments and ablations show the approach not only outperforms recent NST methods in terms of stylization quality, but it also achieves faster inference. Codes are available at https://github.com/Sooyyoungg/AesFA.

{{</citation>}}


### (10/75) Hypergraph-Guided Disentangled Spectrum Transformer Networks for Near-Infrared Facial Expression Recognition (Bingjun Luo et al., 2023)

{{<citation>}}

Bingjun Luo, Haowen Wang, Jinpeng Wang, Junjie Zhu, Xibin Zhao, Yue Gao. (2023)  
**Hypergraph-Guided Disentangled Spectrum Transformer Networks for Near-Infrared Facial Expression Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-HC, cs.CV  
Keywords: Attention, Embedding, Self-Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2312.05907v1)  

---


**ABSTRACT**  
With the strong robusticity on illumination variations, near-infrared (NIR) can be an effective and essential complement to visible (VIS) facial expression recognition in low lighting or complete darkness conditions. However, facial expression recognition (FER) from NIR images presents more challenging problem than traditional FER due to the limitations imposed by the data scale and the difficulty of extracting discriminative features from incomplete visible lighting contents. In this paper, we give the first attempt to deep NIR facial expression recognition and proposed a novel method called near-infrared facial expression transformer (NFER-Former). Specifically, to make full use of the abundant label information in the field of VIS, we introduce a Self-Attention Orthogonal Decomposition mechanism that disentangles the expression information and spectrum information from the input image, so that the expression features can be extracted without the interference of spectrum variation. We also propose a Hypergraph-Guided Feature Embedding method that models some key facial behaviors and learns the structure of the complex correlations between them, thereby alleviating the interference of inter-class similarity. Additionally, we have constructed a large NIR-VIS Facial Expression dataset that includes 360 subjects to better validate the efficiency of NFER-Former. Extensive experiments and ablation studies show that NFER-Former significantly improves the performance of NIR FER and achieves state-of-the-art results on the only two available NIR FER datasets, Oulu-CASIA and Large-HFE.

{{</citation>}}


### (11/75) PSCR: Patches Sampling-based Contrastive Regression for AIGC Image Quality Assessment (Jiquan Yuan et al., 2023)

{{<citation>}}

Jiquan Yuan, Xinyan Cao, Linjing Cao, Jinlong Lin, Xixin Cao. (2023)  
**PSCR: Patches Sampling-based Contrastive Regression for AIGC Image Quality Assessment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, QA  
[Paper Link](http://arxiv.org/abs/2312.05897v1)  

---


**ABSTRACT**  
In recent years, Artificial Intelligence Generated Content (AIGC) has gained widespread attention beyond the computer science community. Due to various issues arising from continuous creation of AI-generated images (AIGI), AIGC image quality assessment (AIGCIQA), which aims to evaluate the quality of AIGIs from human perception perspectives, has emerged as a novel topic in the field of computer vision. However, most existing AIGCIQA methods directly regress predicted scores from a single generated image, overlooking the inherent differences among AIGIs and scores. Additionally, operations like resizing and cropping may cause global geometric distortions and information loss, thus limiting the performance of models. To address these issues, we propose a patches sampling-based contrastive regression (PSCR) framework. We suggest introducing a contrastive regression framework to leverage differences among various generated images for learning a better representation space. In this space, differences and score rankings among images can be measured by their relative scores. By selecting exemplar AIGIs as references, we also overcome the limitations of previous models that could not utilize reference images on the no-reference image databases. To avoid geometric distortions and information loss in image inputs, we further propose a patches sampling strategy. To demonstrate the effectiveness of our proposed PSCR framework, we conduct extensive experiments on three mainstream AIGCIQA databases including AGIQA-1K, AGIQA-3K and AIGCIQA2023. The results show significant improvements in model performance with the introduction of our proposed PSCR framework. Code will be available at \url{https://github.com/jiquan123/PSCR}.

{{</citation>}}


### (12/75) SIFU: Side-view Conditioned Implicit Function for Real-world Usable Clothed Human Reconstruction (Zechuan Zhang et al., 2023)

{{<citation>}}

Zechuan Zhang, Zongxin Yang, Yi Yang. (2023)  
**SIFU: Side-view Conditioned Implicit Function for Real-world Usable Clothed Human Reconstruction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.06704v1)  

---


**ABSTRACT**  
Creating high-quality 3D models of clothed humans from single images for real-world applications is crucial. Despite recent advancements, accurately reconstructing humans in complex poses or with loose clothing from in-the-wild images, along with predicting textures for unseen areas, remains a significant challenge. A key limitation of previous methods is their insufficient prior guidance in transitioning from 2D to 3D and in texture prediction. In response, we introduce SIFU (Side-view Conditioned Implicit Function for Real-world Usable Clothed Human Reconstruction), a novel approach combining a Side-view Decoupling Transformer with a 3D Consistent Texture Refinement pipeline.SIFU employs a cross-attention mechanism within the transformer, using SMPL-X normals as queries to effectively decouple side-view features in the process of mapping 2D features to 3D. This method not only improves the precision of the 3D models but also their robustness, especially when SMPL-X estimates are not perfect. Our texture refinement process leverages text-to-image diffusion-based prior to generate realistic and consistent textures for invisible views. Through extensive experiments, SIFU surpasses SOTA methods in both geometry and texture reconstruction, showcasing enhanced robustness in complex scenarios and achieving an unprecedented Chamfer and P2S measurement. Our approach extends to practical applications such as 3D printing and scene building, demonstrating its broad utility in real-world scenarios. Project page https://river-zhang.github.io/SIFU-projectpage/ .

{{</citation>}}


### (13/75) A Video is Worth 256 Bases: Spatial-Temporal Expectation-Maximization Inversion for Zero-Shot Video Editing (Maomao Li et al., 2023)

{{<citation>}}

Maomao Li, Yu Li, Tianyu Yang, Yunfei Liu, Dongxu Yue, Zhihui Lin, Dong Xu. (2023)  
**A Video is Worth 256 Bases: Spatial-Temporal Expectation-Maximization Inversion for Zero-Shot Video Editing**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2312.05856v1)  

---


**ABSTRACT**  
This paper presents a video inversion approach for zero-shot video editing, which aims to model the input video with low-rank representation during the inversion process. The existing video editing methods usually apply the typical 2D DDIM inversion or na\"ive spatial-temporal DDIM inversion before editing, which leverages time-varying representation for each frame to derive noisy latent. Unlike most existing approaches, we propose a Spatial-Temporal Expectation-Maximization (STEM) inversion, which formulates the dense video feature under an expectation-maximization manner and iteratively estimates a more compact basis set to represent the whole video. Each frame applies the fixed and global representation for inversion, which is more friendly for temporal consistency during reconstruction and editing. Extensive qualitative and quantitative experiments demonstrate that our STEM inversion can achieve consistent improvement on two state-of-the-art video editing methods.

{{</citation>}}


### (14/75) Transformer-based Selective Super-Resolution for Efficient Image Refinement (Tianyi Zhang et al., 2023)

{{<citation>}}

Tianyi Zhang, Kishore Kasichainula, Yaoxin Zhuo, Baoxin Li, Jae-sun Seo, Yu Cao. (2023)  
**Transformer-based Selective Super-Resolution for Efficient Image Refinement**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.05803v1)  

---


**ABSTRACT**  
Conventional super-resolution methods suffer from two drawbacks: substantial computational cost in upscaling an entire large image, and the introduction of extraneous or potentially detrimental information for downstream computer vision tasks during the refinement of the background. To solve these issues, we propose a novel transformer-based algorithm, Selective Super-Resolution (SSR), which partitions images into non-overlapping tiles, selects tiles of interest at various scales with a pyramid architecture, and exclusively reconstructs these selected tiles with deep features. Experimental results on three datasets demonstrate the efficiency and robust performance of our approach for super-resolution. Compared to the state-of-the-art methods, the FID score is reduced from 26.78 to 10.41 with 40% reduction in computation cost for the BDD100K dataset. The source code is available at https://github.com/destiny301/SSR.

{{</citation>}}


### (15/75) Disentangled Representation Learning for Controllable Person Image Generation (Wenju Xu et al., 2023)

{{<citation>}}

Wenju Xu, Chengjiang Long, Yongwei Nie, Guanghui Wang. (2023)  
**Disentangled Representation Learning for Controllable Person Image Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2312.05798v1)  

---


**ABSTRACT**  
In this paper, we propose a novel framework named DRL-CPG to learn disentangled latent representation for controllable person image generation, which can produce realistic person images with desired poses and human attributes (e.g., pose, head, upper clothes, and pants) provided by various source persons. Unlike the existing works leveraging the semantic masks to obtain the representation of each component, we propose to generate disentangled latent code via a novel attribute encoder with transformers trained in a manner of curriculum learning from a relatively easy step to a gradually hard one. A random component mask-agnostic strategy is introduced to randomly remove component masks from the person segmentation masks, which aims at increasing the difficulty of training and promoting the transformer encoder to recognize the underlying boundaries between each component. This enables the model to transfer both the shape and texture of the components. Furthermore, we propose a novel attribute decoder network to integrate multi-level attributes (e.g., the structure feature and the attribute representation) with well-designed Dual Adaptive Denormalization (DAD) residual blocks. Extensive experiments strongly demonstrate that the proposed approach is able to transfer both the texture and shape of different human parts and yield realistic results. To our knowledge, we are the first to learn disentangled latent representations with transformers for person image generation.

{{</citation>}}


### (16/75) AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model (Teng Hu et al., 2023)

{{<citation>}}

Teng Hu, Jiangning Zhang, Ran Yi, Yuzhen Du, Xu Chen, Liang Liu, Yabiao Wang, Chengjie Wang. (2023)  
**AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Embedding, Few-Shot  
[Paper Link](http://arxiv.org/abs/2312.05767v1)  

---


**ABSTRACT**  
Anomaly inspection plays an important role in industrial manufacture. Existing anomaly inspection methods are limited in their performance due to insufficient anomaly data. Although anomaly generation methods have been proposed to augment the anomaly data, they either suffer from poor generation authenticity or inaccurate alignment between the generated anomalies and masks. To address the above problems, we propose AnomalyDiffusion, a novel diffusion-based few-shot anomaly generation model, which utilizes the strong prior information of latent diffusion model learned from large-scale dataset to enhance the generation authenticity under few-shot training data. Firstly, we propose Spatial Anomaly Embedding, which consists of a learnable anomaly embedding and a spatial embedding encoded from an anomaly mask, disentangling the anomaly information into anomaly appearance and location information. Moreover, to improve the alignment between the generated anomalies and the anomaly masks, we introduce a novel Adaptive Attention Re-weighting Mechanism. Based on the disparities between the generated anomaly image and normal sample, it dynamically guides the model to focus more on the areas with less noticeable generated anomalies, enabling generation of accurately-matched anomalous image-mask pairs. Extensive experiments demonstrate that our model significantly outperforms the state-of-the-art methods in generation authenticity and diversity, and effectively improves the performance of downstream anomaly inspection tasks. The code and data are available in https://github.com/sjtuplayer/anomalydiffusion.

{{</citation>}}


### (17/75) Benchmarking of Query Strategies: Towards Future Deep Active Learning (Shiryu Ueno et al., 2023)

{{<citation>}}

Shiryu Ueno, Yusei Yamada, Shunsuke Nakatsuka, Kunihito Kato. (2023)  
**Benchmarking of Query Strategies: Towards Future Deep Active Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2312.05751v1)  

---


**ABSTRACT**  
In this study, we benchmark query strategies for deep actice learning~(DAL). DAL reduces annotation costs by annotating only high-quality samples selected by query strategies. Existing research has two main problems, that the experimental settings are not standardized, making the evaluation of existing methods is difficult, and that most of experiments were conducted on the CIFAR or MNIST datasets. Therefore, we develop standardized experimental settings for DAL and investigate the effectiveness of various query strategies using six datasets, including those that contain medical and visual inspection images. In addition, since most current DAL approaches are model-based, we perform verification experiments using fully-trained models for querying to investigate the effectiveness of these approaches for the six datasets. Our code is available at \href{https://github.com/ia-gu/Benchmarking-of-Query-Strategies-Towards-Future-Deep-Active-Learning}

{{</citation>}}


### (18/75) Open World Object Detection in the Era of Foundation Models (Orr Zohar et al., 2023)

{{<citation>}}

Orr Zohar, Alejandro Lozano, Shelly Goel, Serena Yeung, Kuan-Chieh Wang. (2023)  
**Open World Object Detection in the Era of Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2312.05745v1)  

---


**ABSTRACT**  
Object detection is integral to a bevy of real-world applications, from robotics to medical image analysis. To be used reliably in such applications, models must be capable of handling unexpected - or novel - objects. The open world object detection (OWD) paradigm addresses this challenge by enabling models to detect unknown objects and learn discovered ones incrementally. However, OWD method development is hindered due to the stringent benchmark and task definitions. These definitions effectively prohibit foundation models. Here, we aim to relax these definitions and investigate the utilization of pre-trained foundation models in OWD. First, we show that existing benchmarks are insufficient in evaluating methods that utilize foundation models, as even naive integration methods nearly saturate these benchmarks. This result motivated us to curate a new and challenging benchmark for these models. Therefore, we introduce a new benchmark that includes five real-world application-driven datasets, including challenging domains such as aerial and surgical images, and establish baselines. We exploit the inherent connection between classes in application-driven datasets and introduce a novel method, Foundation Object detection Model for the Open world, or FOMO, which identifies unknown objects based on their shared attributes with the base known objects. FOMO has ~3x unknown object mAP compared to baselines on our benchmark. However, our results indicate a significant place for improvement - suggesting a great research opportunity in further scaling object detection methods to real-world domains. Our code and benchmark are available at https://orrzohar.github.io/projects/fomo/.

{{</citation>}}


### (19/75) Leveraging Generative Language Models for Weakly Supervised Sentence Component Analysis in Video-Language Joint Learning (Zaber Ibn Abdul Hakim et al., 2023)

{{<citation>}}

Zaber Ibn Abdul Hakim, Najibul Haque Sarker, Rahul Pratap Singh, Bishmoy Paul, Ali Dabouei, Min Xu. (2023)  
**Leveraging Generative Language Models for Weakly Supervised Sentence Component Analysis in Video-Language Joint Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.06699v1)  

---


**ABSTRACT**  
A thorough comprehension of textual data is a fundamental element in multi-modal video analysis tasks. However, recent works have shown that the current models do not achieve a comprehensive understanding of the textual data during the training for the target downstream tasks. Orthogonal to the previous approaches to this limitation, we postulate that understanding the significance of the sentence components according to the target task can potentially enhance the performance of the models. Hence, we utilize the knowledge of a pre-trained large language model (LLM) to generate text samples from the original ones, targeting specific sentence components. We propose a weakly supervised importance estimation module to compute the relative importance of the components and utilize them to improve different video-language tasks. Through rigorous quantitative analysis, our proposed method exhibits significant improvement across several video-language tasks. In particular, our approach notably enhances video-text retrieval by a relative improvement of 8.3\% in video-to-text and 1.4\% in text-to-video retrieval over the baselines, in terms of R@1. Additionally, in video moment retrieval, average mAP shows a relative improvement ranging from 2.0\% to 13.7 \% across different baselines.

{{</citation>}}


## cs.AI (12)



### (20/75) Multimodality of AI for Education: Towards Artificial General Intelligence (Gyeong-Geon Lee et al., 2023)

{{<citation>}}

Gyeong-Geon Lee, Lehong Shi, Ehsan Latif, Yizhu Gao, Arne Bewersdorff, Matthew Nyaaba, Shuchen Guo, Zihao Wu, Zhengliang Liu, Hui Wang, Gengchen Mai, Tiaming Liu, Xiaoming Zhai. (2023)  
**Multimodality of AI for Education: Towards Artificial General Intelligence**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.06037v2)  

---


**ABSTRACT**  
This paper presents a comprehensive examination of how multimodal artificial intelligence (AI) approaches are paving the way towards the realization of Artificial General Intelligence (AGI) in educational contexts. It scrutinizes the evolution and integration of AI in educational systems, emphasizing the crucial role of multimodality, which encompasses auditory, visual, kinesthetic, and linguistic modes of learning. This research delves deeply into the key facets of AGI, including cognitive frameworks, advanced knowledge representation, adaptive learning mechanisms, strategic planning, sophisticated language processing, and the integration of diverse multimodal data sources. It critically assesses AGI's transformative potential in reshaping educational paradigms, focusing on enhancing teaching and learning effectiveness, filling gaps in existing methodologies, and addressing ethical considerations and responsible usage of AGI in educational settings. The paper also discusses the implications of multimodal AI's role in education, offering insights into future directions and challenges in AGI development. This exploration aims to provide a nuanced understanding of the intersection between AI, multimodality, and education, setting a foundation for future research and development in AGI.

{{</citation>}}


### (21/75) Modeling Uncertainty in Personalized Emotion Prediction with Normalizing Flows (Piotr Miłkowski et al., 2023)

{{<citation>}}

Piotr Miłkowski, Konrad Karanowski, Patryk Wielopolski, Jan Kocoń, Przemysław Kazienko, Maciej Zięba. (2023)  
**Modeling Uncertainty in Personalized Emotion Prediction with Normalizing Flows**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2312.06034v1)  

---


**ABSTRACT**  
Designing predictive models for subjective problems in natural language processing (NLP) remains challenging. This is mainly due to its non-deterministic nature and different perceptions of the content by different humans. It may be solved by Personalized Natural Language Processing (PNLP), where the model exploits additional information about the reader to make more accurate predictions. However, current approaches require complete information about the recipients to be straight embedded. Besides, the recent methods focus on deterministic inference or simple frequency-based estimations of the probabilities. In this work, we overcome this limitation by proposing a novel approach to capture the uncertainty of the forecast using conditional Normalizing Flows. This allows us to model complex multimodal distributions and to compare various models using negative log-likelihood (NLL). In addition, the new solution allows for various interpretations of possible reader perception thanks to the available sampling function. We validated our method on three challenging, subjective NLP tasks, including emotion recognition and hate speech. The comparative analysis of generalized and personalized approaches revealed that our personalized solutions significantly outperform the baseline and provide more precise uncertainty estimates. The impact on the text interpretability and uncertainty studies are presented as well. The information brought by the developed methods makes it possible to build hybrid models whose effectiveness surpasses classic solutions. In addition, an analysis and visualization of the probabilities of the given decisions for texts with high entropy of annotations and annotators with mixed views were carried out.

{{</citation>}}


### (22/75) Evaluating the Utility of Model Explanations for Model Development (Shawn Im et al., 2023)

{{<citation>}}

Shawn Im, Jacob Andreas, Yilun Zhou. (2023)  
**Evaluating the Utility of Model Explanations for Model Development**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.06032v1)  

---


**ABSTRACT**  
One of the motivations for explainable AI is to allow humans to make better and more informed decisions regarding the use and deployment of AI models. But careful evaluations are needed to assess whether this expectation has been fulfilled. Current evaluations mainly focus on algorithmic properties of explanations, and those that involve human subjects often employ subjective questions to test human's perception of explanation usefulness, without being grounded in objective metrics and measurements. In this work, we evaluate whether explanations can improve human decision-making in practical scenarios of machine learning model development. We conduct a mixed-methods user study involving image data to evaluate saliency maps generated by SmoothGrad, GradCAM, and an oracle explanation on two tasks: model selection and counterfactual simulation. To our surprise, we did not find evidence of significant improvement on these tasks when users were provided with any of the saliency maps, even the synthetic oracle explanation designed to be simple to understand and highly indicative of the answer. Nonetheless, explanations did help users more accurately describe the models. These findings suggest caution regarding the usefulness and potential for misunderstanding in saliency-based explanations.

{{</citation>}}


### (23/75) Class-Aware Pruning for Efficient Neural Networks (Mengnan Jiang et al., 2023)

{{<citation>}}

Mengnan Jiang, Jingcun Wang, Amro Eldebiky, Xunzhao Yin, Cheng Zhuo, Ing-Chao Lin, Grace Li Zhang. (2023)  
**Class-Aware Pruning for Efficient Neural Networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2312.05875v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) have demonstrated remarkable success in various fields. However, the large number of floating-point operations (FLOPs) in DNNs poses challenges for their deployment in resource-constrained applications, e.g., edge devices. To address the problem, pruning has been introduced to reduce the computational cost in executing DNNs. Previous pruning strategies are based on weight values, gradient values and activation outputs. Different from previous pruning solutions, in this paper, we propose a class-aware pruning technique to compress DNNs, which provides a novel perspective to reduce the computational cost of DNNs. In each iteration, the neural network training is modified to facilitate the class-aware pruning. Afterwards, the importance of filters with respect to the number of classes is evaluated. The filters that are only important for a few number of classes are removed. The neural network is then retrained to compensate for the incurred accuracy loss. The pruning iterations end until no filter can be removed anymore, indicating that the remaining filters are very important for many classes. This pruning technique outperforms previous pruning solutions in terms of accuracy, pruning ratio and the reduction of FLOPs. Experimental results confirm that this class-aware pruning technique can significantly reduce the number of weights and FLOPs, while maintaining a high inference accuracy.

{{</citation>}}


### (24/75) Mutual Enhancement of Large and Small Language Models with Cross-Silo Knowledge Transfer (Yongheng Deng et al., 2023)

{{<citation>}}

Yongheng Deng, Ziqing Qiao, Ju Ren, Yang Liu, Yaoxue Zhang. (2023)  
**Mutual Enhancement of Large and Small Language Models with Cross-Silo Knowledge Transfer**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.05842v1)  

---


**ABSTRACT**  
While large language models (LLMs) are empowered with broad knowledge, their task-specific performance is often suboptimal. It necessitates fine-tuning LLMs with task-specific data, but such data may be inaccessible due to privacy concerns. In this paper, we propose a novel approach to enhance LLMs with smaller language models (SLMs) that are trained on clients using their private task-specific data. To enable mutual enhancement between LLMs and SLMs, we propose CrossLM, where the SLMs promote the LLM to generate task-specific high-quality data, and both the LLM and SLMs are enhanced with the generated data. We evaluate CrossLM using publicly accessible language models across a range of benchmark tasks. The results demonstrate that CrossLM significantly enhances the task-specific performance of SLMs on clients and the LLM on the cloud server simultaneously while preserving the LLM's generalization capability.

{{</citation>}}


### (25/75) Toward Open-ended Embodied Tasks Solving (William Wei Wang et al., 2023)

{{<citation>}}

William Wei Wang, Dongqi Han, Xufang Luo, Yifei Shen, Charles Ling, Boyu Wang, Dongsheng Li. (2023)  
**Toward Open-ended Embodied Tasks Solving**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.05822v1)  

---


**ABSTRACT**  
Empowering embodied agents, such as robots, with Artificial Intelligence (AI) has become increasingly important in recent years. A major challenge is task open-endedness. In practice, robots often need to perform tasks with novel goals that are multifaceted, dynamic, lack a definitive "end-state", and were not encountered during training. To tackle this problem, this paper introduces \textit{Diffusion for Open-ended Goals} (DOG), a novel framework designed to enable embodied AI to plan and act flexibly and dynamically for open-ended task goals. DOG synergizes the generative prowess of diffusion models with state-of-the-art, training-free guidance techniques to adaptively perform online planning and control. Our evaluations demonstrate that DOG can handle various kinds of novel task goals not seen during training, in both maze navigation and robot control problems. Our work sheds light on enhancing embodied AI's adaptability and competency in tackling open-ended goals.

{{</citation>}}


### (26/75) Neural Speech Embeddings for Speech Synthesis Based on Deep Generative Networks (Seo-Hyun Lee et al., 2023)

{{<citation>}}

Seo-Hyun Lee, Young-Eun Lee, Soowon Kim, Byung-Kwan Ko, Jun-Young Kim, Seong-Whan Lee. (2023)  
**Neural Speech Embeddings for Speech Synthesis Based on Deep Generative Networks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SD, cs.AI, eess-AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2312.05814v1)  

---


**ABSTRACT**  
Brain-to-speech technology represents a fusion of interdisciplinary applications encompassing fields of artificial intelligence, brain-computer interfaces, and speech synthesis. Neural representation learning based intention decoding and speech synthesis directly connects the neural activity to the means of human linguistic communication, which may greatly enhance the naturalness of communication. With the current discoveries on representation learning and the development of the speech synthesis technologies, direct translation of brain signals into speech has shown great promise. Especially, the processed input features and neural speech embeddings which are given to the neural network play a significant role in the overall performance when using deep generative models for speech generation from brain signals. In this paper, we introduce the current brain-to-speech technology with the possibility of speech synthesis from brain signals, which may ultimately facilitate innovation in non-verbal communication. Also, we perform comprehensive analysis on the neural features and neural speech embeddings underlying the neurophysiological activation while performing speech, which may play a significant role in the speech synthesis works.

{{</citation>}}


### (27/75) Large Multimodal Model Compression via Efficient Pruning and Distillation at AntGroup (Maolin Wang et al., 2023)

{{<citation>}}

Maolin Wang, Yao Zhao, Jiajia Liu, Jingdong Chen, Chenyi Zhuang, Jinjie Gu, Ruocheng Guo, Xiangyu Zhao. (2023)  
**Large Multimodal Model Compression via Efficient Pruning and Distillation at AntGroup**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Pruning  
[Paper Link](http://arxiv.org/abs/2312.05795v1)  

---


**ABSTRACT**  
The deployment of Large Multimodal Models (LMMs) within AntGroup has significantly advanced multimodal tasks in payment, security, and advertising, notably enhancing advertisement audition tasks in Alipay. However, the deployment of such sizable models introduces challenges, particularly in increased latency and carbon emissions, which are antithetical to the ideals of Green AI. This paper introduces a novel multi-stage compression strategy for our proprietary LLM, AntGMM. Our methodology pivots on three main aspects: employing small training sample sizes, addressing multi-level redundancy through multi-stage pruning, and introducing an advanced distillation loss design. In our research, we constructed a dataset, the Multimodal Advertisement Audition Dataset (MAAD), from real-world scenarios within Alipay, and conducted experiments to validate the reliability of our proposed strategy. Furthermore, the effectiveness of our strategy is evident in its operational success in Alipay's real-world multimodal advertisement audition for three months from September 2023. Notably, our approach achieved a substantial reduction in latency, decreasing it from 700ms to 90ms, while maintaining online performance with only a slight performance decrease. Moreover, our compressed model is estimated to reduce electricity consumption by approximately 75 million kWh annually compared to the direct deployment of AntGMM, demonstrating our commitment to green AI initiatives. We will publicly release our code and the MAAD dataset after some reviews\footnote{https://github.com/MorinW/AntGMM$\_$Pruning}.

{{</citation>}}


### (28/75) Graph-based Prediction and Planning Policy Network (GP3Net) for scalable self-driving in dynamic environments using Deep Reinforcement Learning (Jayabrata Chowdhury et al., 2023)

{{<citation>}}

Jayabrata Chowdhury, Venkataramanan Shivaraman, Suresh Sundaram, P B Sujit. (2023)  
**Graph-based Prediction and Planning Policy Network (GP3Net) for scalable self-driving in dynamic environments using Deep Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-RO, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05784v1)  

---


**ABSTRACT**  
Recent advancements in motion planning for Autonomous Vehicles (AVs) show great promise in using expert driver behaviors in non-stationary driving environments. However, learning only through expert drivers needs more generalizability to recover from domain shifts and near-failure scenarios due to the dynamic behavior of traffic participants and weather conditions. A deep Graph-based Prediction and Planning Policy Network (GP3Net) framework is proposed for non-stationary environments that encodes the interactions between traffic participants with contextual information and provides a decision for safe maneuver for AV. A spatio-temporal graph models the interactions between traffic participants for predicting the future trajectories of those participants. The predicted trajectories are utilized to generate a future occupancy map around the AV with uncertainties embedded to anticipate the evolving non-stationary driving environments. Then the contextual information and future occupancy maps are input to the policy network of the GP3Net framework and trained using Proximal Policy Optimization (PPO) algorithm. The proposed GP3Net performance is evaluated on standard CARLA benchmarking scenarios with domain shifts of traffic patterns (urban, highway, and mixed). The results show that the GP3Net outperforms previous state-of-the-art imitation learning-based planning models for different towns. Further, in unseen new weather conditions, GP3Net completes the desired route with fewer traffic infractions. Finally, the results emphasize the advantage of including the prediction module to enhance safety measures in non-stationary environments.

{{</citation>}}


### (29/75) A Comprehensive Survey on Multi-modal Conversational Emotion Recognition with Deep Learning (Yuntao Shou et al., 2023)

{{<citation>}}

Yuntao Shou, Tao Meng, Wei Ai, Nan Yin, Keqin Li. (2023)  
**A Comprehensive Survey on Multi-modal Conversational Emotion Recognition with Deep Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2312.05735v1)  

---


**ABSTRACT**  
Multi-modal conversation emotion recognition (MCER) aims to recognize and track the speaker's emotional state using text, speech, and visual information in the conversation scene. Analyzing and studying MCER issues is significant to affective computing, intelligent recommendations, and human-computer interaction fields. Unlike the traditional single-utterance multi-modal emotion recognition or single-modal conversation emotion recognition, MCER is a more challenging problem that needs to deal with more complex emotional interaction relationships. The critical issue is learning consistency and complementary semantics for multi-modal feature fusion based on emotional interaction relationships. To solve this problem, people have conducted extensive research on MCER based on deep learning technology, but there is still a lack of systematic review of the modeling methods. Therefore, a timely and comprehensive overview of MCER's recent advances in deep learning is of great significance to academia and industry. In this survey, we provide a comprehensive overview of MCER modeling methods and roughly divide MCER methods into four categories, i.e., context-free modeling, sequential context modeling, speaker-differentiated modeling, and speaker-relationship modeling. In addition, we further discuss MCER's publicly available popular datasets, multi-modal feature extraction methods, application areas, existing challenges, and future development directions. We hope that our review can help MCER researchers understand the current research status in emotion recognition, provide some inspiration, and develop more efficient models.

{{</citation>}}


### (30/75) FP8-BERT: Post-Training Quantization for Transformer (Jianwei Li et al., 2023)

{{<citation>}}

Jianwei Li, Tianchi Zhang, Ian En-Hsu Yen, Dongkuan Xu. (2023)  
**FP8-BERT: Post-Training Quantization for Transformer**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, BERT, GLUE, QA, Quantization, Transformer  
[Paper Link](http://arxiv.org/abs/2312.05725v2)  

---


**ABSTRACT**  
Transformer-based models, such as BERT, have been widely applied in a wide range of natural language processing tasks. However, one inevitable side effect is that they require massive memory storage and inference cost when deployed in production. Quantization is one of the popularized ways to alleviate the cost. However, the previous 8-bit quantization strategy based on INT8 data format either suffers from the degradation of accuracy in a Post-Training Quantization (PTQ) fashion or requires an expensive Quantization-Aware Training (QAT) process. Recently, a new numeric format FP8 (i.e. floating-point of 8-bits) has been proposed and supported in commercial AI computing platforms such as H100. In this paper, we empirically validate the effectiveness of FP8 as a way to do Post-Training Quantization without significant loss of accuracy, with a simple calibration and format conversion process. We adopt the FP8 standard proposed by NVIDIA Corp. (2022) in our extensive experiments of BERT variants on GLUE and SQuAD v1.1 datasets, and show that PTQ with FP8 can significantly improve the accuracy upon that with INT8, to the extent of the full-precision model.

{{</citation>}}


### (31/75) Singular Value Penalization and Semantic Data Augmentation for Fully Test-Time Adaptation (Houcheng Su et al., 2023)

{{<citation>}}

Houcheng Su, Daixian Liu, Mengzhu Wang, Wei Wang. (2023)  
**Singular Value Penalization and Semantic Data Augmentation for Fully Test-Time Adaptation**  

---
Primary Category: cs.AI  
Categories: 68A65, I-m, cs-AI, cs-CV, cs.AI  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.08378v1)  

---


**ABSTRACT**  
Fully test-time adaptation (FTTA) adapts a model that is trained on a source domain to a target domain during the testing phase, where the two domains follow different distributions and source data is unavailable during the training phase. Existing methods usually adopt entropy minimization to reduce the uncertainty of target prediction results, and improve the FTTA performance accordingly. However, they fail to ensure the diversity in target prediction results. Recent domain adaptation study has shown that maximizing the sum of singular values of prediction results can simultaneously enhance their confidence (discriminability) and diversity. However, during the training phase, larger singular values usually take up a dominant position in loss maximization. This results in the model being more inclined to enhance discriminability for easily distinguishable classes, and the improvement in diversity is insufficiently effective. Furthermore, the adaptation and prediction in FTTA only use data from the current batch, which may lead to the risk of overfitting. To address the aforementioned issues, we propose maximizing the sum of singular values while minimizing their variance. This enables the model's focus toward the smaller singular values, enhancing discriminability between more challenging classes and effectively increasing the diversity of prediction results. Moreover, we incorporate data from the previous batch to realize semantic data augmentation for the current batch, reducing the risk of overfitting. Extensive experiments on benchmark datasets show our proposed approach outperforms some compared state-of-the-art FTTA methods.

{{</citation>}}


## cs.LG (16)



### (32/75) AI Competitions and Benchmarks: towards impactful challenges with post-challenge papers, benchmarks and other dissemination actions (Antoine Marot et al., 2023)

{{<citation>}}

Antoine Marot, David Rousseau, Zhen Xu. (2023)  
**AI Competitions and Benchmarks: towards impactful challenges with post-challenge papers, benchmarks and other dissemination actions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.06036v3)  

---


**ABSTRACT**  
Organising an AI challenge does not end with the final event. The long-lasting impact also needs to be organised. This chapter covers the various activities after the challenge is formally finished. The target audience of different post-challenge activities is identified. The various outputs of the challenge are listed with the means to collect them. The main part of the chapter is a template for a typical post-challenge paper, including possible graphs as well as advice on how to turn the challenge into a long-lasting benchmark.

{{</citation>}}


### (33/75) Fast Classification of Large Time Series Datasets (Muhammad Marwan Muhammad Fuad, 2023)

{{<citation>}}

Muhammad Marwan Muhammad Fuad. (2023)  
**Fast Classification of Large Time Series Datasets**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2312.06029v1)  

---


**ABSTRACT**  
Time series classification (TSC) is the most import task in time series mining as it has several applications in medicine, meteorology, finance cyber security, and many others. With the ever increasing size of time series datasets, several traditional TSC methods are no longer efficient enough to perform this task on such very large datasets. Yet, most recent papers on TSC focus mainly on accuracy by using methods that apply deep learning, for instance, which require extensive computational resources that cannot be applied efficiently to very large datasets. The method we introduce in this paper focuses on these very large time series datasets with the main objective being efficiency. We achieve this through a simplified representation of the time series. This in turn is enhanced by a distance measure that considers only some of the values of the represented time series. The result of this combination is a very efficient representation method for TSC. This has been tested experimentally against another time series method that is particularly popular for its efficiency. The experiments show that our method is not only 4 times faster, on average, but it is also superior in terms of classification accuracy, as it gives better results on 24 out of the 29 tested time series datasets. .

{{</citation>}}


### (34/75) TransGlow: Attention-augmented Transduction model based on Graph Neural Networks for Water Flow Forecasting (Naghmeh Shafiee Roudbari et al., 2023)

{{<citation>}}

Naghmeh Shafiee Roudbari, Charalambos Poullis, Zachary Patterson, Ursula Eicker. (2023)  
**TransGlow: Attention-augmented Transduction model based on Graph Neural Networks for Water Flow Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.05961v1)  

---


**ABSTRACT**  
The hydrometric prediction of water quantity is useful for a variety of applications, including water management, flood forecasting, and flood control. However, the task is difficult due to the dynamic nature and limited data of water systems. Highly interconnected water systems can significantly affect hydrometric forecasting. Consequently, it is crucial to develop models that represent the relationships between other system components. In recent years, numerous hydrological applications have been studied, including streamflow prediction, flood forecasting, and water quality prediction. Existing methods are unable to model the influence of adjacent regions between pairs of variables. In this paper, we propose a spatiotemporal forecasting model that augments the hidden state in Graph Convolution Recurrent Neural Network (GCRN) encoder-decoder using an efficient version of the attention mechanism. The attention layer allows the decoder to access different parts of the input sequence selectively. Since water systems are interconnected and the connectivity information between the stations is implicit, the proposed model leverages a graph learning module to extract a sparse graph adjacency matrix adaptively based on the data. Spatiotemporal forecasting relies on historical data. In some regions, however, historical data may be limited or incomplete, making it difficult to accurately predict future water conditions. Further, we present a new benchmark dataset of water flow from a network of Canadian stations on rivers, streams, and lakes. Experimental results demonstrate that our proposed model TransGlow significantly outperforms baseline methods by a wide margin.

{{</citation>}}


### (35/75) VAE-IF: Deep feature extraction with averaging for unsupervised artifact detection in routine acquired ICU time-series (Hollan Haule et al., 2023)

{{<citation>}}

Hollan Haule, Ian Piper, Patricia Jones, Chen Qin, Tsz-Yan Milly Lo, Javier Escudero. (2023)  
**VAE-IF: Deep feature extraction with averaging for unsupervised artifact detection in routine acquired ICU time-series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2312.05959v1)  

---


**ABSTRACT**  
Artifacts are a common problem in physiological time-series data collected from intensive care units (ICU) and other settings. They affect the quality and reliability of clinical research and patient care. Manual annotation of artifacts is costly and time-consuming, rendering it impractical. Automated methods are desired. Here, we propose a novel unsupervised approach to detect artifacts in clinical-standard minute-by-minute resolution ICU data without any prior labeling or signal-specific knowledge. Our approach combines a variational autoencoder (VAE) and an isolation forest (iForest) model to learn features and identify anomalies in different types of vital signs, such as blood pressure, heart rate, and intracranial pressure. We evaluate our approach on a real-world ICU dataset and compare it with supervised models based on long short-term memory (LSTM) and XGBoost. We show that our approach achieves comparable sensitivity and generalizes well to an external dataset. We also visualize the latent space learned by the VAE and demonstrate its ability to disentangle clean and noisy samples. Our approach offers a promising solution for cleaning ICU data in clinical research and practice without the need for any labels whatsoever.

{{</citation>}}


### (36/75) Class-Prototype Conditional Diffusion Model for Continual Learning with Generative Replay (Khanh Doan et al., 2023)

{{<citation>}}

Khanh Doan, Quyen Tran, Tuan Nguyen, Dinh Phung, Trung Le. (2023)  
**Class-Prototype Conditional Diffusion Model for Continual Learning with Generative Replay**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.06710v1)  

---


**ABSTRACT**  
Mitigating catastrophic forgetting is a key hurdle in continual learning. Deep Generative Replay (GR) provides techniques focused on generating samples from prior tasks to enhance the model's memory capabilities. With the progression in generative AI, generative models have advanced from Generative Adversarial Networks (GANs) to the more recent Diffusion Models (DMs). A major issue is the deterioration in the quality of generated data compared to the original, as the generator continuously self-learns from its outputs. This degradation can lead to the potential risk of catastrophic forgetting occurring in the classifier. To address this, we propose the Class-Prototype Conditional Diffusion Model (CPDM), a GR-based approach for continual learning that enhances image quality in generators and thus reduces catastrophic forgetting in classifiers. The cornerstone of CPDM is a learnable class-prototype that captures the core characteristics of images in a given class. This prototype, integrated into the diffusion model's denoising process, ensures the generation of high-quality images. It maintains its effectiveness for old tasks even when new tasks are introduced, preserving image generation quality and reducing the risk of catastrophic forgetting in classifiers. Our empirical studies on diverse datasets demonstrate that our proposed method significantly outperforms existing state-of-the-art models, highlighting its exceptional ability to preserve image quality and enhance the model's memory retention.

{{</citation>}}


### (37/75) Temporal Supervised Contrastive Learning for Modeling Patient Risk Progression (Shahriar Noroozizadeh et al., 2023)

{{<citation>}}

Shahriar Noroozizadeh, Jeremy C. Weiss, George H. Chen. (2023)  
**Temporal Supervised Contrastive Learning for Modeling Patient Risk Progression**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2312.05933v1)  

---


**ABSTRACT**  
We consider the problem of predicting how the likelihood of an outcome of interest for a patient changes over time as we observe more of the patient data. To solve this problem, we propose a supervised contrastive learning framework that learns an embedding representation for each time step of a patient time series. Our framework learns the embedding space to have the following properties: (1) nearby points in the embedding space have similar predicted class probabilities, (2) adjacent time steps of the same time series map to nearby points in the embedding space, and (3) time steps with very different raw feature vectors map to far apart regions of the embedding space. To achieve property (3), we employ a nearest neighbor pairing mechanism in the raw feature space. This mechanism also serves as an alternative to data augmentation, a key ingredient of contrastive learning, which lacks a standard procedure that is adequately realistic for clinical tabular data, to our knowledge. We demonstrate that our approach outperforms state-of-the-art baselines in predicting mortality of septic patients (MIMIC-III dataset) and tracking progression of cognitive impairment (ADNI dataset). Our method also consistently recovers the correct synthetic dataset embedding structure across experiments, a feat not achieved by baselines. Our ablation experiments show the pivotal role of our nearest neighbor pairing.

{{</citation>}}


### (38/75) Improving Subgraph-GNNs via Edge-Level Ego-Network Encodings (Nurudin Alvarez-Gonzalez et al., 2023)

{{<citation>}}

Nurudin Alvarez-Gonzalez, Andreas Kaltenbrunner, Vicenç Gómez. (2023)  
**Improving Subgraph-GNNs via Edge-Level Ego-Network Encodings**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2312.05905v1)  

---


**ABSTRACT**  
We present a novel edge-level ego-network encoding for learning on graphs that can boost Message Passing Graph Neural Networks (MP-GNNs) by providing additional node and edge features or extending message-passing formats. The proposed encoding is sufficient to distinguish Strongly Regular Graphs, a family of challenging 3-WL equivalent graphs. We show theoretically that such encoding is more expressive than node-based sub-graph MP-GNNs. In an empirical evaluation on four benchmarks with 10 graph datasets, our results match or improve previous baselines on expressivity, graph classification, graph regression, and proximity tasks -- while reducing memory usage by 18.1x in certain real-world settings.

{{</citation>}}


### (39/75) Take an Irregular Route: Enhance the Decoder of Time-Series Forecasting Transformer (Li Shen et al., 2023)

{{<citation>}}

Li Shen, Yuning Wei, Yangzhu Wang, Hongguang Li. (2023)  
**Take an Irregular Route: Enhance the Decoder of Time-Series Forecasting Transformer**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2312.05792v1)  

---


**ABSTRACT**  
With the development of Internet of Things (IoT) systems, precise long-term forecasting method is requisite for decision makers to evaluate current statuses and formulate future policies. Currently, Transformer and MLP are two paradigms for deep time-series forecasting and the former one is more prevailing in virtue of its exquisite attention mechanism and encoder-decoder architecture. However, data scientists seem to be more willing to dive into the research of encoder, leaving decoder unconcerned. Some researchers even adopt linear projections in lieu of the decoder to reduce the complexity. We argue that both extracting the features of input sequence and seeking the relations of input and prediction sequence, which are respective functions of encoder and decoder, are of paramount significance. Motivated from the success of FPN in CV field, we propose FPPformer to utilize bottom-up and top-down architectures respectively in encoder and decoder to build the full and rational hierarchy. The cutting-edge patch-wise attention is exploited and further developed with the combination, whose format is also different in encoder and decoder, of revamped element-wise attention in this work. Extensive experiments with six state-of-the-art baselines on twelve benchmarks verify the promising performances of FPPformer and the importance of elaborately devising decoder in time-series forecasting Transformer. The source code is released in https://github.com/OrigamiSL/FPPformer.

{{</citation>}}


### (40/75) SimPSI: A Simple Strategy to Preserve Spectral Information in Time Series Data Augmentation (Hyun Ryu et al., 2023)

{{<citation>}}

Hyun Ryu, Sunjae Yoon, Hee Suk Yoon, Eunseop Yoon, Chang D. Yoo. (2023)  
**SimPSI: A Simple Strategy to Preserve Spectral Information in Time Series Data Augmentation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-SP  
Keywords: Augmentation, Time Series  
[Paper Link](http://arxiv.org/abs/2312.05790v1)  

---


**ABSTRACT**  
Data augmentation is a crucial component in training neural networks to overcome the limitation imposed by data size, and several techniques have been studied for time series. Although these techniques are effective in certain tasks, they have yet to be generalized to time series benchmarks. We find that current data augmentation techniques ruin the core information contained within the frequency domain. To address this issue, we propose a simple strategy to preserve spectral information (SimPSI) in time series data augmentation. SimPSI preserves the spectral information by mixing the original and augmented input spectrum weighted by a preservation map, which indicates the importance score of each frequency. Specifically, our experimental contributions are to build three distinct preservation maps: magnitude spectrum, saliency map, and spectrum-preservative map. We apply SimPSI to various time series data augmentations and evaluate its effectiveness across a wide range of time series benchmarks. Our experimental results support that SimPSI considerably enhances the performance of time series data augmentations by preserving core spectral information. The source code used in the paper is available at https://github.com/Hyun-Ryu/simpsi.

{{</citation>}}


### (41/75) Efficient Sparse-Reward Goal-Conditioned Reinforcement Learning with a High Replay Ratio and Regularization (Takuya Hiraoka, 2023)

{{<citation>}}

Takuya Hiraoka. (2023)  
**Efficient Sparse-Reward Goal-Conditioned Reinforcement Learning with a High Replay Ratio and Regularization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05787v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) methods with a high replay ratio (RR) and regularization have gained interest due to their superior sample efficiency. However, these methods have mainly been developed for dense-reward tasks. In this paper, we aim to extend these RL methods to sparse-reward goal-conditioned tasks. We use Randomized Ensemble Double Q-learning (REDQ) (Chen et al., 2021), an RL method with a high RR and regularization. To apply REDQ to sparse-reward goal-conditioned tasks, we make the following modifications to it: (i) using hindsight experience replay and (ii) bounding target Q-values. We evaluate REDQ with these modifications on 12 sparse-reward goal-conditioned tasks of Robotics (Plappert et al., 2018), and show that it achieves about $2 \times$ better sample efficiency than previous state-of-the-art (SoTA) RL methods. Furthermore, we reconsider the necessity of specific components of REDQ and simplify it by removing unnecessary ones. The simplified REDQ with our modifications achieves $\sim 8 \times$ better sample efficiency than the SoTA methods in 4 Fetch tasks of Robotics.

{{</citation>}}


### (42/75) DCIR: Dynamic Consistency Intrinsic Reward for Multi-Agent Reinforcement Learning (Kunyang Lin et al., 2023)

{{<citation>}}

Kunyang Lin, Yufeng Wang, Peihao Chen, Runhao Zeng, Siyuan Zhou, Mingkui Tan, Chuang Gan. (2023)  
**DCIR: Dynamic Consistency Intrinsic Reward for Multi-Agent Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Google, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05783v1)  

---


**ABSTRACT**  
Learning optimal behavior policy for each agent in multi-agent systems is an essential yet difficult problem. Despite fruitful progress in multi-agent reinforcement learning, the challenge of addressing the dynamics of whether two agents should exhibit consistent behaviors is still under-explored. In this paper, we propose a new approach that enables agents to learn whether their behaviors should be consistent with that of other agents by utilizing intrinsic rewards to learn the optimal policy for each agent. We begin by defining behavior consistency as the divergence in output actions between two agents when provided with the same observation. Subsequently, we introduce dynamic consistency intrinsic reward (DCIR) to stimulate agents to be aware of others' behaviors and determine whether to be consistent with them. Lastly, we devise a dynamic scale network (DSN) that provides learnable scale factors for the agent at every time step to dynamically ascertain whether to award consistent behavior and the magnitude of rewards. We evaluate DCIR in multiple environments including Multi-agent Particle, Google Research Football and StarCraft II Micromanagement, demonstrating its efficacy.

{{</citation>}}


### (43/75) QMGeo: Differentially Private Federated Learning via Stochastic Quantization with Mixed Truncated Geometric Distribution (Zixi Wang et al., 2023)

{{<citation>}}

Zixi Wang, M. Cenk Gursoy. (2023)  
**QMGeo: Differentially Private Federated Learning via Stochastic Quantization with Mixed Truncated Geometric Distribution**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2312.05761v1)  

---


**ABSTRACT**  
Federated learning (FL) is a framework which allows multiple users to jointly train a global machine learning (ML) model by transmitting only model updates under the coordination of a parameter server, while being able to keep their datasets local. One key motivation of such distributed frameworks is to provide privacy guarantees to the users. However, preserving the users' datasets locally is shown to be not sufficient for privacy. Several differential privacy (DP) mechanisms have been proposed to provide provable privacy guarantees by introducing randomness into the framework, and majority of these mechanisms rely on injecting additive noise. FL frameworks also face the challenge of communication efficiency, especially as machine learning models grow in complexity and size. Quantization is a commonly utilized method, reducing the communication cost by transmitting compressed representation of the underlying information. Although there have been several studies on DP and quantization in FL, the potential contribution of the quantization method alone in providing privacy guarantees has not been extensively analyzed yet. We in this paper present a novel stochastic quantization method, utilizing a mixed geometric distribution to introduce the randomness needed to provide DP, without any additive noise. We provide convergence analysis for our framework and empirically study its performance.

{{</citation>}}


### (44/75) CLeaRForecast: Contrastive Learning of High-Purity Representations for Time Series Forecasting (Jiaxin Gao et al., 2023)

{{<citation>}}

Jiaxin Gao, Yuxiao Hu, Qinglong Cao, Siqi Dai, Yuntian Chen. (2023)  
**CLeaRForecast: Contrastive Learning of High-Purity Representations for Time Series Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-AP  
Keywords: Contrastive Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2312.05758v1)  

---


**ABSTRACT**  
Time series forecasting (TSF) holds significant importance in modern society, spanning numerous domains. Previous representation learning-based TSF algorithms typically embrace a contrastive learning paradigm featuring segregated trend-periodicity representations. Yet, these methodologies disregard the inherent high-impact noise embedded within time series data, resulting in representation inaccuracies and seriously demoting the forecasting performance. To address this issue, we propose CLeaRForecast, a novel contrastive learning framework to learn high-purity time series representations with proposed sample, feature, and architecture purifying methods. More specifically, to avoid more noise adding caused by the transformations of original samples (series), transformations are respectively applied for trendy and periodic parts to provide better positive samples with obviously less noise. Moreover, we introduce a channel independent training manner to mitigate noise originating from unrelated variables in the multivariate series. By employing a streamlined deep-learning backbone and a comprehensive global contrastive loss function, we prevent noise introduction due to redundant or uneven learning of periodicity and trend. Experimental results show the superior performance of CLeaRForecast in various downstream TSF tasks.

{{</citation>}}


### (45/75) The Generalization Gap in Offline Reinforcement Learning (Ishita Mediratta et al., 2023)

{{<citation>}}

Ishita Mediratta, Qingfei You, Minqi Jiang, Roberta Raileanu. (2023)  
**The Generalization Gap in Offline Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05742v1)  

---


**ABSTRACT**  
Despite recent progress in offline learning, these methods are still trained and tested on the same environment. In this paper, we compare the generalization abilities of widely used online and offline learning methods such as online reinforcement learning (RL), offline RL, sequence modeling, and behavioral cloning. Our experiments show that offline learning algorithms perform worse on new environments than online learning ones. We also introduce the first benchmark for evaluating generalization in offline learning, collecting datasets of varying sizes and skill-levels from Procgen (2D video games) and WebShop (e-commerce websites). The datasets contain trajectories for a limited number of game levels or natural language instructions and at test time, the agent has to generalize to new levels or instructions. Our experiments reveal that existing offline learning algorithms struggle to match the performance of online RL on both train and test environments. Behavioral cloning is a strong baseline, outperforming state-of-the-art offline RL and sequence modeling approaches when trained on data from multiple environments and tested on new ones. Finally, we find that increasing the diversity of the data, rather than its size, improves performance on new environments for all offline learning algorithms. Our study demonstrates the limited generalization of current offline learning algorithms highlighting the need for more research in this area.

{{</citation>}}


### (46/75) ASWT-SGNN: Adaptive Spectral Wavelet Transform-based Self-Supervised Graph Neural Network (Ruyue Liu et al., 2023)

{{<citation>}}

Ruyue Liu, Rong Yin, Yong Liu, Weiping Wang. (2023)  
**ASWT-SGNN: Adaptive Spectral Wavelet Transform-based Self-Supervised Graph Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: GNN, Graph Convolutional Network, Graph Neural Network, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2312.05736v1)  

---


**ABSTRACT**  
Graph Comparative Learning (GCL) is a self-supervised method that combines the advantages of Graph Convolutional Networks (GCNs) and comparative learning, making it promising for learning node representations. However, the GCN encoders used in these methods rely on the Fourier transform to learn fixed graph representations, which is inherently limited by the uncertainty principle involving spatial and spectral localization trade-offs. To overcome the inflexibility of existing methods and the computationally expensive eigen-decomposition and dense matrix multiplication, this paper proposes an Adaptive Spectral Wavelet Transform-based Self-Supervised Graph Neural Network (ASWT-SGNN). The proposed method employs spectral adaptive polynomials to approximate the filter function and optimize the wavelet using contrast loss. This design enables the creation of local filters in both spectral and spatial domains, allowing flexible aggregation of neighborhood information at various scales and facilitating controlled transformation between local and global information. Compared to existing methods, the proposed approach reduces computational complexity and addresses the limitation of graph convolutional neural networks, which are constrained by graph size and lack flexible control over the neighborhood aspect. Extensive experiments on eight benchmark datasets demonstrate that ASWT-SGNN accurately approximates the filter function in high-density spectral regions, avoiding costly eigen-decomposition. Furthermore, ASWT-SGNN achieves comparable performance to state-of-the-art models in node classification tasks.

{{</citation>}}


### (47/75) Beyond Gradient and Priors in Privacy Attacks: Leveraging Pooler Layer Inputs of Language Models in Federated Learning (Jianwei Li et al., 2023)

{{<citation>}}

Jianwei Li, Sheng Liu, Qi Lei. (2023)  
**Beyond Gradient and Priors in Privacy Attacks: Leveraging Pooler Layer Inputs of Language Models in Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.05720v1)  

---


**ABSTRACT**  
Federated learning (FL) emphasizes decentralized training by storing data locally and sending only model updates, underlining user privacy. Recently, a line of works on privacy attacks impairs user privacy by extracting sensitive training text from language models in the context of FL. Yet, these attack techniques face distinct hurdles: some work chiefly with limited batch sizes (e.g., batch size of 1), and others are easily detectable. This paper introduces an innovative approach that is challenging to detect, significantly enhancing the recovery rate of text in various batch-size settings. Building on fundamental gradient matching and domain prior knowledge, we enhance the attack by recovering the input of the Pooler layer of language models, which enables us to provide additional supervised signals at the feature level. Unlike gradient data, these signals do not average across sentences and tokens, thereby offering more nuanced and effective insights. We benchmark our method using text classification tasks on datasets such as CoLA, SST-2, and Rotten Tomatoes. Across different batch sizes and models, our approach consistently outperforms previous state-of-the-art results.

{{</citation>}}


## eess.SP (1)



### (48/75) Stress Management Using Virtual Reality-Based Attention Training (Rojaina Mahmoud et al., 2023)

{{<citation>}}

Rojaina Mahmoud, Mona Mamdouh, Omneya Attallah, Ahmad Al-Kabbany. (2023)  
**Stress Management Using Virtual Reality-Based Attention Training**  

---
Primary Category: eess.SP  
Categories: cs-LG, eess-SP, eess.SP  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.06025v1)  

---


**ABSTRACT**  
In this research, we are concerned with the applicability of virtual reality-based attention training as a tool for stress management. Mental stress is a worldwide challenge that is still far from being fully managed. This has maintained a remarkable research attention on developing and validating tools for detecting and managing stress. Technology-based tools have been at the heart of these endeavors, including virtual reality (VR) technology. Nevertheless, the potential of VR lies, to a large part, in the nature of the content being consumed through such technology. In this study, we investigate the impact of a special type of content, namely, attention training, on the feasibility of using VR for stress management. On a group of fourteen undergraduate engineering students, we conducted a study in which the participants got exposed twice to a stress inducer while their EEG signals were being recorded. The first iteration involved VR-based attention training before starting the stress task while the second time did not. Using multiple features and various machine learning models, we show that VR-based attention training has consistently resulted in reducing the number of recognized stress instances in the recorded EEG signals. This research gives preliminary insights on adopting VR-based attention training for managing stress, and future studies are required to replicate the results in larger samples.

{{</citation>}}


## cs.HC (1)



### (49/75) Thinking Assistants: LLM-Based Conversational Assistants that Help Users Think By Asking rather than Answering (Soya Park et al., 2023)

{{<citation>}}

Soya Park, Chinmay Kulkarni. (2023)  
**Thinking Assistants: LLM-Based Conversational Assistants that Help Users Think By Asking rather than Answering**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.06024v1)  

---


**ABSTRACT**  
We introduce the concept of "thinking assistants", an approach that encourages users to engage in deep reflection and critical thinking through brainstorming and thought-provoking queries. We instantiate one such thinking assistant, Gradschool.chat, as a virtual assistant tailored to assist prospective graduate students. We posit that thinking assistants are particularly relevant to situations like applying to graduate school, a phase often characterized by the challenges of academic preparation and the development of a unique research identity. In such situations, students often lack direct mentorship from professors, or may feel hesitant to approach faculty with their queries, making thinking assistants particularly useful.   Leveraging a Large Language Model (LLM), Gradschool.chat is a demonstration system built as a thinking assistant for working with specific professors in the field of human-computer interaction (HCI). It was designed through training on information specific to these professors and a validation processes in collaboration with these academics. This technical report delineates the system's architecture and offers a preliminary analysis of our deployment study. Additionally, this report covers the spectrum of questions posed to our chatbots by users. The system recorded 223 conversations, with participants responding positively to approximately 65% of responses. Our findings indicate that users who discuss and brainstorm their research interests with Gradschool.chat engage more deeply, often interacting with the chatbot twice as long compared to those who only pose questions about professors.

{{</citation>}}


## cs.CL (8)



### (50/75) Exploiting Representation Bias for Data Distillation in Abstractive Text Summarization (Yash Kumar Atri et al., 2023)

{{<citation>}}

Yash Kumar Atri, Vikram Goyal, Tanmoy Chakraborty. (2023)  
**Exploiting Representation Bias for Data Distillation in Abstractive Text Summarization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Abstractive Text Summarization, BERT, Bias, QA, Rouge, Summarization, Text Summarization  
[Paper Link](http://arxiv.org/abs/2312.06022v1)  

---


**ABSTRACT**  
Abstractive text summarization is surging with the number of training samples to cater to the needs of the deep learning models. These models tend to exploit the training data representations to attain superior performance by improving the quantitative element of the resultant summary. However, increasing the size of the training set may not always be the ideal solution to maximize the performance, and therefore, a need to revisit the quality of training samples and the learning protocol of deep learning models is a must. In this paper, we aim to discretize the vector space of the abstractive text summarization models to understand the characteristics learned between the input embedding space and the models' encoder space. We show that deep models fail to capture the diversity of the input space. Further, the distribution of data points on the encoder space indicates that an unchecked increase in the training samples does not add value; rather, a tear-down of data samples is highly needed to make the models focus on variability and faithfulness. We employ clustering techniques to learn the diversity of a model's sample space and how data points are mapped from the embedding space to the encoder space and vice versa. Further, we devise a metric to filter out redundant data points to make the model more robust and less data hungry. We benchmark our proposed method using quantitative metrics, such as Rouge, and qualitative metrics, such as BERTScore, FEQA and Pyramid score. We also quantify the reasons that inhibit the models from learning the diversity from the varied input samples.

{{</citation>}}


### (51/75) Large Language Models on Lexical Semantic Change Detection: An Evaluation (Ruiyu Wang et al., 2023)

{{<citation>}}

Ruiyu Wang, Matthew Choi. (2023)  
**Large Language Models on Lexical Semantic Change Detection: An Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2312.06002v1)  

---


**ABSTRACT**  
Lexical Semantic Change Detection stands out as one of the few areas where Large Language Models (LLMs) have not been extensively involved. Traditional methods like PPMI, and SGNS remain prevalent in research, alongside newer BERT-based approaches. Despite the comprehensive coverage of various natural language processing domains by LLMs, there is a notable scarcity of literature concerning their application in this specific realm. In this work, we seek to bridge this gap by introducing LLMs into the domain of Lexical Semantic Change Detection. Our work presents novel prompting solutions and a comprehensive evaluation that spans all three generations of language models, contributing to the exploration of LLMs in this research area.

{{</citation>}}


### (52/75) NovaCOMET: Open Commonsense Foundation Models with Symbolic Knowledge Distillation (Peter West et al., 2023)

{{<citation>}}

Peter West, Ronan Le Bras, Taylor Sorensen, Bill Yuchen Lin, Liwei Jiang, Ximing Lu, Khyathi Chandu, Jack Hessel, Ashutosh Baheti, Chandra Bhagavatula, Yejin Choi. (2023)  
**NovaCOMET: Open Commonsense Foundation Models with Symbolic Knowledge Distillation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Knowledge Distillation, T5  
[Paper Link](http://arxiv.org/abs/2312.05979v1)  

---


**ABSTRACT**  
We present NovaCOMET, an open commonsense knowledge model, that combines the best aspects of knowledge and general task models. Compared to previous knowledge models, NovaCOMET allows open-format relations enabling direct application to reasoning tasks; compared to general task models like Flan-T5, it explicitly centers knowledge, enabling superior performance for commonsense reasoning.   NovaCOMET leverages the knowledge of opaque proprietary models to create an open knowledge pipeline. First, knowledge is symbolically distilled into NovATOMIC, a publicly-released discrete knowledge graph which can be audited, critiqued, and filtered. Next, we train NovaCOMET on NovATOMIC by fine-tuning an open-source pretrained model. NovaCOMET uses an open-format training objective, replacing the fixed relation sets of past knowledge models, enabling arbitrary structures within the data to serve as inputs or outputs.   The resulting generation model, optionally augmented with human annotation, matches or exceeds comparable open task models like Flan-T5 on a range of commonsense generation tasks. NovaCOMET serves as a counterexample to the contemporary focus on instruction tuning only, demonstrating a distinct advantage to explicitly modeling commonsense knowledge as well.

{{</citation>}}


### (53/75) Perceiving University Student's Opinions from Google App Reviews (Sakshi Ranjan et al., 2023)

{{<citation>}}

Sakshi Ranjan, Subhankar Mishra. (2023)  
**Perceiving University Student's Opinions from Google App Reviews**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Google, LSTM, NLP  
[Paper Link](http://arxiv.org/abs/2312.06705v1)  

---


**ABSTRACT**  
Google app market captures the school of thought of users from every corner of the globe via ratings and text reviews, in a multilinguistic arena. The potential information from the reviews cannot be extracted manually, due to its exponential growth. So, Sentiment analysis, by machine learning and deep learning algorithms employing NLP, explicitly uncovers and interprets the emotions. This study performs the sentiment classification of the app reviews and identifies the university student's behavior towards the app market via exploratory analysis. We applied machine learning algorithms using the TP, TF, and TF IDF text representation scheme and evaluated its performance on Bagging, an ensemble learning method. We used word embedding, Glove, on the deep learning paradigms. Our model was trained on Google app reviews and tested on Student's App Reviews(SAR). The various combinations of these algorithms were compared amongst each other using F score and accuracy and inferences were highlighted graphically. SVM, amongst other classifiers, gave fruitful accuracy(93.41%), F score(89%) on bigram and TF IDF scheme. Bagging enhanced the performance of LR and NB with accuracy of 87.88% and 86.69% and F score of 86% and 78% respectively. Overall, LSTM on Glove embedding recorded the highest accuracy(95.2%) and F score(88%).

{{</citation>}}


### (54/75) Evidence-based Interpretable Open-domain Fact-checking with Large Language Models (Xin Tan et al., 2023)

{{<citation>}}

Xin Tan, Bowei Zou, Ai Ti Aw. (2023)  
**Evidence-based Interpretable Open-domain Fact-checking with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.05834v1)  

---


**ABSTRACT**  
Universal fact-checking systems for real-world claims face significant challenges in gathering valid and sufficient real-time evidence and making reasoned decisions. In this work, we introduce the Open-domain Explainable Fact-checking (OE-Fact) system for claim-checking in real-world scenarios. The OE-Fact system can leverage the powerful understanding and reasoning capabilities of large language models (LLMs) to validate claims and generate causal explanations for fact-checking decisions. To adapt the traditional three-module fact-checking framework to the open domain setting, we first retrieve claim-related information as relevant evidence from open websites. After that, we retain the evidence relevant to the claim through LLM and similarity calculation for subsequent verification. We evaluate the performance of our adapted three-module OE-Fact system on the Fact Extraction and Verification (FEVER) dataset. Experimental results show that our OE-Fact system outperforms general fact-checking baseline systems in both closed- and open-domain scenarios, ensuring stable and accurate verdicts while providing concise and convincing real-time explanations for fact-checking decisions.

{{</citation>}}


### (55/75) ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models (Zhihang Yuan et al., 2023)

{{<citation>}}

Zhihang Yuan, Yuzhang Shang, Yue Song, Qiang Wu, Yan Yan, Guangyu Sun. (2023)  
**ASVD: Activation-aware Singular Value Decomposition for Compressing Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2312.05821v1)  

---


**ABSTRACT**  
This paper explores a new post-hoc training-free compression paradigm for compressing Large Language Models (LLMs) to facilitate their wider adoption in various computing environments. We delve into the challenges of LLM compression, notably their dependency on extensive training data and computational resources. We propose a training-free approach dubbed Activation-aware Singular Value Decomposition (ASVD) to address these limitations. ASVD effectively manages activation outliers by adjusting the weight matrix based on the activation distribution, improving decomposition accuracy and efficiency. Our method also addresses the varying sensitivity of different LLM layers to decomposition, with an iterative calibration process for optimal layer-specific decomposition. Experiments demonstrate that ASVD can compress network by 10%-20% without losing reasoning capacities. Additionally, it can be seamlessly integrated with other LLM compression paradigms, showcasing its flexible compatibility. Code and compressed models are available at https://github.com/hahnyuan/ASVD4LLM.

{{</citation>}}


### (56/75) Multi-Defendant Legal Judgment Prediction via Hierarchical Reasoning (Yougang Lyu et al., 2023)

{{<citation>}}

Yougang Lyu, Jitai Hao, Zihan Wang, Kai Zhao, Shen Gao, Pengjie Ren, Zhumin Chen, Fang Wang, Zhaochun Ren. (2023)  
**Multi-Defendant Legal Judgment Prediction via Hierarchical Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Legal, Reasoning  
[Paper Link](http://arxiv.org/abs/2312.05762v1)  

---


**ABSTRACT**  
Multiple defendants in a criminal fact description generally exhibit complex interactions, and cannot be well handled by existing Legal Judgment Prediction (LJP) methods which focus on predicting judgment results (e.g., law articles, charges, and terms of penalty) for single-defendant cases. To address this problem, we propose the task of multi-defendant LJP, which aims to automatically predict the judgment results for each defendant of multi-defendant cases. Two challenges arise with the task of multi-defendant LJP: (1) indistinguishable judgment results among various defendants; and (2) the lack of a real-world dataset for training and evaluation. To tackle the first challenge, we formalize the multi-defendant judgment process as hierarchical reasoning chains and introduce a multi-defendant LJP method, named Hierarchical Reasoning Network (HRN), which follows the hierarchical reasoning chains to determine criminal relationships, sentencing circumstances, law articles, charges, and terms of penalty for each defendant. To tackle the second challenge, we collect a real-world multi-defendant LJP dataset, namely MultiLJP, to accelerate the relevant research in the future. Extensive experiments on MultiLJP verify the effectiveness of our proposed HRN.

{{</citation>}}


### (57/75) MISCA: A Joint Model for Multiple Intent Detection and Slot Filling with Intent-Slot Co-Attention (Thinh Pham et al., 2023)

{{<citation>}}

Thinh Pham, Chi Tran, Dat Quoc Nguyen. (2023)  
**MISCA: A Joint Model for Multiple Intent Detection and Slot Filling with Intent-Slot Co-Attention**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Intent Detection  
[Paper Link](http://arxiv.org/abs/2312.05741v1)  

---


**ABSTRACT**  
The research study of detecting multiple intents and filling slots is becoming more popular because of its relevance to complicated real-world situations. Recent advanced approaches, which are joint models based on graphs, might still face two potential issues: (i) the uncertainty introduced by constructing graphs based on preliminary intents and slots, which may transfer intent-slot correlation information to incorrect label node destinations, and (ii) direct incorporation of multiple intent labels for each token w.r.t. token-level intent voting might potentially lead to incorrect slot predictions, thereby hurting the overall performance. To address these two issues, we propose a joint model named MISCA. Our MISCA introduces an intent-slot co-attention mechanism and an underlying layer of label attention mechanism. These mechanisms enable MISCA to effectively capture correlations between intents and slot labels, eliminating the need for graph construction. They also facilitate the transfer of correlation information in both directions: from intents to slots and from slots to intents, through multiple levels of label-specific representations, without relying on token-level intent information. Experimental results show that MISCA outperforms previous models, achieving new state-of-the-art overall accuracy performances on two benchmark datasets MixATIS and MixSNIPS. This highlights the effectiveness of our attention mechanisms.

{{</citation>}}


## cs.CR (3)



### (58/75) A Practical Survey on Emerging Threats from AI-driven Voice Attacks: How Vulnerable are Commercial Voice Control Systems? (Yuanda Wang et al., 2023)

{{<citation>}}

Yuanda Wang, Qiben Yan, Nikolay Ivanov, Xun Chen. (2023)  
**A Practical Survey on Emerging Threats from AI-driven Voice Attacks: How Vulnerable are Commercial Voice Control Systems?**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SD, cs.CR, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.06010v1)  

---


**ABSTRACT**  
The emergence of Artificial Intelligence (AI)-driven audio attacks has revealed new security vulnerabilities in voice control systems. While researchers have introduced a multitude of attack strategies targeting voice control systems (VCS), the continual advancements of VCS have diminished the impact of many such attacks. Recognizing this dynamic landscape, our study endeavors to comprehensively assess the resilience of commercial voice control systems against a spectrum of malicious audio attacks. Through extensive experimentation, we evaluate six prominent attack techniques across a collection of voice control interfaces and devices. Contrary to prevailing narratives, our results suggest that commercial voice control systems exhibit enhanced resistance to existing threats. Particularly, our research highlights the ineffectiveness of white-box attacks in black-box scenarios. Furthermore, the adversaries encounter substantial obstacles in obtaining precise gradient estimations during query-based interactions with commercial systems, such as Apple Siri and Samsung Bixby. Meanwhile, we find that current defense strategies are not completely immune to advanced attacks. Our findings contribute valuable insights for enhancing defense mechanisms in VCS. Through this survey, we aim to raise awareness within the academic community about the security concerns of VCS and advocate for continued research in this crucial area.

{{</citation>}}


### (59/75) Guardians of Trust: Navigating Data Security in AIOps through Vendor Partnerships (Subhadip Kumar, 2023)

{{<citation>}}

Subhadip Kumar. (2023)  
**Guardians of Trust: Navigating Data Security in AIOps through Vendor Partnerships**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2312.06008v1)  

---


**ABSTRACT**  
Artificial Intelligence for IT Operations (AIOps) is a rapidly growing field that applies artificial intelligence and machine learning to automate and optimize IT operations. AIOps vendors provide services that ingest end-to-end logs, traces, and metrics to offer a full stack observability of IT systems. However, these data sources may contain sensitive information such as internal IP addresses, hostnames, HTTP headers, SQLs, method/argument return values, URLs, personal identifiable information (PII), or confidential business data. Therefore, data security is a crucial concern when working with AIOps vendors. In this article, we will discuss the security features offered by different vendors and how we can adopt best practices to ensure data protection and privacy.

{{</citation>}}


### (60/75) A Representative Study on Human Detection of Artificially Generated Media Across Countries (Joel Frank et al., 2023)

{{<citation>}}

Joel Frank, Franziska Herbert, Jonas Ricker, Lea Schönherr, Thorsten Eisenhofer, Asja Fischer, Markus Dürmuth, Thorsten Holz. (2023)  
**A Representative Study on Human Detection of Artificially Generated Media Across Countries**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-CY, cs-LG, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.05976v1)  

---


**ABSTRACT**  
AI-generated media has become a threat to our digital society as we know it. These forgeries can be created automatically and on a large scale based on publicly available technology. Recognizing this challenge, academics and practitioners have proposed a multitude of automatic detection strategies to detect such artificial media. However, in contrast to these technical advances, the human perception of generated media has not been thoroughly studied yet.   In this paper, we aim at closing this research gap. We perform the first comprehensive survey into people's ability to detect generated media, spanning three countries (USA, Germany, and China) with 3,002 participants across audio, image, and text media. Our results indicate that state-of-the-art forgeries are almost indistinguishable from "real" media, with the majority of participants simply guessing when asked to rate them as human- or machine-generated. In addition, AI-generated media receive is voted more human like across all media types and all countries. To further understand which factors influence people's ability to detect generated media, we include personal variables, chosen based on a literature review in the domains of deepfake and fake news research. In a regression analysis, we found that generalized trust, cognitive reflection, and self-reported familiarity with deepfakes significantly influence participant's decision across all media categories.

{{</citation>}}


## cs.SD (1)



### (61/75) mir_ref: A Representation Evaluation Framework for Music Information Retrieval Tasks (Christos Plachouras et al., 2023)

{{<citation>}}

Christos Plachouras, Pablo Alonso-Jiménez, Dmitry Bogdanov. (2023)  
**mir_ref: A Representation Evaluation Framework for Music Information Retrieval Tasks**  

---
Primary Category: cs.SD  
Categories: cs-IR, cs-SD, cs.SD, eess-AS  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2312.05994v2)  

---


**ABSTRACT**  
Music Information Retrieval (MIR) research is increasingly leveraging representation learning to obtain more compact, powerful music audio representations for various downstream MIR tasks. However, current representation evaluation methods are fragmented due to discrepancies in audio and label preprocessing, downstream model and metric implementations, data availability, and computational resources, often leading to inconsistent and limited results. In this work, we introduce mir_ref, an MIR Representation Evaluation Framework focused on seamless, transparent, local-first experiment orchestration to support representation development. It features implementations of a variety of components such as MIR datasets, tasks, embedding models, and tools for result analysis and visualization, while facilitating the implementation of custom components. To demonstrate its utility, we use it to conduct an extensive evaluation of several embedding models across various tasks and datasets, including evaluating their robustness to various audio perturbations and the ease of extracting relevant information from them.

{{</citation>}}


## cs.RO (4)



### (62/75) Modifying RL Policies with Imagined Actions: How Predictable Policies Can Enable Users to Perform Novel Tasks (Isaac Sheidlower et al., 2023)

{{<citation>}}

Isaac Sheidlower, Reuben Aronson, Elaine Short. (2023)  
**Modifying RL Policies with Imagined Actions: How Predictable Policies Can Enable Users to Perform Novel Tasks**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-HC, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05991v1)  

---


**ABSTRACT**  
It is crucial that users are empowered to use the functionalities of a robot to creatively solve problems on the fly. A user who has access to a Reinforcement Learning (RL) based robot may want to use the robot's autonomy and their knowledge of its behavior to complete new tasks. One way is for the user to take control of some of the robot's action space through teleoperation while the RL policy simultaneously controls the rest. However, an out-of-the-box RL policy may not readily facilitate this. For example, a user's control may bring the robot into a failure state from the policy's perspective, causing it to act in a way the user is not familiar with, hindering the success of the user's desired task. In this work, we formalize this problem and present Imaginary Out-of-Distribution Actions, IODA, an initial algorithm for addressing that problem and empowering user's to leverage their expectation of a robot's behavior to accomplish new tasks.

{{</citation>}}


### (63/75) Language-Conditioned Semantic Search-Based Policy for Robotic Manipulation Tasks (Jannik Sheikh et al., 2023)

{{<citation>}}

Jannik Sheikh, Andrew Melnik, Gora Chand Nandi, Robert Haschke. (2023)  
**Language-Conditioned Semantic Search-Based Policy for Robotic Manipulation Tasks**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05925v1)  

---


**ABSTRACT**  
Reinforcement learning and Imitation Learning approaches utilize policy learning strategies that are difficult to generalize well with just a few examples of a task. In this work, we propose a language-conditioned semantic search-based method to produce an online search-based policy from the available demonstration dataset of state-action trajectories. Here we directly acquire actions from the most similar manipulation trajectories found in the dataset. Our approach surpasses the performance of the baselines on the CALVIN benchmark and exhibits strong zero-shot adaptation capabilities. This holds great potential for expanding the use of our online search-based policy approach to tasks typically addressed by Imitation Learning or Reinforcement Learning-based policies.

{{</citation>}}


### (64/75) Explosive Legged Robotic Hopping: Energy Accumulation and Power Amplification via Pneumatic Augmentation (Yifei Chen et al., 2023)

{{<citation>}}

Yifei Chen, Arturo Gamboa-Gonzalez, Michael Wehner, Xiaobin Xiong. (2023)  
**Explosive Legged Robotic Hopping: Energy Accumulation and Power Amplification via Pneumatic Augmentation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2312.05773v1)  

---


**ABSTRACT**  
We present a novel pneumatic augmentation to traditional electric motor-actuated legged robot to increase intermittent power density to perform infrequent explosive hopping behaviors. The pneumatic system is composed of a pneumatic pump, a tank, and a pneumatic actuator. The tank is charged up by the pump during regular hopping motion that is created by the electric motors. At any time after reaching a desired air pressure in the tank, a solenoid valve is utilized to rapidly release the air pressure to the pneumatic actuator (piston) which is used in conjunction with the electric motors to perform explosive hopping, increasing maximum hopping height for one or subsequent cycles. We show that, on a custom-designed one-legged hopping robot, without any additional power source and with this novel pneumatic augmentation system, their associated system identification and optimal control, the robot is able to realize highly explosive hopping with power amplification per cycle by a factor of approximately 5.4 times the power of electric motor actuation alone.

{{</citation>}}


### (65/75) Dynamic Adversarial Attacks on Autonomous Driving Systems (Amirhosein Chahe et al., 2023)

{{<citation>}}

Amirhosein Chahe, Chenan Wang, Abhishek Jeyapratap, Kaidi Xu, Lifeng Zhou. (2023)  
**Dynamic Adversarial Attacks on Autonomous Driving Systems**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2312.06701v1)  

---


**ABSTRACT**  
This paper introduces an attacking mechanism to challenge the resilience of autonomous driving systems. Specifically, we manipulate the decision-making processes of an autonomous vehicle by dynamically displaying adversarial patches on a screen mounted on another moving vehicle. These patches are optimized to deceive the object detection models into misclassifying targeted objects, e.g., traffic signs. Such manipulation has significant implications for critical multi-vehicle interactions such as intersection crossing and lane changing, which are vital for safe and efficient autonomous driving systems. Particularly, we make four major contributions. First, we introduce a novel adversarial attack approach where the patch is not co-located with its target, enabling more versatile and stealthy attacks. Moreover, our method utilizes dynamic patches displayed on a screen, allowing for adaptive changes and movement, enhancing the flexibility and performance of the attack. To do so, we design a Screen Image Transformation Network (SIT-Net), which simulates environmental effects on the displayed images, narrowing the gap between simulated and real-world scenarios. Further, we integrate a positional loss term into the adversarial training process to increase the success rate of the dynamic attack. Finally, we shift the focus from merely attacking perceptual systems to influencing the decision-making algorithms of self-driving systems. Our experiments demonstrate the first successful implementation of such dynamic adversarial attacks in real-world autonomous driving scenarios, paving the way for advancements in the field of robust and secure autonomous driving.

{{</citation>}}


## cs.NE (1)



### (66/75) Cross Fertilizing Empathy from Brain to Machine as a Value Alignment Strategy (Devin Gonier et al., 2023)

{{<citation>}}

Devin Gonier, Adrian Adduci, Cassidy LoCascio. (2023)  
**Cross Fertilizing Empathy from Brain to Machine as a Value Alignment Strategy**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-NE, cs.NE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.07579v1)  

---


**ABSTRACT**  
AI Alignment research seeks to align human and AI goals to ensure independent actions by a machine are always ethical. This paper argues empathy is necessary for this task, despite being often neglected in favor of more deductive approaches. We offer an inside-out approach that grounds morality within the context of the brain as a basis for algorithmically understanding ethics and empathy. These arguments are justified via a survey of relevant literature. The paper concludes with a suggested experimental approach to future research and some initial experimental observations.

{{</citation>}}


## eess.IV (1)



### (67/75) RadImageGAN -- A Multi-modal Dataset-Scale Generative AI for Medical Imaging (Zelong Liu et al., 2023)

{{<citation>}}

Zelong Liu, Alexander Zhou, Arnold Yang, Alara Yilmaz, Maxwell Yoo, Mikey Sullivan, Catherine Zhang, James Grant, Daiqing Li, Zahi A. Fayad, Sean Huver, Timothy Deyer, Xueyan Mei. (2023)  
**RadImageGAN -- A Multi-modal Dataset-Scale Generative AI for Medical Imaging**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI, Generative AI, ImageNet  
[Paper Link](http://arxiv.org/abs/2312.05953v1)  

---


**ABSTRACT**  
Deep learning in medical imaging often requires large-scale, high-quality data or initiation with suitably pre-trained weights. However, medical datasets are limited by data availability, domain-specific knowledge, and privacy concerns, and the creation of large and diverse radiologic databases like RadImageNet is highly resource-intensive. To address these limitations, we introduce RadImageGAN, the first multi-modal radiologic data generator, which was developed by training StyleGAN-XL on the real RadImageNet dataset of 102,774 patients. RadImageGAN can generate high-resolution synthetic medical imaging datasets across 12 anatomical regions and 130 pathological classes in 3 modalities. Furthermore, we demonstrate that RadImageGAN generators can be utilized with BigDatasetGAN to generate multi-class pixel-wise annotated paired synthetic images and masks for diverse downstream segmentation tasks with minimal manual annotation. We showed that using synthetic auto-labeled data from RadImageGAN can significantly improve performance on four diverse downstream segmentation datasets by augmenting real training data and/or developing pre-trained weights for fine-tuning. This shows that RadImageGAN combined with BigDatasetGAN can improve model performance and address data scarcity while reducing the resources needed for annotations for segmentation tasks.

{{</citation>}}


## cs.CY (1)



### (68/75) Exploring Public's Perception of Safety and Video Surveillance Technology: A Survey Approach (Babak Rahimi Ardabili et al., 2023)

{{<citation>}}

Babak Rahimi Ardabili, Armin Danesh Pazho, Ghazal Alinezhad Noghre, Vinit Katariya, Gordon Hull, Shannon Reid, Hamed Tabkhi. (2023)  
**Exploring Public's Perception of Safety and Video Surveillance Technology: A Survey Approach**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2312.06707v1)  

---


**ABSTRACT**  
Addressing public safety effectively requires incorporating diverse stakeholder perspectives, particularly those of the community, which are often underrepresented compared to other stakeholders. This study presents a comprehensive analysis of the community's general public safety concerns, their view of existing surveillance technologies, and their perception of AI-driven solutions for enhancing safety in urban environments, focusing on Charlotte, NC. Through a survey approach, including in-person surveys conducted in August and September 2023 with 410 participants, this research investigates demographic factors such as age, gender, ethnicity, and educational level to gain insights into public perception and concerns toward public safety and possible solutions. Based on the type of dependent variables, we utilized different statistical and significance analyses, such as logit regression and ordinal logistic regression, to explore the effects of demographic factors on the various dependent variables. Our results reveal demographic differences in public safety concerns. Younger females tend to feel less secure yet trust existing video surveillance systems, whereas older, educated individuals are more concerned about violent crimes in malls. Additionally, attitudes towards AI-driven surveillance differ: older Black individuals demonstrate support for it despite having concerns about data privacy, while educated females show a tendency towards skepticism.

{{</citation>}}


## quant-ph (1)



### (69/75) Quantum Private Information Retrieval from Coded Storage Systems (Matteo Allaix, 2023)

{{<citation>}}

Matteo Allaix. (2023)  
**Quantum Private Information Retrieval from Coded Storage Systems**  

---
Primary Category: quant-ph  
Categories: cs-IR, quant-ph, quant-ph  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2312.07570v1)  

---


**ABSTRACT**  
In the era of extensive data growth, robust and efficient mechanisms are needed to store and manage vast amounts of digital information, such as Data Storage Systems (DSSs). Concurrently, privacy concerns have arisen, leading to the development of techniques like Private Information Retrieval (PIR) to enable data access while preserving privacy. A PIR protocol allows users to retrieve information from a database without revealing the specifics of their query or the data they are accessing.   With the advent of quantum computing, researchers have explored the potential of using quantum systems to enhance privacy in information retrieval. In a Quantum Private Information Retrieval (QPIR) protocol, a user can retrieve information from a database by downloading quantum systems from multiple servers, while ensuring that the servers remain oblivious to the specific information being accessed. This scenario offers a unique advantage by leveraging the inherent properties of quantum systems to provide enhanced privacy guarantees and improved communication rates compared to classical PIR protocols.   In this thesis we consider the QPIR setting where the queries and the coded storage systems are classical, while the responses from the servers are quantum. This problem was treated by Song et al. for replicated storage and different collusion patterns. This thesis aims to develop QPIR protocols for coded storage by combining known classical PIR protocols with quantum communication algorithms, achieving enhanced privacy and communication costs. We consider different storage codes and robustness assumptions, and we prove that the achieved communication cost is always lower than the classical counterparts.

{{</citation>}}


## eess.SY (2)



### (70/75) Real-time Estimation of DoS Duration and Frequency for Security Control (Yifan Sun et al., 2023)

{{<citation>}}

Yifan Sun, Jianquan Lu, Daniel W. C. Ho, Lulu Li. (2023)  
**Real-time Estimation of DoS Duration and Frequency for Security Control**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY, math-OC  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2312.05852v1)  

---


**ABSTRACT**  
In this paper, we develop a new denial-of-service (DoS) estimator, enabling defenders to identify duration and frequency parameters of any DoS attacker, except for three edge cases, exclusively using real-time data. The key advantage of the estimator lies in its capability to facilitate security control in a wide range of practical scenarios, even when the attacker's information is previously unknown. We demonstrate the advantage and application of our new estimator in the context of two classical control scenarios, namely consensus of multi-agent systems and impulsive stabilization of nonlinear systems, for illustration.

{{</citation>}}


### (71/75) Synthesis of Temporally-Robust Policies for Signal Temporal Logic Tasks using Reinforcement Learning (Siqi Wang et al., 2023)

{{<citation>}}

Siqi Wang, Shaoyuan Li, Xiang Yin. (2023)  
**Synthesis of Temporally-Robust Policies for Signal Temporal Logic Tasks using Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2312.05764v1)  

---


**ABSTRACT**  
This paper investigates the problem of designing control policies that satisfy high-level specifications described by signal temporal logic (STL) in unknown, stochastic environments. While many existing works concentrate on optimizing the spatial robustness of a system, our work takes a step further by also considering temporal robustness as a critical metric to quantify the tolerance of time uncertainty in STL. To this end, we formulate two relevant control objectives to enhance the temporal robustness of the synthesized policies. The first objective is to maximize the probability of being temporally robust for a given threshold. The second objective is to maximize the worst-case spatial robustness value within a bounded time shift. We use reinforcement learning to solve both control synthesis problems for unknown systems. Specifically, we approximate both control objectives in a way that enables us to apply the standard Q-learning algorithm. Theoretical bounds in terms of the approximations are also derived. We present case studies to demonstrate the feasibility of our approach.

{{</citation>}}


## cs.IR (1)



### (72/75) Towards Global, Socio-Economic, and Culturally Aware Recommender Systems (Kelley Ann Yohe, 2023)

{{<citation>}}

Kelley Ann Yohe. (2023)  
**Towards Global, Socio-Economic, and Culturally Aware Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-CY, cs-IR, cs.IR  
Keywords: Amazon, Google  
[Paper Link](http://arxiv.org/abs/2312.05805v1)  

---


**ABSTRACT**  
Recommender systems have gained increasing attention to personalise consumer preferences. While these systems have primarily focused on applications such as advertisement recommendations (e.g., Google), personalized suggestions (e.g., Netflix and Spotify), and retail selection (e.g., Amazon), there is potential for these systems to benefit from a more global, socio-economic, and culturally aware approach, particularly as companies seek to expand into diverse markets. This paper aims to investigate the potential of a recommender system that considers cultural identity and socio-economic factors. We review the most recent developments in recommender systems and explore the impact of cultural identity and socio-economic factors on consumer preferences. We then propose an ontology and approach for incorporating these factors into recommender systems. To illustrate the potential of our approach, we present a scenario in consumer subscription plan selection within the entertainment industry. We argue that existing recommender systems have limited ability to precisely understand user preferences due to a lack of awareness of socio-economic factors and cultural identity. They also fail to update recommendations in response to changing socio-economic conditions. We explore various machine learning models and develop a final artificial neural network model (ANN) that addresses this gap. We evaluate the effectiveness of socio-economic and culturally aware recommender systems across four dimensions: Precision, Accuracy, F1, and Recall. We find that a highly tuned ANN model incorporating domain-specific data, select cultural indices and relevant socio-economic factors predicts user preference in subscriptions with an accuracy of 95%, a precision of 94%, a F1 Score of 92\%, and a Recall of 90\%.

{{</citation>}}


## cs.SE (1)



### (73/75) Guiding ChatGPT to Fix Web UI Tests via Explanation-Consistency Checking (Zhuolin Xu et al., 2023)

{{<citation>}}

Zhuolin Xu, Yuanzhang Lin, Qiushi Li, Shin Hwei Tan. (2023)  
**Guiding ChatGPT to Fix Web UI Tests via Explanation-Consistency Checking**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2312.05778v1)  

---


**ABSTRACT**  
The rapid evolution of Web UI incurs time and effort in maintaining UI tests. Existing techniques in Web UI test repair focus on finding the target elements on the new web page that match the old ones so that the corresponding broken statements can be repaired. We present the first study that investigates the feasibility of using prior Web UI repair techniques for initial local matching and then using ChatGPT to perform global matching. Our key insight is that given a list of elements matched by prior techniques, ChatGPT can leverage the language understanding to perform global view matching and use its code generation model for fixing the broken statements. To mitigate hallucination in ChatGPT, we design an explanation validator that checks whether the provided explanation for the matching results is consistent, and provides hints to ChatGPT via a self-correction prompt to further improve its results. Our evaluation on a widely used dataset shows that the ChatGPT-enhanced techniques improve the effectiveness of existing Web test repair techniques. Our study also shares several important insights in improving future Web UI test repair techniques.

{{</citation>}}


## cs.SI (1)



### (74/75) GAMC: An Unsupervised Method for Fake News Detection using Graph Autoencoder with Masking (Shu Yin et al., 2023)

{{<citation>}}

Shu Yin, Chao Gao, Zhen Wang. (2023)  
**GAMC: An Unsupervised Method for Fake News Detection using Graph Autoencoder with Masking**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-SI, cs.SI  
Keywords: BERT, Fake News, Transformer  
[Paper Link](http://arxiv.org/abs/2312.05739v1)  

---


**ABSTRACT**  
With the rise of social media, the spread of fake news has become a significant concern, potentially misleading public perceptions and impacting social stability. Although deep learning methods like CNNs, RNNs, and Transformer-based models like BERT have enhanced fake news detection, they primarily focus on content, overlooking social context during news propagation. Graph-based techniques have incorporated this social context but are limited by the need for large labeled datasets. Addressing these challenges, this paper introduces GAMC, an unsupervised fake news detection technique using the Graph Autoencoder with Masking and Contrastive learning. By leveraging both the context and content of news propagation as self-supervised signals, our method negates the requirement for labeled datasets. We augment the original news propagation graph, encode these with a graph encoder, and employ a graph decoder for reconstruction. A unique composite loss function, including reconstruction error and contrast loss, is designed. The method's contributions are: introducing self-supervised learning to fake news detection, proposing a graph autoencoder integrating two distinct losses, and validating our approach's efficacy through real-world dataset experiments.

{{</citation>}}


## cs.MM (1)



### (75/75) AFL-Net: Integrating Audio, Facial, and Lip Modalities with Cross-Attention for Robust Speaker Diarization in the Wild (Yongkang Yin et al., 2023)

{{<citation>}}

Yongkang Yin, Xu Li, Ying Shan, Yuexian Zou. (2023)  
**AFL-Net: Integrating Audio, Facial, and Lip Modalities with Cross-Attention for Robust Speaker Diarization in the Wild**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2312.05730v1)  

---


**ABSTRACT**  
Speaker diarization in real-world videos presents significant challenges due to varying acoustic conditions, diverse scenes, and the presence of off-screen speakers, among other factors. This paper builds upon a previous study (AVR-Net) and introduces a novel multi-modal speaker diarization system, AFL-Net. Unlike AVR-Net, which independently extracts high-level representations from each modality, AFL-Net employs a multi-modal cross-attention mechanism. This approach generates high-level representations from each modality while conditioning on each other, ensuring a more comprehensive information fusion across modalities to enhance identity discrimination. Furthermore, the proposed AFL-Net incorporates dynamic lip movement as an additional modality to aid in distinguishing each segment's identity. We also introduce a masking strategy during training that randomly obscures the face and lip movement modalities, which increases the influence of the audio modality on system outputs.Experimental results demonstrate that our proposed model outperforms state-of-the-art baselines, such as the AVR-Net and DyViSE. Moreover, an ablation study confirms the effectiveness of each modification. Some demos are provided:https://yyk77.github.io/afl_net.github.io.

{{</citation>}}
