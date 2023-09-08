---
draft: false
title: "arXiv @ 2023.09.08"
date: 2023-09-08
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.08"
    identifier: arxiv_20230908
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (19)](#cscv-19)
- [cs.RO (6)](#csro-6)
- [cs.SE (4)](#csse-4)
- [eess.AS (2)](#eessas-2)
- [cs.CR (10)](#cscr-10)
- [cs.LG (22)](#cslg-22)
- [eess.IV (5)](#eessiv-5)
- [cs.AI (3)](#csai-3)
- [cs.CL (19)](#cscl-19)
- [q-fin.ST (1)](#q-finst-1)
- [cs.IR (2)](#csir-2)
- [cs.DC (1)](#csdc-1)
- [cs.SI (3)](#cssi-3)
- [cs.DB (1)](#csdb-1)
- [cs.SD (2)](#cssd-2)
- [cs.HC (1)](#cshc-1)
- [cs.CE (1)](#csce-1)
- [cs.DL (1)](#csdl-1)
- [q-bio.GN (1)](#q-biogn-1)

## cs.CV (19)



### (1/104) Distribution-Aware Prompt Tuning for Vision-Language Models (Eulrang Cho et al., 2023)

{{<citation>}}

Eulrang Cho, Jooyeon Kim, Hyunwoo J. Kim. (2023)  
**Distribution-Aware Prompt Tuning for Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.03406v1)  

---


**ABSTRACT**  
Pre-trained vision-language models (VLMs) have shown impressive performance on various downstream tasks by utilizing knowledge learned from large data. In general, the performance of VLMs on target tasks can be further improved by prompt tuning, which adds context to the input image or text. By leveraging data from target tasks, various prompt-tuning methods have been studied in the literature. A key to prompt tuning is the feature space alignment between two modalities via learnable vectors with model parameters fixed. We observed that the alignment becomes more effective when embeddings of each modality are `well-arranged' in the latent space. Inspired by this observation, we proposed distribution-aware prompt tuning (DAPT) for vision-language models, which is simple yet effective. Specifically, the prompts are learned by maximizing inter-dispersion, the distance between classes, as well as minimizing the intra-dispersion measured by the distance between embeddings from the same class. Our extensive experiments on 11 benchmark datasets demonstrate that our method significantly improves generalizability. The code is available at https://github.com/mlvlab/DAPT.

{{</citation>}}


### (2/104) Reasonable Anomaly Detection in Long Sequences (Yalong Jiang et al., 2023)

{{<citation>}}

Yalong Jiang, Changkang Li. (2023)  
**Reasonable Anomaly Detection in Long Sequences**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2309.03401v1)  

---


**ABSTRACT**  
Video anomaly detection is a challenging task due to the lack in approaches for representing samples. The visual representations of most existing approaches are limited by short-term sequences of observations which cannot provide enough clues for achieving reasonable detections. In this paper, we propose to completely represent the motion patterns of objects by learning from long-term sequences. Firstly, a Stacked State Machine (SSM) model is proposed to represent the temporal dependencies which are consistent across long-range observations. Then SSM model functions in predicting future states based on past ones, the divergence between the predictions with inherent normal patterns and observed ones determines anomalies which violate normal motion patterns. Extensive experiments are carried out to evaluate the proposed approach on the dataset and existing ones. Improvements over state-of-the-art methods can be observed. Our code is available at https://github.com/AllenYLJiang/Anomaly-Detection-in-Sequences.

{{</citation>}}


### (3/104) Self-Supervised Masked Digital Elevation Models Encoding for Low-Resource Downstream Tasks (Priyam Mazumdar et al., 2023)

{{<citation>}}

Priyam Mazumdar, Aiman Soliman, Volodymyr Kindratenko, Luigi Marini, Kenton McHenry. (2023)  
**Self-Supervised Masked Digital Elevation Models Encoding for Low-Resource Downstream Tasks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, BERT, ImageNet, Language Model, Low-Resource, Self-Supervised, Speech Recognition, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.03367v1)  

---


**ABSTRACT**  
The lack of quality labeled data is one of the main bottlenecks for training Deep Learning models. As the task increases in complexity, there is a higher penalty for overfitting and unstable learning. The typical paradigm employed today is Self-Supervised learning, where the model attempts to learn from a large corpus of unstructured and unlabeled data and then transfer that knowledge to the required task. Some notable examples of self-supervision in other modalities are BERT for Large Language Models, Wav2Vec for Speech Recognition, and the Masked AutoEncoder for Vision, which all utilize Transformers to solve a masked prediction task. GeoAI is uniquely poised to take advantage of the self-supervised methodology due to the decades of data collected, little of which is precisely and dependably annotated. Our goal is to extract building and road segmentations from Digital Elevation Models (DEM) that provide a detailed topography of the earths surface. The proposed architecture is the Masked Autoencoder pre-trained on ImageNet (with the limitation that there is a large domain discrepancy between ImageNet and DEM) with an UperNet Head for decoding segmentations. We tested this model with 450 and 50 training images only, utilizing roughly 5% and 0.5% of the original data respectively. On the building segmentation task, this model obtains an 82.1% Intersection over Union (IoU) with 450 Images and 69.1% IoU with only 50 images. On the more challenging road detection task the model obtains an 82.7% IoU with 450 images and 73.2% IoU with only 50 images. Any hand-labeled dataset made today about the earths surface will be immediately obsolete due to the constantly changing nature of the landscape. This motivates the clear necessity for data-efficient learners that can be used for a wide variety of downstream tasks.

{{</citation>}}


### (4/104) ViewMix: Augmentation for Robust Representation in Self-Supervised Learning (Arjon Das et al., 2023)

{{<citation>}}

Arjon Das, Xin Zhong. (2023)  
**ViewMix: Augmentation for Robust Representation in Self-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Augmentation, Embedding, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.03360v1)  

---


**ABSTRACT**  
Joint Embedding Architecture-based self-supervised learning methods have attributed the composition of data augmentations as a crucial factor for their strong representation learning capabilities. While regional dropout strategies have proven to guide models to focus on lesser indicative parts of the objects in supervised methods, it hasn't been adopted by self-supervised methods for generating positive pairs. This is because the regional dropout methods are not suitable for the input sampling process of the self-supervised methodology. Whereas dropping informative pixels from the positive pairs can result in inefficient training, replacing patches of a specific object with a different one can steer the model from maximizing the agreement between different positive pairs. Moreover, joint embedding representation learning methods have not made robustness their primary training outcome. To this end, we propose the ViewMix augmentation policy, specially designed for self-supervised learning, upon generating different views of the same image, patches are cut and pasted from one view to another. By leveraging the different views created by this augmentation strategy, multiple joint embedding-based self-supervised methodologies obtained better localization capability and consistently outperformed their corresponding baseline methods. It is also demonstrated that incorporating ViewMix augmentation policy promotes robustness of the representations in the state-of-the-art methods. Furthermore, our experimentation and analysis of compute times suggest that ViewMix augmentation doesn't introduce any additional overhead compared to other counterparts.

{{</citation>}}


### (5/104) MEGANet: Multi-Scale Edge-Guided Attention Network for Weak Boundary Polyp Segmentation (Nhat-Tan Bui et al., 2023)

{{<citation>}}

Nhat-Tan Bui, Dinh-Hieu Hoang, Quang-Thuc Nguyen, Minh-Triet Tran, Ngan Le. (2023)  
**MEGANet: Multi-Scale Edge-Guided Attention Network for Weak Boundary Polyp Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.03329v1)  

---


**ABSTRACT**  
Efficient polyp segmentation in healthcare plays a critical role in enabling early diagnosis of colorectal cancer. However, the segmentation of polyps presents numerous challenges, including the intricate distribution of backgrounds, variations in polyp sizes and shapes, and indistinct boundaries. Defining the boundary between the foreground (i.e. polyp itself) and the background (surrounding tissue) is difficult. To mitigate these challenges, we propose Multi-Scale Edge-Guided Attention Network (MEGANet) tailored specifically for polyp segmentation within colonoscopy images. This network draws inspiration from the fusion of a classical edge detection technique with an attention mechanism. By combining these techniques, MEGANet effectively preserves high-frequency information, notably edges and boundaries, which tend to erode as neural networks deepen. MEGANet is designed as an end-to-end framework, encompassing three key modules: an encoder, which is responsible for capturing and abstracting the features from the input image, a decoder, which focuses on salient features, and the Edge-Guided Attention module (EGA) that employs the Laplacian Operator to accentuate polyp boundaries. Extensive experiments, both qualitative and quantitative, on five benchmark datasets, demonstrate that our EGANet outperforms other existing SOTA methods under six evaluation metrics. Our code is available at \url{https://github.com/DinhHieuHoang/MEGANet}

{{</citation>}}


### (6/104) Comparative Analysis of Deep-Fake Algorithms (Nikhil Sontakke et al., 2023)

{{<citation>}}

Nikhil Sontakke, Sejal Utekar, Shivansh Rastogi, Shriraj Sonawane. (2023)  
**Comparative Analysis of Deep-Fake Algorithms**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.03295v1)  

---


**ABSTRACT**  
Due to the widespread use of smartphones with high-quality digital cameras and easy access to a wide range of software apps for recording, editing, and sharing videos and images, as well as the deep learning AI platforms, a new phenomenon of 'faking' videos has emerged. Deepfake algorithms can create fake images and videos that are virtually indistinguishable from authentic ones. Therefore, technologies that can detect and assess the integrity of digital visual media are crucial. Deepfakes, also known as deep learning-based fake videos, have become a major concern in recent years due to their ability to manipulate and alter images and videos in a way that is virtually indistinguishable from the original. These deepfake videos can be used for malicious purposes such as spreading misinformation, impersonating individuals, and creating fake news. Deepfake detection technologies use various approaches such as facial recognition, motion analysis, and audio-visual synchronization to identify and flag fake videos. However, the rapid advancement of deepfake technologies has made it increasingly difficult to detect these videos with high accuracy. In this paper, we aim to provide a comprehensive review of the current state of deepfake creation and detection technologies. We examine the various deep learning-based approaches used for creating deepfakes, as well as the techniques used for detecting them. Additionally, we analyze the limitations and challenges of current deepfake detection methods and discuss future research directions in this field. Overall, the paper highlights the importance of continued research and development in deepfake detection technologies in order to combat the negative impact of deepfakes on society and ensure the integrity of digital visual media.

{{</citation>}}


### (7/104) My Art My Choice: Adversarial Protection Against Unruly AI (Anthony Rhodes et al., 2023)

{{<citation>}}

Anthony Rhodes, Ram Bhagat, Umur Aybars Ciftci, Ilke Demir. (2023)  
**My Art My Choice: Adversarial Protection Against Unruly AI**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2309.03198v1)  

---


**ABSTRACT**  
Generative AI is on the rise, enabling everyone to produce realistic content via publicly available interfaces. Especially for guided image generation, diffusion models are changing the creator economy by producing high quality low cost content. In parallel, artists are rising against unruly AI, since their artwork are leveraged, distributed, and dissimulated by large generative models. Our approach, My Art My Choice (MAMC), aims to empower content owners by protecting their copyrighted materials from being utilized by diffusion models in an adversarial fashion. MAMC learns to generate adversarially perturbed "protected" versions of images which can in turn "break" diffusion models. The perturbation amount is decided by the artist to balance distortion vs. protection of the content. MAMC is designed with a simple UNet-based generator, attacking black box diffusion models, combining several losses to create adversarial twins of the original artwork. We experiment on three datasets for various image-to-image tasks, with different user control values. Both protected image and diffusion output results are evaluated in visual, noise, structure, pixel, and generative spaces to validate our claims. We believe that MAMC is a crucial step for preserving ownership information for AI generated content in a flawless, based-on-need, and human-centric way.

{{</citation>}}


### (8/104) PDiscoNet: Semantically consistent part discovery for fine-grained recognition (Robert van der Klis et al., 2023)

{{<citation>}}

Robert van der Klis, Stephan Alaniz, Massimiliano Mancini, Cassio F. Dantas, Dino Ienco, Zeynep Akata, Diego Marcos. (2023)  
**PDiscoNet: Semantically consistent part discovery for fine-grained recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.03173v1)  

---


**ABSTRACT**  
Fine-grained classification often requires recognizing specific object parts, such as beak shape and wing patterns for birds. Encouraging a fine-grained classification model to first detect such parts and then using them to infer the class could help us gauge whether the model is indeed looking at the right details better than with interpretability methods that provide a single attribution map. We propose PDiscoNet to discover object parts by using only image-level class labels along with priors encouraging the parts to be: discriminative, compact, distinct from each other, equivariant to rigid transforms, and active in at least some of the images. In addition to using the appropriate losses to encode these priors, we propose to use part-dropout, where full part feature vectors are dropped at once to prevent a single part from dominating in the classification, and part feature vector modulation, which makes the information coming from each part distinct from the perspective of the classifier. Our results on CUB, CelebA, and PartImageNet show that the proposed method provides substantially better part discovery performance than previous methods while not requiring any additional hyper-parameter tuning and without penalizing the classification performance. The code is available at https://github.com/robertdvdk/part_detection.

{{</citation>}}


### (9/104) Character Queries: A Transformer-based Approach to On-Line Handwritten Character Segmentation (Michael Jungo et al., 2023)

{{<citation>}}

Michael Jungo, Beat Wolf, Andrii Maksai, Claudiu Musat, Andreas Fischer. (2023)  
**Character Queries: A Transformer-based Approach to On-Line Handwritten Character Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.03072v1)  

---


**ABSTRACT**  
On-line handwritten character segmentation is often associated with handwriting recognition and even though recognition models include mechanisms to locate relevant positions during the recognition process, it is typically insufficient to produce a precise segmentation. Decoupling the segmentation from the recognition unlocks the potential to further utilize the result of the recognition. We specifically focus on the scenario where the transcription is known beforehand, in which case the character segmentation becomes an assignment problem between sampling points of the stylus trajectory and characters in the text. Inspired by the $k$-means clustering algorithm, we view it from the perspective of cluster assignment and present a Transformer-based architecture where each cluster is formed based on a learned character query in the Transformer decoder block. In order to assess the quality of our approach, we create character segmentation ground truths for two popular on-line handwriting datasets, IAM-OnDB and HANDS-VNOnDB, and evaluate multiple methods on them, demonstrating that our approach achieves the overall best results.

{{</citation>}}


### (10/104) Prompt-based All-in-One Image Restoration using CNNs and Transformer (Hu Gao et al., 2023)

{{<citation>}}

Hu Gao, Jing Yang, Ning Wang, Jingfan Yang, Ying Zhang, Depeng Dang. (2023)  
**Prompt-based All-in-One Image Restoration using CNNs and Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.03063v1)  

---


**ABSTRACT**  
Image restoration aims to recover the high-quality images from their degraded observations. Since most existing methods have been dedicated into single degradation removal, they may not yield optimal results on other types of degradations, which do not satisfy the applications in real world scenarios. In this paper, we propose a novel data ingredient-oriented approach that leverages prompt-based learning to enable a single model to efficiently tackle multiple image degradation tasks. Specifically, we utilize a encoder to capture features and introduce prompts with degradation-specific information to guide the decoder in adaptively recovering images affected by various degradations. In order to model the local invariant properties and non-local information for high-quality image restoration, we combined CNNs operations and Transformers. Simultaneously, we made several key designs in the Transformer blocks (multi-head rearranged attention with prompts and simple-gate feed-forward network) to reduce computational requirements and selectively determines what information should be persevered to facilitate efficient recovery of potentially sharp images. Furthermore, we incorporate a feature fusion mechanism further explores the multi-scale information to improve the aggregated features. The resulting tightly interlinked hierarchy architecture, named as CAPTNet, despite being designed to handle different types of degradations, extensive experiments demonstrate that our method performs competitively to the task-specific algorithms.

{{</citation>}}


### (11/104) Combining pre-trained Vision Transformers and CIDER for Out Of Domain Detection (Grégor Jouet et al., 2023)

{{<citation>}}

Grégor Jouet, Clément Duhart, Francis Rousseaux, Julio Laborde, Cyril de Runz. (2023)  
**Combining pre-trained Vision Transformers and CIDER for Out Of Domain Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.03047v1)  

---


**ABSTRACT**  
Out-of-domain (OOD) detection is a crucial component in industrial applications as it helps identify when a model encounters inputs that are outside the training distribution. Most industrial pipelines rely on pre-trained models for downstream tasks such as CNN or Vision Transformers. This paper investigates the performance of those models on the task of out-of-domain detection. Our experiments demonstrate that pre-trained transformers models achieve higher detection performance out of the box. Furthermore, we show that pre-trained ViT and CNNs can be combined with refinement methods such as CIDER to improve their OOD detection performance even more. Our results suggest that transformers are a promising approach for OOD detection and set a stronger baseline for this task in many contexts

{{</citation>}}


### (12/104) MCM: Multi-condition Motion Synthesis Framework for Multi-scenario (Zeyu Ling et al., 2023)

{{<citation>}}

Zeyu Ling, Bo Han, Yongkang Wong, Mohan Kangkanhalli, Weidong Geng. (2023)  
**MCM: Multi-condition Motion Synthesis Framework for Multi-scenario**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.03031v1)  

---


**ABSTRACT**  
The objective of the multi-condition human motion synthesis task is to incorporate diverse conditional inputs, encompassing various forms like text, music, speech, and more. This endows the task with the capability to adapt across multiple scenarios, ranging from text-to-motion and music-to-dance, among others. While existing research has primarily focused on single conditions, the multi-condition human motion generation remains underexplored. In this paper, we address these challenges by introducing MCM, a novel paradigm for motion synthesis that spans multiple scenarios under diverse conditions. The MCM framework is able to integrate with any DDPM-like diffusion model to accommodate multi-conditional information input while preserving its generative capabilities. Specifically, MCM employs two-branch architecture consisting of a main branch and a control branch. The control branch shares the same structure as the main branch and is initialized with the parameters of the main branch, effectively maintaining the generation ability of the main branch and supporting multi-condition input. We also introduce a Transformer-based diffusion model MWNet (DDPM-like) as our main branch that can capture the spatial complexity and inter-joint correlations in motion sequences through a channel-dimension self-attention module. Quantitative comparisons demonstrate that our approach achieves SoTA results in both text-to-motion and competitive results in music-to-dance tasks, comparable to task-specific methods. Furthermore, the qualitative evaluation shows that MCM not only streamlines the adaptation of methodologies originally designed for text-to-motion tasks to domains like music-to-dance and speech-to-gesture, eliminating the need for extensive network re-configurations but also enables effective multi-condition modal control, realizing "once trained is motion need".

{{</citation>}}


### (13/104) Sparse 3D Reconstruction via Object-Centric Ray Sampling (Llukman Cerkezi et al., 2023)

{{<citation>}}

Llukman Cerkezi, Paolo Favaro. (2023)  
**Sparse 3D Reconstruction via Object-Centric Ray Sampling**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.03008v1)  

---


**ABSTRACT**  
We propose a novel method for 3D object reconstruction from a sparse set of views captured from a 360-degree calibrated camera rig. We represent the object surface through a hybrid model that uses both an MLP-based neural representation and a triangle mesh. A key contribution in our work is a novel object-centric sampling scheme of the neural representation, where rays are shared among all views. This efficiently concentrates and reduces the number of samples used to update the neural model at each iteration. This sampling scheme relies on the mesh representation to ensure also that samples are well-distributed along its normals. The rendering is then performed efficiently by a differentiable renderer. We demonstrate that this sampling scheme results in a more effective training of the neural representation, does not require the additional supervision of segmentation masks, yields state of the art 3D reconstructions, and works with sparse views on the Google's Scanned Objects, Tank and Temples and MVMC Car datasets.

{{</citation>}}


### (14/104) Dynamic Hyperbolic Attention Network for Fine Hand-object Reconstruction (Zhiying Leng et al., 2023)

{{<citation>}}

Zhiying Leng, Shun-Cheng Wu, Mahdi Saleh, Antonio Montanaro, Hao Yu, Yin Wang, Nassir Navab, Xiaohui Liang, Federico Tombari. (2023)  
**Dynamic Hyperbolic Attention Network for Fine Hand-object Reconstruction**  

---
Primary Category: cs.CV  
Categories: I-4-5, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.02965v1)  

---


**ABSTRACT**  
Reconstructing both objects and hands in 3D from a single RGB image is complex. Existing methods rely on manually defined hand-object constraints in Euclidean space, leading to suboptimal feature learning. Compared with Euclidean space, hyperbolic space better preserves the geometric properties of meshes thanks to its exponentially-growing space distance, which amplifies the differences between the features based on similarity. In this work, we propose the first precise hand-object reconstruction method in hyperbolic space, namely Dynamic Hyperbolic Attention Network (DHANet), which leverages intrinsic properties of hyperbolic space to learn representative features. Our method that projects mesh and image features into a unified hyperbolic space includes two modules, ie. dynamic hyperbolic graph convolution and image-attention hyperbolic graph convolution. With these two modules, our method learns mesh features with rich geometry-image multi-modal information and models better hand-object interaction. Our method provides a promising alternative for fine hand-object reconstruction in hyperbolic space. Extensive experiments on three public datasets demonstrate that our method outperforms most state-of-the-art methods.

{{</citation>}}


### (15/104) Knowledge Distillation Layer that Lets the Student Decide (Ada Gorgun et al., 2023)

{{<citation>}}

Ada Gorgun, Yeti Z. Gurbuz, A. Aydin Alatan. (2023)  
**Knowledge Distillation Layer that Lets the Student Decide**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, stat-ML  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.02843v1)  

---


**ABSTRACT**  
Typical technique in knowledge distillation (KD) is regularizing the learning of a limited capacity model (student) by pushing its responses to match a powerful model's (teacher). Albeit useful especially in the penultimate layer and beyond, its action on student's feature transform is rather implicit, limiting its practice in the intermediate layers. To explicitly embed the teacher's knowledge in feature transform, we propose a learnable KD layer for the student which improves KD with two distinct abilities: i) learning how to leverage the teacher's knowledge, enabling to discard nuisance information, and ii) feeding forward the transferred knowledge deeper. Thus, the student enjoys the teacher's knowledge during the inference besides training. Formally, we repurpose 1x1-BN-ReLU-1x1 convolution block to assign a semantic vector to each local region according to the template (supervised by the teacher) that the corresponding region of the student matches. To facilitate template learning in the intermediate layers, we propose a novel form of supervision based on the teacher's decisions. Through rigorous experimentation, we demonstrate the effectiveness of our approach on 3 popular classification benchmarks. Code is available at: https://github.com/adagorgun/letKD-framework

{{</citation>}}


### (16/104) Image-Object-Specific Prompt Learning for Few-Shot Class-Incremental Learning (In-Ug Yoon et al., 2023)

{{<citation>}}

In-Ug Yoon, Tae-Min Choi, Sun-Kyung Lee, Young-Min Kim, Jong-Hwan Kim. (2023)  
**Image-Object-Specific Prompt Learning for Few-Shot Class-Incremental Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, ImageNet  
[Paper Link](http://arxiv.org/abs/2309.02833v1)  

---


**ABSTRACT**  
While many FSCIL studies have been undertaken, achieving satisfactory performance, especially during incremental sessions, has remained challenging. One prominent challenge is that the encoder, trained with an ample base session training set, often underperforms in incremental sessions. In this study, we introduce a novel training framework for FSCIL, capitalizing on the generalizability of the Contrastive Language-Image Pre-training (CLIP) model to unseen classes. We achieve this by formulating image-object-specific (IOS) classifiers for the input images. Here, an IOS classifier refers to one that targets specific attributes (like wings or wheels) of class objects rather than the image's background. To create these IOS classifiers, we encode a bias prompt into the classifiers using our specially designed module, which harnesses key-prompt pairs to pinpoint the IOS features of classes in each session. From an FSCIL standpoint, our framework is structured to retain previous knowledge and swiftly adapt to new sessions without forgetting or overfitting. This considers the updatability of modules in each session and some tricks empirically found for fast convergence. Our approach consistently demonstrates superior performance compared to state-of-the-art methods across the miniImageNet, CIFAR100, and CUB200 datasets. Further, we provide additional experiments to validate our learned model's ability to achieve IOS classifiers. We also conduct ablation studies to analyze the impact of each module within the architecture.

{{</citation>}}


### (17/104) 3D Trajectory Reconstruction of Drones using a Single Camera (Seobin Hwang et al., 2023)

{{<citation>}}

Seobin Hwang, Hanyoung Kim, Chaeyeon Heo, Youkyoung Na, Cheongeun Lee, Yeongjun Cho. (2023)  
**3D Trajectory Reconstruction of Drones using a Single Camera**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2309.02801v1)  

---


**ABSTRACT**  
Drones have been widely utilized in various fields, but the number of drones being used illegally and for hazardous purposes has increased recently. To prevent those illegal drones, in this work, we propose a novel framework for reconstructing 3D trajectories of drones using a single camera. By leveraging calibrated cameras, we exploit the relationship between 2D and 3D spaces. We automatically track the drones in 2D images using the drone tracker and estimate their 2D rotations. By combining the estimated 2D drone positions with their actual length information and camera parameters, we geometrically infer the 3D trajectories of the drones. To address the lack of public drone datasets, we also create synthetic 2D and 3D drone datasets. The experimental results show that the proposed methods accurately reconstruct drone trajectories in 3D space, and demonstrate the potential of our framework for single camera-based surveillance systems.

{{</citation>}}


### (18/104) DMKD: Improving Feature-based Knowledge Distillation for Object Detection Via Dual Masking Augmentation (Guang Yang et al., 2023)

{{<citation>}}

Guang Yang, Yin Tang, Zhijian Wu, Jun Li, Jianhua Xu, Xili Wan. (2023)  
**DMKD: Improving Feature-based Knowledge Distillation for Object Detection Via Dual Masking Augmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Knowledge Distillation, Object Detection  
[Paper Link](http://arxiv.org/abs/2309.02719v2)  

---


**ABSTRACT**  
Recent mainstream masked distillation methods function by reconstructing selectively masked areas of a student network from the feature map of its teacher counterpart. In these methods, the masked regions need to be properly selected, such that reconstructed features encode sufficient discrimination and representation capability like the teacher feature. However, previous masked distillation methods only focus on spatial masking, making the resulting masked areas biased towards spatial importance without encoding informative channel clues. In this study, we devise a Dual Masked Knowledge Distillation (DMKD) framework which can capture both spatially important and channel-wise informative clues for comprehensive masked feature reconstruction. More specifically, we employ dual attention mechanism for guiding the respective masking branches, leading to reconstructed feature encoding dual significance. Furthermore, fusing the reconstructed features is achieved by self-adjustable weighting strategy for effective feature distillation. Our experiments on object detection task demonstrate that the student networks achieve performance gains of 4.1% and 4.3% with the help of our method when RetinaNet and Cascade Mask R-CNN are respectively used as the teacher networks, while outperforming the other state-of-the-art distillation methods.

{{</citation>}}


### (19/104) Efficient Training for Visual Tracking with Deformable Transformer (Qingmao Wei et al., 2023)

{{<citation>}}

Qingmao Wei, Guotian Zeng, Bi Zeng. (2023)  
**Efficient Training for Visual Tracking with Deformable Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.02676v1)  

---


**ABSTRACT**  
Recent Transformer-based visual tracking models have showcased superior performance. Nevertheless, prior works have been resource-intensive, requiring prolonged GPU training hours and incurring high GFLOPs during inference due to inefficient training methods and convolution-based target heads. This intensive resource use renders them unsuitable for real-world applications. In this paper, we present DETRack, a streamlined end-to-end visual object tracking framework. Our framework utilizes an efficient encoder-decoder structure where the deformable transformer decoder acting as a target head, achieves higher sparsity than traditional convolution heads, resulting in decreased GFLOPs. For training, we introduce a novel one-to-many label assignment and an auxiliary denoising technique, significantly accelerating model's convergence. Comprehensive experiments affirm the effectiveness and efficiency of our proposed method. For instance, DETRack achieves 72.9% AO on challenging GOT-10k benchmarks using only 20% of the training epochs required by the baseline, and runs with lower GFLOPs than all the transformer-based trackers.

{{</citation>}}


## cs.RO (6)



### (20/104) Efficient Baselines for Motion Prediction in Autonomous Driving (Carlos Gómez-Huélamo et al., 2023)

{{<citation>}}

Carlos Gómez-Huélamo, Marcos V. Conde, Rafael Barea, Manuel Ocaña, Luis M. Bergasa. (2023)  
**Efficient Baselines for Motion Prediction in Autonomous Driving**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-MA, cs-RO, cs.RO  
Keywords: GNN, LSTM  
[Paper Link](http://arxiv.org/abs/2309.03387v1)  

---


**ABSTRACT**  
Motion Prediction (MP) of multiple surroundings agents is a crucial task in arbitrarily complex environments, from simple robots to Autonomous Driving Stacks (ADS). Current techniques tackle this problem using end-to-end pipelines, where the input data is usually a rendered top-view of the physical information and the past trajectories of the most relevant agents; leveraging this information is a must to obtain optimal performance. In that sense, a reliable ADS must produce reasonable predictions on time. However, despite many approaches use simple ConvNets and LSTMs to obtain the social latent features, State-Of-The-Art (SOTA) models might be too complex for real-time applications when using both sources of information (map and past trajectories) as well as little interpretable, specially considering the physical information. Moreover, the performance of such models highly depends on the number of available inputs for each particular traffic scenario, which are expensive to obtain, particularly, annotated High-Definition (HD) maps.   In this work, we propose several efficient baselines for the well-known Argoverse 1 Motion Forecasting Benchmark. We aim to develop compact models using SOTA techniques for MP, including attention mechanisms and GNNs. Our lightweight models use standard social information and interpretable map information such as points from the driveable area and plausible centerlines by means of a novel preprocessing step based on kinematic constraints, in opposition to black-box CNN-based or too-complex graphs methods for map encoding, to generate plausible multimodal trajectories achieving up-to-pair accuracy with less operations and parameters than other SOTA methods. Our code is publicly available at https://github.com/Cram3r95/mapfe4mp .

{{</citation>}}


### (21/104) Learning to Recharge: UAV Coverage Path Planning through Deep Reinforcement Learning (Mirco Theile et al., 2023)

{{<citation>}}

Mirco Theile, Harald Bayerlein, Marco Caccamo, Alberto L. Sangiovanni-Vincentelli. (2023)  
**Learning to Recharge: UAV Coverage Path Planning through Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.03157v1)  

---


**ABSTRACT**  
Coverage path planning (CPP) is a critical problem in robotics, where the goal is to find an efficient path that covers every point in an area of interest. This work addresses the power-constrained CPP problem with recharge for battery-limited unmanned aerial vehicles (UAVs). In this problem, a notable challenge emerges from integrating recharge journeys into the overall coverage strategy, highlighting the intricate task of making strategic, long-term decisions. We propose a novel proximal policy optimization (PPO)-based deep reinforcement learning (DRL) approach with map-based observations, utilizing action masking and discount factor scheduling to optimize coverage trajectories over the entire mission horizon. We further provide the agent with a position history to handle emergent state loops caused by the recharge capability. Our approach outperforms a baseline heuristic, generalizes to different target zones and maps, with limited generalization to unseen maps. We offer valuable insights into DRL algorithm design for long-horizon problems and provide a publicly available software framework for the CPP problem.

{{</citation>}}


### (22/104) Serving Time: Real-Time, Safe Motion Planning and Control for Manipulation of Unsecured Objects (Zachary Brei et al., 2023)

{{<citation>}}

Zachary Brei, Jonathan Michaux, Bohao Zhang, Patrick Holmes, Ram Vasudevan. (2023)  
**Serving Time: Real-Time, Safe Motion Planning and Control for Manipulation of Unsecured Objects**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY, math-OC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.03111v1)  

---


**ABSTRACT**  
A key challenge to ensuring the rapid transition of robotic systems from the industrial sector to more ubiquitous applications is the development of algorithms that can guarantee safe operation while in close proximity to humans. Motion planning and control methods, for instance, must be able to certify safety while operating in real-time in arbitrary environments and in the presence of model uncertainty. This paper proposes Wrench Analysis for Inertial Transport using Reachability (WAITR), a certifiably safe motion planning and control framework for serial link manipulators that manipulate unsecured objects in arbitrary environments. WAITR uses reachability analysis to construct over-approximations of the contact wrench applied to unsecured objects, which captures uncertainty in the manipulator dynamics, the object dynamics, and contact parameters such as the coefficient of friction. An optimization problem formulation is presented that can be solved in real-time to generate provably-safe motions for manipulating the unsecured objects. This paper illustrates that WAITR outperforms state of the art methods in a variety of simulation experiments and demonstrates its performance in the real-world.

{{</citation>}}


### (23/104) Natural and Robust Walking using Reinforcement Learning without Demonstrations in High-Dimensional Musculoskeletal Models (Pierre Schumacher et al., 2023)

{{<citation>}}

Pierre Schumacher, Thomas Geijtenbeek, Vittorio Caggiano, Vikash Kumar, Syn Schmitt, Georg Martius, Daniel F. B. Haeufle. (2023)  
**Natural and Robust Walking using Reinforcement Learning without Demonstrations in High-Dimensional Musculoskeletal Models**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.02976v2)  

---


**ABSTRACT**  
Humans excel at robust bipedal walking in complex natural environments. In each step, they adequately tune the interaction of biomechanical muscle dynamics and neuronal signals to be robust against uncertainties in ground conditions. However, it is still not fully understood how the nervous system resolves the musculoskeletal redundancy to solve the multi-objective control problem considering stability, robustness, and energy efficiency. In computer simulations, energy minimization has been shown to be a successful optimization target, reproducing natural walking with trajectory optimization or reflex-based control methods. However, these methods focus on particular motions at a time and the resulting controllers are limited when compensating for perturbations. In robotics, reinforcement learning~(RL) methods recently achieved highly stable (and efficient) locomotion on quadruped systems, but the generation of human-like walking with bipedal biomechanical models has required extensive use of expert data sets. This strong reliance on demonstrations often results in brittle policies and limits the application to new behaviors, especially considering the potential variety of movements for high-dimensional musculoskeletal models in 3D. Achieving natural locomotion with RL without sacrificing its incredible robustness might pave the way for a novel approach to studying human walking in complex natural environments. Videos: https://sites.google.com/view/naturalwalkingrl

{{</citation>}}


### (24/104) Learning Vehicle Dynamics from Cropped Image Patches for Robot Navigation in Unpaved Outdoor Terrains (Jeong Hyun Lee et al., 2023)

{{<citation>}}

Jeong Hyun Lee, Jinhyeok Choi, Simo Ryu, Hyunsik Oh, Suyoung Choi, Jemin Hwangbo. (2023)  
**Learning Vehicle Dynamics from Cropped Image Patches for Robot Navigation in Unpaved Outdoor Terrains**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.02745v1)  

---


**ABSTRACT**  
In the realm of autonomous mobile robots, safe navigation through unpaved outdoor environments remains a challenging task. Due to the high-dimensional nature of sensor data, extracting relevant information becomes a complex problem, which hinders adequate perception and path planning. Previous works have shown promising performances in extracting global features from full-sized images. However, they often face challenges in capturing essential local information. In this paper, we propose Crop-LSTM, which iteratively takes cropped image patches around the current robot's position and predicts the future position, orientation, and bumpiness. Our method performs local feature extraction by paying attention to corresponding image patches along the predicted robot trajectory in the 2D image plane. This enables more accurate predictions of the robot's future trajectory. With our wheeled mobile robot platform Raicart, we demonstrated the effectiveness of Crop-LSTM for point-goal navigation in an unpaved outdoor environment. Our method enabled safe and robust navigation using RGBD images in challenging unpaved outdoor terrains. The summary video is available at https://youtu.be/iIGNZ8ignk0.

{{</citation>}}


### (25/104) Reinforcement Learning of Action and Query Policies with LTL Instructions under Uncertain Event Detector (Wataru Hatanaka et al., 2023)

{{<citation>}}

Wataru Hatanaka, Ryota Yamashina, Takamitsu Matsubara. (2023)  
**Reinforcement Learning of Action and Query Policies with LTL Instructions under Uncertain Event Detector**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.02722v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) with linear temporal logic (LTL) objectives can allow robots to carry out symbolic event plans in unknown environments. Most existing methods assume that the event detector can accurately map environmental states to symbolic events; however, uncertainty is inevitable for real-world event detectors. Such uncertainty in an event detector generates multiple branching possibilities on LTL instructions, confusing action decisions. Moreover, the queries to the uncertain event detector, necessary for the task's progress, may increase the uncertainty further. To cope with those issues, we propose an RL framework, Learning Action and Query over Belief LTL (LAQBL), to learn an agent that can consider the diversity of LTL instructions due to uncertain event detection while avoiding task failure due to the unnecessary event-detection query. Our framework simultaneously learns 1) an embedding of belief LTL, which is multiple branching possibilities on LTL instructions using a graph neural network, 2) an action policy, and 3) a query policy which decides whether or not to query for the event detector. Simulations in a 2D grid world and image-input robotic inspection environments show that our method successfully learns actions to follow LTL instructions even with uncertain event detectors.

{{</citation>}}


## cs.SE (4)



### (26/104) Unity is Strength: Cross-Task Knowledge Distillation to Improve Code Review Generation (Oussama Ben Sghaier et al., 2023)

{{<citation>}}

Oussama Ben Sghaier, Lucas Maes, Houari Sahraoui. (2023)  
**Unity is Strength: Cross-Task Knowledge Distillation to Improve Code Review Generation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BLEU, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.03362v1)  

---


**ABSTRACT**  
Code review is a fundamental process in software development that plays a critical role in ensuring code quality and reducing the likelihood of errors and bugs. However, code review might be complex, subjective, and time-consuming. Comment generation and code refinement are two key tasks of this process and their automation has traditionally been addressed separately in the literature using different approaches. In this paper, we propose a novel deep-learning architecture, DISCOREV, based on cross-task knowledge distillation that addresses these two tasks simultaneously. In our approach, the fine-tuning of the comment generation model is guided by the code refinement model. We implemented this guidance using two strategies, feedback-based learning objective and embedding alignment objective. We evaluated our approach based on cross-task knowledge distillation by comparing it to the state-of-the-art methods that are based on independent training and fine-tuning. Our results show that our approach generates better review comments as measured by the BLEU score.

{{</citation>}}


### (27/104) Method-Level Bug Severity Prediction using Source Code Metrics and LLMs (Ehsan Mashhadi et al., 2023)

{{<citation>}}

Ehsan Mashhadi, Hossein Ahmadvand, Hadi Hemmati. (2023)  
**Method-Level Bug Severity Prediction using Source Code Metrics and LLMs**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.03044v1)  

---


**ABSTRACT**  
In the past couple of decades, significant research efforts are devoted to the prediction of software bugs. However, most existing work in this domain treats all bugs the same, which is not the case in practice. It is important for a defect prediction method to estimate the severity of the identified bugs so that the higher-severity ones get immediate attention. In this study, we investigate source code metrics, source code representation using large language models (LLMs), and their combination in predicting bug severity labels of two prominent datasets. We leverage several source metrics at method-level granularity to train eight different machine-learning models. Our results suggest that Decision Tree and Random Forest models outperform other models regarding our several evaluation metrics. We then use the pre-trained CodeBERT LLM to study the source code representations' effectiveness in predicting bug severity. CodeBERT finetuning improves the bug severity prediction results significantly in the range of 29%-140% for several evaluation metrics, compared to the best classic prediction model on source code metric. Finally, we integrate source code metrics into CodeBERT as an additional input, using our two proposed architectures, which both enhance the CodeBERT model effectiveness.

{{</citation>}}


### (28/104) EdgeFL: A Lightweight Decentralized Federated Learning Framework (Hongyi Zhang et al., 2023)

{{<citation>}}

Hongyi Zhang, Jan Bosch, Helena Holmström Olsson. (2023)  
**EdgeFL: A Lightweight Decentralized Federated Learning Framework**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02936v1)  

---


**ABSTRACT**  
Federated Learning (FL) has emerged as a promising approach for collaborative machine learning, addressing data privacy concerns. However, existing FL platforms and frameworks often present challenges for software engineers in terms of complexity, limited customization options, and scalability limitations. In this paper, we introduce EdgeFL, an edge-only lightweight decentralized FL framework, designed to overcome the limitations of centralized aggregation and scalability in FL deployments. By adopting an edge-only model training and aggregation approach, EdgeFL eliminates the need for a central server, enabling seamless scalability across diverse use cases. With a straightforward integration process requiring just four lines of code (LOC), software engineers can easily incorporate FL functionalities into their AI products. Furthermore, EdgeFL offers the flexibility to customize aggregation functions, empowering engineers to adapt them to specific needs. Based on the results, we demonstrate that EdgeFL achieves superior performance compared to existing FL platforms/frameworks. Our results show that EdgeFL reduces weights update latency and enables faster model evolution, enhancing the efficiency of edge devices. Moreover, EdgeFL exhibits improved classification accuracy compared to traditional centralized FL approaches. By leveraging EdgeFL, software engineers can harness the benefits of federated learning while overcoming the challenges associated with existing FL platforms/frameworks.

{{</citation>}}


### (29/104) Improving Code Generation by Dynamic Temperature Sampling (Yuqi Zhu et al., 2023)

{{<citation>}}

Yuqi Zhu, Jia Allen Li, Ge Li, YunFei Zhao, Jia Li, Zhi Jin, Hong Mei. (2023)  
**Improving Code Generation by Dynamic Temperature Sampling**  

---
Primary Category: cs.SE  
Categories: cs-CL, cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.02772v1)  

---


**ABSTRACT**  
Recently, Large Language Models (LLMs) have shown impressive results in code generation. However, existing decoding strategies are designed for Natural Language (NL) generation, overlooking the differences between NL and programming languages (PL). Due to this oversight, a better decoding strategy for code generation remains an open question. In this paper, we conduct the first systematic study to explore a decoding strategy specialized in code generation. With an analysis of loss distributions of code tokens, we find that code tokens can be divided into two categories: challenging tokens that are difficult to predict and confident tokens that can be easily inferred. Among them, the challenging tokens mainly appear at the beginning of a code block. Inspired by the above findings, we propose a simple yet effective method: Adaptive Temperature (AdapT) sampling, which dynamically adjusts the temperature coefficient when decoding different tokens. We apply a larger temperature when sampling for challenging tokens, allowing LLMs to explore diverse choices. We employ a smaller temperature for confident tokens avoiding the influence of tail randomness noises. We apply AdapT sampling to LLMs with different sizes and conduct evaluations on two popular datasets. Results show that AdapT sampling significantly outperforms state-of-the-art decoding strategy.

{{</citation>}}


## eess.AS (2)



### (30/104) Leveraging Geometrical Acoustic Simulations of Spatial Room Impulse Responses for Improved Sound Event Detection and Localization (Christopher Ick et al., 2023)

{{<citation>}}

Christopher Ick, Brian McFee. (2023)  
**Leveraging Geometrical Acoustic Simulations of Spatial Room Impulse Responses for Improved Sound Event Detection and Localization**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Event Detection  
[Paper Link](http://arxiv.org/abs/2309.03337v1)  

---


**ABSTRACT**  
As deeper and more complex models are developed for the task of sound event localization and detection (SELD), the demand for annotated spatial audio data continues to increase. Annotating field recordings with 360$^{\circ}$ video takes many hours from trained annotators, while recording events within motion-tracked laboratories are bounded by cost and expertise. Because of this, localization models rely on a relatively limited amount of spatial audio data in the form of spatial room impulse response (SRIR) datasets, which limits the progress of increasingly deep neural network based approaches. In this work, we demonstrate that simulated geometrical acoustics can provide an appealing solution to this problem. We use simulated geometrical acoustics to generate a novel SRIR dataset that can train a SELD model to provide similar performance to that of a real SRIR dataset. Furthermore, we demonstrate using simulated data to augment existing datasets, improving on benchmarks set by state of the art SELD models. We explore the potential and limitations of geometric acoustic simulation for localization and event detection. We also propose further studies to verify the limitations of this method, as well as further methods to generate synthetic data for SELD tasks without the need to record more data.

{{</citation>}}


### (31/104) MuLanTTS: The Microsoft Speech Synthesis System for Blizzard Challenge 2023 (Zhihang Xu et al., 2023)

{{<citation>}}

Zhihang Xu, Shaofei Zhang, Xi Wang, Jiajun Zhang, Wenning Wei, Lei He, Sheng Zhao. (2023)  
**MuLanTTS: The Microsoft Speech Synthesis System for Blizzard Challenge 2023**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2309.02743v2)  

---


**ABSTRACT**  
In this paper, we present MuLanTTS, the Microsoft end-to-end neural text-to-speech (TTS) system designed for the Blizzard Challenge 2023. About 50 hours of audiobook corpus for French TTS as hub task and another 2 hours of speaker adaptation as spoke task are released to build synthesized voices for different test purposes including sentences, paragraphs, homographs, lists, etc. Building upon DelightfulTTS, we adopt contextual and emotion encoders to adapt the audiobook data to enrich beyond sentences for long-form prosody and dialogue expressiveness. Regarding the recording quality, we also apply denoise algorithms and long audio processing for both corpora. For the hub task, only the 50-hour single speaker data is used for building the TTS system, while for the spoke task, a multi-speaker source model is used for target speaker fine tuning. MuLanTTS achieves mean scores of quality assessment 4.3 and 4.5 in the respective tasks, statistically comparable with natural speech while keeping good similarity according to similarity assessment. The excellent quality and similarity in this year's new and dense statistical evaluation.

{{</citation>}}


## cs.CR (10)



### (32/104) MALITE: Lightweight Malware Detection and Classification for Constrained Devices (Sidharth Anand et al., 2023)

{{<citation>}}

Sidharth Anand, Barsha Mitra, Soumyadeep Dey, Abhinav Rao, Rupsa Dhar, Jaideep Vaidya. (2023)  
**MALITE: Lightweight Malware Detection and Classification for Constrained Devices**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2309.03294v1)  

---


**ABSTRACT**  
Today, malware is one of the primary cyberthreats to organizations. Malware has pervaded almost every type of computing device including the ones having limited memory, battery and computation power such as mobile phones, tablets and embedded devices like Internet-of-Things (IoT) devices. Consequently, the privacy and security of the malware infected systems and devices have been heavily jeopardized. In recent years, researchers have leveraged machine learning based strategies for malware detection and classification. Malware analysis approaches can only be employed in resource constrained environments if the methods are lightweight in nature. In this paper, we present MALITE, a lightweight malware analysis system, that can classify various malware families and distinguish between benign and malicious binaries. MALITE converts a binary into a gray scale or an RGB image and employs low memory and battery power consuming as well as computationally inexpensive malware analysis strategies. We have designed MALITE-MN, a lightweight neural network based architecture and MALITE-HRF, an ultra lightweight random forest based method that uses histogram features extracted by a sliding window. We evaluate the performance of both on six publicly available datasets (Malimg, Microsoft BIG, Dumpware10, MOTIF, Drebin and CICAndMal2017), and compare them to four state-of-the-art malware classification techniques. The results show that MALITE-MN and MALITE-HRF not only accurately identify and classify malware but also respectively consume several orders of magnitude lower resources (in terms of both memory as well as computation capabilities), making them much more suitable for resource constrained environments.

{{</citation>}}


### (33/104) Provably Unlinkable Smart Card-based Payments (Sergiu Bursuc et al., 2023)

{{<citation>}}

Sergiu Bursuc, Ross Horne, Sjouke Mauw, Semen Yurkov. (2023)  
**Provably Unlinkable Smart Card-based Payments**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Smart Car  
[Paper Link](http://arxiv.org/abs/2309.03128v1)  

---


**ABSTRACT**  
The most prevalent smart card-based payment method, EMV, currently offers no privacy to its users. Transaction details and the card number are sent in cleartext, enabling the profiling and tracking of cardholders. Since public awareness of privacy issues is growing and legislation, such as GDPR, is emerging, we believe it is necessary to investigate the possibility of making payments anonymous and unlinkable without compromising essential security guarantees and functional properties of EMV. This paper draws attention to trade-offs between functional and privacy requirements in the design of such a protocol. We present the UTX protocol - an enhanced payment protocol satisfying such requirements, and we formally certify key security and privacy properties using techniques based on the applied pi-calculus.

{{</citation>}}


### (34/104) ORL-AUDITOR: Dataset Auditing in Offline Deep Reinforcement Learning (Linkang Du et al., 2023)

{{<citation>}}

Linkang Du, Min Chen, Mingyang Sun, Shouling Ji, Peng Cheng, Jiming Chen, Zhikun Zhang. (2023)  
**ORL-AUDITOR: Dataset Auditing in Offline Deep Reinforcement Learning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI, Google, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.03081v1)  

---


**ABSTRACT**  
Data is a critical asset in AI, as high-quality datasets can significantly improve the performance of machine learning models. In safety-critical domains such as autonomous vehicles, offline deep reinforcement learning (offline DRL) is frequently used to train models on pre-collected datasets, as opposed to training these models by interacting with the real-world environment as the online DRL. To support the development of these models, many institutions make datasets publicly available with opensource licenses, but these datasets are at risk of potential misuse or infringement. Injecting watermarks to the dataset may protect the intellectual property of the data, but it cannot handle datasets that have already been published and is infeasible to be altered afterward. Other existing solutions, such as dataset inference and membership inference, do not work well in the offline DRL scenario due to the diverse model behavior characteristics and offline setting constraints. In this paper, we advocate a new paradigm by leveraging the fact that cumulative rewards can act as a unique identifier that distinguishes DRL models trained on a specific dataset. To this end, we propose ORL-AUDITOR, which is the first trajectory-level dataset auditing mechanism for offline RL scenarios. Our experiments on multiple offline DRL models and tasks reveal the efficacy of ORL-AUDITOR, with auditing accuracy over 95% and false positive rates less than 2.88%. We also provide valuable insights into the practical implementation of ORL-AUDITOR by studying various parameter settings. Furthermore, we demonstrate the auditing capability of ORL-AUDITOR on open-source datasets from Google and DeepMind, highlighting its effectiveness in auditing published datasets. ORL-AUDITOR is open-sourced at https://github.com/link-zju/ORL-Auditor.

{{</citation>}}


### (35/104) Disarming Steganography Attacks Inside Neural Network Models (Ran Dubin, 2023)

{{<citation>}}

Ran Dubin. (2023)  
**Disarming Steganography Attacks Inside Neural Network Models**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-MM, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.03071v1)  

---


**ABSTRACT**  
Similar to the revolution of open source code sharing, Artificial Intelligence (AI) model sharing is gaining increased popularity. However, the fast adaptation in the industry, lack of awareness, and ability to exploit the models make them significant attack vectors. By embedding malware in neurons, the malware can be delivered covertly, with minor or no impact on the neural network's performance. The covert attack will use the Least Significant Bits (LSB) weight attack since LSB has a minimal effect on the model accuracy, and as a result, the user will not notice it. Since there are endless ways to hide the attacks, we focus on a zero-trust prevention strategy based on AI model attack disarm and reconstruction. We proposed three types of model steganography weight disarm defense mechanisms. The first two are based on random bit substitution noise, and the other on model weight quantization. We demonstrate a 100\% prevention rate while the methods introduce a minimal decrease in model accuracy based on Qint8 and K-LRBP methods, which is an essential factor for improving AI security.

{{</citation>}}


### (36/104) Hide and Seek (HaS): A Lightweight Framework for Prompt Privacy Protection (Yu Chen et al., 2023)

{{<citation>}}

Yu Chen, Tingxin Li, Huiming Liu, Yang Yu. (2023)  
**Hide and Seek (HaS): A Lightweight Framework for Prompt Privacy Protection**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.03057v1)  

---


**ABSTRACT**  
Numerous companies have started offering services based on large language models (LLM), such as ChatGPT, which inevitably raises privacy concerns as users' prompts are exposed to the model provider. Previous research on secure reasoning using multi-party computation (MPC) has proven to be impractical for LLM applications due to its time-consuming and communication-intensive nature. While lightweight anonymization techniques can protect private information in prompts through substitution or masking, they fail to recover sensitive data replaced in the LLM-generated results. In this paper, we expand the application scenarios of anonymization techniques by training a small local model to de-anonymize the LLM's returned results with minimal computational overhead. We introduce the HaS framework, where "H(ide)" and "S(eek)" represent its two core processes: hiding private entities for anonymization and seeking private entities for de-anonymization, respectively. To quantitatively assess HaS's privacy protection performance, we propose both black-box and white-box adversarial models. Furthermore, we conduct experiments to evaluate HaS's usability in translation and classification tasks. The experimental findings demonstrate that the HaS framework achieves an optimal balance between privacy protection and utility.

{{</citation>}}


### (37/104) Automated CVE Analysis for Threat Prioritization and Impact Prediction (Ehsan Aghaei et al., 2023)

{{<citation>}}

Ehsan Aghaei, Ehab Al-Shaer, Waseem Shadid, Xi Niu. (2023)  
**Automated CVE Analysis for Threat Prioritization and Impact Prediction**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2309.03040v1)  

---


**ABSTRACT**  
The Common Vulnerabilities and Exposures (CVE) are pivotal information for proactive cybersecurity measures, including service patching, security hardening, and more. However, CVEs typically offer low-level, product-oriented descriptions of publicly disclosed cybersecurity vulnerabilities, often lacking the essential attack semantic information required for comprehensive weakness characterization and threat impact estimation. This critical insight is essential for CVE prioritization and the identification of potential countermeasures, particularly when dealing with a large number of CVEs. Current industry practices involve manual evaluation of CVEs to assess their attack severities using the Common Vulnerability Scoring System (CVSS) and mapping them to Common Weakness Enumeration (CWE) for potential mitigation identification. Unfortunately, this manual analysis presents a major bottleneck in the vulnerability analysis process, leading to slowdowns in proactive cybersecurity efforts and the potential for inaccuracies due to human errors. In this research, we introduce our novel predictive model and tool (called CVEDrill) which revolutionizes CVE analysis and threat prioritization. CVEDrill accurately estimates the CVSS vector for precise threat mitigation and priority ranking and seamlessly automates the classification of CVEs into the appropriate CWE hierarchy classes. By harnessing CVEDrill, organizations can now implement cybersecurity countermeasure mitigation with unparalleled accuracy and timeliness, surpassing in this domain the capabilities of state-of-the-art tools like ChaptGPT.

{{</citation>}}


### (38/104) Demystifying RCE Vulnerabilities in LLM-Integrated Apps (Tong Liu et al., 2023)

{{<citation>}}

Tong Liu, Zizhuang Deng, Guozhu Meng, Yuekang Li, Kai Chen. (2023)  
**Demystifying RCE Vulnerabilities in LLM-Integrated Apps**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.02926v1)  

---


**ABSTRACT**  
In recent years, Large Language Models (LLMs) have demonstrated remarkable potential across various downstream tasks. LLM-integrated frameworks, which serve as the essential infrastructure, have given rise to many LLM-integrated web apps. However, some of these frameworks suffer from Remote Code Execution (RCE) vulnerabilities, allowing attackers to execute arbitrary code on apps' servers remotely via prompt injections. Despite the severity of these vulnerabilities, no existing work has been conducted for a systematic investigation of them. This leaves a great challenge on how to detect vulnerabilities in frameworks as well as LLM-integrated apps in real-world scenarios.   To fill this gap, we present two novel strategies, including 1) a static analysis-based tool called LLMSmith to scan the source code of the framework to detect potential RCE vulnerabilities and 2) a prompt-based automated testing approach to verify the vulnerability in LLM-integrated web apps. We discovered 13 vulnerabilities in 6 frameworks, including 12 RCE vulnerabilities and 1 arbitrary file read/write vulnerability. 11 of them are confirmed by the framework developers, resulting in the assignment of 7 CVE IDs. After testing 51 apps, we found vulnerabilities in 17 apps, 16 of which are vulnerable to RCE and 1 to SQL injection. We responsibly reported all 17 issues to the corresponding developers and received acknowledgments. Furthermore, we amplify the attack impact beyond achieving RCE by allowing attackers to exploit other app users (e.g. app responses hijacking, user API key leakage) without direct interaction between the attacker and the victim. Lastly, we propose some mitigating strategies for improving the security awareness of both framework and app developers, helping them to mitigate these risks effectively.

{{</citation>}}


### (39/104) Autonomous and Collaborative Smart Home Security System (ACSHSS) (Hassan Jalil Hadi et al., 2023)

{{<citation>}}

Hassan Jalil Hadi, Khaleeq Un Nisa, Sheetal Harris. (2023)  
**Autonomous and Collaborative Smart Home Security System (ACSHSS)**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Intrusion Detection, Security  
[Paper Link](http://arxiv.org/abs/2309.02899v1)  

---


**ABSTRACT**  
Firstly, the proposed solution provides remotely accessible integrated IoT resources for the safety and security of the building. By using Sha ort Messaging System (SMS), the age is sent to the user by the Global System for Mobile (GSM) system. An SMS alert is sent to the user in case any sensor detects an abnormality in their operation. Secondly, an authentication mechanism is deployed to enable only authorized users to access resources. Thirdly, in case of a malicious approach in accessing IoT resources, a timely alert should be received by the owner. A Network Intrusion Detection System (NIDS) is deployed to detect and real-time information in case of any suspicious activity while accessing the Internet of Things network.

{{</citation>}}


### (40/104) CVE-driven Attack Technique Prediction with Semantic Information Extraction and a Domain-specific Language Model (Ehsan Aghaei et al., 2023)

{{<citation>}}

Ehsan Aghaei, Ehab Al-Shaer. (2023)  
**CVE-driven Attack Technique Prediction with Semantic Information Extraction and a Domain-specific Language Model**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: ChatGPT, GPT, Information Extraction, Language Model  
[Paper Link](http://arxiv.org/abs/2309.02785v1)  

---


**ABSTRACT**  
This paper addresses a critical challenge in cybersecurity: the gap between vulnerability information represented by Common Vulnerabilities and Exposures (CVEs) and the resulting cyberattack actions. CVEs provide insights into vulnerabilities, but often lack details on potential threat actions (tactics, techniques, and procedures, or TTPs) within the ATT&CK framework. This gap hinders accurate CVE categorization and proactive countermeasure initiation. The paper introduces the TTPpredictor tool, which uses innovative techniques to analyze CVE descriptions and infer plausible TTP attacks resulting from CVE exploitation. TTPpredictor overcomes challenges posed by limited labeled data and semantic disparities between CVE and TTP descriptions. It initially extracts threat actions from unstructured cyber threat reports using Semantic Role Labeling (SRL) techniques. These actions, along with their contextual attributes, are correlated with MITRE's attack functionality classes. This automated correlation facilitates the creation of labeled data, essential for categorizing novel threat actions into threat functionality classes and TTPs. The paper presents an empirical assessment, demonstrating TTPpredictor's effectiveness with accuracy rates of approximately 98% and F1-scores ranging from 95% to 98% in precise CVE classification to ATT&CK techniques. TTPpredictor outperforms state-of-the-art language model tools like ChatGPT. Overall, this paper offers a robust solution for linking CVEs to potential attack techniques, enhancing cybersecurity practitioners' ability to proactively identify and mitigate threats.

{{</citation>}}


### (41/104) Malicious Package Detection in NPM and PyPI using a Single Model of Malicious Behavior Sequence (Junan Zhang et al., 2023)

{{<citation>}}

Junan Zhang, Kaifeng Huang, Bihuan Chen, Chong Wang, Zhenhao Tian, Xin Peng. (2023)  
**Malicious Package Detection in NPM and PyPI using a Single Model of Malicious Behavior Sequence**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.02637v1)  

---


**ABSTRACT**  
Open-source software (OSS) supply chain enlarges the attack surface, which makes package registries attractive targets for attacks. Recently, package registries NPM and PyPI have been flooded with malicious packages. The effectiveness of existing malicious NPM and PyPI package detection approaches is hindered by two challenges. The first challenge is how to leverage the knowledge of malicious packages from different ecosystems in a unified way such that multi-lingual malicious package detection can be feasible. The second challenge is how to model malicious behavior in a sequential way such that maliciousness can be precisely captured. To address the two challenges, we propose and implement Cerebro to detect malicious packages in NPM and PyPI. We curate a feature set based on a high-level abstraction of malicious behavior to enable multi-lingual knowledge fusing. We organize extracted features into a behavior sequence to model sequential malicious behavior. We fine-tune the BERT model to understand the semantics of malicious behavior. Extensive evaluation has demonstrated the effectiveness of Cerebro over the state-of-the-art as well as the practically acceptable efficiency. Cerebro has successfully detected 306 and 196 new malicious packages in PyPI and NPM, and received 385 thank letters from the official PyPI and NPM teams.

{{</citation>}}


## cs.LG (22)



### (42/104) Blink: Link Local Differential Privacy in Graph Neural Networks via Bayesian Estimation (Xiaochen Zhu et al., 2023)

{{<citation>}}

Xiaochen Zhu, Vincent Y. F. Tan, Xiaokui Xiao. (2023)  
**Blink: Link Local Differential Privacy in Graph Neural Networks via Bayesian Estimation**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.03190v2)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have gained an increasing amount of popularity due to their superior capability in learning node embeddings for various graph inference tasks, but training them can raise privacy concerns. To address this, we propose using link local differential privacy over decentralized nodes, enabling collaboration with an untrusted server to train GNNs without revealing the existence of any link. Our approach spends the privacy budget separately on links and degrees of the graph for the server to better denoise the graph topology using Bayesian estimation, alleviating the negative impact of LDP on the accuracy of the trained GNNs. We bound the mean absolute error of the inferred link probabilities against the ground truth graph topology. We then propose two variants of our LDP mechanism complementing each other in different privacy settings, one of which estimates fewer links under lower privacy budgets to avoid false positive link estimates when the uncertainty is high, while the other utilizes more information and performs better given relatively higher privacy budgets. Furthermore, we propose a hybrid variant that combines both strategies and is able to perform better across different privacy budgets. Extensive experiments show that our approach outperforms existing methods in terms of accuracy under varying privacy budgets.

{{</citation>}}


### (43/104) Using Multiple Vector Channels Improves E(n)-Equivariant Graph Neural Networks (Daniel Levy et al., 2023)

{{<citation>}}

Daniel Levy, Sékou-Oumar Kaba, Carmelo Gonzales, Santiago Miret, Siamak Ravanbakhsh. (2023)  
**Using Multiple Vector Channels Improves E(n)-Equivariant Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.03139v1)  

---


**ABSTRACT**  
We present a natural extension to E(n)-equivariant graph neural networks that uses multiple equivariant vectors per node. We formulate the extension and show that it improves performance across different physical systems benchmark tasks, with minimal differences in runtime or number of parameters. The proposed multichannel EGNN outperforms the standard singlechannel EGNN on N-body charged particle dynamics, molecular property predictions, and predicting the trajectories of solar system bodies. Given the additional benefits and minimal additional cost of multi-channel EGNN, we suggest that this extension may be of practical use to researchers working in machine learning for the physical sciences

{{</citation>}}


### (44/104) Theoretical Explanation of Activation Sparsity through Flat Minima and Adversarial Robustness (Ze Peng et al., 2023)

{{<citation>}}

Ze Peng, Lei Qi, Yinghuan Shi, Yang Gao. (2023)  
**Theoretical Explanation of Activation Sparsity through Flat Minima and Adversarial Robustness**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.03004v1)  

---


**ABSTRACT**  
A recent empirical observation of activation sparsity in MLP layers offers an opportunity to drastically reduce computation costs for free. Despite several works attributing it to training dynamics, the theoretical explanation of activation sparsity's emergence is restricted to shallow networks, small training steps well as modified training, even though the sparsity has been found in deep models trained by vanilla protocols for large steps. To fill the three gaps, we propose the notion of gradient sparsity as the source of activation sparsity and a theoretical explanation based on it that explains gradient sparsity and then activation sparsity as necessary steps to adversarial robustness w.r.t. hidden features and parameters, which is approximately the flatness of minima for well-learned models. The theory applies to standardly trained LayerNorm-ed pure MLPs, and further to Transformers or other architectures if noises are added to weights during training. To eliminate other sources of flatness when arguing sparsities' necessity, we discover the phenomenon of spectral concentration, i.e., the ratio between the largest and the smallest non-zero singular values of weight matrices is small. We utilize random matrix theory (RMT) as a powerful theoretical tool to analyze stochastic gradient noises and discuss the emergence of spectral concentration. With these insights, we propose two plug-and-play modules for both training from scratch and sparsity finetuning, as well as one radical modification that only applies to from-scratch training. Another under-testing module for both sparsity and flatness is also immediate from our theories. Validational experiments are conducted to verify our explanation. Experiments for productivity demonstrate modifications' improvement in sparsity, indicating further theoretical cost reduction in both training and inference.

{{</citation>}}


### (45/104) DECODE: Data-driven Energy Consumption Prediction leveraging Historical Data and Environmental Factors in Buildings (Aditya Mishra et al., 2023)

{{<citation>}}

Aditya Mishra, Haroon R. Lone, Aayush Mishra. (2023)  
**DECODE: Data-driven Energy Consumption Prediction leveraging Historical Data and Environmental Factors in Buildings**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.02908v1)  

---


**ABSTRACT**  
Energy prediction in buildings plays a crucial role in effective energy management. Precise predictions are essential for achieving optimal energy consumption and distribution within the grid. This paper introduces a Long Short-Term Memory (LSTM) model designed to forecast building energy consumption using historical energy data, occupancy patterns, and weather conditions. The LSTM model provides accurate short, medium, and long-term energy predictions for residential and commercial buildings compared to existing prediction models. We compare our LSTM model with established prediction methods, including linear regression, decision trees, and random forest. Encouragingly, the proposed LSTM model emerges as the superior performer across all metrics. It demonstrates exceptional prediction accuracy, boasting the highest R2 score of 0.97 and the most favorable mean absolute error (MAE) of 0.007. An additional advantage of our developed model is its capacity to achieve efficient energy consumption forecasts even when trained on a limited dataset. We address concerns about overfitting (variance) and underfitting (bias) through rigorous training and evaluation on real-world data. In summary, our research contributes to energy prediction by offering a robust LSTM model that outperforms alternative methods and operates with remarkable efficiency, generalizability, and reliability.

{{</citation>}}


### (46/104) Rethinking Momentum Knowledge Distillation in Online Continual Learning (Nicolas Michel et al., 2023)

{{<citation>}}

Nicolas Michel, Maorong Wang, Ling Xiao, Toshihiko Yamasaki. (2023)  
**Rethinking Momentum Knowledge Distillation in Online Continual Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: ImageNet, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.02870v1)  

---


**ABSTRACT**  
Online Continual Learning (OCL) addresses the problem of training neural networks on a continuous data stream where multiple classification tasks emerge in sequence. In contrast to offline Continual Learning, data can be seen only once in OCL. In this context, replay-based strategies have achieved impressive results and most state-of-the-art approaches are heavily depending on them. While Knowledge Distillation (KD) has been extensively used in offline Continual Learning, it remains under-exploited in OCL, despite its potential. In this paper, we theoretically analyze the challenges in applying KD to OCL. We introduce a direct yet effective methodology for applying Momentum Knowledge Distillation (MKD) to many flagship OCL methods and demonstrate its capabilities to enhance existing approaches. In addition to improving existing state-of-the-arts accuracy by more than $10\%$ points on ImageNet100, we shed light on MKD internal mechanics and impacts during training in OCL. We argue that similar to replay, MKD should be considered a central component of OCL.

{{</citation>}}


### (47/104) On Reducing Undesirable Behavior in Deep Reinforcement Learning Models (Ophir Carmel et al., 2023)

{{<citation>}}

Ophir Carmel, Guy Katz. (2023)  
**On Reducing Undesirable Behavior in Deep Reinforcement Learning Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.02869v1)  

---


**ABSTRACT**  
Deep reinforcement learning (DRL) has proven extremely useful in a large variety of application domains. However, even successful DRL-based software can exhibit highly undesirable behavior. This is due to DRL training being based on maximizing a reward function, which typically captures general trends but cannot precisely capture, or rule out, certain behaviors of the system. In this paper, we propose a novel framework aimed at drastically reducing the undesirable behavior of DRL-based software, while maintaining its excellent performance. In addition, our framework can assist in providing engineers with a comprehensible characterization of such undesirable behavior. Under the hood, our approach is based on extracting decision tree classifiers from erroneous state-action pairs, and then integrating these trees into the DRL training loop, penalizing the system whenever it performs an error. We provide a proof-of-concept implementation of our approach, and use it to evaluate the technique on three significant case studies. We find that our approach can extend existing frameworks in a straightforward manner, and incurs only a slight overhead in training time. Further, it incurs only a very slight hit to performance, or even in some cases - improves it, while significantly reducing the frequency of undesirable behavior.

{{</citation>}}


### (48/104) A Critical Review of Common Log Data Sets Used for Evaluation of Sequence-based Anomaly Detection Techniques (Max Landauer et al., 2023)

{{<citation>}}

Max Landauer, Florian Skopik, Markus Wurzenberger. (2023)  
**A Critical Review of Common Log Data Sets Used for Evaluation of Sequence-based Anomaly Detection Techniques**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2309.02854v1)  

---


**ABSTRACT**  
Log data store event execution patterns that correspond to underlying workflows of systems or applications. While most logs are informative, log data also include artifacts that indicate failures or incidents. Accordingly, log data are often used to evaluate anomaly detection techniques that aim to automatically disclose unexpected or otherwise relevant system behavior patterns. Recently, detection approaches leveraging deep learning have increasingly focused on anomalies that manifest as changes of sequential patterns within otherwise normal event traces. Several publicly available data sets, such as HDFS, BGL, Thunderbird, OpenStack, and Hadoop, have since become standards for evaluating these anomaly detection techniques, however, the appropriateness of these data sets has not been closely investigated in the past. In this paper we therefore analyze six publicly available log data sets with focus on the manifestations of anomalies and simple techniques for their detection. Our findings suggest that most anomalies are not directly related to sequential manifestations and that advanced detection techniques are not required to achieve high detection rates on these data sets.

{{</citation>}}


### (49/104) Combining Thermodynamics-based Model of the Centrifugal Compressors and Active Machine Learning for Enhanced Industrial Design Optimization (Shadi Ghiasi et al., 2023)

{{<citation>}}

Shadi Ghiasi, Guido Pazzi, Concettina Del Grosso, Giovanni De Magistris, Giacomo Veneri. (2023)  
**Combining Thermodynamics-based Model of the Centrifugal Compressors and Active Machine Learning for Enhanced Industrial Design Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2309.02818v1)  

---


**ABSTRACT**  
The design process of centrifugal compressors requires applying an optimization process which is computationally expensive due to complex analytical equations underlying the compressor's dynamical equations. Although the regression surrogate models could drastically reduce the computational cost of such a process, the major challenge is the scarcity of data for training the surrogate model. Aiming to strategically exploit the labeled samples, we propose the Active-CompDesign framework in which we combine a thermodynamics-based compressor model (i.e., our internal software for compressor design) and Gaussian Process-based surrogate model within a deployable Active Learning (AL) setting. We first conduct experiments in an offline setting and further, extend it to an online AL framework where a real-time interaction with the thermodynamics-based compressor's model allows the deployment in production. ActiveCompDesign shows a significant performance improvement in surrogate modeling by leveraging on uncertainty-based query function of samples within the AL framework with respect to the random selection of data points. Moreover, our framework in production has reduced the total computational time of compressor's design optimization to around 46% faster than relying on the internal thermodynamics-based simulator, achieving the same performance.

{{</citation>}}


### (50/104) Norm Tweaking: High-performance Low-bit Quantization of Large Language Models (Liang Li et al., 2023)

{{<citation>}}

Liang Li, Qingyuan Li, Bo Zhang, Xiangxiang Chu. (2023)  
**Norm Tweaking: High-performance Low-bit Quantization of Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GLM, GPT, Language Model, Quantization  
[Paper Link](http://arxiv.org/abs/2309.02784v1)  

---


**ABSTRACT**  
As the size of large language models (LLMs) continues to grow, model compression without sacrificing accuracy has become a crucial challenge for deployment. While some quantization methods, such as GPTQ, have made progress in achieving acceptable 4-bit weight-only quantization, attempts at lower bit quantization often result in severe performance degradation. In this paper, we introduce a technique called norm tweaking, which can be used as a plugin in current PTQ methods to achieve high precision while being cost-efficient. Our approach is inspired by the observation that rectifying the quantized activation distribution to match its float counterpart can readily restore accuracy for LLMs. To achieve this, we carefully design a tweaking strategy that includes calibration data generation and channel-wise distance constraint to update the weights of normalization layers for better generalization. We conduct extensive experiments on various datasets using several open-sourced LLMs. Our method demonstrates significant improvements in both weight-only quantization and joint quantization of weights and activations, surpassing existing PTQ methods. On GLM-130B and OPT-66B, our method even achieves the same level of accuracy at 2-bit quantization as their float ones. Our simple and effective approach makes it more practical for real-world applications.

{{</citation>}}


### (51/104) Unifying over-smoothing and over-squashing in graph neural networks: A physics informed approach and beyond (Zhiqi Shao et al., 2023)

{{<citation>}}

Zhiqi Shao, Dai Shi, Andi Han, Yi Guo, Qibin Zhao, Junbin Gao. (2023)  
**Unifying over-smoothing and over-squashing in graph neural networks: A physics informed approach and beyond**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.02769v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have emerged as one of the leading approaches for machine learning on graph-structured data. Despite their great success, critical computational challenges such as over-smoothing, over-squashing, and limited expressive power continue to impact the performance of GNNs. In this study, inspired from the time-reversal principle commonly utilized in classical and quantum physics, we reverse the time direction of the graph heat equation. The resulted reversing process yields a class of high pass filtering functions that enhance the sharpness of graph node features. Leveraging this concept, we introduce the Multi-Scaled Heat Kernel based GNN (MHKG) by amalgamating diverse filtering functions' effects on node features. To explore more flexible filtering conditions, we further generalize MHKG into a model termed G-MHKG and thoroughly show the roles of each element in controlling over-smoothing, over-squashing and expressive power. Notably, we illustrate that all aforementioned issues can be characterized and analyzed via the properties of the filtering functions, and uncover a trade-off between over-smoothing and over-squashing: enhancing node feature sharpness will make model suffer more from over-squashing, and vice versa. Furthermore, we manipulate the time again to show how G-MHKG can handle both two issues under mild conditions. Our conclusive experiments highlight the effectiveness of proposed models. It surpasses several GNN baseline models in performance across graph datasets characterized by both homophily and heterophily.

{{</citation>}}


### (52/104) Towards Unsupervised Graph Completion Learning on Graphs with Features and Structure Missing (Sichao Fu et al., 2023)

{{<citation>}}

Sichao Fu, Qinmu Peng, Yang He, Baokun Du, Xinge You. (2023)  
**Towards Unsupervised Graph Completion Learning on Graphs with Features and Structure Missing**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2309.02762v1)  

---


**ABSTRACT**  
In recent years, graph neural networks (GNN) have achieved significant developments in a variety of graph analytical tasks. Nevertheless, GNN's superior performance will suffer from serious damage when the collected node features or structure relationships are partially missing owning to numerous unpredictable factors. Recently emerged graph completion learning (GCL) has received increasing attention, which aims to reconstruct the missing node features or structure relationships under the guidance of a specifically supervised task. Although these proposed GCL methods have made great success, they still exist the following problems: the reliance on labels, the bias of the reconstructed node features and structure relationships. Besides, the generalization ability of the existing GCL still faces a huge challenge when both collected node features and structure relationships are partially missing at the same time. To solve the above issues, we propose a more general GCL framework with the aid of self-supervised learning for improving the task performance of the existing GNN variants on graphs with features and structure missing, termed unsupervised GCL (UGCL). Specifically, to avoid the mismatch between missing node features and structure during the message-passing process of GNN, we separate the feature reconstruction and structure reconstruction and design its personalized model in turn. Then, a dual contrastive loss on the structure level and feature level is introduced to maximize the mutual information of node representations from feature reconstructing and structure reconstructing paths for providing more supervision signals. Finally, the reconstructed node features and structure can be applied to the downstream node classification task. Extensive experiments on eight datasets, three GNN variants and five missing rates demonstrate the effectiveness of our proposed method.

{{</citation>}}


### (53/104) GPT Can Solve Mathematical Problems Without a Calculator (Zhen Yang et al., 2023)

{{<citation>}}

Zhen Yang, Ming Ding, Qingsong Lv, Zhihuan Jiang, Zehai He, Yuyi Guo, Jinfeng Bai, Jie Tang. (2023)  
**GPT Can Solve Mathematical Problems Without a Calculator**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: GLM, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.03241v1)  

---


**ABSTRACT**  
Previous studies have typically assumed that large language models are unable to accurately perform arithmetic operations, particularly multiplication of >8 digits, and operations involving decimals and fractions, without the use of calculator tools. This paper aims to challenge this misconception. With sufficient training data, a 2 billion-parameter language model can accurately perform multi-digit arithmetic operations with almost 100% accuracy without data leakage, significantly surpassing GPT-4 (whose multi-digit multiplication accuracy is only 4.3%). We also demonstrate that our MathGLM, fine-tuned from GLM-10B on a dataset with additional multi-step arithmetic operations and math problems described in text, achieves similar performance to GPT-4 on a 5,000-samples Chinese math problem test set.

{{</citation>}}


### (54/104) SWAP: Exploiting Second-Ranked Logits for Adversarial Attacks on Time Series (Chang George Dong et al., 2023)

{{<citation>}}

Chang George Dong, Liangwei Nathan Zheng, Weitong Chen, Wei Emma Zhang, Lin Yue. (2023)  
**SWAP: Exploiting Second-Ranked Logits for Adversarial Attacks on Time Series**  

---
Primary Category: cs.LG  
Categories: I-2-0, cs-AI, cs-CR, cs-LG, cs.LG  
Keywords: Adversarial Attack, Time Series  
[Paper Link](http://arxiv.org/abs/2309.02752v1)  

---


**ABSTRACT**  
Time series classification (TSC) has emerged as a critical task in various domains, and deep neural models have shown superior performance in TSC tasks. However, these models are vulnerable to adversarial attacks, where subtle perturbations can significantly impact the prediction results. Existing adversarial methods often suffer from over-parameterization or random logit perturbation, hindering their effectiveness. Additionally, increasing the attack success rate (ASR) typically involves generating more noise, making the attack more easily detectable. To address these limitations, we propose SWAP, a novel attacking method for TSC models. SWAP focuses on enhancing the confidence of the second-ranked logits while minimizing the manipulation of other logits. This is achieved by minimizing the Kullback-Leibler divergence between the target logit distribution and the predictive logit distribution. Experimental results demonstrate that SWAP achieves state-of-the-art performance, with an ASR exceeding 50% and an 18% increase compared to existing methods.

{{</citation>}}


### (55/104) Unveiling the frontiers of deep learning: innovations shaping diverse domains (Shams Forruque Ahmed et al., 2023)

{{<citation>}}

Shams Forruque Ahmed, Md. Sakib Bin Alam, Maliha Kabir, Shaila Afrin, Sabiha Jannat Rafa, Aanushka Mehjabin, Amir H. Gandomi. (2023)  
**Unveiling the frontiers of deep learning: innovations shaping diverse domains**  

---
Primary Category: cs.LG  
Categories: 68T07, cs-AI, cs-LG, cs-NE, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.02712v1)  

---


**ABSTRACT**  
Deep learning (DL) enables the development of computer models that are capable of learning, visualizing, optimizing, refining, and predicting data. In recent years, DL has been applied in a range of fields, including audio-visual data processing, agriculture, transportation prediction, natural language, biomedicine, disaster management, bioinformatics, drug design, genomics, face recognition, and ecology. To explore the current state of deep learning, it is necessary to investigate the latest developments and applications of deep learning in these disciplines. However, the literature is lacking in exploring the applications of deep learning in all potential sectors. This paper thus extensively investigates the potential applications of deep learning across all major fields of study as well as the associated benefits and challenges. As evidenced in the literature, DL exhibits accuracy in prediction and analysis, makes it a powerful computational tool, and has the ability to articulate itself and optimize, making it effective in processing data with no prior training. Given its independence from training data, deep learning necessitates massive amounts of data for effective analysis and processing, much like data volume. To handle the challenge of compiling huge amounts of medical, scientific, healthcare, and environmental data for use in deep learning, gated architectures like LSTMs and GRUs can be utilized. For multimodal learning, shared neurons in the neural network for all activities and specialized neurons for particular tasks are necessary.

{{</citation>}}


### (56/104) Spatio-Temporal Contrastive Self-Supervised Learning for POI-level Crowd Flow Inference (Songyu Ke et al., 2023)

{{<citation>}}

Songyu Ke, Ting Li, Li Song, Yanping Sun, Qintian Sun, Junbo Zhang, Yu Zheng. (2023)  
**Spatio-Temporal Contrastive Self-Supervised Learning for POI-level Crowd Flow Inference**  

---
Primary Category: cs.LG  
Categories: I-2, cs-AI, cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.03239v1)  

---


**ABSTRACT**  
Accurate acquisition of crowd flow at Points of Interest (POIs) is pivotal for effective traffic management, public service, and urban planning. Despite this importance, due to the limitations of urban sensing techniques, the data quality from most sources is inadequate for monitoring crowd flow at each POI. This renders the inference of accurate crowd flow from low-quality data a critical and challenging task. The complexity is heightened by three key factors: 1) \emph{The scarcity and rarity of labeled data}, 2) \emph{The intricate spatio-temporal dependencies among POIs}, and 3) \emph{The myriad correlations between precise crowd flow and GPS reports}.   To address these challenges, we recast the crowd flow inference problem as a self-supervised attributed graph representation learning task and introduce a novel \underline{C}ontrastive \underline{S}elf-learning framework for \underline{S}patio-\underline{T}emporal data (\model). Our approach initiates with the construction of a spatial adjacency graph founded on the POIs and their respective distances. We then employ a contrastive learning technique to exploit large volumes of unlabeled spatio-temporal data. We adopt a swapped prediction approach to anticipate the representation of the target subgraph from similar instances. Following the pre-training phase, the model is fine-tuned with accurate crowd flow data. Our experiments, conducted on two real-world datasets, demonstrate that the \model pre-trained on extensive noisy data consistently outperforms models trained from scratch.

{{</citation>}}


### (57/104) Implicit Design Choices and Their Impact on Emotion Recognition Model Development and Evaluation (Mimansa Jaiswal, 2023)

{{<citation>}}

Mimansa Jaiswal. (2023)  
**Implicit Design Choices and Their Impact on Emotion Recognition Model Development and Evaluation**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs-SD, cs.LG, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2309.03238v1)  

---


**ABSTRACT**  
Emotion recognition is a complex task due to the inherent subjectivity in both the perception and production of emotions. The subjectivity of emotions poses significant challenges in developing accurate and robust computational models. This thesis examines critical facets of emotion recognition, beginning with the collection of diverse datasets that account for psychological factors in emotion production.   To handle the challenge of non-representative training data, this work collects the Multimodal Stressed Emotion dataset, which introduces controlled stressors during data collection to better represent real-world influences on emotion production. To address issues with label subjectivity, this research comprehensively analyzes how data augmentation techniques and annotation schemes impact emotion perception and annotator labels. It further handles natural confounding variables and variations by employing adversarial networks to isolate key factors like stress from learned emotion representations during model training. For tackling concerns about leakage of sensitive demographic variables, this work leverages adversarial learning to strip sensitive demographic information from multimodal encodings. Additionally, it proposes optimized sociological evaluation metrics aligned with cost-effective, real-world needs for model testing.   This research advances robust, practical emotion recognition through multifaceted studies of challenges in datasets, labels, modeling, demographic and membership variable encoding in representations, and evaluation. The groundwork has been laid for cost-effective, generalizable emotion recognition models that are less likely to encode sensitive demographic information.

{{</citation>}}


### (58/104) RLSynC: Offline-Online Reinforcement Learning for Synthon Completion (Frazier N. Baker et al., 2023)

{{<citation>}}

Frazier N. Baker, Ziqi Chen, Xia Ning. (2023)  
**RLSynC: Offline-Online Reinforcement Learning for Synthon Completion**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.02671v1)  

---


**ABSTRACT**  
Retrosynthesis is the process of determining the set of reactant molecules that can react to form a desired product. Semi-template-based retrosynthesis methods, which imitate the reverse logic of synthesis reactions, first predict the reaction centers in the products, and then complete the resulting synthons back into reactants. These methods enable necessary interpretability and high practical utility to inform synthesis planning. We develop a new offline-online reinforcement learning method RLSynC for synthon completion in semi-template-based methods. RLSynC assigns one agent to each synthon, all of which complete the synthons by conducting actions step by step in a synchronized fashion. RLSynC learns the policy from both offline training episodes and online interactions which allow RLSynC to explore new reaction spaces. RLSynC uses a forward synthesis model to evaluate the likelihood of the predicted reactants in synthesizing a product, and thus guides the action search. We compare RLSynC with the state-of-the-art retrosynthesis methods. Our experimental results demonstrate that RLSynC can outperform these methods with improvement as high as 14.9% on synthon completion, and 14.0% on retrosynthesis, highlighting its potential in synthesis planning.

{{</citation>}}


### (59/104) Marketing Budget Allocation with Offline Constrained Deep Reinforcement Learning (Tianchi Cai et al., 2023)

{{<citation>}}

Tianchi Cai, Jiyan Jiang, Wenpeng Zhang, Shiji Zhou, Xierui Song, Li Yu, Lihong Gu, Xiaodong Zeng, Jinjie Gu, Guannan Zhang. (2023)  
**Marketing Budget Allocation with Offline Constrained Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.02669v1)  

---


**ABSTRACT**  
We study the budget allocation problem in online marketing campaigns that utilize previously collected offline data. We first discuss the long-term effect of optimizing marketing budget allocation decisions in the offline setting. To overcome the challenge, we propose a novel game-theoretic offline value-based reinforcement learning method using mixed policies. The proposed method reduces the need to store infinitely many policies in previous methods to only constantly many policies, which achieves nearly optimal policy efficiency, making it practical and favorable for industrial usage. We further show that this method is guaranteed to converge to the optimal policy, which cannot be achieved by previous value-based reinforcement learning methods for marketing budget allocation. Our experiments on a large-scale marketing campaign with tens-of-millions users and more than one billion budget verify the theoretical results and show that the proposed method outperforms various baseline methods. The proposed method has been successfully deployed to serve all the traffic of this marketing campaign.

{{</citation>}}


### (60/104) Contrastive Learning as Kernel Approximation (Konstantinos Christopher Tsiolis, 2023)

{{<citation>}}

Konstantinos Christopher Tsiolis. (2023)  
**Contrastive Learning as Kernel Approximation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.02651v1)  

---


**ABSTRACT**  
In standard supervised machine learning, it is necessary to provide a label for every input in the data. While raw data in many application domains is easily obtainable on the Internet, manual labelling of this data is prohibitively expensive. To circumvent this issue, contrastive learning methods produce low-dimensional vector representations (also called features) of high-dimensional inputs on large unlabelled datasets. This is done by training with a contrastive loss function, which enforces that similar inputs have high inner product and dissimilar inputs have low inner product in the feature space. Rather than annotating each input individually, it suffices to define a means of sampling pairs of similar and dissimilar inputs. Contrastive features can then be fed as inputs to supervised learning systems on much smaller labelled datasets to obtain high accuracy on end tasks of interest.   The goal of this thesis is to provide an overview of the current theoretical understanding of contrastive learning, specifically as it pertains to the minimizers of contrastive loss functions and their relationship to prior methods for learning features from unlabelled data. We highlight popular contrastive loss functions whose minimizers implicitly approximate a positive semidefinite (PSD) kernel. The latter is a well-studied object in functional analysis and learning theory that formalizes a notion of similarity between elements of a space. PSD kernels provide an implicit definition of features through the theory of reproducing kernel Hilbert spaces.

{{</citation>}}


### (61/104) TFBEST: Dual-Aspect Transformer with Learnable Positional Encoding for Failure Prediction (Rohan Mohapatra et al., 2023)

{{<citation>}}

Rohan Mohapatra, Saptarshi Sengupta. (2023)  
**TFBEST: Dual-Aspect Transformer with Learnable Positional Encoding for Failure Prediction**  

---
Primary Category: cs.LG  
Categories: I-2-0, cs-AI, cs-LG, cs.LG  
Keywords: LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2309.02641v1)  

---


**ABSTRACT**  
Hard Disk Drive (HDD) failures in datacenters are costly - from catastrophic data loss to a question of goodwill, stakeholders want to avoid it like the plague. An important tool in proactively monitoring against HDD failure is timely estimation of the Remaining Useful Life (RUL). To this end, the Self-Monitoring, Analysis and Reporting Technology employed within HDDs (S.M.A.R.T.) provide critical logs for long-term maintenance of the security and dependability of these essential data storage devices. Data-driven predictive models in the past have used these S.M.A.R.T. logs and CNN/RNN based architectures heavily. However, they have suffered significantly in providing a confidence interval around the predicted RUL values as well as in processing very long sequences of logs. In addition, some of these approaches, such as those based on LSTMs, are inherently slow to train and have tedious feature engineering overheads. To overcome these challenges, in this work we propose a novel transformer architecture - a Temporal-fusion Bi-encoder Self-attention Transformer (TFBEST) for predicting failures in hard-drives. It is an encoder-decoder based deep learning technique that enhances the context gained from understanding health statistics sequences and predicts a sequence of the number of days remaining before a disk potentially fails. In this paper, we also provide a novel confidence margin statistic that can help manufacturers replace a hard-drive within a time frame. Experiments on Seagate HDD data show that our method significantly outperforms the state-of-the-art RUL prediction methods during testing over the exhaustive 10-year data from Backblaze (2013-present). Although validated on HDD failure prediction, the TFBEST architecture is well-suited for other prognostics applications and may be adapted for allied regression problems.

{{</citation>}}


### (62/104) Epi-Curriculum: Episodic Curriculum Learning for Low-Resource Domain Adaptation in Neural Machine Translation (Keyu Chen et al., 2023)

{{<citation>}}

Keyu Chen, Di Zhuang, Mingchen Li, J. Morris Chang. (2023)  
**Epi-Curriculum: Episodic Curriculum Learning for Low-Resource Domain Adaptation in Neural Machine Translation**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Low-Resource, Machine Translation  
[Paper Link](http://arxiv.org/abs/2309.02640v1)  

---


**ABSTRACT**  
Neural Machine Translation (NMT) models have become successful, but their performance remains poor when translating on new domains with a limited number of data. In this paper, we present a novel approach Epi-Curriculum to address low-resource domain adaptation (DA), which contains a new episodic training framework along with denoised curriculum learning. Our episodic training framework enhances the model's robustness to domain shift by episodically exposing the encoder/decoder to an inexperienced decoder/encoder. The denoised curriculum learning filters the noised data and further improves the model's adaptability by gradually guiding the learning process from easy to more difficult tasks. Experiments on English-German and English-Romanian translation show that: (i) Epi-Curriculum improves both model's robustness and adaptability in seen and unseen domains; (ii) Our episodic training framework enhances the encoder and decoder's robustness to domain shift.

{{</citation>}}


### (63/104) Deep Reinforcement Learning from Hierarchical Weak Preference Feedback (Alexander Bukharin et al., 2023)

{{<citation>}}

Alexander Bukharin, Yixiao Li, Pengcheng He, Weizhu Chen, Tuo Zhao. (2023)  
**Deep Reinforcement Learning from Hierarchical Weak Preference Feedback**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.02632v1)  

---


**ABSTRACT**  
Reward design is a fundamental, yet challenging aspect of practical reinforcement learning (RL). For simple tasks, researchers typically handcraft the reward function, e.g., using a linear combination of several reward factors. However, such reward engineering is subject to approximation bias, incurs large tuning cost, and often cannot provide the granularity required for complex tasks. To avoid these difficulties, researchers have turned to reinforcement learning from human feedback (RLHF), which learns a reward function from human preferences between pairs of trajectory sequences. By leveraging preference-based reward modeling, RLHF learns complex rewards that are well aligned with human preferences, allowing RL to tackle increasingly difficult problems. Unfortunately, the applicability of RLHF is limited due to the high cost and difficulty of obtaining human preference data. In light of this cost, we investigate learning reward functions for complex tasks with less human effort; simply by ranking the importance of the reward factors. More specifically, we propose a new RL framework -- HERON, which compares trajectories using a hierarchical decision tree induced by the given ranking. These comparisons are used to train a preference-based reward model, which is then used for policy learning. We find that our framework can not only train high performing agents on a variety of difficult tasks, but also provide additional benefits such as improved sample efficiency and robustness. Our code is available at https://github.com/abukharin3/HERON.

{{</citation>}}


## eess.IV (5)



### (64/104) 3D Transformer based on deformable patch location for differential diagnosis between Alzheimer's disease and Frontotemporal dementia (Huy-Dung Nguyen et al., 2023)

{{<citation>}}

Huy-Dung Nguyen, Michaël Clément, Boris Mansencal, Pierrick Coupé. (2023)  
**3D Transformer based on deformable patch location for differential diagnosis between Alzheimer's disease and Frontotemporal dementia**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.03183v1)  

---


**ABSTRACT**  
Alzheimer's disease and Frontotemporal dementia are common types of neurodegenerative disorders that present overlapping clinical symptoms, making their differential diagnosis very challenging. Numerous efforts have been done for the diagnosis of each disease but the problem of multi-class differential diagnosis has not been actively explored. In recent years, transformer-based models have demonstrated remarkable success in various computer vision tasks. However, their use in disease diagnostic is uncommon due to the limited amount of 3D medical data given the large size of such models. In this paper, we present a novel 3D transformer-based architecture using a deformable patch location module to improve the differential diagnosis of Alzheimer's disease and Frontotemporal dementia. Moreover, to overcome the problem of data scarcity, we propose an efficient combination of various data augmentation techniques, adapted for training transformer-based models on 3D structural magnetic resonance imaging data. Finally, we propose to combine our transformer-based model with a traditional machine learning model using brain structure volumes to better exploit the available data. Our experiments demonstrate the effectiveness of the proposed approach, showing competitive results compared to state-of-the-art methods. Moreover, the deformable patch locations can be visualized, revealing the most relevant brain regions used to establish the diagnosis of each disease.

{{</citation>}}


### (65/104) EGIC: Enhanced Low-Bit-Rate Generative Image Compression Guided by Semantic Segmentation (Nikolai Körber et al., 2023)

{{<citation>}}

Nikolai Körber, Eduard Kromer, Andreas Siebert, Sascha Hauke, Daniel Mueller-Gritschneder. (2023)  
**EGIC: Enhanced Low-Bit-Rate Generative Image Compression Guided by Semantic Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.03244v1)  

---


**ABSTRACT**  
We introduce EGIC, a novel generative image compression method that allows traversing the distortion-perception curve efficiently from a single model. Specifically, we propose an implicitly encoded variant of image interpolation that predicts the residual between a MSE-optimized and GAN-optimized decoder output. On the receiver side, the user can then control the impact of the residual on the GAN-based reconstruction. Together with improved GAN-based building blocks, EGIC outperforms a wide-variety of perception-oriented and distortion-oriented baselines, including HiFiC, MRIC and DIRAC, while performing almost on par with VTM-20.0 on the distortion end. EGIC is simple to implement, very lightweight (e.g. 0.18x model parameters compared to HiFiC) and provides excellent interpolation characteristics, which makes it a promising candidate for practical applications targeting the low bit range.

{{</citation>}}


### (66/104) Improving diagnosis and prognosis of lung cancer using vision transformers: A scoping review (Hazrat Ali et al., 2023)

{{<citation>}}

Hazrat Ali, Farida Mohsen, Zubair Shah. (2023)  
**Improving diagnosis and prognosis of lung cancer using vision transformers: A scoping review**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02783v1)  

---


**ABSTRACT**  
Vision transformer-based methods are advancing the field of medical artificial intelligence and cancer imaging, including lung cancer applications. Recently, many researchers have developed vision transformer-based AI methods for lung cancer diagnosis and prognosis. This scoping review aims to identify the recent developments on vision transformer-based AI methods for lung cancer imaging applications. It provides key insights into how vision transformers complemented the performance of AI and deep learning methods for lung cancer. Furthermore, the review also identifies the datasets that contributed to advancing the field. Of the 314 retrieved studies, this review included 34 studies published from 2020 to 2022. The most commonly addressed task in these studies was the classification of lung cancer types, such as lung squamous cell carcinoma versus lung adenocarcinoma, and identifying benign versus malignant pulmonary nodules. Other applications included survival prediction of lung cancer patients and segmentation of lungs. The studies lacked clear strategies for clinical transformation. SWIN transformer was a popular choice of the researchers; however, many other architectures were also reported where vision transformer was combined with convolutional neural networks or UNet model. It can be concluded that vision transformer-based models are increasingly in popularity for developing AI methods for lung cancer applications. However, their computational complexity and clinical relevance are important factors to be considered for future research work. This review provides valuable insights for researchers in the field of AI and healthcare to advance the state-of-the-art in lung cancer diagnosis and prognosis. We provide an interactive dashboard on lung-cancer.onrender.com/.

{{</citation>}}


### (67/104) Improving Image Classification of Knee Radiographs: An Automated Image Labeling Approach (Jikai Zhang et al., 2023)

{{<citation>}}

Jikai Zhang, Carlos Santos, Christine Park, Maciej Mazurowski, Roy Colglazier. (2023)  
**Improving Image Classification of Knee Radiographs: An Automated Image Labeling Approach**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2309.02681v1)  

---


**ABSTRACT**  
Large numbers of radiographic images are available in knee radiology practices which could be used for training of deep learning models for diagnosis of knee abnormalities. However, those images do not typically contain readily available labels due to limitations of human annotations. The purpose of our study was to develop an automated labeling approach that improves the image classification model to distinguish normal knee images from those with abnormalities or prior arthroplasty. The automated labeler was trained on a small set of labeled data to automatically label a much larger set of unlabeled data, further improving the image classification performance for knee radiographic diagnosis. We developed our approach using 7,382 patients and validated it on a separate set of 637 patients. The final image classification model, trained using both manually labeled and pseudo-labeled data, had the higher weighted average AUC (WAUC: 0.903) value and higher AUC-ROC values among all classes (normal AUC-ROC: 0.894; abnormal AUC-ROC: 0.896, arthroplasty AUC-ROC: 0.990) compared to the baseline model (WAUC=0.857; normal AUC-ROC: 0.842; abnormal AUC-ROC: 0.848, arthroplasty AUC-ROC: 0.987), trained using only manually labeled data. DeLong tests show that the improvement is significant on normal (p-value<0.002) and abnormal (p-value<0.001) images. Our findings demonstrated that the proposed automated labeling approach significantly improves the performance of image classification for radiographic knee diagnosis, allowing for facilitating patient care and curation of large knee datasets.

{{</citation>}}


### (68/104) Progressive Attention Guidance for Whole Slide Vulvovaginal Candidiasis Screening (Jiangdong Cai et al., 2023)

{{<citation>}}

Jiangdong Cai, Honglin Xiong, Maosong Cao, Luyan Liu, Lichi Zhang, Qian Wang. (2023)  
**Progressive Attention Guidance for Whole Slide Vulvovaginal Candidiasis Screening**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI, Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2309.02670v1)  

---


**ABSTRACT**  
Vulvovaginal candidiasis (VVC) is the most prevalent human candidal infection, estimated to afflict approximately 75% of all women at least once in their lifetime. It will lead to several symptoms including pruritus, vaginal soreness, and so on. Automatic whole slide image (WSI) classification is highly demanded, for the huge burden of disease control and prevention. However, the WSI-based computer-aided VCC screening method is still vacant due to the scarce labeled data and unique properties of candida. Candida in WSI is challenging to be captured by conventional classification models due to its distinctive elongated shape, the small proportion of their spatial distribution, and the style gap from WSIs. To make the model focus on the candida easier, we propose an attention-guided method, which can obtain a robust diagnosis classification model. Specifically, we first use a pre-trained detection model as prior instruction to initialize the classification model. Then we design a Skip Self-Attention module to refine the attention onto the fined-grained features of candida. Finally, we use a contrastive learning method to alleviate the overfitting caused by the style gap of WSIs and suppress the attention to false positive regions. Our experimental results demonstrate that our framework achieves state-of-the-art performance. Code and example data are available at https://github.com/cjdbehumble/MICCAI2023-VVC-Screening.

{{</citation>}}


## cs.AI (3)



### (69/104) Temporal Inductive Path Neural Network for Temporal Knowledge Graph Reasoning (Hao Dong et al., 2023)

{{<citation>}}

Hao Dong, Pengyang Wang, Meng Xiao, Zhiyuan Ning, Pengfei Wang, Yuanchun Zhou. (2023)  
**Temporal Inductive Path Neural Network for Temporal Knowledge Graph Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Knowledge Graph, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.03251v1)  

---


**ABSTRACT**  
Temporal Knowledge Graph (TKG) is an extension of traditional Knowledge Graph (KG) that incorporates the dimension of time. Reasoning on TKGs is a crucial task that aims to predict future facts based on historical occurrences. The key challenge lies in uncovering structural dependencies within historical subgraphs and temporal patterns. Most existing approaches model TKGs relying on entity modeling, as nodes in the graph play a crucial role in knowledge representation. However, the real-world scenario often involves an extensive number of entities, with new entities emerging over time. This makes it challenging for entity-dependent methods to cope with extensive volumes of entities, and effectively handling newly emerging entities also becomes a significant challenge. Therefore, we propose Temporal Inductive Path Neural Network (TiPNN), which models historical information in an entity-independent perspective. Specifically, TiPNN adopts a unified graph, namely history temporal graph, to comprehensively capture and encapsulate information from history. Subsequently, we utilize the defined query-aware temporal paths to model historical path information related to queries on history temporal graph for the reasoning. Extensive experiments illustrate that the proposed model not only attains significant performance enhancements but also handles inductive settings, while additionally facilitating the provision of reasoning evidence through history temporal graphs.

{{</citation>}}


### (70/104) Universal Preprocessing Operators for Embedding Knowledge Graphs with Literals (Patryk Preisner et al., 2023)

{{<citation>}}

Patryk Preisner, Heiko Paulheim. (2023)  
**Universal Preprocessing Operators for Embedding Knowledge Graphs with Literals**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.03023v1)  

---


**ABSTRACT**  
Knowledge graph embeddings are dense numerical representations of entities in a knowledge graph (KG). While the majority of approaches concentrate only on relational information, i.e., relations between entities, fewer approaches exist which also take information about literal values (e.g., textual descriptions or numerical information) into account. Those which exist are typically tailored towards a particular modality of literal and a particular embedding method. In this paper, we propose a set of universal preprocessing operators which can be used to transform KGs with literals for numerical, temporal, textual, and image information, so that the transformed KGs can be embedded with any method. The results on the kgbench dataset with three different embedding methods show promising results.

{{</citation>}}


### (71/104) Near-continuous time Reinforcement Learning for continuous state-action spaces (Lorenzo Croissant et al., 2023)

{{<citation>}}

Lorenzo Croissant, Marc Abeille, Bruno Bouchard. (2023)  
**Near-continuous time Reinforcement Learning for continuous state-action spaces**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, math-OC, math-ST, stat-TH  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.02815v1)  

---


**ABSTRACT**  
We consider the Reinforcement Learning problem of controlling an unknown dynamical system to maximise the long-term average reward along a single trajectory. Most of the literature considers system interactions that occur in discrete time and discrete state-action spaces. Although this standpoint is suitable for games, it is often inadequate for mechanical or digital systems in which interactions occur at a high frequency, if not in continuous time, and whose state spaces are large if not inherently continuous. Perhaps the only exception is the Linear Quadratic framework for which results exist both in discrete and continuous time. However, its ability to handle continuous states comes with the drawback of a rigid dynamic and reward structure. This work aims to overcome these shortcomings by modelling interaction times with a Poisson clock of frequency $\varepsilon^{-1}$, which captures arbitrary time scales: from discrete ($\varepsilon=1$) to continuous time ($\varepsilon\downarrow0$). In addition, we consider a generic reward function and model the state dynamics according to a jump process with an arbitrary transition kernel on $\mathbb{R}^d$. We show that the celebrated optimism protocol applies when the sub-tasks (learning and planning) can be performed effectively. We tackle learning within the eluder dimension framework and propose an approximate planning method based on a diffusive limit approximation of the jump process. Overall, our algorithm enjoys a regret of order $\tilde{\mathcal{O}}(\varepsilon^{1/2} T+\sqrt{T})$. As the frequency of interactions blows up, the approximation error $\varepsilon^{1/2} T$ vanishes, showing that $\tilde{\mathcal{O}}(\sqrt{T})$ is attainable in near-continuous time.

{{</citation>}}


## cs.CL (19)



### (72/104) Gender-specific Machine Translation with Large Language Models (Eduardo Sánchez et al., 2023)

{{<citation>}}

Eduardo Sánchez, Pierre Andrews, Pontus Stenetorp, Mikel Artetxe, Marta R. Costa-jussà. (2023)  
**Gender-specific Machine Translation with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Machine Translation  
[Paper Link](http://arxiv.org/abs/2309.03175v1)  

---


**ABSTRACT**  
Decoder-only Large Language Models (LLMs) have demonstrated potential in machine translation (MT), albeit with performance slightly lagging behind traditional encoder-decoder Neural Machine Translation (NMT) systems. However, LLMs offer a unique advantage: the ability to control the properties of the output through prompts. In this study, we harness this flexibility to explore LLaMa's capability to produce gender-specific translations for languages with grammatical gender. Our results indicate that LLaMa can generate gender-specific translations with competitive accuracy and gender bias mitigation when compared to NLLB, a state-of-the-art multilingual NMT system. Furthermore, our experiments reveal that LLaMa's translations are robust, showing significant performance drops when evaluated against opposite-gender references in gender-ambiguous datasets but maintaining consistency in less ambiguous contexts. This research provides insights into the potential and challenges of using LLMs for gender-specific translations and highlights the importance of in-context learning to elicit new tasks in LLMs.

{{</citation>}}


### (73/104) J-Guard: Journalism Guided Adversarially Robust Detection of AI-generated News (Tharindu Kumarage et al., 2023)

{{<citation>}}

Tharindu Kumarage, Amrita Bhattacharjee, Djordje Padejski, Kristy Roschke, Dan Gillmor, Scott Ruston, Huan Liu, Joshua Garland. (2023)  
**J-Guard: Journalism Guided Adversarially Robust Detection of AI-generated News**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.03164v1)  

---


**ABSTRACT**  
The rapid proliferation of AI-generated text online is profoundly reshaping the information landscape. Among various types of AI-generated text, AI-generated news presents a significant threat as it can be a prominent source of misinformation online. While several recent efforts have focused on detecting AI-generated text in general, these methods require enhanced reliability, given concerns about their vulnerability to simple adversarial attacks. Furthermore, due to the eccentricities of news writing, applying these detection methods for AI-generated news can produce false positives, potentially damaging the reputation of news organizations. To address these challenges, we leverage the expertise of an interdisciplinary team to develop a framework, J-Guard, capable of steering existing supervised AI text detectors for detecting AI-generated news while boosting adversarial robustness. By incorporating stylistic cues inspired by the unique journalistic attributes, J-Guard effectively distinguishes between real-world journalism and AI-generated news articles. Our experiments on news articles generated by a vast array of AI models, including ChatGPT (GPT3.5), demonstrate the effectiveness of J-Guard in enhancing detection capabilities while maintaining an average performance decrease of as low as 7% when faced with adversarial attacks.

{{</citation>}}


### (74/104) Knowledge Solver: Teaching LLMs to Search for Domain Knowledge from Knowledge Graphs (Chao Feng et al., 2023)

{{<citation>}}

Chao Feng, Xinyu Zhang, Zichu Fei. (2023)  
**Knowledge Solver: Teaching LLMs to Search for Domain Knowledge from Knowledge Graphs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GNN, GPT, GPT-4, Knowledge Graph, QA  
[Paper Link](http://arxiv.org/abs/2309.03118v1)  

---


**ABSTRACT**  
Large language models (LLMs), such as ChatGPT and GPT-4, are versatile and can solve different tasks due to their emergent ability and generalizability. However, LLMs sometimes lack domain-specific knowledge to perform tasks, which would also cause hallucination during inference. In some previous works, additional modules like graph neural networks (GNNs) are trained on retrieved knowledge from external knowledge bases, aiming to mitigate the problem of lacking domain-specific knowledge. However, incorporating additional modules: 1) would need retraining additional modules when encountering novel domains; 2) would become a bottleneck since LLMs' strong abilities are not fully utilized for retrieval. In this paper, we propose a paradigm, termed Knowledge Solver (KSL), to teach LLMs to search for essential knowledge from external knowledge bases by harnessing their own strong generalizability. Specifically, we design a simple yet effective prompt to transform retrieval into a multi-hop decision sequence, which empowers LLMs with searching knowledge ability in zero-shot manner. Additionally, KSL is able to provide complete retrieval paths and therefore increase explainability of LLMs' reasoning processes. We conduct experiments on three datasets: CommonsenseQA, OpenbookQA, and MedQA-USMLE, and found that our approach improves LLM baseline performance by a relatively large margin.

{{</citation>}}


### (75/104) ContrastWSD: Enhancing Metaphor Detection with Word Sense Disambiguation Following the Metaphor Identification Procedure (Mohamad Elzohbi et al., 2023)

{{<citation>}}

Mohamad Elzohbi, Richard Zhao. (2023)  
**ContrastWSD: Enhancing Metaphor Detection with Word Sense Disambiguation Following the Metaphor Identification Procedure**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, Word Sense Disambiguation  
[Paper Link](http://arxiv.org/abs/2309.03103v1)  

---


**ABSTRACT**  
This paper presents ContrastWSD, a RoBERTa-based metaphor detection model that integrates the Metaphor Identification Procedure (MIP) and Word Sense Disambiguation (WSD) to extract and contrast the contextual meaning with the basic meaning of a word to determine whether it is used metaphorically in a sentence. By utilizing the word senses derived from a WSD model, our model enhances the metaphor detection process and outperforms other methods that rely solely on contextual embeddings or integrate only the basic definitions and other external knowledge. We evaluate our approach on various benchmark datasets and compare it with strong baselines, indicating the effectiveness in advancing metaphor detection.

{{</citation>}}


### (76/104) A Multimodal Analysis of Influencer Content on Twitter (Danae Sánchez Villegas et al., 2023)

{{<citation>}}

Danae Sánchez Villegas, Catalina Goanta, Nikolaos Aletras. (2023)  
**A Multimodal Analysis of Influencer Content on Twitter**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2309.03064v1)  

---


**ABSTRACT**  
Influencer marketing involves a wide range of strategies in which brands collaborate with popular content creators (i.e., influencers) to leverage their reach, trust, and impact on their audience to promote and endorse products or services. Because followers of influencers are more likely to buy a product after receiving an authentic product endorsement rather than an explicit direct product promotion, the line between personal opinions and commercial content promotion is frequently blurred. This makes automatic detection of regulatory compliance breaches related to influencer advertising (e.g., misleading advertising or hidden sponsorships) particularly difficult. In this work, we (1) introduce a new Twitter (now X) dataset consisting of 15,998 influencer posts mapped into commercial and non-commercial categories for assisting in the automatic detection of commercial influencer content; (2) experiment with an extensive set of predictive models that combine text and visual information showing that our proposed cross-attention approach outperforms state-of-the-art multimodal models; and (3) conduct a thorough analysis of strengths and limitations of our models. We show that multimodal modeling is useful for identifying commercial posts, reducing the amount of false positives, and capturing relevant context that aids in the discovery of undisclosed commercial posts.

{{</citation>}}


### (77/104) Persona-aware Generative Model for Code-mixed Language (Ayan Sengupta et al., 2023)

{{<citation>}}

Ayan Sengupta, Md Shad Akhtar, Tanmoy Chakraborty. (2023)  
**Persona-aware Generative Model for Code-mixed Language**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BLEU, Rouge, Rouge-L, Transformer  
[Paper Link](http://arxiv.org/abs/2309.02915v1)  

---


**ABSTRACT**  
Code-mixing and script-mixing are prevalent across online social networks and multilingual societies. However, a user's preference toward code-mixing depends on the socioeconomic status, demographics of the user, and the local context, which existing generative models mostly ignore while generating code-mixed texts. In this work, we make a pioneering attempt to develop a persona-aware generative model to generate texts resembling real-life code-mixed texts of individuals. We propose a Persona-aware Generative Model for Code-mixed Generation, PARADOX, a novel Transformer-based encoder-decoder model that encodes an utterance conditioned on a user's persona and generates code-mixed texts without monolingual reference data. We propose an alignment module that re-calibrates the generated sequence to resemble real-life code-mixed texts. PARADOX generates code-mixed texts that are semantically more meaningful and linguistically more valid. To evaluate the personification capabilities of PARADOX, we propose four new metrics -- CM BLEU, CM Rouge-1, CM Rouge-L and CM KS. On average, PARADOX achieves 1.6 points better CM BLEU, 47% better perplexity and 32% better semantic coherence than the non-persona-based counterparts.

{{</citation>}}


### (78/104) Leave no Place Behind: Improved Geolocation in Humanitarian Documents (Enrico M. Belliardo et al., 2023)

{{<citation>}}

Enrico M. Belliardo, Kyriaki Kalimeri, Yelena Mejova. (2023)  
**Leave no Place Behind: Improved Geolocation in Humanitarian Documents**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NER, Named Entity Recognition, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.02914v1)  

---


**ABSTRACT**  
Geographical location is a crucial element of humanitarian response, outlining vulnerable populations, ongoing events, and available resources. Latest developments in Natural Language Processing may help in extracting vital information from the deluge of reports and documents produced by the humanitarian sector. However, the performance and biases of existing state-of-the-art information extraction tools are unknown. In this work, we develop annotated resources to fine-tune the popular Named Entity Recognition (NER) tools Spacy and roBERTa to perform geotagging of humanitarian texts. We then propose a geocoding method FeatureRank which links the candidate locations to the GeoNames database. We find that not only does the humanitarian-domain data improves the performance of the classifiers (up to F1 = 0.92), but it also alleviates some of the bias of the existing tools, which erroneously favor locations in the Western countries. Thus, we conclude that more resources from non-Western documents are necessary to ensure that off-the-shelf NER systems are suitable for the deployment in the humanitarian sector.

{{</citation>}}


### (79/104) On the Challenges of Building Datasets for Hate Speech Detection (Vitthal Bhandari, 2023)

{{<citation>}}

Vitthal Bhandari. (2023)  
**On the Challenges of Building Datasets for Hate Speech Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Hate Speech Detection, NLP  
[Paper Link](http://arxiv.org/abs/2309.02912v1)  

---


**ABSTRACT**  
Detection of hate speech has been formulated as a standalone application of NLP and different approaches have been adopted for identifying the target groups, obtaining raw data, defining the labeling process, choosing the detection algorithm, and evaluating the performance in the desired setting. However, unlike other downstream tasks, hate speech suffers from the lack of large-sized, carefully curated, generalizable datasets owing to the highly subjective nature of the task. In this paper, we first analyze the issues surrounding hate speech detection through a data-centric lens. We then outline a holistic framework to encapsulate the data creation pipeline across seven broad dimensions by taking the specific example of hate speech towards sexual minorities. We posit that practitioners would benefit from following this framework as a form of best practice when creating hate speech datasets in the future.

{{</citation>}}


### (80/104) ViCGCN: Graph Convolutional Network with Contextualized Language Models for Social Media Mining in Vietnamese (Chau-Thang Phan et al., 2023)

{{<citation>}}

Chau-Thang Phan, Quoc-Nam Nguyen, Chi-Thanh Dang, Trong-Hop Do, Kiet Van Nguyen. (2023)  
**ViCGCN: Graph Convolutional Network with Contextualized Language Models for Social Media Mining in Vietnamese**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, BERTology, Graph Convolutional Network, Language Model, Social Media  
[Paper Link](http://arxiv.org/abs/2309.02902v1)  

---


**ABSTRACT**  
Social media processing is a fundamental task in natural language processing with numerous applications. As Vietnamese social media and information science have grown rapidly, the necessity of information-based mining on Vietnamese social media has become crucial. However, state-of-the-art research faces several significant drawbacks, including imbalanced data and noisy data on social media platforms. Imbalanced and noisy are two essential issues that need to be addressed in Vietnamese social media texts. Graph Convolutional Networks can address the problems of imbalanced and noisy data in text classification on social media by taking advantage of the graph structure of the data. This study presents a novel approach based on contextualized language model (PhoBERT) and graph-based method (Graph Convolutional Networks). In particular, the proposed approach, ViCGCN, jointly trained the power of Contextualized embeddings with the ability of Graph Convolutional Networks, GCN, to capture more syntactic and semantic dependencies to address those drawbacks. Extensive experiments on various Vietnamese benchmark datasets were conducted to verify our approach. The observation shows that applying GCN to BERTology models as the final layer significantly improves performance. Moreover, the experiments demonstrate that ViCGCN outperforms 13 powerful baseline models, including BERTology models, fusion BERTology and GCN models, other baselines, and SOTA on three benchmark social media datasets. Our proposed ViCGCN approach demonstrates a significant improvement of up to 6.21%, 4.61%, and 2.63% over the best Contextualized Language Models, including multilingual and monolingual, on three benchmark datasets, UIT-VSMEC, UIT-ViCTSD, and UIT-VSFC, respectively. Additionally, our integrated model ViCGCN achieves the best performance compared to other BERTology integrated with GCN models.

{{</citation>}}


### (81/104) A deep Natural Language Inference predictor without language-specific training data (Lorenzo Corradi et al., 2023)

{{<citation>}}

Lorenzo Corradi, Alessandro Manenti, Francesca Del Bonifro, Francesco Setti, Dario Del Sorbo. (2023)  
**A deep Natural Language Inference predictor without language-specific training data**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Knowledge Distillation, NLI, NLP, Natural Language Inference, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2309.02887v1)  

---


**ABSTRACT**  
In this paper we present a technique of NLP to tackle the problem of inference relation (NLI) between pairs of sentences in a target language of choice without a language-specific training dataset. We exploit a generic translation dataset, manually translated, along with two instances of the same pre-trained model - the first to generate sentence embeddings for the source language, and the second fine-tuned over the target language to mimic the first. This technique is known as Knowledge Distillation. The model has been evaluated over machine translated Stanford NLI test dataset, machine translated Multi-Genre NLI test dataset, and manually translated RTE3-ITA test dataset. We also test the proposed architecture over different tasks to empirically demonstrate the generality of the NLI task. The model has been evaluated over the native Italian ABSITA dataset, on the tasks of Sentiment Analysis, Aspect-Based Sentiment Analysis, and Topic Recognition. We emphasise the generality and exploitability of the Knowledge Distillation technique that outperforms other methodologies based on machine translation, even though the former was not directly trained on the data it was tested over.

{{</citation>}}


### (82/104) Aligning Large Language Models for Clinical Tasks (Supun Manathunga et al., 2023)

{{<citation>}}

Supun Manathunga, Isuru Hettigoda. (2023)  
**Aligning Large Language Models for Clinical Tasks**  

---
Primary Category: cs.CL  
Categories: I-2, I-7, J-3, cs-CL, cs.CL  
Keywords: Clinical, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2309.02884v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable adaptability, showcasing their capacity to excel in tasks for which they were not explicitly trained. However, despite their impressive natural language processing (NLP) capabilities, effective alignment of LLMs remains a crucial challenge when deploying them for specific clinical applications. The ability to generate responses with factually accurate content and to engage in non-trivial reasoning steps are crucial for the LLMs to be eligible for applications in clinical medicine. Employing a combination of techniques including instruction-tuning and in-prompt strategies like few-shot and chain-of-thought prompting has significantly enhanced the performance of LLMs. Our proposed alignment strategy for medical question-answering, known as 'expand-guess-refine', offers a parameter and data-efficient solution. A preliminary analysis of this method demonstrated outstanding performance, achieving a score of 70.63% on a subset of questions sourced from the USMLE dataset.

{{</citation>}}


### (83/104) Promoting Open-domain Dialogue Generation through Learning Pattern Information between Contexts and Responses (Mengjuan Liu et al., 2023)

{{<citation>}}

Mengjuan Liu, Chenyang Liu, Yunfan Yang, Jiang Liu, Mohan Jing. (2023)  
**Promoting Open-domain Dialogue Generation through Learning Pattern Information between Contexts and Responses**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue, GPT  
[Paper Link](http://arxiv.org/abs/2309.02823v1)  

---


**ABSTRACT**  
Recently, utilizing deep neural networks to build the opendomain dialogue models has become a hot topic. However, the responses generated by these models suffer from many problems such as responses not being contextualized and tend to generate generic responses that lack information content, damaging the user's experience seriously. Therefore, many studies try introducing more information into the dialogue models to make the generated responses more vivid and informative. Unlike them, this paper improves the quality of generated responses by learning the implicit pattern information between contexts and responses in the training samples. In this paper, we first build an open-domain dialogue model based on the pre-trained language model (i.e., GPT-2). And then, an improved scheduled sampling method is proposed for pre-trained models, by which the responses can be used to guide the response generation in the training phase while avoiding the exposure bias problem. More importantly, we design a response-aware mechanism for mining the implicit pattern information between contexts and responses so that the generated replies are more diverse and approximate to human replies. Finally, we evaluate the proposed model (RAD) on the Persona-Chat and DailyDialog datasets; and the experimental results show that our model outperforms the baselines on most automatic and manual metrics.

{{</citation>}}


### (84/104) Rubric-Specific Approach to Automated Essay Scoring with Augmentation Training (Brian Cho et al., 2023)

{{<citation>}}

Brian Cho, Youngbin Jang, Jaewoong Yoon. (2023)  
**Rubric-Specific Approach to Automated Essay Scoring with Augmentation Training**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.02740v1)  

---


**ABSTRACT**  
Neural based approaches to automatic evaluation of subjective responses have shown superior performance and efficiency compared to traditional rule-based and feature engineering oriented solutions. However, it remains unclear whether the suggested neural solutions are sufficient replacements of human raters as we find recent works do not properly account for rubric items that are essential for automated essay scoring during model training and validation. In this paper, we propose a series of data augmentation operations that train and test an automated scoring model to learn features and functions overlooked by previous works while still achieving state-of-the-art performance in the Automated Student Assessment Prize dataset.

{{</citation>}}


### (85/104) HC3 Plus: A Semantic-Invariant Human ChatGPT Comparison Corpus (Zhenpeng Su et al., 2023)

{{<citation>}}

Zhenpeng Su, Xing Wu, Wei Zhou, Guangyuan Ma, Songlin Hu. (2023)  
**HC3 Plus: A Semantic-Invariant Human ChatGPT Comparison Corpus**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, BERT, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.02731v1)  

---


**ABSTRACT**  
ChatGPT has gained significant interest due to its impressive performance, but people are increasingly concerned about its potential risks, particularly around the detection of AI-generated content (AIGC), which is often difficult for untrained humans to identify. Current datasets utilized for detecting ChatGPT-generated text primarily center around question-answering, yet they tend to disregard tasks that possess semantic-invariant properties, such as summarization, translation, and paraphrasing. Our primary studies demonstrate that detecting model-generated text on semantic-invariant tasks is more difficult. To fill this gap, we introduce a more extensive and comprehensive dataset that considers more types of tasks than previous work, including semantic-invariant tasks. In addition, the model after a large number of task instruction fine-tuning shows a strong powerful performance. Owing to its previous success, we further instruct fine-tuning Tk-instruct and built a more powerful detection system. Experimental results show that our proposed detector outperforms the previous state-of-the-art RoBERTa-based detector.

{{</citation>}}


### (86/104) Large Language Models for Automated Open-domain Scientific Hypotheses Discovery (Zonglin Yang et al., 2023)

{{<citation>}}

Zonglin Yang, Xinya Du, Junxian Li, Jie Zheng, Soujanya Poria, Erik Cambria. (2023)  
**Large Language Models for Automated Open-domain Scientific Hypotheses Discovery**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2309.02726v1)  

---


**ABSTRACT**  
Hypothetical induction is recognized as the main reasoning type when scientists make observations about the world and try to propose hypotheses to explain those observations. Past research on hypothetical induction has a limited setting that (1) the observation annotations of the dataset are not raw web corpus but are manually selected sentences (resulting in a close-domain setting); and (2) the ground truth hypotheses annotations are mostly commonsense knowledge, making the task less challenging. In this work, we propose the first NLP dataset for social science academic hypotheses discovery, consisting of 50 recent papers published in top social science journals. Raw web corpora that are necessary for developing hypotheses in the published papers are also collected in the dataset, with the final goal of creating a system that automatically generates valid, novel, and helpful (to human researchers) hypotheses, given only a pile of raw web corpora. The new dataset can tackle the previous problems because it requires to (1) use raw web corpora as observations; and (2) propose hypotheses even new to humanity. A multi-module framework is developed for the task, as well as three different feedback mechanisms that empirically show performance gain over the base framework. Finally, our framework exhibits high performance in terms of both GPT-4 based evaluation and social science expert evaluation.

{{</citation>}}


### (87/104) Offensive Hebrew Corpus and Detection using BERT (Nagham Hamad et al., 2023)

{{<citation>}}

Nagham Hamad, Mustafa Jarrar, Mohammad Khalilia, Nadim Nashif. (2023)  
**Offensive Hebrew Corpus and Detection using BERT**  

---
Primary Category: cs.CL  
Categories: I-2-1; I-2-6; I-2-7; I-5-1, cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, Twitter  
[Paper Link](http://arxiv.org/abs/2309.02724v1)  

---


**ABSTRACT**  
Offensive language detection has been well studied in many languages, but it is lagging behind in low-resource languages, such as Hebrew. In this paper, we present a new offensive language corpus in Hebrew. A total of 15,881 tweets were retrieved from Twitter. Each was labeled with one or more of five classes (abusive, hate, violence, pornographic, or none offensive) by Arabic-Hebrew bilingual speakers. The annotation process was challenging as each annotator is expected to be familiar with the Israeli culture, politics, and practices to understand the context of each tweet. We fine-tuned two Hebrew BERT models, HeBERT and AlephBERT, using our proposed dataset and another published dataset. We observed that our data boosts HeBERT performance by 2% when combined with D_OLaH. Fine-tuning AlephBERT on our data and testing on D_OLaH yields 69% accuracy, while fine-tuning on D_OLaH and testing on our data yields 57% accuracy, which may be an indication to the generalizability our data offers. Our dataset and fine-tuned models are available on GitHub and Huggingface.

{{</citation>}}


### (88/104) HAE-RAE Bench: Evaluation of Korean Knowledge in Language Models (Guijin Son et al., 2023)

{{<citation>}}

Guijin Son, Hanwool Lee, Suwan Kim, Huiseo Kim, Jaecheol Lee, Je Won Yeom, Jihyu Jung, Jung Woo Kim, Songseong Kim. (2023)  
**HAE-RAE Bench: Evaluation of Korean Knowledge in Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2309.02706v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) pretrained on massive corpora exhibit remarkable capabilities across a wide range of tasks, however, the attention given to non-English languages has been limited in this field of research. To address this gap and assess the proficiency of language models in the Korean language and culture, we present HAE-RAE Bench, covering 6 tasks including vocabulary, history, and general knowledge. Our evaluation of language models on this benchmark highlights the potential advantages of employing Large Language-Specific Models(LLSMs) over a comprehensive, universal model like GPT-3.5. Remarkably, our study reveals that models approximately 13 times smaller than GPT-3.5 can exhibit similar performance levels in terms of language-specific knowledge retrieval. This observation underscores the importance of homogeneous corpora for training professional-level language-specific models. On the contrary, we also observe a perplexing performance dip in these smaller LMs when they are tasked to generate structured answers.

{{</citation>}}


### (89/104) A Joint Study of Phrase Grounding and Task Performance in Vision and Language Models (Noriyuki Kojima et al., 2023)

{{<citation>}}

Noriyuki Kojima, Hadar Averbuch-Elor, Yoav Artzi. (2023)  
**A Joint Study of Phrase Grounding and Task Performance in Vision and Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.02691v1)  

---


**ABSTRACT**  
Key to tasks that require reasoning about natural language in visual contexts is grounding words and phrases to image regions. However, observing this grounding in contemporary models is complex, even if it is generally expected to take place if the task is addressed in a way that is conductive to generalization. We propose a framework to jointly study task performance and phrase grounding, and propose three benchmarks to study the relation between the two. Our results show that contemporary models demonstrate inconsistency between their ability to ground phrases and solve tasks. We show how this can be addressed through brute-force training on ground phrasing annotations, and analyze the dynamics it creates. Code and at available at https://github.com/lil-lab/phrase_grounding.

{{</citation>}}


### (90/104) Zero-Resource Hallucination Prevention for Large Language Models (Junyu Luo et al., 2023)

{{<citation>}}

Junyu Luo, Cao Xiao, Fenglong Ma. (2023)  
**Zero-Resource Hallucination Prevention for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.02654v1)  

---


**ABSTRACT**  
The prevalent use of large language models (LLMs) in various domains has drawn attention to the issue of "hallucination," which refers to instances where LLMs generate factually inaccurate or ungrounded information. Existing techniques for hallucination detection in language assistants rely on intricate fuzzy, specific free-language-based chain of thought (CoT) techniques or parameter-based methods that suffer from interpretability issues. Additionally, the methods that identify hallucinations post-generation could not prevent their occurrence and suffer from inconsistent performance due to the influence of the instruction format and model style. In this paper, we introduce a novel pre-detection self-evaluation technique, referred to as {\method}, which focuses on evaluating the model's familiarity with the concepts present in the input instruction and withholding the generation of response in case of unfamiliar concepts. This approach emulates the human ability to refrain from responding to unfamiliar topics, thus reducing hallucinations. We validate {\method} across four different large language models, demonstrating consistently superior performance compared to existing techniques. Our findings propose a significant shift towards preemptive strategies for hallucination mitigation in LLM assistants, promising improvements in reliability, applicability, and interpretability.

{{</citation>}}


## q-fin.ST (1)



### (91/104) GPT-InvestAR: Enhancing Stock Investment Strategies through Annual Report Analysis with Large Language Models (Udit Gupta, 2023)

{{<citation>}}

Udit Gupta. (2023)  
**GPT-InvestAR: Enhancing Stock Investment Strategies through Annual Report Analysis with Large Language Models**  

---
Primary Category: q-fin.ST  
Categories: cs-CL, cs-LG, q-fin-ST, q-fin.ST  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.03079v1)  

---


**ABSTRACT**  
Annual Reports of publicly listed companies contain vital information about their financial health which can help assess the potential impact on Stock price of the firm. These reports are comprehensive in nature, going up to, and sometimes exceeding, 100 pages. Analysing these reports is cumbersome even for a single firm, let alone the whole universe of firms that exist. Over the years, financial experts have become proficient in extracting valuable information from these documents relatively quickly. However, this requires years of practice and experience. This paper aims to simplify the process of assessing Annual Reports of all the firms by leveraging the capabilities of Large Language Models (LLMs). The insights generated by the LLM are compiled in a Quant styled dataset and augmented by historical stock price data. A Machine Learning model is then trained with LLM outputs as features. The walkforward test results show promising outperformance wrt S&P500 returns. This paper intends to provide a framework for future work in this direction. To facilitate this, the code has been released as open source.

{{</citation>}}


## cs.IR (2)



### (92/104) Impression-Informed Multi-Behavior Recommender System: A Hierarchical Graph Attention Approach (Dong Li et al., 2023)

{{<citation>}}

Dong Li, Divya Bhargavi, Vidya Sagar Ravipati. (2023)  
**Impression-Informed Multi-Behavior Recommender System: A Hierarchical Graph Attention Approach**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.03169v2)  

---


**ABSTRACT**  
While recommender systems have significantly benefited from implicit feedback, they have often missed the nuances of multi-behavior interactions between users and items. Historically, these systems either amalgamated all behaviors, such as \textit{impression} (formerly \textit{view}), \textit{add-to-cart}, and \textit{buy}, under a singular 'interaction' label, or prioritized only the target behavior, often the \textit{buy} action, discarding valuable auxiliary signals. Although recent advancements tried addressing this simplification, they primarily gravitated towards optimizing the target behavior alone, battling with data scarcity. Additionally, they tended to bypass the nuanced hierarchy intrinsic to behaviors. To bridge these gaps, we introduce the \textbf{H}ierarchical \textbf{M}ulti-behavior \textbf{G}raph Attention \textbf{N}etwork (HMGN). This pioneering framework leverages attention mechanisms to discern information from both inter and intra-behaviors while employing a multi-task Hierarchical Bayesian Personalized Ranking (HBPR) for optimization. Recognizing the need for scalability, our approach integrates a specialized multi-behavior sub-graph sampling technique. Moreover, the adaptability of HMGN allows for the seamless inclusion of knowledge metadata and time-series data. Empirical results attest to our model's prowess, registering a notable performance boost of up to 64\% in NDCG@100 metrics over conventional graph neural network methods.

{{</citation>}}


### (93/104) Prompt-based Effective Input Reformulation for Legal Case Retrieval (Yanran Tang et al., 2023)

{{<citation>}}

Yanran Tang, Ruihong Qiu, Xue Li. (2023)  
**Prompt-based Effective Input Reformulation for Legal Case Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Legal  
[Paper Link](http://arxiv.org/abs/2309.02962v1)  

---


**ABSTRACT**  
Legal case retrieval plays an important role for legal practitioners to effectively retrieve relevant cases given a query case. Most existing neural legal case retrieval models directly encode the whole legal text of a case to generate a case representation, which is then utilised to conduct a nearest neighbour search for retrieval. Although these straightforward methods have achieved improvement over conventional statistical methods in retrieval accuracy, two significant challenges are identified in this paper: (1) Legal feature alignment: the usage of the whole case text as the input will generally incorporate redundant and noisy information because, from the legal perspective, the determining factor of relevant cases is the alignment of key legal features instead of whole text matching; (2) Legal context preservation: furthermore, since the existing text encoding models usually have an input length limit shorter than the case, the whole case text needs to be truncated or divided into paragraphs, which leads to the loss of the global context of legal information. In this paper, a novel legal case retrieval framework, PromptCase, is proposed to tackle these challenges. Firstly, legal facts and legal issues are identified and formally defined as the key features facilitating legal case retrieval based on a thorough study of the definition of relevant cases from a legal perspective. Secondly, with the determining legal features, a prompt-based encoding scheme is designed to conduct an effective encoding with language models. Extensive zero-shot experiments have been conducted on two benchmark datasets in legal case retrieval, which demonstrate the superior retrieval effectiveness of the proposed PromptCase. The code has been released on https://github.com/yanran-tang/PromptCase.

{{</citation>}}


## cs.DC (1)



### (94/104) UMS: Live Migration of Containerized Services across Autonomous Computing Systems (Thanawat Chanikaphon et al., 2023)

{{<citation>}}

Thanawat Chanikaphon, Mohsen Amini Salehi. (2023)  
**UMS: Live Migration of Containerized Services across Autonomous Computing Systems**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs-NI, cs.DC  
Keywords: Azure, Google, Microsoft  
[Paper Link](http://arxiv.org/abs/2309.03168v1)  

---


**ABSTRACT**  
Containerized services deployed within various computing systems, such as edge and cloud, desire live migration support to enable user mobility, elasticity, and load balancing. To enable such a ubiquitous and efficient service migration, a live migration solution needs to handle circumstances where users have various authority levels (full control, limited control, or no control) over the underlying computing systems. Supporting the live migration at these levels serves as the cornerstone of interoperability, and can unlock several use cases across various forms of distributed systems. As such, in this study, we develop a ubiquitous migration solution (called UMS) that, for a given containerized service, can automatically identify the feasible migration approach, and then seamlessly perform the migration across autonomous computing systems. UMS does not interfere with the way the orchestrator handles containers and can coordinate the migration without the orchestrator involvement. Moreover, UMS is orchestrator-agnostic, i.e., it can be plugged into any underlying orchestrator platform. UMS is equipped with novel methods that can coordinate and perform the live migration at the orchestrator, container, and service levels. Experimental results show that for single-process containers, the service-level approach, and for multi-process containers with small (< 128 MiB) memory footprint, the container-level migration approach lead to the lowest migration overhead and service downtime. To demonstrate the potential of UMS in realizing interoperability and multi-cloud scenarios, we examined it to perform live service migration across heterogeneous orchestrators, and between Microsoft Azure and Google Cloud

{{</citation>}}


## cs.SI (3)



### (95/104) Political Issue or Public Health: the Vaccination Debate on Twitter in Europe (Giordano Paoletti et al., 2023)

{{<citation>}}

Giordano Paoletti, Lorenzo Dall'Amico, Kyriaki Kalimeri, Jacopo Lenti, Yelena Mejova, Daniela Paolotti, Michele Starnini, Michele Tizzani. (2023)  
**Political Issue or Public Health: the Vaccination Debate on Twitter in Europe**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, physics-soc-ph  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2309.03078v1)  

---


**ABSTRACT**  
At the beginning of the COVID-19 pandemic, fears grew that making vaccination a political (instead of public health) issue may impact the efficacy of this life-saving intervention, spurring the spread of vaccine-hesitant content. In this study, we examine whether there is a relationship between the political interest of social media users and their exposure to vaccine-hesitant content on Twitter. We focus on 17 European countries using a multilingual, longitudinal dataset of tweets spanning the period before COVID, up to the vaccine roll-out. We find that, in most countries, users' exposure to vaccine-hesitant content is the highest in the early months of the pandemic, around the time of greatest scientific uncertainty. Further, users who follow politicians from right-wing parties, and those associated with authoritarian or anti-EU stances are more likely to be exposed to vaccine-hesitant content, whereas those following left-wing politicians, more pro-EU or liberal parties, are less likely to encounter it. Somewhat surprisingly, politicians did not play an outsized role in the vaccine debates of their countries, receiving a similar number of retweets as other similarly popular users. This systematic, multi-country, longitudinal investigation of the connection of politics with vaccine hesitancy has important implications for public health policy and communication.

{{</citation>}}


### (96/104) Prompt-based Node Feature Extractor for Few-shot Learning on Text-Attributed Graphs (Xuanwen Huang et al., 2023)

{{<citation>}}

Xuanwen Huang, Kaiqiao Han, Dezheng Bao, Quanjin Tao, Zhisheng Zhang, Yang Yang, Qi Zhu. (2023)  
**Prompt-based Node Feature Extractor for Few-shot Learning on Text-Attributed Graphs**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.02848v1)  

---


**ABSTRACT**  
Text-attributed Graphs (TAGs) are commonly found in the real world, such as social networks and citation networks, and consist of nodes represented by textual descriptions. Currently, mainstream machine learning methods on TAGs involve a two-stage modeling approach: (1) unsupervised node feature extraction with pre-trained language models (PLMs); and (2) supervised learning using Graph Neural Networks (GNNs). However, we observe that these representations, which have undergone large-scale pre-training, do not significantly improve performance with a limited amount of training samples. The main issue is that existing methods have not effectively integrated information from the graph and downstream tasks simultaneously. In this paper, we propose a novel framework called G-Prompt, which combines a graph adapter and task-specific prompts to extract node features. First, G-Prompt introduces a learnable GNN layer (\emph{i.e.,} adaptor) at the end of PLMs, which is fine-tuned to better capture the masked tokens considering graph neighborhood information. After the adapter is trained, G-Prompt incorporates task-specific prompts to obtain \emph{interpretable} node representations for the downstream task. Our experiment results demonstrate that our proposed method outperforms current state-of-the-art (SOTA) methods on few-shot node classification. More importantly, in zero-shot settings, the G-Prompt embeddings can not only provide better task interpretability than vanilla PLMs but also achieve comparable performance with fully-supervised baselines.

{{</citation>}}


### (97/104) Hy-DeFake: Hypergraph Neural Networks for Detecting Fake News in Online Social Networks (Xing Su et al., 2023)

{{<citation>}}

Xing Su, Jian Yang, Jia Wu, Zitai Qiu. (2023)  
**Hy-DeFake: Hypergraph Neural Networks for Detecting Fake News in Online Social Networks**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Fake News, Social Network  
[Paper Link](http://arxiv.org/abs/2309.02692v1)  

---


**ABSTRACT**  
Nowadays social media is the primary platform for people to obtain news and share information. Combating online fake news has become an urgent task to reduce the damage it causes to society. Existing methods typically improve their fake news detection performances by utilizing textual auxiliary information (such as relevant retweets and comments) or simple structural information (i.e., graph construction). However, these methods face two challenges. First, an increasing number of users tend to directly forward the source news without adding comments, resulting in a lack of textual auxiliary information. Second, simple graphs are unable to extract complex relations beyond pairwise association in a social context. Given that real-world social networks are intricate and involve high-order relations, we argue that exploring beyond pairwise relations between news and users is crucial for fake news detection. Therefore, we propose constructing an attributed hypergraph to represent non-textual and high-order relations for user participation in news spreading. We also introduce a hypergraph neural network-based method called Hy-DeFake to overcome the challenges. Our proposed method captures semantic information from news content, credibility information from involved users, and high-order correlations between news and users to learn distinctive embeddings for fake news detection. The superiority of Hy-DeFake is demonstrated through experiments conducted on four widely-used datasets, and it is compared against six baselines using four evaluation metrics.

{{</citation>}}


## cs.DB (1)



### (98/104) An Evaluation of Software Sketches (Roy Friedman, 2023)

{{<citation>}}

Roy Friedman. (2023)  
**An Evaluation of Software Sketches**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs-NI, cs.DB  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2309.03045v1)  

---


**ABSTRACT**  
This work presents a detailed evaluation of Rust (software) implementations of several popular sketching solutions, as well as recently proposed optimizations. We compare these solutions in terms of computational speed, memory consumption, and several approximation error metrics. Overall, we find a simple hashing based solution employed with the Nitro sampling technique [22] gives the best trade-off between memory, error and speed. Our findings also include some novel insights about how to best combine sampling with Counting Cuckoo filters depending on the application.

{{</citation>}}


## cs.SD (2)



### (99/104) An Efficient Temporary Deepfake Location Approach Based Embeddings for Partially Spoofed Audio Detection (Yuankun Xie et al., 2023)

{{<citation>}}

Yuankun Xie, Haonan Cheng, Yutian Wang, Long Ye. (2023)  
**An Efficient Temporary Deepfake Location Approach Based Embeddings for Partially Spoofed Audio Detection**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-SD, cs.SD, eess-AS  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.03036v1)  

---


**ABSTRACT**  
Partially spoofed audio detection is a challenging task, lying in the need to accurately locate the authenticity of audio at the frame level. To address this issue, we propose a fine-grained partially spoofed audio detection method, namely Temporal Deepfake Location (TDL), which can effectively capture information of both features and locations. Specifically, our approach involves two novel parts: embedding similarity module and temporal convolution operation. To enhance the identification between the real and fake features, the embedding similarity module is designed to generate an embedding space that can separate the real frames from fake frames. To effectively concentrate on the position information, temporal convolution operation is proposed to calculate the frame-specific similarities among neighboring frames, and dynamically select informative neighbors to convolution. Extensive experiments show that our method outperform baseline models in ASVspoof2019 Partial Spoof dataset and demonstrate superior performance even in the crossdataset scenario. The code is released online.

{{</citation>}}


### (100/104) Self-Supervised Disentanglement of Harmonic and Rhythmic Features in Music Audio Signals (Yiming Wu, 2023)

{{<citation>}}

Yiming Wu. (2023)  
**Self-Supervised Disentanglement of Harmonic and Rhythmic Features in Music Audio Signals**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.02796v1)  

---


**ABSTRACT**  
The aim of latent variable disentanglement is to infer the multiple informative latent representations that lie behind a data generation process and is a key factor in controllable data generation. In this paper, we propose a deep neural network-based self-supervised learning method to infer the disentangled rhythmic and harmonic representations behind music audio generation. We train a variational autoencoder that generates an audio mel-spectrogram from two latent features representing the rhythmic and harmonic content. In the training phase, the variational autoencoder is trained to reconstruct the input mel-spectrogram given its pitch-shifted version. At each forward computation in the training phase, a vector rotation operation is applied to one of the latent features, assuming that the dimensions of the feature vectors are related to pitch intervals. Therefore, in the trained variational autoencoder, the rotated latent feature represents the pitch-related information of the mel-spectrogram, and the unrotated latent feature represents the pitch-invariant information, i.e., the rhythmic content. The proposed method was evaluated using a predictor-based disentanglement metric on the learned features. Furthermore, we demonstrate its application to the automatic generation of music remixes.

{{</citation>}}


## cs.HC (1)



### (101/104) Reviving Static Charts into Live Charts (Lu Ying et al., 2023)

{{<citation>}}

Lu Ying, Yun Wang, Haotian Li, Shuguang Dou, Haidong Zhang, Xinyang Jiang, Huamin Qu, Yingcai Wu. (2023)  
**Reviving Static Charts into Live Charts**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2309.02967v1)  

---


**ABSTRACT**  
Data charts are prevalent across various fields due to their efficacy in conveying complex data relationships. However, static charts may sometimes struggle to engage readers and efficiently present intricate information, potentially resulting in limited understanding. We introduce "Live Charts," a new format of presentation that decomposes complex information within a chart and explains the information pieces sequentially through rich animations and accompanying audio narration. We propose an automated approach to revive static charts into Live Charts. Our method integrates GNN-based techniques to analyze the chart components and extract data from charts. Then we adopt large natural language models to generate appropriate animated visuals along with a voice-over to produce Live Charts from static ones. We conducted a thorough evaluation of our approach, which involved the model performance, use cases, a crowd-sourced user study, and expert interviews. The results demonstrate Live Charts offer a multi-sensory experience where readers can follow the information and understand the data insights better. We analyze the benefits and drawbacks of Live Charts over static charts as a new information consumption experience.

{{</citation>}}


## cs.CE (1)



### (102/104) Reinforcement Learning Based Gasoline Blending Optimization: Achieving More Efficient Nonlinear Online Blending of Fuels (Muyi Huang et al., 2023)

{{<citation>}}

Muyi Huang, Renchu He, Xin Dai, Xin Peng, Wenli Du, Feng Qian. (2023)  
**Reinforcement Learning Based Gasoline Blending Optimization: Achieving More Efficient Nonlinear Online Blending of Fuels**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.02929v1)  

---


**ABSTRACT**  
The online optimization of gasoline blending benefits refinery economies. However, the nonlinear blending mechanism, the oil property fluctuations, and the blending model mismatch bring difficulties to the optimization. To solve the above issues, this paper proposes a novel online optimization method based on deep reinforcement learning algorithm (DRL). The Markov decision process (MDP) expression are given considering a practical gasoline blending system. Then, the environment simulator of gasoline blending process is established based on the MDP expression and the one-year measurement data of a real-world refinery. The soft actor-critic (SAC) DRL algorithm is applied to improve the DRL agent policy by using the data obtained from the interaction between DRL agent and environment simulator. Compared with a traditional method, the proposed method has better economic performance. Meanwhile, it is more robust under property fluctuations and component oil switching. Furthermore, the proposed method maintains performance by automatically adapting to system drift.

{{</citation>}}


## cs.DL (1)



### (103/104) Measuring open access publications: a novel normalized open access indicator (Abdelghani Maddi, 2023)

{{<citation>}}

Abdelghani Maddi. (2023)  
**Measuring open access publications: a novel normalized open access indicator**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs.DL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.03243v1)  

---


**ABSTRACT**  
The issue of open access (OA) to scientific publications is attracting growing interest within the scientific community and among policy makers. Open access indicators are being calculated. In its 2019 ranking, the ''Centre for Science and Technology Studies'' (CWTS) provides the number and the share of OA publications per institution. This gives an idea of the degree of openness of institutions. However, not taking into account the disciplinary specificities and the specialization of institutions makes comparisons based on the shares of OA publications biased. We show that OA publishing practices vary considerably according to discipline. As a result, we propose two methods to normalize OA share; by WoS subject categories and by disciplines. Normalized Open Access Indicator (NOAI) corrects for disciplinary composition and allows a better comparability of institutions or countries.

{{</citation>}}


## q-bio.GN (1)



### (104/104) Automated Bioinformatics Analysis via AutoBA (Juexiao Zhou et al., 2023)

{{<citation>}}

Juexiao Zhou, Bin Zhang, Xiuying Chen, Haoyang Li, Xiaopeng Xu, Siyuan Chen, Xin Gao. (2023)  
**Automated Bioinformatics Analysis via AutoBA**  

---
Primary Category: q-bio.GN  
Categories: cs-AI, cs-LG, cs-MA, q-bio-GN, q-bio.GN  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.03242v1)  

---


**ABSTRACT**  
With the fast-growing and evolving omics data, the demand for streamlined and adaptable tools to handle the analysis continues to grow. In response to this need, we introduce Auto Bioinformatics Analysis (AutoBA), an autonomous AI agent based on a large language model designed explicitly for conventional omics data analysis. AutoBA simplifies the analytical process by requiring minimal user input while delivering detailed step-by-step plans for various bioinformatics tasks. Through rigorous validation by expert bioinformaticians, AutoBA's robustness and adaptability are affirmed across a diverse range of omics analysis cases, including whole genome sequencing (WGS), RNA sequencing (RNA-seq), single-cell RNA-seq, ChIP-seq, and spatial transcriptomics. AutoBA's unique capacity to self-design analysis processes based on input data variations further underscores its versatility. Compared with online bioinformatic services, AutoBA deploys the analysis locally, preserving data privacy. Moreover, different from the predefined pipeline, AutoBA has adaptability in sync with emerging bioinformatics tools. Overall, AutoBA represents a convenient tool, offering robustness and adaptability for complex omics data analysis.

{{</citation>}}
