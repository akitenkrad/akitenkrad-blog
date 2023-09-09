---
draft: false
title: "arXiv @ 2023.09.09"
date: 2023-09-09
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.09"
    identifier: arxiv_20230909
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.MM (1)](#csmm-1)
- [cs.CV (17)](#cscv-17)
- [cs.SD (2)](#cssd-2)
- [cs.CL (18)](#cscl-18)
- [cs.SI (2)](#cssi-2)
- [cs.NI (3)](#csni-3)
- [cs.RO (5)](#csro-5)
- [cs.LG (13)](#cslg-13)
- [cs.AI (4)](#csai-4)
- [eess.IV (4)](#eessiv-4)
- [cs.CR (5)](#cscr-5)
- [cs.DS (1)](#csds-1)
- [stat.ML (1)](#statml-1)
- [cs.CY (1)](#cscy-1)
- [cs.IR (4)](#csir-4)
- [cs.SE (3)](#csse-3)
- [cs.PL (1)](#cspl-1)
- [cs.DC (1)](#csdc-1)
- [cs.GT (1)](#csgt-1)
- [eess.SY (1)](#eesssy-1)

## cs.MM (1)



### (1/88) ImageBind-LLM: Multi-modality Instruction Tuning (Jiaming Han et al., 2023)

{{<citation>}}

Jiaming Han, Renrui Zhang, Wenqi Shao, Peng Gao, Peng Xu, Han Xiao, Kaipeng Zhang, Chris Liu, Song Wen, Ziyu Guo, Xudong Lu, Shuai Ren, Yafei Wen, Xiaoxin Chen, Xiangyu Yue, Hongsheng Li, Yu Qiao. (2023)  
**ImageBind-LLM: Multi-modality Instruction Tuning**  

---
Primary Category: cs.MM  
Categories: cs-CL, cs-CV, cs-LG, cs-MM, cs-SD, cs.MM, eess-AS  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2309.03905v1)  

---


**ABSTRACT**  
We present ImageBind-LLM, a multi-modality instruction tuning method of large language models (LLMs) via ImageBind. Existing works mainly focus on language and image instruction tuning, different from which, our ImageBind-LLM can respond to multi-modality conditions, including audio, 3D point clouds, video, and their embedding-space arithmetic by only image-text alignment training. During training, we adopt a learnable bind network to align the embedding space between LLaMA and ImageBind's image encoder. Then, the image features transformed by the bind network are added to word tokens of all layers in LLaMA, which progressively injects visual instructions via an attention-free and zero-initialized gating mechanism. Aided by the joint embedding of ImageBind, the simple image-text training enables our model to exhibit superior multi-modality instruction-following capabilities. During inference, the multi-modality inputs are fed into the corresponding ImageBind encoders, and processed by a proposed visual cache model for further cross-modal embedding enhancement. The training-free cache model retrieves from three million image features extracted by ImageBind, which effectively mitigates the training-inference modality discrepancy. Notably, with our approach, ImageBind-LLM can respond to instructions of diverse modalities and demonstrate significant language generation quality. Code is released at https://github.com/OpenGVLab/LLaMA-Adapter.

{{</citation>}}


## cs.CV (17)



### (2/88) Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis (Jiapeng Zhu et al., 2023)

{{<citation>}}

Jiapeng Zhu, Ceyuan Yang, Kecheng Zheng, Yinghao Xu, Zifan Shi, Yujun Shen. (2023)  
**Exploring Sparse MoE in GANs for Text-conditioned Image Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.03904v1)  

---


**ABSTRACT**  
Due to the difficulty in scaling up, generative adversarial networks (GANs) seem to be falling from grace on the task of text-conditioned image synthesis. Sparsely-activated mixture-of-experts (MoE) has recently been demonstrated as a valid solution to training large-scale models with limited computational resources. Inspired by such a philosophy, we present Aurora, a GAN-based text-to-image generator that employs a collection of experts to learn feature processing, together with a sparse router to help select the most suitable expert for each feature point. To faithfully decode the sampling stochasticity and the text condition to the final synthesis, our router adaptively makes its decision by taking into account the text-integrated global latent code. At 64x64 image resolution, our model trained on LAION2B-en and COYO-700M achieves 6.2 zero-shot FID on MS COCO. We release the code and checkpoints to facilitate the community for further development.

{{</citation>}}


### (3/88) ProPainter: Improving Propagation and Transformer for Video Inpainting (Shangchen Zhou et al., 2023)

{{<citation>}}

Shangchen Zhou, Chongyi Li, Kelvin C. K. Chan, Chen Change Loy. (2023)  
**ProPainter: Improving Propagation and Transformer for Video Inpainting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.03897v1)  

---


**ABSTRACT**  
Flow-based propagation and spatiotemporal Transformer are two mainstream mechanisms in video inpainting (VI). Despite the effectiveness of these components, they still suffer from some limitations that affect their performance. Previous propagation-based approaches are performed separately either in the image or feature domain. Global image propagation isolated from learning may cause spatial misalignment due to inaccurate optical flow. Moreover, memory or computational constraints limit the temporal range of feature propagation and video Transformer, preventing exploration of correspondence information from distant frames. To address these issues, we propose an improved framework, called ProPainter, which involves enhanced ProPagation and an efficient Transformer. Specifically, we introduce dual-domain propagation that combines the advantages of image and feature warping, exploiting global correspondences reliably. We also propose a mask-guided sparse video Transformer, which achieves high efficiency by discarding unnecessary and redundant tokens. With these components, ProPainter outperforms prior arts by a large margin of 1.46 dB in PSNR while maintaining appealing efficiency.

{{</citation>}}


### (4/88) DiffusionEngine: Diffusion Model is Scalable Data Engine for Object Detection (Manlin Zhang et al., 2023)

{{<citation>}}

Manlin Zhang, Jie Wu, Yuxi Ren, Ming Li, Jie Qin, Xuefeng Xiao, Wei Liu, Rui Wang, Min Zheng, Andy J. Ma. (2023)  
**DiffusionEngine: Diffusion Model is Scalable Data Engine for Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.03893v1)  

---


**ABSTRACT**  
Data is the cornerstone of deep learning. This paper reveals that the recently developed Diffusion Model is a scalable data engine for object detection. Existing methods for scaling up detection-oriented data often require manual collection or generative models to obtain target images, followed by data augmentation and labeling to produce training pairs, which are costly, complex, or lacking diversity. To address these issues, we presentDiffusionEngine (DE), a data scaling-up engine that provides high-quality detection-oriented training pairs in a single stage. DE consists of a pre-trained diffusion model and an effective Detection-Adapter, contributing to generating scalable, diverse and generalizable detection data in a plug-and-play manner. Detection-Adapter is learned to align the implicit semantic and location knowledge in off-the-shelf diffusion models with detection-aware signals to make better bounding-box predictions. Additionally, we contribute two datasets, i.e., COCO-DE and VOC-DE, to scale up existing detection benchmarks for facilitating follow-up research. Extensive experiments demonstrate that data scaling-up via DE can achieve significant improvements in diverse scenarios, such as various detection algorithms, self-supervised pre-training, data-sparse, label-scarce, cross-domain, and semi-supervised learning. For example, when using DE with a DINO-based adapter to scale up data, mAP is improved by 3.1% on COCO, 7.6% on VOC, and 11.5% on Clipart.

{{</citation>}}


### (5/88) Cross-Task Attention Network: Improving Multi-Task Learning for Medical Imaging Applications (Sangwook Kim et al., 2023)

{{<citation>}}

Sangwook Kim, Thomas G. Purdie, Chris McIntosh. (2023)  
**Cross-Task Attention Network: Improving Multi-Task Learning for Medical Imaging Applications**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.03837v1)  

---


**ABSTRACT**  
Multi-task learning (MTL) is a powerful approach in deep learning that leverages the information from multiple tasks during training to improve model performance. In medical imaging, MTL has shown great potential to solve various tasks. However, existing MTL architectures in medical imaging are limited in sharing information across tasks, reducing the potential performance improvements of MTL. In this study, we introduce a novel attention-based MTL framework to better leverage inter-task interactions for various tasks from pixel-level to image-level predictions. Specifically, we propose a Cross-Task Attention Network (CTAN) which utilizes cross-task attention mechanisms to incorporate information by interacting across tasks. We validated CTAN on four medical imaging datasets that span different domains and tasks including: radiation treatment planning prediction using planning CT images of two different target cancers (Prostate, OpenKBP); pigmented skin lesion segmentation and diagnosis using dermatoscopic images (HAM10000); and COVID-19 diagnosis and severity prediction using chest CT scans (STOIC). Our study demonstrates the effectiveness of CTAN in improving the accuracy of medical imaging tasks. Compared to standard single-task learning (STL), CTAN demonstrated a 4.67% improvement in performance and outperformed both widely used MTL baselines: hard parameter sharing (HPS) with an average performance improvement of 3.22%; and multi-task attention network (MTAN) with a relative decrease of 5.38%. These findings highlight the significance of our proposed MTL framework in solving medical imaging tasks and its potential to improve their accuracy across domains.

{{</citation>}}


### (6/88) ClusterFusion: Leveraging Radar Spatial Features for Radar-Camera 3D Object Detection in Autonomous Vehicles (Irfan Tito Kurniawan et al., 2023)

{{<citation>}}

Irfan Tito Kurniawan, Bambang Riyanto Trilaksono. (2023)  
**ClusterFusion: Leveraging Radar Spatial Features for Radar-Camera 3D Object Detection in Autonomous Vehicles**  

---
Primary Category: cs.CV  
Categories: I-4-8; I-4-10, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.03734v1)  

---


**ABSTRACT**  
Thanks to the complementary nature of millimeter wave radar and camera, deep learning-based radar-camera 3D object detection methods may reliably produce accurate detections even in low-visibility conditions. This makes them preferable to use in autonomous vehicles' perception systems, especially as the combined cost of both sensors is cheaper than the cost of a lidar. Recent radar-camera methods commonly perform feature-level fusion which often involves projecting the radar points onto the same plane as the image features and fusing the extracted features from both modalities. While performing fusion on the image plane is generally simpler and faster, projecting radar points onto the image plane flattens the depth dimension of the point cloud which might lead to information loss and makes extracting the spatial features of the point cloud harder. We proposed ClusterFusion, an architecture that leverages the local spatial features of the radar point cloud by clustering the point cloud and performing feature extraction directly on the point cloud clusters before projecting the features onto the image plane. ClusterFusion achieved the state-of-the-art performance among all radar-monocular camera methods on the test slice of the nuScenes dataset with 48.7% nuScenes detection score (NDS). We also investigated the performance of different radar feature extraction strategies on point cloud clusters: a handcrafted strategy, a learning-based strategy, and a combination of both, and found that the handcrafted strategy yielded the best performance. The main goal of this work is to explore the use of radar's local spatial and point-wise features by extracting them directly from radar point cloud clusters for a radar-monocular camera 3D object detection method that performs cross-modal feature fusion on the image plane.

{{</citation>}}


### (7/88) Phasic Content Fusing Diffusion Model with Directional Distribution Consistency for Few-Shot Model Adaption (Teng Hu et al., 2023)

{{<citation>}}

Teng Hu, Jiangning Zhang, Liang Liu, Ran Yi, Siqi Kou, Haokun Zhu, Xu Chen, Yabiao Wang, Chengjie Wang, Lizhuang Ma. (2023)  
**Phasic Content Fusing Diffusion Model with Directional Distribution Consistency for Few-Shot Model Adaption**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.03729v1)  

---


**ABSTRACT**  
Training a generative model with limited number of samples is a challenging task. Current methods primarily rely on few-shot model adaption to train the network. However, in scenarios where data is extremely limited (less than 10), the generative network tends to overfit and suffers from content degradation. To address these problems, we propose a novel phasic content fusing few-shot diffusion model with directional distribution consistency loss, which targets different learning objectives at distinct training stages of the diffusion model. Specifically, we design a phasic training strategy with phasic content fusion to help our model learn content and style information when t is large, and learn local details of target domain when t is small, leading to an improvement in the capture of content, style and local details. Furthermore, we introduce a novel directional distribution consistency loss that ensures the consistency between the generated and source distributions more efficiently and stably than the prior methods, preventing our model from overfitting. Finally, we propose a cross-domain structure guidance strategy that enhances structure consistency during domain adaptation. Theoretical analysis, qualitative and quantitative experiments demonstrate the superiority of our approach in few-shot generative model adaption tasks compared to state-of-the-art methods. The source code is available at: https://github.com/sjtuplayer/few-shot-diffusion.

{{</citation>}}


### (8/88) Interpretable Visual Question Answering via Reasoning Supervision (Maria Parelli et al., 2023)

{{<citation>}}

Maria Parelli, Dimitrios Mallis, Markos Diomataris, Vassilis Pitsikalis. (2023)  
**Interpretable Visual Question Answering via Reasoning Supervision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering, Reasoning, Transformer  
[Paper Link](http://arxiv.org/abs/2309.03726v1)  

---


**ABSTRACT**  
Transformer-based architectures have recently demonstrated remarkable performance in the Visual Question Answering (VQA) task. However, such models are likely to disregard crucial visual cues and often rely on multimodal shortcuts and inherent biases of the language modality to predict the correct answer, a phenomenon commonly referred to as lack of visual grounding. In this work, we alleviate this shortcoming through a novel architecture for visual question answering that leverages common sense reasoning as a supervisory signal. Reasoning supervision takes the form of a textual justification of the correct answer, with such annotations being already available on large-scale Visual Common Sense Reasoning (VCR) datasets. The model's visual attention is guided toward important elements of the scene through a similarity loss that aligns the learned attention distributions guided by the question and the correct reasoning. We demonstrate both quantitatively and qualitatively that the proposed approach can boost the model's visual perception capability and lead to performance increase, without requiring training on explicit grounding annotations.

{{</citation>}}


### (9/88) Efficient Adaptive Human-Object Interaction Detection with Concept-guided Memory (Ting Lei et al., 2023)

{{<citation>}}

Ting Lei, Fabian Caba, Qingchao Chen, Hailin Jin, Yuxin Peng, Yang Liu. (2023)  
**Efficient Adaptive Human-Object Interaction Detection with Concept-guided Memory**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.03696v1)  

---


**ABSTRACT**  
Human Object Interaction (HOI) detection aims to localize and infer the relationships between a human and an object. Arguably, training supervised models for this task from scratch presents challenges due to the performance drop over rare classes and the high computational cost and time required to handle long-tailed distributions of HOIs in complex HOI scenes in realistic settings. This observation motivates us to design an HOI detector that can be trained even with long-tailed labeled data and can leverage existing knowledge from pre-trained models. Inspired by the powerful generalization ability of the large Vision-Language Models (VLM) on classification and retrieval tasks, we propose an efficient Adaptive HOI Detector with Concept-guided Memory (ADA-CM). ADA-CM has two operating modes. The first mode makes it tunable without learning new parameters in a training-free paradigm. Its second mode incorporates an instance-aware adapter mechanism that can further efficiently boost performance if updating a lightweight set of parameters can be afforded. Our proposed method achieves competitive results with state-of-the-art on the HICO-DET and V-COCO datasets with much less training time. Code can be found at https://github.com/ltttpku/ADA-CM.

{{</citation>}}


### (10/88) Towards Comparable Knowledge Distillation in Semantic Image Segmentation (Onno Niemann et al., 2023)

{{<citation>}}

Onno Niemann, Christopher Vox, Thorben Werner. (2023)  
**Towards Comparable Knowledge Distillation in Semantic Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.03659v1)  

---


**ABSTRACT**  
Knowledge Distillation (KD) is one proposed solution to large model sizes and slow inference speed in semantic segmentation. In our research we identify 25 proposed distillation loss terms from 14 publications in the last 4 years. Unfortunately, a comparison of terms based on published results is often impossible, because of differences in training configurations. A good illustration of this problem is the comparison of two publications from 2022. Using the same models and dataset, Structural and Statistical Texture Distillation (SSTKD) reports an increase of student mIoU of 4.54 and a final performance of 29.19, while Adaptive Perspective Distillation (APD) only improves student performance by 2.06 percentage points, but achieves a final performance of 39.25. The reason for such extreme differences is often a suboptimal choice of hyperparameters and a resulting underperformance of the student model used as reference point. In our work, we reveal problems of insufficient hyperparameter tuning by showing that distillation improvements of two widely accepted frameworks, SKD and IFVD, vanish when hyperparameters are optimized sufficiently. To improve comparability of future research in the field, we establish a solid baseline for three datasets and two student models and provide extensive information on hyperparameter tuning. We find that only two out of eight techniques can compete with our simple baseline on the ADE20K dataset.

{{</citation>}}


### (11/88) Enhancing Sample Utilization through Sample Adaptive Augmentation in Semi-Supervised Learning (Guan Gui et al., 2023)

{{<citation>}}

Guan Gui, Zhen Zhao, Lei Qi, Luping Zhou, Lei Wang, Yinghuan Shi. (2023)  
**Enhancing Sample Utilization through Sample Adaptive Augmentation in Semi-Supervised Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.03598v1)  

---


**ABSTRACT**  
In semi-supervised learning, unlabeled samples can be utilized through augmentation and consistency regularization. However, we observed certain samples, even undergoing strong augmentation, are still correctly classified with high confidence, resulting in a loss close to zero. It indicates that these samples have been already learned well and do not provide any additional optimization benefits to the model. We refer to these samples as ``naive samples". Unfortunately, existing SSL models overlook the characteristics of naive samples, and they just apply the same learning strategy to all samples. To further optimize the SSL model, we emphasize the importance of giving attention to naive samples and augmenting them in a more diverse manner. Sample adaptive augmentation (SAA) is proposed for this stated purpose and consists of two modules: 1) sample selection module; 2) sample augmentation module. Specifically, the sample selection module picks out {naive samples} based on historical training information at each epoch, then the naive samples will be augmented in a more diverse manner in the sample augmentation module. Thanks to the extreme ease of implementation of the above modules, SAA is advantageous for being simple and lightweight. We add SAA on top of FixMatch and FlexMatch respectively, and experiments demonstrate SAA can significantly improve the models. For example, SAA helped improve the accuracy of FixMatch from 92.50% to 94.76% and that of FlexMatch from 95.01% to 95.31% on CIFAR-10 with 40 labels.

{{</citation>}}


### (12/88) DropPos: Pre-Training Vision Transformers by Reconstructing Dropped Positions (Haochen Wang et al., 2023)

{{<citation>}}

Haochen Wang, Junsong Fan, Yuxi Wang, Kaiyou Song, Tong Wang, Zhaoxiang Zhang. (2023)  
**DropPos: Pre-Training Vision Transformers by Reconstructing Dropped Positions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.03576v1)  

---


**ABSTRACT**  
As it is empirically observed that Vision Transformers (ViTs) are quite insensitive to the order of input tokens, the need for an appropriate self-supervised pretext task that enhances the location awareness of ViTs is becoming evident. To address this, we present DropPos, a novel pretext task designed to reconstruct Dropped Positions. The formulation of DropPos is simple: we first drop a large random subset of positional embeddings and then the model classifies the actual position for each non-overlapping patch among all possible positions solely based on their visual appearance. To avoid trivial solutions, we increase the difficulty of this task by keeping only a subset of patches visible. Additionally, considering there may be different patches with similar visual appearances, we propose position smoothing and attentive reconstruction strategies to relax this classification problem, since it is not necessary to reconstruct their exact positions in these cases. Empirical evaluations of DropPos show strong capabilities. DropPos outperforms supervised pre-training and achieves competitive results compared with state-of-the-art self-supervised alternatives on a wide range of downstream benchmarks. This suggests that explicitly encouraging spatial reasoning abilities, as DropPos does, indeed contributes to the improved location awareness of ViTs. The code is publicly available at https://github.com/Haochen-Wang409/DropPos.

{{</citation>}}


### (13/88) Toward High Quality Facial Representation Learning (Yue Wang et al., 2023)

{{<citation>}}

Yue Wang, Jinlong Peng, Jiangning Zhang, Ran Yi, Liang Liu, Yabiao Wang, Chengjie Wang. (2023)  
**Toward High Quality Facial Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.03575v1)  

---


**ABSTRACT**  
Face analysis tasks have a wide range of applications, but the universal facial representation has only been explored in a few works. In this paper, we explore high-performance pre-training methods to boost the face analysis tasks such as face alignment and face parsing. We propose a self-supervised pre-training framework, called \textbf{\it Mask Contrastive Face (MCF)}, with mask image modeling and a contrastive strategy specially adjusted for face domain tasks. To improve the facial representation quality, we use feature map of a pre-trained visual backbone as a supervision item and use a partially pre-trained decoder for mask image modeling. To handle the face identity during the pre-training stage, we further use random masks to build contrastive learning pairs. We conduct the pre-training on the LAION-FACE-cropped dataset, a variants of LAION-FACE 20M, which contains more than 20 million face images from Internet websites. For efficiency pre-training, we explore our framework pre-training performance on a small part of LAION-FACE-cropped and verify the superiority with different pre-training settings. Our model pre-trained with the full pre-training dataset outperforms the state-of-the-art methods on multiple downstream tasks. Our model achieves 0.932 NME$_{diag}$ for AFLW-19 face alignment and 93.96 F1 score for LaPa face parsing. Code is available at https://github.com/nomewang/MCF.

{{</citation>}}


### (14/88) Trash to Treasure: Low-Light Object Detection via Decomposition-and-Aggregation (Xiaohan Cui et al., 2023)

{{<citation>}}

Xiaohan Cui, Long Ma, Tengyu Ma, Jinyuan Liu, Xin Fan, Risheng Liu. (2023)  
**Trash to Treasure: Low-Light Object Detection via Decomposition-and-Aggregation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.03548v1)  

---


**ABSTRACT**  
Object detection in low-light scenarios has attracted much attention in the past few years. A mainstream and representative scheme introduces enhancers as the pre-processing for regular detectors. However, because of the disparity in task objectives between the enhancer and detector, this paradigm cannot shine at its best ability. In this work, we try to arouse the potential of enhancer + detector. Different from existing works, we extend the illumination-based enhancers (our newly designed or existing) as a scene decomposition module, whose removed illumination is exploited as the auxiliary in the detector for extracting detection-friendly features. A semantic aggregation module is further established for integrating multi-scale scene-related semantic information in the context space. Actually, our built scheme successfully transforms the "trash" (i.e., the ignored illumination in the detector) into the "treasure" for the detector. Plenty of experiments are conducted to reveal our superiority against other state-of-the-art methods. The code will be public if it is accepted.

{{</citation>}}


### (15/88) Zero-Shot Scene Graph Generation via Triplet Calibration and Reduction (Jiankai Li et al., 2023)

{{<citation>}}

Jiankai Li, Yunhong Wang, Weixin Li. (2023)  
**Zero-Shot Scene Graph Generation via Triplet Calibration and Reduction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.03542v1)  

---


**ABSTRACT**  
Scene Graph Generation (SGG) plays a pivotal role in downstream vision-language tasks. Existing SGG methods typically suffer from poor compositional generalizations on unseen triplets. They are generally trained on incompletely annotated scene graphs that contain dominant triplets and tend to bias toward these seen triplets during inference. To address this issue, we propose a Triplet Calibration and Reduction (T-CAR) framework in this paper. In our framework, a triplet calibration loss is first presented to regularize the representations of diverse triplets and to simultaneously excavate the unseen triplets in incompletely annotated training scene graphs. Moreover, the unseen space of scene graphs is usually several times larger than the seen space since it contains a huge number of unrealistic compositions. Thus, we propose an unseen space reduction loss to shift the attention of excavation to reasonable unseen compositions to facilitate the model training. Finally, we propose a contextual encoder to improve the compositional generalizations of unseen triplets by explicitly modeling the relative spatial relations between subjects and objects. Extensive experiments show that our approach achieves consistent improvements for zero-shot SGG over state-of-the-art methods. The code is available at https://github.com/jkli1998/T-CAR.

{{</citation>}}


### (16/88) Efficient Single Object Detection on Image Patches with Early Exit Enhanced High-Precision CNNs (Arne Moos, 2023)

{{<citation>}}

Arne Moos. (2023)  
**Efficient Single Object Detection on Image Patches with Early Exit Enhanced High-Precision CNNs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.03530v1)  

---


**ABSTRACT**  
This paper proposes a novel approach for detecting objects using mobile robots in the context of the RoboCup Standard Platform League, with a primary focus on detecting the ball. The challenge lies in detecting a dynamic object in varying lighting conditions and blurred images caused by fast movements. To address this challenge, the paper presents a convolutional neural network architecture designed specifically for computationally constrained robotic platforms. The proposed CNN is trained to achieve high precision classification of single objects in image patches and to determine their precise spatial positions. The paper further integrates Early Exits into the existing high-precision CNN architecture to reduce the computational cost of easily rejectable cases in the background class. The training process involves a composite loss function based on confidence and positional losses with dynamic weighting and data augmentation. The proposed approach achieves a precision of 100% on the validation dataset and a recall of almost 87%, while maintaining an execution time of around 170 $\mu$s per hypotheses. By combining the proposed approach with an Early Exit, a runtime optimization of more than 28%, on average, can be achieved compared to the original CNN. Overall, this paper provides an efficient solution for an enhanced detection of objects, especially the ball, in computationally constrained robotic platforms.

{{</citation>}}


### (17/88) Perceptual Quality Assessment of 360$^\circ$ Images Based on Generative Scanpath Representation (Xiangjie Sui et al., 2023)

{{<citation>}}

Xiangjie Sui, Hanwei Zhu, Xuelin Liu, Yuming Fang, Shiqi Wang, Zhou Wang. (2023)  
**Perceptual Quality Assessment of 360$^\circ$ Images Based on Generative Scanpath Representation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.03472v1)  

---


**ABSTRACT**  
Despite substantial efforts dedicated to the design of heuristic models for omnidirectional (i.e., 360$^\circ$) image quality assessment (OIQA), a conspicuous gap remains due to the lack of consideration for the diversity of viewing behaviors that leads to the varying perceptual quality of 360$^\circ$ images. Two critical aspects underline this oversight: the neglect of viewing conditions that significantly sway user gaze patterns and the overreliance on a single viewport sequence from the 360$^\circ$ image for quality inference. To address these issues, we introduce a unique generative scanpath representation (GSR) for effective quality inference of 360$^\circ$ images, which aggregates varied perceptual experiences of multi-hypothesis users under a predefined viewing condition. More specifically, given a viewing condition characterized by the starting point of viewing and exploration time, a set of scanpaths consisting of dynamic visual fixations can be produced using an apt scanpath generator. Following this vein, we use the scanpaths to convert the 360$^\circ$ image into the unique GSR, which provides a global overview of gazed-focused contents derived from scanpaths. As such, the quality inference of the 360$^\circ$ image is swiftly transformed to that of GSR. We then propose an efficient OIQA computational framework by learning the quality maps of GSR. Comprehensive experimental results validate that the predictions of the proposed framework are highly consistent with human perception in the spatiotemporal domain, especially in the challenging context of locally distorted 360$^\circ$ images under varied viewing conditions. The code will be released at https://github.com/xiangjieSui/GSR

{{</citation>}}


### (18/88) Underwater Image Enhancement by Transformer-based Diffusion Model with Non-uniform Sampling for Skip Strategy (Yi Tang et al., 2023)

{{<citation>}}

Yi Tang, Takafumi Iwaguchi, Hiroshi Kawasaki. (2023)  
**Underwater Image Enhancement by Transformer-based Diffusion Model with Non-uniform Sampling for Skip Strategy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.03445v1)  

---


**ABSTRACT**  
In this paper, we present an approach to image enhancement with diffusion model in underwater scenes. Our method adapts conditional denoising diffusion probabilistic models to generate the corresponding enhanced images by using the underwater images and the Gaussian noise as the inputs. Additionally, in order to improve the efficiency of the reverse process in the diffusion model, we adopt two different ways. We firstly propose a lightweight transformer-based denoising network, which can effectively promote the time of network forward per iteration. On the other hand, we introduce a skip sampling strategy to reduce the number of iterations. Besides, based on the skip sampling strategy, we propose two different non-uniform sampling methods for the sequence of the time step, namely piecewise sampling and searching with the evolutionary algorithm. Both of them are effective and can further improve performance by using the same steps against the previous uniform sampling. In the end, we conduct a relative evaluation of the widely used underwater enhancement datasets between the recent state-of-the-art methods and the proposed approach. The experimental results prove that our approach can achieve both competitive performance and high efficiency. Our code is available at \href{mailto:https://github.com/piggy2009/DM_underwater}{\color{blue}{https://github.com/piggy2009/DM\_underwater}}.

{{</citation>}}


## cs.SD (2)



### (19/88) Zero-Shot Audio Captioning via Audibility Guidance (Tal Shaharabany et al., 2023)

{{<citation>}}

Tal Shaharabany, Ariel Shaulov, Lior Wolf. (2023)  
**Zero-Shot Audio Captioning via Audibility Guidance**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: GPT, GPT-4, Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.03884v1)  

---


**ABSTRACT**  
The task of audio captioning is similar in essence to tasks such as image and video captioning. However, it has received much less attention. We propose three desiderata for captioning audio -- (i) fluency of the generated text, (ii) faithfulness of the generated text to the input audio, and the somewhat related (iii) audibility, which is the quality of being able to be perceived based only on audio. Our method is a zero-shot method, i.e., we do not learn to perform captioning. Instead, captioning occurs as an inference process that involves three networks that correspond to the three desired qualities: (i) A Large Language Model, in our case, for reasons of convenience, GPT-2, (ii) A model that provides a matching score between an audio file and a text, for which we use a multimodal matching network called ImageBind, and (iii) A text classifier, trained using a dataset we collected automatically by instructing GPT-4 with prompts designed to direct the generation of both audible and inaudible sentences. We present our results on the AudioCap dataset, demonstrating that audibility guidance significantly enhances performance compared to the baseline, which lacks this objective.

{{</citation>}}


### (20/88) Understanding Self-Supervised Learning of Speech Representation via Invariance and Redundancy Reduction (Yusuf Brima et al., 2023)

{{<citation>}}

Yusuf Brima, Ulf Krumnack, Simone Pika, Gunther Heidemann. (2023)  
**Understanding Self-Supervised Learning of Speech Representation via Invariance and Redundancy Reduction**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.03619v1)  

---


**ABSTRACT**  
The choice of the objective function is crucial in emerging high-quality representations from self-supervised learning. This paper investigates how different formulations of the Barlow Twins (BT) objective impact downstream task performance for speech data. We propose Modified Barlow Twins (MBT) with normalized latents to enforce scale-invariance and evaluate on speaker identification, gender recognition and keyword spotting tasks. Our results show MBT improves representation generalization over original BT, especially when fine-tuning with limited target data. This highlights the importance of designing objectives that encourage invariant and transferable representations. Our analysis provides insights into how the BT learning objective can be tailored to produce speech representations that excel when adapted to new downstream tasks. This study is an important step towards developing reusable self-supervised speech representations.

{{</citation>}}


## cs.CL (18)



### (21/88) DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models (Yung-Sung Chuang et al., 2023)

{{<citation>}}

Yung-Sung Chuang, Yujia Xie, Hongyin Luo, Yoon Kim, James Glass, Pengcheng He. (2023)  
**DoLa: Decoding by Contrasting Layers Improves Factuality in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: LLaMA, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2309.03883v1)  

---


**ABSTRACT**  
Despite their impressive capabilities, large language models (LLMs) are prone to hallucinations, i.e., generating content that deviates from facts seen during pretraining. We propose a simple decoding strategy for reducing hallucinations with pretrained LLMs that does not require conditioning on retrieved external knowledge nor additional fine-tuning. Our approach obtains the next-token distribution by contrasting the differences in logits obtained from projecting the later layers versus earlier layers to the vocabulary space, exploiting the fact that factual knowledge in an LLMs has generally been shown to be localized to particular transformer layers. We find that this Decoding by Contrasting Layers (DoLa) approach is able to better surface factual knowledge and reduce the generation of incorrect facts. DoLa consistently improves the truthfulness across multiple choices tasks and open-ended generation tasks, for example improving the performance of LLaMA family models on TruthfulQA by 12-17% absolute points, demonstrating its potential in making LLMs reliably generate truthful facts.

{{</citation>}}


### (22/88) On Large Language Models' Selection Bias in Multi-Choice Questions (Chujie Zheng et al., 2023)

{{<citation>}}

Chujie Zheng, Hao Zhou, Fandong Meng, Jie Zhou, Minlie Huang. (2023)  
**On Large Language Models' Selection Bias in Multi-Choice Questions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model  
[Paper Link](http://arxiv.org/abs/2309.03882v1)  

---


**ABSTRACT**  
Multi-choice questions (MCQs) serve as a common yet important task format in the research of large language models (LLMs). Our work shows that LLMs exhibit an inherent "selection bias" in MCQs, which refers to LLMs' preferences to select options located at specific positions (like "Option C"). This bias is prevalent across various LLMs, making their performance vulnerable to option position changes in MCQs. We identify that one primary cause resulting in selection bias is option numbering, i.e., the ID symbols A/B/C/D associated with the options. To mitigate selection bias, we propose a new method called PriDe. PriDe first decomposes the observed model prediction distribution into an intrinsic prediction over option contents and a prior distribution over option IDs. It then estimates the prior by permutating option contents on a small number of test samples, which is used to debias the subsequent test samples. We demonstrate that, as a label-free, inference-time method, PriDe achieves a more effective and computation-efficient debiasing than strong baselines. We further show that the priors estimated by PriDe generalize well across different domains, highlighting its practical potential in broader scenarios.

{{</citation>}}


### (23/88) Introducing 'Forecast Utterance' for Conversational Data Science (Md Mahadi Hassan et al., 2023)

{{<citation>}}

Md Mahadi Hassan, Alex Knipper, Shubhra Kanti Karmaker. (2023)  
**Introducing 'Forecast Utterance' for Conversational Data Science**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.03877v1)  

---


**ABSTRACT**  
Envision an intelligent agent capable of assisting users in conducting forecasting tasks through intuitive, natural conversations, without requiring in-depth knowledge of the underlying machine learning (ML) processes. A significant challenge for the agent in this endeavor is to accurately comprehend the user's prediction goals and, consequently, formulate precise ML tasks. In this paper, we take a pioneering step towards this ambitious goal by introducing a new concept called Forecast Utterance and then focus on the automatic and accurate interpretation of users' prediction goals from these utterances. Specifically, we frame the task as a slot-filling problem, where each slot corresponds to a specific aspect of the goal prediction task. We then employ two zero-shot methods for solving the slot-filling task, namely: 1) Entity Extraction (EE), and 2) Question-Answering (QA) techniques. Our experiments, conducted with three meticulously crafted data sets, validate the viability of our ambitious goal and demonstrate the effectiveness of both EE and QA techniques in interpreting Forecast Utterances.

{{</citation>}}


### (24/88) OpinionGPT: Modelling Explicit Biases in Instruction-Tuned LLMs (Patrick Haller et al., 2023)

{{<citation>}}

Patrick Haller, Ansar Aynetdinov, Alan Akbik. (2023)  
**OpinionGPT: Modelling Explicit Biases in Instruction-Tuned LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: Bias, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.03876v1)  

---


**ABSTRACT**  
Instruction-tuned Large Language Models (LLMs) have recently showcased remarkable ability to generate fitting responses to natural language instructions. However, an open research question concerns the inherent biases of trained models and their responses. For instance, if the data used to tune an LLM is dominantly written by persons with a specific political bias, we might expect generated answers to share this bias. Current research work seeks to de-bias such models, or suppress potentially biased answers. With this demonstration, we take a different view on biases in instruction-tuning: Rather than aiming to suppress them, we aim to make them explicit and transparent. To this end, we present OpinionGPT, a web demo in which users can ask questions and select all biases they wish to investigate. The demo will answer this question using a model fine-tuned on text representing each of the selected biases, allowing side-by-side comparison. To train the underlying model, we identified 11 different biases (political, geographic, gender, age) and derived an instruction-tuning corpus in which each answer was written by members of one of these demographics. This paper presents OpinionGPT, illustrates how we trained the bias-aware model and showcases the web application (available at https://opiniongpt.informatik.hu-berlin.de).

{{</citation>}}


### (25/88) FLM-101B: An Open LLM and How to Train It with $100K Budget (Xiang Li et al., 2023)

{{<citation>}}

Xiang Li, Yiqun Yao, Xin Jiang, Xuezhi Fang, Xuying Meng, Siqi Fan, Peng Han, Jing Li, Li Du, Bowen Qin, Zheng Zhang, Aixin Sun, Yequan Wang. (2023)  
**FLM-101B: An Open LLM and How to Train It with $100K Budget**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, GLM, GPT, NLP  
[Paper Link](http://arxiv.org/abs/2309.03852v1)  

---


**ABSTRACT**  
Large language models (LLMs) have achieved remarkable success in NLP and multimodal tasks. Despite these successes, their development faces two main challenges: (i) high computational cost; and (ii) difficulty in conducting fair and objective evaluations. LLMs are prohibitively expensive, making it feasible for only a few major players to undertake their training, thereby constraining both research and application opportunities. This underscores the importance of cost-effective LLM training. In this paper, we utilize a growth strategy to significantly reduce LLM training cost. We demonstrate that an LLM with 101B parameters and 0.31TB tokens can be trained on a $100K budget. We also adopt a systematic evaluation paradigm for the IQ evaluation of LLMs, in complement to existing evaluations that focus more on knowledge-oriented abilities. We introduce our benchmark including evaluations on important aspects of intelligence including symbolic mapping, itrule understanding, pattern mining, and anti-interference. Such evaluations minimize the potential impact of memorization. Experimental results show that our model FLM-101B, trained with a budget of $100K, achieves comparable performance to powerful and well-known models, eg GPT-3 and GLM-130B, especially in the IQ benchmark evaluations with contexts unseen in training data. The checkpoint of FLM-101B will be open-sourced at https://huggingface.co/CofeAI/FLM-101B.

{{</citation>}}


### (26/88) USA: Universal Sentiment Analysis Model & Construction of Japanese Sentiment Text Classification and Part of Speech Dataset (Chengguang Gan et al., 2023)

{{<citation>}}

Chengguang Gan, Qinghao Zhang, Tatsunori Mori. (2023)  
**USA: Universal Sentiment Analysis Model & Construction of Japanese Sentiment Text Classification and Part of Speech Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Sentiment Analysis, Text Classification  
[Paper Link](http://arxiv.org/abs/2309.03787v1)  

---


**ABSTRACT**  
Sentiment analysis is a pivotal task in the domain of natural language processing. It encompasses both text-level sentiment polarity classification and word-level Part of Speech(POS) sentiment polarity determination. Such analysis challenges models to understand text holistically while also extracting nuanced information. With the rise of Large Language Models(LLMs), new avenues for sentiment analysis have opened. This paper proposes enhancing performance by leveraging the Mutual Reinforcement Effect(MRE) between individual words and the overall text. It delves into how word polarity influences the overarching sentiment of a passage. To support our research, we annotated four novel Sentiment Text Classification and Part of Speech(SCPOS) datasets, building upon existing sentiment classification datasets. Furthermore, we developed a Universal Sentiment Analysis(USA) model, with a 7-billion parameter size. Experimental results revealed that our model surpassed the performance of gpt-3.5-turbo across all four datasets, underscoring the significance of MRE in sentiment analysis.

{{</citation>}}


### (27/88) Enhancing Pipeline-Based Conversational Agents with Large Language Models (Mina Foosherian et al., 2023)

{{<citation>}}

Mina Foosherian, Hendrik Purwins, Purna Rathnayake, Touhidul Alam, Rui Teimao, Klaus-Dieter Thoben. (2023)  
**Enhancing Pipeline-Based Conversational Agents with Large Language Models**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-7, cs-AI, cs-CL, cs-LG, cs.CL, stat-ML  
Keywords: AI, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2309.03748v1)  

---


**ABSTRACT**  
The latest advancements in AI and deep learning have led to a breakthrough in large language model (LLM)-based agents such as GPT-4. However, many commercial conversational agent development tools are pipeline-based and have limitations in holding a human-like conversation. This paper investigates the capabilities of LLMs to enhance pipeline-based conversational agents during two phases: 1) in the design and development phase and 2) during operations. In 1) LLMs can aid in generating training data, extracting entities and synonyms, localization, and persona design. In 2) LLMs can assist in contextualization, intent classification to prevent conversational breakdown and handle out-of-scope questions, auto-correcting utterances, rephrasing responses, formulating disambiguation questions, summarization, and enabling closed question-answering capabilities. We conducted informal experiments with GPT-4 in the private banking domain to demonstrate the scenarios above with a practical example. Companies may be hesitant to replace their pipeline-based agents with LLMs entirely due to privacy concerns and the need for deep integration within their existing ecosystems. A hybrid approach in which LLMs' are integrated into the pipeline-based agents allows them to save time and costs of building and running agents by capitalizing on the capabilities of LLMs while retaining the integration and privacy safeguards of their existing systems.

{{</citation>}}


### (28/88) The Daunting Dilemma with Sentence Encoders: Success on Standard Benchmarks, Failure in Capturing Basic Semantic Properties (Yash Mahajan et al., 2023)

{{<citation>}}

Yash Mahajan, Naman Bansal, Shubhra Kanti Karmaker. (2023)  
**The Daunting Dilemma with Sentence Encoders: Success on Standard Benchmarks, Failure in Capturing Basic Semantic Properties**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, NLP  
[Paper Link](http://arxiv.org/abs/2309.03747v1)  

---


**ABSTRACT**  
In this paper, we adopted a retrospective approach to examine and compare five existing popular sentence encoders, i.e., Sentence-BERT, Universal Sentence Encoder (USE), LASER, InferSent, and Doc2vec, in terms of their performance on downstream tasks versus their capability to capture basic semantic properties. Initially, we evaluated all five sentence encoders on the popular SentEval benchmark and found that multiple sentence encoders perform quite well on a variety of popular downstream tasks. However, being unable to find a single winner in all cases, we designed further experiments to gain a deeper understanding of their behavior. Specifically, we proposed four semantic evaluation criteria, i.e., Paraphrasing, Synonym Replacement, Antonym Replacement, and Sentence Jumbling, and evaluated the same five sentence encoders using these criteria. We found that the Sentence-Bert and USE models pass the paraphrasing criterion, with SBERT being the superior between the two. LASER dominates in the case of the synonym replacement criterion. Interestingly, all the sentence encoders failed the antonym replacement and jumbling criteria. These results suggest that although these popular sentence encoders perform quite well on the SentEval benchmark, they still struggle to capture some basic semantic properties, thus, posing a daunting dilemma in NLP research.

{{</citation>}}


### (29/88) Exploring an LM to generate Prolog Predicates from Mathematics Questions (Xiaocheng Yang et al., 2023)

{{<citation>}}

Xiaocheng Yang, Yik-Cheung Tam. (2023)  
**Exploring an LM to generate Prolog Predicates from Mathematics Questions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, LLaMA, NLP  
[Paper Link](http://arxiv.org/abs/2309.03667v1)  

---


**ABSTRACT**  
Recently, there has been a surge in interest in NLP driven by ChatGPT. ChatGPT, a transformer-based generative language model of substantial scale, exhibits versatility in performing various tasks based on natural language. Nevertheless, large language models often exhibit poor performance in solving mathematics questions that require reasoning. Prior research has demonstrated the effectiveness of chain-of-thought prompting in enhancing reasoning capabilities. Now, we aim to investigate whether fine-tuning a model for the generation of Prolog codes, a logic language, and subsequently passing these codes to a compiler can further improve accuracy. Consequently, we employ chain-of-thought to fine-tune LLaMA7B as a baseline model and develop other fine-tuned LLaMA7B models for the generation of Prolog code, Prolog code + chain-of-thought, and chain-of-thought + Prolog code, respectively. The results reveal that the Prolog generation model surpasses the baseline in performance, while the combination generation models do not yield significant improvements. The Prolog corpus based on GSM8K and the correspondingly finetuned Prolog generation model based on LLaMA7B are released to the research community.

{{</citation>}}


### (30/88) BNS-Net: A Dual-channel Sarcasm Detection Method Considering Behavior-level and Sentence-level Conflicts (Liming Zhou et al., 2023)

{{<citation>}}

Liming Zhou, Xiaowei Xu, Xiaodong Wang. (2023)  
**BNS-Net: A Dual-channel Sarcasm Detection Method Considering Behavior-level and Sentence-level Conflicts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Sarcasm Detection  
[Paper Link](http://arxiv.org/abs/2309.03658v1)  

---


**ABSTRACT**  
Sarcasm detection is a binary classification task that aims to determine whether a given utterance is sarcastic. Over the past decade, sarcasm detection has evolved from classical pattern recognition to deep learning approaches, where features such as user profile, punctuation and sentiment words have been commonly employed for sarcasm detection. In real-life sarcastic expressions, behaviors without explicit sentimental cues often serve as carriers of implicit sentimental meanings. Motivated by this observation, we proposed a dual-channel sarcasm detection model named BNS-Net. The model considers behavior and sentence conflicts in two channels. Channel 1: Behavior-level Conflict Channel reconstructs the text based on core verbs while leveraging the modified attention mechanism to highlight conflict information. Channel 2: Sentence-level Conflict Channel introduces external sentiment knowledge to segment the text into explicit and implicit sentences, capturing conflicts between them. To validate the effectiveness of BNS-Net, several comparative and ablation experiments are conducted on three public sarcasm datasets. The analysis and evaluation of experimental results demonstrate that the BNS-Net effectively identifies sarcasm in text and achieves the state-of-the-art performance.

{{</citation>}}


### (31/88) Loquacity and Visible Emotion: ChatGPT as a Policy Advisor (Claudia Biancotti et al., 2023)

{{<citation>}}

Claudia Biancotti, Carolina Camassa. (2023)  
**Loquacity and Visible Emotion: ChatGPT as a Policy Advisor**  

---
Primary Category: cs.CL  
Categories: J-4; K-4-1, cs-CL, cs-HC, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.03595v1)  

---


**ABSTRACT**  
ChatGPT, a software seeking to simulate human conversational abilities, is attracting increasing attention. It is sometimes portrayed as a groundbreaking productivity aid, including for creative work. In this paper, we run an experiment to assess its potential in complex writing tasks. We ask the software to compose a policy brief for the Board of the Bank of Italy. We find that ChatGPT can accelerate workflows by providing well-structured content suggestions, and by producing extensive, linguistically correct text in a matter of seconds. It does, however, require a significant amount of expert supervision, which partially offsets productivity gains. If the app is used naively, output can be incorrect, superficial, or irrelevant. Superficiality is an especially problematic limitation in the context of policy advice intended for high-level audiences.

{{</citation>}}


### (32/88) Evaluating the Efficacy of Supervised Learning vs Large Language Models for Identifying Cognitive Distortions and Suicidal Risks in Chinese Social Media (Hongzhi Qi et al., 2023)

{{<citation>}}

Hongzhi Qi, Qing Zhao, Changwei Song, Wei Zhai, Dan Luo, Shuo Liu, Yi Jing Yu, Fan Wang, Huijing Zou, Bing Xiang Yang, Jianqiang Li, Guanghui Fu. (2023)  
**Evaluating the Efficacy of Supervised Learning vs Large Language Models for Identifying Cognitive Distortions and Suicidal Risks in Chinese Social Media**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-3.5, GPT-4, Language Model, Social Media  
[Paper Link](http://arxiv.org/abs/2309.03564v1)  

---


**ABSTRACT**  
Large language models, particularly those akin to the rapidly progressing GPT series, are gaining traction for their expansive influence. While there is keen interest in their applicability within medical domains such as psychology, tangible explorations on real-world data remain scant. Concurrently, users on social media platforms are increasingly vocalizing personal sentiments; under specific thematic umbrellas, these sentiments often manifest as negative emotions, sometimes escalating to suicidal inclinations. Timely discernment of such cognitive distortions and suicidal risks is crucial to effectively intervene and potentially avert dire circumstances. Our study ventured into this realm by experimenting on two pivotal tasks: suicidal risk and cognitive distortion identification on Chinese social media platforms. Using supervised learning as a baseline, we examined and contrasted the efficacy of large language models via three distinct strategies: zero-shot, few-shot, and fine-tuning. Our findings revealed a discernible performance gap between the large language models and traditional supervised learning approaches, primarily attributed to the models' inability to fully grasp subtle categories. Notably, while GPT-4 outperforms its counterparts in multiple scenarios, GPT-3.5 shows significant enhancement in suicide risk classification after fine-tuning. To our knowledge, this investigation stands as the maiden attempt at gauging large language models on Chinese social media tasks. This study underscores the forward-looking and transformative implications of using large language models in the field of psychology. It lays the groundwork for future applications in psychological research and practice.

{{</citation>}}


### (33/88) All Labels Together: Low-shot Intent Detection with an Efficient Label Semantic Encoding Paradigm (Jiangshu Du et al., 2023)

{{<citation>}}

Jiangshu Du, Congying Xia, Wenpeng Yin, Tingting Liang, Philip S. Yu. (2023)  
**All Labels Together: Low-shot Intent Detection with an Efficient Label Semantic Encoding Paradigm**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Intent Detection  
[Paper Link](http://arxiv.org/abs/2309.03563v1)  

---


**ABSTRACT**  
In intent detection tasks, leveraging meaningful semantic information from intent labels can be particularly beneficial for few-shot scenarios. However, existing few-shot intent detection methods either ignore the intent labels, (e.g. treating intents as indices) or do not fully utilize this information (e.g. only using part of the intent labels). In this work, we present an end-to-end One-to-All system that enables the comparison of an input utterance with all label candidates. The system can then fully utilize label semantics in this way. Experiments on three few-shot intent detection tasks demonstrate that One-to-All is especially effective when the training resource is extremely scarce, achieving state-of-the-art performance in 1-, 3- and 5-shot settings. Moreover, we present a novel pretraining strategy for our model that utilizes indirect supervision from paraphrasing, enabling zero-shot cross-domain generalization on intent detection tasks. Our code is at https://github.com/jiangshdd/AllLablesTogethe.

{{</citation>}}


### (34/88) An Anchor Learning Approach for Citation Field Learning (Zilin Yuan et al., 2023)

{{<citation>}}

Zilin Yuan, Borun Chen, Yimeng Dai, Yinghui Li, Hai-Tao Zheng, Rui Zhang. (2023)  
**An Anchor Learning Approach for Citation Field Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.03559v1)  

---


**ABSTRACT**  
Citation field learning is to segment a citation string into fields of interest such as author, title, and venue. Extracting such fields from citations is crucial for citation indexing, researcher profile analysis, etc. User-generated resources like academic homepages and Curriculum Vitae, provide rich citation field information. However, extracting fields from these resources is challenging due to inconsistent citation styles, incomplete sentence syntax, and insufficient training data. To address these challenges, we propose a novel algorithm, CIFAL (citation field learning by anchor learning), to boost the citation field learning performance. CIFAL leverages the anchor learning, which is model-agnostic for any Pre-trained Language Model, to help capture citation patterns from the data of different citation styles. The experiments demonstrate that CIFAL outperforms state-of-the-art methods in citation field learning, achieving a 2.83% improvement in field-level F1-scores. Extensive analysis of the results further confirms the effectiveness of CIFAL quantitatively and qualitatively.

{{</citation>}}


### (35/88) Machine Learning for Tangible Effects: Natural Language Processing for Uncovering the Illicit Massage Industry & Computer Vision for Tactile Sensing (Rui Ouyang, 2023)

{{<citation>}}

Rui Ouyang. (2023)  
**Machine Learning for Tangible Effects: Natural Language Processing for Uncovering the Illicit Massage Industry & Computer Vision for Tactile Sensing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-SI, cs.CL  
Keywords: Computer Vision, Google, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.03470v1)  

---


**ABSTRACT**  
I explore two questions in this thesis: how can computer science be used to fight human trafficking? And how can computer vision create a sense of touch?   I use natural language processing (NLP) to monitor the United States illicit massage industry (IMI), a multi-billion dollar industry that offers not just therapeutic massages but also commercial sexual services. Employees of this industry are often immigrant women with few job opportunities, leaving them vulnerable to fraud, coercion, and other facets of human trafficking. Monitoring spatiotemporal trends helps prevent trafficking in the IMI. By creating datasets with three publicly-accessible websites: Google Places, Rubmaps, and AMPReviews, combined with NLP techniques such as bag-of-words and Word2Vec, I show how to derive insights into the labor pressures and language barriers that employees face, as well as the income, demographics, and societal pressures affecting sex buyers. I include a call-to-action to other researchers given these datasets. I also consider how to creating synthetic financial data, which can aid with counter-trafficking in the banking sector. I use an agent-based model to create both tabular and payee-recipient graph data.   I then consider the role of computer vision in making tactile sensors. I report on a novel sensor, the Digger Finger, that adapts the Gelsight sensor to finding objects in granular media. Changes include using a wedge shape to facilitate digging, replacing the internal lighting LEDs with fluorescent paint, and adding a vibrator motor to counteract jamming. Finally, I also show how to use a webcam and a printed reference marker, or fiducial, to create a low-cost six-axis force-torque sensor. This sensor is up to a hundred times less expensive than commercial sensors, allowing for a wider range of applications. For this and earlier chapters I release design files and code as open source.

{{</citation>}}


### (36/88) XGen-7B Technical Report (Erik Nijkamp et al., 2023)

{{<citation>}}

Erik Nijkamp, Tian Xie, Hiroaki Hayashi, Bo Pang, Congying Xia, Chen Xing, Jesse Vig, Semih Yavuz, Philippe Laban, Ben Krause, Senthil Purushwalkam, Tong Niu, Wojciech Kryściński, Lidiya Murakhovs'ka, Prafulla Kumar Choubey, Alex Fabbri, Ye Liu, Rui Meng, Lifu Tu, Meghana Bhat, Chien-Sheng Wu, Silvio Savarese, Yingbo Zhou, Shafiq Joty, Caiming Xiong. (2023)  
**XGen-7B Technical Report**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.03450v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have become ubiquitous across various domains, transforming the way we interact with information and conduct research. However, most high-performing LLMs remain confined behind proprietary walls, hindering scientific progress. Most open-source LLMs, on the other hand, are limited in their ability to support longer sequence lengths, which is a key requirement for many tasks that require inference over an input context. To address this, we have trained XGen, a series of 7B parameter models on up to 8K sequence length for up to 1.5T tokens. We have also finetuned the XGen models on public-domain instructional data, creating their instruction-tuned counterparts (XGen-Inst). We open-source our models for both research advancements and commercial applications. Our evaluation on standard benchmarks shows that XGen models achieve comparable or better results when compared with state-of-the-art open-source LLMs. Our targeted evaluation on long sequence modeling tasks shows the benefits of our 8K-sequence models over 2K-sequence open-source LLMs.

{{</citation>}}


### (37/88) Improving Open Information Extraction with Large Language Models: A Study on Demonstration Uncertainty (Chen Ling et al., 2023)

{{<citation>}}

Chen Ling, Xujiang Zhao, Xuchao Zhang, Yanchi Liu, Wei Cheng, Haoyu Wang, Zhengzhang Chen, Takao Osaki, Katsushi Matsuda, Haifeng Chen, Liang Zhao. (2023)  
**Improving Open Information Extraction with Large Language Models: A Study on Demonstration Uncertainty**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Information Extraction, Language Model  
[Paper Link](http://arxiv.org/abs/2309.03433v1)  

---


**ABSTRACT**  
Open Information Extraction (OIE) task aims at extracting structured facts from unstructured text, typically in the form of (subject, relation, object) triples. Despite the potential of large language models (LLMs) like ChatGPT as a general task solver, they lag behind state-of-the-art (supervised) methods in OIE tasks due to two key issues. First, LLMs struggle to distinguish irrelevant context from relevant relations and generate structured output due to the restrictions on fine-tuning the model. Second, LLMs generates responses autoregressively based on probability, which makes the predicted relations lack confidence. In this paper, we assess the capabilities of LLMs in improving the OIE task. Particularly, we propose various in-context learning strategies to enhance LLM's instruction-following ability and a demonstration uncertainty quantification module to enhance the confidence of the generated relations. Our experiments on three OIE benchmark datasets show that our approach holds its own against established supervised methods, both quantitatively and qualitatively.

{{</citation>}}


### (38/88) From Base to Conversational: Japanese Instruction Dataset and Tuning Large Language Models (Masahiro Suzuki et al., 2023)

{{<citation>}}

Masahiro Suzuki, Masanori Hirano, Hiroki Sakaji. (2023)  
**From Base to Conversational: Japanese Instruction Dataset and Tuning Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.03412v1)  

---


**ABSTRACT**  
Instruction tuning is essential for large language models (LLMs) to become interactive. While many instruction tuning datasets exist in English, there is a noticeable lack in other languages. Also, their effectiveness has not been well verified in non-English languages. We construct a Japanese instruction dataset by expanding and filtering existing datasets and apply the dataset to a Japanese pre-trained base model. We performed Low-Rank Adaptation (LoRA) tuning on both Japanese and English existing models using our instruction dataset. We evaluated these models from both quantitative and qualitative perspectives. As a result, the effectiveness of Japanese instruction datasets is confirmed. The results also indicate that even with relatively small LLMs, performances in downstream tasks would be improved through instruction tuning. Our instruction dataset, tuned models, and implementation are publicly available online.

{{</citation>}}


## cs.SI (2)



### (39/88) Network Sampling Methods for Estimating Social Networks, Population Percentages and Totals of People Experiencing Homelessness (Zack W. Almquist et al., 2023)

{{<citation>}}

Zack W. Almquist, Ashley Hazel, Mary-Catherine Anderson, Larisa Ozeryansky, Amy Hagopian. (2023)  
**Network Sampling Methods for Estimating Social Networks, Population Percentages and Totals of People Experiencing Homelessness**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI, stat-ME  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2309.03875v1)  

---


**ABSTRACT**  
In this article, we propose using network-based sampling strategies to estimate the number of unsheltered people experiencing homelessness within a given administrative service unit, known as a Continuum of Care. Further, we specifically advocate for the network sampling method known as Respondent Driven Sampling (RDS), which has been shown to provide unbiased or low-biased estimates of totals and proportions for hard-to-reach populations in contexts where a sampling frame (e.g., housing addresses) not available. To make the RDS estimator work for estimating the total number of unsheltered people, we introduce a new method that leverages administrative data from the HUD-mandated Homeless Management Information System (HMIS). The HMIS provides high-quality counts and demographics for people experiencing homelessness who sleep in emergency shelters. We then demonstrate this method using network data collected in Nashville, TN, combined with simulation methods to illustrate the efficacy of this approach. Finally, we end with discussing how this could be used in practice.

{{</citation>}}


### (40/88) User's Reaction Patterns in Online Social Network Communities (Azza Bouleimen et al., 2023)

{{<citation>}}

Azza Bouleimen, Nicolò Pagan, Stefano Cresci, Aleksandra Urman, Gianluca Nogara, Silvia Giordano. (2023)  
**User's Reaction Patterns in Online Social Network Communities**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2309.03701v1)  

---


**ABSTRACT**  
Several one-fits-all intervention policies were introduced by the Online Social Networks (OSNs) platforms to mitigate potential harms. Nevertheless, some studies showed the limited effectiveness of these approaches. An alternative to this would be a user-centered design of intervention policies. In this context, we study the susceptibility of users to undesired behavior in communities on OSNs. In particular, we explore their reaction to specific events. Our study shows that communities develop different undesired behavior patterns in reaction to specific events. These events can significantly alter the behavior of the community and invert the dynamics of behavior within the whole network. Our findings stress out the importance of understanding the reasons behind the changes in users' reactions and highlights the need of fine-tuning the research to the individual's level. It paves the way towards building better OSNs' intervention strategies centered on the user.

{{</citation>}}


## cs.NI (3)



### (41/88) Experimental Study of Adversarial Attacks on ML-based xApps in O-RAN (Naveen Naik Sapavath et al., 2023)

{{<citation>}}

Naveen Naik Sapavath, Brian Kim, Kaushik Chowdhury, Vijay K Shah. (2023)  
**Experimental Study of Adversarial Attacks on ML-based xApps in O-RAN**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI, eess-SP  
Keywords: AI, Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2309.03844v1)  

---


**ABSTRACT**  
Open Radio Access Network (O-RAN) is considered as a major step in the evolution of next-generation cellular networks given its support for open interfaces and utilization of artificial intelligence (AI) into the deployment, operation, and maintenance of RAN. However, due to the openness of the O-RAN architecture, such AI models are inherently vulnerable to various adversarial machine learning (ML) attacks, i.e., adversarial attacks which correspond to slight manipulation of the input to the ML model. In this work, we showcase the vulnerability of an example ML model used in O-RAN, and experimentally deploy it in the near-real time (near-RT) RAN intelligent controller (RIC). Our ML-based interference classifier xApp (extensible application in near-RT RIC) tries to classify the type of interference to mitigate the interference effect on the O-RAN system. We demonstrate the first-ever scenario of how such an xApp can be impacted through an adversarial attack by manipulating the data stored in a shared database inside the near-RT RIC. Through a rigorous performance analysis deployed on a laboratory O-RAN testbed, we evaluate the performance in terms of capacity and the prediction accuracy of the interference classifier xApp using both clean and perturbed data. We show that even small adversarial attacks can significantly decrease the accuracy of ML application in near-RT RIC, which can directly impact the performance of the entire O-RAN deployment.

{{</citation>}}


### (42/88) HSTF-Model: an HTTP-based Trojan Detection Model via the Hierarchical Spatio-Temporal Features of Traffics (Jiang Xie et al., 2023)

{{<citation>}}

Jiang Xie, Shuhao Lia, Xiaochun Yun, Yongzheng Zhang, Peng Chang. (2023)  
**HSTF-Model: an HTTP-based Trojan Detection Model via the Hierarchical Spatio-Temporal Features of Traffics**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.03724v1)  

---


**ABSTRACT**  
HTTP-based Trojan is extremely threatening, and it is difficult to be effectively detected because of its concealment and confusion. Previous detection methods usually are with poor generalization ability due to outdated datasets and reliance on manual feature extraction, which makes these methods always perform well under their private dataset, but poorly or even fail to work in real network environment. In this paper, we propose an HTTP-based Trojan detection model via the Hierarchical Spatio-Temporal Features of traffics (HSTF-Model) based on the formalized description of traffic spatio-temporal behavior from both packet level and flow level. In this model, we employ Convolutional Neural Network (CNN) to extract spatial information and Long Short-Term Memory (LSTM) to extract temporal information. In addition, we present a dataset consisting of Benign and Trojan HTTP Traffic (BTHT-2018). Experimental results show that our model can guarantee high accuracy (the F1 of 98.62%-99.81% and the FPR of 0.34%-0.02% in BTHT-2018). More importantly, our model has a huge advantage over other related methods in generalization ability. HSTF-Model trained with BTHT-2018 can reach the F1 of 93.51% on the public dataset ISCX-2012, which is 20+% better than the best of related machine learning methods.

{{</citation>}}


### (43/88) Enhancing 5G Radio Planning with Graph Representations and Deep Learning (Paul Almasan et al., 2023)

{{<citation>}}

Paul Almasan, José Suárez-Varela, Andra Lutu, Albert Cabellos-Aparicio, Pere Barlet-Ros. (2023)  
**Enhancing 5G Radio Planning with Graph Representations and Deep Learning**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.03603v1)  

---


**ABSTRACT**  
The roll out of new mobile network generations poses hard challenges due to various factors such as cost-benefit tradeoffs, existing infrastructure, and new technology aspects. In particular, one of the main challenges for the 5G deployment lies in optimal 5G radio coverage while accounting for diverse service performance metrics. This paper introduces a Deep Learning-based approach to assist in 5G radio planning by utilizing data from previous-generation cells. Our solution relies on a custom graph representation to leverage the information available from existing cells, and employs a Graph Neural Network (GNN) model to process such data efficiently. In our evaluation, we test its potential to model the transition from 4G to 5G NSA using real-world data from a UK mobile network operator. The experimental results show that our solution achieves high accuracy in predicting key performance indicators in new 5G cells, with a Mean Absolute Percentage Error (MAPE)~<17\% when evaluated on samples from the same area where it was trained. Moreover, we test its generalization capability over various geographical areas not included in the training, achieving a MAPE~<19\%. This suggests beneficial properties for achieving robust solutions applicable to 5G planning in new areas without the need of retraining.

{{</citation>}}


## cs.RO (5)



### (44/88) Bootstrapping Adaptive Human-Machine Interfaces with Offline Reinforcement Learning (Jensen Gao et al., 2023)

{{<citation>}}

Jensen Gao, Siddharth Reddy, Glen Berseth, Anca D. Dragan, Sergey Levine. (2023)  
**Bootstrapping Adaptive Human-Machine Interfaces with Offline Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-HC, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.03839v1)  

---


**ABSTRACT**  
Adaptive interfaces can help users perform sequential decision-making tasks like robotic teleoperation given noisy, high-dimensional command signals (e.g., from a brain-computer interface). Recent advances in human-in-the-loop machine learning enable such systems to improve by interacting with users, but tend to be limited by the amount of data that they can collect from individual users in practice. In this paper, we propose a reinforcement learning algorithm to address this by training an interface to map raw command signals to actions using a combination of offline pre-training and online fine-tuning. To address the challenges posed by noisy command signals and sparse rewards, we develop a novel method for representing and inferring the user's long-term intent for a given trajectory. We primarily evaluate our method's ability to assist users who can only communicate through noisy, high-dimensional input channels through a user study in which 12 participants performed a simulated navigation task by using their eye gaze to modulate a 128-dimensional command signal from their webcam. The results show that our method enables successful goal navigation more often than a baseline directional interface, by learning to denoise user commands signals and provide shared autonomy assistance. We further evaluate on a simulated Sawyer pushing task with eye gaze control, and the Lunar Lander game with simulated user commands, and find that our method improves over baseline interfaces in these domains as well. Extensive ablation experiments with simulated user commands empirically motivate each component of our method.

{{</citation>}}


### (45/88) Hybrid of representation learning and reinforcement learning for dynamic and complex robotic motion planning (Chengmin Zhou et al., 2023)

{{<citation>}}

Chengmin Zhou, Xin Lu, Jiapeng Dai, Bingding Huang, Xiaoxu Liu, Pasi Fränti. (2023)  
**Hybrid of representation learning and reinforcement learning for dynamic and complex robotic motion planning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.03758v1)  

---


**ABSTRACT**  
Motion planning is the soul of robot decision making. Classical planning algorithms like graph search and reaction-based algorithms face challenges in cases of dense and dynamic obstacles. Deep learning algorithms generate suboptimal one-step predictions that cause many collisions. Reinforcement learning algorithms generate optimal or near-optimal time-sequential predictions. However, they suffer from slow convergence, suboptimal converged results, and overfittings. This paper introduces a hybrid algorithm for robotic motion planning: long short-term memory (LSTM) pooling and skip connection for attention-based discrete soft actor critic (LSA-DSAC). First, graph network (relational graph) and attention network (attention weight) interpret the environmental state for the learning of the discrete soft actor critic algorithm. The expressive power of attention network outperforms that of graph in our task by difference analysis of these two representation methods. However, attention based DSAC faces the overfitting problem in training. Second, the skip connection method is integrated to attention based DSAC to mitigate overfitting and improve convergence speed. Third, LSTM pooling is taken to replace the sum operator of attention weigh and eliminate overfitting by slightly sacrificing convergence speed at early-stage training. Experiments show that LSA-DSAC outperforms the state-of-the-art in training and most evaluations. The physical robot is also implemented and tested in the real world.

{{</citation>}}


### (46/88) Chat Failures and Troubles: Reasons and Solutions (Manal Helal et al., 2023)

{{<citation>}}

Manal Helal, Patrick Holthaus, Gabriella Lakatos, Farshid Amirabdollahian. (2023)  
**Chat Failures and Troubles: Reasons and Solutions**  

---
Primary Category: cs.RO  
Categories: cs-HC, cs-LG, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.03708v1)  

---


**ABSTRACT**  
This paper examines some common problems in Human-Robot Interaction (HRI) causing failures and troubles in Chat. A given use case's design decisions start with the suitable robot, the suitable chatting model, identifying common problems that cause failures, identifying potential solutions, and planning continuous improvement. In conclusion, it is recommended to use a closed-loop control algorithm that guides the use of trained Artificial Intelligence (AI) pre-trained models and provides vocabulary filtering, re-train batched models on new datasets, learn online from data streams, and/or use reinforcement learning models to self-update the trained models and reduce errors.

{{</citation>}}


### (47/88) Fully Onboard SLAM for Distributed Mapping with a Swarm of Nano-Drones (Carl Friess et al., 2023)

{{<citation>}}

Carl Friess, Vlad Niculescu, Tommaso Polonelli, Michele Magno, Luca Benini. (2023)  
**Fully Onboard SLAM for Distributed Mapping with a Swarm of Nano-Drones**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2309.03678v1)  

---


**ABSTRACT**  
The use of Unmanned Aerial Vehicles (UAVs) is rapidly increasing in applications ranging from surveillance and first-aid missions to industrial automation involving cooperation with other machines or humans. To maximize area coverage and reduce mission latency, swarms of collaborating drones have become a significant research direction. However, this approach requires open challenges in positioning, mapping, and communications to be addressed. This work describes a distributed mapping system based on a swarm of nano-UAVs, characterized by a limited payload of 35 g and tightly constrained on-board sensing and computing capabilities. Each nano-UAV is equipped with four 64-pixel depth sensors that measure the relative distance to obstacles in four directions. The proposed system merges the information from the swarm and generates a coherent grid map without relying on any external infrastructure. The data fusion is performed using the iterative closest point algorithm and a graph-based simultaneous localization and mapping algorithm, running entirely on-board the UAV's low-power ARM Cortex-M microcontroller with just 192 kB of SRAM memory. Field results gathered in three different mazes from a swarm of up to 4 nano-UAVs prove a mapping accuracy of 12 cm and demonstrate that the mapping time is inversely proportional to the number of agents. The proposed framework scales linearly in terms of communication bandwidth and on-board computational complexity, supporting communication between up to 20 nano-UAVs and mapping of areas up to 180 m2 with the chosen configuration requiring only 50 kB of memory.

{{</citation>}}


### (48/88) InteractionNet: Joint Planning and Prediction for Autonomous Driving with Transformers (Jiawei Fu et al., 2023)

{{<citation>}}

Jiawei Fu, Yanqing Shen, Zhiqiang Jian, Shitao Chen, Jingmin Xin, Nanning Zheng. (2023)  
**InteractionNet: Joint Planning and Prediction for Autonomous Driving with Transformers**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.03475v1)  

---


**ABSTRACT**  
Planning and prediction are two important modules of autonomous driving and have experienced tremendous advancement recently. Nevertheless, most existing methods regard planning and prediction as independent and ignore the correlation between them, leading to the lack of consideration for interaction and dynamic changes of traffic scenarios. To address this challenge, we propose InteractionNet, which leverages transformer to share global contextual reasoning among all traffic participants to capture interaction and interconnect planning and prediction to achieve joint. Besides, InteractionNet deploys another transformer to help the model pay extra attention to the perceived region containing critical or unseen vehicles. InteractionNet outperforms other baselines in several benchmarks, especially in terms of safety, which benefits from the joint consideration of planning and forecasting. The code will be available at https://github.com/fujiawei0724/InteractionNet.

{{</citation>}}


## cs.LG (13)



### (49/88) Training Acceleration of Low-Rank Decomposed Networks using Sequential Freezing and Rank Quantization (Habib Hajimolahoseini et al., 2023)

{{<citation>}}

Habib Hajimolahoseini, Walid Ahmed, Yang Liu. (2023)  
**Training Acceleration of Low-Rank Decomposed Networks using Sequential Freezing and Rank Quantization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2309.03824v1)  

---


**ABSTRACT**  
Low Rank Decomposition (LRD) is a model compression technique applied to the weight tensors of deep learning models in order to reduce the number of trainable parameters and computational complexity. However, due to high number of new layers added to the architecture after applying LRD, it may not lead to a high training/inference acceleration if the decomposition ranks are not small enough. The issue is that using small ranks increases the risk of significant accuracy drop after decomposition. In this paper, we propose two techniques for accelerating low rank decomposed models without requiring to use small ranks for decomposition. These methods include rank optimization and sequential freezing of decomposed layers. We perform experiments on both convolutional and transformer-based models. Experiments show that these techniques can improve the model throughput up to 60% during training and 37% during inference when combined together while preserving the accuracy close to that of the original models

{{</citation>}}


### (50/88) Deep Learning Safety Concerns in Automated Driving Perception (Stephanie Abrecht et al., 2023)

{{<citation>}}

Stephanie Abrecht, Alexander Hirsch, Shervin Raafatnia, Matthias Woehrle. (2023)  
**Deep Learning Safety Concerns in Automated Driving Perception**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.03774v1)  

---


**ABSTRACT**  
Recent advances in the field of deep learning and impressive performance of deep neural networks (DNNs) for perception have resulted in an increased demand for their use in automated driving (AD) systems. The safety of such systems is of utmost importance and thus requires to consider the unique properties of DNNs.   In order to achieve safety of AD systems with DNN-based perception components in a systematic and comprehensive approach, so-called safety concerns have been introduced as a suitable structuring element. On the one hand, the concept of safety concerns is -- by design -- well aligned to existing standards relevant for safety of AD systems such as ISO 21448 (SOTIF). On the other hand, it has already inspired several academic publications and upcoming standards on AI safety such as ISO PAS 8800.   While the concept of safety concerns has been previously introduced, this paper extends and refines it, leveraging feedback from various domain and safety experts in the field. In particular, this paper introduces an additional categorization for a better understanding as well as enabling cross-functional teams to jointly address the concerns.

{{</citation>}}


### (51/88) TSGBench: Time Series Generation Benchmark (Yihao Ang et al., 2023)

{{<citation>}}

Yihao Ang, Qiang Huang, Yifan Bao, Anthony K. H. Tung, Zhiyong Huang. (2023)  
**TSGBench: Time Series Generation Benchmark**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DB, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2309.03755v1)  

---


**ABSTRACT**  
Synthetic Time Series Generation (TSG) is crucial in a range of applications, including data augmentation, anomaly detection, and privacy preservation. Although significant strides have been made in this field, existing methods exhibit three key limitations: (1) They often benchmark against similar model types, constraining a holistic view of performance capabilities. (2) The use of specialized synthetic and private datasets introduces biases and hampers generalizability. (3) Ambiguous evaluation measures, often tied to custom networks or downstream tasks, hinder consistent and fair comparison.   To overcome these limitations, we introduce \textsf{TSGBench}, the inaugural TSG Benchmark, designed for a unified and comprehensive assessment of TSG methods. It comprises three modules: (1) a curated collection of publicly available, real-world datasets tailored for TSG, together with a standardized preprocessing pipeline; (2) a comprehensive evaluation measures suite including vanilla measures, new distance-based assessments, and visualization tools; (3) a pioneering generalization test rooted in Domain Adaptation (DA), compatible with all methods. We have conducted extensive experiments across ten real-world datasets from diverse domains, utilizing ten advanced TSG methods and twelve evaluation measures, all gauged through \textsf{TSGBench}. The results highlight its remarkable efficacy and consistency. More importantly, \textsf{TSGBench} delivers a statistical breakdown of method rankings, illuminating performance variations across different datasets and measures, and offering nuanced insights into the effectiveness of each method.

{{</citation>}}


### (52/88) A Causal Perspective on Loan Pricing: Investigating the Impacts of Selection Bias on Identifying Bid-Response Functions (Christopher Bockel-Rickermann et al., 2023)

{{<citation>}}

Christopher Bockel-Rickermann, Sam Verboven, Tim Verdonck, Wouter Verbeke. (2023)  
**A Causal Perspective on Loan Pricing: Investigating the Impacts of Selection Bias on Identifying Bid-Response Functions**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, econ-EM  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.03730v1)  

---


**ABSTRACT**  
In lending, where prices are specific to both customers and products, having a well-functioning personalized pricing policy in place is essential to effective business making. Typically, such a policy must be derived from observational data, which introduces several challenges. While the problem of ``endogeneity'' is prominently studied in the established pricing literature, the problem of selection bias (or, more precisely, bid selection bias) is not. We take a step towards understanding the effects of selection bias by posing pricing as a problem of causal inference. Specifically, we consider the reaction of a customer to price a treatment effect. In our experiments, we simulate varying levels of selection bias on a semi-synthetic dataset on mortgage loan applications in Belgium. We investigate the potential of parametric and nonparametric methods for the identification of individual bid-response functions. Our results illustrate how conventional methods such as logistic regression and neural networks suffer adversely from selection bias. In contrast, we implement state-of-the-art methods from causal machine learning and show their capability to overcome selection bias in pricing data.

{{</citation>}}


### (53/88) DiffDefense: Defending against Adversarial Attacks via Diffusion Models (Hondamunige Prasanna Silva et al., 2023)

{{<citation>}}

Hondamunige Prasanna Silva, Lorenzo Seidenari, Alberto Del Bimbo. (2023)  
**DiffDefense: Defending against Adversarial Attacks via Diffusion Models**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2309.03702v1)  

---


**ABSTRACT**  
This paper presents a novel reconstruction method that leverages Diffusion Models to protect machine learning classifiers against adversarial attacks, all without requiring any modifications to the classifiers themselves. The susceptibility of machine learning models to minor input perturbations renders them vulnerable to adversarial attacks. While diffusion-based methods are typically disregarded for adversarial defense due to their slow reverse process, this paper demonstrates that our proposed method offers robustness against adversarial threats while preserving clean accuracy, speed, and plug-and-play compatibility. Code at: https://github.com/HondamunigePrasannaSilva/DiffDefence.

{{</citation>}}


### (54/88) Short-Term Load Forecasting Using A Particle-Swarm Optimized Multi-Head Attention-Augmented CNN-LSTM Network (Paapa Kwesi Quansah, 2023)

{{<citation>}}

Paapa Kwesi Quansah. (2023)  
**Short-Term Load Forecasting Using A Particle-Swarm Optimized Multi-Head Attention-Augmented CNN-LSTM Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs-SY, cs.LG, eess-SY  
Keywords: Attention, LSTM  
[Paper Link](http://arxiv.org/abs/2309.03694v1)  

---


**ABSTRACT**  
Short-term load forecasting is of paramount importance in the efficient operation and planning of power systems, given its inherent non-linear and dynamic nature. Recent strides in deep learning have shown promise in addressing this challenge. However, these methods often grapple with hyperparameter sensitivity, opaqueness in interpretability, and high computational overhead for real-time deployment. In this paper, I propose a novel solution that surmounts these obstacles. Our approach harnesses the power of the Particle-Swarm Optimization algorithm to autonomously explore and optimize hyperparameters, a Multi-Head Attention mechanism to discern the salient features crucial for accurate forecasting, and a streamlined framework for computational efficiency. Our method undergoes rigorous evaluation using a genuine electricity demand dataset. The results underscore its superiority in terms of accuracy, robustness, and computational efficiency. Notably, our Mean Absolute Percentage Error of 1.9376 marks a significant advancement over existing state-of-the-art approaches, heralding a new era in short-term load forecasting.

{{</citation>}}


### (55/88) Characterizing Lipschitz Stability of GNN for Fairness (Yaning Jia et al., 2023)

{{<citation>}}

Yaning Jia, Chunhui Zhang, Jundong Li, Chuxu Zhang. (2023)  
**Characterizing Lipschitz Stability of GNN for Fairness**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.03648v1)  

---


**ABSTRACT**  
The Lipschitz bound, a technique from robust statistics, can limit the maximum changes in the output concerning the input, taking into account associated irrelevant biased factors. It is an efficient and provable method for examining the output stability of machine learning models without incurring additional computation costs. Recently, Graph Neural Networks (GNNs), which operate on non-Euclidean data, have gained significant attention. However, no previous research has investigated the GNN Lipschitz bounds to shed light on stabilizing model outputs, especially when working on non-Euclidean data with inherent biases. Given the inherent biases in common graph data used for GNN training, it poses a serious challenge to constraining the GNN output perturbations induced by input biases, thereby safeguarding fairness during training. Recently, despite the Lipschitz constant's use in controlling the stability of Euclideanneural networks, the calculation of the precise Lipschitz constant remains elusive for non-Euclidean neural networks like GNNs, especially within fairness contexts. To narrow this gap, we begin with the general GNNs operating on an attributed graph, and formulate a Lipschitz bound to limit the changes in the output regarding biases associated with the input. Additionally, we theoretically analyze how the Lipschitz constant of a GNN model could constrain the output perturbations induced by biases learned from data for fairness training. We experimentally validate the Lipschitz bound's effectiveness in limiting biases of the model output. Finally, from a training dynamics perspective, we demonstrate why the theoretical Lipschitz bound can effectively guide the GNN training to better trade-off between accuracy and fairness.

{{</citation>}}


### (56/88) Insights Into the Inner Workings of Transformer Models for Protein Function Prediction (Markus Wenzel et al., 2023)

{{<citation>}}

Markus Wenzel, Erik Grüner, Nils Strodthoff. (2023)  
**Insights Into the Inner Workings of Transformer Models for Protein Function Prediction**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, q-bio-BM  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2309.03631v1)  

---


**ABSTRACT**  
Motivation: We explored how explainable AI (XAI) can help to shed light into the inner workings of neural networks for protein function prediction, by extending the widely used XAI method of integrated gradients such that latent representations inside of transformer models, which were finetuned to Gene Ontology term and Enzyme Commission number prediction, can be inspected too. Results: The approach enabled us to identify amino acids in the sequences that the transformers pay particular attention to, and to show that these relevant sequence parts reflect expectations from biology and chemistry, both in the embedding layer and inside of the model, where we identified transformer heads with a statistically significant correspondence of attribution maps with ground truth sequence annotations (e.g., transmembrane regions, active sites) across many proteins. Availability and Implementation: Source code can be accessed at https://github.com/markuswenzel/xai-proteins .

{{</citation>}}


### (57/88) Filtration Surfaces for Dynamic Graph Classification (Franz Srambical et al., 2023)

{{<citation>}}

Franz Srambical, Bastian Rieck. (2023)  
**Filtration Surfaces for Dynamic Graph Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2309.03616v1)  

---


**ABSTRACT**  
Existing approaches for classifying dynamic graphs either lift graph kernels to the temporal domain, or use graph neural networks (GNNs). However, current baselines have scalability issues, cannot handle a changing node set, or do not take edge weight information into account. We propose filtration surfaces, a novel method that is scalable and flexible, to alleviate said restrictions. We experimentally validate the efficacy of our model and show that filtration surfaces outperform previous state-of-the-art baselines on datasets that rely on edge weight information. Our method does so while being either completely parameter-free or having at most one parameter, and yielding the lowest overall standard deviation.

{{</citation>}}


### (58/88) Sparse Federated Training of Object Detection in the Internet of Vehicles (Luping Rao et al., 2023)

{{<citation>}}

Luping Rao, Chuan Ma, Ming Ding, Yuwen Qian, Lu Zhou, Zhe Liu. (2023)  
**Sparse Federated Training of Object Detection in the Internet of Vehicles**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.03569v1)  

---


**ABSTRACT**  
As an essential component part of the Intelligent Transportation System (ITS), the Internet of Vehicles (IoV) plays a vital role in alleviating traffic issues. Object detection is one of the key technologies in the IoV, which has been widely used to provide traffic management services by analyzing timely and sensitive vehicle-related information. However, the current object detection methods are mostly based on centralized deep training, that is, the sensitive data obtained by edge devices need to be uploaded to the server, which raises privacy concerns. To mitigate such privacy leakage, we first propose a federated learning-based framework, where well-trained local models are shared in the central server. However, since edge devices usually have limited computing power, plus a strict requirement of low latency in IoVs, we further propose a sparse training process on edge devices, which can effectively lighten the model, and ensure its training efficiency on edge devices, thereby reducing communication overheads. In addition, due to the diverse computing capabilities and dynamic environment, different sparsity rates are applied to edge devices. To further guarantee the performance, we propose, FedWeg, an improved aggregation scheme based on FedAvg, which is designed by the inverse ratio of sparsity rates. Experiments on the real-life dataset using YOLO show that the proposed scheme can achieve the required object detection rate while saving considerable communication costs.

{{</citation>}}


### (59/88) Fast FixMatch: Faster Semi-Supervised Learning with Curriculum Batch Size (John Chen et al., 2023)

{{<citation>}}

John Chen, Chen Dun, Anastasios Kyrillidis. (2023)  
**Fast FixMatch: Faster Semi-Supervised Learning with Curriculum Batch Size**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.03469v1)  

---


**ABSTRACT**  
Advances in Semi-Supervised Learning (SSL) have almost entirely closed the gap between SSL and Supervised Learning at a fraction of the number of labels. However, recent performance improvements have often come \textit{at the cost of significantly increased training computation}. To address this, we propose Curriculum Batch Size (CBS), \textit{an unlabeled batch size curriculum which exploits the natural training dynamics of deep neural networks.} A small unlabeled batch size is used in the beginning of training and is gradually increased to the end of training. A fixed curriculum is used regardless of dataset, model or number of epochs, and reduced training computations is demonstrated on all settings. We apply CBS, strong labeled augmentation, Curriculum Pseudo Labeling (CPL) \citep{FlexMatch} to FixMatch \citep{FixMatch} and term the new SSL algorithm Fast FixMatch. We perform an ablation study to show that strong labeled augmentation and/or CPL do not significantly reduce training computations, but, in synergy with CBS, they achieve optimal performance. Fast FixMatch also achieves substantially higher data utilization compared to previous state-of-the-art. Fast FixMatch achieves between $2.1\times$ - $3.4\times$ reduced training computations on CIFAR-10 with all but 40, 250 and 4000 labels removed, compared to vanilla FixMatch, while attaining the same cited state-of-the-art error rate \citep{FixMatch}. Similar results are achieved for CIFAR-100, SVHN and STL-10. Finally, Fast MixMatch achieves between $2.6\times$ - $3.3\times$ reduced training computations in federated SSL tasks and online/streaming learning SSL tasks, which further demonstrate the generializbility of Fast MixMatch to different scenarios and tasks.

{{</citation>}}


### (60/88) Equal Long-term Benefit Rate: Adapting Static Fairness Notions to Sequential Decision Making (Yuancheng Xu et al., 2023)

{{<citation>}}

Yuancheng Xu, Chenghao Deng, Yanchao Sun, Ruijie Zheng, Xiyao Wang, Jieyu Zhao, Furong Huang. (2023)  
**Equal Long-term Benefit Rate: Adapting Static Fairness Notions to Sequential Decision Making**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.03426v1)  

---


**ABSTRACT**  
Decisions made by machine learning models may have lasting impacts over time, making long-term fairness a crucial consideration. It has been shown that when ignoring the long-term effect, naively imposing fairness criterion in static settings can actually exacerbate bias over time. To explicitly address biases in sequential decision-making, recent works formulate long-term fairness notions in Markov Decision Process (MDP) framework. They define the long-term bias to be the sum of static bias over each time step. However, we demonstrate that naively summing up the step-wise bias can cause a false sense of fairness since it fails to consider the importance difference of different time steps during transition. In this work, we introduce a long-term fairness notion called Equal Long-term Benefit Rate (ELBERT), which explicitly considers varying temporal importance and adapts static fairness principles to the sequential setting. Moreover, we show that the policy gradient of Long-term Benefit Rate can be analytically reduced to standard policy gradient. This makes standard policy optimization methods applicable for reducing the bias, leading to our proposed bias mitigation method ELBERT-PO. Experiments on three sequential decision making environments show that ELBERT-PO significantly reduces bias and maintains high utility. Code is available at https://github.com/Yuancheng-Xu/ELBERT.

{{</citation>}}


### (61/88) Large Language Models as Optimizers (Chengrun Yang et al., 2023)

{{<citation>}}

Chengrun Yang, Xuezhi Wang, Yifeng Lu, Hanxiao Liu, Quoc V. Le, Denny Zhou, Xinyun Chen. (2023)  
**Large Language Models as Optimizers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.03409v1)  

---


**ABSTRACT**  
Optimization is ubiquitous. While derivative-based algorithms have been powerful tools for various problems, the absence of gradient imposes challenges on many real-world applications. In this work, we propose Optimization by PROmpting (OPRO), a simple and effective approach to leverage large language models (LLMs) as optimizers, where the optimization task is described in natural language. In each optimization step, the LLM generates new solutions from the prompt that contains previously generated solutions with their values, then the new solutions are evaluated and added to the prompt for the next optimization step. We first showcase OPRO on linear regression and traveling salesman problems, then move on to prompt optimization where the goal is to find instructions that maximize the task accuracy. With a variety of LLMs, we demonstrate that the best prompts optimized by OPRO outperform human-designed prompts by up to 8% on GSM8K, and by up to 50% on Big-Bench Hard tasks.

{{</citation>}}


## cs.AI (4)



### (62/88) Extending Transductive Knowledge Graph Embedding Models for Inductive Logical Relational Inference (Thomas Gebhart et al., 2023)

{{<citation>}}

Thomas Gebhart, John Cobb. (2023)  
**Extending Transductive Knowledge Graph Embedding Models for Inductive Logical Relational Inference**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IR, cs-SI, cs.AI  
Keywords: Embedding, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.03773v1)  

---


**ABSTRACT**  
Many downstream inference tasks for knowledge graphs, such as relation prediction, have been handled successfully by knowledge graph embedding techniques in the transductive setting. To address the inductive setting wherein new entities are introduced into the knowledge graph at inference time, more recent work opts for models which learn implicit representations of the knowledge graph through a complex function of a network's subgraph structure, often parametrized by graph neural network architectures. These come at the cost of increased parametrization, reduced interpretability and limited generalization to other downstream inference tasks. In this work, we bridge the gap between traditional transductive knowledge graph embedding approaches and more recent inductive relation prediction models by introducing a generalized form of harmonic extension which leverages representations learned through transductive embedding methods to infer representations of new entities introduced at inference time as in the inductive setting. This harmonic extension technique provides the best such approximation, can be implemented via an efficient iterative scheme, and can be employed to answer a family of conjunctive logical queries over the knowledge graph, further expanding the capabilities of transductive embedding methods. In experiments on a number of large-scale knowledge graph embedding benchmarks, we find that this approach for extending the functionality of transductive knowledge graph embedding models to perform knowledge graph completion and answer logical queries in the inductive setting is competitive with--and in some scenarios outperforms--several state-of-the-art models derived explicitly for such inductive tasks.

{{</citation>}}


### (63/88) PyGraft: Configurable Generation of Schemas and Knowledge Graphs at Your Fingertips (Nicolas Hubert et al., 2023)

{{<citation>}}

Nicolas Hubert, Pierre Monnin, Mathieu d'Aquin, Armelle Brun, Davy Monticolo. (2023)  
**PyGraft: Configurable Generation of Schemas and Knowledge Graphs at Your Fingertips**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SE, cs.AI  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.03685v1)  

---


**ABSTRACT**  
Knowledge graphs (KGs) have emerged as a prominent data representation and management paradigm. Being usually underpinned by a schema (e.g. an ontology), KGs capture not only factual information but also contextual knowledge. In some tasks, a few KGs established themselves as standard benchmarks. However, recent works outline that relying on a limited collection of datasets is not sufficient to assess the generalization capability of an approach. In some data-sensitive fields such as education or medicine, access to public datasets is even more limited. To remedy the aforementioned issues, we release PyGraft, a Python-based tool that generates highly customized, domain-agnostic schemas and knowledge graphs. The synthesized schemas encompass various RDFS and OWL constructs, while the synthesized KGs emulate the characteristics and scale of real-world KGs. Logical consistency of the generated resources is ultimately ensured by running a description logic (DL) reasoner. By providing a way of generating both a schema and KG in a single pipeline, PyGraft's aim is to empower the generation of a more diverse array of KGs for benchmarking novel approaches in areas such as graph-based machine learning (ML), or more generally KG processing. In graph-based ML in particular, this should foster a more holistic evaluation of model performance and generalization capability, thereby going beyond the limited collection of available benchmarks. PyGraft is available at: https://github.com/nicolas-hbt/pygraft.

{{</citation>}}


### (64/88) Learning of Generalizable and Interpretable Knowledge in Grid-Based Reinforcement Learning Environments (Manuel Eberhardinger et al., 2023)

{{<citation>}}

Manuel Eberhardinger, Johannes Maucher, Setareh Maghsudi. (2023)  
**Learning of Generalizable and Interpretable Knowledge in Grid-Based Reinforcement Learning Environments**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.03651v1)  

---


**ABSTRACT**  
Understanding the interactions of agents trained with deep reinforcement learning is crucial for deploying agents in games or the real world. In the former, unreasonable actions confuse players. In the latter, that effect is even more significant, as unexpected behavior cause accidents with potentially grave and long-lasting consequences for the involved individuals. In this work, we propose using program synthesis to imitate reinforcement learning policies after seeing a trajectory of the action sequence. Programs have the advantage that they are inherently interpretable and verifiable for correctness. We adapt the state-of-the-art program synthesis system DreamCoder for learning concepts in grid-based environments, specifically, a navigation task and two miniature versions of Atari games, Space Invaders and Asterix. By inspecting the generated libraries, we can make inferences about the concepts the black-box agent has learned and better understand the agent's behavior. We achieve the same by visualizing the agent's decision-making process for the imitated sequences. We evaluate our approach with different types of program synthesizers based on a search-only method, a neural-guided search, and a language model fine-tuned on code.

{{</citation>}}


### (65/88) Beyond XAI:Obstacles Towards Responsible AI (Yulu Pi, 2023)

{{<citation>}}

Yulu Pi. (2023)  
**Beyond XAI:Obstacles Towards Responsible AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.03638v1)  

---


**ABSTRACT**  
The rapidly advancing domain of Explainable Artificial Intelligence (XAI) has sparked significant interests in developing techniques to make AI systems more transparent and understandable. Nevertheless, in real-world contexts, the methods of explainability and their evaluation strategies present numerous limitations.Moreover, the scope of responsible AI extends beyond just explainability. In this paper, we explore these limitations and discuss their implications in a boarder context of responsible AI when considering other important aspects, including privacy, fairness and contestability.

{{</citation>}}


## eess.IV (4)



### (66/88) Label-efficient Contrastive Learning-based model for nuclei detection and classification in 3D Cardiovascular Immunofluorescent Images (Nazanin Moradinasab et al., 2023)

{{<citation>}}

Nazanin Moradinasab, Rebecca A. Deaton, Laura S. Shankman, Gary K. Owens, Donald E. Brown. (2023)  
**Label-efficient Contrastive Learning-based model for nuclei detection and classification in 3D Cardiovascular Immunofluorescent Images**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.03744v1)  

---


**ABSTRACT**  
Recently, deep learning-based methods achieved promising performance in nuclei detection and classification applications. However, training deep learning-based methods requires a large amount of pixel-wise annotated data, which is time-consuming and labor-intensive, especially in 3D images. An alternative approach is to adapt weak-annotation methods, such as labeling each nucleus with a point, but this method does not extend from 2D histopathology images (for which it was originally developed) to 3D immunofluorescent images. The reason is that 3D images contain multiple channels (z-axis) for nuclei and different markers separately, which makes training using point annotations difficult. To address this challenge, we propose the Label-efficient Contrastive learning-based (LECL) model to detect and classify various types of nuclei in 3D immunofluorescent images. Previous methods use Maximum Intensity Projection (MIP) to convert immunofluorescent images with multiple slices to 2D images, which can cause signals from different z-stacks to falsely appear associated with each other. To overcome this, we devised an Extended Maximum Intensity Projection (EMIP) approach that addresses issues using MIP. Furthermore, we performed a Supervised Contrastive Learning (SCL) approach for weakly supervised settings. We conducted experiments on cardiovascular datasets and found that our proposed framework is effective and efficient in detecting and classifying various types of nuclei in 3D immunofluorescent images.

{{</citation>}}


### (67/88) MS-UNet-v2: Adaptive Denoising Method and Training Strategy for Medical Image Segmentation with Small Training Data (Haoyuan Chen et al., 2023)

{{<citation>}}

Haoyuan Chen, Yufei Han, Pin Xu, Yanyi Li, Kuan Li, Jianping Yin. (2023)  
**MS-UNet-v2: Adaptive Denoising Method and Training Strategy for Medical Image Segmentation with Small Training Data**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.03686v1)  

---


**ABSTRACT**  
Models based on U-like structures have improved the performance of medical image segmentation. However, the single-layer decoder structure of U-Net is too "thin" to exploit enough information, resulting in large semantic differences between the encoder and decoder parts. Things get worse if the number of training sets of data is not sufficiently large, which is common in medical image processing tasks where annotated data are more difficult to obtain than other tasks. Based on this observation, we propose a novel U-Net model named MS-UNet for the medical image segmentation task in this study. Instead of the single-layer U-Net decoder structure used in Swin-UNet and TransUnet, we specifically design a multi-scale nested decoder based on the Swin Transformer for U-Net. The proposed multi-scale nested decoder structure allows the feature mapping between the decoder and encoder to be semantically closer, thus enabling the network to learn more detailed features. In addition, we propose a novel edge loss and a plug-and-play fine-tuning Denoising module, which not only effectively improves the segmentation performance of MS-UNet, but could also be applied to other models individually. Experimental results show that MS-UNet could effectively improve the network performance with more efficient feature learning capability and exhibit more advanced performance, especially in the extreme case with a small amount of training data, and the proposed Edge loss and Denoising module could significantly enhance the segmentation performance of MS-UNet.

{{</citation>}}


### (68/88) Anatomy-informed Data Augmentation for Enhanced Prostate Cancer Detection (Balint Kovacs et al., 2023)

{{<citation>}}

Balint Kovacs, Nils Netzer, Michael Baumgartner, Carolin Eith, Dimitrios Bounias, Clara Meinzer, Paul F. Jaeger, Kevin S. Zhang, Ralf Floca, Adrian Schrader, Fabian Isensee, Regula Gnirs, Magdalena Goertz, Viktoria Schuetz, Albrecht Stenzinger, Markus Hohenfellner, Heinz-Peter Schlemmer, Ivo Wolf, David Bonekamp, Klaus H. Maier-Hein. (2023)  
**Anatomy-informed Data Augmentation for Enhanced Prostate Cancer Detection**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, eess-IV, eess.IV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.03652v1)  

---


**ABSTRACT**  
Data augmentation (DA) is a key factor in medical image analysis, such as in prostate cancer (PCa) detection on magnetic resonance images. State-of-the-art computer-aided diagnosis systems still rely on simplistic spatial transformations to preserve the pathological label post transformation. However, such augmentations do not substantially increase the organ as well as tumor shape variability in the training set, limiting the model's ability to generalize to unseen cases with more diverse localized soft-tissue deformations. We propose a new anatomy-informed transformation that leverages information from adjacent organs to simulate typical physiological deformations of the prostate and generates unique lesion shapes without altering their label. Due to its lightweight computational requirements, it can be easily integrated into common DA frameworks. We demonstrate the effectiveness of our augmentation on a dataset of 774 biopsy-confirmed examinations, by evaluating a state-of-the-art method for PCa detection with different augmentation settings.

{{</citation>}}


### (69/88) Spatial encoding of BOLD fMRI time series for categorizing static images across visual datasets: A pilot study on human vision (Vamshi K. Kancharala et al., 2023)

{{<citation>}}

Vamshi K. Kancharala, Debanjali Bhattacharya, Neelam Sinha. (2023)  
**Spatial encoding of BOLD fMRI time series for categorizing static images across visual datasets: A pilot study on human vision**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, eess-IV, eess-SP, eess.IV  
Keywords: Clinical, ImageNet, LSTM  
[Paper Link](http://arxiv.org/abs/2309.03590v1)  

---


**ABSTRACT**  
Functional MRI (fMRI) is widely used to examine brain functionality by detecting alteration in oxygenated blood flow that arises with brain activity. In this study, complexity specific image categorization across different visual datasets is performed using fMRI time series (TS) to understand differences in neuronal activities related to vision. Publicly available BOLD5000 dataset is used for this purpose, containing fMRI scans while viewing 5254 images of diverse categories, drawn from three standard computer vision datasets: COCO, ImageNet and SUN. To understand vision, it is important to study how brain functions while looking at different images. To achieve this, spatial encoding of fMRI BOLD TS has been performed that uses classical Gramian Angular Field (GAF) and Markov Transition Field (MTF) to obtain 2D BOLD TS, representing images of COCO, Imagenet and SUN. For classification, individual GAF and MTF features are fed into regular CNN. Subsequently, parallel CNN model is employed that uses combined 2D features for classifying images across COCO, Imagenet and SUN. The result of 2D CNN models is also compared with 1D LSTM and Bi-LSTM that utilizes raw fMRI BOLD signal for classification. It is seen that parallel CNN model outperforms other network models with an improvement of 7% for multi-class classification. Clinical relevance- The obtained result of this analysis establishes a baseline in studying how differently human brain functions while looking at images of diverse complexities.

{{</citation>}}


## cs.CR (5)



### (70/88) Detecting unknown HTTP-based malicious communication behavior via generated adversarial flows and hierarchical traffic features (Xiaochun Yun et al., 2023)

{{<citation>}}

Xiaochun Yun, Jiang Xie, Shuhao Li, Yongzheng Zhang, Peishuai Sun. (2023)  
**Detecting unknown HTTP-based malicious communication behavior via generated adversarial flows and hierarchical traffic features**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-NI, cs.CR  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.03739v1)  

---


**ABSTRACT**  
Malicious communication behavior is the network communication behavior generated by malware (bot-net, spyware, etc.) after victim devices are infected. Experienced adversaries often hide malicious information in HTTP traffic to evade detection. However, related detection methods have inadequate generalization ability because they are usually based on artificial feature engineering and outmoded datasets. In this paper, we propose an HTTP-based Malicious Communication traffic Detection Model (HMCD-Model) based on generated adversarial flows and hierarchical traffic features. HMCD-Model consists of two parts. The first is a generation algorithm based on WGAN-GP to generate HTTP-based malicious communication traffic for data enhancement. The second is a hybrid neural network based on CNN and LSTM to extract hierarchical spatial-temporal features of traffic. In addition, we collect and publish a dataset, HMCT-2020, which consists of large-scale malicious and benign traffic during three years (2018-2020). Taking the data in HMCT-2020(18) as the training set and the data in other datasets as the test set, the experimental results show that the HMCD-Model can effectively detect unknown HTTP-based malicious communication traffic. It can reach F1 = 98.66% in the dataset HMCT-2020(19-20), F1 = 90.69% in the public dataset CIC-IDS-2017, and F1 = 83.66% in the real traffic, which is 20+% higher than other representative methods on average. This validates that HMCD-Model has the ability to discover unknown HTTP-based malicious communication behavior.

{{</citation>}}


### (71/88) ProvG-Searcher: A Graph Representation Learning Approach for Efficient Provenance Graph Search (Enes Altinisik et al., 2023)

{{<citation>}}

Enes Altinisik, Fatih Deniz, Husrev Taha Sencar. (2023)  
**ProvG-Searcher: A Graph Representation Learning Approach for Efficient Provenance Graph Search**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.03647v1)  

---


**ABSTRACT**  
We present ProvG-Searcher, a novel approach for detecting known APT behaviors within system security logs. Our approach leverages provenance graphs, a comprehensive graph representation of event logs, to capture and depict data provenance relations by mapping system entities as nodes and their interactions as edges. We formulate the task of searching provenance graphs as a subgraph matching problem and employ a graph representation learning method. The central component of our search methodology involves embedding of subgraphs in a vector space where subgraph relationships can be directly evaluated. We achieve this through the use of order embeddings that simplify subgraph matching to straightforward comparisons between a query and precomputed subgraph representations. To address challenges posed by the size and complexity of provenance graphs, we propose a graph partitioning scheme and a behavior-preserving graph reduction method. Overall, our technique offers significant computational efficiency, allowing most of the search computation to be performed offline while incorporating a lightweight comparison step during query execution. Experimental results on standard datasets demonstrate that ProvG-Searcher achieves superior performance, with an accuracy exceeding 99% in detecting query behaviors and a false positive rate of approximately 0.02%, outperforming other approaches.

{{</citation>}}


### (72/88) Zero Trust: Applications, Challenges, and Opportunities (Saeid Ghasemshirazi et al., 2023)

{{<citation>}}

Saeid Ghasemshirazi, Ghazaleh Shirvani, Mohammad Ali Alipour. (2023)  
**Zero Trust: Applications, Challenges, and Opportunities**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.03582v1)  

---


**ABSTRACT**  
The escalating complexity of cybersecurity threats necessitates innovative approaches to safeguard digital assets and sensitive information. The Zero Trust paradigm offers a transformative solution by challenging conventional security models and emphasizing continuous verification and least privilege access. This survey comprehensively explores the theoretical foundations, practical implementations, applications, challenges, and future trends of Zero Trust. Through meticulous analysis, we highlight the relevance of Zero Trust in securing cloud environments, facilitating remote work, and protecting the Internet of Things (IoT) ecosystem. While cultural barriers and technical complexities present challenges, their mitigation unlocks Zero Trust's potential. Integrating Zero Trust with emerging technologies like AI and machine learning augments its efficacy, promising a dynamic and responsive security landscape. Embracing Zero Trust empowers organizations to navigate the ever-evolving cybersecurity realm with resilience and adaptability, redefining trust in the digital age.

{{</citation>}}


### (73/88) Caveat (IoT) Emptor: Towards Transparency of IoT Device Presence (Sashidhar Jakkamsetti et al., 2023)

{{<citation>}}

Sashidhar Jakkamsetti, Youngil Kim, Gene Tsudik. (2023)  
**Caveat (IoT) Emptor: Towards Transparency of IoT Device Presence**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.03574v1)  

---


**ABSTRACT**  
As many types of IoT devices worm their way into numerous settings and many aspects of our daily lives, awareness of their presence and functionality becomes a source of major concern. Hidden IoT devices can snoop (via sensing) on nearby unsuspecting users, and impact the environment where unaware users are present, via actuation. This prompts, respectively, privacy and security/safety issues. The dangers of hidden IoT devices have been recognized and prior research suggested some means of mitigation, mostly based on traffic analysis or using specialized hardware to uncover devices. While such approaches are partially effective, there is currently no comprehensive approach to IoT device transparency. Prompted in part by recent privacy regulations (GDPR and CCPA), this paper motivates and constructs a privacy-agile Root-of-Trust architecture for IoT devices, called PAISA: Privacy-Agile IoT Sensing and Actuation. It guarantees timely and secure announcements about IoT devices' presence and their capabilities. PAISA has two components: one on the IoT device that guarantees periodic announcements of its presence even if all device software is compromised, and the other that runs on the user device, which captures and processes announcements. Notably, PAISA requires no hardware modifications; it uses a popular off-the-shelf Trusted Execution Environment (TEE) -- ARM TrustZone. This work also comprises a fully functional (open-sourced) prototype implementation of PAISA, which includes: an IoT device that makes announcements via IEEE 802.11 WiFi beacons and an Android smartphone-based app that captures and processes announcements. Both security and performance of PAISA design and prototype are discussed.

{{</citation>}}


### (74/88) Security assessment of common open source MQTT brokers and clients (Edoardo Di Paolo et al., 2023)

{{<citation>}}

Edoardo Di Paolo, Enrico Bassetti, Angelo Spognardi. (2023)  
**Security assessment of common open source MQTT brokers and clients**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.03547v1)  

---


**ABSTRACT**  
Security and dependability of devices are paramount for the IoT ecosystem. Message Queuing Telemetry Transport protocol (MQTT) is the de facto standard and the most common alternative for those limited devices that cannot leverage HTTP. However, the MQTT protocol was designed with no security concern since initially designed for private networks of the oil and gas industry. Since MQTT is widely used for real applications, it is under the lens of the security community, also considering the widespread attacks targeting IoT devices. Following this direction research, in this paper we present an empirical security evaluation of several widespread implementations of MQTT system components, namely five broker libraries and three client libraries. While the results of our research do not capture very critical flaws, there are several scenarios where some libraries do not fully adhere to the standard and leave some margins that could be maliciously exploited and potentially cause system inconsistencies.

{{</citation>}}


## cs.DS (1)



### (75/88) Adjacency Sketches in Adversarial Environments (Moni Naor et al., 2023)

{{<citation>}}

Moni Naor, Eugene Pekel. (2023)  
**Adjacency Sketches in Adversarial Environments**  

---
Primary Category: cs.DS  
Categories: 68W20, 68W40, 68Q87, E-1; F-2-0; G-2-2, cs-CR, cs-DS, cs.DS  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2309.03728v1)  

---


**ABSTRACT**  
An adjacency sketching or implicit labeling scheme for a family $\cal F$ of graphs is a method that defines for any $n$ vertex $G \in \cal F$ an assignment of labels to each vertex in $G$, so that the labels of two vertices tell you whether or not they are adjacent. The goal is to come up with labeling schemes that use as few bits as possible to represent the labels. By using randomness when assigning labels, it is sometimes possible to produce adjacency sketches with much smaller label sizes, but this comes at the cost of introducing some probability of error. Both deterministic and randomized labeling schemes have been extensively studied, as they have applications for distributed data structures and deeper connections to universal graphs and communication complexity. The main question of interest is which graph families have schemes using short labels, usually $O(\log n)$ in the deterministic case or constant for randomized sketches.   In this work we consider the resilience of probabilistic adjacency sketches against an adversary making adaptive queries to the labels. This differs from the previously analyzed probabilistic setting which is ``one shot". We show that in the adaptive adversarial case the size of the labels is tightly related to the maximal degree of the graphs in $\cal F$. This results in a stronger characterization compared to what is known in the non-adversarial setting. In more detail, we construct sketches that fail with probability $\varepsilon$ for graphs with maximal degree $d$ using $2d\log (1/\varepsilon)$ bit labels and show that this is roughly the best that can be done for any specific graph of maximal degree $d$, e.g.\ a $d$-ary tree.

{{</citation>}}


## stat.ML (1)



### (76/88) A Probabilistic Semi-Supervised Approach with Triplet Markov Chains (Katherine Morales et al., 2023)

{{<citation>}}

Katherine Morales, Yohan Petetin. (2023)  
**A Probabilistic Semi-Supervised Approach with Triplet Markov Chains**  

---
Primary Category: stat.ML  
Categories: cs-LG, math-PR, stat-ME, stat-ML, stat.ML  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.03707v1)  

---


**ABSTRACT**  
Triplet Markov chains are general generative models for sequential data which take into account three kinds of random variables: (noisy) observations, their associated discrete labels and latent variables which aim at strengthening the distribution of the observations and their associated labels. However, in practice, we do not have at our disposal all the labels associated to the observations to estimate the parameters of such models. In this paper, we propose a general framework based on a variational Bayesian inference to train parameterized triplet Markov chain models in a semi-supervised context. The generality of our approach enables us to derive semi-supervised algorithms for a variety of generative models for sequential Bayesian classification.

{{</citation>}}


## cs.CY (1)



### (77/88) Social Media Influence Operations (Raphael Meier, 2023)

{{<citation>}}

Raphael Meier. (2023)  
**Social Media Influence Operations**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs-SI, cs.CY  
Keywords: Language Model, Social Media  
[Paper Link](http://arxiv.org/abs/2309.03670v1)  

---


**ABSTRACT**  
Social media platforms enable largely unrestricted many-to-many communication. In times of crisis, they offer a space for collective sense-making and gave rise to new social phenomena (e.g. open-source investigations). However, they also serve as a tool for threat actors to conduct cyber-enabled social influence operations (CeSIOs) in order to shape public opinion and interfere in decision-making processes. CeSIOs rely on the employment of sock puppet accounts to engage authentic users in online communication, exert influence, and subvert online discourse. Large Language Models (LLMs) may further enhance the deceptive properties of sock puppet accounts. Recent LLMs are able to generate targeted and persuasive text which is for the most part indistinguishable from human-written content -- ideal features for covert influence. This article reviews recent developments at the intersection of LLMs and influence operations, summarizes LLMs' salience, and explores the potential impact of LLM-instrumented sock puppet accounts for CeSIOs. Finally, mitigation measures for the near future are highlighted.

{{</citation>}}


## cs.IR (4)



### (78/88) VideolandGPT: A User Study on a Conversational Recommender System (Mateo Gutierrez Granada et al., 2023)

{{<citation>}}

Mateo Gutierrez Granada, Dina Zilbershtein, Daan Odijk, Francesco Barile. (2023)  
**VideolandGPT: A User Study on a Conversational Recommender System**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.03645v1)  

---


**ABSTRACT**  
This paper investigates how large language models (LLMs) can enhance recommender systems, with a specific focus on Conversational Recommender Systems that leverage user preferences and personalised candidate selections from existing ranking models. We introduce VideolandGPT, a recommender system for a Video-on-Demand (VOD) platform, Videoland, which uses ChatGPT to select from a predetermined set of contents, considering the additional context indicated by users' interactions with a chat interface. We evaluate ranking metrics, user experience, and fairness of recommendations, comparing a personalised and a non-personalised version of the system, in a between-subject user study. Our results indicate that the personalised version outperforms the non-personalised in terms of accuracy and general user satisfaction, while both versions increase the visibility of items which are not in the top of the recommendation lists. However, both versions present inconsistent behavior in terms of fairness, as the system may generate recommendations which are not available on Videoland.

{{</citation>}}


### (79/88) Evaluating ChatGPT as a Recommender System: A Rigorous Approach (Dario Di Palma et al., 2023)

{{<citation>}}

Dario Di Palma, Giovanni Maria Biancofiore, Vito Walter Anelli, Fedelucio Narducci, Tommaso Di Noia, Eugenio Di Sciascio. (2023)  
**Evaluating ChatGPT as a Recommender System: A Rigorous Approach**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs.IR  
Keywords: AI, ChatGPT, GPT, GPT-3.5, PaLM  
[Paper Link](http://arxiv.org/abs/2309.03613v1)  

---


**ABSTRACT**  
Recent popularity surrounds large AI language models due to their impressive natural language capabilities. They contribute significantly to language-related tasks, including prompt-based learning, making them valuable for various specific tasks. This approach unlocks their full potential, enhancing precision and generalization. Research communities are actively exploring their applications, with ChatGPT receiving recognition. Despite extensive research on large language models, their potential in recommendation scenarios still needs to be explored. This study aims to fill this gap by investigating ChatGPT's capabilities as a zero-shot recommender system. Our goals include evaluating its ability to use user preferences for recommendations, reordering existing recommendation lists, leveraging information from similar users, and handling cold-start situations. We assess ChatGPT's performance through comprehensive experiments using three datasets (MovieLens Small, Last.FM, and Facebook Book). We compare ChatGPT's performance against standard recommendation algorithms and other large language models, such as GPT-3.5 and PaLM-2. To measure recommendation effectiveness, we employ widely-used evaluation metrics like Mean Average Precision (MAP), Recall, Precision, F1, normalized Discounted Cumulative Gain (nDCG), Item Coverage, Expected Popularity Complement (EPC), Average Coverage of Long Tail (ACLT), Average Recommendation Popularity (ARP), and Popularity-based Ranking-based Equal Opportunity (PopREO). Through thoroughly exploring ChatGPT's abilities in recommender systems, our study aims to contribute to the growing body of research on the versatility and potential applications of large language models. Our experiment code is available on the GitHub repository: https://github.com/sisinflab/Recommender-ChatGPT

{{</citation>}}


### (80/88) Learning Compact Compositional Embeddings via Regularized Pruning for Recommendation (Xurong Liang et al., 2023)

{{<citation>}}

Xurong Liang, Tong Chen, Quoc Viet Hung Nguyen, Jianxin Li, Hongzhi Yin. (2023)  
**Learning Compact Compositional Embeddings via Regularized Pruning for Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Embedding, Pruning  
[Paper Link](http://arxiv.org/abs/2309.03518v1)  

---


**ABSTRACT**  
Latent factor models are the dominant backbones of contemporary recommender systems (RSs) given their performance advantages, where a unique vector embedding with a fixed dimensionality (e.g., 128) is required to represent each entity (commonly a user/item). Due to the large number of users and items on e-commerce sites, the embedding table is arguably the least memory-efficient component of RSs. For any lightweight recommender that aims to efficiently scale with the growing size of users/items or to remain applicable in resource-constrained settings, existing solutions either reduce the number of embeddings needed via hashing, or sparsify the full embedding table to switch off selected embedding dimensions. However, as hash collision arises or embeddings become overly sparse, especially when adapting to a tighter memory budget, those lightweight recommenders inevitably have to compromise their accuracy. To this end, we propose a novel compact embedding framework for RSs, namely Compositional Embedding with Regularized Pruning (CERP). Specifically, CERP represents each entity by combining a pair of embeddings from two independent, substantially smaller meta-embedding tables, which are then jointly pruned via a learnable element-wise threshold. In addition, we innovatively design a regularized pruning mechanism in CERP, such that the two sparsified meta-embedding tables are encouraged to encode information that is mutually complementary. Given the compatibility with agnostic latent factor models, we pair CERP with two popular recommendation models for extensive experiments, where results on two real-world datasets under different memory budgets demonstrate its superiority against state-of-the-art baselines. The codebase of CERP is available in https://github.com/xurong-liang/CERP.

{{</citation>}}


### (81/88) Behind Recommender Systems: the Geography of the ACM RecSys Community (Lorenzo Porcaro et al., 2023)

{{<citation>}}

Lorenzo Porcaro, João Vinagre, Pedro Frau, Isabelle Hupont, Emilia Gómez. (2023)  
**Behind Recommender Systems: the Geography of the ACM RecSys Community**  

---
Primary Category: cs.IR  
Categories: cs-CY, cs-IR, cs.IR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.03512v1)  

---


**ABSTRACT**  
The amount and dissemination rate of media content accessible online is nowadays overwhelming. Recommender Systems filter this information into manageable streams or feeds, adapted to our personal needs or preferences. It is of utter importance that algorithms employed to filter information do not distort or cut out important elements from our perspectives of the world. Under this principle, it is essential to involve diverse views and teams from the earliest stages of their design and development. This has been highlighted, for instance, in recent European Union regulations such as the Digital Services Act, via the requirement of risk monitoring, including the risk of discrimination, and the AI Act, through the requirement to involve people with diverse backgrounds in the development of AI systems. We look into the geographic diversity of the recommender systems research community, specifically by analyzing the affiliation countries of the authors who contributed to the ACM Conference on Recommender Systems (RecSys) during the last 15 years. This study has been carried out in the framework of the Diversity in AI - DivinAI project, whose main objective is the long-term monitoring of diversity in AI forums through a set of indexes.

{{</citation>}}


## cs.SE (3)



### (82/88) The Devil is in the Tails: How Long-Tailed Code Distributions Impact Large Language Models (Xin Zhou et al., 2023)

{{<citation>}}

Xin Zhou, Kisub Kim, Bowen Xu, Jiakun Liu, DongGyun Han, David Lo. (2023)  
**The Devil is in the Tails: How Long-Tailed Code Distributions Impact Large Language Models**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.03567v1)  

---


**ABSTRACT**  
Learning-based techniques, especially advanced Large Language Models (LLMs) for code, have gained considerable popularity in various software engineering (SE) tasks. However, most existing works focus on designing better learning-based models and pay less attention to the properties of datasets. Learning-based models, including popular LLMs for code, heavily rely on data, and the data's properties (e.g., data distribution) could significantly affect their behavior. We conducted an exploratory study on the distribution of SE data and found that such data usually follows a skewed distribution (i.e., long-tailed distribution) where a small number of classes have an extensive collection of samples, while a large number of classes have very few samples. We investigate three distinct SE tasks and analyze the impacts of long-tailed distribution on the performance of LLMs for code. Our experimental results reveal that the long-tailed distribution has a substantial impact on the effectiveness of LLMs for code. Specifically, LLMs for code perform between 30.0\% and 254.0\% worse on data samples associated with infrequent labels compared to data samples of frequent labels. Our study provides a better understanding of the effects of long-tailed distributions on popular LLMs for code and insights for the future development of SE automation.

{{</citation>}}


### (83/88) Software Testing of Generative AI Systems: Challenges and Opportunities (Aldeida Aleti, 2023)

{{<citation>}}

Aldeida Aleti. (2023)  
**Software Testing of Generative AI Systems: Challenges and Opportunities**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2309.03554v1)  

---


**ABSTRACT**  
Software Testing is a well-established area in software engineering, encompassing various techniques and methodologies to ensure the quality and reliability of software systems. However, with the advent of generative artificial intelligence (GenAI) systems, new challenges arise in the testing domain. These systems, capable of generating novel and creative outputs, introduce unique complexities that require novel testing approaches. In this paper, I aim to explore the challenges posed by generative AI systems and discuss potential opportunities for future research in the field of testing. I will touch on the specific characteristics of GenAI systems that make traditional testing techniques inadequate or insufficient. By addressing these challenges and pursuing further research, we can enhance our understanding of how to safeguard GenAI and pave the way for improved quality assurance in this rapidly evolving domain.

{{</citation>}}


### (84/88) Interactive, Iterative, Tooled, Rule-Based Migration of Microsoft Access to Web Technologies (Santiago Bragagnolo et al., 2023)

{{<citation>}}

Santiago Bragagnolo, Nicolas Anquetil, Stéphane Ducasse, Abdelhak-Djamel Seriai, Mustapha Derras. (2023)  
**Interactive, Iterative, Tooled, Rule-Based Migration of Microsoft Access to Web Technologies**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Microsoft  
[Paper Link](http://arxiv.org/abs/2309.03511v1)  

---


**ABSTRACT**  
In the context of a collaboration with Berger-Levrault, an IT company producing information systems, we are working on migrating Microsoft Access monolithic applications to the web front-end and microservices back-end. Like in most software migrations, developers must learn the target technology, and they will be in charge of the evolution of the migrated system in the future. To respond to this problem, we propose the developers take over the migration project. To enable the developers to drive the migration to the target systems, we propose an Interactive, Iterative, Tooled, Rule-Based Migration approach. The contributions of this article are (i) an iterative, interactive process to language, library, GUI and architectural migration; (ii) proposal of a set of artefacts required to support such an approach; (iii) three different validations of the approach: (a) library and paradigm usage migration to Java and Pharo, (b) tables and queries migration to Java and Typescript, (c) form migration to Java Springboot and Typescript Angular.

{{</citation>}}


## cs.PL (1)



### (85/88) P4R-Type: a Verified API for P4 Control Plane Programs (Technical Report) (Jens Kanstrup Larsen et al., 2023)

{{<citation>}}

Jens Kanstrup Larsen, Roberto Guanciale, Philipp Haller, Alceste Scalas. (2023)  
**P4R-Type: a Verified API for P4 Control Plane Programs (Technical Report)**  

---
Primary Category: cs.PL  
Categories: D-3-1; D-3-2; C-2-3, cs-PL, cs.PL  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.03566v1)  

---


**ABSTRACT**  
Software-Defined Networking (SDN) significantly simplifies programming, reconfiguring, and optimizing network devices, such as switches and routers. The de facto standard for programmming SDN devices is the P4 language. However, the flexibility and power of P4, and SDN more generally, gives rise to important risks. As a number of incidents at major cloud providers have shown, errors in SDN programs can compromise the availability of networks, leaving them in a non-functional state. The focus of this paper are errors in control-plane programs that interact with P4-enabled network devices via the standardized P4Runtime API. For clients of the P4Runtime API it is easy to make mistakes that lead to catastrophic failures, despite the use of Google's Protocol Buffers as an interface definition language.   This paper proposes P4R-Type, a novel verified P4Runtime API for Scala that performs static checks for P4 control plane operations, ruling out mismatches between P4 tables, allowed actions, and action parameters. As a formal foundation of P4R-Type, we present the $F_{\text{P4R}}$ calculus and its typing system, which ensure that well-typed programs never get stuck by issuing invalid P4Runtime operations. We evaluate the safety and flexibility of P4R-Type with 3 case studies. To the best of our knowledge, this is the first work that formalises P4Runtime control plane applications, and a typing discipline ensuring the correctness of P4Runtime operations.

{{</citation>}}


## cs.DC (1)



### (86/88) DGC: Training Dynamic Graphs with Spatio-Temporal Non-Uniformity using Graph Partitioning by Chunks (Fahao Chen et al., 2023)

{{<citation>}}

Fahao Chen, Peng Li, Celimuge Wu. (2023)  
**DGC: Training Dynamic Graphs with Spatio-Temporal Non-Uniformity using Graph Partitioning by Chunks**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs.DC  
Keywords: AI, GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.03523v1)  

---


**ABSTRACT**  
Dynamic Graph Neural Network (DGNN) has shown a strong capability of learning dynamic graphs by exploiting both spatial and temporal features. Although DGNN has recently received considerable attention by AI community and various DGNN models have been proposed, building a distributed system for efficient DGNN training is still challenging. It has been well recognized that how to partition the dynamic graph and assign workloads to multiple GPUs plays a critical role in training acceleration. Existing works partition a dynamic graph into snapshots or temporal sequences, which only work well when the graph has uniform spatio-temporal structures. However, dynamic graphs in practice are not uniformly structured, with some snapshots being very dense while others are sparse. To address this issue, we propose DGC, a distributed DGNN training system that achieves a 1.25x - 7.52x speedup over the state-of-the-art in our testbed. DGC's success stems from a new graph partitioning method that partitions dynamic graphs into chunks, which are essentially subgraphs with modest training workloads and few inter connections. This partitioning algorithm is based on graph coarsening, which can run very fast on large graphs. In addition, DGC has a highly efficient run-time, powered by the proposed chunk fusion and adaptive stale aggregation techniques. Extensive experimental results on 3 typical DGNN models and 4 popular dynamic graph datasets are presented to show the effectiveness of DGC.

{{</citation>}}


## cs.GT (1)



### (87/88) Keep-Alive Caching for the Hawkes process (Sushirdeep Narayana et al., 2023)

{{<citation>}}

Sushirdeep Narayana, Ian A. Kash. (2023)  
**Keep-Alive Caching for the Hawkes process**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs.GT  
Keywords: Azure, Microsoft  
[Paper Link](http://arxiv.org/abs/2309.03521v1)  

---


**ABSTRACT**  
We study the design of caching policies in applications such as serverless computing where there is not a fixed size cache to be filled, but rather there is a cost associated with the time an item stays in the cache. We present a model for such caching policies which captures the trade-off between this cost and the cost of cache misses. We characterize optimal caching policies in general and apply this characterization by deriving a closed form for Hawkes processes. Since optimal policies for Hawkes processes depend on the history of arrivals, we also develop history-independent policies which achieve near-optimal average performance. We evaluate the performances of the optimal policy and approximate polices using simulations and a data trace of Azure Functions, Microsoft's FaaS (Function as a Service) platform for serverless computing.

{{</citation>}}


## eess.SY (1)



### (88/88) Deep Reinforcement Learning Enabled Joint Deployment and Beamforming in STAR-RIS Assisted Networks (Zhuoyuan Ma et al., 2023)

{{<citation>}}

Zhuoyuan Ma, Qi Zhao, Bai Yan, Jin Zhang. (2023)  
**Deep Reinforcement Learning Enabled Joint Deployment and Beamforming in STAR-RIS Assisted Networks**  

---
Primary Category: eess.SY  
Categories: G-1-6, I-2-8, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.03520v1)  

---


**ABSTRACT**  
In the new generation of wireless communication systems, reconfigurable intelligent surfaces (RIS) and simultaneously transmitting and reflecting reconfigurable intelligent surfaces (STAR-RIS) have become competitive network components to achieve intelligent and reconfigurable network environments. However, existing work has not fully studied the deployment freedom of STAR-RIS, which limits further improvements in network communication performance. Therefore, this paper proposes a solution based on a deep reinforcement learning algorithm to dynamically deploy STAR-RIS and hybrid beamforming to improve the total communication rate of users in mobile wireless networks. The paper constructs a STAR-RIS assisted multi-user multiple-input single-output (MU-MISO) mobile wireless network and jointly optimizes the dynamic deployment strategy of STAR-RIS and the hybrid beamforming strategy to maximize the long-term total communication rate of users. To solve this problem, the paper uses the Proximal Policy Optimization (PPO) algorithm to optimize the deployment of STAR-RIS and the joint beamforming strategy of STAR-RIS and the base station. The trained policy can maximize the downlink transmission rate of the system and meet the real-time decision-making needs of the system. Numerical simulation results show that compared with the traditional scheme without using STAR-RIS and fixed STAR-RIS deployment, the PPO method proposed in this paper can effectively improve the total communication rate of wireless network users in the service area.

{{</citation>}}
