---
draft: false
title: "arXiv @ 2023.07.27"
date: 2023-07-27
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.07.27"
    identifier: arxiv_20230727
    parent: 202307_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (23)](#cscv-23)
- [cs.AI (5)](#csai-5)
- [cs.LG (27)](#cslg-27)
- [cs.CL (17)](#cscl-17)
- [cs.CE (1)](#csce-1)
- [stat.ML (3)](#statml-3)
- [cs.GT (1)](#csgt-1)
- [cs.SI (1)](#cssi-1)
- [cs.DC (1)](#csdc-1)
- [cs.CY (2)](#cscy-2)
- [cs.IR (5)](#csir-5)
- [cs.HC (3)](#cshc-3)
- [cs.CR (1)](#cscr-1)
- [q-fin.PM (1)](#q-finpm-1)
- [cs.NI (1)](#csni-1)
- [eess.SY (1)](#eesssy-1)
- [cs.SE (1)](#csse-1)
- [cs.DL (1)](#csdl-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.DS (1)](#csds-1)
- [eess.AS (1)](#eessas-1)
- [cs.DB (1)](#csdb-1)
- [cs.SD (1)](#cssd-1)
- [cs.MM (1)](#csmm-1)
- [cs.RO (1)](#csro-1)
- [eess.IV (1)](#eessiv-1)

## cs.CV (23)



### (1/103) Pretrained Deep 2.5D Models for Efficient Predictive Modeling from Retinal OCT (Taha Emre et al., 2023)

{{<citation>}}

Taha Emre, Marzieh Oghbaie, Arunava Chakravarty, Antoine Rivail, Sophie Riedl, Julia Mai, Hendrik P. N. Scholl, Sobha Sivaprasad, Daniel Rueckert, Andrew Lotery, Ursula Schmidt-Erfurth, Hrvoje Bogunović. (2023)  
**Pretrained Deep 2.5D Models for Efficient Predictive Modeling from Retinal OCT**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: LSTM, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.13865v1)  

---


**ABSTRACT**  
In the field of medical imaging, 3D deep learning models play a crucial role in building powerful predictive models of disease progression. However, the size of these models presents significant challenges, both in terms of computational resources and data requirements. Moreover, achieving high-quality pretraining of 3D models proves to be even more challenging. To address these issues, hybrid 2.5D approaches provide an effective solution for utilizing 3D volumetric data efficiently using 2D models. Combining 2D and 3D techniques offers a promising avenue for optimizing performance while minimizing memory requirements. In this paper, we explore 2.5D architectures based on a combination of convolutional neural networks (CNNs), long short-term memory (LSTM), and Transformers. In addition, leveraging the benefits of recent non-contrastive pretraining approaches in 2D, we enhanced the performance and data efficiency of 2.5D techniques even further. We demonstrate the effectiveness of architectures and associated pretraining on a task of predicting progression to wet age-related macular degeneration (AMD) within a six-month period on two large longitudinal OCT datasets.

{{</citation>}}


### (2/103) On the unreasonable vulnerability of transformers for image restoration -- and an easy fix (Shashank Agnihotri et al., 2023)

{{<citation>}}

Shashank Agnihotri, Kanchana Vaishnavi Gandikota, Julia Grabinski, Paramanand Chandramouli, Margret Keuper. (2023)  
**On the unreasonable vulnerability of transformers for image restoration -- and an easy fix**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.13856v1)  

---


**ABSTRACT**  
Following their success in visual recognition tasks, Vision Transformers(ViTs) are being increasingly employed for image restoration. As a few recent works claim that ViTs for image classification also have better robustness properties, we investigate whether the improved adversarial robustness of ViTs extends to image restoration. We consider the recently proposed Restormer model, as well as NAFNet and the "Baseline network" which are both simplified versions of a Restormer. We use Projected Gradient Descent (PGD) and CosPGD, a recently proposed adversarial attack tailored to pixel-wise prediction tasks for our robustness evaluation. Our experiments are performed on real-world images from the GoPro dataset for image deblurring. Our analysis indicates that contrary to as advocated by ViTs in image classification works, these models are highly susceptible to adversarial attacks. We attempt to improve their robustness through adversarial training. While this yields a significant increase in robustness for Restormer, results on other networks are less promising. Interestingly, the design choices in NAFNet and Baselines, which were based on iid performance, and not on robust generalization, seem to be at odds with the model robustness. Thus, we investigate this further and find a fix.

{{</citation>}}


### (3/103) A real-time material breakage detection for offshore wind turbines based on improved neural network algorithm (Yantong Liu, 2023)

{{<citation>}}

Yantong Liu. (2023)  
**A real-time material breakage detection for offshore wind turbines based on improved neural network algorithm**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.13765v1)  

---


**ABSTRACT**  
The integrity of offshore wind turbines, pivotal for sustainable energy generation, is often compromised by surface material defects. Despite the availability of various detection techniques, limitations persist regarding cost-effectiveness, efficiency, and applicability. Addressing these shortcomings, this study introduces a novel approach leveraging an advanced version of the YOLOv8 object detection model, supplemented with a Convolutional Block Attention Module (CBAM) for improved feature recognition. The optimized loss function further refines the learning process. Employing a dataset of 5,432 images from the Saemangeum offshore wind farm and a publicly available dataset, our method underwent rigorous testing. The findings reveal a substantial enhancement in defect detection stability, marking a significant stride towards efficient turbine maintenance. This study's contributions illuminate the path for future research, potentially revolutionizing sustainable energy practices.

{{</citation>}}


### (4/103) PlaneRecTR: Unified Query learning for 3D Plane Recovery from a Single View (Jingjia Shi et al., 2023)

{{<citation>}}

Jingjia Shi, Shuaifeng Zhi, Kai Xu. (2023)  
**PlaneRecTR: Unified Query learning for 3D Plane Recovery from a Single View**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.13756v1)  

---


**ABSTRACT**  
3D plane recovery from a single image can usually be divided into several subtasks of plane detection, segmentation, parameter estimation and possibly depth estimation. Previous works tend to solve this task by either extending the RCNN-based segmentation network or the dense pixel embedding-based clustering framework. However, none of them tried to integrate above related subtasks into a unified framework but treat them separately and sequentially, which we suspect is potentially a main source of performance limitation for existing approaches. Motivated by this finding and the success of query-based learning in enriching reasoning among semantic entities, in this paper, we propose PlaneRecTR, a Transformer-based architecture, which for the first time unifies all subtasks related to single-view plane recovery with a single compact model. Extensive quantitative and qualitative experiments demonstrate that our proposed unified learning achieves mutual benefits across subtasks, obtaining a new state-of-the-art performance on public ScanNet and NYUv2-Plane datasets. Codes are available at https://github.com/SJingjia/PlaneRecTR.

{{</citation>}}


### (5/103) TMR-RD: Training-based Model Refinement and Representation Disagreement for Semi-Supervised Object Detection (Seyed Mojtaba Marvasti-Zadeh et al., 2023)

{{<citation>}}

Seyed Mojtaba Marvasti-Zadeh, Nilanjan Ray, Nadir Erbilgin. (2023)  
**TMR-RD: Training-based Model Refinement and Representation Disagreement for Semi-Supervised Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2307.13755v1)  

---


**ABSTRACT**  
Semi-supervised object detection (SSOD) can incorporate limited labeled data and large amounts of unlabeled data to improve the performance and generalization of existing object detectors. Despite many advances, recent SSOD methods are still challenged by noisy/misleading pseudo-labels, classical exponential moving average (EMA) strategy, and the consensus of Teacher-Student models in the latter stages of training. This paper proposes a novel training-based model refinement (TMR) stage and a simple yet effective representation disagreement (RD) strategy to address the limitations of classical EMA and the consensus problem. The TMR stage of Teacher-Student models optimizes the lightweight scaling operation to refine the model's weights and prevent overfitting or forgetting learned patterns from unlabeled data. Meanwhile, the RD strategy helps keep these models diverged to encourage the student model to explore complementary representations. In addition, we use cascade regression to generate more reliable pseudo-labels for supervising the student model. Extensive experiments demonstrate the superior performance of our approach over state-of-the-art SSOD methods. Specifically, the proposed approach outperforms the Unbiased-Teacher method by an average mAP margin of 4.6% and 5.3% when using partially-labeled and fully-labeled data on the MS-COCO dataset, respectively.

{{</citation>}}


### (6/103) QuickQual: Lightweight, convenient retinal image quality scoring with off-the-shelf pretrained models (Justin Engelmann et al., 2023)

{{<citation>}}

Justin Engelmann, Amos Storkey, Miguel O. Bernabeu. (2023)  
**QuickQual: Lightweight, convenient retinal image quality scoring with off-the-shelf pretrained models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV, q-bio-QM  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2307.13646v1)  

---


**ABSTRACT**  
Image quality remains a key problem for both traditional and deep learning (DL)-based approaches to retinal image analysis, but identifying poor quality images can be time consuming and subjective. Thus, automated methods for retinal image quality scoring (RIQS) are needed. The current state-of-the-art is MCFNet, composed of three Densenet121 backbones each operating in a different colour space. MCFNet, and the EyeQ dataset released by the same authors, was a huge step forward for RIQS. We present QuickQual, a simple approach to RIQS, consisting of a single off-the-shelf ImageNet-pretrained Densenet121 backbone plus a Support Vector Machine (SVM). QuickQual performs very well, setting a new state-of-the-art for EyeQ (Accuracy: 88.50% vs 88.00% for MCFNet; AUC: 0.9687 vs 0.9588). This suggests that RIQS can be solved with generic perceptual features learned on natural images, as opposed to requiring DL models trained on large amounts of fundus images. Additionally, we propose a Fixed Prior linearisation scheme, that converts EyeQ from a 3-way classification to a continuous logistic regression task. For this task, we present a second model, QuickQual MEga Minified Estimator (QuickQual-MEME), that consists of only 10 parameters on top of an off-the-shelf Densenet121 and can distinguish between gradable and ungradable images with an accuracy of 89.18% (AUC: 0.9537). Code and model are available on GitHub: https://github.com/justinengelmann/QuickQual . QuickQual is so lightweight, that the entire inference code (and even the parameters for QuickQual-MEME) is already contained in this paper.

{{</citation>}}


### (7/103) Learning Transferable Object-Centric Diffeomorphic Transformations for Data Augmentation in Medical Image Segmentation (Nilesh Kumar et al., 2023)

{{<citation>}}

Nilesh Kumar, Prashnna K. Gyawali, Sandesh Ghimire, Linwei Wang. (2023)  
**Learning Transferable Object-Centric Diffeomorphic Transformations for Data Augmentation in Medical Image Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.13645v1)  

---


**ABSTRACT**  
Obtaining labelled data in medical image segmentation is challenging due to the need for pixel-level annotations by experts. Recent works have shown that augmenting the object of interest with deformable transformations can help mitigate this challenge. However, these transformations have been learned globally for the image, limiting their transferability across datasets or applicability in problems where image alignment is difficult. While object-centric augmentations provide a great opportunity to overcome these issues, existing works are only focused on position and random transformations without considering shape variations of the objects. To this end, we propose a novel object-centric data augmentation model that is able to learn the shape variations for the objects of interest and augment the object in place without modifying the rest of the image. We demonstrated its effectiveness in improving kidney tumour segmentation when leveraging shape variations learned both from within the same dataset and transferred from external datasets.

{{</citation>}}


### (8/103) RecursiveDet: End-to-End Region-based Recursive Object Detection (Jing Zhao et al., 2023)

{{<citation>}}

Jing Zhao, Li Sun, Qingli Li. (2023)  
**RecursiveDet: End-to-End Region-based Recursive Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.13619v1)  

---


**ABSTRACT**  
End-to-end region-based object detectors like Sparse R-CNN usually have multiple cascade bounding box decoding stages, which refine the current predictions according to their previous results. Model parameters within each stage are independent, evolving a huge cost. In this paper, we find the general setting of decoding stages is actually redundant. By simply sharing parameters and making a recursive decoder, the detector already obtains a significant improvement. The recursive decoder can be further enhanced by positional encoding (PE) of the proposal box, which makes it aware of the exact locations and sizes of input bounding boxes, thus becoming adaptive to proposals from different stages during the recursion. Moreover, we also design centerness-based PE to distinguish the RoI feature element and dynamic convolution kernels at different positions within the bounding box. To validate the effectiveness of the proposed method, we conduct intensive ablations and build the full model on three recent mainstream region-based detectors. The RecusiveDet is able to achieve obvious performance boosts with even fewer model parameters and slightly increased computation cost. Codes are available at https://github.com/bravezzzzzz/RecursiveDet.

{{</citation>}}


### (9/103) Group Activity Recognition in Computer Vision: A Comprehensive Review, Challenges, and Future Perspectives (Chuanchuan Wang et al., 2023)

{{<citation>}}

Chuanchuan Wang, Ahmad Sufril Azlan Mohamed. (2023)  
**Group Activity Recognition in Computer Vision: A Comprehensive Review, Challenges, and Future Perspectives**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2307.13541v1)  

---


**ABSTRACT**  
Group activity recognition is a hot topic in computer vision. Recognizing activities through group relationships plays a vital role in group activity recognition. It holds practical implications in various scenarios, such as video analysis, surveillance, automatic driving, and understanding social activities. The model's key capabilities encompass efficiently modeling hierarchical relationships within a scene and accurately extracting distinctive spatiotemporal features from groups. Given this technology's extensive applicability, identifying group activities has garnered significant research attention. This work examines the current progress in technology for recognizing group activities, with a specific focus on global interactivity and activities. Firstly, we comprehensively review the pertinent literature and various group activity recognition approaches, from traditional methodologies to the latest methods based on spatial structure, descriptors, non-deep learning, hierarchical recurrent neural networks (HRNN), relationship models, and attention mechanisms. Subsequently, we present the relational network and relational architectures for each module. Thirdly, we investigate methods for recognizing group activity and compare their performance with state-of-the-art technologies. We summarize the existing challenges and provide comprehensive guidance for newcomers to understand group activity recognition. Furthermore, we review emerging perspectives in group activity recognition to explore new directions and possibilities.

{{</citation>}}


### (10/103) HeightFormer: Explicit Height Modeling without Extra Data for Camera-only 3D Object Detection in Bird's Eye View (Yiming Wu et al., 2023)

{{<citation>}}

Yiming Wu, Ruixiang Li, Zequn Qin, Xinhai Zhao, Xi Li. (2023)  
**HeightFormer: Explicit Height Modeling without Extra Data for Camera-only 3D Object Detection in Bird's Eye View**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2307.13510v1)  

---


**ABSTRACT**  
Vision-based Bird's Eye View (BEV) representation is an emerging perception formulation for autonomous driving. The core challenge is to construct BEV space with multi-camera features, which is a one-to-many ill-posed problem. Diving into all previous BEV representation generation methods, we found that most of them fall into two types: modeling depths in image views or modeling heights in the BEV space, mostly in an implicit way. In this work, we propose to explicitly model heights in the BEV space, which needs no extra data like LiDAR and can fit arbitrary camera rigs and types compared to modeling depths. Theoretically, we give proof of the equivalence between height-based methods and depth-based methods. Considering the equivalence and some advantages of modeling heights, we propose HeightFormer, which models heights and uncertainties in a self-recursive way. Without any extra data, the proposed HeightFormer could estimate heights in BEV accurately. Benchmark results show that the performance of HeightFormer achieves SOTA compared with those camera-only methods.

{{</citation>}}


### (11/103) NormAUG: Normalization-guided Augmentation for Domain Generalization (Lei Qi et al., 2023)

{{<citation>}}

Lei Qi, Hongpeng Yang, Yinghuan Shi, Xin Geng. (2023)  
**NormAUG: Normalization-guided Augmentation for Domain Generalization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2307.13492v1)  

---


**ABSTRACT**  
Deep learning has made significant advancements in supervised learning. However, models trained in this setting often face challenges due to domain shift between training and test sets, resulting in a significant drop in performance during testing. To address this issue, several domain generalization methods have been developed to learn robust and domain-invariant features from multiple training domains that can generalize well to unseen test domains. Data augmentation plays a crucial role in achieving this goal by enhancing the diversity of the training data. In this paper, inspired by the observation that normalizing an image with different statistics generated by different batches with various domains can perturb its feature, we propose a simple yet effective method called NormAUG (Normalization-guided Augmentation). Our method includes two paths: the main path and the auxiliary (augmented) path. During training, the auxiliary path includes multiple sub-paths, each corresponding to batch normalization for a single domain or a random combination of multiple domains. This introduces diverse information at the feature level and improves the generalization of the main path. Moreover, our NormAUG method effectively reduces the existing upper boundary for generalization based on theoretical perspectives. During the test stage, we leverage an ensemble strategy to combine the predictions from the auxiliary path of our model, further boosting performance. Extensive experiments are conducted on multiple benchmark datasets to validate the effectiveness of our proposed method.

{{</citation>}}


### (12/103) Cos R-CNN for Online Few-shot Object Detection (Gratianus Wesley Putra Data et al., 2023)

{{<citation>}}

Gratianus Wesley Putra Data, Henry Howard-Jenkins, David Murray, Victor Prisacariu. (2023)  
**Cos R-CNN for Online Few-shot Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Object Detection  
[Paper Link](http://arxiv.org/abs/2307.13485v1)  

---


**ABSTRACT**  
We propose Cos R-CNN, a simple exemplar-based R-CNN formulation that is designed for online few-shot object detection. That is, it is able to localise and classify novel object categories in images with few examples without fine-tuning. Cos R-CNN frames detection as a learning-to-compare task: unseen classes are represented as exemplar images, and objects are detected based on their similarity to these exemplars. The cosine-based classification head allows for dynamic adaptation of classification parameters to the exemplar embedding, and encourages the clustering of similar classes in embedding space without the need for manual tuning of distance-metric hyperparameters. This simple formulation achieves best results on the recently proposed 5-way ImageNet few-shot detection benchmark, beating the online 1/5/10-shot scenarios by more than 8/3/1%, as well as performing up to 20% better in online 20-way few-shot VOC across all shots on novel classes.

{{</citation>}}


### (13/103) An Explainable Model-Agnostic Algorithm for CNN-based Biometrics Verification (Fernando Alonso-Fernandez et al., 2023)

{{<citation>}}

Fernando Alonso-Fernandez, Kevin Hernandez-Diaz, Jose M. Buades, Prayag Tiwari, Josef Bigun. (2023)  
**An Explainable Model-Agnostic Algorithm for CNN-based Biometrics Verification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13428v1)  

---


**ABSTRACT**  
This paper describes an adaptation of the Local Interpretable Model-Agnostic Explanations (LIME) AI method to operate under a biometric verification setting. LIME was initially proposed for networks with the same output classes used for training, and it employs the softmax probability to determine which regions of the image contribute the most to classification. However, in a verification setting, the classes to be recognized have not been seen during training. In addition, instead of using the softmax output, face descriptors are usually obtained from a layer before the classification layer. The model is adapted to achieve explainability via cosine similarity between feature vectors of perturbated versions of the input image. The method is showcased for face biometrics with two CNN models based on MobileNetv2 and ResNet50.

{{</citation>}}


### (14/103) 3DRP-Net: 3D Relative Position-aware Network for 3D Visual Grounding (Zehan Wang et al., 2023)

{{<citation>}}

Zehan Wang, Haifeng Huang, Yang Zhao, Linjun Li, Xize Cheng, Yichen Zhu, Aoxiong Yin, Zhou Zhao. (2023)  
**3DRP-Net: 3D Relative Position-aware Network for 3D Visual Grounding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.13363v1)  

---


**ABSTRACT**  
3D visual grounding aims to localize the target object in a 3D point cloud by a free-form language description. Typically, the sentences describing the target object tend to provide information about its relative relation between other objects and its position within the whole scene. In this work, we propose a relation-aware one-stage framework, named 3D Relative Position-aware Network (3DRP-Net), which can effectively capture the relative spatial relationships between objects and enhance object attributes. Specifically, 1) we propose a 3D Relative Position Multi-head Attention (3DRP-MA) module to analyze relative relations from different directions in the context of object pairs, which helps the model to focus on the specific object relations mentioned in the sentence. 2) We designed a soft-labeling strategy to alleviate the spatial ambiguity caused by redundant points, which further stabilizes and enhances the learning process through a constant and discriminative distribution. Extensive experiments conducted on three benchmarks (i.e., ScanRefer and Nr3D/Sr3D) demonstrate that our method outperforms all the state-of-the-art methods in general. The source code will be released on GitHub.

{{</citation>}}


### (15/103) Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks (Cheeun Hong et al., 2023)

{{<citation>}}

Cheeun Hong, Kyoung Mu Lee. (2023)  
**Overcoming Distribution Mismatch in Quantizing Image Super-Resolution Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2307.13337v1)  

---


**ABSTRACT**  
Quantization is a promising approach to reduce the high computational complexity of image super-resolution (SR) networks. However, compared to high-level tasks like image classification, low-bit quantization leads to severe accuracy loss in SR networks. This is because feature distributions of SR networks are significantly divergent for each channel or input image, and is thus difficult to determine a quantization range. Existing SR quantization works approach this distribution mismatch problem by dynamically adapting quantization ranges to the variant distributions during test time. However, such dynamic adaptation incurs additional computational costs that limit the benefits of quantization. Instead, we propose a new quantization-aware training framework that effectively Overcomes the Distribution Mismatch problem in SR networks without the need for dynamic adaptation. Intuitively, the mismatch can be reduced by directly regularizing the variance in features during training. However, we observe that variance regularization can collide with the reconstruction loss during training and adversely impact SR accuracy. Thus, we avoid the conflict between two losses by regularizing the variance only when the gradients of variance regularization are cooperative with that of reconstruction. Additionally, to further reduce the distribution mismatch, we introduce distribution offsets to layers with a significant mismatch, which either scales or shifts channel-wise features. Our proposed algorithm, called ODM, effectively reduces the mismatch in distributions with minimal computational overhead. Experimental results show that ODM effectively outperforms existing SR quantization approaches with similar or fewer computations, demonstrating the importance of reducing the distribution mismatch problem. Our code is available at https://github.com/Cheeun/ODM.

{{</citation>}}


### (16/103) Mitigating Cross-client GANs-based Attack in Federated Learning (Hong Huang et al., 2023)

{{<citation>}}

Hong Huang, Xinyu Lei, Tao Xiang. (2023)  
**Mitigating Cross-client GANs-based Attack in Federated Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2307.13314v1)  

---


**ABSTRACT**  
Machine learning makes multimedia data (e.g., images) more attractive, however, multimedia data is usually distributed and privacy sensitive. Multiple distributed multimedia clients can resort to federated learning (FL) to jointly learn a global shared model without requiring to share their private samples with any third-party entities. In this paper, we show that FL suffers from the cross-client generative adversarial networks (GANs)-based (C-GANs) attack, in which a malicious client (i.e., adversary) can reconstruct samples with the same distribution as the training samples from other clients (i.e., victims). Since a benign client's data can be leaked to the adversary, this attack brings the risk of local data leakage for clients in many security-critical FL applications. Thus, we propose Fed-EDKD (i.e., Federated Ensemble Data-free Knowledge Distillation) technique to improve the current popular FL schemes to resist C-GANs attack. In Fed-EDKD, each client submits a local model to the server for obtaining an ensemble global model. Then, to avoid model expansion, Fed-EDKD adopts data-free knowledge distillation techniques to transfer knowledge from the ensemble global model to a compressed model. By this way, Fed-EDKD reduces the adversary's control capability over the global model, so Fed-EDKD can effectively mitigate C-GANs attack. Finally, the experimental results demonstrate that Fed-EDKD significantly mitigates C-GANs attack while only incurring a slight accuracy degradation of FL.

{{</citation>}}


### (17/103) CT-Net: Arbitrary-Shaped Text Detection via Contour Transformer (Zhiwen Shao et al., 2023)

{{<citation>}}

Zhiwen Shao, Yuchen Su, Yong Zhou, Fanrong Meng, Hancheng Zhu, Bing Liu, Rui Yao. (2023)  
**CT-Net: Arbitrary-Shaped Text Detection via Contour Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.13310v1)  

---


**ABSTRACT**  
Contour based scene text detection methods have rapidly developed recently, but still suffer from inaccurate frontend contour initialization, multi-stage error accumulation, or deficient local information aggregation. To tackle these limitations, we propose a novel arbitrary-shaped scene text detection framework named CT-Net by progressive contour regression with contour transformers. Specifically, we first employ a contour initialization module that generates coarse text contours without any post-processing. Then, we adopt contour refinement modules to adaptively refine text contours in an iterative manner, which are beneficial for context information capturing and progressive global contour deformation. Besides, we propose an adaptive training strategy to enable the contour transformers to learn more potential deformation paths, and introduce a re-score mechanism that can effectively suppress false positives. Extensive experiments are conducted on four challenging datasets, which demonstrate the accuracy and efficiency of our CT-Net over state-of-the-art methods. Particularly, CT-Net achieves F-measure of 86.1 at 11.2 frames per second (FPS) and F-measure of 87.8 at 10.1 FPS for CTW1500 and Total-Text datasets, respectively.

{{</citation>}}


### (18/103) Conditional Cross Attention Network for Multi-Space Embedding without Entanglement in Only a SINGLE Network (Chull Hwan Song et al., 2023)

{{<citation>}}

Chull Hwan Song, Taebaek Hwang, Jooyoung Yoon, Shunghyun Choi, Yeong Hyeon Gu. (2023)  
**Conditional Cross Attention Network for Multi-Space Embedding without Entanglement in Only a SINGLE Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Attention, Embedding  
[Paper Link](http://arxiv.org/abs/2307.13254v1)  

---


**ABSTRACT**  
Many studies in vision tasks have aimed to create effective embedding spaces for single-label object prediction within an image. However, in reality, most objects possess multiple specific attributes, such as shape, color, and length, with each attribute composed of various classes. To apply models in real-world scenarios, it is essential to be able to distinguish between the granular components of an object. Conventional approaches to embedding multiple specific attributes into a single network often result in entanglement, where fine-grained features of each attribute cannot be identified separately. To address this problem, we propose a Conditional Cross-Attention Network that induces disentangled multi-space embeddings for various specific attributes with only a single backbone. Firstly, we employ a cross-attention mechanism to fuse and switch the information of conditions (specific attributes), and we demonstrate its effectiveness through a diverse visualization example. Secondly, we leverage the vision transformer for the first time to a fine-grained image retrieval task and present a simple yet effective framework compared to existing methods. Unlike previous studies where performance varied depending on the benchmark dataset, our proposed method achieved consistent state-of-the-art performance on the FashionAI, DARN, DeepFashion, and Zappos50K benchmark datasets.

{{</citation>}}


### (19/103) GaPro: Box-Supervised 3D Point Cloud Instance Segmentation Using Gaussian Processes as Pseudo Labelers (Tuan Duc Ngo et al., 2023)

{{<citation>}}

Tuan Duc Ngo, Binh-Son Hua, Khoi Nguyen. (2023)  
**GaPro: Box-Supervised 3D Point Cloud Instance Segmentation Using Gaussian Processes as Pseudo Labelers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13251v1)  

---


**ABSTRACT**  
Instance segmentation on 3D point clouds (3DIS) is a longstanding challenge in computer vision, where state-of-the-art methods are mainly based on full supervision. As annotating ground truth dense instance masks is tedious and expensive, solving 3DIS with weak supervision has become more practical. In this paper, we propose GaPro, a new instance segmentation for 3D point clouds using axis-aligned 3D bounding box supervision. Our two-step approach involves generating pseudo labels from box annotations and training a 3DIS network with the resulting labels. Additionally, we employ the self-training strategy to improve the performance of our method further. We devise an effective Gaussian Process to generate pseudo instance masks from the bounding boxes and resolve ambiguities when they overlap, resulting in pseudo instance masks with their uncertainty values. Our experiments show that GaPro outperforms previous weakly supervised 3D instance segmentation methods and has competitive performance compared to state-of-the-art fully supervised ones. Furthermore, we demonstrate the robustness of our approach, where we can adapt various state-of-the-art fully supervised methods to the weak supervision task by using our pseudo labels for training. The source code and trained models are available at https://github.com/VinAIResearch/GaPro.

{{</citation>}}


### (20/103) Keyword-Aware Relative Spatio-Temporal Graph Networks for Video Question Answering (Yi Cheng et al., 2023)

{{<citation>}}

Yi Cheng, Hehe Fan, Dongyun Lin, Ying Sun, Mohan Kankanhalli, Joo-Hwee Lim. (2023)  
**Keyword-Aware Relative Spatio-Temporal Graph Networks for Video Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.13250v1)  

---


**ABSTRACT**  
The main challenge in video question answering (VideoQA) is to capture and understand the complex spatial and temporal relations between objects based on given questions. Existing graph-based methods for VideoQA usually ignore keywords in questions and employ a simple graph to aggregate features without considering relative relations between objects, which may lead to inferior performance. In this paper, we propose a Keyword-aware Relative Spatio-Temporal (KRST) graph network for VideoQA. First, to make question features aware of keywords, we employ an attention mechanism to assign high weights to keywords during question encoding. The keyword-aware question features are then used to guide video graph construction. Second, because relations are relative, we integrate the relative relation modeling to better capture the spatio-temporal dynamics among object nodes. Moreover, we disentangle the spatio-temporal reasoning into an object-level spatial graph and a frame-level temporal graph, which reduces the impact of spatial and temporal relation reasoning on each other. Extensive experiments on the TGIF-QA, MSVD-QA and MSRVTT-QA datasets demonstrate the superiority of our KRST over multiple state-of-the-art methods.

{{</citation>}}


### (21/103) Multi-Granularity Prediction with Learnable Fusion for Scene Text Recognition (Cheng Da et al., 2023)

{{<citation>}}

Cheng Da, Peng Wang, Cong Yao. (2023)  
**Multi-Granularity Prediction with Learnable Fusion for Scene Text Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NLP, OCR, Transformer  
[Paper Link](http://arxiv.org/abs/2307.13244v1)  

---


**ABSTRACT**  
Due to the enormous technical challenges and wide range of applications, scene text recognition (STR) has been an active research topic in computer vision for years. To tackle this tough problem, numerous innovative methods have been successively proposed, and incorporating linguistic knowledge into STR models has recently become a prominent trend. In this work, we first draw inspiration from the recent progress in Vision Transformer (ViT) to construct a conceptually simple yet functionally powerful vision STR model, which is built upon ViT and a tailored Adaptive Addressing and Aggregation (A$^3$) module. It already outperforms most previous state-of-the-art models for scene text recognition, including both pure vision models and language-augmented methods. To integrate linguistic knowledge, we further propose a Multi-Granularity Prediction strategy to inject information from the language modality into the model in an implicit way, \ie, subword representations (BPE and WordPiece) widely used in NLP are introduced into the output space, in addition to the conventional character level representation, while no independent language model (LM) is adopted. To produce the final recognition results, two strategies for effectively fusing the multi-granularity predictions are devised. The resultant algorithm (termed MGP-STR) is able to push the performance envelope of STR to an even higher level. Specifically, MGP-STR achieves an average recognition accuracy of $94\%$ on standard benchmarks for scene text recognition. Moreover, it also achieves state-of-the-art results on widely-used handwritten benchmarks as well as more challenging scene text datasets, demonstrating the generality of the proposed MGP-STR algorithm. The source code and models will be available at: \url{https://github.com/AlibabaResearch/AdvancedLiterateMachinery/tree/main/OCR/MGP-STR}.

{{</citation>}}


### (22/103) Fashion Matrix: Editing Photos by Just Talking (Zheng Chong et al., 2023)

{{<citation>}}

Zheng Chong, Xujie Zhang, Fuwei Zhao, Zhenyu Xie, Xiaodan Liang. (2023)  
**Fashion Matrix: Editing Photos by Just Talking**  

---
Primary Category: cs.CV  
Categories: 68T42 (Primary) 168T45 (Secondary), I-4-9, cs-CV, cs.CV  
Keywords: AI, Language Model, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2307.13240v1)  

---


**ABSTRACT**  
The utilization of Large Language Models (LLMs) for the construction of AI systems has garnered significant attention across diverse fields. The extension of LLMs to the domain of fashion holds substantial commercial potential but also inherent challenges due to the intricate semantic interactions in fashion-related generation. To address this issue, we developed a hierarchical AI system called Fashion Matrix dedicated to editing photos by just talking. This system facilitates diverse prompt-driven tasks, encompassing garment or accessory replacement, recoloring, addition, and removal. Specifically, Fashion Matrix employs LLM as its foundational support and engages in iterative interactions with users. It employs a range of Semantic Segmentation Models (e.g., Grounded-SAM, MattingAnything, etc.) to delineate the specific editing masks based on user instructions. Subsequently, Visual Foundation Models (e.g., Stable Diffusion, ControlNet, etc.) are leveraged to generate edited images from text prompts and masks, thereby facilitating the automation of fashion editing processes. Experiments demonstrate the outstanding ability of Fashion Matrix to explores the collaborative potential of functionally diverse pre-trained models in the domain of fashion editing.

{{</citation>}}


### (23/103) Multilevel Large Language Models for Everyone (Yuanhao Gong, 2023)

{{<citation>}}

Yuanhao Gong. (2023)  
**Multilevel Large Language Models for Everyone**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CE, cs-CV, cs-DC, cs.CV, econ-GN, q-fin-EC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.13221v1)  

---


**ABSTRACT**  
Large language models have made significant progress in the past few years. However, they are either generic {\it or} field specific, splitting the community into different groups. In this paper, we unify these large language models into a larger map, where the generic {\it and} specific models are linked together and can improve each other, based on the user personal input and information from the internet. The idea of linking several large language models together is inspired by the functionality of human brain. The specific regions on the brain cortex are specific for certain low level functionality. And these regions can jointly work together to achieve more complex high level functionality. Such behavior on human brain cortex sheds the light to design the multilevel large language models that contain global level, field level and user level models. The user level models run on local machines to achieve efficient response and protect the user's privacy. Such multilevel models reduce some redundancy and perform better than the single level models. The proposed multilevel idea can be applied in various applications, such as natural language processing, computer vision tasks, professional assistant, business and healthcare.

{{</citation>}}


## cs.AI (5)



### (24/103) WebArena: A Realistic Web Environment for Building Autonomous Agents (Shuyan Zhou et al., 2023)

{{<citation>}}

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Yonatan Bisk, Daniel Fried, Uri Alon, Graham Neubig. (2023)  
**WebArena: A Realistic Web Environment for Building Autonomous Agents**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: AI, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2307.13854v1)  

---


**ABSTRACT**  
With generative AI advances, the exciting potential for autonomous agents to manage daily tasks via natural language commands has emerged. However, cur rent agents are primarily created and tested in simplified synthetic environments, substantially limiting real-world scenario representation. In this paper, we build an environment for agent command and control that is highly realistic and reproducible. Specifically, we focus on agents that perform tasks on websites, and we create an environment with fully functional websites from four common domains: e-commerce, social forum discussions, collaborative software development, and content management. Our environment is enriched with tools (e.g., a map) and external knowledge bases (e.g., user manuals) to encourage human-like task-solving. Building upon our environment, we release a set of benchmark tasks focusing on evaluating the functional correctness of task completions. The tasks in our benchmark are diverse, long-horizon, and are designed to emulate tasks that humans routinely perform on the internet. We design and implement several autonomous agents, integrating recent techniques such as reasoning before acting. The results demonstrate that solving complex tasks is challenging: our best GPT-4-based agent only achieves an end-to-end task success rate of 10.59%. These results highlight the need for further development of robust agents, that current state-of-the-art LMs are far from perfect performance in these real-life tasks, and that WebArena can be used to measure such progress. Our code, data, environment reproduction resources, and video demonstrations are publicly available at https://webarena.dev/.

{{</citation>}}


### (25/103) ForestMonkey: Toolkit for Reasoning with AI-based Defect Detection and Classification Models (Jiajun Zhang et al., 2023)

{{<citation>}}

Jiajun Zhang, Georgina Cosma, Sarah Bugby, Jason Watkins. (2023)  
**ForestMonkey: Toolkit for Reasoning with AI-based Defect Detection and Classification Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.13815v1)  

---


**ABSTRACT**  
Artificial intelligence (AI) reasoning and explainable AI (XAI) tasks have gained popularity recently, enabling users to explain the predictions or decision processes of AI models. This paper introduces Forest Monkey (FM), a toolkit designed to reason the outputs of any AI-based defect detection and/or classification model with data explainability. Implemented as a Python package, FM takes input in the form of dataset folder paths (including original images, ground truth labels, and predicted labels) and provides a set of charts and a text file to illustrate the reasoning results and suggest possible improvements. The FM toolkit consists of processes such as feature extraction from predictions to reasoning targets, feature extraction from images to defect characteristics, and a decision tree-based AI-Reasoner. Additionally, this paper investigates the time performance of the FM toolkit when applied to four AI models with different datasets. Lastly, a tutorial is provided to guide users in performing reasoning tasks using the FM toolkit.

{{</citation>}}


### (26/103) Argument Attribution Explanations in Quantitative Bipolar Argumentation Frameworks (Xiang Yin et al., 2023)

{{<citation>}}

Xiang Yin, Nico Potyka, Francesca Toni. (2023)  
**Argument Attribution Explanations in Quantitative Bipolar Argumentation Frameworks**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13582v1)  

---


**ABSTRACT**  
Argumentative explainable AI has been advocated by several in recent years, with an increasing interest on explaining the reasoning outcomes of Argumentation Frameworks (AFs). While there is a considerable body of research on qualitatively explaining the reasoning outcomes of AFs with debates/disputes/dialogues in the spirit of \emph{extension-based semantics}, explaining the quantitative reasoning outcomes of AFs under \emph{gradual semantics} has not received much attention, despite widespread use in applications. In this paper, we contribute to filling this gap by proposing a novel theory of \emph{Argument Attribution Explanations (AAEs)} by incorporating the spirit of feature attribution from machine learning in the context of Quantitative Bipolar Argumentation Frameworks (QBAFs): whereas feature attribution is used to determine the influence of features towards outputs of machine learning models, AAEs are used to determine the influence of arguments towards \emph{topic argument}s of interest. We study desirable properties of AAEs, including some new ones and some partially adapted from the literature to our setting. To demonstrate the applicability of our AAEs in practice, we conclude by carrying out two case studies in the scenarios of fake news detection and movie recommender systems.

{{</citation>}}


### (27/103) On Solving the Rubik's Cube with Domain-Independent Planners Using Standard Representations (Bharath Muppasani et al., 2023)

{{<citation>}}

Bharath Muppasani, Vishal Pallagani, Biplav Srivastava, Forest Agostinelli. (2023)  
**On Solving the Rubik's Cube with Domain-Independent Planners Using Standard Representations**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13552v1)  

---


**ABSTRACT**  
Rubik's Cube (RC) is a well-known and computationally challenging puzzle that has motivated AI researchers to explore efficient alternative representations and problem-solving methods. The ideal situation for planning here is that a problem be solved optimally and efficiently represented in a standard notation using a general-purpose solver and heuristics. The fastest solver today for RC is DeepCubeA with a custom representation, and another approach is with Scorpion planner with State-Action-Space+ (SAS+) representation. In this paper, we present the first RC representation in the popular PDDL language so that the domain becomes more accessible to PDDL planners, competitions, and knowledge engineering tools, and is more human-readable. We then bridge across existing approaches and compare performance. We find that in one comparable experiment, DeepCubeA solves all problems with varying complexities, albeit only 18\% are optimal plans. For the same problem set, Scorpion with SAS+ representation and pattern database heuristics solves 61.50\% problems, while FastDownward with PDDL representation and FF heuristic solves 56.50\% problems, out of which all the plans generated were optimal. Our study provides valuable insights into the trade-offs between representational choice and plan optimality that can help researchers design future strategies for challenging domains combining general-purpose solving methods (planning, reinforcement learning), heuristics, and representations (standard or custom).

{{</citation>}}


### (28/103) Counterfactual Explanation Policies in RL (Shripad V. Deshmukh et al., 2023)

{{<citation>}}

Shripad V. Deshmukh, Srivatsan R, Supriti Vijay, Jayakumar Subramanian, Chirag Agarwal. (2023)  
**Counterfactual Explanation Policies in RL**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.13192v1)  

---


**ABSTRACT**  
As Reinforcement Learning (RL) agents are increasingly employed in diverse decision-making problems using reward preferences, it becomes important to ensure that policies learned by these frameworks in mapping observations to a probability distribution of the possible actions are explainable. However, there is little to no work in the systematic understanding of these complex policies in a contrastive manner, i.e., what minimal changes to the policy would improve/worsen its performance to a desired level. In this work, we present COUNTERPOL, the first framework to analyze RL policies using counterfactual explanations in the form of minimal changes to the policy that lead to the desired outcome. We do so by incorporating counterfactuals in supervised learning in RL with the target outcome regulated using desired return. We establish a theoretical connection between Counterpol and widely used trust region-based policy optimization methods in RL. Extensive empirical analysis shows the efficacy of COUNTERPOL in generating explanations for (un)learning skills while keeping close to the original policy. Our results on five different RL environments with diverse state and action spaces demonstrate the utility of counterfactual explanations, paving the way for new frontiers in designing and developing counterfactual policies.

{{</citation>}}


## cs.LG (27)



### (29/103) MAEA: Multimodal Attribution for Embodied AI (Vidhi Jain et al., 2023)

{{<citation>}}

Vidhi Jain, Jayant Sravan Tamarapalli, Sahiti Yerramilli, Yonatan Bisk. (2023)  
**MAEA: Multimodal Attribution for Embodied AI**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13850v1)  

---


**ABSTRACT**  
Understanding multimodal perception for embodied AI is an open question because such inputs may contain highly complementary as well as redundant information for the task. A relevant direction for multimodal policies is understanding the global trends of each modality at the fusion layer. To this end, we disentangle the attributions for visual, language, and previous action inputs across different policies trained on the ALFRED dataset. Attribution analysis can be utilized to rank and group the failure scenarios, investigate modeling and dataset biases, and critically analyze multimodal EAI policies for robustness and user trust before deployment. We present MAEA, a framework to compute global attributions per modality of any differentiable policy. In addition, we show how attributions enable lower-level behavior analysis in EAI policies for language and visual attributions.

{{</citation>}}


### (30/103) Offline Reinforcement Learning with On-Policy Q-Function Regularization (Laixi Shi et al., 2023)

{{<citation>}}

Laixi Shi, Robert Dadashi, Yuejie Chi, Pablo Samuel Castro, Matthieu Geist. (2023)  
**Offline Reinforcement Learning with On-Policy Q-Function Regularization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.13824v1)  

---


**ABSTRACT**  
The core challenge of offline reinforcement learning (RL) is dealing with the (potentially catastrophic) extrapolation error induced by the distribution shift between the history dataset and the desired policy. A large portion of prior work tackles this challenge by implicitly/explicitly regularizing the learning policy towards the behavior policy, which is hard to estimate reliably in practice. In this work, we propose to regularize towards the Q-function of the behavior policy instead of the behavior policy itself, under the premise that the Q-function can be estimated more reliably and easily by a SARSA-style estimate and handles the extrapolation error more straightforwardly. We propose two algorithms taking advantage of the estimated Q-function through regularizations, and demonstrate they exhibit strong performance on the D4RL benchmarks.

{{</citation>}}


### (31/103) Gradient-Based Spectral Embeddings of Random Dot Product Graphs (Marcelo Fiori et al., 2023)

{{<citation>}}

Marcelo Fiori, Bernardo Marenco, Federico Larroca, Paola Bermolen, Gonzalo Mateos. (2023)  
**Gradient-Based Spectral Embeddings of Random Dot Product Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.13818v1)  

---


**ABSTRACT**  
The Random Dot Product Graph (RDPG) is a generative model for relational data, where nodes are represented via latent vectors in low-dimensional Euclidean space. RDPGs crucially postulate that edge formation probabilities are given by the dot product of the corresponding latent positions. Accordingly, the embedding task of estimating these vectors from an observed graph is typically posed as a low-rank matrix factorization problem. The workhorse Adjacency Spectral Embedding (ASE) enjoys solid statistical properties, but it is formally solving a surrogate problem and can be computationally intensive. In this paper, we bring to bear recent advances in non-convex optimization and demonstrate their impact to RDPG inference. We advocate first-order gradient descent methods to better solve the embedding problem, and to organically accommodate broader network embedding applications of practical relevance. Notably, we argue that RDPG embeddings of directed graphs loose interpretability unless the factor matrices are constrained to have orthogonal columns. We thus develop a novel feasible optimization method in the resulting manifold. The effectiveness of the graph representation learning framework is demonstrated on reproducible experiments with both synthetic and real network data. Our open-source algorithm implementations are scalable, and unlike the ASE they are robust to missing edge data and can track slowly-varying latent positions from streaming graphs.

{{</citation>}}


### (32/103) When Multi-Task Learning Meets Partial Supervision: A Computer Vision Review (Maxime Fontana et al., 2023)

{{<citation>}}

Maxime Fontana, Michael Spratling, Miaojing Shi. (2023)  
**When Multi-Task Learning Meets Partial Supervision: A Computer Vision Review**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2307.14382v1)  

---


**ABSTRACT**  
Multi-Task Learning (MTL) aims to learn multiple tasks simultaneously while exploiting their mutual relationships. By using shared resources to simultaneously calculate multiple outputs, this learning paradigm has the potential to have lower memory requirements and inference times compared to the traditional approach of using separate methods for each task. Previous work in MTL has mainly focused on fully-supervised methods, as task relationships can not only be leveraged to lower the level of data-dependency of those methods but they can also improve performance. However, MTL introduces a set of challenges due to a complex optimisation scheme and a higher labeling requirement. This review focuses on how MTL could be utilised under different partial supervision settings to address these challenges. First, this review analyses how MTL traditionally uses different parameter sharing techniques to transfer knowledge in between tasks. Second, it presents the different challenges arising from such a multi-objective optimisation scheme. Third, it introduces how task groupings can be achieved by analysing task relationships. Fourth, it focuses on how partially supervised methods applied to MTL can tackle the aforementioned challenges. Lastly, this review presents the available datasets, tools and benchmarking results of such methods.

{{</citation>}}


### (33/103) Robust Assignment of Labels for Active Learning with Sparse and Noisy Annotations (Daniel Kałuża et al., 2023)

{{<citation>}}

Daniel Kałuża, Andrzej Janusz, Dominik Ślęzak. (2023)  
**Robust Assignment of Labels for Active Learning with Sparse and Noisy Annotations**  

---
Primary Category: cs.LG  
Categories: I-2-6, cs-HC, cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2307.14380v1)  

---


**ABSTRACT**  
Supervised classification algorithms are used to solve a growing number of real-life problems around the globe. Their performance is strictly connected with the quality of labels used in training. Unfortunately, acquiring good-quality annotations for many tasks is infeasible or too expensive to be done in practice. To tackle this challenge, active learning algorithms are commonly employed to select only the most relevant data for labeling. However, this is possible only when the quality and quantity of labels acquired from experts are sufficient. Unfortunately, in many applications, a trade-off between annotating individual samples by multiple annotators to increase label quality vs. annotating new samples to increase the total number of labeled instances is necessary. In this paper, we address the issue of faulty data annotations in the context of active learning. In particular, we propose two novel annotation unification algorithms that utilize unlabeled parts of the sample space. The proposed methods require little to no intersection between samples annotated by different experts. Our experiments on four public datasets indicate the robustness and superiority of the proposed methods in both, the estimation of the annotator's reliability, and the assignment of actual labels, against the state-of-the-art algorithms and the simple majority voting.

{{</citation>}}


### (34/103) RED CoMETS: An ensemble classifier for symbolically represented multivariate time series (Luca A. Bennett et al., 2023)

{{<citation>}}

Luca A. Bennett, Zahraa S. Abdallah. (2023)  
**RED CoMETS: An ensemble classifier for symbolically represented multivariate time series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.13679v1)  

---


**ABSTRACT**  
Multivariate time series classification is a rapidly growing research field with practical applications in finance, healthcare, engineering, and more. The complexity of classifying multivariate time series data arises from its high dimensionality, temporal dependencies, and varying lengths. This paper introduces a novel ensemble classifier called RED CoMETS (Random Enhanced Co-eye for Multivariate Time Series), which addresses these challenges. RED CoMETS builds upon the success of Co-eye, an ensemble classifier specifically designed for symbolically represented univariate time series, and extends its capabilities to handle multivariate data. The performance of RED CoMETS is evaluated on benchmark datasets from the UCR archive, where it demonstrates competitive accuracy when compared to state-of-the-art techniques in multivariate settings. Notably, it achieves the highest reported accuracy in the literature for the 'HandMovementDirection' dataset. Moreover, the proposed method significantly reduces computation time compared to Co-eye, making it an efficient and effective choice for multivariate time series classification.

{{</citation>}}


### (35/103) FedDRL: A Trustworthy Federated Learning Model Fusion Method Based on Staged Reinforcement Learning (Leiming Chen et al., 2023)

{{<citation>}}

Leiming Chen, Cihao Dong, Sibo Qiao, Ziling Huang, Kai Wang, Yuming Nie, Zhaoxiang Hou, Cheewei Tan. (2023)  
**FedDRL: A Trustworthy Federated Learning Model Fusion Method Based on Staged Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.13716v1)  

---


**ABSTRACT**  
Traditional federated learning uses the number of samples to calculate the weights of each client model and uses this fixed weight value to fusion the global model. However, in practical scenarios, each client's device and data heterogeneity leads to differences in the quality of each client's model. Thus the contribution to the global model is not wholly determined by the sample size. In addition, if clients intentionally upload low-quality or malicious models, using these models for aggregation will lead to a severe decrease in global model accuracy. Traditional federated learning algorithms do not address these issues. To solve this probelm, we propose FedDRL, a model fusion approach using reinforcement learning based on a two staged approach. In the first stage, Our method could filter out malicious models and selects trusted client models to participate in the model fusion. In the second stage, the FedDRL algorithm adaptively adjusts the weights of the trusted client models and aggregates the optimal global model. We also define five model fusion scenarios and compare our method with two baseline algorithms in those scenarios. The experimental results show that our algorithm has higher reliability than other algorithms while maintaining accuracy.

{{</citation>}}


### (36/103) Safety Margins for Reinforcement Learning (Alexander Grushin et al., 2023)

{{<citation>}}

Alexander Grushin, Walt Woods, Alvaro Velasquez, Simon Khan. (2023)  
**Safety Margins for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: 68T07, I-2-6, cs-AI, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.13642v1)  

---


**ABSTRACT**  
Any autonomous controller will be unsafe in some situations. The ability to quantitatively identify when these unsafe situations are about to occur is crucial for drawing timely human oversight in, e.g., freight transportation applications. In this work, we demonstrate that the true criticality of an agent's situation can be robustly defined as the mean reduction in reward given some number of random actions. Proxy criticality metrics that are computable in real-time (i.e., without actually simulating the effects of random actions) can be compared to the true criticality, and we show how to leverage these proxy metrics to generate safety margins, which directly tie the consequences of potentially incorrect actions to an anticipated loss in overall performance. We evaluate our approach on learned policies from APE-X and A3C within an Atari environment, and demonstrate how safety margins decrease as agents approach failure states. The integration of safety margins into programs for monitoring deployed agents allows for the real-time identification of potentially catastrophic situations.

{{</citation>}}


### (37/103) Team Intro to AI team8 at CoachAI Badminton Challenge 2023: Advanced ShuttleNet for Shot Predictions (Shih-Hong Chen et al., 2023)

{{<citation>}}

Shih-Hong Chen, Pin-Hsuan Chou, Yong-Fu Liu, Chien-An Han. (2023)  
**Team Intro to AI team8 at CoachAI Badminton Challenge 2023: Advanced ShuttleNet for Shot Predictions**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13715v1)  

---


**ABSTRACT**  
In this paper, our objective is to improve the performance of the existing framework ShuttleNet in predicting badminton shot types and locations by leveraging past strokes. We participated in the CoachAI Badminton Challenge at IJCAI 2023 and achieved significantly better results compared to the baseline. Ultimately, our team achieved the first position in the competition and we made our code available.

{{</citation>}}


### (38/103) Forecasting, capturing and activation of carbon-dioxide (CO$_2$): Integration of Time Series Analysis, Machine Learning, and Material Design (Suchetana Sadhukhan et al., 2023)

{{<citation>}}

Suchetana Sadhukhan, Vivek Kumar Yadav. (2023)  
**Forecasting, capturing and activation of carbon-dioxide (CO$_2$): Integration of Time Series Analysis, Machine Learning, and Material Design**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM, Time Series  
[Paper Link](http://arxiv.org/abs/2307.14374v1)  

---


**ABSTRACT**  
This study provides a comprehensive time series analysis of daily industry-specific, country-wise CO$_2$ emissions from January 2019 to February 2023. The research focuses on the Power, Industry, Ground Transport, Domestic Aviation, and International Aviation sectors in European countries (EU27 & UK, Italy, Germany, Spain) and India, utilizing near-real-time activity data from the Carbon Monitor research initiative. To identify regular emission patterns, the data from the year 2020 is excluded due to the disruptive effects caused by the COVID-19 pandemic. The study then performs a principal component analysis (PCA) to determine the key contributors to CO$_2$ emissions. The analysis reveals that the Power, Industry, and Ground Transport sectors account for a significant portion of the variance in the dataset. A 7-day moving averaged dataset is employed for further analysis to facilitate robust predictions. This dataset captures both short-term and long-term trends and enhances the quality of the data for prediction purposes. The study utilizes Long Short-Term Memory (LSTM) models on the 7-day moving averaged dataset to effectively predict emissions and provide insights for policy decisions, mitigation strategies, and climate change efforts. During the training phase, the stability and convergence of the LSTM models are ensured, which guarantees their reliability in the testing phase. The evaluation of the loss function indicates this reliability. The model achieves high efficiency, as demonstrated by $R^2$ values ranging from 0.8242 to 0.995 for various countries and sectors. Furthermore, there is a proposal for utilizing scandium and boron/aluminium-based thin films as exceptionally efficient materials for capturing CO$_2$ (with a binding energy range from -3.0 to -3.5 eV). These materials are shown to surpass the affinity of graphene and boron nitride sheets in this regard.

{{</citation>}}


### (39/103) Settling the Sample Complexity of Online Reinforcement Learning (Zihan Zhang et al., 2023)

{{<citation>}}

Zihan Zhang, Yuxin Chen, Jason D. Lee, Simon S. Du. (2023)  
**Settling the Sample Complexity of Online Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.13586v1)  

---


**ABSTRACT**  
A central issue lying at the heart of online reinforcement learning (RL) is data efficiency. While a number of recent works achieved asymptotically minimal regret in online RL, the optimality of these results is only guaranteed in a ``large-sample'' regime, imposing enormous burn-in cost in order for their algorithms to operate optimally. How to achieve minimax-optimal regret without incurring any burn-in cost has been an open problem in RL theory.   We settle this problem for the context of finite-horizon inhomogeneous Markov decision processes. Specifically, we prove that a modified version of Monotonic Value Propagation (MVP), a model-based algorithm proposed by \cite{zhang2020reinforcement}, achieves a regret on the order of (modulo log factors) \begin{equation*}   \min\big\{ \sqrt{SAH^3K}, \,HK \big\}, \end{equation*} where $S$ is the number of states, $A$ is the number of actions, $H$ is the planning horizon, and $K$ is the total number of episodes. This regret matches the minimax lower bound for the entire range of sample size $K\geq 1$, essentially eliminating any burn-in requirement. It also translates to a PAC sample complexity (i.e., the number of episodes needed to yield $\varepsilon$-accuracy) of $\frac{SAH^3}{\varepsilon^2}$ up to log factor, which is minimax-optimal for the full $\varepsilon$-range.   Further, we extend our theory to unveil the influences of problem-dependent quantities like the optimal value/cost and certain variances. The key technical innovation lies in the development of a new regret decomposition strategy and a novel analysis paradigm to decouple complicated statistical dependency -- a long-standing challenge facing the analysis of online RL in the sample-hungry regime.

{{</citation>}}


### (40/103) Continuous Time Evidential Distributions for Irregular Time Series (Taylor W. Killian et al., 2023)

{{<citation>}}

Taylor W. Killian, Haoran Zhang, Thomas Hartvigsen, Ava P. Amini. (2023)  
**Continuous Time Evidential Distributions for Irregular Time Series**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.13503v1)  

---


**ABSTRACT**  
Prevalent in many real-world settings such as healthcare, irregular time series are challenging to formulate predictions from. It is difficult to infer the value of a feature at any given time when observations are sporadic, as it could take on a range of values depending on when it was last observed. To characterize this uncertainty we present EDICT, a strategy that learns an evidential distribution over irregular time series in continuous time. This distribution enables well-calibrated and flexible inference of partially observed features at any time of interest, while expanding uncertainty temporally for sparse, irregular observations. We demonstrate that EDICT attains competitive performance on challenging time series classification tasks and enabling uncertainty-guided inference when encountering noisy data.

{{</citation>}}


### (41/103) Finding Money Launderers Using Heterogeneous Graph Neural Networks (Fredrik Johannessen et al., 2023)

{{<citation>}}

Fredrik Johannessen, Martin Jullum. (2023)  
**Finding Money Launderers Using Heterogeneous Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.13499v1)  

---


**ABSTRACT**  
Current anti-money laundering (AML) systems, predominantly rule-based, exhibit notable shortcomings in efficiently and precisely detecting instances of money laundering. As a result, there has been a recent surge toward exploring alternative approaches, particularly those utilizing machine learning. Since criminals often collaborate in their money laundering endeavors, accounting for diverse types of customer relations and links becomes crucial. In line with this, the present paper introduces a graph neural network (GNN) approach to identify money laundering activities within a large heterogeneous network constructed from real-world bank transactions and business role data belonging to DNB, Norway's largest bank. Specifically, we extend the homogeneous GNN method known as the Message Passing Neural Network (MPNN) to operate effectively on a heterogeneous graph. As part of this procedure, we propose a novel method for aggregating messages across different edges of the graph. Our findings highlight the importance of using an appropriate GNN architecture when combining information in heterogeneous graphs. The performance results of our model demonstrate great potential in enhancing the quality of electronic surveillance systems employed by banks to detect instances of money laundering. To the best of our knowledge, this is the first published work applying GNN on a large real-world heterogeneous network for anti-money laundering purposes.

{{</citation>}}


### (42/103) Combinatorial Auctions and Graph Neural Networks for Local Energy Flexibility Markets (Awadelrahman M. A. Ahmed et al., 2023)

{{<citation>}}

Awadelrahman M. A. Ahmed, Frank Eliassen, Yan Zhang. (2023)  
**Combinatorial Auctions and Graph Neural Networks for Local Energy Flexibility Markets**  

---
Primary Category: cs.LG  
Categories: cs-GT, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.13470v1)  

---


**ABSTRACT**  
This paper proposes a new combinatorial auction framework for local energy flexibility markets, which addresses the issue of prosumers' inability to bundle multiple flexibility time intervals. To solve the underlying NP-complete winner determination problems, we present a simple yet powerful heterogeneous tri-partite graph representation and design graph neural network-based models. Our models achieve an average optimal value deviation of less than 5\% from an off-the-shelf optimization tool and show linear inference time complexity compared to the exponential complexity of the commercial solver. Contributions and results demonstrate the potential of using machine learning to efficiently allocate energy flexibility resources in local markets and solving optimization problems in general.

{{</citation>}}


### (43/103) Network Traffic Classification based on Single Flow Time Series Analysis (Josef Koumar et al., 2023)

{{<citation>}}

Josef Koumar, Karel Hynek, Tomáš Čejka. (2023)  
**Network Traffic Classification based on Single Flow Time Series Analysis**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2307.13434v1)  

---


**ABSTRACT**  
Network traffic monitoring using IP flows is used to handle the current challenge of analyzing encrypted network communication. Nevertheless, the packet aggregation into flow records naturally causes information loss; therefore, this paper proposes a novel flow extension for traffic features based on the time series analysis of the Single Flow Time series, i.e., a time series created by the number of bytes in each packet and its timestamp. We propose 69 universal features based on the statistical analysis of data points, time domain analysis, packet distribution within the flow timespan, time series behavior, and frequency domain analysis. We have demonstrated the usability and universality of the proposed feature vector for various network traffic classification tasks using 15 well-known publicly available datasets. Our evaluation shows that the novel feature vector achieves classification performance similar or better than related works on both binary and multiclass classification tasks. In more than half of the evaluated tasks, the classification performance increased by up to 5\%.

{{</citation>}}


### (44/103) On the Learning Dynamics of Attention Networks (Rahul Vashisht et al., 2023)

{{<citation>}}

Rahul Vashisht, Harish G. Ramaswamy. (2023)  
**On the Learning Dynamics of Attention Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2307.13421v2)  

---


**ABSTRACT**  
Attention models are typically learned by optimizing one of three standard loss functions that are variously called -- soft attention, hard attention, and latent variable marginal likelihood (LVML) attention. All three paradigms are motivated by the same goal of finding two models -- a `focus' model that `selects' the right \textit{segment} of the input and a `classification' model that processes the selected segment into the target label. However, they differ significantly in the way the selected segments are aggregated, resulting in distinct dynamics and final results. We observe a unique signature of models learned using these paradigms and explain this as a consequence of the evolution of the classification model under gradient descent when the focus model is fixed. We also analyze these paradigms in a simple setting and derive closed-form expressions for the parameter trajectory under gradient flow. With the soft attention loss, the focus model improves quickly at initialization and splutters later on. On the other hand, hard attention loss behaves in the opposite fashion. Based on our observations, we propose a simple hybrid approach that combines the advantages of the different loss functions and demonstrates it on a collection of semi-synthetic and real-world datasets

{{</citation>}}


### (45/103) Mitigating Memory Wall Effects in CNN Engines with On-the-Fly Weights Generation (Stylianos I. Venieris et al., 2023)

{{<citation>}}

Stylianos I. Venieris, Javier Fernandez-Marques, Nicholas D. Lane. (2023)  
**Mitigating Memory Wall Effects in CNN Engines with On-the-Fly Weights Generation**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13412v1)  

---


**ABSTRACT**  
The unprecedented accuracy of convolutional neural networks (CNNs) across a broad range of AI tasks has led to their widespread deployment in mobile and embedded settings. In a pursuit for high-performance and energy-efficient inference, significant research effort has been invested in the design of FPGA-based CNN accelerators. In this context, single computation engines constitute a popular approach to support diverse CNN modes without the overhead of fabric reconfiguration. Nevertheless, this flexibility often comes with significantly degraded performance on memory-bound layers and resource underutilisation due to the suboptimal mapping of certain layers on the engine's fixed configuration. In this work, we investigate the implications in terms of CNN engine design for a class of models that introduce a pre-convolution stage to decompress the weights at run time. We refer to these approaches as on-the-fly. This paper presents unzipFPGA, a novel CNN inference system that counteracts the limitations of existing CNN engines. The proposed framework comprises a novel CNN hardware architecture that introduces a weights generator module that enables the on-chip on-the-fly generation of weights, alleviating the negative impact of limited bandwidth on memory-bound layers. We further enhance unzipFPGA with an automated hardware-aware methodology that tailors the weights generation mechanism to the target CNN-device pair, leading to an improved accuracy-performance balance. Finally, we introduce an input selective processing element (PE) design that balances the load between PEs in suboptimally mapped layers. The proposed framework yields hardware designs that achieve an average of 2.57x performance efficiency gain over highly optimised GPU designs for the same power constraints and up to 3.94x higher performance density over a diverse range of state-of-the-art FPGA-based CNN accelerators.

{{</citation>}}


### (46/103) Counterfactual Explanation via Search in Gaussian Mixture Distributed Latent Space (Xuan Zhao et al., 2023)

{{<citation>}}

Xuan Zhao, Klaus Broelemann, Gjergji Kasneci. (2023)  
**Counterfactual Explanation via Search in Gaussian Mixture Distributed Latent Space**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13390v1)  

---


**ABSTRACT**  
Counterfactual Explanations (CEs) are an important tool in Algorithmic Recourse for addressing two questions: 1. What are the crucial factors that led to an automated prediction/decision? 2. How can these factors be changed to achieve a more favorable outcome from a user's perspective? Thus, guiding the user's interaction with AI systems by proposing easy-to-understand explanations and easy-to-attain feasible changes is essential for the trustworthy adoption and long-term acceptance of AI systems. In the literature, various methods have been proposed to generate CEs, and different quality measures have been suggested to evaluate these methods. However, the generation of CEs is usually computationally expensive, and the resulting suggestions are unrealistic and thus non-actionable. In this paper, we introduce a new method to generate CEs for a pre-trained binary classifier by first shaping the latent space of an autoencoder to be a mixture of Gaussian distributions. CEs are then generated in latent space by linear interpolation between the query sample and the centroid of the target class. We show that our method maintains the characteristics of the input sample during the counterfactual search. In various experiments, we show that the proposed method is competitive based on different quality measures on image and tabular datasets -- efficiently returns results that are closer to the original data manifold compared to three state-of-the-art methods, which are essential for realistic high-dimensional machine learning applications.

{{</citation>}}


### (47/103) Submodular Reinforcement Learning (Manish Prajapat et al., 2023)

{{<citation>}}

Manish Prajapat, Mojmír Mutný, Melanie N. Zeilinger, Andreas Krause. (2023)  
**Submodular Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.13372v1)  

---


**ABSTRACT**  
In reinforcement learning (RL), rewards of states are typically considered additive, and following the Markov assumption, they are $\textit{independent}$ of states visited previously. In many important applications, such as coverage control, experiment design and informative path planning, rewards naturally have diminishing returns, i.e., their value decreases in light of similar states visited previously. To tackle this, we propose $\textit{submodular RL}$ (SubRL), a paradigm which seeks to optimize more general, non-additive (and history-dependent) rewards modelled via submodular set functions which capture diminishing returns. Unfortunately, in general, even in tabular settings, we show that the resulting optimization problem is hard to approximate. On the other hand, motivated by the success of greedy algorithms in classical submodular optimization, we propose SubPO, a simple policy gradient-based algorithm for SubRL that handles non-additive rewards by greedily maximizing marginal gains. Indeed, under some assumptions on the underlying Markov Decision Process (MDP), SubPO recovers optimal constant factor approximations of submodular bandits. Moreover, we derive a natural policy gradient approach for locally optimizing SubRL instances even in large state- and action- spaces. We showcase the versatility of our approach by applying SubPO to several applications, such as biodiversity monitoring, Bayesian experiment design, informative path planning, and coverage maximization. Our results demonstrate sample efficiency, as well as scalability to high-dimensional state-action spaces.

{{</citation>}}


### (48/103) QuIP: 2-Bit Quantization of Large Language Models With Guarantees (Jerry Chee et al., 2023)

{{<citation>}}

Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, Christopher De Sa. (2023)  
**QuIP: 2-Bit Quantization of Large Language Models With Guarantees**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model, Quantization  
[Paper Link](http://arxiv.org/abs/2307.13304v1)  

---


**ABSTRACT**  
This work studies post-training parameter quantization in large language models (LLMs). We introduce quantization with incoherence processing (QuIP), a new method based on the insight that quantization benefits from incoherent weight and Hessian matrices, i.e., from the weights and the directions in which it is important to round them accurately being unaligned with the coordinate axes. QuIP consists of two steps: (1) an adaptive rounding procedure minimizing a quadratic proxy objective; (2) efficient pre- and post-processing that ensures weight and Hessian incoherence via multiplication by random orthogonal matrices. We complement QuIP with the first theoretical analysis for an LLM-scale quantization algorithm, and show that our theory also applies to an existing method, OPTQ. Empirically, we find that our incoherence preprocessing improves several existing quantization algorithms and yields the first LLM quantization methods that produce viable results using only two bits per weight. Our code can be found at https://github.com/jerry-chee/QuIP .

{{</citation>}}


### (49/103) Curvature-based Transformer for Molecular Property Prediction (Yili Chen et al., 2023)

{{<citation>}}

Yili Chen, Zhengyu Li, Zheng Wan, Hui Yu, Xian Wei. (2023)  
**Curvature-based Transformer for Molecular Property Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-QM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.13275v1)  

---


**ABSTRACT**  
The prediction of molecular properties is one of the most important and challenging tasks in the field of artificial intelligence-based drug design. Among the current mainstream methods, the most commonly used feature representation for training DNN models is based on SMILES and molecular graphs, although these methods are concise and effective, they also limit the ability to capture spatial information. In this work, we propose Curvature-based Transformer to improve the ability of Graph Transformer neural network models to extract structural information on molecular graph data by introducing Discretization of Ricci Curvature. To embed the curvature in the model, we add the curvature information of the graph as positional Encoding to the node features during the attention-score calculation. This method can introduce curvature information from graph data without changing the original network architecture, and it has the potential to be extended to other models. We performed experiments on chemical molecular datasets including PCQM4M-LST, MoleculeNet and compared with models such as Uni-Mol, Graphormer, and the results show that this method can achieve the state-of-the-art results. It is proved that the discretized Ricci curvature also reflects the structural and functional relationship while describing the local geometry of the graph molecular data.

{{</citation>}}


### (50/103) Unbiased Weight Maximization (Stephen Chung, 2023)

{{<citation>}}

Stephen Chung. (2023)  
**Unbiased Weight Maximization**  

---
Primary Category: cs.LG  
Categories: I-2-6; I-2-8; I-5-1, cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.13270v1)  

---


**ABSTRACT**  
A biologically plausible method for training an Artificial Neural Network (ANN) involves treating each unit as a stochastic Reinforcement Learning (RL) agent, thereby considering the network as a team of agents. Consequently, all units can learn via REINFORCE, a local learning rule modulated by a global reward signal, which aligns more closely with biologically observed forms of synaptic plasticity. Nevertheless, this learning method is often slow and scales poorly with network size due to inefficient structural credit assignment, since a single reward signal is broadcast to all units without considering individual contributions. Weight Maximization, a proposed solution, replaces a unit's reward signal with the norm of its outgoing weight, thereby allowing each hidden unit to maximize the norm of the outgoing weight instead of the global reward signal. In this research report, we analyze the theoretical properties of Weight Maximization and propose a variant, Unbiased Weight Maximization. This new approach provides an unbiased learning rule that increases learning speed and improves asymptotic performance. Notably, to our knowledge, this is the first learning rule for a network of Bernoulli-logistic units that is unbiased and scales well with the number of network's units in terms of learning speed.

{{</citation>}}


### (51/103) Structural Credit Assignment with Coordinated Exploration (Stephen Chung, 2023)

{{<citation>}}

Stephen Chung. (2023)  
**Structural Credit Assignment with Coordinated Exploration**  

---
Primary Category: cs.LG  
Categories: I-2-6; I-2-8; I-5-1, cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.13256v1)  

---


**ABSTRACT**  
A biologically plausible method for training an Artificial Neural Network (ANN) involves treating each unit as a stochastic Reinforcement Learning (RL) agent, thereby considering the network as a team of agents. Consequently, all units can learn via REINFORCE, a local learning rule modulated by a global reward signal, which aligns more closely with biologically observed forms of synaptic plasticity. However, this learning method tends to be slow and does not scale well with the size of the network. This inefficiency arises from two factors impeding effective structural credit assignment: (i) all units independently explore the network, and (ii) a single reward is used to evaluate the actions of all units. Accordingly, methods aimed at improving structural credit assignment can generally be classified into two categories. The first category includes algorithms that enable coordinated exploration among units, such as MAP propagation. The second category encompasses algorithms that compute a more specific reward signal for each unit within the network, like Weight Maximization and its variants. In this research report, our focus is on the first category. We propose the use of Boltzmann machines or a recurrent network for coordinated exploration. We show that the negative phase, which is typically necessary to train Boltzmann machines, can be removed. The resulting learning rules are similar to the reward-modulated Hebbian learning rule. Experimental results demonstrate that coordinated exploration significantly exceeds independent exploration in training speed for multiple stochastic and discrete units based on REINFORCE, even surpassing straight-through estimator (STE) backpropagation.

{{</citation>}}


### (52/103) RoSAS: Deep Semi-Supervised Anomaly Detection with Contamination-Resilient Continuous Supervision (Hongzuo Xu et al., 2023)

{{<citation>}}

Hongzuo Xu, Yijie Wang, Guansong Pang, Songlei Jian, Ning Liu, Yongjun Wang. (2023)  
**RoSAS: Deep Semi-Supervised Anomaly Detection with Contamination-Resilient Continuous Supervision**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Anomaly Detection, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2307.13239v1)  

---


**ABSTRACT**  
Semi-supervised anomaly detection methods leverage a few anomaly examples to yield drastically improved performance compared to unsupervised models. However, they still suffer from two limitations: 1) unlabeled anomalies (i.e., anomaly contamination) may mislead the learning process when all the unlabeled data are employed as inliers for model training; 2) only discrete supervision information (such as binary or ordinal data labels) is exploited, which leads to suboptimal learning of anomaly scores that essentially take on a continuous distribution. Therefore, this paper proposes a novel semi-supervised anomaly detection method, which devises \textit{contamination-resilient continuous supervisory signals}. Specifically, we propose a mass interpolation method to diffuse the abnormality of labeled anomalies, thereby creating new data samples labeled with continuous abnormal degrees. Meanwhile, the contaminated area can be covered by new data samples generated via combinations of data with correct labels. A feature learning-based objective is added to serve as an optimization constraint to regularize the network and further enhance the robustness w.r.t. anomaly contamination. Extensive experiments on 11 real-world datasets show that our approach significantly outperforms state-of-the-art competitors by 20%-30% in AUC-PR and obtains more robust and superior performance in settings with different anomaly contamination levels and varying numbers of labeled anomalies. The source code is available at https://github.com/xuhongzuo/rosas/.

{{</citation>}}


### (53/103) FedMEKT: Distillation-based Embedding Knowledge Transfer for Multimodal Federated Learning (Huy Q. Le et al., 2023)

{{<citation>}}

Huy Q. Le, Minh N. H. Nguyen, Chu Myaet Thwal, Yu Qiao, Chaoning Zhang, Choong Seon Hong. (2023)  
**FedMEKT: Distillation-based Embedding Knowledge Transfer for Multimodal Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2307.13214v1)  

---


**ABSTRACT**  
Federated learning (FL) enables a decentralized machine learning paradigm for multiple clients to collaboratively train a generalized global model without sharing their private data. Most existing works simply propose typical FL systems for single-modal data, thus limiting its potential on exploiting valuable multimodal data for future personalized applications. Furthermore, the majority of FL approaches still rely on the labeled data at the client side, which is limited in real-world applications due to the inability of self-annotation from users. In light of these limitations, we propose a novel multimodal FL framework that employs a semi-supervised learning approach to leverage the representations from different modalities. Bringing this concept into a system, we develop a distillation-based multimodal embedding knowledge transfer mechanism, namely FedMEKT, which allows the server and clients to exchange the joint knowledge of their learning models extracted from a small multimodal proxy dataset. Our FedMEKT iteratively updates the generalized global encoders with the joint embedding knowledge from the participating clients. Thereby, to address the modality discrepancy and labeled data constraint in existing FL systems, our proposed FedMEKT comprises local multimodal autoencoder learning, generalized multimodal autoencoder construction, and generalized classifier learning. Through extensive experiments on three multimodal human activity recognition datasets, we demonstrate that FedMEKT achieves superior global encoder performance on linear evaluation and guarantees user privacy for personal data and model parameters while demanding less communication cost than other baselines.

{{</citation>}}


### (54/103) Transferability of Graph Neural Networks using Graphon and Sampling Theories (A. Martina Neuman et al., 2023)

{{<citation>}}

A. Martina Neuman, Jason J. Bramburger. (2023)  
**Transferability of Graph Neural Networks using Graphon and Sampling Theories**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.13206v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) have become powerful tools for processing graph-based information in various domains. A desirable property of GNNs is transferability, where a trained network can swap in information from a different graph without retraining and retain its accuracy. A recent method of capturing transferability of GNNs is through the use of graphons, which are symmetric, measurable functions representing the limit of large dense graphs. In this work, we contribute to the application of graphons to GNNs by presenting an explicit two-layer graphon neural network (WNN) architecture. We prove its ability to approximate bandlimited signals within a specified error tolerance using a minimal number of network weights. We then leverage this result, to establish the transferability of an explicit two-layer GNN over all sufficiently large graphs in a sequence converging to a graphon. Our work addresses transferability between both deterministic weighted graphs and simple random graphs and overcomes issues related to the curse of dimensionality that arise in other GNN results. The proposed WNN and GNN architectures offer practical solutions for handling graph data of varying sizes while maintaining performance guarantees without extensive retraining.

{{</citation>}}


### (55/103) Neural Memory Decoding with EEG Data and Representation Learning (Glenn Bruns et al., 2023)

{{<citation>}}

Glenn Bruns, Michael Haidar, Federico Rubino. (2023)  
**Neural Memory Decoding with EEG Data and Representation Learning**  

---
Primary Category: cs.LG  
Categories: H-3-3; I-2-1; I-2-6; J-3, cs-LG, cs.LG, q-bio-NC  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2307.13181v1)  

---


**ABSTRACT**  
We describe a method for the neural decoding of memory from EEG data. Using this method, a concept being recalled can be identified from an EEG trace with an average top-1 accuracy of about 78.4% (chance 4%). The method employs deep representation learning with supervised contrastive loss to map an EEG recording of brain activity to a low-dimensional space. Because representation learning is used, concepts can be identified even if they do not appear in the training data set. However, reference EEG data must exist for each such concept. We also show an application of the method to the problem of information retrieval. In neural information retrieval, EEG data is captured while a user recalls the contents of a document, and a list of links to predicted documents is produced.

{{</citation>}}


## cs.CL (17)



### (56/103) ARC-NLP at Multimodal Hate Speech Event Detection 2023: Multimodal Methods Boosted by Ensemble Learning, Syntactical and Entity Features (Umitcan Sahin et al., 2023)

{{<citation>}}

Umitcan Sahin, Izzet Emre Kucukkaya, Oguzhan Ozcelik, Cagri Toraman. (2023)  
**ARC-NLP at Multimodal Hate Speech Event Detection 2023: Multimodal Methods Boosted by Ensemble Learning, Syntactical and Entity Features**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: Event Detection, NLP  
[Paper Link](http://arxiv.org/abs/2307.13829v1)  

---


**ABSTRACT**  
Text-embedded images can serve as a means of spreading hate speech, propaganda, and extremist beliefs. Throughout the Russia-Ukraine war, both opposing factions heavily relied on text-embedded images as a vehicle for spreading propaganda and hate speech. Ensuring the effective detection of hate speech and propaganda is of utmost importance to mitigate the negative effect of hate speech dissemination. In this paper, we outline our methodologies for two subtasks of Multimodal Hate Speech Event Detection 2023. For the first subtask, hate speech detection, we utilize multimodal deep learning models boosted by ensemble learning and syntactical text attributes. For the second subtask, target detection, we employ multimodal deep learning models boosted by named entity features. Through experimentation, we demonstrate the superior performance of our models compared to all textual, visual, and text-visual baselines employed in multimodal hate speech detection. Furthermore, our models achieve the first place in both subtasks on the final leaderboard of the shared task.

{{</citation>}}


### (57/103) Watermarking Conditional Text Generation for AI Detection: Unveiling Challenges and a Semantic-Aware Watermark Remedy (Yu Fu et al., 2023)

{{<citation>}}

Yu Fu, Deyi Xiong, Yue Dong. (2023)  
**Watermarking Conditional Text Generation for AI Detection: Unveiling Challenges and a Semantic-Aware Watermark Remedy**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CR, cs.CL  
Keywords: AI, T5, Text Generation  
[Paper Link](http://arxiv.org/abs/2307.13808v1)  

---


**ABSTRACT**  
To mitigate potential risks associated with language models, recent AI detection research proposes incorporating watermarks into machine-generated text through random vocabulary restrictions and utilizing this information for detection. While these watermarks only induce a slight deterioration in perplexity, our empirical investigation reveals a significant detriment to the performance of conditional text generation. To address this issue, we introduce a simple yet effective semantic-aware watermarking algorithm that considers the characteristics of conditional text generation and the input context. Experimental results demonstrate that our proposed method yields substantial improvements across various text generation models, including BART and Flan-T5, in tasks such as summarization and data-to-text generation while maintaining detection ability.

{{</citation>}}


### (58/103) Is GPT a Computational Model of Emotion? Detailed Analysis (Ala N. Tak et al., 2023)

{{<citation>}}

Ala N. Tak, Jonathan Gratch. (2023)  
**Is GPT a Computational Model of Emotion? Detailed Analysis**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-HC, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2307.13779v1)  

---


**ABSTRACT**  
This paper investigates the emotional reasoning abilities of the GPT family of large language models via a component perspective. The paper first examines how the model reasons about autobiographical memories. Second, it systematically varies aspects of situations to impact emotion intensity and coping tendencies. Even without the use of prompt engineering, it is shown that GPT's predictions align significantly with human-provided appraisals and emotional labels. However, GPT faces difficulties predicting emotion intensity and coping responses. GPT-4 showed the highest performance in the initial study but fell short in the second, despite providing superior results after minor prompt engineering. This assessment brings up questions on how to effectively employ the strong points and address the weak areas of these models, particularly concerning response variability. These studies underscore the merits of evaluating models from a componential perspective.

{{</citation>}}


### (59/103) Combating the Curse of Multilinguality in Cross-Lingual WSD by Aligning Sparse Contextualized Word Representations (Gábor Berend, 2023)

{{<citation>}}

Gábor Berend. (2023)  
**Combating the Curse of Multilinguality in Cross-Lingual WSD by Aligning Sparse Contextualized Word Representations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Multilingual, Word Representation, Word Representations  
[Paper Link](http://arxiv.org/abs/2307.13776v1)  

---


**ABSTRACT**  
In this paper, we advocate for using large pre-trained monolingual language models in cross lingual zero-shot word sense disambiguation (WSD) coupled with a contextualized mapping mechanism. We also report rigorous experiments that illustrate the effectiveness of employing sparse contextualized word representations obtained via a dictionary learning procedure. Our experimental results demonstrate that the above modifications yield a significant improvement of nearly 6.5 points of increase in the average F-score (from 62.0 to 68.5) over a collection of 17 typologically diverse set of target languages. We release our source code for replicating our experiments at https://github.com/begab/sparsity_makes_sense.

{{</citation>}}


### (60/103) Evaluating Large Language Models for Radiology Natural Language Processing (Zhengliang Liu et al., 2023)

{{<citation>}}

Zhengliang Liu, Tianyang Zhong, Yiwei Li, Yutong Zhang, Yi Pan, Zihao Zhao, Peixin Dong, Chao Cao, Yuxiao Liu, Peng Shu, Yaonai Wei, Zihao Wu, Chong Ma, Jiaqi Wang, Sheng Wang, Mengyue Zhou, Zuowei Jiang, Chunlin Li, Jason Holmes, Shaochen Xu, Lu Zhang, Haixing Dai, Kai Zhang, Lin Zhao, Yuanhao Chen, Xu Liu, Peilong Wang, Pingkun Yan, Jun Liu, Bao Ge, Lichao Sun, Dajiang Zhu, Xiang Li, Wei Liu, Xiaoyan Cai, Xintao Hu, Xi Jiang, Shu Zhang, Xin Zhang, Tuo Zhang, Shijie Zhao, Quanzheng Li, Hongtu Zhu, Dinggang Shen, Tianming Liu. (2023)  
**Evaluating Large Language Models for Radiology Natural Language Processing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2307.13693v2)  

---


**ABSTRACT**  
The rise of large language models (LLMs) has marked a pivotal shift in the field of natural language processing (NLP). LLMs have revolutionized a multitude of domains, and they have made a significant impact in the medical field. Large language models are now more abundant than ever, and many of these models exhibit bilingual capabilities, proficient in both English and Chinese. However, a comprehensive evaluation of these models remains to be conducted. This lack of assessment is especially apparent within the context of radiology NLP. This study seeks to bridge this gap by critically evaluating thirty two LLMs in interpreting radiology reports, a crucial component of radiology NLP. Specifically, the ability to derive impressions from radiologic findings is assessed. The outcomes of this evaluation provide key insights into the performance, strengths, and weaknesses of these LLMs, informing their practical applications within the medical domain.

{{</citation>}}


### (61/103) ARB: Advanced Reasoning Benchmark for Large Language Models (Tomohiro Sawada et al., 2023)

{{<citation>}}

Tomohiro Sawada, Daniel Paleka, Alexander Havrilla, Pranav Tadepalli, Paula Vidas, Alexander Kranias, John J. Nay, Kshitij Gupta, Aran Komatsuzaki. (2023)  
**ARB: Advanced Reasoning Benchmark for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2307.13692v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable performance on various quantitative reasoning and knowledge benchmarks. However, many of these benchmarks are losing utility as LLMs get increasingly high scores, despite not yet reaching expert performance in these domains. We introduce ARB, a novel benchmark composed of advanced reasoning problems in multiple fields. ARB presents a more challenging test than prior benchmarks, featuring problems in mathematics, physics, biology, chemistry, and law. As a subset of ARB, we introduce a challenging set of math and physics problems which require advanced symbolic reasoning and domain knowledge. We evaluate recent models such as GPT-4 and Claude on ARB and demonstrate that current models score well below 50% on more demanding tasks. In order to improve both automatic and assisted evaluation capabilities, we introduce a rubric-based evaluation approach, allowing GPT-4 to score its own intermediate reasoning steps. Further, we conduct a human evaluation of the symbolic subset of ARB, finding promising agreement between annotators and GPT-4 rubric evaluation scores.

{{</citation>}}


### (62/103) How Can Large Language Models Help Humans in Design and Manufacturing? (Liane Makatura et al., 2023)

{{<citation>}}

Liane Makatura, Michael Foshey, Bohan Wang, Felix HähnLein, Pingchuan Ma, Bolei Deng, Megan Tjandrasuwita, Andrew Spielberg, Crystal Elaine Owens, Peter Yichen Chen, Allan Zhao, Amy Zhu, Wil J Norton, Edward Gu, Joshua Jacob, Yifei Li, Adriana Schulz, Wojciech Matusik. (2023)  
**How Can Large Language Models Help Humans in Design and Manufacturing?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2307.14377v1)  

---


**ABSTRACT**  
The advancement of Large Language Models (LLMs), including GPT-4, provides exciting new opportunities for generative design. We investigate the application of this tool across the entire design and manufacturing workflow. Specifically, we scrutinize the utility of LLMs in tasks such as: converting a text-based prompt into a design specification, transforming a design into manufacturing instructions, producing a design space and design variations, computing the performance of a design, and searching for designs predicated on performance. Through a series of examples, we highlight both the benefits and the limitations of the current LLMs. By exposing these limitations, we aspire to catalyze the continued improvement and progression of these models.

{{</citation>}}


### (63/103) Contributions to the Improvement of Question Answering Systems in the Biomedical Domain (Mourad Sarrouti, 2023)

{{<citation>}}

Mourad Sarrouti. (2023)  
**Contributions to the Improvement of Question Answering Systems in the Biomedical Domain**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.13631v1)  

---


**ABSTRACT**  
This thesis work falls within the framework of question answering (QA) in the biomedical domain where several specific challenges are addressed, such as specialized lexicons and terminologies, the types of treated questions, and the characteristics of targeted documents. We are particularly interested in studying and improving methods that aim at finding accurate and short answers to biomedical natural language questions from a large scale of biomedical textual documents in English. QA aims at providing inquirers with direct, short and precise answers to their natural language questions. In this Ph.D. thesis, we propose four contributions to improve the performance of QA in the biomedical domain. In our first contribution, we propose a machine learning-based method for question type classification to determine the types of given questions which enable to a biomedical QA system to use the appropriate answer extraction method. We also propose an another machine learning-based method to assign one or more topics (e.g., pharmacological, test, treatment, etc.) to given questions in order to determine the semantic types of the expected answers which are very useful in generating specific answer retrieval strategies. In the second contribution, we first propose a document retrieval method to retrieve a set of relevant documents that are likely to contain the answers to biomedical questions from the MEDLINE database. We then present a passage retrieval method to retrieve a set of relevant passages to questions. In the third contribution, we propose specific answer extraction methods to generate both exact and ideal answers. Finally, in the fourth contribution, we develop a fully automated semantic biomedical QA system called SemBioNLQA which is able to deal with a variety of natural language questions and to generate appropriate answers by providing both exact and ideal answers.

{{</citation>}}


### (64/103) GPT-3 Models are Few-Shot Financial Reasoners (Raul Salles de Padua et al., 2023)

{{<citation>}}

Raul Salles de Padua, Imran Qureshi, Mustafa U. Karakaplan. (2023)  
**GPT-3 Models are Few-Shot Financial Reasoners**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Few-Shot, Financial, GPT, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2307.13617v2)  

---


**ABSTRACT**  
Financial analysis is an important tool for evaluating company performance. Practitioners work to answer financial questions to make profitable investment decisions, and use advanced quantitative analyses to do so. As a result, Financial Question Answering (QA) is a question answering task that requires deep reasoning about numbers. Furthermore, it is unknown how well pre-trained language models can reason in the financial domain. The current state-of-the-art requires a retriever to collect relevant facts about the financial question from the text and a generator to produce a valid financial program and a final answer. However, recently large language models like GPT-3 have achieved state-of-the-art performance on wide variety of tasks with just a few shot examples. We run several experiments with GPT-3 and find that a separate retrieval model and logic engine continue to be essential components to achieving SOTA performance in this task, particularly due to the precise nature of financial questions and the complex information stored in financial documents. With this understanding, our refined prompt-engineering approach on GPT-3 achieves near SOTA accuracy without any fine-tuning.

{{</citation>}}


### (65/103) XDLM: Cross-lingual Diffusion Language Model for Machine Translation (Linyao Chen et al., 2023)

{{<citation>}}

Linyao Chen, Aosong Feng, Boming Yang, Zihui Li. (2023)  
**XDLM: Cross-lingual Diffusion Language Model for Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Machine Translation, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2307.13560v1)  

---


**ABSTRACT**  
Recently, diffusion models have excelled in image generation tasks and have also been applied to neural language processing (NLP) for controllable text generation. However, the application of diffusion models in a cross-lingual setting is less unexplored. Additionally, while pretraining with diffusion models has been studied within a single language, the potential of cross-lingual pretraining remains understudied. To address these gaps, we propose XDLM, a novel Cross-lingual diffusion model for machine translation, consisting of pretraining and fine-tuning stages. In the pretraining stage, we propose TLDM, a new training objective for mastering the mapping between different languages; in the fine-tuning stage, we build up the translation system based on the pretrained model. We evaluate the result on several machine translation benchmarks and outperformed both diffusion and Transformer baselines.

{{</citation>}}


### (66/103) FacTool: Factuality Detection in Generative AI -- A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios (I-Chun Chern et al., 2023)

{{<citation>}}

I-Chun Chern, Steffi Chern, Shiqi Chen, Weizhe Yuan, Kehua Feng, Chunting Zhou, Junxian He, Graham Neubig, Pengfei Liu. (2023)  
**FacTool: Factuality Detection in Generative AI -- A Tool Augmented Framework for Multi-Task and Multi-Domain Scenarios**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, Generative AI, NLP, QA  
[Paper Link](http://arxiv.org/abs/2307.13528v2)  

---


**ABSTRACT**  
The emergence of generative pre-trained models has facilitated the synthesis of high-quality text, but it has also posed challenges in identifying factual errors in the generated text. In particular: (1) A wider range of tasks now face an increasing risk of containing factual errors when handled by generative models. (2) Generated texts tend to be lengthy and lack a clearly defined granularity for individual facts. (3) There is a scarcity of explicit evidence available during the process of fact checking. With the above challenges in mind, in this paper, we propose FacTool, a task and domain agnostic framework for detecting factual errors of texts generated by large language models (e.g., ChatGPT). Experiments on four different tasks (knowledge-based QA, code generation, mathematical reasoning, and scientific literature review) show the efficacy of the proposed method. We release the code of FacTool associated with ChatGPT plugin interface at https://github.com/GAIR-NLP/factool .

{{</citation>}}


### (67/103) Zshot: An Open-source Framework for Zero-Shot Named Entity Recognition and Relation Extraction (Gabriele Picco et al., 2023)

{{<citation>}}

Gabriele Picco, Marcos Martínez Galindo, Alberto Purpura, Leopold Fuchs, Vanessa López, Hoang Thanh Lam. (2023)  
**Zshot: An Open-source Framework for Zero-Shot Named Entity Recognition and Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP, Named Entity Recognition, Relation Extraction, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2307.13497v1)  

---


**ABSTRACT**  
The Zero-Shot Learning (ZSL) task pertains to the identification of entities or relations in texts that were not seen during training. ZSL has emerged as a critical research area due to the scarcity of labeled data in specific domains, and its applications have grown significantly in recent years. With the advent of large pretrained language models, several novel methods have been proposed, resulting in substantial improvements in ZSL performance. There is a growing demand, both in the research community and industry, for a comprehensive ZSL framework that facilitates the development and accessibility of the latest methods and pretrained models.In this study, we propose a novel ZSL framework called Zshot that aims to address the aforementioned challenges. Our primary objective is to provide a platform that allows researchers to compare different state-of-the-art ZSL methods with standard benchmark datasets. Additionally, we have designed our framework to support the industry with readily available APIs for production under the standard SpaCy NLP pipeline. Our API is extendible and evaluable, moreover, we include numerous enhancements such as boosting the accuracy with pipeline ensembling and visualization utilities available as a SpaCy extension.

{{</citation>}}


### (68/103) Holistic Exploration on Universal Decompositional Semantic Parsing: Architecture, Data Augmentation, and LLM Paradigm (Hexuan Deng et al., 2023)

{{<citation>}}

Hexuan Deng, Xin Zhang, Meishan Zhang, Xuebo Liu, Min Zhang. (2023)  
**Holistic Exploration on Universal Decompositional Semantic Parsing: Architecture, Data Augmentation, and LLM Paradigm**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2307.13424v1)  

---


**ABSTRACT**  
In this paper, we conduct a holistic exploration of the Universal Decompositional Semantic (UDS) Parsing. We first introduce a cascade model for UDS parsing that decomposes the complex parsing task into semantically appropriate subtasks. Our approach outperforms the prior models, while significantly reducing inference time. We also incorporate syntactic information and further optimized the architecture. Besides, different ways for data augmentation are explored, which further improve the UDS Parsing. Lastly, we conduct experiments to investigate the efficacy of ChatGPT in handling the UDS task, revealing that it excels in attribute parsing but struggles in relation parsing, and using ChatGPT for data augmentation yields suboptimal results. Our code is available at https://github.com/hexuandeng/HExp4UDS.

{{</citation>}}


### (69/103) Towards Resolving Word Ambiguity with Word Embeddings (Matthias Thurnbauer et al., 2023)

{{<citation>}}

Matthias Thurnbauer, Johannes Reisinger, Christoph Goller, Andreas Fischer. (2023)  
**Towards Resolving Word Ambiguity with Word Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, Transformer, Word Embedding  
[Paper Link](http://arxiv.org/abs/2307.13417v1)  

---


**ABSTRACT**  
Ambiguity is ubiquitous in natural language. Resolving ambiguous meanings is especially important in information retrieval tasks. While word embeddings carry semantic information, they fail to handle ambiguity well. Transformer models have been shown to handle word ambiguity for complex queries, but they cannot be used to identify ambiguous words, e.g. for a 1-word query. Furthermore, training these models is costly in terms of time, hardware resources, and training data, prohibiting their use in specialized environments with sensitive data. Word embeddings can be trained using moderate hardware resources. This paper shows that applying DBSCAN clustering to the latent space can identify ambiguous words and evaluate their level of ambiguity. An automatic DBSCAN parameter selection leads to high-quality clusters, which are semantically coherent and correspond well to the perceived meanings of a given word.

{{</citation>}}


### (70/103) Towards Bridging the Digital Language Divide (Gábor Bella et al., 2023)

{{<citation>}}

Gábor Bella, Paula Helm, Gertraud Koch, Fausto Giunchiglia. (2023)  
**Towards Bridging the Digital Language Divide**  

---
Primary Category: cs.CL  
Categories: I-2-7; K-4-2, cs-AI, cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13405v1)  

---


**ABSTRACT**  
It is a well-known fact that current AI-based language technology -- language models, machine translation systems, multilingual dictionaries and corpora -- focuses on the world's 2-3% most widely spoken languages. Recent research efforts have attempted to expand the coverage of AI technology to `under-resourced languages.' The goal of our paper is to bring attention to a phenomenon that we call linguistic bias: multilingual language processing systems often exhibit a hardwired, yet usually involuntary and hidden representational preference towards certain languages. Linguistic bias is manifested in uneven per-language performance even in the case of similar test conditions. We show that biased technology is often the result of research and development methodologies that do not do justice to the complexity of the languages being represented, and that can even become ethically problematic as they disregard valuable aspects of diversity as well as the needs of the language communities themselves. As our attempt at building diversity-aware language resources, we present a new initiative that aims at reducing linguistic bias through both technological design and methodology, based on an eye-level collaboration with local communities.

{{</citation>}}


### (71/103) Empower Your Model with Longer and Better Context Comprehension (Yifei Gao et al., 2023)

{{<citation>}}

Yifei Gao, Lei Wang, Jun Fang, Longhua Hu, Jun Cheng. (2023)  
**Empower Your Model with Longer and Better Context Comprehension**  

---
Primary Category: cs.CL  
Categories: 68T07, 68T50, cs-AI, cs-CL, cs.CL  
Keywords: AI, Attention, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.13365v2)  

---


**ABSTRACT**  
Recently, with the emergence of numerous Large Language Models (LLMs), the implementation of AI has entered a new era. Irrespective of these models' own capacity and structure, there is a growing demand for LLMs to possess enhanced comprehension of longer and more complex contexts with relatively smaller sizes. Models often encounter an upper limit when processing sequences of sentences that extend beyond their comprehension capacity and result in off-topic or even chaotic responses. While several recent works attempt to address this issue in various ways, they rarely focus on "why models are unable to compensate or strengthen their capabilities on their own". In this paper, we thoroughly investigate the nature of information transfer within LLMs and propose a novel technique called Attention Transition. This technique empowers models to achieve longer and better context comprehension with minimal additional training or impact on generation fluency. Our experiments are conducted on the challenging XSum dataset using LLaMa-7b model with context token length ranging from 800 to 1900. Results demonstrate that we achieve substantial improvements compared with the original generation results evaluated by GPT4.

{{</citation>}}


### (72/103) Analyzing Chain-of-Thought Prompting in Large Language Models via Gradient-based Feature Attributions (Skyler Wu et al., 2023)

{{<citation>}}

Skyler Wu, Eric Meng Shen, Charumathi Badrinath, Jiaqi Ma, Himabindu Lakkaraju. (2023)  
**Analyzing Chain-of-Thought Prompting in Large Language Models via Gradient-based Feature Attributions**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2307.13339v1)  

---


**ABSTRACT**  
Chain-of-thought (CoT) prompting has been shown to empirically improve the accuracy of large language models (LLMs) on various question answering tasks. While understanding why CoT prompting is effective is crucial to ensuring that this phenomenon is a consequence of desired model behavior, little work has addressed this; nonetheless, such an understanding is a critical prerequisite for responsible model deployment. We address this question by leveraging gradient-based feature attribution methods which produce saliency scores that capture the influence of input tokens on model output. Specifically, we probe several open-source LLMs to investigate whether CoT prompting affects the relative importances they assign to particular input tokens. Our results indicate that while CoT prompting does not increase the magnitude of saliency scores attributed to semantically relevant tokens in the prompt compared to standard few-shot prompting, it increases the robustness of saliency scores to question perturbations and variations in model output.

{{</citation>}}


## cs.CE (1)



### (73/103) Uncertainty Quantification in the Road-level Traffic Risk Prediction by Spatial-Temporal Zero-Inflated Negative Binomial Graph Neural Network(STZINB-GNN) (Xiaowei Gao et al., 2023)

{{<citation>}}

Xiaowei Gao, James Haworth, Dingyi Zhuang, Huanfa Chen, Xinke Jiang. (2023)  
**Uncertainty Quantification in the Road-level Traffic Risk Prediction by Spatial-Temporal Zero-Inflated Negative Binomial Graph Neural Network(STZINB-GNN)**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2307.13816v1)  

---


**ABSTRACT**  
Urban road-based risk prediction is a crucial yet challenging aspect of research in transportation safety. While most existing studies emphasize accurate prediction, they often overlook the importance of model uncertainty. In this paper, we introduce a novel Spatial-Temporal Zero-Inflated Negative Binomial Graph Neural Network (STZINB-GNN) for road-level traffic risk prediction, with a focus on uncertainty quantification. Our case study, conducted in the Lambeth borough of London, UK, demonstrates the superior performance of our approach in comparison to existing methods. Although the negative binomial distribution may not be the most suitable choice for handling real, non-binary risk levels, our work lays a solid foundation for future research exploring alternative distribution models or techniques. Ultimately, the STZINB-GNN contributes to enhanced transportation safety and data-driven decision-making in urban planning by providing a more accurate and reliable framework for road-level traffic risk prediction and uncertainty quantification.

{{</citation>}}


## stat.ML (3)



### (74/103) How to Scale Your EMA (Dan Busbridge et al., 2023)

{{<citation>}}

Dan Busbridge, Jason Ramapuram, Pierre Ablin, Tatiana Likhomanenko, Eeshan Gunesh Dhekane, Xavier Suau, Russ Webb. (2023)  
**How to Scale Your EMA**  

---
Primary Category: stat.ML  
Categories: cs-AI, cs-LG, stat-ML, stat.ML  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2307.13813v2)  

---


**ABSTRACT**  
Preserving training dynamics across batch sizes is an important tool for practical machine learning as it enables the trade-off between batch size and wall-clock time. This trade-off is typically enabled by a scaling rule, for example, in stochastic gradient descent, one should scale the learning rate linearly with the batch size. Another important tool for practical machine learning is the model Exponential Moving Average (EMA), which is a model copy that does not receive gradient information, but instead follows its target model with some momentum. This model EMA can improve the robustness and generalization properties of supervised learning, stabilize pseudo-labeling, and provide a learning signal for Self-Supervised Learning (SSL). Prior works have treated the model EMA separately from optimization, leading to different training dynamics across batch sizes and lower model performance. In this work, we provide a scaling rule for optimization in the presence of model EMAs and demonstrate its validity across a range of architectures, optimizers, and data modalities. We also show the rule's validity where the model EMA contributes to the optimization of the target model, enabling us to train EMA-based pseudo-labeling and SSL methods at small and large batch sizes. For SSL, we enable training of BYOL up to batch size 24,576 without sacrificing performance, optimally a 6$\times$ wall-clock time reduction.

{{</citation>}}


### (75/103) Implicitly Normalized Explicitly Regularized Density Estimation (Mark Kozdoba et al., 2023)

{{<citation>}}

Mark Kozdoba, Binyamin Perets, Shie Mannor. (2023)  
**Implicitly Normalized Explicitly Regularized Density Estimation**  

---
Primary Category: stat.ML  
Categories: cs-AI, cs-LG, stat-ML, stat.ML  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2307.13763v1)  

---


**ABSTRACT**  
We propose a new approach to non-parametric density estimation, that is based on regularizing a Sobolev norm of the density. This method is provably different from Kernel Density Estimation, and makes the bias of the model clear and interpretable. While there is no closed analytic form for the associated kernel, we show that one can approximate it using sampling. The optimization problem needed to determine the density is non-convex, and standard gradient methods do not perform well. However, we show that with an appropriate initialization and using natural gradients, one can obtain well performing solutions. Finally, while the approach provides unnormalized densities, which prevents the use of log-likelihood for cross validation, we show that one can instead adapt Fisher Divergence based Score Matching methods for this task. We evaluate the resulting method on the comprehensive recent Anomaly Detection benchmark suite, ADBench, and find that it ranks second best, among more than 15 algorithms.

{{</citation>}}


### (76/103) AI and ethics in insurance: a new solution to mitigate proxy discrimination in risk modeling (Marguerite Sauce et al., 2023)

{{<citation>}}

Marguerite Sauce, Antoine Chancel, Antoine Ly. (2023)  
**AI and ethics in insurance: a new solution to mitigate proxy discrimination in risk modeling**  

---
Primary Category: stat.ML  
Categories: cs-CY, cs-LG, stat-ME, stat-ML, stat.ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13616v1)  

---


**ABSTRACT**  
The development of Machine Learning is experiencing growing interest from the general public, and in recent years there have been numerous press articles questioning its objectivity: racism, sexism, \dots Driven by the growing attention of regulators on the ethical use of data in insurance, the actuarial community must rethink pricing and risk selection practices for fairer insurance. Equity is a philosophy concept that has many different definitions in every jurisdiction that influence each other without currently reaching consensus. In Europe, the Charter of Fundamental Rights defines guidelines on discrimination, and the use of sensitive personal data in algorithms is regulated. If the simple removal of the protected variables prevents any so-called `direct' discrimination, models are still able to `indirectly' discriminate between individuals thanks to latent interactions between variables, which bring better performance (and therefore a better quantification of risk, segmentation of prices, and so on). After introducing the key concepts related to discrimination, we illustrate the complexity of quantifying them. We then propose an innovative method, not yet met in the literature, to reduce the risks of indirect discrimination thanks to mathematical concepts of linear algebra. This technique is illustrated in a concrete case of risk selection in life insurance, demonstrating its simplicity of use and its promising performance.

{{</citation>}}


## cs.GT (1)



### (77/103) Strategic Play By Resource-Bounded Agents in Security Games (Xinming Liu et al., 2023)

{{<citation>}}

Xinming Liu, Joseph Y. Halpern. (2023)  
**Strategic Play By Resource-Bounded Agents in Security Games**  

---
Primary Category: cs.GT  
Categories: cs-GT, cs.GT  
Keywords: Amazon, Security  
[Paper Link](http://arxiv.org/abs/2307.13778v1)  

---


**ABSTRACT**  
Many studies have shown that humans are "predictably irrational": they do not act in a fully rational way, but their deviations from rational behavior are quite systematic. Our goal is to see the extent to which we can explain and justify these deviations as the outcome of rational but resource-bounded agents doing as well as they can, given their limitations. We focus on the well-studied ranger-poacher game, where rangers are trying to protect a number of sites from poaching. We capture the computational limitations by modeling the poacher and the ranger as probabilistic finite automata (PFAs). We show that, with sufficiently large memory, PFAs learn to play the Nash equilibrium (NE) strategies of the game and achieve the NE utility. However, if we restrict the memory, we get more "human-like" behaviors, such as probability matching (i.e., visiting sites in proportion to the probability of a rhino being there), and avoiding sites where there was a bad outcome (e.g., the poacher was caught by the ranger), that we also observed in experiments conducted on Amazon Mechanical Turk. Interestingly, we find that adding human-like behaviors such as probability matching and overweighting significant events (like getting caught) actually improves performance, showing that this seemingly irrational behavior can be quite rational.

{{</citation>}}


## cs.SI (1)



### (78/103) The Dynamics of Political Narratives During the Russian Invasion of Ukraine (Ahana Biswas et al., 2023)

{{<citation>}}

Ahana Biswas, Tim Niven, Yu-Ru Lin. (2023)  
**The Dynamics of Political Narratives During the Russian Invasion of Ukraine**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2307.13753v1)  

---


**ABSTRACT**  
The Russian invasion of Ukraine has elicited a diverse array of responses from nations around the globe. During a global conflict, polarized narratives are spread on social media to sway public opinion. We examine the dynamics of the political narratives surrounding the Russia-Ukraine war during the first two months of the Russian invasion of Ukraine (RU) using the Chinese Twitter space as a case study. Since the beginning of the RU, pro-Chinese-state and anti-Chinese-state users have spread divisive opinions, rumors, and conspiracy theories. We investigate how the pro- and anti-state camps contributed to the evolution of RU-related narratives, as well as how a few influential accounts drove the narrative evolution. We identify pro-state and anti-state actors on Twitter using network analysis and text-based classifiers, and we leverage text analysis, along with the users' social interactions (e.g., retweeting), to extract narrative coordination and evolution. We find evidence that both pro-state and anti-state camps spread propaganda narratives about RU. Our analysis illuminates how actors coordinate to advance particular viewpoints or act against one another in the context of global conflict.

{{</citation>}}


## cs.DC (1)



### (79/103) Smartpick: Workload Prediction for Serverless-enabled Scalable Data Analytics Systems (Anshuman Das Mohapatra et al., 2023)

{{<citation>}}

Anshuman Das Mohapatra, Kwangsung Oh. (2023)  
**Smartpick: Workload Prediction for Serverless-enabled Scalable Data Analytics Systems**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: AWS, Amazon, Google  
[Paper Link](http://arxiv.org/abs/2307.13677v1)  

---


**ABSTRACT**  
Many data analytic systems have adopted a newly emerging compute resource, serverless (SL), to handle data analytics queries in a timely and cost-efficient manner, i.e., serverless data analytics. While these systems can start processing queries quickly thanks to the agility and scalability of SL, they may encounter performance- and cost-bottlenecks based on workloads due to SL's worse performance and more expensive cost than traditional compute resources, e.g., virtual machine (VM). In this project, we introduce Smartpick, a SL-enabled scalable data analytics system that exploits SL and VM together to realize composite benefits, i.e., agility from SL and better performance with reduced cost from VM. Smartpick uses a machine learning prediction scheme, decision-tree based Random Forest with Bayesian Optimizer, to determine SL and VM configurations, i.e., how many SL and VM instances for queries, that meet cost-performance goals. Smartpick offers a knob for applications to allow them to explore a richer cost-performance tradeoff space opened by exploiting SL and VM together. To maximize the benefits of SL, Smartpick supports a simple but strong mechanism, called relay-instances. Smartpick also supports event-driven prediction model retraining to deal with workload dynamics. A Smartpick prototype was implemented on Spark and deployed on live test-beds, Amazon AWS and Google Cloud Platform. Evaluation results indicate 97.05% and 83.49% prediction accuracies respectively with up to 50% cost reduction as opposed to the baselines. The results also confirm that Smartpick allows data analytics applications to navigate the richer cost-performance tradeoff space efficiently and to handle workload dynamics effectively and automatically.

{{</citation>}}


## cs.CY (2)



### (80/103) Towards an AI Accountability Policy (Przemyslaw Grabowicz et al., 2023)

{{<citation>}}

Przemyslaw Grabowicz, Nicholas Perello, Yair Zick. (2023)  
**Towards an AI Accountability Policy**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-LG, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13658v1)  

---


**ABSTRACT**  
This white paper is a response to the "AI Accountability Policy Request for Comments" by the National Telecommunications and Information Administration of the United States. The question numbers for which comments were requested are provided in superscripts at the end of key sentences answering the respective questions. The white paper offers a set of interconnected recommendations for an AI accountability policy.

{{</citation>}}


### (81/103) Diversity and Language Technology: How Techno-Linguistic Bias Can Cause Epistemic Injustice (Paula Helm et al., 2023)

{{<citation>}}

Paula Helm, Gábor Bella, Gertraud Koch, Fausto Giunchiglia. (2023)  
**Diversity and Language Technology: How Techno-Linguistic Bias Can Cause Epistemic Injustice**  

---
Primary Category: cs.CY  
Categories: cs-CL, cs-CY, cs.CY  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2307.13714v1)  

---


**ABSTRACT**  
It is well known that AI-based language technology -- large language models, machine translation systems, multilingual dictionaries, and corpora -- is currently limited to 2 to 3 percent of the world's most widely spoken and/or financially and politically best supported languages. In response, recent research efforts have sought to extend the reach of AI technology to ``underserved languages.'' In this paper, we show that many of these attempts produce flawed solutions that adhere to a hard-wired representational preference for certain languages, which we call techno-linguistic bias. Techno-linguistic bias is distinct from the well-established phenomenon of linguistic bias as it does not concern the languages represented but rather the design of the technologies. As we show through the paper, techno-linguistic bias can result in systems that can only express concepts that are part of the language and culture of dominant powers, unable to correctly represent concepts from other communities. We argue that at the root of this problem lies a systematic tendency of technology developer communities to apply a simplistic understanding of diversity which does not do justice to the more profound differences that languages, and ultimately the communities that speak them, embody. Drawing on the concept of epistemic injustice, we point to the broader sociopolitical consequences of the bias we identify and show how it can lead not only to a disregard for valuable aspects of diversity but also to an under-representation of the needs and diverse worldviews of marginalized language communities.

{{</citation>}}


## cs.IR (5)



### (82/103) Mitigating Mainstream Bias in Recommendation via Cost-sensitive Learning (Roger Zhe Li et al., 2023)

{{<citation>}}

Roger Zhe Li, Julián Urbano, Alan Hanjalic. (2023)  
**Mitigating Mainstream Bias in Recommendation via Cost-sensitive Learning**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2307.13632v1)  

---


**ABSTRACT**  
Mainstream bias, where some users receive poor recommendations because their preferences are uncommon or simply because they are less active, is an important aspect to consider regarding fairness in recommender systems. Existing methods to mitigate mainstream bias do not explicitly model the importance of these non-mainstream users or, when they do, it is in a way that is not necessarily compatible with the data and recommendation model at hand. In contrast, we use the recommendation utility as a more generic and implicit proxy to quantify mainstreamness, and propose a simple user-weighting approach to incorporate it into the training process while taking the cost of potential recommendation errors into account. We provide extensive experimental results showing that quantifying mainstreamness via utility is better able at identifying non-mainstream users, and that they are indeed better served when training the model in a cost-sensitive way. This is achieved with negligible or no loss in overall recommendation accuracy, meaning that the models learn a better balance across users. In addition, we show that research of this kind, which evaluates recommendation quality at the individual user level, may not be reliable if not using enough interactions when assessing model performance.

{{</citation>}}


### (83/103) Gaussian Graph with Prototypical Contrastive Learning in E-Commerce Bundle Recommendation (Zhao-Yang Liu et al., 2023)

{{<citation>}}

Zhao-Yang Liu, Liucheng Sun, Chenwei Weng, Qijin Chen, Chengfu Huo. (2023)  
**Gaussian Graph with Prototypical Contrastive Learning in E-Commerce Bundle Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs-LG, cs.IR  
Keywords: Contrastive Learning, GNN  
[Paper Link](http://arxiv.org/abs/2307.13468v1)  

---


**ABSTRACT**  
Bundle recommendation aims to provide a bundle of items to satisfy the user preference on e-commerce platform. Existing successful solutions are based on the contrastive graph learning paradigm where graph neural networks (GNNs) are employed to learn representations from user-level and bundle-level graph views with a contrastive learning module to enhance the cooperative association between different views. Nevertheless, they ignore the uncertainty issue which has a significant impact in real bundle recommendation scenarios due to the lack of discriminative information caused by highly sparsity or diversity. We further suggest that their instancewise contrastive learning fails to distinguish the semantically similar negatives (i.e., sampling bias issue), resulting in performance degradation. In this paper, we propose a novel Gaussian Graph with Prototypical Contrastive Learning (GPCL) framework to overcome these challenges. In particular, GPCL embeds each user/bundle/item as a Gaussian distribution rather than a fixed vector. We further design a prototypical contrastive learning module to capture the contextual information and mitigate the sampling bias issue. Extensive experiments demonstrate that benefiting from the proposed components, we achieve new state-of-the-art performance compared to previous methods on several public datasets. Moreover, GPCL has been deployed on real-world e-commerce platform and achieved substantial improvements.

{{</citation>}}


### (84/103) Comprehensive Review on Semantic Information Retrieval and Ontology Engineering (Sumit Sharma et al., 2023)

{{<citation>}}

Sumit Sharma, Sarika Jain. (2023)  
**Comprehensive Review on Semantic Information Retrieval and Ontology Engineering**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2307.13427v1)  

---


**ABSTRACT**  
Situation awareness is a crucial cognitive skill that enables individuals to perceive, comprehend, and project the current state of their environment accurately. It involves being conscious of relevant information, understanding its meaning, and using that understanding to make well-informed decisions. Awareness systems often need to integrate new knowledge and adapt to changing environments. Ontology reasoning facilitates knowledge integration and evolution, allowing for seamless updates and expansions of the ontology. With the consideration of above, we are providing a quick review on semantic information retrieval and ontology engineering to understand the emerging challenges and future research. In the review we have found that the ontology reasoning addresses the limitations of traditional systems by providing a formal, flexible, and scalable framework for knowledge representation, reasoning, and inference.

{{</citation>}}


### (85/103) An End-to-End Workflow using Topic Segmentation and Text Summarisation Methods for Improved Podcast Comprehension (Andrew Aquilina et al., 2023)

{{<citation>}}

Andrew Aquilina, Sean Diacono, Panagiotis Papapetrou, Maria Movin. (2023)  
**An End-to-End Workflow using Topic Segmentation and Text Summarisation Methods for Improved Podcast Comprehension**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: T5  
[Paper Link](http://arxiv.org/abs/2307.13394v1)  

---


**ABSTRACT**  
The consumption of podcast media has been increasing rapidly. Due to the lengthy nature of podcast episodes, users often carefully select which ones to listen to. Although episode descriptions aid users by providing a summary of the entire podcast, they do not provide a topic-by-topic breakdown. This study explores the combined application of topic segmentation and text summarisation methods to investigate how podcast episode comprehension can be improved. We have sampled 10 episodes from Spotify's English-Language Podcast Dataset and employed TextTiling and TextSplit to segment them. Moreover, three text summarisation models, namely T5, BART, and Pegasus, were applied to provide a very short title for each segment. The segmentation part was evaluated using our annotated sample with the $P_k$ and WindowDiff ($WD$) metrics. A survey was also rolled out ($N=25$) to assess the quality of the generated summaries. The TextSplit algorithm achieved the lowest mean for both evaluation metrics ($\bar{P_k}=0.41$ and $\bar{WD}=0.41$), while the T5 model produced the best summaries, achieving a relevancy score only $8\%$ less to the one achieved by the human-written titles.

{{</citation>}}


### (86/103) An Intent Taxonomy of Legal Case Retrieval (Yunqiu Shao et al., 2023)

{{<citation>}}

Yunqiu Shao, Haitao Li, Yueyue Wu, Yiqun Liu, Qingyao Ai, Jiaxin Mao, Yixiao Ma, Shaoping Ma. (2023)  
**An Intent Taxonomy of Legal Case Retrieval**  

---
Primary Category: cs.IR  
Categories: cs-CL, cs-IR, cs.IR  
Keywords: Information Retrieval, Legal  
[Paper Link](http://arxiv.org/abs/2307.13298v1)  

---


**ABSTRACT**  
Legal case retrieval is a special Information Retrieval~(IR) task focusing on legal case documents. Depending on the downstream tasks of the retrieved case documents, users' information needs in legal case retrieval could be significantly different from those in Web search and traditional ad-hoc retrieval tasks. While there are several studies that retrieve legal cases based on text similarity, the underlying search intents of legal retrieval users, as shown in this paper, are more complicated than that yet mostly unexplored. To this end, we present a novel hierarchical intent taxonomy of legal case retrieval. It consists of five intent types categorized by three criteria, i.e., search for Particular Case(s), Characterization, Penalty, Procedure, and Interest. The taxonomy was constructed transparently and evaluated extensively through interviews, editorial user studies, and query log analysis. Through a laboratory user study, we reveal significant differences in user behavior and satisfaction under different search intents in legal case retrieval. Furthermore, we apply the proposed taxonomy to various downstream legal retrieval tasks, e.g., result ranking and satisfaction prediction, and demonstrate its effectiveness. Our work provides important insights into the understanding of user intents in legal case retrieval and potentially leads to better retrieval techniques in the legal domain, such as intent-aware ranking strategies and evaluation methodologies.

{{</citation>}}


## cs.HC (3)



### (87/103) The Importance of Distrust in AI (Tobias M. Peters et al., 2023)

{{<citation>}}

Tobias M. Peters, Roel W. Visser. (2023)  
**The Importance of Distrust in AI**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13601v1)  

---


**ABSTRACT**  
In recent years the use of Artificial Intelligence (AI) has become increasingly prevalent in a growing number of fields. As AI systems are being adopted in more high-stakes areas such as medicine and finance, ensuring that they are trustworthy is of increasing importance. A concern that is prominently addressed by the development and application of explainability methods, which are purported to increase trust from its users and wider society. While an increase in trust may be desirable, an analysis of literature from different research fields shows that an exclusive focus on increasing trust may not be warranted. Something which is well exemplified by the recent development in AI chatbots, which while highly coherent tend to make up facts. In this contribution, we investigate the concepts of trust, trustworthiness, and user reliance.   In order to foster appropriate reliance on AI we need to prevent both disuse of these systems as well as overtrust. From our analysis of research on interpersonal trust, trust in automation, and trust in (X)AI, we identify the potential merit of the distinction between trust and distrust (in AI). We propose that alongside trust a healthy amount of distrust is of additional value for mitigating disuse and overtrust. We argue that by considering and evaluating both trust and distrust, we can ensure that users can rely appropriately on trustworthy AI, which can both be useful as well as fallible.

{{</citation>}}


### (88/103) The Impact of Imperfect XAI on Human-AI Decision-Making (Katelyn Morrison et al., 2023)

{{<citation>}}

Katelyn Morrison, Philipp Spitzer, Violet Turri, Michelle Feng, Niklas Kühl, Adam Perer. (2023)  
**The Impact of Imperfect XAI on Human-AI Decision-Making**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13566v1)  

---


**ABSTRACT**  
Explainability techniques are rapidly being developed to improve human-AI decision-making across various cooperative work settings. Consequently, previous research has evaluated how decision-makers collaborate with imperfect AI by investigating appropriate reliance and task performance with the aim of designing more human-centered computer-supported collaborative tools. Several human-centered explainable AI (XAI) techniques have been proposed in hopes of improving decision-makers' collaboration with AI; however, these techniques are grounded in findings from previous studies that primarily focus on the impact of incorrect AI advice. Few studies acknowledge the possibility for the explanations to be incorrect even if the AI advice is correct. Thus, it is crucial to understand how imperfect XAI affects human-AI decision-making. In this work, we contribute a robust, mixed-methods user study with 136 participants to evaluate how incorrect explanations influence humans' decision-making behavior in a bird species identification task taking into account their level of expertise and an explanation's level of assertiveness. Our findings reveal the influence of imperfect XAI and humans' level of expertise on their reliance on AI and human-AI team performance. We also discuss how explanations can deceive decision-makers during human-AI collaboration. Hence, we shed light on the impacts of imperfect XAI in the field of computer-supported cooperative work and provide guidelines for designers of human-AI collaboration systems.

{{</citation>}}


### (89/103) Digital Emotion Regulation on Social Media (Akriti Verma et al., 2023)

{{<citation>}}

Akriti Verma, Shama Islam, Valeh Moghaddam, Adnan Anwar. (2023)  
**Digital Emotion Regulation on Social Media**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2307.13187v1)  

---


**ABSTRACT**  
Emotion regulation is the process of consciously altering one's affective state, that is the underlying emotional state such as happiness, confidence, guilt, anger etc. The ability to effectively regulate emotions is necessary for functioning efficiently in everyday life. Today, the pervasiveness of digital technology is being purposefully employed to modify our affective states, a process known as digital emotion regulation. Understanding digital emotion regulation can help support the rise of ethical technology design, development, and deployment. This article presents an overview of digital emotion regulation in social media applications, as well as a synthesis of recent research on emotion regulation interventions for social media. We share our findings from analysing state-of-the-art literature on how different social media applications are utilised at different stages in the process of emotion regulation.

{{</citation>}}


## cs.CR (1)



### (90/103) Node Injection Link Stealing Attack (Oualid Zari et al., 2023)

{{<citation>}}

Oualid Zari, Javier Parra-Arnau, Ayşe Ünsal, Melek Önen. (2023)  
**Node Injection Link Stealing Attack**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2307.13548v1)  

---


**ABSTRACT**  
In this paper, we present a stealthy and effective attack that exposes privacy vulnerabilities in Graph Neural Networks (GNNs) by inferring private links within graph-structured data. Focusing on the inductive setting where new nodes join the graph and an API is used to query predictions, we investigate the potential leakage of private edge information. We also propose methods to preserve privacy while maintaining model utility. Our attack demonstrates superior performance in inferring the links compared to the state of the art. Furthermore, we examine the application of differential privacy (DP) mechanisms to mitigate the impact of our proposed attack, we analyze the trade-off between privacy preservation and model utility. Our work highlights the privacy vulnerabilities inherent in GNNs, underscoring the importance of developing robust privacy-preserving mechanisms for their application.

{{</citation>}}


## q-fin.PM (1)



### (91/103) Deep Reinforcement Learning for Robust Goal-Based Wealth Management (Tessa Bauman et al., 2023)

{{<citation>}}

Tessa Bauman, Bruno Gašperov, Stjepan Begušić, Zvonko Kostanjčar. (2023)  
**Deep Reinforcement Learning for Robust Goal-Based Wealth Management**  

---
Primary Category: q-fin.PM  
Categories: cs-LG, q-fin-PM, q-fin.PM  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.13501v1)  

---


**ABSTRACT**  
Goal-based investing is an approach to wealth management that prioritizes achieving specific financial goals. It is naturally formulated as a sequential decision-making problem as it requires choosing the appropriate investment until a goal is achieved. Consequently, reinforcement learning, a machine learning technique appropriate for sequential decision-making, offers a promising path for optimizing these investment strategies. In this paper, a novel approach for robust goal-based wealth management based on deep reinforcement learning is proposed. The experimental results indicate its superiority over several goal-based wealth management benchmarks on both simulated and historical market data.

{{</citation>}}


## cs.NI (1)



### (92/103) On viewing SpaceX Starlink through the Social Media Lens (Aryan Taneja et al., 2023)

{{<citation>}}

Aryan Taneja, Debopam Bhattacherjee, Saikat Guha, Venkata N. Padmanabhan. (2023)  
**On viewing SpaceX Starlink through the Social Media Lens**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2307.13441v1)  

---


**ABSTRACT**  
Multiple low-Earth orbit satellite constellations, aimed at beaming broadband connectivity from space, are currently under active deployment. While such space-based Internet is set to augment, globally, today's terrestrial connectivity, and has managed to generate significant hype, it has been largely difficult for the community to measure, quantify, or understand the nuances of these offerings in the absence of a global measurement infrastructure -- the research community has mostly resorted to simulators, emulators, and limited measurements till now. In this paper, we identify an opportunity to use the social media `lens' to complement such measurements and mine user-centric insights on the evolving ecosystem at scale.

{{</citation>}}


## eess.SY (1)



### (93/103) Communication-Efficient Orchestrations for URLLC Service via Hierarchical Reinforcement Learning (Wei Shi et al., 2023)

{{<citation>}}

Wei Shi, Milad Ganjalizadeh, Hossein Shokri Ghadikolaei, Marina Petrova. (2023)  
**Communication-Efficient Orchestrations for URLLC Service via Hierarchical Reinforcement Learning**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2307.13415v1)  

---


**ABSTRACT**  
Ultra-reliable low latency communications (URLLC) service is envisioned to enable use cases with strict reliability and latency requirements in 5G. One approach for enabling URLLC services is to leverage Reinforcement Learning (RL) to efficiently allocate wireless resources. However, with conventional RL methods, the decision variables (though being deployed at various network layers) are typically optimized in the same control loop, leading to significant practical limitations on the control loop's delay as well as excessive signaling and energy consumption. In this paper, we propose a multi-agent Hierarchical RL (HRL) framework that enables the implementation of multi-level policies with different control loop timescales. Agents with faster control loops are deployed closer to the base station, while the ones with slower control loops are at the edge or closer to the core network providing high-level guidelines for low-level actions. On a use case from the prior art, with our HRL framework, we optimized the maximum number of retransmissions and transmission power of industrial devices. Our extensive simulation results on the factory automation scenario show that the HRL framework achieves better performance as the baseline single-agent RL method, with significantly less overhead of signal transmissions and delay compared to the one-agent RL methods.

{{</citation>}}


## cs.SE (1)



### (94/103) Predicting Code Coverage without Execution (Michele Tufano et al., 2023)

{{<citation>}}

Michele Tufano, Shubham Chandel, Anisha Agarwal, Neel Sundaresan, Colin Clement. (2023)  
**Predicting Code Coverage without Execution**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI, BARD, GPT, GPT-3.5, GPT-4, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2307.13383v1)  

---


**ABSTRACT**  
Code coverage is a widely used metric for quantifying the extent to which program elements, such as statements or branches, are executed during testing. Calculating code coverage is resource-intensive, requiring code building and execution with additional overhead for the instrumentation. Furthermore, computing coverage of any snippet of code requires the whole program context. Using Machine Learning to amortize this expensive process could lower the cost of code coverage by requiring only the source code context, and the task of code coverage prediction can be a novel benchmark for judging the ability of models to understand code. We propose a novel benchmark task called Code Coverage Prediction for Large Language Models (LLMs). We formalize this task to evaluate the capability of LLMs in understanding code execution by determining which lines of a method are executed by a given test case and inputs. We curate and release a dataset we call COVERAGEEVAL by executing tests and code from the HumanEval dataset and collecting code coverage information. We report the performance of four state-of-the-art LLMs used for code-related tasks, including OpenAI's GPT-4 and GPT-3.5-Turbo, Google's BARD, and Anthropic's Claude, on the Code Coverage Prediction task. Finally, we argue that code coverage as a metric and pre-training data source are valuable for overall LLM performance on software engineering tasks.

{{</citation>}}


## cs.DL (1)



### (95/103) Embedding Models for Supervised Automatic Extraction and Classification of Named Entities in Scientific Acknowledgements (Nina Smirnova et al., 2023)

{{<citation>}}

Nina Smirnova, Philipp Mayr. (2023)  
**Embedding Models for Supervised Automatic Extraction and Classification of Named Entities in Scientific Acknowledgements**  

---
Primary Category: cs.DL  
Categories: J-4; J-5; I-5-1; H-3-3; I-2-7, cs-CL, cs-DL, cs-IR, cs.DL  
Keywords: AI, Embedding, NER, NLP  
[Paper Link](http://arxiv.org/abs/2307.13377v1)  

---


**ABSTRACT**  
Acknowledgments in scientific papers may give an insight into aspects of the scientific community, such as reward systems, collaboration patterns, and hidden research trends. The aim of the paper is to evaluate the performance of different embedding models for the task of automatic extraction and classification of acknowledged entities from the acknowledgment text in scientific papers. We trained and implemented a named entity recognition (NER) task using the Flair NLP framework. The training was conducted using three default Flair NER models with four differently-sized corpora and different versions of the Flair NLP framework. The Flair Embeddings model trained on the medium corpus with the latest FLAIR version showed the best accuracy of 0.79. Expanding the size of a training corpus from very small to medium size massively increased the accuracy of all training algorithms, but further expansion of the training corpus did not bring further improvement. Moreover, the performance of the model slightly deteriorated. Our model is able to recognize six entity types: funding agency, grant number, individuals, university, corporation, and miscellaneous. The model works more precisely for some entity types than for others; thus, individuals and grant numbers showed a very good F1-Score over 0.9. Most of the previous works on acknowledgment analysis were limited by the manual evaluation of data and therefore by the amount of processed data. This model can be applied for the comprehensive analysis of acknowledgment texts and may potentially make a great contribution to the field of automated acknowledgment analysis.

{{</citation>}}


## q-bio.QM (1)



### (96/103) Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers (Hadi Abdine et al., 2023)

{{<citation>}}

Hadi Abdine, Michail Chatzianastasis, Costas Bouyioukos, Michalis Vazirgiannis. (2023)  
**Prot2Text: Multimodal Protein's Function Generation with GNNs and Transformers**  

---
Primary Category: q-bio.QM  
Categories: cs-CL, cs-LG, q-bio-QM, q-bio.QM  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2307.14367v1)  

---


**ABSTRACT**  
The complex nature of big biological systems pushed some scientists to classify its understanding under the inconceivable missions. Different leveled challenges complicated this task, one of is the prediction of a protein's function. In recent years, significant progress has been made in this field through the development of various machine learning approaches. However, most existing methods formulate the task as a multi-classification problem, i.e assigning predefined labels to proteins. In this work, we propose a novel approach, \textbf{Prot2Text}, which predicts a protein function's in a free text style, moving beyond the conventional binary or categorical classifications. By combining Graph Neural Networks(GNNs) and Large Language Models(LLMs), in an encoder-decoder framework, our model effectively integrates diverse data types including proteins' sequences, structures, and textual annotations. This multimodal approach allows for a holistic representation of proteins' functions, enabling the generation of detailed and accurate descriptions. To evaluate our model, we extracted a multimodal protein dataset from SwissProt, and demonstrate empirically the effectiveness of Prot2Text. These results highlight the transformative impact of multimodal models, specifically the fusion of GNNs and LLMs, empowering researchers with powerful tools for more accurate prediction of proteins' functions. The code, the models and a demo will be publicly released.

{{</citation>}}


## cs.DS (1)



### (97/103) Federated Heavy Hitter Recovery under Linear Sketching (Adria Gascon et al., 2023)

{{<citation>}}

Adria Gascon, Peter Kairouz, Ziteng Sun, Ananda Theertha Suresh. (2023)  
**Federated Heavy Hitter Recovery under Linear Sketching**  

---
Primary Category: cs.DS  
Categories: cs-CR, cs-DS, cs-IT, cs.DS, math-IT  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2307.13347v1)  

---


**ABSTRACT**  
Motivated by real-life deployments of multi-round federated analytics with secure aggregation, we investigate the fundamental communication-accuracy tradeoffs of the heavy hitter discovery and approximate (open-domain) histogram problems under a linear sketching constraint. We propose efficient algorithms based on local subsampling and invertible bloom look-up tables (IBLTs). We also show that our algorithms are information-theoretically optimal for a broad class of interactive schemes. The results show that the linear sketching constraint does increase the communication cost for both tasks by introducing an extra linear dependence on the number of users in a round. Moreover, our results also establish a separation between the communication cost for heavy hitter discovery and approximate histogram in the multi-round setting. The dependence on the number of rounds $R$ is at most logarithmic for heavy hitter discovery whereas that of approximate histogram is $\Theta(\sqrt{R})$. We also empirically demonstrate our findings.

{{</citation>}}


## eess.AS (1)



### (98/103) On-Device Speaker Anonymization of Acoustic Embeddings for ASR based onFlexible Location Gradient Reversal Layer (Md Asif Jalal et al., 2023)

{{<citation>}}

Md Asif Jalal, Pablo Peso Parada, Jisi Zhang, Karthikeyan Saravanan, Mete Ozay, Myoungji Han, Jung In Lee, Seokyeong Jung. (2023)  
**On-Device Speaker Anonymization of Acoustic Embeddings for ASR based onFlexible Location Gradient Reversal Layer**  

---
Primary Category: eess.AS  
Categories: cs-CR, cs-SD, eess-AS, eess.AS  
Keywords: AI, Embedding, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2307.13343v1)  

---


**ABSTRACT**  
Smart devices serviced by large-scale AI models necessitates user data transfer to the cloud for inference. For speech applications, this means transferring private user information, e.g., speaker identity. Our paper proposes a privacy-enhancing framework that targets speaker identity anonymization while preserving speech recognition accuracy for our downstream task~-~Automatic Speech Recognition (ASR). The proposed framework attaches flexible gradient reversal based speaker adversarial layers to target layers within an ASR model, where speaker adversarial training anonymizes acoustic embeddings generated by the targeted layers to remove speaker identity. We propose on-device deployment by execution of initial layers of the ASR model, and transmitting anonymized embeddings to the cloud, where the rest of the model is executed while preserving privacy. Experimental results show that our method efficiently reduces speaker recognition relative accuracy by 33%, and improves ASR performance by achieving 6.2% relative Word Error Rate (WER) reduction.

{{</citation>}}


## cs.DB (1)



### (99/103) A Generic Framework for Hidden Markov Models on Biomedical Data (Richard Fechner et al., 2023)

{{<citation>}}

Richard Fechner, Jens Dörpinghaus, Robert Rockenfeller, Jennifer Faber. (2023)  
**A Generic Framework for Hidden Markov Models on Biomedical Data**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB, q-bio-QM  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2307.13288v1)  

---


**ABSTRACT**  
Background: Biomedical data are usually collections of longitudinal data assessed at certain points in time. Clinical observations assess the presences and severity of symptoms, which are the basis for description and modeling of disease progression. Deciphering potential underlying unknowns solely from the distinct observation would substantially improve the understanding of pathological cascades. Hidden Markov Models (HMMs) have been successfully applied to the processing of possibly noisy continuous signals. The aim was to improve the application HMMs to multivariate time-series of categorically distributed data. Here, we used HHMs to study prediction of the loss of free walking ability as one major clinical deterioration in the most common autosomal dominantly inherited ataxia disorder worldwide. We used HHMs to investigate the prediction of loss of the ability to walk freely, representing a major clinical deterioration in the most common autosomal-dominant inherited ataxia disorder worldwide.   Results: We present a prediction pipeline which processes data paired with a configuration file, enabling to construct, validate and query a fully parameterized HMM-based model. In particular, we provide a theoretical and practical framework for multivariate time-series inference based on HMMs that includes constructing multiple HMMs, each to predict a particular observable variable. Our analysis is done on random data, but also on biomedical data based on Spinocerebellar ataxia type 3 disease.   Conclusions: HHMs are a promising approach to study biomedical data that naturally are represented as multivariate time-series. Our implementation of a HHMs framework is publicly available and can easily be adapted for further applications.

{{</citation>}}


## cs.SD (1)



### (100/103) Audio-aware Query-enhanced Transformer for Audio-Visual Segmentation (Jinxiang Liu et al., 2023)

{{<citation>}}

Jinxiang Liu, Chen Ju, Chaofan Ma, Yanfeng Wang, Yu Wang, Ya Zhang. (2023)  
**Audio-aware Query-enhanced Transformer for Audio-Visual Segmentation**  

---
Primary Category: cs.SD  
Categories: cs-CV, cs-LG, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2307.13236v1)  

---


**ABSTRACT**  
The goal of the audio-visual segmentation (AVS) task is to segment the sounding objects in the video frames using audio cues. However, current fusion-based methods have the performance limitations due to the small receptive field of convolution and inadequate fusion of audio-visual features. To overcome these issues, we propose a novel \textbf{Au}dio-aware query-enhanced \textbf{TR}ansformer (AuTR) to tackle the task. Unlike existing methods, our approach introduces a multimodal transformer architecture that enables deep fusion and aggregation of audio-visual features. Furthermore, we devise an audio-aware query-enhanced transformer decoder that explicitly helps the model focus on the segmentation of the pinpointed sounding objects based on audio signals, while disregarding silent yet salient objects. Experimental results show that our method outperforms previous methods and demonstrates better generalization ability in multi-sound and open-set scenarios.

{{</citation>}}


## cs.MM (1)



### (101/103) Text-oriented Modality Reinforcement Network for Multimodal Sentiment Analysis from Unaligned Multimodal Sequences (Yuxuan Lei et al., 2023)

{{<citation>}}

Yuxuan Lei, Dingkang Yang, Mingcheng Li, Shunli Wang, Jiawei Chen, Lihua Zhang. (2023)  
**Text-oriented Modality Reinforcement Network for Multimodal Sentiment Analysis from Unaligned Multimodal Sequences**  

---
Primary Category: cs.MM  
Categories: cs-MM, cs.MM  
Keywords: Attention, Self-Attention, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2307.13205v1)  

---


**ABSTRACT**  
Multimodal Sentiment Analysis (MSA) aims to mine sentiment information from text, visual, and acoustic modalities. Previous works have focused on representation learning and feature fusion strategies. However, most of these efforts ignored the disparity in the semantic richness of different modalities and treated each modality in the same manner. That may lead to strong modalities being neglected and weak modalities being overvalued. Motivated by these observations, we propose a Text-oriented Modality Reinforcement Network (TMRN), which focuses on the dominance of the text modality in MSA. More specifically, we design a Text-Centered Cross-modal Attention (TCCA) module to make full interaction for text/acoustic and text/visual pairs, and a Text-Gated Self-Attention (TGSA) module to guide the self-reinforcement of the other two modalities. Furthermore, we present an adaptive fusion mechanism to decide the proportion of different modalities involved in the fusion process. Finally, we combine the feature matrices into vectors to get the final representation for the downstream tasks. Experimental results show that our TMRN outperforms the state-of-the-art methods on two MSA benchmarks.

{{</citation>}}


## cs.RO (1)



### (102/103) GraspGPT: Leveraging Semantic Knowledge from a Large Language Model for Task-Oriented Grasping (Chao Tang et al., 2023)

{{<citation>}}

Chao Tang, Dehao Huang, Wenqi Ge, Weiyu Liu, Hong Zhang. (2023)  
**GraspGPT: Leveraging Semantic Knowledge from a Large Language Model for Task-Oriented Grasping**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2307.13204v1)  

---


**ABSTRACT**  
Task-oriented grasping (TOG) refers to the problem of predicting grasps on an object that enable subsequent manipulation tasks. To model the complex relationships between objects, tasks, and grasps, existing methods incorporate semantic knowledge as priors into TOG pipelines. However, the existing semantic knowledge is typically constructed based on closed-world concept sets, restraining the generalization to novel concepts out of the pre-defined sets. To address this issue, we propose GraspGPT, a large language model (LLM) based TOG framework that leverages the open-end semantic knowledge from an LLM to achieve zero-shot generalization to novel concepts. We conduct experiments on Language Augmented TaskGrasp (LA-TaskGrasp) dataset and demonstrate that GraspGPT outperforms existing TOG methods on different held-out settings when generalizing to novel concepts out of the training set. The effectiveness of GraspGPT is further validated in real-robot experiments. Our code, data, appendix, and video are publicly available at https://sites.google.com/view/graspgpt/.

{{</citation>}}


## eess.IV (1)



### (103/103) An Investigation into Glomeruli Detection in Kidney H&E and PAS Images using YOLO (Kimia Hemmatirad et al., 2023)

{{<citation>}}

Kimia Hemmatirad, Morteza Babaie, Jeffrey Hodgin, Liron Pantanowitz, H. R. Tizhoosh. (2023)  
**An Investigation into Glomeruli Detection in Kidney H&E and PAS Images using YOLO**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2307.13199v1)  

---


**ABSTRACT**  
Context: Analyzing digital pathology images is necessary to draw diagnostic conclusions by investigating tissue patterns and cellular morphology. However, manual evaluation can be time-consuming, expensive, and prone to inter- and intra-observer variability. Objective: To assist pathologists using computerized solutions, automated tissue structure detection and segmentation must be proposed. Furthermore, generating pixel-level object annotations for histopathology images is expensive and time-consuming. As a result, detection models with bounding box labels may be a feasible solution. Design: This paper studies. YOLO-v4 (You-Only-Look-Once), a real-time object detector for microscopic images. YOLO uses a single neural network to predict several bounding boxes and class probabilities for objects of interest. YOLO can enhance detection performance by training on whole slide images. YOLO-v4 has been used in this paper. for glomeruli detection in human kidney images. Multiple experiments have been designed and conducted based on different training data of two public datasets and a private dataset from the University of Michigan for fine-tuning the model. The model was tested on the private dataset from the University of Michigan, serving as an external validation of two different stains, namely hematoxylin and eosin (H&E) and periodic acid-Schiff (PAS). Results: Average specificity and sensitivity for all experiments, and comparison of existing segmentation methods on the same datasets are discussed. Conclusions: Automated glomeruli detection in human kidney images is possible using modern AI models. The design and validation for different stains still depends on variability of public multi-stain datasets.

{{</citation>}}
