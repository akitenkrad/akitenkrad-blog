---
draft: false
title: "arXiv @ 2023.09.20"
date: 2023-09-20
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.20"
    identifier: arxiv_20230920
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (31)](#cscv-31)
- [cs.CL (22)](#cscl-22)
- [cs.HC (7)](#cshc-7)
- [cs.LG (23)](#cslg-23)
- [cs.NI (2)](#csni-2)
- [cs.RO (18)](#csro-18)
- [cs.CR (7)](#cscr-7)
- [physics.med-ph (1)](#physicsmed-ph-1)
- [physics.ins-det (1)](#physicsins-det-1)
- [eess.SY (1)](#eesssy-1)
- [cs.AI (14)](#csai-14)
- [eess.AS (6)](#eessas-6)
- [cs.CY (4)](#cscy-4)
- [cs.DS (1)](#csds-1)
- [math.NA (1)](#mathna-1)
- [stat.ML (1)](#statml-1)
- [cs.SE (2)](#csse-2)
- [cs.SD (4)](#cssd-4)
- [cs.IR (2)](#csir-2)
- [cs.DL (2)](#csdl-2)
- [cs.IT (1)](#csit-1)
- [eess.SP (1)](#eesssp-1)
- [cs.NE (1)](#csne-1)
- [cs.AR (1)](#csar-1)

## cs.CV (31)



### (1/154) Image-Text Pre-Training for Logo Recognition (Mark Hubenthal et al., 2023)

{{<citation>}}

Mark Hubenthal, Suren Kumar. (2023)  
**Image-Text Pre-Training for Logo Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.10206v1)  

---


**ABSTRACT**  
Open-set logo recognition is commonly solved by first detecting possible logo regions and then matching the detected parts against an ever-evolving dataset of cropped logo images. The matching model, a metric learning problem, is especially challenging for logo recognition due to the mixture of text and symbols in logos. We propose two novel contributions to improve the matching model's performance: (a) using image-text paired samples for pre-training, and (b) an improved metric learning loss function. A standard paradigm of fine-tuning ImageNet pre-trained models fails to discover the text sensitivity necessary to solve the matching problem effectively. This work demonstrates the importance of pre-training on image-text pairs, which significantly improves the performance of a visual embedder trained for the logo retrieval task, especially for more text-dominant classes. We construct a composite public logo dataset combining LogoDet3K, OpenLogo, and FlickrLogos-47 deemed OpenLogoDet3K47. We show that the same vision backbone pre-trained on image-text data, when fine-tuned on OpenLogoDet3K47, achieves $98.6\%$ recall@1, significantly improving performance over pre-training on Imagenet1K ($97.6\%$). We generalize the ProxyNCA++ loss function to propose ProxyNCAHN++ which incorporates class-specific hard negative images. The proposed method sets new state-of-the-art on five public logo datasets considered, with a $3.5\%$ zero-shot recall@1 improvement on LogoDet3K test, $4\%$ on OpenLogo, $6.5\%$ on FlickrLogos-47, $6.2\%$ on Logos In The Wild, and $0.6\%$ on BelgaLogo.

{{</citation>}}


### (2/154) GEDepth: Ground Embedding for Monocular Depth Estimation (Xiaodong Yang et al., 2023)

{{<citation>}}

Xiaodong Yang, Zhuang Ma, Zhiyu Ji, Zhe Ren. (2023)  
**GEDepth: Ground Embedding for Monocular Depth Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.09975v1)  

---


**ABSTRACT**  
Monocular depth estimation is an ill-posed problem as the same 2D image can be projected from infinite 3D scenes. Although the leading algorithms in this field have reported significant improvement, they are essentially geared to the particular compound of pictorial observations and camera parameters (i.e., intrinsics and extrinsics), strongly limiting their generalizability in real-world scenarios. To cope with this challenge, this paper proposes a novel ground embedding module to decouple camera parameters from pictorial cues, thus promoting the generalization capability. Given camera parameters, the proposed module generates the ground depth, which is stacked with the input image and referenced in the final depth prediction. A ground attention is designed in the module to optimally combine ground depth with residual depth. Our ground embedding is highly flexible and lightweight, leading to a plug-in module that is amenable to be integrated into various depth estimation networks. Experiments reveal that our approach achieves the state-of-the-art results on popular benchmarks, and more importantly, renders significant generalization improvement on a wide range of cross-domain tests.

{{</citation>}}


### (3/154) An Empirical Study of Scaling Instruct-Tuned Large Multimodal Models (Yadong Lu et al., 2023)

{{<citation>}}

Yadong Lu, Chunyuan Li, Haotian Liu, Jianwei Yang, Jianfeng Gao, Yelong Shen. (2023)  
**An Empirical Study of Scaling Instruct-Tuned Large Multimodal Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2309.09958v1)  

---


**ABSTRACT**  
Visual instruction tuning has recently shown encouraging progress with open-source large multimodal models (LMM) such as LLaVA and MiniGPT-4. However, most existing studies of open-source LMM are performed using models with 13B parameters or smaller. In this paper we present an empirical study of scaling LLaVA up to 33B and 65B/70B, and share our findings from our explorations in image resolution, data mixing and parameter-efficient training methods such as LoRA/QLoRA. These are evaluated by their impact on the multi-modal and language capabilities when completing real-world tasks in the wild.   We find that scaling LMM consistently enhances model performance and improves language capabilities, and performance of LoRA/QLoRA tuning of LMM are comparable to the performance of full-model fine-tuning. Additionally, the study highlights the importance of higher image resolutions and mixing multimodal-language data to improve LMM performance, and visual instruction tuning can sometimes improve LMM's pure language capability. We hope that this study makes state-of-the-art LMM research at a larger scale more accessible, thus helping establish stronger baselines for future research. Code and checkpoints will be made public.

{{</citation>}}


### (4/154) Hierarchical Attention and Graph Neural Networks: Toward Drift-Free Pose Estimation (Kathia Melbouci et al., 2023)

{{<citation>}}

Kathia Melbouci, Fawzi Nashashibi. (2023)  
**Hierarchical Attention and Graph Neural Networks: Toward Drift-Free Pose Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Attention, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.09934v1)  

---


**ABSTRACT**  
The most commonly used method for addressing 3D geometric registration is the iterative closet-point algorithm, this approach is incremental and prone to drift over multiple consecutive frames. The Common strategy to address the drift is the pose graph optimization subsequent to frame-to-frame registration, incorporating a loop closure process that identifies previously visited places. In this paper, we explore a framework that replaces traditional geometric registration and pose graph optimization with a learned model utilizing hierarchical attention mechanisms and graph neural networks. We propose a strategy to condense the data flow, preserving essential information required for the precise estimation of rigid poses. Our results, derived from tests on the KITTI Odometry dataset, demonstrate a significant improvement in pose estimation accuracy. This improvement is especially notable in determining rotational components when compared with results obtained through conventional multi-way registration via pose graph optimization. The code will be made available upon completion of the review process.

{{</citation>}}


### (5/154) Hyperbolic vs Euclidean Embeddings in Few-Shot Learning: Two Sides of the Same Coin (Gabriel Moreira et al., 2023)

{{<citation>}}

Gabriel Moreira, Manuel Marques, João Paulo Costeira, Alexander Hauptmann. (2023)  
**Hyperbolic vs Euclidean Embeddings in Few-Shot Learning: Two Sides of the Same Coin**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Embedding, Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.10013v1)  

---


**ABSTRACT**  
Recent research in representation learning has shown that hierarchical data lends itself to low-dimensional and highly informative representations in hyperbolic space. However, even if hyperbolic embeddings have gathered attention in image recognition, their optimization is prone to numerical hurdles. Further, it remains unclear which applications stand to benefit the most from the implicit bias imposed by hyperbolicity, when compared to traditional Euclidean features. In this paper, we focus on prototypical hyperbolic neural networks. In particular, the tendency of hyperbolic embeddings to converge to the boundary of the Poincar\'e ball in high dimensions and the effect this has on few-shot classification. We show that the best few-shot results are attained for hyperbolic embeddings at a common hyperbolic radius. In contrast to prior benchmark results, we demonstrate that better performance can be achieved by a fixed-radius encoder equipped with the Euclidean metric, regardless of the embedding dimension.

{{</citation>}}


### (6/154) R2GenGPT: Radiology Report Generation with Frozen LLMs (Zhanyu Wang et al., 2023)

{{<citation>}}

Zhanyu Wang, Lingqiao Liu, Lei Wang, Luping Zhou. (2023)  
**R2GenGPT: Radiology Report Generation with Frozen LLMs**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.09812v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have consistently showcased remarkable generalization capabilities when applied to various language tasks. Nonetheless, harnessing the full potential of LLMs for Radiology Report Generation (R2Gen) still presents a challenge, stemming from the inherent disparity in modality between LLMs and the R2Gen task. To bridge this gap effectively, we propose R2GenGPT, which is a novel solution that aligns visual features with the word embedding space of LLMs using an efficient visual alignment module. This innovative approach empowers the previously static LLM to seamlessly integrate and process image information, marking a step forward in optimizing R2Gen performance. R2GenGPT offers the following benefits. First, it attains state-of-the-art (SOTA) performance by training only the lightweight visual alignment module while freezing all the parameters of LLM. Second, it exhibits high training efficiency, as it requires the training of an exceptionally minimal number of parameters while achieving rapid convergence. By employing delta tuning, our model only trains 5M parameters (which constitute just 0.07\% of the total parameter count) to achieve performance close to the SOTA levels. Our code is available at https://github.com/wang-zhanyu/R2GenGPT.

{{</citation>}}


### (7/154) VisualProg Distiller: Learning to Fine-tune Non-differentiable Visual Programming Frameworks (Wentao Wan et al., 2023)

{{<citation>}}

Wentao Wan, Zeqing Wang, Nan Kang, Keze Wang, Zhiyu Shen, Liang Lin. (2023)  
**VisualProg Distiller: Learning to Fine-tune Non-differentiable Visual Programming Frameworks**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model, QA  
[Paper Link](http://arxiv.org/abs/2309.09809v1)  

---


**ABSTRACT**  
As an interpretable and universal neuro-symbolic paradigm based on Large Language Models, visual programming (VisualProg) can execute compositional visual tasks without training, but its performance is markedly inferior compared to task-specific supervised learning models. To increase its practicality, the performance of VisualProg on specific tasks needs to be improved. However, the non-differentiability of VisualProg limits the possibility of employing the fine-tuning strategy on specific tasks to achieve further improvements. In our analysis, we discovered that significant performance issues in VisualProg's execution originated from errors made by the sub-modules at corresponding visual sub-task steps. To address this, we propose ``VisualProg Distiller", a method of supplementing and distilling process knowledge to optimize the performance of each VisualProg sub-module on decoupled visual sub-tasks, thus enhancing the overall task performance. Specifically, we choose an end-to-end model that is well-performed on the given task as the teacher and further distill the knowledge of the teacher into the invoked visual sub-modules step-by-step based on the execution flow of the VisualProg-generated programs. In this way, our method is capable of facilitating the fine-tuning of the non-differentiable VisualProg frameworks effectively. Extensive and comprehensive experimental evaluations demonstrate that our method can achieve a substantial performance improvement of VisualProg, and outperforms all the compared state-of-the-art methods by large margins. Furthermore, to provide valuable process supervision for the GQA task, we construct a large-scale dataset by utilizing the distillation process of our method.

{{</citation>}}


### (8/154) Drawing the Same Bounding Box Twice? Coping Noisy Annotations in Object Detection with Repeated Labels (David Tschirschwitz et al., 2023)

{{<citation>}}

David Tschirschwitz, Christian Benz, Morris Florek, Henrik Norderhus, Benno Stein, Volker Rodehorst. (2023)  
**Drawing the Same Bounding Box Twice? Coping Noisy Annotations in Object Detection with Repeated Labels**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.09742v1)  

---


**ABSTRACT**  
The reliability of supervised machine learning systems depends on the accuracy and availability of ground truth labels. However, the process of human annotation, being prone to error, introduces the potential for noisy labels, which can impede the practicality of these systems. While training with noisy labels is a significant consideration, the reliability of test data is also crucial to ascertain the dependability of the results. A common approach to addressing this issue is repeated labeling, where multiple annotators label the same example, and their labels are combined to provide a better estimate of the true label. In this paper, we propose a novel localization algorithm that adapts well-established ground truth estimation methods for object detection and instance segmentation tasks. The key innovation of our method lies in its ability to transform combined localization and classification tasks into classification-only problems, thus enabling the application of techniques such as Expectation-Maximization (EM) or Majority Voting (MJV). Although our main focus is the aggregation of unique ground truth for test data, our algorithm also shows superior performance during training on the TexBiG dataset, surpassing both noisy label training and label aggregation using Weighted Boxes Fusion (WBF). Our experiments indicate that the benefits of repeated labels emerge under specific dataset and annotation configurations. The key factors appear to be (1) dataset complexity, the (2) annotator consistency, and (3) the given annotation budget constraints.

{{</citation>}}


### (9/154) Moving Object Detection and Tracking with 4D Radar Point Cloud (Zhijun Pan et al., 2023)

{{<citation>}}

Zhijun Pan, Fangqiang Ding, Hantao Zhong, Chris Xiaoxuan Lu. (2023)  
**Moving Object Detection and Tracking with 4D Radar Point Cloud**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.09737v1)  

---


**ABSTRACT**  
Mobile autonomy relies on the precise perception of dynamic environments. Robustly tracking moving objects in 3D world thus plays a pivotal role for applications like trajectory prediction, obstacle avoidance, and path planning. While most current methods utilize LiDARs or cameras for Multiple Object Tracking (MOT), the capabilities of 4D imaging radars remain largely unexplored. Recognizing the challenges posed by radar noise and point sparsity in 4D radar data, we introduce RaTrack, an innovative solution tailored for radar-based tracking. Bypassing the typical reliance on specific object types and 3D bounding boxes, our method focuses on motion segmentation and clustering, enriched by a motion estimation module. Evaluated on the View-of-Delft dataset, RaTrack showcases superior tracking precision of moving objects, largely surpassing the performance of the state of the art.

{{</citation>}}


### (10/154) CATR: Combinatorial-Dependence Audio-Queried Transformer for Audio-Visual Video Segmentation (Kexin Li et al., 2023)

{{<citation>}}

Kexin Li, Zongxin Yang, Lei Chen, Yi Yang, Jun Xiao. (2023)  
**CATR: Combinatorial-Dependence Audio-Queried Transformer for Audio-Visual Video Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.09709v2)  

---


**ABSTRACT**  
Audio-visual video segmentation~(AVVS) aims to generate pixel-level maps of sound-producing objects within image frames and ensure the maps faithfully adhere to the given audio, such as identifying and segmenting a singing person in a video. However, existing methods exhibit two limitations: 1) they address video temporal features and audio-visual interactive features separately, disregarding the inherent spatial-temporal dependence of combined audio and video, and 2) they inadequately introduce audio constraints and object-level information during the decoding stage, resulting in segmentation outcomes that fail to comply with audio directives. To tackle these issues, we propose a decoupled audio-video transformer that combines audio and video features from their respective temporal and spatial dimensions, capturing their combined dependence. To optimize memory consumption, we design a block, which, when stacked, enables capturing audio-visual fine-grained combinatorial-dependence in a memory-efficient manner. Additionally, we introduce audio-constrained queries during the decoding phase. These queries contain rich object-level information, ensuring the decoded mask adheres to the sounds. Experimental results confirm our approach's effectiveness, with our framework achieving a new SOTA performance on all three datasets using two backbones. The code is available at \url{https://github.com/aspirinone/CATR.github.io}

{{</citation>}}


### (11/154) DGM-DR: Domain Generalization with Mutual Information Regularized Diabetic Retinopathy Classification (Aleksandr Matsun et al., 2023)

{{<citation>}}

Aleksandr Matsun, Dana O. Mohamed, Sharon Chokuwa, Muhammad Ridzuan, Mohammad Yaqub. (2023)  
**DGM-DR: Domain Generalization with Mutual Information Regularized Diabetic Retinopathy Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09670v1)  

---


**ABSTRACT**  
The domain shift between training and testing data presents a significant challenge for training generalizable deep learning models. As a consequence, the performance of models trained with the independent and identically distributed (i.i.d) assumption deteriorates when deployed in the real world. This problem is exacerbated in the medical imaging context due to variations in data acquisition across clinical centers, medical apparatus, and patients. Domain generalization (DG) aims to address this problem by learning a model that generalizes well to any unseen target domain. Many domain generalization techniques were unsuccessful in learning domain-invariant representations due to the large domain shift. Furthermore, multiple tasks in medical imaging are not yet extensively studied in existing literature when it comes to DG point of view. In this paper, we introduce a DG method that re-establishes the model objective function as a maximization of mutual information with a large pretrained model to the medical imaging field. We re-visit the problem of DG in Diabetic Retinopathy (DR) classification to establish a clear benchmark with a correct model selection strategy and to achieve robust domain-invariant representation for an improved generalization. Moreover, we conduct extensive experiments on public datasets to show that our proposed method consistently outperforms the previous state-of-the-art by a margin of 5.25% in average accuracy and a lower standard deviation. Source code available at https://github.com/BioMedIA-MBZUAI/DGM-DR

{{</citation>}}


### (12/154) DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation (Bowen Yin et al., 2023)

{{<citation>}}

Bowen Yin, Xuying Zhang, Zhongyu Li, Li Liu, Ming-Ming Cheng, Qibin Hou. (2023)  
**DFormer: Rethinking RGBD Representation Learning for Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Representation Learning, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.09668v1)  

---


**ABSTRACT**  
We present DFormer, a novel RGB-D pretraining framework to learn transferable representations for RGB-D segmentation tasks. DFormer has two new key innovations: 1) Unlike previous works that aim to encode RGB features,DFormer comprises a sequence of RGB-D blocks, which are tailored for encoding both RGB and depth information through a novel building block design; 2) We pre-train the backbone using image-depth pairs from ImageNet-1K, and thus the DFormer is endowed with the capacity to encode RGB-D representations. It avoids the mismatched encoding of the 3D geometry relationships in depth maps by RGB pre-trained backbones, which widely lies in existing methods but has not been resolved. We fine-tune the pre-trained DFormer on two popular RGB-D tasks, i.e., RGB-D semantic segmentation and RGB-D salient object detection, with a lightweight decoder head. Experimental results show that our DFormer achieves new state-of-the-art performance on these two tasks with less than half of the computational cost of the current best methods on two RGB-D segmentation datasets and five RGB-D saliency datasets. Our code is available at: https://github.com/VCIP-RGBD/DFormer.

{{</citation>}}


### (13/154) Unified Frequency-Assisted Transformer Framework for Detecting and Grounding Multi-Modal Manipulation (Huan Liu et al., 2023)

{{<citation>}}

Huan Liu, Zichang Tan, Qiang Chen, Yunchao Wei, Yao Zhao, Jingdong Wang. (2023)  
**Unified Frequency-Assisted Transformer Framework for Detecting and Grounding Multi-Modal Manipulation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.09667v1)  

---


**ABSTRACT**  
Detecting and grounding multi-modal media manipulation (DGM^4) has become increasingly crucial due to the widespread dissemination of face forgery and text misinformation. In this paper, we present the Unified Frequency-Assisted transFormer framework, named UFAFormer, to address the DGM^4 problem. Unlike previous state-of-the-art methods that solely focus on the image (RGB) domain to describe visual forgery features, we additionally introduce the frequency domain as a complementary viewpoint. By leveraging the discrete wavelet transform, we decompose images into several frequency sub-bands, capturing rich face forgery artifacts. Then, our proposed frequency encoder, incorporating intra-band and inter-band self-attentions, explicitly aggregates forgery features within and across diverse sub-bands. Moreover, to address the semantic conflicts between image and frequency domains, the forgery-aware mutual module is developed to further enable the effective interaction of disparate image and frequency features, resulting in aligned and comprehensive visual forgery representations. Finally, based on visual and textual forgery features, we propose a unified decoder that comprises two symmetric cross-modal interaction modules responsible for gathering modality-specific forgery information, along with a fusing interaction module for aggregation of both modalities. The proposed unified decoder formulates our UFAFormer as a unified framework, ultimately simplifying the overall architecture and facilitating the optimization process. Experimental results on the DGM^4 dataset, containing several perturbations, demonstrate the superior performance of our framework compared to previous methods, setting a new benchmark in the field.

{{</citation>}}


### (14/154) HiT: Building Mapping with Hierarchical Transformers (Mingming Zhang et al., 2023)

{{<citation>}}

Mingming Zhang, Qingjie Liu, Yunhong Wang. (2023)  
**HiT: Building Mapping with Hierarchical Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.09643v1)  

---


**ABSTRACT**  
Deep learning-based methods have been extensively explored for automatic building mapping from high-resolution remote sensing images over recent years. While most building mapping models produce vector polygons of buildings for geographic and mapping systems, dominant methods typically decompose polygonal building extraction in some sub-problems, including segmentation, polygonization, and regularization, leading to complex inference procedures, low accuracy, and poor generalization. In this paper, we propose a simple and novel building mapping method with Hierarchical Transformers, called HiT, improving polygonal building mapping quality from high-resolution remote sensing images. HiT builds on a two-stage detection architecture by adding a polygon head parallel to classification and bounding box regression heads. HiT simultaneously outputs building bounding boxes and vector polygons, which is fully end-to-end trainable. The polygon head formulates a building polygon as serialized vertices with the bidirectional characteristic, a simple and elegant polygon representation avoiding the start or end vertex hypothesis. Under this new perspective, the polygon head adopts a transformer encoder-decoder architecture to predict serialized vertices supervised by the designed bidirectional polygon loss. Furthermore, a hierarchical attention mechanism combined with convolution operation is introduced in the encoder of the polygon head, providing more geometric structures of building polygons at vertex and edge levels. Comprehensive experiments on two benchmarks (the CrowdAI and Inria datasets) demonstrate that our method achieves a new state-of-the-art in terms of instance segmentation and polygonal metrics compared with state-of-the-art methods. Moreover, qualitative results verify the superiority and effectiveness of our model under complex scenes.

{{</citation>}}


### (15/154) Collaborative Three-Stream Transformers for Video Captioning (Hao Wang et al., 2023)

{{<citation>}}

Hao Wang, Libo Zhang, Heng Fan, Tiejian Luo. (2023)  
**Collaborative Three-Stream Transformers for Video Captioning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.09611v1)  

---


**ABSTRACT**  
As the most critical components in a sentence, subject, predicate and object require special attention in the video captioning task. To implement this idea, we design a novel framework, named COllaborative three-Stream Transformers (COST), to model the three parts separately and complement each other for better representation. Specifically, COST is formed by three branches of transformers to exploit the visual-linguistic interactions of different granularities in spatial-temporal domain between videos and text, detected objects and text, and actions and text. Meanwhile, we propose a cross-granularity attention module to align the interactions modeled by the three branches of transformers, then the three branches of transformers can support each other to exploit the most discriminative semantic information of different granularities for accurate predictions of captions. The whole model is trained in an end-to-end fashion. Extensive experiments conducted on three large-scale challenging datasets, i.e., YouCookII, ActivityNet Captions and MSVD, demonstrate that the proposed method performs favorably against the state-of-the-art methods.

{{</citation>}}


### (16/154) Mutual Information-calibrated Conformal Feature Fusion for Uncertainty-Aware Multimodal 3D Object Detection at the Edge (Alex C. Stutts et al., 2023)

{{<citation>}}

Alex C. Stutts, Danilo Erricolo, Sathya Ravi, Theja Tulabandhula, Amit Ranjan Trivedi. (2023)  
**Mutual Information-calibrated Conformal Feature Fusion for Uncertainty-Aware Multimodal 3D Object Detection at the Edge**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-IT, cs-RO, cs.CV, math-IT  
Keywords: AI, Object Detection  
[Paper Link](http://arxiv.org/abs/2309.09593v1)  

---


**ABSTRACT**  
In the expanding landscape of AI-enabled robotics, robust quantification of predictive uncertainties is of great importance. Three-dimensional (3D) object detection, a critical robotics operation, has seen significant advancements; however, the majority of current works focus only on accuracy and ignore uncertainty quantification. Addressing this gap, our novel study integrates the principles of conformal inference (CI) with information theoretic measures to perform lightweight, Monte Carlo-free uncertainty estimation within a multimodal framework. Through a multivariate Gaussian product of the latent variables in a Variational Autoencoder (VAE), features from RGB camera and LiDAR sensor data are fused to improve the prediction accuracy. Normalized mutual information (NMI) is leveraged as a modulator for calibrating uncertainty bounds derived from CI based on a weighted loss function. Our simulation results show an inverse correlation between inherent predictive uncertainty and NMI throughout the model's training. The framework demonstrates comparable or better performance in KITTI 3D object detection benchmarks to similar methods that are not uncertainty-aware, making it suitable for real-time edge robotics.

{{</citation>}}


### (17/154) Multi-Semantic Fusion Model for Generalized Zero-Shot Skeleton-Based Action Recognition (Ming-Zhe Li et al., 2023)

{{<citation>}}

Ming-Zhe Li, Zhen Jia, Zhang Zhang, Zhanyu Ma, Liang Wang. (2023)  
**Multi-Semantic Fusion Model for Generalized Zero-Shot Skeleton-Based Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.09592v1)  

---


**ABSTRACT**  
Generalized zero-shot skeleton-based action recognition (GZSSAR) is a new challenging problem in computer vision community, which requires models to recognize actions without any training samples. Previous studies only utilize the action labels of verb phrases as the semantic prototypes for learning the mapping from skeleton-based actions to a shared semantic space. However, the limited semantic information of action labels restricts the generalization ability of skeleton features for recognizing unseen actions. In order to solve this dilemma, we propose a multi-semantic fusion (MSF) model for improving the performance of GZSSAR, where two kinds of class-level textual descriptions (i.e., action descriptions and motion descriptions), are collected as auxiliary semantic information to enhance the learning efficacy of generalizable skeleton features. Specially, a pre-trained language encoder takes the action descriptions, motion descriptions and original class labels as inputs to obtain rich semantic features for each action class, while a skeleton encoder is implemented to extract skeleton features. Then, a variational autoencoder (VAE) based generative module is performed to learn a cross-modal alignment between skeleton and semantic features. Finally, a classification module is built to recognize the action categories of input samples, where a seen-unseen classification gate is adopted to predict whether the sample comes from seen action classes or not in GZSSAR. The superior performance in comparisons with previous models validates the effectiveness of the proposed MSF model on GZSSAR.

{{</citation>}}


### (18/154) Heterogeneous Generative Knowledge Distillation with Masked Image Modeling (Ziming Wang et al., 2023)

{{<citation>}}

Ziming Wang, Shumin Han, Xiaodi Wang, Jing Hao, Xianbin Cao, Baochang Zhang. (2023)  
**Heterogeneous Generative Knowledge Distillation with Masked Image Modeling**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Knowledge Distillation, Transformer  
[Paper Link](http://arxiv.org/abs/2309.09571v1)  

---


**ABSTRACT**  
Small CNN-based models usually require transferring knowledge from a large model before they are deployed in computationally resource-limited edge devices. Masked image modeling (MIM) methods achieve great success in various visual tasks but remain largely unexplored in knowledge distillation for heterogeneous deep models. The reason is mainly due to the significant discrepancy between the Transformer-based large model and the CNN-based small network. In this paper, we develop the first Heterogeneous Generative Knowledge Distillation (H-GKD) based on MIM, which can efficiently transfer knowledge from large Transformer models to small CNN-based models in a generative self-supervised fashion. Our method builds a bridge between Transformer-based models and CNNs by training a UNet-style student with sparse convolution, which can effectively mimic the visual representation inferred by a teacher over masked modeling. Our method is a simple yet effective learning paradigm to learn the visual representation and distribution of data from heterogeneous teacher models, which can be pre-trained using advanced generative methods. Extensive experiments show that it adapts well to various models and sizes, consistently achieving state-of-the-art performance in image classification, object detection, and semantic segmentation tasks. For example, in the Imagenet 1K dataset, H-GKD improves the accuracy of Resnet50 (sparse) from 76.98% to 80.01%.

{{</citation>}}


### (19/154) RIDE: Self-Supervised Learning of Rotation-Equivariant Keypoint Detection and Invariant Description for Endoscopy (Mert Asim Karaoglu et al., 2023)

{{<citation>}}

Mert Asim Karaoglu, Viktoria Markova, Nassir Navab, Benjamin Busam, Alexander Ladikos. (2023)  
**RIDE: Self-Supervised Learning of Rotation-Equivariant Keypoint Detection and Invariant Description for Endoscopy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.09563v1)  

---


**ABSTRACT**  
Unlike in natural images, in endoscopy there is no clear notion of an up-right camera orientation. Endoscopic videos therefore often contain large rotational motions, which require keypoint detection and description algorithms to be robust to these conditions. While most classical methods achieve rotation-equivariant detection and invariant description by design, many learning-based approaches learn to be robust only up to a certain degree. At the same time learning-based methods under moderate rotations often outperform classical approaches. In order to address this shortcoming, in this paper we propose RIDE, a learning-based method for rotation-equivariant detection and invariant description. Following recent advancements in group-equivariant learning, RIDE models rotation-equivariance implicitly within its architecture. Trained in a self-supervised manner on a large curation of endoscopic images, RIDE requires no manual labeling of training data. We test RIDE in the context of surgical tissue tracking on the SuPeR dataset as well as in the context of relative pose estimation on a repurposed version of the SCARED dataset. In addition we perform explicit studies showing its robustness to large rotations. Our comparison against recent learning-based and classical approaches shows that RIDE sets a new state-of-the-art performance on matching and relative pose estimation tasks and scores competitively on surgical tissue tracking.

{{</citation>}}


### (20/154) Causal-Story: Local Causal Attention Utilizing Parameter-Efficient Tuning For Visual Story Synthesis (Tianyi Song et al., 2023)

{{<citation>}}

Tianyi Song, Jiuxin Cao, Kun Wang, Bo Liu, Xiaofeng Zhang. (2023)  
**Causal-Story: Local Causal Attention Utilizing Parameter-Efficient Tuning For Visual Story Synthesis**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-MM, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.09553v3)  

---


**ABSTRACT**  
The excellent text-to-image synthesis capability of diffusion models has driven progress in synthesizing coherent visual stories. The current state-of-the-art method combines the features of historical captions, historical frames, and the current captions as conditions for generating the current frame. However, this method treats each historical frame and caption as the same contribution. It connects them in order with equal weights, ignoring that not all historical conditions are associated with the generation of the current frame. To address this issue, we propose Causal-Story. This model incorporates a local causal attention mechanism that considers the causal relationship between previous captions, frames, and current captions. By assigning weights based on this relationship, Causal-Story generates the current frame, thereby improving the global consistency of story generation. We evaluated our model on the PororoSV and FlintstonesSV datasets and obtained state-of-the-art FID scores, and the generated frames also demonstrate better storytelling in visuals.

{{</citation>}}


### (21/154) Selective Volume Mixup for Video Action Recognition (Yi Tan et al., 2023)

{{<citation>}}

Yi Tan, Zhaofan Qiu, Yanbin Hao, Ting Yao, Xiangnan He, Tao Mei. (2023)  
**Selective Volume Mixup for Video Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.09534v1)  

---


**ABSTRACT**  
The recent advances in Convolutional Neural Networks (CNNs) and Vision Transformers have convincingly demonstrated high learning capability for video action recognition on large datasets. Nevertheless, deep models often suffer from the overfitting effect on small-scale datasets with a limited number of training videos. A common solution is to exploit the existing image augmentation strategies for each frame individually including Mixup, Cutmix, and RandAugment, which are not particularly optimized for video data. In this paper, we propose a novel video augmentation strategy named Selective Volume Mixup (SV-Mix) to improve the generalization ability of deep models with limited training videos. SV-Mix devises a learnable selective module to choose the most informative volumes from two videos and mixes the volumes up to achieve a new training video. Technically, we propose two new modules, i.e., a spatial selective module to select the local patches for each spatial position, and a temporal selective module to mix the entire frames for each timestamp and maintain the spatial pattern. At each time, we randomly choose one of the two modules to expand the diversity of training samples. The selective modules are jointly optimized with the video action recognition framework to find the optimal augmentation strategy. We empirically demonstrate the merits of the SV-Mix augmentation on a wide range of video action recognition benchmarks and consistently boot the performances of both CNN-based and transformer-based models.

{{</citation>}}


### (22/154) Instant Photorealistic Style Transfer: A Lightweight and Adaptive Approach (Rong Liu et al., 2023)

{{<citation>}}

Rong Liu, Enyu Zhao, Zhiyuan Liu, Andrew Wei-Wen Feng, Scott John Easley. (2023)  
**Instant Photorealistic Style Transfer: A Lightweight and Adaptive Approach**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV, eess-IV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2309.10011v1)  

---


**ABSTRACT**  
In this paper, we propose an Instant Photorealistic Style Transfer (IPST) approach, designed to achieve instant photorealistic style transfer on super-resolution inputs without the need for pre-training on pair-wise datasets or imposing extra constraints. Our method utilizes a lightweight StyleNet to enable style transfer from a style image to a content image while preserving non-color information. To further enhance the style transfer process, we introduce an instance-adaptive optimization to prioritize the photorealism of outputs and accelerate the convergence of the style network, leading to a rapid training completion within seconds. Moreover, IPST is well-suited for multi-frame style transfer tasks, as it retains temporal and multi-view consistency of the multi-frame inputs such as video and Neural Radiance Field (NeRF). Experimental results demonstrate that IPST requires less GPU memory usage, offers faster multi-frame transfer speed, and generates photorealistic outputs, making it a promising solution for various photorealistic transfer applications.

{{</citation>}}


### (23/154) LayoutNUWA: Revealing the Hidden Layout Expertise of Large Language Models (Zecheng Tang et al., 2023)

{{<citation>}}

Zecheng Tang, Chenfei Wu, Juntao Li, Nan Duan. (2023)  
**LayoutNUWA: Revealing the Hidden Layout Expertise of Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.09506v2)  

---


**ABSTRACT**  
Graphic layout generation, a growing research field, plays a significant role in user engagement and information perception. Existing methods primarily treat layout generation as a numerical optimization task, focusing on quantitative aspects while overlooking the semantic information of layout, such as the relationship between each layout element. In this paper, we propose LayoutNUWA, the first model that treats layout generation as a code generation task to enhance semantic information and harness the hidden layout expertise of large language models~(LLMs). More concretely, we develop a Code Instruct Tuning (CIT) approach comprising three interconnected modules: 1) the Code Initialization (CI) module quantifies the numerical conditions and initializes them as HTML code with strategically placed masks; 2) the Code Completion (CC) module employs the formatting knowledge of LLMs to fill in the masked portions within the HTML code; 3) the Code Rendering (CR) module transforms the completed code into the final layout output, ensuring a highly interpretable and transparent layout generation procedure that directly maps code to a visualized layout. We attain significant state-of-the-art performance (even over 50\% improvements) on multiple datasets, showcasing the strong capabilities of LayoutNUWA. Our code is available at https://github.com/ProjectNUWA/LayoutNUWA.

{{</citation>}}


### (24/154) Discovering Sounding Objects by Audio Queries for Audio Visual Segmentation (Shaofei Huang et al., 2023)

{{<citation>}}

Shaofei Huang, Han Li, Yuqing Wang, Hongji Zhu, Jiao Dai, Jizhong Han, Wenge Rong, Si Liu. (2023)  
**Discovering Sounding Objects by Audio Queries for Audio Visual Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.09501v1)  

---


**ABSTRACT**  
Audio visual segmentation (AVS) aims to segment the sounding objects for each frame of a given video. To distinguish the sounding objects from silent ones, both audio-visual semantic correspondence and temporal interaction are required. The previous method applies multi-frame cross-modal attention to conduct pixel-level interactions between audio features and visual features of multiple frames simultaneously, which is both redundant and implicit. In this paper, we propose an Audio-Queried Transformer architecture, AQFormer, where we define a set of object queries conditioned on audio information and associate each of them to particular sounding objects. Explicit object-level semantic correspondence between audio and visual modalities is established by gathering object information from visual features with predefined audio queries. Besides, an Audio-Bridged Temporal Interaction module is proposed to exchange sounding object-relevant information among multiple frames with the bridge of audio features. Extensive experiments are conducted on two AVS benchmarks to show that our method achieves state-of-the-art performances, especially 7.1% M_J and 7.6% M_F gains on the MS3 setting.

{{</citation>}}


### (25/154) CLIP-based Synergistic Knowledge Transfer for Text-based Person Retrieval (Yating liu et al., 2023)

{{<citation>}}

Yating liu, Yaowei Li, Zimo Liu, Wenming Yang, Yaowei Wang, Qingmin Liao. (2023)  
**CLIP-based Synergistic Knowledge Transfer for Text-based Person Retrieval**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2309.09496v1)  

---


**ABSTRACT**  
Text-based Person Retrieval aims to retrieve the target person images given a textual query. The primary challenge lies in bridging the substantial gap between vision and language modalities, especially when dealing with limited large-scale datasets. In this paper, we introduce a CLIP-based Synergistic Knowledge Transfer(CSKT) approach for TBPR. Specifically, to explore the CLIP's knowledge on input side, we first propose a Bidirectional Prompts Transferring (BPT) module constructed by text-to-image and image-to-text bidirectional prompts and coupling projections. Secondly, Dual Adapters Transferring (DAT) is designed to transfer knowledge on output side of Multi-Head Self-Attention (MHSA) in vision and language. This synergistic two-way collaborative mechanism promotes the early-stage feature fusion and efficiently exploits the existing knowledge of CLIP. CSKT outperforms the state-of-the-art approaches across three benchmark datasets when the training parameters merely account for 7.4% of the entire model, demonstrating its remarkable efficiency, effectiveness and generalization.

{{</citation>}}


### (26/154) Target-aware Bi-Transformer for Few-shot Segmentation (Xianglin Wang et al., 2023)

{{<citation>}}

Xianglin Wang, Xiaoliu Luo, Taiping Zhang. (2023)  
**Target-aware Bi-Transformer for Few-shot Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.09492v1)  

---


**ABSTRACT**  
Traditional semantic segmentation tasks require a large number of labels and are difficult to identify unlearned categories. Few-shot semantic segmentation (FSS) aims to use limited labeled support images to identify the segmentation of new classes of objects, which is very practical in the real world. Previous researches were primarily based on prototypes or correlations. Due to colors, textures, and styles are similar in the same image, we argue that the query image can be regarded as its own support image. In this paper, we proposed the Target-aware Bi-Transformer Network (TBTNet) to equivalent treat of support images and query image. A vigorous Target-aware Transformer Layer (TTL) also be designed to distill correlations and force the model to focus on foreground information. It treats the hypercorrelation as a feature, resulting a significant reduction in the number of feature channels. Benefit from this characteristic, our model is the lightest up to now with only 0.4M learnable parameters. Futhermore, TBTNet converges in only 10% to 25% of the training epochs compared to traditional methods. The excellent performance on standard FSS benchmarks of PASCAL-5i and COCO-20i proves the efficiency of our method. Extensive ablation studies were also carried out to evaluate the effectiveness of Bi-Transformer architecture and TTL.

{{</citation>}}


### (27/154) Self-supervised Multi-view Clustering in Computer Vision: A Survey (Jiatai Wang et al., 2023)

{{<citation>}}

Jiatai Wang, Zhiwei Xu, Xuewen Yang, Hailong Li, Bo Li, Xuying Meng. (2023)  
**Self-supervised Multi-view Clustering in Computer Vision: A Survey**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2309.09473v1)  

---


**ABSTRACT**  
Multi-view clustering (MVC) has had significant implications in cross-modal representation learning and data-driven decision-making in recent years. It accomplishes this by leveraging the consistency and complementary information among multiple views to cluster samples into distinct groups. However, as contrastive learning continues to evolve within the field of computer vision, self-supervised learning has also made substantial research progress and is progressively becoming dominant in MVC methods. It guides the clustering process by designing proxy tasks to mine the representation of image and video data itself as supervisory information. Despite the rapid development of self-supervised MVC, there has yet to be a comprehensive survey to analyze and summarize the current state of research progress. Therefore, this paper explores the reasons and advantages of the emergence of self-supervised MVC and discusses the internal connections and classifications of common datasets, data issues, representation learning methods, and self-supervised learning methods. This paper does not only introduce the mechanisms for each category of methods but also gives a few examples of how these techniques are used. In the end, some open problems are pointed out for further investigation and development.

{{</citation>}}


### (28/154) Reconstructing Existing Levels through Level Inpainting (Johor Jara Gonzalez et al., 2023)

{{<citation>}}

Johor Jara Gonzalez, Mathew Guzdial. (2023)  
**Reconstructing Existing Levels through Level Inpainting**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.09472v1)  

---


**ABSTRACT**  
Procedural Content Generation (PCG) and Procedural Content Generation via Machine Learning (PCGML) have been used in prior work for generating levels in various games. This paper introduces Content Augmentation and focuses on the subproblem of level inpainting, which involves reconstructing and extending video game levels. Drawing inspiration from image inpainting, we adapt two techniques from this domain to address our specific use case. We present two approaches for level inpainting: an Autoencoder and a U-net. Through a comprehensive case study, we demonstrate their superior performance compared to a baseline method and discuss their relative merits. Furthermore, we provide a practical demonstration of both approaches for the level inpainting task and offer insights into potential directions for future research.

{{</citation>}}


### (29/154) Progressive Text-to-Image Diffusion with Soft Latent Direction (YuTeng Ye et al., 2023)

{{<citation>}}

YuTeng Ye, Jiale Cai, Hang Zhou, Guanwen Li, Youjia Zhang, Zikai Song, Chenxing Gao, Junqing Yu, Wei Yang. (2023)  
**Progressive Text-to-Image Diffusion with Soft Latent Direction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.09466v1)  

---


**ABSTRACT**  
In spite of the rapidly evolving landscape of text-to-image generation, the synthesis and manipulation of multiple entities while adhering to specific relational constraints pose enduring challenges. This paper introduces an innovative progressive synthesis and editing operation that systematically incorporates entities into the target image, ensuring their adherence to spatial and relational constraints at each sequential step. Our key insight stems from the observation that while a pre-trained text-to-image diffusion model adeptly handles one or two entities, it often falters when dealing with a greater number. To address this limitation, we propose harnessing the capabilities of a Large Language Model (LLM) to decompose intricate and protracted text descriptions into coherent directives adhering to stringent formats. To facilitate the execution of directives involving distinct semantic operations-namely insertion, editing, and erasing-we formulate the Stimulus, Response, and Fusion (SRF) framework. Within this framework, latent regions are gently stimulated in alignment with each operation, followed by the fusion of the responsive latent components to achieve cohesive entity manipulation. Our proposed framework yields notable advancements in object synthesis, particularly when confronted with intricate and lengthy textual inputs. Consequently, it establishes a new benchmark for text-to-image generation tasks, further elevating the field's performance standards.

{{</citation>}}


### (30/154) Reducing Adversarial Training Cost with Gradient Approximation (Huihui Gong et al., 2023)

{{<citation>}}

Huihui Gong, Shuo Yang, Siqi Ma, Seyit Camtepe, Surya Nepal, Chang Xu. (2023)  
**Reducing Adversarial Training Cost with Gradient Approximation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2309.09464v1)  

---


**ABSTRACT**  
Deep learning models have achieved state-of-the-art performances in various domains, while they are vulnerable to the inputs with well-crafted but small perturbations, which are named after adversarial examples (AEs). Among many strategies to improve the model robustness against AEs, Projected Gradient Descent (PGD) based adversarial training is one of the most effective methods. Unfortunately, the prohibitive computational overhead of generating strong enough AEs, due to the maximization of the loss function, sometimes makes the regular PGD adversarial training impractical when using larger and more complicated models. In this paper, we propose that the adversarial loss can be approximated by the partial sum of Taylor series. Furthermore, we approximate the gradient of adversarial loss and propose a new and efficient adversarial training method, adversarial training with gradient approximation (GAAT), to reduce the cost of building up robust models. Additionally, extensive experiments demonstrate that this efficiency improvement can be achieved without any or with very little loss in accuracy on natural and adversarial examples, which show that our proposed method saves up to 60\% of the training time with comparable model test accuracy on MNIST, CIFAR-10 and CIFAR-100 datasets.

{{</citation>}}


### (31/154) FactoFormer: Factorized Hyperspectral Transformers with Self-Supervised Pre-Training (Shaheer Mohamed et al., 2023)

{{<citation>}}

Shaheer Mohamed, Maryam Haghighat, Tharindu Fernando, Sridha Sridharan, Clinton Fookes, Peyman Moghadam. (2023)  
**FactoFormer: Factorized Hyperspectral Transformers with Self-Supervised Pre-Training**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Self-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.09431v1)  

---


**ABSTRACT**  
Hyperspectral images (HSIs) contain rich spectral and spatial information. Motivated by the success of transformers in the field of natural language processing and computer vision where they have shown the ability to learn long range dependencies within input data, recent research has focused on using transformers for HSIs. However, current state-of-the-art hyperspectral transformers only tokenize the input HSI sample along the spectral dimension, resulting in the under-utilization of spatial information. Moreover, transformers are known to be data-hungry and their performance relies heavily on large-scale pre-training, which is challenging due to limited annotated hyperspectral data. Therefore, the full potential of HSI transformers has not been fully realized. To overcome these limitations, we propose a novel factorized spectral-spatial transformer that incorporates factorized self-supervised pre-training procedures, leading to significant improvements in performance. The factorization of the inputs allows the spectral and spatial transformers to better capture the interactions within the hyperspectral data cubes. Inspired by masked image modeling pre-training, we also devise efficient masking strategies for pre-training each of the spectral and spatial transformers. We conduct experiments on three publicly available datasets for HSI classification task and demonstrate that our model achieves state-of-the-art performance in all three datasets. The code for our model will be made available at https://github.com/csiro-robotics/factoformer.

{{</citation>}}


## cs.CL (22)



### (32/154) Stabilizing RLHF through Advantage Model and Selective Rehearsal (Baolin Peng et al., 2023)

{{<citation>}}

Baolin Peng, Linfeng Song, Ye Tian, Lifeng Jin, Haitao Mi, Dong Yu. (2023)  
**Stabilizing RLHF through Advantage Model and Selective Rehearsal**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.10202v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have revolutionized natural language processing, yet aligning these models with human values and preferences using RLHF remains a significant challenge. This challenge is characterized by various instabilities, such as reward hacking and catastrophic forgetting. In this technical report, we propose two innovations to stabilize RLHF training: 1) Advantage Model, which directly models advantage score i.e., extra reward compared to the expected rewards and regulates score distributions across tasks to prevent reward hacking. 2) Selective Rehearsal, which mitigates catastrophic forgetting by strategically selecting data for PPO training and knowledge rehearsing. Our experimental analysis on public and proprietary datasets reveals that the proposed methods not only increase stability in RLHF training but also achieve higher reward scores and win rates.

{{</citation>}}


### (33/154) Few-Shot Adaptation for Parsing Contextual Utterances with LLMs (Kevin Lin et al., 2023)

{{<citation>}}

Kevin Lin, Patrick Xia, Hao Fang. (2023)  
**Few-Shot Adaptation for Parsing Contextual Utterances with LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.10168v1)  

---


**ABSTRACT**  
We evaluate the ability of semantic parsers based on large language models (LLMs) to handle contextual utterances. In real-world settings, there typically exists only a limited number of annotated contextual utterances due to annotation cost, resulting in an imbalance compared to non-contextual utterances. Therefore, parsers must adapt to contextual utterances with a few training examples. We examine four major paradigms for doing so in conversational semantic parsing i.e., Parse-with-Utterance-History, Parse-with-Reference-Program, Parse-then-Resolve, and Rewrite-then-Parse. To facilitate such cross-paradigm comparisons, we construct SMCalFlow-EventQueries, a subset of contextual examples from SMCalFlow with additional annotations. Experiments with in-context learning and fine-tuning suggest that Rewrite-then-Parse is the most promising paradigm when holistically considering parsing accuracy, annotation cost, and error types.

{{</citation>}}


### (34/154) Understanding Catastrophic Forgetting in Language Models via Implicit Inference (Suhas Kotha et al., 2023)

{{<citation>}}

Suhas Kotha, Jacob Mitchell Springer, Aditi Raghunathan. (2023)  
**Understanding Catastrophic Forgetting in Language Models via Implicit Inference**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.10105v1)  

---


**ABSTRACT**  
Fine-tuning (via methods such as instruction-tuning or reinforcement learning from human feedback) is a crucial step in training language models to robustly carry out tasks of interest. However, we lack a systematic understanding of the effects of fine-tuning, particularly on tasks outside the narrow fine-tuning distribution. In a simplified scenario, we demonstrate that improving performance on tasks within the fine-tuning data distribution comes at the expense of suppressing model capabilities on other tasks. This degradation is especially pronounced for tasks "closest" to the fine-tuning distribution. We hypothesize that language models implicitly infer the task of the prompt corresponds, and the fine-tuning process predominantly skews this task inference towards tasks in the fine-tuning distribution. To test this hypothesis, we propose Conjugate Prompting to see if we can recover pretrained capabilities. Conjugate prompting artificially makes the task look farther from the fine-tuning distribution while requiring the same capability. We find that conjugate prompting systematically recovers some of the pretraining capabilities on our synthetic setup. We then apply conjugate prompting to real-world LLMs using the observation that fine-tuning distributions are typically heavily skewed towards English. We find that simply translating the prompts to different languages can cause the fine-tuned models to respond like their pretrained counterparts instead. This allows us to recover the in-context learning abilities lost via instruction tuning, and more concerningly, to recover harmful content generation suppressed by safety fine-tuning in chatbots like ChatGPT.

{{</citation>}}


### (35/154) Not Enough Labeled Data? Just Add Semantics: A Data-Efficient Method for Inferring Online Health Texts (Joseph Gatto et al., 2023)

{{<citation>}}

Joseph Gatto, Sarah M. Preum. (2023)  
**Not Enough Labeled Data? Just Add Semantics: A Data-Efficient Method for Inferring Online Health Texts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Abstract Meaning Representation, NLP  
[Paper Link](http://arxiv.org/abs/2309.09877v1)  

---


**ABSTRACT**  
User-generated texts available on the web and social platforms are often long and semantically challenging, making them difficult to annotate. Obtaining human annotation becomes increasingly difficult as problem domains become more specialized. For example, many health NLP problems require domain experts to be a part of the annotation pipeline. Thus, it is crucial that we develop low-resource NLP solutions able to work with this set of limited-data problems. In this study, we employ Abstract Meaning Representation (AMR) graphs as a means to model low-resource Health NLP tasks sourced from various online health resources and communities. AMRs are well suited to model online health texts as they can represent multi-sentence inputs, abstract away from complex terminology, and model long-distance relationships between co-referring tokens. AMRs thus improve the ability of pre-trained language models to reason about high-complexity texts. Our experiments show that we can improve performance on 6 low-resource health NLP tasks by augmenting text embeddings with semantic graph embeddings. Our approach is task agnostic and easy to merge into any standard text classification pipeline. We experimentally validate that AMRs are useful in the modeling of complex texts by analyzing performance through the lens of two textual complexity measures: the Flesch Kincaid Reading Level and Syntactic Complexity. Our error analysis shows that AMR-infused language models perform better on complex texts and generally show less predictive variance in the presence of changing complexity.

{{</citation>}}


### (36/154) SYNDICOM: Improving Conversational Commonsense with Error-Injection and Natural Language Feedback (Christopher Richardson et al., 2023)

{{<citation>}}

Christopher Richardson, Anirudh Sundar, Larry Heck. (2023)  
**SYNDICOM: Improving Conversational Commonsense with Error-Injection and Natural Language Feedback**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.10015v1)  

---


**ABSTRACT**  
Commonsense reasoning is a critical aspect of human communication. Despite recent advances in conversational AI driven by large language models, commonsense reasoning remains a challenging task. In this work, we introduce SYNDICOM - a method for improving commonsense in dialogue response generation. SYNDICOM consists of two components. The first component is a dataset composed of commonsense dialogues created from a knowledge graph and synthesized into natural language. This dataset includes both valid and invalid responses to dialogue contexts, along with natural language feedback (NLF) for the invalid responses. The second contribution is a two-step procedure: training a model to predict natural language feedback (NLF) for invalid responses, and then training a response generation model conditioned on the predicted NLF, the invalid response, and the dialogue. SYNDICOM is scalable and does not require reinforcement learning. Empirical results on three tasks are evaluated using a broad range of metrics. SYNDICOM achieves a relative improvement of 53% over ChatGPT on ROUGE1, and human evaluators prefer SYNDICOM over ChatGPT 57% of the time. We will publicly release the code and the full dataset.

{{</citation>}}


### (37/154) Instruction-Following Speech Recognition (Cheng-I Jeff Lai et al., 2023)

{{<citation>}}

Cheng-I Jeff Lai, Zhiyun Lu, Liangliang Cao, Ruoming Pang. (2023)  
**Instruction-Following Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: Language Model, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.09843v1)  

---


**ABSTRACT**  
Conventional end-to-end Automatic Speech Recognition (ASR) models primarily focus on exact transcription tasks, lacking flexibility for nuanced user interactions. With the advent of Large Language Models (LLMs) in speech processing, more organic, text-prompt-based interactions have become possible. However, the mechanisms behind these models' speech understanding and "reasoning" capabilities remain underexplored. To study this question from the data perspective, we introduce instruction-following speech recognition, training a Listen-Attend-Spell model to understand and execute a diverse set of free-form text instructions. This enables a multitude of speech recognition tasks -- ranging from transcript manipulation to summarization -- without relying on predefined command sets. Remarkably, our model, trained from scratch on Librispeech, interprets and executes simple instructions without requiring LLMs or pre-trained speech modules. It also offers selective transcription options based on instructions like "transcribe first half and then turn off listening," providing an additional layer of privacy and safety compared to existing LLMs. Our findings highlight the significant potential of instruction-following training to advance speech foundation models.

{{</citation>}}


### (38/154) HypR: A comprehensive study for ASR hypothesis revising with a reference corpus (Yi-Wei Wang et al., 2023)

{{<citation>}}

Yi-Wei Wang, Ke-Han Lu, Kuan-Yu Chen. (2023)  
**HypR: A comprehensive study for ASR hypothesis revising with a reference corpus**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09838v2)  

---


**ABSTRACT**  
With the development of deep learning, automatic speech recognition (ASR) has made significant progress. To further enhance the performance, revising recognition results is one of the lightweight but efficient manners. Various methods can be roughly classified into N-best reranking methods and error correction models. The former aims to select the hypothesis with the lowest error rate from a set of candidates generated by ASR for a given input speech. The latter focuses on detecting recognition errors in a given hypothesis and correcting these errors to obtain an enhanced result. However, we observe that these studies are hardly comparable to each other as they are usually evaluated on different corpora, paired with different ASR models, and even use different datasets to train the models. Accordingly, we first concentrate on releasing an ASR hypothesis revising (HypR) dataset in this study. HypR contains several commonly used corpora (AISHELL-1, TED-LIUM 2, and LibriSpeech) and provides 50 recognition hypotheses for each speech utterance. The checkpoint models of the ASR are also published. In addition, we implement and compare several classic and representative methods, showing the recent research progress in revising speech recognition results. We hope the publicly available HypR dataset can become a reference benchmark for subsequent research and promote the school of research to an advanced level.

{{</citation>}}


### (39/154) Task Selection and Assignment for Multi-modal Multi-task Dialogue Act Classification with Non-stationary Multi-armed Bandits (Xiangheng He et al., 2023)

{{<citation>}}

Xiangheng He, Junjie Chen, Björn W. Schuller. (2023)  
**Task Selection and Assignment for Multi-modal Multi-task Dialogue Act Classification with Non-stationary Multi-armed Bandits**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2309.09832v1)  

---


**ABSTRACT**  
Multi-task learning (MTL) aims to improve the performance of a primary task by jointly learning with related auxiliary tasks. Traditional MTL methods select tasks randomly during training. However, both previous studies and our results suggest that such the random selection of tasks may not be helpful, and can even be harmful to performance. Therefore, new strategies for task selection and assignment in MTL need to be explored. This paper studies the multi-modal, multi-task dialogue act classification task, and proposes a method for selecting and assigning tasks based on non-stationary multi-armed bandits (MAB) with discounted Thompson Sampling (TS) using Gaussian priors. Our experimental results show that in different training stages, different tasks have different utility. Our proposed method can effectively identify the task utility, actively avoid useless or harmful tasks, and realise the task assignment during training. Our proposed method is significantly superior in terms of UAR and F1 to the single-task and multi-task baselines with p-values < 0.05. Further analysis of experiments indicates that for the dataset with the data imbalance problem, our proposed method has significantly higher stability and can obtain consistent and decent performance for minority classes. Our proposed method is superior to the current state-of-the-art model.

{{</citation>}}


### (40/154) AMuRD: Annotated Multilingual Receipts Dataset for Cross-lingual Key Information Extraction and Classification (Abdelrahman Abdallah et al., 2023)

{{<citation>}}

Abdelrahman Abdallah, Mahmoud Abdalla, Mohamed Elkasaby, Yasser Elbendary, Adam Jatowt. (2023)  
**AMuRD: Annotated Multilingual Receipts Dataset for Cross-lingual Key Information Extraction and Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Information Extraction, LLaMA, Multilingual  
[Paper Link](http://arxiv.org/abs/2309.09800v1)  

---


**ABSTRACT**  
Key information extraction involves recognizing and extracting text from scanned receipts, enabling retrieval of essential content, and organizing it into structured documents. This paper presents a novel multilingual dataset for receipt extraction, addressing key challenges in information extraction and item classification. The dataset comprises $47,720$ samples, including annotations for item names, attributes like (price, brand, etc.), and classification into $44$ product categories. We introduce the InstructLLaMA approach, achieving an F1 score of $0.76$ and an accuracy of $0.68$ for key information extraction and item classification. We provide code, datasets, and checkpoints.\footnote{\url{https://github.com/Update-For-Integrated-Business-AI/AMuRD}}.

{{</citation>}}


### (41/154) Watch the Speakers: A Hybrid Continuous Attribution Network for Emotion Recognition in Conversation With Emotion Disentanglement (Shanglin Lei et al., 2023)

{{<citation>}}

Shanglin Lei, Xiaoping Wang, Guanting Dong, Jiang Li, Yingjian Liu. (2023)  
**Watch the Speakers: A Hybrid Continuous Attribution Network for Emotion Recognition in Conversation With Emotion Disentanglement**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2309.09799v2)  

---


**ABSTRACT**  
Emotion Recognition in Conversation (ERC) has attracted widespread attention in the natural language processing field due to its enormous potential for practical applications. Existing ERC methods face challenges in achieving generalization to diverse scenarios due to insufficient modeling of context, ambiguous capture of dialogue relationships and overfitting in speaker modeling. In this work, we present a Hybrid Continuous Attributive Network (HCAN) to address these issues in the perspective of emotional continuation and emotional attribution. Specifically, HCAN adopts a hybrid recurrent and attention-based module to model global emotion continuity. Then a novel Emotional Attribution Encoding (EAE) is proposed to model intra- and inter-emotional attribution for each utterance. Moreover, aiming to enhance the robustness of the model in speaker modeling and improve its performance in different scenarios, A comprehensive loss function emotional cognitive loss $\mathcal{L}_{\rm EC}$ is proposed to alleviate emotional drift and overcome the overfitting of the model to speaker modeling. Our model achieves state-of-the-art performance on three datasets, demonstrating the superiority of our work. Another extensive comparative experiments and ablation studies on three benchmarks are conducted to provided evidence to support the efficacy of each module. Further exploration of generalization ability experiments shows the plug-and-play nature of the EAE module in our method.

{{</citation>}}


### (42/154) Facilitating NSFW Text Detection in Open-Domain Dialogue Systems via Knowledge Distillation (Huachuan Qiu et al., 2023)

{{<citation>}}

Huachuan Qiu, Shuai Zhang, Hongliang He, Anqi Li, Zhenzhong Lan. (2023)  
**Facilitating NSFW Text Detection in Open-Domain Dialogue Systems via Knowledge Distillation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, BERT, ChatGPT, Dialog, Dialogue, GPT, GPT-4, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.09749v2)  

---


**ABSTRACT**  
NSFW (Not Safe for Work) content, in the context of a dialogue, can have severe side effects on users in open-domain dialogue systems. However, research on detecting NSFW language, especially sexually explicit content, within a dialogue context has significantly lagged behind. To address this issue, we introduce CensorChat, a dialogue monitoring dataset aimed at NSFW dialogue detection. Leveraging knowledge distillation techniques involving GPT-4 and ChatGPT, this dataset offers a cost-effective means of constructing NSFW content detectors. The process entails collecting real-life human-machine interaction data and breaking it down into single utterances and single-turn dialogues, with the chatbot delivering the final utterance. ChatGPT is employed to annotate unlabeled data, serving as a training set. Rationale validation and test sets are constructed using ChatGPT and GPT-4 as annotators, with a self-criticism strategy for resolving discrepancies in labeling. A BERT model is fine-tuned as a text classifier on pseudo-labeled data, and its performance is assessed. The study emphasizes the importance of AI systems prioritizing user safety and well-being in digital conversations while respecting freedom of expression. The proposed approach not only advances NSFW content detection but also aligns with evolving user protection needs in AI-driven dialogues.

{{</citation>}}


### (43/154) LLM4Jobs: Unsupervised occupation extraction and standardization leveraging Large Language Models (Nan Li et al., 2023)

{{<citation>}}

Nan Li, Bo Kang, Tijl De Bie. (2023)  
**LLM4Jobs: Unsupervised occupation extraction and standardization leveraging Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.09708v2)  

---


**ABSTRACT**  
Automated occupation extraction and standardization from free-text job postings and resumes are crucial for applications like job recommendation and labor market policy formation. This paper introduces LLM4Jobs, a novel unsupervised methodology that taps into the capabilities of large language models (LLMs) for occupation coding. LLM4Jobs uniquely harnesses both the natural language understanding and generation capacities of LLMs. Evaluated on rigorous experimentation on synthetic and real-world datasets, we demonstrate that LLM4Jobs consistently surpasses unsupervised state-of-the-art benchmarks, demonstrating its versatility across diverse datasets and granularities. As a side result of our work, we present both synthetic and real-world datasets, which may be instrumental for subsequent research in this domain. Overall, this investigation highlights the promise of contemporary LLMs for the intricate task of occupation extraction and standardization, laying the foundation for a robust and adaptable framework relevant to both research and industrial contexts.

{{</citation>}}


### (44/154) Evaluating Gender Bias of Pre-trained Language Models in Natural Language Inference by Considering All Labels (Panatchakorn Anantaprayoon et al., 2023)

{{<citation>}}

Panatchakorn Anantaprayoon, Masahiro Kaneko, Naoaki Okazaki. (2023)  
**Evaluating Gender Bias of Pre-trained Language Models in Natural Language Inference by Considering All Labels**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Language Model, NLI, Natural Language Inference  
[Paper Link](http://arxiv.org/abs/2309.09697v1)  

---


**ABSTRACT**  
Discriminatory social biases, including gender biases, have been found in Pre-trained Language Models (PLMs). In Natural Language Inference (NLI), recent bias evaluation methods have observed biased inferences from the outputs of a particular label such as neutral or entailment. However, since different biased inferences can be associated with different output labels, it is inaccurate for a method to rely on one label. In this work, we propose an evaluation method that considers all labels in the NLI task. We create evaluation data and assign them into groups based on their expected biased output labels. Then, we define a bias measure based on the corresponding label output of each data group. In the experiment, we propose a meta-evaluation method for NLI bias measures, and then use it to confirm that our measure can evaluate bias more accurately than the baseline. Moreover, we show that our evaluation method is applicable to multiple languages by conducting the meta-evaluation on PLMs in three different languages: English, Japanese, and Chinese. Finally, we evaluate PLMs of each language to confirm their bias tendency. To our knowledge, we are the first to build evaluation datasets and measure the bias of PLMs from the NLI task in Japanese and Chinese.

{{</citation>}}


### (45/154) Multi-turn Dialogue Comprehension from a Topic-aware Perspective (Xinbei Ma et al., 2023)

{{<citation>}}

Xinbei Ma, Yi Xu, Hai Zhao, Zhuosheng Zhang. (2023)  
**Multi-turn Dialogue Comprehension from a Topic-aware Perspective**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Dialog, Dialogue, Machine Reading Comprehension  
[Paper Link](http://arxiv.org/abs/2309.09666v1)  

---


**ABSTRACT**  
Dialogue related Machine Reading Comprehension requires language models to effectively decouple and model multi-turn dialogue passages. As a dialogue development goes after the intentions of participants, its topic may not keep constant through the whole passage. Hence, it is non-trivial to detect and leverage the topic shift in dialogue modeling. Topic modeling, although has been widely studied in plain text, deserves far more utilization in dialogue reading comprehension. This paper proposes to model multi-turn dialogues from a topic-aware perspective. We start with a dialogue segmentation algorithm to split a dialogue passage into topic-concentrated fragments in an unsupervised way. Then we use these fragments as topic-aware language processing units in further dialogue comprehension. On one hand, the split segments indict specific topics rather than mixed intentions, thus showing convenient on in-domain topic detection and location. For this task, we design a clustering system with a self-training auto-encoder, and we build two constructed datasets for evaluation. On the other hand, the split segments are an appropriate element of multi-turn dialogue response selection. For this purpose, we further present a novel model, Topic-Aware Dual-Attention Matching (TADAM) Network, which takes topic segments as processing elements and matches response candidates with a dual cross-attention. Empirical studies on three public benchmarks show great improvements over baselines. Our work continues the previous studies on document topic, and brings the dialogue modeling to a novel topic-aware perspective with exhaustive experiments and analyses.

{{</citation>}}


### (46/154) A Novel Method of Fuzzy Topic Modeling based on Transformer Processing (Ching-Hsun Tseng et al., 2023)

{{<citation>}}

Ching-Hsun Tseng, Shin-Jye Lee, Po-Wei Cheng, Chien Lee, Chih-Chieh Hung. (2023)  
**A Novel Method of Fuzzy Topic Modeling based on Transformer Processing**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Topic Model, Topic Modeling, Transformer  
[Paper Link](http://arxiv.org/abs/2309.09658v1)  

---


**ABSTRACT**  
Topic modeling is admittedly a convenient way to monitor markets trend. Conventionally, Latent Dirichlet Allocation, LDA, is considered a must-do model to gain this type of information. By given the merit of deducing keyword with token conditional probability in LDA, we can know the most possible or essential topic. However, the results are not intuitive because the given topics cannot wholly fit human knowledge. LDA offers the first possible relevant keywords, which also brings out another problem of whether the connection is reliable based on the statistic possibility. It is also hard to decide the topic number manually in advance. As the booming trend of using fuzzy membership to cluster and using transformers to embed words, this work presents the fuzzy topic modeling based on soft clustering and document embedding from state-of-the-art transformer-based model. In our practical application in a press release monitoring, the fuzzy topic modeling gives a more natural result than the traditional output from LDA.

{{</citation>}}


### (47/154) Proposition from the Perspective of Chinese Language: A Chinese Proposition Classification Evaluation Benchmark (Conghui Niu et al., 2023)

{{<citation>}}

Conghui Niu, Mengyang Hu, Lin Bo, Xiaoli He, Dong Yu, Pengyuan Liu. (2023)  
**Proposition from the Perspective of Chinese Language: A Chinese Proposition Classification Evaluation Benchmark**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.09602v1)  

---


**ABSTRACT**  
Existing propositions often rely on logical constants for classification. Compared with Western languages that lean towards hypotaxis such as English, Chinese often relies on semantic or logical understanding rather than logical connectives in daily expressions, exhibiting the characteristics of parataxis. However, existing research has rarely paid attention to this issue. And accurately classifying these propositions is crucial for natural language understanding and reasoning. In this paper, we put forward the concepts of explicit and implicit propositions and propose a comprehensive multi-level proposition classification system based on linguistics and logic. Correspondingly, we create a large-scale Chinese proposition dataset PEACE from multiple domains, covering all categories related to propositions. To evaluate the Chinese proposition classification ability of existing models and explore their limitations, We conduct evaluations on PEACE using several different methods including the Rule-based method, SVM, BERT, RoBERTA, and ChatGPT. Results show the importance of properly modeling the semantic features of propositions. BERT has relatively good proposition classification capability, but lacks cross-domain transferability. ChatGPT performs poorly, but its classification ability can be improved by providing more proposition information. Many issues are still far from being resolved and require further study.

{{</citation>}}


### (48/154) Fabricator: An Open Source Toolkit for Generating Labeled Training Data with Teacher LLMs (Jonas Golde et al., 2023)

{{<citation>}}

Jonas Golde, Patrick Haller, Felix Hamborg, Julian Risch, Alan Akbik. (2023)  
**Fabricator: An Open Source Toolkit for Generating Labeled Training Data with Teacher LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.09582v1)  

---


**ABSTRACT**  
Most NLP tasks are modeled as supervised learning and thus require labeled training data to train effective models. However, manually producing such data at sufficient quality and quantity is known to be costly and time-intensive. Current research addresses this bottleneck by exploring a novel paradigm called zero-shot learning via dataset generation. Here, a powerful LLM is prompted with a task description to generate labeled data that can be used to train a downstream NLP model. For instance, an LLM might be prompted to "generate 500 movie reviews with positive overall sentiment, and another 500 with negative sentiment." The generated data could then be used to train a binary sentiment classifier, effectively leveraging an LLM as a teacher to a smaller student model. With this demo, we introduce Fabricator, an open-source Python toolkit for dataset generation. Fabricator implements common dataset generation workflows, supports a wide range of downstream NLP tasks (such as text classification, question answering, and entity recognition), and is integrated with well-known libraries to facilitate quick experimentation. With Fabricator, we aim to support researchers in conducting reproducible dataset generation experiments using LLMs and help practitioners apply this approach to train models for downstream tasks.

{{</citation>}}


### (49/154) Summarization is (Almost) Dead (Xiao Pu et al., 2023)

{{<citation>}}

Xiao Pu, Mingqi Gao, Xiaojun Wan. (2023)  
**Summarization is (Almost) Dead**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2309.09558v1)  

---


**ABSTRACT**  
How well can large language models (LLMs) generate summaries? We develop new datasets and conduct human evaluation experiments to evaluate the zero-shot generation capability of LLMs across five distinct summarization tasks. Our findings indicate a clear preference among human evaluators for LLM-generated summaries over human-written summaries and summaries generated by fine-tuned models. Specifically, LLM-generated summaries exhibit better factual consistency and fewer instances of extrinsic hallucinations. Due to the satisfactory performance of LLMs in summarization tasks (even surpassing the benchmark of reference summaries), we believe that most conventional works in the field of text summarization are no longer necessary in the era of LLMs. However, we recognize that there are still some directions worth exploring, such as the creation of novel datasets with higher quality and more reliable evaluation methods.

{{</citation>}}


### (50/154) Adapting Large Language Models via Reading Comprehension (Daixuan Cheng et al., 2023)

{{<citation>}}

Daixuan Cheng, Shaohan Huang, Furu Wei. (2023)  
**Adapting Large Language Models via Reading Comprehension**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-5, Language Model  
[Paper Link](http://arxiv.org/abs/2309.09530v1)  

---


**ABSTRACT**  
We explore how continued pre-training on domain-specific corpora influences large language models, revealing that training on the raw corpora endows the model with domain knowledge, but drastically hurts its prompting ability for question answering. Taken inspiration from human learning via reading comprehension--practice after reading improves the ability to answer questions based on the learned knowledge--we propose a simple method for transforming raw corpora into reading comprehension texts. Each raw text is enriched with a series of tasks related to its content. Our method, highly scalable and applicable to any pre-training corpora, consistently enhances performance across various tasks in three different domains: biomedicine, finance, and law. Notably, our 7B language model achieves competitive performance with domain-specific models of much larger scales, such as BloombergGPT-50B. Furthermore, we demonstrate that domain-specific reading comprehension texts can improve the model's performance even on general benchmarks, showing the potential to develop a general model across even more domains. Our model, code, and data will be available at https://github.com/microsoft/LMOps.

{{</citation>}}


### (51/154) Understanding Divergent Framing of the Supreme Court Controversies: Social Media vs. News Outlets (Jinsheng Pan et al., 2023)

{{<citation>}}

Jinsheng Pan, Zichen Wang, Weihong Qi, Hanjia Lyu, Jiebo Luo. (2023)  
**Understanding Divergent Framing of the Supreme Court Controversies: Social Media vs. News Outlets**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: Social Media  
[Paper Link](http://arxiv.org/abs/2309.09508v1)  

---


**ABSTRACT**  
Understanding the framing of political issues is of paramount importance as it significantly shapes how individuals perceive, interpret, and engage with these matters. While prior research has independently explored framing within news media and by social media users, there remains a notable gap in our comprehension of the disparities in framing political issues between these two distinct groups. To address this gap, we conduct a comprehensive investigation, focusing on the nuanced distinctions both qualitatively and quantitatively in the framing of social media and traditional media outlets concerning a series of American Supreme Court rulings on affirmative action, student loans, and abortion rights. Our findings reveal that, while some overlap in framing exists between social media and traditional media outlets, substantial differences emerge both across various topics and within specific framing categories. Compared to traditional news media, social media platforms tend to present more polarized stances across all framing categories. Further, we observe significant polarization in the news media's treatment (i.e., Left vs. Right leaning media) of affirmative action and abortion rights, whereas the topic of student loans tends to exhibit a greater degree of consensus. The disparities in framing between traditional and social media platforms carry significant implications for the formation of public opinion, policy decision-making, and the broader political landscape.

{{</citation>}}


### (52/154) Search and Learning for Unsupervised Text Generation (Lili Mou, 2023)

{{<citation>}}

Lili Mou. (2023)  
**Search and Learning for Unsupervised Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: AI, Text Generation  
[Paper Link](http://arxiv.org/abs/2309.09497v1)  

---


**ABSTRACT**  
With the advances of deep learning techniques, text generation is attracting increasing interest in the artificial intelligence (AI) community, because of its wide applications and because it is an essential component of AI. Traditional text generation systems are trained in a supervised way, requiring massive labeled parallel corpora. In this paper, I will introduce our recent work on search and learning approaches to unsupervised text generation, where a heuristic objective function estimates the quality of a candidate sentence, and discrete search algorithms generate a sentence by maximizing the search objective. A machine learning model further learns from the search results to smooth out noise and improve efficiency. Our approach is important to the industry for building minimal viable products for a new task; it also has high social impacts for saving human annotation labor and for processing low-resource languages.

{{</citation>}}


### (53/154) Investigating Zero- and Few-shot Generalization in Fact Verification (Liangming Pan et al., 2023)

{{<citation>}}

Liangming Pan, Yunxiang Zhang, Min-Yen Kan. (2023)  
**Investigating Zero- and Few-shot Generalization in Fact Verification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Fact Verification  
[Paper Link](http://arxiv.org/abs/2309.09444v1)  

---


**ABSTRACT**  
In this paper, we explore zero- and few-shot generalization for fact verification (FV), which aims to generalize the FV model trained on well-resourced domains (e.g., Wikipedia) to low-resourced domains that lack human annotations. To this end, we first construct a benchmark dataset collection which contains 11 FV datasets representing 6 domains. We conduct an empirical analysis of generalization across these FV datasets, finding that current models generalize poorly. Our analysis reveals that several factors affect generalization, including dataset size, length of evidence, and the type of claims. Finally, we show that two directions of work improve generalization: 1) incorporating domain knowledge via pretraining on specialized domains, and 2) automatically generating training data via claim generation.

{{</citation>}}


## cs.HC (7)



### (54/154) Automated Interviewer or Augmented Survey? Collecting Social Data with Large Language Models (Alejandro Cuevas Villalba et al., 2023)

{{<citation>}}

Alejandro Cuevas Villalba, Eva M. Brown, Jennifer V. Scurrell, Jason Entenmann, Madeleine I. G. Daepp. (2023)  
**Automated Interviewer or Augmented Survey? Collecting Social Data with Large Language Models**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.10187v1)  

---


**ABSTRACT**  
Qualitative methods like interviews produce richer data in comparison with quantitative surveys, but are difficult to scale. Switching from web-based questionnaires to interactive chatbots offers a compromise, improving user engagement and response quality. Uptake remains limited, however, because of differences in users' expectations versus the capabilities of natural language processing methods. In this study, we evaluate the potential of large language models (LLMs) to support an information elicitation chatbot that narrows this "gulf of expectations" (Luger & Sellen 2016). We conduct a user study in which participants (N = 399) were randomly assigned to interact with a rule-based chatbot versus one of two LLM-augmented chatbots. We observe limited evidence of differences in user engagement or response richness between conditions. However, the addition of LLM-based dynamic probing skills produces significant improvements in both quantitative and qualitative measures of user experience, consistent with a narrowing of the expectations gulf.

{{</citation>}}


### (55/154) How Do Data Analysts Respond to AI Assistance? A Wizard-of-Oz Study (Ken Gu et al., 2023)

{{<citation>}}

Ken Gu, Madeleine Grunde-McLaughlin, Andrew M. McNutt, Jeffrey Heer, Tim Althoff. (2023)  
**How Do Data Analysts Respond to AI Assistance? A Wizard-of-Oz Study**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10108v1)  

---


**ABSTRACT**  
Data analysis is challenging as analysts must navigate nuanced decisions that may yield divergent conclusions. AI assistants have the potential to support analysts in planning their analyses, enabling more robust decision-making. Though AI-based assistants that target code execution (e.g., Github Copilot) have received significant attention, limited research addresses assistance for both analysis execution and planning. In this work, we characterize helpful planning suggestions and their impacts on analysts' workflows. We first review the analysis planning literature and crowd-sourced analysis studies to categorize suggestion content. We then conduct a Wizard-of-Oz study (n=13) to observe analysts' preferences and reactions to planning assistance in a realistic scenario. Our findings highlight subtleties in contextual factors that impact suggestion helpfulness, emphasizing design implications for supporting different abstractions of assistance, forms of initiative, increased engagement, and alignment of goals between analysts and assistants.

{{</citation>}}


### (56/154) Data Formulator: AI-powered Concept-driven Visualization Authoring (Chenglong Wang et al., 2023)

{{<citation>}}

Chenglong Wang, John Thompson, Bongshin Lee. (2023)  
**Data Formulator: AI-powered Concept-driven Visualization Authoring**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10094v1)  

---


**ABSTRACT**  
With most modern visualization tools, authors need to transform their data into tidy formats to create visualizations they want. Because this requires experience with programming or separate data processing tools, data transformation remains a barrier in visualization authoring. To address this challenge, we present a new visualization paradigm, concept binding, that separates high-level visualization intents and low-level data transformation steps, leveraging an AI agent. We realize this paradigm in Data Formulator, an interactive visualization authoring tool. With Data Formulator, authors first define data concepts they plan to visualize using natural languages or examples, and then bind them to visual channels. Data Formulator then dispatches its AI-agent to automatically transform the input data to surface these concepts and generate desired visualizations. When presenting the results (transformed table and output visualizations) from the AI agent, Data Formulator provides feedback to help authors inspect and understand them. A user study with 10 participants shows that participants could learn and use Data Formulator to create visualizations that involve challenging data transformations, and presents interesting future research directions.

{{</citation>}}


### (57/154) What does ChatGPT know about natural science and engineering? (Lukas Schulze Balhorn et al., 2023)

{{<citation>}}

Lukas Schulze Balhorn, Jana M. Weber, Stefan Buijsman, Julian R. Hildebrandt, Martina Ziefle, Artur M. Schweidtmann. (2023)  
**What does ChatGPT know about natural science and engineering?**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.10048v1)  

---


**ABSTRACT**  
ChatGPT is a powerful language model from OpenAI that is arguably able to comprehend and generate text. ChatGPT is expected to have a large impact on society, research, and education. An essential step to understand ChatGPT's expected impact is to study its domain-specific answering capabilities. Here, we perform a systematic empirical assessment of its abilities to answer questions across the natural science and engineering domains. We collected 594 questions from 198 faculty members across 5 faculties at Delft University of Technology. After collecting the answers from ChatGPT, the participants assessed the quality of the answers using a systematic scheme. Our results show that the answers from ChatGPT are on average perceived as ``mostly correct''. Two major trends are that the rating of the ChatGPT answers significantly decreases (i) as the complexity level of the question increases and (ii) as we evaluate skills beyond scientific knowledge, e.g., critical attitude.

{{</citation>}}


### (58/154) Two Decades of Empirical Research on Trust in AI: A Bibliometric Analysis and HCI Research Agenda (Michaela Benk et al., 2023)

{{<citation>}}

Michaela Benk, Sophie Kerstan, Florian v. Wangenheim, Andrea Ferrario. (2023)  
**Two Decades of Empirical Research on Trust in AI: A Bibliometric Analysis and HCI Research Agenda**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09828v1)  

---


**ABSTRACT**  
Trust is widely regarded as a critical component to build artificial intelligence (AI) systems that people will use and safely rely upon. As research in this area continues to evolve, it becomes imperative that the HCI research community synchronize their empirical efforts and align on the path toward effective knowledge creation. To lay the groundwork toward achieving this objective, we performed a comprehensive bibliometric analysis of two decades of empirical research measuring trust in AI, comprising 538 core articles and 15'548 cited articles across multiple disciplines. A key insight arising from our analysis is the persistence of an exploratory approach across the research landscape. To foster a deeper understanding of trust in AI, we advocate for a contextualized strategy. To pave the way, we outline a research agenda, highlighting questions that require further investigation.

{{</citation>}}


### (59/154) Temporal Analysis of Dark Patterns: A Case Study of a User's Odyssey to Conquer Prime Membership Cancellation through the 'Iliad Flow' (Colin M. Gray et al., 2023)

{{<citation>}}

Colin M. Gray, Thomas Mildner, Nataliia Bielova. (2023)  
**Temporal Analysis of Dark Patterns: A Case Study of a User's Odyssey to Conquer Prime Membership Cancellation through the 'Iliad Flow'**  

---
Primary Category: cs.HC  
Categories: cs-CY, cs-HC, cs.HC  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2309.09635v1)  

---


**ABSTRACT**  
Dark patterns are ubiquitous in digital systems, impacting users throughout their journeys on many popular apps and websites. While substantial efforts from the research community in the last five years have led to consolidated taxonomies of dark patterns, including an emerging ontology, most applications of these descriptors have been focused on analysis of static images or as isolated pattern types. In this paper, we present a case study of Amazon Prime's "Iliad Flow" to illustrate the interplay of dark patterns across a user journey, grounded in insights from a US Federal Trade Commission complaint against the company. We use this case study to lay the groundwork for a methodology of Temporal Analysis of Dark Patterns (TADP), including considerations for characterization of individual dark patterns across a user journey, combinatorial effects of multiple dark patterns types, and implications for expert detection and automated detection.

{{</citation>}}


### (60/154) PwR: Exploring the Role of Representations in Conversational Programming (Pradyumna YM et al., 2023)

{{<citation>}}

Pradyumna YM, Vinod Ganesan, Dinesh Kumar Arumugam, Meghna Gupta, Nischith Shadagopan, Tanay Dixit, Sameer Segal, Pratyush Kumar, Mohit Jain, Sriram Rajamani. (2023)  
**PwR: Exploring the Role of Representations in Conversational Programming**  

---
Primary Category: cs.HC  
Categories: H-5-2, cs-HC, cs-SE, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.09495v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have revolutionized programming and software engineering. AI programming assistants such as GitHub Copilot X enable conversational programming, narrowing the gap between human intent and code generation. However, prior literature has identified a key challenge--there is a gap between user's mental model of the system's understanding after a sequence of natural language utterances, and the AI system's actual understanding. To address this, we introduce Programming with Representations (PwR), an approach that uses representations to convey the system's understanding back to the user in natural language. We conducted an in-lab task-centered study with 14 users of varying programming proficiency and found that representations significantly improve understandability, and instilled a sense of agency among our participants. Expert programmers use them for verification, while intermediate programmers benefit from confirmation. Natural language-based development with LLMs, coupled with representations, promises to transform software development, making it more accessible and efficient.

{{</citation>}}


## cs.LG (23)



### (61/154) Graph-enabled Reinforcement Learning for Time Series Forecasting with Adaptive Intelligence (Thanveer Shaik et al., 2023)

{{<citation>}}

Thanveer Shaik, Xiaohui Tao, Haoran Xie, Lin Li, Jianming Yong, Yuefeng Li. (2023)  
**Graph-enabled Reinforcement Learning for Time Series Forecasting with Adaptive Intelligence**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, LSTM, Reinforcement Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2309.10186v1)  

---


**ABSTRACT**  
Reinforcement learning is well known for its ability to model sequential tasks and learn latent data patterns adaptively. Deep learning models have been widely explored and adopted in regression and classification tasks. However, deep learning has its limitations such as the assumption of equally spaced and ordered data, and the lack of ability to incorporate graph structure in terms of time-series prediction. Graphical neural network (GNN) has the ability to overcome these challenges and capture the temporal dependencies in time-series data. In this study, we propose a novel approach for predicting time-series data using GNN and monitoring with Reinforcement Learning (RL). GNNs are able to explicitly incorporate the graph structure of the data into the model, allowing them to capture temporal dependencies in a more natural way. This approach allows for more accurate predictions in complex temporal structures, such as those found in healthcare, traffic and weather forecasting. We also fine-tune our GraphRL model using a Bayesian optimisation technique to further improve performance. The proposed framework outperforms the baseline models in time-series forecasting and monitoring. The contributions of this study include the introduction of a novel GraphRL framework for time-series prediction and the demonstration of the effectiveness of GNNs in comparison to traditional deep learning models such as RNNs and LSTMs. Overall, this study demonstrates the potential of GraphRL in providing accurate and efficient predictions in dynamic RL environments.

{{</citation>}}


### (62/154) Analysis of the Memorization and Generalization Capabilities of AI Agents: Are Continual Learners Robust? (Minsu Kim et al., 2023)

{{<citation>}}

Minsu Kim, Walid Saad. (2023)  
**Analysis of the Memorization and Generalization Capabilities of AI Agents: Are Continual Learners Robust?**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10149v1)  

---


**ABSTRACT**  
In continual learning (CL), an AI agent (e.g., autonomous vehicles or robotics) learns from non-stationary data streams under dynamic environments. For the practical deployment of such applications, it is important to guarantee robustness to unseen environments while maintaining past experiences. In this paper, a novel CL framework is proposed to achieve robust generalization to dynamic environments while retaining past knowledge. The considered CL agent uses a capacity-limited memory to save previously observed environmental information to mitigate forgetting issues. Then, data points are sampled from the memory to estimate the distribution of risks over environmental change so as to obtain predictors that are robust with unseen changes. The generalization and memorization performance of the proposed framework are theoretically analyzed. This analysis showcases the tradeoff between memorization and generalization with the memory size. Experiments show that the proposed algorithm outperforms memory-based CL baselines across all environments while significantly improving the generalization performance on unseen target environments.

{{</citation>}}


### (63/154) Efficient Low-Rank GNN Defense Against Structural Attacks (Abdullah Alchihabi et al., 2023)

{{<citation>}}

Abdullah Alchihabi, Qing En, Yuhong Guo. (2023)  
**Efficient Low-Rank GNN Defense Against Structural Attacks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.10136v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have been shown to possess strong representation abilities over graph data. However, GNNs are vulnerable to adversarial attacks, and even minor perturbations to the graph structure can significantly degrade their performance. Existing methods either are ineffective against sophisticated attacks or require the optimization of dense adjacency matrices, which is time-consuming and prone to local minima. To remedy this problem, we propose an Efficient Low-Rank Graph Neural Network (ELR-GNN) defense method, which aims to learn low-rank and sparse graph structures for defending against adversarial attacks, ensuring effective defense with greater efficiency. Specifically, ELR-GNN consists of two modules: a Coarse Low-Rank Estimation Module and a Fine-Grained Estimation Module. The first module adopts the truncated Singular Value Decomposition (SVD) to initialize the low-rank adjacency matrix estimation, which serves as a starting point for optimizing the low-rank matrix. In the second module, the initial estimate is refined by jointly learning a low-rank sparse graph structure with the GNN model. Sparsity is incorporated into the learned low-rank adjacency matrix by pruning weak connections, which can reduce redundant data while maintaining valuable information. As a result, instead of using the dense adjacency matrix directly, ELR-GNN can learn a low-rank and sparse estimate of it in a simple, efficient and easy to optimize manner. The experimental results demonstrate that ELR-GNN outperforms the state-of-the-art GNN defense methods in the literature, in addition to being very efficient and easy to train.

{{</citation>}}


### (64/154) GDM: Dual Mixup for Graph Classification with Limited Supervision (Abdullah Alchihabi et al., 2023)

{{<citation>}}

Abdullah Alchihabi, Yuhong Guo. (2023)  
**GDM: Dual Mixup for Graph Classification with Limited Supervision**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.10134v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) require a large number of labeled graph samples to obtain good performance on the graph classification task. The performance of GNNs degrades significantly as the number of labeled graph samples decreases. To reduce the annotation cost, it is therefore important to develop graph augmentation methods that can generate new graph instances to increase the size and diversity of the limited set of available labeled graph samples. In this work, we propose a novel mixup-based graph augmentation method, Graph Dual Mixup (GDM), that leverages both functional and structural information of the graph instances to generate new labeled graph samples. GDM employs a graph structural auto-encoder to learn structural embeddings of the graph samples, and then applies mixup to the structural information of the graphs in the learned structural embedding space and generates new graph structures from the mixup structural embeddings. As for the functional information, GDM applies mixup directly to the input node features of the graph samples to generate functional node feature information for new mixup graph instances. Jointly, the generated input node features and graph structures yield new graph samples which can supplement the set of original labeled graphs. Furthermore, we propose two novel Balanced Graph Sampling methods to enhance the balanced difficulty and diversity for the generated graph samples. Experimental results on the benchmark datasets demonstrate that our proposed method substantially outperforms the state-of-the-art graph augmentation methods when the labeled graphs are scarce.

{{</citation>}}


### (65/154) Deep Prompt Tuning for Graph Transformers (Reza Shirkavand et al., 2023)

{{<citation>}}

Reza Shirkavand, Heng Huang. (2023)  
**Deep Prompt Tuning for Graph Transformers**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.10131v1)  

---


**ABSTRACT**  
Graph transformers have gained popularity in various graph-based tasks by addressing challenges faced by traditional Graph Neural Networks. However, the quadratic complexity of self-attention operations and the extensive layering in graph transformer architectures present challenges when applying them to graph based prediction tasks. Fine-tuning, a common approach, is resource-intensive and requires storing multiple copies of large models. We propose a novel approach called deep graph prompt tuning as an alternative to fine-tuning for leveraging large graph transformer models in downstream graph based prediction tasks. Our method introduces trainable feature nodes to the graph and pre-pends task-specific tokens to the graph transformer, enhancing the model's expressive power. By freezing the pre-trained parameters and only updating the added tokens, our approach reduces the number of free parameters and eliminates the need for multiple model copies, making it suitable for small datasets and scalable to large graphs. Through extensive experiments on various-sized datasets, we demonstrate that deep graph prompt tuning achieves comparable or even superior performance to fine-tuning, despite utilizing significantly fewer task-specific parameters. Our contributions include the introduction of prompt tuning for graph transformers, its application to both graph transformers and message passing graph neural networks, improved efficiency and resource utilization, and compelling experimental results. This work brings attention to a promising approach to leverage pre-trained models in graph based prediction tasks and offers new opportunities for exploring and advancing graph representation learning.

{{</citation>}}


### (66/154) A Semi-Supervised Approach for Power System Event Identification (Nima Taghipourbazargani et al., 2023)

{{<citation>}}

Nima Taghipourbazargani, Lalitha Sankar, Oliver Kosut. (2023)  
**A Semi-Supervised Approach for Power System Event Identification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.10095v1)  

---


**ABSTRACT**  
Event identification is increasingly recognized as crucial for enhancing the reliability, security, and stability of the electric power system. With the growing deployment of Phasor Measurement Units (PMUs) and advancements in data science, there are promising opportunities to explore data-driven event identification via machine learning classification techniques. However, obtaining accurately-labeled eventful PMU data samples remains challenging due to its labor-intensive nature and uncertainty about the event type (class) in real-time. Thus, it is natural to use semi-supervised learning techniques, which make use of both labeled and unlabeled samples. %We propose a novel semi-supervised framework to assess the effectiveness of incorporating unlabeled eventful samples to enhance existing event identification methodologies. We evaluate three categories of classical semi-supervised approaches: (i) self-training, (ii) transductive support vector machines (TSVM), and (iii) graph-based label spreading (LS) method. Our approach characterizes events using physically interpretable features extracted from modal analysis of synthetic eventful PMU data. In particular, we focus on the identification of four event classes whose identification is crucial for grid operations. We have developed and publicly shared a comprehensive Event Identification package which consists of three aspects: data generation, feature extraction, and event identification with limited labels using semi-supervised methodologies. Using this package, we generate and evaluate eventful PMU data for the South Carolina synthetic network. Our evaluation consistently demonstrates that graph-based LS outperforms the other two semi-supervised methods that we consider, and can noticeably improve event identification performance relative to the setting with only a small number of labeled samples.

{{</citation>}}


### (67/154) GAME: Generalized deep learning model towards multimodal data integration for early screening of adolescent mental disorders (Zhicheng Du et al., 2023)

{{<citation>}}

Zhicheng Du, Chenyao Jiang, Xi Yuan, Shiyao Zhai, Zhengyang Lei, Shuyue Ma, Yang Liu, Qihui Ye, Chufan Xiao, Qiming Huang, Ming Xu, Dongmei Yu, Peiwu Qin. (2023)  
**GAME: Generalized deep learning model towards multimodal data integration for early screening of adolescent mental disorders**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.10077v1)  

---


**ABSTRACT**  
The timely identification of mental disorders in adolescents is a global public health challenge.Single factor is difficult to detect the abnormality due to its complex and subtle nature. Additionally, the generalized multimodal Computer-Aided Screening (CAS) systems with interactive robots for adolescent mental disorders are not available. Here, we design an android application with mini-games and chat recording deployed in a portable robot to screen 3,783 middle school students and construct the multimodal screening dataset, including facial images, physiological signs, voice recordings, and textual transcripts.We develop a model called GAME (Generalized Model with Attention and Multimodal EmbraceNet) with novel attention mechanism that integrates cross-modal features into the model. GAME evaluates adolescent mental conditions with high accuracy (73.34%-92.77%) and F1-Score (71.32%-91.06%).We find each modality contributes dynamically to the mental disorders screening and comorbidities among various mental disorders, indicating the feasibility of explainable model. This study provides a system capable of acquiring multimodal information and constructs a generalized multimodal integration algorithm with novel attention mechanisms for the early screening of adolescent mental disorders.

{{</citation>}}


### (68/154) Actively Learning Reinforcement Learning: A Stochastic Optimal Control Approach (Mohammad S. Ramadan et al., 2023)

{{<citation>}}

Mohammad S. Ramadan, Mahmoud A. Hayajnh, Michael T. Tolley, Kyriakos G. Vamvoudakis. (2023)  
**Actively Learning Reinforcement Learning: A Stochastic Optimal Control Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.10831v1)  

---


**ABSTRACT**  
In this paper we provide framework to cope with two problems: (i) the fragility of reinforcement learning due to modeling uncertainties because of the mismatch between controlled laboratory/simulation and real-world conditions and (ii) the prohibitive computational cost of stochastic optimal control. We approach both problems by using reinforcement learning to solve the stochastic dynamic programming equation. The resulting reinforcement learning controller is safe with respect to several types of constraints constraints and it can actively learn about the modeling uncertainties. Unlike exploration and exploitation, probing and safety are employed automatically by the controller itself, resulting real-time learning. A simulation example demonstrates the efficacy of the proposed approach.

{{</citation>}}


### (69/154) Empirical Study of Mix-based Data Augmentation Methods in Physiological Time Series Data (Peikun Guo et al., 2023)

{{<citation>}}

Peikun Guo, Huiyuan Yang, Akane Sano. (2023)  
**Empirical Study of Mix-based Data Augmentation Methods in Physiological Time Series Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Augmentation, Time Series  
[Paper Link](http://arxiv.org/abs/2309.09970v1)  

---


**ABSTRACT**  
Data augmentation is a common practice to help generalization in the procedure of deep model training. In the context of physiological time series classification, previous research has primarily focused on label-invariant data augmentation methods. However, another class of augmentation techniques (\textit{i.e., Mixup}) that emerged in the computer vision field has yet to be fully explored in the time series domain. In this study, we systematically review the mix-based augmentations, including mixup, cutmix, and manifold mixup, on six physiological datasets, evaluating their performance across different sensory data and classification tasks. Our results demonstrate that the three mix-based augmentations can consistently improve the performance on the six datasets. More importantly, the improvement does not rely on expert knowledge or extensive parameter tuning. Lastly, we provide an overview of the unique properties of the mix-based augmentation methods and highlight the potential benefits of using the mix-based augmentation in physiological time series data.

{{</citation>}}


### (70/154) Evaluation of GPT-3 for Anti-Cancer Drug Sensitivity Prediction (Shaika Chowdhury et al., 2023)

{{<citation>}}

Shaika Chowdhury, Sivaraman Rajaganapathy, Lichao Sun, James Cerhan, Nansu Zong. (2023)  
**Evaluation of GPT-3 for Anti-Cancer Drug Sensitivity Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2309.10016v1)  

---


**ABSTRACT**  
In this study, we investigated the potential of GPT-3 for the anti-cancer drug sensitivity prediction task using structured pharmacogenomics data across five tissue types and evaluated its performance with zero-shot prompting and fine-tuning paradigms. The drug's smile representation and cell line's genomic mutation features were predictive of the drug response. The results from this study have the potential to pave the way for designing more efficient treatment protocols in precision oncology.

{{</citation>}}


### (71/154) Context is Environment (Sharut Gupta et al., 2023)

{{<citation>}}

Sharut Gupta, Stefanie Jegelka, David Lopez-Paz, Kartik Ahuja. (2023)  
**Context is Environment**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09888v2)  

---


**ABSTRACT**  
Two lines of work are taking the central stage in AI research. On the one hand, the community is making increasing efforts to build models that discard spurious correlations and generalize better in novel test environments. Unfortunately, the bitter lesson so far is that no proposal convincingly outperforms a simple empirical risk minimization baseline. On the other hand, large language models (LLMs) have erupted as algorithms able to learn in-context, generalizing on-the-fly to eclectic contextual circumstances that users enforce by means of prompting. In this paper, we argue that context is environment, and posit that in-context learning holds the key to better domain generalization. Via extensive theory and experiments, we show that paying attention to context$\unicode{x2013}\unicode{x2013}$unlabeled examples as they arrive$\unicode{x2013}\unicode{x2013}$allows our proposed In-Context Risk Minimization (ICRM) algorithm to zoom-in on the test environment risk minimizer, leading to significant out-of-distribution performance improvements. From all of this, two messages are worth taking home. Researchers in domain generalization should consider environment as context, and harness the adaptive power of in-context learning. Researchers in LLMs should consider context as environment, to better structure data towards generalization.

{{</citation>}}


### (72/154) Deep Reinforcement Learning for the Joint Control of Traffic Light Signaling and Vehicle Speed Advice (Johannes V. S. Busch et al., 2023)

{{<citation>}}

Johannes V. S. Busch, Robert Voelckner, Peter Sossalla, Christian L. Vielhaus, Roberto Calandra, Frank H. P. Fitzek. (2023)  
**Deep Reinforcement Learning for the Joint Control of Traffic Light Signaling and Vehicle Speed Advice**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.09881v1)  

---


**ABSTRACT**  
Traffic congestion in dense urban centers presents an economical and environmental burden. In recent years, the availability of vehicle-to-anything communication allows for the transmission of detailed vehicle states to the infrastructure that can be used for intelligent traffic light control. The other way around, the infrastructure can provide vehicles with advice on driving behavior, such as appropriate velocities, which can improve the efficacy of the traffic system. Several research works applied deep reinforcement learning to either traffic light control or vehicle speed advice. In this work, we propose a first attempt to jointly learn the control of both. We show this to improve the efficacy of traffic systems. In our experiments, the joint control approach reduces average vehicle trip delays, w.r.t. controlling only traffic lights, in eight out of eleven benchmark scenarios. Analyzing the qualitative behavior of the vehicle speed advice policy, we observe that this is achieved by smoothing out the velocity profile of vehicles nearby a traffic light. Learning joint control of traffic signaling and speed advice in the real world could help to reduce congestion and mitigate the economical and environmental repercussions of today's traffic systems.

{{</citation>}}


### (73/154) Prognosis of Multivariate Battery State of Performance and Health via Transformers (Noah H. Paulson et al., 2023)

{{<citation>}}

Noah H. Paulson, Joseph J. Kubal, Susan J. Babinec. (2023)  
**Prognosis of Multivariate Battery State of Performance and Health via Transformers**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.10014v1)  

---


**ABSTRACT**  
Batteries are an essential component in a deeply decarbonized future. Understanding battery performance and "useful life" as a function of design and use is of paramount importance to accelerating adoption. Historically, battery state of health (SOH) was summarized by a single parameter, the fraction of a battery's capacity relative to its initial state. A more useful approach, however, is a comprehensive characterization of its state and complexities, using an interrelated set of descriptors including capacity, energy, ionic and electronic impedances, open circuit voltages, and microstructure metrics. Indeed, predicting across an extensive suite of properties as a function of battery use is a "holy grail" of battery science; it can provide unprecedented insights toward the design of better batteries with reduced experimental effort, and de-risking energy storage investments that are necessary to meet CO2 reduction targets. In this work, we present a first step in that direction via deep transformer networks for the prediction of 28 battery state of health descriptors using two cycling datasets representing six lithium-ion cathode chemistries (LFP, NMC111, NMC532, NMC622, HE5050, and 5Vspinel), multiple electrolyte/anode compositions, and different charge-discharge scenarios. The accuracy of these predictions versus battery life (with an unprecedented mean absolute error of 19 cycles in predicting end of life for an LFP fast-charging dataset) illustrates the promise of deep learning towards providing deeper understanding and control of battery health.

{{</citation>}}


### (74/154) Towards Self-Adaptive Pseudo-Label Filtering for Semi-Supervised Learning (Lei Zhu et al., 2023)

{{<citation>}}

Lei Zhu, Zhanghan Ke, Rynson Lau. (2023)  
**Towards Self-Adaptive Pseudo-Label Filtering for Semi-Supervised Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.09774v1)  

---


**ABSTRACT**  
Recent semi-supervised learning (SSL) methods typically include a filtering strategy to improve the quality of pseudo labels. However, these filtering strategies are usually hand-crafted and do not change as the model is updated, resulting in a lot of correct pseudo labels being discarded and incorrect pseudo labels being selected during the training process. In this work, we observe that the distribution gap between the confidence values of correct and incorrect pseudo labels emerges at the very beginning of the training, which can be utilized to filter pseudo labels. Based on this observation, we propose a Self-Adaptive Pseudo-Label Filter (SPF), which automatically filters noise in pseudo labels in accordance with model evolvement by modeling the confidence distribution throughout the training process. Specifically, with an online mixture model, we weight each pseudo-labeled sample by the posterior of it being correct, which takes into consideration the confidence distribution at that time. Unlike previous handcrafted filters, our SPF evolves together with the deep neural network without manual tuning. Extensive experiments demonstrate that incorporating SPF into the existing SSL methods can help improve the performance of SSL, especially when the labeled data is extremely scarce.

{{</citation>}}


### (75/154) Contrastive Initial State Buffer for Reinforcement Learning (Nico Messikommer et al., 2023)

{{<citation>}}

Nico Messikommer, Yunlong Song, Davide Scaramuzza. (2023)  
**Contrastive Initial State Buffer for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.09752v2)  

---


**ABSTRACT**  
In Reinforcement Learning, the trade-off between exploration and exploitation poses a complex challenge for achieving efficient learning from limited samples. While recent works have been effective in leveraging past experiences for policy updates, they often overlook the potential of reusing past experiences for data collection. Independent of the underlying RL algorithm, we introduce the concept of a Contrastive Initial State Buffer, which strategically selects states from past experiences and uses them to initialize the agent in the environment in order to guide it toward more informative states. We validate our approach on two complex robotic tasks without relying on any prior information about the environment: (i) locomotion of a quadruped robot traversing challenging terrains and (ii) a quadcopter drone racing through a track. The experimental results show that our initial state buffer achieves higher task performance than the nominal baseline while also speeding up training convergence.

{{</citation>}}


### (76/154) Towards Better Modeling with Missing Data: A Contrastive Learning-based Visual Analytics Perspective (Laixin Xie et al., 2023)

{{<citation>}}

Laixin Xie, Yang Ouyang, Longfei Chen, Ziming Wu, Quan Li. (2023)  
**Towards Better Modeling with Missing Data: A Contrastive Learning-based Visual Analytics Perspective**  

---
Primary Category: cs.LG  
Categories: I-1-2; H-1-2; H-4-2, cs-HC, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.09744v1)  

---


**ABSTRACT**  
Missing data can pose a challenge for machine learning (ML) modeling. To address this, current approaches are categorized into feature imputation and label prediction and are primarily focused on handling missing data to enhance ML performance. These approaches rely on the observed data to estimate the missing values and therefore encounter three main shortcomings in imputation, including the need for different imputation methods for various missing data mechanisms, heavy dependence on the assumption of data distribution, and potential introduction of bias. This study proposes a Contrastive Learning (CL) framework to model observed data with missing values, where the ML model learns the similarity between an incomplete sample and its complete counterpart and the dissimilarity between other samples. Our proposed approach demonstrates the advantages of CL without requiring any imputation. To enhance interpretability, we introduce CIVis, a visual analytics system that incorporates interpretable techniques to visualize the learning process and diagnose the model status. Users can leverage their domain knowledge through interactive sampling to identify negative and positive pairs in CL. The output of CIVis is an optimized model that takes specified features and predicts downstream tasks. We provide two usage scenarios in regression and classification tasks and conduct quantitative experiments, expert interviews, and a qualitative user study to demonstrate the effectiveness of our approach. In short, this study offers a valuable contribution to addressing the challenges associated with ML modeling in the presence of missing data by providing a practical solution that achieves high predictive accuracy and model interpretability.

{{</citation>}}


### (77/154) Contrastive Learning and Data Augmentation in Traffic Classification Using a Flowpic Input Representation (Alessandro Finamore et al., 2023)

{{<citation>}}

Alessandro Finamore, Chao Wang, Jonatan Krolikowski, Jose M. Navarro, Fuxing Chen, Dario Rossi. (2023)  
**Contrastive Learning and Data Augmentation in Traffic Classification Using a Flowpic Input Representation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NI, cs.LG  
Keywords: Augmentation, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.09733v1)  

---


**ABSTRACT**  
Over the last years we witnessed a renewed interest towards Traffic Classification (TC) captivated by the rise of Deep Learning (DL). Yet, the vast majority of TC literature lacks code artifacts, performance assessments across datasets and reference comparisons against Machine Learning (ML) methods. Among those works, a recent study from IMC'22 [17] is worth of attention since it adopts recent DL methodologies (namely, few-shot learning, self-supervision via contrastive learning and data augmentation) appealing for networking as they enable to learn from a few samples and transfer across datasets. The main result of [17] on the UCDAVIS19, ISCX-VPN and ISCX-Tor datasets is that, with such DL methodologies, 100 input samples are enough to achieve very high accuracy using an input representation called "flowpic" (i.e., a per-flow 2d histograms of the packets size evolution over time). In this paper (i) we reproduce [17] on the same datasets and (ii) we replicate its most salient aspect (the importance of data augmentation) on three additional public datasets, MIRAGE-19, MIRAGE-22 and UTMOBILENET21. While we confirm most of the original results, we also found a 20% accuracy drop on some of the investigated scenarios due to a data shift in the original dataset that we uncovered. Additionally, our study validates that the data augmentation strategies studied in [17] perform well on other datasets too. In the spirit of reproducibility and replicability we make all artifacts (code and data) available at [10].

{{</citation>}}


### (78/154) Traffic Scene Similarity: a Graph-based Contrastive Learning Approach (Maximilian Zipfl et al., 2023)

{{<citation>}}

Maximilian Zipfl, Moritz Jarosch, J. Marius Zöllner. (2023)  
**Traffic Scene Similarity: a Graph-based Contrastive Learning Approach**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.09720v1)  

---


**ABSTRACT**  
Ensuring validation for highly automated driving poses significant obstacles to the widespread adoption of highly automated vehicles. Scenario-based testing offers a potential solution by reducing the homologation effort required for these systems. However, a crucial prerequisite, yet unresolved, is the definition and reduction of the test space to a finite number of scenarios. To tackle this challenge, we propose an extension to a contrastive learning approach utilizing graphs to construct a meaningful embedding space. Our approach demonstrates the continuous mapping of scenes using scene-specific features and the formation of thematically similar clusters based on the resulting embeddings. Based on the found clusters, similar scenes could be identified in the subsequent test process, which can lead to a reduction in redundant test runs.

{{</citation>}}


### (79/154) FedLALR: Client-Specific Adaptive Learning Rates Achieve Linear Speedup for Non-IID Data (Hao Sun et al., 2023)

{{<citation>}}

Hao Sun, Li Shen, Shixiang Chen, Jingwei Sun, Jing Li, Guangzhong Sun, Dacheng Tao. (2023)  
**FedLALR: Client-Specific Adaptive Learning Rates Achieve Linear Speedup for Non-IID Data**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG, math-OC  
Keywords: Computer Vision, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.09719v1)  

---


**ABSTRACT**  
Federated learning is an emerging distributed machine learning method, enables a large number of clients to train a model without exchanging their local data. The time cost of communication is an essential bottleneck in federated learning, especially for training large-scale deep neural networks. Some communication-efficient federated learning methods, such as FedAvg and FedAdam, share the same learning rate across different clients. But they are not efficient when data is heterogeneous. To maximize the performance of optimization methods, the main challenge is how to adjust the learning rate without hurting the convergence. In this paper, we propose a heterogeneous local variant of AMSGrad, named FedLALR, in which each client adjusts its learning rate based on local historical gradient squares and synchronized learning rates. Theoretical analysis shows that our client-specified auto-tuned learning rate scheduling can converge and achieve linear speedup with respect to the number of clients, which enables promising scalability in federated optimization. We also empirically compare our method with several communication-efficient federated optimization methods. Extensive experimental results on Computer Vision (CV) tasks and Natural Language Processing (NLP) task show the efficacy of our proposed FedLALR method and also coincides with our theoretical findings.

{{</citation>}}


### (80/154) Information based explanation methods for deep learning agents -- with applications on large open-source chess models (Patrik Hammersborg et al., 2023)

{{<citation>}}

Patrik Hammersborg, Inga Strümke. (2023)  
**Information based explanation methods for deep learning agents -- with applications on large open-source chess models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09702v1)  

---


**ABSTRACT**  
With large chess-playing neural network models like AlphaZero contesting the state of the art within the world of computerised chess, two challenges present themselves: The question of how to explain the domain knowledge internalised by such models, and the problem that such models are not made openly available. This work presents the re-implementation of the concept detection methodology applied to AlphaZero in McGrath et al. (2022), by using large, open-source chess models with comparable performance. We obtain results similar to those achieved on AlphaZero, while relying solely on open-source resources. We also present a novel explainable AI (XAI) method, which is guaranteed to highlight exhaustively and exclusively the information used by the explained model. This method generates visual explanations tailored to domains characterised by discrete input spaces, as is the case for chess. Our presented method has the desirable property of controlling the information flow between any input vector and the given model, which in turn provides strict guarantees regarding what information is used by the trained model during inference. We demonstrate the viability of our method by applying it to standard 8x8 chess, using large open-source chess models.

{{</citation>}}


### (81/154) Latent assimilation with implicit neural representations for unknown dynamics (Zhuoyuan Li et al., 2023)

{{<citation>}}

Zhuoyuan Li, Bin Dong, Pingwen Zhang. (2023)  
**Latent assimilation with implicit neural representations for unknown dynamics**  

---
Primary Category: cs.LG  
Categories: 68T07, 49N45, 33C55, cs-LG, cs.LG, math-MP, math-OC, math-ph, physics-ao-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09574v1)  

---


**ABSTRACT**  
Data assimilation is crucial in a wide range of applications, but it often faces challenges such as high computational costs due to data dimensionality and incomplete understanding of underlying mechanisms. To address these challenges, this study presents a novel assimilation framework, termed Latent Assimilation with Implicit Neural Representations (LAINR). By introducing Spherical Implicit Neural Representations (SINR) along with a data-driven uncertainty estimator of the trained neural networks, LAINR enhances efficiency in assimilation process. Experimental results indicate that LAINR holds certain advantage over existing methods based on AutoEncoders, both in terms of accuracy and efficiency.

{{</citation>}}


### (82/154) FedGKD: Unleashing the Power of Collaboration in Federated Graph Neural Networks (Qiying Pan et al., 2023)

{{<citation>}}

Qiying Pan, Ruofan Wu, Tengfei Liu, Tianyi Zhang, Yifei Zhu, Weiqiang Wang. (2023)  
**FedGKD: Unleashing the Power of Collaboration in Federated Graph Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.09517v3)  

---


**ABSTRACT**  
Federated training of Graph Neural Networks (GNN) has become popular in recent years due to its ability to perform graph-related tasks under data isolation scenarios while preserving data privacy. However, graph heterogeneity issues in federated GNN systems continue to pose challenges. Existing frameworks address the problem by representing local tasks using different statistics and relating them through a simple aggregation mechanism. However, these approaches suffer from limited efficiency from two aspects: low quality of task-relatedness quantification and inefficacy of exploiting the collaboration structure. To address these issues, we propose FedGKD, a novel federated GNN framework that utilizes a novel client-side graph dataset distillation method to extract task features that better describe task-relatedness, and introduces a novel server-side aggregation mechanism that is aware of the global collaboration structure. We conduct extensive experiments on six real-world datasets of different scales, demonstrating our framework's outperformance.

{{</citation>}}


### (83/154) An Iterative Method for Unsupervised Robust Anomaly Detection Under Data Contamination (Minkyung Kim et al., 2023)

{{<citation>}}

Minkyung Kim, Jongmin Yu, Junsik Kim, Tae-Hyun Oh, Jun Kyun Choi. (2023)  
**An Iterative Method for Unsupervised Robust Anomaly Detection Under Data Contamination**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2309.09436v1)  

---


**ABSTRACT**  
Most deep anomaly detection models are based on learning normality from datasets due to the difficulty of defining abnormality by its diverse and inconsistent nature. Therefore, it has been a common practice to learn normality under the assumption that anomalous data are absent in a training dataset, which we call normality assumption. However, in practice, the normality assumption is often violated due to the nature of real data distributions that includes anomalous tails, i.e., a contaminated dataset. Thereby, the gap between the assumption and actual training data affects detrimentally in learning of an anomaly detection model. In this work, we propose a learning framework to reduce this gap and achieve better normality representation. Our key idea is to identify sample-wise normality and utilize it as an importance weight, which is updated iteratively during the training. Our framework is designed to be model-agnostic and hyperparameter insensitive so that it applies to a wide range of existing methods without careful parameter tuning. We apply our framework to three different representative approaches of deep anomaly detection that are classified into one-class classification-, probabilistic model-, and reconstruction-based approaches. In addition, we address the importance of a termination condition for iterative methods and propose a termination criterion inspired by the anomaly detection objective. We validate that our framework improves the robustness of the anomaly detection models under different levels of contamination ratios on five anomaly detection benchmark datasets and two image datasets. On various contaminated datasets, our framework improves the performance of three representative anomaly detection methods, measured by area under the ROC curve.

{{</citation>}}


## cs.NI (2)



### (84/154) Self-Sustaining Multiple Access with Continual Deep Reinforcement Learning for Dynamic Metaverse Applications (Hamidreza Mazandarani et al., 2023)

{{<citation>}}

Hamidreza Mazandarani, Masoud Shokrnezhad, Tarik Taleb, Richard Li. (2023)  
**Self-Sustaining Multiple Access with Continual Deep Reinforcement Learning for Dynamic Metaverse Applications**  

---
Primary Category: cs.NI  
Categories: cs-AI, cs-ET, cs-LG, cs-NA, cs-NI, cs.NI, math-NA  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.10177v1)  

---


**ABSTRACT**  
The Metaverse is a new paradigm that aims to create a virtual environment consisting of numerous worlds, each of which will offer a different set of services. To deal with such a dynamic and complex scenario, considering the stringent quality of service requirements aimed at the 6th generation of communication systems (6G), one potential approach is to adopt self-sustaining strategies, which can be realized by employing Adaptive Artificial Intelligence (Adaptive AI) where models are continually re-trained with new data and conditions. One aspect of self-sustainability is the management of multiple access to the frequency spectrum. Although several innovative methods have been proposed to address this challenge, mostly using Deep Reinforcement Learning (DRL), the problem of adapting agents to a non-stationary environment has not yet been precisely addressed. This paper fills in the gap in the current literature by investigating the problem of multiple access in multi-channel environments to maximize the throughput of the intelligent agent when the number of active User Equipments (UEs) may fluctuate over time. To solve the problem, a Double Deep Q-Learning (DDQL) technique empowered by Continual Learning (CL) is proposed to overcome the non-stationary situation, while the environment is unknown. Numerical simulations demonstrate that, compared to other well-known methods, the CL-DDQL algorithm achieves significantly higher throughputs with a considerably shorter convergence time in highly dynamic scenarios.

{{</citation>}}


### (85/154) Network Traffic Classification Based on External Attention by IP Packet Header (Yahui Hu et al., 2023)

{{<citation>}}

Yahui Hu, Ziqian Zeng, Junping Song, Luyang Xu, Xu Zhou. (2023)  
**Network Traffic Classification Based on External Attention by IP Packet Header**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.09440v1)  

---


**ABSTRACT**  
As the emerging services have increasingly strict requirements on quality of service (QoS), such as millisecond network service latency ect., network traffic classification technology is required to assist more advanced network management and monitoring capabilities. So far as we know, the delays of flow-granularity classification methods are difficult to meet the real-time requirements for too long packet-waiting time, whereas the present packet-granularity classification methods may have problems related to privacy protection due to using excessive user payloads. To solve the above problems, we proposed a network traffic classification method only by the IP packet header, which satisfies the requirements of both user's privacy protection and classification performances. We opted to remove the IP address from the header information of the network layer and utilized the remaining 12-byte IP packet header information as input for the model. Additionally, we examined the variations in header value distributions among different categories of network traffic samples. And, the external attention is also introduced to form the online classification framework, which performs well for its low time complexity and strong ability to enhance high-dimensional classification features. The experiments on three open-source datasets show that our average accuracy can reach upon 94.57%, and the classification time is shortened to meet the real-time requirements (0.35ms for a single packet).

{{</citation>}}


## cs.RO (18)



### (86/154) One ACT Play: Single Demonstration Behavior Cloning with Action Chunking Transformers (Abraham George et al., 2023)

{{<citation>}}

Abraham George, Amir Barati Farimani. (2023)  
**One ACT Play: Single Demonstration Behavior Cloning with Action Chunking Transformers**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.10175v1)  

---


**ABSTRACT**  
Learning from human demonstrations (behavior cloning) is a cornerstone of robot learning. However, most behavior cloning algorithms require a large number of demonstrations to learn a task, especially for general tasks that have a large variety of initial conditions. Humans, however, can learn to complete tasks, even complex ones, after only seeing one or two demonstrations. Our work seeks to emulate this ability, using behavior cloning to learn a task given only a single human demonstration. We achieve this goal by using linear transforms to augment the single demonstration, generating a set of trajectories for a wide range of initial conditions. With these demonstrations, we are able to train a behavior cloning agent to successfully complete three block manipulation tasks. Additionally, we developed a novel addition to the temporal ensembling method used by action chunking agents during inference. By incorporating the standard deviation of the action predictions into the ensembling method, our approach is more robust to unforeseen changes in the environment, resulting in significant performance improvements.

{{</citation>}}


### (87/154) Asynchronous Perception-Action-Communication with Graph Neural Networks (Saurav Agarwal et al., 2023)

{{<citation>}}

Saurav Agarwal, Alejandro Ribeiro, Vijay Kumar. (2023)  
**Asynchronous Perception-Action-Communication with Graph Neural Networks**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.10164v1)  

---


**ABSTRACT**  
Collaboration in large robot swarms to achieve a common global objective is a challenging problem in large environments due to limited sensing and communication capabilities. The robots must execute a Perception-Action-Communication (PAC) loop -- they perceive their local environment, communicate with other robots, and take actions in real time. A fundamental challenge in decentralized PAC systems is to decide what information to communicate with the neighboring robots and how to take actions while utilizing the information shared by the neighbors. Recently, this has been addressed using Graph Neural Networks (GNNs) for applications such as flocking and coverage control. Although conceptually, GNN policies are fully decentralized, the evaluation and deployment of such policies have primarily remained centralized or restrictively decentralized. Furthermore, existing frameworks assume sequential execution of perception and action inference, which is very restrictive in real-world applications. This paper proposes a framework for asynchronous PAC in robot swarms, where decentralized GNNs are used to compute navigation actions and generate messages for communication. In particular, we use aggregated GNNs, which enable the exchange of hidden layer information between robots for computational efficiency and decentralized inference of actions. Furthermore, the modules in the framework are asynchronous, allowing robots to perform sensing, extracting information, communication, action inference, and control execution at different frequencies. We demonstrate the effectiveness of GNNs executed in the proposed framework in navigating large robot swarms for collaborative coverage of large environments.

{{</citation>}}


### (88/154) Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions (Yevgen Chebotar et al., 2023)

{{<citation>}}

Yevgen Chebotar, Quan Vuong, Alex Irpan, Karol Hausman, Fei Xia, Yao Lu, Aviral Kumar, Tianhe Yu, Alexander Herzog, Karl Pertsch, Keerthana Gopalakrishnan, Julian Ibarz, Ofir Nachum, Sumedh Sontakke, Grecia Salazar, Huong T Tran, Jodilyn Peralta, Clayton Tan, Deeksha Manjunath, Jaspiar Singht, Brianna Zitkovich, Tomas Jackson, Kanishka Rao, Chelsea Finn, Sergey Levine. (2023)  
**Q-Transformer: Scalable Offline Reinforcement Learning via Autoregressive Q-Functions**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2309.10150v1)  

---


**ABSTRACT**  
In this work, we present a scalable reinforcement learning method for training multi-task policies from large offline datasets that can leverage both human demonstrations and autonomously collected data. Our method uses a Transformer to provide a scalable representation for Q-functions trained via offline temporal difference backups. We therefore refer to the method as Q-Transformer. By discretizing each action dimension and representing the Q-value of each action dimension as separate tokens, we can apply effective high-capacity sequence modeling techniques for Q-learning. We present several design decisions that enable good performance with offline RL training, and show that Q-Transformer outperforms prior offline RL algorithms and imitation learning techniques on a large diverse real-world robotic manipulation task suite. The project's website and videos can be found at https://q-transformer.github.io

{{</citation>}}


### (89/154) Reasoning about the Unseen for Efficient Outdoor Object Navigation (Quanting Xie et al., 2023)

{{<citation>}}

Quanting Xie, Tianyi Zhang, Kedi Xu, Matthew Johnson-Roberson, Yonatan Bisk. (2023)  
**Reasoning about the Unseen for Efficient Outdoor Object Navigation**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.10103v1)  

---


**ABSTRACT**  
Robots should exist anywhere humans do: indoors, outdoors, and even unmapped environments. In contrast, the focus of recent advancements in Object Goal Navigation(OGN) has targeted navigating in indoor environments by leveraging spatial and semantic cues that do not generalize outdoors. While these contributions provide valuable insights into indoor scenarios, the broader spectrum of real-world robotic applications often extends to outdoor settings. As we transition to the vast and complex terrains of outdoor environments, new challenges emerge. Unlike the structured layouts found indoors, outdoor environments lack clear spatial delineations and are riddled with inherent semantic ambiguities. Despite this, humans navigate with ease because we can reason about the unseen. We introduce a new task OUTDOOR, a new mechanism for Large Language Models (LLMs) to accurately hallucinate possible futures, and a new computationally aware success metric for pushing research forward in this more complex domain. Additionally, we show impressive results on both a simulated drone and physical quadruped in outdoor environments. Our agent has no premapping and our formalism outperforms naive LLM-based approaches

{{</citation>}}


### (90/154) Conformal Temporal Logic Planning using Large Language Models: Knowing When to Do What and When to Ask for Help (Jun Wang et al., 2023)

{{<citation>}}

Jun Wang, Jiaming Tong, Kaiyuan Tan, Yevgeniy Vorobeychik, Yiannis Kantaros. (2023)  
**Conformal Temporal Logic Planning using Large Language Models: Knowing When to Do What and When to Ask for Help**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.10092v1)  

---


**ABSTRACT**  
This paper addresses a new motion planning problem for mobile robots tasked with accomplishing multiple high-level sub-tasks, expressed using natural language (NL), in a temporal and logical order. To formally define such missions, we leverage LTL defined over NL-based atomic predicates modeling the considered NL-based sub-tasks. This is contrast to related planning approaches that define LTL tasks over atomic predicates capturing desired low-level system configurations. Our goal is to design robot plans that satisfy LTL tasks defined over NL-based atomic propositions. A novel technical challenge arising in this setup lies in reasoning about correctness of a robot plan with respect to such LTL-encoded tasks. To address this problem, we propose HERACLEs, a hierarchical conformal natural language planner, that relies on a novel integration of existing tools that include (i) automata theory to determine the NL-specified sub-task the robot should accomplish next to make mission progress; (ii) Large Language Models to design robot plans satisfying these sub-tasks; and (iii) conformal prediction to reason probabilistically about correctness of the designed plans and mission satisfaction and to determine if external assistance is required. We provide extensive comparative experiments on mobile manipulation tasks. The project website is ltl-llm.github.io.

{{</citation>}}


### (91/154) Toward collision-free trajectory for autonomous and pilot-controlled unmanned aerial vehicles (Kaya Kuru et al., 2023)

{{<citation>}}

Kaya Kuru, John Michael Pinder, Benjamin Jon Watkinson, Darren Ansell, Keith Vinning, Lee Moore, Chris Gilbert, Aadithya Sujit, David Jones. (2023)  
**Toward collision-free trajectory for autonomous and pilot-controlled unmanned aerial vehicles**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2309.10064v1)  

---


**ABSTRACT**  
For drones, as safety-critical systems, there is an increasing need for onboard detect & avoid (DAA) technology i) to see, sense or detect conflicting traffic or imminent non-cooperative threats due to their high mobility with multiple degrees of freedom and the complexity of deployed unstructured environments, and subsequently ii) to take the appropriate actions to avoid collisions depending upon the level of autonomy. The safe and efficient integration of UAV traffic management (UTM) systems with air traffic management (ATM) systems, using intelligent autonomous approaches, is an emerging requirement where the number of diverse UAV applications is increasing on a large scale in dense air traffic environments for completing swarms of multiple complex missions flexibly and simultaneously. Significant progress over the past few years has been made in detecting UAVs present in aerospace, identifying them, and determining their existing flight path. This study makes greater use of electronic conspicuity (EC) information made available by PilotAware Ltd in developing an advanced collision management methodology -- Drone Aware Collision Management (DACM) -- capable of determining and executing a variety of time-optimal evasive collision avoidance (CA) manoeuvres using a reactive geometric conflict detection and resolution (CDR) technique. The merits of the DACM methodology have been demonstrated through extensive simulations and real-world field tests in avoiding mid-air collisions (MAC) between UAVs and manned aeroplanes. The results show that the proposed methodology can be employed successfully in avoiding collisions while limiting the deviation from the original trajectory in highly dynamic aerospace without requiring sophisticated sensors and prior training.

{{</citation>}}


### (92/154) SMART-LLM: Smart Multi-Agent Robot Task Planning using Large Language Models (Shyam Sundar Kannan et al., 2023)

{{<citation>}}

Shyam Sundar Kannan, Vishnunandan L. N. Venkatesh, Byung-Cheol Min. (2023)  
**SMART-LLM: Smart Multi-Agent Robot Task Planning using Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.10062v1)  

---


**ABSTRACT**  
In this work, we introduce SMART-LLM, an innovative framework designed for embodied multi-robot task planning. SMART-LLM: Smart Multi-Agent Robot Task Planning using Large Language Models (LLMs), harnesses the power of LLMs to convert high-level task instructions provided as input into a multi-robot task plan. It accomplishes this by executing a series of stages, including task decomposition, coalition formation, and task allocation, all guided by programmatic LLM prompts within the few-shot prompting paradigm. We create a benchmark dataset designed for validating the multi-robot task planning problem, encompassing four distinct categories of high-level instructions that vary in task complexity. Our evaluation experiments span both simulation and real-world scenarios, demonstrating that the proposed model can achieve promising results for generating multi-robot task plans. The experimental videos, code, and datasets from the work can be found at https://sites.google.com/view/smart-llm/.

{{</citation>}}


### (93/154) Prompt a Robot to Walk with Large Language Models (Yen-Jen Wang et al., 2023)

{{<citation>}}

Yen-Jen Wang, Bike Zhang, Jianyu Chen, Koushil Sreenath. (2023)  
**Prompt a Robot to Walk with Large Language Models**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.09969v1)  

---


**ABSTRACT**  
Large language models (LLMs) pre-trained on vast internet-scale data have showcased remarkable capabilities across diverse domains. Recently, there has been escalating interest in deploying LLMs for robotics, aiming to harness the power of foundation models in real-world settings. However, this approach faces significant challenges, particularly in grounding these models in the physical world and in generating dynamic robot motions. To address these issues, we introduce a novel paradigm in which we use few-shot prompts collected from the physical environment, enabling the LLM to autoregressively generate low-level control commands for robots without task-specific fine-tuning. Experiments across various robots and environments validate that our method can effectively prompt a robot to walk. We thus illustrate how LLMs can proficiently function as low-level feedback controllers for dynamic motion control even in high-dimensional robotic systems. The project website and source code can be found at: https://prompt2walk.github.io/ .

{{</citation>}}


### (94/154) OptiRoute: A Heuristic-assisted Deep Reinforcement Learning Framework for UAV-UGV Collaborative Route Planning (Md Safwan Mondal et al., 2023)

{{<citation>}}

Md Safwan Mondal, Subramanian Ramasamy, Pranav Bhounsule. (2023)  
**OptiRoute: A Heuristic-assisted Deep Reinforcement Learning Framework for UAV-UGV Collaborative Route Planning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.09942v1)  

---


**ABSTRACT**  
Unmanned aerial vehicles (UAVs) are capable of surveying expansive areas, but their operational range is constrained by limited battery capacity. The deployment of mobile recharging stations using unmanned ground vehicles (UGVs) significantly extends the endurance and effectiveness of UAVs. However, optimizing the routes of both UAVs and UGVs, known as the UAV-UGV cooperative routing problem, poses substantial challenges, particularly with respect to the selection of recharging locations. Here in this paper, we leverage reinforcement learning (RL) for the purpose of identifying optimal recharging locations while employing constraint programming to determine cooperative routes for the UAV and UGV. Our proposed framework is then benchmarked against a baseline solution that employs Genetic Algorithms (GA) to select rendezvous points. Our findings reveal that RL surpasses GA in terms of reducing overall mission time, minimizing UAV-UGV idle time, and mitigating energy consumption for both the UAV and UGV. These results underscore the efficacy of incorporating heuristics to assist RL, a method we refer to as heuristics-assisted RL, in generating high-quality solutions for intricate routing problems.

{{</citation>}}


### (95/154) Zero-Shot Policy Transferability for the Control of a Scale Autonomous Vehicle (Harry Zhang et al., 2023)

{{<citation>}}

Harry Zhang, Stefan Caldararu, Sriram Ashokkumar, Ishaan Mahajan, Aaron Young, Alexis Ruiz, Huzaifa Unjhawala, Luning Bakke, Dan Negrut. (2023)  
**Zero-Shot Policy Transferability for the Control of a Scale Autonomous Vehicle**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.09870v1)  

---


**ABSTRACT**  
We report on a study that employs an in-house developed simulation infrastructure to accomplish zero shot policy transferability for a control policy associated with a scale autonomous vehicle. We focus on implementing policies that require no real world data to be trained (Zero-Shot Transfer), and are developed in-house as opposed to being validated by previous works. We do this by implementing a Neural Network (NN) controller that is trained only on a family of circular reference trajectories. The sensors used are RTK-GPS and IMU, the latter for providing heading. The NN controller is trained using either a human driver (via human in the loop simulation), or a Model Predictive Control (MPC) strategy. We demonstrate these two approaches in conjunction with two operation scenarios: the vehicle follows a waypoint-defined trajectory at constant speed; and the vehicle follows a speed profile that changes along the vehicle's waypoint-defined trajectory. The primary contribution of this work is the demonstration of Zero-Shot Transfer in conjunction with a novel feed-forward NN controller trained using a general purpose, in-house developed simulation platform.

{{</citation>}}


### (96/154) Contrastive Learning for Enhancing Robust Scene Transfer in Vision-based Agile Flight (Jiaxu Xing et al., 2023)

{{<citation>}}

Jiaxu Xing, Leonard Bauersfeld, Yunlong Song, Chunwei Xing, Davide Scaramuzza. (2023)  
**Contrastive Learning for Enhancing Robust Scene Transfer in Vision-based Agile Flight**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.09865v1)  

---


**ABSTRACT**  
Scene transfer for vision-based mobile robotics applications is a highly relevant and challenging problem. The utility of a robot greatly depends on its ability to perform a task in the real world, outside of a well-controlled lab environment. Existing scene transfer end-to-end policy learning approaches often suffer from poor sample efficiency or limited generalization capabilities, making them unsuitable for mobile robotics applications. This work proposes an adaptive multi-pair contrastive learning strategy for visual representation learning that enables zero-shot scene transfer and real-world deployment. Control policies relying on the embedding are able to operate in unseen environments without the need for finetuning in the deployment environment. We demonstrate the performance of our approach on the task of agile, vision-based quadrotor flight. Extensive simulation and real-world experiments demonstrate that our approach successfully generalizes beyond the training domain and outperforms all baselines.

{{</citation>}}


### (97/154) CC-SGG: Corner Case Scenario Generation using Learned Scene Graphs (George Drayson et al., 2023)

{{<citation>}}

George Drayson, Efimia Panagiotaki, Daniel Omeiza, Lars Kunze. (2023)  
**CC-SGG: Corner Case Scenario Generation using Learned Scene Graphs**  

---
Primary Category: cs.RO  
Categories: I-2-4; I-2-6; I-2-9; I-2-10; I-6-3; I-6-4; I-4-8, cs-AI, cs-CV, cs-LG, cs-RO, cs.RO  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.09844v1)  

---


**ABSTRACT**  
Corner case scenarios are an essential tool for testing and validating the safety of autonomous vehicles (AVs). As these scenarios are often insufficiently present in naturalistic driving datasets, augmenting the data with synthetic corner cases greatly enhances the safe operation of AVs in unique situations. However, the generation of synthetic, yet realistic, corner cases poses a significant challenge. In this work, we introduce a novel approach based on Heterogeneous Graph Neural Networks (HGNNs) to transform regular driving scenarios into corner cases. To achieve this, we first generate concise representations of regular driving scenes as scene graphs, minimally manipulating their structure and properties. Our model then learns to perturb those graphs to generate corner cases using attention and triple embeddings. The input and perturbed graphs are then imported back into the simulation to generate corner case scenarios. Our model successfully learned to produce corner cases from input scene graphs, achieving 89.9% prediction accuracy on our testing dataset. We further validate the generated scenarios on baseline autonomous driving methods, demonstrating our model's ability to effectively create critical situations for the baselines.

{{</citation>}}


### (98/154) Grasp-Anything: Large-scale Grasp Dataset from Foundation Models (An Dinh Vuong et al., 2023)

{{<citation>}}

An Dinh Vuong, Minh Nhat Vu, Hieu Le, Baoru Huang, Binh Huynh, Thieu Vo, Andreas Kugi, Anh Nguyen. (2023)  
**Grasp-Anything: Large-scale Grasp Dataset from Foundation Models**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.09818v1)  

---


**ABSTRACT**  
Foundation models such as ChatGPT have made significant strides in robotic tasks due to their universal representation of real-world domains. In this paper, we leverage foundation models to tackle grasp detection, a persistent challenge in robotics with broad industrial applications. Despite numerous grasp datasets, their object diversity remains limited compared to real-world figures. Fortunately, foundation models possess an extensive repository of real-world knowledge, including objects we encounter in our daily lives. As a consequence, a promising solution to the limited representation in previous grasp datasets is to harness the universal knowledge embedded in these foundation models. We present Grasp-Anything, a new large-scale grasp dataset synthesized from foundation models to implement this solution. Grasp-Anything excels in diversity and magnitude, boasting 1M samples with text descriptions and more than 3M objects, surpassing prior datasets. Empirically, we show that Grasp-Anything successfully facilitates zero-shot grasp detection on vision-based tasks and real-world robotic experiments. Our dataset and code are available at https://grasp-anything-2023.github.io.

{{</citation>}}


### (99/154) Learning Inertial Parameter Identification of Unknown Object with Humanoid Robot using Sim-to-Real Adaptation (Donghoon Baek et al., 2023)

{{<citation>}}

Donghoon Baek, Bo Peng, Saurabh Gupta, Joao Ramos. (2023)  
**Learning Inertial Parameter Identification of Unknown Object with Humanoid Robot using Sim-to-Real Adaptation**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.09810v1)  

---


**ABSTRACT**  
Understanding the dynamics of unknown object is crucial for collaborative robots including humanoids to more safely and accurately interact with humans. Most relevant literature leverage a force/torque sensor, prior knowledge of object, vision system, and a long-horizon trajectory which are often impractical. Moreover, these methods often entail solving non-linear optimization problem, sometimes yielding physically inconsistent results. In this work, we propose a fast learningbased inertial parameter estimation as more practical manner. We acquire a reliable dataset in a high-fidelity simulation and train a time-series data-driven regression model (e.g., LSTM) to estimate the inertial parameter of unknown objects. We also introduce a novel sim-to-real adaptation method combining Robot System Identification and Gaussian Processes to directly transfer the trained model to real-world application. We demonstrate our method with a 4-DOF single manipulator of physical wheeled humanoid robot, SATYRR. Results show that our method can identify the inertial parameters of various unknown objects faster and more accurately than conventional methods.

{{</citation>}}


### (100/154) Privileged to Predicted: Towards Sensorimotor Reinforcement Learning for Urban Driving (Ege Onat Özsüer et al., 2023)

{{<citation>}}

Ege Onat Özsüer, Barış Akgün, Fatma Güney. (2023)  
**Privileged to Predicted: Towards Sensorimotor Reinforcement Learning for Urban Driving**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.09756v1)  

---


**ABSTRACT**  
Reinforcement Learning (RL) has the potential to surpass human performance in driving without needing any expert supervision. Despite its promise, the state-of-the-art in sensorimotor self-driving is dominated by imitation learning methods due to the inherent shortcomings of RL algorithms. Nonetheless, RL agents are able to discover highly successful policies when provided with privileged ground truth representations of the environment. In this work, we investigate what separates privileged RL agents from sensorimotor agents for urban driving in order to bridge the gap between the two. We propose vision-based deep learning models to approximate the privileged representations from sensor data. In particular, we identify aspects of state representation that are crucial for the success of the RL agent such as desired route generation and stop zone prediction, and propose solutions to gradually develop less privileged RL agents. We also observe that bird's-eye-view models trained on offline datasets do not generalize to online RL training due to distribution mismatch. Through rigorous evaluation on the CARLA simulation environment, we shed light on the significance of the state representations in RL for autonomous driving and point to unresolved challenges for future research.

{{</citation>}}


### (101/154) Towards Socially Responsive Autonomous Vehicles: A Reinforcement Learning Framework with Driving Priors and Coordination Awareness (Jiaqi Liu et al., 2023)

{{<citation>}}

Jiaqi Liu, Donghao Zhou, Peng Hang, Ying Ni, Jian Sun. (2023)  
**Towards Socially Responsive Autonomous Vehicles: A Reinforcement Learning Framework with Driving Priors and Coordination Awareness**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.09726v1)  

---


**ABSTRACT**  
The advent of autonomous vehicles (AVs) alongside human-driven vehicles (HVs) has ushered in an era of mixed traffic flow, presenting a significant challenge: the intricate interaction between these entities within complex driving environments. AVs are expected to have human-like driving behavior to seamlessly integrate into human-dominated traffic systems. To address this issue, we propose a reinforcement learning framework that considers driving priors and Social Coordination Awareness (SCA) to optimize the behavior of AVs. The framework integrates a driving prior learning (DPL) model based on a variational autoencoder to infer the driver's driving priors from human drivers' trajectories. A policy network based on a multi-head attention mechanism is designed to effectively capture the interactive dependencies between AVs and other traffic participants to improve decision-making quality. The introduction of SCA into the autonomous driving decision-making system, and the use of Coordination Tendency (CT) to quantify the willingness of AVs to coordinate the traffic system is explored. Simulation results show that the proposed framework can not only improve the decision-making quality of AVs but also motivate them to produce social behaviors, with potential benefits for the safety and traffic efficiency of the entire transportation system.

{{</citation>}}


### (102/154) Multi-Agent Deep Reinforcement Learning for Cooperative and Competitive Autonomous Vehicles using AutoDRIVE Ecosystem (Tanmay Vilas Samak et al., 2023)

{{<citation>}}

Tanmay Vilas Samak, Chinmay Vilas Samak, Venkat Krovi. (2023)  
**Multi-Agent Deep Reinforcement Learning for Cooperative and Competitive Autonomous Vehicles using AutoDRIVE Ecosystem**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-MA, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.10007v1)  

---


**ABSTRACT**  
This work presents a modular and parallelizable multi-agent deep reinforcement learning framework for imbibing cooperative as well as competitive behaviors within autonomous vehicles. We introduce AutoDRIVE Ecosystem as an enabler to develop physically accurate and graphically realistic digital twins of Nigel and F1TENTH, two scaled autonomous vehicle platforms with unique qualities and capabilities, and leverage this ecosystem to train and deploy multi-agent reinforcement learning policies. We first investigate an intersection traversal problem using a set of cooperative vehicles (Nigel) that share limited state information with each other in single as well as multi-agent learning settings using a common policy approach. We then investigate an adversarial head-to-head autonomous racing problem using a different set of vehicles (F1TENTH) in a multi-agent learning setting using an individual policy approach. In either set of experiments, a decentralized learning architecture was adopted, which allowed robust training and testing of the approaches in stochastic environments, since the agents were mutually independent and exhibited asynchronous motion behavior. The problems were further aggravated by providing the agents with sparse observation spaces and requiring them to sample control commands that implicitly satisfied the imposed kinodynamic as well as safety constraints. The experimental results for both problem statements are reported in terms of quantitative metrics and qualitative remarks for training as well as deployment phases.

{{</citation>}}


### (103/154) Guided Online Distillation: Promoting Safe Reinforcement Learning by Offline Demonstration (Jinning Li et al., 2023)

{{<citation>}}

Jinning Li, Xinyi Liu, Banghua Zhu, Jiantao Jiao, Masayoshi Tomizuka, Chen Tang, Wei Zhan. (2023)  
**Guided Online Distillation: Promoting Safe Reinforcement Learning by Offline Demonstration**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.09408v1)  

---


**ABSTRACT**  
Safe Reinforcement Learning (RL) aims to find a policy that achieves high rewards while satisfying cost constraints. When learning from scratch, safe RL agents tend to be overly conservative, which impedes exploration and restrains the overall performance. In many realistic tasks, e.g. autonomous driving, large-scale expert demonstration data are available. We argue that extracting expert policy from offline data to guide online exploration is a promising solution to mitigate the conserveness issue. Large-capacity models, e.g. decision transformers (DT), have been proven to be competent in offline policy learning. However, data collected in real-world scenarios rarely contain dangerous cases (e.g., collisions), which makes it prohibitive for the policies to learn safety concepts. Besides, these bulk policy networks cannot meet the computation speed requirements at inference time on real-world tasks such as autonomous driving. To this end, we propose Guided Online Distillation (GOLD), an offline-to-online safe RL framework. GOLD distills an offline DT policy into a lightweight policy network through guided online safe RL training, which outperforms both the offline DT policy and online safe RL algorithms. Experiments in both benchmark safe RL tasks and real-world driving tasks based on the Waymo Open Motion Dataset (WOMD) demonstrate that GOLD can successfully distill lightweight policies and solve decision-making problems in challenging safety-critical scenarios.

{{</citation>}}


## cs.CR (7)



### (104/154) GCNIDS: GCN-based intrusion detection system for CAN Bus (Maloy Kumar Devnath, 2023)

{{<citation>}}

Maloy Kumar Devnath. (2023)  
**GCNIDS: GCN-based intrusion detection system for CAN Bus**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2309.10173v1)  

---


**ABSTRACT**  
The Controller Area Network (CAN) bus serves as a standard protocol for facilitating communication among various electronic control units (ECUs) within contemporary vehicles. However, it has been demonstrated that the CAN bus is susceptible to remote attacks, which pose risks to the vehicle's safety and functionality. To tackle this concern, researchers have introduced intrusion detection systems (IDSs) to identify and thwart such attacks. In this paper, we present an innovative approach to intruder detection within the CAN bus, leveraging Graph Convolutional Network (GCN) techniques as introduced by Zhang, Tong, Xu, and Maciejewski in 2019. By harnessing the capabilities of deep learning, we aim to enhance attack detection accuracy while minimizing the requirement for manual feature engineering. Our experimental findings substantiate that the proposed GCN-based method surpasses existing IDSs in terms of accuracy, precision, and recall. Additionally, our approach demonstrates efficacy in detecting mixed attacks, which are more challenging to identify than single attacks. Furthermore, it reduces the necessity for extensive feature engineering and is particularly well-suited for real-time detection systems. To the best of our knowledge, this represents the pioneering application of GCN to CAN data for intrusion detection. Our proposed approach holds significant potential in fortifying the security and safety of modern vehicles, safeguarding against attacks and preventing them from undermining vehicle functionality.

{{</citation>}}


### (105/154) Efficient Avoidance of Vulnerabilities in Auto-completed Smart Contract Code Using Vulnerability-constrained Decoding (André Storhaug et al., 2023)

{{<citation>}}

André Storhaug, Jingyue Li, Tianyuan Hu. (2023)  
**Efficient Avoidance of Vulnerabilities in Auto-completed Smart Contract Code Using Vulnerability-constrained Decoding**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CL, cs-CR, cs.CR  
Keywords: BLEU, GPT  
[Paper Link](http://arxiv.org/abs/2309.09826v1)  

---


**ABSTRACT**  
Auto-completing code enables developers to speed up coding significantly. Recent advances in transformer-based large language model (LLM) technologies have been applied to code synthesis. However, studies show that many of such synthesized codes contain vulnerabilities. We propose a novel vulnerability-constrained decoding approach to reduce the amount of vulnerable code generated by such models. Using a small dataset of labeled vulnerable lines of code, we fine-tune an LLM to include vulnerability labels when generating code, acting as an embedded classifier. Then, during decoding, we deny the model to generate these labels to avoid generating vulnerable code. To evaluate the method, we chose to automatically complete Ethereum Blockchain smart contracts (SCs) as the case study due to the strict requirements of SC security. We first fine-tuned the 6-billion-parameter GPT-J model using 186,397 Ethereum SCs after removing the duplication from 2,217,692 SCs. The fine-tuning took more than one week using ten GPUs. The results showed that our fine-tuned model could synthesize SCs with an average BLEU (BiLingual Evaluation Understudy) score of 0.557. However, many codes in the auto-completed SCs were vulnerable. Using the code before the vulnerable line of 176 SCs containing different types of vulnerabilities to auto-complete the code, we found that more than 70% of the auto-completed codes were insecure. Thus, we further fine-tuned the model on other 941 vulnerable SCs containing the same types of vulnerabilities and applied vulnerability-constrained decoding. The fine-tuning took only one hour with four GPUs. We then auto-completed the 176 SCs again and found that our approach could identify 62% of the code to be generated as vulnerable and avoid generating 67% of them, indicating the approach could efficiently and effectively avoid vulnerabilities in the auto-completed code.

{{</citation>}}


### (106/154) Towards Model Co-evolution Across Self-Adaptation Steps for Combined Safety and Security Analysis (Thomas Witte et al., 2023)

{{<citation>}}

Thomas Witte, Raffaela Groner, Alexander Raschke, Matthias Tichy, Irdin Pekaric, Michael Felderer. (2023)  
**Towards Model Co-evolution Across Self-Adaptation Steps for Combined Safety and Security Analysis**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-FL, cs-RO, cs-SE, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.09653v1)  

---


**ABSTRACT**  
Self-adaptive systems offer several attack surfaces due to the communication via different channels and the different sensors required to observe the environment. Often, attacks cause safety to be compromised as well, making it necessary to consider these two aspects together. Furthermore, the approaches currently used for safety and security analysis do not sufficiently take into account the intermediate steps of an adaptation. Current work in this area ignores the fact that a self-adaptive system also reveals possible vulnerabilities (even if only temporarily) during the adaptation. To address this issue, we propose a modeling approach that takes into account the different relevant aspects of a system, its adaptation process, as well as safety hazards and security attacks. We present several models that describe different aspects of a self-adaptive system and we outline our idea of how these models can then be combined into an Attack-Fault Tree. This allows modeling aspects of the system on different levels of abstraction and co-evolve the models using transformations according to the adaptation of the system. Finally, analyses can then be performed as usual on the resulting Attack-Fault Tree.

{{</citation>}}


### (107/154) VULNERLIZER: Cross-analysis Between Vulnerabilities and Software Libraries (Irdin Pekaric et al., 2023)

{{<citation>}}

Irdin Pekaric, Michael Felderer, Philipp Steinmüller. (2023)  
**VULNERLIZER: Cross-analysis Between Vulnerabilities and Software Libraries**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs-SE, cs.CR  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2309.09649v1)  

---


**ABSTRACT**  
The identification of vulnerabilities is a continuous challenge in software projects. This is due to the evolution of methods that attackers employ as well as the constant updates to the software, which reveal additional issues. As a result, new and innovative approaches for the identification of vulnerable software are needed. In this paper, we present VULNERLIZER, which is a novel framework for cross-analysis between vulnerabilities and software libraries. It uses CVE and software library data together with clustering algorithms to generate links between vulnerabilities and libraries. In addition, the training of the model is conducted in order to reevaluate the generated associations. This is achieved by updating the assigned weights. Finally, the approach is then evaluated by making the predictions using the CVE data from the test set. The results show that the VULNERLIZER has a great potential in being able to predict future vulnerable libraries based on an initial input CVE entry or a software library. The trained model reaches a prediction accuracy of 75% or higher.

{{</citation>}}


### (108/154) Applying Security Testing Techniques to Automotive Engineering (Irdin Pekaric et al., 2023)

{{<citation>}}

Irdin Pekaric, Clemens Sauerwein, Michael Felderer. (2023)  
**Applying Security Testing Techniques to Automotive Engineering**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.09647v1)  

---


**ABSTRACT**  
The openness of modern IT systems and their permanent change make it challenging to keep these systems secure. A combination of regression and security testing called security regression testing, which ensures that changes made to a system do not harm its security, are therefore of high significance and the interest in such approaches has steadily increased. In this article we present a systematic classification of available security regression testing approaches based on a solid study of background and related work to sketch which parts of the research area seem to be well understood and evaluated, and which ones require further research. For this purpose we extract approaches relevant to security regression testing from computer science digital libraries based on a rigorous search and selection strategy. Then, we provide a classification of these according to security regression approach criteria: abstraction level, security issue, regression testing techniques, and tool support, as well as evaluation criteria, for instance evaluated system, maturity of the system, and evaluation measures. From the resulting classification we derive observations with regard to the abstraction level, regression testing techniques, tool support as well as evaluation, and finally identify several potential directions of future research.

{{</citation>}}


### (109/154) Security Properties through the Lens of Modal Logic (Matvey Soloviev et al., 2023)

{{<citation>}}

Matvey Soloviev, Musard Balliu, Roberto Guanciale. (2023)  
**Security Properties through the Lens of Modal Logic**  

---
Primary Category: cs.CR  
Categories: 68Q60, D-4-6; I-2-4, cs-CR, cs-LO, cs-MA, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.09542v1)  

---


**ABSTRACT**  
We introduce a framework for reasoning about the security of computer systems using modal logic. This framework is sufficiently expressive to capture a variety of known security properties, while also being intuitive and independent of syntactic details and enforcement mechanisms. We show how to use our formalism to represent various progress- and termination-(in)sensitive variants of confidentiality, integrity, robust declassification and transparent endorsement, and prove equivalence to standard definitions. The intuitive nature and closeness to semantic reality of our approach allows us to make explicit several hidden assumptions of these definitions, and identify potential issues and subtleties with them, while also holding the promise of formulating cleaner versions and future extension to entirely novel properties.

{{</citation>}}


### (110/154) Security and Privacy on Generative Data in AIGC: A Survey (Tao Wang et al., 2023)

{{<citation>}}

Tao Wang, Yushu Zhang, Shuren Qi, Ruoyu Zhao, Zhihua Xia, Jian Weng. (2023)  
**Security and Privacy on Generative Data in AIGC: A Survey**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2309.09435v1)  

---


**ABSTRACT**  
The advent of artificial intelligence-generated content (AIGC) represents a pivotal moment in the evolution of information technology. With AIGC, it can be effortless to generate high-quality data that is challenging for the public to distinguish. Nevertheless, the proliferation of generative data across cyberspace brings security and privacy issues, including privacy leakages of individuals and media forgery for fraudulent purposes. Consequently, both academia and industry begin to emphasize the trustworthiness of generative data, successively providing a series of countermeasures for security and privacy. In this survey, we systematically review the security and privacy on generative data in AIGC, particularly for the first time analyzing them from the perspective of information security properties. Specifically, we reveal the successful experiences of state-of-the-art countermeasures in terms of the foundational properties of privacy, controllability, authenticity, and compliance, respectively. Finally, we summarize the open challenges and potential exploration directions from each of theses properties.

{{</citation>}}


## physics.med-ph (1)



### (111/154) RadOnc-GPT: A Large Language Model for Radiation Oncology (Zhengliang Liu et al., 2023)

{{<citation>}}

Zhengliang Liu, Peilong Wang, Yiwei Li, Jason Holmes, Peng Shu, Lian Zhang, Chenbin Liu, Ninghao Liu, Dajiang Zhu, Xiang Li, Quanzheng Li, Samir H. Patel, Terence T. Sio, Tianming Liu, Wei Liu. (2023)  
**RadOnc-GPT: A Large Language Model for Radiation Oncology**  

---
Primary Category: physics.med-ph  
Categories: cs-AI, physics-med-ph, physics.med-ph  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.10160v1)  

---


**ABSTRACT**  
This paper presents RadOnc-GPT, a large language model specialized for radiation oncology through advanced tuning methods. RadOnc-GPT was finetuned on a large dataset of radiation oncology patient records and clinical notes from the Mayo Clinic. The model employs instruction tuning on three key tasks - generating radiotherapy treatment regimens, determining optimal radiation modalities, and providing diagnostic descriptions/ICD codes based on patient diagnostic details. Evaluations conducted by having radiation oncologists compare RadOnc-GPT impressions to general large language model impressions showed that RadOnc-GPT generated outputs with significantly improved clarity, specificity, and clinical relevance. The study demonstrated the potential of using large language models fine-tuned using domain-specific knowledge like RadOnc-GPT to achieve transformational capabilities in highly specialized healthcare fields such as radiation oncology.

{{</citation>}}


## physics.ins-det (1)



### (112/154) Autoencoder-based Anomaly Detection System for Online Data Quality Monitoring of the CMS Electromagnetic Calorimeter (The CMS ECAL Collaboration, 2023)

{{<citation>}}

The CMS ECAL Collaboration. (2023)  
**Autoencoder-based Anomaly Detection System for Online Data Quality Monitoring of the CMS Electromagnetic Calorimeter**  

---
Primary Category: physics.ins-det  
Categories: cs-LG, hep-ex, physics-data-an, physics-ins-det, physics.ins-det  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2309.10157v1)  

---


**ABSTRACT**  
The CMS detector is a general-purpose apparatus that detects high-energy collisions produced at the LHC. Online Data Quality Monitoring of the CMS electromagnetic calorimeter is a vital operational tool that allows detector experts to quickly identify, localize, and diagnose a broad range of detector issues that could affect the quality of physics data. A real-time autoencoder-based anomaly detection system using semi-supervised machine learning is presented enabling the detection of anomalies in the CMS electromagnetic calorimeter data. A novel method is introduced which maximizes the anomaly detection performance by exploiting the time-dependent evolution of anomalies as well as spatial variations in the detector response. The autoencoder-based system is able to efficiently detect anomalies, while maintaining a very low false discovery rate. The performance of the system is validated with anomalies found in 2018 and 2022 LHC collision data. Additionally, the first results from deploying the autoencoder-based system in the CMS online Data Quality Monitoring workflow during the beginning of Run 3 of the LHC are presented, showing its ability to detect issues missed by the existing system.

{{</citation>}}


## eess.SY (1)



### (113/154) Vertical Power Delivery for Emerging Packaging and Integration Platforms -- Power Conversion and Distribution (Sriharini Krishnakumar et al., 2023)

{{<citation>}}

Sriharini Krishnakumar, Dr. Inna Partin-Vaisband. (2023)  
**Vertical Power Delivery for Emerging Packaging and Integration Platforms -- Power Conversion and Distribution**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.10141v1)  

---


**ABSTRACT**  
Efficient delivery of current from PCB to point-of-load (POL) is a primary concern in modern high-power high-density integrated systems. Traditionally, a 48 V power signal is converted to the low, POL voltage at the board and/or package level. As interconnect has become the dominant power loss component, minimizing voltage drop across the laterally routed portions of the board-to-die interconnect (referred to as horizontal interconnect) is a promising approach to enhance the efficiency of the power delivery system. Delivering lower current vertically, at a higher voltage should therefore be considered. High-power conversion near POL, however, results in higher switching and inductor losses, exhibiting an undesired power efficiency tradeoff. To address this problem, four vertical power delivery architectures are proposed in this paper, considering state-of-the-art power converter topologies, integration levels, and voltage conversion schemes. Embedding Silicon (Si) and Gallium Nitride (GaN) power devices and inductors on top of and/or within the interposer is investigated. Integrating GaN power devices on a dedicated power die is also discussed. Various multi-stage 48V-to-1V power conversion schemes are examined and state-of-the-art power conversion circuits are reviewed. Power delivery characteristics with these architectures are determined for a high power (1 kW) high-current density (2 A/mm$^2$) system.

{{</citation>}}


## cs.AI (14)



### (114/154) Adaptive Liquidity Provision in Uniswap V3 with Deep Reinforcement Learning (Haochen Zhang et al., 2023)

{{<citation>}}

Haochen Zhang, Xi Chen, Lin F. Yang. (2023)  
**Adaptive Liquidity Provision in Uniswap V3 with Deep Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.10129v1)  

---


**ABSTRACT**  
Decentralized exchanges (DEXs) are a cornerstone of decentralized finance (DeFi), allowing users to trade cryptocurrencies without the need for third-party authorization. Investors are incentivized to deposit assets into liquidity pools, against which users can trade directly, while paying fees to liquidity providers (LPs). However, a number of unresolved issues related to capital efficiency and market risk hinder DeFi's further development. Uniswap V3, a leading and groundbreaking DEX project, addresses capital efficiency by enabling LPs to concentrate their liquidity within specific price ranges for deposited assets. Nevertheless, this approach exacerbates market risk, as LPs earn trading fees only when asset prices are within these predetermined brackets. To mitigate this issue, this paper introduces a deep reinforcement learning (DRL) solution designed to adaptively adjust these price ranges, maximizing profits and mitigating market risks. Our approach also neutralizes price-change risks by hedging the liquidity position through a rebalancing portfolio in a centralized futures exchange. The DRL policy aims to optimize trading fees earned by LPs against associated costs, such as gas fees and hedging expenses, which is referred to as loss-versus-rebalancing (LVR). Using simulations with a profit-and-loss (PnL) benchmark, our method demonstrates superior performance in ETH/USDC and ETH/USDT pools compared to existing baselines. We believe that this strategy not only offers investors a valuable asset management tool but also introduces a new incentive mechanism for DEX designers.

{{</citation>}}


### (115/154) Automatic Personalized Impression Generation for PET Reports Using Large Language Models (Xin Tie et al., 2023)

{{<citation>}}

Xin Tie, Muheon Shin, Ali Pirasteh, Nevein Ibrahim, Zachary Huemann, Sharon M. Castellino, Kara M. Kelly, John Garrett, Junjie Hu, Steve Y. Cho, Tyler J. Bradshaw. (2023)  
**Automatic Personalized Impression Generation for PET Reports Using Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI, physics-med-ph  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.10066v1)  

---


**ABSTRACT**  
Purpose: To determine if fine-tuned large language models (LLMs) can generate accurate, personalized impressions for whole-body PET reports. Materials and Methods: Twelve language models were trained on a corpus of PET reports using the teacher-forcing algorithm, with the report findings as input and the clinical impressions as reference. An extra input token encodes the reading physician's identity, allowing models to learn physician-specific reporting styles. Our corpus comprised 37,370 retrospective PET reports collected from our institution between 2010 and 2022. To identify the best LLM, 30 evaluation metrics were benchmarked against quality scores from two nuclear medicine (NM) physicians, with the most aligned metrics selecting the model for expert evaluation. In a subset of data, model-generated impressions and original clinical impressions were assessed by three NM physicians according to 6 quality dimensions and an overall utility score (5-point scale). Each physician reviewed 12 of their own reports and 12 reports from other physicians. Bootstrap resampling was used for statistical analysis. Results: Of all evaluation metrics, domain-adapted BARTScore and PEGASUSScore showed the highest Spearman's rho correlations (0.568 and 0.563) with physician preferences. Based on these metrics, the fine-tuned PEGASUS model was selected as the top LLM. When physicians reviewed PEGASUS-generated impressions in their own style, 89% were considered clinically acceptable, with a mean utility score of 4.08/5. Physicians rated these personalized impressions as comparable in overall utility to the impressions dictated by other physicians (4.03, P=0.41). Conclusion: Personalized impressions generated by PEGASUS were clinically useful, highlighting its potential to expedite PET reporting.

{{</citation>}}


### (116/154) MindAgent: Emergent Gaming Interaction (Ran Gong et al., 2023)

{{<citation>}}

Ran Gong, Qiuyuan Huang, Xiaojian Ma, Hoi Vo, Zane Durante, Yusuke Noda, Zilong Zheng, Song-Chun Zhu, Demetri Terzopoulos, Li Fei-Fei, Jianfeng Gao. (2023)  
**MindAgent: Emergent Gaming Interaction**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-MA, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.09971v2)  

---


**ABSTRACT**  
Large Language Models (LLMs) have the capacity of performing complex scheduling in a multi-agent system and can coordinate these agents into completing sophisticated tasks that require extensive collaboration. However, despite the introduction of numerous gaming frameworks, the community has insufficient benchmarks towards building general multi-agents collaboration infrastructure that encompass both LLM and human-NPCs collaborations. In this work, we propose a novel infrastructure - MindAgent - to evaluate planning and coordination emergent capabilities for gaming interaction. In particular, our infrastructure leverages existing gaming framework, to i) require understanding of the coordinator for a multi-agent system, ii) collaborate with human players via un-finetuned proper instructions, and iii) establish an in-context learning on few-shot prompt with feedback. Furthermore, we introduce CUISINEWORLD, a new gaming scenario and related benchmark that dispatch a multi-agent collaboration efficiency and supervise multiple agents playing the game simultaneously. We conduct comprehensive evaluations with new auto-metric CoS for calculating the collaboration efficiency. Finally, our infrastructure can be deployed into real-world gaming scenarios in a customized VR version of CUISINEWORLD and adapted in existing broader Minecraft gaming domain. We hope our findings on LLMs and the new infrastructure for general-purpose scheduling and coordination can help shed light on how such skills can be obtained by learning from large language corpora.

{{</citation>}}


### (117/154) How to Generate Popular Post Headlines on Social Media? (Zhouxiang Fang et al., 2023)

{{<citation>}}

Zhouxiang Fang, Min Yu, Zhendong Fu, Boning Zhang, Xuanwen Huang, Xiaoqi Tang, Yang Yang. (2023)  
**How to Generate Popular Post Headlines on Social Media?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Social Media, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.09949v1)  

---


**ABSTRACT**  
Posts, as important containers of user-generated-content pieces on social media, are of tremendous social influence and commercial value. As an integral components of a post, the headline has a decisive contribution to the post's popularity. However, current mainstream method for headline generation is still manually writing, which is unstable and requires extensive human effort. This drives us to explore a novel research question: Can we automate the generation of popular headlines on social media? We collect more than 1 million posts of 42,447 celebrities from public data of Xiaohongshu, which is a well-known social media platform in China. We then conduct careful observations on the headlines of these posts. Observation results demonstrate that trends and personal styles are widespread in headlines on social medias and have significant contribution to posts's popularity. Motivated by these insights, we present MEBART, which combines Multiple preference-Extractors with Bidirectional and Auto-Regressive Transformers (BART), capturing trends and personal styles to generate popular headlines on social medias. We perform extensive experiments on real-world datasets and achieve state-of-the-art performance compared with several advanced baselines. In addition, ablation and case studies demonstrate that MEBART advances in capturing trends and personal styles.

{{</citation>}}


### (118/154) A Heterogeneous Graph-Based Multi-Task Learning for Fault Event Diagnosis in Smart Grid (Dibaloke Chanda et al., 2023)

{{<citation>}}

Dibaloke Chanda, Nasim Yahya Soltani. (2023)  
**A Heterogeneous Graph-Based Multi-Task Learning for Fault Event Diagnosis in Smart Grid**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2309.09921v1)  

---


**ABSTRACT**  
Precise and timely fault diagnosis is a prerequisite for a distribution system to ensure minimum downtime and maintain reliable operation. This necessitates access to a comprehensive procedure that can provide the grid operators with insightful information in the case of a fault event. In this paper, we propose a heterogeneous multi-task learning graph neural network (MTL-GNN) capable of detecting, locating and classifying faults in addition to providing an estimate of the fault resistance and current. Using a graph neural network (GNN) allows for learning the topological representation of the distribution system as well as feature learning through a message-passing scheme. We investigate the robustness of our proposed model using the IEEE-123 test feeder system. This work also proposes a novel GNN-based explainability method to identify key nodes in the distribution system which then facilitates informed sparse measurements. Numerical tests validate the performance of the model across all tasks.

{{</citation>}}


### (119/154) Evaluation of Human-Understandability of Global Model Explanations using Decision Tree (Adarsa Sivaprasad et al., 2023)

{{<citation>}}

Adarsa Sivaprasad, Ehud Reiter, Nava Tintarev, Nir Oren. (2023)  
**Evaluation of Human-Understandability of Global Model Explanations using Decision Tree**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09917v1)  

---


**ABSTRACT**  
In explainable artificial intelligence (XAI) research, the predominant focus has been on interpreting models for experts and practitioners. Model agnostic and local explanation approaches are deemed interpretable and sufficient in many applications. However, in domains like healthcare, where end users are patients without AI or domain expertise, there is an urgent need for model explanations that are more comprehensible and instil trust in the model's operations. We hypothesise that generating model explanations that are narrative, patient-specific and global(holistic of the model) would enable better understandability and enable decision-making. We test this using a decision tree model to generate both local and global explanations for patients identified as having a high risk of coronary heart disease. These explanations are presented to non-expert users. We find a strong individual preference for a specific type of explanation. The majority of participants prefer global explanations, while a smaller group prefers local explanations. A task based evaluation of mental models of these participants provide valuable feedback to enhance narrative global explanations. This, in turn, guides the design of health informatics systems that are both trustworthy and actionable.

{{</citation>}}


### (120/154) The role of causality in explainable artificial intelligence (Gianluca Carloni et al., 2023)

{{<citation>}}

Gianluca Carloni, Andrea Berti, Sara Colantonio. (2023)  
**The role of causality in explainable artificial intelligence**  

---
Primary Category: cs.AI  
Categories: I-2; I-2-6, cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09901v1)  

---


**ABSTRACT**  
Causality and eXplainable Artificial Intelligence (XAI) have developed as separate fields in computer science, even though the underlying concepts of causation and explanation share common ancient roots. This is further enforced by the lack of review works jointly covering these two fields. In this paper, we investigate the literature to try to understand how and to what extent causality and XAI are intertwined. More precisely, we seek to uncover what kinds of relationships exist between the two concepts and how one can benefit from them, for instance, in building trust in AI systems. As a result, three main perspectives are identified. In the first one, the lack of causality is seen as one of the major limitations of current AI and XAI approaches, and the "optimal" form of explanations is investigated. The second is a pragmatic perspective and considers XAI as a tool to foster scientific exploration for causal inquiry, via the identification of pursue-worthy experimental manipulations. Finally, the third perspective supports the idea that causality is propaedeutic to XAI in three possible manners: exploiting concepts borrowed from causality to support or improve XAI, utilizing counterfactuals for explainability, and considering accessing a causal model as explaining itself. To complement our analysis, we also provide relevant software solutions used to automate causal tasks. We believe our work provides a unified view of the two fields of causality and XAI by highlighting potential domain bridges and uncovering possible limitations.

{{</citation>}}


### (121/154) Towards Ontology Construction with Language Models (Maurice Funk et al., 2023)

{{<citation>}}

Maurice Funk, Simon Hosemann, Jean Christoph Jung, Carsten Lutz. (2023)  
**Towards Ontology Construction with Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.09898v1)  

---


**ABSTRACT**  
We present a method for automatically constructing a concept hierarchy for a given domain by querying a large language model. We apply this method to various domains using OpenAI's GPT 3.5. Our experiments indicate that LLMs can be of considerable help for constructing concept hierarchies.

{{</citation>}}


### (122/154) Bias of AI-Generated Content: An Examination of News Produced by Large Language Models (Xiao Fang et al., 2023)

{{<citation>}}

Xiao Fang, Shangkun Che, Minjia Mao, Hongzhe Zhang, Ming Zhao, Xiaohang Zhao. (2023)  
**Bias of AI-Generated Content: An Examination of News Produced by Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Bias, ChatGPT, GPT, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2309.09825v2)  

---


**ABSTRACT**  
Large language models (LLMs) have the potential to transform our lives and work through the content they generate, known as AI-Generated Content (AIGC). To harness this transformation, we need to understand the limitations of LLMs. Here, we investigate the bias of AIGC produced by seven representative LLMs, including ChatGPT and LLaMA. We collect news articles from The New York Times and Reuters, both known for their dedication to provide unbiased news. We then apply each examined LLM to generate news content with headlines of these news articles as prompts, and evaluate the gender and racial biases of the AIGC produced by the LLM by comparing the AIGC and the original news articles. We further analyze the gender bias of each LLM under biased prompts by adding gender-biased messages to prompts constructed from these news headlines. Our study reveals that the AIGC produced by each examined LLM demonstrates substantial gender and racial biases. Moreover, the AIGC generated by each LLM exhibits notable discrimination against females and individuals of the Black race. Among the LLMs, the AIGC generated by ChatGPT demonstrates the lowest level of bias, and ChatGPT is the sole model capable of declining content generation when provided with biased prompts.

{{</citation>}}


### (123/154) CB-Whisper: Contextual Biasing Whisper using TTS-based Keyword Spotting (Yuang Li et al., 2023)

{{<citation>}}

Yuang Li, Yinglu Li, Min Zhang, Chang Su, Mengyao Piao, Xiaosong Qiao, Jiawei Yu, Miaomiao Ma, Yanqing Zhao, Hao Yang. (2023)  
**CB-Whisper: Contextual Biasing Whisper using TTS-based Keyword Spotting**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2309.09552v1)  

---


**ABSTRACT**  
End-to-end automatic speech recognition (ASR) systems often struggle to recognize rare name entities, such as personal names, organizations, or technical terms that are not frequently encountered in the training data. This paper presents Contextual Biasing Whisper (CB-Whisper), a novel ASR system based on OpenAI's Whisper model that performs keyword-spotting (KWS) before the decoder. The KWS module leverages text-to-speech (TTS) techniques and a convolutional neural network (CNN) classifier to match the features between the entities and the utterances. Experiments demonstrate that by incorporating predicted entities into a carefully designed spoken form prompt, the mixed-error-rate (MER) and entity recall of the Whisper model is significantly improved on three internal datasets and two open-sourced datasets that cover English-only, Chinese-only, and code-switching scenarios.

{{</citation>}}


### (124/154) Pruning Large Language Models via Accuracy Predictor (Yupeng Ji et al., 2023)

{{<citation>}}

Yupeng Ji, Yibo Cao, Jiucai Liu. (2023)  
**Pruning Large Language Models via Accuracy Predictor**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model, NLP, Pruning  
[Paper Link](http://arxiv.org/abs/2309.09507v1)  

---


**ABSTRACT**  
Large language models(LLMs) containing tens of billions of parameters (or even more) have demonstrated impressive capabilities in various NLP tasks. However, substantial model size poses challenges to training, inference, and deployment so that it is necessary to compress the model. At present, most model compression for LLMs requires manual design of pruning features, which has problems such as complex optimization pipeline and difficulty in retaining the capabilities of certain parts of the model.Therefore, we propose a novel pruning approach: firstly, a training set of a certain number of architecture-accuracy pairs is established, and then a non-neural model is trained as an accuracy predictor. Using the accuracy predictor to further optimize the search space and search, the optimal model can be automatically selected. Experiments show that our proposed approach is effective and efficient. Compared with the baseline, the perplexity(PPL) on Wikitext2 and PTB dropped by 9.48% and 5,76% respectively, and the average accuracy of MMLU increased by 6.28%.

{{</citation>}}


### (125/154) Mechanic Maker 2.0: Reinforcement Learning for Evaluating Generated Rules (Johor Jara Gonzalez et al., 2023)

{{<citation>}}

Johor Jara Gonzalez, Seth Cooper, Mathew Guzdial. (2023)  
**Mechanic Maker 2.0: Reinforcement Learning for Evaluating Generated Rules**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.09476v1)  

---


**ABSTRACT**  
Automated game design (AGD), the study of automatically generating game rules, has a long history in technical games research. AGD approaches generally rely on approximations of human play, either objective functions or AI agents. Despite this, the majority of these approximators are static, meaning they do not reflect human player's ability to learn and improve in a game. In this paper, we investigate the application of Reinforcement Learning (RL) as an approximator for human play for rule generation. We recreate the classic AGD environment Mechanic Maker in Unity as a new, open-source rule generation framework. Our results demonstrate that RL produces distinct sets of rules from an A* agent baseline, which may be more usable by humans.

{{</citation>}}


### (126/154) Does Video Summarization Require Videos? Quantifying the Effectiveness of Language in Video Summarization (Yoonsoo Nam et al., 2023)

{{<citation>}}

Yoonsoo Nam, Adam Lehavi, Daniel Yang, Digbalay Bose, Swabha Swayamdipta, Shrikanth Narayanan. (2023)  
**Does Video Summarization Require Videos? Quantifying the Effectiveness of Language in Video Summarization**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-CV, cs.AI  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2309.09405v1)  

---


**ABSTRACT**  
Video summarization remains a huge challenge in computer vision due to the size of the input videos to be summarized. We propose an efficient, language-only video summarizer that achieves competitive accuracy with high data efficiency. Using only textual captions obtained via a zero-shot approach, we train a language transformer model and forego image representations. This method allows us to perform filtration amongst the representative text vectors and condense the sequence. With our approach, we gain explainability with natural language that comes easily for human interpretation and textual summaries of the videos. An ablation study that focuses on modality and data compression shows that leveraging text modality only effectively reduces input data processing while retaining comparable results.

{{</citation>}}


### (127/154) (Deployed Application) Promoting Research Collaboration with Open Data Driven Team Recommendation in Response to Call for Proposals (Siva Likitha Valluru et al., 2023)

{{<citation>}}

Siva Likitha Valluru, Biplav Srivastava, Sai Teja Paladi, Siwen Yan, Sriraam Natarajan. (2023)  
**(Deployed Application) Promoting Research Collaboration with Open Data Driven Team Recommendation in Response to Call for Proposals**  

---
Primary Category: cs.AI  
Categories: H-3-3; I-2-7, cs-AI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09404v1)  

---


**ABSTRACT**  
Building teams and promoting collaboration are two very common business activities. An example of these are seen in the TeamingForFunding problem, where research institutions and researchers are interested to identify collaborative opportunities when applying to funding agencies in response to latter's calls for proposals. We describe a novel system to recommend teams using a variety of AI methods, such that (1) each team achieves the highest possible skill coverage that is demanded by the opportunity, and (2) the workload of distributing the opportunities is balanced amongst the candidate members. We address these questions by extracting skills latent in open data of proposal calls (demand) and researcher profiles (supply), normalizing them using taxonomies, and creating efficient algorithms that match demand to supply. We create teams to maximize goodness along a novel metric balancing short- and long-term objectives. We validate the success of our algorithms (1) quantitatively, by evaluating the recommended teams using a goodness score and find that more informed methods lead to recommendations of smaller number of teams but higher goodness, and (2) qualitatively, by conducting a large-scale user study at a college-wide level, and demonstrate that users overall found the tool very useful and relevant. Lastly, we evaluate our system in two diverse settings in US and India (of researchers and proposal calls) to establish generality of our approach, and deploy it at a major US university for routine use.

{{</citation>}}


## eess.AS (6)



### (128/154) HTEC: Human Transcription Error Correction (Hanbo Sun et al., 2023)

{{<citation>}}

Hanbo Sun, Jian Gao, Xiaomin Wu, Anjie Fang, Cheng Cao, Zheng Du. (2023)  
**HTEC: Human Transcription Error Correction**  

---
Primary Category: eess.AS  
Categories: 68T50, I-2-7, cs-AI, cs-CL, cs-HC, cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.10089v1)  

---


**ABSTRACT**  
High-quality human transcription is essential for training and improving Automatic Speech Recognition (ASR) models. Recent study~\cite{libricrowd} has found that every 1% worse transcription Word Error Rate (WER) increases approximately 2% ASR WER by using the transcriptions to train ASR models. Transcription errors are inevitable for even highly-trained annotators. However, few studies have explored human transcription correction. Error correction methods for other problems, such as ASR error correction and grammatical error correction, do not perform sufficiently for this problem. Therefore, we propose HTEC for Human Transcription Error Correction. HTEC consists of two stages: Trans-Checker, an error detection model that predicts and masks erroneous words, and Trans-Filler, a sequence-to-sequence generative model that fills masked positions. We propose a holistic list of correction operations, including four novel operations handling deletion errors. We further propose a variant of embeddings that incorporates phoneme information into the input of the transformer. HTEC outperforms other methods by a large margin and surpasses human annotators by 2.2% to 4.5% in WER. Finally, we deployed HTEC to assist human annotators and showed HTEC is particularly effective as a co-pilot, which improves transcription quality by 15.1% without sacrificing transcription velocity.

{{</citation>}}


### (129/154) Investigating End-to-End ASR Architectures for Long Form Audio Transcription (Nithin Rao Koluguri et al., 2023)

{{<citation>}}

Nithin Rao Koluguri, Samuel Kriman, Georgy Zelenfroind, Somshubra Majumdar, Dima Rekesh, Vahid Noroozi, Jagadeesh Balam, Boris Ginsburg. (2023)  
**Investigating End-to-End ASR Architectures for Long Form Audio Transcription**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.09950v2)  

---


**ABSTRACT**  
This paper presents an overview and evaluation of some of the end-to-end ASR models on long-form audios. We study three categories of Automatic Speech Recognition(ASR) models based on their core architecture: (1) convolutional, (2) convolutional with squeeze-and-excitation and (3) convolutional models with attention. We selected one ASR model from each category and evaluated Word Error Rate, maximum audio length and real-time factor for each model on a variety of long audio benchmarks: Earnings-21 and 22, CORAAL, and TED-LIUM3. The model from the category of self-attention with local attention and global token has the best accuracy comparing to other architectures. We also compared models with CTC and RNNT decoders and showed that CTC-based models are more robust and efficient than RNNT on long form audio.

{{</citation>}}


### (130/154) Distilling HuBERT with LSTMs via Decoupled Knowledge Distillation (Danilo de Oliveira et al., 2023)

{{<citation>}}

Danilo de Oliveira, Timo Gerkmann. (2023)  
**Distilling HuBERT with LSTMs via Decoupled Knowledge Distillation**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess-SP, eess.AS  
Keywords: BERT, Knowledge Distillation, LSTM, Transformer  
[Paper Link](http://arxiv.org/abs/2309.09920v1)  

---


**ABSTRACT**  
Much research effort is being applied to the task of compressing the knowledge of self-supervised models, which are powerful, yet large and memory consuming. In this work, we show that the original method of knowledge distillation (and its more recently proposed extension, decoupled knowledge distillation) can be applied to the task of distilling HuBERT. In contrast to methods that focus on distilling internal features, this allows for more freedom in the network architecture of the compressed model. We thus propose to distill HuBERT's Transformer layers into an LSTM-based distilled model that reduces the number of parameters even below DistilHuBERT and at the same time shows improved performance in automatic speech recognition.

{{</citation>}}


### (131/154) Corpus Synthesis for Zero-shot ASR domain Adaptation using Large Language Models (Hsuan Su et al., 2023)

{{<citation>}}

Hsuan Su, Ting-Yao Hu, Hema Swetha Koppula, Raviteja Vemulapalli, Jen-Hao Rick Chang, Karren Yang, Gautam Varma Mantena, Oncel Tuzel. (2023)  
**Corpus Synthesis for Zero-shot ASR domain Adaptation using Large Language Models**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Language Model, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.10707v1)  

---


**ABSTRACT**  
While Automatic Speech Recognition (ASR) systems are widely used in many real-world applications, they often do not generalize well to new domains and need to be finetuned on data from these domains. However, target-domain data usually are not readily available in many scenarios. In this paper, we propose a new strategy for adapting ASR models to new target domains without any text or speech from those domains. To accomplish this, we propose a novel data synthesis pipeline that uses a Large Language Model (LLM) to generate a target domain text corpus, and a state-of-the-art controllable speech synthesis model to generate the corresponding speech. We propose a simple yet effective in-context instruction finetuning strategy to increase the effectiveness of LLM in generating text corpora for new domains. Experiments on the SLURP dataset show that the proposed method achieves an average relative word error rate improvement of $28\%$ on unseen target domains without any performance drop in source domains.

{{</citation>}}


### (132/154) RECAP: Retrieval-Augmented Audio Captioning (Sreyan Ghosh et al., 2023)

{{<citation>}}

Sreyan Ghosh, Sonal Kumar, Chandra Kiran Reddy Evuru, Ramani Duraiswami, Dinesh Manocha. (2023)  
**RECAP: Retrieval-Augmented Audio Captioning**  

---
Primary Category: eess.AS  
Categories: cs-AI, cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2309.09836v1)  

---


**ABSTRACT**  
We present RECAP (REtrieval-Augmented Audio CAPtioning), a novel and effective audio captioning system that generates captions conditioned on an input audio and other captions similar to the audio retrieved from a datastore. Additionally, our proposed method can transfer to any domain without the need for any additional fine-tuning. To generate a caption for an audio sample, we leverage an audio-text model CLAP to retrieve captions similar to it from a replaceable datastore, which are then used to construct a prompt. Next, we feed this prompt to a GPT-2 decoder and introduce cross-attention layers between the CLAP encoder and GPT-2 to condition the audio for caption generation. Experiments on two benchmark datasets, Clotho and AudioCaps, show that RECAP achieves competitive performance in in-domain settings and significant improvements in out-of-domain settings. Additionally, due to its capability to exploit a large text-captions-only datastore in a \textit{training-free} fashion, RECAP shows unique capabilities of captioning novel audio events never seen during training and compositional audios with multiple events. To promote research in this space, we also release 150,000+ new weakly labeled captions for AudioSet, AudioCaps, and Clotho.

{{</citation>}}


### (133/154) Enhancing Multilingual Speech Recognition through Language Prompt Tuning and Frame-Level Language Adapter (Song Li et al., 2023)

{{<citation>}}

Song Li, Yongbin You, Xuezhi Wang, Ke Ding, Guanglu Wan. (2023)  
**Enhancing Multilingual Speech Recognition through Language Prompt Tuning and Frame-Level Language Adapter**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: ChatGPT, GPT, Multilingual, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.09443v2)  

---


**ABSTRACT**  
Multilingual intelligent assistants, such as ChatGPT, have recently gained popularity. To further expand the applications of multilingual artificial intelligence assistants and facilitate international communication, it is essential to enhance the performance of multilingual speech recognition, which is a crucial component of speech interaction. In this paper, we propose two simple and parameter-efficient methods: language prompt tuning and frame-level language adapter, to respectively enhance language-configurable and language-agnostic multilingual speech recognition. Additionally, we explore the feasibility of integrating these two approaches using parameter-efficient fine-tuning methods. Our experiments demonstrate significant performance improvements across seven languages using our proposed methods.

{{</citation>}}


## cs.CY (4)



### (134/154) Analyzing the Endeavours of the Supreme Court of India to Transcribe and Translate Court Arguments in Light of the Proposed EU AI Act (Kshitiz Verma, 2023)

{{<citation>}}

Kshitiz Verma. (2023)  
**Analyzing the Endeavours of the Supreme Court of India to Transcribe and Translate Court Arguments in Light of the Proposed EU AI Act**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.10088v1)  

---


**ABSTRACT**  
The Supreme Court of India has been a pioneer in using ICT in courts through its e-Courts project in India. Yet another leap, its recent project, Design, Development, and Implementation of Artificial Intelligence (AI) solution, tools for transcribing arguments and Court proceedings at Supreme Court of India, has potential to impact the way AI algorithms are designed in India, and not just for this particular project. In this paper, we evaluate the endeavours of the Supreme Court of India in light of the state of AI technology as well as the attempts to regulate AI. We argue that since the project aims to transcribe and translate the proceedings of the constitutional benches of the Supreme Court, it has potential to impact rule of law in the country. Hence, we place this application in High Risk AI as per the provisions to the proposed EU AI Act. We provide some guidelines on the approach to transcribe and translate making the maximum use of AI in the Supreme Court of India without running into the dangers it may pose.

{{</citation>}}


### (135/154) Evaluating the Impact of ChatGPT on Exercises of a Software Security Course (Jingyue Li et al., 2023)

{{<citation>}}

Jingyue Li, Per Håkon Meland, Jakob Svennevik Notland, André Storhaug, Jostein Hjortland Tysse. (2023)  
**Evaluating the Impact of ChatGPT on Exercises of a Software Security Course**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: ChatGPT, GPT, GPT-4, Security  
[Paper Link](http://arxiv.org/abs/2309.10085v1)  

---


**ABSTRACT**  
Along with the development of large language models (LLMs), e.g., ChatGPT, many existing approaches and tools for software security are changing. It is, therefore, essential to understand how security-aware these models are and how these models impact software security practices and education. In exercises of a software security course at our university, we ask students to identify and fix vulnerabilities we insert in a web application using state-of-the-art tools. After ChatGPT, especially the GPT-4 version of the model, we want to know how the students can possibly use ChatGPT to complete the exercise tasks. We input the vulnerable code to ChatGPT and measure its accuracy in vulnerability identification and fixing. In addition, we investigated whether ChatGPT can provide a proper source of information to support its outputs. Results show that ChatGPT can identify 20 of the 28 vulnerabilities we inserted in the web application in a white-box setting, reported three false positives, and found four extra vulnerabilities beyond the ones we inserted. ChatGPT makes nine satisfactory penetration testing and fixing recommendations for the ten vulnerabilities we want students to fix and can often point to related sources of information.

{{</citation>}}


### (136/154) ArxNet Model and Data: Building Social Networks from Image Archives (Haley Seaward et al., 2023)

{{<citation>}}

Haley Seaward, Jasmine Talley, David Beskow. (2023)  
**ArxNet Model and Data: Building Social Networks from Image Archives**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2309.09775v1)  

---


**ABSTRACT**  
A corresponding explosion in digital images has accompanied the rapid adoption of mobile technology around the world. People and their activities are routinely captured in digital image and video files. By their very nature, these images and videos often portray social and professional connections. Individuals in the same picture are often connected in some meaningful way. Our research seeks to identify and model social connections found in images using modern face detection technology and social network analysis. The proposed methods are then demonstrated on the public image repository associated with the 2022 Emmy's Award Presentation.

{{</citation>}}


### (137/154) Are You Worthy of My Trust?: A Socioethical Perspective on the Impacts of Trustworthy AI Systems on the Environment and Human Society (Jamell Dacon, 2023)

{{<citation>}}

Jamell Dacon. (2023)  
**Are You Worthy of My Trust?: A Socioethical Perspective on the Impacts of Trustworthy AI Systems on the Environment and Human Society**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-HC, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09450v1)  

---


**ABSTRACT**  
With ubiquitous exposure of AI systems today, we believe AI development requires crucial considerations to be deemed trustworthy. While the potential of AI systems is bountiful, though, is still unknown-as are their risks. In this work, we offer a brief, high-level overview of societal impacts of AI systems. To do so, we highlight the requirement of multi-disciplinary governance and convergence throughout its lifecycle via critical systemic examinations (e.g., energy consumption), and later discuss induced effects on the environment (i.e., carbon footprint) and its users (i.e., social development). In particular, we consider these impacts from a multi-disciplinary perspective: computer science, sociology, environmental science, and so on to discuss its inter-connected societal risks and inability to simultaneously satisfy aspects of well-being. Therefore, we accentuate the necessity of holistically addressing pressing concerns of AI systems from a socioethical impact assessment perspective to explicate its harmful societal effects to truly enable humanity-centered Trustworthy AI.

{{</citation>}}


## cs.DS (1)



### (138/154) Simple and Optimal Online Contention Resolution Schemes for $k$-Uniform Matroids (Atanas Dinev et al., 2023)

{{<citation>}}

Atanas Dinev, S. Matthew Weinberg. (2023)  
**Simple and Optimal Online Contention Resolution Schemes for $k$-Uniform Matroids**  

---
Primary Category: cs.DS  
Categories: cs-DM, cs-DS, cs-GT, cs.DS  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2309.10078v1)  

---


**ABSTRACT**  
We provide a simple $(1-O(\frac{1}{\sqrt{k}}))$-selectable Online Contention Resolution Scheme for $k$-uniform matroids against a fixed-order adversary. If $A_i$ and $G_i$ denote the set of selected elements and the set of realized active elements among the first $i$ (respectively), our algorithm selects with probability $1-\frac{1}{\sqrt{k}}$ any active element $i$ such that $|A_{i-1}| + 1 \leq (1-\frac{1}{\sqrt{k}})\cdot \mathbb{E}[|G_i|]+\sqrt{k}$. This implies a $(1-O(\frac{1}{\sqrt{k}}))$ prophet inequality against fixed-order adversaries for $k$-uniform matroids that is considerably simpler than previous algorithms [Ala14, AKW14, JMZ22].   We also prove that no OCRS can be $(1-\Omega(\sqrt{\frac{\log k}{k}}))$-selectable for $k$-uniform matroids against an almighty adversary. This guarantee is matched by the (known) simple greedy algorithm that accepts every active element with probability $1-\Theta(\sqrt{\frac{\log k}{k}})$ [HKS07].

{{</citation>}}


## math.NA (1)



### (139/154) Recycling Krylov Subspaces for Efficient Partitioned Solution of Aerostructural Adjoint Systems (Christophe Blondeau, 2023)

{{<citation>}}

Christophe Blondeau. (2023)  
**Recycling Krylov Subspaces for Efficient Partitioned Solution of Aerostructural Adjoint Systems**  

---
Primary Category: math.NA  
Categories: 65F10, 65F08, G-1-3, cs-NA, math-NA, math.NA  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2309.09925v2)  

---


**ABSTRACT**  
Robust and efficient solvers for coupled-adjoint linear systems are crucial to successful aerostructural optimization. Monolithic and partitioned strategies can be applied. The monolithic approach is expected to offer better robustness and efficiency for strong fluid-structure interactions. However, it requires a high implementation cost and convergence may depend on appropriate scaling and initialization strategies. On the other hand, the modularity of the partitioned method enables a straightforward implementation while its convergence may require relaxation. In addition, a partitioned solver leads to a higher number of iterations to get the same level of convergence as the monolithic one.   The objective of this paper is to accelerate the fluid-structure coupled-adjoint partitioned solver by considering techniques borrowed from approximate invariant subspace recycling strategies adapted to sequences of linear systems with varying right-hand sides. Indeed, in a partitioned framework, the structural source term attached to the fluid block of equations affects the right-hand side with the nice property of quickly converging to a constant value. We also consider deflation of approximate eigenvectors in conjunction with advanced inner-outer Krylov solvers for the fluid block equations. We demonstrate the benefit of these techniques by computing the coupled derivatives of an aeroelastic configuration of the ONERA-M6 fixed wing in transonic flow. For this exercise the fluid grid was coupled to a structural model specifically designed to exhibit a high flexibility. All computations are performed using RANS flow modeling and a fully linearized one-equation Spalart-Allmaras turbulence model. Numerical simulations show up to 39% reduction in matrix-vector products for GCRO-DR and up to 19% for the nested FGCRO-DR solver.

{{</citation>}}


## stat.ML (1)



### (140/154) Error Reduction from Stacked Regressions (Xin Chen et al., 2023)

{{<citation>}}

Xin Chen, Jason M. Klusowski, Yan Shuo Tan. (2023)  
**Error Reduction from Stacked Regressions**  

---
Primary Category: stat.ML  
Categories: cs-LG, stat-ML, stat.ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09880v1)  

---


**ABSTRACT**  
Stacking regressions is an ensemble technique that forms linear combinations of different regression estimators to enhance predictive accuracy. The conventional approach uses cross-validation data to generate predictions from the constituent estimators, and least-squares with nonnegativity constraints to learn the combination weights. In this paper, we learn these weights analogously by minimizing an estimate of the population risk subject to a nonnegativity constraint. When the constituent estimators are linear least-squares projections onto nested subspaces separated by at least three dimensions, we show that thanks to a shrinkage effect, the resulting stacked estimator has strictly smaller population risk than best single estimator among them. Here ``best'' refers to a model that minimizes a selection criterion such as AIC or BIC. In other words, in this setting, the best single estimator is inadmissible. Because the optimization problem can be reformulated as isotonic regression, the stacked estimator requires the same order of computation as the best single estimator, making it an attractive alternative in terms of both performance and implementation.

{{</citation>}}


## cs.SE (2)



### (141/154) EGFE: End-to-end Grouping of Fragmented Elements in UI Designs with Multimodal Learning (Liuqing Chen et al., 2023)

{{<citation>}}

Liuqing Chen, Yunnong Chen, Shuhong Xiao, Yaxuan Song, Lingyun Sun, Yankun Zhen, Tingting Zhou, Yanfang Chang. (2023)  
**EGFE: End-to-end Grouping of Fragmented Elements in UI Designs with Multimodal Learning**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.09867v1)  

---


**ABSTRACT**  
When translating UI design prototypes to code in industry, automatically generating code from design prototypes can expedite the development of applications and GUI iterations. However, in design prototypes without strict design specifications, UI components may be composed of fragmented elements. Grouping these fragmented elements can greatly improve the readability and maintainability of the generated code. Current methods employ a two-stage strategy that introduces hand-crafted rules to group fragmented elements. Unfortunately, the performance of these methods is not satisfying due to visually overlapped and tiny UI elements. In this study, we propose EGFE, a novel method for automatically End-to-end Grouping Fragmented Elements via UI sequence prediction. To facilitate the UI understanding, we innovatively construct a Transformer encoder to model the relationship between the UI elements with multi-modal representation learning. The evaluation on a dataset of 4606 UI prototypes collected from professional UI designers shows that our method outperforms the state-of-the-art baselines in the precision (by 29.75\%), recall (by 31.07\%), and F1-score (by 30.39\%) at edit distance threshold of 4. In addition, we conduct an empirical study to assess the improvement of the generated front-end code. The results demonstrate the effectiveness of our method on a real software engineering application. Our end-to-end fragmented elements grouping method creates opportunities for improving UI-related software engineering tasks.

{{</citation>}}


### (142/154) TOPr: Enhanced Static Code Pruning for Fast and Precise Directed Fuzzing (Chaitra Niddodi et al., 2023)

{{<citation>}}

Chaitra Niddodi, Stefan Nagy, Darko Marinov, Sibin Mohan. (2023)  
**TOPr: Enhanced Static Code Pruning for Fast and Precise Directed Fuzzing**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2309.09522v1)  

---


**ABSTRACT**  
Directed fuzzing is a dynamic testing technique that focuses exploration on specific, pre targeted program locations. Like other types of fuzzers, directed fuzzers are most effective when maximizing testing speed and precision. To this end, recent directed fuzzers have begun leveraging path pruning: preventing the wasteful testing of program paths deemed irrelevant to reaching a desired target location. Yet, despite code pruning's substantial speedup, current approaches are imprecise failing to capture indirect control flow requiring additional dynamic analyses that diminish directed fuzzers' speeds. Thus, without code pruning that is both fast and precise, directed fuzzers' effectiveness will continue to remain limited. This paper aims to tackle the challenge of upholding both speed and precision in pruning-based directed fuzzing. We show that existing pruning approaches fail to recover common case indirect control flow; and identify opportunities to enhance them with lightweight heuristics namely, function signature matching enabling them to maximize precision without the burden of dynamic analysis. We implement our enhanced pruning as a prototype, TOPr (Target Oriented Pruning), and evaluate it against the leading pruning based and pruning agnostic directed fuzzers SieveFuzz and AFLGo. We show that TOPr's enhanced pruning outperforms these fuzzers in (1) speed (achieving 222% and 73% higher test case throughput, respectively); (2) reachability (achieving 149% and 9% more target relevant coverage, respectively); and (3) bug discovery time (triggering bugs faster 85% and 8%, respectively). Furthermore, TOPr's balance of speed and precision enables it to find 24 new bugs in 5 open source applications, with 18 confirmed by developers, 12 bugs labelled as "Priority - 1. High", and 12 bugs fixed, underscoring the effectiveness of our framework.

{{</citation>}}


## cs.SD (4)



### (143/154) Frame-to-Utterance Convergence: A Spectra-Temporal Approach for Unified Spoofing Detection (Awais Khan et al., 2023)

{{<citation>}}

Awais Khan, Khalid Mahmood Malik, Shah Nawaz. (2023)  
**Frame-to-Utterance Convergence: A Spectra-Temporal Approach for Unified Spoofing Detection**  

---
Primary Category: cs.SD  
Categories: cs-CY, cs-SD, cs.SD, eess-AS  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.09837v1)  

---


**ABSTRACT**  
Voice spoofing attacks pose a significant threat to automated speaker verification systems. Existing anti-spoofing methods often simulate specific attack types, such as synthetic or replay attacks. However, in real-world scenarios, the countermeasures are unaware of the generation schema of the attack, necessitating a unified solution. Current unified solutions struggle to detect spoofing artifacts, especially with recent spoofing mechanisms. For instance, the spoofing algorithms inject spectral or temporal anomalies, which are challenging to identify. To this end, we present a spectra-temporal fusion leveraging frame-level and utterance-level coefficients. We introduce a novel local spectral deviation coefficient (SDC) for frame-level inconsistencies and employ a bi-LSTM-based network for sequential temporal coefficients (STC), which capture utterance-level artifacts. Our spectra-temporal fusion strategy combines these coefficients, and an auto-encoder generates spectra-temporal deviated coefficients (STDC) to enhance robustness. Our proposed approach addresses multiple spoofing categories, including synthetic, replay, and partial deepfake attacks. Extensive evaluation on diverse datasets (ASVspoof2019, ASVspoof2021, VSDC, partial spoofs, and in-the-wild deepfakes) demonstrated its robustness for a wide range of voice applications.

{{</citation>}}


### (144/154) Electrolaryngeal Speech Intelligibility Enhancement Through Robust Linguistic Encoders (Lester Phillip Violeta et al., 2023)

{{<citation>}}

Lester Phillip Violeta, Wen-Chin Huang, Ding Ma, Ryuichi Yamamoto, Kazuhiro Kobayashi, Tomoki Toda. (2023)  
**Electrolaryngeal Speech Intelligibility Enhancement Through Robust Linguistic Encoders**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.09627v1)  

---


**ABSTRACT**  
We propose a novel framework for electrolaryngeal speech intelligibility enhancement through the use of robust linguistic encoders. Pretraining and fine-tuning approaches have proven to work well in this task, but in most cases, various mismatches, such as the speech type mismatch (electrolaryngeal vs. typical) or a speaker mismatch between the datasets used in each stage, can deteriorate the conversion performance of this framework. To resolve this issue, we propose a linguistic encoder robust enough to project both EL and typical speech in the same latent space, while still being able to extract accurate linguistic information, creating a unified representation to reduce the speech type mismatch. Furthermore, we introduce HuBERT output features to the proposed framework for reducing the speaker mismatch, making it possible to effectively use a large-scale parallel dataset during pretraining. We show that compared to the conventional framework using mel-spectrogram input and output features, using the proposed framework enables the model to synthesize more intelligible and naturally sounding speech, as shown by a significant 16% improvement in character error rate and 0.83 improvement in naturalness score.

{{</citation>}}


### (145/154) Face-Driven Zero-Shot Voice Conversion with Memory-based Face-Voice Alignment (Zheng-Yan Sheng et al., 2023)

{{<citation>}}

Zheng-Yan Sheng, Yang Ai, Yan-Nian Chen, Zhen-Hua Ling. (2023)  
**Face-Driven Zero-Shot Voice Conversion with Memory-based Face-Voice Alignment**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.09470v1)  

---


**ABSTRACT**  
This paper presents a novel task, zero-shot voice conversion based on face images (zero-shot FaceVC), which aims at converting the voice characteristics of an utterance from any source speaker to a newly coming target speaker, solely relying on a single face image of the target speaker. To address this task, we propose a face-voice memory-based zero-shot FaceVC method. This method leverages a memory-based face-voice alignment module, in which slots act as the bridge to align these two modalities, allowing for the capture of voice characteristics from face images. A mixed supervision strategy is also introduced to mitigate the long-standing issue of the inconsistency between training and inference phases for voice conversion tasks. To obtain speaker-independent content-related representations, we transfer the knowledge from a pretrained zero-shot voice conversion model to our zero-shot FaceVC model. Considering the differences between FaceVC and traditional voice conversion tasks, systematic subjective and objective metrics are designed to thoroughly evaluate the homogeneity, diversity and consistency of voice characteristics controlled by face images. Through extensive experiments, we demonstrate the superiority of our proposed method on the zero-shot FaceVC task. Samples are presented on our demo website.

{{</citation>}}


### (146/154) Are Soft Prompts Good Zero-shot Learners for Speech Recognition? (Dianwen Ng et al., 2023)

{{<citation>}}

Dianwen Ng, Chong Zhang, Ruixi Zhang, Yukun Ma, Fabian Ritter-Gutierrez, Trung Hieu Nguyen, Chongjia Ni, Shengkui Zhao, Eng Siong Chng, Bin Ma. (2023)  
**Are Soft Prompts Good Zero-shot Learners for Speech Recognition?**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.09413v1)  

---


**ABSTRACT**  
Large self-supervised pre-trained speech models require computationally expensive fine-tuning for downstream tasks. Soft prompt tuning offers a simple parameter-efficient alternative by utilizing minimal soft prompt guidance, enhancing portability while also maintaining competitive performance. However, not many people understand how and why this is so. In this study, we aim to deepen our understanding of this emerging method by investigating the role of soft prompts in automatic speech recognition (ASR). Our findings highlight their role as zero-shot learners in improving ASR performance but also make them vulnerable to malicious modifications. Soft prompts aid generalization but are not obligatory for inference. We also identify two primary roles of soft prompts: content refinement and noise information enhancement, which enhances robustness against background noise. Additionally, we propose an effective modification on noise prompts to show that they are capable of zero-shot learning on adapting to out-of-distribution noise environments.

{{</citation>}}


## cs.IR (2)



### (147/154) Predictive Uncertainty-based Bias Mitigation in Ranking (Maria Heuss et al., 2023)

{{<citation>}}

Maria Heuss, Daniel Cohen, Masoud Mansoury, Maarten de Rijke, Carsten Eickhoff. (2023)  
**Predictive Uncertainty-based Bias Mitigation in Ranking**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.09833v1)  

---


**ABSTRACT**  
Societal biases that are contained in retrieved documents have received increased interest. Such biases, which are often prevalent in the training data and learned by the model, can cause societal harms, by misrepresenting certain groups, and by enforcing stereotypes. Mitigating such biases demands algorithms that balance the trade-off between maximized utility for the user with fairness objectives, which incentivize unbiased rankings. Prior work on bias mitigation often assumes that ranking scores, which correspond to the utility that a document holds for a user, can be accurately determined. In reality, there is always a degree of uncertainty in the estimate of expected document utility. This uncertainty can be approximated by viewing ranking models through a Bayesian perspective, where the standard deterministic score becomes a distribution.   In this work, we investigate whether uncertainty estimates can be used to decrease the amount of bias in the ranked results, while minimizing loss in measured utility. We introduce a simple method that uses the uncertainty of the ranking scores for an uncertainty-aware, post hoc approach to bias mitigation. We compare our proposed method with existing baselines for bias mitigation with respect to the utility-fairness trade-off, the controllability of methods, and computational costs. We show that an uncertainty-based approach can provide an intuitive and flexible trade-off that outperforms all baselines without additional training requirements, allowing for the post hoc use of this approach on top of arbitrary retrieval models.

{{</citation>}}


### (148/154) Selecting which Dense Retriever to use for Zero-Shot Search (Ekaterina Khramtsova et al., 2023)

{{<citation>}}

Ekaterina Khramtsova, Shengyao Zhuang, Mahsa Baktashmotlagh, Xi Wang, Guido Zuccon. (2023)  
**Selecting which Dense Retriever to use for Zero-Shot Search**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.09403v1)  

---


**ABSTRACT**  
We propose the new problem of choosing which dense retrieval model to use when searching on a new collection for which no labels are available, i.e. in a zero-shot setting. Many dense retrieval models are readily available. Each model however is characterized by very differing search effectiveness -- not just on the test portion of the datasets in which the dense representations have been learned but, importantly, also across different datasets for which data was not used to learn the dense representations. This is because dense retrievers typically require training on a large amount of labeled data to achieve satisfactory search effectiveness in a specific dataset or domain. Moreover, effectiveness gains obtained by dense retrievers on datasets for which they are able to observe labels during training, do not necessarily generalise to datasets that have not been observed during training. This is however a hard problem: through empirical experimentation we show that methods inspired by recent work in unsupervised performance evaluation with the presence of domain shift in the area of computer vision and machine learning are not effective for choosing highly performing dense retrievers in our setup. The availability of reliable methods for the selection of dense retrieval models in zero-shot settings that do not require the collection of labels for evaluation would allow to streamline the widespread adoption of dense retrieval. This is therefore an important new problem we believe the information retrieval community should consider. Implementation of methods, along with raw result files and analysis scripts are made publicly available at https://www.github.com/anonymized.

{{</citation>}}


## cs.DL (2)



### (149/154) When Large Language Models Meet Citation: A Survey (Yang Zhang et al., 2023)

{{<citation>}}

Yang Zhang, Yufei Wang, Kai Wang, Quan Z. Sheng, Lina Yao, Adnan Mahmood, Wei Emma Zhang, Rongying Zhao. (2023)  
**When Large Language Models Meet Citation: A Survey**  

---
Primary Category: cs.DL  
Categories: cs-CL, cs-DL, cs.DL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.09727v1)  

---


**ABSTRACT**  
Citations in scholarly work serve the essential purpose of acknowledging and crediting the original sources of knowledge that have been incorporated or referenced. Depending on their surrounding textual context, these citations are used for different motivations and purposes. Large Language Models (LLMs) could be helpful in capturing these fine-grained citation information via the corresponding textual context, thereby enabling a better understanding towards the literature. Furthermore, these citations also establish connections among scientific papers, providing high-quality inter-document relationships and human-constructed knowledge. Such information could be incorporated into LLMs pre-training and improve the text representation in LLMs. Therefore, in this paper, we offer a preliminary review of the mutually beneficial relationship between LLMs and citation analysis. Specifically, we review the application of LLMs for in-text citation analysis tasks, including citation classification, citation-based summarization, and citation recommendation. We then summarize the research pertinent to leveraging citation linkage knowledge to improve text representations of LLMs via citation prediction, network structure information, and inter-document relationship. We finally provide an overview of these contemporary methods and put forth potential promising avenues in combining LLMs and citation analysis for further investigation.

{{</citation>}}


### (150/154) Multi-Affiliated Authors Behave Differently across Fields and Host Country Preferences: A Comparison in G7 and BRICS (Sichao Tong et al., 2023)

{{<citation>}}

Sichao Tong, Liying Yang. (2023)  
**Multi-Affiliated Authors Behave Differently across Fields and Host Country Preferences: A Comparison in G7 and BRICS**  

---
Primary Category: cs.DL  
Categories: cs-DL, cs.DL  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2309.09449v1)  

---


**ABSTRACT**  
This paper study author simultaneously engaged in multiple affiliations based on bibliometric data covered in the Web of Science for the 2017-2021 period. Based on the affiliation information in publication records, we propose a general classification for multiple affiliations within-country or cross-country for analyzing authors' behavior in multiple affiliations and preferences of host countries across research fields. We find a decrease in publications led by international multi-affiliated authorship after 2020, and China has shown a falling trend after 2018. More G7 countries are active in fields like Social Sciences, Clinical and Life Sciences. China, India, and Russia are active in physical sciences-related fields. Countries prefer to affiliate with G7 countries, especially in Clinical and Life Sciences. These findings may provide more insights into the understanding of the behavior and productivity of multi-affiliated researchers in the current academic landscape.

{{</citation>}}


## cs.IT (1)



### (151/154) Turbo Coded OFDM-OQAM Using Hilbert Transform (Kasturi Vasudevan et al., 2023)

{{<citation>}}

Kasturi Vasudevan, Surendra Kota, Lov Kumar, Himanshu Bhusan Mishra. (2023)  
**Turbo Coded OFDM-OQAM Using Hilbert Transform**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.09620v1)  

---


**ABSTRACT**  
Orthogonal frequency division multiplexing (OFDM) with offset quadrature amplitude modulation (OQAM) has been widely discussed in the literature and is considered a popular waveform for 5th generation (5G) wireless telecommunications and beyond. In this work, we show that OFDM-OQAM can be generated using the Hilbert transform and is equivalent to single sideband modulation (SSB), that has roots in analog telecommunications. The transmit filter for OFDM-OQAM is complex valued whose real part is given by the pulse corresponding to the root raised cosine spectrum and the imaginary part is the Hilbert transform of the real part. The real-valued digital information (message) are passed through the transmit filter and frequency division multiplexed on orthogonal subcarriers. The message bandwidth corresponding to each subcarrier is assumed to be narrow enough so that the channel can be considered ideal. Therefore, at the receiver, a matched filter can used to recover the message. Turbo coding is used to achieve bit-error-rate (BER) as low as $10^{-5}$ at an average signal-to-noise ratio (SNR) per bit close to 0 db. The system has been simulated in discrete time.

{{</citation>}}


## eess.SP (1)



### (152/154) AI-Native Transceiver Design for Near-Field Ultra-Massive MIMO: Principles and Techniques (Wentao Yu et al., 2023)

{{<citation>}}

Wentao Yu, Yifan Ma, Hengtao He, Shenghui Song, Jun Zhang, Khaled B. Letaief. (2023)  
**AI-Native Transceiver Design for Near-Field Ultra-Massive MIMO: Principles and Techniques**  

---
Primary Category: eess.SP  
Categories: cs-IT, eess-SP, eess.SP, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.09575v1)  

---


**ABSTRACT**  
Ultra-massive multiple-input multiple-output (UM-MIMO) is a cutting-edge technology that promises to revolutionize wireless networks by providing an unprecedentedly high spectral and energy efficiency. The enlarged array aperture of UM-MIMO facilitates the accessibility of the near-field region, thereby offering a novel degree of freedom for communications and sensing. Nevertheless, the transceiver design for such systems is challenging because of the enormous system scale, the complicated channel characteristics, and the uncertainties in propagation environments. Therefore, it is critical to study scalable, low-complexity, and robust algorithms that can efficiently characterize and leverage the properties of the near-field channel. In this article, we will advocate two general frameworks from an artificial intelligence (AI)-native perspective, which are tailored for the algorithmic design of near-field UM-MIMO transceivers. Specifically, the frameworks for both iterative and non-iterative algorithms are discussed. Near-field beam focusing and channel estimation are presented as two tutorial-style examples to demonstrate the significant advantages of the proposed AI-native frameworks in terms of various key performance indicators.

{{</citation>}}


## cs.NE (1)



### (153/154) Adaptive Reorganization of Neural Pathways for Continual Learning with Hybrid Spiking Neural Networks (Bing Han et al., 2023)

{{<citation>}}

Bing Han, Feifei Zhao, Wenxuan Pan, Zhaoya Zhao, Xianqi Li, Qingqun Kong, Yi Zeng. (2023)  
**Adaptive Reorganization of Neural Pathways for Continual Learning with Hybrid Spiking Neural Networks**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-NE, cs.NE  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.09550v1)  

---


**ABSTRACT**  
The human brain can self-organize rich and diverse sparse neural pathways to incrementally master hundreds of cognitive tasks. However, most existing continual learning algorithms for deep artificial and spiking neural networks are unable to adequately auto-regulate the limited resources in the network, which leads to performance drop along with energy consumption rise as the increase of tasks. In this paper, we propose a brain-inspired continual learning algorithm with adaptive reorganization of neural pathways, which employs Self-Organizing Regulation networks to reorganize the single and limited Spiking Neural Network (SOR-SNN) into rich sparse neural pathways to efficiently cope with incremental tasks. The proposed model demonstrates consistent superiority in performance, energy consumption, and memory capacity on diverse continual learning tasks ranging from child-like simple to complex tasks, as well as on generalized CIFAR100 and ImageNet datasets. In particular, the SOR-SNN model excels at learning more complex tasks as well as more tasks, and is able to integrate the past learned knowledge with the information from the current task, showing the backward transfer ability to facilitate the old tasks. Meanwhile, the proposed model exhibits self-repairing ability to irreversible damage and for pruned networks, could automatically allocate new pathway from the retained network to recover memory for forgotten knowledge.

{{</citation>}}


## cs.AR (1)



### (154/154) From RTL to SVA: LLM-assisted generation of Formal Verification Testbenches (Marcelo Orenes-Vera et al., 2023)

{{<citation>}}

Marcelo Orenes-Vera, Margaret Martonosi, David Wentzlaff. (2023)  
**From RTL to SVA: LLM-assisted generation of Formal Verification Testbenches**  

---
Primary Category: cs.AR  
Categories: cs-AR, cs-SE, cs.AR  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2309.09437v1)  

---


**ABSTRACT**  
Formal property verification (FPV) has existed for decades and has been shown to be effective at finding intricate RTL bugs. However, formal properties, such as those written as SystemVerilog Assertions (SVA), are time-consuming and error-prone to write, even for experienced users. Prior work has attempted to lighten this burden by raising the abstraction level so that SVA is generated from high-level specifications. However, this does not eliminate the manual effort of reasoning and writing about the detailed hardware behavior. Motivated by the increased need for FPV in the era of heterogeneous hardware and the advances in large language models (LLMs), we set out to explore whether LLMs can capture RTL behavior and generate correct SVA properties.   First, we design an FPV-based evaluation framework that measures the correctness and completeness of SVA. Then, we evaluate GPT4 iteratively to craft the set of syntax and semantic rules needed to prompt it toward creating better SVA. We extend the open-source AutoSVA framework by integrating our improved GPT4-based flow to generate safety properties, in addition to facilitating their existing flow for liveness properties. Lastly, our use cases evaluate (1) the FPV coverage of GPT4-generated SVA on complex open-source RTL and (2) using generated SVA to prompt GPT4 to create RTL from scratch.   Through these experiments, we find that GPT4 can generate correct SVA even for flawed RTL, without mirroring design errors. Particularly, it generated SVA that exposed a bug in the RISC-V CVA6 core that eluded the prior work's evaluation.

{{</citation>}}
