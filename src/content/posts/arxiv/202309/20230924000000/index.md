---
draft: false
title: "arXiv @ 2023.09.24"
date: 2023-09-24
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.24"
    identifier: arxiv_20230924
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (24)](#cscv-24)
- [cs.AI (12)](#csai-12)
- [hep-ex (1)](#hep-ex-1)
- [cs.CL (27)](#cscl-27)
- [cs.HC (4)](#cshc-4)
- [cs.LG (18)](#cslg-18)
- [cs.IR (3)](#csir-3)
- [cs.CR (2)](#cscr-2)
- [q-bio.BM (1)](#q-biobm-1)
- [eess.SY (1)](#eesssy-1)
- [cs.CE (1)](#csce-1)
- [eess.AS (8)](#eessas-8)
- [eess.IV (1)](#eessiv-1)
- [cs.SE (3)](#csse-3)
- [cs.MA (1)](#csma-1)
- [cs.NE (2)](#csne-2)
- [cs.AR (1)](#csar-1)
- [cs.RO (4)](#csro-4)
- [cs.GL (1)](#csgl-1)
- [cs.SI (3)](#cssi-3)
- [quant-ph (1)](#quant-ph-1)
- [cs.CY (1)](#cscy-1)

## cs.CV (24)



### (1/120) Poster: Self-Supervised Quantization-Aware Knowledge Distillation (Kaiqi Zhao et al., 2023)

{{<citation>}}

Kaiqi Zhao, Ming Zhao. (2023)  
**Poster: Self-Supervised Quantization-Aware Knowledge Distillation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Knowledge Distillation, QA, Quantization, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.13220v1)  

---


**ABSTRACT**  
Quantization-aware training (QAT) starts with a pre-trained full-precision model and performs quantization during retraining. However, existing QAT works require supervision from the labels and they suffer from accuracy loss due to reduced precision. To address these limitations, this paper proposes a novel Self-Supervised Quantization-Aware Knowledge Distillation framework (SQAKD). SQAKD first unifies the forward and backward dynamics of various quantization functions and then reframes QAT as a co-optimization problem that simultaneously minimizes the KL-Loss and the discretization error, in a self-supervised manner. The evaluation shows that SQAKD significantly improves the performance of various state-of-the-art QAT works. SQAKD establishes stronger baselines and does not require extensive labeled training data, potentially making state-of-the-art QAT research more accessible.

{{</citation>}}


### (2/120) ClusterFormer: Clustering As A Universal Visual Learner (James C. Liang et al., 2023)

{{<citation>}}

James C. Liang, Yiming Cui, Qifan Wang, Tong Geng, Wenguan Wang, Dongfang Liu. (2023)  
**ClusterFormer: Clustering As A Universal Visual Learner**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2309.13196v1)  

---


**ABSTRACT**  
This paper presents CLUSTERFORMER, a universal vision model that is based on the CLUSTERing paradigm with TransFORMER. It comprises two novel designs: 1. recurrent cross-attention clustering, which reformulates the cross-attention mechanism in Transformer and enables recursive updates of cluster centers to facilitate strong representation learning; and 2. feature dispatching, which uses the updated cluster centers to redistribute image features through similarity-based metrics, resulting in a transparent pipeline. This elegant design streamlines an explainable and transferable workflow, capable of tackling heterogeneous vision tasks (i.e., image classification, object detection, and image segmentation) with varying levels of clustering granularity (i.e., image-, box-, and pixel-level). Empirical results demonstrate that CLUSTERFORMER outperforms various well-known specialized architectures, achieving 83.41% top-1 acc. over ImageNet-1K for image classification, 54.2% and 47.0% mAP over MS COCO for object detection and instance segmentation, 52.4% mIoU over ADE20K for semantic segmentation, and 55.8% PQ over COCO Panoptic for panoptic segmentation. For its efficacy, we hope our work can catalyze a paradigm shift in universal models in computer vision.

{{</citation>}}


### (3/120) Contextual Emotion Estimation from Image Captions (Vera Yang et al., 2023)

{{<citation>}}

Vera Yang, Archita Srivastava, Yasaman Etesam, Chuxuan Zhang, Angelica Lim. (2023)  
**Contextual Emotion Estimation from Image Captions**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: GPT, GPT-3.5, Language Model  
[Paper Link](http://arxiv.org/abs/2309.13136v1)  

---


**ABSTRACT**  
Emotion estimation in images is a challenging task, typically using computer vision methods to directly estimate people's emotions using face, body pose and contextual cues. In this paper, we explore whether Large Language Models (LLMs) can support the contextual emotion estimation task, by first captioning images, then using an LLM for inference. First, we must understand: how well do LLMs perceive human emotions? And which parts of the information enable them to determine emotions? One initial challenge is to construct a caption that describes a person within a scene with information relevant for emotion perception. Towards this goal, we propose a set of natural language descriptors for faces, bodies, interactions, and environments. We use them to manually generate captions and emotion annotations for a subset of 331 images from the EMOTIC dataset. These captions offer an interpretable representation for emotion estimation, towards understanding how elements of a scene affect emotion perception in LLMs and beyond. Secondly, we test the capability of a large language model to infer an emotion from the resulting image captions. We find that GPT-3.5, specifically the text-davinci-003 model, provides surprisingly reasonable emotion predictions consistent with human annotations, but accuracy can depend on the emotion concept. Overall, the results suggest promise in the image captioning and LLM approach.

{{</citation>}}


### (4/120) Understanding Calibration of Deep Neural Networks for Medical Image Classification (Abhishek Singh Sambyal et al., 2023)

{{<citation>}}

Abhishek Singh Sambyal, Usma Niyaz, Narayanan C. Krishnan, Deepti R. Bathula. (2023)  
**Understanding Calibration of Deep Neural Networks for Medical Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2309.13132v1)  

---


**ABSTRACT**  
In the field of medical image analysis, achieving high accuracy is not enough; ensuring well-calibrated predictions is also crucial. Confidence scores of a deep neural network play a pivotal role in explainability by providing insights into the model's certainty, identifying cases that require attention, and establishing trust in its predictions. Consequently, the significance of a well-calibrated model becomes paramount in the medical imaging domain, where accurate and reliable predictions are of utmost importance. While there has been a significant effort towards training modern deep neural networks to achieve high accuracy on medical imaging tasks, model calibration and factors that affect it remain under-explored. To address this, we conducted a comprehensive empirical study that explores model performance and calibration under different training regimes. We considered fully supervised training, which is the prevailing approach in the community, as well as rotation-based self-supervised method with and without transfer learning, across various datasets and architecture sizes. Multiple calibration metrics were employed to gain a holistic understanding of model calibration. Our study reveals that factors such as weight distributions and the similarity of learned representations correlate with the calibration trends observed in the models. Notably, models trained using rotation-based self-supervised pretrained regime exhibit significantly better calibration while achieving comparable or even superior performance compared to fully supervised models across different medical imaging datasets. These findings shed light on the importance of model calibration in medical image analysis and highlight the benefits of incorporating self-supervised learning approach to improve both performance and calibration.

{{</citation>}}


### (5/120) Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception? (Xiaoxiao Sun et al., 2023)

{{<citation>}}

Xiaoxiao Sun, Nidham Gazagnadou, Vivek Sharma, Lingjuan Lyu, Hongdong Li, Liang Zheng. (2023)  
**Privacy Assessment on Reconstructed Images: Are Existing Evaluation Metrics Faithful to Human Perception?**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Similarity  
[Paper Link](http://arxiv.org/abs/2309.13038v1)  

---


**ABSTRACT**  
Hand-crafted image quality metrics, such as PSNR and SSIM, are commonly used to evaluate model privacy risk under reconstruction attacks. Under these metrics, reconstructed images that are determined to resemble the original one generally indicate more privacy leakage. Images determined as overall dissimilar, on the other hand, indicate higher robustness against attack. However, there is no guarantee that these metrics well reflect human opinions, which, as a judgement for model privacy leakage, are more trustworthy. In this paper, we comprehensively study the faithfulness of these hand-crafted metrics to human perception of privacy information from the reconstructed images. On 5 datasets ranging from natural images, faces, to fine-grained classes, we use 4 existing attack methods to reconstruct images from many different classification models and, for each reconstructed image, we ask multiple human annotators to assess whether this image is recognizable. Our studies reveal that the hand-crafted metrics only have a weak correlation with the human evaluation of privacy leakage and that even these metrics themselves often contradict each other. These observations suggest risks of current metrics in the community. To address this potential risk, we propose a learning-based measure called SemSim to evaluate the Semantic Similarity between the original and reconstructed images. SemSim is trained with a standard triplet loss, using an original image as an anchor, one of its recognizable reconstructed images as a positive sample, and an unrecognizable one as a negative. By training on human annotations, SemSim exhibits a greater reflection of privacy leakage on the semantic level. We show that SemSim has a significantly higher correlation with human judgment compared with existing metrics. Moreover, this strong correlation generalizes to unseen datasets, models and attack methods.

{{</citation>}}


### (6/120) Deep3DSketch+: Rapid 3D Modeling from Single Free-hand Sketches (Tianrun Chen et al., 2023)

{{<citation>}}

Tianrun Chen, Chenglong Fu, Ying Zang, Lanyun Zhu, Jia Zhang, Papa Mao, Lingyun Sun. (2023)  
**Deep3DSketch+: Rapid 3D Modeling from Single Free-hand Sketches**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2309.13006v1)  

---


**ABSTRACT**  
The rapid development of AR/VR brings tremendous demands for 3D content. While the widely-used Computer-Aided Design (CAD) method requires a time-consuming and labor-intensive modeling process, sketch-based 3D modeling offers a potential solution as a natural form of computer-human interaction. However, the sparsity and ambiguity of sketches make it challenging to generate high-fidelity content reflecting creators' ideas. Precise drawing from multiple views or strategic step-by-step drawings is often required to tackle the challenge but is not friendly to novice users. In this work, we introduce a novel end-to-end approach, Deep3DSketch+, which performs 3D modeling using only a single free-hand sketch without inputting multiple sketches or view information. Specifically, we introduce a lightweight generation network for efficient inference in real-time and a structural-aware adversarial training approach with a Stroke Enhancement Module (SEM) to capture the structural information to facilitate learning of the realistic and fine-detailed shape structures for high-fidelity performance. Extensive experiments demonstrated the effectiveness of our approach with the state-of-the-art (SOTA) performance on both synthetic and real datasets.

{{</citation>}}


### (7/120) License Plate Recognition Based On Multi-Angle View Model (Dat Tran-Anh et al., 2023)

{{<citation>}}

Dat Tran-Anh, Khanh Linh Tran, Hoai-Nam Vu. (2023)  
**License Plate Recognition Based On Multi-Angle View Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2309.12972v1)  

---


**ABSTRACT**  
In the realm of research, the detection/recognition of text within images/videos captured by cameras constitutes a highly challenging problem for researchers. Despite certain advancements achieving high accuracy, current methods still require substantial improvements to be applicable in practical scenarios. Diverging from text detection in images/videos, this paper addresses the issue of text detection within license plates by amalgamating multiple frames of distinct perspectives. For each viewpoint, the proposed method extracts descriptive features characterizing the text components of the license plate, specifically corner points and area. Concretely, we present three viewpoints: view-1, view-2, and view-3, to identify the nearest neighboring components facilitating the restoration of text components from the same license plate line based on estimations of similarity levels and distance metrics. Subsequently, we employ the CnOCR method for text recognition within license plates. Experimental results on the self-collected dataset (PTITPlates), comprising pairs of images in various scenarios, and the publicly available Stanford Cars Dataset, demonstrate the superiority of the proposed method over existing approaches.

{{</citation>}}


### (8/120) Background Activation Suppression for Weakly Supervised Object Localization and Semantic Segmentation (Wei Zhai et al., 2023)

{{<citation>}}

Wei Zhai, Pingyu Wu, Kai Zhu, Yang Cao, Feng Wu, Zheng-Jun Zha. (2023)  
**Background Activation Suppression for Weakly Supervised Object Localization and Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.12943v1)  

---


**ABSTRACT**  
Weakly supervised object localization and semantic segmentation aim to localize objects using only image-level labels. Recently, a new paradigm has emerged by generating a foreground prediction map (FPM) to achieve pixel-level localization. While existing FPM-based methods use cross-entropy to evaluate the foreground prediction map and to guide the learning of the generator, this paper presents two astonishing experimental observations on the object localization learning process: For a trained network, as the foreground mask expands, 1) the cross-entropy converges to zero when the foreground mask covers only part of the object region. 2) The activation value continuously increases until the foreground mask expands to the object boundary. Therefore, to achieve a more effective localization performance, we argue for the usage of activation value to learn more object regions. In this paper, we propose a Background Activation Suppression (BAS) method. Specifically, an Activation Map Constraint (AMC) module is designed to facilitate the learning of generator by suppressing the background activation value. Meanwhile, by using foreground region guidance and area constraint, BAS can learn the whole region of the object. In the inference phase, we consider the prediction maps of different categories together to obtain the final localization results. Extensive experiments show that BAS achieves significant and consistent improvement over the baseline methods on the CUB-200-2011 and ILSVRC datasets. In addition, our method also achieves state-of-the-art weakly supervised semantic segmentation performance on the PASCAL VOC 2012 and MS COCO 2014 datasets. Code and models are available at https://github.com/wpy1999/BAS-Extension.

{{</citation>}}


### (9/120) Zero-Shot Object Counting with Language-Vision Models (Jingyi Xu et al., 2023)

{{<citation>}}

Jingyi Xu, Hieu Le, Dimitris Samaras. (2023)  
**Zero-Shot Object Counting with Language-Vision Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.13097v1)  

---


**ABSTRACT**  
Class-agnostic object counting aims to count object instances of an arbitrary class at test time. It is challenging but also enables many potential applications. Current methods require human-annotated exemplars as inputs which are often unavailable for novel categories, especially for autonomous systems. Thus, we propose zero-shot object counting (ZSC), a new setting where only the class name is available during test time. This obviates the need for human annotators and enables automated operation. To perform ZSC, we propose finding a few object crops from the input image and use them as counting exemplars. The goal is to identify patches containing the objects of interest while also being visually representative for all instances in the image. To do this, we first construct class prototypes using large language-vision models, including CLIP and Stable Diffusion, to select the patches containing the target objects. Furthermore, we propose a ranking model that estimates the counting error of each patch to select the most suitable exemplars for counting. Experimental results on a recent class-agnostic counting dataset, FSC-147, validate the effectiveness of our method.

{{</citation>}}


### (10/120) Bridging Sensor Gaps via Single-Direction Tuning for Hyperspectral Image Classification (Xizhe Xue et al., 2023)

{{<citation>}}

Xizhe Xue, Haokui Zhang, Ying Li, Liuwei Wan, Zongwen Bai, Mike Zheng Shou. (2023)  
**Bridging Sensor Gaps via Single-Direction Tuning for Hyperspectral Image Classification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Classification  
[Paper Link](http://arxiv.org/abs/2309.12865v1)  

---


**ABSTRACT**  
Recently, some researchers started exploring the use of ViTs in tackling HSI classification and achieved remarkable results. However, the training of ViT models requires a considerable number of training samples, while hyperspectral data, due to its high annotation costs, typically has a relatively small number of training samples. This contradiction has not been effectively addressed. In this paper, aiming to solve this problem, we propose the single-direction tuning (SDT) strategy, which serves as a bridge, allowing us to leverage existing labeled HSI datasets even RGB datasets to enhance the performance on new HSI datasets with limited samples. The proposed SDT inherits the idea of prompt tuning, aiming to reuse pre-trained models with minimal modifications for adaptation to new tasks. But unlike prompt tuning, SDT is custom-designed to accommodate the characteristics of HSIs. The proposed SDT utilizes a parallel architecture, an asynchronous cold-hot gradient update strategy, and unidirectional interaction. It aims to fully harness the potent representation learning capabilities derived from training on heterologous, even cross-modal datasets. In addition, we also introduce a novel Triplet-structured transformer (Tri-Former), where spectral attention and spatial attention modules are merged in parallel to construct the token mixing component for reducing computation cost and a 3D convolution-based channel mixer module is integrated to enhance stability and keep structure information. Comparison experiments conducted on three representative HSI datasets captured by different sensors demonstrate the proposed Tri-Former achieves better performance compared to several state-of-the-art methods. Homologous, heterologous and cross-modal tuning experiments verified the effectiveness of the proposed SDT.

{{</citation>}}


### (11/120) SRFNet: Monocular Depth Estimation with Fine-grained Structure via Spatial Reliability-oriented Fusion of Frames and Events (Tianbo Pan et al., 2023)

{{<citation>}}

Tianbo Pan, Zidong Cao, Lin Wang. (2023)  
**SRFNet: Monocular Depth Estimation with Fine-grained Structure via Spatial Reliability-oriented Fusion of Frames and Events**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.12842v1)  

---


**ABSTRACT**  
Monocular depth estimation is a crucial task to measure distance relative to a camera, which is important for applications, such as robot navigation and self-driving. Traditional frame-based methods suffer from performance drops due to the limited dynamic range and motion blur. Therefore, recent works leverage novel event cameras to complement or guide the frame modality via frame-event feature fusion. However, event streams exhibit spatial sparsity, leaving some areas unperceived, especially in regions with marginal light changes. Therefore, direct fusion methods, e.g., RAMNet, often ignore the contribution of the most confident regions of each modality. This leads to structural ambiguity in the modality fusion process, thus degrading the depth estimation performance. In this paper, we propose a novel Spatial Reliability-oriented Fusion Network (SRFNet), that can estimate depth with fine-grained structure at both daytime and nighttime. Our method consists of two key technical components. Firstly, we propose an attention-based interactive fusion (AIF) module that applies spatial priors of events and frames as the initial masks and learns the consensus regions to guide the inter-modal feature fusion. The fused feature are then fed back to enhance the frame and event feature learning. Meanwhile, it utilizes an output head to generate a fused mask, which is iteratively updated for learning consensual spatial priors. Secondly, we propose the Reliability-oriented Depth Refinement (RDR) module to estimate dense depth with the fine-grained structure based on the fused features and masks. We evaluate the effectiveness of our method on the synthetic and real-world datasets, which shows that, even without pretraining, our method outperforms the prior methods, e.g., RAMNet, especially in night scenes. Our project homepage: https://vlislab22.github.io/SRFNet.

{{</citation>}}


### (12/120) Domain Adaptive Few-Shot Open-Set Learning (Debabrata Pal et al., 2023)

{{<citation>}}

Debabrata Pal, Deeptej More, Sai Bhargav, Dipesh Tamboli, Vaneet Aggarwal, Biplab Banerjee. (2023)  
**Domain Adaptive Few-Shot Open-Set Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, ImageNet  
[Paper Link](http://arxiv.org/abs/2309.12814v1)  

---


**ABSTRACT**  
Few-shot learning has made impressive strides in addressing the crucial challenges of recognizing unknown samples from novel classes in target query sets and managing visual shifts between domains. However, existing techniques fall short when it comes to identifying target outliers under domain shifts by learning to reject pseudo-outliers from the source domain, resulting in an incomplete solution to both problems. To address these challenges comprehensively, we propose a novel approach called Domain Adaptive Few-Shot Open Set Recognition (DA-FSOS) and introduce a meta-learning-based architecture named DAFOSNET. During training, our model learns a shared and discriminative embedding space while creating a pseudo open-space decision boundary, given a fully-supervised source domain and a label-disjoint few-shot target domain. To enhance data density, we use a pair of conditional adversarial networks with tunable noise variances to augment both domains closed and pseudo-open spaces. Furthermore, we propose a domain-specific batch-normalized class prototypes alignment strategy to align both domains globally while ensuring class-discriminativeness through novel metric objectives. Our training approach ensures that DAFOS-NET can generalize well to new scenarios in the target domain. We present three benchmarks for DA-FSOS based on the Office-Home, mini-ImageNet/CUB, and DomainNet datasets and demonstrate the efficacy of DAFOS-NET through extensive experimentation

{{</citation>}}


### (13/120) WiCV@CVPR2023: The Eleventh Women In Computer Vision Workshop at the Annual CVPR Conference (Doris Antensteiner et al., 2023)

{{<citation>}}

Doris Antensteiner, Marah Halawa, Asra Aslam, Ivaxi Sheth, Sachini Herath, Ziqi Huang, Sunnie S. Y. Kim, Aparna Akula, Xin Wang. (2023)  
**WiCV@CVPR2023: The Eleventh Women In Computer Vision Workshop at the Annual CVPR Conference**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2309.12768v1)  

---


**ABSTRACT**  
In this paper, we present the details of Women in Computer Vision Workshop - WiCV 2023, organized alongside the hybrid CVPR 2023 in Vancouver, Canada. WiCV aims to amplify the voices of underrepresented women in the computer vision community, fostering increased visibility in both academia and industry. We believe that such events play a vital role in addressing gender imbalances within the field. The annual WiCV@CVPR workshop offers a) opportunity for collaboration between researchers from minority groups, b) mentorship for female junior researchers, c) financial support to presenters to alleviate finanacial burdens and d) a diverse array of role models who can inspire younger researchers at the outset of their careers. In this paper, we present a comprehensive report on the workshop program, historical trends from the past WiCV@CVPR events, and a summary of statistics related to presenters, attendees, and sponsorship for the WiCV 2023 workshop.

{{</citation>}}


### (14/120) Masking Improves Contrastive Self-Supervised Learning for ConvNets, and Saliency Tells You Where (Zhi-Yi Chin et al., 2023)

{{<citation>}}

Zhi-Yi Chin, Chieh-Ming Jiang, Ching-Chun Huang, Pin-Yu Chen, Wei-Chen Chiu. (2023)  
**Masking Improves Contrastive Self-Supervised Learning for ConvNets, and Saliency Tells You Where**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.12757v1)  

---


**ABSTRACT**  
While image data starts to enjoy the simple-but-effective self-supervised learning scheme built upon masking and self-reconstruction objective thanks to the introduction of tokenization procedure and vision transformer backbone, convolutional neural networks as another important and widely-adopted architecture for image data, though having contrastive-learning techniques to drive the self-supervised learning, still face the difficulty of leveraging such straightforward and general masking operation to benefit their learning process significantly. In this work, we aim to alleviate the burden of including masking operation into the contrastive-learning framework for convolutional neural networks as an extra augmentation method. In addition to the additive but unwanted edges (between masked and unmasked regions) as well as other adverse effects caused by the masking operations for ConvNets, which have been discussed by prior works, we particularly identify the potential problem where for one view in a contrastive sample-pair the randomly-sampled masking regions could be overly concentrated on important/salient objects thus resulting in misleading contrastiveness to the other view. To this end, we propose to explicitly take the saliency constraint into consideration in which the masked regions are more evenly distributed among the foreground and background for realizing the masking-based augmentation. Moreover, we introduce hard negative samples by masking larger regions of salient patches in an input image. Extensive experiments conducted on various datasets, contrastive learning mechanisms, and downstream tasks well verify the efficacy as well as the superior performance of our proposed method with respect to several state-of-the-art baselines.

{{</citation>}}


### (15/120) Transformer-based Image Compression with Variable Image Quality Objectives (Chia-Hao Kao et al., 2023)

{{<citation>}}

Chia-Hao Kao, Yi-Hsin Chen, Cheng Chien, Wei-Chen Chiu, Wen-Hsiao Peng. (2023)  
**Transformer-based Image Compression with Variable Image Quality Objectives**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.12717v1)  

---


**ABSTRACT**  
This paper presents a Transformer-based image compression system that allows for a variable image quality objective according to the user's preference. Optimizing a learned codec for different quality objectives leads to reconstructed images with varying visual characteristics. Our method provides the user with the flexibility to choose a trade-off between two image quality objectives using a single, shared model. Motivated by the success of prompt-tuning techniques, we introduce prompt tokens to condition our Transformer-based autoencoder. These prompt tokens are generated adaptively based on the user's preference and input image through learning a prompt generation network. Extensive experiments on commonly used quality metrics demonstrate the effectiveness of our method in adapting the encoding and/or decoding processes to a variable quality objective. While offering the additional flexibility, our proposed method performs comparably to the single-objective methods in terms of rate-distortion performance.

{{</citation>}}


### (16/120) PointSSC: A Cooperative Vehicle-Infrastructure Point Cloud Benchmark for Semantic Scene Completion (Yuxiang Yan et al., 2023)

{{<citation>}}

Yuxiang Yan, Boda Liu, Jianfei Ai, Qinbu Li, Ru Wan, Jian Pu. (2023)  
**PointSSC: A Cooperative Vehicle-Infrastructure Point Cloud Benchmark for Semantic Scene Completion**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.12708v1)  

---


**ABSTRACT**  
Semantic Scene Completion (SSC) aims to jointly generate space occupancies and semantic labels for complex 3D scenes. Most existing SSC models focus on volumetric representations, which are memory-inefficient for large outdoor spaces. Point clouds provide a lightweight alternative but existing benchmarks lack outdoor point cloud scenes with semantic labels. To address this, we introduce PointSSC, the first cooperative vehicle-infrastructure point cloud benchmark for semantic scene completion. These scenes exhibit long-range perception and minimal occlusion. We develop an automated annotation pipeline leveraging Segment Anything to efficiently assign semantics. To benchmark progress, we propose a LiDAR-based model with a Spatial-Aware Transformer for global and local feature extraction and a Completion and Segmentation Cooperative Module for joint completion and segmentation. PointSSC provides a challenging testbed to drive advances in semantic point cloud completion for real-world navigation.

{{</citation>}}


### (17/120) Exploiting Modality-Specific Features For Multi-Modal Manipulation Detection And Grounding (Jiazhen Wang et al., 2023)

{{<citation>}}

Jiazhen Wang, Bin Liu, Changtao Miao, Zhiwei Zhao, Wanyi Zhuang, Qi Chu, Nenghai Yu. (2023)  
**Exploiting Modality-Specific Features For Multi-Modal Manipulation Detection And Grounding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.12657v1)  

---


**ABSTRACT**  
AI-synthesized text and images have gained significant attention, particularly due to the widespread dissemination of multi-modal manipulations on the internet, which has resulted in numerous negative impacts on society. Existing methods for multi-modal manipulation detection and grounding primarily focus on fusing vision-language features to make predictions, while overlooking the importance of modality-specific features, leading to sub-optimal results. In this paper, we construct a simple and novel transformer-based framework for multi-modal manipulation detection and grounding tasks. Our framework simultaneously explores modality-specific features while preserving the capability for multi-modal alignment. To achieve this, we introduce visual/language pre-trained encoders and dual-branch cross-attention (DCA) to extract and fuse modality-unique features. Furthermore, we design decoupled fine-grained classifiers (DFC) to enhance modality-specific feature mining and mitigate modality competition. Moreover, we propose an implicit manipulation query (IMQ) that adaptively aggregates global contextual cues within each modality using learnable queries, thereby improving the discovery of forged details. Extensive experiments on the $\rm DGM^4$ dataset demonstrate the superior performance of our proposed model compared to state-of-the-art approaches.

{{</citation>}}


### (18/120) RHINO: Regularizing the Hash-based Implicit Neural Representation (Hao Zhu et al., 2023)

{{<citation>}}

Hao Zhu, Fengyi Liu, Qi Zhang, Xun Cao, Zhan Ma. (2023)  
**RHINO: Regularizing the Hash-based Implicit Neural Representation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2309.12642v1)  

---


**ABSTRACT**  
The use of Implicit Neural Representation (INR) through a hash-table has demonstrated impressive effectiveness and efficiency in characterizing intricate signals. However, current state-of-the-art methods exhibit insufficient regularization, often yielding unreliable and noisy results during interpolations. We find that this issue stems from broken gradient flow between input coordinates and indexed hash-keys, where the chain rule attempts to model discrete hash-keys, rather than the continuous coordinates. To tackle this concern, we introduce RHINO, in which a continuous analytical function is incorporated to facilitate regularization by connecting the input coordinate and the network additionally without modifying the architecture of current hash-based INRs. This connection ensures a seamless backpropagation of gradients from the network's output back to the input coordinates, thereby enhancing regularization. Our experimental results not only showcase the broadened regularization capability across different hash-based INRs like DINER and Instant NGP, but also across a variety of tasks such as image fitting, representation of signed distance functions, and optimization of 5D static / 6D dynamic neural radiance fields. Notably, RHINO outperforms current state-of-the-art techniques in both quality and speed, affirming its superiority.

{{</citation>}}


### (19/120) Global Context Aggregation Network for Lightweight Saliency Detection of Surface Defects (Feng Yan et al., 2023)

{{<citation>}}

Feng Yan, Xiaoheng Jiang, Yang Lu, Lisha Cui, Shupan Li, Jiale Cao, Mingliang Xu, Dacheng Tao. (2023)  
**Global Context Aggregation Network for Lightweight Saliency Detection of Surface Defects**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Self-Attention  
[Paper Link](http://arxiv.org/abs/2309.12641v1)  

---


**ABSTRACT**  
Surface defect inspection is a very challenging task in which surface defects usually show weak appearances or exist under complex backgrounds. Most high-accuracy defect detection methods require expensive computation and storage overhead, making them less practical in some resource-constrained defect detection applications. Although some lightweight methods have achieved real-time inference speed with fewer parameters, they show poor detection accuracy in complex defect scenarios. To this end, we develop a Global Context Aggregation Network (GCANet) for lightweight saliency detection of surface defects on the encoder-decoder structure. First, we introduce a novel transformer encoder on the top layer of the lightweight backbone, which captures global context information through a novel Depth-wise Self-Attention (DSA) module. The proposed DSA performs element-wise similarity in channel dimension while maintaining linear complexity. In addition, we introduce a novel Channel Reference Attention (CRA) module before each decoder block to strengthen the representation of multi-level features in the bottom-up path. The proposed CRA exploits the channel correlation between features at different layers to adaptively enhance feature representation. The experimental results on three public defect datasets demonstrate that the proposed network achieves a better trade-off between accuracy and running efficiency compared with other 17 state-of-the-art methods. Specifically, GCANet achieves competitive accuracy (91.79% $F_{\beta}^{w}$, 93.55% $S_\alpha$, and 97.35% $E_\phi$) on SD-saliency-900 while running 272fps on a single gpu.

{{</citation>}}


### (20/120) CINFormer: Transformer network with multi-stage CNN feature injection for surface defect segmentation (Xiaoheng Jiang et al., 2023)

{{<citation>}}

Xiaoheng Jiang, Kaiyi Guo, Yang Lu, Feng Yan, Hao Liu, Jiale Cao, Mingliang Xu, Dacheng Tao. (2023)  
**CINFormer: Transformer network with multi-stage CNN feature injection for surface defect segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.12639v1)  

---


**ABSTRACT**  
Surface defect inspection is of great importance for industrial manufacture and production. Though defect inspection methods based on deep learning have made significant progress, there are still some challenges for these methods, such as indistinguishable weak defects and defect-like interference in the background. To address these issues, we propose a transformer network with multi-stage CNN (Convolutional Neural Network) feature injection for surface defect segmentation, which is a UNet-like structure named CINFormer. CINFormer presents a simple yet effective feature integration mechanism that injects the multi-level CNN features of the input image into different stages of the transformer network in the encoder. This can maintain the merit of CNN capturing detailed features and that of transformer depressing noises in the background, which facilitates accurate defect detection. In addition, CINFormer presents a Top-K self-attention module to focus on tokens with more important information about the defects, so as to further reduce the impact of the redundant background. Extensive experiments conducted on the surface defect datasets DAGM 2007, Magnetic tile, and NEU show that the proposed CINFormer achieves state-of-the-art performance in defect detection.

{{</citation>}}


### (21/120) DeFormer: Integrating Transformers with Deformable Models for 3D Shape Abstraction from a Single Image (Di Liu et al., 2023)

{{<citation>}}

Di Liu, Xiang Yu, Meng Ye, Qilong Zhangli, Zhuowei Li, Zhixing Zhang, Dimitris N. Metaxas. (2023)  
**DeFormer: Integrating Transformers with Deformable Models for 3D Shape Abstraction from a Single Image**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.12594v1)  

---


**ABSTRACT**  
Accurate 3D shape abstraction from a single 2D image is a long-standing problem in computer vision and graphics. By leveraging a set of primitives to represent the target shape, recent methods have achieved promising results. However, these methods either use a relatively large number of primitives or lack geometric flexibility due to the limited expressibility of the primitives. In this paper, we propose a novel bi-channel Transformer architecture, integrated with parameterized deformable models, termed DeFormer, to simultaneously estimate the global and local deformations of primitives. In this way, DeFormer can abstract complex object shapes while using a small number of primitives which offer a broader geometry coverage and finer details. Then, we introduce a force-driven dynamic fitting and a cycle-consistent re-projection loss to optimize the primitive parameters. Extensive experiments on ShapeNet across various settings show that DeFormer achieves better reconstruction accuracy over the state-of-the-art, and visualizes with consistent semantic correspondences for improved interpretability.

{{</citation>}}


### (22/120) BGF-YOLO: Enhanced YOLOv8 with Multiscale Attentional Feature Fusion for Brain Tumor Detection (Ming Kang et al., 2023)

{{<citation>}}

Ming Kang, Chee-Ming Ting, Fung Fung Ting, Raphaël C. -W. Phan. (2023)  
**BGF-YOLO: Enhanced YOLOv8 with Multiscale Attentional Feature Fusion for Brain Tumor Detection**  

---
Primary Category: cs.CV  
Categories: 68U10 (Primary) 68T10, 68T07, 62P10 (Secondary), I-4-6; I-5-1; J-3, cs-CV, cs.CV, eess-SP, stat-AP  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.12585v2)  

---


**ABSTRACT**  
You Only Look Once (YOLO)-based object detectors have shown remarkable accuracy for automated brain tumor detection. In this paper, we develop a novel BGF-YOLO architecture by incorporating Bi-level Routing Attention (BRA), Generalized feature pyramid networks (GFPN), and Fourth detecting head into YOLOv8. BGF-YOLO contains an attention mechanism to focus more on important features, and feature pyramid networks to enrich feature representation by merging high-level semantic features with spatial details. Furthermore, we investigate the effect of different attention mechanisms and feature fusions, detection head architectures on brain tumor detection accuracy. Experimental results show that BGF-YOLO gives a 4.7% absolute increase of mAP$_{50}$ compared to YOLOv8x, and achieves state-of-the-art on the brain tumor detection dataset Br35H. The code is available at https://github.com/mkang315/BGF-YOLO.

{{</citation>}}


### (23/120) Classification of Alzheimers Disease with Deep Learning on Eye-tracking Data (Harshinee Sriram et al., 2023)

{{<citation>}}

Harshinee Sriram, Cristina Conati, Thalia Field. (2023)  
**Classification of Alzheimers Disease with Deep Learning on Eye-tracking Data**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.12574v1)  

---


**ABSTRACT**  
Existing research has shown the potential of classifying Alzheimers Disease (AD) from eye-tracking (ET) data with classifiers that rely on task-specific engineered features. In this paper, we investigate whether we can improve on existing results by using a Deep-Learning classifier trained end-to-end on raw ET data. This classifier (VTNet) uses a GRU and a CNN in parallel to leverage both visual (V) and temporal (T) representations of ET data and was previously used to detect user confusion while processing visual displays. A main challenge in applying VTNet to our target AD classification task is that the available ET data sequences are much longer than those used in the previous confusion detection task, pushing the limits of what is manageable by LSTM-based models. We discuss how we address this challenge and show that VTNet outperforms the state-of-the-art approaches in AD classification, providing encouraging evidence on the generality of this model to make predictions from ET data.

{{</citation>}}


### (24/120) Triple-View Knowledge Distillation for Semi-Supervised Semantic Segmentation (Ping Li et al., 2023)

{{<citation>}}

Ping Li, Junjie Chen, Li Yuan, Xianghua Xu, Mingli Song. (2023)  
**Triple-View Knowledge Distillation for Semi-Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation, Semantic Segmentation, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.12557v1)  

---


**ABSTRACT**  
To alleviate the expensive human labeling, semi-supervised semantic segmentation employs a few labeled images and an abundant of unlabeled images to predict the pixel-level label map with the same size. Previous methods often adopt co-training using two convolutional networks with the same architecture but different initialization, which fails to capture the sufficiently diverse features. This motivates us to use tri-training and develop the triple-view encoder to utilize the encoders with different architectures to derive diverse features, and exploit the knowledge distillation skill to learn the complementary semantics among these encoders. Moreover, existing methods simply concatenate the features from both encoder and decoder, resulting in redundant features that require large memory cost. This inspires us to devise a dual-frequency decoder that selects those important features by projecting the features from the spatial domain to the frequency domain, where the dual-frequency channel attention mechanism is introduced to model the feature importance. Therefore, we propose a Triple-view Knowledge Distillation framework, termed TriKD, for semi-supervised semantic segmentation, including the triple-view encoder and the dual-frequency decoder. Extensive experiments were conducted on two benchmarks, \ie, Pascal VOC 2012 and Cityscapes, whose results verify the superiority of the proposed method with a good tradeoff between precision and inference speed.

{{</citation>}}


## cs.AI (12)



### (25/120) AI-Copilot for Business Optimisation: A Framework and A Case Study in Production Scheduling (Pivithuru Thejan Amarasinghe et al., 2023)

{{<citation>}}

Pivithuru Thejan Amarasinghe, Su Nguyen, Yuan Sun, Damminda Alahakoon. (2023)  
**AI-Copilot for Business Optimisation: A Framework and A Case Study in Production Scheduling**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.13218v1)  

---


**ABSTRACT**  
Business optimisation is the process of finding and implementing efficient and cost-effective means of operation to bring a competitive advantage for businesses. Synthesizing problem formulations is an integral part of business optimisation which is centred around human expertise, thus with a high potential of becoming a bottleneck. With the recent advancements in Large Language Models (LLMs), human expertise needed in problem formulation can potentially be minimized using Artificial Intelligence (AI). However, developing a LLM for problem formulation is challenging, due to training data requirements, token limitations, and the lack of appropriate performance metrics in LLMs. To minimize the requirement of large training data, considerable attention has recently been directed towards fine-tuning pre-trained LLMs for downstream tasks, rather than training a LLM from scratch for a specific task. In this paper, we adopt this approach and propose an AI-Copilot for business optimisation by fine-tuning a pre-trained LLM for problem formulation. To address token limitations, we introduce modularization and prompt engineering techniques to synthesize complex problem formulations as modules that fit into the token limits of LLMs. In addition, we design performance evaluation metrics that are more suitable for assessing the accuracy and quality of problem formulations compared to existing evaluation metrics. Experiment results demonstrate that our AI-Copilot can synthesize complex and large problem formulations for a typical business optimisation problem in production scheduling.

{{</citation>}}


### (26/120) AI Risk Profiles: A Standards Proposal for Pre-Deployment AI Risk Disclosures (Eli Sherman et al., 2023)

{{<citation>}}

Eli Sherman, Ian W. Eisenberg. (2023)  
**AI Risk Profiles: A Standards Proposal for Pre-Deployment AI Risk Disclosures**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13176v1)  

---


**ABSTRACT**  
As AI systems' sophistication and proliferation have increased, awareness of the risks has grown proportionally (Sorkin et al. 2023). In response, calls have grown for stronger emphasis on disclosure and transparency in the AI industry (NTIA 2023; OpenAI 2023b), with proposals ranging from standardizing use of technical disclosures, like model cards (Mitchell et al. 2019), to yet-unspecified licensing regimes (Sindhu 2023). Since the AI value chain is complicated, with actors representing various expertise, perspectives, and values, it is crucial that consumers of a transparency disclosure be able to understand the risks of the AI system the disclosure concerns. In this paper we propose a risk profiling standard which can guide downstream decision-making, including triaging further risk assessment, informing procurement and deployment, and directing regulatory frameworks. The standard is built on our proposed taxonomy of AI risks, which reflects a high-level categorization of the wide variety of risks proposed in the literature. We outline the myriad data sources needed to construct informative Risk Profiles and propose a template-based methodology for collating risk information into a standard, yet flexible, structure. We apply this methodology to a number of prominent AI systems using publicly available information. To conclude, we discuss design decisions for the profiles and future work.

{{</citation>}}


### (27/120) KG-MDL: Mining Graph Patterns in Knowledge Graphs with the MDL Principle (Francesco Bariatti et al., 2023)

{{<citation>}}

Francesco Bariatti, Peggy Cellier, Sébastien Ferré. (2023)  
**KG-MDL: Mining Graph Patterns in Knowledge Graphs with the MDL Principle**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-IT, cs.AI, math-IT  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.12908v1)  

---


**ABSTRACT**  
Nowadays, increasingly more data are available as knowledge graphs (KGs). While this data model supports advanced reasoning and querying, they remain difficult to mine due to their size and complexity. Graph mining approaches can be used to extract patterns from KGs. However this presents two main issues. First, graph mining approaches tend to extract too many patterns for a human analyst to interpret (pattern explosion). Second, real-life KGs tend to differ from the graphs usually treated in graph mining: they are multigraphs, their vertex degrees tend to follow a power-law, and the way in which they model knowledge can produce spurious patterns. Recently, a graph mining approach named GraphMDL+ has been proposed to tackle the problem of pattern explosion, using the Minimum Description Length (MDL) principle. However, GraphMDL+, like other graph mining approaches, is not suited for KGs without adaptations. In this paper we propose KG-MDL, a graph pattern mining approach based on the MDL principle that, given a KG, generates a human-sized and descriptive set of graph patterns, and so in a parameter-less and anytime way. We report on experiments on medium-sized KGs showing that our approach generates sets of patterns that are both small enough to be interpreted by humans and descriptive of the KG. We show that the extracted patterns highlight relevant characteristics of the data: both of the schema used to create the data, and of the concrete facts it contains. We also discuss the issues related to mining graph patterns on knowledge graphs, as opposed to other types of graph data.

{{</citation>}}


### (28/120) OpenAi's GPT4 as coding assistant (Lefteris Moussiades et al., 2023)

{{<citation>}}

Lefteris Moussiades, George Zografos. (2023)  
**OpenAi's GPT4 as coding assistant**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SE, cs.AI  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.12732v1)  

---


**ABSTRACT**  
Lately, Large Language Models have been widely used in code generation. GPT4 is considered the most potent Large Language Model from Openai. In this paper, we examine GPT3.5 and GPT4 as coding assistants. More specifically, we have constructed appropriate tests to check whether the two systems can a) answer typical questions that can arise during the code development, b) produce reliable code, and c) contribute to code debugging. The test results are impressive. The performance of GPT4 is outstanding and signals an increase in the productivity of programmers and the reorganization of software development procedures based on these new tools.

{{</citation>}}


### (29/120) Defeasible Reasoning with Knowledge Graphs (Dave Raggett, 2023)

{{<citation>}}

Dave Raggett. (2023)  
**Defeasible Reasoning with Knowledge Graphs**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Knowledge Graph, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.12731v1)  

---


**ABSTRACT**  
Human knowledge is subject to uncertainties, imprecision, incompleteness and inconsistencies. Moreover, the meaning of many everyday terms is dependent on the context. That poses a huge challenge for the Semantic Web. This paper introduces work on an intuitive notation and model for defeasible reasoning with imperfect knowledge, and relates it to previous work on argumentation theory. PKN is to N3 as defeasible reasoning is to deductive logic. Further work is needed on an intuitive syntax for describing reasoning strategies and tactics in declarative terms, drawing upon the AIF ontology for inspiration. The paper closes with observations on symbolic approaches in the era of large language models.

{{</citation>}}


### (30/120) In-context Interference in Chat-based Large Language Models (Eric Nuertey Coleman et al., 2023)

{{<citation>}}

Eric Nuertey Coleman, Julio Hurtado, Vincenzo Lomonaco. (2023)  
**In-context Interference in Chat-based Large Language Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.12727v1)  

---


**ABSTRACT**  
Large language models (LLMs) have had a huge impact on society due to their impressive capabilities and vast knowledge of the world. Various applications and tools have been created that allow users to interact with these models in a black-box scenario. However, one limitation of this scenario is that users cannot modify the internal knowledge of the model, and the only way to add or modify internal knowledge is by explicitly mentioning it to the model during the current interaction. This learning process is called in-context training, and it refers to training that is confined to the user's current session or context. In-context learning has significant applications, but also has limitations that are seldom studied. In this paper, we present a study that shows how the model can suffer from interference between information that continually flows in the context, causing it to forget previously learned knowledge, which can reduce the model's performance. Along with showing the problem, we propose an evaluation benchmark based on the bAbI dataset.

{{</citation>}}


### (31/120) Counterfactual Conservative Q Learning for Offline Multi-agent Reinforcement Learning (Jianzhun Shao et al., 2023)

{{<citation>}}

Jianzhun Shao, Yun Qu, Chen Chen, Hongchang Zhang, Xiangyang Ji. (2023)  
**Counterfactual Conservative Q Learning for Offline Multi-agent Reinforcement Learning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.12696v1)  

---


**ABSTRACT**  
Offline multi-agent reinforcement learning is challenging due to the coupling effect of both distribution shift issue common in offline setting and the high dimension issue common in multi-agent setting, making the action out-of-distribution (OOD) and value overestimation phenomenon excessively severe. Tomitigate this problem, we propose a novel multi-agent offline RL algorithm, named CounterFactual Conservative Q-Learning (CFCQL) to conduct conservative value estimation. Rather than regarding all the agents as a high dimensional single one and directly applying single agent methods to it, CFCQL calculates conservative regularization for each agent separately in a counterfactual way and then linearly combines them to realize an overall conservative value estimation. We prove that it still enjoys the underestimation property and the performance guarantee as those single agent conservative methods do, but the induced regularization and safe policy improvement bound are independent of the agent number, which is therefore theoretically superior to the direct treatment referred to above, especially when the agent number is large. We further conduct experiments on four environments including both discrete and continuous action settings on both existing and our man-made datasets, demonstrating that CFCQL outperforms existing methods on most datasets and even with a remarkable margin on some of them.

{{</citation>}}


### (32/120) TrTr: A Versatile Pre-Trained Large Traffic Model based on Transformer for Capturing Trajectory Diversity in Vehicle Population (Ruyi Feng et al., 2023)

{{<citation>}}

Ruyi Feng, Zhibin Li, Bowen Liu, Yan Ding, Ou Zheng. (2023)  
**TrTr: A Versatile Pre-Trained Large Traffic Model based on Transformer for Capturing Trajectory Diversity in Vehicle Population**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, physics-data-an  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.12677v1)  

---


**ABSTRACT**  
Understanding trajectory diversity is a fundamental aspect of addressing practical traffic tasks. However, capturing the diversity of trajectories presents challenges, particularly with traditional machine learning and recurrent neural networks due to the requirement of large-scale parameters. The emerging Transformer technology, renowned for its parallel computation capabilities enabling the utilization of models with hundreds of millions of parameters, offers a promising solution. In this study, we apply the Transformer architecture to traffic tasks, aiming to learn the diversity of trajectories within vehicle populations. We analyze the Transformer's attention mechanism and its adaptability to the goals of traffic tasks, and subsequently, design specific pre-training tasks. To achieve this, we create a data structure tailored to the attention mechanism and introduce a set of noises that correspond to spatio-temporal demands, which are incorporated into the structured data during the pre-training process. The designed pre-training model demonstrates excellent performance in capturing the spatial distribution of the vehicle population, with no instances of vehicle overlap and an RMSE of 0.6059 when compared to the ground truth values. In the context of time series prediction, approximately 95% of the predicted trajectories' speeds closely align with the true speeds, within a deviation of 7.5144m/s. Furthermore, in the stability test, the model exhibits robustness by continuously predicting a time series ten times longer than the input sequence, delivering smooth trajectories and showcasing diverse driving behaviors. The pre-trained model also provides a good basis for downstream fine-tuning tasks. The number of parameters of our model is over 50 million.

{{</citation>}}


### (33/120) Vision Transformers for Computer Go (Amani Sagri et al., 2023)

{{<citation>}}

Amani Sagri, Tristan Cazenave, Jérôme Arjonilla, Abdallah Saffidine. (2023)  
**Vision Transformers for Computer Go**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs.AI  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.12675v1)  

---


**ABSTRACT**  
Motivated by the success of transformers in various fields, such as language understanding and image analysis, this investigation explores their application in the context of the game of Go. In particular, our study focuses on the analysis of the Transformer in Vision. Through a detailed analysis of numerous points such as prediction accuracy, win rates, memory, speed, size, or even learning rate, we have been able to highlight the substantial role that transformers can play in the game of Go. This study was carried out by comparing them to the usual Residual Networks.

{{</citation>}}


### (34/120) Construction contract risk identification based on knowledge-augmented language model (Saika Wong et al., 2023)

{{<citation>}}

Saika Wong, Chunmo Zheng, Xing Su, Yinqiu Tang. (2023)  
**Construction contract risk identification based on knowledge-augmented language model**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.12626v1)  

---


**ABSTRACT**  
Contract review is an essential step in construction projects to prevent potential losses. However, the current methods for reviewing construction contracts lack effectiveness and reliability, leading to time-consuming and error-prone processes. While large language models (LLMs) have shown promise in revolutionizing natural language processing (NLP) tasks, they struggle with domain-specific knowledge and addressing specialized issues. This paper presents a novel approach that leverages LLMs with construction contract knowledge to emulate the process of contract review by human experts. Our tuning-free approach incorporates construction contract domain knowledge to enhance language models for identifying construction contract risks. The use of a natural language when building the domain knowledge base facilitates practical implementation. We evaluated our method on real construction contracts and achieved solid performance. Additionally, we investigated how large language models employ logical thinking during the task and provide insights and recommendations for future research.

{{</citation>}}


### (35/120) DRG-LLaMA : Tuning LLaMA Model to Predict Diagnosis-related Group for Hospitalized Patients (Hanyin Wang et al., 2023)

{{<citation>}}

Hanyin Wang, Chufan Gao, Christopher Dantona, Bryan Hull, Jimeng Sun. (2023)  
**DRG-LLaMA : Tuning LLaMA Model to Predict Diagnosis-related Group for Hospitalized Patients**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: BERT, Clinical, LLaMA  
[Paper Link](http://arxiv.org/abs/2309.12625v1)  

---


**ABSTRACT**  
In the U.S. inpatient payment system, the Diagnosis-Related Group (DRG) plays a key role but its current assignment process is time-consuming. We introduce DRG-LLaMA, a large language model (LLM) fine-tuned on clinical notes for improved DRG prediction. Using Meta's LLaMA as the base model, we optimized it with Low-Rank Adaptation (LoRA) on 236,192 MIMIC-IV discharge summaries. With an input token length of 512, DRG-LLaMA-7B achieved a macro-averaged F1 score of 0.327, a top-1 prediction accuracy of 52.0% and a macro-averaged Area Under the Curve (AUC) of 0.986. Impressively, DRG-LLaMA-7B surpassed previously reported leading models on this task, demonstrating a relative improvement in macro-averaged F1 score of 40.3% compared to ClinicalBERT and 35.7% compared to CAML. When DRG-LLaMA is applied to predict base DRGs and complication or comorbidity (CC) / major complication or comorbidity (MCC), the top-1 prediction accuracy reached 67.8% for base DRGs and 67.5% for CC/MCC status. DRG-LLaMA performance exhibits improvements in correlation with larger model parameters and longer input context lengths. Furthermore, usage of LoRA enables training even on smaller GPUs with 48 GB of VRAM, highlighting the viability of adapting LLMs for DRGs prediction.

{{</citation>}}


### (36/120) From Text to Trends: A Unique Garden Analytics Perspective on the Future of Modern Agriculture (Parag Saxena, 2023)

{{<citation>}}

Parag Saxena. (2023)  
**From Text to Trends: A Unique Garden Analytics Perspective on the Future of Modern Agriculture**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.12579v1)  

---


**ABSTRACT**  
Data-driven insights are essential for modern agriculture. This research paper introduces a machine learning framework designed to improve how we educate and reach out to people in the field of horticulture. The framework relies on data from the Horticulture Online Help Desk (HOHD), which is like a big collection of questions from people who love gardening and are part of the Extension Master Gardener Program (EMGP). This framework has two main parts. First, it uses special computer programs (machine learning models) to sort questions into categories. This helps us quickly send each question to the right expert, so we can answer it faster. Second, it looks at when questions are asked and uses that information to guess how many questions we might get in the future and what they will be about. This helps us plan on topics that will be really important. It's like knowing what questions will be popular in the coming months. We also take into account where the questions come from by looking at the Zip Code. This helps us make research that fits the challenges faced by gardeners in different places. In this paper, we demonstrate the potential of machine learning techniques to predict trends in horticulture by analyzing textual queries from homeowners. We show that NLP, classification, and time series analysis can be used to identify patterns in homeowners' queries and predict future trends in horticulture. Our results suggest that machine learning could be used to predict trends in other agricultural sectors as well. If large-scale agriculture industries curate and maintain a comparable repository of textual data, the potential for trend prediction and strategic agricultural planning could be revolutionized. This convergence of technology and agriculture offers a promising pathway for the future of sustainable farming and data-informed agricultural practices

{{</citation>}}


## hep-ex (1)



### (37/120) The LHCb ultra-fast simulation option, Lamarr: design and validation (Lucio Anderlini et al., 2023)

{{<citation>}}

Lucio Anderlini, Matteo Barbetti, Simone Capelli, Gloria Corti, Adam Davis, Denis Derkach, Nikita Kazeev, Artem Maevskiy, Maurizio Martinelli, Sergei Mokonenko, Benedetto Gianluca Siddi, Zehua Xu. (2023)  
**The LHCb ultra-fast simulation option, Lamarr: design and validation**  

---
Primary Category: hep-ex  
Categories: cs-LG, hep-ex, hep-ex, physics-ins-det  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.13213v1)  

---


**ABSTRACT**  
Detailed detector simulation is the major consumer of CPU resources at LHCb, having used more than 90% of the total computing budget during Run 2 of the Large Hadron Collider at CERN. As data is collected by the upgraded LHCb detector during Run 3 of the LHC, larger requests for simulated data samples are necessary, and will far exceed the pledged resources of the experiment, even with existing fast simulation options. An evolution of technologies and techniques to produce simulated samples is mandatory to meet the upcoming needs of analysis to interpret signal versus background and measure efficiencies. In this context, we propose Lamarr, a Gaudi-based framework designed to offer the fastest solution for the simulation of the LHCb detector. Lamarr consists of a pipeline of modules parameterizing both the detector response and the reconstruction algorithms of the LHCb experiment. Most of the parameterizations are made of Deep Generative Models and Gradient Boosted Decision Trees trained on simulated samples or alternatively, where possible, on real data. Embedding Lamarr in the general LHCb Gauss Simulation framework allows combining its execution with any of the available generators in a seamless way. Lamarr has been validated by comparing key reconstructed quantities with Detailed Simulation. Good agreement of the simulated distributions is obtained with two-order-of-magnitude speed-up of the simulation phase.

{{</citation>}}


## cs.CL (27)



### (38/120) A Practical Survey on Zero-shot Prompt Design for In-context Learning (Yinheng Li, 2023)

{{<citation>}}

Yinheng Li. (2023)  
**A Practical Survey on Zero-shot Prompt Design for In-context Learning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-ET, cs-LG, cs.CL  
Keywords: NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.13205v1)  

---


**ABSTRACT**  
The remarkable advancements in large language models (LLMs) have brought about significant improvements in Natural Language Processing(NLP) tasks. This paper presents a comprehensive review of in-context learning techniques, focusing on different types of prompts, including discrete, continuous, few-shot, and zero-shot, and their impact on LLM performance. We explore various approaches to prompt design, such as manual design, optimization algorithms, and evaluation methods, to optimize LLM performance across diverse tasks. Our review covers key research studies in prompt engineering, discussing their methodologies and contributions to the field. We also delve into the challenges faced in evaluating prompt performance, given the absence of a single "best" prompt and the importance of considering multiple metrics. In conclusion, the paper highlights the critical role of prompt design in harnessing the full potential of LLMs and provides insights into the combination of manual design, optimization techniques, and rigorous evaluation for more effective and efficient use of LLMs in various NLP tasks.

{{</citation>}}


### (39/120) Large Language Models and Control Mechanisms Improve Text Readability of Biomedical Abstracts (Zihao Li et al., 2023)

{{<citation>}}

Zihao Li, Samuel Belkadi, Nicolo Micheletti, Lifeng Han, Matthew Shardlow, Goran Nenadic. (2023)  
**Large Language Models and Control Mechanisms Improve Text Readability of Biomedical Abstracts**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, BERT, BLEU, GPT, GPT-3.5, GPT-4, Language Model, NLP, Natural Language Processing, T5  
[Paper Link](http://arxiv.org/abs/2309.13202v1)  

---


**ABSTRACT**  
Biomedical literature often uses complex language and inaccessible professional terminologies. That is why simplification plays an important role in improving public health literacy. Applying Natural Language Processing (NLP) models to automate such tasks allows for quick and direct accessibility for lay readers. In this work, we investigate the ability of state-of-the-art large language models (LLMs) on the task of biomedical abstract simplification, using the publicly available dataset for plain language adaptation of biomedical abstracts (\textbf{PLABA}). The methods applied include domain fine-tuning and prompt-based learning (PBL) on: 1) Encoder-decoder models (T5, SciFive, and BART), 2) Decoder-only GPT models (GPT-3.5 and GPT-4) from OpenAI and BioGPT, and 3) Control-token mechanisms on BART-based models. We used a range of automatic evaluation metrics, including BLEU, ROUGE, SARI, and BERTscore, and also conducted human evaluations. BART-Large with Control Token (BART-L-w-CT) mechanisms reported the highest SARI score of 46.54 and T5-base reported the highest BERTscore 72.62. In human evaluation, BART-L-w-CTs achieved a better simplicity score over T5-Base (2.9 vs. 2.2), while T5-Base achieved a better meaning preservation score over BART-L-w-CTs (3.1 vs. 2.6). We also categorised the system outputs with examples, hoping this will shed some light for future research on this task. Our code, fine-tuned models, and data splits are available at \url{https://github.com/HECTA-UoM/PLABA-MU}

{{</citation>}}


### (40/120) Effective Distillation of Table-based Reasoning Ability from LLMs (Bohao Yang et al., 2023)

{{<citation>}}

Bohao Yang, Chen Tang, Kun Zhao, Chenghao Xiao, Chenghua Lin. (2023)  
**Effective Distillation of Table-based Reasoning Ability from LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Reasoning, T5  
[Paper Link](http://arxiv.org/abs/2309.13182v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks. However, their remarkable parameter size and their impressive high requirement of computing resources pose challenges for their practical deployment. Recent research has revealed that specific capabilities of LLMs, such as numerical reasoning, can be transferred to smaller models through distillation. Some studies explore the potential of leveraging LLMs to perform table-based reasoning. Nevertheless, prior to our work, there has been no investigation into the prospect of specialising table reasoning skills in smaller models specifically tailored for table-to-text generation tasks. In this paper, we propose a novel table-based reasoning distillation, with the aim of distilling distilling LLMs into tailored, smaller models specifically designed for table-based reasoning task. Experimental results have shown that a 0.22 billion parameter model (Flan-T5-base) fine-tuned using distilled data, not only achieves a significant improvement compared to traditionally fine-tuned baselines but also surpasses specific LLMs like gpt-3.5-turbo on the scientific table-to-text generation dataset (SciGen). The code and data are released in https://github.com/Bernard-Yang/TableDistill.

{{</citation>}}


### (41/120) BenLLMEval: A Comprehensive Evaluation into the Potentials and Pitfalls of Large Language Models on Bengali NLP (Mohsinul Kabir et al., 2023)

{{<citation>}}

Mohsinul Kabir, Mohammed Saidul Islam, Md Tahmid Rahman Laskar, Mir Tafseer Nayeem, M Saiful Bari, Enamul Hoque. (2023)  
**BenLLMEval: A Comprehensive Evaluation into the Potentials and Pitfalls of Large Language Models on Bengali NLP**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, LLaMA, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2309.13173v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have emerged as one of the most important breakthroughs in natural language processing (NLP) for their impressive skills in language generation and other language-specific tasks. Though LLMs have been evaluated in various tasks, mostly in English, they have not yet undergone thorough evaluation in under-resourced languages such as Bengali (Bangla). In this paper, we evaluate the performance of LLMs for the low-resourced Bangla language. We select various important and diverse Bangla NLP tasks, such as abstractive summarization, question answering, paraphrasing, natural language inference, text classification, and sentiment analysis for zero-shot evaluation with ChatGPT, LLaMA-2, and Claude-2 and compare the performance with state-of-the-art fine-tuned models. Our experimental results demonstrate an inferior performance of LLMs for different Bangla NLP tasks, calling for further effort to develop better understanding of LLMs in low-resource languages like Bangla.

{{</citation>}}


### (42/120) Large Language Models Are Also Good Prototypical Commonsense Reasoners (Chenin Li et al., 2023)

{{<citation>}}

Chenin Li, Qianglong Chen, Yin Zhang, Yifei Zhang, Hongxiang Yao. (2023)  
**Large Language Models Are Also Good Prototypical Commonsense Reasoners**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-3.5, Language Model, NLP, QA  
[Paper Link](http://arxiv.org/abs/2309.13165v1)  

---


**ABSTRACT**  
Commonsense reasoning is a pivotal skill for large language models, yet it presents persistent challenges in specific tasks requiring this competence. Traditional fine-tuning approaches can be resource-intensive and potentially compromise a model's generalization capacity. Furthermore, state-of-the-art language models like GPT-3.5 and Claude are primarily accessible through API calls, which makes fine-tuning models challenging. To address these challenges, we draw inspiration from the outputs of large models for tailored tasks and semi-automatically developed a set of novel prompts from several perspectives, including task-relevance, supportive evidence generation (e.g. chain-of-thought and knowledge), diverse path decoding to aid the model. Experimental results on ProtoQA dataset demonstrate that with better designed prompts we can achieve the new state-of-art(SOTA) on the ProtoQA leaderboard, improving the Max Answer@1 score by 8%, Max Incorrect@1 score by 4% (breakthrough 50% for the first time) compared to the previous SOTA model and achieved an improvement on StrategyQA and CommonsenseQA2.0 (3% and 1%, respectively). Furthermore, with the generated Chain-of-Thought and knowledge, we can improve the interpretability of the model while also surpassing the previous SOTA models. We hope that our work can provide insight for the NLP community to develop better prompts and explore the potential of large language models for more complex reasoning tasks.

{{</citation>}}


### (43/120) Cardiovascular Disease Risk Prediction via Social Media (Al Zadid Sultan Bin Habib et al., 2023)

{{<citation>}}

Al Zadid Sultan Bin Habib, Md Asif Bin Syed, Md Tanvirul Islam, Donald A. Adjeroh. (2023)  
**Cardiovascular Disease Risk Prediction via Social Media**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Natural Language Processing, Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2309.13147v1)  

---


**ABSTRACT**  
Researchers utilize Twitter and sentiment analysis to forecast the risk of Cardiovascular Disease (CVD). We have introduced a novel CVD-related keyword dictionary by scrutinizing the emotions conveyed in tweets. We gathered tweets from eighteen U.S. states, encompassing the Appalachian region. Employing the VADER model for sentiment analysis, we categorized users as potentially at risk for CVD. Machine Learning (ML) models were employed to assess individuals' CVD risk and were subsequently applied to a CDC dataset containing demographic information for comparison. We considered various performance evaluation metrics, including Test Accuracy, Precision, Recall, F1 score, Mathew's Correlation Coefficient (MCC), and Cohen's Kappa (CK) score. Our findings demonstrate that analyzing the emotional content of tweets outperforms the predictive capabilities of demographic data alone, enabling the identification of individuals at potential risk of developing CVD. This research underscores the potential of Natural Language Processing (NLP) and ML techniques in leveraging tweets to identify individuals with CVD risks, offering an alternative approach to traditional demographic information for public health monitoring.

{{</citation>}}


### (44/120) ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs (Justin Chih-Yao Chen et al., 2023)

{{<citation>}}

Justin Chih-Yao Chen, Swarnadeep Saha, Mohit Bansal. (2023)  
**ReConcile: Round-Table Conference Improves Reasoning via Consensus among Diverse LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.13007v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) still struggle with complex reasoning tasks. Motivated by the society of minds (Minsky, 1988), we propose ReConcile, a multi-model multi-agent framework designed as a round table conference among diverse LLM agents to foster diverse thoughts and discussion for improved consensus. ReConcile enhances the reasoning capabilities of LLMs by holding multiple rounds of discussion, learning to convince other agents to improve their answers, and employing a confidence-weighted voting mechanism. In each round, ReConcile initiates discussion between agents via a 'discussion prompt' that consists of (a) grouped answers and explanations generated by each agent in the previous round, (b) their uncertainties, and (c) demonstrations of answer-rectifying human explanations, used for convincing other agents. This discussion prompt enables each agent to revise their responses in light of insights from other agents. Once a consensus is reached and the discussion ends, ReConcile determines the final answer by leveraging the confidence of each agent in a weighted voting scheme. We implement ReConcile with ChatGPT, Bard, and Claude2 as the three agents. Our experimental results on various benchmarks demonstrate that ReConcile significantly enhances the reasoning performance of the agents (both individually and as a team), surpassing prior single-agent and multi-agent baselines by 7.7% and also outperforming GPT-4 on some of these datasets. We also experiment with GPT-4 itself as one of the agents in ReConcile and demonstrate that its initial performance also improves by absolute 10.0% through discussion and feedback from other agents. Finally, we also analyze the accuracy after every round and observe that ReConcile achieves better and faster consensus between agents, compared to a multi-agent debate baseline. Our code is available at: https://github.com/dinobby/ReConcile

{{</citation>}}


### (45/120) Audience-specific Explanations for Machine Translation (Renhan Lou et al., 2023)

{{<citation>}}

Renhan Lou, Jan Niehues. (2023)  
**Audience-specific Explanations for Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Machine Translation  
[Paper Link](http://arxiv.org/abs/2309.12998v1)  

---


**ABSTRACT**  
In machine translation, a common problem is that the translation of certain words even if translated can cause incomprehension of the target language audience due to different cultural backgrounds. A solution to solve this problem is to add explanations for these words. In a first step, we therefore need to identify these words or phrases. In this work we explore techniques to extract example explanations from a parallel corpus. However, the sparsity of sentences containing words that need to be explained makes building the training dataset extremely difficult. In this work, we propose a semi-automatic technique to extract these explanations from a large parallel corpus. Experiments on English->German language pair show that our method is able to extract sentence so that more than 10% of the sentences contain explanation, while only 1.9% of the original sentences contain explanations. In addition, experiments on English->French and English->Chinese language pairs also show similar conclusions. This is therefore an essential first automatic step to create a explanation dataset. Furthermore we show that the technique is robust for all three language pairs.

{{</citation>}}


### (46/120) Nested Event Extraction upon Pivot Element Recogniton (Weicheng Ren et al., 2023)

{{<citation>}}

Weicheng Ren, Zixuan Li, Xiaolong Jin, Long Bai, Miao Su, Yantao Liu, Saiping Guan, Jiafeng Guo, Xueqi Cheng. (2023)  
**Nested Event Extraction upon Pivot Element Recogniton**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Event Extraction  
[Paper Link](http://arxiv.org/abs/2309.12960v1)  

---


**ABSTRACT**  
Nested Event Extraction (NEE) aims to extract complex event structures where an event contains other events as its arguments recursively. Nested events involve a kind of Pivot Elements (PEs) that simultaneously act as arguments of outer events and as triggers of inner events, and thus connect them into nested structures. This special characteristic of PEs brings challenges to existing NEE methods, as they cannot well cope with the dual identities of PEs. Therefore, this paper proposes a new model, called PerNee, which extracts nested events mainly based on recognizing PEs. Specifically, PerNee first recognizes the triggers of both inner and outer events and further recognizes the PEs via classifying the relation type between trigger pairs. In order to obtain better representations of triggers and arguments to further improve NEE performance, it incorporates the information of both event types and argument roles into PerNee through prompt learning. Since existing NEE datasets (e.g., Genia11) are limited to specific domains and contain a narrow range of event types with nested structures, we systematically categorize nested events in generic domain and construct a new NEE dataset, namely ACE2005-Nest. Experimental results demonstrate that PerNee consistently achieves state-of-the-art performance on ACE2005-Nest, Genia11 and Genia13.

{{</citation>}}


### (47/120) Self-Explanation Prompting Improves Dialogue Understanding in Large Language Models (Haoyu Gao et al., 2023)

{{<citation>}}

Haoyu Gao, Ting-En Lin, Hangyu Li, Min Yang, Yuchuan Wu, Wentao Ma, Yongbin Li. (2023)  
**Self-Explanation Prompting Improves Dialogue Understanding in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue, Language Model  
[Paper Link](http://arxiv.org/abs/2309.12940v1)  

---


**ABSTRACT**  
Task-oriented dialogue (TOD) systems facilitate users in executing various activities via multi-turn dialogues, but Large Language Models (LLMs) often struggle to comprehend these intricate contexts. In this study, we propose a novel "Self-Explanation" prompting strategy to enhance the comprehension abilities of LLMs in multi-turn dialogues. This task-agnostic approach requires the model to analyze each dialogue utterance before task execution, thereby improving performance across various dialogue-centric tasks. Experimental results from six benchmark datasets confirm that our method consistently outperforms other zero-shot prompts and matches or exceeds the efficacy of few-shot prompts, demonstrating its potential as a powerful tool in enhancing LLMs' comprehension in complex dialogue tasks.

{{</citation>}}


### (48/120) TopRoBERTa: Topology-Aware Authorship Attribution of Deepfake Texts (Adaku Uchendu et al., 2023)

{{<citation>}}

Adaku Uchendu, Thai Le, Dongwon Lee. (2023)  
**TopRoBERTa: Topology-Aware Authorship Attribution of Deepfake Texts**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.12934v1)  

---


**ABSTRACT**  
Recent advances in Large Language Models (LLMs) have enabled the generation of open-ended high-quality texts, that are non-trivial to distinguish from human-written texts. We refer to such LLM-generated texts as \emph{deepfake texts}. There are currently over 11K text generation models in the huggingface model repo. As such, users with malicious intent can easily use these open-sourced LLMs to generate harmful texts and misinformation at scale. To mitigate this problem, a computational method to determine if a given text is a deepfake text or not is desired--i.e., Turing Test (TT). In particular, in this work, we investigate the more general version of the problem, known as \emph{Authorship Attribution (AA)}, in a multi-class setting--i.e., not only determining if a given text is a deepfake text or not but also being able to pinpoint which LLM is the author. We propose \textbf{TopRoBERTa} to improve existing AA solutions by capturing more linguistic patterns in deepfake texts by including a Topological Data Analysis (TDA) layer in the RoBERTa model. We show the benefits of having a TDA layer when dealing with noisy, imbalanced, and heterogeneous datasets, by extracting TDA features from the reshaped $pooled\_output$ of RoBERTa as input. We use RoBERTa to capture contextual representations (i.e., semantic and syntactic linguistic features), while using TDA to capture the shape and structure of data (i.e., linguistic structures). Finally, \textbf{TopRoBERTa}, outperforms the vanilla RoBERTa in 2/3 datasets, achieving up to 7\% increase in Macro F1 score.

{{</citation>}}


### (49/120) On Separate Normalization in Self-supervised Transformers (Xiaohui Chen et al., 2023)

{{<citation>}}

Xiaohui Chen, Yinkai Wang, Yuanqi Du, Soha Hassoun, Li-Ping Liu. (2023)  
**On Separate Normalization in Self-supervised Transformers**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.12931v1)  

---


**ABSTRACT**  
Self-supervised training methods for transformers have demonstrated remarkable performance across various domains. Previous transformer-based models, such as masked autoencoders (MAE), typically utilize a single normalization layer for both the [CLS] symbol and the tokens. We propose in this paper a simple modification that employs separate normalization layers for the tokens and the [CLS] symbol to better capture their distinct characteristics and enhance downstream task performance. Our method aims to alleviate the potential negative effects of using the same normalization statistics for both token types, which may not be optimally aligned with their individual roles. We empirically show that by utilizing a separate normalization layer, the [CLS] embeddings can better encode the global contextual information and are distributed more uniformly in its anisotropic space. When replacing the conventional normalization layer with the two separate layers, we observe an average 2.7% performance improvement over the image, natural language, and graph domains.

{{</citation>}}


### (50/120) PopBERT. Detecting populism and its host ideologies in the German Bundestag (L. Erhard et al., 2023)

{{<citation>}}

L. Erhard, S. Hanke, U. Remer, A. Falenska, R. Heiberger. (2023)  
**PopBERT. Detecting populism and its host ideologies in the German Bundestag**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.14355v1)  

---


**ABSTRACT**  
The rise of populism concerns many political scientists and practitioners, yet the detection of its underlying language remains fragmentary. This paper aims to provide a reliable, valid, and scalable approach to measure populist stances. For that purpose, we created an annotated dataset based on parliamentary speeches of the German Bundestag (2013 to 2021). Following the ideational definition of populism, we label moralizing references to the virtuous people or the corrupt elite as core dimensions of populist language. To identify, in addition, how the thin ideology of populism is thickened, we annotate how populist statements are attached to left-wing or right-wing host ideologies. We then train a transformer-based model (PopBERT) as a multilabel classifier to detect and quantify each dimension. A battery of validation checks reveals that the model has a strong predictive accuracy, provides high qualitative face validity, matches party rankings of expert surveys, and detects out-of-sample text snippets correctly. PopBERT enables dynamic analyses of how German-speaking politicians and parties use populist language as a strategic device. Furthermore, the annotator-level data may also be applied in cross-domain applications or to develop related classifiers.

{{</citation>}}


### (51/120) ProtoEM: A Prototype-Enhanced Matching Framework for Event Relation Extraction (Zhilei Hu et al., 2023)

{{<citation>}}

Zhilei Hu, Zixuan Li, Daozhu Xu, Long Bai, Cheng Jin, Xiaolong Jin, Jiafeng Guo, Xueqi Cheng. (2023)  
**ProtoEM: A Prototype-Enhanced Matching Framework for Event Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GNN, Graph Neural Network, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2309.12892v1)  

---


**ABSTRACT**  
Event Relation Extraction (ERE) aims to extract multiple kinds of relations among events in texts. However, existing methods singly categorize event relations as different classes, which are inadequately capturing the intrinsic semantics of these relations. To comprehensively understand their intrinsic semantics, in this paper, we obtain prototype representations for each type of event relation and propose a Prototype-Enhanced Matching (ProtoEM) framework for the joint extraction of multiple kinds of event relations. Specifically, ProtoEM extracts event relations in a two-step manner, i.e., prototype representing and prototype matching. In the first step, to capture the connotations of different event relations, ProtoEM utilizes examples to represent the prototypes corresponding to these relations. Subsequently, to capture the interdependence among event relations, it constructs a dependency graph for the prototypes corresponding to these relations and utilized a Graph Neural Network (GNN)-based module for modeling. In the second step, it obtains the representations of new event pairs and calculates their similarity with those prototypes obtained in the first step to evaluate which types of event relations they belong to. Experimental results on the MAVEN-ERE dataset demonstrate that the proposed ProtoEM framework can effectively represent the prototypes of event relations and further obtain a significant improvement over baseline models.

{{</citation>}}


### (52/120) Affect Recognition in Conversations Using Large Language Models (Shutong Feng et al., 2023)

{{<citation>}}

Shutong Feng, Guangzhi Sun, Nurul Lubis, Chao Zhang, Milica Gašić. (2023)  
**Affect Recognition in Conversations Using Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.12881v1)  

---


**ABSTRACT**  
Affect recognition, encompassing emotions, moods, and feelings, plays a pivotal role in human communication. In the realm of conversational artificial intelligence (AI), the ability to discern and respond to human affective cues is a critical factor for creating engaging and empathetic interactions. This study delves into the capacity of large language models (LLMs) to recognise human affect in conversations, with a focus on both open-domain chit-chat dialogues and task-oriented dialogues. Leveraging three diverse datasets, namely IEMOCAP, EmoWOZ, and DAIC-WOZ, covering a spectrum of dialogues from casual conversations to clinical interviews, we evaluated and compared LLMs' performance in affect recognition. Our investigation explores the zero-shot and few-shot capabilities of LLMs through in-context learning (ICL) as well as their model capacities through task-specific fine-tuning. Additionally, this study takes into account the potential impact of automatic speech recognition (ASR) errors on LLM predictions. With this work, we aim to shed light on the extent to which LLMs can replicate human-like affect recognition capabilities in conversations.

{{</citation>}}


### (53/120) AnglE-Optimized Text Embeddings (Xianming Li et al., 2023)

{{<citation>}}

Xianming Li, Jing Li. (2023)  
**AnglE-Optimized Text Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2309.12871v1)  

---


**ABSTRACT**  
High-quality text embedding is pivotal in improving semantic textual similarity (STS) tasks, which are crucial components in Large Language Model (LLM) applications. However, a common challenge existing text embedding models face is the problem of vanishing gradients, primarily due to their reliance on the cosine function in the optimization objective, which has saturation zones. To address this issue, this paper proposes a novel angle-optimized text embedding model called AnglE. The core idea of AnglE is to introduce angle optimization in a complex space. This novel approach effectively mitigates the adverse effects of the saturation zone in the cosine function, which can impede gradient and hinder optimization processes. To set up a comprehensive STS evaluation, we experimented on existing short-text STS datasets and a newly collected long-text STS dataset from GitHub Issues. Furthermore, we examine domain-specific STS scenarios with limited labeled data and explore how AnglE works with LLM-annotated data. Extensive experiments were conducted on various tasks including short-text STS, long-text STS, and domain-specific STS tasks. The results show that AnglE outperforms the state-of-the-art (SOTA) STS models that ignore the cosine saturation zone. These findings demonstrate the ability of AnglE to generate high-quality text embeddings and the usefulness of angle optimization in STS.

{{</citation>}}


### (54/120) Domain Adaptation for Arabic Machine Translation: The Case of Financial Texts (Emad A. Alghamdi et al., 2023)

{{<citation>}}

Emad A. Alghamdi, Jezia Zakraoui, Fares A. Abanmy. (2023)  
**Domain Adaptation for Arabic Machine Translation: The Case of Financial Texts**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, Financial, GPT, GPT-3.5, Machine Translation  
[Paper Link](http://arxiv.org/abs/2309.12863v1)  

---


**ABSTRACT**  
Neural machine translation (NMT) has shown impressive performance when trained on large-scale corpora. However, generic NMT systems have demonstrated poor performance on out-of-domain translation. To mitigate this issue, several domain adaptation methods have recently been proposed which often lead to better translation quality than genetic NMT systems. While there has been some continuous progress in NMT for English and other European languages, domain adaption in Arabic has received little attention in the literature. The current study, therefore, aims to explore the effectiveness of domain-specific adaptation for Arabic MT (AMT), in yet unexplored domain, financial news articles. To this end, we developed carefully a parallel corpus for Arabic-English (AR- EN) translation in the financial domain for benchmarking different domain adaptation methods. We then fine-tuned several pre-trained NMT and Large Language models including ChatGPT-3.5 Turbo on our dataset. The results showed that the fine-tuning is successful using just a few well-aligned in-domain AR-EN segments. The quality of ChatGPT translation was superior than other models based on automatic and human evaluations. To the best of our knowledge, this is the first work on fine-tuning ChatGPT towards financial domain transfer learning. To contribute to research in domain translation, we made our datasets and fine-tuned models available at https://huggingface.co/asas-ai/.

{{</citation>}}


### (55/120) StyloMetrix: An Open-Source Multilingual Tool for Representing Stylometric Vectors (Inez Okulska et al., 2023)

{{<citation>}}

Inez Okulska, Daria Stetsenko, Anna Kołos, Agnieszka Karlińska, Kinga Głąbińska, Adam Nowakowski. (2023)  
**StyloMetrix: An Open-Source Multilingual Tool for Representing Stylometric Vectors**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Multilingual, Transformer  
[Paper Link](http://arxiv.org/abs/2309.12810v1)  

---


**ABSTRACT**  
This work aims to provide an overview on the open-source multilanguage tool called StyloMetrix. It offers stylometric text representations that cover various aspects of grammar, syntax and lexicon. StyloMetrix covers four languages: Polish as the primary language, English, Ukrainian and Russian. The normalized output of each feature can become a fruitful course for machine learning models and a valuable addition to the embeddings layer for any deep learning algorithm. We strive to provide a concise, but exhaustive overview on the application of the StyloMetrix vectors as well as explain the sets of the developed linguistic features. The experiments have shown promising results in supervised content classification with simple algorithms as Random Forest Classifier, Voting Classifier, Logistic Regression and others. The deep learning assessments have unveiled the usefulness of the StyloMetrix vectors at enhancing an embedding layer extracted from Transformer architectures. The StyloMetrix has proven itself to be a formidable source for the machine learning and deep learning algorithms to execute different classification tasks.

{{</citation>}}


### (56/120) ChatPRCS: A Personalized Support System for English Reading Comprehension based on ChatGPT (Xizhe Wang et al., 2023)

{{<citation>}}

Xizhe Wang, Yihua Zhong, Changqin Huang, Xiaodi Huang. (2023)  
**ChatPRCS: A Personalized Support System for English Reading Comprehension based on ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.12808v2)  

---


**ABSTRACT**  
As a common approach to learning English, reading comprehension primarily entails reading articles and answering related questions. However, the complexity of designing effective exercises results in students encountering standardized questions, making it challenging to align with individualized learners' reading comprehension ability. By leveraging the advanced capabilities offered by large language models, exemplified by ChatGPT, this paper presents a novel personalized support system for reading comprehension, referred to as ChatPRCS, based on the Zone of Proximal Development theory. ChatPRCS employs methods including reading comprehension proficiency prediction, question generation, and automatic evaluation, among others, to enhance reading comprehension instruction. First, we develop a new algorithm that can predict learners' reading comprehension abilities using their historical data as the foundation for generating questions at an appropriate level of difficulty. Second, a series of new ChatGPT prompt patterns is proposed to address two key aspects of reading comprehension objectives: question generation, and automated evaluation. These patterns further improve the quality of generated questions. Finally, by integrating personalized ability and reading comprehension prompt patterns, ChatPRCS is systematically validated through experiments. Empirical results demonstrate that it provides learners with high-quality reading comprehension questions that are broadly aligned with expert-crafted questions at a statistical level.

{{</citation>}}


### (57/120) Furthest Reasoning with Plan Assessment: Stable Reasoning Path with Retrieval-Augmented Large Language Models (Yin Zhu et al., 2023)

{{<citation>}}

Yin Zhu, Zhiling Luo, Gong Cheng. (2023)  
**Furthest Reasoning with Plan Assessment: Stable Reasoning Path with Retrieval-Augmented Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, QA, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.12767v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), acting as a powerful reasoner and generator, exhibit extraordinary performance across various natural language tasks, such as question answering (QA). Among these tasks, Multi-Hop Question Answering (MHQA) stands as a widely discussed category, necessitating seamless integration between LLMs and the retrieval of external knowledge. Existing methods employ LLM to generate reasoning paths and plans, and utilize IR to iteratively retrieve related knowledge, but these approaches have inherent flaws. On one hand, Information Retriever (IR) is hindered by the low quality of generated queries by LLM. On the other hand, LLM is easily misguided by the irrelevant knowledge by IR. These inaccuracies, accumulated by the iterative interaction between IR and LLM, lead to a disaster in effectiveness at the end. To overcome above barriers, in this paper, we propose a novel pipeline for MHQA called Furthest-Reasoning-with-Plan-Assessment (FuRePA), including an improved framework (Furthest Reasoning) and an attached module (Plan Assessor). 1) Furthest reasoning operates by masking previous reasoning path and generated queries for LLM, encouraging LLM generating chain of thought from scratch in each iteration. This approach enables LLM to break the shackle built by previous misleading thoughts and queries (if any). 2) The Plan Assessor is a trained evaluator that selects an appropriate plan from a group of candidate plans proposed by LLM. Our methods are evaluated on three highly recognized public multi-hop question answering datasets and outperform state-of-the-art on most metrics (achieving a 10%-12% in answer accuracy).

{{</citation>}}


### (58/120) Semantic similarity prediction is better than other semantic similarity measures (Steffen Herbold, 2023)

{{<citation>}}

Steffen Herbold. (2023)  
**Semantic similarity prediction is better than other semantic similarity measures**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: BERT, BLEU, GLUE  
[Paper Link](http://arxiv.org/abs/2309.12697v1)  

---


**ABSTRACT**  
Semantic similarity between natural language texts is typically measured either by looking at the overlap between subsequences (e.g., BLEU) or by using embeddings (e.g., BERTScore, S-BERT). Within this paper, we argue that when we are only interested in measuring the semantic similarity, it is better to directly predict the similarity using a fine-tuned model for such a task. Using a fine-tuned model for the STS-B from the GLUE benchmark, we define the STSScore approach and show that the resulting similarity is better aligned with our expectations on a robust semantic similarity measure than other approaches.

{{</citation>}}


### (59/120) HRoT: Hybrid prompt strategy and Retrieval of Thought for Table-Text Hybrid Question Answering (Tongxu Luo et al., 2023)

{{<citation>}}

Tongxu Luo, Fangyu Lei, Jiahe Lei, Weihao Liu, Shihu He, Jun Zhao, Kang Liu. (2023)  
**HRoT: Hybrid prompt strategy and Retrieval of Thought for Table-Text Hybrid Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.12669v1)  

---


**ABSTRACT**  
Answering numerical questions over hybrid contents from the given tables and text(TextTableQA) is a challenging task. Recently, Large Language Models (LLMs) have gained significant attention in the NLP community. With the emergence of large language models, In-Context Learning and Chain-of-Thought prompting have become two particularly popular research topics in this field. In this paper, we introduce a new prompting strategy called Hybrid prompt strategy and Retrieval of Thought for TextTableQA. Through In-Context Learning, we prompt the model to develop the ability of retrieval thinking when dealing with hybrid data. Our method achieves superior performance compared to the fully-supervised SOTA on the MultiHiertt dataset in the few-shot setting.

{{</citation>}}


### (60/120) Decoding Affect in Dyadic Conversations: Leveraging Semantic Similarity through Sentence Embedding (Chen-Wei Yu et al., 2023)

{{<citation>}}

Chen-Wei Yu, Yun-Shiuan Chuang, Alexandros N. Lotsos, Claudia M. Haase. (2023)  
**Decoding Affect in Dyadic Conversations: Leveraging Semantic Similarity through Sentence Embedding**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, NLP, Natural Language Processing, Semantic Similarity, Sentence Embedding, Transformer  
[Paper Link](http://arxiv.org/abs/2309.12646v1)  

---


**ABSTRACT**  
Recent advancements in Natural Language Processing (NLP) have highlighted the potential of sentence embeddings in measuring semantic similarity. Yet, its application in analyzing real-world dyadic interactions and predicting the affect of conversational participants remains largely uncharted. To bridge this gap, the present study utilizes verbal conversations within 50 married couples talking about conflicts and pleasant activities. Transformer-based model all-MiniLM-L6-v2 was employed to obtain the embeddings of the utterances from each speaker. The overall similarity of the conversation was then quantified by the average cosine similarity between the embeddings of adjacent utterances. Results showed that semantic similarity had a positive association with wives' affect during conflict (but not pleasant) conversations. Moreover, this association was not observed with husbands' affect regardless of conversation types. Two validation checks further provided support for the validity of the similarity measure and showed that the observed patterns were not mere artifacts of data. The present study underscores the potency of sentence embeddings in understanding the association between interpersonal dynamics and individual affect, paving the way for innovative applications in affective and relationship sciences.

{{</citation>}}


### (61/120) Learning to Diversify Neural Text Generation via Degenerative Model (Jimin Hong et al., 2023)

{{<citation>}}

Jimin Hong, ChaeHun Park, Jaegul Choo. (2023)  
**Learning to Diversify Neural Text Generation via Degenerative Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Text Generation  
[Paper Link](http://arxiv.org/abs/2309.12619v1)  

---


**ABSTRACT**  
Neural language models often fail to generate diverse and informative texts, limiting their applicability in real-world problems. While previous approaches have proposed to address these issues by identifying and penalizing undesirable behaviors (e.g., repetition, overuse of frequent words) from language models, we propose an alternative approach based on an observation: models primarily learn attributes within examples that are likely to cause degeneration problems. Based on this observation, we propose a new approach to prevent degeneration problems by training two models. Specifically, we first train a model that is designed to amplify undesirable patterns. We then enhance the diversity of the second model by focusing on patterns that the first model fails to learn. Extensive experiments on two tasks, namely language modeling and dialogue generation, demonstrate the effectiveness of our approach.

{{</citation>}}


### (62/120) Unlocking Model Insights: A Dataset for Automated Model Card Generation (Shruti Singh et al., 2023)

{{<citation>}}

Shruti Singh, Hitesh Lodwal, Husain Malwat, Rakesh Thakur, Mayank Singh. (2023)  
**Unlocking Model Insights: A Dataset for Automated Model Card Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, ChatGPT, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2309.12616v1)  

---


**ABSTRACT**  
Language models (LMs) are no longer restricted to ML community, and instruction-tuned LMs have led to a rise in autonomous AI agents. As the accessibility of LMs grows, it is imperative that an understanding of their capabilities, intended usage, and development cycle also improves. Model cards are a popular practice for documenting detailed information about an ML model. To automate model card generation, we introduce a dataset of 500 question-answer pairs for 25 ML models that cover crucial aspects of the model, such as its training configurations, datasets, biases, architecture details, and training resources. We employ annotators to extract the answers from the original paper. Further, we explore the capabilities of LMs in generating model cards by answering questions. Our initial experiments with ChatGPT-3.5, LLaMa, and Galactica showcase a significant gap in the understanding of research papers by these aforementioned LMs as well as generating factual textual responses. We posit that our dataset can be used to train models to automate the generation of model cards from paper text and reduce human effort in the model card curation process. The complete dataset is available on https://osf.io/hqt7p/?view_only=3b9114e3904c4443bcd9f5c270158d37

{{</citation>}}


### (63/120) Is it Possible to Modify Text to a Target Readability Level? An Initial Investigation Using Zero-Shot Large Language Models (Asma Farajidizaji et al., 2023)

{{<citation>}}

Asma Farajidizaji, Vatsal Raina, Mark Gales. (2023)  
**Is it Possible to Modify Text to a Target Readability Level? An Initial Investigation Using Zero-Shot Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.12551v1)  

---


**ABSTRACT**  
Text simplification is a common task where the text is adapted to make it easier to understand. Similarly, text elaboration can make a passage more sophisticated, offering a method to control the complexity of reading comprehension tests. However, text simplification and elaboration tasks are limited to only relatively alter the readability of texts. It is useful to directly modify the readability of any text to an absolute target readability level to cater to a diverse audience. Ideally, the readability of readability-controlled generated text should be independent of the source text. Therefore, we propose a novel readability-controlled text modification task. The task requires the generation of 8 versions at various target readability levels for each input text. We introduce novel readability-controlled text modification metrics. The baselines for this task use ChatGPT and Llama-2, with an extension approach introducing a two-step process (generating paraphrases by passing through the language model twice). The zero-shot approaches are able to push the readability of the paraphrases in the desired direction but the final readability remains correlated with the original text's readability. We also find greater drops in semantic and lexical similarity between the source and target texts with greater shifts in the readability.

{{</citation>}}


### (64/120) Automatic Answerability Evaluation for Question Generation (Zifan Wang et al., 2023)

{{<citation>}}

Zifan Wang, Kotaro Funakoshi, Manabu Okumura. (2023)  
**Automatic Answerability Evaluation for Question Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BLEU, ChatGPT, GPT, Question Generation  
[Paper Link](http://arxiv.org/abs/2309.12546v1)  

---


**ABSTRACT**  
Conventional automatic evaluation metrics, such as BLEU and ROUGE, developed for natural language generation (NLG) tasks, are based on measuring the n-gram overlap between the generated and reference text. These simple metrics may be insufficient for more complex tasks, such as question generation (QG), which requires generating questions that are answerable by the reference answers. Developing a more sophisticated automatic evaluation metric, thus, remains as an urgent problem in QG research. This work proposes a Prompting-based Metric on ANswerability (PMAN), a novel automatic evaluation metric to assess whether the generated questions are answerable by the reference answers for the QG tasks. Extensive experiments demonstrate that its evaluation results are reliable and align with human evaluations. We further apply our metric to evaluate the performance of QG models, which shows our metric complements conventional metrics. Our implementation of a ChatGPT-based QG model achieves state-of-the-art (SOTA) performance in generating answerable questions.

{{</citation>}}


## cs.HC (4)



### (65/120) SurrealDriver: Designing Generative Driver Agent Simulation Framework in Urban Contexts based on Large Language Model (Ye Jin et al., 2023)

{{<citation>}}

Ye Jin, Xiaoxi Shen, Huiling Peng, Xiaoan Liu, Jingli Qin, Jiayang Li, Jintao Xie, Peizhong Gao, Guyue Zhou, Jiangtao Gong. (2023)  
**SurrealDriver: Designing Generative Driver Agent Simulation Framework in Urban Contexts based on Large Language Model**  

---
Primary Category: cs.HC  
Categories: H-5-2, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.13193v1)  

---


**ABSTRACT**  
Simulation plays a critical role in the research and development of autonomous driving and intelligent transportation systems. However, the current simulation platforms exhibit limitations in the realism and diversity of agent behaviors, which impede the transfer of simulation outcomes to the real world. In this paper, we propose a generative driver agent simulation framework based on large language models (LLMs), capable of perceiving complex traffic scenarios and providing realistic driving maneuvers. Notably, we conducted interviews with 24 drivers and used their detailed descriptions of driving behavior as chain-of-thought prompts to develop a `coach agent' module, which can evaluate and assist driver agents in accumulating driving experience and developing human-like driving styles. Through practical simulation experiments and user experiments, we validate the feasibility of this framework in generating reliable driver agents and analyze the roles of each module. The results show that the framework with full architect decreased the collision rate by 81.04% and increased the human-likeness by 50%. Our research proposes the first urban context driver agent simulation framework based on LLMs and provides valuable insights into the future of agent simulation for complex tasks.

{{</citation>}}


### (66/120) Using ChatGPT in HCI Research -- A Trioethnography (Smit Desai et al., 2023)

{{<citation>}}

Smit Desai, Tanusree Sharma, Pratyasha Saha. (2023)  
**Using ChatGPT in HCI Research -- A Trioethnography**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.12583v1)  

---


**ABSTRACT**  
This paper explores the lived experience of using ChatGPT in HCI research through a month-long trioethnography. Our approach combines the expertise of three HCI researchers with diverse research interests to reflect on our daily experience of living and working with ChatGPT. Our findings are presented as three provocations grounded in our collective experiences and HCI theories. Specifically, we examine (1) the emotional impact of using ChatGPT, with a focus on frustration and embarrassment, (2) the absence of accountability and consideration of future implications in design, and raise (3) questions around bias from a Global South perspective. Our work aims to inspire critical discussions about utilizing ChatGPT in HCI research and advance equitable and inclusive technological development.

{{</citation>}}


### (67/120) Creativity Support in the Age of Large Language Models: An Empirical Study Involving Emerging Writers (Tuhin Chakrabarty et al., 2023)

{{<citation>}}

Tuhin Chakrabarty, Vishakh Padmakumar, Faeze Brahman, Smaranda Muresan. (2023)  
**Creativity Support in the Age of Large Language Models: An Empirical Study Involving Emerging Writers**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-CL, cs-CY, cs-HC, cs.HC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.12570v2)  

---


**ABSTRACT**  
The development of large language models (LLMs) capable of following instructions and engaging in conversational interactions sparked increased interest in their utilization across various support tools. We investigate the utility of modern LLMs in assisting professional writers via an empirical user study (n=30). The design of our collaborative writing interface is grounded in the cognitive process model of writing that views writing as a goal-oriented thinking process encompassing non-linear cognitive activities: planning, translating, and reviewing. Participants are asked to submit a post-completion survey to provide feedback on the potential and pitfalls of LLMs as writing collaborators. Upon analyzing the writer-LLM interactions, we find that while writers seek LLM's help across all three types of cognitive activities, they find LLMs more helpful in translation and reviewing. Our findings from analyzing both the interactions and the survey responses highlight future research directions in creative writing assistance using LLMs.

{{</citation>}}


### (68/120) PlanFitting: Tailoring Personalized Exercise Plans with Large Language Models (Donghoon Shin et al., 2023)

{{<citation>}}

Donghoon Shin, Gary Hsieh, Young-Ho Kim. (2023)  
**PlanFitting: Tailoring Personalized Exercise Plans with Large Language Models**  

---
Primary Category: cs.HC  
Categories: H-5-2; I-2-7, cs-AI, cs-CL, cs-HC, cs.HC  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.12555v1)  

---


**ABSTRACT**  
A personally tailored exercise regimen is crucial to ensuring sufficient physical activities, yet challenging to create as people have complex schedules and considerations and the creation of plans often requires iterations with experts. We present PlanFitting, a conversational AI that assists in personalized exercise planning. Leveraging generative capabilities of large language models, PlanFitting enables users to describe various constraints and queries in natural language, thereby facilitating the creation and refinement of their weekly exercise plan to suit their specific circumstances while staying grounded in foundational principles. Through a user study where participants (N=18) generated a personalized exercise plan using PlanFitting and expert planners (N=3) evaluated these plans, we identified the potential of PlanFitting in generating personalized, actionable, and evidence-based exercise plans. We discuss future design opportunities for AI assistants in creating plans that better comply with exercise principles and accommodate personal constraints.

{{</citation>}}


## cs.LG (18)



### (69/120) Towards Green AI in Fine-tuning Large Language Models via Adaptive Backpropagation (Kai Huang et al., 2023)

{{<citation>}}

Kai Huang, Hanyun Yin, Heng Huang, Wei Gao. (2023)  
**Towards Green AI in Fine-tuning Large Language Models via Adaptive Backpropagation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.13192v1)  

---


**ABSTRACT**  
Fine-tuning is the most effective way of adapting pre-trained large language models (LLMs) to downstream applications. With the fast growth of LLM-enabled AI applications and democratization of open-souced LLMs, fine-tuning has become possible for non-expert individuals, but intensively performed LLM fine-tuning worldwide could result in significantly high energy consumption and carbon footprint, which may bring large environmental impact. Mitigating such environmental impact towards Green AI directly correlates to reducing the FLOPs of fine-tuning, but existing techniques on efficient LLM fine-tuning can only achieve limited reduction of such FLOPs, due to their ignorance of the backpropagation cost in fine-tuning. To address this limitation, in this paper we present GreenTrainer, a new LLM fine-tuning technique that adaptively evaluates different tensors' backpropagation costs and contributions to the fine-tuned model accuracy, to minimize the fine-tuning cost by selecting the most appropriate set of tensors in training. Such selection in GreenTrainer is made based on a given objective of FLOPs reduction, which can flexibly adapt to the carbon footprint in energy supply and the need in Green AI. Experiment results over multiple open-sourced LLM models and abstractive summarization datasets show that, compared to fine-tuning the whole LLM model, GreenTrainer can save up to 64% FLOPs in fine-tuning without any noticeable model accuracy loss. Compared to the existing fine-tuning techniques such as LoRa, GreenTrainer can achieve up to 4% improvement on model accuracy with on-par FLOPs reduction.

{{</citation>}}


### (70/120) Spatial-frequency channels, shape bias, and adversarial robustness (Ajay Subramanian et al., 2023)

{{<citation>}}

Ajay Subramanian, Elena Sizikova, Najib J. Majaj, Denis G. Pelli. (2023)  
**Spatial-frequency channels, shape bias, and adversarial robustness**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG, eess-IV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.13190v1)  

---


**ABSTRACT**  
What spatial frequency information do humans and neural networks use to recognize objects? In neuroscience, critical band masking is an established tool that can reveal the frequency-selective filters used for object recognition. Critical band masking measures the sensitivity of recognition performance to noise added at each spatial frequency. Existing critical band masking studies show that humans recognize periodic patterns (gratings) and letters by means of a spatial-frequency filter (or "channel'') that has a frequency bandwidth of one octave (doubling of frequency). Here, we introduce critical band masking as a task for network-human comparison and test 14 humans and 76 neural networks on 16-way ImageNet categorization in the presence of narrowband noise. We find that humans recognize objects in natural images using the same one-octave-wide channel that they use for letters and gratings, making it a canonical feature of human object recognition. On the other hand, the neural network channel, across various architectures and training strategies, is 2-4 times as wide as the human channel. In other words, networks are vulnerable to high and low frequency noise that does not affect human performance. Adversarial and augmented-image training are commonly used to increase network robustness and shape bias. Does this training align network and human object recognition channels? Three network channel properties (bandwidth, center frequency, peak noise sensitivity) correlate strongly with shape bias (53% variance explained) and with robustness of adversarially-trained networks (74% variance explained). Adversarial training increases robustness but expands the channel bandwidth even further away from the human bandwidth. Thus, critical band masking reveals that the network channel is more than twice as wide as the human channel, and that adversarial training only increases this difference.

{{</citation>}}


### (71/120) Enhancing Multi-Objective Optimization through Machine Learning-Supported Multiphysics Simulation (Diego Botache et al., 2023)

{{<citation>}}

Diego Botache, Jens Decke, Winfried Ripken, Abhinay Dornipati, Franz Götz-Hahn, Mohamed Ayeb, Bernhard Sick. (2023)  
**Enhancing Multi-Objective Optimization through Machine Learning-Supported Multiphysics Simulation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.13179v1)  

---


**ABSTRACT**  
Multiphysics simulations that involve multiple coupled physical phenomena quickly become computationally expensive. This imposes challenges for practitioners aiming to find optimal configurations for these problems satisfying multiple objectives, as optimization algorithms often require querying the simulation many times. This paper presents a methodological framework for training, self-optimizing, and self-organizing surrogate models to approximate and speed up Multiphysics simulations. We generate two real-world tabular datasets, which we make publicly available, and show that surrogate models can be trained on relatively small amounts of data to approximate the underlying simulations accurately. We conduct extensive experiments combining four machine learning and deep learning algorithms with two optimization algorithms and a comprehensive evaluation strategy. Finally, we evaluate the performance of our combined training and optimization pipeline by verifying the generated Pareto-optimal results using the ground truth simulations. We also employ explainable AI techniques to analyse our surrogates and conduct a preselection strategy to determine the most relevant features in our real-world examples. This approach lets us understand the underlying problem and identify critical partial dependencies.

{{</citation>}}


### (72/120) Flow Factorized Representation Learning (Yue Song et al., 2023)

{{<citation>}}

Yue Song, T. Anderson Keller, Nicu Sebe, Max Welling. (2023)  
**Flow Factorized Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.13167v1)  

---


**ABSTRACT**  
A prominent goal of representation learning research is to achieve representations which are factorized in a useful manner with respect to the ground truth factors of variation. The fields of disentangled and equivariant representation learning have approached this ideal from a range of complimentary perspectives; however, to date, most approaches have proven to either be ill-specified or insufficiently flexible to effectively separate all realistic factors of interest in a learned latent space. In this work, we propose an alternative viewpoint on such structured representation learning which we call Flow Factorized Representation Learning, and demonstrate it to learn both more efficient and more usefully structured representations than existing frameworks. Specifically, we introduce a generative model which specifies a distinct set of latent probability paths that define different input transformations. Each latent flow is generated by the gradient field of a learned potential following dynamic optimal transport. Our novel setup brings new understandings to both \textit{disentanglement} and \textit{equivariance}. We show that our model achieves higher likelihoods on standard representation learning benchmarks while simultaneously being closer to approximately equivariant models. Furthermore, we demonstrate that the transformations learned by our model are flexibly composable and can also extrapolate to new data, implying a degree of robustness and generalizability approaching the ultimate goal of usefully factorized representation learning.

{{</citation>}}


### (73/120) Graph Neural Network for Stress Predictions in Stiffened Panels Under Uniform Loading (Yuecheng Cai et al., 2023)

{{<citation>}}

Yuecheng Cai, Jasmin Jelovica. (2023)  
**Graph Neural Network for Stress Predictions in Stiffened Panels Under Uniform Loading**  

---
Primary Category: cs.LG  
Categories: 74B10 (Primary) 74B02, 68T02 (Secondary), J-2, cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network  
[Paper Link](http://arxiv.org/abs/2309.13022v1)  

---


**ABSTRACT**  
Machine learning (ML) and deep learning (DL) techniques have gained significant attention as reduced order models (ROMs) to computationally expensive structural analysis methods, such as finite element analysis (FEA). Graph neural network (GNN) is a particular type of neural network which processes data that can be represented as graphs. This allows for efficient representation of complex geometries that can change during conceptual design of a structure or a product. In this study, we propose a novel graph embedding technique for efficient representation of 3D stiffened panels by considering separate plate domains as vertices. This approach is considered using Graph Sampling and Aggregation (GraphSAGE) to predict stress distributions in stiffened panels with varying geometries. A comparison between a finite-element-vertex graph representation is conducted to demonstrate the effectiveness of the proposed approach. A comprehensive parametric study is performed to examine the effect of structural geometry on the prediction performance. Our results demonstrate the immense potential of graph neural networks with the proposed graph embedding method as robust reduced-order models for 3D structures.

{{</citation>}}


### (74/120) A Hybrid Deep Learning-based Approach for Optimal Genotype by Environment Selection (Zahra Khalilzadeh et al., 2023)

{{<citation>}}

Zahra Khalilzadeh, Motahareh Kashanian, Saeed Khaki, Lizhi Wang. (2023)  
**A Hybrid Deep Learning-based Approach for Optimal Genotype by Environment Selection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, q-bio-QM, stat-ML  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.13021v1)  

---


**ABSTRACT**  
Precise crop yield prediction is essential for improving agricultural practices and ensuring crop resilience in varying climates. Integrating weather data across the growing season, especially for different crop varieties, is crucial for understanding their adaptability in the face of climate change. In the MLCAS2021 Crop Yield Prediction Challenge, we utilized a dataset comprising 93,028 training records to forecast yields for 10,337 test records, covering 159 locations across 28 U.S. states and Canadian provinces over 13 years (2003-2015). This dataset included details on 5,838 distinct genotypes and daily weather data for a 214-day growing season, enabling comprehensive analysis. As one of the winning teams, we developed two novel convolutional neural network (CNN) architectures: the CNN-DNN model, combining CNN and fully-connected networks, and the CNN-LSTM-DNN model, with an added LSTM layer for weather variables. Leveraging the Generalized Ensemble Method (GEM), we determined optimal model weights, resulting in superior performance compared to baseline models. The GEM model achieved lower RMSE (5.55% to 39.88%), reduced MAE (5.34% to 43.76%), and higher correlation coefficients (1.1% to 10.79%) when evaluated on test data. We applied the CNN-DNN model to identify top-performing genotypes for various locations and weather conditions, aiding genotype selection based on weather variables. Our data-driven approach is valuable for scenarios with limited testing years. Additionally, a feature importance analysis using RMSE change highlighted the significance of location, MG, year, and genotype, along with the importance of weather variables MDNI and AP.

{{</citation>}}


### (75/120) Higher-order Graph Convolutional Network with Flower-Petals Laplacians on Simplicial Complexes (Yiming Huang et al., 2023)

{{<citation>}}

Yiming Huang, Yujie Zeng, Qiang Wu, Linyuan Lü. (2023)  
**Higher-order Graph Convolutional Network with Flower-Petals Laplacians on Simplicial Complexes**  

---
Primary Category: cs.LG  
Categories: cond-mat-stat-mech, cs-AI, cs-LG, cs-SI, cs.LG, physics-soc-ph  
Keywords: GNN, Graph Convolutional Network, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.12971v1)  

---


**ABSTRACT**  
Despite the recent successes of vanilla Graph Neural Networks (GNNs) on many tasks, their foundation on pairwise interaction networks inherently limits their capacity to discern latent higher-order interactions in complex systems. To bridge this capability gap, we propose a novel approach exploiting the rich mathematical theory of simplicial complexes (SCs) - a robust tool for modeling higher-order interactions. Current SC-based GNNs are burdened by high complexity and rigidity, and quantifying higher-order interaction strengths remains challenging. Innovatively, we present a higher-order Flower-Petals (FP) model, incorporating FP Laplacians into SCs. Further, we introduce a Higher-order Graph Convolutional Network (HiGCN) grounded in FP Laplacians, capable of discerning intrinsic features across varying topological scales. By employing learnable graph filters, a parameter group within each FP Laplacian domain, we can identify diverse patterns where the filters' weights serve as a quantifiable measure of higher-order interaction strengths. The theoretical underpinnings of HiGCN's advanced expressiveness are rigorously demonstrated. Additionally, our empirical investigations reveal that the proposed model accomplishes state-of-the-art (SOTA) performance on a range of graph tasks and provides a scalable and flexible solution to explore higher-order interactions in graphs.

{{</citation>}}


### (76/120) BayesDLL: Bayesian Deep Learning Library (Minyoung Kim et al., 2023)

{{<citation>}}

Minyoung Kim, Timothy Hospedales. (2023)  
**BayesDLL: Bayesian Deep Learning Library**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.12928v1)  

---


**ABSTRACT**  
We release a new Bayesian neural network library for PyTorch for large-scale deep networks. Our library implements mainstream approximate Bayesian inference algorithms: variational inference, MC-dropout, stochastic-gradient MCMC, and Laplace approximation. The main differences from other existing Bayesian neural network libraries are as follows: 1) Our library can deal with very large-scale deep networks including Vision Transformers (ViTs). 2) We need virtually zero code modifications for users (e.g., the backbone network definition codes do not neet to be modified at all). 3) Our library also allows the pre-trained model weights to serve as a prior mean, which is very useful for performing Bayesian inference with the large-scale foundation models like ViTs that are hard to optimise from scratch with the downstream data alone. Our code is publicly available at: \url{https://github.com/SamsungLabs/BayesDLL}\footnote{A mirror repository is also available at: \url{https://github.com/minyoungkim21/BayesDLL}.}.

{{</citation>}}


### (77/120) Topological Data Mapping of Online Hate Speech, Misinformation, and General Mental Health: A Large Language Model Based Study (Andrew Alexander et al., 2023)

{{<citation>}}

Andrew Alexander, Hongbin Wang. (2023)  
**Topological Data Mapping of Online Hate Speech, Misinformation, and General Mental Health: A Large Language Model Based Study**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-AT, q-bio-NC  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.13098v1)  

---


**ABSTRACT**  
The advent of social media has led to an increased concern over its potential to propagate hate speech and misinformation, which, in addition to contributing to prejudice and discrimination, has been suspected of playing a role in increasing social violence and crimes in the United States. While literature has shown the existence of an association between posting hate speech and misinformation online and certain personality traits of posters, the general relationship and relevance of online hate speech/misinformation in the context of overall psychological wellbeing of posters remain elusive. One difficulty lies in the lack of adequate data analytics tools capable of adequately analyzing the massive amount of social media posts to uncover the underlying hidden links. Recent progresses in machine learning and large language models such as ChatGPT have made such an analysis possible. In this study, we collected thousands of posts from carefully selected communities on the social media site Reddit. We then utilized OpenAI's GPT3 to derive embeddings of these posts, which are high-dimensional real-numbered vectors that presumably represent the hidden semantics of posts. We then performed various machine-learning classifications based on these embeddings in order to understand the role of hate speech/misinformation in various communities. Finally, a topological data analysis (TDA) was applied to the embeddings to obtain a visual map connecting online hate speech, misinformation, various psychiatric disorders, and general mental health.

{{</citation>}}


### (78/120) Associative Transformer Is A Sparse Representation Learner (Yuwei Sun et al., 2023)

{{<citation>}}

Yuwei Sun, Hideya Ochiai, Zhirong Wu, Stephen Lin, Ryota Kanai. (2023)  
**Associative Transformer Is A Sparse Representation Learner**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs-NE, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.12862v1)  

---


**ABSTRACT**  
Emerging from the monolithic pairwise attention mechanism in conventional Transformer models, there is a growing interest in leveraging sparse interactions that align more closely with biological principles. Approaches including the Set Transformer and the Perceiver employ cross-attention consolidated with a latent space that forms an attention bottleneck with limited capacity. Building upon recent neuroscience studies of Global Workspace Theory and associative memory, we propose the Associative Transformer (AiT). AiT induces low-rank explicit memory that serves as both priors to guide bottleneck attention in the shared workspace and attractors within associative memory of a Hopfield network. Through joint end-to-end training, these priors naturally develop module specialization, each contributing a distinct inductive bias to form attention bottlenecks. A bottleneck can foster competition among inputs for writing information into the memory. We show that AiT is a sparse representation learner, learning distinct priors through the bottlenecks that are complexity-invariant to input quantities and dimensions. AiT demonstrates its superiority over methods such as the Set Transformer, Vision Transformer, and Coordination in various vision tasks.

{{</citation>}}


### (79/120) Reward Function Design for Crowd Simulation via Reinforcement Learning (Ariel Kwiatkowski et al., 2023)

{{<citation>}}

Ariel Kwiatkowski, Vicky Kalogeiton, Julien Pettré, Marie-Paule Cani. (2023)  
**Reward Function Design for Crowd Simulation via Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.12841v1)  

---


**ABSTRACT**  
Crowd simulation is important for video-games design, since it enables to populate virtual worlds with autonomous avatars that navigate in a human-like manner. Reinforcement learning has shown great potential in simulating virtual crowds, but the design of the reward function is critical to achieving effective and efficient results. In this work, we explore the design of reward functions for reinforcement learning-based crowd simulation. We provide theoretical insights on the validity of certain reward functions according to their analytical properties, and evaluate them empirically using a range of scenarios, using the energy efficiency as the metric. Our experiments show that directly minimizing the energy usage is a viable strategy as long as it is paired with an appropriately scaled guiding potential, and enable us to study the impact of the different reward components on the behavior of the simulated crowd. Our findings can inform the development of new crowd simulation techniques, and contribute to the wider study of human-like navigation.

{{</citation>}}


### (80/120) Improving Generalization in Game Agents with Data Augmentation in Imitation Learning (Derek Yadgaroff et al., 2023)

{{<citation>}}

Derek Yadgaroff, Alessandro Sestini, Konrad Tollmar, Linus Gisslén. (2023)  
**Improving Generalization in Game Agents with Data Augmentation in Imitation Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI, Augmentation  
[Paper Link](http://arxiv.org/abs/2309.12815v1)  

---


**ABSTRACT**  
Imitation learning is an effective approach for training game-playing agents and, consequently, for efficient game production. However, generalization - the ability to perform well in related but unseen scenarios - is an essential requirement that remains an unsolved challenge for game AI. Generalization is difficult for imitation learning agents because it requires the algorithm to take meaningful actions outside of the training distribution. In this paper we propose a solution to this challenge. Inspired by the success of data augmentation in supervised learning, we augment the training data so the distribution of states and actions in the dataset better represents the real state-action distribution. This study evaluates methods for combining and applying data augmentations to observations, to improve generalization of imitation learning agents. It also provides a performance benchmark of these augmentations across several 3D environments. These results demonstrate that data augmentation is a promising framework for improving generalization in imitation learning agents.

{{</citation>}}


### (81/120) AMPLIFY:Attention-based Mixup for Performance Improvement and Label Smoothing in Transformer (Leixin Yang et al., 2023)

{{<citation>}}

Leixin Yang, Yaping Zhang, Haoyu Xiong, Yu Xiang. (2023)  
**AMPLIFY:Attention-based Mixup for Performance Improvement and Label Smoothing in Transformer**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Attention, BERT, GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2309.12689v1)  

---


**ABSTRACT**  
Mixup is an effective data augmentation method that generates new augmented samples by aggregating linear combinations of different original samples. However, if there are noises or aberrant features in the original samples, Mixup may propagate them to the augmented samples, leading to over-sensitivity of the model to these outliers . To solve this problem, this paper proposes a new Mixup method called AMPLIFY. This method uses the Attention mechanism of Transformer itself to reduce the influence of noises and aberrant values in the original samples on the prediction results, without increasing additional trainable parameters, and the computational cost is very low, thereby avoiding the problem of high resource consumption in common Mixup methods such as Sentence Mixup . The experimental results show that, under a smaller computational resource cost, AMPLIFY outperforms other Mixup methods in text classification tasks on 7 benchmark datasets, providing new ideas and new ways to further improve the performance of pre-trained models based on the Attention mechanism, such as BERT, ALBERT, RoBERTa, and GPT. Our code can be obtained at https://github.com/kiwi-lilo/AMPLIFY.

{{</citation>}}


### (82/120) How to Fine-tune the Model: Unified Model Shift and Model Bias Policy Optimization (Hai Zhang et al., 2023)

{{<citation>}}

Hai Zhang, Hang Yu, Junqiao Zhao, Di Zhang, ChangHuang, Hongtu Zhou, Xiao Zhang, Chen Ye. (2023)  
**How to Fine-tune the Model: Unified Model Shift and Model Bias Policy Optimization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.12671v1)  

---


**ABSTRACT**  
Designing and deriving effective model-based reinforcement learning (MBRL) algorithms with a performance improvement guarantee is challenging, mainly attributed to the high coupling between model learning and policy optimization. Many prior methods that rely on return discrepancy to guide model learning ignore the impacts of model shift, which can lead to performance deterioration due to excessive model updates. Other methods use performance difference bound to explicitly consider model shift. However, these methods rely on a fixed threshold to constrain model shift, resulting in a heavy dependence on the threshold and a lack of adaptability during the training process. In this paper, we theoretically derive an optimization objective that can unify model shift and model bias and then formulate a fine-tuning process. This process adaptively adjusts the model updates to get a performance improvement guarantee while avoiding model overfitting. Based on these, we develop a straightforward algorithm USB-PO (Unified model Shift and model Bias Policy Optimization). Empirical results show that USB-PO achieves state-of-the-art performance on several challenging benchmark tasks.

{{</citation>}}


### (83/120) OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling (Yi-Fan Zhang et al., 2023)

{{<citation>}}

Yi-Fan Zhang, Qingsong Wen, Xue Wang, Weiqi Chen, Liang Sun, Zhang Zhang, Liang Wang, Rong Jin, Tieniu Tan. (2023)  
**OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling**  

---
Primary Category: cs.LG  
Categories: cs-DS, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2309.12659v1)  

---


**ABSTRACT**  
Online updating of time series forecasting models aims to address the concept drifting problem by efficiently updating forecasting models based on streaming data. Many algorithms are designed for online time series forecasting, with some exploiting cross-variable dependency while others assume independence among variables. Given every data assumption has its own pros and cons in online time series modeling, we propose \textbf{On}line \textbf{e}nsembling \textbf{Net}work (OneNet). It dynamically updates and combines two models, with one focusing on modeling the dependency across the time dimension and the other on cross-variate dependency. Our method incorporates a reinforcement learning-based approach into the traditional online convex programming framework, allowing for the linear combination of the two models with dynamically adjusted weights. OneNet addresses the main shortcoming of classical online learning methods that tend to be slow in adapting to the concept drift. Empirical results show that OneNet reduces online forecasting error by more than $\mathbf{50\%}$ compared to the State-Of-The-Art (SOTA) method. The code is available at \url{https://github.com/yfzhang114/OneNet}.

{{</citation>}}


### (84/120) Sequential Action-Induced Invariant Representation for Reinforcement Learning (Dayang Liang et al., 2023)

{{<citation>}}

Dayang Liang, Qihang Chen, Yunlong Liu. (2023)  
**Sequential Action-Induced Invariant Representation for Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.12628v1)  

---


**ABSTRACT**  
How to accurately learn task-relevant state representations from high-dimensional observations with visual distractions is a realistic and challenging problem in visual reinforcement learning. Recently, unsupervised representation learning methods based on bisimulation metrics, contrast, prediction, and reconstruction have shown the ability for task-relevant information extraction. However, due to the lack of appropriate mechanisms for the extraction of task information in the prediction, contrast, and reconstruction-related approaches and the limitations of bisimulation-related methods in domains with sparse rewards, it is still difficult for these methods to be effectively extended to environments with distractions. To alleviate these problems, in the paper, the action sequences, which contain task-intensive signals, are incorporated into representation learning. Specifically, we propose a Sequential Action--induced invariant Representation (SAR) method, in which the encoder is optimized by an auxiliary learner to only preserve the components that follow the control signals of sequential actions, so the agent can be induced to learn the robust representation against distractions. We conduct extensive experiments on the DeepMind Control suite tasks with distractions while achieving the best performance over strong baselines. We also demonstrate the effectiveness of our method at disregarding task-irrelevant information by deploying SAR to real-world CARLA-based autonomous driving with natural distractions. Finally, we provide the analysis results of generalization drawn from the generalization decay and t-SNE visualization. Code and demo videos are available at https://github.com/DMU-XMU/SAR.git.

{{</citation>}}


### (85/120) Improving Machine Learning Robustness via Adversarial Training (Long Dang et al., 2023)

{{<citation>}}

Long Dang, Thushari Hapuarachchi, Kaiqi Xiong, Jing Lin. (2023)  
**Improving Machine Learning Robustness via Adversarial Training**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2309.12593v1)  

---


**ABSTRACT**  
As Machine Learning (ML) is increasingly used in solving various tasks in real-world applications, it is crucial to ensure that ML algorithms are robust to any potential worst-case noises, adversarial attacks, and highly unusual situations when they are designed. Studying ML robustness will significantly help in the design of ML algorithms. In this paper, we investigate ML robustness using adversarial training in centralized and decentralized environments, where ML training and testing are conducted in one or multiple computers. In the centralized environment, we achieve a test accuracy of 65.41% and 83.0% when classifying adversarial examples generated by Fast Gradient Sign Method and DeepFool, respectively. Comparing to existing studies, these results demonstrate an improvement of 18.41% for FGSM and 47% for DeepFool. In the decentralized environment, we study Federated learning (FL) robustness by using adversarial training with independent and identically distributed (IID) and non-IID data, respectively, where CIFAR-10 is used in this research. In the IID data case, our experimental results demonstrate that we can achieve such a robust accuracy that it is comparable to the one obtained in the centralized environment. Moreover, in the non-IID data case, the natural accuracy drops from 66.23% to 57.82%, and the robust accuracy decreases by 25% and 23.4% in C&W and Projected Gradient Descent (PGD) attacks, compared to the IID data case, respectively. We further propose an IID data-sharing approach, which allows for increasing the natural accuracy to 85.04% and the robust accuracy from 57% to 72% in C&W attacks and from 59% to 67% in PGD attacks.

{{</citation>}}


### (86/120) SPION: Layer-Wise Sparse Training of Transformer via Convolutional Flood Filling (Bokyeong Yoon et al., 2023)

{{<citation>}}

Bokyeong Yoon, Yoonsang Han, Gordon Euhyun Moon. (2023)  
**SPION: Layer-Wise Sparse Training of Transformer via Convolutional Flood Filling**  

---
Primary Category: cs.LG  
Categories: cs-DC, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.12578v1)  

---


**ABSTRACT**  
Sparsifying the Transformer has garnered considerable interest, as training the Transformer is very computationally demanding. Prior efforts to sparsify the Transformer have either used a fixed pattern or data-driven approach to reduce the number of operations involving the computation of multi-head attention, which is the main bottleneck of the Transformer. However, existing methods suffer from inevitable problems, such as the potential loss of essential sequence features due to the uniform fixed pattern applied across all layers, and an increase in the model size resulting from the use of additional parameters to learn sparsity patterns in attention operations. In this paper, we propose a novel sparsification scheme for the Transformer that integrates convolution filters and the flood filling method to efficiently capture the layer-wise sparse pattern in attention operations. Our sparsification approach reduces the computational complexity and memory footprint of the Transformer during training. Efficient implementations of the layer-wise sparsified attention algorithm on GPUs are developed, demonstrating a new SPION that achieves up to 3.08X speedup over existing state-of-the-art sparse Transformer models, with better evaluation quality.

{{</citation>}}


## cs.IR (3)



### (87/120) American Family Cohort, a data resource description (Deepa Balraj et al., 2023)

{{<citation>}}

Deepa Balraj, Ayin Vala, Shiying Hao, Melanie Philofsky, Anna Tsvetkova, Elena Trach, Shravani Priya Narra, Oleg Zhuk, Mary Shamkhorskaya, Jim Singer, Joseph Mesterhazy, Somalee Datta, Isabella Chu, David Rehkopf. (2023)  
**American Family Cohort, a data resource description**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Clinical  
[Paper Link](http://arxiv.org/abs/2309.13175v1)  

---


**ABSTRACT**  
This manuscript is a research resource description and presents a large and novel Electronic Health Records (EHR) data resource, American Family Cohort (AFC). The AFC data is derived from Centers for Medicare and Medicaid Services (CMS) certified American Board of Family Medicine (ABFM) PRIME registry. The PRIME registry is the largest national Qualified Clinical Data Registry (QCDR) for Primary Care. The data is converted to a popular common data model, the Observational Health Data Sciences and Informatics (OHDSI) Observational Medical Outcomes Partnership (OMOP) Common Data Model (CDM).   The resource presents approximately 90 million encounters for 7.5 million patients. All 100% of the patients present age, gender, and address information, and 73% report race. Nealy 93% of patients have lab data in LOINC, 86% have medication data in RxNorm, 93% have diagnosis in SNOWMED and ICD, 81% have procedures in HCPCS or CPT, and 61% have insurance information. The richness, breadth, and diversity of this research accessible and research ready data is expected to accelerate observational studies in many diverse areas. We expect this resource to facilitate research in many years to come.

{{</citation>}}


### (88/120) Diffusion Augmentation for Sequential Recommendation (Qidong Liu et al., 2023)

{{<citation>}}

Qidong Liu, Fan Yan, Xiangyu Zhao, Zhaocheng Du, Huifeng Guo, Ruiming Tang, Feng Tian. (2023)  
**Diffusion Augmentation for Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.12858v1)  

---


**ABSTRACT**  
Sequential recommendation (SRS) has become the technical foundation in many applications recently, which aims to recommend the next item based on the user's historical interactions. However, sequential recommendation often faces the problem of data sparsity, which widely exists in recommender systems. Besides, most users only interact with a few items, but existing SRS models often underperform these users. Such a problem, named the long-tail user problem, is still to be resolved. Data augmentation is a distinct way to alleviate these two problems, but they often need fabricated training strategies or are hindered by poor-quality generated interactions. To address these problems, we propose a Diffusion Augmentation for Sequential Recommendation (DiffuASR) for a higher quality generation. The augmented dataset by DiffuASR can be used to train the sequential recommendation models directly, free from complex training procedures. To make the best of the generation ability of the diffusion model, we first propose a diffusion-based pseudo sequence generation framework to fill the gap between image and sequence generation. Then, a sequential U-Net is designed to adapt the diffusion noise prediction model U-Net to the discrete sequence generation task. At last, we develop two guide strategies to assimilate the preference between generated and origin sequences. To validate the proposed DiffuASR, we conduct extensive experiments on three real-world datasets with three sequential recommendation models. The experimental results illustrate the effectiveness of DiffuASR. As far as we know, DiffuASR is one pioneer that introduce the diffusion model to the recommendation.

{{</citation>}}


### (89/120) KuaiSim: A Comprehensive Simulator for Recommender Systems (Kesen Zhao et al., 2023)

{{<citation>}}

Kesen Zhao, Shuchang Liu, Qingpeng Cai, Xiangyu Zhao, Ziru Liu, Dong Zheng, Peng Jiang, Kun Gai. (2023)  
**KuaiSim: A Comprehensive Simulator for Recommender Systems**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.12645v1)  

---


**ABSTRACT**  
Reinforcement Learning (RL)-based recommender systems (RSs) have garnered considerable attention due to their ability to learn optimal recommendation policies and maximize long-term user rewards. However, deploying RL models directly in online environments and generating authentic data through A/B tests can pose challenges and require substantial resources. Simulators offer an alternative approach by providing training and evaluation environments for RS models, reducing reliance on real-world data. Existing simulators have shown promising results but also have limitations such as simplified user feedback, lacking consistency with real-world data, the challenge of simulator evaluation, and difficulties in migration and expansion across RSs. To address these challenges, we propose KuaiSim, a comprehensive user environment that provides user feedback with multi-behavior and cross-session responses. The resulting simulator can support three levels of recommendation problems: the request level list-wise recommendation task, the whole-session level sequential recommendation task, and the cross-session level retention optimization task. For each task, KuaiSim also provides evaluation protocols and baseline recommendation algorithms that further serve as benchmarks for future research. We also restructure existing competitive simulators on the KuaiRand Dataset and compare them against KuaiSim to future assess their performance and behavioral differences. Furthermore, to showcase KuaiSim's flexibility in accommodating different datasets, we demonstrate its versatility and robustness when deploying it on the ML-1m dataset.

{{</citation>}}


## cs.CR (2)



### (90/120) Investigating Efficient Deep Learning Architectures For Side-Channel Attacks on AES (Yohaï-Eliel Berreby et al., 2023)

{{<citation>}}

Yohaï-Eliel Berreby, Laurent Sauvage. (2023)  
**Investigating Efficient Deep Learning Architectures For Side-Channel Attacks on AES**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.13170v1)  

---


**ABSTRACT**  
Over the past few years, deep learning has been getting progressively more popular for the exploitation of side-channel vulnerabilities in embedded cryptographic applications, as it offers advantages in terms of the amount of attack traces required for effective key recovery. A number of effective attacks using neural networks have already been published, but reducing their cost in terms of the amount of computing resources and data required is an ever-present goal, which we pursue in this work. We focus on the ANSSI Side-Channel Attack Database (ASCAD), and produce a JAX-based framework for deep-learning-based SCA, with which we reproduce a selection of previous results and build upon them in an attempt to improve their performance. We also investigate the effectiveness of various Transformer-based models.

{{</citation>}}


### (91/120) A New Security Threat in MCUs -- SoC-wide timing side channels and how to find them (Johannes Müller et al., 2023)

{{<citation>}}

Johannes Müller, Anna Lena Duque Antón, Lucas Deutschmann, Dino Mehmedagić, Mohammad Rahmani Fadiheh, Dominik Stoffel, Wolfgang Kunz. (2023)  
**A New Security Threat in MCUs -- SoC-wide timing side channels and how to find them**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.12925v1)  

---


**ABSTRACT**  
Microarchitectural timing side channels have been thoroughly investigated as a security threat in hardware designs featuring shared buffers (e.g., caches) and/or parallelism between attacker and victim task execution. Contradicting common intuitions, recent activities demonstrate, however, that this threat is real also in microcontroller SoCs without such features. In this paper, we describe SoC-wide timing side channels previously neglected by security analysis and present a new formal method to close this gap. In a case study with the RISC-V Pulpissimo SoC platform, our method found a vulnerability to a so far unknown attack variant that allows an attacker to obtain information about a victim's memory access behavior. After implementing a conservative fix, we were able to verify that the SoC is now secure w.r.t. timing side channels.

{{</citation>}}


## q-bio.BM (1)



### (92/120) AntiBARTy Diffusion for Property Guided Antibody Design (Jordan Venderley, 2023)

{{<citation>}}

Jordan Venderley. (2023)  
**AntiBARTy Diffusion for Property Guided Antibody Design**  

---
Primary Category: q-bio.BM  
Categories: cs-LG, q-bio-BM, q-bio.BM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.13129v1)  

---


**ABSTRACT**  
Over the past decade, antibodies have steadily grown in therapeutic importance thanks to their high specificity and low risk of adverse effects compared to other drug modalities. While traditional antibody discovery is primarily wet lab driven, the rapid improvement of ML-based generative modeling has made in-silico approaches an increasingly viable route for discovery and engineering. To this end, we train an antibody-specific language model, AntiBARTy, based on BART (Bidirectional and Auto-Regressive Transformer) and use its latent space to train a property-conditional diffusion model for guided IgG de novo design. As a test case, we show that we can effectively generate novel antibodies with improved in-silico solubility while maintaining antibody validity and controlling sequence diversity.

{{</citation>}}


## eess.SY (1)



### (93/120) Modelling, Simulation, and Control of a Flexible Space Launch Vehicle (Muhammad Abdullah Aamer et al., 2023)

{{<citation>}}

Muhammad Abdullah Aamer, Qurat Ul Ain, Ushbah Kaleem, Hafiz Zeeshan Iqbal Khan, Jamshed Riaz. (2023)  
**Modelling, Simulation, and Control of a Flexible Space Launch Vehicle**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: Falcon  
[Paper Link](http://arxiv.org/abs/2309.13032v1)  

---


**ABSTRACT**  
Modern Space Launch Vehicles (SLVs), being slender in shape and due to the use of lightweight materials, are generally flexible in nature. This structural flexibility, when coupled with sensor and actuator dynamics, can adversely affect the control of SLV, which may lead to vehicle instability and, in the worst-case scenario, to structural failure. This work focuses on modelling and simulation of rigid and flexible dynamics of an SLV and its interactions with the control system. SpaceX's Falcon 9 has been selected for this study. The flexible modes are calculated using modal analysis in Ansys. High-fidelity nonlinear simulation is developed which incorporates the flexible modes and their interactions with rigid degrees of freedom. Moreover, linearized models are developed for flexible body dynamics, over the complete trajectory until the first stage's separation. Using classical control methods, attitude controllers, that keep the SLV on its desired trajectory, are developed, and multiple filters are designed to suppress the interactions of flexible dynamics. The designed controllers along with filters are implemented in the nonlinear simulation. Furthermore, to demonstrate the robustness of designed controllers, Monte-Carlo simulations are carried out and results are presented.

{{</citation>}}


## cs.CE (1)



### (94/120) Differential Evolution Algorithm Based Hyperparameter Selection of Gated Recurrent Unit for Electrical Load Forecasting (Anuvab Sen et al., 2023)

{{<citation>}}

Anuvab Sen, Vedica Gupta, Chi Tang. (2023)  
**Differential Evolution Algorithm Based Hyperparameter Selection of Gated Recurrent Unit for Electrical Load Forecasting**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.13019v1)  

---


**ABSTRACT**  
Accurate load forecasting remains a formidable challenge in numerous sectors, given the intricate dynamics of dynamic power systems, which often defy conventional statistical models. As a response, time-series methodologies like ARIMA and sophisticated deep learning techniques such as Artificial Neural Networks (ANN) and Long Short-Term Memory (LSTM) networks have demonstrated their mettle by achieving enhanced predictive performance. In our investigation, we delve into the efficacy of the relatively recent Gated Recurrent Network (GRU) model within the context of load forecasting. GRU models are garnering attention due to their inherent capacity to adeptly capture and model temporal dependencies within data streams. Our methodology entails harnessing the power of Differential Evolution, a versatile optimization technique renowned for its prowess in delivering scalable, robust, and globally optimal solutions, especially in scenarios involving non-differentiable, multi-objective, or constrained optimization challenges. Through rigorous analysis, we undertake a comparative assessment of the proposed Gated Recurrent Network model, collaboratively fused with various metaheuristic algorithms, evaluating their performance by leveraging established numerical benchmarks such as Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE). Our empirical investigations are conducted using power load data originating from the Ontario province, Canada. Our research findings cast a spotlight on the remarkable potential of metaheuristic-augmented Gated Recurrent Network models in substantially augmenting load forecasting precision, offering tailored, optimal hyperparameter configurations uniquely suited to each model's performance characteristics.

{{</citation>}}


## eess.AS (8)



### (95/120) Dynamic ASR Pathways: An Adaptive Masking Approach Towards Efficient Pruning of A Multilingual ASR Model (Jiamin Xie et al., 2023)

{{<citation>}}

Jiamin Xie, Ke Li, Jinxi Guo, Andros Tjandra, Yuan Shangguan, Leda Sari, Chunyang Wu, Junteng Jia, Jay Mahadeokar, Ozlem Kalinli. (2023)  
**Dynamic ASR Pathways: An Adaptive Masking Approach Towards Efficient Pruning of A Multilingual ASR Model**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Multilingual, Pruning  
[Paper Link](http://arxiv.org/abs/2309.13018v1)  

---


**ABSTRACT**  
Neural network pruning offers an effective method for compressing a multilingual automatic speech recognition (ASR) model with minimal performance loss. However, it entails several rounds of pruning and re-training needed to be run for each language. In this work, we propose the use of an adaptive masking approach in two scenarios for pruning a multilingual ASR model efficiently, each resulting in sparse monolingual models or a sparse multilingual model (named as Dynamic ASR Pathways). Our approach dynamically adapts the sub-network, avoiding premature decisions about a fixed sub-network structure. We show that our approach outperforms existing pruning methods when targeting sparse monolingual models. Further, we illustrate that Dynamic ASR Pathways jointly discovers and trains better sub-networks (pathways) of a single multilingual model by adapting from different sub-network initializations, thereby reducing the need for language-specific pruning.

{{</citation>}}


### (96/120) Importance of Smoothness Induced by Optimizers in FL4ASR: Towards Understanding Federated Learning for End-to-End ASR (Sheikh Shams Azam et al., 2023)

{{<citation>}}

Sheikh Shams Azam, Tatiana Likhomanenko, Martin Pelikan, Jan "Honza" Silovsky. (2023)  
**Importance of Smoothness Induced by Optimizers in FL4ASR: Towards Understanding Federated Learning for End-to-End ASR**  

---
Primary Category: eess.AS  
Categories: cs-DC, cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.13102v1)  

---


**ABSTRACT**  
In this paper, we start by training End-to-End Automatic Speech Recognition (ASR) models using Federated Learning (FL) and examining the fundamental considerations that can be pivotal in minimizing the performance gap in terms of word error rate between models trained using FL versus their centralized counterpart. Specifically, we study the effect of (i) adaptive optimizers, (ii) loss characteristics via altering Connectionist Temporal Classification (CTC) weight, (iii) model initialization through seed start, (iv) carrying over modeling setup from experiences in centralized training to FL, e.g., pre-layer or post-layer normalization, and (v) FL-specific hyperparameters, such as number of local epochs, client sampling size, and learning rate scheduler, specifically for ASR under heterogeneous data distribution. We shed light on how some optimizers work better than others via inducing smoothness. We also summarize the applicability of algorithms, trends, and propose best practices from prior works in FL (in general) toward End-to-End ASR models.

{{</citation>}}


### (97/120) Massive End-to-end Models for Short Search Queries (Weiran Wang et al., 2023)

{{<citation>}}

Weiran Wang, Rohit Prabhavalkar, Dongseong Hwang, Qiujia Li, Khe Chai Sim, Bo Li, James Qin, Xingyu Cai, Adam Stooke, Zhong Meng, CJ Zheng, Yanzhang He, Tara Sainath, Pedro Moreno Mengibar. (2023)  
**Massive End-to-end Models for Short Search Queries**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.12963v1)  

---


**ABSTRACT**  
In this work, we investigate two popular end-to-end automatic speech recognition (ASR) models, namely Connectionist Temporal Classification (CTC) and RNN-Transducer (RNN-T), for offline recognition of voice search queries, with up to 2B model parameters. The encoders of our models use the neural architecture of Google's universal speech model (USM), with additional funnel pooling layers to significantly reduce the frame rate and speed up training and inference. We perform extensive studies on vocabulary size, time reduction strategy, and its generalization performance on long-form test sets. Despite the speculation that, as the model size increases, CTC can be as good as RNN-T which builds label dependency into the prediction, we observe that a 900M RNN-T clearly outperforms a 1.8B CTC and is more tolerant to severe time reduction, although the WER gap can be largely removed by LM shallow fusion.

{{</citation>}}


### (98/120) VIC-KD: Variance-Invariance-Covariance Knowledge Distillation to Make Keyword Spotting More Robust Against Adversarial Attacks (Heitor R. Guimarães et al., 2023)

{{<citation>}}

Heitor R. Guimarães, Arthur Pimentel, Anderson Avila, Tiago H. Falk. (2023)  
**VIC-KD: Variance-Invariance-Covariance Knowledge Distillation to Make Keyword Spotting More Robust Against Adversarial Attacks**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Adversarial Attack, Google, Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2309.12914v1)  

---


**ABSTRACT**  
Keyword spotting (KWS) refers to the task of identifying a set of predefined words in audio streams. With the advances seen recently with deep neural networks, it has become a popular technology to activate and control small devices, such as voice assistants. Relying on such models for edge devices, however, can be challenging due to hardware constraints. Moreover, as adversarial attacks have increased against voice-based technologies, developing solutions robust to such attacks has become crucial. In this work, we propose VIC-KD, a robust distillation recipe for model compression and adversarial robustness. Using self-supervised speech representations, we show that imposing geometric priors to the latent representations of both Teacher and Student models leads to more robust target models. Experiments on the Google Speech Commands datasets show that the proposed methodology improves upon current state-of-the-art robust distillation methods, such as ARD and RSLAD, by 12% and 8% in robust accuracy, respectively.

{{</citation>}}


### (99/120) DurIAN-E: Duration Informed Attention Network For Expressive Text-to-Speech Synthesis (Yu Gu et al., 2023)

{{<citation>}}

Yu Gu, Yianrao Bian, Guangzhi Lei, Chao Weng, Dan Su. (2023)  
**DurIAN-E: Duration Informed Attention Network For Expressive Text-to-Speech Synthesis**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: AI, Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2309.12792v1)  

---


**ABSTRACT**  
This paper introduces an improved duration informed attention neural network (DurIAN-E) for expressive and high-fidelity text-to-speech (TTS) synthesis. Inherited from the original DurIAN model, an auto-regressive model structure in which the alignments between the input linguistic information and the output acoustic features are inferred from a duration model is adopted. Meanwhile the proposed DurIAN-E utilizes multiple stacked SwishRNN-based Transformer blocks as linguistic encoders. Style-Adaptive Instance Normalization (SAIN) layers are exploited into frame-level encoders to improve the modeling ability of expressiveness. A denoiser incorporating both denoising diffusion probabilistic model (DDPM) for mel-spectrograms and SAIN modules is conducted to further improve the synthetic speech quality and expressiveness. Experimental results prove that the proposed expressive TTS model in this paper can achieve better performance than the state-of-the-art approaches in both subjective mean opinion score (MOS) and preference tests.

{{</citation>}}


### (100/120) Reduce, Reuse, Recycle: Is Perturbed Data better than Other Language augmentation for Low Resource Self-Supervised Speech Models (Asad Ullah et al., 2023)

{{<citation>}}

Asad Ullah, Alessandro Ragano, Andrew Hines. (2023)  
**Reduce, Reuse, Recycle: Is Perturbed Data better than Other Language augmentation for Low Resource Self-Supervised Speech Models**  

---
Primary Category: eess.AS  
Categories: cs-CL, cs-SD, eess-AS, eess.AS  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.12763v1)  

---


**ABSTRACT**  
Self-supervised representation learning (SSRL) has improved the performance on downstream phoneme recognition versus supervised models. Training SSRL models requires a large amount of pre-training data and this poses a challenge for low resource languages. A common approach is transferring knowledge from other languages. Instead, we propose to use audio augmentation to pre-train SSRL models in a low resource condition and evaluate phoneme recognition as downstream task. We performed a systematic comparison of augmentation techniques, namely: pitch variation, noise addition, accented target-language speech and other language speech. We found combined augmentations (noise/pitch) was the best augmentation strategy outperforming accent and language knowledge transfer. We compared the performance with various quantities and types of pre-training data. We examined the scaling factor of augmented data to achieve equivalent performance to models pre-trained with target domain speech. Our findings suggest that for resource constrained languages, in-domain synthetic augmentation can outperform knowledge transfer from accented or other language speech.

{{</citation>}}


### (101/120) Unsupervised Representations Improve Supervised Learning in Speech Emotion Recognition (Amirali Soltani Tehrani et al., 2023)

{{<citation>}}

Amirali Soltani Tehrani, Niloufar Faridani, Ramin Toosi. (2023)  
**Unsupervised Representations Improve Supervised Learning in Speech Emotion Recognition**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2309.12714v1)  

---


**ABSTRACT**  
Speech Emotion Recognition (SER) plays a pivotal role in enhancing human-computer interaction by enabling a deeper understanding of emotional states across a wide range of applications, contributing to more empathetic and effective communication. This study proposes an innovative approach that integrates self-supervised feature extraction with supervised classification for emotion recognition from small audio segments. In the preprocessing step, to eliminate the need of crafting audio features, we employed a self-supervised feature extractor, based on the Wav2Vec model, to capture acoustic features from audio data. Then, the output featuremaps of the preprocessing step are fed to a custom designed Convolutional Neural Network (CNN)-based model to perform emotion classification. Utilizing the ShEMO dataset as our testing ground, the proposed method surpasses two baseline methods, i.e. support vector machine classifier and transfer learning of a pretrained CNN. comparing the propose method to the state-of-the-art methods in SER task indicates the superiority of the proposed method. Our findings underscore the pivotal role of deep unsupervised feature learning in elevating the landscape of SER, offering enhanced emotional comprehension in the realm of human-computer interactions.

{{</citation>}}


### (102/120) Big model only for hard audios: Sample dependent Whisper model selection for efficient inferences (Hugo Malard et al., 2023)

{{<citation>}}

Hugo Malard, Salah Zaiem, Robin Algayres. (2023)  
**Big model only for hard audios: Sample dependent Whisper model selection for efficient inferences**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.12712v1)  

---


**ABSTRACT**  
Recent progress in Automatic Speech Recognition (ASR) has been coupled with a substantial increase in the model sizes, which may now contain billions of parameters, leading to slow inferences even with adapted hardware. In this context, several ASR models exist in various sizes, with different inference costs leading to different performance levels. Based on the observation that smaller models perform optimally on large parts of testing corpora, we propose to train a decision module, that would allow, given an audio sample, to use the smallest sufficient model leading to a good transcription. We apply our approach to two Whisper models with different sizes. By keeping the decision process computationally efficient, we build a decision module that allows substantial computational savings with reduced performance drops.

{{</citation>}}


## eess.IV (1)



### (103/120) Performance Analysis of UNet and Variants for Medical Image Segmentation (Walid Ehab et al., 2023)

{{<citation>}}

Walid Ehab, Yongmin Li. (2023)  
**Performance Analysis of UNet and Variants for Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.13013v1)  

---


**ABSTRACT**  
Medical imaging plays a crucial role in modern healthcare by providing non-invasive visualisation of internal structures and abnormalities, enabling early disease detection, accurate diagnosis, and treatment planning. This study aims to explore the application of deep learning models, particularly focusing on the UNet architecture and its variants, in medical image segmentation. We seek to evaluate the performance of these models across various challenging medical image segmentation tasks, addressing issues such as image normalization, resizing, architecture choices, loss function design, and hyperparameter tuning. The findings reveal that the standard UNet, when extended with a deep network layer, is a proficient medical image segmentation model, while the Res-UNet and Attention Res-UNet architectures demonstrate smoother convergence and superior performance, particularly when handling fine image details. The study also addresses the challenge of high class imbalance through careful preprocessing and loss function definitions. We anticipate that the results of this study will provide useful insights for researchers seeking to apply these models to new medical imaging problems and offer guidance and best practices for their implementation.

{{</citation>}}


## cs.SE (3)



### (104/120) Smart Fuzzing of 5G Wireless Software Implementation (Huan Wu et al., 2023)

{{<citation>}}

Huan Wu, Brian Fang, Fei Xie. (2023)  
**Smart Fuzzing of 5G Wireless Software Implementation**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2309.12994v1)  

---


**ABSTRACT**  
In this paper, we introduce a comprehensive approach to bolstering the security, reliability, and comprehensibility of OpenAirInterface5G (OAI5G), an open-source software framework for the exploration, development, and testing of 5G wireless communication systems. Firstly, we employ AFL++, a powerful fuzzing tool, to fuzzy-test OAI5G with respect to its configuration files rigorously. This extensive testing process helps identify errors, defects, and security vulnerabilities that may evade conventional testing methods. Secondly, we harness the capabilities of Large Language Models such as Google Bard to automatically decipher and document the meanings of parameters within the OAI5G codebase that are used in fuzzing. This automated parameter interpretation streamlines subsequent analyses and facilitates more informed decision-making. Together, these two techniques contribute to fortifying the OAI5G system, making it more robust, secure, and understandable for developers and analysts alike.

{{</citation>}}


### (105/120) Trusta: Reasoning about Assurance Cases with Formal Methods and Large Language Models (Zezhong Chen et al., 2023)

{{<citation>}}

Zezhong Chen, Yuxin Deng, Wenjie Du. (2023)  
**Trusta: Reasoning about Assurance Cases with Formal Methods and Large Language Models**  

---
Primary Category: cs.SE  
Categories: D-2-1, cs-AI, cs-SE, cs.SE  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Language Model, PaLM, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.12941v1)  

---


**ABSTRACT**  
Assurance cases can be used to argue for the safety of products in safety engineering. In safety-critical areas, the construction of assurance cases is indispensable. Trustworthiness Derivation Trees (TDTs) enhance assurance cases by incorporating formal methods, rendering it possible for automatic reasoning about assurance cases. We present Trustworthiness Derivation Tree Analyzer (Trusta), a desktop application designed to automatically construct and verify TDTs. The tool has a built-in Prolog interpreter in its backend, and is supported by the constraint solvers Z3 and MONA. Therefore, it can solve constraints about logical formulas involving arithmetic, sets, Horn clauses etc. Trusta also utilizes large language models to make the creation and evaluation of assurance cases more convenient. It allows for interactive human examination and modification. We evaluated top language models like ChatGPT-3.5, ChatGPT-4, and PaLM 2 for generating assurance cases. Our tests showed a 50%-80% similarity between machine-generated and human-created cases. In addition, Trusta can extract formal constraints from text in natural languages, facilitating an easier interpretation and validation process. This extraction is subject to human review and correction, blending the best of automated efficiency with human insight. To our knowledge, this marks the first integration of large language models in automatic creating and reasoning about assurance cases, bringing a novel approach to a traditional challenge. Through several industrial case studies, Trusta has proven to quickly find some subtle issues that are typically missed in manual inspection, demonstrating its practical value in enhancing the assurance case development process.

{{</citation>}}


### (106/120) Towards an MLOps Architecture for XAI in Industrial Applications (Leonhard Faubel et al., 2023)

{{<citation>}}

Leonhard Faubel, Thomas Woudsma, Leila Methnani, Amir Ghorbani Ghezeljhemeidan, Fabian Buelow, Klaus Schmid, Willem D. van Driel, Benjamin Kloepper, Andreas Theodorou, Mohsen Nosratinia, Magnus Bång. (2023)  
**Towards an MLOps Architecture for XAI in Industrial Applications**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.12756v1)  

---


**ABSTRACT**  
Machine learning (ML) has become a popular tool in the industrial sector as it helps to improve operations, increase efficiency, and reduce costs. However, deploying and managing ML models in production environments can be complex. This is where Machine Learning Operations (MLOps) comes in. MLOps aims to streamline this deployment and management process. One of the remaining MLOps challenges is the need for explanations. These explanations are essential for understanding how ML models reason, which is key to trust and acceptance. Better identification of errors and improved model accuracy are only two resulting advantages. An often neglected fact is that deployed models are bypassed in practice when accuracy and especially explainability do not meet user expectations. We developed a novel MLOps software architecture to address the challenge of integrating explanations and feedback capabilities into the ML development and deployment processes. In the project EXPLAIN, our architecture is implemented in a series of industrial use cases. The proposed MLOps software architecture has several advantages. It provides an efficient way to manage ML models in production environments. Further, it allows for integrating explanations into the development and deployment processes.

{{</citation>}}


## cs.MA (1)



### (107/120) Boosting Studies of Multi-Agent Reinforcement Learning on Google Research Football Environment: the Past, Present, and Future (Yan Song et al., 2023)

{{<citation>}}

Yan Song, He Jiang, Haifeng Zhang, Zheng Tian, Weinan Zhang, Jun Wang. (2023)  
**Boosting Studies of Multi-Agent Reinforcement Learning on Google Research Football Environment: the Past, Present, and Future**  

---
Primary Category: cs.MA  
Categories: cs-MA, cs.MA  
Keywords: AI, Google, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.12951v1)  

---


**ABSTRACT**  
Even though Google Research Football (GRF) was initially benchmarked and studied as a single-agent environment in its original paper, recent years have witnessed an increasing focus on its multi-agent nature by researchers utilizing it as a testbed for Multi-Agent Reinforcement Learning (MARL). However, the absence of standardized environment settings and unified evaluation metrics for multi-agent scenarios hampers the consistent understanding of various studies. Furthermore, the challenging 5-vs-5 and 11-vs-11 full-game scenarios have received limited thorough examination due to their substantial training complexities. To address these gaps, this paper extends the original environment by not only standardizing the environment settings and benchmarking cooperative learning algorithms across different scenarios, including the most challenging full-game scenarios, but also by discussing approaches to enhance football AI from diverse perspectives and introducing related research tools. Specifically, we provide a distributed and asynchronous population-based self-play framework with diverse pre-trained policies for faster training, two football-specific analytical tools for deeper investigation, and an online leaderboard for broader evaluation. The overall expectation of this work is to advance the study of Multi-Agent Reinforcement Learning on Google Research Football environment, with the ultimate goal of benefiting real-world sports beyond virtual games.

{{</citation>}}


## cs.NE (2)



### (108/120) Emergent mechanisms for long timescales depend on training curriculum and affect performance in memory tasks (Sina Khajehabdollahi et al., 2023)

{{<citation>}}

Sina Khajehabdollahi, Roxana Zeraati, Emmanouil Giannakakis, Tim Jakob Schäfer, Georg Martius, Anna Levina. (2023)  
**Emergent mechanisms for long timescales depend on training curriculum and affect performance in memory tasks**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE, q-bio-NC  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.12927v1)  

---


**ABSTRACT**  
Recurrent neural networks (RNNs) in the brain and in silico excel at solving tasks with intricate temporal dependencies. Long timescales required for solving such tasks can arise from properties of individual neurons (single-neuron timescale, $\tau$, e.g., membrane time constant in biological neurons) or recurrent interactions among them (network-mediated timescale). However, the contribution of each mechanism for optimally solving memory-dependent tasks remains poorly understood. Here, we train RNNs to solve $N$-parity and $N$-delayed match-to-sample tasks with increasing memory requirements controlled by $N$ by simultaneously optimizing recurrent weights and $\tau$s. We find that for both tasks RNNs develop longer timescales with increasing $N$, but depending on the learning objective, they use different mechanisms. Two distinct curricula define learning objectives: sequential learning of a single-$N$ (single-head) or simultaneous learning of multiple $N$s (multi-head). Single-head networks increase their $\tau$ with $N$ and are able to solve tasks for large $N$, but they suffer from catastrophic forgetting. However, multi-head networks, which are explicitly required to hold multiple concurrent memories, keep $\tau$ constant and develop longer timescales through recurrent connectivity. Moreover, we show that the multi-head curriculum increases training speed and network stability to ablations and perturbations, and allows RNNs to generalize better to tasks beyond their training regime. This curriculum also significantly improves training GRUs and LSTMs for large-$N$ tasks. Our results suggest that adapting timescales to task requirements via recurrent interactions allows learning more complex objectives and improves the RNN's performance.

{{</citation>}}


### (109/120) ThinResNet: A New Baseline for Structured Convolutional Networks Pruning (Hugo Tessier et al., 2023)

{{<citation>}}

Hugo Tessier, Ghouti Boukli Hacene, Vincent Gripon. (2023)  
**ThinResNet: A New Baseline for Structured Convolutional Networks Pruning**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2309.12854v1)  

---


**ABSTRACT**  
Pruning is a compression method which aims to improve the efficiency of neural networks by reducing their number of parameters while maintaining a good performance, thus enhancing the performance-to-cost ratio in nontrivial ways. Of particular interest are structured pruning techniques, in which whole portions of parameters are removed altogether, resulting in easier to leverage shrunk architectures. Since its growth in popularity in the recent years, pruning gave birth to countless papers and contributions, resulting first in critical inconsistencies in the way results are compared, and then to a collective effort to establish standardized benchmarks. However, said benchmarks are based on training practices that date from several years ago and do not align with current practices. In this work, we verify how results in the recent literature of pruning hold up against networks that underwent both state-of-the-art training methods and trivial model scaling. We find that the latter clearly and utterly outperform all the literature we compared to, proving that updating standard pruning benchmarks and re-evaluating classical methods in their light is an absolute necessity. We thus introduce a new challenging baseline to compare structured pruning to: ThinResNet.

{{</citation>}}


## cs.AR (1)



### (110/120) AxOCS: Scaling FPGA-based Approximate Operators using Configuration Supersampling (Siva Satyendra Sahoo et al., 2023)

{{<citation>}}

Siva Satyendra Sahoo, Salim Ullah, Soumyo Bhattacharjee, Akash Kumar. (2023)  
**AxOCS: Scaling FPGA-based Approximate Operators using Configuration Supersampling**  

---
Primary Category: cs.AR  
Categories: B-2-4; J-6; J-7; I-2-1, cs-AI, cs-AR, cs-LG, cs.AR, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.12830v1)  

---


**ABSTRACT**  
The rising usage of AI and ML-based processing across application domains has exacerbated the need for low-cost ML implementation, specifically for resource-constrained embedded systems. To this end, approximate computing, an approach that explores the power, performance, area (PPA), and behavioral accuracy (BEHAV) trade-offs, has emerged as a possible solution for implementing embedded machine learning. Due to the predominance of MAC operations in ML, designing platform-specific approximate arithmetic operators forms one of the major research problems in approximate computing. Recently there has been a rising usage of AI/ML-based design space exploration techniques for implementing approximate operators. However, most of these approaches are limited to using ML-based surrogate functions for predicting the PPA and BEHAV impact of a set of related design decisions. While this approach leverages the regression capabilities of ML methods, it does not exploit the more advanced approaches in ML. To this end, we propose AxOCS, a methodology for designing approximate arithmetic operators through ML-based supersampling. Specifically, we present a method to leverage the correlation of PPA and BEHAV metrics across operators of varying bit-widths for generating larger bit-width operators. The proposed approach involves traversing the relatively smaller design space of smaller bit-width operators and employing its associated Design-PPA-BEHAV relationship to generate initial solutions for metaheuristics-based optimization for larger operators. The experimental evaluation of AxOCS for FPGA-optimized approximate operators shows that the proposed approach significantly improves the quality-resulting hypervolume for multi-objective optimization-of 8x8 signed approximate multipliers.

{{</citation>}}


## cs.RO (4)



### (111/120) OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control (Botian Xu et al., 2023)

{{<citation>}}

Botian Xu, Feng Gao, Chao Yu, Ruize Zhang, Yi Wu, Yu Wang. (2023)  
**OmniDrones: An Efficient and Flexible Platform for Reinforcement Learning in Drone Control**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Drone, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.12825v1)  

---


**ABSTRACT**  
In this work, we introduce OmniDrones, an efficient and flexible platform tailored for reinforcement learning in drone control, built on Nvidia's Omniverse Isaac Sim. It employs a bottom-up design approach that allows users to easily design and experiment with various application scenarios on top of GPU-parallelized simulations. It also offers a range of benchmark tasks, presenting challenges ranging from single-drone hovering to over-actuated system tracking. In summary, we propose an open-sourced drone simulation platform, equipped with an extensive suite of tools for drone learning. It includes 4 drone models, 5 sensor modalities, 4 control modes, over 10 benchmark tasks, and a selection of widely used RL baselines. To showcase the capabilities of OmniDrones and to support future research, we also provide preliminary results on these benchmark tasks. We hope this platform will encourage further studies on applying RL to practical drone systems.

{{</citation>}}


### (112/120) Teacher-Student Reinforcement Learning for Mapless Navigation using a Planetary Space Rover (Anton Bjørndahl Mortensen et al., 2023)

{{<citation>}}

Anton Bjørndahl Mortensen, Emil Tribler Pedersen, Laia Vives Benedicto, Lionel Burg, Mads Rossen Madsen, Simon Bøgh. (2023)  
**Teacher-Student Reinforcement Learning for Mapless Navigation using a Planetary Space Rover**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.12807v1)  

---


**ABSTRACT**  
We address the challenge of enhancing navigation autonomy for planetary space rovers using reinforcement learning (RL). The ambition of future space missions necessitates advanced autonomous navigation capabilities for rovers to meet mission objectives. RL's potential in robotic autonomy is evident, but its reliance on simulations poses a challenge. Transferring policies to real-world scenarios often encounters the "reality gap", disrupting the transition from virtual to physical environments. The reality gap is exacerbated in the context of mapless navigation on Mars and Moon-like terrains, where unpredictable terrains and environmental factors play a significant role. Effective navigation requires a method attuned to these complexities and real-world data noise. We introduce a novel two-stage RL approach using offline noisy data. Our approach employs a teacher-student policy learning paradigm, inspired by the "learning by cheating" method. The teacher policy is trained in simulation. Subsequently, the student policy is trained on noisy data, aiming to mimic the teacher's behaviors while being more robust to real-world uncertainties. Our policies are transferred to a custom-designed rover for real-world testing. Comparative analyses between the teacher and student policies reveal that our approach offers improved behavioral performance, heightened noise resilience, and more effective sim-to-real transfer.

{{</citation>}}


### (113/120) Learning Actions and Control of Focus of Attention with a Log-Polar-like Sensor (Robin Göransson et al., 2023)

{{<citation>}}

Robin Göransson, Volker Krueger. (2023)  
**Learning Actions and Control of Focus of Attention with a Log-Polar-like Sensor**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Attention, LSTM  
[Paper Link](http://arxiv.org/abs/2309.12634v1)  

---


**ABSTRACT**  
With the long-term goal of reducing the image processing time on an autonomous mobile robot in mind we explore in this paper the use of log-polar like image data with gaze control. The gaze control is not done on the Cartesian image but on the log-polar like image data. For this we start out from the classic deep reinforcement learning approach for Atari games. We extend an A3C deep RL approach with an LSTM network, and we learn the policy for playing three Atari games and a policy for gaze control. While the Atari games already use low-resolution images of 80 by 80 pixels, we are able to further reduce the amount of image pixels by a factor of 5 without losing any gaming performance.

{{</citation>}}


### (114/120) Real-time Motion Generation and Data Augmentation for Grasping Moving Objects with Dynamic Speed and Position Changes (Kenjiro Yamamoto et al., 2023)

{{<citation>}}

Kenjiro Yamamoto, Hiroshi Ito, Hideyuki Ichiwara, Hiroki Mori, Tetsuya Ogata. (2023)  
**Real-time Motion Generation and Data Augmentation for Grasping Moving Objects with Dynamic Speed and Position Changes**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.12547v1)  

---


**ABSTRACT**  
While deep learning enables real robots to perform complex tasks had been difficult to implement in the past, the challenge is the enormous amount of trial-and-error and motion teaching in a real environment. The manipulation of moving objects, due to their dynamic properties, requires learning a wide range of factors such as the object's position, movement speed, and grasping timing. We propose a data augmentation method for enabling a robot to grasp moving objects with different speeds and grasping timings at low cost. Specifically, the robot is taught to grasp an object moving at low speed using teleoperation, and multiple data with different speeds and grasping timings are generated by down-sampling and padding the robot sensor data in the time-series direction. By learning multiple sensor data in a time series, the robot can generate motions while adjusting the grasping timing for unlearned movement speeds and sudden speed changes. We have shown using a real robot that this data augmentation method facilitates learning the relationship between object position and velocity and enables the robot to perform robust grasping motions for unlearned positions and objects with dynamically changing positions and velocities.

{{</citation>}}


## cs.GL (1)



### (115/120) Computational Natural Philosophy: A Thread from Presocratics through Turing to ChatGPT (Gordana Dodig-Crnkovic, 2023)

{{<citation>}}

Gordana Dodig-Crnkovic. (2023)  
**Computational Natural Philosophy: A Thread from Presocratics through Turing to ChatGPT**  

---
Primary Category: cs.GL  
Categories: cs-AI, cs-GL, cs.GL  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.13094v1)  

---


**ABSTRACT**  
Modern computational natural philosophy conceptualizes the universe in terms of information and computation, establishing a framework for the study of cognition and intelligence. Despite some critiques, this computational perspective has significantly influenced our understanding of the natural world, leading to the development of AI systems like ChatGPT based on deep neural networks. Advancements in this domain have been facilitated by interdisciplinary research, integrating knowledge from multiple fields to simulate complex systems. Large Language Models (LLMs), such as ChatGPT, represent this approach's capabilities, utilizing reinforcement learning with human feedback (RLHF). Current research initiatives aim to integrate neural networks with symbolic computing, introducing a new generation of hybrid computational models.

{{</citation>}}


## cs.SI (3)



### (116/120) Multi-Modal Embeddings for Isolating Cross-Platform Coordinated Information Campaigns on Social Media (Fabio Barbero et al., 2023)

{{<citation>}}

Fabio Barbero, Sander op den Camp, Kristian van Kuijk, Carlos Soto García-Delgado, Gerasimos Spanakis, Adriana Iamnitchi. (2023)  
**Multi-Modal Embeddings for Isolating Cross-Platform Coordinated Information Campaigns on Social Media**  

---
Primary Category: cs.SI  
Categories: H-3-5; H-3-1, cs-SI, cs.SI  
Keywords: Embedding, Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2309.12764v1)  

---


**ABSTRACT**  
Coordinated multi-platform information operations are implemented in a variety of contexts on social media, including state-run disinformation campaigns, marketing strategies, and social activism. Characterized by the promotion of messages via multi-platform coordination, in which multiple user accounts, within a short time, post content advancing a shared informational agenda on multiple platforms, they contribute to an already confusing and manipulated information ecosystem. To make things worse, reliable datasets that contain ground truth information about such operations are virtually nonexistent. This paper presents a multi-modal approach that identifies the social media messages potentially engaged in a coordinated information campaign across multiple platforms. Our approach incorporates textual content, temporal information and the underlying network of user and messages posted to identify groups of messages with unusual coordination patterns across multiple social media platforms. We apply our approach to content posted on four platforms related to the Syrian Civil Defence organization known as the White Helmets: Twitter, Facebook, Reddit, and YouTube. Results show that our approach identifies social media posts that link to news YouTube channels with similar factuality score, which is often an indication of coordinated operations.

{{</citation>}}


### (117/120) Coordinated Information Campaigns on Social Media: A Multifaceted Framework for Detection and Analysis (Kin Wai Ng et al., 2023)

{{<citation>}}

Kin Wai Ng, Adriana Iamnitchi. (2023)  
**Coordinated Information Campaigns on Social Media: A Multifaceted Framework for Detection and Analysis**  

---
Primary Category: cs.SI  
Categories: H-3-5; H-3-1, cs-SI, cs.SI  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2309.12729v1)  

---


**ABSTRACT**  
The prevalence of coordinated information campaigns in social media platforms has significant negative consequences across various domains, including social, political, and economic processes. This paper proposes a multifaceted framework for detecting and analysing coordinated message promotion on social media. By simultaneously considering features related to content, time, and network dimensions, our framework can capture the diverse nature of coordinated activity and identify anomalous user accounts who likely engaged in suspicious behaviour. Unlike existing solutions that rely on specific constraints, our approach is more flexible as it employs specialised components to extract the significant structures within a network and to detect the most unusual interactions. We demonstrate the effectiveness of our framework using two Twitter datasets, the Russian Internet Research Agency (IRA), and long-term discussions on Data Science topics. The results demonstrate our framework's ability to isolate unusual activity from expected normal behaviour and provide valuable insights for further qualitative investigation.

{{</citation>}}


### (118/120) User Migration across Multiple Social Media Platforms (Ujun Jeong et al., 2023)

{{<citation>}}

Ujun Jeong, Ayushi Nirmal, Kritshekhar Jha, Xu Tang, H. Russell Bernard, Huan Liu. (2023)  
**User Migration across Multiple Social Media Platforms**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2309.12613v1)  

---


**ABSTRACT**  
After Twitter's ownership change and policy shifts, many users reconsidered their go-to social media outlets and platforms like Mastodon, Bluesky, and Threads became attractive alternatives in the battle for users. Based on the data from over 16,000 users who migrated to these platforms within the first eight weeks after the launch of Threads, our study examines: (1) distinguishing attributes of Twitter users who migrated, compared to non-migrants; (2) temporal migration patterns and associated challenges for sustainable migration faced by each platform; and (3) how these new platforms are perceived in relation to Twitter. Our research proceeds in three stages. First, we examine migration from a broad perspective, not just one-to-one migration. Second, we leverage behavioral analysis to pinpoint the distinct migration pattern of each platform. Last, we employ a large language model (LLM) to discern stances towards each platform and correlate them with the platform usage. This in-depth analysis illuminates migration patterns amid competition across social media platforms.

{{</citation>}}


## quant-ph (1)



### (119/120) QAL-BP: An Augmented Lagrangian Quantum Approach for Bin Packing Problem (Lorenzo Cellini et al., 2023)

{{<citation>}}

Lorenzo Cellini, Antonio Macaluso, Michele Lombardi. (2023)  
**QAL-BP: An Augmented Lagrangian Quantum Approach for Bin Packing Problem**  

---
Primary Category: quant-ph  
Categories: cs-AI, math-OC, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.12678v1)  

---


**ABSTRACT**  
The bin packing is a well-known NP-Hard problem in the domain of artificial intelligence, posing significant challenges in finding efficient solutions. Conversely, recent advancements in quantum technologies have shown promising potential for achieving substantial computational speedup, particularly in certain problem classes, such as combinatorial optimization. In this study, we introduce QAL-BP, a novel Quadratic Unconstrained Binary Optimization (QUBO) formulation designed specifically for bin packing and suitable for quantum computation. QAL-BP utilizes the augmented Lagrangian method to incorporate the bin packing constraints into the objective function while also facilitating an analytical estimation of heuristic, but empirically robust, penalty multipliers. This approach leads to a more versatile and generalizable model that eliminates the need for empirically calculating instance-dependent Lagrangian coefficients, a requirement commonly encountered in alternative QUBO formulations for similar problems. To assess the effectiveness of our proposed approach, we conduct experiments on a set of bin-packing instances using a real Quantum Annealing device. Additionally, we compare the results with those obtained from two different classical solvers, namely simulated annealing and Gurobi. The experimental findings not only confirm the correctness of the proposed formulation but also demonstrate the potential of quantum computation in effectively solving the bin-packing problem, particularly as more reliable quantum technology becomes available.

{{</citation>}}


## cs.CY (1)



### (120/120) Before Blue Birds Became X-tinct: Understanding the Effect of Regime Change on Twitter's Advertising and Compliance of Advertising Policies (Yash Vekaria et al., 2023)

{{<citation>}}

Yash Vekaria, Zubair Shafiq, Savvas Zannettou. (2023)  
**Before Blue Birds Became X-tinct: Understanding the Effect of Regime Change on Twitter's Advertising and Compliance of Advertising Policies**  

---
Primary Category: cs.CY  
Categories: K-4-1; K-4-2; K-4-3, cs-CR, cs-CY, cs-HC, cs-SI, cs.CY  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2309.12591v1)  

---


**ABSTRACT**  
Social media platforms, including Twitter (now X), have policies in place to maintain a safe and trustworthy advertising environment. However, the extent to which these policies are adhered to and enforced remains a subject of interest and concern. We present the first large-scale audit of advertising on Twitter focusing on compliance with the platform's advertising policies, particularly those related to political and adult content. We investigate the compliance of advertisements on Twitter with the platform's stated policies and the impact of recent acquisition on the advertising activity of the platform. By analyzing 34K advertisements from ~6M tweets, collected over six months, we find evidence of widespread noncompliance with Twitter's political and adult content advertising policies suggesting a lack of effective ad content moderation. We also find that Elon Musk's acquisition of Twitter had a noticeable impact on the advertising landscape, with most existing advertisers either completely stopping their advertising activity or reducing it. Major brands decreased their advertising on Twitter, suggesting a negative immediate effect on the platform's advertising revenue. Our findings underscore the importance of external audits to monitor compliance and improve transparency in online advertising.

{{</citation>}}
