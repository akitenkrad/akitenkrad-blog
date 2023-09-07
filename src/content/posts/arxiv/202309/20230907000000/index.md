---
draft: false
title: "arXiv @ 2023.09.07"
date: 2023-09-07
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.07"
    identifier: arxiv_20230907
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (20)](#cscv-20)
- [eess.IV (3)](#eessiv-3)
- [cs.SD (1)](#cssd-1)
- [cs.NE (1)](#csne-1)
- [cs.LG (29)](#cslg-29)
- [cs.SI (1)](#cssi-1)
- [eess.AS (2)](#eessas-2)
- [cs.RO (5)](#csro-5)
- [cs.CL (20)](#cscl-20)
- [cs.IR (1)](#csir-1)
- [cs.ET (1)](#cset-1)
- [cs.AI (2)](#csai-2)
- [cs.HC (2)](#cshc-2)
- [cs.CR (2)](#cscr-2)
- [cs.SE (5)](#csse-5)
- [astro-ph.EP (1)](#astro-phep-1)
- [physics.acc-ph (1)](#physicsacc-ph-1)
- [q-bio.NC (2)](#q-bionc-2)
- [stat.ML (2)](#statml-2)
- [cs.MA (1)](#csma-1)
- [cs.IT (1)](#csit-1)
- [cs.CY (5)](#cscy-5)
- [math.HO (1)](#mathho-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.NI (1)](#csni-1)
- [cs.DB (1)](#csdb-1)
- [hep-ex (1)](#hep-ex-1)
- [math.NA (1)](#mathna-1)

## cs.CV (20)



### (1/114) Compressing Vision Transformers for Low-Resource Visual Learning (Eric Youn et al., 2023)

{{<citation>}}

Eric Youn, Sai Mitheran J, Sanjana Prabhu, Siyuan Chen. (2023)  
**Compressing Vision Transformers for Low-Resource Visual Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Low-Resource, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.02617v1)  

---


**ABSTRACT**  
Vision transformer (ViT) and its variants have swept through visual learning leaderboards and offer state-of-the-art accuracy in tasks such as image classification, object detection, and semantic segmentation by attending to different parts of the visual input and capturing long-range spatial dependencies. However, these models are large and computation-heavy. For instance, the recently proposed ViT-B model has 86M parameters making it impractical for deployment on resource-constrained devices. As a result, their deployment on mobile and edge scenarios is limited. In our work, we aim to take a step toward bringing vision transformers to the edge by utilizing popular model compression techniques such as distillation, pruning, and quantization.   Our chosen application environment is an unmanned aerial vehicle (UAV) that is battery-powered and memory-constrained, carrying a single-board computer on the scale of an NVIDIA Jetson Nano with 4GB of RAM. On the other hand, the UAV requires high accuracy close to that of state-of-the-art ViTs to ensure safe object avoidance in autonomous navigation, or correct localization of humans in search-and-rescue. Inference latency should also be minimized given the application requirements. Hence, our target is to enable rapid inference of a vision transformer on an NVIDIA Jetson Nano (4GB) with minimal accuracy loss. This allows us to deploy ViTs on resource-constrained devices, opening up new possibilities in surveillance, environmental monitoring, etc. Our implementation is made available at https://github.com/chensy7/efficient-vit.

{{</citation>}}


### (2/114) Self-Supervised Pretraining Improves Performance and Inference Efficiency in Multiple Lung Ultrasound Interpretation Tasks (Blake VanBerlo et al., 2023)

{{<citation>}}

Blake VanBerlo, Brian Li, Jesse Hoey, Alexander Wong. (2023)  
**Self-Supervised Pretraining Improves Performance and Inference Efficiency in Multiple Lung Ultrasound Interpretation Tasks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.02596v1)  

---


**ABSTRACT**  
In this study, we investigated whether self-supervised pretraining could produce a neural network feature extractor applicable to multiple classification tasks in B-mode lung ultrasound analysis. When fine-tuning on three lung ultrasound tasks, pretrained models resulted in an improvement of the average across-task area under the receiver operating curve (AUC) by 0.032 and 0.061 on local and external test sets respectively. Compact nonlinear classifiers trained on features outputted by a single pretrained model did not improve performance across all tasks; however, they did reduce inference time by 49% compared to serial execution of separate fine-tuned models. When training using 1% of the available labels, pretrained models consistently outperformed fully supervised models, with a maximum observed test AUC increase of 0.396 for the task of view classification. Overall, the results indicate that self-supervised pretraining is useful for producing initial weights for lung ultrasound classifiers.

{{</citation>}}


### (3/114) Domain Adaptation for Efficiently Fine-tuning Vision Transformer with Encrypted Images (Teru Nagamori et al., 2023)

{{<citation>}}

Teru Nagamori, Sayaka Shiota, Hitoshi Kiya. (2023)  
**Domain Adaptation for Efficiently Fine-tuning Vision Transformer with Encrypted Images**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.02556v1)  

---


**ABSTRACT**  
In recent years, deep neural networks (DNNs) trained with transformed data have been applied to various applications such as privacy-preserving learning, access control, and adversarial defenses. However, the use of transformed data decreases the performance of models. Accordingly, in this paper, we propose a novel method for fine-tuning models with transformed images under the use of the vision transformer (ViT). The proposed domain adaptation method does not cause the accuracy degradation of models, and it is carried out on the basis of the embedding structure of ViT. In experiments, we confirmed that the proposed method prevents accuracy degradation even when using encrypted images with the CIFAR-10 and CIFAR-100 datasets.

{{</citation>}}


### (4/114) STEP -- Towards Structured Scene-Text Spotting (Sergi Garcia-Bordils et al., 2023)

{{<citation>}}

Sergi Garcia-Bordils, Dimosthenis Karatzas, Marçal Rusiñol. (2023)  
**STEP -- Towards Structured Scene-Text Spotting**  

---
Primary Category: cs.CV  
Categories: 68T01 (Primary) 68T10, 68T45, 68T05, 68T07 (Secondary), I-2-1; I-2-6; I-2-10, cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2309.02356v1)  

---


**ABSTRACT**  
We introduce the structured scene-text spotting task, which requires a scene-text OCR system to spot text in the wild according to a query regular expression. Contrary to generic scene text OCR, structured scene-text spotting seeks to dynamically condition both scene text detection and recognition on user-provided regular expressions. To tackle this task, we propose the Structured TExt sPotter (STEP), a model that exploits the provided text structure to guide the OCR process. STEP is able to deal with regular expressions that contain spaces and it is not bound to detection at the word-level granularity. Our approach enables accurate zero-shot structured text spotting in a wide variety of real-world reading scenarios and is solely trained on publicly available data. To demonstrate the effectiveness of our approach, we introduce a new challenging test dataset that contains several types of out-of-vocabulary structured text, reflecting important reading applications of fields such as prices, dates, serial numbers, license plates etc. We demonstrate that STEP can provide specialised OCR performance on demand in all tested scenarios.

{{</citation>}}


### (5/114) CIEM: Contrastive Instruction Evaluation Method for Better Instruction Tuning (Hongyu Hu et al., 2023)

{{<citation>}}

Hongyu Hu, Jiyuan Zhang, Minyi Zhao, Zhenbang Sun. (2023)  
**CIEM: Contrastive Instruction Evaluation Method for Better Instruction Tuning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.02301v1)  

---


**ABSTRACT**  
Nowadays, the research on Large Vision-Language Models (LVLMs) has been significantly promoted thanks to the success of Large Language Models (LLM). Nevertheless, these Vision-Language Models (VLMs) are suffering from the drawback of hallucination -- due to insufficient understanding of vision and language modalities, VLMs may generate incorrect perception information when doing downstream applications, for example, captioning a non-existent entity. To address the hallucination phenomenon, on the one hand, we introduce a Contrastive Instruction Evaluation Method (CIEM), which is an automatic pipeline that leverages an annotated image-text dataset coupled with an LLM to generate factual/contrastive question-answer pairs for the evaluation of the hallucination of VLMs. On the other hand, based on CIEM, we further propose a new instruction tuning method called CIT (the abbreviation of Contrastive Instruction Tuning) to alleviate the hallucination of VLMs by automatically producing high-quality factual/contrastive question-answer pairs and corresponding justifications for model tuning. Through extensive experiments on CIEM and CIT, we pinpoint the hallucination issues commonly present in existing VLMs, the disability of the current instruction-tuning dataset to handle the hallucination phenomenon and the superiority of CIT-tuned VLMs over both CIEM and public datasets.

{{</citation>}}


### (6/114) ATM: Action Temporality Modeling for Video Question Answering (Junwen Chen et al., 2023)

{{<citation>}}

Junwen Chen, Jie Zhu, Yu Kong. (2023)  
**ATM: Action Temporality Modeling for Video Question Answering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.02290v1)  

---


**ABSTRACT**  
Despite significant progress in video question answering (VideoQA), existing methods fall short of questions that require causal/temporal reasoning across frames. This can be attributed to imprecise motion representations. We introduce Action Temporality Modeling (ATM) for temporality reasoning via three-fold uniqueness: (1) rethinking the optical flow and realizing that optical flow is effective in capturing the long horizon temporality reasoning; (2) training the visual-text embedding by contrastive learning in an action-centric manner, leading to better action representations in both vision and text modalities; and (3) preventing the model from answering the question given the shuffled video in the fine-tuning stage, to avoid spurious correlation between appearance and motion and hence ensure faithful temporality reasoning. In the experiments, we show that ATM outperforms previous approaches in terms of the accuracy on multiple VideoQAs and exhibits better true temporality reasoning ability.

{{</citation>}}


### (7/114) SAM-Deblur: Let Segment Anything Boost Image Deblurring (Siwei Li et al., 2023)

{{<citation>}}

Siwei Li, Mingxuan Liu, Yating Zhang, Shu Chen, Haoxiang Li, Hong Chen, Zifei Dou. (2023)  
**SAM-Deblur: Let Segment Anything Boost Image Deblurring**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.02270v1)  

---


**ABSTRACT**  
Image deblurring is a critical task in the field of image restoration, aiming to eliminate blurring artifacts. However, the challenge of addressing non-uniform blurring leads to an ill-posed problem, which limits the generalization performance of existing deblurring models. To solve the problem, we propose a framework SAM-Deblur, integrating prior knowledge from the Segment Anything Model (SAM) into the deblurring task for the first time. In particular, SAM-Deblur is divided into three stages. First, We preprocess the blurred images, obtain image masks via SAM, and propose a mask dropout method for training to enhance model robustness. Then, to fully leverage the structural priors generated by SAM, we propose a Mask Average Pooling (MAP) unit specifically designed to average SAM-generated segmented areas, serving as a plug-and-play component which can be seamlessly integrated into existing deblurring networks. Finally, we feed the fused features generated by the MAP Unit into the deblurring model to obtain a sharp image. Experimental results on the RealBlurJ, ReloBlur, and REDS datasets reveal that incorporating our methods improves NAFNet's PSNR by 0.05, 0.96, and 7.03, respectively. Code will be available at \href{https://github.com/HPLQAQ/SAM-Deblur}{SAM-Deblur}.

{{</citation>}}


### (8/114) DCP-Net: A Distributed Collaborative Perception Network for Remote Sensing Semantic Segmentation (Zhechao Wang et al., 2023)

{{<citation>}}

Zhechao Wang, Peirui Cheng, Shujing Duan, Kaiqiang Chen, Zhirui Wang, Xinming Li, Xian Sun. (2023)  
**DCP-Net: A Distributed Collaborative Perception Network for Remote Sensing Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2309.02230v1)  

---


**ABSTRACT**  
Onboard intelligent processing is widely applied in emergency tasks in the field of remote sensing. However, it is predominantly confined to an individual platform with a limited observation range as well as susceptibility to interference, resulting in limited accuracy. Considering the current state of multi-platform collaborative observation, this article innovatively presents a distributed collaborative perception network called DCP-Net. Firstly, the proposed DCP-Net helps members to enhance perception performance by integrating features from other platforms. Secondly, a self-mutual information match module is proposed to identify collaboration opportunities and select suitable partners, prioritizing critical collaborative features and reducing redundant transmission cost. Thirdly, a related feature fusion module is designed to address the misalignment between local and collaborative features, improving the quality of fused features for the downstream task. We conduct extensive experiments and visualization analyses using three semantic segmentation datasets, including Potsdam, iSAID and DFC23. The results demonstrate that DCP-Net outperforms the existing methods comprehensively, improving mIoU by 2.61%~16.89% at the highest collaboration efficiency, which promotes the performance to a state-of-the-art level.

{{</citation>}}


### (9/114) Dense Object Grounding in 3D Scenes (Wencan Huang et al., 2023)

{{<citation>}}

Wencan Huang, Daizong Liu, Wei Hu. (2023)  
**Dense Object Grounding in 3D Scenes**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.02224v1)  

---


**ABSTRACT**  
Localizing objects in 3D scenes according to the semantics of a given natural language is a fundamental yet important task in the field of multimedia understanding, which benefits various real-world applications such as robotics and autonomous driving. However, the majority of existing 3D object grounding methods are restricted to a single-sentence input describing an individual object, which cannot comprehend and reason more contextualized descriptions of multiple objects in more practical 3D cases. To this end, we introduce a new challenging task, called 3D Dense Object Grounding (3D DOG), to jointly localize multiple objects described in a more complicated paragraph rather than a single sentence. Instead of naively localizing each sentence-guided object independently, we found that dense objects described in the same paragraph are often semantically related and spatially located in a focused region of the 3D scene. To explore such semantic and spatial relationships of densely referred objects for more accurate localization, we propose a novel Stacked Transformer based framework for 3D DOG, named 3DOGSFormer. Specifically, we first devise a contextual query-driven local transformer decoder to generate initial grounding proposals for each target object. Then, we employ a proposal-guided global transformer decoder that exploits the local object features to learn their correlation for further refining initial grounding proposals. Extensive experiments on three challenging benchmarks (Nr3D, Sr3D, and ScanRefer) show that our proposed 3DOGSFormer outperforms state-of-the-art 3D single-object grounding methods and their dense-object variants by significant margins.

{{</citation>}}


### (10/114) Delving into Ipsilateral Mammogram Assessment under Multi-View Network (Thai Ngoc Toan Truong et al., 2023)

{{<citation>}}

Thai Ngoc Toan Truong, Thanh-Huy Nguyen, Ba Thinh Lam, Vu Minh Duy Nguyen, Hong Phuc Nguyen. (2023)  
**Delving into Ipsilateral Mammogram Assessment under Multi-View Network**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02197v2)  

---


**ABSTRACT**  
In many recent years, multi-view mammogram analysis has been focused widely on AI-based cancer assessment. In this work, we aim to explore diverse fusion strategies (average and concatenate) and examine the model's learning behavior with varying individuals and fusion pathways, involving Coarse Layer and Fine Layer. The Ipsilateral Multi-View Network, comprising five fusion types (Pre, Early, Middle, Last, and Post Fusion) in ResNet-18, is employed. Notably, the Middle Fusion emerges as the most balanced and effective approach, enhancing deep-learning models' generalization performance by +2.06% (concatenate) and +5.29% (average) in VinDr-Mammo dataset and +2.03% (concatenate) and +3% (average) in CMMD dataset on macro F1-Score. The paper emphasizes the crucial role of layer assignment in multi-view network extraction with various strategies.

{{</citation>}}


### (11/114) Exchanging-based Multimodal Fusion with Transformer (Renyu Zhu et al., 2023)

{{<citation>}}

Renyu Zhu, Chengcheng Han, Yong Qian, Qiushi Sun, Xiang Li, Ming Gao, Xuezhi Cao, Yunsen Xian. (2023)  
**Exchanging-based Multimodal Fusion with Transformer**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Named Entity Recognition, Sentiment Analysis, Transformer  
[Paper Link](http://arxiv.org/abs/2309.02190v1)  

---


**ABSTRACT**  
We study the problem of multimodal fusion in this paper. Recent exchanging-based methods have been proposed for vision-vision fusion, which aim to exchange embeddings learned from one modality to the other. However, most of them project inputs of multimodalities into different low-dimensional spaces and cannot be applied to the sequential input data. To solve these issues, in this paper, we propose a novel exchanging-based multimodal fusion model MuSE for text-vision fusion based on Transformer. We first use two encoders to separately map multimodal inputs into different low-dimensional spaces. Then we employ two decoders to regularize the embeddings and pull them into the same space. The two decoders capture the correlations between texts and images with the image captioning task and the text-to-image generation task, respectively. Further, based on the regularized embeddings, we present CrossTransformer, which uses two Transformer encoders with shared parameters as the backbone model to exchange knowledge between multimodalities. Specifically, CrossTransformer first learns the global contextual information of the inputs in the shallow layers. After that, it performs inter-modal exchange by selecting a proportion of tokens in one modality and replacing their embeddings with the average of embeddings in the other modality. We conduct extensive experiments to evaluate the performance of MuSE on the Multimodal Named Entity Recognition task and the Multimodal Sentiment Analysis task. Our results show the superiority of MuSE against other competitors. Our code and data are provided at https://github.com/RecklessRonan/MuSE.

{{</citation>}}


### (12/114) S3C: Semi-Supervised VQA Natural Language Explanation via Self-Critical Learning (Wei Suo et al., 2023)

{{<citation>}}

Wei Suo, Mengyang Sun, Weisong Liu, Yiqi Gao, Peng Wang, Yanning Zhang, Qi Wu. (2023)  
**S3C: Semi-Supervised VQA Natural Language Explanation via Self-Critical Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2309.02155v1)  

---


**ABSTRACT**  
VQA Natural Language Explanation (VQA-NLE) task aims to explain the decision-making process of VQA models in natural language. Unlike traditional attention or gradient analysis, free-text rationales can be easier to understand and gain users' trust. Existing methods mostly use post-hoc or self-rationalization models to obtain a plausible explanation. However, these frameworks are bottlenecked by the following challenges: 1) the reasoning process cannot be faithfully responded to and suffer from the problem of logical inconsistency. 2) Human-annotated explanations are expensive and time-consuming to collect. In this paper, we propose a new Semi-Supervised VQA-NLE via Self-Critical Learning (S3C), which evaluates the candidate explanations by answering rewards to improve the logical consistency between answers and rationales. With a semi-supervised learning framework, the S3C can benefit from a tremendous amount of samples without human-annotated explanations. A large number of automatic measures and human evaluations all show the effectiveness of our method. Meanwhile, the framework achieves a new state-of-the-art performance on the two VQA-NLE datasets.

{{</citation>}}


### (13/114) Self-Supervised Pre-Training Boosts Semantic Scene Segmentation on LiDAR data (Mariona Carós et al., 2023)

{{<citation>}}

Mariona Carós, Ariadna Just, Santi Seguí, Jordi Vitrià. (2023)  
**Self-Supervised Pre-Training Boosts Semantic Scene Segmentation on LiDAR data**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.02139v1)  

---


**ABSTRACT**  
Airborne LiDAR systems have the capability to capture the Earth's surface by generating extensive point cloud data comprised of points mainly defined by 3D coordinates. However, labeling such points for supervised learning tasks is time-consuming. As a result, there is a need to investigate techniques that can learn from unlabeled data to significantly reduce the number of annotated samples. In this work, we propose to train a self-supervised encoder with Barlow Twins and use it as a pre-trained network in the task of semantic scene segmentation. The experimental results demonstrate that our unsupervised pre-training boosts performance once fine-tuned on the supervised task, especially for under-represented categories.

{{</citation>}}


### (14/114) Dual Adversarial Alignment for Realistic Support-Query Shift Few-shot Learning (Siyang Jiang et al., 2023)

{{<citation>}}

Siyang Jiang, Rui Fang, Hsi-Wen Chen, Wei Ding, Ming-Syan Chen. (2023)  
**Dual Adversarial Alignment for Realistic Support-Query Shift Few-shot Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.02088v1)  

---


**ABSTRACT**  
Support-query shift few-shot learning aims to classify unseen examples (query set) to labeled data (support set) based on the learned embedding in a low-dimensional space under a distribution shift between the support set and the query set. However, in real-world scenarios the shifts are usually unknown and varied, making it difficult to estimate in advance. Therefore, in this paper, we propose a novel but more difficult challenge, RSQS, focusing on Realistic Support-Query Shift few-shot learning. The key feature of RSQS is that the individual samples in a meta-task are subjected to multiple distribution shifts in each meta-task. In addition, we propose a unified adversarial feature alignment method called DUal adversarial ALignment framework (DuaL) to relieve RSQS from two aspects, i.e., inter-domain bias and intra-domain variance. On the one hand, for the inter-domain bias, we corrupt the original data in advance and use the synthesized perturbed inputs to train the repairer network by minimizing distance in the feature level. On the other hand, for intra-domain variance, we proposed a generator network to synthesize hard, i.e., less similar, examples from the support set in a self-supervised manner and introduce regularized optimal transportation to derive a smooth optimal transportation plan. Lastly, a benchmark of RSQS is built with several state-of-the-art baselines among three datasets (CIFAR100, mini-ImageNet, and Tiered-Imagenet). Experiment results show that DuaL significantly outperforms the state-of-the-art methods in our benchmark.

{{</citation>}}


### (15/114) Diffusion-based 3D Object Detection with Random Boxes (Xin Zhou et al., 2023)

{{<citation>}}

Xin Zhou, Jinghua Hou, Tingting Yao, Dingkang Liang, Zhe Liu, Zhikang Zou, Xiaoqing Ye, Jianwei Cheng, Xiang Bai. (2023)  
**Diffusion-based 3D Object Detection with Random Boxes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.02049v1)  

---


**ABSTRACT**  
3D object detection is an essential task for achieving autonomous driving. Existing anchor-based detection methods rely on empirical heuristics setting of anchors, which makes the algorithms lack elegance. In recent years, we have witnessed the rise of several generative models, among which diffusion models show great potential for learning the transformation of two distributions. Our proposed Diff3Det migrates the diffusion model to proposal generation for 3D object detection by considering the detection boxes as generative targets. During training, the object boxes diffuse from the ground truth boxes to the Gaussian distribution, and the decoder learns to reverse this noise process. In the inference stage, the model progressively refines a set of random boxes to the prediction results. We provide detailed experiments on the KITTI benchmark and achieve promising performance compared to classical anchor-based 3D detection methods.

{{</citation>}}


### (16/114) Learning Cross-Modal Affinity for Referring Video Object Segmentation Targeting Limited Samples (Guanghui Li et al., 2023)

{{<citation>}}

Guanghui Li, Mingqi Gao, Heng Liu, Xiantong Zhen, Feng Zheng. (2023)  
**Learning Cross-Modal Affinity for Referring Video Object Segmentation Targeting Limited Samples**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2309.02041v1)  

---


**ABSTRACT**  
Referring video object segmentation (RVOS), as a supervised learning task, relies on sufficient annotated data for a given scene. However, in more realistic scenarios, only minimal annotations are available for a new scene, which poses significant challenges to existing RVOS methods. With this in mind, we propose a simple yet effective model with a newly designed cross-modal affinity (CMA) module based on a Transformer architecture. The CMA module builds multimodal affinity with a few samples, thus quickly learning new semantic information, and enabling the model to adapt to different scenarios. Since the proposed method targets limited samples for new scenes, we generalize the problem as - few-shot referring video object segmentation (FS-RVOS). To foster research in this direction, we build up a new FS-RVOS benchmark based on currently available datasets. The benchmark covers a wide range and includes multiple situations, which can maximally simulate real-world scenarios. Extensive experiments show that our model adapts well to different scenarios with only a few samples, reaching state-of-the-art performance on the benchmark. On Mini-Ref-YouTube-VOS, our model achieves an average performance of 53.1 J and 54.8 F, which are 10% better than the baselines. Furthermore, we show impressive results of 77.7 J and 74.8 F on Mini-Ref-SAIL-VOS, which are significantly better than the baselines. Code is publicly available at https://github.com/hengliusky/Few_shot_RVOS.

{{</citation>}}


### (17/114) A survey on efficient vision transformers: algorithms, techniques, and performance benchmarking (Lorenzo Papa et al., 2023)

{{<citation>}}

Lorenzo Papa, Paolo Russo, Irene Amerini, Luping Zhou. (2023)  
**A survey on efficient vision transformers: algorithms, techniques, and performance benchmarking**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.02031v1)  

---


**ABSTRACT**  
Vision Transformer (ViT) architectures are becoming increasingly popular and widely employed to tackle computer vision applications. Their main feature is the capacity to extract global information through the self-attention mechanism, outperforming earlier convolutional neural networks. However, ViT deployment and performance have grown steadily with their size, number of trainable parameters, and operations. Furthermore, self-attention's computational and memory cost quadratically increases with the image resolution. Generally speaking, it is challenging to employ these architectures in real-world applications due to many hardware and environmental restrictions, such as processing and computational capabilities. Therefore, this survey investigates the most efficient methodologies to ensure sub-optimal estimation performances. More in detail, four efficient categories will be analyzed: compact architecture, pruning, knowledge distillation, and quantization strategies. Moreover, a new metric called Efficient Error Rate has been introduced in order to normalize and compare models' features that affect hardware devices at inference time, such as the number of parameters, bits, FLOPs, and model size. Summarizing, this paper firstly mathematically defines the strategies used to make Vision Transformer efficient, describes and discusses state-of-the-art methodologies, and analyzes their performances over different application scenarios. Toward the end of this paper, we also discuss open challenges and promising research directions.

{{</citation>}}


### (18/114) Analyzing domain shift when using additional data for the MICCAI KiTS23 Challenge (George Stoica et al., 2023)

{{<citation>}}

George Stoica, Mihaela Breaban, Vlad Barbu. (2023)  
**Analyzing domain shift when using additional data for the MICCAI KiTS23 Challenge**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02001v1)  

---


**ABSTRACT**  
Using additional training data is known to improve the results, especially for medical image 3D segmentation where there is a lack of training material and the model needs to generalize well from few available data. However, the new data could have been acquired using other instruments and preprocessed such its distribution is significantly different from the original training data. Therefore, we study techniques which ameliorate domain shift during training so that the additional data becomes better usable for preprocessing and training together with the original data. Our results show that transforming the additional data using histogram matching has better results than using simple normalization.

{{</citation>}}


### (19/114) NICE 2023 Zero-shot Image Captioning Challenge (Taehoon Kim et al., 2023)

{{<citation>}}

Taehoon Kim, Pyunghwan Ahn, Sangyun Kim, Sihaeng Lee, Mark Marsden, Alessandra Sala, Seung Hwan Kim, Bohyung Han, Kyoung Mu Lee, Honglak Lee, Kyounghoon Bae, Xiangyu Wu, Yi Gao, Hailiang Zhang, Yang Yang, Weili Guo, Jianfeng Lu, Youngtaek Oh, Jae Won Cho, Dong-jin Kim, In So Kweon, Junmo Kim, Wooyoung Kang, Won Young Jhoo, Byungseok Roh, Jonghwan Mun, Solgil Oh, Kenan Emir Ak, Gwang-Gook Lee, Yan Xu, Mingwei Shen, Kyomin Hwang, Wonsik Shin, Kamin Lee, Wonhark Park, Dongkwan Lee, Nojun Kwak, Yujin Wang, Yimu Wang, Tiancheng Gu, Xingchang Lv, Mingmao Sun. (2023)  
**NICE 2023 Zero-shot Image Captioning Challenge**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Image Captioning  
[Paper Link](http://arxiv.org/abs/2309.01961v2)  

---


**ABSTRACT**  
In this report, we introduce NICE project\footnote{\url{https://nice.lgresearch.ai/}} and share the results and outcomes of NICE challenge 2023. This project is designed to challenge the computer vision community to develop robust image captioning models that advance the state-of-the-art both in terms of accuracy and fairness. Through the challenge, the image captioning models were tested using a new evaluation dataset that includes a large variety of visual concepts from many domains. There was no specific training data provided for the challenge, and therefore the challenge entries were required to adapt to new types of image descriptions that had not been seen during training. This report includes information on the newly proposed NICE dataset, evaluation methods, challenge results, and technical details of top-ranking entries. We expect that the outcomes of the challenge will contribute to the improvement of AI models on various vision-language tasks.

{{</citation>}}


### (20/114) Extract-and-Adaptation Network for 3D Interacting Hand Mesh Recovery (JoonKyu Park et al., 2023)

{{<citation>}}

JoonKyu Park, Daniel Sungho Jung, Gyeongsik Moon, Kyoung Mu Lee. (2023)  
**Extract-and-Adaptation Network for 3D Interacting Hand Mesh Recovery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.01943v1)  

---


**ABSTRACT**  
Understanding how two hands interact with each other is a key component of accurate 3D interacting hand mesh recovery. However, recent Transformer-based methods struggle to learn the interaction between two hands as they directly utilize two hand features as input tokens, which results in distant token problem. The distant token problem represents that input tokens are in heterogeneous spaces, leading Transformer to fail in capturing correlation between input tokens. Previous Transformer-based methods suffer from the problem especially when poses of two hands are very different as they project features from a backbone to separate left and right hand-dedicated features. We present EANet, extract-and-adaptation network, with EABlock, the main component of our network. Rather than directly utilizing two hand features as input tokens, our EABlock utilizes two complementary types of novel tokens, SimToken and JoinToken, as input tokens. Our two novel tokens are from a combination of separated two hand features; hence, it is much more robust to the distant token problem. Using the two type of tokens, our EABlock effectively extracts interaction feature and adapts it to each hand. The proposed EANet achieves the state-of-the-art performance on 3D interacting hands benchmarks. The codes are available at https://github.com/jkpark0825/EANet.

{{</citation>}}


## eess.IV (3)



### (21/114) Generative AI-aided Joint Training-free Secure Semantic Communications via Multi-modal Prompts (Hongyang Du et al., 2023)

{{<citation>}}

Hongyang Du, Guangyuan Liu, Dusit Niyato, Jiayi Zhang, Jiawen Kang, Zehui Xiong, Bo Ai, Dong In Kim. (2023)  
**Generative AI-aided Joint Training-free Secure Semantic Communications via Multi-modal Prompts**  

---
Primary Category: eess.IV  
Categories: cs-LG, cs-NI, eess-IV, eess.IV  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2309.02616v1)  

---


**ABSTRACT**  
Semantic communication (SemCom) holds promise for reducing network resource consumption while achieving the communications goal. However, the computational overheads in jointly training semantic encoders and decoders-and the subsequent deployment in network devices-are overlooked. Recent advances in Generative artificial intelligence (GAI) offer a potential solution. The robust learning abilities of GAI models indicate that semantic decoders can reconstruct source messages using a limited amount of semantic information, e.g., prompts, without joint training with the semantic encoder. A notable challenge, however, is the instability introduced by GAI's diverse generation ability. This instability, evident in outputs like text-generated images, limits the direct application of GAI in scenarios demanding accurate message recovery, such as face image transmission. To solve the above problems, this paper proposes a GAI-aided SemCom system with multi-model prompts for accurate content decoding. Moreover, in response to security concerns, we introduce the application of covert communications aided by a friendly jammer. The system jointly optimizes the diffusion step, jamming, and transmitting power with the aid of the generative diffusion models, enabling successful and secure transmission of the source messages.

{{</citation>}}


### (22/114) Evaluation Kidney Layer Segmentation on Whole Slide Imaging using Convolutional Neural Networks and Transformers (Muhao Liu et al., 2023)

{{<citation>}}

Muhao Liu, Chenyang Qi, Shunxing Bao, Quan Liu, Ruining Deng, Yu Wang, Shilin Zhao, Haichun Yang, Yuankai Huo. (2023)  
**Evaluation Kidney Layer Segmentation on Whole Slide Imaging using Convolutional Neural Networks and Transformers**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.02563v1)  

---


**ABSTRACT**  
The segmentation of kidney layer structures, including cortex, outer stripe, inner stripe, and inner medulla within human kidney whole slide images (WSI) plays an essential role in automated image analysis in renal pathology. However, the current manual segmentation process proves labor-intensive and infeasible for handling the extensive digital pathology images encountered at a large scale. In response, the realm of digital renal pathology has seen the emergence of deep learning-based methodologies. However, very few, if any, deep learning based approaches have been applied to kidney layer structure segmentation. Addressing this gap, this paper assesses the feasibility of performing deep learning based approaches on kidney layer structure segmetnation. This study employs the representative convolutional neural network (CNN) and Transformer segmentation approaches, including Swin-Unet, Medical-Transformer, TransUNet, U-Net, PSPNet, and DeepLabv3+. We quantitatively evaluated six prevalent deep learning models on renal cortex layer segmentation using mice kidney WSIs. The empirical results stemming from our approach exhibit compelling advancements, as evidenced by a decent Mean Intersection over Union (mIoU) index. The results demonstrate that Transformer models generally outperform CNN-based models. By enabling a quantitative evaluation of renal cortical structures, deep learning approaches are promising to empower these medical professionals to make more informed kidney layer segmentation.

{{</citation>}}


### (23/114) INCEPTNET: Precise And Early Disease Detection Application For Medical Images Analyses (Amirhossein Sajedi et al., 2023)

{{<citation>}}

Amirhossein Sajedi, Mohammad Javad Fadaeieslam. (2023)  
**INCEPTNET: Precise And Early Disease Detection Application For Medical Images Analyses**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02147v1)  

---


**ABSTRACT**  
In view of the recent paradigm shift in deep AI based image processing methods, medical image processing has advanced considerably. In this study, we propose a novel deep neural network (DNN), entitled InceptNet, in the scope of medical image processing, for early disease detection and segmentation of medical images in order to enhance precision and performance. We also investigate the interaction of users with the InceptNet application to present a comprehensive application including the background processes, and foreground interactions with users. Fast InceptNet is shaped by the prominent Unet architecture, and it seizes the power of an Inception module to be fast and cost effective while aiming to approximate an optimal local sparse structure. Adding Inception modules with various parallel kernel sizes can improve the network's ability to capture the variations in the scaled regions of interest. To experiment, the model is tested on four benchmark datasets, including retina blood vessel segmentation, lung nodule segmentation, skin lesion segmentation, and breast cancer cell detection. The improvement was more significant on images with small scale structures. The proposed method improved the accuracy from 0.9531, 0.8900, 0.9872, and 0.9881 to 0.9555, 0.9510, 0.9945, and 0.9945 on the mentioned datasets, respectively, which show outperforming of the proposed method over the previous works. Furthermore, by exploring the procedure from start to end, individuals who have utilized a trial edition of InceptNet, in the form of a complete application, are presented with thirteen multiple choice questions in order to assess the proposed method. The outcomes are evaluated through the means of Human Computer Interaction.

{{</citation>}}


## cs.SD (1)



### (24/114) Music Source Separation with Band-Split RoPE Transformer (Wei-Tsung Lu et al., 2023)

{{<citation>}}

Wei-Tsung Lu, Ju-Chiang Wang, Qiuqiang Kong, Yun-Ning Hung. (2023)  
**Music Source Separation with Band-Split RoPE Transformer**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Embedding, Position Embedding, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.02612v1)  

---


**ABSTRACT**  
Music source separation (MSS) aims to separate a music recording into multiple musically distinct stems, such as vocals, bass, drums, and more. Recently, deep learning approaches such as convolutional neural networks (CNNs) and recurrent neural networks (RNNs) have been used, but the improvement is still limited. In this paper, we propose a novel frequency-domain approach based on a Band-Split RoPE Transformer (called BS-RoFormer). BS-RoFormer replies on a band-split module to project the input complex spectrogram into subband-level representations, and then arranges a stack of hierarchical Transformers to model the inner-band as well as inter-band sequences for multi-band mask estimation. To facilitate training the model for MSS, we propose to use the Rotary Position Embedding (RoPE). The BS-RoFormer system trained on MUSDB18HQ and 500 extra songs ranked the first place in the MSS track of Sound Demixing Challenge (SDX23). Benchmarking a smaller version of BS-RoFormer on MUSDB18HQ, we achieve state-of-the-art result without extra training data, with 9.80 dB of average SDR.

{{</citation>}}


## cs.NE (1)



### (25/114) Comparative Evaluation of Metaheuristic Algorithms for Hyperparameter Selection in Short-Term Weather Forecasting (Anuvab Sen et al., 2023)

{{<citation>}}

Anuvab Sen, Arul Rhik Mazumder, Dibyarup Dutta, Udayon Sen, Pathikrit Syam, Sandipan Dhar. (2023)  
**Comparative Evaluation of Metaheuristic Algorithms for Hyperparameter Selection in Short-Term Weather Forecasting**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-NE, cs.NE  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.02600v1)  

---


**ABSTRACT**  
Weather forecasting plays a vital role in numerous sectors, but accurately capturing the complex dynamics of weather systems remains a challenge for traditional statistical models. Apart from Auto Regressive time forecasting models like ARIMA, deep learning techniques (Vanilla ANNs, LSTM and GRU networks), have shown promise in improving forecasting accuracy by capturing temporal dependencies. This paper explores the application of metaheuristic algorithms, namely Genetic Algorithm (GA), Differential Evolution (DE), and Particle Swarm Optimization (PSO), to automate the search for optimal hyperparameters in these model architectures. Metaheuristic algorithms excel in global optimization, offering robustness, versatility, and scalability in handling non-linear problems. We present a comparative analysis of different model architectures integrated with metaheuristic optimization, evaluating their performance in weather forecasting based on metrics such as Mean Squared Error (MSE) and Mean Absolute Percentage Error (MAPE). The results demonstrate the potential of metaheuristic algorithms in enhancing weather forecasting accuracy \& helps in determining the optimal set of hyper-parameters for each model. The paper underscores the importance of harnessing advanced optimization techniques to select the most suitable metaheuristic algorithm for the given weather forecasting task.

{{</citation>}}


## cs.LG (29)



### (26/114) Representation Learning for Sequential Volumetric Design Tasks (Md Ferdous Alam et al., 2023)

{{<citation>}}

Md Ferdous Alam, Yi Wang, Linh Tran, Chin-Yi Cheng, Jieliang Luo. (2023)  
**Representation Learning for Sequential Volumetric Design Tasks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.02583v1)  

---


**ABSTRACT**  
Volumetric design, also called massing design, is the first and critical step in professional building design which is sequential in nature. As the volumetric design process is complex, the underlying sequential design process encodes valuable information for designers. Many efforts have been made to automatically generate reasonable volumetric designs, but the quality of the generated design solutions varies, and evaluating a design solution requires either a prohibitively comprehensive set of metrics or expensive human expertise. While previous approaches focused on learning only the final design instead of sequential design tasks, we propose to encode the design knowledge from a collection of expert or high-performing design sequences and extract useful representations using transformer-based models. Later we propose to utilize the learned representations for crucial downstream applications such as design preference evaluation and procedural design generation. We develop the preference model by estimating the density of the learned representations whereas we train an autoregressive transformer model for sequential design generation. We demonstrate our ideas by leveraging a novel dataset of thousands of sequential volumetric designs. Our preference model can compare two arbitrarily given design sequences and is almost 90% accurate in evaluation against random design sequences. Our autoregressive model is also capable of autocompleting a volumetric design sequence from a partial design sequence.

{{</citation>}}


### (27/114) Unveiling Intractable Epileptogenic Brain Networks with Deep Learning Algorithms: A Novel and Comprehensive Framework for Scalable Seizure Prediction with Unimodal Neuroimaging Data in Pediatric Patients (Bliss Singhal et al., 2023)

{{<citation>}}

Bliss Singhal, Fnu Pooja. (2023)  
**Unveiling Intractable Epileptogenic Brain Networks with Deep Learning Algorithms: A Novel and Comprehensive Framework for Scalable Seizure Prediction with Unimodal Neuroimaging Data in Pediatric Patients**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-IV  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.02580v1)  

---


**ABSTRACT**  
Epilepsy is a prevalent neurological disorder affecting 50 million individuals worldwide and 1.2 million Americans. There exist millions of pediatric patients with intractable epilepsy, a condition in which seizures fail to come under control. The occurrence of seizures can result in physical injury, disorientation, unconsciousness, and additional symptoms that could impede children's ability to participate in everyday tasks. Predicting seizures can help parents and healthcare providers take precautions, prevent risky situations, and mentally prepare children to minimize anxiety and nervousness associated with the uncertainty of a seizure. This research proposes a novel and comprehensive framework to predict seizures in pediatric patients by evaluating machine learning algorithms on unimodal neuroimaging data consisting of electroencephalogram signals. The bandpass filtering and independent component analysis proved to be effective in reducing the noise and artifacts from the dataset. Various machine learning algorithms' performance is evaluated on important metrics such as accuracy, precision, specificity, sensitivity, F1 score and MCC. The results show that the deep learning algorithms are more successful in predicting seizures than logistic Regression, and k nearest neighbors. The recurrent neural network (RNN) gave the highest precision and F1 Score, long short-term memory (LSTM) outperformed RNN in accuracy and convolutional neural network (CNN) resulted in the highest Specificity. This research has significant implications for healthcare providers in proactively managing seizure occurrence in pediatric patients, potentially transforming clinical practices, and improving pediatric care.

{{</citation>}}


### (28/114) A Survey of the Impact of Self-Supervised Pretraining for Diagnostic Tasks with Radiological Images (Blake VanBerlo et al., 2023)

{{<citation>}}

Blake VanBerlo, Jesse Hoey, Alexander Wong. (2023)  
**A Survey of the Impact of Self-Supervised Pretraining for Diagnostic Tasks with Radiological Images**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.02555v1)  

---


**ABSTRACT**  
Self-supervised pretraining has been observed to be effective at improving feature representations for transfer learning, leveraging large amounts of unlabelled data. This review summarizes recent research into its usage in X-ray, computed tomography, magnetic resonance, and ultrasound imaging, concentrating on studies that compare self-supervised pretraining to fully supervised learning for diagnostic tasks such as classification and segmentation. The most pertinent finding is that self-supervised pretraining generally improves downstream task performance compared to full supervision, most prominently when unlabelled examples greatly outnumber labelled examples. Based on the aggregate evidence, recommendations are provided for practitioners considering using self-supervised learning. Motivated by limitations identified in current research, directions and practices for future study are suggested, such as integrating clinical knowledge with theoretically justified self-supervised learning methods, evaluating on public datasets, growing the modest body of evidence for ultrasound, and characterizing the impact of self-supervised pretraining on generalization.

{{</citation>}}


### (29/114) Adaptive Adversarial Training Does Not Increase Recourse Costs (Ian Hardy et al., 2023)

{{<citation>}}

Ian Hardy, Jayanth Yetukuri, Yang Liu. (2023)  
**Adaptive Adversarial Training Does Not Increase Recourse Costs**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Adversarial Training  
[Paper Link](http://arxiv.org/abs/2309.02528v1)  

---


**ABSTRACT**  
Recent work has connected adversarial attack methods and algorithmic recourse methods: both seek minimal changes to an input instance which alter a model's classification decision. It has been shown that traditional adversarial training, which seeks to minimize a classifier's susceptibility to malicious perturbations, increases the cost of generated recourse; with larger adversarial training radii correlating with higher recourse costs. From the perspective of algorithmic recourse, however, the appropriate adversarial training radius has always been unknown. Another recent line of work has motivated adversarial training with adaptive training radii to address the issue of instance-wise variable adversarial vulnerability, showing success in domains with unknown attack radii. This work studies the effects of adaptive adversarial training on algorithmic recourse costs. We establish that the improvements in model robustness induced by adaptive adversarial training show little effect on algorithmic recourse costs, providing a potential avenue for affordable robustness in domains where recoursability is critical.

{{</citation>}}


### (30/114) Graph Self-Contrast Representation Learning (Minjie Chen et al., 2023)

{{<citation>}}

Minjie Chen, Yao Cheng, Ye Wang, Xiang Li, Ming Gao. (2023)  
**Graph Self-Contrast Representation Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-GR, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.02304v1)  

---


**ABSTRACT**  
Graph contrastive learning (GCL) has recently emerged as a promising approach for graph representation learning. Some existing methods adopt the 1-vs-K scheme to construct one positive and K negative samples for each graph, but it is difficult to set K. For those methods that do not use negative samples, it is often necessary to add additional strategies to avoid model collapse, which could only alleviate the problem to some extent. All these drawbacks will undoubtedly have an adverse impact on the generalizability and efficiency of the model. In this paper, to address these issues, we propose a novel graph self-contrast framework GraphSC, which only uses one positive and one negative sample, and chooses triplet loss as the objective. Specifically, self-contrast has two implications. First, GraphSC generates both positive and negative views of a graph sample from the graph itself via graph augmentation functions of various intensities, and use them for self-contrast. Second, GraphSC uses Hilbert-Schmidt Independence Criterion (HSIC) to factorize the representations into multiple factors and proposes a masked self-contrast mechanism to better separate positive and negative samples. Further, Since the triplet loss only optimizes the relative distance between the anchor and its positive/negative samples, it is difficult to ensure the absolute distance between the anchor and positive sample. Therefore, we explicitly reduced the absolute distance between the anchor and positive sample to accelerate convergence. Finally, we conduct extensive experiments to evaluate the performance of GraphSC against 19 other state-of-the-art methods in both unsupervised and transfer learning settings.

{{</citation>}}


### (31/114) Enhancing Semantic Communication with Deep Generative Models -- An ICASSP Special Session Overview (Eleonora Grassucci et al., 2023)

{{<citation>}}

Eleonora Grassucci, Yuki Mitsufuji, Ping Zhang, Danilo Comminiello. (2023)  
**Enhancing Semantic Communication with Deep Generative Models -- An ICASSP Special Session Overview**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, eess-SP  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02478v1)  

---


**ABSTRACT**  
Semantic communication is poised to play a pivotal role in shaping the landscape of future AI-driven communication systems. Its challenge of extracting semantic information from the original complex content and regenerating semantically consistent data at the receiver, possibly being robust to channel corruptions, can be addressed with deep generative models. This ICASSP special session overview paper discloses the semantic communication challenges from the machine learning perspective and unveils how deep generative models will significantly enhance semantic communication frameworks in dealing with real-world complex data, extracting and exploiting semantic information, and being robust to channel corruptions. Alongside establishing this emerging field, this paper charts novel research pathways for the next generative semantic communication frameworks.

{{</citation>}}


### (32/114) Graph-Based Automatic Feature Selection for Multi-Class Classification via Mean Simplified Silhouette (David Levin et al., 2023)

{{<citation>}}

David Levin, Gonen Singer. (2023)  
**Graph-Based Automatic Feature Selection for Multi-Class Classification via Mean Simplified Silhouette**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.02272v1)  

---


**ABSTRACT**  
This paper introduces a novel graph-based filter method for automatic feature selection (abbreviated as GB-AFS) for multi-class classification tasks. The method determines the minimum combination of features required to sustain prediction performance while maintaining complementary discriminating abilities between different classes. It does not require any user-defined parameters such as the number of features to select. The methodology employs the Jeffries-Matusita (JM) distance in conjunction with t-distributed Stochastic Neighbor Embedding (t-SNE) to generate a low-dimensional space reflecting how effectively each feature can differentiate between each pair of classes. The minimum number of features is selected using our newly developed Mean Simplified Silhouette (abbreviated as MSS) index, designed to evaluate the clustering results for the feature selection task. Experimental results on public data sets demonstrate the superior performance of the proposed GB-AFS over other filter-based techniques and automatic feature selection approaches. Moreover, the proposed algorithm maintained the accuracy achieved when utilizing all features, while using only $7\%$ to $30\%$ of the features. Consequently, this resulted in a reduction of the time needed for classifications, from $15\%$ to $70\%$.

{{</citation>}}


### (33/114) MA-VAE: Multi-head Attention-based Variational Autoencoder Approach for Anomaly Detection in Multivariate Time-series Applied to Automotive Endurance Powertrain Testing (Lucas Correia et al., 2023)

{{<citation>}}

Lucas Correia, Jan-Christoph Goos, Philipp Klein, Thomas Bäck, Anna V. Kononova. (2023)  
**MA-VAE: Multi-head Attention-based Variational Autoencoder Approach for Anomaly Detection in Multivariate Time-series Applied to Automotive Endurance Powertrain Testing**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Anomaly Detection, Attention  
[Paper Link](http://arxiv.org/abs/2309.02253v1)  

---


**ABSTRACT**  
A clear need for automatic anomaly detection applied to automotive testing has emerged as more and more attention is paid to the data recorded and manual evaluation by humans reaches its capacity. Such real-world data is massive, diverse, multivariate and temporal in nature, therefore requiring modelling of the testee behaviour. We propose a variational autoencoder with multi-head attention (MA-VAE), which, when trained on unlabelled data, not only provides very few false positives but also manages to detect the majority of the anomalies presented. In addition to that, the approach offers a novel way to avoid the bypass phenomenon, an undesirable behaviour investigated in literature. Lastly, the approach also introduces a new method to remap individual windows to a continuous time series. The results are presented in the context of a real-world industrial data set and several experiments are undertaken to further investigate certain aspects of the proposed model. When configured properly, it is 9% of the time wrong when an anomaly is flagged and discovers 67% of the anomalies present. Also, MA-VAE has the potential to perform well with only a fraction of the training and validation subset, however, to extract it, a more sophisticated threshold estimation method is required.

{{</citation>}}


### (34/114) Sample Size in Natural Language Processing within Healthcare Research (Jaya Chaturvedi et al., 2023)

{{<citation>}}

Jaya Chaturvedi, Diana Shamsutdinova, Felix Zimmer, Sumithra Velupillai, Daniel Stahl, Robert Stewart, Angus Roberts. (2023)  
**Sample Size in Natural Language Processing within Healthcare Research**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: BERT, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2309.02237v1)  

---


**ABSTRACT**  
Sample size calculation is an essential step in most data-based disciplines. Large enough samples ensure representativeness of the population and determine the precision of estimates. This is true for most quantitative studies, including those that employ machine learning methods, such as natural language processing, where free-text is used to generate predictions and classify instances of text. Within the healthcare domain, the lack of sufficient corpora of previously collected data can be a limiting factor when determining sample sizes for new studies. This paper tries to address the issue by making recommendations on sample sizes for text classification tasks in the healthcare domain.   Models trained on the MIMIC-III database of critical care records from Beth Israel Deaconess Medical Center were used to classify documents as having or not having Unspecified Essential Hypertension, the most common diagnosis code in the database. Simulations were performed using various classifiers on different sample sizes and class proportions. This was repeated for a comparatively less common diagnosis code within the database of diabetes mellitus without mention of complication.   Smaller sample sizes resulted in better results when using a K-nearest neighbours classifier, whereas larger sample sizes provided better results with support vector machines and BERT models. Overall, a sample size larger than 1000 was sufficient to provide decent performance metrics.   The simulations conducted within this study provide guidelines that can be used as recommendations for selecting appropriate sample sizes and class proportions, and for predicting expected performance, when building classifiers for textual healthcare data. The methodology used here can be modified for sample size estimates calculations with other datasets.

{{</citation>}}


### (35/114) Distributionally Robust Model-based Reinforcement Learning with Large State Spaces (Shyam Sundhar Ramesh et al., 2023)

{{<citation>}}

Shyam Sundhar Ramesh, Pier Giuseppe Sessa, Yifan Hu, Andreas Krause, Ilija Bogunovic. (2023)  
**Distributionally Robust Model-based Reinforcement Learning with Large State Spaces**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.02236v1)  

---


**ABSTRACT**  
Three major challenges in reinforcement learning are the complex dynamical systems with large state spaces, the costly data acquisition processes, and the deviation of real-world dynamics from the training environment deployment. To overcome these issues, we study distributionally robust Markov decision processes with continuous state spaces under the widely used Kullback-Leibler, chi-square, and total variation uncertainty sets. We propose a model-based approach that utilizes Gaussian Processes and the maximum variance reduction algorithm to efficiently learn multi-output nominal transition dynamics, leveraging access to a generative model (i.e., simulator). We further demonstrate the statistical sample complexity of the proposed method for different uncertainty sets. These complexity bounds are independent of the number of states and extend beyond linear dynamics, ensuring the effectiveness of our approach in identifying near-optimal distributionally-robust policies. The proposed method can be further combined with other model-free distributionally robust reinforcement learning methods to obtain a near-optimal robust policy. Experimental results demonstrate the robustness of our algorithm to distributional shifts and its superior performance in terms of the number of samples needed.

{{</citation>}}


### (36/114) Improving equilibrium propagation without weight symmetry through Jacobian homeostasis (Axel Laborieux et al., 2023)

{{<citation>}}

Axel Laborieux, Friedemann Zenke. (2023)  
**Improving equilibrium propagation without weight symmetry through Jacobian homeostasis**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.02214v1)  

---


**ABSTRACT**  
Equilibrium propagation (EP) is a compelling alternative to the backpropagation of error algorithm (BP) for computing gradients of neural networks on biological or analog neuromorphic substrates. Still, the algorithm requires weight symmetry and infinitesimal equilibrium perturbations, i.e., nudges, to estimate unbiased gradients efficiently. Both requirements are challenging to implement in physical systems. Yet, whether and how weight asymmetry affects its applicability is unknown because, in practice, it may be masked by biases introduced through the finite nudge. To address this question, we study generalized EP, which can be formulated without weight symmetry, and analytically isolate the two sources of bias. For complex-differentiable non-symmetric networks, we show that the finite nudge does not pose a problem, as exact derivatives can still be estimated via a Cauchy integral. In contrast, weight asymmetry introduces bias resulting in low task performance due to poor alignment of EP's neuronal error vectors compared to BP. To mitigate this issue, we present a new homeostatic objective that directly penalizes functional asymmetries of the Jacobian at the network's fixed point. This homeostatic objective dramatically improves the network's ability to solve complex tasks such as ImageNet 32x32. Our results lay the theoretical groundwork for studying and mitigating the adverse effects of imperfections of physical networks on learning algorithms that rely on the substrate's relaxation dynamics.

{{</citation>}}


### (37/114) Language Models for Novelty Detection in System Call Traces (Quentin Fournier et al., 2023)

{{<citation>}}

Quentin Fournier, Daniel Aloise, Leandro R. Costa. (2023)  
**Language Models for Novelty Detection in System Call Traces**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-OS, cs-SE, cs.LG  
Keywords: LSTM, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2309.02206v1)  

---


**ABSTRACT**  
Due to the complexity of modern computer systems, novel and unexpected behaviors frequently occur. Such deviations are either normal occurrences, such as software updates and new user activities, or abnormalities, such as misconfigurations, latency issues, intrusions, and software bugs. Regardless, novel behaviors are of great interest to developers, and there is a genuine need for efficient and effective methods to detect them. Nowadays, researchers consider system calls to be the most fine-grained and accurate source of information to investigate the behavior of computer systems. Accordingly, this paper introduces a novelty detection methodology that relies on a probability distribution over sequences of system calls, which can be seen as a language model. Language models estimate the likelihood of sequences, and since novelties deviate from previously observed behaviors by definition, they would be unlikely under the model. Following the success of neural networks for language models, three architectures are evaluated in this work: the widespread LSTM, the state-of-the-art Transformer, and the lower-complexity Longformer. However, large neural networks typically require an enormous amount of data to be trained effectively, and to the best of our knowledge, no massive modern datasets of kernel traces are publicly available. This paper addresses this limitation by introducing a new open-source dataset of kernel traces comprising over 2 million web requests with seven distinct behaviors. The proposed methodology requires minimal expert hand-crafting and achieves an F-score and AuROC greater than 95% on most novelties while being data- and task-agnostic. The source code and trained models are publicly available on GitHub while the datasets are available on Zenodo.

{{</citation>}}


### (38/114) A Survey of Imitation Learning: Algorithms, Recent Developments, and Challenges (Maryam Zare et al., 2023)

{{<citation>}}

Maryam Zare, Parham M. Kebria, Abbas Khosravi, Saeid Nahavandi. (2023)  
**A Survey of Imitation Learning: Algorithms, Recent Developments, and Challenges**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02473v1)  

---


**ABSTRACT**  
In recent years, the development of robotics and artificial intelligence (AI) systems has been nothing short of remarkable. As these systems continue to evolve, they are being utilized in increasingly complex and unstructured environments, such as autonomous driving, aerial robotics, and natural language processing. As a consequence, programming their behaviors manually or defining their behavior through reward functions (as done in reinforcement learning (RL)) has become exceedingly difficult. This is because such environments require a high degree of flexibility and adaptability, making it challenging to specify an optimal set of rules or reward signals that can account for all possible situations. In such environments, learning from an expert's behavior through imitation is often more appealing. This is where imitation learning (IL) comes into play - a process where desired behavior is learned by imitating an expert's behavior, which is provided through demonstrations.   This paper aims to provide an introduction to IL and an overview of its underlying assumptions and approaches. It also offers a detailed description of recent advances and emerging areas of research in the field. Additionally, the paper discusses how researchers have addressed common challenges associated with IL and provides potential directions for future research. Overall, the goal of the paper is to provide a comprehensive guide to the growing field of IL in robotics and AI.

{{</citation>}}


### (39/114) Bias Propagation in Federated Learning (Hongyan Chang et al., 2023)

{{<citation>}}

Hongyan Chang, Reza Shokri. (2023)  
**Bias Propagation in Federated Learning**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG, stat-ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.02160v1)  

---


**ABSTRACT**  
We show that participating in federated learning can be detrimental to group fairness. In fact, the bias of a few parties against under-represented groups (identified by sensitive attributes such as gender or race) can propagate through the network to all the parties in the network. We analyze and explain bias propagation in federated learning on naturally partitioned real-world datasets. Our analysis reveals that biased parties unintentionally yet stealthily encode their bias in a small number of model parameters, and throughout the training, they steadily increase the dependence of the global model on sensitive attributes. What is important to highlight is that the experienced bias in federated learning is higher than what parties would otherwise encounter in centralized training with a model trained on the union of all their data. This indicates that the bias is due to the algorithm. Our work calls for auditing group fairness in federated learning and designing learning algorithms that are robust to bias propagation.

{{</citation>}}


### (40/114) Generalized Simplicial Attention Neural Networks (Claudio Battiloro et al., 2023)

{{<citation>}}

Claudio Battiloro, Lucia Testa, Lorenzo Giusti, Stefania Sardellitti, Paolo Di Lorenzo, Sergio Barbarossa. (2023)  
**Generalized Simplicial Attention Neural Networks**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, math-AT  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.02138v1)  

---


**ABSTRACT**  
The aim of this work is to introduce Generalized Simplicial Attention Neural Networks (GSANs), i.e., novel neural architectures designed to process data defined on simplicial complexes using masked self-attentional layers. Hinging on topological signal processing principles, we devise a series of self-attention schemes capable of processing data components defined at different simplicial orders, such as nodes, edges, triangles, and beyond. These schemes learn how to weight the neighborhoods of the given topological domain in a task-oriented fashion, leveraging the interplay among simplices of different orders through the Dirac operator and its Dirac decomposition. We also theoretically establish that GSANs are permutation equivariant and simplicial-aware. Finally, we illustrate how our approach compares favorably with other methods when applied to several (inductive and transductive) tasks such as trajectory prediction, missing data imputation, graph classification, and simplex prediction.

{{</citation>}}


### (41/114) Exploiting Spatial-temporal Data for Sleep Stage Classification via Hypergraph Learning (Yuze Liu et al., 2023)

{{<citation>}}

Yuze Liu, Ziming Zhao, Tiehua Zhang, Kang Wang, Xin Chen, Xiaowei Huang, Jun Yin, Zhishu Shen. (2023)  
**Exploiting Spatial-temporal Data for Sleep Stage Classification via Hypergraph Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2309.02124v1)  

---


**ABSTRACT**  
Sleep stage classification is crucial for detecting patients' health conditions. Existing models, which mainly use Convolutional Neural Networks (CNN) for modelling Euclidean data and Graph Convolution Networks (GNN) for modelling non-Euclidean data, are unable to consider the heterogeneity and interactivity of multimodal data as well as the spatial-temporal correlation simultaneously, which hinders a further improvement of classification performance. In this paper, we propose a dynamic learning framework STHL, which introduces hypergraph to encode spatial-temporal data for sleep stage classification. Hypergraphs can construct multi-modal/multi-type data instead of using simple pairwise between two subjects. STHL creates spatial and temporal hyperedges separately to build node correlations, then it conducts type-specific hypergraph learning process to encode the attributes into the embedding space. Extensive experiments show that our proposed STHL outperforms the state-of-the-art models in sleep stage classification tasks.

{{</citation>}}


### (42/114) Efficiency is Not Enough: A Critical Perspective of Environmentally Sustainable AI (Dustin Wright et al., 2023)

{{<citation>}}

Dustin Wright, Christian Igel, Gabrielle Samuel, Raghavendra Selvan. (2023)  
**Efficiency is Not Enough: A Critical Perspective of Environmentally Sustainable AI**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG, stat-ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02065v1)  

---


**ABSTRACT**  
Artificial Intelligence (AI) is currently spearheaded by machine learning (ML) methods such as deep learning (DL) which have accelerated progress on many tasks thought to be out of reach of AI. These ML methods can often be compute hungry, energy intensive, and result in significant carbon emissions, a known driver of anthropogenic climate change. Additionally, the platforms on which ML systems run are associated with environmental impacts including and beyond carbon emissions. The solution lionized by both industry and the ML community to improve the environmental sustainability of ML is to increase the efficiency with which ML systems operate in terms of both compute and energy consumption. In this perspective, we argue that efficiency alone is not enough to make ML as a technology environmentally sustainable. We do so by presenting three high level discrepancies between the effect of efficiency on the environmental sustainability of ML when considering the many variables which it interacts with. In doing so, we comprehensively demonstrate, at multiple levels of granularity both technical and non-technical reasons, why efficiency is not enough to fully remedy the environmental impacts of ML. Based on this, we present and argue for systems thinking as a viable path towards improving the environmental sustainability of ML holistically.

{{</citation>}}


### (43/114) Probabilistic Self-supervised Learning via Scoring Rules Minimization (Amirhossein Vahidi et al., 2023)

{{<citation>}}

Amirhossein Vahidi, Simon Schoßer, Lisa Wimmer, Yawei Li, Bernd Bischl, Eyke Hüllermeier, Mina Rezaei. (2023)  
**Probabilistic Self-supervised Learning via Scoring Rules Minimization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2309.02048v1)  

---


**ABSTRACT**  
In this paper, we propose a novel probabilistic self-supervised learning via Scoring Rule Minimization (ProSMIN), which leverages the power of probabilistic models to enhance representation quality and mitigate collapsing representations. Our proposed approach involves two neural networks; the online network and the target network, which collaborate and learn the diverse distribution of representations from each other through knowledge distillation. By presenting the input samples in two augmented formats, the online network is trained to predict the target network representation of the same sample under a different augmented view. The two networks are trained via our new loss function based on proper scoring rules. We provide a theoretical justification for ProSMIN's convergence, demonstrating the strict propriety of its modified scoring rule. This insight validates the method's optimization process and contributes to its robustness and effectiveness in improving representation quality. We evaluate our probabilistic model on various downstream tasks, such as in-distribution generalization, out-of-distribution detection, dataset corruption, low-shot learning, and transfer learning. Our method achieves superior accuracy and calibration, surpassing the self-supervised baseline in a wide range of experiments on large-scale datasets like ImageNet-O and ImageNet-C, ProSMIN demonstrates its scalability and real-world applicability.

{{</citation>}}


### (44/114) Diffusion Generative Inverse Design (Marin Vlastelica et al., 2023)

{{<citation>}}

Marin Vlastelica, Tatiana López-Guevara, Kelsey Allen, Peter Battaglia, Arnaud Doucet, Kimberley Stachenfeld. (2023)  
**Diffusion Generative Inverse Design**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2309.02040v1)  

---


**ABSTRACT**  
Inverse design refers to the problem of optimizing the input of an objective function in order to enact a target outcome. For many real-world engineering problems, the objective function takes the form of a simulator that predicts how the system state will evolve over time, and the design challenge is to optimize the initial conditions that lead to a target outcome. Recent developments in learned simulation have shown that graph neural networks (GNNs) can be used for accurate, efficient, differentiable estimation of simulator dynamics, and support high-quality design optimization with gradient- or sampling-based optimization procedures. However, optimizing designs from scratch requires many expensive model queries, and these procedures exhibit basic failures on either non-convex or high-dimensional problems.In this work, we show how denoising diffusion models (DDMs) can be used to solve inverse design problems efficiently and propose a particle sampling algorithm for further improving their efficiency. We perform experiments on a number of fluid dynamics design challenges, and find that our approach substantially reduces the number of calls to the simulator compared to standard techniques.

{{</citation>}}


### (45/114) Data-Juicer: A One-Stop Data Processing System for Large Language Models (Daoyuan Chen et al., 2023)

{{<citation>}}

Daoyuan Chen, Yilun Huang, Zhijian Ma, Hesen Chen, Xuchen Pan, Ce Ge, Dawei Gao, Yuexiang Xie, Zhaoyang Liu, Jinyang Gao, Yaliang Li, Bolin Ding, Jingren Zhou. (2023)  
**Data-Juicer: A One-Stop Data Processing System for Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-DB, cs-DC, cs-LG, cs.LG  
Keywords: GPT, GPT-4, LLaMA, Language Model  
[Paper Link](http://arxiv.org/abs/2309.02033v1)  

---


**ABSTRACT**  
The immense evolution in Large Language Models (LLMs) has underscored the importance of massive, diverse, and high-quality data. Despite this, existing open-source tools for LLM data processing remain limited and mostly tailored to specific datasets, with an emphasis on the reproducibility of released data over adaptability and usability, inhibiting potential applications. In response, we propose a one-stop, powerful yet flexible and user-friendly LLM data processing system named Data-Juicer. Our system offers over 50 built-in versatile operators and pluggable tools, which synergize modularity, composability, and extensibility dedicated to diverse LLM data processing needs. By incorporating visualized and automatic evaluation capabilities, Data-Juicer enables a timely feedback loop to accelerate data processing and gain data insights. To enhance usability, Data-Juicer provides out-of-the-box components for users with various backgrounds, and fruitful data recipes for LLM pre-training and post-tuning usages. Further, we employ multi-facet system optimization and seamlessly integrate Data-Juicer with both LLM and distributed computing ecosystems, to enable efficient and scalable data processing. Empirical validation of the generated data recipes reveals considerable improvements in LLaMA performance for various pre-training and post-tuning cases, demonstrating up to 7.45% relative improvement of averaged score across 16 LLM benchmarks and 16.25% higher win rate using pair-wise GPT-4 evaluation. The system's efficiency and scalability are also validated, supported by up to 88.7% reduction in single-machine processing time, 77.1% and 73.1% less memory and CPU usage respectively, and 7.91x processing acceleration when utilizing distributed computing ecosystems. Our system, data recipes, and multiple tutorial demos are released, calling for broader research centered on LLM data.

{{</citation>}}


### (46/114) Non-Parametric Representation Learning with Kernels (Pascal Esser et al., 2023)

{{<citation>}}

Pascal Esser, Maximilian Fleissner, Debarghya Ghoshdastidar. (2023)  
**Non-Parametric Representation Learning with Kernels**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.02028v1)  

---


**ABSTRACT**  
Unsupervised and self-supervised representation learning has become popular in recent years for learning useful features from unlabelled data. Representation learning has been mostly developed in the neural network literature, and other models for representation learning are surprisingly unexplored. In this work, we introduce and analyze several kernel-based representation learning approaches: Firstly, we define two kernel Self-Supervised Learning (SSL) models using contrastive loss functions and secondly, a Kernel Autoencoder (AE) model based on the idea of embedding and reconstructing data. We argue that the classical representer theorems for supervised kernel machines are not always applicable for (self-supervised) representation learning, and present new representer theorems, which show that the representations learned by our kernel models can be expressed in terms of kernel matrices. We further derive generalisation error bounds for representation learning with kernel SSL and AE, and empirically evaluate the performance of these methods in both small data regimes as well as in comparison with neural network based models.

{{</citation>}}


### (47/114) RDGSL: Dynamic Graph Representation Learning with Structure Learning (Siwei Zhang et al., 2023)

{{<citation>}}

Siwei Zhang, Yun Xiong, Yao Zhang, Yiheng Sun, Xi Chen, Yizhu Jiao, Yangyong Zhu. (2023)  
**RDGSL: Dynamic Graph Representation Learning with Structure Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Embedding, Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.02025v1)  

---


**ABSTRACT**  
Temporal Graph Networks (TGNs) have shown remarkable performance in learning representation for continuous-time dynamic graphs. However, real-world dynamic graphs typically contain diverse and intricate noise. Noise can significantly degrade the quality of representation generation, impeding the effectiveness of TGNs in downstream tasks. Though structure learning is widely applied to mitigate noise in static graphs, its adaptation to dynamic graph settings poses two significant challenges. i) Noise dynamics. Existing structure learning methods are ill-equipped to address the temporal aspect of noise, hampering their effectiveness in such dynamic and ever-changing noise patterns. ii) More severe noise. Noise may be introduced along with multiple interactions between two nodes, leading to the re-pollution of these nodes and consequently causing more severe noise compared to static graphs. In this paper, we present RDGSL, a representation learning method in continuous-time dynamic graphs. Meanwhile, we propose dynamic graph structure learning, a novel supervisory signal that empowers RDGSL with the ability to effectively combat noise in dynamic graphs. To address the noise dynamics issue, we introduce the Dynamic Graph Filter, where we innovatively propose a dynamic noise function that dynamically captures both current and historical noise, enabling us to assess the temporal aspect of noise and generate a denoised graph. We further propose the Temporal Embedding Learner to tackle the challenge of more severe noise, which utilizes an attention mechanism to selectively turn a blind eye to noisy edges and hence focus on normal edges, enhancing the expressiveness for representation generation that remains resilient to noise. Our method demonstrates robustness towards downstream tasks, resulting in up to 5.1% absolute AUC improvement in evolving classification versus the second-best baseline.

{{</citation>}}


### (48/114) iLoRE: Dynamic Graph Representation with Instant Long-term Modeling and Re-occurrence Preservation (Siwei Zhang et al., 2023)

{{<citation>}}

Siwei Zhang, Yun Xiong, Yao Zhang, Xixi Wu, Yiheng Sun, Jiawei Zhang. (2023)  
**iLoRE: Dynamic Graph Representation with Instant Long-term Modeling and Re-occurrence Preservation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SI, cs.LG  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2309.02012v1)  

---


**ABSTRACT**  
Continuous-time dynamic graph modeling is a crucial task for many real-world applications, such as financial risk management and fraud detection. Though existing dynamic graph modeling methods have achieved satisfactory results, they still suffer from three key limitations, hindering their scalability and further applicability. i) Indiscriminate updating. For incoming edges, existing methods would indiscriminately deal with them, which may lead to more time consumption and unexpected noisy information. ii) Ineffective node-wise long-term modeling. They heavily rely on recurrent neural networks (RNNs) as a backbone, which has been demonstrated to be incapable of fully capturing node-wise long-term dependencies in event sequences. iii) Neglect of re-occurrence patterns. Dynamic graphs involve the repeated occurrence of neighbors that indicates their importance, which is disappointedly neglected by existing methods. In this paper, we present iLoRE, a novel dynamic graph modeling method with instant node-wise Long-term modeling and Re-occurrence preservation. To overcome the indiscriminate updating issue, we introduce the Adaptive Short-term Updater module that will automatically discard the useless or noisy edges, ensuring iLoRE's effectiveness and instant ability. We further propose the Long-term Updater to realize more effective node-wise long-term modeling, where we innovatively propose the Identity Attention mechanism to empower a Transformer-based updater, bypassing the limited effectiveness of typical RNN-dominated designs. Finally, the crucial re-occurrence patterns are also encoded into a graph module for informative representation learning, which will further improve the expressiveness of our method. Our experimental results on real-world datasets demonstrate the effectiveness of our iLoRE for dynamic graph modeling.

{{</citation>}}


### (49/114) Representation Learning Dynamics of Self-Supervised Models (Pascal Esser et al., 2023)

{{<citation>}}

Pascal Esser, Satyaki Mukherjee, Debarghya Ghoshdastidar. (2023)  
**Representation Learning Dynamics of Self-Supervised Models**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.02011v1)  

---


**ABSTRACT**  
Self-Supervised Learning (SSL) is an important paradigm for learning representations from unlabelled data, and SSL with neural networks has been highly successful in practice. However current theoretical analysis of SSL is mostly restricted to generalisation error bounds. In contrast, learning dynamics often provide a precise characterisation of the behaviour of neural networks based models but, so far, are mainly known in supervised settings. In this paper, we study the learning dynamics of SSL models, specifically representations obtained by minimising contrastive and non-contrastive losses. We show that a naive extension of the dymanics of multivariate regression to SSL leads to learning trivial scalar representations that demonstrates dimension collapse in SSL. Consequently, we formulate SSL objectives with orthogonality constraints on the weights, and derive the exact (network width independent) learning dynamics of the SSL models trained using gradient descent on the Grassmannian manifold. We also argue that the infinite width approximation of SSL models significantly deviate from the neural tangent kernel approximations of supervised models. We numerically illustrate the validity of our theoretical findings, and discuss how the presented results provide a framework for further theoretical analysis of contrastive and non-contrastive SSL.

{{</citation>}}


### (50/114) Establishing a real-time traffic alarm in the city of Valencia with Deep Learning (Miguel Folgado et al., 2023)

{{<citation>}}

Miguel Folgado, Veronica Sanz, Johannes Hirn, Edgar Lorenzo-Saez, Javier Urchueguia. (2023)  
**Establishing a real-time traffic alarm in the city of Valencia with Deep Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.02010v1)  

---


**ABSTRACT**  
Urban traffic emissions represent a significant concern due to their detrimental impacts on both public health and the environment. Consequently, decision-makers have flagged their reduction as a crucial goal. In this study, we first analyze the correlation between traffic flux and pollution in the city of Valencia, Spain. Our results demonstrate that traffic has a significant impact on the levels of certain pollutants (especially $\text{NO}_\text{x}$). Secondly, we develop an alarm system to predict if a street is likely to experience unusually high traffic in the next 30 minutes, using an independent three-tier level for each street. To make the predictions, we use traffic data updated every 10 minutes and Long Short-Term Memory (LSTM) neural networks. We trained the LSTM using traffic data from 2018, and tested it using traffic data from 2019.

{{</citation>}}


### (51/114) An LSTM-Based Predictive Monitoring Method for Data with Time-varying Variability (Jiaqi Qiu et al., 2023)

{{<citation>}}

Jiaqi Qiu, Yu Lin, Inez Zwetsloot. (2023)  
**An LSTM-Based Predictive Monitoring Method for Data with Time-varying Variability**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ME  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.01978v1)  

---


**ABSTRACT**  
The recurrent neural network and its variants have shown great success in processing sequences in recent years. However, this deep neural network has not aroused much attention in anomaly detection through predictively process monitoring. Furthermore, the traditional statistic models work on assumptions and hypothesis tests, while neural network (NN) models do not need that many assumptions. This flexibility enables NN models to work efficiently on data with time-varying variability, a common inherent aspect of data in practice. This paper explores the ability of the recurrent neural network structure to monitor processes and proposes a control chart based on long short-term memory (LSTM) prediction intervals for data with time-varying variability. The simulation studies provide empirical evidence that the proposed model outperforms other NN-based predictive monitoring methods for mean shift detection. The proposed method is also applied to time series sensor data, which confirms that the proposed method is an effective technique for detecting abnormalities.

{{</citation>}}


### (52/114) OHQ: On-chip Hardware-aware Quantization (Wei Huang et al., 2023)

{{<citation>}}

Wei Huang, Haotong Qin, Yangdong Liu, Jingzhuo Liang, Yifu Ding, Ying Li, Xianglong Liu. (2023)  
**OHQ: On-chip Hardware-aware Quantization**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-AR, cs-LG, cs.LG  
Keywords: QA, Quantization  
[Paper Link](http://arxiv.org/abs/2309.01945v1)  

---


**ABSTRACT**  
Quantization emerges as one of the most promising approaches for deploying advanced deep models on resource-constrained hardware. Mixed-precision quantization leverages multiple bit-width architectures to unleash the accuracy and efficiency potential of quantized models. However, existing mixed-precision quantization suffers exhaustive search space that causes immense computational overhead. The quantization process thus relies on separate high-performance devices rather than locally, which also leads to a significant gap between the considered hardware metrics and the real deployment.In this paper, we propose an On-chip Hardware-aware Quantization (OHQ) framework that performs hardware-aware mixed-precision quantization without accessing online devices. First, we construct the On-chip Quantization Awareness (OQA) pipeline, enabling perceive the actual efficiency metrics of the quantization operator on the hardware.Second, we propose Mask-guided Quantization Estimation (MQE) technique to efficiently estimate the accuracy metrics of operators under the constraints of on-chip-level computing power.By synthesizing network and hardware insights through linear programming, we obtain optimized bit-width configurations. Notably, the quantization process occurs on-chip entirely without any additional computing devices and data access. We demonstrate accelerated inference after quantization for various architectures and compression ratios, achieving 70% and 73% accuracy for ResNet-18 and MobileNetV3, respectively. OHQ improves latency by 15~30% compared to INT8 on deployment.

{{</citation>}}


### (53/114) Developing A Fair Individualized Polysocial Risk Score (iPsRS) for Identifying Increased Social Risk of Hospitalizations in Patients with Type 2 Diabetes (T2D) (Yu Huang et al., 2023)

{{<citation>}}

Yu Huang, Jingchuan Guo, William T Donahoo, Zhengkang Fan, Ying Lu, Wei-Han Chen, Huilin Tang, Lori Bilello, Elizabeth A Shenkman, Jiang Bian. (2023)  
**Developing A Fair Individualized Polysocial Risk Score (iPsRS) for Identifying Increased Social Risk of Hospitalizations in Patients with Type 2 Diabetes (T2D)**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02467v1)  

---


**ABSTRACT**  
Background: Racial and ethnic minority groups and individuals facing social disadvantages, which often stem from their social determinants of health (SDoH), bear a disproportionate burden of type 2 diabetes (T2D) and its complications. It is therefore crucial to implement effective social risk management strategies at the point of care. Objective: To develop an EHR-based machine learning (ML) analytical pipeline to identify the unmet social needs associated with hospitalization risk in patients with T2D. Methods: We identified 10,192 T2D patients from the EHR data (from 2012 to 2022) from the University of Florida Health Integrated Data Repository, including contextual SDoH (e.g., neighborhood deprivation) and individual-level SDoH (e.g., housing stability). We developed an electronic health records (EHR)-based machine learning (ML) analytic pipeline, namely individualized polysocial risk score (iPsRS), to identify high social risk associated with hospitalizations in T2D patients, along with explainable AI (XAI) techniques and fairness assessment and optimization. Results: Our iPsRS achieved a C statistic of 0.72 in predicting 1-year hospitalization after fairness optimization across racial-ethnic groups. The iPsRS showed excellent utility for capturing individuals at high hospitalization risk; the actual 1-year hospitalization rate in the top 5% of iPsRS was ~13 times as high as the bottom decile. Conclusion: Our ML pipeline iPsRS can fairly and accurately screen for patients who have increased social risk leading to hospitalization in T2D patients.

{{</citation>}}


### (54/114) A Survey on Physics Informed Reinforcement Learning: Review and Open Problems (Chayan Banerjee et al., 2023)

{{<citation>}}

Chayan Banerjee, Kien Nguyen, Clinton Fookes, Maziar Raissi. (2023)  
**A Survey on Physics Informed Reinforcement Learning: Review and Open Problems**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.01909v1)  

---


**ABSTRACT**  
The inclusion of physical information in machine learning frameworks has revolutionized many application areas. This involves enhancing the learning process by incorporating physical constraints and adhering to physical laws. In this work we explore their utility for reinforcement learning applications. We present a thorough review of the literature on incorporating physics information, as known as physics priors, in reinforcement learning approaches, commonly referred to as physics-informed reinforcement learning (PIRL). We introduce a novel taxonomy with the reinforcement learning pipeline as the backbone to classify existing works, compare and contrast them, and derive crucial insights. Existing works are analyzed with regard to the representation/ form of the governing physics modeled for integration, their specific contribution to the typical reinforcement learning architecture, and their connection to the underlying reinforcement learning pipeline stages. We also identify core learning architectures and physics incorporation biases (i.e., observational, inductive and learning) of existing PIRL approaches and use them to further categorize the works for better understanding and adaptation. By providing a comprehensive perspective on the implementation of the physics-informed capability, the taxonomy presents a cohesive approach to PIRL. It identifies the areas where this approach has been applied, as well as the gaps and opportunities that exist. Additionally, the taxonomy sheds light on unresolved issues and challenges, which can guide future research. This nascent field holds great potential for enhancing reinforcement learning algorithms by increasing their physical plausibility, precision, data efficiency, and applicability in real-world scenarios.

{{</citation>}}


## cs.SI (1)



### (55/114) A Social Network Approach to Analyzing Token Properties and Abnormal Events in Decentralized Exchanges (Aryan Soltani Mohammadi et al., 2023)

{{<citation>}}

Aryan Soltani Mohammadi, Moein Karami, Amir Pasha Motamed, Behnam Bahrak. (2023)  
**A Social Network Approach to Analyzing Token Properties and Abnormal Events in Decentralized Exchanges**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2309.02579v1)  

---


**ABSTRACT**  
The properties of tokens within the Ethereum blockchain, such as their current prices, trade volumes, and potential future values, have been subjects of numerous studies. The complex interaction of the variables related to tokens makes analyzing them challenging. Employing social networks, a powerful tool for modeling connections within groups or communities, can provide valuable guidance. This study mainly focuses on creating and examining networks related to two major decentralized exchanges: Uniswap Version 2 and SushiSwap. We discovered that the distribution of links to nodes follow a power law making them scale-free networks. Additionally, during our analysis, we made an intresting discovery: the centrality of tokens in exchange graphs provide valuable insights into their value and significance in the world of cryptocurrencies. By observing changes in centrality over time, we uncovered noteworthy events in the cryptocurrency domain, that shows the potential of this networks for extracting information about the exchanges.

{{</citation>}}


## eess.AS (2)



### (56/114) Symbolic Music Representations for Classification Tasks: A Systematic Evaluation (Huan Zhang et al., 2023)

{{<citation>}}

Huan Zhang, Emmanouil Karystinaios, Simon Dixon, Gerhard Widmer, Carlos Eduardo Cancino-Chacón. (2023)  
**Symbolic Music Representations for Classification Tasks: A Systematic Evaluation**  

---
Primary Category: eess.AS  
Categories: cs-MM, cs-SD, eess-AS, eess.AS  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2309.02567v1)  

---


**ABSTRACT**  
Music Information Retrieval (MIR) has seen a recent surge in deep learning-based approaches, which often involve encoding symbolic music (i.e., music represented in terms of discrete note events) in an image-like or language like fashion. However, symbolic music is neither an image nor a sentence, and research in the symbolic domain lacks a comprehensive overview of the different available representations. In this paper, we investigate matrix (piano roll), sequence, and graph representations and their corresponding neural architectures, in combination with symbolic scores and performances on three piece-level classification tasks. We also introduce a novel graph representation for symbolic performances and explore the capability of graph representations in global classification tasks. Our systematic evaluation shows advantages and limitations of each input representation. Our results suggest that the graph representation, as the newest and least explored among the three approaches, exhibits promising performance, while being more light-weight in training.

{{</citation>}}


### (57/114) Personalized Adaptation with Pre-trained Speech Encoders for Continuous Emotion Recognition (Minh Tran et al., 2023)

{{<citation>}}

Minh Tran, Yufeng Yin, Mohammad Soleymani. (2023)  
**Personalized Adaptation with Pre-trained Speech Encoders for Continuous Emotion Recognition**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess-SP, eess.AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2309.02418v1)  

---


**ABSTRACT**  
There are individual differences in expressive behaviors driven by cultural norms and personality. This between-person variation can result in reduced emotion recognition performance. Therefore, personalization is an important step in improving the generalization and robustness of speech emotion recognition. In this paper, to achieve unsupervised personalized emotion recognition, we first pre-train an encoder with learnable speaker embeddings in a self-supervised manner to learn robust speech representations conditioned on speakers. Second, we propose an unsupervised method to compensate for the label distribution shifts by finding similar speakers and leveraging their label distributions from the training set. Extensive experimental results on the MSP-Podcast corpus indicate that our method consistently outperforms strong personalization baselines and achieves state-of-the-art performance for valence estimation.

{{</citation>}}


## cs.RO (5)



### (58/114) Physically Grounded Vision-Language Models for Robotic Manipulation (Jensen Gao et al., 2023)

{{<citation>}}

Jensen Gao, Bidipta Sarkar, Fei Xia, Ted Xiao, Jiajun Wu, Brian Ichter, Anirudha Majumdar, Dorsa Sadigh. (2023)  
**Physically Grounded Vision-Language Models for Robotic Manipulation**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-RO, cs.RO  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.02561v1)  

---


**ABSTRACT**  
Recent advances in vision-language models (VLMs) have led to improved performance on tasks such as visual question answering and image captioning. Consequently, these models are now well-positioned to reason about the physical world, particularly within domains such as robotic manipulation. However, current VLMs are limited in their understanding of the physical concepts (e.g., material, fragility) of common objects, which restricts their usefulness for robotic manipulation tasks that involve interaction and physical reasoning about such objects. To address this limitation, we propose PhysObjects, an object-centric dataset of 36.9K crowd-sourced and 417K automated physical concept annotations of common household objects. We demonstrate that fine-tuning a VLM on PhysObjects improves its understanding of physical object concepts, by capturing human priors of these concepts from visual appearance. We incorporate this physically-grounded VLM in an interactive framework with a large language model-based robotic planner, and show improved planning performance on tasks that require reasoning about physical object concepts, compared to baselines that do not leverage physically-grounded VLMs. We additionally illustrate the benefits of our physically-grounded VLM on a real robot, where it improves task success rates. We release our dataset and provide further details and visualizations of our results at https://iliad.stanford.edu/pg-vlm/.

{{</citation>}}


### (59/114) Structural Concept Learning via Graph Attention for Multi-Level Rearrangement Planning (Manav Kulshrestha et al., 2023)

{{<citation>}}

Manav Kulshrestha, Ahmed H. Qureshi. (2023)  
**Structural Concept Learning via Graph Attention for Multi-Level Rearrangement Planning**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-LG, cs-RO, cs.RO  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.02547v1)  

---


**ABSTRACT**  
Robotic manipulation tasks, such as object rearrangement, play a crucial role in enabling robots to interact with complex and arbitrary environments. Existing work focuses primarily on single-level rearrangement planning and, even if multiple levels exist, dependency relations among substructures are geometrically simpler, like tower stacking. We propose Structural Concept Learning (SCL), a deep learning approach that leverages graph attention networks to perform multi-level object rearrangement planning for scenes with structural dependency hierarchies. It is trained on a self-generated simulation data set with intuitive structures, works for unseen scenes with an arbitrary number of objects and higher complexity of structures, infers independent substructures to allow for task parallelization over multiple manipulators, and generalizes to the real world. We compare our method with a range of classical and model-based baselines to show that our method leverages its scene understanding to achieve better performance, flexibility, and efficiency. The dataset, supplementary details, videos, and code implementation are available at: https://manavkulshrestha.github.io/scl

{{</citation>}}


### (60/114) Graph-Based Interaction-Aware Multimodal 2D Vehicle Trajectory Prediction using Diffusion Graph Convolutional Networks (Keshu Wu et al., 2023)

{{<citation>}}

Keshu Wu, Yang Zhou, Haotian Shi, Xiaopeng Li, Bin Ran. (2023)  
**Graph-Based Interaction-Aware Multimodal 2D Vehicle Trajectory Prediction using Diffusion Graph Convolutional Networks**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-GR, cs-RO, cs.RO  
Keywords: Graph Convolutional Network  
[Paper Link](http://arxiv.org/abs/2309.01981v1)  

---


**ABSTRACT**  
Predicting vehicle trajectories is crucial for ensuring automated vehicle operation efficiency and safety, particularly on congested multi-lane highways. In such dynamic environments, a vehicle's motion is determined by its historical behaviors as well as interactions with surrounding vehicles. These intricate interactions arise from unpredictable motion patterns, leading to a wide range of driving behaviors that warrant in-depth investigation. This study presents the Graph-based Interaction-aware Multi-modal Trajectory Prediction (GIMTP) framework, designed to probabilistically predict future vehicle trajectories by effectively capturing these interactions. Within this framework, vehicles' motions are conceptualized as nodes in a time-varying graph, and the traffic interactions are represented by a dynamic adjacency matrix. To holistically capture both spatial and temporal dependencies embedded in this dynamic adjacency matrix, the methodology incorporates the Diffusion Graph Convolutional Network (DGCN), thereby providing a graph embedding of both historical states and future states. Furthermore, we employ a driving intention-specific feature fusion, enabling the adaptive integration of historical and future embeddings for enhanced intention recognition and trajectory prediction. This model gives two-dimensional predictions for each mode of longitudinal and lateral driving behaviors and offers probabilistic future paths with corresponding probabilities, addressing the challenges of complex vehicle interactions and multi-modality of driving behaviors. Validation using real-world trajectory datasets demonstrates the efficiency and potential.

{{</citation>}}


### (61/114) RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking (Homanga Bharadhwaj et al., 2023)

{{<citation>}}

Homanga Bharadhwaj, Jay Vakil, Mohit Sharma, Abhinav Gupta, Shubham Tulsiani, Vikash Kumar. (2023)  
**RoboAgent: Generalization and Efficiency in Robot Manipulation via Semantic Augmentations and Action Chunking**  

---
Primary Category: cs.RO  
Categories: cs-LG, cs-RO, cs.RO  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.01918v1)  

---


**ABSTRACT**  
The grand aim of having a single robot that can manipulate arbitrary objects in diverse settings is at odds with the paucity of robotics datasets. Acquiring and growing such datasets is strenuous due to manual efforts, operational costs, and safety challenges. A path toward such an universal agent would require a structured framework capable of wide generalization but trained within a reasonable data budget. In this paper, we develop an efficient system (RoboAgent) for training universal agents capable of multi-task manipulation skills using (a) semantic augmentations that can rapidly multiply existing datasets and (b) action representations that can extract performant policies with small yet diverse multi-modal datasets without overfitting. In addition, reliable task conditioning and an expressive policy architecture enable our agent to exhibit a diverse repertoire of skills in novel situations specified using language commands. Using merely 7500 demonstrations, we are able to train a single agent capable of 12 unique skills, and demonstrate its generalization over 38 tasks spread across common daily activities in diverse kitchen scenes. On average, RoboAgent outperforms prior methods by over 40% in unseen situations while being more sample efficient and being amenable to capability improvements and extensions through fine-tuning. Videos at https://robopen.github.io/

{{</citation>}}


### (62/114) Improving Drone Imagery For Computer Vision/Machine Learning in Wilderness Search and Rescue (Robin Murphy et al., 2023)

{{<citation>}}

Robin Murphy, Thomas Manzini. (2023)  
**Improving Drone Imagery For Computer Vision/Machine Learning in Wilderness Search and Rescue**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Computer Vision, Drone  
[Paper Link](http://arxiv.org/abs/2309.01904v1)  

---


**ABSTRACT**  
This paper describes gaps in acquisition of drone imagery that impair the use with computer vision/machine learning (CV/ML) models and makes five recommendations to maximize image suitability for CV/ML post-processing. It describes a notional work process for the use of drones in wilderness search and rescue incidents. The large volume of data from the wide area search phase offers the greatest opportunity for CV/ML techniques because of the large number of images that would otherwise have to be manually inspected. The 2023 Wu-Murad search in Japan, one of the largest missing person searches conducted in that area, serves as a case study. Although drone teams conducting wide area searches may not know in advance if the data they collect is going to be used for CV/ML post-processing, there are data collection procedures that can improve the search in general with automated collection software. If the drone teams do expect to use CV/ML, then they can exploit knowledge about the model to further optimize flights.

{{</citation>}}


## cs.CL (20)



### (63/114) Automating Behavioral Testing in Machine Translation (Javier Ferrando et al., 2023)

{{<citation>}}

Javier Ferrando, Matthias Sperber, Hendra Setiawan, Dominic Telaar, Saša Hasan. (2023)  
**Automating Behavioral Testing in Machine Translation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, Machine Translation, NLP  
[Paper Link](http://arxiv.org/abs/2309.02553v1)  

---


**ABSTRACT**  
Behavioral testing in NLP allows fine-grained evaluation of systems by examining their linguistic capabilities through the analysis of input-output behavior. Unfortunately, existing work on behavioral testing in Machine Translation (MT) is currently restricted to largely handcrafted tests covering a limited range of capabilities and languages. To address this limitation, we propose to use Large Language Models (LLMs) to generate a diverse set of source sentences tailored to test the behavior of MT models in a range of situations. We can then verify whether the MT model exhibits the expected behavior through matching candidate sets that are also generated using LLMs. Our approach aims to make behavioral testing of MT systems practical while requiring only minimal human effort. In our experiments, we apply our proposed evaluation framework to assess multiple available MT systems, revealing that while in general pass-rates follow the trends observable from traditional accuracy-based metrics, our method was able to uncover several important differences and potential bugs that go unnoticed when relying only on accuracy.

{{</citation>}}


### (64/114) Substitution-based Semantic Change Detection using Contextual Embeddings (Dallas Card, 2023)

{{<citation>}}

Dallas Card. (2023)  
**Substitution-based Semantic Change Detection using Contextual Embeddings**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2309.02403v2)  

---


**ABSTRACT**  
Measuring semantic change has thus far remained a task where methods using contextual embeddings have struggled to improve upon simpler techniques relying only on static word vectors. Moreover, many of the previously proposed approaches suffer from downsides related to scalability and ease of interpretation. We present a simplified approach to measuring semantic change using contextual embeddings, relying only on the most probable substitutes for masked terms. Not only is this approach directly interpretable, it is also far more efficient in terms of storage, achieves superior average performance across the most frequently cited datasets for this task, and allows for more nuanced investigation of change than is possible with static word vectors.

{{</citation>}}


### (65/114) nanoT5: A PyTorch Framework for Pre-training and Fine-tuning T5-style Models with Limited Resources (Piotr Nawrot, 2023)

{{<citation>}}

Piotr Nawrot. (2023)  
**nanoT5: A PyTorch Framework for Pre-training and Fine-tuning T5-style Models with Limited Resources**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, T5  
[Paper Link](http://arxiv.org/abs/2309.02373v1)  

---


**ABSTRACT**  
State-of-the-art language models like T5 have revolutionized the NLP landscape, but their computational demands hinder a large portion of the research community. To address this challenge, we present nanoT5, a specially-optimized PyTorch framework for efficient pre-training and fine-tuning of T5 models. Drawing on insights from optimizer differences and prioritizing efficiency, nanoT5 allows a T5-Base model to be pre-trained on a single GPU in just 16 hours, without any loss in performance. With the introduction of this open-source framework, we hope to widen the accessibility to language modelling research and cater to the community's demand for more user-friendly T5 (Encoder-Decoder) implementations. Our contributions, including configurations, codebase, software/hardware insights, and pre-trained models, are available to the public, aiming to strike a balance between research accessibility and resource constraints in NLP.

{{</citation>}}


### (66/114) Weigh Your Own Words: Improving Hate Speech Counter Narrative Generation via Attention Regularization (Helena Bonaldi et al., 2023)

{{<citation>}}

Helena Bonaldi, Giuseppe Attanasio, Debora Nozza, Marco Guerini. (2023)  
**Weigh Your Own Words: Improving Hate Speech Counter Narrative Generation via Attention Regularization**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Attention, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2309.02311v1)  

---


**ABSTRACT**  
Recent computational approaches for combating online hate speech involve the automatic generation of counter narratives by adapting Pretrained Transformer-based Language Models (PLMs) with human-curated data. This process, however, can produce in-domain overfitting, resulting in models generating acceptable narratives only for hatred similar to training data, with little portability to other targets or to real-world toxic language. This paper introduces novel attention regularization methodologies to improve the generalization capabilities of PLMs for counter narratives generation. Overfitting to training-specific terms is then discouraged, resulting in more diverse and richer narratives. We experiment with two attention-based regularization techniques on a benchmark English dataset. Regularized models produce better counter narratives than state-of-the-art approaches in most cases, both in terms of automatic metrics and human evaluation, especially when hateful targets are not present in the training data. This work paves the way for better and more flexible counter-speech generation models, a task for which datasets are highly challenging to produce.

{{</citation>}}


### (67/114) Dialog Action-Aware Transformer for Dialog Policy Learning (Huimin Wang et al., 2023)

{{<citation>}}

Huimin Wang, Wai-Chung Kwan, Kam-Fai Wong. (2023)  
**Dialog Action-Aware Transformer for Dialog Policy Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Transformer  
[Paper Link](http://arxiv.org/abs/2309.02240v1)  

---


**ABSTRACT**  
Recent works usually address Dialog policy learning DPL by training a reinforcement learning (RL) agent to determine the best dialog action. However, existing works on deep RL require a large volume of agent-user interactions to achieve acceptable performance. In this paper, we propose to make full use of the plain text knowledge from the pre-trained language model to accelerate the RL agent's learning speed. Specifically, we design a dialog action-aware transformer encoder (DaTrans), which integrates a new fine-tuning procedure named masked last action task to encourage DaTrans to be dialog-aware and distils action-specific features. Then, DaTrans is further optimized in an RL setting with ongoing interactions and evolves through exploration in the dialog action space toward maximizing long-term accumulated rewards. The effectiveness and efficiency of the proposed model are demonstrated with both simulator evaluation and human evaluation.

{{</citation>}}


### (68/114) Augmenting Black-box LLMs with Medical Textbooks for Clinical Question Answering (Yubo Wang et al., 2023)

{{<citation>}}

Yubo Wang, Xueguang Ma, Wenhu Chen. (2023)  
**Augmenting Black-box LLMs with Medical Textbooks for Clinical Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: ChatGPT, Clinical, GPT, Language Model, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.02233v1)  

---


**ABSTRACT**  
Large-scale language models (LLMs), such as ChatGPT, are capable of generating human-like responses for various downstream tasks, such as task-oriented dialogues and question answering. However, applying LLMs to medical domains remains challenging due to their inability to leverage domain-specific knowledge. In this study, we present the Large-scale Language Models Augmented with Medical Textbooks (LLM-AMT), which integrates authoritative medical textbooks as the cornerstone of its design, enhancing its proficiency in the specialized domain through plug-and-play modules, comprised of a Hybrid Textbook Retriever, supplemented by the Query Augmenter and the LLM Reader. Experimental evaluation on three open-domain medical question-answering tasks reveals a substantial enhancement in both the professionalism and accuracy of the LLM responses when utilizing LLM-AMT, exhibiting an improvement ranging from 11.4% to 13.2%. Despite being 100 times smaller, we found that medical textbooks as the retrieval corpus serves as a more valuable external knowledge source than Wikipedia in the medical domain. Our experiments show that textbook augmentation results in a performance improvement ranging from 9.7% to 12.2% over Wikipedia augmentation.

{{</citation>}}


### (69/114) Leveraging BERT Language Models for Multi-Lingual ESG Issue Identification (Elvys Linhares Pontes et al., 2023)

{{<citation>}}

Elvys Linhares Pontes, Mohamed Benjannet, Lam Kim Ming. (2023)  
**Leveraging BERT Language Models for Multi-Lingual ESG Issue Identification**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.02189v1)  

---


**ABSTRACT**  
Environmental, Social, and Governance (ESG) has been used as a metric to measure the negative impacts and enhance positive outcomes of companies in areas such as the environment, society, and governance. Recently, investors have increasingly recognized the significance of ESG criteria in their investment choices, leading businesses to integrate ESG principles into their operations and strategies. The Multi-Lingual ESG Issue Identification (ML-ESG) shared task encompasses the classification of news documents into 35 distinct ESG issue labels. In this study, we explored multiple strategies harnessing BERT language models to achieve accurate classification of news documents across these labels. Our analysis revealed that the RoBERTa classifier emerged as one of the most successful approaches, securing the second-place position for the English test dataset, and sharing the fifth-place position for the French test dataset. Furthermore, our SVM-based binary model tailored for the Chinese language exhibited exceptional performance, earning the second-place rank on the test dataset.

{{</citation>}}


### (70/114) Incorporating Dictionaries into a Neural Network Architecture to Extract COVID-19 Medical Concepts From Social Media (Abul Hasan et al., 2023)

{{<citation>}}

Abul Hasan, Mark Levene, David Weston. (2023)  
**Incorporating Dictionaries into a Neural Network Architecture to Extract COVID-19 Medical Concepts From Social Media**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SI, cs.CL  
Keywords: BERT, Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2309.02188v1)  

---


**ABSTRACT**  
We investigate the potential benefit of incorporating dictionary information into a neural network architecture for natural language processing. In particular, we make use of this architecture to extract several concepts related to COVID-19 from an on-line medical forum. We use a sample from the forum to manually curate one dictionary for each concept. In addition, we use MetaMap, which is a tool for extracting biomedical concepts, to identify a small number of semantic concepts. For a supervised concept extraction task on the forum data, our best model achieved a macro $F_1$ score of 90\%. A major difficulty in medical concept extraction is obtaining labelled data from which to build supervised models. We investigate the utility of our models to transfer to data derived from a different source in two ways. First for producing labels via weak learning and second to perform concept extraction. The dataset we use in this case comprises COVID-19 related tweets and we achieve an $F_1$ score 81\% for symptom concept extraction trained on weakly labelled data. The utility of our dictionaries is compared with a COVID-19 symptom dictionary that was constructed directly from Twitter. Further experiments that incorporate BERT and a COVID-19 version of BERTweet demonstrate that the dictionaries provide a commensurate result. Our results show that incorporating small domain dictionaries to deep learning models can improve concept extraction tasks. Moreover, models built using dictionaries generalize well and are transferable to different datasets on a similar task.

{{</citation>}}


### (71/114) Advancing Text-to-GLOSS Neural Translation Using a Novel Hyper-parameter Optimization Technique (Younes Ouargani et al., 2023)

{{<citation>}}

Younes Ouargani, Noussaima El Khattabi. (2023)  
**Advancing Text-to-GLOSS Neural Translation Using a Novel Hyper-parameter Optimization Technique**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-CL, cs.CL  
Keywords: BLEU, Machine Translation  
[Paper Link](http://arxiv.org/abs/2309.02162v1)  

---


**ABSTRACT**  
In this paper, we investigate the use of transformers for Neural Machine Translation of text-to-GLOSS for Deaf and Hard-of-Hearing communication. Due to the scarcity of available data and limited resources for text-to-GLOSS translation, we treat the problem as a low-resource language task. We use our novel hyper-parameter exploration technique to explore a variety of architectural parameters and build an optimal transformer-based architecture specifically tailored for text-to-GLOSS translation. The study aims to improve the accuracy and fluency of Neural Machine Translation generated GLOSS. This is achieved by examining various architectural parameters including layer count, attention heads, embedding dimension, dropout, and label smoothing to identify the optimal architecture for improving text-to-GLOSS translation performance. The experiments conducted on the PHOENIX14T dataset reveal that the optimal transformer architecture outperforms previous work on the same dataset. The best model reaches a ROUGE (Recall-Oriented Understudy for Gisting Evaluation) score of 55.18% and a BLEU-1 (BiLingual Evaluation Understudy 1) score of 63.6%, outperforming state-of-the-art results on the BLEU1 and ROUGE score by 8.42 and 0.63 respectively.

{{</citation>}}


### (72/114) Bring the Noise: Introducing Noise Robustness to Pretrained Automatic Speech Recognition (Patrick Eickhoff et al., 2023)

{{<citation>}}

Patrick Eickhoff, Matthias Möller, Theresa Pekarek Rosin, Johannes Twiefel, Stefan Wermter. (2023)  
**Bring the Noise: Introducing Noise Robustness to Pretrained Automatic Speech Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.02145v1)  

---


**ABSTRACT**  
In recent research, in the domain of speech processing, large End-to-End (E2E) systems for Automatic Speech Recognition (ASR) have reported state-of-the-art performance on various benchmarks. These systems intrinsically learn how to handle and remove noise conditions from speech. Previous research has shown, that it is possible to extract the denoising capabilities of these models into a preprocessor network, which can be used as a frontend for downstream ASR models. However, the proposed methods were limited to specific fully convolutional architectures. In this work, we propose a novel method to extract the denoising capabilities, that can be applied to any encoder-decoder architecture. We propose the Cleancoder preprocessor architecture that extracts hidden activations from the Conformer ASR model and feeds them to a decoder to predict denoised spectrograms. We train our pre-processor on the Noisy Speech Database (NSD) to reconstruct denoised spectrograms from noisy inputs. Then, we evaluate our model as a frontend to a pretrained Conformer ASR model as well as a frontend to train smaller Conformer ASR models from scratch. We show that the Cleancoder is able to filter noise from speech and that it improves the total Word Error Rate (WER) of the downstream model in noisy conditions for both applications.

{{</citation>}}


### (73/114) Making Large Language Models Better Reasoners with Alignment (Peiyi Wang et al., 2023)

{{<citation>}}

Peiyi Wang, Lei Li, Liang Chen, Feifan Song, Binghuai Lin, Yunbo Cao, Tianyu Liu, Zhifang Sui. (2023)  
**Making Large Language Models Better Reasoners with Alignment**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.02144v1)  

---


**ABSTRACT**  
Reasoning is a cognitive process of using evidence to reach a sound conclusion. The reasoning capability is essential for large language models (LLMs) to serve as the brain of the artificial general intelligence agent. Recent studies reveal that fine-tuning LLMs on data with the chain of thought (COT) reasoning process can significantly enhance their reasoning capabilities. However, we find that the fine-tuned LLMs suffer from an \textit{Assessment Misalignment} problem, i.e., they frequently assign higher scores to subpar COTs, leading to potential limitations in their reasoning abilities. To address this problem, we introduce an \textit{Alignment Fine-Tuning (AFT)} paradigm, which involves three steps: 1) fine-tuning LLMs with COT training data; 2) generating multiple COT responses for each question, and categorizing them into positive and negative ones based on whether they achieve the correct answer; 3) calibrating the scores of positive and negative responses given by LLMs with a novel constraint alignment loss. Specifically, the constraint alignment loss has two objectives: a) Alignment, which guarantees that positive scores surpass negative scores to encourage answers with high-quality COTs; b) Constraint, which keeps the negative scores confined to a reasonable range to prevent the model degradation. Beyond just the binary positive and negative feedback, the constraint alignment loss can be seamlessly adapted to the ranking situations when ranking feedback is accessible. Furthermore, we also delve deeply into recent ranking-based alignment methods, such as DPO, RRHF, and PRO, and discover that the constraint, which has been overlooked by these approaches, is also crucial for their performance. Extensive experiments on four reasoning benchmarks with both binary and ranking feedback demonstrate the effectiveness of AFT.

{{</citation>}}


### (74/114) Leveraging Label Information for Multimodal Emotion Recognition (Peiying Wang et al., 2023)

{{<citation>}}

Peiying Wang, Sunlu Zeng, Junqing Chen, Lu Fan, Meng Chen, Youzheng Wu, Xiaodong He. (2023)  
**Leveraging Label Information for Multimodal Emotion Recognition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2309.02106v1)  

---


**ABSTRACT**  
Multimodal emotion recognition (MER) aims to detect the emotional status of a given expression by combining the speech and text information. Intuitively, label information should be capable of helping the model locate the salient tokens/frames relevant to the specific emotion, which finally facilitates the MER task. Inspired by this, we propose a novel approach for MER by leveraging label information. Specifically, we first obtain the representative label embeddings for both text and speech modalities, then learn the label-enhanced text/speech representations for each utterance via label-token and label-frame interactions. Finally, we devise a novel label-guided attentive fusion module to fuse the label-aware text and speech representations for emotion classification. Extensive experiments were conducted on the public IEMOCAP dataset, and experimental results demonstrate that our proposed approach outperforms existing baselines and achieves new state-of-the-art performance.

{{</citation>}}


### (75/114) Improving Query-Focused Meeting Summarization with Query-Relevant Knowledge (Tiezheng Yu et al., 2023)

{{<citation>}}

Tiezheng Yu, Ziwei Ji, Pascale Fung. (2023)  
**Improving Query-Focused Meeting Summarization with Query-Relevant Knowledge**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2309.02105v1)  

---


**ABSTRACT**  
Query-Focused Meeting Summarization (QFMS) aims to generate a summary of a given meeting transcript conditioned upon a query. The main challenges for QFMS are the long input text length and sparse query-relevant information in the meeting transcript. In this paper, we propose a knowledge-enhanced two-stage framework called Knowledge-Aware Summarizer (KAS) to tackle the challenges. In the first stage, we introduce knowledge-aware scores to improve the query-relevant segment extraction. In the second stage, we incorporate query-relevant knowledge in the summary generation. Experimental results on the QMSum dataset show that our approach achieves state-of-the-art performance. Further analysis proves the competency of our methods in generating relevant and faithful summaries.

{{</citation>}}


### (76/114) Bridging Emotion Role Labeling and Appraisal-based Emotion Analysis (Roman Klinger, 2023)

{{<citation>}}

Roman Klinger. (2023)  
**Bridging Emotion Role Labeling and Appraisal-based Emotion Analysis**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2309.02092v1)  

---


**ABSTRACT**  
The term emotion analysis in text subsumes various natural language processing tasks which have in common the goal to enable computers to understand emotions. Most popular is emotion classification in which one or multiple emotions are assigned to a predefined textual unit. While such setting is appropriate to identify the reader's or author's emotion, emotion role labeling adds the perspective of mentioned entities and extracts text spans that correspond to the emotion cause. The underlying emotion theories agree on one important point; that an emotion is caused by some internal or external event and comprises several subcomponents, including the subjective feeling and a cognitive evaluation. We therefore argue that emotions and events are related in two ways. (1) Emotions are events; and this perspective is the fundament in NLP for emotion role labeling. (2) Emotions are caused by events; a perspective that is made explicit with research how to incorporate psychological appraisal theories in NLP models to interpret events. These two research directions, role labeling and (event-focused) emotion classification, have by and large been tackled separately. We contributed to both directions with the projects SEAT (Structured Multi-Domain Emotion Analysis from Text) and CEAT (Computational Event Evaluation based on Appraisal Theories for Emotion Analysis), both funded by the German Research Foundation. In this paper, we consolidate the findings and point out open research questions.

{{</citation>}}


### (77/114) An Automatic Evaluation Framework for Multi-turn Medical Consultations Capabilities of Large Language Models (Yusheng Liao et al., 2023)

{{<citation>}}

Yusheng Liao, Yutong Meng, Hongcheng Liu, Yanfeng Wang, Yu Wang. (2023)  
**An Automatic Evaluation Framework for Multi-turn Medical Consultations Capabilities of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.02077v1)  

---


**ABSTRACT**  
Large language models (LLMs) have achieved significant success in interacting with human. However, recent studies have revealed that these models often suffer from hallucinations, leading to overly confident but incorrect judgments. This limits their application in the medical domain, where tasks require the utmost accuracy. This paper introduces an automated evaluation framework that assesses the practical capabilities of LLMs as virtual doctors during multi-turn consultations. Consultation tasks are designed to require LLMs to be aware of what they do not know, to inquire about missing medical information from patients, and to ultimately make diagnoses. To evaluate the performance of LLMs for these tasks, a benchmark is proposed by reformulating medical multiple-choice questions from the United States Medical Licensing Examinations (USMLE), and comprehensive evaluation metrics are developed and evaluated on three constructed test sets. A medical consultation training set is further constructed to improve the consultation ability of LLMs. The results of the experiments show that fine-tuning with the training set can alleviate hallucinations and improve LLMs' performance on the proposed benchmark. Extensive experiments and ablation studies are conducted to validate the effectiveness and robustness of the proposed framework.

{{</citation>}}


### (78/114) Enhance Multi-domain Sentiment Analysis of Review Texts through Prompting Strategies (Yajing Wang et al., 2023)

{{<citation>}}

Yajing Wang, Zongwei Luo. (2023)  
**Enhance Multi-domain Sentiment Analysis of Review Texts through Prompting Strategies**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model, Sentiment Analysis  
[Paper Link](http://arxiv.org/abs/2309.02045v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have made significant strides in both scientific research and practical applications. Existing studies have demonstrated the state-of-the-art (SOTA) performance of LLMs in various natural language processing tasks. However, the question of how to further enhance LLMs' performance in specific task using prompting strategies remains a pivotal concern. This paper explores the enhancement of LLMs' performance in sentiment analysis through the application of prompting strategies. We formulate the process of prompting for sentiment analysis tasks and introduce two novel strategies tailored for sentiment analysis: RolePlaying (RP) prompting and Chain-of-thought (CoT) prompting. Specifically, we also propose the RP-CoT prompting strategy which is a combination of RP prompting and CoT prompting. We conduct comparative experiments on three distinct domain datasets to evaluate the effectiveness of the proposed sentiment analysis strategies. The results demonstrate that the adoption of the proposed prompting strategies leads to a increasing enhancement in sentiment analysis accuracy. Further, the CoT prompting strategy exhibits a notable impact on implicit sentiment analysis, with the RP-CoT prompting strategy delivering the most superior performance among all strategies.

{{</citation>}}


### (79/114) Bilevel Scheduled Sampling for Dialogue Generation (Jiawen Liu et al., 2023)

{{<citation>}}

Jiawen Liu, Kan Li. (2023)  
**Bilevel Scheduled Sampling for Dialogue Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Dialog, Dialogue  
[Paper Link](http://arxiv.org/abs/2309.01953v1)  

---


**ABSTRACT**  
Exposure bias poses a common challenge in numerous natural language processing tasks, particularly in the dialog generation. In response to this issue, researchers have devised various techniques, among which scheduled sampling has proven to be an effective method for mitigating exposure bias. However, the existing state-of-the-art scheduled sampling methods solely consider the current sampling words' quality for threshold truncation sampling, which overlooks the importance of sentence-level information and the method of threshold truncation warrants further discussion. In this paper, we propose a bilevel scheduled sampling model that takes the sentence-level information into account and incorporates it with word-level quality. To enhance sampling diversity and improve the model's adaptability, we propose a smooth function that maps the combined result of sentence-level and word-level information to an appropriate range, and employ probabilistic sampling based on the mapped values instead of threshold truncation. Experiments conducted on the DailyDialog and PersonaChat datasets demonstrate the effectiveness of our proposed methods, which significantly alleviate the exposure bias problem and outperform state-of-the-art scheduled sampling methods.

{{</citation>}}


### (80/114) TODM: Train Once Deploy Many Efficient Supernet-Based RNN-T Compression For On-device ASR Models (Yuan Shangguan et al., 2023)

{{<citation>}}

Yuan Shangguan, Haichuan Yang, Danni Li, Chunyang Wu, Yassir Fathullah, Dilin Wang, Ayushi Dalmia, Raghuraman Krishnamoorthi, Ozlem Kalinli, Junteng Jia, Jay Mahadeokar, Xin Lei, Mike Seltzer, Vikas Chandra. (2023)  
**TODM: Train Once Deploy Many Efficient Supernet-Based RNN-T Compression For On-device ASR Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2309.01947v1)  

---


**ABSTRACT**  
Automatic Speech Recognition (ASR) models need to be optimized for specific hardware before they can be deployed on devices. This can be done by tuning the model's hyperparameters or exploring variations in its architecture. Re-training and re-validating models after making these changes can be a resource-intensive task. This paper presents TODM (Train Once Deploy Many), a new approach to efficiently train many sizes of hardware-friendly on-device ASR models with comparable GPU-hours to that of a single training job. TODM leverages insights from prior work on Supernet, where Recurrent Neural Network Transducer (RNN-T) models share weights within a Supernet. It reduces layer sizes and widths of the Supernet to obtain subnetworks, making them smaller models suitable for all hardware types. We introduce a novel combination of three techniques to improve the outcomes of the TODM Supernet: adaptive dropouts, an in-place Alpha-divergence knowledge distillation, and the use of ScaledAdam optimizer. We validate our approach by comparing Supernet-trained versus individually tuned Multi-Head State Space Model (MH-SSM) RNN-T using LibriSpeech. Results demonstrate that our TODM Supernet either matches or surpasses the performance of manually tuned models by up to a relative of 3% better in word error rate (WER), while efficiently keeping the cost of training many models at a small constant.

{{</citation>}}


### (81/114) CodeApex: A Bilingual Programming Evaluation Benchmark for Large Language Models (Lingyue Fu et al., 2023)

{{<citation>}}

Lingyue Fu, Huacan Chai, Shuang Luo, Kounianhua Du, Weiming Zhang, Longteng Fan, Jiayi Lei, Renting Rui, Jianghao Lin, Yuchen Fang, Yifan Liu, Jingkuan Wang, Siyuan Qi, Kangning Zhang, Weinan Zhang, Yong Yu. (2023)  
**CodeApex: A Bilingual Programming Evaluation Benchmark for Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.01940v2)  

---


**ABSTRACT**  
With the emergence of Large Language Models (LLMs), there has been a significant improvement in the programming capabilities of models, attracting growing attention from researchers. We propose CodeApex, a bilingual benchmark dataset focusing on the programming comprehension and code generation abilities of LLMs. CodeApex comprises three types of multiple-choice questions: conceptual understanding, commonsense reasoning, and multi-hop reasoning, designed to evaluate LLMs on programming comprehension tasks. Additionally, CodeApex utilizes algorithmic questions and corresponding test cases to assess the code quality generated by LLMs. We evaluate 14 state-of-the-art LLMs, including both general-purpose and specialized models. GPT exhibits the best programming capabilities, achieving approximate accuracies of 50% and 56% on the two tasks, respectively. There is still significant room for improvement in programming tasks. We hope that CodeApex can serve as a reference for evaluating the coding capabilities of LLMs, further promoting their development and growth. Datasets are released at https://github.com/APEXLAB/CodeApex.git. CodeApex submission website is https://apex.sjtu.edu.cn/codeapex/.

{{</citation>}}


### (82/114) On the Planning, Search, and Memorization Capabilities of Large Language Models (Yunhao Yang et al., 2023)

{{<citation>}}

Yunhao Yang, Anshul Tomar. (2023)  
**On the Planning, Search, and Memorization Capabilities of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2309.01868v1)  

---


**ABSTRACT**  
The rapid advancement of large language models, such as the Generative Pre-trained Transformer (GPT) series, has had significant implications across various disciplines. In this study, we investigate the potential of the state-of-the-art large language model (GPT-4) for planning tasks. We explore its effectiveness in multiple planning subfields, highlighting both its strengths and limitations. Through a comprehensive examination, we identify areas where large language models excel in solving planning problems and reveal the constraints that limit their applicability. Our empirical analysis focuses on GPT-4's performance in planning domain extraction, graph search path planning, and adversarial planning. We then propose a way of fine-tuning a domain-specific large language model to improve its Chain of Thought (CoT) capabilities for the above-mentioned tasks. The results provide valuable insights into the potential applications of large language models in the planning domain and pave the way for future research to overcome their limitations and expand their capabilities.

{{</citation>}}


## cs.IR (1)



### (83/114) Tidying Up the Conversational Recommender Systems' Biases (Armin Moradi et al., 2023)

{{<citation>}}

Armin Moradi, Golnoosh Farnadi. (2023)  
**Tidying Up the Conversational Recommender Systems' Biases**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.02550v1)  

---


**ABSTRACT**  
The growing popularity of language models has sparked interest in conversational recommender systems (CRS) within both industry and research circles. However, concerns regarding biases in these systems have emerged. While individual components of CRS have been subject to bias studies, a literature gap remains in understanding specific biases unique to CRS and how these biases may be amplified or reduced when integrated into complex CRS models. In this paper, we provide a concise review of biases in CRS by surveying recent literature. We examine the presence of biases throughout the system's pipeline and consider the challenges that arise from combining multiple models. Our study investigates biases in classic recommender systems and their relevance to CRS. Moreover, we address specific biases in CRS, considering variations with and without natural language understanding capabilities, along with biases related to dialogue systems and language models. Through our findings, we highlight the necessity of adopting a holistic perspective when dealing with biases in complex CRS models.

{{</citation>}}


## cs.ET (1)



### (84/114) Integrated Photonic AI Accelerators under Hardware Security Attacks: Impacts and Countermeasures (Felipe Gohring de Magalhães et al., 2023)

{{<citation>}}

Felipe Gohring de Magalhães, Mahdi Nikdast, Gabriela Nicolescu. (2023)  
**Integrated Photonic AI Accelerators under Hardware Security Attacks: Impacts and Countermeasures**  

---
Primary Category: cs.ET  
Categories: cs-AR, cs-CR, cs-ET, cs.ET  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2309.02543v1)  

---


**ABSTRACT**  
Integrated photonics based on silicon photonics platform is driving several application domains, from enabling ultra-fast chip-scale communication in high-performance computing systems to energy-efficient optical computation in artificial intelligence (AI) hardware accelerators. Integrating silicon photonics into a system necessitates the adoption of interfaces between the photonic and the electronic subsystems, which are required for buffering data and optical-to-electrical and electrical-to-optical conversions. Consequently, this can lead to new and inevitable security breaches that cannot be fully addressed using hardware security solutions proposed for purely electronic systems. This paper explores different types of attacks profiting from such breaches in integrated photonic neural network accelerators. We show the impact of these attacks on the system performance (i.e., power and phase distributions, which impact accuracy) and possible solutions to counter such attacks.

{{</citation>}}


## cs.AI (2)



### (85/114) Experience and Prediction: A Metric of Hardness for a Novel Litmus Test (Nicos Isaak et al., 2023)

{{<citation>}}

Nicos Isaak, Loizos Michael. (2023)  
**Experience and Prediction: A Metric of Hardness for a Novel Litmus Test**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.02534v1)  

---


**ABSTRACT**  
In the last decade, the Winograd Schema Challenge (WSC) has become a central aspect of the research community as a novel litmus test. Consequently, the WSC has spurred research interest because it can be seen as the means to understand human behavior. In this regard, the development of new techniques has made possible the usage of Winograd schemas in various fields, such as the design of novel forms of CAPTCHAs.   Work from the literature that established a baseline for human adult performance on the WSC has shown that not all schemas are the same, meaning that they could potentially be categorized according to their perceived hardness for humans. In this regard, this \textit{hardness-metric} could be used in future challenges or in the WSC CAPTCHA service to differentiate between Winograd schemas.   Recent work of ours has shown that this could be achieved via the design of an automated system that is able to output the hardness-indexes of Winograd schemas, albeit with limitations regarding the number of schemas it could be applied on. This paper adds to previous research by presenting a new system that is based on Machine Learning (ML), able to output the hardness of any Winograd schema faster and more accurately than any other previously used method. Our developed system, which works within two different approaches, namely the random forest and deep learning (LSTM-based), is ready to be used as an extension of any other system that aims to differentiate between Winograd schemas, according to their perceived hardness for humans. At the same time, along with our developed system we extend previous work by presenting the results of a large-scale experiment that shows how human performance varies across Winograd schemas.

{{</citation>}}


### (86/114) A Survey on Interpretable Cross-modal Reasoning (Dizhan Xue et al., 2023)

{{<citation>}}

Dizhan Xue, Shengsheng Qian, Zuyi Zhou, Changsheng Xu. (2023)  
**A Survey on Interpretable Cross-modal Reasoning**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-MM, cs.AI  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.01955v1)  

---


**ABSTRACT**  
In recent years, cross-modal reasoning (CMR), the process of understanding and reasoning across different modalities, has emerged as a pivotal area with applications spanning from multimedia analysis to healthcare diagnostics. As the deployment of AI systems becomes more ubiquitous, the demand for transparency and comprehensibility in these systems' decision-making processes has intensified. This survey delves into the realm of interpretable cross-modal reasoning (I-CMR), where the objective is not only to achieve high predictive performance but also to provide human-understandable explanations for the results. This survey presents a comprehensive overview of the typical methods with a three-level taxonomy for I-CMR. Furthermore, this survey reviews the existing CMR datasets with annotations for explanations. Finally, this survey summarizes the challenges for I-CMR and discusses potential future directions. In conclusion, this survey aims to catalyze the progress of this emerging research area by providing researchers with a panoramic and comprehensive perspective, illuminating the state of the art and discerning the opportunities.

{{</citation>}}


## cs.HC (2)



### (87/114) Do You Trust ChatGPT? -- Perceived Credibility of Human and AI-Generated Content (Martin Huschens et al., 2023)

{{<citation>}}

Martin Huschens, Martin Briesch, Dominik Sobania, Franz Rothlauf. (2023)  
**Do You Trust ChatGPT? -- Perceived Credibility of Human and AI-Generated Content**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.02524v1)  

---


**ABSTRACT**  
This paper examines how individuals perceive the credibility of content originating from human authors versus content generated by large language models, like the GPT language model family that powers ChatGPT, in different user interface versions. Surprisingly, our results demonstrate that regardless of the user interface presentation, participants tend to attribute similar levels of credibility. While participants also do not report any different perceptions of competence and trustworthiness between human and AI-generated content, they rate AI-generated content as being clearer and more engaging. The findings from this study serve as a call for a more discerning approach to evaluating information sources, encouraging users to exercise caution and critical thinking when engaging with content generated by AI systems.

{{</citation>}}


### (88/114) Designing Interfaces for Human-Computer Communication: An On-Going Collection of Considerations (Elena L. Glassman, 2023)

{{<citation>}}

Elena L. Glassman. (2023)  
**Designing Interfaces for Human-Computer Communication: An On-Going Collection of Considerations**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02257v1)  

---


**ABSTRACT**  
While we do not always use words, communicating what we want to an AI is a conversation -- with ourselves as well as with it, a recurring loop with optional steps depending on the complexity of the situation and our request. Any given conversation of this type may include: (a) the human forming an intent, (b) the human expressing that intent as a command or utterance, (c) the AI performing one or more rounds of inference on that command to resolve ambiguities and/or requesting clarifications from the human, (d) the AI showing the inferred meaning of the command and/or its execution on current and future situations or data, (e) the human hopefully correctly recognizing whether the AI's interpretation actually aligns with their intent. In the process, they may (f) update their model of the AI's capabilities and characteristics, (g) update their model of the situations in which the AI is executing its interpretation of their intent, (h) confirm or refine their intent, and (i) revise their expression of their intent to the AI, where the loop repeats until the human is satisfied. With these critical cognitive and computational steps within this back-and-forth laid out as a framework, it is easier to anticipate where communication can fail, and design algorithms and interfaces that ameliorate those failure points.

{{</citation>}}


## cs.CR (2)



### (89/114) Black-Box Attacks against Signed Graph Analysis via Balance Poisoning (Jialong Zhou et al., 2023)

{{<citation>}}

Jialong Zhou, Yuni Lai, Jian Ren, Kai Zhou. (2023)  
**Black-Box Attacks against Signed Graph Analysis via Balance Poisoning**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SI, cs.CR  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2309.02396v1)  

---


**ABSTRACT**  
Signed graphs are well-suited for modeling social networks as they capture both positive and negative relationships. Signed graph neural networks (SGNNs) are commonly employed to predict link signs (i.e., positive and negative) in such graphs due to their ability to handle the unique structure of signed graphs. However, real-world signed graphs are vulnerable to malicious attacks by manipulating edge relationships, and existing adversarial graph attack methods do not consider the specific structure of signed graphs. SGNNs often incorporate balance theory to effectively model the positive and negative links. Surprisingly, we find that the balance theory that they rely on can ironically be exploited as a black-box attack. In this paper, we propose a novel black-box attack called balance-attack that aims to decrease the balance degree of the signed graphs. We present an efficient heuristic algorithm to solve this NP-hard optimization problem. We conduct extensive experiments on five popular SGNN models and four real-world datasets to demonstrate the effectiveness and wide applicability of our proposed attack method. By addressing these challenges, our research contributes to a better understanding of the limitations and resilience of robust models when facing attacks on SGNNs. This work contributes to enhancing the security and reliability of signed graph analysis in social network modeling. Our PyTorch implementation of the attack is publicly available on GitHub: https://github.com/JialongZhou666/Balance-Attack.git.

{{</citation>}}


### (90/114) Empirical Review of Smart Contract and DeFi Security: Vulnerability Detection and Automated Repair (Peng Qian et al., 2023)

{{<citation>}}

Peng Qian, Rui Cao, Zhenguang Liu, Wenqing Li, Ming Li, Lun Zhang, Yufeng Xu, Jianhai Chen, Qinming He. (2023)  
**Empirical Review of Smart Contract and DeFi Security: Vulnerability Detection and Automated Repair**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-SE, cs.CR  
Keywords: Security, Vulnerability Detection  
[Paper Link](http://arxiv.org/abs/2309.02391v2)  

---


**ABSTRACT**  
Decentralized Finance (DeFi) is emerging as a peer-to-peer financial ecosystem, enabling participants to trade products on a permissionless blockchain. Built on blockchain and smart contracts, the DeFi ecosystem has experienced explosive growth in recent years. Unfortunately, smart contracts hold a massive amount of value, making them an attractive target for attacks. So far, attacks against smart contracts and DeFi protocols have resulted in billions of dollars in financial losses, severely threatening the security of the entire DeFi ecosystem. Researchers have proposed various security tools for smart contracts and DeFi protocols as countermeasures. However, a comprehensive investigation of these efforts is still lacking, leaving a crucial gap in our understanding of how to enhance the security posture of the smart contract and DeFi landscape.   To fill the gap, this paper reviews the progress made in the field of smart contract and DeFi security from the perspective of both vulnerability detection and automated repair. First, we analyze the DeFi smart contract security issues and challenges. Specifically, we lucubrate various DeFi attack incidents and summarize the attacks into six categories. Then, we present an empirical study of 42 state-of-the-art techniques that can detect smart contract and DeFi vulnerabilities. In particular, we evaluate the effectiveness of traditional smart contract bug detection tools in analyzing complex DeFi protocols. Additionally, we investigate 8 existing automated repair tools for smart contracts and DeFi protocols, providing insight into their advantages and disadvantages. To make this work useful for as wide of an audience as possible, we also identify several open issues and challenges in the DeFi ecosystem that should be addressed in the future.

{{</citation>}}


## cs.SE (5)



### (91/114) Mind the Gap: The Difference Between Coverage and Mutation Score Can Guide Testing Efforts (Kush Jain et al., 2023)

{{<citation>}}

Kush Jain, Goutamkumar Tulajappa Kalburgi, Claire Le Goues, Alex Groce. (2023)  
**Mind the Gap: The Difference Between Coverage and Mutation Score Can Guide Testing Efforts**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.02395v1)  

---


**ABSTRACT**  
An "adequate" test suite should effectively find all inconsistencies between a system's requirements/specifications and its implementation. Practitioners frequently use code coverage to approximate adequacy, while academics argue that mutation score may better approximate true (oracular) adequacy coverage. High code coverage is increasingly attainable even on large systems via automatic test generation, including fuzzing. In light of all of these options for measuring and improving testing effort, how should a QA engineer spend their time? We propose a new framework for reasoning about the extent, limits, and nature of a given testing effort based on an idea we call the oracle gap, or the difference between source code coverage and mutation score for a given software element. We conduct (1) a large-scale observational study of the oracle gap across popular Maven projects, (2) a study that varies testing and oracle quality across several of those projects and (3) a small-scale observational study of highly critical, well-tested code across comparable blockchain projects. We show that the oracle gap surfaces important information about the extent and quality of a test effort beyond either adequacy metric alone. In particular, it provides a way for practitioners to identify source files where it is likely a weak oracle tests important code.

{{</citation>}}


### (92/114) Contextual Predictive Mutation Testing (Kush Jain et al., 2023)

{{<citation>}}

Kush Jain, Uri Alon, Alex Groce, Claire Le Goues. (2023)  
**Contextual Predictive Mutation Testing**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.02389v1)  

---


**ABSTRACT**  
Mutation testing is a powerful technique for assessing and improving test suite quality that artificially introduces bugs and checks whether the test suites catch them. However, it is also computationally expensive and thus does not scale to large systems and projects. One promising recent approach to tackling this scalability problem uses machine learning to predict whether the tests will detect the synthetic bugs, without actually running those tests. However, existing predictive mutation testing approaches still misclassify 33% of detection outcomes on a randomly sampled set of mutant-test suite pairs. We introduce MutationBERT, an approach for predictive mutation testing that simultaneously encodes the source method mutation and test method, capturing key context in the input representation. Thanks to its higher precision, MutationBERT saves 33% of the time spent by a prior approach on checking/verifying live mutants. MutationBERT, also outperforms the state-of-the-art in both same project and cross project settings, with meaningful improvements in precision, recall, and F1 score. We validate our input representation, and aggregation approaches for lifting predictions from the test matrix level to the test suite level, finding similar improvements in performance. MutationBERT not only enhances the state-of-the-art in predictive mutation testing, but also presents practical benefits for real-world applications, both in saving developer time and finding hard to detect mutants.

{{</citation>}}


### (93/114) Revisiting File Context for Source Code Summarization (Aakash Bansal et al., 2023)

{{<citation>}}

Aakash Bansal, Chia-Yi Su, Collin McMillan. (2023)  
**Revisiting File Context for Source Code Summarization**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: Summarization, Transformer  
[Paper Link](http://arxiv.org/abs/2309.02326v1)  

---


**ABSTRACT**  
Source code summarization is the task of writing natural language descriptions of source code. A typical use case is generating short summaries of subroutines for use in API documentation. The heart of almost all current research into code summarization is the encoder-decoder neural architecture, and the encoder input is almost always a single subroutine or other short code snippet. The problem with this setup is that the information needed to describe the code is often not present in the code itself -- that information often resides in other nearby code. In this paper, we revisit the idea of ``file context'' for code summarization. File context is the idea of encoding select information from other subroutines in the same file. We propose a novel modification of the Transformer architecture that is purpose-built to encode file context and demonstrate its improvement over several baselines. We find that file context helps on a subset of challenging examples where traditional approaches struggle.

{{</citation>}}


### (94/114) A study on the impact of pre-trained model on Just-In-Time defect prediction (Yuxiang Guo et al., 2023)

{{<citation>}}

Yuxiang Guo, Xiaopeng Gao, Zhenyu Zhang, W. K. Chan, Bo Jiang. (2023)  
**A study on the impact of pre-trained model on Just-In-Time defect prediction**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: BERT, GPT  
[Paper Link](http://arxiv.org/abs/2309.02317v1)  

---


**ABSTRACT**  
Previous researchers conducting Just-In-Time (JIT) defect prediction tasks have primarily focused on the performance of individual pre-trained models, without exploring the relationship between different pre-trained models as backbones. In this study, we build six models: RoBERTaJIT, CodeBERTJIT, BARTJIT, PLBARTJIT, GPT2JIT, and CodeGPTJIT, each with a distinct pre-trained model as its backbone. We systematically explore the differences and connections between these models. Specifically, we investigate the performance of the models when using Commit code and Commit message as inputs, as well as the relationship between training efficiency and model distribution among these six models. Additionally, we conduct an ablation experiment to explore the sensitivity of each model to inputs. Furthermore, we investigate how the models perform in zero-shot and few-shot scenarios. Our findings indicate that each model based on different backbones shows improvements, and when the backbone's pre-training model is similar, the training resources that need to be consumed are much more closer. We also observe that Commit code plays a significant role in defect detection, and different pre-trained models demonstrate better defect detection ability with a balanced dataset under few-shot scenarios. These results provide new insights for optimizing JIT defect prediction tasks using pre-trained models and highlight the factors that require more attention when constructing such models. Additionally, CodeGPTJIT and GPT2JIT achieved better performance than DeepJIT and CC2Vec on the two datasets respectively under 2000 training samples. These findings emphasize the effectiveness of transformer-based pre-trained models in JIT defect prediction tasks, especially in scenarios with limited training data.

{{</citation>}}


### (95/114) Using a Nearest-Neighbour, BERT-Based Approach for Scalable Clone Detection (Muslim Chochlov et al., 2023)

{{<citation>}}

Muslim Chochlov, Gul Aftab Ahmed, James Vincent Patten, Guoxian Lu, Wei Hou, David Gregg, Jim Buckley. (2023)  
**Using a Nearest-Neighbour, BERT-Based Approach for Scalable Clone Detection**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2309.02182v1)  

---


**ABSTRACT**  
Code clones can detrimentally impact software maintenance and manually detecting them in very large codebases is impractical. Additionally, automated approaches find detection of Type 3 and Type 4 (inexact) clones very challenging. While the most recent artificial deep neural networks (for example BERT-based artificial neural networks) seem to be highly effective in detecting such clones, their pairwise comparison of every code pair in the target system(s) is inefficient and scales poorly on large codebases.   We therefore introduce SSCD, a BERT-based clone detection approach that targets high recall of Type 3 and Type 4 clones at scale (in line with our industrial partner's requirements). It does so by computing a representative embedding for each code fragment and finding similar fragments using a nearest neighbour search. SSCD thus avoids the pairwise-comparison bottleneck of other Neural Network approaches while also using parallel, GPU-accelerated search to tackle scalability.   This paper details the approach and an empirical assessment towards configuring and evaluating that approach in industrial setting. The configuration analysis suggests that shorter input lengths and text-only based neural network models demonstrate better efficiency in SSCD, while only slightly decreasing effectiveness. The evaluation results suggest that SSCD is more effective than state-of-the-art approaches like SAGA and SourcererCC. It is also highly efficient: in its optimal setting, SSCD effectively locates clones in the entire 320 million LOC BigCloneBench (a standard clone detection benchmark) in just under three hours.

{{</citation>}}


## astro-ph.EP (1)



### (96/114) Sustainability assessment of Low Earth Orbit (LEO) satellite broadband mega-constellations (Ogutu B. Osoro et al., 2023)

{{<citation>}}

Ogutu B. Osoro, Edward J. Oughton, Andrew R. Wilson, Akhil Rao. (2023)  
**Sustainability assessment of Low Earth Orbit (LEO) satellite broadband mega-constellations**  

---
Primary Category: astro-ph.EP  
Categories: astro-ph-EP, astro-ph.EP, cs-SY, econ-GN, eess-SY, q-fin-EC  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2309.02338v1)  

---


**ABSTRACT**  
The growth of mega-constellations is rapidly increasing the number of rocket launches required to place new satellites in space. While Low Earth Orbit (LEO) broadband satellites help to connect unconnected communities and achieve the Sustainable Development Goals, there are also a range of negative environmental externalities, from the burning of rocket fuels and resulting environmental emissions. We present sustainability analytics for phase 1 of the three main LEO constellations including Amazon Kuiper (3,236 satellites), OneWeb (648 satellites), and SpaceX Starlink (4,425 satellites). In baseline scenarios over five years, we find a per subscriber carbon dioxide equivalent (CO$_2$eq) of 0.70$\pm$0.34 tonnes for Kuiper, 1.41$\pm$0.71 tonnes for OneWeb and 0.47$\pm$0.15 tonnes CO$_2$eq/subscriber for Starlink. However, in the worst-case emissions scenario these values increase to 3.02$\pm$1.48 tonnes for Kuiper, 1.7$\pm$0.71 tonnes for OneWeb and 1.04$\pm$0.33 tonnes CO$_2$eq/subscriber for Starlink, more than 31-91 times higher than equivalent terrestrial mobile broadband. Importantly, phase 2 constellations propose to increase the number of satellites by an order-of-magnitude higher, highlighting the pressing need to mitigate negative environmental impacts. Strategic choices in rocket design and fuel options can help to substantially mitigate negative sustainability impacts.

{{</citation>}}


## physics.acc-ph (1)



### (97/114) Resilient VAE: Unsupervised Anomaly Detection at the SLAC Linac Coherent Light Source (Ryan Humble et al., 2023)

{{<citation>}}

Ryan Humble, William Colocho, Finn O'Shea, Daniel Ratner, Eric Darve. (2023)  
**Resilient VAE: Unsupervised Anomaly Detection at the SLAC Linac Coherent Light Source**  

---
Primary Category: physics.acc-ph  
Categories: cs-LG, physics-acc-ph, physics.acc-ph  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2309.02333v1)  

---


**ABSTRACT**  
Significant advances in utilizing deep learning for anomaly detection have been made in recent years. However, these methods largely assume the existence of a normal training set (i.e., uncontaminated by anomalies) or even a completely labeled training set. In many complex engineering systems, such as particle accelerators, labels are sparse and expensive; in order to perform anomaly detection in these cases, we must drop these assumptions and utilize a completely unsupervised method. This paper introduces the Resilient Variational Autoencoder (ResVAE), a deep generative model specifically designed for anomaly detection. ResVAE exhibits resilience to anomalies present in the training data and provides feature-level anomaly attribution. During the training process, ResVAE learns the anomaly probability for each sample as well as each individual feature, utilizing these probabilities to effectively disregard anomalous examples in the training data. We apply our proposed method to detect anomalies in the accelerator status at the SLAC Linac Coherent Light Source (LCLS). By utilizing shot-to-shot data from the beam position monitoring system, we demonstrate the exceptional capability of ResVAE in identifying various types of anomalies that are visible in the accelerator.

{{</citation>}}


## q-bio.NC (2)



### (98/114) Information Processing by Neuron Populations in the Central Nervous System: Mathematical Structure of Data and Operations (Martin N. P. Nilsson, 2023)

{{<citation>}}

Martin N. P. Nilsson. (2023)  
**Information Processing by Neuron Populations in the Central Nervous System: Mathematical Structure of Data and Operations**  

---
Primary Category: q-bio.NC  
Categories: 92-10 (Primary) 92B20, 68T05 (Secondary), cs-AI, cs-LG, cs-NE, q-bio-NC, q-bio.NC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02332v1)  

---


**ABSTRACT**  
In the intricate architecture of the mammalian central nervous system, neurons form populations. Axonal bundles communicate between these clusters using spike trains as their medium. However, these neuron populations' precise encoding and operations have yet to be discovered. In our analysis, the starting point is a state-of-the-art mechanistic model of a generic neuron endowed with plasticity. From this simple framework emerges a profound mathematical construct: The representation and manipulation of information can be precisely characterized by an algebra of finite convex cones. Furthermore, these neuron populations are not merely passive transmitters. They act as operators within this algebraic structure, mirroring the functionality of a low-level programming language. When these populations interconnect, they embody succinct yet potent algebraic expressions. These networks allow them to implement many operations, such as specialization, generalization, novelty detection, dimensionality reduction, inverse modeling, prediction, and associative memory. In broader terms, this work illuminates the potential of matrix embeddings in advancing our understanding in fields like cognitive science and AI. These embeddings enhance the capacity for concept processing and hierarchical description over their vector counterparts.

{{</citation>}}


### (99/114) Dynamic Brain Transformer with Multi-level Attention for Functional Brain Network Analysis (Xuan Kan et al., 2023)

{{<citation>}}

Xuan Kan, Antonio Aodong Chen Gu, Hejie Cui, Ying Guo, Carl Yang. (2023)  
**Dynamic Brain Transformer with Multi-level Attention for Functional Brain Network Analysis**  

---
Primary Category: q-bio.NC  
Categories: 68T07, 68T05, I-2-6; J-3, cs-AI, cs-LG, q-bio-NC, q-bio.NC  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2309.01941v1)  

---


**ABSTRACT**  
Recent neuroimaging studies have highlighted the importance of network-centric brain analysis, particularly with functional magnetic resonance imaging. The emergence of Deep Neural Networks has fostered a substantial interest in predicting clinical outcomes and categorizing individuals based on brain networks. However, the conventional approach involving static brain network analysis offers limited potential in capturing the dynamism of brain function. Although recent studies have attempted to harness dynamic brain networks, their high dimensionality and complexity present substantial challenges. This paper proposes a novel methodology, Dynamic bRAin Transformer (DART), which combines static and dynamic brain networks for more effective and nuanced brain function analysis. Our model uses the static brain network as a baseline, integrating dynamic brain networks to enhance performance against traditional methods. We innovatively employ attention mechanisms, enhancing model explainability and exploiting the dynamic brain network's temporal variations. The proposed approach offers a robust solution to the low signal-to-noise ratio of blood-oxygen-level-dependent signals, a recurring issue in direct DNN modeling. It also provides valuable insights into which brain circuits or dynamic networks contribute more to final predictions. As such, DRAT shows a promising direction in neuroimaging studies, contributing to the comprehensive understanding of brain organization and the role of neural circuits.

{{</citation>}}


## stat.ML (2)



### (100/114) On the Complexity of Differentially Private Best-Arm Identification with Fixed Confidence (Achraf Azize et al., 2023)

{{<citation>}}

Achraf Azize, Marc Jourdan, Aymen Al Marjani, Debabrota Basu. (2023)  
**On the Complexity of Differentially Private Best-Arm Identification with Fixed Confidence**  

---
Primary Category: stat.ML  
Categories: cs-CR, cs-LG, math-ST, stat-ML, stat-TH, stat.ML  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02202v1)  

---


**ABSTRACT**  
Best Arm Identification (BAI) problems are progressively used for data-sensitive applications, such as designing adaptive clinical trials, tuning hyper-parameters, and conducting user studies to name a few. Motivated by the data privacy concerns invoked by these applications, we study the problem of BAI with fixed confidence under $\epsilon$-global Differential Privacy (DP). First, to quantify the cost of privacy, we derive a lower bound on the sample complexity of any $\delta$-correct BAI algorithm satisfying $\epsilon$-global DP. Our lower bound suggests the existence of two privacy regimes depending on the privacy budget $\epsilon$. In the high-privacy regime (small $\epsilon$), the hardness depends on a coupled effect of privacy and a novel information-theoretic quantity, called the Total Variation Characteristic Time. In the low-privacy regime (large $\epsilon$), the sample complexity lower bound reduces to the classical non-private lower bound. Second, we propose AdaP-TT, an $\epsilon$-global DP variant of the Top Two algorithm. AdaP-TT runs in arm-dependent adaptive episodes and adds Laplace noise to ensure a good privacy-utility trade-off. We derive an asymptotic upper bound on the sample complexity of AdaP-TT that matches with the lower bound up to multiplicative constants in the high-privacy regime. Finally, we provide an experimental analysis of AdaP-TT that validates our theoretical results.

{{</citation>}}


### (101/114) QuantEase: Optimization-based Quantization for Language Models -- An Efficient and Intuitive Algorithm (Kayhan Behdin et al., 2023)

{{<citation>}}

Kayhan Behdin, Ayan Acharya, Aman Gupta, Sathiya Keerthi, Rahul Mazumder. (2023)  
**QuantEase: Optimization-based Quantization for Language Models -- An Efficient and Intuitive Algorithm**  

---
Primary Category: stat.ML  
Categories: cs-CL, cs-LG, stat-ML, stat.ML  
Keywords: GPT, Language Model, Quantization  
[Paper Link](http://arxiv.org/abs/2309.01885v1)  

---


**ABSTRACT**  
With the rising popularity of Large Language Models (LLMs), there has been an increasing interest in compression techniques that enable their efficient deployment. This study focuses on the Post-Training Quantization (PTQ) of LLMs. Drawing from recent advances, our work introduces QuantEase, a layer-wise quantization framework where individual layers undergo separate quantization. The problem is framed as a discrete-structured non-convex optimization, prompting the development of algorithms rooted in Coordinate Descent (CD) techniques. These CD-based methods provide high-quality solutions to the complex non-convex layer-wise quantization problems. Notably, our CD-based approach features straightforward updates, relying solely on matrix and vector operations, circumventing the need for matrix inversion or decomposition. We also explore an outlier-aware variant of our approach, allowing for retaining significant weights (outliers) with complete precision. Our proposal attains state-of-the-art performance in terms of perplexity and zero-shot accuracy in empirical evaluations across various LLMs and datasets, with relative improvements up to 15% over methods such as GPTQ. Particularly noteworthy is our outlier-aware algorithm's capability to achieve near or sub-3-bit quantization of LLMs with an acceptable drop in accuracy, obviating the need for non-uniform quantization or grouping techniques, improving upon methods such as SpQR by up to two times in terms of perplexity.

{{</citation>}}


## cs.MA (1)



### (102/114) Personalized Federated Deep Reinforcement Learning-based Trajectory Optimization for Multi-UAV Assisted Edge Computing (Zhengrong Song et al., 2023)

{{<citation>}}

Zhengrong Song, Chuan Ma, Ming Ding, Howard H. Yang, Yuwen Qian, Xiangwei Zhou. (2023)  
**Personalized Federated Deep Reinforcement Learning-based Trajectory Optimization for Multi-UAV Assisted Edge Computing**  

---
Primary Category: cs.MA  
Categories: cs-LG, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.02193v1)  

---


**ABSTRACT**  
In the era of 5G mobile communication, there has been a significant surge in research focused on unmanned aerial vehicles (UAVs) and mobile edge computing technology. UAVs can serve as intelligent servers in edge computing environments, optimizing their flight trajectories to maximize communication system throughput. Deep reinforcement learning (DRL)-based trajectory optimization algorithms may suffer from poor training performance due to intricate terrain features and inadequate training data. To overcome this limitation, some studies have proposed leveraging federated learning (FL) to mitigate the data isolation problem and expedite convergence. Nevertheless, the efficacy of global FL models can be negatively impacted by the high heterogeneity of local data, which could potentially impede the training process and even compromise the performance of local agents. This work proposes a novel solution to address these challenges, namely personalized federated deep reinforcement learning (PF-DRL), for multi-UAV trajectory optimization. PF-DRL aims to develop individualized models for each agent to address the data scarcity issue and mitigate the negative impact of data heterogeneity. Simulation results demonstrate that the proposed algorithm achieves superior training performance with faster convergence rates, and improves service quality compared to other DRL-based approaches.

{{</citation>}}


## cs.IT (1)



### (103/114) A Wideband MIMO Channel Model for Aerial Intelligent Reflecting Surface-Assisted Wireless Communications (Shaoyi Liu et al., 2023)

{{<citation>}}

Shaoyi Liu, Nan Ma, Yaning Chen, Ke Peng, Dongsheng Xue. (2023)  
**A Wideband MIMO Channel Model for Aerial Intelligent Reflecting Surface-Assisted Wireless Communications**  

---
Primary Category: cs.IT  
Categories: cs-IT, cs.IT, eess-SP, math-IT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02171v1)  

---


**ABSTRACT**  
Compared to traditional intelligent reflecting surfaces(IRS), aerial IRS (AIRS) has unique advantages, such as more flexible deployment and wider service coverage. However, modeling AIRS in the channel presents new challenges due to their mobility. In this paper, a three-dimensional (3D) wideband channel model for AIRS and IRS joint-assisted multiple-input multiple-output (MIMO) communication system is proposed, where considering the rotational degrees of freedom in three directions and the motion angles of AIRS in space. Based on the proposed model, the channel impulse response (CIR), correlation function, and channel capacity are derived, and several feasible joint phase shifts schemes for AIRS and IRS units are proposed. Simulation results show that the proposed model can capture the channel characteristics accurately, and the proposed phase shifts methods can effectively improve the channel statistical characteristics and increase the system capacity. Additionally, we observe that in certain scenarios, the paths involving the IRS and the line-of-sight (LoS) paths exhibit similar characteristics. These findings provide valuable insights for the future development of intelligent communication systems.

{{</citation>}}


## cs.CY (5)



### (104/114) Who are the users of ChatGPT? Implications for the digital divide from web tracking data (Celina Kacperski et al., 2023)

{{<citation>}}

Celina Kacperski, Roberto Ulloa, Denis Bonnay, Juhi Kulshrestha, Peter Selb, Andreas Spitz. (2023)  
**Who are the users of ChatGPT? Implications for the digital divide from web tracking data**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.02142v1)  

---


**ABSTRACT**  
A major challenge of our time is reducing disparities in access to and effective use of digital technologies, with recent discussions highlighting the role of AI in exacerbating the digital divide. We examine user characteristics that predict usage of the AI-powered conversational agent ChatGPT. We combine web tracking and survey data of N=1068 German citizens to investigate differences in activity (usage, visits and duration on chat.openai.com). We examine socio-demographics commonly associated with the digital divide and explore further socio-political attributes identified via stability selection in Lasso regressions. We confirm lower age and more education to affect ChatGPT usage, but not gender and income. We find full-time employment and more children to be barriers to ChatGPT activity. Rural residence, writing and social media activities, as well as more political knowledge, were positively associated with ChatGPT activity. Our research informs efforts to address digital disparities and promote digital literacy among underserved populations.

{{</citation>}}


### (105/114) Exploring the Intersection of Complex Aesthetics and Generative AI for Promoting Cultural Creativity in Rural China after the Post-Pandemic Era (Mengyao Guo et al., 2023)

{{<citation>}}

Mengyao Guo, Xiaolin Zhang, Yuan Zhuang, Jing Chen, Pengfei Wang, Ze Gao. (2023)  
**Exploring the Intersection of Complex Aesthetics and Generative AI for Promoting Cultural Creativity in Rural China after the Post-Pandemic Era**  

---
Primary Category: cs.CY  
Categories: F-2-2, I-2-7, cs-AI, cs-CY, cs-MM, cs.CY  
Keywords: AI, Generative AI  
[Paper Link](http://arxiv.org/abs/2309.02136v1)  

---


**ABSTRACT**  
This paper explores using generative AI and aesthetics to promote cultural creativity in rural China amidst COVID-19's impact. Through literature reviews, case studies, surveys, and text analysis, it examines art and technology applications in rural contexts and identifies key challenges. The study finds artworks often fail to resonate locally, while reliance on external artists limits sustainability. Hence, nurturing grassroots "artist villagers" through AI is proposed. Our approach involves training machine learning on subjective aesthetics to generate culturally relevant content. Interactive AI media can also boost tourism while preserving heritage. This pioneering research puts forth original perspectives on the intersection of AI and aesthetics to invigorate rural culture. It advocates holistic integration of technology and emphasizes AI's potential as a creative enabler versus replacement. Ultimately, it lays the groundwork for further exploration of leveraging AI innovations to empower rural communities. This timely study contributes to growing interest in emerging technologies to address critical issues facing rural China.

{{</citation>}}


### (106/114) The Impact of Artificial Intelligence on the Evolution of Digital Education: A Comparative Study of OpenAI Text Generation Tools including ChatGPT, Bing Chat, Bard, and Ernie (Negin Yazdani Motlagh et al., 2023)

{{<citation>}}

Negin Yazdani Motlagh, Matin Khajavi, Abbas Sharifi, Mohsen Ahmadi. (2023)  
**The Impact of Artificial Intelligence on the Evolution of Digital Education: A Comparative Study of OpenAI Text Generation Tools including ChatGPT, Bing Chat, Bard, and Ernie**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs.CY  
Keywords: AI, ChatGPT, GPT, Text Generation  
[Paper Link](http://arxiv.org/abs/2309.02029v1)  

---


**ABSTRACT**  
In the digital era, the integration of artificial intelligence (AI) in education has ushered in transformative changes, redefining teaching methodologies, curriculum planning, and student engagement. This review paper delves deep into the rapidly evolving landscape of digital education by contrasting the capabilities and impact of OpenAI's pioneering text generation tools like Bing Chat, Bard, Ernie with a keen focus on the novel ChatGPT. Grounded in a typology that views education through the lenses of system, process, and result, the paper navigates the multifaceted applications of AI. From decentralizing global education and personalizing curriculums to digitally documenting competence-based outcomes, AI stands at the forefront of educational modernization. Highlighting ChatGPT's meteoric rise to one million users in just five days, the study underscores its role in democratizing education, fostering autodidacticism, and magnifying student engagement. However, with such transformative power comes the potential for misuse, as text-generation tools can inadvertently challenge academic integrity. By juxtaposing the promise and pitfalls of AI in education, this paper advocates for a harmonized synergy between AI tools and the educational community, emphasizing the urgent need for ethical guidelines, pedagogical adaptations, and strategic collaborations.

{{</citation>}}


### (107/114) Provably safe systems: the only path to controllable AGI (Max Tegmark et al., 2023)

{{<citation>}}

Max Tegmark, Steve Omohundro. (2023)  
**Provably safe systems: the only path to controllable AGI**  

---
Primary Category: cs.CY  
Categories: cs-AI, cs-CY, cs-LG, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.01933v1)  

---


**ABSTRACT**  
We describe a path to humanity safely thriving with powerful Artificial General Intelligences (AGIs) by building them to provably satisfy human-specified requirements. We argue that this will soon be technically feasible using advanced AI for formal verification and mechanistic interpretability. We further argue that it is the only path which guarantees safe controlled AGI. We end with a list of challenge problems whose solution would contribute to this positive outcome and invite readers to join in this work.

{{</citation>}}


### (108/114) Towards Understanding of Deepfake Videos in the Wild (Beomsang Cho et al., 2023)

{{<citation>}}

Beomsang Cho, Binh M. Le, Jiwon Kim, Simon Woo, Shahroz Tariq, Alsharif Abuadbba, Kristen Moore. (2023)  
**Towards Understanding of Deepfake Videos in the Wild**  

---
Primary Category: cs.CY  
Categories: cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.01919v2)  

---


**ABSTRACT**  
Deepfakes have become a growing concern in recent years, prompting researchers to develop benchmark datasets and detection algorithms to tackle the issue. However, existing datasets suffer from significant drawbacks that hamper their effectiveness. Notably, these datasets fail to encompass the latest deepfake videos produced by state-of-the-art methods that are being shared across various platforms. This limitation impedes the ability to keep pace with the rapid evolution of generative AI techniques employed in real-world deepfake production. Our contributions in this IRB-approved study are to bridge this knowledge gap from current real-world deepfakes by providing in-depth analysis. We first present the largest and most diverse and recent deepfake dataset (RWDF-23) collected from the wild to date, consisting of 2,000 deepfake videos collected from 4 platforms targeting 4 different languages span created from 21 countries: Reddit, YouTube, TikTok, and Bilibili. By expanding the dataset's scope beyond the previous research, we capture a broader range of real-world deepfake content, reflecting the ever-evolving landscape of online platforms. Also, we conduct a comprehensive analysis encompassing various aspects of deepfakes, including creators, manipulation strategies, purposes, and real-world content production methods. This allows us to gain valuable insights into the nuances and characteristics of deepfakes in different contexts. Lastly, in addition to the video content, we also collect viewer comments and interactions, enabling us to explore the engagements of internet users with deepfake content. By considering this rich contextual information, we aim to provide a holistic understanding of the {evolving} deepfake phenomenon and its impact on online platforms.

{{</citation>}}


## math.HO (1)



### (109/114) Wordle: A Microcosm of Life. Luck, Skill, Cheating, Loyalty, and Influence! (James P. Dilger, 2023)

{{<citation>}}

James P. Dilger. (2023)  
**Wordle: A Microcosm of Life. Luck, Skill, Cheating, Loyalty, and Influence!**  

---
Primary Category: math.HO  
Categories: cs-CL, math-HO, math.HO  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.02110v1)  

---


**ABSTRACT**  
Wordle is a popular, online word game offered by the New York Times (nytimes.com). Currently there are some 2 million players of the English version worldwide. Players have 6 attempts to guess the daily word (target word) and after each attempt, the player receives color-coded information about the correctness and position of each letter in the guess. After either a successful completion of the puzzle or the final unsuccessful attempt, software can assess the player's luck and skill using Information Theory and can display data for the first, second, ..., sixth guesses of a random sample of all players. Recently, I discovered that the latter data is presented in a format that can easily be copied and pasted into a spreadsheet. I compiled data on Wordle players' first guesses from May 2023 - August 2023 and inferred some interesting information about Wordle players. A) Every day, about 0.2-0.5% of players solve the puzzle in one attempt. Because the odds of guessing the one of 2,315 possible target words at random is 0.043%, this implies that 4,000 - 10,000 players cheat by obtaining the target word outside of playing the game! B) At least 1/3 of the players have a favorite starting word, or cycle through several. And even though players should be aware that target words are never repeated, most players appear to remain loyal to their starting word even after its appearance as a target word. C) On August 15, 2023, about 30,000 players abruptly changed their starting word, presumably based on a crossword puzzle clue! Wordle players can be influenced! This study goes beyond social media postings, surveys, and Google Trends to provide solid, quantitative evidence about cheating in Wordle.

{{</citation>}}


## q-bio.QM (1)



### (110/114) BeeTLe: A Framework for Linear B-Cell Epitope Prediction and Classification (Xiao Yuan, 2023)

{{<citation>}}

Xiao Yuan. (2023)  
**BeeTLe: A Framework for Linear B-Cell Epitope Prediction and Classification**  

---
Primary Category: q-bio.QM  
Categories: cs-LG, q-bio-QM, q-bio.QM  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.02071v1)  

---


**ABSTRACT**  
The process of identifying and characterizing B-cell epitopes, which are the portions of antigens recognized by antibodies, is important for our understanding of the immune system, and for many applications including vaccine development, therapeutics, and diagnostics. Computational epitope prediction is challenging yet rewarding as it significantly reduces the time and cost of laboratory work. Most of the existing tools do not have satisfactory performance and only discriminate epitopes from non-epitopes. This paper presents a new deep learning-based multi-task framework for linear B-cell epitope prediction as well as antibody type-specific epitope classification. Specifically, a sequenced-based neural network model using recurrent layers and Transformer blocks is developed. We propose an amino acid encoding method based on eigen decomposition to help the model learn the representations of epitopes. We introduce modifications to standard cross-entropy loss functions by extending a logit adjustment technique to cope with the class imbalance. Experimental results on data curated from the largest public epitope database demonstrate the validity of the proposed methods and the superior performance compared to competing ones.

{{</citation>}}


## cs.NI (1)



### (111/114) How Can AI be Distributed in the Computing Continuum? Introducing the Neural Pub/Sub Paradigm (Lauri Lovén et al., 2023)

{{<citation>}}

Lauri Lovén, Roberto Morabito, Abhishek Kumar, Susanna Pirttikangas, Jukka Riekki, Sasu Tarkoma. (2023)  
**How Can AI be Distributed in the Computing Continuum? Introducing the Neural Pub/Sub Paradigm**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.02058v1)  

---


**ABSTRACT**  
This paper proposes the neural publish/subscribe paradigm, a novel approach to orchestrating AI workflows in large-scale distributed AI systems in the computing continuum. Traditional centralized broker methodologies are increasingly struggling with managing the data surge resulting from the proliferation of 5G systems, connected devices, and ultra-reliable applications. Moreover, the advent of AI-powered applications, particularly those leveraging advanced neural network architectures, necessitates a new approach to orchestrate and schedule AI processes within the computing continuum. In response, the neural pub/sub paradigm aims to overcome these limitations by efficiently managing training, fine-tuning and inference workflows, improving distributed computation, facilitating dynamic resource allocation, and enhancing system resilience across the computing continuum. We explore this new paradigm through various design patterns, use cases, and discuss open research questions for further exploration.

{{</citation>}}


## cs.DB (1)



### (112/114) Automatic Data Transformation Using Large Language Model: An Experimental Study on Building Energy Data (Ankita Sharma et al., 2023)

{{<citation>}}

Ankita Sharma, Xuanmao Li, Hong Guan, Guoxin Sun, Liang Zhang, Lanjun Wang, Kesheng Wu, Lei Cao, Erkang Zhu, Alexander Sim, Teresa Wu, Jia Zou. (2023)  
**Automatic Data Transformation Using Large Language Model: An Experimental Study on Building Energy Data**  

---
Primary Category: cs.DB  
Categories: cs-DB, cs.DB  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.01957v2)  

---


**ABSTRACT**  
Existing approaches to automatic data transformation are insufficient to meet the requirements in many real-world scenarios, such as the building sector. First, there is no convenient interface for domain experts to provide domain knowledge easily. Second, they require significant training data collection overheads. Third, the accuracy suffers from complicated schema changes. To bridge this gap, we present a novel approach that leverages the unique capabilities of large language models (LLMs) in coding, complex reasoning, and zero-shot learning to generate SQL code that transforms the source datasets into the target datasets. We demonstrate the viability of this approach by designing an LLM-based framework, termed SQLMorpher, which comprises a prompt generator that integrates the initial prompt with optional domain knowledge and historical patterns in external databases. It also implements an iterative prompt optimization mechanism that automatically improves the prompt based on flaw detection. The key contributions of this work include (1) pioneering an end-to-end LLM-based solution for data transformation, (2) developing a benchmark dataset of 105 real-world building energy data transformation problems, and (3) conducting an extensive empirical evaluation where our approach achieved 96% accuracy in all 105 problems. SQLMorpher demonstrates the effectiveness of utilizing LLMs in complex, domain-specific challenges, highlighting the potential of their potential to drive sustainable solutions.

{{</citation>}}


## hep-ex (1)



### (113/114) Extended Symmetry Preserving Attention Networks for LHC Analysis (Michael James Fenton et al., 2023)

{{<citation>}}

Michael James Fenton, Alexander Shmakov, Hideki Okawa, Yuji Li, Ko-Yang Hsiao, Shih-Chieh Hsu, Daniel Whiteson, Pierre Baldi. (2023)  
**Extended Symmetry Preserving Attention Networks for LHC Analysis**  

---
Primary Category: hep-ex  
Categories: cs-LG, hep-ex, hep-ex, hep-ph  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.01886v1)  

---


**ABSTRACT**  
Reconstructing unstable heavy particles requires sophisticated techniques to sift through the large number of possible permutations for assignment of detector objects to partons. An approach based on a generalized attention mechanism, symmetry preserving attention networks (SPANet), has been previously applied to top quark pair decays at the Large Hadron Collider, which produce six hadronic jets. Here we extend the SPANet architecture to consider multiple input streams, such as leptons, as well as global event features, such as the missing transverse momentum. In addition, we provide regression and classification outputs to supplement the parton assignment. We explore the performance of the extended capability of SPANet in the context of semi-leptonic decays of top quark pairs as well as top quark pairs produced in association with a Higgs boson. We find significant improvements in the power of three representative studies: search for ttH, measurement of the top quark mass and a search for a heavy Z' decaying to top quark pairs. We present ablation studies to provide insight on what the network has learned in each case.

{{</citation>}}


## math.NA (1)



### (114/114) Variable Time Step Method of DAHLQUIST, LINIGER and NEVANLINNA (DLN) for a Corrected Smagorinsky Model (Farjana Siddiqua et al., 2023)

{{<citation>}}

Farjana Siddiqua, Wenlong Pei. (2023)  
**Variable Time Step Method of DAHLQUIST, LINIGER and NEVANLINNA (DLN) for a Corrected Smagorinsky Model**  

---
Primary Category: math.NA  
Categories: cs-NA, math-NA, math.NA  
Keywords: NLI  
[Paper Link](http://arxiv.org/abs/2309.01867v1)  

---


**ABSTRACT**  
Turbulent flows strain resources, both memory and CPU speed. The DLN method has greater accuracy and allows larger time steps, requiring less memory and fewer FLOPS. The DLN method can also be implemented adaptively. The classical Smagorinsky model, as an effective way to approximate a (resolved) mean velocity, has recently been corrected to represent a flow of energy from unresolved fluctuations to the (resolved) mean velocity. In this paper, we apply a family of second-order, G-stable time-stepping methods proposed by Dahlquist, Liniger, and Nevanlinna (the DLN method) to one corrected Smagorinsky model and provide the detailed numerical analysis of the stability and consistency. We prove that the numerical solutions under any arbitrary time step sequences are unconditionally stable in the long term and converge at second order. We also provide error estimate under certain time step condition. Numerical tests are given to confirm the rate of convergence and also to show that the adaptive DLN algorithm helps to control numerical dissipation so that backscatter is visible.

{{</citation>}}
