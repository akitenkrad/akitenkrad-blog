---
draft: false
title: "arXiv @ 2023.09.11"
date: 2023-09-11
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.11"
    identifier: arxiv_20230911
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (12)](#cscv-12)
- [cs.HC (3)](#cshc-3)
- [cs.LG (7)](#cslg-7)
- [eess.SP (1)](#eesssp-1)
- [cs.CL (15)](#cscl-15)
- [eess.SY (2)](#eesssy-2)
- [quant-ph (1)](#quant-ph-1)
- [cs.CR (2)](#cscr-2)
- [cs.SE (1)](#csse-1)
- [cs.SI (2)](#cssi-2)
- [cs.SD (2)](#cssd-2)
- [cs.RO (1)](#csro-1)
- [eess.IV (3)](#eessiv-3)

## cs.CV (12)



### (1/52) How to Evaluate Semantic Communications for Images with ViTScore Metric? (Tingting Zhu et al., 2023)

{{<citation>}}

Tingting Zhu, Bo Peng, Jifan Liang, Tingchen Han, Hai Wan, Jingqiao Fu, Junjie Chen. (2023)  
**How to Evaluate Semantic Communications for Images with ViTScore Metric?**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-IT, cs.CV, math-IT  
Keywords: BERT, NLP, Transformer  
[Paper Link](http://arxiv.org/abs/2309.04891v1)  

---


**ABSTRACT**  
Semantic communications (SC) have been expected to be a new paradigm shifting to catalyze the next generation communication, whose main concerns shift from accurate bit transmission to effective semantic information exchange in communications. However, the previous and widely-used metrics for images are not applicable to evaluate the image semantic similarity in SC. Classical metrics to measure the similarity between two images usually rely on the pixel level or the structural level, such as the PSNR and the MS-SSIM. Straightforwardly using some tailored metrics based on deep-learning methods in CV community, such as the LPIPS, is infeasible for SC. To tackle this, inspired by BERTScore in NLP community, we propose a novel metric for evaluating image semantic similarity, named Vision Transformer Score (ViTScore). We prove theoretically that ViTScore has 3 important properties, including symmetry, boundedness, and normalization, which make ViTScore convenient and intuitive for image measurement. To evaluate the performance of ViTScore, we compare ViTScore with 3 typical metrics (PSNR, MS-SSIM, and LPIPS) through 5 classes of experiments. Experimental results demonstrate that ViTScore can better evaluate the image semantic similarity than the other 3 typical metrics, which indicates that ViTScore is an effective performance metric when deployed in SC scenarios.

{{</citation>}}


### (2/52) Few-Shot Medical Image Segmentation via a Region-enhanced Prototypical Transformer (Yazhou Zhu et al., 2023)

{{<citation>}}

Yazhou Zhu, Shidong Wang, Tong Xin, Haofeng Zhang. (2023)  
**Few-Shot Medical Image Segmentation via a Region-enhanced Prototypical Transformer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias, Few-Shot, Transformer  
[Paper Link](http://arxiv.org/abs/2309.04825v1)  

---


**ABSTRACT**  
Automated segmentation of large volumes of medical images is often plagued by the limited availability of fully annotated data and the diversity of organ surface properties resulting from the use of different acquisition protocols for different patients. In this paper, we introduce a more promising few-shot learning-based method named Region-enhanced Prototypical Transformer (RPT) to mitigate the effects of large intra-class diversity/bias. First, a subdivision strategy is introduced to produce a collection of regional prototypes from the foreground of the support prototype. Second, a self-selection mechanism is proposed to incorporate into the Bias-alleviated Transformer (BaT) block to suppress or remove interferences present in the query prototype and regional support prototypes. By stacking BaT blocks, the proposed RPT can iteratively optimize the generated regional prototypes and finally produce rectified and more accurate global prototypes for Few-Shot Medical Image Segmentation (FSMS). Extensive experiments are conducted on three publicly available medical image datasets, and the obtained results show consistent improvements compared to state-of-the-art FSMS methods. The source code is available at: https://github.com/YazhouZhu19/RPT.

{{</citation>}}


### (3/52) Timely Fusion of Surround Radar/Lidar for Object Detection in Autonomous Driving Systems (Wenjing Xie et al., 2023)

{{<citation>}}

Wenjing Xie, Tao Hu, Neiwen Ling, Guoliang Xing, Shaoshan Liu, Nan Guan. (2023)  
**Timely Fusion of Surround Radar/Lidar for Object Detection in Autonomous Driving Systems**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.04806v1)  

---


**ABSTRACT**  
Fusing Radar and Lidar sensor data can fully utilize their complementary advantages and provide more accurate reconstruction of the surrounding for autonomous driving systems. Surround Radar/Lidar can provide 360-degree view sampling with the minimal cost, which are promising sensing hardware solutions for autonomous driving systems. However, due to the intrinsic physical constraints, the rotating speed of surround Radar, and thus the frequency to generate Radar data frames, is much lower than surround Lidar. Existing Radar/Lidar fusion methods have to work at the low frequency of surround Radar, which cannot meet the high responsiveness requirement of autonomous driving systems.This paper develops techniques to fuse surround Radar/Lidar with working frequency only limited by the faster surround Lidar instead of the slower surround Radar, based on the state-of-the-art object detection model MVDNet. The basic idea of our approach is simple: we let MVDNet work with temporally unaligned data from Radar/Lidar, so that fusion can take place at any time when a new Lidar data frame arrives, instead of waiting for the slow Radar data frame. However, directly applying MVDNet to temporally unaligned Radar/Lidar data greatly degrades its object detection accuracy. The key information revealed in this paper is that we can achieve high output frequency with little accuracy loss by enhancing the training procedure to explore the temporal redundancy in MVDNet so that it can tolerate the temporal unalignment of input data. We explore several different ways of training enhancement and compare them quantitatively with experiments.

{{</citation>}}


### (4/52) Towards Real-World Burst Image Super-Resolution: Benchmark and Method (Pengxu Wei et al., 2023)

{{<citation>}}

Pengxu Wei, Yujing Sun, Xingbei Guo, Chang Liu, Jie Chen, Xiangyang Ji, Liang Lin. (2023)  
**Towards Real-World Burst Image Super-Resolution: Benchmark and Method**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.04803v1)  

---


**ABSTRACT**  
Despite substantial advances, single-image super-resolution (SISR) is always in a dilemma to reconstruct high-quality images with limited information from one input image, especially in realistic scenarios. In this paper, we establish a large-scale real-world burst super-resolution dataset, i.e., RealBSR, to explore the faithful reconstruction of image details from multiple frames. Furthermore, we introduce a Federated Burst Affinity network (FBAnet) to investigate non-trivial pixel-wise displacements among images under real-world image degradation. Specifically, rather than using pixel-wise alignment, our FBAnet employs a simple homography alignment from a structural geometry aspect and a Federated Affinity Fusion (FAF) strategy to aggregate the complementary information among frames. Those fused informative representations are fed to a Transformer-based module of burst representation decoding. Besides, we have conducted extensive experiments on two versions of our datasets, i.e., RealBSR-RAW and RealBSR-RGB. Experimental results demonstrate that our FBAnet outperforms existing state-of-the-art burst SR methods and also achieves visually-pleasant SR image predictions with model details. Our dataset, codes, and models are publicly available at https://github.com/yjsunnn/FBANet.

{{</citation>}}


### (5/52) Self-Supervised Transformer with Domain Adaptive Reconstruction for General Face Forgery Video Detection (Daichi Zhang et al., 2023)

{{<citation>}}

Daichi Zhang, Zihao Xiao, Jianmin Li, Shiming Ge. (2023)  
**Self-Supervised Transformer with Domain Adaptive Reconstruction for General Face Forgery Video Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised, Transformer  
[Paper Link](http://arxiv.org/abs/2309.04795v1)  

---


**ABSTRACT**  
Face forgery videos have caused severe social public concern, and various detectors have been proposed recently. However, most of them are trained in a supervised manner with limited generalization when detecting videos from different forgery methods or real source videos. To tackle this issue, we explore to take full advantage of the difference between real and forgery videos by only exploring the common representation of real face videos. In this paper, a Self-supervised Transformer cooperating with Contrastive and Reconstruction learning (CoReST) is proposed, which is first pre-trained only on real face videos in a self-supervised manner, and then fine-tuned a linear head on specific face forgery video datasets. Two specific auxiliary tasks incorporated contrastive and reconstruction learning are designed to enhance the representation learning. Furthermore, a Domain Adaptive Reconstruction (DAR) module is introduced to bridge the gap between different forgery domains by reconstructing on unlabeled target videos when fine-tuning. Extensive experiments on public datasets demonstrate that our proposed method performs even better than the state-of-the-art supervised competitors with impressive generalization.

{{</citation>}}


### (6/52) When to Learn What: Model-Adaptive Data Augmentation Curriculum (Chengkai Hou et al., 2023)

{{<citation>}}

Chengkai Hou, Jieyu Zhang, Tianyi Zhou. (2023)  
**When to Learn What: Model-Adaptive Data Augmentation Curriculum**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.04747v1)  

---


**ABSTRACT**  
Data augmentation (DA) is widely used to improve the generalization of neural networks by enforcing the invariances and symmetries to pre-defined transformations applied to input data. However, a fixed augmentation policy may have different effects on each sample in different training stages but existing approaches cannot adjust the policy to be adaptive to each sample and the training model. In this paper, we propose Model Adaptive Data Augmentation (MADAug) that jointly trains an augmentation policy network to teach the model when to learn what. Unlike previous work, MADAug selects augmentation operators for each input image by a model-adaptive policy varying between training stages, producing a data augmentation curriculum optimized for better generalization. In MADAug, we train the policy through a bi-level optimization scheme, which aims to minimize a validation-set loss of a model trained using the policy-produced data augmentations. We conduct an extensive evaluation of MADAug on multiple image classification tasks and network architectures with thorough comparisons to existing DA approaches. MADAug outperforms or is on par with other baselines and exhibits better fairness: it brings improvement to all classes and more to the difficult ones. Moreover, MADAug learned policy shows better performance when transferred to fine-grained datasets. In addition, the auto-optimized policy in MADAug gradually introduces increasing perturbations and naturally forms an easy-to-hard curriculum.

{{</citation>}}


### (7/52) Frequency-Aware Self-Supervised Long-Tailed Learning (Ci-Siang Lin et al., 2023)

{{<citation>}}

Ci-Siang Lin, Min-Hung Chen, Yu-Chiang Frank Wang. (2023)  
**Frequency-Aware Self-Supervised Long-Tailed Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.04723v1)  

---


**ABSTRACT**  
Data collected from the real world typically exhibit long-tailed distributions, where frequent classes contain abundant data while rare ones have only a limited number of samples. While existing supervised learning approaches have been proposed to tackle such data imbalance, the requirement of label supervision would limit their applicability to real-world scenarios in which label annotation might not be available. Without the access to class labels nor the associated class frequencies, we propose Frequency-Aware Self-Supervised Learning (FASSL) in this paper. Targeting at learning from unlabeled data with inherent long-tailed distributions, the goal of FASSL is to produce discriminative feature representations for downstream classification tasks. In FASSL, we first learn frequency-aware prototypes, reflecting the associated long-tailed distribution. Particularly focusing on rare-class samples, the relationships between image data and the derived prototypes are further exploited with the introduced self-supervised learning scheme. Experiments on long-tailed image datasets quantitatively and qualitatively verify the effectiveness of our learning scheme.

{{</citation>}}


### (8/52) UnitModule: A Lightweight Joint Image Enhancement Module for Underwater Object Detection (Zhuoyan Liu et al., 2023)

{{<citation>}}

Zhuoyan Liu, Bo Wang, Ye Li, Jiaxian He, Yunfeng Li. (2023)  
**UnitModule: A Lightweight Joint Image Enhancement Module for Underwater Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.04708v1)  

---


**ABSTRACT**  
Underwater object detection faces the problem of underwater image degradation, which affects the performance of the detector. Underwater object detection methods based on noise reduction and image enhancement usually do not provide images preferred by the detector or require additional datasets. In this paper, we propose a plug-and-play Underwater joint image enhancement Module (UnitModule) that provides the input image preferred by the detector. We design an unsupervised learning loss for the joint training of UnitModule with the detector without additional datasets to improve the interaction between UnitModule and the detector. Furthermore, a color cast predictor with the assisting color cast loss and a data augmentation called Underwater Color Random Transfer (UCRT) are designed to improve the performance of UnitModule on underwater images with different color casts. Extensive experiments are conducted on DUO for different object detection models, where UnitModule achieves the highest performance improvement of 2.6 AP for YOLOv5-S and gains the improvement of 3.3 AP on the brand-new test set (URPCtest). And UnitModule significantly improves the performance of all object detection models we test, especially for models with a small number of parameters. In addition, UnitModule with a small number of parameters of 31K has little effect on the inference speed of the original object detection model. Our quantitative and visual analysis also demonstrates the effectiveness of UnitModule in enhancing the input image and improving the perception ability of the detector for object features.

{{</citation>}}


### (9/52) A Spatial-Temporal Deformable Attention based Framework for Breast Lesion Detection in Videos (Chao Qin et al., 2023)

{{<citation>}}

Chao Qin, Jiale Cao, Huazhu Fu, Rao Muhammad Anwer, Fahad Shahbaz Khan. (2023)  
**A Spatial-Temporal Deformable Attention based Framework for Breast Lesion Detection in Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.04702v1)  

---


**ABSTRACT**  
Detecting breast lesion in videos is crucial for computer-aided diagnosis. Existing video-based breast lesion detection approaches typically perform temporal feature aggregation of deep backbone features based on the self-attention operation. We argue that such a strategy struggles to effectively perform deep feature aggregation and ignores the useful local information. To tackle these issues, we propose a spatial-temporal deformable attention based framework, named STNet. Our STNet introduces a spatial-temporal deformable attention module to perform local spatial-temporal feature fusion. The spatial-temporal deformable attention module enables deep feature aggregation in each stage of both encoder and decoder. To further accelerate the detection speed, we introduce an encoder feature shuffle strategy for multi-frame prediction during inference. In our encoder feature shuffle strategy, we share the backbone and encoder features, and shuffle encoder features for decoder to generate the predictions of multiple frames. The experiments on the public breast lesion ultrasound video dataset show that our STNet obtains a state-of-the-art detection performance, while operating twice as fast inference speed. The code and model are available at https://github.com/AlfredQin/STNet.

{{</citation>}}


### (10/52) DeNoising-MOT: Towards Multiple Object Tracking with Severe Occlusions (Teng Fu et al., 2023)

{{<citation>}}

Teng Fu, Xiaocong Wang, Haiyang Yu, Ke Niu, Bin Li, Xiangyang Xue. (2023)  
**DeNoising-MOT: Towards Multiple Object Tracking with Severe Occlusions**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.04682v1)  

---


**ABSTRACT**  
Multiple object tracking (MOT) tends to become more challenging when severe occlusions occur. In this paper, we analyze the limitations of traditional Convolutional Neural Network-based methods and Transformer-based methods in handling occlusions and propose DNMOT, an end-to-end trainable DeNoising Transformer for MOT. To address the challenge of occlusions, we explicitly simulate the scenarios when occlusions occur. Specifically, we augment the trajectory with noises during training and make our model learn the denoising process in an encoder-decoder architecture, so that our model can exhibit strong robustness and perform well under crowded scenes. Additionally, we propose a Cascaded Mask strategy to better coordinate the interaction between different types of queries in the decoder to prevent the mutual suppression between neighboring trajectories under crowded scenes. Notably, the proposed method requires no additional modules like matching strategy and motion state estimation in inference. We conduct extensive experiments on the MOT17, MOT20, and DanceTrack datasets, and the experimental results show that our method outperforms previous state-of-the-art methods by a clear margin.

{{</citation>}}


### (11/52) BiLMa: Bidirectional Local-Matching for Text-based Person Re-identification (Takuro Fujii et al., 2023)

{{<citation>}}

Takuro Fujii, Shuhei Tarashima. (2023)  
**BiLMa: Bidirectional Local-Matching for Text-based Person Re-identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.04675v1)  

---


**ABSTRACT**  
Text-based person re-identification (TBPReID) aims to retrieve person images represented by a given textual query. In this task, how to effectively align images and texts globally and locally is a crucial challenge. Recent works have obtained high performances by solving Masked Language Modeling (MLM) to align image/text parts. However, they only performed uni-directional (i.e., from image to text) local-matching, leaving room for improvement by introducing opposite-directional (i.e., from text to image) local-matching. In this work, we introduce Bidirectional Local-Matching (BiLMa) framework that jointly optimize MLM and Masked Image Modeling (MIM) in TBPReID model training. With this framework, our model is trained so as the labels of randomly masked both image and text tokens are predicted by unmasked tokens. In addition, to narrow the semantic gap between image and text in MIM, we propose Semantic MIM (SemMIM), in which the labels of masked image tokens are automatically given by a state-of-the-art human parser. Experimental results demonstrate that our BiLMa framework with SemMIM achieves state-of-the-art Rank@1 and mAP scores on three benchmarks.

{{</citation>}}


### (12/52) Unified Language-Vision Pretraining with Dynamic Discrete Visual Tokenization (Yang Jin et al., 2023)

{{<citation>}}

Yang Jin, Kun Xu, Kun Xu, Liwei Chen, Chao Liao, Jianchao Tan, Bin Chen, Chenyi Lei, An Liu, Chengru Song, Xiaoqiang Lei, Yadong Mu, Di Zhang, Wenwu Ou, Kun Gai. (2023)  
**Unified Language-Vision Pretraining with Dynamic Discrete Visual Tokenization**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2309.04669v1)  

---


**ABSTRACT**  
Recently, the remarkable advance of the Large Language Model (LLM) has inspired researchers to transfer its extraordinary reasoning capability to data across several modalities. The prevailing approaches primarily regard visual input as the prompt and focus exclusively on optimizing the text generation process conditioned upon vision content by a frozen LLM. Such an inequitable treatment of vision and language heavily constrains the model's potential. In this paper, we break through this limitation by representing both vision and language in a unified representation. To this end, we craft a visual tokenizer that translates the non-linguistic image into a sequence of discrete tokens like a foreign language that LLM can read. The resulting visual tokens encompass high-level semantics worthy of a word and also support dynamic sequence length varying from the image content. Coped with this visual tokenizer, the presented foundation model called LaVIT (Language-VIsion Transformer) can handle both image and text indiscriminately under a unified generative learning paradigm. Pre-trained on the web-scale image-text corpus, LaVIT is empowered with impressive multi-modal comprehension capability. The extensive experiments showcase that it outperforms existing models by a large margin on downstream tasks. Our code and models will be available at https://github.com/jy0205/LaVIT.

{{</citation>}}


## cs.HC (3)



### (13/52) Evaluating Chatbots to Promote Users' Trust -- Practices and Open Problems (Biplav Srivastava et al., 2023)

{{<citation>}}

Biplav Srivastava, Kausik Lakkaraju, Tarmo Koppel, Vignesh Narayanan, Ashish Kundu, Sachindra Joshi. (2023)  
**Evaluating Chatbots to Promote Users' Trust -- Practices and Open Problems**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs-SE, cs.HC  
Keywords: AI, ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.05680v2)  

---


**ABSTRACT**  
Chatbots, the common moniker for collaborative assistants, are Artificial Intelligence (AI) software that enables people to naturally interact with them to get tasks done. Although chatbots have been studied since the dawn of AI, they have particularly caught the imagination of the public and businesses since the launch of easy-to-use and general-purpose Large Language Model-based chatbots like ChatGPT. As businesses look towards chatbots as a potential technology to engage users, who may be end customers, suppliers, or even their own employees, proper testing of chatbots is important to address and mitigate issues of trust related to service or product performance, user satisfaction and long-term unintended consequences for society. This paper reviews current practices for chatbot testing, identifies gaps as open problems in pursuit of user trust, and outlines a path forward.

{{</citation>}}


### (14/52) A Visual Analytic Environment to Co-locate Peoples' Tweets with City Factual Data (Snehal Patil et al., 2023)

{{<citation>}}

Snehal Patil, Shah Rukh Humayoun. (2023)  
**A Visual Analytic Environment to Co-locate Peoples' Tweets with City Factual Data**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Social Media, Twitter  
[Paper Link](http://arxiv.org/abs/2309.04724v1)  

---


**ABSTRACT**  
Social Media platforms (e.g., Twitter, Facebook, etc.) are used heavily by public to provide news, opinions, and reactions towards events or topics. Integrating such data with the event or topic factual data could provide a more comprehensive understanding of the underlying event or topic. Targeting this, we present our visual analytics tool, called VC-FaT, that integrates peoples' tweet data regarding crimes in San Francisco city with the city factual crime data. VC-FaT provides a number of interactive visualizations using both data sources for better understanding and exploration of crime activities happened in the city during a period of five years.

{{</citation>}}


### (15/52) TECVis: A Visual Analytics Tool to Compare People's Emotion Feelings (Ilya Nemtsov et al., 2023)

{{<citation>}}

Ilya Nemtsov, MST Jasmine Jahan, Chuting Yan, Shah Rukh Humayoun. (2023)  
**TECVis: A Visual Analytics Tool to Compare People's Emotion Feelings**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2309.04722v1)  

---


**ABSTRACT**  
Twitter is one of the popular social media platforms where people share news or reactions towards an event or topic using short text messages called "tweets". Emotion analysis in these tweets can play a vital role in understanding peoples' feelings towards the underlying event or topic. In this work, we present our visual analytics tool, called TECVis, that focuses on providing comparison views of peoples' emotion feelings in tweets towards an event or topic. The comparison is done based on geolocations or timestamps. TECVis provides several interaction and filtering options for navigation and better exploration of underlying tweet data for emotion feelings comparison.

{{</citation>}}


## cs.LG (7)



### (16/52) Symplectic Structure-Aware Hamiltonian (Graph) Embeddings (Jiaxu Liu et al., 2023)

{{<citation>}}

Jiaxu Liu, Xinping Yi, Tianle Zhang, Xiaowei Huang. (2023)  
**Symplectic Structure-Aware Hamiltonian (Graph) Embeddings**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-SG  
Keywords: Embedding, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.04885v1)  

---


**ABSTRACT**  
In traditional Graph Neural Networks (GNNs), the assumption of a fixed embedding manifold often limits their adaptability to diverse graph geometries. Recently, Hamiltonian system-inspired GNNs are proposed to address the dynamic nature of such embeddings by incorporating physical laws into node feature updates. In this work, we present SAH-GNN, a novel approach that generalizes Hamiltonian dynamics for more flexible node feature updates. Unlike existing Hamiltonian-inspired GNNs, SAH-GNN employs Riemannian optimization on the symplectic Stiefel manifold to adaptively learn the underlying symplectic structure during training, circumventing the limitations of existing Hamiltonian GNNs that rely on a pre-defined form of standard symplectic structure. This innovation allows SAH-GNN to automatically adapt to various graph datasets without extensive hyperparameter tuning. Moreover, it conserves energy during training such that the implicit Hamiltonian system is physically meaningful. To this end, we empirically validate SAH-GNN's superior performance and adaptability in node classification tasks across multiple types of graph datasets.

{{</citation>}}


### (17/52) Reverse-Engineering Decoding Strategies Given Blackbox Access to a Language Generation System (Daphne Ippolito et al., 2023)

{{<citation>}}

Daphne Ippolito, Nicholas Carlini, Katherine Lee, Milad Nasr, Yun William Yu. (2023)  
**Reverse-Engineering Decoding Strategies Given Blackbox Access to a Language Generation System**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.04858v1)  

---


**ABSTRACT**  
Neural language models are increasingly deployed into APIs and websites that allow a user to pass in a prompt and receive generated text. Many of these systems do not reveal generation parameters. In this paper, we present methods to reverse-engineer the decoding method used to generate text (i.e., top-$k$ or nucleus sampling). Our ability to discover which decoding strategy was used has implications for detecting generated text. Additionally, the process of discovering the decoding strategy can reveal biases caused by selecting decoding settings which severely truncate a model's predicted distributions. We perform our attack on several families of open-source language models, as well as on production systems (e.g., ChatGPT).

{{</citation>}}


### (18/52) RR-CP: Reliable-Region-Based Conformal Prediction for Trustworthy Medical Image Classification (Yizhe Zhang et al., 2023)

{{<citation>}}

Yizhe Zhang, Shuo Wang, Yejia Zhang, Danny Z. Chen. (2023)  
**RR-CP: Reliable-Region-Based Conformal Prediction for Trustworthy Medical Image Classification**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI, Image Classification  
[Paper Link](http://arxiv.org/abs/2309.04760v1)  

---


**ABSTRACT**  
Conformal prediction (CP) generates a set of predictions for a given test sample such that the prediction set almost always contains the true label (e.g., 99.5\% of the time). CP provides comprehensive predictions on possible labels of a given test sample, and the size of the set indicates how certain the predictions are (e.g., a set larger than one is `uncertain'). Such distinct properties of CP enable effective collaborations between human experts and medical AI models, allowing efficient intervention and quality check in clinical decision-making. In this paper, we propose a new method called Reliable-Region-Based Conformal Prediction (RR-CP), which aims to impose a stronger statistical guarantee so that the user-specified error rate (e.g., 0.5\%) can be achieved in the test time, and under this constraint, the size of the prediction set is optimized (to be small). We consider a small prediction set size an important measure only when the user-specified error rate is achieved. Experiments on five public datasets show that our RR-CP performs well: with a reasonably small-sized prediction set, it achieves the user-specified error rate (e.g., 0.5\%) significantly more frequently than exiting CP methods.

{{</citation>}}


### (19/52) A Spatiotemporal Deep Neural Network for Fine-Grained Multi-Horizon Wind Prediction (Fanling Huang et al., 2023)

{{<citation>}}

Fanling Huang, Yangdong Deng. (2023)  
**A Spatiotemporal Deep Neural Network for Fine-Grained Multi-Horizon Wind Prediction**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Seq2Seq  
[Paper Link](http://arxiv.org/abs/2309.04733v1)  

---


**ABSTRACT**  
The prediction of wind in terms of both wind speed and direction, which has a crucial impact on many real-world applications like aviation and wind power generation, is extremely challenging due to the high stochasticity and complicated correlation in the weather data. Existing methods typically focus on a sub-set of influential factors and thus lack a systematic treatment of the problem. In addition, fine-grained forecasting is essential for efficient industry operations, but has been less attended in the literature. In this work, we propose a novel data-driven model, Multi-Horizon SpatioTemporal Network (MHSTN), generally for accurate and efficient fine-grained wind prediction. MHSTN integrates multiple deep neural networks targeting different factors in a sequence-to-sequence (Seq2Seq) backbone to effectively extract features from various data sources and produce multi-horizon predictions for all sites within a given region. MHSTN is composed of four major modules. First, a temporal module fuses coarse-grained forecasts derived by Numerical Weather Prediction (NWP) and historical on-site observation data at stations so as to leverage both global and local atmospheric information. Second, a spatial module exploits spatial correlation by modeling the joint representation of all stations. Third, an ensemble module weighs the above two modules for final predictions. Furthermore, a covariate selection module automatically choose influential meteorological variables as initial input. MHSTN is already integrated into the scheduling platform of one of the busiest international airports of China. The evaluation results demonstrate that our model outperforms competitors by a significant margin.

{{</citation>}}


### (20/52) TCGAN: Convolutional Generative Adversarial Network for Time Series Classification and Clustering (Fanling Huang et al., 2023)

{{<citation>}}

Fanling Huang, Yangdong Deng. (2023)  
**TCGAN: Convolutional Generative Adversarial Network for Time Series Classification and Clustering**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2309.04732v1)  

---


**ABSTRACT**  
Recent works have demonstrated the superiority of supervised Convolutional Neural Networks (CNNs) in learning hierarchical representations from time series data for successful classification. These methods require sufficiently large labeled data for stable learning, however acquiring high-quality labeled time series data can be costly and potentially infeasible. Generative Adversarial Networks (GANs) have achieved great success in enhancing unsupervised and semi-supervised learning. Nonetheless, to our best knowledge, it remains unclear how effectively GANs can serve as a general-purpose solution to learn representations for time series recognition, i.e., classification and clustering. The above considerations inspire us to introduce a Time-series Convolutional GAN (TCGAN). TCGAN learns by playing an adversarial game between two one-dimensional CNNs (i.e., a generator and a discriminator) in the absence of label information. Parts of the trained TCGAN are then reused to construct a representation encoder to empower linear recognition methods. We conducted comprehensive experiments on synthetic and real-world datasets. The results demonstrate that TCGAN is faster and more accurate than existing time-series GANs. The learned representations enable simple classification and clustering methods to achieve superior and stable performance. Furthermore, TCGAN retains high efficacy in scenarios with few-labeled and imbalanced-labeled data. Our work provides a promising path to effectively utilize abundant unlabeled time series data.

{{</citation>}}


### (21/52) Toward Reproducing Network Research Results Using Large Language Models (Qiao Xiang et al., 2023)

{{<citation>}}

Qiao Xiang, Yuling Lin, Mingjun Fang, Bang Huang, Siyong Huang, Ridi Wen, Franck Le, Linghe Kong, Jiwu Shu. (2023)  
**Toward Reproducing Network Research Results Using Large Language Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-LG, cs.LG  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.04716v1)  

---


**ABSTRACT**  
Reproducing research results in the networking community is important for both academia and industry. The current best practice typically resorts to three approaches: (1) looking for publicly available prototypes; (2) contacting the authors to get a private prototype; and (3) manually implementing a prototype following the description of the publication. However, most published network research does not have public prototypes and private prototypes are hard to get. As such, most reproducing efforts are spent on manual implementation based on the publications, which is both time and labor consuming and error-prone. In this paper, we boldly propose reproducing network research results using the emerging large language models (LLMs). In particular, we first prove its feasibility with a small-scale experiment, in which four students with essential networking knowledge each reproduces a different networking system published in prominent conferences and journals by prompt engineering ChatGPT. We report the experiment's observations and lessons and discuss future open research questions of this proposal. This work raises no ethical issue.

{{</citation>}}


### (22/52) Redundancy-Free Self-Supervised Relational Learning for Graph Clustering (Si-Yu Yi et al., 2023)

{{<citation>}}

Si-Yu Yi, Wei Ju, Yifang Qin, Xiao Luo, Luchen Liu, Yong-Dao Zhou, Ming Zhang. (2023)  
**Redundancy-Free Self-Supervised Relational Learning for Graph Clustering**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2309.04694v1)  

---


**ABSTRACT**  
Graph clustering, which learns the node representations for effective cluster assignments, is a fundamental yet challenging task in data analysis and has received considerable attention accompanied by graph neural networks in recent years. However, most existing methods overlook the inherent relational information among the non-independent and non-identically distributed nodes in a graph. Due to the lack of exploration of relational attributes, the semantic information of the graph-structured data fails to be fully exploited which leads to poor clustering performance. In this paper, we propose a novel self-supervised deep graph clustering method named Relational Redundancy-Free Graph Clustering (R$^2$FGC) to tackle the problem. It extracts the attribute- and structure-level relational information from both global and local views based on an autoencoder and a graph autoencoder. To obtain effective representations of the semantic information, we preserve the consistent relation among augmented nodes, whereas the redundant relation is further reduced for learning discriminative embeddings. In addition, a simple yet valid strategy is utilized to alleviate the over-smoothing issue. Extensive experiments are performed on widely used benchmark datasets to validate the superiority of our R$^2$FGC over state-of-the-art baselines. Our codes are available at https://github.com/yisiyu95/R2FGC.

{{</citation>}}


## eess.SP (1)



### (23/52) Recall-driven Precision Refinement: Unveiling Accurate Fall Detection using LSTM (Rishabh Mondal et al., 2023)

{{<citation>}}

Rishabh Mondal, Prasun Ghosal. (2023)  
**Recall-driven Precision Refinement: Unveiling Accurate Fall Detection using LSTM**  

---
Primary Category: eess.SP  
Categories: cs-AI, eess-SP, eess.SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2309.07154v1)  

---


**ABSTRACT**  
This paper presents an innovative approach to address the pressing concern of fall incidents among the elderly by developing an accurate fall detection system. Our proposed system combines state-of-the-art technologies, including accelerometer and gyroscope sensors, with deep learning models, specifically Long Short-Term Memory (LSTM) networks. Real-time execution capabilities are achieved through the integration of Raspberry Pi hardware. We introduce pruning techniques that strategically fine-tune the LSTM model's architecture and parameters to optimize the system's performance. We prioritize recall over precision, aiming to accurately identify falls and minimize false negatives for timely intervention. Extensive experimentation and meticulous evaluation demonstrate remarkable performance metrics, emphasizing a high recall rate while maintaining a specificity of 96\%. Our research culminates in a state-of-the-art fall detection system that promptly sends notifications, ensuring vulnerable individuals receive timely assistance and improve their overall well-being. Applying LSTM models and incorporating pruning techniques represent a significant advancement in fall detection technology, offering an effective and reliable fall prevention and intervention solution.

{{</citation>}}


## cs.CL (15)



### (24/52) Distributional Data Augmentation Methods for Low Resource Language (Mosleh Mahamud et al., 2023)

{{<citation>}}

Mosleh Mahamud, Zed Lee, Isak Samsten. (2023)  
**Distributional Data Augmentation Methods for Low Resource Language**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Augmentation, NLP  
[Paper Link](http://arxiv.org/abs/2309.04862v1)  

---


**ABSTRACT**  
Text augmentation is a technique for constructing synthetic data from an under-resourced corpus to improve predictive performance. Synthetic data generation is common in numerous domains. However, recently text augmentation has emerged in natural language processing (NLP) to improve downstream tasks. One of the current state-of-the-art text augmentation techniques is easy data augmentation (EDA), which augments the training data by injecting and replacing synonyms and randomly permuting sentences. One major obstacle with EDA is the need for versatile and complete synonym dictionaries, which cannot be easily found in low-resource languages. To improve the utility of EDA, we propose two extensions, easy distributional data augmentation (EDDA) and type specific similar word replacement (TSSR), which uses semantic word context information and part-of-speech tags for word replacement and augmentation. In an extensive empirical evaluation, we show the utility of the proposed methods, measured by F1 score, on two representative datasets in Swedish as an example of a low-resource language. With the proposed methods, we show that augmented data improve classification performances in low-resource settings.

{{</citation>}}


### (25/52) Speech Emotion Recognition with Distilled Prosodic and Linguistic Affect Representations (Debaditya Shome et al., 2023)

{{<citation>}}

Debaditya Shome, Ali Etemad. (2023)  
**Speech Emotion Recognition with Distilled Prosodic and Linguistic Affect Representations**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2309.04849v1)  

---


**ABSTRACT**  
We propose EmoDistill, a novel speech emotion recognition (SER) framework that leverages cross-modal knowledge distillation during training to learn strong linguistic and prosodic representations of emotion from speech. During inference, our method only uses a stream of speech signals to perform unimodal SER thus reducing computation overhead and avoiding run-time transcription and prosodic feature extraction errors. During training, our method distills information at both embedding and logit levels from a pair of pre-trained Prosodic and Linguistic teachers that are fine-tuned for SER. Experiments on the IEMOCAP benchmark demonstrate that our method outperforms other unimodal and multimodal techniques by a considerable margin, and achieves state-of-the-art performance of 77.49% unweighted accuracy and 78.91% weighted accuracy. Detailed ablation studies demonstrate the impact of each component of our method.

{{</citation>}}


### (26/52) Leveraging Large Language Models for Exploiting ASR Uncertainty (Pranay Dighe et al., 2023)

{{<citation>}}

Pranay Dighe, Yi Su, Shangshang Zheng, Yunshu Liu, Vineet Garg, Xiaochuan Niu, Ahmed Tewfik. (2023)  
**Leveraging Large Language Models for Exploiting ASR Uncertainty**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs-SD, cs.CL, eess-AS  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2309.04842v2)  

---


**ABSTRACT**  
While large language models excel in a variety of natural language processing (NLP) tasks, to perform well on spoken language understanding (SLU) tasks, they must either rely on off-the-shelf automatic speech recognition (ASR) systems for transcription, or be equipped with an in-built speech modality. This work focuses on the former scenario, where LLM's accuracy on SLU tasks is constrained by the accuracy of a fixed ASR system on the spoken input. Specifically, we tackle speech-intent classification task, where a high word-error-rate can limit the LLM's ability to understand the spoken intent. Instead of chasing a high accuracy by designing complex or specialized architectures regardless of deployment costs, we seek to answer how far we can go without substantially changing the underlying ASR and LLM, which can potentially be shared by multiple unrelated tasks. To this end, we propose prompting the LLM with an n-best list of ASR hypotheses instead of only the error-prone 1-best hypothesis. We explore prompt-engineering to explain the concept of n-best lists to the LLM; followed by the finetuning of Low-Rank Adapters on the downstream tasks. Our approach using n-best lists proves to be effective on a device-directed speech detection task as well as on a keyword spotting task, where systems using n-best list prompts outperform those using 1-best ASR hypothesis; thus paving the way for an efficient method to exploit ASR uncertainty via LLMs for speech-based applications.

{{</citation>}}


### (27/52) Neurons in Large Language Models: Dead, N-gram, Positional (Elena Voita et al., 2023)

{{<citation>}}

Elena Voita, Javier Ferrando, Christoforos Nalmpantis. (2023)  
**Neurons in Large Language Models: Dead, N-gram, Positional**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.04827v1)  

---


**ABSTRACT**  
We analyze a family of large language models in such a lightweight manner that can be done on a single GPU. Specifically, we focus on the OPT family of models ranging from 125m to 66b parameters and rely only on whether an FFN neuron is activated or not. First, we find that the early part of the network is sparse and represents many discrete features. Here, many neurons (more than 70% in some layers of the 66b model) are "dead", i.e. they never activate on a large collection of diverse data. At the same time, many of the alive neurons are reserved for discrete features and act as token and n-gram detectors. Interestingly, their corresponding FFN updates not only promote next token candidates as could be expected, but also explicitly focus on removing the information about triggering them tokens, i.e., current input. To the best of our knowledge, this is the first example of mechanisms specialized at removing (rather than adding) information from the residual stream. With scale, models become more sparse in a sense that they have more dead neurons and token detectors. Finally, some neurons are positional: them being activated or not depends largely (or solely) on position and less so (or not at all) on textual data. We find that smaller models have sets of neurons acting as position range indicators while larger models operate in a less explicit manner.

{{</citation>}}


### (28/52) FaNS: a Facet-based Narrative Similarity Metric (Mousumi Akter et al., 2023)

{{<citation>}}

Mousumi Akter, Shubhra Kanti Karmaker Santu. (2023)  
**FaNS: a Facet-based Narrative Similarity Metric**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.04823v1)  

---


**ABSTRACT**  
Similar Narrative Retrieval is a crucial task since narratives are essential for explaining and understanding events, and multiple related narratives often help to create a holistic view of the event of interest. To accurately identify semantically similar narratives, this paper proposes a novel narrative similarity metric called Facet-based Narrative Similarity (FaNS), based on the classic 5W1H facets (Who, What, When, Where, Why, and How), which are extracted by leveraging the state-of-the-art Large Language Models (LLMs). Unlike existing similarity metrics that only focus on overall lexical/semantic match, FaNS provides a more granular matching along six different facets independently and then combines them. To evaluate FaNS, we created a comprehensive dataset by collecting narratives from AllSides, a third-party news portal. Experimental results demonstrate that the FaNS metric exhibits a higher correlation (37\% higher) than traditional text similarity metrics that directly measure the lexical/semantic match between narratives, demonstrating its effectiveness in comparing the finer details between a pair of narratives.

{{</citation>}}


### (29/52) MMHQA-ICL: Multimodal In-context Learning for Hybrid Question Answering over Text, Tables and Images (Weihao Liu et al., 2023)

{{<citation>}}

Weihao Liu, Fangyu Lei, Tongxu Luo, Jiahe Lei, Shizhu He, Jun Zhao, Kang Liu. (2023)  
**MMHQA-ICL: Multimodal In-context Learning for Hybrid Question Answering over Text, Tables and Images**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.04790v1)  

---


**ABSTRACT**  
In the real world, knowledge often exists in a multimodal and heterogeneous form. Addressing the task of question answering with hybrid data types, including text, tables, and images, is a challenging task (MMHQA). Recently, with the rise of large language models (LLM), in-context learning (ICL) has become the most popular way to solve QA problems. We propose MMHQA-ICL framework for addressing this problems, which includes stronger heterogeneous data retriever and an image caption module. Most importantly, we propose a Type-specific In-context Learning Strategy for MMHQA, enabling LLMs to leverage their powerful performance in this task. We are the first to use end-to-end LLM prompting method for this task. Experimental results demonstrate that our framework outperforms all baselines and methods trained on the full dataset, achieving state-of-the-art results under the few-shot setting on the MultimodalQA dataset.

{{</citation>}}


### (30/52) SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning (Bin Wang et al., 2023)

{{<citation>}}

Bin Wang, Zhengyuan Liu, Xin Huang, Fangkai Jiao, Yang Ding, Ai Ti Aw, Nancy F. Chen. (2023)  
**SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Multilingual, NLP, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.04766v1)  

---


**ABSTRACT**  
We present SeaEval, a benchmark for multilingual foundation models. In addition to characterizing how these models understand and reason with natural language, we also investigate how well they comprehend cultural practices, nuances, and values. Alongside standard accuracy metrics, we investigate the brittleness of foundation models in the dimensions of semantics and multilinguality. Our analyses span both open-sourced and closed models, leading to empirical results across classic NLP tasks, reasoning, and cultural comprehension. Key findings indicate (1) Most models exhibit varied behavior when given paraphrased instructions. (2) Many models still suffer from exposure bias (e.g., positional bias, majority label bias). (3) For questions rooted in factual, scientific, and commonsense knowledge, consistent responses are expected across multilingual queries that are semantically equivalent. Yet, most models surprisingly demonstrate inconsistent performance on these queries. (4) Multilingually-trained models have not attained "balanced multilingual" capabilities. Our endeavors underscore the need for more generalizable semantic representations and enhanced multilingual contextualization. SeaEval can serve as a launchpad for more thorough investigations and evaluations for multilingual and multicultural scenarios.

{{</citation>}}


### (31/52) Data Augmentation for Conversational AI (Heydar Soudani et al., 2023)

{{<citation>}}

Heydar Soudani, Evangelos Kanoulas, Faegheh Hasibi. (2023)  
**Data Augmentation for Conversational AI**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-IR, cs.CL  
Keywords: AI, Augmentation  
[Paper Link](http://arxiv.org/abs/2309.04739v1)  

---


**ABSTRACT**  
Advancements in conversational systems have revolutionized information access, surpassing the limitations of single queries. However, developing dialogue systems requires a large amount of training data, which is a challenge in low-resource domains and languages. Traditional data collection methods like crowd-sourcing are labor-intensive and time-consuming, making them ineffective in this context. Data augmentation (DA) is an affective approach to alleviate the data scarcity problem in conversational systems. This tutorial provides a comprehensive and up-to-date overview of DA approaches in the context of conversational systems. It highlights recent advances in conversation augmentation, open domain and task-oriented conversation generation, and different paradigms of evaluating these models. We also discuss current challenges and future directions in order to help researchers and practitioners to further advance the field in this area.

{{</citation>}}


### (32/52) EPA: Easy Prompt Augmentation on Large Language Models via Multiple Sources and Multiple Targets (Hongyuan Lu et al., 2023)

{{<citation>}}

Hongyuan Lu, Wai Lam. (2023)  
**EPA: Easy Prompt Augmentation on Large Language Models via Multiple Sources and Multiple Targets**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation, Language Model, NLP, NLU  
[Paper Link](http://arxiv.org/abs/2309.04725v1)  

---


**ABSTRACT**  
Large language models (LLMs) have shown promising performance on various NLP tasks via task prompting. And their performance can be further improved by appending task demonstrations to the head of the prompt. And usually, a better performance can be achieved with more demonstrations. However, asking the users to write the demonstrations can be cumbersome. As a simple yet cost-effective workaround, this paper proposes a novel method called EPA (\textbf{E}asy \textbf{P}rompt \textbf{A}ugmentation)\footnote{While this paper considers augmenting prompts via demonstrations, we name it EPA as the name EDA is already taken by a well-known NLP method \citep{wei-zou-2019-eda}.} that effectively minimizes user efforts in writing demonstrations while improving the model performance at the same time. EPA achieves these goals by automatically augmenting the demonstrations with multiple sources/targets, where each of them paraphrases each other. This is well motivated as augmenting data via paraphrasing effectively improves neural language models. EPA thus employs paraphrasing as an augmentation method for in-context learning. Extensive experiments indicate that EPA effectively improves both NLU and NLG tasks, covering from natural language inference to machine translation in translating tens of languages.\footnote{Code and data will be released upon publication.}

{{</citation>}}


### (33/52) Analysis of Disinformation and Fake News Detection Using Fine-Tuned Large Language Model (Bohdan M. Pavlyshenko, 2023)

{{<citation>}}

Bohdan M. Pavlyshenko. (2023)  
**Analysis of Disinformation and Fake News Detection Using Fine-Tuned Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-IR, cs-LG, cs.CL  
Keywords: Fake News, Language Model  
[Paper Link](http://arxiv.org/abs/2309.04704v1)  

---


**ABSTRACT**  
The paper considers the possibility of fine-tuning Llama 2 large language model (LLM) for the disinformation analysis and fake news detection. For fine-tuning, the PEFT/LoRA based approach was used. In the study, the model was fine-tuned for the following tasks: analysing a text on revealing disinformation and propaganda narratives, fact checking, fake news detection, manipulation analytics, extracting named entities with their sentiments. The obtained results show that the fine-tuned Llama 2 model can perform a deep analysis of texts and reveal complex styles and narratives. Extracted sentiments for named entities can be considered as predictive features in supervised machine learning models.

{{</citation>}}


### (34/52) Code-Style In-Context Learning for Knowledge-Based Question Answering (Zhijie Nie et al., 2023)

{{<citation>}}

Zhijie Nie, Richong Zhang, Zhongyuan Wang, Xudong Liu. (2023)  
**Code-Style In-Context Learning for Knowledge-Based Question Answering**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2309.04695v1)  

---


**ABSTRACT**  
Current methods for Knowledge-Based Question Answering (KBQA) usually rely on complex training techniques and model frameworks, leading to many limitations in practical applications. Recently, the emergence of In-Context Learning (ICL) capabilities in Large Language Models (LLMs) provides a simple and training-free semantic parsing paradigm for KBQA: Given a small number of questions and their labeled logical forms as demo examples, LLMs can understand the task intent and generate the logic form for a new question. However, current powerful LLMs have little exposure to logic forms during pre-training, resulting in a high format error rate. To solve this problem, we propose a code-style in-context learning method for KBQA, which converts the generation process of unfamiliar logical form into the more familiar code generation process for LLMs. Experimental results on three mainstream datasets show that our method dramatically mitigated the formatting error problem in generating logic forms while realizing a new SOTA on WebQSP, GrailQA, and GraphQ under the few-shot setting.

{{</citation>}}


### (35/52) Embedding structure matters: Comparing methods to adapt multilingual vocabularies to new languages (C. M. Downey et al., 2023)

{{<citation>}}

C. M. Downey, Terra Blevins, Nora Goldfine, Shane Steinert-Threlkeld. (2023)  
**Embedding structure matters: Comparing methods to adapt multilingual vocabularies to new languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Embedding, NLP  
[Paper Link](http://arxiv.org/abs/2309.04679v1)  

---


**ABSTRACT**  
Pre-trained multilingual language models underpin a large portion of modern NLP tools outside of English. A strong baseline for specializing these models for specific languages is Language-Adaptive Pre-Training (LAPT). However, retaining a large cross-lingual vocabulary and embedding matrix comes at considerable excess computational cost during adaptation. In this study, we propose several simple techniques to replace a cross-lingual vocabulary with a compact, language-specific one. Namely, we address strategies for re-initializing the token embedding matrix after vocabulary specialization. We then provide a systematic experimental comparison of our techniques, in addition to the recently-proposed Focus method. We demonstrate that: 1) Embedding-replacement techniques in the monolingual transfer literature are inadequate for adapting multilingual models. 2) Replacing cross-lingual vocabularies with smaller specialized ones provides an efficient method to improve performance in low-resource languages. 3) Simple embedding re-initialization techniques based on script-wise sub-distributions rival techniques such as Focus, which rely on similarity scores obtained from an auxiliary model.

{{</citation>}}


### (36/52) MADLAD-400: A Multilingual And Document-Level Large Audited Dataset (Sneha Kudugunta et al., 2023)

{{<citation>}}

Sneha Kudugunta, Isaac Caswell, Biao Zhang, Xavier Garcia, Christopher A. Choquette-Choo, Katherine Lee, Derrick Xin, Aditya Kusupati, Romi Stella, Ankur Bapna, Orhan Firat. (2023)  
**MADLAD-400: A Multilingual And Document-Level Large Audited Dataset**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2309.04662v1)  

---


**ABSTRACT**  
We introduce MADLAD-400, a manually audited, general domain 3T token monolingual dataset based on CommonCrawl, spanning 419 languages. We discuss the limitations revealed by self-auditing MADLAD-400, and the role data auditing had in the dataset creation process. We then train and release a 10.7B-parameter multilingual machine translation model on 250 billion tokens covering over 450 languages using publicly available data, and find that it is competitive with models that are significantly larger, and report the results on different domains. In addition, we train a 8B-parameter language model, and assess the results on few-shot translation. We make the baseline models available to the research community.

{{</citation>}}


### (37/52) Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf (Yuzhuang Xu et al., 2023)

{{<citation>}}

Yuzhuang Xu, Shuo Wang, Peng Li, Fuwen Luo, Xiaolong Wang, Weidong Liu, Yang Liu. (2023)  
**Exploring Large Language Models for Communication Games: An Empirical Study on Werewolf**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.04658v1)  

---


**ABSTRACT**  
Communication games, which we refer to as incomplete information games that heavily depend on natural language communication, hold significant research value in fields such as economics, social science, and artificial intelligence. In this work, we explore the problem of how to engage large language models (LLMs) in communication games, and in response, propose a tuning-free framework. Our approach keeps LLMs frozen, and relies on the retrieval and reflection on past communications and experiences for improvement. An empirical study on the representative and widely-studied communication game, ``Werewolf'', demonstrates that our framework can effectively play Werewolf game without tuning the parameters of the LLMs. More importantly, strategic behaviors begin to emerge in our experiments, suggesting that it will be a fruitful journey to engage LLMs in communication games and associated domains.

{{</citation>}}


### (38/52) Efficient Finetuning Large Language Models For Vietnamese Chatbot (Vu-Thuan Doan et al., 2023)

{{<citation>}}

Vu-Thuan Doan, Quoc-Truong Truong, Duc-Vu Nguyen, Vinh-Tiep Nguyen, Thuy-Ngan Nguyen Luu. (2023)  
**Efficient Finetuning Large Language Models For Vietnamese Chatbot**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model, Multilingual, PaLM  
[Paper Link](http://arxiv.org/abs/2309.04646v1)  

---


**ABSTRACT**  
Large language models (LLMs), such as GPT-4, PaLM, and LLaMa, have been shown to achieve remarkable performance across a variety of natural language tasks. Recent advancements in instruction tuning bring LLMs with ability in following user's instructions and producing human-like responses. However, the high costs associated with training and implementing LLMs pose challenges to academic research. Furthermore, the availability of pretrained LLMs and instruction-tune datasets for Vietnamese language is limited. To tackle these concerns, we leverage large-scale instruction-following datasets from open-source projects, namely Alpaca, GPT4All, and Chat-Doctor, which cover general domain and specific medical domain. To the best of our knowledge, these are the first instructional dataset for Vietnamese. Subsequently, we utilize parameter-efficient tuning through Low-Rank Adaptation (LoRA) on two open LLMs: Bloomz (Multilingual) and GPTJ-6B (Vietnamese), resulting four models: Bloomz-Chat, Bloomz-Doctor, GPTJ-Chat, GPTJ-Doctor.Finally, we assess the effectiveness of our methodology on a per-sample basis, taking into consideration the helpfulness, relevance, accuracy, level of detail in their responses. This evaluation process entails the utilization of GPT-4 as an automated scoring mechanism. Despite utilizing a low-cost setup, our method demonstrates about 20-30\% improvement over the original models in our evaluation tasks.

{{</citation>}}


## eess.SY (2)



### (39/52) Verifiable Reinforcement Learning Systems via Compositionality (Cyrus Neary et al., 2023)

{{<citation>}}

Cyrus Neary, Aryaman Singh Samyal, Christos Verginis, Murat Cubuktepe, Ufuk Topcu. (2023)  
**Verifiable Reinforcement Learning Systems via Compositionality**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.06420v1)  

---


**ABSTRACT**  
We propose a framework for verifiable and compositional reinforcement learning (RL) in which a collection of RL subsystems, each of which learns to accomplish a separate subtask, are composed to achieve an overall task. The framework consists of a high-level model, represented as a parametric Markov decision process, which is used to plan and analyze compositions of subsystems, and of the collection of low-level subsystems themselves. The subsystems are implemented as deep RL agents operating under partial observability. By defining interfaces between the subsystems, the framework enables automatic decompositions of task specifications, e.g., reach a target set of states with a probability of at least 0.95, into individual subtask specifications, i.e. achieve the subsystem's exit conditions with at least some minimum probability, given that its entry conditions are met. This in turn allows for the independent training and testing of the subsystems. We present theoretical results guaranteeing that if each subsystem learns a policy satisfying its subtask specification, then their composition is guaranteed to satisfy the overall task specification. Conversely, if the subtask specifications cannot all be satisfied by the learned policies, we present a method, formulated as the problem of finding an optimal set of parameters in the high-level model, to automatically update the subtask specifications to account for the observed shortcomings. The result is an iterative procedure for defining subtask specifications, and for training the subsystems to meet them. Experimental results demonstrate the presented framework's novel capabilities in environments with both full and partial observability, discrete and continuous state and action spaces, as well as deterministic and stochastic dynamics.

{{</citation>}}


### (40/52) Integrated Robotics Networks with Co-optimization of Drone Placement and Air-Ground Communications (Menghao Hu et al., 2023)

{{<citation>}}

Menghao Hu, Tong Zhang, Shuai Wang, Guoliang Li, Yingyang Chen, Qiang Li, Gaojie Chen. (2023)  
**Integrated Robotics Networks with Co-optimization of Drone Placement and Air-Ground Communications**  

---
Primary Category: eess.SY  
Categories: cs-DC, cs-SY, eess-SY, eess.SY  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2309.04730v1)  

---


**ABSTRACT**  
Terrestrial robots, i.e., unmanned ground vehicles (UGVs), and aerial robots, i.e., unmanned aerial vehicles (UAVs), operate in separate spaces. To exploit their complementary features (e.g., fields of views, communication links, computing capabilities), a promising paradigm termed integrated robotics network emerges, which provides communications for cooperative UAVs-UGVs applications. However, how to efficiently deploy UAVs and schedule the UAVs-UGVs connections according to different UGV tasks become challenging. In this paper, we propose a sum-rate maximization problem, where UGVs plan their trajectories autonomously and are dynamically associated with UAVs according to their planned trajectories. Although the problem is a NP-hard mixed integer program, a fast polynomial time algorithm using alternating gradient descent and penalty-based binary relaxation, is devised. Simulation results demonstrate the effectiveness of the proposed algorithm.

{{</citation>}}


## quant-ph (1)



### (41/52) Fast Simulation of High-Depth QAOA Circuits (Danylo Lykov et al., 2023)

{{<citation>}}

Danylo Lykov, Ruslan Shaydulin, Yue Sun, Yuri Alexeev, Marco Pistoia. (2023)  
**Fast Simulation of High-Depth QAOA Circuits**  

---
Primary Category: quant-ph  
Categories: cs-DC, cs-PF, quant-ph, quant-ph  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.04841v2)  

---


**ABSTRACT**  
Until high-fidelity quantum computers with a large number of qubits become widely available, classical simulation remains a vital tool for algorithm design, tuning, and validation. We present a simulator for the Quantum Approximate Optimization Algorithm (QAOA). Our simulator is designed with the goal of reducing the computational cost of QAOA parameter optimization and supports both CPU and GPU execution. Our central observation is that the computational cost of both simulating the QAOA state and computing the QAOA objective to be optimized can be reduced by precomputing the diagonal Hamiltonian encoding the problem. We reduce the time for a typical QAOA parameter optimization by eleven times for $n = 26$ qubits compared to a state-of-the-art GPU quantum circuit simulator based on cuQuantum. Our simulator is available on GitHub: https://github.com/jpmorganchase/QOKit

{{</citation>}}


## cs.CR (2)



### (42/52) The Effectiveness of Security Interventions on GitHub (Felix Fischer et al., 2023)

{{<citation>}}

Felix Fischer, Jonas Höbenreich, Jens Grossklags. (2023)  
**The Effectiveness of Security Interventions on GitHub**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.04833v2)  

---


**ABSTRACT**  
In 2017, GitHub was the first online open source platform to show security alerts to its users. It has since introduced further security interventions to help developers improve the security of their open source software. In this study, we investigate and compare the effects of these interventions. This offers a valuable empirical perspective on security interventions in the context of software development, enriching the predominantly qualitative and survey-based literature landscape with substantial data-driven insights. We conduct a time series analysis on security-altering commits covering the entire history of a large-scale sample of over 50,000 GitHub repositories to infer the causal effects of the security alert, security update, and code scanning interventions. Our analysis shows that while all of GitHub's security interventions have a significant positive effect on security, they differ greatly in their effect size. By comparing the design of each intervention, we identify the building blocks that worked well and those that did not. We also provide recommendations on how practitioners can improve the design of their interventions to enhance their effectiveness.

{{</citation>}}


### (43/52) Security Analysis of Pairing-based Cryptography (Xiaofeng Wang et al., 2023)

{{<citation>}}

Xiaofeng Wang, Peng Zheng, Qianqian Xing. (2023)  
**Security Analysis of Pairing-based Cryptography**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.04693v1)  

---


**ABSTRACT**  
Recent progress in number field sieve (NFS) has shaken the security of Pairing-based Cryptography. For the discrete logarithm problem (DLP) in finite field, we present the first systematic review of the NFS algorithms from three perspectives: the degree $\alpha$, constant $c$, and hidden constant $o(1)$ in the asymptotic complexity $L_Q\left(\alpha,c\right)$ and indicate that further research is required to optimize the hidden constant. Using the special extended tower NFS algorithm, we conduct a thorough security evaluation for all the existing standardized PF curves as well as several commonly utilized curves, which reveals that the BN256 curves recommended by the SM9 and the previous ISO/IEC standard exhibit only 99.92 bits of security, significantly lower than the intended 128-bit level. In addition, we comprehensively analyze the security and efficiency of BN, BLS, and KSS curves for different security levels. Our analysis suggests that the BN curve exhibits superior efficiency for security strength below approximately 105 bit. For a 128-bit security level, BLS12 and BLS24 curves are the optimal choices, while the BLS24 curve offers the best efficiency for security levels of 160bit, 192bit, and 256bit.

{{</citation>}}


## cs.SE (1)



### (44/52) FAIR: Flow Type-Aware Pre-Training of Compiler Intermediate Representations (Changan Niu et al., 2023)

{{<citation>}}

Changan Niu, Chuanyi Li, Vincent Ng, David Lo, Bin Luo. (2023)  
**FAIR: Flow Type-Aware Pre-Training of Compiler Intermediate Representations**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI, Transformer  
[Paper Link](http://arxiv.org/abs/2309.04828v1)  

---


**ABSTRACT**  
While the majority of existing pre-trained models from code learn source code features such as code tokens and abstract syntax trees, there are some other works that focus on learning from compiler intermediate representations (IRs). Existing IR-based models typically utilize IR features such as instructions, control and data flow graphs (CDFGs), call graphs, etc. However, these methods confuse variable nodes and instruction nodes in a CDFG and fail to distinguish different types of flows, and the neural networks they use fail to capture long-distance dependencies and have over-smoothing and over-squashing problems. To address these weaknesses, we propose FAIR, a Flow type-Aware pre-trained model for IR that involves employing (1) a novel input representation of IR programs; (2) Graph Transformer to address over-smoothing, over-squashing and long-dependencies problems; and (3) five pre-training tasks that we specifically propose to enable FAIR to learn the semantics of IR tokens, flow type information, and the overall representation of IR. Experimental results show that FAIR can achieve state-of-the-art results on four code-related downstream tasks.

{{</citation>}}


## cs.SI (2)



### (45/52) Finding Influencers in Complex Networks: An Effective Deep Reinforcement Learning Approach (Changan Liu et al., 2023)

{{<citation>}}

Changan Liu, Changjun Fan, Zhongzhi Zhang. (2023)  
**Finding Influencers in Complex Networks: An Effective Deep Reinforcement Learning Approach**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-LG, cs-SI, cs.SI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.07153v1)  

---


**ABSTRACT**  
Maximizing influences in complex networks is a practically important but computationally challenging task for social network analysis, due to its NP- hard nature. Most current approximation or heuristic methods either require tremendous human design efforts or achieve unsatisfying balances between effectiveness and efficiency. Recent machine learning attempts only focus on speed but lack performance enhancement. In this paper, different from previous attempts, we propose an effective deep reinforcement learning model that achieves superior performances over traditional best influence maximization algorithms. Specifically, we design an end-to-end learning framework that combines graph neural network as the encoder and reinforcement learning as the decoder, named DREIM. Trough extensive training on small synthetic graphs, DREIM outperforms the state-of-the-art baseline methods on very large synthetic and real-world networks on solution quality, and we also empirically show its linear scalability with regard to the network size, which demonstrates its superiority in solving this problem.

{{</citation>}}


### (46/52) Influence Maximization in Social Networks: A Survey (Hui Li et al., 2023)

{{<citation>}}

Hui Li, Susu Yang, Mengting Xu, Sourav S Bhowmick, Jiangtao Cui. (2023)  
**Influence Maximization in Social Networks: A Survey**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Social Network  
[Paper Link](http://arxiv.org/abs/2309.04668v1)  

---


**ABSTRACT**  
Online social networks have become an important platform for people to communicate, share knowledge and disseminate information. Given the widespread usage of social media, individuals' ideas, preferences and behavior are often influenced by their peers or friends in the social networks that they participate in. Since the last decade, influence maximization (IM) problem has been extensively adopted to model the diffusion of innovations and ideas. The purpose of IM is to select a set of k seed nodes who can influence the most individuals in the network.   In this survey, we present a systematical study over the researches and future directions with respect to IM problem. We review the information diffusion models and analyze a variety of algorithms for the classic IM algorithms. We propose a taxonomy for potential readers to understand the key techniques and challenges. We also organize the milestone works in time order such that the readers of this survey can experience the research roadmap in this field. Moreover, we also categorize other application-oriented IM studies and correspondingly study each of them. What's more, we list a series of open questions as the future directions for IM-related researches, where a potential reader of this survey can easily observe what should be done next in this field.

{{</citation>}}


## cs.SD (2)



### (47/52) AudRandAug: Random Image Augmentations for Audio Classification (Teerath Kumar et al., 2023)

{{<citation>}}

Teerath Kumar, Muhammad Turab, Alessandra Mileo, Malika Bendechache, Takfarinas Saber. (2023)  
**AudRandAug: Random Image Augmentations for Audio Classification**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-CV, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2309.04762v1)  

---


**ABSTRACT**  
Data augmentation has proven to be effective in training neural networks. Recently, a method called RandAug was proposed, randomly selecting data augmentation techniques from a predefined search space. RandAug has demonstrated significant performance improvements for image-related tasks while imposing minimal computational overhead. However, no prior research has explored the application of RandAug specifically for audio data augmentation, which converts audio into an image-like pattern. To address this gap, we introduce AudRandAug, an adaptation of RandAug for audio data. AudRandAug selects data augmentation policies from a dedicated audio search space. To evaluate the effectiveness of AudRandAug, we conducted experiments using various models and datasets. Our findings indicate that AudRandAug outperforms other existing data augmentation methods regarding accuracy performance.

{{</citation>}}


### (48/52) Mask-CTC-based Encoder Pre-training for Streaming End-to-End Speech Recognition (Huaibo Zhao et al., 2023)

{{<citation>}}

Huaibo Zhao, Yosuke Higuchi, Yusuke Kida, Tetsuji Ogawa, Tetsunori Kobayashi. (2023)  
**Mask-CTC-based Encoder Pre-training for Streaming End-to-End Speech Recognition**  

---
Primary Category: cs.SD  
Categories: cs-SD, cs.SD, eess-AS  
Keywords: Speech Recognition, Transformer  
[Paper Link](http://arxiv.org/abs/2309.04654v1)  

---


**ABSTRACT**  
Achieving high accuracy with low latency has always been a challenge in streaming end-to-end automatic speech recognition (ASR) systems. By attending to more future contexts, a streaming ASR model achieves higher accuracy but results in larger latency, which hurts the streaming performance. In the Mask-CTC framework, an encoder network is trained to learn the feature representation that anticipates long-term contexts, which is desirable for streaming ASR. Mask-CTC-based encoder pre-training has been shown beneficial in achieving low latency and high accuracy for triggered attention-based ASR. However, the effectiveness of this method has not been demonstrated for various model architectures, nor has it been verified that the encoder has the expected look-ahead capability to reduce latency. This study, therefore, examines the effectiveness of Mask-CTCbased pre-training for models with different architectures, such as Transformer-Transducer and contextual block streaming ASR. We also discuss the effect of the proposed pre-training method on obtaining accurate output spike timing.

{{</citation>}}


## cs.RO (1)



### (49/52) A Review on Robot Manipulation Methods in Human-Robot Interactions (Haoxu Zhang et al., 2023)

{{<citation>}}

Haoxu Zhang, Parham M. Kebria, Shady Mohamed, Samson Yu, Saeid Nahavandi. (2023)  
**A Review on Robot Manipulation Methods in Human-Robot Interactions**  

---
Primary Category: cs.RO  
Categories: cs-HC, cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.04687v1)  

---


**ABSTRACT**  
Robot manipulation is an important part of human-robot interaction technology. However, traditional pre-programmed methods can only accomplish simple and repetitive tasks. To enable effective communication between robots and humans, and to predict and adapt to uncertain environments, this paper reviews recent autonomous and adaptive learning in robotic manipulation algorithms. It includes typical applications and challenges of human-robot interaction, fundamental tasks of robot manipulation and one of the most widely used formulations of robot manipulation, Markov Decision Process. Recent research focusing on robot manipulation is mainly based on Reinforcement Learning and Imitation Learning. This review paper shows the importance of Deep Reinforcement Learning, which plays an important role in manipulating robots to complete complex tasks in disturbed and unfamiliar environments. With the introduction of Imitation Learning, it is possible for robot manipulation to get rid of reward function design and achieve a simple, stable and supervised learning process. This paper reviews and compares the main features and popular algorithms for both Reinforcement Learning and Imitation Learning.

{{</citation>}}


## eess.IV (3)



### (50/52) SSHNN: Semi-Supervised Hybrid NAS Network for Echocardiographic Image Segmentation (Renqi Chen et al., 2023)

{{<citation>}}

Renqi Chen, Jingjing Luo, Fan Nian, Yuhui Cen, Yiheng Peng, Zekuan Yu. (2023)  
**SSHNN: Semi-Supervised Hybrid NAS Network for Echocardiographic Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Semi-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.04672v1)  

---


**ABSTRACT**  
Accurate medical image segmentation especially for echocardiographic images with unmissable noise requires elaborate network design. Compared with manual design, Neural Architecture Search (NAS) realizes better segmentation results due to larger search space and automatic optimization, but most of the existing methods are weak in layer-wise feature aggregation and adopt a ``strong encoder, weak decoder" structure, insufficient to handle global relationships and local details. To resolve these issues, we propose a novel semi-supervised hybrid NAS network for accurate medical image segmentation termed SSHNN. In SSHNN, we creatively use convolution operation in layer-wise feature fusion instead of normalized scalars to avoid losing details, making NAS a stronger encoder. Moreover, Transformers are introduced for the compensation of global context and U-shaped decoder is designed to efficiently connect global context with local features. Specifically, we implement a semi-supervised algorithm Mean-Teacher to overcome the limited volume problem of labeled medical image dataset. Extensive experiments on CAMUS echocardiography dataset demonstrate that SSHNN outperforms state-of-the-art approaches and realizes accurate segmentation. Code will be made publicly available.

{{</citation>}}


### (51/52) ConvFormer: Plug-and-Play CNN-Style Transformers for Improving Medical Image Segmentation (Xian Lin et al., 2023)

{{<citation>}}

Xian Lin, Zengqiang Yan, Xianbo Deng, Chuansheng Zheng, Li Yu. (2023)  
**ConvFormer: Plug-and-Play CNN-Style Transformers for Improving Medical Image Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.05674v1)  

---


**ABSTRACT**  
Transformers have been extensively studied in medical image segmentation to build pairwise long-range dependence. Yet, relatively limited well-annotated medical image data makes transformers struggle to extract diverse global features, resulting in attention collapse where attention maps become similar or even identical. Comparatively, convolutional neural networks (CNNs) have better convergence properties on small-scale training data but suffer from limited receptive fields. Existing works are dedicated to exploring the combinations of CNN and transformers while ignoring attention collapse, leaving the potential of transformers under-explored. In this paper, we propose to build CNN-style Transformers (ConvFormer) to promote better attention convergence and thus better segmentation performance. Specifically, ConvFormer consists of pooling, CNN-style self-attention (CSA), and convolutional feed-forward network (CFFN) corresponding to tokenization, self-attention, and feed-forward network in vanilla vision transformers. In contrast to positional embedding and tokenization, ConvFormer adopts 2D convolution and max-pooling for both position information preservation and feature size reduction. In this way, CSA takes 2D feature maps as inputs and establishes long-range dependency by constructing self-attention matrices as convolution kernels with adaptive sizes. Following CSA, 2D convolution is utilized for feature refinement through CFFN. Experimental results on multiple datasets demonstrate the effectiveness of ConvFormer working as a plug-and-play module for consistent performance improvement of transformer-based frameworks. Code is available at https://github.com/xianlin7/ConvFormer.

{{</citation>}}


### (52/52) Video and Synthetic MRI Pre-training of 3D Vision Architectures for Neuroimage Analysis (Nikhil J. Dhinagar et al., 2023)

{{<citation>}}

Nikhil J. Dhinagar, Amit Singh, Saket Ozarkar, Ketaki Buwa, Sophia I. Thomopoulos, Conor Owens-Walton, Emily Laltoo, Yao-Liang Chen, Philip Cook, Corey McMillan, Chih-Chien Tsai, J-J Wang, Yih-Ru Wu, Paul M. Thompson. (2023)  
**Video and Synthetic MRI Pre-training of 3D Vision Architectures for Neuroimage Analysis**  

---
Primary Category: eess.IV  
Categories: cs-AI, cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.04651v1)  

---


**ABSTRACT**  
Transfer learning represents a recent paradigm shift in the way we build artificial intelligence (AI) systems. In contrast to training task-specific models, transfer learning involves pre-training deep learning models on a large corpus of data and minimally fine-tuning them for adaptation to specific tasks. Even so, for 3D medical imaging tasks, we do not know if it is best to pre-train models on natural images, medical images, or even synthetically generated MRI scans or video data. To evaluate these alternatives, here we benchmarked vision transformers (ViTs) and convolutional neural networks (CNNs), initialized with varied upstream pre-training approaches. These methods were then adapted to three unique downstream neuroimaging tasks with a range of difficulty: Alzheimer's disease (AD) and Parkinson's disease (PD) classification, "brain age" prediction. Experimental tests led to the following key observations: 1. Pre-training improved performance across all tasks including a boost of 7.4% for AD classification and 4.6% for PD classification for the ViT and 19.1% for PD classification and reduction in brain age prediction error by 1.26 years for CNNs, 2. Pre-training on large-scale video or synthetic MRI data boosted performance of ViTs, 3. CNNs were robust in limited-data settings, and in-domain pretraining enhanced their performances, 4. Pre-training improved generalization to out-of-distribution datasets and sites. Overall, we benchmarked different vision architectures, revealing the value of pre-training them with emerging datasets for model initialization. The resulting pre-trained models can be adapted to a range of downstream neuroimaging tasks, even when training data for the target task is limited.

{{</citation>}}
