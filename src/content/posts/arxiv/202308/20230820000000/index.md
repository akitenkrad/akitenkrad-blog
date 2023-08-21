---
draft: false
title: "arXiv @ 2023.08.20"
date: 2023-08-20
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.08.20"
    identifier: arxiv_20230820
    parent: 202308_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (38)](#cscv-38)
- [cs.CL (15)](#cscl-15)
- [cs.SI (2)](#cssi-2)
- [cs.LG (13)](#cslg-13)
- [cs.CR (6)](#cscr-6)
- [cs.AI (4)](#csai-4)
- [cs.RO (3)](#csro-3)
- [cs.SD (1)](#cssd-1)
- [cs.CE (1)](#csce-1)
- [cs.IR (2)](#csir-2)
- [cs.SE (2)](#csse-2)

## cs.CV (38)



### (1/87) Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training (Xiaoyang Wu et al., 2023)

{{<citation>}}

Xiaoyang Wu, Zhuotao Tian, Xin Wen, Bohao Peng, Xihui Liu, Kaicheng Yu, Hengshuang Zhao. (2023)  
**Towards Large-scale 3D Representation Learning with Multi-dataset Point Prompt Training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.09718v1)  

---


**ABSTRACT**  
The rapid advancement of deep learning models often attributes to their ability to leverage massive training data. In contrast, such privilege has not yet fully benefited 3D deep learning, mainly due to the limited availability of large-scale 3D datasets. Merging multiple available data sources and letting them collaboratively train a single model is a potential solution. However, due to the large domain gap between 3D point cloud datasets, such mixed supervision could adversely affect the model's performance and lead to degenerated performance (i.e., negative transfer) compared to single-dataset training. In view of this challenge, we introduce Point Prompt Training (PPT), a novel framework for multi-dataset synergistic learning in the context of 3D representation learning that supports multiple pre-training paradigms. Based on this framework, we propose Prompt-driven Normalization, which adapts the model to different datasets with domain-specific prompts and Language-guided Categorical Alignment that decently unifies the multiple-dataset label spaces by leveraging the relationship between label text. Extensive experiments verify that PPT can overcome the negative transfer associated with synergistic learning and produce generalizable representations. Notably, it achieves state-of-the-art performance on each dataset using a single weight-shared model with supervised multi-dataset training. Moreover, when served as a pre-training framework, it outperforms other pre-training approaches regarding representation quality and attains remarkable state-of-the-art performance across over ten diverse downstream tasks spanning both indoor and outdoor 3D scenarios.

{{</citation>}}


### (2/87) Smoothness Similarity Regularization for Few-Shot GAN Adaptation (Vadim Sushko et al., 2023)

{{<citation>}}

Vadim Sushko, Ruyu Wang, Juergen Gall. (2023)  
**Smoothness Similarity Regularization for Few-Shot GAN Adaptation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.09717v1)  

---


**ABSTRACT**  
The task of few-shot GAN adaptation aims to adapt a pre-trained GAN model to a small dataset with very few training images. While existing methods perform well when the dataset for pre-training is structurally similar to the target dataset, the approaches suffer from training instabilities or memorization issues when the objects in the two domains have a very different structure. To mitigate this limitation, we propose a new smoothness similarity regularization that transfers the inherently learned smoothness of the pre-trained GAN to the few-shot target domain even if the two domains are very different. We evaluate our approach by adapting an unconditional and a class-conditional GAN to diverse few-shot target domains. Our proposed method significantly outperforms prior few-shot GAN adaptation methods in the challenging case of structurally dissimilar source-target domains, while performing on par with the state of the art for similar source-target domains.

{{</citation>}}


### (3/87) SimDA: Simple Diffusion Adapter for Efficient Video Generation (Zhen Xing et al., 2023)

{{<citation>}}

Zhen Xing, Qi Dai, Han Hu, Zuxuan Wu, Yu-Gang Jiang. (2023)  
**SimDA: Simple Diffusion Adapter for Efficient Video Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2308.09710v1)  

---


**ABSTRACT**  
The recent wave of AI-generated content has witnessed the great development and success of Text-to-Image (T2I) technologies. By contrast, Text-to-Video (T2V) still falls short of expectations though attracting increasing interests. Existing works either train from scratch or adapt large T2I model to videos, both of which are computation and resource expensive. In this work, we propose a Simple Diffusion Adapter (SimDA) that fine-tunes only 24M out of 1.1B parameters of a strong T2I model, adapting it to video generation in a parameter-efficient way. In particular, we turn the T2I model for T2V by designing light-weight spatial and temporal adapters for transfer learning. Besides, we change the original spatial attention to the proposed Latent-Shift Attention (LSA) for temporal consistency. With similar model architecture, we further train a video super-resolution model to generate high-definition (1024x1024) videos. In addition to T2V generation in the wild, SimDA could also be utilized in one-shot video editing with only 2 minutes tuning. Doing so, our method could minimize the training effort with extremely few tunable parameters for model adaptation.

{{</citation>}}


### (4/87) Invariant Training 2D-3D Joint Hard Samples for Few-Shot Point Cloud Recognition (Xuanyu Yi et al., 2023)

{{<citation>}}

Xuanyu Yi, Jiajun Deng, Qianru Sun, Xian-Sheng Hua, Joo-Hwee Lim, Hanwang Zhang. (2023)  
**Invariant Training 2D-3D Joint Hard Samples for Few-Shot Point Cloud Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.09694v1)  

---


**ABSTRACT**  
We tackle the data scarcity challenge in few-shot point cloud recognition of 3D objects by using a joint prediction from a conventional 3D model and a well-trained 2D model. Surprisingly, such an ensemble, though seems trivial, has hardly been shown effective in recent 2D-3D models. We find out the crux is the less effective training for the ''joint hard samples'', which have high confidence prediction on different wrong labels, implying that the 2D and 3D models do not collaborate well. To this end, our proposed invariant training strategy, called InvJoint, does not only emphasize the training more on the hard samples, but also seeks the invariance between the conflicting 2D and 3D ambiguous predictions. InvJoint can learn more collaborative 2D and 3D representations for better ensemble. Extensive experiments on 3D shape classification with widely adopted ModelNet10/40, ScanObjectNN and Toys4K, and shape retrieval with ShapeNet-Core validate the superiority of our InvJoint.

{{</citation>}}


### (5/87) A Lightweight Transformer for Faster and Robust EBSD Data Collection (Harry Dong et al., 2023)

{{<citation>}}

Harry Dong, Sean Donegan, Megna Shah, Yuejie Chi. (2023)  
**A Lightweight Transformer for Faster and Robust EBSD Data Collection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.09693v1)  

---


**ABSTRACT**  
Three dimensional electron back-scattered diffraction (EBSD) microscopy is a critical tool in many applications in materials science, yet its data quality can fluctuate greatly during the arduous collection process, particularly via serial-sectioning. Fortunately, 3D EBSD data is inherently sequential, opening up the opportunity to use transformers, state-of-the-art deep learning architectures that have made breakthroughs in a plethora of domains, for data processing and recovery. To be more robust to errors and accelerate this 3D EBSD data collection, we introduce a two step method that recovers missing slices in an 3D EBSD volume, using an efficient transformer model and a projection algorithm to process the transformer's outputs. Overcoming the computational and practical hurdles of deep learning with scarce high dimensional data, we train this model using only synthetic 3D EBSD data with self-supervision and obtain superior recovery accuracy on real 3D EBSD data, compared to existing methods.

{{</citation>}}


### (6/87) Is context all you need? Scaling Neural Sign Language Translation to Large Domains of Discourse (Ozge Mercanoglu Sincan et al., 2023)

{{<citation>}}

Ozge Mercanoglu Sincan, Necati Cihan Camgoz, Richard Bowden. (2023)  
**Is context all you need? Scaling Neural Sign Language Translation to Large Domains of Discourse**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: BLEU, Machine Translation  
[Paper Link](http://arxiv.org/abs/2308.09622v1)  

---


**ABSTRACT**  
Sign Language Translation (SLT) is a challenging task that aims to generate spoken language sentences from sign language videos, both of which have different grammar and word/gloss order. From a Neural Machine Translation (NMT) perspective, the straightforward way of training translation models is to use sign language phrase-spoken language sentence pairs. However, human interpreters heavily rely on the context to understand the conveyed information, especially for sign language interpretation, where the vocabulary size may be significantly smaller than their spoken language equivalent.   Taking direct inspiration from how humans translate, we propose a novel multi-modal transformer architecture that tackles the translation task in a context-aware manner, as a human would. We use the context from previous sequences and confident predictions to disambiguate weaker visual cues. To achieve this we use complementary transformer encoders, namely: (1) A Video Encoder, that captures the low-level video features at the frame-level, (2) A Spotting Encoder, that models the recognized sign glosses in the video, and (3) A Context Encoder, which captures the context of the preceding sign sequences. We combine the information coming from these encoders in a final transformer decoder to generate spoken language translations.   We evaluate our approach on the recently published large-scale BOBSL dataset, which contains ~1.2M sequences, and on the SRF dataset, which was part of the WMT-SLT 2022 challenge. We report significant improvements on state-of-the-art translation performance using contextual information, nearly doubling the reported BLEU-4 scores of baseline approaches.

{{</citation>}}


### (7/87) Far3D: Expanding the Horizon for Surround-view 3D Object Detection (Xiaohui Jiang et al., 2023)

{{<citation>}}

Xiaohui Jiang, Shuailin Li, Yingfei Liu, Shihao Wang, Fan Jia, Tiancai Wang, Lijin Han, Xiangyu Zhang. (2023)  
**Far3D: Expanding the Horizon for Surround-view 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.09616v1)  

---


**ABSTRACT**  
Recently 3D object detection from surround-view images has made notable advancements with its low deployment cost. However, most works have primarily focused on close perception range while leaving long-range detection less explored. Expanding existing methods directly to cover long distances poses challenges such as heavy computation costs and unstable convergence. To address these limitations, this paper proposes a novel sparse query-based framework, dubbed Far3D. By utilizing high-quality 2D object priors, we generate 3D adaptive queries that complement the 3D global queries. To efficiently capture discriminative features across different views and scales for long-range objects, we introduce a perspective-aware aggregation module. Additionally, we propose a range-modulated 3D denoising approach to address query error propagation and mitigate convergence issues in long-range tasks. Significantly, Far3D demonstrates SoTA performance on the challenging Argoverse 2 dataset, covering a wide range of 150 meters, surpassing several LiDAR-based approaches. Meanwhile, Far3D exhibits superior performance compared to previous methods on the nuScenes dataset. The code will be available soon.

{{</citation>}}


### (8/87) On the Effectiveness of LayerNorm Tuning for Continual Learning in Vision Transformers (Thomas De Min et al., 2023)

{{<citation>}}

Thomas De Min, Massimiliano Mancini, Karteek Alahari, Xavier Alameda-Pineda, Elisa Ricci. (2023)  
**On the Effectiveness of LayerNorm Tuning for Continual Learning in Vision Transformers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.09610v1)  

---


**ABSTRACT**  
State-of-the-art rehearsal-free continual learning methods exploit the peculiarities of Vision Transformers to learn task-specific prompts, drastically reducing catastrophic forgetting. However, there is a tradeoff between the number of learned parameters and the performance, making such models computationally expensive. In this work, we aim to reduce this cost while maintaining competitive performance. We achieve this by revisiting and extending a simple transfer learning idea: learning task-specific normalization layers. Specifically, we tune the scale and bias parameters of LayerNorm for each continual learning task, selecting them at inference time based on the similarity between task-specific keys and the output of the pre-trained model. To make the classifier robust to incorrect selection of parameters during inference, we introduce a two-stage training procedure, where we first optimize the task-specific parameters and then train the classifier with the same selection procedure of the inference time. Experiments on ImageNet-R and CIFAR-100 show that our method achieves results that are either superior or on par with {the state of the art} while being computationally cheaper.

{{</citation>}}


### (9/87) PUMGPT: A Large Vision-Language Model for Product Understanding (Shuhui Wu et al., 2023)

{{<citation>}}

Shuhui Wu, Zengming Tang, Zongyi Guo, Weiwei Zhang, Baoliang Cui, Haihong Tang, Weiming Lu. (2023)  
**PUMGPT: A Large Vision-Language Model for Product Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.09568v1)  

---


**ABSTRACT**  
Recent developments of multi-modal large language models have demonstrated its strong ability in solving vision-language tasks. In this paper, we focus on the product understanding task, which plays an essential role in enhancing online shopping experience. Product understanding task includes a variety of sub-tasks, which require models to respond diverse queries based on multi-modal product information. Traditional methods design distinct model architectures for each sub-task. On the contrary, we present PUMGPT, a large vision-language model aims at unifying all product understanding tasks under a singular model structure. To bridge the gap between vision and text representations, we propose Layer-wise Adapters (LA), an approach that provides enhanced alignment with fewer visual tokens and enables parameter-efficient fine-tuning. Moreover, the inherent parameter-efficient fine-tuning ability allows PUMGPT to be readily adapted to new product understanding tasks and emerging products. We design instruction templates to generate diverse product instruction datasets. Simultaneously, we utilize open-domain datasets during training to improve the performance of PUMGPT and its generalization ability. Through extensive evaluations, PUMGPT demonstrates its superior performance across multiple product understanding tasks, including product captioning, category question-answering, attribute extraction, attribute question-answering, and even free-form question-answering about products.

{{</citation>}}


### (10/87) Deep Equilibrium Object Detection (Shuai Wang et al., 2023)

{{<citation>}}

Shuai Wang, Yao Teng, Limin Wang. (2023)  
**Deep Equilibrium Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.09564v1)  

---


**ABSTRACT**  
Query-based object detectors directly decode image features into object instances with a set of learnable queries. These query vectors are progressively refined to stable meaningful representations through a sequence of decoder layers, and then used to directly predict object locations and categories with simple FFN heads. In this paper, we present a new query-based object detector (DEQDet) by designing a deep equilibrium decoder. Our DEQ decoder models the query vector refinement as the fixed point solving of an {implicit} layer and is equivalent to applying {infinite} steps of refinement. To be more specific to object decoding, we use a two-step unrolled equilibrium equation to explicitly capture the query vector refinement. Accordingly, we are able to incorporate refinement awareness into the DEQ training with the inexact gradient back-propagation (RAG). In addition, to stabilize the training of our DEQDet and improve its generalization ability, we devise the deep supervision scheme on the optimization path of DEQ with refinement-aware perturbation~(RAP). Our experiments demonstrate DEQDet converges faster, consumes less memory, and achieves better results than the baseline counterpart (AdaMixer). In particular, our DEQDet with ResNet50 backbone and 300 queries achieves the $49.5$ mAP and $33.0$ AP$_s$ on the MS COCO benchmark under $2\times$ training scheme (24 epochs).

{{</citation>}}


### (11/87) Decoupled conditional contrastive learning with variable metadata for prostate lesion detection (Camille Ruppli et al., 2023)

{{<citation>}}

Camille Ruppli, Pietro Gori, Roberto Ardon, Isabelle Bloch. (2023)  
**Decoupled conditional contrastive learning with variable metadata for prostate lesion detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09542v1)  

---


**ABSTRACT**  
Early diagnosis of prostate cancer is crucial for efficient treatment. Multi-parametric Magnetic Resonance Images (mp-MRI) are widely used for lesion detection. The Prostate Imaging Reporting and Data System (PI-RADS) has standardized interpretation of prostate MRI by defining a score for lesion malignancy. PI-RADS data is readily available from radiology reports but is subject to high inter-reports variability. We propose a new contrastive loss function that leverages weak metadata with multiple annotators per sample and takes advantage of inter-reports variability by defining metadata confidence. By combining metadata of varying confidence with unannotated data into a single conditional contrastive loss function, we report a 3% AUC increase on lesion detection on the public PI-CAI challenge dataset.   Code is available at: https://github.com/camilleruppli/decoupled_ccl

{{</citation>}}


### (12/87) Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning (Xiang Yuan et al., 2023)

{{<citation>}}

Xiang Yuan, Gong Cheng, Kebing Yan, Qinghua Zeng, Junwei Han. (2023)  
**Small Object Detection via Coarse-to-fine Proposal Generation and Imitation Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.09534v1)  

---


**ABSTRACT**  
The past few years have witnessed the immense success of object detection, while current excellent detectors struggle on tackling size-limited instances. Concretely, the well-known challenge of low overlaps between the priors and object regions leads to a constrained sample pool for optimization, and the paucity of discriminative information further aggravates the recognition. To alleviate the aforementioned issues, we propose CFINet, a two-stage framework tailored for small object detection based on the Coarse-to-fine pipeline and Feature Imitation learning. Firstly, we introduce Coarse-to-fine RPN (CRPN) to ensure sufficient and high-quality proposals for small objects through the dynamic anchor selection strategy and cascade regression. Then, we equip the conventional detection head with a Feature Imitation (FI) branch to facilitate the region representations of size-limited instances that perplex the model in an imitation manner. Moreover, an auxiliary imitation loss following supervised contrastive learning paradigm is devised to optimize this branch. When integrated with Faster RCNN, CFINet achieves state-of-the-art performance on the large-scale small object detection benchmarks, SODA-D and SODA-A, underscoring its superiority over baseline detector and other mainstream detection approaches.

{{</citation>}}


### (13/87) Denoising Diffusion for 3D Hand Pose Estimation from Images (Maksym Ivashechkin et al., 2023)

{{<citation>}}

Maksym Ivashechkin, Oscar Mendez, Richard Bowden. (2023)  
**Denoising Diffusion for 3D Hand Pose Estimation from Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.09523v1)  

---


**ABSTRACT**  
Hand pose estimation from a single image has many applications. However, approaches to full 3D body pose estimation are typically trained on day-to-day activities or actions. As such, detailed hand-to-hand interactions are poorly represented, especially during motion. We see this in the failure cases of techniques such as OpenPose or MediaPipe. However, accurate hand pose estimation is crucial for many applications where the global body motion is less important than accurate hand pose estimation.   This paper addresses the problem of 3D hand pose estimation from monocular images or sequences. We present a novel end-to-end framework for 3D hand regression that employs diffusion models that have shown excellent ability to capture the distribution of data for generative purposes. Moreover, we enforce kinematic constraints to ensure realistic poses are generated by incorporating an explicit forward kinematic layer as part of the network. The proposed model provides state-of-the-art performance when lifting a 2D single-hand image to 3D. However, when sequence data is available, we add a Transformer module over a temporal window of consecutive frames to refine the results, overcoming jittering and further increasing accuracy.   The method is quantitatively and qualitatively evaluated showing state-of-the-art robustness, generalization, and accuracy on several different datasets.

{{</citation>}}


### (14/87) Learnt Contrastive Concept Embeddings for Sign Recognition (Ryan Wong et al., 2023)

{{<citation>}}

Ryan Wong, Necati Cihan Camgoz, Richard Bowden. (2023)  
**Learnt Contrastive Concept Embeddings for Sign Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding, NLP  
[Paper Link](http://arxiv.org/abs/2308.09515v1)  

---


**ABSTRACT**  
In natural language processing (NLP) of spoken languages, word embeddings have been shown to be a useful method to encode the meaning of words. Sign languages are visual languages, which require sign embeddings to capture the visual and linguistic semantics of sign. Unlike many common approaches to Sign Recognition, we focus on explicitly creating sign embeddings that bridge the gap between sign language and spoken language. We propose a learning framework to derive LCC (Learnt Contrastive Concept) embeddings for sign language, a weakly supervised contrastive approach to learning sign embeddings. We train a vocabulary of embeddings that are based on the linguistic labels for sign video. Additionally, we develop a conceptual similarity loss which is able to utilise word embeddings from NLP methods to create sign embeddings that have better sign language to spoken language correspondence. These learnt representations allow the model to automatically localise the sign in time. Our approach achieves state-of-the-art keypoint-based sign recognition performance on the WLASL and BOBSL datasets.

{{</citation>}}


### (15/87) ResQ: Residual Quantization for Video Perception (Davide Abati et al., 2023)

{{<citation>}}

Davide Abati, Haitam Ben Yahia, Markus Nagel, Amirhossein Habibian. (2023)  
**ResQ: Residual Quantization for Video Perception**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Quantization  
[Paper Link](http://arxiv.org/abs/2308.09511v1)  

---


**ABSTRACT**  
This paper accelerates video perception, such as semantic segmentation and human pose estimation, by levering cross-frame redundancies. Unlike the existing approaches, which avoid redundant computations by warping the past features using optical-flow or by performing sparse convolutions on frame differences, we approach the problem from a new perspective: low-bit quantization. We observe that residuals, as the difference in network activations between two neighboring frames, exhibit properties that make them highly quantizable. Based on this observation, we propose a novel quantization scheme for video networks coined as Residual Quantization. ResQ extends the standard, frame-by-frame, quantization scheme by incorporating temporal dependencies that lead to better performance in terms of accuracy vs. bit-width. Furthermore, we extend our model to dynamically adjust the bit-width proportional to the amount of changes in the video. We demonstrate the superiority of our model, against the standard quantization and existing efficient video perception models, using various architectures on semantic segmentation and human pose estimation benchmarks.

{{</citation>}}


### (16/87) Vision Relation Transformer for Unbiased Scene Graph Generation (Gopika Sudhakaran et al., 2023)

{{<citation>}}

Gopika Sudhakaran, Devendra Singh Dhami, Kristian Kersting, Stefan Roth. (2023)  
**Vision Relation Transformer for Unbiased Scene Graph Generation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09472v1)  

---


**ABSTRACT**  
Recent years have seen a growing interest in Scene Graph Generation (SGG), a comprehensive visual scene understanding task that aims to predict entity relationships using a relation encoder-decoder pipeline stacked on top of an object encoder-decoder backbone. Unfortunately, current SGG methods suffer from an information loss regarding the entities local-level cues during the relation encoding process. To mitigate this, we introduce the Vision rElation TransfOrmer (VETO), consisting of a novel local-level entity relation encoder. We further observe that many existing SGG methods claim to be unbiased, but are still biased towards either head or tail classes. To overcome this bias, we introduce a Mutually Exclusive ExperT (MEET) learning strategy that captures important relation features without bias towards head or tail classes. Experimental results on the VG and GQA datasets demonstrate that VETO + MEET boosts the predictive performance by up to 47 percentage over the state of the art while being 10 times smaller.

{{</citation>}}


### (17/87) Artificial-Spiking Hierarchical Networks for Vision-Language Representation Learning (Yeming Chen et al., 2023)

{{<citation>}}

Yeming Chen, Siyu Zhang, Yaoru Sun, Weijian Liang, Haoran Wang. (2023)  
**Artificial-Spiking Hierarchical Networks for Vision-Language Representation Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2308.09455v1)  

---


**ABSTRACT**  
With the success of self-supervised learning, multimodal foundation models have rapidly adapted a wide range of downstream tasks driven by vision and language (VL) pretraining. State-of-the-art methods achieve impressive performance by pre-training on large-scale datasets. However, bridging the semantic gap between the two modalities remains a nonnegligible challenge for VL tasks. In this work, we propose an efficient computation framework for multimodal alignment by introducing a novel visual semantic module to further improve the performance of the VL tasks. Specifically, we propose a flexible model, namely Artificial-Spiking Hierarchical Networks (ASH-Nets), which combines the complementary advantages of Artificial neural networks (ANNs) and Spiking neural networks (SNNs) to enrich visual semantic representations. In particular, a visual concrete encoder and a semantic abstract encoder are constructed to learn continuous and discrete latent variables to enhance the flexibility of semantic encoding. Considering the spatio-temporal properties of SNNs modeling, we introduce a contrastive learning method to optimize the inputs of similar samples. This can improve the computational efficiency of the hierarchical network, while the augmentation of hard samples is beneficial to the learning of visual representations. Furthermore, the Spiking to Text Uni-Alignment Learning (STUA) pre-training method is proposed, which only relies on text features to enhance the encoding ability of abstract semantics. We validate the performance on multiple well-established downstream VL tasks. Experiments show that the proposed ASH-Nets achieve competitive results.

{{</citation>}}


### (18/87) Transformer-based Detection of Microorganismson High-Resolution Petri Dish Images (Nikolas Ebert et al., 2023)

{{<citation>}}

Nikolas Ebert, Didier Stricker, Oliver Wasenmüller. (2023)  
**Transformer-based Detection of Microorganismson High-Resolution Petri Dish Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.09436v1)  

---


**ABSTRACT**  
Many medical or pharmaceutical processes have strict guidelines regarding continuous hygiene monitoring. This often involves the labor-intensive task of manually counting microorganisms in Petri dishes by trained personnel. Automation attempts often struggle due to major challenges: significant scaling differences, low separation, low contrast, etc. To address these challenges, we introduce AttnPAFPN, a high-resolution detection pipeline that leverages a novel transformer variation, the efficient-global self-attention mechanism. Our streamlined approach can be easily integrated in almost any multi-scale object detection pipeline. In a comprehensive evaluation on the publicly available AGAR dataset, we demonstrate the superior accuracy of our network over the current state-of-the-art. In order to demonstrate the task-independent performance of our approach, we perform further experiments on COCO and LIVECell datasets.

{{</citation>}}


### (19/87) Self-Supervised Single-Image Deconvolution with Siamese Neural Networks (Mikhail Papkov et al., 2023)

{{<citation>}}

Mikhail Papkov, Kaupo Palo, Leopold Parts. (2023)  
**Self-Supervised Single-Image Deconvolution with Siamese Neural Networks**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.09426v1)  

---


**ABSTRACT**  
Inverse problems in image reconstruction are fundamentally complicated by unknown noise properties. Classical iterative deconvolution approaches amplify noise and require careful parameter selection for an optimal trade-off between sharpness and grain. Deep learning methods allow for flexible parametrization of the noise and learning its properties directly from the data. Recently, self-supervised blind-spot neural networks were successfully adopted for image deconvolution by including a known point-spread function in the end-to-end training. However, their practical application has been limited to 2D images in the biomedical domain because it implies large kernels that are poorly optimized. We tackle this problem with Fast Fourier Transform convolutions that provide training speed-up in 3D microscopy deconvolution tasks. Further, we propose to adopt a Siamese invariance loss for deconvolution and empirically identify its optimal position in the neural network between blind-spot and full image branches. The experimental results show that our improved framework outperforms the previous state-of-the-art deconvolution methods with a known point spread function.

{{</citation>}}


### (20/87) MonoNeRD: NeRF-like Representations for Monocular 3D Object Detection (Junkai Xu et al., 2023)

{{<citation>}}

Junkai Xu, Liang Peng, Haoran Cheng, Hao Li, Wei Qian, Ke Li, Wenxiao Wang, Deng Cai. (2023)  
**MonoNeRD: NeRF-like Representations for Monocular 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.09421v1)  

---


**ABSTRACT**  
In the field of monocular 3D detection, it is common practice to utilize scene geometric clues to enhance the detector's performance. However, many existing works adopt these clues explicitly such as estimating a depth map and back-projecting it into 3D space. This explicit methodology induces sparsity in 3D representations due to the increased dimensionality from 2D to 3D, and leads to substantial information loss, especially for distant and occluded objects. To alleviate this issue, we propose MonoNeRD, a novel detection framework that can infer dense 3D geometry and occupancy. Specifically, we model scenes with Signed Distance Functions (SDF), facilitating the production of dense 3D representations. We treat these representations as Neural Radiance Fields (NeRF) and then employ volume rendering to recover RGB images and depth maps. To the best of our knowledge, this work is the first to introduce volume rendering for M3D, and demonstrates the potential of implicit reconstruction for image-based 3D perception. Extensive experiments conducted on the KITTI-3D benchmark and Waymo Open Dataset demonstrate the effectiveness of MonoNeRD. Codes are available at https://github.com/cskkxjk/MonoNeRD.

{{</citation>}}


### (21/87) Diffusion Models for Image Restoration and Enhancement -- A Comprehensive Survey (Xin Li et al., 2023)

{{<citation>}}

Xin Li, Yulin Ren, Xin Jin, Cuiling Lan, Xingrui Wang, Wenjun Zeng, Xinchao Wang, Zhibo Chen. (2023)  
**Diffusion Models for Image Restoration and Enhancement -- A Comprehensive Survey**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09388v1)  

---


**ABSTRACT**  
Image restoration (IR) has been an indispensable and challenging task in the low-level vision field, which strives to improve the subjective quality of images distorted by various forms of degradation. Recently, the diffusion model has achieved significant advancements in the visual generation of AIGC, thereby raising an intuitive question, "whether diffusion model can boost image restoration". To answer this, some pioneering studies attempt to integrate diffusion models into the image restoration task, resulting in superior performances than previous GAN-based methods. Despite that, a comprehensive and enlightening survey on diffusion model-based image restoration remains scarce. In this paper, we are the first to present a comprehensive review of recent diffusion model-based methods on image restoration, encompassing the learning paradigm, conditional strategy, framework design, modeling strategy, and evaluation. Concretely, we first introduce the background of the diffusion model briefly and then present two prevalent workflows that exploit diffusion models in image restoration. Subsequently, we classify and emphasize the innovative designs using diffusion models for both IR and blind/real-world IR, intending to inspire future development. To evaluate existing methods thoroughly, we summarize the commonly-used dataset, implementation details, and evaluation metrics. Additionally, we present the objective comparison for open-sourced methods across three tasks, including image super-resolution, deblurring, and inpainting. Ultimately, informed by the limitations in existing works, we propose five potential and challenging directions for the future research of diffusion model-based IR, including sampling efficiency, model compression, distortion simulation and estimation, distortion invariant learning, and framework design.

{{</citation>}}


### (22/87) DReg-NeRF: Deep Registration for Neural Radiance Fields (Yu Chen et al., 2023)

{{<citation>}}

Yu Chen, Gim Hee Lee. (2023)  
**DReg-NeRF: Deep Registration for Neural Radiance Fields**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09386v1)  

---


**ABSTRACT**  
Although Neural Radiance Fields (NeRF) is popular in the computer vision community recently, registering multiple NeRFs has yet to gain much attention. Unlike the existing work, NeRF2NeRF, which is based on traditional optimization methods and needs human annotated keypoints, we propose DReg-NeRF to solve the NeRF registration problem on object-centric scenes without human intervention. After training NeRF models, our DReg-NeRF first extracts features from the occupancy grid in NeRF. Subsequently, our DReg-NeRF utilizes a transformer architecture with self-attention and cross-attention layers to learn the relations between pairwise NeRF blocks. In contrast to state-of-the-art (SOTA) point cloud registration methods, the decoupled correspondences are supervised by surface fields without any ground truth overlapping labels. We construct a novel view synthesis dataset with 1,700+ 3D objects obtained from Objaverse to train our network. When evaluated on the test set, our proposed method beats the SOTA point cloud registration methods by a large margin, with a mean $\text{RPE}=9.67^{\circ}$ and a mean $\text{RTE}=0.038$.   Our code is available at https://github.com/AIBluefisher/DReg-NeRF.

{{</citation>}}


### (23/87) Which Transformer to Favor: A Comparative Analysis of Efficiency in Vision Transformers (Tobias Christian Nauen et al., 2023)

{{<citation>}}

Tobias Christian Nauen, Sebastian Palacio, Andreas Dengel. (2023)  
**Which Transformer to Favor: A Comparative Analysis of Efficiency in Vision Transformers**  

---
Primary Category: cs.CV  
Categories: 68T07, I-4-0; I-2-10; I-5-1, cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2308.09372v1)  

---


**ABSTRACT**  
The growing popularity of Vision Transformers as the go-to models for image classification has led to an explosion of architectural modifications claiming to be more efficient than the original ViT. However, a wide diversity of experimental conditions prevents a fair comparison between all of them, based solely on their reported results. To address this gap in comparability, we conduct a comprehensive analysis of more than 30 models to evaluate the efficiency of vision transformers and related architectures, considering various performance metrics. Our benchmark provides a comparable baseline across the landscape of efficiency-oriented transformers, unveiling a plethora of surprising insights. For example, we discover that ViT is still Pareto optimal across multiple efficiency metrics, despite the existence of several alternative approaches claiming to be more efficient. Results also indicate that hybrid attention-CNN models fare particularly well when it comes to low inference memory and number of parameters, and also that it is better to scale the model size, than the image size. Furthermore, we uncover a strong positive correlation between the number of FLOPS and the training memory, which enables the estimation of required VRAM from theoretical measurements alone.   Thanks to our holistic evaluation, this study offers valuable insights for practitioners and researchers, facilitating informed decisions when selecting models for specific applications. We publicly release our code and data at https://github.com/tobna/WhatTransformerToFavor

{{</citation>}}


### (24/87) Single Frame Semantic Segmentation Using Multi-Modal Spherical Images (Suresh Guttikonda et al., 2023)

{{<citation>}}

Suresh Guttikonda, Jason Rambach. (2023)  
**Single Frame Semantic Segmentation Using Multi-Modal Spherical Images**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.09369v1)  

---


**ABSTRACT**  
In recent years, the research community has shown a lot of interest to panoramic images that offer a 360-degree directional perspective. Multiple data modalities can be fed, and complimentary characteristics can be utilized for more robust and rich scene interpretation based on semantic segmentation, to fully realize the potential. Existing research, however, mostly concentrated on pinhole RGB-X semantic segmentation. In this study, we propose a transformer-based cross-modal fusion architecture to bridge the gap between multi-modal fusion and omnidirectional scene perception. We employ distortion-aware modules to address extreme object deformations and panorama distortions that result from equirectangular representation. Additionally, we conduct cross-modal interactions for feature rectification and information exchange before merging the features in order to communicate long-range contexts for bi-modal and tri-modal feature streams. In thorough tests using combinations of four different modality types in three indoor panoramic-view datasets, our technique achieved state-of-the-art mIoU performance: 60.60% on Stanford2D3DS (RGB-HHA), 71.97% Structured3D (RGB-D-N), and 35.92% Matterport3D (RGB-D). We plan to release all codes and trained models soon.

{{</citation>}}


### (25/87) A tailored Handwritten-Text-Recognition System for Medieval Latin (Philipp Koch et al., 2023)

{{<citation>}}

Philipp Koch, Gilary Vera Nuñez, Esteban Garces Arias, Christian Heumann, Matthias Schöffel, Alexander Häberlin, Matthias Aßenmacher. (2023)  
**A tailored Handwritten-Text-Recognition System for Medieval Latin**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs-CY, cs-LG, cs.CV, stat-ML  
Keywords: GPT, Google  
[Paper Link](http://arxiv.org/abs/2308.09368v1)  

---


**ABSTRACT**  
The Bavarian Academy of Sciences and Humanities aims to digitize its Medieval Latin Dictionary. This dictionary entails record cards referring to lemmas in medieval Latin, a low-resource language. A crucial step of the digitization process is the Handwritten Text Recognition (HTR) of the handwritten lemmas found on these record cards. In our work, we introduce an end-to-end pipeline, tailored to the medieval Latin dictionary, for locating, extracting, and transcribing the lemmas. We employ two state-of-the-art (SOTA) image segmentation models to prepare the initial data set for the HTR task. Furthermore, we experiment with different transformer-based models and conduct a set of experiments to explore the capabilities of different combinations of vision encoders with a GPT-2 decoder. Additionally, we also apply extensive data augmentation resulting in a highly competitive model. The best-performing setup achieved a Character Error Rate (CER) of 0.015, which is even superior to the commercial Google Cloud Vision model, and shows more stable performance.

{{</citation>}}


### (26/87) Overlap Bias Matching is Necessary for Point Cloud Registration (Pengcheng Shi et al., 2023)

{{<citation>}}

Pengcheng Shi, Jie Zhang, Haozhe Cheng, Junyang Wang, Yiyang Zhou, Chenlin Zhao, Jihua Zhu. (2023)  
**Overlap Bias Matching is Necessary for Point Cloud Registration**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2308.09364v1)  

---


**ABSTRACT**  
Point cloud registration is a fundamental problem in many domains. Practically, the overlap between point clouds to be registered may be relatively small. Most unsupervised methods lack effective initial evaluation of overlap, leading to suboptimal registration accuracy. To address this issue, we propose an unsupervised network Overlap Bias Matching Network (OBMNet) for partial point cloud registration. Specifically, we propose a plug-and-play Overlap Bias Matching Module (OBMM) comprising two integral components, overlap sampling module and bias prediction module. These two components are utilized to capture the distribution of overlapping regions and predict bias coefficients of point cloud common structures, respectively. Then, we integrate OBMM with the neighbor map matching module to robustly identify correspondences by precisely merging matching scores of points within the neighborhood, which addresses the ambiguities in single-point features. OBMNet can maintain efficacy even in pair-wise registration scenarios with low overlap ratios. Experimental results on extensive datasets demonstrate that our approach's performance achieves a significant improvement compared to the state-of-the-art registration approach.

{{</citation>}}


### (27/87) Open-vocabulary Video Question Answering: A New Benchmark for Evaluating the Generalizability of Video Question Answering Models (Dohwan Ko et al., 2023)

{{<citation>}}

Dohwan Ko, Ji Soo Lee, Miso Choi, Jaewon Chu, Jihwan Park, Hyunwoo J. Kim. (2023)  
**Open-vocabulary Video Question Answering: A New Benchmark for Evaluating the Generalizability of Video Question Answering Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GNN, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2308.09363v1)  

---


**ABSTRACT**  
Video Question Answering (VideoQA) is a challenging task that entails complex multi-modal reasoning. In contrast to multiple-choice VideoQA which aims to predict the answer given several options, the goal of open-ended VideoQA is to answer questions without restricting candidate answers. However, the majority of previous VideoQA models formulate open-ended VideoQA as a classification task to classify the video-question pairs into a fixed answer set, i.e., closed-vocabulary, which contains only frequent answers (e.g., top-1000 answers). This leads the model to be biased toward only frequent answers and fail to generalize on out-of-vocabulary answers. We hence propose a new benchmark, Open-vocabulary Video Question Answering (OVQA), to measure the generalizability of VideoQA models by considering rare and unseen answers. In addition, in order to improve the model's generalization power, we introduce a novel GNN-based soft verbalizer that enhances the prediction on rare and unseen answers by aggregating the information from their similar words. For evaluation, we introduce new baselines by modifying the existing (closed-vocabulary) open-ended VideoQA models and improve their performances by further taking into account rare and unseen answers. Our ablation studies and qualitative analyses demonstrate that our GNN-based soft verbalizer further improves the model performance, especially on rare and unseen answers. We hope that our benchmark OVQA can serve as a guide for evaluating the generalizability of VideoQA models and inspire future research. Code is available at https://github.com/mlvlab/OVQA.

{{</citation>}}


### (28/87) Unlimited Knowledge Distillation for Action Recognition in the Dark (Ruibing Jin et al., 2023)

{{<citation>}}

Ruibing Jin, Guosheng Lin, Min Wu, Jie Lin, Zhengguo Li, Xiaoli Li, Zhenghua Chen. (2023)  
**Unlimited Knowledge Distillation for Action Recognition in the Dark**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2308.09327v1)  

---


**ABSTRACT**  
Dark videos often lose essential information, which causes the knowledge learned by networks is not enough to accurately recognize actions. Existing knowledge assembling methods require massive GPU memory to distill the knowledge from multiple teacher models into a student model. In action recognition, this drawback becomes serious due to much computation required by video process. Constrained by limited computation source, these approaches are infeasible. To address this issue, we propose an unlimited knowledge distillation (UKD) in this paper. Compared with existing knowledge assembling methods, our UKD can effectively assemble different knowledge without introducing high GPU memory consumption. Thus, the number of teaching models for distillation is unlimited. With our UKD, the network's learned knowledge can be remarkably enriched. Our experiments show that the single stream network distilled with our UKD even surpasses a two-stream network. Extensive experiments are conducted on the ARID dataset.

{{</citation>}}


### (29/87) Audio-Visual Glance Network for Efficient Video Recognition (Muhammad Adi Nugroho et al., 2023)

{{<citation>}}

Muhammad Adi Nugroho, Sangmin Woo, Sumin Lee, Changick Kim. (2023)  
**Audio-Visual Glance Network for Efficient Video Recognition**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-MM, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09322v1)  

---


**ABSTRACT**  
Deep learning has made significant strides in video understanding tasks, but the computation required to classify lengthy and massive videos using clip-level video classifiers remains impractical and prohibitively expensive. To address this issue, we propose Audio-Visual Glance Network (AVGN), which leverages the commonly available audio and visual modalities to efficiently process the spatio-temporally important parts of a video. AVGN firstly divides the video into snippets of image-audio clip pair and employs lightweight unimodal encoders to extract global visual features and audio features. To identify the important temporal segments, we use an Audio-Visual Temporal Saliency Transformer (AV-TeST) that estimates the saliency scores of each frame. To further increase efficiency in the spatial dimension, AVGN processes only the important patches instead of the whole images. We use an Audio-Enhanced Spatial Patch Attention (AESPA) module to produce a set of enhanced coarse visual features, which are fed to a policy network that produces the coordinates of the important patches. This approach enables us to focus only on the most important spatio-temporally parts of the video, leading to more efficient video recognition. Moreover, we incorporate various training techniques and multi-modal feature fusion to enhance the robustness and effectiveness of our AVGN. By combining these strategies, our AVGN sets new state-of-the-art performance in multiple video recognition benchmarks while achieving faster processing speed.

{{</citation>}}


### (30/87) Retro-FPN: Retrospective Feature Pyramid Network for Point Cloud Semantic Segmentation (Peng Xiang et al., 2023)

{{<citation>}}

Peng Xiang, Xin Wen, Yu-Shen Liu, Hui Zhang, Yi Fang, Zhizhong Han. (2023)  
**Retro-FPN: Retrospective Feature Pyramid Network for Point Cloud Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2308.09314v1)  

---


**ABSTRACT**  
Learning per-point semantic features from the hierarchical feature pyramid is essential for point cloud semantic segmentation. However, most previous methods suffered from ambiguous region features or failed to refine per-point features effectively, which leads to information loss and ambiguous semantic identification. To resolve this, we propose Retro-FPN to model the per-point feature prediction as an explicit and retrospective refining process, which goes through all the pyramid layers to extract semantic features explicitly for each point. Its key novelty is a retro-transformer for summarizing semantic contexts from the previous layer and accordingly refining the features in the current stage. In this way, the categorization of each point is conditioned on its local semantic pattern. Specifically, the retro-transformer consists of a local cross-attention block and a semantic gate unit. The cross-attention serves to summarize the semantic pattern retrospectively from the previous layer. And the gate unit carefully incorporates the summarized contexts and refines the current semantic features. Retro-FPN is a pluggable neural network that applies to hierarchical decoders. By integrating Retro-FPN with three representative backbones, including both point-based and voxel-based methods, we show that Retro-FPN can significantly improve performance over state-of-the-art backbones. Comprehensive experiments on widely used benchmarks can justify the effectiveness of our design. The source is available at https://github.com/AllenXiangX/Retro-FPN

{{</citation>}}


### (31/87) Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering (Haiwei Wu et al., 2023)

{{<citation>}}

Haiwei Wu, Yiming Chen, Jiantao Zhou. (2023)  
**Rethinking Image Forgery Detection via Contrastive Learning and Unsupervised Clustering**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2308.09307v1)  

---


**ABSTRACT**  
Image forgery detection aims to detect and locate forged regions in an image. Most existing forgery detection algorithms formulate classification problems to classify pixels into forged or pristine. However, the definition of forged and pristine pixels is only relative within one single image, e.g., a forged region in image A is actually a pristine one in its source image B (splicing forgery). Such a relative definition has been severely overlooked by existing methods, which unnecessarily mix forged (pristine) regions across different images into the same category. To resolve this dilemma, we propose the FOrensic ContrAstive cLustering (FOCAL) method, a novel, simple yet very effective paradigm based on contrastive learning and unsupervised clustering for the image forgery detection. Specifically, FOCAL 1) utilizes pixel-level contrastive learning to supervise the high-level forensic feature extraction in an image-by-image manner, explicitly reflecting the above relative definition; 2) employs an on-the-fly unsupervised clustering algorithm (instead of a trained one) to cluster the learned features into forged/pristine categories, further suppressing the cross-image influence from training data; and 3) allows to further boost the detection performance via simple feature-level concatenation without the need of retraining. Extensive experimental results over six public testing datasets demonstrate that our proposed FOCAL significantly outperforms the state-of-the-art competing algorithms by big margins: +24.3% on Coverage, +18.6% on Columbia, +17.5% on FF++, +14.2% on MISD, +13.5% on CASIA and +10.3% on NIST in terms of IoU. The paradigm of FOCAL could bring fresh insights and serve as a novel benchmark for the image forgery detection task. The code is available at https://github.com/HighwayWu/FOCAL.

{{</citation>}}


### (32/87) Human Part-wise 3D Motion Context Learning for Sign Language Recognition (Taeryung Lee et al., 2023)

{{<citation>}}

Taeryung Lee, Yeonguk Oh, Kyoung Mu Lee. (2023)  
**Human Part-wise 3D Motion Context Learning for Sign Language Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.09305v1)  

---


**ABSTRACT**  
In this paper, we propose P3D, the human part-wise motion context learning framework for sign language recognition. Our main contributions lie in two dimensions: learning the part-wise motion context and employing the pose ensemble to utilize 2D and 3D pose jointly. First, our empirical observation implies that part-wise context encoding benefits the performance of sign language recognition. While previous methods of sign language recognition learned motion context from the sequence of the entire pose, we argue that such methods cannot exploit part-specific motion context. In order to utilize part-wise motion context, we propose the alternating combination of a part-wise encoding Transformer (PET) and a whole-body encoding Transformer (WET). PET encodes the motion contexts from a part sequence, while WET merges them into a unified context. By learning part-wise motion context, our P3D achieves superior performance on WLASL compared to previous state-of-the-art methods. Second, our framework is the first to ensemble 2D and 3D poses for sign language recognition. Since the 3D pose holds rich motion context and depth information to distinguish the words, our P3D outperformed the previous state-of-the-art methods employing a pose ensemble.

{{</citation>}}


### (33/87) V2A-Mapper: A Lightweight Solution for Vision-to-Audio Generation by Connecting Foundation Models (Heng Wang et al., 2023)

{{<citation>}}

Heng Wang, Jianbo Ma, Santiago Pascual, Richard Cartwright, Weidong Cai. (2023)  
**V2A-Mapper: A Lightweight Solution for Vision-to-Audio Generation by Connecting Foundation Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-MM, cs-SD, cs.CV, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09300v1)  

---


**ABSTRACT**  
Building artificial intelligence (AI) systems on top of a set of foundation models (FMs) is becoming a new paradigm in AI research. Their representative and generative abilities learnt from vast amounts of data can be easily adapted and transferred to a wide range of downstream tasks without extra training from scratch. However, leveraging FMs in cross-modal generation remains under-researched when audio modality is involved. On the other hand, automatically generating semantically-relevant sound from visual input is an important problem in cross-modal generation studies. To solve this vision-to-audio (V2A) generation problem, existing methods tend to design and build complex systems from scratch using modestly sized datasets. In this paper, we propose a lightweight solution to this problem by leveraging foundation models, specifically CLIP, CLAP, and AudioLDM. We first investigate the domain gap between the latent space of the visual CLIP and the auditory CLAP models. Then we propose a simple yet effective mapper mechanism (V2A-Mapper) to bridge the domain gap by translating the visual input between CLIP and CLAP spaces. Conditioned on the translated CLAP embedding, pretrained audio generative FM AudioLDM is adopted to produce high-fidelity and visually-aligned sound. Compared to previous approaches, our method only requires a quick training of the V2A-Mapper. We further analyze and conduct extensive experiments on the choice of the V2A-Mapper and show that a generative mapper is better at fidelity and variability (FD) while a regression mapper is slightly better at relevance (CS). Both objective and subjective evaluation on two V2A datasets demonstrate the superiority of our proposed method compared to current state-of-the-art approaches - trained with 86% fewer parameters but achieving 53% and 19% improvement in FD and CS, respectively.

{{</citation>}}


### (34/87) NAPA-VQ: Neighborhood Aware Prototype Augmentation with Vector Quantization for Continual Learning (Tamasha Malepathirana et al., 2023)

{{<citation>}}

Tamasha Malepathirana, Damith Senanayake, Saman Halgamuge. (2023)  
**NAPA-VQ: Neighborhood Aware Prototype Augmentation with Vector Quantization for Continual Learning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Augmentation, ImageNet, Quantization  
[Paper Link](http://arxiv.org/abs/2308.09297v1)  

---


**ABSTRACT**  
Catastrophic forgetting; the loss of old knowledge upon acquiring new knowledge, is a pitfall faced by deep neural networks in real-world applications. Many prevailing solutions to this problem rely on storing exemplars (previously encountered data), which may not be feasible in applications with memory limitations or privacy constraints. Therefore, the recent focus has been on Non-Exemplar based Class Incremental Learning (NECIL) where a model incrementally learns about new classes without using any past exemplars. However, due to the lack of old data, NECIL methods struggle to discriminate between old and new classes causing their feature representations to overlap. We propose NAPA-VQ: Neighborhood Aware Prototype Augmentation with Vector Quantization, a framework that reduces this class overlap in NECIL. We draw inspiration from Neural Gas to learn the topological relationships in the feature space, identifying the neighboring classes that are most likely to get confused with each other. This neighborhood information is utilized to enforce strong separation between the neighboring classes as well as to generate old class representative prototypes that can better aid in obtaining a discriminative decision boundary between old and new classes. Our comprehensive experiments on CIFAR-100, TinyImageNet, and ImageNet-Subset demonstrate that NAPA-VQ outperforms the State-of-the-art NECIL methods by an average improvement of 5%, 2%, and 4% in accuracy and 10%, 3%, and 9% in forgetting respectively. Our code can be found in https://github.com/TamashaM/NAPA-VQ.git.

{{</citation>}}


### (35/87) Self-Calibrated Cross Attention Network for Few-Shot Segmentation (Qianxiong Xu et al., 2023)

{{<citation>}}

Qianxiong Xu, Wenting Zhao, Guosheng Lin, Cheng Long. (2023)  
**Self-Calibrated Cross Attention Network for Few-Shot Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Few-Shot  
[Paper Link](http://arxiv.org/abs/2308.09294v1)  

---


**ABSTRACT**  
The key to the success of few-shot segmentation (FSS) lies in how to effectively utilize support samples. Most solutions compress support foreground (FG) features into prototypes, but lose some spatial details. Instead, others use cross attention to fuse query features with uncompressed support FG. Query FG could be fused with support FG, however, query background (BG) cannot find matched BG features in support FG, yet inevitably integrates dissimilar features. Besides, as both query FG and BG are combined with support FG, they get entangled, thereby leading to ineffective segmentation. To cope with these issues, we design a self-calibrated cross attention (SCCA) block. For efficient patch-based attention, query and support features are firstly split into patches. Then, we design a patch alignment module to align each query patch with its most similar support patch for better cross attention. Specifically, SCCA takes a query patch as Q, and groups the patches from the same query image and the aligned patches from the support image as K&V. In this way, the query BG features are fused with matched BG features (from query patches), and thus the aforementioned issues will be mitigated. Moreover, when calculating SCCA, we design a scaled-cosine mechanism to better utilize the support features for similarity calculation. Extensive experiments conducted on PASCAL-5^i and COCO-20^i demonstrate the superiority of our model, e.g., the mIoU score under 5-shot setting on COCO-20^i is 5.6%+ better than previous state-of-the-arts. The code is available at https://github.com/Sam1224/SCCAN.

{{</citation>}}


### (36/87) Diverse Cotraining Makes Strong Semi-Supervised Segmentor (Yijiang Li et al., 2023)

{{<citation>}}

Yijiang Li, Xinjiang Wang, Lihe Yang, Litong Feng, Wayne Zhang, Ying Gao. (2023)  
**Diverse Cotraining Makes Strong Semi-Supervised Segmentor**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2308.09281v1)  

---


**ABSTRACT**  
Deep co-training has been introduced to semi-supervised segmentation and achieves impressive results, yet few studies have explored the working mechanism behind it. In this work, we revisit the core assumption that supports co-training: multiple compatible and conditionally independent views. By theoretically deriving the generalization upper bound, we prove the prediction similarity between two models negatively impacts the model's generalization ability. However, most current co-training models are tightly coupled together and violate this assumption. Such coupling leads to the homogenization of networks and confirmation bias which consequently limits the performance. To this end, we explore different dimensions of co-training and systematically increase the diversity from the aspects of input domains, different augmentations and model architectures to counteract homogenization. Our Diverse Co-training outperforms the state-of-the-art (SOTA) methods by a large margin across different evaluation protocols on the Pascal and Cityscapes. For example. we achieve the best mIoU of 76.2%, 77.7% and 80.2% on Pascal with only 92, 183 and 366 labeled images, surpassing the previous best results by more than 5%.

{{</citation>}}


### (37/87) Point Contrastive Prediction with Semantic Clustering for Self-Supervised Learning on Point Cloud Videos (Xiaoxiao Sheng et al., 2023)

{{<citation>}}

Xiaoxiao Sheng, Zhiqiang Shen, Gang Xiao, Longguang Wang, Yulan Guo, Hehe Fan. (2023)  
**Point Contrastive Prediction with Semantic Clustering for Self-Supervised Learning on Point Cloud Videos**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Self-Supervised  
[Paper Link](http://arxiv.org/abs/2308.09247v1)  

---


**ABSTRACT**  
We propose a unified point cloud video self-supervised learning framework for object-centric and scene-centric data. Previous methods commonly conduct representation learning at the clip or frame level and cannot well capture fine-grained semantics. Instead of contrasting the representations of clips or frames, in this paper, we propose a unified self-supervised framework by conducting contrastive learning at the point level. Moreover, we introduce a new pretext task by achieving semantic alignment of superpoints, which further facilitates the representations to capture semantic cues at multiple scales. In addition, due to the high redundancy in the temporal dimension of dynamic point clouds, directly conducting contrastive learning at the point level usually leads to massive undesired negatives and insufficient modeling of positive representations. To remedy this, we propose a selection strategy to retain proper negatives and make use of high-similarity samples from other instances as positive supplements. Extensive experiments show that our method outperforms supervised counterparts on a wide range of downstream tasks and demonstrates the superior transferability of the learned representations.

{{</citation>}}


### (38/87) SparseBEV: High-Performance Sparse 3D Object Detection from Multi-Camera Videos (Haisong Liu et al., 2023)

{{<citation>}}

Haisong Liu, Yao Teng, Tao Lu, Haiguang Wang, Limin Wang. (2023)  
**SparseBEV: High-Performance Sparse 3D Object Detection from Multi-Camera Videos**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2308.09244v1)  

---


**ABSTRACT**  
Camera-based 3D object detection in BEV (Bird's Eye View) space has drawn great attention over the past few years. Dense detectors typically follow a two-stage pipeline by first constructing a dense BEV feature and then performing object detection in BEV space, which suffers from complex view transformations and high computation cost. On the other side, sparse detectors follow a query-based paradigm without explicit dense BEV feature construction, but achieve worse performance than the dense counterparts. In this paper, we find that the key to mitigate this performance gap is the adaptability of the detector in both BEV and image space. To achieve this goal, we propose SparseBEV, a fully sparse 3D object detector that outperforms the dense counterparts. SparseBEV contains three key designs, which are (1) scale-adaptive self attention to aggregate features with adaptive receptive field in BEV space, (2) adaptive spatio-temporal sampling to generate sampling locations under the guidance of queries, and (3) adaptive mixing to decode the sampled features with dynamic weights from the queries. On the test split of nuScenes, SparseBEV achieves the state-of-the-art performance of 67.5 NDS. On the val split, SparseBEV achieves 55.8 NDS while maintaining a real-time inference speed of 23.5 FPS. Code is available at https://github.com/MCG-NJU/SparseBEV.

{{</citation>}}


## cs.CL (15)



### (39/87) Graph of Thoughts: Solving Elaborate Problems with Large Language Models (Maciej Besta et al., 2023)

{{<citation>}}

Maciej Besta, Nils Blach, Ales Kubicek, Robert Gerstenberger, Lukas Gianinazzi, Joanna Gajda, Tomasz Lehmann, Michal Podstawski, Hubert Niewiadomski, Piotr Nyczyk, Torsten Hoefler. (2023)  
**Graph of Thoughts: Solving Elaborate Problems with Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.09687v1)  

---


**ABSTRACT**  
We introduce Graph of Thoughts (GoT): a framework that advances prompting capabilities in large language models (LLMs) beyond those offered by paradigms such as Chain-ofThought or Tree of Thoughts (ToT). The key idea and primary advantage of GoT is the ability to model the information generated by an LLM as an arbitrary graph, where units of information ("LLM thoughts") are vertices, and edges correspond to dependencies between these vertices. This approach enables combining arbitrary LLM thoughts into synergistic outcomes, distilling the essence of whole networks of thoughts, or enhancing thoughts using feedback loops. We illustrate that GoT offers advantages over state of the art on different tasks, for example increasing the quality of sorting by 62% over ToT, while simultaneously reducing costs by >31%. We ensure that GoT is extensible with new thought transformations and thus can be used to spearhead new prompting schemes. This work brings the LLM reasoning closer to human thinking or brain mechanisms such as recurrence, both of which form complex networks.

{{</citation>}}


### (40/87) OCR Language Models with Custom Vocabularies (Peter Garst et al., 2023)

{{<citation>}}

Peter Garst, Reeve Ingle, Yasuhisa Fujii. (2023)  
**OCR Language Models with Custom Vocabularies**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, OCR  
[Paper Link](http://arxiv.org/abs/2308.09671v1)  

---


**ABSTRACT**  
Language models are useful adjuncts to optical models for producing accurate optical character recognition (OCR) results. One factor which limits the power of language models in this context is the existence of many specialized domains with language statistics very different from those implied by a general language model - think of checks, medical prescriptions, and many other specialized document classes. This paper introduces an algorithm for efficiently generating and attaching a domain specific word based language model at run time to a general language model in an OCR system. In order to best use this model the paper also introduces a modified CTC beam search decoder which effectively allows hypotheses to remain in contention based on possible future completion of vocabulary words. The result is a substantial reduction in word error rate in recognizing material from specialized domains.

{{</citation>}}


### (41/87) Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment (Rishabh Bhardwaj et al., 2023)

{{<citation>}}

Rishabh Bhardwaj, Soujanya Poria. (2023)  
**Red-Teaming Large Language Models using Chain of Utterances for Safety-Alignment**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT, GPT-4, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2308.09662v1)  

---


**ABSTRACT**  
Larger language models (LLMs) have taken the world by storm with their massive multi-tasking capabilities simply by optimizing over a next-word prediction objective. With the emergence of their properties and encoded knowledge, the risk of LLMs producing harmful outputs increases, making them unfit for scalable deployment for the public. In this work, we propose a new safety evaluation benchmark RED-EVAL that carries out red-teaming. We show that even widely deployed models are susceptible to the Chain of Utterances-based (CoU) prompting, jailbreaking closed source LLM-based systems such as GPT-4 and ChatGPT to unethically respond to more than 65% and 73% of harmful queries. We also demonstrate the consistency of the RED-EVAL across 8 open-source LLMs in generating harmful responses in more than 86% of the red-teaming attempts. Next, we propose RED-INSTRUCT--An approach for the safety alignment of LLMs. It constitutes two phases: 1) HARMFULQA data collection: Leveraging CoU prompting, we collect a dataset that consists of 1.9K harmful questions covering a wide range of topics, 9.5K safe and 7.3K harmful conversations from ChatGPT; 2) SAFE-ALIGN: We demonstrate how the conversational dataset can be used for the safety alignment of LLMs by minimizing the negative log-likelihood over helpful responses and penalizing over harmful responses by gradient accent over sample loss. Our model STARLING, a fine-tuned Vicuna-7B, is observed to be more safely aligned when evaluated on RED-EVAL and HHH benchmarks while preserving the utility of the baseline models (TruthfulQA, MMLU, and BBH).

{{</citation>}}


### (42/87) Tree-of-Mixed-Thought: Combining Fast and Slow Thinking for Multi-hop Visual Reasoning (Pengbo Hu et al., 2023)

{{<citation>}}

Pengbo Hu, Ji Qi, Xingyu Li, Hong Li, Xinqi Wang, Bing Quan, Ruiyu Wang, Yi Zhou. (2023)  
**Tree-of-Mixed-Thought: Combining Fast and Slow Thinking for Multi-hop Visual Reasoning**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CV, cs.CL  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.09658v1)  

---


**ABSTRACT**  
There emerges a promising trend of using large language models (LLMs) to generate code-like plans for complex inference tasks such as visual reasoning. This paradigm, known as LLM-based planning, provides flexibility in problem solving and endows better interpretability. However, current research is mostly limited to basic scenarios of simple questions that can be straightforward answered in a few inference steps. Planning for the more challenging multi-hop visual reasoning tasks remains under-explored. Specifically, under multi-hop reasoning situations, the trade-off between accuracy and the complexity of plan-searching becomes prominent. The prevailing algorithms either address the efficiency issue by employing the fast one-stop generation or adopt a complex iterative generation method to improve accuracy. Both fail to balance the need for efficiency and performance. Drawing inspiration from the dual system of cognition in the human brain, the fast and the slow think processes, we propose a hierarchical plan-searching algorithm that integrates the one-stop reasoning (fast) and the Tree-of-thought (slow). Our approach succeeds in performance while significantly saving inference steps. Moreover, we repurpose the PTR and the CLEVER datasets, developing a systematic framework for evaluating the performance and efficiency of LLMs-based plan-search algorithms under reasoning tasks at different levels of difficulty. Extensive experiments demonstrate the superiority of our proposed algorithm in terms of performance and efficiency. The dataset and code will be release soon.

{{</citation>}}


### (43/87) ChatHaruhi: Reviving Anime Character in Reality via Large Language Model (Cheng Li et al., 2023)

{{<citation>}}

Cheng Li, Ziang Leng, Chenxi Yan, Junyi Shen, Hao Wang, Weishi MI, Yaying Fei, Xiaoyang Feng, Song Yan, HaoSheng Wang, Linkang Zhan, Yaokai Jia, Pingyu Wu, Haozhen Sun. (2023)  
**ChatHaruhi: Reviving Anime Character in Reality via Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.09597v1)  

---


**ABSTRACT**  
Role-playing chatbots built on large language models have drawn interest, but better techniques are needed to enable mimicking specific fictional characters. We propose an algorithm that controls language models via an improved prompt and memories of the character extracted from scripts. We construct ChatHaruhi, a dataset covering 32 Chinese / English TV / anime characters with over 54k simulated dialogues. Both automatic and human evaluations show our approach improves role-playing ability over baselines. Code and data are available at https://github.com/LC1332/Chat-Haruhi-Suzumiya .

{{</citation>}}


### (44/87) WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct (Haipeng Luo et al., 2023)

{{<citation>}}

Haipeng Luo, Qingfeng Sun, Can Xu, Pu Zhao, Jianguang Lou, Chongyang Tao, Xiubo Geng, Qingwei Lin, Shifeng Chen, Dongmei Zhang. (2023)  
**WizardMath: Empowering Mathematical Reasoning for Large Language Models via Reinforced Evol-Instruct**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Language Model, NLP, PaLM, Reasoning, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.09583v1)  

---


**ABSTRACT**  
Large language models (LLMs), such as GPT-4, have shown remarkable performance in natural language processing (NLP) tasks, including challenging mathematical reasoning. However, most existing open-source models are only pre-trained on large-scale internet data and without math-related optimization. In this paper, we present WizardMath, which enhances the mathematical reasoning abilities of Llama-2, by applying our proposed Reinforcement Learning from Evol-Instruct Feedback (RLEIF) method to the domain of math. Through extensive experiments on two mathematical reasoning benchmarks, namely GSM8k and MATH, we reveal the extraordinary capabilities of our model. WizardMath surpasses all other open-source LLMs by a substantial margin. Furthermore, our model even outperforms ChatGPT-3.5, Claude Instant-1, PaLM-2 and Minerva on GSM8k, simultaneously surpasses Text-davinci-002, PaLM-1 and GPT-3 on MATH. More details and model weights are public at https://github.com/nlpxucan/WizardLM and https://huggingface.co/WizardLM.

{{</citation>}}


### (45/87) Predictive Authoring for Brazilian Portuguese Augmentative and Alternative Communication (Jayr Pereira et al., 2023)

{{<citation>}}

Jayr Pereira, Rodrigo Nogueira, Cleber Zanchettin, Robson Fidalgo. (2023)  
**Predictive Authoring for Brazilian Portuguese Augmentative and Alternative Communication**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: BERT  
[Paper Link](http://arxiv.org/abs/2308.09497v1)  

---


**ABSTRACT**  
Individuals with complex communication needs (CCN) often rely on augmentative and alternative communication (AAC) systems to have conversations and communique their wants. Such systems allow message authoring by arranging pictograms in sequence. However, the difficulty of finding the desired item to complete a sentence can increase as the user's vocabulary increases. This paper proposes using BERTimbau, a Brazilian Portuguese version of BERT, for pictogram prediction in AAC systems. To finetune BERTimbau, we constructed an AAC corpus for Brazilian Portuguese to use as a training corpus. We tested different approaches to representing a pictogram for prediction: as a word (using pictogram captions), as a concept (using a dictionary definition), and as a set of synonyms (using related terms). We also evaluated the usage of images for pictogram prediction. The results demonstrate that using embeddings computed from the pictograms' caption, synonyms, or definitions have a similar performance. Using synonyms leads to lower perplexity, but using captions leads to the highest accuracies. This paper provides insight into how to represent a pictogram for prediction using a BERT-like model and the potential of using images for pictogram prediction.

{{</citation>}}


### (46/87) Scope is all you need: Transforming LLMs for HPC Code (Tal Kadosh et al., 2023)

{{<citation>}}

Tal Kadosh, Niranjan Hasabnis, Vy A. Vo, Nadav Schneider, Neva Krien, Abdul Wasay, Nesreen Ahmed, Ted Willke, Guy Tamir, Yuval Pinter, Timothy Mattson, Gal Oren. (2023)  
**Scope is all you need: Transforming LLMs for HPC Code**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09440v1)  

---


**ABSTRACT**  
With easier access to powerful compute resources, there is a growing trend in the field of AI for software development to develop larger and larger language models (LLMs) to address a variety of programming tasks. Even LLMs applied to tasks from the high-performance computing (HPC) domain are huge in size (e.g., billions of parameters) and demand expensive compute resources for training. We found this design choice confusing - why do we need large LLMs trained on natural languages and programming languages unrelated to HPC for HPC-specific tasks? In this line of work, we aim to question design choices made by existing LLMs by developing smaller LLMs for specific domains - we call them domain-specific LLMs. Specifically, we start off with HPC as a domain and propose a novel tokenizer named Tokompiler, designed specifically for preprocessing code in HPC and compilation-centric tasks. Tokompiler leverages knowledge of language primitives to generate language-oriented tokens, providing a context-aware understanding of code structure while avoiding human semantics attributed to code structures completely. We applied Tokompiler to pre-train two state-of-the-art models, SPT-Code and Polycoder, for a Fortran code corpus mined from GitHub. We evaluate the performance of these models against the conventional LLMs. Results demonstrate that Tokompiler significantly enhances code completion accuracy and semantic understanding compared to traditional tokenizers in normalized-perplexity tests, down to ~1 perplexity score. This research opens avenues for further advancements in domain-specific LLMs, catering to the unique demands of HPC and compilation tasks.

{{</citation>}}


### (47/87) A Methodology for Generative Spelling Correction via Natural Spelling Errors Emulation across Multiple Domains and Languages (Nikita Martynov et al., 2023)

{{<citation>}}

Nikita Martynov, Mark Baushenko, Anastasia Kozlova, Katerina Kolomeytseva, Aleksandr Abramov, Alena Fenogenova. (2023)  
**A Methodology for Generative Spelling Correction via Natural Spelling Errors Emulation across Multiple Domains and Languages**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2308.09435v1)  

---


**ABSTRACT**  
Modern large language models demonstrate impressive capabilities in text generation and generalization. However, they often struggle with solving text editing tasks, particularly when it comes to correcting spelling errors and mistypings. In this paper, we present a methodology for generative spelling correction (SC), which was tested on English and Russian languages and potentially can be extended to any language with minor changes. Our research mainly focuses on exploring natural spelling errors and mistypings in texts and studying the ways those errors can be emulated in correct sentences to effectively enrich generative models' pre-train procedure. We investigate the impact of such emulations and the models' abilities across different text domains. In this work, we investigate two spelling corruption techniques: 1) first one mimics human behavior when making a mistake through leveraging statistics of errors from particular dataset and 2) second adds the most common spelling errors, keyboard miss clicks, and some heuristics within the texts. We conducted experiments employing various corruption strategies, models' architectures and sizes on the pre-training and fine-tuning stages and evaluated the models using single-domain and multi-domain test sets. As a practical outcome of our work, we introduce SAGE (Spell checking via Augmentation and Generative distribution Emulation) is a library for automatic generative SC that includes a family of pre-trained generative models and built-in augmentation algorithms.

{{</citation>}}


### (48/87) Leveraging Large Language Models for DRL-Based Anti-Jamming Strategies in Zero Touch Networks (Abubakar S. Ali et al., 2023)

{{<citation>}}

Abubakar S. Ali, Dimitrios Michael Manias, Abdallah Shami, Sami Muhaidat. (2023)  
**Leveraging Large Language Models for DRL-Based Anti-Jamming Strategies in Zero Touch Networks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2308.09376v1)  

---


**ABSTRACT**  
As the dawn of sixth-generation (6G) networking approaches, it promises unprecedented advancements in communication and automation. Among the leading innovations of 6G is the concept of Zero Touch Networks (ZTNs), aiming to achieve fully automated, self-optimizing networks with minimal human intervention. Despite the advantages ZTNs offer in terms of efficiency and scalability, challenges surrounding transparency, adaptability, and human trust remain prevalent. Concurrently, the advent of Large Language Models (LLMs) presents an opportunity to elevate the ZTN framework by bridging the gap between automated processes and human-centric interfaces. This paper explores the integration of LLMs into ZTNs, highlighting their potential to enhance network transparency and improve user interactions. Through a comprehensive case study on deep reinforcement learning (DRL)-based anti-jamming technique, we demonstrate how LLMs can distill intricate network operations into intuitive, human-readable reports. Additionally, we address the technical and ethical intricacies of melding LLMs with ZTNs, with an emphasis on data privacy, transparency, and bias reduction. Looking ahead, we identify emerging research avenues at the nexus of LLMs and ZTNs, advocating for sustained innovation and interdisciplinary synergy in the domain of automated networks.

{{</citation>}}


### (49/87) TrOMR:Transformer-Based Polyphonic Optical Music Recognition (Yixuan Li et al., 2023)

{{<citation>}}

Yixuan Li, Huaping Liu, Qiang Jin, Miaomiao Cai, Peng Li. (2023)  
**TrOMR:Transformer-Based Polyphonic Optical Music Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.09370v1)  

---


**ABSTRACT**  
Optical Music Recognition (OMR) is an important technology in music and has been researched for a long time. Previous approaches for OMR are usually based on CNN for image understanding and RNN for music symbol classification. In this paper, we propose a transformer-based approach with excellent global perceptual capability for end-to-end polyphonic OMR, called TrOMR. We also introduce a novel consistency loss function and a reasonable approach for data annotation to improve recognition accuracy for complex music scores. Extensive experiments demonstrate that TrOMR outperforms current OMR methods, especially in real-world scenarios. We also develop a TrOMR system and build a camera scene dataset for full-page music scores in real-world. The code and datasets will be made available for reproducibility.

{{</citation>}}


### (50/87) Accelerated materials language processing enabled by GPT (Jaewoong Choi et al., 2023)

{{<citation>}}

Jaewoong Choi, Byungju Lee. (2023)  
**Accelerated materials language processing enabled by GPT**  

---
Primary Category: cs.CL  
Categories: cond-mat-mtrl-sci, cs-CL, cs.CL  
Keywords: GPT, NER, QA  
[Paper Link](http://arxiv.org/abs/2308.09354v1)  

---


**ABSTRACT**  
Materials language processing (MLP) is one of the key facilitators of materials science research, as it enables the extraction of structured information from massive materials science literature. Prior works suggested high-performance MLP models for text classification, named entity recognition (NER), and extractive question answering (QA), which require complex model architecture, exhaustive fine-tuning and a large number of human-labelled datasets. In this study, we develop generative pretrained transformer (GPT)-enabled pipelines where the complex architectures of prior MLP models are replaced with strategic designs of prompt engineering. First, we develop a GPT-enabled document classification method for screening relevant documents, achieving comparable accuracy and reliability compared to prior models, with only small dataset. Secondly, for NER task, we design an entity-centric prompts, and learning few-shot of them improved the performance on most of entities in three open datasets. Finally, we develop an GPT-enabled extractive QA model, which provides improved performance and shows the possibility of automatically correcting annotations. While our findings confirm the potential of GPT-enabled MLP models as well as their value in terms of reliability and practicability, our scientific methods and systematic approach are applicable to any materials science domain to accelerate the information extraction of scientific literature.

{{</citation>}}


### (51/87) Document Automation Architectures: Updated Survey in Light of Large Language Models (Mohammad Ahmadi Achachlouei et al., 2023)

{{<citation>}}

Mohammad Ahmadi Achachlouei, Omkar Patil, Tarun Joshi, Vijayan N. Nair. (2023)  
**Document Automation Architectures: Updated Survey in Light of Large Language Models**  

---
Primary Category: cs.CL  
Categories: 68T50, I-7-0; I-2-7; I-2-4, cs-CL, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2308.09341v1)  

---


**ABSTRACT**  
This paper surveys the current state of the art in document automation (DA). The objective of DA is to reduce the manual effort during the generation of documents by automatically creating and integrating input from different sources and assembling documents conforming to defined templates. There have been reviews of commercial solutions of DA, particularly in the legal domain, but to date there has been no comprehensive review of the academic research on DA architectures and technologies. The current survey of DA reviews the academic literature and provides a clearer definition and characterization of DA and its features, identifies state-of-the-art DA architectures and technologies in academic research, and provides ideas that can lead to new research opportunities within the DA field in light of recent advances in generative AI and large language models.

{{</citation>}}


### (52/87) KESDT: knowledge enhanced shallow and deep Transformer for detecting adverse drug reactions (Yunzhi Qiu et al., 2023)

{{<citation>}}

Yunzhi Qiu, Xiaokun Zhang, Weiwei Wang, Tongxuan Zhang, Bo Xu, Hongfei Lin. (2023)  
**KESDT: knowledge enhanced shallow and deep Transformer for detecting adverse drug reactions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Transformer, Twitter  
[Paper Link](http://arxiv.org/abs/2308.09329v1)  

---


**ABSTRACT**  
Adverse drug reaction (ADR) detection is an essential task in the medical field, as ADRs have a gravely detrimental impact on patients' health and the healthcare system. Due to a large number of people sharing information on social media platforms, an increasing number of efforts focus on social media data to carry out effective ADR detection. Despite having achieved impressive performance, the existing methods of ADR detection still suffer from three main challenges. Firstly, researchers have consistently ignored the interaction between domain keywords and other words in the sentence. Secondly, social media datasets suffer from the challenges of low annotated data. Thirdly, the issue of sample imbalance is commonly observed in social media datasets. To solve these challenges, we propose the Knowledge Enhanced Shallow and Deep Transformer(KESDT) model for ADR detection. Specifically, to cope with the first issue, we incorporate the domain keywords into the Transformer model through a shallow fusion manner, which enables the model to fully exploit the interactive relationships between domain keywords and other words in the sentence. To overcome the low annotated data, we integrate the synonym sets into the Transformer model through a deep fusion manner, which expands the size of the samples. To mitigate the impact of sample imbalance, we replace the standard cross entropy loss function with the focal loss function for effective model training. We conduct extensive experiments on three public datasets including TwiMed, Twitter, and CADEC. The proposed KESDT outperforms state-of-the-art baselines on F1 values, with relative improvements of 4.87%, 47.83%, and 5.73% respectively, which demonstrates the effectiveness of our proposed KESDT.

{{</citation>}}


### (53/87) Conversational Ontology Alignment with ChatGPT (Sanaz Saki Norouzi et al., 2023)

{{<citation>}}

Sanaz Saki Norouzi, Mohammad Saeid Mahdavinejad, Pascal Hitzler. (2023)  
**Conversational Ontology Alignment with ChatGPT**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2308.09217v1)  

---


**ABSTRACT**  
This study evaluates the applicability and efficiency of ChatGPT for ontology alignment using a naive approach. ChatGPT's output is compared to the results of the Ontology Alignment Evaluation Initiative 2022 campaign using conference track ontologies. This comparison is intended to provide insights into the capabilities of a conversational large language model when used in a naive way for ontology matching, and to investigate the potential advantages and disadvantages of this approach.

{{</citation>}}


## cs.SI (2)



### (54/87) A Potts model approach to unsupervised graph clustering with Graph Neural Networks (Co Tran et al., 2023)

{{<citation>}}

Co Tran, Mo Badawy, Tyler McDonnell. (2023)  
**A Potts model approach to unsupervised graph clustering with Graph Neural Networks**  

---
Primary Category: cs.SI  
Categories: cs-CE, cs-SI, cs.SI  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.09644v1)  

---


**ABSTRACT**  
Numerous approaches have been explored for graph clustering, including those which optimize a global criteria such as modularity. More recently, Graph Neural Networks (GNNs), which have produced state-of-the-art results in graph analysis tasks such as node classification and link prediction, have been applied for unsupervised graph clustering using these modularity-based metrics. Modularity, though robust for many practical applications, suffers from the resolution limit problem, in which optimization may fail to identify clusters smaller than a scale that is dependent on properties of the network. In this paper, we propose a new GNN framework which draws from the Potts model in physics to overcome this limitation. Experiments on a variety of real world datasets show that this model achieves state-of-the-art clustering results.

{{</citation>}}


### (55/87) Profile Update: The Effects of Identity Disclosure on Network Connections and Language (Minje Choi et al., 2023)

{{<citation>}}

Minje Choi, Daniel M. Romero, David Jurgens. (2023)  
**Profile Update: The Effects of Identity Disclosure on Network Connections and Language**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: NLP, Twitter  
[Paper Link](http://arxiv.org/abs/2308.09270v1)  

---


**ABSTRACT**  
Our social identities determine how we interact and engage with the world surrounding us. In online settings, individuals can make these identities explicit by including them in their public biography, possibly signaling a change to what is important to them and how they should be viewed. Here, we perform the first large-scale study on Twitter that examines behavioral changes following identity signal addition on Twitter profiles. Combining social networks with NLP and quasi-experimental analyses, we discover that after disclosing an identity on their profiles, users (1) generate more tweets containing language that aligns with their identity and (2) connect more to same-identity users. We also examine whether adding an identity signal increases the number of offensive replies and find that (3) the combined effect of disclosing identity via both tweets and profiles is associated with a reduced number of offensive replies from others.

{{</citation>}}


## cs.LG (13)



### (56/87) Development of a Neural Network-based Method for Improved Imputation of Missing Values in Time Series Data by Repurposing DataWig (Daniel Zhang, 2023)

{{<citation>}}

Daniel Zhang. (2023)  
**Development of a Neural Network-based Method for Improved Imputation of Missing Values in Time Series Data by Repurposing DataWig**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Time Series  
[Paper Link](http://arxiv.org/abs/2308.09635v1)  

---


**ABSTRACT**  
Time series data are observations collected over time intervals. Successful analysis of time series data captures patterns such as trends, cyclicity and irregularity, which are crucial for decision making in research, business, and governance. However, missing values in time series data occur often and present obstacles to successful analysis, thus they need to be filled with alternative values, a process called imputation. Although various approaches have been attempted for robust imputation of time series data, even the most advanced methods still face challenges including limited scalability, poor capacity to handle heterogeneous data types and inflexibility due to requiring strong assumptions of data missing mechanisms. Moreover, the imputation accuracy of these methods still has room for improvement. In this study, I developed tsDataWig (time-series DataWig) by modifying DataWig, a neural network-based method that possesses the capacity to process large datasets and heterogeneous data types but was designed for non-time series data imputation. Unlike the original DataWig, tsDataWig can directly handle values of time variables and impute missing values in complex time series datasets. Using one simulated and three different complex real-world time series datasets, I demonstrated that tsDataWig outperforms the original DataWig and the current state-of-the-art methods for time series data imputation and potentially has broad application due to not requiring strong assumptions of data missing mechanisms. This study provides a valuable solution for robustly imputing missing values in challenging time series datasets, which often contain millions of samples, high dimensional variables, and heterogeneous data types.

{{</citation>}}


### (57/87) Learning Computational Efficient Bots with Costly Features (Anthony Kobanda et al., 2023)

{{<citation>}}

Anthony Kobanda, Valliappan C. A., Joshua Romoff, Ludovic Denoyer. (2023)  
**Learning Computational Efficient Bots with Costly Features**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2308.09629v1)  

---


**ABSTRACT**  
Deep reinforcement learning (DRL) techniques have become increasingly used in various fields for decision-making processes. However, a challenge that often arises is the trade-off between both the computational efficiency of the decision-making process and the ability of the learned agent to solve a particular task. This is particularly critical in real-time settings such as video games where the agent needs to take relevant decisions at a very high frequency, with a very limited inference time.   In this work, we propose a generic offline learning approach where the computation cost of the input features is taken into account. We derive the Budgeted Decision Transformer as an extension of the Decision Transformer that incorporates cost constraints to limit its cost at inference. As a result, the model can dynamically choose the best input features at each timestep. We demonstrate the effectiveness of our method on several tasks, including D4RL benchmarks and complex 3D environments similar to those found in video games, and show that it can achieve similar performance while using significantly fewer computational resources compared to classical approaches.

{{</citation>}}


### (58/87) Disparity, Inequality, and Accuracy Tradeoffs in Graph Neural Networks for Node Classification (Arpit Merchant et al., 2023)

{{<citation>}}

Arpit Merchant, Carlos Castillo. (2023)  
**Disparity, Inequality, and Accuracy Tradeoffs in Graph Neural Networks for Node Classification**  

---
Primary Category: cs.LG  
Categories: cs-CY, cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.09596v1)  

---


**ABSTRACT**  
Graph neural networks (GNNs) are increasingly used in critical human applications for predicting node labels in attributed graphs. Their ability to aggregate features from nodes' neighbors for accurate classification also has the capacity to exacerbate existing biases in data or to introduce new ones towards members from protected demographic groups. Thus, it is imperative to quantify how GNNs may be biased and to what extent their harmful effects may be mitigated. To this end, we propose two new GNN-agnostic interventions namely, (i) PFR-AX which decreases the separability between nodes in protected and non-protected groups, and (ii) PostProcess which updates model predictions based on a blackbox policy to minimize differences between error rates across demographic groups. Through a large set of experiments on four datasets, we frame the efficacies of our approaches (and three variants) in terms of their algorithmic fairness-accuracy tradeoff and benchmark our results against three strong baseline interventions on three state-of-the-art GNN models. Our results show that no single intervention offers a universally optimal tradeoff, but PFR-AX and PostProcess provide granular control and improve model confidence when correctly predicting positive outcomes for nodes in protected groups.

{{</citation>}}


### (59/87) Adapt Your Teacher: Improving Knowledge Distillation for Exemplar-free Continual Learning (Filip Szatkowski et al., 2023)

{{<citation>}}

Filip Szatkowski, Mateusz Pyla, Marcin Przewięźlikowski, Sebastian Cygert, Bartłomiej Twardowski, Tomasz Trzciński. (2023)  
**Adapt Your Teacher: Improving Knowledge Distillation for Exemplar-free Continual Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2308.09544v1)  

---


**ABSTRACT**  
In this work, we investigate exemplar-free class incremental learning (CIL) with knowledge distillation (KD) as a regularization strategy, aiming to prevent forgetting. KD-based methods are successfully used in CIL, but they often struggle to regularize the model without access to exemplars of the training data from previous tasks. Our analysis reveals that this issue originates from substantial representation shifts in the teacher network when dealing with out-of-distribution data. This causes large errors in the KD loss component, leading to performance degradation in CIL. Inspired by recent test-time adaptation methods, we introduce Teacher Adaptation (TA), a method that concurrently updates the teacher and the main model during incremental training. Our method seamlessly integrates with KD-based CIL approaches and allows for consistent enhancement of their performance across multiple exemplar-free CIL benchmarks.

{{</citation>}}


### (60/87) Transitivity-Preserving Graph Representation Learning for Bridging Local Connectivity and Role-based Similarity (Van Thuy Hoang et al., 2023)

{{<citation>}}

Van Thuy Hoang, O-Joun Lee. (2023)  
**Transitivity-Preserving Graph Representation Learning for Bridging Local Connectivity and Role-based Similarity**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Representation Learning, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09517v1)  

---


**ABSTRACT**  
Graph representation learning (GRL) methods, such as graph neural networks and graph transformer models, have been successfully used to analyze graph-structured data, mainly focusing on node classification and link prediction tasks. However, the existing studies mostly only consider local connectivity while ignoring long-range connectivity and the roles of nodes. In this paper, we propose Unified Graph Transformer Networks (UGT) that effectively integrate local and global structural information into fixed-length vector representations. First, UGT learns local structure by identifying the local substructures and aggregating features of the $k$-hop neighborhoods of each node. Second, we construct virtual edges, bridging distant nodes with structural similarity to capture the long-range dependencies. Third, UGT learns unified representations through self-attention, encoding structural distance and $p$-step transition probability between node pairs. Furthermore, we propose a self-supervised learning task that effectively learns transition probability to fuse local and global structural features, which could then be transferred to other downstream tasks. Experimental results on real-world benchmark datasets over various downstream tasks showed that UGT significantly outperformed baselines that consist of state-of-the-art models. In addition, UGT reaches the expressive power of the third-order Weisfeiler-Lehman isomorphism test (3d-WL) in distinguishing non-isomorphic graph pairs. The source code is available at https://github.com/NSLab-CUK/Unified-Graph-Transformer.

{{</citation>}}


### (61/87) Bridged-GNN: Knowledge Bridge Learning for Effective Knowledge Transfer (Wendong Bi et al., 2023)

{{<citation>}}

Wendong Bi, Xueqi Cheng, Bingbing Xu, Xiaoqian Sun, Li Xu, Huawei Shen. (2023)  
**Bridged-GNN: Knowledge Bridge Learning for Effective Knowledge Transfer**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.09499v1)  

---


**ABSTRACT**  
The data-hungry problem, characterized by insufficiency and low-quality of data, poses obstacles for deep learning models. Transfer learning has been a feasible way to transfer knowledge from high-quality external data of source domains to limited data of target domains, which follows a domain-level knowledge transfer to learn a shared posterior distribution. However, they are usually built on strong assumptions, e.g., the domain invariant posterior distribution, which is usually unsatisfied and may introduce noises, resulting in poor generalization ability on target domains. Inspired by Graph Neural Networks (GNNs) that aggregate information from neighboring nodes, we redefine the paradigm as learning a knowledge-enhanced posterior distribution for target domains, namely Knowledge Bridge Learning (KBL). KBL first learns the scope of knowledge transfer by constructing a Bridged-Graph that connects knowledgeable samples to each target sample and then performs sample-wise knowledge transfer via GNNs.KBL is free from strong assumptions and is robust to noises in the source data. Guided by KBL, we propose the Bridged-GNN} including an Adaptive Knowledge Retrieval module to build Bridged-Graph and a Graph Knowledge Transfer module. Comprehensive experiments on both un-relational and relational data-hungry scenarios demonstrate the significant improvements of Bridged-GNN compared with SOTA methods

{{</citation>}}


### (62/87) Balancing Transparency and Risk: The Security and Privacy Risks of Open-Source Machine Learning Models (Dominik Hintersdorf et al., 2023)

{{<citation>}}

Dominik Hintersdorf, Lukas Struppek, Kristian Kersting. (2023)  
**Balancing Transparency and Risk: The Security and Privacy Risks of Open-Source Machine Learning Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CR, cs-CY, cs-LG, cs.LG  
Keywords: AI, Security  
[Paper Link](http://arxiv.org/abs/2308.09490v1)  

---


**ABSTRACT**  
The field of artificial intelligence (AI) has experienced remarkable progress in recent years, driven by the widespread adoption of open-source machine learning models in both research and industry. Considering the resource-intensive nature of training on vast datasets, many applications opt for models that have already been trained. Hence, a small number of key players undertake the responsibility of training and publicly releasing large pre-trained models, providing a crucial foundation for a wide range of applications. However, the adoption of these open-source models carries inherent privacy and security risks that are often overlooked. To provide a concrete example, an inconspicuous model may conceal hidden functionalities that, when triggered by specific input patterns, can manipulate the behavior of the system, such as instructing self-driving cars to ignore the presence of other vehicles. The implications of successful privacy and security attacks encompass a broad spectrum, ranging from relatively minor damage like service interruptions to highly alarming scenarios, including physical harm or the exposure of sensitive user data. In this work, we present a comprehensive overview of common privacy and security threats associated with the use of open-source models. By raising awareness of these dangers, we strive to promote the responsible and secure use of AI systems.

{{</citation>}}


### (63/87) Data augmentation and explainability for bias discovery and mitigation in deep learning (Agnieszka Mikołajczyk-Bareła, 2023)

{{<citation>}}

Agnieszka Mikołajczyk-Bareła. (2023)  
**Data augmentation and explainability for bias discovery and mitigation in deep learning**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: AI, Augmentation, Bias, Style Transfer  
[Paper Link](http://arxiv.org/abs/2308.09464v1)  

---


**ABSTRACT**  
This dissertation explores the impact of bias in deep neural networks and presents methods for reducing its influence on model performance. The first part begins by categorizing and describing potential sources of bias and errors in data and models, with a particular focus on bias in machine learning pipelines. The next chapter outlines a taxonomy and methods of Explainable AI as a way to justify predictions and control and improve the model. Then, as an example of a laborious manual data inspection and bias discovery process, a skin lesion dataset is manually examined. A Global Explanation for the Bias Identification method is proposed as an alternative semi-automatic approach to manual data exploration for discovering potential biases in data. Relevant numerical methods and metrics are discussed for assessing the effects of the identified biases on the model. Whereas identifying errors and bias is critical, improving the model and reducing the number of flaws in the future is an absolute priority. Hence, the second part of the thesis focuses on mitigating the influence of bias on ML models. Three approaches are proposed and discussed: Style Transfer Data Augmentation, Targeted Data Augmentations, and Attribution Feedback. Style Transfer Data Augmentation aims to address shape and texture bias by merging a style of a malignant lesion with a conflicting shape of a benign one. Targeted Data Augmentations randomly insert possible biases into all images in the dataset during the training, as a way to make the process random and, thus, destroy spurious correlations. Lastly, Attribution Feedback is used to fine-tune the model to improve its accuracy by eliminating obvious mistakes and teaching it to ignore insignificant input parts via an attribution loss. The goal of these approaches is to reduce the influence of bias on machine learning models, rather than eliminate it entirely.

{{</citation>}}


### (64/87) An Efficient 1 Iteration Learning Algorithm for Gaussian Mixture Model And Gaussian Mixture Embedding For Neural Network (Weiguo Lu et al., 2023)

{{<citation>}}

Weiguo Lu, Xuan Wu, Deng Ding, Gangnan Yuan. (2023)  
**An Efficient 1 Iteration Learning Algorithm for Gaussian Mixture Model And Gaussian Mixture Embedding For Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2308.09444v1)  

---


**ABSTRACT**  
We propose an Gaussian Mixture Model (GMM) learning algorithm, based on our previous work of GMM expansion idea. The new algorithm brings more robustness and simplicity than classic Expectation Maximization (EM) algorithm. It also improves the accuracy and only take 1 iteration for learning. We theoretically proof that this new algorithm is guarantee to converge regardless the parameters initialisation. We compare our GMM expansion method with classic probability layers in neural network leads to demonstrably better capability to overcome data uncertainty and inverse problem. Finally, we test GMM based generator which shows a potential to build further application that able to utilized distribution random sampling for stochastic variation as well as variation control.

{{</citation>}}


### (65/87) From Hope to Safety: Unlearning Biases of Deep Models by Enforcing the Right Reasons in Latent Space (Maximilian Dreyer et al., 2023)

{{<citation>}}

Maximilian Dreyer, Frederik Pahde, Christopher J. Anders, Wojciech Samek, Sebastian Lapuschkin. (2023)  
**From Hope to Safety: Unlearning Biases of Deep Models by Enforcing the Right Reasons in Latent Space**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-CY, cs-LG, cs.LG  
Keywords: Bias, ImageNet  
[Paper Link](http://arxiv.org/abs/2308.09437v1)  

---


**ABSTRACT**  
Deep Neural Networks are prone to learning spurious correlations embedded in the training data, leading to potentially biased predictions. This poses risks when deploying these models for high-stake decision-making, such as in medical applications. Current methods for post-hoc model correction either require input-level annotations, which are only possible for spatially localized biases, or augment the latent feature space, thereby hoping to enforce the right reasons. We present a novel method ensuring the right reasons on the concept level by reducing the model's sensitivity towards biases through the gradient. When modeling biases via Concept Activation Vectors, we highlight the importance of choosing robust directions, as traditional regression-based approaches such as Support Vector Machines tend to result in diverging directions. We effectively mitigate biases in controlled and real-world settings on the ISIC, Bone Age, ImageNet and CelebA datasets using VGG, ResNet and EfficientNet architectures.

{{</citation>}}


### (66/87) Deciphering knee osteoarthritis diagnostic features with explainable artificial intelligence: A systematic review (Yun Xin Teoh et al., 2023)

{{<citation>}}

Yun Xin Teoh, Alice Othmani, Siew Li Goh, Juliana Usman, Khin Wee Lai. (2023)  
**Deciphering knee osteoarthritis diagnostic features with explainable artificial intelligence: A systematic review**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09380v1)  

---


**ABSTRACT**  
Existing artificial intelligence (AI) models for diagnosing knee osteoarthritis (OA) have faced criticism for their lack of transparency and interpretability, despite achieving medical-expert-like performance. This opacity makes them challenging to trust in clinical practice. Recently, explainable artificial intelligence (XAI) has emerged as a specialized technique that can provide confidence in the model's prediction by revealing how the prediction is derived, thus promoting the use of AI systems in healthcare. This paper presents the first survey of XAI techniques used for knee OA diagnosis. The XAI techniques are discussed from two perspectives: data interpretability and model interpretability. The aim of this paper is to provide valuable insights into XAI's potential towards a more reliable knee OA diagnosis approach and encourage its adoption in clinical practice.

{{</citation>}}


### (67/87) CARLA: A Self-supervised Contrastive Representation Learning Approach for Time Series Anomaly Detection (Zahra Zamanzadeh Darban et al., 2023)

{{<citation>}}

Zahra Zamanzadeh Darban, Geoffrey I. Webb, Shirui Pan, Mahsa Salehi. (2023)  
**CARLA: A Self-supervised Contrastive Representation Learning Approach for Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-NE, cs.LG  
Keywords: Anomaly Detection, Representation Learning, Time Series  
[Paper Link](http://arxiv.org/abs/2308.09296v1)  

---


**ABSTRACT**  
We introduce a Self-supervised Contrastive Representation Learning Approach for Time Series Anomaly Detection (CARLA), an innovative end-to-end self-supervised framework carefully developed to identify anomalous patterns in both univariate and multivariate time series data. By taking advantage of contrastive representation learning, We introduce an innovative end-to-end self-supervised deep learning framework carefully developed to identify anomalous patterns in both univariate and multivariate time series data. By taking advantage of contrastive representation learning, CARLA effectively generates robust representations for time series windows. It achieves this by 1) learning similar representations for temporally close windows and dissimilar representations for windows and their equivalent anomalous windows and 2) employing a self-supervised approach to classify normal/anomalous representations of windows based on their nearest/furthest neighbours in the representation space. Most of the existing models focus on learning normal behaviour. The normal boundary is often tightly defined, which can result in slight deviations being classified as anomalies, resulting in a high false positive rate and limited ability to generalise normal patterns. CARLA's contrastive learning methodology promotes the production of highly consistent and discriminative predictions, thereby empowering us to adeptly address the inherent challenges associated with anomaly detection in time series data. Through extensive experimentation on 7 standard real-world time series anomaly detection benchmark datasets, CARLA demonstrates F1 and AU-PR superior to existing state-of-the-art results. Our research highlights the immense potential of contrastive representation learning in advancing the field of time series anomaly detection, thus paving the way for novel applications and in-depth exploration in this domain.

{{</citation>}}


### (68/87) Distribution shift mitigation at test time with performance guarantees (Rui Ding et al., 2023)

{{<citation>}}

Rui Ding, Jielong Yang, Feng Ji, Xionghu Zhong, Linbo Xie. (2023)  
**Distribution shift mitigation at test time with performance guarantees**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2308.09259v1)  

---


**ABSTRACT**  
Due to inappropriate sample selection and limited training data, a distribution shift often exists between the training and test sets. This shift can adversely affect the test performance of Graph Neural Networks (GNNs). Existing approaches mitigate this issue by either enhancing the robustness of GNNs to distribution shift or reducing the shift itself. However, both approaches necessitate retraining the model, which becomes unfeasible when the model structure and parameters are inaccessible. To address this challenge, we propose FR-GNN, a general framework for GNNs to conduct feature reconstruction. FRGNN constructs a mapping relationship between the output and input of a well-trained GNN to obtain class representative embeddings and then uses these embeddings to reconstruct the features of labeled nodes. These reconstructed features are then incorporated into the message passing mechanism of GNNs to influence the predictions of unlabeled nodes at test time. Notably, the reconstructed node features can be directly utilized for testing the well-trained model, effectively reducing the distribution shift and leading to improved test performance. This remarkable achievement is attained without any modifications to the model structure or parameters. We provide theoretical guarantees for the effectiveness of our framework. Furthermore, we conduct comprehensive experiments on various public datasets. The experimental results demonstrate the superior performance of FRGNN in comparison to mainstream methods.

{{</citation>}}


## cs.CR (6)



### (69/87) An AI-Driven VM Threat Prediction Model for Multi-Risks Analysis-Based Cloud Cybersecurity (Deepika Saxena et al., 2023)

{{<citation>}}

Deepika Saxena, Ishu Gupta, Rishabh Gupta, Ashutosh Kumar Singh, Xiaoqing Wen. (2023)  
**An AI-Driven VM Threat Prediction Model for Multi-Risks Analysis-Based Cloud Cybersecurity**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-DC, cs.CR  
Keywords: AI, Google  
[Paper Link](http://arxiv.org/abs/2308.09578v1)  

---


**ABSTRACT**  
Cloud virtualization technology, ingrained with physical resource sharing, prompts cybersecurity threats on users' virtual machines (VM)s due to the presence of inevitable vulnerabilities on the offsite servers. Contrary to the existing works which concentrated on reducing resource sharing and encryption and decryption of data before transfer for improving cybersecurity which raises computational cost overhead, the proposed model operates diversely for efficiently serving the same purpose. This paper proposes a novel Multiple Risks Analysis based VM Threat Prediction Model (MR-TPM) to secure computational data and minimize adversary breaches by proactively estimating the VMs threats. It considers multiple cybersecurity risk factors associated with the configuration and management of VMs, along with analysis of users' behaviour. All these threat factors are quantified for the generation of respective risk score values and fed as input into a machine learning based classifier to estimate the probability of threat for each VM. The performance of MR-TPM is evaluated using benchmark Google Cluster and OpenNebula VM threat traces. The experimental results demonstrate that the proposed model efficiently computes the cybersecurity risks and learns the VM threat patterns from historical and live data samples. The deployment of MR-TPM with existing VM allocation policies reduces cybersecurity threats up to 88.9%.

{{</citation>}}


### (70/87) Intrusion Detection based on Federated Learning: a systematic review (Jose L. Hernandez-Ramos et al., 2023)

{{<citation>}}

Jose L. Hernandez-Ramos, Georgios Karopoulos, Efstratios Chatzoglou, Vasileios Kouliaridis, Enrique Marmol, Aurora Gonzalez-Vidal, Georgios Kambourakis. (2023)  
**Intrusion Detection based on Federated Learning: a systematic review**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: AI, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2308.09522v1)  

---


**ABSTRACT**  
The evolution of cybersecurity is undoubtedly associated and intertwined with the development and improvement of artificial intelligence (AI). As a key tool for realizing more cybersecure ecosystems, Intrusion Detection Systems (IDSs) have evolved tremendously in recent years by integrating machine learning (ML) techniques for the detection of increasingly sophisticated cybersecurity attacks hidden in big data. However, these approaches have traditionally been based on centralized learning architectures, in which data from end nodes are shared with data centers for analysis. Recently, the application of federated learning (FL) in this context has attracted great interest to come up with collaborative intrusion detection approaches where data does not need to be shared. Due to the recent rise of this field, this work presents a complete, contemporary taxonomy for FL-enabled IDS approaches that stems from a comprehensive survey of the literature in the time span from 2018 to 2022. Precisely, our discussion includes an analysis of the main ML models, datasets, aggregation functions, as well as implementation libraries, which are employed by the proposed FL-enabled IDS approaches. On top of everything else, we provide a critical view of the current state of the research around this topic, and describe the main challenges and future directions based on the analysis of the literature and our own experience in this area.

{{</citation>}}


### (71/87) Proceedings of the 2nd International Workshop on Adaptive Cyber Defense (Marco Carvalho et al., 2023)

{{<citation>}}

Marco Carvalho, Damian Marriott, Mark Bilinski, Ahmad Ridley. (2023)  
**Proceedings of the 2nd International Workshop on Adaptive Cyber Defense**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09520v1)  

---


**ABSTRACT**  
The 2nd International Workshop on Adaptive Cyber Defense was held at the Florida Institute of Technology, Florida. This workshop was organized to share research that explores unique applications of Artificial Intelligence (AI) and Machine Learning (ML) as foundational capabilities for the pursuit of adaptive cyber defense. The cyber domain cannot currently be reliably and effectively defended without extensive reliance on human experts. Skilled cyber defenders are in short supply and often cannot respond fast enough to cyber threats.   Building on recent advances in AI and ML the Cyber defense research community has been motivated to develop new dynamic and sustainable defenses through the adoption of AI and ML techniques to cyber settings. Bridging critical gaps between AI and Cyber researchers and practitioners can accelerate efforts to create semi-autonomous cyber defenses that can learn to recognize and respond to cyber attacks or discover and mitigate weaknesses in cooperation with other cyber operation systems and human experts. Furthermore, these defenses are expected to be adaptive and able to evolve over time to thwart changes in attacker behavior, changes in the system health and readiness, and natural shifts in user behavior over time.   The workshop was comprised of invited keynote talks, technical presentations and a panel discussion about how AI/ML can enable autonomous mitigation of current and future cyber attacks. Workshop submissions were peer reviewed by a panel of domain experts with a proceedings consisting of six technical articles exploring challenging problems of critical importance to national and global security. Participation in this workshop offered new opportunities to stimulate research and innovation in the emerging domain of adaptive and autonomous cyber defense.

{{</citation>}}


### (72/87) Poison Dart Frog: A Clean-Label Attack with Low Poisoning Rate and High Attack Success Rate in the Absence of Training Data (Binhao Ma et al., 2023)

{{<citation>}}

Binhao Ma, Jiahui Wang, Dejun Wang, Bo Meng. (2023)  
**Poison Dart Frog: A Clean-Label Attack with Low Poisoning Rate and High Attack Success Rate in the Absence of Training Data**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2308.09487v1)  

---


**ABSTRACT**  
To successfully launch backdoor attacks, injected data needs to be correctly labeled; otherwise, they can be easily detected by even basic data filters. Hence, the concept of clean-label attacks was introduced, which is more dangerous as it doesn't require changing the labels of injected data. To the best of our knowledge, the existing clean-label backdoor attacks largely relies on an understanding of the entire training set or a portion of it. However, in practice, it is very difficult for attackers to have it because of training datasets often collected from multiple independent sources. Unlike all current clean-label attacks, we propose a novel clean label method called 'Poison Dart Frog'. Poison Dart Frog does not require access to any training data; it only necessitates knowledge of the target class for the attack, such as 'frog'. On CIFAR10, Tiny-ImageNet, and TSRD, with a mere 0.1\%, 0.025\%, and 0.4\% poisoning rate of the training set size, respectively, Poison Dart Frog achieves a high Attack Success Rate compared to LC, HTBA, BadNets, and Blend. Furthermore, compared to the state-of-the-art attack, NARCISSUS, Poison Dart Frog achieves similar attack success rates without any training data. Finally, we demonstrate that four typical backdoor defense algorithms struggle to counter Poison Dart Frog.

{{</citation>}}


### (73/87) Attacking logo-based phishing website detectors with adversarial perturbations (Jehyun Lee et al., 2023)

{{<citation>}}

Jehyun Lee, Zhe Xin, Melanie Ng Pei See, Kanav Sabharwal, Giovanni Apruzzese, Dinil Mon Divakaran. (2023)  
**Attacking logo-based phishing website detectors with adversarial perturbations**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Computer Vision  
[Paper Link](http://arxiv.org/abs/2308.09392v1)  

---


**ABSTRACT**  
Recent times have witnessed the rise of anti-phishing schemes powered by deep learning (DL). In particular, logo-based phishing detectors rely on DL models from Computer Vision to identify logos of well-known brands on webpages, to detect malicious webpages that imitate a given brand. For instance, Siamese networks have demonstrated notable performance for these tasks, enabling the corresponding anti-phishing solutions to detect even "zero-day" phishing webpages. In this work, we take the next step of studying the robustness of logo-based phishing detectors against adversarial ML attacks. We propose a novel attack exploiting generative adversarial perturbations to craft "adversarial logos" that evade phishing detectors. We evaluate our attacks through: (i) experiments on datasets containing real logos, to evaluate the robustness of state-of-the-art phishing detectors; and (ii) user studies to gauge whether our adversarial logos can deceive human eyes. The results show that our proposed attack is capable of crafting perturbed logos subtle enough to evade various DL models-achieving an evasion rate of up to 95%. Moreover, users are not able to spot significant differences between generated adversarial logos and original ones.

{{</citation>}}


### (74/87) Blockchain-Based and Fuzzy Logic-Enabled False Data Discovery for the Intelligent Autonomous Vehicular System (Ziaur Rahman et al., 2023)

{{<citation>}}

Ziaur Rahman, Xun Yi, Ibrahim Khalil, Adnan Anwar, Shantanu Pal. (2023)  
**Blockchain-Based and Fuzzy Logic-Enabled False Data Discovery for the Intelligent Autonomous Vehicular System**  

---
Primary Category: cs.CR  
Categories: 11T71, 68T05, E-3-1; I-2-1, cs-CR, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09237v1)  

---


**ABSTRACT**  
Since the beginning of this decade, several incidents report that false data injection attacks targeting intelligent connected vehicles cause huge industrial damage and loss of lives. Data Theft, Flooding, Fuzzing, Hijacking, Malware Spoofing and Advanced Persistent Threats have been immensely growing attack that leads to end-user conflict by abolishing trust on autonomous vehicle. Looking after those sensitive data that contributes to measure the localisation factors of the vehicle, conventional centralised techniques can be misused to update the legitimate vehicular status maliciously. As investigated, the existing centralized false data detection approach based on state and likelihood estimation has a reprehensible trade-off in terms of accuracy, trust, cost, and efficiency. Blockchain with Fuzzy-logic Intelligence has shown its potential to solve localisation issues, trust and false data detection challenges encountered by today's autonomous vehicular system. The proposed Blockchain-based fuzzy solution demonstrates a novel false data detection and reputation preservation technique. The illustrated proposed model filters false and anomalous data based on the vehicles' rules and behaviours. Besides improving the detection accuracy and eliminating the single point of failure, the contributions include appropriating fuzzy AI functions within the Road-side Unit node before authorizing status data by a Blockchain network. Finally, thorough experimental evaluation validates the effectiveness of the proposed model.

{{</citation>}}


## cs.AI (4)



### (75/87) AI Hilbert: From Data and Background Knowledge to Automated Scientific Discovery (Ryan Cory-Wright et al., 2023)

{{<citation>}}

Ryan Cory-Wright, Bachir El Khadir, Cristina Cornelio, Sanjeeb Dash, Lior Horesh. (2023)  
**AI Hilbert: From Data and Background Knowledge to Automated Scientific Discovery**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-SC, cs.AI, math-OC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09474v1)  

---


**ABSTRACT**  
The discovery of scientific formulae that parsimoniously explain natural phenomena and align with existing background theory is a key goal in science. Historically, scientists have derived natural laws by manipulating equations based on existing knowledge, forming new equations, and verifying them experimentally. In recent years, data-driven scientific discovery has emerged as a viable competitor in settings with large amounts of experimental data. Unfortunately, data-driven methods often fail to discover valid laws when data is noisy or scarce. Accordingly, recent works combine regression and reasoning to eliminate formulae inconsistent with background theory. However, the problem of searching over the space of formulae consistent with background theory to find one that fits the data best is not well solved. We propose a solution to this problem when all axioms and scientific laws are expressible via polynomial equalities and inequalities and argue that our approach is widely applicable. We further model notions of minimal complexity using binary variables and logical constraints, solve polynomial optimization problems via mixed-integer linear or semidefinite optimization, and automatically prove the validity of our scientific discoveries via Positivestellensatz certificates. Remarkably, the optimization techniques leveraged in this paper allow our approach to run in polynomial time with fully correct background theory, or non-deterministic polynomial (NP) time with partially correct background theory. We experimentally demonstrate that some famous scientific laws, including Kepler's Third Law of Planetary Motion, the Hagen-Poiseuille Equation, and the Radiated Gravitational Wave Power equation, can be automatically derived from sets of partially correct background axioms.

{{</citation>}}


### (76/87) Preference-conditioned Pixel-based AI Agent For Game Testing (Sherif Abdelfattah et al., 2023)

{{<citation>}}

Sherif Abdelfattah, Adrian Brown, Pushi Zhang. (2023)  
**Preference-conditioned Pixel-based AI Agent For Game Testing**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-MM, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2308.09289v1)  

---


**ABSTRACT**  
The game industry is challenged to cope with increasing growth in demand and game complexity while maintaining acceptable quality standards for released games. Classic approaches solely depending on human efforts for quality assurance and game testing do not scale effectively in terms of time and cost. Game-testing AI agents that learn by interaction with the environment have the potential to mitigate these challenges with good scalability properties on time and costs. However, most recent work in this direction depends on game state information for the agent's state representation, which limits generalization across different game scenarios. Moreover, game test engineers usually prefer exploring a game in a specific style, such as exploring the golden path. However, current game testing AI agents do not provide an explicit way to satisfy such a preference. This paper addresses these limitations by proposing an agent design that mainly depends on pixel-based state observations while exploring the environment conditioned on a user's preference specified by demonstration trajectories. In addition, we propose an imitation learning method that couples self-supervised and supervised learning objectives to enhance the quality of imitation behaviors. Our agent significantly outperforms state-of-the-art pixel-based game testing agents over exploration coverage and test execution quality when evaluated on a complex open-world environment resembling many aspects of real AAA games.

{{</citation>}}


### (77/87) Enhancing Reasoning Capabilities of Large Language Models: A Graph-Based Verification Approach (Lang Cao, 2023)

{{<citation>}}

Lang Cao. (2023)  
**Enhancing Reasoning Capabilities of Large Language Models: A Graph-Based Verification Approach**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2308.09267v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have showcased impressive reasoning capabilities, particularly when guided by specifically designed prompts in complex reasoning tasks such as math word problems. These models typically solve tasks using a chain-of-thought approach, which not only bolsters their reasoning abilities but also provides valuable insights into their problem-solving process. However, there is still significant room for enhancing the reasoning abilities of LLMs. Some studies suggest that the integration of an LLM output verifier can boost reasoning accuracy without necessitating additional model training. In this paper, we follow these studies and introduce a novel graph-based method to further augment the reasoning capabilities of LLMs. We posit that multiple solutions to a reasoning task, generated by an LLM, can be represented as a reasoning graph due to the logical connections between intermediate steps from different reasoning paths. Therefore, we propose the Reasoning Graph Verifier (RGV) to analyze and verify the solutions generated by LLMs. By evaluating these graphs, models can yield more accurate and reliable results.Our experimental results show that our graph-based verification method not only significantly enhances the reasoning abilities of LLMs but also outperforms existing verifier methods in terms of improving these models' reasoning performance.

{{</citation>}}


### (78/87) Learning in Cooperative Multiagent Systems Using Cognitive and Machine Models (Thuy Ngoc Nguyen et al., 2023)

{{<citation>}}

Thuy Ngoc Nguyen, Duy Nhat Phan, Cleotilde Gonzalez. (2023)  
**Learning in Cooperative Multiagent Systems Using Cognitive and Machine Models**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-MA, cs.AI  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.09219v1)  

---


**ABSTRACT**  
Developing effective Multi-Agent Systems (MAS) is critical for many applications requiring collaboration and coordination with humans. Despite the rapid advance of Multi-Agent Deep Reinforcement Learning (MADRL) in cooperative MAS, one major challenge is the simultaneous learning and interaction of independent agents in dynamic environments in the presence of stochastic rewards. State-of-the-art MADRL models struggle to perform well in Coordinated Multi-agent Object Transportation Problems (CMOTPs), wherein agents must coordinate with each other and learn from stochastic rewards. In contrast, humans often learn rapidly to adapt to nonstationary environments that require coordination among people. In this paper, motivated by the demonstrated ability of cognitive models based on Instance-Based Learning Theory (IBLT) to capture human decisions in many dynamic decision making tasks, we propose three variants of Multi-Agent IBL models (MAIBL). The idea of these MAIBL algorithms is to combine the cognitive mechanisms of IBLT and the techniques of MADRL models to deal with coordination MAS in stochastic environments from the perspective of independent learners. We demonstrate that the MAIBL models exhibit faster learning and achieve better coordination in a dynamic CMOTP task with various settings of stochastic rewards compared to current MADRL models. We discuss the benefits of integrating cognitive insights into MADRL models.

{{</citation>}}


## cs.RO (3)



### (79/87) Integrating Expert Guidance for Efficient Learning of Safe Overtaking in Autonomous Driving Using Deep Reinforcement Learning (Jinxiong Lu et al., 2023)

{{<citation>}}

Jinxiong Lu, Gokhan Alcan, Ville Kyrki. (2023)  
**Integrating Expert Guidance for Efficient Learning of Safe Overtaking in Autonomous Driving Using Deep Reinforcement Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.09456v1)  

---


**ABSTRACT**  
Overtaking on two-lane roads is a great challenge for autonomous vehicles, as oncoming traffic appearing on the opposite lane may require the vehicle to change its decision and abort the overtaking. Deep reinforcement learning (DRL) has shown promise for difficult decision problems such as this, but it requires massive number of data, especially if the action space is continuous. This paper proposes to incorporate guidance from an expert system into DRL to increase its sample efficiency in the autonomous overtaking setting. The guidance system developed in this study is composed of constrained iterative LQR and PID controllers. The novelty lies in the incorporation of a fading guidance function, which gradually decreases the effect of the expert system, allowing the agent to initially learn an appropriate action swiftly and then improve beyond the performance of the expert system. This approach thus combines the strengths of traditional control engineering with the flexibility of learning systems, expanding the capabilities of the autonomous system. The proposed methodology for autonomous vehicle overtaking does not depend on a particular DRL algorithm and three state-of-the-art algorithms are used as baselines for evaluation. Simulation results show that incorporating expert system guidance improves state-of-the-art DRL algorithms greatly in both sample efficiency and driving safety.

{{</citation>}}


### (80/87) Robust Quadrupedal Locomotion via Risk-Averse Policy Learning (Jiyuan Shi et al., 2023)

{{<citation>}}

Jiyuan Shi, Chenjia Bai, Haoran He, Lei Han, Dong Wang, Bin Zhao, Xiu Li, Xuelong Li. (2023)  
**Robust Quadrupedal Locomotion via Risk-Averse Policy Learning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2308.09405v1)  

---


**ABSTRACT**  
The robustness of legged locomotion is crucial for quadrupedal robots in challenging terrains. Recently, Reinforcement Learning (RL) has shown promising results in legged locomotion and various methods try to integrate privileged distillation, scene modeling, and external sensors to improve the generalization and robustness of locomotion policies. However, these methods are hard to handle uncertain scenarios such as abrupt terrain changes or unexpected external forces. In this paper, we consider a novel risk-sensitive perspective to enhance the robustness of legged locomotion. Specifically, we employ a distributional value function learned by quantile regression to model the aleatoric uncertainty of environments, and perform risk-averse policy learning by optimizing the worst-case scenarios via a risk distortion measure. Extensive experiments in both simulation environments and a real Aliengo robot demonstrate that our method is efficient in handling various external disturbances, and the resulting policy exhibits improved robustness in harsh and uncertain situations in legged locomotion. Videos are available at https://risk-averse-locomotion.github.io/.

{{</citation>}}


### (81/87) Multi-Level Compositional Reasoning for Interactive Instruction Following (Suvaansh Bhambri et al., 2023)

{{<citation>}}

Suvaansh Bhambri, Byeonghwi Kim, Jonghyun Choi. (2023)  
**Multi-Level Compositional Reasoning for Interactive Instruction Following**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2308.09387v1)  

---


**ABSTRACT**  
Robotic agents performing domestic chores by natural language directives are required to master the complex job of navigating environment and interacting with objects in the environments. The tasks given to the agents are often composite thus are challenging as completing them require to reason about multiple subtasks, e.g., bring a cup of coffee. To address the challenge, we propose to divide and conquer it by breaking the task into multiple subgoals and attend to them individually for better navigation and interaction. We call it Multi-level Compositional Reasoning Agent (MCR-Agent). Specifically, we learn a three-level action policy. At the highest level, we infer a sequence of human-interpretable subgoals to be executed based on language instructions by a high-level policy composition controller. At the middle level, we discriminatively control the agent's navigation by a master policy by alternating between a navigation policy and various independent interaction policies. Finally, at the lowest level, we infer manipulation actions with the corresponding object masks using the appropriate interaction policy. Our approach not only generates human interpretable subgoals but also achieves 2.03% absolute gain to comparable state of the arts in the efficiency metric (PLWSR in unseen set) without using rule-based planning or a semantic spatial memory.

{{</citation>}}


## cs.SD (1)



### (82/87) Exploring Sampling Techniques for Generating Melodies with a Transformer Language Model (Mathias Rose Bjare et al., 2023)

{{<citation>}}

Mathias Rose Bjare, Stefan Lattner, Gerhard Widmer. (2023)  
**Exploring Sampling Techniques for Generating Melodies with a Transformer Language Model**  

---
Primary Category: cs.SD  
Categories: cs-CL, cs-SD, cs.SD, eess-AS  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09454v1)  

---


**ABSTRACT**  
Research in natural language processing has demonstrated that the quality of generations from trained autoregressive language models is significantly influenced by the used sampling strategy. In this study, we investigate the impact of different sampling techniques on musical qualities such as diversity and structure. To accomplish this, we train a high-capacity transformer model on a vast collection of highly-structured Irish folk melodies and analyze the musical qualities of the samples generated using distribution truncation sampling techniques. Specifically, we use nucleus sampling, the recently proposed "typical sampling", and conventional ancestral sampling. We evaluate the effect of these sampling strategies in two scenarios: optimal circumstances with a well-calibrated model and suboptimal circumstances where we systematically degrade the model's performance. We assess the generated samples using objective and subjective evaluations. We discover that probability truncation techniques may restrict diversity and structural patterns in optimal circumstances, but may also produce more musical samples in suboptimal circumstances.

{{</citation>}}


## cs.CE (1)



### (83/87) BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine (Yizhen Luo et al., 2023)

{{<citation>}}

Yizhen Luo, Jiahuan Zhang, Siqi Fan, Kai Yang, Yushuai Wu, Mu Qiao, Zaiqing Nie. (2023)  
**BioMedGPT: Open Multimodal Generative Pre-trained Transformer for BioMedicine**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs.CE  
Keywords: GPT, QA, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09442v1)  

---


**ABSTRACT**  
Foundation models (FMs) have exhibited remarkable performance across a wide range of downstream tasks in various domains. Nevertheless, general-purpose FMs often face challenges when confronted with domain-specific problems, due to their limited access to the proprietary training data in a particular domain. In life science, a large portion of biomedical knowledge is encapsulated within heterogeneous biological data, such as molecular structures, wet-lab experiment results, and knowledge bases, which exhibit significant modality gaps with human natural language. In this paper, we introduce BioMedGPT, an open multimodal generative pre-trained transformer (GPT) for biomedicine, to bridge the gap between biomedical data and natural language. BioMedGPT allows users to easily "communicate" with various biological modalities through free text, which is the first of its kind. BioMedGPT aligns different biological modalities with the text modality via a large generative language model, namely, BioMedGPT-LM. We publish BioMedGPT-10B, which unifies the feature spaces of molecules, proteins, and natural language via encoding and alignment. Through fine-tuning, BioMedGPT-10B outperforms or is on par with human and significantly larger general-purpose foundation models on the biomedical QA task. It also demonstrates promising performance in the molecule QA and protein QA tasks, which could greatly accelerate the discovery of new drugs and therapeutic targets. In addition, BioMedGPT-LM-7B is the first large generative language model based on Llama2 in the biomedical domain, therefore is commercial friendly. Both BioMedGPT-10B and BioMedGPT-LM-7B are open-sourced to the research community. We also publish the datasets that are meticulously curated for the alignment of multi-modalities, i.e., PubChemQA and UniProtQA. All the models, codes, and datasets are publicly available.

{{</citation>}}


## cs.IR (2)



### (84/87) Attention Calibration for Transformer-based Sequential Recommendation (Peilin Zhou et al., 2023)

{{<citation>}}

Peilin Zhou, Qichen Ye, Yueqi Xie, Jingqi Gao, Shoujin Wang, Jae Boum Kim, Chenyu You, Sunghun Kim. (2023)  
**Attention Calibration for Transformer-based Sequential Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: AI, Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2308.09419v1)  

---


**ABSTRACT**  
Transformer-based sequential recommendation (SR) has been booming in recent years, with the self-attention mechanism as its key component. Self-attention has been widely believed to be able to effectively select those informative and relevant items from a sequence of interacted items for next-item prediction via learning larger attention weights for these items. However, this may not always be true in reality. Our empirical analysis of some representative Transformer-based SR models reveals that it is not uncommon for large attention weights to be assigned to less relevant items, which can result in inaccurate recommendations. Through further in-depth analysis, we find two factors that may contribute to such inaccurate assignment of attention weights: sub-optimal position encoding and noisy input. To this end, in this paper, we aim to address this significant yet challenging gap in existing works. To be specific, we propose a simple yet effective framework called Attention Calibration for Transformer-based Sequential Recommendation (AC-TSR). In AC-TSR, a novel spatial calibrator and adversarial calibrator are designed respectively to directly calibrates those incorrectly assigned attention weights. The former is devised to explicitly capture the spatial relationships (i.e., order and distance) among items for more precise calculation of attention weights. The latter aims to redistribute the attention weights based on each item's contribution to the next-item prediction. AC-TSR is readily adaptable and can be seamlessly integrated into various existing transformer-based SR models. Extensive experimental results on four benchmark real-world datasets demonstrate the superiority of our proposed ACTSR via significant recommendation performance enhancements. The source code is available at https://github.com/AIM-SE/AC-TSR.

{{</citation>}}


### (85/87) Differentiable Retrieval Augmentation via Generative Language Modeling for E-commerce Query Intent Classification (Chenyu Zhao et al., 2023)

{{<citation>}}

Chenyu Zhao, Yunjiang Jiang, Yiming Qiu, Han Zhang, Wen-Yun Yang. (2023)  
**Differentiable Retrieval Augmentation via Generative Language Modeling for E-commerce Query Intent Classification**  

---
Primary Category: cs.IR  
Categories: H-3-3, cs-CL, cs-IR, cs.IR  
Keywords: Augmentation, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2308.09308v1)  

---


**ABSTRACT**  
Retrieval augmentation, which enhances downstream models by a knowledge retriever and an external corpus instead of by merely increasing the number of model parameters, has been successfully applied to many natural language processing (NLP) tasks such as text classification, question answering and so on. However, existing methods that separately or asynchronously train the retriever and downstream model mainly due to the non-differentiability between the two parts, usually lead to degraded performance compared to end-to-end joint training.

{{</citation>}}


## cs.SE (2)



### (86/87) AutoLog: A Log Sequence Synthesis Framework for Anomaly Detection (Yintong Huo et al., 2023)

{{<citation>}}

Yintong Huo, Yichen Li, Yuxin Su, Pinjia He, Zifan Xie, Michael R. Lyu. (2023)  
**AutoLog: A Log Sequence Synthesis Framework for Anomaly Detection**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2308.09324v1)  

---


**ABSTRACT**  
The rapid progress of modern computing systems has led to a growing interest in informative run-time logs. Various log-based anomaly detection techniques have been proposed to ensure software reliability. However, their implementation in the industry has been limited due to the lack of high-quality public log resources as training datasets.   While some log datasets are available for anomaly detection, they suffer from limitations in (1) comprehensiveness of log events; (2) scalability over diverse systems; and (3) flexibility of log utility. To address these limitations, we propose AutoLog, the first automated log generation methodology for anomaly detection. AutoLog uses program analysis to generate run-time log sequences without actually running the system. AutoLog starts with probing comprehensive logging statements associated with the call graphs of an application. Then, it constructs execution graphs for each method after pruning the call graphs to find log-related execution paths in a scalable manner. Finally, AutoLog propagates the anomaly label to each acquired execution path based on human knowledge. It generates flexible log sequences by walking along the log execution paths with controllable parameters. Experiments on 50 popular Java projects show that AutoLog acquires significantly more (9x-58x) log events than existing log datasets from the same system, and generates log messages much faster (15x) with a single machine than existing passive data collection approaches. We hope AutoLog can facilitate the benchmarking and adoption of automated log analysis techniques.

{{</citation>}}


### (87/87) Domain Adaptive Code Completion via Language Models and Decoupled Domain Databases (Ze Tang et al., 2023)

{{<citation>}}

Ze Tang, Jidong Ge, Shangqing Liu, Tingwei Zhu, Tongtong Xu, Liguo Huang, Bin Luo. (2023)  
**Domain Adaptive Code Completion via Language Models and Decoupled Domain Databases**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2308.09313v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have demonstrated remarkable performance in code completion. However, due to the lack of domain-specific knowledge, they may not be optimal in completing code that requires intensive domain knowledge for example completing the library names. Although there are several works that have confirmed the effectiveness of fine-tuning techniques to adapt language models for code completion in specific domains. They are limited by the need for constant fine-tuning of the model when the project is in constant iteration.   To address this limitation, in this paper, we propose $k$NM-LM, a retrieval-augmented language model (R-LM), that integrates domain knowledge into language models without fine-tuning. Different from previous techniques, our approach is able to automatically adapt to different language models and domains. Specifically, it utilizes the in-domain code to build the retrieval-based database decoupled from LM, and then combines it with LM through Bayesian inference to complete the code. The extensive experiments on the completion of intra-project and intra-scenario have confirmed that $k$NM-LM brings about appreciable enhancements when compared to CodeGPT and UnixCoder. A deep analysis of our tool including the responding speed, storage usage, specific type code completion, and API invocation completion has confirmed that $k$NM-LM provides satisfactory performance, which renders it highly appropriate for domain adaptive code completion. Furthermore, our approach operates without the requirement for direct access to the language model's parameters. As a result, it can seamlessly integrate with black-box code completion models, making it easy to integrate our approach as a plugin to further enhance the performance of these models.

{{</citation>}}
