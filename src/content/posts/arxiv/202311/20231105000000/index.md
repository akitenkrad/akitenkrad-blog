---
draft: false
title: "arXiv @ 2023.11.05"
date: 2023-11-05
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.11.05"
    identifier: arxiv_20231105
    parent: 202311_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (14)](#cscv-14)
- [cs.CL (29)](#cscl-29)
- [cs.LG (15)](#cslg-15)
- [cs.AI (3)](#csai-3)
- [eess.IV (4)](#eessiv-4)
- [cs.RO (2)](#csro-2)
- [cs.CR (3)](#cscr-3)
- [cs.HC (1)](#cshc-1)
- [cs.IR (1)](#csir-1)
- [cs.CY (1)](#cscy-1)
- [cs.DC (2)](#csdc-2)
- [cs.SI (1)](#cssi-1)
- [math.OC (1)](#mathoc-1)
- [cs.MA (1)](#csma-1)
- [eess.SY (1)](#eesssy-1)
- [cs.IT (1)](#csit-1)
- [quant-ph (1)](#quant-ph-1)

## cs.CV (14)



### (1/81) EmerNeRF: Emergent Spatial-Temporal Scene Decomposition via Self-Supervision (Jiawei Yang et al., 2023)

{{<citation>}}

Jiawei Yang, Boris Ivanovic, Or Litany, Xinshuo Weng, Seung Wook Kim, Boyi Li, Tong Che, Danfei Xu, Sanja Fidler, Marco Pavone, Yue Wang. (2023)  
**EmerNeRF: Emergent Spatial-Temporal Scene Decomposition via Self-Supervision**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.02077v1)  

---


**ABSTRACT**  
We present EmerNeRF, a simple yet powerful approach for learning spatial-temporal representations of dynamic driving scenes. Grounded in neural fields, EmerNeRF simultaneously captures scene geometry, appearance, motion, and semantics via self-bootstrapping. EmerNeRF hinges upon two core components: First, it stratifies scenes into static and dynamic fields. This decomposition emerges purely from self-supervision, enabling our model to learn from general, in-the-wild data sources. Second, EmerNeRF parameterizes an induced flow field from the dynamic field and uses this flow field to further aggregate multi-frame features, amplifying the rendering precision of dynamic objects. Coupling these three fields (static, dynamic, and flow) enables EmerNeRF to represent highly-dynamic scenes self-sufficiently, without relying on ground truth object annotations or pre-trained models for dynamic object segmentation or optical flow estimation. Our method achieves state-of-the-art performance in sensor simulation, significantly outperforming previous methods when reconstructing static (+2.93 PSNR) and dynamic (+3.70 PSNR) scenes. In addition, to bolster EmerNeRF's semantic generalization, we lift 2D visual foundation model features into 4D space-time and address a general positional bias in modern Transformers, significantly boosting 3D perception performance (e.g., 37.50% relative improvement in occupancy prediction accuracy on average). Finally, we construct a diverse and challenging 120-sequence dataset to benchmark neural fields under extreme and highly-dynamic settings.

{{</citation>}}


### (2/81) Towards Unsupervised Object Detection From LiDAR Point Clouds (Lunjun Zhang et al., 2023)

{{<citation>}}

Lunjun Zhang, Anqi Joyce Yang, Yuwen Xiong, Sergio Casas, Bin Yang, Mengye Ren, Raquel Urtasun. (2023)  
**Towards Unsupervised Object Detection From LiDAR Point Clouds**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-RO, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2311.02007v1)  

---


**ABSTRACT**  
In this paper, we study the problem of unsupervised object detection from 3D point clouds in self-driving scenes. We present a simple yet effective method that exploits (i) point clustering in near-range areas where the point clouds are dense, (ii) temporal consistency to filter out noisy unsupervised detections, (iii) translation equivariance of CNNs to extend the auto-labels to long range, and (iv) self-supervision for improving on its own. Our approach, OYSTER (Object Discovery via Spatio-Temporal Refinement), does not impose constraints on data collection (such as repeated traversals of the same location), is able to detect objects in a zero-shot manner without supervised finetuning (even in sparse, distant regions), and continues to self-improve given more rounds of iterative self-training. To better measure model performance in self-driving scenarios, we propose a new planning-centric perception metric based on distance-to-collision. We demonstrate that our unsupervised object detector significantly outperforms unsupervised baselines on PandaSet and Argoverse 2 Sensor dataset, showing promise that self-supervision combined with object priors can enable object discovery in the wild. For more information, visit the project website: https://waabi.ai/research/oyster

{{</citation>}}


### (3/81) Assessing Fidelity in XAI post-hoc techniques: A Comparative Study with Ground Truth Explanations Datasets (M. Miró-Nicolau et al., 2023)

{{<citation>}}

M. Miró-Nicolau, A. Jaume-i-Capó, G. Moyà-Alcover. (2023)  
**Assessing Fidelity in XAI post-hoc techniques: A Comparative Study with Ground Truth Explanations Datasets**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.01961v1)  

---


**ABSTRACT**  
The evaluation of the fidelity of eXplainable Artificial Intelligence (XAI) methods to their underlying models is a challenging task, primarily due to the absence of a ground truth for explanations. However, assessing fidelity is a necessary step for ensuring a correct XAI methodology. In this study, we conduct a fair and objective comparison of the current state-of-the-art XAI methods by introducing three novel image datasets with reliable ground truth for explanations. The primary objective of this comparison is to identify methods with low fidelity and eliminate them from further research, thereby promoting the development of more trustworthy and effective XAI techniques. Our results demonstrate that XAI methods based on the backpropagation of output information to input yield higher accuracy and reliability compared to methods relying on sensitivity analysis or Class Activation Maps (CAM). However, the backpropagation method tends to generate more noisy saliency maps. These findings have significant implications for the advancement of XAI methods, enabling the elimination of erroneous explanations and fostering the development of more robust and reliable XAI.

{{</citation>}}


### (4/81) ProS: Facial Omni-Representation Learning via Prototype-based Self-Distillation (Xing Di et al., 2023)

{{<citation>}}

Xing Di, Yiyu Zheng, Xiaoming Liu, Yu Cheng. (2023)  
**ProS: Facial Omni-Representation Learning via Prototype-based Self-Distillation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.01929v1)  

---


**ABSTRACT**  
This paper presents a novel approach, called Prototype-based Self-Distillation (ProS), for unsupervised face representation learning. The existing supervised methods heavily rely on a large amount of annotated training facial data, which poses challenges in terms of data collection and privacy concerns. To address these issues, we propose ProS, which leverages a vast collection of unlabeled face images to learn a comprehensive facial omni-representation. In particular, ProS consists of two vision-transformers (teacher and student models) that are trained with different augmented images (cropping, blurring, coloring, etc.). Besides, we build a face-aware retrieval system along with augmentations to obtain the curated images comprising predominantly facial areas. To enhance the discrimination of learned features, we introduce a prototype-based matching loss that aligns the similarity distributions between features (teacher or student) and a set of learnable prototypes. After pre-training, the teacher vision transformer serves as a backbone for downstream tasks, including attribute estimation, expression recognition, and landmark alignment, achieved through simple fine-tuning with additional layers. Extensive experiments demonstrate that our method achieves state-of-the-art performance on various tasks, both in full and few-shot settings. Furthermore, we investigate pre-training with synthetic face images, and ProS exhibits promising performance in this scenario as well.

{{</citation>}}


### (5/81) Holistic Representation Learning for Multitask Trajectory Anomaly Detection (Alexandros Stergiou et al., 2023)

{{<citation>}}

Alexandros Stergiou, Brent De Weerdt, Nikos Deligiannis. (2023)  
**Holistic Representation Learning for Multitask Trajectory Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection, Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.01851v1)  

---


**ABSTRACT**  
Video anomaly detection deals with the recognition of abnormal events in videos. Apart from the visual signal, video anomaly detection has also been addressed with the use of skeleton sequences. We propose a holistic representation of skeleton trajectories to learn expected motions across segments at different times. Our approach uses multitask learning to reconstruct any continuous unobserved temporal segment of the trajectory allowing the extrapolation of past or future segments and the interpolation of in-between segments. We use an end-to-end attention-based encoder-decoder. We encode temporally occluded trajectories, jointly learn latent representations of the occluded segments, and reconstruct trajectories based on expected motions across different temporal segments. Extensive experiments on three trajectory-based video anomaly detection datasets show the advantages and effectiveness of our approach with state-of-the-art results on anomaly detection in skeleton trajectories.

{{</citation>}}


### (6/81) Towards a Unified Transformer-based Framework for Scene Graph Generation and Human-object Interaction Detection (Tao He et al., 2023)

{{<citation>}}

Tao He, Lianli Gao, Jingkuan Song, Yuan-Fang Li. (2023)  
**Towards a Unified Transformer-based Framework for Scene Graph Generation and Human-object Interaction Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.01755v1)  

---


**ABSTRACT**  
Scene graph generation (SGG) and human-object interaction (HOI) detection are two important visual tasks aiming at localising and recognising relationships between objects, and interactions between humans and objects, respectively.   Prevailing works treat these tasks as distinct tasks, leading to the development of task-specific models tailored to individual datasets. However, we posit that the presence of visual relationships can furnish crucial contextual and intricate relational cues that significantly augment the inference of human-object interactions. This motivates us to think if there is a natural intrinsic relationship between the two tasks, where scene graphs can serve as a source for inferring human-object interactions. In light of this, we introduce SG2HOI+, a unified one-step model based on the Transformer architecture. Our approach employs two interactive hierarchical Transformers to seamlessly unify the tasks of SGG and HOI detection. Concretely, we initiate a relation Transformer tasked with generating relation triples from a suite of visual features. Subsequently, we employ another transformer-based decoder to predict human-object interactions based on the generated relation triples. A comprehensive series of experiments conducted across established benchmark datasets including Visual Genome, V-COCO, and HICO-DET demonstrates the compelling performance of our SG2HOI+ model in comparison to prevalent one-stage SGG models. Remarkably, our approach achieves competitive performance when compared to state-of-the-art HOI methods. Additionally, we observe that our SG2HOI+ jointly trained on both SGG and HOI tasks in an end-to-end manner yields substantial improvements for both tasks compared to individualized training paradigms.

{{</citation>}}


### (7/81) MixCon3D: Synergizing Multi-View and Cross-Modal Contrastive Learning for Enhancing 3D Representation (Yipeng Gao et al., 2023)

{{<citation>}}

Yipeng Gao, Zeyu Wang, Wei-Shi Zheng, Cihang Xie, Yuyin Zhou. (2023)  
**MixCon3D: Synergizing Multi-View and Cross-Modal Contrastive Learning for Enhancing 3D Representation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2311.01734v1)  

---


**ABSTRACT**  
Contrastive learning has emerged as a promising paradigm for 3D open-world understanding, jointly with text, image, and point cloud. In this paper, we introduce MixCon3D, which combines the complementary information between 2D images and 3D point clouds to enhance contrastive learning. With the further integration of multi-view 2D images, MixCon3D enhances the traditional tri-modal representation by offering a more accurate and comprehensive depiction of real-world 3D objects and bolstering text alignment. Additionally, we pioneer the first thorough investigation of various training recipes for the 3D contrastive learning paradigm, building a solid baseline with improved performance. Extensive experiments conducted on three representative benchmarks reveal that our method renders significant improvement over the baseline, surpassing the previous state-of-the-art performance on the challenging 1,156-category Objaverse-LVIS dataset by 5.7%. We further showcase the effectiveness of our approach in more applications, including text-to-3D retrieval and point cloud captioning. The code is available at https://github.com/UCSC-VLAA/MixCon3D.

{{</citation>}}


### (8/81) Towards Calibrated Robust Fine-Tuning of Vision-Language Models (Changdae Oh et al., 2023)

{{<citation>}}

Changdae Oh, Mijoo Kim, Hyesu Lim, Junhyeok Park, Euiseog Jeong, Zhi-Qi Cheng, Kyungwoo Song. (2023)  
**Towards Calibrated Robust Fine-Tuning of Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2311.01723v1)  

---


**ABSTRACT**  
While fine-tuning unleashes the potential of a pre-trained model to a specific task, it trades off the model's generalization capability on out-of-distribution (OOD) datasets. To mitigate this, robust fine-tuning aims to ensure performance on OOD datasets as well as an in-distribution (ID) dataset for which the model is being tuned. However, another criterion for reliable machine learning (ML), confidence calibration, has been overlooked despite its increasing demand for real-world high-stakes ML applications (e.g., autonomous driving and medical diagnosis). For the first time, we raise concerns about the calibration of fine-tuned vision-language models (VLMs) under distribution shift by showing that naive fine-tuning and even state-of-the-art robust fine-tuning methods hurt the calibration of pre-trained VLMs, especially on OOD datasets. To address this, we provide a simple approach, called a calibrated robust fine-tuning (CaRot) that incentivizes the calibration and robustness on both ID and OOD datasets. Empirical results on ImageNet-1K distribution shift evaluation verify the effectiveness of our method.

{{</citation>}}


### (9/81) Disentangled Representation Learning with Transmitted Information Bottleneck (Zhuohang Dang et al., 2023)

{{<citation>}}

Zhuohang Dang, Minnan Luo, Chengyou Jia, Guang Dai, Jihong Wang, Xiaojun Chang, Jingdong Wang, Qinghua Zheng. (2023)  
**Disentangled Representation Learning with Transmitted Information Bottleneck**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2311.01686v1)  

---


**ABSTRACT**  
Encoding only the task-related information from the raw data, \ie, disentangled representation learning, can greatly contribute to the robustness and generalizability of models. Although significant advances have been made by regularizing the information in representations with information theory, two major challenges remain: 1) the representation compression inevitably leads to performance drop; 2) the disentanglement constraints on representations are in complicated optimization. To these issues, we introduce Bayesian networks with transmitted information to formulate the interaction among input and representations during disentanglement. Building upon this framework, we propose \textbf{DisTIB} (\textbf{T}ransmitted \textbf{I}nformation \textbf{B}ottleneck for \textbf{Dis}entangled representation learning), a novel objective that navigates the balance between information compression and preservation. We employ variational inference to derive a tractable estimation for DisTIB. This estimation can be simply optimized via standard gradient descent with a reparameterization trick. Moreover, we theoretically prove that DisTIB can achieve optimal disentanglement, underscoring its superior efficacy. To solidify our claims, we conduct extensive experiments on various downstream tasks to demonstrate the appealing efficacy of DisTIB and validate our theoretical analyses.

{{</citation>}}


### (10/81) Flow-Based Feature Fusion for Vehicle-Infrastructure Cooperative 3D Object Detection (Haibao Yu et al., 2023)

{{<citation>}}

Haibao Yu, Yingjuan Tang, Enze Xie, Jilei Mao, Ping Luo, Zaiqing Nie. (2023)  
**Flow-Based Feature Fusion for Vehicle-Infrastructure Cooperative 3D Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Object Detection  
[Paper Link](http://arxiv.org/abs/2311.01682v1)  

---


**ABSTRACT**  
Cooperatively utilizing both ego-vehicle and infrastructure sensor data can significantly enhance autonomous driving perception abilities. However, the uncertain temporal asynchrony and limited communication conditions can lead to fusion misalignment and constrain the exploitation of infrastructure data. To address these issues in vehicle-infrastructure cooperative 3D (VIC3D) object detection, we propose the Feature Flow Net (FFNet), a novel cooperative detection framework. FFNet is a flow-based feature fusion framework that uses a feature flow prediction module to predict future features and compensate for asynchrony. Instead of transmitting feature maps extracted from still-images, FFNet transmits feature flow, leveraging the temporal coherence of sequential infrastructure frames. Furthermore, we introduce a self-supervised training approach that enables FFNet to generate feature flow with feature prediction ability from raw infrastructure sequences. Experimental results demonstrate that our proposed method outperforms existing cooperative detection methods while only requiring about 1/100 of the transmission cost of raw data and covers all latency in one model on the DAIR-V2X dataset. The code is available at \href{https://github.com/haibao-yu/FFNet-VIC3D}{https://github.com/haibao-yu/FFNet-VIC3D}.

{{</citation>}}


### (11/81) MineSegSAT: An automated system to evaluate mining disturbed area extents from Sentinel-2 imagery (Ezra MacDonald et al., 2023)

{{<citation>}}

Ezra MacDonald, Derek Jacoby, Yvonne Coady. (2023)  
**MineSegSAT: An automated system to evaluate mining disturbed area extents from Sentinel-2 imagery**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV, eess-IV  
Keywords: AWS, Amazon  
[Paper Link](http://arxiv.org/abs/2311.01676v1)  

---


**ABSTRACT**  
Assessing the environmental impact of the mineral extraction industry plays a critical role in understanding and mitigating the ecological consequences of extractive activities. This paper presents MineSegSAT, a model that presents a novel approach to predicting environmentally impacted areas of mineral extraction sites using the SegFormer deep learning segmentation architecture trained on Sentinel-2 data. The data was collected from non-overlapping regions over Western Canada in 2021 containing areas of land that have been environmentally impacted by mining activities that were identified from high-resolution satellite imagery in 2021. The SegFormer architecture, a state-of-the-art semantic segmentation framework, is employed to leverage its advanced spatial understanding capabilities for accurate land cover classification. We investigate the efficacy of loss functions including Dice, Tversky, and Lovasz loss respectively. The trained model was utilized for inference over the test region in the ensuing year to identify potential areas of expansion or contraction over these same periods. The Sentinel-2 data is made available on Amazon Web Services through a collaboration with Earth Daily Analytics which provides corrected and tiled analytics-ready data on the AWS platform. The model and ongoing API to access the data on AWS allow the creation of an automated tool to monitor the extent of disturbed areas surrounding known mining sites to ensure compliance with their environmental impact goals.

{{</citation>}}


### (12/81) Content Significance Distribution of Sub-Text Blocks in Articles and Its Application to Article-Organization Assessment (You Zhou et al., 2023)

{{<citation>}}

You Zhou, Jie Wang. (2023)  
**Content Significance Distribution of Sub-Text Blocks in Articles and Its Application to Article-Organization Assessment**  

---
Primary Category: cs.CV  
Categories: I-5-4, cs-CV, cs.CV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.01673v1)  

---


**ABSTRACT**  
We explore how to capture the significance of a sub-text block in an article and how it may be used for text mining tasks. A sub-text block is a sub-sequence of sentences in the article. We formulate the notion of content significance distribution (CSD) of sub-text blocks, referred to as CSD of the first kind and denoted by CSD-1. In particular, we leverage Hugging Face's SentenceTransformer to generate contextual sentence embeddings, and use MoverScore over text embeddings to measure how similar a sub-text block is to the entire text. To overcome the exponential blowup on the number of sub-text blocks, we present an approximation algorithm and show that the approximated CSD-1 is almost identical to the exact CSD-1. Under this approximation, we show that the average and median CSD-1's for news, scholarly research, argument, and narrative articles share the same pattern. We also show that under a certain linear transformation, the complement of the cumulative distribution function of the beta distribution with certain values of $\alpha$ and $\beta$ resembles a CSD-1 curve. We then use CSD-1's to extract linguistic features to train an SVC classifier for assessing how well an article is organized. Through experiments, we show that this method achieves high accuracy for assessing student essays. Moreover, we study CSD of sentence locations, referred to as CSD of the second kind and denoted by CSD-2, and show that average CSD-2's for different types of articles possess distinctive patterns, which either conform common perceptions of article structures or provide rectification with minor deviation.

{{</citation>}}


### (13/81) Efficient Cloud Pipelines for Neural Radiance Fields (Derek Jacoby et al., 2023)

{{<citation>}}

Derek Jacoby, Donglin Xu, Weder Ribas, Minyi Xu, Ting Liu, Vishwanath Jayaraman, Mengdi Wei, Emma De Blois, Yvonne Coady. (2023)  
**Efficient Cloud Pipelines for Neural Radiance Fields**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI, Azure, Microsoft  
[Paper Link](http://arxiv.org/abs/2311.01659v1)  

---


**ABSTRACT**  
Since their introduction in 2020, Neural Radiance Fields (NeRFs) have taken the computer vision community by storm. They provide a multi-view representation of a scene or object that is ideal for eXtended Reality (XR) applications and for creative endeavors such as virtual production, as well as change detection operations in geospatial analytics. The computational cost of these generative AI models is quite high, however, and the construction of cloud pipelines to generate NeRFs is neccesary to realize their potential in client applications. In this paper, we present pipelines on a high performance academic computing cluster and compare it with a pipeline implemented on Microsoft Azure. Along the way, we describe some uses of NeRFs in enabling novel user interaction scenarios.

{{</citation>}}


### (14/81) SemiGPC: Distribution-Aware Label Refinement for Imbalanced Semi-Supervised Learning Using Gaussian Processes (Abdelhak Lemkhenter et al., 2023)

{{<citation>}}

Abdelhak Lemkhenter, Manchen Wang, Luca Zancato, Gurumurthy Swaminathan, Paolo Favaro, Davide Modolo. (2023)  
**SemiGPC: Distribution-Aware Label Refinement for Imbalanced Semi-Supervised Learning Using Gaussian Processes**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2311.01646v1)  

---


**ABSTRACT**  
In this paper we introduce SemiGPC, a distribution-aware label refinement strategy based on Gaussian Processes where the predictions of the model are derived from the labels posterior distribution. Differently from other buffer-based semi-supervised methods such as CoMatch and SimMatch, our SemiGPC includes a normalization term that addresses imbalances in the global data distribution while maintaining local sensitivity. This explicit control allows SemiGPC to be more robust to confirmation bias especially under class imbalance. We show that SemiGPC improves performance when paired with different Semi-Supervised methods such as FixMatch, ReMixMatch, SimMatch and FreeMatch and different pre-training strategies including MSN and Dino. We also show that SemiGPC achieves state of the art results under different degrees of class imbalance on standard CIFAR10-LT/CIFAR100-LT especially in the low data-regime. Using SemiGPC also results in about 2% avg.accuracy increase compared to a new competitive baseline on the more challenging benchmarks SemiAves, SemiCUB, SemiFungi and Semi-iNat.

{{</citation>}}


## cs.CL (29)



### (15/81) Grounded Intuition of GPT-Vision's Abilities with Scientific Images (Alyssa Hwang et al., 2023)

{{<citation>}}

Alyssa Hwang, Andrew Head, Chris Callison-Burch. (2023)  
**Grounded Intuition of GPT-Vision's Abilities with Scientific Images**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2311.02069v1)  

---


**ABSTRACT**  
GPT-Vision has impressed us on a range of vision-language tasks, but it comes with the familiar new challenge: we have little idea of its capabilities and limitations. In our study, we formalize a process that many have instinctively been trying already to develop "grounded intuition" of this new model. Inspired by the recent movement away from benchmarking in favor of example-driven qualitative evaluation, we draw upon grounded theory and thematic analysis in social science and human-computer interaction to establish a rigorous framework for qualitative evaluation in natural language processing. We use our technique to examine alt text generation for scientific figures, finding that GPT-Vision is particularly sensitive to prompting, counterfactual text in images, and relative spatial relationships. Our method and analysis aim to help researchers ramp up their own grounded intuitions of new models while exposing how GPT-Vision can be applied to make information more accessible.

{{</citation>}}


### (16/81) Post Turing: Mapping the landscape of LLM Evaluation (Alexey Tikhonov et al., 2023)

{{<citation>}}

Alexey Tikhonov, Ivan P. Yamshchikov. (2023)  
**Post Turing: Mapping the landscape of LLM Evaluation**  

---
Primary Category: cs.CL  
Categories: 68T50, I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2311.02049v1)  

---


**ABSTRACT**  
In the rapidly evolving landscape of Large Language Models (LLMs), introduction of well-defined and standardized evaluation methodologies remains a crucial challenge. This paper traces the historical trajectory of LLM evaluations, from the foundational questions posed by Alan Turing to the modern era of AI research. We categorize the evolution of LLMs into distinct periods, each characterized by its unique benchmarks and evaluation criteria. As LLMs increasingly mimic human-like behaviors, traditional evaluation proxies, such as the Turing test, have become less reliable. We emphasize the pressing need for a unified evaluation system, given the broader societal implications of these models. Through an analysis of common evaluation methodologies, we advocate for a qualitative shift in assessment approaches, underscoring the importance of standardization and objective criteria. This work serves as a call for the AI community to collaboratively address the challenges of LLM evaluation, ensuring their reliability, fairness, and societal benefit.

{{</citation>}}


### (17/81) Vicinal Risk Minimization for Few-Shot Cross-lingual Transfer in Abusive Language Detection (Gretel Liz De la Peña Sarracén et al., 2023)

{{<citation>}}

Gretel Liz De la Peña Sarracén, Paolo Rosso, Robert Litschko, Goran Glavaš, Simone Paolo Ponzetto. (2023)  
**Vicinal Risk Minimization for Few-Shot Cross-lingual Transfer in Abusive Language Detection**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2311.02025v1)  

---


**ABSTRACT**  
Cross-lingual transfer learning from high-resource to medium and low-resource languages has shown encouraging results. However, the scarcity of resources in target languages remains a challenge. In this work, we resort to data augmentation and continual pre-training for domain adaptation to improve cross-lingual abusive language detection. For data augmentation, we analyze two existing techniques based on vicinal risk minimization and propose MIXAG, a novel data augmentation method which interpolates pairs of instances based on the angle of their representations. Our experiments involve seven languages typologically distinct from English and three different domains. The results reveal that the data augmentation strategies can enhance few-shot cross-lingual abusive language detection. Specifically, we observe that consistently in all target languages, MIXAG improves significantly in multidomain and multilingual environments. Finally, we show through an error analysis how the domain adaptation can favour the class of abusive texts (reducing false negatives), but at the same time, declines the precision of the abusive language detection model.

{{</citation>}}


### (18/81) ProSG: Using Prompt Synthetic Gradients to Alleviate Prompt Forgetting of RNN-like Language Models (Haotian Luo et al., 2023)

{{<citation>}}

Haotian Luo, Kunming Wu, Cheng Dai, Sixian Ding, Xinhao Chen. (2023)  
**ProSG: Using Prompt Synthetic Gradients to Alleviate Prompt Forgetting of RNN-like Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.01981v1)  

---


**ABSTRACT**  
RNN-like language models are getting renewed attention from NLP researchers in recent years and several models have made significant progress, which demonstrates performance comparable to traditional transformers. However, due to the recurrent nature of RNNs, this kind of language model can only store information in a set of fixed-length state vectors. As a consequence, they still suffer from forgetfulness though after a lot of improvements and optimizations, when given complex instructions or prompts. As the prompted generation is the main and most concerned function of LMs, solving the problem of forgetting in the process of generation is no wonder of vital importance. In this paper, focusing on easing the prompt forgetting during generation, we proposed an architecture to teach the model memorizing prompt during generation by synthetic gradient. To force the model to memorize the prompt, we derive the states that encode the prompt, then transform it into model parameter modification using low-rank gradient approximation, which hard-codes the prompt into model parameters temporarily. We construct a dataset for experiments, and the results have demonstrated the effectiveness of our method in solving the problem of forgetfulness in the process of prompted generation. We will release all the code upon acceptance.

{{</citation>}}


### (19/81) The language of prompting: What linguistic properties make a prompt successful? (Alina Leidinger et al., 2023)

{{<citation>}}

Alina Leidinger, Robert van Rooij, Ekaterina Shutova. (2023)  
**The language of prompting: What linguistic properties make a prompt successful?**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NLP  
[Paper Link](http://arxiv.org/abs/2311.01967v1)  

---


**ABSTRACT**  
The latest generation of LLMs can be prompted to achieve impressive zero-shot or few-shot performance in many NLP tasks. However, since performance is highly sensitive to the choice of prompts, considerable effort has been devoted to crowd-sourcing prompts or designing methods for prompt optimisation. Yet, we still lack a systematic understanding of how linguistic properties of prompts correlate with task performance. In this work, we investigate how LLMs of different sizes, pre-trained and instruction-tuned, perform on prompts that are semantically equivalent, but vary in linguistic structure. We investigate both grammatical properties such as mood, tense, aspect and modality, as well as lexico-semantic variation through the use of synonyms. Our findings contradict the common assumption that LLMs achieve optimal performance on lower perplexity prompts that reflect language use in pretraining or instruction-tuning data. Prompts transfer poorly between datasets or models, and performance cannot generally be explained by perplexity, word frequency, ambiguity or prompt length. Based on our results, we put forward a proposal for a more robust and comprehensive evaluation standard for prompting research.

{{</citation>}}


### (20/81) Too Much Information: Keeping Training Simple for BabyLMs (Lukas Edman et al., 2023)

{{<citation>}}

Lukas Edman, Lisa Bylinina. (2023)  
**Too Much Information: Keeping Training Simple for BabyLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLUE  
[Paper Link](http://arxiv.org/abs/2311.01955v1)  

---


**ABSTRACT**  
This paper details the work of the University of Groningen for the BabyLM Challenge. We follow the idea that, like babies, language models should be introduced to simpler concepts first and build off of that knowledge to understand more complex concepts. We examine this strategy of simple-then-complex through a variety of lenses, namely context size, vocabulary, and overall linguistic complexity of the data. We find that only one, context size, is truly beneficial to training a language model. However this simple change to context size gives us improvements of 2 points on average on (Super)GLUE tasks, 1 point on MSGS tasks, and 12\% on average on BLiMP tasks. Our context-limited model outperforms the baseline that was trained on 10$\times$ the amount of data.

{{</citation>}}


### (21/81) Hint-enhanced In-Context Learning wakes Large Language Models up for knowledge-intensive tasks (Yifan Wang et al., 2023)

{{<citation>}}

Yifan Wang, Qingyan Guo, Xinzhe Ni, Chufan Shi, Lemao Liu, Haiyun Jiang, Yujiu Yang. (2023)  
**Hint-enhanced In-Context Learning wakes Large Language Models up for knowledge-intensive tasks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: LLaMA, Language Model, QA  
[Paper Link](http://arxiv.org/abs/2311.01949v1)  

---


**ABSTRACT**  
In-context learning (ICL) ability has emerged with the increasing scale of large language models (LLMs), enabling them to learn input-label mappings from demonstrations and perform well on downstream tasks. However, under the standard ICL setting, LLMs may sometimes neglect query-related information in demonstrations, leading to incorrect predictions. To address this limitation, we propose a new paradigm called Hint-enhanced In-Context Learning (HICL) to explore the power of ICL in open-domain question answering, an important form in knowledge-intensive tasks. HICL leverages LLMs' reasoning ability to extract query-related knowledge from demonstrations, then concatenates the knowledge to prompt LLMs in a more explicit way. Furthermore, we track the source of this knowledge to identify specific examples, and introduce a Hint-related Example Retriever (HER) to select informative examples for enhanced demonstrations. We evaluate HICL with HER on 3 open-domain QA benchmarks, and observe average performance gains of 2.89 EM score and 2.52 F1 score on gpt-3.5-turbo, 7.62 EM score and 7.27 F1 score on LLaMA-2-Chat-7B compared with standard setting.

{{</citation>}}


### (22/81) Constructing Temporal Dynamic Knowledge Graphs from Interactive Text-based Games (Keunwoo Peter Yu, 2023)

{{<citation>}}

Keunwoo Peter Yu. (2023)  
**Constructing Temporal Dynamic Knowledge Graphs from Interactive Text-based Games**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI, Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2311.01928v1)  

---


**ABSTRACT**  
In natural language processing, interactive text-based games serve as a test bed for interactive AI systems. Prior work has proposed to play text-based games by acting based on discrete knowledge graphs constructed by the Discrete Graph Updater (DGU) to represent the game state from the natural language description. While DGU has shown promising results with high interpretability, it suffers from lower knowledge graph accuracy due to its lack of temporality and limited generalizability to complex environments with objects with the same label. In order to address DGU's weaknesses while preserving its high interpretability, we propose the Temporal Discrete Graph Updater (TDGU), a novel neural network model that represents dynamic knowledge graphs as a sequence of timestamped graph events and models them using a temporal point based graph neural network. Through experiments on the dataset collected from a text-based game TextWorld, we show that TDGU outperforms the baseline DGU. We further show the importance of temporal information for TDGU's performance through an ablation study and demonstrate that TDGU has the ability to generalize to more complex environments with objects with the same label. All the relevant code can be found at \url{https://github.com/yukw777/temporal-discrete-graph-updater}.

{{</citation>}}


### (23/81) Large Language Models Illuminate a Progressive Pathway to Artificial Healthcare Assistant: A Review (Mingze Yuan et al., 2023)

{{<citation>}}

Mingze Yuan, Peng Bao, Jiajia Yuan, Yunhao Shen, Zifan Chen, Yi Xie, Jie Zhao, Yang Chen, Li Zhang, Lin Shen, Bin Dong. (2023)  
**Large Language Models Illuminate a Progressive Pathway to Artificial Healthcare Assistant: A Review**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.01918v1)  

---


**ABSTRACT**  
With the rapid development of artificial intelligence, large language models (LLMs) have shown promising capabilities in mimicking human-level language comprehension and reasoning. This has sparked significant interest in applying LLMs to enhance various aspects of healthcare, ranging from medical education to clinical decision support. However, medicine involves multifaceted data modalities and nuanced reasoning skills, presenting challenges for integrating LLMs. This paper provides a comprehensive review on the applications and implications of LLMs in medicine. It begins by examining the fundamental applications of general-purpose and specialized LLMs, demonstrating their utilities in knowledge retrieval, research support, clinical workflow automation, and diagnostic assistance. Recognizing the inherent multimodality of medicine, the review then focuses on multimodal LLMs, investigating their ability to process diverse data types like medical imaging and EHRs to augment diagnostic accuracy. To address LLMs' limitations regarding personalization and complex clinical reasoning, the paper explores the emerging development of LLM-powered autonomous agents for healthcare. Furthermore, it summarizes the evaluation methodologies for assessing LLMs' reliability and safety in medical contexts. Overall, this review offers an extensive analysis on the transformative potential of LLMs in modern medicine. It also highlights the pivotal need for continuous optimizations and ethical oversight before these models can be effectively integrated into clinical practice. Visit https://github.com/mingze-yuan/Awesome-LLM-Healthcare for an accompanying GitHub repository containing latest papers.

{{</citation>}}


### (24/81) BoschAI @ PLABA 2023: Leveraging Edit Operations in End-to-End Neural Sentence Simplification (Valentin Knappich et al., 2023)

{{<citation>}}

Valentin Knappich, Simon Razniewski, Annemarie Friedrich. (2023)  
**BoschAI @ PLABA 2023: Leveraging Edit Operations in End-to-End Neural Sentence Simplification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.01907v1)  

---


**ABSTRACT**  
Automatic simplification can help laypeople to comprehend complex scientific text. Language models are frequently applied to this task by translating from complex to simple language. In this paper, we describe our system based on Llama 2, which ranked first in the PLABA shared task addressing the simplification of biomedical text. We find that the large portion of shared tokens between input and output leads to weak training signals and conservatively editing models. To mitigate these issues, we propose sentence-level and token-level loss weights. They give higher weight to modified tokens, indicated by edit distance and edit operations, respectively. We conduct an empirical evaluation on the PLABA dataset and find that both approaches lead to simplifications closer to those created by human annotators (+1.8% / +3.5% SARI), simpler language (-1 / -1.1 FKGL) and more edits (1.6x / 1.8x edit distance) compared to the same model fine-tuned with standard cross entropy. We furthermore show that the hyperparameter $\lambda$ in token-level loss weights can be used to control the edit distance and the simplicity level (FKGL).

{{</citation>}}


### (25/81) Indicative Summarization of Long Discussions (Shahbaz Syed et al., 2023)

{{<citation>}}

Shahbaz Syed, Dominik Schwabe, Khalid Al-Khatib, Martin Potthast. (2023)  
**Indicative Summarization of Long Discussions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Summarization  
[Paper Link](http://arxiv.org/abs/2311.01882v1)  

---


**ABSTRACT**  
Online forums encourage the exchange and discussion of different stances on many topics. Not only do they provide an opportunity to present one's own arguments, but may also gather a broad cross-section of others' arguments. However, the resulting long discussions are difficult to overview. This paper presents a novel unsupervised approach using large language models (LLMs) to generating indicative summaries for long discussions that basically serve as tables of contents. Our approach first clusters argument sentences, generates cluster labels as abstractive summaries, and classifies the generated cluster labels into argumentation frames resulting in a two-level summary. Based on an extensively optimized prompt engineering approach, we evaluate 19~LLMs for generative cluster labeling and frame classification. To evaluate the usefulness of our indicative summaries, we conduct a purpose-driven user study via a new visual interface called Discussion Explorer: It shows that our proposed indicative summaries serve as a convenient navigation tool to explore long discussions.

{{</citation>}}


### (26/81) Sentiment Analysis through LLM Negotiations (Xiaofei Sun et al., 2023)

{{<citation>}}

Xiaofei Sun, Xiaoya Li, Shengyu Zhang, Shuhe Wang, Fei Wu, Jiwei Li, Tianwei Zhang, Guoyin Wang. (2023)  
**Sentiment Analysis through LLM Negotiations**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Sentiment Analysis, Twitter  
[Paper Link](http://arxiv.org/abs/2311.01876v1)  

---


**ABSTRACT**  
A standard paradigm for sentiment analysis is to rely on a singular LLM and makes the decision in a single round under the framework of in-context learning. This framework suffers the key disadvantage that the single-turn output generated by a single LLM might not deliver the perfect decision, just as humans sometimes need multiple attempts to get things right. This is especially true for the task of sentiment analysis where deep reasoning is required to address the complex linguistic phenomenon (e.g., clause composition, irony, etc) in the input.   To address this issue, this paper introduces a multi-LLM negotiation framework for sentiment analysis. The framework consists of a reasoning-infused generator to provide decision along with rationale, a explanation-deriving discriminator to evaluate the credibility of the generator. The generator and the discriminator iterate until a consensus is reached. The proposed framework naturally addressed the aforementioned challenge, as we are able to take the complementary abilities of two LLMs, have them use rationale to persuade each other for correction.   Experiments on a wide range of sentiment analysis benchmarks (SST-2, Movie Review, Twitter, yelp, amazon, IMDB) demonstrate the effectiveness of proposed approach: it consistently yields better performances than the ICL baseline across all benchmarks, and even superior performances to supervised baselines on the Twitter and movie review datasets.

{{</citation>}}


### (27/81) Efficient Black-Box Adversarial Attacks on Neural Text Detectors (Vitalii Fishchuk et al., 2023)

{{<citation>}}

Vitalii Fishchuk, Daniel Braun. (2023)  
**Efficient Black-Box Adversarial Attacks on Neural Text Detectors**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Adversarial Attack, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2311.01873v1)  

---


**ABSTRACT**  
Neural text detectors are models trained to detect whether a given text was generated by a language model or written by a human. In this paper, we investigate three simple and resource-efficient strategies (parameter tweaking, prompt engineering, and character-level mutations) to alter texts generated by GPT-3.5 that are unsuspicious or unnoticeable for humans but cause misclassification by neural text detectors. The results show that especially parameter tweaking and character-level mutations are effective strategies.

{{</citation>}}


### (28/81) Multi-EuP: The Multilingual European Parliament Dataset for Analysis of Bias in Information Retrieval (Jinrui Yang et al., 2023)

{{<citation>}}

Jinrui Yang, Timothy Baldwin, Trevor Cohn. (2023)  
**Multi-EuP: The Multilingual European Parliament Dataset for Analysis of Bias in Information Retrieval**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs.CL  
Keywords: Bias, Information Retrieval, Multilingual  
[Paper Link](http://arxiv.org/abs/2311.01870v1)  

---


**ABSTRACT**  
We present Multi-EuP, a new multilingual benchmark dataset, comprising 22K multi-lingual documents collected from the European Parliament, spanning 24 languages. This dataset is designed to investigate fairness in a multilingual information retrieval (IR) context to analyze both language and demographic bias in a ranking context. It boasts an authentic multilingual corpus, featuring topics translated into all 24 languages, as well as cross-lingual relevance judgments. Furthermore, it offers rich demographic information associated with its documents, facilitating the study of demographic bias. We report the effectiveness of Multi-EuP for benchmarking both monolingual and multilingual IR. We also conduct a preliminary experiment on language bias caused by the choice of tokenization strategy.

{{</citation>}}


### (29/81) Towards Concept-Aware Large Language Models (Chen Shani et al., 2023)

{{<citation>}}

Chen Shani, Jilles Vreeken, Dafna Shahaf. (2023)  
**Towards Concept-Aware Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.01866v1)  

---


**ABSTRACT**  
Concepts play a pivotal role in various human cognitive functions, including learning, reasoning and communication. However, there is very little work on endowing machines with the ability to form and reason with concepts. In particular, state-of-the-art large language models (LLMs) work at the level of tokens, not concepts.   In this work, we analyze how well contemporary LLMs capture human concepts and their structure. We then discuss ways to develop concept-aware LLMs, taking place at different stages of the pipeline. We sketch a method for pretraining LLMs using concepts, and also explore the simpler approach that uses the output of existing LLMs. Despite its simplicity, our proof-of-concept is shown to better match human intuition, as well as improve the robustness of predictions. These preliminary results underscore the promise of concept-aware LLMs.

{{</citation>}}


### (30/81) $R^3$-NL2GQL: A Hybrid Models Approach for for Accuracy Enhancing and Hallucinations Mitigation (Yuhang Zhou et al., 2023)

{{<citation>}}

Yuhang Zhou, He Yu, Siyu Tian, Dan Chen, Liuzhi Zhou, Xinlin Yu, Chuanjun Ji, Sen Liu, Guangnan Ye, Hongfeng Chai. (2023)  
**$R^3$-NL2GQL: A Hybrid Models Approach for for Accuracy Enhancing and Hallucinations Mitigation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-DB, cs.CL  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2311.01862v1)  

---


**ABSTRACT**  
While current NL2SQL tasks constructed using Foundation Models have achieved commendable results, their direct application to Natural Language to Graph Query Language (NL2GQL) tasks poses challenges due to the significant differences between GQL and SQL expressions, as well as the numerous types of GQL. Our extensive experiments reveal that in NL2GQL tasks, larger Foundation Models demonstrate superior cross-schema generalization abilities, while smaller Foundation Models struggle to improve their GQL generation capabilities through fine-tuning. However, after fine-tuning, smaller models exhibit better intent comprehension and higher grammatical accuracy. Diverging from rule-based and slot-filling techniques, we introduce R3-NL2GQL, which employs both smaller and larger Foundation Models as reranker, rewriter and refiner. The approach harnesses the comprehension ability of smaller models for information reranker and rewriter, and the exceptional generalization and generation capabilities of larger models to transform input natural language queries and code structure schema into any form of GQLs. Recognizing the lack of established datasets in this nascent domain, we have created a bilingual dataset derived from graph database documentation and some open-source Knowledge Graphs (KGs). We tested our approach on this dataset and the experimental results showed that delivers promising performance and robustness.Our code and dataset is available at https://github.com/zhiqix/NL2GQL

{{</citation>}}


### (31/81) Mitigating Framing Bias with Polarity Minimization Loss (Yejin Bang et al., 2023)

{{<citation>}}

Yejin Bang, Nayeon Lee, Pascale Fung. (2023)  
**Mitigating Framing Bias with Polarity Minimization Loss**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2311.01817v1)  

---


**ABSTRACT**  
Framing bias plays a significant role in exacerbating political polarization by distorting the perception of actual events. Media outlets with divergent political stances often use polarized language in their reporting of the same event. We propose a new loss function that encourages the model to minimize the polarity difference between the polarized input articles to reduce framing bias. Specifically, our loss is designed to jointly optimize the model to map polarity ends bidirectionally. Our experimental results demonstrate that incorporating the proposed polarity minimization loss leads to a substantial reduction in framing bias when compared to a BART-based multi-document summarization model. Notably, we find that the effectiveness of this approach is most pronounced when the model is trained to minimize the polarity loss associated with informational framing bias (i.e., skewed selection of information to report).

{{</citation>}}


### (32/81) AFPQ: Asymmetric Floating Point Quantization for LLMs (Yijia Zhang et al., 2023)

{{<citation>}}

Yijia Zhang, Sicheng Zhang, Shijie Cao, Dayou Du, Jianyu Wei, Ting Cao, Ningyi Xu. (2023)  
**AFPQ: Asymmetric Floating Point Quantization for LLMs**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Quantization  
[Paper Link](http://arxiv.org/abs/2311.01792v1)  

---


**ABSTRACT**  
Large language models (LLMs) show great performance in various tasks, but face deployment challenges from limited memory capacity and bandwidth. Low-bit weight quantization can save memory and accelerate inference. Although floating-point (FP) formats show good performance in LLM quantization, they tend to perform poorly with small group sizes or sub-4 bits. We find the reason is that the absence of asymmetry in previous FP quantization makes it unsuitable for handling asymmetric value distribution of LLM weight tensors. In this work, we propose asymmetric FP quantization (AFPQ), which sets separate scales for positive and negative values. Our method leads to large accuracy improvements and can be easily plugged into other quantization methods, including GPTQ and AWQ, for better performance. Besides, no additional storage is needed compared with asymmetric integer (INT) quantization. The code is available at https://github.com/zhangsichengsjtu/AFPQ.

{{</citation>}}


### (33/81) TCM-GPT: Efficient Pre-training of Large Language Models for Domain Adaptation in Traditional Chinese Medicine (Guoxing Yang et al., 2023)

{{<citation>}}

Guoxing Yang, Jianyu Shi, Zan Wang, Xiaohong Liu, Guangyu Wang. (2023)  
**TCM-GPT: Efficient Pre-training of Large Language Models for Domain Adaptation in Traditional Chinese Medicine**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.01786v1)  

---


**ABSTRACT**  
Pre-training and fine-tuning have emerged as a promising paradigm across various natural language processing (NLP) tasks. The effectiveness of pretrained large language models (LLM) has witnessed further enhancement, holding potential for applications in the field of medicine, particularly in the context of Traditional Chinese Medicine (TCM). However, the application of these general models to specific domains often yields suboptimal results, primarily due to challenges like lack of domain knowledge, unique objectives, and computational efficiency. Furthermore, their effectiveness in specialized domains, such as Traditional Chinese Medicine, requires comprehensive evaluation. To address the above issues, we propose a novel domain specific TCMDA (TCM Domain Adaptation) approach, efficient pre-training with domain-specific corpus. Specifically, we first construct a large TCM-specific corpus, TCM-Corpus-1B, by identifying domain keywords and retreving from general corpus. Then, our TCMDA leverages the LoRA which freezes the pretrained model's weights and uses rank decomposition matrices to efficiently train specific dense layers for pre-training and fine-tuning, efficiently aligning the model with TCM-related tasks, namely TCM-GPT-7B. We further conducted extensive experiments on two TCM tasks, including TCM examination and TCM diagnosis. TCM-GPT-7B archived the best performance across both datasets, outperforming other models by relative increments of 17% and 12% in accuracy, respectively. To the best of our knowledge, our study represents the pioneering validation of domain adaptation of a large language model with 7 billion parameters in TCM domain. We will release both TCMCorpus-1B and TCM-GPT-7B model once accepted to facilitate interdisciplinary development in TCM and NLP, serving as the foundation for further study.

{{</citation>}}


### (34/81) PPTC Benchmark: Evaluating Large Language Models for PowerPoint Task Completion (Yiduo Guo et al., 2023)

{{<citation>}}

Yiduo Guo, Zekai Zhang, Yaobo Liang, Dongyan Zhao, Duan Nan. (2023)  
**PPTC Benchmark: Evaluating Large Language Models for PowerPoint Task Completion**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2311.01767v1)  

---


**ABSTRACT**  
Recent evaluations of Large Language Models (LLMs) have centered around testing their zero-shot/few-shot capabilities for basic natural language tasks and their ability to translate instructions into tool APIs. However, the evaluation of LLMs utilizing complex tools to finish multi-turn, multi-modal instructions in a complex multi-modal environment has not been investigated. To address this gap, we introduce the PowerPoint Task Completion (PPTC) benchmark to assess LLMs' ability to create and edit PPT files based on user instructions. It contains 279 multi-turn sessions covering diverse topics and hundreds of instructions involving multi-modal operations. We also propose the PPTX-Match Evaluation System that evaluates if LLMs finish the instruction based on the prediction file rather than the label API sequence, thus it supports various LLM-generated API sequences. We measure 3 closed LLMs and 6 open-source LLMs. The results show that GPT-4 outperforms other LLMs with 75.1\% accuracy in single-turn dialogue testing but faces challenges in completing entire sessions, achieving just 6\% session accuracy. We find three main error causes in our benchmark: error accumulation in the multi-turn session, long PPT template processing, and multi-modality perception. These pose great challenges for future LLM and agent systems. We release the data, code, and evaluation system of PPTC at \url{https://github.com/gydpku/PPTC}.

{{</citation>}}


### (35/81) Indo LEGO-ABSA: A Multitask Generative Aspect Based Sentiment Analysis for Indonesian Language (Randy Zakya Suchrady et al., 2023)

{{<citation>}}

Randy Zakya Suchrady, Ayu Purwarianti. (2023)  
**Indo LEGO-ABSA: A Multitask Generative Aspect Based Sentiment Analysis for Indonesian Language**  

---
Primary Category: cs.CL  
Categories: I-2-7, cs-AI, cs-CL, cs.CL  
Keywords: Sentiment Analysis, T5  
[Paper Link](http://arxiv.org/abs/2311.01757v1)  

---


**ABSTRACT**  
Aspect-based sentiment analysis is a method in natural language processing aimed at identifying and understanding sentiments related to specific aspects of an entity. Aspects are words or phrases that represent an aspect or attribute of a particular entity. Previous research has utilized generative pre-trained language models to perform aspect-based sentiment analysis. LEGO-ABSA is one framework that has successfully employed generative pre-trained language models in aspect-based sentiment analysis, particularly in English. LEGO-ABSA uses a multitask learning and prompting approach to enhance model performance. However, the application of this approach has not been done in the context of Bahasa Indonesia. Therefore, this research aims to implement the multitask learning and prompting approach in aspect-based sentiment analysis for Bahasa Indonesia using generative pre-trained language models. In this study, the Indo LEGO-ABSA model is developed, which is an aspect-based sentiment analysis model utilizing generative pre-trained language models and trained with multitask learning and prompting. Indo LEGO-ABSA is trained with a hotel domain dataset in the Indonesian language. The obtained results include an f1-score of 79.55% for the Aspect Sentiment Triplet Extraction task, 86.09% for Unified Aspect-based Sentiment Analysis, 79.85% for Aspect Opinion Pair Extraction, 87.45% for Aspect Term Extraction, and 88.09% for Opinion Term Extraction. Indo LEGO-ABSA adopts the LEGO-ABSA framework that employs the T5 model, specifically mT5, by applying multitask learning to train all tasks within aspect-based sentiment analysis.

{{</citation>}}


### (36/81) SAC$^3$: Reliable Hallucination Detection in Black-Box Language Models via Semantic-aware Cross-check Consistency (Jiaxin Zhang et al., 2023)

{{<citation>}}

Jiaxin Zhang, Zhuohang Li, Kamalika Das, Bradley A. Malin, Sricharan Kumar. (2023)  
**SAC$^3$: Reliable Hallucination Detection in Black-Box Language Models via Semantic-aware Cross-check Consistency**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.01740v1)  

---


**ABSTRACT**  
Hallucination detection is a critical step toward understanding the trustworthiness of modern language models (LMs). To achieve this goal, we re-examine existing detection approaches based on the self-consistency of LMs and uncover two types of hallucinations resulting from 1) question-level and 2) model-level, which cannot be effectively identified through self-consistency check alone. Building upon this discovery, we propose a novel sampling-based method, i.e., semantic-aware cross-check consistency (SAC$^3$) that expands on the principle of self-consistency checking. Our SAC$^3$ approach incorporates additional mechanisms to detect both question-level and model-level hallucinations by leveraging advances including semantically equivalent question perturbation and cross-model response consistency checking. Through extensive and systematic empirical analysis, we demonstrate that SAC$^3$ outperforms the state of the art in detecting both non-factual and factual statements across multiple question-answering and open-domain generation benchmarks.

{{</citation>}}


### (37/81) Proto-lm: A Prototypical Network-Based Framework for Built-in Interpretability in Large Language Models (Sean Xie et al., 2023)

{{<citation>}}

Sean Xie, Soroush Vosoughi, Saeed Hassanpour. (2023)  
**Proto-lm: A Prototypical Network-Based Framework for Built-in Interpretability in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2311.01732v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have significantly advanced the field of Natural Language Processing (NLP), but their lack of interpretability has been a major concern. Current methods for interpreting LLMs are post hoc, applied after inference time, and have limitations such as their focus on low-level features and lack of explainability at higher level text units. In this work, we introduce proto-lm, a prototypical network-based white-box framework that allows LLMs to learn immediately interpretable embeddings during the fine-tuning stage while maintaining competitive performance. Our method's applicability and interpretability are demonstrated through experiments on a wide range of NLP tasks, and our results indicate a new possibility of creating interpretable models without sacrificing performance. This novel approach to interpretability in LLMs can pave the way for more interpretable models without the need to sacrifice performance.

{{</citation>}}


### (38/81) An Empirical Study of Benchmarking Chinese Aspect Sentiment Quad Prediction (Junxian Zhou et al., 2023)

{{<citation>}}

Junxian Zhou, Haiqin Yang, Ye Junpeng, Yuxuan He, Hao Mou. (2023)  
**An Empirical Study of Benchmarking Chinese Aspect Sentiment Quad Prediction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, Transformer  
[Paper Link](http://arxiv.org/abs/2311.01713v1)  

---


**ABSTRACT**  
Aspect sentiment quad prediction (ASQP) is a critical subtask of aspect-level sentiment analysis. Current ASQP datasets are characterized by their small size and low quadruple density, which hinders technical development. To expand capacity, we construct two large Chinese ASQP datasets crawled from multiple online platforms. The datasets hold several significant characteristics: larger size (each with 10,000+ samples) and rich aspect categories, more words per sentence, and higher density than existing ASQP datasets. Moreover, we are the first to evaluate the performance of Generative Pre-trained Transformer (GPT) series models on ASQP and exhibit potential issues. The experiments with state-of-the-art ASQP baselines underscore the need to explore additional techniques to address ASQP, as well as the importance of further investigation into methods to improve the performance of GPTs.

{{</citation>}}


### (39/81) A New Korean Text Classification Benchmark for Recognizing the Political Intents in Online Newspapers (Beomjune Kim et al., 2023)

{{<citation>}}

Beomjune Kim, Eunsun Lee, Dongbin Na. (2023)  
**A New Korean Text Classification Benchmark for Recognizing the Political Intents in Online Newspapers**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Text Classification  
[Paper Link](http://arxiv.org/abs/2311.01712v1)  

---


**ABSTRACT**  
Many users reading online articles in various magazines may suffer considerable difficulty in distinguishing the implicit intents in texts. In this work, we focus on automatically recognizing the political intents of a given online newspaper by understanding the context of the text. To solve this task, we present a novel Korean text classification dataset that contains various articles. We also provide deep-learning-based text classification baseline models trained on the proposed dataset. Our dataset contains 12,000 news articles that may contain political intentions, from the politics section of six of the most representative newspaper organizations in South Korea. All the text samples are labeled simultaneously in two aspects (1) the level of political orientation and (2) the level of pro-government. To the best of our knowledge, our paper is the most large-scale Korean news dataset that contains long text and addresses multi-task classification problems. We also train recent state-of-the-art (SOTA) language models that are based on transformer architectures and demonstrate that the trained models show decent text classification performance. All the codes, datasets, and trained models are available at https://github.com/Kdavid2355/KoPolitic-Benchmark-Dataset.

{{</citation>}}


### (40/81) Data-Free Distillation of Language Model by Text-to-Text Transfer (Zheyuan Bai et al., 2023)

{{<citation>}}

Zheyuan Bai, Xinduo Liu, Hailin Hu, Tianyu Guo, Qinghua Zhang, Yunhe Wang. (2023)  
**Data-Free Distillation of Language Model by Text-to-Text Transfer**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Knowledge Distillation, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2311.01689v1)  

---


**ABSTRACT**  
Data-Free Knowledge Distillation (DFKD) plays a vital role in compressing the model when original training data is unavailable. Previous works for DFKD in NLP mainly focus on distilling encoder-only structures like BERT on classification tasks, which overlook the notable progress of generative language modeling. In this work, we propose a novel DFKD framework, namely DFKD-T$^{3}$, where the pretrained generative language model can also serve as a controllable data generator for model compression. This novel framework DFKD-T$^{3}$ leads to an end-to-end learnable text-to-text framework to transform the general domain corpus to compression-friendly task data, targeting to improve both the \textit{specificity} and \textit{diversity}. Extensive experiments show that our method can boost the distillation performance in various downstream tasks such as sentiment analysis, linguistic acceptability, and information extraction. Furthermore, we show that the generated texts can be directly used for distilling other language models and outperform the SOTA methods, making our method more appealing in a general DFKD setting. Our code is available at https://gitee.com/mindspore/models/tree/master/research/nlp/DFKD\_T3.

{{</citation>}}


### (41/81) CASE: Commonsense-Augmented Score with an Expanded Answer Space (Wenkai Chen et al., 2023)

{{<citation>}}

Wenkai Chen, Sahithya Ravi, Vered Shwartz. (2023)  
**CASE: Commonsense-Augmented Score with an Expanded Answer Space**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, QA  
[Paper Link](http://arxiv.org/abs/2311.01684v1)  

---


**ABSTRACT**  
LLMs have demonstrated impressive zero-shot performance on NLP tasks thanks to the knowledge they acquired in their training. In multiple-choice QA tasks, the LM probabilities are used as an imperfect measure of the plausibility of each answer choice. One of the major limitations of the basic score is that it treats all words as equally important. We propose CASE, a Commonsense-Augmented Score with an Expanded Answer Space. CASE addresses this limitation by assigning importance weights for individual words based on their semantic relations to other words in the input. The dynamic weighting approach outperforms basic LM scores, not only because it reduces noise from unimportant words, but also because it informs the model of implicit commonsense knowledge that may be useful for answering the question. We then also follow prior work in expanding the answer space by generating lexically-divergent answers that are conceptually-similar to the choices. When combined with answer space expansion, our method outperforms strong baselines on 5 commonsense benchmarks. We further show these two approaches are complementary and may be especially beneficial when using smaller LMs.

{{</citation>}}


### (42/81) DialogBench: Evaluating LLMs as Human-like Dialogue Systems (Jiao Ou et al., 2023)

{{<citation>}}

Jiao Ou, Junda Lu, Che Liu, Yihong Tang, Fuzheng Zhang, Di Zhang, Zhongyuan Wang, Kun Gai. (2023)  
**DialogBench: Evaluating LLMs as Human-like Dialogue Systems**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Dialog, Dialogue, GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2311.01677v1)  

---


**ABSTRACT**  
Large language models (LLMs) have achieved remarkable breakthroughs in new dialogue capabilities, refreshing human's impressions on dialogue systems. The long-standing goal of dialogue systems is to be human-like enough to establish long-term connections with users by satisfying the need for communication, affection and social belonging. Therefore, there has been an urgent need to evaluate LLMs as human-like dialogue systems. In this paper, we propose DialogBench, a dialogue evaluation benchmark that currently contains $12$ dialogue tasks to assess the capabilities of LLMs as human-like dialogue systems should have. Specifically, we prompt GPT-4 to generate evaluation instances for each task. We first design the basic prompt based on widely-used design principles and further mitigate the existing biases to generate higher-quality evaluation instances. Our extensive test over $28$ LLMs (including pre-trained and supervised instruction-tuning) shows that instruction fine-tuning benefits improve the human likeness of LLMs to a certain extent, but there is still much room to improve those capabilities for most LLMs as human-like dialogue systems. In addition, experimental results also indicate that LLMs perform differently in various abilities that human-like dialogue systems should have. We will publicly release DialogBench, along with the associated evaluation code for the broader research community.

{{</citation>}}


### (43/81) MARRS: Multimodal Reference Resolution System (Halim Cagri Ates et al., 2023)

{{<citation>}}

Halim Cagri Ates, Shruti Bhargava, Site Li, Jiarui Lu, Siddhardha Maddula, Joel Ruben Antony Moniz, Anil Kumar Nalamalapu, Roman Hoang Nguyen, Melis Ozyildirim, Alkesh Patel, Dhivya Piraviperumal, Vincent Renkens, Ankit Samal, Thy Tran, Bo-Hsiang Tseng, Hong Yu, Yuan Zhang, Rong Zou. (2023)  
**MARRS: Multimodal Reference Resolution System**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Natural Language Understanding  
[Paper Link](http://arxiv.org/abs/2311.01650v1)  

---


**ABSTRACT**  
Successfully handling context is essential for any dialog understanding task. This context maybe be conversational (relying on previous user queries or system responses), visual (relying on what the user sees, for example, on their screen), or background (based on signals such as a ringing alarm or playing music). In this work, we present an overview of MARRS, or Multimodal Reference Resolution System, an on-device framework within a Natural Language Understanding system, responsible for handling conversational, visual and background context. In particular, we present different machine learning models to enable handing contextual queries; specifically, one to enable reference resolution, and one to handle context via query rewriting. We also describe how these models complement each other to form a unified, coherent, lightweight system that can understand context while preserving user privacy.

{{</citation>}}


## cs.LG (15)



### (44/81) Active Learning-Based Species Range Estimation (Christian Lange et al., 2023)

{{<citation>}}

Christian Lange, Elijah Cole, Grant Van Horn, Oisin Mac Aodha. (2023)  
**Active Learning-Based Species Range Estimation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2311.02061v1)  

---


**ABSTRACT**  
We propose a new active learning approach for efficiently estimating the geographic range of a species from a limited number of on the ground observations. We model the range of an unmapped species of interest as the weighted combination of estimated ranges obtained from a set of different species. We show that it is possible to generate this candidate set of ranges by using models that have been trained on large weakly supervised community collected observation data. From this, we develop a new active querying approach that sequentially selects geographic locations to visit that best reduce our uncertainty over an unmapped species' range. We conduct a detailed evaluation of our approach and compare it to existing active learning methods using an evaluation dataset containing expert-derived ranges for one thousand species. Our results demonstrate that our method outperforms alternative active learning methods and approaches the performance of end-to-end trained models, even when only using a fraction of the data. This highlights the utility of active learning via transfer learned spatial representations for species range estimation. It also emphasizes the value of leveraging emerging large-scale crowdsourced datasets, not only for modeling a species' range, but also for actively discovering them.

{{</citation>}}


### (45/81) DeliverAI: Reinforcement Learning Based Distributed Path-Sharing Network for Food Deliveries (Ashman Mehra et al., 2023)

{{<citation>}}

Ashman Mehra, Snehanshu Saha, Vaskar Raychoudhury, Archana Mathur. (2023)  
**DeliverAI: Reinforcement Learning Based Distributed Path-Sharing Network for Food Deliveries**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: AI, Amazon, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.02017v1)  

---


**ABSTRACT**  
Delivery of items from the producer to the consumer has experienced significant growth over the past decade and has been greatly fueled by the recent pandemic. Amazon Fresh, Shopify, UberEats, InstaCart, and DoorDash are rapidly growing and are sharing the same business model of consumer items or food delivery. Existing food delivery methods are sub-optimal because each delivery is individually optimized to go directly from the producer to the consumer via the shortest time path. We observe a significant scope for reducing the costs associated with completing deliveries under the current model. We model our food delivery problem as a multi-objective optimization, where consumer satisfaction and delivery costs, both, need to be optimized. Taking inspiration from the success of ride-sharing in the taxi industry, we propose DeliverAI - a reinforcement learning-based path-sharing algorithm. Unlike previous attempts for path-sharing, DeliverAI can provide real-time, time-efficient decision-making using a Reinforcement learning-enabled agent system. Our novel agent interaction scheme leverages path-sharing among deliveries to reduce the total distance traveled while keeping the delivery completion time under check. We generate and test our methodology vigorously on a simulation setup using real data from the city of Chicago. Our results show that DeliverAI can reduce the delivery fleet size by 12\%, the distance traveled by 13%, and achieve 50% higher fleet utilization compared to the baselines.

{{</citation>}}


### (46/81) Score Models for Offline Goal-Conditioned Reinforcement Learning (Harshit Sikchi et al., 2023)

{{<citation>}}

Harshit Sikchi, Rohan Chitnis, Ahmed Touati, Alborz Geramifard, Amy Zhang, Scott Niekum. (2023)  
**Score Models for Offline Goal-Conditioned Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.02013v1)  

---


**ABSTRACT**  
Offline Goal-Conditioned Reinforcement Learning (GCRL) is tasked with learning to achieve multiple goals in an environment purely from offline datasets using sparse reward functions. Offline GCRL is pivotal for developing generalist agents capable of leveraging pre-existing datasets to learn diverse and reusable skills without hand-engineering reward functions. However, contemporary approaches to GCRL based on supervised learning and contrastive learning are often suboptimal in the offline setting. An alternative perspective on GCRL optimizes for occupancy matching, but necessitates learning a discriminator, which subsequently serves as a pseudo-reward for downstream RL. Inaccuracies in the learned discriminator can cascade, negatively influencing the resulting policy. We present a novel approach to GCRL under a new lens of mixture-distribution matching, leading to our discriminator-free method: SMORe. The key insight is combining the occupancy matching perspective of GCRL with a convex dual formulation to derive a learning objective that can better leverage suboptimal offline data. SMORe learns scores or unnormalized densities representing the importance of taking an action at a state for reaching a particular goal. SMORe is principled and our extensive experiments on the fully offline GCRL benchmark composed of robot manipulation and locomotion tasks, including high-dimensional observations, show that SMORe can outperform state-of-the-art baselines by a significant margin.

{{</citation>}}


### (47/81) Conditions on Preference Relations that Guarantee the Existence of Optimal Policies (Jonathan Colaco Carr et al., 2023)

{{<citation>}}

Jonathan Colaco Carr, Prakash Panangaden, Doina Precup. (2023)  
**Conditions on Preference Relations that Guarantee the Existence of Optimal Policies**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2311.01990v1)  

---


**ABSTRACT**  
Learning from Preferential Feedback (LfPF) plays an essential role in training Large Language Models, as well as certain types of interactive learning agents. However, a substantial gap exists between the theory and application of LfPF algorithms. Current results guaranteeing the existence of optimal policies in LfPF problems assume that both the preferences and transition dynamics are determined by a Markov Decision Process. We introduce the Direct Preference Process, a new framework for analyzing LfPF problems in partially-observable, non-Markovian environments. Within this framework, we establish conditions that guarantee the existence of optimal policies by considering the ordinal structure of the preferences. Using the von Neumann-Morgenstern Expected Utility Theorem, we show that the Direct Preference Process generalizes the standard reinforcement learning problem. Our findings narrow the gap between the empirical success and theoretical understanding of LfPF algorithms and provide future practitioners with the tools necessary for a more principled design of LfPF agents.

{{</citation>}}


### (48/81) ForecastPFN: Synthetically-Trained Zero-Shot Forecasting (Samuel Dooley et al., 2023)

{{<citation>}}

Samuel Dooley, Gurnoor Singh Khurana, Chirag Mohapatra, Siddartha Naidu, Colin White. (2023)  
**ForecastPFN: Synthetically-Trained Zero-Shot Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2311.01933v1)  

---


**ABSTRACT**  
The vast majority of time-series forecasting approaches require a substantial training dataset. However, many real-life forecasting applications have very little initial observations, sometimes just 40 or fewer. Thus, the applicability of most forecasting methods is restricted in data-sparse commercial applications. While there is recent work in the setting of very limited initial data (so-called `zero-shot' forecasting), its performance is inconsistent depending on the data used for pretraining. In this work, we take a different approach and devise ForecastPFN, the first zero-shot forecasting model trained purely on a novel synthetic data distribution. ForecastPFN is a prior-data fitted network, trained to approximate Bayesian inference, which can make predictions on a new time series dataset in a single forward pass. Through extensive experiments, we show that zero-shot predictions made by ForecastPFN are more accurate and faster compared to state-of-the-art forecasting methods, even when the other methods are allowed to train on hundreds of additional in-distribution data points.

{{</citation>}}


### (49/81) GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling (Tobias Katsch, 2023)

{{<citation>}}

Tobias Katsch. (2023)  
**GateLoop: Fully Data-Controlled Linear Recurrence for Sequence Modeling**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CL, cs-DS, cs-LG, cs.LG  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2311.01927v1)  

---


**ABSTRACT**  
Linear Recurrence has proven to be a powerful tool for modeling long sequences efficiently. In this work, we show that existing models fail to take full advantage of its potential. Motivated by this finding, we develop GateLoop, a foundational sequence model that generalizes linear recurrent models such as S4, S5, LRU and RetNet, by employing data-controlled state transitions. Utilizing this theoretical advance, GateLoop empirically outperforms existing models for auto-regressive language modeling. Our method comes with a low-cost $O(l)$ recurrent mode and an efficient $O(l \log_{2} l)$ parallel mode making use of highly optimized associative scan implementations. Furthermore, we derive an $O(l^2)$ surrogate attention mode, revealing remarkable implications for Transformer and recently proposed architectures. Specifically, we prove that our approach can be interpreted as providing data-controlled relative-positional information to Attention. While many existing models solely rely on data-controlled cumulative sums for context aggregation, our findings suggest that incorporating data-controlled complex cumulative products may be a crucial step towards more powerful sequence models.

{{</citation>}}


### (50/81) Simplifying Transformer Blocks (Bobby He et al., 2023)

{{<citation>}}

Bobby He, Thomas Hofmann. (2023)  
**Simplifying Transformer Blocks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: BERT, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2311.01906v1)  

---


**ABSTRACT**  
A simple design recipe for deep Transformers is to compose identical building blocks. But standard transformer blocks are far from simple, interweaving attention and MLP sub-blocks with skip connections & normalisation layers in precise arrangements. This complexity leads to brittle architectures, where seemingly minor changes can significantly reduce training speed, or render models untrainable.   In this work, we ask to what extent the standard transformer block can be simplified? Combining signal propagation theory and empirical observations, we motivate modifications that allow many block components to be removed with no loss of training speed, including skip connections, projection or value parameters, sequential sub-blocks and normalisation layers. In experiments on both autoregressive decoder-only and BERT encoder-only models, our simplified transformers emulate the per-update training speed and performance of standard transformers, while enjoying 15% faster training throughput, and using 15% fewer parameters.

{{</citation>}}


### (51/81) Domain Randomization via Entropy Maximization (Gabriele Tiboni et al., 2023)

{{<citation>}}

Gabriele Tiboni, Pascal Klink, Jan Peters, Tatiana Tommasi, Carlo D'Eramo, Georgia Chalvatzaki. (2023)  
**Domain Randomization via Entropy Maximization**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-RO, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.01885v1)  

---


**ABSTRACT**  
Varying dynamics parameters in simulation is a popular Domain Randomization (DR) approach for overcoming the reality gap in Reinforcement Learning (RL). Nevertheless, DR heavily hinges on the choice of the sampling distribution of the dynamics parameters, since high variability is crucial to regularize the agent's behavior but notoriously leads to overly conservative policies when randomizing excessively. In this paper, we propose a novel approach to address sim-to-real transfer, which automatically shapes dynamics distributions during training in simulation without requiring real-world data. We introduce DOmain RAndomization via Entropy MaximizatiON (DORAEMON), a constrained optimization problem that directly maximizes the entropy of the training distribution while retaining generalization capabilities. In achieving this, DORAEMON gradually increases the diversity of sampled dynamics parameters as long as the probability of success of the current policy is sufficiently high. We empirically validate the consistent benefits of DORAEMON in obtaining highly adaptive and generalizable policies, i.e. solving the task at hand across the widest range of dynamics parameters, as opposed to representative baselines from the DR literature. Notably, we also demonstrate the Sim2Real applicability of DORAEMON through its successful zero-shot transfer in a robotic manipulation setup under unknown real-world parameters.

{{</citation>}}


### (52/81) TinyFormer: Efficient Transformer Design and Deployment on Tiny Devices (Jianlei Yang et al., 2023)

{{<citation>}}

Jianlei Yang, Jiacheng Liao, Fanding Lei, Meichen Liu, Junyi Chen, Lingkun Long, Han Wan, Bei Yu, Weisheng Zhao. (2023)  
**TinyFormer: Efficient Transformer Design and Deployment on Tiny Devices**  

---
Primary Category: cs.LG  
Categories: cs-AR, cs-LG, cs.LG  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.01759v1)  

---


**ABSTRACT**  
Developing deep learning models on tiny devices (e.g. Microcontroller units, MCUs) has attracted much attention in various embedded IoT applications. However, it is challenging to efficiently design and deploy recent advanced models (e.g. transformers) on tiny devices due to their severe hardware resource constraints. In this work, we propose TinyFormer, a framework specifically designed to develop and deploy resource-efficient transformers on MCUs. TinyFormer mainly consists of SuperNAS, SparseNAS and SparseEngine. Separately, SuperNAS aims to search for an appropriate supernet from a vast search space. SparseNAS evaluates the best sparse single-path model including transformer architecture from the identified supernet. Finally, SparseEngine efficiently deploys the searched sparse models onto MCUs. To the best of our knowledge, SparseEngine is the first deployment framework capable of performing inference of sparse models with transformer on MCUs. Evaluation results on the CIFAR-10 dataset demonstrate that TinyFormer can develop efficient transformers with an accuracy of $96.1\%$ while adhering to hardware constraints of $1$MB storage and $320$KB memory. Additionally, TinyFormer achieves significant speedups in sparse inference, up to $12.2\times$, when compared to the CMSIS-NN library. TinyFormer is believed to bring powerful transformers into TinyML scenarios and greatly expand the scope of deep learning applications.

{{</citation>}}


### (53/81) Epidemic Decision-making System Based Federated Reinforcement Learning (Yangxi Zhou et al., 2023)

{{<citation>}}

Yangxi Zhou, Junping Du, Zhe Xue, Zhenhui Pan, Weikang Chen. (2023)  
**Epidemic Decision-making System Based Federated Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-IR, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.01749v1)  

---


**ABSTRACT**  
Epidemic decision-making can effectively help the government to comprehensively consider public security and economic development to respond to public health and safety emergencies. Epidemic decision-making can effectively help the government to comprehensively consider public security and economic development to respond to public health and safety emergencies. Some studies have shown that intensive learning can effectively help the government to make epidemic decision, thus achieving the balance between health security and economic development. Some studies have shown that intensive learning can effectively help the government to make epidemic decision, thus achieving the balance between health security and economic development. However, epidemic data often has the characteristics of limited samples and high privacy. However, epidemic data often has the characteristics of limited samples and high privacy. This model can combine the epidemic situation data of various provinces for cooperative training to use as an enhanced learning model for epidemic situation decision, while protecting the privacy of data. The experiment shows that the enhanced federated learning can obtain more optimized performance and return than the enhanced learning, and the enhanced federated learning can also accelerate the training convergence speed of the training model. accelerate the training convergence speed of the client. At the same time, through the experimental comparison, A2C is the most suitable reinforcement learning model for the epidemic situation decision-making. learning model for the epidemic situation decision-making scenario, followed by the PPO model, and the performance of DDPG is unsatisfactory.

{{</citation>}}


### (54/81) Heterogeneous federated collaborative filtering using FAIR: Federated Averaging in Random Subspaces (Aditya Desai et al., 2023)

{{<citation>}}

Aditya Desai, Benjamin Meisburger, Zichang Liu, Anshumali Shrivastava. (2023)  
**Heterogeneous federated collaborative filtering using FAIR: Federated Averaging in Random Subspaces**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.01722v1)  

---


**ABSTRACT**  
Recommendation systems (RS) for items (e.g., movies, books) and ads are widely used to tailor content to users on various internet platforms. Traditionally, recommendation models are trained on a central server. However, due to rising concerns for data privacy and regulations like the GDPR, federated learning is an increasingly popular paradigm in which data never leaves the client device. Applying federated learning to recommendation models is non-trivial due to large embedding tables, which often exceed the memory constraints of most user devices. To include data from all devices in federated learning, we must enable collective training of embedding tables on devices with heterogeneous memory capacities. Current solutions to heterogeneous federated learning can only accommodate a small range of capacities and thus limit the number of devices that can participate in training. We present Federated Averaging in Random subspaces (FAIR), which allows arbitrary compression of embedding tables based on device capacity and ensures the participation of all devices in training. FAIR uses what we call consistent and collapsible subspaces defined by hashing-based random projections to jointly train large embedding tables while using varying amounts of compression on user devices. We evaluate FAIR on Neural Collaborative Filtering tasks with multiple datasets and verify that FAIR can gather and share information from a wide range of devices with varying capacities, allowing for seamless collaboration. We prove the convergence of FAIR in the homogeneous setting with non-i.i.d data distribution. Our code is open source at {https://github.com/apd10/FLCF}

{{</citation>}}


### (55/81) Adversarial Attacks on Cooperative Multi-agent Bandits (Jinhang Zuo et al., 2023)

{{<citation>}}

Jinhang Zuo, Zhiyao Zhang, Xuchuang Wang, Cheng Chen, Shuai Li, John C. S. Lui, Mohammad Hajiesmaili, Adam Wierman. (2023)  
**Adversarial Attacks on Cooperative Multi-agent Bandits**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs-MA, cs.LG  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2311.01698v1)  

---


**ABSTRACT**  
Cooperative multi-agent multi-armed bandits (CMA2B) consider the collaborative efforts of multiple agents in a shared multi-armed bandit game. We study latent vulnerabilities exposed by this collaboration and consider adversarial attacks on a few agents with the goal of influencing the decisions of the rest. More specifically, we study adversarial attacks on CMA2B in both homogeneous settings, where agents operate with the same arm set, and heterogeneous settings, where agents have distinct arm sets. In the homogeneous setting, we propose attack strategies that, by targeting just one agent, convince all agents to select a particular target arm $T-o(T)$ times while incurring $o(T)$ attack costs in $T$ rounds. In the heterogeneous setting, we prove that a target arm attack requires linear attack costs and propose attack strategies that can force a maximum number of agents to suffer linear regrets while incurring sublinear costs and only manipulating the observations of a few target agents. Numerical experiments validate the effectiveness of our proposed attack strategies.

{{</citation>}}


### (56/81) Detecting Spurious Correlations via Robust Visual Concepts in Real and AI-Generated Image Classification (Preetam Prabhu Srikar Dammu et al., 2023)

{{<citation>}}

Preetam Prabhu Srikar Dammu, Chirag Shah. (2023)  
**Detecting Spurious Correlations via Robust Visual Concepts in Real and AI-Generated Image Classification**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs.LG  
Keywords: AI, Image Classification  
[Paper Link](http://arxiv.org/abs/2311.01655v1)  

---


**ABSTRACT**  
Often machine learning models tend to automatically learn associations present in the training data without questioning their validity or appropriateness. This undesirable property is the root cause of the manifestation of spurious correlations, which render models unreliable and prone to failure in the presence of distribution shifts. Research shows that most methods attempting to remedy spurious correlations are only effective for a model's known spurious associations. Current spurious correlation detection algorithms either rely on extensive human annotations or are too restrictive in their formulation. Moreover, they rely on strict definitions of visual artifacts that may not apply to data produced by generative models, as they are known to hallucinate contents that do not conform to standard specifications. In this work, we introduce a general-purpose method that efficiently detects potential spurious correlations, and requires significantly less human interference in comparison to the prior art. Additionally, the proposed method provides intuitive explanations while eliminating the need for pixel-level annotations. We demonstrate the proposed method's tolerance to the peculiarity of AI-generated images, which is a considerably challenging task, one where most of the existing methods fall short. Consequently, our method is also suitable for detecting spurious correlations that may propagate to downstream applications originating from generative models.

{{</citation>}}


### (57/81) Calibrate and Boost Logical Expressiveness of GNN Over Multi-Relational and Temporal Graphs (Yeyuan Chen et al., 2023)

{{<citation>}}

Yeyuan Chen, Dingmin Wang. (2023)  
**Calibrate and Boost Logical Expressiveness of GNN Over Multi-Relational and Temporal Graphs**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-LO, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2311.01647v1)  

---


**ABSTRACT**  
As a powerful framework for graph representation learning, Graph Neural Networks (GNNs) have garnered significant attention in recent years. However, to the best of our knowledge, there has been no formal analysis of the logical expressiveness of GNNs as Boolean node classifiers over multi-relational graphs, where each edge carries a specific relation type. In this paper, we investigate $\mathcal{FOC}_2$, a fragment of first-order logic with two variables and counting quantifiers. On the negative side, we demonstrate that the R$^2$-GNN architecture, which extends the local message passing GNN by incorporating global readout, fails to capture $\mathcal{FOC}_2$ classifiers in the general case. Nevertheless, on the positive side, we establish that R$^2$-GNNs models are equivalent to $\mathcal{FOC}_2$ classifiers under certain restricted yet reasonable scenarios. To address the limitations of R$^2$-GNNs regarding expressiveness, we propose a simple graph transformation technique, akin to a preprocessing step, which can be executed in linear time. This transformation enables R$^2$-GNNs to effectively capture any $\mathcal{FOC}_2$ classifiers when applied to the "transformed" input graph. Moreover, we extend our analysis of expressiveness and graph transformation to temporal graphs, exploring several temporal GNN architectures and providing an expressiveness hierarchy for them. To validate our findings, we implement R$^2$-GNNs and the graph transformation technique and conduct empirical tests in node classification tasks against various well-known GNN architectures that support multi-relational or temporal graphs. Our experimental results consistently demonstrate that R$^2$-GNN with the graph transformation outperforms the baseline methods on both synthetic and real-world datasets

{{</citation>}}


### (58/81) Robust Adversarial Reinforcement Learning via Bounded Rationality Curricula (Aryaman Reddi et al., 2023)

{{<citation>}}

Aryaman Reddi, Maximilian Tölle, Jan Peters, Georgia Chalvatzaki, Carlo D'Eramo. (2023)  
**Robust Adversarial Reinforcement Learning via Bounded Rationality Curricula**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: QA, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.01642v1)  

---


**ABSTRACT**  
Robustness against adversarial attacks and distribution shifts is a long-standing goal of Reinforcement Learning (RL). To this end, Robust Adversarial Reinforcement Learning (RARL) trains a protagonist against destabilizing forces exercised by an adversary in a competitive zero-sum Markov game, whose optimal solution, i.e., rational strategy, corresponds to a Nash equilibrium. However, finding Nash equilibria requires facing complex saddle point optimization problems, which can be prohibitive to solve, especially for high-dimensional control. In this paper, we propose a novel approach for adversarial RL based on entropy regularization to ease the complexity of the saddle point optimization problem. We show that the solution of this entropy-regularized problem corresponds to a Quantal Response Equilibrium (QRE), a generalization of Nash equilibria that accounts for bounded rationality, i.e., agents sometimes play random actions instead of optimal ones. Crucially, the connection between the entropy-regularized objective and QRE enables free modulation of the rationality of the agents by simply tuning the temperature coefficient. We leverage this insight to propose our novel algorithm, Quantal Adversarial RL (QARL), which gradually increases the rationality of the adversary in a curriculum fashion until it is fully rational, easing the complexity of the optimization problem while retaining robustness. We provide extensive evidence of QARL outperforming RARL and recent baselines across several MuJoCo locomotion and navigation problems in overall performance and robustness.

{{</citation>}}


## cs.AI (3)



### (59/81) APRICOT: Acuity Prediction in Intensive Care Unit (ICU): Predicting Stability, Transitions, and Life-Sustaining Therapies (Miguel Contreras et al., 2023)

{{<citation>}}

Miguel Contreras, Brandon Silva, Benjamin Shickel, Tezcan Ozrazgat Baslanti, Yuanfang Ren, Ziyuan Guan, Sabyasachi Bandyopadhyay, Kia Khezeli, Azra Bihorac, Parisa Rashidi. (2023)  
**APRICOT: Acuity Prediction in Intensive Care Unit (ICU): Predicting Stability, Transitions, and Life-Sustaining Therapies**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.02026v1)  

---


**ABSTRACT**  
The acuity state of patients in the intensive care unit (ICU) can quickly change from stable to unstable, sometimes leading to life-threatening conditions. Early detection of deteriorating conditions can result in providing more timely interventions and improved survival rates. Current approaches rely on manual daily assessments. Some data-driven approaches have been developed, that use mortality as a proxy of acuity in the ICU. However, these methods do not integrate acuity states to determine the stability of a patient or the need for life-sustaining therapies. In this study, we propose APRICOT (Acuity Prediction in Intensive Care Unit), a Transformer-based neural network to predict acuity state in real-time in ICU patients. We develop and extensively validate externally, temporally, and prospectively the APRICOT model on three large datasets: University of Florida Health (UFH), eICU Collaborative Research Database (eICU), and Medical Information Mart for Intensive Care (MIMIC)-IV. The performance of APRICOT shows comparable results to state-of-the-art mortality prediction models (external AUROC 0.93-0.93, temporal AUROC 0.96-0.98, and prospective AUROC 0.98) as well as acuity prediction models (external AUROC 0.80-0.81, temporal AUROC 0.77-0.78, and prospective AUROC 0.87). Furthermore, APRICOT can make predictions for the need for life-sustaining therapies, showing comparable results to state-of-the-art ventilation prediction models (external AUROC 0.80-0.81, temporal AUROC 0.87-0.88, and prospective AUROC 0.85), and vasopressor prediction models (external AUROC 0.82-0.83, temporal AUROC 0.73-0.75, prospective AUROC 0.87). This tool allows for real-time acuity monitoring of a patient and can provide helpful information to clinicians to make timely interventions. Furthermore, the model can suggest life-sustaining therapies that the patient might need in the next hours in the ICU.

{{</citation>}}


### (60/81) Active Reasoning in an Open-World Environment (Manjie Xu et al., 2023)

{{<citation>}}

Manjie Xu, Guangyuan Jiang, Wei Liang, Chi Zhang, Yixin Zhu. (2023)  
**Active Reasoning in an Open-World Environment**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CV, cs.AI  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2311.02018v1)  

---


**ABSTRACT**  
Recent advances in vision-language learning have achieved notable success on complete-information question-answering datasets through the integration of extensive world knowledge. Yet, most models operate passively, responding to questions based on pre-stored knowledge. In stark contrast, humans possess the ability to actively explore, accumulate, and reason using both newfound and existing information to tackle incomplete-information questions. In response to this gap, we introduce $Conan$, an interactive open-world environment devised for the assessment of active reasoning. $Conan$ facilitates active exploration and promotes multi-round abductive inference, reminiscent of rich, open-world settings like Minecraft. Diverging from previous works that lean primarily on single-round deduction via instruction following, $Conan$ compels agents to actively interact with their surroundings, amalgamating new evidence with prior knowledge to elucidate events from incomplete observations. Our analysis on $Conan$ underscores the shortcomings of contemporary state-of-the-art models in active exploration and understanding complex scenarios. Additionally, we explore Abduction from Deduction, where agents harness Bayesian rules to recast the challenge of abduction as a deductive process. Through $Conan$, we aim to galvanize advancements in active reasoning and set the stage for the next generation of artificial intelligence agents adept at dynamically engaging in environments.

{{</citation>}}


### (61/81) Supermind Ideator: Exploring generative AI to support creative problem-solving (Steven R. Rick et al., 2023)

{{<citation>}}

Steven R. Rick, Gianni Giacomelli, Haoran Wen, Robert J. Laubacher, Nancy Taubenslag, Jennifer L. Heyman, Max Sina Knicker, Younes Jeddi, Hendrik Maier, Stephen Dwyer, Pranav Ragupathy, Thomas W. Malone. (2023)  
**Supermind Ideator: Exploring generative AI to support creative problem-solving**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-HC, cs.AI  
Keywords: AI, GPT  
[Paper Link](http://arxiv.org/abs/2311.01937v1)  

---


**ABSTRACT**  
Previous efforts to support creative problem-solving have included (a) techniques (such as brainstorming and design thinking) to stimulate creative ideas, and (b) software tools to record and share these ideas. Now, generative AI technologies can suggest new ideas that might never have occurred to the users, and users can then select from these ideas or use them to stimulate even more ideas. Here, we describe such a system, Supermind Ideator. The system uses a large language model (GPT 3.5) and adds prompting, fine tuning, and a user interface specifically designed to help people use creative problem-solving techniques. Some of these techniques can be applied to any problem; others are specifically intended to help generate innovative ideas about how to design groups of people and/or computers ("superminds"). We also describe our early experiences with using this system and suggest ways it could be extended to support additional techniques for other specific problem-solving domains.

{{</citation>}}


## eess.IV (4)



### (62/81) A Structured Pruning Algorithm for Model-based Deep Learning (Chicago Park et al., 2023)

{{<citation>}}

Chicago Park, Weijie Gan, Zihao Zou, Yuyang Hu, Zhixin Sun, Ulugbek S. Kamilov. (2023)  
**A Structured Pruning Algorithm for Model-based Deep Learning**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2311.02003v1)  

---


**ABSTRACT**  
There is a growing interest in model-based deep learning (MBDL) for solving imaging inverse problems. MBDL networks can be seen as iterative algorithms that estimate the desired image using a physical measurement model and a learned image prior specified using a convolutional neural net (CNNs). The iterative nature of MBDL networks increases the test-time computational complexity, which limits their applicability in certain large-scale applications. We address this issue by presenting structured pruning algorithm for model-based deep learning (SPADE) as the first structured pruning algorithm for MBDL networks. SPADE reduces the computational complexity of CNNs used within MBDL networks by pruning its non-essential weights. We propose three distinct strategies to fine-tune the pruned MBDL networks to minimize the performance loss. Each fine-tuning strategy has a unique benefit that depends on the presence of a pre-trained model and a high-quality ground truth. We validate SPADE on two distinct inverse problems, namely compressed sensing MRI and image super-resolution. Our results highlight that MBDL models pruned by SPADE can achieve substantial speed up in testing time while maintaining competitive performance.

{{</citation>}}


### (63/81) LLM-driven Multimodal Target Volume Contouring in Radiation Oncology (Yujin Oh et al., 2023)

{{<citation>}}

Yujin Oh, Sangjoon Park, Hwa Kyung Byun, Jin Sung Kim, Jong Chul Ye. (2023)  
**LLM-driven Multimodal Target Volume Contouring in Radiation Oncology**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.01908v1)  

---


**ABSTRACT**  
Target volume contouring for radiation therapy is considered significantly more challenging than the normal organ segmentation tasks as it necessitates the utilization of both image and text-based clinical information. Inspired by the recent advancement of large language models (LLMs) that can facilitate the integration of the textural information and images, here we present a novel LLM-driven multi-modal AI that utilizes the clinical text information and is applicable to the challenging task of target volume contouring for radiation therapy, and validate it within the context of breast cancer radiation therapy target volume contouring. Using external validation and data-insufficient environments, which attributes highly conducive to real-world applications, we demonstrate that the proposed model exhibits markedly improved performance compared to conventional vision-only AI models, particularly exhibiting robust generalization performance and data-efficiency. To our best knowledge, this is the first LLM-driven multimodal AI model that integrates the clinical text information into target volume delineation for radiation oncology.

{{</citation>}}


### (64/81) Simulation of acquisition shifts in T2 Flair MR images to stress test AI segmentation networks (Christiane Posselt et al., 2023)

{{<citation>}}

Christiane Posselt, Mehmet Yigit Avci, Mehmet Yigitsoy, Patrick Schünke, Christoph Kolbitsch, Tobias Schäffter, Stefanie Remmele. (2023)  
**Simulation of acquisition shifts in T2 Flair MR images to stress test AI segmentation networks**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.01894v1)  

---


**ABSTRACT**  
Purpose: To provide a simulation framework for routine neuroimaging test data, which allows for "stress testing" of deep segmentation networks against acquisition shifts that commonly occur in clinical practice for T2 weighted (T2w) fluid attenuated inversion recovery (FLAIR) Magnetic Resonance Imaging (MRI) protocols.   Approach: The approach simulates "acquisition shift derivatives" of MR images based on MR signal equations. Experiments comprise the validation of the simulated images by real MR scans and example stress tests on state-of-the-art MS lesion segmentation networks to explore a generic model function to describe the F1 score in dependence of the contrast-affecting sequence parameters echo time (TE) and inversion time (TI).   Results: The differences between real and simulated images range up to 19 % in gray and white matter for extreme parameter settings. For the segmentation networks under test the F1 score dependency on TE and TI can be well described by quadratic model functions (R^2 > 0.9). The coefficients of the model functions indicate that changes of TE have more influence on the model performance than TI.   Conclusions: We show that these deviations are in the range of values as may be caused by erroneous or individual differences of relaxation times as described by literature. The coefficients of the F1 model function allow for quantitative comparison of the influences of TE and TI. Limitations arise mainly from tissues with the low baseline signal (like CSF) and when the protocol contains contrast-affecting measures that cannot be modelled due to missing information in the DICOM header.

{{</citation>}}


### (65/81) Capturing Local and Global Features in Medical Images by Using Ensemble CNN-Transformer (Javad Mirzapour Kaleybar et al., 2023)

{{<citation>}}

Javad Mirzapour Kaleybar, Hooman Saadat, Hooman Khaloo. (2023)  
**Capturing Local and Global Features in Medical Images by Using Ensemble CNN-Transformer**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.01731v1)  

---


**ABSTRACT**  
This paper introduces a groundbreaking classification model called the Controllable Ensemble Transformer and CNN (CETC) for the analysis of medical images. The CETC model combines the powerful capabilities of convolutional neural networks (CNNs) and transformers to effectively capture both local and global features present in medical images. The model architecture comprises three main components: a convolutional encoder block (CEB), a transposed-convolutional decoder block (TDB), and a transformer classification block (TCB). The CEB is responsible for capturing multi-local features at different scales and draws upon components from VGGNet, ResNet, and MobileNet as backbones. By leveraging this combination, the CEB is able to effectively detect and encode local features. The TDB, on the other hand, consists of sub-decoders that decode and sum the captured features using ensemble coefficients. This enables the model to efficiently integrate the information from multiple scales. Finally, the TCB utilizes the SwT backbone and a specially designed prediction head to capture global features, ensuring a comprehensive understanding of the entire image. The paper provides detailed information on the experimental setup and implementation, including the use of transfer learning, data preprocessing techniques, and training settings. The CETC model is trained and evaluated using two publicly available COVID-19 datasets. Remarkably, the model outperforms existing state-of-the-art models across various evaluation metrics. The experimental results clearly demonstrate the superiority of the CETC model, emphasizing its potential for accurately and efficiently analyzing medical images.

{{</citation>}}


## cs.RO (2)



### (66/81) RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches (Jiayuan Gu et al., 2023)

{{<citation>}}

Jiayuan Gu, Sean Kirmani, Paul Wohlhart, Yao Lu, Montserrat Gonzalez Arenas, Kanishka Rao, Wenhao Yu, Chuyuan Fu, Keerthana Gopalakrishnan, Zhuo Xu, Priya Sundaresan, Peng Xu, Hao Su, Karol Hausman, Chelsea Finn, Quan Vuong, Ted Xiao. (2023)  
**RT-Trajectory: Robotic Task Generalization via Hindsight Trajectory Sketches**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2311.01977v1)  

---


**ABSTRACT**  
Generalization remains one of the most important desiderata for robust robot learning systems. While recently proposed approaches show promise in generalization to novel objects, semantic concepts, or visual distribution shifts, generalization to new tasks remains challenging. For example, a language-conditioned policy trained on pick-and-place tasks will not be able to generalize to a folding task, even if the arm trajectory of folding is similar to pick-and-place. Our key insight is that this kind of generalization becomes feasible if we represent the task through rough trajectory sketches. We propose a policy conditioning method using such rough trajectory sketches, which we call RT-Trajectory, that is practical, easy to specify, and allows the policy to effectively perform new tasks that would otherwise be challenging to perform. We find that trajectory sketches strike a balance between being detailed enough to express low-level motion-centric guidance while being coarse enough to allow the learned policy to interpret the trajectory sketch in the context of situational visual observations. In addition, we show how trajectory sketches can provide a useful interface to communicate with robotic policies: they can be specified through simple human inputs like drawings or videos, or through automated methods such as modern image-generating or waypoint-generating methods. We evaluate RT-Trajectory at scale on a variety of real-world robotic tasks, and find that RT-Trajectory is able to perform a wider range of tasks compared to language-conditioned and goal-conditioned policies, when provided the same training data.

{{</citation>}}


### (67/81) Depth-guided Free-space Segmentation for a Mobile Robot (Christos Sevastopoulos et al., 2023)

{{<citation>}}

Christos Sevastopoulos, Joey Hussain, Stasinos Konstantopoulos, Vangelis Karkaletsis, Fillia Makedon. (2023)  
**Depth-guided Free-space Segmentation for a Mobile Robot**  

---
Primary Category: cs.RO  
Categories: cs-CV, cs-RO, cs.RO  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2311.01966v1)  

---


**ABSTRACT**  
Accurate indoor free-space segmentation is a challenging task due to the complexity and the dynamic nature that indoor environments exhibit. We propose an indoors free-space segmentation method that associates large depth values with navigable regions. Our method leverages an unsupervised masking technique that, using positive instances, generates segmentation labels based on textural homogeneity and depth uniformity. Moreover, we generate superpixels corresponding to areas of higher depth and align them with features extracted from a Dense Prediction Transformer (DPT). Using the estimated free-space masks and the DPT feature representation, a SegFormer model is fine-tuned on our custom-collected indoor dataset. Our experiments demonstrate sufficient performance in intricate scenarios characterized by cluttered obstacles and challenging identification of free space.

{{</citation>}}


## cs.CR (3)



### (68/81) Architecture of Smart Certificates for Web3 Applications Against Cyberthreats in Financial Industry (Stefan Kambiz Behfar et al., 2023)

{{<citation>}}

Stefan Kambiz Behfar, Jon Crowcroft. (2023)  
**Architecture of Smart Certificates for Web3 Applications Against Cyberthreats in Financial Industry**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Financial  
[Paper Link](http://arxiv.org/abs/2311.01956v1)  

---


**ABSTRACT**  
This study addresses the security challenges associated with the current internet transformations, specifically focusing on emerging technologies such as blockchain and decentralized storage. It also investigates the role of Web3 applications in shaping the future of the internet. The primary objective is to propose a novel design for 'smart certificates,' which are digital certificates that can be programmatically enforced. Utilizing such certificates, an enterprise can better protect itself from cyberattacks and ensure the security of its data and systems. Web3 recent security solutions by companies and projects like Certik, Forta, Slither, and Securify are the equivalent of code scanning tool that were originally developed for Web1 and Web2 applications, and definitely not like certificates to help enterprises feel safe against cyberthreats. We aim to improve the resilience of enterprises' digital infrastructure by building on top of Web3 application and put methodologies in place for vulnerability analysis and attack correlation, focusing on architecture of different layers, Wallet/Client, Application and Smart Contract, where specific components are provided to identify and predict threats and risks. Furthermore, Certificate Transparency is used for enhancing the security, trustworthiness and decentralized management of the certificates, and detecting misuses, compromises, and malfeasances.

{{</citation>}}


### (69/81) CoPriv: Network/Protocol Co-Optimization for Communication-Efficient Private Inference (Wenxuan Zeng et al., 2023)

{{<citation>}}

Wenxuan Zeng, Meng Li, Haichuan Yang, Wen-jie Lu, Runsheng Wang, Ru Huang. (2023)  
**CoPriv: Network/Protocol Co-Optimization for Communication-Efficient Private Inference**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2311.01737v1)  

---


**ABSTRACT**  
Deep neural network (DNN) inference based on secure 2-party computation (2PC) can offer cryptographically-secure privacy protection but suffers from orders of magnitude latency overhead due to enormous communication. Previous works heavily rely on a proxy metric of ReLU counts to approximate the communication overhead and focus on reducing the ReLUs to improve the communication efficiency. However, we observe these works achieve limited communication reduction for state-of-the-art (SOTA) 2PC protocols due to the ignorance of other linear and non-linear operations, which now contribute to the majority of communication. In this work, we present CoPriv, a framework that jointly optimizes the 2PC inference protocol and the DNN architecture. CoPriv features a new 2PC protocol for convolution based on Winograd transformation and develops DNN-aware optimization to significantly reduce the inference communication. CoPriv further develops a 2PC-aware network optimization algorithm that is compatible with the proposed protocol and simultaneously reduces the communication for all the linear and non-linear operations. We compare CoPriv with the SOTA 2PC protocol, CrypTFlow2, and demonstrate 2.1x communication reduction for both ResNet-18 and ResNet-32 on CIFAR-100. We also compare CoPriv with SOTA network optimization methods, including SNL, MetaPruning, etc. CoPriv achieves 9.98x and 3.88x online and total communication reduction with a higher accuracy compare to SNL, respectively. CoPriv also achieves 3.87x online communication reduction with more than 3% higher accuracy compared to MetaPruning.

{{</citation>}}


### (70/81) Universal Perturbation-based Secret Key-Controlled Data Hiding (Donghua Wang et al., 2023)

{{<citation>}}

Donghua Wang, Wen Yao, Tingsong Jiang, Xiaoqian Chen. (2023)  
**Universal Perturbation-based Secret Key-Controlled Data Hiding**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-CV, cs.CR  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2311.01696v1)  

---


**ABSTRACT**  
Deep neural networks (DNNs) are demonstrated to be vulnerable to universal perturbation, a single quasi-perceptible perturbation that can deceive the DNN on most images. However, the previous works are focused on using universal perturbation to perform adversarial attacks, while the potential usability of universal perturbation as data carriers in data hiding is less explored, especially for the key-controlled data hiding method. In this paper, we propose a novel universal perturbation-based secret key-controlled data-hiding method, realizing data hiding with a single universal perturbation and data decoding with the secret key-controlled decoder. Specifically, we optimize a single universal perturbation, which serves as a data carrier that can hide multiple secret images and be added to most cover images. Then, we devise a secret key-controlled decoder to extract different secret images from the single container image constructed by the universal perturbation by using different secret keys. Moreover, a suppress loss function is proposed to prevent the secret image from leakage. Furthermore, we adopt a robust module to boost the decoder's capability against corruption. Finally, A co-joint optimization strategy is proposed to find the optimal universal perturbation and decoder. Extensive experiments are conducted on different datasets to demonstrate the effectiveness of the proposed method. Additionally, the physical test performed on platforms (e.g., WeChat and Twitter) verifies the usability of the proposed method in practice.

{{</citation>}}


## cs.HC (1)



### (71/81) ChartGPT: Leveraging LLMs to Generate Charts from Abstract Natural Language (Yuan Tian et al., 2023)

{{<citation>}}

Yuan Tian, Weiwei Cui, Dazhen Deng, Xinjing Yi, Yurun Yang, Haidong Zhang, Yingcai Wu. (2023)  
**ChartGPT: Leveraging LLMs to Generate Charts from Abstract Natural Language**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: GPT, NLI  
[Paper Link](http://arxiv.org/abs/2311.01920v1)  

---


**ABSTRACT**  
The use of natural language interfaces (NLIs) for the creation of charts is becoming increasingly popular due to the intuitiveness of natural language interactions. One key challenge in this approach is to accurately capture user intents and transform them to proper chart specifications. This obstructs the wide use of NLI in chart generation, as users' natural language inputs are generally abstract (i.e., ambiguous or under-specified), without a clear specification of visual encodings. Recently, pre-trained large language models (LLMs) have exhibited superior performance in understanding and generating natural language, demonstrating great potential for downstream tasks. Inspired by this major trend, we propose ChartGPT, generating charts from abstract natural language inputs. However, LLMs are struggling to address complex logic problems. To enable the model to accurately specify the complex parameters and perform operations in chart generation, we decompose the generation process into a step-by-step reasoning pipeline, so that the model only needs to reason a single and specific sub-task during each run. Moreover, LLMs are pre-trained on general datasets, which might be biased for the task of chart generation. To provide adequate visualization knowledge, we create a dataset consisting of abstract utterances and charts and improve model performance through fine-tuning. We further design an interactive interface for ChartGPT that allows users to check and modify the intermediate outputs of each step. The effectiveness of the proposed system is evaluated through quantitative evaluations and a user study.

{{</citation>}}


## cs.IR (1)



### (72/81) Enhancing search engine precision and user experience through sentiment-based polysemy resolution (Mike Nkongolo, 2023)

{{<citation>}}

Mike Nkongolo. (2023)  
**Enhancing search engine precision and user experience through sentiment-based polysemy resolution**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: NLP, Reasoning  
[Paper Link](http://arxiv.org/abs/2311.01895v1)  

---


**ABSTRACT**  
With the proliferation of digital content and the need for efficient information retrieval, this study's insights can be applied to various domains, including news services, e-commerce, and digital marketing, to provide users with more meaningful and tailored experiences. The study addresses the common problem of polysemy in search engines, where the same keyword may have multiple meanings. It proposes a solution to this issue by embedding a smart search function into the search engine, which can differentiate between different meanings based on sentiment. The study leverages sentiment analysis, a powerful natural language processing (NLP) technique, to classify and categorize news articles based on their emotional tone. This can provide more insightful and nuanced search results. The article reports an impressive accuracy rate of 85% for the proposed smart search function, which outperforms conventional search engines. This indicates the effectiveness of the sentiment-based approach. The research explores multiple sentiment analysis models, including Sentistrength and Valence Aware Dictionary for Sentiment Reasoning (VADER), to determine the best-performing approach. The findings can be applied to enhance search engines, making them more capable of understanding the context and intent behind users 'queries. This can lead to better search results that are more aligned with what users are looking for. The proposed smart search function can improve the user experience by reducing the need to sift through irrelevant search results. This is particularly important in an age where information overload is common.

{{</citation>}}


## cs.CY (1)



### (73/81) When fairness is an abstraction: Equity and AI in Swedish compulsory education (Marie Utterberg Modén et al., 2023)

{{<citation>}}

Marie Utterberg Modén, Marisa Ponti, Johan Lundin, Martin Tallvid. (2023)  
**When fairness is an abstraction: Equity and AI in Swedish compulsory education**  

---
Primary Category: cs.CY  
Categories: K-3; K-4, cs-CY, cs.CY  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.01838v1)  

---


**ABSTRACT**  
Artificial intelligence experts often question whether AI is fair. They view fairness as a property of AI systems rather than of sociopolitical and economic systems. This paper emphasizes the need to be fair in the social, political, and economic contexts within which an educational system operates and uses AI. Taking Swedish decentralized compulsory education as the context, this paper examines whether and how the use of AI envisaged by national authorities and edtech companies exacerbates unfairness. A qualitative content analysis of selected Swedish policy documents and edtech reports was conducted using the concept of relevant social groups to understand how different groups view the risks and benefits of AI for fairness. Three groups that view efficiency as a key value of AI are identified, and interpreted as economical, pedagogical and accessibility-related. By separating fairness from social justice, this paper challenges the notion of fairness as the formal equality of opportunities.

{{</citation>}}


## cs.DC (2)



### (74/81) Large Language Models to the Rescue: Reducing the Complexity in Scientific Workflow Development Using ChatGPT (Mario Sänger et al., 2023)

{{<citation>}}

Mario Sänger, Ninon De Mecquenem, Katarzyna Ewa Lewińska, Vasilis Bountris, Fabian Lehmann, Ulf Leser, Thomas Kosch. (2023)  
**Large Language Models to the Rescue: Reducing the Complexity in Scientific Workflow Development Using ChatGPT**  

---
Primary Category: cs.DC  
Categories: cs-CL, cs-DC, cs-HC, cs.DC  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2311.01825v1)  

---


**ABSTRACT**  
Scientific workflow systems are increasingly popular for expressing and executing complex data analysis pipelines over large datasets, as they offer reproducibility, dependability, and scalability of analyses by automatic parallelization on large compute clusters. However, implementing workflows is difficult due to the involvement of many black-box tools and the deep infrastructure stack necessary for their execution. Simultaneously, user-supporting tools are rare, and the number of available examples is much lower than in classical programming languages. To address these challenges, we investigate the efficiency of Large Language Models (LLMs), specifically ChatGPT, to support users when dealing with scientific workflows. We performed three user studies in two scientific domains to evaluate ChatGPT for comprehending, adapting, and extending workflows. Our results indicate that LLMs efficiently interpret workflows but achieve lower performance for exchanging components or purposeful workflow extensions. We characterize their limitations in these challenging scenarios and suggest future research directions.

{{</citation>}}


### (75/81) Efficient Algorithms for Monte Carlo Particle Transport on AI Accelerator Hardware (John Tramm et al., 2023)

{{<citation>}}

John Tramm, Bryce Allen, Kazutomo Yoshii, Andrew Siegel, Leighton Wilson. (2023)  
**Efficient Algorithms for Monte Carlo Particle Transport on AI Accelerator Hardware**  

---
Primary Category: cs.DC  
Categories: D-1-3; J-2, cs-DC, cs-PF, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2311.01739v1)  

---


**ABSTRACT**  
The recent trend toward deep learning has led to the development of a variety of highly innovative AI accelerator architectures. One such architecture, the Cerebras Wafer-Scale Engine 2 (WSE-2), features 40 GB of on-chip SRAM, making it a potentially attractive platform for latency- or bandwidth-bound HPC simulation workloads. In this study, we examine the feasibility of performing continuous energy Monte Carlo (MC) particle transport on the WSE-2 by porting a key kernel from the MC transport algorithm to Cerebras's CSL programming model. New algorithms for minimizing communication costs and for handling load balancing are developed and tested. The WSE-2 is found to run \SPEEDUP~times faster than a highly optimized CUDA version of the kernel run on an NVIDIA A100 GPU -- significantly outpacing the expected performance increase given the difference in transistor counts between the architectures.

{{</citation>}}


## cs.SI (1)



### (76/81) Cross-modal Consistency Learning with Fine-grained Fusion Network for Multimodal Fake News Detection (Jun Li et al., 2023)

{{<citation>}}

Jun Li, Yi Bin, Jie Zou, Jie Zou, Guoqing Wang, Yang Yang. (2023)  
**Cross-modal Consistency Learning with Fine-grained Fusion Network for Multimodal Fake News Detection**  

---
Primary Category: cs.SI  
Categories: cs-SI, cs.SI  
Keywords: Fake News  
[Paper Link](http://arxiv.org/abs/2311.01807v1)  

---


**ABSTRACT**  
Previous studies on multimodal fake news detection have observed the mismatch between text and images in the fake news and attempted to explore the consistency of multimodal news based on global features of different modalities. However, they fail to investigate this relationship between fine-grained fragments in multimodal content. To gain public trust, fake news often includes relevant parts in the text and the image, making such multimodal content appear consistent. Using global features may suppress potential inconsistencies in irrelevant parts. Therefore, in this paper, we propose a novel Consistency-learning Fine-grained Fusion Network (CFFN) that separately explores the consistency and inconsistency from high-relevant and low-relevant word-region pairs. Specifically, for a multimodal post, we divide word-region pairs into high-relevant and low-relevant parts based on their relevance scores. For the high-relevant part, we follow the cross-modal attention mechanism to explore the consistency. For low-relevant part, we calculate inconsistency scores to capture inconsistent points. Finally, a selection module is used to choose the primary clue (consistency or inconsistency) for identifying the credibility of multimodal news. Extensive experiments on two public datasets demonstrate that our CFFN substantially outperforms all the baselines.

{{</citation>}}


## math.OC (1)



### (77/81) Sketching for Convex and Nonconvex Regularized Least Squares with Sharp Guarantees (Yingzhen Yang et al., 2023)

{{<citation>}}

Yingzhen Yang, Ping Li. (2023)  
**Sketching for Convex and Nonconvex Regularized Least Squares with Sharp Guarantees**  

---
Primary Category: math.OC  
Categories: cs-LG, math-OC, math-ST, math.OC, stat-ML, stat-TH  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2311.01806v1)  

---


**ABSTRACT**  
Randomized algorithms are important for solving large-scale optimization problems. In this paper, we propose a fast sketching algorithm for least square problems regularized by convex or nonconvex regularization functions, Sketching for Regularized Optimization (SRO). Our SRO algorithm first generates a sketch of the original data matrix, then solves the sketched problem. Different from existing randomized algorithms, our algorithm handles general Frechet subdifferentiable regularization functions in an unified framework. We present general theoretical result for the approximation error between the optimization results of the original problem and the sketched problem for regularized least square problems which can be convex or nonconvex. For arbitrary convex regularizer, relative-error bound is proved for the approximation error. Importantly, minimax rates for sparse signal estimation by solving the sketched sparse convex or nonconvex learning problems are also obtained using our general theoretical result under mild conditions. To the best of our knowledge, our results are among the first to demonstrate minimax rates for convex or nonconvex sparse learning problem by sketching under a unified theoretical framework. We further propose an iterative sketching algorithm which reduces the approximation error exponentially by iteratively invoking the sketching algorithm. Experimental results demonstrate the effectiveness of the proposed SRO and Iterative SRO algorithms.

{{</citation>}}


## cs.MA (1)



### (78/81) RiskQ: Risk-sensitive Multi-Agent Reinforcement Learning Value Factorization (Siqi Shen et al., 2023)

{{<citation>}}

Siqi Shen, Chennan Ma, Chao Li, Weiquan Liu, Yongquan Fu, Songzhu Mei, Xinwang Liu, Cheng Wang. (2023)  
**RiskQ: Risk-sensitive Multi-Agent Reinforcement Learning Value Factorization**  

---
Primary Category: cs.MA  
Categories: cs-AI, cs-LG, cs-MA, cs.MA  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.01753v1)  

---


**ABSTRACT**  
Multi-agent systems are characterized by environmental uncertainty, varying policies of agents, and partial observability, which result in significant risks. In the context of Multi-Agent Reinforcement Learning (MARL), learning coordinated and decentralized policies that are sensitive to risk is challenging. To formulate the coordination requirements in risk-sensitive MARL, we introduce the Risk-sensitive Individual-Global-Max (RIGM) principle as a generalization of the Individual-Global-Max (IGM) and Distributional IGM (DIGM) principles. This principle requires that the collection of risk-sensitive action selections of each agent should be equivalent to the risk-sensitive action selection of the central policy. Current MARL value factorization methods do not satisfy the RIGM principle for common risk metrics such as the Value at Risk (VaR) metric or distorted risk measurements. Therefore, we propose RiskQ to address this limitation, which models the joint return distribution by modeling quantiles of it as weighted quantile mixtures of per-agent return distribution utilities. RiskQ satisfies the RIGM principle for the VaR and distorted risk metrics. We show that RiskQ can obtain promising performance through extensive experiments. The source code of RiskQ is available in https://github.com/xmu-rl-3dv/RiskQ.

{{</citation>}}


## eess.SY (1)



### (79/81) Low Overhead Beam Alignment for Mobile Millimeter Channel Based on Continuous-Time Prediction (Huang-Chou Lin et al., 2023)

{{<citation>}}

Huang-Chou Lin, Kuang-Hao, Liu. (2023)  
**Low Overhead Beam Alignment for Mobile Millimeter Channel Based on Continuous-Time Prediction**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2311.01752v1)  

---


**ABSTRACT**  
In millimeter-wave (mmWave) communications, directional transmission based on beamforming is important to compensate for high pathloss. To maintain the desired direction transmission gain, beam scanning that involves the transmitter sending the pilot signal over all available beam directions to find the optimal beam is often considered. Alternatively, beam tracking using partial beams can save the beam training overhead through algorithms such as statistical analysis models and kalman filter (KF). Unfortunately, existing beam tracking solutions are limited to a fixed beam variation pattern. In this work, we propose a beam alignment scheme called adaptive online beam alignment (AOBA), which aims to reduce training overhead and achieve accurate beam alignment for any movement profile. The proposed AOBA periodically performs beam tracking using a small amount but carefully selected candidate beams and switches to beam scanning using all available beams based on a given switching rule. During the interval without the pilot signal, the optimal beam at an arbitrary time instant is predicted with the aid of the recently proposed ordinary differential equation (ODE)-long short-term memory (LSTM) model. Extensive simulations are conducted to evaluate the performance of the proposed AOBA in comparison with several existing beam alignment schemes.

{{</citation>}}


## cs.IT (1)



### (80/81) Energy Efficiency Optimization for Subterranean LoRaWAN Using A Reinforcement Learning Approach: A Direct-to-Satellite Scenario (Kaiqiang Lin et al., 2023)

{{<citation>}}

Kaiqiang Lin, Muhammad Asad Ullah, Hirley Alves, Konstantin Mikhaylov, Tong Hao. (2023)  
**Energy Efficiency Optimization for Subterranean LoRaWAN Using A Reinforcement Learning Approach: A Direct-to-Satellite Scenario**  

---
Primary Category: cs.IT  
Categories: cs-AI, cs-IT, cs-LG, cs-NI, cs.IT, math-IT  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2311.01743v1)  

---


**ABSTRACT**  
The integration of subterranean LoRaWAN and non-terrestrial networks (NTN) delivers substantial economic and societal benefits in remote agriculture and disaster rescue operations. The LoRa modulation leverages quasi-orthogonal spreading factors (SFs) to optimize data rates, airtime, coverage and energy consumption. However, it is still challenging to effectively assign SFs to end devices for minimizing co-SF interference in massive subterranean LoRaWAN NTN. To address this, we investigate a reinforcement learning (RL)-based SFs allocation scheme to optimize the system's energy efficiency (EE). To efficiently capture the device-to-environment interactions in dense networks, we proposed an SFs allocation technique using the multi-agent dueling double deep Q-network (MAD3QN) and the multi-agent advantage actor-critic (MAA2C) algorithms based on an analytical reward mechanism. Our proposed RL-based SFs allocation approach evinces better performance compared to four benchmarks in the extreme underground direct-to-satellite scenario. Remarkably, MAD3QN shows promising potentials in surpassing MAA2C in terms of convergence rate and EE.

{{</citation>}}


## quant-ph (1)



### (81/81) Flexible Error Mitigation of Quantum Processes with Data Augmentation Empowered Neural Model (Manwen Liao et al., 2023)

{{<citation>}}

Manwen Liao, Yan Zhu, Giulio Chiribella, Yuxiang Yang. (2023)  
**Flexible Error Mitigation of Quantum Processes with Data Augmentation Empowered Neural Model**  

---
Primary Category: quant-ph  
Categories: cs-AI, cs-LG, quant-ph, quant-ph  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2311.01727v1)  

---


**ABSTRACT**  
Neural networks have shown their effectiveness in various tasks in the realm of quantum computing. However, their application in quantum error mitigation, a crucial step towards realizing practical quantum advancements, has been restricted by reliance on noise-free statistics. To tackle this critical challenge, we propose a data augmentation empowered neural model for error mitigation (DAEM). Our model does not require any prior knowledge about the specific noise type and measurement settings and can estimate noise-free statistics solely from the noisy measurement results of the target quantum process, rendering it highly suitable for practical implementation. In numerical experiments, we show the model's superior performance in mitigating various types of noise, including Markovian noise and Non-Markovian noise, compared with previous error mitigation methods. We further demonstrate its versatility by employing the model to mitigate errors in diverse types of quantum processes, including those involving large-scale quantum systems and continuous-variable quantum states. This powerful data augmentation-empowered neural model for error mitigation establishes a solid foundation for realizing more reliable and robust quantum technologies in practical applications.

{{</citation>}}
