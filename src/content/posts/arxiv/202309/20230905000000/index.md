---
draft: false
title: "arXiv @ 2023.09.05"
date: 2023-09-05
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.09.05"
    identifier: arxiv_20230905
    parent: 202309_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.GT (1)](#csgt-1)
- [cs.CR (3)](#cscr-3)
- [cs.CV (16)](#cscv-16)
- [cs.RO (2)](#csro-2)
- [cs.AI (3)](#csai-3)
- [cs.CL (5)](#cscl-5)
- [cs.LG (7)](#cslg-7)
- [cs.HC (1)](#cshc-1)
- [eess.IV (2)](#eessiv-2)
- [cs.GR (1)](#csgr-1)
- [cs.IR (3)](#csir-3)
- [cs.DC (1)](#csdc-1)
- [eess.AS (3)](#eessas-3)
- [quant-ph (1)](#quant-ph-1)
- [q-bio.QM (1)](#q-bioqm-1)
- [cs.CE (1)](#csce-1)
- [cs.NI (1)](#csni-1)

## cs.GT (1)



### (1/52) Generative Social Choice (Sara Fish et al., 2023)

{{<citation>}}

Sara Fish, Paul Gölz, David C. Parkes, Ariel D. Procaccia, Gili Rusak, Itai Shapira, Manuel Wüthrich. (2023)  
**Generative Social Choice**  

---
Primary Category: cs.GT  
Categories: cs-AI, cs-GT, cs-LG, cs.GT  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.01291v1)  

---


**ABSTRACT**  
Traditionally, social choice theory has only been applicable to choices among a few predetermined alternatives but not to more complex decisions such as collectively selecting a textual statement. We introduce generative social choice, a framework that combines the mathematical rigor of social choice theory with large language models' capability to generate text and extrapolate preferences. This framework divides the design of AI-augmented democratic processes into two components: first, proving that the process satisfies rigorous representation guarantees when given access to oracle queries; second, empirically validating that these queries can be approximately implemented using a large language model. We illustrate this framework by applying it to the problem of generating a slate of statements that is representative of opinions expressed as free-form text, for instance in an online deliberative process.

{{</citation>}}


## cs.CR (3)



### (2/52) Game Theory in Distributed Systems Security: Foundations, Challenges, and Future Directions (Mustafa Abdallah et al., 2023)

{{<citation>}}

Mustafa Abdallah, Saurabh Bagchi, Shaunak D. Bopardikar, Kevin Chan, Xing Gao, Murat Kantarcioglu, Congmiao Li, Peng Liu, Quanyan Zhu. (2023)  
**Game Theory in Distributed Systems Security: Foundations, Challenges, and Future Directions**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-DC, cs-GT, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.01281v1)  

---


**ABSTRACT**  
Many of our critical infrastructure systems and personal computing systems have a distributed computing systems structure. The incentives to attack them have been growing rapidly as has their attack surface due to increasing levels of connectedness. Therefore, we feel it is time to bring in rigorous reasoning to secure such systems. The distributed system security and the game theory technical communities can come together to effectively address this challenge. In this article, we lay out the foundations from each that we can build upon to achieve our goals. Next, we describe a set of research challenges for the community, organized into three categories -- analytical, systems, and integration challenges, each with "short term" time horizon (2-3 years) and "long term" (5-10 years) items. This article was conceived of through a community discussion at the 2022 NSF SaTC PI meeting.

{{</citation>}}


### (3/52) Multidomain transformer-based deep learning for early detection of network intrusion (Jinxin Liu et al., 2023)

{{<citation>}}

Jinxin Liu, Murat Simsek, Michele Nogueira, Burak Kantarci. (2023)  
**Multidomain transformer-based deep learning for early detection of network intrusion**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs.CR  
Keywords: Attention, Intrusion Detection, Time Series, Transformer  
[Paper Link](http://arxiv.org/abs/2309.01070v1)  

---


**ABSTRACT**  
Timely response of Network Intrusion Detection Systems (NIDS) is constrained by the flow generation process which requires accumulation of network packets. This paper introduces Multivariate Time Series (MTS) early detection into NIDS to identify malicious flows prior to their arrival at target systems. With this in mind, we first propose a novel feature extractor, Time Series Network Flow Meter (TS-NFM), that represents network flow as MTS with explainable features, and a new benchmark dataset is created using TS-NFM and the meta-data of CICIDS2017, called SCVIC-TS-2022. Additionally, a new deep learning-based early detection model called Multi-Domain Transformer (MDT) is proposed, which incorporates the frequency domain into Transformer. This work further proposes a Multi-Domain Multi-Head Attention (MD-MHA) mechanism to improve the ability of MDT to extract better features. Based on the experimental results, the proposed methodology improves the earliness of the conventional NIDS (i.e., percentage of packets that are used for classification) by 5x10^4 times and duration-based earliness (i.e., percentage of duration of the classified packets of a flow) by a factor of 60, resulting in a 84.1% macro F1 score (31% higher than Transformer) on SCVIC-TS-2022. Additionally, the proposed MDT outperforms the state-of-the-art early detection methods by 5% and 6% on ECG and Wafer datasets, respectively.

{{</citation>}}


### (4/52) Digital Twins and Blockchain for IoT Management (Mayra Samaniego et al., 2023)

{{<citation>}}

Mayra Samaniego, Ralph Deters. (2023)  
**Digital Twins and Blockchain for IoT Management**  

---
Primary Category: cs.CR  
Categories: H-3-4, cs-CR, cs-DC, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2309.01042v1)  

---


**ABSTRACT**  
Security and privacy are primary concerns in IoT management. Security breaches in IoT resources, such as smart sensors, can leak sensitive data and compromise the privacy of individuals. Effective IoT management requires a comprehensive approach to prioritize access security and data privacy protection. Digital twins create virtual representations of IoT resources. Blockchain adds decentralization, transparency, and reliability to IoT systems. This research integrates digital twins and blockchain to manage access to IoT data streaming. Digital twins are used to encapsulate data access and view configurations. Access is enabled on digital twins, not on IoT resources directly. Trust structures programmed as smart contracts are the ones that manage access to digital twins. Consequently, IoT resources are not exposed to third parties, and access security breaches can be prevented. Blockchain has been used to validate digital twins and store their configuration. The research presented in this paper enables multitenant access and customization of data streaming views and abstracts the complexity of data access management. This approach provides access and configuration security and data privacy protection.

{{</citation>}}


## cs.CV (16)



### (5/52) COMEDIAN: Self-Supervised Learning and Knowledge Distillation for Action Spotting using Transformers (Julien Denize et al., 2023)

{{<citation>}}

Julien Denize, Mykola Liashuha, Jaonary Rabarisoa, Astrid Orcesi, Romain Hérault. (2023)  
**COMEDIAN: Self-Supervised Learning and Knowledge Distillation for Action Spotting using Transformers**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Knowledge Distillation, Self-Supervised, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2309.01270v1)  

---


**ABSTRACT**  
We present COMEDIAN, a novel pipeline to initialize spatio-temporal transformers for action spotting, which involves self-supervised learning and knowledge distillation. Action spotting is a timestamp-level temporal action detection task. Our pipeline consists of three steps, with two initialization stages. First, we perform self-supervised initialization of a spatial transformer using short videos as input. Additionally, we initialize a temporal transformer that enhances the spatial transformer's outputs with global context through knowledge distillation from a pre-computed feature bank aligned with each short video segment. In the final step, we fine-tune the transformers to the action spotting task. The experiments, conducted on the SoccerNet-v2 dataset, demonstrate state-of-the-art performance and validate the effectiveness of COMEDIAN's pretraining paradigm. Our results highlight several advantages of our pretraining pipeline, including improved performance and faster convergence compared to non-pretrained models.

{{</citation>}}


### (6/52) Multimodal Contrastive Learning with Hard Negative Sampling for Human Activity Recognition (Hyeongju Choi et al., 2023)

{{<citation>}}

Hyeongju Choi, Apoorva Beedu, Irfan Essa. (2023)  
**Multimodal Contrastive Learning with Hard Negative Sampling for Human Activity Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-HC, cs-LG, cs.CV, eess-SP  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.01262v1)  

---


**ABSTRACT**  
Human Activity Recognition (HAR) systems have been extensively studied by the vision and ubiquitous computing communities due to their practical applications in daily life, such as smart homes, surveillance, and health monitoring.   Typically, this process is supervised in nature and the development of such systems requires access to large quantities of annotated data.   However, the higher costs and challenges associated with obtaining good quality annotations have rendered the application of self-supervised methods an attractive option and contrastive learning comprises one such method.   However, a major component of successful contrastive learning is the selection of good positive and negative samples.   Although positive samples are directly obtainable, sampling good negative samples remain a challenge.   As human activities can be recorded by several modalities like camera and IMU sensors, we propose a hard negative sampling method for multimodal HAR with a hard negative sampling loss for skeleton and IMU data pairs.   We exploit hard negatives that have different labels from the anchor but are projected nearby in the latent space using an adjustable concentration parameter.   Through extensive experiments on two benchmark datasets: UTD-MHAD and MMAct, we demonstrate the robustness of our approach forlearning strong feature representation for HAR tasks, and on the limited data setting.   We further show that our model outperforms all other state-of-the-art methods for UTD-MHAD dataset, and self-supervised methods for MMAct: Cross session, even when uni-modal data are used during downstream activity recognition.

{{</citation>}}


### (7/52) BDC-Adapter: Brownian Distance Covariance for Better Vision-Language Reasoning (Yi Zhang et al., 2023)

{{<citation>}}

Yi Zhang, Ce Zhang, Zihan Liao, Yushun Tang, Zhihai He. (2023)  
**BDC-Adapter: Brownian Distance Covariance for Better Vision-Language Reasoning**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model, Reasoning  
[Paper Link](http://arxiv.org/abs/2309.01256v1)  

---


**ABSTRACT**  
Large-scale pre-trained Vision-Language Models (VLMs), such as CLIP and ALIGN, have introduced a new paradigm for learning transferable visual representations. Recently, there has been a surge of interest among researchers in developing lightweight fine-tuning techniques to adapt these models to downstream visual tasks. We recognize that current state-of-the-art fine-tuning methods, such as Tip-Adapter, simply consider the covariance between the query image feature and features of support few-shot training samples, which only captures linear relations and potentially instigates a deceptive perception of independence. To address this issue, in this work, we innovatively introduce Brownian Distance Covariance (BDC) to the field of vision-language reasoning. The BDC metric can model all possible relations, providing a robust metric for measuring feature dependence. Based on this, we present a novel method called BDC-Adapter, which integrates BDC prototype similarity reasoning and multi-modal reasoning network prediction to perform classification tasks. Our extensive experimental results show that the proposed BDC-Adapter can freely handle non-linear relations and fully characterize independence, outperforming the current state-of-the-art methods by large margins.

{{</citation>}}


### (8/52) Holistic Dynamic Frequency Transformer for Image Fusion and Exposure Correction (Xiaoke Shang et al., 2023)

{{<citation>}}

Xiaoke Shang, Gehui Li, Zhiying Jiang, Shaomin Zhang, Nai Ding, Jinyuan Liu. (2023)  
**Holistic Dynamic Frequency Transformer for Image Fusion and Exposure Correction**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Transformer  
[Paper Link](http://arxiv.org/abs/2309.01183v1)  

---


**ABSTRACT**  
The correction of exposure-related issues is a pivotal component in enhancing the quality of images, offering substantial implications for various computer vision tasks. Historically, most methodologies have predominantly utilized spatial domain recovery, offering limited consideration to the potentialities of the frequency domain. Additionally, there has been a lack of a unified perspective towards low-light enhancement, exposure correction, and multi-exposure fusion, complicating and impeding the optimization of image processing. In response to these challenges, this paper proposes a novel methodology that leverages the frequency domain to improve and unify the handling of exposure correction tasks. Our method introduces Holistic Frequency Attention and Dynamic Frequency Feed-Forward Network, which replace conventional correlation computation in the spatial-domain. They form a foundational building block that facilitates a U-shaped Holistic Dynamic Frequency Transformer as a filter to extract global information and dynamically select important frequency bands for image restoration. Complementing this, we employ a Laplacian pyramid to decompose images into distinct frequency bands, followed by multiple restorers, each tuned to recover specific frequency-band information. The pyramid fusion allows a more detailed and nuanced image restoration process. Ultimately, our structure unifies the three tasks of low-light enhancement, exposure correction, and multi-exposure fusion, enabling comprehensive treatment of all classical exposure errors. Benchmarking on mainstream datasets for these tasks, our proposed method achieves state-of-the-art results, paving the way for more sophisticated and unified solutions in exposure correction.

{{</citation>}}


### (9/52) LoGoPrompt: Synthetic Text Images Can Be Good Visual Prompts for Vision-Language Models (Cheng Shi et al., 2023)

{{<citation>}}

Cheng Shi, Sibei Yang. (2023)  
**LoGoPrompt: Synthetic Text Images Can Be Good Visual Prompts for Vision-Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, ImageNet, Language Model  
[Paper Link](http://arxiv.org/abs/2309.01155v1)  

---


**ABSTRACT**  
Prompt engineering is a powerful tool used to enhance the performance of pre-trained models on downstream tasks. For example, providing the prompt ``Let's think step by step" improved GPT-3's reasoning accuracy to 63% on MutiArith while prompting ``a photo of" filled with a class name enables CLIP to achieve $80$\% zero-shot accuracy on ImageNet. While previous research has explored prompt learning for the visual modality, analyzing what constitutes a good visual prompt specifically for image recognition is limited. In addition, existing visual prompt tuning methods' generalization ability is worse than text-only prompting tuning. This paper explores our key insight: synthetic text images are good visual prompts for vision-language models! To achieve that, we propose our LoGoPrompt, which reformulates the classification objective to the visual prompt selection and addresses the chicken-and-egg challenge of first adding synthetic text images as class-wise visual prompts or predicting the class first. Without any trainable visual prompt parameters, experimental results on 16 datasets demonstrate that our method consistently outperforms state-of-the-art methods in few-shot learning, base-to-new generalization, and domain generalization.

{{</citation>}}


### (10/52) EdaDet: Open-Vocabulary Object Detection Using Early Dense Alignment (Cheng Shi et al., 2023)

{{<citation>}}

Cheng Shi, Sibei Yang. (2023)  
**EdaDet: Open-Vocabulary Object Detection Using Early Dense Alignment**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.01151v1)  

---


**ABSTRACT**  
Vision-language models such as CLIP have boosted the performance of open-vocabulary object detection, where the detector is trained on base categories but required to detect novel categories. Existing methods leverage CLIP's strong zero-shot recognition ability to align object-level embeddings with textual embeddings of categories. However, we observe that using CLIP for object-level alignment results in overfitting to base categories, i.e., novel categories most similar to base categories have particularly poor performance as they are recognized as similar base categories. In this paper, we first identify that the loss of critical fine-grained local image semantics hinders existing methods from attaining strong base-to-novel generalization. Then, we propose Early Dense Alignment (EDA) to bridge the gap between generalizable local semantics and object-level prediction. In EDA, we use object-level supervision to learn the dense-level rather than object-level alignment to maintain the local fine-grained semantics. Extensive experiments demonstrate our superior performance to competing approaches under the same strict setting and without using external training resources, i.e., improving the +8.4% novel box AP50 on COCO and +3.9% rare mask AP on LVIS.

{{</citation>}}


### (11/52) Attention Where It Matters: Rethinking Visual Document Understanding with Selective Region Concentration (Haoyu Cao et al., 2023)

{{<citation>}}

Haoyu Cao, Changcun Bao, Chaohu Liu, Huang Chen, Kun Yin, Hao Liu, Yinsong Liu, Deqiang Jiang, Xing Sun. (2023)  
**Attention Where It Matters: Rethinking Visual Document Understanding with Selective Region Concentration**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.01131v1)  

---


**ABSTRACT**  
We propose a novel end-to-end document understanding model called SeRum (SElective Region Understanding Model) for extracting meaningful information from document images, including document analysis, retrieval, and office automation.   Unlike state-of-the-art approaches that rely on multi-stage technical schemes and are computationally expensive,   SeRum converts document image understanding and recognition tasks into a local decoding process of the visual tokens of interest, using a content-aware token merge module.   This mechanism enables the model to pay more attention to regions of interest generated by the query decoder, improving the model's effectiveness and speeding up the decoding speed of the generative scheme.   We also designed several pre-training tasks to enhance the understanding and local awareness of the model.   Experimental results demonstrate that SeRum achieves state-of-the-art performance on document understanding tasks and competitive results on text spotting tasks.   SeRum represents a substantial advancement towards enabling efficient and effective end-to-end document understanding.

{{</citation>}}


### (12/52) RSDiff: Remote Sensing Image Generation from Text Using Diffusion Model (Ahmad Sebaq et al., 2023)

{{<citation>}}

Ahmad Sebaq, Mohamed ElHelw. (2023)  
**RSDiff: Remote Sensing Image Generation from Text Using Diffusion Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Image Captioning  
[Paper Link](http://arxiv.org/abs/2309.02455v1)  

---


**ABSTRACT**  
Satellite imagery generation and super-resolution are pivotal tasks in remote sensing, demanding high-quality, detailed images for accurate analysis and decision-making. In this paper, we propose an innovative and lightweight approach that employs two-stage diffusion models to gradually generate high-resolution Satellite images purely based on text prompts. Our innovative pipeline comprises two interconnected diffusion models: a Low-Resolution Generation Diffusion Model (LR-GDM) that generates low-resolution images from text and a Super-Resolution Diffusion Model (SRDM) conditionally produced. The LR-GDM effectively synthesizes low-resolution by (computing the correlations of the text embedding and the image embedding in a shared latent space), capturing the essential content and layout of the desired scenes. Subsequently, the SRDM takes the generated low-resolution image and its corresponding text prompts and efficiently produces the high-resolution counterparts, infusing fine-grained spatial details and enhancing visual fidelity. Experiments are conducted on the commonly used dataset, Remote Sensing Image Captioning Dataset (RSICD). Our results demonstrate that our approach outperforms existing state-of-the-art (SoTA) models in generating satellite images with realistic geographical features, weather conditions, and land structures while achieving remarkable super-resolution results for increased spatial precision.

{{</citation>}}


### (13/52) AdvMono3D: Advanced Monocular 3D Object Detection with Depth-Aware Robust Adversarial Training (Xingyuan Li et al., 2023)

{{<citation>}}

Xingyuan Li, Jinyuan Liu, Long Ma, Xin Fan, Risheng Liu. (2023)  
**AdvMono3D: Advanced Monocular 3D Object Detection with Depth-Aware Robust Adversarial Training**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Adversarial Training, Object Detection  
[Paper Link](http://arxiv.org/abs/2309.01106v1)  

---


**ABSTRACT**  
Monocular 3D object detection plays a pivotal role in the field of autonomous driving and numerous deep learning-based methods have made significant breakthroughs in this area. Despite the advancements in detection accuracy and efficiency, these models tend to fail when faced with such attacks, rendering them ineffective. Therefore, bolstering the adversarial robustness of 3D detection models has become a crucial issue that demands immediate attention and innovative solutions. To mitigate this issue, we propose a depth-aware robust adversarial training method for monocular 3D object detection, dubbed DART3D. Specifically, we first design an adversarial attack that iteratively degrades the 2D and 3D perception capabilities of 3D object detection models(IDP), serves as the foundation for our subsequent defense mechanism. In response to this attack, we propose an uncertainty-based residual learning method for adversarial training. Our adversarial training approach capitalizes on the inherent uncertainty, enabling the model to significantly improve its robustness against adversarial attacks. We conducted extensive experiments on the KITTI 3D datasets, demonstrating that DART3D surpasses direct adversarial training (the most popular approach) under attacks in 3D object detection $AP_{R40}$ of car category for the Easy, Moderate, and Hard settings, with improvements of 4.415%, 4.112%, and 3.195%, respectively.

{{</citation>}}


### (14/52) CoTDet: Affordance Knowledge Prompting for Task Driven Object Detection (Jiajin Tang et al., 2023)

{{<citation>}}

Jiajin Tang, Ge Zheng, Jingyi Yu, Sibei Yang. (2023)  
**CoTDet: Affordance Knowledge Prompting for Task Driven Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.01093v1)  

---


**ABSTRACT**  
Task driven object detection aims to detect object instances suitable for affording a task in an image. Its challenge lies in object categories available for the task being too diverse to be limited to a closed set of object vocabulary for traditional object detection. Simply mapping categories and visual features of common objects to the task cannot address the challenge. In this paper, we propose to explore fundamental affordances rather than object categories, i.e., common attributes that enable different objects to accomplish the same task. Moreover, we propose a novel multi-level chain-of-thought prompting (MLCoT) to extract the affordance knowledge from large language models, which contains multi-level reasoning steps from task to object examples to essential visual attributes with rationales. Furthermore, to fully exploit knowledge to benefit object recognition and localization, we propose a knowledge-conditional detection framework, namely CoTDet. It conditions the detector from the knowledge to generate object queries and regress boxes. Experimental results demonstrate that our CoTDet outperforms state-of-the-art methods consistently and significantly (+15.6 box AP and +14.8 mask AP) and can generate rationales for why objects are detected to afford the task.

{{</citation>}}


### (15/52) MILA: Memory-Based Instance-Level Adaptation for Cross-Domain Object Detection (Onkar Krishna et al., 2023)

{{<citation>}}

Onkar Krishna, Hiroki Ohashi, Saptarshi Sinha. (2023)  
**MILA: Memory-Based Instance-Level Adaptation for Cross-Domain Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.01086v1)  

---


**ABSTRACT**  
Cross-domain object detection is challenging, and it involves aligning labeled source and unlabeled target domains. Previous approaches have used adversarial training to align features at both image-level and instance-level. At the instance level, finding a suitable source sample that aligns with a target sample is crucial. A source sample is considered suitable if it differs from the target sample only in domain, without differences in unimportant characteristics such as orientation and color, which can hinder the model's focus on aligning the domain difference. However, existing instance-level feature alignment methods struggle to find suitable source instances because their search scope is limited to mini-batches. Mini-batches are often so small in size that they do not always contain suitable source instances. The insufficient diversity of mini-batches becomes problematic particularly when the target instances have high intra-class variance. To address this issue, we propose a memory-based instance-level domain adaptation framework. Our method aligns a target instance with the most similar source instance of the same category retrieved from a memory storage. Specifically, we introduce a memory module that dynamically stores the pooled features of all labeled source instances, categorized by their labels. Additionally, we introduce a simple yet effective memory retrieval module that retrieves a set of matching memory slots for target instances. Our experiments on various domain shift scenarios demonstrate that our approach outperforms existing non-memory-based methods significantly.

{{</citation>}}


### (16/52) Chinese Text Recognition with A Pre-Trained CLIP-Like Model Through Image-IDS Aligning (Haiyang Yu et al., 2023)

{{<citation>}}

Haiyang Yu, Xiaocong Wang, Bin Li, Xiangyang Xue. (2023)  
**Chinese Text Recognition with A Pre-Trained CLIP-Like Model Through Image-IDS Aligning**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: OCR  
[Paper Link](http://arxiv.org/abs/2309.01083v1)  

---


**ABSTRACT**  
Scene text recognition has been studied for decades due to its broad applications. However, despite Chinese characters possessing different characteristics from Latin characters, such as complex inner structures and large categories, few methods have been proposed for Chinese Text Recognition (CTR). Particularly, the characteristic of large categories poses challenges in dealing with zero-shot and few-shot Chinese characters. In this paper, inspired by the way humans recognize Chinese texts, we propose a two-stage framework for CTR. Firstly, we pre-train a CLIP-like model through aligning printed character images and Ideographic Description Sequences (IDS). This pre-training stage simulates humans recognizing Chinese characters and obtains the canonical representation of each character. Subsequently, the learned representations are employed to supervise the CTR model, such that traditional single-character recognition can be improved to text-line recognition through image-IDS matching. To evaluate the effectiveness of the proposed method, we conduct extensive experiments on both Chinese character recognition (CCR) and CTR. The experimental results demonstrate that the proposed method performs best in CCR and outperforms previous methods in most scenarios of the CTR benchmark. It is worth noting that the proposed method can recognize zero-shot Chinese characters in text images without fine-tuning, whereas previous methods require fine-tuning when new classes appear. The code is available at https://github.com/FudanVI/FudanOCR/tree/main/image-ids-CTR.

{{</citation>}}


### (17/52) UnsMOT: Unified Framework for Unsupervised Multi-Object Tracking with Geometric Topology Guidance (Son Tran et al., 2023)

{{<citation>}}

Son Tran, Cong Tran, Anh Tran, Cuong Pham. (2023)  
**UnsMOT: Unified Framework for Unsupervised Multi-Object Tracking with Geometric Topology Guidance**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2309.01078v1)  

---


**ABSTRACT**  
Object detection has long been a topic of high interest in computer vision literature. Motivated by the fact that annotating data for the multi-object tracking (MOT) problem is immensely expensive, recent studies have turned their attention to the unsupervised learning setting. In this paper, we push forward the state-of-the-art performance of unsupervised MOT methods by proposing UnsMOT, a novel framework that explicitly combines the appearance and motion features of objects with geometric information to provide more accurate tracking. Specifically, we first extract the appearance and motion features using CNN and RNN models, respectively. Then, we construct a graph of objects based on their relative distances in a frame, which is fed into a GNN model together with CNN features to output geometric embedding of objects optimized using an unsupervised loss function. Finally, associations between objects are found by matching not only similar extracted features but also geometric embedding of detections and tracklets. Experimental results show remarkable performance in terms of HOTA, IDF1, and MOTA metrics in comparison with state-of-the-art methods.

{{</citation>}}


### (18/52) Spatial and Visual Perspective-Taking via View Rotation and Relation Reasoning for Embodied Reference Understanding (Cheng Shi et al., 2023)

{{<citation>}}

Cheng Shi, Sibei Yang. (2023)  
**Spatial and Visual Perspective-Taking via View Rotation and Relation Reasoning for Embodied Reference Understanding**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2309.01073v1)  

---


**ABSTRACT**  
Embodied Reference Understanding studies the reference understanding in an embodied fashion, where a receiver is required to locate a target object referred to by both language and gesture of the sender in a shared physical environment. Its main challenge lies in how to make the receiver with the egocentric view access spatial and visual information relative to the sender to judge how objects are oriented around and seen from the sender, i.e., spatial and visual perspective-taking. In this paper, we propose a REasoning from your Perspective (REP) method to tackle the challenge by modeling relations between the receiver and the sender and the sender and the objects via the proposed novel view rotation and relation reasoning. Specifically, view rotation first rotates the receiver to the position of the sender by constructing an embodied 3D coordinate system with the position of the sender as the origin. Then, it changes the orientation of the receiver to the orientation of the sender by encoding the body orientation and gesture of the sender. Relation reasoning models the nonverbal and verbal relations between the sender and the objects by multi-modal cooperative reasoning in gesture, language, visual content, and spatial position. Experiment results demonstrate the effectiveness of REP, which consistently surpasses all existing state-of-the-art algorithms by a large margin, i.e., +5.22% absolute accuracy in terms of Prec0.5 on YouRefIt.

{{</citation>}}


### (19/52) AB2CD: AI for Building Climate Damage Classification and Detection (Maximilian Nitsche et al., 2023)

{{<citation>}}

Maximilian Nitsche, S. Karthik Mukkavilli, Niklas Kühl, Thomas Brunschwiler. (2023)  
**AB2CD: AI for Building Climate Damage Classification and Detection**  

---
Primary Category: cs.CV  
Categories: 68T07 (Primary), 68T45, 86A08, 74A45 (Secondary), I-2-10; I-4-8; I-4-6; I-5-4; I-2-6, cs-AI, cs-CV, cs-CY, cs.CV, eess-IV, physics-geo-ph  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.01066v1)  

---


**ABSTRACT**  
We explore the implementation of deep learning techniques for precise building damage assessment in the context of natural hazards, utilizing remote sensing data. The xBD dataset, comprising diverse disaster events from across the globe, serves as the primary focus, facilitating the evaluation of deep learning models. We tackle the challenges of generalization to novel disasters and regions while accounting for the influence of low-quality and noisy labels inherent in natural hazard data. Furthermore, our investigation quantitatively establishes that the minimum satellite imagery resolution essential for effective building damage detection is 3 meters and below 1 meter for classification using symmetric and asymmetric resolution perturbation analyses. To achieve robust and accurate evaluations of building damage detection and classification, we evaluated different deep learning models with residual, squeeze and excitation, and dual path network backbones, as well as ensemble techniques. Overall, the U-Net Siamese network ensemble with F-1 score of 0.812 performed the best against the xView2 challenge benchmark. Additionally, we evaluate a Universal model trained on all hazards against a flood expert model and investigate generalization gaps across events, and out of distribution from field data in the Ahr Valley. Our research findings showcase the potential and limitations of advanced AI solutions in enhancing the impact assessment of climate change-induced extreme weather events, such as floods and hurricanes. These insights have implications for disaster impact assessment in the face of escalating climate challenges.

{{</citation>}}


### (20/52) Semi-supervised 3D Video Information Retrieval with Deep Neural Network and Bi-directional Dynamic-time Warping Algorithm (Yintai Ma et al., 2023)

{{<citation>}}

Yintai Ma, Diego Klabjan. (2023)  
**Semi-supervised 3D Video Information Retrieval with Deep Neural Network and Bi-directional Dynamic-time Warping Algorithm**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Information Retrieval  
[Paper Link](http://arxiv.org/abs/2309.01063v1)  

---


**ABSTRACT**  
This paper presents a novel semi-supervised deep learning algorithm for retrieving similar 2D and 3D videos based on visual content. The proposed approach combines the power of deep convolutional and recurrent neural networks with dynamic time warping as a similarity measure. The proposed algorithm is designed to handle large video datasets and retrieve the most related videos to a given inquiry video clip based on its graphical frames and contents. We split both the candidate and the inquiry videos into a sequence of clips and convert each clip to a representation vector using an autoencoder-backed deep neural network. We then calculate a similarity measure between the sequences of embedding vectors using a bi-directional dynamic time-warping method. This approach is tested on multiple public datasets, including CC\_WEB\_VIDEO, Youtube-8m, S3DIS, and Synthia, and showed good results compared to state-of-the-art. The algorithm effectively solves video retrieval tasks and outperforms the benchmarked state-of-the-art deep learning model.

{{</citation>}}


## cs.RO (2)



### (21/52) Outlining the design space of eXplainable swarm (xSwarm): experts perspective (Mohammad Naiseh et al., 2023)

{{<citation>}}

Mohammad Naiseh, Mohammad D. Soorati, Sarvapali Ramchurn. (2023)  
**Outlining the design space of eXplainable swarm (xSwarm): experts perspective**  

---
Primary Category: cs.RO  
Categories: cs-CY, cs-HC, cs-RO, cs.RO  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.01269v1)  

---


**ABSTRACT**  
In swarm robotics, agents interact through local roles to solve complex tasks beyond an individual's ability. Even though swarms are capable of carrying out some operations without the need for human intervention, many safety-critical applications still call for human operators to control and monitor the swarm. There are novel challenges to effective Human-Swarm Interaction (HSI) that are only beginning to be addressed. Explainability is one factor that can facilitate effective and trustworthy HSI and improve the overall performance of Human-Swarm team. Explainability was studied across various Human-AI domains, such as Human-Robot Interaction and Human-Centered ML. However, it is still ambiguous whether explanations studied in Human-AI literature would be beneficial in Human-Swarm research and development. Furthermore, the literature lacks foundational research on the prerequisites for explainability requirements in swarm robotics, i.e., what kind of questions an explainable swarm is expected to answer, and what types of explanations a swarm is expected to generate. By surveying 26 swarm experts, we seek to answer these questions and identify challenges experts faced to generate explanations in Human-Swarm environments. Our work contributes insights into defining a new area of research of eXplainable Swarm (xSwarm) which looks at how explainability can be implemented and developed in swarm systems. This paper opens the discussion on xSwarm and paves the way for more research in the field.

{{</citation>}}


### (22/52) Integration of Vision-based Object Detection and Grasping for Articulated Manipulator in Lunar Conditions (Camille Boucher et al., 2023)

{{<citation>}}

Camille Boucher, Gustavo H. Diaz, Shreya Santra, Kentaro Uno, Kazuya Yoshida. (2023)  
**Integration of Vision-based Object Detection and Grasping for Articulated Manipulator in Lunar Conditions**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs.RO  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2309.01055v1)  

---


**ABSTRACT**  
The integration of vision-based frameworks to achieve lunar robot applications faces numerous challenges such as terrain configuration or extreme lighting conditions. This paper presents a generic task pipeline using object detection, instance segmentation and grasp detection, that can be used for various applications by using the results of these vision-based systems in a different way. We achieve a rock stacking task on a non-flat surface in difficult lighting conditions with a very good success rate of 92%. Eventually, we present an experiment to assemble 3D printed robot components to initiate more complex tasks in the future.

{{</citation>}}


## cs.AI (3)



### (23/52) Large AI Model Empowered Multimodal Semantic Communications (Feibo Jiang et al., 2023)

{{<citation>}}

Feibo Jiang, Yubo Peng, Li Dong, Kezhi Wang, Kun Yang, Cunhua Pan, Xiaohu You. (2023)  
**Large AI Model Empowered Multimodal Semantic Communications**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs-LG, cs.AI  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.01249v1)  

---


**ABSTRACT**  
Multimodal signals, including text, audio, image and video, can be integrated into Semantic Communication (SC) for providing an immersive experience with low latency and high quality at the semantic level. However, the multimodal SC has several challenges, including data heterogeneity, semantic ambiguity, and signal fading. Recent advancements in large AI models, particularly in Multimodal Language Model (MLM) and Large Language Model (LLM), offer potential solutions for these issues. To this end, we propose a Large AI Model-based Multimodal SC (LAM-MSC) framework, in which we first present the MLM-based Multimodal Alignment (MMA) that utilizes the MLM to enable the transformation between multimodal and unimodal data while preserving semantic consistency. Then, a personalized LLM-based Knowledge Base (LKB) is proposed, which allows users to perform personalized semantic extraction or recovery through the LLM. This effectively addresses the semantic ambiguity. Finally, we apply the Conditional Generative adversarial networks-based channel Estimation (CGE) to obtain Channel State Information (CSI). This approach effectively mitigates the impact of fading channels in SC. Finally, we conduct simulations that demonstrate the superior performance of the LAM-MSC framework.

{{</citation>}}


### (24/52) A Survey on Service Route and Time Prediction in Instant Delivery: Taxonomy, Progress, and Prospects (Haomin Wen et al., 2023)

{{<citation>}}

Haomin Wen, Youfang Lin, Lixia Wu, Xiaowei Mao, Tianyue Cai, Yunfeng Hou, Shengnan Guo, Yuxuan Liang, Guangyin Jin, Yiji Zhao, Roger Zimmermann, Jieping Ye, Huaiyu Wan. (2023)  
**A Survey on Service Route and Time Prediction in Instant Delivery: Taxonomy, Progress, and Prospects**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2309.01194v1)  

---


**ABSTRACT**  
Instant delivery services, such as food delivery and package delivery, have achieved explosive growth in recent years by providing customers with daily-life convenience. An emerging research area within these services is service Route\&Time Prediction (RTP), which aims to estimate the future service route as well as the arrival time of a given worker. As one of the most crucial tasks in those service platforms, RTP stands central to enhancing user satisfaction and trimming operational expenditures on these platforms. Despite a plethora of algorithms developed to date, there is no systematic, comprehensive survey to guide researchers in this domain. To fill this gap, our work presents the first comprehensive survey that methodically categorizes recent advances in service route and time prediction. We start by defining the RTP challenge and then delve into the metrics that are often employed. Following that, we scrutinize the existing RTP methodologies, presenting a novel taxonomy of them. We categorize these methods based on three criteria: (i) type of task, subdivided into only-route prediction, only-time prediction, and joint route\&time prediction; (ii) model architecture, which encompasses sequence-based and graph-based models; and (iii) learning paradigm, including Supervised Learning (SL) and Deep Reinforcement Learning (DRL). Conclusively, we highlight the limitations of current research and suggest prospective avenues. We believe that the taxonomy, progress, and prospects introduced in this paper can significantly promote the development of this field.

{{</citation>}}


### (25/52) A Study on the Implementation of Generative AI Services Using an Enterprise Data-Based LLM Application Architecture (Cheonsu Jeong, 2023)

{{<citation>}}

Cheonsu Jeong. (2023)  
**A Study on the Implementation of Generative AI Services Using an Enterprise Data-Based LLM Application Architecture**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CL, cs.AI  
Keywords: AI, Generative AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.01105v1)  

---


**ABSTRACT**  
This study presents a method for implementing generative AI services by utilizing the Large Language Model (LLM) application architecture. With recent advancements in generative AI technology, LLMs have gained prominence across various domains. In this context, the research addresses the challenge of information scarcity and proposes specific remedies by harnessing LLM capabilities. The investigation delves into strategies for mitigating the issue of inadequate data, offering tailored solutions. The study delves into the efficacy of employing fine-tuning techniques and direct document integration to alleviate data insufficiency. A significant contribution of this work is the development of a Retrieval-Augmented Generation (RAG) model, which tackles the aforementioned challenges. The RAG model is carefully designed to enhance information storage and retrieval processes, ensuring improved content generation. The research elucidates the key phases of the information storage and retrieval methodology underpinned by the RAG model. A comprehensive analysis of these steps is undertaken, emphasizing their significance in addressing the scarcity of data. The study highlights the efficacy of the proposed method, showcasing its applicability through illustrative instances. By implementing the RAG model for information storage and retrieval, the research not only contributes to a deeper comprehension of generative AI technology but also facilitates its practical usability within enterprises utilizing LLMs. This work holds substantial value in advancing the field of generative AI, offering insights into enhancing data-driven content generation and fostering active utilization of LLM-based services within corporate settings.

{{</citation>}}


## cs.CL (5)



### (26/52) Representations Matter: Embedding Modes of Large Language Models using Dynamic Mode Decomposition (Mohamed Akrout, 2023)

{{<citation>}}

Mohamed Akrout. (2023)  
**Representations Matter: Embedding Modes of Large Language Models using Dynamic Mode Decomposition**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Embedding, Language Model  
[Paper Link](http://arxiv.org/abs/2309.01245v1)  

---


**ABSTRACT**  
Existing large language models (LLMs) are known for generating "hallucinated" content, namely a fabricated text of plausibly looking, yet unfounded, facts. To identify when these hallucination scenarios occur, we examine the properties of the generated text in the embedding space. Specifically, we draw inspiration from the dynamic mode decomposition (DMD) tool in analyzing the pattern evolution of text embeddings across sentences. We empirically demonstrate how the spectrum of sentence embeddings over paragraphs is constantly low-rank for the generated text, unlike that of the ground-truth text. Importantly, we find that evaluation cases having LLM hallucinations correspond to ground-truth embedding patterns with a higher number of modes being poorly approximated by the few modes associated with LLM embedding patterns. In analogy to near-field electromagnetic evanescent waves, the embedding DMD eigenmodes of the generated text with hallucinations vanishes quickly across sentences as opposed to those of the ground-truth text. This suggests that the hallucinations result from both the generation techniques and the underlying representation.

{{</citation>}}


### (27/52) Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models (Yue Zhang et al., 2023)

{{<citation>}}

Yue Zhang, Yafu Li, Leyang Cui, Deng Cai, Lemao Liu, Tingchen Fu, Xinting Huang, Enbo Zhao, Yu Zhang, Yulong Chen, Longyue Wang, Anh Tuan Luu, Wei Bi, Freda Shi, Shuming Shi. (2023)  
**Siren's Song in the AI Ocean: A Survey on Hallucination in Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs-LG, cs.CL  
Keywords: AI, Language Model  
[Paper Link](http://arxiv.org/abs/2309.01219v1)  

---


**ABSTRACT**  
While large language models (LLMs) have demonstrated remarkable capabilities across a range of downstream tasks, a significant concern revolves around their propensity to exhibit hallucinations: LLMs occasionally generate content that diverges from the user input, contradicts previously generated context, or misaligns with established world knowledge. This phenomenon poses a substantial challenge to the reliability of LLMs in real-world scenarios. In this paper, we survey recent efforts on the detection, explanation, and mitigation of hallucination, with an emphasis on the unique challenges posed by LLMs. We present taxonomies of the LLM hallucination phenomena and evaluation benchmarks, analyze existing approaches aiming at mitigating LLM hallucination, and discuss potential directions for future research.

{{</citation>}}


### (28/52) A Visual Interpretation-Based Self-Improved Classification System Using Virtual Adversarial Training (Shuai Jiang et al., 2023)

{{<citation>}}

Shuai Jiang, Sayaka Kamei, Chen Li, Shengzhe Hou, Yasuhiko Morimoto. (2023)  
**A Visual Interpretation-Based Self-Improved Classification System Using Virtual Adversarial Training**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Adversarial Training, BERT, Twitter  
[Paper Link](http://arxiv.org/abs/2309.01196v1)  

---


**ABSTRACT**  
The successful application of large pre-trained models such as BERT in natural language processing has attracted more attention from researchers. Since the BERT typically acts as an end-to-end black box, classification systems based on it usually have difficulty in interpretation and low robustness. This paper proposes a visual interpretation-based self-improving classification model with a combination of virtual adversarial training (VAT) and BERT models to address the above problems. Specifically, a fine-tuned BERT model is used as a classifier to classify the sentiment of the text. Then, the predicted sentiment classification labels are used as part of the input of another BERT for spam classification via a semi-supervised training manner using VAT. Additionally, visualization techniques, including visualizing the importance of words and normalizing the attention head matrix, are employed to analyze the relevance of each component to classification accuracy. Moreover, brand-new features will be found in the visual analysis, and classification performance will be improved. Experimental results on Twitter's tweet dataset demonstrate the effectiveness of the proposed model on the classification task. Furthermore, the ablation study results illustrate the effect of different components of the proposed model on the classification results.

{{</citation>}}


### (29/52) MedChatZH: a Better Medical Adviser Learns from Better Instructions (Yang Tan et al., 2023)

{{<citation>}}

Yang Tan, Mingchen Li, Zijie Huang, Huiqun Yu, Guisheng Fan. (2023)  
**MedChatZH: a Better Medical Adviser Learns from Better Instructions**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2309.01114v1)  

---


**ABSTRACT**  
Generative large language models (LLMs) have shown great success in various applications, including question-answering (QA) and dialogue systems. However, in specialized domains like traditional Chinese medical QA, these models may perform unsatisfactorily without fine-tuning on domain-specific datasets. To address this, we introduce MedChatZH, a dialogue model designed specifically for traditional Chinese medical QA. Our model is pre-trained on Chinese traditional medical books and fine-tuned with a carefully curated medical instruction dataset. It outperforms several solid baselines on a real-world medical dialogue dataset. We release our model, code, and dataset on https://github.com/tyang816/MedChatZH to facilitate further research in the domain of traditional Chinese medicine and LLMs.

{{</citation>}}


### (30/52) Business Process Text Sketch Automation Generation Using Large Language Model (Rui Zhu et al., 2023)

{{<citation>}}

Rui Zhu, Quanzhou Hu, Wenxin Li, Honghao Xiao, Chaogang Wang, Zixin Zhou. (2023)  
**Business Process Text Sketch Automation Generation Using Large Language Model**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Sketch  
[Paper Link](http://arxiv.org/abs/2309.01071v1)  

---


**ABSTRACT**  
Business Process Management (BPM) is gaining increasing attention as it has the potential to cut costs while boosting output and quality. Business process document generation is a crucial stage in BPM. However, due to a shortage of datasets, data-driven deep learning techniques struggle to deliver the expected results. We propose an approach to transform Conditional Process Trees (CPTs) into Business Process Text Sketches (BPTSs) using Large Language Models (LLMs). The traditional prompting approach (Few-shot In-Context Learning) tries to get the correct answer in one go, and it can find the pattern of transforming simple CPTs into BPTSs, but for close-domain and CPTs with complex hierarchy, the traditional prompts perform weakly and with low correctness. We suggest using this technique to break down a difficult CPT into a number of basic CPTs and then solve each one in turn, drawing inspiration from the divide-and-conquer strategy. We chose 100 process trees with depths ranging from 2 to 5 at random, as well as CPTs with many nodes, many degrees of selection, and cyclic nesting. Experiments show that our method can achieve a correct rate of 93.42%, which is 45.17% better than traditional prompting methods. Our proposed method provides a solution for business process document generation in the absence of datasets, and secondly, it becomes potentially possible to provide a large number of datasets for the process model extraction (PME) domain.

{{</citation>}}


## cs.LG (7)



### (31/52) Saturn: An Optimized Data System for Large Model Deep Learning Workloads (Kabir Nagrecha et al., 2023)

{{<citation>}}

Kabir Nagrecha, Arun Kumar. (2023)  
**Saturn: An Optimized Data System for Large Model Deep Learning Workloads**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DC, cs-LG, cs.LG  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.01226v1)  

---


**ABSTRACT**  
Large language models such as GPT-3 & ChatGPT have transformed deep learning (DL), powering applications that have captured the public's imagination. These models are rapidly being adopted across domains for analytics on various modalities, often by finetuning pre-trained base models. Such models need multiple GPUs due to both their size and computational load, driving the development of a bevy of "model parallelism" techniques & tools. Navigating such parallelism choices, however, is a new burden for end users of DL such as data scientists, domain scientists, etc. who may lack the necessary systems knowhow. The need for model selection, which leads to many models to train due to hyper-parameter tuning or layer-wise finetuning, compounds the situation with two more burdens: resource apportioning and scheduling. In this work, we tackle these three burdens for DL users in a unified manner by formalizing them as a joint problem that we call SPASE: Select a Parallelism, Allocate resources, and SchedulE. We propose a new information system architecture to tackle the SPASE problem holistically, representing a key step toward enabling wider adoption of large DL models. We devise an extensible template for existing parallelism schemes and combine it with an automated empirical profiler for runtime estimation. We then formulate SPASE as an MILP.   We find that direct use of an MILP-solver is significantly more effective than several baseline heuristics. We optimize the system runtime further with an introspective scheduling approach. We implement all these techniques into a new data system we call Saturn. Experiments with benchmark DL workloads show that Saturn achieves 39-49% lower model selection runtimes than typical current DL practice.

{{</citation>}}


### (32/52) LogGPT: Exploring ChatGPT for Log-Based Anomaly Detection (Jiaxing Qi et al., 2023)

{{<citation>}}

Jiaxing Qi, Shaohan Huang, Zhongzhi Luan, Carol Fung, Hailong Yang, Depei Qian. (2023)  
**LogGPT: Exploring ChatGPT for Log-Based Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-SE, cs.LG  
Keywords: Anomaly Detection, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2309.01189v1)  

---


**ABSTRACT**  
The increasing volume of log data produced by software-intensive systems makes it impractical to analyze them manually. Many deep learning-based methods have been proposed for log-based anomaly detection. These methods face several challenges such as high-dimensional and noisy log data, class imbalance, generalization, and model interpretability. Recently, ChatGPT has shown promising results in various domains. However, there is still a lack of study on the application of ChatGPT for log-based anomaly detection. In this work, we proposed LogGPT, a log-based anomaly detection framework based on ChatGPT. By leveraging the ChatGPT's language interpretation capabilities, LogGPT aims to explore the transferability of knowledge from large-scale corpora to log-based anomaly detection. We conduct experiments to evaluate the performance of LogGPT and compare it with three deep learning-based methods on BGL and Spirit datasets. LogGPT shows promising results and has good interpretability. This study provides preliminary insights into prompt-based models, such as ChatGPT, for the log-based anomaly detection task.

{{</citation>}}


### (33/52) Cognition-Mode Aware Variational Representation Learning Framework for Knowledge Tracing (Moyu Zhang et al., 2023)

{{<citation>}}

Moyu Zhang, Xinning Zhu, Chunhong Zhang, Feng Pan, Wenchen Qian, Hui Zhao. (2023)  
**Cognition-Mode Aware Variational Representation Learning Framework for Knowledge Tracing**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CY, cs-LG, cs.LG  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2309.01179v1)  

---


**ABSTRACT**  
The Knowledge Tracing (KT) task plays a crucial role in personalized learning, and its purpose is to predict student responses based on their historical practice behavior sequence. However, the KT task suffers from data sparsity, which makes it challenging to learn robust representations for students with few practice records and increases the risk of model overfitting. Therefore, in this paper, we propose a Cognition-Mode Aware Variational Representation Learning Framework (CMVF) that can be directly applied to existing KT methods. Our framework uses a probabilistic model to generate a distribution for each student, accounting for uncertainty in those with limited practice records, and estimate the student's distribution via variational inference (VI). In addition, we also introduce a cognition-mode aware multinomial distribution as prior knowledge that constrains the posterior student distributions learning, so as to ensure that students with similar cognition modes have similar distributions, avoiding overwhelming personalization for students with few practice records. At last, extensive experimental results confirm that CMVF can effectively aid existing KT methods in learning more robust student representations. Our code is available at https://github.com/zmy-9/CMVF.

{{</citation>}}


### (34/52) End-to-End Learning on Multimodal Knowledge Graphs (W. X. Wilcke et al., 2023)

{{<citation>}}

W. X. Wilcke, P. Bloem, V. de Boer, R. H. van t Veer. (2023)  
**End-to-End Learning on Multimodal Knowledge Graphs**  

---
Primary Category: cs.LG  
Categories: I-2-6; I-2-4, cs-AI, cs-LG, cs.LG  
Keywords: Knowledge Graph  
[Paper Link](http://arxiv.org/abs/2309.01169v1)  

---


**ABSTRACT**  
Knowledge graphs enable data scientists to learn end-to-end on heterogeneous knowledge. However, most end-to-end models solely learn from the relational information encoded in graphs' structure: raw values, encoded as literal nodes, are either omitted completely or treated as regular nodes without consideration for their values. In either case we lose potentially relevant information which could have otherwise been exploited by our learning methods. We propose a multimodal message passing network which not only learns end-to-end from the structure of graphs, but also from their possibly divers set of multimodal node features. Our model uses dedicated (neural) encoders to naturally learn embeddings for node features belonging to five different types of modalities, including numbers, texts, dates, images and geometries, which are projected into a joint representation space together with their relational information. We implement and demonstrate our model on node classification and link prediction for artificial and real-worlds datasets, and evaluate the effect that each modality has on the overall performance in an inverse ablation study. Our results indicate that end-to-end multimodal learning from any arbitrary knowledge graph is indeed possible, and that including multimodal information can significantly affect performance, but that much depends on the characteristics of the data.

{{</citation>}}


### (35/52) AutoML-GPT: Large Language Model for AutoML (Yun-Da Tsai et al., 2023)

{{<citation>}}

Yun-Da Tsai, Yu-Che Tsai, Bo-Wei Huang, Chun-Pai Yang, Shou-De Lin. (2023)  
**AutoML-GPT: Large Language Model for AutoML**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2309.01125v1)  

---


**ABSTRACT**  
With the emerging trend of GPT models, we have established a framework called AutoML-GPT that integrates a comprehensive set of tools and libraries. This framework grants users access to a wide range of data preprocessing techniques, feature engineering methods, and model selection algorithms. Through a conversational interface, users can specify their requirements, constraints, and evaluation metrics. Throughout the process, AutoML-GPT employs advanced techniques for hyperparameter optimization and model selection, ensuring that the resulting model achieves optimal performance. The system effectively manages the complexity of the machine learning pipeline, guiding users towards the best choices without requiring deep domain knowledge. Through our experimental results on diverse datasets, we have demonstrated that AutoML-GPT significantly reduces the time and effort required for machine learning tasks. Its ability to leverage the vast knowledge encoded in large language models enables it to provide valuable insights, identify potential pitfalls, and suggest effective solutions to common challenges faced during model training.

{{</citation>}}


### (36/52) Double Clipping: Less-Biased Variance Reduction in Off-Policy Evaluation (Jan Malte Lichtenberg et al., 2023)

{{<citation>}}

Jan Malte Lichtenberg, Alexander Buchholz, Giuseppe Di Benedetto, Matteo Ruffini, Ben London. (2023)  
**Double Clipping: Less-Biased Variance Reduction in Off-Policy Evaluation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2309.01120v1)  

---


**ABSTRACT**  
"Clipping" (a.k.a. importance weight truncation) is a widely used variance-reduction technique for counterfactual off-policy estimators. Like other variance-reduction techniques, clipping reduces variance at the cost of increased bias. However, unlike other techniques, the bias introduced by clipping is always a downward bias (assuming non-negative rewards), yielding a lower bound on the true expected reward. In this work we propose a simple extension, called $\textit{double clipping}$, which aims to compensate this downward bias and thus reduce the overall bias, while maintaining the variance reduction properties of the original estimator.

{{</citation>}}


### (37/52) M2HGCL: Multi-Scale Meta-Path Integrated Heterogeneous Graph Contrastive Learning (Yuanyuan Guo et al., 2023)

{{<citation>}}

Yuanyuan Guo, Yu Xia, Rui Wang, Rongcheng Duan, Lu Li, Jiangmeng Li. (2023)  
**M2HGCL: Multi-Scale Meta-Path Integrated Heterogeneous Graph Contrastive Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.01101v1)  

---


**ABSTRACT**  
Inspired by the successful application of contrastive learning on graphs, researchers attempt to impose graph contrastive learning approaches on heterogeneous information networks. Orthogonal to homogeneous graphs, the types of nodes and edges in heterogeneous graphs are diverse so that specialized graph contrastive learning methods are required. Most existing methods for heterogeneous graph contrastive learning are implemented by transforming heterogeneous graphs into homogeneous graphs, which may lead to ramifications that the valuable information carried by non-target nodes is undermined thereby exacerbating the performance of contrastive learning models. Additionally, current heterogeneous graph contrastive learning methods are mainly based on initial meta-paths given by the dataset, yet according to our deep-going exploration, we derive empirical conclusions: only initial meta-paths cannot contain sufficiently discriminative information; and various types of meta-paths can effectively promote the performance of heterogeneous graph contrastive learning methods. To this end, we propose a new multi-scale meta-path integrated heterogeneous graph contrastive learning (M2HGCL) model, which discards the conventional heterogeneity-homogeneity transformation and performs the graph contrastive learning in a joint manner. Specifically, we expand the meta-paths and jointly aggregate the direct neighbor information, the initial meta-path neighbor information and the expanded meta-path neighbor information to sufficiently capture discriminative information. A specific positive sampling strategy is further imposed to remedy the intrinsic deficiency of contrastive learning, i.e., the hard negative sample sampling issue. Through extensive experiments on three real-world datasets, we demonstrate that M2HGCL outperforms the current state-of-the-art baseline models.

{{</citation>}}


## cs.HC (1)



### (38/52) Immersive Technologies in Virtual Companions: A Systematic Literature Review (Ziaullah Momand et al., 2023)

{{<citation>}}

Ziaullah Momand, Jonathan H. Chan, Pornchai Mongkolnam. (2023)  
**Immersive Technologies in Virtual Companions: A Systematic Literature Review**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: Google  
[Paper Link](http://arxiv.org/abs/2309.01214v1)  

---


**ABSTRACT**  
The emergence of virtual companions is transforming the evolution of intelligent systems that effortlessly cater to the unique requirements of users. These advanced systems not only take into account the user present capabilities, preferences, and needs but also possess the capability to adapt dynamically to changes in the environment, as well as fluctuations in the users emotional state or behavior. A virtual companion is an intelligent software or application that offers support, assistance, and companionship across various aspects of users lives. Various enabling technologies are involved in building virtual companion, among these, Augmented Reality (AR), and Virtual Reality (VR) are emerging as transformative tools. While their potential for use in virtual companions or digital assistants is promising, their applications in these domains remain relatively unexplored. To address this gap, a systematic review was conducted to investigate the applications of VR, AR, and MR immersive technologies in the development of virtual companions. A comprehensive search across PubMed, Scopus, and Google Scholar yielded 28 relevant articles out of a pool of 644. The review revealed that immersive technologies, particularly VR and AR, play a significant role in creating digital assistants, offering a wide range of applications that brings various facilities in the individuals life in areas such as addressing social isolation, enhancing cognitive abilities and dementia care, facilitating education, and more. Additionally, AR and MR hold potential for enhancing Quality of life (QoL) within the context of virtual companion technology. The findings of this review provide a valuable foundation for further research in this evolving field.

{{</citation>}}


## eess.IV (2)



### (39/52) Spectral Adversarial MixUp for Few-Shot Unsupervised Domain Adaptation (Jiajin Zhang et al., 2023)

{{<citation>}}

Jiajin Zhang, Hanqing Chao, Amit Dhurandhar, Pin-Yu Chen, Ali Tajer, Yangyang Xu, Pingkun Yan. (2023)  
**Spectral Adversarial MixUp for Few-Shot Unsupervised Domain Adaptation**  

---
Primary Category: eess.IV  
Categories: cs-CV, cs-LG, eess-IV, eess.IV  
Keywords: Few-Shot  
[Paper Link](http://arxiv.org/abs/2309.01207v1)  

---


**ABSTRACT**  
Domain shift is a common problem in clinical applications, where the training images (source domain) and the test images (target domain) are under different distributions. Unsupervised Domain Adaptation (UDA) techniques have been proposed to adapt models trained in the source domain to the target domain. However, those methods require a large number of images from the target domain for model training. In this paper, we propose a novel method for Few-Shot Unsupervised Domain Adaptation (FSUDA), where only a limited number of unlabeled target domain samples are available for training. To accomplish this challenging task, first, a spectral sensitivity map is introduced to characterize the generalization weaknesses of models in the frequency domain. We then developed a Sensitivity-guided Spectral Adversarial MixUp (SAMix) method to generate target-style images to effectively suppresses the model sensitivity, which leads to improved model generalizability in the target domain. We demonstrated the proposed method and rigorously evaluated its performance on multiple tasks using several public datasets.

{{</citation>}}


### (40/52) Channel Attention Separable Convolution Network for Skin Lesion Segmentation (Changlu Guo et al., 2023)

{{<citation>}}

Changlu Guo, Jiangyan Dai, Marton Szemenyei, Yugen Yi. (2023)  
**Channel Attention Separable Convolution Network for Skin Lesion Segmentation**  

---
Primary Category: eess.IV  
Categories: cs-CV, eess-IV, eess.IV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.01072v1)  

---


**ABSTRACT**  
Skin cancer is a frequently occurring cancer in the human population, and it is very important to be able to diagnose malignant tumors in the body early. Lesion segmentation is crucial for monitoring the morphological changes of skin lesions, extracting features to localize and identify diseases to assist doctors in early diagnosis. Manual de-segmentation of dermoscopic images is error-prone and time-consuming, thus there is a pressing demand for precise and automated segmentation algorithms. Inspired by advanced mechanisms such as U-Net, DenseNet, Separable Convolution, Channel Attention, and Atrous Spatial Pyramid Pooling (ASPP), we propose a novel network called Channel Attention Separable Convolution Network (CASCN) for skin lesions segmentation. The proposed CASCN is evaluated on the PH2 dataset with limited images. Without excessive pre-/post-processing of images, CASCN achieves state-of-the-art performance on the PH2 dataset with Dice similarity coefficient of 0.9461 and accuracy of 0.9645.

{{</citation>}}


## cs.GR (1)



### (41/52) MAGMA: Music Aligned Generative Motion Autodecoder (Sohan Anisetty et al., 2023)

{{<citation>}}

Sohan Anisetty, Amit Raj, James Hays. (2023)  
**MAGMA: Music Aligned Generative Motion Autodecoder**  

---
Primary Category: cs.GR  
Categories: cs-CV, cs-GR, cs-MM, cs-SD, cs.GR, eess-AS  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2309.01202v1)  

---


**ABSTRACT**  
Mapping music to dance is a challenging problem that requires spatial and temporal coherence along with a continual synchronization with the music's progression. Taking inspiration from large language models, we introduce a 2-step approach for generating dance using a Vector Quantized-Variational Autoencoder (VQ-VAE) to distill motion into primitives and train a Transformer decoder to learn the correct sequencing of these primitives. We also evaluate the importance of music representations by comparing naive music feature extraction using Librosa to deep audio representations generated by state-of-the-art audio compression algorithms. Additionally, we train variations of the motion generator using relative and absolute positional encodings to determine the effect on generated motion quality when generating arbitrarily long sequence lengths. Our proposed approach achieve state-of-the-art results in music-to-motion generation benchmarks and enables the real-time generation of considerably longer motion sequences, the ability to chain multiple motion sequences seamlessly, and easy customization of motion sequences to meet style requirements.

{{</citation>}}


## cs.IR (3)



### (42/52) Pre-trained Neural Recommenders: A Transferable Zero-Shot Framework for Recommendation Systems (Junting Wang et al., 2023)

{{<citation>}}

Junting Wang, Adit Krishnan, Hari Sundaram, Yunzhe Li. (2023)  
**Pre-trained Neural Recommenders: A Transferable Zero-Shot Framework for Recommendation Systems**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-IR, cs.IR  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2309.01188v1)  

---


**ABSTRACT**  
Modern neural collaborative filtering techniques are critical to the success of e-commerce, social media, and content-sharing platforms. However, despite technical advances -- for every new application domain, we need to train an NCF model from scratch. In contrast, pre-trained vision and language models are routinely applied to diverse applications directly (zero-shot) or with limited fine-tuning. Inspired by the impact of pre-trained models, we explore the possibility of pre-trained recommender models that support building recommender systems in new domains, with minimal or no retraining, without the use of any auxiliary user or item information. Zero-shot recommendation without auxiliary information is challenging because we cannot form associations between users and items across datasets when there are no overlapping users or items. Our fundamental insight is that the statistical characteristics of the user-item interaction matrix are universally available across different domains and datasets. Thus, we use the statistical characteristics of the user-item interaction matrix to identify dataset-independent representations for users and items. We show how to learn universal (i.e., supporting zero-shot adaptation without user or item auxiliary information) representations for nodes and edges from the bipartite user-item interaction graph. We learn representations by exploiting the statistical properties of the interaction data, including user and item marginals, and the size and density distributions of their clusters.

{{</citation>}}


### (43/52) Large Language Models for Generative Recommendation: A Survey and Visionary Discussions (Lei Li et al., 2023)

{{<citation>}}

Lei Li, Yongfeng Zhang, Dugang Liu, Li Chen. (2023)  
**Large Language Models for Generative Recommendation: A Survey and Visionary Discussions**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs.IR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2309.01157v1)  

---


**ABSTRACT**  
Recent years have witnessed the wide adoption of large language models (LLM) in different fields, especially natural language processing and computer vision. Such a trend can also be observed in recommender systems (RS). However, most of related work treat LLM as a component of the conventional recommendation pipeline (e.g., as a feature extractor) which may not be able to fully leverage the generative power of LLM. Instead of separating the recommendation process into multiple stages such as score computation and re-ranking, this process can be simplified to one stage with LLM: directly generating recommendations from the complete pool of items. This survey reviews the progress, methods and future directions of LLM-based generative recommendation by examining three questions: 1) What generative recommendation is, 2) Why RS should advance to generative recommendation, and 3) How to implement LLM-based generative recommendation for various RS tasks. We hope that the survey can provide the context and guidance needed to explore this interesting and emerging topic.

{{</citation>}}


### (44/52) Multi-Relational Contrastive Learning for Recommendation (Wei Wei et al., 2023)

{{<citation>}}

Wei Wei, Lianghao Xia, Chao Huang. (2023)  
**Multi-Relational Contrastive Learning for Recommendation**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2309.01103v1)  

---


**ABSTRACT**  
Personalized recommender systems play a crucial role in capturing users' evolving preferences over time to provide accurate and effective recommendations on various online platforms. However, many recommendation models rely on a single type of behavior learning, which limits their ability to represent the complex relationships between users and items in real-life scenarios. In such situations, users interact with items in multiple ways, including clicking, tagging as favorite, reviewing, and purchasing. To address this issue, we propose the Relation-aware Contrastive Learning (RCL) framework, which effectively models dynamic interaction heterogeneity. The RCL model incorporates a multi-relational graph encoder that captures short-term preference heterogeneity while preserving the dedicated relation semantics for different types of user-item interactions. Moreover, we design a dynamic cross-relational memory network that enables the RCL model to capture users' long-term multi-behavior preferences and the underlying evolving cross-type behavior dependencies over time. To obtain robust and informative user representations with both commonality and diversity across multi-behavior interactions, we introduce a multi-relational contrastive learning paradigm with heterogeneous short- and long-term interest modeling. Our extensive experimental studies on several real-world datasets demonstrate the superiority of the RCL recommender system over various state-of-the-art baselines in terms of recommendation accuracy and effectiveness.

{{</citation>}}


## cs.DC (1)



### (45/52) FusionAI: Decentralized Training and Deploying LLMs with Massive Consumer-Level GPUs (Zhenheng Tang et al., 2023)

{{<citation>}}

Zhenheng Tang, Yuxin Wang, Xin He, Longteng Zhang, Xinglin Pan, Qiang Wang, Rongfei Zeng, Kaiyong Zhao, Shaohuai Shi, Bingsheng He, Xiaowen Chu. (2023)  
**FusionAI: Decentralized Training and Deploying LLMs with Massive Consumer-Level GPUs**  

---
Primary Category: cs.DC  
Categories: cs-AI, cs-DC, cs-LG, cs-NI, cs.DC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.01172v1)  

---


**ABSTRACT**  
The rapid growth of memory and computation requirements of large language models (LLMs) has outpaced the development of hardware, hindering people who lack large-scale high-end GPUs from training or deploying LLMs. However, consumer-level GPUs, which constitute a larger market share, are typically overlooked in LLM due to their weaker computing performance, smaller storage capacity, and lower communication bandwidth. Additionally, users may have privacy concerns when interacting with remote LLMs. In this paper, we envision a decentralized system unlocking the potential vast untapped consumer-level GPUs in pre-training, inference and fine-tuning of LLMs with privacy protection. However, this system faces critical challenges, including limited CPU and GPU memory, low network bandwidth, the variability of peer and device heterogeneity. To address these challenges, our system design incorporates: 1) a broker with backup pool to implement dynamic join and quit of computing providers; 2) task scheduling with hardware performance to improve system efficiency; 3) abstracting ML procedures into directed acyclic graphs (DAGs) to achieve model and task universality; 4) abstracting intermediate represention and execution planes to ensure compatibility of various devices and deep learning (DL) frameworks. Our performance analysis demonstrates that 50 RTX 3080 GPUs can achieve throughputs comparable to those of 4 H100 GPUs, which are significantly more expensive.

{{</citation>}}


## eess.AS (3)



### (46/52) Noise robust speech emotion recognition with signal-to-noise ratio adapting speech enhancement (Yu-Wen Chen et al., 2023)

{{<citation>}}

Yu-Wen Chen, Julia Hirschberg, Yu Tsao. (2023)  
**Noise robust speech emotion recognition with signal-to-noise ratio adapting speech enhancement**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2309.01164v1)  

---


**ABSTRACT**  
Speech emotion recognition (SER) often experiences reduced performance due to background noise. In addition, making a prediction on signals with only background noise could undermine user trust in the system. In this study, we propose a Noise Robust Speech Emotion Recognition system, NRSER. NRSER employs speech enhancement (SE) to effectively reduce the noise in input signals. Then, the signal-to-noise-ratio (SNR)-level detection structure and waveform reconstitution strategy are introduced to reduce the negative impact of SE on speech signals with no or little background noise. Our experimental results show that NRSER can effectively improve the noise robustness of the SER system, including preventing the system from making emotion recognition on signals consisting solely of background noise. Moreover, the proposed SNR-level detection structure can be used individually for tasks such as data selection.

{{</citation>}}


### (47/52) MSM-VC: High-fidelity Source Style Transfer for Non-Parallel Voice Conversion by Multi-scale Style Modeling (Zhichao Wang et al., 2023)

{{<citation>}}

Zhichao Wang, Xinsheng Wang, Qicong Xie, Tao Li, Lei Xie, Qiao Tian, Yuping Wang. (2023)  
**MSM-VC: High-fidelity Source Style Transfer for Non-Parallel Voice Conversion by Multi-scale Style Modeling**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2309.01142v1)  

---


**ABSTRACT**  
In addition to conveying the linguistic content from source speech to converted speech, maintaining the speaking style of source speech also plays an important role in the voice conversion (VC) task, which is essential in many scenarios with highly expressive source speech, such as dubbing and data augmentation. Previous work generally took explicit prosodic features or fixed-length style embedding extracted from source speech to model the speaking style of source speech, which is insufficient to achieve comprehensive style modeling and target speaker timbre preservation. Inspired by the style's multi-scale nature of human speech, a multi-scale style modeling method for the VC task, referred to as MSM-VC, is proposed in this paper. MSM-VC models the speaking style of source speech from different levels. To effectively convey the speaking style and meanwhile prevent timbre leakage from source speech to converted speech, each level's style is modeled by specific representation. Specifically, prosodic features, pre-trained ASR model's bottleneck features, and features extracted by a model trained with a self-supervised strategy are adopted to model the frame, local, and global-level styles, respectively. Besides, to balance the performance of source style modeling and target speaker timbre preservation, an explicit constraint module consisting of a pre-trained speech emotion recognition model and a speaker classifier is introduced to MSM-VC. This explicit constraint module also makes it possible to simulate the style transfer inference process during the training to improve the disentanglement ability and alleviate the mismatch between training and inference. Experiments performed on the highly expressive speech corpus demonstrate that MSM-VC is superior to the state-of-the-art VC methods for modeling source speech style while maintaining good speech quality and speaker similarity.

{{</citation>}}


### (48/52) Acoustic-to-articulatory inversion for dysarthric speech: Are pre-trained self-supervised representations favorable? (Sarthak Kumar Maharana et al., 2023)

{{<citation>}}

Sarthak Kumar Maharana, Krishna Kamal Adidam, Shoumik Nandi, Ajitesh Srivastava. (2023)  
**Acoustic-to-articulatory inversion for dysarthric speech: Are pre-trained self-supervised representations favorable?**  

---
Primary Category: eess.AS  
Categories: cs-LG, cs-SD, eess-AS, eess.AS  
Keywords: AI, LSTM  
[Paper Link](http://arxiv.org/abs/2309.01108v1)  

---


**ABSTRACT**  
$ $Acoustic-to-articulatory inversion (AAI) involves mapping from the acoustic space to the articulatory space. Signal-processing features like the MFCCs, have been widely used for the AAI task. For subjects with dysarthric speech, AAI is challenging because of an imprecise and indistinct pronunciation. In this work, we perform AAI for dysarthric speech using representations from pre-trained self-supervised learning (SSL) models. We demonstrate the impact of different pre-trained features on this challenging AAI task, at low-resource conditions. In addition, we also condition x-vectors to the extracted SSL features to train a BLSTM network. In the seen case, we experiment with three AAI training schemes (subject-specific, pooled, and fine-tuned). The results, consistent across training schemes, reveal that DeCoAR, in the fine-tuned scheme, achieves a relative improvement of the Pearson Correlation Coefficient (CC) by ${\sim}$1.81\% and ${\sim}$4.56\% for healthy controls and patients, respectively, over MFCCs. In the unseen case, we observe similar average trends for different SSL features. Overall, SSL networks like wav2vec, APC, and DeCoAR, which are trained with feature reconstruction or future timestep prediction tasks, perform well in predicting dysarthric articulatory trajectories.

{{</citation>}}


## quant-ph (1)



### (49/52) Financial Fraud Detection using Quantum Graph Neural Networks (Nouhaila Innan et al., 2023)

{{<citation>}}

Nouhaila Innan, Abhishek Sawaika, Ashim Dhor, Siddhant Dutta, Sairupa Thota, Husayn Gokal, Nandan Patel, Muhammad Al-Zafar Khan, Ioannis Theodonis, Mohamed Bennai. (2023)  
**Financial Fraud Detection using Quantum Graph Neural Networks**  

---
Primary Category: quant-ph  
Categories: cs-AI, cs-LG, quant-ph, quant-ph  
Keywords: Financial, Fraud Detection, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2309.01127v1)  

---


**ABSTRACT**  
Financial fraud detection is essential for preventing significant financial losses and maintaining the reputation of financial institutions. However, conventional methods of detecting financial fraud have limited effectiveness, necessitating the need for new approaches to improve detection rates. In this paper, we propose a novel approach for detecting financial fraud using Quantum Graph Neural Networks (QGNNs). QGNNs are a type of neural network that can process graph-structured data and leverage the power of Quantum Computing (QC) to perform computations more efficiently than classical neural networks. Our approach uses Variational Quantum Circuits (VQC) to enhance the performance of the QGNN. In order to evaluate the efficiency of our proposed method, we compared the performance of QGNNs to Classical Graph Neural Networks using a real-world financial fraud detection dataset. The results of our experiments showed that QGNNs achieved an AUC of $0.85$, which outperformed classical GNNs. Our research highlights the potential of QGNNs and suggests that QGNNs are a promising new approach for improving financial fraud detection.

{{</citation>}}


## q-bio.QM (1)



### (50/52) AI driven B-cell Immunotherapy Design (Bruna Moreira da Silva et al., 2023)

{{<citation>}}

Bruna Moreira da Silva, David B. Ascher, Nicholas Geard, Douglas E. V. Pires. (2023)  
**AI driven B-cell Immunotherapy Design**  

---
Primary Category: q-bio.QM  
Categories: cs-CE, cs-LG, q-bio-QM, q-bio.QM  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2309.01122v1)  

---


**ABSTRACT**  
Antibodies, a prominent class of approved biologics, play a crucial role in detecting foreign antigens. The effectiveness of antigen neutralisation and elimination hinges upon the strength, sensitivity, and specificity of the paratope-epitope interaction, which demands resource-intensive experimental techniques for characterisation. In recent years, artificial intelligence and machine learning methods have made significant strides, revolutionising the prediction of protein structures and their complexes. The past decade has also witnessed the evolution of computational approaches aiming to support immunotherapy design. This review focuses on the progress of machine learning-based tools and their frameworks in the domain of B-cell immunotherapy design, encompassing linear and conformational epitope prediction, paratope prediction, and antibody design. We mapped the most commonly used data sources, evaluation metrics, and method availability and thoroughly assessed their significance and limitations, discussing the main challenges ahead.

{{</citation>}}


## cs.CE (1)



### (51/52) MQENet: A Mesh Quality Evaluation Neural Network Based on Dynamic Graph Attention (Haoxuan Zhang et al., 2023)

{{<citation>}}

Haoxuan Zhang, Haisheng Li, Nan Li, Xiaochuan Wang. (2023)  
**MQENet: A Mesh Quality Evaluation Neural Network Based on Dynamic Graph Attention**  

---
Primary Category: cs.CE  
Categories: cs-CE, cs-LG, cs-NA, cs.CE, math-NA  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2309.01067v1)  

---


**ABSTRACT**  
With the development of computational fluid dynamics, the requirements for the fluid simulation accuracy in industrial applications have also increased. The quality of the generated mesh directly affects the simulation accuracy. However, previous mesh quality metrics and models cannot evaluate meshes comprehensively and objectively. To this end, we propose MQENet, a structured mesh quality evaluation neural network based on dynamic graph attention. MQENet treats the mesh evaluation task as a graph classification task for classifying the quality of the input structured mesh. To make graphs generated from structured meshes more informative, MQENet introduces two novel structured mesh preprocessing algorithms. These two algorithms can also improve the conversion efficiency of structured mesh data. Experimental results on the benchmark structured mesh dataset NACA-Market show the effectiveness of MQENet in the mesh quality evaluation task.

{{</citation>}}


## cs.NI (1)



### (52/52) Optimizing Mobile-Edge AI-Generated Everything (AIGX) Services by Prompt Engineering: Fundamental, Framework, and Case Study (Yinqiu Liu et al., 2023)

{{<citation>}}

Yinqiu Liu, Hongyang Du, Dusit Niyato, Jiawen Kang, Shuguang Cui, Xuemin Shen, Ping Zhang. (2023)  
**Optimizing Mobile-Edge AI-Generated Everything (AIGX) Services by Prompt Engineering: Fundamental, Framework, and Case Study**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: AI, ChatGPT, GPT, Generative AI  
[Paper Link](http://arxiv.org/abs/2309.01065v1)  

---


**ABSTRACT**  
As the next-generation paradigm for content creation, AI-Generated Content (AIGC), i.e., generating content automatically by Generative AI (GAI) based on user prompts, has gained great attention and success recently. With the ever-increasing power of GAI, especially the emergence of Pretrained Foundation Models (PFMs) that contain billions of parameters and prompt engineering methods (i.e., finding the best prompts for the given task), the application range of AIGC is rapidly expanding, covering various forms of information for human, systems, and networks, such as network designs, channel coding, and optimization solutions. In this article, we present the concept of mobile-edge AI-Generated Everything (AIGX). Specifically, we first review the building blocks of AIGX, the evolution from AIGC to AIGX, as well as practical AIGX applications. Then, we present a unified mobile-edge AIGX framework, which employs edge devices to provide PFM-empowered AIGX services and optimizes such services via prompt engineering. More importantly, we demonstrate that suboptimal prompts lead to poor generation quality, which adversely affects user satisfaction, edge network performance, and resource utilization. Accordingly, we conduct a case study, showcasing how to train an effective prompt optimizer using ChatGPT and investigating how much improvement is possible with prompt engineering in terms of user experience, quality of generation, and network performance.

{{</citation>}}
