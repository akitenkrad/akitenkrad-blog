---
draft: false
title: "arXiv @ 2023.10.31"
date: 2023-10-31
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2023"]
menu:
  sidebar:
    name: "arXiv @ 2023.10.31"
    identifier: arxiv_20231031
    parent: 202310_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.RO (2)](#csro-2)
- [cs.CV (19)](#cscv-19)
- [cs.CR (3)](#cscr-3)
- [cs.SD (1)](#cssd-1)
- [cs.CL (20)](#cscl-20)
- [cs.AI (7)](#csai-7)
- [cs.LG (19)](#cslg-19)
- [cs.HC (1)](#cshc-1)
- [eess.SY (1)](#eesssy-1)
- [cs.SE (2)](#csse-2)
- [cs.DS (1)](#csds-1)
- [cs.IR (2)](#csir-2)
- [cs.NE (1)](#csne-1)
- [stat.ML (1)](#statml-1)
- [cs.DM (1)](#csdm-1)
- [cs.SI (1)](#cssi-1)

## cs.RO (2)



### (1/82) Immersive 3D Simulator for Drone-as-a-Service (Jiamin Lin et al., 2023)

{{<citation>}}

Jiamin Lin, Balsam Alkouz, Athman Bouguettaya, Amani Abusafia. (2023)  
**Immersive 3D Simulator for Drone-as-a-Service**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2310.19199v1)  

---


**ABSTRACT**  
We propose a 3D simulator tailored for the Drone-as-a-Service framework. The simulator enables employing dynamic algorithms for addressing realistic delivery scenarios. We present the simulator's architectural design and its use of an energy consumption model for drone deliveries. We introduce two primary operational modes within the simulator: the edit mode and the runtime mode. Beyond its simulation capabilities, our simulator serves as a valuable data collection resource, facilitating the creation of datasets through simulated scenarios. Our simulator empowers researchers by providing an intuitive platform to visualize and interact with delivery environments. Moreover, it enables rigorous algorithm testing in a safe simulation setting, thus obviating the need for real-world drone deployments. Demo: https://youtu.be/HOLfo1JiFJ0

{{</citation>}}


### (2/82) Spacecraft Autonomous Decision-Planning for Collision Avoidance: a Reinforcement Learning Approach (Nicolas Bourriez et al., 2023)

{{<citation>}}

Nicolas Bourriez, Adrien Loizeau, Adam F. Abdin. (2023)  
**Spacecraft Autonomous Decision-Planning for Collision Avoidance: a Reinforcement Learning Approach**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-CV, cs-RO, cs.RO  
Keywords: AI, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.18966v1)  

---


**ABSTRACT**  
The space environment around the Earth is becoming increasingly populated by both active spacecraft and space debris. To avoid potential collision events, significant improvements in Space Situational Awareness (SSA) activities and Collision Avoidance (CA) technologies are allowing the tracking and maneuvering of spacecraft with increasing accuracy and reliability. However, these procedures still largely involve a high level of human intervention to make the necessary decisions. For an increasingly complex space environment, this decision-making strategy is not likely to be sustainable. Therefore, it is important to successfully introduce higher levels of automation for key Space Traffic Management (STM) processes to ensure the level of reliability needed for navigating a large number of spacecraft. These processes range from collision risk detection to the identification of the appropriate action to take and the execution of avoidance maneuvers. This work proposes an implementation of autonomous CA decision-making capabilities on spacecraft based on Reinforcement Learning (RL) techniques. A novel methodology based on a Partially Observable Markov Decision Process (POMDP) framework is developed to train the Artificial Intelligence (AI) system on board the spacecraft, considering epistemic and aleatory uncertainties. The proposed framework considers imperfect monitoring information about the status of the debris in orbit and allows the AI system to effectively learn stochastic policies to perform accurate Collision Avoidance Maneuvers (CAMs). The objective is to successfully delegate the decision-making process for autonomously implementing a CAM to the spacecraft without human intervention. This approach would allow for a faster response in the decision-making process and for highly decentralized operations.

{{</citation>}}


## cs.CV (19)



### (3/82) 3DMiner: Discovering Shapes from Large-Scale Unannotated Image Datasets (Ta-Ying Cheng et al., 2023)

{{<citation>}}

Ta-Ying Cheng, Matheus Gadelha, Soren Pirk, Thibault Groueix, Radomir Mech, Andrew Markham, Niki Trigoni. (2023)  
**3DMiner: Discovering Shapes from Large-Scale Unannotated Image Datasets**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.19188v1)  

---


**ABSTRACT**  
We present 3DMiner -- a pipeline for mining 3D shapes from challenging large-scale unannotated image datasets. Unlike other unsupervised 3D reconstruction methods, we assume that, within a large-enough dataset, there must exist images of objects with similar shapes but varying backgrounds, textures, and viewpoints. Our approach leverages the recent advances in learning self-supervised image representations to cluster images with geometrically similar shapes and find common image correspondences between them. We then exploit these correspondences to obtain rough camera estimates as initialization for bundle-adjustment. Finally, for every image cluster, we apply a progressive bundle-adjusting reconstruction method to learn a neural occupancy field representing the underlying shape. We show that this procedure is robust to several types of errors introduced in previous steps (e.g., wrong camera poses, images containing dissimilar shapes, etc.), allowing us to obtain shape and pose annotations for images in-the-wild. When using images from Pix3D chairs, our method is capable of producing significantly better results than state-of-the-art unsupervised 3D reconstruction techniques, both quantitatively and qualitatively. Furthermore, we show how 3DMiner can be applied to in-the-wild data by reconstructing shapes present in images from the LAION-5B dataset. Project Page: https://ttchengab.github.io/3dminerOfficial

{{</citation>}}


### (4/82) BirdSAT: Cross-View Contrastive Masked Autoencoders for Bird Species Classification and Mapping (Srikumar Sastry et al., 2023)

{{<citation>}}

Srikumar Sastry, Subash Khanal, Aayush Dhakal, Di Huang, Nathan Jacobs. (2023)  
**BirdSAT: Cross-View Contrastive Masked Autoencoders for Bird Species Classification and Mapping**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.19168v1)  

---


**ABSTRACT**  
We propose a metadata-aware self-supervised learning~(SSL)~framework useful for fine-grained classification and ecological mapping of bird species around the world. Our framework unifies two SSL strategies: Contrastive Learning~(CL) and Masked Image Modeling~(MIM), while also enriching the embedding space with metadata available with ground-level imagery of birds. We separately train uni-modal and cross-modal ViT on a novel cross-view global bird species dataset containing ground-level imagery, metadata (location, time), and corresponding satellite imagery. We demonstrate that our models learn fine-grained and geographically conditioned features of birds, by evaluating on two downstream tasks: fine-grained visual classification~(FGVC) and cross-modal retrieval. Pre-trained models learned using our framework achieve SotA performance on FGVC of iNAT-2021 birds and in transfer learning settings for CUB-200-2011 and NABirds datasets. Moreover, the impressive cross-modal retrieval performance of our model enables the creation of species distribution maps across any geographic region. The dataset and source code will be released at https://github.com/mvrl/BirdSAT}.

{{</citation>}}


### (5/82) Out-of-distribution Object Detection through Bayesian Uncertainty Estimation (Tianhao Zhang et al., 2023)

{{<citation>}}

Tianhao Zhang, Shenglin Wang, Nidhal Bouaynaya, Radu Calinescu, Lyudmila Mihaylova. (2023)  
**Out-of-distribution Object Detection through Bayesian Uncertainty Estimation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection  
[Paper Link](http://arxiv.org/abs/2310.19119v1)  

---


**ABSTRACT**  
The superior performance of object detectors is often established under the condition that the test samples are in the same distribution as the training data. However, in many practical applications, out-of-distribution (OOD) instances are inevitable and usually lead to uncertainty in the results. In this paper, we propose a novel, intuitive, and scalable probabilistic object detection method for OOD detection. Unlike other uncertainty-modeling methods that either require huge computational costs to infer the weight distributions or rely on model training through synthetic outlier data, our method is able to distinguish between in-distribution (ID) data and OOD data via weight parameter sampling from proposed Gaussian distributions based on pre-trained networks. We demonstrate that our Bayesian object detector can achieve satisfactory OOD identification performance by reducing the FPR95 score by up to 8.19% and increasing the AUROC score by up to 13.94% when trained on BDD100k and VOC datasets as the ID datasets and evaluated on COCO2017 dataset as the OOD dataset.

{{</citation>}}


### (6/82) Dynamic Task and Weight Prioritization Curriculum Learning for Multimodal Imagery (Huseyin Fuat Alsan et al., 2023)

{{<citation>}}

Huseyin Fuat Alsan, Taner Arsan. (2023)  
**Dynamic Task and Weight Prioritization Curriculum Learning for Multimodal Imagery**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.19109v1)  

---


**ABSTRACT**  
This paper explores post-disaster analytics using multimodal deep learning models trained with curriculum learning method. Studying post-disaster analytics is important as it plays a crucial role in mitigating the impact of disasters by providing timely and accurate insights into the extent of damage and the allocation of resources. We propose a curriculum learning strategy to enhance the performance of multimodal deep learning models. Curriculum learning emulates the progressive learning sequence in human education by training deep learning models on increasingly complex data. Our primary objective is to develop a curriculum-trained multimodal deep learning model, with a particular focus on visual question answering (VQA) capable of jointly processing image and text data, in conjunction with semantic segmentation for disaster analytics using the FloodNet\footnote{https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021} dataset. To achieve this, U-Net model is used for semantic segmentation and image encoding. A custom built text classifier is used for visual question answering. Existing curriculum learning methods rely on manually defined difficulty functions. We introduce a novel curriculum learning approach termed Dynamic Task and Weight Prioritization (DATWEP), which leverages a gradient-based method to automatically decide task difficulty during curriculum learning training, thereby eliminating the need for explicit difficulty computation. The integration of DATWEP into our multimodal model shows improvement on VQA performance. Source code is available at https://github.com/fualsan/DATWEP.

{{</citation>}}


### (7/82) Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery (Katie Z Luo et al., 2023)

{{<citation>}}

Katie Z Luo, Zhenzhen Liu, Xiangyu Chen, Yurong You, Sagie Benaim, Cheng Perng Phoo, Mark Campbell, Wen Sun, Bharath Hariharan, Kilian Q. Weinberger. (2023)  
**Reward Finetuning for Faster and More Accurate Unsupervised Object Discovery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.19080v1)  

---


**ABSTRACT**  
Recent advances in machine learning have shown that Reinforcement Learning from Human Feedback (RLHF) can improve machine learning models and align them with human preferences. Although very successful for Large Language Models (LLMs), these advancements have not had a comparable impact in research for autonomous vehicles -- where alignment with human expectations can be imperative. In this paper, we propose to adapt similar RL-based methods to unsupervised object discovery, i.e. learning to detect objects from LiDAR points without any training labels. Instead of labels, we use simple heuristics to mimic human feedback. More explicitly, we combine multiple heuristics into a simple reward function that positively correlates its score with bounding box accuracy, \ie, boxes containing objects are scored higher than those without. We start from the detector's own predictions to explore the space and reinforce boxes with high rewards through gradient updates. Empirically, we demonstrate that our approach is not only more accurate, but also orders of magnitudes faster to train compared to prior works on object discovery.

{{</citation>}}


### (8/82) Myriad: Large Multimodal Model by Applying Vision Experts for Industrial Anomaly Detection (Yuanze Li et al., 2023)

{{<citation>}}

Yuanze Li, Haolin Wang, Shihao Yuan, Ming Liu, Yiwen Guo, Chen Xu, Guangming Shi, Wangmeng Zuo. (2023)  
**Myriad: Large Multimodal Model by Applying Vision Experts for Industrial Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection, GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2310.19070v1)  

---


**ABSTRACT**  
Existing industrial anomaly detection (IAD) methods predict anomaly scores for both anomaly detection and localization. However, they struggle to perform a multi-turn dialog and detailed descriptions for anomaly regions, e.g., color, shape, and categories of industrial anomalies. Recently, large multimodal (i.e., vision and language) models (LMMs) have shown eminent perception abilities on multiple vision tasks such as image captioning, visual understanding, visual reasoning, etc., making it a competitive potential choice for more comprehensible anomaly detection. However, the knowledge about anomaly detection is absent in existing general LMMs, while training a specific LMM for anomaly detection requires a tremendous amount of annotated data and massive computation resources. In this paper, we propose a novel large multi-modal model by applying vision experts for industrial anomaly detection (dubbed Myriad), which leads to definite anomaly detection and high-quality anomaly description. Specifically, we adopt MiniGPT-4 as the base LMM and design an Expert Perception module to embed the prior knowledge from vision experts as tokens which are intelligible to Large Language Models (LLMs). To compensate for the errors and confusions of vision experts, we introduce a domain adapter to bridge the visual representation gaps between generic and industrial images. Furthermore, we propose a Vision Expert Instructor, which enables the Q-Former to generate IAD domain vision-language tokens according to vision expert prior. Extensive experiments on MVTec-AD and VisA benchmarks demonstrate that our proposed method not only performs favorably against state-of-the-art methods under the 1-class and few-shot settings, but also provide definite anomaly prediction along with detailed descriptions in IAD domain.

{{</citation>}}


### (9/82) Multimodal ChatGPT for Medical Applications: an Experimental Study of GPT-4V (Zhiling Yan et al., 2023)

{{<citation>}}

Zhiling Yan, Kai Zhang, Rong Zhou, Lifang He, Xiang Li, Lichao Sun. (2023)  
**Multimodal ChatGPT for Medical Applications: an Experimental Study of GPT-4V**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ChatGPT, GPT, GPT-4, QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2310.19061v1)  

---


**ABSTRACT**  
In this paper, we critically evaluate the capabilities of the state-of-the-art multimodal large language model, i.e., GPT-4 with Vision (GPT-4V), on Visual Question Answering (VQA) task. Our experiments thoroughly assess GPT-4V's proficiency in answering questions paired with images using both pathology and radiology datasets from 11 modalities (e.g. Microscopy, Dermoscopy, X-ray, CT, etc.) and fifteen objects of interests (brain, liver, lung, etc.). Our datasets encompass a comprehensive range of medical inquiries, including sixteen distinct question types. Throughout our evaluations, we devised textual prompts for GPT-4V, directing it to synergize visual and textual information. The experiments with accuracy score conclude that the current version of GPT-4V is not recommended for real-world diagnostics due to its unreliable and suboptimal accuracy in responding to diagnostic medical questions. In addition, we delineate seven unique facets of GPT-4V's behavior in medical VQA, highlighting its constraints within this complex arena. The complete details of our evaluation cases are accessible at https://github.com/ZhilingYan/GPT4V-Medical-Report.

{{</citation>}}


### (10/82) TESTA: Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding (Shuhuai Ren et al., 2023)

{{<citation>}}

Shuhuai Ren, Sishuo Chen, Shicheng Li, Xu Sun, Lu Hou. (2023)  
**TESTA: Temporal-Spatial Token Aggregation for Long-form Video-Language Understanding**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CL, cs-CV, cs.CV  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2310.19060v1)  

---


**ABSTRACT**  
Large-scale video-language pre-training has made remarkable strides in advancing video-language understanding tasks. However, the heavy computational burden of video encoding remains a formidable efficiency bottleneck, particularly for long-form videos. These videos contain massive visual tokens due to their inherent 3D properties and spatiotemporal redundancy, making it challenging to capture complex temporal and spatial relationships. To tackle this issue, we propose an efficient method called TEmporal-Spatial Token Aggregation (TESTA). TESTA condenses video semantics by adaptively aggregating similar frames, as well as similar patches within each frame. TESTA can reduce the number of visual tokens by 75% and thus accelerate video encoding. Building upon TESTA, we introduce a pre-trained video-language model equipped with a divided space-time token aggregation module in each video encoder block. We evaluate our model on five datasets for paragraph-to-video retrieval and long-form VideoQA tasks. Experimental results show that TESTA improves computing efficiency by 1.7 times, and achieves significant performance gains from its scalability in processing longer input frames, e.g., +13.7 R@1 on QuerYD and +6.5 R@1 on Condensed Movie.

{{</citation>}}


### (11/82) Uncovering Prototypical Knowledge for Weakly Open-Vocabulary Semantic Segmentation (Fei Zhang et al., 2023)

{{<citation>}}

Fei Zhang, Tianfei Zhou, Boyang Li, Hao He, Chaofan Ma, Tianjiao Zhang, Jiangchao Yao, Ya Zhang, Yanfeng Wang. (2023)  
**Uncovering Prototypical Knowledge for Weakly Open-Vocabulary Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.19001v1)  

---


**ABSTRACT**  
This paper studies the problem of weakly open-vocabulary semantic segmentation (WOVSS), which learns to segment objects of arbitrary classes using mere image-text pairs. Existing works turn to enhance the vanilla vision transformer by introducing explicit grouping recognition, i.e., employing several group tokens/centroids to cluster the image tokens and perform the group-text alignment. Nevertheless, these methods suffer from a granularity inconsistency regarding the usage of group tokens, which are aligned in the all-to-one v.s. one-to-one manners during the training and inference phases, respectively. We argue that this discrepancy arises from the lack of elaborate supervision for each group token. To bridge this granularity gap, this paper explores explicit supervision for the group tokens from the prototypical knowledge. To this end, this paper proposes the non-learnable prototypical regularization (NPR) where non-learnable prototypes are estimated from source features to serve as supervision and enable contrastive matching of the group tokens. This regularization encourages the group tokens to segment objects with less redundancy and capture more comprehensive semantic regions, leading to increased compactness and richness. Based on NPR, we propose the prototypical guidance segmentation network (PGSeg) that incorporates multi-modal regularization by leveraging prototypical sources from both images and texts at different levels, progressively enhancing the segmentation capability with diverse prototypical patterns. Experimental results show that our proposed method achieves state-of-the-art performance on several benchmark datasets. The source code is available at https://github.com/Ferenas/PGSeg.

{{</citation>}}


### (12/82) Blacksmith: Fast Adversarial Training of Vision Transformers via a Mixture of Single-step and Multi-step Methods (Mahdi Salmani et al., 2023)

{{<citation>}}

Mahdi Salmani, Alireza Dehghanpour Farashah, Mohammad Azizmalayeri, Mahdi Amiri, Navid Eslami, Mohammad Taghi Manzuri, Mohammad Hossein Rohban. (2023)  
**Blacksmith: Fast Adversarial Training of Vision Transformers via a Mixture of Single-step and Multi-step Methods**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-LG, cs.CV  
Keywords: Adversarial Training, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.18975v1)  

---


**ABSTRACT**  
Despite the remarkable success achieved by deep learning algorithms in various domains, such as computer vision, they remain vulnerable to adversarial perturbations. Adversarial Training (AT) stands out as one of the most effective solutions to address this issue; however, single-step AT can lead to Catastrophic Overfitting (CO). This scenario occurs when the adversarially trained network suddenly loses robustness against multi-step attacks like Projected Gradient Descent (PGD). Although several approaches have been proposed to address this problem in Convolutional Neural Networks (CNNs), we found out that they do not perform well when applied to Vision Transformers (ViTs). In this paper, we propose Blacksmith, a novel training strategy to overcome the CO problem, specifically in ViTs. Our approach utilizes either of PGD-2 or Fast Gradient Sign Method (FGSM) randomly in a mini-batch during the adversarial training of the neural network. This will increase the diversity of our training attacks, which could potentially mitigate the CO issue. To manage the increased training time resulting from this combination, we craft the PGD-2 attack based on only the first half of the layers, while FGSM is applied end-to-end. Through our experiments, we demonstrate that our novel method effectively prevents CO, achieves PGD-2 level performance, and outperforms other existing techniques including N-FGSM, which is the state-of-the-art method in fast training for CNNs.

{{</citation>}}


### (13/82) Analyzing Vision Transformers for Image Classification in Class Embedding Space (Martina G. Vilas et al., 2023)

{{<citation>}}

Martina G. Vilas, Timothy Schaumlöffel, Gemma Roig. (2023)  
**Analyzing Vision Transformers for Image Classification in Class Embedding Space**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Embedding, Image Classification, NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.18969v1)  

---


**ABSTRACT**  
Despite the growing use of transformer models in computer vision, a mechanistic understanding of these networks is still needed. This work introduces a method to reverse-engineer Vision Transformers trained to solve image classification tasks. Inspired by previous research in NLP, we demonstrate how the inner representations at any level of the hierarchy can be projected onto the learned class embedding space to uncover how these networks build categorical representations for their predictions. We use our framework to show how image tokens develop class-specific representations that depend on attention mechanisms and contextual information, and give insights on how self-attention and MLP layers differentially contribute to this categorical composition. We additionally demonstrate that this method (1) can be used to determine the parts of an image that would be important for detecting the class of interest, and (2) exhibits significant advantages over traditional linear probing approaches. Taken together, our results position our proposed framework as a powerful tool for mechanistic interpretability and explainability research.

{{</citation>}}


### (14/82) AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection (Qihang Zhou et al., 2023)

{{<citation>}}

Qihang Zhou, Guansong Pang, Yu Tian, Shibo He, Jiming Chen. (2023)  
**AnomalyCLIP: Object-agnostic Prompt Learning for Zero-shot Anomaly Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Anomaly Detection  
[Paper Link](http://arxiv.org/abs/2310.18961v1)  

---


**ABSTRACT**  
Zero-shot anomaly detection (ZSAD) requires detection models trained using auxiliary data to detect anomalies without any training sample in a target dataset. It is a crucial task when training data is not accessible due to various concerns, \eg, data privacy, yet it is challenging since the models need to generalize to anomalies across different domains where the appearance of foreground objects, abnormal regions, and background features, such as defects/tumors on different products/organs, can vary significantly. Recently large pre-trained vision-language models (VLMs), such as CLIP, have demonstrated strong zero-shot recognition ability in various vision tasks, including anomaly detection. However, their ZSAD performance is weak since the VLMs focus more on modeling the class semantics of the foreground objects rather than the abnormality/normality in the images. In this paper we introduce a novel approach, namely AnomalyCLIP, to adapt CLIP for accurate ZSAD across different domains. The key insight of AnomalyCLIP is to learn object-agnostic text prompts that capture generic normality and abnormality in an image regardless of its foreground objects. This allows our model to focus on the abnormal image regions rather than the object semantics, enabling generalized normality and abnormality recognition on diverse types of objects. Large-scale experiments on 17 real-world anomaly detection datasets show that AnomalyCLIP achieves superior zero-shot performance of detecting and segmenting anomalies in datasets of highly diverse class semantics from various defect inspection and medical imaging domains. Code will be made available at https://github.com/zqhang/AnomalyCLIP.

{{</citation>}}


### (15/82) Mask Propagation for Efficient Video Semantic Segmentation (Yuetian Weng et al., 2023)

{{<citation>}}

Yuetian Weng, Mingfei Han, Haoyu He, Mingjie Li, Lina Yao, Xiaojun Chang, Bohan Zhuang. (2023)  
**Mask Propagation for Efficient Video Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2310.18954v1)  

---


**ABSTRACT**  
Video Semantic Segmentation (VSS) involves assigning a semantic label to each pixel in a video sequence. Prior work in this field has demonstrated promising results by extending image semantic segmentation models to exploit temporal relationships across video frames; however, these approaches often incur significant computational costs. In this paper, we propose an efficient mask propagation framework for VSS, called MPVSS. Our approach first employs a strong query-based image segmentor on sparse key frames to generate accurate binary masks and class predictions. We then design a flow estimation module utilizing the learned queries to generate a set of segment-aware flow maps, each associated with a mask prediction from the key frame. Finally, the mask-flow pairs are warped to serve as the mask predictions for the non-key frames. By reusing predictions from key frames, we circumvent the need to process a large volume of video frames individually with resource-intensive segmentors, alleviating temporal redundancy and significantly reducing computational costs. Extensive experiments on VSPW and Cityscapes demonstrate that our mask propagation framework achieves SOTA accuracy and efficiency trade-offs. For instance, our best model with Swin-L backbone outperforms the SOTA MRCFA using MiT-B5 by 4.0% mIoU, requiring only 26% FLOPs on the VSPW dataset. Moreover, our framework reduces up to 4x FLOPs compared to the per-frame Mask2Former baseline with only up to 2% mIoU degradation on the Cityscapes validation set. Code is available at https://github.com/ziplab/MPVSS.

{{</citation>}}


### (16/82) Customize StyleGAN with One Hand Sketch (Shaocong Zhang, 2023)

{{<citation>}}

Shaocong Zhang. (2023)  
**Customize StyleGAN with One Hand Sketch**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2310.18949v1)  

---


**ABSTRACT**  
Generating images from human sketches typically requires dedicated networks trained from scratch. In contrast, the emergence of the pre-trained Vision-Language models (e.g., CLIP) has propelled generative applications based on controlling the output imagery of existing StyleGAN models with text inputs or reference images. Parallelly, our work proposes a framework to control StyleGAN imagery with a single user sketch. In particular, we learn a conditional distribution in the latent space of a pre-trained StyleGAN model via energy-based learning and propose two novel energy functions leveraging CLIP for cross-domain semantic supervision. Once trained, our model can generate multi-modal images semantically aligned with the input sketch. Quantitative evaluations on synthesized datasets have shown that our approach improves significantly from previous methods in the one-shot regime. The superiority of our method is further underscored when experimenting with a wide range of human sketches of diverse styles and poses. Surprisingly, our models outperform the previous baseline regarding both the range of sketch inputs and image qualities despite operating with a stricter setting: with no extra training data and single sketch input.

{{</citation>}}


### (17/82) CHAIN: Exploring Global-Local Spatio-Temporal Information for Improved Self-Supervised Video Hashing (Rukai Wei et al., 2023)

{{<citation>}}

Rukai Wei, Yu Liu, Jingkuan Song, Heng Cui, Yanzhao Xie, Ke Zhou. (2023)  
**CHAIN: Exploring Global-Local Spatio-Temporal Information for Improved Self-Supervised Video Hashing**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.18926v1)  

---


**ABSTRACT**  
Compressing videos into binary codes can improve retrieval speed and reduce storage overhead. However, learning accurate hash codes for video retrieval can be challenging due to high local redundancy and complex global dependencies between video frames, especially in the absence of labels. Existing self-supervised video hashing methods have been effective in designing expressive temporal encoders, but have not fully utilized the temporal dynamics and spatial appearance of videos due to less challenging and unreliable learning tasks. To address these challenges, we begin by utilizing the contrastive learning task to capture global spatio-temporal information of videos for hashing. With the aid of our designed augmentation strategies, which focus on spatial and temporal variations to create positive pairs, the learning framework can generate hash codes that are invariant to motion, scale, and viewpoint. Furthermore, we incorporate two collaborative learning tasks, i.e., frame order verification and scene change regularization, to capture local spatio-temporal details within video frames, thereby enhancing the perception of temporal structure and the modeling of spatio-temporal relationships. Our proposed Contrastive Hashing with Global-Local Spatio-temporal Information (CHAIN) outperforms state-of-the-art self-supervised video hashing methods on four video benchmark datasets. Our codes will be released.

{{</citation>}}


### (18/82) QWID: Quantized Weed Identification Deep neural network (Parikshit Singh Rathore, 2023)

{{<citation>}}

Parikshit Singh Rathore. (2023)  
**QWID: Quantized Weed Identification Deep neural network**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.18921v1)  

---


**ABSTRACT**  
In this paper, we present an efficient solution for weed classification in agriculture. We focus on optimizing model performance at inference while respecting the constraints of the agricultural domain. We propose a Quantized Deep Neural Network model that classifies a dataset of 9 weed classes using 8-bit integer (int8) quantization, a departure from standard 32-bit floating point (fp32) models. Recognizing the hardware resource limitations in agriculture, our model balances model size, inference time, and accuracy, aligning with practical requirements. We evaluate the approach on ResNet-50 and InceptionV3 architectures, comparing their performance against their int8 quantized versions. Transfer learning and fine-tuning are applied using the DeepWeeds dataset. The results show staggering model size and inference time reductions while maintaining accuracy in real-world production scenarios like Desktop, Mobile and Raspberry Pi. Our work sheds light on a promising direction for efficient AI in agriculture, holding potential for broader applications.   Code: https://github.com/parikshit14/QNN-for-weed

{{</citation>}}


### (19/82) Identifiable Contrastive Learning with Automatic Feature Importance Discovery (Qi Zhang et al., 2023)

{{<citation>}}

Qi Zhang, Yifei Wang, Yisen Wang. (2023)  
**Identifiable Contrastive Learning with Automatic Feature Importance Discovery**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.18904v1)  

---


**ABSTRACT**  
Existing contrastive learning methods rely on pairwise sample contrast $z_x^\top z_{x'}$ to learn data representations, but the learned features often lack clear interpretability from a human perspective. Theoretically, it lacks feature identifiability and different initialization may lead to totally different features. In this paper, we study a new method named tri-factor contrastive learning (triCL) that involves a 3-factor contrast in the form of $z_x^\top S z_{x'}$, where $S=\text{diag}(s_1,\dots,s_k)$ is a learnable diagonal matrix that automatically captures the importance of each feature. We show that by this simple extension, triCL can not only obtain identifiable features that eliminate randomness but also obtain more interpretable features that are ordered according to the importance matrix $S$. We show that features with high importance have nice interpretability by capturing common classwise features, and obtain superior performance when evaluated for image retrieval using a few features. The proposed triCL objective is general and can be applied to different contrastive learning methods like SimCLR and CLIP. We believe that it is a better alternative to existing 2-factor contrastive learning by improving its identifiability and interpretability with minimal overhead. Code is available at https://github.com/PKU-ML/Tri-factor-Contrastive-Learning.

{{</citation>}}


### (20/82) Emergence of Shape Bias in Convolutional Neural Networks through Activation Sparsity (Tianqin Li et al., 2023)

{{<citation>}}

Tianqin Li, Ziqi Wen, Yangfan Li, Tai Sing Lee. (2023)  
**Emergence of Shape Bias in Convolutional Neural Networks through Activation Sparsity**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.18894v1)  

---


**ABSTRACT**  
Current deep-learning models for object recognition are known to be heavily biased toward texture. In contrast, human visual systems are known to be biased toward shape and structure. What could be the design principles in human visual systems that led to this difference? How could we introduce more shape bias into the deep learning models? In this paper, we report that sparse coding, a ubiquitous principle in the brain, can in itself introduce shape bias into the network. We found that enforcing the sparse coding constraint using a non-differential Top-K operation can lead to the emergence of structural encoding in neurons in convolutional neural networks, resulting in a smooth decomposition of objects into parts and subparts and endowing the networks with shape bias. We demonstrated this emergence of shape bias and its functional benefits for different network structures with various datasets. For object recognition convolutional neural networks, the shape bias leads to greater robustness against style and pattern change distraction. For the image synthesis generative adversary networks, the emerged shape bias leads to more coherent and decomposable structures in the synthesized images. Ablation studies suggest that sparse codes tend to encode structures, whereas the more distributed codes tend to favor texture. Our code is host at the github repository: \url{https://github.com/Crazy-Jack/nips2023_shape_vs_texture}

{{</citation>}}


### (21/82) HDMNet: A Hierarchical Matching Network with Double Attention for Large-scale Outdoor LiDAR Point Cloud Registration (Weiyi Xue et al., 2023)

{{<citation>}}

Weiyi Xue, Fan Lu, Guang Chen. (2023)  
**HDMNet: A Hierarchical Matching Network with Double Attention for Large-scale Outdoor LiDAR Point Cloud Registration**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.18874v1)  

---


**ABSTRACT**  
Outdoor LiDAR point clouds are typically large-scale and complexly distributed. To achieve efficient and accurate registration, emphasizing the similarity among local regions and prioritizing global local-to-local matching is of utmost importance, subsequent to which accuracy can be enhanced through cost-effective fine registration. In this paper, a novel hierarchical neural network with double attention named HDMNet is proposed for large-scale outdoor LiDAR point cloud registration. Specifically, A novel feature consistency enhanced double-soft matching network is introduced to achieve two-stage matching with high flexibility while enlarging the receptive field with high efficiency in a patch-to patch manner, which significantly improves the registration performance. Moreover, in order to further utilize the sparse matching information from deeper layer, we develop a novel trainable embedding mask to incorporate the confidence scores of correspondences obtained from pose estimation of deeper layer, eliminating additional computations. The high-confidence keypoints in the sparser point cloud of the deeper layer correspond to a high-confidence spatial neighborhood region in shallower layer, which will receive more attention, while the features of non-key regions will be masked. Extensive experiments are conducted on two large-scale outdoor LiDAR point cloud datasets to demonstrate the high accuracy and efficiency of the proposed HDMNet.

{{</citation>}}


## cs.CR (3)



### (22/82) From Chatbots to PhishBots? -- Preventing Phishing scams created using ChatGPT, Google Bard and Claude (Sayak Saha Roy et al., 2023)

{{<citation>}}

Sayak Saha Roy, Poojitha Thota, Krishna Vamsi Naragam, Shirin Nilizadeh. (2023)  
**From Chatbots to PhishBots? -- Preventing Phishing scams created using ChatGPT, Google Bard and Claude**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs.CR  
Keywords: BERT, ChatGPT, GPT, Google, Language Model  
[Paper Link](http://arxiv.org/abs/2310.19181v1)  

---


**ABSTRACT**  
The advanced capabilities of Large Language Models (LLMs) have made them invaluable across various applications, from conversational agents and content creation to data analysis, research, and innovation. However, their effectiveness and accessibility also render them susceptible to abuse for generating malicious content, including phishing attacks. This study explores the potential of using four popular commercially available LLMs - ChatGPT (GPT 3.5 Turbo), GPT 4, Claude and Bard to generate functional phishing attacks using a series of malicious prompts. We discover that these LLMs can generate both phishing emails and websites that can convincingly imitate well-known brands, and also deploy a range of evasive tactics for the latter to elude detection mechanisms employed by anti-phishing systems. Notably, these attacks can be generated using unmodified, or "vanilla," versions of these LLMs, without requiring any prior adversarial exploits such as jailbreaking. As a countermeasure, we build a BERT based automated detection tool that can be used for the early detection of malicious prompts to prevent LLMs from generating phishing content attaining an accuracy of 97\% for phishing website prompts, and 94\% for phishing email prompts.

{{</citation>}}


### (23/82) RAIFLE: Reconstruction Attacks on Interaction-based Federated Learning with Active Data Manipulation (Dzung Pham et al., 2023)

{{<citation>}}

Dzung Pham, Shreyas Kulkarni, Amir Houmansadr. (2023)  
**RAIFLE: Reconstruction Attacks on Interaction-based Federated Learning with Active Data Manipulation**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs-LG, cs.CR  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.19163v1)  

---


**ABSTRACT**  
Federated learning (FL) has recently emerged as a privacy-preserving approach for machine learning in domains that rely on user interactions, particularly recommender systems (RS) and online learning to rank (OLTR). While there has been substantial research on the privacy of traditional FL, little attention has been paid to studying the privacy properties of these interaction-based FL (IFL) systems. In this work, we show that IFL can introduce unique challenges concerning user privacy, particularly when the central server has knowledge and control over the items that users interact with. Specifically, we demonstrate the threat of reconstructing user interactions by presenting RAIFLE, a general optimization-based reconstruction attack framework customized for IFL. RAIFLE employs Active Data Manipulation (ADM), a novel attack technique unique to IFL, where the server actively manipulates the training features of the items to induce adversarial behaviors in the local FL updates. We show that RAIFLE is more impactful than existing FL privacy attacks in the IFL context, and describe how it can undermine privacy defenses like secure aggregation and private information retrieval. Based on our findings, we propose and discuss countermeasure guidelines to mitigate our attack in the context of federated RS/OLTR specifically and IFL more broadly.

{{</citation>}}


### (24/82) Updated Standard for Secure Satellite Communications: Analysis of Satellites, Attack Vectors, Existing Standards, and Enterprise and Security Architectures (Rupok Chowdhury Protik, 2023)

{{<citation>}}

Rupok Chowdhury Protik. (2023)  
**Updated Standard for Secure Satellite Communications: Analysis of Satellites, Attack Vectors, Existing Standards, and Enterprise and Security Architectures**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Security  
[Paper Link](http://arxiv.org/abs/2310.19105v1)  

---


**ABSTRACT**  
Satellites play a vital role in remote communication where traditional communication mediums struggle to provide benefits over associated costs and efficiency. In recent years, satellite communication has achieved utter interest in the industry due to the achievement of high data rates through the massive deployment of LEO satellites. Because of the complex diversity in types of satellites, communication methodologies, technological obstacles, environmental limitations, elements in the entire ecosystem, massive financial impact, geopolitical conflict and domination, easier access to satellite communications, and various other reasons, the threat vectors are rising in the threat landscape. To achieve resilience against those, only technological solutions are not enough. An effective approach will be through security standards. However, there is a considerable gap in the industry regarding a generic security standard framework for satellite communication and space data systems. A few countries and space agencies have their own standard framework and private policies. However, many of those are either private, serve the specific requirements of specific missions, or have not been updated for a long time.   This project report will focus on identifying, categorizing, comparing, and assessing elements, threat landscape, enterprise security architectures, and available public standards of satellite communication and space data systems. After that, it will utilize the knowledge to propose an updated standard framework for secure satellite communications and space data systems.

{{</citation>}}


## cs.SD (1)



### (25/82) JEN-1 Composer: A Unified Framework for High-Fidelity Multi-Track Music Generation (Yao Yao et al., 2023)

{{<citation>}}

Yao Yao, Peike Li, Boyu Chen, Alex Wang. (2023)  
**JEN-1 Composer: A Unified Framework for High-Fidelity Multi-Track Music Generation**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-CV, cs-MM, cs-SD, cs.SD, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.19180v1)  

---


**ABSTRACT**  
With rapid advances in generative artificial intelligence, the text-to-music synthesis task has emerged as a promising direction for music generation from scratch. However, finer-grained control over multi-track generation remains an open challenge. Existing models exhibit strong raw generation capability but lack the flexibility to compose separate tracks and combine them in a controllable manner, differing from typical workflows of human composers. To address this issue, we propose JEN-1 Composer, a unified framework to efficiently model marginal, conditional, and joint distributions over multi-track music via a single model. JEN-1 Composer framework exhibits the capacity to seamlessly incorporate any diffusion-based music generation system, \textit{e.g.} Jen-1, enhancing its capacity for versatile multi-track music generation. We introduce a curriculum training strategy aimed at incrementally instructing the model in the transition from single-track generation to the flexible generation of multi-track combinations. During the inference, users have the ability to iteratively produce and choose music tracks that meet their preferences, subsequently creating an entire musical composition incrementally following the proposed Human-AI co-composition workflow. Quantitative and qualitative assessments demonstrate state-of-the-art performance in controllable and high-fidelity multi-track music synthesis. The proposed JEN-1 Composer represents a significant advance toward interactive AI-facilitated music creation and composition. Demos will be available at https://jenmusic.ai/audio-demos.

{{</citation>}}


## cs.CL (20)



### (26/82) Robustifying Language Models with Test-Time Adaptation (Noah Thomas McDermott et al., 2023)

{{<citation>}}

Noah Thomas McDermott, Junfeng Yang, Chengzhi Mao. (2023)  
**Robustifying Language Models with Test-Time Adaptation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.19177v1)  

---


**ABSTRACT**  
Large-scale language models achieved state-of-the-art performance over a number of language tasks. However, they fail on adversarial language examples, which are sentences optimized to fool the language models but with similar semantic meanings for humans. While prior work focuses on making the language model robust at training time, retraining for robustness is often unrealistic for large-scale foundation models. Instead, we propose to make the language models robust at test time. By dynamically adapting the input sentence with predictions from masked words, we show that we can reverse many language adversarial attacks. Since our approach does not require any training, it works for novel tasks at test time and can adapt to novel adversarial corruptions. Visualizations and empirical results on two popular sentence classification datasets demonstrate that our method can repair adversarial language attacks over 65% o

{{</citation>}}


### (27/82) Women Wearing Lipstick: Measuring the Bias Between an Object and Its Related Gender (Ahmed Sabir et al., 2023)

{{<citation>}}

Ahmed Sabir, Lluís Padró. (2023)  
**Women Wearing Lipstick: Measuring the Bias Between an Object and Its Related Gender**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-CV, cs.CL  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.19130v1)  

---


**ABSTRACT**  
In this paper, we investigate the impact of objects on gender bias in image captioning systems. Our results show that only gender-specific objects have a strong gender bias (e.g., women-lipstick). In addition, we propose a visual semantic-based gender score that measures the degree of bias and can be used as a plug-in for any image captioning system. Our experiments demonstrate the utility of the gender score, since we observe that our score can measure the bias relation between a caption and its related gender; therefore, our score can be used as an additional metric to the existing Object Gender Co-Occ approach. Code and data are publicly available at \url{https://github.com/ahmedssabir/GenderScore}.

{{</citation>}}


### (28/82) Unified Representation for Non-compositional and Compositional Expressions (Ziheng Zeng et al., 2023)

{{<citation>}}

Ziheng Zeng, Suma Bhat. (2023)  
**Unified Representation for Non-compositional and Compositional Expressions**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLU  
[Paper Link](http://arxiv.org/abs/2310.19127v1)  

---


**ABSTRACT**  
Accurate processing of non-compositional language relies on generating good representations for such expressions. In this work, we study the representation of language non-compositionality by proposing a language model, PIER, that builds on BART and can create semantically meaningful and contextually appropriate representations for English potentially idiomatic expressions (PIEs). PIEs are characterized by their non-compositionality and contextual ambiguity in their literal and idiomatic interpretations. Via intrinsic evaluation on embedding quality and extrinsic evaluation on PIE processing and NLU tasks, we show that representations generated by PIER result in 33% higher homogeneity score for embedding clustering than BART, whereas 3.12% and 3.29% gains in accuracy and sequence accuracy for PIE sense classification and span detection compared to the state-of-the-art IE representation model, GIEA. These gains are achieved without sacrificing PIER's performance on NLU tasks (+/- 1% accuracy) compared to BART.

{{</citation>}}


### (29/82) PACuna: Automated Fine-Tuning of Language Models for Particle Accelerators (Antonin Sulc et al., 2023)

{{<citation>}}

Antonin Sulc, Raimund Kammering, Annika Eichler, Tim Wilksen. (2023)  
**PACuna: Automated Fine-Tuning of Language Models for Particle Accelerators**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.19106v1)  

---


**ABSTRACT**  
Navigating the landscape of particle accelerators has become increasingly challenging with recent surges in contributions. These intricate devices challenge comprehension, even within individual facilities. To address this, we introduce PACuna, a fine-tuned language model refined through publicly available accelerator resources like conferences, pre-prints, and books. We automated data collection and question generation to minimize expert involvement and make the data publicly available. PACuna demonstrates proficiency in addressing intricate accelerator questions, validated by experts. Our approach shows adapting language models to scientific domains by fine-tuning technical texts and auto-generated corpora capturing the latest developments can further produce pre-trained models to answer some intricate questions that commercially available assistants cannot and can serve as intelligent assistants for individual facilities.

{{</citation>}}


### (30/82) Pushdown Layers: Encoding Recursive Structure in Transformer Language Models (Shikhar Murty et al., 2023)

{{<citation>}}

Shikhar Murty, Pratyusha Sharma, Jacob Andreas, Christopher D. Manning. (2023)  
**Pushdown Layers: Encoding Recursive Structure in Transformer Language Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GLUE, GPT, Language Model, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.19089v1)  

---


**ABSTRACT**  
Recursion is a prominent feature of human language, and fundamentally challenging for self-attention due to the lack of an explicit recursive-state tracking mechanism. Consequently, Transformer language models poorly capture long-tail recursive structure and exhibit sample-inefficient syntactic generalization. This work introduces Pushdown Layers, a new self-attention layer that models recursive state via a stack tape that tracks estimated depths of every token in an incremental parse of the observed prefix. Transformer LMs with Pushdown Layers are syntactic language models that autoregressively and synchronously update this stack tape as they predict new tokens, in turn using the stack tape to softly modulate attention over tokens -- for instance, learning to "skip" over closed constituents. When trained on a corpus of strings annotated with silver constituency parses, Transformers equipped with Pushdown Layers achieve dramatically better and 3-5x more sample-efficient syntactic generalization, while maintaining similar perplexities. Pushdown Layers are a drop-in replacement for standard self-attention. We illustrate this by finetuning GPT2-medium with Pushdown Layers on an automatically parsed WikiText-103, leading to improvements on several GLUE text classification tasks.

{{</citation>}}


### (31/82) Roles of Scaling and Instruction Tuning in Language Perception: Model vs. Human Attention (Changjiang Gao et al., 2023)

{{<citation>}}

Changjiang Gao, Shujian Huang, Jixing Li, Jiajun Chen. (2023)  
**Roles of Scaling and Instruction Tuning in Language Perception: Model vs. Human Attention**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: Attention, LLaMA  
[Paper Link](http://arxiv.org/abs/2310.19084v1)  

---


**ABSTRACT**  
Recent large language models (LLMs) have revealed strong abilities to understand natural language. Since most of them share the same basic structure, i.e. the transformer block, possible contributors to their success in the training process are scaling and instruction tuning. However, how these factors affect the models' language perception is unclear. This work compares the self-attention of several existing LLMs (LLaMA, Alpaca and Vicuna) in different sizes (7B, 13B, 30B, 65B), together with eye saccade, an aspect of human reading attention, to assess the effect of scaling and instruction tuning on language perception. Results show that scaling enhances the human resemblance and improves the effective attention by reducing the trivial pattern reliance, while instruction tuning does not. However, instruction tuning significantly enhances the models' sensitivity to instructions. We also find that current LLMs are consistently closer to non-native than native speakers in attention, suggesting a sub-optimal language perception of all models. Our code and data used in the analysis is available on GitHub.

{{</citation>}}


### (32/82) A Survey on Recent Named Entity Recognition and Relation Classification Methods with Focus on Few-Shot Learning Approaches (Sakher Alqaaidi et al., 2023)

{{<citation>}}

Sakher Alqaaidi, Elika Bozorgi. (2023)  
**A Survey on Recent Named Entity Recognition and Relation Classification Methods with Focus on Few-Shot Learning Approaches**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Few-Shot, Named Entity Recognition  
[Paper Link](http://arxiv.org/abs/2310.19055v1)  

---


**ABSTRACT**  
Named entity recognition and relation classification are key stages for extracting information from unstructured text. Several natural language processing applications utilize the two tasks, such as information retrieval, knowledge graph construction and completion, question answering and other domain-specific applications, such as biomedical data mining. We present a survey of recent approaches in the two tasks with focus on few-shot learning approaches. Our work compares the main approaches followed in the two paradigms. Additionally, we report the latest metric scores in the two tasks with a structured analysis that considers the results in the few-shot learning scope.

{{</citation>}}


### (33/82) ArBanking77: Intent Detection Neural Model and a New Dataset in Modern and Dialectical Arabic (Mustafa Jarrar et al., 2023)

{{<citation>}}

Mustafa Jarrar, Ahmet Birim, Mohammed Khalilia, Mustafa Erden, Sana Ghanem. (2023)  
**ArBanking77: Intent Detection Neural Model and a New Dataset in Modern and Dialectical Arabic**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Intent Detection, NLP  
[Paper Link](http://arxiv.org/abs/2310.19034v1)  

---


**ABSTRACT**  
This paper presents the ArBanking77, a large Arabic dataset for intent detection in the banking domain. Our dataset was arabized and localized from the original English Banking77 dataset, which consists of 13,083 queries to ArBanking77 dataset with 31,404 queries in both Modern Standard Arabic (MSA) and Palestinian dialect, with each query classified into one of the 77 classes (intents). Furthermore, we present a neural model, based on AraBERT, fine-tuned on ArBanking77, which achieved an F1-score of 0.9209 and 0.8995 on MSA and Palestinian dialect, respectively. We performed extensive experimentation in which we simulated low-resource settings, where the model is trained on a subset of the data and augmented with noisy queries to simulate colloquial terms, mistakes and misspellings found in real NLP systems, especially live chat queries. The data and the models are publicly available at https://sina.birzeit.edu/arbanking77.

{{</citation>}}


### (34/82) SALMA: Arabic Sense-Annotated Corpus and WSD Benchmarks (Mustafa Jarrar et al., 2023)

{{<citation>}}

Mustafa Jarrar, Sanad Malaysha, Tymaa Hammouda, Mohammed Khalilia. (2023)  
**SALMA: Arabic Sense-Annotated Corpus and WSD Benchmarks**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Word Sense Disambiguation  
[Paper Link](http://arxiv.org/abs/2310.19029v1)  

---


**ABSTRACT**  
SALMA, the first Arabic sense-annotated corpus, consists of ~34K tokens, which are all sense-annotated. The corpus is annotated using two different sense inventories simultaneously (Modern and Ghani). SALMA novelty lies in how tokens and senses are associated. Instead of linking a token to only one intended sense, SALMA links a token to multiple senses and provides a score to each sense. A smart web-based annotation tool was developed to support scoring multiple senses against a given word. In addition to sense annotations, we also annotated the corpus using six types of named entities. The quality of our annotations was assessed using various metrics (Kappa, Linear Weighted Kappa, Quadratic Weighted Kappa, Mean Average Error, and Root Mean Square Error), which show very high inter-annotator agreement. To establish a Word Sense Disambiguation baseline using our SALMA corpus, we developed an end-to-end Word Sense Disambiguation system using Target Sense Verification. We used this system to evaluate three Target Sense Verification models available in the literature. Our best model achieved an accuracy with 84.2% using Modern and 78.7% using Ghani. The full corpus and the annotation tool are open-source and publicly available at https://sina.birzeit.edu/salma/.

{{</citation>}}


### (35/82) TeacherLM: Teaching to Fish Rather Than Giving the Fish, Language Modeling Likewise (Nan He et al., 2023)

{{<citation>}}

Nan He, Hanyu Lai, Chenyang Zhao, Zirui Cheng, Junting Pan, Ruoyu Qin, Ruofan Lu, Rui Lu, Yunchen Zhang, Gangming Zhao, Zhaohui Hou, Zhiyuan Huang, Shaoqing Lu, Ding Liang, Mingjie Zhan. (2023)  
**TeacherLM: Teaching to Fish Rather Than Giving the Fish, Language Modeling Likewise**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BLOOM, Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2310.19019v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) exhibit impressive reasoning and data augmentation capabilities in various NLP tasks. However, what about small models? In this work, we propose TeacherLM-7.1B, capable of annotating relevant fundamentals, chain of thought, and common mistakes for most NLP samples, which makes annotation more than just an answer, thus allowing other models to learn "why" instead of just "what". The TeacherLM-7.1B model achieved a zero-shot score of 52.3 on MMLU, surpassing most models with over 100B parameters. Even more remarkable is its data augmentation ability. Based on TeacherLM-7.1B, we augmented 58 NLP datasets and taught various student models with different parameters from OPT and BLOOM series in a multi-task setting. The experimental results indicate that the data augmentation provided by TeacherLM has brought significant benefits. We will release the TeacherLM series of models and augmented datasets as open-source.

{{</citation>}}


### (36/82) Bipartite Graph Pre-training for Unsupervised Extractive Summarization with Graph Convolutional Auto-Encoders (Qianren Mao et al., 2023)

{{<citation>}}

Qianren Mao, Shaobo Zhao, Jiarui Li, Xiaolei Gu, Shizhu He, Bo Li, Jianxin Li. (2023)  
**Bipartite Graph Pre-training for Unsupervised Extractive Summarization with Graph Convolutional Auto-Encoders**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, Summarization  
[Paper Link](http://arxiv.org/abs/2310.18992v1)  

---


**ABSTRACT**  
Pre-trained sentence representations are crucial for identifying significant sentences in unsupervised document extractive summarization. However, the traditional two-step paradigm of pre-training and sentence-ranking, creates a gap due to differing optimization objectives. To address this issue, we argue that utilizing pre-trained embeddings derived from a process specifically designed to optimize cohensive and distinctive sentence representations helps rank significant sentences. To do so, we propose a novel graph pre-training auto-encoder to obtain sentence embeddings by explicitly modelling intra-sentential distinctive features and inter-sentential cohesive features through sentence-word bipartite graphs. These pre-trained sentence representations are then utilized in a graph-based ranking algorithm for unsupervised summarization. Our method produces predominant performance for unsupervised summarization frameworks by providing summary-worthy sentence representations. It surpasses heavy BERT- or RoBERTa-based sentence representations in downstream tasks.

{{</citation>}}


### (37/82) EtiCor: Corpus for Analyzing LLMs for Etiquettes (Ashutosh Dwivedi et al., 2023)

{{<citation>}}

Ashutosh Dwivedi, Pradhyumna Lavania, Ashutosh Modi. (2023)  
**EtiCor: Corpus for Analyzing LLMs for Etiquettes**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: Falcon, GPT, GPT-3.5  
[Paper Link](http://arxiv.org/abs/2310.18974v1)  

---


**ABSTRACT**  
Etiquettes are an essential ingredient of day-to-day interactions among people. Moreover, etiquettes are region-specific, and etiquettes in one region might contradict those in other regions. In this paper, we propose EtiCor, an Etiquettes Corpus, having texts about social norms from five different regions across the globe. The corpus provides a test bed for evaluating LLMs for knowledge and understanding of region-specific etiquettes. Additionally, we propose the task of Etiquette Sensitivity. We experiment with state-of-the-art LLMs (Delphi, Falcon40B, and GPT-3.5). Initial results indicate that LLMs, mostly fail to understand etiquettes from regions from non-Western world.

{{</citation>}}


### (38/82) S2F-NER: Exploring Sequence-to-Forest Generation for Complex Entity Recognition (Yongxiu Xu et al., 2023)

{{<citation>}}

Yongxiu Xu, Heyan Huang, Yue Hu. (2023)  
**S2F-NER: Exploring Sequence-to-Forest Generation for Complex Entity Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NER, Named Entity Recognition, Seq2Seq  
[Paper Link](http://arxiv.org/abs/2310.18944v1)  

---


**ABSTRACT**  
Named Entity Recognition (NER) remains challenging due to the complex entities, like nested, overlapping, and discontinuous entities. Existing approaches, such as sequence-to-sequence (Seq2Seq) generation and span-based classification, have shown impressive performance on various NER subtasks, but they are difficult to scale to datasets with longer input text because of either exposure bias issue or inefficient computation. In this paper, we propose a novel Sequence-to-Forest generation paradigm, S2F-NER, which can directly extract entities in sentence via a Forest decoder that decode multiple entities in parallel rather than sequentially. Specifically, our model generate each path of each tree in forest autoregressively, where the maximum depth of each tree is three (which is the shortest feasible length for complex NER and is far smaller than the decoding length of Seq2Seq). Based on this novel paradigm, our model can elegantly mitigates the exposure bias problem and keep the simplicity of Seq2Seq. Experimental results show that our model significantly outperforms the baselines on three discontinuous NER datasets and on two nested NER datasets, especially for discontinuous entity recognition.

{{</citation>}}


### (39/82) Retrofitting Light-weight Language Models for Emotions using Supervised Contrastive Learning (Sapan Shah et al., 2023)

{{<citation>}}

Sapan Shah, Sreedhar Reddy, Pushpak Bhattacharyya. (2023)  
**Retrofitting Light-weight Language Models for Emotions using Supervised Contrastive Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Contrastive Learning, Language Model  
[Paper Link](http://arxiv.org/abs/2310.18930v1)  

---


**ABSTRACT**  
We present a novel retrofitting method to induce emotion aspects into pre-trained language models (PLMs) such as BERT and RoBERTa. Our method updates pre-trained network weights using contrastive learning so that the text fragments exhibiting similar emotions are encoded nearby in the representation space, and the fragments with different emotion content are pushed apart. While doing so, it also ensures that the linguistic knowledge already present in PLMs is not inadvertently perturbed. The language models retrofitted by our method, i.e., BERTEmo and RoBERTaEmo, produce emotion-aware text representations, as evaluated through different clustering and retrieval metrics. For the downstream tasks on sentiment analysis and sarcasm detection, they perform better than their pre-trained counterparts (about 1% improvement in F1-score) and other existing approaches. Additionally, a more significant boost in performance is observed for the retrofitted models over pre-trained ones in few-shot learning setting.

{{</citation>}}


### (40/82) Debiasing Algorithm through Model Adaptation (Tomasz Limisiewicz et al., 2023)

{{<citation>}}

Tomasz Limisiewicz, David Mareček, Tomáš Musil. (2023)  
**Debiasing Algorithm through Model Adaptation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL, stat-ML  
Keywords: LLaMA  
[Paper Link](http://arxiv.org/abs/2310.18913v1)  

---


**ABSTRACT**  
Large language models are becoming the go-to solution for various language tasks. However, with growing capacity, models are prone to rely on spurious correlations stemming from biases and stereotypes present in the training data. This work proposes a novel method for detecting and mitigating gender bias in language models. We perform causal analysis to identify problematic model components and discover that mid-upper feed-forward layers are most prone to convey biases. Based on the analysis results, we adapt the model by multiplying these layers by a linear projection. Our titular method, DAMA, significantly decreases bias as measured by diverse metrics while maintaining the model's performance on downstream tasks. We release code for our method and models, which retrain LLaMA's state-of-the-art performance while being significantly less biased.

{{</citation>}}


### (41/82) Stacking the Odds: Transformer-Based Ensemble for AI-Generated Text Detection (Duke Nguyen et al., 2023)

{{<citation>}}

Duke Nguyen, Khaing Myat Noe Naing, Aditya Joshi. (2023)  
**Stacking the Odds: Transformer-Based Ensemble for AI-Generated Text Detection**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: AI, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2310.18906v1)  

---


**ABSTRACT**  
This paper reports our submission under the team name `SynthDetectives' to the ALTA 2023 Shared Task. We use a stacking ensemble of Transformers for the task of AI-generated text detection. Our approach is novel in terms of its choice of models in that we use accessible and lightweight models in the ensemble. We show that ensembling the models results in an improved accuracy in comparison with using them individually. Our approach achieves an accuracy score of 0.9555 on the official test data provided by the shared task organisers.

{{</citation>}}


### (42/82) Pre-trained Speech Processing Models Contain Human-Like Biases that Propagate to Speech Emotion Recognition (Isaac Slaughter et al., 2023)

{{<citation>}}

Isaac Slaughter, Craig Greenberg, Reva Schwartz, Aylin Caliskan. (2023)  
**Pre-trained Speech Processing Models Contain Human-Like Biases that Propagate to Speech Emotion Recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-LG, cs-SD, cs.CL, eess-AS  
Keywords: BERT, Bias, Embedding, Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2310.18877v1)  

---


**ABSTRACT**  
Previous work has established that a person's demographics and speech style affect how well speech processing models perform for them. But where does this bias come from? In this work, we present the Speech Embedding Association Test (SpEAT), a method for detecting bias in one type of model used for many speech tasks: pre-trained models. The SpEAT is inspired by word embedding association tests in natural language processing, which quantify intrinsic bias in a model's representations of different concepts, such as race or valence (something's pleasantness or unpleasantness) and capture the extent to which a model trained on large-scale socio-cultural data has learned human-like biases. Using the SpEAT, we test for six types of bias in 16 English speech models (including 4 models also trained on multilingual data), which come from the wav2vec 2.0, HuBERT, WavLM, and Whisper model families. We find that 14 or more models reveal positive valence (pleasantness) associations with abled people over disabled people, with European-Americans over African-Americans, with females over males, with U.S. accented speakers over non-U.S. accented speakers, and with younger people over older people. Beyond establishing that pre-trained speech models contain these biases, we also show that they can have real world effects. We compare biases found in pre-trained models to biases in downstream models adapted to the task of Speech Emotion Recognition (SER) and find that in 66 of the 96 tests performed (69%), the group that is more associated with positive valence as indicated by the SpEAT also tends to be predicted as speaking with higher valence by the downstream model. Our work provides evidence that, like text and image-based models, pre-trained speech based-models frequently learn human-like biases. Our work also shows that bias found in pre-trained models can propagate to the downstream task of SER.

{{</citation>}}


### (43/82) Prompt-Engineering and Transformer-based Question Generation and Evaluation (Rubaba Amyeen, 2023)

{{<citation>}}

Rubaba Amyeen. (2023)  
**Prompt-Engineering and Transformer-based Question Generation and Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: BERT, LLaMA, Question Generation, Transformer  
[Paper Link](http://arxiv.org/abs/2310.18867v1)  

---


**ABSTRACT**  
Question generation has numerous applications in the educational context. Question generation can prove helpful for students when reviewing content and testing themselves. Furthermore, a question generation model can aid teachers by lessening the burden of creating assessments and other practice material. This paper aims to find the best method to generate questions from textual data through a transformer model and prompt engineering. In this research, we finetuned a pretrained distilBERT model on the SQuAD question answering dataset to generate questions. In addition to training a transformer model, prompt engineering was applied to generate questions effectively using the LLaMA model. The generated questions were compared against the baseline questions in the SQuAD dataset to evaluate the effectiveness of four different prompts. All four prompts demonstrated over 60% similarity on average. Of the prompt-generated questions, 30% achieved a high similarity score greater than 70%.

{{</citation>}}


### (44/82) MUST: A Multilingual Student-Teacher Learning approach for low-resource speech recognition (Muhammad Umar Farooq et al., 2023)

{{<citation>}}

Muhammad Umar Farooq, Rehan Ahmad, Thomas Hain. (2023)  
**MUST: A Multilingual Student-Teacher Learning approach for low-resource speech recognition**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-SD, cs.CL, eess-AS  
Keywords: Multilingual  
[Paper Link](http://arxiv.org/abs/2310.18865v1)  

---


**ABSTRACT**  
Student-teacher learning or knowledge distillation (KD) has been previously used to address data scarcity issue for training of speech recognition (ASR) systems. However, a limitation of KD training is that the student model classes must be a proper or improper subset of the teacher model classes. It prevents distillation from even acoustically similar languages if the character sets are not same. In this work, the aforementioned limitation is addressed by proposing a MUltilingual Student-Teacher (MUST) learning which exploits a posteriors mapping approach. A pre-trained mapping model is used to map posteriors from a teacher language to the student language ASR. These mapped posteriors are used as soft labels for KD learning. Various teacher ensemble schemes are experimented to train an ASR model for low-resource languages. A model trained with MUST learning reduces relative character error rate (CER) up to 9.5% in comparison with a baseline monolingual ASR.

{{</citation>}}


### (45/82) Counterfactually Probing Language Identity in Multilingual Models (Anirudh Srinivasan et al., 2023)

{{<citation>}}

Anirudh Srinivasan, Venkata S Govindarajan, Kyle Mahowald. (2023)  
**Counterfactually Probing Language Identity in Multilingual Models**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: BERT, Multilingual  
[Paper Link](http://arxiv.org/abs/2310.18862v1)  

---


**ABSTRACT**  
Techniques in causal analysis of language models illuminate how linguistic information is organized in LLMs. We use one such technique, AlterRep, a method of counterfactual probing, to explore the internal structure of multilingual models (mBERT and XLM-R). We train a linear classifier on a binary language identity task, to classify tokens between Language X and Language Y. Applying a counterfactual probing procedure, we use the classifier weights to project the embeddings into the null space and push the resulting embeddings either in the direction of Language X or Language Y. Then we evaluate on a masked language modeling task. We find that, given a template in Language X, pushing towards Language Y systematically increases the probability of Language Y words, above and beyond a third-party control language. But it does not specifically push the model towards translation-equivalent words in Language Y. Pushing towards Language X (the same direction as the template) has a minimal effect, but somewhat degrades these models. Overall, we take these results as further evidence of the rich structure of massive multilingual language models, which include both a language-specific and language-general component. And we show that counterfactual probing can be fruitfully applied to multilingual models.

{{</citation>}}


## cs.AI (7)



### (46/82) Predicting recovery following stroke: deep learning, multimodal data and feature selection using explainable AI (Adam White et al., 2023)

{{<citation>}}

Adam White, Margarita Saranti, Artur d'Avila Garcez, Thomas M. H. Hope, Cathy J. Price, Howard Bowman. (2023)  
**Predicting recovery following stroke: deep learning, multimodal data and feature selection using explainable AI**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CY, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.19174v1)  

---


**ABSTRACT**  
Machine learning offers great potential for automated prediction of post-stroke symptoms and their response to rehabilitation. Major challenges for this endeavour include the very high dimensionality of neuroimaging data, the relatively small size of the datasets available for learning, and how to effectively combine neuroimaging and tabular data (e.g. demographic information and clinical characteristics). This paper evaluates several solutions based on two strategies. The first is to use 2D images that summarise MRI scans. The second is to select key features that improve classification accuracy. Additionally, we introduce the novel approach of training a convolutional neural network (CNN) on images that combine regions-of-interest extracted from MRIs, with symbolic representations of tabular data. We evaluate a series of CNN architectures (both 2D and a 3D) that are trained on different representations of MRI and tabular data, to predict whether a composite measure of post-stroke spoken picture description ability is in the aphasic or non-aphasic range. MRI and tabular data were acquired from 758 English speaking stroke survivors who participated in the PLORAS study. The classification accuracy for a baseline logistic regression was 0.678 for lesion size alone, rising to 0.757 and 0.813 when initial symptom severity and recovery time were successively added. The highest classification accuracy 0.854 was observed when 8 regions-of-interest was extracted from each MRI scan and combined with lesion size, initial severity and recovery time in a 2D Residual Neural Network.Our findings demonstrate how imaging and tabular data can be combined for high post-stroke classification accuracy, even when the dataset is small in machine learning terms. We conclude by proposing how the current models could be improved to achieve even higher levels of accuracy using images from hospital scanners.

{{</citation>}}


### (47/82) Web3 Meets AI Marketplace: Exploring Opportunities, Analyzing Challenges, and Suggesting Solutions (Peihao Li, 2023)

{{<citation>}}

Peihao Li. (2023)  
**Web3 Meets AI Marketplace: Exploring Opportunities, Analyzing Challenges, and Suggesting Solutions**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-CE, cs-CR, cs-GT, cs-NI, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.19099v1)  

---


**ABSTRACT**  
Web3 and AI have been among the most discussed fields over the recent years, with substantial hype surrounding each field's potential to transform the world as we know it. However, as the hype settles, it's evident that neither AI nor Web3 can address all challenges independently. Consequently, the intersection of AI and Web3 is gaining increased attention, emerging as a new field with the potential to address the limitations of each. In this article, we will focus on the integration of web3 and the AI marketplace, where AI services and products can be provided in a decentralized manner (DeAI). A comprehensive review is provided by summarizing the opportunities and challenges on this topic. Additionally, we offer analyses and solutions to address these challenges. We've developed a framework that lets users pay with any kind of cryptocurrency to get AI services. Additionally, they can also enjoy AI services for free on our platform by simply locking up their assets temporarily in the protocol. This unique approach is a first in the industry. Before this, offering free AI services in the web3 community wasn't possible. Our solution opens up exciting opportunities for the AI marketplace in the web3 space to grow and be widely adopted.

{{</citation>}}


### (48/82) A Unique Training Strategy to Enhance Language Models Capabilities for Health Mention Detection from Social Media Content (Pervaiz Iqbal Khan et al., 2023)

{{<citation>}}

Pervaiz Iqbal Khan, Muhammad Nabeel Asim, Andreas Dengel, Sheraz Ahmed. (2023)  
**A Unique Training Strategy to Enhance Language Models Capabilities for Health Mention Detection from Social Media Content**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Language Model, Social Media  
[Paper Link](http://arxiv.org/abs/2310.19057v1)  

---


**ABSTRACT**  
An ever-increasing amount of social media content requires advanced AI-based computer programs capable of extracting useful information. Specifically, the extraction of health-related content from social media is useful for the development of diverse types of applications including disease spread, mortality rate prediction, and finding the impact of diverse types of drugs on diverse types of diseases. Language models are competent in extracting the syntactic and semantics of text. However, they face a hard time extracting similar patterns from social media texts. The primary reason for this shortfall lies in the non-standardized writing style commonly employed by social media users. Following the need for an optimal language model competent in extracting useful patterns from social media text, the key goal of this paper is to train language models in such a way that they learn to derive generalized patterns. The key goal is achieved through the incorporation of random weighted perturbation and contrastive learning strategies. On top of a unique training strategy, a meta predictor is proposed that reaps the benefits of 5 different language models for discriminating posts of social media text into non-health and health-related classes. Comprehensive experimentation across 3 public benchmark datasets reveals that the proposed training strategy improves the performance of the language models up to 3.87%, in terms of F1-score, as compared to their performance with traditional training. Furthermore, the proposed meta predictor outperforms existing health mention classification predictors across all 3 benchmark datasets.

{{</citation>}}


### (49/82) DCQA: Document-Level Chart Question Answering towards Complex Reasoning and Common-Sense Understanding (Anran Wu et al., 2023)

{{<citation>}}

Anran Wu, Luwei Xiao, Xingjiao Wu, Shuwen Yang, Junjie Xu, Zisong Zhuang, Nian Xie, Cheng Jin, Liang He. (2023)  
**DCQA: Document-Level Chart Question Answering towards Complex Reasoning and Common-Sense Understanding**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: OCR, QA, Question Answering, Reasoning  
[Paper Link](http://arxiv.org/abs/2310.18983v1)  

---


**ABSTRACT**  
Visually-situated languages such as charts and plots are omnipresent in real-world documents. These graphical depictions are human-readable and are often analyzed in visually-rich documents to address a variety of questions that necessitate complex reasoning and common-sense responses. Despite the growing number of datasets that aim to answer questions over charts, most only address this task in isolation, without considering the broader context of document-level question answering. Moreover, such datasets lack adequate common-sense reasoning information in their questions. In this work, we introduce a novel task named document-level chart question answering (DCQA). The goal of this task is to conduct document-level question answering, extracting charts or plots in the document via document layout analysis (DLA) first and subsequently performing chart question answering (CQA). The newly developed benchmark dataset comprises 50,010 synthetic documents integrating charts in a wide range of styles (6 styles in contrast to 3 for PlotQA and ChartQA) and includes 699,051 questions that demand a high degree of reasoning ability and common-sense understanding. Besides, we present the development of a potent question-answer generation engine that employs table data, a rich color set, and basic question templates to produce a vast array of reasoning question-answer pairs automatically. Based on DCQA, we devise an OCR-free transformer for document-level chart-oriented understanding, capable of DLA and answering complex reasoning and common-sense questions over charts in an OCR-free manner. Our DCQA dataset is expected to foster research on understanding visualizations in documents, especially for scenarios that require complex reasoning for charts in the visually-rich document. We implement and evaluate a set of baselines, and our proposed method achieves comparable results.

{{</citation>}}


### (50/82) Language Agents with Reinforcement Learning for Strategic Play in the Werewolf Game (Zelai Xu et al., 2023)

{{<citation>}}

Zelai Xu, Chao Yu, Fei Fang, Yu Wang, Yi Wu. (2023)  
**Language Agents with Reinforcement Learning for Strategic Play in the Werewolf Game**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs-MA, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.18940v1)  

---


**ABSTRACT**  
Agents built with large language models (LLMs) have recently achieved great advancements. However, most of the efforts focus on single-agent or cooperative settings, leaving more general multi-agent environments underexplored. We propose a new framework powered by reinforcement learning (RL) to develop strategic language agents, i.e., LLM-based agents with strategic thinking ability, for a popular language game, Werewolf. Werewolf is a social deduction game with hidden roles that involves both cooperation and competition and emphasizes deceptive communication and diverse gameplay. Our agent tackles this game by first using LLMs to reason about potential deceptions and generate a set of strategically diverse actions. Then an RL policy, which selects an action from the candidates, is learned by population-based training to enhance the agents' decision-making ability. By combining LLMs with the RL policy, our agent produces a variety of emergent strategies, achieves the highest win rate against other LLM-based agents, and stays robust against adversarial human players in the Werewolf game.

{{</citation>}}


### (51/82) The Utility of 'Even if...' Semifactual Explanation to Optimise Positive Outcomes (Eoin M. Kenny et al., 2023)

{{<citation>}}

Eoin M. Kenny, Weipeng Huang. (2023)  
**The Utility of 'Even if...' Semifactual Explanation to Optimise Positive Outcomes**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-LG, cs.AI  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.18937v1)  

---


**ABSTRACT**  
When users receive either a positive or negative outcome from an automated system, Explainable AI (XAI) has almost exclusively focused on how to mutate negative outcomes into positive ones by crossing a decision boundary using counterfactuals (e.g., \textit{"If you earn 2k more, we will accept your loan application"}). Here, we instead focus on \textit{positive} outcomes, and take the novel step of using XAI to optimise them (e.g., \textit{"Even if you wish to half your down-payment, we will still accept your loan application"}). Explanations such as these that employ "even if..." reasoning, and do not cross a decision boundary, are known as semifactuals. To instantiate semifactuals in this context, we introduce the concept of \textit{Gain} (i.e., how much a user stands to benefit from the explanation), and consider the first causal formalisation of semifactuals. Tests on benchmark datasets show our algorithms are better at maximising gain compared to prior work, and that causality is important in the process. Most importantly however, a user study supports our main hypothesis by showing people find semifactual explanations more useful than counterfactuals when they receive the positive outcome of a loan acceptance.

{{</citation>}}


### (52/82) Self Attention with Temporal Prior: Can We Learn More from Arrow of Time? (Kyung Geun Kim et al., 2023)

{{<citation>}}

Kyung Geun Kim, Byeong Tak Lee. (2023)  
**Self Attention with Temporal Prior: Can We Learn More from Arrow of Time?**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.18932v1)  

---


**ABSTRACT**  
Many of diverse phenomena in nature often inherently encode both short and long term temporal dependencies, short term dependencies especially resulting from the direction of flow of time. In this respect, we discovered experimental evidences suggesting that {\it interrelations} of these events are higher for closer time stamps. However, to be able for attention based models to learn these regularities in short term dependencies, it requires large amounts of data which are often infeasible. This is due to the reason that, while they are good at learning piece wised temporal dependencies, attention based models lack structures that encode biases in time series. As a resolution, we propose a simple and efficient method that enables attention layers to better encode short term temporal bias of these data sets by applying learnable, adaptive kernels directly to the attention matrices. For the experiments, we chose various prediction tasks using Electronic Health Records (EHR) data sets since they are great examples that have underlying long and short term temporal dependencies. The results of our experiments show exceptional classification results compared to best performing models on most of the task and data sets.

{{</citation>}}


## cs.LG (19)



### (53/82) Transfer Learning in Transformer-Based Demand Forecasting For Home Energy Management System (Gargya Gokhale et al., 2023)

{{<citation>}}

Gargya Gokhale, Jonas Van Gompel, Bert Claessens, Chris Develder. (2023)  
**Transfer Learning in Transformer-Based Demand Forecasting For Home Energy Management System**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SY, cs.LG, eess-SY  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2310.19159v1)  

---


**ABSTRACT**  
Increasingly, homeowners opt for photovoltaic (PV) systems and/or battery storage to minimize their energy bills and maximize renewable energy usage. This has spurred the development of advanced control algorithms that maximally achieve those goals. However, a common challenge faced while developing such controllers is the unavailability of accurate forecasts of household power consumption, especially for shorter time resolutions (15 minutes) and in a data-efficient manner. In this paper, we analyze how transfer learning can help by exploiting data from multiple households to improve a single house's load forecasting. Specifically, we train an advanced forecasting model (a temporal fusion transformer) using data from multiple different households, and then finetune this global model on a new household with limited data (i.e. only a few days). The obtained models are used for forecasting power consumption of the household for the next 24 hours~(day-ahead) at a time resolution of 15 minutes, with the intention of using these forecasts in advanced controllers such as Model Predictive Control. We show the benefit of this transfer learning setup versus solely using the individual new household's data, both in terms of (i) forecasting accuracy ($\sim$15\% MAE reduction) and (ii) control performance ($\sim$2\% energy cost reduction), using real-world household data.

{{</citation>}}


### (54/82) BERT Lost Patience Won't Be Robust to Adversarial Slowdown (Zachary Coalson et al., 2023)

{{<citation>}}

Zachary Coalson, Gabriel Ritter, Rakesh Bobba, Sanghyun Hong. (2023)  
**BERT Lost Patience Won't Be Robust to Adversarial Slowdown**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-CR, cs-LG, cs.LG  
Keywords: BERT, ChatGPT, GLUE, GPT  
[Paper Link](http://arxiv.org/abs/2310.19152v1)  

---


**ABSTRACT**  
In this paper, we systematically evaluate the robustness of multi-exit language models against adversarial slowdown. To audit their robustness, we design a slowdown attack that generates natural adversarial text bypassing early-exit points. We use the resulting WAFFLE attack as a vehicle to conduct a comprehensive evaluation of three multi-exit mechanisms with the GLUE benchmark against adversarial slowdown. We then show our attack significantly reduces the computational savings provided by the three methods in both white-box and black-box settings. The more complex a mechanism is, the more vulnerable it is to adversarial slowdown. We also perform a linguistic analysis of the perturbed text inputs, identifying common perturbation patterns that our attack generates, and comparing them with standard adversarial text attacks. Moreover, we show that adversarial training is ineffective in defeating our slowdown attack, but input sanitization with a conversational model, e.g., ChatGPT, can remove perturbations effectively. This result suggests that future work is needed for developing efficient yet robust multi-exit models. Our code is available at: https://github.com/ztcoalson/WAFFLE

{{</citation>}}


### (55/82) MAG-GNN: Reinforcement Learning Boosted Graph Neural Network (Lecheng Kong et al., 2023)

{{<citation>}}

Lecheng Kong, Jiarui Feng, Hao Liu, Dacheng Tao, Yixin Chen, Muhan Zhang. (2023)  
**MAG-GNN: Reinforcement Learning Boosted Graph Neural Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks, Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.19142v1)  

---


**ABSTRACT**  
While Graph Neural Networks (GNNs) recently became powerful tools in graph learning tasks, considerable efforts have been spent on improving GNNs' structural encoding ability. A particular line of work proposed subgraph GNNs that use subgraph information to improve GNNs' expressivity and achieved great success. However, such effectivity sacrifices the efficiency of GNNs by enumerating all possible subgraphs. In this paper, we analyze the necessity of complete subgraph enumeration and show that a model can achieve a comparable level of expressivity by considering a small subset of the subgraphs. We then formulate the identification of the optimal subset as a combinatorial optimization problem and propose Magnetic Graph Neural Network (MAG-GNN), a reinforcement learning (RL) boosted GNN, to solve the problem. Starting with a candidate subgraph set, MAG-GNN employs an RL agent to iteratively update the subgraphs to locate the most expressive set for prediction. This reduces the exponential complexity of subgraph enumeration to the constant complexity of a subgraph search algorithm while keeping good expressivity. We conduct extensive experiments on many datasets, showing that MAG-GNN achieves competitive performance to state-of-the-art methods and even outperforms many subgraph GNNs. We also demonstrate that MAG-GNN effectively reduces the running time of subgraph GNNs.

{{</citation>}}


### (56/82) Automaton Distillation: Neuro-Symbolic Transfer Learning for Deep Reinforcement Learning (Suraj Singireddy et al., 2023)

{{<citation>}}

Suraj Singireddy, Andre Beckus, George Atia, Sumit Jha, Alvaro Velasquez. (2023)  
**Automaton Distillation: Neuro-Symbolic Transfer Learning for Deep Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.19137v1)  

---


**ABSTRACT**  
Reinforcement learning (RL) is a powerful tool for finding optimal policies in sequential decision processes. However, deep RL methods suffer from two weaknesses: collecting the amount of agent experience required for practical RL problems is prohibitively expensive, and the learned policies exhibit poor generalization on tasks outside of the training distribution. To mitigate these issues, we introduce automaton distillation, a form of neuro-symbolic transfer learning in which Q-value estimates from a teacher are distilled into a low-dimensional representation in the form of an automaton. We then propose two methods for generating Q-value estimates: static transfer, which reasons over an abstract Markov Decision Process constructed based on prior knowledge, and dynamic transfer, where symbolic information is extracted from a teacher Deep Q-Network (DQN). The resulting Q-value estimates from either method are used to bootstrap learning in the target environment via a modified DQN loss function. We list several failure modes of existing automaton-based transfer methods and demonstrate that both static and dynamic automaton distillation decrease the time required to find optimal policies for various decision tasks.

{{</citation>}}


### (57/82) Atom: Low-bit Quantization for Efficient and Accurate LLM Serving (Yilong Zhao et al., 2023)

{{<citation>}}

Yilong Zhao, Chien-Yu Lin, Kan Zhu, Zihao Ye, Lequn Chen, Size Zheng, Luis Ceze, Arvind Krishnamurthy, Tianqi Chen, Baris Kasikci. (2023)  
**Atom: Low-bit Quantization for Efficient and Accurate LLM Serving**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Language Model, Quantization  
[Paper Link](http://arxiv.org/abs/2310.19102v1)  

---


**ABSTRACT**  
The growing demand for Large Language Models (LLMs) in applications such as content generation, intelligent chatbots, and sentiment analysis poses considerable challenges for LLM service providers. To efficiently use GPU resources and boost throughput, batching multiple requests has emerged as a popular paradigm; to further speed up batching, LLM quantization techniques reduce memory consumption and increase computing capacity. However, prevalent quantization schemes (e.g., 8-bit weight-activation quantization) cannot fully leverage the capabilities of modern GPUs, such as 4-bit integer operators, resulting in sub-optimal performance.   To maximize LLMs' serving throughput, we introduce Atom, a low-bit quantization method that achieves high throughput improvements with negligible accuracy loss. Atom significantly boosts serving throughput by using low-bit operators and considerably reduces memory consumption via low-bit quantization. It attains high accuracy by applying a novel mixed-precision and fine-grained quantization process. We evaluate Atom on 4-bit weight-activation quantization setups in the serving context. Atom improves end-to-end throughput by up to $7.73\times$ compared to the FP16 and by $2.53\times$ compared to INT8 quantization, while maintaining the same latency target.

{{</citation>}}


### (58/82) Bespoke Solvers for Generative Flow Models (Neta Shaul et al., 2023)

{{<citation>}}

Neta Shaul, Juan Perez, Ricky T. Q. Chen, Ali Thabet, Albert Pumarola, Yaron Lipman. (2023)  
**Bespoke Solvers for Generative Flow Models**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2310.19075v1)  

---


**ABSTRACT**  
Diffusion or flow-based models are powerful generative paradigms that are notoriously hard to sample as samples are defined as solutions to high-dimensional Ordinary or Stochastic Differential Equations (ODEs/SDEs) which require a large Number of Function Evaluations (NFE) to approximate well. Existing methods to alleviate the costly sampling process include model distillation and designing dedicated ODE solvers. However, distillation is costly to train and sometimes can deteriorate quality, while dedicated solvers still require relatively large NFE to produce high quality samples. In this paper we introduce "Bespoke solvers", a novel framework for constructing custom ODE solvers tailored to the ODE of a given pre-trained flow model. Our approach optimizes an order consistent and parameter-efficient solver (e.g., with 80 learnable parameters), is trained for roughly 1% of the GPU time required for training the pre-trained model, and significantly improves approximation and generation quality compared to dedicated solvers. For example, a Bespoke solver for a CIFAR10 model produces samples with Fr\'echet Inception Distance (FID) of 2.73 with 10 NFE, and gets to 1% of the Ground Truth (GT) FID (2.59) for this model with only 20 NFE. On the more challenging ImageNet-64$\times$64, Bespoke samples at 2.2 FID with 10 NFE, and gets within 2% of GT FID (1.71) with 20 NFE.

{{</citation>}}


### (59/82) Object-centric architectures enable efficient causal representation learning (Amin Mansouri et al., 2023)

{{<citation>}}

Amin Mansouri, Jason Hartford, Yan Zhang, Yoshua Bengio. (2023)  
**Object-centric architectures enable efficient causal representation learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2310.19054v1)  

---


**ABSTRACT**  
Causal representation learning has showed a variety of settings in which we can disentangle latent variables with identifiability guarantees (up to some reasonable equivalence class). Common to all of these approaches is the assumption that (1) the latent variables are represented as $d$-dimensional vectors, and (2) that the observations are the output of some injective generative function of these latent variables. While these assumptions appear benign, we show that when the observations are of multiple objects, the generative function is no longer injective and disentanglement fails in practice. We can address this failure by combining recent developments in object-centric learning and causal representation learning. By modifying the Slot Attention architecture arXiv:2006.15055, we develop an object-centric architecture that leverages weak supervision from sparse perturbations to disentangle each object's properties. This approach is more data-efficient in the sense that it requires significantly fewer perturbations than a comparable approach that encodes to a Euclidean space and we show that this approach successfully disentangles the properties of a set of objects in a series of simple image-based disentanglement experiments.

{{</citation>}}


### (60/82) Boosting Decision-Based Black-Box Adversarial Attack with Gradient Priors (Han Liu et al., 2023)

{{<citation>}}

Han Liu, Xingshuo Huang, Xiaotong Zhang, Qimai Li, Fenglong Ma, Wei Wang, Hongyang Chen, Hong Yu, Xianchao Zhang. (2023)  
**Boosting Decision-Based Black-Box Adversarial Attack with Gradient Priors**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: Adversarial Attack  
[Paper Link](http://arxiv.org/abs/2310.19038v1)  

---


**ABSTRACT**  
Decision-based methods have shown to be effective in black-box adversarial attacks, as they can obtain satisfactory performance and only require to access the final model prediction. Gradient estimation is a critical step in black-box adversarial attacks, as it will directly affect the query efficiency. Recent works have attempted to utilize gradient priors to facilitate score-based methods to obtain better results. However, these gradient priors still suffer from the edge gradient discrepancy issue and the successive iteration gradient direction issue, thus are difficult to simply extend to decision-based methods. In this paper, we propose a novel Decision-based Black-box Attack framework with Gradient Priors (DBA-GP), which seamlessly integrates the data-dependent gradient prior and time-dependent prior into the gradient estimation procedure. First, by leveraging the joint bilateral filter to deal with each random perturbation, DBA-GP can guarantee that the generated perturbations in edge locations are hardly smoothed, i.e., alleviating the edge gradient discrepancy, thus remaining the characteristics of the original image as much as possible. Second, by utilizing a new gradient updating strategy to automatically adjust the successive iteration gradient direction, DBA-GP can accelerate the convergence speed, thus improving the query efficiency. Extensive experiments have demonstrated that the proposed method outperforms other strong baselines significantly.

{{</citation>}}


### (61/82) Does Invariant Graph Learning via Environment Augmentation Learn Invariance? (Yongqiang Chen et al., 2023)

{{<citation>}}

Yongqiang Chen, Yatao Bian, Kaiwen Zhou, Binghui Xie, Bo Han, James Cheng. (2023)  
**Does Invariant Graph Learning via Environment Augmentation Learn Invariance?**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Augmentation  
[Paper Link](http://arxiv.org/abs/2310.19035v1)  

---


**ABSTRACT**  
Invariant graph representation learning aims to learn the invariance among data from different environments for out-of-distribution generalization on graphs. As the graph environment partitions are usually expensive to obtain, augmenting the environment information has become the de facto approach. However, the usefulness of the augmented environment information has never been verified. In this work, we find that it is fundamentally impossible to learn invariant graph representations via environment augmentation without additional assumptions. Therefore, we develop a set of minimal assumptions, including variation sufficiency and variation consistency, for feasible invariant graph learning. We then propose a new framework Graph invAriant Learning Assistant (GALA). GALA incorporates an assistant model that needs to be sensitive to graph environment changes or distribution shifts. The correctness of the proxy predictions by the assistant model hence can differentiate the variations in spurious subgraphs. We show that extracting the maximally invariant subgraph to the proxy predictions provably identifies the underlying invariant subgraph for successful OOD generalization under the established minimal assumptions. Extensive experiments on datasets including DrugOOD with various graph distribution shifts confirm the effectiveness of GALA.

{{</citation>}}


### (62/82) TRIAGE: Characterizing and auditing training data for improved regression (Nabeel Seedat et al., 2023)

{{<citation>}}

Nabeel Seedat, Jonathan Crabbé, Zhaozhi Qian, Mihaela van der Schaar. (2023)  
**TRIAGE: Characterizing and auditing training data for improved regression**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.18970v1)  

---


**ABSTRACT**  
Data quality is crucial for robust machine learning algorithms, with the recent interest in data-centric AI emphasizing the importance of training data characterization. However, current data characterization methods are largely focused on classification settings, with regression settings largely understudied. To address this, we introduce TRIAGE, a novel data characterization framework tailored to regression tasks and compatible with a broad class of regressors. TRIAGE utilizes conformal predictive distributions to provide a model-agnostic scoring method, the TRIAGE score. We operationalize the score to analyze individual samples' training dynamics and characterize samples as under-, over-, or well-estimated by the model. We show that TRIAGE's characterization is consistent and highlight its utility to improve performance via data sculpting/filtering, in multiple regression settings. Additionally, beyond sample level, we show TRIAGE enables new approaches to dataset selection and feature acquisition. Overall, TRIAGE highlights the value unlocked by data characterization in real-world regression applications

{{</citation>}}


### (63/82) Building a Safer Maritime Environment Through Multi-Path Long-Term Vessel Trajectory Forecasting (Gabriel Spadon et al., 2023)

{{<citation>}}

Gabriel Spadon, Jay Kumar, Matthew Smith, Sarah Vela, Romina Gehrmann, Derek Eden, Joshua van Berkel, Amilcar Soares, Ronan Fablet, Ronald Pelot, Stan Matwin. (2023)  
**Building a Safer Maritime Environment Through Multi-Path Long-Term Vessel Trajectory Forecasting**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-DM, cs-LG, cs.LG, math-PR  
Keywords: AI, LSTM  
[Paper Link](http://arxiv.org/abs/2310.18948v1)  

---


**ABSTRACT**  
Maritime transport is paramount to global economic growth and environmental sustainability. In this regard, the Automatic Identification System (AIS) data plays a significant role by offering real-time streaming data on vessel movement, which allows for enhanced traffic surveillance, assisting in vessel safety by avoiding vessel-to-vessel collisions and proactively preventing vessel-to-whale ones. This paper tackles an intrinsic problem to trajectory forecasting: the effective multi-path long-term vessel trajectory forecasting on engineered sequences of AIS data. We utilize an encoder-decoder model with Bidirectional Long Short-Term Memory Networks (Bi-LSTM) to predict the next 12 hours of vessel trajectories using 1 to 3 hours of AIS data. We feed the model with probabilistic features engineered from the AIS data that refer to the potential route and destination of each trajectory so that the model, leveraging convolutional layers for spatial feature learning and a position-aware attention mechanism that increases the importance of recent timesteps of a sequence during temporal feature learning, forecasts the vessel trajectory taking the potential route and destination into account. The F1 Score of these features is approximately 85% and 75%, indicating their efficiency in supplementing the neural network. We trialed our model in the Gulf of St. Lawrence, one of the North Atlantic Right Whales (NARW) habitats, achieving an R2 score exceeding 98% with varying techniques and features. Despite the high R2 score being attributed to well-defined shipping lanes, our model demonstrates superior complex decision-making during path selection. In addition, our model shows enhanced accuracy, with average and median forecasting errors of 11km and 6km, respectively. Our study confirms the potential of geographical data engineering and trajectory forecasting models for preserving marine life species.

{{</citation>}}


### (64/82) Implicit Bias of Gradient Descent for Two-layer ReLU and Leaky ReLU Networks on Nearly-orthogonal Data (Yiwen Kou et al., 2023)

{{<citation>}}

Yiwen Kou, Zixiang Chen, Quanquan Gu. (2023)  
**Implicit Bias of Gradient Descent for Two-layer ReLU and Leaky ReLU Networks on Nearly-orthogonal Data**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, math-OC, stat-ML  
Keywords: Bias  
[Paper Link](http://arxiv.org/abs/2310.18935v1)  

---


**ABSTRACT**  
The implicit bias towards solutions with favorable properties is believed to be a key reason why neural networks trained by gradient-based optimization can generalize well. While the implicit bias of gradient flow has been widely studied for homogeneous neural networks (including ReLU and leaky ReLU networks), the implicit bias of gradient descent is currently only understood for smooth neural networks. Therefore, implicit bias in non-smooth neural networks trained by gradient descent remains an open question. In this paper, we aim to answer this question by studying the implicit bias of gradient descent for training two-layer fully connected (leaky) ReLU neural networks. We showed that when the training data are nearly-orthogonal, for leaky ReLU activation function, gradient descent will find a network with a stable rank that converges to $1$, whereas for ReLU activation function, gradient descent will find a neural network with a stable rank that is upper bounded by a constant. Additionally, we show that gradient descent will find a neural network such that all the training data points have the same normalized margin asymptotically. Experiments on both synthetic and real data backup our theoretical findings.

{{</citation>}}


### (65/82) Label Poisoning is All You Need (Rishi D. Jha et al., 2023)

{{<citation>}}

Rishi D. Jha, Jonathan Hayase, Sewoong Oh. (2023)  
**Label Poisoning is All You Need**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-CV, cs-LG, cs.LG  
Keywords: ImageNet, Transformer  
[Paper Link](http://arxiv.org/abs/2310.18933v1)  

---


**ABSTRACT**  
In a backdoor attack, an adversary injects corrupted data into a model's training dataset in order to gain control over its predictions on images with a specific attacker-defined trigger. A typical corrupted training example requires altering both the image, by applying the trigger, and the label. Models trained on clean images, therefore, were considered safe from backdoor attacks. However, in some common machine learning scenarios, the training labels are provided by potentially malicious third-parties. This includes crowd-sourced annotation and knowledge distillation. We, hence, investigate a fundamental question: can we launch a successful backdoor attack by only corrupting labels? We introduce a novel approach to design label-only backdoor attacks, which we call FLIP, and demonstrate its strengths on three datasets (CIFAR-10, CIFAR-100, and Tiny-ImageNet) and four architectures (ResNet-32, ResNet-18, VGG-19, and Vision Transformer). With only 2% of CIFAR-10 labels corrupted, FLIP achieves a near-perfect attack success rate of 99.4% while suffering only a 1.8% drop in the clean test accuracy. Our approach builds upon the recent advances in trajectory matching, originally introduced for dataset distillation.

{{</citation>}}


### (66/82) Remaining Useful Life Prediction of Lithium-ion Batteries using Spatio-temporal Multimodal Attention Networks (Sungho Suh et al., 2023)

{{<citation>}}

Sungho Suh, Dhruv Aditya Mittal, Hymalai Bello, Bo Zhou, Mayank Shekhar Jha, Paul Lukowicz. (2023)  
**Remaining Useful Life Prediction of Lithium-ion Batteries using Spatio-temporal Multimodal Attention Networks**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Attention, LSTM  
[Paper Link](http://arxiv.org/abs/2310.18924v1)  

---


**ABSTRACT**  
Lithium-ion batteries are widely used in various applications, including electric vehicles and renewable energy storage. The prediction of the remaining useful life (RUL) of batteries is crucial for ensuring reliable and efficient operation, as well as reducing maintenance costs. However, determining the life cycle of batteries in real-world scenarios is challenging, and existing methods have limitations in predicting the number of cycles iteratively. In addition, existing works often oversimplify the datasets, neglecting important features of the batteries such as temperature, internal resistance, and material type. To address these limitations, this paper proposes a two-stage remaining useful life prediction scheme for Lithium-ion batteries using a spatio-temporal multimodal attention network (ST-MAN). The proposed model is designed to iteratively predict the number of cycles required for the battery to reach the end of its useful life, based on available data. The proposed ST-MAN is to capture the complex spatio-temporal dependencies in the battery data, including the features that are often neglected in existing works. Experimental results demonstrate that the proposed ST-MAN model outperforms existing CNN and LSTM-based methods, achieving state-of-the-art performance in predicting the remaining useful life of Li-ion batteries. The proposed method has the potential to improve the reliability and efficiency of battery operations and is applicable in various industries, including automotive and renewable energy.

{{</citation>}}


### (67/82) Posterior Sampling with Delayed Feedback for Reinforcement Learning with Linear Function Approximation (Nikki Lijing Kuang et al., 2023)

{{<citation>}}

Nikki Lijing Kuang, Ming Yin, Mengdi Wang, Yu-Xiang Wang, Yi-An Ma. (2023)  
**Posterior Sampling with Delayed Feedback for Reinforcement Learning with Linear Function Approximation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.18919v1)  

---


**ABSTRACT**  
Recent studies in reinforcement learning (RL) have made significant progress by leveraging function approximation to alleviate the sample complexity hurdle for better performance. Despite the success, existing provably efficient algorithms typically rely on the accessibility of immediate feedback upon taking actions. The failure to account for the impact of delay in observations can significantly degrade the performance of real-world systems due to the regret blow-up. In this work, we tackle the challenge of delayed feedback in RL with linear function approximation by employing posterior sampling, which has been shown to empirically outperform the popular UCB algorithms in a wide range of regimes. We first introduce Delayed-PSVI, an optimistic value-based algorithm that effectively explores the value function space via noise perturbation with posterior sampling. We provide the first analysis for posterior sampling algorithms with delayed feedback in RL and show our algorithm achieves $\widetilde{O}(\sqrt{d^3H^3 T} + d^2H^2 E[\tau])$ worst-case regret in the presence of unknown stochastic delays. Here $E[\tau]$ is the expected delay. To further improve its computational efficiency and to expand its applicability in high-dimensional RL problems, we incorporate a gradient-based approximate sampling scheme via Langevin dynamics for Delayed-LPSVI, which maintains the same order-optimal regret guarantee with $\widetilde{O}(dHK)$ computational cost. Empirical evaluations are performed to demonstrate the statistical and computational efficacy of our algorithms.

{{</citation>}}


### (68/82) Hyperbolic Graph Neural Networks at Scale: A Meta Learning Approach (Nurendra Choudhary et al., 2023)

{{<citation>}}

Nurendra Choudhary, Nikhil Rao, Chandan K. Reddy. (2023)  
**Hyperbolic Graph Neural Networks at Scale: A Meta Learning Approach**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.18918v1)  

---


**ABSTRACT**  
The progress in hyperbolic neural networks (HNNs) research is hindered by their absence of inductive bias mechanisms, which are essential for generalizing to new tasks and facilitating scalable learning over large datasets. In this paper, we aim to alleviate these issues by learning generalizable inductive biases from the nodes' local subgraph and transfer them for faster learning over new subgraphs with a disjoint set of nodes, edges, and labels in a few-shot setting. We introduce a novel method, Hyperbolic GRAph Meta Learner (H-GRAM), that, for the tasks of node classification and link prediction, learns transferable information from a set of support local subgraphs in the form of hyperbolic meta gradients and label hyperbolic protonets to enable faster learning over a query set of new tasks dealing with disjoint subgraphs. Furthermore, we show that an extension of our meta-learning framework also mitigates the scalability challenges seen in HNNs faced by existing approaches. Our comparative analysis shows that H-GRAM effectively learns and transfers information in multiple challenging few-shot settings compared to other state-of-the-art baselines. Additionally, we demonstrate that, unlike standard HNNs, our approach is able to scale over large graph datasets and improve performance over its Euclidean counterparts.

{{</citation>}}


### (69/82) Sentence Bag Graph Formulation for Biomedical Distant Supervision Relation Extraction (Hao Zhang et al., 2023)

{{<citation>}}

Hao Zhang, Yang Liu, Xiaoyan Liu, Tianming Liang, Gaurav Sharma, Liang Xue, Maozu Guo. (2023)  
**Sentence Bag Graph Formulation for Biomedical Distant Supervision Relation Extraction**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Relation Extraction  
[Paper Link](http://arxiv.org/abs/2310.18912v1)  

---


**ABSTRACT**  
We introduce a novel graph-based framework for alleviating key challenges in distantly-supervised relation extraction and demonstrate its effectiveness in the challenging and important domain of biomedical data. Specifically, we propose a graph view of sentence bags referring to an entity pair, which enables message-passing based aggregation of information related to the entity pair over the sentence bag. The proposed framework alleviates the common problem of noisy labeling in distantly supervised relation extraction and also effectively incorporates inter-dependencies between sentences within a bag. Extensive experiments on two large-scale biomedical relation datasets and the widely utilized NYT dataset demonstrate that our proposed framework significantly outperforms the state-of-the-art methods for biomedical distant supervision relation extraction while also providing excellent performance for relation extraction in the general text mining domain.

{{</citation>}}


### (70/82) Ever Evolving Evaluator (EV3): Towards Flexible and Reliable Meta-Optimization for Knowledge Distillation (Li Ding et al., 2023)

{{<citation>}}

Li Ding, Masrour Zoghi, Guy Tennenholtz, Maryam Karimzadehgan. (2023)  
**Ever Evolving Evaluator (EV3): Towards Flexible and Reliable Meta-Optimization for Knowledge Distillation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs-NE, cs.LG  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2310.18893v1)  

---


**ABSTRACT**  
We introduce EV3, a novel meta-optimization framework designed to efficiently train scalable machine learning models through an intuitive explore-assess-adapt protocol. In each iteration of EV3, we explore various model parameter updates, assess them using pertinent evaluation methods, and adapt the model based on the optimal updates and previous progress history. EV3 offers substantial flexibility without imposing stringent constraints like differentiability on the key objectives relevant to the tasks of interest. Moreover, this protocol welcomes updates with biased gradients and allows for the use of a diversity of losses and optimizers. Additionally, in scenarios with multiple objectives, it can be used to dynamically prioritize tasks. With inspiration drawn from evolutionary algorithms, meta-learning, and neural architecture search, we investigate an application of EV3 to knowledge distillation. Our experimental results illustrate EV3's capability to safely explore model spaces, while hinting at its potential applicability across numerous domains due to its inherent flexibility and adaptability.

{{</citation>}}


### (71/82) Simple and Asymmetric Graph Contrastive Learning without Augmentations (Teng Xiao et al., 2023)

{{<citation>}}

Teng Xiao, Huaisheng Zhu, Zhengyu Chen, Suhang Wang. (2023)  
**Simple and Asymmetric Graph Contrastive Learning without Augmentations**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Augmentation, Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2310.18884v1)  

---


**ABSTRACT**  
Graph Contrastive Learning (GCL) has shown superior performance in representation learning in graph-structured data. Despite their success, most existing GCL methods rely on prefabricated graph augmentation and homophily assumptions. Thus, they fail to generalize well to heterophilic graphs where connected nodes may have different class labels and dissimilar features. In this paper, we study the problem of conducting contrastive learning on homophilic and heterophilic graphs. We find that we can achieve promising performance simply by considering an asymmetric view of the neighboring nodes. The resulting simple algorithm, Asymmetric Contrastive Learning for Graphs (GraphACL), is easy to implement and does not rely on graph augmentations and homophily assumptions. We provide theoretical and empirical evidence that GraphACL can capture one-hop local neighborhood information and two-hop monophily similarity, which are both important for modeling heterophilic graphs. Experimental results show that the simple GraphACL significantly outperforms state-of-the-art graph contrastive learning and self-supervised learning methods on homophilic and heterophilic graphs. The code of GraphACL is available at https://github.com/tengxiao1/GraphACL.

{{</citation>}}


## cs.HC (1)



### (72/82) Perspectives from India: Challenges and Opportunities for Computational Tools to Enhance Confidence in Published Research (Tatiana Chakravorti et al., 2023)

{{<citation>}}

Tatiana Chakravorti, Chuhao Wu, Sai Koneru, Sarah Rajtmajer. (2023)  
**Perspectives from India: Challenges and Opportunities for Computational Tools to Enhance Confidence in Published Research**  

---
Primary Category: cs.HC  
Categories: cs-HC, cs.HC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.19158v1)  

---


**ABSTRACT**  
Over the past decade, a crisis of confidence in published scientific findings has catalyzed widespread response from the research community, particularly in the West. These responses have included policy discussions and changes to existing practice as well as computational infrastructure to support and evaluate research. Our work studies Indian researchers' awareness, perceptions, and challenges around research integrity. We explore opportunities for Artificial Intelligence (AI)-powered tools to evaluate reproducibility and replicability, centering cultural perspectives. We discuss requirements for such tools, including signals within papers and metadata to be included, and system hybridity (fully-AI vs. collaborative human-AI). We draw upon 19 semi-structured interviews and 72 follow-up surveys with researchers at universities throughout India. Our findings highlight the need for computational tools to contextualize confidence in published research. In particular, researchers prefer approaches that enable human-AI collaboration. Additionally, our findings emphasize the shortcomings of current incentive structures for publication, funding, and promotion.

{{</citation>}}


## eess.SY (1)



### (73/82) Real-World Implementation of Reinforcement Learning Based Energy Coordination for a Cluster of Households (Gargya Gokhale et al., 2023)

{{<citation>}}

Gargya Gokhale, Niels Tiben, Marie-Sophie Verwee, Manu Lahariya, Bert Claessens, Chris Develder. (2023)  
**Real-World Implementation of Reinforcement Learning Based Energy Coordination for a Cluster of Households**  

---
Primary Category: eess.SY  
Categories: cs-LG, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2310.19155v1)  

---


**ABSTRACT**  
Given its substantial contribution of 40\% to global power consumption, the built environment has received increasing attention to serve as a source of flexibility to assist the modern power grid. In that respect, previous research mainly focused on energy management of individual buildings. In contrast, in this paper, we focus on aggregated control of a set of residential buildings, to provide grid supporting services, that eventually should include ancillary services. In particular, we present a real-life pilot study that studies the effectiveness of reinforcement-learning (RL) in coordinating the power consumption of 8 residential buildings to jointly track a target power signal. Our RL approach relies solely on observed data from individual households and does not require any explicit building models or simulators, making it practical to implement and easy to scale. We show the feasibility of our proposed RL-based coordination strategy in a real-world setting. In a 4-week case study, we demonstrate a hierarchical control system, relying on an RL-based ranking system to select which households to activate flex assets from, and a real-time PI control-based power dispatch mechanism to control the selected assets. Our results demonstrate satisfactory power tracking, and the effectiveness of the RL-based ranks which are learnt in a purely data-driven manner.

{{</citation>}}


## cs.SE (2)



### (74/82) Partial Orderings as Heuristic for Multi-Objective Model-Based Reasoning (Andre Lustosa et al., 2023)

{{<citation>}}

Andre Lustosa, Tim Menzies. (2023)  
**Partial Orderings as Heuristic for Multi-Objective Model-Based Reasoning**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2310.19125v1)  

---


**ABSTRACT**  
Model-based reasoning is becoming increasingly common in software engineering. The process of building and analyzing models helps stakeholders to understand the ramifications of their software decisions. But complex models can confuse and overwhelm stakeholders when these models have too many candidate solutions. We argue here that a technique based on partial orderings lets humans find acceptable solutions via a binary chop needing $O(log(N))$ queries (or less). This paper checks the value of this approach via the iSNEAK partial ordering tool. Pre-experimentally, we were concerned that (a)~our automated methods might produce models that were unacceptable to humans; and that (b)~our human-in-the-loop methods might actual overlooking significant optimizations. Hence, we checked the acceptability of the solutions found by iSNEAK via a human-in-the-loop double-blind evaluation study of 20 Brazilian programmers. We also checked if iSNEAK misses significant optimizations (in a corpus of 16 SE models of size ranging up to 1000 attributes by comparing it against two rival technologies (the genetic algorithms preferred by the interactive search-based SE community; and the sequential model optimizers developed by the SE configuration community~\citep{flash_vivek}). iSNEAK 's solutions were found to be human acceptable (and those solutions took far less time to generate, with far fewer questions to any stakeholder). Significantly, our methods work well even for multi-objective models with competing goals (in this work we explore models with four to five goals). These results motivate more work on partial ordering for many-goal model-based problems.

{{</citation>}}


### (75/82) Software engineering for deep learning applications: usage of SWEng and MLops tools in GitHub repositories (Evangelia Panourgia et al., 2023)

{{<citation>}}

Evangelia Panourgia, Theodoros Plessas, Diomidis Spinellis. (2023)  
**Software engineering for deep learning applications: usage of SWEng and MLops tools in GitHub repositories**  

---
Primary Category: cs.SE  
Categories: cs-LG, cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2310.19124v1)  

---


**ABSTRACT**  
The rising popularity of deep learning (DL) methods and techniques has invigorated interest in the topic of SE4DL, the application of software engineering (SE) practices on deep learning software. Despite the novel engineering challenges brought on by the data-driven and non-deterministic paradigm of DL software, little work has been invested into developing AI-targeted SE tools. On the other hand, tools tackling more general engineering issues in DL are actively used and referred to under the umbrella term of ``MLOps tools''. Furthermore, the available literature supports the utility of conventional SE tooling in DL software development. Building upon previous MSR research on tool usage in open-source software works, we identify conventional and MLOps tools adopted in popular applied DL projects that use Python as the main programming language. About 70% of the GitHub repositories mined contained at least one conventional SE tool. Software configuration management tools are the most adopted, while the opposite applies to maintenance tools. Substantially fewer MLOps tools were in use, with only 9 tools out of a sample of 80 used in at least one repository. The majority of them were open-source rather than proprietary. One of these tools, TensorBoard, was found to be adopted in about half of the repositories in our study. Consequently, the use of conventional SE tooling demonstrates its relevance to DL software. Further research is recommended on the adoption of MLOps tooling by open-source projects, focusing on the relevance of particular tool types, the development of required tools, as well as ways to promote the use of already available tools.

{{</citation>}}


## cs.DS (1)



### (76/82) Sketching Algorithms for Sparse Dictionary Learning: PTAS and Turnstile Streaming (Gregory Dexter et al., 2023)

{{<citation>}}

Gregory Dexter, Petros Drineas, David P. Woodruff, Taisuke Yasuda. (2023)  
**Sketching Algorithms for Sparse Dictionary Learning: PTAS and Turnstile Streaming**  

---
Primary Category: cs.DS  
Categories: cs-DS, cs-LG, cs.DS  
Keywords: Sketch  
[Paper Link](http://arxiv.org/abs/2310.19068v1)  

---


**ABSTRACT**  
Sketching algorithms have recently proven to be a powerful approach both for designing low-space streaming algorithms as well as fast polynomial time approximation schemes (PTAS). In this work, we develop new techniques to extend the applicability of sketching-based approaches to the sparse dictionary learning and the Euclidean $k$-means clustering problems. In particular, we initiate the study of the challenging setting where the dictionary/clustering assignment for each of the $n$ input points must be output, which has surprisingly received little attention in prior work. On the fast algorithms front, we obtain a new approach for designing PTAS's for the $k$-means clustering problem, which generalizes to the first PTAS for the sparse dictionary learning problem. On the streaming algorithms front, we obtain new upper bounds and lower bounds for dictionary learning and $k$-means clustering. In particular, given a design matrix $\mathbf A\in\mathbb R^{n\times d}$ in a turnstile stream, we show an $\tilde O(nr/\epsilon^2 + dk/\epsilon)$ space upper bound for $r$-sparse dictionary learning of size $k$, an $\tilde O(n/\epsilon^2 + dk/\epsilon)$ space upper bound for $k$-means clustering, as well as an $\tilde O(n)$ space upper bound for $k$-means clustering on random order row insertion streams with a natural "bounded sensitivity" assumption. On the lower bounds side, we obtain a general $\tilde\Omega(n/\epsilon + dk/\epsilon)$ lower bound for $k$-means clustering, as well as an $\tilde\Omega(n/\epsilon^2)$ lower bound for algorithms which can estimate the cost of a single fixed set of candidate centers.

{{</citation>}}


## cs.IR (2)



### (77/82) MILL: Mutual Verification with Large Language Models for Zero-Shot Query Expansion (Pengyue Jia et al., 2023)

{{<citation>}}

Pengyue Jia, Yiding Liu, Xiangyu Zhao, Xiaopeng Li, Changying Hao, Shuaiqiang Wang, Dawei Yin. (2023)  
**MILL: Mutual Verification with Large Language Models for Zero-Shot Query Expansion**  

---
Primary Category: cs.IR  
Categories: cs-AI, cs-CL, cs-IR, cs.IR  
Keywords: Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2310.19056v1)  

---


**ABSTRACT**  
Query expansion is a commonly-used technique in many search systems to better represent users' information needs with additional query terms. Existing studies for this task usually propose to expand a query with retrieved or generated contextual documents. However, both types of methods have clear limitations. For retrieval-based methods, the documents retrieved with the original query might not be accurate enough to reveal the search intent, especially when the query is brief or ambiguous. For generation-based methods, existing models can hardly be trained or aligned on a particular corpus, due to the lack of corpus-specific labeled data. In this paper, we propose a novel Large Language Model (LLM) based mutual verification framework for query expansion, which alleviates the aforementioned limitations. Specifically, we first design a query-query-document generation pipeline, which can effectively leverage the contextual knowledge encoded in LLMs to generate sub-queries and corresponding documents from multiple perspectives. Next, we employ a mutual verification method for both generated and retrieved contextual documents, where 1) retrieved documents are filtered with the external contextual knowledge in generated documents, and 2) generated documents are filtered with the corpus-specific knowledge in retrieved documents. Overall, the proposed method allows retrieved and generated documents to complement each other to finalize a better query expansion. We conduct extensive experiments on three information retrieval datasets, i.e., TREC-DL-2020, TREC-COVID, and MSMARCO. The results demonstrate that our method outperforms other baselines significantly.

{{</citation>}}


### (78/82) A Multimodal Ecological Civilization Pattern Recommendation Method Based on Large Language Models and Knowledge Graph (Zhihang Yu et al., 2023)

{{<citation>}}

Zhihang Yu, Shu Wang, Yunqiang Zhu, Zhiqiang Zou. (2023)  
**A Multimodal Ecological Civilization Pattern Recommendation Method Based on Large Language Models and Knowledge Graph**  

---
Primary Category: cs.IR  
Categories: cs-IR, cs.IR  
Keywords: Knowledge Graph, Language Model, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2310.18951v1)  

---


**ABSTRACT**  
The Ecological Civilization Pattern Recommendation System (ECPRS) aims to recommend suitable ecological civilization patterns for target regions, promoting sustainable development and reducing regional disparities. However, the current representative recommendation methods are not suitable for recommending ecological civilization patterns in a geographical context. There are two reasons for this. Firstly, regions have spatial heterogeneity, and the (ECPRS)needs to consider factors like climate, topography, vegetation, etc., to recommend civilization patterns adapted to specific ecological environments, ensuring the feasibility and practicality of the recommendations. Secondly, the abstract features of the ecological civilization patterns in the real world have not been fully utilized., resulting in poor richness in their embedding representations and consequently, lower performance of the recommendation system. Considering these limitations, we propose the ECPR-MML method. Initially, based on the novel method UGPIG, we construct a knowledge graph to extract regional representations incorporating spatial heterogeneity features. Following that, inspired by the significant progress made by Large Language Models (LLMs) in the field of Natural Language Processing (NLP), we employ Large LLMs to generate multimodal features for ecological civilization patterns in the form of text and images. We extract and integrate these multimodal features to obtain semantically rich representations of ecological civilization. Through extensive experiments, we validate the performance of our ECPR-MML model. Our results show that F1@5 is 2.11% higher compared to state-of-the-art models, 2.02% higher than NGCF, and 1.16% higher than UGPIG. Furthermore, multimodal data can indeed enhance recommendation performance. However, the data generated by LLM is not as effective as real data to a certain extent.

{{</citation>}}


## cs.NE (1)



### (79/82) Large Language Models as Evolutionary Optimizers (Shengcai Liu et al., 2023)

{{<citation>}}

Shengcai Liu, Caishun Chen, Xinghua Qu, Ke Tang, Yew-Soon Ong. (2023)  
**Large Language Models as Evolutionary Optimizers**  

---
Primary Category: cs.NE  
Categories: cs-NE, cs.NE  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2310.19046v1)  

---


**ABSTRACT**  
Evolutionary algorithms (EAs) have achieved remarkable success in tackling complex combinatorial optimization problems. However, EAs often demand carefully-designed operators with the aid of domain expertise to achieve satisfactory performance. In this work, we present the first study on large language models (LLMs) as evolutionary combinatorial optimizers. The main advantage is that it requires minimal domain knowledge and human efforts, as well as no additional training of the model. This approach is referred to as LLM-driven EA (LMEA). Specifically, in each generation of the evolutionary search, LMEA instructs the LLM to select parent solutions from current population, and perform crossover and mutation to generate offspring solutions. Then, LMEA evaluates these new solutions and include them into the population for the next generation. LMEA is equipped with a self-adaptation mechanism that controls the temperature of the LLM. This enables it to balance between exploration and exploitation and prevents the search from getting stuck in local optima. We investigate the power of LMEA on the classical traveling salesman problems (TSPs) widely used in combinatorial optimization research. Notably, the results show that LMEA performs competitively to traditional heuristics in finding high-quality solutions on TSP instances with up to 20 nodes. Additionally, we also study the effectiveness of LLM-driven crossover/mutation and the self-adaptation mechanism in evolutionary search. In summary, our results reveal the great potentials of LLMs as evolutionary optimizers for solving combinatorial problems. We hope our research shall inspire future explorations on LLM-driven EAs for complex optimization challenges.

{{</citation>}}


## stat.ML (1)



### (80/82) On Linear Separation Capacity of Self-Supervised Representation Learning (Shulei Wang, 2023)

{{<citation>}}

Shulei Wang. (2023)  
**On Linear Separation Capacity of Self-Supervised Representation Learning**  

---
Primary Category: stat.ML  
Categories: cs-LG, math-ST, stat-ML, stat-TH, stat.ML  
Keywords: Representation Learning, Self-Supervised  
[Paper Link](http://arxiv.org/abs/2310.19041v1)  

---


**ABSTRACT**  
Recent advances in self-supervised learning have highlighted the efficacy of data augmentation in learning data representation from unlabeled data. Training a linear model atop these enhanced representations can yield an adept classifier. Despite the remarkable empirical performance, the underlying mechanisms that enable data augmentation to unravel nonlinear data structures into linearly separable representations remain elusive. This paper seeks to bridge this gap by investigating under what conditions learned representations can linearly separate manifolds when data is drawn from a multi-manifold model. Our investigation reveals that data augmentation offers additional information beyond observed data and can thus improve the information-theoretic optimal rate of linear separation capacity. In particular, we show that self-supervised learning can linearly separate manifolds with a smaller distance than unsupervised learning, underscoring the additional benefits of data augmentation. Our theoretical analysis further underscores that the performance of downstream linear classifiers primarily hinges on the linear separability of data representations rather than the size of the labeled data set, reaffirming the viability of constructing efficient classifiers with limited labeled data amid an expansive unlabeled data set.

{{</citation>}}


## cs.DM (1)



### (81/82) The Weisfeiler-Leman Dimension of Existential Conjunctive Queries (Andreas Göbel et al., 2023)

{{<citation>}}

Andreas Göbel, Leslie Ann Goldberg, Marc Roth. (2023)  
**The Weisfeiler-Leman Dimension of Existential Conjunctive Queries**  

---
Primary Category: cs.DM  
Categories: cs-DB, cs-DM, cs-LO, cs.DM  
Keywords: AI, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2310.19006v1)  

---


**ABSTRACT**  
The Weisfeiler-Leman (WL) dimension of a graph parameter $f$ is the minimum $k$ such that, if $G_1$ and $G_2$ are indistinguishable by the $k$-dimensional WL-algorithm then $f(G_1)=f(G_2)$. The WL-dimension of $f$ is $\infty$ if no such $k$ exists. We study the WL-dimension of graph parameters characterised by the number of answers from a fixed conjunctive query to the graph. Given a conjunctive query $\varphi$, we quantify the WL-dimension of the function that maps every graph $G$ to the number of answers of $\varphi$ in $G$.   The works of Dvor\'ak (J. Graph Theory 2010), Dell, Grohe, and Rattan (ICALP 2018), and Neuen (ArXiv 2023) have answered this question for full conjunctive queries, which are conjunctive queries without existentially quantified variables. For such queries $\varphi$, the WL-dimension is equal to the treewidth of the Gaifman graph of $\varphi$.   In this work, we give a characterisation that applies to all conjunctive qureies. Given any conjunctive query $\varphi$, we prove that its WL-dimension is equal to the semantic extension width $\mathsf{sew}(\varphi)$, a novel width measure that can be thought of as a combination of the treewidth of $\varphi$ and its quantified star size, an invariant introduced by Durand and Mengel (ICDT 2013) describing how the existentially quantified variables of $\varphi$ are connected with the free variables. Using the recently established equivalence between the WL-algorithm and higher-order Graph Neural Networks (GNNs) due to Morris et al. (AAAI 2019), we obtain as a consequence that the function counting answers to a conjunctive query $\varphi$ cannot be computed by GNNs of order smaller than $\mathsf{sew}(\varphi)$.

{{</citation>}}


## cs.SI (1)



### (82/82) Uncovering Gender Bias within Journalist-Politician Interaction in Indian Twitter (Brisha Jain et al., 2023)

{{<citation>}}

Brisha Jain, Mainack Mondal. (2023)  
**Uncovering Gender Bias within Journalist-Politician Interaction in Indian Twitter**  

---
Primary Category: cs.SI  
Categories: cs-CY, cs-HC, cs-SI, cs.SI  
Keywords: Bias, Twitter  
[Paper Link](http://arxiv.org/abs/2310.18911v1)  

---


**ABSTRACT**  
Gender bias in political discourse is a significant problem on today's social media. Previous studies found that the gender of politicians indeed influences the content directed towards them by the general public. However, these works are particularly focused on the global north, which represents individualistic culture. Furthermore, they did not address whether there is gender bias even within the interaction between popular journalists and politicians in the global south. These understudied journalist-politician interactions are important (more so in collectivistic cultures like the global south) as they can significantly affect public sentiment and help set gender-biased social norms. In this work, using large-scale data from Indian Twitter we address this research gap.   We curated a gender-balanced set of 100 most-followed Indian journalists on Twitter and 100 most-followed politicians. Then we collected 21,188 unique tweets posted by these journalists that mentioned these politicians. Our analysis revealed that there is a significant gender bias -- the frequency with which journalists mention male politicians vs. how frequently they mention female politicians is statistically significantly different ($p<<0.05$). In fact, median tweets from female journalists mentioning female politicians received ten times fewer likes than median tweets from female journalists mentioning male politicians. However, when we analyzed tweet content, our emotion score analysis and topic modeling analysis did not reveal any significant gender-based difference within the journalists' tweets towards politicians. Finally, we found a potential reason for the significant gender bias: the number of popular male Indian politicians is almost twice as large as the number of popular female Indian politicians, which might have resulted in the observed bias. We conclude by discussing the implications of this work.

{{</citation>}}
