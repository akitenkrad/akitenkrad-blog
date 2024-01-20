---
draft: false
title: "arXiv @ 2024.01.20"
date: 2024-01-20
author: "akitenkrad"
description: ""
tags: ["arXiv", "Published:2024"]
menu:
  sidebar:
    name: "arXiv @ 2024.01.20"
    identifier: arxiv_20240120
    parent: 202401_arxiv
    weight: 10
math: true
---

<figure style="border:none; width:100%; display:flex; justify-content: center">
    <iframe src="pie.html" width=900 height=620 style="border:none"></iframe>
</figure>


## Primary Categories

- [cs.CV (27)](#cscv-27)
- [cs.CL (22)](#cscl-22)
- [cs.NI (1)](#csni-1)
- [cs.CR (4)](#cscr-4)
- [cs.LG (20)](#cslg-20)
- [stat.ME (1)](#statme-1)
- [cs.AI (3)](#csai-3)
- [cs.NE (3)](#csne-3)
- [cs.RO (3)](#csro-3)
- [cs.SE (2)](#csse-2)
- [cs.DC (3)](#csdc-3)
- [cs.AR (1)](#csar-1)
- [cs.SD (2)](#cssd-2)
- [cs.ET (1)](#cset-1)
- [q-bio.BM (1)](#q-biobm-1)
- [cs.GT (1)](#csgt-1)
- [eess.AS (2)](#eessas-2)
- [eess.SY (2)](#eesssy-2)
- [cs.MM (1)](#csmm-1)
- [cs.SI (1)](#cssi-1)
- [cs.SC (1)](#cssc-1)
- [math.NA (1)](#mathna-1)
- [cs.HC (1)](#cshc-1)
- [astro-ph.IM (1)](#astro-phim-1)

## cs.CV (27)



### (1/105) Towards Language-Driven Video Inpainting via Multimodal Large Language Models (Jianzong Wu et al., 2024)

{{<citation>}}

Jianzong Wu, Xiangtai Li, Chenyang Si, Shangchen Zhou, Jingkang Yang, Jiangning Zhang, Yining Li, Kai Chen, Yunhai Tong, Ziwei Liu, Chen Change Loy. (2024)  
**Towards Language-Driven Video Inpainting via Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10226v1)  

---


**ABSTRACT**  
We introduce a new task -- language-driven video inpainting, which uses natural language instructions to guide the inpainting process. This approach overcomes the limitations of traditional video inpainting methods that depend on manually labeled binary masks, a process often tedious and labor-intensive. We present the Remove Objects from Videos by Instructions (ROVI) dataset, containing 5,650 videos and 9,091 inpainting results, to support training and evaluation for this task. We also propose a novel diffusion-based language-driven video inpainting framework, the first end-to-end baseline for this task, integrating Multimodal Large Language Models to understand and execute complex language-based inpainting requests effectively. Our comprehensive results showcase the dataset's versatility and the model's effectiveness in various language-instructed inpainting scenarios. We will make datasets, code, and models publicly available.

{{</citation>}}


### (2/105) GPAvatar: Generalizable and Precise Head Avatar from Image(s) (Xuangeng Chu et al., 2024)

{{<citation>}}

Xuangeng Chu, Yu Li, Ailing Zeng, Tianyu Yang, Lijian Lin, Yunfei Liu, Tatsuya Harada. (2024)  
**GPAvatar: Generalizable and Precise Head Avatar from Image(s)**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.10215v1)  

---


**ABSTRACT**  
Head avatar reconstruction, crucial for applications in virtual reality, online meetings, gaming, and film industries, has garnered substantial attention within the computer vision community. The fundamental objective of this field is to faithfully recreate the head avatar and precisely control expressions and postures. Existing methods, categorized into 2D-based warping, mesh-based, and neural rendering approaches, present challenges in maintaining multi-view consistency, incorporating non-facial information, and generalizing to new identities. In this paper, we propose a framework named GPAvatar that reconstructs 3D head avatars from one or several images in a single forward pass. The key idea of this work is to introduce a dynamic point-based expression field driven by a point cloud to precisely and effectively capture expressions. Furthermore, we use a Multi Tri-planes Attention (MTA) fusion module in the tri-planes canonical field to leverage information from multiple input images. The proposed method achieves faithful identity reconstruction, precise expression control, and multi-view consistency, demonstrating promising results for free-viewpoint rendering and novel view synthesis.

{{</citation>}}


### (3/105) Neural Echos: Depthwise Convolutional Filters Replicate Biological Receptive Fields (Zahra Babaiee et al., 2024)

{{<citation>}}

Zahra Babaiee, Peyman M. Kiasari, Daniela Rus, Radu Grosu. (2024)  
**Neural Echos: Depthwise Convolutional Filters Replicate Biological Receptive Fields**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-NE, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.10178v1)  

---


**ABSTRACT**  
In this study, we present evidence suggesting that depthwise convolutional kernels are effectively replicating the structural intricacies of the biological receptive fields observed in the mammalian retina. We provide analytics of trained kernels from various state-of-the-art models substantiating this evidence. Inspired by this intriguing discovery, we propose an initialization scheme that draws inspiration from the biological receptive fields. Experimental analysis of the ImageNet dataset with multiple CNN architectures featuring depthwise convolutions reveals a marked enhancement in the accuracy of the learned model when initialized with biologically derived weights. This underlies the potential for biologically inspired computational models to further our understanding of vision processing systems and to improve the efficacy of convolutional networks.

{{</citation>}}


### (4/105) VMamba: Visual State Space Model (Yue Liu et al., 2024)

{{<citation>}}

Yue Liu, Yunjie Tian, Yuzhong Zhao, Hongtian Yu, Lingxi Xie, Yaowei Wang, Qixiang Ye, Yunfan Liu. (2024)  
**VMamba: Visual State Space Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.10166v1)  

---


**ABSTRACT**  
Convolutional Neural Networks (CNNs) and Vision Transformers (ViTs) stand as the two most popular foundation models for visual representation learning. While CNNs exhibit remarkable scalability with linear complexity w.r.t. image resolution, ViTs surpass them in fitting capabilities despite contending with quadratic complexity. A closer inspection reveals that ViTs achieve superior visual modeling performance through the incorporation of global receptive fields and dynamic weights. This observation motivates us to propose a novel architecture that inherits these components while enhancing computational efficiency. To this end, we draw inspiration from the recently introduced state space model and propose the Visual State Space Model (VMamba), which achieves linear complexity without sacrificing global receptive fields. To address the encountered direction-sensitive issue, we introduce the Cross-Scan Module (CSM) to traverse the spatial domain and convert any non-causal visual image into order patch sequences. Extensive experimental results substantiate that VMamba not only demonstrates promising capabilities across various visual perception tasks, but also exhibits more pronounced advantages over established benchmarks as the image resolution increases. Source code has been available at https://github.com/MzeroMiko/VMamba.

{{</citation>}}


### (5/105) Motion-Zero: Zero-Shot Moving Object Control Framework for Diffusion-Based Video Generation (Changgu Chen et al., 2024)

{{<citation>}}

Changgu Chen, Junwei Shu, Lianggangxu Chen, Gaoqi He, Changbo Wang, Yang Li. (2024)  
**Motion-Zero: Zero-Shot Moving Object Control Framework for Diffusion-Based Video Generation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.10150v1)  

---


**ABSTRACT**  
Recent large-scale pre-trained diffusion models have demonstrated a powerful generative ability to produce high-quality videos from detailed text descriptions. However, exerting control over the motion of objects in videos generated by any video diffusion model is a challenging problem. In this paper, we propose a novel zero-shot moving object trajectory control framework, Motion-Zero, to enable a bounding-box-trajectories-controlled text-to-video diffusion model.To this end, an initial noise prior module is designed to provide a position-based prior to improve the stability of the appearance of the moving object and the accuracy of position. In addition, based on the attention map of the U-net, spatial constraints are directly applied to the denoising process of diffusion models, which further ensures the positional and spatial consistency of moving objects during the inference. Furthermore, temporal consistency is guaranteed with a proposed shift temporal attention mechanism. Our method can be flexibly applied to various state-of-the-art video diffusion models without any training process. Extensive experiments demonstrate our proposed method can control the motion trajectories of objects and generate high-quality videos.

{{</citation>}}


### (6/105) Explicitly Disentangled Representations in Object-Centric Learning (Riccardo Majellaro et al., 2024)

{{<citation>}}

Riccardo Majellaro, Jonathan Collu, Aske Plaat, Thomas M. Moerland. (2024)  
**Explicitly Disentangled Representations in Object-Centric Learning**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs-LG, cs.CV  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.10148v1)  

---


**ABSTRACT**  
Extracting structured representations from raw visual data is an important and long-standing challenge in machine learning. Recently, techniques for unsupervised learning of object-centric representations have raised growing interest. In this context, enhancing the robustness of the latent features can improve the efficiency and effectiveness of the training of downstream tasks. A promising step in this direction is to disentangle the factors that cause variation in the data. Previously, Invariant Slot Attention disentangled position, scale, and orientation from the remaining features. Extending this approach, we focus on separating the shape and texture components. In particular, we propose a novel architecture that biases object-centric models toward disentangling shape and texture components into two non-overlapping subsets of the latent space dimensions. These subsets are known a priori, hence before the training process. Experiments on a range of object-centric benchmarks reveal that our approach achieves the desired disentanglement while also numerically improving baseline performance in most cases. In addition, we show that our method can generate novel textures for a specific object or transfer textures between objects with distinct shapes.

{{</citation>}}


### (7/105) Exposing Lip-syncing Deepfakes from Mouth Inconsistencies (Soumyya Kanti Datta et al., 2024)

{{<citation>}}

Soumyya Kanti Datta, Shan Jia, Siwei Lyu. (2024)  
**Exposing Lip-syncing Deepfakes from Mouth Inconsistencies**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.10113v1)  

---


**ABSTRACT**  
A lip-syncing deepfake is a digitally manipulated video in which a person's lip movements are created convincingly using AI models to match altered or entirely new audio. Lip-syncing deepfakes are a dangerous type of deepfakes as the artifacts are limited to the lip region and more difficult to discern. In this paper, we describe a novel approach, LIP-syncing detection based on mouth INConsistency (LIPINC), for lip-syncing deepfake detection by identifying temporal inconsistencies in the mouth region. These inconsistencies are seen in the adjacent frames and throughout the video. Our model can successfully capture these irregularities and outperforms the state-of-the-art methods on several benchmark deepfake datasets.

{{</citation>}}


### (8/105) DiffusionGPT: LLM-Driven Text-to-Image Generation System (Jie Qin et al., 2024)

{{<citation>}}

Jie Qin, Jie Wu, Weifeng Chen, Yuxi Ren, Huixia Li, Hefeng Wu, Xuefeng Xiao, Rui Wang, Shilei Wen. (2024)  
**DiffusionGPT: LLM-Driven Text-to-Image Generation System**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.10061v1)  

---


**ABSTRACT**  
Diffusion models have opened up new avenues for the field of image generation, resulting in the proliferation of high-quality models shared on open-source platforms. However, a major challenge persists in current text-to-image systems are often unable to handle diverse inputs, or are limited to single model results. Current unified attempts often fall into two orthogonal aspects: i) parse Diverse Prompts in input stage; ii) activate expert model to output. To combine the best of both worlds, we propose DiffusionGPT, which leverages Large Language Models (LLM) to offer a unified generation system capable of seamlessly accommodating various types of prompts and integrating domain-expert models. DiffusionGPT constructs domain-specific Trees for various generative models based on prior knowledge. When provided with an input, the LLM parses the prompt and employs the Trees-of-Thought to guide the selection of an appropriate model, thereby relaxing input constraints and ensuring exceptional performance across diverse domains. Moreover, we introduce Advantage Databases, where the Tree-of-Thought is enriched with human feedback, aligning the model selection process with human preferences. Through extensive experiments and comparisons, we demonstrate the effectiveness of DiffusionGPT, showcasing its potential for pushing the boundaries of image synthesis in diverse domains.

{{</citation>}}


### (9/105) GPT4Ego: Unleashing the Potential of Pre-trained Models for Zero-Shot Egocentric Action Recognition (Guangzhao Dai et al., 2024)

{{<citation>}}

Guangzhao Dai, Xiangbo Shu, Wenhao Wu. (2024)  
**GPT4Ego: Unleashing the Potential of Pre-trained Models for Zero-Shot Egocentric Action Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, Language Model, Zero-Shot  
[Paper Link](http://arxiv.org/abs/2401.10039v1)  

---


**ABSTRACT**  
Vision-Language Models (VLMs), pre-trained on large-scale datasets, have shown impressive performance in various visual recognition tasks. This advancement paves the way for notable performance in Zero-Shot Egocentric Action Recognition (ZS-EAR). Typically, VLMs handle ZS-EAR as a global video-text matching task, which often leads to suboptimal alignment of vision and linguistic knowledge. We propose a refined approach for ZS-EAR using VLMs, emphasizing fine-grained concept-description alignment that capitalizes on the rich semantic and contextual details in egocentric videos. In this paper, we introduce GPT4Ego, a straightforward yet remarkably potent VLM framework for ZS-EAR, designed to enhance the fine-grained alignment of concept and description between vision and language. Extensive experiments demonstrate GPT4Ego significantly outperforms existing VLMs on three large-scale egocentric video benchmarks, i.e., EPIC-KITCHENS-100 (33.2%, +9.4%), EGTEA (39.6%, +5.5%), and CharadesEgo (31.5%, +2.6%).

{{</citation>}}


### (10/105) Depth Over RGB: Automatic Evaluation of Open Surgery Skills Using Depth Camera (Ido Zuckerman et al., 2024)

{{<citation>}}

Ido Zuckerman, Nicole Werner, Jonathan Kouchly, Emma Huston, Shannon DiMarco, Paul DiMusto, Shlomi Laufer. (2024)  
**Depth Over RGB: Automatic Evaluation of Open Surgery Skills Using Depth Camera**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Azure  
[Paper Link](http://arxiv.org/abs/2401.10037v1)  

---


**ABSTRACT**  
Purpose: In this paper, we present a novel approach to the automatic evaluation of open surgery skills using depth cameras. This work is intended to show that depth cameras achieve similar results to RGB cameras, which is the common method in the automatic evaluation of open surgery skills. Moreover, depth cameras offer advantages such as robustness to lighting variations, camera positioning, simplified data compression, and enhanced privacy, making them a promising alternative to RGB cameras.   Methods: Experts and novice surgeons completed two simulators of open suturing. We focused on hand and tool detection, and action segmentation in suturing procedures. YOLOv8 was used for tool detection in RGB and depth videos. Furthermore, UVAST and MSTCN++ were used for action segmentation. Our study includes the collection and annotation of a dataset recorded with Azure Kinect.   Results: We demonstrated that using depth cameras in object detection and action segmentation achieves comparable results to RGB cameras. Furthermore, we analyzed 3D hand path length, revealing significant differences between experts and novice surgeons, emphasizing the potential of depth cameras in capturing surgical skills. We also investigated the influence of camera angles on measurement accuracy, highlighting the advantages of 3D cameras in providing a more accurate representation of hand movements.   Conclusion: Our research contributes to advancing the field of surgical skill assessment by leveraging depth cameras for more reliable and privacy evaluations. The findings suggest that depth cameras can be valuable in assessing surgical skills and provide a foundation for future research in this area.

{{</citation>}}


### (11/105) CPCL: Cross-Modal Prototypical Contrastive Learning for Weakly Supervised Text-based Person Re-Identification (Yanwei Zheng et al., 2024)

{{<citation>}}

Yanwei Zheng, Xinpeng Zhao, Chuanlin Lan, Xiaowei Zhang, Bowen Huang, Jibin Yang, Dongxiao Yu. (2024)  
**CPCL: Cross-Modal Prototypical Contrastive Learning for Weakly Supervised Text-based Person Re-Identification**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Contrastive Learning  
[Paper Link](http://arxiv.org/abs/2401.10011v1)  

---


**ABSTRACT**  
Weakly supervised text-based person re-identification (TPRe-ID) seeks to retrieve images of a target person using textual descriptions, without relying on identity annotations and is more challenging and practical. The primary challenge is the intra-class differences, encompassing intra-modal feature variations and cross-modal semantic gaps. Prior works have focused on instance-level samples and ignored prototypical features of each person which are intrinsic and invariant. Toward this, we propose a Cross-Modal Prototypical Contrastive Learning (CPCL) method. In practice, the CPCL introduces the CLIP model to weakly supervised TPRe-ID for the first time, mapping visual and textual instances into a shared latent space. Subsequently, the proposed Prototypical Multi-modal Memory (PMM) module captures associations between heterogeneous modalities of image-text pairs belonging to the same person through the Hybrid Cross-modal Matching (HCM) module in a many-to-many mapping fashion. Moreover, the Outlier Pseudo Label Mining (OPLM) module further distinguishes valuable outlier samples from each modality, enhancing the creation of more reliable clusters by mining implicit relationships between image-text pairs. Experimental results demonstrate that our proposed CPCL attains state-of-the-art performance on all three public datasets, with a significant improvement of 11.58%, 8.77% and 5.25% in Rank@1 accuracy on CUHK-PEDES, ICFG-PEDES and RSTPReid datasets, respectively. The code is available at https://github.com/codeGallery24/CPCL.

{{</citation>}}


### (12/105) Advancing Large Multi-modal Models with Explicit Chain-of-Reasoning and Visual Question Generation (Kohei Uehara et al., 2024)

{{<citation>}}

Kohei Uehara, Nabarun Goswami, Hanqin Wang, Toshiaki Baba, Kohtaro Tanaka, Tomohiro Hashimoto, Kai Wang, Rei Ito, Takagi Naoya, Ryo Umagami, Yingyi Wen, Tanachai Anakewat, Tatsuya Harada. (2024)  
**Advancing Large Multi-modal Models with Explicit Chain-of-Reasoning and Visual Question Generation**  

---
Primary Category: cs.CV  
Categories: cs-CL, cs-CV, cs.CV  
Keywords: Language Model, Question Generation, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.10005v1)  

---


**ABSTRACT**  
The increasing demand for intelligent systems capable of interpreting and reasoning about visual content requires the development of Large Multi-Modal Models (LMMs) that are not only accurate but also have explicit reasoning capabilities. This paper presents a novel approach to imbue an LMM with the ability to conduct explicit reasoning based on visual content and textual instructions. We introduce a system that can ask a question to acquire necessary knowledge, thereby enhancing the robustness and explicability of the reasoning process. Our method comprises the development of a novel dataset generated by a Large Language Model (LLM), designed to promote chain-of-thought reasoning combined with a question-asking mechanism. We designed an LMM, which has high capabilities on region awareness to address the intricate requirements of image-text alignment. The model undergoes a three-stage training phase, starting with large-scale image-text alignment using a large-scale datasets, followed by instruction tuning, and fine-tuning with a focus on chain-of-thought reasoning. The results demonstrate a stride toward a more robust, accurate, and interpretable LMM, capable of reasoning explicitly and seeking information proactively when confronted with ambiguous visual input.

{{</citation>}}


### (13/105) MAMBA: Multi-level Aggregation via Memory Bank for Video Object Detection (Guanxiong Sun et al., 2024)

{{<citation>}}

Guanxiong Sun, Yang Hua, Guosheng Hu, Neil Robertson. (2024)  
**MAMBA: Multi-level Aggregation via Memory Bank for Video Object Detection**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet, Object Detection  
[Paper Link](http://arxiv.org/abs/2401.09923v1)  

---


**ABSTRACT**  
State-of-the-art video object detection methods maintain a memory structure, either a sliding window or a memory queue, to enhance the current frame using attention mechanisms. However, we argue that these memory structures are not efficient or sufficient because of two implied operations: (1) concatenating all features in memory for enhancement, leading to a heavy computational cost; (2) frame-wise memory updating, preventing the memory from capturing more temporal information. In this paper, we propose a multi-level aggregation architecture via memory bank called MAMBA. Specifically, our memory bank employs two novel operations to eliminate the disadvantages of existing methods: (1) light-weight key-set construction which can significantly reduce the computational cost; (2) fine-grained feature-wise updating strategy which enables our method to utilize knowledge from the whole video. To better enhance features from complementary levels, i.e., feature maps and proposals, we further propose a generalized enhancement operation (GEO) to aggregate multi-level features in a unified manner. We conduct extensive evaluations on the challenging ImageNetVID dataset. Compared with existing state-of-the-art methods, our method achieves superior performance in terms of both speed and accuracy. More remarkably, MAMBA achieves mAP of 83.7/84.6% at 12.6/9.1 FPS with ResNet-101. Code is available at https://github.com/guanxiongsun/video_feature_enhancement.

{{</citation>}}


### (14/105) BlenDA: Domain Adaptive Object Detection through diffusion-based blending (Tzuhsuan Huang et al., 2024)

{{<citation>}}

Tzuhsuan Huang, Chen-Che Huang, Chung-Hao Ku, Jun-Cheng Chen. (2024)  
**BlenDA: Domain Adaptive Object Detection through diffusion-based blending**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Object Detection, Transformer  
[Paper Link](http://arxiv.org/abs/2401.09921v1)  

---


**ABSTRACT**  
Unsupervised domain adaptation (UDA) aims to transfer a model learned using labeled data from the source domain to unlabeled data in the target domain. To address the large domain gap issue between the source and target domains, we propose a novel regularization method for domain adaptive object detection, BlenDA, by generating the pseudo samples of the intermediate domains and their corresponding soft domain labels for adaptation training. The intermediate samples are generated by dynamically blending the source images with their corresponding translated images using an off-the-shelf pre-trained text-to-image diffusion model which takes the text label of the target domain as input and has demonstrated superior image-to-image translation quality. Based on experimental results from two adaptation benchmarks, our proposed approach can significantly enhance the performance of the state-of-the-art domain adaptive object detector, Adversarial Query Transformer (AQT). Particularly, in the Cityscapes to Foggy Cityscapes adaptation, we achieve an impressive 53.4% mAP on the Foggy Cityscapes dataset, surpassing the previous state-of-the-art by 1.5%. It is worth noting that our proposed method is also applicable to various paradigms of domain adaptive object detection. The code is available at:https://github.com/aiiu-lab/BlenDA

{{</citation>}}


### (15/105) XAI-Enhanced Semantic Segmentation Models for Visual Quality Inspection (Tobias Clement et al., 2024)

{{<citation>}}

Tobias Clement, Truong Thanh Hung Nguyen, Mohamed Abdelaal, Hung Cao. (2024)  
**XAI-Enhanced Semantic Segmentation Models for Visual Quality Inspection**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI, Augmentation, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.09900v1)  

---


**ABSTRACT**  
Visual quality inspection systems, crucial in sectors like manufacturing and logistics, employ computer vision and machine learning for precise, rapid defect detection. However, their unexplained nature can hinder trust, error identification, and system improvement. This paper presents a framework to bolster visual quality inspection by using CAM-based explanations to refine semantic segmentation models. Our approach consists of 1) Model Training, 2) XAI-based Model Explanation, 3) XAI Evaluation, and 4) Annotation Augmentation for Model Enhancement, informed by explanations and expert insights. Evaluations show XAI-enhanced models surpass original DeepLabv3-ResNet101 models, especially in intricate object segmentation.

{{</citation>}}


### (16/105) Skeleton-Guided Instance Separation for Fine-Grained Segmentation in Microscopy (Jun Wang et al., 2024)

{{<citation>}}

Jun Wang, Chengfeng Zhou, Zhaoyan Ming, Lina Wei, Xudong Jiang, Dahong Qian. (2024)  
**Skeleton-Guided Instance Separation for Fine-Grained Segmentation in Microscopy**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Semi-Supervised  
[Paper Link](http://arxiv.org/abs/2401.09895v1)  

---


**ABSTRACT**  
One of the fundamental challenges in microscopy (MS) image analysis is instance segmentation (IS), particularly when segmenting cluster regions where multiple objects of varying sizes and shapes may be connected or even overlapped in arbitrary orientations. Existing IS methods usually fail in handling such scenarios, as they rely on coarse instance representations such as keypoints and horizontal bounding boxes (h-bboxes). In this paper, we propose a novel one-stage framework named A2B-IS to address this challenge and enhance the accuracy of IS in MS images. Our approach represents each instance with a pixel-level mask map and a rotated bounding box (r-bbox). Unlike two-stage methods that use box proposals for segmentations, our method decouples mask and box predictions, enabling simultaneous processing to streamline the model pipeline. Additionally, we introduce a Gaussian skeleton map to aid the IS task in two key ways: (1) It guides anchor placement, reducing computational costs while improving the model's capacity to learn RoI-aware features by filtering out noise from background regions. (2) It ensures accurate isolation of densely packed instances by rectifying erroneous box predictions near instance boundaries. To further enhance the performance, we integrate two modules into the framework: (1) An Atrous Attention Block (A2B) designed to extract high-resolution feature maps with fine-grained multiscale information, and (2) A Semi-Supervised Learning (SSL) strategy that leverages both labeled and unlabeled images for model training. Our method has been thoroughly validated on two large-scale MS datasets, demonstrating its superiority over most state-of-the-art approaches.

{{</citation>}}


### (17/105) Question-Answer Cross Language Image Matching for Weakly Supervised Semantic Segmentation (Songhe Deng et al., 2024)

{{<citation>}}

Songhe Deng, Wei Zhuo, Jinheng Xie, Linlin Shen. (2024)  
**Question-Answer Cross Language Image Matching for Weakly Supervised Semantic Segmentation**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: QA, Question Answering, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.09883v1)  

---


**ABSTRACT**  
Class Activation Map (CAM) has emerged as a popular tool for weakly supervised semantic segmentation (WSSS), allowing the localization of object regions in an image using only image-level labels. However, existing CAM methods suffer from under-activation of target object regions and false-activation of background regions due to the fact that a lack of detailed supervision can hinder the model's ability to understand the image as a whole. In this paper, we propose a novel Question-Answer Cross-Language-Image Matching framework for WSSS (QA-CLIMS), leveraging the vision-language foundation model to maximize the text-based understanding of images and guide the generation of activation maps. First, a series of carefully designed questions are posed to the VQA (Visual Question Answering) model with Question-Answer Prompt Engineering (QAPE) to generate a corpus of both foreground target objects and backgrounds that are adaptive to query images. We then employ contrastive learning in a Region Image Text Contrastive (RITC) network to compare the obtained foreground and background regions with the generated corpus. Our approach exploits the rich textual information from the open vocabulary as additional supervision, enabling the model to generate high-quality CAMs with a more complete object region and reduce false-activation of background regions. We conduct extensive analysis to validate the proposed method and show that our approach performs state-of-the-art on both PASCAL VOC 2012 and MS COCO datasets. Code is available at: https://github.com/CVI-SZU/QA-CLIMS

{{</citation>}}


### (18/105) Boosting Few-Shot Segmentation via Instance-Aware Data Augmentation and Local Consensus Guided Cross Attention (Li Guo et al., 2024)

{{<citation>}}

Li Guo, Haoming Liu, Yuxuan Xia, Chengyu Zhang, Xiaochen Lu. (2024)  
**Boosting Few-Shot Segmentation via Instance-Aware Data Augmentation and Local Consensus Guided Cross Attention**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Attention, Augmentation, Few-Shot  
[Paper Link](http://arxiv.org/abs/2401.09866v1)  

---


**ABSTRACT**  
Few-shot segmentation aims to train a segmentation model that can fast adapt to a novel task for which only a few annotated images are provided. Most recent models have adopted a prototype-based paradigm for few-shot inference. These approaches may have limited generalization capacity beyond the standard 1- or 5-shot settings. In this paper, we closely examine and reevaluate the fine-tuning based learning scheme that fine-tunes the classification layer of a deep segmentation network pre-trained on diverse base classes. To improve the generalizability of the classification layer optimized with sparsely annotated samples, we introduce an instance-aware data augmentation (IDA) strategy that augments the support images based on the relative sizes of the target objects. The proposed IDA effectively increases the support set's diversity and promotes the distribution consistency between support and query images. On the other hand, the large visual difference between query and support images may hinder knowledge transfer and cripple the segmentation performance. To cope with this challenge, we introduce the local consensus guided cross attention (LCCA) to align the query feature with support features based on their dense correlation, further improving the model's generalizability to the query image. The significant performance improvements on the standard few-shot segmentation benchmarks PASCAL-$5^i$ and COCO-$20^i$ verify the efficacy of our proposed method.

{{</citation>}}


### (19/105) Temporal Insight Enhancement: Mitigating Temporal Hallucination in Multimodal Large Language Models (Li Sun et al., 2024)

{{<citation>}}

Li Sun, Liuan Wang, Jun Sun, Takayuki Okatani. (2024)  
**Temporal Insight Enhancement: Mitigating Temporal Hallucination in Multimodal Large Language Models**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09861v1)  

---


**ABSTRACT**  
Recent advancements in Multimodal Large Language Models (MLLMs) have significantly enhanced the comprehension of multimedia content, bringing together diverse modalities such as text, images, and videos. However, a critical challenge faced by these models, especially when processing video inputs, is the occurrence of hallucinations - erroneous perceptions or interpretations, particularly at the event level. This study introduces an innovative method to address event-level hallucinations in MLLMs, focusing on specific temporal understanding in video content. Our approach leverages a novel framework that extracts and utilizes event-specific information from both the event query and the provided video to refine MLLMs' response. We propose a unique mechanism that decomposes on-demand event queries into iconic actions. Subsequently, we employ models like CLIP and BLIP2 to predict specific timestamps for event occurrences. Our evaluation, conducted using the Charades-STA dataset, demonstrates a significant reduction in temporal hallucinations and an improvement in the quality of event-related responses. This research not only provides a new perspective in addressing a critical limitation of MLLMs but also contributes a quantitatively measurable method for evaluating MLLMs in the context of temporal-related questions.

{{</citation>}}


### (20/105) Enhancing the Fairness and Performance of Edge Cameras with Explainable AI (Truong Thanh Hung Nguyen et al., 2024)

{{<citation>}}

Truong Thanh Hung Nguyen, Vo Thanh Khang Nguyen, Quoc Hung Cao, Van Binh Truong, Quoc Khanh Nguyen, Hung Cao. (2024)  
**Enhancing the Fairness and Performance of Edge Cameras with Explainable AI**  

---
Primary Category: cs.CV  
Categories: cs-AI, cs-CV, cs.CV  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09852v1)  

---


**ABSTRACT**  
The rising use of Artificial Intelligence (AI) in human detection on Edge camera systems has led to accurate but complex models, challenging to interpret and debug. Our research presents a diagnostic method using Explainable AI (XAI) for model debugging, with expert-driven problem identification and solution creation. Validated on the Bytetrack model in a real-world office Edge network, we found the training dataset as the main bias source and suggested model augmentation as a solution. Our approach helps identify model biases, essential for achieving fair and trustworthy models.

{{</citation>}}


### (21/105) Exploring Latent Cross-Channel Embedding for Accurate 3D Human Pose Reconstruction in a Diffusion Framework (Junkun Jiang et al., 2024)

{{<citation>}}

Junkun Jiang, Jie Chen. (2024)  
**Exploring Latent Cross-Channel Embedding for Accurate 3D Human Pose Reconstruction in a Diffusion Framework**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Embedding  
[Paper Link](http://arxiv.org/abs/2401.09836v1)  

---


**ABSTRACT**  
Monocular 3D human pose estimation poses significant challenges due to the inherent depth ambiguities that arise during the reprojection process from 2D to 3D. Conventional approaches that rely on estimating an over-fit projection matrix struggle to effectively address these challenges and often result in noisy outputs. Recent advancements in diffusion models have shown promise in incorporating structural priors to address reprojection ambiguities. However, there is still ample room for improvement as these methods often overlook the exploration of correlation between the 2D and 3D joint-level features. In this study, we propose a novel cross-channel embedding framework that aims to fully explore the correlation between joint-level features of 3D coordinates and their 2D projections. In addition, we introduce a context guidance mechanism to facilitate the propagation of joint graph attention across latent channels during the iterative diffusion process. To evaluate the effectiveness of our proposed method, we conduct experiments on two benchmark datasets, namely Human3.6M and MPI-INF-3DHP. Our results demonstrate a significant improvement in terms of reconstruction accuracy compared to state-of-the-art methods. The code for our method will be made available online for further reference.

{{</citation>}}


### (22/105) Boosting Few-Shot Semantic Segmentation Via Segment Anything Model (Chen-Bin Feng et al., 2024)

{{<citation>}}

Chen-Bin Feng, Qi Lai, Kangdao Liu, Houcheng Su, Chi-Man Vong. (2024)  
**Boosting Few-Shot Semantic Segmentation Via Segment Anything Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: Few-Shot, Semantic Segmentation  
[Paper Link](http://arxiv.org/abs/2401.09826v1)  

---


**ABSTRACT**  
In semantic segmentation, accurate prediction masks are crucial for downstream tasks such as medical image analysis and image editing. Due to the lack of annotated data, few-shot semantic segmentation (FSS) performs poorly in predicting masks with precise contours. Recently, we have noticed that the large foundation model segment anything model (SAM) performs well in processing detailed features. Inspired by SAM, we propose FSS-SAM to boost FSS methods by addressing the issue of inaccurate contour. The FSS-SAM is training-free. It works as a post-processing tool for any FSS methods and can improve the accuracy of predicted masks. Specifically, we use predicted masks from FSS methods to generate prompts and then use SAM to predict new masks. To avoid predicting wrong masks with SAM, we propose a prediction result selection (PRS) algorithm. The algorithm can remarkably decrease wrong predictions. Experiment results on public datasets show that our method is superior to base FSS methods in both quantitative and qualitative aspects.

{{</citation>}}


### (23/105) Enhancing Small Object Encoding in Deep Neural Networks: Introducing Fast&Focused-Net with Volume-wise Dot Product Layer (Ali Tofik et al., 2024)

{{<citation>}}

Ali Tofik, Roy Partha Pratim. (2024)  
**Enhancing Small Object Encoding in Deep Neural Networks: Introducing Fast&Focused-Net with Volume-wise Dot Product Layer**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: ImageNet  
[Paper Link](http://arxiv.org/abs/2401.09823v1)  

---


**ABSTRACT**  
In this paper, we introduce Fast&Focused-Net, a novel deep neural network architecture tailored for efficiently encoding small objects into fixed-length feature vectors. Contrary to conventional Convolutional Neural Networks (CNNs), Fast&Focused-Net employs a series of our newly proposed layer, the Volume-wise Dot Product (VDP) layer, designed to address several inherent limitations of CNNs. Specifically, CNNs often exhibit a smaller effective receptive field than their theoretical counterparts, limiting their vision span. Additionally, the initial layers in CNNs produce low-dimensional feature vectors, presenting a bottleneck for subsequent learning. Lastly, the computational overhead of CNNs, particularly in capturing diverse image regions by parameter sharing, is significantly high. The VDP layer, at the heart of Fast&Focused-Net, aims to remedy these issues by efficiently covering the entire image patch information with reduced computational demand. Experimental results demonstrate the prowess of Fast&Focused-Net in a variety of applications. For small object classification tasks, our network outperformed state-of-the-art methods on datasets such as CIFAR-10, CIFAR-100, STL-10, SVHN-Cropped, and Fashion-MNIST. In the context of larger image classification, when combined with a transformer encoder (ViT), Fast&Focused-Net produced competitive results for OpenImages V6, ImageNet-1K, and Places365 datasets. Moreover, the same combination showcased unparalleled performance in text recognition tasks across SVT, IC15, SVTP, and HOST datasets. This paper presents the architecture, the underlying motivation, and extensive empirical evidence suggesting that Fast&Focused-Net is a promising direction for efficient and focused deep learning.

{{</citation>}}


### (24/105) SlideAVSR: A Dataset of Paper Explanation Videos for Audio-Visual Speech Recognition (Hao Wang et al., 2024)

{{<citation>}}

Hao Wang, Shuhei Kurita, Shuichiro Shimizu, Daisuke Kawahara. (2024)  
**SlideAVSR: A Dataset of Paper Explanation Videos for Audio-Visual Speech Recognition**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs-MM, cs-SD, cs.CV, eess-AS  
Keywords: Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.09759v1)  

---


**ABSTRACT**  
Audio-visual speech recognition (AVSR) is a multimodal extension of automatic speech recognition (ASR), using video as a complement to audio. In AVSR, considerable efforts have been directed at datasets for facial features such as lip-readings, while they often fall short in evaluating the image comprehension capabilities in broader contexts. In this paper, we construct SlideAVSR, an AVSR dataset using scientific paper explanation videos. SlideAVSR provides a new benchmark where models transcribe speech utterances with texts on the slides on the presentation recordings. As technical terminologies that are frequent in paper explanations are notoriously challenging to transcribe without reference texts, our SlideAVSR dataset spotlights a new aspect of AVSR problems. As a simple yet effective baseline, we propose DocWhisper, an AVSR model that can refer to textual information from slides, and confirm its effectiveness on SlideAVSR.

{{</citation>}}


### (25/105) Image Translation as Diffusion Visual Programmers (Cheng Han et al., 2024)

{{<citation>}}

Cheng Han, James C. Liang, Qifan Wang, Majid Rabbani, Sohail Dianat, Raghuveer Rao, Ying Nian Wu, Dongfang Liu. (2024)  
**Image Translation as Diffusion Visual Programmers**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2401.09742v1)  

---


**ABSTRACT**  
We introduce the novel Diffusion Visual Programmer (DVP), a neuro-symbolic image translation framework. Our proposed DVP seamlessly embeds a condition-flexible diffusion model within the GPT architecture, orchestrating a coherent sequence of visual programs (i.e., computer vision models) for various pro-symbolic steps, which span RoI identification, style transfer, and position manipulation, facilitating transparent and controllable image translation processes. Extensive experiments demonstrate DVP's remarkable performance, surpassing concurrent arts. This success can be attributed to several key features of DVP: First, DVP achieves condition-flexible translation via instance normalization, enabling the model to eliminate sensitivity caused by the manual guidance and optimally focus on textual descriptions for high-quality content generation. Second, the framework enhances in-context reasoning by deciphering intricate high-dimensional concepts in feature spaces into more accessible low-dimensional symbols (e.g., [Prompt], [RoI object]), allowing for localized, context-free editing while maintaining overall coherence. Last but not least, DVP improves systemic controllability and explainability by offering explicit symbolic representations at each programming stage, empowering users to intuitively interpret and modify results. Our research marks a substantial step towards harmonizing artificial image translation processes with cognitive intelligence, promising broader applications.

{{</citation>}}


### (26/105) SkyEyeGPT: Unifying Remote Sensing Vision-Language Tasks via Instruction Tuning with Large Language Model (Yang Zhan et al., 2024)

{{<citation>}}

Yang Zhan, Zhitong Xiong, Yuan Yuan. (2024)  
**SkyEyeGPT: Unifying Remote Sensing Vision-Language Tasks via Instruction Tuning with Large Language Model**  

---
Primary Category: cs.CV  
Categories: cs-CV, cs.CV  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09712v1)  

---


**ABSTRACT**  
Large language models (LLMs) have recently been extended to the vision-language realm, obtaining impressive general multi-modal capabilities. However, the exploration of multi-modal large language models (MLLMs) for remote sensing (RS) data is still in its infancy, and the performance is not satisfactory. In this work, we introduce SkyEyeGPT, a unified multi-modal large language model specifically designed for RS vision-language understanding. To this end, we meticulously curate an RS multi-modal instruction tuning dataset, including single-task and multi-task conversation instructions. After manual verification, we obtain a high-quality RS instruction-following dataset with 968k samples. Our research demonstrates that with a simple yet effective design, SkyEyeGPT works surprisingly well on considerably different tasks without the need for extra encoding modules. Specifically, after projecting RS visual features to the language domain via an alignment layer, they are fed jointly with task-specific instructions into an LLM-based RS decoder to predict answers for RS open-ended tasks. In addition, we design a two-stage tuning method to enhance instruction-following and multi-turn dialogue ability at different granularities. Experiments on 8 datasets for RS vision-language tasks demonstrate SkyEyeGPT's superiority in image-level and region-level tasks, such as captioning and visual grounding. In particular, SkyEyeGPT exhibits encouraging results compared to GPT-4V in some qualitative tests. The online demo, code, and dataset will be released in https://github.com/ZhanYang-nwpu/SkyEyeGPT.

{{</citation>}}


### (27/105) Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack (Zhongliang Guo et al., 2024)

{{<citation>}}

Zhongliang Guo, Kaixuan Wang, Weiye Li, Yifei Qian, Ognjen Arandjelović, Lei Fang. (2024)  
**Artwork Protection Against Neural Style Transfer Using Locally Adaptive Adversarial Color Attack**  

---
Primary Category: cs.CV  
Categories: cs-CR, cs-CV, cs-LG, cs.CV, eess-IV  
Keywords: Style Transfer  
[Paper Link](http://arxiv.org/abs/2401.09673v1)  

---


**ABSTRACT**  
Neural style transfer (NST) is widely adopted in computer vision to generate new images with arbitrary styles. This process leverages neural networks to merge aesthetic elements of a style image with the structural aspects of a content image into a harmoniously integrated visual result. However, unauthorized NST can exploit artwork. Such misuse raises socio-technical concerns regarding artists' rights and motivates the development of technical approaches for the proactive protection of original creations. Adversarial attack is a concept primarily explored in machine learning security. Our work introduces this technique to protect artists' intellectual property. In this paper Locally Adaptive Adversarial Color Attack (LAACA), a method for altering images in a manner imperceptible to the human eyes but disruptive to NST. Specifically, we design perturbations targeting image areas rich in high-frequency content, generated by disrupting intermediate features. Our experiments and user study confirm that by attacking NST using the proposed method results in visually worse neural style transfer, thus making it an effective solution for visual artwork protection.

{{</citation>}}


## cs.CL (22)



### (28/105) ChatQA: Building GPT-4 Level Conversational QA Models (Zihan Liu et al., 2024)

{{<citation>}}

Zihan Liu, Wei Ping, Rajarshi Roy, Peng Xu, Mohammad Shoeybi, Bryan Catanzaro. (2024)  
**ChatQA: Building GPT-4 Level Conversational QA Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-IR, cs-LG, cs.CL  
Keywords: AI, GPT, GPT-4, QA  
[Paper Link](http://arxiv.org/abs/2401.10225v1)  

---


**ABSTRACT**  
In this work, we introduce ChatQA, a family of conversational question answering (QA) models, that obtain GPT-4 level accuracies. Specifically, we propose a two-stage instruction tuning method that can significantly improve the zero-shot conversational QA results from large language models (LLMs). To handle retrieval in conversational QA, we fine-tune a dense retriever on a multi-turn QA dataset, which provides comparable results to using the state-of-the-art query rewriting model while largely reducing deployment cost. Notably, our ChatQA-70B can outperform GPT-4 in terms of average score on 10 conversational QA datasets (54.14 vs. 53.90), without relying on any synthetic data from OpenAI GPT models.

{{</citation>}}


### (29/105) Chem-FINESE: Validating Fine-Grained Few-shot Entity Extraction through Text Reconstruction (Qingyun Wang et al., 2024)

{{<citation>}}

Qingyun Wang, Zixuan Zhang, Hongxiang Li, Xuan Liu, Jiawei Han, Heng Ji, Huimin Zhao. (2024)  
**Chem-FINESE: Validating Fine-Grained Few-shot Entity Extraction through Text Reconstruction**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-LG, cs.CL  
Keywords: NER  
[Paper Link](http://arxiv.org/abs/2401.10189v1)  

---


**ABSTRACT**  
Fine-grained few-shot entity extraction in the chemical domain faces two unique challenges. First, compared with entity extraction tasks in the general domain, sentences from chemical papers usually contain more entities. Moreover, entity extraction models usually have difficulty extracting entities of long-tailed types. In this paper, we propose Chem-FINESE, a novel sequence-to-sequence (seq2seq) based few-shot entity extraction approach, to address these two challenges. Our Chem-FINESE has two components: a seq2seq entity extractor to extract named entities from the input sentence and a seq2seq self-validation module to reconstruct the original input sentence from extracted entities. Inspired by the fact that a good entity extraction system needs to extract entities faithfully, our new self-validation module leverages entity extraction results to reconstruct the original input sentence. Besides, we design a new contrastive loss to reduce excessive copying during the extraction process. Finally, we release ChemNER+, a new fine-grained chemical entity extraction dataset that is annotated by domain experts with the ChemNER schema. Experiments in few-shot settings with both ChemNER+ and CHEMET datasets show that our newly proposed framework has contributed up to 8.26% and 6.84% absolute F1-score gains respectively.

{{</citation>}}


### (30/105) Beyond Reference-Based Metrics: Analyzing Behaviors of Open LLMs on Data-to-Text Generation (Zdeněk Kasner et al., 2024)

{{<citation>}}

Zdeněk Kasner, Ondřej Dušek. (2024)  
**Beyond Reference-Based Metrics: Analyzing Behaviors of Open LLMs on Data-to-Text Generation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, GPT-4, Text Generation  
[Paper Link](http://arxiv.org/abs/2401.10186v1)  

---


**ABSTRACT**  
We investigate to which extent open large language models (LLMs) can generate coherent and relevant text from structured data. To prevent bias from benchmarks leaked into LLM training data, we collect Quintd-1: an ad-hoc benchmark for five data-to-text (D2T) generation tasks, consisting of structured data records in standard formats gathered from public APIs. We leverage reference-free evaluation metrics and LLMs' in-context learning capabilities, allowing us to test the models with no human-written references. Our evaluation focuses on annotating semantic accuracy errors on token-level, combining human annotators and a metric based on GPT-4. Our systematic examination of the models' behavior across domains and tasks suggests that state-of-the-art open LLMs with 7B parameters can generate fluent and coherent text from various standard data formats in zero-shot settings. However, we also show that semantic accuracy of the outputs remains a major issue: on our benchmark, 80% of outputs of open LLMs contain a semantic error according to human annotators (91% according to GPT-4). Our code, data, and model outputs are available at https://d2t-llm.github.io.

{{</citation>}}


### (31/105) Marrying Adapters and Mixup to Efficiently Enhance the Adversarial Robustness of Pre-Trained Language Models for Text Classification (Tuc Nguyen et al., 2024)

{{<citation>}}

Tuc Nguyen, Thai Le. (2024)  
**Marrying Adapters and Mixup to Efficiently Enhance the Adversarial Robustness of Pre-Trained Language Models for Text Classification**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, Text Classification  
[Paper Link](http://arxiv.org/abs/2401.10111v1)  

---


**ABSTRACT**  
Existing works show that augmenting training data of neural networks using both clean and adversarial examples can enhance their generalizability under adversarial attacks. However, this training approach often leads to performance degradation on clean inputs. Additionally, it requires frequent re-training of the entire model to account for new attack types, resulting in significant and costly computations. Such limitations make adversarial training mechanisms less practical, particularly for complex Pre-trained Language Models (PLMs) with millions or even billions of parameters. To overcome these challenges while still harnessing the theoretical benefits of adversarial training, this study combines two concepts: (1) adapters, which enable parameter-efficient fine-tuning, and (2) Mixup, which train NNs via convex combinations of pairs data pairs. Intuitively, we propose to fine-tune PLMs through convex combinations of non-data pairs of fine-tuned adapters, one trained with clean and another trained with adversarial examples. Our experiments show that the proposed method achieves the best trade-off between training efficiency and predictive performance, both with and without attacks compared to other baselines on a variety of downstream tasks.

{{</citation>}}


### (32/105) Power in Numbers: Robust reading comprehension by finetuning with four adversarial sentences per example (Ariel Marcus, 2024)

{{<citation>}}

Ariel Marcus. (2024)  
**Power in Numbers: Robust reading comprehension by finetuning with four adversarial sentences per example**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Question Answering  
[Paper Link](http://arxiv.org/abs/2401.10091v1)  

---


**ABSTRACT**  
Recent models have achieved human level performance on the Stanford Question Answering Dataset when using F1 scores to evaluate the reading comprehension task. Yet, teaching machines to comprehend text has not been solved in the general case. By appending one adversarial sentence to the context paragraph, past research has shown that the F1 scores from reading comprehension models drop almost in half. In this paper, I replicate past adversarial research with a new model, ELECTRA-Small, and demonstrate that the new model's F1 score drops from 83.9% to 29.2%. To improve ELECTRA-Small's resistance to this attack, I finetune the model on SQuAD v1.1 training examples with one to five adversarial sentences appended to the context paragraph. Like past research, I find that the finetuned model on one adversarial sentence does not generalize well across evaluation datasets. However, when finetuned on four or five adversarial sentences the model attains an F1 score of more than 70% on most evaluation datasets with multiple appended and prepended adversarial sentences. The results suggest that with enough examples we can make models robust to adversarial attacks.

{{</citation>}}


### (33/105) Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs (Haritz Puerto et al., 2024)

{{<citation>}}

Haritz Puerto, Martin Tutek, Somak Aditya, Xiaodan Zhu, Iryna Gurevych. (2024)  
**Code Prompting Elicits Conditional Reasoning Abilities in Text+Code LLMs**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: GPT, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.10065v1)  

---


**ABSTRACT**  
Reasoning is a fundamental component for achieving language understanding. Among the multiple types of reasoning, conditional reasoning, the ability to draw different conclusions depending on some condition, has been understudied in large language models (LLMs). Recent prompting methods, such as chain of thought, have significantly improved LLMs on reasoning tasks. Nevertheless, there is still little understanding of what triggers reasoning abilities in LLMs. We hypothesize that code prompts can trigger conditional reasoning in LLMs trained on text and code. We propose a chain of prompts that transforms a natural language problem into code and prompts the LLM with the generated code. Our experiments find that code prompts exhibit a performance boost between 2.6 and 7.7 points on GPT 3.5 across multiple datasets requiring conditional reasoning. We then conduct experiments to discover how code prompts elicit conditional reasoning abilities and through which features. We observe that prompts need to contain natural language text accompanied by high-quality code that closely represents the semantics of the instance text. Furthermore, we show that code prompts are more efficient, requiring fewer demonstrations, and that they trigger superior state tracking of variables or key entities.

{{</citation>}}


### (34/105) Large Language Models for Scientific Information Extraction: An Empirical Study for Virology (Mahsa Shamsabadi et al., 2024)

{{<citation>}}

Mahsa Shamsabadi, Jennifer D'Souza, Sören Auer. (2024)  
**Large Language Models for Scientific Information Extraction: An Empirical Study for Virology**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-DL, cs-IT, cs.CL, math-IT  
Keywords: Amazon, GPT, Information Extraction, Language Model, T5  
[Paper Link](http://arxiv.org/abs/2401.10040v1)  

---


**ABSTRACT**  
In this paper, we champion the use of structured and semantic content representation of discourse-based scholarly communication, inspired by tools like Wikipedia infoboxes or structured Amazon product descriptions. These representations provide users with a concise overview, aiding scientists in navigating the dense academic landscape. Our novel automated approach leverages the robust text generation capabilities of LLMs to produce structured scholarly contribution summaries, offering both a practical solution and insights into LLMs' emergent abilities.   For LLMs, the prime focus is on improving their general intelligence as conversational agents. We argue that these models can also be applied effectively in information extraction (IE), specifically in complex IE tasks within terse domains like Science. This paradigm shift replaces the traditional modular, pipelined machine learning approach with a simpler objective expressed through instructions. Our results show that finetuned FLAN-T5 with 1000x fewer parameters than the state-of-the-art GPT-davinci is competitive for the task.

{{</citation>}}


### (35/105) Self-Rewarding Language Models (Weizhe Yuan et al., 2024)

{{<citation>}}

Weizhe Yuan, Richard Yuanzhe Pang, Kyunghyun Cho, Sainbayar Sukhbaatar, Jing Xu, Jason Weston. (2024)  
**Self-Rewarding Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.10020v1)  

---


**ABSTRACT**  
We posit that to achieve superhuman agents, future models require superhuman feedback in order to provide an adequate training signal. Current approaches commonly train reward models from human preferences, which may then be bottlenecked by human performance level, and secondly these separate frozen reward models cannot then learn to improve during LLM training. In this work, we study Self-Rewarding Language Models, where the language model itself is used via LLM-as-a-Judge prompting to provide its own rewards during training. We show that during Iterative DPO training that not only does instruction following ability improve, but also the ability to provide high-quality rewards to itself. Fine-tuning Llama 2 70B on three iterations of our approach yields a model that outperforms many existing systems on the AlpacaEval 2.0 leaderboard, including Claude 2, Gemini Pro, and GPT-4 0613. While only a preliminary study, this work opens the door to the possibility of models that can continually improve in both axes.

{{</citation>}}


### (36/105) R-Judge: Benchmarking Safety Risk Awareness for LLM Agents (Tongxin Yuan et al., 2024)

{{<citation>}}

Tongxin Yuan, Zhiwei He, Lingzhong Dong, Yiming Wang, Ruijie Zhao, Tian Xia, Lizhen Xu, Binglin Zhou, Fangqi Li, Zhuosheng Zhang, Rui Wang, Gongshen Liu. (2024)  
**R-Judge: Benchmarking Safety Risk Awareness for LLM Agents**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs.CL  
Keywords: GPT, GPT-4  
[Paper Link](http://arxiv.org/abs/2401.10019v1)  

---


**ABSTRACT**  
Large language models (LLMs) have exhibited great potential in autonomously completing tasks across real-world applications. Despite this, these LLM agents introduce unexpected safety risks when operating in interactive environments. Instead of centering on LLM-generated content safety in most prior studies, this work addresses the imperative need for benchmarking the behavioral safety of LLM agents within diverse environments. We introduce R-Judge, a benchmark crafted to evaluate the proficiency of LLMs in judging safety risks given agent interaction records. R-Judge comprises 162 agent interaction records, encompassing 27 key risk scenarios among 7 application categories and 10 risk types. It incorporates human consensus on safety with annotated safety risk labels and high-quality risk descriptions. Utilizing R-Judge, we conduct a comprehensive evaluation of 8 prominent LLMs commonly employed as the backbone for agents. The best-performing model, GPT-4, achieves 72.29% in contrast to the human score of 89.38%, showing considerable room for enhancing the risk awareness of LLMs. Notably, leveraging risk descriptions as environment feedback significantly improves model performance, revealing the importance of salient safety risk feedback. Furthermore, we design an effective chain of safety analysis technique to help the judgment of safety risks and conduct an in-depth case study to facilitate future research. R-Judge is publicly available at https://github.com/Lordog/R-Judge.

{{</citation>}}


### (37/105) Gender Bias in Machine Translation and The Era of Large Language Models (Eva Vanmassenhove, 2024)

{{<citation>}}

Eva Vanmassenhove. (2024)  
**Gender Bias in Machine Translation and The Era of Large Language Models**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: Bias, ChatGPT, GPT, GPT-3.5, Language Model, Machine Translation, Transformer  
[Paper Link](http://arxiv.org/abs/2401.10016v1)  

---


**ABSTRACT**  
This chapter examines the role of Machine Translation in perpetuating gender bias, highlighting the challenges posed by cross-linguistic settings and statistical dependencies. A comprehensive overview of relevant existing work related to gender bias in both conventional Neural Machine Translation approaches and Generative Pretrained Transformer models employed as Machine Translation systems is provided. Through an experiment using ChatGPT (based on GPT-3.5) in an English-Italian translation context, we further assess ChatGPT's current capacity to address gender bias. The findings emphasize the ongoing need for advancements in mitigating bias in Machine Translation systems and underscore the importance of fostering fairness and inclusivity in language technologies.

{{</citation>}}


### (38/105) Towards Hierarchical Spoken Language Dysfluency Modeling (Jiachen Lian et al., 2024)

{{<citation>}}

Jiachen Lian, Gopala Anumanchipalli. (2024)  
**Towards Hierarchical Spoken Language Dysfluency Modeling**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL, eess-AS  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.10015v1)  

---


**ABSTRACT**  
Speech dysfluency modeling is the bottleneck for both speech therapy and language learning. However, there is no AI solution to systematically tackle this problem. We first propose to define the concept of dysfluent speech and dysfluent speech modeling. We then present Hierarchical Unconstrained Dysfluency Modeling (H-UDM) approach that addresses both dysfluency transcription and detection to eliminate the need for extensive manual annotation. Furthermore, we introduce a simulated dysfluent dataset called VCTK++ to enhance the capabilities of H-UDM in phonetic transcription. Our experimental results demonstrate the effectiveness and robustness of our proposed methods in both transcription and detection tasks.

{{</citation>}}


### (39/105) Distantly Supervised Morpho-Syntactic Model for Relation Extraction (Nicolas Gutehrlé et al., 2024)

{{<citation>}}

Nicolas Gutehrlé, Iana Atanassova. (2024)  
**Distantly Supervised Morpho-Syntactic Model for Relation Extraction**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Information Extraction, Relation Extraction  
[Paper Link](http://arxiv.org/abs/2401.10002v1)  

---


**ABSTRACT**  
The task of Information Extraction (IE) involves automatically converting unstructured textual content into structured data. Most research in this field concentrates on extracting all facts or a specific set of relationships from documents. In this paper, we present a method for the extraction and categorisation of an unrestricted set of relationships from text. Our method relies on morpho-syntactic extraction patterns obtained by a distant supervision method, and creates Syntactic and Semantic Indices to extract and classify candidate graphs. We evaluate our approach on six datasets built on Wikidata and Wikipedia. The evaluation shows that our approach can achieve Precision scores of up to 0.85, but with lower Recall and F1 scores. Our approach allows to quickly create rule-based systems for Information Extraction and to build annotated datasets to train machine-learning and deep-learning based classifiers.

{{</citation>}}


### (40/105) Gradable ChatGPT Translation Evaluation (Hui Jiao et al., 2024)

{{<citation>}}

Hui Jiao, Bei Peng, Lu Zong, Xiaojun Zhang, Xinwei Li. (2024)  
**Gradable ChatGPT Translation Evaluation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.09984v1)  

---


**ABSTRACT**  
ChatGPT, as a language model based on large-scale pre-training, has exerted a profound influence on the domain of machine translation. In ChatGPT, a "Prompt" refers to a segment of text or instruction employed to steer the model towards generating a specific category of response. The design of the translation prompt emerges as a key aspect that can wield influence over factors such as the style, precision and accuracy of the translation to a certain extent. However, there is a lack of a common standard and methodology on how to design and select a translation prompt. Accordingly, this paper proposes a generic taxonomy, which defines gradable translation prompts in terms of expression type, translation style, POS information and explicit statement, thus facilitating the construction of prompts endowed with distinct attributes tailored for various translation tasks. Specific experiments and cases are selected to validate and illustrate the effectiveness of the method.

{{</citation>}}


### (41/105) Better Explain Transformers by Illuminating Important Information (Linxin Song et al., 2024)

{{<citation>}}

Linxin Song, Yan Cui, Ao Luo, Freddy Lecue, Irene Li. (2024)  
**Better Explain Transformers by Illuminating Important Information**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: NLP, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.09972v1)  

---


**ABSTRACT**  
Transformer-based models excel in various natural language processing (NLP) tasks, attracting countless efforts to explain their inner workings. Prior methods explain Transformers by focusing on the raw gradient and attention as token attribution scores, where non-relevant information is often considered during explanation computation, resulting in confusing results. In this work, we propose highlighting the important information and eliminating irrelevant information by a refined information flow on top of the layer-wise relevance propagation (LRP) method. Specifically, we consider identifying syntactic and positional heads as important attention heads and focus on the relevance obtained from these important heads. Experimental results demonstrate that irrelevant information does distort output attribution scores and then should be masked during explanation computation. Compared to eight baselines on both classification and question-answering datasets, our method consistently outperforms with over 3\% to 33\% improvement on explanation metrics, providing superior explanation performance. Our anonymous code repository is available at: https://github.com/LinxinS97/Mask-LRP

{{</citation>}}


### (42/105) Sketch-Guided Constrained Decoding for Boosting Blackbox Large Language Models without Logit Access (Saibo Geng et al., 2024)

{{<citation>}}

Saibo Geng, Berkay Döner, Chris Wendler, Martin Josifoski, Robert West. (2024)  
**Sketch-Guided Constrained Decoding for Boosting Blackbox Large Language Models without Logit Access**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Language Model, NLP, Sketch  
[Paper Link](http://arxiv.org/abs/2401.09967v1)  

---


**ABSTRACT**  
Constrained decoding, a technique for enforcing constraints on language model outputs, offers a way to control text generation without retraining or architectural modifications. Its application is, however, typically restricted to models that give users access to next-token distributions (usually via softmax logits), which poses a limitation with blackbox large language models (LLMs). This paper introduces sketch-guided constrained decoding (SGCD), a novel approach to constrained decoding for blackbox LLMs, which operates without access to the logits of the blackbox LLM. SGCD utilizes a locally hosted auxiliary model to refine the output of an unconstrained blackbox LLM, effectively treating this initial output as a "sketch" for further elaboration. This approach is complementary to traditional logit-based techniques and enables the application of constrained decoding in settings where full model transparency is unavailable. We demonstrate the efficacy of SGCD through experiments in closed information extraction and constituency parsing, showing how it enhances the utility and flexibility of blackbox LLMs for complex NLP tasks.

{{</citation>}}


### (43/105) MatSciRE: Leveraging Pointer Networks to Automate Entity and Relation Extraction for Material Science Knowledge-base Construction (Ankan Mullick et al., 2024)

{{<citation>}}

Ankan Mullick, Akash Ghosh, G Sai Chaitanya, Samir Ghui, Tapas Nayak, Seung-Cheol Lee, Satadeep Bhattacharjee, Pawan Goyal. (2024)  
**MatSciRE: Leveraging Pointer Networks to Automate Entity and Relation Extraction for Material Science Knowledge-base Construction**  

---
Primary Category: cs.CL  
Categories: cs-CE, cs-CL, cs-IR, cs.CL  
Keywords: Relation Extraction  
[Paper Link](http://arxiv.org/abs/2401.09839v1)  

---


**ABSTRACT**  
Material science literature is a rich source of factual information about various categories of entities (like materials and compositions) and various relations between these entities, such as conductivity, voltage, etc. Automatically extracting this information to generate a material science knowledge base is a challenging task. In this paper, we propose MatSciRE (Material Science Relation Extractor), a Pointer Network-based encoder-decoder framework, to jointly extract entities and relations from material science articles as a triplet ($entity1, relation, entity2$). Specifically, we target the battery materials and identify five relations to work on - conductivity, coulombic efficiency, capacity, voltage, and energy. Our proposed approach achieved a much better F1-score (0.771) than a previous attempt using ChemDataExtractor (0.716). The overall graphical framework of MatSciRE is shown in Fig 1. The material information is extracted from material science literature in the form of entity-relation triplets using MatSciRE.

{{</citation>}}


### (44/105) All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks (Kazuhiro Takemoto, 2024)

{{<citation>}}

Kazuhiro Takemoto. (2024)  
**All in How You Ask for It: Simple Black-Box Method for Jailbreak Attacks**  

---
Primary Category: cs.CL  
Categories: cs-AI, cs-CL, cs-CY, cs.CL  
Keywords: ChatGPT, GPT, GPT-3.5, GPT-4, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09798v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) like ChatGPT face `jailbreak' challenges, where safeguards are bypassed to produce ethically harmful prompts. This study introduces a simple black-box method to effectively generate jailbreak prompts, overcoming the limitations of high complexity and computational costs associated with existing methods. The proposed technique iteratively rewrites harmful prompts into non-harmful expressions using the target LLM itself, based on the hypothesis that LLMs can directly sample safeguard-bypassing expressions. Demonstrated through experiments with ChatGPT (GPT-3.5 and GPT-4) and Gemini-Pro, this method achieved an attack success rate of over 80% within an average of 5 iterations and remained effective despite model updates. The jailbreak prompts generated were naturally-worded and concise, suggesting they are less detectable. The results indicate that creating effective jailbreak prompts is simpler than previously considered, and black-box jailbreak attacks pose a more serious security threat.

{{</citation>}}


### (45/105) Instant Answering in E-Commerce Buyer-Seller Messaging (Besnik Fetahu et al., 2024)

{{<citation>}}

Besnik Fetahu, Tejas Mehta, Qun Song, Nikhita Vedula, Oleg Rokhlenko, Shervin Malmasi. (2024)  
**Instant Answering in E-Commerce Buyer-Seller Messaging**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Question Answering  
[Paper Link](http://arxiv.org/abs/2401.09785v1)  

---


**ABSTRACT**  
E-commerce customers frequently seek detailed product information for purchase decisions, commonly contacting sellers directly with extended queries. This manual response requirement imposes additional costs and disrupts buyer's shopping experience with response time fluctuations ranging from hours to days. We seek to automate buyer inquiries to sellers in a leading e-commerce store using a domain-specific federated Question Answering (QA) system. The main challenge is adapting current QA systems, designed for single questions, to address detailed customer queries. We address this with a low-latency, sequence-to-sequence approach, MESSAGE-TO-QUESTION ( M2Q ). It reformulates buyer messages into succinct questions by identifying and extracting the most salient information from a message. Evaluation against baselines shows that M2Q yields relative increases of 757% in question understanding, and 1,746% in answering rate from the federated QA system. Live deployment shows that automatic answering saves sellers from manually responding to millions of messages per year, and also accelerates customer purchase decisions by eliminating the need for buyers to wait for a reply

{{</citation>}}


### (46/105) Leveraging Biases in Large Language Models: 'bias-kNN'' for Effective Few-Shot Learning (Yong Zhang et al., 2024)

{{<citation>}}

Yong Zhang, Hanzhang Li, Zhitao Li, Ning Cheng, Ming Li, Jing Xiao, Jianzong Wang. (2024)  
**Leveraging Biases in Large Language Models: 'bias-kNN'' for Effective Few-Shot Learning**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: Bias, Few-Shot, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09783v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have shown significant promise in various applications, including zero-shot and few-shot learning. However, their performance can be hampered by inherent biases. Instead of traditionally sought methods that aim to minimize or correct these biases, this study introduces a novel methodology named ``bias-kNN''. This approach capitalizes on the biased outputs, harnessing them as primary features for kNN and supplementing with gold labels. Our comprehensive evaluations, spanning diverse domain text classification datasets and different GPT-2 model sizes, indicate the adaptability and efficacy of the ``bias-kNN'' method. Remarkably, this approach not only outperforms conventional in-context learning in few-shot scenarios but also demonstrates robustness across a spectrum of samples, templates and verbalizers. This study, therefore, presents a unique perspective on harnessing biases, transforming them into assets for enhanced model performance.

{{</citation>}}


### (47/105) Controllable Decontextualization of Yes/No Question and Answers into Factual Statements (Lingbo Mo et al., 2024)

{{<citation>}}

Lingbo Mo, Besnik Fetahu, Oleg Rokhlenko, Shervin Malmasi. (2024)  
**Controllable Decontextualization of Yes/No Question and Answers into Factual Statements**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs.CL  
Keywords: QA, Transformer  
[Paper Link](http://arxiv.org/abs/2401.09775v1)  

---


**ABSTRACT**  
Yes/No or polar questions represent one of the main linguistic question categories. They consist of a main interrogative clause, for which the answer is binary (assertion or negation). Polar questions and answers (PQA) represent a valuable knowledge resource present in many community and other curated QA sources, such as forums or e-commerce applications. Using answers to polar questions alone in other contexts is not trivial. Answers are contextualized, and presume that the interrogative question clause and any shared knowledge between the asker and answerer are provided.   We address the problem of controllable rewriting of answers to polar questions into decontextualized and succinct factual statements. We propose a Transformer sequence to sequence model that utilizes soft-constraints to ensure controllable rewriting, such that the output statement is semantically equivalent to its PQA input. Evaluation on three separate PQA datasets as measured through automated and human evaluation metrics show that our proposed approach achieves the best performance when compared to existing baselines.

{{</citation>}}


### (48/105) A Comparative Study on Annotation Quality of Crowdsourcing and LLM via Label Aggregation (Jiyi Li, 2024)

{{<citation>}}

Jiyi Li. (2024)  
**A Comparative Study on Annotation Quality of Crowdsourcing and LLM via Label Aggregation**  

---
Primary Category: cs.CL  
Categories: cs-CL, cs-HC, cs.CL  
Keywords: Language Model, NLP  
[Paper Link](http://arxiv.org/abs/2401.09760v1)  

---


**ABSTRACT**  
Whether Large Language Models (LLMs) can outperform crowdsourcing on the data annotation task is attracting interest recently. Some works verified this issue with the average performance of individual crowd workers and LLM workers on some specific NLP tasks by collecting new datasets. However, on the one hand, existing datasets for the studies of annotation quality in crowdsourcing are not yet utilized in such evaluations, which potentially provide reliable evaluations from a different viewpoint. On the other hand, the quality of these aggregated labels is crucial because, when utilizing crowdsourcing, the estimated labels aggregated from multiple crowd labels to the same instances are the eventually collected labels. Therefore, in this paper, we first investigate which existing crowdsourcing datasets can be used for a comparative study and create a benchmark. We then compare the quality between individual crowd labels and LLM labels and make the evaluations on the aggregated labels. In addition, we propose a Crowd-LLM hybrid label aggregation method and verify the performance. We find that adding LLM labels from good LLMs to existing crowdsourcing datasets can enhance the quality of the aggregated labels of the datasets, which is also higher than the quality of LLM labels themselves.

{{</citation>}}


### (49/105) Curriculum Recommendations Using Transformer Base Model with InfoNCE Loss And Language Switching Method (Xiaonan Xu et al., 2024)

{{<citation>}}

Xiaonan Xu, Bin Yuan, Yongyao Mo, Tianbo Song, Shulin Li. (2024)  
**Curriculum Recommendations Using Transformer Base Model with InfoNCE Loss And Language Switching Method**  

---
Primary Category: cs.CL  
Categories: 68T50, cs-AI, cs-CL, cs.CL  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.09699v1)  

---


**ABSTRACT**  
The Curriculum Recommendations paradigm is dedicated to fostering learning equality within the ever-evolving realms of educational technology and curriculum development. In acknowledging the inherent obstacles posed by existing methodologies, such as content conflicts and disruptions from language translation, this paradigm aims to confront and overcome these challenges. Notably, it addresses content conflicts and disruptions introduced by language translation, hindrances that can impede the creation of an all-encompassing and personalized learning experience. The paradigm's objective is to cultivate an educational environment that not only embraces diversity but also customizes learning experiences to suit the distinct needs of each learner. To overcome these challenges, our approach builds upon notable contributions in curriculum development and personalized learning, introducing three key innovations. These include the integration of Transformer Base Model to enhance computational efficiency, the implementation of InfoNCE Loss for accurate content-topic matching, and the adoption of a language switching strategy to alleviate translation-related ambiguities. Together, these innovations aim to collectively tackle inherent challenges and contribute to forging a more equitable and effective learning journey for a diverse range of learners. Competitive cross-validation scores underscore the efficacy of sentence-transformers/LaBSE, achieving 0.66314, showcasing our methodology's effectiveness in diverse linguistic nuances for content alignment prediction. Index Terms-Curriculum Recommendation, Transformer model with InfoNCE Loss, Language Switching.

{{</citation>}}


## cs.NI (1)



### (50/105) Tailoring Semantic Communication at Network Edge: A Novel Approach Using Dynamic Knowledge Distillation (Abdullatif Albaseer et al., 2024)

{{<citation>}}

Abdullatif Albaseer, Mohamed Abdallah. (2024)  
**Tailoring Semantic Communication at Network Edge: A Novel Approach Using Dynamic Knowledge Distillation**  

---
Primary Category: cs.NI  
Categories: cs-NI, cs.NI  
Keywords: Knowledge Distillation  
[Paper Link](http://arxiv.org/abs/2401.10214v1)  

---


**ABSTRACT**  
Semantic Communication (SemCom) systems, empowered by deep learning (DL), represent a paradigm shift in data transmission. These systems prioritize the significance of content over sheer data volume. However, existing SemCom designs face challenges when applied to diverse computational capabilities and network conditions, particularly in time-sensitive applications. A key challenge is the assumption that diverse devices can uniformly benefit from a standard, large DL model in SemCom systems. This assumption becomes increasingly impractical, especially in high-speed, high-reliability applications such as industrial automation or critical healthcare. Therefore, this paper introduces a novel SemCom framework tailored for heterogeneous, resource-constrained edge devices and computation-intensive servers. Our approach employs dynamic knowledge distillation (KD) to customize semantic models for each device, balancing computational and communication constraints while ensuring Quality of Service (QoS). We formulate an optimization problem and develop an adaptive algorithm that iteratively refines semantic knowledge on edge devices, resulting in better models tailored to their resource profiles. This algorithm strategically adjusts the granularity of distilled knowledge, enabling devices to maintain high semantic accuracy for precise inference tasks, even under unstable network conditions. Extensive simulations demonstrate that our approach significantly reduces model complexity for edge devices, leading to better semantic extraction and achieving the desired QoS.

{{</citation>}}


## cs.CR (4)



### (51/105) Eclectic Rule Extraction for Explainability of Deep Neural Network based Intrusion Detection Systems (Jesse Ables et al., 2024)

{{<citation>}}

Jesse Ables, Nathaniel Childers, William Anderson, Sudip Mittal, Shahram Rahimi, Ioana Banicescu, Maria Seale. (2024)  
**Eclectic Rule Extraction for Explainability of Deep Neural Network based Intrusion Detection Systems**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-LG, cs.CR  
Keywords: AI, Intrusion Detection  
[Paper Link](http://arxiv.org/abs/2401.10207v1)  

---


**ABSTRACT**  
This paper addresses trust issues created from the ubiquity of black box algorithms and surrogate explainers in Explainable Intrusion Detection Systems (X-IDS). While Explainable Artificial Intelligence (XAI) aims to enhance transparency, black box surrogate explainers, such as Local Interpretable Model-Agnostic Explanation (LIME) and SHapley Additive exPlanation (SHAP), are difficult to trust. The black box nature of these surrogate explainers makes the process behind explanation generation opaque and difficult to understand. To avoid this problem, one can use transparent white box algorithms such as Rule Extraction (RE). There are three types of RE algorithms: pedagogical, decompositional, and eclectic. Pedagogical methods offer fast but untrustworthy white-box explanations, while decompositional RE provides trustworthy explanations with poor scalability. This work explores eclectic rule extraction, which strikes a balance between scalability and trustworthiness. By combining techniques from pedagogical and decompositional approaches, eclectic rule extraction leverages the advantages of both, while mitigating some of their drawbacks. The proposed Hybrid X-IDS architecture features eclectic RE as a white box surrogate explainer for black box Deep Neural Networks (DNN). The presented eclectic RE algorithm extracts human-readable rules from hidden layers, facilitating explainable and trustworthy rulesets. Evaluations on UNSW-NB15 and CIC-IDS-2017 datasets demonstrate the algorithm's ability to generate rulesets with 99.9% accuracy, mimicking DNN outputs. The contributions of this work include the hybrid X-IDS architecture, the eclectic rule extraction algorithm applicable to intrusion detection datasets, and a thorough analysis of performance and explainability, demonstrating the trade-offs involved in rule extraction speed and accuracy.

{{</citation>}}


### (52/105) LOCALINTEL: Generating Organizational Threat Intelligence from Global and Local Cyber Knowledge (Shaswata Mitra et al., 2024)

{{<citation>}}

Shaswata Mitra, Subash Neupane, Trisha Chakraborty, Sudip Mittal, Aritran Piplai, Manas Gaur, Shahram Rahimi. (2024)  
**LOCALINTEL: Generating Organizational Threat Intelligence from Global and Local Cyber Knowledge**  

---
Primary Category: cs.CR  
Categories: cs-AI, cs-CR, cs-IR, cs-LO, cs.CR  
Keywords: Language Model, Security  
[Paper Link](http://arxiv.org/abs/2401.10036v1)  

---


**ABSTRACT**  
Security Operations Center (SoC) analysts gather threat reports from openly accessible global threat databases and customize them manually to suit a particular organization's needs. These analysts also depend on internal repositories, which act as private local knowledge database for an organization. Credible cyber intelligence, critical operational details, and relevant organizational information are all stored in these local knowledge databases. Analysts undertake a labor intensive task utilizing these global and local knowledge databases to manually create organization's unique threat response and mitigation strategies. Recently, Large Language Models (LLMs) have shown the capability to efficiently process large diverse knowledge sources. We leverage this ability to process global and local knowledge databases to automate the generation of organization-specific threat intelligence.   In this work, we present LOCALINTEL, a novel automated knowledge contextualization system that, upon prompting, retrieves threat reports from the global threat repositories and uses its local knowledge database to contextualize them for a specific organization. LOCALINTEL comprises of three key phases: global threat intelligence retrieval, local knowledge retrieval, and contextualized completion generation. The former retrieves intelligence from global threat repositories, while the second retrieves pertinent knowledge from the local knowledge database. Finally, the fusion of these knowledge sources is orchestrated through a generator to produce a contextualized completion.

{{</citation>}}


### (53/105) Conning the Crypto Conman: End-to-End Analysis of Cryptocurrency-based Technical Support Scams (Bhupendra Acharya et al., 2024)

{{<citation>}}

Bhupendra Acharya, Muhammad Saad, Antonio Emanuele Cinà, Lea Schönherr, Hoang Dai Nguyen, Adam Oest, Phani Vadrevu, Thorsten Holz. (2024)  
**Conning the Crypto Conman: End-to-End Analysis of Cryptocurrency-based Technical Support Scams**  

---
Primary Category: cs.CR  
Categories: cs-CR, cs.CR  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.09824v1)  

---


**ABSTRACT**  
The mainstream adoption of cryptocurrencies has led to a surge in wallet-related issues reported by ordinary users on social media platforms. In parallel, there is an increase in an emerging fraud trend called cryptocurrency-based technical support scam, in which fraudsters offer fake wallet recovery services and target users experiencing wallet-related issues.   In this paper, we perform a comprehensive study of cryptocurrency-based technical support scams. We present an analysis apparatus called HoneyTweet to analyze this kind of scam. Through HoneyTweet, we lure over 9K scammers by posting 25K fake wallet support tweets (so-called honey tweets). We then deploy automated systems to interact with scammers to analyze their modus operandi. In our experiments, we observe that scammers use Twitter as a starting point for the scam, after which they pivot to other communication channels (eg email, Instagram, or Telegram) to complete the fraud activity. We track scammers across those communication channels and bait them into revealing their payment methods. Based on the modes of payment, we uncover two categories of scammers that either request secret key phrase submissions from their victims or direct payments to their digital wallets. Furthermore, we obtain scam confirmation by deploying honey wallet addresses and validating private key theft. We also collaborate with the prominent payment service provider by sharing scammer data collections. The payment service provider feedback was consistent with our findings, thereby supporting our methodology and results. By consolidating our analysis across various vantage points, we provide an end-to-end scam lifecycle analysis and propose recommendations for scam mitigation.

{{</citation>}}


### (54/105) Large Language Model Lateral Spear Phishing: A Comparative Study in Large-Scale Organizational Settings (Mazal Bethany et al., 2024)

{{<citation>}}

Mazal Bethany, Athanasios Galiopoulos, Emet Bethany, Mohammad Bahrami Karkevandi, Nishant Vishwamitra, Peyman Najafirad. (2024)  
**Large Language Model Lateral Spear Phishing: A Comparative Study in Large-Scale Organizational Settings**  

---
Primary Category: cs.CR  
Categories: cs-CL, cs-CR, cs.CR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09727v1)  

---


**ABSTRACT**  
The critical threat of phishing emails has been further exacerbated by the potential of LLMs to generate highly targeted, personalized, and automated spear phishing attacks. Two critical problems concerning LLM-facilitated phishing require further investigation: 1) Existing studies on lateral phishing lack specific examination of LLM integration for large-scale attacks targeting the entire organization, and 2) Current anti-phishing infrastructure, despite its extensive development, lacks the capability to prevent LLM-generated attacks, potentially impacting both employees and IT security incident management. However, the execution of such investigative studies necessitates a real-world environment, one that functions during regular business operations and mirrors the complexity of a large organizational infrastructure. This setting must also offer the flexibility required to facilitate a diverse array of experimental conditions, particularly the incorporation of phishing emails crafted by LLMs. This study is a pioneering exploration into the use of Large Language Models (LLMs) for the creation of targeted lateral phishing emails, targeting a large tier 1 university's operation and workforce of approximately 9,000 individuals over an 11-month period. It also evaluates the capability of email filtering infrastructure to detect such LLM-generated phishing attempts, providing insights into their effectiveness and identifying potential areas for improvement. Based on our findings, we propose machine learning-based detection techniques for such emails to detect LLM-generated phishing emails that were missed by the existing infrastructure, with an F1-score of 98.96.

{{</citation>}}


## cs.LG (20)



### (55/105) Multi-Agent Reinforcement Learning for Maritime Operational Technology Cyber Security (Alec Wilson et al., 2024)

{{<citation>}}

Alec Wilson, Ryan Menzies, Neela Morarji, David Foster, Marco Casassa Mont, Esin Turkbeyler, Lisa Gralewski. (2024)  
**Multi-Agent Reinforcement Learning for Maritime Operational Technology Cyber Security**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs-MA, cs.LG  
Keywords: Cyber Security, Reinforcement Learning, Security  
[Paper Link](http://arxiv.org/abs/2401.10149v1)  

---


**ABSTRACT**  
This paper demonstrates the potential for autonomous cyber defence to be applied on industrial control systems and provides a baseline environment to further explore Multi-Agent Reinforcement Learning's (MARL) application to this problem domain. It introduces a simulation environment, IPMSRL, of a generic Integrated Platform Management System (IPMS) and explores the use of MARL for autonomous cyber defence decision-making on generic maritime based IPMS Operational Technology (OT). OT cyber defensive actions are less mature than they are for Enterprise IT. This is due to the relatively brittle nature of OT infrastructure originating from the use of legacy systems, design-time engineering assumptions, and lack of full-scale modern security controls. There are many obstacles to be tackled across the cyber landscape due to continually increasing cyber-attack sophistication and the limitations of traditional IT-centric cyber defence solutions. Traditional IT controls are rarely deployed on OT infrastructure, and where they are, some threats aren't fully addressed. In our experiments, a shared critic implementation of Multi Agent Proximal Policy Optimisation (MAPPO) outperformed Independent Proximal Policy Optimisation (IPPO). MAPPO reached an optimal policy (episode outcome mean of 1) after 800K timesteps, whereas IPPO was only able to reach an episode outcome mean of 0.966 after one million timesteps. Hyperparameter tuning greatly improved training performance. Across one million timesteps the tuned hyperparameters reached an optimal policy whereas the default hyperparameters only managed to win sporadically, with most simulations resulting in a draw. We tested a real-world constraint, attack detection alert success, and found that when alert success probability is reduced to 0.75 or 0.9, the MARL defenders were still able to win in over 97.5% or 99.5% of episodes, respectively.

{{</citation>}}


### (56/105) Spatial-Temporal Large Language Model for Traffic Prediction (Chenxi Liu et al., 2024)

{{<citation>}}

Chenxi Liu, Sun Yang, Qianxiong Xu, Zhishuai Li, Cheng Long, Ziyue Li, Rui Zhao. (2024)  
**Spatial-Temporal Large Language Model for Traffic Prediction**  

---
Primary Category: cs.LG  
Categories: cs-CL, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.10134v1)  

---


**ABSTRACT**  
Traffic prediction, a critical component for intelligent transportation systems, endeavors to foresee future traffic at specific locations using historical data. Although existing traffic prediction models often emphasize developing complex neural network structures, their accuracy has not seen improvements accordingly. Recently, Large Language Models (LLMs) have shown outstanding capabilities in time series analysis. Differing from existing models, LLMs progress mainly through parameter expansion and extensive pre-training while maintaining their fundamental structures. In this paper, we propose a Spatial-Temporal Large Language Model (ST-LLM) for traffic prediction. Specifically, ST-LLM redefines the timesteps at each location as tokens and incorporates a spatial-temporal embedding module to learn the spatial location and global temporal representations of tokens. Then these representations are fused to provide each token with unified spatial and temporal information. Furthermore, we propose a novel partially frozen attention strategy of the LLM, which is designed to capture spatial-temporal dependencies for traffic prediction. Comprehensive experiments on real traffic datasets offer evidence that ST-LLM outperforms state-of-the-art models. Notably, the ST-LLM also exhibits robust performance in both few-shot and zero-shot prediction scenarios.

{{</citation>}}


### (57/105) Towards Principled Graph Transformers (Luis Müller et al., 2024)

{{<citation>}}

Luis Müller, Christopher Morris. (2024)  
**Towards Principled Graph Transformers**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.10119v1)  

---


**ABSTRACT**  
Graph learning architectures based on the k-dimensional Weisfeiler-Leman (k-WL) hierarchy offer a theoretically well-understood expressive power. However, such architectures often fail to deliver solid predictive performance on real-world tasks, limiting their practical impact. In contrast, global attention-based models such as graph transformers demonstrate strong performance in practice, but comparing their expressive power with the k-WL hierarchy remains challenging, particularly since these architectures rely on positional or structural encodings for their expressivity and predictive performance. To address this, we show that the recently proposed Edge Transformer, a global attention model operating on node pairs instead of nodes, has at least 3-WL expressive power. Empirically, we demonstrate that the Edge Transformer surpasses other theoretically aligned architectures regarding predictive performance while not relying on positional or structural encodings.

{{</citation>}}


### (58/105) Optimizing Medication Decisions for Patients with Atrial Fibrillation through Path Development Network (Tian Xie, 2024)

{{<citation>}}

Tian Xie. (2024)  
**Optimizing Medication Decisions for Patients with Atrial Fibrillation through Path Development Network**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, eess-SP  
Keywords: LSTM  
[Paper Link](http://arxiv.org/abs/2401.10014v1)  

---


**ABSTRACT**  
Atrial fibrillation (AF) is a common cardiac arrhythmia characterized by rapid and irregular contractions of the atria. It significantly elevates the risk of strokes due to slowed blood flow in the atria, especially in the left atrial appendage, which is prone to blood clot formation. Such clots can migrate into cerebral arteries, leading to ischemic stroke. To assess whether AF patients should be prescribed anticoagulants, doctors often use the CHA2DS2-VASc scoring system. However, anticoagulant use must be approached with caution as it can impact clotting functions. This study introduces a machine learning algorithm that predicts whether patients with AF should be recommended anticoagulant therapy using 12-lead ECG data. In this model, we use STOME to enhance time-series data and then process it through a Convolutional Neural Network (CNN). By incorporating a path development layer, the model achieves a specificity of 30.6% under the condition of an NPV of 1. In contrast, LSTM algorithms without path development yield a specificity of only 2.7% under the same NPV condition.

{{</citation>}}


### (59/105) Developing an AI-based Integrated System for Bee Health Evaluation (Andrew Liang, 2024)

{{<citation>}}

Andrew Liang. (2024)  
**Developing an AI-based Integrated System for Bee Health Evaluation**  

---
Primary Category: cs.LG  
Categories: cs-CV, cs-LG, cs-SD, cs.LG, eess-AS  
Keywords: AI, Attention  
[Paper Link](http://arxiv.org/abs/2401.09988v1)  

---


**ABSTRACT**  
Honey bees pollinate about one-third of the world's food supply, but bee colonies have alarmingly declined by nearly 40% over the past decade due to several factors, including pesticides and pests. Traditional methods for monitoring beehives, such as human inspection, are subjective, disruptive, and time-consuming. To overcome these limitations, artificial intelligence has been used to assess beehive health. However, previous studies have lacked an end-to-end solution and primarily relied on data from a single source, either bee images or sounds. This study introduces a comprehensive system consisting of bee object detection and health evaluation. Additionally, it utilized a combination of visual and audio signals to analyze bee behaviors. An Attention-based Multimodal Neural Network (AMNN) was developed to adaptively focus on key features from each type of signal for accurate bee health assessment. The AMNN achieved an overall accuracy of 92.61%, surpassing eight existing single-signal Convolutional Neural Networks and Recurrent Neural Networks. It outperformed the best image-based model by 32.51% and the top sound-based model by 13.98% while maintaining efficient processing times. Furthermore, it improved prediction robustness, attaining an F1-score higher than 90% across all four evaluated health conditions. The study also shows that audio signals are more reliable than images for assessing bee health. By seamlessly integrating AMNN with image and sound data in a comprehensive bee health monitoring system, this approach provides a more efficient and non-invasive solution for the early detection of bee diseases and the preservation of bee colonies.

{{</citation>}}


### (60/105) Through the Dual-Prism: A Spectral Perspective on Graph Data Augmentation for Graph Classification (Yutong Xia et al., 2024)

{{<citation>}}

Yutong Xia, Runpeng Yu, Yuxuan Liang, Xavier Bresson, Xinchao Wang, Roger Zimmermann. (2024)  
**Through the Dual-Prism: A Spectral Perspective on Graph Data Augmentation for Graph Classification**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Augmentation, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.09953v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have become the preferred tool to process graph data, with their efficacy being boosted through graph data augmentation techniques. Despite the evolution of augmentation methods, issues like graph property distortions and restricted structural changes persist. This leads to the question: Is it possible to develop more property-conserving and structure-sensitive augmentation methods? Through a spectral lens, we investigate the interplay between graph properties, their augmentation, and their spectral behavior, and found that keeping the low-frequency eigenvalues unchanged can preserve the critical properties at a large scale when generating augmented graphs. These observations inform our introduction of the Dual-Prism (DP) augmentation method, comprising DP-Noise and DP-Mask, which adeptly retains essential graph properties while diversifying augmented graphs. Extensive experiments validate the efficiency of our approach, providing a new and promising direction for graph data augmentation.

{{</citation>}}


### (61/105) SymbolNet: Neural Symbolic Regression with Adaptive Dynamic Pruning (Ho Fung Tsoi et al., 2024)

{{<citation>}}

Ho Fung Tsoi, Vladimir Loncar, Sridhara Dasu, Philip Harris. (2024)  
**SymbolNet: Neural Symbolic Regression with Adaptive Dynamic Pruning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, hep-ex, physics-ins-det  
Keywords: Pruning  
[Paper Link](http://arxiv.org/abs/2401.09949v1)  

---


**ABSTRACT**  
Contrary to the use of genetic programming, the neural network approach to symbolic regression can scale well with high input dimension and leverage gradient methods for faster equation searching. Common ways of constraining expression complexity have relied on multistage pruning methods with fine-tuning, but these often lead to significant performance loss. In this work, we propose SymbolNet, a neural network approach to symbolic regression in a novel framework that enables dynamic pruning of model weights, input features, and mathematical operators in a single training, where both training loss and expression complexity are optimized simultaneously. We introduce a sparsity regularization term per pruning type, which can adaptively adjust its own strength and lead to convergence to a target sparsity level. In contrast to most existing symbolic regression methods that cannot efficiently handle datasets with more than $O$(10) inputs, we demonstrate the effectiveness of our model on the LHC jet tagging task (16 inputs), MNIST (784 inputs), and SVHN (3072 inputs).

{{</citation>}}


### (62/105) HGAttack: Transferable Heterogeneous Graph Adversarial Attack (He Zhao et al., 2024)

{{<citation>}}

He Zhao, Zhiwei Zeng, Yongwei Wang, Deheng Ye, Chunyan Miao. (2024)  
**HGAttack: Transferable Heterogeneous Graph Adversarial Attack**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-IR, cs-LG, cs.LG  
Keywords: Adversarial Attack, GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.09945v1)  

---


**ABSTRACT**  
Heterogeneous Graph Neural Networks (HGNNs) are increasingly recognized for their performance in areas like the web and e-commerce, where resilience against adversarial attacks is crucial. However, existing adversarial attack methods, which are primarily designed for homogeneous graphs, fall short when applied to HGNNs due to their limited ability to address the structural and semantic complexity of HGNNs. This paper introduces HGAttack, the first dedicated gray box evasion attack method for heterogeneous graphs. We design a novel surrogate model to closely resemble the behaviors of the target HGNN and utilize gradient-based methods for perturbation generation. Specifically, the proposed surrogate model effectively leverages heterogeneous information by extracting meta-path induced subgraphs and applying GNNs to learn node embeddings with distinct semantics from each subgraph. This approach improves the transferability of generated attacks on the target HGNN and significantly reduces memory costs. For perturbation generation, we introduce a semantics-aware mechanism that leverages subgraph gradient information to autonomously identify vulnerable edges across a wide range of relations within a constrained perturbation budget. We validate HGAttack's efficacy with comprehensive experiments on three datasets, providing empirical analyses of its generated perturbations. Outperforming baseline methods, HGAttack demonstrated significant efficacy in diminishing the performance of target HGNN models, affirming the effectiveness of our approach in evaluating the robustness of HGNNs against adversarial attacks.

{{</citation>}}


### (63/105) Infinite-Horizon Graph Filters: Leveraging Power Series to Enhance Sparse Information Aggregation (Ruizhe Zhang et al., 2024)

{{<citation>}}

Ruizhe Zhang, Xinke Jiang, Yuchen Fang, Jiayuan Luo, Yongxin Xu, Yichen Zhu, Xu Chu, Junfeng Zhao, Yasha Zhao. (2024)  
**Infinite-Horizon Graph Filters: Leveraging Power Series to Enhance Sparse Information Aggregation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.09943v1)  

---


**ABSTRACT**  
Graph Neural Networks (GNNs) have shown considerable effectiveness in a variety of graph learning tasks, particularly those based on the message-passing approach in recent years. However, their performance is often constrained by a limited receptive field, a challenge that becomes more acute in the presence of sparse graphs. In light of the power series, which possesses infinite expansion capabilities, we propose a novel \underline{G}raph \underline{P}ower \underline{F}ilter \underline{N}eural Network (GPFN) that enhances node classification by employing a power series graph filter to augment the receptive field. Concretely, our GPFN designs a new way to build a graph filter with an infinite receptive field based on the convergence power series, which can be analyzed in the spectral and spatial domains. Besides, we theoretically prove that our GPFN is a general framework that can integrate any power series and capture long-range dependencies. Finally, experimental results on three datasets demonstrate the superiority of our GPFN over state-of-the-art baselines.

{{</citation>}}


### (64/105) Biases in Expected Goals Models Confound Finishing Ability (Jesse Davis et al., 2024)

{{<citation>}}

Jesse Davis, Pieter Robberechts. (2024)  
**Biases in Expected Goals Models Confound Finishing Ability**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-AP  
Keywords: AI, Bias  
[Paper Link](http://arxiv.org/abs/2401.09940v1)  

---


**ABSTRACT**  
Expected Goals (xG) has emerged as a popular tool for evaluating finishing skill in soccer analytics. It involves comparing a player's cumulative xG with their actual goal output, where consistent overperformance indicates strong finishing ability. However, the assessment of finishing skill in soccer using xG remains contentious due to players' difficulty in consistently outperforming their cumulative xG. In this paper, we aim to address the limitations and nuances surrounding the evaluation of finishing skill using xG statistics. Specifically, we explore three hypotheses: (1) the deviation between actual and expected goals is an inadequate metric due to the high variance of shot outcomes and limited sample sizes, (2) the inclusion of all shots in cumulative xG calculation may be inappropriate, and (3) xG models contain biases arising from interdependencies in the data that affect skill measurement. We found that sustained overperformance of cumulative xG requires both high shot volumes and exceptional finishing, including all shot types can obscure the finishing ability of proficient strikers, and that there is a persistent bias that makes the actual and expected goals closer for excellent finishers than it really is. Overall, our analysis indicates that we need more nuanced quantitative approaches for investigating a player's finishing ability, which we achieved using a technique from AI fairness to learn an xG model that is calibrated for multiple subgroups of players. As a concrete use case, we show that (1) the standard biased xG model underestimates Messi's GAX by 17% and (2) Messi's GAX is 27% higher than the typical elite high-shot-volume attacker, indicating that Messi is even a more exceptional finisher than people commonly believed.

{{</citation>}}


### (65/105) Cooperative Edge Caching Based on Elastic Federated and Multi-Agent Deep Reinforcement Learning in Next-Generation Network (Qiong Wu et al., 2024)

{{<citation>}}

Qiong Wu, Wenhua Wang, Pingyi Fan, Qiang Fan, Huiling Zhu, Khaled B. Letaief. (2024)  
**Cooperative Edge Caching Based on Elastic Federated and Multi-Agent Deep Reinforcement Learning in Next-Generation Network**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09886v1)  

---


**ABSTRACT**  
Edge caching is a promising solution for next-generation networks by empowering caching units in small-cell base stations (SBSs), which allows user equipments (UEs) to fetch users' requested contents that have been pre-cached in SBSs. It is crucial for SBSs to predict accurate popular contents through learning while protecting users' personal information. Traditional federated learning (FL) can protect users' privacy but the data discrepancies among UEs can lead to a degradation in model quality. Therefore, it is necessary to train personalized local models for each UE to predict popular contents accurately. In addition, the cached contents can be shared among adjacent SBSs in next-generation networks, thus caching predicted popular contents in different SBSs may affect the cost to fetch contents. Hence, it is critical to determine where the popular contents are cached cooperatively. To address these issues, we propose a cooperative edge caching scheme based on elastic federated and multi-agent deep reinforcement learning (CEFMR) to optimize the cost in the network. We first propose an elastic FL algorithm to train the personalized model for each UE, where adversarial autoencoder (AAE) model is adopted for training to improve the prediction accuracy, then {a popular} content prediction algorithm is proposed to predict the popular contents for each SBS based on the trained AAE model. Finally, we propose a multi-agent deep reinforcement learning (MADRL) based algorithm to decide where the predicted popular contents are collaboratively cached among SBSs. Our experimental results demonstrate the superiority of our proposed scheme to existing baseline caching schemes.

{{</citation>}}


### (66/105) GA-SmaAt-GNet: Generative Adversarial Small Attention GNet for Extreme Precipitation Nowcasting (Eloy Reulen et al., 2024)

{{<citation>}}

Eloy Reulen, Siamak Mehrkanoon. (2024)  
**GA-SmaAt-GNet: Generative Adversarial Small Attention GNet for Extreme Precipitation Nowcasting**  

---
Primary Category: cs.LG  
Categories: I-2; I-5, cs-LG, cs.LG, physics-ao-ph  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.09881v1)  

---


**ABSTRACT**  
In recent years, data-driven modeling approaches have gained considerable traction in various meteorological applications, particularly in the realm of weather forecasting. However, these approaches often encounter challenges when dealing with extreme weather conditions. In light of this, we propose GA-SmaAt-GNet, a novel generative adversarial architecture that makes use of two methodologies aimed at enhancing the performance of deep learning models for extreme precipitation nowcasting. Firstly, it uses a novel SmaAt-GNet built upon the successful SmaAt-UNet architecture as generator. This network incorporates precipitation masks (binarized precipitation maps) as an additional data source, leveraging valuable information for improved predictions. Additionally, GA-SmaAt-GNet utilizes an attention-augmented discriminator inspired by the well-established Pix2Pix architecture. Furthermore, we assess the performance of GA-SmaAt-GNet using real-life precipitation dataset from the Netherlands. Our experimental results reveal a notable improvement in both overall performance and for extreme precipitation events. Furthermore, we conduct uncertainty analysis on the proposed GA-SmaAt-GNet model as well as on the precipitation dataset, providing additional insights into the predictive capabilities of the model. Finally, we offer further insights into the predictions of our proposed model using Grad-CAM. This visual explanation technique generates activation heatmaps, illustrating areas of the input that are more activated for various parts of the network.

{{</citation>}}


### (67/105) Reconciling Spatial and Temporal Abstractions for Goal Representation (Mehdi Zadem et al., 2024)

{{<citation>}}

Mehdi Zadem, Sergio Mover, Sao Mai Nguyen. (2024)  
**Reconciling Spatial and Temporal Abstractions for Goal Representation**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09870v1)  

---


**ABSTRACT**  
Goal representation affects the performance of Hierarchical Reinforcement Learning (HRL) algorithms by decomposing the complex learning problem into easier subtasks. Recent studies show that representations that preserve temporally abstract environment dynamics are successful in solving difficult problems and provide theoretical guarantees for optimality. These methods however cannot scale to tasks where environment dynamics increase in complexity i.e. the temporally abstract transition relations depend on larger number of variables. On the other hand, other efforts have tried to use spatial abstraction to mitigate the previous issues. Their limitations include scalability to high dimensional environments and dependency on prior knowledge.   In this paper, we propose a novel three-layer HRL algorithm that introduces, at different levels of the hierarchy, both a spatial and a temporal goal abstraction. We provide a theoretical study of the regret bounds of the learned policies. We evaluate the approach on complex continuous control tasks, demonstrating the effectiveness of spatial and temporal abstractions learned by this approach.

{{</citation>}}


### (68/105) A Fast, Performant, Secure Distributed Training Framework For Large Language Model (Wei Huang et al., 2024)

{{<citation>}}

Wei Huang, Yinggui Wang, Anda Cheng, Aihui Zhou, Chaofan Yu, Lei Wang. (2024)  
**A Fast, Performant, Secure Distributed Training Framework For Large Language Model**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs.LG  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09796v1)  

---


**ABSTRACT**  
The distributed (federated) LLM is an important method for co-training the domain-specific LLM using siloed data. However, maliciously stealing model parameters and data from the server or client side has become an urgent problem to be solved. In this paper, we propose a secure distributed LLM based on model slicing. In this case, we deploy the Trusted Execution Environment (TEE) on both the client and server side, and put the fine-tuned structure (LoRA or embedding of P-tuning v2) into the TEE. Then, secure communication is executed in the TEE and general environments through lightweight encryption. In order to further reduce the equipment cost as well as increase the model performance and accuracy, we propose a split fine-tuning scheme. In particular, we split the LLM by layers and place the latter layers in a server-side TEE (the client does not need a TEE). We then combine the proposed Sparsification Parameter Fine-tuning (SPF) with the LoRA part to improve the accuracy of the downstream task. Numerous experiments have shown that our method guarantees accuracy while maintaining security.

{{</citation>}}


### (69/105) PatchAD: Patch-based MLP-Mixer for Time Series Anomaly Detection (Zhijie Zhong et al., 2024)

{{<citation>}}

Zhijie Zhong, Zhiwen Yu, Yiyuan Yang, Weizheng Wang, Kaixiang Yang. (2024)  
**PatchAD: Patch-based MLP-Mixer for Time Series Anomaly Detection**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Anomaly Detection, Time Series  
[Paper Link](http://arxiv.org/abs/2401.09793v1)  

---


**ABSTRACT**  
Anomaly detection stands as a crucial aspect of time series analysis, aiming to identify abnormal events in time series samples. The central challenge of this task lies in effectively learning the representations of normal and abnormal patterns in a label-lacking scenario. Previous research mostly relied on reconstruction-based approaches, restricting the representational abilities of the models. In addition, most of the current deep learning-based methods are not lightweight enough, which prompts us to design a more efficient framework for anomaly detection. In this study, we introduce PatchAD, a novel multi-scale patch-based MLP-Mixer architecture that leverages contrastive learning for representational extraction and anomaly detection. Specifically, PatchAD is composed of four distinct MLP Mixers, exclusively utilizing the MLP architecture for high efficiency and lightweight architecture. Additionally, we also innovatively crafted a dual project constraint module to mitigate potential model degradation. Comprehensive experiments demonstrate that PatchAD achieves state-of-the-art results across multiple real-world multivariate time series datasets. Our code is publicly available.\footnote{\url{https://github.com/EmorZz1G/PatchAD}}

{{</citation>}}


### (70/105) Querying Easily Flip-flopped Samples for Deep Active Learning (Seong Jin Cho et al., 2024)

{{<citation>}}

Seong Jin Cho, Gwangsu Kim, Junghyun Lee, Jinwoo Shin, Chang D. Yoo. (2024)  
**Querying Easily Flip-flopped Samples for Deep Active Learning**  

---
Primary Category: cs.LG  
Categories: cs-AI, cs-LG, cs.LG, stat-ML  
Keywords: Active Learning  
[Paper Link](http://arxiv.org/abs/2401.09787v1)  

---


**ABSTRACT**  
Active learning is a machine learning paradigm that aims to improve the performance of a model by strategically selecting and querying unlabeled data. One effective selection strategy is to base it on the model's predictive uncertainty, which can be interpreted as a measure of how informative a sample is. The sample's distance to the decision boundary is a natural measure of predictive uncertainty, but it is often intractable to compute, especially for complex decision boundaries formed in multiclass classification tasks. To address this issue, this paper proposes the {\it least disagree metric} (LDM), defined as the smallest probability of disagreement of the predicted label, and an estimator for LDM proven to be asymptotically consistent under mild assumptions. The estimator is computationally efficient and can be easily implemented for deep learning models using parameter perturbation. The LDM-based active learning is performed by querying unlabeled data with the smallest LDM. Experimental results show that our LDM-based active learning algorithm obtains state-of-the-art overall performance on all considered datasets and deep architectures.

{{</citation>}}


### (71/105) Universally Robust Graph Neural Networks by Preserving Neighbor Similarity (Yulin Zhu et al., 2024)

{{<citation>}}

Yulin Zhu, Yuni Lai, Xing Ai, Kai Zhou. (2024)  
**Universally Robust Graph Neural Networks by Preserving Neighbor Similarity**  

---
Primary Category: cs.LG  
Categories: cs-CR, cs-LG, cs-SI, cs.LG  
Keywords: GNN, Graph Neural Network, Graph Neural Networks  
[Paper Link](http://arxiv.org/abs/2401.09754v1)  

---


**ABSTRACT**  
Despite the tremendous success of graph neural networks in learning relational data, it has been widely investigated that graph neural networks are vulnerable to structural attacks on homophilic graphs. Motivated by this, a surge of robust models is crafted to enhance the adversarial robustness of graph neural networks on homophilic graphs. However, the vulnerability based on heterophilic graphs remains a mystery to us. To bridge this gap, in this paper, we start to explore the vulnerability of graph neural networks on heterophilic graphs and theoretically prove that the update of the negative classification loss is negatively correlated with the pairwise similarities based on the powered aggregated neighbor features. This theoretical proof explains the empirical observations that the graph attacker tends to connect dissimilar node pairs based on the similarities of neighbor features instead of ego features both on homophilic and heterophilic graphs. In this way, we novelly introduce a novel robust model termed NSPGNN which incorporates a dual-kNN graphs pipeline to supervise the neighbor similarity-guided propagation. This propagation utilizes the low-pass filter to smooth the features of node pairs along the positive kNN graphs and the high-pass filter to discriminate the features of node pairs along the negative kNN graphs. Extensive experiments on both homophilic and heterophilic graphs validate the universal robustness of NSPGNN compared to the state-of-the-art methods.

{{</citation>}}


### (72/105) Applications of Machine Learning to Optimizing Polyolefin Manufacturing (Niket Sharma et al., 2024)

{{<citation>}}

Niket Sharma, Y. A. Liu. (2024)  
**Applications of Machine Learning to Optimizing Polyolefin Manufacturing**  

---
Primary Category: cs.LG  
Categories: cs-CE, cs-LG, cs.LG  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09753v1)  

---


**ABSTRACT**  
This chapter is a preprint from our book by , focusing on leveraging machine learning (ML) in chemical and polyolefin manufacturing optimization. It's crafted for both novices and seasoned professionals keen on the latest ML applications in chemical processes. We trace the evolution of AI and ML in chemical industries, delineate core ML components, and provide resources for ML beginners. A detailed discussion on various ML methods is presented, covering regression, classification, and unsupervised learning techniques, with performance metrics and examples. Ensemble methods, deep learning networks, including MLP, DNNs, RNNs, CNNs, and transformers, are explored for their growing role in chemical applications. Practical workshops guide readers through predictive modeling using advanced ML algorithms. The chapter culminates with insights into science-guided ML, advocating for a hybrid approach that enhances model accuracy. The extensive bibliography offers resources for further research and practical implementation. This chapter aims to be a thorough primer on ML's practical application in chemical engineering, particularly for polyolefin production, and sets the stage for continued learning in subsequent chapters. Please cite the original work [169,170] when referencing.

{{</citation>}}


### (73/105) Exploration and Anti-Exploration with Distributional Random Network Distillation (Kai Yang et al., 2024)

{{<citation>}}

Kai Yang, Jian Tao, Jiafei Lyu, Xiu Li. (2024)  
**Exploration and Anti-Exploration with Distributional Random Network Distillation**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG  
Keywords: Network Distillation  
[Paper Link](http://arxiv.org/abs/2401.09750v1)  

---


**ABSTRACT**  
Exploration remains a critical issue in deep reinforcement learning for an agent to attain high returns in unknown environments. Although the prevailing exploration Random Network Distillation (RND) algorithm has been demonstrated to be effective in numerous environments, it often needs more discriminative power in bonus allocation. This paper highlights the ``bonus inconsistency'' issue within RND, pinpointing its primary limitation. To address this issue, we introduce the Distributional RND (DRND), a derivative of the RND. DRND enhances the exploration process by distilling a distribution of random networks and implicitly incorporating pseudo counts to improve the precision of bonus allocation. This refinement encourages agents to engage in more extensive exploration. Our method effectively mitigates the inconsistency issue without introducing significant computational overhead. Both theoretical analysis and experimental results demonstrate the superiority of our approach over the original RND algorithm. Our method excels in challenging online exploration scenarios and effectively serves as an anti-exploration mechanism in D4RL offline tasks.

{{</citation>}}


### (74/105) Harnessing Density Ratios for Online Reinforcement Learning (Philip Amortila et al., 2024)

{{<citation>}}

Philip Amortila, Dylan J. Foster, Nan Jiang, Ayush Sekhari, Tengyang Xie. (2024)  
**Harnessing Density Ratios for Online Reinforcement Learning**  

---
Primary Category: cs.LG  
Categories: cs-LG, cs.LG, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09681v1)  

---


**ABSTRACT**  
The theories of offline and online reinforcement learning, despite having evolved in parallel, have begun to show signs of the possibility for a unification, with algorithms and analysis techniques for one setting often having natural counterparts in the other. However, the notion of density ratio modeling, an emerging paradigm in offline RL, has been largely absent from online RL, perhaps for good reason: the very existence and boundedness of density ratios relies on access to an exploratory dataset with good coverage, but the core challenge in online RL is to collect such a dataset without having one to start. In this work we show -- perhaps surprisingly -- that density ratio-based algorithms have online counterparts. Assuming only the existence of an exploratory distribution with good coverage, a structural condition known as coverability (Xie et al., 2023), we give a new algorithm (GLOW) that uses density ratio realizability and value function realizability to perform sample-efficient online exploration. GLOW addresses unbounded density ratios via careful use of truncation, and combines this with optimism to guide exploration. GLOW is computationally inefficient; we complement it with a more efficient counterpart, HyGLOW, for the Hybrid RL setting (Song et al., 2022) wherein online RL is augmented with additional offline data. HyGLOW is derived as a special case of a more general meta-algorithm that provides a provable black-box reduction from hybrid RL to offline RL, which may be of independent interest.

{{</citation>}}


## stat.ME (1)



### (75/105) Lower Ricci Curvature for Efficient Community Detection (Yun Jin Park et al., 2024)

{{<citation>}}

Yun Jin Park, Didong Li. (2024)  
**Lower Ricci Curvature for Efficient Community Detection**  

---
Primary Category: stat.ME  
Categories: cs-SI, physics-soc-ph, stat-AP, stat-ME, stat.ME  
Keywords: Amazon  
[Paper Link](http://arxiv.org/abs/2401.10124v1)  

---


**ABSTRACT**  
This study introduces the Lower Ricci Curvature (LRC), a novel, scalable, and scale-free discrete curvature designed to enhance community detection in networks. Addressing the computational challenges posed by existing curvature-based methods, LRC offers a streamlined approach with linear computational complexity, making it well-suited for large-scale network analysis. We further develop an LRC-based preprocessing method that effectively augments popular community detection algorithms. Through comprehensive simulations and applications on real-world datasets, including the NCAA football league network, the DBLP collaboration network, the Amazon product co-purchasing network, and the YouTube social network, we demonstrate the efficacy of our method in significantly improving the performance of various community detection algorithms.

{{</citation>}}


## cs.AI (3)



### (76/105) Counterfactual Reasoning with Probabilistic Graphical Models for Analyzing Socioecological Systems (Rafael Cabañas et al., 2024)

{{<citation>}}

Rafael Cabañas, Ana D. Maldonado, María Morales, Pedro A. Aguilera, Antonio Salmerón. (2024)  
**Counterfactual Reasoning with Probabilistic Graphical Models for Analyzing Socioecological Systems**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI, math-PR, stat-AP  
Keywords: Reasoning  
[Paper Link](http://arxiv.org/abs/2401.10101v1)  

---


**ABSTRACT**  
Causal and counterfactual reasoning are emerging directions in data science that allow us to reason about hypothetical scenarios. This is particularly useful in domains where experimental data are usually not available. In the context of environmental and ecological sciences, causality enables us, for example, to predict how an ecosystem would respond to hypothetical interventions. A structural causal model is a class of probabilistic graphical models for causality, which, due to its intuitive nature, can be easily understood by experts in multiple fields. However, certain queries, called unidentifiable, cannot be calculated in an exact and precise manner. This paper proposes applying a novel and recent technique for bounding unidentifiable queries within the domain of socioecological systems. Our findings indicate that traditional statistical analysis, including probabilistic graphical models, can identify the influence between variables. However, such methods do not offer insights into the nature of the relationship, specifically whether it involves necessity or sufficiency. This is where counterfactual reasoning becomes valuable.

{{</citation>}}


### (77/105) Towards Generative Abstract Reasoning: Completing Raven's Progressive Matrix via Rule Abstraction and Selection (Fan Shi et al., 2024)

{{<citation>}}

Fan Shi, Bin Li, Xiangyang Xue. (2024)  
**Towards Generative Abstract Reasoning: Completing Raven's Progressive Matrix via Rule Abstraction and Selection**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs.AI  
Keywords: AI, Reasoning  
[Paper Link](http://arxiv.org/abs/2401.09966v1)  

---


**ABSTRACT**  
Endowing machines with abstract reasoning ability has been a long-term research topic in artificial intelligence. Raven's Progressive Matrix (RPM) is widely used to probe abstract visual reasoning in machine intelligence, where models need to understand the underlying rules and select the missing bottom-right images out of candidate sets to complete image matrices. The participators can display powerful reasoning ability by inferring the underlying attribute-changing rules and imagining the missing images at arbitrary positions. However, existing solvers can hardly manifest such an ability in realistic RPM problems. In this paper, we propose a conditional generative model to solve answer generation problems through Rule AbstractIon and SElection (RAISE) in the latent space. RAISE encodes image attributes as latent concepts and decomposes underlying rules into atomic rules by means of concepts, which are abstracted as global learnable parameters. When generating the answer, RAISE selects proper atomic rules out of the global knowledge set for each concept and composes them into the integrated rule of an RPM. In most configurations, RAISE outperforms the compared generative solvers in tasks of generating bottom-right and arbitrary-position answers. We test RAISE in the odd-one-out task and two held-out configurations to demonstrate how learning decoupled latent concepts and atomic rules helps find the image breaking the underlying rules and handle RPMs with unseen combinations of rules and attributes.

{{</citation>}}


### (78/105) Tiny Multi-Agent DRL for Twins Migration in UAV Metaverses: A Multi-Leader Multi-Follower Stackelberg Game Approach (Jiawen Kang et al., 2024)

{{<citation>}}

Jiawen Kang, Yue Zhong, Minrui Xu, Jiangtian Nie, Jinbo Wen, Hongyang Du, Dongdong Ye, Xumin Huang, Dusit Niyato, Shengli Xie. (2024)  
**Tiny Multi-Agent DRL for Twins Migration in UAV Metaverses: A Multi-Leader Multi-Follower Stackelberg Game Approach**  

---
Primary Category: cs.AI  
Categories: cs-AI, cs-GT, cs.AI  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09680v1)  

---


**ABSTRACT**  
The synergy between Unmanned Aerial Vehicles (UAVs) and metaverses is giving rise to an emerging paradigm named UAV metaverses, which create a unified ecosystem that blends physical and virtual spaces, transforming drone interaction and virtual exploration. UAV Twins (UTs), as the digital twins of UAVs that revolutionize UAV applications by making them more immersive, realistic, and informative, are deployed and updated on ground base stations, e.g., RoadSide Units (RSUs), to offer metaverse services for UAV Metaverse Users (UMUs). Due to the dynamic mobility of UAVs and limited communication coverages of RSUs, it is essential to perform real-time UT migration to ensure seamless immersive experiences for UMUs. However, selecting appropriate RSUs and optimizing the required bandwidth is challenging for achieving reliable and efficient UT migration. To address the challenges, we propose a tiny machine learning-based Stackelberg game framework based on pruning techniques for efficient UT migration in UAV metaverses. Specifically, we formulate a multi-leader multi-follower Stackelberg model considering a new immersion metric of UMUs in the utilities of UAVs. Then, we design a Tiny Multi-Agent Deep Reinforcement Learning (Tiny MADRL) algorithm to obtain the tiny networks representing the optimal game solution. Specifically, the actor-critic network leverages the pruning techniques to reduce the number of network parameters and achieve model size and computation reduction, allowing for efficient implementation of Tiny MADRL. Numerical results demonstrate that our proposed schemes have better performance than traditional schemes.

{{</citation>}}


## cs.NE (3)



### (79/105) Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap (Xingyu Wu et al., 2024)

{{<citation>}}

Xingyu Wu, Sheng-hao Wu, Jibin Wu, Liang Feng, Kay Chen Tan. (2024)  
**Evolutionary Computation in the Era of Large Language Model: Survey and Roadmap**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-CL, cs-NE, cs.NE  
Keywords: Language Model, Transformer  
[Paper Link](http://arxiv.org/abs/2401.10034v1)  

---


**ABSTRACT**  
Large Language Models (LLMs), built upon Transformer-based architectures with massive pretraining on diverse data, have not only revolutionized natural language processing but also extended their prowess to various domains, marking a significant stride towards artificial general intelligence. The interplay between LLMs and Evolutionary Algorithms (EAs), despite differing in objectives and methodologies, reveals intriguing parallels, especially in their shared optimization nature, black-box characteristics, and proficiency in handling complex problems. Meanwhile, EA can not only provide an optimization framework for LLM's further enhancement under black-box settings but also empower LLM with flexible global search and iterative mechanism in applications. On the other hand, LLM's abundant domain knowledge enables EA to perform smarter searches, while its text processing capability assist in deploying EA across various tasks. Based on their complementary advantages, this paper presents a comprehensive review and forward-looking roadmap, categorizing their mutual inspiration into LLM-enhanced evolutionary optimization and EA-enhanced LLM. Some integrated synergy methods are further introduced to exemplify the amalgamation of LLMs and EAs in various application scenarios, including neural architecture search, code generation, software engineering, and text generation. As the first comprehensive review specifically focused on the EA research in the era of LLMs, this paper provides a foundational stepping stone for understanding and harnessing the collaborative potential of LLMs and EAs. By presenting a comprehensive review, categorization, and critical analysis, we contribute to the ongoing discourse on the cross-disciplinary study of these two powerful paradigms. The identified challenges and future directions offer guidance to unlock the full potential of this innovative collaboration.

{{</citation>}}


### (80/105) Evolutionary Multi-Objective Optimization of Large Language Model Prompts for Balancing Sentiments (Jill Baumann et al., 2024)

{{<citation>}}

Jill Baumann, Oliver Kramer. (2024)  
**Evolutionary Multi-Objective Optimization of Large Language Model Prompts for Balancing Sentiments**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-CL, cs-LG, cs-NE, cs.NE  
Keywords: ChatGPT, GPT, Language Model  
[Paper Link](http://arxiv.org/abs/2401.09862v1)  

---


**ABSTRACT**  
The advent of large language models (LLMs) such as ChatGPT has attracted considerable attention in various domains due to their remarkable performance and versatility. As the use of these models continues to grow, the importance of effective prompt engineering has come to the fore. Prompt optimization emerges as a crucial challenge, as it has a direct impact on model performance and the extraction of relevant information. Recently, evolutionary algorithms (EAs) have shown promise in addressing this issue, paving the way for novel optimization strategies. In this work, we propose a evolutionary multi-objective (EMO) approach specifically tailored for prompt optimization called EMO-Prompts, using sentiment analysis as a case study. We use sentiment analysis capabilities as our experimental targets. Our results demonstrate that EMO-Prompts effectively generates prompts capable of guiding the LLM to produce texts embodying two conflicting emotions simultaneously.

{{</citation>}}


### (81/105) A Comparative Analysis on Metaheuristic Algorithms Based Vision Transformer Model for Early Detection of Alzheimer's Disease (Anuvab Sen et al., 2024)

{{<citation>}}

Anuvab Sen, Udayon Sen, Subhabrata Roy. (2024)  
**A Comparative Analysis on Metaheuristic Algorithms Based Vision Transformer Model for Early Detection of Alzheimer's Disease**  

---
Primary Category: cs.NE  
Categories: cs-AI, cs-NE, cs.NE  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.09795v1)  

---


**ABSTRACT**  
A number of life threatening neuro-degenerative disorders had degraded the quality of life for the older generation in particular. Dementia is one such symptom which may lead to a severe condition called Alzheimer's disease if not detected at an early stage. It has been reported that the progression of such disease from a normal stage is due to the change in several parameters inside the human brain. In this paper, an innovative metaheuristic algorithms based ViT model has been proposed for the identification of dementia at different stage. A sizeable number of test data have been utilized for the validation of the proposed scheme. It has also been demonstrated that our model exhibits superior performance in terms of accuracy, precision, recall as well as F1-score.

{{</citation>}}


## cs.RO (3)



### (82/105) A-KIT: Adaptive Kalman-Informed Transformer (Nadav Cohen et al., 2024)

{{<citation>}}

Nadav Cohen, Itzik Klein. (2024)  
**A-KIT: Adaptive Kalman-Informed Transformer**  

---
Primary Category: cs.RO  
Categories: cs-AI, cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Transformer  
[Paper Link](http://arxiv.org/abs/2401.09987v1)  

---


**ABSTRACT**  
The extended Kalman filter (EKF) is a widely adopted method for sensor fusion in navigation applications. A crucial aspect of the EKF is the online determination of the process noise covariance matrix reflecting the model uncertainty. While common EKF implementation assumes a constant process noise, in real-world scenarios, the process noise varies, leading to inaccuracies in the estimated state and potentially causing the filter to diverge. To cope with such situations, model-based adaptive EKF methods were proposed and demonstrated performance improvements, highlighting the need for a robust adaptive approach. In this paper, we derive and introduce A-KIT, an adaptive Kalman-informed transformer to learn the varying process noise covariance online. The A-KIT framework is applicable to any type of sensor fusion. Here, we present our approach to nonlinear sensor fusion based on an inertial navigation system and Doppler velocity log. By employing real recorded data from an autonomous underwater vehicle, we show that A-KIT outperforms the conventional EKF by more than 49.5% and model-based adaptive EKF by an average of 35.4% in terms of position accuracy.

{{</citation>}}


### (83/105) Robotic Test Tube Rearrangement Using Combined Reinforcement Learning and Motion Planning (Hao Chen et al., 2024)

{{<citation>}}

Hao Chen, Weiwei Wan, Masaki Matsushita, Takeyuki Kotaka, Kensuke Harada. (2024)  
**Robotic Test Tube Rearrangement Using Combined Reinforcement Learning and Motion Planning**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs.RO  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09772v1)  

---


**ABSTRACT**  
A combined task-level reinforcement learning and motion planning framework is proposed in this paper to address a multi-class in-rack test tube rearrangement problem. At the task level, the framework uses reinforcement learning to infer a sequence of swap actions while ignoring robotic motion details. At the motion level, the framework accepts the swapping action sequences inferred by task-level agents and plans the detailed robotic pick-and-place motion. The task and motion-level planning form a closed loop with the help of a condition set maintained for each rack slot, which allows the framework to perform replanning and effectively find solutions in the presence of low-level failures. Particularly for reinforcement learning, the framework leverages a distributed deep Q-learning structure with the Dueling Double Deep Q Network (D3QN) to acquire near-optimal policies and uses an A${}^\star$-based post-processing technique to amplify the collected training data. The D3QN and distributed learning help increase training efficiency. The post-processing helps complete unfinished action sequences and remove redundancy, thus making the training data more effective. We carry out both simulations and real-world studies to understand the performance of the proposed framework. The results verify the performance of the RL and post-processing and show that the closed-loop combination improves robustness. The framework is ready to incorporate various sensory feedback. The real-world studies also demonstrated the incorporation.

{{</citation>}}


### (84/105) Learning Hybrid Policies for MPC with Application to Drone Flight in Unknown Dynamic Environments (Zhaohan Feng et al., 2024)

{{<citation>}}

Zhaohan Feng, Jie Chen, Wei Xiao, Jian Sun, Bin Xin, Gang Wang. (2024)  
**Learning Hybrid Policies for MPC with Application to Drone Flight in Unknown Dynamic Environments**  

---
Primary Category: cs.RO  
Categories: cs-RO, cs-SY, cs.RO, eess-SY  
Keywords: Drone  
[Paper Link](http://arxiv.org/abs/2401.09705v1)  

---


**ABSTRACT**  
In recent years, drones have found increased applications in a wide array of real-world tasks. Model predictive control (MPC) has emerged as a practical method for drone flight control, owing to its robustness against modeling errors/uncertainties and external disturbances. However, MPC's sensitivity to manually tuned parameters can lead to rapid performance degradation when faced with unknown environmental dynamics. This paper addresses the challenge of controlling a drone as it traverses a swinging gate characterized by unknown dynamics. This paper introduces a parameterized MPC approach named hyMPC that leverages high-level decision variables to adapt to uncertain environmental conditions. To derive these decision variables, a novel policy search framework aimed at training a high-level Gaussian policy is presented. Subsequently, we harness the power of neural network policies, trained on data gathered through the repeated execution of the Gaussian policy, to provide real-time decision variables. The effectiveness of hyMPC is validated through numerical simulations, achieving a 100\% success rate in 20 drone flight tests traversing a swinging gate, demonstrating its capability to achieve safe and precise flight with limited prior knowledge of environmental dynamics.

{{</citation>}}


## cs.SE (2)



### (85/105) When Neural Code Completion Models Size up the Situation: Attaining Cheaper and Faster Completion through Dynamic Model Inference (Zhensu Sun et al., 2024)

{{<citation>}}

Zhensu Sun, Xiaoning Du, Fu Song, Shangwen Wang, Li Li. (2024)  
**When Neural Code Completion Models Size up the Situation: Attaining Cheaper and Faster Completion through Dynamic Model Inference**  

---
Primary Category: cs.SE  
Categories: cs-AI, cs-SE, cs.SE  
Keywords: GPT  
[Paper Link](http://arxiv.org/abs/2401.09964v1)  

---


**ABSTRACT**  
Leveraging recent advancements in large language models, modern neural code completion models have demonstrated the capability to generate highly accurate code suggestions. However, their massive size poses challenges in terms of computational costs and environmental impact, hindering their widespread adoption in practical scenarios. Dynamic inference emerges as a promising solution, as it allocates minimal computation during inference while maintaining the model's performance. In this research, we explore dynamic inference within the context of code completion. Initially, we conducted an empirical investigation on GPT-2, focusing on the inference capabilities of intermediate layers for code completion. We found that 54.4% of tokens can be accurately generated using just the first layer, signifying significant computational savings potential. Moreover, despite using all layers, the model still fails to predict 14.5% of tokens correctly, and the subsequent completions continued from them are rarely considered helpful, with only a 4.2% Acceptance Rate. These findings motivate our exploration of dynamic inference in code completion and inspire us to enhance it with a decision-making mechanism that stops the generation of incorrect code. We thus propose a novel dynamic inference method specifically tailored for code completion models. This method aims not only to produce correct predictions with largely reduced computation but also to prevent incorrect predictions proactively. Our extensive evaluation shows that it can averagely skip 1.7 layers out of 16 layers in the models, leading to an 11.2% speedup with only a marginal 1.1% reduction in ROUGE-L.

{{</citation>}}


### (86/105) SensoDat: Simulation-based Sensor Dataset of Self-driving Cars (Christian Birchler et al., 2024)

{{<citation>}}

Christian Birchler, Cyrill Rohrbach, Timo Kehrer, Sebastiano Panichella. (2024)  
**SensoDat: Simulation-based Sensor Dataset of Self-driving Cars**  

---
Primary Category: cs.SE  
Categories: cs-SE, cs.SE  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09808v1)  

---


**ABSTRACT**  
Developing tools in the context of autonomous systems [22, 24 ], such as self-driving cars (SDCs), is time-consuming and costly since researchers and practitioners rely on expensive computing hardware and simulation software. We propose SensoDat, a dataset of 32,580 executed simulation-based SDC test cases generated with state-of-the-art test generators for SDCs. The dataset consists of trajectory logs and a variety of sensor data from the SDCs (e.g., rpm, wheel speed, brake thermals, transmission, etc.) represented as a time series. In total, SensoDat provides data from 81 different simulated sensors. Future research in the domain of SDCs does not necessarily depend on executing expensive test cases when using SensoDat. Furthermore, with the high amount and variety of sensor data, we think SensoDat can contribute to research, particularly for AI development, regression testing techniques for simulation-based SDC testing, flakiness in simulation, etc. Link to the dataset: https://doi.org/10.5281/zenodo.10307479

{{</citation>}}


## cs.DC (3)



### (87/105) Deep Back-Filling: a Split Window Technique for Deep Online Cluster Job Scheduling (Lingfei Wang et al., 2024)

{{<citation>}}

Lingfei Wang, Aaron Harwood, Maria A. Rodriguez. (2024)  
**Deep Back-Filling: a Split Window Technique for Deep Online Cluster Job Scheduling**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09910v1)  

---


**ABSTRACT**  
Job scheduling is a critical component of workload management systems that can significantly influence system performance, e.g., in HPC clusters. The scheduling objectives are often mixed, such as maximizing resource utilization and minimizing job waiting time. An increasing number of researchers are moving from heuristic-based approaches to Deep Reinforcement Learning approaches in order to optimize scheduling objectives. However, the job scheduler's state space is partially observable to a DRL-based agent because the job queue is practically unbounded. The agent's observation of the state space is constant in size since the input size of the neural networks is predefined. All existing solutions to this problem intuitively allow the agent to observe a fixed window size of jobs at the head of the job queue. In our research, we have seen that such an approach can lead to "window staleness" where the window becomes full of jobs that can not be scheduled until the cluster has completed sufficient work. In this paper, we propose a novel general technique that we call \emph{split window}, which allows the agent to observe both the head \emph{and tail} of the queue. With this technique, the agent can observe all arriving jobs at least once, which completely eliminates the window staleness problem. By leveraging the split window, the agent can significantly reduce the average job waiting time and average queue length, alternatively allowing the use of much smaller windows and, therefore, faster training times. We show a range of simulation results using HPC job scheduling trace data that supports the effectiveness of our technique.

{{</citation>}}


### (88/105) A HPC Co-Scheduler with Reinforcement Learning (Abel Souza et al., 2024)

{{<citation>}}

Abel Souza, Kristiaan Pelckmans, Johan Tordsson. (2024)  
**A HPC Co-Scheduler with Reinforcement Learning**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09706v1)  

---


**ABSTRACT**  
Although High Performance Computing (HPC) users understand basic resource requirements such as the number of CPUs and memory limits, internal infrastructural utilization data is exclusively leveraged by cluster operators, who use it to configure batch schedulers. This task is challenging and increasingly complex due to ever larger cluster scales and heterogeneity of modern scientific workflows. As a result, HPC systems achieve low utilization with long job completion times (makespans). To tackle these challenges, we propose a co-scheduling algorithm based on an adaptive reinforcement learning algorithm, where application profiling is combined with cluster monitoring. The resulting cluster scheduler matches resource utilization to application performance in a fine-grained manner (i.e., operating system level). As opposed to nominal allocations, we apply decision trees to model applications' actual resource usage, which are used to estimate how much resource capacity from one allocation can be co-allocated to additional applications. Our algorithm learns from incorrect co-scheduling decisions and adapts from changing environment conditions, and evaluates when such changes cause resource contention that impacts quality of service metrics such as jobs slowdowns. We integrate our algorithm in an HPC resource manager that combines Slurm and Mesos for job scheduling and co-allocation, respectively. Our experimental evaluation performed in a dedicated cluster executing a mix of four real different scientific workflows demonstrates improvements on cluster utilization of up to 51% even in high load scenarios, with 55% average queue makespan reductions under low loads.

{{</citation>}}


### (89/105) DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving (Yinmin Zhong et al., 2024)

{{<citation>}}

Yinmin Zhong, Shengyu Liu, Junda Chen, Jianbo Hu, Yibo Zhu, Xuanzhe Liu, Xin Jin, Hao Zhang. (2024)  
**DistServe: Disaggregating Prefill and Decoding for Goodput-optimized Large Language Model Serving**  

---
Primary Category: cs.DC  
Categories: cs-DC, cs.DC  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09670v1)  

---


**ABSTRACT**  
DistServe improves the performance of large language models (LLMs) serving by disaggregating the prefill and decoding computation. Existing LLM serving systems colocate the two phases and batch the computation of prefill and decoding across all users and requests. We find that this strategy not only leads to strong prefill-decoding interferences but also couples the resource allocation and parallelism plans for both phases. LLM applications often emphasize individual latency for each phase: time to first token (TTFT) for the prefill phase and time per output token (TPOT) of each request for the decoding phase. In the presence of stringent latency requirements, existing systems have to prioritize one latency over the other, or over-provision compute resources to meet both.   DistServe assigns prefill and decoding computation to different GPUs, hence eliminating prefill-decoding interferences. Given the application's TTFT and TPOT requirements, DistServe co-optimizes the resource allocation and parallelism strategy tailored for each phase. DistServe also places the two phases according to the serving cluster's bandwidth to minimize the communication caused by disaggregation. As a result, DistServe significantly improves LLM serving performance in terms of the maximum rate that can be served within both TTFT and TPOT constraints on each GPU. Our evaluations show that on various popular LLMs, applications, and latency requirements, DistServe can serve 4.48x more requests or 10.2x tighter SLO, compared to state-of-the-art systems, while staying within latency constraints for > 90% of requests.

{{</citation>}}


## cs.AR (1)



### (90/105) A Survey on Hardware Accelerators for Large Language Models (Christoforos Kachris, 2024)

{{<citation>}}

Christoforos Kachris. (2024)  
**A Survey on Hardware Accelerators for Large Language Models**  

---
Primary Category: cs.AR  
Categories: B-5; C-1; C-3, cs-AR, cs-CL, cs-LG, cs.AR  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09890v1)  

---


**ABSTRACT**  
Large Language Models (LLMs) have emerged as powerful tools for natural language processing tasks, revolutionizing the field with their ability to understand and generate human-like text. As the demand for more sophisticated LLMs continues to grow, there is a pressing need to address the computational challenges associated with their scale and complexity. This paper presents a comprehensive survey on hardware accelerators designed to enhance the performance and energy efficiency of Large Language Models. By examining a diverse range of accelerators, including GPUs, FPGAs, and custom-designed architectures, we explore the landscape of hardware solutions tailored to meet the unique computational demands of LLMs. The survey encompasses an in-depth analysis of architecture, performance metrics, and energy efficiency considerations, providing valuable insights for researchers, engineers, and decision-makers aiming to optimize the deployment of LLMs in real-world applications.

{{</citation>}}


## cs.SD (2)



### (91/105) Attention-Based Recurrent Neural Network For Automatic Behavior Laying Hen Recognition (Fréjus A. A. Laleye et al., 2024)

{{<citation>}}

Fréjus A. A. Laleye, Mikaël A. Mousse. (2024)  
**Attention-Based Recurrent Neural Network For Automatic Behavior Laying Hen Recognition**  

---
Primary Category: cs.SD  
Categories: cs-AI, cs-CL, cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Attention  
[Paper Link](http://arxiv.org/abs/2401.09880v1)  

---


**ABSTRACT**  
One of the interests of modern poultry farming is the vocalization of laying hens which contain very useful information on health behavior. This information is used as health and well-being indicators that help breeders better monitor laying hens, which involves early detection of problems for rapid and more effective intervention. In this work, we focus on the sound analysis for the recognition of the types of calls of the laying hens in order to propose a robust system of characterization of their behavior for a better monitoring. To do this, we first collected and annotated laying hen call signals, then designed an optimal acoustic characterization based on the combination of time and frequency domain features. We then used these features to build the multi-label classification models based on recurrent neural network to assign a semantic class to the vocalization that characterize the laying hen behavior. The results show an overall performance with our model based on the combination of time and frequency domain features that obtained the highest F1-score (F1=92.75) with a gain of 17% on the models using the frequency domain features and of 8% on the compared approaches from the litterature.

{{</citation>}}


### (92/105) Improving Speaker-independent Speech Emotion Recognition Using Dynamic Joint Distribution Adaptation (Cheng Lu et al., 2024)

{{<citation>}}

Cheng Lu, Yuan Zong, Hailun Lian, Yan Zhao, Björn Schuller, Wenming Zheng. (2024)  
**Improving Speaker-independent Speech Emotion Recognition Using Dynamic Joint Distribution Adaptation**  

---
Primary Category: cs.SD  
Categories: cs-LG, cs-SD, cs.SD, eess-AS  
Keywords: Emotion Recognition  
[Paper Link](http://arxiv.org/abs/2401.09752v1)  

---


**ABSTRACT**  
In speaker-independent speech emotion recognition, the training and testing samples are collected from diverse speakers, leading to a multi-domain shift challenge across the feature distributions of data from different speakers. Consequently, when the trained model is confronted with data from new speakers, its performance tends to degrade. To address the issue, we propose a Dynamic Joint Distribution Adaptation (DJDA) method under the framework of multi-source domain adaptation. DJDA firstly utilizes joint distribution adaptation (JDA), involving marginal distribution adaptation (MDA) and conditional distribution adaptation (CDA), to more precisely measure the multi-domain distribution shifts caused by different speakers. This helps eliminate speaker bias in emotion features, allowing for learning discriminative and speaker-invariant speech emotion features from coarse-level to fine-level. Furthermore, we quantify the adaptation contributions of MDA and CDA within JDA by using a dynamic balance factor based on $\mathcal{A}$-Distance, promoting to effectively handle the unknown distributions encountered in data from new speakers. Experimental results demonstrate the superior performance of our DJDA as compared to other state-of-the-art (SOTA) methods.

{{</citation>}}


## cs.ET (1)



### (93/105) Improving the Accuracy of Analog-Based In-Memory Computing Accelerators Post-Training (Corey Lammie et al., 2024)

{{<citation>}}

Corey Lammie, Athanasios Vasilopoulos, Julian Büchel, Giacomo Camposampiero, Manuel Le Gallo, Malte Rasch, Abu Sebastian. (2024)  
**Improving the Accuracy of Analog-Based In-Memory Computing Accelerators Post-Training**  

---
Primary Category: cs.ET  
Categories: cs-ET, cs.ET  
Keywords: AI, BERT, GLUE  
[Paper Link](http://arxiv.org/abs/2401.09859v1)  

---


**ABSTRACT**  
Analog-Based In-Memory Computing (AIMC) inference accelerators can be used to efficiently execute Deep Neural Network (DNN) inference workloads. However, to mitigate accuracy losses, due to circuit and device non-idealities, Hardware-Aware (HWA) training methodologies must be employed. These typically require significant information about the underlying hardware. In this paper, we propose two Post-Training (PT) optimization methods to improve accuracy after training is performed. For each crossbar, the first optimizes the conductance range of each column, and the second optimizes the input, i.e, Digital-to-Analog Converter (DAC), range. It is demonstrated that, when these methods are employed, the complexity during training, and the amount of information about the underlying hardware can be reduced, with no notable change in accuracy ($\leq$0.1%) when finetuning the pretrained RoBERTa transformer model for all General Language Understanding Evaluation (GLUE) benchmark tasks. Additionally, it is demonstrated that further optimizing learned parameters PT improves accuracy.

{{</citation>}}


## q-bio.BM (1)



### (94/105) FREED++: Improving RL Agents for Fragment-Based Molecule Generation by Thorough Reproduction (Alexander Telepov et al., 2024)

{{<citation>}}

Alexander Telepov, Artem Tsypin, Kuzma Khrabrov, Sergey Yakukhnov, Pavel Strashnov, Petr Zhilyaev, Egor Rumiantsev, Daniel Ezhov, Manvel Avetisian, Olga Popova, Artur Kadurin. (2024)  
**FREED++: Improving RL Agents for Fragment-Based Molecule Generation by Thorough Reproduction**  

---
Primary Category: q-bio.BM  
Categories: cs-LG, q-bio-BM, q-bio.BM, stat-ML  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09840v1)  

---


**ABSTRACT**  
A rational design of new therapeutic drugs aims to find a molecular structure with desired biological functionality, e.g., an ability to activate or suppress a specific protein via binding to it. Molecular docking is a common technique for evaluating protein-molecule interactions. Recently, Reinforcement Learning (RL) has emerged as a promising approach to generating molecules with the docking score (DS) as a reward. In this work, we reproduce, scrutinize and improve the recent RL model for molecule generation called FREED (arXiv:2110.01219). Extensive evaluation of the proposed method reveals several limitations and challenges despite the outstanding results reported for three target proteins. Our contributions include fixing numerous implementation bugs and simplifying the model while increasing its quality, significantly extending experiments, and conducting an accurate comparison with current state-of-the-art methods for protein-conditioned molecule generation. We show that the resulting fixed model is capable of producing molecules with superior docking scores compared to alternative approaches.

{{</citation>}}


## cs.GT (1)



### (95/105) Clickbait vs. Quality: How Engagement-Based Optimization Shapes the Content Landscape in Online Platforms (Nicole Immorlica et al., 2024)

{{<citation>}}

Nicole Immorlica, Meena Jagadeesan, Brendan Lucier. (2024)  
**Clickbait vs. Quality: How Engagement-Based Optimization Shapes the Content Landscape in Online Platforms**  

---
Primary Category: cs.GT  
Categories: cs-CY, cs-GT, cs-LG, cs.GT  
Keywords: Twitter  
[Paper Link](http://arxiv.org/abs/2401.09804v1)  

---


**ABSTRACT**  
Online content platforms commonly use engagement-based optimization when making recommendations. This encourages content creators to invest in quality, but also rewards gaming tricks such as clickbait. To understand the total impact on the content landscape, we study a game between content creators competing on the basis of engagement metrics and analyze the equilibrium decisions about investment in quality and gaming. First, we show the content created at equilibrium exhibits a positive correlation between quality and gaming, and we empirically validate this finding on a Twitter dataset. Using the equilibrium structure of the content landscape, we then examine the downstream performance of engagement-based optimization along several axes. Perhaps counterintuitively, the average quality of content consumed by users can decrease at equilibrium as gaming tricks become more costly for content creators to employ. Moreover, engagement-based optimization can perform worse in terms of user utility than a baseline with random recommendations, and engagement-based optimization is also suboptimal in terms of realized engagement relative to quality-based optimization. Altogether, our results highlight the need to consider content creator incentives when evaluating a platform's choice of optimization metric.

{{</citation>}}


## eess.AS (2)



### (96/105) Multilingual Visual Speech Recognition with a Single Model by Learning with Discrete Visual Speech Units (Minsu Kim et al., 2024)

{{<citation>}}

Minsu Kim, Jeong Hun Yeo, Jeongsoo Choi, Se Jin Park, Yong Man Ro. (2024)  
**Multilingual Visual Speech Recognition with a Single Model by Learning with Discrete Visual Speech Units**  

---
Primary Category: eess.AS  
Categories: cs-CV, cs-SD, eess-AS, eess.AS  
Keywords: Multilingual, Speech Recognition  
[Paper Link](http://arxiv.org/abs/2401.09802v1)  

---


**ABSTRACT**  
This paper explores sentence-level Multilingual Visual Speech Recognition with a single model for the first time. As the massive multilingual modeling of visual data requires huge computational costs, we propose a novel strategy, processing with visual speech units. Motivated by the recent success of the audio speech unit, the proposed visual speech unit is obtained by discretizing the visual speech features extracted from the self-supervised visual speech model. To correctly capture multilingual visual speech, we first train the self-supervised visual speech model on 5,512 hours of multilingual audio-visual data. Through analysis, we verify that the visual speech units mainly contain viseme information while suppressing non-linguistic information. By using the visual speech units as the inputs of our system, we pre-train the model to predict corresponding text outputs on massive multilingual data constructed by merging several VSR databases. As both the inputs and outputs are discrete, we can greatly improve the training efficiency compared to the standard VSR training. Specifically, the input data size is reduced to 0.016% of the original video inputs. In order to complement the insufficient visual information in speech recognition, we apply curriculum learning where the inputs of the system begin with audio-visual speech units and gradually change to visual speech units. After pre-training, the model is finetuned on continuous features. We set new state-of-the-art multilingual VSR performances by achieving comparable performances to the previous language-specific VSR models, with a single trained model.

{{</citation>}}


### (97/105) An Empirical Study on the Impact of Positional Encoding in Transformer-based Monaural Speech Enhancement (Qiquan Zhang et al., 2024)

{{<citation>}}

Qiquan Zhang, Meng Ge, Hongxu Zhu, Eliathamby Ambikairajah, Qi Song, Zhaoheng Ni, Haizhou Li. (2024)  
**An Empirical Study on the Impact of Positional Encoding in Transformer-based Monaural Speech Enhancement**  

---
Primary Category: eess.AS  
Categories: cs-SD, eess-AS, eess.AS  
Keywords: T5, Transformer, Transformers  
[Paper Link](http://arxiv.org/abs/2401.09686v1)  

---


**ABSTRACT**  
Transformer architecture has enabled recent progress in speech enhancement. Since Transformers are position-agostic, positional encoding is the de facto standard component used to enable Transformers to distinguish the order of elements in a sequence. However, it remains unclear how positional encoding exactly impacts speech enhancement based on Transformer architectures. In this paper, we perform a comprehensive empirical study evaluating five positional encoding methods, i.e., Sinusoidal and learned absolute position embedding (APE), T5-RPE, KERPLE, as well as the Transformer without positional encoding (No-Pos), across both causal and noncausal configurations. We conduct extensive speech enhancement experiments, involving spectral mapping and masking methods. Our findings establish that positional encoding is not quite helpful for the models in a causal configuration, which indicates that causal attention may implicitly incorporate position information. In a noncausal configuration, the models significantly benefit from the use of positional encoding. In addition, we find that among the four position embeddings, relative position embeddings outperform APEs.

{{</citation>}}


## eess.SY (2)



### (98/105) Power System Fault Diagnosis with Quantum Computing and Efficient Gate Decomposition (Xiang Fei et al., 2024)

{{<citation>}}

Xiang Fei, Huan Zhao, Xiyuan Zhou, Junhua Zhao, Ting Shu, Fushuan Wen. (2024)  
**Power System Fault Diagnosis with Quantum Computing and Efficient Gate Decomposition**  

---
Primary Category: eess.SY  
Categories: cs-SY, eess-SY, eess.SY  
Keywords: QA  
[Paper Link](http://arxiv.org/abs/2401.09800v1)  

---


**ABSTRACT**  
Power system fault diagnosis is crucial for identifying the location and causes of faults and providing decision-making support for power dispatchers. However, most classical methods suffer from significant time-consuming, memory overhead, and computational complexity issues as the scale of the power system concerned increases. With rapid development of quantum computing technology, the combinatorial optimization method based on quantum computing has shown certain advantages in computational time over existing methods. Given this background, this paper proposes a quantum computing based power system fault diagnosis method with the Quantum Approximate Optimization Algorithm (QAOA). The proposed method reformulates the fault diagnosis problem as a Hamiltonian by using Ising model, which completely preserves the coupling relationship between faulty components and various operations of protective relays and circuit breakers. Additionally, to enhance problem-solving efficiency under current equipment limitations, the symmetric equivalent decomposition method of multi-z-rotation gate is proposed. Furthermore, the small probability characteristics of power system events is utilized to reduce the number of qubits. Simulation results based on the test system show that the proposed methods can achieve the same optimal results with a faster speed compared with the classical higher-order solver provided by D-Wave.

{{</citation>}}


### (99/105) Traffic Smoothing Controllers for Autonomous Vehicles Using Deep Reinforcement Learning and Real-World Trajectory Data (Nathan Lichtlé et al., 2024)

{{<citation>}}

Nathan Lichtlé, Kathy Jang, Adit Shah, Eugene Vinitsky, Jonathan W. Lee, Alexandre M. Bayen. (2024)  
**Traffic Smoothing Controllers for Autonomous Vehicles Using Deep Reinforcement Learning and Real-World Trajectory Data**  

---
Primary Category: eess.SY  
Categories: cs-AI, cs-MA, cs-SY, eess-SY, eess.SY  
Keywords: Reinforcement Learning  
[Paper Link](http://arxiv.org/abs/2401.09666v1)  

---


**ABSTRACT**  
Designing traffic-smoothing cruise controllers that can be deployed onto autonomous vehicles is a key step towards improving traffic flow, reducing congestion, and enhancing fuel efficiency in mixed autonomy traffic. We bypass the common issue of having to carefully fine-tune a large traffic microsimulator by leveraging real-world trajectory data from the I-24 highway in Tennessee, replayed in a one-lane simulation. Using standard deep reinforcement learning methods, we train energy-reducing wave-smoothing policies. As an input to the agent, we observe the speed and distance of only the vehicle in front, which are local states readily available on most recent vehicles, as well as non-local observations about the downstream state of the traffic. We show that at a low 4% autonomous vehicle penetration rate, we achieve significant fuel savings of over 15% on trajectories exhibiting many stop-and-go waves. Finally, we analyze the smoothing effect of the controllers and demonstrate robustness to adding lane-changing into the simulation as well as the removal of downstream information.

{{</citation>}}


## cs.MM (1)



### (100/105) On the Audio Hallucinations in Large Audio-Video Language Models (Taichi Nishimura et al., 2024)

{{<citation>}}

Taichi Nishimura, Shota Nakada, Masayoshi Kondo. (2024)  
**On the Audio Hallucinations in Large Audio-Video Language Models**  

---
Primary Category: cs.MM  
Categories: cs-CL, cs-CV, cs-MM, cs-SD, cs.MM, eess-AS  
Keywords: Language Model  
[Paper Link](http://arxiv.org/abs/2401.09774v1)  

---


**ABSTRACT**  
Large audio-video language models can generate descriptions for both video and audio. However, they sometimes ignore audio content, producing audio descriptions solely reliant on visual information. This paper refers to this as audio hallucinations and analyzes them in large audio-video language models. We gather 1,000 sentences by inquiring about audio information and annotate them whether they contain hallucinations. If a sentence is hallucinated, we also categorize the type of hallucination. The results reveal that 332 sentences are hallucinated with distinct trends observed in nouns and verbs for each hallucination type. Based on this, we tackle a task of audio hallucination classification using pre-trained audio-text models in the zero-shot and fine-tuning settings. Our experimental results reveal that the zero-shot models achieve higher performance (52.2% in F1) than the random (40.3%) and the fine-tuning models achieve 87.9%, outperforming the zero-shot models.

{{</citation>}}


## cs.SI (1)



### (101/105) Towards Learning from Graphs with Heterophily: Progress and Future (Chenghua Gong et al., 2024)

{{<citation>}}

Chenghua Gong, Yao Cheng, Xiang Li, Caihua Shan, Siqiang Luo, Chuan Shi. (2024)  
**Towards Learning from Graphs with Heterophily: Progress and Future**  

---
Primary Category: cs.SI  
Categories: cs-AI, cs-LG, cs-SI, cs.SI  
Keywords: GNN  
[Paper Link](http://arxiv.org/abs/2401.09769v1)  

---


**ABSTRACT**  
Graphs are structured data that models complex relations between real-world entities. Heterophilous graphs, where linked nodes are prone to be with different labels or dissimilar features, have recently attracted significant attention and found many applications. Meanwhile, increasing efforts have been made to advance learning from heterophilous graphs. Although there exist surveys on the relevant topic, they focus on heterophilous GNNs, which are only sub-topics of heterophilous graph learning. In this survey, we comprehensively overview existing works on learning from graphs with heterophily.First, we collect over 180 publications and introduce the development of this field. Then, we systematically categorize existing methods based on a hierarchical taxonomy including learning strategies, model architectures and practical applications. Finally, we discuss the primary challenges of existing studies and highlight promising avenues for future research.More publication details and corresponding open-source codes can be accessed and will be continuously updated at our repositories:https://github.com/gongchenghua/Awesome-Survey-Graphs-with-Heterophily.

{{</citation>}}


## cs.SC (1)



### (102/105) Bootstrapping OTS-Funcimg Pre-training Model (Botfip) -- A Comprehensive Symbolic Regression Framework (Tianhao Chen et al., 2024)

{{<citation>}}

Tianhao Chen, Pengbo Xu, Haibiao Zheng. (2024)  
**Bootstrapping OTS-Funcimg Pre-training Model (Botfip) -- A Comprehensive Symbolic Regression Framework**  

---
Primary Category: cs.SC  
Categories: cs-AI, cs-LG, cs-SC, cs.SC  
Keywords: AI  
[Paper Link](http://arxiv.org/abs/2401.09748v1)  

---


**ABSTRACT**  
In the field of scientific computing, many problem-solving approaches tend to focus only on the process and final outcome, even in AI for science, there is a lack of deep multimodal information mining behind the data, missing a multimodal framework akin to that in the image-text domain. In this paper, we take Symbolic Regression(SR) as our focal point and, drawing inspiration from the BLIP model in the image-text domain, propose a scientific computing multimodal framework based on Function Images (Funcimg) and Operation Tree Sequence (OTS), named Bootstrapping OTS-Funcimg Pre-training Model (Botfip). In SR experiments, we validate the advantages of Botfip in low-complexity SR problems, showcasing its potential. As a MED framework, Botfip holds promise for future applications in a broader range of scientific computing problems.

{{</citation>}}


## math.NA (1)



### (103/105) Fast Updating Truncated SVD for Representation Learning with Sparse Matrices (Haoran Deng et al., 2024)

{{<citation>}}

Haoran Deng, Yang Yang, Jiahe Li, Cheng Chen, Weihao Jiang, Shiliang Pu. (2024)  
**Fast Updating Truncated SVD for Representation Learning with Sparse Matrices**  

---
Primary Category: math.NA  
Categories: cs-NA, math-NA, math.NA  
Keywords: Representation Learning  
[Paper Link](http://arxiv.org/abs/2401.09703v1)  

---


**ABSTRACT**  
Updating a truncated Singular Value Decomposition (SVD) is crucial in representation learning, especially when dealing with large-scale data matrices that continuously evolve in practical scenarios. Aligning SVD-based models with fast-paced updates becomes increasingly important. Existing methods for updating truncated SVDs employ Rayleigh-Ritz projection procedures, where projection matrices are augmented based on original singular vectors. However, these methods suffer from inefficiency due to the densification of the update matrix and the application of the projection to all singular vectors. To address these limitations, we introduce a novel method for dynamically approximating the truncated SVD of a sparse and temporally evolving matrix. Our approach leverages sparsity in the orthogonalization process of augmented matrices and utilizes an extended decomposition to independently store projections in the column space of singular vectors. Numerical experiments demonstrate a remarkable efficiency improvement of an order of magnitude compared to previous methods. Remarkably, this improvement is achieved while maintaining a comparable precision to existing approaches.

{{</citation>}}


## cs.HC (1)



### (104/105) Should ChatGPT Write Your Breakup Text? Exploring the Role of AI in Relationship Dissolution (Yue Fu et al., 2024)

{{<citation>}}

Yue Fu, Yixin Chen, Zelia Gomes Da Costa Lai, Alexis Hiniker. (2024)  
**Should ChatGPT Write Your Breakup Text? Exploring the Role of AI in Relationship Dissolution**  

---
Primary Category: cs.HC  
Categories: cs-AI, cs-HC, cs.HC  
Keywords: AI, ChatGPT, GPT  
[Paper Link](http://arxiv.org/abs/2401.09695v1)  

---


**ABSTRACT**  
Relationships are essential to our happiness and wellbeing. The dissolution of a relationship, the final stage of relationship's lifecycle and one of the most stressful events in an individual's life, can have profound and long-lasting impacts on people. With the breakup process increasingly facilitated by computer-mediated communication (CMC), and the likely future influence of AI-mediated communication (AIMC) tools, we conducted a semi-structured interview study with 21 participants. We aim to understand: 1) the current role of technology in the breakup process, 2) the needs and support individuals have during the process, and 3) how AI might address these needs. Our research shows that people have distinct needs at various stages of ending a relationship. Presently, technology is used for information gathering and community support, acting as a catalyst for breakups, enabling ghosting and blocking, and facilitating communication. Participants anticipate that AI could aid in sense-making of their relationship leading up to the breakup, act as a mediator, assist in crafting appropriate wording, tones, and language during breakup conversations, and support companionship, reflection, recovery, and growth after a breakup. Our findings also demonstrate an overlap between the breakup process and the Transtheoretical Model (TTM) of behavior change. Through the lens of TTM, we explore the potential support and affordances AI could offer in breakups, including its benefits and the necessary precautions regarding AI's role in this sensitive process.

{{</citation>}}


## astro-ph.IM (1)



### (105/105) Decades of Transformation: Evolution of the NASA Astrophysics Data System's Infrastructure (Alberto Accomazzi, 2024)

{{<citation>}}

Alberto Accomazzi. (2024)  
**Decades of Transformation: Evolution of the NASA Astrophysics Data System's Infrastructure**  

---
Primary Category: astro-ph.IM  
Categories: astro-ph-IM, astro-ph.IM, cs-DL  
Keywords: AI, NLP, Natural Language Processing  
[Paper Link](http://arxiv.org/abs/2401.09685v1)  

---


**ABSTRACT**  
The NASA Astrophysics Data System (ADS) is the primary Digital Library portal for researchers in astronomy and astrophysics. Over the past 30 years, the ADS has gone from being an astronomy-focused bibliographic database to an open digital library system supporting research in space and (soon) earth sciences. This paper describes the evolution of the ADS system, its capabilities, and the technological infrastructure underpinning it.   We give an overview of the ADS's original architecture, constructed primarily around simple database models. This bespoke system allowed for the efficient indexing of metadata and citations, the digitization and archival of full-text articles, and the rapid development of discipline-specific capabilities running on commodity hardware. The move towards a cloud-based microservices architecture and an open-source search engine in the late 2010s marked a significant shift, bringing full-text search capabilities, a modern API, higher uptime, more reliable data retrieval, and integration of advanced visualizations and analytics.   Another crucial evolution came with the gradual and ongoing incorporation of Machine Learning and Natural Language Processing algorithms in our data pipelines. Originally used for information extraction and classification tasks, NLP and ML techniques are now being developed to improve metadata enrichment, search, notifications, and recommendations. we describe how these computational techniques are being embedded into our software infrastructure, the challenges faced, and the benefits reaped.   Finally, we conclude by describing the future prospects of ADS and its ongoing expansion, discussing the challenges of managing an interdisciplinary information system in the era of AI and Open Science, where information is abundant, technology is transformative, but their trustworthiness can be elusive.

{{</citation>}}
